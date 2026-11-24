"""Run: python3 -m unittest test_twotime_postprocess -v (standard library only)."""
import base64
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import twotime_postprocess as pp

HERE = Path(__file__).resolve().parent


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.folder = self.base / 'run'
        self.times = [-1., 0., 1., 2.]

    def fixture(self, label="<P2|c'|P3>"):
        for step in range(2):
            work = self.folder / f'step_{step:04d}'
            work.mkdir(parents=True)
            tp = float(step)
            for direction, times in [('forward', self.times[step+1:-1]), ('backward', self.times[1:step+2])]:
                echo = base64.b64encode(b'TotalNumberOfSites=2;').decode()
                lines = ['PsiApp::echoBase64: Echo', echo, 'FiniteLoops printing ends']
                for t in [tp] + [t for t in times if t != tp]:
                    for site in range(2):
                        lines.append(f'{site} (1,0) {t} {label} (0,0)')
                        if t == tp:
                            # Maximum pairwise distance is 5 forward, 0 backward.
                            if direction == 'forward':
                                lines.extend([f'{site} (4,4) {t} {label}', f'{site} (2,0) {t} {label}'])
                lines.append('DmrgSolver: Turning off the engine.')
                (work / f'runForinput_evolve_{direction}.cout').write_text('\n'.join(lines))

    def process(self, workers=1, label="<P2|c'|P3>", token='P2_cdag_P3'):
        with redirect_stdout(io.StringIO()):
            return pp.process_twotime_results(self.base, 'run', 2, self.times, 0, workers, label, token)

    def test_report_extrema_and_all_values(self):
        self.fixture()
        path = self.process()
        report = path.with_name(path.stem+'_equal_time_diagnostics.txt').read_text()
        self.assertLess(report.index('SUMMARY'), report.index('ALL EQUAL-TIME'))
        self.assertIn('5.00000000e+00', report)
        self.assertIn('Min_ties Max_ties', report)
        rows = list(csv.DictReader(report.split('ALL EQUAL-TIME DIAGNOSTICS\n')[1].splitlines()))
        self.assertEqual(len(rows), 4)
        self.assertEqual({float(r['forward_spread']) for r in rows}, {5.})
        self.assertEqual({float(r['backward_spread']) for r in rows}, {0.})
        self.assertEqual({float(r['forward_backward_difference']) for r in rows}, {1.})

    def test_physical_csv_retains_last_forward_not_backward_or_mean(self):
        self.fixture()
        with self.process().open() as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 8)
        for r in rows:
            expected = 2. if r['t_prime'] == r['t'] else 1.
            self.assertEqual(float(r['real_part']), expected)

    def test_parallel_matches_serial(self):
        self.fixture()
        path = self.process()
        before = path.read_bytes()
        report = path.with_name(path.stem+'_equal_time_diagnostics.txt')
        before_report = report.read_bytes()
        self.process(workers=2)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(report.read_bytes(), before_report)

    def test_greater_label_and_filename(self):
        self.fixture('<P2|c|P3>')
        self.assertIn('P2_c_P3', self.process(label='<P2|c|P3>', token='P2_c_P3').name)

    def test_non_equal_time_conflict_still_fails(self):
        self.fixture()
        path = self.folder/'step_0000/runForinput_evolve_forward.cout'
        with path.open('a') as file:
            file.write("\n0 (9,0) 1 <P2|c'|P3>\n")
        with self.assertRaisesRegex(ValueError, 'non-equal-time'):
            self.process()

    def test_incomplete_grid_fails(self):
        self.fixture()
        path = self.folder/'step_0000/runForinput_evolve_forward.cout'
        path.write_text(path.read_text().replace("1 (1,0) 1.0 <P2|c'|P3> (0,0)", ''))
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            self.process()

    def test_unfinished_log_fails(self):
        self.fixture()
        path = self.folder/'step_0000/runForinput_evolve_forward.cout'
        path.write_text(path.read_text().replace('Turning off the engine.', 'not finished'))
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            self.process()

    def test_malformed_measurement_fails(self):
        self.fixture()
        path = self.folder/'step_0000/runForinput_evolve_forward.cout'
        path.write_text(path.read_text().replace('(1,0)', '(nan,0)', 1))
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            self.process()

    def test_summary_extrema_locate_different_tprime_sites(self):
        path = self.base/'report.txt'
        rows = [(0.,0,3.,4.,5.), (1.,1,1.,8.,2.)]
        with redirect_stdout(io.StringIO()):
            pp.save_diagnostics(rows,path)
        text = path.read_text()
        line = next(l for l in text.splitlines() if l.startswith('forward_spread '))
        fields = line.split()
        self.assertEqual(fields[1:7], ['1.00000000e+00','1','1','3.00000000e+00','0','0'])

    def test_cli_both_branches(self):
        for branch, label in [('G<', "<P2|c'|P3>"), ('G>', '<P2|c|P3>')]:
            if self.folder.exists():
                import shutil
                shutil.rmtree(self.folder)
            self.fixture(label)
            pump = self.base/'pump.csv'
            pump.write_text('time,pump\n-1,0\n0,0\n1,0\n2,0\n')
            result = subprocess.run([sys.executable, str(HERE/f'my_hubbard_chain_twotime_{branch}_parallel_process.py'),
                '0', str(self.base/'gs.hd5'), str(pump), str(self.folder), '--workers', '2'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('WARNING', result.stdout)
            self.assertIn('Wrote equal-time diagnostics', result.stdout)


if __name__ == '__main__':
    unittest.main()
