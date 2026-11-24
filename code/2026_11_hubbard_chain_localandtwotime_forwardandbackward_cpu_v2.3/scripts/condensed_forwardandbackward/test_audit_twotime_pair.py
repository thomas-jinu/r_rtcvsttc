"""Run with python3 -m unittest test_audit_twotime_pair -v."""
import base64
import tempfile
import unittest
from pathlib import Path

from audit_twotime_pair import audit


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def output(self, name, times, extra="", finish=True, missing=False):
        tau = 0.1 if times[1] > times[0] else -0.1
        text = 'TotalNumberOfSites=2;\nRestartFilename="checkpoint";\n'
        text += 'matrix AversusTime = [\n' + '\n'.join(f'[{t}, 0],' for t in times) + '\n];\n'
        for v in ('P2', 'P3'):
            text += f'string {v}="TimeEvolve{{tau={tau},startLoop=9}}*|P0>";\n'
        lines = ['PsiApp::echoBase64: Echo', base64.b64encode(text.encode()).decode(),
                 'FiniteLoops printing ends', 'Finite loop number 0']
        for t in times[:-1]:
            for site in range(2):
                if missing and site == 1 and t == times[0]:
                    continue
                lines.append(f"{site} (0.5,0) {t} <P2|c'|P3> (0,0)")
        lines.extend(['Finite loop number 1', extra])
        if finish:
            lines.append('DmrgSolver: Turning off the engine.')
        path = self.root / name
        path.write_text('\n'.join(lines))
        return path

    def pair(self, **kwargs):
        return self.output('f', [1., 1.1, 1.2], **kwargs), self.output('b', [1., .9, .8])

    def test_physical_grid_excludes_padding(self):
        report = audit(*self.pair())
        self.assertEqual(report['structural_errors'], [])
        self.assertEqual(report['directions']['forward']['physical_times'], [1., 1.1])
        self.assertEqual(report['directions']['forward']['padding_time'], 1.2)
        self.assertFalse(report['accuracy_validated'])

    def test_restart_drift_is_diagnostic_not_structural_failure(self):
        report = audit(*self.pair(extra="0 (0.50001,0) 1 <P2|c'|P3> (0,0)"))
        self.assertEqual(report['structural_errors'], [])
        repeated = report['directions']['forward']['repeated_measurements'][0]
        self.assertTrue(repeated['exceeds_diagnostic_tolerance'])
        self.assertEqual([s['loop'] for s in repeated['samples']], [0, 1])
        self.assertGreater(report['equal_time_comparisons'][0]['last_difference'], 1e-7)

    def test_nonrestart_duplicate_fails(self):
        report = audit(*self.pair(extra="0 (0.5,0) 1.1 <P2|c'|P3> (0,0)"))
        self.assertTrue(report['structural_errors'])

    def test_missing_measurement_fails(self):
        self.assertTrue(audit(*self.pair(missing=True))['structural_errors'])

    def test_incomplete_engine_fails(self):
        self.assertTrue(audit(*self.pair(finish=False))['structural_errors'])

    def test_padding_measurement_fails(self):
        report = audit(*self.pair(extra="0 (0.5,0) 1.2 <P2|c'|P3> (0,0)"))
        self.assertTrue(report['structural_errors'])

    def test_invalid_tolerance_rejected(self):
        with self.assertRaises(ValueError):
            audit(*self.pair(), tolerance=float('nan'))

    def test_concatenated_logs_rejected(self):
        f, b = self.pair()
        f.write_text(f.read_text() * 2)
        with self.assertRaises(ValueError):
            audit(f, b)


if __name__ == '__main__':
    unittest.main()
