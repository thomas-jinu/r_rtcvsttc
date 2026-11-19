#!/usr/bin/env python3
"""Helpers for forward-only relative-time G< and G> calculations."""
import csv
import math
import re
from pathlib import Path

from dmrg_config import link_binary


def generate(a, parent, run_name, executable, evolve_input, pump_axis):
    """Create singletime inputs alongside existing TD and two-time folders."""
    if not math.isfinite(a.TSPTau) or a.TSPTau <= 0:
        raise ValueError('TSPTau must be finite and positive for singletime calculations')
    if a.Pump_time_steps < 1:
        raise ValueError('Pump_time_steps must be positive')
    folders = []
    for prefix, creator in (('G<', False), ('G>', True)):
        folder = parent / f'{prefix}_singletime_{run_name}'
        folder.mkdir(parents=True, exist_ok=False)
        for step in range(a.Pump_time_steps):
            work = folder / f'step_{step:04d}'
            work.mkdir()
            link_binary(parent / run_name / 'dmrg', work)
            # The table starts at the restart time step*TSPTau and ends one step past the
            # last needed sample ((step+N-1)*TSPTau): one trailing padding row, as in the
            # G< / G> tables.
            times = [round((step + i) * a.TSPTau, 9) for i in range(0, a.Pump_time_steps + 1)]
            pump = pump_axis(a, times)
            restart = f'../../{a.gs_filename.resolve().name}' if step == 0 else f'../../{run_name}/Recovery{step - 1}{run_name}'
            name = f'input_singletime_{run_name}_evolve_forward.ain'
            text = evolve_input(a, f'singletime_{run_name}', times, pump, 0, True, restart, creator)
            # Same rule as the G< / G> inputs: one loop per table entry from the restart
            # time (padding included), plus 8 warm-up loops. startLoop=9 follows from this.
            loops = len(times) * (a.TSPAdvanceEach // (a.sites - 2)) + 8
            rows = ',\n'.join(f'    [@auto, {a.finite_kept}, 2]' for _ in range(loops))
            text, count = re.subn(r'FiniteLoops = \[.*?\];', lambda _: 'FiniteLoops = [\n' + rows + '\n];', text, count=1, flags=re.S)
            if count != 1:
                raise ValueError('Could not set singletime finite loops')
            (work / name).write_text(text, encoding='utf-8')
            with (work / 'pump_table.csv').open('w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(('time', 'pump'))
                w.writerows(zip(times, pump))
        folders.append(folder)
    return tuple(folders)


def process(folder: Path, run_name: str, n: int, dt: float, center: int, label: str):
    """Extract forward correlators, retaining only the N requested time samples."""
    pattern = re.compile(r'^\s*(\d+)\s+\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)\s+([-+\d.eE]+)\s+' + re.escape(label))
    rows = {}
    tolerance = max(1e-8, abs(dt) * 1e-6)
    for step in range(n):
        step_dir = folder / f'step_{step:04d}'
        matches = sorted(step_dir.glob('runFor*_evolve_forward.cout'))
        if len(matches) != 1:
            raise FileNotFoundError(f'Expected one forward singletime output in {step_dir}, found {len(matches)}')
        path = matches[0]
        collecting = False
        with path.open(encoding='utf-8') as f:
            for line in f:
                if 'FiniteLoops printing ends' in line:
                    collecting = True
                    continue
                if not collecting:
                    continue
                match = pattern.search(line)
                if not match:
                    continue
                site, real, imag, absolute = int(match[1]), float(match[2]), float(match[3]), float(match[4])
                relative_index = round((absolute - step * dt) / dt)
                expected = (step + relative_index) * dt
                if 0 <= relative_index < n and abs(absolute - expected) <= tolerance:
                    rows[(step, relative_index, site)] = (real, imag)
        if not any(key[0] == step for key in rows):
            raise ValueError(f'No requested correlator measurements found in {path}')
    # Fail instead of silently producing an incomplete relative-time grid.
    sites = {site for _, _, site in rows}
    missing = [(step, k, site) for step in range(n) for k in range(n) for site in sites if (step, k, site) not in rows]
    if missing:
        raise ValueError(f'Incomplete singletime grid in {folder}: {len(missing)} missing values; first {missing[0]}')
    def write_csv(path, entries):
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(('t_prime', 'relative_time', 't', 'center_site', 'site_index', 'real_part', 'imaginary_part'))
            for (step, k, site), (real, imag) in entries:
                writer.writerow((f'{step * dt:.15g}', f'{k * dt:.15g}', f'{(step + k) * dt:.15g}', center, site, f'{real:.15g}', f'{imag:.15g}'))

    entries = sorted(rows.items())
    processed = folder.parent / 'processed_data'
    for step in range(n):
        step_dir = processed / folder.name / f'step_{step:04d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        step_csv = step_dir / f'singletime_centersite={center}.csv'
        write_csv(step_csv, (entry for entry in entries if entry[0][0] == step))
    output = processed / f'{folder.name}_centersite={center}.csv'
    write_csv(output, entries)
    print(f'Wrote {n} per-step singletime CSV files in {processed / folder.name}', flush=True)
    print(f'Wrote combined singletime data: {output}', flush=True)
    return output


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Process one singletime correlator folder')
    p.add_argument('folder', type=Path)
    p.add_argument('run_name')
    p.add_argument('count', type=int)
    p.add_argument('dt', type=float)
    p.add_argument('center', type=int)
    p.add_argument('label')
    a = p.parse_args()
    process(a.folder, a.run_name, a.count, a.dt, a.center, a.label)
