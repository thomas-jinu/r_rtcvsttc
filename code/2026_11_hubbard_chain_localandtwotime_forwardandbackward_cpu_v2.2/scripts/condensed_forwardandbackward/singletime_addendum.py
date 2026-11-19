#!/usr/bin/env python3
"""Singletime (forward-only, relative-time) G< and G> helpers.

generate()  creates the singletime folders and inputs next to the TD and two-time folders.
process()   collects the forward outputs into relative-time CSV files.
"""
import csv
import math
import re
from pathlib import Path

from dmrg_config import link_binary

# A sample printed more than once (for example in a resumed run) must agree within this.
OVERLAP_TOLERANCE = 1e-7

def _singletime_times(step, a):
    """Pump times for one singletime step: from the restart time to one step past the last sample."""
    # One trailing padding row, as in the G< / G> tables.
    return [round((step + i) * a.TSPTau, 9) for i in range(0, a.Pump_time_steps + 1)]


def _set_finite_loops(text, a, count_entries):
    """Replace the FiniteLoops block with the singletime loop count (entries + 8 warm-up loops)."""
    loops = count_entries * (a.TSPAdvanceEach // (a.sites - 2)) + 8
    rows = ',\n'.join(f'    [@auto, {a.finite_kept}, 2]' for _ in range(loops))
    new_block = 'FiniteLoops = [\n' + rows + '\n];'
    text, count = re.subn(r'FiniteLoops = \[.*?\];', lambda _: new_block, text, count=1, flags=re.S)
    if count != 1:
        raise ValueError('Could not set singletime finite loops')
    return text


def _write_pump_table(path, times, pump):
    """Save the singletime time/pump table next to its input."""
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('time', 'pump'))
        writer.writerows(zip(times, pump))


def generate(a, parent, run_name, evolve_input, pump_axis):
    """Create the G< and G> singletime folders, one step folder per pump time."""
    if not math.isfinite(a.TSPTau) or a.TSPTau <= 0:
        raise ValueError('TSPTau must be finite and positive for singletime calculations')
    if a.Pump_time_steps < 1:
        raise ValueError('Pump_time_steps must be positive')

    folders = []
    # G< uses creator=False and G> uses creator=True, matching the two-time runs.
    for prefix, creator in (('G<', False), ('G>', True)):
        folder = parent / f'{prefix}_singletime_{run_name}'
        folder.mkdir(parents=True, exist_ok=False)
        for step in range(a.Pump_time_steps):
            work = folder / f'step_{step:04d}'
            work.mkdir()
            link_binary(parent / run_name / 'dmrg', work)

            times = _singletime_times(step, a)
            pump = pump_axis(a, times)
            # Step 0 restarts from the ground state; later steps from the TD checkpoint.
            if step == 0:
                restart = f'../../{a.gs_filename.resolve().name}'
            else:
                restart = f'../../{run_name}/Recovery{step - 1}{run_name}'
            name = f'input_singletime_{run_name}_evolve_forward.ain'

            text = evolve_input(a, f'singletime_{run_name}', times, pump, 0, True, restart, creator)
            text = _set_finite_loops(text, a, len(times))

            (work / name).write_text(text, encoding='utf-8')
            _write_pump_table(work / 'pump_table.csv', times, pump)
        folders.append(folder)
    return tuple(folders)


def _find_forward_output(step_dir):
    """The one forward DMRG++ output in a singletime step folder."""
    matches = sorted(step_dir.glob('runFor*_evolve_forward.cout'))
    if len(matches) != 1:
        raise FileNotFoundError(f'Expected one forward singletime output in {step_dir}, found {len(matches)}')
    return matches[0]


def _write_csv(path, entries, dt, center):
    """Write relative-time correlator rows: entries are ((step, k, site), (real, imag))."""
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(('t_prime', 'relative_time', 't', 'center_site', 'site_index', 'real_part', 'imaginary_part'))
        for (step, k, site), (real, imag) in entries:
            writer.writerow((
                f'{step * dt:.15g}', f'{k * dt:.15g}', f'{(step + k) * dt:.15g}',
                center, site, f'{real:.15g}', f'{imag:.15g}',
            ))


def process(folder: Path, run_name: str, n: int, dt: float, center: int, label: str):
    """Collect the forward correlators into relative-time CSV files.

    Keeps only the N requested samples per step. Fails if any value of the N x N grid is missing.
    """
    pattern = re.compile(
        r'^\s*(\d+)\s+\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)\s+([-+\d.eE]+)\s+' + re.escape(label)
    )
    rows = {}
    tolerance = max(1e-8, abs(dt) * 1e-6)

    for step in range(n):
        step_dir = folder / f'step_{step:04d}'
        path = _find_forward_output(step_dir)
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
                site = int(match[1])
                real, imag, absolute = float(match[2]), float(match[3]), float(match[4])
                # Keep a sample only if its time is on the grid: step*dt plus k*dt, k in 0..n-1.
                relative_index = round((absolute - step * dt) / dt)
                expected = (step + relative_index) * dt
                on_grid = 0 <= relative_index < n and abs(absolute - expected) <= tolerance
                if on_grid:
                    key = (step, relative_index, site)
                    if key in rows and abs(complex(*rows[key]) - complex(real, imag)) > OVERLAP_TOLERANCE:
                        raise ValueError(
                            f'Repeated value differs in {path}: step {step}, time {absolute}, '
                            f'site {site}: {rows[key]} vs {(real, imag)}'
                        )
                    rows[key] = (real, imag)
        if not any(key[0] == step for key in rows):
            raise ValueError(f'No requested correlator measurements found in {path}')

    # Fail instead of writing an incomplete relative-time grid.
    sites = {site for _, _, site in rows}
    missing = [(step, k, site) for step in range(n) for k in range(n) for site in sites
               if (step, k, site) not in rows]
    if missing:
        raise ValueError(f'Incomplete singletime grid in {folder}: {len(missing)} missing values; first {missing[0]}')

    entries = sorted(rows.items())
    processed = folder.parent / 'processed_data'
    for step in range(n):
        step_dir = processed / folder.name / f'step_{step:04d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        step_csv = step_dir / f'singletime_centersite={center}.csv'
        _write_csv(step_csv, (entry for entry in entries if entry[0][0] == step), dt, center)

    output = processed / f'{folder.name}_centersite={center}.csv'
    _write_csv(output, entries, dt, center)
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
    args = p.parse_args()
    process(args.folder, args.run_name, args.count, args.dt, args.center, args.label)
