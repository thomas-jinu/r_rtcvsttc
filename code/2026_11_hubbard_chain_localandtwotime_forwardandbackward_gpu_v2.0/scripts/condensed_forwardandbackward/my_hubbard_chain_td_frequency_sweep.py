#!/usr/bin/env python3
"""Generate and optionally run or submit independent TD-only pump-frequency jobs.

Uses the TD input, pump and batch helpers from the combined script in the same
folder. No two-time G inputs or jobs are generated. --run executes frequencies
sequentially; --submit submits one independent Slurm job per frequency.
"""
import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import my_hubbard_chain_td_Gless_Ggreater_singlejob_streaming as combined


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('sites', type=int)
    p.add_argument('up', type=int)
    p.add_argument('down', type=int)
    p.add_argument('t', type=float)
    p.add_argument('U', type=float)
    p.add_argument('potentialV', type=float)
    p.add_argument('gs_filename', type=Path)
    p.add_argument('finite_kept', type=int)
    p.add_argument('TSPAdvanceEach', type=int)
    p.add_argument('TSPTau', type=float)
    p.add_argument('Pump_Amplitude', type=float)
    p.add_argument('Pump_time_delay', type=float)
    p.add_argument('Pump_pulse_width', type=float)
    p.add_argument('Pump_time_steps', type=int)
    p.add_argument('cluster', choices=combined.DMRG_EXECUTABLES)
    p.add_argument('--frequency-min', type=float, required=True)
    p.add_argument('--frequency-max', type=float, required=True)
    p.add_argument('--frequency-count', type=int, required=True, help='Equally spaced frequencies including both endpoints')
    p.add_argument('--cpus-per-task', type=int, default=128)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument('--run', action='store_true', help='Run TD jobs sequentially on the current node')
    execution.add_argument('--submit', action='store_true', help='Submit independent NERSC TD Slurm jobs')
    a = p.parse_args()
    if a.sites <= 2 or a.TSPAdvanceEach != a.sites - 2:
        p.error('TSPAdvanceEach must equal sites - 2, with sites > 2')
    if a.finite_kept < 1 or a.Pump_time_steps < 1 or a.TSPTau <= 0 or a.Pump_pulse_width <= 0 or a.cpus_per_task < 1:
        p.error('finite_kept, Pump_time_steps, TSPTau, Pump_pulse_width and cpus-per-task must be positive')
    if a.frequency_count < 1 or not math.isfinite(a.frequency_min) or not math.isfinite(a.frequency_max) or a.frequency_min > a.frequency_max:
        p.error('frequency count must be positive and finite frequency bounds must be ordered')
    if a.frequency_count == 1 and a.frequency_min != a.frequency_max:
        p.error('for one frequency, frequency-min and frequency-max must match')
    if a.frequency_count > 1 and a.frequency_min == a.frequency_max:
        p.error('frequency bounds must differ for multiple frequencies')
    if a.submit and a.cluster != 'nersc':
        p.error('--submit requires cluster=nersc')
    if a.run and a.cluster == 'nersc' and not os.environ.get('SLURM_JOB_ID'):
        p.error('--run on NERSC requires an interactive Slurm compute-node allocation')
    gs = a.gs_filename.resolve()
    executable = combined.DMRG_EXECUTABLES[a.cluster]
    if not gs.is_file():
        p.error(f'Ground-state file not found: {gs}')
    if not executable.is_file():
        p.error(f'DMRG++ executable not found: {executable}')
    frequencies = [a.frequency_min] if a.frequency_count == 1 else [
        a.frequency_min + (a.frequency_max - a.frequency_min) * i / (a.frequency_count - 1)
        for i in range(a.frequency_count)
    ]
    tags = [combined.frequency_tag(frequency) for frequency in frequencies]
    if len(set(tags)) != len(tags):
        p.error('Frequency values collide when formatted as folder names; reduce frequency count or widen the range')
    # Validate every pump before creating or submitting any jobs.
    times = combined.time_axis(a)
    pumps = []
    for frequency in frequencies:
        a.pump_frequency = frequency
        pumps.append(combined.pump_axis(a, times))
    stamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
    run_base = f'N={a.sites}_Nup={a.up}_Ndown={a.down}_{stamp}'
    for frequency, tag, pump in zip(frequencies, tags, pumps):
        a.pump_frequency = frequency
        run_name = f'td_{run_base}_{tag}'
        folder = gs.parent / run_name
        folder.mkdir(exist_ok=False)
        values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
        (folder / f'input_args_td_{run_base}.json').write_text(json.dumps(values, indent=2), encoding='utf-8')
        shutil.copy2(executable, folder / 'dmrg')
        combined.write_pump(times, pump, folder / 'pump_table.csv')
        input_path = folder / f'input_td_{run_base}.ain'
        input_path.write_text(combined.td_input(a, run_name, times, pump, f'../{gs.name}'), encoding='utf-8')
        batch = combined.write_td_batch(a, folder, input_path) if a.cluster == 'nersc' else None
        print(f'Frequency {frequency:.15g}: {folder}', flush=True)
        if a.submit:
            subprocess.run(['sbatch', str(batch)], cwd=folder, check=True)
        elif a.run:
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(max(1, a.cpus_per_task // 2) if a.cluster == 'nersc' else a.cpus_per_task)
            env.setdefault('OMP_PLACES', 'cores')
            env.setdefault('OMP_PROC_BIND', 'spread')
            with (folder / 'td_streamed.stdout').open('w') as out, (folder / 'td_streamed.stderr').open('w') as err:
                subprocess.run(['./dmrg', '-f', input_path.name, '-p', str(combined.DMRG_PRECISION), combined.TD_OPERATORS], cwd=folder, env=env, stdout=out, stderr=err, check=True)


if __name__ == '__main__':
    main()
