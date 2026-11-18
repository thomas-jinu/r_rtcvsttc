#!/usr/bin/env python3
"""Generate and optionally run or submit independent TD-only pump-frequency jobs.

Uses the TD input, pump and batch helpers from the combined script in the same
folder. No two-time G inputs or jobs are generated. --run runs frequencies
concurrently within one allocation; --submit requests one node per frequency.
"""
import argparse
from datetime import UTC, datetime
import json
import math
import os
from pathlib import Path
import shlex
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    p.add_argument('--gpus-per-task', type=int, default=1, help='GPUs assigned to each frequency run (default: 1)')
    execution = p.add_mutually_exclusive_group()
    execution.add_argument('--run', action='store_true', help='Run TD jobs concurrently on the current node')
    execution.add_argument('--submit', action='store_true', help='Submit one NERSC job with one node per frequency')
    a = p.parse_args()
    if a.sites <= 2 or a.TSPAdvanceEach != a.sites - 2:
        p.error('TSPAdvanceEach must equal sites - 2, with sites > 2')
    if a.finite_kept < 1 or a.Pump_time_steps < 1 or a.TSPTau <= 0 or a.Pump_pulse_width <= 0 or a.cpus_per_task < 1:
        p.error('finite_kept, Pump_time_steps, TSPTau, Pump_pulse_width and cpus-per-task must be positive')
    if not 1 <= a.gpus_per_task <= 4:
        p.error('gpus-per-task must be between 1 and 4 for a Perlmutter GPU node')
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
    jobs = []
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
        if batch is not None:
            batch.write_text(batch.read_text(encoding='utf-8').replace('#SBATCH --gpus-per-task=4\n', f'#SBATCH --gpus-per-task={a.gpus_per_task}\n'), encoding='utf-8')
        print(f'Frequency {frequency:.15g}: {folder}', flush=True)
        jobs.append((folder, input_path, batch))

    if a.submit:
        # Reuse the TD batch module environment, but launch one exclusive step per node.
        template = jobs[0][2].read_text(encoding='utf-8')
        header = template.split('\ncd ', 1)[0]
        header = header.replace('#SBATCH --nodes=1\n', f'#SBATCH --nodes={len(jobs)}\n')
        header = header.replace('#SBATCH --ntasks=1\n', f'#SBATCH --ntasks={len(jobs)}\n')
        lines = [header, 'pids=()']
        for folder, input_path, _ in jobs:
            command = shlex.join(['./dmrg', '-f', input_path.name, '-p', str(combined.DMRG_PRECISION), combined.TD_OPERATORS])
            lines.append(f'(cd {shlex.quote(str(folder))} && srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task={a.cpus_per_task} --gpus-per-task={a.gpus_per_task} {command} > td_streamed.stdout 2> td_streamed.stderr) &')
            lines.append('pids+=("$!")')
        lines += ['failed=0', 'for pid in "${pids[@]}"; do wait "$pid" || failed=1; done', 'exit "$failed"']
        sweep_batch = gs.parent / f'batch_td_sweep_{run_base}.slurm'
        sweep_batch.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(f'Submitting {sweep_batch}', flush=True)
        subprocess.run(['sbatch', str(sweep_batch)], cwd=gs.parent, check=True)
    elif a.run:
        # SLURM_CPUS_ON_NODE is the CPU count available to this allocation.
        total = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 1))
        if a.cpus_per_task > total:
            raise ValueError(f'cpus-per-task ({a.cpus_per_task}) exceeds allocated CPUs ({total})')
        if a.cluster == 'nersc':
            visible = os.environ.get('CUDA_VISIBLE_DEVICES')
            if visible is not None:
                gpu_count = len([x for x in visible.split(',') if x.strip() and x.strip() != '-1'])
            else:
                result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
                gpu_count = len([line for line in result.stdout.splitlines() if line.strip()])
            if gpu_count < a.gpus_per_task:
                raise RuntimeError(f'Requested {a.gpus_per_task} GPUs per task but only {gpu_count} GPUs are visible')
            parallel = min(len(jobs), total // a.cpus_per_task, gpu_count // a.gpus_per_task)
        else:
            # Without Slurm GPU isolation, do not share an unspecified local GPU concurrently.
            parallel = 1
        print(f'Running up to {parallel} frequencies concurrently, {a.cpus_per_task} CPUs and {a.gpus_per_task} GPUs per task.', flush=True)
        def execute(job):
            folder, input_path, _ = job
            env = os.environ.copy()
            env['OMP_NUM_THREADS'] = str(max(1, a.cpus_per_task // 2) if a.cluster == 'nersc' else a.cpus_per_task)
            env.setdefault('OMP_PLACES', 'cores')
            env.setdefault('OMP_PROC_BIND', 'spread')
            command = ['./dmrg', '-f', input_path.name, '-p', str(combined.DMRG_PRECISION), combined.TD_OPERATORS]
            if a.cluster == 'nersc':
                command = ['srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1', f'--cpus-per-task={a.cpus_per_task}', '--cpu-bind=cores', f'--gpus-per-task={a.gpus_per_task}', '--gpu-bind=closest'] + command
            with (folder / 'td_streamed.stdout').open('w') as out, (folder / 'td_streamed.stderr').open('w') as err:
                subprocess.run(command, cwd=folder, env=env, stdout=out, stderr=err, check=True)
        failed = []
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(execute, job): job[0] for job in jobs}
            for future in as_completed(futures):
                try:
                    future.result()
                except (subprocess.CalledProcessError, OSError) as exc:
                    print(f'TD failed in {futures[future]}: {exc}', flush=True)
                    failed.append(futures[future])
        if failed:
            raise RuntimeError(f'{len(failed)} TD frequency runs failed; see their td_streamed.stderr files')


if __name__ == '__main__':
    main()
