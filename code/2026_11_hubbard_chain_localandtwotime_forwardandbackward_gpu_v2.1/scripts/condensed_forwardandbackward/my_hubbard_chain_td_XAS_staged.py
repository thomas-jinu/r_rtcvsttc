#!/usr/bin/env python3
"""Experimental TD + staged greater XAS two-time and singletime workflow.

Each time interval advances the reference and excited branches in separate
DMRG++ restarts, retaining both in a shared basis. Requires h5dump to validate
saved P-vector times. The backward branch uses H-Vch*n in reverse time; its
physical interpretation must be chosen deliberately. See --help.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
import json
import math
import os
import queue
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import my_hubbard_chain_td_Gless_Ggreater_singlejob_streaming as common

LABEL = '<P0|c|P1>'
ROW = re.compile(r'^\s*(\d+)\s+\(\s*([-+\d.eE]+)\s*,\s*([-+\d.eE]+)\s*\)\s+([-+\d.eE]+)\s+' + re.escape(LABEL))


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name, typ in [('sites', int), ('up', int), ('down', int), ('t', float), ('U', float),
                      ('potentialV', float), ('gs_filename', Path), ('finite_kept', int),
                      ('TSPAdvanceEach', int), ('TSPTau', float), ('Pump_Amplitude', float),
                      ('Pump_time_delay', float), ('Pump_pulse_width', float), ('Pump_time_steps', int)]:
        p.add_argument(name, type=typ)
    p.add_argument('cluster', choices=('local', 'nersc'))
    p.add_argument('center_site', type=int)
    p.add_argument('--pump-frequency', type=float, required=True)
    p.add_argument('--core-hole-potential', type=float, required=True, help='Vch >= 0; excited Hamiltonian H - Vch*n at center_site')
    p.add_argument('--cpus-per-task', type=int, default=128, help='Total CPU budget for --run; CPUs per node for --submit')
    p.add_argument('--run-gpus', type=int, default=4, help='Maximum GPUs for --run; one GPU per probe pipeline')
    p.add_argument('--xas-worker-nodes', type=int, default=None, help='Nodes in dependent XAS batch (default: one per reference time)')
    action = p.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true', help='Run TD then XAS pipelines on the current node')
    action.add_argument('--submit', action='store_true', help='Submit TD and dependent XAS Slurm jobs on NERSC')
    return p


def model(a, excited):
    s = common.model_block(a)
    v = [a.potentialV] * (2 * a.sites)
    if excited:
        v[a.center_site] -= a.core_hole_potential
        v[a.center_site + a.sites] -= a.core_hole_potential
    replacement = 'potentialV = [' + ', '.join(f'{x:.17g}' for x in v) + '];'
    return s.replace(f'potentialV = [{a.potentialV}, ...];', replacement)


def stage_input(a, times, pump, restart, output, mapping, stage, direction=1):
    # 10 finite loops: one primary time advance in the validated 4-site test.
    # The checkpoint is checked on every stage; never assume it advanced once.
    loops = 10
    expressions = ['string P0="|P0>";']
    if stage == 'initialize':
        expressions.append(f'string P1="c[{a.center_site}]\'*|P0>";')
    else:
        expressions.append('string P1="|P1>";')
    if stage in ('reference', 'excited'):
        source = 'P0' if stage == 'reference' else 'P1'
        expressions.append(f'string P2="TimeEvolve{{tau={direction * a.TSPTau},steps=5,startLoop=9,advanceEach={a.TSPAdvanceEach},depends={source},EnergyForExp=0}}*|{source}>";')
    if direction < 0:
        times, pump = list(reversed(times)), list(reversed(pump))
    return '\n\n'.join((
        '##Ainur1.0', model(a, stage in ('excited', 'measure_excited')),
        '\n'.join(('matrix AversusTime = [', common.table(times, pump), '];', 'GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;')),
        f'TargetElectronsUp = {a.up};\nTargetElectronsDown = {a.down};',
        '\n'.join(('TruncationTolerance = 1e-10;', 'FiniteLoops = [', common.rows(loops, a), '];', 'RestartMapStages=0;',
                   'RestartMappingTvs=[' + ', '.join(str(x) for x in mapping) + '];', *expressions)),
        '\n'.join(('SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead,BatchedGemm";',
                   'Version = "staged_xas_experimental";', 'string RecoverySave = "%l%%1,@keep,@M=1";',
                   f'OutputFile = "{output}";', f'RestartFilename = "{restart}";', 'GsWeight = 0.0;')),
    )) + '\n'


def saved_times(checkpoint):
    result = subprocess.run(['h5dump', '-d', '/Def/FinalPsi/TimeSerializer/PvectorTimes/Values', str(checkpoint)],
                            capture_output=True, text=True, check=True)
    data = re.search(r'DATA\s*\{(.*?)\}', result.stdout, re.S)
    if not data:
        raise ValueError(f'Cannot read P-vector times from {checkpoint}')
    values = re.sub(r'\(\d+\):', '', data.group(1))
    return [float(x) for x in re.findall(r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?', values)]


def check_times(checkpoint, expected):
    values = saved_times(checkpoint)
    for index, time in expected.items():
        if index >= len(values) or not math.isclose(values[index], time, rel_tol=1e-8, abs_tol=1e-8):
            raise RuntimeError(f'{checkpoint}: P{index} has time {values[index] if index < len(values) else "missing"}; expected {time}')


def extract(path, requested_time):
    collecting = False
    points = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if 'FiniteLoops printing ends' in line:
                collecting = True
                continue
            match = ROW.search(line) if collecting else None
            if match and math.isclose(float(match[4]), requested_time, abs_tol=1e-8, rel_tol=1e-8):
                points[int(match[1])] = complex(float(match[2]), float(match[3]))
    if not points:
        raise ValueError(f'No {LABEL} output at time {requested_time:g} in {path}')
    return points


def run_stage(a, work, times, pump, restart, name, mapping, stage, direction, expected):
    path = work / f'{name}.ain'
    path.write_text(stage_input(a, times, pump, str(restart), name, mapping, stage, direction), encoding='utf-8')
    command = ['./dmrg', '-f', path.name, '-p', str(common.DMRG_PRECISION), LABEL] if stage in ('initialize', 'measure_excited') else ['./dmrg', '-f', path.name, '-p', str(common.DMRG_PRECISION)]
    with (work / f'{name}.stdout').open('w') as out, (work / f'{name}.stderr').open('w') as err:
        subprocess.run(command, cwd=work, stdout=out, stderr=err, check=True)
    checkpoint = work / f'Recovery0{name}.hd5'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'Expected checkpoint missing: {checkpoint}')
    check_times(checkpoint, expected)
    points = extract(work / f'runFor{name}.cout', list(expected.values())[0]) if stage in ('initialize', 'measure_excited') else None
    return checkpoint, points


def save_step(path, data, step, dt, center, singletime):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        header = ['t_prime', 't']
        if singletime:
            header.append('relative_time')
        w.writerow(header + ['center_site', 'site_index', 'real_part', 'imaginary_part'])
        for (t, site), value in sorted(data.items()):
            row = [f'{step * dt:.15g}', f'{t:.15g}']
            if singletime:
                row.append(f'{t - step * dt:.15g}')
            w.writerow(row + [center, site, f'{value.real:.15g}', f'{value.imag:.15g}'])


def worker(manifest, probe):
    if not shutil.which('h5dump'):
        raise RuntimeError('h5dump is required on the compute node to validate checkpoint times')
    meta = json.loads(Path(manifest).read_text(encoding='utf-8'))
    raw = meta['args']
    raw['gs_filename'] = Path(raw['gs_filename'])
    a = argparse.Namespace(**raw)
    n, dt = a.Pump_time_steps, a.TSPTau
    folder = Path(meta['work']) / f'step_{probe:04d}'
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copy2(Path(meta['executable']), folder / 'dmrg')
    # Two padding points cover the look-ahead of the one-step Krylov targets.
    times = [round(i * dt, 9) for i in range(-2, 2 * n + 2)]
    pump = common.pump_axis(a, times)
    initial = Path(meta['gs']) if probe == 0 else Path(meta['td_folder']) / f'Recovery{probe - 1}{meta["run_name"]}.hd5'
    if not initial.is_file():
        raise FileNotFoundError(f'TD checkpoint not found: {initial}')
    time0 = probe * dt
    start, equal = run_stage(a, folder, times, pump, initial, 'initialize', [0, -1], 'initialize', 1, {0: time0, 1: time0})
    data = {(time0, site): value for site, value in equal.items()}
    single = dict(data)
    for direction in (1, -1):
        checkpoint = start
        intervals = n - 1 if direction > 0 else probe
        for j in range(intervals):
            before = (probe + direction * j) * dt
            after = (probe + direction * (j + 1)) * dt
            stem = ('forward' if direction > 0 else 'backward') + f'_{j:04d}'
            firstmap = [0, 1, -1] if j == 0 else [0, 2, -1]
            checkpoint, _ = run_stage(a, folder, times, pump, checkpoint, stem + '_reference', firstmap, 'reference', direction, {2: after, 1: before})
            checkpoint, _ = run_stage(a, folder, times, pump, checkpoint, stem + '_excited', [2, 1, -1], 'excited', direction, {0: after, 2: after})
            _, values = run_stage(a, folder, times, pump, checkpoint, stem + '_measure', [0, 2], 'measure_excited', direction, {0: after, 1: after})
            if -1e-9 <= after <= (n - 1) * dt + 1e-9:
                data.update({(after, site): value for site, value in values.items()})
            if direction > 0:
                single.update({(after, site): value for site, value in values.items()})
    processed = Path(meta['processed'])
    save_step(processed / f'XAS_twotime_{meta["run_name"]}' / f'step_{probe:04d}.csv', data, probe, dt, a.center_site, False)
    save_step(processed / f'XAS_singletime_{meta["run_name"]}' / f'step_{probe:04d}.csv', single, probe, dt, a.center_site, True)
    print(f'XAS probe {probe} completed', flush=True)


def aggregate(meta):
    a = meta['args']
    processed = Path(meta['processed'])
    td_folder = Path(meta['td_folder'])
    td_output = td_folder / f"runFor{Path(meta['td_input']).stem}.cout"
    td_processor = Path(__file__).resolve().with_name('my_hubbard_chain_td_targetingexpression_process.py')
    subprocess.run([sys.executable, str(td_processor), str(td_output), str(td_folder / 'pump_table.csv')], check=True)
    for kind in ('twotime', 'singletime'):
        directory = processed / f'XAS_{kind}_{meta["run_name"]}'
        output = processed / f'XAS_{kind}_{meta["run_name"]}_centersite={a["center_site"]}.csv'
        with output.open('w', newline='', encoding='utf-8') as f:
            writer = None
            for i in range(a['Pump_time_steps']):
                path = directory / f'step_{i:04d}.csv'
                with path.open(newline='', encoding='utf-8') as inp:
                    reader = csv.DictReader(inp)
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                        writer.writeheader()
                    writer.writerows(reader)
        print(f'Wrote {output}', flush=True)


def orchestrate(manifest):
    meta = json.loads(Path(manifest).read_text(encoding='utf-8'))
    a = meta['args']
    n = a['Pump_time_steps']
    total = int(os.environ.get('SLURM_CPUS_ON_NODE', os.cpu_count() or 1))
    cpus = a['cpus_per_task']
    if cpus > total:
        raise ValueError(f'cpus-per-task={cpus} exceeds allocated CPUs={total}')
    if a['cluster'] == 'nersc' and not os.environ.get('SLURM_JOB_ID'):
        raise ValueError('--run on NERSC requires a Slurm allocation')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        result = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    else:
        devices = [x.strip() for x in visible.split(',') if x.strip() and x.strip() != '-1']
    workers = min(n, len(devices), a['run_gpus'], cpus)
    if workers < 1:
        raise ValueError('No GPUs available for --run')
    cpus_per_worker = max(1, cpus // workers)
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(max(1, cpus_per_worker // 2) if a['cluster'] == 'nersc' else cpus_per_worker)
    env.setdefault('OMP_PLACES', 'cores')
    env['OMP_PROC_BIND'] = 'false'
    available = queue.Queue()
    for device in devices[:workers]: available.put(device)
    def execute(i):
        device = available.get()
        try:
            command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(manifest), str(i)]
            worker_env = env.copy()
            if a['cluster'] == 'nersc':
                command = ['srun', '--exclusive', '--exact', '--nodes=1', '--ntasks=1', f'--cpus-per-task={cpus_per_worker}', '--cpu-bind=cores', '--gpus-per-task=1', '--gpu-bind=closest'] + command
            else:
                worker_env['CUDA_VISIBLE_DEVICES'] = device
            subprocess.run(command, env=worker_env, check=True)
        finally:
            available.put(device)
    print(f'Running up to {workers} XAS probe pipelines concurrently', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in as_completed([pool.submit(execute, i) for i in range(n)]):
            result.result()
    aggregate(meta)


def write_xas_batch(a, td_batch, work, manifest):
    header = td_batch.read_text(encoding='utf-8').split('\ncd ', 1)[0]
    header = header.replace('#SBATCH --nodes=1\n', f'#SBATCH --nodes={a.xas_worker_nodes}\n').replace('#SBATCH --ntasks=1\n', f'#SBATCH --ntasks={a.xas_worker_nodes}\n').replace('#SBATCH --job-name=dmrg_td', '#SBATCH --job-name=dmrg_xas')
    script = work / 'batch_xas.slurm'
    lines = [header, 'pids=()', 'failed=0', f'limit={a.xas_worker_nodes}']
    for i in range(a.Pump_time_steps):
        cmd = shlex.join(['python3', str(Path(__file__).resolve()), '--worker', str(manifest), str(i)])
        lines += [f'srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task={a.cpus_per_task} --gpus-per-task=4 {cmd} &', 'pids+=("$!")', 'if (( ${#pids[@]} >= limit )); then wait "${pids[0]}" || failed=1; pids=("${pids[@]:1}"); fi']
    lines += ['for pid in "${pids[@]}"; do wait "$pid" || failed=1; done', 'if (( failed )); then exit 1; fi', shlex.join(['python3', str(Path(__file__).resolve()), '--aggregate', str(manifest)])]
    script.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return script


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        worker(Path(sys.argv[2]), int(sys.argv[3])); return
    if len(sys.argv) == 3 and sys.argv[1] == '--aggregate':
        aggregate(json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))); return
    p = args_parser(); a = p.parse_args()
    if a.sites <= 2 or a.TSPAdvanceEach != a.sites - 2 or not 0 <= a.center_site < a.sites:
        p.error('sites > 2, TSPAdvanceEach = sites - 2 and a valid center_site are required')
    if a.Pump_time_steps < 1 or a.finite_kept < 1 or a.cpus_per_task < 1 or a.run_gpus < 1 or a.TSPTau <= 0 or a.Pump_pulse_width <= 0:
        p.error('time steps, finite kept, CPUs, TSPTau and pulse width must be positive')
    if not math.isfinite(a.core_hole_potential) or a.core_hole_potential < 0:
        p.error('core-hole-potential must be finite and nonnegative')
    if a.submit and a.cluster != 'nersc': p.error('--submit requires nersc')
    if a.xas_worker_nodes is None: a.xas_worker_nodes = a.Pump_time_steps
    if a.xas_worker_nodes < 1: p.error('xas-worker-nodes must be positive')
    if a.run and not shutil.which('h5dump'): p.error('h5dump is required to validate checkpoints')
    gs = a.gs_filename.resolve(); exe = common.DMRG_EXECUTABLES[a.cluster]
    if not gs.is_file() or not exe.is_file(): p.error(f'Ground-state or executable not found: {gs}, {exe}')
    stamp = datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')
    base = f'N={a.sites}_Nup={a.up}_Ndown={a.down}_{stamp}'
    run_name = f'td_{base}_{common.frequency_tag(a.pump_frequency)}'
    parent = gs.parent; td = parent / run_name; td.mkdir()
    work = parent / f'XAS_staged_{run_name}'; work.mkdir()
    processed = parent / 'processed_data'; processed.mkdir(exist_ok=True)
    shutil.copy2(exe, td / 'dmrg')
    times = common.time_axis(a); pump = common.pump_axis(a, times)
    common.write_pump(times, pump, td / 'pump_table.csv')
    inp = td / f'input_td_{base}.ain'; inp.write_text(common.td_input(a, run_name, times, pump, f'../{gs.name}'), encoding='utf-8')
    meta = {'args': {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, 'td_folder': str(td), 'work': str(work),
            'gs': str(gs), 'executable': str(exe), 'processed': str(processed), 'run_name': run_name, 'td_input': str(inp)}
    manifest = work / 'manifest.json'; manifest.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    td_batch = common.write_td_batch(a, td, inp) if a.cluster == 'nersc' else None
    script = write_xas_batch(a, td_batch, work, manifest) if td_batch else None
    print(f'TD folder: {td}\nXAS work folder: {work}\nManifest: {manifest}', flush=True)
    if script:
        print(f'TD batch: {td_batch}\nXAS batch: {script}', flush=True)
    if a.run:
        if a.cluster == 'nersc' and not os.environ.get('SLURM_JOB_ID'): p.error('--run requires a Slurm allocation on NERSC')
        env = os.environ.copy(); env['OMP_NUM_THREADS'] = str(max(1, a.cpus_per_task // 2) if a.cluster == 'nersc' else a.cpus_per_task)
        with (td / 'td_streamed.stdout').open('w') as out, (td / 'td_streamed.stderr').open('w') as err:
            subprocess.run(['./dmrg', '-f', inp.name, '-p', str(common.DMRG_PRECISION), common.TD_OPERATORS], cwd=td, env=env, stdout=out, stderr=err, check=True)
        orchestrate(manifest)
    elif a.submit:
        result = subprocess.run(['sbatch', str(td_batch)], cwd=td, text=True, capture_output=True, check=True)
        print(result.stdout.strip(), flush=True)
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if not match: raise RuntimeError('Could not parse submitted TD job ID')
        subprocess.run(['sbatch', f'--dependency=afterok:{match[1]}', str(script)], cwd=work, check=True)
    else:
        print('Generated TD inputs and XAS manifest; use --run or --submit to execute.', flush=True)


if __name__ == '__main__':
    main()
