#!/usr/bin/env python3
"""Generate a streamed, single-allocation TD -> G< / G> Hubbard-chain workflow.

TD runs on one node.  As each TD recovery file becomes available, one exclusive
node runs the complete G< step and another runs the complete G> step:
apply -> forward evolution -> backward evolution.  The original TD and G
scripts are deliberately not used or modified by this script.
"""

import argparse
import atexit
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

DMRG_EXECUTABLES = {
    "local": Path("/Users/qqt/Documents/Codes/dmrgpp_backwardevolve/copy_dmrg/build/dmrg/dmrg"),
    "nersc": Path("/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/10_4sitetestruns_GPU/copy_dmrg/build_perlmutter_nvidia_cuda/dmrg/dmrg"),
}
DMRG_PRECISION = 12
TD_DEFINE_OPERATORS = "double:nup*ndown,hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,parity:identity+(-2.0)*n+4.0*nup*ndown,local_moment:0.75*n+(-1.5)*nup*ndown"
TD_OPERATORS = "<P0|n|P0>,<gs|n|gs>,<P0|sz|P0>,<gs|sz|gs>,<P0|local_moment|P0>,<gs|local_moment|gs>,<P0|hole|P0>,<gs|hole|gs>,<P0|parity|P0>,<gs|parity|gs>,<P0|double|P0>,<gs|double|gs>,<P0|n*n|P0>,<gs|n*n|gs>"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create one streamed TD + G< + G> Slurm workflow.")
    p.add_argument("sites", type=int)
    p.add_argument("up", type=int)
    p.add_argument("down", type=int)
    p.add_argument("t", type=float)
    p.add_argument("U", type=float)
    p.add_argument("potentialV", type=float)
    p.add_argument("gs_filename", type=Path)
    p.add_argument("finite_kept", type=int)
    p.add_argument("TSPAdvanceEach", type=int)
    p.add_argument("TSPTau", type=float)
    p.add_argument("Pump_Amplitude", type=float)
    p.add_argument("Pump_time_delay", type=float)
    p.add_argument("Pump_pulse_width", type=float)
    p.add_argument("Pump_time_steps", type=int, help="Number of physical pump times")
    p.add_argument("cluster", choices=DMRG_EXECUTABLES)
    p.add_argument("center_site", type=int)
    p.add_argument("finite_loops_apply", type=int)
    p.add_argument("--pump-frequency", type=float, required=True)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--submit", action="store_true", help="Submit the generated Slurm batch script")
    execution.add_argument("--run", action="store_true", help="Run TD and all G steps serially on the current compute node")
    p.add_argument("--cpus-per-task", type=int, default=128)
    p.add_argument(
        "--g-worker-nodes", type=int, default=None,
        help="Exclusive nodes available to G work (default: 2 per time step).",
    )
    p.add_argument(
        "--initial-g-delay", type=int, default=60,
        help="Seconds to wait after TD starts before launching G step 0000 (default: 60).",
    )
    p.add_argument(
        "--recovery-poll-seconds", type=int, default=30,
        help="Seconds between checks for the next TD recovery file (default: 30).",
    )
    p.add_argument(
        "--recovery-settle-seconds", type=int, default=30,
        help="Extra seconds to wait after the next recovery file appears (default: 30).", 
    )
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    if a.submit and a.cluster != "nersc":
        raise ValueError("--submit requires cluster=nersc")
    if a.sites <= 2 or a.TSPAdvanceEach != a.sites - 2:
        raise ValueError("TSPAdvanceEach must equal sites - 2, with sites > 2")
    if a.Pump_time_steps < 1 or a.finite_kept < 1 or a.finite_loops_apply < 1:
        raise ValueError("Pump_time_steps, finite_kept, and finite_loops_apply must be positive")
    if not 0 <= a.center_site < a.sites:
        raise ValueError("center_site must be a valid site index")
    if a.cpus_per_task < 1 or a.initial_g_delay < 0 or a.recovery_poll_seconds < 1 or a.recovery_settle_seconds < 0:
        raise ValueError("Invalid worker, delay, or polling value")
    if a.g_worker_nodes is None:
        a.g_worker_nodes = 2 * a.Pump_time_steps
    if a.g_worker_nodes < 2:
        raise ValueError("g-worker-nodes must be at least 2 so G< and G> can launch as a pair")


def frequency_tag(value: float) -> str:
    return f"freq_{value:g}".replace(".", "p")


def time_axis(a: argparse.Namespace) -> list[float]:
    maximum = (a.Pump_time_steps - 1) * a.TSPTau
    return [-0.1, *(round(i * a.TSPTau, 9) for i in range(a.Pump_time_steps)), round(maximum + 0.1, 9)]


def pump_axis(a: argparse.Namespace, times: list[float]) -> list[float]:
    values = []
    for time in times:
        value = a.Pump_Amplitude * math.exp(-0.5 * ((time - a.Pump_time_delay) / a.Pump_pulse_width) ** 2) * math.cos(a.pump_frequency * (time - a.Pump_time_delay))
        values.append(round(value, 7) or 0.0)
    if next(value for time, value in zip(times, values) if time == 0.0) != 0.0:
        raise ValueError("Pump signal at time 0 must be zero")
    return values


def rows(count: int, a: argparse.Namespace) -> str:
    return ",\n".join(f"    [@auto, {a.finite_kept}, 2]" for _ in range(count))


def table(times: list[float], pump: list[float]) -> str:
    return "\n".join(f"    [{t:.15g}, {v:.15g}]," for t, v in zip(times, pump)).rstrip(",")


def model_block(a: argparse.Namespace) -> str:
    return "\n".join((
        f"TotalNumberOfSites = {a.sites};", "NumberOfTerms = 1;", "DegreesOfFreedom = 1;",
        'GeometryKind = "chain";', 'GeometryOptions = "ConstantValues";', f"dir0:Connectors = [{a.t}];",
        f"hubbardU = [{a.U}, ...];", f"potentialV = [{a.potentialV}, ...];", 'Model = "HubbardOneBand";',
    ))


def td_input(a: argparse.Namespace, output: str, times: list[float], pump: list[float], restart: str) -> str:
    loops = a.Pump_time_steps * (a.TSPAdvanceEach // (a.sites - 2))
    return "\n\n".join((
        "##Ainur1.0", model_block(a),
        "\n".join(("matrix AversusTime = [", table(times, pump), "];", "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;")),
        "\n".join((f"TargetElectronsUp = {a.up};", f"TargetElectronsDown = {a.down};", f'DefineOperators="{TD_DEFINE_OPERATORS}";')),
        "\n".join(("TruncationTolerance = 1e-10;", "FiniteLoops = [", rows(loops, a), "];", "RestartMapStages=0;", "RestartMappingTvs=[-1];", f'string P0="TimeEvolve{{tau={a.TSPTau},steps=5,advanceEach={a.TSPAdvanceEach}}}*|gs>";')),
        "\n".join(('SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead,BatchedGemm";', 'Version = "stc_vs_ttc";', f'string RecoverySave = "%l%%1,@keep,@M={a.Pump_time_steps}";', f'OutputFile = "{output}";', f'RestartFilename = "{restart}";', "GsWeight = 0.0;")),
    )) + "\n"


def apply_input(a: argparse.Namespace, output: str, time: float, pump: float, restart: str, creator: bool) -> str:
    op = f"c[{a.center_site}]" + ("'" if creator else "")
    return "\n\n".join((
        "##Ainur1.0", model_block(a),
        "\n".join(("matrix AversusTime = [", f"    [{time:.15g}, {pump:.15g}]", "];", f"GeometryFactor = exp:*:1.0i:!readTableAversusTime,{time:.15g};")),
        f"TargetElectronsUp = {a.up};\nTargetElectronsDown = {a.down};",
        "\n".join(("TruncationTolerance = 1e-10;", "FiniteLoops = [", rows(a.finite_loops_apply, a), "];", "RestartMapStages=0;", "RestartMappingTvs=[0,-1,-1,-1];", "string P0 = |P0>;", f'string P1 = "{op}*|P0>";')),
        "\n".join(('SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,BatchedGemm";', 'Version = "stc_vs_ttc_apply";', f'OutputFile = "{output}_apply";', f'RestartFilename = "{restart}";', "GsWeight = 0.0;")),
    )) + "\n"


def evolve_input(a: argparse.Namespace, output: str, times: list[float], pump: list[float], step: int, forward: bool) -> str:
    # step is the index in the padded table where the operator was applied.
    selected = list(zip(times[step:], pump[step:])) if forward else list(reversed(list(zip(times[:step + 1], pump[:step + 1]))))
    count = len(selected) * (a.TSPAdvanceEach // (a.sites - 2)) - 1
    if forward and len(selected) == 1:
        count = 1
        expressions = ('RestartMappingTvs=[0, 1, -1, -1];', 'string P0="|P0>";', 'string P1="|P1>";', 'string P2="|P0>";', 'string P3="|P1>";')
    else:
        tau = a.TSPTau if forward else -a.TSPTau
        expressions = ('RestartMappingTvs=[0, 1, -1, -1];', 'string P0="|P0>";', 'string P1="|P1>";', f'string P2="TimeEvolve{{tau={tau},steps=5,advanceEach={a.TSPAdvanceEach}}}*|P0>";', f'string P3="TimeEvolve{{tau={tau},steps=5,advanceEach={a.TSPAdvanceEach}}}*|P1>";')
    stage = "evolve_forward" if forward else "evolve_backward"
    return "\n\n".join((
        "##Ainur1.0", model_block(a),
        "\n".join(("matrix AversusTime = [", table([x[0] for x in selected], [x[1] for x in selected]), "];", "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;")),
        f"TargetElectronsUp = {a.up};\nTargetElectronsDown = {a.down};",
        "\n".join(("TruncationTolerance = 1e-10;", "FiniteLoops = [", rows(count, a), "];", "RestartMapStages=0;", *expressions)),
        "\n".join(('SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead,BatchedGemm";', 'Version = "stc_vs_ttc_apply_evolve";', 'string RecoverySave = "%l%%1,@keep,@M=1";', f'OutputFile = "{output}_{stage}";', f'RestartFilename = "{output}_apply";', "GsWeight = 0.0;")),
    )) + "\n"


def write_pump(times: Iterable[float], pump: Iterable[float], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time", "pump"))
        writer.writerows((f"{t:.7f}", f"{p:.7f}") for t, p in zip(times, pump))


def write_g_inputs(a: argparse.Namespace, parent: Path, td_folder: Path, run_name: str, times: list[float], pump: list[float], creator: bool, executable: Path) -> Path:
    prefix = "G<" if creator else "G>"
    folder = parent / f"{prefix}_twotime_{run_name}"
    folder.mkdir(parents=True, exist_ok=True)
    output = f"twotime_{run_name}"
    for physical_step in range(a.Pump_time_steps):
        step = physical_step + 1
        work = folder / f"step_{physical_step:04d}"
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executable, work / "dmrg")
        restart = f"../../{a.gs_filename.resolve().name}" if step == 1 else f"../../{run_name}/Recovery{step - 2}{run_name}"
        # G< applies annihilation and measures creation; G> does the reverse.
        (work / f"input_twotime_{run_name}_apply.ain").write_text(apply_input(a, output, times[step], pump[step], restart, not creator), encoding="utf-8")
        (work / f"input_twotime_{run_name}_evolve_forward.ain").write_text(evolve_input(a, output, times, pump, step, True), encoding="utf-8")
        (work / f"input_twotime_{run_name}_evolve_backward.ain").write_text(evolve_input(a, output, times, pump, step, False), encoding="utf-8")
    return folder


def write_batch(a: argparse.Namespace, td_folder: Path, gless: Path, ggreater: Path, run_name: str) -> Path:
    """Write one allocation: TD is streamed while paired G tasks consume recoveries."""
    path = td_folder / f"batch_streamed_td_G_{run_name}.slurm"
    nodes = 1 + a.g_worker_nodes
    post_commands = postprocess_commands(a, td_folder, gless, ggreater, run_name)
    td_post, less_post, greater_post = (shlex.join(command) for command in post_commands)
    # A later recovery appearing means TD has closed the prior recovery.
    body = f'''#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes={nodes}
#SBATCH --ntasks={nodes}
#SBATCH --cpus-per-task={a.cpus_per_task}
#SBATCH --gpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --job-name=dmrg_td_streamed_g
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail
module load cray-mpich/9.0.1
module load cray-libsci/25.09.0
module load cray-hdf5/1.14.3.7
module load PrgEnv-nvidia
module load cudatoolkit
export OMP_NUM_THREADS={max(1, a.cpus_per_task // 2)}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

TD_FOLDER="{td_folder}"
GLESS_FOLDER="{gless}"
GGREATER_FOLDER="{ggreater}"
RUN_NAME="{run_name}"
TIME_STEPS={a.Pump_time_steps}
CPUS_PER_TASK={a.cpus_per_task}
G_WORKER_NODES={a.g_worker_nodes}
INITIAL_G_DELAY={a.initial_g_delay}
POLL_SECONDS={a.recovery_poll_seconds}
SETTLE_SECONDS={a.recovery_settle_seconds}
PRECISION={DMRG_PRECISION}

# Initialize a status for every planned step before launching TD.
for (( step=0; step<TIME_STEPS; ++step )); do
  step_name="step_$(printf '%04d' "$step")"
  printf '%s\\n' 'not started' > "$GLESS_FOLDER/$step_name/.g_pipeline_status"
  printf '%s\\n' 'not started' > "$GGREATER_FOLDER/$step_name/.g_pipeline_status"
done
write_failure_report() {{
  local status="$?" folder step_name state step
  local report="$TD_FOLDER/failed_g_steps.txt"
  trap - EXIT
  : > "$report"
  for folder in "$GLESS_FOLDER" "$GGREATER_FOLDER"; do
    for (( step=0; step<TIME_STEPS; ++step )); do
      step_name="step_$(printf '%04d' "$step")"
      state=$(<"$folder/$step_name/.g_pipeline_status") || state="unknown"
      if [[ "$state" != complete ]]; then
        printf '%s\\t%s\\n' "$state" "$folder/$step_name" >> "$report"
      fi
    done
  done
  if [[ -s "$report" ]]; then
    echo "Incomplete G steps listed in $report" >&2
  fi
  exit "$status"
}}
trap write_failure_report EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# TD permanently occupies one node. Its exit status is checked at the end.
(
  cd "$TD_FOLDER"
  srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" \\
    ./dmrg -f {shlex.quote('input_td_' + run_name.removeprefix('td_').removesuffix('_' + frequency_tag(a.pump_frequency)) + '.ain')} -p "$PRECISION" "{TD_OPERATORS}"
) >"$TD_FOLDER/td_streamed.stdout" 2>"$TD_FOLDER/td_streamed.stderr" &
TD_PID=$!

active=0
G_PIDS=()
launch_g_step() {{
  local folder="$1"
  local step="$2"
  local apply_op="$3"
  local twotime_op="$4"
  local step_folder="$folder/step_$(printf '%04d' "$step")"
  printf '%s\\n' running > "$step_folder/.g_pipeline_status"
  (
    cd "$step_folder" || exit 1
    if srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" \\
          bash -c '
            set -e
            run_name="$1"; precision="$2"; apply_op="$3"; twotime_op="$4"
            ./dmrg -f "input_twotime_${{run_name}}_apply.ain" -p "$precision" "$apply_op"
            ./dmrg -f "input_twotime_${{run_name}}_evolve_forward.ain" -p "$precision" "$twotime_op"
            ./dmrg -f "input_twotime_${{run_name}}_evolve_backward.ain" -p "$precision" "$twotime_op"
          ' _ "$RUN_NAME" "$PRECISION" "$apply_op" "$twotime_op"; then
      printf '%s\\n' complete > .g_pipeline_status
      exit 0
    fi
    echo "G pipeline failed: $folder step $step" >&2
    printf '%s\\n' failed > .g_pipeline_status
    exit 1
  ) >"$folder/step_$(printf '%04d' "$step")/streamed.stdout" \\
    2>"$folder/step_$(printf '%04d' "$step")/streamed.stderr" &
  G_PIDS+=("$!")
  active=$((active + 1))
}}

reap_one() {{
  # Portable to older Bash: wait for the oldest tracked G pipeline.
  # Other pipelines continue running while this one is awaited.
  local pid="${{G_PIDS[0]}}"
  if ! wait "$pid"; then
    echo "G pipeline failed (PID $pid); continuing other steps" >&2
  fi
  G_PIDS=("${{G_PIDS[@]:1}}")
  active=$((active - 1))
}}
wait_for_pair_slots() {{
  while (( active + 2 > G_WORKER_NODES )); do reap_one; done
}}
wait_for_recovery() {{
  local recovery="$1"
  local next_recovery="$2"
  while [[ ! -e "$next_recovery" ]]; do
    if ! kill -0 "$TD_PID" 2>/dev/null; then
      # If TD finished successfully, its final recovery is closed even when
      # no subsequent recovery exists. A failed TD must stop the workflow.
      if ! wait "$TD_PID"; then
        echo "TD failed before recovery was ready: $recovery" >&2
        exit 1
      fi
      if [[ ! -e "$recovery" ]]; then
        echo "TD completed without required recovery: $recovery" >&2
        exit 1
      fi
      return
    fi
    sleep "$POLL_SECONDS"
  done
  if [[ ! -e "$recovery" ]]; then
    echo "Next recovery exists but required recovery is missing: $recovery" >&2
    exit 1
  fi
  sleep "$SETTLE_SECONDS"
}}

# Step 0000 restarts from the GS file. Start it only after the requested TD lead time.
sleep "$INITIAL_G_DELAY"
wait_for_pair_slots
launch_g_step "$GLESS_FOLDER" 0 "<P0|c'|P1>" "<P2|c'|P3>"
launch_g_step "$GGREATER_FOLDER" 0 "<P0|c|P1>" "<P2|c|P3>"

# Wait for the next recovery to appear, indicating the required one is closed.
# If there is no next recovery, successful TD completion is the fallback.
for (( step=1; step<TIME_STEPS; ++step )); do
  wait_for_recovery "$TD_FOLDER/Recovery$((step - 1))$RUN_NAME" \\
    "$TD_FOLDER/Recovery${{step}}$RUN_NAME"
  wait_for_pair_slots
  launch_g_step "$GLESS_FOLDER" "$step" "<P0|c'|P1>" "<P2|c'|P3>"
  launch_g_step "$GGREATER_FOLDER" "$step" "<P0|c|P1>" "<P2|c|P3>"
done

while (( active > 0 )); do reap_one; done
wait "$TD_PID"
FAILURE_LOG="$TD_FOLDER/failed_g_steps.txt"
less_failed=0
greater_failed=0
for (( step=0; step<TIME_STEPS; ++step )); do
  step_name="step_$(printf '%04d' "$step")"
  [[ $(<"$GLESS_FOLDER/$step_name/.g_pipeline_status") == complete ]] || less_failed=1
  [[ $(<"$GGREATER_FOLDER/$step_name/.g_pipeline_status") == complete ]] || greater_failed=1
done
{td_post}
if (( ! less_failed )); then {less_post}; else echo "Skipping incomplete G< post-processing" >&2; fi
if (( ! greater_failed )); then {greater_post}; else echo "Skipping incomplete G> post-processing" >&2; fi
if (( less_failed || greater_failed )); then
  echo "Incomplete G steps will be listed in $FAILURE_LOG" >&2
  exit 1
fi
echo "TD and all streamed G< / G> step pipelines completed and post-processed."
'''
    path.write_text(body, encoding="utf-8")
    return path


def postprocess_commands(a: argparse.Namespace, td_folder: Path, gless: Path, ggreater: Path, run_name: str) -> list[list[str]]:
    """Build commands for the existing TD and two-time output processors."""
    scripts = Path(__file__).resolve().parent
    pump = str(td_folder / "pump_table.csv")
    td_output = td_folder / f"runForinput_td_{run_name.removeprefix('td_').removesuffix('_' + frequency_tag(a.pump_frequency))}.cout"
    commands = [["python3", str(scripts / "my_hubbard_chain_td_targetingexpression_process.py"), str(td_output), pump]]
    for prefix, folder in (("G<", gless), ("G>", ggreater)):
        commands.append([
            "python3", str(scripts / f"my_hubbard_chain_twotime_{prefix}_parallel_process.py"),
            str(a.center_site), str(a.gs_filename.resolve()), pump, str(folder), "--workers", "1",
        ])
    return commands


def run_serial(a: argparse.Namespace, td_folder: Path, td_input_path: Path, gless: Path, ggreater: Path, run_name: str) -> None:
    """Run TD, then each G< and G> step sequentially on the current node."""
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(
        max(1, a.cpus_per_task // 2) if a.cluster == "nersc" else a.cpus_per_task
    )
    environment.setdefault("OMP_PLACES", "cores")
    environment.setdefault("OMP_PROC_BIND", "spread")

    def run(work: Path, input_name: str, operator: str) -> None:
        print(f"Running {work / input_name}", flush=True)
        subprocess.run(
            ["./dmrg", "-f", input_name, "-p", str(DMRG_PRECISION), operator],
            cwd=work, env=environment, check=True,
        )

    all_steps = [folder / f"step_{step:04d}" for folder in (gless, ggreater) for step in range(a.Pump_time_steps)]
    for work in all_steps:
        (work / ".g_pipeline_status").write_text("not started\n", encoding="utf-8")

    def write_failure_report() -> None:
        lines = []
        for work in all_steps:
            try:
                state = (work / ".g_pipeline_status").read_text(encoding="utf-8").strip()
            except OSError:
                state = "unknown"
            if state != "complete":
                lines.append(f"{state}\t{work}\n")
        failure_log = td_folder / "failed_g_steps.txt"
        failure_log.write_text("".join(lines), encoding="utf-8")
        if lines:
            print(f"Incomplete G steps listed in {failure_log}", flush=True)

    atexit.register(write_failure_report)
    run(td_folder, td_input_path.name, TD_OPERATORS)
    failed_steps = []
    for step in range(a.Pump_time_steps):
        for folder, apply_op, evolve_op in (
            (gless, "<P0|c'|P1>", "<P2|c'|P3>"),
            (ggreater, "<P0|c|P1>", "<P2|c|P3>"),
        ):
            work = folder / f"step_{step:04d}"
            (work / ".g_pipeline_status").write_text("running\n", encoding="utf-8")
            try:
                for stage, operator in (("apply", apply_op), ("evolve_forward", evolve_op), ("evolve_backward", evolve_op)):
                    run(work, f"input_twotime_{run_name}_{stage}.ain", operator)
                (work / ".g_pipeline_status").write_text("complete\n", encoding="utf-8")
            except subprocess.CalledProcessError:
                (work / ".g_pipeline_status").write_text("failed\n", encoding="utf-8")
                failed_steps.append(work)
                print(f"G pipeline failed for {work}; continuing other steps.", flush=True)
    failure_log = td_folder / "failed_g_steps.txt"
    write_failure_report()
    commands = postprocess_commands(a, td_folder, gless, ggreater, run_name)
    subprocess.run(commands[0], check=True)
    for folder, command in zip((gless, ggreater), commands[1:]):
        if any(work.parent == folder for work in failed_steps):
            print(f"Skipping incomplete post-processing: {folder}", flush=True)
        else:
            subprocess.run(command, check=True)
    if failed_steps:
        raise RuntimeError(f"Failed G steps to rerun are listed in {failure_log}")
    print("TD and all serial G< / G> step pipelines completed and post-processed.")


def main() -> int:
    a = parse_args()
    validate(a)
    gs = a.gs_filename.resolve()
    executable = DMRG_EXECUTABLES[a.cluster]
    if not gs.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {gs}")
    if not executable.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable}")

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_base = f"N={a.sites}_Nup={a.up}_Ndown={a.down}_{stamp}"
    tag = frequency_tag(a.pump_frequency)
    run_name = f"td_{run_base}_{tag}"
    parent = gs.parent
    td_folder = parent / run_name
    td_folder.mkdir(parents=True, exist_ok=True)
    times = time_axis(a)
    pump = pump_axis(a, times)

    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    (td_folder / f"input_args_td_{run_base}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    shutil.copy2(executable, td_folder / "dmrg")
    write_pump(times, pump, td_folder / "pump_table.csv")
    td_output = f"td_{run_base}_{tag}"
    td_input_path = td_folder / f"input_td_{run_base}.ain"
    td_input_path.write_text(td_input(a, td_output, times, pump, f"../{gs.name}"), encoding="utf-8")

    gless = write_g_inputs(a, parent, td_folder, run_name, times, pump, True, executable)
    ggreater = write_g_inputs(a, parent, td_folder, run_name, times, pump, False, executable)
    batch = write_batch(a, td_folder, gless, ggreater, run_name) if a.cluster == "nersc" else None
    print(f"Created TD folder: {td_folder}")
    print(f"Created G< folder: {gless}")
    print(f"Created G> folder: {ggreater}")
    if batch is not None:
        print(f"Wrote streamed batch script: {batch}")
    if a.submit:
        subprocess.run(["sbatch", str(batch)], cwd=td_folder, check=True)
    elif a.run:
        if a.cluster == "nersc" and not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("--run on NERSC requires an interactive Slurm compute-node allocation")
        run_serial(a, td_folder, td_input_path, gless, ggreater, run_name)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
