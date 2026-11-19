#!/usr/bin/env python3
"""Create the TD run and the dependent G< / G> runs for a Hubbard chain.

Workflow:
  1. TD: time evolution of the ground state under the pump (one Slurm job or local run).
  2. G< and G>: two-time correlators, one pipeline per pump time, started from the TD checkpoints.
  4. Post-processing: CSV files written next to the ground state.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from dmrg_config import (
    DMRG_EXECUTABLES,
    DMRG_PRECISION,
    RECOVERY_FILES_KEPT,
    batch_environment,
    link_binary,
    sbatch_header,
)

# Operators measured by the TD run (DMRG++ DefineOperators syntax).
TD_DEFINE_OPERATORS = (
    "double:nup*ndown,hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,"
    "parity:identity+(-2.0)*n+4.0*nup*ndown,local_moment:0.75*n+(-1.5)*nup*ndown"
)
# Expectation values printed by the TD run, in the order the processors expect.
TD_OPERATORS = (
    "<P0|n|P0>,<gs|n|gs>,<P0|sz|P0>,<gs|sz|gs>,<P0|local_moment|P0>,<gs|local_moment|gs>,"
    "<P0|hole|P0>,<gs|hole|gs>,<P0|parity|P0>,<gs|parity|gs>,<P0|double|P0>,<gs|double|gs>,"
    "<P0|n*n|P0>,<gs|n*n|gs>"
)


# ---------------------------------------------------------------------------
# Command line and checks
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Read the command line. CPUs and walltime have cluster-dependent defaults."""
    p = argparse.ArgumentParser(description="Create TD and dependent combined G< + G> Slurm jobs.")
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
    p.add_argument("--pump-frequency", type=float, required=True)
    execution = p.add_mutually_exclusive_group()
    execution.add_argument("--submit", action="store_true", help="Submit the generated Slurm batch script")
    execution.add_argument("--run", action="store_true", help="Run TD, then G pipelines concurrently on the current node")
    p.add_argument(
        "--cpus-per-task", type=int, default=None,
        help="CPUs per task (default: local core count for local, 256 for nersc)",
    )
    p.add_argument("--time", dest="walltime", default="16:00:00", help="Slurm walltime for the TD and G batch jobs (HH:MM:SS)")
    p.add_argument(
        "--g-worker-nodes", type=int, default=None,
        help="Exclusive nodes available to G work (default: 2 per time step).",
    )
    args = p.parse_args()
    if args.cpus_per_task is None:
        args.cpus_per_task = (os.cpu_count() or 1) if args.cluster == "local" else 256
    return args


def validate(a: argparse.Namespace) -> None:
    """Reject inconsistent options before anything is written."""
    if a.submit and a.cluster != "nersc":
        raise ValueError("--submit requires cluster=nersc")
    # One advance per sweep requires TSPAdvanceEach = sites - 2.
    if a.sites <= 2 or a.TSPAdvanceEach != a.sites - 2:
        raise ValueError("TSPAdvanceEach must equal sites - 2, with sites > 2")
    if a.Pump_time_steps < 1 or a.finite_kept < 1:
        raise ValueError("Pump_time_steps and finite_kept must be positive")
    if not (a.TSPTau > 0 and a.Pump_pulse_width > 0):
        raise ValueError("TSPTau and Pump_pulse_width must be positive")
    if not 0 <= a.center_site < a.sites:
        raise ValueError("center_site must be a valid site index")
    if a.cpus_per_task < 1:
        raise ValueError("cpus-per-task must be positive")
    if a.g_worker_nodes is None:
        a.g_worker_nodes = 2 * a.Pump_time_steps
    # G< and G> run as a pair, so at least two nodes are needed.
    if a.g_worker_nodes < 2:
        raise ValueError("g-worker-nodes must be at least 2 so G< and G> can launch as a pair")


# ---------------------------------------------------------------------------
# Time and pump axes
# ---------------------------------------------------------------------------

def frequency_tag(value: float) -> str:
    """Folder-name tag for a pump frequency, e.g. 0.5 -> freq_0p5."""
    return f"freq_{value:g}".replace(".", "p")


def time_axis(a: argparse.Namespace) -> list[float]:
    """Pump times: one leading point (-TSPTau), the N physical times, one trailing point."""
    maximum = (a.Pump_time_steps - 1) * a.TSPTau
    physical = [round(i * a.TSPTau, 9) for i in range(a.Pump_time_steps)]
    return [-a.TSPTau, *physical, round(maximum + a.TSPTau, 9)]


def pump_axis(a: argparse.Namespace, times: list[float]) -> list[float]:
    """Gaussian-enveloped cosine pump evaluated at each time."""
    values = []
    for time in times:
        envelope = math.exp(-0.5 * ((time - a.Pump_time_delay) / a.Pump_pulse_width) ** 2)
        carrier = math.cos(a.pump_frequency * (time - a.Pump_time_delay))
        value = a.Pump_Amplitude * envelope * carrier
        if time == 0.0:
            # The initial state is the unperturbed ground state, so the pump must be off at t = 0.
            if abs(value) > 1e-7:
                raise ValueError("Pump signal at time 0 must be within 1e-7 of zero")
            value = 0.0
        values.append(round(value, 7) or 0.0)
    return values


def write_pump(times: Iterable[float], pump: Iterable[float], path: Path) -> None:
    """Save the time/pump table used by the processors."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("time", "pump"))
        writer.writerows((f"{t:.7f}", f"{p:.7f}") for t, p in zip(times, pump))


# ---------------------------------------------------------------------------
# DMRG++ input text (kept as whole builders: each returns one complete input file)
# ---------------------------------------------------------------------------

def rows(count: int, a: argparse.Namespace, flags: int = 2) -> str:
    """FiniteLoops rows. flags is the bit field: 1 = save, 2 = onlyfastwft, 3 = both."""
    return ",\n".join(f"    [@auto, {a.finite_kept}, {flags}]" for _ in range(count))


def table(times: list[float], pump: list[float]) -> str:
    """The AversusTime matrix body: one [time, value] row per entry."""
    return "\n".join(f"    [{t:.15g}, {v:.15g}]," for t, v in zip(times, pump)).rstrip(",")


def model_block(a: argparse.Namespace) -> str:
    """Model section shared by all DMRG++ inputs."""
    return "\n".join((
        f"TotalNumberOfSites = {a.sites};",
        "NumberOfTerms = 1;",
        "DegreesOfFreedom = 1;",
        'GeometryKind = "chain";',
        'GeometryOptions = "ConstantValues";',
        f"dir0:Connectors = [{a.t}];",
        f"hubbardU = [{a.U}, ...];",
        f"potentialV = [{a.potentialV}, ...];",
        'Model = "HubbardOneBand";',
    ))


def td_input(a: argparse.Namespace, output: str, times: list[float], pump: list[float], restart: str) -> str:
    """Complete TD input: starts from the ground state and evolves under the pump."""
    loops = a.Pump_time_steps * (a.TSPAdvanceEach // (a.sites - 2))
    sections = (
        "##Ainur1.0",
        model_block(a),
        "\n".join((
            "matrix AversusTime = [",
            table(times, pump),
            "];",
            "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
        )),
        "\n".join((
            f"TargetElectronsUp = {a.up};",
            f"TargetElectronsDown = {a.down};",
            f'DefineOperators="{TD_DEFINE_OPERATORS}";',
        )),
        "\n".join((
            "TruncationTolerance = 1e-10;",
            "FiniteLoops = [",
            rows(loops, a, 3),
            "];",
            "RestartMapStages=0;",
            "RestartMappingTvs=[-1];",
            f'string P0="TimeEvolve{{tau={a.TSPTau},steps=5,advanceEach={a.TSPAdvanceEach}}}*|gs>";',
        )),
        "\n".join((
            'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead";',
            'Version = "stc_vs_ttc";',
            f'string RecoverySave = "%l%%1,@keep,@M={a.Pump_time_steps}";',
            f'OutputFile = "{output}";',
            f'RestartFilename = "{restart}";',
            "GsWeight = 0.0;",
        )),
    )
    return "\n\n".join(sections) + "\n"


def evolve_input(a: argparse.Namespace, output: str, times: list[float], pump: list[float],
                 step: int, forward: bool, restart: str, creator: bool) -> str:
    """Complete two-time evolve input for one G step, forward or backward in time.

    Loop count: the table entries after the restart point, plus 8. Loops 0-8 are warm-up
    (startLoop=9); each later loop advances the clock once. The trailing padding entry is
    reached by the clock but not printed.
    """
    if forward:
        selected = list(zip(times[step:], pump[step:]))
    else:
        selected = list(reversed(list(zip(times[:step + 1], pump[:step + 1]))))
    count = len(selected) * (a.TSPAdvanceEach // (a.sites - 2)) + 8
    tau = a.TSPTau if forward else -a.TSPTau
    # G< uses c and measures c'; G> uses c' and measures c. Keep these in step with the processors.
    op = f"c[{a.center_site}]" + ("'" if creator else "")
    expressions = (
        "RestartMappingTvs=[0, -1, -1, -1];",
        'string P0="|P0>";',
        f'string P1="{op}*|P0>";',
        f'string P2="TimeEvolve{{tau={tau},steps=5,startLoop=9,advanceEach={a.TSPAdvanceEach},depends=P0}}*|P0>";',
        f'string P3="TimeEvolve{{tau={tau},steps=5,startLoop=9,advanceEach={a.TSPAdvanceEach},depends=P1}}*|P1>";',
    )
    stage = "evolve_forward" if forward else "evolve_backward"
    sections = (
        "##Ainur1.0",
        model_block(a),
        "\n".join((
            "matrix AversusTime = [",
            table([x[0] for x in selected], [x[1] for x in selected]),
            "];",
            "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
        )),
        f"TargetElectronsUp = {a.up};\nTargetElectronsDown = {a.down};",
        "\n".join((
            "TruncationTolerance = 1e-10;",
            "FiniteLoops = [",
            rows(count, a),
            "];",
            "RestartMapStages=0;",
            *expressions,
        )),
        "\n".join((
            'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead";',
            'Version = "stc_vs_ttc_apply_evolve";',
            f'string RecoverySave = "%l%%1,@keep,@M={RECOVERY_FILES_KEPT}";',
            f'OutputFile = "{output}_{stage}";',
            f'RestartFilename = "{restart}";',
            "GsWeight = 0.0;",
        )),
    )
    return "\n\n".join(sections) + "\n"


# ---------------------------------------------------------------------------
# Folder and file creation
# ---------------------------------------------------------------------------

def write_g_inputs(a: argparse.Namespace, parent: Path, td_folder: Path, run_name: str,
                   times: list[float], pump: list[float], creator: bool) -> Path:
    """Create the G< (creator=True) or G> (creator=False) folder with one step folder per pump time."""
    prefix = "G<" if creator else "G>"
    folder = parent / f"{prefix}_twotime_{run_name}"
    if any(folder.glob("step_*/input_*_apply.ain")):
        raise ValueError(f"Existing G folder uses a separate apply stage; move it aside before using condensed inputs: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    output = f"twotime_{run_name}"
    for physical_step in range(a.Pump_time_steps):
        step = physical_step + 1  # evolve_input is indexed one ahead of the physical step
        work = folder / f"step_{physical_step:04d}"
        work.mkdir(parents=True, exist_ok=True)
        link_binary(td_folder / "dmrg", work)
        # Step 1 restarts from the ground state; later steps restart from the TD checkpoint.
        if step == 1:
            restart = f"../../{a.gs_filename.resolve().name}"
        else:
            restart = f"../../{run_name}/Recovery{step - 2}{run_name}"
        # G< and G> pass opposite creator flags to evolve_input.
        forward_text = evolve_input(a, output, times, pump, step, True, restart, not creator)
        backward_text = evolve_input(a, output, times, pump, step, False, restart, not creator)
        (work / f"input_twotime_{run_name}_evolve_forward.ain").write_text(forward_text, encoding="utf-8")
        (work / f"input_twotime_{run_name}_evolve_backward.ain").write_text(backward_text, encoding="utf-8")
    return folder


# ---------------------------------------------------------------------------
# Batch scripts (templates are kept whole; comments stay in Python, not in the Slurm text)
# ---------------------------------------------------------------------------

def write_td_batch(a: argparse.Namespace, td_folder: Path, input_path: Path) -> Path:
    """Write the one-node Slurm job that runs the TD input."""
    path = td_folder / f"batch_td_{td_folder.name}.slurm"
    body = f'''{sbatch_header("dmrg_td", 1, a.cpus_per_task, getattr(a, "walltime", "16:00:00"))}
{batch_environment(a.cpus_per_task)}
cd {shlex.quote(str(td_folder))}
srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task={a.cpus_per_task} \\
    ./dmrg -f {shlex.quote(input_path.name)} -p {DMRG_PRECISION} {shlex.quote(TD_OPERATORS)} \\
    > td_streamed.stdout 2> td_streamed.stderr
'''
    path.write_text(body, encoding="utf-8")
    return path


def write_batch(a: argparse.Namespace, td_folder: Path, gless: Path, ggreater: Path, run_name: str) -> Path:
    """Write the combined G job. It must be submitted with an afterok dependency on TD."""
    path = td_folder / f"batch_combined_G_{run_name}.slurm"
    nodes = a.g_worker_nodes
    post_commands = postprocess_commands(a, td_folder, gless, ggreater, run_name)
    td_post, less_post, greater_post = (shlex.join(command) for command in post_commands)
    body = f'''{sbatch_header("dmrg_combined_g", nodes, a.cpus_per_task, getattr(a, "walltime", "16:00:00"), ntasks=nodes)}

{batch_environment(a.cpus_per_task)}

TD_FOLDER="{td_folder}"
GLESS_FOLDER="{gless}"
GGREATER_FOLDER="{ggreater}"
RUN_NAME="{run_name}"
TIME_STEPS={a.Pump_time_steps}
CPUS_PER_TASK={a.cpus_per_task}
G_WORKER_NODES={a.g_worker_nodes}
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

active=0
G_PIDS=()
launch_g_step() {{
  local folder="$1"
  local step="$2"
  local twotime_op="$3"
  local step_folder="$folder/step_$(printf '%04d' "$step")"
  printf '%s\\n' running > "$step_folder/.g_pipeline_status"
  (
    cd "$step_folder" || exit 1
    if srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" \\
          bash -c '
            set -e
            run_name="$1"; precision="$2"; twotime_op="$3"
            ./dmrg -f "input_twotime_${{run_name}}_evolve_forward.ain" -p "$precision" "$twotime_op"
            ./dmrg -f "input_twotime_${{run_name}}_evolve_backward.ain" -p "$precision" "$twotime_op"
          ' _ "$RUN_NAME" "$PRECISION" "$twotime_op"; then
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
  # Return as soon as ANY tracked G pipeline has exited (polls; works on any bash).
  # Bash keeps the exit status of finished background jobs until waited on.
  local pid n i
  while true; do
    n=${{#G_PIDS[@]}}
    for (( i=0; i<n; i++ )); do
      pid=${{G_PIDS[$i]}}
      if ! kill -0 "$pid" 2>/dev/null; then
        if ! wait "$pid"; then
          echo "G pipeline failed (PID $pid); continuing other steps" >&2
        fi
        G_PIDS=("${{G_PIDS[@]:0:i}}" "${{G_PIDS[@]:i+1}}")
        active=$((active - 1))
        return 0
      fi
    done
    sleep 2
  done
}}
wait_for_pair_slots() {{
  while (( active + 2 > G_WORKER_NODES )); do reap_one; done
}}
# This batch must be submitted with an afterok dependency on TD.
# Check every required TD recovery file before launching any pipeline, so a missing
# file stops the batch with nothing running.
for (( step=1; step<TIME_STEPS; ++step )); do
  if [[ ! -f "$TD_FOLDER/Recovery$((step - 1))$RUN_NAME.hd5" ]]; then
    echo "Missing required TD recovery for step $step: $TD_FOLDER/Recovery$((step - 1))$RUN_NAME.hd5" >&2
    exit 1
  fi
done
for (( step=0; step<TIME_STEPS; ++step )); do
  wait_for_pair_slots
  launch_g_step "$GLESS_FOLDER" "$step" "<P2|c'|P3>"
  launch_g_step "$GGREATER_FOLDER" "$step" "<P2|c|P3>"
done

while (( active > 0 )); do reap_one; done
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


# ---------------------------------------------------------------------------
# Post-processing commands
# ---------------------------------------------------------------------------

def td_output_path(td_folder: Path) -> Path:
    """Name of the TD DMRG++ output, taken from the TD input file (runFor<input stem>.cout)."""
    inputs = sorted(td_folder.glob("input_td_*.ain"))
    if len(inputs) != 1:
        raise FileNotFoundError(f"Expected one TD input in {td_folder}, found {len(inputs)}")
    return td_folder / f"runFor{inputs[0].stem}.cout"


def postprocess_commands(a: argparse.Namespace, td_folder: Path, gless: Path, ggreater: Path,
                         run_name: str) -> list[list[str]]:
    """Commands that turn the TD and G outputs into CSV files: TD first, then G< and G>."""
    scripts = Path(__file__).resolve().parent
    pump = str(td_folder / "pump_table.csv")
    commands = [[
        "python3", str(scripts / "my_hubbard_chain_td_targetingexpression_process.py"),
        str(td_output_path(td_folder)), pump,
    ]]
    for prefix, folder in (("G<", gless), ("G>", ggreater)):
        commands.append([
            "python3", str(scripts / f"my_hubbard_chain_twotime_{prefix}_parallel_process.py"),
            str(a.center_site), str(a.gs_filename.resolve()), pump, str(folder), "--workers", "1",
        ])
    return commands


# ---------------------------------------------------------------------------
# Local execution (--run)
# ---------------------------------------------------------------------------

def run_serial(a: argparse.Namespace, td_folder: Path, td_input_path: Path, gless: Path,
               ggreater: Path, run_name: str) -> None:
    """Run TD, then the G pipelines concurrently, all on the current machine or allocation."""
    environment = os.environ.copy()
    # One OpenMP thread per CPU (hardware thread), as in the batch jobs.
    environment["OMP_NUM_THREADS"] = str(a.cpus_per_task)
    environment.setdefault("OMP_PLACES", "threads")
    environment.setdefault("OMP_PROC_BIND", "spread")

    def run_td() -> None:
        print(f"Running {td_folder / td_input_path.name}", flush=True)
        subprocess.run(
            ["./dmrg", "-f", td_input_path.name, "-p", str(DMRG_PRECISION), TD_OPERATORS],
            cwd=td_folder, env=environment, check=True,
        )

    # Each G step (one < and one >) has a status file, read by the failure report.
    all_steps = [folder / f"step_{step:04d}" for folder in (gless, ggreater)
                 for step in range(a.Pump_time_steps)]
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

    # Check the CPU allocation before the long TD run starts.
    total_cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    if a.cpus_per_task > total_cpus:
        raise ValueError(f"cpus-per-task ({a.cpus_per_task}) exceeds available CPUs ({total_cpus})")
    atexit.register(write_failure_report)
    run_td()
    parallel = min(len(all_steps), total_cpus // a.cpus_per_task)
    print(f"Running up to {parallel} G pipelines concurrently, {a.cpus_per_task} CPUs per pipeline.", flush=True)

    def pipeline(work: Path, operator: str) -> bool:
        """Run one G step (forward, then backward). Returns True on success."""
        (work / ".g_pipeline_status").write_text("running\n", encoding="utf-8")
        try:
            # NERSC: one srun owns the allocation for both sequential stages.
            if a.cluster == "nersc":
                command = [
                    "srun", "--exclusive", "--exact", "--nodes=1", "--ntasks=1",
                    f"--cpus-per-task={a.cpus_per_task}", "--cpu-bind=cores", "bash", "-c",
                    'set -e; ./dmrg -f "$1" -p "$3" "$4"; ./dmrg -f "$2" -p "$3" "$4"',
                    "_", f"input_twotime_{run_name}_evolve_forward.ain",
                    f"input_twotime_{run_name}_evolve_backward.ain", str(DMRG_PRECISION), operator,
                ]
                commands = [command]
            else:
                commands = [
                    ["./dmrg", "-f", f"input_twotime_{run_name}_{stage}.ain", "-p", str(DMRG_PRECISION), operator]
                    for stage in ("evolve_forward", "evolve_backward")
                ]
            with (work / "streamed.stdout").open("w", encoding="utf-8") as out, \
                    (work / "streamed.stderr").open("w", encoding="utf-8") as err:
                for command in commands:
                    subprocess.run(command, cwd=work, env=environment, stdout=out, stderr=err, check=True)

            (work / ".g_pipeline_status").write_text("complete\n", encoding="utf-8")
            return True
        except (subprocess.CalledProcessError, OSError) as exc:
            (work / ".g_pipeline_status").write_text("failed\n", encoding="utf-8")
            print(f"G pipeline failed for {work}: {exc}; continuing other steps.", flush=True)
            return False

    failed_steps = []
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {
            executor.submit(pipeline, folder / f"step_{step:04d}", operator): folder / f"step_{step:04d}"
            for step in range(a.Pump_time_steps)
            for folder, operator in ((gless, "<P2|c'|P3>"), (ggreater, "<P2|c|P3>"))
        }
        for future in as_completed(futures):
            if not future.result():
                failed_steps.append(futures[future])

    failure_log = td_folder / "failed_g_steps.txt"
    write_failure_report()
    commands = postprocess_commands(a, td_folder, gless, ggreater, run_name)
    subprocess.run(commands[0], check=True)
    for folder, command in zip((gless, ggreater), commands[1:]):
        if any(work.parent == folder for work in failed_steps):
            print(f"Skipping incomplete post-processing: {folder}", flush=True)
            continue
        subprocess.run(command, check=True)
    if failed_steps:
        raise RuntimeError(f"Failed G steps to rerun are listed in {failure_log}")
    print("TD and all G< / G> step pipelines completed and post-processed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    """Create the run folders and inputs, then submit, run, or stop, as requested."""
    a = parse_args()
    validate(a)
    gs = a.gs_filename.resolve()
    executable = DMRG_EXECUTABLES[a.cluster]
    if not gs.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {gs}")
    if not executable.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable}")

    # Microseconds keep run names unique when several runs start in the same second.
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_base = f"N={a.sites}_Nup={a.up}_Ndown={a.down}_{stamp}"
    tag = frequency_tag(a.pump_frequency)
    run_name = f"td_{run_base}_{tag}"
    parent = gs.parent
    td_folder = parent / run_name
    td_folder.mkdir(parents=True, exist_ok=False)
    times = time_axis(a)
    pump = pump_axis(a, times)

    # Refuse to write into G folders from the old separate-apply layout.
    for prefix in ("G<", "G>"):
        folder = parent / f"{prefix}_twotime_{run_name}"
        if any(folder.glob("step_*/input_*_apply.ain")):
            raise ValueError(f"Existing G folder uses a separate apply stage; move it aside before using condensed inputs: {folder}")

    values = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    (td_folder / f"input_args_td_{run_base}.json").write_text(json.dumps(values, indent=2), encoding="utf-8")
    shutil.copy2(executable, td_folder / "dmrg")
    write_pump(times, pump, td_folder / "pump_table.csv")
    td_output = f"td_{run_base}_{tag}"
    td_input_path = td_folder / f"input_td_{run_base}.ain"
    td_input_path.write_text(td_input(a, td_output, times, pump, f"../{gs.name}"), encoding="utf-8")

    gless = write_g_inputs(a, parent, td_folder, run_name, times, pump, True)
    ggreater = write_g_inputs(a, parent, td_folder, run_name, times, pump, False)
    batch = None
    td_batch = None
    if a.cluster == "nersc":
        batch = write_batch(a, td_folder, gless, ggreater, run_name)
        td_batch = write_td_batch(a, td_folder, td_input_path)

    print(f"Created TD folder: {td_folder}")
    print(f"Created G< folder: {gless}")
    print(f"Created G> folder: {ggreater}")
    if batch is not None:
        print(f"Wrote TD batch script: {td_batch}")
        print(f"Wrote combined G batch script: {batch}")

    if a.submit:
        # The G job starts only after TD finishes successfully (afterok).
        result = subprocess.run(["sbatch", str(td_batch)], cwd=td_folder, check=True,
                                stdout=subprocess.PIPE, text=True)
        print(result.stdout.strip(), flush=True)
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if not match:
            raise RuntimeError("TD submission succeeded but its job ID could not be parsed; check Slurm before resubmitting")
        td_job_id = match.group(1)
        print(f"Submitting G job dependent on TD job {td_job_id}", flush=True)
        subprocess.run(["sbatch", f"--dependency=afterok:{td_job_id}", str(batch)], cwd=td_folder, check=True)
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
