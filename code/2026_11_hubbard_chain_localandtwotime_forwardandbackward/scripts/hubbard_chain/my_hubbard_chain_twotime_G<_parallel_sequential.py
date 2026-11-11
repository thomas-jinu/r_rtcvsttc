"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Location of the current DMRG++ executables for different clusters. Adjust these paths as needed. The location is printed when the script is completed.
DMRG_EXECUTABLES = {
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_backwardevolve/copy_dmrg/build/dmrg/dmrg"
    ),
    "nersc": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_backwardevolve/copy_dmrg/build/dmrg/dmrg"
    ),
    # "nersc": Path(
    #     "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    # ),
}


# DMRG Settings
DMRG_PRECISION = 12

TWOTIME_OPERATOR_LABEL = "<P2|c'|P3>"
APPLY_OPERATOR_LABEL = "<P0|c'|P1>"

TWOTIME_COLLECTION_MARKER = "FiniteLoops printing ends"

TWOTIME_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

TWOTIME_OPERATOR_PATTERN = re.compile(
    rf"^\s*(\d+)\s+"
    rf"\(\s*({TWOTIME_NUMBER_PATTERN})\s*,\s*({TWOTIME_NUMBER_PATTERN})\s*\)\s+"
    rf"({TWOTIME_NUMBER_PATTERN})\s+"
    rf"{re.escape(TWOTIME_OPERATOR_LABEL)}"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int, help="Number of spin-up electrons")
    parser.add_argument("down", type=int, help="Number of spin-down electrons")
    parser.add_argument("t", type=float, help="Hopping parameter t")
    parser.add_argument("U", type=float, help="Hubbard interaction U")
    parser.add_argument("potentialV", type=float, help="Potential V")
    parser.add_argument("gs_filename", type=Path, help="Ground-state file location")
    parser.add_argument(
        "finite_kept", type=int, help="Number of states in finite loops"
    )
    parser.add_argument("cluster", choices=DMRG_EXECUTABLES)
    parser.add_argument("center_site", type=int, help="Index of the center site")
    parser.add_argument(
        "finite_loops_apply",
        type=int,
        help="Number of finite loops for applying the operator step",
    )
    parser.add_argument("td_folder", type=Path, help="Folder name for td files")
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating inputs"
    )
    execution_group.add_argument(
        "--submit", action="store_true", help="Submit DMRG++ job after generating input"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=1,
        help="CPU cores assigned to each DMRG++ process.",
    )
    parser.add_argument(
        "--n-tasks",
        type=int,
        default=2,
        help=(
            "Number of independent time steps to run simultaneously. "
            "Must be at least 2 so the first and last steps can run together."
        ),
    )

    return parser.parse_args()


def load_td_run_data(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    """Load time/pump parameters and the exact pump table from the TD run."""
    args.td_folder = args.td_folder.resolve()
    if not args.td_folder.is_dir():
        raise FileNotFoundError(f"TD run folder not found: {args.td_folder}")

    argument_files = sorted(args.td_folder.glob("input_args_td_*.json"))
    if len(argument_files) != 1:
        raise FileNotFoundError(
            "Expected exactly one input_args_td_*.json file in "
            f"{args.td_folder}, found {len(argument_files)}"
        )

    with argument_files[0].open("r", encoding="utf-8") as file:
        td_args = json.load(file)

    required_parameters = (
        "TSPAdvanceEach",
        "TSPTau",
        "Pump_Amplitude",
        "Pump_Frequency",
        "Pump_time_delay",
        "Pump_pulse_width",
        "Pump_time_steps",
    )
    missing_parameters = [key for key in required_parameters if key not in td_args]
    if missing_parameters:
        raise ValueError(
            "TD argument file is missing: " + ", ".join(missing_parameters)
        )
    for key in required_parameters:
        setattr(args, key, td_args[key])

    pump_table_path = args.td_folder / "pump_table.csv"
    if not pump_table_path.is_file():
        raise FileNotFoundError(f"Pump table not found: {pump_table_path}")

    with pump_table_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"Pump table is empty: {pump_table_path}")

    try:
        time_axis = [float(row["time"]) for row in rows]
        pump_axis = [float(row["pump"]) for row in rows]
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid pump table: {pump_table_path}") from error

    return time_axis, pump_axis


def validate_args(args: argparse.Namespace) -> None:
    """Validate constraints required by the finite-loop construction."""
    if args.sites <= 2:
        raise ValueError("sites must be greater than 2")
    if args.TSPAdvanceEach <= 0:
        raise ValueError("TSPAdvanceEach must be positive")
    if args.TSPAdvanceEach != (args.sites - 2):
        raise ValueError("This script only supports TSPAdvanceEach equal to sites - 2")
    if args.Pump_pulse_width == 0:
        raise ValueError("Pump_pulse_width must be nonzero")
    if args.n_tasks < 2:
        raise ValueError("n_tasks must be at least 2")
    if args.submit and args.cluster != "nersc":
        raise ValueError("--submit is only supported for the nersc cluster")


def save_arguments(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments as JSON."""
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)


def build_input_apply(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    step: int,
    restart_filename_apply: str,
) -> str:
    """Build the input that applies the operator at one time step."""

    # Finite rows for the apply step.
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(args.finite_loops_apply)
    )

    # Generate the pump table at the specific time step.
    aversus_t_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

    return (
        "\n\n".join(
            [
                "##Ainur1.0",
                "\n".join(
                    [
                        "# --- Model parameters ---",
                        f"TotalNumberOfSites = {args.sites};",
                        "NumberOfTerms = 1;",
                        "DegreesOfFreedom = 1;",
                        'GeometryKind = "chain";',
                        'GeometryOptions = "ConstantValues";',
                        f"dir0:Connectors = [{args.t}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Pump parameters --- #",
                        "matrix AversusTime = [",
                        f"{aversus_t_table}",
                        "];",
                        f"GeometryFactor = exp:*:1.0i:!readTableAversusTime,{time_axis[step]:.15g};",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {args.up};",
                        f"TargetElectronsDown = {args.down};",
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                        "RestartMapStages=0;",
                        "RestartMappingTvs=[0,-1,-1,-1];",
                        "string P0 = |P0>;",
                        f'string P1 = "c[{args.center_site}]*|P0>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression";',
                        'Version = "stc_vs_ttc_apply";',
                        f'OutputFile = "{run_name}_apply";',
                        f'RestartFilename = "{restart_filename_apply}";',
                        "GsWeight = 0.0;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def build_input_evolve_forward(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    step: int,
    restart_filename_evolve: str,
) -> str:
    """Build the time-evolution input after applying the operator."""
    remaining_steps = len(time_axis) - step

    if remaining_steps == 1:
        finite_loops = args.TSPAdvanceEach // (args.sites - 2)

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        aversus_t_table = f"    [{time_axis[step]:.15g}, {pump_axis[step]:.15g}]"

        targeting_expression = [
            "RestartMappingTvs=[0, 1, -1, -1];",
            'string P0="|P0>";',
            'string P1="|P1>";',
            'string P2="|P0>";',
            'string P3="|P1>";',
        ]

    else:
        finite_loops = remaining_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

        finite_rows = ",\n".join(
            f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
        )

        aversus_t_table = "\n".join(
            f"    [{time:.15g}, {pump:.15g}],"
            for time, pump in zip(time_axis[step:], pump_axis[step:])
        ).rstrip(",")

        targeting_expression = [
            "RestartMappingTvs=[0, 1, -1, -1];",
            'string P0="|P0>";',
            'string P1="|P1>";',
            f'string P2="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P0>";',
            f'string P3="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P1>";',
        ]

    return (
        "\n\n".join(
            [
                "##Ainur1.0",
                "\n".join(
                    [
                        "# --- Model parameters ---",
                        f"TotalNumberOfSites = {args.sites};",
                        "NumberOfTerms = 1;",
                        "DegreesOfFreedom = 1;",
                        'GeometryKind = "chain";',
                        'GeometryOptions = "ConstantValues";',
                        f"dir0:Connectors = [{args.t}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Pump parameters --- #",
                        "matrix AversusTime = [",
                        f"{aversus_t_table}",
                        "];",
                        "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {args.up};",
                        f"TargetElectronsDown = {args.down};",
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                        "RestartMapStages=0;",
                        *targeting_expression,
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead";',
                        'Version = "stc_vs_ttc_apply_evolve";',
                        'string RecoverySave = "%l%%1,@keep,@M=1";',
                        f'OutputFile = "{run_name}_evolve_forward";',
                        f'RestartFilename = "{restart_filename_evolve}";',
                        "GsWeight = 0.0;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def build_input_evolve_backward(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    step: int,
    restart_filename_evolve: str,
) -> str:
    """Build the time-evolution input after applying the operator."""
    remaining_steps = step + 1

    finite_loops = remaining_steps * (args.TSPAdvanceEach // (args.sites - 2)) - 1

    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
    )

    aversus_t_table = "\n".join(
        f"    [{time:.15g}, {pump:.15g}],"
        for time, pump in reversed(
            list(zip(time_axis[: step + 1], pump_axis[: step + 1]))
        )
    ).rstrip(",")

    targeting_expression = [
        "RestartMappingTvs=[0, 1, -1, -1];",
        'string P0="|P0>";',
        'string P1="|P1>";',
        f'string P2="TimeEvolve{{tau={-args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P0>";',
        f'string P3="TimeEvolve{{tau={-args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|P1>";',
    ]

    return (
        "\n\n".join(
            [
                "##Ainur1.0",
                "\n".join(
                    [
                        "# --- Model parameters ---",
                        f"TotalNumberOfSites = {args.sites};",
                        "NumberOfTerms = 1;",
                        "DegreesOfFreedom = 1;",
                        'GeometryKind = "chain";',
                        'GeometryOptions = "ConstantValues";',
                        f"dir0:Connectors = [{args.t}];",
                        f"hubbardU = [{args.U}, ...];",
                        f"potentialV = [{args.potentialV}, ...];",
                        'Model = "HubbardOneBand";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Pump parameters --- #",
                        "matrix AversusTime = [",
                        f"{aversus_t_table}",
                        "];",
                        "GeometryFactor = exp:*:1.0i:!readTableAversusTime,%t;",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {args.up};",
                        f"TargetElectronsDown = {args.down};",
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                        "RestartMapStages=0;",
                        *targeting_expression,
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead";',
                        'Version = "stc_vs_ttc_apply_evolve";',
                        'string RecoverySave = "%l%%1,@keep,@M=1";',
                        f'OutputFile = "{run_name}_evolve_backward";',
                        f'RestartFilename = "{restart_filename_evolve}";',
                        "GsWeight = 0.0;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def run_dmrg(
    working_directory: Path,
    input_path: Path,
    operator: str | None = None,
    cpus_per_task: int = 1,
) -> None:
    dmrg_command = [
        "./dmrg",
        "-f",
        input_path.name,
        "-p",
        str(DMRG_PRECISION),
    ]

    if operator is not None:
        dmrg_command.append(operator)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpus_per_task)

    subprocess.run(
        dmrg_command,
        cwd=working_directory,
        check=True,
        env=environment,
    )


def run_one_step(
    step_folder: Path,
    input_path_apply: Path,
    input_path_evolve_forward: Path,
    input_path_evolve_backward: Path,
    cpus_per_task: int,
) -> None:
    """Run the apply and evolve calculations for one time step."""
    run_dmrg(
        step_folder,
        input_path_apply,
        operator=APPLY_OPERATOR_LABEL,
        cpus_per_task=cpus_per_task,
    )
    run_dmrg(
        step_folder,
        input_path_evolve_forward,
        operator=TWOTIME_OPERATOR_LABEL,
        cpus_per_task=cpus_per_task,
    )

    run_dmrg(
        step_folder,
        input_path_evolve_backward,
        operator=TWOTIME_OPERATOR_LABEL,
        cpus_per_task=cpus_per_task,
    )


def write_multinode_batch(
    args: argparse.Namespace,
    run_folder: Path,
    run_name: str,
    restart_path: Path,
) -> Path:
    """Write a single-frequency, multi-node G< batch workflow."""
    node_count = (args.Pump_time_steps + 1) // 2
    process_script = Path(__file__).with_name(
        "my_hubbard_chain_twotime_G<_parallel_process.py"
    )
    batch_path = run_folder / f"batch_multinode_{run_name}.slurm"
    body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=shared
#SBATCH --constraint=cpu
#SBATCH --nodes={node_count}
#SBATCH --ntasks={node_count}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --time=48:00:00
#SBATCH --job-name=dmrg_gless
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load PrgEnv-gnu/8.7.0
module load cray-mpich/9.1.0
module load cray-libsci/26.03.0
module load cray-hdf5/1.14.3.7

export OMP_NUM_THREADS={max(1, args.cpus_per_task // 2)}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

RUN_FOLDER="{run_folder}"
RUN_NAME="{run_name}"
NODE_COUNT={node_count}
TIME_STEPS={args.Pump_time_steps}
CPUS_PER_TASK={args.cpus_per_task}
active=0

run_task() {{
    local stage="$1"
    local step="$2"
    local operator="$3"
    (
        cd "$RUN_FOLDER/step_$(printf '%04d' "$step")"
        srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task="$CPUS_PER_TASK" \\
            ./dmrg -f "input_twotime_${{RUN_NAME}}_${{stage}}.ain" \\
            -p {DMRG_PRECISION} "$operator"
    ) &
}}

launch() {{
    run_task "$1" "$2" "$3"
    active=$((active + 1))
    if (( active >= NODE_COUNT )); then
        wait -n
        active=$((active - 1))
    fi
}}

wait_for_all() {{
    while (( active > 0 )); do
        wait -n
        active=$((active - 1))
    done
}}

# Phase A: all equal-time operator applications complete before evolution starts.
for ((step = 0; step < TIME_STEPS; ++step)); do
    launch "apply" "$step" "{APPLY_OPERATOR_LABEL}"
done
wait_for_all

# Phase B: longest forward and backward evolutions are launched first.
for ((rank = 0; rank < TIME_STEPS; ++rank)); do
    launch "evolve_forward" "$rank" "{TWOTIME_OPERATOR_LABEL}"
    launch "evolve_backward" "$((TIME_STEPS - 1 - rank))" "{TWOTIME_OPERATOR_LABEL}"
done
wait_for_all

python3 "{process_script}" {args.center_site} "{restart_path}" \\
    "{args.td_folder / "pump_table.csv"}" "$RUN_FOLDER" --workers=1
"""
    batch_path.write_text(body, encoding="utf-8")
    return batch_path


def main() -> int:
    """Generate inputs, optionally run DMRG++, and collect the results."""
    args = parse_args()
    time_axis, pump_axis = load_td_run_data(args)
    validate_args(args)
    restart_path = args.gs_filename.resolve()

    run_name = args.td_folder.name
    print(f"Run name: {run_name}")
    executable_path = DMRG_EXECUTABLES[args.cluster]

    if not restart_path.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {restart_path}")
    if not executable_path.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable_path}")

    # Name file with pump frequency for clarity
    run_folder_twotime = restart_path.parent / f"G<_twotime_{run_name}"
    run_folder_twotime.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_twotime}")

    # Save the command-line arguments to a JSON file.
    args_path = run_folder_twotime / f"input_args_twotime_{run_name}.json"
    save_arguments(args, args_path)
    print(f"Wrote arguments: {args_path}")

    step_jobs = []

    for step in range(args.Pump_time_steps):
        # The padded axis starts at -0.1; physical operator times begin at 0.
        time_index = step + 1
        step_folder = run_folder_twotime / f"step_{step:04d}"
        step_folder.mkdir(parents=True, exist_ok=True)
        print(f"Created step folder: {step_folder}")
        shutil.copy2(executable_path, step_folder / "dmrg")

        if time_index == 1:
            restart_filename = f"../../{args.gs_filename.name}"
        else:
            restart_filename = f"../../{run_name}/Recovery{time_index - 2}{run_name}"

        input_path_apply = step_folder / f"input_twotime_{run_name}_apply.ain"
        input_path_apply.write_text(
            build_input_apply(
                args,
                f"twotime_{run_name}",
                step=time_index,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_apply=restart_filename,
            ),
            encoding="utf-8",
        )

        print(f"Wrote Ainur input: {input_path_apply}")

        input_path_evolve_forward = (
            step_folder / f"input_twotime_{run_name}_evolve_forward.ain"
        )
        input_path_evolve_forward.write_text(
            build_input_evolve_forward(
                args,
                f"twotime_{run_name}",
                step=time_index,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_evolve=f"twotime_{run_name}_apply",
            ),
            encoding="utf-8",
        )

        print(f"Wrote Ainur input: {input_path_evolve_forward}")

        input_path_evolve_backward = (
            step_folder / f"input_twotime_{run_name}_evolve_backward.ain"
        )
        input_path_evolve_backward.write_text(
            build_input_evolve_backward(
                args,
                f"twotime_{run_name}",
                step=time_index,
                time_axis=time_axis,
                pump_axis=pump_axis,
                restart_filename_evolve=f"twotime_{run_name}_apply",
            ),
            encoding="utf-8",
        )
        print(f"Wrote Ainur input: {input_path_evolve_backward}")

        step_jobs.append(
            (
                step_folder,
                input_path_apply,
                input_path_evolve_forward,
                input_path_evolve_backward,
            )
        )

    # step_jobs remains in chronological order: 0, 1, 2, ..., final step.
    if args.cluster == "nersc":
        slurm_path = write_multinode_batch(
            args, run_folder_twotime, run_name, restart_path
        )
        print(f"Wrote multi-node batch script: {slurm_path}")

    if args.run:
        with ThreadPoolExecutor(max_workers=args.n_tasks) as executor:
            futures = [
                executor.submit(
                    run_one_step,
                    step_folder,
                    input_path_apply,
                    input_path_evolve_forward,
                    input_path_evolve_backward,
                    args.cpus_per_task,
                )
                for step_folder, input_path_apply, input_path_evolve_forward, input_path_evolve_backward in step_jobs
            ]

            for future in as_completed(futures):
                future.result()
    elif args.submit:
        subprocess.run(
            ["sbatch", str(slurm_path)],
            cwd=run_folder_twotime,
            check=True,
        )
    else:
        print("Inputs generated. Use --run to execute DMRG++.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
