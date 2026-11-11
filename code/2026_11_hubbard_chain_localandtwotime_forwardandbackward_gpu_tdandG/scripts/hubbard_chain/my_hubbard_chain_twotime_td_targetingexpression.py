"""
Generate and optionally run a two-time Hubbard-chain calculation.
"""

import argparse
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

# Location of the current DMRG++ executables for different clusters. Adjust these paths as needed. The location is printed when the script is completed.
DMRG_EXECUTABLES = {
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_backwardevolve/copy_dmrg/build/dmrg/dmrg"
    ),
    "nersc": Path(
        "/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/10_4sitetestruns_GPU/copy_dmrg/build_perlmutter_nvidia_cuda/dmrg/dmrg"
    ),
    # "nersc": Path(
    #     "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    # ),
}


# DMRG Settings
DMRG_PRECISION = 12

TD_DEFINE_OPERATORS_DMRGPP = "double:nup*ndown,hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,parity:identity+(-2.0)*n+4.0*nup*ndown,local_moment:0.75*n+(-1.5)*nup*ndown"

TD_OPERATOR_LABELS = {
    "<P0|n|P0>",
    "<gs|n|gs>",
    "<P0|sz|P0>",
    "<gs|sz|gs>",
    "<P0|local_moment|P0>",
    "<gs|local_moment|gs>",
    "<P0|hole|P0>",
    "<gs|hole|gs>",
    "<P0|parity|P0>",
    "<gs|parity|gs>",
    "<P0|double|P0>",
    "<gs|double|gs>",
    "<P0|n*n|P0>",
    "<gs|n*n|gs>",
}

TD_COLLECTION_MARKER = "FiniteLoops printing ends"

TD_NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

TD_OPERATOR_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<site>\d+)\s+
    \(
        (?P<real>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?),
        (?P<imag>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)
    \)\s+
    (?P<time>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s+
    (?P<label><[^>]+>)
    """,
    re.VERBOSE,
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
    parser.add_argument(
        "TSPAdvanceEach",
        type=int,
        help="Multiples of N-2",
    )
    parser.add_argument("TSPTau", type=float, help="Time step size for the pump table")
    parser.add_argument("Pump_Amplitude", type=float)
    parser.add_argument("Pump_time_delay", type=float)
    parser.add_argument("Pump_pulse_width", type=float)
    parser.add_argument("Pump_time_steps", type=int, help="Always do N+2 for N steps")
    parser.add_argument("cluster", choices=DMRG_EXECUTABLES)
    parser.add_argument(
        "--pump-frequencies",
        type=float,
        nargs="+",
        required=True,
        help="One or more pump frequencies; one TD run is generated for each.",
    )
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--run", action="store_true", help="Run DMRG++ after generating input"
    )
    execution_group.add_argument(
        "--submit", action="store_true", help="Submit DMRG++ job after generating input"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=128,
        help="CPU cores assigned for this run.",
    )
    parser.add_argument(
        "--generate-twotime",
        action="store_true",
        help="Generate G< and G> two-time calculations from this TD run.",
    )
    parser.add_argument(
        "--center-site",
        type=int,
        help="Center site for G< and G> operator insertion.",
    )
    parser.add_argument(
        "--finite-loops-apply",
        type=int,
        help="Finite loops for the G< and G> operator-application stage.",
    )
    parser.add_argument("--run-timestamp", help=argparse.SUPPRESS)

    return parser.parse_args()


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
    if args.Pump_time_steps <= 0:
        raise ValueError("Pump_time_steps must be positive")
    if args.cpus_per_task <= 0:
        raise ValueError("cpus_per_task must be positive")
    if args.submit and args.cluster != "nersc":
        raise ValueError("--submit is only supported for the nersc cluster")
    if args.generate_twotime:
        if args.center_site is None or args.finite_loops_apply is None:
            raise ValueError(
                "--generate-twotime requires --center-site and --finite-loops-apply"
            )
        if not 0 <= args.center_site < args.sites:
            raise ValueError("--center-site must be a valid site index")
        if args.finite_loops_apply <= 0:
            raise ValueError("--finite-loops-apply must be positive")


def save_arguments(args: argparse.Namespace, output_path: Path) -> None:
    """Save command-line arguments as JSON."""
    values = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(values, file, indent=2)


def build_input_td(
    args: argparse.Namespace,
    run_name: str,
    time_axis: list[float],
    pump_axis: list[float],
    restart_filename: str | None = None,
) -> str:
    """Build the time-dependent reference-state input."""
    if restart_filename is None:
        restart_filename = f"../{args.gs_filename.resolve().name}"

    # Number of finite loops is determined by the number of pump time steps and the TSPAdvanceEach parameter.
    finite_loops = args.Pump_time_steps * (args.TSPAdvanceEach // (args.sites - 2))

    # Finite rows is the ordering of loops in DMRG++ input.
    # Numeric value 3 = 1 + 2; 1: save and 2: usefastwft
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, 2]" for _ in range(finite_loops)
    )

    # Build the AversusTime table for the pump values.
    aversus_t_table = "\n".join(
        f"    [{time:.15g}, {pump:.15g}]," for time, pump in zip(time_axis, pump_axis)
    ).rstrip(",")

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
                        f'DefineOperators="{TD_DEFINE_OPERATORS_DMRGPP}";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        "TruncationTolerance = 1e-10;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                        "RestartMapStages=0;",
                        "RestartMappingTvs=[-1];",
                        f'string P0="TimeEvolve{{tau={args.TSPTau},steps=5,advanceEach={args.TSPAdvanceEach}}}*|gs>";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,restart,TargetingExpression,recoveryEnableRead,BatchedGemm";',
                        'Version = "stc_vs_ttc";',
                        f'string RecoverySave = "%l%%1,@keep,@M={args.Pump_time_steps}";',
                        f'OutputFile = "{run_name}";',
                        f'RestartFilename = "{restart_filename}";',
                        "GsWeight = 0.0;",
                    ]
                ),
            ]
        )
        + "\n"
    )


def create_time_axis(nsteps: int, step_size: float) -> list[float]:
    """Create exactly nsteps physical points with fixed 0.1 padding."""
    time_max = (nsteps - 1) * step_size
    return [
        -0.1,
        *(round(index * step_size, 9) for index in range(nsteps)),
        round(time_max + 0.1, 9),
    ]


def create_pump_axis(
    time_axis: list[float],
    amplitude: float,
    omega_pump: float,
    t_delay: float,
    sigma: float,
) -> list[float]:
    """Create Gaussian-envelope cosine pump values."""
    return [
        (rounded_value if rounded_value != 0 else 0.0)
        for rounded_value in (
            round(
                amplitude
                * math.exp(-0.5 * ((time - t_delay) / sigma) ** 2)
                * math.cos(omega_pump * (time - t_delay)),
                7,
            )
            for time in time_axis
        )
    ]


def validate_pump_at_zero(time_axis: list[float], pump_axis: list[float]) -> None:
    """Require the pump signal to vanish at time zero."""
    try:
        pump_at_zero = next(
            pump for time, pump in zip(time_axis, pump_axis) if time == 0.0
        )
    except StopIteration as error:
        raise ValueError("Pump time axis does not contain 0.0") from error

    if pump_at_zero != 0.0:
        raise ValueError(f"Pump signal at time 0.0 must be 0.0, got {pump_at_zero}")


def write_pump_table(
    pump_table: Iterable[tuple[float, float]],
    output_path: Path,
) -> None:
    """Write time and pump values to CSV."""
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["time", "pump"])
        writer.writerows((f"{time:.7f}", f"{pump:.7f}") for time, pump in pump_table)


def run_dmrg(
    working_directory: Path,
    input_path: Path,
    operator: str | None = None,
    cpus_per_task: int = 1,
) -> None:
    command = [
        "./dmrg",
        "-f",
        input_path.name,
        "-p",
        str(DMRG_PRECISION),
    ]

    if operator is not None:
        command.append(operator)

    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(cpus_per_task)

    subprocess.run(
        command,
        cwd=working_directory,
        check=True,
        env=environment,
    )


def twotime_command(
    args: argparse.Namespace, td_folder: Path, script_name: str, execution: str | None
) -> list[str]:
    """Build a command for one of the G< or G> two-time scripts."""
    command = [
        "python3",
        str(Path(__file__).with_name(script_name)),
        str(args.sites),
        str(args.up),
        str(args.down),
        str(args.t),
        str(args.U),
        str(args.potentialV),
        str(args.gs_filename.resolve()),
        str(args.finite_kept),
        args.cluster,
        str(args.center_site),
        str(args.finite_loops_apply),
        str(td_folder),
        "--cpus-per-task",
        str(args.cpus_per_task),
    ]
    if execution is not None:
        command.append(execution)
    return command


def generate_twotime_batches(args: argparse.Namespace, td_folder: Path) -> list[Path]:
    """Generate G< and G> batch scripts and return their paths."""
    script_names = (
        "my_hubbard_chain_twotime_G<_parallel_sequential.py",
        "my_hubbard_chain_twotime_G>_parallel_sequential.py",
    )
    batch_paths = []
    for script_name in script_names:
        subprocess.run(
            twotime_command(args, td_folder, script_name, execution=None), check=True
        )
        prefix = "G<" if "G<" in script_name else "G>"
        run_name = td_folder.name
        batch_paths.append(
            args.gs_filename.resolve().parent
            / f"{prefix}_twotime_{run_name}"
            / f"batch_multinode_{run_name}.slurm"
        )
    return batch_paths


def frequency_tag(frequency: float) -> str:
    """Convert a frequency into a filesystem-safe tag."""
    return f"freq_{frequency:g}".replace(".", "p")


def td_run_folder(args: argparse.Namespace, frequency: float, timestamp: str) -> Path:
    """Return the deterministic TD output folder for one pump frequency."""
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    return args.gs_filename.resolve().parent / (
        f"td_{run_name}_{frequency_tag(frequency)}"
    )


def frequency_command(
    args: argparse.Namespace, frequency: float, run: bool, timestamp: str
) -> list[str]:
    """Build a command that handles one frequency from a multi-frequency request."""
    command = [
        "python3",
        str(Path(__file__).resolve()),
        str(args.sites),
        str(args.up),
        str(args.down),
        str(args.t),
        str(args.U),
        str(args.potentialV),
        str(args.gs_filename.resolve()),
        str(args.finite_kept),
        str(args.TSPAdvanceEach),
        str(args.TSPTau),
        str(args.Pump_Amplitude),
        str(args.Pump_time_delay),
        str(args.Pump_pulse_width),
        str(args.Pump_time_steps),
        args.cluster,
        "--pump-frequencies",
        str(frequency),
        "--cpus-per-task",
        str(args.cpus_per_task),
        "--run-timestamp",
        timestamp,
    ]
    if run:
        command.append("--run")
    return command


def write_multi_frequency_batch(args: argparse.Namespace, timestamp: str) -> Path:
    """Write one NERSC batch script that runs one frequency per node."""
    restart_path = args.gs_filename.resolve()
    executable_path = DMRG_EXECUTABLES[args.cluster]
    if not restart_path.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {restart_path}")
    if not executable_path.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable_path}")

    batch_path = restart_path.parent / f"batch_td_frequencies_{timestamp}.slurm"
    commands = [
        "srun --exclusive --nodes=1 --ntasks=1 "
        f"--cpus-per-task={args.cpus_per_task} "
        f"{shlex.join(frequency_command(args, frequency, run=True, timestamp=timestamp))} &"
        for frequency in args.pump_frequencies
    ]

    frequency_count = len(args.pump_frequencies)
    body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes={frequency_count}
#SBATCH --ntasks={frequency_count}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --gpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --job-name=dmrg_td_frequencies
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load cray-mpich/9.0.1
module load cray-libsci/25.09.0
module load cray-hdf5/1.14.3.7
module load PrgEnv-nvidia
module load cudatoolkit

export OMP_NUM_THREADS={max(1, args.cpus_per_task // 2)}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

{chr(10).join(commands)}

wait
"""
    batch_path.write_text(body, encoding="utf-8")
    return batch_path


def main() -> int:
    """Generate inputs, optionally run DMRG++, and collect the results."""
    args = parse_args()
    validate_args(args)

    if len(args.pump_frequencies) > 1:
        if args.run:
            raise ValueError("Use --submit to run multiple pump frequencies")

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        td_folders = []
        for frequency in args.pump_frequencies:
            subprocess.run(
                frequency_command(args, frequency, run=False, timestamp=timestamp),
                check=True,
            )
            td_folders.append(td_run_folder(args, frequency, timestamp))

        if args.submit:
            batch_path = write_multi_frequency_batch(args, timestamp)
            result = subprocess.run(
                ["sbatch", "--parsable", str(batch_path)],
                check=True,
                capture_output=True,
                text=True,
            )
            td_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
            if not td_job_id:
                raise RuntimeError("sbatch did not return a TD batch job ID")
            print(f"Submitted multi-frequency TD batch job: {td_job_id}")
            if args.generate_twotime:
                for td_folder in td_folders:
                    for twotime_batch in generate_twotime_batches(args, td_folder):
                        result = subprocess.run(
                            [
                                "sbatch",
                                f"--dependency=afterok:{td_job_id}",
                                str(twotime_batch),
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        print(
                            f"Submitted dependent two-time batch: "
                            f"{result.stdout.strip()} ({twotime_batch})"
                        )
        elif args.generate_twotime:
            for td_folder in td_folders:
                generate_twotime_batches(args, td_folder)
        return 0

    args.Pump_Frequency = args.pump_frequencies[0]
    restart_path = args.gs_filename.resolve()

    time_axis = create_time_axis(
        nsteps=args.Pump_time_steps,
        step_size=args.TSPTau,
    )

    pump_axis = create_pump_axis(
        time_axis,
        amplitude=args.Pump_Amplitude,
        omega_pump=args.Pump_Frequency,
        t_delay=args.Pump_time_delay,
        sigma=args.Pump_pulse_width,
    )
    validate_pump_at_zero(time_axis, pump_axis)

    timestamp = args.run_timestamp or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"
    executable_path = DMRG_EXECUTABLES[args.cluster]

    if not restart_path.is_file():
        raise FileNotFoundError(f"Ground-state file not found: {restart_path}")
    if not executable_path.is_file():
        raise FileNotFoundError(f"DMRG++ executable not found: {executable_path}")

    # Name file with pump frequency for clarity
    frequency_tag_value = frequency_tag(args.Pump_Frequency)
    run_folder_td = restart_path.parent / f"td_{run_name}_{frequency_tag_value}"
    run_folder_td.mkdir(parents=True, exist_ok=True)
    print(f"Created run folder: {run_folder_td}")

    # Save the command-line arguments to a JSON file.
    args_path = run_folder_td / f"input_args_td_{run_name}.json"
    save_arguments(args, args_path)
    print(f"Wrote arguments: {args_path}")

    # Copy the DMRG++ executable to the run folder.
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder_td / "dmrg")

    # Write the pump table to a CSV file for reference.
    write_pump_table(
        pump_table=zip(time_axis, pump_axis),
        output_path=run_folder_td / "pump_table.csv",
    )

    input_path = run_folder_td / f"input_td_{run_name}.ain"
    input_path.write_text(
        build_input_td(
            args,
            "td_" + run_name + "_" + frequency_tag_value,
            time_axis,
            pump_axis,
            restart_filename=f"../{restart_path.name}",
        ),
        encoding="utf-8",
    )
    print(f"Wrote Ainur input: {input_path}")

    operator_argument = ",".join(sorted(TD_OPERATOR_LABELS))

    if args.cluster == "nersc":
        slurm_path = run_folder_td / f"batch_{run_name}.slurm"

        physical_cores_per_task = max(
            1, args.cpus_per_task // 2
        )  # Assuming hyperthreading, adjust as needed

        body = f"""#!/bin/bash
#SBATCH --account=m5228
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --time=48:00:00
#SBATCH --gpus-per-task=4
#SBATCH --job-name=dmrg_frequency
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

module load cray-mpich/9.0.1
module load cray-libsci/25.09.0
module load cray-hdf5/1.14.3.7
module load PrgEnv-nvidia
module load cudatoolkit

export OMP_NUM_THREADS={max(1, args.cpus_per_task // 2)}
export OMP_PLACES=cores
export OMP_PROC_BIND=spread

cd "{run_folder_td}"

srun \\
    --exclusive \\
    --cpu-bind=cores \\
    --ntasks=1 \\
    --cpus-per-task={args.cpus_per_task} \\
    ./dmrg \\
    -f {input_path.name} \\
    -p {DMRG_PRECISION} \\
    "{operator_argument}"



wait

echo "All frequency jobs completed."
"""

        slurm_path.write_text(body, encoding="utf-8")
        print(f"Wrote batch script: {slurm_path}")

    # If the --run flag is provided, execute DMRG++ with the generated input and collect the results.

    if args.run:
        # Run DMRG++ with the generated input and collect the results.
        run_dmrg(
            run_folder_td,
            input_path,
            cpus_per_task=args.cpus_per_task,
            operator=operator_argument,
        )
        if args.generate_twotime:
            for script_name in (
                "my_hubbard_chain_twotime_G<_parallel_sequential.py",
                "my_hubbard_chain_twotime_G>_parallel_sequential.py",
            ):
                subprocess.run(
                    twotime_command(args, run_folder_td, script_name, execution="--run"),
                    check=True,
                )

    elif args.submit:
        result = subprocess.run(
            ["sbatch", "--parsable", str(slurm_path)],
            cwd=run_folder_td,
            check=True,
            capture_output=True,
            text=True,
        )
        td_job_id = result.stdout.strip().split(";", maxsplit=1)[0]
        if not td_job_id:
            raise RuntimeError("sbatch did not return a TD batch job ID")
        print(f"Submitted TD batch job: {td_job_id}")
        if args.generate_twotime:
            for batch_path in generate_twotime_batches(args, run_folder_td):
                result = subprocess.run(
                    ["sbatch", f"--dependency=afterok:{td_job_id}", str(batch_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                print(
                    f"Submitted dependent two-time batch: "
                    f"{result.stdout.strip()} ({batch_path})"
                )
    else:
        if args.generate_twotime:
            generate_twotime_batches(args, run_folder_td)
        print("Inputs generated. Use --run to execute DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
