"""Hubbard ground-state script (GPU workflow).

Creates the run folder with the ground state |gs> (saved at every finite loop) and the
starting vector |P0> = identity[center]*|gs>, which the two-time runs restart from.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from dmrg_config import GPU_MODULE_LOAD_BLOCK, gpu_layout, gpu_sbatch_header

# Make the project root importable, so the shared settings can be found from this folder.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Executables per cluster. Each run folder gets a copy as ./dmrg.
DMRG_EXECUTABLES = {
    "local": Path(
        "/Users/qqt/Documents/Codes/dmrgpp_alltimerecovery_fixes/dmrgpp/build/dmrg/dmrg"
    ),
    "nersc": Path("/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/14_12site_GPU_twotime/dmrgpp_fixrestart/dmrgpp/build_perlmutter_nvidia_cuda/dmrg/dmrg"),
    # "nersc": Path(
    #     "/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"
    # ),
}

# Precision passed to DMRG++ with -p.
DMRG_PRECISION = 12


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def parse_args_input() -> argparse.Namespace:
    """Read the command line. Positional order: sites up down t U potentialV infinite_kept finite_loops finite_kept cluster."""
    parser = argparse.ArgumentParser(
        description="Generate and optionally run a DMRG++ Hubbard-chain calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Order of positional arguments: sites up down t U potentialV infinite_kept finite_loops finite_kept cluster

        Example:
        python my_hubbard_chain_twotime_gs.py 32 16 16 -1 8 0 128 5 1000 local --run
        """,
    )
    parser.add_argument("sites", type=int, help="Chain length N")
    parser.add_argument("up", type=int)
    parser.add_argument("down", type=int)
    parser.add_argument("t", type=float)
    parser.add_argument("U", type=float)
    parser.add_argument("potentialV", type=float)
    parser.add_argument("infinite_kept", type=int)
    parser.add_argument("finite_loops", type=int)
    parser.add_argument("finite_kept", type=int)
    parser.add_argument("cluster", choices=["local", "nersc"])
    execution_group = parser.add_mutually_exclusive_group()
    execution_group.add_argument("--run", action="store_true", help="Run DMRG++ after generating input")
    execution_group.add_argument("--submit", action="store_true", help="Submit DMRG++ job after generating input")
    args = parser.parse_args()
    if args.submit and args.cluster != "nersc":
        parser.error("--submit requires cluster=nersc")
    return args


# ---------------------------------------------------------------------------
# DMRG++ input text
# ---------------------------------------------------------------------------

def build_input(args: argparse.Namespace, up: int, down: int, run_name: str) -> str:
    """Complete ground-state input file. Every finite loop saves a checkpoint (@save)."""
    finite_rows = ",\n".join(
        f"    [@auto, {args.finite_kept}, @save]" for _ in range(args.finite_loops)
    )

    # The centre site of the chain, used for the |P0> starting vector.
    center_site = args.sites // 2 - 1

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
                        "# --- Fock Space parameters ---",
                        f"TargetElectronsUp = {up};",
                        f"TargetElectronsDown = {down};",
                        'DefineOperators="double:nup*ndown,'  # noqa: ISC004
                        "hole:identity+(-1.0)*nup+(-1.0)*ndown+nup*ndown,"
                        "parity:identity+(-2.0)*n+4.0*nup*ndown,"
                        'local_moment:0.75*n+(-1.5)*nup*ndown";',
                    ]
                ),
                "\n".join(
                    [
                        "# --- DMRG++ control parameters ---",
                        f"InfiniteLoopKeptStates = {args.infinite_kept};",
                        "TruncationTolerance = 1e-12;",
                        "FiniteLoops = [",
                        finite_rows,
                        "];",
                    ]
                ),
                "\n".join(
                    [
                        "# --- Solver / run control ---",
                        'SolverOptions = "twositedmrg,usecomplex,TargetingExpression,BatchedGemm";',
                        f'string P0="identity[{center_site}]*|gs>";',
                        "GsWeight=1.0;",
                        'Version = "stc_vs_ttc";',
                        f'OutputFile = "{run_name}";',
                    ]
                ),
            ]
        )
        + "\n"
    )


# ---------------------------------------------------------------------------
# Run folder and batch script
# ---------------------------------------------------------------------------

def write_batch_script(run_folder: Path, input_path: Path, run_name: str) -> Path:
    """Slurm job for NERSC (one OpenMP thread per CPU, as in the other jobs): one GPU node, one srun for the ground-state run."""
    slurm_path = run_folder / f"batch_{run_name}.slurm"
    # The ground state runs on a whole GPU node: 128 CPUs, so 4 GPUs for its step.
    cpus = 128
    gpus, _ = gpu_layout(cpus)
    body = f"""{gpu_sbatch_header('dmrg_job', 1, cpus, '48:00:00')}

set -euo pipefail

{GPU_MODULE_LOAD_BLOCK}

export OMP_PLACES=threads
export OMP_PROC_BIND=spread

export CC=cc
export CXX=CC
export OMP_NUM_THREADS={cpus}
export BASE=/global/common/software/m5228
export LOCAL="$BASE/local"

cd "{run_folder}"

date
srun --gpus-per-task={gpus} ./dmrg -f {input_path.name} -p {DMRG_PRECISION}
date
"""
    slurm_path.write_text(body, encoding="utf-8")
    print(f"Wrote batch script: {slurm_path}")
    return slurm_path


def main() -> int:
    """Create the run folder, the executable copy and the input; then run, submit, or stop."""
    args = parse_args_input()

    # Run names carry the chain parameters and the start time.
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_name = f"N={args.sites}_Nup={args.up}_Ndown={args.down}_{timestamp}"

    # The run folder sits under the project root.
    run_folder = PROJECT_ROOT / run_name
    run_folder.mkdir(parents=True, exist_ok=False)
    print(f"Created run folder: {run_folder}")

    executable_path = DMRG_EXECUTABLES[args.cluster]
    print(f"Using DMRG++ executable for cluster '{args.cluster}': {executable_path}")
    shutil.copy2(executable_path, run_folder / "dmrg")

    input_path = run_folder / f"input_{run_name}.ain"
    input_path.write_text(build_input(args, args.up, args.down, run_name), encoding="utf-8")
    print(f"Wrote Ainur input: {input_path}")

    slurm_path = None
    if args.cluster == "nersc":
        slurm_path = write_batch_script(run_folder, input_path, run_name)

    if args.run:
        subprocess.run(["./dmrg", "-f", str(input_path), "-p", f"{DMRG_PRECISION}"], cwd=run_folder, check=True)
    elif args.submit:
        subprocess.run(["sbatch", str(slurm_path)], cwd=run_folder, check=True)
    else:
        print("Input generated. Use --run to start DMRG++.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"Error: {error}")
