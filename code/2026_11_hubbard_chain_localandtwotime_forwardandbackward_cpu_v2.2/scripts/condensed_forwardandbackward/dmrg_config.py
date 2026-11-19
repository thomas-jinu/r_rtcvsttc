"""Shared settings and helpers for the Hubbard-chain DMRG++ scripts.

Change paths, precision, and NERSC settings here, not in each script.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# DMRG++ executables and precision
# ---------------------------------------------------------------------------

# Each run folder gets its own copy (or a link to one) of this binary, as ./dmrg.
DMRG_EXECUTABLES = {
    "local": Path("/Users/qqt/Documents/Codes/dmrgpp_alltimerecovery_fixes/dmrgpp/build/dmrg/dmrg"),
    "nersc": Path("/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/14_12site_GPU_twotime/dmrgpp_fixrestart/dmrgpp_cpu/build_perlmutter_cpu_threads/dmrg/dmrg"),
    # "nersc": Path("/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"),
}

# Precision passed to DMRG++ with -p.
DMRG_PRECISION = 12

# ---------------------------------------------------------------------------
# Recovery files
# ---------------------------------------------------------------------------

# Recovery files kept by the G< / G> / singletime / XAS runs (RecoverySave @M=).
# Two files let DMRG++ restart from the older one if the newest is unusable (for example,
# the job was killed while writing it). Tested locally; see the notes in the project docs.
# The TD run keeps one file per time step.
RECOVERY_FILES_KEPT = 2

# ---------------------------------------------------------------------------
# NERSC / Slurm
# ---------------------------------------------------------------------------

SLURM_ACCOUNT = "m5228"
SLURM_QOS = "regular"
SLURM_CONSTRAINT = "cpu"
SOFTWARE_BASE = "/global/common/software/m5228"

# Modules loaded at the top of every NERSC batch script. Same versions and order as
# nersc_build_cpu.sh, so the jobs run with the libraries the binary was built against.
MODULE_LOADS = (
    "module load PrgEnv-gnu/8.7.0",
    "module load cray-mpich/9.1.0",
    "module load cray-libsci/26.03.0",
    "module load cray-hdf5/1.14.3.7",
)
MODULE_LOAD_BLOCK = "\n".join(MODULE_LOADS)


def sbatch_header(job_name: str, nodes: int, cpus_per_task: int, walltime: str, ntasks: int | None = None) -> str:
    """Shebang and #SBATCH lines. No trailing newline."""
    # Tasks default to one per node.
    ntasks = nodes if ntasks is None else ntasks
    return "\n".join((
        "#!/bin/bash",
        f"#SBATCH --account={SLURM_ACCOUNT}",
        f"#SBATCH --qos={SLURM_QOS}",
        f"#SBATCH --constraint={SLURM_CONSTRAINT}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=%x-%j.out",
        "#SBATCH --error=%x-%j.err",
    ))


def batch_environment(cpus_per_task: int) -> str:
    """Strict mode, module loads and OpenMP settings for the TD, G and XAS batches. No trailing newline."""
    return "\n".join((
        "set -euo pipefail",
        MODULE_LOAD_BLOCK,
        # One OpenMP thread per CPU (hardware thread) of the step.
        f"export OMP_NUM_THREADS={cpus_per_task}",
        "export OMP_PLACES=threads",
        "export OMP_PROC_BIND=spread",
    ))


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def link_binary(run_binary: Path, target_dir: Path) -> None:
    """Make target_dir/dmrg a relative symlink to the run's binary copy.

    Relative, so the run folder still works if it is moved together with its contents.
    """
    link = target_dir / "dmrg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(run_binary, target_dir))
