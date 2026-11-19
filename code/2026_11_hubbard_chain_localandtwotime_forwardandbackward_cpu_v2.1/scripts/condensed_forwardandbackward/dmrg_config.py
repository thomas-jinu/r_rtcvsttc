"""Shared settings for the Hubbard-chain DMRG++ scripts.

Change paths, precision, and NERSC/Slurm settings here, not in each script.
"""

import os
from pathlib import Path

# DMRG++ executables per cluster. Each run folder gets a copy as ./dmrg.
DMRG_EXECUTABLES = {
    "local": Path("/Users/qqt/Documents/Codes/dmrgpp_alltimerecovery_fixes/dmrgpp/build/dmrg/dmrg"),
    "nersc": Path("/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/14_12site_GPU_twotime/dmrgpp_fixrestart/dmrgpp_cpu/build_perlmutter_cpu_threads/dmrg/dmrg"),
    # "nersc": Path("/Users/qqt/Documents/Codes/dmrgpp_pvector/copy_dmrg/installdir/bin/dmrg"),
}

# DMRG++ precision flag passed as -p.
DMRG_PRECISION = 12

# Number of recovery files kept by the G< / G> / singletime / XAS runs (RecoverySave @M=).
# Why two: a job killed while writing a checkpoint leaves that file incomplete, and DMRG++
# skips any file that fails its validity check. With two files, if the newest checkpoint is
# unusable, DMRG++ restarts from the older one and recomputes only the loops after it.
# Tested locally: with either file corrupted, or only one present, the run restarted from the
# other file and reproduced the uninterrupted results. DMRG++ picks the newest file by its
# save counter, not by timestamp. The TD run keeps one file per time step, as before.
RECOVERY_FILES_KEPT = 2

# NERSC / Slurm settings used by the generated batch scripts.
SLURM_ACCOUNT = "m5228"
SLURM_QOS = "regular"
SLURM_CONSTRAINT = "cpu"
SOFTWARE_BASE = "/global/common/software/m5228"

# Modules loaded at the top of every NERSC batch script.
MODULE_LOADS = (
    "module load cray-mpich/9.0.1",
    "module load cray-libsci/25.09.0",
    "module load cray-hdf5/1.14.3.7",
    "module load PrgEnv-gnu",
)
MODULE_LOAD_BLOCK = "\n".join(MODULE_LOADS)


def sbatch_header(job_name: str, nodes: int, cpus_per_task: int, walltime: str, ntasks: int | None = None) -> str:
    """Shebang and #SBATCH lines shared by every NERSC batch script (no trailing newline)."""
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
    """Strict-mode line, module loads and OpenMP settings used by the TD/G/XAS batches (no trailing newline)."""
    return "\n".join((
        "set -euo pipefail",
        MODULE_LOAD_BLOCK,
        f"export OMP_NUM_THREADS={max(1, cpus_per_task // 2)}",
        "export OMP_PLACES=cores",
        "export OMP_PROC_BIND=spread",
    ))


def link_binary(run_binary: Path, target_dir: Path) -> None:
    """Make target_dir/dmrg a relative symlink to the run's own binary copy.

    Step folders share the run's binary instead of holding copies. The link is
    relative, so the run folder can be moved together with its contents.
    """
    link = target_dir / "dmrg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(run_binary, target_dir))
