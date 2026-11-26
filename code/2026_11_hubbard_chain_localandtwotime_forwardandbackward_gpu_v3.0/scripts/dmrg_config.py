"""Shared helpers for the Hubbard-chain DMRG++ scripts (GPU workflow).

Change paths and recovery settings here, not in each script.
"""

import os
from pathlib import Path

# Executable paths shared by all workflow drivers. Local uses the macOS CPU build.
DMRG_EXECUTABLES = {
    "local": Path("/Users/qqt/Library/CloudStorage/Dropbox/ORNL/Codes/dmrgpp_alltimerecovery_fixes/dmrgpp/build/dmrg/dmrg"),
    "nersc": Path("/pscratch/sd/t/thomasj2/r_rtcvsttc/data/2026_november/14_12site_GPU_twotime/dmrgpp_fixrestart/dmrgpp/build_perlmutter_nvidia_cuda/dmrg/dmrg"),
}

# Precision passed to DMRG++ with -p.
DMRG_PRECISION = 12

# Recovery files kept by the G< / G> / XAS runs (RecoverySave @M=).
# Two files let DMRG++ restart from the older one if the newest is unusable. The TD run keeps
# one file per time step.
RECOVERY_FILES_KEPT = 2


# ---------------------------------------------------------------------------
# NERSC GPU batch settings
# ---------------------------------------------------------------------------

GPU_SLURM_ACCOUNT = "m5228"
GPU_SLURM_QOS = "regular"
GPU_SLURM_CONSTRAINT = "gpu"
# Perlmutter GPU nodes: 128 CPUs and 4 GPUs, so 32 CPUs per GPU. Each batch job requests
# every GPU on its nodes; each srun step (one DMRG++ run) gets GPUs in proportion to its CPUs.
CPUS_PER_NODE = 128
GPUS_PER_NODE = 4
CPUS_PER_GPU = CPUS_PER_NODE // GPUS_PER_NODE
STEP_CPU_SIZES = (32, 64, 128)


def gpu_layout(cpus_per_task: int) -> tuple[int, int]:
    """(GPUs per step, steps per node): 32 CPUs -> (1, 4), 64 -> (2, 2), 128 -> (4, 1)."""
    if cpus_per_task not in STEP_CPU_SIZES:
        sizes = ', '.join(str(size) for size in STEP_CPU_SIZES)
        raise ValueError(f"cpus-per-task must be one of {sizes} on a GPU node, got {cpus_per_task}")
    gpus = cpus_per_task // CPUS_PER_GPU
    return gpus, GPUS_PER_NODE // gpus

# Modules loaded at the top of every GPU batch script.
GPU_MODULE_LOADS = (
    "module load cray-mpich/9.0.1",
    "module load cray-libsci/25.09.0",
    "module load cray-hdf5/1.14.3.7",
    "module load PrgEnv-nvidia",
    "module load cudatoolkit",
)


def gpu_sbatch_header(job_name: str, nodes: int, cpus_per_task: int, walltime: str) -> str:
    """Shebang and #SBATCH lines for GPU jobs. No trailing newline.

    The job takes every GPU on its nodes, with as many steps per node as the CPU size allows.
    """
    _, steps_per_node = gpu_layout(cpus_per_task)
    return "\n".join((
        "#!/bin/bash",
        f"#SBATCH --account={GPU_SLURM_ACCOUNT}",
        f"#SBATCH --qos={GPU_SLURM_QOS}",
        f"#SBATCH --constraint={GPU_SLURM_CONSTRAINT}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks-per-node={steps_per_node}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --gpus-per-node={GPUS_PER_NODE}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --output=%x-%j.out",
        "#SBATCH --error=%x-%j.err",
    ))


GPU_MODULE_LOAD_BLOCK = "\n".join(GPU_MODULE_LOADS)


def gpu_batch_environment(cpus_per_task: int) -> str:
    """Strict mode, GPU module loads and OpenMP settings. No trailing newline."""
    return "\n".join((
        "set -euo pipefail",
        GPU_MODULE_LOAD_BLOCK,
        # One OpenMP thread per CPU (hardware thread) of the step.
        f"export OMP_NUM_THREADS={cpus_per_task}",
        "export OMP_PLACES=threads",
        "export OMP_PROC_BIND=spread",
    ))


def link_binary(run_binary: Path, target_dir: Path) -> None:
    """Make target_dir/dmrg a relative symlink to the run's binary copy.

    Relative, so the run folder still works if it is moved together with its contents.
    """
    link = target_dir / "dmrg"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(run_binary, target_dir))
