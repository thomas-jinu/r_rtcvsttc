# r_rtcvsttc research log

_This log has only been regularly updated since 2026-11-08._

## Analysis

## Code

__Repo__: https://github.com/thomas-jinu/r_rtcvsttc

__2026-09-22 – The two-time scripts all handle TSPAdvanceEach = 2 * (N-2)__: Otherwise time changes at the border for every loop and for the first and last loop only half of the lattice is on the same time.

__2026-09-24 - Gonzalo promised a possible MPI routine for these simulations;__

__2026-10-02 - Reviewed basic pump parameters at Pump Definition and Parameters;__ Considering reverting scripts back to TSPAdvanceEach=N-2 due to how long 32-site simulations are.

__2026-10-05 - P_threads for dmrg++ internal have segmentation fault issues__: Likely due to BLAS libraries not being thread-safe. Known problem.

__2026-10-14 - Added tests for CPU vs GPU for performance checks__: Tested on NERSC Perlmutter.

__2026-10-20 - Tested GPUs with magma__: GPU's are faster. Magma needs a PR for Krylove Time Evolution. Fix implemented with Codex in my personal fork of dmrg++. PR for this added as a future task.

## Data

__2026-10-02 - 4-site and 8-site calculations are running.__: Data has been added to folders.

__2026-11-08 - 12-site runs for GPU.__: This atleast can go on manuscript if I cannot go higher. ED runs are at 10 sites.

## Home

__2026-11-08 - Reorganized project directory.__ Moved markdown notes here instead of Zettler.__

## Reading

## Updates

__2026-11-06 - Thomas sked to run more GPU jobs to keep NERSC hours.__ Needs updated scripts with TSPAdvanceEach=N-2.

## Writing
