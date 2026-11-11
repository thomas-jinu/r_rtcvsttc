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

__2026-11-08 - GPU testing on NERSC with 12-site runs__: dmrg++ compiles with Release on NERSC without issues. Kokkos with threads is the fastest followed by Kokkos with openmp and Kokkos with serial for GPU runs. Multi-gpu runs failed.

__2026-11-09 - Backward Time Evolution with Gonzalo__: Gonzalo put in a fix but the tests reveal a new issue in apply stage that was not caught before. Probably wasn't caught from my earlier versions of code from copy_dmrg either. 

## Data

__2026-10-02 - 4-site and 8-site calculations are running.__: Data has been added to folders.

__2026-11-08 - 12-site runs for GPU scheduled on NERSC.__: This atleast can go on manuscript if I cannot go higher. ED runs are at 10 sites. 12 site runs are for Tmax=50 and Tmax=100 with delta_t=0.1. Kokkos_Threads is used. 

## Home

__2026-11-08 - Reorganized project directory.__ Moved markdown notes here instead of Zettler.__

## Reading

## Updates

__2026-11-06 - Thomas sked to run more GPU jobs to keep NERSC hours.__ Needs updated scripts with TSPAdvanceEach=N-2.

## Writing

## Things to remember

- In the TspAdvanceEach = N-2, the first time step never has all the sites. Time is changed at the boundary and since the first loop starts at the second or second to last site, the border site is always missing. Important for momentum-resolved measurements.
