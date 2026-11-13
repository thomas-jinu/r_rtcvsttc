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

__2026-10-20 - Tested GPUs with magma__: GPU's are faster. Magma needs a PR for Krylov Time Evolution. Fix implemented with Codex in my personal fork of dmrg++. PR for this added as a future task.

__2026-11-08 - GPU testing on NERSC with 12-site runs__: dmrg++ compiles with Release on NERSC without issues. Kokkos with threads is the fastest followed by Kokkos with openmp and Kokkos with serial for GPU runs. Multi-gpu runs failed.

__2026-11-09 - Backward Time Evolution with Gonzalo__: Gonzalo put in a fix which I tested locally and on NERSC.

__2026-11-10 - Automatic restart with Gonzalo__: 
Gonzalo put in a fix for automatic restart; it didn't test correctly; will need additional fix.

__2026-11-10 - Batch scripts now use N-2, are padded with values at ends to get all sites and TD and G inputs can be generated together and run.__

__2026-11-12 - Automatic restart now works__: There maybe an additional issue that showed up on NERSC but I haven't tested it yet. 

__2026-11-12 - Gonzalo implemented startLoop__ : The Hamiltonian looks for t=0 for non-time loops it doesn't help with my workflow. 


## Data

__2026-10-02 - 4-site and 8-site calculations are running.__: Data has been added to folders.

__2026-11-08 - 12-site runs for GPU scheduled on NERSC.__: This atleast can go on manuscript if I cannot go higher. ED runs are at 10 sites. 12 site runs are for Tmax=50 and Tmax=100 with delta_t=0.1. Kokkos_Threads is used. 

__2026-11-10 - Massively parallel 12-site runs for GPU scheduled on NERSC__: Just queued; won't run until the weekend or longer. The automatic restart is included but I expect that it won't be needed. 

__2026-11-12 - Parallel runs takes a long time to start__: NERSC only allows two jobs to accrue priority by time so the rest acts as if they were just submitted. Need to rethink this workflow!

__2026-11-13 - 16 site jobs started but not actively tracking__: Testing to see how this performs; I think this is also doable in the 48 hour window. 

## Home

__2026-11-08 - Reorganized project directory.__ Moved markdown notes here instead of Zettler.

## Reading

## Updates

__2026-11-06 - Thomas asked to run more GPU jobs to keep NERSC hours.__ Needs updated scripts with TSPAdvanceEach=N-2.

## Writing

## Things to remember

- In the TspAdvanceEach = N-2, the first time step never has all the sites. Time is changed at the boundary and since the first loop starts at the second or second to last site, the border site is always missing. Important for momentum-resolved measurements.
- Scripts now use N-2; the equal $t=t^\prime$ are calculated on apply stage while the others in the time evolution.
- Timeaxis are padded (at both ends) so that the border-sites are now computed and available in the Fourier transform.
- NERSC is a pain if submitting 100's of jobs. 1 large GPU job has more priority.
