""" 
ENSURE: 
For every script 
    - ADF CGNS FILE
    - USE FAMILY CONDITIONS = FALSE IN POINTWISE
    - EXPORT > CAE

For complex runs and Rohit's addition of eigenvalue solvers: 
    - PETSC_ARCH=complex-opt 
    - mpirun -np 4 python3.11 run_eigensolve.py

fix_bc.py will implement the family names and any other BC data

==================================================

STAGE 1 of 2 -- standard SA with ANK/NK to produce a converged restart file.
Writes ./output_SA/comp_corner_sa_000_vol.cgns in DOUBLE precision, which
adflow_run2.py then reloads as its restartFile.

USAGE: mpiexec -n 16 python3 adflow_run1.py

    nProc MATTERS. comp_corner_20_fixed.cgns is one 1185x145x3 node block, i.e.
    1184x144x2 cells; ADflow cuts it into nProc pieces and every cut plane must
    still land on a cell boundary after multigrid coarsening, i.e. at an even
    cell index. Probed on this grid with MGCycle "2w": nProc = 8 and 16 work,
    14 aborts in checkCoarse1to1 with "Non-matching block-to-block face".
    Use 16.

    16 is also the right number for the machine, and not because nproc says 32.
    This host is an AMD Ryzen 9 7950X: 16 PHYSICAL cores, 32 threads. nproc
    counts SMT siblings, which are not extra compute -- 16 ranks is one per
    physical core, i.e. saturation, not headroom. Running 32 would put two
    ranks on each core sharing one FP unit and one L2, which for a
    bandwidth-bound solve is neutral at best; it is moot anyway, since
    1184/32 = 37 is odd and would abort.

    The counts are not an accident -- mesh.py picks its default point counts so
    ni-1 factorises well, and prints the workable nProc list when it runs. The
    earlier 550/220 defaults gave ni-1 = 926 = 2 x 463 with 463 prime, and
    *every* nProc from 2 to 32 aborted, nProc = 2 included.

"""

import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--gridFile", type=str, default="./meshes/comp_corner_20_fixed.cgns")
parser.add_argument("--task", choices=["analysis", "polar"], default="analysis")
args = parser.parse_args()
outputDirectory = "./output_SA"

comm = MPI.COMM_WORLD
if not os.path.exists(outputDirectory):
    if comm.rank == 0:
        os.mkdir(outputDirectory)

aeroOptions = {
    # I/O Parameters
    "gridFile": args.gridFile,
    "outputDirectory": outputDirectory,
    "monitorVariables": ["resrho", "resturb", "yplus"],
    "surfaceVariables": ["cf", "cfx", "cfy", "cfz", "p", "vx", "vy", "vz", "temp", "rho", "mach"], 
    # Added to enable Rohits' implementation of SA-Edwards: enables double precision and volume restart file
    "volumeVariables": ["resrho", "resturb", "eddyratio", "eddy", "temp", "mach"],
    "writeVolumeSolution": True,
    "solutionPrecision": "double",
    "gridPrecision": "double",
    # Physics Parameters
    "eddyVisInfRatio": 0.2104, # makes ~v/v = 3, default is 1.342; apparently affects the location of the transition
    "equationType": "RANS",
    "turbulenceModel": "SA", # SA-Edwards will be implemented in the second stage
    # Drop the ft2 trip term -- i.e. run SA-noft2, which is what the NASA TMR
    # calls the standard fully-turbulent form of the model.
    #
    # This is here for CONSISTENCY WITH STAGE 2, not just convention.  Stage 2
    # runs SA-Edwards, and sa.F90:290-296 forces ft2 = 0 whenever useSAEdwards
    # is set, *before* it ever looks at useft2SA:
    #
    #     if (useSAEdwards) then
    #         ft2 = zero            <- Edwards drops ft2 by construction
    #     else if (useft2SA) then
    #         ft2 = rsaCt3 * exp(-rsaCt4 * chi2)
    #
    # So stage 2 has no ft2 whatever this option says.  Leaving it on here
    # means stage 1 converges a *different model* from the one stage 2 then
    # solves, and the restart carries a model discontinuity on top of the
    # Edwards change that stage 2 has to unwind.  Turning it off here makes the
    # two stages differ by the Edwards terms alone, which is the entire point
    # of splitting them.
    #
    # It bites hardest exactly where this case lives.  ft2 = 1.2*exp(-0.5*chi^2)
    # damps production while nuTilde is small, which is what "keeps a laminar
    # solution laminar"; with eddyVisInfRatio tuned to chi = 3 the freestream
    # ft2 is 0.013, small but not zero, and it is the mechanism behind the
    # "affects the location of the transition" note on that option.
    #
    # Unlike saVariant, this one is NOT a blockette trap: useft2SA is read by
    # both SA implementations -- sa.F90:293 (DADI path) and blockette.F90:1124
    # (ANK/NK path) -- so it is honoured whatever useBlockettes is set to.
    "useft2SA": False,
    "turbResScale": 1e5, # default
    # Solver Parameters
    # 2w needs one coarsening. The grid is a single 700x200x2 block that ADflow
    # cuts into nProc sub-blocks, and every cut has to stay 1-to-1 matching after
    # coarsening or coarseUtils.F90:1528 aborts with "Non-matching block-to-block
    # face". On this grid that only holds for nProc = 10 or 14 -- 4, 8 and 12 all
    # abort. Use 14 (see the usage line above); it is also the physical core count.
    "MGCycle": "2w",
    # Start on the COARSE grid (level 2 = 350x100x1) and prolong to the fine grid.
    # This is the single most important change for grid 15. Starting ANK on the
    # fine grid straight from uniform freestream drives a near-wall cell
    # unphysical on iteration 8 -- Y+_max jumps 9 -> 5297 -- and the line search
    # then rejects every step forever (totalRes frozen at 6.03e8, above the 3.74e8
    # it started from). The coarse grid has 4x fewer cells and a 2x larger first
    # off-wall cell, so the startup transient is survivable there; the prolonged
    # coarse solution is then a good enough initial guess that ANK takes 0.9-sized
    # steps on the fine grid from iteration 1.
    "MGStartLevel": 2,
    "nSubiter": 1, # how many checks before the next timestep is taken
    "nSubiterTurb": 7,  # was 3; turbulence lags badly in segregated ANK at 3
    # CFL/CFLCoarse drive the DADI smoother, NOT ANK (ANK has its own ANKCFL*).
    # CFLCoarse is the live one here: it is what runs the level-2 startup, and
    # 1.0 gets to totalRes 1.2e6 in 500 coarse cycles where 0.3 only reaches 1.1e7.
    # CFL is fine-grid DADI, which with the settings below never actually runs --
    # but keep it at 1.0, not 5.0: fine-grid DADI at CFL 5 from freestream NaNs on
    # this grid in two iterations (totalRes 4.8e51 then NaN).
    "CFL": 1.0,
    "CFLCoarse": 1.0,
    # ANK Solver Parameters
    "useANKSolver": True,
    # Switch tolerances are RELATIVE to the free-stream residual totalR0
    # (solvers.F90:1107: `if (totalR > ANK_switchTol * totalR0) call executeMGCycle`).
    # 1e11 is far above 1, so ANK owns the fine grid from its first iteration and
    # the fine-grid DADI smoother is never used. That is deliberate -- see the CFL
    # note above for what happens if it does run.
    "ANKSwitchTol": 1e11,
    # ANK's CFL sets the size of the I*V/(CFL*dt) term added to the diagonal of
    # the Jacobian, i.e. how much the pseudo-transient continuation regularises
    # the Newton step. ANKCFL0=5 (the default, and what grid 14 used) is too
    # aggressive here: grid 15's first off-wall cell is 9.56e-7 m against grid
    # 14's 1.60e-6 m, so the wall-normal Jacobian entries are stiffer and a
    # near-Newton first step overshoots into unphysical territory. 1.0 gives
    # enough regularisation to get started, and the ramp takes it back over 200
    # within ~35 iterations anyway.
    "ANKCFL0": 1.0,
    # NB: the bare "ILUFill" option that used to sit here was a no-op. It maps to
    # inputADjoint::fillLevel (pyADflow.py:6869) and is only read by the ADJOINT
    # KSP setup in adjointAPI.F90:917 -- this script runs no adjoint, so it never
    # did anything. The flow solvers have their own private copies below. ILU fill
    # is the level-of-fill of the incomplete LU used as the *local* (per-subdomain)
    # preconditioner inside additive Schwarz; it has no connection to CFL.
    "ANKPCILUFill": 2,
    "NKPCILUFill": 2,
    # Default ANK is segregated: it solves the mean flow implicitly and leaves
    # nuTilde to nSubiterTurb DADI sweeps. On this case that stalls -- res rho
    # drops 4 orders while res nuturb *climbs* (8.7e-6 -> 1.6e-3) and totalRes
    # flattens at ~8e4, well above the 1.95e1 NK switch, so NK is never reached.
    # Both switch tolerances default to 1e-16 in this build, i.e. never fire.
    # Coupled ANK puts nuTilde in the same implicit system as the mean flow;
    # second-order ANK is what actually gets the residual down asymptotically.
    "ANKSecondOrdSwitchTol": 1e-3,
    # Coupled ANK is deliberately left OFF (1e-16 = never). It was tried at 1e-3
    # and made things worse: SANK descended 1.85e6 -> 1.98e5 over iters 110-299,
    # then coupled mode engaged at iter 302 and flatlined -- 80+ iterations with
    # the line search rejecting nearly every step (step 0.00) and totalRes
    # drifting back up. Segregated SANK + more turb subiterations is what works.
    "ANKCoupledSwitchTol": 1e-16,
    "ANKNSubiterTurb": 3,
    # NK Solver Parameters
    "useNKSolver": True,
    # SANK's descent flattens around totalR ~2e5 (rel ~1e-3). NK is a true Newton
    # method and punches through where ANK's line search stalls, so hand off there
    # rather than at 1e-7 -- which ANK never reached in either earlier attempt.
    "NKSwitchTol": 1e-4,
    # Termination Criteria
    "L2Convergence": 1e-10,
    "nCycles": 20000,
}

ap = AeroProblem(
        name = "comp_corner_sa",
        mach = 2.95,
        reynolds = 63560,
        T = 108.0,
        # rho = 0.314,
        reynoldsLength = 2.27e-3, # change this so that density matches up
        areaRef = 1.0,
        chordRef = 1.0,
        evalFuncs = []
)

# echo the derived free-stream state ADflow will use 
print("=" * 50)
print(f"Mach              : {ap.mach:.4f}")
print(f"Velocity V        : {ap.V:.3f} m/s")
print(f"Speed of sound a  : {ap.a:.3f} m/s")
print(f"Density rho       : {ap.rho:.5f} kg/m^3")
print(f"Temperature T     : {ap.T:.3f} K")
print(f"Pressure P        : {ap.P:.3f} Pa")
print(f"Viscosity mu      : {ap.mu:.4e} Pa.s")
print(f"reynoldsLength L  : {ap.reynoldsLength:.4e} m")
print(f"Re = rho*V*L/mu   : {ap.re:.1f}")
print("=" * 50)

# Create solver
CFDSolver = ADFLOW(options = aeroOptions)
CFDSolver(ap)

funcs = {}
CFDSolver.evalFunctions(ap, funcs)
 
restartPath = os.path.join(outputDirectory, f"{ap.name}_000_vol.cgns")
if comm.rank == 0:
    print(funcs)
    print("=" * 50)
    print("Stage 1 complete. Pass this file to stage 2:")
    print(f"  --restartFile {restartPath}")
    print("=" * 50)