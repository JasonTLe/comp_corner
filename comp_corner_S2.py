"""
STAGE 2 of 2 -- SA-Edwards&Chandra restarted from the stage-1 SA solution.

HOW SA-Edwards IS ACTUALLY SELECTED IN THIS FORK
------------------------------------------------
There are two *different* knobs and only one of them does anything:

  * turbulenceModel = "SA-Edwards"  -> inputphysics.turbmodel = 3.
    This is DEAD. Every solver dispatch is a `select case (turbModel)` that
    only lists `spalartAllmaras` (=2):
        src/turbulence/turbAPI.F90:68   turbSolveDDADI   (DADI / multigrid)
        src/turbulence/turbAPI.F90:140  turbResidual
        src/NKSolver/blockette.F90:622,812                (ANK / NK)
        src/output/outputMod.F90:3301                     (turb output vars)
    With turbmodel=3 no case matches, sa_block() is never called, saResScale()
    never fills dw(:,:,:,itu1), so resturb is exactly 0.000 and nuTilde is
    frozen -- in EVERY solver path, not just ANK/NK.

  * saVariant = "SA-Edwards"        -> inputphysics.savariant = 2.
    This is the live knob. src/turbulence/sa.F90:150-151 is the only consumer:
        useSAEdwards = (saVariant == saEdwards) or (saVariant == saCompEdwards)
    It defaults to "SA" (pyADflow.py:6185), so it must be set explicitly.

So the correct combination is turbulenceModel="SA" + saVariant="SA-Edwards".
Because turbmodel stays 2, ANK/NK are perfectly usable; they are left off here
only to keep the restart gentle while the new source term settles in.

WHY STAGE 2 USED TO "CONVERGE" IN ONE CYCLE
-------------------------------------------
Not multigrid -- MGCycle "sg" is correct for a 1-cell-thick mesh and multigrid
is optional. ADflow's L2Convergence is measured against the FREE-STREAM
residual, not the restart residual:
    src/solver/solvers.F90:972   call getFreeStreamResidual(rhoRes0, totalR0)
    src/solver/solvers.F90:1743  if (totalR < L2Conv * totalR0) absConv = .True.
Stage 1 already drove totalR down to 1e-10*totalR0, so reusing
L2Convergence=1e-10 here is satisfied on cycle 1. L2ConvergenceRel (line 1748)
is the one measured against totalRStart, i.e. the residual of the restart
state; the two checks are OR'd (line 1761).

USAGE:
    BE CAREFUL WITH THE NUMBER OF PROCS YOU USE. COARSENING DOES NOT ALLOW A CERTAIN AMOUNT OF PROCS DUE TO UNEVEN SPLIT

    For comp_corner_15_fixed.cgns (one 700x200x2 block) that means nProc = 10 or
    14 only. 4, 8 and 12 all abort in coarseUtils.F90:1528 with "Non-matching
    block-to-block face" because the sub-block cuts stop being 1-to-1 after the
    "2w" coarsening. 14 is also the physical core count on this machine.
    Multigrid matters more here than in stage 1: stage 2 is a pure DADI run, so
    the coarse level is doing real work rather than sitting unused.

    mpiexec -n 14 python3.11 comp_corner_S2.py \
        --gridFile ./meshes/comp_corner_15_fixed.cgns \
        --restartFile ./output_SA/comp_corner_sa_000_vol.cgns
"""

import argparse
import os

from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--gridFile", type=str, default="./meshes/comp_corner_15_fixed.cgns")
parser.add_argument(
    "--restartFile",
    type=str,
    default="./output_SA/comp_corner_sa_000_vol.cgns",
    help="Double-precision volume CGNS written by comp_corner_sa.py",
)
args = parser.parse_args()
outputDirectory = "./output_SAE"

comm = MPI.COMM_WORLD
if comm.rank == 0 and not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
comm.barrier()

if not os.path.isfile(args.restartFile):
    raise FileNotFoundError(f"restartFile not found: {args.restartFile}. Run comp_corner_S1.py first.")

aeroOptions = {
    # I/O Parameters
    "gridFile": args.gridFile,
    "restartFile": args.restartFile,
    "outputDirectory": outputDirectory,
    "monitorVariables": ["resrho", "resturb", "yplus"],
    "surfaceVariables": ["cf", "cfx", "cfy", "cfz", "p", "vx", "vy", "vz", "temp", "rho", "mach", "yplus"],
    "volumeVariables": ["resrho", "resturb", "eddyratio", "eddy", "temp", "mach"],
    "solutionPrecision": "double",
    "gridPrecision": "double",
    # Physics Parameters
    "eddyVisInfRatio": 0.2104, # makes ~v/v = 3, default is 1.342; apparently affects the location of the transition
    "equationType": "RANS",
    "turbulenceModel": "SA",  # must stay "SA" (turbmodel=2) or nothing happens
    "saVariant": "SA-Edwards",  # knob to turn on edwards
    "turbResScale": 1e5,  
    # Solver Parameters -- DADI
    "smoother": "DADI", # default
    "MGCycle": "2w",
    # sg settings
    #"nSubiterTurb": 10,
    # multigrid settings
    "MGStartLevel": 1,  # Coarsening is not available on restart files?
    "nSubiter": 1,
    "nSubiterTurb": 3,
    # CFL 1.5 is far too conservative for DADI here: it stalls at resrho ~1.6 and
    # the rate decays to ~3e-4/iter, which extrapolates to ~62k cycles -- past the
    # 20k budget. DADI is implicit and takes a much larger CFL than this.
    "CFL": 5.0,
    "CFLCoarse": 1.0,
    # -----------------------------------------------------------------------
    # useBlockettes MUST BE FALSE WHENEVER ANK/NK IS ON WITH saVariant SET.
    # -----------------------------------------------------------------------
    # There are TWO SA implementations in this fork and only one knows about
    # saVariant:
    #
    #   src/turbulence/sa.F90       sa_block -> saSource/saViscous. HAS the
    #                               Edwards terms (useSAEdwards is decoded at
    #                               sa.F90:150-151 and used at 212, 290, 300, 326).
    #   src/NKSolver/blockette.F90  its OWN private saSource/saAdvection/saViscous
    #                               (blockette.F90:976, 1170, 1392). Contains ZERO
    #                               references to saVariant or Edwards.
    #
    # ANK and NK evaluate the residual through blocketteRes, which dispatches on
    # useBlockettes (blockette.F90:271-275):
    #
    #     if (useBlockettes) then         ! DEFAULT IS TRUE
    #         call blocketteResCore(...)  ! -> blockette's private, standard-SA
    #     else
    #         call blockResCore(...)      ! -> sa_block from sa.F90, Edwards OK
    #
    # So with the default useBlockettes=True, turning ANK/NK on silently swaps
    # the turbulence model back to standard SA. This is not subtle when you look
    # for it: restarting the Edwards-converged field with useBlockettes=True
    # reports res nuturb 1.6403e-03 / totalRes 8.68e+04 (instead of the true
    # 8.77e-11 / 4.64e-03) and then freezes -- identical to 13 significant
    # figures after 99 iterations -- because the DADI turbulence update is
    # solving Edwards while the monitor is measuring standard SA. With
    # useBlockettes=False the same restart reports the correct 4.64e-03 and NK
    # converges it in FOUR iterations.
    #
    # This is the same class of trap as turbulenceModel="SA-Edwards" being dead
    # (see the header): the option is accepted, nothing errors, and the physics
    # is quietly wrong.
    "useBlockettes": False,
    # DADI alone cannot finish this problem. It drives the MEAN FLOW to
    # res rho 4.4e-10 (14 orders) but res nuturb floors at ~8.8e-11, which pins
    # totalRes at 4.64e-03 against the 3.74e-04 target. Verified as a genuine
    # stall, not a budget problem: a control run of 400 further DADI cycles from
    # that state moved totalRes only 4.6410e-03 -> 4.6325e-03.
    #
    # So DADI still does the settling-in that this stage exists for -- the
    # Edwards source term fires on cycle 1 and throws the residual to ~8.2e+04,
    # and the smoother walks that back down -- but NK is handed the endgame.
    # NKSwitchTol 1e-6 means NK takes over at totalRes = 1e-6 * totalR0 = 374,
    # which DADI reaches around cycle 4000, long after the source term has
    # settled. ANK stays off: it is segregated here, so its turbulence update is
    # the same DADI that is stalling, and it would not touch the floor.
    "useANKSolver": False,
    "useNKSolver": True,
    "NKSwitchTol": 1e-6,
    # Termination Criteria
    # L2Convergence is measured against the FREE-STREAM residual. On
    # comp_corner_15_fixed that is totalR0 = 3.74e8 (grid 14 was 1.947e8), so
    # 1e-12 means totalR <= 3.74e-4. It must stay tighter than ~3e-11 or stage 2
    # exits on cycle 1: stage 1 hands over a state whose residual is already
    # 1.19e-2, and only on cycle 1 does the Edwards source term kick in and throw
    # the residual up to 8.2e4.
    #
    # L2ConvergenceRel is NOT the useful knob here, contrary to what one might
    # expect for a restart. It is measured against totalRStart, which ADflow
    # takes at cycle 0 -- i.e. stage 1's *converged* 1.37e-2, not the 4.5e4 the
    # Edwards term jumps to. 1e-8 therefore asks for 1.4e-10 and never fires.
    # It is left on only as a harmless OR'd backstop.
    "L2Convergence": 1e-12,
    "L2ConvergenceRel": 1e-8,
    "nCycles": 40000,
}

# Must match comp_corner_sa.py exactly.
ap = AeroProblem(
    name="comp_corner_sa_edwards",
    mach=2.95,
    reynolds=63560,
    T=108.0,
    reynoldsLength=2.27e-3,
    areaRef=1.0,
    chordRef=1.0,
    evalFuncs=[],
)

CFDSolver = ADFLOW(options=aeroOptions)

# ---------------------------------------------------------------------------
# Check 1: did the option actually reach the Fortran layer?
# constants.F90: turbModel  spalartAllmaras = 2, spalartAllmarasEdwards = 3
# constants.F90: saVariant  saStandard = 0, saComp = 1, saEdwards = 2,
#                           saCompEdwards = 3
# Checking turbmodel == 3 is NOT a valid test -- it passes while the turbulence
# equation is never solved at all. saVariant is the one that matters.
# ---------------------------------------------------------------------------
turbModel = int(CFDSolver.adflow.inputphysics.turbmodel)
saVariant = int(CFDSolver.adflow.inputphysics.savariant)
if comm.rank == 0:
    print("=" * 60)
    print(f"inputphysics.turbmodel = {turbModel}  (must be 2; 3 is never dispatched)")
    print(f"inputphysics.savariant = {saVariant}  (2 = SA-Edwards)")
    if turbModel != 2:
        print("WARNING: turbmodel != 2 -- sa_block() will never be called, resturb will be 0.")
    if saVariant != 2:
        print("WARNING: saVariant != 2 -- the Edwards terms in sa.F90 are inactive.")
    print("=" * 60)


NW_RANS = 6  # rho, u, v, w, rhoE, nuTilde


def nuTildeRange():
    """min/max of nuTilde over all cells on all ranks (state index 5 for RANS)."""
    nuTilde = CFDSolver.getStates().reshape(-1, NW_RANS)[:, 5]
    return comm.allreduce(nuTilde.min(), op=MPI.MIN), comm.allreduce(nuTilde.max(), op=MPI.MAX)


CFDSolver.setAeroProblem(ap)
lo0, hi0 = nuTildeRange()

CFDSolver(ap)

lo1, hi1 = nuTildeRange()

funcs = {}
CFDSolver.evalFunctions(ap, funcs)

# ---------------------------------------------------------------------------
# Check 2: did nuTilde actually evolve? If the field is frozen, the turbulence
# equation was never solved, regardless of what the residual column showed.
# ---------------------------------------------------------------------------
if comm.rank == 0:
    print(funcs)
    print("=" * 60)
    print(f"nuTilde before solve : [{lo0:.6e}, {hi0:.6e}]")
    print(f"nuTilde after  solve : [{lo1:.6e}, {hi1:.6e}]")
    spread = (hi1 - lo1) / max(abs(hi1), 1e-30)
    if spread < 1e-8:
        print("FAIL: nuTilde is spatially uniform -- the turbulence equation was NOT solved.")
    else:
        print("OK: nuTilde varies spatially -- the turbulence equation is being solved.")
    print("=" * 60)
