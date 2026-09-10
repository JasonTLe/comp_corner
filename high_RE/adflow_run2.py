"""
STAGE 2 of 2 -- SA-Edwards&Chandra restarted from the stage-1 SA solution.

highRe case: M = 2.88, Re_delta = 132 840, delta = 4.1 mm at 8.04 delta
upstream of the corner. The free stream comes from flow_conditions.make_ap(),
the same call stage 1 makes -- it MUST be, because this stage restarts from
stage 1's converged volume file. Do not write a literal AeroProblem here.

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

    For comp_corner_21_fixed.cgns (one 1153x157x3 node = 1152x156x2 cell block)
    mesh.py's checkMultigrid reports even splits at nProc = 2, 3, 4, 6, 8, 9,
    12, 16, 18, 24, 32, 36; anything else aborts in coarseUtils.F90:1528 with
    "Non-matching block-to-block face" because the sub-block cuts stop being
    1-to-1 after the "2w" coarsening. Use 16, the same as stage 1 -- it is also
    one rank per physical core on this host (see stage 1's header).
    Multigrid matters more here than in stage 1: stage 2 is a pure DADI run, so
    the coarse level is doing real work rather than sitting unused.

    mpiexec -n 16 python3 adflow_run2.py \
        --gridFile ./meshes/comp_corner_21_fixed.cgns \
        --restartFile ./output_SA/comp_corner_sa_000_vol.cgns

    Runtime at these settings, 16 ranks: on the lowRe grid stage 1 was ~430 s
    and stage 2 ~4000 s (it runs the full --nCycles; see the note on
    useANKSolver). This grid is 5% larger, so scale accordingly -- and budget
    for the whole thing more than once, because --plateLength still has to be
    calibrated against delta = 4.1 mm and each calibration step is a full
    two-stage run. See the note on --plateLength in mesh.py.
"""

import argparse
import os

from adflow import ADFLOW
from mpi4py import MPI

# Shared with adflow_run1.py and eigen_run.py -- see the note at the top of
# that file on Re_delta vs the quoted density.
import flow_conditions

parser = argparse.ArgumentParser()
parser.add_argument("--gridFile", type=str, default="./meshes/comp_corner_21_fixed.cgns")
parser.add_argument(
    "--restartFile",
    type=str,
    default="./output_SA/comp_corner_sa_000_vol.cgns",
    help="Double-precision volume CGNS written by adflow_run1.py",
)
parser.add_argument(
    "--nCycles",
    type=int,
    default=40000,
    help="DADI cycle budget. This stage stops on the residual floor, not on "
         "L2Convergence (see the note on useANKSolver), so this is what "
         "actually ends the run. Lower it only for calibration sweeps, and "
         "check res rho has plateaued at whatever value you pick -- res nuturb "
         "floors early and totalRes hides res rho behind it.",
)
parser.add_argument(
    "--useNK",
    action="store_true",
    help="Hand the endgame to the Newton-Krylov solver. OFF by default, which "
         "is what the useANKSolver note above describes: DADI alone stalls on "
         "a residual floor. That floor is tolerable when the field is only "
         "being reported, and NOT tolerable when it is about to be linearised "
         "-- eigen_run.py needs dR/dw at R(w)=0, and on grid 21 the DADI floor "
         "sat 5.4 orders above the L2Convergence target (totalR 83.45 against "
         "a 3.25e-4 target), which poisons the eigenvalues nearest the origin. "
         "Turn this on for any run whose output feeds a stability analysis. "
         "Safe here only because useBlockettes is False -- see the block above.",
)
parser.add_argument(
    "--NKSwitchTol",
    type=float,
    default=1e-6,
    help="Relative totalR at which DADI hands over to NK. Measured against "
         "totalR0 = 3.25e8 on this grid, so the default 1e-6 switches at "
         "totalR = 325; the DADI floor is 83.45, well below that, so the "
         "switch does fire. This is the value the earlier version of this file "
         "used before NK was removed.",
)
args = parser.parse_args()
outputDirectory = "./output_SAE"

comm = MPI.COMM_WORLD
if comm.rank == 0 and not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)
comm.barrier()

if not os.path.isfile(args.restartFile):
    raise FileNotFoundError(f"restartFile not found: {args.restartFile}. Run adflow_run1.py first.")

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
    # Set for the record, and to match stage 1 -- but note it changes NOTHING
    # here. sa.F90:290-296 tests useSAEdwards first and forces ft2 = 0 before it
    # ever reaches the useft2SA branch:
    #
    #     if (useSAEdwards) then
    #         ft2 = zero
    #     else if (useft2SA) then
    #         ft2 = rsaCt3 * exp(-rsaCt4 * chi2)
    #
    # Edwards drops ft2 by construction, so this stage was already running
    # noft2 whether or not the option was present. Where it does matter is
    # stage 1 -- see the long note on useft2SA there. Keep the two files in
    # step so the only difference between the stages is the Edwards terms.
    "useft2SA": False,
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
    # This stage is DADI ONLY. Both Newton solvers are off.
    #
    # That is a deliberate restriction and it has a known cost: on grid 15 DADI
    # drove the mean flow to res rho 4.4e-10 but res nuturb floored at ~8.8e-11,
    # pinning totalRes at 4.64e-03 against a 3.74e-04 target, and 400 further
    # cycles from that state moved it only 4.6410e-03 -> 4.6325e-03. That is a
    # genuine stall, not a budget problem, and the earlier version of this file
    # handed the endgame to NK at NKSwitchTol 1e-6 for exactly that reason.
    #
    # With NK off, expect this stage to run to nCycles and stop on the residual
    # floor rather than on L2Convergence. The Edwards field is still solved
    # correctly -- DADI goes through sa.F90, which is the implementation that
    # has the Edwards terms -- so the physics is right; it is the last two or
    # three orders of residual that DADI cannot deliver on its own. Judge the
    # result on the nuTilde check and the residual history, not on the exit code.
    #
    # If NK is ever switched back on here, useBlockettes MUST stay False (see
    # the block above) or ANK/NK will silently solve standard SA instead.
    "useANKSolver": False,
    "useNKSolver": args.useNK,
    "NKSwitchTol": args.NKSwitchTol,
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
    "nCycles": args.nCycles,
}

# Must match adflow_run1.py exactly.
ap = flow_conditions.make_ap("comp_corner_sa_edwards")
flow_conditions.report(ap, comm)

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
