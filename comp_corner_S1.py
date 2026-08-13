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
Writes ./output/comp_corner_sa_vol.cgns in DOUBLE precision, which
comp_corner_sa_edwards.py then reloads as its restartFile.

USAGE: mpiexec -n # python3.11 comp_corner_S1.py 

""" 

import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="./output_SA")
parser.add_argument("--gridFile", type=str, default="./meshes/comp_corner_14_fixed.cgns")
parser.add_argument("--task", choices=["analysis", "polar"], default="analysis")
args = parser.parse_args()

comm = MPI.COMM_WORLD
if not os.path.exists(args.output):
    if comm.rank == 0:
        os.mkdir(args.output)

aeroOptions = {
    # I/O Parameters
    "gridFile": args.gridFile,
    "outputDirectory": args.output,
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
    "turbResScale": 1e5, # default
    # Solver Parameters
    "MGCycle": "2w",
    "MGStartLevel": 1,
    "nSubiter": 1, # how many checks before the next timestep is taken
    "nSubiterTurb": 10,  # was 3; turbulence lags badly in segregated ANK at 3
    "CFL": 1.5, # Value that directly affects timestep size
    "CFLCoarse": 1.0, # Value that directly affects timestep size in coarse regions, usually lower than CFL 
    # ANK Solver Parameters
    "useANKSolver": True,
    "ANKSwitchTol": 1e10,
    "ILUFill": 3, # Your steering system, eats up memory but controls your CFL/timesteps
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
 
restartPath = os.path.join(args.output, f"{ap.name}_000_vol.cgns")
if comm.rank == 0:
    print(funcs)
    print("=" * 50)
    print("Stage 1 complete. Pass this file to stage 2:")
    print(f"  --restartFile {restartPath}")
    print("=" * 50)