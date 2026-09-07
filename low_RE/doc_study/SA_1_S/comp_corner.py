""" 
ENSURE:
    - ADF CGNS FILE
    - USE FAMILY CONDITIONS = FALSE IN POINTWISE
    - EXPORT > CAE

    PETSC_ARCH=complex-opt mpirun -np 4 python3.11 run_eigensolve.py

fix_bc.py will implement the family names and any other BC data
USAGE: mpiexec -n # python3.11 comp_corner.py 
""" 

import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="./output")
parser.add_argument("--gridFile", type=str, default="./meshes/comp_corner_10_fixed.cgns")
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
    # Physics Parameters
    "equationType": "RANS",
    "turbulenceModel": "SA", # supposed to be SA-Edwards
    "turbResScale": 1e5, # default
    # Solver Parameters
    "MGCycle": "sg",
    "nSubiterTurb": 10,
    # ANK Solver Parameters
    "useANKSolver": True,
    # NK Solver Parameters
    "useNKSolver": True,
    "NKSwitchTol": 1e-6,
    # Termination Criteria
    "L2Convergence": 1e-10,
    "nCycles": 50000,
}

ap = AeroProblem(
        name = "comp_corner",
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
#print(f"ap.reynolds attr  : {ap.reynolds}")   # None if not supplied as input
print("=" * 50)

# Create solver
CFDSolver = ADFLOW(options = aeroOptions)

if args.task == "analysis":
    CFDSolver.setAeroProblem(ap)
    CFDSolver(ap)
    # rst Evaluate and printan
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)
    # Print the evaluated functions
    if comm.rank == 0:
        print(funcs)
