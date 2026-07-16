# ENSURE THAT THE CGNS FILE IS ADF FORMAT AND USE FAMILY CONDITIONS=FALSE IN POINTWISE

import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="./output")
parser.add_argument("--gridFile", type=str, default="./comp_corner_5.cgns")
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
    "monitorVariables": ["resrho", "yplus", "resturb"],
    "surfaceVariables": ["cf", "p", "vx", "vy", "vz", "temp", "rho", "mach", "yplus"], 
    # Physics Parameters
    "equationType": "RANS",
    "turbulenceModel": "SA", # SA-Edwards
    "turbResScale": 10e4, # no clue
    # Solver Parameters
    "MGCycle": "sg",
    "nSubiterTurb": 10,
    # ANK Solver Parameters
    "useANKSolver": True,
    # NK Solver Parameters
    "useNKSolver": True,
    "NKSwitchTol": 1e-4,
    # Termination Criteria
    "L2Convergence": 1e-15,
    "nCycles": 10000,
}

# all weird, random init conditions
ap = AeroProblem(
        name="comp_corner",
        mach=2.95, 
        reynolds=63560,
        reynoldsLength=30.0,
        T=108.0,
        areaRef=30.0,
        chordRef=30.0,
        evalFuncs=[]
)
ap.setBCVar("Temperature", 275.4, "wall")

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

if args.task == "analysis":
    CFDSolver.setAeroProblem(ap)

    #dummy_fluxes = CFDSolver.getHeatFluxes(ap, groupName="wall")
    #wall_temp_array = np.full_like(dummy_fluxes, 275.4)
    #CFDSolver.setWallTemperature(ap, wall_temp_array, groupName="wall")
    CFDSolver(ap)
    # rst Evaluate and printan
    funcs = {}
    CFDSolver.evalFunctions(ap, funcs)
    # Print the evaluated functions
    if comm.rank == 0:
        print(funcs)
