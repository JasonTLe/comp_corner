"""Add the missing Dirichlet Temperature BCDataSet to the isothermal wall
BCs in comp_corner_5.cgns. ADFlow's preprocessing (BCDataIsothermalWall)
requires the wall temperature to be present in the CGNS file itself --
AeroProblem.setBCVar() only *updates* existing BC data after init.

Also assigns family names matching what ADFlow auto-generates, so the
"does not have a family" warnings go away and setBCVar("...", "wall")
keeps working.

Usage: python fix_bc_temperature.py [inFile] [outFile] [Twall]
"""

import sys
import numpy as np
from cgnsutilities.cgnsutilities import (
    readGrid,
    BocoDataSet,
    BocoDataSetArray,
    CGNSDATATYPES,
)

inFile = sys.argv[1] if len(sys.argv) > 1 else "comp_corner_5.cgns"
outFile = sys.argv[2] if len(sys.argv) > 2 else "comp_corner_5_isoT.cgns"
Twall = float(sys.argv[3]) if len(sys.argv) > 3 else 275.4

# family names per BC type (same names ADFlow auto-assigns)
famMap = {
    "bcsymmetryplane": "sym",
    "bcfarfield": "far",
    "bcwallviscousisothermal": "wall",
    "bcinflowssupersonic": "inflow",
    "bcinflowsupersonic": "inflow",
    "bcoutflow": "outflow",
}

grid = readGrid(inFile)
nFixed = 0
for blk in grid.blocks:
    for boco in blk.bocos:
        bcType = boco.internalType
        if bcType in famMap:
            boco.family = famMap[bcType]
        if bcType == "bcwallviscousisothermal":
            ds = BocoDataSet("BCDataSet_1", "bcwallviscousisothermal")
            dataArr = np.array([Twall], dtype=np.float64)
            dataDims = np.ones(3, dtype=np.int32, order="F")
            dataDims[0] = dataArr.size
            arr = BocoDataSetArray(
                "Temperature",
                CGNSDATATYPES["RealDouble"],
                1,          # nDims
                dataDims,   # dataDims (length-3 int32, Fortran convention)
                dataArr,
            )
            ds.addDirichletDataSet(arr)
            boco.addBocoDataSet(ds)
            nFixed += 1
            print(f"  {blk.name} / {boco.name}: added Dirichlet Temperature = {Twall} K")

grid.writeToCGNS(outFile)
print(f"Fixed {nFixed} isothermal wall BC(s) -> {outFile}")
