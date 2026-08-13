"""Add the missing Dirichlet Temperature BCDataSet to the isothermal wall
BCs in comp_corner_5.cgns. ADFlow's preprocessing (BCDataIsothermalWall)
requires the wall temperature to be present in the CGNS file itself --
AeroProblem.setBCVar() only *updates* existing BC data after init.

Also assigns family names matching what ADFlow auto-generates, so the
"does not have a family" warnings go away and setBCVar("...", "wall")
keeps working.

Usage: python3.11 fix_bc.py [inFile] [Twall]

must specify every BC
"""

import ctypes
import os
import sys
from tabulate import tabulate
import numpy as np
from cgnsutilities.cgnsutilities import (
    readGrid,
    BocoDataSet,
    BocoDataSetArray,
    CGNSDATATYPES,
)


def setSIUnits(fileName):
    """Stamp DataClass=Dimensional + DimensionalUnits=SI on every BCData_t node.

    cgnsutilities has no notion of units, so the BCDataSets it writes carry no
    DimensionalUnits_t node. ADflow then prints

        BC data set Temperature: No units specified, assuming SI units

    (readCGNSGrid.F90:readBCDataArrays). The assumption is correct -- our data
    *is* SI -- so this only silences a cosmetic warning. Done by re-opening the
    file with the CGNS mid-level library through ctypes.

    Returns the number of BCData_t nodes stamped.
    """
    lib = ctypes.CDLL("libcgns.so")  # on LD_LIBRARY_PATH via $CGNS_HOME/lib
    lib.cg_get_error.restype = ctypes.c_char_p

    CG_MODE_MODIFY, CG_OK = 2, 0
    Dimensional = 2
    Kilogram, Meter, Second, Kelvin, Radian = 2, 2, 2, 2, 3
    # BCData_t is indexed by BCDataType_t, not 1..n
    Dirichlet, Neumann = 2, 3

    def chk(ierr, what):
        if ierr != CG_OK:
            raise RuntimeError(f"{what} failed: {lib.cg_get_error().decode()}")

    def goto(fn, B, Z, BC, DS, dirNeu):
        labels = [b"Zone_t", b"ZoneBC_t", b"BC_t", b"BCDataSet_t", b"BCData_t"]
        nums = [Z, 1, BC, DS, dirNeu]
        return lib.cg_golist(
            fn,
            B,
            len(labels),
            (ctypes.c_char_p * len(labels))(*labels),
            (ctypes.c_int * len(nums))(*nums),
        )

    fnRef = ctypes.c_int()
    chk(lib.cg_open(fileName.encode(), CG_MODE_MODIFY, ctypes.byref(fnRef)), "cg_open")
    fn = fnRef.value

    nStamped = 0
    try:
        nBases = ctypes.c_int()
        chk(lib.cg_nbases(fn, ctypes.byref(nBases)), "cg_nbases")
        for B in range(1, nBases.value + 1):
            nZones = ctypes.c_int()
            chk(lib.cg_nzones(fn, B, ctypes.byref(nZones)), "cg_nzones")
            for Z in range(1, nZones.value + 1):
                nBocos = ctypes.c_int()
                chk(lib.cg_nbocos(fn, B, Z, ctypes.byref(nBocos)), "cg_nbocos")
                for BC in range(1, nBocos.value + 1):
                    DS = 1
                    # no cg_ndataset in the API -- walk until the goto misses
                    while True:
                        hit = False
                        for dirNeu in (Dirichlet, Neumann):
                            if goto(fn, B, Z, BC, DS, dirNeu) != CG_OK:
                                continue
                            hit = True
                            chk(lib.cg_dataclass_write(Dimensional), "cg_dataclass_write")
                            chk(
                                lib.cg_units_write(
                                    Kilogram, Meter, Second, Kelvin, Radian
                                ),
                                "cg_units_write",
                            )
                            nStamped += 1
                        if not hit:
                            break
                        DS += 1
    finally:
        lib.cg_close(fn)

    return nStamped


if len(sys.argv) > 1:
    inFile = sys.argv[1]  
else:
    print("="*50)
    raise ValueError("No input file provided.")
    print("="*50)

base, ext = os.path.splitext(inFile)
outFile = f"{base}_fixed{ext}"
print(f'Fixed file outputting to "{outFile}"...')
print("="*50)

if len(sys.argv) > 2:
    Twall = float(sys.argv[2]) 
else:
    Twall = None
    print("No Dirichlet temperature provided. Checking for isothermal walls...")
    print("="*50)

# family names per BC type (same names ADFlow auto-assigns)
famMap = {
    "bcaxisymmetricwedge": "asym_wedge",
    "bcfarfield": "farfield",
    "bcinflow": "inflow",
    "bcinflowsubsonic": "inflow_subsonic",
    "bcinflowssupersonic": "inflow_supersonic",
    "bcoutflow": "outflow",
    "bcoutflowsubsonic": "outflow_subsonic",
    "bcoutflowsupersonic": "outflow_supersonic",
    "bcsymmetryplane": "sym_plane",
    "bcsymmetrypolar": "sym_polar",
    "bcwall": "wall",
    "bcwallinviscid": "wall_inviscid",
    "bcwallviscous": "wall_viscous",
    "bcwallviscousheatflux": "wall_viscous_hf",
    "bcwallviscousisothermal": "wall_viscous_iso",
}

grid = readGrid(inFile)
nRenamed = 0
nFixed = 0
renameRows = []
fixedRows = []
for blk in grid.blocks:
    for boco in blk.bocos:
        bcType = boco.internalType
        if bcType in famMap:
            oldFam = getattr(boco, "family", None) or "<none>"
            renameRows.append(
                [blk.name, boco.name, bcType, f'"{oldFam}" --> "{famMap[bcType]}"']
            )
            boco.family = famMap[bcType]
            nRenamed += 1
        if bcType == "bcwallviscousisothermal":
            if Twall is None: 
                raise ValueError("****ERROR: Dirichlet temperature must be provided.*****")
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
            fixedRows.append(
                [blk.name, boco.name, bcType, f"Added Dirichlet Temperature = {Twall} K"]
            )

if renameRows:
    print(
        tabulate(
            renameRows,
            headers=["Block", "Domain", "BC Type", "Family Name"],
            tablefmt="grid",
        )
    )

if fixedRows:
    print(
        tabulate(
            fixedRows,
            headers=["Block", "Domain", "BC Type", "Note"],
            tablefmt="grid",
        )
    )

grid.writeToCGNS(outFile)
nUnits = setSIUnits(outFile)
print("="*50)
print(f"Tagged {nUnits} BCData node(s) with DataClass=Dimensional, units = SI (kg, m, s, K, rad)")
print(f"Successfully renamed {nRenamed} BC(s) and fixed {nFixed} isothermal wall BC(s) --> {outFile}")
print("="*50)