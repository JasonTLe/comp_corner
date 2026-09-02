#!/usr/bin/env python3.11
"""
make_comp_corner_mesh.py
=========================

Generate a 25-deg compression-corner volume mesh with the MDO Lab toolchain
(pySpline/pyGeo for the wall curve, pyHyp for the hyperbolic volume
extrusion) that matches the geometry and point distribution of
meshes/comp_corner_14_fixed.cgns -- which was hand-built in Pointwise.

Reference geometry extracted from comp_corner_14_fixed.cgns:
    single block, (i, j, k) = (769, 129, 3)
    i : streamwise wall curve
        flat plate  x in [-0.125758, 0.0], 550 pts, uniform spacing
        25 deg ramp x in [0.0, 0.04],      220 pts, uniform spacing in x
        (corner point at x=0 is shared -> 550 + 220 - 1 = 769)
    j : wall-normal, N = 129, first cell height s0 = 1.6e-6 m,
        geometric growth ratio ~1.06, outer boundary y ~ 0.029-0.047 m
    k : spanwise, 3 planes at z = 0, 0.001, 0.002 m (2 cells -- ADflow
        needs >=2 cells here so multigrid coarsening has something to cut)

BCs mirror the reference mesh (see fix_bc.py / docs/DEBUG_REPORT.md):
    wall   (i=1..550 and 550..769, split into two families) -> isothermal,
           Twall = 275.4 K
    iLow   (upstream)   -> farfield
    iHigh  (downstream) -> outflow
    jHigh  (outer/top)  -> farfield
    kLow/kHigh          -> symmetry planes

Usage:
    python3.11 make_comp_corner_mesh.py
    python3.11 make_comp_corner_mesh.py --output meshes/comp_corner_16.cgns
"""

import os
import argparse
import numpy as np

from pyspline import Curve
from pyhyp import pyHyp
from cgnsutilities.cgnsutilities import readGrid, BocoDataSet, BocoDataSetArray, CGNSDATATYPES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# Part 1: wall curve (pySpline / pyGeo geometry engine)
# ----------------------------------------------------------------------
def buildWallCurve(plateLength, rampLength, angleDeg, nPlate, nRamp):
    """Two straight pySpline.Curve segments (flat plate + ramp), each
    sampled at a uniform parameter -- which for a straight, chord-length
    parametrized curve means uniform spacing along the segment, matching
    the reference mesh's point distribution exactly."""
    angle = np.radians(angleDeg)

    plateCtrl = np.array([[-plateLength, 0.0], [0.0, 0.0]])
    rampCtrl = np.array(
        [[0.0, 0.0], [rampLength, rampLength * np.tan(angle)]]
    )

    plateCurve = Curve(X=plateCtrl, k=2)
    rampCurve = Curve(X=rampCtrl, k=2)

    plateXY = plateCurve.getValue(np.linspace(0.0, 1.0, nPlate))
    rampXY = rampCurve.getValue(np.linspace(0.0, 1.0, nRamp))

    # corner point (x=0) is shared between the two segments
    xy = np.vstack([plateXY, rampXY[1:]])
    return xy[:, 0], xy[:, 1]


# ----------------------------------------------------------------------
# Part 2: PLOT3D surface (curve extruded trivially across the span)
# ----------------------------------------------------------------------
def writeSurfaceFile(fileName, x, y, zPlanes):
    ni = len(x)
    nj = len(zPlanes)
    with open(fileName, "w") as f:
        f.write("1\n")
        f.write(f"{ni} {nj} 1\n")
        for iDim in range(3):
            for j in range(nj):
                for i in range(ni):
                    if iDim == 0:
                        f.write(f"{x[i]:.10e}\n")
                    elif iDim == 1:
                        f.write(f"{y[i]:.10e}\n")
                    else:
                        f.write(f"{zPlanes[j]:.10e}\n")


# ----------------------------------------------------------------------
# Part 3: pyHyp hyperbolic extrusion
# ----------------------------------------------------------------------
def extrudeVolume(surfaceFile, volumeFile, N, s0, marchDist):
    options = {
        # Input
        "inputFile": surfaceFile,
        "unattachedEdgesAreSymmetry": False,
        "autoConnect": True,
        "families": "wall",
        # curve endpoints (inflow/outflow) are free; the two duplicated
        # spanwise edges are a symmetry plane
        "BC": {1: {"iLow": "splay", "iHigh": "splay", "jLow": "zSymm", "jHigh": "zSymm"}},
        "outerFaceBC": "farfield",
        # Grid parameters, matched to comp_corner_14_fixed.cgns
        "N": N,
        "s0": s0,
        "marchDist": marchDist,
        "splay": 0.3,
        # Pseudo grid parameters
        "ps0": -1.0,
        "pGridRatio": -1.0,
        "cMax": 6.0,
        # Smoothing parameters
        "epsE": 1.0,
        "epsI": 2.0,
        "theta": 3.0,
        "volCoef": 0.25,
        "volBlend": 0.0001,
        "volSmoothIter": 100,
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(volumeFile)
    return hyp


# ----------------------------------------------------------------------
# Part 4: BCs -- mirror comp_corner_14_fixed.cgns (see fix_bc.py)
# ----------------------------------------------------------------------
def setBCs(volumeFile, fixedFile, nPlate, Twall):
    grid = readGrid(volumeFile)
    blk = grid.blocks[0]
    # pyHyp's own axis convention: i = streamwise (curve), j = spanwise
    # (the duplicated input curve, size 3), k = wall-normal (marched, N)
    ni, nj, nk = blk.dims

    blk.bocos = []

    def addBoco(name, internalType, family, ptRange):
        from cgnsutilities.cgnsutilities import Boco

        boco = Boco(name, internalType, np.array(ptRange, dtype="intc"), family)
        blk.bocos.append(boco)

    # wall (physical surface, k=1), split into plate / ramp families like
    # the reference mesh's BC5 / BC6
    addBoco("BC1", "bcwallviscousisothermal", "wall_plate", [[1, nPlate], [1, nj], [1, 1]])
    addBoco("BC2", "bcwallviscousisothermal", "wall_ramp", [[nPlate, ni], [1, nj], [1, 1]])
    # upstream / downstream
    addBoco("BC3", "bcfarfield", "farfield", [[1, 1], [1, nj], [1, nk]])
    addBoco("BC4", "bcoutflow", "outflow", [[ni, ni], [1, nj], [1, nk]])
    # outer boundary (k=nk, the marched/outer face)
    addBoco("BC5", "bcfarfield", "farfield", [[1, ni], [1, nj], [nk, nk]])
    # symmetry planes (span, j=1 and j=nj)
    addBoco("BC6", "bcsymmetryplane", "sym_plane", [[1, ni], [1, 1], [1, nk]])
    addBoco("BC7", "bcsymmetryplane", "sym_plane", [[1, ni], [nj, nj], [1, nk]])

    for boco in blk.bocos:
        if boco.internalType == "bcwallviscousisothermal":
            ds = BocoDataSet("BCDataSet_1", "bcwallviscousisothermal")
            dataArr = np.array([Twall], dtype=np.float64)
            dataDims = np.ones(3, dtype=np.int32, order="F")
            dataDims[0] = dataArr.size
            arr = BocoDataSetArray(
                "Temperature", CGNSDATATYPES["RealDouble"], 1, dataDims, dataArr
            )
            ds.addDirichletDataSet(arr)
            boco.addBocoDataSet(ds)

    grid.writeToCGNS(fixedFile)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plateLength", type=float, default=0.125758)
    parser.add_argument("--rampLength", type=float, default=0.04)
    parser.add_argument("--angleDeg", type=float, default=25.0)
    parser.add_argument("--nPlate", type=int, default=550)
    parser.add_argument("--nRamp", type=int, default=220)
    parser.add_argument("--N", type=int, default=129, help="wall-normal points")
    parser.add_argument("--s0", type=float, default=1.6e-6, help="first cell height [m]")
    parser.add_argument("--marchDist", type=float, default=0.047, help="wall-normal march distance [m]")
    parser.add_argument("--spanWidth", type=float, default=0.002, help="total span [m]")
    parser.add_argument("--Twall", type=float, default=275.4)
    parser.add_argument("--output", type=str, default="meshes/comp_corner_16.cgns")
    args = parser.parse_args()

    surfaceFile = os.path.join(BASE_DIR, "meshes", "comp_corner_16_surf.fmt")
    volumeFile = os.path.join(BASE_DIR, args.output)
    base, ext = os.path.splitext(volumeFile)
    fixedFile = f"{base}_fixed{ext}"

    x, y = buildWallCurve(args.plateLength, args.rampLength, args.angleDeg, args.nPlate, args.nRamp)
    # span order (not curve order) sets which way pyHyp's cross-product normal
    # points; descending z makes the march go +y (away from the wall) here.
    zPlanes = np.array([args.spanWidth, args.spanWidth / 2, 0.0])
    writeSurfaceFile(surfaceFile, x, y, zPlanes)
    print(f"Wrote wall curve: {surfaceFile}  ({len(x)} streamwise pts)")

    hyp = extrudeVolume(surfaceFile, volumeFile, args.N, args.s0, args.marchDist)
    print(f"Wrote raw volume mesh: {volumeFile}")

    setBCs(volumeFile, fixedFile, args.nPlate, args.Twall)
    print(f"Wrote BC-tagged mesh: {fixedFile}")


if __name__ == "__main__":
    main()
