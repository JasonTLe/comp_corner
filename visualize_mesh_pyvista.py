#!/usr/bin/env python3.11
"""
visualize_mesh_pyvista.py
==========================

Render a structured CGNS volume mesh (e.g. one produced by
make_comp_corner_mesh.py, or any of meshes/comp_corner_*.cgns) with pyvista:
one PNG of the full 3D block with the wall/symmetry surfaces shaded, and one
PNG zoomed on the corner region showing the wall-normal grid lines so mesh
density/quality near the wall can be inspected.

Usage:
    python3.11 visualize_mesh_pyvista.py meshes/comp_corner_16_fixed.cgns
"""

import os
import sys
import argparse

import numpy as np
import pyvista as pv
from cgnsutilities.cgnsutilities import readGrid

pv.OFF_SCREEN = True


def loadStructuredGrid(fileName):
    """Read the single-block structured grid directly via cgnsutilities
    (skips CGNS BC_t nodes vtkCGNSReader doesn't support, and gives us the
    coordinate array in a known (i, j, k) order for slicing)."""
    grid = readGrid(fileName)
    blk = grid.blocks[0]
    coords = blk.coords  # shape (ni, nj, nk, 3)
    ni, nj, nk = blk.dims

    sgrid = pv.StructuredGrid()
    # VTK StructuredGrid expects points ordered with i fastest, so transpose
    # to (k, j, i) before flattening in C order.
    pts = coords.transpose(2, 1, 0, 3).reshape(-1, 3)
    sgrid.points = pts
    sgrid.dimensions = [ni, nj, nk]
    return sgrid, blk


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gridFile", help="structured CGNS mesh file")
    parser.add_argument("--outDir", default=".", help="directory for output PNGs")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.gridFile))[0]
    sgrid, blk = loadStructuredGrid(args.gridFile)
    ni, nj, nk = blk.dims
    print(f"Loaded {args.gridFile}: dims (i,j,k) = ({ni}, {nj}, {nk}), {sgrid.n_points} points")

    # Both figures use the j=0 symmetry-plane slice (full i x k grid, single
    # spanwise plane) -- that is the actual 2D flow-relevant cross-section:
    # streamwise along the wall, wall-normal growth away from it.
    symPlane = sgrid.extract_subset((0, ni - 1, 0, 0, 0, nk - 1)).extract_surface()

    # ------------------------------------------------------------------
    # Figure 1: full symmetry-plane mesh
    # ------------------------------------------------------------------
    p = pv.Plotter(off_screen=True, window_size=(1600, 900))
    p.add_mesh(symPlane, style="wireframe", color="black", line_width=0.4)
    p.view_xy()
    p.add_axes()
    p.background_color = "white"
    fullPng = os.path.join(args.outDir, f"{base}_full.png")
    p.screenshot(fullPng)
    print(f"Wrote {fullPng}")

    # ------------------------------------------------------------------
    # Figure 2: zoomed corner region, near-wall layers, to show clustering.
    # The first cell (~1.6e-6 m) is ~5 orders of magnitude smaller than the
    # domain, so a true-scale zoom is still invisible -- exaggerate the
    # wall-normal (y) coordinate for this plot only, as is standard practice
    # for boundary-layer mesh QA figures.
    # ------------------------------------------------------------------
    coords = blk.coords
    xw = coords[:, 0, 0, 0]
    corner_i = int(np.argmin(np.abs(xw)))
    iLo = max(corner_i - 20, 0)
    iHi = min(corner_i + 20, ni - 1)
    kHi = min(45, nk - 1)  # near-wall layers, comparable scale to the streamwise spacing here

    corner = sgrid.extract_subset((iLo, iHi, 0, 0, 0, kHi)).extract_surface()
    EXAGGERATION = 4.0
    yWall = corner.points[:, 1].min()
    corner.points[:, 1] = yWall + (corner.points[:, 1] - yWall) * EXAGGERATION

    p2 = pv.Plotter(off_screen=True, window_size=(1600, 900))
    p2.add_mesh(corner, style="wireframe", color="black", line_width=0.6)
    p2.view_xy()
    p2.add_axes()
    p2.background_color = "white"
    p2.add_text(f"wall-normal (y) exaggerated {EXAGGERATION:.0f}x", font_size=12, color="black")
    cornerPng = os.path.join(args.outDir, f"{base}_corner_zoom.png")
    p2.screenshot(cornerPng)
    print(f"Wrote {cornerPng}")


if __name__ == "__main__":
    main()
