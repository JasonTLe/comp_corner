#!/usr/bin/env python3.11
"""
mesh.py
=======

Generate a 25-deg compression-corner volume mesh with the MDO Lab toolchain
(pySpline/pyGeo for the wall curve, pyHyp for the hyperbolic volume
extrusion).  The viscous part of the geometry and the point distribution come
from meshes/comp_corner_14_fixed.cgns, which was hand-built in Pointwise.

THIS COPY IS THE highRe CASE (M = 2.88, Re_delta = 132 840, delta = 4.1 mm at
8.04 delta upstream of the corner -- see flow_conditions.py).  It is low_RE/
mesh.py with every length that carries a boundary-layer scale multiplied by

    k = delta_highRe / delta_lowRe = 4.10 / 2.27 = 1.8062

and the point counts adjusted to hold the same cells-per-delta, so the two
grids resolve their own boundary layers identically and a difference between
the two solutions is a Reynolds-number difference and not a mesh difference.
The code below is UNCHANGED from low_RE/mesh.py; only the argparse defaults
and the notes on them differ.  Summary of what moved:

    option            lowRe        highRe    why
    --invFrontLength  0.05         0.09      k, holds the same stretching ratio
    --plateLength     0.1759264    0.321360  recalibrated (3 solves), see its note
    --rampLength      0.04         0.04      NOT scaled: the paper plots both
                                             cases over the same x/L window
                                             (fig. 2 reaches x/L = +30 highRe)
    --invBackLength   0.02         0.036     k
    --nPlate          808          874       ds 218 -> 368 um, ~11.1 cells/delta
    --nRamp           220          122       ds 202 -> 365 um, ~11.2 cells/delta
    --nInvFront       109          109       length/ds is unchanged, so is the
    --nInvBack        51           51        stretching each has to do
    --N               145          157       12 more layers to span the taller
                                             march at the same outer ratio
    --s0              8.0e-7       8.0e-7    NOT scaled -- y+ is set by s0 and
                                             would rise with it; see the note
    --marchDist       0.0285977    0.0516    k
    --wnFineHeight    0.0035       0.0063    k, still ~1.54 delta
    --output          ..._20.cgns  ..._21.cgns

meshes/comp_corner_20*.cgns in this directory are STALE COPIES of the lowRe
grid left over from the repository reorganisation.  They are calibrated to
delta = 2.27 mm and must not be used here; comp_corner_21 is this case's grid.

The wall, upstream to downstream, is four segments with three adjustable
lengths (--invFrontLength, --plateLength, --invBackLength; the ramp has its
own --rampLength):

    --invFrontLength   inviscid slip wall, horizontal, ahead of the plate.
                       Coarsens going upstream.  Gives the freestream somewhere
                       to enter without a boundary layer, so the plate's
                       leading edge is a slip/no-slip junction inside the
                       domain rather than a corner of it.
    --plateLength      viscous flat plate, horizontal, uniform spacing.
    --rampLength       viscous 25 deg ramp, uniform spacing.
    --invBackLength    inviscid slip wall, horizontal again, past the ramp.
                       Coarsens going downstream.  Turning the wall back to
                       horizontal here is a 25 deg *expansion*, so expect a
                       Prandtl-Meyer fan off that convex corner.

Both inviscid segments pick up the spacing of the viscous segment they join,
so cell size is continuous across each junction and only the BC changes.
Setting either inviscid length to 0 drops that segment; with both at 0 the
geometry is comp_corner_14's exactly.

Reference distribution, from comp_corner_14_fixed.cgns (the lowRe ancestor
of this geometry -- kept verbatim because it is what the code reproduces):
    i : flat plate  x in [-0.125758, 0.0], 550 pts, uniform
        25 deg ramp x in [0.0, 0.04],      220 pts, uniform in x
        (junction nodes are shared, so ni = sum(counts) - #junctions)
    j : wall-normal, N = 129, first cell height s0 = 1.6e-6 m, marched
        0.0285977 m at a geometric growth ratio ~1.055
    k : spanwise, 3 planes at z = 0, 0.001, 0.002 m (2 cells -- ADflow
        needs >=2 cells here so multigrid coarsening has something to cut)

Note the axis order: pyHyp always marches along its *last* index, so its raw
output is (streamwise, spanwise, wall-normal).  setBCs() swaps the last two
axes back to the reference's convention before writing.

The grid lines leave the wall orthogonal to it: the pyHyp smoothing terms are
ramped up layer by layer from zero rather than held constant, so a corner's
turn is absorbed out in the boundary layer instead of in the first cell.  See
the table in extrudeVolume() for what that costs elsewhere.  The only nodes
that cannot be orthogonal are the two kinks themselves, whose grid lines
bisect the turn -- the same as in comp_corner_14.

Labelling the BCs
-----------------
Two labels sit on every CGNS boundary, and only one of them survives:

    BC name   the CGNS BC_t node's own name.  Do not rely on it -- cgnsutilities
              discards whatever you pass and writes BC1, BC2, ... in the order
              the bocos were added.  The names in setBCs() below are for
              reading the code and the summary table, nothing more.
    family    a FamilyName_t node under the BC_t.  This one is written through
              verbatim, and it is what ADflow's groupName / addFamilyGroup /
              setBCVar arguments take, and what the force and heat-transfer
              breakdowns get reported per.

So the family is the label.  setBCs() gives every face its own, one per face
rather than one per BC type:

    wall_inv_front   bcwallinviscid            i 1 .. nInvFront
    wall_plate       bcwallviscousisothermal   the flat plate
    wall_ramp        bcwallviscousisothermal   the 25 deg ramp
    wall_inv_back    bcwallinviscid            past the ramp
    inflow           bcfarfield                iLow
    outflow          bcfarfield                iHigh
    outer            bcfarfield                jHigh
    sym_low          bcsymmetryplane           kLow
    sym_high         bcsymmetryplane           kHigh

which is what makes it possible to, say, hold the ramp at a different Twall
than the plate, or integrate the heating on the ramp alone:

    CFDSolver.setBCVar("Temperature", 300.0, "wall_ramp")
    CFDSolver.addFamilyGroup("viscous_wall", ["wall_plate", "wall_ramp"])

fix_bc.py used to overwrite the family from a table keyed on BC type, which
collapsed wall_plate and wall_ramp into one "wall_viscous_iso" and inflow,
outflow and outer into one "farfield".  It now only names a family that is
still "default", so a Pointwise grid like comp_corner_14 still gets ADflow's
auto-generated names and these stay as written.

Inflow and outflow are both bcfarfield, as requested: at Mach 2.88 (the
free-stream both adflow_run1.py and adflow_run2.py set) the upstream face is
supersonic inflow and the downstream one supersonic outflow, and
ADflow's farfield picks which from the local Riemann invariants rather than
needing to be told.  They keep separate families, so they are still separable
in post-processing.

The Dirichlet wall temperature, the SI unit nodes and the ADF encoding
ADflow's libcgns needs are applied by that same fix_bc.py pass -- the step that
produced comp_corner_14_fixed.cgns.

Usage:
    python3.11 mesh.py
    python3.11 mesh.py --invFrontLength 0.08 --plateLength 0.1 --invBackLength 0.03
    python3.11 mesh.py --invFrontLength 0 --invBackLength 0   # comp_corner_14 geometry
    python3.11 mesh.py --output meshes/comp_corner_20.cgns

writes <output> and, via fix_bc.py, <output stem>_fixed.cgns
"""

import os
import sys
import argparse
import subprocess
import numpy as np
from tabulate import tabulate

from pyspline import Curve
from pyhyp import pyHyp
from cgnsutilities.cgnsutilities import readGrid, Block, Boco

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIX_BC = os.path.join(BASE_DIR, "fix_bc.py")


# ----------------------------------------------------------------------
# Part 1: wall curve (pySpline / pyGeo geometry engine)
# ----------------------------------------------------------------------
def solveGrowthRatio(length, ds0, nInt, what):
    """Geometric growth ratio r for which nInt intervals starting at ds0 span
    exactly `length`:  ds0 * (r**nInt - 1) / (r - 1) == length.

    The sum is monotone in r, so plain bisection does it.  The bracket is
    deliberately narrow: a segment that needs a ratio outside [0.5, 2.0] is
    not a mesh anyone wants, and failing loudly with the name of the knob to
    turn beats handing back a grid with a 2x jump between neighbouring cells.
    """
    target = length / ds0

    def sumGeom(r):
        return float(nInt) if abs(r - 1.0) < 1e-14 else (r ** nInt - 1.0) / (r - 1.0)

    if abs(target - nInt) < 1e-9 * nInt:
        return 1.0

    lo, hi = 0.5, 2.0
    if sumGeom(hi) < target:
        raise SystemExit(
            f"{what}: {nInt + 1} points cannot span {length:g} m from a first cell of "
            f"{ds0:.3e} m without a growth ratio above {hi} -- add points or shorten it."
        )
    if sumGeom(lo) > target:
        raise SystemExit(
            f"{what}: {nInt + 1} points over {length:g} m from a first cell of "
            f"{ds0:.3e} m would have to shrink faster than {lo} per cell -- remove "
            f"points or lengthen it."
        )
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if sumGeom(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stretchedParams(length, ds0, nPts, what):
    """Curve parameters (0 -> 1) for nPts points spanning `length`, the first
    interval ds0 and every one after it a fixed ratio larger.  A straight
    pySpline curve is chord-length parametrized, so normalized arc length and
    curve parameter are the same thing here.  Returns (u, ratio)."""
    r = solveGrowthRatio(length, ds0, nPts - 1, what)
    u = np.r_[0.0, np.cumsum(ds0 * r ** np.arange(nPts - 1))]
    return u / u[-1], r


def refinedParams(length, nPts, ratio, factor):
    """Curve parameters (0 -> 1) for a straight segment that is geometrically
    REFINED toward its start and uniform over the rest.  Returns
    (u, ds_first, ds_uniform).

    This is the mirror image of stretchedParams: that one starts at a spacing
    it is told and coarsens away from a junction, this one starts `factor`
    times finer than its own uniform spacing and grows back up to it.

    It exists for the two slip/no-slip junctions.  Across each of them the wall
    shear jumps from zero to its full plate value in a single cell -- an
    inviscid wall carries no boundary layer -- and a uniform viscous segment
    meets that jump with a cell ~280x wider than it is tall.  The first-cell y+
    spikes there (0.95 against 0.20 along the rest of the plate, and 0.98 at
    the ramp/slip junction) because the cell straddles the whole rise instead
    of resolving it.

    The refinement is laid out backwards from the uniform end so the two pieces
    join at exactly one spacing, with no jump to smooth over:

        k       intervals of geometric growth, r^k ~ factor
        ds_end  the uniform spacing, solved so the whole segment still spans
                `length` in exactly nPts-1 intervals
        ds[i]   ds_end / r^(k-i)   for i < k,   ds_end after that

    factor = 1 gives k = 0 and reproduces a plain uniform distribution
    exactly, which is what --leRefine 1 is for.
    """
    n = nPts - 1
    k = 0 if factor <= 1.0 else int(round(np.log(factor)/np.log(ratio)))
    k = max(0, min(k, n - 1))
    # geometric part measured in units of ds_end: sum_{i=1..k} r^-i
    g = (ratio ** k - 1.0)/(ratio ** k * (ratio - 1.0)) if k else 0.0
    dsEnd = length/(g + (n - k))
    ds = np.r_[dsEnd/ratio ** np.arange(k, 0, -1), np.full(n - k, dsEnd)]
    u = np.r_[0.0, np.cumsum(ds)]
    return u/u[-1], float(ds[0]), float(dsEnd)


def buildWallCurve(invFrontLength, plateLength, rampLength, invBackLength,
                   angleDeg, nInvFront, nPlate, nRamp, nInvBack,
                   leRefine=1.0, leRatio=1.05):
    """The wall as up to four straight pySpline.Curve segments laid end to end,
    with the (name, BC type, first i, last i) label of each.  Running
    downstream:

        wall_inv_front  inviscid slip wall, horizontal, ahead of the plate.
                        Coarsens going upstream -- its spacing matches the
                        plate's at the junction and grows geometrically from
                        there, so the far-upstream cells are cheap.
        wall_plate      viscous flat plate, horizontal, uniform spacing.  This
                        is where the boundary layer lives.
        wall_ramp       viscous 25 deg ramp, uniform spacing.
        wall_inv_back   inviscid slip wall, horizontal again, past the ramp.
                        Same idea as the leading one, coarsening downstream.

    The two inviscid segments start at the spacing of the viscous segment they
    join, so there is no jump in cell size across either junction -- only a
    jump in the wall BC.

    Point counts are per segment and the shared node at each junction belongs
    to both, so ni = nInvFront + nPlate + nRamp + nInvBack - 3.  Setting a
    length (or a count below 2) to zero drops that segment and its BC
    entirely; zeroing both inviscid lengths reproduces comp_corner_14's
    geometry exactly.

    Note that only the wall is split up here.  The wall-normal distribution is
    the same at every i -- one structured block cannot cluster to s0 = 1.6e-6 m
    under the plate and not under the slip walls -- so the inviscid stretches
    carry the full 129-point boundary-layer resolution whether they need it or
    not.  That is the price of keeping this a single block, and it is why the
    inviscid segments coarsen in i as fast as they reasonably can.
    """
    angle = np.radians(angleDeg)

    def sample(p0, p1, u):
        return Curve(X=np.array([p0, p1]), k=2).getValue(u)

    plateStart = (-plateLength, 0.0)
    corner = (0.0, 0.0)
    rampEnd = (rampLength, rampLength * np.tan(angle))

    # The viscous segments are uniform except for a refined run of cells at
    # whichever end meets an inviscid wall, and it is that refined spacing --
    # not the uniform one -- that the inviscid segments have to match at their
    # junctions.  The plate is refined at its leading edge (upstream end), the
    # ramp at its crest (downstream end); the compression corner between them
    # is viscous on both sides, so it gets no refinement and none is wanted.
    uPlate, dsPlateLE, dsPlate = refinedParams(plateLength, nPlate, leRatio, leRefine)
    rampArc = np.hypot(rampEnd[0] - corner[0], rampEnd[1] - corner[1])
    uRampRev, dsRampEnd, dsRamp = refinedParams(rampArc, nRamp, leRatio, leRefine)
    uRamp = 1.0 - uRampRev[::-1]        # mirror: fine at the crest, not the corner

    pieces = []  # (xy, name, bcType)
    ratios = {}

    if invFrontLength > 0.0 and nInvFront > 1:
        # sampled from the plate junction *backwards*, so the fine end is the
        # one that has to match dsPlate, then flipped into ascending x
        u, ratios["wall_inv_front"] = stretchedParams(
            invFrontLength, dsPlateLE, nInvFront, "--invFrontLength/--nInvFront")
        back = sample(plateStart, (plateStart[0] - invFrontLength, 0.0), u)
        pieces.append((back[::-1], "wall_inv_front", "bcwallinviscid"))

    pieces.append((sample(plateStart, corner, uPlate),
                   "wall_plate", "bcwallviscousisothermal"))
    pieces.append((sample(corner, rampEnd, uRamp),
                   "wall_ramp", "bcwallviscousisothermal"))

    if invBackLength > 0.0 and nInvBack > 1:
        u, ratios["wall_inv_back"] = stretchedParams(
            invBackLength, dsRampEnd, nInvBack, "--invBackLength/--nInvBack")
        pieces.append((sample(rampEnd, (rampEnd[0] + invBackLength, rampEnd[1]), u),
                       "wall_inv_back", "bcwallinviscid"))

    xy = pieces[0][0]
    segments = [(pieces[0][1], pieces[0][2], 1, len(xy))]
    for part, name, bcType in pieces[1:]:
        iShared = len(xy)  # 1-based index of the node the two segments share
        xy = np.vstack([xy, part[1:]])
        segments.append((name, bcType, iShared, len(xy)))

    spacings = dict(dsPlateLE=dsPlateLE, dsPlate=dsPlate,
                    dsRampEnd=dsRampEnd, dsRamp=dsRamp)
    return xy[:, 0], xy[:, 1], segments, ratios, spacings


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
def achievedMarch(volumeFile):
    """Wall-to-outer distance actually produced, measured along the inflow grid
    line.  pyHyp's raw index order is (streamwise, spanwise, wall-normal)."""
    c = readGrid(volumeFile).blocks[0].coords
    return float(np.linalg.norm(c[0, 0, -1] - c[0, 0, 0]))


def marchRatios(N, s0, marchDist, r0, yFine, nBlend):
    """Per-layer wall-normal growth ratios for pyHyp's `growthRatios` option.

    pyHyp's own N/s0/marchDist route spends one ratio on the whole march: it
    solves s0*(r**(N-1) - 1)/(r - 1) == marchDist and uses that r from the wall
    to the farfield.  That is a bad bargain here, because the two ends of the
    march want opposite things.  Below the boundary layer the ratio is a
    resolution knob -- 1.048 is what N = 161 was chosen to buy, against the
    1.056 the mesh had before it.  Above it, the cells are freestream: the
    outer 20% of the march is 90% of its height, nothing in it but the
    departing junction shock and the corner shock on their way to a farfield
    boundary, and holding 1.048 up there costs points at 1.048-per-cell
    granularity to resolve nothing.

    So run two ratios and let pyHyp interpolate between them:

        r0      from the wall out to `yFine`, i.e. the resolution ratio
        ramp    a linear blend over `nBlend` layers, so there is no jump in
                cell size where the two meet
        r1      constant after that, solved by bisection so the march still
                spans exactly `marchDist`

    yFine is a height, not a layer count, so it stays meaningful when s0 or N
    move: on this highRe grid 6.3 mm is ~1.54x the 4.1 mm boundary layer at the
    reference station, which keeps the layer itself, the separation bubble and
    the near-wall half of the interaction inside the fine region.  At N = 157
    that is 127 of the 156 layers at r0, and r1 comes out ~1.110 -- the same
    outer ratio the lowRe grid ran at (3.5 mm / N = 145 / 114 of 144 / 1.108).

    Returns (ratios, nFine, r1).  `ratios` is the length N-1 list pyHyp wants:
    _expandPerLayerOption writes it to growthRatios[1:] and _computeDeltaS then
    reads growthRatios[2:], so the leading entry is a placeholder and never
    used -- ratios[k] is the step from layer k to layer k+1.
    """
    if yFine <= 0.0:
        # single ratio the whole way, i.e. what pyHyp does on its own
        nFine, nBlend = 0, 0

    def heights(r1):
        # nFine is located on the provisional all-r0 march, which is exact
        # below yFine because that part of the march *is* all r0
        ds = s0 * r0 ** np.arange(N - 1)
        nF = 0 if yFine <= 0 else min(
            int(np.searchsorted(np.cumsum(ds), yFine)) + 1, N - 1)
        step = np.arange(N) - nF
        blend = np.clip(step if nBlend == 0 else step / float(nBlend), 0.0, 1.0)
        r = r0 + (r1 - r0) * blend
        ds = s0 * np.r_[1.0, np.cumprod(r[2:])]
        return ds, r, nF

    # bracket from just above 1, not from r0: with --wnFineHeight 0 the whole
    # march is the "outer" ratio and the answer is legitimately below r0
    lo, hi = 1.0 + 1e-12, 2.0
    if heights(hi)[0].sum() < marchDist:
        raise SystemExit(
            f"--N {N} cannot span {marchDist:g} m from s0 = {s0:.3e} m holding "
            f"{r0} out to {yFine:g} m -- add points, lower --wnRatio, or pull "
            f"--wnFineHeight in.")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if heights(mid)[0].sum() < marchDist:
            lo = mid
        else:
            hi = mid
    r1 = 0.5 * (lo + hi)
    ds, r, nFine = heights(r1)
    if nFine and r1 < r0:
        raise SystemExit(
            f"--N {N} overshoots: holding {r0} out to {yFine:g} m already spans more "
            f"than {marchDist:g} m, so the outer layers would have to *refine* "
            f"({r1:.4f}).  Drop --N or pull --wnFineHeight in.")
    return np.r_[1.0, r[2:]], nFine, r1


def runPyHyp(surfaceFile, volumeFile, N, s0, marchDist, volBlend=1e-4,
             wnRatio=1.048, wnFineHeight=0.0063, wnBlend=10):
    ratios, nFine, r1 = marchRatios(N, s0, marchDist, wnRatio, wnFineHeight, wnBlend)
    print(f"  wall-normal: {wnRatio:.4f} for {nFine} layers (to {wnFineHeight*1e3:.1f} mm), "
          f"blended over {wnBlend} to {r1:.4f} for the remaining "
          f"{max(N - 1 - nFine - wnBlend, 0)}")

    options = {
        # Input
        "inputFile": surfaceFile,
        "unattachedEdgesAreSymmetry": False,
        "autoConnect": True,
        "families": "wall",
        # The end faces are square to the wall: the inflow face is the plane
        # x = -(invFrontLength + plateLength), and the outflow face is vertical
        # because the wall it stands on is horizontal by then.  "xConst" pins
        # the former exactly; the latter is what an unsplayed march produces on
        # its own, so iHigh stays free with splay switched off.  Splaying (the
        # old 0.3) bowed both faces outward -- inflow reached back to
        # x = -0.154 -- which is the single biggest reason the extrusion did
        # not reproduce comp_corner_14.  The two duplicated spanwise edges are
        # a symmetry plane.
        "BC": {1: {"iLow": "xConst", "iHigh": "splay", "jLow": "zSymm", "jHigh": "zSymm"}},
        "outerFaceBC": "farfield",
        # Grid parameters.  marchDist is listed for the record only: setting
        # growthRatios makes pyHyp derive the march from the ratios instead
        # (_determineMarchingParameters), which is exactly why marchRatios
        # solves its outer ratio against the number below.
        "N": N,
        "s0": s0,
        "marchDist": marchDist,
        # a plain list, not an ndarray: baseclasses type-checks this one
        "growthRatios": ratios.tolist(),
        "splay": 0.0,
        # Pseudo grid parameters
        "ps0": -1.0,
        "pGridRatio": -1.0,
        "cMax": 6.0,
        # Smoothing parameters, ramped layer by layer (pyHyp takes these as
        # [[fraction of N, value], ...] and interpolates linearly between the
        # points).  This is what keeps the grid orthogonal at the wall.
        #
        # The march has to give *something* up at each kink in the wall: at the
        # 25 deg compression corner the wall turns into the fluid, so normals
        # off the plate and off the ramp converge above it and an
        # everywhere-normal march would fold cells over.  Explicit dissipation
        # is what stops the fold -- but a constant dissipation applies it at the
        # very first layer too, and that is precisely where it costs wall
        # orthogonality.  Holding it at zero for the first ~2% of layers and
        # only then ramping up to pyHyp's default strength puts the turn where
        # there is room for it (the normals do not actually cross until j ~ 60)
        # and leaves the first layer square to the wall.
        #
        # Measured on the k=0 plane: wall skew is the worst deviation from
        # 90 deg between a j-line and the wall segment it stands on, with the
        # kink nodes excluded (their j-lines bisect the turn, 12.5 deg off each
        # side, in comp_corner_14 too); cell skew is the standard equiangle
        # measure; area ratio is the worst ratio between neighbouring cells.
        #
        #   schedule                        | wall skew  cell skew  area ratio
        #   epsE 1.0 const (pyHyp default)  |  6.39 deg   13.3 deg     1.135
        #   epsE 0.3 const                  |  4.33 deg   13.3 deg     1.328
        #   0 -> 1.0 over 2%..65% of N      |  0.001 deg  13.3 deg     1.201
        #   0 -> 1.0 over 2%..12% of N      |  0.001 deg  13.3 deg     1.143  <- here
        #   comp_corner_14 (Pointwise)      |  0.017 deg  12.9 deg     1.135
        #
        # Note what does *not* move: 13.3 deg of cell skew is there in every
        # row, and in the Pointwise mesh too.  It is the two cells either side
        # of each kink, whose j-line is the bisector -- geometry, not
        # smoothing, and nothing in this table touches it.  What the ramp end
        # fraction does control is the area ratio, and 12% is where that is
        # best; past ~35% it drifts up to 1.23 for no gain anywhere else.
        "epsE": [[0.0, 0.0], [0.02, 0.0], [0.12, 1.0], [1.0, 1.0]],
        "epsI": [[0.0, 0.0], [0.02, 0.0], [0.12, 2.0], [1.0, 2.0]],
        "theta": [[0.0, 0.0], [0.02, 0.0], [0.12, 3.0], [1.0, 3.0]],
        "volCoef": 0.25,
        # volBlend pulls each layer's cell volumes toward their local average,
        # and it must stay small.  Raising it is tempting: a hyperbolic march
        # converges grid lines over the concave corner, and at 1e-4 the outer
        # boundary is crowded to 0.34x the local wall spacing there (0.01 takes
        # that to 0.64x, against comp_corner_14's 0.76x).
        #
        # But equalizing *volumes* means ds_j has to compensate for whatever
        # ds_i is doing, and ds_i is not uniform here -- the inviscid run-in is
        # deliberately stretched 4x coarser going upstream.  So the march
        # distance ends up tracking the inverse of the streamwise spacing, and
        # the outer boundary sags over exactly the part of the wall that is
        # coarsest.  Measured over the flat wall alone (i = 1 .. corner, all of
        # it one straight horizontal line, so the height should not vary at
        # all):
        #
        #   volBlend   height spread over the flat wall   corner crowding
        #    0.0001                 8.5%                      0.34x
        #    0.001                 13.3%                      0.42x
        #    0.003                 22.4%                      0.52x
        #    0.01                  46.0%                      0.64x
        #
        # 46% is indefensible -- the outer boundary is a farfield, and how far
        # it stands off the wall is a modelling choice, not something that
        # should follow the local cell size.  The crowding it buys back is
        # cosmetic by comparison: those cells are three orders of magnitude
        # larger than the near-wall ones, so they never touch the time step.
        # Stay at pyHyp's default.  (The 8.5% that remains at 1e-4 is almost
        # all the corner approach lifting the outer boundary, which the
        # Pointwise mesh does too; between the far upstream and the plate the
        # height varies 0.9%.)
        "volBlend": volBlend,
        "volSmoothIter": [[0.0, 0], [0.02, 0], [0.12, 100], [1.0, 100]],
    }

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(volumeFile)
    return hyp


def extrudeVolume(surfaceFile, volumeFile, N, s0, marchDist, volBlend=1e-4,
                  wnRatio=1.048, wnFineHeight=0.0063, wnBlend=10, tol=1e-3, maxIter=5):
    """Extrude to a wall-to-outer distance of `marchDist`, for real.

    pyHyp's own marchDist is a request, not a result, and volBlend (see above)
    makes it a badly underdelivered one.  With growthRatios set it is not even
    read -- marchRatios is what enforces the distance, by solving its outer
    ratio against it -- so the request scaled here is the target handed to
    marchRatios on each pass.  Since the outer boundary's height is a physical
    modelling choice -- it decides whether the corner shock reaches the
    farfield -- it should not quietly depend on a smoothing coefficient.
    So scale the request until the delivered distance matches.  Each pass is
    about a second and it converges in one or two.
    """
    request = marchDist
    for it in range(maxIter):
        hyp = runPyHyp(surfaceFile, volumeFile, N, s0, request, volBlend,
                       wnRatio, wnFineHeight, wnBlend)
        got = achievedMarch(volumeFile)
        if abs(got - marchDist) <= tol * marchDist:
            break
        request *= marchDist / got
    print(f"March calibration: asked pyHyp for {request:.6f} m, "
          f"got {got:.6f} m against a target of {marchDist:.6f} m ({it + 1} pass(es))")
    return hyp


# ----------------------------------------------------------------------
# Part 4: index convention + BCs -- mirror comp_corner_14.cgns
# ----------------------------------------------------------------------
def setBCs(volumeFile, wallSegments):
    """Rewrite the pyHyp volume in the reference mesh's index convention and
    tag its faces with the reference's BC types and point ranges.

    pyHyp hands back (i, j, k) = (streamwise, spanwise, wall-normal): the
    marched direction is always last, and the span is whatever the input
    PLOT3D surface's second index was.  comp_corner_14.cgns is instead
    (streamwise, wall-normal, spanwise), so the last two axes are swapped
    here.  The span is also reversed, for two reasons: writeSurfaceFile lays
    the planes down in descending z (that is what points pyHyp's march at +y,
    away from the wall), whereas the reference runs z = 0 -> 0.002; and
    swapping two axes on its own would leave the block left-handed.

    `wallSegments` is what buildWallCurve returned: (family, BC type, first i,
    last i) for each stretch of wall.  Each becomes one boco on jLow; the
    remaining five faces are added whole below.

    Families are set here rather than left to fix_bc.py, because they are the
    only label that survives -- see "Labelling the BCs" at the top of the file.
    The Dirichlet wall temperature, SI units on the BC data and the ADF
    encoding are still fix_bc.py's job, exactly as they were for
    comp_corner_14 -> comp_corner_14_fixed.
    """
    grid = readGrid(volumeFile)
    old = grid.blocks[0]

    coords = old.coords.transpose(0, 2, 1, 3)[:, :, ::-1, :].copy()
    ni, nj, nk = coords.shape[:3]

    # Re-flatten the span.  The case is 2.5D -- every spanwise plane is meant
    # to be the same x-y grid at a constant z -- but pyHyp smooths in all three
    # directions, so it hands back planes that wander by ~1e-10 m in z and
    # ~1e-10 m in x-y off the k=0 plane.  That is 1e-4 of the first cell height,
    # harmless in itself, but it makes the k faces something other than exact
    # constant-z symmetry planes and gives the solver a spanwise gradient to
    # chew on that is not physically there.  The reference mesh's planes are
    # exact, so match it: copy the k=0 plane across and pin z per plane.
    coords[:, :, :, :2] = coords[:, :, :1, :2]
    coords[:, :, :, 2] = np.round(coords[:, :, :, 2].mean(axis=(0, 1)), 12)
    coords = np.asfortranarray(coords)
    blk = Block(old.name, np.array([ni, nj, nk], dtype="intc"), coords)
    grid.blocks = [blk]

    # The wall is jLow, split in i into the segments buildWallCurve laid out;
    # everything else is a whole face.  Ranges are 1-based and inclusive on
    # both ends, and neighbouring wall segments share their junction node --
    # that is the CGNS convention Pointwise exported comp_corner_14 with.
    faces = [(fam, bcType, [[i1, i2], [1, 1], [1, nk]])
             for fam, bcType, i1, i2 in wallSegments]
    faces += [
        # inflow and outflow are both farfield: at Mach 2.88 the upstream face
        # is supersonic inflow and the downstream one is supersonic outflow, and
        # ADflow's farfield BC picks which it is from the local Riemann
        # invariants rather than needing to be told.  Naming them separately
        # still lets them be told apart in post-processing.
        ("inflow", "bcfarfield", [[1, 1], [1, nj], [1, nk]]),
        ("outflow", "bcfarfield", [[ni, ni], [1, nj], [1, nk]]),
        ("outer", "bcfarfield", [[1, ni], [nj, nj], [1, nk]]),
        ("sym_low", "bcsymmetryplane", [[1, ni], [1, nj], [1, 1]]),
        ("sym_high", "bcsymmetryplane", [[1, ni], [1, nj], [nk, nk]]),
    ]
    for fam, bcType, ptRange in faces:
        boco = Boco(fam, bcType, np.array(ptRange, dtype="intc"), None)
        # The family is the label that matters.  cgnsutilities discards the
        # BC_t *name* on write -- everything comes back as BC1..BCn in the
        # order added -- so a family is the only way to address one of these
        # faces from ADflow afterwards.  Set one per face, not one per BC type,
        # or wall_plate and wall_ramp become the same thing.  fix_bc.py leaves
        # an explicitly named family alone.
        boco.family = fam
        blk.addBoco(boco)

    grid.writeToCGNS(volumeFile)
    return blk.dims, faces


def checkMultigrid(dims):
    """Warn if the cell counts will fight ADflow's multigrid.

    ADflow cuts the single block into nProc pieces and every cut plane has to
    still land on a cell boundary after coarsening, i.e. at an even cell index.
    So it is not enough for the total cell count to be even -- it has to stay
    even after being divided by the number of ranks.  The default 550/220 point
    counts give ni - 1 = 926 = 2 x 463 with 463 prime, and *every* nProc from 2
    to 32 then aborts in coarseUtils.F90:1528 with "Non-matching block-to-block
    face", including nProc = 2.  That is a mesh problem, not a solver one, and
    it is invisible until ADflow refuses to start.
    """
    nCell = np.array(dims) - 1
    print(f"Cells (i, j, k) = {tuple(nCell)}")
    for axis, n in zip("ijk", nCell):
        if axis == "k":
            continue  # 2 cells by design: enough for one coarsening, never split
        f = [d for d in range(2, 65) if n % d == 0 and (n // d) % 2 == 0]
        if not f:
            print(f"  WARNING: {n} cells in {axis} cannot be split into equal even "
                  f"pieces -- ADflow multigrid will abort at every nProc.")
        else:
            print(f"  {axis}: {n} cells, even split at nProc = {f[:12]}{' ...' if len(f) > 12 else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- the three wall lengths, upstream to downstream --------------------
    # 0.09 m = lowRe's 0.05 x k.  Scaled because it is a stretched run-out from
    # the plate's own spacing, and the plate's spacing scaled: what matters is
    # length/ds, which stays at ~230 over 108 cells and so keeps the stretching
    # ratio at ~1.013 (see --nInvFront).
    parser.add_argument("--invFrontLength", type=float, default=0.09,
                        help="inviscid slip wall ahead of the plate [m]; 0 disables it")
    # THE calibrated quantity, and the one thing in this file that cannot be
    # scaled from low_RE by a factor.  The boundary layer starts at the
    # slip/no-slip junction at x = -plateLength, so this length alone sets how
    # much running length the layer has when it reaches the experiment's
    # measuring station, which for the highRe case is
    #
    #     x_exp = -8.04 * 4.1 mm = -32.964 mm
    #
    # (vs the lowRe case's -15.4 * 2.27 = -34.958 mm -- the two experiments
    # measured at nearly the same physical place, which is a coincidence and
    # not something to lean on).  Lengthening the *inviscid* run-in instead
    # would do nothing -- it carries no boundary layer.
    #
    # Target: delta = 4.1 mm at x_exp, which then gives Re_delta = 132 840
    # automatically, because flow_conditions.py has already pinned
    # Re_m = 132840/4.1e-3 = 3.240e7 /m.  delta and Re_delta are ONE target,
    # not two.
    #
    # ---- 0.321360 m is CALIBRATED ------------------------------------------
    # Converged on comp_corner_21 at nProc 16, measured on the stage-2
    # (SA-Edwards) surface, both stages run to their residual floors:
    #
    #   delta    = 4.099968 mm  against 4.100  (-32 nm, -0.001%)
    #   Re_delta = 132 796      against 132 840 (-0.03%)
    #
    # i.e. the same standard low_RE/ is held to (2.269999 mm, -1 nm, -0.03%).
    #
    # It took three solves, and the seed is worth recording because it is what
    # a reader would otherwise recompute: the lowRe anchor pushed through
    # delta = C * L_run^0.8 * Re_m^-0.2, with C fitted on the converged lowRe
    # result rather than taken as the textbook 0.37,
    #
    #   C = 2.27e-3 / (0.140968^0.8 * (2.800e7)^-0.2) = 0.33581   (0.908*0.37)
    #   L_run = (4.1e-3 / (C * (3.240e7)^-0.2))^1.25 = 0.30612 m
    #   plateLength = L_run + 32.964 mm             = 0.33909 m
    #
    # seeds 339.09 mm and is 5.4% THICK.  Extrapolating 1.8x in delta and 1.16x
    # in Re_m off one anchor on another grid does not work; the converged C here
    # is 0.35459, not the lowRe grid's 0.33581.  Do not trust the seed formula
    # for a third case either -- use it to start, then iterate.
    #
    # If flow_conditions.MATCH is switched to "rho" (Re_m = 2.8595e7 /m instead
    # of 3.2400e7), THIS NUMBER IS WRONG: a different Re_m grows a different
    # layer over the same plate.  Rescaling the converged result rather than the
    # seed, delta ~ Re_m^-0.2 at fixed L_run gives 4.100*(2.8595/3.2400)^-0.2 =
    # 4.205 mm, so L_run would step to 288.396*(4.100/4.205)^(1/0.8823) =
    # 280.35 mm, i.e. --plateLength 0.31332 m as the STARTING guess.  Calibrate
    # it properly and record the result here next to this one.
    #
    # ---- how to calibrate ---------------------------------------------------
    # Calibrate PER GRID and per stage: the layer's virtual origin moves with
    # the streamwise resolution at the junction, so --leRefine, --nPlate and the
    # inviscid counts all shift it.  Take delta at full precision -- plot_flow's
    # table rounds to 3 decimals, coarser than the target:
    #
    #   import plot_flow
    #   c = plot_flow.Case("output_SAE/comp_corner_sa_edwards_000_surf.cgns")
    #   c.delta_exp, c.Re_delta
    #
    # then step on the LOCAL exponent of delta ~ L_run^n, where L_run is
    # plateLength - 32.964 mm.  The lowRe grid fitted n = 0.693 over its own
    # two converged runs (L_run 140.6250 -> delta 2.266168 mm, 140.9180 ->
    # 2.269438 mm).  n is a local slope, not a constant: the lowRe grid had
    # earlier been fitted at 0.847 on a different distribution, and using that
    # value overshot each step by ~9 um of plate.  So use 0.693 for the FIRST
    # step here, then refit from this grid's own two points -- L_run is 2.2x
    # longer and the exponent has no obligation to have come along.
    #
    #   L_run_new = L_run_old * (delta_target / delta_measured)^(1/n)
    #
    # ---- measured -----------------------------------------------------------
    #   L_run [mm]   delta [mm]   Re_delta   err       stage  n used for the step
    #   ----------   ----------   --------   -------   -----  -------------------
    #   306.036       4.527699     146 650   +10.43%     1     (preview only)
    #   306.036       4.320498     139 939    +5.38%     2     0.8, correlation
    #   286.639       4.077953     132 083    -0.54%     2     0.8823, fitted
    #   288.396       4.099968     132 796    -0.001%    2     <- CONVERGED
    #
    # Three things to read off that table.
    #
    # 1. CALIBRATE ON STAGE 2, and do not shortcut it with stage 1.  The same
    #    grid and the same free stream give delta = 4.5277 mm under SA-noft2 and
    #    4.3205 mm under SA-Edwards -- 4.8% apart, which is most of the error
    #    being chased.  Stage 1 is worth running as an ~11-minute preview of
    #    whether a step went the right way; it is not the number.
    #
    # 2. THE LOCAL EXPONENT IS 0.8823 ON THIS GRID, fitted from the two stage-2
    #    points at L_run 306.036 and 286.639.  Not the 0.8 of the flat-plate
    #    correlation, and emphatically not the 0.693 the lowRe grid fitted --
    #    that value would have overshot the second step by 1.8 mm of plate, i.e.
    #    one wasted 85-minute iteration.  n is local to the grid, the Reynolds
    #    number and the turbulence model; refit it, do not inherit it.
    #
    # 3. The first step, taken at the correlation's 0.8 because only one point
    #    existed, overshot from +5.38% to -0.54%.  That overshoot is what made
    #    the fit possible, so it was not wasted -- but it is why the seed
    #    formula is documented above as a starting point and nothing more.
    #
    # calibrate.py does this arithmetic and carries the same table in HISTORY;
    # run it against a converged stage-2 surface rather than doing it by hand.
    parser.add_argument("--plateLength", type=float, default=0.321360,
                        help="viscous flat plate, from the slip wall to the corner [m]")
    # 0.036 m = lowRe's 0.02 x k, for the same length/ds reason as
    # --invFrontLength: 0.036/0.3648 mm = 98.7 against lowRe's 0.02/0.2015 mm
    # = 99.3, so the same 50 cells do the same stretching.
    parser.add_argument("--invBackLength", type=float, default=0.036,
                        help="inviscid slip wall past the ramp, horizontal [m]; 0 disables it")
    # NOT scaled by k, unlike every other length here.  The ramp is not sized by
    # delta, it is sized by where the interaction has to be resolved out to, and
    # Hao plots both cases over the same window: figure 3 runs x/L = -40 .. +20
    # for both, and figure 2 reaches x/L = +30 for the highRe case (L = 1 mm).
    # 0.04 m of x carries the ramp to x/L = +40, past both.  The highRe bubble
    # is longer than the lowRe one, but it grows UPSTREAM onto the plate, which
    # is where the extra 163 mm of --plateLength already went.
    parser.add_argument("--rampLength", type=float, default=0.04, help="ramp, in x [m]")
    parser.add_argument("--angleDeg", type=float, default=25.0)
    # --- point counts per segment (junction nodes shared) ------------------
    # 109/51, where the refined mesh needed 224/96.  Those counts were set when
    # --leRefine clustered the plate's leading edge to 56 um: an inviscid
    # segment starts at the spacing of the viscous one it joins, so it was
    # starting 4x finer than the plate and needed the points to climb back out.
    # At --leRefine 1 both junctions sit at the plate's own 218 um and the climb
    # is 4x shorter, so the same stretching costs half the points.
    #
    # How fast the run-in coarsens shows up in the OUTER boundary, not just in
    # the wall spacing.  A hyperbolic march lets the wall-normal step respond to
    # the streamwise one, so the wall-to-outer distance tracks the inverse of
    # ds_i -- it lifts where the wall is finely spaced and sags where it is
    # coarse.  Over a wall that is one straight horizontal line the height
    # should not vary at all, so any variation is pure numerics.  Measured over
    # the run-in alone (x = -0.2256 .. -0.1756 m, one straight horizontal slip
    # wall, with volBlend at 0):
    #
    #   leRefine  nInvFront   ds_i over the run-in   ratio   outer-height spread
    #      4         224        56 -  577 um         1.012        0.38%
    #      1          77       218 - 1469 um         1.026        1.80%
    #      1         109       218 -  845 um         1.013        0.61%   <- here
    #
    # 77 is the split that equalizes the two inviscid stretching ratios, and it
    # is the cheapest count that still lands ni-1 on a multigrid-friendly
    # number -- but it stretches the run-in at 1.026 and the outer boundary
    # sags nearly 2% because of it.  109 puts the ratio back at grid 19's 1.013
    # for 32 cells, i.e. 0.8% of the mesh, and that is the better trade: the
    # farfield standoff is a modelling choice and should not follow ds_i.
    #
    # Unchanged at 109 from the lowRe grid, and for the reason in the table
    # above: the run-in is 1.8x longer but it starts from a 1.8x coarser plate
    # spacing, so length/ds is 0.09/0.3681 mm = 244.5 against lowRe's
    # 0.05/0.2180 mm = 229.4.  The same 108 cells therefore do the same job at
    # very nearly the same ratio -- 1.01289 measured here against lowRe's
    # 1.013; scaling the count as well would have refined the run-in for no
    # reason.
    #
    # 109/51 puts ni-1 at 109+874+122+51-4 = 1152 = 2^7 x 9, which splits into
    # equal even pieces at nProc = 8, 16 and 32 -- 16 is what adflow_run1.py
    # uses, and 1152/16 = 72 is even.  (lowRe's was 1184 = 2^5 x 37.)  See
    # checkMultigrid.
    parser.add_argument("--nInvFront", type=int, default=109)
    # 874 holds the plate's spacing at 0.3681 mm over the calibrated 0.32136 m
    # plate, i.e. 4.1/0.3681 = 11.1 cells per delta against the lowRe grid's
    # 2.27/0.218 = 10.4.  That is the invariant being held: cells per delta, not
    # cells per millimetre.  Scaling 808 by the plate-length ratio instead
    # (808 x 0.3214/0.1759 = 1476) would have resolved the highRe layer 1.7x
    # finer than the lowRe one and made the two solutions incomparable at 1.7x
    # the cost.
    #
    # It also lands the plate's spacing within 1% of the ramp's 0.3648 mm, so
    # the corner is very nearly a spacing-continuous junction -- better than the
    # lowRe grid managed (0.218 vs 0.2015 mm, 8% apart).
    #
    # It also has to land ni-1 on a multigrid-friendly number.  ni-1 =
    # nPlate + 278 (see --nInvFront), and the nearest multiples of 32 to the
    # 861 the resolution target alone wants are 842 (ni-1 = 1120) and 874
    # (ni-1 = 1152); 874 is the one that keeps ds under the lowRe grid's
    # cells-per-delta rather than over it.
    #
    # NOTE this count is fixed while --plateLength is being calibrated, so ds
    # drifts a little with each step: it ran 0.3883 -> 0.3661 -> 0.3681 mm over
    # the three solves.  That is deliberate -- changing the count mid-calibration
    # moves the virtual origin and invalidates the exponent fit.
    parser.add_argument("--nPlate", type=int, default=874)
    # 122, down from 220, because --rampLength did NOT scale while delta did.
    # dx = 0.04/121 = 0.3306 mm, arc ds = dx/cos(25 deg) = 0.3648 mm, i.e.
    # 4.1/0.3648 = 11.2 cells per delta against the lowRe grid's
    # 2.27/0.2015 = 11.3.  Same resolution of the same physics, fewer points,
    # because the physics is 1.8x bigger over the same 40 mm of ramp.
    parser.add_argument("--nRamp", type=int, default=122)
    parser.add_argument("--nInvBack", type=int, default=51)
    # s0 alone sets y+; N only decides what the march costs to get from s0 to
    # the outer boundary.  comp_corner_14's 1.6e-6 m left the first cell centre
    # at 0.80 um, y+ ~ 0.20 along the plate but spiking to 0.95 (leading edge)
    # and 0.98 (ramp crest) at the two slip/no-slip junctions, where the wall
    # shear is largest.  y+ scales linearly with s0 and nothing else here does,
    # so 8.0e-7 halves both the spikes and the mean.
    #
    # s0 is the ONE length in this file that is deliberately NOT scaled by k.
    # Scaling it would scale y+ straight up with it, and y+ is the quantity the
    # value was chosen to hold.  It does not stay put by itself either:
    # y+ ~ s0 * V * sqrt(cf) / nu, and going lowRe -> highRe (at MATCH = "Re")
    #
    #   nu   = mu/rho  2.195e-5 -> 1.909e-5   (x0.870, rho is 22% higher)
    #   V                614.6  ->    618.6   (x1.007)
    #   sqrt(cf) at the calibrated station    (x0.912, Re_x 3.9e6 -> 9.9e6)
    #
    # nets to y+ x1.055 at the same s0.  A 5.5% rise off ~0.2 is nothing; a
    # 1.8x rise off it would not be.  Leave s0 alone.  (At MATCH = "rho" the
    # density is the quoted 0.368 instead of 0.417, nu is 2.163e-5, and y+ comes
    # out slightly BELOW the lowRe grid's.)
    #
    # N is 157, up from the lowRe grid's 145, purely to span the 1.8x taller
    # march at the same outer growth ratio.  marchRatios holds --wnRatio 1.048
    # out to --wnFineHeight and solves for whatever outer ratio r1 still reaches
    # --marchDist, so N is chosen by asking which value reproduces the lowRe
    # grid's r1:
    #
    #     N    r1       largest cell    cells below delta
    #    145  1.3044      10.47 mm            117
    #    149  1.2019       7.67 mm            117
    #    153  1.1451       5.88 mm            117
    #    157  1.1098       4.68 mm            117    <- here
    #    161  1.0860       3.82 mm            117
    #   (lowRe: N = 145, r1 = 1.1078, 2.55 mm, 104 cells below delta)
    #
    # 157 lands r1 = 1.1098 against lowRe's 1.1078, and its largest cell is
    # 4.68/2.55 = 1.83x the lowRe grid's, i.e. k.  The same mesh, 1.8x bigger.
    # 117 cells below delta rather than 104 is the one place this grid is
    # RICHER than its ancestor, and it is the direct consequence of holding s0
    # fixed while delta grew -- free resolution, not a design choice.
    #
    # 156 cells stay even through two coarsenings (156 -> 78 -> 39), which is
    # what ADflow's "2w" cycle needs.
    parser.add_argument("--N", type=int, default=157, help="wall-normal points")
    parser.add_argument("--s0", type=float, default=8.0e-7, help="first cell height [m]")
    # 0.0285977 m is comp_corner_14's actual wall-normal march: the distance
    # from wall to outer boundary measured along the first inflow grid line.
    # (The 0.029-0.047 m in the header is the *y* range of the outer face,
    # which is larger only because the ramp lifts its downstream half.)
    # 0.0516 m = lowRe's 0.0285977 x k, i.e. the same 12.6 delta of standoff.
    # The farfield placement is a modelling choice and the thing it has to stay
    # clear of -- the separation shock, the bubble, the corner shock -- all
    # scale with delta, so this is one of the lengths that must scale.
    #
    # It also still lets the leading-edge junction wave leave the domain well
    # upstream of the interaction: at M = 2.88 the Mach angle is
    # asin(1/2.88) = 20.3 deg, so a wave off the junction at x = -321 mm clears
    # a 51.6 mm outer boundary in 51.6/tan(20.3 deg) = 140 mm of x, i.e. by
    # x = -181 mm.  A shock of finite strength is steeper and exits sooner.
    parser.add_argument("--marchDist", type=float, default=0.0516, help="wall-normal march distance [m]")
    parser.add_argument("--spanWidth", type=float, default=0.002, help="total span [m]")
    # Streamwise refinement at the two slip/no-slip junctions.  --leRefine is
    # how much finer the cell at the junction is than the segment's uniform
    # spacing, --leRatio how fast it grows back up; 4x at 1.05 per cell takes
    # ~28 cells to relax, i.e. ~10 mm of the 321 mm plate.
    #
    # Off (1.0), i.e. the junction cell is the plate's own 218 um.  What the
    # refinement was buying was a first cell that resolved the shear rise
    # instead of straddling it, and what it cost was the leading-edge shock
    # captured sharply -- which is not a shock this case wants sharp.  It is an
    # artefact of switching the wall BC from slip to no-slip at a point, not
    # the experiment's leading edge; it leaves the domain through the farfield
    # at least ~181 mm upstream of the corner (a wave off the junction at the
    # Mach angle, asin(1/2.88) = 20.3 deg, clears the 51.6 mm outer boundary in
    # 140 mm of x, and a shock of finite strength is steeper and so exits
    # sooner); and a conservative scheme gets the jump across it right however
    # many cells it is smeared over.
    #
    # Two things follow from turning it off and both are why the counts above
    # moved: the inviscid segments no longer start 4x finer than the plate
    # (see --nInvFront), and the wall cell at the junction now averages the
    # leading-edge shear singularity over 218 um rather than 56 um, which
    # lowers the y+ it reports there.  Note that "lowers the y+ it reports" is
    # not "resolves it better" -- y+ is rho_w*u_tau*y/mu_w and depends on the
    # wall-NORMAL spacing, so the only real lever on it is --s0.  See
    # refinedParams for what the knob does.
    parser.add_argument("--leRefine", type=float, default=1.0,
                        help="junction cell refinement factor; 1 disables it")
    parser.add_argument("--leRatio", type=float, default=1.05,
                        help="growth ratio out of the refined junction cells")
    # Wall-normal growth schedule; see marchRatios.  --wnRatio is the ratio
    # held from the wall out to --wnFineHeight, --wnBlend the number of layers
    # over which it ramps to whatever the outer ratio has to be to still reach
    # the outer boundary.  --wnFineHeight 0 collapses this to a single ratio
    # and reproduces pyHyp's own distribution.  6.3 mm is 1.54x delta at the
    # reference station, the same multiple the lowRe grid's 3.5 mm was of its
    # 2.27 mm -- so the fine region still covers the layer, the separation
    # bubble and the near-wall half of the interaction, and nothing more.
    parser.add_argument("--wnRatio", type=float, default=1.048,
                        help="wall-normal growth ratio held below --wnFineHeight")
    parser.add_argument("--wnFineHeight", type=float, default=0.0063,
                        help="height [m] out to which --wnRatio is held")
    parser.add_argument("--wnBlend", type=int, default=10,
                        help="layers over which the growth ratio ramps to its outer value")
    # 0, not pyHyp's 1e-4 default.  volBlend pulls each layer's cell volumes
    # toward their local average, and a cell volume is ds_i * ds_j, so it makes
    # the march distance compensate for whatever the streamwise spacing is
    # doing -- exactly the coupling described under --nInvFront.  With the
    # leading-edge refinement the streamwise spacing now varies 10x over a
    # dead-flat wall, so that coupling is worth switching off entirely:
    #
    #   volBlend   run-in height spread   plate height spread   LE bump
    #     1e-4            1.42%                 0.37%          +0.107 mm
    #     3e-5            1.03%                 0.25%          +0.067 mm
    #     0               0.92%                 0.25%          +0.066 mm
    #
    # (that column is at nInvFront = 160; with 224 the 0.92% becomes 0.38%).
    # What it costs is corner crowding, 0.35x -> 0.33x of the local wall
    # spacing at the outer boundary above the compression corner.  Those cells
    # are three orders of magnitude larger than the near-wall ones and never
    # touch the time step, so it is a cosmetic loss against a real gain.
    parser.add_argument("--volBlend", type=float, default=0.0,
                        help="pyHyp cell-volume blending; see the note in runPyHyp")
    parser.add_argument("--Twall", type=float, default=275.4)
    # comp_corner_21, not _20: meshes/comp_corner_20*.cgns here are stale copies
    # of the lowRe grid and writing over them would destroy the only record of
    # which is which.  See the header.
    parser.add_argument("--output", type=str, default="meshes/comp_corner_21.cgns")
    args = parser.parse_args()

    volumeFile = os.path.join(BASE_DIR, args.output)
    # keep the intermediate PLOT3D surface next to the mesh it belongs to, so
    # a second --output does not silently overwrite the first one's input
    surfaceFile = f"{os.path.splitext(volumeFile)[0]}_surf.fmt"

    x, y, wallSegments, ratios, sp = buildWallCurve(
        args.invFrontLength, args.plateLength, args.rampLength, args.invBackLength,
        args.angleDeg, args.nInvFront, args.nPlate, args.nRamp, args.nInvBack,
        args.leRefine, args.leRatio)
    # span order (not curve order) sets which way pyHyp's cross-product normal
    # points; descending z makes the march go +y (away from the wall) here.
    # setBCs puts the span back in ascending order afterwards.
    zPlanes = np.array([args.spanWidth, args.spanWidth / 2, 0.0])
    writeSurfaceFile(surfaceFile, x, y, zPlanes)
    print(f"Wrote wall curve: {surfaceFile}  ({len(x)} streamwise pts, "
          f"x from {x[0]:.6f} to {x[-1]:.6f} m)")
    for name, r in ratios.items():
        print(f"  {name}: stretched at ratio {r:.5f} per cell")
    print(f"  wall_plate: {sp['dsPlate']*1e6:.1f} um uniform, refined to "
          f"{sp['dsPlateLE']*1e6:.1f} um at the leading edge "
          f"({sp['dsPlate']/sp['dsPlateLE']:.2f}x)")
    print(f"  wall_ramp : {sp['dsRamp']*1e6:.1f} um uniform, refined to "
          f"{sp['dsRampEnd']*1e6:.1f} um at the crest "
          f"({sp['dsRamp']/sp['dsRampEnd']:.2f}x)")

    extrudeVolume(surfaceFile, volumeFile, args.N, args.s0, args.marchDist, args.volBlend,
                  args.wnRatio, args.wnFineHeight, args.wnBlend)
    dims, faces = setBCs(volumeFile, wallSegments)
    checkMultigrid(dims)
    print(f"Wrote volume mesh: {volumeFile}  (i, j, k) = {tuple(dims)}")
    print(tabulate(
        [[n, t, f"i {r[0][0]}..{r[0][1]}", f"j {r[1][0]}..{r[1][1]}", f"k {r[2][0]}..{r[2][1]}"]
         for n, t, r in faces],
        headers=["family", "CGNS BCType_t", "", "point range", ""], tablefmt="grid"))

    # Hand off to the same script that turned comp_corner_14.cgns into
    # comp_corner_14_fixed.cgns, so the two files end up with identical
    # family names, wall-temperature data sets, SI unit nodes and ADF
    # encoding rather than a second, drifting copy of that logic here.
    # fix_bc.py is what turns the BC *types* above into the family names
    # ADflow addresses groups by -- see the "Labelling the BCs" note up top.
    subprocess.run([sys.executable, FIX_BC, volumeFile, str(args.Twall)], check=True)


if __name__ == "__main__":
    main()
