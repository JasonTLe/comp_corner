#!/usr/bin/env python3.11
"""
plot_mesh.py
============

Render a structured CGNS volume mesh (one produced by make_comp_corner_mesh.py,
or any of meshes/comp_corner_*.cgns) and check how square its grid lines stand
on the wall.  Three PNGs:

    <stem>_full.png         the whole k=0 plane
    <stem>_corner_zoom.png  the corner, near-wall layers, y off the wall
                            exaggerated so the boundary-layer clustering and
                            the wall-normal lines are actually visible
    <stem>_ortho.png        wall-normal angle along the wall + the skew field,
                            i.e. the quantitative version of the zoom

Everything is drawn from the k=0 spanwise plane: the span is a trivial
extrusion, so that plane is the whole 2D story.  Axes are
(i, j, k) = (streamwise, wall-normal, spanwise), which is comp_corner_14's
convention -- *not* pyHyp's raw order, where the marched direction is last;
make_comp_corner_mesh.setBCs swaps it back.

matplotlib rather than pyvista: this is a 2.5D grid whose only interesting
plane is k=0, and the numbers below (skew, spacing) want axes and a colorbar
more than they want a 3D view.

Usage:
    python3.11 plot_mesh.py meshes/comp_corner_16.cgns
    python3.11 plot_mesh.py meshes/comp_corner_16.cgns --outDir figs
"""

import os
import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cgnsutilities.cgnsutilities import readGrid


def loadPlane(fileName):
    """k=0 spanwise plane of the single block, as an (ni, nj, 2) x-y array."""
    blk = readGrid(fileName).blocks[0]
    return blk.coords[:, :, 0, :2], tuple(blk.dims)


def wallKinks(c, tolDeg=0.01):
    """Indices of the wall nodes where the wall tangent is discontinuous, and
    how far it turns at each, in degrees.  The tolerance is well above the
    round-off in a unit tangent (~1e-5 deg on a stretched segment) and well
    below any turn worth calling a corner.  With the inviscid run-in and run-out
    segments there are two of these -- the 25 deg compression corner at the
    foot of the ramp and the 25 deg expansion corner at its top -- and neither
    can have an orthogonal grid line, so callers need to know where they are
    rather than assuming the only one is at x = 0."""
    tw = c[1:, 0] - c[:-1, 0]
    tw /= np.linalg.norm(tw, axis=-1, keepdims=True)
    turn = np.degrees(np.arccos(np.clip((tw[1:] * tw[:-1]).sum(-1), -1, 1)))
    idx = np.where(turn > tolDeg)[0] + 1
    return idx, turn[idx - 1]


def wallSkew(c):
    """Deviation from 90 deg, in degrees, between each j-line and the wall
    segment it stands on.  Reported per wall node as the worse of the segment
    on either side, so a line that is square to the plate but not to the ramp
    still shows up.

    Returned with the kink nodes included; those are *unavoidably* skew -- the
    wall tangent is discontinuous there, so the grid line bisects the turn and
    sits half of it off each side.  Callers reporting a "max" should exclude
    them, which is why wallKinks() exists.
    """
    tj = c[:, 1] - c[:, 0]
    tj /= np.linalg.norm(tj, axis=-1, keepdims=True)
    tw = c[1:, 0] - c[:-1, 0]  # wall segment tangents
    tw /= np.linalg.norm(tw, axis=-1, keepdims=True)
    fwd = np.abs(np.degrees(np.arccos(np.clip((tj[:-1] * tw).sum(-1), -1, 1))) - 90)
    bwd = np.abs(np.degrees(np.arccos(np.clip((tj[1:] * tw).sum(-1), -1, 1))) - 90)
    return np.maximum(np.r_[fwd, 0.0], np.r_[0.0, bwd])


def cellSkew(c):
    """Equiangle skew of every cell in the plane: the largest departure of any
    of its four corner angles from 90 deg, in degrees.  This is the standard
    mesh-quality measure (what Pointwise, Fluent and ICEM report), and it is
    the honest one here -- a node-centred i-vs-j angle uses central differences
    that average across the corner seam and quietly halves it.

    Returned on the cell grid, shape (ni-1, nj-1).
    """
    a, b, d, e = c[:-1, :-1], c[1:, :-1], c[1:, 1:], c[:-1, 1:]
    angs = []
    for u, v, w in [(e, a, b), (a, b, d), (b, d, e), (d, e, a)]:
        v1, v2 = u - v, w - v
        v1 /= np.linalg.norm(v1, axis=-1, keepdims=True)
        v2 /= np.linalg.norm(v2, axis=-1, keepdims=True)
        angs.append(np.degrees(np.arccos(np.clip((v1 * v2).sum(-1), -1, 1))))
    angs = np.stack(angs)
    return np.maximum(angs.max(0) - 90.0, 90.0 - angs.min(0))


def cellMetrics(c):
    """Aspect ratio and neighbour area ratios, on the same cell grid."""
    a, b, d, e = c[:-1, :-1], c[1:, :-1], c[1:, 1:], c[:-1, 1:]
    area = 0.5 * np.abs(np.cross(d - a, e - b))
    li = 0.5 * (np.linalg.norm(b - a, axis=-1) + np.linalg.norm(d - e, axis=-1))
    lj = 0.5 * (np.linalg.norm(e - a, axis=-1) + np.linalg.norm(d - b, axis=-1))
    ar = np.maximum(li / lj, lj / li)
    ri = np.maximum(area[1:] / area[:-1], area[:-1] / area[1:])
    rj = np.maximum(area[:, 1:] / area[:, :-1], area[:, :-1] / area[:, 1:])
    return ar, ri, rj, area


def wallFitted(pts, wall):
    """Project pts (..., 2) onto the wall polyline and return, for each point,
    the arc length of its foot along the wall and its perpendicular distance
    from the wall.  Brute force over every wall segment -- a few million
    distance evaluations, which is nothing, and it avoids assuming the foot of
    a node sits under its own i index (exactly the assumption that would make
    this plot show orthogonality whether or not the mesh has it)."""
    a, b = wall[:-1], wall[1:]
    seg = b - a
    segLen2 = (seg ** 2).sum(-1)
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(seg, axis=-1))]

    flat = pts.reshape(-1, 1, 2)
    t = np.clip(((flat - a) * seg).sum(-1) / segLen2, 0.0, 1.0)   # (npts, nseg)
    foot = a + t[..., None] * seg
    d = np.linalg.norm(flat - foot, axis=-1)
    k = d.argmin(axis=1)
    r = np.arange(len(k))
    return (arc[k] + t[r, k] * np.sqrt(segLen2[k])).reshape(pts.shape[:-1]), d[r, k].reshape(pts.shape[:-1])


def wire(ax, c, di, dj, **kw):
    ni, nj = c.shape[:2]
    kw = {"color": "black", "lw": 0.35, **kw}
    for i in list(range(0, ni, di)) + [ni - 1]:
        ax.plot(c[i, :, 0], c[i, :, 1], **kw)
    for j in list(range(0, nj, dj)) + [nj - 1]:
        ax.plot(c[:, j, 0], c[:, j, 1], **kw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("gridFile", help="structured CGNS mesh file")
    parser.add_argument("--outDir", default=".", help="directory for output PNGs")
    parser.add_argument("--kink", type=int, default=0,
                        help="which wall kink the zoom figure centres on (0 = the "
                             "compression corner at the foot of the ramp)")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.gridFile))[0]
    c, dims = loadPlane(args.gridFile)
    ni, nj = c.shape[:2]
    print(f"Loaded {args.gridFile}: dims (i, j, k) = {dims}")

    dev = wallSkew(c)
    kinks, turns = wallKinks(c)
    smooth = np.ones(len(dev), dtype=bool)
    smooth[kinks] = False
    ic = int(kinks[min(args.kink, len(kinks) - 1)]) if len(kinks) else ni // 2
    fs = cellSkew(c)
    ar, ri, rj, area = cellMetrics(c)
    print(f"  wall kinks at i = {list(kinks)}, turning "
          f"{', '.join(f'{t:.1f}deg' for t in turns)} -- their grid lines bisect the turn "
          f"({', '.join(f'{dev[k]:.2f}deg' for k in kinks)}), which no mesh can avoid")
    print(f"  wall skew away from the kinks: max {dev[smooth].max():.4f} deg, "
          f"mean {dev[smooth].mean():.5f} deg")
    print(f"  cell equiangle skew: max {fs.max():.2f} deg, p99.9 {np.percentile(fs, 99.9):.2f}, "
          f"p99 {np.percentile(fs, 99):.2f}, >10deg {(fs > 10).sum()} cells "
          f"({100 * (fs > 10).mean():.2f}%)")
    print(f"  aspect ratio: max {ar.max():.0f}, median {np.median(ar):.1f}    "
          f"neighbour area ratio: i {ri.max():.3f}, j {rj.max():.3f}    "
          f"min area {area.min():.2e}, negatives {(area <= 0).sum()}")

    # ------------------------------------------------------------------
    # Figure 1: full k=0 plane
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 5))
    wire(ax, c, 12, 3)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{base} — k=0 plane, every 12th streamwise / 3rd wall-normal line", loc="left")
    fullPng = os.path.join(args.outDir, f"{base}_full.png")
    fig.savefig(fullPng, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {fullPng}")

    # ------------------------------------------------------------------
    # Figure 2: the corner, two ways.
    #
    # Left, true scale: no exaggeration at all, but enough layers that the
    # block is thick enough to see.  Orthogonality reads directly here -- the
    # grid lines meet the plate and the ramp square.
    #
    # Right, wall-fitted: every node projected onto the wall polyline and
    # replotted as (arc length of the foot, perpendicular distance).  That
    # rectifies the corner and lets the near-wall layers be stretched without
    # distorting any angle, because the stretch is now normal to the wall
    # everywhere rather than along a fixed global axis.  In these coordinates
    # a wall-orthogonal j-line is exactly vertical, so the figure *is* the
    # orthogonality check: any lean is skew.
    # ------------------------------------------------------------------
    NJ_PHYS, NJ_FIT = 90, 30
    sl = slice(max(ic - 45, 0), min(ic + 46, ni))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6),
                                   gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.25))

    wire(axL, c[sl, :NJ_PHYS], 3, 4, lw=0.5)
    axL.set_aspect("equal")
    axL.set_xlabel("x [m]")
    axL.set_ylabel("y [m]")
    axL.set_title(f"Kink at i={ic}, first {NJ_PHYS} layers — true scale", loc="left")

    sFoot, nDist = wallFitted(c[sl, :NJ_FIT], c[:, 0])
    sKink = np.linalg.norm(np.diff(c[:ic + 1, 0], axis=0), axis=1).sum()
    fit = np.stack([(sFoot - sKink) * 1e3, nDist * 1e6], axis=-1)
    wire(axR, fit, 3, 2, lw=0.5)
    axR.axvline(0.0, color="#d62728", lw=0.8, ls="--")
    axR.set_xlabel("arc length along the wall, from the kink [mm]")
    axR.set_ylabel("perpendicular distance off the wall [µm]")
    axR.set_title(f"Same {NJ_FIT} layers, wall-fitted — vertical = orthogonal", loc="left")

    fig.suptitle(f"{base} — wall kink at i={ic}", fontsize=13)
    cornerPng = os.path.join(args.outDir, f"{base}_corner_zoom.png")
    fig.savefig(cornerPng, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {cornerPng}")

    # ------------------------------------------------------------------
    # Figure 3: orthogonality, quantified
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), gridspec_kw=dict(hspace=0.38))

    ax = axes[0]
    ax.semilogy(np.maximum(dev, 1e-6), color="#1f77b4", lw=0.9)
    for n, (k, t) in enumerate(zip(kinks, turns)):
        ax.axvline(k, color="#d62728", lw=0.8, ls="--")
        # stagger left/right so labels on neighbouring kinks do not collide
        side = -1 if n % 2 == 0 else 1
        ax.annotate(f"i={k}: {dev[k]:.1f}° — bisector of the {t:.0f}° turn",
                    xy=(k, dev[k]), xytext=(k + side * 15, dev[k] * (0.25 if n % 2 else 1.0)),
                    fontsize=9, color="#d62728", va="center",
                    ha="right" if side < 0 else "left")
    ax.set_xlabel("i (streamwise node)")
    ax.set_ylabel("|angle to wall − 90°|  [deg]")
    ax.set_title(f"Wall orthogonality along the wall — max {dev[smooth].max():.4f}° "
                 f"away from the {len(kinks)} kink(s)", loc="left")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    iLo = max(int(kinks.min()) - 60, 0) if len(kinks) else 0
    iHi = min(int(kinks.max()) + 61, ni) if len(kinks) else ni
    ax.plot(np.arange(iLo, iHi), dev[iLo:iHi], "o-", ms=2.5, lw=0.8, color="#1f77b4")
    ax.set_xlabel("i (streamwise node)")
    ax.set_ylabel("|angle to wall − 90°|  [deg]")
    ax.set_title("Same, zoomed on the kinks — a clean mesh is flat either side of each spike",
                 loc="left")
    ax.grid(alpha=0.3)

    ax = axes[2]
    pc = ax.pcolormesh(c[..., 0], c[..., 1], fs, shading="flat", cmap="magma", vmin=0.0)
    ax.set_aspect("equal")
    fig.colorbar(pc, ax=ax, label="equiangle skew [deg]", shrink=0.9)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Cell equiangle skew — max {fs.max():.2f}°, in the two cells either side of "
                 f"each kink; {100 * (fs > 10).mean():.2f}% of cells above 10°", loc="left")

    fig.suptitle(f"{base}: grid-line orthogonality", fontsize=14, y=0.98)
    orthoPng = os.path.join(args.outDir, f"{base}_ortho.png")
    fig.savefig(orthoPng, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {orthoPng}")


if __name__ == "__main__":
    main()
