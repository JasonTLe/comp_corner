#!/usr/bin/env python3.11
"""
calibrate.py  --  propose the next mesh.py --plateLength.

The one free parameter in this case is the length of the viscous flat plate.
Everything else is pinned by the experiment (flow_conditions.py); the plate
length is what makes the incoming boundary layer come out 4.1 mm thick at the
station 8.04 delta upstream of the corner.  Hao, JFM 2023, 971 A28, section 2:
"The length of the flat plate is determined by matching the experimental
boundary-layer thickness for the highRe and lowRe cases, respectively."

There is no closed form for it -- the layer's virtual origin depends on the
grid as well as the flow -- so it is a fixed-point iteration, and each step
costs a full two-stage solve.  This script is the arithmetic between steps,
kept here rather than in a comment so the fit is reproducible and the recorded
points accumulate in one place.

USAGE
    # after a converged stage 2, from high_RE/:
    python3 calibrate.py output_SAE/comp_corner_sa_edwards_000_surf.cgns

    # a stage-1 surface works too, as a cheap ~11-minute preview; it is NOT
    # the calibration target (see --stage)
    python3 calibrate.py output_SA/comp_corner_sa_000_surf.cgns --stage 1

    # then rebuild and rerun:
    python3 mesh.py --plateLength <proposed>
    python3 fix_bc.py meshes/comp_corner_21.cgns 275.4
    mpiexec -n 16 python3 adflow_run1.py
    mpiexec -n 16 python3 adflow_run2.py

THE STEP
    delta ~ L_run^n,  L_run = plateLength - |X_EXP|

    L_run_next = L_run * (delta_target / delta_measured)^(1/n)

n is a LOCAL exponent, not a constant.  With one measured point this script
falls back to the flat-plate correlation's 0.8; with two or more in HISTORY it
fits n from the last two and says so.  The lowRe grid fitted 0.693 over a 0.2%
step in L_run, and a value fitted on one grid does not transfer to another --
see the --plateLength note in mesh.py for what that cost there.
"""

import argparse
import os
import sys

import plot_flow
import flow_conditions

# Measured points on comp_corner_21, appended as they are produced.  Keep
# stage-2 (SA-Edwards) points only in the fit -- stage 1 is a different
# turbulence model and sits at a different delta for the same plate.
#
#   (L_run [m], delta [m], stage, note)
HISTORY = [
    (0.306036, 4.527699e-3, 1, "seed from the lowRe-anchored correlation"),
    (0.306036, 4.320498e-3, 2, "same grid, SA-Edwards -- the calibration target"),
    (0.286639, 4.077953e-3, 2, "step 1, taken at the correlation's n = 0.8"),
    (0.288396, 4.099968e-3, 2, "step 2, at the fitted n = 0.8823 -- CONVERGED"),
]

# CONVERGED at L_run = 288.396 mm, i.e. --plateLength 0.321360 m, which is now
# mesh.py's default.  delta = 4.099968 mm against the target 4.100 (-32 nm,
# -0.001%) and Re_delta = 132 796 against 132 840 (-0.03%) -- the same standard
# the lowRe grid was calibrated to.  Nothing further to run unless the wall
# distribution changes; if it does, this whole table is void (see mesh.py).

DELTA_TARGET = flow_conditions.DELTA_EXP      # 4.1 mm
X_EXP = abs(flow_conditions.X_EXP)            # 32.964 mm


def fitExponent(points):
    """Local exponent n in delta ~ L_run^n from the last two points, or None."""
    if len(points) < 2:
        return None
    import math
    (l0, d0), (l1, d1) = points[-2], points[-1]
    if l0 == l1 or d0 == d1:
        return None
    return math.log(d1/d0) / math.log(l1/l0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("surfFile", help="converged surface CGNS to measure delta on")
    p.add_argument("--plate-length", type=float, default=None, metavar="M",
                   help="the --plateLength this solution was run at [m]; "
                        "defaults to mesh.py's current default")
    p.add_argument("--stage", type=int, choices=(1, 2), default=2,
                   help="which stage produced the file (default 2, the target)")
    p.add_argument("-n", "--exponent", type=float, default=None,
                   help="override the exponent in delta ~ L_run^n")
    a = p.parse_args()

    if a.plate_length is None:
        # read mesh.py's default rather than making the user retype it
        import re
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "mesh.py")).read()
        m = re.search(r'"--plateLength",\s*type=float,\s*default=([0-9.eE+-]+)', src)
        if not m:
            sys.exit("could not read --plateLength from mesh.py; pass --plate-length")
        a.plate_length = float(m.group(1))

    plot_flow.DELTA_EXP = DELTA_TARGET
    plot_flow.X_EXP = -X_EXP
    case = plot_flow.Case(a.surfFile)

    lRun = a.plate_length - X_EXP
    delta = case.delta_exp
    err = delta/DELTA_TARGET - 1

    stage2 = [(l, d) for l, d, s, _ in HISTORY if s == 2]
    nFit = fitExponent(stage2)
    n = a.exponent if a.exponent is not None else (nFit if nFit else 0.8)
    source = ("--exponent" if a.exponent is not None else
              f"fitted from the last two stage-2 points" if nFit else
              "flat-plate correlation (only one point so far)")

    lNext = lRun * (DELTA_TARGET/delta)**(1.0/n)
    plateNext = lNext + X_EXP

    print("=" * 68)
    print(f"  file            : {a.surfFile}  (stage {a.stage})")
    print(f"  plateLength     : {a.plate_length*1e3:.4f} mm"
          f"   -> L_run = {lRun*1e3:.4f} mm")
    print(f"  station x_exp   : {-X_EXP*1e3:.3f} mm"
          f"  ({X_EXP/DELTA_TARGET:.2f} delta upstream of the corner)")
    print("-" * 68)
    print(f"  delta measured  : {delta*1e3:.6f} mm"
          f"   vs target {DELTA_TARGET*1e3:.3f} mm   ({100*err:+.3f}%)")
    print(f"  Re_m            : {case.Re_m:.6e} 1/m")
    print(f"  Re_delta        : {case.Re_delta:.0f}"
          f"   vs {flow_conditions.RE_DELTA} ({100*(case.Re_delta/flow_conditions.RE_DELTA - 1):+.2f}%)")
    print("-" * 68)
    print(f"  exponent n      : {n:.4f}   ({source})")
    print(f"  next L_run      : {lNext*1e3:.4f} mm")
    print(f"  next plateLength: {plateNext:.6f} m   ({plateNext*1e3:.4f} mm)")
    print()
    print(f"    python3 mesh.py --plateLength {plateNext:.6f}")
    print(f"    python3 fix_bc.py meshes/comp_corner_21.cgns 275.4")
    print("=" * 68)
    if a.stage == 1:
        print("NOTE: stage 1 is SA-noft2, not the SA-Edwards model the case is")
        print("reported with.  Use it to get close cheaply, then confirm on a")
        print("stage-2 solution before recording the point in HISTORY.")
    if abs(err) < 1e-3:
        print(f"Converged: |{100*err:+.3f}%| is inside 0.1%.  Record the point in")
        print("HISTORY and in mesh.py's --plateLength note, and stop.")


if __name__ == "__main__":
    main()
