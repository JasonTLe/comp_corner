#!/usr/bin/env python3.11
"""
flow_conditions.py  --  free-stream state for the highRe compression corner.

Single source of truth for the AeroProblem.  adflow_run1.py (stage 1),
adflow_run2.py (stage 2) and eigen_run.py all import make_ap() from here,
because they MUST agree exactly: stage 2 restarts from stage 1's volume file
and eigen_run.py linearises about stage 2's, and a mismatched free stream
silently rescales the base flow the eigensolver is handed.

Run it directly for the derived-state table:

    python3.11 flow_conditions.py


================================================================================
THE CASE
================================================================================
Hao, JFM 2023, 971 A28, section 2, quoting Zheltovodov et al. (1990):

    "The highRe case has M_inf = 2.88 and Re_delta = 132 840 with
     delta = 4.1 mm measured 8.04 delta upstream of the corner.  The
     free-stream density is 0.368 kg m^-3, and the free-stream temperature
     is 114.8 K."

    "The length of the flat plate is determined by matching the experimental
     boundary-layer thickness for the highRe and lowRe cases, respectively."

So the experiment pins five numbers, and mesh.py's --plateLength is the free
parameter that makes the sixth (delta at the station) come out right.  The
station is fixed in space at

    x_exp = -8.04 * 4.1 mm = -32.964 mm      (x = 0 is the corner)

which is very nearly the lowRe case's -15.4 * 2.27 = -34.958 mm -- the two
experiments measured at almost the same physical place, with boundary layers
1.8x apart in thickness.  Wall temperature is 275.4 K for both cases.


================================================================================
WHY THIS FILE EXISTS: rho AND Re_delta ARE NOT MUTUALLY CONSISTENT
================================================================================
The five quoted numbers are one too many.  Given M, T and Sutherland's law,
the free stream has one degree of freedom left, and rho and Re_delta both
claim it:

    a   = sqrt(gamma*R*T)                        = 214.7917 m/s
    V   = M*a                                    = 618.6001 m/s
    mu  = Sutherland(114.8 K)                    = 7.960907e-06 Pa.s
    Re/m = rho*V/mu                              <- ONE equation, TWO givens

    from rho = 0.368        ->  Re/m = 2.859534e7 /m  ->  Re_delta = 117 241
    from Re_delta = 132 840 ->  Re/m = 3.240000e7 /m  ->  rho      = 0.416963

i.e. they disagree by 13.3%.  This is not a units slip or a Sutherland-constant
quibble; it is in the source data.  The lowRe case has the same problem in the
same direction, 8.0%:

    case     T [K]   M     quoted rho   rho from Re_delta   quoted Re_delta   Re_delta from rho
    lowRe    108.0   2.95    0.314          0.341248             63 560            58 485
    highRe   114.8   2.88    0.368          0.416963            132 840           117 241

(no single viscosity offset reconciles both rows -- the implied mu is 8% low
for lowRe and 12% low for highRe -- so this is experimental scatter between
two separately reported quantities, not a systematic convention difference.)

--------------------------------------------------------------------------------
Which one to feed ADflow, and why "input density instead of Reynolds" changes
LESS than it looks like it should
--------------------------------------------------------------------------------
AeroProblem accepts either group and both reach the same three numbers:

    {mach, T, reynolds, reynoldsLength}  -> _updateFromRe:  rho = Re/L * mu / V
    {mach, T, rho}                       -> _updateFromM :  Re/L = rho * V / mu

and pyADflow._setAeroProblemData (pyADflow.py:4113-4115) hands the solver

    flowvarrefstate.pinfdim   = P
    flowvarrefstate.tinfdim   = T
    flowvarrefstate.rhoinfdim = rho
    inputphysics.mach         = mach
    inputphysics.{musuthdim, tsuthdim, ssuthdim, rgasdim, prandtl}

and NOTHING else.  `reynolds` and `reynoldsLength` are never passed to the
Fortran layer at all.  So specifying Re is not an alternative *physics* input
to specifying rho -- it is an alternative *spelling* of rho, evaluated through
Sutherland's law at T.  Whichever group you write, the solver sees (M, T, rho).

That is why the old comment in the lowRe scripts, "reynoldsLength = 2.27e-3,
change this so that density matches up", works: reynoldsLength is a pure
divisor on rho.  Setting reynoldsLength = 4.645513e-3 here with
reynolds = 132 840 would land rho exactly on 0.368 -- and would then be a
strictly worse way of writing rho = 0.368, because the "4.6 mm" would look
like a boundary-layer thickness while being nothing of the kind.

So the real choice is not "Re or rho", it is WHICH OF THE TWO INCONSISTENT
EXPERIMENTAL NUMBERS TO HONOUR, and the honest way to write it is:

    MATCH = "Re"   ->  reynolds/reynoldsLength, Re_delta exact, rho +13.3%
    MATCH = "rho"  ->  rho, rho exact, Re_delta -11.7%

--------------------------------------------------------------------------------
Default is MATCH = "Re".  Three reasons:
--------------------------------------------------------------------------------
1. Re_delta IS the case.  The paper names its two cases "Re_delta = 63 560"
   and "Re_delta = 132 840" -- every figure is labelled by it.  Reproducing
   "the highRe case" means reproducing that number.

2. Nothing in the problem is sensitive to rho on its own.  The solution
   depends on (M, gamma, Pr, Re, T_wall/T_inf) and on nothing else dimensional;
   rho only ever enters through Re, and Cf, p/p_inf and the eigenvalues are all
   normalised by free-stream quantities.  Missing Re by 12% moves Cf and the
   separation length; missing rho by 13% at fixed Re moves nothing at all.

3. It is what low_RE/ already does (reynolds = 63560, reynoldsLength = 2.27e-3,
   rho = 0.3412 rather than the quoted 0.314).  Two cases calibrated on
   different conventions cannot be compared against each other, and comparing
   them is the whole point of running both.

Switch MATCH to "rho" to run the other convention.  It is a real sensitivity
study -- 13% in Re is roughly a factor of 1.13 in Re_delta, which at these
conditions shortens the separation bubble a few per cent -- but note it needs
its OWN --plateLength calibration, because a different Re/m grows a different
boundary layer over the same plate.  Both calibrated lengths are recorded in
mesh.py.
"""

from baseclasses import AeroProblem

# ---------------------------------------------------------------------------
# Experiment (Zheltovodov et al. 1990, as tabulated by Hao 2023).  These are
# the quoted numbers, verbatim -- do not "fix" them to be self-consistent.
# ---------------------------------------------------------------------------
MACH = 2.88            # free-stream Mach number
T_INF = 114.8          # free-stream static temperature, K
RHO_INF = 0.368        # free-stream density, kg/m^3   ) mutually inconsistent
RE_DELTA = 132840      # Re based on DELTA_EXP         ) by 13.3%, see above
DELTA_EXP = 4.1e-3     # incoming boundary-layer thickness, m
X_EXP = -8.04*DELTA_EXP  # station where it was measured, m (x = 0 is the corner)
T_WALL = 275.4         # isothermal wall temperature, K (both cases)
L_REF = 1.0e-3         # Hao's characteristic length L, m -- x/L in every figure

# "Re"  -- honour Re_delta = 132 840 exactly, let rho come out at 0.416963.
# "rho" -- honour rho = 0.368 exactly, let Re_delta come out at 117 241.
# See the long note above.  Changing this REQUIRES a new --plateLength; see
# mesh.py.
MATCH = "Re"


def make_ap(name):
    """The highRe AeroProblem.  `name` sets the output file stem.

    Stage 1 uses "comp_corner_sa", stage 2 and eigen_run.py use
    "comp_corner_sa_edwards"; everything else about the state is identical
    between them by construction.
    """
    common = dict(name=name, mach=MACH, T=T_INF,
                  areaRef=1.0, chordRef=1.0, evalFuncs=[])
    if MATCH == "Re":
        # reynoldsLength is DELTA_EXP, so `reynolds` reads as Re_delta and
        # ap.re comes out as Re per metre.  Both are only a spelling of rho.
        return AeroProblem(reynolds=RE_DELTA, reynoldsLength=DELTA_EXP, **common)
    elif MATCH == "rho":
        return AeroProblem(rho=RHO_INF, **common)
    raise ValueError(f"MATCH must be 'Re' or 'rho', got {MATCH!r}")


def report(ap, comm=None):
    """Print the derived free-stream state, and how far each convention lands
    from the experimental number it is NOT matching.  Rank-0 only under MPI."""
    if comm is not None and comm.rank != 0:
        return
    print("=" * 68)
    print(f"highRe free stream  --  MATCH = {MATCH!r}"
          f"  ({'Re_delta exact' if MATCH == 'Re' else 'rho exact'})")
    print("-" * 68)
    print(f"  Mach              : {ap.mach:.4f}")
    print(f"  Temperature T     : {ap.T:.4f} K")
    print(f"  Speed of sound a  : {ap.a:.4f} m/s")
    print(f"  Velocity V        : {ap.V:.4f} m/s")
    print(f"  Viscosity mu      : {ap.mu:.6e} Pa.s   (Sutherland at T)")
    print(f"  Density rho       : {ap.rho:.6f} kg/m^3")
    print(f"  Pressure P        : {ap.P:.4f} Pa")
    print(f"  Re per metre      : {ap.re:.6e} 1/m")
    print("-" * 68)
    reDelta = ap.re*DELTA_EXP
    print(f"  rho      vs exp   : {ap.rho:.6f} vs {RHO_INF:.3f} kg/m^3"
          f"   ({100*(ap.rho/RHO_INF - 1):+.2f}%)")
    print(f"  Re_delta vs exp   : {reDelta:.0f} vs {RE_DELTA}"
          f"          ({100*(reDelta/RE_DELTA - 1):+.2f}%)")
    print(f"  (delta = {DELTA_EXP*1e3:.2f} mm at x = {X_EXP*1e3:.3f} mm"
          f" = {X_EXP/DELTA_EXP:.2f} delta upstream of the corner)")
    print("=" * 68)


if __name__ == "__main__":
    # Both conventions side by side, so the 13.3% is on the record and nobody
    # has to rederive it to know which knob they just turned.
    saved = MATCH
    for mode in ("Re", "rho"):
        MATCH = mode
        report(make_ap("compare"))
    MATCH = saved

    apRe = AeroProblem(name="c", mach=MACH, T=T_INF, reynolds=RE_DELTA,
                       reynoldsLength=DELTA_EXP, areaRef=1.0, chordRef=1.0,
                       evalFuncs=[])
    print("\nFor reference, the reynoldsLength that would put rho exactly on "
          f"{RHO_INF} kg/m^3\nwhile still writing reynolds = {RE_DELTA}: "
          f"{RE_DELTA*apRe.mu/(RHO_INF*apRe.V):.6e} m."
          "\nThat is just rho = 0.368 in disguise -- use MATCH = 'rho' instead.")
