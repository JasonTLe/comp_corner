#!/usr/bin/env python3
"""
eigen_spectra.py
================

Global stability analysis of the converged compression-corner base flow,
following Hao, JFM 2023, 971 A28 section 3.3, using the LST eigenvalue
machinery in this ADflow fork (src/lst/, driven through solveLSTEigenMatrix).

It drives solveLSTEigenMatrix, which is the fork's OWN eigensolver: SLEPc
shift-invert with a MUMPS LU inner solve, all in Fortran
(src/lst/eigenSolveMatrix.F90).  Nothing here goes through petsc4py, and it
does NOT need the complex build -- eigenSolveMatrix.F90 carries an #else
branch for real scalars, where SLEPc returns each conjugate pair through
eigValImagDummy.  The one thing real arithmetic costs is the shift: SLEPc
cannot take a complex shift without complex scalars, so --shift is a real
number and the transformation targets the rightmost eigenvalues, which for a
stability question is the interesting end anyway.

Physics options are adflow_run2.py's, so the operator is linearized about the
SA-Edwards field that stage 2 actually converged.

    mpiexec -n 16 python3 eigen_spectra.py \
        --restart-file ./output_SAE/comp_corner_sa_edwards_000_vol.cgns

WHAT THIS DOES AND DOES NOT COVER
---------------------------------
The fork's LST module assembles dR/dw on the mesh it is given and solves

    (-dR/dw) x = lambda (dU/dw) x

There is no spanwise wavenumber anywhere in src/lst/ -- grep it -- so beta is
set entirely by the spanwise discretization of the grid.  comp_corner_21 is two
cells across a 2 mm span between symmetry planes, i.e. this script computes the
beta = 0 spectrum, which is Hao's figure 5(b).

Figure 5(a), at beta*L = 0.36 with L = 1 mm (paper p.4: "a characteristic length
of L = 1 mm"), needs beta = 360 rad/m.  Mirror symmetry at both spanwise ends
admits beta = n*pi/Lz, so a HALF wavelength Lz = pi/360 = 8.727 mm carries the
n = 1 mode.  That is a different grid and a much larger eigenproblem; see the
note at the bottom of this file.

EIGENVALUE CONVENTION
---------------------
ADflow returns lambda for perturbations ~ exp(lambda*t).  Hao writes them as
exp[i*beta*z - i*(omega_r + i*omega_i)*t], so

    omega_i (growth rate) =  Re(lambda)
    omega_r (frequency)   = -Im(lambda)

and figure 5 plots them scaled by L/u_inf.  Both the raw and the scaled values
are written, because the raw ones are in ADflow's internal nondimensionalization
and the scaling below is the thing most likely to need checking first.
"""
import argparse
import os
import sys
import csv
import math

from mpi4py import MPI
from adflow import ADFLOW

# Same free stream as adflow_run1/2.py -- the base flow this script linearises
# about was converged with it, so it cannot be restated here.
import flow_conditions

ET_ROOT = "/home/jason/packages/ADflowContribute/eigen_tests"
if ET_ROOT not in sys.path:
    sys.path.insert(0, ET_ROOT)
from common import mumps_safe_tokens, set_petsc_options_env  # noqa: E402

L_REF = flow_conditions.L_REF   # Hao's characteristic length, 1 mm


def make_ap():
    """Must match adflow_run2.py exactly -- the base flow was converged with it.

    Which is now guaranteed rather than asserted: both go through
    flow_conditions.make_ap(), so the free stream cannot drift between the
    solve and the linearisation about it."""
    return flow_conditions.make_ap("comp_corner_sa_edwards")


def build_options(args):
    """adflow_run2.py's physics with the solver switched off (nCycles 0): the
    base flow is read from the restart, not recomputed.  useBlockettes MUST stay
    False with saVariant set -- see the long note in adflow_run2.py."""
    return {
        "gridFile": args.grid_file,
        "restartFile": args.restart_file,
        "outputDirectory": args.output_dir,
        "equationType": "RANS",
        "turbulenceModel": "SA",
        "saVariant": "SA-Edwards",
        "useBlockettes": False,
        "useft2SA": False,
        "turbResScale": 1e5,
        "eddyVisInfRatio": 0.2104,
        "solutionPrecision": "double",
        "gridPrecision": "double",
        "MGCycle": "sg",
        "nCycles": 0,
        "useANKSolver": False,
        "useNKSolver": False,
        "writeVolumeSolution": False,
        "writeSurfaceSolution": False,
        "writeTecplotSurfaceSolution": False,
        "monitorVariables": ["resrho", "resturb", "totalr"],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid-file", default="./meshes/comp_corner_21_fixed.cgns")
    p.add_argument("--restart-file",
                   default="./output_SAE/comp_corner_sa_edwards_000_vol.cgns")
    p.add_argument("--output-dir", default="./output_LST")
    p.add_argument("--csv", default="./output_LST/spectra_beta0.csv")
    p.add_argument("--panel", default="0", help="panel label for plot_stability.py (the betaL value)")
    # Hao: 50 eigenvalues, Krylov subspace 120, residual < 1e-9.
    p.add_argument("--nev", type=int, default=50)
    p.add_argument("--ncv", type=int, default=120)
    p.add_argument("--tol", type=float, default=1e-9)
    p.add_argument("--max-its", type=int, default=1000)
    # real, not complex: see the note in the module docstring
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--target", type=float, default=0.0)
    p.add_argument("--linear-solver", default="mumps",
                   choices=["mumps", "superlu_dist", "asm_ilu", "asm_lu"])
    p.add_argument("--eps-view-values", action="store_true", default=True,
                   help="ask SLEPc to print every converged eigenvalue, not "
                        "just the one the API returns")
    p.add_argument("--no-eps-view-values", dest="eps_view_values", action="store_false")
    p.add_argument("--mumps-icntl14", type=int, default=400,
                   help="MUMPS ICNTL(14), %% workspace growth over the analysis "
                        "estimate. Default 20 is not enough here (INFOG(1)=-9).")
    p.add_argument("--frozen-turbulence", action="store_true", default=False,
                   help="Hao linearizes the SA equation too; leave this OFF to match.")
    p.add_argument("--use-ad", action="store_true", default=False,
                   help="Build dR/dw by AUTOMATIC DIFFERENTIATION.  Leave this "
                        "OFF and pyADflow's default useAD=False applies, which "
                        "adjointUtils.F90:13-14 documents as 'if False, FD is "
                        "used' -- i.e. the operator being linearised is a "
                        "finite-difference Jacobian, and this script inherited "
                        "that silently.  Measured on grid 21 at target 6.8: FD "
                        "gives lambda 7.157 (omega_i L/u = 0.002100) with a "
                        "mode peaking on the ramp shoulder at x/L = +38.5; AD "
                        "gives 6.799 (0.001995, against Hao's 0.002) with the "
                        "peak moved into the interaction at x/L = +8.4.  AD is "
                        "the correct operator; turn it on.  NOTE it is not "
                        "sufficient -- both modes stay pinned within a few "
                        "cells of the wall (50%% of |v|^2 in 70 of 179712 "
                        "cells) rather than filling the bubble the way Hao's "
                        "figure 5(c-e) does, so the spectrum is still not his.")
    p.add_argument("--standard-evp", action="store_true", default=False,
                   help="solve the STANDARD EVP J_mod = inv(dU/dw)*(dR/dw) "
                        "instead of the generalized pencil.  This is the only "
                        "way to get shift != 0 without patching the fork: with "
                        "a B matrix, SLEPc's shift-invert forms J - shift*B in "
                        "STMatMAXPY_Private, and B carries nonzeros outside "
                        "J's preallocated pattern, so MatAXPY dies with "
                        "'New nonzero caused a malloc'.  With no B, the same "
                        "step is J_mod - shift*I, which only touches the "
                        "diagonal -- always allocated in a finite-volume "
                        "Jacobian -- so it goes through.")
    args = p.parse_args()

    comm = MPI.COMM_WORLD
    if comm.rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    comm.barrier()

    if not os.path.isfile(args.restart_file):
        raise FileNotFoundError(f"restart not found: {args.restart_file}. Run adflow_run2.py first.")

    # MUMPS needs a serial-safe ordering here; eigen_tests/common.py carries
    # the exact token list the shipped drivers use.  On top of that, ICNTL(14)
    # -- the percentage MUMPS is allowed to grow its internal work array beyond
    # the estimate from the analysis phase -- has to come up.  At the default
    # 20% this factorization dies with INFOG(1) = -9 ("main internal real
    # workarray S too small"); the interaction's Jacobian fills in far more
    # than the symbolic phase predicts.  The -st_ copy is the one that matters,
    # since the LU that actually gets built is SLEPc's shift-invert operator.
    tokens = list(mumps_safe_tokens(args.linear_solver == 'mumps'))
    if args.linear_solver == 'mumps':
        tokens += ["-mat_mumps_icntl_14", str(args.mumps_icntl14),
                   "-st_mat_mumps_icntl_14", str(args.mumps_icntl14)]
    if args.eps_view_values:
        # The API returns only eigenvalue 0, but SLEPc has all nConv of them in
        # the EPS object.  -eps_view_values makes SLEPc print the whole
        # converged set to stdout, which is a spectrum for the price of one
        # solve and needs no change to eigenSolveMatrix.F90.
        tokens += ["-eps_view_values", "::ascii_info_detail"]
    set_petsc_options_env(tokens, comm)
    ap = make_ap()
    solver = ADFLOW(options=build_options(args))
    solver.setAeroProblem(ap)

    # Scaling to figure 5's axes.  ADflow does NOT return lambda in 1/s: it
    # nondimensionalizes by the free-stream state (initializeFlow.F90:55-78),
    #     pRef = pInfDim,  rhoRef = rhoInfDim,  uRef = sqrt(pRef/rhoRef),
    #     timeRef = sqrt(rhoRef/pRef) = 1/uRef,
    # with a reference LENGTH of 1 m, because the grid is in metres.  So a
    # nondimensional rate becomes a physical one by multiplying by uRef, and
    # Hao's axis is that times L/u_inf with his L = 1 mm:
    #
    #     omega * L/u_inf = lambda_ADflow * uRef * L / u_inf
    #
    # Read the reference state off the solver rather than recomputing it, so
    # this cannot drift from whatever reference state ADflow actually built.
    # NOTE: flowVarRefState.F90 declares uRef, but the f2py layer does not
    # export it (dir(flowvarrefstate) has pref/rhoref/timeref/tref/muref/lref
    # and no uref) -- so build it from the members that ARE exported.  Both
    # routes below are exact identities in initializeFlow.F90, not fits:
    #     uRef = sqrt(pRef/rhoRef)   and   timeRef = sqrt(rhoRef/pRef) = 1/uRef
    # so they are cross-checked against each other, and a mismatch means the
    # reference state is not what this comment assumes.
    fvrs = solver.adflow.flowvarrefstate
    uRef = math.sqrt(float(fvrs.pref)/float(fvrs.rhoref))
    uRef_from_time = 1.0/float(fvrs.timeref)
    if abs(uRef - uRef_from_time) > 1e-8*max(uRef, 1.0):
        raise RuntimeError(
            f"uRef is ambiguous: sqrt(pRef/rhoRef) = {uRef!r} but "
            f"1/timeRef = {uRef_from_time!r}; the nondimensionalization is not "
            "the one this script's scaling assumes.")
    a_inf = math.sqrt(1.4*287.085*ap.T)
    u_inf = ap.mach*a_inf
    scale = uRef*L_REF/u_inf
    if comm.rank == 0:
        print("=" * 64)
        print(f"uRef = {uRef:.4f} m/s   u_inf = {u_inf:.3f} m/s")
        print(f"omega*L/u_inf = lambda_ADflow * {scale:.6e}")
        print(f"  (Hao's leading lowRe mode, omega_i L/u = 0.002, "
              f"sits at lambda_ADflow ~ {0.002/scale:.1f})")
        print(f"nev={args.nev} ncv={args.ncv} tol={args.tol:g} "
              f"shift={args.shift} solver={args.linear_solver}")
        print("=" * 64)

    res = solver.solveLSTEigenMatrix(
        frozenTurbulence=args.frozen_turbulence,
        negateJacobian=True,
        linearSolver=args.linear_solver,
        useAD=args.use_ad,
        useInvMassJacobianStandardEVP=args.standard_evp,
        shift=args.shift, target=args.target,
        nev=args.nev, ncv=args.ncv, maxIts=args.max_its, tol=args.tol,
    )

    # NOTE: this returns a DICT with a single "eigval", not a list.  nev/ncv
    # size the Krylov space SLEPc works in, but pyADflow.py only passes back
    # eigenvalue 0 -- the one nearest --target -- and eigenSolveMatrix.F90 only
    # prints that one too.  So one call is one eigenvalue, and a spectrum in
    # Hao's sense (his figure 5 shows ~50) needs either repeated calls at
    # different targets or the Fortran extended to loop EPSGetEigenpair over
    # all nConv.  Repeated calls currently need shift != 0, which trips
    # MatAXPY "Argument out of range" in STMatMAXPY_Private when SLEPc forms
    # J - shift*M: the mass matrix has nonzeros the Jacobian's preallocation
    # does not carry.  At shift = 0 that AXPY is skipped, which is why only
    # shift = 0 works today.
    if comm.rank == 0:
        lam = res["eigval"]
        print(f"converged {res['nConv']} eigenvalue(s) in {res['its']} iterations, "
              f"{res['solveTimeMinutes']:.2f} min   relError = {res['relError']:.3e}")
        print(f"lambda = {lam.real:.10e} {lam.imag:+.10e}j")
        row = dict(panel=args.panel,
                   omega_r=-lam.imag*scale,
                   omega_i=lam.real*scale,
                   lambda_re=lam.real, lambda_im=lam.imag,
                   rel_error=res["relError"], shift=args.shift,
                   n_conv=res["nConv"], solve_min=res["solveTimeMinutes"])
        write_header = not os.path.exists(args.csv)
        # append: a spectrum is built up one target at a time
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                w.writeheader()
            w.writerow(row)
        print(f"appended to {args.csv}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# Getting to beta*L = 0.36
# ---------------------------------------------------------------------------
# beta = 0.36/L_REF = 360 rad/m -> half-wavelength Lz = pi/beta = 8.727 mm
# between symmetry planes (u, v even and w odd in z, which is what a mirror
# plane admits).  mesh.py's --spanWidth would become 0.008727 and the span
# would need ~16-24 cells instead of 2 to resolve sin(beta*z).
#
# Cost, at comp_corner_21's 1152 x 156 in (x, y): 20 spanwise cells is
# 3.41M cells = 20.5M dofs.  eigen_tests/README.md puts the global complex LU
# (path 2, mumps) at a ~5M-dof memory wall, so that needs path 3 (asm_lu) or
# the real-mode time-stepper (tsa_cyl.py --pc-mode asm_ilu), and the base flow
# has to be re-converged on the 3-D grid first.
#
# NOTE ALSO that figure 5 is the paper's highRe case -- Re_delta = 132 840,
# M = 2.88, delta = 4.1 mm measured 8.04*delta upstream (p.3) -- and beta*L =
# 0.36 is where the highRe growth rate peaks (Hao, figure 4), and comp_corner_21
# IS the highRe case (M = 2.88, Re_delta = 132 840, delta = 4.1 mm at
# 8.04*delta), so beta*L = 0.36 is directly the right number here -- unlike on
# the lowRe grid, where it was borrowed from the wrong case.
# Running THIS base flow at beta*L = 0.36 is perfectly well posed, but it is a
# point on figure 4's lowRe curve, not a reproduction of figure 5(a).
