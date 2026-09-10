#!/usr/bin/env python3
"""
eigvec_probe.py -- is a given eigenvalue a physical mode or numerical junk?

Every eigenvalue lambda has an eigenvector: the spatial SHAPE of the
perturbation that grows at rate lambda.  That shape is the test.  A physical
global mode lives where the physics is -- inside the separation bubble, along
the shear layer, around the shock foot (Hao's figure 5c-e).  A numerical mode
does not: it pins to a boundary, to the far field, or to a single layer of
cells, and often oscillates cell-to-cell.

This drives eigen_run.py's solver call, then writes the converged eigenvector
into a volume CGNS through the fork's own lstevec* volumeVariables
(inputParamRoutines.F90:2752-2770 -> LSTEvecDensity, LSTEvecVelocityX/Y/Z,
LSTEvecEnergyStagnationDensity), so the mode can be looked at on the mesh.

    mpiexec -n 16 python3 eigvec_probe.py --target 130 --tag t130
"""
import argparse, os, sys, math
from mpi4py import MPI
from adflow import ADFLOW
import flow_conditions

ET_ROOT = "/home/jason/packages/ADflowContribute/eigen_tests"
if ET_ROOT not in sys.path:
    sys.path.insert(0, ET_ROOT)
from common import mumps_safe_tokens, set_petsc_options_env  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--grid-file", default="./meshes/comp_corner_21_fixed.cgns")
p.add_argument("--restart-file", default="./output_SAE/comp_corner_sa_edwards_000_vol.cgns")
p.add_argument("--output-dir", default="./output_LST/evec")
p.add_argument("--target", type=float, required=True)
p.add_argument("--tag", required=True)
p.add_argument("--nev", type=int, default=50)
p.add_argument("--ncv", type=int, default=120)
p.add_argument("--tol", type=float, default=1e-9)
p.add_argument("--generalized", action="store_true",
               help="solve the GENERALIZED pencil (keep the mass matrix) instead of\n"
                    "the standard EVP J_mod=inv(M)J. Only works at shift=0.")
p.add_argument("--use-ad", action="store_true",
               help="build dR/dw by automatic differentiation instead of finite "
                    "differences (adjointUtils.F90:13-14). pyADflow defaults "
                    "useAD=False = FD, which eigen_run.py silently inherits.")
args = p.parse_args()

comm = MPI.COMM_WORLD
if comm.rank == 0:
    os.makedirs(args.output_dir, exist_ok=True)
comm.barrier()

tokens = list(mumps_safe_tokens(True)) + [
    "-mat_mumps_icntl_14", "400", "-st_mat_mumps_icntl_14", "400"]
set_petsc_options_env(tokens, comm)

opts = {
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
    "writeVolumeSolution": True,
    "writeSurfaceSolution": False,
    "writeTecplotSurfaceSolution": False,
    # the fork's eigenvector output fields
    "volumeVariables": ["lstevecrho", "lstevecvelx", "lstevecvely",
                        "lstevecvelz", "lstevecrhoe", "mach", "temp"],
    "monitorVariables": ["resrho", "resturb", "totalr"],
}

ap = flow_conditions.make_ap("comp_corner_sa_edwards")
solver = ADFLOW(options=opts)
solver.setAeroProblem(ap)

fvrs = solver.adflow.flowvarrefstate
uRef = math.sqrt(float(fvrs.pref)/float(fvrs.rhoref))
u_inf = ap.mach*math.sqrt(1.4*287.085*ap.T)
scale = uRef*flow_conditions.L_REF/u_inf

binf = os.path.join(args.output_dir, f"evec_{args.tag}.bin")
res = solver.solveLSTEigenMatrix(
    useAD=args.use_ad,
    frozenTurbulence=False, negateJacobian=True, linearSolver="mumps",
    useInvMassJacobianStandardEVP=(not args.generalized),
    shift=args.target, target=args.target,
    nev=args.nev, ncv=args.ncv, maxIts=1000, tol=args.tol,
    eigenvectorFile=binf,
)
lam = res["eigval"]
if comm.rank == 0:
    print("=" * 64)
    print(f"target {args.target}  ->  lambda = {lam.real:.6f}{lam.imag:+.6f}j")
    print(f"omega_i L/u = {lam.real*scale:+.6f}   omega_r L/u = {-lam.imag*scale:+.6f}")
    print(f"nConv={res['nConv']} relError={res['relError']:.3e}")
    print("=" * 64)

# push the eigenvector into LSTEvecReal so the CGNS writer emits it
solver.readAndSetLSTEigenvector(binf)
cgns = os.path.join(args.output_dir, f"evec_{args.tag}_vol.cgns")
solver.writeVolumeSolutionFile(cgns, writeGrid=True)
if comm.rank == 0:
    print(f"wrote {cgns}")
