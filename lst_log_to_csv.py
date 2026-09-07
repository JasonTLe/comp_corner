#!/usr/bin/env python3
"""
lst_log_to_csv.py -- pull every converged eigenvalue out of an eigen_spectra.py
log and write plot_stability.py's spectra CSV.

Why this exists: solveLSTEigenMatrix returns ONE eigenvalue (pyADflow.py packs
a dict with a single "eigval"), but SLEPc converges many -- 17 on the first
beta = 0 run -- and prints them all when eigen_spectra.py passes
-eps_view_values.  So the log is the only place the full set appears until
eigenSolveMatrix.F90 is extended to loop EPSGetEigenpair over nConv.

Convention: ADflow returns lambda for perturbations ~ exp(lambda*t); Hao writes
exp[i*beta*z - i*(omega_r + i*omega_i)t], so omega_i = Re(lambda) and
omega_r = -Im(lambda).

SCALING.  lambda comes back in ADflow's internal nondimensionalization, not in
1/s.  ADflow scales by the free-stream state (initializeFlow.F90:55-78) with a
reference length of 1 m, so uRef = sqrt(pInfDim/rhoInfDim) and

    omega * L/u_inf = lambda_ADflow * uRef * L / u_inf

--scale is that factor; eigen_spectra.py prints it at startup ("omega*L/u_inf =
lambda_ADflow * ...").  Pass it, or leave it at 1.0 to keep raw ADflow units.

    python3 lst_log_to_csv.py logs/lst_native5.log out.csv "betaL = 0" 2.864919e-04
"""
import csv
import re
import sys

NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
ROW = re.compile(rf"^\s*({NUM})(?:\s*([-+]\s*{NUM})i)?\s*$")


def parse(path):
    vals, inside = [], False
    for line in open(path):
        if line.strip().startswith("Eigenvalues"):
            inside = True
            continue
        if inside:
            m = ROW.match(line)
            if not m:
                if line.strip() == "":
                    continue
                break                      # block ended
            re_, im_ = m.group(1), m.group(2)
            vals.append(complex(float(re_),
                                float(im_.replace(" ", "")) if im_ else 0.0))
    return vals


def main():
    log = sys.argv[1]
    out = sys.argv[2]
    panel = sys.argv[3] if len(sys.argv) > 3 else "0"
    scale = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
    vals = parse(log)
    if not vals:
        raise SystemExit(f"no 'Eigenvalues =' block found in {log}")
    rows = [dict(panel=panel, omega_r=-v.imag*scale, omega_i=v.real*scale,
                 lambda_re=v.real, lambda_im=v.imag) for v in vals]
    rows.sort(key=lambda r: -r["omega_i"])
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} eigenvalues -> {out}   (scale = {scale:g})")
    print(f"  most unstable: lambda = {rows[0]['lambda_re']:+.6f} "
          f"{rows[0]['lambda_im']:+.6f}j")
    print(f"  {sum(1 for r in rows if r['omega_i'] > 0)} with Re(lambda) > 0, "
          f"{sum(1 for r in rows if abs(r['lambda_im']) > 0)} oscillatory")


if __name__ == "__main__":
    main()
