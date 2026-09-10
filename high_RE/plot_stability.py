#!/usr/bin/env python3
"""
plot_stability_figures.py
===========================

Reproduce the global-stability-analysis figures of Hao, JFM 2023, 971 A28,
from your own eigenvalue/growth-rate data. Each subcommand writes a PNG.

    python plot_stability_figures.py spectra example_eigenvalue_spectra.csv
    python plot_stability_figures.py spectra logs/lst_native5.log --panel "0" --scale 2.864919e-04
    python plot_stability_figures.py growth  example_growth_rate_wavenumber.csv
    python plot_stability_figures.py log2csv logs/lst_native5.log out.csv "betaL = 0" 2.864919e-04


spectra   Figure 5(a,b): eigenvalue spectra, ω_i L/u∞ vs ω_r L/u∞, one
          scatter panel per case (e.g. per spanwise wavenumber βL).

          Input is either a CSV with columns panel,omega_r,omega_i
              panel     label for the case this eigenvalue belongs to
                        (e.g. the βL value) -- one subplot per unique
                        value, in first-seen order
              omega_r   real part of the eigenvalue, ω_r L/u∞
              omega_i   imaginary part (growth rate), ω_i L/u∞
          or a raw eigen_spectra.py log file (single panel, parsed the same
          way as log2csv -- see --panel/--scale/--csv-out below).

          NOTE: figure 5's panels (c-e) are eigenfunction contours over the
          base-flow mesh (a scalar field on (x, y)), not a list of
          eigenvalues -- a different data product from a stability solver's
          eigenvalue output, and not reproduced here.

growth    Figure 6(a,b): growth rate and spanwise wavenumber/wavelength of
          the most unstable mode vs ramp angle α, one line per case (e.g.
          per Reynolds number). Panel (a) is growth rate vs α; panel (b) is
          βL (left axis, black) and 2π/(βL_sep) (right axis, magenta) vs α.

          Input CSV columns: case,alpha,omega_i,beta_L,wavelength_Lsep
              case               label for the case (e.g. "Re_delta=63560")
              alpha              ramp angle, degrees
              omega_i            growth rate of the most unstable mode
              beta_L             spanwise wavenumber of that mode, βL
              wavelength_Lsep    wavelength normalised by the separation
                                 length, 2π/(βL_sep)

          Rows are sorted by alpha within each case, so input order does
          not matter.

log2csv   Pull every converged eigenvalue out of an eigen_spectra.py log and
          write a spectra CSV (the format the spectra subcommand reads).

          Why this exists: solveLSTEigenMatrix returns ONE eigenvalue
          (pyADflow.py packs a dict with a single "eigval"), but SLEPc
          converges many -- 17 on the first beta = 0 run -- and prints them
          all when eigen_spectra.py passes -eps_view_values. So the log is
          the only place the full set appears until eigenSolveMatrix.F90 is
          extended to loop EPSGetEigenpair over nConv.

          Convention: ADflow returns lambda for perturbations ~ exp(lambda*t),
          Hao writes exp[i*beta*z - i*(omega_r + i*omega_i)t], so
          omega_i = Re(lambda) and omega_r = -Im(lambda).

          SCALING. lambda comes back in ADflow's internal
          nondimensionalization, not in 1/s. ADflow scales by the
          free-stream state (initializeFlow.F90:55-78) with a reference
          length of 1 m, so uRef = sqrt(pInfDim/rhoInfDim) and

              omega * L/u_inf = lambda_ADflow * uRef * L / u_inf

          --scale is that factor; eigen_spectra.py prints it at startup
          ("omega*L/u_inf = lambda_ADflow * ..."). Pass it, or leave it at
          1.0 to keep raw ADflow units.

All subcommands need only numpy and matplotlib. See
example_eigenvalue_spectra.csv and example_growth_rate_wavenumber.csv for
runnable samples.
"""
import argparse
import csv
import re
import sys
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

MARKERS = ['o', 'D', 's', '^', 'v', 'P']
MAGENTA = '#e6007e'

NUM = r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?"
ROW = re.compile(rf"^\s*({NUM})(?:\s*([-+]\s*{NUM})i)?\s*$")


# ============================================================ log2csv (eigenvalues)

def parse_log(path):
    """Pull the complex eigenvalues out of an eigen_spectra.py 'Eigenvalues =' block."""
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


def eigenvalues_to_rows(vals, panel, scale):
    rows = [dict(panel=panel, omega_r=-v.imag*scale, omega_i=v.real*scale,
                 lambda_re=v.real, lambda_im=v.imag) for v in vals]
    rows.sort(key=lambda r: -r["omega_i"])
    return rows


def cmd_log2csv(args):
    vals = parse_log(args.log)
    if not vals:
        raise SystemExit(f"no 'Eigenvalues =' block found in {args.log}")
    rows = eigenvalues_to_rows(vals, args.panel, args.scale)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} eigenvalues -> {args.out}   (scale = {args.scale:g})")
    print(f"  most unstable: lambda = {rows[0]['lambda_re']:+.6f} "
          f"{rows[0]['lambda_im']:+.6f}j")
    print(f"  {sum(1 for r in rows if r['omega_i'] > 0)} with Re(lambda) > 0, "
          f"{sum(1 for r in rows if abs(r['lambda_im']) > 0)} oscillatory")


# ============================================================== spectra (fig 5)

def read_spectra(path):
    """Group (omega_r, omega_i) pairs by panel label, preserving first-seen order."""
    panels = OrderedDict()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        missing = {'panel', 'omega_r', 'omega_i'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError('%s is missing column(s): %s' % (path, ', '.join(sorted(missing))))
        for row in reader:
            key = row['panel'].strip()
            panels.setdefault(key, {'omega_r': [], 'omega_i': []})
            panels[key]['omega_r'].append(float(row['omega_r']))
            panels[key]['omega_i'].append(float(row['omega_i']))
    if not panels:
        raise ValueError('%s has no data rows' % path)
    return panels


def figure_eigenvalue_spectra(panels, xlim=None, ylim=None, panel_title_fmt='βL = {panel}'):
    """One eigenvalue-spectrum scatter subplot per panel label, side by side."""
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.2*n, 4.4), squeeze=False)
    axes = axes[0]

    letters = 'abcdefghij'
    for ax, letter, (label, xy) in zip(axes, letters, panels.items()):
        wr, wi = np.asarray(xy['omega_r']), np.asarray(xy['omega_i'])
        ax.axhline(0, ls='--', color='k', lw=0.8)
        ax.axvline(0, ls='--', color='k', lw=0.8)
        ax.scatter(wr, wi, s=45, facecolors='none', edgecolors='k', linewidths=1.1)

        ax.text(-0.16, 1.05, '(%s)' % letter, transform=ax.transAxes,
                fontsize=13, fontweight='bold', va='bottom')
        ax.text(0.06, 0.85, panel_title_fmt.format(panel=label), transform=ax.transAxes,
                fontsize=11, va='top',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5))

        # Cap the tick count.  Autoscaled spectra land on ranges like
        # +-0.0018, where matplotlib's default locator emits ~9 ticks labelled
        # '-0.00150' etc. and neighbouring labels run into each other.  Hao's
        # figure 5 uses 5 x-ticks and 6 y-ticks; match that and the labels have
        # room whatever the range turns out to be.
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.set_xlabel(r'$\omega_r L/u_\infty$')
        ax.set_ylabel(r'$\omega_i L/u_\infty$')
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.tight_layout()
    return fig


def cmd_spectra(args):
    if args.input.endswith('.log'):
        vals = parse_log(args.input)
        if not vals:
            raise SystemExit(f"no 'Eigenvalues =' block found in {args.input}")
        rows = eigenvalues_to_rows(vals, args.panel, args.scale)
        panels = OrderedDict()
        panels[args.panel] = {'omega_r': [r['omega_r'] for r in rows],
                               'omega_i': [r['omega_i'] for r in rows]}
        if args.csv_out:
            with open(args.csv_out, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            print('wrote', args.csv_out)
    else:
        panels = read_spectra(args.input)
    fig = figure_eigenvalue_spectra(panels, xlim=args.xlim, ylim=args.ylim,
                                     panel_title_fmt=args.panel_title)

    out = args.output or (args.input.rsplit('.', 1)[0] + '_spectra.png')
    fig.savefig(out, dpi=200)
    print('wrote', out)
    for label, xy in panels.items():
        wi = np.asarray(xy['omega_i'])
        i = int(np.argmax(wi))
        print('  panel %-8s  %d eigenvalues, most unstable: omega_r=%.4f omega_i=%.4f'
              % (label, len(wi), xy['omega_r'][i], wi[i]))


# ================================================================ growth (fig 6)

def read_growth_data(path):
    """Group rows by case, each case sorted by alpha."""
    cases = OrderedDict()
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        needed = {'case', 'alpha', 'omega_i', 'beta_L', 'wavelength_Lsep'}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise ValueError('%s is missing column(s): %s' % (path, ', '.join(sorted(missing))))
        for row in reader:
            key = row['case'].strip()
            cases.setdefault(key, {'alpha': [], 'omega_i': [], 'beta_L': [], 'wavelength_Lsep': []})
            for col in ('alpha', 'omega_i', 'beta_L', 'wavelength_Lsep'):
                cases[key][col].append(float(row[col]))
    if not cases:
        raise ValueError('%s has no data rows' % path)
    for key, d in cases.items():
        order = np.argsort(d['alpha'])
        for col in d:
            d[col] = np.asarray(d[col])[order]
    return cases


def figure_growth_rate_wavenumber(cases):
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    # ---- panel (a): growth rate vs alpha ----
    axa.axhline(0, ls='--', color='k', lw=0.8)
    for marker, (label, d) in zip(MARKERS, cases.items()):
        axa.plot(d['alpha'], d['omega_i'], marker=marker, mfc='none', mec='k',
                  color='k', lw=1.0, label=label)
    axa.set_xlabel(r'$\alpha$ (deg.)')
    axa.set_ylabel(r'$\omega_i L/u_\infty$')
    axa.text(-0.16, 1.05, '(a)', transform=axa.transAxes, fontsize=13, fontweight='bold', va='bottom')
    axa.legend(frameon=False, fontsize=9, loc='lower right')
    axa.spines['top'].set_visible(False)
    axa.spines['right'].set_visible(False)

    # ---- panel (b): beta*L (left, black) and wavelength (right, magenta) ----
    axb2 = axb.twinx()
    for marker, (label, d) in zip(MARKERS, cases.items()):
        axb.plot(d['alpha'], d['beta_L'], marker=marker, mfc='none', mec='k',
                  color='k', lw=1.0)
        axb2.plot(d['alpha'], d['wavelength_Lsep'], marker=marker, mfc='none',
                   mec=MAGENTA, color=MAGENTA, lw=1.2)
    axb.set_xlabel(r'$\alpha$ (deg.)')
    axb.set_ylabel(r'$\beta L$')
    axb2.set_ylabel(r'$2\pi/(\beta L_{sep})$', color=MAGENTA)
    axb2.tick_params(axis='y', colors=MAGENTA)
    axb.text(-0.16, 1.05, '(b)', transform=axb.transAxes, fontsize=13, fontweight='bold', va='bottom')
    axb.spines['top'].set_visible(False)

    fig.tight_layout()
    return fig


def cmd_growth(args):
    cases = read_growth_data(args.input)
    fig = figure_growth_rate_wavenumber(cases)

    out = args.output or (args.input.rsplit('.', 1)[0] + '_growth.png')
    fig.savefig(out, dpi=200)
    print('wrote', out)
    for label, d in cases.items():
        i = int(np.argmax(d['omega_i']))
        print('  case %-20s  peak growth rate omega_i=%.4f at alpha=%.1f deg'
              % (label, d['omega_i'][i], d['alpha'][i]))


# =================================================================== command line

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)

    ps = sub.add_parser('spectra', help='figure 5(a,b): eigenvalue spectra')
    ps.add_argument('input', help='CSV file with columns panel,omega_r,omega_i, '
                                   'or a raw eigen_spectra.py .log file (single panel)')
    ps.add_argument('-o', '--output', default=None,
                     help='output PNG path (default: <input>_spectra.png)')
    ps.add_argument('--xlim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'))
    ps.add_argument('--ylim', nargs=2, type=float, default=None, metavar=('MIN', 'MAX'))
    ps.add_argument('--panel-title', default='βL = {panel}',
                     help='subplot title, {panel} is replaced by the panel label '
                          '(default: "βL = {panel}")')
    ps.add_argument('--panel', default='0',
                     help='panel label to use when input is a .log file (default: "0")')
    ps.add_argument('--scale', type=float, default=1.0,
                     help='lambda_ADflow -> omega*L/u_inf scale factor when input is a '
                          '.log file (default: 1.0, i.e. raw ADflow units)')
    ps.add_argument('--csv-out', default=None,
                     help='also write the parsed eigenvalues to this CSV when input is a .log file')
    ps.set_defaults(func=cmd_spectra)

    pg = sub.add_parser('growth', help='figure 6(a,b): growth rate and wavenumber vs ramp angle')
    pg.add_argument('input', help='CSV file with columns case,alpha,omega_i,beta_L,wavelength_Lsep')
    pg.add_argument('-o', '--output', default=None,
                     help='output PNG path (default: <input>_growth.png)')
    pg.set_defaults(func=cmd_growth)

    pl = sub.add_parser('log2csv', help='pull eigenvalues out of an eigen_spectra.py log into a spectra CSV')
    pl.add_argument('log', help='eigen_spectra.py log file')
    pl.add_argument('out', help='output CSV path')
    pl.add_argument('panel', nargs='?', default='0', help='panel label (default: "0")')
    pl.add_argument('scale', nargs='?', type=float, default=1.0,
                     help='lambda_ADflow -> omega*L/u_inf scale factor (default: 1.0)')
    pl.set_defaults(func=cmd_log2csv)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    sys.exit(main())
