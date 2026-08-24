# Converging `comp_corner_15_fixed.cgns` (stages S1 + S2)

Date: 2026-08-13. Machine: `JasonLenovo`, i7-14700HX, 14 physical cores / 28 threads,
15 GB RAM. ADflow 2.13.1 (`~/packages/ADflowContribute`), PETSc 3.21.6, python3.11.

This is a record of what was wrong, how each problem was identified, and what was
changed. Logs referenced here are all under `logs/`.

---

## 0. TL;DR — what changed

| File | Change | Why |
|---|---|---|
| `comp_corner_S1.py` | `args.output` → `outputDirectory` | `NameError`: `--output` had been deleted from argparse but was still referenced |
| `comp_corner_S2.py` | added `outputDirectory`, restored `"restartFile": args.restartFile` | same `NameError`, **plus** `restartFile` was pointing at the `./output_SAE` *directory* instead of the restart file |
| `comp_corner_S1.py` | `ILUFill: 3` → `ANKPCILUFill: 2`, `NKPCILUFill: 2` | `ILUFill` is the *adjoint* solver's knob and was a no-op in a flow-only run |
| `comp_corner_S1.py` | `MGStartLevel: 1` → `2` | fine-grid ANK from freestream blows up on this grid; start coarse and prolong |
| `comp_corner_S1.py` | `ANKCFL0: 5.0` (default) → `1.0` | grid 15's first cell is 1.7× finer than grid 14's; needs more pseudo-transient regularisation to start |
| `comp_corner_S1.py` | `CFL 0.5 / CFLCoarse 0.3` → `1.0 / 1.0` | `CFLCoarse` now actually does the startup work; `CFL 5.0` was verified to NaN |
| `comp_corner_S2.py` | added `useBlockettes: False` | **mandatory** with `saVariant` + ANK/NK, or the turbulence model silently reverts to standard SA |
| `comp_corner_S2.py` | `useNKSolver: True`, `NKSwitchTol: 1e-6` | DADI alone stalls at `totalRes 4.64e-3`; NK finishes in 8 iterations |
| run command | `mpiexec -n 12` → `-n 14` | 12 procs cannot coarsen this grid at all (hard abort) |

Both stages now converge: **S1 in 563 s, S2 in 257 s**, on 14 cores.

---

## 1. Both scripts were broken before they reached the solver

The working tree had `--output` removed from `argparse` in both scripts but the
variable was still used:

* `comp_corner_S1.py:127` — `os.path.join(args.output, ...)`, after the solve, so
  S1 would have run for minutes and then crashed while printing its summary.
* `comp_corner_S2.py:65` — `os.path.exists(args.output)`, at the very top, so S2
  crashed instantly.

S2 had a second, more damaging bug in the same edit — the restart path and the
output directory had been swapped:

```python
"restartFile": "./output_SAE",   # a DIRECTORY, not the restart file
"outputDirectory": args.output,  # NameError
```

Both are fixed. S2 now reads `args.restartFile` and writes to `./output_SAE`.

---

## 2. `ILUFill` was doing nothing (teaching note)

`comp_corner_S1.py` carried `"ILUFill": 3` with the comment *"your steering
system … controls your CFL/timesteps"*. Neither half of that is true.

ADflow builds **three** independent linear systems and each has its own private
copy of the same preconditioner knobs:

| Option | Fortran symbol | Which linear system |
|---|---|---|
| `ANKPCILUFill` | `anksolver::ANK_iluFill` (`NKSolvers.F90:1668,2027`) | ANK's pseudo-transient system |
| `NKPCILUFill` | `nksolver::NK_iluFill` (`NKSolvers.F90:37,423`) | NK's true Newton system |
| `ILUFill` | `inputADjoint::fillLevel` (`adjointAPI.F90:917`) | **the adjoint solve only** |

The bare `ILUFill` maps to `["adjoint", "filllevel"]` (`pyADflow.py:6869`) and in
this build is read only by the adjoint KSP setup. These scripts run no adjoint, so
the option never did anything — and 3 vs the default 2 was moot for a second
reason: the grid-14 solve that everyone treats as the working baseline was
therefore running at ILU fill **2** on both ANK and NK, not 3.

What the knob actually is: the level-of-fill `k` of an incomplete LU factorisation
(`PCFactorSetLevels`, `adjointUtils.F90:1559`), used as the *local* per-subdomain
solver inside additive Schwarz. ILU(0) keeps the sparsity of the matrix; ILU(k)
keeps fill entries within `k` elimination steps of an original nonzero. Higher `k`
= fewer Krylov iterations, more memory, slower factorisation. It has no
relationship to CFL or timestep size whatsoever.

Replaced with explicit `ANKPCILUFill: 2` / `NKPCILUFill: 2` so the effective value
is visible and tunable.

Related dead knobs in S1, for the same reason (the option exists but the code path
never runs):

* `CFL`, `CFLCoarse` and `nSubiterTurb` were all inert in the *old* S1, because
  `ANKSwitchTol: 1e11` handed the fine grid to ANK at iteration 1 and the
  DADI/multigrid smoother never executed. `nSubiterTurb` drives `turbAPI.F90:35`
  (smoother path); ANK's turbulence sweeps are the separate `ANKNSubiterTurb`
  (`NKSolvers.F90:3374`). After the changes in §5, `CFLCoarse` is live again — it
  is what runs the level-2 startup.
* `MGCycle` is the exception: it too had no effect on *convergence* (the coarse
  level was allocated and never used), but it is emphatically not inert — it is
  what imposes the coarsening constraint that makes 12 procs abort outright (§4).

---

## 3. Grid 15 is not a refinement of grid 14 — it is a different grid

This matters, because the grid-14 solver recipe was tuned to grid 14.

| | `comp_corner_14_fixed` | `comp_corner_15_fixed` |
|---|---|---|
| node dims | 769 × 129 × 3 | 701 × 201 × 3 |
| cells | 196,608 | 280,000 |
| wall located at | `j = jmin` | **`j = jmax`** |
| streamwise index | `i` increases downstream | **`i` *decreases* downstream** |
| inflow (farfield) | `i = imin` | `i = imax` |
| outflow | `i = imax` | `i = imin` |
| corner at | wall node i=550 | wall node i=233 (x=0, y=0) |
| ramp angle | — | 25.0° |
| domain x | −0.1258 … +0.0400 m | −0.0404 … +0.0182 m |
| first off-wall spacing | 1.60e-6 … 1.68e-6 m | **9.56e-7 … 1.08e-6 m** |
| wall Δx | 2.02e-4 … 2.29e-4 m | 8.63e-5 m |
| wall-cell aspect ratio | ≤ 143 | ≤ 90 |
| j growth ratio | 1.052 … 1.079 | 1.001 … 1.032 |
| isothermal wall T | 275.4 K | 275.4 K |

Both grids are single-zone, both carry the same 275.4 K isothermal wall data
(checked — this was a suspect and was cleared), and neither has inverted cells.
Grid 15 is the *better* grid geometrically: lower aspect ratio, gentler stretching,
1.7× finer first cell. The index reversal is harmless — `i` and `j` are both
flipped, so handedness is preserved.

The consequential differences are (a) the finer first cell, which is what breaks
the startup (§5), and (b) the cell counts, which is what breaks the partitioning
(§4). Note also the domain is ~3× shorter, so the boundary layer arriving at the
corner is a different one — S1/S2 results on grid 15 are not directly comparable
to grid 14 results.

---

## 4. The run aborted immediately: 12 procs cannot coarsen this grid

```
Run-time error in procedure checkCoarse1to1
Error message: Non-matching block-to-block face on zone blk-1
               ...Support not implemented yet.
```

Grid 15 is a single block. ADflow splits it into `nProc` sub-blocks
(`loadBalance.F90:2790 splitBlock`), and every cut creates an internal 1-to-1
interface. `MGCycle: "2w"` needs one level of coarsening, and each of those
interfaces must still be 1-to-1 on the coarse grid or `coarseUtils.F90:1528`
hard-aborts. With 700 × 200 × 2 cells and 12 sub-blocks the splitter cannot find
a cut set that satisfies this; with 768 × 128 × 2 (grid 14, both powers-of-two
friendly) it always could, which is why nobody hit this before.

Probed directly (`_probe_partition.py`, deleted after use):

| nProc | `sg` | `2w` |
|---|---|---|
| 4 | OK | **abort** |
| 8 | OK | **abort** |
| 10 | OK | OK |
| 12 | OK | **abort** |
| 14 | OK | OK |
| ≥16 | — | — (OpenMPI: only 14 slots; 28 "CPUs" are 14 cores × 2 threads) |

**Use `mpiexec -n 14`.** It is the only count that both keeps multigrid and uses
every physical core. This is recorded in the S1 and S2 docstrings.

(`MGCycle: "sg"` would also dodge the abort at any proc count, and costs S1
nothing since ANK owns the fine grid — but S2 is a pure DADI run where multigrid
is doing real work, so it is worth keeping 2w for both stages.)

---

## 5. The real problem: ANK cannot start on grid 15 from freestream

With 14 procs the run started and then died a different death
(`logs/S1_m15_run2.log`):

```
 iter  type   CFL      Step  Res rho      totalRes     Y+_max
    0  None   ----     ----  6.151e+04    3.738e+08    3.001e+00
    7  *ANK   2.50E+00 0.01  9.137e+04    5.017e+08    9.103e+00
    8   ANK   2.50E+00 0.07  1.009e+05    5.430e+08    5.297e+03   <-- blow-up
   ...
  160   ANK   6.31E-01 0.00  1.015e+05    6.029e+08    6.502e+03   <-- frozen
```

Y+_max jumps from 9 to 5297 in one iteration, the residual ends up **above** where
it started, and from iteration ~65 onwards the ANK line search rejects every step
(`Step 0.00`) with the state completely frozen. Compare grid 14, where Y+ never
leaves 3–6 and `Step` reaches 0.93 by iteration 6.

A y+ of 6500 in the first off-wall cell implies a friction velocity of ~3e5 m/s —
i.e. one or more near-wall cells has gone unphysical. ANK's line search then
correctly refuses to move, but the Jacobian it is building is around a garbage
state, so it can never recover. This is a startup failure, not slow convergence;
no amount of extra iterations helps.

**Mechanism.** ANK is pseudo-transient continuation: it solves
`(I·V/(CFL·Δt) + ∂R/∂w) Δw = −R`. The `CFL` here is `ANKCFL0` (default **5.0**),
and it sets how much the diagonal regularises the Newton step — high CFL means a
small diagonal, i.e. nearly a full Newton step. A full Newton step taken from a
uniform freestream initial condition is a wild extrapolation. Grid 15's first
off-wall cell is 9.56e-7 m against grid 14's 1.60e-6 m, so its wall-normal
Jacobian entries are stiffer, and the same `ANKCFL0` that grid 14 tolerated
overshoots here.

### The sweep

Five startup strategies, ~250 cycles each, all at `-n 14` (`logs/sweep/`):

| tag | change from baseline | totalRes after ~250 cycles | rejected steps | verdict |
|---|---|---|---|---|
| — | baseline (grid-14 recipe) | 6.03e+08 (**above** 3.74e8 start) | nearly all | dead |
| B | `MGStartLevel: 2` | 1.19e+07 | 0 | healthy |
| C | `ANKSwitchTol 0.1`, fine DADI at `CFL 5.0` first | **NaN on iteration 2** | — | catastrophic |
| D | `ANKCFL0: 1.0` | 1.17e+08 | 0 | healthy |
| E | `MGStartLevel 2` + `CFLCoarse 1.0` + `ANKCFL0 1.0` | **1.02e+06** | 0 | best |
| F | as C but two orders of smoother | **NaN on iteration 2** | — | catastrophic |

Reading it:

* **D alone works.** Just lowering `ANKCFL0` from 5.0 to 1.0 is enough to make
  the startup survivable — `Step` goes 0.37 → 0.65 → 0.91 in three iterations.
  This confirms the mechanism above.
* **B alone works** and is stronger than D. Starting on the coarse grid means the
  violent transient is resolved on a grid with 4× fewer cells and a 2× larger
  first cell, where it is survivable; the prolonged solution is then a good
  initial guess.
* **E, the combination, is 2.6 orders down in the same budget** and hands the
  fine grid to ANK with `Step 0.90` and `CFL 7.67` on the very first fine
  iteration. That is what was adopted.
* **C and F are a warning, not a fix.** Fine-grid DADI at `CFL 5.0` from a uniform
  freestream produces `totalRes 4.8e51` on iteration 1 and `NaN` on iteration 2.
  E only escapes this because after prolongation the residual is already below
  the ANK switch tolerance, so fine-grid DADI never actually executes — verified,
  zero fine-grid DADI iterations in E's log. That is too fragile to ship, so the
  adopted config sets `ANKSwitchTol: 1e11` (ANK owns the fine grid unconditionally)
  **and** drops `CFL` to 1.0 as a second line of defence.

### Adopted S1 configuration

```python
"MGCycle":      "2w",
"MGStartLevel": 2,      # start coarse, prolong to fine
"CFL":          1.0,    # fine-grid DADI -- should never run; 5.0 NaNs if it does
"CFLCoarse":    1.0,    # this is what actually runs the level-2 startup
"ANKSwitchTol": 1e11,   # ANK owns the fine grid from iteration 1
"ANKCFL0":      1.0,    # was 5.0; too aggressive for this grid's near-wall spacing
"ANKPCILUFill": 2,
"NKPCILUFill":  2,
```

Everything else — `ANKSecondOrdSwitchTol 1e-3`, `ANKCoupledSwitchTol 1e-16`,
`ANKNSubiterTurb 3`, `NKSwitchTol 1e-4`, `eddyVisInfRatio 0.2104` — is unchanged
from the grid-14 recipe.

---

## 6. Results

### S1 — converged (`logs/S1_m15_final.log`)

`mpiexec -n 14 python3.11 comp_corner_S1.py --gridFile ./meshes/comp_corner_15_fixed.cgns`

**563 s wall clock on 14 cores.** Free-stream residual `totalR0 = 3.74e8`, so
`L2Convergence: 1e-10` asks for `totalRes <= 3.74e-2`; the run finished at
`1.19e-2`.

| phase | iters | totalRes | Y+_max |
|---|---|---|---|
| level 2 (coarse) start | 0 | 8.797e+07 | 4.242 |
| level 2 (coarse) end, DADI @ CFLCoarse 1.0 | 500 | 1.199e+06 | 1.358 |
| prolonged to level 1 | 0 | 6.355e+06 | 0.959 |
| ANK, first order | 1–104 | → 4.31e+05 | 0.762 |
| SANK, second order (`ANKSecondOrdSwitchTol 1e-3`) | 105–299 | → 3.32e+04 | 0.762 |
| NK, true Newton (`NKSwitchTol 1e-4` → 3.74e4) | 300–376 | → **1.191e-02** | 0.762 |

Final residuals: `res rho = 2.66e-06` (10.4 orders below the initial 6.15e+04),
`res nuturb = 2.42e-11`, `Y+_max = 0.7617`.

Two things worth noting in that history:

* The ANK line search never rejected a step (`Step 0.00` never appears on level 1),
  in contrast to the failed run where it rejected essentially all of them.
* SANK **plateaus around totalRes 1.0–1.55e5 for roughly 80 iterations**
  (iters 129–209) with `res nuturb` climbing 1.7e-3 → 2.5e-3, then resumes
  descending on its own. This is the segregated mean-flow/turbulence lag, and on
  this grid it is a plateau, not a limit cycle — do not intervene, it clears.
  Once past it, NK closes the last 6 orders in 77 iterations.

Output: `output_SA/comp_corner_sa_000_vol.cgns` (64.7 MB, double precision),
`output_SA/comp_corner_sa_000_surf.cgns`.

**Physics check.** Converged is not the same as correct, so the wall skin friction
was pulled out of the surface file (`SkinFrictionX` on `NSWallIsothermalBCZone7/8`)
and checked for the shock-wave/boundary-layer interaction that a 25° compression
corner at Mach 2.95 must produce:

| quantity | value |
|---|---|
| Cf,x on the upstream plate (x = −30 mm) | +2.85e-3 |
| **separation** (Cf,x → negative) | x = **−3.63 mm** |
| Cf,x minimum (reversed flow) | −2.03e-3 at x = +0.55 mm |
| **reattachment** (Cf,x → positive) | x = **+2.90 mm** |
| separation bubble length | **6.52 mm** |
| Cf,x on the ramp (x = +10 mm) | +3.52e-3 |

A single closed separation bubble straddling the corner (x = 0), attached flow on
both the incoming plate and the downstream ramp. That is the expected structure,
so the converged field is a real solution and not a numerical artefact.

### S2 — SA-Edwards restart

`mpiexec -n 14 python3.11 comp_corner_S2.py --gridFile ./meshes/comp_corner_15_fixed.cgns --restartFile ./output_SA/comp_corner_sa_000_vol.cgns`

S2's `CFL: 5.0` DADI is **safe here** even though fine-grid DADI at CFL 5.0 NaN'd
in sweep configs C/F (§5). Those started from a uniform freestream; S2 starts from
S1's converged field, which is an entirely different conditioning problem. Verified
in the log — the first cycles are stable.

The documented grid-14 pattern reproduces exactly: S1 hands over at
`totalRes 1.19e-2`, cycle 1 switches on the Edwards source term and throws the
residual up to `8.23e+04`, and DADI then has to walk it back down. Same `totalR0`
(3.74e8), so `L2Convergence: 1e-12` asks for `totalRes <= 3.74e-4`.

#### First attempt: DADI alone stalls (`logs/S2_m15_run1.log`)

It ran the full 40,000 cycles in 4,249 s and **did not converge** — final
`totalRes 4.64e-3` against the 3.74e-4 target. Crucially this is a *stall*, not a
budget problem. The mean flow is fully converged; the turbulence is not:

| iter | res rho | res nuturb | totalRes |
|---|---|---|---|
| 9,999 | 1.51e-04 | 8.40e-10 | 6.94e-01 |
| 17,499 | 4.04e-06 | **9.997e-11** | 2.01e-02 |
| 27,499 | 3.55e-08 | **9.31e-11** | 4.93e-03 |
| 39,999 | 4.41e-10 | **8.77e-11** | 4.64e-03 |

`res rho` falls 14 orders. `res nuturb` moves 12% in the last 22,500 cycles.
`totalRes` tracks `res nuturb` exactly once the mean flow drops out, so the
turbulence residual floor *is* the convergence limit. A control run of 400 extra
DADI cycles from that state moved `totalRes` only 4.6410e-3 → 4.6325e-3.

#### The `useBlockettes` trap

The fix is to let NK finish, but doing that naively breaks the physics. **There are
two SA implementations in this fork and only one honours `saVariant`:**

| implementation | Edwards terms? |
|---|---|
| `src/turbulence/sa.F90` → `sa_block` → `saSource`/`saViscous` | **yes** — `useSAEdwards` decoded at `sa.F90:150-151`, used at 212, 290, 300, 326 |
| `src/NKSolver/blockette.F90` → its own private `saSource`/`saAdvection`/`saViscous` (976, 1170, 1392) | **no** — zero references to `saVariant` or Edwards anywhere in the file |

ANK and NK evaluate residuals through `blocketteRes`, which dispatches on
`useBlockettes` (`blockette.F90:271-275`):

```fortran
if (useBlockettes) then         ! DEFAULT IS TRUE
    call blocketteResCore(...)  ! -> blockette's private, standard-SA routines
else
    call blockResCore(...)      ! -> sa_block from sa.F90, Edwards honoured
```

So **turning ANK/NK on with the default `useBlockettes: True` silently reverts the
turbulence model to standard SA.** Measured from the Edwards-converged field
(`logs/sweep/S2_endgame_*.log`), 400 cycles each:

| | restart res nuturb | restart totalRes | after |
|---|---|---|---|
| **A** ANK/NK, `useBlockettes: True` | **1.6403e-03** (wrong) | **8.68e+04** (wrong) | frozen — `1.6403227620708058e-3` → `1.6403227620711081e-3` over 99 iters |
| **B** ANK/NK, `useBlockettes: False` | 8.77e-11 ✓ | 4.641e-03 ✓ | **8.34e-16 / 8.17e-07 in 4 NK iterations** |
| **C** DADI only (control) | 8.77e-11 | 4.641e-03 | 8.75e-11 / 4.633e-03 — the floor |

Config A is the tell: the residual it reports on a *converged* field is 8.68e+04,
and it never moves, because the DADI turbulence update is solving Edwards while
the monitor is measuring standard SA. Nothing errors. This is the same class of
trap as `turbulenceModel: "SA-Edwards"` being dead — the option is accepted, and
the physics is quietly wrong.

#### Adopted S2 configuration — converged (`logs/S2_m15_final.log`)

```python
"useBlockettes": False,   # mandatory with saVariant + ANK/NK
"useANKSolver":  False,   # segregated ANK's turbulence update is the same
                          # stalling DADI, so it would not touch the floor
"useNKSolver":   True,
"NKSwitchTol":   1e-6,    # NK takes over at totalRes = 1e-6 * 3.74e8 = 374
```

**257 s on 14 cores** — 16.5× faster than the 40,000-cycle run that did *not*
converge.

| phase | cycles | totalRes |
|---|---|---|
| restart from S1 | 0 | 1.19e-02 |
| Edwards source term fires | 1 | 8.23e+04 |
| DADI settling | 1 → 2281 | → 3.73e+02 |
| **NK** | 2282 → 2289 (8 iters) | → **1.275e-04** |

Final: `res rho = 3.92e-08`, `res nuturb = 2.61e-13` (vs the DADI floor of
8.8e-11), `Y+_max = 0.7617`. `L2Convergence: 1e-12` needs ≤ 3.74e-4 — met.
`turbmodel = 2` and `savariant = 2` confirmed at runtime, and the script's own
nuTilde check passes (`nuTilde` spans 1.01e-8 … 3.03e-5, so the turbulence
equation really is being solved).

Output: `output_SAE/comp_corner_sa_edwards_000_vol.cgns`, `..._surf.cgns`.

**Physics check, and SA vs SA-Edwards:**

| | SA (S1) | SA-Edwards (S2) | Δ |
|---|---|---|---|
| separation | −3.626 mm | −3.799 mm | −0.173 mm |
| reattachment | +2.896 mm | +2.818 mm | −0.078 mm |
| bubble length | 6.522 mm | 6.617 mm | +1.4% |
| Cf,x minimum | −2.0256e-3 | −2.0475e-3 | −1.1% |

The Edwards modification moves separation slightly upstream and grows the bubble
by 1.4% — a small, physically sensible shift in the near-wall production/destruction
balance rather than a qualitative change. Both solutions have a single closed
bubble straddling the corner.

---

## 7. Reproducing

```bash
cd ~/ADFlow/2_comp_corner
mpiexec -n 14 python3.11 comp_corner_S1.py \
    --gridFile ./meshes/comp_corner_15_fixed.cgns          # -> output_SA/comp_corner_sa_000_vol.cgns
mpiexec -n 14 python3.11 comp_corner_S2.py \
    --gridFile ./meshes/comp_corner_15_fixed.cgns \
    --restartFile ./output_SA/comp_corner_sa_000_vol.cgns  # -> output_SAE/
```

`-n 14` is required, not a preference (§4).
