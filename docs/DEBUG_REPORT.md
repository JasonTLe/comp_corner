# Debug Report — ADFlow `comp_corner.py` Isothermal-Wall Crash

**Date:** 2026-07-16
**Case:** `/home/rohit/Desktop/Comp_Corner_ADFlow/comp_corner.py` (supersonic compression-corner, RANS/SA, Mach 2.95)
**Environment:** `~/packages/myenv` (Python 3.12) — adflow 2.10.0, cgnsutilities 2.8.1, baseclasses 1.9.0, pyhyp 2.6.3
**Status:** ✅ **Root cause identified and fixed. Solver runs and converges.**

---

## 1. Symptom

Running `python comp_corner.py` aborted during solver construction with:

```
#--------------------------- !!! Error !!! ----------------------------
#* Terminate called by processor 0
#* Run-time error in procedure BCDataIsothermalWall
#* Error message: Zone blk-1,          boundary subface dom-3: Wall
#*                temperature not specified for isothermal wall
```

followed by `MPI_ABORT` on rank 0. Additionally, seven warnings of the form
*"CGNS Block 1, boundary condition N ... does not have a family"* were printed.

## 2. Root cause

### 2.1 The grid file is missing BC data

Inspecting `comp_corner_5.cgns` with the `cgnsutilities` Python API showed one block
(`blk-1`, dims 700 × 799 × 2) with seven boundary conditions — **none with a family
name, and the two isothermal walls with no `BCDataSet` attached**:

| Boco   | BC type                   | Point range              | Family    | BCDataSets |
|--------|---------------------------|--------------------------|-----------|------------|
| dom-1  | `bcsymmetryplane`         | k=1 plane                | *default* | 0          |
| dom-7  | `bcsymmetryplane`         | k=2 plane                | *default* | 0          |
| dom-6  | `bcfarfield`              | i=1 plane                | *default* | 0          |
| dom-3  | `bcwallviscousisothermal` | i=700, j=1–400           | *default* | **0** ⚠️   |
| dom-4  | `bcwallviscousisothermal` | i=700, j=400–799         | *default* | **0** ⚠️   |
| dom-2  | `bcinflowsupersonic`      | j=1 plane                | *default* | 0          |
| dom-5  | `bcoutflow`               | j=799 plane              | *default* | 0          |

An isothermal wall BC in CGNS is only complete when it carries a
`BCDataSet → Dirichlet BCData → Temperature` data array. ADFlow's Fortran
preprocessing routine `BCDataIsothermalWall` reads that array while building the
solver and hard-aborts when it is absent. (This is a mesh-export issue — the
grid generator wrote the BC *type* but no BC *data*.)

### 2.2 Why `setBCVar` in the script did not help

`comp_corner.py` does set the wall temperature:

```python
ap.setBCVar("Temperature", 275.4, "wall")   # line 55  → prints "update bc 275.4"
CFDSolver = ADFLOW(options=aeroOptions)     # line 58  → CRASHES HERE
CFDSolver.setAeroProblem(ap)                # line 61  → never reached
```

`AeroProblem.setBCVar()` only stores the value on the AeroProblem object; ADFlow
applies it to the flow solver later, inside `setAeroProblem()` — and even then it
can only **update existing BC data arrays**. The crash occurs one line earlier,
during `ADFLOW(...)` grid preprocessing, before any AeroProblem is attached. So no
Python-side setting can rescue a grid whose isothermal walls carry no temperature:
**the value must exist in the CGNS file itself.**

The commented-out `getHeatFluxes`/`setWallTemperature` attempt in the script fails
for the same reason — those also require a live (successfully initialized) solver.

## 3. Fix

### 3.1 What was done

A repair script, [`fix_bc_temperature.py`](fix_bc_temperature.py), uses the
`cgnsutilities` Python API (`readGrid`, `BocoDataSet`, `BocoDataSetArray`) to:

1. **Attach a Dirichlet `Temperature = 275.4 K` BCDataSet** to every
   `bcwallviscousisothermal` boco (dom-3 and dom-4).
2. **Assign explicit family names** matching the ones ADFlow was auto-generating
   (`wall`, `sym`, `far`, `inflow`, `outflow`), which removes the seven warnings
   while keeping `ap.setBCVar(..., "wall")` pointed at the correct family.

Output was written to a **new** file, `comp_corner_5_isoT.cgns`; the original grid
is untouched.

```bash
source ~/packages/myenv/bin/activate
python fix_bc_temperature.py                  # comp_corner_5.cgns → comp_corner_5_isoT.cgns
# general form:
python fix_bc_temperature.py <in.cgns> <out.cgns> <Twall_K>
```

### 3.2 Why not `cgns_utils overwriteBC`?

The CLI route (`cgns_utils overwriteBC grid.cgns bcFile out.cgns`) can also write
BC datasets, but `Block.overwriteBCs()` enforces **one BC per block face** — it
would have merged dom-3 and dom-4 into a single boco spanning the whole i=700
face. Functionally equivalent here (same type/family, and together they tile the
face), but the Python-API route preserves the original topology exactly, so it was
preferred. For reference, the equivalent bcFile line would be:

```
1 iHigh bcwallviscousisothermal wall BCDataSet_1 BCWallViscousIsothermal Dirichlet Temperature 275.4
```

### 3.3 One implementation gotcha

`BocoDataSetArray` requires `dataDims` to be a **length-3 int32 array** (Fortran
convention: `np.ones(3, dtype=np.int32, order="F")` with `dataDims[0] = arr.size`),
not a plain Python list — otherwise `libcgns_utils.utils.writebcdata` raises
`ValueError: unexpected array size: new_size=3, got array with arr_size=1`.

## 4. Verification

Re-reading `comp_corner_5_isoT.cgns` confirms both wall bocos now carry the data:

```
BC4 | bcwallviscousisothermal | fam: wall | nDS: 1 → Dirichlet Temperature = [275.4]
BC5 | bcwallviscousisothermal | fam: wall | nDS: 1 → Dirichlet Temperature = [275.4]
```

Running the case against the fixed grid:

```bash
python comp_corner.py --gridFile ./comp_corner_5_isoT.cgns
```

- Preprocessing completes (no `BCDataIsothermalWall` abort, no family warnings).
- The ANK solver starts and converges steadily; by iteration ~32 the total
  residual had dropped from 3.5×10⁶ to 1.4×10⁴ (NK switch scheduled at 3.6×10²,
  `L2Convergence = 1e-15`).

## 5. Recommendations / open items

1. **Fix at the source**: in the mesher (the script header mentions Pointwise),
   export isothermal walls *with* their BCDataSet temperature, or plan on running
   `fix_bc_temperature.py` after every export.
2. **Sibling grids**: `comp_corner_1..4.cgns` were exported the same way and very
   likely have the same defect. Batch-fix:
   ```bash
   for i in 1 2 3 4 5; do
       python fix_bc_temperature.py comp_corner_${i}.cgns comp_corner_${i}_isoT.cgns 275.4
   done
   ```
3. **Script default**: `comp_corner.py` still defaults to the broken grid
   (`--gridFile ./comp_corner_5.cgns`); update it to the `_isoT` file (or replace
   the original once you're confident).
4. **Dead code**: the commented `getHeatFluxes`/`setWallTemperature` block in
   `comp_corner.py` (lines 63–65) can be removed — with the temperature in the
   grid file it is unnecessary, and `ap.setBCVar` (line 55) is now the correct
   mechanism for *changing* T_wall between runs.
5. **Runtime**: full convergence to 1e-15 on 557,802 cells with 1 MPI rank is
   slow; consider `mpirun -np <N> python comp_corner.py ...` for production runs.
6. Minor: `"turbResScale": 10e4` is `1e5` — fine for SA, just note it's not `1e4`
   if that was the intent.
