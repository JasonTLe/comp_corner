#!/usr/bin/env python3
"""
visualize_corner.py
===================

Turn one ADflow surface-solution file into publication-quality figures.

    python visualize_corner.py comp_corner_000_surf.cgns

writes two PNGs next to wherever you run it, named after the input file:

    comp_corner_000_surf_flowfield.png   Mach field, numerical schlieren,
                                         close-up of the separation bubble
    comp_corner_000_surf_wall.png        p/p_inf, Cf, y+ and the incoming
                                         boundary-layer profile

If a context/ directory sits next to this script, the digitized paper curves
in it (p*.csv, cf*.csv) are resampled into continuous lines and drawn over
the p/p_inf and Cf panels.  --context DIR points somewhere else, --no-context
turns the overlay off.

and prints a table of derived quantities (separation length, shock angle,
boundary-layer thickness, near-wall resolution, ...).


REQUIREMENTS
------------
numpy and matplotlib.  Nothing else.  In particular no h5py, no CGNS library
and no VTK -- the CGNS container is parsed directly by this file, in both of
its flavours:

    HDF5   the default container since CGNS 3.x
    ADF    the legacy container (ParaView's IOSS reader cannot open these,
           which is a common reason a perfectly good file "won't load")

The container is identified by its magic number and the matching reader is
used automatically, so both produce identical numpy arrays.


WHAT THIS EXPECTS
-----------------
An ADflow *_surf.cgns file for a 2-D ramp / compression-corner case:

  * one pair of symmetry-plane zones carrying the 2-D flow field
    (zone names containing "Symmetry")
  * one or more no-slip wall zones (zone names containing "Wall")
  * a flat plate leading into a single straight ramp, corner at x = 0

The figures will draw whatever you give them, but quantities such as
delta_0, Re_tau and L_sep/delta_0 assume that geometry.


CONVENTIONS IN ADflow SURFACE FILES
-----------------------------------
Four things about these files are easy to get wrong.  All four are handled
in Part 3; they are spelled out here because they bite anyone writing their
own post-processing:

  1. HDF5 stores array dimensions in the reverse of CGNS order, so a zone of
     size (ni, nj) yields arrays shaped (nj, ni).  ADF stores them in CGNS
     order, and Part 1 reshapes it to match so the rest of the code is
     unaware of which container it came from.

  2. FlowSolution_t is CellCenter with Rind = [1, 1, 1, 1], i.e. one layer of
     ghost cells all round.  It is stripped with [1:-1, 1:-1].

  3. Pressure, Density and Temperature are normalised so that the freestream
     value is exactly 1.0; velocities by sqrt(p_ref / rho_ref).  The
     dimensional reference values live in ReferenceState_t.

  4. SkinFriction{Magnitude,X,Y,Z} are ALREADY coefficients, normalised by
     0.5 * rho_inf * U_inf^2.  Do not divide by dynamic pressure again.
     How to check: in the viscous sublayer du+/dy+ must equal 1.  Treating
     these as tau_w/p_ref gives 6.01 instead -- wrong by exactly
     0.5 * gamma * M^2 = 6.09.

Index directions are also not consistent between files, so Part 3 works out
which side is the wall and which way the flow goes from the data itself
rather than assuming.


FILE LAYOUT
-----------
    Part 1   container readers      HDF5File, ADFFile
    Part 2   CGNS node tree         Node, load, tree
    Part 3   physics extraction     Case
    Part 4   streamline tracing     contravariant, streamline
    Part 5   figures                figure_flowfield, figure_wall
    Part 6   command line           report, main

Parts 1 and 2 are plumbing -- read them only if a file fails to open.
Part 3 is where the fluid dynamics lives.
"""
import sys
import os
import struct
import argparse
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')          # render to file; no display needed
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.tri import Triangulation, LinearTriInterpolator

GAM = 1.4                      # ratio of specific heats
KARMAN, BLOG = 0.41, 5.2       # log-law constants
MM = 1e3                       # metres -> millimetres, used in every plot

# Target incoming boundary layer, from the experiments of Zheltovodov et al.
# (1990) as used by Hao, JFM 2023, 971 A28 (low-Re case): delta = 2.27 mm
# measured 15.4 delta upstream of the corner, giving Re_delta = 63560.  The
# station is therefore fixed in space, unlike Case.x_ref which follows x_sep.
DELTA_EXP = 2.27e-3            # experimental boundary-layer thickness, metres
X_EXP = -15.4*DELTA_EXP        # station where it was measured, metres (x=0 = corner)
RE_DELTA_EXP = 63560           # Re based on DELTA_EXP and free-stream properties

# Sutherland's law for the dynamic viscosity of air, SI units.
mu = lambda T: 1.458e-6 * T ** 1.5 / (T + 110.4)


# ============================================================ Part 1: HDF5
H5MAGIC = b'\x89HDF\r\n\x1a\n'



class HDF5File:
    def __init__(self, path):
        self.f = open(path, 'rb')
        self.f.seek(0, 2)
        self.filesize = self.f.tell()
        b = self.at(0, 96)
        if b[:8] != H5MAGIC:
            raise ValueError('not an HDF5 file')
        self.sbver = b[8]
        if self.sbver >= 2:
            self.O, self.L = b[9], b[10]
            o = 12
            self.base = self.uu(b, o, self.O); o += self.O
            self.sbext = self.uu(b, o, self.O); o += self.O
            self.eof = self.uu(b, o, self.O); o += self.O
            self.root_addr = self.uu(b, o, self.O)
        else:
            self.O, self.L = b[13], b[14]
            o = 24 + (4 if self.sbver == 1 else 0)
            self.base = self.uu(b, o, self.O); o += self.O
            o += self.O
            self.eof = self.uu(b, o, self.O); o += self.O
            o += self.O
            ste = self.at(o, 2 * self.O + 24)
            self.root_addr = self.uu(ste, self.O, self.O)
        self.undef = (1 << (8 * self.O)) - 1

    # ---------- low level ----------
    def at(self, addr, n):
        self.f.seek(addr)
        return self.f.read(n)

    @staticmethod
    def uu(buf, off, n):
        return int.from_bytes(buf[off:off + n], 'little')

    # ---------- object header ----------
    def obj_msgs(self, addr):
        msgs = []
        sig = self.at(addr, 4)
        if sig == b'OHDR':
            h = self.at(addr, 6)
            flags = h[5]
            o = addr + 6
            if flags & 0x20: o += 16
            if flags & 0x10: o += 4
            ss = 1 << (flags & 0x03)
            csize = self.uu(self.at(o, ss), 0, ss)
            o += ss
            self._msgs2(o, o + csize, flags, msgs)
        else:
            h = self.at(addr, 16)
            if h[0] != 1:
                raise ValueError('bad object header at %d' % addr)
            nmsg = self.uu(h, 2, 2)
            ohsize = self.uu(h, 8, 4)
            self._msgs1(addr + 16, addr + 16 + ohsize, nmsg, msgs)
        return msgs

    def _msgs2(self, o, end, ohflags, msgs):
        while o + 4 <= end:
            mtype = self.uu(self.at(o, 1), 0, 1)
            hb = self.at(o, 4)
            mtype, msize, mflags = hb[0], self.uu(hb, 1, 2), hb[3]
            o += 4
            if ohflags & 0x04: o += 2
            data = self.at(o, msize)
            self._handle(mtype, mflags, data, msgs, v2=True, ohflags=ohflags)
            o += msize

    def _msgs1(self, o, end, nmsg, msgs):
        cnt = 0
        while cnt < nmsg and o + 8 <= end:
            hb = self.at(o, 8)
            mtype = self.uu(hb, 0, 2)
            msize = self.uu(hb, 2, 2)
            mflags = hb[4]
            o += 8
            data = self.at(o, msize)
            self._handle(mtype, mflags, data, msgs, v2=False, ohflags=0)
            o += msize
            cnt += 1

    def _handle(self, mtype, mflags, data, msgs, v2, ohflags):
        if mtype == 0x0010:  # continuation
            caddr = self.uu(data, 0, self.O)
            clen = self.uu(data, self.O, self.L)
            if v2:
                self._msgs2(caddr + 4, caddr + clen - 4, ohflags, msgs)
            else:
                self._msgs1(caddr, caddr + clen, 10 ** 6, msgs)
        else:
            msgs.append((mtype, mflags, data))

    # ---------- messages ----------
    def dataspace(self, data):
        ver, ndim, flags = data[0], data[1], data[2]
        o = 8 if ver == 1 else 4
        dims = [self.uu(data, o + i * self.L, self.L) for i in range(ndim)]
        return dims

    def datatype(self, data):
        b0 = data[0]
        cls = b0 & 0x0F
        bf = data[1] | (data[2] << 8) | (data[3] << 16)
        size = self.uu(data, 4, 4)
        be = '>' if (bf & 0x01) else '<'
        if cls == 0:
            k = 'i' if (bf & 0x08) else 'u'
            return np.dtype('%s%s%d' % (be, k, size)), cls
        if cls == 1:
            return np.dtype('%sf%d' % (be, size)), cls
        if cls == 3:
            return np.dtype('S%d' % size), cls
        if cls == 8:  # enum -> use base type
            return self.datatype(data[8:])[0], cls
        return np.dtype('V%d' % size), cls

    def layout(self, data):
        ver = data[0]
        if ver in (1, 2):
            ndim, cls = data[1], data[2]
            o = 8
            addr = self.uu(data, o, self.O); o += self.O
            dims = [self.uu(data, o + 4 * i, 4) for i in range(ndim)]
            return dict(cls=cls, addr=addr, cdims=dims)
        if ver == 3:
            cls = data[1]; o = 2
            if cls == 0:
                n = self.uu(data, o, 2)
                return dict(cls=0, raw=data[o + 2:o + 2 + n])
            if cls == 1:
                addr = self.uu(data, o, self.O); o += self.O
                return dict(cls=1, addr=addr, size=self.uu(data, o, self.L))
            if cls == 2:
                ndim = data[o]; o += 1
                addr = self.uu(data, o, self.O); o += self.O
                dims = [self.uu(data, o + 4 * i, 4) for i in range(ndim)]
                o += 4 * ndim
                return dict(cls=2, addr=addr, cdims=dims, esize=self.uu(data, o, 4))
        if ver == 4:
            cls = data[1]; o = 2
            if cls == 0:
                n = self.uu(data, o, 2)
                return dict(cls=0, raw=data[o + 2:o + 2 + n])
            if cls == 1:
                addr = self.uu(data, o, self.O); o += self.O
                return dict(cls=1, addr=addr, size=self.uu(data, o, self.L))
            if cls == 2:
                flags = data[o]; o += 1
                esz = data[o]; o += 1
                ndim = data[o]; o += 1
                dsz = data[o]; o += 1
                dims = [self.uu(data, o + dsz * i, dsz) for i in range(ndim)]
                o += dsz * ndim
                itype = data[o]; o += 1
                return dict(cls=2, v4=True, itype=itype, cdims=dims, rest=data[o:])
        raise ValueError('layout ver %d cls?' % ver)

    def attrs(self, addr):
        out = {}
        for mtype, mflags, data in self.obj_msgs(addr):
            if mtype != 0x000C:
                continue
            ver = data[0]
            o = 1
            if ver == 1:
                o += 1
                nsz = self.uu(data, o, 2); dtsz = self.uu(data, o + 2, 2); dssz = self.uu(data, o + 4, 2)
                o += 6
                pad = lambda n: (n + 7) & ~7
                name = data[o:o + nsz].split(b'\x00')[0].decode('utf-8', 'replace'); o += pad(nsz)
                dt = data[o:o + dtsz]; o += pad(dtsz)
                ds = data[o:o + dssz]; o += pad(dssz)
            else:
                o += 1
                nsz = self.uu(data, o, 2); dtsz = self.uu(data, o + 2, 2); dssz = self.uu(data, o + 4, 2)
                o += 6
                if ver == 3: o += 1
                name = data[o:o + nsz].split(b'\x00')[0].decode('utf-8', 'replace'); o += nsz
                dt = data[o:o + dtsz]; o += dtsz
                ds = data[o:o + dssz]; o += dssz
            npdt, cls = self.datatype(dt)
            dims = self.dataspace(ds)
            n = int(np.prod(dims)) if dims else 1
            raw = data[o:o + n * npdt.itemsize]
            arr = np.frombuffer(raw, dtype=npdt, count=n)
            if cls == 3:
                v = b''.join(arr.tolist()).split(b'\x00')[0].decode('utf-8', 'replace')
            else:
                v = arr.reshape(dims) if dims else arr[0]
            out[name] = v
        return out

    def dataset(self, addr):
        dims = None; npdt = None; lay = None; filters = []
        for mtype, mflags, data in self.obj_msgs(addr):
            if mtype == 0x0001: dims = self.dataspace(data)
            elif mtype == 0x0003: npdt, cls = self.datatype(data)
            elif mtype == 0x0008: lay = self.layout(data)
            elif mtype == 0x000B: filters = self._filters(data)
        if npdt is None or lay is None:
            return None
        if dims is None: dims = []
        n = int(np.prod(dims)) if dims else 1
        if lay['cls'] == 0:
            raw = lay['raw']
        elif lay['cls'] == 1:
            raw = self.at(lay['addr'], n * npdt.itemsize)
        elif lay['cls'] == 2:
            raw = self._chunked(lay, dims, npdt, filters)
        arr = np.frombuffer(raw[:n * npdt.itemsize], dtype=npdt, count=n)
        return arr.reshape(dims) if dims else arr[0]

    def _filters(self, data):
        ver = data[0]; nf = data[1]; o = 8 if ver == 1 else 2
        out = []
        for _ in range(nf):
            fid = self.uu(data, o, 2); o += 2
            if ver == 1:
                nlen = self.uu(data, o, 2); o += 2
            else:
                nlen = 0 if fid < 256 else self.uu(data, o, 2)
                if fid >= 256: o += 2
            flags = self.uu(data, o, 2); o += 2
            ncd = self.uu(data, o, 2); o += 2
            if ver == 1:
                nm = ((nlen + 7) & ~7) if nlen else 0
                o += nm
            else:
                o += nlen
            cd = [self.uu(data, o + 4 * i, 4) for i in range(ncd)]
            o += 4 * ncd
            if ver == 1 and ncd % 2: o += 4
            out.append((fid, cd))
        return out

    def _chunked(self, lay, dims, npdt, filters):
        if lay.get('v4'):
            raise NotImplementedError('layout v4 chunk index %d' % lay['itype'])
        cdims = lay['cdims'][:-1]
        esize = npdt.itemsize
        full = np.zeros(dims, dtype=npdt)
        for off, cbuf in self._btree1_chunks(lay['addr'], len(cdims)):
            for fid, cd in reversed(filters):
                if fid == 1: cbuf = zlib.decompress(cbuf)
                elif fid == 2:
                    a = np.frombuffer(cbuf, dtype=np.uint8).reshape(cd[0], -1)
                    cbuf = a.T.tobytes()
            c = np.frombuffer(cbuf, dtype=npdt, count=int(np.prod(cdims))).reshape(cdims)
            sl = tuple(slice(off[i], min(off[i] + cdims[i], dims[i])) for i in range(len(dims)))
            cs = tuple(slice(0, sl[i].stop - sl[i].start) for i in range(len(dims)))
            full[sl] = c[cs]
        return full.tobytes()

    def _btree1_chunks(self, addr, ndim):
        out = []
        stack = [addr]
        while stack:
            a = stack.pop()
            if a in (self.undef, 0): continue
            h = self.at(a, 24)
            if h[:4] != b'TREE': continue
            level = h[5]
            nent = self.uu(h, 6, 2)
            o = a + 24
            rec = 8 + 8 * (ndim + 1)
            blk = self.at(o, nent * (rec + self.O) + rec)
            p = 0
            for i in range(nent):
                csize = self.uu(blk, p, 4); mask = self.uu(blk, p + 4, 4)
                offs = [self.uu(blk, p + 8 + 8 * k, 8) for k in range(ndim)]
                p += rec
                child = self.uu(blk, p, self.O); p += self.O
                if level == 0:
                    out.append((offs, self.at(child, csize)))
                else:
                    stack.append(child)
        return out

    # ---------- groups / links ----------
    def links(self, addr):
        out = {}
        linkinfo = None; symtab = None
        for mtype, mflags, data in self.obj_msgs(addr):
            if mtype == 0x0006:
                nm, tgt, _ = self._link_rec(data, 0)
                if tgt is not None: out[nm] = tgt
            elif mtype == 0x0002:
                linkinfo = data
            elif mtype == 0x0011:
                symtab = data
        if linkinfo is not None:
            ver, flags = linkinfo[0], linkinfo[1]
            o = 2
            if flags & 0x01: o += 8
            fh = self.uu(linkinfo, o, self.O); o += self.O
            if fh not in (self.undef, 0):
                for rec in self._heap_objects(fh):
                    try:
                        nm, tgt, _ = self._link_rec(rec, 0)
                    except Exception:
                        continue
                    if tgt is not None: out[nm] = tgt
        if symtab is not None:
            bt = self.uu(symtab, 0, self.O)
            lh = self.uu(symtab, self.O, self.O)
            out.update(self._symtab(bt, lh))
        return out

    def _link_rec(self, b, o):
        ver = b[o]; flags = b[o + 1]; o += 2
        if ver != 1: raise ValueError('link ver')
        ltype = 0
        if flags & 0x08: ltype = b[o]; o += 1
        if flags & 0x04: o += 8
        if flags & 0x10: o += 1
        nl = 1 << (flags & 0x03)
        nsz = self.uu(b, o, nl); o += nl
        name = b[o:o + nsz].decode('utf-8', 'replace'); o += nsz
        if ltype == 0:
            tgt = self.uu(b, o, self.O); o += self.O
        else:
            n = self.uu(b, o, 2); o += 2 + n
            tgt = None
        return name, tgt, o

    def _symtab(self, bt, lh):
        out = {}
        hh = self.at(lh, 32)
        dseg = self.uu(hh, 8 + 2 * self.L, self.O) if hh[:4] == b'HEAP' else None
        stack = [bt]
        snods = []
        while stack:
            a = stack.pop()
            if a in (self.undef, 0): continue
            h = self.at(a, 24)
            if h[:4] != b'TREE': continue
            level = h[5]; nent = self.uu(h, 6, 2)
            o = a + 24
            blk = self.at(o, (nent * 2 + 1) * self.O)
            p = 0
            for i in range(nent):
                p += self.O
                child = self.uu(blk, p, self.O); p += self.O
                (snods if level == 0 else stack).append(child)
        for sa in snods:
            h = self.at(sa, 8)
            if h[:4] != b'SNOD': continue
            nsym = self.uu(h, 6, 2)
            ent = self.at(sa + 8, nsym * (2 * self.O + 24))
            for i in range(nsym):
                q = i * (2 * self.O + 24)
                noff = self.uu(ent, q, self.O)
                oha = self.uu(ent, q + self.O, self.O)
                nm = self._heapstr(dseg, noff)
                out[nm] = oha
        return out

    def _heapstr(self, dseg, off):
        b = self.at(dseg + off, 256)
        return b.split(b'\x00')[0].decode('utf-8', 'replace')

    def _heap_objects(self, addr):
        hd = self.at(addr, 256)
        if hd[:4] != b'FRHP': return []
        p = 4
        ver = hd[p]; p += 1
        idlen = self.uu(hd, p, 2); p += 2
        iof = self.uu(hd, p, 2); p += 2
        flags = hd[p]; p += 1
        p += 4
        p += self.L + self.O + self.L + self.O + self.L * 3
        p += self.L * 5
        width = self.uu(hd, p, 2); p += 2
        sbs = self.uu(hd, p, self.L); p += self.L
        mds = self.uu(hd, p, self.L); p += self.L
        mhs = self.uu(hd, p, 2); p += 2
        p += 2
        root = self.uu(hd, p, self.O); p += self.O
        rows = self.uu(hd, p, 2)
        hsz = (mhs + 7) // 8
        chk = bool(flags & 0x02)
        out = []
        if root in (self.undef, 0):
            return out
        if rows == 0:
            self._heap_direct(root, sbs, hsz, chk, out)
        else:
            self._heap_indirect(root, rows, width, sbs, mds, hsz, chk, out)
        return out

    def _rowsize(self, i, sbs):
        return sbs if i < 2 else sbs * (2 ** (i - 1))

    def _heap_indirect(self, addr, rows, width, sbs, mds, hsz, chk, out):
        b = self.at(addr, 4)
        if b != b'FHIB': return
        maxdrows = int(np.log2(mds)) - int(np.log2(sbs)) + 2
        ndr = min(rows, maxdrows)
        o = addr + 4 + 1 + self.O + hsz
        blk = self.at(o, (rows * width) * self.O + 64)
        p = 0
        for r in range(ndr):
            bs = self._rowsize(r, sbs)
            for c in range(width):
                a = self.uu(blk, p, self.O); p += self.O
                if a not in (self.undef, 0):
                    self._heap_direct(a, bs, hsz, chk, out)
        for r in range(ndr, rows):
            for c in range(width):
                a = self.uu(blk, p, self.O); p += self.O
                if a not in (self.undef, 0):
                    self._heap_indirect(a, rows, width, sbs, mds, hsz, chk, out)

    def _heap_direct(self, addr, bsize, hsz, chk, out):
        b = self.at(addr, bsize)
        if b[:4] != b'FHDB': return
        o = 4 + 1 + self.O + hsz
        if chk: o += 4
        while o < len(b):
            if b[o] != 1:
                o += 1
                continue
            try:
                nm, tgt, no = self._link_rec(b, o)
            except Exception:
                o += 1
                continue
            if no <= o or no > len(b):
                o += 1
                continue
            out.append(b[o:no])
            o = no



SKIP = (' data', ' hdf5version', ' format', ' link', ' mother', ' file')


# ============================================================ Part 1b: ADF

ADF_MAGIC = b'\xc0\xa8\xa3\xa9'
NODE_SIZE = 246
BLOCK = 4096

_DT = {'R4': 'f4', 'R8': 'f8', 'I4': 'i4', 'I8': 'i8', 'U4': 'u4', 'U8': 'u8',
       'X4': 'c8', 'X8': 'c16', 'B1': 'u1', 'C1': 'S1', 'LK': 'S1'}


class ADFFile:
    def __init__(self, path):
        self.f = open(path, 'rb')
        h = self.at(0, 0x100)
        if h[:4] != ADF_MAGIC:
            raise ValueError('not an ADF file')
        self.version = h[4:0x20].decode('latin1').strip()
        i = h.find(b'AdF2')
        self.numfmt = chr(h[i+4])                       # 'L' little, 'B' big, 'N' native
        self.end = '>' if self.numfmt == 'B' else '<'
        j = h.find(b'AdF4')
        self.root_addr = self.dp(j+4)
        self.eof = self.dp(j+16)

    # ---- primitives ----
    def at(self, a, n):
        self.f.seek(a); return self.f.read(n)

    def dp(self, a):
        # An ADF disk pointer is 12 ASCII characters, not 12 binary bytes:
        # 8 hex digits of block number followed by 4 hex digits of offset
        # inside that block.  block 0 / offset BLOCK is the "unset" value.
        b = self.at(a, 12)
        blk = int(b[:8], 16)
        off = int(b[8:12], 16)
        v = blk*BLOCK + off
        return None if (blk == 0 and off == BLOCK) else v

    def hexint(self, a, n):
        s = self.at(a, n).strip()
        return int(s, 16) if s else 0

    # ---- node ----
    def node(self, a):
        b = self.at(a, NODE_SIZE)
        if b[:4] != b'NoDe' or b[242:246] != b'TaiL':
            raise ValueError('bad ADF node at %d' % a)
        nd = int(b[128:130], 16)
        dims = [int(b[130+8*i:138+8*i], 16) for i in range(nd)]   # ASCII hex, like everything else in the header
        return dict(addr=a,
                    name=b[4:36].decode('latin1').strip(),
                    label=b[36:68].decode('latin1').strip(),
                    nsub=int(b[68:76], 16), nent=int(b[76:84], 16),
                    sntb=self.dp(a+84),
                    dt=b[96:128].decode('latin1').strip(),
                    dims=dims, nchunk=int(b[226:230], 16), data=self.dp(a+230))

    def children(self, h):
        if not h['nsub'] or h['sntb'] is None:
            return []
        t = h['sntb']
        if self.at(t, 4) != b'SNTb':
            return []
        out = []
        blk = self.at(t+16, 44*h['nsub'])
        for i in range(h['nsub']):
            nm = blk[44*i:44*i+32].decode('latin1').strip()
            a = self.dp(t+16+44*i+32)
            if a is not None and not nm.startswith('unused'):
                out.append((nm, a))
        return out

    def _chunk(self, a):
        tag = self.at(a, 4)
        if tag != b'DaTa':
            raise ValueError('expected DaTa at %d, got %r' % (a, tag))
        end = self.dp(a+4)
        return self.at(a+16, end-(a+16))

    def data(self, h):
        if h['dt'] in ('MT', '') or not h['dims'] or h['data'] is None:
            return None
        if h['nchunk'] <= 1:
            raw = self._chunk(h['data'])
        else:                                    # chunk table (untested path)
            t = h['data']
            if self.at(t, 4) != b'CKTb':
                raise ValueError('expected CKTb at %d' % t)
            raw = b''.join(self._chunk(self.dp(t+16+24*i)) for i in range(h['nchunk']))
        code = _DT.get(h['dt'])
        if code is None:
            return np.frombuffer(raw, dtype='u1')
        dt = np.dtype(code if code == 'S1' else self.end+code)
        n = int(np.prod(h['dims']))
        return np.frombuffer(raw[:n*dt.itemsize], dtype=dt, count=n).reshape(h['dims'][::-1])


# ==================================================== Part 2: CGNS node tree

class Node:
    def __init__(self, name, label, dtype, data, children):
        self.name, self.label, self.dtype, self.data, self.children = name, label, dtype, data, children
    def get(self, *labels_or_names):
        for c in self.children:
            if c.name in labels_or_names or c.label in labels_or_names:
                return c
        return None
    def all(self, label):
        return [c for c in self.children if c.label == label]
    def __repr__(self):
        s = '' if self.data is None else str(getattr(self.data, 'shape', self.data))
        return '<%s %s %s %s>' % (self.name, self.label, self.dtype, s)


def _load_adf(path):
    """ADF container -> same Node tree as the HDF5 path."""
    a = ADFFile(path)
    def build(addr):
        h = a.node(addr)
        return Node(h['name'], h['label'], h['dt'], a.data(h),
                    [build(ad) for _, ad in a.children(h)])
    return a, build(a.root_addr)


def load(path):
    with open(path, 'rb') as f:
        if f.read(4) == b'\xc0\xa8\xa3\xa9':
            return _load_adf(path)
    h = HDF5File(path)
    def build(addr):
        a = h.attrs(addr)
        lk = h.links(addr)
        data = None
        if ' data' in lk:
            data = h.dataset(lk[' data'])
        kids = []
        for nm, ad in lk.items():
            if nm in SKIP or nm.startswith(' '):
                continue
            kids.append(build(ad))
        return Node(a.get('name', '?'), a.get('label', '?'), a.get('type', '?'), data, kids)
    return h, build(h.root_addr)


def tree(n, d=0, maxd=6):
    sh = ''
    if n.data is not None:
        sh = ' shape=%s' % (list(n.data.shape) if hasattr(n.data, 'shape') else n.data,)
        if n.dtype == 'C1' and hasattr(n.data, 'tobytes'):
            sh += ' = %r' % n.data.tobytes().split(b'\x00')[0][:60]
        elif n.data is not None and getattr(n.data, 'size', 0) and n.data.size <= 12 and n.dtype != 'C1':
            sh += ' = %s' % n.data.ravel().tolist()
    print('  ' * d + '%s [%s] <%s>%s' % (n.name, n.label, n.dtype, sh))
    if d < maxd:
        for c in n.children:
            tree(c, d + 1, maxd)


# =============================================================================
#  Part 3   Physics extraction
# =============================================================================

def oblique_shock_beta(M, theta):
    """Wave angle of the *weak* oblique shock that turns flow by `theta`.

    Solves the theta-beta-M relation

        tan(theta) = 2 cot(beta) (M^2 sin^2(beta) - 1)
                     / (M^2 (gamma + cos 2beta) + 2)

    by scanning beta from the Mach angle upwards and taking the FIRST sign
    change.  Taking any other root silently returns the strong solution,
    which for M = 2.95 and a 25 deg ramp is 79 deg instead of 44.6 deg.

    Parameters
    ----------
    M     : freestream Mach number
    theta : flow deflection angle, radians

    Returns
    -------
    beta : shock angle, radians
    """
    b = np.linspace(np.arcsin(1/M) + 1e-9, np.pi/2 - 1e-9, 400001)
    f = 2/np.tan(b)*(M**2*np.sin(b)**2 - 1)/(M**2*(GAM + np.cos(2*b)) + 2) - np.tan(theta)
    return b[np.where(np.sign(f[:-1]) != np.sign(f[1:]))[0][0]]


class Case:
    """One ADflow surface solution, parsed into fields, wall data and metrics.

    Everything is computed in the constructor, so after

        c = Case("comp_corner_000_surf.cgns")

    the useful attributes are:

    Geometry and grid
        X, Y        vertex coordinates of the symmetry plane, shape (nj+1, ni+1)
        Xc, Yc      cell-centre coordinates,                  shape (nj,   ni  )
        ramp        ramp angle in degrees, fitted from the wall

    Flow field (cell-centred, ghost layer removed, non-dimensional)
        rho, p, T   density, pressure, temperature   (freestream = 1)
        u, v        velocity components              (freestream speed = ue)
        M           Mach number
        gmag        |grad rho|, for the numerical schlieren

    Wall distributions, ordered by increasing x
        xw, yw      wall coordinates
        pw          wall pressure ratio p/p_inf (freestream = 1)
        cpw         pressure coefficient (freestream = 0), derived from pw
        cfw         skin-friction coefficient projected on the wall tangent
                    (negative => reversed flow => separated)
        yplus       y+ of the first cell off the wall
        dw          height of that first cell, metres

    Derived quantities
        x_sep,      separation and reattachment points, metres
        x_rea
        Lsep        separation length
        delta0      boundary-layer thickness at the reference station
        Retau       friction Reynolds number there
        delta_exp,  the same, at the experiment's fixed station X_EXP, plus
        Retau_exp,  Re_delta = Re_m*delta_exp, for comparison against
        Re_delta    DELTA_EXP and RE_DELTA_EXP
        profile     (y+, u+, u+_vanDriest) at the reference station
        beta_fit    shock angle fitted from the pressure field, degrees
        x_trans     laminar-to-turbulent transition location, NaN when the
                    plate is fully turbulent (which is every plain-SA run)

    Reference state (dimensional, from ReferenceState_t)
        Minf, Pref, Rref, Tref, Uinf, ainf, Re_m
        QD          0.5*gamma*Minf^2, the non-dimensional dynamic pressure
                    used to form Cp
    """

    # --- construction -------------------------------------------------------

    def __init__(self, path, label=None):
        self.path = path
        self.label = label or os.path.basename(path)
        _, root = load(path)
        base = root.all('CGNSBase_t')[0]

        # Dimensional reference values, used to convert back to SI for y+.
        rs = base.get('ReferenceState_t')
        r = {c.name: float(c.data[0]) for c in rs.children
             if c.data is not None and getattr(c.data, 'size', 0) == 1}
        self.Minf = r['Mach']
        self.Pref, self.Rref, self.Tref = r['Pressure'], r['Density'], r['Temperature']

        self.QD = 0.5*GAM*self.Minf**2              # = 0.5 rho_inf U_inf^2 / p_ref
        self.Rgas = self.Pref/(self.Rref*self.Tref)  # gas constant, ~287 for air
        self.ainf = np.sqrt(GAM*self.Rgas*self.Tref)
        self.Uinf = self.Minf*self.ainf
        self.uref = np.sqrt(self.Pref/self.Rref)     # ADflow's velocity scale
        self.ue = self.Uinf/self.uref                # freestream speed, non-dim
        self.Re_m = self.Rref*self.Uinf/mu(self.Tref)

        zones = base.all('Zone_t')
        self._load_field([z for z in zones if 'Symmetry' in z.name])
        self._load_wall(sorted([z for z in zones if 'Wall' in z.name],
                               key=lambda z: z.get('GridCoordinates_t')
                                              .get('CoordinateX').data.min()))
        self._derive_quantities()

    # --- reading zones ------------------------------------------------------

    @staticmethod
    def _zone_arrays(zone):
        """Coordinates and flow variables of one zone, ghost cells removed."""
        gc, fs = zone.get('GridCoordinates_t'), zone.get('FlowSolution_t')
        X = gc.get('CoordinateX').data.astype(float)
        Y = gc.get('CoordinateY').data.astype(float)
        rind = fs.get('Rind_t')
        # Rind = [1,1,1,1] means one ghost layer on every side.
        cut = (slice(1, -1), slice(1, -1)) if rind is not None and rind.data[0] == 1 \
              else (slice(None), slice(None))
        vals = {c.name: c.data.astype(float)[cut]
                for c in fs.children if c.label == 'DataArray_t'}
        return X, Y, vals

    def _load_field(self, candidates):
        """Load the 2-D flow field and normalise its index directions.

        The symmetry-plane zone holds the whole flow field.  Its index
        directions are NOT consistent between files, so rather than assume,
        we detect:

          * the wall is the edge where the velocity is small (no-slip), so we
            flip j until the wall sits at j = 0;
          * the inflow is the edge with the lower pressure (the outflow is
            behind the shock), so we flip i until it increases downstream.

        After this, self.u[0] is the row of cells against the wall and
        self.Xc[0] runs from the inlet to the outlet.
        """
        z = max(candidates,
                key=lambda z: z.get('GridCoordinates_t').get('CoordinateX').data.size)
        X, Y, D = self._zone_arrays(z)

        speed = np.hypot(D['VelocityX'], D['VelocityY'])
        if speed[0].mean() > speed[-1].mean():           # wall is currently at j = -1
            X, Y = X[::-1], Y[::-1]
            D = {k: v[::-1] for k, v in D.items()}
        if D['Pressure'][:, 0].mean() > D['Pressure'][:, -1].mean():   # i runs upstream
            X, Y = X[:, ::-1], Y[:, ::-1]
            D = {k: v[:, ::-1] for k, v in D.items()}

        self.X, self.Y, self.D = X, Y, D
        self.Xc = 0.25*(X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:])
        self.Yc = 0.25*(Y[:-1, :-1] + Y[1:, :-1] + Y[:-1, 1:] + Y[1:, 1:])
        self.rho, self.p, self.T = D['Density'], D['Pressure'], D['Temperature']
        self.u, self.v, self.M = D['VelocityX'], D['VelocityY'], D['Mach']

        # |grad rho| on a curvilinear grid, via the chain rule.  With the
        # Jacobian J = x_i y_j - x_j y_i,
        #     rho_x = (rho_i y_j - rho_j y_i) / J
        #     rho_y = (rho_j x_i - rho_i x_j) / J
        xi, xj = np.gradient(self.Xc, axis=1), np.gradient(self.Xc, axis=0)
        yi, yj = np.gradient(self.Yc, axis=1), np.gradient(self.Yc, axis=0)
        J = xi*yj - xj*yi
        self.J = np.where(np.abs(J) < 1e-30, np.nan, J)
        ri, rj = np.gradient(self.rho, axis=1), np.gradient(self.rho, axis=0)
        self.gmag = np.hypot((ri*yj - rj*yi)/self.J, (rj*xi - ri*xj)/self.J)

    def _load_wall(self, zones):
        """Concatenate the wall zones into single distributions along x.

        Cf is projected onto the local wall tangent so that its sign is
        meaningful: positive means the flow at the wall moves downstream,
        negative means it is reversed.  The tangent is taken from the wall
        geometry itself, so a plate and a ramp are handled the same way.
        """
        # Which points are on a VISCOUS wall.  ADflow names the surface zones
        # after the BC type -- EulerWallBCZone* for the two inviscid slip
        # segments, NSWallIsothermal*/NSWallAdiabatic* for the plate and ramp --
        # so the split is readable straight off the file.  It matters: y+ and
        # T_w are only defined on a no-slip wall, and averaging the slip
        # segments into them is what used to make this script report a T_w/T_inf
        # well below the wall's actual value and a y+ minimum of 0.000 that was
        # just the inviscid wall's zero shear.
        xs, ys, pr, cf, rw, Tw, cfm, vs = [], [], [], [], [], [], [], []
        wallBC = set()
        for z in zones:
            viscous = 'NSWall' in z.name
            if viscous:
                wallBC.add('isothermal' if 'Isothermal' in z.name else
                           'adiabatic' if 'Adiabatic' in z.name else 'viscous')
            X, Y, D = self._zone_arrays(z)
            order = np.argsort(X[0])
            xv, yv = X[0][order], Y[0][order]
            xs.append(0.5*(xv[:-1] + xv[1:]))          # vertex -> cell centre
            ys.append(0.5*(yv[:-1] + yv[1:]))

            tx, ty = np.diff(xv), np.diff(yv)          # unit wall tangent
            seg = np.hypot(tx, ty)
            tx, ty = tx/seg, ty/seg

            cell = np.argsort(0.5*(X[0][:-1] + X[0][1:]))
            get = lambda n: D[n][0][cell]
            pr.append(get('Pressure'))          # already p/p_inf; see note 3 above
            cf.append(get('SkinFrictionX')*tx + get('SkinFrictionY')*ty)
            rw.append(get('Density'))
            Tw.append(get('Temperature'))
            cfm.append(get('SkinFrictionMagnitude'))
            vs.append(np.full(xs[-1].shape, viscous))

        cat = np.concatenate
        o = np.argsort(cat(xs))
        self.xw, self.yw = cat(xs)[o], cat(ys)[o]
        self.pw, self.cfw = cat(pr)[o], cat(cf)[o]
        self.rww, self.Tww, self.cfm = cat(rw)[o], cat(Tw)[o], cat(cfm)[o]
        self.viscw = cat(vs)[o]
        self.wallBC = '/'.join(sorted(wallBC)) if wallBC else 'viscous'
        # Same mask on the field-cell abscissa self.xs, for y+ (set in
        # _derive_quantities, which runs after this).
        self._visc_x = (self.xw[self.viscw].min(), self.xw[self.viscw].max()) \
                       if self.viscw.any() else (-np.inf, np.inf)

    # --- derived quantities -------------------------------------------------

    def _derive_quantities(self):
        """Separation, y+, boundary-layer state, ramp angle, shock angle."""
        x, f = self.xw, self.cfw

        # Separation and reattachment are the zero crossings of Cf: falling
        # through zero = separation, rising through zero = reattachment.
        def crossings(rising):
            out = []
            for i in range(len(f) - 1):
                if f[i] != 0 and (f[i] > 0) != (f[i+1] > 0) \
                        and (f[i+1] > f[i]) == rising:
                    out.append(x[i] + (x[i+1] - x[i])*(-f[i])/(f[i+1] - f[i]))
            return out
        sep, rea = crossings(False), crossings(True)
        self.x_sep = min(sep) if sep else np.nan
        self.x_rea = max(rea) if rea else np.nan
        self.Lsep = self.x_rea - self.x_sep

        # y+ of the first cell:  y+ = rho_w u_tau y / mu_w,  u_tau = sqrt(tau_w/rho_w).
        # The wall quantities are non-dimensional, so multiply back to SI first.
        self.xs = self.Xc[0]
        self.dw = np.hypot(self.Xc[0] - 0.5*(self.X[0, :-1] + self.X[0, 1:]),
                           self.Yc[0] - 0.5*(self.Y[0, :-1] + self.Y[0, 1:]))
        tau = np.interp(self.xs, x, self.cfm)*self.QD*self.Pref
        rho_w = np.interp(self.xs, x, self.rww)*self.Rref
        T_w = np.interp(self.xs, x, self.Tww)*self.Tref
        self.utau = np.sqrt(np.abs(tau)/rho_w)
        self.yplus = rho_w*self.utau*self.dw/mu(T_w)
        # y+ is a no-slip quantity.  Keep the full array for plotting (the
        # inviscid stretches read ~0 and that is honest on a graph), but expose
        # a viscous-only view for the min/mean/max the table reports.
        lo, hi = self._visc_x
        self.yplusv = self.yplus[(self.xs >= lo) & (self.xs <= hi)]

        # Incoming boundary layer, sampled 15 mm upstream of separation.
        self.x_ref = self.x_sep - 0.015
        self.delta0, self.Retau, self.profile = self.boundary_layer(self.x_ref)

        # The same layer at the experiment's fixed station, for comparison with
        # DELTA_EXP / RE_DELTA_EXP.  Kept separate from delta0 above: that one
        # follows x_sep and so sits at a different x in every run, which makes
        # it useless for checking whether the incoming layer matches the target.
        self.x_exp = X_EXP
        self.delta_exp, self.Retau_exp, self.profile_exp = self.boundary_layer(X_EXP)
        self.Re_delta = self.Re_m*self.delta_exp

        # Ramp angle: straight-line fit through the ramp, and the ramp ONLY.
        #
        # The old window was x > 0.25*x.max(), which is wrong whenever the mesh
        # carries an inviscid slip wall past the ramp (mesh.py --invBackLength):
        # x.max() is then the end of that horizontal wall, so the window spans
        # the ramp AND the flat stretch behind it and one line through both
        # returns the average.  On comp_corner_18 that is 16.40 deg for a
        # geometrically exact 25 deg ramp.
        #
        # It is not a cosmetic error -- self.ramp is the wedge angle handed to
        # oblique_shock_beta, so it also sets `beta theory` and the inviscid
        # p/p_inf drawn as the dashed line on the wall-pressure panel.  At
        # M = 2.95 the difference is beta 34.10 vs 44.59 deg and p/p_inf 3.025
        # vs 4.836, i.e. the figure claimed the solution missed inviscid theory
        # by 60% when it actually sits within 0.1% of it.
        #
        # The ramp runs from the compression corner up to the crest, so take
        # the first node at max y as its downstream end and trim 5% off each
        # end to drop the two kink cells, whose j-lines bisect the turn.
        x_top = x[int(np.argmax(self.yw))]
        m = (x > 0.05*x_top) & (x < 0.95*x_top)
        self.ramp = np.degrees(np.arctan(np.polyfit(x[m], self.yw[m], 1)[0]))

        self._fit_shock()

        self.x_trans = self._find_transition()

    # Fraction by which Cf must climb past its minimum before the rise counts
    # as transition rather than as noise on a monotone turbulent decay.
    TRANS_RISE = 0.25
    # Keep the search window this far clear of separation, and of the corner
    # when the flow stays attached, so the interaction is never inside it.
    TRANS_MARGIN = 0.005

    def _find_transition(self):
        """Transition location on the plate, or NaN if there is none.

        A transition leaves a minimum in Cf followed by a climb to the
        turbulent level; the transition point is the steepest part of that
        climb.  Where Cf instead decays monotonically from the leading edge
        the layer is turbulent throughout and there is nothing to report, so
        this returns NaN rather than snapping to the end of the search window.

        Note that in a SA run this is a numerical transition, not a physical
        one: SA has no transition model, and the eddy viscosity simply takes
        some distance to grow from its freestream level to the turbulent one.
        A run started with a high enough freestream eddy-viscosity ratio is
        turbulent from the leading edge and yields NaN here.

        The window is confined to attached flow on the plate, clear of the
        interaction, so the Cf recovery through reattachment and the rise at
        the corner cannot be mistaken for a transition.
        """
        x, f = self.xw, self.cfw
        x_end = self.x_sep if np.isfinite(self.x_sep) else 0.0
        m = (x < min(x_end, 0.0) - self.TRANS_MARGIN) & (f > 0)
        if m.sum() < 10:
            return np.nan

        xm, fm = x[m], f[m]
        i = int(np.argmin(fm))                       # end of the laminar run
        if i >= len(fm) - 3:                         # minimum sits at the window
            return np.nan                            # edge => Cf never recovers
        if fm[i+1:].max() < (1.0 + self.TRANS_RISE)*fm[i]:
            return np.nan                            # rise too weak to be real

        g = np.gradient(fm, xm)
        return float(xm[i + 1 + int(np.argmax(g[i+1:]))])

    def _fit_shock(self):
        """Fit the shock angle from the pressure field.

        Along each grid row that lies well outside the boundary layer, the
        shock is located as the first point where pressure crosses halfway
        between freestream and the post-shock value.  A straight line through
        those points gives beta.  This is steadier than chasing max|grad rho|,
        which can lock onto the top boundary or the shear layer.
        """
        pts = []
        for j in range(self.Yc.shape[0]):
            y_row = self.Yc[j]
            height = y_row - np.interp(self.Xc[j], self.xw, self.yw)
            p_row = self.p[j]
            if not (height.mean() > 5*self.delta0 and p_row.max() > 2.0):
                continue
            k = np.where(p_row > 0.5*(1.0 + p_row.max()))[0]
            if len(k) and 3 < k[0] < len(p_row) - 3:
                pts.append((self.Xc[j, k[0]], y_row[k[0]]))
        self.shock_pts = np.array(pts)
        self.beta_fit = np.nan
        if len(pts) > 5:
            slope = np.polyfit(self.shock_pts[:, 1], self.shock_pts[:, 0], 1)[0]
            self.beta_fit = np.degrees(np.arctan(1/slope))

    def boundary_layer(self, x_station):
        """Boundary-layer thickness, Re_tau and wall-scaled profile at a station.

        Returns
        -------
        delta99 : thickness where the speed reaches 0.99 of freestream, metres
        Re_tau  : friction Reynolds number, delta99 in wall units
        profile : (y+, u+, u+_vanDriest)

        The van Driest transform,

            u+_VD = (1/u_tau) * integral sqrt(rho/rho_w) du,

        removes the density variation across a compressible boundary layer so
        that the profile can be compared against the incompressible log law.
        A turbulent layer should follow u+ = y+ near the wall and
        u+ = ln(y+)/0.41 + 5.2 in the log region; if it does not, the layer is
        not turbulent there.
        """
        i = int(np.argmin(np.abs(self.xs - x_station)))
        V = np.hypot(self.u[:, i], self.v[:, i])
        k = int(np.argmax(V >= 0.99*self.ue))
        delta = np.interp(0.99*self.ue, V[:k+1], self.Yc[:k+1, i])

        cf = np.interp(x_station, self.xw, self.cfm)
        rho_w = np.interp(x_station, self.xw, self.rww)*self.Rref
        T_w = np.interp(x_station, self.xw, self.Tww)*self.Tref
        utau = np.sqrt(cf*self.QD*self.Pref/rho_w)

        yplus = rho_w*utau*self.Yc[:, i]/mu(T_w)
        Vdim = V*self.uref
        rho = self.rho[:, i]*self.Rref
        uvd = np.concatenate([[0], np.cumsum(
            np.sqrt(0.5*(rho[1:] + rho[:-1])/rho_w)*np.diff(Vdim))])/utau
        return delta, rho_w*utau*delta/mu(T_w), (yplus, Vdim/utau, uvd)

    # --- convenience --------------------------------------------------------

    @property
    def cpw(self):
        """Wall Cp, from the pressure ratio.

        The two normalisations differ only in what you divide by:

            p/p_inf = 1 + (gamma/2) M_inf^2 Cp = 1 + QD Cp

        because for a perfect gas q_inf = 0.5 rho_inf U_inf^2 = 0.5 gamma
        p_inf M_inf^2, i.e. exactly self.QD in these non-dimensional units.
        p/p_inf is 1 in the freestream, Cp is 0 there.
        """
        return (self.pw - 1.0)/self.QD

    @property
    def p_inviscid(self):
        """p/p_inf behind the inviscid oblique shock for this Mach and ramp.

        Rankine-Hugoniot across the wave, using the Mach number normal to it:

            p2/p1 = (2 gamma Mn1^2 - (gamma - 1)) / (gamma + 1),
            Mn1   = M_inf sin(beta).
        """
        beta = oblique_shock_beta(self.Minf, np.radians(self.ramp))
        Mn = self.Minf*np.sin(beta)
        return (2*GAM*Mn**2 - (GAM - 1))/(GAM + 1)

    @property
    def cp_inviscid(self):
        """Cp behind the inviscid oblique shock, i.e. p_inviscid as a Cp."""
        return (self.p_inviscid - 1.0)/self.QD


# =============================================================================
#  Part 3b   Reference data digitized from the paper
# =============================================================================
#
# The comparison curves in context/ come out of a plot digitizer, so they are
# not quite a function of x.  On the steep parts -- the pressure jump, the Cf
# plunge into the bubble, the spike at the corner -- the operator's x scatters
# by a few tenths of a mm while y marches on, and in two places the trace
# doubles back over itself.  x therefore decreases 31 times in cf_xl.csv.
# Joined with a line those show up as zig-zags; drawn as markers the file
# reads as scattered data, which it is not -- the paper draws one curve.
#
# Turning it back into that curve takes three steps:
#
#   1. sort by x and average y over the samples that share one x.  The jitter
#      is small compared with the spacing of genuinely distinct stations, so
#      sorting reorders only within the doubled-back runs, and averaging puts
#      the curve through the middle of them.  What survives is the height of
#      those near-vertical runs (~4e-4 in Cf at x = -9.7 and x = -0.3), which
#      no single-valued y(x) can reproduce and which is invisible at the
#      scale these panels are drawn at.
#   2. interpolate with a monotone cubic (Fritsch-Carlson / PCHIP).  A natural
#      cubic spline would ring at the pressure jump and dip Cf below its own
#      data inside the bubble; the monotone form cannot overshoot, so the
#      plateau stays flat and the jump stays a jump.
#   3. evaluate on a dense uniform grid, so the drawn line is smooth at any
#      figure size.
#
# Implemented here rather than pulled from scipy so that the script keeps its
# numpy + matplotlib only requirement.

REF_LABEL = 'Hao'
# Electric violet, chosen by maximizing the smallest CIELAB distance to every
# other colour on panels (a) and (b): ADflow's dark green, the crimson of the
# inviscid line and the S/R markers, the pale orange band, and the black/grey
# of the axes.  Its nearest neighbour is crimson at dE 125 -- and crimson,
# marking separation and reattachment, is the one it must not be confused
# with.  Magenta, the obvious alternative, manages only dE 55 there.  The
# dashes keep the curve separable in greyscale as well; short ones (3 on,
# 1.4 off) so the Cf spike at the corner still resolves as a spike.
REF_COLOR = '#8b00ff'
REF_DASH = (0, (3, 1.4))

# Both wall panels draw ADflow in one colour, so the eye carries "this is the
# computation" from the pressure panel down to the skin-friction panel.
ADFLOW_COLOR = '#0b6b3a'
N_REF = 1200                   # samples in the resampled reference curve


def _pchip_slopes(x, y):
    """Fritsch-Carlson tangents: the derivative estimate that cannot overshoot.

    At an interior node the harmonic mean of the two neighbouring secants is
    used, which vanishes whenever they disagree in sign.  That is what pins
    the interpolant to the data at a local extremum (the bottom of the Cf
    bubble, the top of the pressure plateau) instead of letting it swing past.
    """
    h = np.diff(x)
    d = np.diff(y)/h                                  # secant slopes
    m = np.zeros_like(y)

    # Interior nodes: harmonic mean, weighted by the interval lengths.
    same = np.sign(d[:-1])*np.sign(d[1:]) > 0
    w1, w2 = 2*h[1:] + h[:-1], h[1:] + 2*h[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        m[1:-1] = np.where(same, (w1 + w2)/(w1/d[:-1] + w2/d[1:]), 0.0)

    # Ends: one-sided three-point formula, clipped so it stays monotone.
    def end(d0, d1, h0, h1):
        s = ((2*h0 + h1)*d0 - h0*d1)/(h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3*d0):
            return 3*d0
        return s

    m[0] = end(d[0], d[1], h[0], h[1]) if len(d) > 1 else d[0]
    m[-1] = end(d[-1], d[-2], h[-1], h[-2]) if len(d) > 1 else d[-1]
    return m


def _pchip(x, y, xq):
    """Evaluate the monotone cubic through (x, y) at xq, on the cubic Hermite
    basis of each interval."""
    m = _pchip_slopes(x, y)
    k = np.clip(np.searchsorted(x, xq) - 1, 0, len(x) - 2)
    h = x[k+1] - x[k]
    t = (xq - x[k])/h
    t2, t3 = t*t, t*t*t
    return ((2*t3 - 3*t2 + 1)*y[k] + (t3 - 2*t2 + t)*h*m[k]
            + (-2*t3 + 3*t2)*y[k+1] + (t3 - t2)*h*m[k+1])


def load_reference(path, n=N_REF):
    """One digitized curve as a continuous line.

    Returns (x, y) sampled on a uniform grid across the file's x range, or
    None if the file is missing or too short to interpolate.  The CSV is a
    two-column "x, y" export with a header line; x is in mm, y in whatever
    the panel plots.
    """
    if not os.path.isfile(path):
        return None
    d = np.atleast_2d(np.loadtxt(path, delimiter=',', skiprows=1))
    if d.shape[0] < 2:
        return None
    x, y = d[:, 0], d[:, 1]

    # Sort, then average the ties.  np.unique gives, for every sample, the
    # index of the station it belongs to, so bincount sums and counts the
    # duplicates in one pass.
    xu, inv = np.unique(x, return_inverse=True)
    yu = np.bincount(inv, weights=y)/np.bincount(inv)
    if len(xu) < 2:
        return None

    xq = np.linspace(xu[0], xu[-1], n)
    return xq, _pchip(xu, yu, xq)


def zero_crossings(x, y):
    """(x_sep, x_rea) of a Cf curve: where it falls through zero and where it
    rises back.

    Same convention as Case._derive_quantities -- the FIRST falling crossing
    and the LAST rising one -- so that the small positive spike the corner
    puts in the middle of the bubble is stepped over rather than mistaken for
    reattachment followed by a second separation.  Kept here as a free
    function because it has to run on the digitized curve too, which is not a
    Case and has none of a Case's geometry.
    """
    def crossings(rising):
        out = []
        for i in range(len(y) - 1):
            if y[i] != 0 and (y[i] > 0) != (y[i+1] > 0) \
                    and (y[i+1] > y[i]) == rising:
                out.append(x[i] + (x[i+1] - x[i])*(-y[i])/(y[i+1] - y[i]))
        return out
    sep, rea = crossings(False), crossings(True)
    return (min(sep) if sep else np.nan,
            max(rea) if rea else np.nan)


def load_context(dirname):
    """The paper's wall curves, keyed by the panel they belong to.

    p*.csv    wall pressure ratio p/p_inf   -> panel (a)
    cf*.csv   skin friction Cf              -> panel (b)

    Matched by prefix rather than by exact name: the lowRe and highRe
    directories hold the same two curves under different names (cf_xl.csv vs
    cf_xl_high.csv), and a file that has been renamed should still be found.
    The cf glob is tried first so that "cf*" never loses its files to "p*".
    """
    if not dirname or not os.path.isdir(dirname):
        return {}
    out = {}
    for key, pattern in (('cf', 'cf*.csv'), ('p', 'p*.csv')):
        for path in sorted(glob.glob(os.path.join(dirname, pattern))):
            curve = load_reference(path)
            if curve is not None:
                out[key] = curve
                break
    return out


# =============================================================================
#  Part 4   Streamline tracing
# =============================================================================
#
# The grid is curvilinear, so matplotlib's streamplot (which needs an evenly
# spaced grid) cannot be used without interpolating first.  Instead we trace
# in *index* space: convert the velocity to its contravariant components, so
# that a step in (i, j) follows the flow, and integrate there.  No resampling,
# no loss of near-wall detail.

def contravariant(Xc, Yc, u, v):
    """Velocity expressed in grid-index directions.

        U^i = (u y_j - v x_j) / J
        U^j = (v x_i - u y_i) / J     with  J = x_i y_j - x_j y_i
    """
    xi, xj = np.gradient(Xc, axis=1), np.gradient(Xc, axis=0)
    yi, yj = np.gradient(Yc, axis=1), np.gradient(Yc, axis=0)
    J = xi*yj - xj*yi
    J = np.where(np.abs(J) < 1e-30, np.nan, J)
    return (u*yj - v*xj)/J, (v*xi - u*yi)/J


def bilinear(A, fi, fj):
    """Value of array A at fractional index (fi, fj)."""
    nj, ni = A.shape
    i0 = int(np.clip(np.floor(fi), 0, ni - 2))
    j0 = int(np.clip(np.floor(fj), 0, nj - 2))
    a, b = fi - i0, fj - j0
    return ((1-a)*(1-b)*A[j0, i0] + a*(1-b)*A[j0, i0+1]
            + (1-a)*b*A[j0+1, i0] + a*b*A[j0+1, i0+1])


def streamline(Ui, Uj, Xc, Yc, fi, fj, ds=0.4, nmax=6000, sign=1):
    """Trace one streamline from index position (fi, fj).

    Second-order Runge-Kutta with a fixed step in index space.  `sign = -1`
    traces upstream, which is how the recirculating streamlines inside the
    separation bubble are drawn.  Returns an (n, 2) array of x, y in metres.
    """
    nj, ni = Ui.shape
    pts = []

    def direction(a, b):
        p, q = bilinear(Ui, a, b)*sign, bilinear(Uj, a, b)*sign
        mag = np.hypot(p, q)
        return (0, 0) if (not np.isfinite(mag) or mag < 1e-14) else (p/mag, q/mag)

    for _ in range(nmax):
        if not (0 <= fi <= ni - 1.001 and 0 <= fj <= nj - 1.001):
            break
        pts.append((bilinear(Xc, fi, fj), bilinear(Yc, fi, fj)))
        k1 = direction(fi, fj)
        if k1 == (0, 0):
            break
        k2 = direction(fi + 0.5*ds*k1[0], fj + 0.5*ds*k1[1]) or k1
        if k2 == (0, 0):
            k2 = k1
        fi += ds*k2[0]
        fj += ds*k2[1]
    return np.array(pts) if pts else np.zeros((0, 2))


# =============================================================================
#  Part 5   Figures
# =============================================================================

def _draw_wall(case, ax, floor):
    """Solid wall line with the solid body shaded below it."""
    ax.plot(case.X[0]*MM, case.Y[0]*MM, 'k-', lw=1.7, zorder=6)
    ax.fill_between(case.X[0]*MM, case.Y[0]*MM, floor, color='0.82', zorder=5)


def figure_flowfield(case, fname, exag=3.5):
    """Three-panel view of the flow field.

    (a) Mach number over the whole domain, with the sonic line.
    (b) Numerical schlieren, exp(-k |grad rho|).  This mimics a schlieren
        photograph: shocks and shear layers appear dark because density
        changes sharply across them.  It is the clearest way to see the
        separation shock, the reattachment shock and where they merge.
    (c) Close-up of the interaction, stretched vertically so the thin
        separation bubble is visible, with streamlines, the sonic line, the
        reverse-flow region hatched, and the inviscid shock angle for
        reference.
    """
    beta = oblique_shock_beta(case.Minf, np.radians(case.ramp))
    Xm, Ym = case.X*MM, case.Y*MM
    Xcm, Ycm = case.Xc*MM, case.Yc*MM
    x0d, x1d, y1d = Xm.min(), Xm.max(), Ym.max()
    vmax = np.ceil(case.Minf*10)/10

    fig = plt.figure(figsize=(13.4, 11.6))
    # right=0.93, not 0.985.  Panels (a) and (b) are set_aspect('equal'), so
    # their axes shrink to fit and their colorbars land inside the canvas
    # whatever this is.  Panel (c) is vertically exaggerated and has no aspect
    # constraint, so it fills the full width and pushes its colorbar -- ticks
    # and the 'M' label -- off the right edge.  This is the margin that keeps
    # the key on the page; it costs (c) a little width and (a)/(b) nothing.
    gs = GridSpec(3, 1, height_ratios=[1, 1, 1.7], hspace=0.30,
                  left=0.07, right=0.93, top=0.955, bottom=0.05)

    # --- (a) Mach number ---------------------------------------------------
    ax = fig.add_subplot(gs[0])
    pc = ax.pcolormesh(Xm, Ym, case.M, cmap='turbo', vmin=0, vmax=vmax,
                       shading='flat', rasterized=True)
    ax.contour(Xcm, Ycm, case.M, [1.0], colors='w', linewidths=0.9, zorder=8)
    _draw_wall(case, ax, -3)
    ax.set_xlim(x0d, x1d); ax.set_ylim(-1.2, y1d*1.02); ax.set_aspect('equal')
    ax.set_ylabel('y [mm]')
    fig.colorbar(pc, ax=ax, pad=.01, fraction=.022).set_label('M')
    # The wall label used to be the hard-coded string 'adiabatic wall', which was
    # simply false: both Zheltovodov cases are run ISOTHERMAL at 275.4 K (see
    # mesh.py --Twall, applied to the CGNS by fix_bc.py).
    #
    # Take the label from the BOUNDARY CONDITION, not from comparing T_w against
    # the recovery temperature.  A temperature test looks reasonable and is a
    # trap: in the lowRe case T_w/T_inf = 2.550 sits only 0.6% below
    # the adiabatic 2.566,
    # so a "within 1% of adiabatic" rule would label an isothermal wall adiabatic
    # on a numerical coincidence.  ADflow names the surface zone after the BC it
    # applied -- NSWallIsothermalBCZone* here -- so the file already says it.
    Tw_ratio = float(np.nanmean(case.Tww[case.viscw]))
    wall = ('adiabatic wall' if case.wallBC == 'adiabatic' else
            'isothermal wall T$_w$=%.1f K (T$_w$/T$_\\infty$=%.2f)'
            % (Tw_ratio*case.Tref, Tw_ratio) if case.wallBC == 'isothermal' else
            'viscous wall T$_w$/T$_\\infty$=%.2f' % Tw_ratio)
    ax.set_title('(a)  Mach number  —  %s  |  M$_\\infty$=%.2f, %.1f° corner, '
                 '%s, Re=%.2f×10$^7$ m$^{-1}$'
                 % (case.label, case.Minf, case.ramp, wall, case.Re_m/1e7),
                 loc='left', fontsize=10.5, pad=5)
    ax.text(x0d + 0.02*(x1d - x0d), 0.83*y1d, 'white: sonic line M = 1',
            color='w', fontsize=8.5)

    # --- (b) numerical schlieren -------------------------------------------
    ax = fig.add_subplot(gs[1])
    g = np.nan_to_num(case.gmag)
    g = g/np.percentile(g, 99.5)               # clip the few extreme cells
    ax.pcolormesh(Xm, Ym, np.exp(-6.0*np.clip(g, 0, 1.6)), cmap='gray',
                  vmin=0.02, vmax=1.0, shading='flat', rasterized=True)
    if len(case.shock_pts):
        ax.plot(case.shock_pts[:, 0]*MM, case.shock_pts[:, 1]*MM,
                ls=(0, (2, 3)), color='tab:red', lw=1.2, zorder=8)
    _draw_wall(case, ax, -3)
    ax.set_xlim(x0d, x1d); ax.set_ylim(-1.2, y1d*1.02); ax.set_aspect('equal')
    ax.set_ylabel('y [mm]')
    ax.set_title('(b)  Numerical schlieren $\\exp(-k|\\nabla\\rho|)$  —  red dots: '
                 'fitted shock locus, β = %.1f° (inviscid %.1f°)'
                 % (case.beta_fit, np.degrees(beta)), loc='left', fontsize=10.5, pad=5)
    if np.isfinite(case.x_trans):
        ax.annotate('laminar→turbulent\ntransition', xy=(case.x_trans*MM, 1.0),
                    xytext=(case.x_trans*MM - 0.20*(x1d - x0d), 0.35*y1d),
                    fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=.9))

    # --- (c) interaction close-up ------------------------------------------
    ax = fig.add_subplot(gs[2])
    d0 = case.delta0*MM
    x0 = case.x_sep*MM - 9*d0
    x1 = case.x_rea*MM + 9*d0
    # Always keep the experiment's measuring station in frame -- it is what
    # --plateLength is calibrated against, and it is annotated below.  It lands
    # inside the 9*delta0 run-in for the highRe case but ~2 mm outside it for
    # the lowRe one, so widen rather than let the marker fall off the axes.
    x0 = min(x0, X_EXP*MM - 2.2*d0)
    y_reatt = np.interp(case.x_rea, case.xw, case.yw)*MM
    y0, y1 = -0.95*d0, y_reatt + 3.0*d0

    pc = ax.pcolormesh(Xm, Ym, case.M, cmap='turbo', vmin=0, vmax=vmax,
                       shading='flat', rasterized=True)
    ax.contour(Xcm, Ycm, case.M, [1.0], colors='w', linewidths=1.1, zorder=8)
    ax.contour(Xcm, Ycm, case.u, [0.0], colors='k', linewidths=1.8, zorder=9)
    ax.contourf(Xcm, Ycm, np.where(case.u < 0, -1., 1.), levels=[-2, 0],
                colors='none', hatches=['////'], zorder=6.5)

    Ui, Uj = contravariant(case.Xc, case.Yc, case.u, case.v)
    xs = case.Xc[0]
    index_of = lambda xt: float(np.argmin(np.abs(xs - xt)))
    # White: streamlines entering from upstream, seeded across the layer.
    for j in [1, 3, 6, 10, 16, 24, 34, 46, 60, 76, 95, 115, 138]:
        if j >= case.u.shape[0]:
            continue
        P = streamline(Ui, Uj, case.Xc, case.Yc,
                       index_of(x0/MM + 2*d0/MM), float(j), ds=0.5, nmax=5000)
        if len(P) > 3:
            ax.plot(P[:, 0]*MM, P[:, 1]*MM, color='w', lw=0.85, alpha=.9, zorder=7)
    # Dark: short traces seeded inside the bubble, both directions.
    for x_seed, j in [(case.x_sep + 0.4*case.Lsep, 6), (case.x_sep + 0.75*case.Lsep, 10)]:
        for s in (1, -1):
            P = streamline(Ui, Uj, case.Xc, case.Yc, index_of(x_seed), float(j),
                           ds=0.3, nmax=700, sign=s)
            if len(P) > 3:
                ax.plot(P[:, 0]*MM, P[:, 1]*MM, color='0.1', lw=0.8, zorder=7)

    _draw_wall(case, ax, y0 - 3)

    # Separation and reattachment.  The letters sit next to their own markers
    # rather than out in the free stream, on a dark patch so they read over
    # whatever the colormap is doing there (R used to be white-on-cyan), and
    # each one drops a dashed line to the axis carrying its x.  The dash
    # pattern and the white halo keep these apart from the inviscid-shock line,
    # which is also dashed and also white.
    halo = [pe.withStroke(linewidth=3.0, foreground='w')]
    y_sep = np.interp(case.x_sep, case.xw, case.yw)*MM
    marks = [('S', case.x_sep*MM, y_sep,   -1),
             ('R', case.x_rea*MM, y_reatt, +1)]
    for tag, xm, ym, side in marks:
        ax.plot([xm], [ym], 'o', ms=7, mfc='none', mec='w', mew=1.8, zorder=10)
        ax.plot([xm, xm], [y0, ym], color='0.1', lw=1.3, ls=(0, (5, 3)),
                zorder=9.5, path_effects=halo)
        # leader from the badge to the exact point, so the letter cannot be
        # read as labelling whichever streamline it happens to sit on
        ax.annotate(tag, xy=(xm, ym), xytext=(xm + side*0.85*d0, ym + 0.95*d0),
                    color='w', fontsize=12, weight='bold', ha='center',
                    va='center', zorder=11,
                    bbox=dict(boxstyle='circle,pad=0.22', fc='0.1', ec='w', lw=1.2),
                    arrowprops=dict(arrowstyle='-', color='w', lw=1.5,
                                    shrinkA=3, shrinkB=5,
                                    path_effects=[pe.withStroke(linewidth=3.0,
                                                                foreground='0.1')]))
        ax.text(xm, y0 + 0.08*d0, '%.2f' % xm, color='0.1', fontsize=9,
                weight='bold', ha='center', va='bottom', zorder=12,
                bbox=dict(boxstyle='round,pad=0.18', fc='w', ec='0.1', lw=0.8))

    # Span between them, i.e. the separation length the title quotes.
    y_span = y0 + 0.62*d0
    ax.annotate('', xy=(case.x_sep*MM, y_span), xytext=(case.x_rea*MM, y_span),
                arrowprops=dict(arrowstyle='<->', color='0.1', lw=1.3,
                                shrinkA=0, shrinkB=0), zorder=11)
    ax.text(0.5*(case.x_sep + case.x_rea)*MM, y_span + 0.06*d0,
            r'$L_{sep}$ = %.2f mm' % (case.Lsep*MM), color='0.1', fontsize=9,
            weight='bold', ha='center', va='bottom', zorder=11,
            path_effects=halo)
    # --- the experiment's measuring station ---------------------------------
    # delta at this fixed station, and the Re_delta built on it, are the two
    # numbers the whole case is calibrated to (mesh.py --plateLength), so draw
    # them rather than leaving them in the table.  The bar is the boundary-layer
    # thickness TO SCALE, i.e. it carries the panel's vertical exaggeration like
    # everything else here.
    xe = X_EXP*MM
    ye = np.interp(X_EXP, case.xw, case.yw)*MM
    de = case.delta_exp*MM
    cap = 0.20*d0
    ax.plot([xe, xe], [ye, ye + de], color='#ffd400', lw=2.6,
            solid_capstyle='butt', zorder=10.6, path_effects=halo)
    for yy in (ye, ye + de):
        ax.plot([xe - cap, xe + cap], [yy, yy], color='#ffd400', lw=2.2,
                zorder=10.6, path_effects=halo)
    # Anchor the caption in AXES fraction, not data coordinates, with a leader
    # to the bar.  Placing it beside the bar in data units overflows the left
    # spine whenever the station sits near the edge of the window -- which is
    # exactly the lowRe case, where -15.4*delta lands 2 mm outside the
    # separation-centred view and the caption ran off the axes.
    ax.annotate('$\\delta$ = %.3f mm\n$Re_\\delta$ = %s\nat x = %.2f mm = %.2f$\\delta$'
                % (de, format(int(round(case.Re_delta)), ',').replace(',', '\u2009'),
                   xe, X_EXP/DELTA_EXP),
                xy=(xe, ye + de), xycoords='data',
                xytext=(0.015, 0.72), textcoords='axes fraction',
                color='0.1', fontsize=8.5, ha='left', va='top', zorder=11,
                bbox=dict(boxstyle='round,pad=0.30', fc='#fff4c2',
                          ec='#b38f00', lw=0.9),
                arrowprops=dict(arrowstyle='-', color='#ffd400', lw=1.6,
                                shrinkA=4, shrinkB=2, path_effects=halo))

    xl = np.array([0, (x1 - 0)*0.85])
    ax.plot(xl, np.tan(beta)*xl, color='w', ls=(0, (6, 4)), lw=1.3, zorder=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xlabel('x [mm]   (corner at x = 0)'); ax.set_ylabel('y [mm]')
    fig.colorbar(pc, ax=ax, pad=.01, fraction=.022).set_label('M')

    # Report the exaggeration that actually resulted from the layout.
    fig.canvas.draw()
    bb = ax.get_window_extent()
    stretch = ((x1 - x0)/bb.width)/((y1 - y0)/bb.height)
    ax.set_title('(c)  Interaction close-up, vertical exaggeration %.1f×  —  streamlines, '
                 'hatched: reverse flow (u < 0), white: sonic line, dashed: inviscid shock'
                 % stretch, loc='left', fontsize=10.5, pad=5)

    fig.savefig(fname, dpi=165)
    plt.close(fig)


def figure_wall(case, fname, ref=None):
    """Four-panel view of what happens at the wall.

    (a) p/p_inf: the plateau marks the separated region, and the final level
        should reach the inviscid oblique-shock value.
    (b) Cf: its zero crossings define separation (S) and reattachment (R).
    (c) y+ of the first cell.  Below 1 is wall-resolved; the shaded band is
        the buffer layer, which is the one place you do not want to sit.
    (d) The incoming boundary layer in wall units.  If the points follow
        u+ = y+ then the log law, the layer is genuinely turbulent there.

    ref, if given, is the dict from load_context(): the paper's digitized
    wall curves, drawn over panels (a) and (b) as continuous lines.
    """
    ref = ref or {}
    p_inv = case.p_inviscid
    beta = oblique_shock_beta(case.Minf, np.radians(case.ramp))
    xw = case.xw*MM
    xs, xr = case.x_sep*MM, case.x_rea*MM
    W = 4*case.Lsep*MM                      # plot window, scaled to the bubble

    fig = plt.figure(figsize=(12.4, 10.4))
    # right margin leaves room for the Cp axis mirrored on panel (a)
    gs = GridSpec(3, 2, height_ratios=[1, 1, 1], width_ratios=[2.1, 1],
                  hspace=0.40, wspace=0.24, left=0.075, right=0.935,
                  top=0.945, bottom=0.06)

    # --- (a) wall pressure ratio -------------------------------------------
    # p/p_inf is what ADflow stores, so it is plotted directly.  The right-hand
    # axis carries the same curve as Cp; the two are one affine map apart,
    #     p/p_inf = 1 + QD*Cp,   QD = 0.5*gamma*M_inf^2,
    # which for M_inf = 2.95 is p/p_inf = 1 + 6.092 Cp.
    ax = fig.add_subplot(gs[0, :])
    ax.plot(xw, case.pw, color=ADFLOW_COLOR, lw=2.0, label='ADflow (SA-Edwards)')
    if 'p' in ref:
        ax.plot(*ref['p'], color=REF_COLOR, lw=1.8, ls=REF_DASH,
                zorder=3, label=REF_LABEL)
    ax.axhline(p_inv, color='crimson', ls='--', lw=1.2,
               label='inviscid oblique shock, $p/p_\\infty$ = %.3f  ($C_p$ = %.3f)'
                     % (p_inv, case.cp_inviscid))
    ax.axvspan(xs, xr, color='#ffd9a0', alpha=.6, zorder=0,
               label='separated region')
    ax.axhline(1.0, color='.6', lw=.8)
    ax.set_xlim(-40, 30); ax.set_ylim(0, 5) # ax.set_xlim(-W, W); ax.set_ylim(0.85, 1 + (p_inv - 1)*1.25)
    ax.set_ylabel('$p/p_\\infty$')
    ax.legend(fontsize=9, loc='upper left'); ax.grid(alpha=.25)
    sec = ax.secondary_yaxis('right', functions=(lambda p: (p - 1.0)/case.QD,
                                                 lambda cp: 1.0 + case.QD*cp))
    sec.set_ylabel('$C_p$')
    ax.set_title('(a)  Wall pressure  —  %s' % case.label, loc='left', fontsize=10.5, pad=5)

    # --- (b) skin friction --------------------------------------------------
    ax = fig.add_subplot(gs[1, :])
    ax.plot(xw, case.cfw, color=ADFLOW_COLOR, lw=2.0, label='ADflow (SA-Edwards)')
    if 'cf' in ref:
        ax.plot(*ref['cf'], color=REF_COLOR, lw=1.8, ls=REF_DASH,
                zorder=3, label=REF_LABEL)
    ax.axhline(0, color='k', lw=1.0)
    ax.axvspan(xs, xr, color='#ffd9a0', alpha=.6, zorder=0,
               label='separated region')
    # After the span, not before: a legend only picks up what already exists.
    ax.legend(fontsize=9, loc='upper left')
    ax.set_xlim(-40, 30); ax.set_ylabel('$C_f$ (wall-tangent)'); ax.grid(alpha=.25); ax.set_ylim(-0.0030, 0.004)
    # Label S and R against the Cf curve -- Cf = 0 is what *defines* them, so
    # this is the panel where they mean something, and each badge is anchored
    # to the point (x, 0) on the curve it belongs to.
    #
    # Hao's stations are read off the digitized curve with the same rule the
    # solution gets, rather than taken from the paper's text: whatever bias
    # the digitizing introduced then lands on both the curve and its markers,
    # so the S-to-S and R-to-R gaps drawn here are a like-for-like comparison.
    #
    # Two rows, because the two curves separate within a millimetre of each
    # other and badges on one row would collide.  ADflow's sit at Cf =
    # -0.0019, Hao's below at -0.0026, which is what the extra 1e-3 of bottom
    # margin above is for: both rows have to clear the floor of the bubble
    # (-0.0014 here) as well as each other, since the offsets are not enough
    # to carry them clear of it horizontally.
    #
    # The outer offset belongs to the UPPER row.  Give both rows the same one
    # and the lower badge's leader, climbing inward from further down, rules a
    # line straight through the upper badge's text -- which is what it did.
    # Pairing outer-with-upper (8 mm) and inner-with-lower (3 mm) makes the
    # lower leader the steeper of the two, so it starts inboard of the upper
    # badge and never reaches it.  Each badge takes the colour of its own
    # curve -- the crimson these used to be matched neither one.  Offsets go
    # outward, S left and R right, so the leaders splay apart rather than
    # lean across the bubble between them.
    halo = [pe.withStroke(linewidth=2.6, foreground='w')]
    rows = [(xs, xr, ADFLOW_COLOR, -0.0019, 8.0)]
    if 'cf' in ref:
        hs, hr = zero_crossings(*ref['cf'])
        if np.isfinite(hs) and np.isfinite(hr):
            rows.append((hs, hr, REF_COLOR, -0.0026, 3.0))
    for x_s, x_r, colour, y_lab, dx in rows:
        for tag, xx, side in [('S', x_s, -1), ('R', x_r, +1)]:
            ax.plot([xx], [0.0], 'o', ms=6, mfc='w', mec=colour, mew=1.6,
                    zorder=6)
            ax.annotate('%s  %.2f mm' % (tag, xx), xy=(xx, 0.0),
                        xytext=(xx + side*dx, y_lab),
                        color=colour, fontsize=9, weight='bold',
                        ha='center', va='center', zorder=7, path_effects=halo,
                        arrowprops=dict(arrowstyle='-', color=colour, lw=1.1,
                                        shrinkA=3, shrinkB=5,
                                        path_effects=halo))
    title = ('(b)  Skin friction  —  L$_{sep}$ = %.2f mm = %.1f δ$_0$'
             % (case.Lsep*MM, case.Lsep/case.delta0))
    if len(rows) > 1:
        title += '   (%s: %.2f mm, %+.1f%%)' % (
            REF_LABEL, rows[1][1] - rows[1][0],
            100*((rows[1][1] - rows[1][0])/(case.Lsep*MM) - 1))
    ax.set_title(title, loc='left', fontsize=10.5, pad=5)

    # --- (c) near-wall resolution -------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    ax.semilogy(case.xs*MM, case.yplus, color='#7b3fa0', lw=1.6)
    ax.axhline(1.0, color='crimson', ls='--', lw=1.2)
    ax.axhspan(5, 30, color='0.85', zorder=0)
    x_left = case.xs.min()*MM
    ax.text(x_left*0.97, 1.15, '$y^+=1$', color='crimson', fontsize=9)
    ax.text(x_left*0.97, 9, 'buffer layer $5<y^+<30$', fontsize=9, color='0.35')
    ax.set_xlim(x_left, case.xs.max()*MM); ax.set_ylim(1e-2, 60)
    ax.set_xlabel('x [mm]'); ax.set_ylabel('$y^+$ of first cell')
    ax.grid(alpha=.25, which='both')
    ax.set_title('(c)  $y^+$ = %.2f – %.2f (mean %.2f) on the viscous wall;'
                 '  first cell %.2f–%.2f µm'
                 % (case.yplusv.min(), case.yplusv.max(), case.yplusv.mean(),
                    case.dw.min()*1e6, case.dw.max()*1e6),
                 loc='left', fontsize=10.5, pad=5)

    # --- (d) boundary-layer profile in wall units ---------------------------
    ax = fig.add_subplot(gs[2, 1])
    yp, up, uvd = case.profile
    ax.semilogx(yp, uvd, 'o', ms=2.6, color='#1f4e9c', label='CFD (van Driest)')
    yy = np.logspace(0, np.log10(300), 50)
    ax.semilogx(yy, yy, 'k:', lw=1.2, label='$u^+=y^+$')
    yy2 = np.logspace(np.log10(20), np.log10(600), 50)
    ax.semilogx(yy2, np.log(yy2)/KARMAN + BLOG, 'r--', lw=1.2, label='log law')
    ax.set_xlim(0.05, 900); ax.set_ylim(0, 28); ax.grid(alpha=.25, which='both')
    ax.set_xlabel('$y^+$'); ax.set_ylabel('$u^+_{VD}$')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_title('(d)  Profile at x = %.1f mm\n$Re_\\tau$ = %.0f, δ$_0$ = %.2f mm'
                 % (case.x_ref*MM, case.Retau, case.delta0*MM), loc='left', fontsize=10)

    fig.savefig(fname, dpi=165)
    plt.close(fig)


# =============================================================================
#  Part 5b   Hao's flow-field panel, reproduced
# =============================================================================
#
# A like-for-like redraw of figure 8(a) of Hao (JFM 2023): filled bands of
# u/u_inf, the sonic-free "u = 0.99" contour, the bubble outline and the S/R
# circles, on the paper's own axes.  The point is that the two can be laid
# side by side and read as one picture, so every styling choice below is
# copied from the screenshot in context/ rather than chosen.
#
# WHAT L IS.  The paper labels its axes x/L and y/L and never fixes L in the
# caption, but the screenshot pins it: its boundary-layer edge sits at
# y/L = 2.27 on the plate and its S and R circles at x/L = -10.4 and 8.8.
# Those are delta = 2.27 mm and the separation and reattachment this case
# already reports in millimetres, so L = 1 mm and the paper's axes are simply
# x and y in mm.  (The same reading makes the highRe panel's edge y/L = 4.1
# match its delta = 4.1 mm, and it is what lets the digitized wall curves in
# context/ overlay the solution directly -- had L been delta, they would have
# missed by a factor of 2.27 and nothing would have lined up.)
#
# THE COLOURS.  The screenshot's bands were sampled and fitted against
# matplotlib's families: RdYlBu_r truncated to [0.06, 0.84] over 12 bands
# reproduces them to a mean error of 8/255 per channel.  The truncation is
# the part worth keeping -- the paper's top band is a coral, not RdYlBu_r's
# maroon endpoint, and the bubble bottoms out at a mid blue rather than navy,
# so an untruncated map reads far darker at both ends than the original.
PAPER_CMAP = ('RdYlBu_r', 0.06, 0.84)
PAPER_BANDS = 12
PAPER_ULIM = (-0.1, 1.0)       # the colour range the paper annotates

# Pixel frame of panel (a) inside context/flowfield.png: left, right, top,
# bottom.  Used only to crop the screenshot for the side-by-side, so that its
# data area and ours line up edge to edge instead of by eye.
#
# The limits below are measured off the screenshot rather than read from its
# tick labels, because the labels only pin x.  Tracing the white-to-colour
# boundary gives the wall: flat to col 347, then a ramp of -0.461 px/px.  The
# ramp is 25 deg by construction, so px-per-y = 0.461*px-per-x/tan(25 deg) =
# px-per-x -- the panel is equal aspect, which is worth knowing because a
# guess of "y runs 0 to 17" is off by enough to bend every shock angle in the
# comparison.  y = 0 lands on row 258, sixteen rows above the frame, so the
# panel carries a white strip below the wall; reproducing that strip is what
# keeps the two wall lines at the same height when the panels are stacked.
PAPER_CROP = (105, 589, 48, 274)
PAPER_XLIM = (-20.0, 20.0)
PAPER_YLIM = (-1.32, 17.33)


def paper_cmap():
    """The screenshot's colormap: RdYlBu_r with both ends trimmed."""
    name, lo, hi = PAPER_CMAP
    base = plt.get_cmap(name)
    return mcolors.LinearSegmentedColormap.from_list(
        'paper', base(np.linspace(lo, hi, 256)))


# Confidence radius for calling a pixel "flat fill".  The distance-to-nearest
# -band distribution has its bulk under 0.09 (JPEG noise inside a band) and a
# long tail past 0.19 (ink, edges, white); 0.10 sits in the gap.
FILL_TAU = 0.10


def load_shot(ref_png):
    """The screenshot cropped to panel (a), plus the (x/L, y/L) of every pixel."""
    l, r, t, b = PAPER_CROP
    img = plt.imread(ref_png)[t:b, l:r, :3].astype(float)
    if img.max() > 1.001:                      # 8-bit PNG rather than float
        img /= 255.0
    H, W = img.shape[:2]
    X, Y = np.meshgrid(np.linspace(*PAPER_XLIM, W),
                       np.linspace(PAPER_YLIM[1], PAPER_YLIM[0], H))
    return img, X, Y


def restore_shot(img, X, Y):
    """Undo the screenshot's JPEG noise, inventing nothing.

    The paper's panel is a DISCRETE contour plot: away from its lines, every
    pixel was originally one of twelve flat colours.  Compression turned each
    of those flats into a cloud with a standard deviation of about 0.01 per
    channel, plus ringing along every band edge.  That is recoverable damage,
    because the clean value of a flat pixel is not a guess -- it is the colour
    the rest of its own band already has.

    So: classify each pixel to a band, take the MEASURED median colour of each
    band from this image, and repaint the confidently-flat pixels with it.
    The median is the point -- an earlier version repainted with the fitted
    RdYlBu_r palette instead, which is a different picture's colours, off by
    0.044 in green on the free stream.  Nothing here comes from outside the
    screenshot.

    Everything that is not confidently flat is left exactly as it was: the
    dashed contours, the wall, the S and R circles, the text, the colour key,
    the antialiased pixels along every edge, and the white under the wall.
    Those keep their original blur.  This raises no detail and moves no
    boundary; it only flattens what was already meant to be flat.

    Returns (restored image, fraction of pixels repainted).
    """
    bands = paper_cmap()((np.arange(PAPER_BANDS) + 0.5)/PAPER_BANDS)[:, :3]
    d = np.linalg.norm(img[:, :, None, :] - bands[None, None, :, :], axis=3)
    idx, dmin = d.argmin(2), d.min(2)

    # Below the wall is white by construction, never a band; letting it snap
    # to the palest band would repaint the blank half of the panel.
    wall = np.maximum(X, 0.0)*np.tan(np.radians(25.0))
    fill = (dmin < FILL_TAU) & (Y > wall + 0.15)
    fill &= ~((X > -18.5) & (X < -2.5) & (Y > 11.5))          # colour key

    # Drop isolated misclassifications: a pixel whose band is shared by at
    # most one of its eight neighbours, inside a neighbourhood that agrees on
    # some other band.  A pixel ON a band edge has neighbours of both bands,
    # so it fails the test and is left alone -- the filter cannot walk an edge.
    pad = np.pad(np.where(fill, idx, -1), 1, constant_values=-1)
    shifts = [pad[a:a+idx.shape[0], b:b+idx.shape[1]]
              for a in (0, 1, 2) for b in (0, 1, 2) if (a, b) != (1, 1)]
    counts = np.stack([sum((s == k).astype(np.int8) for s in shifts)
                       for k in range(PAPER_BANDS)])
    own = np.take_along_axis(counts, idx[None], 0)[0]
    best = counts.argmax(0)
    lone = fill & (own <= 1) & (counts.max(0) >= 7)
    idx = np.where(lone, best, idx)

    out = img.copy()
    for k in range(PAPER_BANDS):
        sel = fill & (idx == k)
        if sel.sum() >= 40:                       # enough to trust a median
            out[sel] = np.median(img[sel], axis=0)
    return out, float(fill.mean())


def paper_raster(img, X, Y):
    """Invert the screenshot back into a field of u/u_inf.

    Because the bands are discrete and their colours are now known, the
    screenshot can be read rather than merely looked at: classify each pixel
    to the nearest of the twelve band colours and it returns the band, hence
    u to within half a band (+-0.046).  That is enough to subtract one
    solution from the other and see WHERE they differ rather than guessing
    from two pictures side by side.

    Returns (u, valid) on the screenshot's own pixel raster.  `valid` drops
    everything that is not field: the region under the wall, the black of the
    dashed contours and the wall line, the colour key, and any pixel whose
    colour is too far from every band to classify.  Fed the restored image
    rather than the raw one, that last category shrinks to the edges alone.
    """
    edges = np.linspace(PAPER_ULIM[0], PAPER_ULIM[1], PAPER_BANDS + 1)
    centres = 0.5*(edges[:-1] + edges[1:])
    bands = paper_cmap()((np.arange(PAPER_BANDS) + 0.5)/PAPER_BANDS)[:, :3]
    d = np.linalg.norm(img[:, :, None, :] - bands[None, None, :, :], axis=3)

    # Black text and black contour lines are far from every band colour, so
    # the distance test throws them out on its own -- the Re label needs no
    # rectangle, and cutting one would have blanked a stripe of real field
    # across the ramp.  The colour key does need one: it is a smooth ramp of
    # the very colours being matched, so every pixel of it classifies as
    # good field.
    wall = np.maximum(X, 0.0)*np.tan(np.radians(25.0))
    valid = (d.min(2) < 0.10) & (Y > wall + 0.15)
    valid &= ~((X > -18.5) & (X < -2.5) & (Y > 11.5))            # colour key
    return centres[d.argmin(2)], valid


def sample_field(case, F, X, Y):
    """F, given on the curvilinear cell centres, sampled at (X, Y).

    The grid is curvilinear, so the structured cells are handed to
    matplotlib's triangulation as two triangles each -- exact connectivity,
    no Delaunay pass and no scipy.  Masked wherever (X, Y) falls outside the
    mesh, which is everything under the wall and above the far field.
    """
    Xc, Yc = case.Xc*MM, case.Yc*MM
    i = np.where((Xc[0] > PAPER_XLIM[0] - 4) & (Xc[0] < PAPER_XLIM[1] + 4))[0]
    Xc, Yc, F = Xc[:, i], Yc[:, i], F[:, i]
    j = np.where(Yc.min(1) < PAPER_YLIM[1] + 5)[0]
    Xc, Yc, F = Xc[j], Yc[j], F[j]

    nj, ni = Xc.shape
    ids = np.arange(nj*ni).reshape(nj, ni)
    a, b, c, d = ids[:-1, :-1], ids[:-1, 1:], ids[1:, 1:], ids[1:, :-1]
    tris = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                           np.stack([a, c, d], -1).reshape(-1, 3)])
    interp = LinearTriInterpolator(
        Triangulation(Xc.ravel(), Yc.ravel(), tris), F.ravel())
    return interp(X, Y)


def _paper_axes(ax, xlabel=True, ylabel=True):
    """The screenshot's axes, to the pixel."""
    ax.set_xlim(*PAPER_XLIM); ax.set_ylim(*PAPER_YLIM); ax.set_aspect('equal')
    ax.set_xticks([-20, -10, 0, 10, 20]); ax.set_yticks([0, 5, 10, 15])
    ax.tick_params(direction='in', top=True, right=True, labelsize=9)
    if xlabel:
        ax.set_xlabel('$x/L$')
    if ylabel:
        ax.set_ylabel('$y/L$')


def _draw_paper_field(case, ax):
    """Panel (b): this solution, in the paper's own bands."""
    U = np.clip(case.u/case.ue, *PAPER_ULIM)
    levels = np.linspace(PAPER_ULIM[0], PAPER_ULIM[1], PAPER_BANDS + 1)
    ax.contourf(case.Xc*MM, case.Yc*MM, U, levels=levels, cmap=paper_cmap(),
                zorder=1)
    ax.contour(case.Xc*MM, case.Yc*MM, U, levels=[0.0, 0.99], colors='k',
               linestyles='--', linewidths=1.0, zorder=4)
    _draw_paper_wall(case, ax)


def _draw_paper_wall(case, ax):
    ax.plot(case.X[0]*MM, case.Y[0]*MM, 'k-', lw=1.1, zorder=6)
    ax.fill_between(case.X[0]*MM, case.Y[0]*MM, PAPER_YLIM[0] - 5,
                    color='w', zorder=5)
    for xm in (case.x_sep*MM, case.x_rea*MM):
        ax.plot([xm], [np.interp(xm, case.xw*MM, case.yw*MM)], 'o', ms=5,
                mfc='none', mec='0.25', mew=1.2, zorder=7)


def _draw_paper_key(ax):
    """The inset colour key, where the screenshot puts it."""
    cax = ax.inset_axes([0.161, 0.750, 0.093, 0.168])
    cax.imshow(np.linspace(1, 0, 256).reshape(-1, 1), cmap=paper_cmap(),
               aspect='auto')
    cax.set_xticks([]); cax.set_yticks([])
    for sp in cax.spines.values():
        sp.set_linewidth(0.8)
    ax.text(0.150, 0.834, '$u/u_\\infty$', transform=ax.transAxes,
            fontsize=10, ha='right', va='center')
    for frac, txt in ((0.905, '1'), (0.762, '$-0.1$')):
        ax.text(0.268, frac, txt, transform=ax.transAxes, fontsize=9,
                ha='left', va='center')


def figure_paper(case, fname, ref_png=None):
    """Hao figure 8(a) and this solution, side by side and differenced.

    Without the screenshot, one panel: this solution drawn on the paper's
    axes.  With it, four:

      (a) the screenshot, cropped to its own frame
      (b) this solution in the same bands
      (c) this solution's u = 0 and u = 0.99 contours laid over (a), which is
          the registration check -- if the bubble and the shock sit on the
          paper's own dashed lines, the two runs agree where it matters
      (d) the two fields subtracted, by reading the screenshot's bands back
          into numbers.  Blue means this solution is slower than Hao's there,
          red faster; +-1 band is the quantisation floor of (a), so anything
          inside the palest tint is agreement to the limit of what a
          screenshot can be asked.
    """
    have_ref = bool(ref_png) and os.path.isfile(ref_png)
    if not have_ref:
        fig = plt.figure(figsize=(7.6, 3.9))
        ax = fig.add_axes([0.105, 0.145, 0.87, 0.80])
        _draw_paper_field(case, ax)
        _draw_paper_key(ax)
        _paper_axes(ax)
        ax.text(0.94, 0.17, '$Re_\\delta$ = %s'
                % format(int(round(case.Re_delta)), ',').replace(',', '\\,'),
                transform=ax.transAxes, fontsize=10, ha='right', va='center')
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        return

    raw, Xp, Yp = load_shot(ref_png)
    shot, repainted = restore_shot(raw, Xp, Yp)
    extent = [PAPER_XLIM[0], PAPER_XLIM[1], PAPER_YLIM[0], PAPER_YLIM[1]]

    # The restored crop is worth having on its own, not only as a tile here.
    stem = os.path.splitext(fname)[0]
    plt.imsave(stem.replace('_paper', '') + '_hao_restored.png', shot)

    fig = plt.figure(figsize=(13.2, 7.4))
    gs = GridSpec(2, 2, hspace=0.16, wspace=0.10,
                  left=0.055, right=0.985, top=0.945, bottom=0.075)
    axes = [fig.add_subplot(gs[k//2, k % 2]) for k in range(4)]

    # (a) the screenshot ----------------------------------------------------
    axes[0].imshow(shot, aspect='auto', extent=extent, origin='upper')
    _paper_axes(axes[0], xlabel=False)
    axes[0].set_title('(a)  Hao, JFM 2023, figure 8(a)   $Re_\\delta$ = %d'
                      '   (%.0f%% of pixels de-compressed)'
                      % (RE_DELTA_EXP, 100*repainted), loc='left', fontsize=10)

    # (b) this solution -----------------------------------------------------
    _draw_paper_field(case, axes[1])
    _draw_paper_key(axes[1])
    _paper_axes(axes[1], xlabel=False, ylabel=False)
    axes[1].set_title('(b)  this solution   $Re_\\delta$ = %d   %s'
                      % (int(round(case.Re_delta)), case.label),
                      loc='left', fontsize=10)

    # (c) registration ------------------------------------------------------
    # Lime over a red-and-blue screenshot, haloed in black: the one hue that
    # stays visible over the free stream, the bubble and the white below the
    # wall alike.  Hao's own contours are the black dashes underneath, so
    # where the two agree the lime sits directly on top of them.
    axes[2].imshow(shot, aspect='auto', extent=extent, origin='upper')
    U = np.clip(case.u/case.ue, *PAPER_ULIM)
    cs = axes[2].contour(case.Xc*MM, case.Yc*MM, U, levels=[0.0, 0.99],
                         colors='#39ff14', linewidths=1.6, zorder=4)
    for coll in cs.collections:
        coll.set_path_effects([pe.withStroke(linewidth=3.0, foreground='k')])
    _paper_axes(axes[2])
    axes[2].set_title('(c)  this solution\'s $u$ = 0 and $u$ = 0.99 '
                      '(green) over (a)', loc='left', fontsize=10)

    # (d) difference --------------------------------------------------------
    # Quantise THIS solution into the same twelve bands before subtracting.
    # The screenshot can only report a band centre, so its free stream reads
    # 0.954 where ours reads 1.0; differencing the raw fields printed that
    # 0.046 offset as a uniform half-band tint over the whole free stream --
    # an artefact of the palette, not a disagreement.  Binned the same way,
    # the difference is in whole bands and answers the question actually
    # being asked: how far apart are the two pictures, in the units the
    # picture is drawn in.
    up, valid = paper_raster(shot, Xp, Yp)
    X, Y = Xp, Yp
    edges = np.linspace(PAPER_ULIM[0], PAPER_ULIM[1], PAPER_BANDS + 1)
    centres = 0.5*(edges[:-1] + edges[1:])
    um = sample_field(case, U, X, Y)
    ok = valid & ~np.ma.getmaskarray(um)
    umq = centres[np.clip(np.digitize(np.ma.getdata(um), edges) - 1,
                          0, PAPER_BANDS - 1)]
    dU = np.ma.masked_where(~ok, umq - up)
    band = (PAPER_ULIM[1] - PAPER_ULIM[0])/PAPER_BANDS
    # Grey, not white, for what could not be read: white would pass for
    # "no difference here" next to the pale centre of the diverging map.
    dcmap = plt.get_cmap('RdBu_r').copy()
    dcmap.set_bad('0.86')
    im = axes[3].imshow(dU/band, aspect='auto', extent=extent, origin='upper',
                        cmap=dcmap, vmin=-2.5, vmax=2.5,
                        interpolation='nearest')
    _draw_paper_wall(case, axes[3])
    _paper_axes(axes[3], ylabel=False)
    cb = fig.colorbar(im, ax=axes[3], fraction=0.031, pad=0.015,
                      ticks=[-2, -1, 0, 1, 2])
    cb.set_label('bands apart  (this $-$ Hao)', fontsize=9)
    cb.ax.tick_params(labelsize=8)
    nb = np.abs(dU[ok])/band
    axes[3].set_title('(d)  difference:  %.0f%% same band, %.0f%% within one, '
                      'rms %.2f'
                      % (100*float((nb < 0.5).mean()),
                         100*float((nb < 1.5).mean()),
                         float(np.sqrt((nb**2).mean()))),
                      loc='left', fontsize=10)

    fig.savefig(fname, dpi=170)
    plt.close(fig)

# =============================================================================
#  Part 6   Command line
# =============================================================================

REPORT = [
    ('grid (i x j)',        lambda c: '%d x %d' % (c.M.shape[1]+1, c.M.shape[0]+1)),
    ('M_inf',               lambda c: '%.3f' % c.Minf),
    ('ramp angle [deg]',    lambda c: '%.2f' % c.ramp),
    ('Re [1/m]',            lambda c: '%.3e' % c.Re_m),
    ('plate length [mm]',   lambda c: '%.1f' % (-c.xw.min()*MM)),
    ('transition x [mm]',   lambda c: '%.1f' % (c.x_trans*MM)
                                if np.isfinite(c.x_trans)
                                else 'none (fully turbulent)'),
    ('delta_0 [mm]',        lambda c: '%.3f' % (c.delta0*MM)),
    ('Re_tau',              lambda c: '%.0f' % c.Retau),
    ('delta @ x_exp [mm]',  lambda c: '%.3f  at x = %.2f mm  (exp %.2f, %+.1f%%)'
                                % (c.delta_exp*MM, c.x_exp*MM, DELTA_EXP*MM,
                                   100*(c.delta_exp/DELTA_EXP - 1))),
    ('Re_delta',            lambda c: '%.0f  (exp %d, %+.1f%%)'
                                % (c.Re_delta, RE_DELTA_EXP,
                                   100*(c.Re_delta/RE_DELTA_EXP - 1))),
    ('Re_tau @ x_exp',      lambda c: '%.0f' % c.Retau_exp),
    ('x_sep [mm]',          lambda c: '%.2f' % (c.x_sep*MM)),
    ('x_reatt [mm]',        lambda c: '%.2f' % (c.x_rea*MM)),
    ('L_sep [mm]',          lambda c: '%.2f' % (c.Lsep*MM)),
    ('L_sep / delta_0',     lambda c: '%.2f' % (c.Lsep/c.delta0)),
    ('p/p_inf peak',        lambda c: '%.4f  (Cp = %.4f)' % (np.nanmax(c.pw),
                                np.nanmax(c.cpw))),
    ('p/p_inf inviscid',    lambda c: '%.4f  (Cp = %.4f)' % (c.p_inviscid,
                                c.cp_inviscid)),
    ('beta fitted [deg]',   lambda c: '%.2f' % c.beta_fit),
    ('beta theory [deg]',   lambda c: '%.2f' % np.degrees(
                                oblique_shock_beta(c.Minf, np.radians(c.ramp)))),
    ('1st cell [um]',       lambda c: '%.2f - %.2f' % (c.dw.min()*1e6, c.dw.max()*1e6)),
    ('y+ min/mean/max',     lambda c: '%.3f / %.3f / %.3f  (viscous wall only)'
                                % (c.yplusv.min(), c.yplusv.mean(), c.yplusv.max())),
    ('T_w/T_inf',           lambda c: '%.4f' % np.nanmean(c.Tww[c.viscw])),
    ('  adiabatic theory',  lambda c: '%.4f' % (1 + 0.9*(GAM-1)/2*c.Minf**2)),
    ('reverse-flow cells',  lambda c: '%d' % int((c.u < 0).sum())),
]


def report(case):
    """Fixed-width table of derived quantities.

    Every cell is padded to its column width so the pipes line up in any
    monospaced terminal.
    """
    rows = [('quantity', 'value')] + [(n, f(case)) for n, f in REPORT]
    w = [max(len(r[i]) for r in rows) for i in (0, 1)]
    bar = '+' + '+'.join('-'*(w[i] + 2) for i in (0, 1)) + '+'
    line = lambda r: '| ' + ' | '.join(r[i].ljust(w[i]) for i in (0, 1)) + ' |'
    return '\n'.join([bar, line(rows[0]), bar] + [line(r) for r in rows[1:]] + [bar])


def main():
    global DELTA_EXP, X_EXP
    ap = argparse.ArgumentParser(
        description='Visualize one ADflow surface-solution CGNS file.',
        epilog='Figures are named after the input file, e.g. '
               'foo.cgns -> foo_flowfield.png and foo_wall.png',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gridFile', help='an ADflow *_surf.cgns file (HDF5 or ADF)')
    ap.add_argument('-o', '--outdir', default='.',
                    help='directory for the PNGs (default: current directory)')
    ap.add_argument('--tree', action='store_true',
                    help='print the CGNS tree and exit, making no figures')
    ap.add_argument('--no-figs', action='store_true',
                    help='print the table only, make no figures')
    ap.add_argument('--context', default=None, metavar='DIR',
                    help='directory of digitized reference curves (p*.csv, '
                         'cf*.csv) to overlay on the wall figure '
                         '(default: a context/ directory next to this script)')
    ap.add_argument('--no-context', action='store_true',
                    help='skip the reference overlay even if context/ exists')
    ap.add_argument('--exag', type=float, default=3.5,
                    help='vertical exaggeration hint for the close-up panel')
    ap.add_argument('--delta-exp', type=float, default=DELTA_EXP*MM, metavar='MM',
                    help='experimental boundary-layer thickness in mm '
                         '(default: %.2f, Zheltovodov low-Re case)' % (DELTA_EXP*MM))
    ap.add_argument('--x-exp', type=float, default=None, metavar='MM',
                    help='station where it is measured, mm upstream of the corner '
                         '(default: 15.4 * delta-exp)')
    a = ap.parse_args()

    # The station defaults to 15.4 delta, so it follows --delta-exp unless pinned.
    DELTA_EXP = a.delta_exp/MM
    X_EXP = -abs(a.x_exp)/MM if a.x_exp is not None else -15.4*DELTA_EXP

    if a.tree:
        h, root = load(a.gridFile)
        kind = ('HDF5, superblock v%d' % h.sbver) if hasattr(h, 'sbver') \
               else ('ADF, %s' % h.version)
        print('== %s  [%s] ==' % (a.gridFile, kind))
        tree(root)
        return

    case = Case(a.gridFile)
    stem = os.path.splitext(os.path.basename(a.gridFile))[0]

    # The overlay defaults to context/ beside the script, so the usual
    # invocation picks it up with no extra flag.
    ctxdir = a.context or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'context')
    ref = {} if a.no_context else load_context(ctxdir)
    if ref:
        print('reference curves from %s: %s' % (ctxdir, ', '.join(sorted(ref))))

    if not a.no_figs:
        os.makedirs(a.outdir, exist_ok=True)
        # The paper panel takes the screenshot rather than the CSVs, so it
        # looks for it in the same context/ directory the curves came from.
        ref_png = os.path.join(ctxdir, 'flowfield.png') if not a.no_context else None
        draws = (('flowfield', figure_flowfield, a.exag),
                 ('wall', figure_wall, ref),
                 ('paper', figure_paper, ref_png))
        for suffix, draw, extra in draws:
            path = os.path.join(a.outdir, '%s_%s.png' % (stem, suffix))
            draw(case, path, extra)
            print('wrote', path)
        print()

    print(report(case))


if __name__ == '__main__':
    main()
