#!/usr/bin/env python3
"""exam2d - scientifically accurate 2D black-and-white diagrams in exam-paper style.

Everything is computed from real equations / reference data (never hand-placed):
  physics  : point-charge E-fields + equipotentials, parallel plates (Laplace solve,
             real fringing), B-fields of wires / loops / solenoids / bar magnets
             (exact flux function, elliptic integrals), ray diagrams (lens/mirror
             equation), waves (transverse, longitudinal, standing, two-source),
             Kepler orbits, projectiles with drag, graphs, contours, vector fields
  chemistry: skeletal / displayed formulae (RDKit 2D depiction), VSEPR shapes with
             wedge/dash bonds, dot-and-cross diagrams, crystal layers and plan views
             (ASE lattices, experimental a), hydrogen-like orbital cross-sections and
             radial distribution graphs (exact wavefunctions)

CLI : python exam2d.py <command> [args] -o out.png        (see --help / SKILL.md)
API : from exam2d import *; f = Fig(); efield(f, ...); f.save("x.png")
"""
import os, sys, re, argparse, warnings
from fractions import Fraction

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle, Arc, Ellipse

INK, PAPER, LW = "black", "white", 1.2  # ink / fill colours; Fig(ink=..., paper=...) switches them
FONTS = {"serif": dict(family="STIXGeneral", math="stix"),
         "sans": dict(family="DejaVu Sans", math="dejavusans")}


def set_font(kind="serif"):
    f = FONTS.get(kind, FONTS["serif"])
    plt.rcParams.update({"font.family": f["family"], "mathtext.fontset": f["math"], "font.size": 11,
                         "axes.linewidth": 1.0, "xtick.direction": "out", "ytick.direction": "out",
                         "xtick.major.size": 4, "ytick.major.size": 4, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "hatch.linewidth": 0.6, "hatch.color": INK,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})


set_font()
FRAC = {Fraction(0): "0", Fraction(1, 2): "½", Fraction(1, 4): "¼", Fraction(3, 4): "¾",
        Fraction(1, 3): "⅓", Fraction(2, 3): "⅔", Fraction(1, 8): "⅛", Fraction(3, 8): "⅜",
        Fraction(5, 8): "⅝", Fraction(7, 8): "⅞", Fraction(1, 6): "⅙", Fraction(5, 6): "⅚", Fraction(1): "1"}


def qfmt(q):
    """+1 -> '+q', -2 -> '−2q' (true minus sign)."""
    a = abs(q); s = "+" if q > 0 else "−"
    return f"{s}{'' if a == 1 else f'{a:g}'}q"


def frac_txt(x):
    fr = Fraction(x).limit_denominator(8)
    return FRAC.get(fr, f"{fr.numerator}/{fr.denominator}" if abs(float(fr) - x) < 1e-3 else f"{x:.2f}")


# =============================================================== Figure
class Fig:
    """Black-on-white matplotlib figure. Diagram mode (equal aspect, no axes) by default;
    call .plot_axes(...) for an exam-style graph with arrowed axes. All methods chain."""

    def __init__(self, title=None, caption=None, figno=None, size=(6.0, 4.5), font="serif", ink="black",
                 paper="white", px=None, transparent=False):
        """px=(W, H): exact pixel size (no tight crop; limits grow to keep 1:1 scale).
        ink/paper: e.g. ink="white", paper="#141d35" for dark slides (paper fills shapes and label boxes)."""
        global INK, PAPER
        INK, PAPER = ink, paper
        set_font(font)
        plt.rcParams.update({"text.color": ink, "axes.edgecolor": ink, "xtick.color": ink, "ytick.color": ink,
                             "axes.labelcolor": ink, "hatch.color": ink})
        self.px, self.transparent, self.dpi = px, transparent, 200
        self.fig, self.ax = plt.subplots(figsize=(px[0] / 200, px[1] / 200) if px else size)
        if px:
            self.ax.set_position([0.015, 0.015, 0.97, 0.97])
        self.ax.set_aspect("equal", adjustable="datalim" if px else "box"); self.ax.axis("off")
        self.title, self.caption, self.figno = title, caption, figno
        self.notes, self.lims = [], None

    # ---------- primitives (data coordinates)
    def span(self):
        x0, x1 = self.ax.get_xlim(); y0, y1 = self.ax.get_ylim()
        return max(abs(x1 - x0), abs(y1 - y0))

    def limits(self, xlim, ylim):
        self.lims = (tuple(xlim), tuple(ylim))
        self.ax.set_xlim(*xlim); self.ax.set_ylim(*ylim)
        return self

    def line(self, pts, dashed=False, lw=LW, ls=None, z=2, **kw):
        p = np.asarray(pts, float)
        self.ax.plot(p[:, 0], p[:, 1], color=INK, lw=lw, ls=ls or ("--" if dashed else "-"),
                     zorder=z, solid_capstyle="round", **kw)
        return self

    def head(self, p, d, size=9):
        """Solid arrowhead at p pointing along d (for marking direction on a line)."""
        d = np.asarray(d, float); n = np.hypot(*d)
        if n == 0:
            return self
        e = d / n * 1e-4 * self.span(); p = np.asarray(p, float)
        self.ax.annotate("", xy=p + e, xytext=p - e, zorder=4,
                         arrowprops=dict(arrowstyle="-|>,head_length=0.55,head_width=0.24", color=INK,
                                         lw=0.6, mutation_scale=size, shrinkA=0, shrinkB=0))
        return self

    def heads_on(self, pts, at=(0.5,), reverse=False, size=9):
        """Arrowheads at fractions of arc length along a polyline."""
        p = np.asarray(pts, float)
        if len(p) < 3:
            return self
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        for t in at:
            i = int(np.clip(np.searchsorted(s, t * s[-1]), 1, len(p) - 2))
            d = p[i + 1] - p[i - 1]
            self.head(p[i], -d if reverse else d, size)
        return self

    def arrow(self, start, vec, label=None, lw=LW, size=11, both=False, label_off=(0, 0), **tkw):
        a = np.asarray(start, float); b = a + np.asarray(vec, float)
        st = "<|-|>" if both else "-|>"
        self.ax.annotate("", xy=b, xytext=a, zorder=4, arrowprops=dict(
            arrowstyle=f"{st},head_length=0.5,head_width=0.2", color=INK, lw=lw, mutation_scale=size,
            shrinkA=0, shrinkB=0))
        if label:
            off = np.asarray(label_off, float)
            if not off.any():  # default: just beyond the tip, along the arrow
                v = b - a; off = v / (np.hypot(*v) or 1) * 0.035 * self.span()
            self.text(b + off, label, **tkw)
        return self

    def text(self, xy, s, size=11, ha="center", va="center", bg=False, **kw):
        bb = dict(boxstyle="square,pad=0.08", fc=PAPER, ec="none") if bg else None
        self.ax.text(*xy, s, fontsize=size, ha=ha, va=va, color=INK, zorder=6, bbox=bb, **kw)
        return self

    def leader(self, point, text_xy, s, size=10, **kw):
        """Exam-style label: text joined to the feature by a thin straight line."""
        self.ax.annotate(s, xy=point, xytext=text_xy, fontsize=size, ha=kw.pop("ha", "center"),
                         va=kw.pop("va", "center"), zorder=6,
                         arrowprops=dict(arrowstyle="-", color=INK, lw=0.7, shrinkA=2, shrinkB=0), **kw)
        return self

    def dim(self, p1, p2, s, off=(0, 0), size=10, text_off=None):
        """Dimension line with arrowheads at both ends and extension lines."""
        p1, p2, o = np.asarray(p1, float), np.asarray(p2, float), np.asarray(off, float)
        if np.hypot(*o) > 0:
            for p in (p1, p2):
                self.line([p, p + o * 1.15], lw=0.6)
        self.arrow(p1 + o, p2 - p1, lw=0.8, size=9, both=True)
        d = p2 - p1; nrm = np.array([-d[1], d[0]]) / (np.hypot(*d) or 1)
        if text_off is None:
            text_off = nrm * 0.035 * self.span() * (1 if np.dot(nrm, o) >= 0 else -1)
        self.text((p1 + p2) / 2 + o + np.asarray(text_off), s, size=size, bg=True)
        return self

    def circle(self, c, r, fill=None, lw=LW, hatch=None, z=5, ls="-"):
        self.ax.add_patch(Circle(c, r, fc=PAPER if fill is None else fill, ec=INK, lw=lw, hatch=hatch, zorder=z, ls=ls))
        return self

    def dot(self, c, r):
        return self.circle(c, r, fill=INK, lw=0)

    def current(self, c, r, out=True):
        """Conductor cross-section: ⊙ current out of page, ⊗ into page."""
        self.circle(c, r, lw=1.0, z=6)
        if out:
            self.ax.add_patch(Circle(c, r * 0.22, fc=INK, ec=INK, zorder=7))
        else:
            k = r * 0.62
            for s in (1, -1):
                self.ax.plot([c[0] - k, c[0] + k], [c[1] - s * k, c[1] + s * k], color=INK, lw=1.0, zorder=7)
        return self

    def rot_arrow(self, c, r, a0, a1, label=None, lw=LW, size=12, label_r=None, rx=None, ry=None, **tkw):
        """Curved arrow (angular velocity, rotation sense) on a circle/ellipse about c from angle a0 to a1
        (degrees; a1 > a0 = anticlockwise, a1 < a0 = clockwise). Head at a1."""
        rx, ry = rx or r, ry or r
        t = np.radians(np.linspace(a0, a1, 80)); c = np.asarray(c, float)
        p = c + np.c_[rx * np.cos(t), ry * np.sin(t)]
        self.ax.plot(p[:-3, 0], p[:-3, 1], color=INK, lw=lw, zorder=4, solid_capstyle="round")
        self.head(p[-1], p[-1] - p[-4], size)
        if label:
            am = np.radians((a0 + a1) / 2); lr = label_r or 1.25
            self.text(c + np.array([rx * lr * np.cos(am), ry * lr * np.sin(am)]), label, **tkw)
        return self

    def angle(self, v, a0, a1, r, label=None, lr=1.45, lw=0.9, **tkw):
        """Angle arc at vertex v between directions a0 and a1 (degrees), with label."""
        self.ax.add_patch(Arc(v, 2 * r, 2 * r, theta1=min(a0, a1), theta2=max(a0, a1), color=INK, lw=lw, zorder=4))
        if label:
            am = np.radians((a0 + a1) / 2)
            self.text(np.asarray(v, float) + lr * r * np.array([np.cos(am), np.sin(am)]), label, **tkw)
        return self

    def right_angle(self, p, d1, d2, s):
        """Right-angle box at p between unit directions d1 and d2, side s."""
        p, d1, d2 = (np.asarray(x, float) for x in (p, d1, d2))
        d1, d2 = d1 / np.hypot(*d1), d2 / np.hypot(*d2)
        return self.line([p + d1 * s, p + (d1 + d2) * s, p + d2 * s], lw=0.8)

    def surface(self, p0, p1, side=1, n=None, lw=1.6, tick=None):
        """Fixed surface (wall, ground, ceiling): thick line p0-p1 with hatching on one side
        (side=+1: left of the direction p0->p1, -1: right)."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0; L = np.hypot(*d); u = d / L; nrm = np.array([-u[1], u[0]]) * side
        tick = tick or 0.04 * self.span(); n = n or max(3, int(L / (0.55 * tick)))
        self.line([p0, p1], lw=lw)
        for t in np.linspace(0, 1, n + 1)[:-1]:
            q = p0 + d * t + u * 0.5 * tick / 1.5
            self.line([q, q + nrm * tick - u * tick * 0.7], lw=0.7)
        return self

    def poly(self, pts, fill=None, lw=LW, dashed=False, hatch=None, z=3):
        self.ax.add_patch(Polygon(np.asarray(pts, float), closed=True, fc=PAPER if fill is None else fill, ec=INK,
                                  lw=lw, ls="--" if dashed else "-", hatch=hatch, zorder=z))
        return self

    def note(self, k, v):
        self.notes.append(f"{k}: {v}")
        return self

    # ---------- graph mode
    def plot_axes(self, xlim, ylim, xlabel="", ylabel="", grid=False, equal=False, ticks=True,
                  origin=True, at_zero=True):
        ax = self.ax; ax.axis("on"); ax.set_aspect("equal" if equal else "auto")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        x0 = 0 if at_zero and xlim[0] <= 0 <= xlim[1] else xlim[0]
        y0 = 0 if at_zero and ylim[0] <= 0 <= ylim[1] else ylim[0]
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_position(("data", x0)); ax.spines["bottom"].set_position(("data", y0))
        ax.plot(1, y0, ">", color=INK, ms=5, transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot(x0, 1, "^", color=INK, ms=5, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.annotate(xlabel, xy=(1, y0), xycoords=ax.get_yaxis_transform(), xytext=(4, -4),
                    textcoords="offset points", ha="left", va="top", fontsize=11)
        ax.annotate(ylabel, xy=(x0, 1), xycoords=ax.get_xaxis_transform(), xytext=(0, 8),
                    textcoords="offset points", ha="center", va="bottom", fontsize=11)
        if not ticks:
            ax.set_xticks([]); ax.set_yticks([])
        if grid:
            ax.minorticks_on()
            ax.grid(which="major", color="0.55", lw=0.5); ax.grid(which="minor", color="0.82", lw=0.3)
            ax.set_axisbelow(True)
        if origin and x0 == 0 and y0 == 0:
            ax.annotate("O", xy=(0, 0), xytext=(-6, -6), textcoords="offset points", ha="right",
                        va="top", fontsize=11)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "" if v == 0 else f"{v:g}".replace("-", "−")))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda v, _: "" if v == 0 else f"{v:g}".replace("-", "−")))
        self.lims = None
        return self

    # ---------- output
    def save(self, path="out.png", svg=False, pdf=False, dpi=300):
        if self.lims:
            self.ax.set_xlim(*self.lims[0]); self.ax.set_ylim(*self.lims[1])
        if self.title:
            self.ax.set_title(self.title, fontsize=13, pad=12)
        foot = " ".join(x for x in (self.figno, self.caption) if x)
        if foot:
            self.ax.text(0.5, -0.03 if not self.ax.axison else -0.12, foot, transform=self.ax.transAxes,
                         ha="center", va="top", fontsize=11)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        out = []
        kw = dict(dpi=self.dpi) if self.px else dict(dpi=dpi, bbox_inches="tight", pad_inches=0.12)
        kw["transparent"] = self.transparent
        if not self.transparent:
            kw["facecolor"] = PAPER
        for ext, on in ((os.path.splitext(path)[1] or ".png", True), (".svg", svg), (".pdf", pdf)):
            if on:
                p = os.path.splitext(path)[0] + ext
                self.fig.savefig(p, **kw)
                if p not in out:
                    out.append(p)
        plt.close(self.fig)
        print("saved: " + ", ".join(out))
        for n in self.notes:
            print(n)
        return out


# =============================================================== field-line helpers
def _trace(field, seeds, ds, nmax, stop, sign=1.0):
    """Vectorised RK4 streamline tracer along +/- normalised field. Returns list of point arrays."""
    X = np.array(seeds, float); n = len(X)
    alive = np.ones(n, bool); P = [X.copy()]

    def f(Y):
        v = field(Y); m = np.linalg.norm(v, axis=1, keepdims=True)
        return sign * v / np.where(m == 0, 1, m)

    end = np.full(n, nmax)
    for k in range(nmax):
        idx = np.where(alive)[0]
        if not len(idx):
            break
        Y = X[idx]
        k1 = f(Y); k2 = f(Y + .5 * ds * k1); k3 = f(Y + .5 * ds * k2); k4 = f(Y + ds * k3)
        X[idx] = Y + ds * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        P.append(X.copy())
        s = stop(X[idx], k)
        alive[idx[s]] = False; end[idx[s]] = k + 1
    P = np.array(P)
    return [P[:end[i] + 1, i] for i in range(n)]


def _visible(p, lims):
    (x0, x1), (y0, y1) = lims
    return (p[:, 0] >= x0) & (p[:, 0] <= x1) & (p[:, 1] >= y0) & (p[:, 1] <= y1)


def _mid_visible(f, p, lims, size=9):
    """Arrowhead at the arc-length middle of the first visible stretch of a line."""
    v = _visible(p, lims)
    if v.sum() < 4:
        return
    i0 = np.argmax(v); i1 = i0 + (np.argmin(v[i0:]) if not v[i0:].all() else len(v) - i0)
    f.heads_on(p[i0:i1], size=size)


def _runs(mask):
    """(start, stop) index pairs of consecutive True stretches."""
    m = np.r_[False, np.asarray(mask, bool), False].astype(int)
    d = np.diff(m)
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _heads_outside(f, p, keep, lims, Bf, min_len, size=9):
    """Arrowhead (oriented by B) at the arc-length middle of every visible stretch where keep is True."""
    ok = keep & _visible(p, lims)
    for a, b in _runs(ok):
        q = p[a:b]
        if len(q) < 4:
            continue
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(q, axis=0).T))]
        if s[-1] < min_len:
            continue
        i = int(np.clip(np.searchsorted(s, s[-1] / 2), 1, len(q) - 2)); d = q[i + 1] - q[i - 1]
        f.head(q[i], d if np.dot(d, Bf(q[i:i + 1])[0]) >= 0 else -d, size)


def _contour_lines(X, Y, Z, level):
    import contourpy
    return [l for l in contourpy.contour_generator(X, Y, Z).lines(level) if len(l) > 3]


def _box(lims):
    (x0, x1), (y0, y1) = lims
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])


# =============================================================== electrostatics
def efield(f, charges, lines_per_q=12, equipotential=True, extent=None):
    """charges: [(q, (x, y)), ...]. 3D Coulomb field in the plane of the charges.
    Lines start on + and end on − charges; number of lines from each charge ∝ |q|."""
    Q = np.array([c[0] for c in charges], float)
    R = np.array([c[1][:2] for c in charges], float)
    d = max(1.0, float(np.max(np.ptp(R, axis=0)))) if len(R) > 1 else 1.0
    c0 = R.mean(0)
    W = 0.5 * np.ptp(R[:, 0]) + (extent or 1.15 * d); H = max(0.5 * np.ptp(R[:, 1]) + 0.9 * d, 0.72 * W)
    W = max(W, H / 0.75)
    lims = ((c0[0] - W, c0[0] + W), (c0[1] - H, c0[1] + H))
    rc = 0.085 * d

    def E(X):
        D = X[:, None, :] - R[None]; r = np.maximum(np.linalg.norm(D, axis=2), 1e-6)
        return np.sum(Q[None, :, None] * D / r[..., None] ** 3, axis=1)

    E0 = np.abs(Q).max() / d ** 2

    def stop(X, k):
        r = np.linalg.norm(X[:, None, :] - R[None], axis=2)
        null = np.linalg.norm(E(X), axis=1) < 2e-3 * E0          # neutral point: E = 0
        return ((r.min(1) < 0.6 * rc) & (k > 3)) | (np.abs(X - c0).max(1) > 3 * W) | null

    ds = d / 250; nmax = int(12 * W / ds)
    lines = []
    for sign, mask in ((1, Q > 0), (-1, Q < 0)):
        if sign == -1 and Q.sum() >= 0 and (Q > 0).any():
            continue
        for q, r in zip(Q[mask], R[mask]):
            n = max(4, int(round(lines_per_q * abs(q))))
            a = np.linspace(0, 2 * np.pi, n, endpoint=False)
            got = _trace(E, r + rc * np.c_[np.cos(a), np.sin(a)], ds, nmax, stop, sign)
            if sign == -1:  # keep only lines escaping to infinity (others are drawn from +)
                got = [g for g in got if np.abs(g[-1] - c0).max() > 2 * W]
            lines += [g[::-1] if sign == -1 else g for g in got]
    for p in lines:
        f.line(p, lw=0.9)
        _mid_visible(f, p, lims)
    if equipotential:
        x = np.linspace(*lims[0], 500); y = np.linspace(*lims[1], 380); X, Y = np.meshgrid(x, y)
        V = sum(q / np.maximum(np.hypot(X - r[0], Y - r[1]), 1e-6) for q, r in zip(Q, R))
        Vq = np.abs(Q).max() / (np.array([0.28, 0.45, 0.75]) * d)
        lev = sorted(set(np.r_[Vq, -Vq]) | ({0.0} if (Q > 0).any() and (Q < 0).any() else set()))
        lev = [v for v in lev if V.min() < v < V.max()]
        if lev:
            f.ax.contour(X, Y, V, levels=lev, colors=INK, linewidths=0.6, linestyles="--", zorder=1)
    for q, r in zip(Q, R):
        rr = rc * (0.85 + 0.15 * min(abs(q), 3))
        f.circle(r, rr, lw=1.1)
        f.text(r, "+" if q > 0 else "−", size=13 if abs(q) < 2 else 14)
        if abs(q) != 1 or len(set(np.abs(Q))) > 1:
            f.text(r + [0, -rr - 0.07 * d], qfmt(q), size=10, va="top", bg=True)
    f.limits(*lims)
    f.note("E-field", f"{len(Q)} charges, net {Q.sum():+g}q, {len(lines)} field lines "
                      f"(number ∝ |q|); dashed = equipotentials" if equipotential else "")
    return f


def plates(f, V=100.0, d=0.02, L=0.06, lines=24, equipotential=True):
    """Parallel plates: numerical Laplace solution (finite differences) incl. edge fringing.
    Field lines start on the + plate with density ∝ local surface charge (Gauss)."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from scipy.interpolate import RegularGridInterpolator as RGI
    Wd, Hd = L / 2 + 1.0 * d, d / 2 + 0.9 * d          # displayed half-sizes
    Wc, Hc = 2.2 * Wd, 2.4 * Hd                         # computational box (V = 0 far away)
    h = d / 36
    x = np.arange(-Wc, Wc + h / 2, h); y = np.arange(-Hc, Hc + h / 2, h)
    X, Y = np.meshgrid(x, y); ny, nx = X.shape
    fixed = np.zeros_like(X, bool); val = np.zeros_like(X)
    fixed[[0, -1], :] = True; fixed[:, [0, -1]] = True
    jt, jb = np.argmin(abs(y - d / 2)), np.argmin(abs(y + d / 2))
    inx = np.abs(x) <= L / 2
    fixed[jt, inx] = True; val[jt, inx] = V / 2
    fixed[jb, inx] = True; val[jb, inx] = -V / 2
    unk = np.flatnonzero(~fixed.ravel()); idx = -np.ones(nx * ny, int); idx[unk] = np.arange(len(unk))
    rows, cols, data = [np.arange(len(unk))], [np.arange(len(unk))], [np.full(len(unk), -4.0)]
    b = np.zeros(len(unk)); fv = val.ravel(); ff = fixed.ravel()
    for off in (1, -1, nx, -nx):
        nb = unk + off; isu = ~ff[nb]
        rows.append(np.arange(len(unk))[isu]); cols.append(idx[nb[isu]]); data.append(np.ones(isu.sum()))
        b[~isu] -= fv[nb[~isu]]
    A = sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(len(unk),) * 2)
    Vg = val.ravel().copy(); Vg[unk] = spsolve(A, b); Vg = Vg.reshape(ny, nx)
    gy, gx = np.gradient(Vg, h)
    iEx, iEy = RGI((y, x), -gx, bounds_error=False, fill_value=0), RGI((y, x), -gy, bounds_error=False, fill_value=0)
    Ef = lambda P: np.c_[iEx(P[:, ::-1]), iEy(P[:, ::-1])]
    yt = y[jt]
    xs = np.linspace(-L / 2, L / 2, 600)
    s_dn = np.maximum(-iEy(np.c_[np.full_like(xs, yt - 1.5 * h), xs]), 0)
    s_up = np.maximum(iEy(np.c_[np.full_like(xs, yt + 1.5 * h), xs]), 0)
    cum = np.cumsum(np.r_[s_dn, s_up]); cum /= cum[-1]
    k = np.searchsorted(cum, (np.arange(lines) + 0.5) / lines)
    seeds = np.array([[xs[i % 600], yt + (1.6 * h if i >= 600 else -1.6 * h)] for i in k])

    def stop(P, it):
        hit = (np.abs(P[:, 1] - y[jb]) < 1.2 * h) & (np.abs(P[:, 0]) <= L / 2 + h)
        return hit | (np.abs(P[:, 0]) > 0.95 * Wc) | (np.abs(P[:, 1]) > 0.95 * Hc)

    lims = ((-Wd, Wd), (-Hd, Hd))
    paths = _trace(Ef, seeds, h / 2, 20000, stop)
    for p in paths:
        f.line(p, lw=0.9); _mid_visible(f, p, lims)
    if equipotential:
        lev = [v * V for v in (-0.375, -0.25, -0.125, 0, 0.125, 0.25, 0.375)]
        f.ax.contour(X, Y, Vg, levels=lev, colors=INK, linewidths=0.6, linestyles="--", zorder=1)
    t = 0.13 * d
    for yy, sgn in ((yt, "+"), (y[jb], "−")):
        f.ax.add_patch(Rectangle((-L / 2, yy - t / 2), L, t, fc=INK, ec=INK, zorder=5))
        for xx in np.linspace(-L / 2, L / 2, 11)[1:-1]:
            f.ax.text(xx, yy, sgn, color=PAPER, fontsize=10, fontweight="bold", ha="center",
                      va="center_baseline", zorder=7)
    f.text((-L / 2 - 0.08 * d, yt), "+", size=14, ha="right"); f.text((-L / 2 - 0.08 * d, y[jb]), "−", size=14, ha="right")
    Ec = float(np.hypot(*Ef(np.array([[0.0, 0.0]]))[0]))
    f.limits(*lims)
    f.note("plates", f"V={V:g} V, d={d:g} m, L={L:g} m; E at centre {Ec:.4g} V/m "
                     f"(ideal V/d = {V / d:.4g} V/m); {lines} lines ∝ surface charge; dashed = equipotentials "
                     f"every {V / 8:g} V")
    return f


# =============================================================== magnetism
MU0 = 4e-7 * np.pi


def _loop_psi(a, z0, I, Xa, Yr):
    """Flux function psi = r*A_phi of a circular loop (radius a, axial position z0, current I),
    exact via elliptic integrals. Field lines in a meridian plane are contours of psi."""
    from scipy.special import ellipk, ellipe
    r = np.abs(Yr) + 1e-12; z = Xa - z0
    Dq = (a + r) ** 2 + z ** 2; m = np.clip(4 * a * r / Dq, 0, 1 - 1e-12)
    exact = MU0 * I / np.pi * np.sqrt(a / r) / np.sqrt(np.maximum(m, 1e-30)) * ((1 - m / 2) * ellipk(m) - ellipe(m))
    series = MU0 * I * a ** 2 * r / (4 * Dq ** 1.5)
    return r * np.where(m < 1e-4, series, exact)


def _loop_B(a, z0, I, P):
    """(B_axial, B_radial) of a loop at plot points P[:, (axial, radial-signed)]."""
    from scipy.special import ellipk, ellipe
    r = np.abs(P[:, 1]); z = P[:, 0] - z0
    Dq = (a + r) ** 2 + z ** 2; m = np.clip(4 * a * r / Dq, 0, 1 - 1e-12)
    K, E = ellipk(m), ellipe(m); den = np.sqrt(Dq); g = (a - r) ** 2 + z ** 2
    Bz = MU0 * I / (2 * np.pi * den) * (K + (a * a - r * r - z * z) / g * E)
    Br = MU0 * I / (2 * np.pi * den) * z / np.maximum(r, 1e-12) * (-K + (a * a + r * r + z * z) / g * E)
    return np.c_[Bz, np.where(r < 1e-9, 0, Br) * np.sign(P[:, 1])]


def coils(f, loops, kind="coil", lims=None, nlines=7, R=1.0, L=0.0, levels_at=0.0, wire_r=None):
    """Axisymmetric current loops [(radius, axial_pos, I), ...]; axis drawn horizontal.
    kind: 'coil' (show ⊙/⊗ wire sections) or 'magnet' (uniformly magnetised bar)."""
    if lims is None:
        hw, hh = L / 2 + 1.6 * R, 2.1 * R
        hw = max(hw, hh / 0.75); hh = max(hh, 0.75 * hw)
        lims = ((-hw, hw), (-hh, hh))
    x = np.linspace(*lims[0], 760); y = np.linspace(*lims[1], 560); X, Y = np.meshgrid(x, y)
    psi = sum(_loop_psi(a, z0, I, X, Y) for a, z0, I in loops)
    rk = R * (np.arange(nlines) + 0.5) / nlines * (0.97 if kind == "coil" else 1.0)
    lev = sorted(set(np.round(sum(_loop_psi(a, z0, I, np.full_like(rk, levels_at), rk) for a, z0, I in loops), 30)))
    if any(I < 0 for _, _, I in loops) and any(I > 0 for _, _, I in loops):
        lev = sorted(set(lev) | {-v for v in lev})
    Bf = lambda P: sum(_loop_B(a, z0, I, P) for a, z0, I in loops)
    nl = 0; span = min(np.ptp(lims[0]), np.ptp(lims[1]))
    zs = [z0 for _, z0, _ in loops]; zlo, zhi = min(zs), max(zs)
    for k, v in enumerate(lev):
        for p in _contour_lines(X, Y, psi, v):
            nl += 1
            f.line(p, lw=0.9, z=2)
            outside = (np.abs(p[:, 1]) > 1.12 * R) | (p[:, 0] < zlo - 0.25 * R) | (p[:, 0] > zhi + 0.25 * R)
            _heads_outside(f, p, outside, lims, Bf, 0.12 * span)
            if kind != "magnet" and k % 2 == 0:  # inside the coil: every other line, at the centre plane
                cr = [i for i in np.where(np.diff(np.sign(p[:, 0])) != 0)[0]
                      if 0 < i < len(p) - 1 and abs(p[i, 1]) < 0.85 * R]
                for i in cr:
                    d = p[i + 1] - p[i - 1]; f.head(p[i], d if np.dot(d, Bf(p[i:i + 1])[0]) >= 0 else -d)
    if kind == "magnet":
        f.ax.add_patch(Rectangle((-L / 2, -R), L, 2 * R, fc=PAPER, ec=INK, lw=1.4, zorder=5))
        f.line([[0, -R], [0, R]], lw=1.0, z=6)
        north = sum(I for _, _, I in loops) > 0
        f.text((L / 4, 0), "N" if north else "S", size=16, weight="bold")
        f.text((-L / 4, 0), "S" if north else "N", size=16, weight="bold")
    else:
        rw = wire_r or 0.07 * R
        for a, z0, I in loops:
            f.current((z0, a), rw, out=I > 0); f.current((z0, -a), rw, out=I < 0)
    f.limits(*lims)
    return f, nl


def bfield(f, kind="solenoid", I=1.0, R=1.0, length=4.0, turns=10, wires=None, lines=7):
    if kind in ("wire", "wires"):
        W = wires or [((0.0, 0.0), I)]
        P = np.array([w[0] for w in W], float); Iw = np.array([w[1] for w in W], float)
        span = max(1.0, float(np.max(np.ptp(P, 0)))) if len(P) > 1 else 1.0
        c = P.mean(0); Wd = 0.5 * np.ptp(P[:, 0]) + 1.3 * span; Hd = max(0.5 * np.ptp(P[:, 1]) + 1.0 * span, 0.72 * Wd)
        lims = ((c[0] - Wd, c[0] + Wd), (c[1] - Hd, c[1] + Hd))
        x = np.linspace(*lims[0], 700); y = np.linspace(*lims[1], 520); X, Y = np.meshgrid(x, y)
        A = -sum(i * np.log(np.maximum(np.hypot(X - p[0], Y - p[1]), 1e-9)) for p, i in zip(P, Iw))
        if len(P) == 1:  # single wire: circles with r_k in geometric progression (density ∝ 1/r)
            rk = 0.16 * span * 1.38 ** np.arange(lines + 4); rk = rk[rk < 0.95 * Hd]
            lev = -Iw[0] * np.log(rk)
        else:
            far = np.min([np.hypot(X - p[0], Y - p[1]) for p in P], axis=0) > 0.12 * span
            lo, hi = np.percentile(A[far], [3, 99.5])
            if (Iw > 0).any() and (Iw < 0).any():  # keep the pattern symmetric for opposite currents
                M = max(abs(lo), abs(hi)); lev = np.linspace(-M, M, 2 * (3 * lines // 2) + 1)
            else:
                lev = np.linspace(lo, hi, 3 * lines)
        Bw = lambda Q: sum(i * np.c_[-(Q[:, 1] - p[1]), Q[:, 0] - p[0]] /
                           np.maximum(np.hypot(Q[:, 0] - p[0], Q[:, 1] - p[1]), 1e-9)[:, None] ** 2
                           for p, i in zip(P, Iw))
        nl = 0
        for v in sorted(set(lev)):
            for p in _contour_lines(X, Y, A, v):
                nl += 1; f.line(p, lw=0.9)
                closed = np.hypot(*(p[0] - p[-1])) < 1e-6 * span
                if closed and np.ptp(p[:, 1]) < 0.1 * span:
                    continue                                   # tiny loop hugging the wire: no head
                if closed:  # head at the top and bottom of every closed line
                    for i in (np.argmax(p[:, 1]), np.argmin(p[:, 1])):
                        i = int(np.clip(i, 1, len(p) - 2)); d = p[i + 1] - p[i - 1]
                        f.head(p[i], d if np.dot(d, Bw(p[i:i + 1])[0]) >= 0 else -d)
                else:
                    v = _visible(p, lims)
                    if v.sum() > 4:
                        q = p[v]
                        for t in (0.2, 0.8):
                            i = int(np.clip(t * len(q), 1, len(q) - 2)); d = q[i + 1] - q[i - 1]
                            f.head(q[i], d if np.dot(d, Bw(q[i:i + 1])[0]) >= 0 else -d)
        for p, i in zip(P, Iw):
            f.current(p, 0.07 * span, out=i > 0)
        f.limits(*lims)
        f.note("B-field", f"{len(P)} long straight wire(s) (⊙ out of page, ⊗ into page), {nl} field lines = "
                          f"contours of A_z (equal flux between lines; density ∝ |B|)")
        return f
    if kind == "loop":
        lp = [(R, 0.0, I)]; L = 0.0
    elif kind in ("helmholtz", "antihelmholtz"):
        lp = [(R, -R / 2, I), (R, R / 2, I if kind == "helmholtz" else -I)]; L = R
    elif kind == "magnet":
        n = 48; lp = [(R, z, I / n * turns) for z in np.linspace(-length / 2, length / 2, n)]; L = length
    else:
        lp = [(R, z, I) for z in np.linspace(-length / 2, length / 2, turns)]; L = length
    lat = R / 2 if kind == "antihelmholtz" else 0.0
    wr = min(0.07 * R, 0.4 * length / max(turns - 1, 1)) if kind == "solenoid" else None
    _, nl = coils(f, lp, "magnet" if kind == "magnet" else "coil", nlines=lines, R=R, L=L, levels_at=lat, wire_r=wr)
    Bcen = sum(_loop_B(a, z0, Ii, np.array([[0.0, 0.0]]))[0, 0] for a, z0, Ii in lp)
    extra = ""
    if kind == "solenoid":
        extra = f"; B(centre) {Bcen:.4g} T vs ideal μ0nI = {MU0 * turns / length * I:.4g} T"
    elif kind == "loop":
        extra = f"; B(centre) {Bcen:.4g} T = μ0I/2R"
    f.note("B-field", f"{kind}: R={R:g} m, I={I:g} A, {len(lp)} turn(s), {nl} field lines = contours of the "
                      f"exact flux function (elliptic integrals); ⊙ out of page, ⊗ into page{extra}")
    return f


# =============================================================== optics
def rays(f, kind="convex-lens", focal=10.0, u=30.0, h=4.0):
    """Ray diagram from the lens/mirror equation 1/u + 1/v = 1/f (real-is-positive).
    convex-lens / concave-mirror: f > 0; concave-lens / convex-mirror: f < 0."""
    fl = abs(focal) * (1 if kind in ("convex-lens", "concave-mirror") else -1)
    mirror = "mirror" in kind
    inf = abs(u - fl) < 1e-9
    v = np.inf if inf else 1 / (1 / fl - 1 / u)
    m = -v / u if not inf else np.inf
    xo = -u
    xi = (-v if mirror else v) if not inf else None
    hi = m * h if not inf else None
    xs = [xo, fl, -fl, 2 * fl, -2 * fl] + ([xi] if xi is not None else [])
    xmin, xmax = min(xs) - 0.25 * abs(fl), max(xs) + 0.35 * abs(fl)
    if mirror:
        xmax = max(0.6 * abs(fl), max(xs) + 0.3 * abs(fl))
    H = max(abs(h), abs(hi) if hi is not None and abs(hi) < 6 * abs(h) else 0) * 1.45
    real = (not inf) and ((xi < 0) if mirror else (xi > 0))
    P = np.array([xo, h])
    # three principal rays: hit points on the lens/mirror plane x = 0
    hits = [np.array([0.0, h]), np.array([0.0, 0.0])]
    if not inf:
        hits.append(np.array([0.0, hi]))
    xend = xmin if mirror else xmax
    I = np.array([xi, hi]) if not inf else None
    for k, Hh in enumerate(hits):
        f.line([P, Hh], lw=1.0); f.heads_on([P, (P + Hh) / 2, Hh], size=8)
        if inf:  # image at infinity: emergent rays parallel to the central/pole ray
            dirv = np.array([-1, -h / u]) if mirror else np.array([1, -h / u])
        else:
            dirv = (I - Hh) if real else (Hh - I)
            if mirror and not real:
                dirv = Hh - I
        dirv = dirv / np.hypot(*dirv)
        if (dirv[0] < 0) != mirror:
            dirv = -dirv
        tmax = abs((xend - Hh[0]) / dirv[0]) if abs(dirv[0]) > 1e-9 else 10 * abs(fl)
        tmax = min(tmax, abs((1.4 * H) / dirv[1])) if abs(dirv[1]) > 1e-9 else tmax
        end = Hh + dirv * tmax
        f.line([Hh, end], lw=1.0); f.heads_on([Hh, (Hh + end) / 2, end], size=8)
        if not inf and not real:  # virtual: dashed backward extension to the image
            f.line([Hh, I], dashed=True, lw=0.8)
    f.line([[xmin, 0], [xmax, 0]], lw=0.7)                       # principal axis
    if mirror:
        Rm = 2 * abs(fl); th = np.arcsin(min(H / Rm, 0.9))
        cx = -Rm if fl > 0 else Rm                                # centre of curvature side
        t = np.linspace(-th, th, 80)
        arc = np.c_[cx + Rm * np.cos(t) * (1 if fl > 0 else -1), Rm * np.sin(t)]
        arc[:, 0] -= arc[40, 0]
        f.line(arc, lw=1.8)
        for p in arc[::6]:
            f.line([p, p + [0.035 * (xmax - xmin), 0.06 * H]], lw=0.6)
        f.text((-fl, -0.06 * H), "F", va="top", bg=True); f.text((-2 * fl, -0.06 * H), "C", va="top", bg=True)
        for xx in (-fl, -2 * fl):
            f.line([[xx, -0.03 * H], [xx, 0.03 * H]], lw=0.8)
    else:
        f.line([[0, -H], [0, H]], lw=1.3)
        if fl > 0:
            f.head((0, H), (0, 1), 10); f.head((0, -H), (0, -1), 10)
        else:
            f.head((0, H), (0, -1), 10); f.head((0, -H), (0, 1), 10)
        for xx, s in ((fl, "F"), (-fl, "F"), (2 * fl, "2F"), (-2 * fl, "2F")):
            f.line([[xx, -0.03 * H], [xx, 0.03 * H]], lw=0.8)
            f.text((xx, -0.06 * H), s, va="top", size=10, bg=True)
    dx = 0.012 * (xmax - xmin)
    f.arrow((xo, 0), (0, h), lw=1.8, size=12)
    f.text((xo - dx, h / 2), "O", size=12, ha="right")
    if not inf:
        if real:
            f.arrow((xi, 0), (0, hi), lw=1.8, size=12)
        else:
            f.line([[xi, 0], [xi, hi]], dashed=True, lw=1.6); f.head((xi, hi), (0, np.sign(hi)), 12)
        f.text((xi + dx, hi / 2), "I", size=12, ha="left")
    f.limits((xmin, xmax), (-1.25 * H, 1.25 * H))
    f.ax.set_aspect("auto")  # vertical scale stretched: an affine map keeps every ray construction exact
    if inf:
        f.note("image", "object at F: image at infinity (emergent rays parallel)")
    else:
        nat = ("real" if real else "virtual") + ", " + ("inverted" if m < 0 else "upright") + ", " + \
              ("magnified" if abs(m) > 1 else "diminished" if abs(m) < 1 else "same size")
        f.note("image", f"{kind}: f={fl:+g}, u={u:g} -> v={v:+.4g} (real-is-positive), "
                        f"magnification {m:+.4g}; image {nat}, height {hi:+.4g}")
    return f


# =============================================================== waves
def wave_transverse(f, A=2.0, wavelength=4.0, length=10.0, phase=0.0, xlabel="distance / m",
                    ylabel="displacement / cm", grid=False, time=False):
    x = np.linspace(0, length, 1200); y = A * np.sin(2 * np.pi * x / wavelength + phase)
    if time:
        xlabel = "time / s" if xlabel == "distance / m" else xlabel
    f.plot_axes((0, length * 1.06), (-1.45 * A, 1.45 * A), xlabel, ylabel, grid=grid)
    f.ax.plot(x, y, color=INK, lw=1.4)
    crest = (np.pi / 2 - phase) / (2 * np.pi) * wavelength % wavelength
    if crest + wavelength <= length:
        f.dim((crest, A * 1.18), (crest + wavelength, A * 1.18), "T" if time else "λ",
              text_off=(0, 0.1 * A))
        for xx in (crest, crest + wavelength):
            f.ax.plot([xx, xx], [A, A * 1.25], color=INK, lw=0.6, ls=":")
        tr = crest + wavelength / 2
        f.ax.annotate("", xy=(tr, -A), xytext=(tr, 0), arrowprops=dict(arrowstyle="<|-|>,head_length=0.5,head_width=0.2",
                      color=INK, lw=0.8, mutation_scale=9, shrinkA=0, shrinkB=0))
        f.ax.text(tr + 0.02 * length, -A / 2, "A", fontsize=11, va="center")
    f.note("wave", f"A={A:g}, {'T' if time else 'λ'}={wavelength:g}; "
                   + (f"f = 1/T = {1 / wavelength:.4g} Hz" if time else f"k = 2π/λ = {2 * np.pi / wavelength:.4g}"))
    return f


def wave_longitudinal(f, A=0.35, wavelength=4.0, length=12.0, n=72):
    """Particle positions x0 + A sin(kx0): compressions (C) / rarefactions (R) are exact."""
    x0 = np.linspace(0, length, n); k = 2 * np.pi / wavelength
    xs = x0 + A * np.sin(k * x0)
    for x in xs:
        if 0 <= x <= length:
            f.line([[x, -1], [x, 1]], lw=0.9)
    # compression where dx/dx0 = 1 + A k cos(k x0) is minimum: k x0 = pi (mod 2 pi)
    for j in range(int(length / wavelength) + 1):
        for x0c, s in ((wavelength / 2 + j * wavelength, "C"), (j * wavelength, "R")):
            if 0.03 * length < x0c < 0.97 * length:
                f.text((x0c, 1.25), s, size=12)
    c1 = wavelength / 2
    if c1 + wavelength < length:
        f.dim((c1, -1.25), (c1 + wavelength, -1.25), "λ", text_off=(0, -0.25))
    f.arrow((0.1 * length, -2.0), (0.18 * length, 0), lw=1.0, size=10)
    f.text((0.3 * length, -2.0), "direction of travel", size=10, ha="left")
    f.limits((-0.03 * length, 1.03 * length), (-2.5, 1.8))
    f.note("wave", f"longitudinal: λ={wavelength:g}, particle amplitude {A:g}; C = compression, R = rarefaction")
    return f


def wave_standing(f, harmonic=3, ends="fixed-fixed", length=1.0, speed=None):
    L = length; n = harmonic
    if ends == "closed-open" and n % 2 == 0:
        raise SystemExit("closed-open pipe supports only odd harmonics (1, 3, 5, ...)")
    lam = 4 * L / n if ends == "closed-open" else 2 * L / n
    x = np.linspace(0, L, 1000); k = 2 * np.pi / lam
    y = np.sin(k * x) if ends in ("fixed-fixed", "closed-open") else np.cos(k * x)
    A = 0.6
    f.line(np.c_[x, A * y], lw=1.4); f.line(np.c_[x, -A * y], dashed=True, lw=1.1)
    xn = [i * lam / 2 + (0 if ends in ("fixed-fixed", "closed-open") else lam / 4) for i in range(2 * n + 2)]
    xn = [v for v in xn if -1e-9 <= v <= L + 1e-9]
    xa = [v + lam / 4 for v in xn] + [v - lam / 4 for v in xn]
    xa = sorted({round(v, 9) for v in xa if -1e-9 <= v <= L + 1e-9})
    for v in xn:
        f.text((v, -0.8 if ends == "fixed-fixed" else -1.15), "N", size=12, va="top", bg=True)
    for v in xa:
        f.text((v, 0.8 if ends == "fixed-fixed" else 1.15), "A", size=12, va="bottom", bg=True)
    if ends == "fixed-fixed":
        for xx in (0, L):
            f.line([[xx, -0.45], [xx, 0.45]], lw=2.2)
            s = -1 if xx == 0 else 1
            for yy in np.linspace(-0.45, 0.33, 7):
                f.line([[xx, yy], [xx + s * 0.025 * L, yy + 0.1]], lw=0.6)
        f.ax.plot([0, L], [0, 0], color=INK, lw=0.5, ls=":")
    else:
        for yy in (1.0, -1.0):
            f.line([[0, yy], [L, yy]], lw=1.6)
        if ends == "closed-open":
            f.line([[0, -1.0], [0, 1.0]], lw=3.0)
        f.ax.plot([0, L], [0, 0], color=INK, lw=0.5, ls=":")
    f.dim((0, -1.6), (L, -1.6), f"{L:g} m", text_off=(0, -0.22))
    f.limits((-0.08 * L, 1.08 * L), (-2.0, 1.6))
    f.ax.set_aspect("auto")
    fr = f", f = v/λ = {speed / lam:.4g} Hz" if speed else ""
    f.note("standing wave", f"{ends}, harmonic n={n}: λ = {lam:.4g} m{fr}; nodes at "
                            + ", ".join(f"{v:.3g}" for v in xn) + " m; N = node, A = antinode "
                            + ("(displacement)" if ends != "fixed-fixed" else ""))
    return f


def interference(f, d=2.0, wavelength=0.6, extent=6.0, nodal=False):
    """Two coherent in-phase sources: crests (solid) / troughs (dashed) at one instant,
    antinodal lines = loci of path difference mλ (exact hyperbolae)."""
    S = np.array([[0, -d / 2], [0, d / 2]])
    lims = ((-0.12 * extent, extent), (-0.65 * extent, 0.65 * extent))
    th = np.linspace(-np.pi / 2, np.pi / 2, 400)
    for s in S:
        for nr in np.arange(0.5, 2 * extent / wavelength, 0.5):
            r = nr * wavelength
            if r > 1.15 * extent:
                break
            f.line(np.c_[s[0] + r * np.cos(th), s[1] + r * np.sin(th)], dashed=(nr % 1 != 0), lw=0.5)
    x = np.linspace(0.02, lims[0][1], 600); y = np.linspace(*lims[1], 600); X, Y = np.meshgrid(x, y)
    D = np.hypot(X - S[0, 0], Y - S[0, 1]) - np.hypot(X - S[1, 0], Y - S[1, 1])
    mmax = int(np.floor(d / wavelength - 1e-9))
    for mm in range(-mmax, mmax + 1):
        for p in _contour_lines(X, Y, D, mm * wavelength):
            f.line(p, lw=2.0, z=3)
            q = p[np.argmax(p[:, 0])]
            f.text(q + [0.04 * extent, 0], f"m = {mm}".replace("-", "−"), size=9, ha="left", bg=True)
        if nodal and mm < mmax + 1:
            for p in _contour_lines(X, Y, D, (mm + 0.5) * wavelength):
                f.line(p, lw=1.0, ls=(0, (1, 2)), z=3)
    for s, lab in zip(S, ("S$_1$", "S$_2$")):
        f.dot(s, 0.06 * wavelength + 0.012 * extent); f.text(s + [-0.05 * extent, 0], lab, ha="right")
    f.limits((lims[0][0] - 0.05 * extent, lims[0][1] + 0.18 * extent), lims[1])
    f.note("interference", f"d={d:g}, λ={wavelength:g}: solid arcs = crests, dashed = troughs; thick lines = "
                           f"maxima (path difference mλ), |m| ≤ {mmax}; far-field angles sinθ = mλ/d: "
                           + ", ".join(f"{np.degrees(np.arcsin(mm * wavelength / d)):.1f}°" for mm in range(mmax + 1)))
    return f


# =============================================================== mechanics
def kepler(f, a=1.0, e=0.6, M=1.0, n_marks=12, sectors=True):
    b = a * np.sqrt(1 - e ** 2); T = np.sqrt(a ** 3 / M)
    E = np.linspace(0, 2 * np.pi, 800)
    f.line(np.c_[a * (np.cos(E) - e), b * np.sin(E)], lw=1.3)

    def pos(Mean):
        Ek = np.array(Mean, float).copy()
        for _ in range(60):
            Ek -= (Ek - e * np.sin(Ek) - Mean) / (1 - e * np.cos(Ek))
        return np.c_[a * (np.cos(Ek) - e), b * np.sin(Ek)]

    if sectors:
        w = 2 * np.pi / n_marks
        for M0, hatch in ((-w / 2, "////"), (np.pi - w / 2, "\\\\\\\\")):
            arc = pos(np.linspace(M0, M0 + w, 60))
            f.ax.add_patch(Polygon(np.vstack([[0, 0], arc]), closed=True, fc=PAPER, ec=INK, lw=0.8,
                                   hatch=hatch, zorder=1))
    for p in pos(np.linspace(0, 2 * np.pi, n_marks, endpoint=False)):
        f.circle(p, 0.018 * a, fill=INK, lw=0)
    f.circle((0, 0), 0.05 * a, lw=1.2)
    f.text((0, 0), "S", size=9)
    f.text((-2 * a * e, 0), "×", size=12)
    f.ax.plot([-a * (1 + e), a * (1 - e)], [0, 0], color=INK, lw=0.5, ls=":")
    f.dim((-a * (1 + e), -b - 0.12 * a), (a * (1 - e), -b - 0.12 * a), "2a", text_off=(0, -0.06 * a))
    f.text((a * (1 - e) + 0.04 * a, 0.08 * a), "perihelion", size=10, ha="left")
    f.text((-a * (1 + e) - 0.04 * a, 0.08 * a), "aphelion", size=10, ha="right")
    f.limits((-a * (1 + e) - 0.45 * a, a * (1 - e) + 0.5 * a), (-b - 0.3 * a, b + 0.15 * a))
    f.note("orbit", f"a={a:g} AU, e={e:g}, b={b:.4g} AU, r_p={a * (1 - e):.4g} AU, r_a={a * (1 + e):.4g} AU, "
                    f"T={T:.4g} yr; dots = equal time steps; hatched sectors swept in equal times have equal "
                    f"area (Kepler II); S = Sun at focus, × = empty focus")
    return f


def projectile(f, v0=30.0, angle=45.0, k=0.0, g=9.81, arrows=5):
    th = np.radians(angle)

    def run(kk):
        X = np.array([0, 0.0]); V = v0 * np.array([np.cos(th), np.sin(th)])
        dt = 2 * v0 * np.sin(th) / g / 4000; P = [X.copy()]; Vs = [V.copy()]
        acc = lambda v: np.array([0, -g]) - kk * np.linalg.norm(v) * v
        while True:
            k1v = acc(V); k1x = V
            k2v = acc(V + .5 * dt * k1v); k2x = V + .5 * dt * k1v
            k3v = acc(V + .5 * dt * k2v); k3x = V + .5 * dt * k2v
            k4v = acc(V + dt * k3v); k4x = V + dt * k3v
            X = X + dt * (k1x + 2 * k2x + 2 * k3x + k4x) / 6
            V = V + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
            P.append(X.copy()); Vs.append(V.copy())
            if X[1] < 0:
                break
        P = np.array(P); fr = P[-2, 1] / (P[-2, 1] - P[-1, 1]); P[-1] = P[-2] + fr * (P[-1] - P[-2])
        return P, np.array(Vs), (len(P) - 2 + fr) * dt

    P, Vs, T = run(k); R, H = P[-1, 0], P[:, 1].max()
    P0, _, T0 = run(0) if k > 0 else (P, Vs, T)
    Rm = max(R, P0[-1, 0]); Hm = max(H, P0[:, 1].max())
    f.plot_axes((0, Rm * 1.12), (0, max(Hm * 1.35, Rm * 0.25)), "x / m", "y / m", equal=True)
    f.ax.plot(P[:, 0], P[:, 1], color=INK, lw=1.5)
    if k > 0:
        f.ax.plot(P0[:, 0], P0[:, 1], color=INK, lw=1.0, ls="--")
        f.note("no-drag (dashed)", f"range {P0[-1, 0]:.4g} m, h_max {P0[:, 1].max():.4g} m, T {T0:.4g} s")
    vmax = np.linalg.norm(Vs, axis=1).max(); sc = 0.16 * Rm / vmax
    for i in np.linspace(len(P) // 6, len(P) - 2, arrows).astype(int)[:-1]:
        f.arrow(P[i], Vs[i] * sc, lw=1.0, size=9)
    f.arrow((0, 0), Vs[0] * sc * 1.2, lw=1.4, size=11, label="$v_0$", label_off=(-0.03 * Rm, 0.02 * Rm), ha="right")
    f.ax.add_patch(Arc((0, 0), 0.16 * Rm, 0.16 * Rm, theta1=0, theta2=angle, color=INK, lw=0.8))
    f.text((0.1 * Rm * np.cos(th / 2), 0.1 * Rm * np.sin(th / 2)), "θ", size=11, ha="left")
    ia = np.argmax(P[:, 1])
    f.ax.plot([P[ia, 0]] * 2, [0, H], color=INK, lw=0.6, ls=":")
    f.text((P[ia, 0] + 0.015 * Rm, H / 2), "$h_{max}$", size=10, ha="left")
    f.text((R, 0.025 * Rm), "R", size=11, va="bottom")
    f.note("projectile", f"v0={v0:g} m/s at {angle:g}°, k={k:g} 1/m, g={g:g}: range {R:.4g} m, "
                         f"h_max {H:.4g} m, flight time {T:.4g} s (RK4)")
    return f


# =============================================================== maths / generic
_NS = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "arctan2", "sinh",
                                   "cosh", "tanh", "pi", "e", "hypot", "where", "maximum", "minimum",
                                   "arctan", "arcsin", "arccos", "log10", "sign", "heaviside")}
STYLES = ["-", "--", "-.", ":"]


def _ev(expr, **v):
    return eval(expr, {"__builtins__": {}}, dict(_NS, **v))


def graph(f, exprs, xr=(0, 10), yr=None, xlabel="x", ylabel="y", labels=None, grid=False, ticks=True):
    x = np.linspace(*xr, 2000); ys = []
    for e in exprs:
        y = np.broadcast_to(np.asarray(_ev(e, x=x), float), x.shape).copy(); y[~np.isfinite(y)] = np.nan
        ys.append(y)
    if yr is None:
        lo = min(0, np.nanmin([np.nanmin(y) for y in ys])); hi = max(0, np.nanmax([np.nanmax(y) for y in ys]))
        pad = 0.12 * (hi - lo or 1); yr = (lo - (pad if lo < 0 else 0), hi + pad)
    f.plot_axes((xr[0], xr[1] + 0.04 * (xr[1] - xr[0])), yr, xlabel, ylabel, grid=grid, ticks=ticks)
    for i, (e, y) in enumerate(zip(exprs, ys)):
        f.ax.plot(x, y, color=INK, lw=1.4, ls=STYLES[i % 4])
        if labels and i < len(labels) and labels[i]:
            ok = np.where(np.isfinite(y) & (y > yr[0]) & (y < yr[1]))[0]
            if len(ok):
                j = ok[-1]; f.ax.text(x[j], y[j], " " + labels[i], fontsize=10, va="center", ha="left")
        f.note(f"curve {i + 1}", f"y = {e}; y in [{np.nanmin(y):.4g}, {np.nanmax(y):.4g}] on x in [{xr[0]:g}, {xr[1]:g}]")
    return f


def contour(f, expr, xr=(-3, 3), yr=(-3, 3), n=12, xlabel="x", ylabel="y", labels=True):
    x = np.linspace(*xr, 400); y = np.linspace(*yr, 400); X, Y = np.meshgrid(x, y)
    Z = np.broadcast_to(np.asarray(_ev(expr, x=X, y=Y), float), X.shape)
    f.plot_axes(xr, yr, xlabel, ylabel, equal=True, origin=False, at_zero=False)
    cs = f.ax.contour(X, Y, Z, levels=n, colors=INK, linewidths=0.9)
    cs.set_linestyles(["--" if v < 0 else "-" for v in cs.levels])
    if labels:
        f.ax.clabel(cs, fontsize=7, inline=True, fmt=lambda v: f"{v:g}".replace("-", "−"))
    f.note("contour", f"z = {expr}; z in [{Z.min():.4g}, {Z.max():.4g}]; dashed = negative levels")
    return f


def vfield(f, fx, fy, extent=2.0, n=13, stream=False):
    t = np.linspace(-extent, extent, n); X, Y = np.meshgrid(t, t)
    U = np.broadcast_to(_ev(fx, x=X, y=Y), X.shape).astype(float); V = np.broadcast_to(_ev(fy, x=X, y=Y), X.shape).astype(float)
    m = np.hypot(U, V)
    f.plot_axes((-extent * 1.1, extent * 1.1), (-extent * 1.1, extent * 1.1), "x", "y", equal=True, origin=False,
                at_zero=False)
    if stream:
        tt = np.linspace(-extent, extent, 200); XX, YY = np.meshgrid(tt, tt)
        f.ax.streamplot(XX, YY, np.broadcast_to(_ev(fx, x=XX, y=YY), XX.shape).astype(float),
                        np.broadcast_to(_ev(fy, x=XX, y=YY), XX.shape).astype(float), color="0.35",
                        linewidth=0.6, density=0.8, arrowsize=0.8)
    f.ax.quiver(X, Y, U, V, color=INK, pivot="mid", angles="xy", scale_units="xy",
                scale=max(m.max(), 1e-12) / (0.85 * (t[1] - t[0])), width=0.0045, headwidth=4, zorder=3)
    f.note("vfield", f"F = ({fx}, {fy}); |F| in [{m.min():.3g}, {m.max():.3g}]")
    return f


# =============================================================== chemistry
COMMON = {  # common name -> SMILES
    "water": "O", "ammonia": "N", "methane": "C", "ethane": "CC", "propane": "CCC", "butane": "CCCC",
    "ethene": "C=C", "ethylene": "C=C", "ethyne": "C#C", "acetylene": "C#C", "methanol": "CO",
    "ethanol": "CCO", "propan-1-ol": "CCCO", "propan-2-ol": "CC(C)O", "methanal": "C=O",
    "formaldehyde": "C=O", "ethanal": "CC=O", "propanone": "CC(C)=O", "acetone": "CC(C)=O",
    "ethanoic acid": "CC(=O)O", "acetic acid": "CC(=O)O", "methanoic acid": "OC=O",
    "carbon dioxide": "O=C=O", "co2": "O=C=O", "nitrogen": "N#N", "oxygen": "O=O", "chlorine": "ClCl",
    "hydrogen": "[H][H]", "hydrogen chloride": "Cl", "hcl": "Cl", "hydrogen fluoride": "F",
    "hydrogen sulfide": "S", "phosphine": "P", "boron trifluoride": "FB(F)F", "bf3": "FB(F)F",
    "beryllium chloride": "Cl[Be]Cl", "becl2": "Cl[Be]Cl", "sulfur dioxide": "O=S=O",
    "sulfur hexafluoride": "FS(F)(F)(F)(F)F", "sf6": "FS(F)(F)(F)(F)F",
    "phosphorus pentachloride": "ClP(Cl)(Cl)(Cl)Cl", "pcl5": "ClP(Cl)(Cl)(Cl)Cl",
    "xenon tetrafluoride": "F[Xe](F)(F)F", "xef4": "F[Xe](F)(F)F", "xenon difluoride": "F[Xe]F",
    "chlorine trifluoride": "FCl(F)F", "clf3": "FCl(F)F", "sulfur tetrafluoride": "FS(F)(F)F",
    "sf4": "FS(F)(F)F", "brf5": "FBr(F)(F)(F)F", "xef2": "F[Xe]F", "nh3": "N", "h2o": "O", "ch4": "C",
    "h2s": "S", "ph3": "P", "nf3": "FN(F)F", "so2": "O=S=O", "ccl4": "ClC(Cl)(Cl)Cl", "cl2": "ClCl",
    "n2": "N#N", "o2": "O=O", "h2": "[H][H]", "hf": "F",
    "bromine pentafluoride": "FBr(F)(F)(F)F", "nitrogen trifluoride": "FN(F)F",
    "tetrachloromethane": "ClC(Cl)(Cl)Cl", "silane": "[SiH4]", "ammonium": "[NH4+]",
    "hydroxonium": "[OH3+]", "oxonium": "[OH3+]", "urea": "NC(N)=O", "benzene": "c1ccccc1",
    "methylbenzene": "Cc1ccccc1", "toluene": "Cc1ccccc1", "phenol": "Oc1ccccc1",
    "phenylamine": "Nc1ccccc1", "cyclohexane": "C1CCCCC1", "cyclohexene": "C1=CCCCC1",
    "glycine": "NCC(=O)O", "alanine": "C[C@H](N)C(=O)O", "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "ethyl ethanoate": "CCOC(C)=O", "chloroethane": "CCCl", "bromoethane": "CCBr",
    "ethylamine": "CCN", "but-2-ene": "CC=CC", "2-methylpropane": "CC(C)C", "lactic acid": "CC(O)C(=O)O",
    "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
}
# experimental bond angles (gas phase; NIST CCCBDB / standard data), keyed by canonical heavy-atom SMILES
EXP_ANGLE = {"O": 104.5, "N": 106.7, "C": 109.5, "S": 92.1, "P": 93.3, "FN(F)F": 102.2, "FB(F)F": 120.0,
             "O=C=O": 180.0, "O=S=O": 119.3, "F[S](F)(F)(F)(F)F": 90.0, "FS(F)(F)(F)(F)F": 90.0,
             "ClP(Cl)(Cl)(Cl)Cl": 90.0, "F[Xe](F)(F)F": 90.0, "FCl(F)F": 87.5, "FS(F)(F)F": 173.1,
             "FBr(F)(F)(F)F": 84.8, "F[Xe]F": 180.0, "Cl[Be]Cl": 180.0, "ClC(Cl)(Cl)Cl": 109.5,
             "[SiH4]": 109.5, "[NH4+]": 109.5, "F": None, "Cl": None}
VSEPR = {  # (domains, lone pairs): (shape, ideal angle, bond slots, lone-pair slots); slot = (dx, dy, kind)
    (2, 0): ("linear", 180, [(-1, 0, "p"), (1, 0, "p")], []),
    (3, 0): ("trigonal planar", 120, [(0, 1, "p"), (-.866, -.5, "p"), (.866, -.5, "p")], []),
    (3, 1): ("bent (V-shaped)", "<120", [(-.866, -.5, "p"), (.866, -.5, "p")], [(0, 1)]),
    (4, 0): ("tetrahedral", 109.5, [(0, 1, "p"), (-.94, -.34, "p"), (.62, -.62, "w"), (.12, -.95, "h")], []),
    (4, 1): ("trigonal pyramidal", 107, [(-.87, -.5, "p"), (.66, -.6, "w"), (.08, -.98, "h")], [(0, 1)]),
    (4, 2): ("bent (V-shaped)", 104.5, [(-.791, -.612, "p"), (.791, -.612, "p")], [(-.6, .8), (.6, .8)]),
    (4, 3): ("linear", 180, [(1, 0, "p")], [(0, 1), (-1, 0), (0, -1)]),
    (5, 0): ("trigonal bipyramidal", "120 and 90", [(0, 1, "p"), (0, -1, "p"), (-1, 0, "p"), (.7, -.4, "w"), (.62, .32, "h")], []),
    (5, 1): ("see-saw", "<120 and <90", [(0, 1, "p"), (0, -1, "p"), (.7, -.4, "w"), (.62, .32, "h")], [(-1, 0)]),
    (5, 2): ("T-shaped", "<90", [(0, 1, "p"), (0, -1, "p"), (1, 0, "p")], [(-.75, .5), (-.75, -.5)]),
    (5, 3): ("linear", 180, [(0, 1, "p"), (0, -1, "p")], [(-1, 0), (.75, -.45), (.75, .45)]),
    (6, 0): ("octahedral", 90, [(0, 1, "p"), (0, -1, "p"), (-1, 0, "p"), (1, 0, "p"), (.5, -.55, "w"), (-.5, .55, "h")], []),
    (6, 1): ("square pyramidal", "<90", [(0, 1, "p"), (-1, 0, "p"), (1, 0, "p"), (.5, -.55, "w"), (-.5, .55, "h")], [(0, -1)]),
    (6, 2): ("square planar", 90, [(-1, 0, "p"), (1, 0, "p"), (0, 1, "p"), (0, -1, "p")], [(.6, .6), (-.6, -.6)]),
}


def load_mol(spec):
    from rdkit import Chem
    s = COMMON.get(spec.strip().lower(), spec.split("smiles:", 1)[-1])
    if os.path.exists(spec):
        m = Chem.MolFromMolFile(spec) if spec.endswith((".mol", ".sdf")) else None
    else:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        m = Chem.MolFromSmiles(s)
        if m is None:  # hypervalent species (ClF3, BrF5, XeF4 ...): accept without valence check
            m = Chem.MolFromSmiles(s, sanitize=False)
            if m is not None:
                m.UpdatePropertyCache(strict=False)
                Chem.SanitizeMol(m, Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES,
                                 catchErrors=True)
    if m is None:
        raise SystemExit(f"Could not parse '{spec}'. Use a common name (see `list`), a SMILES string "
                         f"(optionally 'smiles:...') or a .mol file.")
    return m


def _mol_note(f, m):
    from rdkit.Chem import Descriptors, rdMolDescriptors
    f.note("molecule", f"{rdMolDescriptors.CalcMolFormula(m)}, M = {Descriptors.MolWt(m):.2f} g/mol")


def _png_into(f, png_bytes):
    import io
    img = plt.imread(io.BytesIO(png_bytes))[..., :3]
    ink = np.where(img.min(axis=2) < 0.9)
    if len(ink[0]):  # crop to the drawing plus a margin
        pad = 40; y0, y1 = max(ink[0].min() - pad, 0), ink[0].max() + pad; x0, x1 = max(ink[1].min() - pad, 0), ink[1].max() + pad
        img = img[y0:y1, x0:x1]
    f.ax.imshow(img, interpolation="lanczos"); f.ax.set_aspect("equal")
    f.limits((0, img.shape[1]), (img.shape[0], 0))


def structure(f, spec, style="skeletal", size=(2400, 1800), svg_path=None, label_c=False):
    """2D structural formula via RDKit depiction in black and white.
    skeletal: carbon skeleton, H on heteroatoms only; displayed: every atom and bond."""
    from rdkit import Chem
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    m = load_mol(spec)
    _mol_note(f, m)
    if style == "displayed":
        m = Chem.AddHs(m)
        for a in m.GetAtoms():
            if a.GetSymbol() == "C":
                a.SetProp("atomLabel", "C")
    elif label_c:
        for a in m.GetAtoms():
            if a.GetSymbol() == "C":
                a.SetProp("atomLabel", "C")
    rdDepictor.SetPreferCoordGen(True); rdDepictor.Compute2DCoords(m)

    def draw(d):
        o = d.drawOptions(); o.useBWAtomPalette(); o.bondLineWidth = 5
        o.fixedBondLength = 170; o.fixedFontSize = 58; o.padding = 0.05
        o.clearBackground = True; o.multipleBondOffset = 0.17; o.additionalAtomLabelPadding = 0.08
        stix = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "STIXGeneral.ttf")
        if os.path.exists(stix):
            o.fontFile = stix
        rdMolDraw2D.PrepareAndDrawMolecule(d, m); d.FinishDrawing()
        return d.GetDrawingText()

    png = draw(rdMolDraw2D.MolDraw2DCairo(*size))
    if svg_path:
        open(svg_path, "w").write(draw(rdMolDraw2D.MolDraw2DSVG(*size)))
        f.note("svg", svg_path)
    _png_into(f, png)
    return f


def _star(m):
    """Return (centre index, [(terminal index, bond order)]) for an AXn molecule, else None."""
    n = m.GetNumAtoms()
    if n == 2:
        a, b = m.GetAtomWithIdx(0), m.GetAtomWithIdx(1)
        c = 0 if (a.GetAtomicNum() > b.GetAtomicNum() or b.GetSymbol() == "H") else 1
        bd = m.GetBondBetweenAtoms(0, 1)
        return c, [(1 - c, int(bd.GetBondTypeAsDouble()))]
    for a in m.GetAtoms():
        if a.GetDegree() == n - 1:
            return a.GetIdx(), [(nb.GetIdx(), int(m.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble()))
                                for nb in a.GetNeighbors()]
    return None


def _valence_e(Z):
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetNOuterElecs(Z)


def shape(f, spec, show_lp=True, angle_text=None):
    """VSEPR shape (AXnEm) drawn with in-plane, wedge (towards viewer) and hashed (away) bonds."""
    from rdkit import Chem
    m0 = load_mol(spec); key = Chem.MolToSmiles(m0)
    m = Chem.AddHs(m0); st = _star(m)
    if st is None:
        raise SystemExit("shape needs a molecule/ion with one central atom (AXnEm), e.g. water, NH3, SF6, [NH4+]")
    c, term = st; ca = m.GetAtomWithIdx(c)
    lp = (_valence_e(ca.GetAtomicNum()) - sum(o for _, o in term) - ca.GetFormalCharge()) // 2
    lp = max(lp, 0); dom = len(term) + lp
    if (dom, lp) not in VSEPR:
        raise SystemExit(f"no VSEPR layout for {dom} electron domains with {lp} lone pairs")
    name, ideal, bslots, lslots = VSEPR[(dom, lp)]
    term = sorted(term, key=lambda t: -t[1])  # multiple bonds take in-plane slots first
    L = 1.0
    sym_c = ca.GetSymbol()
    f.text((0, 0), sym_c, size=20)
    ends = []
    for (t, o), (dx, dy, kind) in zip(term, bslots):
        u = np.array([dx, dy]) / np.hypot(dx, dy); p0, p1 = u * 0.2, u * (L - 0.17)
        nrm = np.array([-u[1], u[0]])
        if kind == "w":
            f.ax.add_patch(Polygon([p0, p1 + nrm * 0.075, p1 - nrm * 0.075], closed=True, fc=INK, ec=INK, zorder=3))
        elif kind == "h":
            for tt in np.linspace(0.0, 1.0, 8):
                q = p0 + (p1 - p0) * tt; w = 0.01 + 0.07 * tt
                f.line([q + nrm * w, q - nrm * w], lw=1.3)
        else:
            offs = {1: [0], 2: [-0.045, 0.045], 3: [-0.07, 0, 0.07]}[min(o, 3)]
            for of in offs:
                f.line([p0 + nrm * of, p1 + nrm * of], lw=1.6)
        f.text(u * L, m.GetAtomWithIdx(t).GetSymbol(), size=18)
        ends.append((u, kind))
    if show_lp:
        for dx, dy in lslots:
            u = np.array([dx, dy]) / np.hypot(dx, dy); nrm = np.array([-u[1], u[0]])
            f.ax.add_patch(Ellipse(u * 0.42, 0.42, 0.2, angle=np.degrees(np.arctan2(dy, dx)), fc="none", ec=INK, lw=1.0))
            for s in (1, -1):
                f.dot(u * 0.45 + nrm * 0.035 * s, 0.018)
    exp = EXP_ANGLE.get(key)
    pl = [u for u, k in ends if k == "p"]
    ang = angle_text or (f"{exp:g}°" if isinstance(exp, (int, float)) else None)
    if len(pl) >= 2 and len(term) > 1:
        import itertools
        others = [u for u, k in ends if k != "p"] + [np.array(v) / np.hypot(*v) for v in lslots]
        prs = list(itertools.combinations(pl, 2)); best = max(np.dot(a, b) for a, b in prs)

        def clear(pr):  # bisector as far as possible from wedge/hash bonds and lone pairs
            bis = pr[0] + pr[1]; bis = bis / (np.hypot(*bis) or 1)
            return -max([np.dot(bis, o) for o in others] or [-1])
        u1, u2 = max([pr for pr in prs if np.dot(*pr) > best - 1e-6], key=clear)
        a1, a2 = np.degrees(np.arctan2(u1[1], u1[0])), np.degrees(np.arctan2(u2[1], u2[0]))
        lo, hi = sorted((a1, a2))
        if hi - lo > 180:
            lo, hi = hi, lo + 360
        f.ax.add_patch(Arc((0, 0), 0.8, 0.8, theta1=lo, theta2=hi, color=INK, lw=0.8))
        mid = np.radians((lo + hi) / 2)
        if ang is None:  # no data: ideal VSEPR angle of the drawn (in-plane) pair, '<' if lone pairs compress it
            ang = ("<" if lp and dom < 5 else "") + f"{hi - lo:.4g}°"
        f.text(0.6 * np.array([np.cos(mid), np.sin(mid)]), ang, size=12, bg=True)
    chg = Chem.GetFormalCharge(m)
    if chg:
        for s in (-1, 1):
            f.line([[1.17 * s, 1.3], [1.25 * s, 1.3], [1.25 * s, -1.3], [1.17 * s, -1.3]], lw=1.4)
        f.text((1.35, 1.35), ("+" if chg > 0 else "−") if abs(chg) == 1 else f"{abs(chg)}{'+' if chg > 0 else '−'}", size=14)
    f.limits((-1.55, 1.7), (-1.35, 1.45))
    _mol_note(f, m0)
    f.note("VSEPR", f"{sym_c}: {len(term)} bonding domain(s) + {lp} lone pair(s) = {dom} electron domains -> "
                    f"{name}; labelled angle {'(experimental) ' if isinstance(exp, (int, float)) and not angle_text else ''}{ang}; "
                    f"wedge = towards viewer, hashed = away")
    return f


def dotcross(f, spec, shells=True):
    """Dot-and-cross diagram (outer shells) for AXn molecules: central-atom electrons = dots,
    other atoms' electrons = crosses; shared pairs drawn in the overlap of the shells."""
    from rdkit import Chem
    m0 = load_mol(spec); m = Chem.AddHs(m0)
    if Chem.GetFormalCharge(m):
        raise SystemExit("dotcross supports neutral molecules only")
    st = _star(m)
    if st is None:
        raise SystemExit("dotcross needs a molecule with one central atom or a diatomic, e.g. water, CO2, Cl2")
    c, term = st; ca = m.GetAtomWithIdx(c)
    r0 = 1.0; rh = 0.72
    rad = lambda a: rh if a.GetSymbol() == "H" else r0
    nt = len(term)
    angs = {1: [0], 2: [180, 0], 3: [180, 0, 270], 4: [180, 0, 90, 270]}.get(nt, list(np.linspace(0, 360, nt, endpoint=False)))
    rc = rad(ca)
    f.circle((0, 0), rc, fill="none", lw=1.1, z=2); f.text((0, 0), ca.GetSymbol(), size=18)
    used = []
    es = 0.08

    def mark(p, kind):
        if kind == "dot":
            f.dot(p, 0.045)
        else:
            k = 0.05
            f.line([p - k, p + k], lw=1.1); f.line([p + [-k, k], p + [k, -k]], lw=1.1)

    def pairs_on(centre, r, kind, n, avoid):
        """n non-bonding electrons in pairs on a shell, away from the bond directions."""
        cand = [a for a in (90, 270, 180, 0, 45, 135, 225, 315) if all(abs((a - b + 180) % 360 - 180) > 50 for b in avoid)]
        for k in range(n // 2 + n % 2):
            a = np.radians(cand[k % len(cand)]); u = np.array([np.cos(a), np.sin(a)]); t = np.array([-u[1], u[0]])
            for s in ((1, -1) if (k < n // 2) else (0,)):
                mark(np.asarray(centre) + u * r + t * s * es * 1.1, kind)

    for (ti, o), a in zip(term, angs):
        ta = m.GetAtomWithIdx(ti); rt = rad(ta); D = rc + rt - 0.62 * min(rc, rt)
        u = np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]); tp = np.array([-u[1], u[0]])
        ct = u * D
        f.circle(ct, rt, fill="none", lw=1.1, z=2); f.text(ct, ta.GetSymbol(), size=18)
        # overlap centre on the bond axis (radical axis of the two circles)
        xo = (D ** 2 + rc ** 2 - rt ** 2) / (2 * D); oc = u * xo
        for j, of in enumerate(np.linspace(-(o - 1) / 2, (o - 1) / 2, o) * es * 2.6):
            mark(oc + tp * of - u * es, "dot"); mark(oc + tp * of + u * es, "cross")
        nb = _valence_e(ta.GetAtomicNum()) - o
        pairs_on(ct, rt, "cross", nb, [(a + 180) % 360])
        used.append(a)
    nbc = _valence_e(ca.GetAtomicNum()) - sum(o for _, o in term)
    pairs_on((0, 0), rc, "dot", nbc, used)
    pts = [np.zeros(2)] + [np.array([np.cos(np.radians(a)), np.sin(np.radians(a))]) * 1.5 for a in angs]
    P = np.array(pts); ext = 1.25
    f.limits((P[:, 0].min() - ext, P[:, 0].max() + ext), (P[:, 1].min() - ext, P[:, 1].max() + ext))
    _mol_note(f, m0)
    shared = sum(o for _, o in term)
    f.note("dot-and-cross", f"outer shells only; {ca.GetSymbol()} electrons = dots, others = crosses; "
                            f"{shared} shared pair(s), {nbc // 2} lone pair(s) on {ca.GetSymbol()}")
    return f


CRYSTALS = {  # name: (formula, structure, kwargs); a, c in Å (experimental, ~300 K)
    "nacl": ("NaCl", "rocksalt", dict(a=5.640)), "mgo": ("MgO", "rocksalt", dict(a=4.212)),
    "kcl": ("KCl", "rocksalt", dict(a=6.293)), "licl": ("LiCl", "rocksalt", dict(a=5.130)),
    "cscl": ("CsCl", "cesiumchloride", dict(a=4.123)),
    "zns": ("ZnS", "zincblende", dict(a=5.409)), "gaas": ("GaAs", "zincblende", dict(a=5.653)),
    "caf2": ("CaF2", "fluorite", dict(a=5.463)),
    "diamond": ("C", "diamond", dict(a=3.567)), "si": ("Si", "diamond", dict(a=5.431)),
    "cu": ("Cu", "fcc", dict(a=3.615)), "al": ("Al", "fcc", dict(a=4.050)), "au": ("Au", "fcc", dict(a=4.078)),
    "ag": ("Ag", "fcc", dict(a=4.086)), "fe": ("Fe", "bcc", dict(a=2.867)), "na": ("Na", "bcc", dict(a=4.291)),
    "w": ("W", "bcc", dict(a=3.165)), "mg": ("Mg", "hcp", dict(a=3.209, c=5.211)),
    "zn": ("Zn", "hcp", dict(a=2.665, c=4.947)),
}
IONIC = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Cs": 1.74, "Mg": 0.72, "Ca": 1.12, "F": 1.33, "Cl": 1.81,
         "O": 1.40}  # Shannon (1976), Å
ION_CHG = {"Li": "+", "Na": "+", "K": "+", "Cs": "+", "Mg": "2+", "Ca": "2+", "F": "−", "Cl": "−", "O": "2−"}
NONMETALS = set("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe".split())


def crystal(f, name, view="layer", repeat=2, style="ball"):
    """layer: atoms in the z = 0 face (repeat x repeat cells), true positions;
    plan : one unit cell projected along c with fractional heights written beside atoms."""
    from ase.build import bulk, graphene
    from ase.neighborlist import neighbor_list
    from ase.data import atomic_masses, atomic_numbers
    key = name.lower().replace(" ", "")
    if os.path.exists(name):
        from ase.io import read
        at = read(name); src = f"file {os.path.basename(name)}"
    elif key == "graphene":
        at = graphene(a=2.46, vacuum=0); at.cell[2, 2] = 3.35; src = "graphene, a=2.46 Å"
    elif key in CRYSTALS:
        fo, stt, kw = CRYSTALS[key]
        at = bulk(fo, stt, cubic=stt != "hcp", **kw); src = f"{stt}, " + ", ".join(f"{k}={v} Å" for k, v in kw.items())
    else:
        at = bulk(name, cubic=True); src = "ASE reference lattice"
    cell = np.array(at.cell); frac = np.round(at.get_scaled_positions(wrap=True), 6) % 1.0; sym0 = at.get_chemical_symbols()
    elems = sorted(set(sym0), key=lambda e: -IONIC.get(e, 1.0))
    fills = [(PAPER, None), (INK, None), (PAPER, "////"), ("0.6", None)]
    sty = {e: fills[i % 4] for i, e in enumerate(elems)}
    ii, dd = neighbor_list("id", at, 7.0); dmin = dd[ii == 0].min(); cn = int(np.sum(dd[ii == 0] < dmin * 1.05))
    if set(elems) <= set(IONIC) and len(elems) > 1:
        rad = {e: IONIC[e] for e in elems}; rk = "Shannon ionic radii"
    else:
        rad = {e: dmin / 2 for e in elems}; rk = "d_nn/2"
    k = 1.0 if style == "space" else 0.32
    a1, a2 = cell[0, :2], cell[1, :2]
    if view == "plan":
        pts = {}
        for fr, e in zip(frac, sym0):
            for sx in (0, 1):
                for sy in (0, 1):
                    q = fr[:2] + (sx, sy)
                    if np.all(q <= 1 + 1e-6):
                        key2 = (round(q[0], 4), round(q[1], 4), e)
                        z = [round(fr[2], 4)] + ([1.0] if abs(fr[2]) < 1e-4 else [])
                        pts.setdefault(key2, set()).update(z)
        poly = np.array([[0, 0], a1, a1 + a2, a2, [0, 0]])
        f.line(poly, lw=1.2, z=1)
        rs = {e: rad[e] * 0.24 for e in elems}
        for (qx, qy, e), zs in sorted(pts.items(), key=lambda kv: -rs[kv[0][2]]):  # big first, small on top
            p = qx * a1 + qy * a2; fl, ht = sty[e]; k = elems.index(e)
            f.circle(p, rs[e], fill=fl, hatch=ht, lw=1.0, z=5 + k)
            d = np.array([1, 1 if k % 2 == 0 else -1]) / np.sqrt(2)
            f.text(p + d * (max(rs.values()) + 0.12), ", ".join(frac_txt(z) for z in sorted(zs)), size=9,
                   ha="left", va="bottom" if k % 2 == 0 else "top", bg=True)
        lo = np.min(poly, 0); hi = np.max(poly, 0); pad = 0.25 * np.linalg.norm(a1)
        f.dim((0, lo[1] - 0.55 * pad), (a1[0], a1[1] + lo[1] - 0.55 * pad), f"a = {np.linalg.norm(cell[0]) / 10:.4g} nm",
              text_off=(0, -0.3 * pad))
        f.limits((lo[0] - pad, hi[0] + 2.4 * pad), (lo[1] - 1.1 * pad, hi[1] + pad)); xleg = hi[0] + 1.1 * pad
    else:
        n = repeat
        P, S = [], []
        for i in range(-1, n + 1):
            for j in range(-1, n + 1):
                for fr, e in zip(frac, sym0):
                    if abs(fr[2] - frac[:, 2].min()) < 1e-4:  # lowest (001) layer
                        q = fr[:2] + (i, j)
                        if np.all(q >= -1e-6) and np.all(q <= n + 1e-6):
                            P.append(q[0] * a1 + q[1] * a2); S.append(e)
        P = np.array(P)
        for _ in range(3):  # trim dangling edge atoms (hexagonal sheets)
            D = np.linalg.norm(P[:, None] - P[None], axis=2); dl = D[D > 1e-6].min()
            keep = ((D > 1e-6) & (D < dl * 1.05)).sum(1) >= 2
            P = P[keep]; S = [e for e, k in zip(S, keep) if k]
        if style != "space":
            D = np.linalg.norm(P[:, None] - P[None], axis=2); dl = D[D > 1e-6].min()
            for i, j in zip(*np.where(np.triu((D > 1e-6) & (D < dl * 1.05)))):
                f.line([P[i], P[j]], lw=0.9, z=1)
        for p, e in zip(P, S):
            fl, ht = sty[e]; f.circle(p, rad[e] * k, fill=fl, hatch=ht, lw=1.0)
        lo, hi = P.min(0), P.max(0); pad = 0.18 * max(np.ptp(P, 0)) + max(rad.values()) * k
        f.dim((0, lo[1] - pad * 0.8), (a1[0], a1[1] + lo[1] - pad * 0.8), f"a = {np.linalg.norm(cell[0]) / 10:.4g} nm",
              text_off=(0, -0.3 * pad))
        f.limits((lo[0] - pad, hi[0] + pad * 2.4), (lo[1] - pad * 1.5, hi[1] + pad)); xleg = hi[0] + 0.9 * pad
    x0 = xleg
    yl = f.lims[1][1]
    for i, e in enumerate(elems):
        fl, ht = sty[e]; yy = yl - (0.12 + 0.13 * i) * np.ptp(f.lims[1])
        f.circle((x0, yy), 0.035 * np.ptp(f.lims[1]), fill=fl, hatch=ht, lw=1.0)
        lab = e + (f"$^{{{ION_CHG[e]}}}$" if set(elems) <= set(IONIC) and len(elems) > 1 else "")
        f.text((x0 + 0.06 * np.ptp(f.lims[1]), yy), lab, size=11, ha="left")
    vol = abs(np.linalg.det(cell)); mass = sum(atomic_masses[atomic_numbers[e]] for e in sym0)
    f.note("crystal", f"{at.get_chemical_formula()} {src}; atoms/cell {len(sym0)}, density "
                      f"{mass * 1.66054 / vol:.4g} g/cm3, nearest neighbour {dmin:.3f} Å, coordination number {cn}; "
                      f"radii: {rk}; view = {view}" + (" (numbers = fractional height z/c)" if view == "plan" else " (lowest (001) layer)"))
    return f


ORB = {"s": {"": 0}, "p": {"z": 0, "x": 1, "y": -1},
       "d": {"z2": 0, "xz": 1, "yz": -1, "x2-y2": 2, "x2y2": 2, "xy": -2},
       "f": {"z3": 0, "xz2": 1, "yz2": -1, "z(x2-y2)": 2, "xyz": -2, "x(x2-3y2)": 3, "y(3x2-y2)": -3}}


def parse_orbital(name):
    m = re.fullmatch(r"(\d)([spdf])_?(.*)", name.strip().lower())
    if not m:
        raise SystemExit(f"bad orbital '{name}' (e.g. 1s 2p_z 3d_z2 3d_xy 4f_xyz)")
    n, L, sub = int(m[1]), "spdf".index(m[2]), m[3]
    table = ORB["spdf"[L]]
    if sub not in table:
        raise SystemExit(f"'{sub}' not valid for {m[2]}; options: {list(table)}")
    if L >= n:
        raise SystemExit(f"l={L} needs n>{L}")
    return n, L, table[sub]


def _nl(name):
    """'3d' or '3d_xy' -> (n, l); the radial function does not depend on m."""
    m = re.fullmatch(r"(\d)([spdf]).*", name.strip().lower())
    if not m or "spdf".index(m[2]) >= int(m[1]):
        raise SystemExit(f"bad orbital '{name}' (e.g. 1s 2s 2p 3d)")
    return int(m[1]), "spdf".index(m[2]), 0


def radial_R(n, l, r, Zn=1):
    from scipy.special import genlaguerre, factorial
    rho = 2 * Zn * r / n
    N = np.sqrt((2 * Zn / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    return N * np.exp(-rho / 2) * rho ** l * genlaguerre(n - l - 1, 2 * l + 1)(rho)


def psi(n, l, m, X, Y, Z, Zn=1):
    """Real hydrogen-like wavefunction (atomic units, r in bohr)."""
    try:
        from scipy.special import sph_harm_y
        Ylm = lambda ll, mm, th, ph: sph_harm_y(ll, mm, th, ph)
    except ImportError:
        from scipy.special import sph_harm
        Ylm = lambda ll, mm, th, ph: sph_harm(mm, ll, ph, th)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2); th = np.arccos(np.clip(Z / np.where(r == 0, 1, r), -1, 1))
    ph = np.arctan2(Y, X)
    R = radial_R(n, l, r, Zn)
    if m == 0:
        Yr = Ylm(l, 0, th, ph).real
    elif m > 0:
        Yr = np.sqrt(2) * (-1) ** m * Ylm(l, m, th, ph).real
    else:
        Yr = np.sqrt(2) * (-1) ** m * Ylm(l, -m, th, ph).imag
    return R * Yr


BOHR = 0.529177210903  # Å


def orbital(f, name="2p_z", Zn=1, prob=0.9, style="boundary", phase="hatch", nodes=True, plane=None, res=90):
    """Cross-section through the nucleus of a hydrogen-like orbital (Å). The boundary curve is
    the section of the 3D surface enclosing `prob` of |psi|^2; dashed lines = nodes."""
    from scipy import ndimage
    n, l, m = parse_orbital(name)
    r = np.linspace(0, 60 * n * n / Zn, 20000); pr = r ** 2 * radial_R(n, l, r, Zn) ** 2
    cdf = np.cumsum(pr); cdf /= cdf[-1]; Rm = r[np.searchsorted(cdf, 0.997)] * 1.02
    t = np.linspace(-Rm, Rm, res)
    X3, Y3, Z3 = np.meshgrid(t, t, t, indexing="ij")
    P3 = psi(n, l, m, X3, Y3, Z3, Zn)
    d = np.sort((P3 ** 2).ravel())[::-1]; c = np.cumsum(d); iso = np.sqrt(d[np.searchsorted(c, prob * c[-1])])
    g = np.linspace(-Rm, Rm, 700); A, B = np.meshgrid(g, g)
    planes = {"xz": (A, 0 * A, B, "x", "z"), "yz": (0 * A, A, B, "y", "z"), "xy": (A, B, 0 * A, "x", "y"),
              "diag": (A / np.sqrt(2), A / np.sqrt(2), B, "x = y", "z")}
    if plane is None:  # the coordinate plane holding most density; diagonal (x = y) if all vanish, e.g. f_xyz
        w = {k: np.sum(psi(n, l, m, *v[:3], Zn) ** 2) for k, v in planes.items()}
        plane = next(k for k in ("xz", "yz", "xy", "diag") if w[k] > 0.98 * max(w.values()))
    Xp, Yp, Zp, hl, vl = planes[plane]
    P = psi(n, l, m, Xp, Yp, Zp, Zn); G = g * BOHR; GA, GB = np.meshgrid(G, G)
    Rb = Rm * BOHR
    if style == "contours":
        mx = np.abs(P).max(); levs = iso * 2.0 ** np.arange(-2, 8); levs = levs[levs < mx]
        f.ax.contour(GA, GB, P, levels=list(levs), colors=INK, linewidths=0.8, linestyles="-")
        f.ax.contour(GA, GB, P, levels=list(-levs[::-1]), colors=INK, linewidths=0.8, linestyles="--")
    elif style == "dots":
        rng = np.random.default_rng(1); pr2 = (P ** 2).ravel(); pr2 /= pr2.sum()
        k = rng.choice(pr2.size, 9000, p=pr2); q = np.c_[GA.ravel()[k], GB.ravel()[k]]
        q += rng.uniform(-0.5, 0.5, q.shape) * (G[1] - G[0])
        f.ax.scatter(q[:, 0], q[:, 1], s=0.6, c=INK, lw=0, zorder=2)
    else:
        for v, sgn in ((iso, 1), (-iso, -1)):
            if (sgn * P > iso).any():
                hatch = (phase == "hatch" and sgn < 0)
                fill = "0.75" if (phase == "shade" and sgn > 0) else (PAPER if hatch else "none")
                f.ax.contourf(GA, GB, sgn * P, levels=[iso, np.inf], colors=[fill if fill != "none" else (1, 1, 1, 0)],
                              hatches=["////"] if hatch else [None], zorder=1)
                f.ax.contour(GA, GB, P, levels=[v], colors=INK, linewidths=1.3, zorder=3, linestyles="solid")
    if style != "dots":
        for sgn in (1, -1):
            lab, nreg = ndimage.label(sgn * P > (iso if style != "contours" else 0.3 * np.abs(P).max()))
            for k in range(1, nreg + 1):
                reg = lab == k
                if reg.sum() < 30:
                    continue
                cy, cx = ndimage.center_of_mass(reg)
                iy, ix = (int(round(cy)), int(round(cx)))
                if not reg[iy, ix]:
                    iy, ix = np.unravel_index(np.argmax(np.where(reg, np.abs(P), 0)), P.shape)
                f.text((G[ix], G[iy]), "+" if sgn > 0 else "−", size=15, bg=True)
    if nodes and (l > 0 or n > 1):
        f.ax.contour(GA, GB, P, levels=[0], colors=INK, linewidths=0.6, linestyles=[(0, (4, 3))], zorder=2)
    L = Rb * 1.05
    f.arrow((-L, 0), (2 * L, 0), lw=0.8, size=9); f.arrow((0, -L), (0, 2 * L), lw=0.8, size=9)
    f.text((L * 1.04, 0), hl, size=13, ha="left", style="italic"); f.text((0, L * 1.04), vl, size=13, va="bottom", style="italic")
    f.dot((0, 0), 0.012 * Rb)
    f.limits((-1.15 * Rb, 1.2 * Rb), (-1.12 * Rb, 1.2 * Rb))
    rmean = (3 * n * n - l * (l + 1)) / (2 * Zn) * BOHR
    f.note("orbital", f"{name}: n={n}, l={l}, m(real)={m}, Z={Zn}; {plane}-plane section through nucleus; radial "
                      f"nodes {n - l - 1}, angular nodes {l}; <r> = {rmean:.3f} Å; boundary encloses {prob:.0%} of "
                      f"|ψ|² in 3D; +/− = phase of ψ" + ("; hatched = negative phase" if phase == "hatch" and style == "boundary" else "")
                      + ("; dashed = nodes" if nodes else ""))
    return f


def radial(f, names, Zn=1, func="P", unit="A"):
    """Radial distribution P(r) = r²R²(r) (or R(r)) for several orbitals on one exam-style graph."""
    from scipy.optimize import brentq
    nmax = max(_nl(s)[0] for s in names)
    rmax = (nmax ** 2 * 3.2 + 6) / Zn
    r = np.linspace(1e-6, rmax, 4000); sc = BOHR if unit == "A" else 1.0
    ys = []
    for i, s in enumerate(names):
        n, l, _ = _nl(s); R = radial_R(n, l, r, Zn)
        y = r ** 2 * R ** 2 if func == "P" else R
        ys.append((s, n, l, y / np.abs(y).max()))
    f.plot_axes((0, rmax * sc), (-1.1 if func == "R" and any(y.min() < -1e-3 for *_, y in ys) else 0, 1.2),
                "r / Å" if unit == "A" else "r / a$_0$", "r$^2$R$^2$(r)" if func == "P" else "R(r)", ticks=True)
    f.ax.set_yticks([])
    for i, (s, n, l, y) in enumerate(ys):
        f.ax.plot(r * sc, y, color=INK, lw=1.4, ls=STYLES[i % 4])
        j = np.argmax(y); f.ax.text(r[j] * sc, y[j] + 0.04, s, ha="center", va="bottom", fontsize=11)
        Rr = radial_R(n, l, r, Zn); sgn = np.where(np.diff(np.sign(Rr[5:])) != 0)[0] + 5
        nodes = [brentq(lambda x: radial_R(n, l, x, Zn), r[k], r[k + 1]) for k in sgn]
        f.note(s, f"most probable r = {r[np.argmax(r ** 2 * Rr ** 2)] * sc:.4g} {'Å' if unit == 'A' else 'a0'}; radial nodes at "
                  + (", ".join(f"{x * sc:.4g}" for x in nodes) or "none"))
    return f


# =============================================================== CLI
def _common(sp):
    sp.add_argument("-o", "--out", default=None, help="output .png (default <command>.png)")
    sp.add_argument("--title", help="title above (exam figures normally have none)")
    sp.add_argument("--fig", help="figure number under the diagram, e.g. 'Fig. 1.1'")
    sp.add_argument("--caption", help="text after the figure number")
    sp.add_argument("--size", default="6x4.5", help="W x H in inches (default 6x4.5)")
    sp.add_argument("--font", choices=["serif", "sans"], default="serif")
    sp.add_argument("--dpi", type=int, default=300)
    sp.add_argument("--svg", action="store_true", help="also write vector .svg")
    sp.add_argument("--pdf", action="store_true", help="also write vector .pdf")


def _floats(t):
    return [float(v) for v in t.replace(" ", "").split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exam2d", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("efield", help="point charges: field lines + equipotentials")
    a.add_argument("charges", nargs="+", help="q@x,y  e.g. +1@-1,0 -1@1,0")
    a.add_argument("--lines", type=int, default=12, help="lines per unit charge"); a.add_argument("--no-equi", action="store_true")
    a = sub.add_parser("plates", help="parallel-plate capacitor field (Laplace solve, fringing)")
    a.add_argument("--V", type=float, default=100); a.add_argument("--d", type=float, default=0.02)
    a.add_argument("--L", type=float, default=0.06); a.add_argument("--lines", type=int, default=24)
    a.add_argument("--no-equi", action="store_true")
    a = sub.add_parser("bfield", help="magnetic field: wire | wires | loop | solenoid | helmholtz | antihelmholtz | magnet")
    a.add_argument("kind", choices=["wire", "wires", "loop", "solenoid", "helmholtz", "antihelmholtz", "magnet"])
    a.add_argument("--I", type=float, default=1.0); a.add_argument("--R", type=float, default=None, help="coil radius / magnet half-width (default 1, magnet 0.5)")
    a.add_argument("--length", type=float, default=4.0); a.add_argument("--turns", type=int, default=10)
    a.add_argument("--lines", type=int, default=7); a.add_argument("--into", action="store_true", help="wire: current into page")
    a.add_argument("--wires", nargs="+", help="wires: I@x,y ... e.g. 1@-1,0 -1@1,0 (I>0 out of page)")
    a = sub.add_parser("rays", help="ray diagram: convex-lens | concave-lens | concave-mirror | convex-mirror")
    a.add_argument("kind", choices=["convex-lens", "concave-lens", "concave-mirror", "convex-mirror"])
    a.add_argument("--f", type=float, default=10); a.add_argument("--u", type=float, default=30)
    a.add_argument("--h", type=float, default=4)
    a = sub.add_parser("wave", help="transverse | longitudinal | standing | interference")
    a.add_argument("kind", choices=["transverse", "longitudinal", "standing", "interference"])
    a.add_argument("--A", type=float, default=None); a.add_argument("--wavelength", type=float, default=None)
    a.add_argument("--length", type=float, default=None); a.add_argument("--time", action="store_true", help="transverse: x-axis is time (wavelength = period)")
    a.add_argument("--xlabel"); a.add_argument("--ylabel"); a.add_argument("--grid", action="store_true")
    a.add_argument("--harmonic", type=int, default=3)
    a.add_argument("--ends", choices=["fixed-fixed", "open-open", "closed-open"], default="fixed-fixed")
    a.add_argument("--speed", type=float); a.add_argument("--d", type=float, default=2.0)
    a.add_argument("--extent", type=float, default=6.0); a.add_argument("--nodal", action="store_true")
    a = sub.add_parser("orbit", help="Kepler orbit with equal-area sectors")
    a.add_argument("--a", type=float, default=1.0); a.add_argument("--e", type=float, default=0.6)
    a.add_argument("--M", type=float, default=1.0); a.add_argument("--marks", type=int, default=12)
    a = sub.add_parser("projectile", help="projectile with optional quadratic drag")
    a.add_argument("--v0", type=float, default=30); a.add_argument("--angle", type=float, default=45)
    a.add_argument("--k", type=float, default=0.0, help="drag rho*Cd*A/(2m) [1/m]"); a.add_argument("--g", type=float, default=9.81)
    a = sub.add_parser("graph", help="exam-style graph of y = f(x) (numpy expressions)")
    a.add_argument("exprs", nargs="+"); a.add_argument("--x", default="0,10"); a.add_argument("--y")
    a.add_argument("--xlabel", default="x"); a.add_argument("--ylabel", default="y"); a.add_argument("--labels")
    a.add_argument("--grid", action="store_true"); a.add_argument("--no-ticks", action="store_true")
    a = sub.add_parser("contour", help="contour map of z = f(x, y)")
    a.add_argument("expr"); a.add_argument("--x", default="-3,3"); a.add_argument("--y", default="-3,3")
    a.add_argument("--levels", type=int, default=12)
    a = sub.add_parser("vfield", help="2D vector field F = (fx, fy)")
    a.add_argument("fx"); a.add_argument("fy"); a.add_argument("--extent", type=float, default=2.0)
    a.add_argument("--n", type=int, default=13); a.add_argument("--stream", action="store_true", help="add streamlines")
    a = sub.add_parser("molecule", help="skeletal | displayed formula, VSEPR shape, dot-and-cross")
    a.add_argument("spec"); a.add_argument("--style", choices=["skeletal", "displayed", "shape", "dotcross"], default="skeletal")
    a.add_argument("--no-lp", action="store_true", help="shape: hide lone pairs"); a.add_argument("--angle", help="shape: override angle text")
    a.add_argument("--label-c", action="store_true", help="skeletal: write C on every carbon")
    a = sub.add_parser("crystal", help="crystal: z=0 layer or plan view with fractional heights")
    a.add_argument("name"); a.add_argument("--view", choices=["layer", "plan"], default="layer")
    a.add_argument("--repeat", type=int, default=2); a.add_argument("--style", choices=["ball", "space"], default="ball")
    a = sub.add_parser("orbital", help="hydrogen-like orbital cross-section, e.g. 2p_z 3d_xy 3s")
    a.add_argument("name"); a.add_argument("--Z", type=int, default=1); a.add_argument("--prob", type=float, default=0.9)
    a.add_argument("--style", choices=["boundary", "contours", "dots"], default="boundary")
    a.add_argument("--phase", choices=["hatch", "shade", "none"], default="hatch")
    a.add_argument("--plane", choices=["xz", "yz", "xy", "diag"]); a.add_argument("--no-nodes", action="store_true")
    a = sub.add_parser("radial", help="radial distribution graph, e.g. radial 1s 2s 3s")
    a.add_argument("names", nargs="+"); a.add_argument("--Z", type=int, default=1)
    a.add_argument("--func", choices=["P", "R"], default="P"); a.add_argument("--unit", choices=["A", "a0"], default="A")
    sub.add_parser("list", help="list built-in molecules, crystals, orbitals")
    for k, v in sub.choices.items():
        if k != "list":
            _common(v)
    argv = sys.argv[1:] if argv is None else argv
    flags = {o for p in [ap, *sub.choices.values()] for x in p._actions for o in x.option_strings}
    argv = [" " + t if t.startswith("-") and t.split("=")[0] not in flags else t for t in argv]
    A = ap.parse_args(argv)
    if A.cmd == "list":
        print("molecules:", ", ".join(sorted(COMMON)), "\n  + any SMILES ('smiles:...') or .mol file")
        print("crystals:", ", ".join(sorted(CRYSTALS)), ", graphene\n  + any element symbol or .cif file")
        print("orbitals:", "; ".join(f"{k}: {', '.join(x or '-' for x in v)}" for k, v in ORB.items()))
        return
    W, H = map(float, A.size.lower().split("x"))
    f = Fig(A.title, A.caption, A.fig, (W, H), A.font)
    c = A.cmd; out = A.out or f"{c}.png"
    if c == "efield":
        ch = []
        for t in A.charges:
            q, p = t.strip().split("@"); ch.append((float(q), _floats(p)))
        efield(f, ch, A.lines, not A.no_equi)
    elif c == "plates":
        plates(f, A.V, A.d, A.L, A.lines, not A.no_equi)
    elif c == "bfield":
        wires = None
        if A.wires:
            wires = [(tuple(_floats(t.strip().split("@")[1])), float(t.strip().split("@")[0])) for t in A.wires]
        elif A.kind == "wires":
            wires = [((-1.0, 0.0), A.I), ((1.0, 0.0), -A.I)]
        bfield(f, A.kind, -A.I if A.into else A.I, A.R or (0.5 if A.kind == "magnet" else 1.0), A.length, A.turns, wires, A.lines)
    elif c == "rays":
        rays(f, A.kind, A.f, A.u, A.h)
    elif c == "wave":
        if A.kind == "transverse":
            wave_transverse(f, A.A or 2.0, A.wavelength or 4.0, A.length or 10.0,
                            xlabel=A.xlabel or ("time / s" if A.time else "distance / m"),
                            ylabel=A.ylabel or "displacement / cm", grid=A.grid, time=A.time)
        elif A.kind == "longitudinal":
            wave_longitudinal(f, A.A or 0.35, A.wavelength or 4.0, A.length or 12.0)
        elif A.kind == "standing":
            wave_standing(f, A.harmonic, A.ends, A.length or 1.0, A.speed)
        else:
            interference(f, A.d, A.wavelength or 0.6, A.extent, A.nodal)
    elif c == "orbit":
        kepler(f, A.a, A.e, A.M, A.marks)
    elif c == "projectile":
        projectile(f, A.v0, A.angle, A.k, A.g)
    elif c == "graph":
        graph(f, A.exprs, _floats(A.x), _floats(A.y) if A.y else None, A.xlabel, A.ylabel,
              A.labels.split(",") if A.labels else None, A.grid, not A.no_ticks)
    elif c == "contour":
        contour(f, A.expr, _floats(A.x), _floats(A.y), A.levels)
    elif c == "vfield":
        vfield(f, A.fx.strip(), A.fy.strip(), A.extent, A.n, A.stream)
    elif c == "molecule":
        if A.style in ("skeletal", "displayed"):
            structure(f, A.spec, A.style, svg_path=os.path.splitext(out)[0] + ".svg" if A.svg else None, label_c=A.label_c)
            A.svg = False
        elif A.style == "shape":
            shape(f, A.spec, not A.no_lp, A.angle)
        else:
            dotcross(f, A.spec)
    elif c == "crystal":
        crystal(f, A.name, A.view, A.repeat, A.style)
    elif c == "orbital":
        orbital(f, A.name, A.Z, A.prob, A.style, A.phase, not A.no_nodes, A.plane)
    elif c == "radial":
        radial(f, A.names, A.Z, A.func, A.unit)
    f.save(out, A.svg, A.pdf, A.dpi)


if __name__ == "__main__":
    main()
