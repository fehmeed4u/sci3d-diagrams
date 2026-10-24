"""A.4 Rigid body mechanics: IB-style 2D figures (1800 x 1280 transparent PNG).

python make_figures.py            -> fig-*.png (black ink) and dark/fig-*.png (white ink for #141d35 slides)
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "plugins", "sci3d-diagrams", "skills", "sci3d-diagrams", "scripts"))
import exam2d as X
from exam2d import Fig

PX = (1800, 1280)
LIM = ((0, 9.0), (0, 6.4))          # 900:640 drawing units
FS, FSS = 17, 14                    # label / small label sizes (pt at 200 dpi)


def new(dark):
    f = Fig(font="sans", px=PX, transparent=True, ink="white" if dark else "black",
            paper="#141d35" if dark else "white")
    f.limits(*LIM)
    return f


def T(f, xy, s, size=FS, **kw):
    return f.text(xy, s, size=size, **kw)


def u(a):
    a = np.radians(a)
    return np.array([np.cos(a), np.sin(a)])


def dimline(f, p1, p2, s, side=1, size=FSS, gap=0.22):
    """Double-headed dimension arrow with the value written midway (beside the line)."""
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    f.arrow(p1, p2 - p1, lw=1.0, size=12, both=True)
    d = (p2 - p1) / np.hypot(*(p2 - p1)); n = np.array([-d[1], d[0]]) * side
    T(f, (p1 + p2) / 2 + n * gap, s, size=size, bg=True)


def ext(f, p, q):
    f.line([p, q], lw=0.6, ls=(0, (2, 2)))


# ------------------------------------------------------------------ Lesson 1
def fig_1a(f):
    O = np.array([1.9, 2.6]); P = np.array([6.9, 2.6]); th = 55
    # spanner: ring head round a hexagonal nut, handle to P
    f.line([O + [0.75, 0.24], P + [0.25, 0.24]], lw=1.6); f.line([O + [0.75, -0.24], P + [0.25, -0.24]], lw=1.6)
    f.ax.add_patch(X.Arc(P + [0.25, 0], 0.48, 0.48, theta1=-90, theta2=90, color=X.INK, lw=1.6))
    f.circle(O, 0.8, lw=1.6, z=3)
    hexa = O + 0.42 * np.array([u(30 + 60 * k) for k in range(6)])
    f.poly(hexa, lw=1.3, z=4)
    f.circle(O, 0.13, lw=1.0, z=5)
    f.dot(O, 0.045)
    T(f, O + [-0.15, -1.05], "pivot (nut centre)", size=FSS)
    # r along the handle
    dimline(f, O + [0, 0.95], P + [0, 0.95], "$r$", size=FS)
    ext(f, O + [0, 0.15], O + [0, 1.1]); ext(f, P + [0, 0.3], P + [0, 1.1])
    # force at P at angle theta to the handle
    d = u(th); f.arrow(P, d * 1.7, lw=2.2, size=16)
    T(f, P + d * 1.95, "$F$", size=FS + 2)
    f.line([P, P + [1.15, 0]], lw=0.8, ls=(0, (5, 3)))                       # handle direction continued
    f.angle(P, 0, th, 0.6, "$\\theta$", lr=1.55, size=FS)
    # line of action extended back and perpendicular from pivot
    foot = P + d * np.dot(O - P, d)
    f.line([foot - d * 0.6, P], lw=1.0, ls=(0, (6, 4)))
    T(f, P - d * 1.6 + [0.3, -0.05], "line of action", size=FSS, ha="left", rotation=th)
    f.line([O, foot], lw=1.2)
    f.right_angle(foot, O - foot, -d, 0.2)
    m = (O + foot) / 2
    T(f, m + [-0.3, -0.25], "$r_\\perp$", size=FS)
    T(f, (7.9, 1.3), "$r_\\perp = r\\,\\sin\\theta$", size=FS)
    # inset: couple on a steering wheel
    bx, by, bw, bh = 0.15, 4.25, 3.0, 2.05
    f.line([(bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh), (bx, by)], lw=0.8)
    C = np.array([1.25, 5.25]); r = 0.68
    f.circle(C, r, lw=1.6, z=3); f.circle(C, 0.12, lw=1.0, z=4)
    for a in (90, 210, 330):
        f.line([C + u(a) * 0.12, C + u(a) * r], lw=1.0)
    f.arrow(C + [0.15, r], (-0.95, 0), lw=1.8, size=13); T(f, C + [-0.55, r + 0.24], "$F$", size=FSS + 1)
    f.arrow(C + [-0.15, -r], (0.95, 0), lw=1.8, size=13); T(f, C + [0.55, -r - 0.24], "$F$", size=FSS + 1)
    x2 = C[0] + 1.25
    ext(f, C + [0.1, r], (x2 + 0.1, C[1] + r)); ext(f, C + [0.1, -r], (x2 + 0.1, C[1] - r))
    dimline(f, (x2, C[1] - r), (x2, C[1] + r), "$2r$", side=-1, size=FSS, gap=0.3)
    T(f, (bx + bw / 2 + 0.25, by + bh - 0.2), "couple", size=FSS - 1)


def fig_1b(f):
    s = 0.9; xw, yb = 1.3, 2.6; L = 6.0 * s; t = 0.22
    f.surface((xw, 0.25), (xw, 6.3), side=1)                                   # wall, hatched on the left
    f.poly([(xw, yb - t / 2), (xw + L, yb - t / 2), (xw + L, yb + t / 2), (xw, yb + t / 2)], lw=1.4)
    f.circle((xw, yb), 0.1, lw=1.2, z=6); f.dot((xw, yb), 0.03)              # hinge pin
    T(f, (xw + 0.75, yb + 0.33), "hinge", size=FSS, ha="left")
    E = np.array([xw + L, yb + t / 2]); W = np.array([xw, E[1] + L * np.tan(np.radians(30))])
    f.line([E, W], lw=1.4); f.dot(W, 0.05)
    f.angle(E, 150, 180, 0.9, "30°", lr=1.4, size=FSS)
    dc = (W - E) / np.hypot(*(W - E))
    f.arrow(E + dc * 0.15, dc * 1.4, lw=2.0, size=15)
    T(f, E + dc * 1.2 + [0.25, 0.3], "$T$", size=FS + 2)
    T(f, (3.95, yb + 0.45), "uniform beam, 40 kg", size=FSS)
    # beam weight from the midpoint
    M = np.array([xw + 3.0 * s, yb - t / 2])
    f.dot(M + [0, t / 2], 0.05)
    f.arrow(M, (0, -0.8), lw=2.0, size=15); T(f, M + [0.2, -0.6], "$W$", size=FS, ha="left")
    # hanging 20 kg load at 4.5 m
    xs = xw + 4.5 * s
    f.line([(xs, yb - t / 2), (xs, 1.2)], lw=1.0)
    f.poly([(xs - 0.42, 0.62), (xs + 0.42, 0.62), (xs + 0.42, 1.2), (xs - 0.42, 1.2)], lw=1.4)
    T(f, (xs, 0.91), "20 kg", size=FSS)
    # unknown hinge reaction
    f.ax.annotate("", xy=(xw, yb) + u(62) * 1.25, xytext=(xw, yb), zorder=5, arrowprops=dict(
        arrowstyle="-|>,head_length=0.5,head_width=0.2", color=X.INK, lw=2.0, ls="--", mutation_scale=15,
        shrinkA=0, shrinkB=0))
    T(f, (xw, yb) + u(62) * 1.5 + [0.1, 0.05], "$R$ (?)", size=FS)
    # dimensions from the hinge
    ext(f, (xw, yb - t / 2), (xw, 0.2)); ext(f, (xs, 1.2), (xs, 1.45)); ext(f, (xw + L, yb - t / 2), (xw + L, 0.2))
    dimline(f, (xw, 1.45), (xs, 1.45), "4.5 m", side=1, gap=0.0)
    dimline(f, (xw, 0.35), (xw + L, 0.35), "6.0 m", side=1, gap=0.0)
    dimline(f, (xw, 2.05), (xw + 3.0 * s, 2.05), "3.0 m", side=1, gap=0.0)


# ------------------------------------------------------------------ Lesson 2
def fig_2a(f):
    O = np.array([3.3, 3.2]); R = 2.7; r = 1.15
    f.circle(O, R, lw=1.8, z=1); f.dot(O, 0.06); T(f, O + [-0.28, -0.28], "O", size=FS)
    f.rot_arrow(O, R + 0.32, 100, 165, lw=1.6, size=15)
    T(f, O + u(132) * (R + 0.75), "$\\omega$", size=FS + 3)
    f.line([O, O + [R, 0]], lw=0.9, ls=(0, (6, 4)))
    P, Q = O + [r, 0], O + [2 * r, 0]
    for p, n in ((P, "P"), (Q, "Q")):
        f.dot(p, 0.07); T(f, p + [0.22, -0.3], n, size=FS)
    k = 0.95
    f.arrow(P, (0, k), lw=2.0, size=15); T(f, P + [-0.3, k * 0.75], "$v$", size=FS)
    f.arrow(Q, (0, 2 * k), lw=2.0, size=15); T(f, Q + [-0.34, 2 * k * 0.8], "$2v$", size=FS)
    f.arrow(Q + [-0.1, 0], (-0.7, 0), lw=2.0, size=15)
    T(f, Q + [-0.45, 0.32], "$a$", size=FS)
    dimline(f, O + [0, -0.7], P + [0, -0.7], "$r$", side=-1, gap=0.28)
    dimline(f, O + [0, -1.45], Q + [0, -1.45], "$2r$", side=-1, gap=0.28)
    ext(f, P, P + [0, -0.85]); ext(f, Q, Q + [0, -1.6]); ext(f, O, O + [0, -1.6])
    T(f, (7.65, 4.7), "$v = \\omega r$", size=FS)
    T(f, (7.65, 3.9), "$a = \\omega^2 r$", size=FS)
    T(f, (7.65, 3.25), "(at Q:  $a = \\omega^2\\,(2r)$)", size=FSS - 1)


def fig_2b(f):
    C = np.array([2.35, 3.2]); R = 1.85
    f.circle(C, R, lw=2.2, z=2); f.circle(C, R - 0.22, lw=1.0, z=2)
    for a in range(0, 360, 45):
        f.line([C + u(a) * 0.25, C + u(a) * (R - 0.22)], lw=1.0)
    f.circle(C, 0.25, lw=1.2, z=3); f.dot(C, 0.05)
    # brake pad pressing on the rim (right side) with its arm
    pa = C + [R, 0]
    f.poly([pa + [0, -0.45], pa + [0.28, -0.45], pa + [0.28, 0.45], pa + [0, 0.45]], fill=X.INK)
    f.line([pa + [0.28, 0], pa + [0.95, 0]], lw=2.0); f.circle(pa + [0.95, 0], 0.07, lw=1.2, z=6)
    T(f, pa + [0.55, 0.75], "brake pad", size=FSS)
    f.rot_arrow(C, R + 0.32, 150, 60, lw=1.6, size=15)          # clockwise
    T(f, C + u(105) * (R + 0.75), "1200 rpm", size=FS)
    # omega-t sketch
    o = np.array([5.3, 1.25]); w, h = 3.0, 3.7
    f.arrow(o, (w + 0.3, 0), lw=1.3, size=12); f.arrow(o, (0, h + 0.4), lw=1.3, size=12)
    T(f, o + [w + 0.55, -0.02], "$t$", size=FS); T(f, o + [0, h + 0.65], "$\\omega$", size=FS)
    T(f, o + [-0.25, -0.25], "0", size=FSS)
    A, B = o + [0, h * 0.85], o + [w * 0.85, 0]
    f.poly([o, A, B], hatch="///", lw=0, z=1)
    f.line([A, B], lw=2.0)
    T(f, A + [-0.38, 0], "$\\omega_0$", size=FS)
    f.line([B, B + [0, -0.12]], lw=1.0); T(f, B + [0, -0.38], "$t$ = ?", size=FSS)
    T(f, o + [w * 0.25, h * 0.25], "25 rev", size=FS, bg=True)


# ------------------------------------------------------------------ Lesson 3
def fig_3a(f):
    cx = [1.15, 3.35, 5.6, 7.85]; y = 3.6; Lr = 1.8
    labels = ["rod about centre", "rod about end", "disc, central axis", "solid sphere"]
    formulas = ["$I = \\frac{1}{12}\\,mL^2$", "$I = \\frac{1}{3}\\,mL^2$", "$I = \\frac{1}{2}\\,mr^2$",
                "$I = \\frac{2}{5}\\,mr^2$"]
    axis = dict(lw=1.0, ls=(0, (7, 3, 1.5, 3)))
    # rod about centre
    x = cx[0]
    f.poly([(x - Lr / 2, y - 0.07), (x + Lr / 2, y - 0.07), (x + Lr / 2, y + 0.07), (x - Lr / 2, y + 0.07)], lw=1.4)
    f.line([(x, y - 1.15), (x, y + 1.15)], **axis)
    dimline(f, (x - Lr / 2, y - 0.5), (x + Lr / 2, y - 0.5), "$L$", gap=0.0)
    # rod about end
    x = cx[1]; x0 = x - Lr / 2
    f.poly([(x0, y - 0.07), (x0 + Lr, y - 0.07), (x0 + Lr, y + 0.07), (x0, y + 0.07)], lw=1.4)
    f.line([(x0, y - 1.15), (x0, y + 1.15)], **axis)
    dimline(f, (x0, y - 0.5), (x0 + Lr, y - 0.5), "$L$", gap=0.0)
    # disc about its central axis (drawn edge-on in perspective)
    x = cx[2]; rx, ry = 0.95, 0.3
    f.ax.add_patch(X.Ellipse((x, y - 0.09), 2 * rx, 2 * ry, fc=X.PAPER, ec=X.INK, lw=1.4, zorder=2))
    f.ax.add_patch(X.Rectangle((x - rx, y - 0.09), 2 * rx, 0.18, fc=X.PAPER, ec="none", zorder=2))
    f.line([(x - rx, y - 0.09), (x - rx, y + 0.09)], lw=1.4, z=3); f.line([(x + rx, y - 0.09), (x + rx, y + 0.09)], lw=1.4, z=3)
    f.ax.add_patch(X.Ellipse((x, y + 0.09), 2 * rx, 2 * ry, fc=X.PAPER, ec=X.INK, lw=1.4, zorder=3))
    f.line([(x, y - 1.15), (x, y + 1.15)], z=4, **axis)
    f.line([(x, y + 0.09), (x + rx, y + 0.09)], lw=1.0, z=4); T(f, (x + rx / 2, y + 0.3), "$r$", size=FS)
    # solid sphere
    x = cx[3]; rs = 0.85
    f.circle((x, y), rs, lw=1.4, z=2)
    f.ax.add_patch(X.Arc((x, y), 2 * rs, 0.5, theta1=180, theta2=360, color=X.INK, lw=0.8, zorder=3))
    f.line([(x, y - 1.15), (x, y + 1.15)], z=4, **axis)
    f.line([(x, y), (x + rs * np.cos(np.radians(35)), y + rs * np.sin(np.radians(35)))], lw=1.0, z=4)
    T(f, (x + 0.42, y + 0.5), "$r$", size=FS)
    for x, lab, fo in zip(cx, labels, formulas):
        T(f, (x, 1.7), lab, size=FSS)
        T(f, (x, 1.0), fo, size=FS + 2)
    T(f, (4.5, 5.6), "dashed line = rotation axis", size=FSS - 1)


def fig_3b(f):
    k = 4.6                                     # drawing units per metre
    xc = 4.75
    for (y, xm, lab, dl) in ((4.55, 0.60, "(a)", "0.60 m"), (1.55, 0.30, "(b)", "0.30 m")):
        T(f, (0.45, y + 0.9), lab, size=FS + 1)
        x0, x1 = xc - 0.6 * k, xc + 0.6 * k
        f.poly([(x0, y - 0.06), (x1, y - 0.06), (x1, y + 0.06), (x0, y + 0.06)], lw=1.4)
        f.line([(xc, y - 1.0), (xc, y + 1.05)], lw=1.0, ls=(0, (7, 3, 1.5, 3)))
        f.rot_arrow((xc, y + 0.8), 0.32, 200, 340, lw=1.1, size=11, ry=0.1)
        for s in (-1, 1):
            p = (xc + s * xm * k, y)
            f.circle(p, 0.2, fill=X.INK, lw=0, z=6)
            T(f, (p[0], y + 0.5), "0.20 kg", size=FSS)
            dimline(f, (xc, y - 0.55), (p[0], y - 0.55), dl, gap=0.0)
            ext(f, p, (p[0], y - 0.7))
        T(f, (x1 + 0.25, y), "rod 1.2 m,\n0.60 kg", size=FSS, ha="left")


# ------------------------------------------------------------------ Lesson 4
def fig_4a(f):
    yg = 0.75; R = 2.0; C = np.array([3.0, yg + R])
    f.surface((0.3, yg), (8.7, yg), side=-1)
    f.circle(C, R, lw=2.0, z=2); f.dot(C, 0.06)
    k = 1.75
    top, bot = C + [0, R], C - [0, R]
    f.arrow(top, (2 * k, 0), lw=2.2, size=16); T(f, top + [2 * k + 0.4, 0], "$2v$", size=FS)
    f.arrow(C, (k, 0), lw=2.2, size=16); T(f, C + [k + 0.35, 0], "$v$", size=FS)
    f.dot(bot, 0.08); T(f, bot + [-0.5, 0.32], "$v = 0$", size=FSS)
    end = bot + (top + [2 * k, 0] - bot) * 1.12
    f.line([bot, end], lw=1.0, ls=(0, (6, 4)))
    f.line([C, top], lw=0.6, ls=(0, (2, 2))); f.line([C, bot], lw=0.6, ls=(0, (2, 2)))
    f.rot_arrow(C, 0.8, 150, 30, lw=1.6, size=14)          # clockwise
    T(f, C + [-0.05, 1.12], "$\\omega$", size=FS + 2)
    f.arrow((6.4, 1.3), (1.4, 0), lw=1.2, size=12); T(f, (7.1, 1.6), "rolling", size=FSS)


def fig_4b(f):
    a = np.radians(45); H = 3.2; B = H / np.tan(a); yb = 0.9
    d = np.array([np.cos(a), -np.sin(a)]); nrm = np.array([np.sin(a), np.cos(a)])
    # (a)/(b) rolling cylinder; slope 1 has its vertical face at x0
    x0 = 1.55; top, foot = np.array([x0, yb + H]), np.array([x0 + B, yb])
    f.poly([(x0, yb), foot, top], lw=1.5, z=1)
    f.surface((0.3, yb), (x0 + B + 0.3, yb), side=-1, lw=1.5)
    Rc = 0.42
    c1 = top + d * 0.75 + nrm * Rc; c2 = foot - d * 0.75 + nrm * Rc
    f.circle(c1, Rc, lw=1.8, z=4); f.dot(c1, 0.04)
    f.circle(c2, Rc, lw=1.2, z=4, ls=(0, (4, 3))); f.dot(c2, 0.03)
    f.arrow(c2 + d * 0.5, d * 0.55, lw=1.8, size=14)
    T(f, c2 + [-0.95, -0.5], "$v$ = ?", size=FSS, bg=True)
    f.rot_arrow(c2, 0.68, 165, 55, lw=1.3, size=12)
    T(f, c2 + [-0.6, 0.95], "$\\omega$ = ?", size=FSS)
    T(f, c1 + [0.55, 0.45], "solid cylinder,\nreleased from rest", size=FSS - 1, ha="left")
    xd = 1.2
    ext(f, c1, (xd - 0.12, c1[1])); ext(f, c2, (xd - 0.12, c2[1]))
    dimline(f, (xd, c2[1]), (xd, c1[1]), "", side=1)
    T(f, (xd - 0.12, (c1[1] + c2[1]) / 2), "$\\Delta h$ =\n0.90 m", size=FSS, ha="right")
    # (c) identical frictionless slope with a sliding block
    x0 = 5.35; top, foot = np.array([x0, yb + H]), np.array([x0 + B, yb])
    f.poly([(x0, yb), foot, top], lw=1.5, z=1)
    f.surface((x0 - 0.1, yb), (x0 + B + 0.3, yb), side=-1, lw=1.5)
    s_ = 0.6

    def block(base, dashed=False):
        pts = [base, base + d * s_, base + d * s_ + nrm * s_, base + nrm * s_]
        f.poly(pts, lw=1.2 if dashed else 1.8, dashed=dashed, z=4)
        return base + d * s_ / 2 + nrm * s_ / 2
    block(top + d * 0.45)
    b2 = block(foot - d * 1.05, dashed=True)
    f.arrow(b2 + d * 0.45, d * 0.55, lw=1.8, size=14)
    T(f, b2 + [-0.95, -0.45], "$v$ = ?", size=FSS, bg=True)
    T(f, (x0 + 1.9, yb + H + 0.45), "(c) block on frictionless\nslope (same height)", size=FSS - 1)


# ------------------------------------------------------------------ Lesson 5
def skater(f, c, arms_out):
    c = np.asarray(c, float)
    f.ax.add_patch(X.Ellipse(c, 1.25, 0.55, fc=X.PAPER, ec=X.INK, lw=1.5, zorder=3))   # shoulders (top view)
    f.circle(c + [0, 0.05], 0.27, lw=1.5, z=4)                                         # head
    if arms_out:
        for s in (-1, 1):
            f.line([c + [s * 0.6, 0], c + [s * 1.75, 0.05]], lw=2.4, z=2); f.circle(c + [s * 1.86, 0.05], 0.1, lw=1.2, z=3)
    else:
        for s in (-1, 1):
            f.line([c + [s * 0.55, -0.05], c + [s * 0.5, -0.42], c + [s * 0.12, -0.5]], lw=2.4, z=5)
            f.circle(c + [s * 0.18, -0.5], 0.08, lw=1.2, z=6)


def fig_5a(f):
    A, B = np.array([2.2, 3.4]), np.array([6.95, 3.4])
    skater(f, A, True); skater(f, B, False)
    f.rot_arrow(A, 2.2, 60, 120, lw=1.4, size=12, ry=1.25)
    T(f, A + [0, 1.6], "$\\omega_1$", size=FS + 1)
    f.rot_arrow(B, 1.05, -30, 200, lw=2.0, size=16, ry=0.95)
    T(f, B + [0, 1.3], "$\\omega_2$", size=FS + 3)
    T(f, A + [0, -1.45], "arms out: large $I_1$, small $\\omega_1$", size=FSS)
    T(f, B + [0, -1.45], "arms in: small $I_2$, large $\\omega_2$", size=FSS)
    f.arrow((4.35, 3.4), (1.3, 0), lw=1.8, size=15)
    T(f, (5.0, 3.95), "$I_1\\omega_1 = I_2\\omega_2$", size=FS + 2, bg=True)
    T(f, (4.5, 0.8), "view from above", size=FSS - 1)


def turntable(f, cx, y, R, label, clay=None):
    f.poly([(cx - R, y), (cx + R, y), (cx + R, y + 0.22), (cx - R, y + 0.22)], lw=1.6)
    f.line([(cx, y), (cx, y - 0.9)], lw=2.2); f.surface((cx - 0.7, y - 0.9), (cx + 0.7, y - 0.9), side=-1, lw=1.4)
    f.line([(cx, y + 0.22), (cx, y + 1.5)], lw=0.9, ls=(0, (7, 3, 1.5, 3)))
    f.rot_arrow((cx, y + 1.05), 0.55, 210, 330 + 0, lw=1.5, size=13, ry=0.18)
    T(f, (cx, y + 1.62), label, size=FS)
    dimline(f, (cx, y - 0.35), (cx + R, y - 0.35), "0.50 m", gap=0.0)
    if clay is not None:
        f.ax.add_patch(X.Ellipse(clay, 0.36, 0.26, fc=X.INK, ec=X.INK, zorder=6))


def fig_5b(f):
    R = 1.55; y = 2.3
    xa, xb = 2.35, 6.75
    turntable(f, xa, y, R, "3.0 rad s$^{-1}$")
    T(f, (xa, y + 0.11), "4.0 kg", size=FSS - 1)
    c0 = np.array([xa + R - 0.15, y + 3.15])
    f.ax.add_patch(X.Ellipse(c0, 0.36, 0.26, fc=X.INK, ec=X.INK, zorder=6))
    f.line([c0 - [0, 0.18], (c0[0], y + 0.3)], lw=1.0, ls=(0, (5, 4)))
    f.head((c0[0], y + 0.45), (0, -1), 13)
    T(f, c0 + [0.25, 0.05], "1.0 kg clay", size=FSS, ha="left")
    turntable(f, xb, y, R, "$\\omega_2$ = ?", clay=(xb + R - 0.15, y + 0.33))
    T(f, (xb - 0.15, y + 0.11), "4.0 kg", size=FSS - 1)
    T(f, (xa, 0.45), "before", size=FS); T(f, (xb, 0.45), "after", size=FS)
    f.arrow((4.15, 3.0), (0.75, 0), lw=1.6, size=14)


FIGS = {"fig-1a-review": fig_1a, "fig-1b-challenge": fig_1b, "fig-2a-review": fig_2a, "fig-2b-challenge": fig_2b,
        "fig-3a-review": fig_3a, "fig-3b-challenge": fig_3b, "fig-4a-review": fig_4a, "fig-4b-challenge": fig_4b,
        "fig-5a-review": fig_5a, "fig-5b-challenge": fig_5b}

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in FIGS.items():
        if only and name not in only:
            continue
        for dark in (False, True):
            f = new(dark); fn(f)
            f.save(os.path.join(HERE, "dark" if dark else "", name + ".png"))
