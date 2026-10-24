"""Thermodynamics topic diagrams (b4-1a … b4-4x), 2D textbook style.

Run:  python make_thermo.py [outdir]
Every curve is computed: isotherms pV = nRT, adiabats pV^γ = const (γ = 5/3).
"""
import sys, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle, FancyBboxPatch, Circle, Polygon, FancyArrowPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "png")
os.makedirs(OUT, exist_ok=True)
R, GAMMA = 8.31, 5 / 3

INK, HOT, COLD, WORK, GAS, SHADE, ISO = "#222222", "#d1495b", "#2e86ab", "#2a9d8f", "#6c5ce7", "#ffd166", "#9a9a9a"
plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "mathtext.fontset": "dejavusans", "savefig.dpi": 200})
rng = np.random.default_rng(4)


# ---------- helpers ----------------------------------------------------------
def save(fig, name, title):
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.savefig(os.path.join(OUT, f"{name}.png"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", name)


def canvas(w=9, h=6, xlim=(0, 10), ylim=(0, 7)):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def arrow(ax, p0, p1, color=INK, lw=2.5, ms=22, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


def wavy(ax, p0, p1, color=HOT, amp=0.12, waves=4, lw=2.5, label=None, loff=(0.25, 0), fs=14):
    """Wavy heat arrow from p0 to p1."""
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0; L = np.hypot(*d); u = d / L; n = np.array([-u[1], u[0]])
    t = np.linspace(0, L * 0.85, 200)
    pts = p0 + np.outer(t, u) + np.outer(amp * np.sin(2 * np.pi * waves * t / (L * 0.85)), n)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=lw, solid_capstyle="round")
    arrow(ax, pts[-1], p1, color=color, lw=lw)
    if label:
        m = p0 + d / 2
        ax.text(m[0] + loff[0], m[1] + loff[1], label, color=color, fontsize=fs, fontweight="bold",
                ha="left" if loff[0] >= 0 else "right", va="center")


def gas_dots(ax, x0, y0, x1, y1, n=40, color=GAS, s=18):
    ax.scatter(rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), s=s, color=color, zorder=5)


def isotherms(ax, V, nRTs, **kw):
    for nRT in nRTs:
        ax.plot(V, nRT / V, ":", color=ISO, lw=1.2, zorder=1, **kw)


def pv_axes(ax, xl="V", yl="p", ticks=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel(xl); ax.set_ylabel(yl)
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, ms=7)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, ms=7)


def mid_arrow(ax, x, y, color, frac=0.5, ms=20):
    i = int(frac * (len(x) - 1))
    arrow(ax, (x[i - 1], y[i - 1]), (x[i + 1], y[i + 1]), color=color, lw=0, ms=ms)


# =========================== TOPIC 1: FIRST LAW ==============================
def b4_1a():
    fig, ax = canvas(10, 6.5, (0, 11), (0, 7))
    # reservoir
    ax.add_patch(FancyBboxPatch((0.6, 0.4), 8.8, 1.3, boxstyle="round,pad=0.05", fc="#f6c1c1", ec=HOT, lw=2))
    ax.text(5, 1.05, "thermal reservoir,  T ≈ constant", ha="center", va="center", fontsize=14, color="#8b1e2d", fontweight="bold")
    # cylinder (open on the right where piston is)
    x0, x1, y0, y1 = 1.5, 7.8, 2.4, 4.9
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#eef3f8", ec="none"))
    ax.plot([x1, x0, x0, x1], [y1, y1, y0, y0], color=INK, lw=4, solid_joinstyle="miter")
    px = 6.2
    ax.add_patch(Rectangle((px, y0 + 0.04), 0.35, y1 - y0 - 0.08, fc="#7f8c8d", ec=INK, lw=1.5, zorder=6))
    ax.add_patch(Rectangle((px + 0.35, (y0 + y1) / 2 - 0.12), 2.3, 0.24, fc="#95a5a6", ec=INK, lw=1.2, zorder=6))
    gas_dots(ax, x0 + 0.25, y0 + 0.25, px - 0.2, y1 - 0.25, n=45)
    # system boundary
    ax.add_patch(Rectangle((x0 + 0.1, y0 + 0.1), px - x0 - 0.15, y1 - y0 - 0.2, fill=False, ec=WORK, lw=2.2, ls=(0, (6, 4)), zorder=7))
    ax.text((x0 + px) / 2, y1 + 0.3, "system (closed)", color=WORK, ha="center", fontsize=14, fontweight="bold")
    ax.text(9.6, 6.4, "surroundings", ha="center", fontsize=15, style="italic", color="#555")
    ax.text(1.0, 6.4, "surroundings", ha="center", fontsize=15, style="italic", color="#555")
    # Q and W
    wavy(ax, (3.9, 1.75), (3.9, 3.0), color=HOT, amp=0.1, waves=3)
    ax.text(3.6, 2.05, "Q", color=HOT, fontsize=16, fontweight="bold", ha="right")
    arrow(ax, (8.9, 5.35), (10.6, 5.35), color=WORK, lw=3)
    ax.plot([px + 0.17, px + 0.17], [y1, 5.35], color=WORK, lw=1, ls=":")
    ax.text(9.75, 5.6, "work W", color=WORK, ha="center", fontsize=14, fontweight="bold")
    arrow(ax, (px + 0.17, 5.35), (8.9, 5.35), color=WORK, lw=3, style="-")
    save(fig, "b4-1a", "b4-1a  System, surroundings and reservoir")


def b4_1b():
    fig, ax = plt.subplots(figsize=(9, 6))
    V1, V2, p = 0.020, 0.050, 1.0
    ax.fill_between([V1, V2], 0, p, color=SHADE, alpha=0.75, lw=0)
    ax.plot([V1, V2], [p, p], color=HOT, lw=3)
    arrow(ax, (0.033, p), (0.037, p), color=HOT, lw=0, ms=24)
    for v in (V1, V2):
        ax.plot([v, v], [0, p], "--", color=INK, lw=1)
    ax.text((V1 + V2) / 2, 0.5, r"$W = p\Delta V = 3.0\ \mathrm{kJ}$", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(V1, p + 0.05, "1", ha="center", fontsize=13, fontweight="bold"); ax.text(V2, p + 0.05, "2", ha="center", fontsize=13, fontweight="bold")
    ax.annotate("", (V2, 0.15), (V1, 0.15), arrowprops=dict(arrowstyle="<->", lw=1.3))
    ax.text(0.035, 0.2, r"$\Delta V = 0.030\ \mathrm{m^3}$", ha="center", fontsize=12)
    ax.set_xlim(0, 0.060); ax.set_ylim(0, 1.5)
    ax.set_xticks(np.arange(0, 0.061, 0.010)); ax.set_yticks(np.arange(0, 1.51, 0.5))
    ax.set_xlabel(r"$V\ /\ \mathrm{m^3}$"); ax.set_ylabel(r"$p\ /\ 10^5\ \mathrm{Pa}$")
    ax.grid(alpha=0.25); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    # inset: piston moving out by Δx
    ia = ax.inset_axes([0.0, 0.7, 0.32, 0.28]); ia.set_xlim(0, 6); ia.set_ylim(0, 3); ia.axis("off"); ia.set_aspect("equal")
    ia.plot([5.8, 0.3, 0.3, 5.8], [2.4, 2.4, 0.6, 0.6], color=INK, lw=2.5)
    ia.add_patch(Rectangle((0.3, 0.6), 3.0, 1.8, fc="#eef3f8", ec="none"))
    ia.add_patch(Rectangle((3.0, 0.62), 0.3, 1.76, fc="#ccc", ec="#888", ls="--", lw=1))
    ia.add_patch(Rectangle((4.2, 0.62), 0.3, 1.76, fc="#7f8c8d", ec=INK, lw=1.2))
    ia.add_patch(Rectangle((3.3, 0.62), 0.9, 1.76, fc=SHADE, alpha=0.7, ec="none"))
    ia.annotate("", (4.2, 2.75), (3.15, 2.75), arrowprops=dict(arrowstyle="->", lw=1.3))
    ia.text(3.7, 2.85, r"$\Delta x$", ha="center", va="bottom", fontsize=12)
    ia.text(1.6, 1.5, "area A", ha="center", va="center", fontsize=10)
    ia.text(3.0, 0.1, r"$\Delta V = A\Delta x$", ha="center", fontsize=12, fontweight="bold")
    save(fig, "b4-1b", "b4-1b  Isobaric expansion on a p–V graph")


def b4_1c():
    fig, ax = canvas(10, 7, (0, 10), (0, 8))
    ax.add_patch(FancyBboxPatch((3.7, 4.6), 2.6, 2.0, boxstyle="round,pad=0.1", fc="#e9e4ff", ec=GAS, lw=2.5))
    ax.text(5, 6.15, "gas", ha="center", fontsize=15, fontweight="bold", color=GAS)
    ax.text(5, 5.2, r"$\Delta U$", ha="center", fontsize=22)
    arrow(ax, (0.6, 5.6), (3.6, 5.6), color=HOT, lw=6, ms=30)
    ax.text(2.1, 6.0, "Q", ha="center", fontsize=20, fontweight="bold", color=HOT)
    ax.text(2.1, 5.0, "in = +", ha="center", fontsize=13, color=HOT)
    arrow(ax, (6.4, 5.6), (9.4, 5.6), color=WORK, lw=6, ms=30)
    ax.text(7.9, 6.0, "W", ha="center", fontsize=20, fontweight="bold", color=WORK)
    ax.text(7.9, 5.0, "by gas = +", ha="center", fontsize=13, color=WORK)
    ax.text(5, 3.95, r"$\Delta U = Q - W$", ha="center", fontsize=17, fontweight="bold")
    rows = [("Q", "+ heat supplied TO the gas", "− heat lost BY the gas"),
            (r"$\Delta U$", "+ internal energy rises (T ↑)", "− internal energy falls (T ↓)"),
            ("W", "+ work done BY the gas (expands)", "− work done ON the gas (compressed)")]
    tbl = ax.table(cellText=[[r[0], r[1], r[2]] for r in rows], colLabels=["quantity", "positive (+)", "negative (−)"],
                   bbox=[-0.04, 0.04, 1.08, 0.38], cellLoc="center", colWidths=[0.12, 0.44, 0.44])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#888")
        if r == 0:
            cell.set_facecolor("#ececec"); cell.set_text_props(fontweight="bold")
        elif c == 0:
            cell.set_text_props(fontweight="bold", color=[HOT, GAS, WORK][r - 1])
    save(fig, "b4-1c", "b4-1c  Energy accounting: first law sign convention")


def b4_1x():
    n, p = 0.20, 1.0e5
    T1, T2 = 300, 450
    V1, V2 = n * R * T1 / p, n * R * T2 / p
    fig = plt.figure(figsize=(13, 6))
    a = fig.add_axes([0.0, 0.05, 0.36, 0.85]); a.set_xlim(0, 6); a.set_ylim(0, 8); a.axis("off"); a.set_aspect("equal")
    a.plot([1.2, 1.2, 4.8, 4.8], [7.4, 1.6, 1.6, 7.4], color=INK, lw=4)
    a.add_patch(Rectangle((1.2, 1.6), 3.6, 3.4, fc="#eef3f8", ec="none"))
    a.add_patch(Rectangle((1.25, 5.0), 3.5, 0.45, fc="#7f8c8d", ec=INK, lw=1.5))
    a.add_patch(Rectangle((2.6, 5.45), 0.8, 0.6, fc="#95a5a6", ec=INK, lw=1))
    a.text(3.0, 6.3, "free piston", ha="center", fontsize=11)
    a.annotate("", (3.0, 7.7), (3.0, 6.7), arrowprops=dict(arrowstyle="->", lw=2, color=WORK))
    a.text(3.3, 7.3, "moves up", fontsize=10, color=WORK)
    gas_dots(a, 1.4, 1.8, 4.6, 4.8, n=35)
    a.text(3.0, 3.3, "0.20 mol", ha="center", fontsize=14, fontweight="bold",
           bbox=dict(fc="white", ec="none", alpha=0.8))
    xs = np.linspace(1.4, 4.6, 200)
    a.plot(xs, 1.05 + 0.22 * np.sin((xs - 1.4) / 3.2 * 2 * np.pi * 5), color="#e67e22", lw=3)
    a.plot([1.4, 1.4], [1.05, 0.4], color="#e67e22", lw=2); a.plot([4.6, 4.6], [1.05, 0.4], color="#e67e22", lw=2)
    a.text(3.0, 0.2, "heater coil", ha="center", fontsize=11, color="#e67e22")
    a.text(3.0, 7.95, r"$p = 1.0\times10^5\ \mathrm{Pa}$ constant", ha="center", fontsize=12, fontweight="bold")
    ax = fig.add_axes([0.45, 0.12, 0.52, 0.76])
    V = np.linspace(1.5e-3, 10e-3, 300)
    for T in (T1, T2):
        ax.plot(V * 1e3, n * R * T / V / 1e5, ":", color=ISO, lw=1.6)
    ax.text(9.6, n * R * T1 / 9.6e-3 / 1e5 + 0.05, "300 K", color="#777", fontsize=10)
    ax.text(9.6, n * R * T2 / 9.6e-3 / 1e5 + 0.05, "450 K", color="#777", fontsize=10)
    ax.fill_between([V1 * 1e3, V2 * 1e3], 0, 1.0, color=SHADE, alpha=0.75, lw=0)
    ax.plot([V1 * 1e3, V2 * 1e3], [1, 1], color=HOT, lw=3)
    arrow(ax, (6.1, 1), (6.4, 1), color=HOT, lw=0, ms=24)
    ax.plot(V1 * 1e3, 1, "o", color=INK, ms=8); ax.plot(V2 * 1e3, 1, "o", color=INK, ms=8)
    ax.text(V1 * 1e3, 1.12, "1\n300 K", ha="center", fontsize=12, fontweight="bold")
    ax.text(V2 * 1e3, 1.12, "2\n450 K", ha="center", fontsize=12, fontweight="bold")
    ax.text((V1 + V2) / 2 * 1e3, 0.45, "W", ha="center", fontsize=22, fontweight="bold")
    ax.set_xlim(0, 10.5); ax.set_ylim(0, 2.0)
    ax.set_xlabel(r"$V\ /\ 10^{-3}\ \mathrm{m^3}$"); ax.set_ylabel(r"$p\ /\ 10^5\ \mathrm{Pa}$")
    [ax.spines[s].set_visible(False) for s in ("top", "right")]
    save(fig, "b4-1x", "b4-1x  Isobaric heating (exam)")


# =========================== TOPIC 2: p–V PROCESSES ==========================
def b4_2a():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8.5))
    V = np.linspace(0.6, 5, 300)
    titles = ["(a) isothermal  (T constant)", "(b) isovolumetric  (V constant)",
              "(c) adiabatic  (Q = 0)", "(d) isobaric  (p constant)"]
    for ax, t in zip(axs.flat, titles):
        isotherms(ax, V, [1.5, 2.5, 3.5, 4.5, 5.5])
        ax.set_xlim(0, 5); ax.set_ylim(0, 5); pv_axes(ax); ax.set_title(t, loc="left", fontsize=13, fontweight="bold")
    ax = axs[0, 0]; v = np.linspace(1, 4, 200); ax.plot(v, 3.5 / v, color=HOT, lw=3); mid_arrow(ax, v, 3.5 / v, HOT)
    ax.text(3.3, 1.35, "pV = const", color=HOT, fontsize=12)
    ax = axs[0, 1]; ax.plot([2.2, 2.2], [1.2, 3.8], color=HOT, lw=3); arrow(ax, (2.2, 2.4), (2.2, 2.7), color=HOT, lw=0, ms=22)
    ax.text(2.4, 3.9, "W = 0", color=HOT, fontsize=12)
    ax = axs[1, 0]; v = np.linspace(1, 3.5, 200); pa = 4.5 * (1 / v) ** GAMMA
    ax.plot(v, pa, color=HOT, lw=3); mid_arrow(ax, v, pa, HOT, 0.35)
    ax.text(2.3, 1.6, r"$pV^{\gamma}$ = const" + "\n(steeper; crosses isotherms)", color=HOT, fontsize=11)
    ax = axs[1, 1]; ax.fill_between([1.2, 3.8], 0, 2.5, color=SHADE, alpha=0.75, lw=0)
    ax.plot([1.2, 3.8], [2.5, 2.5], color=HOT, lw=3); arrow(ax, (2.4, 2.5), (2.7, 2.5), color=HOT, lw=0, ms=22)
    ax.text(2.5, 1.2, r"$W = p\Delta V$", ha="center", fontsize=13, fontweight="bold")
    fig.text(0.5, 0.005, "dotted grey lines = isotherms (pV = nRT)", ha="center", color="#777", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "b4-2a", "b4-2a  The four thermodynamic processes")


def b4_2b():
    fig, ax = plt.subplots(figsize=(9, 6.5))
    VA, pA, VF = 1.0, 4.0, 3.0
    v = np.linspace(VA, VF, 300)
    pi, pa = pA * VA / v, pA * (VA / v) ** GAMMA
    ax.plot(v, pi, "--", color=COLD, lw=2.8, label=r"isothermal  $pV$ = const")
    ax.plot(v, pa, "-", color=HOT, lw=3, label=r"adiabatic  $pV^{5/3}$ = const")
    mid_arrow(ax, v, pi, COLD, 0.55); mid_arrow(ax, v, pa, HOT, 0.4)
    ax.plot([VF, VF], [0, pi[-1]], ":", color=ISO)
    for (x, y, s, off) in [(VA, pA, "A", (-0.12, 0.1)), (VF, pi[-1], "B", (0.1, 0.05)), (VF, pa[-1], "C", (0.1, -0.1))]:
        ax.plot(x, y, "o", color=INK, ms=8, zorder=5); ax.text(x + off[0], y + off[1], s, fontsize=15, fontweight="bold")
    ax.annotate("", (VF + 0.35, pi[-1]), (VF + 0.35, pa[-1]), arrowprops=dict(arrowstyle="]-[", lw=1.5, mutation_scale=6))
    ax.text(VF + 0.5, (pi[-1] + pa[-1]) / 2, "temperature\ndrop", va="center", fontsize=12, color=HOT, fontweight="bold")
    ax.set_xlim(0, 4.2); ax.set_ylim(0, 4.6); pv_axes(ax)
    ax.set_xticks([VA, VF]); ax.set_xticklabels([r"$V_A$", r"$V_{final}$"])
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    save(fig, "b4-2b", "b4-2b  Isothermal vs adiabatic expansion")


def b4_2c():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.35, 1]})
    th = np.array([0, 20, 40, 60, 80, 100]); p0 = 100.0
    p = p0 * (th + 273) / 273 + rng.normal(0, 0.9, th.size)
    m, c = np.polyfit(th, p, 1)
    a1.errorbar(th, p, yerr=2.0, fmt="o", color=INK, capsize=4, ms=6, zorder=5)
    a1.plot([0, 105], m * np.array([0, 105]) + c, color=HOT, lw=2.2)
    x0 = -c / m
    a1.plot([x0, 0], [0, c], "--", color=HOT, lw=1.8)
    a1.plot(x0, 0, "o", color=HOT, ms=8, clip_on=False, zorder=6)
    a1.annotate("≈ −273 °C\n(absolute zero)", (x0, 0), (x0 + 25, 25), fontsize=12, color=HOT, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=HOT))
    a1.set_xlim(-300, 120); a1.set_ylim(0, 150); a1.axvline(0, color="#bbb", lw=0.8)
    a1.set_xlabel(r"$\theta\ /\ ^\circ\mathrm{C}$"); a1.set_ylabel(r"$p\ /\ \mathrm{kPa}$")
    a1.set_title("Pressure law (constant V)", loc="left", fontweight="bold")
    [a1.spines[s].set_visible(False) for s in ("top", "right")]; a1.grid(alpha=0.2)
    V = np.array([20, 25, 30, 40, 50, 60.0]); pk = 3000 / V
    pk_meas = pk * (1 + rng.normal(0, 0.01, V.size)); pk_meas[3] = pk[3] * 1.22
    vv = np.linspace(17, 65, 200)
    a2.plot(vv, 3000 / vv, color=COLD, lw=2.2, zorder=2)
    a2.errorbar(V, pk_meas, yerr=3, fmt="o", color=INK, capsize=4, ms=6, zorder=5)
    a2.add_patch(Circle((V[3], pk_meas[3]), 2.6, fill=False, ec=HOT, lw=2, zorder=6))
    a2.annotate("anomalous", (V[3] + 2.0, pk_meas[3] + 2.5), (V[3] + 6, pk_meas[3] + 25), color=HOT, fontsize=13, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=HOT))
    a2.set_xlim(0, 70); a2.set_ylim(0, 170)
    a2.set_xlabel(r"$V\ /\ \mathrm{cm^3}$"); a2.set_ylabel(r"$p\ /\ \mathrm{kPa}$")
    a2.set_title("Boyle's law (constant T)", loc="left", fontweight="bold")
    [a2.spines[s].set_visible(False) for s in ("top", "right")]; a2.grid(alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "b4-2c", "b4-2c  Student experiments")


def b4_2x():
    VX, pX = 2.0e-3, 1.0e5; VY = 1.0e-3
    fig, ax = plt.subplots(figsize=(10, 7))
    v = np.linspace(VY, VX, 300)
    pa = pX * (VX / v) ** GAMMA; pi = pX * VX / v
    ax.plot(v * 1e3, pa / 1e5, color=HOT, lw=3, label=r"adiabat  $pV^{5/3}$ = const")
    ax.plot(v * 1e3, pi / 1e5, "--", color=COLD, lw=2.5, label="isotherm  300 K")
    mid_arrow(ax, (v * 1e3)[::-1], (pa / 1e5)[::-1], HOT, 0.5); mid_arrow(ax, (v * 1e3)[::-1], (pi / 1e5)[::-1], COLD, 0.5)
    ax.plot(VX * 1e3, 1, "o", color=INK, ms=9, zorder=5)
    ax.text(VX * 1e3 + 0.05, 1.12, "X\n300 K", fontsize=13, fontweight="bold")
    ax.plot(VY * 1e3, pa[0] / 1e5, "o", color=HOT, ms=9, zorder=5)
    ax.text(VY * 1e3 - 0.1, pa[0] / 1e5, "Y", fontsize=14, fontweight="bold", ha="right", va="center")
    ax.plot([0, VY * 1e3], [pa[0] / 1e5] * 2, ":", color=HOT); ax.text(0.05, pa[0] / 1e5 + 0.07, "p = ?", color=HOT, fontsize=14, fontweight="bold")
    ax.plot(VY * 1e3, 2.0, "o", color=COLD, ms=9, zorder=5)
    ax.text(VY * 1e3 - 0.1, 2.0, "Z", fontsize=14, fontweight="bold", ha="right", va="center")
    ax.plot([0, VY * 1e3], [2, 2], ":", color=COLD)
    ax.plot([VY * 1e3] * 2, [0, 2], ":", color=ISO); ax.plot([VX * 1e3] * 2, [0, 1], ":", color=ISO)
    ax.set_xlim(0, 2.5); ax.set_ylim(0, 3.5)
    ax.set_xticks(np.arange(0, 2.51, 0.5)); ax.set_yticks(np.arange(0, 3.51, 0.5))
    ax.set_xlabel(r"$V\ /\ 10^{-3}\ \mathrm{m^3}$"); ax.set_ylabel(r"$p\ /\ 10^5\ \mathrm{Pa}$")
    ax.grid(alpha=0.2); [ax.spines[s].set_visible(False) for s in ("top", "right")]
    ax.legend(loc="lower left", frameon=False, fontsize=12)
    ia = fig.add_axes([0.66, 0.5, 0.25, 0.25]); ia.set_xlim(0, 6); ia.set_ylim(0, 4); ia.axis("off"); ia.set_aspect("equal")
    ia.add_patch(Rectangle((0.2, 0.6), 4.4, 2.6, fc="#f3d9a4", ec="none", hatch="///"))
    ia.add_patch(Rectangle((0.5, 0.9), 4.1, 2.0, fc="#eef3f8", ec=INK, lw=2))
    ia.add_patch(Rectangle((2.4, 0.92), 0.3, 1.96, fc="#7f8c8d", ec=INK, lw=1.2))
    ia.add_patch(Rectangle((2.7, 1.78), 2.6, 0.24, fc="#95a5a6", ec=INK, lw=1))
    gas_dots(ia, 0.65, 1.05, 2.3, 2.75, n=22, s=8)
    arrow(ia, (5.9, 1.9), (5.35, 1.9), color=HOT, lw=2.5, ms=14)
    ia.text(2.4, 3.45, "insulated", ha="center", fontsize=10); ia.text(2.4, 0.15, "pushed in quickly", ha="center", fontsize=10, color=HOT)
    save(fig, "b4-2x", "b4-2x  Adiabatic compression (exam)")


# =========================== TOPIC 3: CARNOT CYCLE ==========================
def b4_3a():
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    VA, VB = 1.0, 4.0
    v = np.linspace(VA, VB, 200)
    up = 4.0 * VA / v + 0.4 * np.sin(np.pi * (v - VA) / (VB - VA))
    lo = 1.2 * (VA / v) ** 0.7
    for ax in axs:
        ax.set_xlim(0, 5); ax.set_ylim(0, 4.8); pv_axes(ax)
    a = axs[0]; a.fill_between(v, 0, up, color=SHADE, alpha=0.8, lw=0); a.plot(v, up, color=HOT, lw=3); mid_arrow(a, v, up, HOT, 0.45)
    a.text(2.5, 0.8, "work done\nby gas", ha="center", fontsize=13, fontweight="bold")
    a.set_title("(a) expansion A→B", loc="left", fontweight="bold")
    a = axs[1]; a.fill_between(v, 0, lo, color="#a8dadc", alpha=0.9, lw=0); a.plot(v, lo, color=COLD, lw=3)
    mid_arrow(a, v[::-1], lo[::-1], COLD, 0.5); a.text(2.5, 0.35, "work done on gas", ha="center", fontsize=13, fontweight="bold")
    a.set_title("(b) compression B→A", loc="left", fontweight="bold")
    a = axs[2]; a.fill(np.r_[v, v[::-1]], np.r_[up, lo[::-1]], color="#b8e0d2", lw=0)
    a.plot(v, up, color=HOT, lw=3); a.plot(v, lo, color=COLD, lw=3)
    mid_arrow(a, v, up, HOT, 0.45); mid_arrow(a, v[::-1], lo[::-1], COLD, 0.5)
    a.text(2.4, 1.7, "net work", ha="center", fontsize=14, fontweight="bold", color="#1d6b5f")
    a.set_title("(c) closed cycle (clockwise)", loc="left", fontweight="bold")
    for a in axs:
        a.plot(VA, 4.0, "o", color=INK, ms=7); a.plot(VB, up[-1], "o", color=INK, ms=7)
        a.text(VA - 0.25, 4.0, "A", fontsize=14, fontweight="bold"); a.text(VB + 0.1, up[-1] + 0.1, "B", fontsize=14, fontweight="bold")
    axs[1].plot(VA, lo[0], "o", color=INK, ms=7); axs[2].plot(VA, lo[0], "o", color=INK, ms=7)
    axs[2].plot([VA, VA], [lo[0], 4.0], color=INK, lw=1.2, ls=":"); axs[2].plot([VB, VB], [lo[-1], up[-1]], color=INK, lw=1.2, ls=":")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "b4-3a", "b4-3a  Net work from a cycle")


def flow_diagram(ax, cx, reverse, QH=100, W=40):
    QC = QH - W
    ax.add_patch(FancyBboxPatch((cx - 2.0, 7.0), 4.0, 1.1, boxstyle="round,pad=0.05", fc="#f6c1c1", ec=HOT, lw=2))
    ax.text(cx, 7.55, r"hot reservoir  $T_H$", ha="center", va="center", fontsize=13, fontweight="bold", color="#8b1e2d")
    ax.add_patch(FancyBboxPatch((cx - 2.0, 0.4), 4.0, 1.1, boxstyle="round,pad=0.05", fc="#c6e2f5", ec=COLD, lw=2))
    ax.text(cx, 0.95, r"cold reservoir  $T_C$", ha="center", va="center", fontsize=13, fontweight="bold", color="#1b4f72")
    ax.add_patch(Circle((cx, 4.25), 1.0, fc="white", ec=INK, lw=2.5, zorder=4))
    ax.text(cx, 4.25, "heat\npump" if reverse else "engine", ha="center", va="center", fontsize=13, fontweight="bold", zorder=5)

    def band(x0, y0, x1, y1, E, color, label, lx, ly):
        if reverse:
            x0, y0, x1, y1 = x1, y1, x0, y0
        w = E * 0.009  # band width in data units, proportional to energy
        ax.add_patch(FancyArrow(x0, y0, x1 - x0, y1 - y0, width=w, head_width=w + 0.35, head_length=0.4,
                                length_includes_head=True, fc=color, ec="none", alpha=0.9, zorder=3))
        ax.text(lx, ly, label, fontsize=15, fontweight="bold", color=color, va="center")

    band(cx, 6.95, cx, 5.27, QH, HOT, r"$Q_H$", cx + 0.75, 6.1)
    band(cx, 3.23, cx, 1.55, QC, COLD, r"$Q_C$", cx + 0.6, 2.4)
    band(cx + 1.02, 4.25, cx + 3.0, 4.25, W, WORK, "W", cx + 1.9, 4.9)
    ax.text(cx, -0.25, (r"$W = Q_H - Q_C$" if not reverse else r"$Q_H = Q_C + W$"), ha="center", fontsize=13)


def b4_3b():
    fig, ax = canvas(13, 7, (0, 13), (-0.6, 8.6))
    flow_diagram(ax, 3.2, False); flow_diagram(ax, 9.6, True)
    ax.text(3.2, 8.4, "Heat engine", ha="center", fontsize=15, fontweight="bold")
    ax.text(9.6, 8.4, "Heat pump / refrigerator", ha="center", fontsize=15, fontweight="bold")
    ax.text(6.4, -0.5, "arrow widths ∝ energy  (Q_H = 100 %, W = 40 %, Q_C = 60 %)", ha="center", fontsize=10, color="#777")
    save(fig, "b4-3b", "b4-3b  Heat engine and heat pump")


def b4_3c():
    TH, TC = 500.0, 300.0; nR = 1.0
    VA, VB = 1.0, 2.6
    r = (TH / TC) ** (1 / (GAMMA - 1))
    VC, VD = VB * r, VA * r
    fig, (ax, at) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.4, 1]})
    vAB = np.linspace(VA, VB, 200); pAB = nR * TH / vAB
    vBC = np.linspace(VB, VC, 200); pBC = nR * TH / VB * (VB / vBC) ** GAMMA
    vCD = np.linspace(VC, VD, 200); pCD = nR * TC / vCD
    vDA = np.linspace(VD, VA, 200); pDA = nR * TC / VD * (VD / vDA) ** GAMMA
    vv = np.linspace(0.6, 6.2, 300)
    ax.plot(vv, nR * TH / vv, ":", color=HOT, lw=1.2, alpha=0.7); ax.plot(vv, nR * TC / vv, ":", color=COLD, lw=1.2, alpha=0.7)
    ax.text(6.25, nR * TH / 6.25, r"$T_H$", color=HOT, fontsize=13); ax.text(6.25, nR * TC / 6.25, r"$T_C$", color=COLD, fontsize=13)
    ax.fill(np.r_[vAB, vBC, vCD, vDA], np.r_[pAB, pBC, pCD, pDA], color="#b8e0d2", lw=0)
    ax.text(2.25, 190, "W", fontsize=24, fontweight="bold", color="#1d6b5f", ha="center", va="center")
    for v, p, c, ls in [(vAB, pAB, HOT, "-"), (vBC, pBC, INK, "-"), (vCD, pCD, COLD, "-"), (vDA, pDA, INK, "-")]:
        ax.plot(v, p, color=c, lw=3, ls=ls); mid_arrow(ax, v, p, c, 0.5)
    pts = {"A": (VA, nR * TH / VA), "B": (VB, nR * TH / VB), "C": (VC, nR * TC / VC), "D": (VD, nR * TC / VD)}
    offs = {"A": (-0.22, 5), "B": (0.08, 10), "C": (0.1, -8), "D": (-0.2, -22)}
    for k, (x, y) in pts.items():
        ax.plot(x, y, "o", color=INK, ms=8, zorder=5); ax.text(x + offs[k][0], y + offs[k][1], k, fontsize=15, fontweight="bold")
    wavy(ax, (2.5, 440), (1.72, 305), color=HOT, amp=0.05, waves=3)
    ax.text(2.55, 450, r"$Q_H$ in", color=HOT, fontsize=14, fontweight="bold")
    wavy(ax, (3.1, 92), (2.75, 25), color=COLD, amp=0.05, waves=3)
    ax.text(2.7, 12, r"$Q_C$ out", color=COLD, fontsize=14, fontweight="bold", ha="right")
    ax.text(3.45, 145, "adiabat", fontsize=11)
    ax.text(1.25, 230, "adiabat", fontsize=11, ha="right")
    ax.set_xlim(0, 6.6); ax.set_ylim(0, 560); pv_axes(ax)
    ax.set_title("p–V diagram (clockwise)", loc="left", fontweight="bold")
    S1, S2 = 1.0, 2.0
    at.fill([S1, S2, S2, S1], [TC, TC, TH, TH], color="#b8e0d2")
    loop = [(S1, TH, S2, TH, HOT), (S2, TH, S2, TC, INK), (S2, TC, S1, TC, COLD), (S1, TC, S1, TH, INK)]
    for x0, y0, x1, y1, c in loop:
        at.plot([x0, x1], [y0, y1], color=c, lw=3); arrow(at, (x0 + 0.45 * (x1 - x0), y0 + 0.45 * (y1 - y0)), (x0 + 0.55 * (x1 - x0), y0 + 0.55 * (y1 - y0)), color=c, lw=0, ms=22)
    for k, x, y, o in [("A", S1, TH, (-0.12, 8)), ("B", S2, TH, (0.04, 8)), ("C", S2, TC, (0.04, -22)), ("D", S1, TC, (-0.12, -22))]:
        at.plot(x, y, "o", color=INK, ms=8); at.text(x + o[0], y + o[1], k, fontsize=15, fontweight="bold")
    at.text(1.5, 400, "W", fontsize=24, fontweight="bold", color="#1d6b5f", ha="center")
    at.text(1.5, TH + 15, r"$Q_H = T_H\Delta S$", ha="center", color=HOT, fontsize=12)
    at.text(1.5, TC - 35, r"$Q_C = T_C\Delta S$", ha="center", color=COLD, fontsize=12)
    at.set_xlim(0, 2.6); at.set_ylim(0, 600); pv_axes(at, "S", "T")
    at.set_yticks([TC, TH]); at.set_yticklabels([r"$T_C$", r"$T_H$"])
    at.set_title("T–S diagram", loc="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "b4-3c", "b4-3c  The Carnot cycle")


def b4_3x():
    Pout, TH, TC = 500.0, 800.0, 300.0
    eta = 0.60 * (1 - TC / TH); Pin = Pout / eta; Prej = Pin - Pout
    k = 2.2 / Pin  # band width per MW
    fig, ax = canvas(13, 7.5, (0, 13), (-0.5, 8))
    win, wout, wrej = Pin * k, Pout * k, Prej * k
    yc = 5.0
    # boiler
    ax.add_patch(FancyBboxPatch((0.2, yc - 1.4), 2.0, 2.8, boxstyle="round,pad=0.05", fc="#f6c1c1", ec=HOT, lw=2))
    ax.text(1.2, yc + 0.25, "boiler", ha="center", fontsize=14, fontweight="bold", color="#8b1e2d")
    ax.text(1.2, yc - 0.4, "800 K", ha="center", fontsize=14, fontweight="bold", color="#8b1e2d")
    # input band
    ax.add_patch(Rectangle((2.25, yc - win / 2), 2.6, win, fc=HOT, alpha=0.55, ec="none"))
    ax.text(3.55, yc + win / 2 + 0.25, "input = ?", ha="center", fontsize=14, fontweight="bold", color=HOT)
    # engine box
    ex0, ex1 = 4.85, 7.15
    ax.add_patch(FancyBboxPatch((ex0, yc - 1.4), ex1 - ex0, 2.8, boxstyle="round,pad=0.05", fc="white", ec=INK, lw=2.5, zorder=3))
    ax.text(6.0, yc + 0.4, "turbine\n(engine)", ha="center", va="center", fontsize=13, fontweight="bold", zorder=4)
    ax.text(6.0, yc - 0.85, "η = 60 % of Carnot", ha="center", fontsize=11, zorder=4, color="#444")
    # useful output to right (top part of band)
    ytop = yc + win / 2
    ax.add_patch(Polygon([(ex1, ytop), (11.6, ytop), (12.6, ytop - wout / 2), (11.6, ytop - wout), (ex1, ytop - wout)],
                         fc=WORK, alpha=0.75, ec="none"))
    ax.text(9.6, ytop + 0.3, "useful output 500 MW", ha="center", fontsize=14, fontweight="bold", color=WORK)
    # rejected heat: bottom part of band, bending down
    yb0, yb1 = yc - win / 2, yc - win / 2 + wrej
    th = np.linspace(0, np.pi / 2, 60); r_in, r_out = 0.6, 0.6 + wrej
    cx, cy = ex1, yb0 - r_in + 0  # centre of bend
    cy = yb1 - r_out + wrej  # keep inner edge at yb1
    cy = yb0 + wrej - r_in - wrej  # simple: arc centred below band
    cx = ex1 + 0.2
    outer = np.c_[cx + r_out * np.sin(th), cy + r_out * np.cos(th)]
    inner = np.c_[cx + r_in * np.sin(th[::-1]), cy + r_in * np.cos(th[::-1])]
    ytop_b = cy + r_out; ybot_b = cy + r_in
    poly = np.r_[[[ex1, ytop_b]], [[cx, ytop_b]], outer, [[cx + r_out, 1.6]], [[cx + (r_in + r_out) / 2, 1.1]],
                 [[cx + r_in, 1.6]], inner, [[cx, ybot_b]], [[ex1, ybot_b]]]
    ax.add_patch(Polygon(poly, fc=COLD, alpha=0.55, ec="none"))
    # cooling tower
    tx = cx + (r_in + r_out) / 2
    ax.add_patch(Polygon([(tx - 1.0, -0.4), (tx - 0.6, 0.6), (tx - 0.75, 1.05), (tx + 0.75, 1.05), (tx + 0.6, 0.6), (tx + 1.0, -0.4)],
                         fc="#d6dbdf", ec=INK, lw=2))
    ax.text(tx + 1.25, 0.35, "cooling tower\n300 K, rejected = ?", fontsize=13, fontweight="bold", color=COLD, va="center")
    ax.text(0.2, -0.35, "band widths drawn to scale", fontsize=10, color="#777")
    save(fig, "b4-3x", "b4-3x  Power station energy flow (exam)")


# =========================== TOPIC 4: ENTROPY ================================
def b4_4a():
    fig, ax = canvas(12, 7.5, (0, 12), (-1.2, 7))
    ax.add_patch(Rectangle((0.3, 0.3), 11.4, 6.0, fill=False, ec=INK, lw=2, ls=(0, (7, 4))))
    ax.text(6, 6.5, "isolated system: fridge + kitchen", ha="center", fontsize=14, fontweight="bold")
    # fridge
    ax.add_patch(FancyBboxPatch((1.6, 0.7), 2.6, 4.6, boxstyle="round,pad=0.08", fc="#eaf4fb", ec=INK, lw=2.5))
    ax.plot([1.6, 4.2], [3.6, 3.6], color=INK, lw=2); ax.plot([3.9, 3.9], [3.9, 4.8], color=INK, lw=4); ax.plot([3.9, 3.9], [1.6, 3.0], color=INK, lw=4)
    ax.text(2.9, 2.4, "S ↓", ha="center", fontsize=24, fontweight="bold", color=COLD)
    ax.text(2.9, 4.4, "fridge", ha="center", fontsize=12)
    # back coils
    yy = np.linspace(1.0, 5.0, 400); ax.plot(4.45 + 0.15 * np.sign(np.sin(yy * 6)), yy, color="#555", lw=1.2)
    ax.text(4.5, 0.45, "coils", ha="center", fontsize=10)
    wavy(ax, (4.8, 3.6), (7.6, 3.6), color=HOT, amp=0.15, waves=4)
    ax.text(6.2, 4.1, "Q + W", ha="center", fontsize=16, fontweight="bold", color=HOT)
    ax.text(9.2, 4.7, "kitchen air", ha="center", fontsize=13, style="italic")
    ax.text(9.2, 3.6, "S ↑ (larger)", ha="center", fontsize=22, fontweight="bold", color=HOT)
    # plug and cable
    ax.add_patch(Rectangle((8.9, 1.2), 0.9, 0.6, fc="#ddd", ec=INK, lw=1.5))
    ax.plot([9.6, 10.1], [1.35, 1.35], color=INK, lw=2); ax.plot([9.6, 10.1], [1.65, 1.65], color=INK, lw=2)
    ax.plot(np.r_[8.9, 7.5, 6.0, 4.2], np.r_[1.5, 1.2, 1.0, 1.2], color=INK, lw=2)
    ax.text(9.35, 0.75, "work W in", ha="center", fontsize=13, fontweight="bold", color=WORK)
    arrow(ax, (7.2, 0.75), (5.8, 0.75), color=WORK, lw=2.5)
    ax.add_patch(FancyBboxPatch((0.6, -1.05), 10.8, 0.75, boxstyle="round,pad=0.05", fc="#fff4d6", ec="#e0b84d", lw=1.5))
    ax.text(6, -0.68, r"$\Delta S_{total} = \Delta S_{kitchen} + \Delta S_{fridge} > 0$", ha="center", va="center", fontsize=16, fontweight="bold")
    save(fig, "b4-4a", "b4-4a  A fridge in a kitchen")


def b4_4b():
    fig, ax = canvas(11, 6.5, (0, 11), (-0.5, 6.5))
    ax.add_patch(Rectangle((0.6, 0.0), 9.8, 6.0, fill=False, ec=INK, lw=2, ls=(0, (7, 4))))
    ax.text(9.9, 5.6, "isolated", ha="right", fontsize=13, style="italic")
    ax.add_patch(Rectangle((1.5, 1.8), 4.0, 2.6, fc="#f6a5a5", ec=HOT, lw=2.5))
    ax.add_patch(Rectangle((5.5, 1.8), 4.0, 2.6, fc="#a9d3ee", ec=COLD, lw=2.5))
    ax.text(3.5, 4.0, r"A  (hot, $T_A$)", ha="center", fontsize=14, fontweight="bold", color="#8b1e2d")
    ax.text(7.5, 4.0, r"B  (cold, $T_B$)", ha="center", fontsize=14, fontweight="bold", color="#1b4f72")
    arrow(ax, (2.2, 3.15), (8.8, 3.15), color=HOT, lw=7, ms=34)
    ax.text(4.3, 3.5, "Q", ha="center", fontsize=16, fontweight="bold", color=HOT)
    arrow(ax, (6.6, 2.4), (4.4, 2.4), color=COLD, lw=1.4, ms=12)
    ax.text(4.3, 2.4, "Q", ha="right", fontsize=11, color=COLD, va="center")
    ax.text(3.5, 1.15, r"$\Delta S_A = -Q/T_A$ (small)", ha="center", fontsize=14, color="#8b1e2d")
    ax.text(7.5, 1.15, r"$\Delta S_B = +Q/T_B$ (large)", ha="center", fontsize=14, color="#1b4f72")
    ax.text(5.5, 0.35, r"$T_A > T_B \Rightarrow \Delta S_{total} = Q\left(\frac{1}{T_B}-\frac{1}{T_A}\right) > 0$", ha="center", fontsize=13)
    save(fig, "b4-4b", "b4-4b  Heat flow between two bodies")


def b4_4c():
    fig = plt.figure(figsize=(15, 6))
    a = fig.add_axes([0.05, 0.12, 0.38, 0.75])
    tot = np.arange(2, 13); ways = 6 - np.abs(tot - 7)
    bars = a.bar(tot, ways, color=[HOT if t == 7 else "#8fb8de" for t in tot], ec=INK)
    for t, w in zip(tot, ways):
        a.text(t, w + 0.1, str(w), ha="center", fontsize=11)
    a.set_xticks(tot); a.set_yticks(range(7)); a.set_ylim(0, 7)
    a.set_xlabel("total of two dice"); a.set_ylabel("number of ways (microstates)")
    a.set_title("Two dice: 36 microstates", loc="left", fontweight="bold")
    [a.spines[s].set_visible(False) for s in ("top", "right")]
    a.text(7, 6.55, "most probable", ha="center", fontsize=11, color=HOT, fontweight="bold")
    b = fig.add_axes([0.48, 0.02, 0.5, 0.88]); b.set_xlim(0, 4 * 3.1); b.set_ylim(-0.4, 2 * 2.6 + 0.4); b.axis("off"); b.set_aspect("equal")
    b.set_title("3 particles in two halves: 2³ = 8 microstates", loc="left", fontweight="bold")
    arrs = list(itertools.product("LR", repeat=3))
    arrs.sort(key=lambda a: (a.count("L") not in (0, 3), -a.count("L")))
    cols = {"A": "#e76f51", "B": "#2a9d8f", "C": "#6c5ce7"}
    for i, arr in enumerate(arrs):
        x0 = (i % 4) * 3.1 + 0.15; y0 = (1 - i // 4) * 2.6 + 0.4
        ordered = arr.count("L") in (0, 3)
        b.add_patch(Rectangle((x0, y0), 2.8, 1.6, fc="white", ec=HOT if ordered else INK, lw=3.5 if ordered else 1.5))
        b.plot([x0 + 1.4, x0 + 1.4], [y0, y0 + 1.6], color=INK, lw=1, ls="--")
        side_n = {"L": 0, "R": 0}
        for p, s in zip("ABC", arr):
            k = side_n[s]; side_n[s] += 1
            cx = x0 + (0.35 if s == "L" else 1.75) + (k % 2) * 0.7; cy = y0 + 1.15 - (k // 2) * 0.7 - (k % 2) * 0.2
            b.add_patch(Circle((cx, cy), 0.25, fc=cols[p], ec=INK, lw=1)); b.text(cx, cy, p, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        b.text(x0 + 1.4, y0 - 0.3, "ordered" if ordered else "disordered", ha="center", fontsize=11,
               color=HOT if ordered else "#444", fontweight="bold" if ordered else "normal")
    save(fig, "b4-4c", "b4-4c  Microstates and disorder")


def b4_4x():
    fig, ax = canvas(11, 7, (0, 11), (0, 7))
    ax.add_patch(Rectangle((0.4, 0.4), 10.2, 6.2, fill=False, ec=INK, lw=2, ls=(0, (7, 4))))
    ax.text(0.7, 6.15, "room, 293 K", fontsize=15, fontweight="bold", color=HOT)
    ax.add_patch(Rectangle((1.5, 1.2), 8.0, 0.35, fc="#c8a27a", ec=INK, lw=1.5))
    ax.text(8.9, 0.85, "bench", ha="center", fontsize=11)
    bx0, bx1, by0, by1 = 4.2, 6.8, 1.55, 4.3
    ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, 1.9, fc="#d8eefa", ec="none"))
    for (x, y) in [(4.4, 1.7), (5.3, 1.75), (6.0, 1.7), (4.7, 2.5), (5.6, 2.45), (4.9, 3.0), (5.8, 2.95)]:
        ax.add_patch(Rectangle((x, y), 0.65, 0.6, fc="#f4fbff", ec="#7fb3d5", lw=1.5, angle=rng.uniform(-12, 12)))
    ax.plot([bx0, bx0, bx1, bx1], [by1, by0, by0, by1], color=INK, lw=3)
    ax.text(5.5, 4.65, "0.50 kg ice, 273 K", ha="center", fontsize=14, fontweight="bold", color="#1b4f72")
    for (p0, p1) in [((1.4, 3.0), (3.9, 2.7)), ((9.6, 3.0), (7.1, 2.7)), ((3.0, 5.6), (4.6, 4.0)), ((8.0, 5.6), (6.4, 4.0))]:
        wavy(ax, p0, p1, color=HOT, amp=0.1, waves=3)
    ax.text(2.6, 3.3, "Q", fontsize=15, fontweight="bold", color=HOT); ax.text(8.3, 3.3, "Q", fontsize=15, fontweight="bold", color=HOT)
    ax.text(5.5, 2.05, r"$\Delta S_{ice} = ?$", ha="center", fontsize=15, fontweight="bold", bbox=dict(fc="white", ec=COLD, boxstyle="round"))
    ax.text(8.6, 5.5, r"$\Delta S_{room} = ?$", ha="center", fontsize=15, fontweight="bold", bbox=dict(fc="white", ec=HOT, boxstyle="round"))
    save(fig, "b4-4x", "b4-4x  Melting ice (exam)")


if __name__ == "__main__":
    for f in [b4_1a, b4_1b, b4_1c, b4_1x, b4_2a, b4_2b, b4_2c, b4_2x,
              b4_3a, b4_3b, b4_3c, b4_3x, b4_4a, b4_4b, b4_4c, b4_4x]:
        f()
