#!/usr/bin/env python3
"""sci3d - realistic, scientifically accurate 3D physics & chemistry diagrams.

Everything is computed from real equations / reference data (never hand-placed):
  physics : Coulomb E-fields + equipotentials, Biot-Savart B-fields, wave
            interference, Kepler orbits, projectile with drag, z=f(x,y)
            surfaces, vector fields
  chemistry: molecules (name / formula / SMILES / file) with real 3D geometry
            (RDKit ETKDG + MMFF, or ASE reference data), crystals (ASE lattices),
            hydrogen-like atomic orbitals (exact wavefunctions, 90 % isosurface)

CLI : python sci3d.py <command> [args] -o out.png      (see --help / SKILL.md)
API : from sci3d import *; s = Scene("Title"); efield(s, ...); s.save("x.png")
"""
import os, sys, math, re, argparse, warnings

warnings.filterwarnings("ignore")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
import numpy as np
import pyvista as pv
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()
pv.OFF_SCREEN = True

BOHR = 0.529177210903  # Å
POS, NEG = "#e63946", "#3a86ff"  # +/- (charge, phase)
THEMES = {
    "dark": dict(bg="#04060d", top="#1b2442", fg="#f2f4f8"),
    "light": dict(bg="#e9edf3", top="#ffffff", fg="#14161a"),
    "white": dict(bg="#ffffff", top=None, fg="#000000"),
}


def nice(x):
    """Round to 1/2/5 x 10^n (for scale bars)."""
    if x <= 0:
        return 1.0
    e = math.floor(math.log10(x)); f = x / 10 ** e
    return (1 if f < 1.5 else 2 if f < 3.5 else 5 if f < 7.5 else 10) * 10 ** e


def fib_sphere(n, r=1.0, c=(0, 0, 0)):
    """n near-uniform points on a sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n); th = np.pi * (1 + 5 ** 0.5) * i
    p = np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]
    return np.asarray(c, float) + r * p


# =============================================================== Scene
class Scene:
    """Off-screen PyVista scene with publication defaults. All add methods chain."""

    def __init__(self, title=None, caption=None, theme="dark", size=(1600, 1200),
                 unit="", ssao=False):
        self.t = THEMES.get(theme, THEMES["dark"])
        self.p = pv.Plotter(off_screen=True, window_size=list(size), lighting="three lights")
        if self.t["top"]:
            self.p.set_background(self.t["bg"], top=self.t["top"])
        else:
            self.p.set_background(self.t["bg"])
        self.title, self.caption, self.unit = title, caption, unit
        self.size, self.ssao = size, ssao
        self.notes, self.scale, self._view, self._zoom = [], 1.0, "iso", 1.0
        self.transparent, self.bar, self.bar_axis = False, True, "y"
        self.fs = max(10, int(size[1] / 55))

    # ---------- primitives
    def cmap(self, name, lo=0.28):
        """Colormap; on dark themes drop the darkest end so lines stay visible."""
        if not self.t["top"] or self.t["bg"] != THEMES["dark"]["bg"]:
            return name
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap
        return ListedColormap(colormaps[name](np.linspace(lo, 1, 256)))

    def r0(self, k=1.0):
        return 0.006 * self.scale * k

    def mesh(self, m, color="#cccccc", opacity=1.0, **kw):
        kw.setdefault("smooth_shading", True)
        kw.setdefault("specular", 0.45); kw.setdefault("specular_power", 30)
        if "scalars" in kw:
            kw.setdefault("show_scalar_bar", False)
            color = None
        if opacity < 1:
            self.transparent = True
        self.p.add_mesh(m, color=color, opacity=opacity, **kw)
        return self

    def sphere(self, c, r, color="#cccccc", **kw):
        return self.mesh(pv.Sphere(r, c, theta_resolution=48, phi_resolution=48), color, **kw)

    def line(self, pts, color=None, radius=None, closed=False, dashed=False,
             scalars=None, cmap=None, opacity=1.0, clim=None):
        pts = np.asarray(pts, float)
        if closed:
            pts = np.vstack([pts, pts[:1]])
        r = radius or self.r0()
        if dashed:
            n = max(2, len(pts) // 30)
            for k in range(0, len(pts) - n, 2 * n):
                self.mesh(pv.lines_from_points(pts[k:k + n + 1]).tube(radius=r, n_sides=16),
                          color or self.t["fg"], opacity)
            return self
        poly = pv.lines_from_points(pts)
        kw = {}
        if scalars is not None:
            poly["s"] = np.asarray(scalars)
            kw = dict(scalars="s", cmap=cmap or "plasma", clim=clim)
        return self.mesh(poly.tube(radius=r, n_sides=20), color or self.t["fg"], opacity, **kw)

    def arrow(self, start, vec, color=None, label=None, radius=None):
        start, vec = np.asarray(start, float), np.asarray(vec, float)
        L = np.linalg.norm(vec)
        if L == 0:
            return self
        r = radius or self.r0(1.6)
        tip = min(0.35, 6 * r / L)
        a = pv.Arrow(start, vec / L, tip_length=tip, tip_radius=2.6 * r / L,
                     shaft_radius=r / L, scale=L, shaft_resolution=24, tip_resolution=32)
        self.mesh(a, color or self.t["fg"])
        if label:
            self.label(start + vec * 1.08, [label], color)
        return self

    def cones(self, pts, dirs, color, size):
        """Small direction cones (arrowheads) at pts pointing along dirs."""
        if len(pts) == 0:
            return self
        pd = pv.PolyData(np.asarray(pts, float)); pd["v"] = np.asarray(dirs, float)
        g = pd.glyph(orient="v", scale=False, factor=size,
                     geom=pv.Cone(direction=(1, 0, 0), height=1.0, radius=0.38, resolution=24))
        return self.mesh(g, color)

    def label(self, pts, texts, color=None, size=None):
        self.p.add_point_labels(np.atleast_2d(np.asarray(pts, float)), [str(t) for t in texts],
                                font_size=size or self.fs, text_color=color or self.t["fg"],
                                shape=None, show_points=False, always_visible=True, bold=True)
        return self

    def axes_lines(self, L, labels=("x", "y", "z")):
        for i, n in enumerate(labels):
            d = np.zeros(3); d[i] = L
            self.line([-d, d], self.t["fg"], radius=self.r0(0.35), opacity=0.55)
            self.label([d * 1.06], [n])
        return self

    def note(self, k, v):
        self.notes.append(f"{k}: {v}")
        return self

    def view(self, v="iso", zoom=1.0):
        self._view, self._zoom = v, zoom
        return self

    # ---------- output
    def _apply_view(self):
        p, v = self.p, str(self._view)
        base, _, tilt = v.partition(":") if ":" in v else (("xz", "", v) if "," in v else (v, "", ""))
        {"xy": p.view_xy, "top": p.view_xy, "xz": p.view_xz, "front": p.view_xz,
         "yz": p.view_yz, "side": p.view_yz}.get(base, p.view_isometric)()
        if tilt:
            e, a = map(float, tilt.split(","))
            p.camera.azimuth = a; p.camera.elevation = e
        p.reset_camera(); p.camera.zoom(self._zoom)

    def _scale_bar(self):
        """Screen-parallel bar at lower-left, so its length reads correctly in any view."""
        cam = self.p.camera
        b = np.array(self.p.bounds).reshape(3, 2); c = b.mean(1)
        R = 0.5 * np.linalg.norm(b[:, 1] - b[:, 0])
        up = np.array(cam.up, float); up /= np.linalg.norm(up)
        fwd = np.array(cam.focal_point) - np.array(cam.position); fwd /= np.linalg.norm(fwd)
        rt = np.cross(fwd, up); rt /= np.linalg.norm(rt); up = np.cross(rt, fwd)
        L = nice(R / 2.2)
        o = c - up * 0.82 * R - rt * 0.85 * R; e = o + rt * L
        r = self.r0(0.5)
        self.line([o, e], self.t["fg"], radius=r)
        for q in (o, e):
            self.line([q - up * 0.03 * R, q + up * 0.03 * R], self.t["fg"], radius=r)
        self.label([o + rt * L / 2 - up * 0.07 * R], [f"{L:g} {self.unit}"])

    def export_glb(self, path, total=90000):
        """Binary glTF with true colours baked as vertex colours (incl. colormapped data)."""
        import json, struct
        blob, views, accs, meshes, nodes, mats = bytearray(), [], [], [], [], []

        def add(arr, target, comp, typ, minmax=False):
            arr = np.ascontiguousarray(arr)
            while len(blob) % 4:
                blob.append(0)
            views.append(dict(buffer=0, byteOffset=len(blob), byteLength=arr.nbytes, target=target))
            blob.extend(arr.tobytes())
            acc = dict(bufferView=len(views) - 1, componentType=comp, count=len(arr), type=typ)
            if minmax:
                acc.update(min=arr.min(0).tolist(), max=arr.max(0).tolist())
            accs.append(acc); return len(accs) - 1

        acts = [a for a in self.p.renderer.actors.values() if isinstance(a, pv.Actor)
                and a.GetVisibility() and a.mapper is not None and a.mapper.dataset is not None]
        npts = sum(a.mapper.dataset.n_points for a in acts) or 1
        for a in acts:
            ds = a.mapper.dataset
            budget = max(1500, int(total * ds.n_points / npts))
            if ds is None or ds.n_points == 0:
                continue
            ds = ds.copy()
            rgba = None
            if a.mapper.GetScalarVisibility() and a.mapper.GetLookupTable() is not None:
                try:
                    c = pv.convert_array(a.mapper.MapScalars(1.0))
                    if c is not None and len(c) == ds.n_points:
                        rgba = c[:, :4].astype(np.float32) / 255
                except Exception:
                    rgba = None
            if rgba is None:
                rgba = np.tile(np.r_[a.prop.color.float_rgb, 1.0].astype(np.float32), (ds.n_points, 1))
            ds.point_data["_rgba"] = rgba
            m = ds.extract_surface().triangulate()
            if m.n_points > budget:  # keep web files small; colours kept via attribute error
                try:
                    m = m.decimate(1 - budget / m.n_points, attribute_error=True, scalars=True)
                except Exception:
                    pass
            if m.n_cells == 0 or "_rgba" not in m.point_data:
                continue
            try:
                tri = m.regular_faces
            except Exception:
                continue
            if tri.size == 0:
                continue
            op = float(a.prop.opacity)
            col = (np.clip(np.array(m.point_data["_rgba"], np.float32), 0, 1) * 255).astype(np.uint8)
            col[:, 3] = int(op * 255)
            attrs = dict(POSITION=add(np.asarray(m.points, np.float32), 34962, 5126, "VEC3", True),
                         NORMAL=add(np.asarray(m.point_normals, np.float32), 34962, 5126, "VEC3"),
                         COLOR_0=add(col, 34962, 5121, "VEC4"))
            accs[attrs["COLOR_0"]]["normalized"] = True
            idx = add(tri.astype(np.uint32).ravel(), 34963, 5125, "SCALAR")
            mats.append(dict(pbrMetallicRoughness=dict(baseColorFactor=[1, 1, 1, op], metallicFactor=0.05,
                                                       roughnessFactor=0.4), doubleSided=True,
                             **({"alphaMode": "BLEND"} if op < 1 else {})))
            meshes.append(dict(primitives=[dict(attributes=attrs, indices=idx, material=len(mats) - 1)]))
            nodes.append(dict(mesh=len(meshes) - 1))
        while len(blob) % 4:
            blob.append(0)
        gl = dict(asset=dict(version="2.0", generator="sci3d"), scene=0,
                  scenes=[dict(nodes=list(range(len(nodes))))], nodes=nodes, meshes=meshes,
                  materials=mats, accessors=accs, bufferViews=views, buffers=[dict(byteLength=len(blob))])
        js = json.dumps(gl, separators=(",", ":")).encode()
        js += b" " * (-len(js) % 4)
        with open(path, "wb") as fh:
            fh.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
            fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
            fh.write(struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))
        return path

    def _export_html(self, path):
        """Self-contained interactive page: GLB embedded in <model-viewer> (rotate/zoom/pan)."""
        import base64, tempfile, html as H
        with tempfile.TemporaryDirectory() as d:
            g = self.export_glb(os.path.join(d, "m.glb"))
            b64 = base64.b64encode(open(g, "rb").read()).decode()
        t = H.escape(self.title or "3D diagram"); notes = "<br>".join(H.escape(n) for n in self.notes)
        bg, fg = self.t["bg"], self.t["fg"]
        open(path, "w").write(f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{t}</title>
<script type="module" src="https://unpkg.com/@google/model-viewer@3/dist/model-viewer.min.js"></script>
<style>html,body{{margin:0;height:100%;background:radial-gradient({self.t['top'] or bg},{bg});color:{fg};
font-family:system-ui,sans-serif}}model-viewer{{width:100%;height:100%}}
h1{{position:fixed;top:12px;width:100%;text-align:center;margin:0;font-weight:500;font-size:22px;pointer-events:none}}
p{{position:fixed;bottom:10px;left:14px;right:14px;font-size:12px;opacity:.75;pointer-events:none}}</style></head>
<body><h1>{t}</h1><model-viewer src="data:model/gltf-binary;base64,{b64}" camera-controls
touch-action="pan-y" exposure="1.0" shadow-intensity="0.5" environment-image="neutral"
interaction-prompt="none"></model-viewer><p>{notes}</p></body></html>""")

    def save(self, path="out.png", html=False, gltf=False, scale=1):
        """PNG always; html=True -> interactive page; gltf=True -> .glb model."""
        p = self.p
        if self.bar and self.unit:
            self._apply_view(); self._scale_bar()
        if self.title:
            p.add_text(self.title, position="upper_edge", font_size=int(self.fs * 1.25),
                       color=self.t["fg"], shadow=True)
        if self.caption:
            p.add_text(self.caption, position="lower_left", font_size=int(self.fs * 0.75),
                       color=self.t["fg"])
        self._apply_view()
        if self.transparent:
            p.enable_depth_peeling(12)
        if self.ssao and not self.transparent:
            try:
                p.enable_ssao(radius=self.scale * 0.15, kernel_size=128)
            except Exception:
                pass
        p.enable_anti_aliasing("ssaa")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        p.screenshot(path, scale=scale)
        out = [path]
        base = os.path.splitext(path)[0]
        if gltf:
            out.append(self.export_glb(base + ".glb"))
        if html:
            try:
                self._export_html(base + ".html"); out.append(base + ".html")
            except Exception as e:
                self.note("html", f"failed ({e})")
        p.close()
        print("saved: " + ", ".join(out))
        for n in self.notes:
            print(n)
        return out


# =============================================================== physics
def _trace(field, seeds, ds, nmax, stop, sign=1.0):
    """Vectorised RK4 streamline tracer along +/- normalised field. Returns list of (pts, |F|)."""
    X = np.array(seeds, float); n = len(X)
    alive = np.ones(n, bool); P = [X.copy()]; M = [np.linalg.norm(field(X), axis=1)]

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
        P.append(X.copy()); M.append(np.linalg.norm(field(X), axis=1))
        s = stop(X[idx], k)
        alive[idx[s]] = False; end[idx[s]] = k + 1
    P = np.array(P); M = np.array(M)
    return [(P[:end[i] + 1, i], M[:end[i] + 1, i]) for i in range(n)]


def efield(s, charges, lines_per_q=26, equipotential=True, plane=False, extent=None,
           cmap="plasma", planar=False):
    """charges: [(q, (x,y,z)), ...] in units of e-charge q and length units.
    Field lines start on + and end on - charges with density proportional to |q| (Gauss)."""
    Q = np.array([c[0] for c in charges], float)
    R = np.array([c[1] for c in charges], float)
    d = max(1.0, np.max(np.ptp(R, axis=0))) if len(R) > 1 else 1.0
    s.scale = max(s.scale, d)
    ext = extent or 1.6 * d + 1.0
    rc = 0.1 * d

    def E(X):
        D = X[:, None, :] - R[None]; r = np.linalg.norm(D, axis=2)
        r = np.maximum(r, 1e-6)
        return np.sum(Q[None, :, None] * D / r[..., None] ** 3, axis=1)

    def stop(X, k):
        r = np.linalg.norm(X[:, None, :] - R[None], axis=2)
        return ((r.min(1) < 0.7 * rc) & (k > 3)) | (np.abs(X).max(1) > ext)

    ds = d / 200; nmax = int(8 * ext / ds)
    lines = []
    for sign, mask in ((1, Q > 0), (-1, Q < 0)):
        if sign == -1 and Q.sum() >= 0 and (Q > 0).any():
            continue
        for q, r in zip(Q[mask], R[mask]):
            n = max(4, int(round(lines_per_q * abs(q))))
            if planar:  # textbook 2D slice: seeds on a circle in the xy-plane
                a = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / n
                sd = r + rc * np.c_[np.cos(a), np.sin(a), 0 * a]
            else:
                sd = fib_sphere(n, rc, r)
            got = _trace(E, sd, ds, nmax, stop, sign)
            if sign == -1:  # keep only lines escaping to infinity (others drawn from +)
                got = [g for g in got if np.abs(g[0][-1]).max() > ext]
            lines += got
    allm = np.concatenate([m for _, m in lines]) if lines else np.ones(1)
    clim = np.log10(np.percentile(allm, [5, 95]))
    cp, cd = [], []
    for pts, m in lines:
        s.line(pts, scalars=np.log10(m), cmap=s.cmap(cmap), clim=clim, radius=s.r0(1.1))
        i = len(pts) // 2
        if len(pts) > 4:
            cp.append(pts[i]); cd.append(pts[i + 1] - pts[i - 1])
    s.cones(cp, cd, s.t["fg"], 0.07 * d)
    for q, r in zip(Q, R):
        s.sphere(r, rc * (0.8 + 0.2 * min(abs(q), 3)), POS if q > 0 else NEG,
                 specular=0.8, specular_power=60)
        s.label([r + [0, 0, rc * 2.6]], [f"{q:+g}q"])
    if equipotential or plane:
        n = 90
        g = pv.ImageData(dimensions=(n, n, n), spacing=(2 * ext / (n - 1),) * 3,
                         origin=(-ext,) * 3)
        P = g.points
        V = np.sum(Q[None] / np.maximum(np.linalg.norm(P[:, None] - R[None], axis=2), 1e-3), 1)
        g["V"] = V
        if equipotential:
            Vq = np.abs(Q).max() / (np.array([0.3, 0.5, 0.8]) * d)
            for v in np.r_[Vq, -Vq]:
                if v > V.max() or v < V.min():
                    continue
                c = g.contour([v], scalars="V")
                if c.n_points:
                    s.mesh(c, POS if v > 0 else NEG, opacity=0.2, specular=0.3)
        if plane:
            sl = g.slice(normal="z", origin=(0, 0, 0))
            vm = np.percentile(np.abs(V), 97)
            s.mesh(sl, scalars="V", cmap="RdBu_r", clim=(-vm, vm), opacity=0.85, smooth_shading=False)
    s.note("E-field", f"{len(Q)} charges, net {Q.sum():+g}q, {len(lines)} field lines "
                      f"(density proportional to |q|; colour = log10|E|)")
    if s.title is None:
        s.title = "Electric field lines" + (" & equipotentials" if equipotential else "")
    if s._view == "iso":
        s.view("25,-35", max(s._zoom, 1.3))
    return s


class Current:
    """Conductor path (closed or open polyline) carrying current I (A)."""

    def __init__(self, path, I=1.0, seeds=None, closed=False):
        self.path = np.asarray(path, float); self.I = I; self.closed = closed
        self.seeds = seeds if seeds is not None else np.zeros((0, 3))


def wire(I=1.0, length=6.0, radii=(0.3, 0.6, 1.0, 1.5, 2.1)):
    seeds = np.array([[r, 0, z] for r in radii for z in (-1.5, 0.0, 1.5)])
    return Current([[0, 0, -length / 2], [0, 0, length / 2]], I, seeds)


def loop(R=1.0, I=1.0, n=240, planes=4):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    seeds = [[x * np.cos(a), x * np.sin(a), 0] for a in np.linspace(0, np.pi, planes, endpoint=False)
             for x in np.r_[np.linspace(-0.85, -0.15, 4), np.linspace(0.15, 0.85, 4)] * R]
    return Current(np.c_[R * np.cos(t), R * np.sin(t), 0 * t], I, np.array(seeds), closed=True)


def solenoid(R=1.0, length=4.0, turns=12, I=1.0, planes=3):
    t = np.linspace(0, 2 * np.pi * turns, turns * 48)
    path = np.c_[R * np.cos(t), R * np.sin(t), length * (t / t[-1] - 0.5)]
    seeds = [[x * np.cos(a), x * np.sin(a), 0] for a in np.linspace(0, np.pi, planes, endpoint=False)
             for x in np.r_[np.linspace(-0.8, -0.1, 4), np.linspace(0.1, 0.8, 4)] * R]
    return Current(path, I, np.array(seeds))


def bfield(s, currents, cmap="viridis", max_len=None):
    """Exact Biot-Savart (finite straight segments) for any list of Current objects."""
    MU = 1e-7  # mu0/4pi
    segs = []
    for c in currents:
        P = np.vstack([c.path, c.path[:1]]) if c.closed else c.path
        segs.append((P[:-1], P[1:], np.full(len(P) - 1, c.I)))
    A = np.concatenate([a for a, _, _ in segs]); Bv = np.concatenate([b for _, b, _ in segs])
    Iv = np.concatenate([i for _, _, i in segs])
    allp = np.concatenate([c.path for c in currents])
    L = max(np.max(np.ptp(allp, 0)), 1.0)
    s.scale = max(s.scale, L)

    def B(X):
        out = np.zeros_like(X)
        for k in range(0, len(A), 400):
            a = A[None, k:k + 400] - X[:, None]; b = Bv[None, k:k + 400] - X[:, None]
            na = np.linalg.norm(a, axis=2); nb = np.linalg.norm(b, axis=2)
            den = na * nb * (na * nb + np.sum(a * b, 2)) + 1e-12
            f = MU * Iv[None, k:k + 400] * (na + nb) / den
            out += np.sum(f[..., None] * np.cross(a, b), 1)
        return out

    ext = 1.4 * L
    seeds = np.concatenate([c.seeds for c in currents])
    ds = L / 220; nmax = int((max_len or 5 * L) / ds)

    def stop(X, k):
        return np.abs(X).max(1) > ext

    fw = _trace(B, seeds, ds, nmax, stop, 1)
    lines = []
    for (pts, m), sd in zip(fw, seeds):
        dd = np.linalg.norm(pts - sd, axis=1)
        # closed-loop detection: first return to seed after leaving it
        far = np.where(dd > 10 * ds)[0]
        if len(far):
            back = np.where((dd < 1.2 * ds) & (np.arange(len(dd)) > far[0]))[0]
            if len(back):
                k = back[0]; lines.append((np.vstack([pts[:k + 1], pts[:1]]), np.r_[m[:k + 1], m[:1]]))
                continue
        bw = _trace(B, [sd], ds, nmax, stop, -1)[0]
        lines.append((np.vstack([bw[0][::-1], pts[1:]]), np.r_[bw[1][::-1], m[1:]]))
    # de-duplicate lines that retrace each other
    keep = []
    for pts, m in lines:
        if all(np.min(np.linalg.norm(q[::5, None] - pts[None, ::5], axis=2), 1).mean() > 2 * ds
               for q, _ in keep):
            keep.append((pts, m))
    allm = np.concatenate([m for _, m in keep])
    clim = np.log10(np.percentile(allm, [5, 95]))
    cp, cd = [], []
    for pts, m in keep:
        s.line(pts, scalars=np.log10(m), cmap=s.cmap(cmap), clim=clim, radius=s.r0(0.9))
        for i in (len(pts) // 4, 3 * len(pts) // 4):
            if 1 <= i < len(pts) - 1:
                cp.append(pts[i]); cd.append(pts[i + 1] - pts[i - 1])
    s.cones(cp, cd, s.t["fg"], 0.045 * L)
    for c in currents:
        s.mesh(pv.lines_from_points(np.vstack([c.path, c.path[:1]]) if c.closed else c.path)
            .tube(radius=s.r0(2.2), n_sides=24), "#c87533", pbr=True, metallic=0.9, roughness=0.3)
        n = len(c.path); idx = np.linspace(0, n - 2, 6 if c.closed else 3).astype(int)[1:]
        d = (c.path[idx + 1] - c.path[idx]) * np.sign(c.I)
        s.cones(c.path[idx] , d, "#ffd166", 0.07 * L)
    s.note("B-field", f"{len(currents)} conductor(s), {len(keep)} field lines, exact "
                      f"Biot-Savart; colour = log10|B|; yellow cones = current direction")
    if s.title is None:
        s.title = "Magnetic field lines (Biot-Savart)"
    s.unit = s.unit or "m"
    s.view("15,-30" if s._view == "iso" else s._view)
    return s


def wave(s, sources=((-1, 0), (1, 0)), wavelength=0.6, extent=5.0, mode="instant",
         height=0.6, n=400, phases=None):
    """Superposition of 2D circular waves A cos(kr+phi)/sqrt(kr) from point sources."""
    S = np.asarray(sources, float); k = 2 * np.pi / wavelength
    ph = np.zeros(len(S)) if phases is None else np.asarray(phases, float)
    x = np.linspace(-extent, extent, n)
    Xg, Yg = np.meshgrid(x, x)
    F = np.zeros_like(Xg, complex)
    for (sx, sy), p in zip(S, ph):
        r = np.hypot(Xg - sx, Yg - sy)
        F += np.exp(1j * (k * r + p)) / np.sqrt(k * r + 1.0)
    Z = F.real if mode == "instant" else np.abs(F) ** 2
    Zn = Z / np.abs(Z).max()
    g = pv.StructuredGrid(Xg, Yg, height * extent * 0.25 * Zn); g["A"] = Zn.ravel(order="F")
    s.scale = max(s.scale, 2 * extent)
    v = np.percentile(np.abs(Zn), 97)
    s.mesh(g, scalars="A", cmap="RdBu_r" if mode == "instant" else "inferno",
           clim=(-v, v) if mode == "instant" else (0, v), specular=0.6, specular_power=40)
    for sx, sy in S:
        s.sphere((sx, sy, height * extent * 0.3), 0.045 * extent, "#ffd166", ambient=0.5)
    s.unit = s.unit or "units"
    if len(S) == 2:
        dsep = np.linalg.norm(S[1] - S[0])
        s.note("fringes", f"d={dsep:g}, λ={wavelength:g}; maxima at d·sinθ = mλ, "
                          f"m ≤ {int(dsep // wavelength)}")
    s.title = s.title or ("Two-source interference" if len(S) == 2 else "Wave superposition")
    s.view("40,-30" if s._view == "iso" else s._view)
    return s


def kepler(s, a=1.0, e=0.6, M=1.0, n_marks=16, sectors=True):
    """Exact Keplerian ellipse (AU, years, solar masses); marks at equal time steps."""
    b = a * np.sqrt(1 - e ** 2); T = np.sqrt(a ** 3 / M)
    E = np.linspace(0, 2 * np.pi, 600)
    orbit = np.c_[a * (np.cos(E) - e), b * np.sin(E), 0 * E]

    def pos(Mean):
        Ek = Mean.copy()
        for _ in range(50):
            Ek -= (Ek - e * np.sin(Ek) - Mean) / (1 - e * np.cos(Ek))
        return np.c_[a * (np.cos(Ek) - e), b * np.sin(Ek), 0 * Ek]

    s.scale = max(s.scale, 2 * a)
    s.line(orbit, "#9fb4d9", radius=s.r0(0.5))
    s.mesh(pv.Sphere(0.07 * a, (0, 0, 0), theta_resolution=60, phi_resolution=60), "#ffcc33",
           ambient=0.9, diffuse=0.3, specular=0.0)
    marks = pos(np.linspace(0, 2 * np.pi, n_marks, endpoint=False))
    for p in marks:
        s.sphere(p, 0.022 * a, "#4cc9f0")
    if sectors:
        w = 2 * np.pi / n_marks
        for M0, col in ((-w / 2, "#f72585"), (np.pi - w / 2, "#7bd389")):
            arc = pos(np.linspace(M0, M0 + w, 40))
            fan = pv.PolyData(np.vstack([[0, 0, 0], arc]),
                              faces=np.hstack([[3, 0, i, i + 1] for i in range(1, 40)]))
            s.mesh(fan, col, opacity=0.55, smooth_shading=False)
    rp, ra = a * (1 - e), a * (1 + e)
    s.label([[rp * 1.05, 0, 0.05 * a], [-ra * 1.02, 0, 0.05 * a]],
            [f"perihelion {rp:.3g} AU", f"aphelion {ra:.3g} AU"])
    s.unit = s.unit or "AU"
    s.note("orbit", f"a={a} AU, e={e}, b={b:.4g} AU, T={T:.4g} yr; dots = equal time steps; "
                    f"shaded sectors = equal areas (Kepler II)")
    s.title = s.title or "Kepler orbit - equal areas in equal times"
    s.view("55,-15" if s._view == "iso" else s._view)
    return s


def projectile(s, v0=30.0, angle=45.0, k=0.0, g=9.81, arrows=8):
    """2D projectile, quadratic drag a = g - k|v|v (k = rho*Cd*A/2m, 1/m). RK4."""
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
        P = np.array(P); f = P[-2, 1] / (P[-2, 1] - P[-1, 1])
        P[-1] = P[-2] + f * (P[-1] - P[-2])
        return P, np.array(Vs), (len(P) - 2 + f) * dt

    P, Vs, T = run(k)
    R, H = P[-1, 0], P[:, 1].max()
    span = max(R, 1.0)
    if k > 0:
        P0, _, T0 = run(0)
        s.line(np.c_[P0[:, 0], 0 * P0[:, 0], P0[:, 1]], "#9aa5b1", radius=span * 0.003, dashed=True)
        s.note("no-drag", f"range {P0[-1,0]:.4g} m, h_max {P0[:,1].max():.4g} m, T {T0:.4g} s")
    s.scale = max(s.scale, span)
    s.line(np.c_[P[:, 0], 0 * P[:, 0], P[:, 1]], "#ff9f1c", radius=span * 0.005)
    vmax = np.linalg.norm(Vs, axis=1).max()
    for i in np.linspace(0, len(P) - 2, arrows).astype(int):
        v = Vs[i] / vmax * span * 0.16
        s.arrow([P[i, 0], 0, P[i, 1]], [v[0], 0, v[1]], "#2ec4b6", radius=span * 0.0045)
    gnd = pv.Plane((R / 2, 0, -0.002 * span), (0, 0, 1), R * 1.3, span * 0.35)
    s.mesh(gnd, "#55606e", opacity=0.6, specular=0.1)
    s.label([[P[np.argmax(P[:, 1]), 0], 0, H * 1.08]], [f"h_max {H:.3g} m"])
    s.label([[R, 0, 0.04 * span]], [f"R {R:.3g} m"])
    s.unit = s.unit or "m"
    s.note("projectile", f"v0={v0} m/s @ {angle}°, k={k} 1/m: range {R:.4g} m, h_max {H:.4g} m, T {T:.4g} s")
    s.title = s.title or ("Projectile with quadratic drag" if k else "Projectile motion")
    s.view("xz" if s._view == "iso" else s._view)
    s.bar_axis = "z"
    return s


_NS = {k: getattr(np, k) for k in ("sin", "cos", "tan", "exp", "log", "sqrt", "abs", "arctan2",
                                   "sinh", "cosh", "tanh", "pi", "e", "hypot", "where", "maximum",
                                   "minimum", "arctan", "arcsin", "arccos", "log10", "sign")}


def surface(s, expr, xr=(-3, 3), yr=(-3, 3), n=300, cmap="viridis", contours=12, zscale=1.0):
    """z = f(x,y) from a numpy expression string (or callable)."""
    x = np.linspace(*xr, n); y = np.linspace(*yr, n); X, Y = np.meshgrid(x, y)
    Z = expr(X, Y) if callable(expr) else eval(expr, {"__builtins__": {}}, dict(_NS, x=X, y=Y))
    Z = np.broadcast_to(np.asarray(Z, float), X.shape) * zscale
    g = pv.StructuredGrid(X, Y, Z); g["z"] = Z.ravel(order="F")
    s.scale = max(s.scale, max(np.ptp(x), np.ptp(y)))
    s.mesh(g, scalars="z", cmap=cmap, specular=0.5)
    if contours:
        c = g.contour(contours, scalars="z")
        if c.n_points:
            s.mesh(c.tube(radius=s.r0(0.25)), s.t["fg"], opacity=0.6)
    s.note("surface", f"z = {expr if isinstance(expr, str) else 'f(x,y)'}; z in [{Z.min():.4g}, {Z.max():.4g}]")
    s.title = s.title or f"z = {expr}" if isinstance(expr, str) else s.title
    s.view("35,-40" if s._view == "iso" else s._view)
    return s


def vfield(s, fx, fy, fz="0", extent=2.0, n=9, cmap="cividis", stream=True):
    """Vector field F=(fx,fy,fz) from numpy expressions in x,y,z: arrows + streamlines."""
    t = np.linspace(-extent, extent, n)
    g = pv.ImageData(dimensions=(n, n, n), spacing=(t[1] - t[0],) * 3, origin=(-extent,) * 3)
    P = g.points; ns = dict(_NS, x=P[:, 0], y=P[:, 1], z=P[:, 2])
    F = np.c_[[np.broadcast_to(eval(e, {"__builtins__": {}}, ns), (len(P),)) for e in (fx, fy, fz)]].T
    m = np.linalg.norm(F, axis=1); g["F"] = F; g["m"] = m
    s.scale = max(s.scale, 2 * extent)
    gl = g.glyph(orient="F", scale="m", factor=0.9 * (t[1] - t[0]) / max(m.max(), 1e-12),
                 geom=pv.Arrow(tip_length=0.3, tip_radius=0.12, shaft_radius=0.04))
    s.mesh(gl, scalars="m", cmap=cmap)
    if stream:
        fine = pv.ImageData(dimensions=(40,) * 3, spacing=(2 * extent / 39,) * 3, origin=(-extent,) * 3)
        Q = fine.points; ns = dict(_NS, x=Q[:, 0], y=Q[:, 1], z=Q[:, 2])
        fine["F"] = np.c_[[np.broadcast_to(eval(e, {"__builtins__": {}}, ns), (len(Q),)) for e in (fx, fy, fz)]].T
        st = fine.streamlines_from_source(pv.PolyData(fib_sphere(24, extent * 0.6)), vectors="F",
                                          max_length=8 * extent, integration_direction="both")
        if st.n_points:
            s.mesh(st.tube(radius=s.r0(0.25)), "#ffd166", opacity=0.45)
    s.note("vfield", f"F=({fx}, {fy}, {fz}); |F| in [{m.min():.3g}, {m.max():.3g}]")
    s.title = s.title or f"F = ({fx}, {fy}, {fz})"
    return s


# =============================================================== chemistry
from ase.data import atomic_numbers, chemical_symbols, covalent_radii, vdw_radii, atomic_masses
from ase.data.colors import jmol_colors

COMMON = {  # common name -> SMILES (forced)
    "water": "O", "ammonia": "N", "methane": "C", "ethane": "CC", "propane": "CCC",
    "butane": "CCCC", "ethylene": "C=C", "ethene": "C=C", "acetylene": "C#C", "ethyne": "C#C",
    "methanol": "CO", "ethanol": "CCO", "formaldehyde": "C=O", "acetone": "CC(C)=O",
    "acetic acid": "CC(=O)O", "carbon dioxide": "O=C=O", "carbon monoxide": "[C-]#[O+]",
    "nitrogen": "N#N", "oxygen": "O=O", "hydrogen": "[H][H]", "ozone": "[O-][O+]=O",
    "hydrogen peroxide": "OO", "hydrogen chloride": "Cl", "hcl": "Cl", "urea": "NC(N)=O",
    "benzene": "c1ccccc1", "toluene": "Cc1ccccc1", "phenol": "Oc1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1", "cyclohexane": "C1CCCCC1", "pyridine": "c1ccncc1",
    "sulfuric acid": "OS(=O)(=O)O", "nitric acid": "O[N+](=O)[O-]", "phosphoric acid": "OP(=O)(O)O",
    "glycine": "NCC(=O)O", "alanine": "C[C@H](N)C(=O)O", "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O",
    "fructose": "OC[C@H]1O[C@](O)(CO)[C@@H](O)[C@@H]1O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "nicotine": "CN1CCC[C@H]1c1cccnc1", "dopamine": "NCCc1ccc(O)c(O)c1",
    "serotonin": "NCCc1c[nH]c2ccc(O)cc12", "adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1",
    "adenine": "Nc1ncnc2[nH]cnc12", "guanine": "Nc1nc2[nH]cnc2c(=O)[nH]1",
    "cytosine": "Nc1cc[nH]c(=O)n1", "thymine": "Cc1c[nH]c(=O)[nH]c1=O", "uracil": "O=c1cc[nH]c(=O)[nH]1",
    "cholesterol": "C[C@H](CCCC(C)C)[C@H]1CC[C@@H]2[C@@]1(CC[C@H]3[C@H]2CC=C4[C@@]3(CC[C@@H](C4)O)C)C",
    "sucrose": "OC[C@H]1O[C@@](CO)(O[C@H]2O[C@H](CO)[C@@H](O)[C@H](O)[C@H]2O)[C@@H](O)[C@@H]1O",
    "vitamin c": "OC[C@H](O)[C@H]1OC(=O)C(O)=C1O", "citric acid": "OC(=O)CC(O)(CC(=O)O)C(=O)O",
    "sf6": "FS(F)(F)(F)(F)F", "methylamine": "CN", "chloroform": "ClC(Cl)Cl",
}
IONIC = {"Li": 0.76, "Na": 1.02, "K": 1.38, "Cs": 1.74, "Mg": 0.72, "Ca": 1.12,
         "F": 1.33, "Cl": 1.81, "O": 1.40}  # Shannon (1976) radii, Å, at the CN of these lattices
NONMETALS = set("H He B C N O F Ne Si P S Cl Ar Ge As Se Br Kr Sb Te I Xe".split())


def _elem_rgb(Z):
    return tuple(jmol_colors[Z])


def _radius(Z, style):
    v = vdw_radii[Z]; c = covalent_radii[Z]
    v = v if np.isfinite(v) else c * 1.25 + 0.4
    return {"spacefill": v, "stick": 0.18}.get(style, 0.22 * v + 0.05)


def load_molecule(spec, optimize=True):
    """spec: common name | ASE formula (H2O, C60, CH3CH2OH...) | SMILES | 'smiles:...' | file.
    Returns symbols, positions (Å), bonds [(i,j,order)], source string."""
    key = spec.strip().lower()
    if key in COMMON:
        spec = "smiles:" + COMMON[key]
    if spec.startswith("smiles:") or not (os.path.exists(spec) or _is_ase_name(spec)):
        return _from_smiles(spec.split("smiles:", 1)[-1], optimize)
    if os.path.exists(spec):
        from ase.io import read
        at = read(spec); src = f"file {os.path.basename(spec)}"
    else:
        from ase.build import molecule
        at = molecule(spec); src = "ASE reference geometry (G2/expt.)"
    sym, pos = at.get_chemical_symbols(), at.get_positions()
    return sym, pos, _distance_bonds(sym, pos), src


def _is_ase_name(s):
    try:
        from ase.build import molecule
        molecule(s); return True
    except Exception:
        return False


def _from_smiles(smi, optimize=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        raise SystemExit(f"Could not parse '{smi}' as name, formula, file or SMILES. "
                         f"Use 'smiles:<SMILES>' or a .xyz/.pdb/.mol/.cif file.")
    m = Chem.AddHs(m)
    ps = AllChem.ETKDGv3(); ps.randomSeed = 42
    if AllChem.EmbedMolecule(m, ps) != 0:
        ps.useRandomCoords = True; AllChem.EmbedMolecule(m, ps)
    ff = "ETKDG"
    if optimize:
        try:
            if AllChem.MMFFHasAllMoleculeParams(m):
                AllChem.MMFFOptimizeMolecule(m, maxIters=5000); ff = "ETKDG+MMFF94"
            else:
                AllChem.UFFOptimizeMolecule(m, maxIters=5000); ff = "ETKDG+UFF"
        except Exception:
            pass
    pos = m.GetConformer().GetPositions()
    sym = [a.GetSymbol() for a in m.GetAtoms()]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), b.GetBondTypeAsDouble()) for b in m.GetBonds()]
    return sym, pos, bonds, f"SMILES {smi} -> {ff}"


def _distance_bonds(sym, pos, tol=1.15):
    Z = np.array([atomic_numbers[x] for x in sym]); r = covalent_radii[Z]
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    i, j = np.where(np.triu(D < tol * (r[:, None] + r[None]), 1))
    return [(a, b, 1.0) for a, b in zip(i, j)]


def _orient(pos):
    """Centre and rotate so principal axes = x (longest), y, z (thinnest)."""
    c = pos - pos.mean(0)
    if len(c) < 3:
        return c
    w, v = np.linalg.eigh(np.cov(c.T))
    out = c @ v[:, ::-1]
    return out if np.linalg.det(v[:, ::-1]) > 0 else out * [1, 1, -1]


def draw_atoms(s, sym, pos, bonds=(), style="ballstick", labels=False, measure=False,
               radii=None):
    """radii: optional {symbol: physical radius Å} (ionic/metallic) overriding vdW."""
    Z = np.array([atomic_numbers[x] for x in sym])
    if radii:
        k = {"spacefill": 1.0, "stick": None}.get(style, 0.42)
        rad = np.array([radii[e] * k if k else 0.18 for e in sym])
    else:
        rad = np.array([_radius(z, style) for z in Z])
    sph = pv.Sphere(1.0, theta_resolution=40, phi_resolution=40)
    for z in np.unique(Z):
        idx = Z == z
        pd = pv.PolyData(pos[idx]); pd["r"] = rad[idx]
        s.mesh(pd.glyph(scale="r", geom=sph, orient=False), _elem_rgb(z),
               specular=0.55, specular_power=45, ambient=0.12, diffuse=0.85)
    if style != "spacefill" and len(bonds):
        br = 0.18 if style == "stick" else 0.11
        nb = {i: set() for i in range(len(pos))}
        for i, j, _ in bonds:
            nb[i].add(j); nb[j].add(i)
        segs = {}
        for i, j, o in bonds:
            a, b = pos[i], pos[j]; u = b - a; L = np.linalg.norm(u); u /= L
            # offset direction: towards a neighbour, perpendicular to bond (planar for pi bonds)
            ref = next((pos[k] for k in (nb[i] - {j}) | (nb[j] - {i})), a + [0, 0, 1] if abs(u[2]) < .9 else a + [1, 0, 0])
            pdir = (ref - a) - np.dot(ref - a, u) * u
            pdir = pdir / (np.linalg.norm(pdir) or 1)
            n = int(round(o)) if o != 1.5 else 2
            w = 0.6 if n == 3 else 0.55
            offs = {1: [0], 2: [-0.5, 0.5], 3: [-1, 0, 1]}[min(n, 3)]
            rr = br if n == 1 else br * 0.62
            if o == 1.5:  # aromatic: full bond + thin inner bond
                offs, rr = [0, 1.0], br * 0.75
            for k, off in enumerate(offs):
                dlt = pdir * off * w * br * (2.4 if o == 1.5 else 3.2)
                r_this = rr if not (o == 1.5 and k == 1) else br * 0.45
                mid = (a + b) / 2 + dlt
                aa, bb = a + dlt + u * (0.3 * rad[i] if o == 1.5 and k else 0), b + dlt - u * (0.3 * rad[j] if o == 1.5 and k else 0)
                for end, at in ((aa, Z[i]), (bb, Z[j])):
                    segs.setdefault((at, round(r_this, 4)), []).append((end, mid))
        for (z, r), ss in segs.items():
            pts = np.array([p for seg in ss for p in seg])
            lines = np.hstack([[2, 2 * k, 2 * k + 1] for k in range(len(ss))])
            s.mesh(pv.PolyData(pts, lines=lines).tube(radius=r, n_sides=28), _elem_rgb(z),
                   specular=0.5, specular_power=40)
    if labels:
        show = [i for i in range(len(sym)) if sym[i] != "H" or len(sym) <= 12]
        s.label(pos[show] + [0, 0, 0], [sym[i] for i in show])
    if measure and len(bonds):
        sel = [(i, j) for i, j, _ in bonds if len(sym) <= 14 or ("H" not in (sym[i], sym[j]))]
        s.label([(pos[i] + pos[j]) / 2 for i, j in sel],
                [f"{np.linalg.norm(pos[i] - pos[j]):.3f}" for i, j in sel], size=int(s.fs * 0.7))
    return s


def molecule(s, spec, style="ballstick", labels=False, measure=False, optimize=True):
    sym, pos, bonds, src = load_molecule(spec, optimize)
    pos = _orient(np.asarray(pos, float))
    s.scale = max(s.scale, np.ptp(pos, 0).max() + 2)
    s.ssao = True
    draw_atoms(s, sym, pos, bonds, style, labels, measure)
    Z = [atomic_numbers[x] for x in sym]
    from collections import Counter
    cnt = Counter(sym); order = sorted(cnt, key=lambda e: (e != "C", e != "H", e))
    formula = "".join(f"{e}{cnt[e] if cnt[e] > 1 else ''}" for e in order)
    mw = sum(atomic_masses[z] for z in Z)
    bl = {}
    for i, j, o in bonds:
        k = "".join(sorted((sym[i], sym[j]))) + {1: "", 1.5: "(ar)", 2: "=", 3: "#"}.get(o, "")
        bl.setdefault(k, []).append(np.linalg.norm(pos[i] - pos[j]))
    s.note("molecule", f"{formula}, M={mw:.2f} g/mol, {len(sym)} atoms; geometry: {src}")
    if bl:
        s.note("bonds (Å)", ", ".join(f"{k} {np.mean(v):.3f}" for k, v in sorted(bl.items())))
    s.unit = s.unit or "Å"
    s.title = s.title or (spec.split("smiles:")[-1] if len(spec) < 30 else formula)
    if s._view == "iso":
        s.view("xy:-28,18", max(s._zoom, 1.15))
    return s


CRYSTALS = {  # name: (formula, structure, kwargs) ; a,c in Å (experimental, ~300 K)
    "nacl": ("NaCl", "rocksalt", dict(a=5.640)), "mgo": ("MgO", "rocksalt", dict(a=4.212)),
    "kcl": ("KCl", "rocksalt", dict(a=6.293)), "licl": ("LiCl", "rocksalt", dict(a=5.130)),
    "cscl": ("CsCl", "cesiumchloride", dict(a=4.123)),
    "zns": ("ZnS", "zincblende", dict(a=5.409)), "gaas": ("GaAs", "zincblende", dict(a=5.653)),
    "caf2": ("CaF2", "fluorite", dict(a=5.463)),
    "diamond": ("C", "diamond", dict(a=3.567)), "si": ("Si", "diamond", dict(a=5.431)),
    "ge": ("Ge", "diamond", dict(a=5.658)),
    "cu": ("Cu", "fcc", dict(a=3.615)), "al": ("Al", "fcc", dict(a=4.050)),
    "au": ("Au", "fcc", dict(a=4.078)), "ag": ("Ag", "fcc", dict(a=4.086)), "ni": ("Ni", "fcc", dict(a=3.524)),
    "fe": ("Fe", "bcc", dict(a=2.867)), "na": ("Na", "bcc", dict(a=4.291)), "w": ("W", "bcc", dict(a=3.165)),
    "cr": ("Cr", "bcc", dict(a=2.910)), "k": ("K", "bcc", dict(a=5.328)),
    "mg": ("Mg", "hcp", dict(a=3.209, c=5.211)), "zn": ("Zn", "hcp", dict(a=2.665, c=4.947)),
    "ti": ("Ti", "hcp", dict(a=2.951, c=4.686)), "co": ("Co", "hcp", dict(a=2.507, c=4.070)),
}


def crystal(s, name, repeat=(1, 1, 1), style="ballstick", bonds=None, labels=False, cubic=True):
    """Crystal from preset, ASE element default, or .cif/.vasp file; shows unit cell(s)."""
    from ase.build import bulk, graphene
    key = name.lower().replace(" ", "")
    if os.path.exists(name):
        from ase.io import read
        at = read(name); src = f"file {os.path.basename(name)}"
    elif key == "graphene":
        at = graphene(a=2.46, vacuum=0); at.cell[2, 2] = 3.35; src = "graphene a=2.46 Å"
    elif key in CRYSTALS:
        f, st, kw = CRYSTALS[key]
        at = bulk(f, st, cubic=cubic and st not in ("hcp",), **kw); src = f"{st}, " + ", ".join(f"{k}={v} Å" for k, v in kw.items())
    else:
        at = bulk(name, cubic=cubic); src = "ASE reference lattice"
    cell = np.array(at.cell); frac = at.get_scaled_positions(wrap=True)
    sym0 = at.get_chemical_symbols(); rep = np.array(repeat if np.ndim(repeat) else (repeat,) * 3)
    sym, pos = [], []
    pbc = np.array(at.pbc, bool)  # boundary images only along periodic axes (2D sheets stay 1 layer)
    for sh in np.ndindex(*(rep + pbc)):
        for f, e in zip(frac, sym0):
            fr = f + sh
            if np.all(np.where(pbc, fr <= rep + 1e-6, fr < rep - 1e-6)):
                sym.append(e); pos.append(fr @ cell)
    pos = np.array(pos)
    ctr = (rep / 2) @ cell
    pos = pos - ctr
    elems = set(sym0)
    if bonds is None:
        bonds = not (len(elems) == 1 and elems.isdisjoint(NONMETALS))
    bl = _distance_bonds(sym, pos, 1.12) if bonds else []
    if bonds and len(elems) > 1:  # ionic/polar: only hetero-atom contacts
        bl = [b for b in bl if sym[b[0]] != sym[b[1]]] or bl
    s.scale = max(s.scale, np.ptp(pos, 0).max())
    s.ssao = True
    D0 = np.linalg.norm(pos[:, None] - pos[None], axis=2); np.fill_diagonal(D0, np.inf)
    if elems <= set(IONIC):
        radii, rk = {e: IONIC[e] for e in elems}, "Shannon ionic radii"
    elif len(elems) == 1:
        radii = {e: D0.min() / 2 for e in elems}
        rk = ("covalent" if elems & NONMETALS else "metallic") + " radius = d_nn/2"
    else:
        radii, rk = None, "covalent/vdW radii"
    draw_atoms(s, sym, pos, bl, style, labels, radii=radii)
    s.note("radii", rk + ("" if not radii else ": " + ", ".join(f"{e} {r:.3f} Å" for e, r in radii.items())))
    # unit-cell edges for every repeated cell
    corners = np.array(list(np.ndindex(*(1 + pbc.astype(int)))))  # flat outline for 2D sheets
    nc = len(corners)
    edges = [(i, j) for i in range(nc) for j in range(i + 1, nc) if np.abs(corners[i] - corners[j]).sum() == 1]
    pts, ln = [], []
    for sh in np.ndindex(*rep):
        c = (corners + sh) @ cell - ctr
        for i, j in edges:
            ln.append([2, len(pts), len(pts) + 1]); pts += [c[i], c[j]]
    s.mesh(pv.PolyData(np.array(pts), lines=np.hstack(ln)).tube(radius=0.03, n_sides=12),
           s.t["fg"], opacity=0.7)
    vol = abs(np.linalg.det(cell)); mass = sum(atomic_masses[atomic_numbers[e]] for e in sym0)
    from ase.neighborlist import neighbor_list  # periodic (bulk) neighbours, not the finite cluster
    ii, dd = neighbor_list("id", at, 7.0)
    dmin = dd[ii == 0].min(); cn = int(np.sum(dd[ii == 0] < dmin * 1.05))
    s.note("crystal", f"{at.get_chemical_formula()} {src}; atoms/cell {len(sym0)}, "
                      f"density {mass * 1.66054 / vol:.4g} g/cm3, nearest neighbour {dmin:.3f} Å, CN {cn}")
    s.unit = s.unit or "Å"
    s.title = s.title or f"{name} crystal structure"
    s.view("20,-30" if s._view == "iso" else s._view)
    return s


ORB = {"s": {"": 0}, "p": {"z": 0, "x": 1, "y": -1},
       "d": {"z2": 0, "xz": 1, "yz": -1, "x2-y2": 2, "x2y2": 2, "xy": -2},
       "f": {"z3": 0, "xz2": 1, "yz2": -1, "z(x2-y2)": 2, "zx2-zy2": 2, "xyz": -2,
             "x(x2-3y2)": 3, "x3": 3, "y(3x2-y2)": -3, "y3": -3}}


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


def psi(n, l, m, X, Y, Z, Zn=1):
    """Real hydrogen-like wavefunction (atomic units, r in bohr)."""
    from scipy.special import genlaguerre, factorial
    try:
        from scipy.special import sph_harm_y
        Ylm = lambda ll, mm, th, ph: sph_harm_y(ll, mm, th, ph)
    except ImportError:
        from scipy.special import sph_harm
        Ylm = lambda ll, mm, th, ph: sph_harm(mm, ll, ph, th)
    r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2); th = np.arccos(np.clip(Z / np.where(r == 0, 1, r), -1, 1))
    ph = np.arctan2(Y, X); rho = 2 * Zn * r / n
    N = np.sqrt((2 * Zn / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    R = N * np.exp(-rho / 2) * rho ** l * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    if m == 0:
        Yr = Ylm(l, 0, th, ph).real
    elif m > 0:
        Yr = np.sqrt(2) * (-1) ** m * Ylm(l, m, th, ph).real
    else:
        Yr = np.sqrt(2) * (-1) ** m * Ylm(l, -m, th, ph).imag
    return R * Yr


def orbital(s, name="2p_z", Zn=1, prob=0.9, style="surface", cut=None, res=120, dots=90000):
    """Hydrogen-like orbital: exact psi, isosurface enclosing `prob` of |psi|^2 (Å units)."""
    from scipy.special import genlaguerre, factorial
    n, l, m = parse_orbital(name) if isinstance(name, str) else name
    # extent from radial distribution (99.7 %)
    r = np.linspace(0, 60 * n * n / Zn, 20000)
    pr = (r ** 2) * psi(n, l, 0, 0 * r, 0 * r, r, Zn) ** 2
    cdf = np.cumsum(pr); cdf /= cdf[-1]
    R = r[np.searchsorted(cdf, 0.997)] * 1.05
    t = np.linspace(-R, R, res); h = t[1] - t[0]
    X, Y, Zg = np.meshgrid(t, t, t, indexing="ij")
    P = psi(n, l, m, X, Y, Zg, Zn)
    d = np.sort((P ** 2).ravel())[::-1]; c = np.cumsum(d) * h ** 3; tot = c[-1]
    thr = d[np.searchsorted(c, prob * tot)]; iso = np.sqrt(thr)
    g = pv.ImageData(dimensions=(res,) * 3, spacing=(h * BOHR,) * 3, origin=(-R * BOHR,) * 3)
    g["psi"] = P.ravel(order="F"); g["p"] = (P ** 2).ravel(order="F") / d[0]
    radial_nodes = n - l - 1
    cut = (radial_nodes > 0) if cut is None else cut
    s.scale = max(s.scale, 2 * R * BOHR)
    Rb = R * BOHR
    if style == "dots":  # Monte-Carlo sample of |psi|^2: each dot = one electron position draw
        rng = np.random.default_rng(1)
        pr = (P ** 2).ravel(); pr = pr / pr.sum()
        k = rng.choice(pr.size, size=dots, p=pr)
        pts = np.c_[np.unravel_index(k, P.shape)].astype(float) * h - R
        pts = (pts + rng.uniform(-h / 2, h / 2, pts.shape)) * BOHR
        sg = np.sign(P.ravel()[k])
        for sgn, col in ((1, POS), (-1, NEG)):
            q = pts[sg == sgn]
            if cut:
                q = q[~((q[:, 0] > 0) & (q[:, 1] > 0) & (q[:, 2] > 0))]
            if len(q):
                s.p.add_mesh(pv.PolyData(q), color=col, point_size=max(3, s.size[1] // 300),
                             render_points_as_spheres=True, opacity=0.55)
        s.note("dots", f"{dots} samples drawn from |psi|^2")
    else:
        for v, col in ((iso, POS), (-iso, NEG)):
            surf = g.contour([v], scalars="psi")
            if not surf.n_points:
                continue
            surf = surf.smooth_taubin(n_iter=30)
            if cut:  # remove the octant facing the camera to expose inner shells/lobes
                surf = surf.clip_box((0, Rb, 0, Rb, 0, Rb), invert=True)
            s.mesh(surf, col, specular=0.7, specular_power=50, ambient=0.12)
    s.sphere((0, 0, 0), 0.03 * R * BOHR, "#ffd166", ambient=0.6)
    s.axes_lines(R * BOHR * (0.6 if style == "dots" else 0.9))
    rmean = (3 * n * n - l * (l + 1)) / (2 * Zn) * BOHR
    lbl = name if isinstance(name, str) else f"n={n} l={l} m={m}"
    s.note("orbital", f"{lbl}: n={n}, l={l}, m(real)={m}, Z={Zn}; radial nodes {radial_nodes}, "
                      f"angular nodes {l}; <r>={rmean:.3f} Å; surface encloses {prob:.0%} probability "
                      f"(|psi|={iso:.3e} a0^-3/2); red +, blue - phase")
    s.unit = s.unit or "Å"
    s.title = s.title or (f"H {lbl} orbital - electron density" if style == "dots"
                          else f"H {lbl} orbital ({prob:.0%} surface)")
    if s._view == "iso" and not cut:  # in-plane (xy) orbitals: look from above, others: side-tilt
        s.view("xy:-40,15" if (l >= 2 and abs(m) == l) else "xz:25,-30")
    return s


# =============================================================== CLI
def _common(sp):
    sp.add_argument("-o", "--out", default=None, help="output .png (default <command>.png)")
    sp.add_argument("--title"); sp.add_argument("--caption")
    sp.add_argument("--theme", default="dark", choices=list(THEMES))
    sp.add_argument("--size", default="1600x1200", help="WxH pixels")
    sp.add_argument("--view", default="iso", help="iso|xy|xz|yz|'elev,azim'")
    sp.add_argument("--zoom", type=float, default=1.0)
    sp.add_argument("--no-scale", action="store_true", help="hide scale bar")
    sp.add_argument("--html", action="store_true", help="also export interactive .html")
    sp.add_argument("--gltf", action="store_true", help="also export .glb 3D model (Blender/PowerPoint/web)")


def _floats(t):
    return [float(v) for v in t.replace(" ", "").split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sci3d", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("efield", help="point charges: field lines + equipotentials")
    a.add_argument("charges", nargs="+", help="q@x,y,z  e.g. +1@-1,0,0 -1@1,0,0")
    a.add_argument("--lines", type=int, default=26, help="lines per unit charge")
    a.add_argument("--no-equi", action="store_true"); a.add_argument("--plane", action="store_true")
    a.add_argument("--planar", action="store_true", help="field lines in xy-plane only (textbook)")
    a = sub.add_parser("bfield", help="magnetic field: wire | loop | solenoid | helmholtz")
    a.add_argument("kind", choices=["wire", "loop", "solenoid", "helmholtz", "antihelmholtz"])
    a.add_argument("--I", type=float, default=1.0); a.add_argument("--R", type=float, default=1.0)
    a.add_argument("--length", type=float, default=4.0); a.add_argument("--turns", type=int, default=12)
    a = sub.add_parser("wave", help="interference of circular waves")
    a.add_argument("--sources", nargs="+", default=["-1,0", "1,0"], help="x,y ...")
    a.add_argument("--wavelength", type=float, default=0.6); a.add_argument("--extent", type=float, default=5)
    a.add_argument("--mode", choices=["instant", "intensity"], default="instant")
    a.add_argument("--height", type=float, default=0.6)
    a = sub.add_parser("orbit", help="Kepler orbit")
    a.add_argument("--a", type=float, default=1.0); a.add_argument("--e", type=float, default=0.6)
    a.add_argument("--M", type=float, default=1.0); a.add_argument("--marks", type=int, default=16)
    a = sub.add_parser("projectile", help="projectile with optional quadratic drag")
    a.add_argument("--v0", type=float, default=30); a.add_argument("--angle", type=float, default=45)
    a.add_argument("--k", type=float, default=0.0, help="drag rho*Cd*A/(2m) [1/m]")
    a.add_argument("--g", type=float, default=9.81)
    a = sub.add_parser("surface", help="z=f(x,y) potential/energy surface")
    a.add_argument("expr"); a.add_argument("--x", default="-3,3"); a.add_argument("--y", default="-3,3")
    a.add_argument("--cmap", default="viridis"); a.add_argument("--contours", type=int, default=12)
    a.add_argument("--zscale", type=float, default=1.0)
    a = sub.add_parser("vfield", help="vector field from expressions in x,y,z")
    a.add_argument("fx"); a.add_argument("fy"); a.add_argument("fz", nargs="?", default="0")
    a.add_argument("--extent", type=float, default=2.0); a.add_argument("--n", type=int, default=9)
    a = sub.add_parser("molecule", help="molecule from name/formula/SMILES/file")
    a.add_argument("spec"); a.add_argument("--style", default="ballstick", choices=["ballstick", "spacefill", "stick"])
    a.add_argument("--labels", action="store_true"); a.add_argument("--measure", action="store_true")
    a.add_argument("--no-opt", action="store_true")
    a = sub.add_parser("crystal", help="crystal lattice from preset/element/.cif")
    a.add_argument("name"); a.add_argument("--repeat", default="1,1,1")
    a.add_argument("--style", default="ballstick", choices=["ballstick", "spacefill", "stick"])
    a.add_argument("--bonds", choices=["auto", "on", "off"], default="auto"); a.add_argument("--labels", action="store_true")
    a.add_argument("--primitive", action="store_true")
    a = sub.add_parser("orbital", help="hydrogen-like orbital e.g. 1s 2p_z 3d_z2 4f_xyz")
    a.add_argument("name"); a.add_argument("--Z", type=int, default=1)
    a.add_argument("--prob", type=float, default=0.9); a.add_argument("--style", choices=["surface", "dots"], default="surface")
    a.add_argument("--cut", choices=["auto", "on", "off"], default="auto"); a.add_argument("--res", type=int, default=120)
    sub.add_parser("list", help="list built-in molecules, crystals, orbitals")
    for k, v in sub.choices.items():
        if k != "list":
            _common(v)
    argv = sys.argv[1:] if argv is None else argv
    # values that start with '-' (e.g. -1@1,0,0  -2,2  -y) are not options: protect them
    flags = {o for p in [ap, *sub.choices.values()] for a in p._actions for o in a.option_strings}
    argv = [" " + t if t.startswith("-") and t.split("=")[0] not in flags else t for t in argv]
    A = ap.parse_args(argv)
    if A.cmd == "list":
        print("molecules:", ", ".join(sorted(COMMON)), "\n  + any ASE formula (H2O, C60, CH3COOH...), SMILES, .xyz/.pdb/.mol/.sdf")
        print("crystals:", ", ".join(sorted(CRYSTALS)), ", graphene\n  + any element symbol or .cif file")
        print("orbitals:", "; ".join(f"{k}: {', '.join(x or '-' for x in v)}" for k, v in ORB.items()))
        return
    W, H = map(int, A.size.lower().split("x"))
    s = Scene(A.title, A.caption, A.theme, (W, H)); s._view, s._zoom = A.view, A.zoom
    c = A.cmd
    if c == "efield":
        ch = []
        for t in A.charges:
            q, p = t.split("@"); ch.append((float(q), _floats(p)))
        efield(s, ch, A.lines, not A.no_equi, A.plane, planar=A.planar)
    elif c == "bfield":
        if A.kind == "wire":
            cur = [wire(A.I, A.length * 1.5)]
        elif A.kind == "loop":
            cur = [loop(A.R, A.I)]
        elif A.kind == "solenoid":
            cur = [solenoid(A.R, A.length, A.turns, A.I)]
        else:
            l1, l2 = loop(A.R, A.I, planes=3), loop(A.R, A.I if A.kind == "helmholtz" else -A.I, planes=3)
            l1.path[:, 2] += A.R / 2; l2.path[:, 2] -= A.R / 2
            l2.seeds = np.zeros((0, 3))
            zs = (0.0,) if A.kind == "helmholtz" else (A.R / 2, -A.R / 2)  # anti: seed in both coil planes
            l1.seeds = np.array([[x * np.cos(t), x * np.sin(t), z0] for z0 in zs
                                 for t in (0, np.pi / 3, 2 * np.pi / 3)
                                 for x in np.r_[np.linspace(-0.9, -0.1, 5), np.linspace(0.1, 0.9, 5)] * A.R])
            cur = [l1, l2]
        bfield(s, cur)
    elif c == "wave":
        wave(s, [_floats(x) for x in A.sources], A.wavelength, A.extent, A.mode, A.height)
    elif c == "orbit":
        kepler(s, A.a, A.e, A.M, A.marks)
    elif c == "projectile":
        projectile(s, A.v0, A.angle, A.k, A.g)
    elif c == "surface":
        surface(s, A.expr, _floats(A.x), _floats(A.y), cmap=A.cmap, contours=A.contours, zscale=A.zscale)
    elif c == "vfield":
        vfield(s, A.fx, A.fy, A.fz, A.extent, A.n)
    elif c == "molecule":
        molecule(s, A.spec, A.style, A.labels, A.measure, not A.no_opt)
    elif c == "crystal":
        rp = _floats(A.repeat); rp = [int(x) for x in (rp * 3 if len(rp) == 1 else rp)]
        crystal(s, A.name, rp, A.style, {"auto": None, "on": True, "off": False}[A.bonds], A.labels, not A.primitive)
    elif c == "orbital":
        orbital(s, A.name, A.Z, A.prob, A.style, {"auto": None, "on": True, "off": False}[A.cut], A.res)
    s.bar = not A.no_scale
    s.save(A.out or f"{c}.png", A.html, A.gltf)


if __name__ == "__main__":
    main()
