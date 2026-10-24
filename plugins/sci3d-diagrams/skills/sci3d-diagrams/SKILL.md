---
name: sci3d-diagrams
description: Exam-paper style, scientifically accurate 2D black-and-white physics and chemistry diagrams rendered in pure Python (matplotlib + RDKit + ASE) — electric field lines and equipotentials, parallel plates, magnetic fields of wires/coils/solenoids/bar magnets, lens and mirror ray diagrams, transverse/longitudinal/standing waves, two-source interference, Kepler orbits, projectiles, graphs, skeletal and displayed formulae, VSEPR shapes with wedge/dash bonds, dot-and-cross diagrams, crystal layers and plan views, orbital cross-sections and radial distribution graphs. Use for any request to draw, sketch or illustrate a physics or chemistry diagram for a worksheet, exam, test, textbook or printed document.
---

# Exam-style 2D science diagrams (black and white)

Every diagram is flat 2D line art in black on white with a serif font, like the figures in an exam paper. Never produce 3D renders, colour or shading gradients with this skill.

All geometry is **computed**, never hand-placed:
- E-fields come from Coulomb's law. Parallel plates use a finite-difference Laplace solve, so edge fringing is real.
- B-fields use the exact flux function (elliptic integrals) for loops, solenoids and magnets, and A_z for straight wires.
- Ray diagrams come from 1/u + 1/v = 1/f. Kepler orbits come from Kepler's equation. Projectiles use RK4.
- Orbitals use exact hydrogen wavefunctions. Molecules use RDKit 2D depiction and VSEPR electron counting. Crystals use experimental lattice constants.

## Token rules (important)
- **Do not read `scripts/exam2d.py`.** Everything you need is on this page. Run `--help` on one subcommand only if a flag is missing here.
- Use one shell command per diagram. Each run prints `saved:` plus verified facts (image distance, B at centre, bond angle, density, nodes…). Quote these instead of re-deriving them.
- View the PNG once to check it. Re-render only if something is actually wrong.

## Setup (once per session)
```bash
pip install -q matplotlib scipy rdkit ase 2>/dev/null || pip install -q --break-system-packages matplotlib scipy rdkit ase
S="${CLAUDE_SKILL_DIR}/scripts/exam2d.py"   # Claude Code expands this; elsewhere use this skill's folder path
```

## Commands — `python $S <cmd> ... -o out.png`
| Command | Example | Key options |
|---|---|---|
| `efield` | `efield +1@-1,0 -1@1,0` · `efield +2@-1,0 -1@1,0` | `q@x,y` per charge; `--lines 12` (per unit charge, so number ∝ \|q\|); `--no-equi` (hide dashed equipotentials) |
| `plates` | `plates --V 100 --d 0.02 --L 0.06` | `--lines 24` (seeded ∝ surface charge); prints E at centre vs V/d |
| `bfield` | `bfield solenoid` · `bfield wire --into` · `bfield wires --wires 1@-1,0 -1@1,0` | `wire wires loop solenoid helmholtz antihelmholtz magnet`; `--I --R --length --turns --lines`; ⊙ = out of page, ⊗ = into page |
| `rays` | `rays convex-lens --f 10 --u 30 --h 4` | `convex-lens concave-lens concave-mirror convex-mirror`; three principal rays; virtual rays/images dashed; prints v, m, nature |
| `wave` | `wave transverse` · `wave longitudinal` · `wave standing --ends closed-open --harmonic 3` · `wave interference --d 2 --wavelength 0.6` | transverse: `--A --wavelength --length --time --grid --xlabel --ylabel`; standing: `--ends fixed-fixed\|open-open\|closed-open --harmonic --length --speed`; interference: `--extent --nodal` |
| `orbit` | `orbit --a 1 --e 0.6` | `--M --marks`; hatched equal-area sectors (Kepler II) |
| `projectile` | `projectile --v0 30 --angle 45 --k 0.004` | `--k` quadratic drag (1/m); no-drag path dashed |
| `graph` | `graph "5*x" "20-2*x" --x 0,10 --xlabel "t / s" --ylabel "v / m s\$^{-1}\$" --labels A,B` | numpy expressions in `x`; `--y lo,hi --grid` (graph paper) `--no-ticks` (sketch graph) |
| `contour` | `contour "x**2 - y**2"` | `--x --y --levels`; negative levels dashed |
| `vfield` | `vfield " -y" x` | `--extent --n --stream` |
| `molecule` | `molecule aspirin` · `molecule ethanoic\ acid --style displayed` · `molecule water --style shape` · `molecule co2 --style dotcross` | `--style skeletal\|displayed\|shape\|dotcross`; shape: `--no-lp --angle "107°"`; skeletal: `--label-c` |
| `crystal` | `crystal NaCl` · `crystal CsCl --view plan` | `--view layer\|plan` (plan = unit cell projected on the base, fractional heights written beside atoms); `--repeat 2 --style ball\|space` |
| `orbital` | `orbital 3d_z2` · `orbital 3p_x --style contours` | `--style boundary\|contours\|dots`; `--phase hatch\|shade\|none`; `--plane xz\|yz\|xy\|diag`; `--no-nodes`; `--Z` |
| `radial` | `radial 1s 2s 3s` | `--func P\|R` (r²R² or R); `--unit A\|a0`; prints most probable r and node radii |
| `list` | `list` | built-in molecule names, crystals and orbitals |

**Common flags (all commands):**
- `-o out.png`
- `--fig "Fig. 2.1"` adds a centred figure number under the diagram, as in an exam paper. `--caption` adds text after it.
- `--title` adds a title above. Exam figures normally have none.
- `--svg` / `--pdf` also write a vector copy for print or Word.
- `--size 6x4.5` (inches), `--dpi 300`, `--font serif|sans`.

**Molecule input:**
- A common name: `water`, `ammonia`, `ethanol`, `propanone`, `sf6`, `ammonium`… Run `list` to see them all.
- Any SMILES string, e.g. `CC(=O)O` or `smiles:CCO`.
- A `.mol` file.

`shape` and `dotcross` need one central atom, either AXnEm or a diatomic. Use `skeletal` or `displayed` for anything bigger. Ions such as `[NH4+]` work with `shape`, which draws them in brackets with the charge. `dotcross` accepts neutral molecules only.

**Crystal presets** use experimental a and c: NaCl MgO KCl LiCl CsCl ZnS GaAs CaF2 diamond Si Cu Al Au Ag Fe Na W Mg Zn graphene. Any other element symbol, or a `.cif` file, also works.

## Exam conventions used (state when relevant)
- **Field lines** carry arrowheads mid-line. Line density reflects field strength:
  - E-lines start on + charges and end on − charges, with the number of lines ∝ |q|.
  - For wires and coils there is equal magnetic flux between neighbouring B-lines.
  - Dashed lines are equipotentials.
- **Ray diagrams:**
  - Real rays are solid and virtual rays are dashed. A virtual image is a dashed arrow.
  - Lenses are drawn as a line with arrowheads: pointing outward for converging, inward for diverging.
  - Mirrors are hatched on the back.
  - The vertical scale may be stretched to fill the figure. This keeps every ray construction exact.
- **Waves:** N = node and A = antinode. For pipes these are displacement nodes and antinodes. C = compression and R = rarefaction. In the interference figure, solid arcs are crests, dashed arcs are troughs and thick lines are maxima (path difference mλ).
- **VSEPR shapes:**
  - Wedge = bond towards the viewer, hashed = away. Lone pairs are drawn as lobes with two dots.
  - The labelled angle is experimental where known (H₂O 104.5°, NH₃ 106.7°, CH₄ 109.5°…). Otherwise it is the ideal VSEPR angle, with `<` where lone pairs compress it.
- **Dot-and-cross:** outer shells only. The central atom's electrons are dots and the other atoms' electrons are crosses.
- **Orbitals:**
  - The figure is a cross-section through the nucleus. The boundary is the section of the 3D surface that encloses 90% of |ψ|².
  - The negative phase is hatched, or the positive phase is shaded with `--phase shade`. Dashed lines are nodes.
  - Length unit is Å (a₀ = 0.529 Å).
- **Crystals:** In the layer view, filled and open circles distinguish the species. Sizes are to scale for ionic radii (Shannon), with `--style space` showing true size and `ball` showing 0.32×. In the plan view, the numbers are fractional heights z/c.

## Python API (composite or custom figures)
```python
import sys; sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts"); from exam2d import *
f = Fig(figno="Fig. 3.1")                      # black-and-white, equal aspect, no axes
efield(f, [(+1, (-1, 0)), (-1, (1, 0))])       # any domain function: (f, ...)
f.arrow((0, 1.2), (0.8, 0), label="F")         # force / velocity arrows
f.leader((1, 0), (1.8, 0.9), "point P")        # exam-style labelling line
f.dim((-1, -1.3), (1, -1.3), "d")              # dimension line with arrowheads
f.text((0, 1.5), "X"); f.line(pts, dashed=True); f.circle(c, r); f.current(c, r, out=True)
f.save("out.png", svg=True)
```
Mechanics helpers:
- `f.rot_arrow(c, r, a0, a1, label)` draws a curved ω arrow. Angles are in degrees; `a1 < a0` makes it clockwise, and `rx=`/`ry=` give an elliptical arc.
- `f.angle(v, a0, a1, r, "θ")` draws an angle arc with a label. `f.right_angle(p, d1, d2, s)` draws a right-angle box.
- `f.surface(p0, p1, side)` draws a hatched wall or ground. `f.poly(pts, dashed=True, hatch="///")` draws a polygon.

Slide output: `Fig(font="sans", px=(1800, 1280), transparent=True)` gives an exact pixel size and a transparent background. Add `ink="white", paper="#141d35"` for dark slides; `paper` fills shapes and label boxes. A worked set of ten mechanics figures is in `examples/a4-rigid-body/make_figures.py`.
Domain functions: `efield plates bfield rays wave_transverse wave_longitudinal wave_standing interference kepler projectile graph contour vfield structure shape dotcross crystal orbital radial`. Use `f.plot_axes(xlim, ylim, "x / m", "y / m", grid=True)` for a graph with arrowed axes.

**Circuits:** use `schemdraw`, which is black and white by default (`pip install schemdraw`):
```python
import schemdraw, schemdraw.elements as e
schemdraw.config(font="serif")
with schemdraw.Drawing(file="circuit.png", dpi=300, show=False) as d:
    d += (b := e.Battery().up().label("12 V"))
    d += e.Resistor().right().label("4.0 Ω"); d += e.Lamp().down(); d += e.Line().left().to(b.start)
```

## Troubleshooting
- `Could not parse …` → pass a SMILES string (`smiles:...`) or a `.mol` file.
- `shape needs a molecule … with one central atom` → use `--style skeletal` or `--style displayed`.
- Negative numbers in arguments (`-1@1,0`, `-y`) are handled automatically. Quote expressions that contain spaces.
