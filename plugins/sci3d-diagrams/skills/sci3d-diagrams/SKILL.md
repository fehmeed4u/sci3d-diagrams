---
name: sci3d-diagrams
description: Realistic, scientifically accurate 3D physics and chemistry diagrams rendered in pure Python (PyVista + ASE + RDKit) — electric/magnetic field lines, waves, orbits, projectiles, potential surfaces, vector fields, molecules, crystal lattices and atomic orbitals. Use for any request to draw, render, visualise or illustrate a physics or chemistry concept in 3D, or for PNG / interactive HTML / GLB science figures.
---

# sci3d — accurate 3D physics & chemistry diagrams

All geometry is **computed** (Coulomb, exact Biot–Savart, Kepler's equation, RK4, exact hydrogen wavefunctions, MMFF-optimised molecules, experimental lattice constants). Never hand-place scientific geometry.

## Token rules (important)
- **Do not read `scripts/sci3d.py`** — everything needed is on this page. Run `--help` on one subcommand only if a flag is missing here.
- One shell command per diagram. Each run prints `saved:` plus verified facts (bond lengths, density, range, nodes…) — quote these instead of re-deriving.
- View the PNG once to check it. Re-render only if something is actually wrong.

## Setup (once per session)
```bash
pip install -q pyvista ase rdkit scipy matplotlib 2>/dev/null || pip install -q --break-system-packages pyvista ase rdkit scipy matplotlib
S="${CLAUDE_SKILL_DIR}/scripts/sci3d.py"   # Claude Code expands this; elsewhere use this skill's folder path
```
Rendering is off-screen. If you see a blank image or an OpenGL error, prefix the command with `xvfb-run -a`.

## Commands — `python $S <cmd> ... -o out.png`
| Command | Example | Key options |
|---|---|---|
| `orbital` | `orbital 3d_z2` · `orbital 2p_x --Z 2` | names: 1s 2s 2p_x/y/z 3d_z2/xz/yz/xy/x2-y2 4f_xyz/z3/… ; `--style surface\|dots` (dots = Monte-Carlo electron density) ; `--prob 0.9` ; `--cut on\|off` (auto octant cutaway when radial nodes exist) |
| `molecule` | `molecule caffeine` · `molecule CCO` · `molecule smiles:CO` · `molecule C60` · `molecule file.xyz` | `--style ballstick\|spacefill\|stick` `--labels` `--measure` (bond lengths Å) |
| `crystal` | `crystal NaCl --repeat 2` · `crystal graphene --repeat 4,4,1` · `crystal my.cif` | `--style …` `--bonds auto\|on\|off` `--labels` `--primitive` |
| `efield` | `efield +1@-1,0,0 -1@1,0,0` | `--planar` (textbook 2D) `--plane` (potential heat-map) `--no-equi` `--lines 26` (per unit charge) |
| `bfield` | `bfield loop` · `bfield solenoid --turns 15` | `wire\|loop\|solenoid\|helmholtz\|antihelmholtz` `--I --R --length --turns` |
| `wave` | `wave --sources -1,0 1,0 --wavelength 0.6` | `--mode instant\|intensity` `--extent` `--height` |
| `orbit` | `orbit --a 1 --e 0.6` | `--M` (solar masses) `--marks` (equal-time dots) — shows Kepler's 2nd law sectors |
| `projectile` | `projectile --v0 30 --angle 45 --k 0.004` | `--k` quadratic drag ρC_dA/2m (1/m), `--g`; no-drag path shown dashed |
| `surface` | `surface "x**2 - y**2" --x -2,2 --y -2,2` | numpy expression; `--cmap --contours --zscale` |
| `vfield` | `vfield -y x 0.3` | `--extent --n` |
| `list` | `list` | built-in molecules, crystals, orbital names |

Molecule lookup order: built-in name (≈60 common molecules) → file → ASE formula (H2O, CH4, C60, C6H6…) → SMILES. Prefix `smiles:` to force SMILES (e.g. `smiles:CO` = methanol, while `CO` = carbon monoxide).
Crystal presets (experimental a, c): NaCl MgO KCl LiCl CsCl ZnS GaAs CaF2 diamond Si Ge Cu Al Au Ag Ni Fe Na W Cr K Mg Zn Ti Co graphene. Any other element uses ASE reference data.

**Common flags (all commands):** `-o out.png` · `--title` `--caption` · `--theme dark|light|white` (white = print/paper) · `--size 1600x1200` · `--view iso|xy|xz|yz|"xy:elev,azim"` · `--zoom 1.2` · `--no-scale` · `--html` (self-contained interactive page) · `--gltf` (.glb model for Blender/PowerPoint/web)

## Python API (composite scenes)
Use this only when one command isn't enough, for example to combine elements or add arrows and labels.
```python
import sys; sys.path.insert(0, "${CLAUDE_SKILL_DIR}/scripts"); from sci3d import *
s = Scene("Title", caption="source / equation", theme="dark", unit="m")   # unit drives the scale bar
efield(s, [(+1,(0,0,0)), (-1,(3,0,0))], planar=True)       # also: bfield(s,[loop(R,I), wire(I), solenoid(...), Current(path,I,seeds)])
s.arrow(start, vector, "#7bd389", label="F")               # size arrows relative to the scene (here ~3 units)
s.label([(x,y,z)], ["text"]); s.sphere(c, r, color); s.line(pts, color, dashed=True)
s.view("xy").save("out.png", html=False, gltf=False)
```
Domain functions: `efield bfield wave kepler projectile surface vfield molecule crystal orbital`, all called as `(s, …)`. Low level: `load_molecule(spec)` → symbols, positions (Å), bonds; `draw_atoms(s, sym, pos, bonds)`; `psi(n,l,m,X,Y,Z)` gives the real hydrogen-like wavefunction.

## Accuracy conventions (state these when relevant)
- **Orbitals:** the surface encloses the stated fraction of |ψ|² (default 90%). Red is + phase, blue is − phase. Length unit is Å (a₀ = 0.529 Å).
- **E-fields:** the number of lines from each charge is proportional to |q| (Gauss's law). Lines start on + charges and end on − charges or at infinity. Line colour shows log₁₀|E|.
- **B-fields:** exact finite-segment Biot–Savart. Yellow cones show current direction. Line colour shows log₁₀|B|.
- **Molecules:** geometry from ETKDG + MMFF94 (or UFF), or from ASE experimental data. Bond orders are drawn as single, double, triple or aromatic. Colours are Jmol CPK.
- **Crystals:** experimental lattice constants. Ionic crystals use Shannon ionic radii, and metals use d_nn/2. The printed density and coordination number come from the periodic lattice.
- **Kepler orbit:** dots mark equal time intervals. The shaded sectors have equal area.
- **Projectile:** RK4 integration. Range, h_max and flight time are printed.

## Troubleshooting
- `Could not parse …` → use `smiles:<SMILES>` or a structure file.
- Long titles can clip at the edges → shorten them with `--title`.
- Exported HTML files are 3–8 MB because the 3D model is embedded. The PNG is the main deliverable.
