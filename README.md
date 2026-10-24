# sci3d-diagrams

A Claude skill for **scientifically accurate 2D physics and chemistry diagrams in black and white**, drawn in the style of an exam paper. It is written in pure Python with matplotlib, RDKit and ASE.

All geometry is computed rather than drawn by hand:
- Electric fields come from Coulomb's law. Parallel plates use a Laplace solve, so edge fringing is real.
- Magnetic fields of wires, loops, solenoids and bar magnets use the exact flux function.
- Ray diagrams come from the lens/mirror equation. Kepler orbits come from Kepler's equation. Projectiles use RK4.
- Orbitals use exact hydrogen wavefunctions.
- Molecules use RDKit 2D depiction and VSEPR electron counting. Crystals use experimental lattice constants.

Output is flat line art: black on white, serif labels, arrowheads on field lines, hatching instead of colour, and optional figure numbers such as "Fig. 1.1". Every diagram is saved as a 300 dpi PNG, with optional SVG or PDF for print.

![gallery](docs/gallery.png)

## What it draws

| Physics | Chemistry |
|---|---|
| E-field lines and equipotentials, parallel plates | skeletal and displayed formulae |
| B-fields: wires (⊙/⊗), loop, solenoid, Helmholtz, bar magnet | VSEPR shapes with wedge/hashed bonds and lone pairs |
| ray diagrams: converging/diverging lens, concave/convex mirror | dot-and-cross diagrams |
| transverse, longitudinal and standing waves; two-source interference | crystal layers and plan views (fractional heights) |
| Kepler orbit, projectile with drag, graphs, contours, vector fields | orbital cross-sections, radial distribution graphs |

## Install in Claude Code

**Option A: plugin (recommended, easy to update)**

```bash
claude plugin marketplace add <your-github-user>/sci3d-diagrams
claude plugin install sci3d-diagrams@sci3d
```

**Option B: plain skill (copy the folder)**

```bash
git clone https://github.com/<your-github-user>/sci3d-diagrams.git
mkdir -p ~/.claude/skills
cp -r sci3d-diagrams/plugins/sci3d-diagrams/skills/sci3d-diagrams ~/.claude/skills/
```

Dependencies (installed by the skill on first use):

```bash
pip install matplotlib scipy rdkit ase
```

## Use

Ask Claude for a diagram in plain language. For example: "draw the field lines between two opposite charges for my worksheet", "ray diagram for a convex lens with the object beyond 2F", "VSEPR shape of ammonia", "dot-and-cross diagram for CO2" or "standing wave in a closed pipe, third harmonic".

You can also run the script directly:

```bash
S=plugins/sci3d-diagrams/skills/sci3d-diagrams/scripts/exam2d.py
python $S efield +1@-1,0 -1@1,0 --fig "Fig. 1.1" -o dipole.png
python $S bfield solenoid --turns 12 -o solenoid.png
python $S rays convex-lens --f 10 --u 30 -o lens.png
python $S wave standing --ends closed-open --harmonic 3 -o pipe.png
python $S molecule water --style shape -o water.png
python $S molecule co2 --style dotcross -o co2.png
python $S crystal NaCl --view plan --svg -o nacl.png
python $S orbital 3d_z2 -o dz2.png
python $S list
```

See [SKILL.md](plugins/sci3d-diagrams/skills/sci3d-diagrams/SKILL.md) for every command and option.

## License

MIT
