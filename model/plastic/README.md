# Elasto-plasticity

Six models demonstrating permanent plastic deformation across joint and tendon springs (`springyield`) and 2D shell and
3D continuum flexes (`yield`). Except for [`clay.xml`](clay.xml), each scene places an elastic reference (`yield = 0`)
next to an elasto-plastic counterpart under the same load. Plastic state lives in `mjData.plastic`, so resetting the
simulation restores the undeformed reference geometry.

<p float="left">
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/wire.xml" title="Open wire.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/wire_v2.webp" width="32%"></a>
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/paper.xml" title="Open paper.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/paper_v2.webp" width="32%"></a>
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/foil.xml" title="Open foil.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/foil.webp" width="32%"></a>
</p>
<p float="left">
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/clay.xml" title="Open clay.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/clay.webp" width="32%"></a>
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/roller.xml" title="Open roller.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/roller_v2.webp" width="32%"></a>
  <a href="https://live.mujoco.org/?model=github:google-deepmind/mujoco/main/model/plastic/torsion.xml" title="Open torsion.xml in live.mujoco.org"><img src="https://www.gstatic.com/mujoco/model/plastic/torsion_v2.webp" width="32%"></a>
</p>

Click an animation to open the model in the browser viewer at [live.mujoco.org](https://live.mujoco.org).

| Model | Formulation | Yield parameter | DOFs | Plastic state | Timestep |
| --- | --- | --- | --- | --- | --- |
| [`wire.xml`](wire.xml) | Hinge joints & spatial tendons | `springyield="2.5"`, `"0.5"` (joints); `"0.085"`, `"0.042"` (tendons) | 54 | 28 | 2 ms |
| [`paper.xml`](paper.xml) | 2D shell (`elastic2d="bend"`, edge equalities) | `yield="0.00001"` | 1736 | 800 | 4 ms |
| [`foil.xml`](foil.xml) | 2D shell (`elastic2d="both"`) | `yield="0.00016 0.0016"` (stretch, bend) | 792 | 1320 | 3 ms |
| [`clay.xml`](clay.xml) | 3D tetrahedral grid (23 × 23 × 5) | `yield="0.0030"` | 6349 | 69696 | 3 ms |
| [`roller.xml`](roller.xml) | 3D trilinear cells (4 × 6 × 2 vs. 6 × 10 × 2) | `yield="0.003"` | 794 | 720 | 3 ms |
| [`torsion.xml`](torsion.xml) | 3D trilinear cells (2 × 2 × 8) | `yield="0.025"` | 438 | 192 | 3 ms |

## Models

* **1D joint and tendon springs ([`wire.xml`](wire.xml)):** Three 12-segment cantilevered wires (spring steel, brass,
  soft copper) and three matching suspension cables are loaded by falling spheres that roll away after impact. When
  spring deflection exceeds `springyield`, the rest configuration shifts to clamp the elastic force at the yield limit.
* **Bend-only inextensible shell ([`paper.xml`](paper.xml)):** Two 17 × 17 sheets are pulled by their centers through
  brass rings (`solver="Newton"`). `<edge equality="true"/>` keeps membrane edges inextensible while `elastic2d="bend"`
  provides bending stiffness; the elastic cloth unfolds after clearing the ring, whereas the paper napkin retains radial
  creases.
* **Separate stretching and bending yield ([`foil.xml`](foil.xml)):** Three steel spheres drop onto tilted 13 × 13
  sheets (`elastic2d="both"`). Passing two thresholds in `yield="0.00016 0.0016"` gives membrane stretching a 10×
  smaller yield strain than bending so impacts leave crisp craters without runaway bending drift. Two non-shadow
  spotlights placed at the mirror image of the default camera cast specular glints that distort around each dent.
* **3D continuum stamping ([`clay.xml`](clay.xml)):** A spring-loaded hinged press stamps a cross (`+`) die into a
  23 × 23 × 5 tetrahedral block anchored at its base. Volumetric–deviatoric (von Mises J₂) return mapping preserves
  volume under plastic shear, leaving a permanent impression when the press rebounds.
* **Cogged stamping roller ([`roller.xml`](roller.xml)):** A rolling axle carrying two four-toothed wheels traverses an
  elastic trilinear slab (`4 × 6 × 2` cells) and a plastic trilinear slab (`6 × 10 × 2` cells), stamping spaced dents
  into the plastic block while the elastic block rebounds flat.
* **Torsion twist and release ([`torsion.xml`](torsion.xml)):** Eccentric escapement wrenches twist two `2 × 2 × 8`
  trilinear bars past 180° before slipping off their drive pins, leaving an ~82° permanent twist in the plastic bar
  while the elastic bar untwists to zero.

## Modeling notes

* **Trilinear cells with subdivided skin:** Interpolated flexes (`dof="trilinear"`) decouple simulation DOFs
  (`cellcount`) from the visual and collision surface mesh (`count`). Setting `count = 2 * cellcount + 1` in
  [`roller.xml`](roller.xml) and [`torsion.xml`](torsion.xml) smooths surface shading and contact geometry at zero DOF
  cost.

## Changelog

* 04-11-2026: Initial release.
