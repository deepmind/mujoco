---
name: mujoco-spec-editing
description: >-
  Procedural model authoring, editing, and tuning with MjSpec (Python and C++),
  bodies, geoms, joints, sites, cameras, sub-model attachment with prefix
  namespaces, actuators (position servos, motors, velocity drives, trntype),
  baseline option/default overrides (implicitfast, elliptic cones, solref,
  solimp, armature, damping), collision bitmasks (contype, conaffinity),
  friction (condim), default classes (MjsDefault), procedural meshes, deletion,
  spec recompilation (spec.recompile). Use for authoring or tuning MJCF/MjSpec
  models, adding bodies/geoms/actuators, and attaching sub-models.
  Do NOT use for stepping simulations (mujoco-python) or rendering (mujoco-rendering).
---

# MuJoCo Spec Editing: Building & Modifying Models

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`):** Read short chapters like
> [`../../programming/modeledit.rst`](../../programming/modeledit.rst)
> (`.. _meOverview:`) and [`../../overview.rst`](../../overview.rst) directly
> when in a MuJoCo checkout; for large reference files like
> [`../../modeling.rst`](../../modeling.rst) (`.. _ProceduralModeling:`,
> `.. _Mechanisms:`, `.. _CTuning:`),
> [`../../XMLreference.rst`](../../XMLreference.rst),
> [`../../python.rst`](../../python.rst) (`.. _PyModelEdit:`), and
> [`../../APIreference/functions.rst`](../../APIreference/functions.rst), grep
> the literal `.. _Label:` anchor. Outside a checkout, the same `doc/*.rst` tree
> ships offline in `pip install mujoco` at
> `pathlib.Path(mujoco.__file__).parent / 'doc'`; otherwise fetch
> `https://mujoco.readthedocs.io/en/latest/<path>.html` (or `/en/stable/` for
> released pages; `functions.rst` renders inside
> `APIreference/APIfunctions.html`, and Sphinx anchors lowercase the label,
> e.g. `.. _meOverview:` → `#meoverview`) and browse repo paths (`../../../`)
> under `https://github.com/google-deepmind/mujoco/tree/main/`.
>
> **Related skills:**
>
> -   [mujoco-python](../python/SKILL.md) — Simulation lifecycle, stepping, and
>     diagnostics.
> -   [mujoco-rendering](../rendering/SKILL.md) — Offscreen image rendering and
>     camera setup.
> -   [mujoco-studio](../studio/SKILL.md) — Interactive viewers and Studio
>     plugins.

## 1. Procedural Modeling with MjSpec

`MjSpec` is a mutable C++ and Python data structure that mirrors the MJCF XML
schema, allowing models to be constructed and modified programmatically.

```python
import mujoco

spec = mujoco.MjSpec()

# Add a body with a freejoint and sphere geom
body = spec.worldbody.add_body(name='ball', pos=[0, 0, 1.0])
body.add_freejoint()
body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.1], rgba=[1, 0, 0, 1])

# Compile to MjModel
model = spec.compile()
```

In C++:
```cpp
#include <mujoco/mujoco.h>

mjSpec* spec = mj_makeSpec();
mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), nullptr);
body->pos[2] = 1.0;
mjs_addFreeJoint(body);
mjsGeom* geom = mjs_addGeom(body, nullptr);
geom->type = mjGEOM_SPHERE;
geom->size[0] = 0.1;

mjModel* model = mj_compile(spec, nullptr);
if (!model) printf("%s\n", mjs_getError(spec));
mj_deleteSpec(spec);
```

## 2. Building Hierarchies: Bodies, Geoms, Joints & Sites

### Adding Elements

```python
body = spec.worldbody.add_body(name='arm', pos=[0, 0, 0.5])

# Geom with capsule shape
geom = body.add_geom(
    name='arm_geom',
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=[0.05, 0.25],
    rgba=[0.2, 0.6, 1.0, 1.0],
)

# Hinge joint
joint = body.add_joint(
    name='shoulder',
    type=mujoco.mjtJoint.mjJNT_HINGE,
    axis=[0, 1, 0],
    range=[-1.57, 1.57],
)

# Sensor site and camera
site = body.add_site(name='grip_site', pos=[0, 0, 0.25])
cam = body.add_camera(name='tool_cam', pos=[0, -0.5, 0.25], xyaxes=[1, 0, 0, 0, 0, 1])
```

### Geom Dimensions Reference

| Geom Type | `size` Parameter Semantics |
| :--- | :--- |
| `mjGEOM_SPHERE` | `[radius]` |
| `mjGEOM_CAPSULE` | `[radius, half_length]` (or `[radius]` if using `fromto`) |
| `mjGEOM_CYLINDER` | `[radius, half_length]` (or `[radius]` if using `fromto`) |
| `mjGEOM_BOX` | `[half_x, half_y, half_z]` |
| `mjGEOM_ELLIPSOID`| `[radius_x, radius_y, radius_z]` |
| `mjGEOM_PLANE` | `[half_x, half_y, grid_spacing]` |

> [!WARNING]
>
> When `fromto=[x1, y1, z1, x2, y2, z2]` is specified for capsules or cylinders,
> `size` takes only **one value** (`[radius]`); length is calculated from endpoints.

### Orientation Keywords

Only **one** orientation specification can be used at a time:
-   `quat=[w, x, y, z]` (native representation)
-   `euler=[a1, a2, a3]` (sequence set by `spec.compiler.eulerseq`, default `'xyz'`)
-   `axisangle=[x, y, z, angle]`
-   `xyaxes=[x_x, x_y, x_z, y_x, y_y, y_z]`
-   `zaxis=[x, y, z]`

Angles in `euler` and `axisangle` are in degrees unless
`spec.compiler.degree = False` (see [modeling.rst](../../modeling.rst),
`.. _COrientation:`, and [XMLreference.rst](../../XMLreference.rst),
`.. _body-geom:`).

## 3. Sub-model Composition & Attachments

Compose models modularly by attaching sub-specs with prefix namespaces (see
[programming/modeledit.rst](../../programming/modeledit.rst),
`.. _meAttachment:` and `.. _meAttributeMerging:`):

```python
parent = mujoco.MjSpec()
child = mujoco.MjSpec()

child_body = child.worldbody.add_body(name='gripper')
child_body.add_geom(size=[0.05], type=mujoco.mjtGeom.mjGEOM_SPHERE)

# Attach child spec to a mount site on the parent
mount_site = parent.worldbody.add_site(name='mount', pos=[0, 0, 1.0])
parent.attach(child, site=mount_site, prefix='gripper_')
# Child body is renamed to 'gripper_gripper' in parent
```

### Cross-Spec Equality Constraints

When linking bodies from two different child specs (e.g. dual-arm setups),
attach both children to the common parent first, then declare constraints on the
parent spec:

```python
left_mount = parent.worldbody.add_frame(pos=[0, 0.3, 0])
right_mount = parent.worldbody.add_frame(pos=[0, -0.3, 0])
parent.attach(arm1_spec, frame=left_mount, prefix='left_')
parent.attach(arm2_spec, frame=right_mount, prefix='right_')

eq = parent.add_equality(
    type=mujoco.mjtEq.mjEQ_CONNECT,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    name1='left_hand',
    name2='right_hand',
)
eq.data[0:3] = [0, 0, 0]  # connect anchor, in name1's body frame
```

> [!NOTE]
>
> `attach()` requires exactly one of `frame=` or `site=`; omitting both raises
> `ValueError`.

## 4. Deleting & Dynamically Modifying Elements

### Deleting Elements

Pass the element object to `spec.delete()`; deleting a body removes its subtree:

```python
spec.delete(spec.body('unwanted_body'))
spec.delete(spec.geom('old_geom'))
spec.delete(spec.joint('redundant_joint'))
spec.delete(spec.actuator('motor'))
```

### State-Preserving Recompilation

When changing model architecture at runtime, use `spec.recompile` to keep
existing physics state:

```python
# Add new tool to robot
spec.worldbody.add_geom(name='tool_tip', size=[0.02], type=mujoco.mjtGeom.mjGEOM_SPHERE)

# Recompile keeping positions/velocities intact
model, data = spec.recompile(model, data)
```

## 5. Actuators & Servos

Always explicitly set the transmission type (`trntype`):

```python
# 1. Torque Motor
motor = spec.add_actuator(
    name='torque_actuator',
    target='shoulder',
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
)
motor.set_to_motor()

# 2. Stiff Position Servo (Recommended for position control)
servo = spec.add_actuator(
    name='pos_servo',
    target='shoulder',
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
)
servo.set_to_position(kp=200.0, dampratio=1.0)  # Pair with implicitfast (kv defaults to 0)
servo.forcelimited = True
servo.forcerange = [-50.0, 50.0]     # Peak torque in Nm
servo.ctrllimited = True
servo.ctrlrange = [-1.57, 1.57]      # Target angle in radians

# 3. Velocity Actuator
vel = spec.add_actuator(
    name='vel_driver',
    target='wheel_joint',
    trntype=mujoco.mjtTrn.mjTRN_JOINT,
)
vel.set_to_velocity(kv=10.0)
```

> [!CAUTION]
>
> The helper functions `set_to_position()` and `set_to_motor()` configure gains
> but do **not** set the transmission. You must always specify
> `trntype=mujoco.mjtTrn.mjTRN_JOINT` (or another valid transmission type). See
> [modeling.rst](../../modeling.rst) (`.. _CActuators:`, `.. _CActShortcuts:`)
> and [XMLreference.rst](../../XMLreference.rst) (`.. _actuator:`).

## 6. Default Classes (Inheritance & Styling)

Default classes (`MjsDefault`) provide cascaded styling and properties (see
[modeling.rst](../../modeling.rst), `.. _CDefault:`, and
[programming/modeledit.rst](../../programming/modeledit.rst), `.. _meDefault:`):

```python
# Create child class under root 'main' class
main_class = spec.default
rubber = spec.add_default('rubber', main_class)
rubber.geom.friction = [1.5, 0.01, 0.001]
rubber.geom.density = 1200.0

# Assign class at creation
body.add_geom(rubber, name='tire', type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.1, 0.05])

# Assign class post-creation (must pass MjsDefault object, NOT a string)
other_geom.classname = rubber
```

### Baseline Tuning Overrides (`<option>` & Root `<default>`)

Several internal defaults (`Euler`, `pyramidal` cones with `impratio=1`, soft
`solref="0.02 1"`, zero joint `armature`/`damping`) are kept for backwards
compatibility rather than new robotics models. Override them at the top of a
project (full rationale and priority-ordered parameter guide in
[modeling.rst](../../modeling.rst), `.. _CTuning:`, `.. _CTuningDefaults:`,
`.. _CTuningGlobal:`, `.. _CTuningModel:`):

```python
spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST  # DISCRETE for flex elasticity
spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
spec.option.impratio = 10.0

spec.default.geom.solref = [0.005, 1.0]                 # Keep solref[0] >= 2 * timestep
spec.default.joint.armature = 0.01                      # Reflected rotor inertia (I_rotor * N^2)
spec.default.joint.damping = 0.1                        # Implicit in M + h*D at zero solver cost
```

Keep the default `solimp`: raising `solimp[0]` (e.g. to `0.95`) halves rest
penetration and creep but costs solver iterations, so reach for it only after
`solref` and elliptic cones.

## 7. Collisions & Friction Tuning

### Collision Filtering Bitmasks

Two geoms collide if and only if:
`(g1.contype & g2.conaffinity) || (g2.contype & g1.conaffinity)`

```python
# Visual-only geom: never collides; set mass=0 if omitting <inertial> so it
# does not add water-density (1000 kg/m^3) mass (inertiagrouprange defaults to "0 5")
body.add_geom(size=[0.1], contype=0, conaffinity=0, group=1, mass=0)

# Team A collides only with Team B
geom_a = body.add_geom(size=[0.1], contype=1, conaffinity=2)
geom_b = body.add_geom(size=[0.1], contype=2, conaffinity=1)
```

### Contact Dimension (`condim`) & Friction

| `condim` | Active Friction Components | Friction Index Used |
| :--- | :--- | :--- |
| 1 | Normal force only (frictionless) | — |
| 3 | Tangential sliding friction | `friction[0]` |
| 4 | Tangential + torsional (twisting) | `friction[0:2]` |
| 6 | Tangential + torsional + rolling | `friction[0:3]` |

```python
# High-friction contact for grasping
finger.add_geom(
    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
    size=[0.01, 0.03],
    condim=4,
    friction=[1.2, 0.02, 0.0001],
)
```

For contact softness, solver impedance/reference tuning (`solref`, `solimp`,
`priority`/mixing), and why parent-child geoms omit collisions by default, read
[modeling.rst](../../modeling.rst) (`.. _CTuning:`, `.. _CTuningModel:`,
`.. _CSolver:`, `.. _CContact:`) and [overview.rst](../../overview.rst)
(`.. _SurprisingCollisions:`, `.. _Soft:`, `.. _Units:`).

## 8. Procedural Meshes, Materials & Sensors

### Procedural Meshes

```python
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
mesh = spec.add_mesh(name='tetra', uservert=vertices.flatten(), userface=faces.flatten())
body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname='tetra')
```

Concave collision meshes are replaced by their convex hulls (unless using SDF
plugins); prefer primitives (`sphere`, `capsule`, `box`, `cylinder`) for both
speed and cleaner normals, or pre-decompose non-convex assets offline with
`CoACD` or `mjfit` (see [computation/index.rst](../../computation/index.rst),
`.. _coChecking:` and `.. _coDecomposition:`).

### Procedural Sensors

```python
# IMU Accelerometer and Gyroscope
spec.add_sensor(name='torso_acc', type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu_site')
spec.add_sensor(name='torso_gyro', type=mujoco.mjtSensor.mjSENS_GYRO,
                objtype=mujoco.mjtObj.mjOBJ_SITE, objname='imu_site')
```

## 9. Common Gotchas & Best Practices

1.  **Non-zero geom sizes**: Geoms with radius or half-extents of zero cause
    immediate compilation errors.
2.  **`classname` takes an object**: `geom.classname = 'rubber'` raises a
    `TypeError`. Pass the `MjsDefault` instance: `geom.classname = rubber`.
3.  **Always assign `recompile()`**: `spec.recompile(m, d)` produces newly
    allocated model and data instances. Reassign them:
    `m, d = spec.recompile(m, d)` (see
    [programming/modeledit.rst](../../programming/modeledit.rst),
    `.. _meRecompilation:`).
4.  **Declare high-contact "hub" bodies first (reverse-Cholesky fill-in)**:
    When a moving body with DOFs (e.g., an actuated tray, sieve, container, or
    robot base) contacts many independent clutter bodies, declare the hub body
    **before** the independent bodies so `mju_cholFactorSparse` factors the
    mass/Hessian arrowhead in $O(N)$ with zero fill-in rather than $O(N^3)$.
    Static geoms on `worldbody` have 0 DOFs and are unaffected — see the full
    derivation in [modeling.rst](../../modeling.rst) (`.. _CBodyOrdering:`).
5.  **Never add `<freejoint/>` (or joints) to the parent body of an unpinned
    `<flexcomp>`**:
    For `<flexcomp>` with `dof="full"`, `"trilinear"`, or `"quadratic"`
    and `rigid="false"`, unpinned vertices/nodes already carry 3 orthogonal
    slide joints providing full 3D motion. Do not add joints or a dummy
    `<inertial/>` on the parent body unless at least one vertex/node is pinned
    via `<pin>`, `rigid="true"`, or `dof="radial"` / `"2d"`. Reposition an
    unpinned `<flexcomp>` via static parent `pos`/`quat` at compile time or
    slide-joint `qpos` offsets at runtime — see
    [modeling.rst](../../modeling.rst) (`.. _CDeformable:`) and
    [XMLreference.rst](../../XMLreference.rst) (`.. _body-flexcomp:`).

## 10. Reference Models & Key References

### Reference MJCF Models & Fixtures

Before authoring complex mechanisms or specialized elements from scratch, check
existing models:

-   **Built-in showcase models (`../../../model/`, or online at
    https://github.com/google-deepmind/mujoco/tree/main/model)**:
    `model/humanoid/`, `model/tendon_arm/arm26.xml`, `model/car/car.xml`,
    `model/slider_crank/slider_crank.xml`, `model/flex/` (`bag.xml`,
    `jelly.xml`, `drape.xml`, `poncho.xml`), `model/replicate/`
    (`newton_cradle.xml`, `particle.xml`, `bunnies.xml`), `model/plugin/`
    (actuator, sensor, SDF), `model/tactile/`, `model/adhesion/`,
    `model/sleep/`, `model/speculative/`, and `model/plastic/`.
-   **MuJoCo Menagerie ([models.rst](../../models.rst), `.. _Menagerie:`)**:
    Production robot descriptions (Franka Panda/FR3, UR5e/UR10e, Unitree
    G1/H1/Go2, Spot, Barkour, Shadow Hand, Robotiq 2F-85, ALOHA 2, Skydio X2,
    Flybody) at https://github.com/google-deepmind/mujoco_menagerie.
-   **Minimal test fixtures (`../../../test/**/testdata/`, or online under
    https://github.com/google-deepmind/mujoco/tree/main/test)**: Focused
    single-feature XMLs in `test/user/testdata/` (`MjSpec` & compiler),
    `test/xml/testdata/` (`parent.xml` / `child.xml` attachment fixtures,
    OpenUSD), and `test/engine/testdata/` (`collisions.xml`, `connect.xml`,
    `equality_site.xml`, `catenary.xml`).

### Documentation

-   [MjSpec & Procedural Model Editing](../../programming/modeledit.rst)
    (`.. _meOverview:`, `.. _meAttachment:`, `.. _meAttributeMerging:`,
    `.. _meDefault:`, `.. _meRecompilation:`)
-   [Python Model Editing & OpenUSD Bindings](../../python.rst)
    (`.. _PyModelEdit:`, `.. _PyEditConvenience:`, `.. _PyMJCF:`,
    `.. _PyUSDexport:`) and [OpenUSD Guide](../../OpenUSD/index.rst)
-   [MJCF Modeling Guide](../../modeling.rst) (`.. _CDefault:`,
    `.. _COrientation:`, `.. _CSolver:`, `.. _CContact:`, `.. _CBodyOrdering:`,
    `.. _CActuators:`, `.. _CDeformable:`, `.. _CTuning:`,
    `.. _CTuningDefaults:`, `.. _CTuningGlobal:`, `.. _CTuningModel:`) and
    [Modeling Pitfalls & Units](../../overview.rst) (`.. _Clarifications:`,
    `.. _Units:`, `.. _SurprisingCollisions:`, `.. _Soft:`)
-   [MJCF XML Schema Reference](../../XMLreference.rst) (`.. _body-geom:`,
    `.. _body-joint:`, `.. _actuator:`, `.. _equality:`, `.. _body-flexcomp:`,
    `.. _replicate:`)
-   [Engine Plugins & Resource Providers](../../programming/extension.rst)
    (`.. _exExtension:`, `.. _exSDF:`)
-   [C Model Editing API Functions](../../APIreference/functions.rst)
    (`.. _Parseandcompile:`, `.. _Attachment:`, `.. _AddTreeElements:`,
    `.. _AddNonTreeElements:`)

### Source Code & Notebook Examples

-   `python/mjspec.ipynb` (`MjSpec` tutorial & `dm_control` `PyMJCF`
    reimplementation; see [python.rst](../../python.rst), `.. _PyMJCF:`),
    [MjSpec Tests](../../../python/mujoco/specs_test.py), and
    [C++ Compile Sample](../../programming/samples.rst) (`.. _saCompile:`)
