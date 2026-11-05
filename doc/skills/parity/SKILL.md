---
name: mujoco-parity
description: >-
  Cross-engine physics parity comparison between MuJoCo (CPU, float64) and
  accelerated backends (MJWarp CUDA, MJX JAX, float32): stage-by-stage
  mj_forward vs mjw.forward isolation, normalized tolerances, stiff affine
  actuator and ill-conditioned mass matrix (qM) float32 rounding bounds,
  actuation-conditioned forward passes, order-insensitive contact matching per
  geom pair, threshold-grazing contact filtering (includemargin), batched
  single-step trajectory evaluation, and float32-perturbed CPU baseline
  rollouts. Use when comparing MuJoCo CPU physics against MJWarp or MJX,
  debugging sim-to-sim trajectory divergence, missing contacts, or force/qacc
  discrepancies. Do NOT use for general GPU batch simulation setup
  (mujoco-accelerated) or CPU-only simulation (mujoco-python).
---

# MuJoCo Parity: Comparing CPU (C) vs Accelerated (MJWarp / MJX) Physics

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`):** Read short chapters like
> [`../../mjwarp/index.rst`](../../mjwarp/index.rst) (`.. _mjwDiff:`,
> `.. _mjwOverflow:`, `.. _mjwPairwise:`, `.. _mjwBoxBox:`),
> [`../../mjx.rst`](../../mjx.rst) (`.. _MjxFeatureParity:`), and
> [`../../programming/simulation.rst`](../../programming/simulation.rst)
> (`.. _siIntegrationState:`, `.. _siWarmstart:`) directly when in a MuJoCo
> checkout; for large files like
> [`../../computation/index.rst`](../../computation/index.rst) (`.. _Pipeline:`,
> `.. _piStages:`) and
> [`../../APIreference/functions.rst`](../../APIreference/functions.rst), grep
> the literal `.. _Label:` anchor. Outside a checkout, fetch
> `https://mujoco.readthedocs.io/en/latest/<path>.html` (or `/en/stable/` for
> released pages; `functions.rst` renders inside
> `APIreference/APIfunctions.html`, and Sphinx anchors lowercase the label,
> e.g. `.. _mjwDiff:` → `#mjwdiff`).
>
> **Related skills:**
>
> -   [mujoco-accelerated](../accelerated/SKILL.md) — GPU/TPU batch simulation
>     with MJWarp and MJX.
> -   [mujoco-python](../python/SKILL.md) — CPU simulation lifecycle, state
>     access, and stepping.

> [!IMPORTANT]
>
> By default, MuJoCo (C) executes in `float64` with dynamic buffer allocation,
> whereas MJWarp and MJX execute in `float32` with pre-allocated
> contact/constraint buffers and GPU-parallel collision and solver kernels.
> Comparing raw multi-step trajectories without accounting for `float32`
> conditioning, contact ordering, and buffer capacities leads to false
> positives and misdiagnosed root causes. For direct `float32`-to-`float32`
> comparison against MJWarp/MJX (or to isolate pure precision effects by
> comparing `float32` CPU against `float64` CPU), build MuJoCo C and its Python
> bindings in single precision (`mjUSESINGLE`) with
> `--@mujoco//:single_precision` (Bazel) or
> `-DCMAKE_C_FLAGS="-DmjUSESINGLE" -DCMAKE_CXX_FLAGS="-DmjUSESINGLE"` (CMake),
> and verify `mujoco.MJTNUM_DTYPE == np.float32`.

## 1. The Three-Level Parity Hierarchy

Contact-rich rigid-body dynamics are chaotic: a $10^{-7}$ `float32` difference
that flips a single grazing contact (`dist ≈ includemargin`) or convex collision
witness point on step $k$ causes two trajectories to diverge exponentially
thereafter, even when both engines are bug-free. Always diagnose parity using
three levels of increasing scope:

| Level | Method | What It Isolates | When to Use |
| :--- | :--- | :--- | :--- |
| **1. Single-State Forward** (`mj_forward` vs `mjw.forward`) | Evaluate both engines on identical states batched as worlds (`nworld = len(states)`) and compare intermediate `Data` fields in pipeline order | Pinpoints the exact pipeline stage (kinematics, tendons, mass matrix, actuation, smooth dynamics, collision, or constraint solver) and model element where a discrepancy originates | Primary unit test and root-cause diagnostic tool |
| **2. Batched Single-Step Trajectory** (`mj_step` vs `mjw.step`) | Record a reference CPU trajectory of $T$ steps, then take **one** MJWarp step from all $T$ pre-step states in parallel (`nworld = T`) | Prevents error accumulation over time; a spike at step $t$ points directly to a dynamics or contact discrepancy at state $t$ rather than earlier drift | Finding state-dependent bugs (e.g., specific contact configurations) along realistic rollouts |
| **3. Open-Loop Rollout with Float32 Baseline** | Step both engines from $t=0$ with identical controls alongside a **float32-perturbed CPU shadow rollout** | Distinguishes genuine parity bugs (MJWarp diverges before the `float32` CPU baseline on unambiguous steps) from benign chaotic divergence | Evaluating end-to-end rollout horizon fidelity for RL policies or controllers |

---

## 2. Pre-Flight Buffer Sizing, Overflow Detection & State Sync

### 1. Size `nconmax` and `njmax` for $2\times$ Peak CPU Counts

When MJWarp exceeds `nconmax`/`naconmax` or `njmax`, excess contacts or
constraint rows are dropped, which looks identical to missing collision contacts
or solver failure. Size capacities to **at least twice** the maximum CPU counts
(`d.ncon`, `d.nefc`) across the evaluated states so extra contacts generated by
MJWarp are caught as contact mismatches rather than clipped by buffer overflow:

```python
from typing import Optional, Sequence
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

MIN_NCONMAX = 64
MIN_NJMAX = 256


def capacities(
    mjds: Sequence[mujoco.MjData],
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
) -> tuple[int, int]:
  if nconmax is None:
    nconmax = max(MIN_NCONMAX, 2 * max(d.ncon for d in mjds))
  if njmax is None:
    njmax = max(MIN_NJMAX, 2 * max(d.nefc for d in mjds))
  return nconmax, njmax


def check_overflow(d: mjw.Data) -> None:
  nacon = int(d.nacon.numpy()[0])
  if nacon > d.naconmax:
    raise RuntimeError(
        f"MJWarp contact overflow ({nacon} > naconmax {d.naconmax}); "
        "increase nconmax."
    )
  nefc = int(d.nefc.numpy().max())
  if nefc > d.njmax:
    raise RuntimeError(
        f"MJWarp constraint overflow ({nefc} > njmax {d.njmax}); "
        "increase njmax."
    )
  buf_mask = ~(mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS)
  overflow = int(np.bitwise_or.reduce(d.overflow.numpy()) & buf_mask)
  if overflow:
    raise RuntimeError(
        f"MJWarp buffer overflow: {mjw.OverflowType(overflow)!r}"
    )
```

### 2. Synchronize the Full Simulation State (`set_states`)

Copying only `qpos` and `qvel` leaves actuator activations, mocap poses, applied
wrenches, equality constraint toggles, delay history buffers, and solver
warmstarts out of sync. In particular, unless `mjDSBL_WARMSTART` is set, MJWarp
initializes its solver directly from `qacc_warmstart` (whereas CPU MuJoCo
compares `qacc_warmstart` against `qacc_smooth`). Use `mujoco.mj_getState` and
`mjw.set_state` with `mjSTATE_INTEGRATION` to copy the complete integration
state across worlds:

```python
def copy_data(mjm: mujoco.MjModel, src: mujoco.MjData) -> mujoco.MjData:
  dst = mujoco.MjData(mjm)
  mujoco.mj_copyData(dst, mjm, src)
  return dst


def set_states(
    mjm: mujoco.MjModel,
    m: mjw.Model,
    d: mjw.Data,
    states: Sequence[mujoco.MjData],
) -> None:
  """Copies the full integration state of each CPU MjData into the corresponding MJWarp world."""
  if len(states) != d.nworld:
    raise ValueError(f"Got {len(states)} states for {d.nworld} worlds.")
  sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  size = mujoco.mj_stateSize(mjm, sig)
  mj_states = np.empty((len(states), size))
  for i, s in enumerate(states):
    mujoco.mj_getState(mjm, s, mj_states[i], sig)
  mjw.set_state(m, d, wp.array(mj_states, dtype=float), sig)


def put_batched(
    mjm: mujoco.MjModel,
    mjds: Sequence[mujoco.MjData],
    nconmax: Optional[int] = None,
    njmax: Optional[int] = None,
) -> tuple[mjw.Model, mjw.Data]:
  nconmax, njmax = capacities(mjds, nconmax, njmax)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjds[0], nworld=len(mjds), nconmax=nconmax, njmax=njmax)
  set_states(mjm, m, d, mjds)
  return m, d
```

---

## 3. Stage-by-Stage `forward` Isolation & Conditioning Bounds

### 1. Pipeline Stage Order & Baseline Tolerances

Each stage of `mj_forward` feeds the next. Always inspect errors in topological
pipeline order—the **first** stage with normalized error $> 1$ is the root
cause:

| Pipeline Stage | `Data` Fields | Default `atol` | Default `rtol` | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **1. Kinematics** | `xpos`, `xquat`, `site_xpos`, `subtree_com` | `1e-5` | `1e-4` | Forward kinematics (`mj_kinematics`, `mj_comPos`) |
| **2. Tendons & Lengths** | `ten_length`, `actuator_length` | `1e-5` | `1e-4` | Spatial/fixed tendon & transmission lengths |
| **3. Mass Matrix** | `M` | `1e-8` | `1e-5` | CSR lower triangle (`mjm.M_rownnz`, `mjm.M_colind`); add CRB scale bound |
| **4. Velocity & Forces** | `actuator_force`, `qfrc_actuator`, `qfrc_bias`, `qfrc_passive` | `1e-3` | `1e-3` | RNE bias, passive springs/dampers, affine actuators; add stiff-actuator bound |
| **5. Smooth Dynamics** | `qfrc_smooth`, `qacc_smooth` | `1e-3`, `1e-2` | `1e-3` | Unconstrained net force and $M^{-1} q_{\text{frc,smooth}}$; compare on actuation-conditioned CPU `MjData` |
| **6. Constraint Dynamics** | `qfrc_constraint`, `qacc` | `1e-1` | `1e-2` | Iterative solver output (CG/Newton stop at `opt.tolerance`); compare on actuation-conditioned CPU `MjData` |

Normalized error is defined as:

```text
err = |mujoco - warp| / (atol + rtol * |mujoco| + extra_atol)
```

Values $\le 1.0$ are within tolerance.

### 2. Gotcha 1: Stiff Affine Actuators & Actuation Conditioning

High-gain position or tendon servos (e.g. $k_p \sim 10^4\text{--}10^6$) compute
force as a difference of large terms:
$f = (g_0 + g_1 L + g_2 \dot{L}) u + (b_0 + b_1 L + b_2 \dot{L})$. A `float32`
rounding of $L$ (`actuator_length`) or $\dot{L}$ (`actuator_velocity`) is
multiplied by $g$ and $b$, producing legitimate force differences of size
$O(\epsilon_{32} (|g| |u| + |b|))$ that exceed `1e-3`. When these forces act on
light DOFs (such as fingers), they cause large downstream differences in
`qacc_smooth`, `qfrc_constraint`, and `qacc` that masquerade as mass-matrix or
constraint-solver bugs.

**Two-part fix:**
1. Add a componentwise `float32` rounding bound (`actuator_rounding_atol`) to
   `actuator_force` and `qfrc_actuator`.
2. **Condition the CPU reference on MJWarp's actuator forces** when checking
   downstream fields (`qfrc_smooth`, `qacc_smooth`, `qfrc_constraint`, `qacc`)
   by adding `warp_qfrc_actuator - mjd.qfrc_actuator` into `mjd.qfrc_applied`
   and re-running `mj_forward`:

```python
FLOAT32_ROUNDING_FACTOR = 8.0
ACTUATION_DOWNSTREAM_FIELDS = frozenset(
    {"qfrc_smooth", "qacc_smooth", "qfrc_constraint", "qacc"}
)


def actuator_rounding_atol(
    mjm: mujoco.MjModel, mjd: mujoco.MjData
) -> dict[str, np.ndarray]:
  """Returns float32 rounding bounds for actuator_force and qfrc_actuator."""
  if not mjm.nu or (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_ACTUATION):
    return {}
  eps = float(np.finfo(np.float32).eps)
  length, velocity = mjd.actuator_length, mjd.actuator_velocity
  gain, bias = mjm.actuator_gainprm, mjm.actuator_biasprm
  gaintype, biastype = mjm.actuator_gaintype, mjm.actuator_biastype

  ctrl = mjd.ctrl.copy()
  if not (mjm.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL):
    limited = mjm.actuator_ctrllimited.astype(bool)
    ctrl[limited] = np.clip(
        ctrl[limited],
        mjm.actuator_ctrlrange[limited, 0],
        mjm.actuator_ctrlrange[limited, 1],
    )
  act_input = np.abs(ctrl)
  stateful = mjm.actuator_actnum > 0
  if np.any(stateful):
    act_ind = mjm.actuator_actadr[stateful] + mjm.actuator_actnum[stateful] - 1
    act = mjd.act[act_ind].copy()
    act_lim = mjm.actuator_actlimited[stateful].astype(bool)
    act[act_lim] = np.clip(
        act[act_lim],
        mjm.actuator_actrange[stateful][act_lim, 0],
        mjm.actuator_actrange[stateful][act_lim, 1],
    )
    act_input[stateful] = np.abs(act)

  gain_terms = np.abs(gain[:, 0]) + np.where(
      gaintype == mujoco.mjtGain.mjGAIN_AFFINE,
      np.abs(gain[:, 1] * length) + np.abs(gain[:, 2] * velocity),
      0.0,
  )
  bias_terms = np.where(
      biastype == mujoco.mjtBias.mjBIAS_AFFINE,
      np.abs(bias[:, 0])
      + np.abs(bias[:, 1] * length)
      + np.abs(bias[:, 2] * velocity),
      0.0,
  )
  affine = np.isin(
      gaintype,
      [int(mujoco.mjtGain.mjGAIN_FIXED), int(mujoco.mjtGain.mjGAIN_AFFINE)],
  ) & np.isin(
      biastype,
      [int(mujoco.mjtBias.mjBIAS_NONE), int(mujoco.mjtBias.mjBIAS_AFFINE)],
  )
  force_atol = np.where(
      affine,
      FLOAT32_ROUNDING_FACTOR * eps * (gain_terms * act_input + bias_terms),
      0.0,
  )

  moment = np.zeros((mjm.nu, mjm.nv))
  mujoco.mju_sparse2dense(
      moment,
      mjd.actuator_moment,
      mjd.moment_rownnz,
      mjd.moment_rowadr,
      mjd.moment_colind,
  )
  return {
      "actuator_force": force_atol,
      "qfrc_actuator": np.abs(moment).T @ force_atol,
  }


def actuation_conditioned(
    mjm: mujoco.MjModel, mjd: mujoco.MjData, warp_qfrc_actuator: np.ndarray
) -> mujoco.MjData:
  """Re-runs CPU mj_forward using MJWarp's qfrc_actuator to isolate downstream stages."""
  conditioned = copy_data(mjm, mjd)
  conditioned.qfrc_applied += (
      np.asarray(warp_qfrc_actuator, dtype=np.float64) - mjd.qfrc_actuator
  )
  mujoco.mj_forward(mjm, conditioned)
  return conditioned
```

### 3. Gotcha 2: Ill-Conditioned Mass Matrices (`M`) & Parallel-Axis Cancellation

`M[i, j] = cdof_i' crb_b cdof_j` accumulates composite rigid-body inertias
(`crb`) expressed about the kinematic tree root's subtree center of mass. For
light distal bodies far from the root (e.g. humanoid hands/fingers), the
parallel-axis terms $m |r|^2$ are much larger than the resulting entries of `M`,
and solving $M x = f$ amplifies `float32` rounding by $|M^{-1}|$:

```python
def mass_matrix_scale(mjm: mujoco.MjModel, mjd: mujoco.MjData) -> np.ndarray:
  """Magnitude of spatial inertia terms summed into each entry of M."""
  cinert = np.abs(mjd.cinert)
  inertia = np.zeros((mjm.nbody, 6, 6))
  for k, (r, c) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
    inertia[:, r, c] = inertia[:, c, r] = cinert[:, k]
  mc = cinert[:, 6:9]
  skew = np.zeros((mjm.nbody, 3, 3))
  skew[:, 0, 1] = skew[:, 1, 0] = mc[:, 2]
  skew[:, 0, 2] = skew[:, 2, 0] = mc[:, 1]
  skew[:, 1, 2] = skew[:, 2, 1] = mc[:, 0]
  inertia[:, :3, 3:] = skew
  inertia[:, 3:, :3] = skew
  inertia[:, 3:, 3:] = cinert[:, 9, None, None] * np.eye(3)

  for b in range(mjm.nbody - 1, 0, -1):
    inertia[mjm.body_parentid[b]] += inertia[b]

  cdof = np.abs(mjd.cdof)
  scale = np.zeros((mjm.nv, mjm.nv))
  for i in range(mjm.nv):
    crb_cdof = inertia[mjm.dof_bodyid[i]] @ cdof[i]
    j = i
    while j >= 0:
      scale[i, j] = scale[j, i] = cdof[j] @ crb_cdof
      j = mjm.dof_parentid[j]
  scale[np.diag_indices(mjm.nv)] += mjm.dof_armature
  return scale


def mass_rounding_atol(
    mjm: mujoco.MjModel, mjd: mujoco.MjData
) -> dict[str, np.ndarray]:
  """Componentwise float32 forward-error bounds for M, qacc_smooth, and qacc."""
  if not mjm.nv:
    return {}
  eps = float(np.finfo(np.float32).eps)
  mass = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, mjd, mass)
  scale = mass_matrix_scale(mjm, mjd)
  abs_inv = np.abs(np.linalg.inv(mass))

  def bound(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    return FLOAT32_ROUNDING_FACTOR * eps * (abs_inv @ (scale @ np.abs(x) + f))

  rows = np.repeat(np.arange(mjm.nv), mjm.M_rownnz)
  return {
      "M": FLOAT32_ROUNDING_FACTOR * eps * scale[rows, mjm.M_colind],
      "qacc_smooth": bound(mjd.qacc_smooth, np.abs(mjd.qfrc_smooth)),
      "qacc": bound(
          mjd.qacc, np.abs(mjd.qfrc_smooth) + np.abs(mjd.qfrc_constraint)
      ),
  }
```

### 4. Complete Batched `forward_errors` Recipe

```python
DEFAULT_FORWARD_TOLERANCES = {
    "xpos": (1e-5, 1e-4),
    "xquat": (1e-5, 1e-4),
    "site_xpos": (1e-5, 1e-4),
    "subtree_com": (1e-5, 1e-4),
    "ten_length": (1e-5, 1e-4),
    "actuator_length": (1e-5, 1e-4),
    "M": (1e-8, 1e-5),
    "actuator_force": (1e-3, 1e-3),
    "qfrc_actuator": (1e-3, 1e-3),
    "qfrc_bias": (1e-3, 1e-3),
    "qfrc_passive": (1e-3, 1e-3),
    "qfrc_smooth": (1e-3, 1e-3),
    "qacc_smooth": (1e-2, 1e-3),
    "qfrc_constraint": (1e-1, 1e-2),
    "qacc": (1e-1, 1e-2),
}


def forward_errors(
    mjm: mujoco.MjModel,
    states: Sequence[mujoco.MjData],
    tolerances: dict[str, tuple[float, float]] = DEFAULT_FORWARD_TOLERANCES,
) -> dict[str, dict[str, object]]:
  mjds = [copy_data(mjm, s) for s in states]
  for mjd in mjds:
    mujoco.mj_forward(mjm, mjd)

  m, d = put_batched(mjm, mjds)
  mjw.forward(m, d)
  check_overflow(d)
  warp_fields = {
      k: getattr(d, k).numpy() for k in set(tolerances) | {"qfrc_actuator"}
  }

  worst = {k: {"error": 0.0, "state": 0, "index": ()} for k in tolerances}
  for i, mjd in enumerate(mjds):
    mjd_cond = actuation_conditioned(mjm, mjd, warp_fields["qfrc_actuator"][i])
    extra = actuator_rounding_atol(mjm, mjd) | mass_rounding_atol(mjm, mjd_cond)
    for name, (atol, rtol) in tolerances.items():
      ref_data = mjd_cond if name in ACTUATION_DOWNSTREAM_FIELDS else mjd
      ref = np.asarray(getattr(ref_data, name), dtype=np.float64)
      if ref.size == 0:
        continue
      act = np.asarray(warp_fields[name][i], dtype=np.float64).reshape(ref.shape)
      if name == "xquat":
        sign = np.where(np.sum(ref * act, axis=-1, keepdims=True) < 0, -1.0, 1.0)
        act = act * sign
      norm = np.abs(ref - act) / (atol + rtol * np.abs(ref) + extra.get(name, 0.0))
      idx = np.unravel_index(np.argmax(norm), ref.shape)
      if float(norm[idx]) > worst[name]["error"]:
        worst[name] = {
            "error": float(norm[idx]),
            "state": i,
            "index": tuple(int(j) for j in idx),
            "mujoco": float(ref[idx]),
            "warp": float(act[idx]),
        }
  return worst
```

---

## 4. Order-Insensitive Contact Matching & Ambiguous Contacts

### 1. Why `d.contact` Cannot Be Compared Index-by-Index

1. **Non-deterministic GPU ordering**: Atomic contact allocation across parallel
   broadphase/narrowphase threads permutes contact slot indices.
2. **Geom pair orientation**: A contact between geoms $A$ and $B$ may be stored
   as `(g1, g2)` or `(g2, g1)` (with negated normal). Canonicalize pairs as
   `(min(g1, g2), max(g1, g2))`.
3. **MJWarp sensor-only contact slots**: MJWarp's main collision pipeline writes
   a contact entry for geom pairs participating in collision sensors
   (`distance`, `normal`, `fromto`) even when they do not collide in physics.
   Filter active physics contacts by masking `d.contact.type` with
   `ContactType.CONSTRAINT | ContactType.PASSIVE` and slicing to
   `nacon = min(int(d.nacon.numpy()[0]), d.naconmax)`.
4. **Threshold-grazing contacts (`includemargin`)**: Contacts with distance
   within `dist_tolerance` (`1e-5` m) of `includemargin` (`margin - gap`) can
   legitimately fall on opposite sides of the threshold in `float32` vs
   `float64` and must be excluded on both sides.

```python
import collections
from mujoco_warp._src.types import ContactType

DEFAULT_CONTACT_DIST_TOLERANCE = 1e-5
EXPORTED_CONTACT_TYPES = ContactType.CONSTRAINT | ContactType.PASSIVE


def _contact_counts(
    geoms: np.ndarray,
    dists: np.ndarray,
    includemargins: np.ndarray,
    dist_tolerance: float,
) -> collections.Counter[tuple[int, int]]:
  counts = collections.Counter()
  for geom, dist, margin in zip(geoms, dists, includemargins):
    if abs(dist - margin) <= dist_tolerance:
      continue
    counts[(int(min(geom)), int(max(geom)))] += 1
  return counts


def contact_mismatches(
    mjd: mujoco.MjData,
    d: mjw.Data,
    world: int = 0,
    dist_tolerance: float = DEFAULT_CONTACT_DIST_TOLERANCE,
) -> dict[tuple[int, int], tuple[int, int]]:
  """Returns {(g1, g2): (mj_count, warp_count)} for geom pairs whose contact counts differ."""
  mj_c = mjd.contact
  mj_counts = _contact_counts(
      mj_c.geom, mj_c.dist, mj_c.includemargin, dist_tolerance
  )

  nacon = min(int(d.nacon.numpy()[0]), d.naconmax)
  exported = (d.contact.type.numpy()[:nacon] & EXPORTED_CONTACT_TYPES) != 0
  worldid = d.contact.worldid.numpy()[:nacon][exported]
  w = worldid == world
  warp_counts = _contact_counts(
      d.contact.geom.numpy()[:nacon][exported][w],
      d.contact.dist.numpy()[:nacon][exported][w],
      d.contact.includemargin.numpy()[:nacon][exported][w],
      dist_tolerance,
  )
  return {
      pair: (mj_counts[pair], warp_counts[pair])
      for pair in sorted(mj_counts.keys() | warp_counts.keys())
      if mj_counts[pair] != warp_counts[pair]
  }
```

### 2. Detecting Float32-Ambiguous Contact States (`contacts_ambiguous`)

Beyond grazing the `includemargin` boundary, convex collision (GJK/EPA) on
nearly parallel faces (e.g. stacked boxes or meshes) is only defined up to
`mjm.opt.ccd_tolerance`. A state is `float32`-ambiguous if CPU MuJoCo itself
changes its contact counts under:
- `float32` state rounding,
- random `qpos` perturbations of magnitude `mjm.opt.ccd_tolerance`, or
- scaling `mjm.opt.ccd_tolerance` by $10\times$ or $0.1\times$.

```python
def round_state_float32(mjm: mujoco.MjModel, d: mujoco.MjData) -> None:
  sig = int(mujoco.mjtState.mjSTATE_INTEGRATION)
  state = np.empty(mujoco.mj_stateSize(mjm, sig))
  mujoco.mj_getState(mjm, d, state, sig)
  mujoco.mj_setState(mjm, d, state.astype(np.float32), sig)


def contacts_ambiguous(
    mjm: mujoco.MjModel,
    state: mujoco.MjData,
    dist_tolerance: float = DEFAULT_CONTACT_DIST_TOLERANCE,
    num_perturbations: int = 2,
    seed: int = 0,
) -> bool:
  """Returns True if contact counts at `state` are sensitive to float32 / CCD tolerance."""

  def counts_at(qpos_delta=None):
    d = copy_data(mjm, state)
    if qpos_delta is not None:
      round_state_float32(mjm, d)
      mujoco.mj_integratePos(mjm, d.qpos, qpos_delta, 1.0)
    mujoco.mj_fwdPosition(mjm, d)
    c = d.contact
    near = bool(np.any(np.abs(c.dist - c.includemargin) <= dist_tolerance))
    return near, _contact_counts(
        c.geom, c.dist, c.includemargin, dist_tolerance
    )

  near, base_counts = counts_at()
  if near:
    return True
  rng = np.random.default_rng(seed)
  tol = mjm.opt.ccd_tolerance
  deltas = [np.zeros(mjm.nv)] + [
      rng.uniform(-tol, tol, mjm.nv) for _ in range(num_perturbations)
  ]
  if any(counts_at(dq)[1] != base_counts for dq in deltas):
    return True
  try:
    for factor in (10.0, 0.1):
      mjm.opt.ccd_tolerance = tol * factor
      if counts_at()[1] != base_counts:
        return True
    return False
  finally:
    mjm.opt.ccd_tolerance = tol
```

> [!NOTE]
>
> **Known Collider Differences (Box-Box & Max Contacts):** By default
> (`NATIVECCD` enabled and zero box-box margin), MJWarp routes box-box pairs to
> its general convex GJK/EPA collider (up to 4 contacts), whereas CPU MuJoCo
> always uses its specialized `mjc_BoxBox` primitive collider (up to 8
> contacts). Setting `m.opt.disableflags |= mjw.DisableBit.NATIVECCD` (or
> `<flag nativeccd="disable"/>`) switches MJWarp to the primitive box-box port.

---

## 5. Batched Single-Step Parity & Float32 Baseline Rollouts

### 1. Emulating Float32 Dynamics Perturbations on CPU

To establish how much trajectory divergence is caused purely by `float32`
arithmetic on a given model, run a shadow CPU `MjData` that rounds state to
`float32` and injects a random generalized force perturbation `dqfrc` matching
one `float32` ULP of the actuation and mass-matrix solve:

```python
def float32_perturbation(
    mjm: mujoco.MjModel, d: mujoco.MjData, rng: np.random.Generator
) -> np.ndarray:
  atol = actuator_rounding_atol(mjm, d) | mass_rounding_atol(mjm, d)
  if "qacc" not in atol:
    return np.zeros(mjm.nv)
  mass = np.zeros((mjm.nv, mjm.nv))
  mujoco.mj_fullM(mjm, d, mass)
  dqfrc = mass @ (atol["qacc"] * rng.uniform(-1, 1, mjm.nv))
  if "qfrc_actuator" in atol:
    dqfrc += atol["qfrc_actuator"] * rng.uniform(-1, 1, mjm.nv)
  return dqfrc / FLOAT32_ROUNDING_FACTOR
```

### 2. Batched Single-Step Trajectory Evaluation (`step_errors`)

Batching all $T$ trajectory states as `nworld = T` worlds in a single
`mjw.step(m, d)` call evaluates single-step accuracy across an entire rollout in
one GPU launch without accumulating drift:

```python
def step_errors(
    mjm: mujoco.MjModel,
    initial_state: mujoco.MjData,
    ctrls: np.ndarray,
    seed: int = 0,
) -> dict[str, np.ndarray]:
  """Compares 1-step MJWarp vs CPU along a CPU trajectory of length len(ctrls)."""
  rng = np.random.default_rng(seed)
  mjd = copy_data(mjm, initial_state)
  mujoco.mj_forward(mjm, mjd)
  mjd_f32 = mujoco.MjData(mjm)

  nstep = len(ctrls)
  pre_states, post_states = [], []
  qvel_baseline = np.zeros(nstep)
  ambiguous = np.zeros(nstep, dtype=bool)

  for i, ctrl in enumerate(ctrls):
    mjd.ctrl[:] = ctrl
    ambiguous[i] = contacts_ambiguous(mjm, mjd)
    pre_states.append(copy_data(mjm, mjd))

    mujoco.mj_copyData(mjd_f32, mjm, mjd)
    round_state_float32(mjm, mjd_f32)
    mujoco.mj_forward(mjm, mjd_f32)
    mjd_f32.qfrc_applied[:] += float32_perturbation(mjm, mjd_f32, rng)
    mujoco.mj_step(mjm, mjd_f32)

    mujoco.mj_step(mjm, mjd)
    post_states.append(copy_data(mjm, mjd))
    qvel_baseline[i] = np.max(np.abs(mjd.qvel - mjd_f32.qvel), initial=0.0)

  nconmax, njmax = capacities(post_states)
  m, d = put_batched(mjm, pre_states, nconmax, njmax)
  mjw.step(m, d)
  check_overflow(d)

  qvel_mj = np.stack([s.qvel for s in post_states])
  qvel_diff = np.abs(qvel_mj - d.qvel.numpy())
  return {
      "qvel_err": np.max(qvel_diff, axis=1, initial=0.0),
      "qvel_baseline": qvel_baseline,
      "ambiguous": ambiguous,
  }
```

### 3. Open-Loop Rollout Comparison (`rollout_errors`)

When comparing full open-loop rollouts, plot or compare `qpos_err` / `qvel_err`
against `qpos_baseline` / `qvel_baseline`:
- If `qvel_err` stays comparable to `qvel_baseline` (or only grows after
  `ambiguous[t]` is `True`), the rollout divergence is expected `float32`
  trajectory sensitivity.
- If `qvel_err >> qvel_baseline` at step $t$ while `ambiguous[:t+1]` is `False`,
  extract `pre_states[t]` and run `forward_errors(mjm, [pre_states[t]])` and
  `contact_mismatches` to pinpoint the exact stage and element causing the bug.

### 4. Adapting to MJX (`mujoco.mjx`)

The same state-conditioning and tolerance principles apply when comparing CPU
MuJoCo against MJX (`from mujoco import mjx`):
- Replace `put_batched` + `mjw.forward(m, d)` with `mjx.put_model(mjm)`,
  `jax.vmap(lambda s: mjx.put_data(mjm, s))`, and
  `jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))(mx, dx_batch)`.
- In MJX's JAX backend, all candidate geom-pair contact slots are statically
  present in `dx.contact`; active contacts satisfy
  `dx.contact.dist < dx.contact.includemargin` (ignore slots within
  `dist_tolerance` of `includemargin`).

---

## 6. Key References

-   [MJWarp Differences from MuJoCo & Colliders](../../mjwarp/index.rst) (`.. _mjwDiff:`, `.. _mjwOverflow:`, `.. _mjwPairwise:`, `.. _mjwBoxBox:`)
-   [MJX Feature Parity & Sharp Bits](../../mjx.rst) (`.. _MjxFeatureParity:`, `.. _MjxSharpBits:`)
-   [Forward Pipeline Stages, Actuation & Solvers](../../computation/index.rst) (`.. _Pipeline:`, `.. _piStages:`, `.. _geActuation:`, `.. _Solver:`, `.. _Collision:`)
-   [Integration State (`mjSTATE_INTEGRATION`) & Solver Warmstart](../../programming/simulation.rst) (`.. _siIntegrationState:`, `.. _siWarmstart:`)
-   [C Forward Subcomponent API](../../APIreference/functions.rst) (`.. _Components:`, `.. _Subcomponents:`)
