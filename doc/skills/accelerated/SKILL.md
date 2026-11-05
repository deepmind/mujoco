---
name: mujoco-accelerated
description: >-
  GPU and TPU batch simulation, MJX (JAX), MJWarp (CUDA), parallel rollouts
  (jax.vmap), trajectory scan (jax.lax.scan), differentiability, device memory
  (put_model, put_data), static contact sizing (max_contact_points,
  max_geom_pairs, nconmax, njmax), float32 precision, XLA compilation caching.
  Use for RL environment batches, GPU rollouts, differentiable physics.
  Do NOT use for CPU simulation, noslip/PGS solvers (mujoco-python), cross-engine parity debugging (mujoco-parity), rendering (mujoco-rendering), or GUI (mujoco-gui).
---

# MuJoCo Accelerated: MJX & MJWarp GPU Backends

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`):** Read
> [`../../mjx.rst`](../../mjx.rst) (`.. _Mjx:`, `.. _MjxWarp:`, `.. _MjxJAX:`),
> [`../../mjx_api.rst`](../../mjx_api.rst),
> [`../../mjwarp/index.rst`](../../mjwarp/index.rst) (`.. _MJW:`), and
> [`../../mjwarp/api.rst`](../../mjwarp/api.rst) directly when in a MuJoCo
> checkout (reserve `.. _Label:` grep for large files like `python.rst`).
> Outside a checkout, fetch
> `https://mujoco.readthedocs.io/en/latest/<path>.html` (or `/en/stable/` for
> released pages; Sphinx anchors lowercase the label, e.g. `.. _PyBatch:` →
> `#pybatch`) and browse repo paths (`../../../`) under
> `https://github.com/google-deepmind/mujoco/tree/main/`.
>
> **Related skills:**
>
> -   [mujoco-python](../python/SKILL.md) — CPU physics simulation, kinematics,
>     and full solver suite.
> -   [mujoco-parity](../parity/SKILL.md) — Cross-engine physics parity checks
>     between CPU MuJoCo and MJWarp/MJX (`forward` isolation, contact matching,
>     and `float32` baseline rollouts).
> -   [mujoco-studio](../studio/SKILL.md) — Interactive simulation and Studio
>     viewer plugins.

> [!IMPORTANT]
>
> MJX and MJWarp are **Python-only** accelerated backends. Standard CPU
> simulation runs in full `float64` via `import mujoco`.

## 1. Backend Comparison: C++ vs MJX-JAX vs MJX-Warp vs MJWarp

| Feature | C++ Engine (Default) | MJX-JAX (`impl='jax'`) | MJX-Warp (`impl='warp'`) | MJWarp (CUDA) |
| :--- | :--- | :--- | :--- | :--- |
| **Import** | `import mujoco` (`mujoco.batch`, `mujoco.rollout`) | `from mujoco import mjx` | `from mujoco import mjx` | `import mujoco_warp as mjw` |
| **Hardware** | CPU | CPU / GPU / TPU | NVIDIA GPU | NVIDIA GPU |
| **Precision** | float64 | float32 | float32 | float32 |
| **Differentiable** | ❌ No | ✅ Yes (JAX gradients) | ❌ No | ❌ No |
| **Batch Stepping** | ✅ `mujoco.batch.Batch` & `mujoco.rollout` (CPU threads) | ✅ `jax.vmap` | ✅ `jax.vmap` (CUDA kernels) | ✅ Native CUDA batches |
| **Solvers** | Newton, CG, PGS, **noslip** | Newton, CG | Newton, CG | Newton, CG |
| **Constraint Islands**| ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Plugins** | All engine plugins | ❌ No | SDF plugins only | SDF plugins only |

See [mjx.rst](../../mjx.rst) (`.. _MjxWarp:`, `.. _MjxJAX:`,
`.. _MjxFeatureParity:`) and [mjwarp/index.rst](../../mjwarp/index.rst)
(`.. _mjwDiff:`) for full feature parity and solver/collision details. For
multi-threaded CPU batch simulation supporting `float64`, `noslip`/`PGS`, and
all engine plugins without JAX or CUDA, use stateful `mujoco.batch.Batch` (for
interactive RL stepping, subset resets, and per-sim domain randomization — see
[python.rst](../../python.rst), `.. _PyBatch:`) or stateless `mujoco.rollout`
(`.. _PyRollout:`, and [programming/simulation.rst](../../programming/simulation.rst),
`.. _siBatch:`).

## 2. MJX Workflow (JAX)

MJX translates MuJoCo data structures into immutable JAX pytrees that execute
natively on GPUs and TPUs.

```python
import mujoco
from mujoco import mjx

# 1. Load CPU model and data
model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# 2. Transfer to device (GPU/TPU)
mjx_model = mjx.put_model(model)
mjx_data = mjx.put_data(model, data)

# 3. Step on device (functional paradigm returns new data)
mjx_data = mjx.step(mjx_model, mjx_data)

# 4. Copy data back to host CPU
mjx.get_data_into(data, model, mjx_data)
print(data.body('torso').xpos)
```

On NVIDIA GPUs when JAX autodiff is not required, pass `impl='warp'` to
`mjx.put_model(model, impl='warp')` and
`mjx.make_data(model, impl='warp', naconmax=..., njmax=...)` (`pip install
mujoco-mjx[warp]`) to run MJWarp's CUDA contact/constraint kernels inside JAX
(`jax.jit` / `jax.vmap`); use `Data.where(...)` for conditional per-world resets
and read [mjx.rst](../../mjx.rst) (`.. _MjxWarp:`, `.. _MjxWarpGraphModes:`,
`.. _MjxPerformanceWarp:`) for CUDA graph modes and batch rendering.

## 3. High-Throughput Rollouts: jax.vmap and jax.lax.scan

### Parallel Batches (`jax.vmap`)

```python
import jax
from mujoco import mjx

# Vectorize step over batch dimension (model unbatched, data batched)
batched_step = jax.vmap(mjx.step, in_axes=(None, 0))

# JIT-compile for maximum device throughput
batched_step_jit = jax.jit(batched_step)
batch_data = batched_step_jit(mjx_model, batch_data)
```

### Multi-Step Trajectory Rollouts (`jax.lax.scan`)

For RL environment steps, avoid Python loops by scanning across steps:

```python
def rollout(model: mjx.Model, init_data: mjx.Data, ctrl_sequence: jax.Array):
  def step_fn(d, ctrl):
    d = d.replace(ctrl=ctrl)
    d = mjx.step(model, d)
    return d, (d.qpos, d.sensordata)

  final_data, trajectory = jax.lax.scan(step_fn, init_data, ctrl_sequence)
  return final_data, trajectory
```

## 4. Static Contact and Constraint Sizing

JAX requires static array shapes, so contact and constraint buffers are sized
when the device data is created, not per step.

### MJX (default JAX implementation)

The contact buffer is sized **from the model**, not from `data.ncon`: every geom
pair that can collide (after `contype`/`conaffinity`, `exclude` and `pair`
filtering) gets a fixed number of contact slots, which is why large scenes
produce large `efc_*` arrays and slow compilation. `put_data` raises a
`ValueError` if the CPU `MjData` already holds more contacts than this capacity.
The `njmax` argument of `mjx.put_data` is ignored here.

To shrink the buffer, cap it with custom numerics in the MJCF:

```xml
<custom>
  <!-- keep only the N closest geom pairs (bounding spheres) per geom-type pair -->
  <numeric name="max_geom_pairs" data="16"/>
  <!-- keep only the N deepest contacts per condim group -->
  <numeric name="max_contact_points" data="32"/>
</custom>
```

Also reduce the number of candidate pairs with `contype`/`conaffinity` and
`<exclude>`, and prefer primitive geoms over meshes.

### MJWarp (and `mjx.put_data(..., impl='warp')`)

Buffers are sized by explicit arguments; omitted values fall back to defaults
derived from the model:

-   `nconmax`: contacts per world (MJWarp `put_data`/`make_data`).
-   `naconmax`: contacts across all worlds.
-   `njmax`: constraint rows (`efc`) **per world**, a hard cap per world.

`put_data` raises if the input `MjData` exceeds a capacity. During stepping,
excess contacts/constraints are dropped, a warning is printed, and the per-world
`d.overflow` bitmask (`mjw.OverflowType`) is set, so size for the peak.

## 5. MJX Named Access & `bind()`

MJX provides declarative named access over JAX arrays via `bind()`:

```python
spec = mujoco.MjSpec.from_file('model.xml')
model = spec.compile()
data = mujoco.MjData(model)

mx = mjx.put_model(model)
dx = mjx.put_data(model, data)

# Read body positions from JAX device arrays
torso_pos = dx.bind(mx, spec.body('torso')).xpos

# Functional update of control inputs
dx = dx.bind(mx, spec.actuators).set('ctrl', [1.0, 0.5])
```

## 6. MJWarp Guide (NVIDIA CUDA)

MJWarp (`mujoco_warp`) provides a CUDA-native simulator optimized for maximum
batch throughput on NVIDIA hardware without JAX:

```python
import mujoco
import mujoco_warp as mjw

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

# Transfer to CUDA: one model, nworld copies of the data
mjw_model = mjw.put_model(model)
mjw_data = mjw.put_data(model, data, nworld=4096, nconmax=64, njmax=256)
# or allocate fresh: mjw.make_data(model, nworld=4096, nconmax=64, njmax=256)

# Advance all 4096 environments in parallel (in place)
mjw.step(mjw_model, mjw_data)

# Fields are Warp arrays with a leading world dimension
qpos = mjw_data.qpos.numpy()  # (4096, nq)

# Copy one world back to CPU MjData
mjw.get_data_into(data, model, mjw_data, world_id=0)
```

## 7. Feature Parity, Tuning & Gotchas

Instead of relying on static feature snapshots, check the authoritative parity
and performance sections in the RST manual when choosing colliders, solvers, or
tuning throughput:

-   **Supported geometries, solvers & differences from CPU MuJoCo**:
    [mjx.rst](../../mjx.rst) (`.. _MjxFeatureParity:`, `.. _MjxSharpBits:`) and
    [mjwarp/index.rst](../../mjwarp/index.rst) (`.. _mjwDiff:`,
    `.. _mjwPairwise:`, `.. _mjwBoxBox:`). Notably, `noslip` and `PGS` solvers
    are CPU-only, and MJX-JAX solves constraints monolithically without
    constraint islands.
-   **Performance tuning, CUDA graphs & domain randomization**:
    [mjx.rst](../../mjx.rst) (`.. _MjxPerformance:`, `.. _MjxPerformanceJAX:`,
    `.. _MjxPerformanceWarp:`), [mjwarp/index.rst](../../mjwarp/index.rst)
    (`.. _mjwPerf:`, `.. _mjwGC:`, `.. _mjwOverflow:`, `.. _mjwBatch:`), and
    [modeling.rst](../../modeling.rst) (`.. _CTuning:`, `.. _CTuningGlobal:`,
    `.. _CTuningModel:`) for general timestep, solver/iteration, friction-cone,
    and primitive-vs-mesh collision trade-offs.

### High-Frequency Gotchas

1.  **Float32 precision**: GPU backends run in single precision (`float32`).
    Stiff systems may experience numerical drift or NaN errors. Counteract this
    by increasing `model.opt.iterations` or decreasing `model.opt.timestep`.
    For comparing `float32` GPU physics against CPU MuJoCo, refer to
    [`mujoco-parity`](../parity/SKILL.md).
2.  **Immutability in MJX**: `mjx.Data` is a JAX pytree. In-place assignments
    fail; use `.replace()`:
    ```python
    mjx_data = mjx_data.replace(qpos=new_qpos)
    ```
3.  **Compilation caching**: Avoid long recompilations on startup by enabling
    persistent XLA caching:
    ```python
    import jax
    jax.config.update('jax_compilation_cache_dir', '/tmp/jax_cache')
    ```

## 8. Key References

### Documentation

-   [MJX Guide (`MJX-JAX` & `MJX-Warp`)](../../mjx.rst) (`.. _Mjx:`,
    `.. _MjxWarp:`, `.. _MjxJAX:`, `.. _MjxFeatureParity:`,
    `.. _MjxPerformance:`, `.. _MjxSharpBits:`) and
    [MJX API Reference](../../mjx_api.rst)
-   [MuJoCo Warp (MJWarp) Guide](../../mjwarp/index.rst) (`.. _MJW:`,
    `.. _MJW_Usage:`, `.. _mjwPerf:`, `.. _mjwOverflow:`, `.. _mjwBatch:`,
    `.. _mjwDiff:`) and [MJWarp API Reference](../../mjwarp/api.rst)
-   [Python Bindings (`mujoco.batch` & `mujoco.rollout`)](../../python.rst)
    (`.. _PyNamed:`, `.. _PyBatch:`, `.. _PyRollout:`),
    [Model Tuning Guide](../../modeling.rst) (`.. _CTuning:`), and
    [MuJoCo Menagerie](../../models.rst) (`.. _Menagerie:`)

### Notebooks, Models & RL Ecosystem

-   **Tutorials & tests**: `mjx/tutorial.ipynb` (`.. _MjxNotebook:` in
    [mjx.rst](../../mjx.rst)),
    [MJX Support Tests](../../../mjx/mujoco/mjx/_src/support_test.py),
    [Humanoid Models](../../../model/humanoid/humanoid.xml), and
    [Benchmark Test Models](../../../test/benchmark/testdata/100_humanoids_chol.xml)
-   **RL & simulation ecosystem** (see
    [mjwarp/index.rst](../../mjwarp/index.rst), `.. _mjwFAQ:`):
    -   [MuJoCo Playground (`google-deepmind/mujoco_playground`)](https://github.com/google-deepmind/mujoco_playground)
        — JAX/MJX (`MJX-JAX` & `MJX-Warp`) RL environments for locomotion,
        manipulation, dexterous hands, and vision.
    -   [`mjlab` (`mujocolab/mjlab`)](https://github.com/mujocolab/mjlab) —
        Isaac Lab-style manager API for training policies directly with
        `mujoco_warp` and PyTorch.
    -   [MuJoCo Warp (`google-deepmind/mujoco_warp`)](https://github.com/google-deepmind/mujoco_warp)
        — Standalone `mujoco_warp` repository (`notebooks/tutorial.ipynb`,
        benchmarks, and Warp kernels).
    -   [MuJoCo MPC (`google-deepmind/mujoco_mpc`)](https://github.com/google-deepmind/mujoco_mpc)
        — Real-time model-predictive control and MJX planning baselines.
