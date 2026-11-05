# MuJoCo Agent Skills

This directory contains workflow guides and guardrails for AI coding agents and
developers working with MuJoCo across C++ and Python.

## Official Documentation (`doc/*.rst`)

These skills focus on concise code skeletons, idiomatic patterns, and
high-frequency gotchas (`# Good:` vs `# Avoid:`). They complement—rather than
duplicate—the official MuJoCo Sphinx manual in `../index.rst`:

- **Local `.rst` files + Sphinx anchors**: Relative links in each skill (such as
  `../../modeling.rst` with `` `.. _CBodyOrdering:` ``) point directly into this
  `doc/` tree. Read short `.rst` chapters (such as `../studio/*.rst`) directly;
  for large reference files like `../XMLreference.rst` (e.g.,
  `` `.. _sensor-contact:` ``) or `../APIreference/functions.rst` (e.g.,
  `` `.. _mj_step:` ``), grep for the literal `.. _Label:` string to jump to the
  target section.
- **Online fallback (standalone skill installs)**: If `doc/skills/` is installed
  outside a MuJoCo checkout, every `doc/<path>.rst` page is published at
  https://mujoco.readthedocs.io/en/stable/ as `<path>.html` (or `/en/latest/`
  for `main` and unreleased chapters like `studio/`;
  `APIreference/functions.rst` renders inside `APIreference/APIfunctions.html`,
  and Sphinx anchors lowercase the label as `#<lowercase-label>`, e.g.
  `.. _StudioDiffSync:` → `#studiodiffsync`), and repo-relative paths
  (`../../model/`, `../../test/`, `../../sample/`, `python/*.ipynb`) live under
  https://github.com/google-deepmind/mujoco/tree/main/.
- **Core manual chapters**: `../overview.rst` | `../modeling.rst` |
  `../XMLreference.rst` | `../models.rst` (`.. _Menagerie:`) |
  `../computation/index.rst` | `../computation/speculative.rst` |
  `../computation/fluid.rst` | `../programming/index.rst` |
  `../programming/samples.rst` | `../programming/extension.rst` |
  `../OpenUSD/index.rst` | `../python.rst` | `../mjx.rst` |
  `../mjwarp/index.rst` | `../studio/index.rst` | `../APIreference/index.rst`.

## Reference Models, Test Fixtures & Samples

Before inventing synthetic XMLs or guessing syntax for specialized features,
check the canonical models and samples shipped with MuJoCo:

- **Showcase MJCF models (`../../model/`)**: Self-contained models covering
  articulated robots and mechanisms (`humanoid/`, `car/`, `tendon_arm/`,
  `slider_crank/`, `mug/`) as well as specialized physics features (`flex/`,
  `replicate/`, `plugin/`, `tactile/`, `adhesion/`, `sleep/`, `speculative/`,
  `cards/`, `cube/`, `hammock/`, `balloons/`).
- **MuJoCo Menagerie (`../models.rst`, `` `.. _Menagerie:` ``)**: Curated,
  ready-to-simulate robot descriptions (Franka Panda/FR3, UR5e/UR10e, Unitree
  G1/H1/Go2, Spot, Barkour, Shadow Hand, Robotiq 2F-85, ALOHA 2, Skydio X2,
  Flybody) at https://github.com/google-deepmind/mujoco_menagerie.
- **Minimal feature & unit-test XMLs (`../../test/**/testdata/`)**: Compact
  single-feature fixtures for engine physics (`test/engine/testdata/`),
  `MjSpec`/compiler (`test/user/testdata/`), XML attachments & OpenUSD
  (`test/xml/testdata/`), Filament PBR materials & lighting
  (`test/render/testdata/`), and multi-body benchmarks
  (`test/benchmark/testdata/`).
- **C++ & Python samples**: `../../sample/*.cc` (`../programming/samples.rst`,
  `` `.. _Sample:` ``) and `../../python/mujoco/sample/`.
- **Interactive Colab / Jupyter notebooks**: `python/tutorial.ipynb`
  (`` `.. _PyNotebook:` ``), `python/mjspec.ipynb` (`MjSpec` & `PyMJCF`
  comparison), `python/rollout.ipynb` (`mujoco.rollout`), `python/LQR.ipynb`
  (`mjd_transitionFD` & LQR), `python/least_squares.ipynb`
  (`mujoco.minimize.least_squares`), `python/mujoco/sysid/sysid.ipynb`
  (`mujoco.sysid`, `` `.. _PySysid:` ``), and `mjx/tutorial.ipynb`
  (`` `.. _MjxNotebook:` ``).
- **RL, Control & Simulation Ecosystem**:
  - **MuJoCo Playground**
    (https://github.com/google-deepmind/mujoco_playground) — JAX/MJX
    (`MJX-JAX` & `MJX-Warp`) RL environments for locomotion, manipulation,
    dexterous hands, and vision.
  - **mjlab** (https://github.com/mujocolab/mjlab) — Isaac Lab-style modular
    manager API for training policies directly with `mujoco_warp` and PyTorch
    (see `../mjwarp/index.rst`, `` `.. _mjwFAQ:` ``).
  - **MuJoCo Warp** (https://github.com/google-deepmind/mujoco_warp, mirrored in
    `../mjwarp/index.rst`) — Standalone NVIDIA Warp GPU physics repository
    (`notebooks/tutorial.ipynb`, benchmarks, and kernel sources).
  - **MuJoCo MPC (MJPC)** (https://github.com/google-deepmind/mujoco_mpc) —
    Interactive real-time model-predictive control and trajectory optimization.
  - **dm_control** (https://github.com/google-deepmind/dm_control) — Python RL
    suites and legacy `PyMJCF` (see `../python.rst`, `` `.. _PyMJCF:` ``).

## Skill Catalog

| Skill | Directory | Primary Scope | Key APIs & Capabilities |
| :--- | :--- | :--- | :--- |
| **`mujoco-python`** | [`python/`](python/SKILL.md) | Python simulation lifecycle & physics | `MjSpec` → `MjModel` → `MjData`, `mj_step`, `mj_forward`, named access, `bind()`, `mujoco.batch` (`Batch`), `mujoco.rollout`, `mujoco.minimize`, `mujoco.sysid`, sensors, `mju_*` math. |
| **`mujoco-spec-editing`** | [`spec_editing/`](spec_editing/SKILL.md) | Procedural model authoring & tuning | Building bodies, geoms, joints, sub-spec attachments with prefixes, actuators (servos, motors), collision bitmasks (`contype`/`conaffinity`), default classes, baseline `<option>`/solver/contact tuning (`.. _CTuning:`). |
| **`mujoco-rendering`** | [`rendering/`](rendering/SKILL.md) | Offscreen & PBR rendering | Headless RGB, metric depth maps, segmentation masks (`mujoco.Renderer`), Filament PBR (`mujoco.mjrf`, `mujoco.rendering.filament`), C++ `mjr`/`mjrf` pipelines. |
| **`mujoco-accelerated`** | [`accelerated/`](accelerated/SKILL.md) | GPU & TPU batch simulation | High-throughput batch rollouts with MJX (`MJX-JAX`, `MJX-Warp`, `jax.vmap`, `jax.lax.scan`), differentiability, CUDA batching with MJWarp (`mujoco_warp`), contact/constraint sizing. |
| **`mujoco-parity`** | [`parity/`](parity/SKILL.md) | Cross-engine physics parity | Comparing CPU (`float64`) vs MJWarp/MJX (`float32`): stage-by-stage `forward` isolation, stiff actuator & `qM` rounding bounds, contact matching, batched 1-step & `float32` baseline rollouts. |
| **`mujoco-gui`** | [`gui/`](gui/SKILL.md) | Dear ImGui & ImPlot UI design | Immediate-mode UI patterns, layout, styling, ID stack rules (`##` vs `###`), list virtualization (`ImGuiListClipper`), automated testing hooks. |
| **`mujoco-studio`** | [`studio/`](studio/SKILL.md) | Studio apps & interactive viewers | Standalone desktop app (`.. _StudioApp:`) & Python viewers, decoupled Sim/Viewer threads, message channels (Snapshots vs Events), plugin architecture in Python and C++, hot reload, WebViewer WASM streaming. |

## How to Use These Skills

- **For AI Coding Agents**: Each skill contains standardized YAML frontmatter
  with `Use when:` triggers and `Don't use for:` exclusions to assist automated
  orchestrators in choosing the relevant guide.
- **For Developers**: Skills pair quick-start recipes and common pitfalls with
  targeted pointers into the `doc/*.rst` reference manual.
