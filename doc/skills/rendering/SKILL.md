---
name: mujoco-rendering
description: >-
  Offscreen image rendering, Python Renderer class, C++ pipeline (mjrContext,
  mjvScene, mjr_render, mjr_readPixels), RGB, metric depth maps, segmentation
  masks (mjtObj), offwidth/offheight framebuffer sizing, MUJOCO_GL (osmesa, egl),
  cameras (fixed, tracking, free, MjvCamera), scene options (geomgroup, flags).
  Use for headless camera rendering, dataset recording, visual observations.
  Do NOT use for interactive viewers/Studio (mujoco-studio) or ImGui controls (mujoco-gui).
---

# MuJoCo Rendering: Cameras, Contexts & Offscreen Buffers

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`):** Read
> [`../../programming/rendering.rst`](../../programming/rendering.rst)
> (`.. _Rendering:`, `.. _FilamentRendering:`, `.. _ClassicRendering:`) and
> [`../../programming/visualization.rst`](../../programming/visualization.rst)
> (`.. _Visualization:`) directly when in a MuJoCo checkout; for large
> reference files like [`../../python.rst`](../../python.rst) (`.. _PyRender:`),
> [`../../XMLreference.rst`](../../XMLreference.rst), and
> [`../../APIreference/functions.rst`](../../APIreference/functions.rst), grep
> the literal `.. _Label:` anchor. Outside a checkout, fetch
> `https://mujoco.readthedocs.io/en/latest/<path>.html` (or `/en/stable/` for
> released pages; `functions.rst` renders inside
> `APIreference/APIfunctions.html`, and Sphinx anchors lowercase the label,
> e.g. `.. _PyRender:` → `#pyrender`) and browse repo paths (`../../../`) under
> `https://github.com/google-deepmind/mujoco/tree/main/`.
>
> **Related skills:**
>
> -   [mujoco-python](../python/SKILL.md) — Core physics simulation, stepping,
>     and state management.
> -   [mujoco-studio](../studio/SKILL.md) — Interactive simulation, Studio
>     viewer apps, and plugin architecture.
> -   [mujoco-gui](../gui/SKILL.md) — Dear ImGui UI design and custom controls.

## 1. Overview: Rendering Backends & Workflows

MuJoCo provides two built-in 3D rendering backends (compared side-by-side in
[programming/rendering.rst](../../programming/rendering.rst), `.. _Rendering:`):

-   **Classic OpenGL (`mujoco.Renderer` / C `mjr`)**: Immediate-mode OpenGL 1.5
    renderer (`mjv_updateScene` -> `mjr_render` -> `mjr_readPixels`), now in
    maintenance mode. Bundled out of the box in `pip install mujoco`, making
    `mujoco.Renderer` the simplest single-class helper for headless RGB, depth,
    and segmentation capture. See
    [programming/rendering.rst](../../programming/rendering.rst)
    (`.. _ClassicRendering:`) and [python.rst](../../python.rst)
    (`.. _PyClassicRender:`).
-   **Filament PBR (`mujoco.mjrf` & `mujoco.rendering.filament` / C `mjrf`)**:
    Retained-mode, asynchronous physically-based renderer supporting OpenGL,
    Vulkan, Metal, and WebGL (owns its own headless or windowed GPU context and
    arbitrary-size `mjrfRenderTarget` buffers). Recommended for PBR materials,
    image-based/photometric lighting, and new C++/Studio applications; see §7
    and [programming/rendering.rst](../../programming/rendering.rst)
    (`.. _FilamentRendering:`).
-   **Interactive Viewers (`mujoco.studio` / `mujoco.viewer`)**: Desktop or
    web-streamed interactive viewers; use [`mujoco-studio`](../studio/SKILL.md)
    instead.

## 2. Headless Contexts on Linux: `MUJOCO_GL`

On headless Linux systems (servers, containers), configure the OpenGL context
provider via the `MUJOCO_GL` environment variable **before importing `mujoco`**:

```python
import os
os.environ['MUJOCO_GL'] = 'egl'      # Hardware-accelerated offscreen on NVIDIA GPUs
# os.environ['MUJOCO_GL'] = 'osmesa' # Portable CPU software rasterization

import mujoco  # Must import AFTER setting MUJOCO_GL
```

Or from the command line:
```bash
MUJOCO_GL=egl python3 my_script.py
```

## 3. The Python Renderer Class (RGB, Depth & Segmentation)

`mujoco.Renderer` manages GL context creation, scene geometry synchronization,
and buffer readback. **Always use it as a context manager**:

```python
import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path('scene.xml')
data = mujoco.MjData(model)

with mujoco.Renderer(model, height=480, width=640) as renderer:
  mujoco.mj_forward(model, data)

  # 1. RGB Rendering
  renderer.update_scene(data, camera='overhead_cam')
  rgb = renderer.render()  # Shape: (480, 640, 3), dtype: uint8

  # 2. Metric Depth Rendering (distance in meters)
  renderer.enable_depth_rendering()
  renderer.update_scene(data, camera='overhead_cam')
  depth = renderer.render()  # Shape: (480, 640), dtype: float32
  renderer.disable_depth_rendering()

  # 3. Semantic Segmentation Rendering
  renderer.enable_segmentation_rendering()
  renderer.update_scene(data, camera='overhead_cam')
  seg = renderer.render()  # Shape: (480, 640, 2), dtype: int32
  # seg[:, :, 0] = object ID (-1 for background)
  # seg[:, :, 1] = object type (mjtObj enum, e.g. mjOBJ_GEOM)
  renderer.disable_segmentation_rendering()
```

### Rendering Inside Simulation Loops

Creating a `Renderer` allocates GPU framebuffers. In RL or evaluation loops,
**create the renderer once outside the loop**:

```python
with mujoco.Renderer(model, height=84, width=84) as renderer:
  for step in range(1000):
    mujoco.mj_step(model, data)
    if step % render_interval == 0:
      renderer.update_scene(data, camera='agent_cam')
      # Call .copy() because the renderer's internal buffer is reused
      frame = renderer.render().copy()
```

## 4. C++ Offscreen Rendering Pipeline (mjr & mjv)

In C++, offscreen rendering is performed using the `mjvScene` and `mjrContext`
APIs:

```cpp
#include <mujoco/mujoco.h>

// 1. Initialize visualization structures
mjvCamera cam;
mjv_defaultCamera(&cam);
cam.type = mjCAMERA_FIXED;
cam.fixedcamid = mj_name2id(m, mjOBJ_CAMERA, "overhead_cam");

mjvOption opt;
mjv_defaultOption(&opt);

mjvScene scn;
mjv_makeScene(m, &scn, 2000);  // Buffer up to 2000 geoms

// 2. Initialize offscreen rendering context
mjrContext con;
mjr_defaultContext(&con);
mjr_makeContext(m, &con, mjFONTSCALE_100);

// 3. Render frame
mjrRect viewport = {0, 0, 640, 480};
mjv_updateScene(m, d, &opt, nullptr, &cam, mjCAT_ALL, &scn);
mjr_render(viewport, &scn, &con);

// 4. Read pixels into user buffers
unsigned char rgb[640 * 480 * 3];
float depth[640 * 480];
mjr_readPixels(rgb, depth, viewport, &con);

// 5. Clean up
mjr_freeContext(&con);
mjv_freeScene(&scn);
```

## 5. Camera Definition & Placement

MuJoCo cameras look down their **negative Z-axis**:
-   **-Z**: Viewing forward direction
-   **+X**: Right
-   **+Y**: Up

### Defining Cameras in MjSpec

```python
# Fixed camera pointing down at ground plane
cam = spec.worldbody.add_camera(
    name='overhead',
    pos=[0, 0, 3.0],
    quat=[0.707, 0.707, 0, 0],  # 90 deg rotation around X
    fovy=60,
)

# Camera attached to robot wrist
wrist_cam = wrist_body.add_camera(
    name='wrist_cam',
    pos=[0.05, 0, 0],
    xyaxes=[0, -1, 0, 0, 0, -1],
    fovy=75,
)
```

### Free Camera (`MjvCamera`)

For custom interactive or orbital viewpoints:

```python
cam = mujoco.MjvCamera()
cam.type = mujoco.mjtCamera.mjCAMERA_FREE
cam.lookat[:] = [0, 0, 0.5]
cam.distance = 2.5
cam.azimuth = 90.0
cam.elevation = -20.0

renderer.update_scene(data, camera=cam)
```

## 6. Framebuffers & Scene Options

### Offscreen Framebuffer Resolution

The offscreen rendering resolution cannot exceed the dimensions pre-allocated in
the model. Adjust the buffer limit in the spec before compilation:

```python
spec.visual.global_.offwidth = 1920
spec.visual.global_.offheight = 1080
model = spec.compile()
```

### The Single `<visual>` Block Rule

MuJoCo uses only the **last** `<visual>` block defined in an XML file. Multiple
visual blocks will silently overwrite earlier settings:

```xml
<!-- ❌ WRONG — second visual block silently resets offwidth/offheight to default -->
<visual><global offwidth="1920" offheight="1080"/></visual>
<visual><headlight ambient="0.3 0.3 0.3"/></visual>

<!-- ✅ RIGHT — consolidate into one visual block -->
<visual>
  <global offwidth="1920" offheight="1080"/>
  <headlight ambient="0.3 0.3 0.3"/>
</visual>
```

### Geom Group Visibility & Visualization Flags

Control rendering layers via `MjvOption`:

```python
opt = mujoco.MjvOption()
opt.geomgroup[:] = False   # Hide all groups
opt.geomgroup[0] = True    # Show primary geometry (default group)
opt.geomgroup[1] = True    # Show secondary visuals

# Toggle visual debugging flags
opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

renderer.update_scene(data, camera='overhead', scene_option=opt)
```

## 7. Filament Physically-Based Rendering (`mjrf`)

MuJoCo's Filament backend (`mjrfilament.h` in C/C++, `from mujoco import mjrf`
and `mujoco.rendering.filament` in Python) provides retained-mode PBR with
OpenGL, Vulkan, Metal, and WebGL support, arbitrary `mjrfRenderTarget` sizes
(independent of `<global offwidth offheight/>`), and asynchronous
`mjrfReadRequest` pixel readback:

-   **Python (`ModelScene` + `Renderer`)**: Create an `mjrf.Context`, bind
    model lights and renderables once via
    `mujoco.rendering.filament.model_scene.ModelScene`, update in place with
    `ms.update(data)`, and render into a NumPy target buffer via
    `mujoco.rendering.filament.renderer.Renderer` — see the complete example in
    [python.rst](../../python.rst) (`.. _PyFilamentRender:`).
-   **C/C++ & CMake**: Build with `-DMUJOCO_USE_FILAMENT=ON` (or
    `-DMUJOCO_USE_FILAMENT_MJR_COMPAT=ON` for drop-in `mjr_` compatibility) and
    consult [programming/rendering.rst](../../programming/rendering.rst)
    (`.. _reProgramming:`, `.. _reSupport:`).
-   **Photometric light-intensity gotcha**: If *any* `<light>` has positive
    `intensity`, Filament switches the entire scene from classic headlight +
    default environment lighting to photometric mode (turning off the headlight
    and darkening zero-intensity lights). For PBR materials, textures, and
    lighting authoring, read
    [programming/rendering.rst](../../programming/rendering.rst)
    (`.. _reAuthoring:`, `.. _reMaterials:`, `.. _reLighting:`) and inspect the
    reference PBR/IBL test scenes in `../../../test/render/testdata/`
    (`pbr.xml`, `lighting.xml`, `emissive_bloom.xml`, `reflect_pbr.xml`,
    `ibl.ktx`) and `../../../model/mug/mug.xml`.

## 8. Common Gotchas & Best Practices

1.  **GL context leaks**: Always close `mujoco.Renderer` or use a `with` block.
    Failing to free renderers in loops exhausts GL contexts.
2.  **Destruction order**: In C++, `mjr_freeContext` must be called before the
    underlying OpenGL context is destroyed (see
    [programming/rendering.rst](../../programming/rendering.rst),
    `.. _reContext:`).
3.  **Buffer re-use**: `renderer.render()` returns an internal buffer view. Call
    `.copy()` if storing frames in a list.
4.  **Camera lookup by name**: Pass `camera='camera_name'` or the integer ID.
    Avoid passing raw string IDs that are not present in the model.

## 9. Key References

### Documentation

-   [Rendering Guide (Filament `mjrf` & Classic OpenGL `mjr`)](../../programming/rendering.rst)
    (`.. _Rendering:`, `.. _FilamentRendering:`, `.. _ClassicRendering:`)
-   [Abstract Visualization (`mjvCamera`, `mjvOption`, `mjvScene`)](../../programming/visualization.rst)
    (`.. _Visualization:`, `.. _viCamera:`, `.. _viOption:`)
-   [Python Rendering Bindings](../../python.rst) (`.. _PyRender:`,
    `.. _PyFilamentRender:`, `.. _PyClassicRender:`)
-   [MJCF `<visual>` & `<camera>` Reference](../../XMLreference.rst)
    (`.. _visual-global:`, `.. _body-camera:`),
    [Modeling Cameras](../../modeling.rst) (`.. _CCamera:`), and
    [MuJoCo Menagerie](../../models.rst) (`.. _Menagerie:`)
-   [C Rendering API Functions](../../APIreference/functions.rst)
    (`.. _Visualization-api:`, `.. _OpenGLrendering:`,
    `.. _FilamentRenderingApi:`) and
    [C++ Render/Record Samples](../../programming/samples.rst) (`.. _saRender:`,
    `.. _saRecord:`)

### Source Code & Model Examples

-   [C API Headers](../../../include/mujoco/mujoco.h) and
    [`mjrfilament.h`](../../../include/mujoco/mjrfilament.h)
-   [C++ Offscreen & Filament Sample](../../../sample/render.cc) and
    [Filament PBR Test Scenes](../../../test/render/testdata/pbr.xml)
