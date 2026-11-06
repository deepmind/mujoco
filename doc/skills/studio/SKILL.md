---
name: mujoco-studio
description: >-
  Builds and launches interactive MuJoCo Studio viewers and plugins in Python
  and C++, covering decoupled sim/viewer threads, Snapshot and Event messaging,
  lifecycle events, WebViewer browser streaming, and Python hot reload. Use when
  launching Studio viewers (launch_web, launch_native, launch_passive,
  LaunchStudio), authoring Python ViewerPlugin/SimPlugin or C++ RegisterPlugin
  (mjPLUGIN_LIB_INIT) extensions, synchronizing sim and viewer state
  (ViewerHandle.sync, @messages.handler), hot-reloading Python plugins or models
  (HotReloadEvent), iterating on a live hot-reload Studio session, configuring
  graphics modes or WebViewer WASM streaming, or debugging Studio and
  browser console logs. Don't use for Dear ImGui/ImPlot widget layout or styling
  (use mujoco-gui), core physics stepping (use mujoco-python), or headless
  offscreen rendering (use mujoco-rendering).
---

# MuJoCo Studio: Interactive Simulation, UI & Plugin Architecture

Studio replaces the legacy `simulate` tools with a modular interactive
visualization stack built on [Dear ImGui](https://github.com/ocornut/imgui) and
ImPlot for UI, Google Filament (PBR with Vulkan, OpenGL, Metal, and WebGL
backends) for rendering, and out-of-the-box browser streaming via `WebViewer`
(WASM). Its default interface, the **Studio Viewer App**, is designed
around composable **Studio Plugins** that teams can share across environment
authoring (e.g., Gymnasium or robotics suites), asset inspection/loading tools,
and agent monitoring/control workflows.

> [!TIP]
>
> **Official MuJoCo documentation (`doc/*.rst`):** Detailed Studio architecture,
> lifecycle tables, and C++/web docs live in
> [`../../studio/`](../../studio/index.rst)—read these short `.rst` chapters
> directly when present locally or in `pip install mujoco` under
> `pathlib.Path(mujoco.__file__).parent / 'doc'` (reserve `.. _Label:` grep for
> large files like `XMLreference.rst`), or fetch
> `https://mujoco.readthedocs.io/en/latest/studio/<page>.html` (or `/en/stable/`
> once released; Sphinx anchors lowercase the label, e.g. `#studiodiffsync`).
>
> **Related skills:**
>
> -   [mujoco-gui](../gui/SKILL.md) — Dear ImGui and ImPlot widget layout,
>     Tables API alignment, `##` vs `###` IDs, and `ImGuiListClipper`.
> -   [mujoco-python](../python/SKILL.md) — Core simulation lifecycle
>     (`MjSpec` -> `MjModel` -> `MjData`), named access, and spatial math.
> -   [mujoco-spec-editing](../spec_editing/SKILL.md) — Procedural `MjSpec`
>     modeling, attachments, and runtime recompilation.
> -   [mujoco-rendering](../rendering/SKILL.md) — Headless offscreen RGB, depth,
>     and segmentation rendering.

## Official Studio Docs (Load on Demand)

Instead of duplicating reference tables here, read the matching `.rst` chapter
when your task touches these areas:

-   **[studio/index.rst](../../studio/index.rst)** (`.. _Studio:`,
    `.. _StudioQuickStart:`, `.. _StudioComparison:`) — Overview, CLI quick
    start, and feature comparison across `NativeViewer`, `WebViewer`, C++
    Studio, MuJoCo Live, and legacy `simulate` / `mujoco.viewer`.
-   **[studio/architecture.rst](../../studio/architecture.rst)**
    (`.. _StudioThreads:`, `.. _StudioCppComponents:`, `.. _StudioMessaging:`,
    `.. _StudioSnapshotsVsEvents:`) — Sim/viewer thread split in Python & C++,
    when to implement heavy or WASM-portable widgets in C++ (`ux/`) vs Python
    (GIL release, MuJoCo Live vs `WebViewer`), and `Snapshot` vs `Event` channel
    semantics.
-   **[studio/native_viewers.rst](../../studio/native_viewers.rst)**
    (`.. _StudioApp:`, `.. _StudioPythonNative:`, `.. _StudioLaunchers:`,
    `.. _StudioExecutionModes:`, `.. _StudioBackends:`) — Prebuilt standalone
    desktop application (`MuJoCoStudio.app` on macOS, `bin\mujoco_studio.exe` on
    Windows, `bin/mujoco_studio` on Linux) vs Python `NativeViewer` (and when
    shared custom components belong in C++ bound to Python), `launch_native` /
    `launch_web` / `launch_passive`, blocking `.run(...)` vs non-blocking
    `.launch(...)`, and `ViewerConfig.gfx` modes (`opengl`, `vulkan`,
    `opengl_headless`, `opengl_software`, `vulkan_software`, `web`).
-   **[studio/web_viewers.rst](../../studio/web_viewers.rst)**
    (`.. _StudioLive:`, `.. _StudioWebViewer:`,
    `.. _StudioBrowserSandboxing:`) — MuJoCo Live (`?model=github:...`),
    `WebViewer` `/ui` + `/model` + `/state` + `/drop` streams, Controller vs
    Spectator roles, COOP/COEP + `localhost`/HTTPS `SharedArrayBuffer` rules
    (`ssh -L 8080:localhost:8080`), and persistent C++ global/static state
    across WASM and sequential Python sessions.
-   **[studio/python_plugins.rst](../../studio/python_plugins.rst)**
    (`.. _StudioViewerPlugins:`, `.. _StudioSceneOverlays:`,
    `.. _StudioImGuiTips:`, `.. _StudioSplitPlugins:`, `.. _StudioLifecycle:`,
    `.. _StudioDiffSync:`, `.. _StudioPriorities:`, `.. _StudioHotReload:`) —
    Authoring `ViewerPlugin` & `SimPlugin` extensions, `viewer.extra_geoms`
    (`MjvGeom`) vs `mujoco.mjrf`, full Viewer & Sim lifecycle event tables
    (including `HotReloadEvent`), `send_if_changes` diff sync, handler
    priorities, and `--hot_reload`.
-   **[studio/cpp.rst](../../studio/cpp.rst)** (`.. _StudioCppLauncher:`,
    `.. _StudioCppPlugins:`, `.. _StudioLiveCpp:`) — C++ `LaunchStudio`,
    `LauncherConfig`, `GuiPlugin` / `ScenePlugin` / `ModelPlugin` /
    `KeyHandlerPlugin` / `SpecEditorPlugin` callbacks, `mjPLUGIN_LIB_INIT` vs
    explicit `RegisterPlugin`, and the Emscripten WASM build.

## 1. Architecture & Threading Model

### Studio vs Legacy `simulate`

| Feature | Legacy `simulate` / `mujoco.viewer` | Studio (`mujoco.studio`) |
| :--- | :--- | :--- |
| **GUI & Plotting** | Fixed `mjui` panels & `mjr_figure` | **Dear ImGui** & **ImPlot** |
| **Render Backend** | Classic OpenGL (`mjrContext`) | **Filament** PBR (Vulkan, OpenGL, Metal, WebGL) |
| **Web & Remote** | None | **WebViewer** (streamed ImGui UI + state over HTTP/WS) & **MuJoCo Live** (WASM) |
| **Extensibility** | Hardcoded panels and keybinds | **Plugin Architecture** in Python and C++ |
| **Threading & Sync** | Shared `model`/`data` with locking | Python & desktop C++: decoupled sim/viewer threads; WASM (`MuJoCo Live`): single-threaded |

### Decoupled Sim & Viewer Threads (Python)

`viewer_protocol.py` and `viewer_handle.py` decouple simulation execution from
rendering and UI through non-blocking message endpoints:

```text
┌────────────────────────────────────────────────────────┐
│              Simulation Thread (Caller)                │
│  - Owns MjModel and MjData                             │
│  - Runs physics loop, RL policies, or MPC planners     │
│  - Executes SimPlugin instances (e.g. StepControl)     │
│  - Drives simulation via handle.sync()                 │
└────────────────────────────────────────────────────────┘
                        ▲
                        │   Queued Events & Snapshots
                        │ (SimEndpoint <-> ViewerEndpoint)
                        ▼
┌────────────────────────────────────────────────────────┐
│              Viewer Thread (or Web Client)             │
│  - Owns Window, Renderer, Camera, VisOptions, Perturb  │
│  - Owns local copy of MjModel and MjData for rendering │
│  - Executes ViewerPlugin instances (studio_app, UI)    │
│  - Dispatches lifecycle events (UpdateEvent, BuildGuiEvent) │
└────────────────────────────────────────────────────────┘
```

-   **Priority-based Plugin System (`plugin.py`, `plugin_registry.py`)**:
    `plugin_registry.PluginRegistry` discovers `@messages.handler` methods and
    dispatches messages in priority order. Launchers accept a single
    `plugins=[...]` list and partition instances by base class:
    -   **`plugin.SimPlugin`**: Runs on the simulation thread, automatically
        caches `self.handle` on `SimInitEvent`, and handles physics stepping,
        resets, perturbations, model reloads, and policy updates.
    -   **`plugin.ViewerPlugin`**: Runs on the viewer thread, automatically
        caches `self.viewer` on `ViewerInitEvent`, and handles visual updates,
        rendering overlays, and Dear ImGui construction.
-   **Canonical Samples (`python/mujoco/sample/studio/`)**:
    -   `python/mujoco/studio/viewer.py` — Standard Studio viewer
        (`studio_app.App`); run with
        `viewer.py --model=<file> --plugins=a.py,b.py` (instantiates every
        `ViewerPlugin`/`SimPlugin` defined in those files).
    -   `python/mujoco/sample/studio/ghost.py` — `GhostRenderer` injecting transient `extra_geoms` on
        `UpdateEvent` and ImGui controls on `BuildGuiEvent`
        (`viewer.py --model=... --plugins=python/mujoco/sample/studio/ghost.py`).
    -   `python/mujoco/sample/studio/implot.py` — `BodyInspector` demonstrating real-time ImGui +
        ImPlot charting (`viewer.py --model=... --plugins=python/mujoco/sample/studio/implot.py`).
    -   `python/mujoco/sample/studio/custom_sim_loop.py` — External script owning its stepping and
        pacing loop via `handle.sync(model, data)`.
    -   `python/mujoco/sample/studio/python_scene.py` — `MyScene` and
        `MySceneGui` demonstrating a hot-reloading scene built in Python paired
        with UI (run `viewer.py --hot_reload=all` with
        `--plugins=python/mujoco/sample/studio/python_scene.py`).

## 2. Launching Python Viewers

Studio provides three runner modules sharing the same `.run(...)` and
`.launch(...)` interfaces:

1.  **`launch_web.run`** (`python/mujoco/studio/launch_web.py`) —
    **Recommended default.** Spawns `WebViewer` in a background daemon thread
    with **zero host graphics dependencies** (no X11/Wayland display server or
    host Filament libraries required): physics and headless Dear ImGui layout
    run on the host while UI draw commands and state stream over
    WebSockets to a browser WASM client. Ideal for remote workstations,
    containers, and headless servers.
2.  **`launch_native.run`**
    (`python/mujoco/studio/launch_native.py`) — Opens a local
    desktop window with native Filament graphics (Vulkan or OpenGL).
3.  **`launch_passive.run`**
    (`python/mujoco/studio/launch_passive.py`) — Inspects
    `config.gfx` and delegates to `launch_web`
    (`config.gfx in ('web', 'webgl')`) or `launch_native`. Use `launch_web` or
    `launch_native` directly unless you need runtime switching via `config.gfx`.

### Blocking Runner (`.run`)

Accepts either `model_path=...` (loaded on the sim thread by `ViewerHandle` via
`LoadModelEvent`, setting `handle.model_path` so **Reload** works) or an
in-memory `model=...` (with optional `data=...`, published via
`handle.set_model`). `studio_app.SIM_PLUGINS` bundles
`step_control.StepControl()` and `sim_history.SimHistory()`:

```python
from mujoco.studio import app as studio_app
from mujoco.studio import launch_web  # Or launch_native / launch_passive
from mujoco.studio import viewer_protocol

config = viewer_protocol.ViewerConfig(title='Studio')

launch_web.run(
    config,
    model_path=model_path,  # Or model=model, data=data for an in-memory model.
    plugins=[
        studio_app.App(),
        *studio_app.SIM_PLUGINS,
    ],
)
```

> [!WARNING]
>
> **Stepping and pacing are plugin responsibilities.** `studio_app.SIM_PLUGINS`
> includes `step_control.StepControl()` (real-time-paced CPU stepping) and
> `sim_history.SimHistory()` (history scrubbing). Without `StepControl()` (or
> your own `StepEvent` plugin), `handle.sync()` does **not** advance physics or
> sleep—a tight `while handle.is_running(): handle.sync(...)` loop will spin at
> 100% CPU. When replacing `StepControl()` (e.g., for GPU rollouts, custom RL
> loops, or a pure non-physics mesh/asset viewer via a `NullSimulation` plugin
> that throttles each iteration without calling `mj_step`), pass individual
> plugins and manage pacing explicitly.

### Non-Blocking Launchers (`launch_web.launch`, `launch_native.launch`, `launch_passive.launch`)

Use `.launch(...)` only when attaching a viewer to an external loop that you do
not want to invert into a `StepEvent` plugin (e.g., an existing RL training
loop, trajectory replay script, or test):

```python
with launch_web.launch(
    config,
    plugins=[
        studio_app.App(),
        *studio_app.SIM_PLUGINS,
    ],
) as handle:
  while handle.is_running():
    # Mutate model.opt or data in step-event handlers, NOT here (see §5.2).
    model, data = handle.sync(model, data)
```

To step and pace physics directly in the caller loop instead, omit
`StepControl()` and step after `model, data = handle.sync(model, data)` (see
`python/mujoco/sample/studio/custom_sim_loop.py`).

## 3. Plugin & Messaging Essentials

All cross-thread communication in `messages.py` uses two channel base classes
(custom classes end in `*Snapshot` or `*Event` by convention):

-   **`Snapshot` (Lossy, Latest-Wins)**: Replaces any unread previous value of
    the same type; use for continuous state or complete control configs where
    only the latest value matters (e.g., `StateSnapshot`, `StepControlSnapshot`,
    `MjOptionSnapshot`).
-   **`Event` (Lossless FIFO Queue)**: Guaranteed in-order, exactly-once
    delivery; use for discrete actions, reloads, and distinct edits that must
    not clobber one another (e.g., `ModelEvent`, `StateEvent`, `ResetEvent`,
    `SingleStepEvent`, `RequestPauseEvent`).

Plugins subscribe to messages via `@messages.handler`:

```python
from mujoco.studio import messages
from mujoco.studio import plugin

class MyPlugin(plugin.ViewerPlugin):
  @messages.handler
  def on_build_gui(self, event: messages.BuildGuiEvent) -> bool | None:
    # Build Dear ImGui widgets here (self.viewer is cached automatically).
    return False  # Return True ONLY to consume and block lower-priority handlers.
```

-   **Prefer `Pre*`/`Post*` lifecycle events over `messages.Priority`**: Use a
    bare `@messages.handler` on the phase event matching your intent rather than
    tweaking priorities on `ModelEvent` or `StepEvent`:
    -   **After model load/reload**: Subscribe to `PostModelEvent` (so `model`
        and `data` are already installed and `mj_forward` has run).
    -   **Sim step phases**: `PreStepEvent` (write controls, apply forces, or
        call `handle.set_model`), `StepEvent` (physics stepper only),
        `PostStepEvent` (observe/record stepped state).
    -   **Viewer frame phases**: `UpdateEvent` (visual state, `extra_geoms`, GUI
        data prep, `viewer.upload_image`), `BuildGuiEvent` (Dear ImGui / ImPlot
        submission only).
-   **Handler return values**: Return `False` or `None` so downstream handlers
    also receive the message; return `True` only to consume the event and halt
    propagation. **Never return `True` from an `ExitEvent` handler**—teardown
    must reach every registered plugin.

For bidirectional `send_if_changes` synchronization, the complete chronological
Viewer and Simulation lifecycle event catalogs, `viewer.extra_geoms` 3D
overlays, and `messages.Priority` levels, read
[studio/python_plugins.rst](../../studio/python_plugins.rst)
(`.. _StudioSceneOverlays:`, `.. _StudioLifecycle:`, `.. _StudioDiffSync:`,
`.. _StudioPriorities:`).

### Scenes Built in Python (`LoadModelEvent` + `handle.set_model()`)

A sim plugin that handles `LoadModelEvent`, calls `self.handle.set_model(spec)`,
and returns `True` provides the model; handle `HotReloadEvent` (`if self in
event.plugins: self.handle.reload()`) so saving the file rebuilds the scene in
place. Pair it with a `ViewerPlugin` (same file is fine) whose ImGui controls
send it `Event`s or `Snapshot`s, and run with
`viewer.py --plugins=my_scene.py --hot_reload=all`. The contract — when
`LoadModelEvent` is dispatched and how handlers handle it, `event.path`,
Reload, errors, hot reload — is documented on `messages.LoadModelEvent`;
`ViewerHandle.set_model` and `ViewerHandle.reload` cover the API, and
`python/mujoco/sample/studio/python_scene.py` is a complete, runnable scene
paired with UI. Define the plugins in a module, not in `__main__`, to
hot-reload them; don't name your own handlers `_sim_plugin_on_*` /
`_viewer_plugin_on_*` (the base classes' handlers). Plugins must be registered
at launch: when live-coding, stub a `SimPlugin` and a `ViewerPlugin` up front
and grow them.

> [!IMPORTANT]
>
> **Edit the model where it lives.** A scene is either a model file
> (`--model=scene.xml`: edit the XML, and `--hot_reload=all` or `model`, or the
> Reload button, picks the edit up with its `<include>`s and assets) or a
> Python plugin (edit the `.py` file). Don't write a `LoadModelEvent` plugin
> just to load an existing XML with `MjSpec.from_file` and patch it in Python:
> the model would then live in two places and edits to the XML would not
> hot-reload. When asked to change a model and it is unclear whether the user
> wants the file edited or a Python scene (e.g. to drive a parameter from UI at
> runtime), ask before choosing.
>
> **UI controls for an XML scene are a temporary scaffold.** There is no
> framework that writes UI values back into an XML file, and `MjSpec.to_xml()`
> is not an in-place save (it regenerates the whole file, dropping comments,
> formatting and `<include>`s). When the user is working on an XML scene and
> asks for controls over some of its parameters, tell them so and propose this
> workflow:
>
> 1.  **Scaffold**: a `LoadModelEvent` sim plugin that keeps the XML path
>     itself, loads the file into a spec (`mujoco.MjSpec.from_file(path)`),
>     applies the parameters (`spec.body('torso').pos = ...`) and publishes it
>     with `self.handle.set_model(spec, watch=[path])`, plus a `ViewerPlugin`
>     whose UI controls send the values (as in `sample/studio/python_scene.py`).
>     Reload and hot reload arrive with an empty `event.path`, so leave out the
>     `if event.path: return False` guard here. Run with `--hot_reload=all` to
>     grow the plugins while the viewer runs; saving the XML, its `<include>`s
>     or assets then rebuilds the scene like saving the plugin does.
> 2.  **Tune** in the UI; the values live in the plugin for now.
> 3.  **Bake**: once the user is happy, write the final values into the XML by
>     hand (edit the attributes in place, keeping `<include>`s and comments),
>     delete the scaffold plugin, and reload the XML on its own to confirm it
>     matches.
>
> If the parameters should stay live, the scene belongs in Python for good
> (`MjSpec` built in code, no XML); if the file is generated anyway and its
> formatting does not matter, `spec.to_xml()` can replace step 3. Say so
> briefly if asked, but do not build anything more elaborate up front.

### C++ Studio Plugins & `LaunchStudio`

In C++ (`studio/ux/plugin.h`, `studio/launcher.h`), plugins are POD callback
structs (`GuiPlugin`, `ScenePlugin`, `ModelPlugin`, `KeyHandlerPlugin`,
`SpecEditorPlugin`) copied by value into `mujoco::studio::RegisterPlugin(...)`
(store custom state in `void* data`, never by subclassing; see
[studio/cpp.rst](../../studio/cpp.rst), `.. _StudioCppPlugins:`):

```cpp
static float gain = 1.0f;
// Or call RegisterPlugin(p) before LaunchStudio(argc, argv, cfg):
mjPLUGIN_LIB_INIT(telemetry_plugin) {
  mujoco::studio::GuiPlugin p{
      .active = true, .name = "Telemetry", .data = &gain};
  p.update = [](mujoco::studio::GuiPlugin* self) {
    ImGui::SliderFloat("Gain", static_cast<float*>(self->data), 0.0f, 10.0f);
  };
  mujoco::studio::RegisterPlugin(p);
}
```

## 4. Live Hot-Reload Iteration & Automated Log Inspection

### Live Iteration Workflow When `--hot_reload` Is Active

When a user is running Studio with `--hot_reload` enabled (see
[studio/python_plugins.rst](../../studio/python_plugins.rst),
`.. _StudioHotReload:`) and asks for changes, optimize for **minimum
time-to-visual-feedback** rather than upfront batch verification:

-   **Keep Plugins Out of `__main__`**: Python's `__main__` entrypoint script
    cannot be swapped by `--hot_reload`. Always put `ViewerPlugin` and
    `SimPlugin` classes (and procedural `MjSpec` helpers) in separate importable
    `.py` modules. If an existing script defines plugins inline in `__main__`,
    ask the user whether to split them out to unlock hot reload.
-   **Edit Reloadable Files Immediately & Postpone Tests**: Save changes to
    watched plugin `.py` or model files right away so the live viewer updates on
    the next tick. Defer test suites, full rebuilds, and slow static checks
    until interactive iteration wraps up (or a non-reloadable file changes).
-   **Ship in Small, Incremental Steps**: Save multi-part UI or scene changes in
    small, self-contained increments so the user can steer layout or behavior
    live.
-   **Only Plugin Classes Are Swapped on Reload**: Hot reload swaps
    `plugin.__class__ = new_cls` in place without re-running `__init__`, and any
    helper objects a plugin holds (such as a state dataclass) keep their *old*
    class—so calling a newly added method or property on that helper raises
    `AttributeError` on the live instance. Either declare new state on the
    plugin itself as **class-level defaults** (e.g., `gain: float = 1.0`) or
    handle `messages.HotReloadEvent` (`if self in event.plugins:`) and rebuild
    the helper from its old fields.
-   **Keep `Begin`/`End` and `Push*`/`Pop*` Calls Balanced**: Dear ImGui
    asserts at frame end on a missing `End()`, `PopStyleVar()`, or
    `PopStyleColor()` and aborts the whole viewer process, not just the plugin.
-   **Rely on Error Isolation & Stay Strictly On-Task**: Syntax, import, model
    compile, and handler errors retain the last working state without crashing
    the session—if an edit ever breaks isolation and crashes the viewer, track
    down and fix that gap, and always ask the user before taking detours from
    the requested feature.

### Programmatic Host & Browser Console Inspection

-   **Host Logs**: Engine warnings/messages, Python `logging`, `warnings.warn`,
    and uncaught thread exceptions are captured into the Studio Console
    (`ux.console_gui`). Scripts and tests can inspect captured host logs
    programmatically via `ux.get_console_lines()` and `ux.get_console_counts()`.
-   **Browser Logs (`WebViewer` / `MuJoCo Live`)**: Client-side WebGL, WASM, or
    WebSocket errors appear only in the browser developer console. Inspect them
    via the `chrome-devtools` MCP (`list_console_messages`) when available, or
    launch headless Chrome with `--remote-debugging-port=9222` and subscribe to
    CDP `Runtime.consoleAPICalled` and `Runtime.exceptionThrown`.

## 5. High-Frequency Gotchas

1.  **Respect Thread Boundaries & Keep `ViewerHandle` Minimal**: In Python, the
    sim thread must never read or mutate viewer objects (`viewer.camera`,
    `viewer.vis_options`, viewer model/data copies), and the viewer thread must
    never directly mutate sim state outside of handling inbound events and
    sending requests; always communicate via `handle.send_to_viewer(...)` and
    `viewer.send_to_sim(...)`, because direct concurrent access causes data
    races and rendering artifacts. Never block `UpdateEvent` or `BuildGuiEvent`
    with file/network I/O or `MjSpec` compilation (which stalls `WebViewer`
    heartbeats and triggers a disconnected banner). Keep `ViewerHandle` minimal
    (`sync()`, message dispatch, model/data lifecycle)—put domain sim logic in
    `SimPlugin` subclasses (`StepControl`, `SimHistory`) and do not add `ux` or
    `sim` imports to `viewer_handle.py`.
2.  **Preserve `run_sim_loop` Boilerplate & Edit State in Step Handlers**: Keep
    `viewer_handle.run_sim_loop(handle)`
    (`while handle.is_running(): handle.sync()`) free of ad-hoc logic; put
    custom behavior (custom stepping, perturbation forces, sensor logging, RL
    resets) in a `SimPlugin` subscribing to `SimInitEvent`, `PostModelEvent`,
    `PreStepEvent`, `StepEvent`, `PostStepEvent`, or custom messages, or
    dispatch an `Event` to the sim thread. `sync()` only detects `model.opt`
    changes made inside `PreStepEvent` / `StepEvent` / `PostStepEvent` (edits in
    an outer loop are not sent and get overwritten by the viewer's next
    snapshot; structural model edits require
    `handle.set_model(new_model, path=handle.model_path)`), and `sync()` applies
    viewer resets, slider edits, and perturbations before `PreStepEvent`
    (overwriting controls/forces set in an outer loop). In `.launch(...)` loops
    that own `(model, data)`, always rebind
    `model, data = handle.sync(model, data)` so reloads (`LoadModelEvent` or
    `handle.set_model()`) do not leave the loop stepping a stale model.
3.  **Call `.copy()` on MuJoCo NumPy Views**: MuJoCo array bindings return views
    into mutable C buffers; copy them when recording history across timesteps
    (e.g., position trajectories, Euler angles):

    ```python
    # Avoid: view updates in place, so all history entries alias latest state
    self.history.append(app.data.xpos[body_id])

    # Good: stores an independent snapshot
    self.history.append(app.data.xpos[body_id].copy())
    ```

4.  **Remote `WebViewer` Requires `localhost` or HTTPS (`SharedArrayBuffer`)**:
    Opening a remote `WebViewer` over plain `http://<remote-ip>:<port>` disables
    `SharedArrayBuffer` and blocks WASM pthreads initialization. Tunnel via
    `ssh -L 8080:localhost:8080` and open `http://localhost:8080`, and never
    expose `WebViewer` to untrusted networks (see
    [studio/web_viewers.rst](../../studio/web_viewers.rst),
    `.. _StudioBrowserSandboxing:`).
5.  **Build and Run with `-c opt`**: Always compile and run Studio targets with
    `-c opt` so physics stepping, Filament rendering, and WebViewer WebSocket
    streaming execute with compiler optimizations enabled.
