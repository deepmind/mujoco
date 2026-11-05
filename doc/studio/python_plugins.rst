.. _StudioPythonPlugins:
.. _StudioComposition:

==============
Python Plugins
==============

Python plugins (`plugin.py <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/plugin.py>`__)
extend Studio by decorating methods with ``@messages.handler`` on one of two base classes:

- ``plugin.ViewerPlugin``: Runs on the viewer thread (``NativeViewer`` or ``WebViewer``) for custom Dear ImGui panels,
  input handling, and 3D overlays. Provides ``self.viewer`` for reading the viewer's local ``model`` and ``data`` and
  sending messages to the simulation via ``self.viewer.send_to_sim(...)``.
- ``plugin.SimPlugin``: Runs on the simulation thread inside ``ViewerHandle`` for physics stepping, controllers, and
  model or state mutations. Provides ``self.handle`` for mutating the authoritative ``model`` and ``data``, publishing
  new models via ``self.handle.set_model(...)``, and sending messages to the viewer via
  ``self.handle.send_to_viewer(...)``.

Pass plugin instances via ``plugins=[...]`` in Python with ``launch_native`` or ``launch_web``, or load ``.py`` files
directly from the command line with :ref:`live hot reload <StudioHotReload>`:

.. code-block:: shell

   python -m mujoco.studio.viewer --model=humanoid.xml --plugins=my_plugin.py --hot_reload=all

.. code-block:: python

   from mujoco.studio.dear_imgui import dear_imgui as imgui
   from mujoco.studio import messages, plugin


   class StatusWindow(plugin.ViewerPlugin):

     @messages.handler
     def on_build_gui(self, _: messages.BuildGuiEvent) -> None:
       if imgui.Begin('Simulation Time'):
         if self.viewer.data is not None:
           imgui.TextUnformatted(f't = {self.viewer.data.time:.3f} s')
       imgui.End()

Applications can mix and match built-in and custom plugins to suit different workflows:

- **Full Studio UI with Custom Overlays**: Load the viewer application's default plugins—``studio_app.App()``
  and ``*studio_app.SIM_PLUGINS`` (`viewer.py
  <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/viewer.py>`__)—alongside custom viewer or
  simulation plugins to add dockable charts, control panels, or 3D overlays while retaining all standard
  ``studio_app.App`` features.
- **Minimal Viewport**: Omit ``studio_app.App`` and register only a custom ``plugin.ViewerPlugin`` to get an
  uncluttered 3D viewport with a custom control panel.
- **Custom Stepping Backends**: Replace ``step_control.StepControl()`` with a custom ``plugin.SimPlugin`` to drive the
  viewer from GPU simulations such as :ref:`MuJoCo XLA <Mjx>` or :ref:`MuJoCo Warp <MJW>`, reinforcement learning
  environments, or recorded trajectories while keeping ``studio_app.App()`` and ``sim_history.SimHistory()`` for
  inspection and timeline scrubbing.

See `python/mujoco/sample/studio/
<https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/sample/studio>`__ for sample applications
illustrating these patterns.

.. note::
   Python plugins use a message-handler interface distinct from the callback-struct API currently used by :ref:`C++
   plugins <StudioCppPlugins>`; future work will unify the two plugin systems.

.. _StudioViewerPlugins:

Viewer Plugins
==============

Viewer plugins own the user interface, keyboard and mouse interaction, camera control, and 3D scene overlays.

studio_app.App
--------------

The Studio plugin framework is independent of any particular user interface and gives ``studio_app.App``
(`app/plugins.py <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/app/plugins.py>`__) no special
privileges—it is an ordinary ``ViewerPlugin`` that you can omit or replace entirely with your own UI.
``studio_app.App`` ports MuJoCo's classic :ref:`simulate <saSimulate>` application to Dear ImGui and is
typically paired with ``*studio_app.SIM_PLUGINS`` on the simulation side. When included in ``plugins``, it adds the menu
bar, model hierarchy
and property inspector, visualization options panel, simulation profiler, timeline scrubber, status bar, and an
integrated Studio Console that captures MuJoCo log messages, Python ``logging`` records, ``warnings.warn`` calls, and
uncaught thread exceptions. Click the status bar counts to toggle the Console, or right-click inside it to clear, copy,
or configure engine logging.

.. _StudioSceneOverlays:

3D Scene Overlays
-----------------

Viewer plugins can inject custom 3D decorative geometry—such as command arrows, force vectors, trajectory markers, or
ghost bodies—into the rendered scene without modifying :ref:`mjModel`. During ``UpdateEvent``, clear and populate
``self.viewer.extra_geoms`` with ``mujoco.MjvGeom`` instances initialized via :ref:`mjv_initGeom` and
:ref:`mjv_connector`. For example, when training or debugging a locomotion policy, a viewer plugin can draw the desired
velocity vector above the robot's torso:

.. code-block:: python

   import mujoco
   from mujoco.studio import messages
   from mujoco.studio import plugin
   import numpy as np


   class VelocityCommandOverlay(plugin.ViewerPlugin):
     """Draws a 3D arrow above the torso showing the desired XY velocity."""

     def __init__(self, cmd_vel: tuple[float, float] = (1.0, 0.0)) -> None:
       self.cmd_vel = np.array([cmd_vel[0], cmd_vel[1], 0.0])

     @messages.handler
     def on_update(self, _: messages.UpdateEvent) -> None:
       self.viewer.extra_geoms.clear()
       if self.viewer.data is None or self.viewer.model.nbody < 2:
         return

       origin = self.viewer.data.xpos[1] + np.array([0.0, 0.0, 0.3])
       geom = mujoco.MjvGeom()
       mujoco.mjv_initGeom(
           geom,
           int(mujoco.mjtGeom.mjGEOM_ARROW),
           np.zeros(3),
           np.zeros(3),
           np.eye(3).flatten(),
           np.array([0.2, 0.8, 0.3, 0.9], dtype=np.float32),
       )
       mujoco.mjv_connector(
           geom,
           int(mujoco.mjtGeom.mjGEOM_ARROW),
           0.02,
           origin,
           origin + 0.5 * self.cmd_vel,
       )
       self.viewer.extra_geoms.append(geom)

Both ``NativeViewer`` and ``WebViewer`` automatically forward ``viewer.extra_geoms`` to the underlying renderer each
frame. See `ghost.py <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sample/studio/ghost.py>`__ for a
complete time-delayed ghost-body sample.

.. note::
   ``viewer.extra_geoms`` provides transient per-frame decorative overlays via :ref:`mjvGeom`; persistent scene objects
   are managed directly through :ref:`mujoco.mjrf <FilamentRendering>`.

.. _StudioUiPlugins:

Custom UI Panels
----------------

Custom viewer plugins build user interfaces by handling ``BuildGuiEvent`` and calling MuJoCo's Python bindings for `Dear
ImGui <https://github.com/ocornut/imgui>`__.

.. _StudioViewerBestPractices:
.. _StudioImGuiTips:

Best Practices
--------------

When writing :ref:`viewer plugins <StudioViewerPlugins>`:

- **Read-only viewer state**: Treat ``self.viewer.model`` and ``self.viewer.data`` as read-only, and reset
  model-dependent viewer buffers on ``PostModelEvent``.
- **Never load or compile models on the viewer thread**: Authoritative model loading, :ref:`mjSpec` compilation, and
  state mutations belong on the simulation thread. Dispatch ``LoadModelEvent`` or a custom ``Event`` via
  ``self.viewer.send_to_sim(...)`` so a ``plugin.SimPlugin`` loads or compiles the model and publishes it via
  ``self.handle.set_model(...)``.
- **Keep the viewer thread running at 60 Hz**: Offload file I/O, network requests, or heavy computation to background
  worker threads or the simulation thread. Slowing down ``UpdateEvent`` or ``BuildGuiEvent`` drops the frame rate in
  ``NativeViewer`` and stalls the ``/ui`` and ``/state`` WebSocket heartbeats in :ref:`WebViewer <StudioWebViewer>`,
  causing the browser client to display a disconnected banner.
- **Treat the WebViewer host-to-browser protocol as fixed**: User code should be written strictly as
  ``plugin.ViewerPlugin`` and ``plugin.SimPlugin`` classes. You should not need to modify the wire protocol between the
  Python viewer thread and the WebAssembly browser client when using ``WebViewer``—if a capability is missing there, it
  should be designed as a general-purpose feature for other applications too, so consider opening a pull request
  upstream.

MuJoCo provides its own `Python bindings for Dear ImGui
<https://github.com/google-deepmind/mujoco/tree/main/python/mujoco/studio/dear_imgui>`__ that mirror the C++ API
and share the viewer's active Dear ImGui context. To see what Dear ImGui is capable of and browse available widgets
alongside their code, try the interactive `online demo <https://pthom.github.io/imgui_explorer/?lib=imgui>`__ or call
``imgui.ShowDemoWindow()`` from a ``BuildGuiEvent`` handler. When building user interfaces, consult the `Getting
Started guide <https://github.com/ocornut/imgui/wiki/Getting-Started>`__, the upstream `docs/ folder
<https://github.com/ocornut/imgui/tree/master/docs>`__ and `FAQ
<https://github.com/ocornut/imgui/blob/master/docs/FAQ.md>`__, and the ``CONTROLS GUIDE`` and ``PROGRAMMER GUIDE`` in
the header comments at the top of `imgui.cpp <https://github.com/ocornut/imgui/blob/master/imgui.cpp>`__:

- **Window granularity & docking**: Implement each viewer plugin as one or two ``imgui.Begin(...)`` windows so panels
  dock cleanly into ``studio_app.App``'s dockspaces alongside built-in tools.
- **Separate data preparation from layout** (``UpdateEvent`` **vs.** ``BuildGuiEvent``): Dear ImGui builds draw command
  lists rather than rasterizing pixels directly. Prepare plot buffers or upload GUI textures via
  ``self.viewer.upload_image(...)`` during ``UpdateEvent``, and keep ``BuildGuiEvent`` strictly focused on widget
  submission, naming layout methods ``build_gui`` or ``build_*_gui`` rather than ``render`` or ``draw``. Never run
  simulation steps, file I/O, or blocking calls inside GUI methods.
- **Runtime IDs** (``##`` **vs.** ``###``): Dear ImGui hashes each window or widget's label string against the current
  ID stack seed; any text after ``##`` or ``###`` is hidden on screen:

  - **Disambiguating duplicate labels** (``##``): ``Label##Suffix`` hashes the entire string while displaying only
    ``Label``. Append ``##{i}`` or wrap loop iterations in ``imgui.PushID(i)`` / ``imgui.PopID()`` when repeating
    buttons or ``imgui.Selectable`` rows share the same visible text.
  - **Persistent IDs across dynamic labels** (``###``): ``Dynamic Text###ConstantId`` resets the hash seed and hashes
    *only* ``ConstantId``. Append ``###ConstantId`` when a window title changes across frames—for example,
    ``f'Bodies ({n})###Bodies'`` preserves its docked position, focus, and collapse state—or to give icon-only buttons a
    readable ID such as ``f'{ICON_FA_PLAY}###Play'`` for UI test automation.
- **Grid alignment with the Tables API**: Prefer ``imgui.BeginTable`` with ``TableColumnFlags.WidthFixed`` and
  ``WidthStretch`` for key-value inspectors and multi-column layouts instead of manual ``SameLine()`` pixel offsets or
  legacy ``Columns()``.
- **Virtualize large lists with** ``imgui.ListClipper``: Wrap long lists, log views, or tables in
  ``clipper = imgui.ListClipper(); clipper.Begin(len(items))`` so Dear ImGui only submits visible rows; unclipped tables
  generate vertices for every offscreen item and can saturate :ref:`WebViewer <StudioWebViewer>` WebSocket bandwidth.
  Use ``imgui.TextUnformatted`` for read-only strings to avoid formatting and copy overhead.
- **Theme-aware colors & keyboard capture**: Query semantic theme colors via
  ``imgui.GetStyleColorVec4(imgui.Col.Text)`` rather than hardcoding RGB values in ``PushStyleColor`` so custom widgets
  remain legible across dark and light themes. When handling custom keyboard shortcuts, check
  ``not imgui.GetIO().WantCaptureKeyboard`` so typing into an active ``InputText`` or filter box does not trigger
  simulation actions.
- **Dear ImGui skill guide**: Refer to the `mujoco-gui skill file
  <https://github.com/google-deepmind/mujoco/blob/main/doc/skills/gui/SKILL.md>`__ for detailed C++ and Python Dear
  ImGui conventions and ``# Good:`` vs. ``# Avoid:`` examples.

.. _StudioSimPlugins:
.. _StudioSplitPlugins:

Simulation Plugins
==================

Simulation plugins own physics stepping, real-time pacing, trajectory history buffers, and authoritative mutations to
:ref:`mjModel`, :ref:`mjData`, and :ref:`mjSpec`. When a feature needs both interactive UI controls and changes to the
simulation or model, pair a ``plugin.ViewerPlugin`` with a ``plugin.SimPlugin`` in the same ``.py`` file: the viewer
plugin builds the widgets and sends custom ``*Snapshot`` or ``*Event`` messages via ``self.viewer.send_to_sim(...)``,
and the simulation plugin handles those messages to mutate state or recompile and publish a new model via
``self.handle.set_model(...)``.

.. TODO(matijak): Add a split-plugin chain editing sample under sample/studio/ and link to it here once Python modeling
   via SimPlugin is supported.

Built-in Plugins
----------------

``studio_app.SIM_PLUGINS`` bundles Studio's two standard simulation plugins so they can be unpacked alongside
``studio_app.App()`` via ``plugins=[studio_app.App(), *studio_app.SIM_PLUGINS]``, or passed
individually when replacing one:

- ``step_control.StepControl`` (`step_control.py
  <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/step_control.py>`__): Steps CPU physics
  (:ref:`mj_step`) with real-time wall-clock pacing during ``StepEvent``, applies control noise configured from the UI
  (``StepControlSnapshot``), and handles pause (``RequestPauseEvent``) and single-step (``SingleStepEvent``) requests.
  It reports its settings to the viewer whenever they change, and the speed it achieves whenever it has moved by more
  than a deadband amount.
- ``sim_history.SimHistory`` (`sim_history.py
  <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/sim_history.py>`__): Records a frame into an
  authoritative ring buffer after every physics step (``PostStepEvent``), broadcasts buffer metadata to the viewer via
  ``SimHistorySnapshot``, and loads historical frames when the user scrubs the timeline in ``studio_app.App``.
  ``studio_app.App`` and ``sim_history.SimHistory`` are decoupled: ``SimHistorySnapshot`` is the sole contract
  between them.

Stepping & Control
------------------

To inject custom control laws before each physics step, handle ``PreStepEvent``. To replace CPU physics stepping—for
example, to step a :ref:`MuJoCo XLA <Mjx>` or :ref:`MuJoCo Warp <MJW>` batch simulation, or an RL environment—omit
``step_control.StepControl()`` or consume ``StepEvent`` at a higher priority, and advance and pace ``event.data`` inside
your own ``StepEvent`` handler:

.. code-block:: python

   import time
   import mujoco
   from mujoco.studio import messages
   from mujoco.studio import plugin


   class CustomStepperPlugin(plugin.SimPlugin):
     """Applies PD joint torques and replaces default StepControl stepping."""

     kp: float = 50.0
     kd: float = 5.0

     @messages.handler
     def on_pre_step(self, event: messages.PreStepEvent) -> None:
       qpos_err = -event.data.qpos[: event.model.nu]
       qvel_err = -event.data.qvel[: event.model.nu]
       event.data.ctrl[:] = self.kp * qpos_err + self.kd * qvel_err

     @messages.handler(priority=messages.Priority.USER)
     def on_step(self, event: messages.StepEvent) -> bool:
       # Runs before StepControl (Priority.INTERNAL); returning True consumes StepEvent.
       mujoco.mj_step(event.model, event.data)
       time.sleep(event.model.opt.timestep)
       return True

.. _StudioBestPractices:

Best Practices
--------------

When writing :ref:`simulation plugins <StudioSimPlugins>`:

- **Own authoritative state & model reloads**: Mutate ``self.handle.model`` and ``self.handle.data`` only on the
  simulation thread, republish structural or non-``opt`` model changes via
  ``self.handle.set_model(new_model, path=self.handle.model_path)``, and reset model-dependent buffers on
  ``PostModelEvent``.
- **Edit** ``model.opt`` **and** ``data`` **inside step-event handlers**: ``ViewerHandle.sync()`` tracks ``model.opt``
  diffs across ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent`` via ``send_if_changes``, and applies incoming
  viewer resets, ``StateEvent`` overrides, and ``PerturbSnapshot`` forces *before* ``PreStepEvent``. Edits outside
  those handlers are not synced and may be overwritten in the same call.
- **Simulation loop pacing & pure viewers**: ``handle.sync()`` never sleeps; ``step_control.StepControl`` paces against
  wall-clock time and yields ``1 ms`` while paused so the loop does not spin at 100% CPU. If you omit
  ``step_control.StepControl`` for a custom physics stepper or a pure model viewer that does not call :ref:`mj_step`,
  your ``StepEvent`` plugin or outer loop must pace each iteration.
- **Copy NumPy views when recording history**: Slices of :ref:`mjData` arrays such as ``data.xpos[body_id]``,
  ``data.geom_xpos``, or ``data.qpos`` are zero-copy NumPy views backed by MuJoCo's C buffers and are overwritten in
  place on the next step or :ref:`mj_forward` call. Always call ``.copy()`` before storing slices across frames.
- **Handler return values**: Returning ``True`` from an ``@messages.handler`` method consumes the message and skips
  lower-priority handlers. Return ``None`` or ``False`` unless intentionally overriding a message, and never return
  ``True`` from ``ExitEvent``.

.. _StudioLifecycle:

Lifecycle Events
================

Viewer Thread
-------------

The viewer loop in `viewer_protocol.py
<https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/viewer_protocol.py>`__ dispatches lifecycle
events to ``ViewerPlugin`` instances in chronological order:

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Event Type
     - Description
   * - ``ViewerInitEvent(viewer)``
     - Dispatched once on startup. ``ViewerPlugin`` caches ``self.viewer`` automatically above ``Priority.CRITICAL``
       before subclass handlers run.
   * - ``ModelEvent(model, path)``
     - Received when the simulation publishes a model; ``Viewer`` deep-copies ``viewer.model``, creates ``viewer.data``,
       records ``viewer.model_path``, runs :ref:`mj_forward`, clears ``viewer.extra_geoms``, and dispatches
       ``PostModelEvent``.
   * - ``PostModelEvent(model, data, path)``
     - Dispatched locally once ``viewer.model`` and ``viewer.data`` are initialized. Subscribe here rather than
       ``ModelEvent`` to reset viewer-plugin state; reloading the same ``path`` preserves camera, pause, and playback
       speed in ``studio_app.App``.
   * - ``PreloadModelEvent(model)``
     - A hint from a simulation plugin that ``model`` may become the current model soon, for example the next scene
       of a playlist. Viewers may ignore it; the :ref:`web viewer <StudioWebViewers>` serializes the model and has
       connected browsers download it in the background, so the later ``ModelEvent`` switches without a download.
   * - ``HotReloadEvent(plugins)``
     - Dispatched between frames when :ref:`hot reload <StudioHotReload>` swaps modified plugin classes.
   * - ``UpdateEvent()``
     - Dispatched every active frame after incoming simulation events and snapshots are applied, to update visuals,
       populate ``viewer.extra_geoms``, prepare GUI data, or upload textures via ``viewer.upload_image``.
   * - ``BuildGuiEvent()``
     - Dispatched every active frame immediately after ``UpdateEvent`` to submit Dear ImGui widgets.
   * - ``ExitEvent(reason)``
     - Dispatched once when the viewer shuts down. Release resources here; never return ``True`` so teardown reaches
       every plugin.

When ``studio_app.App`` is included in ``plugins``, it also dispatches
``studio_app.InitEvent(app)`` once during startup so companion viewer plugins can cache
``event.app``.

Simulation Thread
-----------------

``ViewerHandle`` (`viewer_handle.py
<https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/viewer_handle.py>`__) dispatches lifecycle
events to ``SimPlugin`` instances in chronological order:

.. list-table::
   :widths: 34 66
   :header-rows: 1

   * - Event Type
     - Description
   * - ``SimInitEvent(handle)``
     - Dispatched once by ``ViewerHandle.__init__``. ``SimPlugin`` caches ``self.handle`` automatically above
       ``Priority.CRITICAL`` before subclass handlers run.
   * - ``LoadModelEvent(path)``
     - Sent on startup, UI Reload, or file drag-and-drop. ``ViewerHandle`` loads the file and publishes it via
       ``handle.set_model(model, path=path)``, keeping the previous model if loading fails.
   * - ``ModelEvent(model, path)``
     - Dispatched when ``handle.set_model(...)`` publishes a model or ``handle.sync(model, data)`` sees a new model
       object, followed immediately by ``PostModelEvent``.
   * - ``PostModelEvent(model, data, path)``
     - Dispatched locally once ``handle.model`` and ``handle.data`` are installed and forwarded. Subscribe here rather
       than ``ModelEvent`` to reset simulation-plugin state; reloading the same ``path`` preserves ``StepControl``
       pause, speed, and noise settings.
   * - ``HotReloadEvent(plugins)``
     - Dispatched inside ``handle.sync()`` when :ref:`hot reload <StudioHotReload>` swaps modified simulation plugin
       classes.
   * - ``PreStepEvent(model, data)``
     - Dispatched on each ``handle.sync()`` after incoming viewer resets, state overrides, and perturbations are applied
       and before physics advances. Write controls, external forces, or ``model.opt`` edits, or call
       ``handle.set_model``, here.
   * - ``StepEvent(model, data)``
     - Dispatched immediately after ``PreStepEvent`` to advance physics; ``StepControl`` steps CPU physics with
       real-time pacing here.
   * - ``PostStepEvent(model, data)``
     - Dispatched immediately after ``StepEvent`` and before ``StateSnapshot`` is sent to the viewer; ``SimHistory``
       records the stepped state here.
   * - ``ExitEvent(reason)``
     - Dispatched once by ``ViewerHandle.close()`` on every exit path. Release resources here; never return ``True``.

.. _StudioDiffSync:

Diff-Based Sync
---------------

Bidirectional :ref:`snapshots <StudioMessaging>` such as ``MjOptionSnapshot`` and ``StepControlSnapshot`` must only be
sent by the side that changed the value so neither side echoes incoming snapshots back over the other side's edits. Both
``Viewer`` and ``ViewerHandle`` provide a ``send_if_changes(*make_messages)`` context manager that calls each
zero-argument lambda before and after the ``with`` block and sends only the messages whose value changed:

.. code-block:: python

   with self.viewer.send_if_changes(
       lambda: messages.StepControlSnapshot.from_step_control(self.step_control_state),
       lambda: messages.MjOptionSnapshot(opt=copy.deepcopy(self.model.opt)),
   ):
     ux.build_options_gui(self.model, self.viewer.vis_options)

Incoming snapshots are applied outside ``send_if_changes`` blocks so they never echo back. On the simulation side,
``ViewerHandle.sync()`` wraps ``PreStepEvent``, ``StepEvent``, and ``PostStepEvent`` in ``send_if_changes`` for
``model.opt``—so mutate ``model.opt`` inside step-event handlers rather than in an outer loop between ``sync()`` calls,
and change another plugin's state by dispatching an event it handles, such as ``RequestPauseEvent``, rather than
mutating it directly.

.. _StudioPriorities:

Handler Priorities
==================

.. note::
   Handler priorities are an advanced feature. Most plugins only need plain ``@messages.handler`` at the default
   ``Priority.USER`` level together with explicit lifecycle hooks such as ``PostModelEvent``, ``PreStepEvent``,
   ``StepEvent``, and ``PostStepEvent``.

Plugin methods are decorated with ``@messages.handler`` or ``@messages.handler(priority=...)``. ``PluginRegistry``
inspects each decorated method's type annotation to determine which ``Event`` or ``Snapshot`` subclass it handles:

.. code-block:: python

   from mujoco.studio import messages
   from mujoco.studio import plugin


   class CustomSimPlugin(plugin.SimPlugin):

     @messages.handler(priority=messages.Priority.CRITICAL)
     def on_pre_step(self, event: messages.PreStepEvent) -> None:
       # Run before standard Priority.USER PreStepEvent handlers.
       event.data.ctrl[:] = 0.0

Handlers for the same message type execute from **highest** numerical ``Priority`` value to lowest:

.. list-table::
   :widths: 28 14 58
   :header-rows: 1

   * - Priority Constant
     - Value
     - Intended Use
   * - ``Priority.CRITICAL``
     - ``1000``
     - Handlers that must run before everything else, such as core viewer state updates and ``studio_app.App`` layout.
   * - ``Priority.USER``
     - ``100``
     - Default priority for user plugins when ``@messages.handler`` is used without arguments.
   * - ``Priority.LIBRARY``
     - ``10``
     - Reusable library extensions that run after user plugins by default.
   * - ``Priority.INTERNAL``
     - ``1``
     - Built-in framework handlers: ``StepControl``, ``SimHistory``, ``Viewer``, and ``ViewerHandle``.

Stopping propagation: If a handler returns ``True``, dispatching of that message stops immediately and lower-priority
handlers are skipped. Because built-in handlers run at ``Priority.INTERNAL`` (``1``), a user plugin at
``Priority.USER`` (``100``) can override built-in behavior for a specific message by returning ``True``, or augment it
by returning ``None`` or ``False``. Never return ``True`` from an ``ExitEvent`` handler.

Events vs. Priority
-------------------

Studio provides both explicit ``Pre*`` / ``Post*`` lifecycle events and handler priorities:

- **Prefer lifecycle phase events** such as ``PostModelEvent``, ``PreStepEvent``, ``StepEvent``, ``PostStepEvent``,
  ``UpdateEvent``, and ``BuildGuiEvent`` with a bare ``@messages.handler`` decorator. Ordering distinct lifecycle phases
  via ``Priority`` on ``ModelEvent`` or ``StepEvent`` can expose uninitialized ``MjData`` before :ref:`mj_forward` runs
  or leave an in-flight ``StepEvent`` holding stale ``model``/``data`` references after ``handle.set_model()``.
- **Reserve** ``Priority`` strictly for cases where multiple handlers target the *exact same* event phase and one
  handler must override or consume the message by returning ``True`` before a fallback handler runs.

.. _StudioHotReload:

Hot Reload
==========

.. tip::
   Developing Dear ImGui panels and 3D overlays with ``--hot_reload=all`` updates the live viewer on every file save
   while keeping the running simulation, camera pose, and docked window layout intact. Compiled C++ plugins and
   ``pybind11`` extensions cannot be hot-reloaded without rebuilding and restarting the process, whereas MuJoCo's Python
   ``dear_imgui`` bindings mirror the C++ ``ImGui::`` API directly. Prototyping a UI in Python with hot reload and
   porting it to C++ once the layout settles is therefore the fastest way to build C++ Studio components too.

Passing ``--hot_reload=all`` to ``viewer.py`` or setting ``ViewerConfig(hot_reload='all')`` enables live reloading
(`hot_reload.py <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/hot_reload.py>`__) without
restarting the process:

- **Scope**: Accepts ``'all'`` for plugin modules and the active model file, ``'plugins'``, ``'model'``, or
  comma-separated plugin class names such as ``'VelocityCommandOverlay'`` or ``'model,App'``.
- **Live class swapping**: Saving a registered plugin's ``.py`` file or a helper module imported from the same
  directory reloads the module, swaps ``plugin.__class__`` in place—preserving existing instance attributes, camera
  pose, UI docking layout, and ``WebViewer`` browser connections—and dispatches ``HotReloadEvent(plugins=...)`` so
  swapped plugins can initialize new buffers or invalidate cached MuJoCo IDs. Saving the watched XML/URDF model file or
  its ``<include>`` assets automatically triggers a model reload.
- **Fixed boundary & error recovery**: Framework modules, ``__main__``, the C++ binary, and the set of registered plugin
  instances are fixed across reloads. Syntax, import, or model compilation errors leave the previous code and model
  running and retry on the next save; exceptions raised by a swapped plugin's handlers are logged once to the Studio
  Console rather than terminating the loop.
