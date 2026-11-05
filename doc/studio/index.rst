.. _Studio:

=============
MuJoCo Studio
=============

**MuJoCo Studio** is a Python and C++ SDK for building interactive simulations, research tools, and custom visualizers
that run in a native desktop window or in a web browser with no changes to user code. It is the next iteration of the
:ref:`simulate <saSimulate>` application and :ref:`mujoco.viewer <PyViewer>`, replacing the classic OpenGL renderer and
fixed :ref:`mjUI <Ui>` panels with :ref:`Filament <FilamentRendering>` physically based rendering, `Dear ImGui
<https://github.com/ocornut/imgui>`__ immediate-mode UI, and a modular plugin system.

.. _StudioQuickStart:

Quick Start
===========

- **In the browser (zero install)**: Open `live.mujoco.org <https://live.mujoco.org>`__ and drag-and-drop any ``.xml``,
  ``.mjb``, or ``.mjz`` model onto the canvas, or share a model URL via ``?model=github:<org>/<repo>/<branch>/<path>``
  (see :ref:`MuJoCo Live <StudioLive>`).
- **Standalone desktop app**: Download the `latest release <https://github.com/google-deepmind/mujoco/releases>`__ and
  open ``MuJoCoStudio.app`` on macOS, ``bin\mujoco_studio.exe`` on Windows, or ``bin/mujoco_studio`` on Linux (see
  :ref:`Standalone Application <StudioApp>`).
- **From the Python CLI**: Launch the Studio viewer in a :ref:`native window <StudioNativeViewers>`
  or stream it headlessly to a :ref:`browser <StudioWebViewer>`:

  .. code-block:: shell

     python -m mujoco.studio.viewer --model=humanoid.xml            # native desktop window
     python -m mujoco.studio.viewer --model=humanoid.xml --gfx=web  # headless web viewer

- **Inside a Python simulation loop**: Attach a non-blocking viewer (``launch_native.launch`` or ``launch_web.launch``)
  without locking ``model`` or ``data``:

  .. code-block:: python

     import mujoco
     from mujoco.studio import app as studio_app, launch_native  # or launch_web

     model = mujoco.MjModel.from_xml_path('humanoid.xml')
     data = mujoco.MjData(model)

     with launch_native.launch(plugins=[studio_app.App()]) as handle:
       while handle.is_running():
         model, data = handle.sync(model, data)
         mujoco.mj_step(model, data)

.. _StudioOverview:

Overview
========

MuJoCo also ships self-contained `agent SKILL files <https://github.com/google-deepmind/mujoco/tree/main/doc/skills>`__
under ``doc/skills/``—including `mujoco-studio
<https://github.com/google-deepmind/mujoco/blob/main/doc/skills/studio/SKILL.md>`__ and `mujoco-gui
<https://github.com/google-deepmind/mujoco/blob/main/doc/skills/gui/SKILL.md>`__—with idiomatic API patterns and common
pitfalls for both developers and AI coding assistants.

.. toctree::
   :maxdepth: 2

   web_viewers
   native_viewers
   architecture
   python_plugins
   cpp

Core Components
---------------

MuJoCo Studio is organized around five building blocks shared across C++ and Python:

- **Studio Viewer Application**: The full-featured interactive visualizer, implemented as ``mujoco::studio::App``
  (`app.h <https://github.com/google-deepmind/mujoco/blob/main/studio/app.h>`__) in :ref:`C++ <StudioCpp>` and
  ``studio_app.App`` (`app/plugins.py
  <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/studio/app/plugins.py>`__) in :ref:`Python
  <StudioViewerPlugins>`. It provides model loading, simulation history scrubbing, step controls, performance profiling,
  an integrated log and warning console, and scene hierarchy inspection.
- **Composable Plugins**: Modular components that extend the viewer UI, 3D scene, and physics stepping.
  In Python, the standard Studio UI (:ref:`studio_app.App <StudioViewerPlugins>`), real-time physics stepping
  (:ref:`StepControl <StudioSimPlugins>`), and trajectory history recording (:ref:`SimHistory <StudioSimPlugins>`) are
  themselves implemented as :ref:`ViewerPlugin and SimPlugin <StudioPythonPlugins>` subclasses communicating over
  non-blocking :ref:`message channels <StudioMessaging>`.
- **Immediate-mode UI**: Powered by `Dear ImGui <https://github.com/ocornut/imgui>`__, the `immediate-mode GUI (IMGUI)
  paradigm <https://github.com/ocornut/imgui/wiki#about-the-imgui-paradigm>`__ minimizes superfluous state duplication,
  state synchronization, and UI-side state retention—making it less error-prone than traditional retained-mode
  interfaces and well suited for fast iteration on dynamic visualization and debugging tools that directly reflect live
  simulation data. Rather than touching GPU state directly, widget calls output compact vertex buffers and draw command
  lists that can be rendered locally or streamed remotely (see :ref:`WebViewer <StudioWebViewer>`).
- **Running in the Browser**: Studio offers two web viewers: :ref:`MuJoCo Live <StudioLive>` runs entirely in the
  browser and can be hosted on a static page, whereas :ref:`WebViewer <StudioWebViewer>` streams a host-driven Python
  simulation and UI to the browser with no GPU or display server required on the host.
- **Physically Based Rendering**: Hardware-accelerated physically based rendering (PBR) via :ref:`Filament
  <FilamentRendering>` across OpenGL, Vulkan, and WebGL, plus headless and software rasterizers for automated testing.
  For background on Filament and physically based rendering theory, see the `Filament documentation
  <https://google.github.io/filament/>`__, the online `Physically Based Rendering in Filament
  <https://google.github.io/filament/Filament.html>`__ guide, and `Physically Based Rendering: From Theory to
  Implementation <https://www.pbr-book.org/>`__.

.. _StudioComparison:

Legacy Visualizers
------------------

.. list-table::
   :widths: 22 36 42
   :header-rows: 1

   * - Feature
     - Legacy :ref:`mujoco.viewer <PyViewer>`
     - MuJoCo Studio
   * - **GUI Framework**
     - Fixed custom C :ref:`mjUI <Ui>` panels (:ref:`simulate <saSimulate>`)
     - Dockable `Dear ImGui <https://github.com/ocornut/imgui>`__ windows (``studio_app.App`` is the port of :ref:`simulate
       <saSimulate>`)
   * - **3D Renderer**
     - Fixed-function :ref:`classic OpenGL <ClassicRendering>` (``mjr_``)
     - :ref:`Filament <FilamentRendering>` (OpenGL, Vulkan, WebGL, headless/software)
   * - **Web Support**
     - None (requires local display or X11/VNC)
     - :ref:`MuJoCo Live <StudioLive>` (standalone WASM) & Python :ref:`WebViewer <StudioWebViewer>` (NetImgui + WASM)
   * - **Architecture**
     - Monolithic loop with coupled UI and physics
     - Composable :ref:`ViewerPlugin / SimPlugin <StudioPythonPlugins>` classes (Python) and :ref:`callback plugins
       <StudioCppPlugins>` (C++)
   * - **Synchronization**
     - Shared :ref:`mjModel` / :ref:`mjData` under ``viewer.lock()``
     - :ref:`Message passing <StudioMessaging>` in Python; internal physics mutex in C++ (``mujoco::studio::App``)
   * - **Live Iteration**
     - Static model reload only
     - Model file watching and live Python plugin :ref:`hot reload <StudioHotReload>`
