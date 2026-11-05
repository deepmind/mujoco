.. _StudioCpp:

===
C++
===

.. _StudioCppLauncher:

Build & Launch
==============

The native C++ Studio application lives in `studio/
<https://github.com/google-deepmind/mujoco/tree/main/studio>`__ and ships prebuilt in MuJoCo binary releases (see
:ref:`Standalone Application <StudioApp>`). To build and launch ``mujoco_studio`` from the
repository root, use the ``build_studio_cpp_viewer`` step in `.github/workflows/build_steps.sh
<https://github.com/google-deepmind/mujoco/blob/main/.github/workflows/build_steps.sh>`__ (see the workflows `README
<https://github.com/google-deepmind/mujoco/blob/main/.github/workflows/README.md>`__ for prerequisites), or run
`studio/build.sh <https://github.com/google-deepmind/mujoco/blob/main/studio/build.sh>`__ (which also supports Git Bash
on Windows, ``debug``/``configure``/``build`` arguments, and ``CMAKE_ARGS``):

.. code-block:: shell

   bash .github/workflows/build_steps.sh build_studio_cpp_viewer
   ./build_host/bin/mujoco_studio model/humanoid/humanoid.xml

To configure an isolated CMake build tree manually without helper scripts:

.. code-block:: shell

   cmake -B build-studio \
     -DCMAKE_BUILD_TYPE=Release \
     -DUSE_STATIC_LIBCXX=OFF \
     -DBUILD_SHARED_LIB=OFF \
     -DMUJOCO_USE_FILAMENT=ON \
     -DMUJOCO_BUILD_EXAMPLES=OFF \
     -DMUJOCO_BUILD_SIMULATE=OFF \
     -DMUJOCO_BUILD_TESTS=OFF \
     -DMUJOCO_TEST_PYTHON_UTIL=OFF \
     -DMUJOCO_WITH_USD=OFF \
     -DMUJOCO_BUILD_STUDIO=ON \
     -DFILAMENT_SKIP_SAMPLES=ON \
     -DCMAKE_CXX_FLAGS=-Wno-error=deprecated-declarations \
     -DFILAMENT_SHORTEN_MSVC_COMPILATION=OFF
   cmake --build build-studio --config Release --target mujoco_studio --parallel
   ./build-studio/bin/mujoco_studio model/humanoid/humanoid.xml

``mujoco_studio`` resolves its staged font and Filament assets relative to the executable directory, and supports the
following ``--gfx`` graphics modes (`graphics_mode.h
<https://github.com/google-deepmind/mujoco/blob/main/studio/hal/graphics_mode.h>`__):

- ``opengl`` / ``opengl_headless`` / ``opengl_software``: :ref:`Filament <FilamentRendering>` OpenGL desktop window,
  offscreen headless context (blitted to a software SDL2 window in the C++ app), or software rasterizer.
- ``vulkan`` / ``vulkan_software``: Filament Vulkan desktop window or software rasterizer.

On Linux, direct window rendering expects X11 (``--gfx=opengl`` is rejected under Wayland). In a Wayland session,
omitting ``--gfx`` falls back to ``opengl_headless``; for direct GPU presentation, use an X11 session.

Custom C++ executables can embed and launch Studio via ``mujoco::studio::LaunchStudio`` declared in `launcher.h
<https://github.com/google-deepmind/mujoco/blob/main/studio/launcher.h>`__ after calling ``absl::ParseCommandLine``.
``LaunchStudio`` accepts a ``LauncherConfig`` and manages resource provider registration, ``mujoco::studio::App``
(`app.h <https://github.com/google-deepmind/mujoco/blob/main/studio/app.h>`__) initialization, the background physics
thread on desktop builds, and the main UI/render loop.

.. _StudioCppPlugins:

C++ Plugins
===========

.. note::
   C++ plugins currently use a POD callback-struct interface distinct from the message-handler API used by
   :ref:`Python plugins <StudioPythonPlugins>`; future work will unify the two plugin systems.

In C++, Studio plugins are plain-old-data (POD) structs with function pointers defined in `ux/plugin.h
<https://github.com/google-deepmind/mujoco/blob/main/studio/ux/plugin.h>`__. Plugins are copied by value into a global
registry via ``mujoco::studio::RegisterPlugin(plugin)``. Do not subclass them; store custom state in the ``void* data``
field:

.. list-table::
   :widths: 24 34 42
   :header-rows: 1

   * - Struct (``mujoco::studio``)
     - Callbacks
     - Purpose
   * - ``GuiPlugin``
     - ``update(GuiPlugin* self)``
     - Registers a toggleable window in the Plugins main menu and builds custom ``ImGui::`` widgets when ``active`` is
       ``true``.
   * - ``ScenePlugin``
     - ``enhance_scene(ScenePlugin* self, const mjModel* model, mjData* data, mjvScene* scene)``
     - Appends custom :ref:`mjvGeom` objects to the :ref:`mjvScene` before rendering.
   * - ``ModelPlugin``
     - ``get_model_to_load``, ``post_model_loaded``, ``do_update``, ``pre_step``, ``post_step``
     - Supplies models from custom sources and hooks into physics updates and :ref:`mj_step` calls.
   * - ``KeyHandlerPlugin``
     - ``on_key_pressed(KeyHandlerPlugin* self)``
     - Invokes a callback when the configured ImGui ``key_chord`` is pressed.
   * - ``SpecEditorPlugin``
     - ``pre_compile``, ``post_compile``
     - Mutates :ref:`mjSpec` before compilation; returning ``true`` from ``pre_compile`` triggers recompilation and
       ``post_compile``.

Plugin Registration
-------------------

C++ plugins can self-register inside a linked library or translation unit using :ref:`mjPLUGIN_LIB_INIT`, which runs
before ``main()`` without modifying the launcher, or register explicitly in ``main()`` before calling
``LaunchStudio``:

.. code-block:: C++

   #include "absl/flags/parse.h"
   #include <imgui.h>
   #include <mujoco/mujoco.h>
   #include "studio/launcher.h"
   #include "studio/ux/plugin.h"

   struct TelemetryState {
     float gain = 1.0f;
   };

   void RegisterTelemetryPlugin() {
     static TelemetryState state;

     mujoco::studio::GuiPlugin gui_plugin;
     gui_plugin.name = "Telemetry";
     gui_plugin.active = true;
     gui_plugin.data = &state;
     gui_plugin.update = [](mujoco::studio::GuiPlugin* self) {
       auto* s = static_cast<TelemetryState*>(self->data);
       ImGui::SliderFloat("Gain", &s->gain, 0.0f, 10.0f);
     };
     mujoco::studio::RegisterPlugin(gui_plugin);
   }

   // Option A: Register automatically when linked or loaded as a plugin library.
   mjPLUGIN_LIB_INIT(telemetry_plugin) {
     RegisterTelemetryPlugin();
   }

   // Option B: Register explicitly in a custom executable before LaunchStudio.
   int main(int argc, char** argv) {
     absl::ParseCommandLine(argc, argv);
     RegisterTelemetryPlugin();

     mujoco::studio::LauncherConfig config;
     config.title = "Custom Studio";
     config.model_file = "path/to/robot.xml";
     return mujoco::studio::LaunchStudio(argc, argv, config);
   }

.. _StudioLiveCpp:

WebAssembly Build
=================

The :ref:`MuJoCo Live <StudioLive>` web application (`live.mujoco.org <https://live.mujoco.org>`__) compiles
``mujoco::studio::App`` to WebAssembly via Emscripten (`emscripten.cc
<https://github.com/google-deepmind/mujoco/blob/main/studio/emscripten.cc>`__). JavaScript drives the application loop
through Embind bindings—``init``, ``loadFile``, ``loadUrl``, ``loadKeyframe``, ``renderFrame``, and ``deinit``—and
``emscripten.cc`` registers custom :ref:`mjpResourceProvider` handlers for ``http:``, ``https:``, and ``github:`` URLs
backed by an asynchronous ``fetch`` cache. For example, clicking any robot thumbnail on the `MuJoCo Menagerie
<https://github.com/google-deepmind/mujoco_menagerie>`__ page opens that robot directly in ``live.mujoco.org`` via its
``github:`` model URL. See :ref:`MuJoCo Live <StudioLive>` in :ref:`Web Viewers <StudioWebViewers>` for browser usage,
drag-and-drop archives, and ``?model=`` URL parameters.
