===========
Programming
===========

.. _inIntro:

Introduction
~~~~~~~~~~~~

This chapter is the MuJoCo programming guide. A separate chapter contains the :doc:`../APIreference/index`
documentation. MuJoCo is a dynamic library compatible with Windows, Linux and macOS, which requires a processor with AVX
instructions. The library exposes the full functionality of the simulator through a compiler-independent shared-memory C
API. It can also be used in C++ programs.

The MuJoCo codebase is organized into subdirectories corresponding to different major areas of functionality:

Engine
   The simulator (or physics engine) is written in C. It is responsible for all runtime computations.
Parser
   The XML parser is written in C++. It can parse MJCF models and URDF models, converting them into an internal mjCModel
   C++ object which is exposed to the user via mjSpec.
Compiler
   The compiler is written in C++. It takes an mjCModel C++ object constructed by the parser, and converts it into an
   mjModel C structure used at runtime.
Thread
   The threading framework is written in C++ and exposed in C. It provides a thread pool interface
   to process tasks asynchronously. To enable use in MuJoCo, call ``mju_threadpool``.
Abstract visualizer
   The abstract visualizer is written in C and is independent of any graphics library. It provides abstract mouse
   hooks for camera and perturbation control, object selection, visualization options, and generates a list of
   abstract geometric entities representing the simulation state and its decorations. It is shared by both renderers.
Rendering
   There are two rendering libraries provided by MuJoCo. The :ref:`classic rendering<ClassicRendering>` library is
   written in C and uses OpenGL 1.5. It provides a simple and efficient way to visualize MuJoCo models. The
   :ref:`filament rendering<FilamentRendering>` library is written in C++ and uses the externally-developed Filament
   rendering engine. It provides more modern and feature-rich real-time rendering capabilities.
UI framework
   The UI framework is written in C and is designed to work with the :ref:`classic OpenGL renderer<OpenGLrendering>`.
   UI elements are rendered in OpenGL. It has its own event mechanism and abstract hooks for keyboard and mouse input.
   The code samples use it with GLFW, but it can also be used with other window libraries.

.. _inStart:

Getting started
~~~~~~~~~~~~~~~

MuJoCo is an open-source project. Pre-built dynamic libraries are available for x86_64 and arm64 machines running
Windows, Linux, and macOS. These can be downloaded from the `GitHub Releases page
<https://github.com/google-deepmind/mujoco/releases>`_. Users who do not intend to develop or modify core MuJoCo code
are encouraged to use our pre-built libraries, as these come bundled with the same versions of dependencies that we
regularly test against, and benefit from build flags that have been tuned for performance. Our pre-built libraries are
almost entirely self-contained and do not require any other library to be present, outside the standard C runtime. We
also hide all symbols apart from those that form MuJoCo's public API, thus ensuring that it can coexist with any other
libraries that may be loaded into the process (including other versions of libraries that MuJoCo depends on).

The pre-built distribution is a single ``.dmg`` on macOS, ``.zip`` on Windows, and ``.tar.gz`` on Linux; there is no
installer. On macOS and Windows, both :ref:`MuJoCo Studio <StudioApp>` and :ref:`simulate <saSimulate>` are packaged as
standalone desktop applications (``MuJoCoStudio.app`` and ``MuJoCo.app`` in the macOS ``.dmg``;
``bin\mujoco_studio.exe`` and ``bin\simulate.exe`` in the Windows ``.zip``) that you can double-click to launch and
drop model files onto.

On Windows and Linux, extract the archive in a directory of your choice. From a terminal in the ``bin`` subdirectory,
you can also run :ref:`MuJoCo Studio <StudioApp>` or the precompiled code samples such as :ref:`simulate <saSimulate>`:

.. code-block:: Text

     Windows:   mujoco_studio ..\model\humanoid\humanoid.xml
     Linux:     ./mujoco_studio ../model/humanoid/humanoid.xml

The directory structure of the Windows and Linux archives is shown below. Users can re-organize it if needed, as well as
install the dynamic libraries in other directories and set the path accordingly. The only file created automatically is
MUJOCO_LOG.TXT in the executable directory; it contains error and warning messages, and can be deleted at any time.

.. code-block:: Text

     bin     - dynamic libraries, executables, MUJOCO_LOG.TXT
     doc     - README.txt and REFERENCE.txt
     include - header files needed to develop with MuJoCo
     model   - model collection
     sample  - code samples and CMakeLists.txt needed to build them

After verifying that the simulator works, you may also want to re-compile the code samples to ensure that you have a
working development environment. We provide a cross-platform `CMake
<https://github.com/google-deepmind/mujoco/blob/main/sample/CMakeLists.txt>`__ setup that can be used to build sample
applications independently of the MuJoCo library itself.

On macOS, you can drag ``MuJoCoStudio.app`` and ``MuJoCo.app`` from the ``.dmg`` into ``/Applications`` to install them
like any other `application bundle <https://developer.apple.com/go/?id=bundle-structure>`__ (their executables inside
``Contents/MacOS/`` can also be run from a terminal). Alongside the app bundles, the DMG includes the
``mujoco.framework`` subdirectory containing the MuJoCo dynamic library and all of
its public headers. If you are using Xcode, you can import it as a framework dependency on your project. (This also
works for Swift projects without any modification). If you are building manually, you can use ``-F`` and
``-framework mujoco`` to specify the header search path and the library search path respectively.

.. _inBuild:

Building from source
~~~~~~~~~~~~~~~~~~~~

To build MuJoCo from source, you will need CMake and a working C++17 compiler installed. The steps are:

#. Clone the ``mujoco`` repository: ``git clone https://github.com/google-deepmind/mujoco.git``
#. Create a new build directory and ``cd`` into it.
#. Run :shell:`cmake $PATH_TO_CLONED_REPO` to configure the build.
#. Run ``cmake --build .`` to build.

MuJoCo's build system automatically fetches dependencies from upstream repositories over the Internet using CMake's
`FetchContent <https://cmake.org/cmake/help/latest/module/FetchContent.html>`_ module.

The main CMake setup will build the MuJoCo library itself along with all sample applications, but the Python
bindings are not built. Those come with their own build instructions, which can be found in the :doc:`../python`
section of the documentation.

Additionally, the CMake setup also implements an installation phase which will copy and organize the output files to a
target directory.

#. Select the directory: :shell:`cmake $PATH_TO_CLONED_REPO -DCMAKE_INSTALL_PREFIX=<my_install_dir>`
#. After building, install with ``cmake --install .``
#. If desired, proceed to building the Python bindings - see
   :ref:`PyBuild`.

**Notes:**

- To optimize runtime performance build with ``-DCMAKE_BUILD_TYPE=Release``.
- When building on Windows with MSVC, use Visual Studio 2019 or later and make sure Windows SDK version 10.0.22000 or
  later is installed (see :issue:`862` for more details).
- We've found that performance on Windows is best when building with Clang, rather than MSVC.

.. tip::
   As a reference, a working build configuration can be found in MuJoCo's
   `continuous integration setup <https://github.com/google-deepmind/mujoco/blob/main/.github/workflows/build.yml>`_ on
   GitHub.

.. _inBuildBazel:

Building with Bazel
~~~~~~~~~~~~~~~~~~~

MuJoCo can also be built with `Bazel <https://bazel.build/>`__ 8.0 or later. The Bazel version it is tested with is
pinned in ``.bazelversion``, which `Bazelisk <https://github.com/bazelbuild/bazelisk>`__ uses automatically. To build
and test MuJoCo from the root of the cloned repository:

.. code-block:: shell

   bazel build //...
   bazel test //...

To use MuJoCo from your own Bazel module, add it to your ``MODULE.bazel``. MuJoCo isn't in the
`Bazel Central Registry <https://registry.bazel.build/>`__ yet, so use an override, for example:

.. code-block:: python

   bazel_dep(name = "mujoco")
   git_override(
       module_name = "mujoco",
       remote = "https://github.com/google-deepmind/mujoco.git",
       commit = "<commit>",
   )

The main targets are:

- ``@mujoco``: the MuJoCo library. Its headers are included as ``#include <mujoco/mujoco.h>``.
- ``@mujoco//:mujoco_shared``: MuJoCo as a shared library (``libmujoco.so``, or ``libmujoco.dylib`` on macOS) which,
  like the one built with CMake, exports only the MuJoCo API. Add it to the ``dynamic_deps`` of a ``cc_binary`` or
  ``cc_shared_library`` that depends on ``@mujoco`` to link MuJoCo dynamically, for example to share one copy of MuJoCo
  with the Python bindings.
- ``@mujoco//plugin/<name>:register``, where ``<name>`` is ``actuator``, ``elasticity``, ``sdf`` or ``sensor``: the
  bundled plugins, which register themselves when linked in.
- ``@mujoco//python/mujoco``: the Python bindings, see :ref:`PyBuildBazel`.

To build your own :ref:`plugin<exPlugin>` as a shared library, like the bundled ones, use the ``mujoco_plugin`` macro.
The library links against ``@mujoco//:mujoco_shared`` and exports no symbols; load it at run time with
:ref:`mj_loadPluginLibrary`, which runs its :ref:`registration<exRegistration>` code:

.. code-block:: python

   load("@mujoco//plugin:mujoco_plugin.bzl", "mujoco_plugin")

   cc_library(
       name = "my_plugin_lib",
       srcs = ["my_plugin.cc"],
       deps = ["@mujoco"],
       alwayslink = True,
   )

   # Builds libmy_plugin.so (libmy_plugin.dylib on macOS).
   mujoco_plugin(
       name = "my_plugin",
       deps = [":my_plugin_lib"],
   )

To use single-precision floating point (``mjUSESINGLE``), build with ``--@mujoco//:single_precision``.

The Bazel build is tested on Linux and macOS, and covers the MuJoCo library, ``simulate``, the plugins, the samples and
the core Python bindings.

.. tip::
   The ``sample`` directory is a standalone Bazel module that depends on MuJoCo this way.

.. _inBuildDocs:

Building the docs
~~~~~~~~~~~~~~~~~

If you wish to build the documentation locally, for example to test pull-requests that improve it, do:

1. Clone the ``mujoco`` repository: ``git clone https://github.com/google-deepmind/mujoco.git``
2. Go to the ``doc/`` directory: ``cd mujoco/doc``
3. Install the dependencies: ``pip install -r requirements.txt``
   |br| Note that the MuJoCo Warp API documentation is autogenerated and requires additional dependencies.
   See `.readthedocs.yml <https://github.com/google-deepmind/mujoco/blob/main/.readthedocs.yml>`__ for details.
4. Build the HTML: ``make html``
5. Open ``_build/html/index.html`` in your browser of choice.

.. _inHeader:

Header files
~~~~~~~~~~~~

The distribution contains several header files which are identical on all platforms. They are also available from the
links below, to make this documentation self-contained.

`mujoco.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h>`__
   This is the main header file and must be included in all programs using MuJoCo. It defines all API functions and
   global variables, and includes all other header files except mjxmacro.h and mjspecmacro.h.
`mjmodel.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h>`__
   Defines the C structure :ref:`mjModel` which is the runtime representation of the
   model being simulated.
`mjdata.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h>`__
   Defines the C structure :ref:`mjData` which is the workspace where all computations
   read their inputs and write their outputs.
`mjvisualize.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjvisualize.h>`__
   Defines the primitive types and structures needed by the abstract visualizer.
`mjrender.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrender.h>`__
   Defines the primitive types and structures needed by the OpenGL renderer.
`mjrfilament.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrfilament.h>`__
   Defines the primitive types and structures needed by the filament renderer.
`mjui.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjui.h>`__
   Defines the primitive types and structures needed by the UI framework.
`mjtype.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h>`__
   Defines primitive types and enums, including the ``mjtNum`` floating-point type to be either ``double`` or ``float``
   (see :ref:`mjtNum`).
`mjspec.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjspec.h>`__
   Defines enums and structs used for :doc:`procedural model editing <modeledit>`.
`mjplugin.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjplugin.h>`__
   Defines data structures required by :ref:`engine plugins<exPlugin>`.
`mjmacro.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmacro.h>`__
   Defines C macros that are useful in user code.
`mjxmacro.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjxmacro.h>`__
   This file is optional and is not included by mujoco.h. It defines :ref:`X Macros <tyXMacro>` that can
   automate the mapping of mjModel and mjData into scripting languages, as well as other operations that require
   accessing all fields of mjModel and mjData.
`mjspecmacro.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjspecmacro.h>`__
   This file is optional and is not included by mujoco.h. It defines :ref:`X Macros <tyXMacro>` that can
   automate the mapping of mjSpec and its element structs into scripting languages, as well as other operations that require
   accessing all fields of mjSpec during procedural model editing.
`mjexport.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjexport.h>`__
   Macros used for exporting public symbols from the MuJoCo library. This header should not be used directly by client
   code.
`mjsan.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjsan.h>`__
   Definitions required when building with sanitizer instrumentation.
`mjassert.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjassert.h>`__
   Compile-time size assertions verifying MuJoCo ABI stability across C and C++ compilers.

.. _inVersion:

Versions and compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~

MuJoCo has been used extensively since 2010 and is quite mature (even though our version numbering scheme is quite
conservative). Nevertheless it remains under active development, and we have many exciting ideas for new features and
are also making changes based on user feedback. This leads to unavoidable changes in both the modeling language and the
API. While we encourage users to upgrade to the latest version, we recognize that this is not always feasible,
especially when other developers release software that relies on MuJoCo. Therefore we have introduced simple
mechanisms to help avoid version conflicts, as follows.

The situation is more subtle if existing code was developed with a certain version of MuJoCo, and is now being
compiled and linked with a different version. If the definitions of the API functions used in that code have changed,
either the compiler or the linker will generate errors. But even if the function definitions have not changed, it may
still be a good idea to assert that the software version is the same. To this end, the main header (mujoco.h) defines
the symbol :ref:`mjVERSION_HEADER <glNumericVersion>` and the library provides the function
:ref:`mj_version`. Thus the header and library versions can be compared with:

.. code-block:: C

   // recommended version check
   if (mjVERSION_HEADER != mj_version())
     complain();

Note that only the main header defines this symbol. We assume that the collection of headers released with each software
version will stay together and will not be mixed between versions. To avoid complications with floating-point
comparisons, the above symbol and function use integers rather than floating-point numbers. See
`VERSIONING.md <https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`__ for the encoding formula and
version semantics.

.. _inNaming:

Naming convention
~~~~~~~~~~~~~~~~~

All symbols defined in the API start with the prefix "mj". The character after "mj" in the prefix determines the family
to which the symbol belongs. First we list the prefixes corresponding to type definitions.

``mj``
   Core simulation data structure (C struct), for example :ref:`mjModel`. If all characters
   after the prefix are capital, for example :ref:`mjMIN`, this is a macro or a symbol (#define).
``mjt``
   Primitive type, for example :ref:`mjtNum` and :ref:`mjtGeom`. Most types in this family are enums.
``mjf``
   Callback function type, for example :ref:`mjfGeneric`.
``mjs``
   Data structure related to :doc:`procedural model editing <modeledit>`, for example :ref:`mjsJoint`.
``mjv``
   Data structure related to abstract visualization, for example :ref:`mjvCamera`.
``mjrf``
   Data structure related to filament rendering, for example :ref:`mjrfContext`.
``mjr``
   Data structure related to OpenGL rendering, for example :ref:`mjrContext`.
``mjui``
   Data structure related to UI framework, for example :ref:`mjuiSection`.

Next we list the prefixes corresponding to function definitions. Note that function prefixes always end with underscore.

``mj_``
   Core simulation function, for example :ref:`mj_step`. Almost all such functions have
   pointers to mjModel and mjData as their first two arguments, possibly followed by other arguments. They usually write
   their outputs to mjData.
``mju_``
   Utility function, for example :ref:`mju_mulMatVec`. These functions are self-contained
   in the sense that they do not have mjModel and mjData pointers as their arguments.
``mjv_``
   Function related to abstract visualization, for example :ref:`mjv_updateScene`.
``mjrf_``
   Function related to filament rendering, for example :ref:`mjrf_render`.
``mjr_``
   Function related to OpenGL rendering, for example :ref:`mjr_render`.
``mjui_``
   Function related to UI framework, for example :ref:`mjui_update`.
``mjcb_``
   Global callback function pointer, for example :ref:`mjcb_control`. The user can install
   custom callbacks by setting these global pointers to user-defined functions.
``mjd_``
   Functions for computing derivatives, for example :ref:`mjd_transitionFD`.
``mjs_``
   Functions for :doc:`procedural model editing <modeledit>`, for example :ref:`mjs_addJoint`.
``mjb_``
   Functions for :ref:`batched simulation<siBatch>`, declared in ``mujoco/mjbatch.h``, for example :ref:`mjb_step`.

.. toctree::
    :hidden:

    simulation
    visualization
    rendering
    ui
    modeledit
    samples
    extension
