.. _Sample:

Code samples
------------

MuJoCo comes with several code samples providing useful functionality. Some of them are quite elaborate
(:ref:`simulate.cc <saSimulate>` in particular) but nevertheless we hope that they will help users learn how to program
with the library.

.. _saTestspeed:

`testspeed <https://github.com/google-deepmind/mujoco/blob/main/sample/testspeed.cc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample times the simulation of a given model. The timing is straightforward: the simulation of the passive
dynamics (with optional control noise) is rolled-out for the specified number of steps, while collecting statistics
about the number of contacts, scalar constraints, and CPU times from internal profiling. The results are then printed to
the console. To simulate controlled dynamics instead of passive dynamics one can either install a control callback
:ref:`mjcb_control`, or modify the code to set control signals explicitly, as explained in the :ref:`simulation loop
<siSimulation>` section below. This command-line utility is run with

.. code-block:: Shell

   testspeed [options] model

Where the command-line options and arguments are

.. list-table::
   :width: 95%
   :align: left
   :widths: 2 1 4
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``model``
     - (required)
     - path to model (positional argument)
   * - ``--nstep=N``
     - 10000
     - number of steps per rollout
   * - ``--nthread=N``
     - 1
     - number of threads running parallel rollouts
   * - ``--noisestd=X``
     - 0.01
     - scale of pseudo-random noise injected into actuators
   * - ``--noiserate=X``
     - 0.1
     - rate of convergence to ctrl keyframe/midpoint
   * - ``--nenginethread=N``
     - 0
     - number of worker threads in engine-internal threadpool
   * - ``--solver=S``
     - Newton
     - override constraint solver algorithm (PGS, CG, Newton)
   * - ``--cone=C``
     - Pyramidal
     - override friction cone type (Pyramidal, Elliptic)
   * - ``--jacobian=J``
     - Auto
     - override constraint Jacobian type (Dense, Sparse, Auto)
   * - ``--integrator=I``
     - Euler
     - override integration mode (Euler, RK4, Implicit, ImplicitFast)
   * - ``--iterations=N``
     - 100
     - override solver iterations limit
   * - ``--tolerance=X``
     - 1e-8
     - override solver convergence tolerance
   * - ``--sleep_tolerance=X``
     - 1e-4
     - override sleep tolerance
   * - ``--noslip_iterations=N``
     - 0
     - override noslip solver iterations limit

**Notes:**

- When ``nthread > 1`` is specified, the code allocates a single mjModel and per-thread mjData, and runs ``nthread``
  identical simulations in parallel. This tests performance with all cores active, as in Reinforcement
  Learning scenarios where samples are collected in parallel. The optimal ``nthread`` usually equals the number of
  logical cores.
- By default, the simulation starts from the model reference configuration with zero velocities. However, if a
  keyframe named "test" is present in the model, it is used as the initial state.
- The physics option override flags (such as ``--solver``) only override the model settings if they are explicitly
  specified on the command line; otherwise, the model options configured in the XML file are preserved.
- The control noise arguments (``noisestd`` and ``noiserate``) prevent models from settling into a static state where,
  due to warmstarts, one can measure artificially faster simulation.
- When ``nenginethread > 0`` is specified, an engine-internal thread pool is created with the specified number of
  worker threads (in addition to the calling thread), to speed up simulation of large scenes. Note that while it is
  possible to use both ``nthread`` and ``nenginethread``, the scenarios for which one would want these different types
  of multithreading are usually mutually exclusive.
- For more repeatable performance statistics, run the tool with the ``performance``
  `governor <https://www.kernel.org/doc/Documentation/cpu-freq/governors.txt>`__ on Linux, or the
  ``High Performance`` power plan on Windows, to reduce noise from CPU scaling.
- Many modern CPUs contain a mixture of "performance" and "efficiency" cores. Users should consider restricting the
  process to only run on the same type of cores for more interpretable performance statistics. This can be done via the
  `taskset <https://man7.org/linux/man-pages/man1/taskset.1.html>`__ command on Linux, or the
  `start /affinity <https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/start>`__
  command on Windows (processor affinity cannot be specified through documented API means on macOS).

.. _saSimulate:

`simulate <https://github.com/google-deepmind/mujoco/blob/main/simulate>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample is a fully-featured interactive simulator. It opens an OpenGL window using the platform-independent
GLFW library, and renders the simulation state in it. There is built-in help, simulation statistics, profiler, sensor
data plots. The model file can be specified as a command-line argument, or loaded at runtime using drag-and-drop
functionality. This code sample uses the native UI to render various controls, and provides an
illustration of how the new UI framework is intended to be used. Below is a screen-capture of ``simulate`` in action:

..  youtube:: P83tKA1iz2Y
    :aspect: 16:9
    :width: 95%
    :align: center

Interaction is done with the mouse; built-in help with a summary of available commands is available by pressing the
``F1`` key. Briefly, an object is selected by left-double-click. The user can then apply forces and torques on the
selected object by holding Ctrl and dragging the mouse. Dragging the mouse alone (without Ctrl) moves the camera. There
are keyboard shortcuts for pausing the simulation, resetting, and re-loading the model file. The latter functionality is
very useful while editing the model in an XML editor. The complete set of shortcuts is given in
:ref:`saSimulateShortcuts` below.

.. _saSimulateShortcuts:

Shortcuts
^^^^^^^^^

The ``F1`` help overlay lists the most common commands. The tables below are the complete reference, and also apply to
the Python viewer launched with :ref:`mujoco.viewer<PyViewer>`, which is built on the same ``Simulate`` UI. Shortcuts
that step or pause the simulation have no effect in passive mode, where stepping is driven by user code.

In addition to the shortcuts listed here, every UI control shows its own shortcut when the right mouse button is held
over the UI panel.

Simulation and camera
"""""""""""""""""""""

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 3
   :header-rows: 1

   * - Key
     - Action
   * - ``Space``
     - Play / pause
   * - ``Right arrow``
     - Step forward
   * - ``Left arrow``
     - Step backward, through the history buffer
   * - ``+`` (actually ``=``)
     - Speed up
   * - ``-``
     - Slow down
   * - ``Backspace``
     - Reset
   * - ``Ctrl C``
     - Copy state to clipboard
   * - ``Ctrl L``
     - Reload the model
   * - ``Ctrl A``
     - Align the free camera
   * - ``Esc``
     - Switch to the free camera
   * - ``[`` / ``]``
     - Cycle down / up through the fixed cameras defined in the model
   * - ``Page Up``
     - Select the parent of the currently selected body
   * - ``Tab`` / ``Shift Tab``
     - Toggle the left / right UI panel

Panels and files
""""""""""""""""

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 3
   :header-rows: 1

   * - Key
     - Action
   * - ``F1``
     - Toggle the help overlay
   * - ``F2``
     - Toggle the info overlay
   * - ``F3``
     - Toggle the profiler
   * - ``F4``
     - Toggle the sensor plots
   * - ``F5``
     - Toggle fullscreen
   * - ``F6``
     - Cycle the frame visualization
   * - ``F7``
     - Cycle the label visualization
   * - ``Ctrl M``
     - Print the model to ``MJMODEL.TXT``
   * - ``Ctrl D``
     - Print the data to ``MJDATA.TXT``
   * - ``Ctrl P``
     - Save a screenshot
   * - ``Ctrl Q``
     - Quit
   * - ``Alt`` + letter
     - Expand / collapse a UI section: ``F`` File, ``O`` Option, ``S`` Simulation, ``W`` Watch, ``P`` Physics,
       ``R`` Rendering, ``V`` Visualization, ``G`` Group enable, ``L`` Logging, ``J`` Joint, ``C`` Control,
       ``E`` Equality

Visibility groups
"""""""""""""""""

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 3
   :header-rows: 1

   * - Key
     - Action
   * - ``0`` … ``5``
     - Toggle visibility of geom group 0 … 5
   * - ``Shift 0`` … ``Shift 5``
     - Toggle visibility of site group 0 … 5

Visualization flags
"""""""""""""""""""

These toggle the abstract visualization flags in the Rendering section, and correspond to the shortcuts declared in
``mjVISSTRING``. Flags with no shortcut assigned (Select Point, the Flex flags and SDF iters) are omitted.

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 3 1 3
   :header-rows: 1

   * - Key
     - Flag
     - Key
     - Flag
   * - ``H``
     - Convex Hull
     - ``X``
     - Texture
   * - ``J``
     - Joint
     - ``Q``
     - Camera
   * - ``U``
     - Actuator
     - ``,``
     - Activation
   * - ``Z``
     - Light
     - ``V``
     - Tendon
   * - ``Y``
     - Range Finder
     - ``E``
     - Equality
   * - ``I``
     - Inertia
     - ``'``
     - Scale Inertia
   * - ``B``
     - Perturb Force
     - ``O``
     - Perturb Object
   * - ``C``
     - Contact Point
     - ``N``
     - Island
   * - ``F``
     - Contact Force
     - ``P``
     - Contact Split
   * - ``T``
     - Transparent
     - ``A``
     - Auto Connect
   * - ``M``
     - Center of Mass
     - ``D``
     - Static Body
   * - ``;``
     - Skin
     - :literal:`\``
     - Body Tree
   * - ``\``
     - Mesh Tree
     -
     -

Rendering flags
"""""""""""""""

These toggle the OpenGL effects in the Rendering section, and correspond to the shortcuts declared in ``mjRNDSTRING``.
Flags with no shortcut assigned (Depth, Id Color and Cull Face) are omitted.

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 3 1 3
   :header-rows: 1

   * - Key
     - Flag
     - Key
     - Flag
   * - ``S``
     - Shadow
     - ``W``
     - Wireframe
   * - ``R``
     - Reflection
     - ``L``
     - Additive
   * - ``K``
     - Skybox
     - ``G``
     - Fog
   * - ``/``
     - Haze
     -
     -

Mouse
"""""

.. list-table::
   :width: 95%
   :align: left
   :widths: 2 3
   :header-rows: 1

   * - Action
     - Effect
   * - Left drag
     - Orbit the camera
   * - Right drag
     - Pan camera in vertical plane
   * - ``Shift`` right drag
     - Pan camera in horizontal plane
   * - Scroll, or middle drag
     - Zoom
   * - Double-click
     - Select an object
   * - Right double-click
     - Center the camera on the clicked point
   * - ``Ctrl`` right double-click
     - Track the selected body
   * - ``Ctrl`` drag
     - Rotate the selected object
   * - ``Ctrl`` right drag
     - Translate object in vertical plane
   * - ``Ctrl`` ``Shift`` right drag
     - Translate object in horizontal plane
   * - Right-button hold over the UI
     - Show the shortcut of each UI control
   * - Double-click a UI section title
     - Expand / collapse all sections

The code is long yet reasonably commented, so it is best to just read it. Here we provide a high-level overview.
The ``main()`` function initializes both MuJoCo and GLFW, opens a window, and install GLFW callbacks for mouse and
keyboard handling. Note that there is no render callback; GLFW puts the user in charge, instead of running a rendering
loop behind the scenes. The main loop handles UI events and rendering. The simulation is handled in a background
thread, which is synchronized with the main thread.

The mouse and keyboard callbacks perform whatever action is necessary. Many of these actions invoke functionality
provided by MuJoCo's :ref:`abstract visualization <Abstract>` mechanism. Indeed this mechanism is designed to be
hooked to mouse and keyboard events more or less directly, and provides camera as well as perturbation control.

The profiler and sensor data plots illustrate the use of the :ref:`mjr_figure` function
that can plot elaborate 2D figures with grids, annotation, axis scaling etc. The information presented in the profiler
is extracted from the diagnostic fields of mjData. It is a very useful tool for tuning the parameters of the
constraint solver algorithms. The outputs of the sensors defined in the model are visualized as a bar graph.

Note that the profiler shows timing information collected with high-resolution timers. On Windows, depending on the
power settings, the OS may reduce the CPU frequency; this is because :ref:`simulate.cc <saSimulate>` sleeps most of
the time in order to slow down to realtime. This results in inaccurate timings. To avoid this problem, change the
Windows power plan so that the minimum processor state is 100%.

.. _saCompile:

`compile <https://github.com/google-deepmind/mujoco/blob/main/sample/compile.cc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample invokes the built-in parser and compiler. It implements all possible model conversions from (MJCF,
URDF, MJB) format to (MJCF, MJB, TXT) format. Models are saved as MJCF as :ref:`Model Encoding & Saving <meSaving>`
describes: by default with the values which compilation made of them, in a canonical subset of the format, so that
MJCF-to-MJCF conversion will generally result in a different file; with :ref:`savecompiled<compiler-savecompiled>`
"false", as they are written, which keeps their notation but not their comments, :ref:`include<include>` elements and
the elements which the parser expands.
The TXT format is a human-readable road-map to the model. It cannot be loaded by MuJoCo, but can be a very useful aid
during model development. It is in one-to-one correspondence with the compiled mjModel. Note also that one can use the
function :ref:`mj_printData` to create a text file which is in one-to-one correspondence
with mjData, although this is not done by the code sample.

If the input file is MJCF or URDF and the output file is empty, compilation is performed twice to measure the impact
of the compiler's :ref:`asset cache<Assetcache>`. A detailed timing breakdown is printed for each compilation, showing
total time, asset processing time (wall clock), and per-category CPU times for meshes and textures. These timings are
read from the :ref:`mjtCTimer` fields via :ref:`mjs_getTimer`, which can be read programmatically
after any call to :ref:`mj_compile`.

.. _saBasic:

`basic <https://github.com/google-deepmind/mujoco/blob/main/sample/basic.cc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample is a minimal interactive simulator. The model file must be provided as command-line argument. It
opens an OpenGL window using the platform-independent GLFW library, and renders the simulation state at 60 fps while
advancing the simulation in real-time. Press Backspace to reset the simulation. The mouse can be used to control the
camera: left drag to rotate, right drag to translate in the vertical plane, shift right drag to translate in the
horizontal plane, scroll or middle drag to zoom.

The :ref:`Visualization` programming guide below explains how visualization works. This code sample is a minimal
illustration of the concepts in that guide.

.. _saRecord:

`record <https://github.com/google-deepmind/mujoco/blob/main/sample/record.cc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample simulates the passive dynamics of a given model, renders it offscreen, reads the color and depth pixel
values, and saves them into a raw data file that can then be converted into a movie file with tools such as ffmpeg. The
rendering is simplified compared to :ref:`simulate.cc <saSimulate>` because there is no user interaction, visualization
options or timing; instead we simply render with the default settings as fast as possible. The dimensions and number of
multi-samples for the offscreen buffer are specified in the MuJoCo model with the visual/global/{`offwidth
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global-offwidth>`__, `offheight
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-global-offheight>`__} and visual/quality/`offsamples
<https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-quality-offsamples>`_ attributes, while the simulation
duration, frames-per-second to be rendered (usually much less than the physics simulation rate), and output file name
are specified as command-line arguments.

.. code-block:: Shell

   record modelfile duration fps rgbfile [adddepth]

Where the command line arguments are

.. list-table::
   :width: 95%
   :align: left
   :widths: 1 1 5
   :header-rows: 1

   * - Argument
     - Default
     - Meaning
   * - ``modelfile``
     - (required)
     - path to model
   * - ``duration``
     - (required)
     - duration of the recording in seconds
   * - ``fps``
     - (required)
     - number of frames per second
   * - ``rgbfile``
     - (required)
     - path to raw recording file
   * - ``adddepth``
     - 1
     - overlay depth image in the lower left corner (0: none)

For example, a 5 second animation at 60 frames per second is created with:

.. code-block:: Shell

   record humanoid.xml 5 60 rgb.out

The default `humanoid.xml <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid.xml>`__ model
specifies offscreen rendering with 2560x1440 resolution. With this information in hand, we can compress the (large) raw
data file into a playable movie file:

.. code-block:: Shell

   ffmpeg -f rawvideo -pixel_format rgb24 -video_size 2560x1440
          -framerate 60 -i rgb.out -vf "vflip,format=yuv420p" video.mp4

Note that the offscreen rendering resolution of the model and ffmpeg's video_size must be identical.

This sample can be compiled in three ways which differ in how the OpenGL context is created: using GLFW with an
invisible window, using OSMesa, or using EGL. The latter two options are only available on Linux and are invoked by
defining the symbols MJ_OSMESA or MJ_EGL when compiling record.cc. The functions ``initOpenGL`` and ``closeOpenGL``
create and close the OpenGL context in three different ways depending on which of the above symbols is defined.

Note that the MuJoCo rendering code does not depend on how the OpenGL context was created. This is the beauty of
OpenGL: it leaves context creation to the platform, and the actual rendering is then standard and works in the same
way on all platforms. In retrospect, the decision to leave context creation out of the standard has led to unnecessary
proliferation of overlapping technologies, which differ not only between platforms but also within a platform in the
case of Linux. The addition of a couple of extra functions (such as those provided by OSMesa for example) could have
avoided a lot of confusion. EGL is a newer standard from Khronos which aims to do this, and it is gaining popularity.
But we cannot yet assume that all users have it installed.

.. _saRender:

`render <https://github.com/google-deepmind/mujoco/blob/main/sample/render.cc>`_
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This code sample renders a model configuration offscreen and saves the result directly to a PNG image file. It supports
both the modern Filament rendering backend (:ref:`mjrf <FilamentRenderingApi>`) with physically-based rendering (PBR),
and the classic OpenGL renderer (:ref:`mjrContext`). The rendering runs headless without creating an interactive window.

.. code-block:: Shell

   render model [output] [options]

Where the command-line arguments and options are

.. list-table::
   :width: 95%
   :align: left
   :widths: 2 1 4
   :header-rows: 1

   * - Option
     - Default
     - Meaning
   * - ``model``
     - (required)
     - path to model (positional argument)
   * - ``output``
     - (default)
     - path to output image file (positional argument, default: ``<model>.png``)
   * - ``--backend=B``
     - ``filament``
     - rendering backend (``filament``, ``classic``)
   * - ``--width=N``
     - 640
     - image width in pixels
   * - ``--height=N``
     - 480
     - image height in pixels
   * - ``--camera=C``
     - (free)
     - camera name or integer index (default: free camera)
   * - ``--lookat=X,Y,Z``
     - (center)
     - free camera lookat position
   * - ``--distance=D``
     - (fit)
     - free camera distance
   * - ``--azimuth=A``
     - ``90``
     - free camera azimuth in degrees
   * - ``--elevation=E``
     - ``-45``
     - free camera elevation in degrees
   * - ``--key=K``
     - (none)
     - keyframe name or integer index to load prior to rendering
   * - ``--steps=N``
     - 0
     - number of simulation steps to advance before rendering
   * - ``--geomgroup=G``
     - ``111000``
     - 6-character string enabling/disabling geom groups 0-5 (e.g. ``111111``)
   * - ``--sitegroup=S``
     - ``111000``
     - 6-character string enabling/disabling site groups 0-5
   * - ``--label=L``
     - (none)
     - type of label (e.g. ``--label=geom``, ``--label=body``, ``--label=site``)
   * - ``--frame=F``
     - (none)
     - type of coordinate frame (e.g. ``--frame=geom``, ``--frame=body``, ``--frame=site``)
   * - ``--<visflag>=0|1``
     - (default)
     - toggle visualization flag declared in :ref:`mjVISSTRING` (e.g. ``--joint=1``, ``--contactpoint=1``)
   * - ``--<rndflag>=0|1``
     - (default)
     - toggle rendering flag declared in :ref:`mjRNDSTRING` (e.g. ``--shadow=0``, ``--wireframe=1``)

**Notes:**

- Running ``render`` with no arguments (or with ``--help`` / ``-h``) prints the help message.
- The non-optional ``model`` path and optional ``output`` path are positional arguments. If ``output`` is
  omitted, the output filename is derived from the model filename with the ``.png`` extension (e.g.
  ``humanoid.xml`` becomes ``humanoid.png``).
- ``--backend=filament`` uses MuJoCo's Filament C API to configure a scene from the model, setting up lighting,
  materials, and textures, and rendering via Filament's offscreen pipeline. ``--backend=classic`` uses standard OpenGL
  offscreen rendering with an EGL context.
- The default camera is the model's free camera looking at the center of the scene. Specifying ``--camera=name`` selects
  a named camera defined in the model XML, while integer indices ``0, 1, ...`` select fixed cameras in order of definition.
  When using the free camera, its viewpoint can be adjusted via ``--lookat=x,y,z``, ``--distance=d``, ``--azimuth=a``,
  and ``--elevation=e``.
- If a keyframe is specified via ``--key``, the model state is reset to that keyframe before evaluating kinematics and
  rendering. If no keyframe is specified and a keyframe named ``test`` exists in the model, it is used by default.
- ``--steps=N`` advances the simulation by ``N`` steps (after any keyframe reset) before rendering, so a state reached by
  simulating can be captured rather than only the initial or keyframe pose.
- Visualization and rendering flags can be individually toggled using normalized flag names from :ref:`mjVISSTRING`
  and :ref:`mjRNDSTRING`.
- Labels can be enabled using ``--label=L`` where ``L`` is a label type name from :ref:`mjLABELSTRING` (e.g. ``geom``,
  ``body``, ``joint``) or an integer index.
- Coordinate frames can be rendered using ``--frame=F`` where ``F`` is a frame type name from :ref:`mjFRAMESTRING` (e.g.
  ``geom``, ``body``, ``site``, ``world``) or an integer index.

