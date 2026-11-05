=====
Types
=====

MuJoCo defines a large number of types:

- Four :ref:`primitive types<tyPrimitive>`: :ref:`mjtNum<mjtNum>`, :ref:`mjtByte<mjtByte>`, :ref:`mjtBool<mjtBool>`, and
  :ref:`mjtSize<mjtSize>`.

- :ref:`C enum types<tyEnums>` used to define categorical values. These can be classified as:

  - Enums used in :ref:`mjModel<tyModelEnums>`.
  - Enums used in :ref:`mjData<tyDataEnums>`.
  - Enums for abstract :ref:`visualization<tyVisEnums>`.
  - Enums used by the :ref:`classic renderer<tyRenderEnums>`.
  - Enums used by the :ref:`filament renderer<tyFilamentRenderEnums>`.
  - Enums used by the :ref:`mjUI<tyUIEnums>` user interface package.
  - Enums used by :ref:`engine plugins<tyPluginEnums>`.
  - Enums used for :ref:`procedural model manipulation<tySpecEnums>`.

  Note that the API does not use these enum types directly. Instead it uses ints, and the documentation/comments state
  that certain ints correspond to certain enum types. This is because we want the API to be compiler-independent, and
  the C standard does not dictate how many bytes must be used to represent an enum type. Nevertheless, for improved
  readability, we recommend using these types when calling API functions which take them as arguments.

- :ref:`C struct types<tyStructure>`. These can be classified as:

  - Main structs:

    - :ref:`mjModel`.
    - :ref:`mjOption` (embedded in :ref:`mjModel`).
    - :ref:`mjData`.

  - :ref:`Auxiliary struct types<tyAuxStructure>`, also used by the engine.
  - Structs for collecting :ref:`simulation statistics<tyStatStructure>`.
  - Structs for :ref:`abstract visualization<tyVisStructure>`.
  - Structs used by the :ref:`classic renderer<tyRenderStructure>`.
  - Structs used by the :ref:`filament renderer<tyFilamentRenderStructure>`.
  - Structs used by the :ref:`UI framework<tyUIStructure>`.
  - Structs used for :ref:`procedural model manipulation<tySpecStructure>`.
  - Structs used by :ref:`engine plugins<tyPluginStructure>`.

- Several :ref:`function types<tyFunction>` for user-defined callbacks.
- :ref:`tyNotes` regarding specific data structures that require detailed description.



.. _tyPrimitive:

Primitive types
---------------

The three types below are defined in `mjtype.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h>`__.


.. _mjtNum:

mjtNum
^^^^^^

This is the floating-point type used throughout the simulator. When using the default build configuration, ``mjtNum`` is
defined as ``double``. If the symbol ``mjUSESINGLE`` is defined, ``mjtNum`` is defined as ``float``.

Currently only the double-precision version of MuJoCo is distributed, although the entire code base works with
single-precision as well. We may release the single-precision version in the future, but the
double-precision version will always be available. Thus it is safe to write user code assuming double precision.
However, our preference is to write code that works with either single or double precision. To this end we provide math
utility functions that are always defined with the correct floating-point type.

Note that changing ``mjUSESINGLE`` in ``mjtype.h`` will not change how the library was compiled, and instead will
result in numerous link errors. In general, the header files distributed with precompiled MuJoCo should never be
changed by the user.

.. code-block:: C

   // floating point data type and minval
   #ifndef mjUSESINGLE
     typedef double mjtNum;
     #define mjMINVAL    1E-15       // minimum value in any denominator
   #else
     typedef float mjtNum;
     #define mjMINVAL    1E-15f
   #endif


.. _mjtByte:

mjtByte
^^^^^^^

Byte type used to represent small integers and binary data.

.. code-block:: C

   typedef unsigned char mjtByte;


.. _mjtBool:

mjtBool
^^^^^^^

Boolean type used to represent true/false values.

.. code-block:: C

   #ifndef __cplusplus
     typedef _Bool mjtBool;
   #else
     typedef bool mjtBool;
   #endif


.. _mjtSize:

mjtSize
^^^^^^^

Size type used to represent buffer sizes.

.. code-block:: C

   typedef int64_t mjtSize;


.. _tyEnums:

Enum types
----------

All enum types use the ``mjt`` prefix.

.. _tyModelEnums:

Model
^^^^^

The enums below are defined in `mjtype.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h>`__.


.. _mjtDisableBit:

mjtDisableBit
~~~~~~~~~~~~~

Constants which are powers of 2. They are used as bitmasks for the field ``disableflags`` of :ref:`mjOption`.
At runtime this field is ``m->opt.disableflags``. The number of these constants is given by ``mjNDISABLE`` which is
also the length of the global string array :ref:`mjDISABLESTRING` with text descriptions of these flags.

.. mujoco-include:: mjtDisableBit


.. _mjtEnableBit:

mjtEnableBit
~~~~~~~~~~~~

Constants which are powers of 2. They are used as bitmasks for the field ``enableflags`` of :ref:`mjOption`.
At runtime this field is ``m->opt.enableflags``. The number of these constants is given by ``mjNENABLE`` which is also
the length of the global string array :ref:`mjENABLESTRING` with text descriptions of these flags.

.. mujoco-include:: mjtEnableBit


.. _mjtJoint:

mjtJoint
~~~~~~~~

Primitive joint types. These values are used in ``m->jnt_type``. The numbers in the comments indicate how many
positional coordinates each joint type has. Note that ball joints and rotational components of free joints are
represented as unit quaternions - which have 4 positional coordinates but 3 degrees of freedom each.

.. mujoco-include:: mjtJoint


.. _mjtGeom:

mjtGeom
~~~~~~~

Geometric types supported by MuJoCo. The first group are "official" geom types that can be used in the model. The
second group are geom types that cannot be used in the model but are used by the visualizer to add decorative
elements. These values are used in ``m->geom_type`` and ``m->site_type``.

.. mujoco-include:: mjtGeom


.. _mjtProjection:

mjtProjection
~~~~~~~~~~~~~

Type of camera projection. Used in ``m->cam_projection``.

.. mujoco-include:: mjtProjection


.. _mjtCamLight:

mjtCamLight
~~~~~~~~~~~

Dynamic modes for cameras and lights, specifying how the camera/light position and orientation are computed. These
values are used in ``m->cam_mode`` and ``m->light_mode``.

.. mujoco-include:: mjtCamLight


.. _mjtLightType:

mjtLightType
~~~~~~~~~~~~

The type of a light source describing how its position, orientation and other properties will interact with the
objects in the scene. These values are used in ``m->light_type``.

.. mujoco-include:: mjtLightType


.. _mjtTexture:

mjtTexture
~~~~~~~~~~

Texture types, specifying how the texture will be mapped. These values are used in ``m->tex_type``.

.. mujoco-include:: mjtTexture


.. _mjtTextureRole:

mjtTextureRole
~~~~~~~~~~~~~~

Texture roles, specifying how the renderer should interpret the texture.  Note that the MuJoCo built-in renderer only
uses RGB textures.  These values are used to store the texture index in the material's array ``m->mat_texid``.

.. mujoco-include:: mjtTextureRole


.. _mjtColorSpace:

mjtColorSpace
~~~~~~~~~~~~~

Type of color space encoding for textures.

.. mujoco-include:: mjtColorSpace


.. _mjtIntegrator:

mjtIntegrator
~~~~~~~~~~~~~

Numerical integrator types. These values are used in ``m->opt.integrator``.

.. mujoco-include:: mjtIntegrator

.. _mjtCone:

mjtCone
~~~~~~~

Available friction cone types. These values are used in ``m->opt.cone``.

.. mujoco-include:: mjtCone

.. _mjtJacobian:

mjtJacobian
~~~~~~~~~~~

Available Jacobian types. These values are used in ``m->opt.jacobian``.

.. mujoco-include:: mjtJacobian


.. _mjtSolver:

mjtSolver
~~~~~~~~~

Available constraint solver algorithms. These values are used in ``m->opt.solver``.

.. mujoco-include:: mjtSolver


.. _mjtEq:

mjtEq
~~~~~

Equality constraint types. These values are used in ``m->eq_type``.

.. mujoco-include:: mjtEq


.. _mjtTendon:

mjtTendon
~~~~~~~~~

Tendon types. These values are used in ``m->tendon_type``.

.. mujoco-include:: mjtTendon


.. _mjtWrap:

mjtWrap
~~~~~~~

Tendon wrapping object types. These values are used in ``m->wrap_type``.

.. mujoco-include:: mjtWrap


.. _mjtTrn:

mjtTrn
~~~~~~

Actuator transmission types. These values are used in ``m->actuator_trntype``.

.. mujoco-include:: mjtTrn


.. _mjtDyn:

mjtDyn
~~~~~~

Actuator dynamics types. These values are used in ``m->actuator_dyntype``.

.. mujoco-include:: mjtDyn


.. _mjtGain:

mjtGain
~~~~~~~

Actuator gain types. These values are used in ``m->actuator_gaintype``.

.. mujoco-include:: mjtGain


.. _mjtBias:

mjtBias
~~~~~~~

Actuator bias types. These values are used in ``m->actuator_biastype``.

.. mujoco-include:: mjtBias


.. _mjtCtrlChart:

mjtCtrlChart
~~~~~~~~~~~~

Orientation input charts of so3 actuators. These values are used in ``m->actuator_ctrlspec``.

.. mujoco-include:: mjtCtrlChart


.. _mjtCtrlInput:

mjtCtrlInput
~~~~~~~~~~~~

Input bitflags of actuators: the input subset of pid and dcmotor actuators, and the declared input of actuators with
fixed or affine gain. These values are used in ``m->actuator_ctrlspec``.

.. mujoco-include:: mjtCtrlInput


.. _mjtObj:

mjtObj
~~~~~~

MuJoCo object types. These are used, for example, in the support functions :ref:`mj_name2id` and
:ref:`mj_id2name` to convert between object names and integer ids.

.. mujoco-include:: mjtObj


.. _mjtSensor:

mjtSensor
~~~~~~~~~

Sensor types. These values are used in ``m->sensor_type``.

.. mujoco-include:: mjtSensor


.. _mjtStage:

mjtStage
~~~~~~~~

These are the compute stages for the skipstage parameters of :ref:`mj_forwardSkip` and
:ref:`mj_inverseSkip`.

.. mujoco-include:: mjtStage


.. _mjtDataType:

mjtDataType
~~~~~~~~~~~

These are the possible sensor data types, used in ``mjData.sensor_datatype``.

.. mujoco-include:: mjtDataType


.. _mjtConDataField:

mjtConDataField
~~~~~~~~~~~~~~~

Types of data fields returned by contact sensors.

.. mujoco-include:: mjtConDataField


.. _mjtRayDataField:

mjtRayDataField
~~~~~~~~~~~~~~~

Data fields returned by rangefinder sensors.

.. mujoco-include:: mjtRayDataField


.. _mjtCamOutBit:

mjtCamOutBit
~~~~~~~~~~~~

Camera output type bitflags. These are used in ``m->cam_output``.

.. mujoco-include:: mjtCamOutBit


.. _mjtSameFrame:

mjtSameFrame
~~~~~~~~~~~~

Types of frame alignment of elements with their parent bodies. Used as shortcuts during :ref:`mj_kinematics` in the
last argument to :ref:`mj_local2global`.

.. mujoco-include:: mjtSameFrame


.. _mjtSleepPolicy:

mjtSleepPolicy
~~~~~~~~~~~~~~

Sleep policy associated with a tree. The compiler automatically chooses between ``NEVER`` and ``ALLOWED``, but the user
can override this choice. Only the user can set the ``INIT`` policy (initialized as asleep).

.. mujoco-include:: mjtSleepPolicy


.. _mjtLRMode:

mjtLRMode
~~~~~~~~~

Mode for actuator length range computation. Used in ``mjLROpt.mode``.

.. mujoco-include:: mjtLRMode


.. _mjtFlexSelf:

mjtFlexSelf
~~~~~~~~~~~~

Types of flex self-collisions midphase.

.. mujoco-include:: mjtFlexSelf


.. _mjtSDFType:

mjtSDFType
~~~~~~~~~~~

Formulas used to combine SDFs when calling mjc_distance and mjc_gradient.

.. mujoco-include:: mjtSDFType


.. _tyDataEnums:

Data
^^^^

The enums below are defined in `mjtype.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h>`__.



.. _mjtState:

mjtState
~~~~~~~~

State component elements as integer bitflags and several convenient combinations of these flags. Used by
:ref:`mj_getState`, :ref:`mj_setState` and :ref:`mj_stateSize`.

.. mujoco-include:: mjtState


.. _mjtConstraint:

mjtConstraint
~~~~~~~~~~~~~

Constraint types. These values are not used in mjModel, but are used in the mjData field ``d->efc_type`` when the list
of active constraints is constructed at each simulation time step.

.. mujoco-include:: mjtConstraint


.. _mjtConstraintState:

mjtConstraintState
~~~~~~~~~~~~~~~~~~

These values are used by the solver internally to keep track of the constraint states.

.. mujoco-include:: mjtConstraintState


.. _mjtWarning:

mjtWarning
~~~~~~~~~~

Warning types. The number of warning types is given by ``mjNWARNING`` which is also the length of the array
``mjData.warning``.

.. mujoco-include:: mjtWarning


.. _mjtTimer:

mjtTimer
~~~~~~~~

Timer types. The number of timer types is given by ``mjNTIMER`` which is also the length of the array
``mjData.timer``, as well as the length of the string array :ref:`mjTIMERSTRING` with timer names.

.. mujoco-include:: mjtTimer


.. _mjtSleepState:

mjtSleepState
~~~~~~~~~~~~~

Sleep state of an object.

.. mujoco-include:: mjtSleepState


.. _tyLogEnums:

Logging
~~~~~~~

.. _mjtLogLevel:

mjtLogLevel
"""""""""""

Log message severity level.

.. mujoco-include:: mjtLogLevel


.. _mjtLogTopic:

mjtLogTopic
"""""""""""

Topic identifiers for informational messages. Used with :ref:`mju_info` for topic-based filtering.
Topic 0 (``mjTOPIC_NONE``) always passes through the default handler's filter. Other topics must be enabled in
the :ref:`mjLogConfig` bitmask. Since topics are 1-indexed, the bitmask for topic ``t`` is ``(1 << (t - 1))``.
The number of filterable topics is given by ``mjNTOPIC``, which is also the length of the string array
:ref:`mjTOPICSTRING`.

.. mujoco-include:: mjtLogTopic


.. _tyVisEnums:

Visualization
^^^^^^^^^^^^^

The enums below are defined in `mjvisualize.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjvisualize.h>`__.


.. _mjtCatBit:

mjtCatBit
~~~~~~~~~

These are the available categories of geoms in the abstract visualizer. The bitmask can be used in the function
:ref:`mjr_render` to specify which categories should be rendered.

.. mujoco-include:: mjtCatBit


.. _mjtMouse:

mjtMouse
~~~~~~~~

These are the mouse actions that the abstract visualizer recognizes. It is up to the user to intercept mouse events
and translate them into these actions, as illustrated in :ref:`simulate.cc <saSimulate>`.

.. mujoco-include:: mjtMouse


.. _mjtPertBit:

mjtPertBit
~~~~~~~~~~

These bitmasks enable the translational and rotational components of the mouse perturbation. For the regular mouse,
only one can be enabled at a time. For the 3D mouse (SpaceNavigator) both can be enabled simultaneously. They are used
in ``mjvPerturb.active``.

.. mujoco-include:: mjtPertBit


.. _mjtCamera:

mjtCamera
~~~~~~~~~

These are the possible camera types, used in ``mjvCamera.type``.

.. mujoco-include:: mjtCamera


.. _mjtLabel:

mjtLabel
~~~~~~~~

These are the abstract visualization elements that can have text labels. Used in ``mjvOption.label``.
The corresponding string array is :ref:`mjLABELSTRING`.

.. mujoco-include:: mjtLabel


.. _mjtFrame:

mjtFrame
~~~~~~~~

These are the MuJoCo objects whose spatial frames can be rendered. Used in ``mjvOption.frame``.
The corresponding string array is :ref:`mjFRAMESTRING`.

.. mujoco-include:: mjtFrame


.. _mjtVisFlag:

mjtVisFlag
~~~~~~~~~~

These are indices in the array ``mjvOption.flags``, whose elements enable/disable the visualization of the
corresponding model or decoration element. The corresponding string array is :ref:`mjVISSTRING`.

.. mujoco-include:: mjtVisFlag


.. _mjtRndFlag:

mjtRndFlag
~~~~~~~~~~

These are indices in the array ``mjvScene.flags``, whose elements enable/disable OpenGL rendering effects.
The corresponding string array is :ref:`mjRNDSTRING`.

.. mujoco-include:: mjtRndFlag


.. _mjtStereo:

mjtStereo
~~~~~~~~~

These are the possible stereo rendering types. They are used in ``mjvScene.stereo``.

.. mujoco-include:: mjtStereo



.. _tyRenderEnums:

Rendering
^^^^^^^^^

The enums below are defined in `mjrender.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrender.h>`__.


.. _mjtGridPos:

mjtGridPos
~~~~~~~~~~

These are the possible grid positions for text overlays. They are used as an argument to the function
:ref:`mjr_overlay`.

.. mujoco-include:: mjtGridPos


.. _mjtFramebuffer:

mjtFramebuffer
~~~~~~~~~~~~~~

These are the possible framebuffers. They are used as an argument to the function :ref:`mjr_setBuffer`.

.. mujoco-include:: mjtFramebuffer


.. _mjtDepthMap:

mjtDepthMap
~~~~~~~~~~~

These are the depth mapping options. They are used as a value for the ``readPixelDepth`` attribute of the
:ref:`mjrContext` struct, to control how the depth returned by :ref:`mjr_readPixels` is mapped from
``znear`` to ``zfar``.

.. mujoco-include:: mjtDepthMap


.. _mjtFontScale:

mjtFontScale
~~~~~~~~~~~~

These are the possible font sizes. The fonts are predefined bitmaps stored in the dynamic library at three different
sizes.

.. mujoco-include:: mjtFontScale


.. _mjtFont:

mjtFont
~~~~~~~

These are the possible font types.

.. mujoco-include:: mjtFont

.. _mjrPixelFormat:

mjrPixelFormat
~~~~~~~~~~~~~~

These are the possible values:

.. mujoco-include:: mjrPixelFormat

.. _mjrVertexAttributeUsage:

mjrVertexAttributeUsage
~~~~~~~~~~~~~~~~~~~~~~~

These are the possible values:

.. mujoco-include:: mjrVertexAttributeUsage

.. _mjrVertexAttributeType:

mjrVertexAttributeType
~~~~~~~~~~~~~~~~~~~~~~

These are the possible values:

.. mujoco-include:: mjrVertexAttributeType

.. _mjrIndexType:

mjrIndexType
~~~~~~~~~~~~

These are the possible values:

.. mujoco-include:: mjrIndexType

.. _mjrMeshPrimitiveType:

mjrMeshPrimitiveType
~~~~~~~~~~~~~~~~~~~~

There are the possible values:

.. mujoco-include:: mjrMeshPrimitiveType


.. _tyFilamentRenderEnums:

Filament Rendering
^^^^^^^^^^^^^^^^^^

The enums below are defined in `mjrfilament.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrfilament.h>`__.


.. _mjrGraphicsApi:

mjrGraphicsApi
~~~~~~~~~~~~~~

The underlying graphics API to use for Filament rendering.

.. mujoco-include:: mjrGraphicsApi


.. _mjrDrawMode:

mjrDrawMode
~~~~~~~~~~~

High-level control for how to draw objects in the scene for Filament rendering.

.. mujoco-include:: mjrDrawMode


.. _tyUIEnums:

User Interface
^^^^^^^^^^^^^^

The enums below are defined in `mjui.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjui.h>`__.


.. _mjtButton:

mjtButton
~~~~~~~~~

Mouse button IDs used in the UI framework.

.. mujoco-include:: mjtButton


.. _mjtEvent:

mjtEvent
~~~~~~~~

Event types used in the UI framework.

.. mujoco-include:: mjtEvent


.. _mjtItem:

mjtItem
~~~~~~~

Item types used in the UI framework.

.. mujoco-include:: mjtItem


.. _mjtSection:

mjtSection
~~~~~~~~~~

State of a UI section.

.. mujoco-include:: mjtSection



.. _tySpecEnums:

Spec
^^^^

The enums below are defined in `mjspec.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjspec.h>`__.

.. _mjtGeomInertia:

mjtGeomInertia
~~~~~~~~~~~~~~

Type of inertia inference.

.. mujoco-include:: mjtGeomInertia

.. _mjtBuiltin:

mjtBuiltin
~~~~~~~~~~

Type of built-in procedural texture.

.. mujoco-include:: mjtBuiltin

.. _mjtMark:

mjtMark
~~~~~~~

Mark type for procedural textures.

.. mujoco-include:: mjtMark

.. _mjtLimited:

mjtLimited
~~~~~~~~~~

Type of limit specification.

.. mujoco-include:: mjtLimited

.. _mjtAlignFree:

mjtAlignFree
~~~~~~~~~~~~

Whether to align free joints with the inertial frame.

.. mujoco-include:: mjtAlignFree

.. _mjtPassthrough:

mjtPassthrough
~~~~~~~~~~~~~~

Which assets the compiler leaves :ref:`by reference<CAssetReference>` instead of loading into :ref:`mjModel`, a
bitmask; see :ref:`compiler/passthrough<compiler-passthrough>`.

.. mujoco-include:: mjtPassthrough

.. _mjtEmbed:

mjtEmbed
~~~~~~~~

Whether an asset's payload is in the model or the asset is :ref:`by reference<CAssetReference>`; see
:ref:`texture/embed<asset-texture-embed>`.

.. mujoco-include:: mjtEmbed

.. _mjtInertiaFromGeom:

mjtInertiaFromGeom
~~~~~~~~~~~~~~~~~~

Whether to infer body inertias from child geoms.

.. mujoco-include:: mjtInertiaFromGeom

.. _mjtOrientation:

mjtOrientation
~~~~~~~~~~~~~~

Type of orientation specifier.

.. mujoco-include:: mjtOrientation

.. _mjtMeshInertia:

mjtMeshInertia
~~~~~~~~~~~~~~

Type of mesh inertia computation.

.. mujoco-include:: mjtMeshInertia

.. _mjtMeshBuiltin:

mjtMeshBuiltin
~~~~~~~~~~~~~~

Type of built-in procedural mesh.

.. mujoco-include:: mjtMeshBuiltin

.. _mjtFlexSection:

mjtFlexSection
~~~~~~~~~~~~~~

Cross-section of 1D flex :ref:`elasticity<flex-elasticity>`, see :ref:`section<flex-elasticity-section>`.

.. mujoco-include:: mjtFlexSection

.. _mjtConflict:

mjtConflict
~~~~~~~~~~~

Conflict resolution mode for attach.

.. mujoco-include:: mjtConflict

.. _mjtScaling:

mjtScaling
~~~~~~~~~~

Physical policy of a frame's :ref:`scale<frame-scale>`, see :ref:`scaling<frame-scaling>`.

.. mujoco-include:: mjtScaling

.. _mjtActuator:

mjtActuator
~~~~~~~~~~~

The :ref:`actuator element<actuator>` an actuator is written with: :ref:`general<actuator-general>` or one of its
:ref:`shortcuts<CActShortcuts>`. Set by the ``mjs_setTo*`` functions and by the parser; shortcut parameters are inherited
only from a default written with the same shortcut.

.. mujoco-include:: mjtActuator


.. _mjtCTimer:

mjtCTimer
~~~~~~~~~

Compiler timing categories, used in :ref:`mjs_getTimer`. Top-level timers (``TOTAL``, ``ASSETS``) measure wall-clock
time. Asset sub-timers measure CPU time summed across all assets; with multi-threaded compilation their sum can exceed
the ``ASSETS`` wall-clock time.

.. mujoco-include:: mjtCTimer


.. _tyPluginEnums:

Plugins
^^^^^^^

The enums below are defined in `mjplugin.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjplugin.h>`__.
See :ref:`exPlugin` for details.


.. _mjtPluginCapabilityBit:

mjtPluginCapabilityBit
~~~~~~~~~~~~~~~~~~~~~~

Capabilities declared by an engine plugin.

.. mujoco-include:: mjtPluginCapabilityBit



.. _tyStructure:

Struct types
------------

The three central struct types for physics simulation are :ref:`mjModel`, :ref:`mjOption` (embedded in :ref:`mjModel`)
and :ref:`mjData`. An introductory discussion of these structures can be found in the :ref:`Overview<ModelAndData>`.


.. _mjModel:

mjModel
^^^^^^^

This is the main data structure holding the MuJoCo model. It is treated as constant by the simulator. Some specific
details regarding datastructures in :ref:`mjModel` can be found below in :ref:`tyNotes`.

.. mujoco-include:: mjModel



.. _mjOption:

mjOption
^^^^^^^^

This is the data structure with simulation options. It corresponds to the MJCF element
:ref:`option <option>`. One instance of it is embedded in mjModel.

.. mujoco-include:: mjOption


.. _mjData:

mjData
^^^^^^

This is the main data structure holding the simulation state. It is the workspace where all functions read their
modifiable inputs and write their outputs.

.. mujoco-include:: mjData



.. _tyAuxStructure:

Auxiliary
^^^^^^^^^

These struct types are used in the engine and their names are prefixed with ``mj``. :ref:`mjVisual`
and :ref:`mjStatistic` are embedded in :ref:`mjModel`, :ref:`mjContact` is embedded in :ref:`mjData`, and :ref:`mjVFS`
is a library-level struct used for loading assets.


.. _mjVisual:

mjVisual
~~~~~~~~

This is the data structure with abstract visualization options. It corresponds to the MJCF element
:ref:`visual <visual>`. One instance of it is embedded in mjModel.

.. mujoco-include:: mjVisual


.. _mjStatistic:

mjStatistic
~~~~~~~~~~~

This is the data structure with model statistics precomputed by the compiler or set by the user. It corresponds to the
MJCF element :ref:`statistic <statistic>`. One instance of it is embedded in mjModel.

.. mujoco-include:: mjStatistic


.. _mjPreContact:

mjPreContact
~~~~~~~~~~~~

This is the data structure holding information about one contact filled out by a narrowphase collision detector.

.. mujoco-include:: mjPreContact

.. _mjContact:

mjContact
~~~~~~~~~

This is the data structure holding information about one contact. ``mjData.contact`` is a preallocated array of
mjContact data structures, populated at runtime with the contacts found by the collision detector. Additional contact
information is then filled-in by the simulator.

.. mujoco-include:: mjContact


.. _mjResource:

mjResource
~~~~~~~~~~

A resource is an abstraction of a file in a filesystem. The name field is the unique name of the resource while the
other fields are populated by a :ref:`resource provider <exProvider>`.

.. mujoco-include:: mjResource


.. _mjVFS:

mjVFS
~~~~~

This is the data structure of the virtual file system. It can only be constructed programmatically, and does not
have an analog in MJCF.

.. mujoco-include:: mjVFS


.. _mjLROpt:

mjLROpt
~~~~~~~

Options for configuring the automatic :ref:`actuator length-range computation<CLengthRange>`.

.. mujoco-include:: mjLROpt

.. _mjCache:

mjCache
~~~~~~~

Asset cache used by the compiler to avoid repeated slow recompilation. See :ref:`Asset cache<Assetcache>`.

.. mujoco-include:: mjCache


.. _tyLogStructure:

Logging
^^^^^^^

.. _mjLogMessage:

mjLogMessage
~~~~~~~~~~~~

Structured log message passed to :ref:`mjfLogHandler` callbacks. Contains the severity level, optional topic for
info messages, a one-line subject, an optional multi-line body, and optional source location (function name, file
name, line number).

.. mujoco-include:: mjLogMessage


.. _mjLogConfig:

mjLogConfig
~~~~~~~~~~~

Configuration for the default log handler. Controls whether messages are printed to the console and/or written to
a log file (default: ``MUJOCO_LOG.TXT``). The ``logto_file`` field enables file logging, while ``logfile`` specifies
the file path. The ``topics`` field is a bitmask of :ref:`mjtLogTopic` values: bit ``(topic - 1)`` enables
that topic. Topic 0 (``mjTOPIC_NONE``) always passes through.

.. mujoco-include:: mjLogConfig


.. _tyStatStructure:

Sim statistics
^^^^^^^^^^^^^^

These structs are all embedded in :ref:`mjData`, and collect simulation-related statistics.


.. _mjWarningStat:

mjWarningStat
~~~~~~~~~~~~~

This is the data structure holding information about one warning type. ``mjData.warning`` is a preallocated array of
mjWarningStat data structures, one for each warning type.

.. mujoco-include:: mjWarningStat


.. _mjTimerStat:

mjTimerStat
~~~~~~~~~~~

This is the data structure holding information about one timer. ``mjData.timer`` is a preallocated array of
mjTimerStat data structures, one for each timer type.

.. mujoco-include:: mjTimerStat


.. _mjSolverStat:

mjSolverStat
~~~~~~~~~~~~

This is the data structure holding information about one solver iteration. ``mjData.solver`` is a preallocated array
of mjSolverStat data structures, one for each iteration of the solver, up to a maximum of mjNSOLVER. The actual number
of solver iterations is given by ``mjData.solver_niter``.

.. mujoco-include:: mjSolverStat



.. _tyVisStructure:

Visualisation
^^^^^^^^^^^^^

The names of these struct types are prefixed with ``mjv``.

.. _mjvPerturb:

mjvPerturb
~~~~~~~~~~

This is the data structure holding information about mouse perturbations.

.. mujoco-include:: mjvPerturb


.. _mjvCamera:

mjvCamera
~~~~~~~~~

This is the data structure describing one abstract camera.

.. mujoco-include:: mjvCamera


.. _mjvGLCamera:

mjvGLCamera
~~~~~~~~~~~

This is the data structure describing one OpenGL camera.

.. mujoco-include:: mjvGLCamera


.. _mjvGeom:

mjvGeom
~~~~~~~

This is the data structure describing one abstract visualization geom - which could correspond to a model geom or to a
decoration element constructed by the visualizer.

.. mujoco-include:: mjvGeom


.. _mjvLight:

mjvLight
~~~~~~~~

This is the data structure describing one OpenGL light.

.. mujoco-include:: mjvLight


.. _mjvOption:

mjvOption
~~~~~~~~~

This structure contains options that enable and disable the visualization of various elements.

.. mujoco-include:: mjvOption


.. _mjvScene:

mjvScene
~~~~~~~~

This structure contains everything needed to render the 3D scene in OpenGL.

.. mujoco-include:: mjvScene


.. _mjvFigure:

mjvFigure
~~~~~~~~~

This structure contains everything needed to render a 2D plot in OpenGL. The buffers for line points etc. are
preallocated, and the user has to populate them before calling the function :ref:`mjr_figure` with this
data structure as an argument.

.. mujoco-include:: mjvFigure


.. _tyRenderStructure:

Rendering
^^^^^^^^^

The names of these struct types are prefixed with ``mjr``.

.. _mjrRect:

mjrRect
~~~~~~~

This structure specifies a rectangle.

.. mujoco-include:: mjrRect


.. _mjrVertexAttribute:

mjrVertexAttribute
~~~~~~~~~~~~~~~~~~

This structure specifies the attributes for a single vertex.

.. mujoco-include:: mjrVertexAttribute



.. _mjrRendererInfo:

mjrRendererInfo
~~~~~~~~~~~~~~~

This structure contains information about the available renderer and its current context.

.. mujoco-include:: mjrRendererInfo


.. _mjrContext:

mjrContext
~~~~~~~~~~

This structure contains the custom OpenGL rendering context, with the ids of all OpenGL resources uploaded to the GPU.

.. mujoco-include:: mjrContext


.. _tyFilamentRenderStructure:

Filament Rendering
^^^^^^^^^^^^^^^^^^

The names of these struct types are prefixed with ``mjrf``. They are defined in
`mjrfilament.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrfilament.h>`__.

There are seven key types defined by this API: :ref:`mjrfContext<mjrfContext>`, :ref:`mjrfTexture<mjrfTexture>`,
:ref:`mjrfMesh<mjrfMesh>`, :ref:`mjrfLight<mjrfLight>`, :ref:`mjrfRenderable<mjrfRenderable>`,
:ref:`mjrfScene<mjrfScene>`, and :ref:`mjrfRenderTarget<mjrfRenderTarget>`.

Each object is created using a `create` function and destroyed using a `destroy` function, e.g. `mjrf_createTexture` and
`mjrf_destroyTexture`. All objects require a :ref:`Context<mjrfContext>` in order to be created (with the exception of
the :ref:`Context<mjrfContext>` object itself). Additionally, the `create` functions accept a pointer to a configuration
struct (e.g. `mjrTextureConfig`) which describes the parameters for the object to be created. Each of these structs has
a corresponding `default` function (e.g. `mjrf_defaultTextureConfig`) which can be used to initialize the struct to
default values. The default values are assumed to be `0` or `NULL` unless otherwise specified.


.. _mjrfContext:

mjrfContext
~~~~~~~~~~~

The Context is the main entry point for the filament rendering library. It manages all the core filament objects that
are responsible for the rendering of an image. All other objects (e.g. Textures, Meshes, Scenes, etc.) need a Context in
order to be created.

Otherwise, the main function to use with the Context is :ref:`mjrf_render()<mjrf_render>` which performs the actual
rendering of an image.

Filament uses a separate thread for rendering. However, despite that, this API is not thread-safe; calls are expected to
be made from a single thread. Due to the asynchronous nature of filament, some APIs provide handles or callbacks to
signal when an operation is complete. (Note: for WASM builds, filament does not use a separate thread.)

There are two key differences between the :ref:`mjrfContext<mjrfContext>` and the classic :ref:`mjrContext<mjrContext>`.
Firstly, the filament context will manage the underlying graphics context itself. This means users do not need to
initialize EGL or similar libraries beforehand. Secondly, the filament context is independent of a MuJoCo model. That
means you can use a single :ref:`mjrfContext<mjrfContext>` instance to render images for multiple models.

.. _mjrfContextConfig:

mjrfContextConfig
~~~~~~~~~~~~~~~~~

Parameters for creating :ref:`filament graphics context<mjrfContext>`.

.. mujoco-include:: mjrfContextConfig


.. _mjrfTexture:

mjrfTexture
~~~~~~~~~~~

A texture is a 2D or 3D (cubemap) image that adds visual detail to a rendered model, such as color or bumpiness, without
increasing geometric complexity. A texture is simply a memory buffer holds pixel data, as well as metadata such as the
dimensions of the image or the format of the pixels (e.g. 8-bit RGB).

.. _mjrfTextureConfig:

mjrfTextureConfig
~~~~~~~~~~~~~~~~~

Parameters for creating a :ref:`texture<mjrfTexture>`.

.. mujoco-include:: mjrfTextureConfig


.. _mjrfTextureData:

mjrfTextureData
~~~~~~~~~~~~~~~

Binary data payload for a :ref:`texture<mjrfTexture>`.

.. mujoco-include:: mjrfTextureData


.. _mjrfMesh:

mjrfMesh
~~~~~~~~

A mesh describes the surface geometry of an object to be rendered. It is defined as a collection of vertices (i.e. a
VertexBuffer), a set of indices (i.e. an IndexBuffer) that describes the order in which the vertices should be
processed, and a primitive type that defined how the vertices are to be interpreted (e.g. triangles, lines, etc.) when
rendering the surface.

Filament does not directly support normals. Instead, it encodes the normal, tangent, and bitangent into a 4-component
quaternion describing the "orientation" of the vertex. Ideally, you should preprocess your assets to generate this data
offline, but we will compute it on the fly if needed (at a performance cost).

Vertex data may or may not be interleaved. Interleaved data assumes that the attributes are packed in the order
specified in the attributes array, with no padding in-between. Additionally, the `data` pointer for each attribute is
assumed to point to the first element of that type. For non-interleaved data, each attribute is assumed to be stored in
a separate array.

Additionally, the bounds of the mesh should be computed in order to allow the filament renderer to perform frustum-based
culling. Alternatively, the bounds can be computed at runtime (though there is a small performance cost). If no bounds
are provided (or calculated), then frustum culling will not be performed.


.. _mjrfMeshConfig:

mjrfMeshConfig
~~~~~~~~~~~~~~~~~

Parameters for creating a :ref:`mesh<mjrfMesh>`.

.. mujoco-include:: mjrfMeshConfig


.. _mjrfMeshData:

mjrfMeshData
~~~~~~~~~~~~

Binary data for a :ref:`mesh<mjrfMesh>`.

.. mujoco-include:: mjrfMeshData


.. _mjrfScene:

mjrfScene
~~~~~~~~~

A Scene is a collection of :ref:`Lights<mjrfLight>` and :ref:`Renderables<mjrfRenderable>` that describes what is to be
rendered.


.. _mjrfSceneParams:

mjrfSceneParams
~~~~~~~~~~~~~~~

Parameters for creating a :ref:`scene<mjrfScene>`.

.. mujoco-include: mjrfSceneParams


.. _mjrfLight:

mjrfLight
~~~~~~~~~

A light is a source of illumination. (Without lights, a rendered image will be completely black.) There are several
different types of lights such as directional, spot, point, and image lights.

The primary light in a scene is the image light (also sometimes known as the environment light). This is a light that
"surrounds" the entire scene and is defined as a 3D texture. Each "pixel" of the cubemap is interpreted as the color of
projected into the scene from a particular direction.

The texture used for image-based lighting can be generated using filament's `cmgen` tool. The tool should be configured
to output a KTX file from your source image. This tool calculates additional data (i.e. the spherical harmonics) and
encodes that information into the KTX file.

Directional lights are the next most common type of light and is usually used to simulate the sun; a uniformly colored
light that is emitted in a single direction.

Filament only supports a single image and directional light. You can define as many point or spot lights as you want.
Each light source (except image based lights) may or may not cast shadows. Each shadow-casting light incurs a
performance cost.


.. _mjrfLightParams:

mjrfLightParams
~~~~~~~~~~~~~~~

Parameters for creating a :ref:`light<mjrfLight>`.

.. mujoco-include:: mjrfLightParams


.. _mjrfRenderable:

mjrfRenderable
~~~~~~~~~~~~~~

A renderable is a single object that is to be drawn. It is defined as a combination of a :ref:`mjrfMesh<mjrfMesh>`
(i.e. the shape or surface geometry, as described above) and a :ref:`mjrfMaterial<mjrfMaterial>` (i.e. a description of
how the surface interacts with lights to product the final visual appearance).


.. _mjrfRenderableParams:

mjrfRenderableParams
~~~~~~~~~~~~~~~~~~~~

Parameters for creating a :ref:`renderable<mjrfRenderable>`.

.. mujoco-include:: mjrfRenderableParams


.. _mjrfMaterial:

mjrfMaterial
~~~~~~~~~~~~

Materials describe the properties of the surface of a renderable, effectively dictating how the surface interacts with
lights to produce a final pixel color in the output image. Different lighting models will be applied to the surface
depending on the values of the material properties. There there are three lighting models currently supported:

1. Metallic-roughness (PBR): this is the preferred model for rendering models based standard
metallic-roughness workflows.

2. Specular-glossiness (non-PBR): this is a legacy model designed to be compatible with classic
:ref:`mjr<ClassicRendering>` renderer, though it is not 100% identical.

3. Unlit: this model ignores lighting and is used for rendering UX or decorative elements like
contact forces and labels.

.. mujoco-include:: mjrfMaterial


.. _mjrfRenderTarget:

mjrfRenderTarget
~~~~~~~~~~~~~~~~

A RenderTarget is a memory buffer that holds the results of a rendering operation. (This is an alternative to rendering
directly to the screen.)


.. _mjrfRenderTargetConfig:

mjrfRenderTargetConfig
~~~~~~~~~~~~~~~~~~~~~~

Parameters for creating a :ref:`render target<mjrfRenderTarget>`.

.. mujoco-include:: mjrfRenderTargetConfig


.. _mjrfRenderRequest:

mjrfRenderRequest
~~~~~~~~~~~~~~~~~

A single rendering operation.

.. mujoco-include:: mjrfRenderRequest


.. _mjrfReadRequest:

mjrfReadRequest
~~~~~~~~~~~~~~~~~~~~~

A single pixel read operation.

.. mujoco-include:: mjrfReadRequest


.. _mjrfFrameStats:

mjrfFrameStats
~~~~~~~~~~~~~~

Information about a single frame of rendering.

.. mujoco-include:: mjrfFrameStats


.. _tyUIStructure:

User Interface
^^^^^^^^^^^^^^

For a high-level description of the UI framework, see :ref:`UI`.
The names of these struct types are prefixed with ``mjui``, except for the main :ref:`mjUI` struct itself.


.. _mjuiState:

mjuiState
~~~~~~~~~

This C struct represents the global state of the window, keyboard and mouse, input event descriptors, and all window
rectangles (including the visible UI rectangles). There is only one ``mjuiState`` per application, even if there are
multiple UIs. This struct would normally be defined as a global variable.

.. mujoco-include:: mjuiState


.. _mjuiThemeSpacing:

mjuiThemeSpacing
~~~~~~~~~~~~~~~~

This structure defines the spacing of UI items in the theme.

.. mujoco-include:: mjuiThemeSpacing


.. _mjuiThemeColor:

mjuiThemeColor
~~~~~~~~~~~~~~

This structure defines the colors of UI items in the theme.

.. mujoco-include:: mjuiThemeColor


.. _mjuiItem:

mjuiItem
~~~~~~~~

This structure defines one UI item.

.. mujoco-include:: mjuiItem


.. _mjuiSection:

mjuiSection
~~~~~~~~~~~

This structure defines one section of the UI.

.. mujoco-include:: mjuiSection


.. _mjuiDef:

mjuiDef
~~~~~~~

This structure defines one entry in the definition table used for simplified UI construction. It contains everything
needed to define one UI item. Some translation is performed by the helper functions, so that multiple mjuiDefs can be
defined as a static table.

.. mujoco-include:: mjuiDef


.. _mjUI:

mjUI
~~~~

This C struct represents an entire UI. The same application could have multiple UIs, for example on the left and the
right of the window. This would normally be defined as a global variable. As explained earlier, it contains static
allocation for a maximum number of supported UI sections (:ref:`mjuiSection<mjuiSection>`) each with a maximum number
of supported items (:ref:`mjuiItem<mjuiItem>`). It also contains the color and spacing themes, enable/disable
callback, virtual window descriptor, text edit state, mouse focus. Some of these fields are set only once when the UI
is initialized, others change at runtime.

.. mujoco-include:: mjUI



.. _tySpecStructure:

Model Editing
^^^^^^^^^^^^^

The structs below are defined in
`mjspec.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjspec.h>`__ and, with the exception of
the top level :ref:`mjSpec` struct, begin with the ``mjs`` prefix. For more details, see the :doc:`Model Editing
<../programming/modeledit>` chapter.

.. _mjSpec:

mjSpec
~~~~~~

Model specification.

.. mujoco-include:: mjSpec


.. _mjsElement:

mjsElement
~~~~~~~~~~

Special type corresponding to any element. This struct is the first member of all other elements; in the low-level C++
implementation, it is not included as a member but via class inheritance. Inclusion via inheritance allows the compiler
to ``static_cast`` an ``mjsElement`` to the correct C++ object class. Unlike all other attributes of the structs below,
which are user-settable by design, modifying the contents of an ``mjsElement`` is not allowed and leads to undefined
behavior.

.. mujoco-include:: mjsElement


.. _mjsCompiler:

mjsCompiler
~~~~~~~~~~~

Compiler options.

.. mujoco-include:: mjsCompiler


.. _mjsAuthored:

mjsAuthored
~~~~~~~~~~~

Authored tracking bitmasks for ``mjModel`` structs.

.. mujoco-include:: mjsAuthored


.. _mjsBody:

mjsBody
~~~~~~~

Body specification.

.. mujoco-include:: mjsBody


.. _mjsFrame:

mjsFrame
~~~~~~~~

Frame specification.

.. mujoco-include:: mjsFrame


.. _mjsJoint:

mjsJoint
~~~~~~~~

Joint specification.

.. mujoco-include:: mjsJoint


.. _mjsGeom:

mjsGeom
~~~~~~~

Geom specification.

.. mujoco-include:: mjsGeom


.. _mjsSite:

mjsSite
~~~~~~~

Site specification.

.. mujoco-include:: mjsSite


.. _mjsCamera:

mjsCamera
~~~~~~~~~

Camera specification.

.. mujoco-include:: mjsCamera


.. _mjsLight:

mjsLight
~~~~~~~~

Light specification.

.. mujoco-include:: mjsLight


.. _mjsFlex:

mjsFlex
~~~~~~~

Flex specification.

The experimental integer ``elastic3d`` selects the material for non-interpolated 3D flexes: ``0`` (the default) is
Saint Venant-Kirchhoff (StVK) elasticity, and ``1`` is simplified Stable Neo-Hookean (SNH) elasticity. Other values are rejected.
SNH requires the :ref:`discrete integrator<option-integrator>`. The setting is available only through :ref:`mjSpec`,
with no MJCF attribute. Specs with SNH enabled cannot be saved to MJCF; compiled models can be saved to MJB.
SNH projects each element's material Hessian to positive semidefiniteness before mapping it to vertex coordinates.
The discrete solver and Rayleigh damping reuse this projected operator. Energy and elastic forces are unchanged and
remain defined through inversion. Material damping has non-positive instantaneous power, but projection does not
guarantee inversion-free motion or stability at arbitrary timesteps.

.. mujoco-include:: mjsFlex


.. _mjsMesh:

mjsMesh
~~~~~~~

Mesh specification.

.. mujoco-include:: mjsMesh


.. _mjsHField:

mjsHField
~~~~~~~~~

Height field specification.

.. mujoco-include:: mjsHField


.. _mjsSkin:

mjsSkin
~~~~~~~

Skin specification.

.. mujoco-include:: mjsSkin


.. _mjsTexture:

mjsTexture
~~~~~~~~~~

Texture specification.

.. mujoco-include:: mjsTexture


.. _mjsMaterial:

mjsMaterial
~~~~~~~~~~~

Material specification.

.. mujoco-include:: mjsMaterial


.. _mjsPair:

mjsPair
~~~~~~~

Pair specification.

.. mujoco-include:: mjsPair


.. _mjsExclude:

mjsExclude
~~~~~~~~~~

Exclude specification.

.. mujoco-include:: mjsExclude


.. _mjsEquality:

mjsEquality
~~~~~~~~~~~

Equality specification.

.. mujoco-include:: mjsEquality


.. _mjsTendon:

mjsTendon
~~~~~~~~~

Tendon specification.

.. mujoco-include:: mjsTendon


.. _mjsWrap:

mjsWrap
~~~~~~~

Wrapping object specification.

.. mujoco-include:: mjsWrap


.. _mjsActuator:

mjsActuator
~~~~~~~~~~~

Actuator specification.

.. mujoco-include:: mjsActuator


.. _mjsSensor:

mjsSensor
~~~~~~~~~

Sensor specification.

.. mujoco-include:: mjsSensor


.. _mjsNumeric:

mjsNumeric
~~~~~~~~~~

Custom numeric field specification.

.. mujoco-include:: mjsNumeric


.. _mjsText:

mjsText
~~~~~~~

Custom text specification.

.. mujoco-include:: mjsText


.. _mjsTuple:

mjsTuple
~~~~~~~~

Tuple specification.

.. mujoco-include:: mjsTuple


.. _mjsKey:

mjsKey
~~~~~~

Keyframe specification.

.. mujoco-include:: mjsKey


.. _mjsDefault:

mjsDefault
~~~~~~~~~~

Default specification.

.. mujoco-include:: mjsDefault


.. _mjsPlugin:

mjsPlugin
~~~~~~~~~

Plugin specification.

.. mujoco-include:: mjsPlugin


.. _mjsOrientation:

mjsOrientation
~~~~~~~~~~~~~~

Alternative orientation specifiers.

.. mujoco-include:: mjsOrientation


.. _ArrayHandles:

.. _mjByteVec:

.. _mjString:

.. _mjStringVec:

.. _mjIntVec:

.. _mjIntVecVec:

.. _mjFloatVec:

.. _mjFloatVecVec:

.. _mjDoubleVec:

Array handles
~~~~~~~~~~~~~

C handles for C++ strings and vector types. When using from C, use the provided :ref:`getters<AttributeGetters>` and
:ref:`setters<AttributeSetters>`.

.. code-block:: C++

   #ifdef __cplusplus
     // C++: defined to be compatible with corresponding std types
     using mjString      = std::string;
     using mjStringVec   = std::vector<std::string>;
     using mjIntVec      = std::vector<int>;
     using mjIntVecVec   = std::vector<std::vector<int>>;
     using mjFloatVec    = std::vector<float>;
     using mjFloatVecVec = std::vector<std::vector<float>>;
     using mjDoubleVec   = std::vector<double>;
     using mjByteVec     = std::vector<std::byte>;
   #else
     // C: opaque types
     typedef void mjString;
     typedef void mjStringVec;
     typedef void mjIntVec;
     typedef void mjIntVecVec;
     typedef void mjFloatVec;
     typedef void mjFloatVecVec;
     typedef void mjDoubleVec;
     typedef void mjByteVec;
   #endif


.. _tyPluginStructure:

Plugins
^^^^^^^

The names of these struct types are prefixed with ``mjp``. See :ref:`exPlugin` for more details.


.. _mjpPlugin:

mjpPlugin
~~~~~~~~~

This structure contains the definition of a single engine plugin. It mostly contains a set of callbacks, which are
triggered by the compiler and the engine during various phases of the computation pipeline.

.. mujoco-include:: mjpPlugin

.. _mjSDF:

mjSDF
~~~~~

Data structure used by the :ref:`Signed Distance Functions<Signeddistancefunction>` API for computing distances and
gradients between SDF geoms.

.. mujoco-include:: mjSDF

.. _mjpResourceProvider:

mjpResourceProvider
~~~~~~~~~~~~~~~~~~~

This data structure contains the definition of a :ref:`resource provider <exProvider>`. It contains a set of callbacks
used for opening and reading resources.

.. mujoco-include:: mjpResourceProvider

.. _mjpDecoder:

mjpDecoder
~~~~~~~~~~~~~~~~~~~

This data structure defines a decoder. It contains a set of callbacks used for decoding :ref:`mjResource`
into :ref:`mjSpec`.

.. mujoco-include:: mjpDecoder

.. _mjpEncoder:

mjpEncoder
~~~~~~~~~~~~~~~~~~~

This data structure defines an encoder. It contains a set of callbacks used for encoding of :ref:`mjSpec` and
:ref:`mjModel` into :ref:`mjResource`.

.. mujoco-include:: mjpEncoder



.. _tyBatchStructure:

Batched simulation
^^^^^^^^^^^^^^^^^^

These types are used by the :ref:`batch API<Batchedsimulation>`, declared in `mjbatch.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjbatch.h>`__.


.. _mjBatch:

mjBatch
~~~~~~~

An opaque handle to ``nsim`` simulations of one model and the thread pool that runs them, created by
:ref:`mjb_makeBatch` and freed by :ref:`mjb_deleteBatch`. See :ref:`Batched simulation<siBatch>`.


.. _mjBatchConfig:

mjBatchConfig
~~~~~~~~~~~~~

The parameters of :ref:`mjb_makeBatch`, set to their defaults by :ref:`mjb_defaultConfig`.

.. mujoco-include:: mjBatchConfig


.. _mjBatchRecord:

mjBatchRecord
~~~~~~~~~~~~~

A field or state recorded by :ref:`mjb_rollout` after every substep. ``buf`` holds ``n x nstep x size`` values for the
``n`` simulations of the call, in call order, where ``size`` is the field's number of elements per simulation, or
``mj_stateSize(m, spec)`` for a state given by the :ref:`mjtState` bits ``spec``, in which case the values are
:ref:`mjtNum`.

.. mujoco-include:: mjBatchRecord



.. _tyFunction:

Function types
--------------

MuJoCo callbacks have corresponding function types. They are defined in `mjdata.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h>`__ and in `mjui.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjui.h>`__. The actual callback functions are documented
in the :doc:`globals<APIglobals>` page.


.. _tyPhysicsCallbacks:

Physics Callbacks
^^^^^^^^^^^^^^^^^

These function types are used by :ref:`physics callbacks<glPhysics>`.


.. _mjfGeneric:

mjfGeneric
~~~~~~~~~~

.. code-block:: C

   typedef void (*mjfGeneric)(const mjModel* m, mjData* d);

This is the function type of the callbacks :ref:`mjcb_passive` and :ref:`mjcb_control`.


.. _mjfConFilt:

mjfConFilt
~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfConFilt)(const mjModel* m, mjData* d, int geom1, int geom2);

This is the function type of the callback :ref:`mjcb_contactfilter`. The return value is 1: discard,
0: proceed with collision check.


.. _mjfSensor:

mjfSensor
~~~~~~~~~

.. code-block:: C

   typedef void (*mjfSensor)(const mjModel* m, mjData* d, int stage);

This is the function type of the callback :ref:`mjcb_sensor`.


.. _mjfTime:

mjfTime
~~~~~~~

.. code-block:: C

   typedef mjtNum (*mjfTime)(void);

This is the function type of the callback :ref:`mjcb_time`.


.. _mjfAct:

mjfAct
~~~~~~

.. code-block:: C

   typedef mjtNum (*mjfAct)(const mjModel* m, const mjData* d, int id);

This is the function type of the callbacks :ref:`mjcb_act_dyn`, :ref:`mjcb_act_gain` and :ref:`mjcb_act_bias`.


.. _mjfCollision:

mjfCollision
~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfCollision)(const mjModel* m, const mjData* d,
                               mjPreContact* con, int g1, int g2, mjtNum margin);

This is the function type of the callbacks in the collision table :ref:`mjCOLLISIONFUNC`.


.. _tyLogCallbacks:

Log Callbacks
^^^^^^^^^^^^^

.. _mjfLogHandler:

mjfLogHandler
~~~~~~~~~~~~~

.. code-block:: C

   typedef void (*mjfLogHandler)(const mjLogMessage*);

This is the function type of the log handler callback installed via :ref:`mju_setLogHandler`. The handler receives
all errors, warnings and informational messages as structured :ref:`mjLogMessage` data. It must be thread-safe.

It must not call :ref:`mju_error` from within the callback.


.. _tyUICallbacks:

UI Callbacks
^^^^^^^^^^^^

These function types are used by the UI framework.

.. _mjfItemEnable:

mjfItemEnable
~~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfItemEnable)(int category, void* data);

This is the function type of the predicate function used by the UI framework to determine if each item is enabled or
disabled.

.. _tyRPCallbacks:

Resource Provider Callbacks
^^^^^^^^^^^^^^^^^^^^^^^^^^^

These callbacks are used by :ref:`resource providers<exProvider>`.

.. _mjfOpenResource:

mjfOpenResource
~~~~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfOpenResource)(mjResource* resource);

This callback is for opening a resource; returns zero on failure. Note that
if this callback returns zero, the ``close`` callback will not be called.
Therefore, the ``open`` callback is responsible for cleaning up any allocated
memory or resources before returning zero to avoid memory leaks.

.. _mjfReadResource:

mjfReadResource
~~~~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfReadResource)(mjResource* resource, const void** buffer);

This callback is for reading a resource. Returns number of bytes stored in buffer and returns -1 on error.

.. _mjfCloseResource:

mjfCloseResource
~~~~~~~~~~~~~~~~

.. code-block:: C

   typedef void (*mjfCloseResource)(mjResource* resource);

This callback is for closing a resource, and is responsible for freeing any allocated memory.

.. _mjfGetResourceDir:

mjfGetResourceDir
~~~~~~~~~~~~~~~~~

.. code-block:: C

   typedef void (*mjfGetResourceDir)(mjResource* resource, const char** dir, int* ndir);

This callback is for returning the directory of a resource, by setting dir to the directory string with ndir being size
of directory string.

.. _mjfResourceModified:

mjfResourceModified
~~~~~~~~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfResourceModified)(const mjResource* resource);

This callback is for checking if a resource was modified since it was last read.
Returns positive value if the resource was modified since last open, 0 if resource was not modified,
and negative value if inconclusive.

.. _mjfDecode:

mjfDecode
~~~~~~~~~

.. code-block:: C

   typedef mjSpec* (*mjfDecode)(mjResource* resource, const mjVFS* vfs, char* error, int error_sz);


This callback is given an opened resource, and is responsible for decoding it into a :ref:`mjSpec`.
Ownership of the resource and the returned spec is responsibility of the caller.
When decoding fails, the callback should return NULL and, if ``error`` is not NULL, write a description of the
failure into ``error`` (of size ``error_sz``).

.. _mjfCanDecode:

mjfCanDecode
~~~~~~~~~~~~

.. code-block:: C

   typedef int (*mjfCanDecode)(const mjResource* resource);


This callback is given an opened resource, and is responsible for returning true if the resource can
be decoded by the :ref:`mjpDecoder<mjpDecoder>`.

.. _mjfEncode:

mjfEncode
~~~~~~~~~

.. code-block:: C

   typedef mjtSize (*mjfEncode)(const mjSpec* s, const mjModel* m, const mjVFS* vfs,
                                mjResource* resource);


This callback populates the :ref:`mjResource<mjResource>` `data` member with bytes representing the
given spec in the format associated with the owning plugin. This may be called with the associated
compiled :ref:`mjModel`.


.. _tyBatchCallbacks:

Batch callbacks
^^^^^^^^^^^^^^^

This function type is used by the :ref:`batch API<Batchedsimulation>`.


.. _mjfBatchFunc:

mjfBatchFunc
~~~~~~~~~~~~

.. code-block:: C

   typedef void (*mjfBatchFunc)(const mjModel* m, mjData* d, int sim, void* arg);

This is the function type run by :ref:`mjb_apply` on simulation ``sim``, on a batch thread. The model has the
simulation's per-simulation fields applied, and ``d`` holds its state.


.. _tyNotes:

Notes
-----

This section contains miscellaneous notes regarding data-structure conventions in MuJoCo struct types.


.. _tyNotesCom:

c-frame variables
^^^^^^^^^^^^^^^^^

:ref:`mjData` contains two arrays with the ``c`` prefix, which are used for internal calculations: ``cdof`` and
``cinert``, both computed by :ref:`mj_comPos`. The ``c`` prefix means that quantities are with respect to the "c-frame",
a frame at the center-of-mass of the local kinematic subtree (``mjData.subtree_com``), oriented like the world frame.
This choice increases the precision of kinematic computations for mechanisms that are distant from the global origin.

``cdof``:
  These 6D motion vectors (3 rotation, 3 translation) describe the instantaneous axis of a degree-of-freedom and are
  used by all Jacobian functions. The minimal computation required for analytic Jacobians is :ref:`mj_kinematics`
  followed by :ref:`mj_comPos`.

``cinert``:
  These 10-vectors describe the inertial properties of a body in the c-frame and are used by the Composite Rigid Body
  algorithm (:ref:`mj_crb`). The 10 numbers are packed arrays of lengths (6, 3, 1) with semantics:

  ``cinert[0-5]``: Upper triangle of the body's inertia matrix.

  ``cinert[6-8]``: Body mass multiplied by the body CoM's offset from the c-frame origin.

  ``cinert[9]``: Body mass.

.. _tyNotesConvex:

Convex hulls
^^^^^^^^^^^^

The convex hull descriptors are stored in :ref:`mjModel`:

.. code-block:: C

   int*      mesh_graphadr;     // graph data address; -1: no graph      (nmesh x 1)
   int*      mesh_graph;        // convex graph data                     (nmeshgraph x 1)

If mesh ``N`` has a convex hull stored in :ref:`mjModel` (which is optional), then ``m->mesh_graphadr[N]`` is the offset
of mesh ``N``'s convex hull data in ``m->mesh_graph``. The convex hull data for each mesh is a record with the following
format:

.. code-block:: C

   int numvert;
   int numface;
   int vert_edgeadr[numvert];
   int vert_globalid[numvert];
   int edge_localid[numvert+3*numface];
   int face_globalid[3*numface];

Note that the convex hull contains a subset of the vertices of the full mesh. We use the nomenclature ``globalid`` to
refer to vertex indices in the full mesh, and ``localid`` to refer to vertex indices in the convex hull. The meaning of
the fields is as follows:

``numvert``
   Number of vertices in the convex hull.

``numface``
   Number of faces in the convex hull.

``vert_edgeadr[numvert]``
   For each vertex in the convex hull, this is the offset of the edge record for that vertex in edge_localid.

``vert_globalid[numvert]``
   For each vertex in the convex hull, this is the corresponding vertex index in the full mesh

``edge_localid[numvert+3*numface]``
   This contains a sequence of edge records, one for each vertex in the convex hull. Each edge record is an array of
   vertex indices (in localid format) terminated with -1. For example, say the record for vertex 7 is: 3, 4, 5, 9, -1.
   This means that vertex 7 belongs to 4 edges, and the other ends of these edges are vertices 3, 4, 5, 9. In this way
   every edge is represented twice, in the edge records of its two vertices. Note that for a closed triangular mesh
   (such as the convex hulls used here), the number of edges is ``3*numface/2``. Thus when each edge is represented
   twice, we have ``3*numface edges``. And since we are using the separator -1 at the end of each edge record (one
   separator per vertex), the length of ``edge_localid`` is ``numvert+3*numface``.

``face_globalid[3*numface]``
   For each face of the convex hull, this contains the indices of the three vertices in the full mesh
