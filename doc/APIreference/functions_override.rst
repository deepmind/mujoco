..
  This file contains each section text along with function doc overrides.  By default the docs use the function doc
  pulled from the header files.

.. _Virtualfilesystem:

Virtual file system (VFS) enables the user to load all necessary files in memory, including MJB binary model files, XML
files (MJCF, URDF and included files), STL meshes, PNGs for textures and height fields, and HF files in our custom
height field format. Model and resource files in the VFS can also be constructed programmatically (say using a Python
library that writes to memory). Once all desired files are in the VFS, the user can call :ref:`mj_loadModel` or
:ref:`mj_loadXML` with a pointer to the VFS. When this pointer is not NULL, the loaders will first check the VFS for any
files they are about to load, and only access the disk if the file is not found in the VFS.

The VFS must first be allocated using :ref:`mj_defaultVFS` and must be freed with :ref:`mj_deleteVFS`.


.. _mj_defaultVFS:

Initialize an empty VFS, :ref:`mj_deleteVFS` must be called to deallocate the VFS.

.. _mj_addFileVFS:

Add file to VFS. The directory argument is optional and can be NULL or empty. Returns 0 on success,
2 on name collision, or -1 when an internal error occurs.

*Nullable:* ``directory``

.. _Assetcache:

The asset cache is a mechanism for caching assets (e.g. textures, meshes, etc.) to avoid repeated slow recompilation.
The following methods provide way to control the capacity of the cache or to disable it altogether.

.. _Resources:

Resources are the interface between :ref:`resource providers <exProvider>` and MuJoCo model compilation code.
These functions provide the means to query the resource provider and obtain resources.

.. _Parseandcompile:

The key function here is :ref:`mj_loadXML`. It invokes the built-in parser and compiler, and either returns a pointer to
a valid mjModel, or NULL - in which case the user should check the error information in the user-provided string.
The model and all files referenced in it can be loaded from disk or from a VFS when provided.

.. _mj_compile:

Compile :ref:`mjSpec` to :ref:`mjModel`. A spec can be edited and compiled multiple times, returning a new
:ref:`mjModel` instance that takes the edits into account.
If compilation fails, :ref:`mj_compile` returns ``NULL``; the error can be read with :ref:`mjs_getError`.

Compilation leaves the spec as it is written, with two exceptions. It completes :ref:`keyframes<keyframe>`: their
vectors are sized for the model, and keyframes are added up to :ref:`nkey<size-nkey>`. And the compiler attributes
:ref:`discardvisual<compiler-discardvisual>` and :ref:`fusestatic<compiler-fusestatic>` apply
:ref:`mjs_discardVisual` and :ref:`mjs_fuseStatic` to the spec before it is compiled, which changes it: pointers to
the elements which they remove are no longer valid. An operation which fails changes nothing, but one which succeeded
stays applied if a later one, or the compilation, fails; the error then says so.

.. _mj_copyBack:

Copy to an :ref:`mjSpec` the values which were changed in the :ref:`mjModel` that was compiled from it. The changes
can then be read in the spec, are saved with it, and are in the models which are compiled from it afterwards. An
attribute is copied if the model no longer has what compilation gave it, all of its numbers; everything else in the
spec is left as it is, including what was written there since it was compiled. The position and the orientation of a
pose, and the bounds of a range, are separate attributes. Returns 1 on success. Returns 0 if the model was not compiled
from the spec as it is structured now, or if a value was changed which the spec cannot express; nothing is copied
then, and the error can be read with :ref:`mjs_getError`. A model can also be copied back to a copy of the spec made
by :ref:`mj_copySpec`.

The values which are copied are ``mjModel.opt``, ``mjModel.vis`` and ``mjModel.stat``, the real-valued parameters of
bodies, joints, geoms, sites, cameras, lights, materials, contact pairs, equalities, tendons, actuators and sensors,
custom numeric and tuple data, keyframes and the elevation data of height fields. Each is written as what compiles to
it:

- A pose is written in the :ref:`frame<frame>` which the element is in. An orientation which was not changed stays as
  it was written, for example as Euler angles, and one which was changed is written as a quaternion. Where
  compilation offsets the position by the orientation, for a mesh geom whose mesh is not centered at its origin and a
  body which is :ref:`aligned<body-freejoint-align>` with its free joint, a new orientation also writes the position.
  The reference pose of a body with a free joint is taken from ``mjModel.qpos0`` if that was changed.
- Angles are written in the :ref:`unit<compiler-angle>` of the spec.
- The size and pose of a geom or site which were written as ``fromto``, or computed by fitting a geom to a mesh, are
  written in its place. A new radius of a capsule or cylinder leaves ``fromto`` as it is.
- A mass or inertia which was changed gives the body an explicit :ref:`inertial<body-inertial>`, which is then no
  longer inferred from its geoms; a body which has one keeps it as it was written if only the mass was changed. The
  inertia of the other bodies still follows their geoms, including a size or pose which was copied.
- A range whose ``limited`` attribute is "auto" keeps the limited state which it has in the model: the attribute is
  set if the new range would be inferred otherwise.
- A stiffness or damping of a joint with :ref:`springdamper<body-joint-springdamper>`, and a control or activation
  range which was inherited with :ref:`inheritrange<actuator-position-inheritrange>`, are written in its place.
- Elevation data of a height field which was read from a file is written in place of the file.
- The relative pose of a :ref:`weld<equality-weld>` between bodies which is marked by a nonzero
  ``mjModel.eq_data[11]`` is written as computed, with a zero quaternion.

The following changes cannot be expressed in the spec and are errors: the mass or inertia of a body when
:ref:`inertiafromgeom<compiler-inertiafromgeom>` is "true" or :ref:`settotalmass<compiler-settotalmass>` is set; the
inertial frame of a body which is :ref:`aligned<body-freejoint-align>` with its free joint; the pose of the world
body; the anchor or axis of a free joint and the axis of a ball joint; the size of a mesh or height field geom or of a
mesh site, and the frame of a mesh; the field of view of a camera which has a sensor size, and the intrinsics of one
which has none; and different control ranges for the inputs of an actuator which has one. So are changes which the
next compilation would not keep: damping, armature, friction loss or their solver parameters which differ between the
degrees of freedom of a ball or free joint; elevation data of a height field whose lowest and highest values are not
0 and 1; and the anchor of a :ref:`connect<equality-connect>` between bodies in its second body, or the relative pose
of a marked weld, other than the one which :ref:`mj_setConst` computes.

.. _mj_recompile:

Recompile spec to model, preserving the state. Like :ref:`mj_compile`, this function compiles an :ref:`mjSpec` to an
:ref:`mjModel`, with two differences. First, rather than returning an entirely new model, it will
reallocate existing :ref:`mjModel` and :ref:`mjData` instances in-place. Second, it will preserve the
:ref:`integration state<siIntegrationState>`, as given in the provided :ref:`mjData` instance, while accounting for
newly added or removed degrees of freedom. This allows the user to continue simulation with the same model and data
struct pointers while editing the model programmatically.

:ref:`mj_recompile` returns 0 if compilation succeed. In the case of failure, the given :ref:`mjModel` and :ref:`mjData`
instances will be deleted; as in :ref:`mj_compile`, the compilation error can be read with :ref:`mjs_getError`.

.. _mj_saveLastXML:

Update XML data structures with info from low-level model created with :ref:`mj_loadXML`, save as MJCF.
If error is not NULL, it must have size error_sz.

Note that this function only saves models that have been loaded with :ref:`mj_loadXML`, the legacy loading mechanism.
See the :ref:`model editing<meOverview>` chapter to understand the difference between the old and new model loading and
saving mechanisms.

.. _mj_saveXMLString:

Save spec to XML string, return 0 on success, -1 on failure. If the length of the output buffer is too small, returns
the required size. See :ref:`mj_saveXML` for what is saved.

.. _mj_saveXML:

Save spec to XML file, return 0 on success, -1 otherwise. The compiler attributes
:ref:`savecompiled<compiler-savecompiled>` and :ref:`savecanonical<compiler-savecanonical>` of the spec say whether
the model is saved as it was compiled or as it is written in the spec, and in which notation; see
:ref:`Model Encoding & Saving <meSaving>`. Saving the compiled values requires that the spec first be compiled, and
compiled again after it is structurally edited (elements added, deleted or attached).

.. _mj_encode:

Encode :ref:`mjSpec` or :ref:`mjModel` to a file. The output format is determined by the file extension
(case insensitive) or ``content_type``. Returns the number of bytes written on success, -1 on failure.

For detailed documentation, supported output formats (``.xml``, ``.mjb``, ``.txt``, ``.mjz``), and custom encoder
plugins, see :ref:`Model Encoding & Saving <meSaving>`.

*Nullable:* ``s``, ``m``, ``vfs``, ``error``

.. _Mainsimulation:

These are the main entry points to the simulator. Most users will only need to call :ref:`mj_step`, which computes
everything and advanced the simulation state by one time step. Controls and applied forces must either be set in advance
(in ``mjData.{ctrl, qfrc_applied, xfrc_applied}``), or a control callback :ref:`mjcb_control` must be installed which
will be called just before the controls and applied forces are needed. Alternatively, one can use :ref:`mj_step1` and
:ref:`mj_step2` which break down the simulation pipeline into computations that are executed before and after the
controls are needed; in this way one can set controls that depend on the results from :ref:`mj_step1`. Keep in mind
though that the RK4 solver does not work with mj_step1/2. See :ref:`Pipeline` for a more detailed description.

mj_forward performs the same computations as :ref:`mj_step` but without the integration. It is useful after loading or
resetting a model (to put the entire mjData in a valid state), and also for out-of-order computations that involve
sampling or finite-difference approximations.

:ref:`mj_inverse` runs the inverse dynamics, and writes its output in ``mjData.qfrc_inverse``. Note that ``mjData.qacc``
must be set before calling this function. Given the state (qpos, qvel, act), mj_forward maps from force to acceleration,
while mj_inverse maps from acceleration to force. Mathematically these functions are inverse of each other, but
numerically this may not always be the case because the forward dynamics rely on a constraint optimization algorithm
which is usually terminated early. The difference between the results of forward and inverse dynamics can be computed
with the function :ref:`mj_compareFwdInv`, which can be thought of as another solver accuracy check (as well as a
general sanity check).

The skip version of :ref:`mj_forward` and :ref:`mj_inverse` are useful for example when qpos was unchanged but qvel was
changed (usually in the context of finite differencing). Then there is no point repeating the computations that only
depend on qpos. Calling the dynamics with skipstage = :ref:`mjSTAGE_POS<mjtStage>` will achieve these savings.

.. _Initialization:

This section contains functions that load/initialize the model or other data structures. Their use is well illustrated
in the code samples.

.. _mj_saveModel:

Save model to binary MJB file or memory buffer; buffer has precedence when given. The model is written as it is: a
model with :ref:`by-reference<CAssetReference>` textures keeps them as references, and the file loads wherever the
paths resolve. A hermetic file is :ref:`mj_embedAssets`, then :ref:`mj_saveModel` on the result, then
:ref:`mj_deleteModel`.

*Nullable:* ``filename``, ``buffer``

.. _mjv_copyData:

Copy :ref:`mjData`, skipping large arrays not required for visualization, like the constraint Jacobian. The copy is
meant for visualization, e.g. with :ref:`mjv_updateScene`; the skipped arrays are ``NULL``, so it is not a valid input
to simulation functions.

.. _Printing:

These functions can be used to print various quantities to the screen for debugging purposes.

.. _Components:

These are components of the simulation pipeline, called internally from :ref:`mj_step`, :ref:`mj_forward` and
:ref:`mj_inverse`. It is unlikely that the user will need to call them.

.. _mj_implicit:

Integrates the simulation state using an implicit-in-velocity integrator (either "implicit" or "implicitfast", see
:ref:`Numerical Integration<geIntegration>`), and advances simulation time. See `mjdata.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h>`__ for fields computed by this function.

.. _Subcomponents:

These are sub-components of the simulation pipeline, called internally from the components above.

.. _mj_makeM:

Compute the composite rigid body inertia with :ref:`mj_crb`, add terms due
to :ref:`tendon armature<tendon-spatial-armature>`. The joint-space inertia matrix is stored in both ``mjData.qM`` and
``mjData.M``. These arrays represent the same quantity using different layouts (parent-based and compressed sparse row,
respectively).

.. _mj_factorM:

Compute sparse :math:`L^T D L` factorizaton of inertia matrix.

.. _mj_solveM:

Solve linear system :math:`M x = y` using factorization: :math:`x = (L^T D L)^{-1} y`

.. _mj_solveM2:

Half of linear solve: :math:`x = \sqrt{D^{-1}} (L^T)^{-1} y`

.. _mj_subtreeVel:

Sub-tree linear velocity and angular momentum: compute ``subtree_linvel``, ``subtree_angmom``.
This function is triggered automatically if the subtree :ref:`velocity<sensor-subtreelinvel>` or
:ref:`momentum<sensor-subtreeangmom>` sensors are present in the model.
It is also triggered for :ref:`user sensors<sensor-user>` of :ref:`stage<sensor-user-needstage>` "vel".

.. _mj_rne:

Recursive Newton Euler: compute :math:`M(q) \ddot q + C(q,\dot q)`. ``flg_acc=0`` removes the inertial term (i.e.
assumes :math:`\ddot q = 0`).

.. _mj_rnePostConstraint:

Recursive Newton Euler with final computed forces and accelerations.
Computes three body-level ``nv x 6`` arrays, all defined in the subtreecom-based
:ref:`c-frame<tyNotesCom>` and arranged in ``[rotation(3), translation(3)]`` order.

- ``cacc``: Body acceleration, required for :ref:`mj_objectAcceleration`.
- ``cfrc_int``: Interaction force with the parent body.
- ``cfrc_ext``: External force acting on the body.

This function is triggered automatically if the following sensors are present in the model:
:ref:`accelerometer<sensor-accelerometer>`, :ref:`force<sensor-force>`, :ref:`torque<sensor-torque>`,
:ref:`framelinacc<sensor-framelinacc>`, :ref:`frameangacc<sensor-frameangacc>`.
It is also triggered for :ref:`user sensors<sensor-user>` of :ref:`stage<sensor-user-needstage>` "acc".

``cfrc_ext`` collects the forces that are not transmitted through the joints: applied Cartesian forces
(``xfrc_applied``), contacts, connect and weld constraints, and spatial tendons (spring, damper, actuator, constraint
and armature forces along the tendon path). ``cfrc_int`` is then the wrench transmitted through the joint, and its
projection on the joint axes is the total joint-space force. Forces of actuators with site, slider-crank and body
transmissions, gravity compensation, fluid forces, flex forces and custom passive forces are not yet collected and are
attributed to the joints.

.. _mj_constraintUpdate:

Compute ``efc_state``, ``efc_force``, ``qfrc_constraint``, and (optionally) cone Hessians.
If ``cost`` is not ``NULL``, set ``*cost = s(jar)`` where ``jar = Jac*qacc - aref``.

*Nullable:* ``cost``

.. _Support:

These are support functions that need access to :ref:`mjModel` and :ref:`mjData`, unlike the utility functions which do
not need such access. Support functions are called within the simulator but some of them can also be useful for custom
computations, and are documented in more detail below.

.. _mj_stateSize:

Returns the number of :ref:`mjtNum` |-| s required for a given state signature. The bits of the integer ``sig``
correspond to element fields of :ref:`mjtState`.

.. _mj_getState:

Copy concatenated state components specified by ``sig`` from ``d`` into ``state``. The bits of the integer
``sig`` correspond to element fields of :ref:`mjtState`. Fails with :ref:`mju_error` if ``sig`` is invalid.

.. _mj_extractState:

Extract into ``dst`` the subset of components specified by ``dstsig`` from a state ``src`` previously obtained via
:ref:`mj_getState` with components specified by ``srcsig``. Fails with :ref:`mju_error` if the bits set in ``dstsig``
is not a subset of the bits set in ``srcsig``.

.. _mj_setState:

Copy concatenated state components specified by ``sig`` from  ``state`` into ``d``. The bits of the integer
``sig`` correspond to element fields of :ref:`mjtState`. Fails with :ref:`mju_error` if ``sig`` is invalid.

.. _mj_readCtrl:

Read the control value for an actuator at a given time, taking delays into account. If no history buffer exists, return
a pointer to the actuator's slice of ``mjData.ctrl``. If a history buffer exists (:ref:`nsample<actuator-general-nsample>` > 0),
read from the delay buffer at ``time - actuator_delay[id]``. Note that the subtraction of the delay changes the semantic
of the ``time`` argument from "time at which values were pushed into the delay buffer" to "time at which values come out
of the delay buffer". See :ref:`Delays<CDelay>` for details.

**Return value semantics:**

- If no history buffer exists (:ref:`nsample<actuator-general-nsample>` = 0), returns a pointer to the actuator's slice
  of ``mjData.ctrl``.
- If a history buffer exists (:ref:`nsample<actuator-general-nsample>` > 0) and the requested time matches a stored
  sample (always true for ``interp = 0``), returns a pointer to the data in the history buffer.
- If interpolation is required (``interp = 1 or 2``), returns ``NULL`` and writes the interpolated result to
  ``result`` (must be of size ``actuator_ctrlnum[id]``).

**Interpolation:**

- ``interp = 0``: Zero-order hold (piecewise constant)
- ``interp = 1``: Piecewise Linear
- ``interp = 2``: Cubic Spline (Catmull-Rom)
- ``interp = -1``: Use the actuator's :ref:`interp<actuator-general-interp>` value.

Constant extrapolation is used outside of buffer bounds.

.. _mj_readSensor:

Read a sensor value at a given time, taking delays into account. If no history buffer exists, return a pointer to the
sensor's slice of ``mjData.sensordata``. If a history buffer exists (:ref:`nsample<sensor-nsample>` > 0), read from the
history buffer at ``time - sensor_delay[id]``. Note that the subtraction of the delay changes the semantic of the
``time`` argument from "time at which values were pushed into the delay buffer" to "time at which values come out of the
delay buffer". See :ref:`Delays<CDelay>` for details.

**Return value semantics:**

- If no history buffer exists (:ref:`nsample<sensor-nsample>` = 0), returns a pointer to the sensor's slice of
  ``mjData.sensordata``.
- If a history buffer exists (:ref:`nsample<sensor-nsample>` > 0) and the requested time matches a stored sample
  (always true for ``interp = 0``), returns a pointer to the data in the history buffer.
- If interpolation is required (``interp = 1 or 2``), returns ``NULL`` and writes the interpolated result to
  ``result`` (must be of size ``dim``).

**Interpolation:**

- ``interp = 0``: Zero-order hold (piecewise constant)
- ``interp = 1``: Piecewise Linear
- ``interp = 2``: Cubic Spline (Catmull-Rom)
- ``interp = -1``: Use the value in :ref:`interp<sensor-interp>`

Constant extrapolation is used outside of buffer bounds.


**Usage:**

.. code-block:: C

   // read sensor 0 of data size `dim` at time t
   mjtNum result[dim];
   const mjtNum* ptr = mj_readSensor(m, d, 0, t, result, /* interp = */ 1);
   const mjtNum* data = ptr ? ptr : result;

.. _mj_initCtrlHistory:

Initialize the history buffer for an actuator with custom values. The ``times`` array specifies the timestamps for each
sample (must be length :ref:`nsample<actuator-general-nsample>`), and ``values`` specifies the control values (must be of
size ``nsample * actuator_ctrlnum[id]``). If ``times`` is ``NULL``, the existing timestamps in the buffer are used, and
only the values are updated.
See :ref:`Delays<CDelay>` for details.

.. _mj_initSensorHistory:

Initialize the history buffer for a sensor with custom values. The ``times`` array specifies the timestamps for each
sample (must be length :ref:`nsample<sensor-nsample>`), and ``values`` specifies the sensor values (must be of size
``nsample * dim``). If ``times`` is ``NULL``, the existing timestamps in the buffer are used.
The ``phase`` argument sets the user slot, which stores the last computation time for interval sensors.
See :ref:`Delays<CDelay>` for details.

.. _mj_mulJacVec:

This function multiplies the constraint Jacobian mjData.efc_J by a vector. Note that the Jacobian can be either dense or
sparse; the function is aware of this setting. Multiplication by J maps velocities from joint space to constraint space.

.. _mj_mulJacTVec:

Same as mj_mulJacVec but multiplies by the transpose of the Jacobian. This maps forces from constraint space to joint
space.

.. _mj_jac:

This function computes an end-effector kinematic Jacobian, describing the local linear relationship between the
degrees-of-freedom and a given point. Given a body specified by its integer id (``body``) and a 3D point in the world
frame (``point``) treated as attached to the body, the Jacobian has both translational (``jacp``) and rotational
(``jacr``) components. Passing ``NULL`` for either pointer will skip that part of the computation. Each component is a
3-by-nv matrix. Each row of this matrix is the gradient of the corresponding coordinate of the specified point with
respect to the degrees-of-freedom. The frame with respect to which the Jacobian is computed is centered at the body
center-of-mass but aligned with the world frame. The minimal :ref:`pipeline stages<piForward>` required for Jacobian
computations to be consistent with the current generalized positions ``mjData.qpos`` are :ref:`mj_kinematics` followed
by :ref:`mj_comPos`.

*Nullable:* ``jacp``, ``jacr``

.. _mj_jacBody:

This and the remaining variants of the Jacobian function call mj_jac internally, with the center of the body, geom or
site. They are just shortcuts; the same can be achieved by calling mj_jac directly.

*Nullable:* ``jacp``, ``jacr``

.. _mj_jacDot:

This function computes the time-derivative of an end-effector kinematic Jacobian computed by :ref:`mj_jac`.
The minimal :ref:`pipeline stages<piStages>` required for computation to be
consistent with the current generalized positions and velocities ``mjData.{qpos, qvel}`` are
:ref:`mj_kinematics`, :ref:`mj_comPos`, :ref:`mj_comVel` (in that order).

*Nullable:* ``jacp``, ``jacr``

.. _mj_angmomMat:

This function computes the ``3 x nv`` angular momentum matrix :math:`H(q)`, providing the linear mapping from
generalized velocities to subtree angular momentum. More precisely if :math:`h` is the subtree angular momentum of
body index ``body`` in ``mjData.subtree_angmom`` (reported by the :ref:`subtreeangmom<sensor-subtreeangmom>` sensor)
and :math:`\dot q` is the generalized velocity ``mjData.qvel``, then :math:`h = H \dot q`.

.. _mj_name2id:

Get id of object with the specified :ref:`mjtObj` type and name, returns -1 if id not found.

.. _mj_id2name:

Get name of object with the specified :ref:`mjtObj` type and id, returns ``NULL`` if name not found.

.. _mj_geomDistance:

Returns the smallest signed distance between two geoms and optionally the segment from ``geom1`` to ``geom2``.
Returned distances are bounded from above by ``distmax``. |br| If no collision of distance smaller than ``distmax`` is
found, the function will return ``distmax`` and ``fromto``, if given, will be set to (0, 0, 0, 0, 0, 0).

*Nullable:* ``fromto``

.. admonition:: different (correct) behavior under `nativeccd`
   :class: note

   As explained in :ref:`Collision Detection<coDistance>`, distances are inaccurate when using the
   :ref:`legacy CCD pipeline<coCCD>`, and its use is discouraged.

.. _mj_mulM:

This function multiplies the joint-space inertia matrix stored in ``mjData.M`` by a vector.

.. _mj_applyFT:

This function can be used to apply a Cartesian force and torque to a point on a body, and add the result to the vector
mjData.qfrc_applied of all applied forces. Note that the function requires a pointer to this vector, because sometimes
we want to add the result to a different vector.

.. _mj_objectAcceleration:

Compute object 6D acceleration (rot:lin) in object-centered frame, world/local orientation. If acceleration or force
sensors are not present in the model, :ref:`mj_rnePostConstraint` must be manually called in order to calculate
mjData.cacc -- the total body acceleration, including contributions from the constraint solver.

.. _mj_differentiatePos:

This function subtracts two vectors in the format of qpos (and divides the result by dt), while respecting the
properties of quaternions. Recall that unit quaternions represent spatial orientations. They are points on the unit
sphere in 4D. The tangent to that sphere is a 3D plane of rotational velocities. Thus when we subtract two quaternions
in the right way, the result is a 3D vector and not a 4D vector. Thus the output qvel has dimensionality nv while the
inputs have dimensionality nq.

.. _mj_integratePos:

This is the opposite of mj_differentiatePos. It adds a vector in the format of qvel (scaled by dt) to a vector in the
format of qpos.

.. _Raycollisions:

Ray collisions, also known as ray casting, find the distance ``x`` of a ray's intersection with a geom, where a ray is
a line emanating from the 3D point ``p`` in the direction ``v`` i.e., ``(p + x*v, x >= 0)``. All functions in this
family return the distance to the nearest geom surface, or -1 if there is no intersection. Note that if ``p`` is inside
a geom, the ray will intersect the surface from the inside which still counts as an intersection.

All ray collision functions rely on quantities computed by :ref:`mj_kinematics` (see :ref:`mjData`), so must be called
after  :ref:`mj_kinematics`, or functions that call it (e.g. :ref:`mj_fwdPosition`). The top level functions, which
intersect with all geoms types, are :ref:`mj_ray` which casts a single ray, and :ref:`mj_multiRay` which casts multiple
rays from a single point.

.. _mj_ray:

Intersect ray ``pnt+x*vec, x >= 0`` with geoms.

- Return distance ``x`` to nearest surface, or -1 if no intersection.
- If ``geomid`` is not NULL, write the id of the intersected geom or -1 if not intersection.
- If ``normal`` is not NULL, write the surface normal at the intersection point. The normal always points **out of the
  geometry**, regardless of the ray's direction (i.e., including rays hitting the surface from the inside).
- Exclude geoms in body with id ``bodyexclude``, use -1 to include all bodies.
- ``geomgroup`` is an array of length :ref:`mjNGROUP<glNumericVisualization>`, where 1 means the group should be included. Pass
  NULL to skip geom group exclusion.
- If ``flg_static`` is 0, static geoms will be excluded.

*Nullable:* ``geomgroup``, ``geomid``, ``normal``

.. _Interaction:

These functions implement abstract mouse interactions, allowing control over cameras and perturbations. Their use is well
illustrated in :ref:`simulate<saSimulate>`.

.. _mjv_select:

This function is used for mouse selection, relying on ray intersections. aspectratio is the viewport width/height. relx
and rely are the relative coordinates of the 2D point of interest in the viewport (usually mouse cursor). The function
returns the id of the geom under the specified 2D point, or -1 if there is no geom (note that they skybox if present is
not a model geom). The 3D coordinates of the clicked point are returned in selpnt. See :ref:`simulate<saSimulate>` for
an illustration.

.. _Visualization-api:

The functions in this section implement abstract visualization. The results are used by the OpenGL renderer, and can
also be used by users wishing to implement their own renderer, or hook up MuJoCo to advanced rendering tools such as
Unity or Unreal Engine. See :ref:`simulate<saSimulate>` for illustration of how to use these functions.

.. _FilamentRenderingApi:

Rendering functions using the Filament rendering engine. These functions are prefixed with ``mjrf``. See
:ref:`Filament Rendering<tyFilamentRenderStructure>` for an overview of the core types and their uses.

.. _OpenGLrendering:

These functions expose the OpenGL renderer. See :ref:`simulate<saSimulate>` for an illustration
of how to use these functions.

.. _UIframework:

For a high-level description of the UI framework, see :ref:`UI`.

.. _mjui_add:

This is the helper function used to construct a UI. The second argument points to an array of :ref:`mjuiDef` structs,
each corresponding to one item. The last (unused) item has its type set to -1, to mark termination. The items are added
after the end of the last used section. There is also another version of this function
(:ref:`mjui_addToSection<mjui_addToSection>`) which adds items to a specified section instead of adding them at the end
of the UI. Keep in mind that there is a maximum preallocated number of sections and items per section, given by
:ref:`mjMAXUISECT<glNumericUI>` and :ref:`mjMAXUIITEM<glNumericUI>`. Exceeding these maxima results in low-level errors.

.. _mjui_update:

This is the main UI update function. It needs to be called whenever the user data (pointed to by the item data pointers)
changes, or when the UI state itself changes. It is normally called by a higher-level function implemented by the user
(``UiModify`` in :ref:`simulate.cc <saSimulate>`) which also recomputes the layout of all rectangles and associated
auxiliary buffers. The function updates the pixels in the offscreen OpenGL buffer. To perform minimal updates, the user
specifies the section and the item that was modified. A value of -1 means all items and/or sections need to be updated
(which is needed following major changes.)

.. _mjui_event:

This function is the low-level event handler. It makes the necessary changes in the UI and returns a pointer to the item
that received the event (or ``NULL`` if no valid event was recorded). This is normally called within the event handler
implemented by the user (``UiEvent`` in :ref:`simulate.cc <saSimulate>`), and then some action is taken by user code
depending on which UI item was modified and what the state of that item is after the event is handled.


.. _mjui_render:

This function is called in the screen refresh loop. It copies the offscreen OpenGL buffer to the window framebuffer. If
there are multiple UIs in the application, it should be called once for each UI. Thus ``mjui_render`` is called all the
time, while :ref:`mjui_update` is called only when changes in the UI take place. dsffsdg




.. _Errorandmemory:

.. _mju_error:

Main error function. The error message is dispatched to the active log handler (see :ref:`mju_setLogHandler`).
Errors are always fatal: if the handler returns, the process is terminated with ``exit(EXIT_FAILURE)``. Handlers
wishing to recover must ``longjmp`` or otherwise transfer control before returning.

.. _mju_warning:

Main warning function; returns to caller. The warning message is dispatched to the active log handler.

.. _mju_clearHandlers:

Clear all user handlers and restore defaults. Resets the legacy error/warning/memory callbacks to ``NULL``, restores
the default log handler, and resets the log configuration to its defaults (console and file output enabled, all info
topics disabled).

.. _mju_setLogHandler:

Set the active global log handler. Returns the previous handler (which is never ``NULL``), intended for save/restore
or callback chaining. If ``handler`` is ``NULL``, the default handler is restored. The handler receives all errors,
warnings and informational messages as a structured :ref:`mjLogMessage`. See :ref:`siLogHandler` for usage examples.

.. _mju_getLogConfig:

Get the current default handler configuration. See :ref:`mjLogConfig`.

.. _mju_setLogConfig:

Set the default handler configuration. Controls console output, file output, and info topic filtering.
See :ref:`mjLogConfig`.

Example usage (disabling file output):

.. code-block:: C

   mjLogConfig config = mju_getLogConfig();
   config.logto_file = false;
   mju_setLogConfig(config);

.. _mju_info:

Log an informational message with optional topic filtering. The ``topic`` argument is a :ref:`mjtLogTopic` value.
Topic 0 (``mjTOPIC_NONE``) always passes through. Other topics must be enabled in the default handler configuration
via :ref:`mju_setLogConfig`. Note that topic filtering is implemented in the default handler; custom handlers
receive all info messages regardless.

.. _mju_message:

Dispatch a structured :ref:`mjLogMessage` to the active log handler. This is the primary entry point for emitting
log messages with full control over all fields. The convenience functions :ref:`mju_error`, :ref:`mju_warning`, and
:ref:`mju_info` are thin wrappers that populate an ``mjLogMessage`` and call this function.

The ``subject`` field is a one-line summary (up to 1024 bytes, inline in the struct). The ``body`` field is an
optional ``const char*`` pointer to multi-line detail text, owned by the caller. When ``body`` is ``NULL``, only the
subject line is printed.

The default handler formats the output as follows:

.. code-block:: text

   LEVEL FUNC (FILE:LINE) TIME: SUBJECT
   BODY

where:

- ``LEVEL`` is ``ERROR``, ``WARNING``, ``INFO``, or ``DEBUG``.
- ``FUNC`` is present when the ``func`` field is set.
- ``(FILE:LINE)`` is present when the ``file`` and ``line`` fields are set.
- ``TIME`` is present when the ``timestamp`` field is set or file logging is active.
- ``SUBJECT`` is the contents of the ``subject`` field.
- ``BODY`` follows on the next line(s), printed raw without indentation or separators, only if non-NULL.

The default handler appends a trailing blank line after ``ERROR``, ``WARNING``, and ``INFO`` messages for visual
separation. ``DEBUG`` messages are printed compactly without a trailing blank line.

Example usage:

.. code-block:: C

   mjLogMessage msg = {
     .level = mjLOG_INFO,
     .timestamp = true,
     .body = "  height:     0.001 m\n  velocity:   0.000 m/s\n  bounces:    47",
   };
   snprintf(msg.subject, sizeof(msg.subject), "The ball has come to rest");
   mju_message(&msg);

This produces:

.. code-block:: text

   INFO Mon Jun  9 15:04:05 2026: The ball has come to rest
     height:     0.001 m
     velocity:   0.000 m/s
     bounces:    47

.. _Batchedsimulation:

Functions for running many simulations of one model on a thread pool, declared in ``mujoco/mjbatch.h``. See
:ref:`Batched simulation<siBatch>` for an overview and an example.

The functions that run simulations take a selection: ``ids`` holds ``nid`` sorted, unique simulation indices, or is
``NULL`` to select every simulation, in which case ``nid`` is ignored. They return the number of simulations in which
MuJoCo raised an error. Those simulations keep the state they had before the call, and the others run to completion.
:ref:`mjb_status` and :ref:`mjb_error` report the outcome of each simulation's last call.

Invalid arguments, such as unsorted ``ids`` or an unknown ``stage``, are a different kind of error: the function raises
it with :ref:`mju_error` on the caller's thread, before any simulation runs, and it is not counted in the return value.

Errors in simulations are trapped by a thread-local log handler. Therefore, :ref:`physics callbacks<glPhysics>`, engine
plugins and functions run by :ref:`mjb_apply` must not raise errors with :ref:`mju_error` or throw exceptions. Other
messages, such as warnings, go to the log handler the thread would use anyway. On the batch's threads this is the
:ref:`global handler<mju_setLogHandler>`. On the caller's thread it can also be a thread-local handler, such as the one
the Python bindings install. Either way, the counters of :ref:`mjb_warning` count every simulation's warnings.

The arrays that the batch returns, such as the state rows, outputs and expanded fields, are owned by the batch. They
stay at the same address until :ref:`mjb_deleteBatch` frees them.

Timers (``mjData.timer``, see :ref:`mjTimerStat`) are not part of the state. In a persistent batch, each simulation's
mjData times that simulation, and :ref:`mjb_getData` copies its timers. Otherwise, each thread's mjData accumulates the
timing of every simulation the thread runs, until a reset of one of them clears it. These timers profile the batch as a
whole rather than any one simulation.

.. _mjb_defaultConfig:

Set the default :ref:`mjBatchConfig` parameters: one thread per logical CPU, and one mjData per thread. Call it first,
then change the parameters you need.

.. _mjb_makeBatch:

Allocate a batch of ``nsim`` simulations of a copy of ``m``, and return it. ``config`` holds the :ref:`mjBatchConfig`
parameters; if it is ``NULL``, the defaults of :ref:`mjb_defaultConfig` apply. If the model cannot be batched, or making
an mjData raises an error, return ``NULL`` and write the reason into ``error``, a buffer of ``error_sz`` bytes.

The batch runs on ``config->nthread`` threads, including the caller's. If ``nthread <= 0``, it uses one thread per
logical CPU. Either way, it uses at most ``nsim`` threads. The number of logical CPUs is the machine's, which can exceed
what the process may use, for example in a container; set ``nthread`` explicitly there.

If ``config->persistent`` is nonzero, the batch keeps one mjData per simulation instead of one per thread. Models with
:ref:`sleep<option-flag-sleep>` or :ref:`IPC<option-flag-ipc>` enabled require this. In a batch that is not persistent,
:ref:`engine plugins<exPlugin>` that keep state outside ``plugin_state`` share that state among the simulations a thread
runs.

.. _mjb_deleteBatch:

Free the batch, its threads, and every array it returned. ``b`` may be ``NULL``. A function run by :ref:`mjb_apply`
must not delete its own batch.

.. _mjb_state:

Return the state rows, ``nsim x nstate``. Row ``i`` is simulation ``i``'s :ref:`integration state<siIntegrationState>`,
in :ref:`mj_getState` order. Writing to a row sets the simulation's state for its next call, which then overwrites the
row with the new state.

.. _mjb_stateComponent:

Return the address of state component ``component`` (one bit of :ref:`mjtState`) in the first state row, and write its
number of elements into ``size``. Simulation ``i``'s component is ``mjb_nstate(b) * i`` elements further on. Raise an
error if ``component`` is not one bit of ``mjSTATE_INTEGRATION``.

.. _mjb_warning:

Return the warning counters, ``nsim x mjNWARNING`` :ref:`mjWarningStat` |-| s. They are saved and restored with the
state.

.. _mjb_step:

Advance each selected simulation by ``nstep`` calls to :ref:`mj_step`, then compute the stages of :ref:`mj_forward` up
to ``stage`` (an :ref:`mjtStage`) at the new state. Return the number of simulations in which MuJoCo raised an error.

As with :ref:`mj_step`, derived fields after the steps describe the state before the last substep. ``stage`` brings
them up to date:

- ``mjSTAGE_NONE`` leaves them as :ref:`mj_step` does.
- ``mjSTAGE_VEL`` makes position- and velocity-dependent fields and sensors current, as :ref:`mj_step1` does.
  Acceleration-dependent fields remain those of the last substep, under the controls it applied.
- ``mjSTAGE_ACC`` runs :ref:`mj_forward`, at nearly the cost of another step. Its acceleration-dependent values use
  controls that the next step will replace.

In a persistent batch, a simulation's next step continues from these stages with :ref:`mj_step2`, so ``mjSTAGE_VEL``
costs nothing extra. The step does not continue if anything the stages depend on changed in between, other than
``ctrl``, ``qfrc_applied`` and ``xfrc_applied``; the results are the same either way. As with :ref:`mj_step1` and
:ref:`mj_step2`, this assumes that plugins do not make the position and velocity stages depend on those inputs. A loop
of :ref:`mj_forward` and :ref:`mj_step` computes these stages twice at the same state, while a continued step computes
them once; this matters only to plugins whose computations have side effects. Installed :ref:`physics
callbacks<glPhysics>` turn the continuation off, but removing one between a call and the next step is not detected. A
continued step counts as one step in ``mjData.timer``, but its ``mjTIMER_STEP`` duration excludes the position and
velocity stages, which are timed in the call that computed them.

.. _mjb_forward:

Compute the stages of :ref:`mj_forward` up to ``stage`` (an :ref:`mjtStage`) on each selected simulation.
``mjSTAGE_ACC`` runs :ref:`mj_forward`. Return the number of simulations in which MuJoCo raised an error.

.. _mjb_reset:

Reset each selected simulation, then compute the stages of :ref:`mj_forward` up to ``stage`` (an :ref:`mjtStage`).
Return the number of simulations in which MuJoCo raised an error.

``key`` holds one keyframe per selected simulation, in call order; a keyframe of -1, or a ``NULL`` ``key``, resets to
the model's defaults. ``mjSTAGE_ACC`` runs :ref:`mj_forward`. ``mjSTAGE_NONE`` suits a reset that is followed by writes
to the state and :ref:`mjb_forward`. Writes to the state rows made before the call are discarded.

.. _mjb_status:

Return the status of each simulation's last call: ``nsim`` values, nonzero where MuJoCo raised an error. Each call
updates the values of the simulations it runs.

.. _mjb_error:

Return the error message of simulation ``sim``'s last call, or ``""`` if there was none. The string remains valid until
the next call on the simulation.

.. _mjb_output:

Register :ref:`mjData` field ``name`` as an output, and return its storage: ``nsim x size`` elements of the field's
type. Write the number of elements per simulation into ``size`` and the element size in bytes into ``elemsize``. After
every call on a simulation except :ref:`mjb_apply`, the batch copies the field from the simulation's mjData into its
row of the storage. Registering a field again returns the same storage.

``name`` can be an array field, or one of the counts ``ncon``, ``ne``, ``nf``, ``nl``, ``nefc``, ``nJ``, ``nA``,
``nisland`` and ``nidof``. Return ``NULL`` for unknown fields, and for fields of the state row, which :ref:`mjb_state`
already holds.

In a persistent batch, the storage starts with each simulation's current values. Otherwise, it is zero until the next
call on a simulation, and a field that the call does not compute holds another simulation's value. Examples are
``qfrc_inverse`` after :ref:`mj_step`, quantities computed only for the sensors that need them, and fields of stages
above the call's ``stage``.

.. _mjb_expand:

Give :ref:`mjModel` field ``name`` per-simulation values, and return their storage: ``nsim x size`` elements of the
field's type, initialized from the model. Before each call on a simulation, the batch copies the simulation's row into
its model. ``size`` and ``elemsize`` are written as by :ref:`mjb_output`. Expanding a field again returns the same
storage.

Array fields are named as in :ref:`mjModel`, and :ref:`mjOption` fields as ``"opt.<name>"``. The storage of
``bvh_aabb`` holds only the bodies' bounding volumes, which come before the meshes'. Return ``NULL`` for unknown fields
and for asset fields (meshes, heightfields, textures, skins), which all simulations share.

Writing a field that :ref:`mj_setConst` reads, an input in ``MJSETCONST_ARRAYS`` of ``mjxmacro.h`` or an
:ref:`mjOption` field, does not update the constants derived from it until :ref:`mjb_setConst` runs.

Expanding a field that :ref:`mjb_setConst` expanded makes it an input of later calls to :ref:`mjb_setConst`, if
:ref:`mj_setConst` reads it before writing it. To read such a field without that effect, use :ref:`mjb_expanded`.

.. _mjb_expanded:

Return the per-simulation storage of :ref:`mjModel` field ``name`` if it is expanded, by :ref:`mjb_expand` or by
:ref:`mjb_setConst`. Otherwise, return ``NULL``: every simulation has the model's value. Unlike :ref:`mjb_expand`, this
expands nothing, and fields that :ref:`mjb_setConst` derives stay derived. Treat the storage as read-only, since
:ref:`mjb_setConst` overwrites derived fields. ``size`` and ``elemsize`` are written as by :ref:`mjb_expand`, also when
the field is not expanded.

.. _mjb_setConst:

Run :ref:`mj_setConst` on each selected simulation's model. Return the number of simulations in which MuJoCo raised an
error; those simulations keep their constants.

Every model field that :ref:`mj_setConst` changes becomes per-simulation, as with :ref:`mjb_expand`, so each simulation
gets its own derived constants; only the arrays marked ``mjSC_OUT`` in ``MJSETCONST_ARRAYS`` and the scalars in
``MJSETCONST_SCALARS`` can change. If a call makes a new field per-simulation, :ref:`mj_setConst` runs again on the
call's simulations. Simulations outside the call need no update, since the new field starts from the model's value,
which is theirs.

Only the groups of :ref:`mj_setConst` computations (the columns of ``MJSETCONST_ARRAYS``) that read a per-simulation
field, or what an earlier such group writes, run; the others would give every simulation the model's values. With no
per-simulation field that :ref:`mj_setConst` reads, for example only friction or actuator gains, the call does nothing.
Per-simulation :ref:`mjOption` fields count as read by every group. Skipping groups assumes that the constants of the
model given to :ref:`mjb_makeBatch` are current, as after compilation or :ref:`mj_setConst`.

On each call, the fields that :ref:`mj_setConst` derives are recomputed from the model's values. Expanding such a field
with :ref:`mjb_expand` makes its values inputs instead if :ref:`mj_setConst` reads it before writing it, as the anchors
of connect constraints are in ``eq_data``; otherwise they are recomputed all the same. :ref:`mjb_expanded` reads them
without that effect.

.. _mjb_setModels:

Run each simulation ``i`` on ``models[i]``, a model of the caller's, instead of the batch's model. If ``models`` is
``NULL``, return to the batch's model.

Each model must give mjData the same layout as the batch's model, that is, the same sizes for everything mjData holds.
Models compiled from one spec with different parameters qualify, and so do models whose meshes have other shapes but the
same numbers of vertices and faces. The batch only reads the models, so one model can serve several simulations. After
editing a model, call :ref:`mjb_setModels` again; otherwise a persistent batch may continue a step from stages computed
before the edit.

Caller models replace per-simulation fields, and the two cannot be combined: raise an error if the batch has fields from
:ref:`mjb_expand`. While caller models are set:

- :ref:`mjb_expand` and :ref:`mjb_setConst` raise an error.
- Keyframes are those of each simulation's model.
- :ref:`mjb_getModel` copies the simulation's model.

.. _mjb_rollout:

Advance each selected simulation by ``nstep`` calls to :ref:`mj_step`, as :ref:`rollout<PyRollout>` does, recording
after each substep. Return the number of simulations in which MuJoCo raised an error.

Before each substep, set the state components in ``control_spec`` (a subset of the :ref:`mjtState` bits in
``mjSTATE_USER``) from ``control``. ``control`` is ``n x nstep x mj_stateSize(m, control_spec)``, for the ``n`` selected
simulations in call order; if it is ``NULL``, the inputs are left as they are. After each substep, copy each of the
``nrecord`` :ref:`mjBatchRecord` |-| s in ``record`` into its buffer. A simulation whose warning counters rise stops
stepping, and its remaining records repeat its last ones.

.. _mjb_apply:

Run ``func`` (an :ref:`mjfBatchFunc`) on each selected simulation's model and data, on the batch's threads, passing it
the simulation's index and ``arg``. If ``save`` is nonzero, save the state back afterwards. Return the number of
simulations in which MuJoCo raised an error.

Fields registered with :ref:`mjb_output` are not refreshed; ``func`` returns its results through ``arg``. ``func`` may
call other batches, as long as they do not call back into this one; a call on its own batch raises an error in the
simulation. An error raised by a MuJoCo function that ``func`` calls unwinds to the batch with ``longjmp``, so ``func``
must not hold C++ objects with destructors across MuJoCo calls.

In a persistent batch:

- With ``save`` zero, ``func`` should not change the simulation's state. The mjData keeps what lies outside the state,
  such as the sleep bookkeeping, which would then no longer match the row.
- The mjData that ``func`` receives may hold the stages that :ref:`mjb_step` computed at the new state. Its contacts
  are then those of the new state, but its constraint forces are not yet computed, so do not read contact forces before
  the acceleration stage runs.

For example, this computes the finite-differenced transition matrices of every simulation at its current state, into
arrays indexed by simulation. :ref:`mjd_transitionFD` restores the state it perturbs, so ``save`` is zero:

.. code-block:: C

   // arrays of TransitionFD, nx = 2*nv + na
   struct TransitionArgs {
     mjtNum* A;  // nsim x nx x nx
     mjtNum* B;  // nsim x nx x nu
   };

   // mjb_apply function: transition matrices of simulation sim
   void TransitionFD(const mjModel* m, mjData* d, int sim, void* arg) {
     struct TransitionArgs* a = arg;
     int nx = 2*m->nv + m->na;
     mjd_transitionFD(m, d, 1e-6, /*flg_centered=*/0, a->A + sim*nx*nx, a->B + sim*nx*m->nu, NULL, NULL);
   }

   struct TransitionArgs args = {A, B};
   mjb_apply(b, /*ids=*/NULL, 0, TransitionFD, &args, /*save=*/0);

A function can also read one field of one simulation. The mjData it receives holds the simulation's state, but in a
batch that is not persistent its derived fields belong to another simulation, so the function computes what it reads.
In a persistent batch, it can instead read the fields that the simulation's last call left in its mjData:

.. code-block:: C

   // mjb_apply function: subtree centers of mass, into arg (3*nbody)
   void SubtreeCom(const mjModel* m, mjData* d, int sim, void* arg) {
     mj_kinematics(m, d);
     mj_comPos(m, d);
     mju_copy(arg, d->subtree_com, 3*m->nbody);
   }

   int sim = 7;
   mjb_apply(b, &sim, 1, SubtreeCom, com, /*save=*/0);

.. _mjb_getModel:

Write simulation ``sim``'s model into ``m``, ready to render or to pass to any function. Normally, this is the batch's
model with the simulation's per-simulation fields and the constants :ref:`mjb_setConst` computed for it; ``m`` must then
have the batch's sizes, such as a copy of :ref:`mjb_model`. With caller models from :ref:`mjb_setModels`, it is a copy
of the simulation's model, and ``m`` must have that model's sizes. Raise an error if ``sim`` is out of range or the
sizes differ.

.. _mjb_getData:

Write simulation ``sim``'s state and warning counters into ``d``, including writes to its row since its last call;
``d`` must have the batch's sizes. In a persistent batch, first copy the simulation's mjData, so that ``d`` also holds
its derived fields and sleep state. Otherwise, run :ref:`mj_forward` with the model from :ref:`mjb_getModel` to compute
them. Raise an error if ``sim`` is out of range or the sizes differ.

.. _mjb_model:

Return the batch's copy of the model, made by :ref:`mjb_makeBatch`. Per-simulation fields and caller models never
change it; :ref:`mjb_getModel` gives a simulation's own model.

.. _Standardmath:

The "functions" in this section are preprocessor macros replaced with the corresponding C standard library math
functions. When MuJoCo is compiled with single precision (which is not currently available to the public, but we
sometimes use it internally) these macros are replaced with the corresponding single-precision functions (not shown
here). So one can think of them as having inputs and outputs of type mjtNum, where mjtNum is defined as double or float
depending on how MuJoCo is compiled. We will not document these functions here; see the C standard library
specification.

mju_sqrt
~~~~~~~~

.. code-block:: C

   #define mju_sqrt    sqrt

mju_exp
~~~~~~~

.. code-block:: C

   #define mju_exp     exp

mju_sin
~~~~~~~

.. code-block:: C

   #define mju_sin     sin

mju_cos
~~~~~~~

.. code-block:: C

   #define mju_cos     cos

mju_tan
~~~~~~~

.. code-block:: C

   #define mju_tan     tan

mju_asin
~~~~~~~~

.. code-block:: C

   #define mju_asin    asin

mju_acos
~~~~~~~~

.. code-block:: C

   #define mju_acos    acos

mju_atan2
~~~~~~~~~

.. code-block:: C

   #define mju_atan2   atan2

mju_tanh
~~~~~~~~

.. code-block:: C

   #define mju_tanh    tanh

mju_pow
~~~~~~~

.. code-block:: C

   #define mju_pow     pow

mju_abs
~~~~~~~

.. code-block:: C

   #define mju_abs     fabs

mju_log
~~~~~~~

.. code-block:: C

   #define mju_log     log

mju_log10
~~~~~~~~~

.. code-block:: C

   #define mju_log10   log10

mju_floor
~~~~~~~~~

.. code-block:: C

   #define mju_floor   floor

mju_ceil
~~~~~~~~

.. code-block:: C

   #define mju_ceil    ceil

.. _Vectormath:

.. _Quaternions:

.. _Poses:

.. _Decompositions:

.. _mju_cholFactorBand:

Band-dense Cholesky decomposition.
|br| Add ``diagadd + diagmul*mat_ii`` to diagonal before decomposition.
|br| Returns the minimum value of the factorized diagonal or 0 if rank-deficient.

   **Symmetric band-dense matrices**

   :ref:`mju_cholFactorBand` and subsequent functions containing the substring "band" operate on matrices which are a
   generalization of symmetric `band matrices <https://en.wikipedia.org/wiki/Band_matrix>`_. *Symmetric band-dense* or
   "arrowhead" matrices have non-zeros along proximal diagonal bands and dense blocks on the bottom rows and right
   columns. These matrices have the property that Cholesky factorization creates no fill-in and can therefore be
   performed efficiently in-place. Matrix structure is defined by three integers:

   - ``ntotal``: the number of rows (columns) of the symmetric matrix.
   - ``nband``: the number of bands under (over) the diagonal, inclusive of the diagonal.
   - ``ndense``: the number of dense rows (columns) at the bottom (right).

   The non-zeros are stored in memory as two contiguous row-major blocks, colored green and blue in the illustration
   below. The first block has size ``nband x (ntotal-ndense)`` and contains the diagonal and the bands below it. The
   second block has size ``ndense x ntotal`` and contains the dense part. Total required memory is the sum of the block
   sizes.

   .. figure:: /images/APIreference/arrowhead.svg
      :width: 750px
      :align: left

   For example, consider an arrowhead matrix with ``nband = 3``, ``ndense = 2`` and ``ntotal = 8``. In this example, the
   total memory required is ``3*(8-2) + 2*8 = 34`` mjtNum's, laid out as follows:

   .. code-block::

      0   1   2
          3   4   5
              6   7   8
                  9   10  11
                      12  13  14
                          15  16  17
              18  19  20  21  22  23  24  25
              26  27  28  29  30  31  32  33


   The diagonal elements are ``2, 5, 8, 11, 14, 17, 24, 33``.
   |br| Elements ``0, 1, 3, 25`` are present in memory but never touched.

.. _mju_boxQP:

Minimize :math:`\tfrac{1}{2} x^T H x + x^T g \quad \text{s.t.} \quad l \le x \le u`, return rank or -1 if failed.

inputs:
  ``n``           - problem dimension

  ``H``           - SPD matrix                ``n*n``

  ``g``           - bias vector               ``n``

  ``lower``       - lower bounds              ``n``

  ``upper``       - upper bounds              ``n``

  ``res``         - solution warmstart        ``n``

return value:
  ``nfree <= n``  - rank of unconstrained subspace, -1 if failure

outputs (required):
  ``res``         - solution                  ``n``

  ``R``           - subspace Cholesky factor  ``nfree*nfree``,    allocated: ``n*(n+7)``

outputs (optional):
  ``index``       - set of free dimensions    ``nfree``,          allocated: ``n``

notes:
  The initial value of ``res`` is used to warmstart the solver.
  ``R`` must have allocated size ``n*(n+7)``, but only ``nfree*nfree`` values are used as output.
  ``index`` (if given) must have allocated size ``n``, but only ``nfree`` values are used as output.
  The convenience function :ref:`mju_boxQPmalloc` allocates the required data structures.
  Only the lower triangles of H and R are read from and written to, respectively.

.. _mju_boxQPmalloc:

Allocate heap memory for box-constrained Quadratic Program.
As in :ref:`mju_boxQP`, ``index``, ``lower``, and ``upper`` are optional.
Free all pointers with ``mju_free()``.

.. _mju_symmetrize:

Symmetrize square matrix :math:`R = \frac{1}{2}(M + M^T)`.

.. _Miscellaneous:

.. _mju_sigmoid:

Twice continuously differentiable sigmoid function using a quintic polynomial:

.. math::
   s(x) =
   \begin{cases}
      0,                    &       & x \le 0  \\
      6x^5 - 15x^4 + 10x^3, & 0 \lt & x \lt 1  \\
      1,                    & 1 \le & x \qquad
   \end{cases}

.. _Derivatives-api:

The functions below provide useful derivatives of various functions, both analytic and
finite-differenced. The latter have names with the suffix ``FD``. Note that unlike much of the API,
outputs of derivative functions are the trailing rather than leading arguments.

.. _mjd_transitionFD:

Compute finite-differenced discrete-time transition matrices.

Letting :math:`x, u` denote the current :ref:`state<siPhysicsState>` and :ref:`control<siInput>`
vector in an mjData instance, and letting :math:`y, s` denote the next state and sensor
values, the top-level :ref:`mj_step` function computes :math:`(x,u) \rightarrow (y,s)`
:ref:`mjd_transitionFD` computes the four associated Jacobians using finite-differencing.
These matrices and their dimensions are:

.. csv-table::
   :header: "matrix", "Jacobian", "dimension"
   :widths: auto
   :align: left

   ``A``, :math:`\partial y / \partial x`, ``2*nv+na x 2*nv+na``
   ``B``, :math:`\partial y / \partial u`, ``2*nv+na x nu``
   ``C``, :math:`\partial s / \partial x`, ``nsensordata x 2*nv+na``
   ``D``, :math:`\partial s / \partial u`, ``nsensordata x nu``

- All outputs are optional (can be NULL).
- ``eps`` is the finite-differencing epsilon.
- ``flg_centered`` denotes whether to use forward (0) or centered (1) differences.
- The Runge-Kutta integrator (:ref:`mjINT_RK4<mjtIntegrator>`) is not supported.
- :ref:`Sleeping<Sleeping>` is not supported. Disable the :ref:`sleep<option-flag-sleep>` flag before calling.
- :ref:`Delays<CDelay>` are not supported.

.. admonition:: Improving speed and accuracy
   :class: tip

   warmstart
     If warm-starts are not :ref:`disabled<option-flag-warmstart>`, the warm-start accelerations
     ``mjData.qacc_warmstart`` which are present at call-time are loaded at the start of every relevant pipeline call,
     to preserve determinism. If solver computations are an expensive part of the simulation, the following trick can
     lead to significant speed-ups: First call :ref:`mj_forward` to let the solver converge, then reduce :ref:`solver
     iterations<option-iterations>` significantly, then call :ref:`mjd_transitionFD`, finally, restore the original
     value of :ref:`iterations<option-iterations>`. Because we are already near the solution, few iteration are required
     to find the new minimum. This is especially true for the :ref:`Newton<option-solver>` solver, where the required
     number of iteration for convergence near the minimum can be as low as 1.

   tolerance
      Accuracy can be improved if solver :ref:`tolerance<option-tolerance>` is set to 0. This means that all calls to
      the solver will perform exactly the same number of iterations, preventing numerical errors due to early
      termination. Of course, this means that :ref:`solver iterations<option-iterations>` should be small, to not tread
      water at the minimum. This method and the one described above can and should be combined.

*Nullable:* ``A``, ``B``, ``D``, ``C``

.. _mjd_inverseFD:

Finite differenced continuous-time inverse-dynamics Jacobians.

Letting :math:`x, a` denote the current :ref:`state<siPhysicsState>` and acceleration vectors in an mjData instance, and
letting :math:`f, s` denote the forces computed by the inverse dynamics (``qfrc_inverse``), the function
:ref:`mj_inverse` computes :math:`(x,a) \rightarrow (f,s)`. :ref:`mjd_inverseFD` computes seven associated Jacobians
using finite-differencing. These matrices and their dimensions are:

.. csv-table::
   :header: "matrix", "Jacobian", "dimension"
   :widths: auto
   :align: left

   ``DfDq``, :math:`\partial f / \partial q`, ``nv x nv``
   ``DfDv``, :math:`\partial f / \partial v`, ``nv x nv``
   ``DfDa``, :math:`\partial f / \partial a`, ``nv x nv``
   ``DsDq``, :math:`\partial s / \partial q`, ``nv x nsensordata``
   ``DsDv``, :math:`\partial s / \partial v`, ``nv x nsensordata``
   ``DsDa``, :math:`\partial s / \partial a`, ``nv x nsensordata``
   ``DmDq``, :math:`\partial M / \partial q`, ``nv x nC``

- All outputs are optional (can be NULL).
- All outputs are transposed relative to Control Theory convention (i.e., column major).
- ``DmDq``, which contains a sparse representation of the ``nv x nv x nv`` tensor :math:`\partial M / \partial q`, is
  not strictly an inverse dynamics Jacobian but is useful in related applications. It is provided as a convenience to
  the user, since the required values are already computed if either of the other two :math:`\partial / \partial q`
  Jacobians are requested.
- ``eps`` is the (forward) finite-differencing epsilon.
- ``flg_actuation`` denotes whether to subtract actuation forces (``qfrc_actuator``) from the output of the inverse
  dynamics. If this flag is positive, actuator forces are not considered as external.
- The model option flag ``invdiscrete`` should correspond to the representation of ``mjData.qacc`` in order to compute
  the correct derivative information.

.. attention::
   - The Runge-Kutta 4th-order integrator (``mjINT_RK4``) is not supported.
   - The noslip solver is not supported.
   - :ref:`Sleeping<Sleeping>` is not supported. Disable the :ref:`sleep<option-flag-sleep>` flag before calling.

*Nullable:* ``DfDq``, ``DfDv``, ``DfDa``, ``DsDq``, ``DsDv``, ``DsDa``, ``DmDq``

.. _mjd_subQuat:

Derivatives of :ref:`mju_subQuat` (quaternion difference).

*Nullable:* ``Da``, ``Db``

.. _mjd_quatIntegrate:

Derivatives of :ref:`mju_quatIntegrate`.

:math:`{\tt \small mju\_quatIntegrate}(q, v, h)` performs the in-place rotation :math:`q \leftarrow q + v h`,
where :math:`q \in \mathbf{S}^3` is a unit quaternion, :math:`v \in \mathbf{R}^3` is a 3D angular velocity and
:math:`h \in \mathbf{R^+}` is a timestep. This is equivalent to :math:`{\tt \small mju\_quatIntegrate}(q, s, 1.0)`,
where :math:`s` is the scaled velocity :math:`s = h v`.

:math:`{\tt \small mjd\_quatIntegrate}(v, h, D_q, D_v, D_h)` computes the Jacobians of the output :math:`q` with respect
to the inputs. Below, :math:`\bar q` denotes the pre-modified quaternion:

.. math::
   \begin{aligned}
      D_q &= \partial q / \partial \bar q \\
      D_v &= \partial q / \partial v \\
      D_h &= \partial q / \partial h
   \end{aligned}

Note that derivatives depend only on :math:`h` and :math:`v` (in fact, on :math:`s = h v`).
All outputs are optional.

*Nullable:* ``Dquat``, ``Dvel``, ``Dscale``

.. _mjs_delete:

Remove object corresponding to the given element; return 0 on success. Deleting a body or a frame also deletes
everything inside it, along with all the elements that reference a deleted element. A
:ref:`plugin instance<plugin-instance>` is deleted along with the last element that references it.

.. _mjs_fuseStatic:

Fuse the static bodies of the spec with their parents. A body without joints is replaced by a :ref:`frame<frame>` in
its parent, which has the pose of the body and holds its geoms, sites, cameras, lights, frames and child bodies, in the
same coordinates as before. The inertia of the body is added to that of the parent. The compiled model has the same
kinematics and dynamics, with fewer bodies. Bodies which cannot be fused without changing the model are kept, see
:ref:`fusestatic<compiler-fusestatic>`, which applies this function before compiling. Pointers to the bodies which were
fused are no longer valid. Assets are read as in :ref:`mj_compile`. Returns the number of bodies fused, 0 if the spec
is unchanged, or -1 on error, in which case nothing is fused.

.. _mjs_discardVisual:

Discard the elements of the spec which are only visual: materials and textures, the geoms which do not collide, and
the meshes which are then not used; an element which another refers to by name is kept. If a discarded geom was used for
inferring the inertia of its body, that inertia becomes the explicit inertial of the body, as in
:ref:`mjs_adoptInertial`, so the compiled model has the same dynamics. See
:ref:`discardvisual<compiler-discardvisual>`, which applies this function before compiling. Pointers to the elements
which were discarded are no longer valid. Assets are read as in :ref:`mj_compile`, and only if inertia is inferred from
a discarded geom. Returns the number of elements discarded, 0 if the spec is unchanged, or -1 on error, in which case
nothing is discarded.

.. _mjs_dedupAssets:

Merge the textures and meshes of the spec which would compile to the same asset: textures with the same resolved file
or buffer and the same parameters, meshes with the same resolved file or arrays and the same parameters, names
excluded. The first asset of each group keeps its name, every reference to the others (materials, lights, geoms, sites,
default classes, sensors, tuples) is rewritten to it, and the others are deleted as by :ref:`mjs_delete`, so pointers
to them are no longer valid. An unnamed asset is left alone, as is a mesh with a plugin. Nothing is read from disk: a
model which attaches the same sub-model several times is merged without its asset files. Returns the number of assets
merged, 0 if the spec is unchanged, or -1 on error, in which case nothing is merged.

.. _mjs_adoptInertial:

Make the inertial which compilation infers for a body part of the spec. The mass, inertia and inertial frame which the
compiler calculates from the geoms of the body, as the spec is now, are written as the explicit inertial of the body,
so that later changes to its geoms no longer affect them. Compiling before and after gives the same model: the values
are those before the scaling of :ref:`settotalmass<compiler-settotalmass>`. Nothing is done if the inertial of the body
is not inferred. It is an error when :ref:`inertiafromgeom<compiler-inertiafromgeom>` is "true", which infers the
inertia of every body whatever its inertial. Assets are read as in :ref:`mj_compile`. Returns 0 on success.

To read an inferred inertial without adopting it, compile and look it up in :ref:`mjModel` with :ref:`mjs_getId`.

.. _mjs_makeFlex:

Create a :ref:`flexcomp<body-flexcomp>` in ``body`` and return the resulting flex; the arguments correspond to the
flexcomp attributes. If ``file`` is the name of an existing :ref:`mesh asset<asset-mesh>`, the flexcomp uses that mesh as
with the :ref:`mesh<body-flexcomp-mesh>` attribute, otherwise ``file`` is the name of a mesh file. The flexcomp is
expanded immediately, so later changes to the mesh asset do not affect the created flex.

.. _mjs_makeMesh:

Generate a built-in procedural mesh of type :ref:`mjtMeshBuiltin` with the given parameters. Returns 0 on success; on
failure, returns a nonzero value and sets the error of the spec. The parameters of the sphere, hemisphere, cone,
supersphere, supertorus, wedge and plate types are those of the corresponding :ref:`mesh/builtin<asset-mesh-builtin>`
attribute. The procedural types below generate the same shapes as the corresponding :ref:`flexcomp<body-flexcomp>`
types; their nodes are stored in double precision in ``usernode``.

- ``mjMESH_BUILTIN_GRID`` (count[3], spacing[3], [dim]): A regular grid with ``count[i] > 0`` points along each axis,
  separated by ``spacing[i]``. ``dim`` (default 3) creates tetrahedra (3) or triangles (2);
  ``count`` must be 1 along the axes beyond ``dim``.
- ``mjMESH_BUILTIN_BOX``, ``mjMESH_BUILTIN_CYLINDER``, ``mjMESH_BUILTIN_ELLIPSOID`` (count[3], spacing[3], [dim]): The
  surface of a box grid with ``count[i] >= 2`` points along each axis, projected onto the shape. ``dim`` (default 3)
  creates tetrahedra connecting the surface to the center (3) or surface triangles (2).
- ``mjMESH_BUILTIN_SQUARE``, ``mjMESH_BUILTIN_DISC`` (count[3], spacing[3]): A triangulated square, or a disc obtained by
  radial projection of the square; ``count[2]`` must be 1.

Types without volume (square, disc, and
grid, box, cylinder and ellipsoid with ``dim`` < 3) change a ``legacy`` :ref:`inertia<asset-mesh-inertia>` to
``shell``.
