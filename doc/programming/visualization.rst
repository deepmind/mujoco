.. _Visualization:
.. _Abstract:

Visualization
-------------

MuJoCo has a native 3D visualization layer that is independent of any graphics library. It provides interactive camera
control, object picking via raycasting, mouse-driven force and pose perturbations, visualization options, and automatic
generation of decorative and diagnostic geometry such as contact points and forces, equivalent inertia boxes, convex
hulls, kinematic trees, constraint violations, spatial frames, and text labels. These elements provide insight into the
physics simulation and help fine-tune models across both interactive viewers (:ref:`MuJoCo Studio <Studio>`,
:ref:`simulate.cc <saSimulate>`) and offscreen rendering pipelines (:ref:`render.cc <saRender>`,
:ref:`record.cc <saRecord>`).

All data structures and functions in the abstract visualization layer have the prefix ``mjv`` and are declared in
``mjvisualize.h``:

- :ref:`mjvCamera` and :ref:`mjvGLCamera` represent high-level interactive cameras and low-level frustum cameras.
- :ref:`mjvOption` controls which element categories, geom groups, and diagnostic overlays are visible.
- :ref:`mjvPerturb` tracks interactive object selection and mouse-applied forces and torques.
- :ref:`mjvScene` and :ref:`mjvGeom` hold a complete per-frame geometric description of the 3D scene and its decorations.

Separating abstract visualization from :ref:`rendering <Rendering>` keeps interaction and scene logic independent of
graphics APIs and allows multiple rendering backends to share the same camera, selection, perturbation, and option
mechanisms. The :ref:`classic OpenGL renderer <ClassicRendering>` consumes the entire :ref:`mjvScene` on every frame,
while the :ref:`Filament renderer <FilamentRendering>` updates its own persistent scene from :ref:`mjData` and uses
``mjvScene`` only for transient decorations; see the :ref:`Rendering <Rendering>` chapter for a side-by-side comparison.
Third-party renderers can also consume ``mjvScene`` directly.

.. _viCamera:

Cameras
~~~~~~~

There are two types of camera objects: an abstract camera represented with the stand-alone data structure
:ref:`mjvCamera`, and a low-level camera represented with the data structure :ref:`mjvGLCamera` (also aliased as
``mjrCamera``) which is embedded in :ref:`mjvScene` or computed directly via :ref:`mjv_camera2GLCamera`. When present,
the abstract camera is used to automatically compute the low-level camera parameters consumed by the renderer.
Alternatively, the user can bypass the abstract camera mechanism and set the low-level camera parameters directly, as
discussed in the :ref:`Virtual reality <viVR>` section below.

The abstract camera can represent three different camera types as determined by mjvCamera.type. The possible settings
are defined by the enum mjtCamera:

mjCAMERA_FREE
   This is the most commonly used abstract camera. It can be freely moved with the mouse. It has a lookat point,
   distance to the lookat point, azimuth, and elevation; twist around the line of sight is not allowed. The function
   :ref:`mjv_moveCamera` is a mouse hook for controlling all these camera properties interactively with the mouse.
   When :ref:`simulate.cc <saSimulate>` first starts, it uses the free camera.
mjCAMERA_TRACKING
   This is similar to the free camera, except the lookat point is no longer a free parameter but instead is coupled to
   the MuJoCo body whose id is given by mjvCamera.trackbodyid. At each update, the lookat point is set to the center of
   mass of the kinematic subtree rooted at the specified body. There is also some filtering which produces smooth camera
   motion. The distance, azimuth, and elevation are controlled by the user and are not modified automatically. This is
   useful for tracking a body as it moves around, without turning the camera. To switch from the free to the tracking
   camera in :ref:`simulate.cc <saSimulate>`, hold Ctrl and right-double-click on the body of interest. Press Esc to go
   back to the free camera.
mjCAMERA_FIXED
   This refers to a camera explicitly defined in the model, unlike the free and tracking cameras which only exist in the
   visualizer and are not defined in the model. The id of the model camera is given by mjvCamera.fixedcamid. This camera
   is fixed in the sense that the visualizer cannot change its pose or any other parameters. However the simulator
   computes the camera pose at each time step, and if the camera is attached to a moving body or is in tracking or
   targeting mode, it will move.
mjCAMERA_USER
   This means that the abstract camera is ignored during an update and the low-level OpenGL cameras are not changed. It
   is equivalent to not specifying an abstract camera at all, i.e., passing a NULL pointer to mjvCamera in the update
   functions explained below.

The low-level mjvGLCamera is what determines the actual rendering. There are two such cameras embedded in mjvScene, one
for each eye. Each has position, forward and up directions. Forward corresponds to the negative Z axis of the camera
frame, while up corresponds to the positive Y axis. There is also a frustum in the sense of OpenGL, except we store the
average of the left and right frustum edges and then during rendering compute the actual edges from the viewport aspect
ratio assuming 1:1 pixel aspect ratio. The distance between the two camera positions corresponds to the inter-pupillary
distance (ipd). When the low-level camera parameters are computed automatically from an abstract camera, the ipd as well
as vertical field of view (fovy) are taken from ``mjModel.vis.global.ipd``/``fovy`` for free and tracking cameras, and
from the camera-specific ``mjModel.cam_ipd/fovy`` for cameras defined in the model. When stereoscopic mode is not
enabled, as determined by mjvScene.stereo, the camera data for the two eyes are internally averaged during rendering.

.. _viSelect:

Selection
~~~~~~~~~

In many applications we need to click on a point and determine the 3D object to which this point/pixel belongs. This is
done with the function :ref:`mjv_select`, which uses :ref:`ray collisions <Raycollisions>`. Ray collisions functionality
is engine-level and does not depend on the visualizer (indeed it is also used to simulate :ref:`rangefinder
<sensor-rangefinder>` sensors independent of visualization), but the select function is implemented in the visualizer
because it needs information about the camera and viewport.

The function mjv_select returns the index of the geom at the specified window coordinates, or -1 if there is no geom
at those coordinates. The 3D position is also returned. See the code sample :ref:`simulate.cc <saSimulate>` for an
example of how to use this function. Internally, mjv_select calls the engine-level function :ref:`mj_ray` which in turn
calls the per-geom functions :ref:`mj_rayMesh`, :ref:`mj_rayHfield`, and :ref:`mju_rayGeom`. The user can implement
custom selection mechanisms by calling these functions directly. In a VR application for example, it would make sense to
use the hand-held controller as a "laser pointer" that can select objects.

.. _viPerturb:

Perturbations
~~~~~~~~~~~~~

Interactive perturbations have proven very useful in exploring the model dynamics as well as probing closed-loop control
systems. The user is free to implement any perturbation mechanism of their choice by setting ``mjData.qfrc_applied`` or
``mjData.xfrc_applied`` to suitable forces (in generalized and Cartesian coordinates respectively).

All objects needed to implement interactive perturbations are grouped into the data structure :ref:`mjvPerturb`.
Its use is illustrated in :ref:`simulate.cc <saSimulate>`.
The idea is to select a MuJoCo body of interest, and provide a reference pose (i.e., a 3D position and quaternion
orientation) for that body. These are stored in mjPerturb.refpos/quat. The function :ref:`mjv_movePerturb` is a mouse
hook for controlling the reference pose with the mouse. The function :ref:`mjv_initPerturb` is used to set the
reference pose equal to the selected body pose at the onset of perturbation, so as to avoid jumps.

This perturbation object can then be used to move the selected body directly (when the simulation is paused or when the
selected body is a mocap body), or to apply forces and torques to the body. This is done with the functions
:ref:`mjv_applyPerturbPose` and :ref:`mjv_applyPerturbForce` respectively. The latter function writes the external
perturbation force to ``mjData.xfrc_applied`` for the selected body. However it does not clear ``mjData.xfrc_applied``
for the remaining bodies, thus it is recommended to clear it in user code, in case the selected body changed and some
perturbation force was left over from a previous time step. If there is more than one device that can apply
perturbations or user code needs to add perturbations from other sources, the user must implement the necessary logic so
that only the desired perturbations are present in ``mjData.xfrc_applied`` and any old perturbations are cleared.

In addition to affecting the simulation, the perturbation object is recognized by the abstract visualizer and can be
rendered. This is done by adding a visual string to denote the positional difference, and a rotating cube to denote
the reference orientation of the selected body. The perturbation forces themselves can also be rendered when the
corresponding visualization flag in :ref:`mjvOption` is enabled.

.. _viOption:

Visualization options
~~~~~~~~~~~~~~~~~~~~~

The data structure :ref:`mjvOption` controls what is shown, independently of how it is rendered. It is consumed by
:ref:`mjv_updateScene` when populating ``mjvScene`` and read directly by the :ref:`Filament renderer
<FilamentRendering>` to set the visibility of its persistent objects. Its fields are:

- ``flags``, indexed by :ref:`mjtVisFlag`, enable the decorative and diagnostic elements: contact points and forces,
  joint and actuator axes, constraint violations, inertia boxes, convex hulls, skins, tendons, perturbation forces,
  and so on.
- ``geomgroup``, ``sitegroup``, ``jointgroup``, ``tendongroup``, ``actuatorgroup``, ``flexgroup`` and ``skingroup``
  toggle the visibility of model elements by their :ref:`group<body-geom-group>` attribute.
- ``label`` (:ref:`mjtLabel`) and ``frame`` (:ref:`mjtFrame`) select which category of objects is labeled with text and
  which spatial frames are drawn.
- ``bvh_depth`` and ``flex_layer`` select which level of the bounding volume hierarchy and which layer of 3D flex
  elements are shown, respectively.

The function :ref:`mjv_defaultOption` sets all fields to their defaults, in which groups 0, 1 and 2 are visible and
flags take the defaults listed in :ref:`mjVISSTRING`.

.. _viUpdate:

Scene update
~~~~~~~~~~~~

Finally, we bring all of the above elements together and explain how :ref:`mjvScene` is updated. In immediate-mode
renderers such as the classic :ref:`OpenGL renderer <ClassicRendering>`, ``mjvScene`` is the primary input to the
rendering stage; in retained-mode renderers such as :ref:`Filament <FilamentRendering>`, model geometry is updated
directly from ``mjData`` and ``mjvScene`` is used only for decorative elements (``mjCAT_DECOR``).

Updating ``mjvScene`` can be done with a single call to the function :ref:`mjv_updateScene` at each frame.
mjvCamera and mjvPerturb are arguments to this function, or they can be NULL pointers in which case the corresponding
functionality is disabled. In VR applications the parameters of mjvScene.camera[n], n=0,1 must also be set at each
frame; this is done by user code outside mjv_updateScene. The function mjv_updateScene examines mjModel and mjData,
constructs all geoms that need to be rendered (according to the specified visualization options), and populates the
array mjvScene.geoms with :ref:`mjvGeom` objects. Note that mjvGeom is an abstract geom, and is not in one-to-one
correspondence with the simulation geoms in mjModel and mjData. In particular, mjvGeom contains the geom pose, scale,
shape (primitive or mesh index in mjModel), material properties, textures (index in mjModel), labeling, and everything
else needed to specify how rendering should be done. mjvScene also contains up to ``mjMAXLIGHT`` lights copied from the
model, as well as a headlight which is in light position 0 when present.

The above procedure is the most common approach, and it updates the entire scene at each frame. In addition, we
provide two functions for finer control. :ref:`mjv_updateCamera` updates only the camera (i.e., maps the abstract
mjvCamera to the low-level mjvGLCamera) but does not touch the geoms or lights; alternatively,
:ref:`mjv_camera2GLCamera` returns a single ``mjvGLCamera`` directly without requiring an ``mjvScene``. This is useful
when the user is moving the camera rapidly while the simulation is paused, or when using the Filament renderer where
camera parameters are passed via :ref:`mjrfRenderRequest`.

More advanced rendering effects can be achieved by manipulating the list of abstract geoms. For example, the user can
add custom geoms at the end of the list. Sometimes it is desirable to render a sequence of simulation states (i.e., a
trajectory) and not just the current state. For this purpose, we have provided the function :ref:`mjv_addGeoms` which
adds the geoms corresponding to the current simulation state to the list already in mjvScene. It does not change the
list of lights, because lighting is additive and having too many lights will make the scene too bright. Importantly,
the user can select which geom categories will be added, via a bitmask of enum type mjtCatBit:

mjCAT_STATIC
   This selects MuJoCo geoms and sites belonging to the world body (which has body id 0).
mjCAT_DYNAMIC
   This selects MuJoCo geoms and sites belonging to bodies other than the world body.
mjCAT_DECOR
   This selects decorative elements such as force arrows, automatically-generated skeletons, equivalent inertia boxes,
   and any other elements that were added by the abstract visualizer and do not correspond to MuJoCo geoms and sites
   defined in the model.
mjCAT_ALL
   This selects all of the above categories.

The main update function mjv_updateScene would normally be called with mjCAT_ALL. It clears the geom list and calls
mjv_addGeoms to add only the geoms for the current model state. If we want to render a trajectory, we have to be careful
to avoid visual clutter. So it makes sense to render one of the frames with mjCAT_ALL (usually the first or the last
depending on the use case), and all other frames with mjCAT_DYNAMIC. Since the static/world objects are not moving,
rendering them in each frame will only slow down the GPU and create visual aliasing. As for the decor elements, there
could be situations where we want to render all of them - for example to visualize the evolution of contact forces over
time. In summary, there is plenty of flexibility in how mjvScene is constructed. We have provided automation for the
main use cases, but the user can also make programmatic changes as needed.

.. _viVR:

Virtual reality
~~~~~~~~~~~~~~~

In desktop applications it is convenient to use an abstract mjvCamera allowing intuitive mouse control, and then
automatically map it to mjvGLCamera used for rendering. In VR applications the situation is very different. In that
case the head/eyes of the user as well as the projection surface are being tracked, and therefore have physical
presence in the room. If anything can be moved by the user (with a mouse or other input device) it is the position,
orientation and scale of the model relative to the room. This is called model transformation, and is represented in
mjvScene. The function :ref:`mjv_moveModel` is a mouse hook for controlling this transformation. When using an
abstract mjvCamera during update, the model transformation is automatically disabled, by setting the flag
mjvScene.enabletransform = 0 rather than clearing the actual parameters. In this way the user can switch between VR
and desktop camera mode without losing the model transformation parameters.

Since we have introduced two spaces, namely model space and room space, we need to map between them as well as clarify
which spatial quantities are defined with respect to which spatial frame. Everything accessible by the simulator lives
in the model space. The room space is only accessible by the visualizer. The only quantities defined in room space are
the mjvGLCamera parameters. The functions :ref:`mjv_room2model`, :ref:`mjv_model2room`, :ref:`mjv_cameraInModel`,
and :ref:`mjv_cameraInRoom` perform the necessary transformations, and are needed for VR applications.

While MuJoCo does not provide a built-in VR application, it provides data structures and functions to support VR
integration in user code.

**Head tracking and cameras**
   In a typical VR application, a tracking device provides the positions and orientations of the user's eyes in
   real-time. These data can be copied directly into the two ``mjvGLCamera`` structures in ``mjvScene.camera[n]``
   (where ``n=0`` is the left eye and ``n=1`` is the right eye). The ``mjvGLCamera`` frustum parameters must also be
   set according to the physical characteristics of the tracked display.

**Controllers and mocap bodies**
   Hand-held spatial controllers are also tracked in room space. The function :ref:`mjv_room2model` can map these
   poses to model space. Once in model space, the controller poses can be used to update the position of MuJoCo
   *mocap bodies*. Mocap bodies are treated as fixed from the viewpoint of physics, yet the user is expected to move
   them programmatically at each simulation step. They can interact with the simulation through contacts, or better
   yet, through soft equality constraints to regular bodies which in turn make contacts. This provides effective
   dynamic filtering and avoids contacts involving bodies that behave as if they are infinitely heavy. The
   time-varying positions and orientations of the mocap bodies are stored in ``mjData.mocap_pos`` and
   ``mjData.mocap_quat``.
