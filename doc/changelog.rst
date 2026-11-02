=========
Changelog
=========

Upcoming Version (not yet released)
-----------------------------------

Significant new features
^^^^^^^^^^^^^^^^^^^^^^^^
- Added the :ref:`speculative<option-flag-speculative>` enable flag for :doc:`speculative contacts
  <computation/speculative>` to prevent fast-moving convex geoms from tunneling through thin obstacles within a single
  timestep.

.. image:: https://www.gstatic.com/mujoco/doc/images/changelog/wire.gif
   :align: right
   :width: 40%
   :target: https://github.com/google-deepmind/mujoco/blob/main/model/plastic/wire.xml

- Added ideal elasto-plasticity for joint (:ref:`springyield<body-joint-springyield>`) and tendon
  (:ref:`spatial<tendon-spatial-springyield>`, :ref:`fixed<tendon-fixed-springyield>`) springs, with stateful plastic
  spring references in ``mjData.plastic`` (``mjSTATE_PLASTIC``).

.. image:: https://www.gstatic.com/mujoco/doc/images/changelog/clay.gif
   :align: right
   :width: 40%
   :target: https://github.com/google-deepmind/mujoco/blob/main/model/flex/clay.xml

- Added continuum elasto-plasticity (:ref:`yield<flex-elasticity-yield>`) for 2D and 3D
  :ref:`flexes<deformable-flex>`, with volumetric-deviatoric (von Mises :math:`J_2`) stretching and 2D shell bending
  return mapping in ``mjData.plastic``.

.. container:: custom-clear

   .. raw:: html

      <div style="clear: both;"></div>

General
^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   - Flexes now support per-element visual materials: ``mjModel.flex_matid`` is now ``(nflexelem x 1)`` instead of
     ``(nflex x 1)``, and the ``material`` attribute on :ref:`flex<deformable-flex>` and :ref:`flexcomp<body-flexcomp>`
     (``mjsFlex.material``) is now ``mjStringVec`` rather than ``mjString`` (accepting either a single material name
     broadcast to all elements or one material name per element).

     **Migration:** Index ``mjModel.flex_matid`` by element ``m->flex_elemadr[f] + e`` rather than by flex ``f``. In C,
     set ``mjsFlex.material`` with ``mjs_setStringVec`` (or ``mjs_appendString``) instead of ``mjs_setString``; in
     Python, assign a list of material names (e.g. ``flex.material = ['mat']``).

   - Removed ``flexvert`` equality constraints (``<equality><flexvert>`` and ``<edge equality="vert"/>`` in
     :ref:`flexcomp<body-flexcomp>`), along with the associated :ref:`mjModel` fields (``nJfv``, ``flex_vertedgeadr``,
     ``flex_vertedgenum``, ``flex_vertedge``, ``flex_vertmetric``, ``flexvert_J_rownnz``, ``flexvert_J_rowadr``,
     ``flexvert_J_colind``) and :ref:`mjData` fields (``flexvert_length``, ``flexvert_J``). ``mjEQ_FLEXVERT`` is
     retained as an unsupported placeholder in :ref:`mjtEq` and causes an error if used.

     **Migration:** Use continuum :ref:`elasticity<flex-elasticity>` with ``integrator="discrete"`` or
     :ref:`edge equality<flexcomp-edge-equality>` for 2D cloth models.

   - Unpinned :ref:`flexcomp<body-flexcomp>` models (with **full**, **trilinear**, or **quadratic**
     :ref:`dofs<body-flexcomp-dof>`) now fail to compile if their parent or ancestor body has joints when no active
     vertices or nodes are :ref:`pinned<flexcomp-pin>` to it.

     **Migration:** Remove joints from the parent/ancestor body of an unpinned flex, or pin at least one vertex/node to
     the parent body with :ref:`pin<flexcomp-pin>`.

   - Records of 1D flexes in ``mjModel.flex_bending`` grow from 4 to 10 values, to store curved reference
     configurations and pairs of adjacent bent vertices. Previously, each record was ``[v_prev, v, v_next, k]``.

     **Migration:** Read 1D records with stride 10. A record whose fourth value is negative is a single hinge
     ``[v0, v1, v2, -1, k, 0, |kb0|, 0, 0, 0]``; otherwise it is a hinge pair ``[v0, v1, v2, v3, w0, w1, om0[4]]``.

- Added a :ref:`castshadow<body-geom-castshadow>` attribute to :ref:`geom<body-geom>` (``mjsGeom.castshadow``,
  ``mjModel.geom_castshadow``, ``mjvGeom.castshadow``; default ``true``) to disable shadow casting per geom in both the
  classic OpenGL and Filament renderers.
- Added ``mjModel.tendon_type`` with values from the new :ref:`mjtTendon` enum, distinguishing fixed and spatial
  tendons. Previously, the type was implied by whether the first path element was a joint.

Engine
^^^^^^
.. admonition:: Breaking changes
   :class: attention

   - Changed the semantics of :ref:`sidesite<spatial-geom-sidesite>`: the tendon now goes around the geom the same way
     as the path from the first endpoint, through the side site, to the second endpoint (thin grey line below).
     Formally, the tendon is the shortest path homotopic to this polyline in the wrapping plane with the geom removed.

     .. image:: images/XMLreference/sidesite_homotopy_light.svg
        :align: center
        :width: 90%
        :class: only-light

     .. image:: images/XMLreference/sidesite_homotopy_dark.svg
        :align: center
        :width: 90%
        :class: only-dark

     Previously, a half-plane test was used, which could make the tendon jump onto the geom, a discontinuity in tendon
     length, when the segment between the endpoints moved away from the side site while clear of the geom. It could
     also select the wrong side for wraps longer than 180 degrees. Top row: old rule; bottom row: new rule.

     .. image:: images/changelog/sidesite_old_vs_new_light.svg
        :align: center
        :width: 90%
        :class: only-light

     .. image:: images/changelog/sidesite_old_vs_new_dark.svg
        :align: center
        :width: 90%
        :class: only-dark

     The new rule has a single discontinuity: when an endpoint crosses the ray opposite the side site, the path
     through the side site changes its winding around the geom and the tendon snaps off. This can change the behavior
     of existing models.

     **Migration:** Place side sites so that the endpoints do not cross this ray over the range of motion, typically
     near the middle of the wrapped arc. For example, the side sites of the
     `two-link arm <https://github.com/google-deepmind/mujoco/blob/main/model/tendon_arm/arm26.xml>`__ were moved from
     directly below the joints to 45 degrees below the arm, which reproduces the previous tendon lengths exactly.

   - Removed the ``cable`` plugin (``plugin/elasticity``) and the ``composite`` element, which are superseded by 1D
     :ref:`flexcomp<body-flexcomp>` with :ref:`elasticity<flex-elasticity>`.

     **Migration:**

     - ``count`` and ``curve`` / ``size`` / ``vertex``: use :ref:`flexcomp<body-flexcomp>` ``dim="1"`` with
       ``type="grid"``, ``count``, and ``spacing`` (or ``type="circle"`` for closed loops), or ``type="direct"`` with
       ``point``. Add ``<edge equality="true"/>`` to keep edges inextensible.
     - Geom ``size`` (capsule radius): set flexcomp ``radius``. For geom ``type="box"``, set
       ``<elasticity section="box" sectionsize="..."/>``.
     - Plugin ``bend`` and ``twist``: set ``<elasticity young="..." shear="..."/>``.
     - Root attachment ``initial``: ``initial="none"`` maps to ``<pin id="0 1"/>`` (clamps the first edge's material
       frame to the parent body and creates the ``<name>_twist`` body and hinge joint); ``initial="ball"`` maps to
       ``<pin id="0"/>`` (spherical anchor at the first vertex, no twist joint); ``initial="free"`` maps to no
       ``<pin/>``.
     - Plugin ``flat="true"``: define the rod straight in ``qpos0`` (e.g., ``type="grid"``) and set the curved initial
       pose via ``qpos`` or a :ref:`keyframe<keyframe>` (curves defined directly in ``point`` are at rest in ``qpos0``,
       corresponding to ``flat="false"``).
     - Plugin ``vmax`` stress coloring: not supported.
     - Integrator: use ``integrator="discrete"`` for implicit integration (or ``Euler`` / ``RK4``; ``implicit`` and
       ``implicitfast`` are not supported for 1D rod elasticity).
     - Body names ``B_first`` / ``B_<i>`` / ``B_last``: unpinned vertices create bodies ``<name>_<i>`` (pinned vertices
       attach directly to the parent body), plus ``<name>_twist`` when ``<pin id="0 1"/>`` is used.

   - 1D rod bending :ref:`elasticity<flex-elasticity>` (``dim="1"`` with positive ``young``) is no longer supported under
     ``implicit`` and ``implicitfast``, matching 2D and 3D flex elasticity.

     **Migration:** Use ``integrator="discrete"`` for implicit integration, or ``integrator="Euler"`` with a small
     timestep for explicit integration.

   - Flex :ref:`edge stiffness<flex-edge-stiffness>` and :ref:`damping<flex-edge-damping>` are no longer supported under
     ``integrator="discrete"`` (where they were integrated explicitly rather than in the effective metric).

     **Migration:** Use :ref:`elasticity<flex-elasticity>` (or :ref:`edge equality<flexcomp-edge-equality>`), or use
     ``Euler``/``RK4``/``implicitfast``.

- Flex improvements:

  - .. admonition:: Breaking change
       :class: attention

       Fixed the elastic constants of non-interpolated :ref:`flexes<deformable-flex>` (**full** and **radial**
       :ref:`dofs<body-flexcomp-dof>`), which now match :ref:`young<flex-elasticity-young>`,
       :ref:`poisson<flex-elasticity-poisson>` and :ref:`thickness<flex-elasticity-thickness>` as documented, and the
       interpolated flexes:

       - 3D Saint Venant-Kirchhoff stretching used twice the first Lamé parameter :math:`\lambda`.
       - 2D stretching used a quarter of the shear modulus :math:`\mu` and half of the 3D :math:`\lambda`, instead of
         :math:`\mu` and the plane-stress :math:`\lambda^* = E \nu / (1 - \nu^2)`.
       - 2D bending used :math:`\tfrac{3}{4}(1-\nu) D` instead of the plate bending rigidity
         :math:`D = E t^3 / (12 (1 - \nu^2))`.

       Models with these flexes become stiffer, except 3D models with zero Poisson's ratio, which are unchanged.

       **Migration:** To keep the previous behavior, use these parameters:

       - 3D: :math:`\nu' = 2\nu / (1 + 2\nu)` and :math:`E' = E (1 + \nu') / (1 + \nu)`.
       - 2D bending only (``elastic2d="bend"``): :math:`E' = \tfrac{3}{4}(1 - \nu) E`.
       - 2D stretching (``elastic2d="stretch"`` or ``"both"``): :math:`\nu' = 2\nu`, which is exact for
         :math:`\nu < 1/4` (so that :math:`\nu' < 1/2`). Then:

         - ``"stretch"``: :math:`E' = E (1 + 2\nu) / (4 (1 + \nu))`.
         - ``"both"``: :math:`t' = t \sqrt{3 (1 - 2\nu)}` and :math:`E' = E t (1 + 2\nu) / (4 (1 + \nu) t')`.

       The flex models in ``model/flex`` were remapped in this way, so their behavior is unchanged.

  - Volumetric interpolated flexes (**trilinear** or **quadratic** :ref:`dofs<body-flexcomp-dof>` with ``elastic2d="none"``)
    now integrate cell stiffness and lump nodal masses over the mesh volume using Finite Cell Method (FCM) subcell
    quadrature instead of treating every non-empty grid cell as a full cube. This changes the stiffness and nodal mass
    distribution of volumetric interpolated flexes whose mesh does not fill its bounding box, as well as the nodal mass
    distribution of multi-cell **trilinear** flexes.
  - Interpolated flexes (**trilinear** or **quadratic** :ref:`dofs<body-flexcomp-dof>`) now support
    :ref:`passive<flex-contact-passive>` contact.
  - Contacts with flex elements now distribute the contact to the element vertices with the barycentric coordinates of the
    point on the element closest to the contact. Previously the vertex weights were inversely proportional to the
    distances from the contact, which do not interpolate the contact point: the contact Jacobian measured the velocity of
    a different material point, and the contact force exerted a spurious torque on the element. This changes the
    simulation of models with flex contacts.
  - Fixed contact Jacobians of quadratic flexes, which dropped nodes with negative basis weights.
  - Flex :ref:`edge damping<flex-edge-damping>` is now integrated explicitly under ``implicit`` and ``implicitfast``
    (previously it also damped rigid translation).
  - 1D rod bending :ref:`elasticity<flex-elasticity>` now supports curved reference configurations and torsional
    stiffness: vertices bent by more than :math:`10^{-4}` rad in ``qpos0`` keep their reference curvature and dihedral
    twist angle, so that rods such as rings, arcs, and coil springs are at rest in their compiled shape. Previously,
    such models failed to compile. The new :ref:`shear<flex-elasticity-shear>` attribute optionally overrides the
    shear modulus used for torsion. Open elastic 1D flexcomps with a clamped root edge get a ``<name>_twist`` hinge
    joint that holds the material frame of the last edge, with the polar inertia and Rayleigh torsional damping of the
    rod; the geometric twist is carried in time in the plastic state, so the twist energy stays continuous when the rod
    coils. Box sections solve the twist of the interior edges quasistatically. See the
    :ref:`documentation<flex-elasticity>`.

- Improved numerical stability of sphere and cylinder :ref:`spatial tendon<tendon-spatial>` :ref:`geom<spatial-geom>`
  wrapping at wrap onset. Thanks to :github:user:`M-Colley` for reporting and analyzing the issue in
  :doc:`MJWarp <mjwarp/index>` issue `1772 <https://github.com/google-deepmind/mujoco_warp/issues/1772>`__.
- The ``RK4`` :ref:`integrator<option-integrator>` is now fourth-order accurate for the orientations of ball and free
  joints. Previously, the angular velocities of the Runge-Kutta stages were combined without the Runge-Kutta-Munthe-Kaas
  correction, so orientations converged only at second order. This changes the simulation of models with ball or free
  joints that use ``RK4``.
- Added ``surfacevel[3]`` to :ref:`mjContact`, storing the relative :ref:`surface velocity<body-geom-surfacevel>`
  (``geom[1]`` minus ``geom[0]``) in the contact frame ``(tangent1, tangent2, spin)``. Collision detection populates it
  from ``geom_surfacevel`` when ``mjModel.flg_surfacevel`` is set, and external contact generators can write it
  directly (setting ``flg_surfacevel = 1`` if no geom specifies ``surfacevel``). Contact-point visualization now draws
  a single relative tangential velocity arrow per active frictional contact.
- Added ``xiquat``, ``geom_xquat``, and ``site_xquat`` to :ref:`mjData`, storing the Cartesian quaternion orientations
  of body inertias, geoms, and sites alongside ``ximat``, ``geom_xmat``, and ``site_xmat``.
- ``mju_mat2Rot`` now normalizes the input quaternion and escapes 180-degree stationary points (saddles and minima)
  that do not maximize :math:`\operatorname{tr}(R^\top M)`.

Compiler
^^^^^^^^
.. admonition:: Breaking changes
   :class: attention

   - Models that :ref:`include<include>` the same file through different paths, which previously loaded, are now
     rejected with ``already included``.

     **Migration:** Remove redundant ``<include>`` elements referencing the same file.

   - :ref:`Height fields<asset-hfield>` with :ref:`nrow<asset-hfield-nrow>` or :ref:`ncol<asset-hfield-ncol>` less than
     2 are now rejected at compile time.

     **Migration:** Ensure height fields have ``nrow >= 2`` and ``ncol >= 2``.

   - :ref:`Fixed tendons<tendon-fixed>` referencing ``ball``/``free`` or duplicate joints or with negative
     :ref:`armature<tendon-fixed-armature>`, :ref:`spatial tendon<tendon-spatial>`
     :ref:`pulleys<spatial-pulley>` with non-positive :ref:`divisor<spatial-pulley-divisor>`, and
     :ref:`camprojection<sensor-camprojection>` or camera-attached :ref:`rangefinder<sensor-rangefinder>` sensors
     referencing an invalid camera or a camera with non-positive :ref:`resolution<body-camera-resolution>` are now
     rejected at compile time.

     **Migration:** Ensure fixed tendons wrap distinct ``slide`` or ``hinge`` joints with non-negative armature,
     pulley divisors are positive, and cameras referenced by ``camprojection`` or ``rangefinder`` sensors have
     positive resolution.

   - :ref:`Skins<deformable-skin>` with out-of-range face vertex indices and :ref:`mjs_makeMesh`
     ``mjMESH_BUILTIN_CONE`` calls with fewer than 3 edges or ``radius`` outside ``[0, 1]`` are now rejected at
     compile time.

     **Migration:** Ensure skin face indices are in ``[0, nvert)`` and cone mesh parameters specify at least 3 edges
     and a radius in ``[0, 1]``.

- :ref:`default/tendon<default-tendon>` in MJCF now accepts :ref:`armature<tendon-fixed-armature>`,
  :ref:`actuatorfrclimited<tendon-spatial-actuatorfrclimited>`, and
  :ref:`actuatorfrcrange<tendon-spatial-actuatorfrcrange>` (previously emitted when saving XML, which then failed
  schema validation on reload).
- :ref:`mj_encode` now supports binary MJB output for :ref:`mjSpec` via the ``model/vnd.mujoco.mjb`` content type.

Filament Rendering
^^^^^^^^^^^^^^^^^^
- Added horizon haze and skybox visibility support: infinite planes are centered beneath the camera and blend smoothly
  into the skybox horizon according to :ref:`visual/map/haze<visual-map-haze>` and
  :ref:`visual/rgba/haze<visual-rgba-haze>`, toggled per view via ``haze`` and ``skybox`` on :ref:`mjrfRenderRequest`
  (:ref:`mjRND_HAZE<mjtRndFlag>` and :ref:`mjRND_SKYBOX<mjtRndFlag>`).
- Added :ref:`mjrf_flush` (``mjrf.Context.flush`` in Python) to flush pending operations to the rendering backend and
  invoke any completed callbacks.
- Added :ref:`mjrf_setRenderableCastShadow` (``mjrf.Renderable.set_cast_shadow`` in Python) to dynamically toggle
  whether a renderable casts shadows.
- When no light specifies an :ref:`intensity<body-light-intensity>`, the fallback intensity given to each light is now
  scaled by the brightness of its diffuse color, which is factored out of the light's color. Lights look the same,
  but the brightest directional light now casts the shadows, rather than an arbitrary one that could differ between
  runs.
- Added :ref:`filament.material.base_color_type<reFlags>` flag for configuring how the base color material parameter is
  interpreted.

Rendering
^^^^^^^^^
- Added :ref:`mjRND_POST_PROCESSING<mjtRndFlag>` rendering flag.

Python bindings
^^^^^^^^^^^^^^^
- Removed support for Python 3.10, now that it's `deprecated upstream <https://devguide.python.org/versions>`__.

Bug fixes
^^^^^^^^^
- :ref:`fullinertia<body-inertial-fullinertia>` and inertias inferred from geoms which are not positive definite are
  now bounded by :ref:`boundinertia<compiler-boundinertia>` when it is positive. Previously they gave a compile error.
- Contact pairs created with an undefined ``mjsPair.solref`` now mix the ``solref`` of their geoms using ``solmix``
  when both are positive, like contacts not defined by pairs. Previously they always took the element-wise minimum.
- :ref:`mj_setConst` now wakes all :ref:`sleeping<siSleep>` trees, since it recomputes their kinematics at ``qpos0`` and
  ``qpos_spring``. Previously their cached poses could be left at ``qpos_spring``. It also no longer overwrites
  ``cam_xpos``, ``cam_xmat``, ``light_xpos`` and ``light_xdir`` in :ref:`mjData`, which left static cameras and lights
  with tracking or targeting at their fixed-mode poses.
- Fixed :ref:`sleep<siSleep>` filtering bugs in history sensors (:at:`nsample`, :at:`delay`, :at:`interval`),
  :ref:`mj_energyPos`, flex passive forces, tendon limit and frictionloss constraints, two-tree tendon
  :ref:`armature<tendon-fixed-armature>` and bias, :ref:`mj_subtreeVel`, and caller-supplied :ref:`mj_rne` buffers.
- Fixed :ref:`dcmotor<actuator-dcmotor>` parameter handling: allow zero :ref:`motorconst<actuator-dcmotor-motorconst>`
  for ``input="none"``, reject negative constants (e.g. ``motorconst="-a b"`` is now rejected instead of silently
  using ``K = b``), replace capacitance when an inconsistent :ref:`thermal<actuator-dcmotor-thermal>` triple is given,
  clear inherited :ref:`forcerange<actuator-general-forcerange>` when :ref:`saturation<actuator-dcmotor-saturation>`
  is authored, let zero ``saturation`` or ``inductance`` clear class defaults, and validate
  :ref:`actdim<actuator-general-actdim>` at compile time.
- Fixed a bug where tendon actuators saturated by :ref:`actuatorfrcrange<tendon-spatial-actuatorfrcrange>` still
  contributed to implicit actuator derivatives.
- Actuators saturated by joint :ref:`actuatorfrcrange<body-joint-actuatorfrcrange>` now also skip implicit actuator
  derivatives, changing integration under ``implicit``, ``implicitfast``, and ``discrete``. Because actuators on
  saturated joints no longer contribute implicit damping, models with light links and no armature may need joint
  armature or a smaller timestep.
- Fixed several MJCF XML loading and saving bugs:

  - :ref:`include<include>` no longer double-prefixes the model directory on nested or relative paths.
  - Loading MJCF now applies enclosing :ref:`frame<frame>` transforms to :ref:`flexcomp<body-flexcomp>`, and allows
    explicit :at:`range` or :ref:`inheritrange<actuator-position-inheritrange>` attributes on actuator shortcuts to
    override conflicting defaults.
  - Saving MJCF (:ref:`mj_saveLastXML`, :ref:`mj_saveXML`, :ref:`mj_saveXMLString`) now preserves
    :ref:`texture<asset-texture>` :ref:`nchannel<asset-texture-nchannel>` and clamps emitted
    :ref:`gridlayout<asset-texture-gridlayout>` to ``gridsize`` (while loading resets ``gridlayout`` when an authored
    ``gridsize`` differs from the inherited default), compares material :ref:`layer<material-layer>` textures, camera
    :ref:`sensorsize<body-camera-sensorsize>`, and light :ref:`texture<body-light-texture>` against default classes, and
    writes all attributes on plugin-backed meshes and :at:`reftype`/:at:`refname` on plugin sensors.
- :ref:`mj_ray` now scales SDF hit distances by ``1 / ||vec||`` for non-unit ray vectors, and ``mju_rayTriangle``
  rejects triangle intersections behind the ray origin.
- :ref:`mj_resetData`, :ref:`mj_copyData`, and :ref:`mj_recompile` now preserve the destination :ref:`mjData` thread
  pool instead of dropping and leaking it.
- :ref:`mjs_delete` (``MjSpec.delete`` in Python) now removes deleted default classes from the class lookup map
  instead of leaving dangling pointers; :ref:`mj_copyBack` (``MjSpec.copy_back`` in Python) preserves material
  ``metallic``, ``roughness``, and ``texuniform`` attributes; :ref:`mjs_findChild` no longer matches a non-world body
  against itself; and :ref:`mjs_setToMuscle` accepts ``NULL`` ``timeconst`` and ``range`` pointers.
- Legacy ``.msh`` meshes with mirrored :ref:`scale<asset-mesh-scale>` no longer double-flip face winding (which
  inverted faces), and ``.msh`` files without vertex normals no longer populate empty face normals or reject custom
  normals on recompilation.
- :ref:`accelerometer<sensor-accelerometer>` and :ref:`framelinacc<sensor-framelinacc>` sensors and
  :ref:`mj_objectAcceleration` on static and mocap bodies now report the gravitational proper acceleration, like a
  dynamic body at rest. Previously they returned zero, a regression introduced in 3.3.7.

Documentation
^^^^^^^^^^^^^
- Added documentation for :ref:`MuJoCo Studio <Studio>`, covering :ref:`web <StudioWebViewers>` and
  :ref:`native <StudioNativeViewers>` viewers, :ref:`architecture <StudioArchitecture>`, and
  :ref:`Python <StudioPythonPlugins>` and :ref:`C++ <StudioCpp>` development.
- Added a :ref:`Model tuning <CTuning>` guide to the Modeling chapter---covering the fidelity--speed trade-off,
  :ref:`tuning defaults <CTuningDefaults>`, priority-ordered :ref:`global <CTuningGlobal>` and
  :ref:`model-level <CTuningModel>` parameters, :ref:`deformables <CTuningDeformable>`, and
  :ref:`system identification <CTuningSysid>`---along with module documentation for :ref:`mujoco.sysid <PySysid>`.

Version 3.17.0 (October 22, 2026)
---------------------------------

Significant new features
^^^^^^^^^^^^^^^^^^^^^^^^
1. :commit:`c00ed8153` Added a :ref:`batch API<siBatch>` in ``mujoco/mjbatch.h`` (in Python as
   :ref:`mujoco.batch<PyBatch>`): many simulations of one model stepped on a thread pool of its own, each held as its
   integration state and loaded into an ``mjData`` for each call, so memory scales with threads and results are
   bit-identical to a plain loop at any thread count. It supports per-simulation model and option fields or models of
   the caller's, observations of the state after a step, recorded trajectories, persistent simulations (required for
   sleep), and arbitrary per-simulation functions. Follows the design of
   `mjbatch <https://github.com/kevinzakka/mjbatch>`__ by :github:user:`kevinzakka`.

Engine
^^^^^^
2. Flex improvements:

   - :commit:`94483e166` Added bending :ref:`elasticity<flex-elasticity>` for 1D flexes, using a twist-free isotropic
     discrete elastic rod model. The cross-section is selected with the new :ref:`section<flex-elasticity-section>`
     attribute: a circle of the flex radius, or a box with half-sizes :ref:`sectionsize<flex-elasticity-sectionsize>`
     that uses the stiffness of its weak axis. It is enabled by a positive :ref:`young<flex-elasticity-young>` modulus,
     requires a straight reference configuration (curved initial poses can be set with ``qpos`` or keyframes), is
     compatible with edge equality constraints, and is integrated implicitly by the :ref:`discrete<option-integrator>`
     integrator. Previously ``young`` was ignored for 1D flexes: existing models that set it now bend, and fail to
     compile if the cross-section is degenerate (zero radius with the default circle section), if a vertex joins a
     zero-length edge, or if the rod is not straight.
   - :commit:`36d0df9a5`, :commit:`897e3fcb1` Interpolated flexes (**trilinear** or **quadratic**
     :ref:`dofs<body-flexcomp-dof>`) now support :ref:`self-collision<flex-contact-selfcollide>`, excluding elements in
     the same or neighboring cells.
   - :commit:`39dd10f2d` The default arena :ref:`memory<size-memory>` for flex models is now much smaller, sized from
     the flex mesh topology instead of ``nv^2``. If the Newton solver runs out of arena memory it falls back to CG with
     a warning.
3. :ref:`mj_setConst` now computes only quantities derived from other fields of :ref:`mjModel`, while the compiler alone
   resolves spec-level conventions:

   - :commit:`55c1b585a`, :commit:`d6d41f324` The damping of position actuators from
     :ref:`dampratio<actuator-position-dampratio>`, the spring length of tendons from ``(-1, -1)``, and model statistics
     are now resolved only at compile time, not by :ref:`mj_setConst`. Set the model fields directly when editing a
     compiled model (:ref:`mj_copyBack` now copies tendon spring lengths back to the spec).
   - :commit:`c8362756d` Constraint solvers now scale their termination criteria by the trace of the current inertia
     matrix (of the model or island) instead of ``mjModel.stat.meaninertia``. Models solved with PGS, noslip or islands
     disabled can stop at slightly different iterations. The :ref:`meaninertia<statistic-meaninertia>` attribute is
     deprecated and now emits a compile warning; remove it to migrate.
   - :commit:`abddec5b9` For body-based :ref:`weld<equality-weld>` constraints without a specified
     :ref:`relpose<equality-weld-relpose>`, the relative pose in ``qpos0`` is now recomputed by :ref:`mj_setConst` when
     the model is modified after compilation (matching the second body's :ref:`anchor<equality-connect-anchor>` in
     body-based ``connect`` constraints). The compiler marks such welds with a nonzero ``eq_data[11]``, and
     :ref:`mjNEQDATA<glNumericSizes>` increased from 11 to 12.

     .. admonition:: Breaking ABI changes
        :class: attention

        :ref:`mjNEQDATA<glNumericSizes>` increased from 11 to 12, and :ref:`mj_setConst` overwrites ``eq_data[3:10]`` of
        body-based welds that had no ``relpose`` at compile time. Zero ``eq_data[11]`` before calling :ref:`mj_setConst`
        to set the relative pose of such a weld directly.
   - :commit:`865316f19` Added ``MJSETCONST_ARRAYS`` and ``MJSETCONST_SCALARS`` in ``mjxmacro.h``, listing the
     :ref:`mjModel` arrays and scalars read and written by each phase of :ref:`mj_setConst`.
4. :commit:`e15ecd842` Added single-shot :ref:`multicontact<coMultiCCD>` for geoms with positive contact
   :ref:`margins<body-geom-margin>`.
5. :commit:`7394a8672` :ref:`mju_boxQP` now updates its Cholesky factor incrementally with :ref:`mju_cholUpdate`, making
   cold-started solves 1.4-1.6x faster and warm-started solves 1.9-2.9x faster.
6. :commit:`aa0f88e77` Trees marked :ref:`sleep="init"<body-sleep>` which cannot be put to sleep now cause a warning
   rather than an error, and start awake. See :ref:`Sleeping<siSleepSleeping>` for details.

Compiler
^^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   7. :commit:`aad2e2982` :ref:`mjs_fuseStatic` and :ref:`mjs_discardVisual` now return the number of bodies fused or
      elements discarded, and -1 on error. Previously they returned 0 on success, so a caller could not tell whether the
      spec changed. In Python, ``MjSpec.fuse_static`` and ``MjSpec.discard_visual`` return the same number instead of
      ``None``.

      **Migration:** Test for an error with ``< 0`` rather than ``!= 0``.

   8. :commit:`084828896` :ref:`mjs_addTexture` now takes a default class, like :ref:`mjs_addMesh` and
      :ref:`mjs_addMaterial`.

      **Migration:** ``mjs_addTexture(spec)`` → ``mjs_addTexture(spec, NULL)``. ``MjSpec.add_texture`` is unchanged.

9. :commit:`3576818f5`, :commit:`b0f206d04` Textures can be left :ref:`by reference<CAssetReference>` instead of being
   decoded into :ref:`mjModel`. With :ref:`compiler/passthrough<compiler-passthrough>` set to "textures", or
   :ref:`texture/embed<asset-texture-embed>` set to "false", a single-file texture is not read at compile time; the
   model holds its path instead of its pixels, so a scene that attaches many textured sub-models loads in a fraction of
   the memory and time. In addition, a compiled :ref:`mjSpec` no longer keeps the decoded pixels of its textures between
   compilations.

   - :commit:`3576818f5` What the model holds: ``tex_adr`` is -1, ``tex_width`` and ``tex_height`` are 0, ``ntexdata``
     excludes the texture, the resolved path is at ``tex_pathadr``, and any authored
     :ref:`content_type<asset-texture-content_type>` is at the new ``tex_contenttypeadr``. Flipped, composite, six-file,
     procedural, buffer and in-memory textures stay embedded, as does any texture with :at:`embed` "true".
   - :commit:`1be762d0c` Getting the pixels: :ref:`mj_loadTexture` decodes a texture of either kind, and
     :ref:`mj_embedAssets` returns a copy of the model with every by-reference texture loaded (which followed by
     :ref:`mj_saveModel` gives a hermetic MJB). Both are in Python as ``mujoco.mj_loadTexture`` and
     ``mujoco.mj_embedAssets``.
   - :commit:`fd92ba1db` Rendering: the built-in and Filament renderers load by-reference textures on demand, and
     Filament uploads each file once for all textures that share it. A file that fails to resolve produces a warning and
     an untextured material.
10. :commit:`084828896` Textures can have :ref:`default classes<default-texture>`: ``mjsDefault`` has a ``texture``
    member and :ref:`texture<asset-texture>` takes a :ref:`class<asset-texture-class>`.
11. :commit:`4e9a2de2d` Added :ref:`mjs_dedupAssets` (``MjSpec.dedup_assets``), which merges the textures and meshes of
    an :ref:`mjSpec` that would compile to the same asset, as those of a sub-model attached several times do: the first
    of each group is kept and every reference to the others is rewritten to it. Like :ref:`mjs_fuseStatic`, it returns
    the number of assets merged.
12. :commit:`027cb3dd8` The asset paths in :ref:`mjModel` (``paths`` at ``mesh_pathadr``, ``hfield_pathadr``,
    ``skin_pathadr`` and ``tex_pathadr``) now store the resolved path or URI from which the compiler read the asset
    (combining the model file's directory and ``meshdir`` or ``texturedir`` with :at:`file`, unless :at:`file` is an
    absolute path or URI), so :ref:`mju_openResource` can open them directly even after :ref:`attaching<body-attach>`
    models with different asset directories. Code that prefixed these paths with its own directory should pass them to
    :ref:`mju_openResource` as is.
13. :commit:`9f035384d` Mesh polygon normals (``mjModel.mesh_polynormal``) are now computed from the whole polygon
    outline using Newell's method rather than the first three vertices (:issue:`3691`).

Python bindings
^^^^^^^^^^^^^^^
14. :commit:`08330629b` Added a :ref:`mujoco.batch<PyBatch>` module over the batch API: ``Batch(model, nsim)`` with live
    ``(nsim, ...)`` arrays from ``bind()`` and named views such as ``joint(name)``, per-simulation model fields from
    ``expand()``, ``step``, ``forward``, ``reset``, ``set_const`` and ``rollout`` with the GIL released, and ``jac`` and
    ``ray`` queries.
15. :commit:`1313ef78e` Added support for building the Python bindings with `Bazel <https://bazel.build/>`__ (initially
    covers the core bindings, without rendering or the viewer).
16. :commit:`39b7193e8` Fixed several bugs in :ref:`named access<PyNamed>` (``data.actuator('a')``,
    ``model.joint('j')``, ``data.bind(...)``), and the corresponding generated accessors in the JavaScript bindings:

    - Multi-control or multi-output actuators (such as :ref:`orientation<actuator-orientation>`) now follow
      ``actuator_ctrladr`` and ``actuator_outadr`` with leading dimension ``ctrlnum`` or ``outnum``, instead of reading
      a single row at the actuator index.
    - ``data.actuator('a').moment`` is now a read-only dense copy of the actuator's moment arm of shape ``(nv,)`` or
      ``(outnum, nv)``, instead of reading the sparse ``actuator_moment`` array as if it were dense.
    - Views of multi-column per-dof fields (``cdof``, ``cdof_dot``, ``dof_dampingpoly``) on joints after the first now
      index the right rows.
17. :commit:`7cdc8857e` Unallocated arena arrays of ``MjData`` (such as ``efm_*`` when the ``discrete`` integrator is
    not used, or island arrays when no islands are formed) are now filled with zeros, or -1 for index fields, instead of
    returning uninitialized memory.

Bug fixes
^^^^^^^^^
18. Fixed several bugs in interpolated :ref:`flexcomp<body-flexcomp>` elasticity:

    - :commit:`4bbc96957` Volume (solid) interpolated flexes whose first cell is empty (e.g. a trilinear
      :ref:`flexcomp<body-flexcomp>` from a mesh that does not fill the corner of its bounding box) now retain their
      elasticity; previously elasticity was gated on the first cell's stiffness.
    - :commit:`b0b20166d` Editing a flex to remove a strain :ref:`equality<flexcomp-edge-equality>` (e.g. from
      ``body.make_flex``) and recompiling now recomputes stiffness instead of keeping the strain constraint eigenmodes.
    - :commit:`08bf94b18` Shell-mode interpolated flexes with :ref:`elastic2d<flex-elasticity-elastic2d>` set to
      ``"stretch"`` or ``"both"`` now apply membrane stretch forces and implicit stiffness, and ``"bend"`` shells
      without strain constraints no longer allocate ``flex_stiffness``.
19. Fixed several bugs in the Filament renderer:

    - :commit:`5c7c1e3ae` Textures now allocate mipmap levels (including sRGB textures), fixing minification aliasing.
    - :commit:`e81c8aed4` Skins that share vertices across faces no longer fail with an index count mismatch
      (:issue:`3719`).
    - :commit:`001aa3203` Material colors now apply only when an object has the default color, instead of overwriting
      custom object colors.
    - :commit:`6cfc76e32` Textures with an unsupported number of channels are now skipped with a warning, matching the
      classic renderer, instead of calling :ref:`mju_error`.
20. Fixed several bugs on error paths which called :ref:`mju_error` while C++ objects or locks were held (an error
    handler that does not return, as in the Python bindings and the model compiler, skips C++ destructors):

    - :commit:`4cc38db84`, :commit:`8c1e45559`, :commit:`526583e5b` Invalid configurations of the touch grid, cable and
      SDF plugins leaked more than 10 MB per failed compilation (they now fail plugin initialization with a warning);
      :ref:`mju_getXMLDependencies` and unknown material layer :ref:`roles<material-layer-role>` also leaked memory on
      invalid XML, and unsupported equality types in the XML writer are now reported as write errors instead of calling
      :ref:`mju_error`.
    - :commit:`4cc38db84` An error while compiling meshes or textures on the compiler's worker threads crashed the
      process, and a failed library load in :ref:`mj_loadAllPluginLibraries` left its mutex locked.
    - :commit:`ae15ab978` :ref:`mj_recompile` freed the given :ref:`mjData` and called :ref:`mju_error` if a plugin
      failed to initialize, and did not delete the model and data on some other failures, contrary to its documentation.

Models
^^^^^^
21. :commit:`94483e166` The
    `elastic mechanism <https://github.com/google-deepmind/mujoco/tree/main/model/flex/mechanisms/>`__ example models
    now use 1D flex rods: the cantilever comparison of quadratic flexes adds a rod, and the elastic slider-crank and the
    tendon-loaded dipper arm are built from rods.

Version 3.16.0 (October 14, 2026)
---------------------------------

Significant new features
^^^^^^^^^^^^^^^^^^^^^^^^
1. :commit:`cf89a6375`, :commit:`4316935b3`, :commit:`6d98d9fab`, :commit:`5f17e92eb`, :commit:`89d59b11f`,
   :commit:`147dc67d0` :ref:`Frames<frame>` can now scale their contents at compile time—without modifying the
   :ref:`mjSpec`—via the new :ref:`scale<frame-scale>`, :ref:`scaling<frame-scaling>`, and
   :ref:`massscale<frame-massscale>` attributes. The default ``similar`` policy scales all physical quantities so that
   dynamics are preserved under proportionally scaled gravity, whereas ``geometry`` scales only spatial dimensions and
   inertias (and supports anisotropic scaling). Masses scale with volume unless overridden by ``massscale``. External
   references to scaled elements—including actuators, tendons, constraints, sensors, keyframes, and deformable/mesh
   assets—are scaled consistently, and :ref:`mj_copyBack` maps values back into unscaled spec coordinates. See
   :ref:`frame/scale<frame-scale>` for details.
2. :commit:`592c54ad3`, :commit:`92b8f4b27`, :commit:`d4595d020` Added compiler attributes
   :ref:`savecompiled<compiler-savecompiled>` and :ref:`savecanonical<compiler-savecanonical>` (both defaulting to
   ``"true"``) to preserve original authoring when saving models to XML. Setting ``savecompiled="false"`` saves the
   uncompiled :ref:`mjSpec` directly—retaining default classes, frames, and authored numeric values—so the saved XML
   reloads into an identical spec. Setting ``savecanonical="false"`` additionally preserves the original input
   conventions, such as orientation representations, degree/radian units, and actuator shortcuts; see
   :ref:`Model Encoding & Saving <meSaving>`. :ref:`mj_copyBack` (``MjSpec.copy_back`` in Python) now updates only the
   modified fields in the spec while leaving all other authored values untouched.
3. :commit:`9d953b40c` Added support for building MuJoCo with `Bazel <https://bazel.build/>`__ (initially covers core
   library, ``simulate``, plugins, and samples).

General
^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   4. :commit:`ea5e711c1` :ref:`mjfDecode`, the ``decode`` callback of :ref:`mjpDecoder` plugins, and
      :ref:`mju_decodeResource` now take ``char* error, int error_sz`` arguments. A decoder which fails writes its
      reason into ``error``, which is then reported by :ref:`mj_parse` and in mesh compilation errors, rather than only
      as a warning with a generic error.

      **Migration:** Add the two arguments to custom decoders and write failure messages into ``error`` (if not
      ``NULL``) instead of calling :ref:`mju_warning`. Pass an error buffer, or ``NULL, 0``, to
      :ref:`mju_decodeResource`.

   5. :commit:`2757493d9` Shortened the names of :ref:`mjpResourceProvider` functions. ``mjp_getResourceProvider`` is
      renamed to :ref:`mjp_findProvider`, consistent with other prefix and extension lookups.

      **Migration:** Rename as follows:
      |br| ``mjp_defaultResourceProvider`` → :ref:`mjp_defaultProvider`
      |br| ``mjp_registerResourceProvider`` → :ref:`mjp_registerProvider`
      |br| ``mjp_resourceProviderCount`` → :ref:`mjp_providerCount`
      |br| ``mjp_getResourceProvider`` → :ref:`mjp_findProvider`
      |br| ``mjp_getResourceProviderAtSlot`` → :ref:`mjp_getProviderAtSlot`
      |br| ``mjp_registerArchiveResourceProvider`` → :ref:`mjp_registerArchiveProvider`
      |br| ``mjp_findArchiveResourceProvider`` → :ref:`mjp_findArchiveProvider`
      |br| ``mjp_archiveResourceProviderCount`` → :ref:`mjp_archiveProviderCount`

Engine
^^^^^^
6. :commit:`0998957cf` The :ref:`contact<sensor-contact>` sensor now supports ``linvel`` and ``angvel``
   :ref:`data<sensor-contact-data>` fields for reporting relative contact velocities, and its computation stage
   (:ref:`mjtStage`) is now inferred from the requested :ref:`data<sensor-contact-data>` and
   :ref:`reduce<sensor-contact-reduce>` attributes.
7. :commit:`d823f9212` :ref:`mj_setConst` now computes the bounding volumes of geoms and bodies, the collision masks and
   margins of bodies, and the added mass of geoms in the ellipsoid :ref:`fluid model<flEllipsoid>`. Changes to the size,
   pose, margin and gap of geoms, to the inertial frames of bodies and to the size of height fields are then safe;
   previously contacts could be missed. :ref:`mj_setConst` also computes the constants of sleeping bodies, which it
   skipped, and resolves the automatic :ref:`sleep policy<body-sleep>` of trees anew.
8. :commit:`52824daf6` Corrected the shear contribution to solid and membrane stiffness for interpolated flexes, and the
   three-point Gauss quadrature used by quadratic flexes. Material parameters now produce the standard isotropic
   linear-elastic stiffness; existing models using these flexes can change behavior.
9. :commit:`3bc4b769a` Improved numerical stability of sphere and cylinder :ref:`spatial tendon<tendon-spatial>`
   :ref:`geom<spatial-geom>` wrapping near wrap onset and for inside wrapping. Thanks to :github:user:`M-Colley` for
   reporting and analyzing the issues in :doc:`MJWarp <mjwarp/index>` issues
   `1713 <https://github.com/google-deepmind/mujoco_warp/issues/1713>`__ and
   `1714 <https://github.com/google-deepmind/mujoco_warp/issues/1714>`__.
10. :commit:`871e3725b` Flex contacts are now reduced to ``mjMAXCONPAIR`` per geom only for rigid and interpolated
    flexes; other flexes keep all their contacts. Previously, cloth resting on a plane or on many static geoms could be
    held by at most 50 contacts and fall through.
11. :commit:`efffd3a5f` Fixed floating-point cancellation in capsule-capsule collision detection for parallel and
    near-parallel axes. Thanks to :github:user:`ayonga` for reporting the issue in :doc:`MJWarp <mjwarp/index>` issue
    `1758 <https://github.com/google-deepmind/mujoco_warp/issues/1758>`__.

Compiler
^^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   12. :commit:`592c54ad3` :ref:`mj_copyBack` and :ref:`mj_saveLastXML` now fail, with an error which names the element,
       if a value was changed in the model which the spec cannot express, for example the mass of a body when
       :ref:`inertiafromgeom<compiler-inertiafromgeom>` is "true". Nothing is copied or saved then; previously such a
       change was left out of the saved file without notice. :ref:`mj_saveLastXML` also fails if it is given a model
       which was not compiled from the XML that was loaded last, and a save which fails no longer empties the file.

       **Migration:** Undo the change in the model, or change what makes it inexpressible, as listed for
       :ref:`mj_copyBack`.

   13. :commit:`2267bf299` An asset added to an :ref:`mjSpec` through the API is no longer named after its ``file`` when
       the spec is compiled. A mesh, height field or texture which is left without a name is now a compilation error, as
       it already was for one without a file. MJCF is unaffected: the XML parser still names assets after their files.

       **Migration:** Name the asset when adding it, e.g. ``spec.add_mesh(name='link', file='link.stl')`` in Python or
       :ref:`mjs_setName` in C.

   14. :commit:`1961e8a2d` :ref:`Keyframes<keyframe>` in MJCF are no longer copied by :ref:`replicate<replicate>`: a
       keyframe describes the model with all replicas in place, as in a model without replicates. Previously each
       replica received a copy of every keyframe, setting only the joints of that replica, and a keyframe written for
       the complete model failed to load (:issue:`3071`, reported by :github:user:`eholum-nasa`).

       **Migration:** Write the vectors of a keyframe for the complete model. A keyframe which was written for a single
       replica still loads, as one keyframe whose vectors are completed with the default configuration.

   15. :commit:`1db889162` With :ref:`inertiafromgeom<compiler-inertiafromgeom>` "true",
       :ref:`discardvisual<compiler-discardvisual>` is now a compilation error if a geom which it discards has mass.
       Previously the first compilation kept the inertia of the discarded geoms, while compiling the same spec again, or
       saving it and loading the file, dropped it.

       **Migration:** Use ``inertiafromgeom="auto"``, the default, under which the inertia is kept as an explicit
       :ref:`inertial<body-inertial>`, or give the visual geoms zero mass.

16. :commit:`0696cdc85` :ref:`flexcomp<body-flexcomp>` can now reference a :ref:`mesh asset<asset-mesh>` directly using
    the :ref:`mesh<body-flexcomp-mesh>` attribute, in place of :at:`file`. Volumetric meshes (e.g. from GMSH) supply
    nodes and tetrahedra directly; surface meshes can create 1D or 2D flexes or extrude to volumetric tetrahedra using
    :at:`origin`. Added the ``mesh`` parameter to Python ``body.make_flex``; in :ref:`mjs_makeFlex`, the ``file``
    parameter can also be the name of a mesh asset.
17. :commit:`ea019c212`, :commit:`0696cdc85` Added procedural mesh types ``grid``, ``box``, ``cylinder``, ``ellipsoid``,
    ``square`` and ``disc`` to :ref:`mesh/builtin<asset-mesh-builtin>` and :ref:`mjs_makeMesh`, generating the same
    shapes as the corresponding :ref:`flexcomp<body-flexcomp>` types, which flexcomp and geoms can use directly. Types
    without volume change a ``legacy`` mesh inertia to ``shell``. Added Python convenience methods ``mjsMesh.make_*``.
18. :commit:`dc8210572`, :commit:`08490e897`, :commit:`831994e2a` Added :ref:`mjs_fuseStatic`, which fuses the static
    bodies of an :ref:`mjSpec` with their parents as an operation on the spec: each fused body becomes a frame in its
    parent. The :ref:`fusestatic<compiler-fusestatic>` compiler option now applies it and then compiles; previously it
    fused inside compilation and left the spec inconsistent with the model. Static bodies whose fusing would change the
    forces on the model (a plugin, a sleep policy, another gravcomp, a fluid), or which a flex or skin refers to, are
    now kept, and cameras and lights are no longer misplaced. Elements which refer by name to a geom, site, camera or
    light are no longer bound to a different one, and referencing a body by name no longer leaves other static bodies
    unfused. The new :ref:`fuse<body-fuse>` body attribute keeps a static body when set to "false".
19. :commit:`1db889162` Added :ref:`mjs_discardVisual`, which discards the visual elements of an :ref:`mjSpec` as an
    operation on the spec. The :ref:`discardvisual<compiler-discardvisual>` compiler option now applies it and then
    compiles; previously it discarded inside compilation and left references to what it discarded in the spec, so that
    some models failed to compile. Visual meshes are no longer loaded unless body inertia is inferred from them, and
    geoms which use the ellipsoid :ref:`fluid model<body-geom-fluidshape>` are kept.
20. :commit:`d48b586ae` Added :ref:`mjs_adoptInertial`, which makes the inertial that compilation infers from the geoms
    of a body part of the :ref:`mjSpec`, so that it no longer follows later changes to the geoms.
21. :commit:`5558bd783` The :ref:`settotalmass<compiler-settotalmass>` compiler attribute is deprecated and will be
    removed in a future release. It still scales the masses, and compiling a model which sets it now gives a warning.

    **Migration:** Scale the masses and densities in the model, or call :ref:`mj_setTotalmass` on the compiled model,
    followed by :ref:`mj_setConst`.

Compiler bug fixes
^^^^^^^^^^^^^^^^^^
22. :commit:`9fd6ee025` The ``quat`` of a geom, site or camera is no longer ignored when its
    :ref:`default class<default>` gives an orientation as ``euler``, ``axisangle``, ``xyaxes`` or ``zaxis``.
23. :commit:`9fd6ee025`, :commit:`92b8f4b27`, :commit:`239647a1d` Saving MJCF no longer changes or loses the following:
    the order of the elements of a body which has :ref:`frames<frame>`, which gave the saved model other ids;
    ``class="main"`` inside another ``childclass``; the :ref:`nchannel<asset-texture-nchannel>` of a texture; the names
    of energy sensors, which could not be loaded; ``inertiafromgeom="false"``, and the inertia of the bodies of an
    attached model which has another ``inertiafromgeom``; the frame of the :ref:`surfacevel<body-geom-surfacevel>` of
    mesh geoms; the precision of mesh, flex, skin and height field data; and the empty cells of a
    :ref:`flexcomp<body-flexcomp>` with trilinear or quadratic dofs.
24. :commit:`592c54ad3` :ref:`mj_saveXML` and :ref:`mj_saveXMLString` now save the compiled values of an :ref:`mjSpec`
    as :ref:`mj_saveLastXML` does, including those which are computed at the end of compilation, such as the damping
    which :ref:`dampratio<actuator-position-dampratio>` gives, and no longer lose the scaling of
    :ref:`settotalmass<compiler-settotalmass>`.
25. :commit:`1961e8a2d` :ref:`Keyframes<keyframe>` of an :ref:`mjSpec` now keep their place and their values when its
    kinematic tree changes (:ref:`mjs_delete`, :ref:`mjs_attach`) and when the spec is copied; previously the next
    compilation created them again, and values were lost or misplaced in several cases. A vector which is shorter than
    the model is completed with the default configuration of the compiled model.
26. :commit:`3a35e7712` :ref:`mjs_attach` and the :ref:`attach<body-attach>` element now apply the prefix and suffix to
    the name and material of skins and to the material of tendons and meshes. Attaching a model in which one of these
    has a material used to fail to compile.
27. :commit:`cbd16feac` Fixed attaching by reference, the default of :ref:`mjs_attach`: :ref:`mj_recompile` gave the
    attached elements the state of other elements of the parent, and attaching another body or frame of the same model
    failed or left out the elements which refer to it. Attaching what the parent already has is now an error.
28. Fixed several bugs when editing or copying an :ref:`mjSpec`:

    - :commit:`239647a1d` Saving the compiled values of a spec which was structurally edited since it was compiled
      (elements added, deleted or attached) is now an error until it is compiled again; it used to crash or save wrong
      values.
    - :commit:`2267bf299`, :commit:`dc8210572` Compiling no longer reorders contact :ref:`pairs<contact-pair>` and
      :ref:`excludes<contact-exclude>`, writes into authored fields such as the data of a texture, or keeps references
      removed since the last compilation.
    - :commit:`2267bf299`, :commit:`dc8210572` :ref:`mjs_findElement` now finds elements added or renamed after
      compilation, and returns the named element for contact pairs, excludes and after
      :ref:`discardvisual<compiler-discardvisual>`.
    - :commit:`3a9b10a3c`, :commit:`3a35e7712` :ref:`mjs_delete` now deletes a :ref:`plugin instance<plugin-instance>`
      together with the last element referencing it, and deleting a body removes the elements referencing it after
      :ref:`discardvisual<compiler-discardvisual>` or later edits. Both cases used to make the next compilation fail or
      crash.
    - :commit:`592c54ad3` :ref:`mj_copyBack` and :ref:`mj_saveLastXML` no longer lose a change to the reference pose of
      a free joint, or write out of bounds for a :ref:`numeric<custom-numeric>` which has less data than its ``size``.

Actuation
^^^^^^^^^
29. :commit:`ab7f005c9` Actuator :ref:`shortcuts<CActShortcuts>` now inherit their parameters (``kp``, ``kv``, ``area``,
    ...) only from a :ref:`default<default>` written with the same shortcut or with :ref:`general<actuator-general>`,
    looking up the class hierarchy for one, and otherwise use their documented defaults; mechanical attributes like
    ``ctrlrange`` and ``gear`` are inherited from any default, as before. A :ref:`motor<actuator-motor>` in a class with
    a :ref:`position<actuator-position>` default used to be a position servo (:issue:`3561`). Every shortcut now sets
    its full documented table, including its dynamics and activation layout, and a :ref:`pid<actuator-pid>` without
    ``kp`` has the documented gain of 1, not 0. Added ``mjsActuator.type`` (:ref:`mjtActuator`), the element an actuator
    is written with.

    **Migration:** Write the gains on the actuator itself, or in a default of the same shortcut or of
    :ref:`general<actuator-general>`.
30. :commit:`fcb83a173` Actuators with "fixed" or "affine" :ref:`gaintype<actuator-general-gaintype>` can declare what
    their control is with :ref:`input<actuator-general-input>`: a position setpoint ("pos"), a velocity setpoint ("vel")
    or a pressure ("pressure"); without a declaration it is a command. The declaration does not affect the simulation.
    The :ref:`position<actuator-position>`, :ref:`velocity<actuator-velocity>`, :ref:`intvelocity<actuator-intvelocity>`
    and :ref:`cylinder<actuator-cylinder>` shortcuts declare their inputs, which are recorded in
    ``mjModel.actuator_ctrlspec`` and kept when a model is saved; :ref:`mj_actuatorInputName` returns the declaration.
    An input signature inherited from a default class is now discarded when the gaintype changes.
31. :commit:`92b8f4b27` Added the :ref:`inheritrange<actuator-general-inheritrange>` attribute to the
    :ref:`general<actuator-general>` actuator, which so far only the :ref:`position<actuator-position>`,
    :ref:`intvelocity<actuator-intvelocity>` and :ref:`pid<actuator-pid>` shortcuts had.

Filament Rendering
^^^^^^^^^^^^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   32. :commit:`4419c1be6`, :commit:`6678efd66`, :commit:`39b17ef2f`, :commit:`6c2137fe0` Renamed ``mjrf`` functions
       and types so that the owning object appears first (``mjrf_<verb><Owner><Thing>``) and names match MJCF attributes
       and existing MuJoCo vocabulary (cf. :ref:`mjr_uploadMesh`, ``mjModel.light_active``).

       **Migration:** Rename as follows:
       |br| ``mjrf_addRenderableToScene`` → :ref:`mjrf_addSceneRenderable`
       |br| ``mjrf_removeRenderableFromScene`` → :ref:`mjrf_removeSceneRenderable`
       |br| ``mjrf_addLightToScene`` → :ref:`mjrf_addSceneLight`
       |br| ``mjrf_removeLightFromScene`` → :ref:`mjrf_removeSceneLight`
       |br| ``mjrf_setMeshData`` → :ref:`mjrf_uploadMesh`
       |br| ``mjrf_setTextureData`` → :ref:`mjrf_uploadTexture`
       |br| ``mjrf_setRenderableTransform`` → :ref:`mjrf_setRenderablePose`
       |br| ``mjrf_setLightTransform`` → :ref:`mjrf_setLightPose`
       |br| ``mjrf_setLightEnabled`` → :ref:`mjrf_setLightActive`
       |br| ``mjrf_setLightCutoffAngle`` → :ref:`mjrf_setLightCutoff`
       |br| ``mjrf_setLightShadowsEnabled`` → :ref:`mjrf_setLightCastShadow`
       |br| ``mjrfReadPixelsRequest`` → :ref:`mjrfReadRequest`
       |br| :ref:`mjrfLightParams`: ``spot_cone_angle`` → ``cutoff``, ``spot_softness`` → ``softness``, ``cast_shadows`` →
       |br| ``cast_shadow``
       |br| :ref:`mjrfRenderableParams`: ``cast_shadows`` → ``cast_shadow``, ``receive_shadows`` → ``receive_shadow``
       |br| :ref:`mjrfRenderRequest`: ``enable_post_processing``, ``enable_reflections``, ``enable_shadows`` and
       |br| ``enable_headlight`` drop the ``enable_`` prefix
       |br| :ref:`mjrfContextConfig`: ``force_software_rendering`` → ``software_rendering``
       |br| :ref:`mjrfMaterial`: ``decor_ux`` → ``decor``
       |br| Support class ``ModelObjects`` → ``ModelResources``

       Python names changed accordingly, e.g. ``mjrf.ReadPixelsRequest`` → ``mjrf.ReadRequest``,
       ``Light.set_cutoff_angle`` → ``Light.set_cutoff``, ``Light.set_shadows_enabled`` → ``Light.set_cast_shadow``,
       ``Mesh.set_data``/``Texture.set_data`` → ``upload``, ``set_transform`` → ``set_pose``, ``Light.set_enabled`` →
       ``set_active``, ``Material.decor_ux`` → ``decor``, ``mjrf.ModelObjects`` → ``mjrf.ModelResources``.

   33. :commit:`e87cafb26` Renamed :ref:`custom flags<reFlags>` for consistency, and changed enum flags to take strings
       instead of integers.

       **Migration:** Rename as follows:
       |br| ``filament.clearColor`` → ``filament.clear_color``
       |br| ``filament.fog.cutOffDistance`` → ``filament.fog.cut_off_distance``
       |br| ``filament.fog.maximumOpacity`` → ``filament.fog.maximum_opacity``
       |br| ``filament.fog.heightFalloff`` → ``filament.fog.height_falloff``
       |br| ``filament.fog.inScatteringStart`` → ``filament.fog.in_scattering_start``
       |br| ``filament.fog.inScatteringSize`` → ``filament.fog.in_scattering_size``

       ``filament.ao.quality``, ``filament.ao.low_pass_filter``, ``filament.ao.upsampling`` and
       ``filament.bloom.quality`` take "low", "medium", "high" or "ultra" instead of 0-3; ``filament.shadows.type``
       takes "pcf", "vsm" or "pcss" instead of 0-2.

   34. :commit:`e7b0bc104` Removed ``mjrf_setLightBlurWidth`` and ``mjrfLightParams::vsm_blur_width``, and their Python
       counterparts ``mjrf.Light.setBlurWidth`` and ``mjrf.LightParams.vsm_blur_width``. This removes soft shadows with
       VSM.

       **Migration:** Use PCSS shadows instead by setting the custom flag ``filament.shadows.type`` to ``pcss``. PCSS
       shadows provide similar quality and are more performant.

   35. :commit:`e71e9f6db` Removed the ``filament.msaa.enabled`` custom flag; multi-sample anti-aliasing is now
       controlled by :ref:`visual/quality/offsamples<visual-quality-offsamples>` (``<= 1`` disables MSAA) and can be
       changed at runtime with :ref:`mjrf_setSceneMsaa`.

   36. :commit:`6c2137fe0` Removed ``mjrf_DEBUG_drawImguiEditor`` from the public API; the ImGui scene editor is an
       internal debugging aid for MuJoCo Studio.

Python bindings
^^^^^^^^^^^^^^^
37. :commit:`43eff4c76` The ``opt``, ``vis`` and ``stat`` attributes of ``MjModel``, and the sub-struct attributes of
    ``MjVisual``, can now be assigned to (e.g. ``model.opt = new_opt``); previously this raised ``AttributeError``.
    Assignment copies the fields into the underlying C struct in place, so existing references and NumPy array views
    into the destination remain valid.

Models
^^^^^^
38. :commit:`dc0c237df` Added three
    `elastic mechanism <https://github.com/google-deepmind/mujoco/tree/main/model/flex/mechanisms/>`__ example models
    using multicell trilinear flexes: a cantilever material comparison, an elastic slider-crank, and a tendon-loaded
    dipper arm. Inspired by
    `Miles Macklin's Reduced Elastic Links experiments <https://reports.mmacklin.com/newton-reduced/reduced_elastic_links_implementation.html>`__.

Documentation
^^^^^^^^^^^^^
39. :commit:`641652525` Added documentation for :ref:`Filament rendering <FilamentRendering>`.
40. :commit:`1b7d9e83b` The :ref:`XML reference<CDimension>` now gives the physical dimension of every real-valued
    attribute, declared in the MJCF schema.

Version 3.15.0 (October 5, 2026)
--------------------------------

General
^^^^^^^
1. :commit:`1aa68e1ca` :ref:`.mjz <MJZArchives>` archives written on Windows now always use ``/`` separators.
2. :commit:`1aa68e1ca` Assets in a :ref:`meshdir<compiler-meshdir>` or :ref:`texturedir<compiler-texturedir>` now
   resolve correctly in :ref:`.mjz <MJZArchives>` archives when rewritten or when the directory path is absolute, uses
   ``..``, or uses a URI scheme.

Engine
^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   3. :commit:`b3dd9e617` Actuator :ref:`dampratio<actuator-position-dampratio>` now computes the reflected inertia
      :math:`m` at ``mjModel.qpos0`` as the operational-space inertia :math:`(J M^{-1} J^T)^{-1}` (averaged across force
      outputs for multi-output actuators such as :ref:`orientation<actuator-orientation>`), rather than summing
      :math:`\text{dof\_M0}_j / J_j^2` across degrees of freedom. This accounts for tendon and actuator armature as well
      as off-diagonal inertial coupling, and avoids extreme damping values when a multi-DOF transmission has small
      Jacobian entries. In practice, actuators using ``dampratio`` on multi-link kinematic chains may exhibit lower
      damping than before, while actuators driving tendons or sites that span multiple joints will be damped much more
      accurately instead of being overdamped.

   4. :commit:`5023a4a50` Removed the deprecated ``internal`` flex collision option and associated ``evpair``
      structures.

5. :commit:`1b973add4` Added experimental simplified Stable Neo-Hookean elasticity for non-interpolated 3D flexes,
   enabled only through ``mjsFlex.elastic3d = 1``. It projects the material Hessian to positive semidefiniteness for the
   solver and Rayleigh damping, and preserves the Saint Venant-Kirchhoff (StVK) default (``0``).
   SNH requires the discrete integrator. The setting is not available in MJCF.
6. :commit:`2742de438` The cached flex bending factor now retains cross-coordinate couplings between differently
   oriented vertex bodies.
7. :commit:`eed008905` Flex bending and stretching now include the motion and reaction forces of articulated vertex
   attachments. The discrete integrator supports these attachments with the CG solver; fixed and independent XYZ-slide
   attachments retain their optimized assembly. Elastic flexes attached to mocap bodies produce a compiler error.
8. :commit:`6224e95f6` Added single-shot :ref:`multicontact<coMultiCCD>` for collisions with capsule geoms.
9. :commit:`ebfb4662d` Added the :ref:`enclosed<sensor-insidesite-enclosed>` attribute to
   :ref:`insidesite<sensor-insidesite>` sensors, measuring how much an object juts out of a site (using directed
   Hausdorff distance) and reporting signed clearance/protrusion.
10. :commit:`f80ef4244` Fixed 64-bit model size narrowing in binary MJB serialization: :ref:`mj_saveModel` and
    :ref:`mj_loadModelBuffer` now accept :ref:`mjtSize` for ``buffer_sz``, supporting models larger than 2 GiB.

Version 3.14.0 (September 22, 2026)
-----------------------------------

General
^^^^^^^

1. :commit:`458189298` Added the ``ipc`` :ref:`flag<option-flag-ipc>`, an experimental contact mode of the ``discrete``
   integrator for penetration-free flex contact. Each step minimizes an incremental potential subject to linearized
   contact constraints, using a barrier-free augmented Lagrangian whose subproblems are the discrete solve. Every
   committed position update is verified intersection-free by continuous collision detection, so flex contact cannot
   tunnel. Supported for dim-2 flexes: a flex with edge equality constraints keeps its elasticity in the constraint
   solver, while ``elastic2d`` elasticity is integrated implicitly through the effective metric. The contacts the mode
   resolves are frictionless. The mode keeps contact multipliers across steps that no state specification covers, so
   ``mj_getState``/``mj_setState`` do not capture its full state and exact replay is not supported.
2. :commit:`2fec92237` Introduced :ref:`archive resource providers<mjp_registerArchiveProvider>`
   (``mjp_registerArchiveResourceProvider``). Archive providers use the
   :ref:`mjpResourceProvider` interface to mount and read archive containers (such as
   ``.mjz``/``.zip``), decoupling container handling from format decoders and enabling
   on-demand asset extraction without requiring a pre-allocated :ref:`VFS<Virtualfilesystem>`.
3. :commit:`e312ce82d` :ref:`Frames<frame>` are now preserved when saving MJCF: ``<frame>`` elements are written with
   their ``pos`` and ``quat`` and their contents in frame-relative coordinates, so a saved model reloads with the same
   frames. Previously the frame transformation was accumulated into the children on save and only the frame's name was
   kept. Changes to the ``pos`` or ``quat`` of a frame in an :ref:`mjSpec` now also take effect when the spec is
   recompiled.


Engine
^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   4. :commit:`dc8bb1364` :ref:`mj_readCtrl` now matches the signature and semantics of :ref:`mj_readSensor` to support
      multi-input actuators (e.g., :ref:`pid<actuator-pid>`, :ref:`dcmotor<actuator-dcmotor>`,
      :ref:`orientation<actuator-orientation>`): it takes an output buffer ``result`` of size ``actuator_ctrlnum[id]``
      and returns ``const mjtNum*`` (a pointer into ``mjData.ctrl`` or ``mjData.history`` on zero-order-hold or exact
      timestamp matches, or ``NULL`` when interpolated values are written to ``result``). Actuator history buffers now
      store ``actuator_ctrlnum[id]`` values per sample, and :ref:`mj_initCtrlHistory` expects ``nsample *
      actuator_ctrlnum[id]`` values (:issue:`3597`).

5. :commit:`9e0c6b577` The :ref:`mjWARN_INERTIA <mjtWarning>` warning is now also raised by the modified-inertia
   factorizations of the :at:`implicitfast` and damped-:at:`Euler` integrators (previously silent) and of the
   :at:`implicit` integrator (previously a fatal error).
6. :commit:`35c0631de` The sparse Newton solver no longer aborts with a "rank-deficient sparse Hessian" error when
   rounding loses a pivot of its Hessian, as can happen in single precision with ill-conditioned inertia. The pivot is
   now clamped and its row decoupled, as in the dense factorization.

Bug fixes
^^^^^^^^^
7. :commit:`2a3f1f5a4` Procedural model editing is no longer quadratic in the number of elements. ``mjSpec`` previously
   recomputed the compilation signature on every :ref:`mjs_addGeom`-like call, and :ref:`mjs_setName` rescanned every
   name of the element's type. Both are now incremental. Building a spec with 8000 geoms is roughly 10x faster.

   .. admonition:: Breaking API changes
      :class: attention

      ``mjsElement.signature`` is now ``0`` whenever the spec has been structurally edited since it was last compiled,
      rather than being eagerly recomputed on each edit. It is still equal to ``mjModel.signature`` after a successful
      :ref:`mj_compile`, which is what :ref:`mj_copyBack` and the Python ``bind`` methods rely on. Code that compared the
      signature of an uncompiled spec against anything other than ``0`` must compile first. Additionally, when
      :ref:`mjs_setName` fails on a duplicate name, the element's previous name is now preserved rather than being
      overwritten with the duplicate name.
8. :commit:`ac329bd17` Fixed the torque applied by :ref:`weld<equality-weld>` constraints in
   :ref:`mj_rnePostConstraint`. The rotational constraint force was used directly as a world-frame torque, missing the
   scaling by ``0.5 * torquescale`` and the rotation into the world frame. :ref:`Force<sensor-force>` and
   :ref:`torque<sensor-torque>` sensors on bodies held by a weld that carries torque were wrong (:issue:`2533`).
9. :commit:`263c638c0` :ref:`mj_rnePostConstraint` now accounts for the forces of spatial tendons (spring, damper,
   actuator, limit, friction loss, equality and armature), so :ref:`force<sensor-force>` and
   :ref:`torque<sensor-torque>` sensors register the forces that tendons apply to bodies (:issue:`832`).
10. :commit:`eb662fcaf` Tendon :ref:`actuatorfrclimited<tendon-spatial-actuatorfrclimited>` now defaults to "auto" as
    documented. Previously the default was "false" and :at:`actuatorfrcrange` was silently ignored unless
    :at:`actuatorfrclimited` was set explicitly.
11. :commit:`54be9cce6` :ref:`mjd_transitionFD` and :ref:`mjd_inverseFD` now raise an error when
    :ref:`sleeping<Sleeping>` is enabled. Previously, their repeated evaluations changed the sleep state, leading to
    internal errors or wrong derivatives.
12. :commit:`e38573c64` Fixed a memory leak in :ref:`mjs_delete` where deleted elements and subtrees were retained in
    ``mjSpec`` until :ref:`mj_deleteSpec` (:issue:`2882`).
13. :commit:`65ea54c3c` Fixed an out-of-bounds read when parsing the header of a :ref:`GMSH file<gmsh-file-docs>` loaded
    by :ref:`flexcomp<body-flexcomp>`. Truncated headers are now reported as an error.

Version 3.13.0 (September 8, 2026)
----------------------------------

General
^^^^^^^
1. :commit:`eb18d77ca` The :ref:`.mjz <MJZArchives>` encoder now writes the root file as ``model.xml`` in the archive as
   this is less susceptible to breakage due to file renaming.
2. :commit:`4affbb64a` Site geometries can now also be associated with meshes (:ref:`type="mesh"<body-site-type>` with
   :ref:`mesh="name"<body-site-mesh>`), supporting visualization, the :ref:`insidesite<sensor-insidesite>`
   sensor, and the new :ref:`mj_insideSite` function.
3. :commit:`b59b07fae` Added support for Python 3.15 (GIL and Free-Threading).

Engine
^^^^^^
4. :commit:`5660353ec` Added a new integrator ``discrete``: the constraint solve and the implicit velocity update merge
   into one operation, performed in the effective metric :math:`\widehat{M} = M + hD + h^2K`, which incorporates both
   implicit damping :math:`hD` and implicit position stiffness :math:`h^2 K`. Under this integrator ``mjData.qacc`` is
   the discrete step map :math:`(v^+ - v)/h`, and joint, tendon and actuator stiffness and damping join the solver's
   metric, making passive springs and actuator position gains stable at timesteps far beyond the explicit stability
   limit. Constraint rows are treated implicitly as well: :at:`solref` spring--dampers are evaluated at the end of the
   step, so constraints are stable at any ``timeconst``; under this integrator the :ref:`refsafe<option-flag-refsafe>`
   flag replaces contact and limit rows stiffer than the timestep can resolve by the stiffest zero-restitution row
   instead of clamping ``timeconst``. The actuator-gain treatment resolves the stiff-servo timestep limitation of
   :issue:`3443` (analysis contributed by :github:user:`qiayuanl`). See the
   :ref:`integrator documentation<geIntegrators>` for semantics and current limitations.

   .. admonition:: Breaking API changes
      :class: attention

      Removed the implicit flex effective-metric special case under ``implicit``/``implicitfast`` with the ``CG``
      solver, introduced in 3.11.0. This behavior now requires ``integrator="discrete"`` (with a primal solver:
      ``CG`` or ``Newton``), which additionally treats joint damping and stiffness implicitly inside the solve. Models
      relying on the old behavior should set :ref:`integrator<option-integrator>` to ``discrete``; models with flex
      elasticity or passive flex contact under ``implicit``/``implicitfast`` now raise a runtime error carrying this
      migration note.

5. :commit:`0b4e17747` Restored clamping of non-positive pivots in the sparse inertia factorization, along with the
   associated ``mjWARN_INERTIA`` warning. The guard was inadvertently dropped in the 3.3.0 conversion of ``qLD`` to CSR
   format; since then, models with singular mass matrices silently produced non-finite accelerations, typically
   surfacing as divergence warnings and automatic resets.
6. :commit:`c9d997610` Added single-shot :ref:`multicontact<coMultiCCD>` for collisions with cylinder geoms.
7. :commit:`10eeb8289` The Newton solver with :ref:`elliptic cones<option-cone>` now rebuilds the cone-augmented Hessian
   factor with a single refactorization instead of per-contact rank-1 updates when a flop-count model predicts this is
   faster. Scenes with many simultaneously sliding contacts speed up by 1.4-2x on average and 3-4x on the slowest steps.
   Contribution by :github:user:`kevinzakka`.

Compiler
^^^^^^^^
8. :commit:`2a3957558` Custom text fields (:ref:`custom/text<custom-text>`) in MJCF now accept their values inside a
   ``<![CDATA[ ... ]]>`` block in addition to the ``data`` attribute. When saving a model via :ref:`mj_saveXML`, custom
   text containing newlines or XML characters is exported as CDATA.
9. :commit:`47ebfe48a` Added :ref:`cutoff<sensor-tactile-cutoff>` attribute to :ref:`sensor/tactile<sensor-tactile>`.

Samples
^^^^^^^
10. :commit:`834983fbb` Added :ref:`render.cc<saRender>`: a headless offscreen rendering sample that saves an image to a
    PNG image. Supports both the Filament and classic OpenGL backends, with command-line options for camera, resolution,
    keyframe, simulation steps, geom/site groups, visualization and rendering flags, labels, and frames.

Bug fixes
^^^^^^^^^
11. :commit:`67562d95f` Rewrote the Plane-Mesh collider to fix bugs and improve performance. Fixes :issue:`3524`.

.. youtube:: MNQsV_etq24
   :aspect: 16:7
   :align: right
   :width: 35%

Models
^^^^^^
12. :commit:`f6c1004b7` Added `house of cards <https://github.com/google-deepmind/mujoco/tree/main/model/cards/>`__
    example model and three `free-standing arch <https://github.com/google-deepmind/mujoco/tree/main/model/arch/>`__
    models: roman, gothic, and hyperbolic. Inspired by `Song et al <https://www.cs.ubc.ca/research/fbf-friction/>`__.

Rendering
^^^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   13. :commit:`cb57b526f` For materials with physically-based rendering parameters that do *not* provide a metallic
       texture the fallback is now white. Previously the fallback was black, which caused scalar metallic values to be
       discarded when multiplied in the PBR shader.

Version 3.12.0 (August 20, 2026)
--------------------------------

General
^^^^^^^

1. :commit:`3f8db4c1` The MJCF grammar is now defined in a single source of truth schema file,
   `src/xml/mjcf.schema <https://github.com/google-deepmind/mujoco/tree/main/src/xml/mjcf.schema>`__. The parser's
   grammar table, presence constraints, keyword maps, typed attribute bindings and save policies are generated from it
   and gated by tests, as are the schema's enum keywords and declared defaults against the C headers and
   default-constructors.

.. admonition:: Breaking API changes
   :class: attention

   2. :commit:`6fe04aa8` Removed the custom binary texture format (``image/vnd.mujoco.texture``) and the automatic
      fallback to custom textures when loading files with unrecognized extensions. Textures can now only be loaded from
      PNG (``image/png``) and KTX (``image/ktx``) files.

Actuation
^^^^^^^^^
3. :commit:`279df98c` Added the :ref:`pid<actuator-pid>` actuator: a PID controller with real position and velocity
   setpoint inputs, optional integral action (:ref:`ki<actuator-pid-ki>`, integrating the position error with
   :ref:`imax<actuator-pid-imax>` anti-windup), setpoint rate limiting (:ref:`slewmax<actuator-pid-slewmax>`), and an
   optional feedforward input. This subsumes the functionality of the ``mujoco.pid`` plugin with proper activation
   state: correct under all integrators and visible to keyframes and sensors. With a zero velocity setpoint it is
   identical to :ref:`position<actuator-position>`. The input signature is any subset of ``[pos, vel, ff]``, selected
   by :ref:`input<actuator-pid-input>`; absent setpoint inputs are fixed at zero, so the control vector contains no
   inert entries.
4. :commit:`2f1843f4` The :ref:`dcmotor<actuator-dcmotor>` on-board controller is redesigned: the
   :ref:`input<actuator-dcmotor-input>` attribute selects any subset of ``[pos, vel, ff, voltage]``, where ``pos``
   and ``vel`` are setpoints for the controller, ``ff`` is a torque feedforward, and ``voltage`` is the raw terminal
   voltage (the default, a plain voltage-commanded motor). Controller gains are in torque space, as for
   :ref:`pid<actuator-pid>`, and the drive voltage compensates back-EMF as in a current-controlled driver: commanded
   torque is delivered exactly until a limit is reached. The keyword ``input="none"`` selects the empty signature:
   the actuator has no control inputs and is purely passive, so friction, cogging and back-EMF braking can be used
   as passive joint forces.

.. admonition:: Breaking API changes
   :class: attention

   5. :commit:`2f1843f4` The mode-flag semantics of :ref:`dcmotor/input<actuator-dcmotor-input>` ("voltage", "position",
      "velocity", selecting the interpretation of a single control) are replaced by input signatures, and the controller
      gains changed from voltage space to torque space. The old velocity mode's integral term (integrated-velocity
      tracking) is retired without replacement; the integrator always accumulates position error.

      **Migration:** Voltage-commanded motors (the default) are unchanged. Replace ``input="position"`` with
      ``input="pos"`` and ``input="velocity"`` with ``input="vel"``, and multiply the controller gains by
      :math:`K/R` (torque per volt). The motor's back-EMF damping, previously felt in addition to the controller's
      damping, is now compensated: to preserve behavior when the velocity setpoint is zero, add :math:`K^2/R` to the
      converted :at-val:`kd`.

Engine
^^^^^^

6. :commit:`83e621d7` Optimized large-mesh convex collision detection with up to 2x speedup in certain cases.
7. :commit:`55d13aec` Replaced the per-step sparse Cholesky factorization of the flex block of the implicit effective
   metric M + K with its prefactored per-vertex 3x3 diagonal blocks. The blocks precondition the CG constraint solver
   and drive an iterative solve for ``qacc_smooth``, which now converges on :ref:`tolerance<option-tolerance>` rather
   than a fixed threshold. Flexes with :ref:`elastic2d<flex-elasticity-elastic2d>` stretch stiffness step roughly twice
   as fast; bending-only flexes keep the exact constant factor and are unchanged.
8. :commit:`86e98601` Rewrote cleaner box-box SAT collider.

.. admonition:: Breaking API changes
   :class: attention

   9. :commit:`2a3554c8` Contacts of a flex with :ref:`passive<flexcomp-contact-passive>` collisions are now integrated
      implicitly: their stiffness is carried by the effective metric rather than applied as an explicit spring, and can
      be far stiffer than the timestep would otherwise permit. Models using passive collisions should be re-checked: the
      feature now requires :at:`implicit` or :at:`implicitfast` with the CG solver, pyramidal cones and sleep
      disabled; passive handling covers flex-flex, self-, and static-geometry contact, while contact with a moving
      body stays on the constraint solver; and the stiffness is now a mass-scaled natural frequency rather than a
      fixed 1e4.

   10. :commit:`55d13aec` Removed ``mjData.efm_L_rownnz``, ``mjData.efm_L_rowadr`` and ``mjData.efm_L_colind``. They
       described the sparsity of the effective-metric Cholesky factor, which no longer exists; ``mjData.efm_L`` now
       holds dense 3x3 blocks, 9 numbers per covered vertex. ``mjData.efm_active`` no longer takes the value 2: nothing
       selects a solve path on preconditioner exactness, so it is now a plain 0/1 flag.
   11. :commit:`1362a8bd` Changed the default value of :ref:`bvactive<visual-global-bvactive>` from "true" to "false".
       This avoids unnecessarily clearing bounding volume hierarchy visualization flags at every simulation step, which
       can be a bottleneck for models with large meshes.
   12. :commit:`ed13bf56` Mocap bodies and their dof-less descendants are now the root of their own weld group:
       ``mjModel.body_weldid`` of a mocap body equals its own id rather than 0. Consequences: dragging a mocap body into
       sleeping objects now wakes them; children of mocap bodies receive standard
       :ref:`parent-child collision filtering<SurprisingCollisions>`; mocap bodies no longer count as static geometry
       for ray casting, and contact-matching sensors aggregate their contacts under the mocap body rather than the
       world; and geom pairs where neither body can move no longer generate contacts.

Models
^^^^^^

13. :commit:`2a3554c8` Added `drape <https://github.com/google-deepmind/mujoco/blob/main/model/flex/drape.xml>`__
    example model: three cloths draped over a sphere, demonstrating :ref:`passive<flex-contact-passive>` collisions. It
    replaces the ``sphere_passive`` model, which has been removed.

14. :commit:`55d13aec` Added `bag <https://github.com/google-deepmind/mujoco/blob/main/model/flex/bag.xml>`__ example
    model: a cloth bag, held open by pinning the ring of vertices around its mouth, catching the standard humanoid
    dropped in from above. Unlike the poncho models, which are bending-only, this model exercises the 2D
    :ref:`stretch<flex-elasticity-elastic2d>` elasticity of a flex.

Rendering
^^^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   .. image:: https://www.gstatic.com/mujoco/doc/images/changelog/primitives_textured.gif
      :align: right
      :width: 40%

   15. :commit:`cc7fb98c` Added explicit texture coordinates to built-in geometries (Plane, Box, Sphere, Ellipsoid,
       Capsule, Cylinder) in both the Classic renderer and Filament. 2D textures applied to primitive shapes will look
       different as textures are mapped using canonical UV parameterizations rather than projecting onto the :math:`x,y`
       plane.

       For finite planes, textures are now anchored to the bottom-left corner instead of the center. This will cause the
       most common visual breakage, as common procedural checker textures will be phase shifted. Infinite planes continue
       to be anchored at the origin with no visual changes.

       .. image:: images/changelog/plane_uv_tiling.png
          :align: center
          :width: 70%
   16. :commit:`f9a00bd5` Added :ref:`light/softness<body-light-softness>`: edge softness for spotlights under
       physically-based lighting models, given as the fraction of the cone over which intensity falls to zero. The
       default of 0.2 is a semi-soft cone which delivers the full :ref:`intensity<body-light-intensity>` everywhere
       inside it, so that illuminance follows :math:`E = I/d^2` independent of the :ref:`cutoff<body-light-cutoff>`
       angle. Previously the filament renderer treated the entire cone as penumbra, dimming spotlights well below their
       rated intensity, increasingly so for narrow cutoffs.

       **Migration:** Set :at:`softness` to 1 to reproduce the previous appearance of existing models.

MJX
^^^

.. admonition:: Breaking API changes
   :class: attention

   17. :commit:`5e3464f4` :func:`mjx.render` and :func:`mjx.render_with_segmentation` now return the updated
       :class:`mjx.Data` as the last element in their return tuple (i.e. ``(rgb, depth, d)`` and
       ``(rgb, depth, seg, d)``). This ensures JAX/XLA strictly enforces causal scheduling between sequential
       ``refit_bvh`` and ``render`` calls.

       **Migration:** Update unpacking calls from ``pixels, depth = mjx.render(mx, d, rc)`` to
       ``pixels, depth, d = mjx.render(mx, d, rc)``.

Bug fixes
^^^^^^^^^

18. :commit:`95539261` Fixed a bug where models with pinned interpolated flex nodes (e.g. a
    :ref:`flexcomp<body-flexcomp>` with :at:`dof` "trilinear" and pinned vertices) could not be reloaded after saving:
    node coordinates within their body frames were not saved, degenerating the interpolation grid. They are now saved in
    the new flex :ref:`nodecoord<deformable-flex-nodecoord>` attribute.
19. :commit:`8655446f` Fixed a bug in the box-box collider where near-degenerate face clipping could generate contacts
    with spuriously large penetration depth between nearly touching thin boxes with positive margin, causing resting
    stacks to explode.
20. :commit:`fb07a9ca` Fixed a bug in the box-box collider where penetrations deeper than a box's smallest half-size
    could produce no contacts, letting boxes tunnel through thin boxes. Fixes :issue:`1800`.
21. :commit:`54979947` Fixed the flex stretch stiffness operator, which was the Gauss-Newton Hessian of the stretch
    force rather than its Jacobian: the geometric (stress-proportional) term was missing. Only the tensile part of that
    term is added, since it is positive semi-definite exactly when the edge is in tension, and its consumers require an
    SPD operator; the stretch force itself is unchanged. This affects the implicit integrators and the implicit
    effective metric, so flexes using ``elastic2d="stretch"`` integrate slightly differently. Bending-only flexes are
    unaffected.

OpenUSD
^^^^^^^

22. :commit:`39e44588` Upgraded Newton USD schemas support to version 0.4.0:

    - ``NewtonJointAPI`` (``newton:armature``, ``newton:damping``, ``newton:friction``) deprecates the ``MjcJointAPI``
      equivalent ``mjc:armature``, ``mjc:damping``, and ``mjc:frictionloss`` attributes.
    - ``NewtonMassAPI`` (``newton:massModel``, ``newton:inertia``) deprecates the ``MjcCollisionAPI``
      equivalent ``mjc:shellinertia`` and ``MjcMeshCollisionAPI`` ``mjc:inertia`` attributes. This completes the
      deprecation of all ``MjcMeshCollisionAPI`` attributes, slating it for removal in a future release.
    - Added support for ``NewtonSiteAPI`` to declare sites, ``MjcSiteAPI`` auto applies this schema, but remains
      as an extension for the ``mjc:group`` attribute.
    - Added support for ``NewtonMaterialAPI`` (``newton:contactAdhesion``, ``newton:torsionalFriction``,
      ``newton:rollingFriction``). This deprecates ``MjcMaterialAPI`` which will be removed in a future release.
    - Added support for ``NewtonMimicAPI`` (``newton:mimicJoint``, ``newton:mimicCoef0``, ``newton:mimicCoef1``) as a
      base for ``MjcEqualityJointAPI``, this deprecates the ``mjc:coef0`` and ``mjc:coef1`` attributes and the
      ``mjc:target`` relationship.
    - Added support for ``NewtonArticulationRootAPI`` (``newton:jointsAddMobility``).

.. admonition:: Breaking ABI changes
   :class: caution

   23. :commit:`279df98c` :ref:`mjsActuator` gained ``velrange`` and ``ffrange`` fields, changing its size and layout.
       The :ref:`mjtGain` and :ref:`mjtDyn` enums gained ``pid`` members, shifting the values of ``mjGAIN_USER`` and
       ``mjDYN_USER``.
   24. :commit:`596b6f43` :ref:`mjResource` gained an ``args`` field (changing its size and layout), used to hold
       optional extra encoding and decoding arguments formatted as URI query parameters (separated by ``&``).

Version 3.11.0 (July 27, 2026)
------------------------------

Engine
^^^^^^

.. youtube:: PdSdrqhSiZA
   :aspect: 16:7
   :align: right
   :width: 35%

1. :commit:`4787c809` Added :ref:`geom/surfacevel<body-geom-surfacevel>`: the velocity of a geom's surface as seen by
   contacts, given as a velocity field with a constant component and a rotational component about the geom frame origin.
   This allows conveyor belts, treadmills and turntables to be modeled with static geoms and no degrees of freedom:
   friction drives touching bodies along the motion of the surface, with the field projected onto each contact's tangent
   plane. Surface velocities compose correctly with each other and with body motion.
   Note that the contact rows of ``mjData.efc_vel``, and the constraint-state sensors that read them, report the
   velocity relative to the moving surface rather than to the geom, since that is the quantity the constraint acts
   on; for geoms without :at:`surfacevel` the two are identical. Contact-point visualization draws an arrow along the
   surface velocity at contacts with moving surfaces.

.. youtube:: GioWwB36XHI
   :aspect: 16:7
   :align: right
   :width: 35%

2. :commit:`a264d0bc` Added :ref:`geom/adhesion<body-geom-adhesion>` and :ref:`pair/adhesion<contact-pair-adhesion>`: an
   adhesive force associated with a contact, useful for modeling sticky materials. Contacts can pull with up to the
   given force before breaking, and the friction budget becomes :math:`\mu(f_N + \text{adhesion})`. Combined with
   :ref:`gap<body-geom-gap>`, adhesive contacts apply "adhesion at a distance", useful for modeling magnets. Resting
   penetration is unaffected by adhesion. :ref:`mj_contactForce` reports the net interface force, whose normal component
   can now be negative.
3. :commit:`f0fa3d82` Replaced midpoint integration of free bodies with :ref:`gyroscopic derivatives<geFreeBody>` in the
   ``implicitfast`` :ref:`integrator<geIntegrators>`: the bias-force derivative of every standalone free body is applied
   via a local unsymmetric solve of its decoupled block, making ``implicitfast`` identical to ``implicit`` for such
   bodies. Unlike midpoint integration, which required vacuum and no constraints, this applies in all environments
   (contacts, fluid, constraints), and is compatible with discrete-time inverse dynamics. Spinning free bodies no
   longer gain energy, but tumbling motion is now mildly damped; models requiring long-horizon energy conservation of
   tumbling bodies in vacuum should use ``RK4``. The :ref:`invdiscrete<option-flag-invdiscrete>` flag no longer has any
   effect on forward dynamics.
4. :commit:`5618666a` Added :ref:`body/simple<body-simple>` attribute ("false"/"auto") to disable the *simple body* mass
   matrix optimization. This is useful for domain randomization, where model parameters may change post-compilation.
5. :commit:`14c0b0c9` :ref:`mj_setConst` now recomputes the ``mjModel.{body,geom,site}_sameframe`` flags, to account for
   changes in body/geom/site frames after compilation.
6. :commit:`2444defc` Added support for :ref:`multiccd <coMultiCCD>` with arbitrarily large meshes.
7. :commit:`a04b0c5b` Added ``flg_gravcomp`` and ``flg_surfacevel`` boolean flags to ``mjModel``. These flags replace
   the fast-path checks as originally guarded by ``ngravcomp``. Since the engine uses these integers as flags (zero vs.
   non-zero), the new flags are honest boolean properties, writeable from the Python bindings at runtime. The field
   ``ngravcomp`` is deprecated and will be removed in a future release.
8. :commit:`a1f38c8e` Replaced quadratic scratch in DFS flood-fill island discovery with a linear-memory Union-Find
   (disjoint set). Contribution by :github:user:`teerthsharma`.

.. admonition:: Breaking API changes
   :class: attention

   9. :commit:`ff629889` Changed the default value of :ref:`sleep_tolerance<option-sleep_tolerance>` from 1e-4 to 1e-3
      (1mm/sec in SI units).
   10. :commit:`315bcfbf` Removed the legacy sparse ancestor-walk inertia matrix ``mjData.qM``. The joint-space inertia
       matrix is now stored exclusively in the compressed sparse row (CSR) format ``mjData.M``.
   11. :commit:`7e9ac58f` Switched :ref:`mjd_inverseFD` to use the CSR-format ``mjData.M`` representation instead of the
       legacy ``mjData.qM`` for the mass matrix derivative. This changes the shape of the ``DmDq`` parameter from
       ``(nv x nM)`` to ``(nv x nC)``.
   12. :commit:`1ea2d884` :ref:`mju_round` now breaks ties away from zero rather than towards :math:`+\infty`. This only
       affects negative half-integers, e.g. ``mju_round(-2.5)`` now returns -3 rather than -2.
   13. :commit:`fa36015b` Removed unneeded `mjvScene` argument from :ref:`mjv_moveCamera`.
   14. :commit:`ba9a6503` Split up :ref:`mjrfMeshData` into `mjrfMeshData` and `mjrfMeshConfig` to allow reuploading of
       mesh data without having to recreate the mesh object. Introduces ``mjrf_defaultMeshConfig`` and
       ``mjrf_setMeshData`` functions.
   15. :commit:`ba9a6503` Removed `bytes` field from :ref:`mjrVertexAttribute`.

.. admonition:: Breaking ABI changes
   :class: caution

   16. :commit:`072e963f` :ref:`mjModel` gained the ``actuator_ctrlspec`` field (input signature of each actuator), and
       :ref:`mjsActuator` gained ``ctrlspec``, changing their size and layout. The :ref:`mjtGain` and :ref:`mjtBias`
       enums gained ``so3`` members, shifting the values of ``mjGAIN_USER`` and ``mjBIAS_USER``.
   17. :commit:`d43c3ed4` Added ``texid``, ``texuniform`` and ``texrepeat`` fields to :ref:`mjvGeom`.
   18. :commit:`a264d0bc` The :ref:`mjContact` struct gained an ``adhesion`` member, changing its size and layout.

.. admonition:: Bug fixes
   :class: admonition

   19. :commit:`dddb2767` Fixed a bug where ``body_margin`` excluded ``gap``, causing the mid-phase collision filter to
       incorrectly prune in-gap contacts on multi-geom bodies.

Actuation
^^^^^^^^^
20. :commit:`d507e921` Refactored actuator infrastructure in preparation for MIMO (multi-input multi-output) actuator
    support. Each actuator now has ``ctrlnum`` (number of controls) and ``outnum`` (number of force outputs). The total
    counts ``nu = sum(ctrlnum)`` and ``nout = sum(outnum)`` dimension ``mjData.ctrl`` and ``mjData.actuator_force``,
    respectively, ``nactuator`` is the number of actuators. For existing actuators ``ctrnum = outnum = 1``, so
    ``nactuator == nu == nout`` and existing code is unaffected.
21. :commit:`56a93979` Setpoints of :ref:`position<actuator-position>` and :ref:`intvelocity<actuator-intvelocity>`
    servos acting on 3D rotational transmissions (ball joints, or site transmissions with a
    :ref:`refsite<actuator-general-refsite>` and purely rotational gear) are now interpreted on the circle: the force
    uses the setpoint representative nearest the current angle, so targets winding beyond half a turn are tracked
    continuously instead of slipping by full turns. Behavior is identical whenever the error does not exceed π.
    Relatedly, ``intvelocity`` actuators now expose :ref:`actlimited<actuator-intvelocity-actlimited>`, which was
    previously hardcoded to "true": as for :ref:`general<actuator-general>` actuators it defaults to "auto", so
    activation clamping is enabled by specifying ``actrange``. Unclamped integrated setpoints are well-behaved on
    rotational transmissions, where they wrap.

.. youtube:: 17XpwnqyCXs
   :aspect: 16:7
   :align: right
   :width: 35%

22. :commit:`072e963f` Added the :ref:`orientation<actuator-orientation>` actuator: a geodesic servo on a new SO(3)
    transmission (ball joints, or a site with a :ref:`refsite<actuator-general-refsite>`), acting jointly on the full
    relative orientation with an exact equilibrium at every commanded orientation. This is the first actuator with
    multiple force outputs (3), and, with ``input="quat"``, the first with different input and output dimensions (4
    controls, 3 outputs). The input signature is recorded in the new ``mjModel.actuator_ctrlspec``, exposed as the
    :ref:`input<actuator-general-input>` attribute.
23. :commit:`072e963f` Added :ref:`mj_actuatorInputName`, returning the name of an actuator input (e.g. "qw" for the
    first control of a quaternion-commanded orientation actuator). The control sliders in :ref:`simulate<saSimulate>`
    and MuJoCo Studio are now generated per control and labeled with the actuator name plus the input name suffix.
24. :commit:`072e963f` Viewer control sliders now use a defined :ref:`ctrlrange<actuator-general-ctrlrange>` even when
    :ref:`ctrllimited<actuator-general-ctrllimited>` is "false": the range sets the slider span, while clamping remains
    controlled by :at:`ctrllimited`.
25. :commit:`072e963f` Added :ref:`mj_resetCtrl`, setting controls to neutral values: zero, except quaternion inputs
    which reset to the identity quaternion. Called by :ref:`mj_resetData` and the viewers' "Clear All".

Solvers
^^^^^^^
26. :commit:`ea230a95` Flex elasticity (stretch, bending, interpolation stiffness) is now integrated implicitly inside
    the CG constraint solver via an *effective metric*: the mass matrix is augmented with the stiffness Hessian,
    so contact and elastic forces are solved against one consistent metric. This replaces the previous post-hoc CG
    correction, which modified ``qacc`` after the constraint solve. The gate is ``solver="CG"`` with an implicit
    integrator and flex stiffness present; Newton and PGS are unaffected. Bending-only models pay zero per-step
    factorization cost (the factor is precomputed in :ref:`mj_setConst<mj_setConst>`). Inverse dynamics
    (:ref:`mj_inverse<mj_inverse>`) is now discrete-consistent with forward dynamics for gated models.
27. :commit:`c499f7f2` Added Nesterov momentum extrapolation with adaptive gradient restart (O'Donoghue-Candès) to the
    PGS solver, significantly improving convergence. Overall PGS now requires ~2x fewer iterations.
28. :commit:`1e66efd1` Added the Newton decrement -- the quadratic model's predicted cost improvement of the next
    iteration -- as a third early-termination criterion of the :ref:`Newton solver<soAlgorithms>`, alongside cost
    improvement and gradient norm. This reduces iteration counts at no accuracy cost. Proposed by
    :github:user:`adenzler-nvidia` in :doc:`MJWarp <mjwarp/index>` pull request
    `1520 <https://github.com/google-deepmind/mujoco_warp/pull/1520>`__.
29. :commit:`c69ef030` The CG and Newton solvers now terminate with zero iterations when a duality-gap certificate
    proves that the warmstarted solution already satisfies the tolerance. The certificate requires only the existing
    mass-matrix factorization, so quiescent scenes skip Hessian construction, factorization and the line search
    entirely. Newton zero-iteration exits additionally require the gradient criterion, preserving Newton's
    characteristic force-level accuracy. See :ref:`Warmstart<soAlgorithms>` in the Computation chapter for details.


Compiler
^^^^^^^^
30. :commit:`4e1795b9` :ref:`mj_encode` now supports encoding of MJB and TXT files.
31. :commit:`c6c3ec31` The :ref:`attach<body-attach>` element now supports self-attachment (attaching elements of the
    current model to itself) by omitting the :ref:`model<body-attach-model>` attribute. It also supports attaching a
    frame via the new :ref:`frame<body-attach-frame>` attribute, which is mutually exclusive with
    :ref:`body<body-attach-body>`.
32. :commit:`040872fd` Fixed loading of :ref:`.mjz <MJZArchives>` archives in :ref:`simulate<saSimulate>`: the archive
    was unmounted before model compilation, so assets failed to load. Failures in the :ref:`mjz <MJZArchives>` decoder
    now emit a warning with the underlying error instead of the generic "could not decode content" message.
33. :commit:`ebd4abae` The :ref:`mjz <MJZArchives>` decoder now searches for ``model.xml`` and ``<stem>/model.xml`` as a
    fallback if ``<stem>.xml`` and ``<stem>/<stem>.xml`` are not found.
34. :commit:`dc7581ac` Added support for resource writing via :ref:`mju_writeResource` and the ``write`` callback in
    :ref:`mjpResourceProvider`.

.. admonition:: Breaking API changes
   :class: attention

   35. :commit:`d83ef0b6` The return type of :ref:`mj_encode` and the :ref:`mjfEncode` callback changed from ``int`` to
       ``mjtSize`` (64-bit).

.. admonition:: Bug fixes
   :class: admonition

   36. :commit:`f5f9d9ef` Fixed a bug in the mesh compiler where normals were scaled as vectors rather than covectors.

Python bindings
^^^^^^^^^^^^^^^
37. :commit:`a07ae6f8` The bindings now support free threading (`PEP 703 <https://peps.python.org/pep-0703/>`__) for
    Python 3.14.

Documentation
^^^^^^^^^^^^^
38. :commit:`1f1bfa9e` Expanded documentation for :ref:`spec.encode <meSaving>` workflows and added detailed
    documentation for the :ref:`MJZ Archive <MJZArchives>` format (``.mjz`` / ``.zip``).


Version 3.10.0 (June 22, 2026)
------------------------------

General
^^^^^^^
1. :commit:`b935d415` Added :ref:`mju_threadpool`, a new function for creating a thread pool on an ``mjData`` instance.
   When a thread pool is initialized, parts of the simulation pipeline, such as collision detection and constraint
   solving across islands, are parallelized. The thread pool is automatically destroyed when the ``mjData`` is freed.
2. :commit:`58f6d524` Added a unified :ref:`logging API<siError>`:

   - All errors, warnings, and informational messages are now routed through a single :ref:`mjfLogHandler` callback
     receiving a structured :ref:`mjLogMessage`.
   - Users can install a custom handler via :ref:`mju_setLogHandler`,
     configure the default handler's behavior (console/file output, topic filtering) via :ref:`mju_setLogConfig`.
   - Messages can be emitted via :ref:`mju_info` and :ref:`mju_message`.
   - New types: :ref:`mjtLogLevel`, :ref:`mjtLogTopic`, :ref:`mjLogMessage`, :ref:`mjLogConfig`.
   - The legacy callbacks :ref:`mju_user_error` and :ref:`mju_user_warning` are deprecated but remain functional.

3. :commit:`6f8bb5ef` Added :ref:`mjs_numWarnings` and :ref:`mjs_getWarning` for retrieving all warnings accumulated
   during model compilation and attachment. Deprecated :ref:`mjs_isWarning` in favor of ``mjs_numWarnings(s) > 0``.
4. :commit:`410c7316` Added the :ref:`compiler/conflict<compiler-conflict>` attribute for controlling how conflicting
   global attributes are resolved during :ref:`attachment<mjs_attach>`. Possible values are "warning" (default: parent
   values take precedence, warnings emitted on conflicts), "merge" (per-field min/max/error strategy), and "error" (any
   conflict raises an error). See :ref:`Attribute Merging <meAttributeMerging>` for details.

   .. admonition:: Future breaking API changes
      :class: warning

      The current default conflict resolution policy "warn" (ignore the child model) is backward compatible.
      However, the default policy will change to "merge" in a future release.

5. :commit:`cd6db9eb` Improved primal solver convergence under float32. Improvements initially proposed by
   :github:user:`n3b` in :issue:`2313` and :github:user:`adenzler-nvidia` in :doc:`MJWarp <mjwarp/index>` pull request
   `1374 <https://github.com/google-deepmind/mujoco_warp/pull/1374>`__.
6. :commit:`828052e6` The :ref:`CG solver<soAlgorithms>` now uses the Hager-Zhang conjugate direction update instead of
   the Polak-Ribiere-Plus formula. This improves convergence and leads to a significant speedup under float32.
7. :commit:`4c381635` Added :ref:`mjs_makeFlex`, a new C API function equivalent to the :ref:`flexcomp<body-flexcomp>`
   element for programmatically creating flex objects with auto-generated bodies, joints, and equality constraints.
   Exposed as ``body.make_flex()`` in Python.
8. :commit:`7a7dc7cc` Added support for loading 1D flex components from OBJ line segments
9. :commit:`ea2d785e` Significantly improved the quality of coarse convex hulls produced by the
   :ref:`maxhullvert<asset-mesh-maxhullvert>` attribute by invoking Qhull's
   `Q9 <http://www.qhull.org/html/qh-optq.htm#Q9>`__ option.

.. admonition:: Breaking API changes
   :class: attention

   10. :commit:`b935d415` The header file ``mjthread.h`` was removed along with the old engine threading API.
       |br| **Migration:** Use :ref:`mju_threadpool` to set number of worker threads for the engine.
   11. :commit:`96bf8aea` Moved island sparse matrix construction from :ref:`mj_island` (single threaded) into
       :ref:`mj_fwdConstraint` (multi-threaded). The island-specific matrices ``iM, iLD, iefc_J`` were removed from
       the arena and are now allocated on the stack.
   12. :commit:`4548e81e` Following the introduction of the :ref:`diagexact<option-flag-diagexact>` flag, the
       ``mjData`` field ``efc_diagApprox`` was renamed to ``efc_diagA``, as it can now be either the exact or
       approximate diagonal of the :math:`A` ("Delassus") matrix.
   13. :commit:`062b0f1e` The deprecated functions ``mju_{error,warning}_{i,s}`` have been removed.
   14. :commit:`7b9b8806` Changed the signature of :ref:`mj_fullM` from ``mj_fullM(m, dst, M)`` to
       ``mj_fullM(m, d, dst)`` as part of the planned deprecation of ``mjData.qM`` in favor of the CSR-format
       ``mjData.M``.

       **Migration:** For inertia matrix conversion, replace ``mj_fullM(m, dst, d->qM)`` with ``mj_fullM(m, d, dst)``
       or ``mju_sym2dense(dst, d->M, m->nv, m->M_rownnz, m->M_rowadr, m->M_colind)``.

Bug fixes
^^^^^^^^^
15. :commit:`a8eaccd2` Fixed a vulnerability in the System Identification toolbox where loading a trajectory or time
    series called ``np.load`` with ``allow_pickle=True``, allowing arbitrary code execution from a malicious ``.npz``
    file. Signal metadata is now serialized as JSON and loaded with ``allow_pickle=False``.
16. :commit:`b9fb817a` Fixed a bug in the :ref:`mjz <MJZArchives>` :ref:`decoder <mjpDecoder>` where unnormalized paths
    would fail to be read.
17. :commit:`986d73c0` Fixed a bug where the mesh compiler would produce non-unit convex hull polygon normals.

Version 3.9.0 (May 27, 2026)
----------------------------

General
^^^^^^^
1. :commit:`71d1014e` Added ``mjData.efc_Y``, the whitened constraint Jacobian :math:`Y = J M^{-1/2}`, allocated in the
   arena when dual solvers (PGS or NoSlip) are used or when :ref:`diagexact<option-flag-diagexact>` is enabled.
2. :commit:`71d1014e` Added the :ref:`diagexact<option-flag-diagexact>` enable flag, which computes the exact diagonal
   of the constraint-space inertia matrix at the current configuration, replacing the default compile-time
   approximation. This improves solver quality for models with anisotropic inertias or complex kinematic coupling. See
   :ref:`Exact diagonal <soExactDiag>` for details.
3. :commit:`7bfdbad8` The pseudo-random constraint visitation order in the :ref:`PGS solver<soAlgorithms>`, introduced
   in the previous release, now uses a fixed seed. The previous implementation seeded with ``mjData.time``, which
   introduced subtle yet undesirable time dependence.
4. :commit:`f712eed4` Flexes are now allowed to sleep, with the exception of completely passive (constraint-free)
   flexes.
5. :commit:`bdf00966` Added compiler timing diagnostics via the new :ref:`mjtCTimer` enum and the :ref:`mjs_getTimer` C
   API. After :ref:`mj_compile`, per-category timings (total, assets, mesh loading, convex hull, normals, inertia, BVH,
   octree, textures) are available via ``mjs_getTimer(spec)``. The :ref:`compile<saCompile>` sample prints a detailed
   timing breakdown when run without an output file.
6. :commit:`393c1e42` Added :ref:`mjtBool` to represent boolean variables, replacing :ref:`mjtByte` across all boolean
   fields in :ref:`mjModel`, :ref:`mjData`, and public C API function signatures.

.. admonition:: Breaking API changes
   :class: attention

   7. :commit:`a4e49f2d` The semantics of the contact ``margin`` and ``gap`` parameters have been redesigned for
      conceptual clarity and consistency with `Newton <https://github.com/newton-physics/newton>`__. See the new
      :ref:`margin and gap<coMarginGap>` documentation section for details.

      Previously, ``margin`` controlled the *detection threshold* (contacts exist when ``dist < margin``) and ``gap``
      was subtracted from it to produce the *force threshold* (forces generated when ``dist < margin - gap``). This was
      unintuitive: users expected ``margin`` to mean geometric inflation and ``gap`` to mean a spatial gap.

      Under the new semantics, ``margin`` is the geometric inflation of the geom surface and ``gap`` is an additional
      detection buffer beyond the inflated surface:

      - **Detection**: contacts are created when ``dist < margin + gap``.
      - **Force generation**: constraint forces are applied when ``dist < margin``.
      - **Inactive contacts**: contacts with ``margin < dist ≤ margin + gap`` are included in ``mjData.contact`` but
        generate no force (``efc_address = -1``). This is useful for :ref:`adhesion<actuator-adhesion>` actuators and
        custom callbacks.

      With the default values ``margin = 0``, ``gap = 0``, the behavior is unchanged.

      .. image:: images/modeling/margin_gap_light.svg
         :width: 80%
         :align: center
         :class: only-light

      .. image:: images/modeling/margin_gap_dark.svg
         :width: 80%
         :align: center
         :class: only-dark

      |

      **Migration:** Models that use the default ``gap="0"`` (the vast majority) require no changes. For models with
      ``gap > 0``, apply the following transformation to preserve identical behavior:

      .. code-block::

         margin_new = margin_old - gap_old
         gap_new    = gap_old

      For example, a geom with the old attributes ``margin="0.1" gap="0.1"`` should be changed to
      ``margin="0" gap="0.1"``.

      Negative ``margin`` values are now permitted (corresponding to ``gap > margin`` under the old semantics). The
      constraint ``margin + gap >= 0`` should be maintained to ensure valid collision detection.

   8. :commit:`7174d33f` The :ref:`mjfCollision` functions now populate the :ref:`mjPreContact` struct instead of the
      :ref:`mjContact` struct. The :ref:`mjPreContact` only contains the necessary fields needed for the narrowphase
      collision detection.

   9. :commit:`2810edd2` The header file ``mjtnum.h`` was renamed to
      `mjtype.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjtype.h>` and now includes all
      enum type definitions.

   10. :commit:`f6cd0234` The :ref:`tactile<sensor-tactile>` sensor now reports raw depth instead of an estimated
       pressure.

   11. :commit:`072125c4` MJX: Removed the deprecated ``nconmax`` argument from ``mjx.make_data`` and ``mjx.put_data``
       in favor of ``naconmax``.

   12. :commit:`15d27b36` Maybe-breaking: Added `mjassert.h
       <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjassert.h>`__, a new header containing
       compile-time assertions that verify the sizes of MuJoCo's public types for ABI stability. This is a first step
       towards replacing ``int`` with strongly-typed enums in the public API. If these assertions fail on your compiler
       or platform, please report the issue on GitHub.


Version 3.8.1 (May 11, 2026)
----------------------------

General
^^^^^^^
1. :commit:`647af382` Added island support for the :ref:`PGS solver<soAlgorithms>`.
2. :commit:`4ed69b5c` The :ref:`PGS solver<soAlgorithms>` now iterates over constraints in pseudo-random order,
   improving performance by ~20%.
3. :commit:`b9c1877e` Added support for :ref:`elastic2d<flex-elasticity-elastic2d>` for trilinear and quadratic flex
   :ref:`dofs<body-flexcomp-dof>`.
4. :commit:`910b3336` Midpoint integration is now restricted to the ``implicitfast``
   :ref:`integrator<geIntegrators>` and is disabled when fluid forces are active
   (nonzero :ref:`density<option-density>` or :ref:`viscosity<option-viscosity>`).
   Midpoint integration treats external forces as zero-order-hold constants, which causes
   energy gain in the presence of contacts and in fluid media.
5. :commit:`ec50260e` Added :ref:`mjs_getOriginSpec`, returning the spec that originally defined an element, prior to
   attachment. This is in contrast to :ref:`mjs_getSpec` which returns the spec currently owning the element. If the
   element is not the result of an attach operation, the functions are identical.
6. :commit:`767c607f` Added :ref:`mju_sym2dense`, converting a lower-triangular, implicitly symmetric CSR matrix to a
   dense symmetric matrix. The inertia matrix ``mjData.M`` is an example of such a matrix.

.. admonition:: Future breaking API changes
   :class: warning

   7. :commit:`767c607f` The introduction of :ref:`mju_sym2dense` is a step towards the removal of the legacy-format
      ``mjData.qM`` in favor of the CSR-format ``mjData.M``. This removal will involve a future breaking change to
      :ref:`mj_fullM` (which currently accepts a ``qM``-like matrix as an argument). To prevent a future breakage,
      replace ``mj_fullM(m, dst, d->qM)``  with
      |br| ``mju_sym2dense(dst, d->M, m->nv, m->M_rownnz, m->M_rowadr, m->M_colind)``.


Bug fixes
^^^^^^^^^

8. :commit:`3e960ba3` Fixed default for multiccd in :doc:`mjcPhysics <OpenUSD/mjcPhysics>`.

Python
^^^^^^

9. :commit:`d92fe081` Added ``MjSpec.encode`` method, wrapping :ref:`mj_encode`.
10. :commit:`723b8b1e` Added ``mujoco.MjVfs`` Python binding to interact with the Virtual File System directly from
    Python. See :ref:`Virtual File System <PyVFS>` for usage details.

    .. warning::
       The previous way of passing assets via a dictionary mapping asset names to bytes is **deprecated** and will be
       removed in an upcoming release. You cannot specify both the ``assets`` dictionary and the ``vfs`` argument at the
       same time. ``MjVfs`` should be used as a drop-in replacement.


Version 3.8.0 (April 24, 2026)
------------------------------

General
^^^^^^^
1. :commit:`a04cf1b2` Added support for Python 3.14.
2. :commit:`2f2d00da` Added :ref:`multi-cell support<body-flexcomp-cellcount>` for trilinear and quadratic flexes. Note
   that the implicit integrator uses a dense solver for the flex degrees of freedom, which can be slow for multi-cell
   flexes.
3. :commit:`3d45a331` Refactored ``strain`` flex :ref:`equality constraints<flexcomp-edge-equality>` to be instantiated
   per cell instead of per flex object, reducing the number of degrees of freedom per constraint row. The equality can
   be associated with a specific cell with the new attribute :ref:`cell <equality-flexstrain-cell>`
4. :commit:`33259718` Added new :ref:`mj_maxContact<mj_maxContact>` function to get the maximum number of possible
   contacts returned by colliding two geoms.
5. :commit:`4cfebcc3` Added ``mj_containsBufferVFS`` and ``mj_containsFileVFS`` to check for existence of buffers and
   files in VFS.

.. admonition:: Breaking API changes
   :class: attention

   6. :commit:`6cb6e5a9` The :ref:`multiccd<coMultiCCD>` option (multiple contacts returned from the convex collision
      detection pipeline) is now enabled by default. The new implementation (as opposed to the legacy pipeline) has
      little performance overhead and improves stability.

      **Migration:** Disable :ref:`multiccd<option-flag-multiccd>` to recover the previous behavior.

Documentation
^^^^^^^^^^^^^

7. :commit:`2f5e5d3d` Added :ref:`documentation<exDecoder>` for :ref:`mjpDecoder` plugins.

Bug fixes
^^^^^^^^^

8. :commit:`da01bd37` Asset paths in attached child specs are now resolved relative to the model file directory of the
   child spec, rather than the parent spec. This prevents the origin of the parent spec to affect the resolution of
   asset paths in the child spec.

Version 3.7.0 (April 14, 2026)
------------------------------

General
^^^^^^^

1. :commit:`70a7647a` Added the :ref:`dcmotor<actuator-dcmotor>` actuator for modeling DC motors. Supports optional
   electrical dynamics (inductance), cogging torque, thermal resistance variation, and LuGre friction. See the
   `technical note <_static/dcmotor.pdf>`__ for more details.
2. :commit:`510d75f4` Actuators with joint or tendon transmissions can now contribute
   :ref:`damping<actuator-general-damping>` and :ref:`armature<actuator-general-armature>` to their transmission target.
   These are applied during the passive force and inertia computations, respectively, and are scaled by gear\ :sup:`2`
   ("reflected" damping/inertia).

.. youtube:: aKa3ZlEF9_Y
   :align: right
   :width: 35%

3. :commit:`efae9157` Stiffness in :ref:`joints<body-joint-stiffness>` and :ref:`tendons<tendon-spatial-stiffness>` and
   damping in :ref:`joints<body-joint-damping>` and :ref:`tendons<tendon-spatial-damping>` now support nonlinear
   polynomial :ref:`force profiles<gePolynomial>`. New ``mjModel`` arrays (``jnt_stiffnesspoly``,
   ``tendon_stiffnesspoly``, ``dof_dampingpoly``, ``tendon_dampingpoly``) hold higher-order coefficients. The existing
   scalar arrays (``jnt_stiffness``, ``dof_damping``, etc.) continue to hold the linear coefficient and are unchanged.
   The polynomial order is defined by the new constant :ref:`mjNPOLY<glNumericSizes>`. A future breaking C-API change
   may unify the linear and higher-order coefficients into a single array.
4. :commit:`0c337799` Added midpoint integration for standalone free bodies in ``implicit`` and
   ``implicitfast`` :ref:`integrators<geIntegrators>`. This applies the implicit midpoint rule to the rotational
   dynamics of free bodies with no children, conserving kinetic energy to machine precision in the absence of external
   torques. The :ref:`invdiscrete<option-flag-invdiscrete>` flag now also disables midpoint integration, providing an
   opt-out mechanism.
5. :commit:`412cee20` Added the centripetal/Coriolis acceleration term :math:`\dot{J}v` to the constraint solver bias
   for :ref:`connect<equality-connect>` and :ref:`weld<equality-weld>` equality constaints. This significantly improves
   the stability of constrained mechanisms like four-bar linkages. See :ref:`Dual problem<soDual>` for details.

6. :commit:`f5d3ce34` Introduced :ref:`mjpEncoder`, the counterpart to :ref:`mjpDecoder` for encoding of :ref:`mjSpec`
   and :ref:`mjModel` into :ref:`mjResource`.

7. :commit:`f5d3ce34` Added :ref:`mj_encode`, :ref:`mjp_registerEncoder`, :ref:`mjp_defaultEncoder`, and
   :ref:`mjp_findEncoder`.

.. admonition:: Breaking API changes
   :class: attention

   8. :commit:`efae9157` The ``mjs`` layer fields ``stiffness`` and ``damping`` in :ref:`mjsJoint` and :ref:`mjsTendon`
      have been widened from ``mjtNum`` scalars to ``mjtNum[mjNPOLY+1]`` arrays. The first element is the linear
      coefficient (previously the scalar), and subsequent elements are the higher-order :ref:`polynomial<gePolynomial>`
      coefficients.

      **Migration:** Replace assignments like ``joint.stiffness = val`` with ``joint.stiffness[0] = val``.
   9. :commit:`15ca42ff` ``.obj`` and ``.stl`` decoders are now included as source when building MuJoCo with CMake. This
      fixes the behaviour from the previous release where it required downstream code to load these plugins explicitly.

   10. :commit:`4b1667e4` The ``vertcollide`` field in :ref:`mjsFlex` has been removed. It is no longer required since
       :doc:`MuJoCo Warp <mjwarp/index>` supports native flex collisions.

   11. :commit:`f2461f9c` :ref:`mjPLUGIN_LIB_INIT` macro now requires a name argument to avoid initialization function
       name collisions. When building with MSVC, we now use the C runtime initialization section to initialize plugins
       instead of ``DllMain``. See :ref:`plugin registration<exRegistration>` for more details.

   12. :commit:`0e04436d` The :ref:`mjtWarning` enum value ``mjWARN_VGEOMFULL`` is removed. Exhaustion of visual geoms
       is now handled internally by the :ref:`mjvScene`.
   13. :commit:`7ae07d81` URDF parsing no longer hardcodes :ref:`strippath<compiler-strippath>` to "true". The setting
       is now respected and the default is "false". Setting this is attribute is now the responsibility of the user.

       **Migration:** Set :ref:`strippath<compiler-strippath>` to "true" in MJCF or programmatically using

       .. code-block:: python

         spec = mujoco.MjSpec.from_file("path/to/model.urdf")
         spec.compiler.strippath = True


Bug fixes
^^^^^^^^^

14. :commit:`ecc22667` The compiler now correctly accounts for negative scaling when loading user specified mesh data.

Version 3.6.0 (March 10, 2026)
------------------------------

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`9efe41c0` The tendon Jacobian ``ten_J`` is now always sparse. The fields ``ten_J_rownnz``,
      ``ten_J_rowadr``, and ``ten_J_colind`` have been moved from :ref:`mjData` to :ref:`mjModel` and are no longer
      computed at run time by ``mj_tendon`` but at compile time.

2. :commit:`6890e133` Added :ref:`mjs_getCompiler` C API function and a ``compiler`` read-only property to all Python
   spec element types. This allows querying the compiler settings (e.g., ``meshdir``) from any element, with the correct
   originating spec's compiler preserved after attachment.
3. :commit:`713b5524` Added a new ``strain`` :ref:`equality constraint<flexcomp-edge-equality>` type for trilinear and
   quadratic :ref:`dofs<body-flexcomp-dof>`.
4. :commit:`bf74d01d` Flexes now support collisions with SDF geoms.
5. :commit:`5903d482` Improved memory requirements for ``ten_J`` and ``ten_J_colind`` by reducing the upper bound for
   the number of non-zeros ``nJten``.
6. :commit:`37e993f6` Improved memory requirements for ``actuator_moment`` and ``moment_colind`` by reducing the upper
   bound for the number of non-zeros ``nJmom``.

MJX
^^^

7. :commit:`62a32386` Add batch rendering support for MJX-Warp. See the
   :ref:`MJX-Warp batch rendering<MjxWarpBatchRendering>` section for details.

Bug fixes
^^^^^^^^^

8. :commit:`6ec808e2` Fixed a bug where :ref:`mjs_attach` silently dropped spatial tendons with wrapping geometries that
   had no ``sidesite`` attribute (:issue:`3119`, reported by :github:user:`tomstewart89`).

Version 3.5.0 (February 12, 2026)
---------------------------------

Significant new features
^^^^^^^^^^^^^^^^^^^^^^^^

1. :commit:`b64b527e` :doc:`MuJoCo Warp <mjwarp/index>` is now officially released.
2. :commit:`146a5c08` Added a new **System Identification** toolbox (Python), see
   `README <https://github.com/google-deepmind/mujoco/blob/main/python/mujoco/sysid/README.md>`__ for details.
   |br| A Colab notebook demonstrating the toolbox is available here: |sysid_colab|
   |br| Contribution by :github:user:`kevinzakka`, :github:user:`aftersomemath`, :github:user:`jonathanembleyriches`,
   :github:user:`qiayuanl`, :github:user:`spjardim` and :github:user:`gizemozd`.

.. |sysid_colab| image:: https://colab.research.google.com/assets/colab-badge.png
                 :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mujoco/sysid/sysid.ipynb


3. :commit:`6419534b` Actuators and sensors now support arbitrary delays via history buffers, and sensor values can be
   computed at intervals larger than the simulation timestep. Using a delay or interval introduces a new
   ``mjData.history`` variable to the :ref:`Physics state<siPhysicsState>`. See :ref:`Delays<CDelay>` for details.

.. image:: images/changelog/poncho.png
   :width: 45%
   :align: right
   :target: https://github.com/google-deepmind/mujoco/blob/main/model/flex/poncho.xml

4. :commit:`7da271c6` Added new ``flexvert`` equality constraints that enable cloth simulations
   with coarser meshes. This adds a new attribute value ``vert`` to flexcomp edge
   :ref:`equality<flexcomp-edge-equality>` and the new equality type ``flexvert``. Uses the method
   described in `Chen, Kry and Vouga, 2019 <https://arxiv.org/abs/1911.05204>`__.
5. :commit:`0041fdcb` Added implicit integration support for deformable objects (flex) in ``implicit`` and
   ``implicitfast`` :ref:`integrators<geIntegration>`. This method extracts the flex degrees of freedom and solves them
   as a dense block, enabling increased stability for stiff flex objects without reducing the timestep. It is compatible
   with the ``trilinear`` and ``quadratic`` :ref:`dof<body-flexcomp-dof>` types.

.. image:: images/XMLreference/rfcamera.png
   :width: 45%
   :align: right
   :target: https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/sensor/rfcamera.xml

6. :commit:`9d646e65` Rangefinder sensors can now be attached to a camera using the
   :ref:`rangefinder/camera<sensor-rangefinder-camera>` attribute. In this case, the sensor respects the
   :ref:`camera/resolution<body-camera-resolution>` attribute and casts multiple rays, one for each pixel.
7. :commit:`ed15493a` :ref:`Rangefinders<sensor-rangefinder>` can now report various kinds of information besides ray
   distances, including surface normals and intersection points.

.. container:: custom-clear

   .. raw:: html

      <div style="clear: both;"></div>

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   8. :commit:`218226fc` Ray-cast functions now optionally compute the surface normal at the ray intersection. This is a
      breaking change due to the addition of the ``mjtNum normal[3]`` argument. The modified functions are
      :ref:`mj_ray`, :ref:`mj_multiRay`, :ref:`mju_rayGeom`, :ref:`mj_rayFlex`, :ref:`mj_rayHfield` and
      :ref:`mj_rayMesh`.

      **Migration:** In C/C++, pass ``NULL`` to the ``normal`` argument. In Python, in all functions except
      :ref:`mj_multiRay`, it defaults to ``None``, so no action is required.

   9. :commit:`218226fc` ``mju_rayFlex`` has been renamed to :ref:`mj_rayFlex` for consistency with other functions that
      take ``mjModel*`` and ``mjData*`` arguments.

   10. :commit:`b8a4ac5d` The ``mjModel.cam_orthographic`` field has been renamed to ``cam_projection``, with the
       semantic of a new enum type :ref:`mjtProjection`. This will allow for more projection types in the future like
       fisheye cameras. Relatedly, the ``camera/orthographic`` MJCF attribute for cameras has been renamed to
       :ref:`camera/projection<body-camera-projection>` and now accepts the values ``orthographic`` and ``perspective``.

       **Migration:** Replace ``orthographic = "false/true"`` with ``projection="perspective/orthographic"``,
       respectively.

   11. :commit:`cb9a9c15` Removed ``getdir`` from the ``mjpResourceProvider`` struct. All Resource Providers now use the
       same shared implementation.
   12. :commit:`6af0d4c8` When combining the ``margin`` or ``gap`` :ref:`parameters<CContact>` of two geoms to obtain
       the parameters of a contact, the respective values are now **summed** rather than taking the maximum. This allows
       geom margins to be a proper "inflation" of the geom.

13. :commit:`c7f57663` Camera frustum visualization is now triggered by setting
    :ref:`resolution<body-camera-resolution>` to values larger than 1. Relatedly, frustum visualization also works for
    :ref:`orthographic<body-camera-projection>` cameras. See :ref:`rangefinder<sensor-rangefinder>` for details.
14. :commit:`608115ab` Cameras now have an :ref:`output<body-camera-output>` attribute, parsed into the
    ``mjModel.cam_output`` bitfield. Unused by the renderer, it serves as a convenient location to store a camera's
    supported output types.
15. :commit:`37762e3f` Added :ref:`mj_mountVFS` and :ref:`mj_unmountVFS` functions for mounting a custom VFS provider.
    Mounting allows providers to be used to open/read/close resources dynamically at arbitrary paths.
16. :commit:`1d2d0b1c` The optimization whereby sequential :ref:`collision sensors<collision-sensors>` with identical
    attributes shared computation has been removed. This results in a (likely minor) performance regression for models
    which exploited this optimization. To recover the performance, use the :ref:`fromto<sensor-fromto>` and compute the
    other values manually. If ``from = fromto[0:3]`` and ``to = fromto[3:6]`` then ``distance = norm(to-from)`` and
    ``normal = normalize(to-from)``.
17. :commit:`a5dc57c0` :doc:`OpenUSD <OpenUSD/index>`:

    - Parsing has been moved out of experimental into a mjpDecoder plugin. (documentation pending)
    - OpenUSD can now be built with the
      `third_party_deps/openusd <https://github.com/google-deepmind/mujoco/tree/main/cmake/third_party_deps/openusd>`__
      CMake utility project.
    - ``USD_DIR`` is no longer used by the MuJoCo CMake project, instead use ``pxr_DIR`` if you have a pre-built USD
      library.
    - Users no longer have to set ``PXR_PLUGINPATH_NAME`` environment variable, MuJoCo should load USD plugins
      automatically.
18. :commit:`1ff74ba8` Non-breaking ABI changes:

    - The type of the ``sig`` (signature) argument of :ref:`mj_stateSize` and related functions has been changed from
      ``unsigned int`` to ``int``. Before this change, invalid negative arguments passed to this function would result
      in a silent implicit cast; now, negativity will trigger an error.
    - Added a :ref:`depth<mjtRndFlag>` rendering flag.
    - Allocation sizes in :ref:`mjModel` now use 64-bit rather than 32-bit integers to accommodate larger scenes.


MJX
^^^
19. :commit:`1483aefe` Added ``actuator_length``, ``cdof`` and ``cdof_dof`` fields to ``mjx.Data``.
20. :commit:`d07f39b4` Add ``graph_mode`` argument to ``put_model`` to support multiple Warp graph capture modes.

Documentation
^^^^^^^^^^^^^
21. :commit:`b77977a9` General improvements to the :ref:`Programming/Simulation<Simulation>` chapter. Notably, the main
    discussion of :ref:`state<siStateControl>` has been moved there, and the section on :ref:`mjModel changes<siChange>`
    has been expanded.
22. :commit:`c5925e7b` The usability of the :ref:`MJCF schema<CSchema>` is improved with a collapsible dropdown menu
    with links to elements and attributes.
23. :commit:`c54f1fe3` MuJoCo version numbering is now based on Semantic Versioning, see
    `VERSIONING.md <https://github.com/google-deepmind/mujoco/blob/main/VERSIONING.md>`__.


Bug fixes
^^^^^^^^^
24. :commit:`b4668294` Fixed a bug in :ref:`implicit integrator<geIntegrators>` derivatives where actuator velocity
    derivatives were incorrectly computed when the force was clamped by :ref:`forcerange<actuator-general-forcerange>`.
25. :commit:`7f74487a` Fixed a bug in :ref:`implicit integrator<geIntegrators>` derivatives where actuator velocity
    derivatives did not account for the :ref:`actearly<actuator-general-actearly>` flag.
26. :commit:`64a2345c` Multi-threaded mesh processing, enabled by the :ref:`usethread<compiler-usethread>` compiler flag
    (on by default), was in fact disabled by the flag. Fixing this bug speeds up compilation of mesh-heavy models by (up
    to) the number of available cores.
27. :commit:`223ba99e` The ``vertid`` argument of :ref:`mj_rayFlex` and :ref:`mju_raySkin` was marked as nullable but
    was not; it is now nullable.
28. :commit:`c1b3b306` Fixed :ref:`gravcomp<body-gravcomp>` being ignored for bodies with no joints nested inside
    jointed parent bodies (:issue:`3066`, reported by :github:user:`Alex108306`).

Version 3.4.0 (December 5, 2025)
--------------------------------

General
^^^^^^^

.. youtube:: vct493lGQ8Q
   :aspect: 2:1
   :align: right
   :width: 35%

1. :commit:`8734cab3` Introduced a major new feature: :ref:`sleeping islands<Sleeping>`. Preliminary release for early
   testing, see documentation for details.
2. :commit:`3a7aa84e` Added "quadratic" option to :ref:`flexcomp/dof<body-flexcomp-dof>`. This type of fast
   :ref:`deformable<CDeformable>` flex object is similar to the "trilinear" option, but it includes curved deformations.
3. :commit:`b6f25ca6` Raise an error if there are name collisions also during parsing.
4. :commit:`b6f25ca6` Increase Windows stack size to 16MB to enable models with deep nested body hierarchies.
5. :commit:`19e2d0ae` Added a new pipeline component function :ref:`mj_fwdKinematics` that combines all kinematics-like
   sub-components. Relatedly, added a clarifying table at the top of the :ref:`Simulation Pipeline<Pipeline>` chapter.
6. :commit:`2f65e237` Added a new :ref:`mj_extractState` function that allows a subset of a state that was previously
   returned by :ref:`mj_getState` to be extracted without having to be written back into ``mjData`` first.
7. :commit:`888d3a7b` Added a new :ref:`mj_copyState` function that copies state components from one ``mjData`` to
   another.
8. :commit:`ac2cd5df` Tendon paths can now be queried from Python via ``MjsTendon.path``, the returned object
   is iterable and indexing it will give the ``MjsWrap`` at the given index in the path.
9. :commit:`ac2cd5df` ``MjsWrap`` now exposes:

   - ``type -> mujoco.mjtWrap``
   - ``target -> MjsSite|MjsJoint|MjsGeom|None``
   - ``sidesite -> MjsSite|None``
   - ``coef -> real``
   - ``divisor -> real``

10. :commit:`86a77ff8` Non-breaking ABI changes:

    - :ref:`mjtSize` is now defined as ``int64_t`` rather than ``uint64_t`` to avoid future type-promotion bugs.
    - :ref:`mj_sizeModel` now returns an :ref:`mjtSize` rather than an ``int``.

MJX
^^^

11. :commit:`c34ac712` ``warp-lang`` optional dependency is updated to 1.10.0. ``pmap`` now works with MuJoCo Warp from
    MJX.

.. admonition:: Breaking ABI changes
   :class: attention

   12. :commit:`d0f32a7f` ``mjx.Model.tex_data`` is now a numpy ndarray instead of a jax.Array, to avoid vmapping over
       this potentially large array. This may break certain use-cases with Madrona MJX, but we are no longer supporting
       this codepath. We will be migrating users to a Warp-based batch renderer.


Bug fixes
^^^^^^^^^

13. :commit:`88383684` Fixed a bug in the box-box distance computation. Reported by :github:user:`nvtw`.

Version 3.3.7 (October 13, 2025)
--------------------------------

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`77e025ea` The mjSpec C API fields :ref:`meshdir<compiler-meshdir>` and
      :ref:`texturedir<compiler-texturedir>` have been moved to
      `compiler.meshdir <https://github.com/google-deepmind/mujoco/blob/0baac589993220095cf09e153f194f35ca0f0738/include/mujoco/mjspec.h#L154>`__
      and
      `compiler.texturedir <https://github.com/google-deepmind/mujoco/blob/0baac589993220095cf09e153f194f35ca0f0738/include/mujoco/mjspec.h#L155>`__
      respectively. For backwards compatibility, the old fields are still available in the Python API but will be
      removed in a future release.

      **Migration:** Replace ``meshdir`` and ``texturedir`` with ``compiler.meshdir`` and ``compiler.texturedir``.
   2. :commit:`192da874` Remove ``_full_compat`` from ``mjx.put_data`` and ``mjx.put_model``.
   3. :commit:`b56cf98e` ``nconmax`` and ``njmax`` fields in ``mjx.make_data`` now default to ``None`` instead of -1.
      ``nconmax`` will be deprecated in favor of ``naconmax`` in a future release.
   4. :commit:`401bf431` The ``Simulate::InjectNoise`` method needs to set the target of exponential decay to a value
      from ``key_ctrl`` if a valid key is provided as input, else -1.


5. :commit:`fe8384b6` Joint decorators and spatial tendons which have limits defined and whose current value (angle or
   length) exceeds the limit, are recolored by using the :ref:`constraint impedance<soParameters>` :math:`d` to mix the
   existing color with :ref:`visual/rgba/constraint<visual-rgba-constraint>`. For spatial tendons, this visualization
   aid is active only if no :ref:`material<tendon-spatial-material>` is set and :ref:`rgba<tendon-spatial-rgba>` is
   default.
6. :commit:`6320b959` Added :ref:`mju_getXMLDependencies` for computing a list of unique asset dependencies from an MJCF
   file.
7. :commit:`e4704cd2` Added the code sample ``dependencies`` which provides command line utility for printing the result
   of :ref:`mju_getXMLDependencies`.
8. :commit:`bd68f0c6` The minimum C++ standard required to compile MuJoCo is now C++20, this has been the case within
   Google since 2023 but the CMake update was forgotten.

.. admonition:: Breaking ABI changes
   :class: attention

   9. :commit:`431f9657` The attribute ``mjOption.apirate`` was unused and has been removed.
   10. :commit:`b56cf98e` MJX ``nconmax`` and ``njmax`` fields in ``mjx.make_data`` now default to ``None`` instead of
       -1.

MJX
^^^
11. :commit:`6ae9cc80` Fix :issue:`2508`, ``qLD`` shapes mismatched mjModel during ``get_data_into``.
12. :commit:`b56cf98e` Pull in MuJoCo Warp update to ``io.py``, and use ``naconmax`` instead of ``nconmax`` to set the
    maximum number of contacts over all environments.

Bug fixes
^^^^^^^^^
13. :commit:`98682ae2` Fix :issue:`2881`, :at:`fitaabb` was adding an offset to the mesh and applying an incorrect frame
    transformation. Also, unify the meaning of fitting a geom to a mesh AABB: it now means to find the smallest geom
    such that its AABB contains the mesh AABB.

Version 3.3.6 (September 15, 2025)
----------------------------------

General
^^^^^^^
1. :commit:`ec94bb49` Constraint island discovery and construction, previously an experimental feature, is now
   :ref:`documented<soIsland>` and promoted to default; disable it with :ref:`option/flag/island <option-flag-island>`.
   We expect islanding to be a strict improvement over the monolithic constraint solver, please let us know if you
   experience any issues.
2. :commit:`7443e685` :ref:`Contact sensor<sensor-contact>` :at-val:`subtree1/subtree2` specification is now available
   for any body, not just direct children of the world.

.. admonition:: Breaking API changes
   :class: attention

   3. :commit:`6ec5f8b9` The update of ``mjData.qacc_warmstart`` was moved from the end of the solver call
      (:ref:`mj_fwdConstraint`) to the end of :ref:`mj_step`, and is now updated with all other state variables. This
      change makes :ref:`mj_forward` fully idempotent.

      Before this change, calling :ref:`mj_forward` repeatedly would make the constraint solver converge,
      since each subsequent call would start from the previously updated ``qacc_warmstart`` value.
      Indeed, this is precisely what happened in the viewer, which calls :ref:`mj_forward` repeatedly in PAUSE mode.

      **Migration:** If your code depended on this behavior, you can recover it by updating manually after each
      :ref:`mj_forward`: ``qacc_warmstart ← qacc``. The behavior is available in :ref:`simulate<saSimulate>` by
      clicking the "Pause update" toggle (off by default).

      Furthermore, this change has a numerical impact on the output of the :ref:`RK4 <geIntegrators>` integrator.
      Before this change, due to the ``qacc_warmstart`` update occurring after each of the four Runge-Kutta substeps,
      the solver convergence of RK4 was faster, at the cost of unprincipled integration. This change makes the RK4
      integration principled and well-defined. Since this change to RK4 is effectively a bug fix, migration to the
      previous behavior is not provided.

   4. :commit:`b092563c` The ``mjDSBL_PASSIVE`` flag for disabling passive forces was removed and replaced by
      :ref:`mjDSBL_SPRING<mjtDisableBit>` and :ref:`mjDSBL_DAMPER<mjtDisableBit>` with corresponding
      :ref:`mjcf<option-flag-spring>` :ref:`attributes<option-flag-damper>`. Each flag disables only joint and tendon
      springs or dampers, respectively. When both flags are set, **all** passive forces are disabled, including gravity
      compensation, fluid forces, forces computed by the :ref:`mjcb_passive` callback, and forces computed by
      :ref:`plugins <exPlugin>` when passed the :ref:`mjPLUGIN_PASSIVE<mjtPluginCapabilityBit>` capability flag.

      **Migration:** Set both flags to recover the behavior of the previous flag.


.. admonition:: Breaking ABI changes
   :class: attention

   5. :commit:`ed6fa7fe` Removed ``mjMOUSE_SELECT`` flag for :ref:`mjtMouse` as it is no longer in use.

   6. :commit:`ec94bb49` The promotion of islanding to default involved removing the enable flag ``mjENBL_ISLAND`` and
      converting it to a disable flag :ref:`mjDSBL_ISLAND <mjtDisableBit>`.

7. :commit:`b66175eb` Added support for shells with a curved reference configuration. See this `example
   <https://github.com/google-deepmind/mujoco/blob/main/model/flex/basket.xml>`__.
8. :commit:`8acd83f3` Added experimental option for :ref:`passive<flex-contact-passive>` contacts involving flexes.

9. :commit:`1fb1810b` Added support for assigning a default material to a mesh asset using the
   :ref:`mesh/material <asset-mesh-material>` attribute.

MJX
^^^
10. :commit:`1763fa53` Promote ``ten_length`` to the public MJX API. Add Warp support for ``mjx.tendon``.

.. admonition:: Breaking API changes
   :class: attention

   11. :commit:`1763fa53` ``ten_length`` was moved from ``mjx.Data._impl.ten_length`` to a public field
       ``mjx.Data.ten_length``.

Bug fixes
^^^^^^^^^
12. :commit:`ec94bb49` Fixed a latent bug where MjData objects were not serialized correctly by the Python bindings when
    islanding was enabled.


Version 3.3.5 (August 8, 2025)
------------------------------

General
^^^^^^^
1. :commit:`e6c57159` Added the :ref:`insidesite<sensor-insidesite>` sensor, for checking if an object is inside the
   volume of a site. It is useful for triggering events in surrounding environment logic.
2. :commit:`d0e4771c` Added the :ref:`contact<sensor-contact>` sensor, for reporting contact information according to
   user-defined criteria.
   The purpose of the :el:`contact` sensor is to report contact-related information in a fixed-size array. This is
   useful as input to learning-based agents and in environment logic.
3. :commit:`51babec9` Added the :ref:`tactile<sensor-tactile>` sensor, for measuring the penetration depth between two
   objects at given points and the sliding velocities in the tangent frame. The sensor reports tactile data only when
   colliding with SDFs.
4. :commit:`0b11e3e3` Removed the SdfLib plugin and the dependency on `SdfLib <https://github.com/UPC-ViRVIG/SdfLib>`__.
   SDFs are now supported natively in mjModel.
5. :commit:`5e666635` Removed ``oct_depth`` from :ref:`mjvOption` (unused).
6. :commit:`89f47890` Added the functionality to create a builtin meshes, see :ref:`mesh/builtin<asset-mesh-builtin>`.
7. :commit:`ad0dc0de` Inertia computation in MuJoCo C is now performed by a new :ref:`pipeline<piStages>` function
   :ref:`mj_makeM`, which combines the Composite Rigid Body algorithm in :ref:`mj_crb` and additional terms related to
   :ref:`tendon armature<tendon-spatial-armature>`. Code that uses :ref:`mj_crb` to compute the inertia should now use
   :ref:`mj_makeM` instead.

.. admonition:: Breaking API changes
   :class: attention

   8. :commit:`5e666635` Removed the ``mjVIS_FLEXBVH`` enum value, its functionality is now provided by
      :ref:`mjVIS_MESHBVH<mjtVisFlag>`.

Bug fixes
^^^^^^^^^
9. :commit:`6e7aaacb` Fixed a bug that caused object lists in the child to have missing elements after attaching an
   mjSpec. This was caused by adding to the lists only the objects that belong to the tree of the requested body, but
   this causes to skip objects that were attached, since they belong to the tree of the parent.
10. :commit:`3434f5d9` Fixed a bug where the convex hull of a collision mesh was not being computed if the mesh could
    only collide via a :ref:`contact pair<contact-pair>`.

Python
^^^^^^
11. :commit:`2e60d058` On Linux, built distribution packages (wheels) now target the ``manylinux_2_28`` platform tag.
    Previously MuJoCo wheels targeted ``manylinux2014`` based on CentOS 7, which reached end-of-life in June 2024.

MJX
^^^
12. :commit:`47bc16a3` Add Warp as a backend implementation for MJX. The implementation can be specified via
    ``mjx.put_model(m, impl='warp')`` and ``mjx.make_data(m, impl='warp')``. The warp implementation requires
    a CUDA device and ``warp-lang`` to be installed (``pip install mujoco-mjx[warp]``). This feature is available in
    "beta" and some bugs are expected.

Version 3.3.4 (July 8, 2025)
----------------------------

.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`18d5d5d0` The functions ``mjs_detachBody`` and ``mjs_detachDefault`` have been replaced by
      :ref:`mjs_delete`.
   2. :commit:`0488d9f4` The Python functions ``element.delete`` have been replaced by ``spec.delete(element)``.
   3. :commit:`564c51dd` In the mjSpec C API, directly setting an element's name using :ref:`mjs_setString` has been
      replaced with a new function :ref:`mjs_setName` which allows checking for naming collisions at set-time rather
      than compile-time, for earlier catching of errors. Relatedly, the ``name`` attribute has been removed from all mjs
      elements. Known issue: the error is not raised during parsing.
   4. :commit:`47bc16a3` For MJX, the ``mjx.Option`` dataclass now has private and public fields similar to
      ``mjx.Model`` and ``mjx.Data``. Some fields are no longer publicly available due to differences in the
      underlying implementations of this data structure.

General
^^^^^^^
5. :commit:`14dc7c2a` Added support for setting the initial camera in the viewer using
   :ref:`visual/global/cameraid<visual-global-cameraid>`.
6. :commit:`09f7154e` Added support to only sync the state in the Python :ref:`passive viewer<PyViewerPassive>`'s
   ``Sync`` method, this is useful to improve performance. The default behavior is unchanged and copies the entire model
   and data.

Bug fixes
^^^^^^^^^
7. :commit:`4ce62932` Inverse dynamics were not being computed correctly when
   :ref:`tendon armature<tendon-spatial-armature>` was present, now fixed.
8. :commit:`45d4cacc` Fix bug in ``mjx.put_data`` where ``actuator_moment`` was not being copied correctly for the C
   implementation.

Documentation
^^^^^^^^^^^^^
9. :commit:`7548d370` Added missing item documentation and clarified the nature of breaking changes in the 3.3.3
   changelog. See items 3 and 4 below.

Version 3.3.3 (June 10, 2025)
-----------------------------

General
^^^^^^^
1. :commit:`ecb769fc` Refactored island implementation so that island data is memory-contiguous. This speeds up island
   processing in the solver and clears the way for the addition of the Newton and PGS solvers (currently only CG is
   supported).
2. :commit:`7edbdd0a` Removed the :at:`shell` plugin. This is now supported by :ref:`flexcomp<body-flexcomp>` and is
   active depending on the :ref:`elastic2d<flexcomp-elasticity-elastic2d>` attribute (off by default).

.. admonition:: Breaking API changes
   :class: attention

   3. :commit:`74cc904e` Replaced the :ref:`directional<body-light-directional>` (boolean) field for lights with a
      :ref:`type<body-light-type>` field (of type :ref:`mjtLightType<mjtLightType>`) to allow for additional lighting
      types.

      **Migration:** Replace light/directional="false/true" with light/type="spot/directional", respectively.

   4. :commit:`3e9bc79b` Added :ref:`mjtColorSpace` enum and associated :ref:`colorspace<asset-texture-colorspace>`
      attribute that allows to specify the color space of textures (either linear or
      `sRGB <https://en.wikipedia.org/wiki/SRGB>`__). Since this property is now read correctly from PNG files, textures
      files which use sRGB will now be rendered differently.

      **Migration:** Set :ref:`colorspace<asset-texture-colorspace>` to "linear" for all textures that should look like
      they did before this change.

5. :commit:`89e39dc0` Added new sub-component :ref:`mj_makeM` which combines the :ref:`mj_crb` call with additional
   logic to support the introduction in 3.3.1 of :ref:`tendon armature<tendon-spatial-armature>`. In addition to the
   traditional ``mjData.qM``, :ref:`mj_makeM` also computes ``mjData.M``, a CSR representation of the same matrix.
6. :commit:`84ad22a5` Added a new function :ref:`mj_copyBack` to copy real-valued arrays in an mjModel to a compatible
   mjSpec.
7. :commit:`b8768aa1` Removed the limitation of :ref:`fusestatic<compiler-fusestatic>` to models which contain no
   references. The fusestatic flag will now fuse all bodies which are not referenced and ignore bodies which are
   referenced.

Simulate
^^^^^^^^
8. :commit:`ced63018` The struct ``mjv_sceneState`` has been removed. This struct was used for partial synchronization
   of ``mjModel`` and ``mjData`` when the Python viewer is used in passive mode. This functionality is now provided by
   :ref:`mjv_copyModel` and :ref:`mjv_copyData`, which don't copy arrays which are not required for visualization.

.. image:: images/changelog/procedural_terrain_generation.png
   :width: 33%
   :align: right

Python bindings
^^^^^^^^^^^^^^^

9. :commit:`3a4b6e6c` Added examples of procedural terrain generation to the Model Editing tutorial: |mjspec_colab|

MJX
^^^
10. :commit:`caaf7b3a` Added tendon armature.

Version 3.3.2 (April 28, 2025)
------------------------------

MJX
^^^
1. :commit:`51c489fc` Added inverse dynamics.
2. :commit:`f317bd17` Added tendon actuator force sensor.
3. :commit:`421c487d` Fix :issue:`2606` such that ``make_data`` copies over ``mocap_pos`` and ``mocap_quat``
   from ``body_pos`` and ``body_quat``.

Version 3.3.1 (Apr 9, 2025)
----------------------------

.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`f25fc63f` The default value of the flag for toggling ``internal flex contacts``
      was changed from "true" to "false". This feature has proven to be counterintuitive for users.
   2. :commit:`a02a27d4` All of the attach functions (``mjs_attachBody``, ``mjs_attachFrame``, ``mjs_attachToSite``,
      ``mjs_attachFrameToSite``) have been removed and replaced by a single function :ref:`mjs_attach`.

General
^^^^^^^
3. :commit:`d05251af` Added :ref:`tendon armature<tendon-spatial-armature>`: inertia associated with changes in tendon
   length.
4. :commit:`f96f3e1c` Added the :ref:`compiler/saveinertial<compiler-saveinertial>` flag, writing explicit inertial
   clauses for all bodies when saving to XML.
5. :commit:`e8c67ca5` Added ``quat`` orientation attribute to ``composite``.
   Moreover, allow the composite to be the direct child of a frame.
6. :commit:`96dda6ea` Added :ref:`tendon actuator force limits<tendon-spatial-actuatorfrclimited>` and
   :ref:`tendon actuator force sensor<sensor-tendonactuatorfrc>`.

MJX
^^^
7. :commit:`8fc616bf` Added tendon actuator force limits.

Bug fixes
^^^^^^^^^
8. :commit:`de48f417` :ref:`mj_jacDot` was missing a term that accounts for the motion of the point with respect to
   which the Jacobian is computed, now fixed.
9. :commit:`1bf24e9f` Fixed a bug that caused the parent frame of elements in the child worldbody to be incorrectly set
   when attaching an mjSpec to a frame or a site.
10. :commit:`40393f46` Fixed a bug that caused shadow rendering to flicker on platforms (e.g., MacOS) that do not
    support ARB_clip_control. Fixed in collaboration with :github:user:`aftersomemath`.

Python bindings
^^^^^^^^^^^^^^^

.. youtube:: LbANnKMDOHg
   :aspect: 16:7
   :align: right
   :width: 240px

11. :commit:`16e49f27` Added examples of procedural model creation to the Model Editing tutorial: |mjspec_colab|
12. :commit:`ebd30493` Added support for nameless :ref:`mjSpec` objects in the ``bind`` method, see the corresponding
    :ref:`section<PyMJCF>` in the documentation.

.. |mjspec_colab| image:: https://colab.research.google.com/assets/colab-badge.png
                  :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/mjspec.ipynb

Version 3.3.0 (Feb 26, 2025)
----------------------------

Feature promotion
^^^^^^^^^^^^^^^^^
.. youtube:: qJFbx-FR7Bc
   :aspect: 16:7
   :align: right
   :width: 240px

1. :commit:`7cdf1806` Introduced a new kind of **fast deformable body**, activated by setting
   :ref:`flexcomp/dof<body-flexcomp-dof>` to "trilinear". This type of :ref:`deformable<CDeformable>` flex object has
   the same collision geometry as a regular flex, but has far fewer degrees of freedom. Instead of 3 dofs per vertex,
   only the corners of the bounding box are free to move, with the positions of the interior vertices computed with
   trilinear interpolation of the 8 corners, for a total of 24 dofs for the entire flex object (or less, if some of the
   corners are pinned). This limits the types of deformation achievable by the flex, but allows for much faster
   simulation. For example, see the video on the right comparing
   `full <https://github.com/google-deepmind/mujoco/blob/main/model/flex/gripper.xml>`__ and
   `trilinear <https://github.com/google-deepmind/mujoco/blob/main/model/flex/gripper_trilinear.xml>`__ flexes for
   modeling deformable gripper pads.


.. image:: images/computation/ccd_light.gif
      :width: 20%
      :align: right
      :class: only-light

.. image:: images/computation/ccd_dark.gif
   :width: 20%
   :align: right
   :class: only-dark

2. :commit:`ed16f2da` The native convex collision detection pipeline introduced in 3.2.3 and enabled by the
   :ref:`nativeccd<option-flag-nativeccd>` flag, is now the default. See the section on
   :ref:`Convex Collision Detection<coCCD>` for more details.

   **Migration:** If the new pipeline breaks your workflow, set :ref:`nativeccd<option-flag-nativeccd>` to "disable".

General
^^^^^^^
3. :commit:`37d7591c` Add support for custom plots in the MuJoCo viewer by exposing a ``viewport`` property, a
   ``set_figures`` method, and a ``clear_figures`` method.
4. :commit:`7cdf1806` Separate collision and deformation meshes for :ref:`flex<deformable-flex>`. This enables a fixed
   cost for the soft body computations, while preserving the fidelity of high-resolution collisions.
5. :commit:`240a7afd` Added :ref:`potential<sensor-e_potential>` and :ref:`kinetic<sensor-e_kinetic>` energy sensors.
6. :commit:`ac2a324f` Improved shadow rendering in the native renderer.
7. :commit:`b0e9d086` Moved ``introspect`` to ``python/introspect``.

.. admonition:: Breaking API changes
   :class: attention

   8. :commit:`ed16f2da` As mentioned above, the native convex collision detection pipeline is now the default, which
      may break some workflows. In this case, set :ref:`nativeccd<option-flag-nativeccd>` to "disable" to restore the
      old behavior.
   9. :commit:`c2138c3f` Added :ref:`mjs_setDeepCopy` API function. When the deep copy flag is 0, attaching a model will
      not copy it to the parent, so the original references to the child can be used to modify the parent after
      attachment. The default behavior is to perform such a shallow copy. The old behavior of creating a deep copy of
      the child model while attaching can be restored by setting the deep copy flag to 1.
   10. :commit:`89253d95` Changes to inertia inference from meshes:

       Previously, in order to specify that the mass lies on the surface, :ref:`geom/shellinertia<body-geom-shellinertia>`
       could be used for any geom type. Now this attribute is ignored if the geom is a mesh; instead, inertia inference
       for meshes is specified in the asset, using the :ref:`asset/mesh/inertia<asset-mesh-inertia>` attribute.

       Previously, if the volumetric inertia computation failed (for example due to a very flat mesh), the compiler
       would silently fall back to surface inertia computation. Now, the compiler will throw an informative error.
   11. :commit:`0fcd20f0` Removed the composite type ``grid``. Users should instead use :ref:`flexcomp<body-flexcomp>`.
   12. :commit:`c52d1b39` Removed the ``particle`` composite type. It is recommended to use the more generic
       :ref:`replicate<replicate>` instead, see for example `this model
       <https://github.com/google-deepmind/mujoco/blob/main/model/replicate/particle.xml>`__.

MJX
^^^
13. :commit:`f4096bca` Added support for spatial tendons with internal sphere and cylinder wrapping.
14. :commit:`e0664b1b` Fix a bug with box-box collisions :issue:`2356`.

Python bindings
^^^^^^^^^^^^^^^

15. :commit:`7a2ad8fd` Added a pedagogical colab notebook for ``mujoco.rollout``, a Python module for multithreaded
    simulation rollouts. It is available here |rollout_colab|.
    |br| Contribution by :github:user:`aftersomemath`.

.. |rollout_colab| image:: https://colab.research.google.com/assets/colab-badge.png
                   :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/rollout.ipynb

Version 3.2.7 (Jan 14, 2025)
----------------------------

Python bindings
^^^^^^^^^^^^^^^
1. :commit:`a7eb6efd` :ref:`rollout<PyRollout>` now features native multi-threading. If a sequence of ``MjData``
   instances of length ``nthread`` is passed in, ``rollout`` will automatically create a thread pool and parallelize
   the computation. The thread pool can be reused across calls, but then the function cannot be called simultaneously
   from multiple threads. To run multiple threaded rollouts simultaneously, use the new class ``Rollout`` which
   encapsulates the thread pool. Contribution by :github:user:`aftersomemath`.
2. :commit:`40ef08c8` Fix global namespace pollution when using ``mjpython`` (:issue:`2265`).

General
^^^^^^^

.. admonition:: Breaking API changes (minor)
   :class: attention

   3. :commit:`69c9ac07` The field ``mjData.qLDiagSqrtInv`` has been removed. This field is only required for the dual
      solvers. It is now computed as-needed rather than unconditionally. Relatedly, added the corresponding argument to
      :ref:`mj_solveM2`.

4. :commit:`d4ca66a4` Reduced the memory footprint of the PGS solver's :ref:`A matrix<soDual>`. This was the last
   remaining dense-memory allocation in MuJoCo, allowing for a significant reduction of the
   :ref:`dynamic memory allocation heuristic<CSize>`.

Bug fixes
^^^^^^^^^
5. :commit:`0e7d2ef6` Fixed a bug in the box-sphere collider, depth was incorrect for deep penetrations (:issue:`2206`).
6. :commit:`ec322641` Fixed a bug in :ref:`mj_mulM2` and added a test.

Version 3.2.6 (Dec 2, 2024)
---------------------------

General
^^^^^^^
1. :commit:`300450f8` Removed rope and loop from ``composite``. The user is encouraged to instead use
   the :at:`cable` plugin or :ref:`flexcomp<body-flexcomp>`, respectively.

MJX
^^^
2. :commit:`0f381a9e` Added muscle actuators.

Python bindings
^^^^^^^^^^^^^^^
3. :commit:`74dcd51d` Provide prebuilt wheels for Python 3.13.
4. :commit:`3a12db9a` Added ``bind`` method and removed id attribute from :ref:`mjSpec` objects. Using ids is error
   prone in scenarios of repeated attachment and detachment. Python users are encouraged to use names for unique
   identification of model elements.
5. :commit:`943eb6bc` :ref:`rollout<PyRollout>` can now accept sequences of MjModel of length ``nroll``. Also removed
   the ``nroll`` argument because its value can always be inferred.

Bug fixes
^^^^^^^^^
6. :commit:`f9569cda` Fixed :issue:`2212`, type error in ``mjx.get_data``.
7. :commit:`5c23ae11` Fixed bug introduced in 3.2.0 in handling of :ref:`texrepeat<asset-material-texrepeat>` attribute,
   was mistakenly cast from ``float`` to ``int``, (fixed :issue:`2223`).

Version 3.2.5 (Nov 4, 2024)
---------------------------

Feature promotion
^^^^^^^^^^^^^^^^^
1. :commit:`b6037d17` The :doc:`Model Editing<programming/modeledit>` framework afforded by :ref:`mjSpec`, introduced in
   3.2.0 as an in-development feature, is now stable and recommended for general use.
2. :commit:`298ce31e` The native convex collision detection pipeline introduced in 3.2.3 and enabled by the
   :ref:`nativeccd<option-flag-nativeccd>` flag, is not yet the default but is already recommended for general use.
   Please try it when encountering collision-related problems and report any issues you encounter.

General
^^^^^^^

3. :commit:`b598d79b` The global compiler flag ``exactmeshinertia`` has been removed and replaced with the mesh-specific
   :ref:`inertia<asset-mesh-inertia>` attribute.
4. :commit:`7dc8aef8` The not-useful ``convexhull`` compiler option (to disable computation of mesh convex hulls) has
   been removed.
5. :commit:`d8494fef` Removed the deprecated ``mju_rotVecMat``, ``mju_rotVecMatT`` and ``mjv_makeConnector`` functions.
6. :commit:`2b0629d7` Sorting now uses a faster, native sort function (fixes :issue:`1638`).
7. :commit:`61cb552f` The PBR texture layers introduced in 3.2.1 were refactored from separate sub-elements to a single
   :ref:`layer<material-layer>` sub-element.
8. :commit:`831d9881` The composite types box, cylinder, and sphere have been removed. Users should instead use the
   equivalent types available in :ref:`flexcomp<body-flexcomp>`.

MJX
^^^
9. :commit:`680fb3e5` Added ``apply_ft``, ``jac``, and  ``xfrc_accumulate`` as public functions.
10. :commit:`b00a7c67` Added ``TOUCH`` sensor.
11. :commit:`f24de91c` Added support for ``eq_active``. Fixes :issue:`2173`.
12. :commit:`3c21abc0` Added ray intersection with ellipsoid.

Bug fixes
^^^^^^^^^
13. :commit:`864b805a` Fixed several bugs related to connect and weld constraints with site semantics (fixes
    :issue:`2179`, reported by :github:user:`yinfanyi`). The introduction of site specification to connects and welds in
    3.2.3 conditionally changed the semantics of `mjData.eq_obj1id` and `mjData.eq_obj2id`, but these changes were not
    properly propagated in several places leading to incorrect computations of constraint inertia, readings of affected
    force/torque sensors and runtime enabling/disabling of such constraints.
14. :commit:`7620aef5` Fixed a bug in slider-crank :ref:`transmission<geTransmission>`. The bug was introduced in 3.0.0.
15. :commit:`831d9881` Fixed a bug in flex texture coordinates that prevented the correct allocation of textures in
    mjModel.


Documentation
^^^^^^^^^^^^^
16. :commit:`1d58576d` Function headers in the :doc:`API reference <../APIreference/APIfunctions>` now link to their
    source definitions in GitHub.

Version 3.2.4 (Oct 15, 2024)
----------------------------

General
^^^^^^^

.. youtube:: e8lUuykQPGs
   :aspect: 16:7
   :align: right
   :width: 240px

1. :commit:`2dd51873` The Newton solver no longer requires ``nv*nv`` memory allocation, allowing for much larger models.
   See e.g.,
   `100_humanoids.xml  <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/100_humanoids.xml>`__. Two
   quadratic-memory allocations still remain to be fully sparsified: ``mjData.actuator_moment`` and the matrices used by
   the PGS solver.
2. :commit:`4998e7b3` Removed the :at:`solid` and :at:`membrane` plugins and moved the associated computations into the
   engine. See `3D example model <https://github.com/google-deepmind/mujoco/blob/main/model/flex/floppy.xml>`__ and `2D
   example model <https://github.com/google-deepmind/mujoco/blob/main/model/flex/trampoline.xml>`__ for examples of flex
   objects that previously required these plugins.
3. :commit:`6832df30` Replaced the function ``mjs_setActivePlugins`` with :ref:`mjs_activatePlugin`.

MJX
^^^
4. :commit:`1a9d3070` Added ``mocap_pos`` and ``mocap_quat`` in kinematics.
5. :commit:`160ed9bf` Added support for :ref:`spatial tendons <tendon-spatial>` with pulleys and external sphere and
   cylinder wrapping.
6. :commit:`fa22e6d0` Added sphere-cylinder and sphere-ellipsoid collision functions (:issue:`2126`).
7. :commit:`22e4f7fc` Fixed a bug with frictionloss constraints.
8. :commit:`ac91a763` Added ``TENDONPOS`` and ``TENDONVEL`` sensors.
9. :commit:`19459263` Fixed a bug with the computation of tangential contact forces in ``_decode_pyramid``.
10. :commit:`096853e1` Added ``JOINTINPARENT`` actuator transmission type.

Python bindings
^^^^^^^^^^^^^^^
11. :commit:`6881ce24` Removed support for Python 3.8, now that it's
    `deprecated upstream <https://devguide.python.org/versions>`__.

Bug fixes
^^^^^^^^^
12. :commit:`ab3954d8` Fixed a bug where ``actuator_force`` was not set in MJX (:issue:`2068`).
13. :commit:`5838f847` Fixed bug where MJX data tendon fields were incorrect after calling ``mjx.put_data``.
14. :commit:`8d84b5f6` The compiler now returns an error if height fields are used with
    :ref:`collision sensors<collision-sensors>` as they are not yet supported.


Version 3.2.3 (Sep 16, 2024)
----------------------------

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`088079ef` The runtime options ``mpr_tolerance`` and ``mpr_iterations`` were renamed to
      :ref:`ccd_tolerance<option-ccd_tolerance>` and :ref:`ccd_iterations<option-ccd_iterations>`, both in XML and in
      the :ref:`mjOption` struct. This is because the new convex collision detection pipeline (see below) does not use
      the MPR algorithm. The semantics of these options remain identical.
   2. :commit:`d3dfa6f9` The functions ``mjs_findMesh`` and ``mjs_findKeyframe`` were replaced by ``mjs_findElement``,
      which allows to look for any object type.
   3. :commit:`4862b9e7` The experimental use of 2D/3D elasticity plugins with ``composite`` has been
      removed. Users should instead use :ref:`flexcomp<body-flexcomp>`, which provides the correct collision behavior.

4. :commit:`0bffd744` Added the :ref:`nativeccd<option-flag-nativeccd>` flag. When this flag is enabled, general convex
   collision detection is handled with a new native code path, rather than
   `libccd <https://github.com/danfis/libccd>`__. This feature is in early stages of testing, but users who've
   experienced issues related to collision detection are welcome to experiment with it and report any issues.

.. youtube:: kcM_oauk3ZA
   :aspect: 16:7
   :align: right
   :width: 240px

5. :commit:`60a1921b` Added a new way of defining :ref:`connect<equality-connect>` and :ref:`weld<equality-weld>`
   equality constraints, using two sites. The new semantic is useful when the assumption that the constraint is
   satisfied in the base configuration does not hold. In this case the sites will "snap together" at the beginning of
   the simulation. Additionally, changing the site positions (``mjModel.site_pos``) and orientations
   ( ``mjModel.site_quat``) at runtime will correctly modify the constraint definition. This
   `example model <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/equality_site.xml>`__ using
   the new semantic is shown in the video on the right.
6. :commit:`8954a088` Introduced **free joint alignment**, an optimization that applies to bodies with a free joint and
   no child bodies (simple free-floating bodies): automatically aligning the body frame with the inertial frame. This
   feature can be toggled individually using the :ref:`freejoint/align<body-freejoint-align>` attribute or globally
   using the compiler :ref:`alignfree<compiler-alignfree>` attribute. The alignment diagonalizes the related 6x6 inertia
   sub-matrix, leading to both faster and more stable simulation of free bodies.

   While this optimization is a strict improvement, it changes the semantics of the joint's degrees-of-freedom.
   Therefore, ``qpos`` and ``qvel`` values saved in older versions (for example, in :ref:`keyframes<keyframe>`) will
   become invalid. The global compiler attribute currently defaults to "false" due to this potential breakage, but could
   be changed to "true" in a future release. Aligned free joints are recommended for all new models.

7. :commit:`851bb6ee` Added an :ref:`mjSpec` option for creating a texture directly from a buffer.
8. :commit:`466368ef` :ref:`shell (surface) inertia <body-geom-shellinertia>` is now supported by all geom types.
9. :commit:`afd7c73f` When :ref:`attaching<meAttachment>` sub-models, :ref:`keyframes<keyframe>` will now be correctly
   merged into the parent model, but only on the first attachment.
10. :commit:`8b03daa0` Added the :ref:`mjtSameFrame` enum which contains the possible frame alignments of bodies and
    their children. These alignments are used for computation shortcuts in :ref:`mj_kinematics`.
11. :commit:`2d3d5415` Added :ref:`mj_jacDot` for computing time-derivatives of kinematic Jacobians. Fixes :issue:`411`.

MJX
^^^
12. :commit:`a74c184f` Added ``efc_pos`` to ``mjx.Data`` (:issue:`1388`).
13. :commit:`6a12787a` Added position-dependent sensors: ``MAGNETOMETER``, ``CAMPROJECTION``, ``RANGEFINDER``,
    ``JOINTPOS``, ``ACTUATORPOS``, ``BALLQUAT``, ``FRAMEPOS``, ``FRAMEXAXIS``, ``FRAMEYAXIS``, ``FRAMEZAXIS``,
    ``FRAMEQUAT``, ``SUBTREECOM``, ``CLOCK``.
14. :commit:`9805df61` Added velocity-dependent sensors: ``VELOCIMETER``, ``GYRO``, ``JOINTVEL``, ``ACTUATORVEL``,
    ``BALLANGVEL``, ``FRAMELINVEL``, ``FRAMEANGVEL``, ``SUBTREELINVEL``, ``SUBTREEANGMOM``.
15. :commit:`9d732117` Added acceleration/force-dependent sensors: ``ACCELEROMETER``, ``FORCE``, ``TORQUE``,
    ``ACTUATORFRC``, ``JOINTACTFRC``, ``FRAMELINACC``, ``FRAMEANGACC``.
16. :commit:`390bce23` Changed default policy to avoid placing unused (MuJoCo-only) arrays on device.
17. :commit:`390bce23` Added ``device`` parameter to ``mjx.make_data`` to bring it to parity with ``mjx.put_model`` and
    ``mjx.put_data``.
18. :commit:`a68141ee` Added support for :ref:`implicitfast integration<geIntegration>` for all cases except
    :doc:`fluid drag <computation/fluid>`.
19. :commit:`494e166f` Fixed a bug where ``qLDiagInv`` had the wrong size for sparse mass matrices.
20. :commit:`49711fa1` Added support for joint and tendon :ref:`frictionloss <coFriction>`.
21. :commit:`cd8ff440` Added support for :ref:`connect<equality-connect>` equality constraints using two sites.
22. :commit:`e3d3a24b` Added support for :ref:`spatial tendons <tendon-spatial>` with site wrapping.

Bug fixes
^^^^^^^^^
23. :commit:`39896f80` Fixed a performance regression introduced in 3.1.7 in mesh Bounding Volume Hierarchies
    (:issue:`1875`, contribution by :github:user:`michael-ahn`).
24. :commit:`0bcaa856` Fixed a bug wherein, for models that have both muscles and stateless actuators and used one of
    the implicit integrators, wrong derivatives would be computed.
25. :commit:`3e701b21` Fixed a bug in tendon wrapping around spheres. Before this fix, tendons that wrapped around
    spheres with an externally-placed :ref:`sidesite<spatial-geom-sidesite>` could jump inside the sphere instead of
    wrapping around it.
26. :commit:`567793c2` Fixed a bug that caused :at:`meshdir` and :at:`texturedir` to be overwritten during model
    :ref:`attachment<meAttachment>`, preventing model attachment for models with assets in different directories.

Python bindings
^^^^^^^^^^^^^^^
27. :commit:`cfc7dc98` Added support for engine plugins in :ref:`mjSpec` (:issue:`1903`).
28. :commit:`9a27fc14` Better error reporting for issues with the assets dictionary, when loading models.


Version 3.2.2 (Aug 8, 2024)
---------------------------

General
^^^^^^^
1. :commit:`9db9df73` Increase texture and material limit back to 1000. 3.2.0 inadvertently reduced this limit to 100,
   breaking some existing models (:issue:`1877`).

Version 3.2.1 (Aug 5, 2024)
---------------------------

General
^^^^^^^
1. :commit:`e92af73c` Renamed ``mjModel.tex_rgb`` to ``mjModel.tex_data``.
2. :commit:`24a55506` Added a new :ref:`autoreset<option-flag-autoreset>` flag to disable automatic reset when NaNs or
   infinities are detected.
3. :commit:`33e59606` Added sub-elements to the MJCF :ref:`material<asset-material>` element, to allow specification of
   multiple textures for rendering (e.g., ``occlusion, roughness, metallic``). Note that the MuJoCo renderer doesn't
   support these new features, and they are made available for use with external renderers.
4. :commit:`82c27165` Sorting (``mjQUICKSORT``) now calls ``std::sort`` when building with C++ (:issue:`1638`).

MJX
^^^
5. :commit:`dbe18f57` Added more fields to ``mjx.Model`` and ``mjx.Data`` for further compatibility with the
   corresponding MuJoCo structs.
6. :commit:`2d24c588` Added support for :ref:`fixed tendons <tendon-fixed>`.
7. :commit:`2d24c588` Added support for tendon length limits (``mjCNSTR_LIMIT_TENDON`` in :ref:`mjtConstraint`).
8. :commit:`2d24c588` Added support for tendon equality constraints (``mjEQ_TENDON`` in :ref:`mjtEq`).
9. :commit:`2d24c588` Added support for tendon actuator transmission (``mjTRN_TENDON`` in :ref:`mjtTrn`).

Python bindings
^^^^^^^^^^^^^^^
10. :commit:`70ac76bb` Added support for asset dictionary argument in ``mujoco.spec.from_file``,
    ``mujoco.spec.from_string`` and ``mujoco.spec.compile``.


Bug fixes
^^^^^^^^^
11. :commit:`a4bd2bec` Fixed a bug where implicit integrators did not take into account disabled actuators
    (:issue:`1838`).

Version 3.2.0 (Jul 15, 2024)
----------------------------

New features
^^^^^^^^^^^^

1. :commit:`e13ddfa2` Introduced a major new feature: **procedural model creation and editing**, using a new top-level
   data-structure :ref:`mjSpec`. See the :doc:`Model Editing<programming/modeledit>` chapter for details.
   Note that as of this release this feature is still in testing and subject to future breaking changes.
   Fixes :issue:`364`.

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   2. :commit:`e66b9a36` Removed deprecated ``mj_makeEmptyFileVFS`` and ``mj_findFileVFS`` functions. The constants
      ``mjMAXVFS`` and ``mjMAXVFSNAME`` are also removed as they are no longer needed.

      **Migration:** Use :ref:`mj_addBufferVFS` to copy a buffer into a VFS file directly.

   3. :commit:`57e6760e` Calls to :ref:`mj_defaultVFS` may allocate memory inside VFS, and the corresponding
      :ref:`mj_deleteVFS` must be called to deallocate any internal allocated memory.

   4. :commit:`60670485` Deprecated ``mju_rotVecMat`` and ``mju_rotVecMatT`` in favor of :ref:`mju_mulMatVec3` and
      :ref:`mju_mulMatTVec3`. These function names and argument order are more consistent with the rest of the API.
      The older functions have been removed from the Python bindings and will be removed from the C API in the next
      release.
   5. :commit:`393895bb` Removed the ``actuator_actdim`` callback from actuator plugins. They now have the ``actdim``
      attribute, which must be used with actuators that write state to the ``act`` array. This fixed a crash which
      happened when keyframes were used in a model with stateful actuator plugins. The PID plugin will give an error
      when the wrong value of actdim is provided.

6. :commit:`27b9ddda` Added :ref:`attach<body-attach>` meta-element to MJCF, which allows :ref:`attaching<meAttachment>`
   a subtree from a different model to a body in the current model.
7. :commit:`57e6760e` The :ref:`VFS<Virtualfilesystem>` implementation has been rewritten in C++ and is now considerably
   more efficient in speed and memory footprint.

.. youtube:: ZXBTEIDWHhs
   :aspect: 16:7
   :align: right
   :width: 240px

8. :commit:`07fc95ca` Added support for orthographic cameras. This is available for both fixed cameras and the free
   camera, using the ``camera/orthographic`` and :ref:`global/orthographic<visual-global-orthographic>`
   attributes, respectively.
9. :commit:`ace0c8f0` Added :ref:`maxhullvert<asset-mesh-maxhullvert>`, the maximum number of vertices in a mesh's
   convex hull.
10. :commit:`c9bcf837` Added :ref:`mj_setKeyframe` for saving the current state into a model keyframe.
11. :commit:`5ed464b4` Added support for ``ball`` joints in the URDF parser ("spherical" in URDF).
12. :commit:`3f3b39bb` Replaced ``mjUSEDOUBLE`` which was previously hard-coded in
    `mjtnum.h <https://github.com/google-deepmind/mujoco/blob/3577e2cf8bf841475b489aefff52276a39f24d51/include/mjtnum.h>`__
    with the build-time flag ``mjUSESINGLE``. If this symbol is not defined, MuJoCo will use double-precision floating
    point, as usual. If ``mjUSESINGLE`` is defined, MuJoCo will use single-precision floating point. See :ref:`mjtNum`.

    Relatedly, fixed various type errors that prevented building with single-precision.
13. :commit:`22a10fd2` Quaternions in ``mjData.qpos`` and ``mjData.mocap_quat`` are no longer normalized in-place by
    :ref:`mj_kinematics`. Instead they are normalized when they are used. After the first step, quaternions in
    ``mjData.qpos`` will be normalized.
14. :commit:`3d1d1d07` Mesh loading in the compiler, which is usually the slowest part of the loading process, is now
    multi-threaded.

MJX
~~~
15. :commit:`4c3d9461` Added support for :ref:`elliptic friction cones<option-cone>`.
16. :commit:`4c3d9461` Fixed a bug that resulted in less-optimal linesearch solutions for some difficult constraint
    settings.
17. :commit:`4c3d9461` Fixed a bug in the Newton solver that sometimes resulted in less-optimal gradients.


.. youtube:: P83tKA1iz2Y
   :align: right
   :width: 360px

Simulate
^^^^^^^^
18. :commit:`1eb70864` Added improved tutorial video.
19. :commit:`f37f8408` Improved the Brownian noise generator.
20. :commit:`3d1d1d07` Now displaying model load times if they are longer than 0.25 seconds.

Python bindings
^^^^^^^^^^^^^^^
21. :commit:`2188cba4` Fixed a memory leak when using ``copy.deepcopy()`` on a ``mujoco.MjData`` instance
    (:issue:`1572`).

Bug fixes
^^^^^^^^^
22. :commit:`e7301edd` Fix an issue where ``mj_copyData`` (or ``copy.copy()`` in the Python bindings) was not copying
    contact information correctly (:issue:`1710`).
23. :commit:`8e827d0d` Fix an issue with saving to XML that caused frames to be written multiple times (:issue:`1802`).

Version 3.1.6 (Jun 3, 2024)
---------------------------

General
^^^^^^^

1. :commit:`02d01545` Added :ref:`mj_geomDistance` for computing the shortest signed distance between two geoms and
   optionally a segment connecting them. Relatedly, added the 3 sensors: :ref:`distance<sensor-distance>`,
   :ref:`normal<sensor-normal>`, :ref:`fromto<sensor-fromto>`. See the function and sensor documentation for details.
   Fixes :issue:`51`.
2. :commit:`2830a407` Improvements to position actuators:

   - Added :ref:`timeconst<actuator-position-timeconst>` attribute to the :ref:`position actuator<actuator-position>`.
     When set to a positive value, the actuator is made stateful with :at:`filterexact` dynamics.
   - Added :ref:`dampratio<actuator-position-dampratio>` to both :el:`position` and :el:`intvelocity` actuators. An
     alternative to the :at:`kv` attribute, it provides a convenient way to set actuator damping using natural units.
     See attribute documentation for details.

MJX
^^^

3. :commit:`c511d022` Add height-field collision support. Fixes :issue:`1491`.
4. :commit:`c511d022` Add a pre-compiled field ``mesh_convex`` to ``mjx.Model`` so that mesh properties can be vmapped
   over. Fixes :issue:`1655`.
5. :commit:`c511d022` Fix a bug in convex mesh collisions, where erroneous edge contacts were being created even though
   face separating axes were found. Fixes :issue:`1695`.

Bug fixes
^^^^^^^^^

6. :commit:`96844db9` Fixed a bug the could cause collisions to be missed when :ref:`fusestatic<compiler-fusestatic>` is
   enabled, as is often the case for URDF imports. Fixes :issue:`1069`, :issue:`1577`.
7. :commit:`1d181786` Fixed a bug that was causing the visualization of SDF iterations to write outside the size of the
   vector storing them. Fixes :issue:`1539`.

Version 3.1.5 (May 7, 2024)
---------------------------

General
^^^^^^^

.. youtube:: 5k0_wsIRAFc
   :aspect: 16:7
   :align: right
   :width: 240px

1. :commit:`26f23066` Added the :ref:`replicate<replicate>` to MJCF, a :ref:`meta-element<meta-element>` which permits
   to repeat a subtree with incremental translational and rotational offsets.
2. :commit:`ad045968` Enabled an internal cache in the MuJoCo compiler resulting in recompilation speedup. Currently,
   processed textures, hfields, and OBJ meshes are cached. Support for Unity environments is not yet available.
3. :commit:`6481a838` Added ``mjModel.mesh_scale``: the scaling applied to asset vertices, as specified in the
   :ref:`scale<asset-mesh-scale>` attribute.
4. :commit:`caf215e3` Added visual properties which are ignored by the native renderer, but can be used by external
   renderers:

   - :ref:`light/bulbradius<body-light-bulbradius>` attribute and corresponding ``mjModel.light_bulbradius`` field.
   - :ref:`material/metallic<asset-material-metallic>` attribute and corresponding ``mjModel.material_metallic`` field.
   - :ref:`material/roughness<asset-material-roughness>` attribute and corresponding ``mjModel.material_roughness``
     field.
5. :commit:`546a27ca` The type of the ``size`` argument of :ref:`mj_stackAllocNum` and :ref:`mj_stackAllocInt` was
   changed from ``int`` to ``size_t``.
6. :commit:`131b1745` Added support for gmsh format version 2.2 surface meshes in :ref:`flexcomp<body-flexcomp-file>`.

MJX
^^^
.. admonition:: Breaking API changes
   :class: attention

   7. :commit:`718e079c` Removed deprecated ``mjx.device_get_into`` and ``mjx.device_put`` functions as they lack
      critical new functionality.

      **Migration:** Use ``mjx.get_data_into`` instead of ``mjx.device_get_into``, and ``mjx.put_data`` instead of
      ``mjx.device_put``.

8. :commit:`0cd28d24` Added cylinder plane collisions.
9. :commit:`71333938` Added ``efc_type`` to ``mjx.Data`` and ``dim``, ``efc_address`` to ``mjx.Contact``.
10. :commit:`71333938` Added ``geom`` to ``mjx.Contact`` and marked ``geom1``, ``geom2`` deprecated.
11. :commit:`3b64217b` Added ``ne``, ``nf``, ``nl``, ``nefc``, and ``ncon`` to ``mjx.Data`` to match ``mujoco.MjData``.
12. :commit:`a4df9120` Given the above added fields, removed ``mjx.get_params``, ``mjx.ncon``, and
    ``mjx.count_constraints``.
13. :commit:`a4df9120` Changed the way meshes are organized on device to speed up collision detection when a mesh is
    replicated for many geoms.
14. :commit:`a4df9120` Fixed a bug where capsules might be ignored in broadphase colliision checking.
15. :commit:`c2d0c5dd` Added cylinder collisions using SDFs.
16. :commit:`71333938` Added support for all :ref:`condim <coContact>`: 1, 3, 4, 6.
17. :commit:`d15db545` Add support functions for ``id2name`` and ``name2id``, MJX versions of :ref:`mj_id2name` and
    :ref:`mj_name2id`.
18. :commit:`e9709900` Added support for :ref:`gravcomp<body-gravcomp>` and
    :ref:`actuatorgravcomp<body-joint-actuatorgravcomp>`.
19. :commit:`719476c2` Fixed a bug in ``mjx.ray`` for sometimes allowed negative distances for ray-mesh tests.
20. :commit:`24bc1c8b` Added a new
    `differentiable physics tutorial <https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/training_apg.ipynb>`__
    that demonstrates training locomotion policies with analytical gradients automatically derived from the MJX physics
    step. Contribution by :github:user:`Andrew-Luo1`.

Bug fixes
^^^^^^^^^
21. :commit:`0cd28d24` Defaults of lights were not being saved, now fixed.
22. :commit:`4b6c07cd` Prevent overwriting of frame names by body names when saving an XML. Bug introduced in 3.1.4.
23. :commit:`2b497581` Fixed bug in Python binding of :ref:`mj_saveModel`: ``buffer`` argument was documented as
    optional but was actually not optional.
24. :commit:`546a27ca` Fixed bug that prevented memory allocations larger than 2.15 GB. Fixes :issue:`1606`.


Version 3.1.4 (April 10th, 2024)
--------------------------------

General
^^^^^^^
.. admonition:: Breaking API changes
   :class: attention

   1. :commit:`5d26b50f` Removed the ability to natively add noise to sensors. Note that the ``mjModel.sensor_noise``
      field and :ref:`corresponding attribute<CSensor>` are kept and now function as a convenient location for the user
      to save standard-deviation information for their own use. This feature was removed because:

      - There was no mechanism to seed the random noise generator.
      - It was not thread-safe, even if seeding would have been provided, sampling on multiple threads would lead to
        non-reproducible results.
      - This feature was seen as overreach by the engine. Adding noise should be the user's responsibility.
      - We are not aware of anyone who was actually using the feature.

      **Migration:** Add noise to sensor values yourself.

2. :commit:`47ba72ea` Added the :ref:`actuatorgravcomp<body-joint-actuatorgravcomp>` joint attribute. When enabled,
   gravity compensation forces on the joint are treated as applied by actuators. See attribute documentation for more
   details. The example model
   `refsite.xml <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/actuation/refsite.xml>`__,
   which demonstrates Cartesian actuation of an arm, has been updated to use this attribute.
3. :commit:`4f0293c6` Added support for gmsh format 2.2 , tetrahedral mesh, as generated by e.g.
   `fTetwild <https://github.com/wildmeshing/fTetWild>`__.

4. :commit:`5a365603` Added :ref:`mju_euler2Quat` for converting an Euler-angle sequence to quaternion.

MJX
^^^
5. :commit:`22cd96fb` Improved performance of SAT for convex collisions.
6. :commit:`f2e107f7` Fixed bug for sphere/capsule-convex deep penetration.
7. :commit:`02c62c11` Fixed bug where ``mjx.Data`` produced by ``mjx.put_data`` had different treedef than
   ``mjx.make_data``.
8. :commit:`2386353b` Throw an error for margin/gap for convex mesh collisions, since they are not supported.
9. :commit:`2b3f336b` Added ellipsoid plane collisions.
10. :commit:`b4419235` Added support for userdata.
11. :commit:`2b3f336b` Added ellipsoid-ellipsoid and ellipsoid-capsule collisions using signed distance functions
    (SDFs).

Simulate
^^^^^^^^
12. :commit:`bb42ff16` Fixed bug in order of enable flag strings. Before this change, using the simulate UI to toggle
    the :ref:`invdiscrete<option-flag-invdiscrete>` or the (now removed) ``sensornoise`` flags would actually toggle the
    other flag.

Python bindings
^^^^^^^^^^^^^^^

.. youtube:: xHDS0n5DpqM
   :aspect: 16:7
   :align: right
   :width: 240px

13. :commit:`adc4b92c` Added the ``mujoco.minimize`` Python module for nonlinear least-squares, designed for System
    Identification (sysID). The sysID tutorial is work in progress, but a pedagogical colab notebook with examples,
    including Inverse Kinematics, is available here: |ls_colab|
    |br| The video on the right shows example clips from the tutorial.

.. |ls_colab| image:: https://colab.research.google.com/assets/colab-badge.png
              :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/least_squares.ipynb


Version 3.1.3 (March 5th, 2024)
-------------------------------

General
^^^^^^^
1. :commit:`05150546` Added the :at:`inheritrange` attribute to :ref:`position<actuator-position>` and
   :ref:`intvelocity<actuator-intvelocity>` actuators, allowing convenient setting of the actuator's
   :at:`ctrlrange` or :at:`actrange` (respectively), according to the range of the transmission
   target (joint or tendon). See :ref:`position/inheritrange<actuator-position-inheritrange>` for
   details.
2. :commit:`a4a621f6` Deprecated ``mj_makeEmptyFileVFS`` in favor of :ref:`mj_addBufferVFS`. :ref:`mjVFS` now computes
   checksums of its internal file buffers. :ref:`mj_addBufferVFS` allocates an empty buffer with a given name in an
   mjVFS and copies the data buffer into it, combining and replacing the deprecated two-step process of calling
   ``mj_makeEmptyFileVFS`` followed by a direct copy into the given mjVFS internal file buffer.
3. :commit:`6b7d7142` Added :ref:`mj_angmomMat` which computes the ``3 x nv`` angular momentum matrix :math:`H(q)`,
   providing the linear mapping from generalized velocities to subtree angular momentum :math:`h = H \dot q`.
   Contribution by :github:user:`v-r-a`.

MJX
^^^
4. :commit:`4933a2c7` Improved performance of getting and putting device data.

   - Use ``tobytes()`` for numpy array serialization, which is orders of magnitude faster than converting to tuples.
   - Avoid reallocating host ``mjData`` arrays when array shapes are unchanged.
   - Speed up calculation of ``mjx.ncon`` for models with many geoms.
   - Avoid calling ``mjx.ncon`` in ``mjx.get_data_into`` when ``nc`` can be derived from ``mjx.Data``.
5. :commit:`e77c3cb2` Fixed a bug in ``mjx-viewer`` that prevented it from running.  Updated ``mjx-viewer`` to use newer
   ``mjx.get_data_into`` function call.
6. :commit:`47bb4a82` Fixed a bug in ``mjx.euler`` that applied incorrect damping when using dense mass matrices.
7. :commit:`47bb4a82` Fixed a bug in ``mjx.solve`` that was causing slow convergence when using ``mjSOL_NEWTON`` in
   :ref:`mjtSolver`.
8. :commit:`6a346c42` Added support for :ref:`mjOption.impratio<mjOption>` to ``mjx.Model``.
9. :commit:`2067e208` Added support for cameras in ``mjx.Model`` and ``mjx.Data``. Fixes :issue:`1422`.
10. :commit:`419be4c6` Added an implementation of broadphase using `top_k` and bounding spheres.

Python bindings
^^^^^^^^^^^^^^^
11. :commit:`abf6d41b` Fixed incorrect data types in the bindings for the ``geom``, ``vert``, ``elem``, and ``flex``
    array members of the ``mjContact`` struct, and all array members of the ``mjrContext`` struct.

Version 3.1.2 (February 05, 2024)
---------------------------------

General
^^^^^^^
1. :commit:`e0864ab7` Improved the :ref:`discardvisual<compiler-discardvisual>` compiler flag, which now discards all
   visual-only assets. See :ref:`discardvisual<compiler-discardvisual>` for details.
2. :commit:`2feefbc5` Removed the :ref:`timer<mjtTimer>` for midphase colllision detection, it is now folded in with the
   narrowphase timer. This is because timing the two phases separately required fine-grained timers inside the collision
   functions; these functions are so small and fast that the timer itself was incurring a measurable cost.
3. :commit:`fea7c10b` Added the flag :ref:`bvactive<visual-global-bvactive>` to ``visual/global``, allowing users to
   turn off visualisation of active bounding volumes (the red/green boxes in this :ref:`this changelog item<midphase>`).
   For models with very high-resolution meshes, the computation required for this visualization can slow down simulation
   speed. Fixes :issue:`1279`.

   - Added color of :ref:`bounding volumes<visual-rgba-bv>` and :ref:`active bounding volumes<visual-rgba-bvactive>`
     to :ref:`visual/rgba<visual-rgba>`.
4. :commit:`e143b3db` Height-field elevation data can now be specified directly in XML with the
   :ref:`elevation<asset-hfield-elevation>` attribute (and not only with PNG files).
   See `example model <https://github.com/google-deepmind/mujoco/blob/main/test/user/testdata/hfield_xml.xml>`__.

MJX
^^^
5. :commit:`80f50c94` Added :ref:`dyntype<actuator-general-dyntype>` ``filterexact``.
6. :commit:`8ce2c920` Added :at:`site` transmission.
7. :commit:`feb92bf5` Updated MJX colab tutorial with more stable quadruped environment.
8. :commit:`a02fc405` Added ``mjx.ray`` which mirrors :ref:`mj_ray` for planes, spheres, capsules, boxes, and meshes.
9. :commit:`0a7be173` Added ``mjx.is_sparse`` which mirrors :ref:`mj_isSparse` and ``mjx.full_m`` which mirrors
   :ref:`mj_fullM`.
10. :commit:`0a7be173` Added support for specifying sparse or dense mass matrices via :ref:`option-jacobian`.
11. :commit:`508669a9` Raise a not implemented error when nonzero frictionloss is present. Fixes :issue:`1344`.

Python bindings
^^^^^^^^^^^^^^^
12. :commit:`aceb52bd` Improved the implementation of the :ref:`rollout<PyRollout>` module. Note the changes below are
    breaking, dependent code will require modification.

    - Uses :ref:`mjSTATE_FULLPHYSICS<siFullPhysics>` as state spec, enabling divergence detection by inspecting time.
    - Allows user-defined control spec for any combination of :ref:`user input<siInput>` fields as controls.
    - Outputs are no longer squeezed and always have dim=3.
13. :commit:`7bb0ce42` The ``sync`` function for the :ref:`passive viewer<PyViewerPassive>` can now pick up changes to
    rendering flags in ``user_scn``, as requested in :issue:`1190`.

Bug fixes
^^^^^^^^^
14. :commit:`1e2e0b30` Fixed a bug that prevented the use of pins with plugins if flexes are not in the worldbody. Fixes
    :issue:`1270`.
15. :commit:`a14a584f` Fixed a bug in the :ref:`muscle model<CMuscle>` that led to non-zero values outside the lower
    bound of the length range. Fixes :issue:`1342`.


Version 3.1.1 (December 18, 2023)
-----------------------------------

Bug fixes
^^^^^^^^^
1. :commit:`d39ed1d3` Fixed a bug (introduced in 3.1.0) where box-box collisions produced no contacts if one box was
   deeply embedded in the other.
2. :commit:`dc0d0c59` Fixed a bug in :ref:`simulate<saSimulate>` where the "LOADING..." message was not showing
   correctly.
3. :commit:`d39ed1d3` Fixed a crash in the Python :ref:`passive viewer<PyViewerPassive>`, when used with models
   containing Flex objects.
4. :commit:`0915d69c` Fixed a bug in MJX where ``site_xmat`` was ignored in ``get_data`` and ``put_data``
5. :commit:`d39ed1d3` Fixed a bug in MJX where ``efc_address`` was sometimes incorrectly calculated in ``get_data``.


Version 3.1.0 (December 12, 2023)
---------------------------------

General
^^^^^^^
1. :commit:`8ca51b53` Improved convergence of Signed Distance Function (SDF) collisions by using line search and a new
   objective function for the optimization. This allows to decrease the number of initial points needed for finding the
   contacts and is more robust for very small or large geom sizes.
2. :commit:`eb9568a4` Added :ref:`frame<frame>` to MJCF, a :ref:`meta-element<meta-element>` which defines a pure
   coordinate transformation on its direct children, without requiring a :ref:`body<body>`.
3. :commit:`762371c3` Added the :at:`kv` attribute to the :ref:`position<actuator-position>` and
   :ref:`intvelocity<actuator-intvelocity>` actuators, for specifying actuator-applied damping. This can be used to
   implement a PD controller with 0 reference velocity. When using this attribute, it is recommended to use the
   implicitfast or implicit :ref:`integrators<geIntegration>`.

Plugins
^^^^^^^
4. :commit:`f2025c6a` Allow actuator plugins to use activation variables in ``mjData.act`` as their internal state,
   rather than ``mjData.plugin_state``. Actuator plugins can now specify :ref:`callbacks<mjpPlugin>` that compute
   activation variables, and they can be used with built-in :ref:`dyntype<actuator-plugin-dyntype>` actuator dynamics.
5. :commit:`ca046cbf` Added the `pid <https://github.com/deepmind/mujoco/blob/main/plugin/actuator/README.md>`__
   actuator plugin, a configurable PID controller that implements the Integral term, which is not available with native
   MuJoCo actuators.

MJX
^^^
6. :commit:`35c90844` Added ``site_xpos`` and ``site_xmat`` to MJX.
7. :commit:`67fa7c1d` Added ``put_data``, ``put_model``, ``get_data`` to replace ``device_put`` and ``device_get_into``,
   which will be deprecated. These new functions correctly translate fields that are the result of intermediate
   calculations such as ``efc_J``.

Bug fixes
^^^^^^^^^
8. :commit:`cd56a41f` Fix bug in Cartesian actuation with movable refsite, as when using body-centric Cartesian
   actuators on a quadruped. Before this fix such actuators could lead to non-conservation of momentum.
9. :commit:`cd56a41f` Fix bug that prevented using flex with :ref:`simulate<saSimulate>`.
10. :commit:`7d8d4d39` Fix bug that prevented the use of elasticity plugins in combination with pinned flex vertices.
11. :commit:`3c05f9fa` Release Python wheels targeting macOS 10.16 to support x86_64 systems where
    ``SYSTEM_VERSION_COMPAT`` is set. The minimum supported version is still 11.0, but we release these wheels to fix
    compatibility for those users. See :issue:`1213`.
12. :commit:`49ddb7ca` Fixed mass computation of meshes: Use the correct mesh volume instead of approximating it using
    the inertia box.

Version 3.0.1 (November 15, 2023)
---------------------------------

General
^^^^^^^
1. :commit:`a89412bb` Added sub-terms of total passive forces in ``mjData.qfrc_passive`` to :ref:`mjData`:
   ``qfrc_{spring, damper, gravcomp, fluid}``. The sum of these vectors equals ``qfrc_passive``.

.. youtube:: H9qG9Zf2W44
   :aspect: 16:7
   :align: right
   :width: 240px

2. :commit:`893c4042` Added :ref:`actuatorgroupdisable<option-actuatorgroupdisable>` attribute and associated
   :ref:`mjOption.disableactuator<mjOption>` integer bitfield, which can be used to disable sets of actuators at runtime
   according to their :ref:`group<actuator-general-group>`. Fixes :issue:`1092`. See :ref:`CActDisable`.

   - The first 6 actuator groups are toggleable in the :ref:`simulate<saSimulate>` viewer. See `example model
     <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/actuation/actuator_group_disable.xml>`__
     and associated screen-capture on the right.

3. :commit:`7e419276` Increased ``mjMAXUIITEM`` (maximum number of UI elements per section in Simulate) to 200.

MJX
^^^
4. :commit:`3c0a56c1` Added support for Newton solver (``mjSOL_NEWTON`` in :ref:`mjtSolver`). The Newton solver
   significantly speeds up simulation on GPU:

   .. list-table:: Steps-per-second, Conjugate Gradient vs. Newton on A100
      :header-rows: 1
      :align: left

      * - Model
        - CG
        - Newton
        - Speedup
      * - `Humanoid <https://github.com/google-deepmind/mujoco/tree/56006355b29424658b56aedb48a4269bd4361c68/mjx/mujoco/mjx/benchmark/model/humanoid>`__
        - 640,000
        - 1,020,000
        - **1.6 x**
      * - `Barkour v0 <https://github.com/google-deepmind/mujoco/tree/56006355b29424658b56aedb48a4269bd4361c68/mjx/mujoco/mjx/benchmark/model/barkour_v0>`__
        - 1,290,000
        - 1,750,000
        - **1.35 x**
      * - `Shadow Hand <https://github.com/google-deepmind/mujoco/tree/56006355b29424658b56aedb48a4269bd4361c68/mjx/mujoco/mjx/benchmark/model/shadow_hand>`__
        - 215,000
        - 270,000
        - **1.25 x**

   Humanoid is the standard MuJoCo humanoid,
   `Google Barkour <https://blog.research.google/2023/05/barkour-benchmarking-animal-level.html>`__ and the Shadow Hand
   are both available in the :ref:`MuJoCo Menagerie<Menagerie>`.
5. :commit:`70699765` Added support for joint equality constraints (``mjEQ_JOINT`` in :ref:`mjtEq`).
6. :commit:`f6ed57e9` Fixed bug where mixed ``jnt_limited`` joints were not being constrained correctly.
7. :commit:`e4a9f535` Made ``device_put`` type validation more verbose (fixes :issue:`1113`).
8. :commit:`843360b7` Removed empty EFC rows from ``MJX``, for joints with no limits (fixes :issue:`1117`).
9. :commit:`c8146372` Fixed bug in ``scan.body_tree`` that led to incorrect smooth dynamics for some kinematic tree
   layouts.

Python bindings
^^^^^^^^^^^^^^^

10. :commit:`4c24be9e` Fix the macOS ``mjpython`` launcher to work with the Python interpreter from Apple Command Line
    Tools.
11. :commit:`084facc9` Fixed a crash when copying instances of ``mujoco.MjData`` for models that use plugins. Introduced
    a ``model`` attribute to ``MjData`` which is reference to the model that was used to create that ``MjData``
    instance.

Simulate
^^^^^^^^
12. :commit:`f6ed57e9` :ref:`simulate<saSimulate>`: correct handling of "Pause update", "Fullscreen" and "VSync"
    buttons.

Documentation
^^^^^^^^^^^^^

.. youtube:: cE3s_IfO4g4
   :aspect: 16:7
   :align: right
   :width: 240px

13. :commit:`8d5966ee` Added cell to the `tutorial colab <https://github.com/google-deepmind/mujoco#getting-started>`__
    providing an example of procedural camera control:
14. :commit:`2bb8652b` Added documentation for the :ref:`UI` framework.
15. :commit:`dc8bac2f` Fixed typos and supported fields in docs (fixes :issue:`1105` and :issue:`1106`).


Bug fixes
^^^^^^^^^
16. :commit:`86d9c84e` Fixed bug relating to welds modified with :ref:`torquescale<equality-weld-torquescale>`.

Version 3.0.0 (October 18, 2023)
--------------------------------

New features
^^^^^^^^^^^^

1. :commit:`8f9c690c` Added simulation on GPU and TPU via the new :doc:`mjx` (MJX) Python module. Python users can now
   natively run MuJoCo simulations at millions of steps per second on Google TPU or their own accelerator hardware.

   - MJX is designed to work with on-device reinforcement learning algorithms.  This Colab notebook demonstrates using
     MJX along with reinforcement learning to train humanoid and quadruped robots to locomote: |colab|
   - The MJX API is compatible with MuJoCo but is missing some features in this release.  See the outline of
     :ref:`MJX feature parity <MjxFeatureParity>` for more details.

.. |colab| image:: https://colab.research.google.com/assets/colab-badge.png
           :target: https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb

.. youtube:: QewlEqIZi1o
   :aspect: 16:7
   :align: right
   :width: 240px

2. :commit:`fdb04158` Added new signed distance field (SDF) collision primitive. SDFs can take any shape and are not
   constrained to be convex. Collision points are found by minimizing the maximum of the two colliding SDFs via gradient
   descent.

   - Added new SDF plugin for defining implicit geometries. The plugin must define methods computing an SDF and its
     gradient at query points. See the :ref:`documentation<exWriting>` for more details.

.. youtube:: ra2bTiZHGlw
   :aspect: 16:7
   :align: right
   :width: 240px

3. :commit:`139a8ae2` Added new low-level model element called ``flex``, used to define deformable objects. These
   `simplicial complexes <https://en.wikipedia.org/wiki/Simplicial_complex>`__ can be of dimension 1, 2
   or 3, corresponding to stretchable lines, triangles or tetrahedra. Two new MJCF elements are used
   to define flexes. The top-level :ref:`deformable<deformable>` section contains the low-level flex definition.
   The :ref:`flexcomp<body-flexcomp>` element, similar to ``composite`` is a convenience macro for
   creating deformables, and supports the GMSH tetrahedral file format.

   - Added `shell <https://github.com/google-deepmind/mujoco/blob/45fc15b8447b56d3e6f12d3a158e2bc0d4d30dc8/plugin/elasticity/shell.cc>`__ passive force plugin,
     computing bending forces using a constant precomputed Hessian (cotangent operator).

   **Note**: This feature is still under development and subject to change. In particular, deformable object
   functionality is currently available both via :ref:`deformable<CDeformable>` and ``composite``,
   and both are modifiable by the first-party
   `elasticity plugins <https://github.com/google-deepmind/mujoco/tree/45fc15b8447b56d3e6f12d3a158e2bc0d4d30dc8/plugin/elasticity>`__. We expect some of
   this functionality to be unified in the future.

.. youtube:: Vc1tq0fFvQA
   :aspect: 16:7
   :align: right
   :width: 240px

4. :commit:`3e034e38` Added constraint island discovery with :ref:`mj_island`. Constraint islands are disjoint sets of
   constraints and degrees-of-freedom that do not interact. The only solver which currently supports islands is
   :ref:`CG<option-solver>`. Island discovery can be activated using a new :ref:`enable flag<option-flag-island>`.
   If island discovery is enabled, geoms, contacts and tendons will be colored according to the corresponding island,
   see video. Island discovery is currently disabled for models that have deformable objects (see previous item).

5. :commit:`62251869` Added ``mjThreadPool`` and ``mjTask`` which allow for multi-threaded operations within the MuJoCo
   engine pipeline. If engine-internal threading is enabled, the following operations will be multi-threaded:

   - Island constraint resolution, if island discovery is :ref:`enabled<option-flag-island>` and the
     :ref:`CG solver<option-solver>` is selected. The
     `22 humanoids <https://github.com/deepmind/mujoco/blob/main/model/humanoid/22_humanoids.xml>`__ model shows a 3x
     speedup compared to the single threaded simulation.
   - Inertia-related computations and collision detection will happen in parallel.

   Engine-internal threading is a work in progress and currently only available in first-party code via the
   :ref:`testspeed<saTestspeed>` utility, exposed with the ``npoolthread`` flag.

6. :commit:`139a8ae2` Added capability to initialize ``composite`` particles from OBJ files. Fixes
   :issue:`642` and :issue:`674`.

General
^^^^^^^

.. admonition:: Breaking API changes
   :class: attention

   7. :commit:`ba66bd4f` Removed the macros ``mjMARKSTACK`` and ``mjFREESTACK``.

      **Migration:** These macros have been replaced by new functions :ref:`mj_markStack` and
      :ref:`mj_freeStack`. These functions manage the :ref:`mjData stack<siStack>` in a fully encapsulated way (i.e.,
      without introducing a local variable at the call site).

   8. :commit:`9902b735` Renamed ``mj_stackAlloc`` to :ref:`mj_stackAllocNum`. The new function :ref:`mj_stackAllocByte`
      allocates an arbitrary number of bytes and has an additional argument for specifying the alignment of the returned
      pointer.

      **Migration:** The functionality for allocating ``mjtNum`` arrays is now available via :ref:`mj_stackAllocNum`.

   9. :commit:`49290772` Renamed the ``nstack`` field in :ref:`mjModel` and :ref:`mjData` to ``narena``. Changed
      ``narena``, ``pstack``, and ``maxuse_stack`` to count number of bytes rather than number of :ref:`mjtNum` |-| s.

   10. :commit:`86d8b912` Changed :ref:`mjData.solver<mjData>`, the array used to collect solver diagnostic information.
       This array of :ref:`mjSolverStat` structs is now of length ``mjNISLAND * mjNSOLVER``, interpreted as as a matrix.
       Each row of length ``mjNSOLVER`` contains separate solver statistics for each constraint island.
       If the solver does not use islands, only row 0 is filled.

       - The new constant :ref:`mjNISLAND<glNumericSizes>` was set to 20.
       - :ref:`mjNSOLVER<glNumericSizes>` was reduced from 1000 to 200.
       - Added :ref:`mjData.solver_nisland<mjData>`: the number of islands for which the solver ran.
       - Renamed ``mjData.solver_iter`` to ``solver_niter``. Both this member and ``mjData.solver_nnz`` are now integer
         vectors of length ``mjNISLAND``.

   11. :commit:`9cf1f6eb` Removed ``mjOption.collision`` and the associated ``option/collision`` attribute.

       **Migration:**

       - For models which have ``<option collision="all"/>``, delete the attribute.
       - For models which have ``<option collision="dynamic"/>``, delete all :ref:`pair<contact-pair>` elements.
       - For models which have ``<option collision="predefined"/>``, disable all dynamic collisions (determined
         via contype/conaffinity) by first deleting all :ref:`contype<body-geom-contype>` and
         :ref:`conaffinity<body-geom-conaffinity>` attributes in the model and then setting them globally to ``0`` using
         |br| ``<default> <geom contype="0" conaffinity="0"/> </default>``.

   12. :commit:`d3d46e16` Removed the :at:`rope` and :at:`cloth` composite objects.

       **Migration:** Users should use the :at:`cable` and :at:`shell` elasticity plugins.

   13. :commit:`ee78b8f7` Added :ref:`mjData.eq_active<mjData>` user input variable, for enabling/disabling the state of
       equality constraints. Renamed ``mjModel.eq_active`` to :ref:`mjModel.eq_active0<mjModel>`, which now has the
       semantic of "initial value of ``mjData.eq_active``". Fixes :issue:`876`.

       **Migration:** Replace uses of ``mjModel.eq_active`` with ``mjData.eq_active``.

   14. :commit:`d88675a0` Changed the default of :ref:`autolimits<compiler-autolimits>` from "false" to "true". This is
       a minor breaking change. The potential breakage applies to models which have elements with "range" defined and
       "limited" not set. Such models cannot be loaded since version 2.2.2 (July 2022).

15. :commit:`2c3297b3` Added a new :ref:`dyntype<actuator-general-dyntype>`, ``filterexact``, which updates first-order
    filter states with the exact formula rather than with Euler integration.
16. :commit:`2c3297b3` Added an actuator attribute, :ref:`actearly<actuator-general-actearly>`, which uses semi-implicit
    integration for actuator forces: using the next step's actuator state to compute the current actuator forces.
17. :commit:`a276c49c` Renamed ``actuatorforcerange`` and ``actuatorforcelimited``, introduced in the previous version
    to :ref:`actuatorfrcrange<body-joint-actuatorfrcrange>` and
    :ref:`actuatorfrclimited<body-joint-actuatorfrclimited>`, respectively.
18. :commit:`b7cf479a` Added the flag :ref:`eulerdamp<option-flag-eulerdamp>`, which disables implicit integration of
    joint damping in the Euler integrator. See the :ref:`Numerical Integration<geIntegration>` section for more details.
19. :commit:`819b5cb9` Added the flag :ref:`invdiscrete<option-flag-invdiscrete>`, which enables discrete-time inverse
    dynamics for all :ref:`integrators<option-integrator>` other than ``RK4``. See the flag documentation for more
    details.
20. :commit:`1a3215e3` Added :ref:`ls_iterations<option-ls_iterations>` and :ref:`ls_tolerance<option-ls_tolerance>`
    options for adjusting linesearch stopping criteria in CG and Newton solvers. These can be useful for performance
    tuning.
21. :commit:`ccda87aa` Added ``mesh_pos`` and ``mesh_quat`` fields to :ref:`mjModel` to store the normalizing
    transformation applied to mesh assets. Fixes :issue:`409`.
22. :commit:`8064ad59` Added camera :ref:`resolution<body-camera-resolution>` attribute and
    :ref:`camprojection<sensor-camprojection>` sensor. If camera resolution is set to positive values, the camera
    projection sensor will report the location of a target site, projected onto the camera image, in pixel coordinates.
23. :commit:`36d2ffe4` Added :ref:`camera<body-camera>` calibration attributes:

    - The new attributes are :ref:`resolution<body-camera-resolution>`, :ref:`focal<body-camera-focal>`,
      :ref:`focalpixel<body-camera-focalpixel>`, :ref:`principal<body-camera-principal>`,
      :ref:`principalpixel<body-camera-principalpixel>` and :ref:`sensorsize<body-camera-sensorsize>`.
    - Visualize the calibrated frustum using the :ref:`mjVIS_CAMERA<mjtVisFlag>` visualization flag when these
      attributes are specified. See the following
      `example model <https://github.com/deepmind/mujoco/blob/main/test/engine/testdata/vis_visualize/frustum.xml>`__.
    - Note that these attributes only take effect for offline rendering and do not affect interactive visualisation.
24. :commit:`59164702` Implemented reversed Z rendering for better depth precision. An enum :ref:`mjtDepthMap` was added
    with values ``mjDEPTH_ZERONEAR`` and ``mjDEPTH_ZEROFAR``, which can be used to set the new ``readDepthMap``
    attribute in :ref:`mjrContext` to control how the depth returned by :ref:`mjr_readPixels` is mapped from ``znear``
    to ``zfar``. Contribution :pr:`978` by `Levi Burner <https://github.com/aftersomemath>`__.
25. :commit:`fb4cf472` Deleted the code sample ``testxml``. The functionality provided by this utility is implemented in
    the `WriteReadCompare <https://github.com/google-deepmind/mujoco/blob/main/test/xml/xml_native_writer_test.cc>`__
    test.
26. :commit:`a1b6026b` Deleted the code sample ``derivative``. Functionality provided by :ref:`mjd_transitionFD`.

Python bindings
^^^^^^^^^^^^^^^

27. :commit:`631b16e7` Fixed :issue:`870` where calling ``update_scene`` with an invalid camera name used the default
    camera.
28. :commit:`2e15574b` Added ``user_scn`` to the :ref:`passive viewer<PyViewerPassive>` handle, which allows users to
    add custom visualization geoms (:issue:`1023`).
29. :commit:`a1d0cbd6` Added optional boolean keyword arguments ``show_left_ui`` and ``show_right_ui`` to the functions
    ``viewer.launch`` and ``viewer.launch_passive``, which allow users to launch a viewer with UI panels hidden.

Simulate
^^^^^^^^

.. youtube:: YSvWn_poqWs
   :aspect: 16:7
   :align: right
   :width: 240px

30. :commit:`3e12f0d5` Added **state history** mechanism to :ref:`simulate<saSimulate>` and the managed
    :ref:`Python viewer<PyViewerManaged>`. State history can be viewed by scrubbing the History slider and (more
    precisely) with the left and right arrow keys. See screen capture:

31. :commit:`93d1c3c9` The ``LOADING...`` label is now shown correctly. Contribution :pr:`1070` by
    `Levi Burner <https://github.com/aftersomemath>`__.

Documentation
^^^^^^^^^^^^^

.. youtube:: nljr0X79vI0
   :aspect: 16:7
   :align: right
   :width: 240px

32. :commit:`18e4e101` Added :doc:`detailed documentation <computation/fluid>` of fluid force modeling, and an
    illustrative example model showing
    `tumbling cards <https://github.com/google-deepmind/mujoco/blob/main/model/cards/cards.xml>`__ using the
    ellipsoid-based fluid model.

Bug fixes
^^^^^^^^^

33. :commit:`b0077e40` Fixed a bug that was causing :ref:`geom margin<body-geom-margin>` to be ignored during the
    construction of midphase collision trees.

34. :commit:`1bd6c94e` Fixed a bug that was generating incorrect values in ``efc_diagApprox`` for weld equality
    constraints.


Version 2.3.7 (July 20, 2023)
-----------------------------

General
^^^^^^^

1. :commit:`a7021df6` Added primitive collider for sphere-cylinder contacts, previously this pair used the generic
   convex-convex collider.
#. :commit:`51aa375a` Added :ref:`joint-actuatorforcerange<body-joint-actuatorfrcrange>` for clamping total actuator
   force at joints and :ref:`sensor-jointactuatorfrc<sensor-jointactuatorfrc>` for measuring total actuation force
   applied at a joint. The most important use case for joint-level actuator force clamping is to ensure that
   :ref:`Cartesian actuator<actuator-general-refsite>` forces are realizable by individual motors at the joints.
   See :ref:`CForceRange` for details.
#. :commit:`5fcdae77` Added an optional ``content_type`` attribute to hfield, texture, and mesh assets. This attribute
   supports a formatted `Media Type <https://www.iana.org/assignments/media-types/media-types.xhtml>`_ (previously known
   as MIME type) string used to determine the type of the asset file without resorting to pulling the type from the file
   extension.
#. :commit:`5cfbb6ac` Added analytic derivatives for quaternion :ref:`subtraction<mjd_subQuat>` and
   :ref:`integration<mjd_quatIntegrate>` (rotation with an angular velocity). Derivatives are in the 3D tangent space.
#. :commit:`c50c92cc` Added :ref:`mjv_connector` which has identical functionality to ``mjv_makeConnector``, but with
   more convenient "from-to" argument parametrization. ``mjv_makeConnector`` is now deprecated.
#. :commit:`242aea93` Bumped oldest supported MacOS from version 10.12 to 11. MacOS 11 is the oldest version still
   maintained by Apple.

Python bindings
^^^^^^^^^^^^^^^

7. :commit:`0ccfef73` The :ref:`passive viewer<PyViewerPassive>` handle now exposes ``update_hfield``, ``update_mesh``,
   and ``update_texture`` methods to allow users to update renderable assets.
   (Issues :issue:`812`, :issue:`958`, :issue:`965`).
#. :commit:`06b70832` Allow a custom keyboard event callback to be specified in the
   :ref:`passive viewer<PyViewerPassive>` (:issue:`766`).
#. :commit:`f7847ba7` Fix GLFW crash when Python exits while the passive viewer is running (:issue:`790`).

Models
^^^^^^

10. :commit:`51aa375a` Added simple `car <https://github.com/google-deepmind/mujoco/blob/main/model/car/car.xml>`__
    example model.


Version 2.3.6 (June 20, 2023)
-----------------------------

.. note::
   MuJoCo 2.3.6 is the last version to officially support Python 3.7.

.. youtube:: ZppeDArq6AU
   :align: right
   :width: 240px

Models
^^^^^^

1. :commit:`3d71b160` Added
   `3x3x3 cube <https://github.com/google-deepmind/mujoco/blob/main/model/cube/cube_3x3x3.xml>`__ example model. See
   `README <https://github.com/google-deepmind/mujoco/blob/main/model/cube/README.md>`__ for details.

Bug fixes
^^^^^^^^^

2. :commit:`41a70499` Fixed a bug that was causing an incorrect computation of the mesh bounding box and coordinate
   frame if the volume was invalid. In such case, now MuJoCo only accepts a non-watertight geometry if
   :ref:`shellinertia<body-geom-shellinertia>` is equal to ``true``.
#. :commit:`55c7f45d` Fixed the sparse Jacobian multiplication logic that is used to compute derivatives for tendon
   damping and fluid force, which affects the behaviour of the
   :ref:`implicit and implicitfast integrators<geIntegration>`.
#. :commit:`c66d7940` Fixes to :ref:`mj_ray`, in line with geom visualisation conventions:

   - Planes and height-fields respect the ``geom_group`` and ``flg_static`` arguments. Before this change, rays would
     intersect planes and height-fields unconditionally.
   - ``flg_static`` now applies to all static geoms, not just those which are direct children of the world body.

.. youtube:: hqIMTNGaLF4
   :align: right
   :width: 240px

Plugins
^^^^^^^

5. :commit:`b397b312` Added touch-grid sensor plugin. See
   `documentation <https://github.com/google-deepmind/mujoco/blob/main/plugin/sensor/README.md>`__ for details, and
   associated
   `touch_grid.xml <https://github.com/google-deepmind/mujoco/blob/main/model/plugin/sensor/touch_grid.xml>`__ example
   model. The plugin includes `in-scene visualisation <https://youtu.be/0LOJ3WMnqeA>`__.

Simulate
^^^^^^^^

.. youtube:: mXVPbppGk5I
   :aspect: 16:7
   :align: right
   :width: 240px

6. :commit:`d40c3959` Added Visualization tab to simulate UI, corresponding to elements of the :ref:`visual<visual>`
   MJCF element. After modifying values in the GUI, a saved XML will contain the new values. The modifiable members of
   :ref:`mjStatistic` (:ref:`extent<statistic-extent>`, :ref:`meansize<statistic-meansize>` and
   :ref:`center<statistic-center>`) are computed by the compiler and therefore do not have defaults. In order for these
   attributes to appear in the saved XML, a value must be specified in the loaded XML.

.. image:: images/changelog/simulate_text_width.png
   :align: right
   :width: 380px
   :alt: Before / After

7. :commit:`d40c3959` Increased text width for UI elements in the default spacing. [before / after]:

General
^^^^^^^

8. :commit:`f67e3595` Added :ref:`mj_getState` and :ref:`mj_setState` for getting and setting the simulation state as a
   concatenated vector of floating point numbers. See the :ref:`State<geState>`  section for details.
#. :commit:`d82f5ce5` Added :ref:`mjContact.solreffriction<mjContact>`, allowing different :ref:`solref<CSolver>`
   parameters for the normal and frictional axes of contacts when using :ref:`elliptic friction cones<option-cone>`.
   This attribute is required for elastic frictional collisions, see associated
   `example model <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/spin_recoil.xml>`__
   mimicking the spin-bounce recoil behaviour of
   `elastic rubber balls <https://www.youtube.com/watch?v=uFLJcRegIVQ&t=3s>`__. This is an advanced option currently
   only supported by explicit :ref:`contact pairs<contact-pair>`, using the
   :ref:`solreffriction<contact-pair-solreffriction>` attribute.
#. :commit:`c50177d3` Added :ref:`mjd_inverseFD` for finite-differenced inverse-dynamics derivatives.
#. :commit:`49efa9cc` Added functions for operations on banded-then-dense "arrowhead" matrices. Such matrices are common
   when doing direct trajectory optimization. See :ref:`mju_cholFactorBand` documentation for details.
#. :commit:`2ad82d59` Added :ref:`mj_multiRay` function for intersecting multiple rays emanating from a single point.
   This is significantly faster than calling :ref:`mj_ray` multiple times.
#. :commit:`9d1a21f9` Ray-mesh collisions are now up to 10x faster, using a bounding volume hierarchy of mesh faces.
#. :commit:`3d82d2a4` Increased ``mjMAXUIITEM`` (maximum number of UI elements per section in Simulate) to 100.
#. :commit:`67f0f515` Added :ref:`documentation<exProvider>` for resource providers.
#. :commit:`49efa9cc` Changed the formula for :ref:`mju_sigmoid`, a finite-support sigmoid
   :math:`s \colon \mathbf R \rightarrow [0, 1]`. Previously, the smooth part consisted of two stitched quadratics, once
   continuously differentiable. It is now a single quintic, twice continuously differentiable:

   .. math::
      s(x) =
      \begin{cases}
         0,                    &       & x \le 0  \\
         6x^5 - 15x^4 + 10x^3, & 0 \lt & x \lt 1  \\
         1,                    & 1 \le & x \qquad
      \end{cases}

17. :commit:`770b4b36` Added optional :ref:`tausmooth<actuator-muscle-tausmooth>` attribute to muscle actuators. When
    positive, the time-constant :math:`\tau` of muscle activation/deactivation uses :ref:`mju_sigmoid` to transition
    smoothly between the two extremal values given by the `Millard et al. (2013) <https://doi.org/10.1115/1.4023390>`__
    muscle model, within a range of width tausmooth.  See :ref:`Muscle actuators<CMuscle>` for more details.
    Relatedly, :ref:`mju_muscleDynamics` now takes 3 parameters instead of 2, adding the new smoothing-width parameter.
#. :commit:`f887c1e9` Moved public C macro definitions out of mujoco.h into a new public header file called
    `mjmacro.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmacro.h>`__. The new file
    is included by mujoco.h so this change does not break existing user code.
#. :commit:`f887c1e9` Added instrumentation for the
    `Address Sanitizer (ASAN) <https://clang.llvm.org/docs/AddressSanitizer.html>`__ and
    `Memory Sanitizer (MSAN) <https://clang.llvm.org/docs/MemorySanitizer.html>`__ to detect memory bugs when allocating
    from the ``mjData`` stack and arena.
#. :commit:`f887c1e9` Removed ``pstack`` and ``parena`` from the output of ``mj_printData``, since these are
    implementation details of the ``mjData`` allocators that are affected by diagnostic paddings in instrumented builds.
#. :commit:`466c9aca` Removed the ``mj_activate`` and ``mj_deactivate`` functions. These had been kept around for
    compatibility with old user code from when MuJoCo was closed source, but have been no-op functions since open
    sourcing.


Version 2.3.5 (April 25, 2023)
------------------------------

Bug fixes
^^^^^^^^^

1. :commit:`d23c5e78` Fix asset loading bug that prevented OBJ and PNG files from being read from disk when :ref:`mjVFS`
   is used.
#. :commit:`d23c5e78` Fix occasional segmentation faults on macOS when mouse perturbations are applied in the Python
   passive viewer.

Plugins
^^^^^^^

3. :commit:`d23c5e78` The ``visualize`` callback in :ref:`mjpPlugin` now receives an :ref:`mjvOption` as an input
   argument.


Version 2.3.4 (April 20, 2023)
------------------------------

.. note::

   This version is affected by an asset loading bug that prevents OBJ and PNG files from being read from disk when
   ``mjVFS`` is used. Users are advised to skip to version 2.3.5 instead.

General
^^^^^^^

1. :commit:`7cc42ecf` Removed the "global" setting of the :ref:`compiler/coordinate<compiler-coordinate>` attribute.
   This rarely-used setting complicates the compiler logic and is blocking future improvements. In order to convert
   older models which used this option, load and save them in MuJoCo 2.3.3 or older.

.. image:: images/changelog/ellipsoidinertia.gif
   :align: right
   :width: 240px

2. :commit:`7cc42ecf` Added :ref:`visual-global<visual-global>` flag
   :ref:`ellipsoidinertia<visual-global-ellipsoidinertia>` to visualize equivalent body inertias with ellipsoids instead
   of the default boxes.
#. :commit:`5f132af6` Added midphase and broadphase collision statistics to :ref:`mjData`.
#. :commit:`01b84089` Added documentation for :ref:`engine plugins<exPlugin>`.
#. :commit:`2e23594f` Added struct information to the ``introspect`` module.
#. :commit:`fe3dccfd` Added a new extension mechanism called :ref:`resource providers<exProvider>`. This extensible
   mechanism allows MuJoCo to read assets from data sources other than the local OS filesystem or
   the :ref:`Virtual file system<Virtualfilesystem>`.

Python bindings
^^^^^^^^^^^^^^^

7. :commit:`6d63e046` Offscreen rendering on macOS is no longer restricted to the main thread. This is achieved by using
   the low-level Core OpenGL (CGL) API to create the OpenGL context, rather than going via GLFW which relies on Cocoa's
   NSOpenGL. The resulting context is not tied to a Cocoa window, and is therefore not tied to the main thread.
#. :commit:`34226bf5` Fixed a race condition in ``viewer.launch_passive`` and ``viewer.launch_repl``. These functions
   could previously return before an internal call to ``mj_forward``. This allows user code to continue and potentially
   modify physics state concurrently with the internal ``mj_forward``, resulting in e.g.
   `MuJoCo stack overflow error <https://github.com/google-deepmind/mujoco/issues/783>`__
   or `segmentation fault <https://github.com/google-deepmind/mujoco/issues/790>`__.
#. :commit:`0db9a453` The ``viewer.launch_passive`` function now returns a handle which can be used to interact with the
   viewer. The passive viewer now also requires an explicit call to ``sync`` on its handle to pick up any update to the
   physics state. This is to avoid race conditions that can result in visual artifacts. See
   :ref:`documentation<PyViewerPassive>` for details.
#. :commit:`b362cb49` The ``viewer.launch_repl`` function has been removed since its functionality is superseded by
   ``launch_passive``.
#. :commit:`2e23594f` Added a small number of missing struct fields discovered through the new ``introspect`` metadata.

Bug fixes
^^^^^^^^^

12. :commit:`101c647a` Fixed bug in the handling of ellipsoid-based fluid model forces in the new implicitfast
    integrator.
#. :commit:`0db9a453` Removed spurious whole-arena copying in `mj_copyData`, which can considerably
    `slow down <https://github.com/google-deepmind/mujoco/issues/568>`__ the copying operation.
#. :commit:`2d69b158` Make :ref:`shellinertia<body-geom-shellinertia>` ignore ``exactmeshinertia``, which is
    only used for legacy volume computations (`#759 <https://github.com/google-deepmind/mujoco/issues/759>`__).


Version 2.3.3 (March 20, 2023)
------------------------------

General
^^^^^^^

1. :commit:`8c7f6ce5` Improvements to implicit integration:

   - The derivatives of the RNE algorithm are now computed using sparse math, leading to significant speed
     improvements for large models when using the :ref:`implicit integrator<geIntegration>`.
   - A new integrator called ``implicitfast`` was added. It is similar to the existing implicit integrator, but skips the
     derivatives of Coriolis and centripetal forces. See the :ref:`numerical integration<geIntegration>` section for a
     detailed motivation and discussion. The implicitfast integrator is recommended for all new models and will
     become the default integrator in a future version.

   The table below shows the compute cost of the 627-DoF `humanoid100
   <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid100.xml>`__ model using different
   integrators. "implicit (old)" uses dense RNE derivatives, "implicit (new)" is after the sparsification mentioned
   above. Timings were measured on a single core of an AMD 3995WX CPU.

.. csv-table::
   :header: "timing", "Euler", "implicitfast", "implicit (new)", "implicit (old)"
   :widths: auto
   :align: left

   one step (ms),  0.5,   0.53,  0.77,  5.0
   steps/second,   2000,  1900,  1300,  200

.. image:: images/computation/midphase.gif
   :align: right
   :width: 350px

.. _midphase:

2. :commit:`70959c1a` Added a collision mid-phase for pruning geoms in body pairs, see :ref:`documentation<coSelection>`
   for more details. This is based on static AABB bounding volume hierarchy (a BVH binary tree) in the body inertial
   frame. The GIF on the right is cut from `this longer video <https://youtu.be/e0babIM8hBo>`__.
#. :commit:`49c939ea` The ``mjd_transitionFD`` function no longer triggers sensor calculation unless explicitly
   requested.
#. :commit:`c57a588a` Corrected the spelling of the ``inteval`` attribute to ``interval`` in the :ref:`mjLROpt` struct.
#. :commit:`9756ed0d` Mesh texture and normal mappings are now 3-per-triangle rather than 1-per-vertex. Mesh vertices
   are no longer duplicated in order to circumvent this limitation as they previously were.
#. :commit:`50bebcb4` The non-zeros for the sparse constraint Jacobian matrix are now precounted and used for matrix
   memory allocation. For instance, the constraint Jacobian matrix from the `humanoid100
   <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid100.xml>`__ model, which previously
   required ~500,000 ``mjtNum``'s, now only requires ~6000. Very large models can now load and run with the CG solver.
#. :commit:`d3d789cf` Modified :ref:`mju_error` and :ref:`mju_warning` to be variadic functions (support for printf-like
   arguments). The functions ``mju_error_i``, ``mju_error_s``, ``mju_warning_i``, and ``mju_warning_s`` are now
   deprecated.
#. :commit:`056e8492` Implemented a performant ``mju_sqrMatTDSparse`` function that doesn't require dense memory
   allocation.
#. :commit:`8696247f` Added ``mj_stackAllocInt`` to get correct size for allocating ints on mjData stack. Reducing stack
   memory usage by 10% - 15%.


Python bindings
^^^^^^^^^^^^^^^

10. :commit:`e58d53e7` Fixed IPython history corruption when using ``viewer.launch_repl``. The ``launch_repl`` function
    now provides seamless continuation of an IPython interactive shell session, and is no longer considered experimental
    feature.
#. :commit:`7a0def97` Added ``viewer.launch_passive`` which launches the interactive viewer in a passive, non-blocking
    mode. Calls to ``launch_passive`` return immediately, allowing user code to continue execution, with the viewer
    automatically reflecting any changes to the physics state. (Note that this functionality is currently in
    experimental/beta stage, and is not yet described in our :ref:`viewer documentation<PyViewer>`.)
#. :commit:`7a0def97` Added the ``mjpython`` launcher for macOS, which is required for ``viewer.launch_passive`` to
   function there.
#. :commit:`71c5f179` Removed ``efc_`` fields from joint indexers. Since the introduction of arena memory, these fields
    now have dynamic sizes that change between time steps depending on the number of active constraints, breaking strict
    correspondence between joints and ``efc_`` rows.
#. :commit:`e20ab6c9` Added a number of missing fields to the bindings of ``mjVisual`` and ``mjvPerturb`` structs.

Simulate
^^^^^^^^

15. :commit:`2f0fb1e4` Implemented a workaround for `broken VSync <https://github.com/glfw/glfw/issues/2249>`__ on macOS
    so that the frame rate is correctly capped when the Vertical Sync toggle is enabled.

.. image:: images/changelog/contactlabel.png
   :align: right
   :width: 400px

16. :commit:`2607e67f` Added optional labels to contact visualization, indicating which two geoms are contacting (names
    if defined, ids otherwise). This can be useful in cluttered scenes.

|br|


Version 2.3.2 (February 7, 2023)
--------------------------------

General
^^^^^^^

1. :commit:`c741dfce` A more performant mju_transposeSparse has been implemented that doesn't require dense memory
   allocation. For a constraint Jacobian matrix from the
   `humanoid100.xml <https://github.com/google-deepmind/mujoco/blob/main/model/humanoid/humanoid100.xml>`__ model,
   this function is 35% faster.
#. :commit:`f1007df0` The function :ref:`mj_name2id` is now implemented using a hash function instead of a linear search
   for better performance.
#. :commit:`929e09f8` Geom names are now parsed from URDF. Any duplicate names are ignored.
   ``mj_printData`` output now contains contacting geom names.

Bug fixes
^^^^^^^^^

4. :commit:`6ba4d6f0` Fixed a bug that for :at:`shellinertia` equal to ``true`` caused the mesh orientation to be
   overwritten by the principal components of the shell inertia, while the vertex coordinates are rotated using the
   volumetric inertia. Now the volumetric inertia orientation is used also in the shell case.
#. :commit:`19b6c70e` Fixed misalignment bug in mesh-to-primitive fitting when using the bounding box fitting option
   :at:`fitaabb`.

.. image:: images/changelog/meshfit.png
   :align: right
   :width: 300px

6. :commit:`d022cd1a` The ``launch_repl`` functionality in the Python viewer has been fixed.
#. :commit:`06557c14` Set ``time`` correctly in ``mjd_transitionFD``, to support time-dependent user code.
#. :commit:`6b80b010` Fixed sensor data dimension validation when ``user`` type sensors are present.
#. :commit:`093125c2` Fixed incorrect plugin error message when a null ``nsensordata`` callback is encountered during
   model compilation.
#. :commit:`e9869d3b` Correctly end the timer (``TM_END``) ``mj_fwdConstraint`` returns early.
#. :commit:`436fc6e7` Fixed an infinite loop in ``mj_deleteFileVFS``.

Simulate
^^^^^^^^

12. :commit:`bc0184d6` Increased precision of simulate sensor plot y-axis by 1 digit
    (`#719 <https://github.com/google-deepmind/mujoco/issues/719>`_).
#. :commit:`79fffdc0` Body labels are now drawn at the body frame rather than inertial frame, unless inertia is being
   visualised.

Plugins
^^^^^^^

14. :commit:`5c021d04` The ``reset`` callback now receives instance-specific ``plugin_state`` and ``plugin_data`` as
    arguments, rather than the entire ``mjData``. Since ``reset`` is called inside ``mj_resetData`` before any physics
    forwarding call has been made, it is an error to read anything from ``mjData`` at this stage.
#. :commit:`30b4309a` The ``capabilities`` field in ``mjpPlugin`` is renamed ``capabilityflags`` to more clearly
    indicate that this is a bit field.


Version 2.3.1 (December 6, 2022)
--------------------------------

Python bindings
^^^^^^^^^^^^^^^

1. :commit:`0846f38c` The ``simulate`` GUI is now available through the ``mujoco`` Python package as ``mujoco.viewer``.
   See :ref:`documentation<PyViewer>` for details. (Contribution by `Levi Burner <https://github.com/aftersomemath>`__.)
#. :commit:`ef695bb8` The ``Renderer`` class from the MuJoCo tutorial Colab is now available directly in the native
   Python bindings.

General
^^^^^^^

3. :commit:`893942a7` The tendon :at:`springlength` attribute can now take two values. Given two non-decreasing values,
   `springlength` specifies a `deadband  <https://en.wikipedia.org/wiki/Deadband>`__ range for spring stiffness. If the
   tendon length is between the two values, the force is 0. If length is outside this range, the force behaves like a
   regular spring, with the spring resting length corresponding to the nearest :at:`springlength` value. This can be
   used to create tendons whose limits are enforced by springs rather than constraints, which are cheaper and easier to
   analyse. See
   `tendon_springlength.xml <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/tendon_springlength.xml>`__
   example model.

   .. attention::
     This is a minor breaking API change. ``mjModel.tendon_lengthspring`` now has size ``ntendon x 2`` rather than
     ``ntendon x 1``.

   .. youtube:: -PJ6afdETUg
      :align: right
      :height: 150px

#. :commit:`0eb7f871` Removed the requirement that stateless actuators come before stateful actuators.
#. :commit:`f905c7fb` Added :ref:`mju_fill`, :ref:`mju_symmetrize` and :ref:`mju_eye` utility functions.
#. :commit:`23092a11` Added :at:`gravcomp` attribute to :ref:`body<body>`, implementing gravity compensation and
   buoyancy. See `balloons.xml <https://github.com/google-deepmind/mujoco/blob/main/model/balloons/balloons.xml>`__
   example model.
#. :commit:`36b30e45` Renamed the ``cable`` plugin library to ``elasticity``.
#. :commit:`0b45129c` Added :at:`actdim` attribute to :ref:`general actuators<actuator-general>`. Values greater than 1
   are only allowed for dyntype :at-val:`user`, as native activation dynamics are all scalar. Added example test
   implementing 2nd-order activation dynamics to
   `engine_forward_test.cc <https://github.com/google-deepmind/mujoco/blob/main/test/engine/engine_forward_test.cc>`__.
#. :commit:`3b89b0fd` Improved particle ``composite`` type, which now permits a user-specified
   geometry and multiple joints. See the two new examples:
   `particle_free.xml <https://github.com/google-deepmind/mujoco/blob/0fcd20f0da67ede7b16ef7e439a28be67505a037/model/composite/particle_free.xml>`__ and
   `particle_free2d.xml <https://github.com/google-deepmind/mujoco/blob/0fcd20f0da67ede7b16ef7e439a28be67505a037/model/composite/particle_free2d.xml>`__.
#. :commit:`7b0fbc63` Performance improvements for non-AVX configurations:

   - 14% faster ``mj_solveLD`` using `restrict <https://en.wikipedia.org/wiki/Restrict>`__. See `engine_core_smooth_benchmark_test
     <https://github.com/google-deepmind/mujoco/blob/main/test/benchmark/solveLD_benchmark_test.cc>`__.
   - 50% faster ``mju_dotSparse`` using manual loop unroll. See `engine_util_sparse_benchmark_test
     <https://github.com/google-deepmind/mujoco/blob/main/test/benchmark/engine_util_sparse_benchmark_test.cc>`__.
#. :commit:`d0b1a973` Added new :at:`solid` passive force plugin:

   .. youtube:: AGcTGHbbze4
      :align: right
      :height: 150px

   - This is new force field compatible with the ``composite`` particles.
   - Generates a tetrahedral mesh having particles with mass concentrated at vertices.
   - Uses a piecewise-constant strain model equivalent to finite elements but expressed in a coordinate-free
     formulation. This implies that all quantities can be precomputed except edge elongation, as in a mass-spring model.
   - Only suitable for small strains (large displacements but small deformations). Tetrahedra may invert if subject to
     large loads.

#. :commit:`9ecade07` Added API functions ``mj_loadPluginLibrary`` and ``mj_loadAllPluginLibraries``. The first function
   is identical to ``dlopen`` on a POSIX system, and to ``LoadLibraryA`` on Windows. The second function scans a
   specified directory for all dynamic libraries file and loads each library found. Dynamic libraries opened by these
   functions are assumed to register one or more MuJoCo plugins on load.
#. :commit:`0d52feaa` Added an optional ``visualize`` callback to plugins, which is called during ``mjv_updateScene``.
   This callback allows custom plugin visualizations. Enable stress visualization for the Cable plugin as an example.
#. :commit:`dee1d602` Sensors of type :ref:`user<sensor-user>` no longer require :at:`objtype`, :at:`objname` and
   :at:`needstage`. If unspecified, the objtype is now :ref:`mjOBJ_UNKNOWN<mjtObj>`. ``user`` sensors :at:`datatype`
   default is now :at-val:`"real"`, :at:`needstage` default is now :at-val:`"acc"`.
#. :commit:`638c9a69` Added support for capsules in URDF import.
#. :commit:`df25d7d6` On macOS, issue an informative error message when run under
   `Rosetta 2 <https://support.apple.com/en-gb/HT211861>`__ translation on an Apple Silicon machine. Pre-built MuJoCo
   binaries make use of `AVX <https://en.wikipedia.org/wiki/Advanced_Vector_Extensions>`__ instructions on x86-64
   machines, which is not supported by Rosetta 2. (Before this version, users only get a cryptic "Illegal instruction"
   message.)

Bug fixes
^^^^^^^^^

17. :commit:`89185b4a` Fixed bug in ``mj_addFileVFS`` that was causing the file path to be ignored (introduced in
    2.1.4).

Simulate
^^^^^^^^

18. :commit:`2ebc5f09` Renamed the directory in which the ``simulate`` application searches for plugins from ``plugin``
    to ``mujoco_plugin``.
#. :commit:`165e72f7` Mouse force perturbations are now applied at the selection point rather than the body center of
   mass.


Version 2.3.0 (October 18, 2022)
--------------------------------

General
^^^^^^^

1. :commit:`58fd72f5` The ``contact`` array and arrays prefixed with ``efc_`` in ``mjData`` were moved out of the
   ``buffer`` into a new ``arena`` memory space. These arrays are no longer allocated with fixed sizes when ``mjData``
   is created. Instead, the exact memory requirement is determined during each call to :ref:`mj_forward` (specifically,
   in :ref:`mj_collision` and :ref:`mj_makeConstraint`) and the arrays are allocated from the ``arena`` space. The
   ``stack`` now also shares its available memory with ``arena``. This change reduces the memory footprint of ``mjData``
   in models that do not use the PGS solver, and will allow for significant memory reductions in the future.
   See the :ref:`Memory allocation <CSize>` section for details.

   .. youtube:: RHnXD6uO3Mg
      :aspect: 16:7
      :align: right
      :height: 150px

#. :commit:`f151e84a` Added colab notebook tutorial showing how to balance the humanoid on one leg with a Linear
   Quadratic Regulator. The notebook uses MuJoCo's native Python bindings, and includes a draft ``Renderer`` class, for
   easy rendering in Python. |br| Try it yourself:  |LQRopenincolab|

   .. |LQRopenincolab| image:: https://colab.research.google.com/assets/colab-badge.png
                       :target: https://colab.research.google.com/github/deepmind/mujoco/blob/main/python/LQR.ipynb

#. :commit:`04d44e1e` Updates to humanoid model:
   - Added two keyframes (stand-on-one-leg and squat).
   - Increased maximum hip flexion angle.
   - Added hamstring tendons which couple the hip and knee at high hip flexion angles.
   - General cosmetic improvements, including improved use of defaults and better naming scheme.

#. :commit:`89579766` Added :ref:`mju_boxQP` and allocation function :ref:`mju_boxQPmalloc` for solving the
   box-constrained Quadratic Program:

   .. math::

      x^* = \text{argmin} \; \tfrac{1}{2} x^T H x + x^T g \quad \text{s.t.} \quad l \le x \le u

   The algorithm, introduced in `Tassa et al. 2014 <https://doi.org/10.1109/ICRA.2014.6907001>`__,
   converges after 2-5 Cholesky factorisations, independent of problem size.

#. :commit:`f4e7fa97` Added :ref:`mju_mulVecMatVec` to multiply a square matrix :math:`M` with vectors :math:`x` and
   :math:`y` on both sides. The function returns :math:`x^TMy`.

#. :commit:`1e2a9a53` Added new plugin API. Plugins allow developers to extend MuJoCo's capability without modifying
   core engine code. The plugin mechanism is intended to replace the existing callbacks, though these will remain for
   the time being as an option for simple use cases and backward compatibility. The new mechanism manages stateful
   plugins and supports multiple plugins from different sources, allowing MuJoCo extensions to be introduced in a
   modular fashion, rather than as global overrides. Note the new mechanism is currently undocumented except in code, as
   we test it internally. If you are interested in using the plugin mechanism, please get in touch first.

#. :commit:`cce35e18` Added :at:`assetdir` compiler option, which sets the values of both :at:`meshdir` and
   :at:`texturedir`. Values in the latter attributes take precedence over :at:`assetdir`.

#. :commit:`84d16844` Added :at:`realtime` option to :ref:`visual<visual>` for starting a simulation at a slower speed.

#. :commit:`e250ff0d` Added new :at:`cable` composite type:

   - Cable elements are connected with ball joints.
   - The `initial` parameter specifies the joint at the starting boundary: :at:`free`, :at:`ball`, or :at:`none`.
   - The boundary bodies are exposed with the names :at:`B_last` and :at:`B_first`.
   - The vertex initial positions can be specified directly in the XML with the parameter :at:`vertex`.
   - The orientation of the body frame **is** the orientation of the material frame of the curve.

#. :commit:`e250ff0d` Added new :at:`cable` passive force plugin:

   - Twist and bending stiffness can be set separately with the parameters :at:`twist` and :at:`bend`.
   - The stress-free configuration can be set to be the initial one or flat with the flag :at:`flat`.
   - New `cable.xml <https://github.com/google-deepmind/mujoco/blob/e250ff0d/model/plugin/elasticity/cable.xml>`__ example
     showing the formation of plectoneme.
   - New `coil.xml <https://github.com/google-deepmind/mujoco/blob/e250ff0d/model/plugin/elasticity/coil.xml>`__  example
     showing a curved equilibrium configuration.
   - New `belt.xml <https://github.com/google-deepmind/mujoco/blob/e250ff0d/model/plugin/elasticity/belt.xml>`__  example
     showing interaction between twist and anisotropy.
   - Added test using cantilever exact solution.

   +--------------------------+--------------------------+--------------------------+
   | .. youtube:: 25kQP671fJE | .. youtube:: 4DvGe-BodFU | .. youtube:: QcGdpUd5H0c |
   |   :align: center         |   :align: center         |    :align: center        |
   |   :height: 140px         |   :height: 140px         |    :height: 140px        |
   +--------------------------+--------------------------+--------------------------+

Python bindings
^^^^^^^^^^^^^^^
11. :commit:`c13979cd` Added ``id`` and ``name`` properties to
    `named accessor <https://mujoco.readthedocs.io/en/latest/python.html#named-access>`__ objects.
    These provide more Pythonic API access to ``mj_name2id`` and ``mj_id2name`` respectively.

#. :commit:`58fd72f5` The length of ``MjData.contact`` is now ``ncon`` rather than ``nconmax``, allowing it to be
   straightforwardly used as an iterator without needing to check ``ncon``.

#. :commit:`ec6ea6a6` Fix a memory leak when a Python callable is installed as callback
   (`#527 <https://github.com/google-deepmind/mujoco/issues/527>`__).


Version 2.2.2 (September 7, 2022)
---------------------------------

General
^^^^^^^

.. youtube:: BcHZ5BFeTmU
   :aspect: 16:7
   :align: right
   :height: 150px

1. :commit:`3d77eb1e` Added :ref:`adhesion actuators<actuator-adhesion>` mimicking vacuum grippers and adhesive
   biomechanical appendages.
#. :commit:`3d77eb1e` Added related
   `example model <https://github.com/google-deepmind/mujoco/tree/main/model/adhesion>`__ and video:
#. :commit:`fcf41317` Added :ref:`mj_jacSubtreeCom` for computing the translational Jacobian of the center-of-mass of a
   subtree.
#. :commit:`d26501c0` Added :at:`torquescale` and :at:`anchor` attributes to :el:`weld` constraints. :at:`torquescale`
   sets the torque-to-force ratio exerted by the constraint, :at:`anchor` sets the point at which the weld wrench is
   applied. See :ref:`weld <equality-weld>` for more details.
#. :commit:`d26501c0` Increased ``mjNEQDATA``, the row length of equality constraint parameters in ``mjModel.eq_data``,
   from 7 to 11.
#. :commit:`d26501c0` Added visualisation of anchor points for both :el:`connect` and :el:`weld` constraints (activated
   by the 'N' key in ``simulate``).
#. :commit:`8ca5887c` Added
   `weld.xml <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/weld.xml>`__ showing different
   uses of new weld attributes.

   .. youtube:: s-0JHanqV1A
      :aspect: 16:7
      :align: right
      :height: 150px

#. :commit:`46da1285` Cartesian 6D end-effector control is now possible by adding a reference site to actuators with
   :at:`site` transmission. See description of new :at:`refsite` attribute in the :ref:`actuator<actuator-general>`
   documentation and
   `refsite.xml <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/actuation/refsite.xml>`__
   example model.

#. :commit:`a693a2d9` Added :at:`autolimits` compiler option. If ``true``, joint and tendon :at:`limited` attributes and
   actuator :at:`ctrllimited`, :at:`forcelimited` and :at:`actlimited` attributes will automatically be set to ``true``
   if the corresponding range *is defined* and ``false`` otherwise.

   If ``autolimits="false"`` (the default) models where a :at:`range` attribute is specified without the :at:`limited`
   attribute will fail to compile. A future release will change the default of :at:`autolimits` to ``true``, and this
   compilation error allows users to catch this future change of behavior.

   .. attention::
     This is a breaking change. In models where a range was defined but :at:`limited` was unspecified, explicitly set
     limited to ``false`` or remove the range to maintain the current behavior of your model.

#. :commit:`8ca5887c` Added moment of inertia computation for all well-formed meshes. This option is activated by
   setting the compiler flag :at:`exactmeshinertia` to ``true`` (defaults to ``false``). This default may change in the
   future.
#. :commit:`5c5449bf` Added parameter :at:`shellinertia` to :at:`geom`, for locating the inferred inertia on the
   boundary (shell). Currently only meshes are supported.
#. :commit:`833dc740` For meshes from which volumetric inertia is inferred, raise error if the orientation of mesh faces
   is not consistent. If this occurs, fix the mesh in e.g., MeshLab or Blender.

   .. youtube:: I2q7D0Vda-A
      :align: right
      :height: 150px

#. :commit:`ae0ac86e` Added catenary visualisation for hanging tendons. The model seen in the video can be found
   `here <https://github.com/google-deepmind/mujoco/blob/main/test/engine/testdata/catenary.xml>`__.
#. :commit:`b966a378` Added ``azimuth`` and ``elevation`` attributes to :ref:`visual/global<visual-global>`, defining
   the initial orientation of the free camera at model load time.
#. :commit:`b966a378` Added ``mjv_defaultFreeCamera`` which sets the default free camera, respecting the above
   attributes.
#. :commit:`80b4ffdd` ``simulate`` now supports taking a screenshot via a button in the File section or via ``Ctrl-P``.
#. :commit:`834e8dd5` Improvements to time synchronisation in `simulate`, in particular report actual real-time factor
   if different from requested factor (if e.g., the timestep is so small that simulation cannot keep up with real-time).
#. :commit:`090fe2db` Added a disable flag for sensors.
#. :commit:`fdbbc8bb` :ref:`mju_mulQuat` and :ref:`mju_mulQuatAxis` support in place computation. For example
   |br| ``mju_mulQuat(a, a, b);`` sets the quaternion ``a`` equal to the product of ``a`` and ``b``.
#. :commit:`e3a82247` Added sensor matrices to ``mjd_transitionFD`` (note this is an API change).

Deleted/deprecated features
^^^^^^^^^^^^^^^^^^^^^^^^^^^

21. :commit:`c8ff7b3d` Removed ``distance`` constraints.

Bug fixes
^^^^^^^^^

22. :commit:`f71daed6` Fixed rendering of some transparent geoms in reflection.
#. :commit:`0b2f19bb` Fixed ``intvelocity`` defaults parsing.


Version 2.2.1 (July 18, 2022)
-----------------------------

General
^^^^^^^

1. :commit:`228264c9` Added ``mjd_transitionFD`` to compute efficient finite difference approximations of the
   state-transition and control-transition matrices, :ref:`see here<derivatives>` for more details.
#. :commit:`373cc894` Added derivatives for the ellipsoid fluid model.
#. :commit:`09a5efc0` Added ``ctrl`` attribute to :ref:`keyframes<keyframe>`.
#. :commit:`c14a7ef4` Added ``clock`` sensor which :ref:`measures time<sensor-clock>`.
#. :commit:`2d0995b4` Added visualisation groups to skins.
#. :commit:`6ead1461` Added actuator visualisation for ``free`` and ``ball`` joints and for actuators with ``site``
   transmission.
#. :commit:`d3a86bb7` Added visualisation for actuator activations.
#. :commit:`d3a86bb7` Added ``<actuator-intvelocity>`` actuator shortcut for "integrated velocity" actuators, documented
   :ref:`here <actuator-intvelocity>`.
#. :commit:`d3a86bb7` Added ``<actuator-damper>`` actuator shortcut for active-damping actuators, documented
   :ref:`here <actuator-damper>`.
#. :commit:`ec7133b0` ``mju_rotVecMat`` and ``mju_rotVecMatT`` now support in-place multiplication.
#. :commit:`d3a86bb7` ``mjData.ctrl`` values are no longer clamped in-place, remain untouched by the engine.
#. :commit:`d3a86bb7` Arrays in mjData's buffer now align to 64-byte boundaries rather than 8-byte.
#. :commit:`f887c1e9` Added memory poisoning when building with
   `Address Sanitizer (ASAN) <https://clang.llvm.org/docs/AddressSanitizer.html>`__ and
   `Memory Sanitizer (MSAN) <https://clang.llvm.org/docs/MemorySanitizer.html>`__. This allows ASAN to detect reads and
   writes to regions in ``mjModel.buffer`` and ``mjData.buffer`` that do not lie within an array, and for MSAN to detect
   reads from uninitialised fields in ``mjData`` following ``mj_resetData``.
#. :commit:`373cc894` Added a
   `slider-crank example model <https://github.com/google-deepmind/mujoco/tree/main/model/slider_crank>`__.

Bug fixes
^^^^^^^^^

15. :commit:`d3a86bb7` :ref:`Activation clamping <CActRange>` was not being applied in the
    :ref:`implicit integrator<geIntegration>`.
#. :commit:`d3a86bb7` Stricter parsing of orientation specifiers. Before this change, a specification that included both
   ``quat`` and an :ref:`alternative specifier<COrientation>` e.g., ``<geom ... quat=".1 .2 .3 .4" euler="10 20 30">``,
   would lead to the ``quat`` being ignored and only ``euler`` being used. After this change a parse error will be
   thrown.
#. :commit:`f3453cf8` Stricter parsing of XML attributes. Before this change an erroneous XML snippet like
   ``<geom size="1/2 3 4">`` would have been parsed as ``size="1 0 0"`` and no error would have been thrown. Now throws
   an error.
#. :commit:`d3a86bb7` Trying to load a ``NaN`` via XML like ``<geom size="1 NaN 4">``, while allowed for debugging
   purposes, will now print a warning.
#. :commit:`d5672639` Fixed null pointer dereference in ``mj_loadModel``.
#. :commit:`dbef8e6c` Fixed memory leaks when loading an invalid model from MJB.
#. :commit:`d3a86bb7` Integer overflows are now avoided when computing ``mjModel`` buffer sizes.
#. :commit:`d3a86bb7` Added missing warning string for ``mjWARN_BADCTRL``.

Packaging
^^^^^^^^^

23. :commit:`d3a86bb7` Changed MacOS packaging so that the copy of ``mujoco.framework`` embedded in ``MuJoCo.app`` can
    be used to build applications externally.


Version 2.2.0 (May 23, 2022)
----------------------------

Open Sourcing
^^^^^^^^^^^^^

1. :commit:`1913a02b` MuJoCo is now fully open-source software. Newly available top level directories are:

   a. ``src/``: All source files. Subdirectories correspond to the modules described in the Programming chapter
   :ref:`introduction<inIntro>`:

   - ``src/engine/``: Core engine.
   - ``src/xml/``: XML parser.
   - ``src/user/``: Model compiler.
   - ``src/visualize/``: Abstract visualizer.
   - ``src/ui/``: UI framework.

   b. ``test/``: Tests and corresponding asset files.

   c. ``dist/``: Files related to packaging and binary distribution.

#. :commit:`1913a02b` Added
   `contributor's guide <https://github.com/google-deepmind/mujoco/blob/main/CONTRIBUTING.md>`__ and
   `style guide <https://github.com/google-deepmind/mujoco/blob/main/STYLEGUIDE.md>`__.

General
^^^^^^^

3. :commit:`64bc6d27` Added analytic derivatives of smooth (unconstrained) dynamics forces, with respect to velocities:

   - Centripetal and Coriolis forces computed by the Recursive Newton-Euler algorithm.
   - Damping and fluid-drag passive forces.
   - Actuation forces.

#. :commit:`64bc6d27` Added ``implicit`` integrator. Using the analytic derivatives above, a new implicit-in-velocity
   integrator was added. This integrator lies between the Euler and Runge Kutta integrators in terms of both stability
   and computational cost. It is most useful for models which use fluid drag (e.g. for flying or swimming) and for
   models which use :ref:`velocity actuators<actuator-velocity>`. For more details, see the
   :ref:`Numerical Integration<geIntegration>` section.

#. :commit:`1913a02b` Added :at:`actlimited` and :at:`actrange` attributes to
   :ref:`general actuators<actuator-general>`, for clamping actuator internal states (activations). This clamping is
   useful for integrated-velocity actuators, see the :ref:`Activation clamping <CActRange>` section for details.

#. :commit:`1913a02b` ``mjData`` fields ``qfrc_unc`` (unconstrained forces) and ``qacc_unc`` (unconstrained
   accelerations) were renamed ``qfrc_smooth`` and ``qacc_smooth``, respectively. While "unconstrained" is precise,
   "smooth" is more intelligible than "unc".

#. :commit:`1913a02b` Public headers have been moved from ``/include`` to ``/include/mujoco/``, in line with the
   directory layout common in other open source projects. Developers are encouraged to include MuJoCo public headers in
   their own codebase via ``#include <mujoco/filename.h>``.

#. :commit:`1913a02b` The default shadow resolution specified by the :ref:`shadowsize<visual-quality>` attribute was
   increased from 1024 to 4096.

#. :commit:`1913a02b` Saved XMLs now use 2-space indents.

Bug fixes
^^^^^^^^^

10. :commit:`1913a02b` Antialiasing was disabled for segmentation rendering. Before this change, if the
    :ref:`offsamples<visual-quality>` attribute was greater than 0 (the default value is 4), pixels that overlapped with
    multiple geoms would receive averaged segmentation IDs, leading to incorrect or non-existent IDs. After this change
    :at:`offsamples` is ignored during segmentation rendering.

#. :commit:`1913a02b` The value of the enable flag for the experimental multiCCD feature was made sequential with other
    enable flags. Sequentiality is assumed in the ``simulate`` UI and elsewhere.

#. :commit:`1913a02b` Fix issue of duplicated meshes when saving models with OBJ meshes using mj_saveLastXML.


Version 2.1.5 (Apr. 13, 2022)
-----------------------------

General
^^^^^^^

1. :commit:`87539dbd` Added an experimental feature: multi-contact convex collision detection, activated by an enable
   flag. See full description :ref:`here <option-flag>`.

Bug fixes
^^^^^^^^^

2. :commit:`87539dbd` GLAD initialization logic on Linux now calls ``dlopen`` to load a GL platform dynamic library if a
   ``*GetProcAddress`` function is not already present in the process' global symbol table. In particular, processes
   that use GLFW to set up a rendering context that are not explicitly linked against ``libGLX.so`` (this applies to the
   Python interpreter, for example) will now work correctly rather than fail with a ``gladLoadGL`` error when
   ``mjr_makeContext`` is called.

#. :commit:`87539dbd` In the Python bindings, named indexers for scalar fields (e.g. the ``ctrl`` field for actuators)
   now return a NumPy array of shape ``(1,)`` rather than ``()``. This allows values to be assigned to these fields more
   straightforwardly.

Version 2.1.4 (Apr. 4, 2022)
----------------------------

General
^^^^^^^

1. :commit:`90dea1bd` MuJoCo now uses GLAD to manage OpenGL API access instead of GLEW. On Linux, there is no longer a
   need to link against different GL wrangling libraries depending on whether GLX, EGL, or OSMesa is being used.
   Instead, users can simply use GLX, EGL, or OSMesa to create a GL context and ``mjr_makeContext`` will detect which
   one is being used.

#. :commit:`90dea1bd` Added visualisation for contact frames. This is useful when writing or modifying collision
   functions, when the actual direction of the x and y axes of a contact can be important.

Binary build
^^^^^^^^^^^^

3. :commit:`90dea1bd` The ``_nogl`` dynamic library is no longer provided on Linux and Windows. The switch to GLAD
   allows us to resolve OpenGL symbols when ``mjr_makeContext`` is called rather than when the library is loaded. As a
   result, the MuJoCo library no longer has an explicit dynamic dependency on OpenGL, and can be used on system where
   OpenGL is not present.

Simulate
^^^^^^^^

4. :commit:`90dea1bd` Fixed a bug in simulate where pressing '[' or ']' when a model is not loaded causes a crash.

#. :commit:`90dea1bd` Contact frame visualisation was added to the Simulate GUI.

#. :commit:`90dea1bd` Renamed "set key", "reset to key" to "save key" and "load key", respectively.

#. :commit:`90dea1bd` Changed bindings of F6 and F7 from the not very useful "vertical sync" and "busy wait" to the more
   useful cycling of frames and labels.

Bug fixes
^^^^^^^^^

8. :commit:`90dea1bd` ``mj_resetData`` zeroes out the ``solver_nnz`` field.

#. :commit:`90dea1bd` Removed a special branch in ``mju_quat2mat`` for unit quaternions. Previously, ``mju_quat2mat``
   skipped all computation if the real part of the quaternion equals 1.0. For very small angles (e.g. when finite
   differencing), the cosine can evaluate to exactly 1.0 at double precision while the sine is still nonzero.


Version 2.1.3 (Mar. 23, 2022)
-----------------------------

General
^^^^^^^

1. :commit:`df048e54` ``simulate`` now supports cycling through cameras (with the ``[`` and ``]`` keys).
#. :commit:`df048e54` ``mjVIS_STATIC`` toggles all static bodies, not just direct children of the world.

Python bindings
^^^^^^^^^^^^^^^

3. :commit:`df048e54` Added a ``free()`` method to ``MjrContext``.
#. :commit:`df048e54` Enums now support arithmetic and bitwise operations with numbers.

Bug fixes
^^^^^^^^^

5. :commit:`df048e54` Fixed rendering bug for planes, introduced in 2.1.2. This broke maze environments in
   `dm_control <https://github.com/google-deepmind/dm_control>`__.


Version 2.1.2 (Mar. 15, 2022)
-----------------------------

New modules
^^^^^^^^^^^

1. :commit:`3577e2cf` Added new :doc:`Python bindings<python>`, which can be installed via ``pip install mujoco``,
   and imported as ``import mujoco``.
#. :commit:`3577e2cf` Added new :doc:`Unity plug-in<unity>`.
#. :commit:`3577e2cf` Added a new ``introspect`` module, which provides reflection-like capability for MuJoCo's public
   API, currently describing functions and enums. While implemented in Python, this module is expected to be generally
   useful for automatic code generation targeting multiple languages. (This is not shipped as part of the ``mujoco``
   Python bindings package.)

API changes
^^^^^^^^^^^

4. :commit:`3577e2cf` Moved definition of ``mjtNum`` floating point type into a new header
   `mjtnum.h <https://github.com/google-deepmind/mujoco/blob/3577e2cf8bf841475b489aefff52276a39f24d51/include/mjtnum.h>`__.
#. :commit:`3577e2cf` Renamed header `mujoco_export.h` to :ref:`mjexport.h<inHeader>`.
#. :commit:`3577e2cf` Added ``mj_printFormattedData``, which accepts a format string for floating point numbers, for
   example to increase precision.

General
^^^^^^^

7. :commit:`3577e2cf` MuJoCo can load `OBJ <https://en.wikipedia.org/wiki/Wavefront_.obj_file>`__ mesh files.

   a. Meshes containing polygons with more than 4 vertices are not supported.
   #. In OBJ files containing multiple object groups, any groups after the first one will be ignored.
   #. Added (post-release, not included in the 2.1.2 archive) textured
      `mug <https://github.com/google-deepmind/mujoco/blob/main/model/mug/mug.xml>`__ example model:

      .. image:: images/changelog/mug.png
         :width: 300px


#. :commit:`3577e2cf` Added optional frame-of-reference specification to :ref:`framepos<sensor-framepos>`,
   :ref:`framequat<sensor-framequat>`, :ref:`framexaxis<sensor-framexaxis>`, :ref:`frameyaxis<sensor-frameyaxis>`,
   :ref:`framezaxis<sensor-framezaxis>`, :ref:`framelinvel<sensor-framelinvel>`, and
   :ref:`frameangvel<sensor-frameangvel>` sensors. The frame-of-reference is specified by new :at:`reftype` and
   :at:`refname` attributes.

#. :commit:`3577e2cf` Sizes of :ref:`user parameters <CUser>` are now automatically inferred.

   a. Declarations of user parameters in the top-level :ref:`size <size>` clause (e.g. :at:`nuser_body`,
      :at:`nuser_jnt`, etc.) now accept a value of -1, which is the default. This will automatically set the value to
      the length of the maximum associated :at:`user` attribute defined in the model.
   #. Setting a value smaller than -1 will lead to a compiler error (previously a segfault).
   #. Setting a value to a length smaller than some :at:`user` attribute defined in the model will lead to an error
      (previously additional values were ignored).

#. :commit:`3577e2cf` Increased the maximum number of lights in an :ref:`mjvScene` from 8 to 100.

#. :commit:`3577e2cf` Saved XML files only contain explicit :ref:`inertial <body-inertial>` elements if the original XML
   included them. Inertias that were automatically inferred by the compiler's :ref:`inertiafromgeom <compiler>`
   mechanism remain unspecified.

#. :commit:`3577e2cf` User-selected geoms are always rendered as opaque. This is useful in interactive visualizers.

#. :commit:`3577e2cf` Static geoms now respect their :ref:`geom group<body-geom>` for visualisation. Until this change
   rendering of static geoms could only be toggled using the :ref:`mjVIS_STATIC<mjtVisFlag>` visualisation flag . After
   this change, both the geom group and the visualisation flag need to be enabled for the geom to be rendered.

#. :commit:`3577e2cf` Pointer parameters in function declarations in :ref:`mujoco.h<inHeader>` that are supposed to
   represent fixed-length arrays are now spelled as arrays with extents, e.g. ``mjtNum quat[4]`` rather than
   ``mjtNum* quat``. From the perspective of C and C++, this is a non-change since array types in function signatures
   decay to pointer types. However, it allows autogenerated code to be aware of expected input shapes.

#. :commit:`147efb51` Experimental stateless fluid interaction model. As described :ref:`here <gePassive>`, fluid forces
   use sizes computed from body inertia. While sometimes convenient, this is very rarely a good approximation. In the
   new model forces act on geoms, rather than bodies, and have a several user-settable parameters. The model is
   activated by setting a new attribute: ``<geom fluidshape="ellipsoid"/>``. The parameters are described succinctly
   :ref:`here<body-geom>`, but we leave a full description or the model and its parameters to when this feature leaves
   experimental status.

Bug fixes
^^^^^^^^^

16. :commit:`3577e2cf` ``mj_loadXML`` and ``mj_saveLastXML`` are now locale-independent. The Unity plugin should now
    work correctly for users whose system locales use commas as decimal separators.
#. :commit:`3577e2cf` XML assets in VFS no longer need to end in a null character. Instead, the file size is determined
    by the size parameter of the corresponding VFS entry.
#. :commit:`3577e2cf` Fix a vertex buffer object memory leak in ``mjrContext`` when skins are used.
#. :commit:`3577e2cf` Camera quaternions are now normalized during XML compilation.

Binary build
^^^^^^^^^^^^

20. :commit:`3577e2cf` Windows binaries are now built with Clang.

Version 2.1.1 (Dec. 16, 2021)
-----------------------------

API changes
^^^^^^^^^^^

1. :commit:`a499b381` Added ``mj_printFormattedModel``, which accepts a format string for floating point numbers, for
   example to increase precision.
#. :commit:`a499b381` Added ``mj_versionString``, which returns human-readable string that represents the version of the
   MuJoCo binary.
#. :commit:`a499b381` Converted leading underscores to trailing underscores in private instances of API struct
   definitions, to conform to reserved identifier directive, see
   `C standard: Section 7.1.3 <https://www.open-std.org/jtc1/sc22/wg14/www/docs/n1570.pdf>`__.

   .. attention::
      This is a minor breaking change. Code which references private instances will break. To fix, replace leading
      underscores with trailing underscores, e.g. ``_mjModel`` |rarr| ``mjModel_``.

General
^^^^^^^

4. :commit:`a499b381` Safer string handling: replaced ``strcat``, ``strcpy``, and ``sprintf`` with ``strncat``,
   ``strncpy``, and ``snprintf`` respectively.
#. :commit:`ee39340a` Changed indentation from 4 spaces to 2 spaces, K&R bracing style, added braces to one-line
   conditionals.

Bug Fixes
^^^^^^^^^

6. :commit:`a499b381` Fixed reading from uninitialized memory in PGS solver.
#. :commit:`a499b381` Computed capsule inertias are now exact. Until this change, capsule masses and inertias computed
   by the :ref:`compiler <compiler>`'s :at:`inertiafromgeom` mechanism were approximated by a cylinder, formed by the
   capsule's cylindrical middle section, extended on both ends by half the capsule radius. Capsule inertias are now
   computed with the `Parallel Axis theorem <https://en.wikipedia.org/wiki/Parallel_axis_theorem>`__, applied to the two
   hemispherical end-caps.

   .. attention::
      This is a minor breaking change. Simulation of a model with automatically-computed capsule inertias will be
      numerically different, leading to, for example, breakage of golden-value tests.
#. :commit:`a499b381` Fixed bug related to :ref:`force <sensor-force>` and :ref:`torque <sensor-torque>` sensors. Until
   this change, forces and torques reported by F/T sensors ignored out-of-tree constraint wrenches except those produced
   by contacts. Force and torque sensors now correctly take into account the effects of
   :ref:`connect <equality-connect>` and :ref:`weld <equality-weld>` constraints.

   .. note::
      Forces generated by :ref:`spatial tendons <tendon-spatial>` which are outside the kinematic tree (i.e., between
      bodies which have no ancestral relationship) are still not taken into account by force and torque sensors. This
      remains a future work item.

Code samples
^^^^^^^^^^^^

9. :commit:`a499b381` ``testspeed``: Added injection of pseudo-random control noise, turned on by default. This is to
   avoid settling into some fixed contact configuration and providing an unrealistic timing measure.
#. :commit:`a499b381` ``simulate``:

   a. Added slower-than-real-time functionality, which is controlled via the '+' and '-' keys.
   #. Added sliders for injecting Brownian noise into the controls.
   #. Added "Print Camera" button to print an MJCF clause with the pose of the current camera.
   #. The camera pose is not reset when reloading the same model file.

Updated dependencies
^^^^^^^^^^^^^^^^^^^^

11. :commit:`a499b381` ``TinyXML`` was replaced with ``TinyXML2`` 6.2.0.
#. :commit:`a499b381` ``qhull`` was upgraded to version 8.0.2.
#. :commit:`a499b381` ``libCCD`` was upgraded to version 1.4.
#. :commit:`a499b381` On Linux, ``libstdc++`` was replaced with ``libc++``.

Binary build
^^^^^^^^^^^^

15. :commit:`a499b381` MacOS packaging. We now ship Universal binaries that natively support both Apple Silicon and
    Intel CPUs.

    a. MuJoCo library is now packaged as a `Framework Bundle
       <https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPFrameworks/Concepts/FrameworkAnato
       my.html>`__, allowing it to be incorporated more easily into Xcode projects (including Swift projects). Developers
       are encouraged to compile and link against MuJoCo using the ``-framework mujoco`` flag, however all header files
       and the ``libmujoco.2.1.1.dylib`` library can still be directly accessed inside the framework.
    #. Sample applications are now packaged into an Application Bundle called ``MuJoCo.app``. When launched via GUI,
       the bundle launches the ``simulate`` executable. Other precompiled sample programs are shipped inside that bundle
       (in ``MuJoCo.app/Contents/MacOS``) and can be launched via command line.
    #. Binaries are now signed and the disk image is notarized.

#. :commit:`a499b381` Windows binaries and libraries are now signed.
#. :commit:`a499b381` Link-time optimization is enabled on Linux and macOS, leading to an average of \~20% speedup when
   benchmarked on three test models (``cloth.xml``, ``humanoid.xml``, and ``humanoid100.xml``).
#. :commit:`a499b381` Linux binaries are now built with LLVM/Clang instead of GCC.
#. :commit:`a499b381` An AArch64 (aka ARM64) Linux build is also provided.
#. :commit:`a499b381` Private symbols are no longer stripped from shared libraries on Linux and MacOS.

Sample models
^^^^^^^^^^^^^
21. :commit:`a499b381` Clean-up of the ``model/`` directory.

    a. Rearranged into subdirectories which include all dependencies.
    #. Added descriptions in XML comments, cleaned up XMLs.
    #. Deleted some composite models: ``grid1``, ``grid1pin``, ``grid2``, ``softcylinder``, ``softellipsoid``.

#. :commit:`a499b381` Added descriptive animations in ``docs/images/models/`` :

|humanoid|   |particle|


Version 2.1.0 (Oct. 18, 2021)
-----------------------------

New features
^^^^^^^^^^^^

1. Keyframes now have ``mocap_pos`` and ``mocap_quat`` fields (mpos and quat attributes in the XML) allowing mocap
   poses to be stored in keyframes.
2. New utility functions: ``mju_insertionSortInt`` (integer insertion sort) and ``mju_sigmoid`` (constructing a
   sigmoid from two half-quadratics).

General
^^^^^^^

3. The preallocated sizes in the virtual file system (VFS) increased to 2000 and 1000, to allow for larger projects.
#. The C structs in the ``mjuiItem`` union are now named, for compatibility.
#. Fixed: ``mjcb_contactfilter`` type is ``mjfConFilt`` (was ``mjfGeneric``).
#. Fixed: The array of sensors in ``mjCModel`` was not cleared.
#. Cleaned up cross-platform code (internal changes, not visible via the API).
#. Fixed a bug in parsing of XML ``texcoord`` data (related to number of vertices).
#. Fixed a bug in `simulate.cc <https://github.com/google-deepmind/mujoco/blob/main/simulate/simulate.cc>`__ related to
   ``nkey`` (the number of keyframes).
#. Accelerated collision detection in the presence of large numbers of non-colliding geoms (with ``contype==0 and
   conaffinity==0``).

UI
^^

11. Figure selection type changed from ``int`` to ``float``.
#. Figures now show data coordinates, when selection and highlight are enabled.
#. Changed ``mjMAXUIMULTI`` to 35, ``mjMAXUITEXT`` to 300, ``mjMAXUIRECT`` to 25.
#. Added collapsible sub-sections, implemented as separators with state: ``mjSEPCLOSED`` collapsed, ``mjSEPCLOSED+1``
   expanded.
#. Added ``mjITEM_RADIOLINE`` item type.
#. Added function ``mjui_addToSection`` to simplify UI section construction.
#. Added subplot titles to ``mjvFigure``.

Rendering
^^^^^^^^^

18. ``render_gl2`` guards against non-finite floating point data in the axis range computation.
#. ``render_gl2`` draws lines from back to front for better visibility.
#. Added function ``mjr_label`` (for text labels).
#. ``mjr_render`` exits immediately if ``ngeom==0``, to avoid errors from uninitialized scenes (e.g. ``frustrum==0``).
#. Added scissor box in ``mjr_render``, so we don't clear the entire window at every frame.


License manager
^^^^^^^^^^^^^^^

23. Removed the entire license manager. The functions ``mj_activate`` and ``mj_deactivate`` are still there for
    backward compatibility, but now they do nothing and it is no longer necessary to call them.
#. Removed the remote license certificate functions ``mj_certXXX``.

Earlier versions
----------------

For changelogs of earlier versions please see `roboti.us <https://www.roboti.us/download.html>`__.

.. |humanoid| image:: images/models/humanoid.gif
   :width: 270px
.. |particle| image:: images/models/particle.gif
   :width: 270px
