Speculative contacts
====================

Because standard collision detection only inspects geom poses at discrete timesteps, geoms can *tunnel* through each
other whenever their displacement over a single step exceeds their thickness, passing from one side to the other without
overlapping at either step. This is especially problematic for simulations with fast-moving bodies, thin geoms, or large
timesteps.

Speculative contacts work by *speculating* contacts that would have occurred between the current and the next timestep.
This feature can be enabled using the :ref:`speculative<option-flag-speculative>` flag. Speculative contacts work like
normal contacts and are treated the same by the solver. They are produced by the following pipeline:

1. :ref:`Broadphase and sphere filtering <spBroadphase>` expands the sweep-and-prune AABBs and bounding-sphere tests to
   account for each fast-moving geom's linear and rotational motion over the step.
2. :ref:`Conservative advancement <spAdvancement>` sweeps candidate geom pairs to determine the time of impact (TOI)
   :math:`0 < t < 1` as a fraction of the timestep.
3. :ref:`Line-of-sight raycasting <spRaycasting>` casts a ray between the two geoms to exclude speculative contacts
   obscured by another geom in the middle.
4. :ref:`Effective contact distance <spDistance>` inverts the solver's reference acceleration equation
   (:math:`\ari = -b_i (J v)_i - k_i r_i`) to compute a synthetic contact distance ``dist`` so that the constraint
   solver arrests :math:`(1 - t)` of the relative normal velocity over the step and lands the geom on the contact
   surface at :math:`t = 1`.

Example models demonstrating speculative contacts can be found in
`model/speculative/ <https://github.com/google-deepmind/mujoco/tree/main/model/speculative>`__.

Enabling speculative contacts adds scene-dependent collision overhead per step, scaling with the number of geoms whose
single-step travel exceeds their thickness and the candidate pairs they sweep across. At the same time, preventing
tunneling often allows a larger timestep than discrete collision detection would tolerate, reducing overall wall-clock
time.

.. note::
   Speculative sweeps extrapolate geom motion using only pre-step velocities (``mjData.qvel``) and ignore
   acceleration. This is both because collision detection runs in the position stage before forces and accelerations are
   evaluated, and because acceleration contributes an :math:`O(\Delta t^2)` displacement over a single step compared to
   the :math:`O(\Delta t)` velocity term. Consequently, the timestep :math:`\Delta t` cannot be made arbitrarily large:
   if a slow or stationary geom is accelerated across an obstacle within a single step by a large applied force,
   gravity, or another body pushing it during the constraint solve, its pre-step velocity will not predict the motion
   and it can still tunnel.

.. _spBroadphase:

Broadphase and sphere filtering
--------------------------------

To avoid overhead on slow-moving objects, a geom is only treated as fast-moving if the maximum distance traveled by
any point on its surface over the timestep :math:`\Delta t` exceeds the minimum dimension of its local AABB
half-extents. Given pre-step linear displacement :math:`\Delta x = \Delta t \, v` and angular displacement
:math:`\Delta \omega = \Delta t \, \omega` about the geom center :math:`c`, any point :math:`p` on the geom surface
moves over the step by

.. math::
   \Delta p = \Delta x + \Delta \omega \times (p - c)

Since :math:`\|p - c\| \le r_{\text{bound}}`, where :math:`r_{\text{bound}}` is the geom's bounding-sphere radius, this
surface travel is bounded above via the triangle inequality by the sweep distance

.. math::
   s = \|\Delta x\| + \|\Delta \omega\| \, r_{\text{bound}}

Geoms whose sweep distance :math:`s` is smaller than the minimum dimension of their local AABB half-extents cannot
tunnel in a single step and use standard discrete bounds.

Swept bounding-sphere filter
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each candidate geom pair, the bounding-sphere filter tests whether the two bounding spheres can overlap at any
fraction :math:`t \in [0, 1]` of the timestep. Writing :math:`p = x_2 - x_1` for the initial center separation and
:math:`v = \Delta x_2 - \Delta x_1` for the relative linear displacement over the step, the relative center trajectory
is :math:`r(t) = p + t v`. We find the global minimizer :math:`\tau` of :math:`\|r(t)\|^2` on :math:`[0, 1]` by finding
the root :math:`t^*` of the derivative :math:`\frac{d}{dt}\|r(t)\|^2 = 2(p + t v) \cdot v = 0`:

.. math::
   \tau =
   \begin{cases}
      0,   & \phantom{0 \lt {}} t^* \le 0 \\
      t^*, & 0 \lt t^* \lt 1 \\
      1,   & 1 \le t^*
   \end{cases}
   \qquad \text{where } t^* = -\frac{p \cdot v}{\|v\|^2}

The pair is pruned if :math:`\|r(\tau)\| > r_{\text{bound},1} + r_{\text{bound},2} + \text{margin} + \text{gap}`. When
one of the geoms is a plane with unit normal :math:`n`, the filter checks the minimum signed distance at the
endpoints :math:`t \in \{0, 1\}`, pruning the pair if
:math:`\min(d_0,\, d_0 + n \cdot \Delta x) > r_{\text{bound}} + \text{margin} + \text{gap}`.

Swept broadphase AABBs
~~~~~~~~~~~~~~~~~~~~~~~

During sweep-and-prune (:ref:`Selection <coSelection>`), each geom's oriented bounding box is projected onto the three
principal axes :math:`\hat{u}_j` of the scene covariance frame. For a fast-moving geom, the static half-projection along
:math:`\hat{u}_j` is first inflated by its rotational sweep radius :math:`\|\Delta \omega\| \, r_{\text{bound}}` and
intersected with its bounding sphere radius :math:`r_{\text{bound}}` (since rotation about the geom center cannot move
any point outside :math:`r_{\text{bound}}`). The resulting interval :math:`[\text{lo}_j, \text{hi}_j]` is then extended
directionally by the linear displacement projection :math:`\Delta x \cdot \hat{u}_j`:

.. math::
   \big[\text{lo}_j + \min(0, \Delta x \cdot \hat{u}_j) - \text{margin},\;
        \text{hi}_j + \max(0, \Delta x \cdot \hat{u}_j) + \text{margin}\big]

Because per-body bounding volume hierarchies (``mjModel.bvh_aabb``) are precomputed statically in the body frame at
compile time, mid-phase BVH tree traversal is bypassed for body pairs containing a fast-moving geom so that swept geom
pairs are not culled by uninflated internal tree nodes.

.. _spAdvancement:

Conservative advancement
------------------------

There are many approaches to computing the time of impact (see Erin Catto's
`GDC 2013 presentation <https://box2d.org/files/ErinCatto_ContinuousCollision_GDC2013.pdf>`__ for an overview). In
MuJoCo, when discrete narrowphase collision detection finds no active surface contact (``dist < margin``) and at least
one geom in the pair exceeds the fast-sweep threshold, we use a unilateral, *directed conservative advancement*
algorithm to find the time of impact :math:`t \in (0, 1)`.

.. raw:: html

   <style>
     .ca-widget {
       border: 1px solid var(--color-background-border, #e2e8f0);
       border-radius: 10px;
       background: var(--color-background-secondary, #f8fafc);
       padding: 12px 14px;
       margin: 1em 0;
       font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
     }
     .ca-scenes {
       display: flex;
       flex-wrap: wrap;
       align-items: center;
       justify-content: center;
       gap: 6px;
       margin-bottom: 8px;
       padding-bottom: 8px;
       border-bottom: 1px solid var(--color-background-border, #e2e8f0);
     }
     .ca-controls {
       display: flex;
       flex-wrap: wrap;
       align-items: center;
       justify-content: center;
       gap: 4px;
       margin-bottom: 10px;
     }
     .ca-btn {
       border: 1px solid var(--color-background-border, #cbd5e1);
       background: var(--color-background-primary, #ffffff);
       color: var(--color-foreground-primary, #1e293b);
       border-radius: 6px;
       padding: 4px 10px;
       font-size: 12.5px;
       font-weight: 600;
       cursor: pointer;
       transition: all 0.15s ease;
       font-family: inherit;
     }
     .ca-btn:hover {
       border-color: #3b82f6;
       color: #2563eb;
     }
     .ca-btn.ca-active {
       background: #2563eb;
       color: #ffffff;
       border-color: #2563eb;
     }
     .ca-svg-wrap {
       background: var(--color-background-primary, #ffffff);
       border: 1px solid var(--color-background-border, #e2e8f0);
       border-radius: 8px;
       overflow: hidden;
     }
     .ca-svg-wrap svg {
       width: 100%;
       height: auto;
       display: block;
     }
     .ca-caption {
       margin: 10px 0 0 0;
       font-size: 13px;
       line-height: 1.5;
       color: var(--color-foreground-primary, #334155);
     }
     /* Theme-aware SVG classes */
     .ca-widget .ca-txt { fill: #334155; }
     .ca-widget .ca-muted { fill: #64748b; }
     .ca-widget .ca-g1-fill { fill: #dbeafe; fill-opacity: 0.75; stroke: #2563eb; }
     .ca-widget .ca-g2-fill { fill: #ede9fe; fill-opacity: 0.75; stroke: #7c3aed; }
     .ca-widget .ca-sphere1 { fill: none; stroke: #60a5fa; stroke-opacity: 0.45; }
     .ca-widget .ca-sphere2 { fill: none; stroke: #a78bfa; stroke-opacity: 0.45; }
     .ca-widget .ca-path1 { stroke: #93c5fd; stroke-opacity: 0.6; }
     .ca-widget .ca-path2 { stroke: #c4b5fd; stroke-opacity: 0.6; }
     .ca-widget .ca-plane { stroke: #10b981; stroke-opacity: 0.75; }

     body[data-theme="dark"] .ca-widget .ca-txt { fill: #e2e8f0; }
     body[data-theme="dark"] .ca-widget .ca-muted { fill: #94a3b8; }
     body[data-theme="dark"] .ca-widget .ca-g1-fill { fill: #1e3a8a; fill-opacity: 0.65; stroke: #60a5fa; }
     body[data-theme="dark"] .ca-widget .ca-g2-fill { fill: #4c1d95; fill-opacity: 0.65; stroke: #a78bfa; }
     body[data-theme="dark"] .ca-widget .ca-sphere1 { stroke: #60a5fa; stroke-opacity: 0.35; }
     body[data-theme="dark"] .ca-widget .ca-sphere2 { stroke: #a78bfa; stroke-opacity: 0.35; }
     body[data-theme="dark"] .ca-widget .ca-path1 { stroke: #3b82f6; stroke-opacity: 0.4; }
     body[data-theme="dark"] .ca-widget .ca-path2 { stroke: #8b5cf6; stroke-opacity: 0.4; }
     body[data-theme="dark"] .ca-widget .ca-plane { stroke: #34d399; stroke-opacity: 0.75; }
     @media (prefers-color-scheme: dark) {
       body[data-theme="auto"] .ca-widget .ca-txt { fill: #e2e8f0; }
       body[data-theme="auto"] .ca-widget .ca-muted { fill: #94a3b8; }
       body[data-theme="auto"] .ca-widget .ca-g1-fill { fill: #1e3a8a; fill-opacity: 0.65; stroke: #60a5fa; }
       body[data-theme="auto"] .ca-widget .ca-g2-fill { fill: #4c1d95; fill-opacity: 0.65; stroke: #a78bfa; }
       body[data-theme="auto"] .ca-widget .ca-sphere1 { stroke: #60a5fa; stroke-opacity: 0.35; }
       body[data-theme="auto"] .ca-widget .ca-sphere2 { stroke: #a78bfa; stroke-opacity: 0.35; }
       body[data-theme="auto"] .ca-widget .ca-path1 { stroke: #3b82f6; stroke-opacity: 0.4; }
       body[data-theme="auto"] .ca-widget .ca-path2 { stroke: #8b5cf6; stroke-opacity: 0.4; }
       body[data-theme="auto"] .ca-widget .ca-plane { stroke: #34d399; stroke-opacity: 0.75; }
     }
   </style>

   <div class="ca-widget" id="mj-ca-widget">
     <div class="ca-scenes">
       <button type="button" class="ca-btn ca-scene-btn ca-active" data-scene="0">Basic: Sphere &amp; Box (Δω = 0)</button>
       <button type="button" class="ca-btn ca-scene-btn" data-scene="1">Complex: Two Rotating Boxes (Δω ≠ 0)</button>
     </div>
     <div class="ca-controls" id="ca-step-btn-container"></div>

     <div class="ca-svg-wrap">
       <svg viewBox="0 0 660 255" xmlns="http://www.w3.org/2000/svg">
         <defs>
           <marker id="ca-arr-b" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
             <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#3b82f6"/>
           </marker>
           <marker id="ca-arr-p" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
             <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#8b5cf6"/>
           </marker>
           <marker id="ca-arr-g" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto">
             <path d="M 0 1.5 L 8.5 5 L 0 8.5 z" fill="#10b981"/>
           </marker>
         </defs>

         <!-- Dynamic layer populated by script -->
         <g id="ca-layer-Main"></g>
       </svg>
     </div>

     <p class="ca-caption" id="ca-status-text"></p>
   </div>

   <script>

.. raw:: html
   :file: speculative.js

.. raw:: html

   </script>

At each iteration, GJK evaluates the minimal distance :math:`\text{dist}(t)` between the two geoms,
closest witness points :math:`(x_1(t), x_2(t))`, and separating unit normal
:math:`n = (x_2(t) - x_1(t)) / \text{dist}(t)` (pointing from geom 1 to geom 2) at the current advanced poses. By the
Separating Axis Theorem (SAT), the supporting planes perpendicular to :math:`n` at :math:`x_1(t)` and
:math:`x_2(t)` bound every pair of points :math:`p_1(t) \in \text{geom } 1` and :math:`p_2(t) \in \text{geom } 2` along
:math:`n`:

.. math::
   (p_2(t) - p_1(t)) \cdot n \;\ge\; (x_2(t) - x_1(t)) \cdot n \;=\; \text{dist}(t)

As each geom translates and rotates about its center :math:`c_i` with step displacements
:math:`(\Delta x_i, \Delta \omega_i)`, any point :math:`p_i` on geom :math:`i` moves with velocity
:math:`\dot{p}_i = \Delta x_i + \Delta \omega_i \times (p_i - c_i)` with respect to :math:`t`. Because every point on
geom :math:`i` lies within its bounding sphere of radius :math:`r_{\text{bound},i}` centered at :math:`c_i`
(:math:`\|p_i - c_i\| \le r_{\text{bound},i}`) and :math:`\|n\| = 1`, each rotational term satisfies
:math:`\big|n \cdot (\Delta \omega_i \times (p_i - c_i))\big| \le \|\Delta \omega_i\| \, r_{\text{bound},i}`.
Projecting the relative closing velocity :math:`\dot{p}_1 - \dot{p}_2` onto the fixed axis :math:`n` therefore gives an
upper bound :math:`v_{\text{bound}}` on how fast any two points on the geoms can close across the separating plane:

.. math::
   \begin{aligned}
   n \cdot (\dot{p}_1 - \dot{p}_2)
   &= n \cdot (\Delta x_1 - \Delta x_2)
      + n \cdot \big(\Delta \omega_1 \times (p_1 - c_1)\big)
      - n \cdot \big(\Delta \omega_2 \times (p_2 - c_2)\big) \\
   &\le n \cdot (\Delta x_1 - \Delta x_2)
      + \|\Delta \omega_1\| \, r_{\text{bound},1}
      + \|\Delta \omega_2\| \, r_{\text{bound},2}
   \;\equiv\; v_{\text{bound}}
   \end{aligned}

Unlike the worst-case isotropic sweep :math:`s_1 + s_2`—which bounds linear motion in all directions and stalls when a
geom moves rapidly tangentially while grazing a surface—projecting :math:`\Delta x_1 - \Delta x_2` onto :math:`n`
discards tangential linear motion parallel to the separating plane.

If :math:`v_{\text{bound}} \le 0`, the geoms cannot close along :math:`n` and the search terminates without
contact. Otherwise, advancing by :math:`\Delta t \ge 0` can reduce the projection along :math:`n` by at most
:math:`\Delta t \, v_{\text{bound}}`, so the true separation at :math:`t + \Delta t` is bounded below by

.. math::
   \text{dist}(t + \Delta t) \;\ge\; \min_{p_1, p_2}\,(p_2(t + \Delta t) - p_1(t + \Delta t)) \cdot n
                             \;\ge\; \text{dist}(t) - \Delta t \, v_{\text{bound}}

Setting this lower bound equal to a target clearance
:math:`\text{dist}_{\text{target}} = \text{margin} + 0.02\,(\text{dist}(0) - \text{margin})` just above ``margin``
(:math:`\text{dist}(t) - \Delta t \, v_{\text{bound}} = \text{dist}_{\text{target}}`) yields the largest guaranteed-safe
step:

.. math::
   \Delta t = \frac{\text{dist}(t) - \text{dist}_{\text{target}}}{v_{\text{bound}}}, \qquad t \leftarrow t + \Delta t

Unlike *bilateral advancement*—which can overshoot into overlap and step both forward and backward using
signed penetration depth queries—this unilateral advancement approaches :math:`\text{dist}_{\text{target}}` strictly
from the separated side (:math:`\text{dist}(t) > \text{margin}`), so GJK always returns well-defined witness points and
a separating normal without invoking EPA. When :ref:`multiccd<option-flag-multiccd>` is enabled (or one geom is a
plane), polygon clipping extracts up to four contact points at the advanced impact pose.

.. _spRaycasting:

Line-of-sight raycasting
------------------------

The witness points :math:`x_1(t)` and :math:`x_2(t)` found at impact time :math:`t` are mapped from each geom's advanced
local frame back to its :math:`t = 0` world pose :math:`x_1(0)` and :math:`x_2(0)`, and interpolated by each geom's
sweep fraction :math:`w_i = s_i / (s_1 + s_2)` to form the contact position:

.. math::
   x_{\text{contact}} = w_1 \, x_1(0) + w_2 \, x_2(0)

Because narrowphase evaluates each geom pair independently, a fast-moving geom could otherwise generate a speculative
contact with a second object sitting *behind* an intervening obstacle (for example, a falling box registering a
speculative contact with the floor underneath a thin tabletop it is about to strike). To enforce line-of-sight, a ray
segment is cast between the :math:`t = 0` witness points :math:`x_1(0)` and :math:`x_2(0)` and tested against all other
collidable geoms compatible with both bodies; if any intervening geom intersects the segment, the occluded speculative
contact is discarded.

.. _spDistance:

Effective contact distance
--------------------------

Because a speculative contact is created at :math:`t = 0` before geometric overlap occurs, its true separation exceeds
``margin``. Passing the true distance to the constraint solver would either leave the contact inactive or, if activated
with zero penetration, cause the velocity damper in the reference acceleration

.. math::
   \ari = -b_i (J v)_i - k_i r_i

to arrest :math:`100\%` of the closing speed immediately—freezing the body in mid-air at :math:`t = 0`.

Instead, collision detection inverts this reference acceleration equation to solve for an effective contact distance
:math:`\text{dist} = \text{margin} - \delta` such that the constraint solver arrests :math:`(1 - t)` of the incoming
normal speed :math:`|v_n|` over the timestep, leaving the remaining fraction :math:`t` of the velocity to carry the geom
onto the contact surface at the end of the step (:math:`z_1 = z_0 + h v_1 = 0`).

Under the standard Euler integrator, a contact with dimensionless single-step stiffness :math:`k_0 = h^2 k` and damping
:math:`b_0 = h b` (derived from :at:`solref` and :at:`solimp`) produces a single-step normal velocity change

.. math::
   h \ari = b_0 |v_n| + k_0 \frac{\delta}{h}

Setting :math:`h \ari = (1 - t)|v_n|` and solving for the effective penetration depth :math:`\delta` yields

.. math::
   \delta = \frac{1 - t - b_0}{k_0} \, h |v_n|, \qquad \text{dist} = \text{margin} - \delta

Under the implicit :ref:`discrete<option-integrator>` integrator, where the spring-damper is evaluated at the
end-of-step position and divided by the implicit factor :math:`1 + b_0 + k_0`, the numerator :math:`1 - t - b_0` is
replaced by :math:`1 - t - t(b_0 + k_0)`:

.. math::
   \delta_{\text{discrete}} = \frac{1 - t - t(b_0 + k_0)}{k_0} \, h |v_n|

Notice that for default critically damped contacts (:math:`b_0 = 1`), :math:`1 - t - b_0 = -t < 0`, so
:math:`\delta < 0` and :math:`\text{dist} > \text{margin}`: the spring term pulls *back* against the damper by
:math:`t |v_n|` so that only :math:`(1 - t)|v_n|` is removed during the approach step. Speculative contacts remain
active even when :math:`\text{dist} \ge \text{margin}`.

Modeling notes
~~~~~~~~~~~~~~

* **Elastic restitution:** Standard speculative contacts split impact across two steps—arresting :math:`(1 - t)|v_n|` to
  land on the surface on the first step, and resolving surface contact on the next. When modeling nearly elastic
  collisions via direct negative :at:`solref` (e.g. :math:`b_0 \approx 2` so the damper reverses :math:`v_n \to -v_n` in
  a single step, as in
  `newton.xml <https://github.com/google-deepmind/mujoco/blob/main/model/speculative/newton.xml>`__),
  set the stiffness to zero (e.g. ``solref="0 -499"``) with a stiff impedance (``solimp="0.995 0.995 0.001"``) so the
  speculative position spring term does not cancel out the elastic damper during the approach step (noting that zero
  stiffness provides no static position spring for resting contacts).
* **Single-pass pre-step velocities:** Because speculative sweeps use pre-step velocities (``mjData.qvel``), a
  stationary intermediate body struck by a fast neighbor acquires its post-impact velocity only after the solver runs.
  In multi-body impact chains (such as Newton's cradle), keep a small clearance between adjacent bodies or choose a
  timestep where a single step's travel does not jump across two gaps at once.
