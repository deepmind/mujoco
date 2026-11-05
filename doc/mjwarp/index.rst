.. _MJW:

====================
MuJoCo Warp (MJWarp)
====================

.. toctree::
    :hidden:

    API <api.rst>

MuJoCo Warp (MJWarp) is an implementation of MuJoCo written in `Warp <https://nvidia.github.io/warp/>`__ and optimized
for `NVIDIA <https://nvidia.com>`__ hardware and parallel simulation. MJWarp lives in the
`google-deepmind/mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`__ GitHub repository.

MJWarp is developed and maintained as a joint effort by `NVIDIA <https://nvidia.com>`__ and
`Google DeepMind <https://deepmind.google/>`__.

.. _MJW_tutorial:

Tutorial notebook
=================

The MJWarp basics are covered in a tutorial |notebook| |open in colab|.

.. |notebook| replace:: `[notebook] <https://github.com/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb>`__

.. |open in colab| replace:: `[open in colab] <https://colab.research.google.com/github/google-deepmind/mujoco_warp/blob/main/notebooks/tutorial.ipynb>`__


When To Use MJWarp?
===================

High throughput
---------------

The MuJoCo ecosystem offers multiple options for batched simulation.

- :ref:`mujoco.rollout <PyRollout>`: Python API for multi-threaded calls to :ref:`mj_step` on CPU. High throughput
  can be achieved with hardware that has fast cores and large thread counts, but overall performance of applications
  requiring frequent host<>device transfers (e.g., reinforcement learning with simulation on CPU and learning on GPU)
  may be bottlenecked by transfer overhead.
- :ref:`mjx.step <MjxFunctions>`: `jax.vmap` and `jax.pmap` enable multi-threaded and multi-device simulation with JAX on CPUs, GPUs, or
  TPUs.
- :func:`mujoco_warp.step <mujoco_warp.step>`: Python API for high throughput simulation targeting NVIDIA GPUs. Scales
  better than MJX on nearly every workload, in particular complex simulation scenes. Works well with PyTorch.

.. TODO(robotics-simulation): add step/time comparison plot

Low latency
-----------

MJWarp is optimized for throughput: the total number of simulation steps per unit time whereas MuJoCo is optimized for
latency: time for one simulation step. It is expected that a simulation step with MJWarp will be less performant
than a step with MuJoCo for the same simulation.

As a result, MJWarp is well suited for applications where large numbers of
samples are required, like reinforcement learning, while MuJoCo is likely more useful for real-time applications like
online control (e.g., model predictive control) or interactive graphical interfaces (e.g., simulation-based
teleoperation).

Complex scenes
--------------

MJWarp scales better than MJX for scenes with many geoms or degrees of freedom, but not as well as MuJoCo for single
large kinematic trees. MJWarp can scale to scenes with hundreds of degrees of freedom as long as
:ref:`sleeping <Sleeping>` is enabled and the scene can be partitioned into independent islands with dormant bodies.
Improving performance for single connected mechanisms beyond ~60 DoFs remains an active priority.

.. TODO(robotic-simulation): add graph for ngeom and nv scaling

Differentiability
-----------------

The dynamics API in MJX is automatically differentiable via JAX. We are considering whether to support this in MJWarp
via Warp - if this feature is important to you, please chime in on this issue
`here <https://github.com/google-deepmind/mujoco_warp/issues/500>`__.

.. TODO(robotics-simulation): Newton multi-physics

.. _MJW_install:

Installation
============

**From PyPI:**

.. code-block:: shell

    pip install mujoco-warp

**From source:**

.. code-block:: shell

    git clone https://github.com/google-deepmind/mujoco_warp.git
    cd mujoco_warp
    uv sync --all-extras

To make sure everything is working:

.. code-block:: shell

    uv run pytest -n 8

.. _MJW_Usage:

Basic Usage
===========

Once installed, the package can be imported via ``import mujoco_warp as mjw``. Structs, functions, and enums are
available directly from the top-level :mod:`mjw <mujoco_warp>` module.

Structs
-------
Before running MJWarp functions on an NVIDIA GPU, structs must be copied onto the device via
:func:`mjw.put_model <mujoco_warp.put_model>` and :func:`mjw.make_data <mujoco_warp.make_data>` or
:func:`mjw.put_data <mujoco_warp.put_data>` functions. Placing an :ref:`mjModel` on device yields an
:class:`mjw.Model <mujoco_warp.Model>`. Placing an :ref:`mjData` on device yields an
:class:`mjw.Data <mujoco_warp.Data>`.

.. code-block:: python

  mjm = mujoco.MjModel.from_xml_string("...")
  mjd = mujoco.MjData(mjm)
  m = mjw.put_model(mjm)
  d = mjw.put_data(mjm, mjd)

These MJWarp variants mirror their MuJoCo counterparts but have a few key differences:

#. :class:`mjw.Model <mujoco_warp.Model>` and :class:`mjw.Data <mujoco_warp.Data>` contain Warp arrays that are copied
   onto device.
#. Some fields are missing from :class:`mjw.Model <mujoco_warp.Model>` and :class:`mjw.Data <mujoco_warp.Data>` for
   features that are unsupported.

Batch sizes
-----------

MJWarp is optimized for parallel simulation. A batch of simulations can be specified with three parameters:

- :attr:`nworld <mujoco_warp.Data.nworld>`: Number of worlds to simulate.
- _`nconmax`: Expected number of contacts per world. The maximum number of contacts for all worlds is
  ``nconmax * nworld``.
- _`naconmax`: Alternative to `nconmax`_, maximum number of contacts over all worlds. If `nconmax`_ and `naconmax`_ are
  both set then `nconmax`_ is ignored.
- _`njmax`: Maximum number of constraints per world.

.. admonition:: Semantic difference for `nconmax`_ and `njmax`_.
  :class: note

  It is possible for the number of contacts per world to exceed `nconmax`_ if the total number of contacts for all
  worlds does not exceed ``nworld x nconmax``. However, the number of constraints per world is strictly limited by
  `njmax`_.

.. admonition:: XML parsing
  :class: note

  Values for `nconmax`_ and `njmax`_ are not parsed from :ref:`size/nconmax <size-nconmax>` and
  :ref:`size/njmax <size-njmax>` (these parameters are deprecated). Values for these parameters must be provided to
  :func:`mjw.make_data <mujoco_warp.make_data>` or :func:`mjw.put_data <mujoco_warp.put_data>`.

Functions
---------

MuJoCo functions are exposed as MJWarp functions of the same name, but following
`PEP 8 <https://peps.python.org/pep-0008/>`__-compliant names. Most of the :ref:`main simulation <Mainsimulation>` and
some of the :ref:`sub-components <Subcomponents>` for forward simulation are available from the top-level
:mod:`mjw <mujoco_warp>` module.

Minimal example
---------------

.. code-block:: python

   # Throw a ball at 100 different velocities.

   import mujoco
   import mujoco_warp as mjw
   import warp as wp

   _MJCF=r"""
   <mujoco>
     <worldbody>
       <body>
         <freejoint/>
         <geom size=".15" mass="1" type="sphere"/>
       </body>
     </worldbody>
   </mujoco>
   """

   mjm = mujoco.MjModel.from_xml_string(_MJCF)
   m = mjw.put_model(mjm)
   d = mjw.make_data(mjm, nworld=100)

   # initialize velocities
   wp.copy(d.qvel, wp.array([[float(i) / 100, 0, 0, 0, 0, 0] for i in range(100)], dtype=float))

   # simulate physics
   mjw.step(m, d)

   print(f'qpos:\n{d.qpos.numpy()}')

.. _mjwCLI:

Command line scripts
--------------------

Benchmark an environment with _`testspeed`

.. code-block:: shell

    mjwarp-testspeed benchmark/humanoid/humanoid.xml

.. _mjwViewer:

Interactive environment simulation with MJWarp

.. code-block:: shell

    mjwarp-viewer benchmark/humanoid/humanoid.xml

Feature Parity
==============

MJWarp supports most of the main simulation features of MuJoCo, with a few exceptions. MJWarp will raise an exception if
asked to copy to device an :ref:`mjModel` with field values referencing unsupported features. For the most up-to-date
feature availability, please see
`MuJoCo API Compatibility <https://github.com/google-deepmind/mujoco_warp#mujoco-api-compatibility>`__.

.. _mjwPerf:

Performance Tuning
==================

The following are considerations for optimizing the performance of MJWarp.

.. _mjwGC:

Graph capture
-------------

MJWarp functions, for example :func:`mjw.step <mujoco_warp.step>`, often comprise a collection of kernel launches. Warp
will launch these kernels individually if the function is called directly. To improve performance, especially if the
function will be called multiple times, it is recommended to capture the operations that comprise the function as a CUDA
graph

.. code-block:: python

  with wp.ScopedCapture() as capture:
    mjw.step(m, d)

The graph can then be launched or re-launched

.. code-block:: python

  wp.capture_launch(capture.graph)

and will typically be significantly faster compared to calling the function directly. Please see the
`Warp Graph API reference <https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html#graph-launches>`__ for details.

Batch sizes
-----------

The maximum numbers of contacts and constraints, `nconmax`_ / `naconmax`_ and `njmax`_ respectively, are specified when
creating :class:`mjw.Data <mujoco_warp.Data>` with :func:`mjw.make_data <mujoco_warp.make_data>` or
:func:`mjw.put_data <mujoco_warp.put_data>`. Memory and computation scales with the values of these parameters. For best
performance, the values of these parameters should be set as small as possible while ensuring the simulation does not
exceed these limits.

It is expected that good values for these limits will be environment specific. In practice, selecting good values
typically involves trial-and-error. :func:`mjwarp-testspeed <mujoco_warp.testspeed>` with the flag `--measure_alloc` for
printing the number of contacts and constraints at each simulation step, `--overflow_behavior=error` (default) for
detecting overflows, and programmatic inspection via :ref:`Overflow detection <mjwOverflow>` can all be useful
techniques for iteratively testing values for these parameters.

Solver iterations
-----------------

MuJoCo's default solver settings for the maximum numbers of :ref:`solver iterations<option-iterations>` and
:ref:`linesearch iterations<option-ls_iterations>` are expected to provide reasonable performance. Reducing MJWarp's
settings :attr:`Option.iterations <mujoco_warp.Option.iterations>` and/or
:attr:`Option.ls_iterations <mujoco_warp.Option.ls_iterations>` limits may improve performance and should be secondary
considerations after tuning `nconmax`_ / `naconmax`_ and `njmax`_.

Reducing these limits too much may prevent the constraint solver from converging and can lead to inaccurate or unstable
simulation.

.. admonition:: Impact on Performance: MJX (JAX) and MJWarp
  :class: note

  In :ref:`MJX<mjx>` these solver parameters are key for controlling simulation performance. With MJWarp, in contrast,
  once all worlds have converged the solver can early exit and avoid unnecessary computation. As a result, the values
  of these settings have comparatively less impact on performance.

Contact sensor matching
-----------------------

Scenes that include :ref:`contact sensors<sensor-contact>` have a parameter that specifies the maximum number of matched
contacts per sensor :attr:`Option.contact_sensor_max_match <mujoco_warp.Option.contact_sensor_max_match>`. For best
performance, the value of this parameter should be as small as possible while ensuring the simulation does not exceed
the limit. Matched contacts that exceed this limit will be ignored.

The value of this parameter can be set directly, for example ``model.opt.contact_sensor_maxmatch = 16``, or via an XML
custom numeric field

.. code-block:: xml

   <custom>
     <numeric name="contact_sensor_maxmatch" data="16"/>
   </custom>

Similar to the maximum numbers of contacts and constraints, a good value for this setting is expected to be environment
specific. :func:`mjwarp-testspeed <mujoco_warp.testspeed>` and :func:`mjwarp-viewer <mujoco_warp.viewer>` may be useful
for tuning the value of this parameter.


Memory
------

Simulation throughput is often limited by memory requirements for large numbers of worlds. Considerations for optimizing
memory utilization include:

- CCD colliders require more memory than primitive colliders, see the :ref:`pair-wise colliders table <mjwPairwise>`
  for information about colliders.
- :ref:`multiccd <option-flag-multiccd>` requires more memory than CCD.
- CCD memory requirements scale linearly with :ref:`Option.ccd_iterations <option-ccd_iterations>`.
- A scene with at least one mesh geom and using :ref:`multiccd <option-flag-multiccd>` will have memory requirements
  that scale linearly with the maximum number of vertices per face and with the maximum number of edges per vertex,
  computed over all meshes.

`testspeed`_ provides the flag ``--memory`` for reporting a simulation's total memory utilization and information about
:class:`mjw.Model <mujoco_warp.Model>` and :class:`mjw.Data <mujoco_warp.Data>` fields that require significant memory.
Memory allocated inline, including for CCD and the constraint solver, can also be significant and is reported as
``Other memory``.

.. admonition:: Maximum number of contacts per collider
  :class: note

  Some MJWarp colliders have a different maximum number of contacts compared to MuJoCo (see the
  :ref:`pair-wise colliders table <mjwPairwise>`):

  - ``PLANE<>MESH``: 4 versus 3
  - ``HFieldCCD``: 4 versus ``mjMAXCONPAIR``

Sparse Jacobians (e.g., ``efc.J``, ``ten_J``, ``flexedge_J``, and ``actuator_moment``) store only potentially non-zero entries,
reducing memory and skipping zero arithmetic. For high-DoF scenes (:math:`n_v > 60`), sparsity
enables simulations that are unsupported in dense mode. For example, in the Aloha clutter benchmark
(:math:`n_v = 136`, 2048 worlds, ``njmax = 384``), the sparse representation of ``efc.J`` and its column indices
``efc.J_colind`` requires ~84 MB combined (~42 MB each) compared to ~408 MB for dense ``efc.J``, a +4x reduction in memory.

The :func:`mjw.make_data <mujoco_warp.make_data>` or :func:`mjw.put_data <mujoco_warp.put_data>` argument
``nccdmax`` / ``naccdmax`` can be set to a value less than `nconmax`_ / `naconmax`_ in order to reduce the memory
requirements for CCD. The value for this parameter should be the maximum number of contacts generated by a CCD
collider, per world or for all worlds, respectively. For example, a batched simulation with 10 worlds that generates 80
total contacts with per-collider contacts: mesh-mesh: 30 (CCD), ellipsoid-ellipsoid: 10 (CCD), and sphere-sphere: 40
(primitive) should set `nconmax`_ / `naconmax`_ to at least 8 / 80 (may require more for broadphase) and
``nccdmax`` / ``naccdmax`` to 3 / 30.

Large scenes
------------

Simulating scenes with many DoFs (i.e., `nv`) can be computationally
expensive. However, in many scenarios, a significant portion of the scene may
be stationary. MJWarp can put stationary objects to *sleep* (see
:ref:`Sleeping`), excluding them from the working set of many of its
calculations. Furthermore, MJWarp groups bodies into independent
:ref:`islands <soIsland>`; if all bodies in an island are stationary, the
entire island is put to sleep. Currently, both the collision pipeline and the
constraint solver benefit from sleeping, and more sleeping-aware components
may be added in the future.

Compact solver
~~~~~~~~~~~~~~

To optimize performance in scenes with many total DoFs but a relatively
small number of active DoFs (typically fewer than 64, such as two robot
arms with grippers (16 DoFs) and 8 active objects (48 DoFs)), MJWarp
provides a **compact solver** that leverages this sleeping mechanism:

1. Identifies the set of active DOFs for each world, determined from the active islands.
2. **Compacts** these active DOFs into a single, contiguous dense workspace of a known maximum size (``nvmax``).
3. Executes the constraint solver (Newton) using GPU-optimized tile operations (such as blocked Cholesky
   factorization) of fixed size on this compacted space.
4. Scatters the results back to the global state, freezing the inactive DOFs.

By using a fixed-size compacted workspace, the solver avoids GPU thread divergence and leverages
high-performance tensor/matrix operations optimized for fixed tile sizes.

.. rubric:: Enabling the compact solver

1. Enable the Newton solver:

   - Via XML:

     .. code-block:: xml

        <option solver="Newton"/>

   - Via Python ``MjSpec``:

     .. code-block:: python

        spec = mujoco.MjSpec()
        spec.option.solver = mujoco.mjtSolver.mjSOL_NEWTON

2. Enable sleep:

   - Via XML:

     .. code-block:: xml

        <option>
          <flag sleep="enable"/>
        </option>

   - Via Python ``MjSpec``:

     .. code-block:: python

        spec = mujoco.MjSpec()
        spec.option.enableflags |= mujoco.mjtEnableBit.mjENBL_SLEEP

3. Specify the maximum expected active DOFs for any world (``nvmax``) when allocating data. This sizes the
   compacted workspace.

   - In Python:

     .. code-block:: python

        # Allocate data with a maximum of 64 active DOFs per world
        d = mjw.make_data(mjm, nworld=2048, nvmax=64)

   - Via the command line:

     .. code-block:: shell

        mjwarp-testspeed scene.xml --nvmax=64

   If ``nvmax`` is not specified, it defaults to the full number of DOFs (``nv``). Sizing ``nvmax``
   to a tight upper bound of the expected active DOFs significantly reduces GPU memory usage and improves
   throughput.

.. TODO(taylorhowell): update example with correct island cycles initialization

4. When setting ``nvmax < nv`` it is recommended to initialize all trees to asleep in order to avoid initial dof
   overflow.

   .. code-block:: python

      d.tree_asleep.assign(np.array(np.arange(mjm.ntree, dtype=np.int32)), dtype=np.int32)

.. note::
   Consider increasing the sleep tolerance setting (e.g., ``sleep_tolerance="0.01"`` in XML options or
   ``spec.option.sleep_tolerance = 0.01`` in Python) from its default value (0.001) to more quickly
   sleep objects.

.. _mjwOverflow:

Overflow detection
------------------

MJWarp relies on fixed-size buffer allocations (`nconmax`_ / `naconmax`_, `njmax`_, ``nccdmax`` / ``naccdmax``,
``nvmax``, :attr:`Option.contact_sensor_maxmatch <mujoco_warp.Option.contact_sensor_maxmatch>`) for high-throughput GPU
execution. When simulation demands exceed these pre-allocated limits, an overflow occurs, leading to undefined behavior.

MJWarp tracks overflows per world in :attr:`Data.overflow <mujoco_warp.Data.overflow>`, a 1D Warp array of shape
``(nworld,)`` storing bitmasks of :class:`mjw.OverflowType <mujoco_warp.OverflowType>`.

:attr:`Data.overflow <mujoco_warp.Data.overflow>` is initialized to zero upon creation and cleared on
:func:`mjw.reset_data <mujoco_warp.reset_data>`.

.. rubric:: Checking for overflows programmatically

.. code-block:: python

   mjw.step(m, d)

   # device-to-host transfer
   overflow = d.overflow.numpy()

   # check worlds for any overflow
   any_overflow = np.any(overflow != 0)

   # check for each world for (narrowphase) contact overflow
   has_contact_overflow = (overflow & mjw.OverflowType.NARROWPHASE) != 0

.. rubric:: Controlling warnings

By default, :attr:`Option.warn_overflow <mujoco_warp.Option.warn_overflow>` is ``True``, causing kernels to print
warning messages to stdout via ``wp.printf`` when an overflow occurs. Setting ``m.opt.warn_overflow = False`` suppresses
these GPU-side warning prints while still recording the overflow bitmasks in :attr:`Data.overflow <mujoco_warp.Data.overflow>`.

In `testspeed`_, the command-line flag ``--overflow_behavior=error`` (default) automatically checks
:attr:`Data.overflow <mujoco_warp.Data.overflow>` and aborts with a diagnostic error if any world overflows, while
``--overflow_behavior=continue`` allows execution to proceed with warnings enabled.

.. rubric:: Performance implications

- **Device printf overhead**: Printing warnings from GPU kernels (``m.opt.warn_overflow = True``) serializes execution
  across threads, introduces CUDA device synchronizations, and can severely degrade simulation throughput when many
  parallel worlds overflow. Disabling warnings (``m.opt.warn_overflow = False``) eliminates this GPU I/O overhead.
- **Host-device synchronization**: Reading :attr:`Data.overflow <mujoco_warp.Data.overflow>` on CPU via ``d.overflow.numpy()``
  requires a device-to-host memory copy and pipeline synchronization. In high-throughput reinforcement learning
  pipelines, avoid synchronizing on every simulation step; instead, check :attr:`Data.overflow <mujoco_warp.Data.overflow>`
  periodically (e.g., at environment reset or rollout boundaries) or inspect the tensor directly on device.
- **Simulation fidelity**: If warnings are disabled and :attr:`Data.overflow <mujoco_warp.Data.overflow>` is not checked,
  overflows will lead to undefined behavior.

.. _mjwBatch:

Batched :class:`Model <mujoco_warp.Model>` Fields
=================================================

To enable batched simulation with different model parameter values, many :class:`mjw.Model <mujoco_warp.Model>` fields
have a leading batch dimension. By default, the leading dimension is 1 (i.e., ``field.shape[0] == 1``) and the same
value(s) will be applied to all worlds. It is possible to set batch sizes per field

.. code-block:: python

  m = mjw.put_model(mjm, batch_sizes={"dof_damping": 2})
  m.dof_damping.assign(np.array([[0.1], [0.2]], dtype=float))

and then override the default values. Batched fields will be indexed with a modulo operation of the world id and batch
dimension: ``field[worldid % field.shape[0]]``.

Modifying fields
----------------

The recommended workflow for modifying an :ref:`mjModel` field is to first modify the corresponding :ref:`mjSpec` and
then compile to create a new :ref:`mjModel` with the updated field. However, compilation currently requires a host call:
1 call per new field instance, i.e., ``nworld`` host calls for ``nworld`` instances.

Certain fields are safe to modify directly without compilation, enabling on-device updates. Please see
:ref:`mjModel changes<sichange>` for details about specific fields. Additionally,
`GitHub issue 893 <https://github.com/google-deepmind/mujoco_warp/issues/893>`__ tracks adding on-device updates for a
subset of fields.

Per-world assets
----------------

Per-world assets enable heterogeneous worlds where different worlds simulate different
`assets <https://mujoco.readthedocs.io/en/latest/XMLreference.html#asset>`__ including meshes, height fields, materials,
and textures. The general workflow is:

1. Create an :ref:`mjSpec` with **all** assets.
2. Compile each variant by mutating the spec and calling ``spec.compile()``.
3. Compile a **base** model and create :class:`mjw.Model <mujoco_warp.Model>` from it.
4. Override the relevant :class:`mjw.Model <mujoco_warp.Model>` fields with per-world arrays built from the compiled
   variants.

.. rubric:: Per-world meshes

**Example 1 — Per-world meshes: Geom-level** randomization (1 body, 1 geom, 2 mesh assets):

The base scene includes all mesh assets. The geom references one mesh (``mesh_a``); a second mesh
(``mesh_b``) is available for per-world substitution.

.. code-block:: xml

  <mujoco>
    <asset>
      <mesh name="mesh_a" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="mesh_b" vertex="0 0 0 2 0 0 0 2 0 0 0 2"/>
    </asset>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom name="obj" type="mesh" mesh="mesh_a"/>
      </body>
    </worldbody>
  </mujoco>

.. code-block:: python

  nworld = 4

  # base spec: 1 body with 1 mesh geom, all mesh assets
  spec = mujoco.MjSpec()
  mesh_a = spec.add_mesh()
  mesh_a.name = "mesh_a"
  mesh_a.uservert = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

  mesh_b = spec.add_mesh()
  mesh_b.name = "mesh_b"
  mesh_b.uservert = [0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2]

  body = spec.worldbody.add_body()
  body.pos = [0, 0, 1]
  body.add_freejoint()
  geom = body.add_geom()
  geom.name = "obj"
  geom.type = mujoco.mjtGeom.mjGEOM_MESH
  geom.meshname = "mesh_a"

  # compile each variant
  geom.meshname = "mesh_a"
  mjm_a = spec.compile()
  geom.meshname = "mesh_b"
  mjm_b = spec.compile()

  # restore and compile base
  geom.meshname = "mesh_a"
  mjm = spec.compile()

  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=nworld)

  # build per-world arrays: worlds 0-1 use mesh_a, worlds 2-3 use mesh_b
  geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "obj")
  variants = [mjm_a, mjm_b]
  assignment = [0, 0, 1, 1]  # variant index per world

  # build per-world arrays
  dataid = np.tile(mjm.geom_dataid, (nworld, 1))
  geom_size = np.zeros((nworld, mjm.ngeom, 3))
  geom_aabb = np.zeros((nworld, mjm.ngeom, 2, 3))
  geom_rbound = np.zeros((nworld, mjm.ngeom))
  geom_pos = np.zeros((nworld, mjm.ngeom, 3))
  body_mass = np.zeros((nworld, mjm.nbody))
  body_subtreemass = np.zeros((nworld, mjm.nbody))
  body_inertia = np.zeros((nworld, mjm.nbody, 3))
  body_invweight0 = np.zeros((nworld, mjm.nbody, 2))
  body_ipos = np.zeros((nworld, mjm.nbody, 3))
  body_iquat = np.zeros((nworld, mjm.nbody, 4))

  for w in range(nworld):
    ref = variants[assignment[w]]
    dataid[w, geom_id] = ref.geom_dataid[geom_id]
    geom_size[w] = ref.geom_size
    geom_aabb[w] = ref.geom_aabb.reshape(mjm.ngeom, 2, 3)
    geom_rbound[w] = ref.geom_rbound
    geom_pos[w] = ref.geom_pos
    body_mass[w] = ref.body_mass
    body_subtreemass[w] = ref.body_subtreemass
    body_inertia[w] = ref.body_inertia
    body_invweight0[w] = ref.body_invweight0
    body_ipos[w] = ref.body_ipos
    body_iquat[w] = ref.body_iquat

  m.geom_dataid = wp.array(dataid, dtype=int)
  m.geom_size = wp.array(geom_size, dtype=wp.vec3)
  m.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)
  m.geom_rbound = wp.array(geom_rbound, dtype=float)
  m.geom_pos = wp.array(geom_pos, dtype=wp.vec3)
  m.body_mass = wp.array(body_mass, dtype=float)
  m.body_subtreemass = wp.array(body_subtreemass, dtype=float)
  m.body_inertia = wp.array(body_inertia, dtype=wp.vec3)
  m.body_invweight0 = wp.array(body_invweight0, dtype=wp.vec2)
  m.body_ipos = wp.array(body_ipos, dtype=wp.vec3)
  m.body_iquat = wp.array(body_iquat, dtype=wp.quat)

**Example 2 — Per-world meshes: Body-level** randomization (1 body, 1 or 2 geoms, 3 mesh assets):

.. admonition:: Maximum geom count
   :class: important

   For body-level randomization, the base ``mjModel`` provided to ``mjw.put_model`` should specify the **maximum number
   of geoms** required across all variants. Geom slots that are unused in a particular variant can be disabled
   (e.g., ``contype=0``, ``conaffinity=0``, ``dataid=-1``), but they should still be present as part of the body in the
   base model.

.. code-block:: xml

  <mujoco>
    <asset>
      <mesh name="mA" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="mB" vertex="0 0 0 2 0 0 0 2 0 0 0 2"/>
      <mesh name="mC" vertex="0 0 0 3 0 0 0 3 0 0 0 3"/>
    </asset>
    <worldbody>
      <body name="obj" pos="0 0 1">
        <freejoint/>
        <geom name="obj_0" type="mesh" mesh="mA"/>
        <geom name="obj_1" size=".001" contype="0" conaffinity="0" mass="0"/>
      </body>
    </worldbody>
  </mujoco>

.. code-block:: python

  nworld = 6

  # base spec: body with 2 geom slots (max across variants), all mesh assets
  spec = mujoco.MjSpec()
  for name, scale in [("mA", 1), ("mB", 2), ("mC", 3)]:
    mesh = spec.add_mesh()
    mesh.name = name
    mesh.uservert = [0, 0, 0, scale, 0, 0, 0, scale, 0, 0, 0, scale]

  body = spec.worldbody.add_body()
  body.name = "obj"
  body.pos = [0, 0, 1]
  body.add_freejoint()

  g0 = body.add_geom()
  g0.name = "obj_0"
  g0.type = mujoco.mjtGeom.mjGEOM_MESH
  g0.meshname = "mA"

  # null geom slot: disabled collision, no mesh
  g1 = body.add_geom()
  g1.name = "obj_1"
  g1.size = [0.001, 0, 0]
  g1.contype = 0
  g1.conaffinity = 0
  g1.mass = 0

  # variant A: 1 geom (mesh mA), g1 stays null
  mjm_a = spec.compile()

  # variant B: 2 geoms (mesh mB + mC)
  g0.meshname = "mB"
  g1.type = mujoco.mjtGeom.mjGEOM_MESH
  g1.meshname = "mC"
  g1.contype = 1
  g1.conaffinity = 1
  mjm_b = spec.compile()

  # restore base and compile
  g0.meshname = "mA"
  g1.type = mujoco.mjtGeom.mjGEOM_SPHERE
  g1.contype = 0
  g1.conaffinity = 0
  mjm = spec.compile()

  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm, nworld=nworld)

  # worlds 0-2: variant A (1 active geom), worlds 3-5: variant B (2 active geoms)
  variants = [mjm_a, mjm_b]
  assignment = [0, 0, 0, 1, 1, 1]

  geom0_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "obj_0")
  geom1_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "obj_1")
  body_id = mjm.geom_bodyid[geom0_id]

  # build per-world arrays
  dataid = np.tile(mjm.geom_dataid, (nworld, 1))
  geom_size = np.zeros((nworld, mjm.ngeom, 3))
  geom_rbound = np.zeros((nworld, mjm.ngeom))
  geom_aabb = np.zeros((nworld, mjm.ngeom, 2, 3))
  geom_pos = np.zeros((nworld, mjm.ngeom, 3))
  body_mass = np.zeros((nworld, mjm.nbody))
  body_subtreemass = np.zeros((nworld, mjm.nbody))
  body_inertia = np.zeros((nworld, mjm.nbody, 3))
  body_invweight0 = np.zeros((nworld, mjm.nbody, 2))
  body_ipos = np.zeros((nworld, mjm.nbody, 3))
  body_iquat = np.zeros((nworld, mjm.nbody, 4))

  for w in range(nworld):
    ref = variants[assignment[w]]
    dataid[w] = ref.geom_dataid
    # disable unused geom slot for variant A
    if assignment[w] == 0:
      dataid[w, geom1_id] = -1
    geom_size[w] = ref.geom_size
    geom_rbound[w] = ref.geom_rbound
    geom_aabb[w] = ref.geom_aabb.reshape(mjm.ngeom, 2, 3)
    geom_pos[w] = ref.geom_pos
    body_mass[w] = ref.body_mass
    body_subtreemass[w] = ref.body_subtreemass
    body_inertia[w] = ref.body_inertia
    body_invweight0[w] = ref.body_invweight0
    body_ipos[w] = ref.body_ipos
    body_iquat[w] = ref.body_iquat

  m.geom_dataid = wp.array(dataid, dtype=int)
  m.geom_size = wp.array(geom_size, dtype=wp.vec3)
  m.geom_rbound = wp.array(geom_rbound, dtype=float)
  m.geom_aabb = wp.array(geom_aabb, dtype=wp.vec3)
  m.geom_pos = wp.array(geom_pos, dtype=wp.vec3)
  m.body_mass = wp.array(body_mass, dtype=float)
  m.body_subtreemass = wp.array(body_subtreemass, dtype=float)
  m.body_inertia = wp.array(body_inertia, dtype=wp.vec3)
  m.body_invweight0 = wp.array(body_invweight0, dtype=wp.vec2)
  m.body_ipos = wp.array(body_ipos, dtype=wp.vec3)
  m.body_iquat = wp.array(body_iquat, dtype=wp.quat)

**Batched fields** — fields that must be overridden for per-world meshes:

.. list-table::
   :width: 90%
   :align: left
   :widths: 3 2 3
   :header-rows: 1

   * - Field
     - dtype
     - Shape
   * - ``geom_dataid``
     - ``int``
     - ``(nworld, ngeom)``
   * - ``geom_size``
     - ``wp.vec3``
     - ``(nworld, ngeom)``
   * - ``geom_aabb``
     - ``wp.vec3``
     - ``(nworld, ngeom, 2)``
   * - ``geom_rbound``
     - ``float``
     - ``(nworld, ngeom)``
   * - ``geom_pos``
     - ``wp.vec3``
     - ``(nworld, ngeom)``
   * - ``body_mass``
     - ``float``
     - ``(nworld, nbody)``
   * - ``body_subtreemass``
     - ``float``
     - ``(nworld, nbody)``
   * - ``body_inertia``
     - ``wp.vec3``
     - ``(nworld, nbody)``
   * - ``body_invweight0``
     - ``wp.vec2``
     - ``(nworld, nbody)``
   * - ``body_ipos``
     - ``wp.vec3``
     - ``(nworld, nbody)``
   * - ``body_iquat``
     - ``wp.quat``
     - ``(nworld, nbody)``

Per-world height fields, materials, and textures can be similarly formulated.

.. admonition:: Per-world asset dependent field construction
  :class: note

  MJWarp enables per-world asset functionality but does not provide utilities for construction of dependent per-world
  field variants. Construction is left to the user or environment authoring frameworks.

Batch Rendering
===============

MJWarp provides a batch renderer for high-throughput ray tracing built on
`Warp's accelerated BVHs <https://nvidia.github.io/warp/stable/api_reference/_generated/warp.Bvh.html>`__ for
rendering worlds with multiple cameras in parallel.

Key features:

- **Mesh rendering with textures**: BVH-accelerated mesh rendering with full texture support.
- **Heightfield rendering**: Optimized rendering for heightfields.
- **Flex rendering**: Render :ref:`flex<deformable-flex>` objects.
- **3D Gaussian Splatting (3DGS)**: BVH-accelerated ray tracing and alpha compositing of 3D Gaussian splats with per-world grouping and bidirectional physical occlusion.
- **Lighting and shadows**: Dynamic lighting with configurable shadows; domain randomizable: `light_active`,
  `light_type`, `light_castshadow`, `light_xpos`, `light_xdir`.
- **Heterogeneous multi-camera**: Multiple cameras per world and each camera can have a different resolution
  (`cam_resolution`), field of view (`cam_fovy`, `cam_sensorsize`, `cam_intrinsic`), and output mode (`cam_output`).
- **Domain Randomization**: Per-world :class:`mjw.Model <mujoco_warp.Model>` fields (see
  :ref:`Batched Model Fields <mjwBatch>` above): `geom_matid`, `geom_size`, `geom_rgba`, `mat_texid`, `mat_texrepeat`,
  `mat_rgba`.
- **BVH-accelerated ray/rays API**: Ray casting: Accelerated :func:`mjw.ray <mujoco_warp.ray>`,
  :func:`mjw.rays <mujoco_warp.rays>`, and :ref:`rangefinder sensors <sensor-rangefinder>` via
  `Warp's BVHs <https://nvidia.github.io/warp/stable/api_reference/_generated/warp.Bvh.html>`__.

Basic Usage
-----------

Rendering or raycasting requires a :class:`mjw.RenderContext <mujoco_warp.RenderContext>` which contains BVH structures,
rendering specific fields, and output buffers.

.. code-block:: python

    rc = mjw.create_render_context(
        mjm,
        nworld=1,
        cam_res=(256, 256),           # Override camera resolution (or per-camera list)
        render_rgb=True,              # Enable RGB output (or per-camera list)
        render_depth=True,            # Enable depth output (or per-camera list)
        use_textures=True,            # Apply material textures
        use_shadows=False,            # Enable shadow casting (slower)
        enabled_geom_groups=[0, 1],   # Only render geoms in groups 0 and 1
        cam_active=[True, False],     # Selectively enable/disable cameras
        flex_render_smooth=True,      # Smooth shading for soft bodies
        splat_position=pos,           # Optional (nsplat, 3) 3DGS centers
        splat_rotation=rot,           # Optional (nsplat, 4) 3DGS quaternions (w, x, y, z)
        splat_scale=scale,            # Optional (nsplat, 3) 3DGS standard deviations
        splat_rgba=rgba,              # Optional (nsplat, 4) 3DGS RGB color and opacity
        splat_adr=adr,                # Optional (ngroup+1,) group offsets for per-world splats
        splat_group_id=group_id,      # Optional (nworld,) splat group index per world
    )

Each :class:`mjw.RenderContext <mujoco_warp.RenderContext>` parameter can be applied globally or per camera.
Additionally, values for :class:`mjw.RenderContext <mujoco_warp.RenderContext>` parameters can be parsed from XML:

.. code-block:: xml

    <camera name="front_camera" pos="3 0 2" xyaxes="0 1 0 -0.6 0 0.8" resolution="64 64" output="rgb depth"/>

or set via :ref:`mjSpec <mjspec>` for camera customization.

To render, first call :func:`mjw.refit_bvh <mujoco_warp.refit_bvh>` to update the BVH trees,
followed by :func:`mjw.render <mujoco_warp.render>` to write to output buffers.

.. code-block:: python

    mjw.refit_bvh(m, d, rc)
    mjw.render(m, d, rc)

The output buffers contain stacked pixels for all cameras with shape `(nworld, npixel)` and RGB data is
packed into one `uint32` variable. `RenderContext.rgb_adr` and `RenderContext.depth_adr` provide per-camera indexing.
For convenience, :func:`mjw.get_rgb <mujoco_warp.get_rgb>` and :func:`mjw.get_depth <mujoco_warp.get_depth>`
return processed and reshaped RGB and depth data for a given camera batched for all worlds.

.. code-block:: python

    nworld = 1
    cam_index = 0
    resolution = rc.cam_res.numpy()[cam_index]
    rgb_data = wp.zeros((nworld, resolution[1], resolution[0]), dtype=wp.vec3)
    mjw.get_rgb(rc, rgb_data=rgb_data, cam_id=cam_index)

A complete example can be found in the MJWarp tutorial |notebook| |open in colab|.

Benchmarks
----------

Rendering can be benchmarked using `testspeed`_:

.. code-block:: shell

    mjwarp-testspeed benchmarks/primitives.xml --function=render

For benchmark results across a variety of scenes, see the
`released benchmarks <https://github.com/google-deepmind/mujoco_warp/pull/1113>`__.

3D Gaussian Splatting (3DGS)
----------------------------

MJWarp supports rendering 3D Gaussian Splatting (3DGS) scenes composited with simulated physical MuJoCo objects. Splats are raytraced via a dedicated SAH-constructed BVH (`build_splat_bvh`) and evaluated using a single-hit planar slice approximation along each ray.

Physical Occlusion
~~~~~~~~~~~~~~~~~~
Splats are raytraced and alpha-composited *prior* to solid MuJoCo geometry intersection. If a camera ray hits a physical body (such as a mesh, primitive, or flex), splats located behind the hit distance (`dist`) are occluded by the physical object, while splats in front of the hit distance are blended over the physical object's surface color.

Loading PLY Files
~~~~~~~~~~~~~~~~~
Standard binary little-endian 3DGS `.ply` files can be loaded using the helper in `contrib/render.py` or prepared manually as NumPy arrays:

- `splat_position`: Splat centers in world coordinates `(nsplat, 3)` (`float32`).
- `splat_rotation`: Splat unit quaternions in MuJoCo convention `(w, x, y, z)` `(nsplat, 4)` (`float32`).
- `splat_scale`: Splat standard deviations in each local axis `(nsplat, 3)` (`float32`), exponentiated from log-scale PLY properties.
- `splat_rgba`: Diffuse RGB color (converted from degree-0 spherical harmonics `f_dc_0..2`) and sigmoid-activated opacity `(nsplat, 4)` (`float32`).

.. code-block:: python

    import mujoco_warp as mjw
    import numpy as np

    # Example defining a single splat
    splat_pos = np.array([[0.0, 0.0, 0.5]], dtype=np.float32)
    splat_rot = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # w, x, y, z
    splat_scale = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
    splat_rgba = np.array([[1.0, 0.2, 0.2, 0.9]], dtype=np.float32)

    rc = mjw.create_render_context(
        mjm,
        nworld=1,
        render_rgb=True,
        splat_position=splat_pos,
        splat_rotation=splat_rot,
        splat_scale=splat_scale,
        splat_rgba=splat_rgba,
    )

Per-World Splat Groups
~~~~~~~~~~~~~~~~~~~~~~
Similar to per-world meshes, MJWarp supports heterogeneous per-world 3DGS scenes. Multiple splat scenes can be concatenated into single attribute arrays with group offsets specified by `splat_adr` `(ngroup + 1,)` and assigned to each world via `splat_group_id` `(nworld,)`.

.. code-block:: python

    # Group 0 has 100 splats (indices 0..99), Group 1 has 250 splats (indices 100..349)
    splat_adr = np.array([0, 100, 350], dtype=np.int32)
    # Assign Group 0 to World 0 and Group 1 to World 1
    splat_group_id = np.array([0, 1], dtype=np.int32)

Dynamic Splat Refitting
~~~~~~~~~~~~~~~~~~~~~~~
If splat positions, rotations, or scales change dynamically during simulation (e.g., splats attached to moving rigid bodies), call :func:`mjw.refit_splat_bvh <mujoco_warp.refit_splat_bvh>` before rendering:

.. code-block:: python

    # Update splat positions on device
    wp.copy(rc.splat_position, updated_positions)
    mjw.refit_splat_bvh(rc)
    mjw.render(m, d, rc)

Performance & Raycasting Thresholds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- **BVH Response Cutoff**: Splat bounding boxes are sized to contain density responses down to `SPLAT_MIN_RESPONSE = 0.01`.
- **Bounded Compositing**: Raycasting accumulates up to the first `_MAX_SPLAT_HITS = 32` BVH intersections per ray ordered by depth.
- **Early Termination**: Compositing stops early once remaining ray transmittance drops below `0.005` or individual splat alpha is below 8-bit quantization (`1 / 255`).

CLI Previewer & Camera Orbits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The command-line tool `mjwarp-render` supports rendering 3DGS PLY scenes and generating camera orbit animations:

.. code-block:: shell

    # Render single frame with 3DGS PLY
    mjwarp-render scene.xml --splat=my_scene.ply --width=512 --height=512

    # Render a 128-frame circular camera orbit video around a target
    mjwarp-render scene.xml --splat=my_scene.ply --orbit --orbit_center="0,0,1" --orbit_radius=4.0 --output_video=orbit.gif

Notes
-----

- **Meshes**: Rendering computation scales with mesh complexity, specifically the number of vertices and faces. A
  primitive is expected to have better performance (i.e., higher throughput) compared to a similar-sized
  :ref:`mesh<body-geom-mesh>` or :ref:`heightfield <body-geom-hfield>`.
- **Scaling**: Rendering scales linearly with resolution (total pixel count) and camera count.

.. _mjwFAQ:

Frequently Asked Questions
==========================

Learning frameworks
-------------------

**Does MJWarp work with JAX?**

Yes. MJWarp is interoperable with `JAX <https://jax.readthedocs.io/>`__. Please see the
`Warp Interoperability <https://nvidia.github.io/warp/stable/user_guide/interoperability/jax.html>`__ documentation for details.

Additionally, :ref:`MJX <mjx>` provides a JAX API for a subset of MJWarp's :doc:`API <api>`. The implementation is
specified with ``impl='warp'``.

**Does MJWarp work with PyTorch?**

Yes. MJWarp is interoperable with `PyTorch <https://pytorch.org>`__. Please see the
`Warp Interoperability <https://nvidia.github.io/warp/stable/user_guide/interoperability/pytorch.html>`__ documentation for
details.

**How to train policies with MJWarp physics?**

For examples that train policies with MJWarp physics, please see:

- `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`__: Train via
  `Newton API <https://github.com/newton-physics/newton>`__.
- `mjlab <https://github.com/mujocolab/mjlab>`__: Train directly with MJWarp using PyTorch.
- `MuJoCo Playground <https://github.com/google-deepmind/mujoco_playground>`__: Train via :ref:`MJX API <mjx>`.

Features
--------

.. _mjwDiff:

**Is MJWarp differentiable?**

No. MJWarp is not currently differentiable via
Warp's `automatic differentiation <https://nvidia.github.io/warp/stable/user_guide/differentiability.html>`__
functionality. Updates from the team related to enabling automatic differentiation for MJWarp are tracked in this
`GitHub issue <https://github.com/google-deepmind/mujoco_warp/issues/500>`__.

**Does MJWarp work with multiple GPUs?**

Yes. Warp's ``wp.ScopedDevice`` enables multi-GPU computation

.. code-block:: python

   # create a graph for each device
   graph = {}
   for device in wp.get_cuda_devices():
     with wp.ScopedDevice(device):
       m = mjw.put_model(mjm)
       d = mjw.make_data(mjm)
       with wp.ScopedCapture(device) as capture:
         mjw.step(m, d)
       graph[device] = capture.graph

   # launch a graph on each device
   for device in wp.get_cuda_devices():
     wp.capture_launch(graph[device])

Please see the
`Warp documentation <https://nvidia.github.io/warp/stable/user_guide/execution_and_performance/concurrency.html>`__
for details and
`mjlab distributed training <https://mujocolab.github.io/mjlab/main/source/training/distributed_training.html>`__ for a
reinforcement learning example.

**Is MJWarp on GPU deterministic?**

No. There may be ordering or *small* numerical differences between results computed by different executions of the same
code. This is characteristic of non-deterministic atomic operations on GPU. Set device to CPU with
``wp.set_device("cpu")`` for deterministic results.

Developments for deterministic results on GPU are tracked in this
`GitHub issue <https://github.com/google-deepmind/mujoco_warp/issues/562>`__.

**How are orientations represented?**

Orientations are represented as unit quaternions and follow :ref:`MuJoCo's conventions<siLayout>`:
``w, x, y, z`` or ``scalar, vector``.

.. admonition:: ``wp.quat``
  :class: note

  MJWarp utilizes Warp's `built-in type <https://nvidia.github.io/warp/stable/api_reference/_generated/warp.quat.html>`__
  ``wp.quat``. Importantly however, MJWarp does not utilize Warp's ``x, y, z, w`` quaternion convention or
  operations and instead implements quaternion routines that follow MuJoCo's conventions. Please see
  `math.py <https://github.com/google-deepmind/mujoco_warp/blob/main/mujoco_warp/_src/math.py>`__ for the
  implementations.

**Does MJWarp have a named access API / bind?**

No. Updates for this feature are tracked in this
`GitHub issue <https://github.com/google-deepmind/mujoco_warp/issues/884>`__.

**Why are contacts reported when there are no collisions?**

1 contact will be reported for each unique geom pair that contributes to any collision sensor, even if this geom pair is
not in collision. Unlike MuJoCo or MJX where :ref:`collision sensors<collision-sensors>` make separate calls to
collision routines while computing sensor data, MJWarp computes and stores the data for these sensors in contacts while
running its main collision pipeline.

:ref:`Contact sensors<sensor-contact>` will report the correct information for contacts affecting the physics.

**Why do some arrays have different shapes compared to mjModel or mjData?**

By default for batched simulation, many :class:`mjw.Data <mujoco_warp.Data>` fields having a leading batch dimension of
size ``Data.nworld``. Some :class:`mjw.Model <mujoco_warp.Model>` fields having a leading batch dimension with size
``1``, indicating that this
:ref:`field can be overridden with an array of batched parameters for domain randomization <mjwBatch>`.

Additionally, certain fields including ``Model.M``, ``Data.efc.J``, and ``Data.efc.D`` are padded to enable fast
loading on GPU.

**Why are numerical results from MJWarp and MuJoCo different?**

MJWarp utilizes single precision `float32
<https://nvidia.github.io/warp/stable/api_reference/_generated/warp.float32.html>`__ in contrast to MuJoCo's default
double precision (float64) representation for :ref:`mjtNum`. Solver settings, including iterations, collision detection,
and small friction values may be sensitive to differences in floating point representation.

If you encounter unexpected results, including NaNs, please open a GitHub issue.

**How to fix simulation runtime warnings?**

Warnings are provided when memory requirements exceed existing allocations during simulation:

- `nconmax`_ / `njmax`_: The maximum number of contacts / constraints has been exceeded. Increase the value of the
  setting by updating the relevant argument to :func:`mjw.make_data <mujoco_warp.make_data>` or
  :func:`mjw.put_data <mujoco_warp.put_data>`.
- ``mjw.Option.ccd_iterations``: The convex collision detection algorithm has exceeded the maximum number of iterations.
  Increase the value of this setting in the XML / :ref:`mjSpec` / :ref:`mjModel`. Importantly, this change must be made
  to the :ref:`mjModel` instance that is provided to :func:`mjw.put_model <mujoco_warp.put_model>` and
  :func:`mjw.make_data <mujoco_warp.make_data>` / :func:`mjw.put_data <mujoco_warp.put_data>`.
- ``mjw.Option.contact_sensor_maxmatch``: The maximum number of contact matches for a
  :ref:`contact sensor<sensor-contact>`'s matching criteria has been exceeded. Increase the value of this MJWarp-only
  setting `m.opt.contact_sensor_maxmatch`. Alternatively, refactor the contact sensor matching criteria, for example if
  the 2 geoms of interest are known, specify ``geom1`` and ``geom2``.
- ``height field collision overflow``: The number of potential contacts generated by a height field exceeds
  :ref:`mjMAXCONPAIR <glNumericEngine>` and some contacts are ignored. To resolve this warning, reduce the height field
  resolution or reduce the size of the geom interacting with the height field.

For programmatic inspection of these overflow conditions, check :attr:`Data.overflow <mujoco_warp.Data.overflow>`
and :class:`mjw.OverflowType <mujoco_warp.OverflowType>`. To suppress stdout warning prints during high-throughput
runs, set :attr:`Option.warn_overflow <mujoco_warp.Option.warn_overflow>` to ``False``. See
:ref:`Overflow detection <mjwOverflow>` for details.

Compilation
-----------

**How can compilation time be improved?**

Limit the number of unique colliders that require the general convex collision pipeline. These colliders are listed as
``CONVEX`` in ``MJ_COLLISION_TABLE`` in
`collision_driver.py <https://github.com/google-deepmind/mujoco_warp/blob/e357e88b6b47166d325b564c805d9ecbae659a64/mujoco_warp/_src/collision_driver.py#L45>`__
(and marked as ``CCD`` in the :ref:`pair-wise colliders table <mjwPairwise>`).

**Why are the physics not working as expected after upgrading MJWarp?**

The Warp cache may be incompatible with the current code and should be cleared as part of the debugging process. This
can be accomplished by deleting the directory ``~/.cache/warp`` or via Python

.. code-block:: python

   import warp as wp
   wp.clear_kernel_cache()

**Is it possible to compile MJWarp ahead of time instead of at runtime?**

Yes. Please see Warp's
`Ahead-of-Time Compilation Workflows <https://nvidia.github.io/warp/stable/user_guide/programming_model/code_generation.html#ahead-of-time-compilation-workflows>`__
documentation for details.

Differences from MuJoCo
=======================

This section notes differences between MJWarp and MuJoCo.

Warmstart
---------

If warmstarts are not :ref:`disabled <option-flag-warmstart>`, the MJWarp solver warmstart always initializes the
acceleration with ``qacc_warmstart``. In contrast, MuJoCo performs a comparison between ``qacc_smooth`` and
``qacc_warmstart`` to determine which one is utilized for the initialization.

Inertia matrix factorization
----------------------------

MJWarp performs a per-tree factorization of the inertia matrix that is stored in ``qLD`` where the size of the tree
determines if Warp's ``U'U`` Cholesky factorization
`wp.tile_cholesky <https://nvidia.github.io/warp/stable/language_reference/_generated/warp.tile_cholesky.html>`__,
MuJoCo's sparse reverse-mode ``L'DL`` routine, or a simple diagonal inverse routine is employed.

Options
-------

:class:`mjw.Option <mujoco_warp.Option>` fields correspond to their :ref:`mjOption` counterparts with the following
exceptions:

- :ref:`impratio <option-impratio>` is stored as its inverse square root ``impratio_invsqrt``.
- The constraint solver setting :ref:`tolerance <option-tolerance>` is clamped to a minimum value of ``1e-6``.
- Contact :ref:`override <option-flag-override>` parameters :ref:`o_margin <option-o_margin>`,
  :ref:`o_solref <option-o_solref>`, :ref:`o_solimp <option-o_solimp>`, and :ref:`o_friction <option-o_friction>` are
  not available.

:ref:`disableflags <option-flag>` has the following differences:

- :ref:`mjDSBL_MIDPHASE <mjtDisablebit>` is not available.
- :ref:`mjDSBL_AUTORESET <mjtDisablebit>` is not available.
- :ref:`mjDSBL_NATIVECCD <mjtDisablebit>` changes the default box-box collider from CCD to a primitive collider.

:ref:`enableflags <option-flag>` has the following differences:

- :ref:`mjENBL_OVERRIDE <mjtEnablebit>` is not available.
- :ref:`mjENBL_FWDINV <mjtEnablebit>` is not available.
- :ref:`mjENBL_SPECULATIVE <mjtEnablebit>` is not available.

Additional MJWarp-only options are available:

- ``broadphase``: type of broadphase algorithm (:class:`mjw.BroadphaseType <mujoco_warp.BroadphaseType>`)
- ``broadphase_filter``: type of filtering utilized by broadphase
  (:class:`mjw.BroadphaseFilter <mujoco_warp.BroadphaseFilter>`)
- ``graph_conditional``: use CUDA graph conditional
- ``run_collision_detection``: use collision detection routine
- ``contact_sensor_maxmatch``: maximum number of contacts for contact sensor matching criteria
- ``warn_overflow``: warn if simulation overflow is encountered
- ``deterministic``: bitmask for deterministic execution (:class:`mjw.DeterminismType <mujoco_warp.DeterminismType>`)

.. admonition:: Fluid model
  :class: note

  Modifying fluid model parameters: ``density``, ``viscosity``, or ``wind`` may require updating
  ``Model.has_fluid``.

.. admonition:: Graph capture
  :class: note

  A new :ref:`graph capture <mjwGC>` may be necessary after modifying an :class:`mjw.Option <mujoco_warp.Option>` field
  in order for the updated setting to take effect.

SDF plugins
-----------

SDF collisions support plugins. The following example for
`plugin/sdf/bowl.xml <https://github.com/google-deepmind/mujoco/blob/main/model/plugin/sdf/bowl.xml>`__ illustrates how to implement the
SDF plugin implementation in `bowl.cc <https://github.com/google-deepmind/mujoco/blob/main/plugin/sdf/bowl.cc>`__:

.. code-block:: python

  import mujoco_warp as mjw
  import warp as wp

  # distance function
  @wp.func
  def bowl(p: wp.vec3, attr: wp.vec3) -> float:
    """Signed distance function for a bowl shape.

    attr[0] = height
    attr[1] = radius
    attr[2] = thickness
    """
    height = attr[0]
    radius = attr[1]
    thick = attr[2]
    width = wp.sqrt(radius * radius - height * height)

    # q = (norm_xy(p), p.z)
    q0 = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    q1 = p[2]

    # qdiff = q - (width, height)
    qdiff0 = q0 - width
    qdiff1 = q1 - height

    if height * q0 < width * q1:
      dist = wp.sqrt(qdiff0 * qdiff0 + qdiff1 * qdiff1)
    else:
      q_norm = wp.sqrt(q0 * q0 + q1 * q1)
      dist = wp.abs(q_norm - radius)

    return dist - thick


  # gradient of distance function
  @wp.func
  def bowl_sdf_grad(p: wp.vec3, attr: wp.vec3) -> wp.vec3:
    """Gradient of bowl SDF via finite differences."""
    eps = float(1e-6)
    f0 = bowl(p, attr)

    px = wp.vec3(p[0] + eps, p[1], p[2])
    py = wp.vec3(p[0], p[1] + eps, p[2])
    pz = wp.vec3(p[0], p[1], p[2] + eps)

    grad = wp.vec3(
      (bowl(px, attr) - f0) / eps,
      (bowl(py, attr) - f0) / eps,
      (bowl(pz, attr) - f0) / eps,
    )
    return grad


  # register the bowl SDF plugin
  @wp.func
  def user_sdf(p: wp.vec3, attr: wp.vec3, sdf_type: int) -> float:
    return bowl(p, attr)


  @wp.func
  def user_sdf_grad(p: wp.vec3, attr: wp.vec3, sdf_type: int) -> wp.vec3:
    return bowl_sdf_grad(p, attr)


  # override the module-level hooks
  mjw._src.collision_sdf.user_sdf = user_sdf
  mjw._src.collision_sdf.user_sdf_grad = user_sdf_grad

Physics callbacks
-----------------

MuJoCo provides global `physics callbacks <https://mujoco.readthedocs.io/en/latest/APIreference/APIglobals.html#physics-callbacks>`__
that allow users to inject custom logic into the simulation pipeline. MJWarp supports a similar mechanism, but callbacks
are Python functions set per-model on the :class:`mjw.Model <mujoco_warp.Model>` instance via ``Model.callback`` rather than as global
function pointers.

The following callbacks are available:

.. list-table::
   :width: 90%
   :align: left
   :widths: 3 5
   :header-rows: 1

   * - Callback
     - Description
   * - ``control``
     - Custom control laws, writes to ``Data.ctrl``
   * - ``passive``
     - Custom passive forces, writes to ``Data.qfrc_passive``
   * - ``act_dyn``
     - Custom actuator dynamics, writes to ``Data.act_dot``
   * - ``act_gain``
     - Custom actuator gains, writes to ``Data.actuator_force``
   * - ``act_bias``
     - Custom actuator biases, writes to ``Data.actuator_force``
   * - ``sensor``
     - Custom sensors, writes to ``Data.sensordata``; receives an additional ``stage`` argument
   * - ``contactfilter``
     - Custom contact filtering, writes to ``Data.contact``

.. code-block:: python

  import mujoco
  import mujoco_warp as mjw
  import warp as wp

  _MJCF = r"""
  <mujoco>
    <worldbody>
      <body>
        <geom size=".1"/>
        <joint name="hinge"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  """

  @wp.kernel
  def _ctrl_callback(ctrl_out: wp.array2d(dtype=float)):
    worldid = wp.tid()
    ctrl_out[worldid, 0] = 2.0

  def ctrl_callback(m, d):
    wp.launch(_ctrl_callback, dim=(d.nworld,), outputs=[d.ctrl])

  mjm = mujoco.MjModel.from_xml_string(_MJCF)
  m = mjw.put_model(mjm)
  d = mjw.make_data(mjm)

  m.callback.control = ctrl_callback
  mjw.step(m, d)
  assert d.ctrl.numpy()[0, 0] == 2.0

.. _mjwPairwise:

Pair-wise colliders
-------------------

The table below provides information about the colliders and maximum number of contacts generated for different geom
pairs in MJWarp. Use the toggles to see the maximum number of contacts with the options
:ref:`nativeccd<option-flag-nativeccd>` and :ref:`multiccd<option-flag-multiccd>`.

.. raw:: html

   <div class="pairwise-toggles">
     <div class="pairwise-toggle-item">
       <label class="pairwise-switch">
         <input type="checkbox" id="nativeccd-checkbox" checked>
         <span class="pairwise-slider"></span>
       </label>
       <span>nativeccd</span>
     </div>
     <div class="pairwise-toggle-item">
       <label class="pairwise-switch">
         <input type="checkbox" id="multiccd-checkbox" checked>
         <span class="pairwise-slider"></span>
       </label>
       <span>multiccd</span>
     </div>
   </div>

.. list-table::
   :header-rows: 1
   :stub-columns: 1
   :widths: auto
   :class: table-pairwise

   * -
     - Sphere
     - Capsule
     - Ellipsoid
     - Cylinder
     - Box
     - Mesh
     - SDF
   * - Plane
     - | primitive
       | **1**
     - | primitive
       | **2**
     - | primitive
       | **1**
     - | primitive
       | **4**
     - | primitive
       | **4**
     - | primitive
       | **4**
     - | primitive
       | **1**
   * - HField
     - | HFieldCCD
       | **4**
     - | HFieldCCD
       | **4**
     - | HFieldCCD
       | **4**
     - | HFieldCCD
       | **4**
     - | HFieldCCD
       | **4**
     - | HFieldCCD
       | **4**
     - | HFieldSDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Sphere
     - | primitive
       | **1**
     - | primitive
       | **1**
     - | CCD
       | **1**
     - | primitive
       | **1**
     - | primitive
       | **1**
     - | CCD
       | **1**
     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Capsule
     -
     - | primitive
       | **2**
     - | CCD
       | **1**
     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-cap-ccd">2</strong></div>

     - | primitive
       | **2**
     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-cap-ccd">2</strong></div>

     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Ellipsoid
     -
     -
     - | CCD
       | **1**
     - | CCD
       | **1**
     - | CCD
       | **1**
     - | CCD
       | **1**
     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Cylinder
     -
     -
     -
     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-ccd">4</strong></div>

     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-ccd">4</strong></div>

     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-ccd">4</strong></div>

     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Box
     -
     -
     -
     -
     -
       .. raw:: html

         <div class="mjw-boxbox">
           <div class="line">CCD</div>
           <div class="line"><strong>4</strong></div>
         </div>

     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-ccd">4</strong></div>

     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - Mesh
     -
     -
     -
     -
     -
     -
       .. raw:: html

         <div class="line">CCD</div>
         <div class="line"><strong class="mjw-ccd">4</strong></div>

     - | MeshSDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`
   * - SDF
     -
     -
     -
     -
     -
     -
     - | SDF
       | :ref:`sdf_initpoints <option-sdf_initpoints>`

.. raw:: html

   <script>
     (() => {
       const nativeccd = document.getElementById('nativeccd-checkbox');
       const multiccd = document.getElementById('multiccd-checkbox');

       const update = () => {
         const isNative = nativeccd && nativeccd.checked;
         const isMulti = multiccd && multiccd.checked;

         const ccdVal = isMulti ? '4' : '1';
         document.querySelectorAll('.mjw-ccd').forEach(el => el.textContent = ccdVal);

         const capVal = isMulti ? '2' : '1';
         document.querySelectorAll('.mjw-cap-ccd').forEach(el => el.textContent = capVal);

         const boxbox = document.querySelector('.mjw-boxbox');
         if (boxbox) {
           if (!isNative) {
             boxbox.innerHTML = '<div class="line">primitive</div><div class="line"><strong>8</strong></div>';
           } else {
             boxbox.innerHTML = `<div class="line">CCD</div><div class="line"><strong>${ccdVal}</strong></div>`;
           }
         }
       };

       [nativeccd, multiccd].forEach(cb => cb && cb.addEventListener('change', update));
     })();
   </script>

.. _mjwBoxBox:

Box-box collisions
------------------

By default, box-box collisions use the general-purpose convex collision pipeline (GJK/EPA). A specialized primitive
collider based on
`engine_collision_box.c <https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_collision_box.c>`__
is available by setting the ``NATIVECCD`` disable flag:

.. code-block:: python

   m.opt.disableflags |= mjw.DisableBit.NATIVECCD

The specialized collider generates up to 8 contact points, compared to up to 4 for the convex pipeline, and may improve
contact stability for tasks involving box stacking or manipulation.

Rendering
---------

The batch renderer included in MJWarp serves a different purpose than MuJoCo's renderer. The MJWarp batch
renderer is a single hit raycaster optimized for high throughput and low fidelity.

It supports:
 * Simple lambertian diffuse shading
 * Basic point lights and directional lights
 * Textures
 * Shadows
 * 3D Gaussian Splatting (3DGS) diffuse rendering with physical occlusion

It does not support:
 * Advanced lighting effects such as global illumination
 * Physically based material properties
 * View-dependent spherical harmonics (higher degree SH > 0) for 3DGS
