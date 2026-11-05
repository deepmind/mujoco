.. raw:: html

    <div id="fetchlines"/>


.. _API:

=========
Functions
=========

.. tip::
   Clicking on a function name below will take you to the source implementation in the GitHub repository.

The main header `mujoco.h <https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mujoco.h>`_ exposes a
large number of functions. However the functions that most users are likely to need are a small fraction.

API function can be classified as:

- **Main entry points**
   - :ref:`Parse and compile<Parseandcompile>` an :ref:`mjModel` from XML files and assets.
   - :ref:`Main simulation<Mainsimulation>` entry points, including :ref:`mj_step`.

- **Support functions**
   - :ref:`Support<Support>` functions requiring :ref:`mjModel` and :ref:`mjData`.
   - Pipeline :ref:`components<Components>`, called from :ref:`mj_step`, :ref:`mj_forward` and :ref:`mj_inverse`.
   - :ref:`Sub components<Subcomponents>` of the simulation pipeline.
   - :ref:`Ray casting<Raycollisions>`.
   - :ref:`Printing<Printing>` of various quantities.
   - :ref:`Virtual file system<Virtualfilesystem>`, used to load assets from memory.
   - :ref:`Asset cache<Assetcache>`, used to speed up model compilation.
   - :ref:`Resources<Resources>`, interfacing with resource providers to load assets.
   - :ref:`Initialization<Initialization>` of data structures.
   - :ref:`Error and memory<Errorandmemory>`.
   - :ref:`Miscellaneous<Miscellaneous>` functions.

- **Visualization, Rendering, UI**
   - :ref:`Abstract interaction<Interaction>`: mouse control of cameras and perturbations.
   - :ref:`Abstract Visualization<Visualization-api>`.
   - :ref:`OpenGL rendering<OpenGLrendering>`.
   - :ref:`Filament rendering<FilamentRenderingApi>`.
   - :ref:`UI framework<UIframework>`.

- **Threads, Plugins, Derivatives**
   - :ref:`Derivatives<Derivatives-api>`.
   - :ref:`Signed Distance Functions<Signeddistancefunction>`.
   - :ref:`Thread<Thread>` |-| -related functions.
   - :ref:`Batched simulation<Batchedsimulation>`.
   - :ref:`Plugin<Plugins-api>` |-| -related functions.

- **Math**
   - Aliases for C :ref:`standard math<Standardmath>` functions.
   - :ref:`Vector math<Vectormath>`.
   - :ref:`Sparse math<Sparsemath>`.
   - :ref:`Quaternions<Quaternions>`.
   - :ref:`Pose transformations<Poses>`.
   - :ref:`Matrix decompositions and solvers<Decompositions>`.

- **Model editing**
   - :ref:`Attachment<Attachment>`.
   - :ref:`Tree elements<AddTreeElements>`.
   - :ref:`Non-tree elements<AddNonTreeElements>`.
   - :ref:`Set actuator parameters<Setactuatorparameters>`.
   - :ref:`Assets<AddAssets>`.
   - :ref:`Find and get utilities<FindAndGetUtilities>`.
   - :ref:`Attribute setters<AttributeSetters>`.
   - :ref:`Attribute getters<AttributeGetters>`.
   - :ref:`Spec utilities<SpecUtilities>`.
   - :ref:`Element initialization<ElementInitialization>`.
   - :ref:`Element casting<ElementCasting>`.

.. include:: functions.rst
