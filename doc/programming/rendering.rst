.. _Rendering:

Rendering
---------

MuJoCo provides two built-in 3D rendering backends that support both interactive onscreen viewers and headless offscreen
rendering:

- :ref:`Filament rendering <FilamentRendering>` (``mjrf``): A modern, physically-based rendering (PBR) engine supporting
  OpenGL, Vulkan, Metal, and WebGL. It uses a **retained-mode**, asynchronous architecture in which meshes, textures,
  lights, and :ref:`renderables <mjrfRenderable>` are created once from :ref:`mjModel` and updated in place from
  :ref:`mjData` each frame. Filament powers :ref:`MuJoCo Studio <Studio>` and is the recommended renderer for new
  applications.
- :ref:`OpenGL rendering <ClassicRendering>` (``mjr``): The classic fixed-function OpenGL 1.5 renderer, now in
  maintenance mode. It uses an **immediate-mode** scene architecture in which :ref:`mjv_updateScene` populates an
  :ref:`mjvScene` with all geometric objects and lights on every frame, and :ref:`mjr_render` draws it. It powers the
  classic :ref:`simulate.cc <saSimulate>` and :ref:`basic.cc <saBasic>` samples.

Both renderers share the graphics-independent :ref:`Visualization <Visualization>` (``mjv``) layer for camera control
(:ref:`mjvCamera`), visualization options (:ref:`mjvOption`), object picking (:ref:`mjv_select`), and interactive
perturbations (:ref:`mjvPerturb`). While the classic OpenGL renderer relies on :ref:`mjvScene` for the entire scene, the
Filament renderer updates model geometry directly from :ref:`mjData` and uses :ref:`mjvScene` only for transient
decorative overlays (``mjCAT_DECOR``). The table below maps the main concepts of the two renderers onto each other:

.. list-table::
   :widths: 22 39 39
   :header-rows: 1

   * -
     - OpenGL (``mjr``)
     - Filament (``mjrf``)
   * - GPU context
     - :ref:`mjrContext`, requires an externally created OpenGL context
     - :ref:`mjrfContext`, owns its graphics context (OpenGL or Vulkan, windowed or headless)
   * - Model resources
     - Uploaded by :ref:`mjr_makeContext`
     - :ref:`mjrfMesh` and :ref:`mjrfTexture`, created by the ``ModelResources`` :ref:`support class <reSupport>`
   * - Per-frame scene
     - :ref:`mjv_updateScene` fills :ref:`mjvScene`
     - ``ModelRenderables::Update`` updates persistent :ref:`mjrfRenderable` objects in place; :ref:`mjvScene` only
       for decorations
   * - Offscreen buffer
     - ``mjFB_OFFSCREEN``, sized by ``mjModel.vis``
     - :ref:`mjrfRenderTarget`, any size and format
   * - Pixel readback
     - :ref:`mjr_readPixels`
     - :ref:`mjrfReadRequest`, submitted together with the render request
   * - Modes and effects
     - ``mjvScene.flags`` (:ref:`mjtRndFlag`) and ``mjvScene.stereo``
     - :ref:`mjrDrawMode` and per-request flags in :ref:`mjrfRenderRequest`; stereo is two requests with
       different cameras
   * - 2D overlays and text
     - :ref:`mjr_overlay`, :ref:`mjr_text`, :ref:`mjr_rectangle`, :ref:`mjr_figure`
     - None; labels are reported through a callback and drawn by the application

.. _FilamentRendering:

Filament rendering
~~~~~~~~~~~~~~~~~~

MuJoCo provides a renderer based on `Filament <https://github.com/google/filament>`__, Google's real-time physically
based rendering engine. Filament has a small and fast runtime, runs on Linux, Windows, macOS, Android, iOS and the Web,
and supports the OpenGL, Vulkan, Metal and WebGL graphics backends. The Filament renderer powers :ref:`MuJoCo Studio
<Studio>` and `live.mujoco.org <https://live.mujoco.org>`__.

The Filament renderer is an optional, self-contained library named ``mujoco_filament``. It implements the ``mjrf``
:ref:`types<tyFilamentRenderStructure>` and :ref:`functions<FilamentRenderingApi>` defined in `mjrfilament.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrfilament.h>`__.

While the filament runtime is very small and efficient, it is a rather large build dependency. As such, if you intend to
:ref:`build MuJoCo from source<inBuild>`, you will need to set the CMake ``MUJOCO_USE_FILAMENT`` option in order to
include the Filament renderer.

.. _reFilamentCompat:

It is also possible to use Filament rendering with the classic ``mjr`` :ref:`API<OpenGLrendering>` by additionally
setting the ``MUJOCO_USE_FILAMENT_MJR_COMPAT`` CMake option. This option effectively replaces the OpenGL-based ``mjr``
functions with Filament-based implementations. This is useful for experimenting with the Filament renderer without
having to rewrite application code. However, this compatibility mode does not support the full range of customization
and effects available in the Filament runtime and comes with a slight performance overhead.

You can find an example of using the Filament renderer in the :ref:`render.cc <saRender>` sample and, specifically, in
the ``RenderFilament()`` function.

.. admonition:: Light intensity switches the lighting model
   :class: tip

   The Filament renderer decides how to light the **entire scene** from a single property: whether any
   :ref:`light<body-light>` in the model has a positive :ref:`intensity<body-light-intensity>`.

   - If **no** light has positive intensity, the model is assumed to be authored for the classic renderer. A default
     environment light is added, the headlight is enabled, and a fixed total intensity is distributed among the model's
     lights.
   - If **any** light has positive intensity, all lights are treated as photometric: each light contributes exactly its
     own intensity (so lights left at zero go dark), the headlight is turned off, and there is no ambient illumination
     unless the model defines an ``image`` light.

   Setting ``intensity`` on a single light therefore changes the appearance of everything else. See :ref:`Lighting
   <reLighting>` below for details.

.. _reProgramming:

Programming
^^^^^^^^^^^

The Filament renderer is organized in three layers. At the base is the externally developed `Filament library
<https://github.com/google/filament>`__ itself, which provides the physically-based rendering pipeline and graphics API
backends. The public interface to the renderer is the ``mjrf`` C API, declared in `mjrfilament.h
<https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjrfilament.h>`__. Unlike the classic ``mjr`` API,
which redraws an :ref:`mjvScene` every frame, ``mjrf`` is *retained-mode* and *asynchronous*: applications create
long-lived objects, mutate them incrementally, and submit batches of render and pixel-readback requests. On top of the C
API, the C++ :ref:`support layer <reSupport>` in `src/render/filament/support
<https://github.com/google-deepmind/mujoco/blob/main/src/render/filament/support>`__ (also bound in Python under
``mujoco.mjrf``) automates creating and updating ``mjrf`` scene objects from :ref:`mjModel`, :ref:`mjData`, and
:ref:`mjvOption`.

Below is a simplified C++ outline (adapted from ``RenderFilament()`` in :ref:`render.cc <saRender>`) illustrating how
these components fit together for offscreen rendering:

.. code-block:: C

   // create Filament context and scene
   mjrfContextConfig context_cfg;
   mjrf_defaultContextConfig(&context_cfg);
   auto ctx = mujoco::CreateContext(context_cfg);

   mjrfSceneParams scene_params;
   mjrf_defaultSceneParams(&scene_params);
   auto scene = mujoco::CreateScene(ctx.get(), scene_params);
   mjrf_configureSceneFromModel(scene.get(), m);

   // create retained model components and decoration helper
   auto model_resources   = std::make_unique<ModelResources>(m, ctx.get());
   auto model_lights      = std::make_unique<ModelLights>(scene.get(), model_resources.get());
   auto model_renderables = std::make_unique<ModelRenderables>(scene.get(), model_resources.get());
   auto model_decorations = std::make_unique<ModelDecorations>(ctx.get(), scene.get(), m);

   // create offscreen render target
   mjrfRenderTargetConfig target_cfg;
   mjrf_defaultRenderTargetConfig(&target_cfg);
   target_cfg.width        = width;
   target_cfg.height       = height;
   target_cfg.color_format = mjPIXEL_FORMAT_RGB8;
   target_cfg.depth_format = mjPIXEL_FORMAT_DEPTH32F;
   auto render_target      = mujoco::CreateRenderTarget(ctx.get(), target_cfg);

   // update retained lights, renderables, and transient decorations from mjData
   mjrRect viewport = {0, 0, width, height};
   model_renderables->SetOptions(opt);
   model_lights->Update(d);
   model_renderables->Update(d);
   model_decorations->Update(d, &opt, nullptr, &cam, viewport);

   // submit render and pixel-readback requests and wait for completion
   mjrfRenderRequest request;
   mjrf_defaultRenderRequest(&request);
   request.scene               = scene.get();
   request.camera              = mjv_camera2GLCamera(m, d, &cam);
   request.viewport            = viewport;
   request.target              = render_target.get();
   request.headlight    = m->vis.headlight.active;
   request.headlight_intensity = model_lights->GetHeadlightIntensity();

   mjrfReadRequest read_request;
   mjrf_defaultReadRequest(&read_request);
   read_request.target    = render_target.get();
   read_request.output    = rgb.data();
   read_request.num_bytes = width * height * 3;

   mjrfFrameHandle frame = mjrf_render(ctx.get(), &request, 1, &read_request, 1);
   mjrf_waitForFrame(ctx.get(), frame);

   // remove transient decorations before destroying the scene
   model_decorations->Clear();

.. _reConcepts:

Core concepts
'''''''''''''

The ``mjrf`` C API revolves around seven opaque object types, each created with an ``mjrf_create*`` function and freed
with ``mjrf_destroy*``: :ref:`mjrfContext`, :ref:`mjrfScene`, :ref:`mjrfMesh`, :ref:`mjrfTexture`,
:ref:`mjrfRenderable`, :ref:`mjrfLight`, and :ref:`mjrfRenderTarget`. Creation takes a struct initialized by
``mjrf_default*``, whose name encodes a rule: a ``*Config`` (context, texture, mesh, render target) is fixed for the
lifetime of the object, while ``*Params`` (scene, light, renderable) are initial values that can later be changed with
``mjrf_set*`` functions. Plain structs such as :ref:`mjrfMaterial` and :ref:`mjrfRenderRequest` describe how these
objects are configured and drawn.

**Context and scene** (:ref:`mjrfContext`, :ref:`mjrfScene`)
   :ref:`mjrfContext` is the entry point to the Filament renderer. Unlike the classic :ref:`mjrContext`, it manages its
   own graphics context (OpenGL or Vulkan, selected via :ref:`mjrGraphicsApi` in :ref:`mjrfContextConfig`, with optional
   window handle or headless software rendering) and is independent of any single :ref:`mjModel`. A :ref:`mjrfScene` is
   a container of lights, renderables, and an optional skybox texture; scene-level rendering options and custom
   ``filament.*`` flags from a model are applied via :ref:`mjrf_configureSceneFromModel`.

**Meshes and textures** (:ref:`mjrfMesh`, :ref:`mjrfTexture`)
   GPU geometry and image assets are created once and retained across frames. :ref:`mjrfMesh` stores vertex attributes
   (positions, packed tangent-frame orientations, UVs) and indices uploaded via :ref:`mjrf_uploadMesh`, along with
   bounding boxes used for frustum culling. :ref:`mjrfTexture` stores 2D, cubemap, or KTX-compressed image data uploaded
   via :ref:`mjrf_uploadTexture`. Both upload functions accept an optional completion callback for freeing host memory
   once the asynchronous GPU transfer finishes; calling :ref:`mjrf_flush` flushes pending operations to the rendering
   backend and invokes any completed callbacks before the next :ref:`mjrf_render` call.

**Renderables and materials** (:ref:`mjrfRenderable`, :ref:`mjrfMaterial`)
   An :ref:`mjrfRenderable` is a persistent drawable entity in a scene, combining surface geometry, material appearance,
   and a 3D transform:

   - **Geometry**: A renderable can reference either a custom :ref:`mjrfMesh` (via :ref:`mjrf_setRenderableMesh`) or a
     built-in geometric primitive such as a sphere, capsule, cylinder, box, or ellipsoid (via
     :ref:`mjrf_setRenderableGeomMesh`, which caches and shares underlying mesh buffers across renderables with matching
     quality parameters).
   - **Material**: :ref:`mjrf_setRenderableMaterial` applies an :ref:`mjrfMaterial` struct, which selects between three
     shading models: metallic-roughness PBR, legacy specular-glossiness (Phong), or unlit rendering for UI and
     decorative overlays (``decor``, which also places the object on the decor layer and disables shadows). The
     material also carries texture bindings (color, normal, metallic, roughness, occlusion, packed
     ORM, emissive), UV scale/offset, selection highlighting (``selected``), and object/island IDs for segmentation and
     solver-island visualization.
   - **Pose, size, and shadows**: On each frame, :ref:`mjrf_setRenderablePose` updates the renderable's 3D position and
     rotation matrix in place, :ref:`mjrf_setRenderableSize` updates its scale (handling non-uniform primitive scaling
     such as keeping capsule end-caps spherical), and :ref:`mjrf_setRenderableCastShadow` toggles whether it casts
     shadows. Renderables are added to or removed from a scene with :ref:`mjrf_addSceneRenderable` and
     :ref:`mjrf_removeSceneRenderable`.

**Lights** (:ref:`mjrfLight`)
   :ref:`mjrfLight` represents directional, spot, point, or image-based (IBL environment cubemap) light sources added to
   a scene via :ref:`mjrf_addSceneLight`. Their poses, intensities (in candela), colors, and shadow map resolutions
   can be updated dynamically. Note that the camera-aligned **headlight** is specified on :ref:`mjrfRenderRequest`
   (with intensity in lux) rather than as a scene light, so rendering the same scene from multiple cameras does not
   cross-illuminate views.

**Render targets, requests, and frame handles** (:ref:`mjrfRenderTarget`, :ref:`mjrfRenderRequest`, :ref:`mjrfReadRequest`)
   Rendering is submitted in batches via :ref:`mjrf_render`. Each :ref:`mjrfRenderRequest` pairs an :ref:`mjrfScene`
   with a camera (``mjrCamera``), viewport, optional offscreen :ref:`mjrfRenderTarget` (or ``NULL`` for the native
   window), draw mode (:ref:`mjrDrawMode`: default, no-textures, wireframe, depth, solver islands, or segmentation by
   ID/color), and per-view flags (shadows, reflections, post-processing, headlight). Because camera and draw mode belong
   to the request rather than the scene, a single retained :ref:`mjrfScene` can be rendered from multiple cameras or in
   multiple draw modes in one call. Offscreen pixel readbacks are requested alongside render requests via
   :ref:`mjrfReadRequest`. :ref:`mjrf_render` dispatches work to a dedicated render thread and returns an
   ``mjrfFrameHandle``, which the caller synchronizes on with :ref:`mjrf_waitForFrame`. Despite using an internal render
   thread, the ``mjrf`` API is not thread-safe and all calls must be made from a single thread.

.. _reSupport:

Support layer
'''''''''''''

While applications can manage low-level ``mjrf`` objects directly, most MuJoCo applications use the support layer in
`src/render/filament/support <https://github.com/google-deepmind/mujoco/blob/main/src/render/filament/support>`__ (and
`support/mjrf_ptr.h <https://github.com/google-deepmind/mujoco/blob/main/src/render/filament/support/mjrf_ptr.h>`__ for
RAII ``UniquePtr`` wrappers), which is also exposed in Python under ``mujoco.mjrf``:

- ``ModelResources``: Creates and owns all :ref:`mjrfMesh` (meshes, convex hulls, and heightfields) and
  :ref:`mjrfTexture` resources from an :ref:`mjModel`, and supports re-uploading modified assets dynamically.
- ``ModelLights``: Populates and updates the scene's :ref:`mjrfLight` instances from :ref:`mjModel` and :ref:`mjData`,
  and automatically configures fallback environment and directional lighting when a model does not define photometric
  light intensities. Since the headlight belongs to the render request, ``GetHeadlightIntensity()`` reports the
  intensity that the application should pass in :ref:`mjrfRenderRequest`.
- ``ModelRenderables``: Creates persistent :ref:`mjrfRenderable` entities for all model geoms, sites, skins, flexes, and
  spatial tendons. Calling ``Update(data)`` updates their poses, sizes, and dynamic material properties in place from
  :ref:`mjData` without constructing an :ref:`mjvScene`, while ``SetOptions(opt)`` applies group and category visibility
  from :ref:`mjvOption`.
- ``ModelDecorations``: Bridges the abstract visualizer for transient decorative geometry. Because decorative overlays
  (contact force arrows, inertia boxes, joint/actuator axes, perturbation tethers, text labels) are generated
  procedurally each frame rather than stored as persistent model elements, ``ModelDecorations`` internally calls
  :ref:`mjv_updateScene` and converts only the ``mjCAT_DECOR`` geoms (and non-skin flex debug wireframes) into temporary
  decor renderables. The Filament renderer has no 2D text API; text labels are instead reported to an optional callback
  passed to ``Update()``, and the application is responsible for drawing them (see ``DrawBitmapText()`` in
  :ref:`render.cc <saRender>`).

.. _reAuthoring:

Authoring scenes
^^^^^^^^^^^^^^^^

.. _rePbr:

The :ref:`classic renderer<ClassicRendering>` implements a traditional Phong lighting model: surface color is a sum of
ambient, diffuse, specular and emissive terms, and light intensities are dimensionless values tuned by eye. Modern
physically-based rendering (PBR) engines describe surfaces with empirically measurable properties and lights with
photometric units. Because material and lighting parameters are physical, assets retain their appearance across scenes
and lighting conditions, and effects like `image-based lighting
<https://google.github.io/filament/Filament.html#lighting/imagebasedlights>`__ and realistic metals become possible. For
an introduction to the underlying theory, see Filament's `PBR documentation
<https://google.github.io/filament/Filament.html>`__.

The Filament renderer supports both models. In general, we recommend that scenes are authored with modern PBR
(e.g. metallic-roughness) methods to produce the best results. However, we assume that many existing models are authored
for the classic renderer and so we default to using Filament's PBR-based approximation of Phong shading. This fallback
mode has been tuned to produce reasonable results, but the final image can vary significantly compared to the classic
renderer. The two decisions, how to shade materials and how to light the scene, are made independently and are described
below.

.. _reMaterials:

Materials
'''''''''

A material is shaded with the PBR lighting model if it defines any physical property: a non-negative
:ref:`metallic<asset-material-metallic>` or :ref:`roughness<asset-material-roughness>` coefficient, or texture
:ref:`layers<material-layer>` with PBR roles (metallic, roughness, packed occlusion-roughness-metallic, normal,
emissive, occlusion). Materials with no physical properties are assumed to be designed with the classic renderer in mind
and are shaded with a Phong lighting model, whose classic parameters are rescaled by the following
:ref:`custom flags<reFlags>`:

.. list-table::
   :widths: 32 10 12 46
   :header-rows: 1

   * - Name (``filament.<name>``)
     - Type
     - Default
     - Description
   * - ``phong.specular_multiplier``
     - real
     - 0.2
     - Multiplier applied to legacy (Phong) material :ref:`specular<asset-material-specular>`.
   * - ``phong.shininess_multiplier``
     - real
     - 0.1
     - Multiplier applied to legacy (Phong) material :ref:`shininess<asset-material-shininess>`.
   * - ``phong.emissive_multiplier``
     - real
     - 0.3
     - Multiplier applied to legacy (Phong) material :ref:`emission<asset-material-emission>`.

.. _reLighting:

Lighting
''''''''

Physical lighting is enabled by the lights themselves: if any light in the model has positive
:ref:`intensity<body-light-intensity>`, all lights are assumed to be photometric, with intensities measured in
`candela <https://en.wikipedia.org/wiki/Candela>`__. Lights of :ref:`type<body-light-type>` ``image`` provide
image-based lighting: the referenced :ref:`texture<body-light-texture>` illuminates the scene as an environment map. A
light's :ref:`bulbradius<body-light-bulbradius>` controls the softness of its shadows. In this mode the headlight is
disabled and no environment light is added automatically, so a model with physical lights but no ``image`` light has no
ambient illumination.

If no light has positive intensity, the model is treated as authored for the classic renderer: a default environment
light is added, the headlight is enabled, and a default total intensity is distributed equally among the model's lights.
These defaults are set by the following :ref:`custom flags<reFlags>`:

.. list-table::
   :widths: 32 10 12 46
   :header-rows: 1

   * - Name (``filament.<name>``)
     - Type
     - Default
     - Description
   * - ``fallback.environment_light_intensity``
     - real
     - 5000
     - Intensity of the fallback environment light.
   * - ``fallback.scene_light_intensity``
     - real
     - 80000
     - Total intensity distributed among the model's lights.
   * - ``fallback.head_light_intensity``
     - real
     - 40000
     - Intensity of the :ref:`headlight<visual-headlight>`, in lux.

.. _reFlags:

Rendering options
^^^^^^^^^^^^^^^^^

The Filament renderer provides a set of options for controlling how the final image is rendered. While some of these
options can be set directly on the scene elements as described above (e.g. setting the PBR properties of the material or
the intensities of the lights), some features are controlled by :ref:`custom<custom>` model elements (e.g.
:ref:`numeric<custom-numeric>` or :ref:`text<custom-text>`) whose names begin with ``filament.`` As the feature set
stabilizes, important flags will be promoted to first-class MJCF attributes.

.. code-block:: xml

   <custom>
     <numeric name="filament.clear_color" data="0.5 0.6 0.7 1"/>
   </custom>

Most flags correspond directly to Filament's `rendering options
<https://github.com/google/filament/blob/main/filament/include/filament/Options.h>`__ and are read by
:ref:`mjrf_configureSceneFromModel`. Boolean flags are numerics taking values 0 or 1. Enum flags (e.g.
``filament.ao.quality``) take string values that correspond to the underlying enum's name identifier. Flags not
specified in the model use the defaults listed below. All flag names begin with ``filament.``

.. collapse:: Shadows

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``shadows.type``
        - string
        - "pcf"
        - `Shadow algorithm <https://github.com/google/filament/blob/main/filament/include/filament/Options.h>`__; "pcf", "vsm", or "pcss".
      * - ``shadows.map_size``
        - int
        - min( :ref:`shadowsize<visual-quality-shadowsize>`, 2048)
        - Shadow map resolution, in texels.

.. collapse:: Screen-space ambient occlusion

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``ao.enabled``
        - bool
        - 1 (true)
        - Enable screen-space ambient occlusion.
      * - ``ao.quality``
        - string
        - "ultra"
        - Sampling quality; "low", "medium", "high", or "ultra".
      * - ``ao.low_pass_filter``
        - string
        - "ultra"
        - Quality of the depth-aware blur filter; "low", "medium", "high", or "ultra".
      * - ``ao.upsampling``
        - string
        - "ultra"
        - Quality of the occlusion buffer upsampling; "low", "medium", "high", or "ultra".
      * - ``ao.bent_normals``
        - bool
        - 0 (false)
        - Compute bent normals for specular occlusion.
      * - ``ao.ssct``
        - bool
        - 0 (false)
        - Enable screen-space cone tracing.
      * - ``ao.bilateral_threshold``
        - real
        - 0.5
        - Depth difference treated as an edge by the blur filter.

.. collapse:: Bloom

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``bloom.enabled``
        - bool
        - 0 (false)
        - Enable bloom.
      * - ``bloom.strength``
        - real
        - [filament default]
        - Strength of the bloom effect, between 0 and 1.
      * - ``bloom.dirt_strength``
        - real
        - [filament default]
        - Strength of the lens-dirt effect.
      * - ``bloom.quality``
        - string
        - "low"
        - Quality of the bloom passes; "low", "medium", "high", or "ultra".
      * - ``bloom.resolution``
        - int
        - [filament default]
        - Resolution of the bloom's minor axis, in pixels.
      * - ``bloom.levels``
        - int
        - [filament default]
        - Number of blur levels.

.. collapse:: Color grading

   See Filament's `ColorGrading.h
   <https://github.com/google/filament/blob/main/filament/include/filament/ColorGrading.h>`__.

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``cg.exposure``
        - real
        - 0
        - Exposure adjustment, in stops.
      * - ``cg.temperature``
        - real
        - 0
        - White balance temperature, between -1 (cool) and 1 (warm).
      * - ``cg.tint``
        - real
        - 0
        - White balance tint, between -1 (green) and 1 (magenta).
      * - ``cg.slope``
        - real(3)
        - 1 1 1
        - `ASC CDL <https://en.wikipedia.org/wiki/ASC_CDL>`__ slope.
      * - ``cg.offset``
        - real(3)
        - 0 0 0
        - `ASC CDL <https://en.wikipedia.org/wiki/ASC_CDL>`__ offset.
      * - ``cg.power``
        - real(3)
        - 1 1 1
        - `ASC CDL <https://en.wikipedia.org/wiki/ASC_CDL>`__ power.
      * - ``cg.shadows``
        - real(4)
        - 1 1 1 1
        - Shadow color shift; the fourth component shifts luminance.
      * - ``cg.midtones``
        - real(4)
        - 1 1 1 1
        - Midtone color shift; the fourth component shifts luminance.
      * - ``cg.highlights``
        - real(4)
        - 1 1 1 1
        - Highlight color shift; the fourth component shifts luminance.
      * - ``cg.tonal_ranges``
        - real(4)
        - 0 0.33 0.55 1
        - Boundaries of the shadow and highlight tonal ranges.
      * - ``cg.shadow_gamma``
        - real(3)
        - 1 1 1
        - Gamma adjustment of shadows.
      * - ``cg.mid_point``
        - real(3)
        - 1 1 1
        - Boundary between shadows and highlights.
      * - ``cg.highlight_scale``
        - real(3)
        - 1 1 1
        - Scale of highlights.
      * - ``cg.contrast``
        - real
        - 1
        - Contrast, between 0 and 2.
      * - ``cg.vibrance``
        - real
        - 1
        - Vibrance, between 0 and 2.
      * - ``cg.saturation``
        - real
        - 1
        - Saturation, between 0 and 2.
      * - ``cg.tone_mapping``
        - string
        - "pbr_neutral"
        - Tone mapping operator; "aces", "aces_legacy", "filmic", "linear", or "pbr_neutral".
      * - ``cg.output_color_space``
        - string
        - "srgb"
        - Output color space; "linear" or "srgb".
      * - ``cg.luminance_scaling``
        - bool
        - 0
        - Scale luminance to desaturate very bright highlights.
      * - ``cg.gamut_mapping``
        - bool
        - 0
        - Map out-of-gamut colors into the output gamut.

.. collapse:: Exponential height fog

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``fog.enabled``
        - bool
        - 0 (false)
        - Enable fog.
      * - ``fog.color``
        - real(3)
        - [filament default]
        - Fog color.
      * - ``fog.distance``
        - real
        - [filament default]
        - Distance from the camera at which the fog starts.
      * - ``fog.density``
        - real
        - [filament default]
        - Fog density.
      * - ``fog.cut_off_distance``
        - real
        - [filament default]
        - Distance beyond which fog is not applied.
      * - ``fog.maximum_opacity``
        - real
        - [filament default]
        - Maximum opacity of the fog, between 0 and 1.
      * - ``fog.height``
        - real
        - [filament default]
        - Height above which the fog density decreases.
      * - ``fog.height_falloff``
        - real
        - [filament default]
        - Falloff of fog density with height.
      * - ``fog.in_scattering_start``
        - real
        - [filament default]
        - Distance at which light in-scattering starts.
      * - ``fog.in_scattering_size``
        - real
        - [filament default]
        - Size of the light in-scattering halo (>0 to activate).

.. collapse:: Vignette

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``vignette.enabled``
        - bool
        - 0 (false)
        - Enable vignette.
      * - ``vignette.midpoint``
        - real
        - [filament default]
        - How close to the corners the vignette is restricted, between 0 and 1.

.. collapse:: Material

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``material.base_color_type``
        - string
        - "srgb"
        - How to interpret the color of a material; "linear" or "srgb".

.. collapse:: Background

   .. list-table::
      :widths: 32 10 12 46
      :header-rows: 1

      * - Name (``filament.<name>``)
        - Type
        - Default
        - Description
      * - ``clear_color``
        - real(4)
        - 0 0 0 1
        - Background color, RGBA (see the note below).

   The ``filament.clear_color`` flag is not read by the renderer library itself, because the background is a property
   of the renderer rather than of a scene: multiple scenes can, for example, be rendered to the same target to display
   them simultaneously. Applications that render a single scene, such as MuJoCo Studio and the ``mjr`` compatibility
   mode, read this flag from the model and apply it with :ref:`mjrf_setClearColor`. If you are managing your own render
   targets, you are responsible for setting the clear color yourself.

.. _ClassicRendering:

OpenGL rendering
~~~~~~~~~~~~~~~~

The classic OpenGL renderer (``mjr``) takes the :ref:`mjvScene` data structure populated in the
:ref:`Visualization <Visualization>` stage and renders it. It also provides basic 2D drawing and framebuffer access, so
that most applications do not need to call OpenGL directly. It is in maintenance mode; new applications should prefer
the :ref:`Filament renderer <FilamentRendering>`.

Below is a mixture of C code and pseudo-code in comments, illustrating the structure of an application that combines
simulation, abstract visualization, and classic OpenGL rendering. This is a short version of the :ref:`basic.cc
<saBasic>` code sample using GLFW.

.. code-block:: C

   // MuJoCo data structures
   mjModel* m = NULL;                  // MuJoCo model
   mjData* d = NULL;                   // MuJoCo data
   mjvCamera cam;                      // abstract camera
   mjvOption opt;                      // visualization options
   mjvScene scn;                       // abstract scene
   mjrContext con;                     // custom GPU context

   // ... load model and data

   // init GLFW, create window, make OpenGL context current, request v-sync
   glfwInit();
   GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
   glfwMakeContextCurrent(window);
   glfwSwapInterval(1);

   // initialize visualization data structures
   mjv_defaultCamera(&cam);
   mjv_defaultPerturb(&pert);
   mjv_defaultOption(&opt);
   mjr_defaultContext(&con);

   // create scene and context
   mjv_makeScene(m, &scn, 1000);
   mjr_makeContext(m, &con, mjFONTSCALE_100);

   // ... install GLFW keyboard and mouse callbacks

   // run main loop, target real-time simulation and 60 fps rendering
   while( !glfwWindowShouldClose(window) ) {
     // advance interactive simulation for 1/60 sec
     //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
     //  this loop will finish on time for the next frame to be rendered at 60 fps.
     //  Otherwise add a cpu timer and exit this loop when it is time to render.
     mjtNum simstart = d->time;
     while( d->time - simstart < 1.0/60.0 )
         mj_step(m, d);

     // get framebuffer viewport
     mjrRect viewport = {0, 0, 0, 0};
     glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

     // update scene and render
     mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
     mjr_render(viewport, &scn, &con);

     // swap OpenGL buffers (blocking call due to v-sync)
     glfwSwapBuffers(window);

     // process pending GUI events, call GLFW callbacks
     glfwPollEvents();
   }

   // close GLFW, free visualization storage
   glfwTerminate();
   mjv_freeScene(&scn);
   mjr_freeContext(&con);

   // ... free MuJoCo model and data

.. _reOpenGL:

Using OpenGL
^^^^^^^^^^^^

MuJoCo uses OpenGL 1.5 in the compatibility profile with the ``ARB_framebuffer_object`` and ``ARB_vertex_buffer_object``
extensions. OpenGL symbols are loaded via `GLAD <https://github.com/Dav1dde/glad>`_ the first time the
:ref:`mjr_makeContext` function is called. This means that the MuJoCo library itself does not have an explicit
dependency on OpenGL and can be used on systems without OpenGL support, as long as ``mjr_`` functions are not called.

Applications that use MuJoCo's built-in rendering functionalities are responsible for linking against an appropriate
OpenGL context creation library and for ensuring that there is an OpenGL context that is made current on the running
thread. On Windows and macOS, there is a canonical OpenGL library provided by the operating system. On Linux, MuJoCo
currently supports GLX for rendering to an X11 window, OSMesa for headless software rendering, and EGL for hardware
accelerated headless rendering.


.. _reContext:

Context and GPU resources
^^^^^^^^^^^^^^^^^^^^^^^^^

The first step in the rendering process is to create the model-specific GPU context :ref:`mjrContext`. This is done by
first clearing the data structure with the function :ref:`mjr_defaultContext`, and then calling the function
:ref:`mjr_makeContext`. This was already illustrated earlier; the relevant code is:

.. code-block:: C

   mjModel* m;
   mjrContext con;

   // clear mjrContext only once before first use
   mjr_defaultContext(&con);

   // create window with OpenGL context, make it current
   GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
   glfwMakeContextCurrent(window);

   // ... load MuJoCo model

   // make model-specific mjrContext
   mjr_makeContext(m, &con, mjFONTSCALE_100);

   // ... load another MuJoCo model

   // make mjrContext for new model (old context freed automatically)
   mjr_makeContext(m, &con, mjFONTSCALE_100);

   // free context when done
   mjr_freeContext(&con);

How is mjrContext related to an OpenGL context? An OpenGL context is what enables the application to talk to the video
driver and send rendering commands. It must exist and must be current in the calling thread before mjr_makeContext is
called. GLFW and related libraries provide the necessary functions as shown above.

mjrContext is specific to MuJoCo. After creation, it contains references (called "names" in OpenGL) to all the
resources that were uploaded to the GPU by mjr_makeContext. These include model-specific resources such as meshes and
textures, as well as generic resources such as font bitmaps for the specified font scale, framebuffer objects for
shadow mapping and offscreen rendering, and associated renderbuffers. It also contains OpenGL-related options copied
from ``mjModel.vis``, capabilities of the default window framebuffer that are discovered automatically, and the
currently active buffer for rendering; see :ref:`buffers <reBuffer>` below. Note that even though MuJoCo uses
fixed-function OpenGL, it avoids immediate mode rendering and instead uploads all resources to the GPU upfront. This
makes it as efficient as a modern shader, and possibly more efficient, because fixed-function OpenGL is now implemented
via internal shaders that have been written by the video driver developers and tuned extensively.

Most of the fields of mjrContext remain constant after the call to mjr_makeContext. The only exception is
mjrContext.currentBuffer which changes whenever the active buffer changes. Some of the GPU resources may also change
because the user can upload modified resources with the functions :ref:`mjr_uploadTexture`, :ref:`mjr_uploadMesh`,
:ref:`mjr_uploadHField`. This can be used to achieve dynamic effects such as inserting a video feed into the
rendering, or modulating a terrain map. Such modifications affect the resources residing on the GPU, but their OpenGL
names are reused; thus, the change is not actually visible in mjrContext.

The user should **never** make changes to mjrContext directly. MuJoCo's renderer assumes that only it can manage
mjrContext. In fact this kind of object would normally be opaque and its internal structure would not be exposed to
the user. We are exposing it because MuJoCo has an open design and also because users may want to interleave their own
OpenGL code with MuJoCo's renderer, in which case they may need read access to some fields of mjrContext. For example
in VR applications the user needs to blit from MuJoCo's offscreen buffer to a texture provided by a VR SDK.

When a different MuJoCo model is loaded, mjr_makeContext must be called again. There is also the function
:ref:`mjr_freeContext` which frees the GPU resources while preserving the initialization and capabilities flags. This
function should be called when the application is about to exit. It is called automatically from within
mjr_makeContext, so you do not need to call it directly when a different model is loaded, although it is not an error
to do so. The function mjr_defaultContext must be called once before rendering starts, to clear the memory allocated
for the data structure mjrContext. If you call it after calling mjr_makeContext, it will wipe out any record that GPU
resources were allocated without freeing those resources, so don't do that.

.. _reBuffer:

Buffers for rendering
^^^^^^^^^^^^^^^^^^^^^

In addition to the default window framebuffer, OpenGL can support unlimited framebuffer objects (FBOs) for custom
rendering. In MuJoCo we provide systematic support for two framebuffers: the default window framebuffer, and one
offscreen framebuffer. They are referred to by the constants in the enum type :ref:`mjtFramebuffer`, namely
mjFB_WINDOW and mjFB_OFFSCREEN. At any time, one of these two buffers is active for the purposes of MuJoCo rendering,
meaning that all subsequent commands are directed to it. There are two additional framebuffer objects referenced in
mjrContext, needed for shadow mapping and resolving multi-sample buffers, but these are used internally and the user
should not attempt to access them directly.

The active buffer is set with the function :ref:`mjr_setBuffer`. This sets the value of mjrContext.activeBuffer and
configures the OpenGL state accordingly. When mjr_makeContext is called, internally it calls mjr_setBuffer with
argument mjFB_WINDOW, so that rendering starts in the window buffer by default. If the specified buffer does not
exist, mjr_setBuffer automatically defaults to the other buffer (note that when using headless rendering on Linux,
there may be no window framebuffer).

From the perspective of OpenGL, there are important differences between the window framebuffer and offscreen
framebuffer, and these differences affect how the MuJoCo user interacts with the renderer. The window framebuffer is
created and managed by the operating system and not by OpenGL. As a result, properties such as resolution,
double-buffering, quad-buffered stereo, multi-samples, v-sync are set outside OpenGL; this is done by GLFW calls in our
code samples. All OpenGL can do is detect these properties; we do this in mjr_makeContext and record the results in the
various window capabilities fields of mjrContext. This is why such properties are not part of the MuJoCo model; they are
session/software-specific and not model-specific. In contrast, the offscreen framebuffer is managed entirely by OpenGL,
and so we can create that buffer with whatever properties we want, namely with the resolution and multi-sample
properties specified in ``mjModel.vis``.

The user can directly access the pixels in the two buffers. This is done with the functions :ref:`mjr_readPixels`,
:ref:`mjr_drawPixels` and :ref:`mjr_blitBuffer`. Read/draw transfer pixels from/to the active buffer to/from the CPU.
Blit transfers pixels between the two buffers on the GPU and is therefore much faster. The direction is from the
active buffer to the buffer that is not active. Note that mjr_blitBuffer has source and destination viewports that can
have different size, allowing the image to be scaled in the process.

.. _reDraw:

Drawing pixels
^^^^^^^^^^^^^^

The main rendering function is :ref:`mjr_render`. Its arguments are a rectangular viewport for rendering, the mjvScene
which was populated by the abstract visualizer, and the mjrContext which was created by mjr_makeContext. The viewport
can be the entire active buffer, or part of it for custom effects. A viewport corresponding to the entire buffer can
be obtained with the function :ref:`mjr_maxViewport`. Note that while the offscreen buffer size does not change, the
window buffer size changes whenever the user resizes or maximizes the window. Therefore user code should not assume
fixed viewport size. In the code sample :ref:`simulate.cc <saSimulate>` we use a callback which is triggered whenever
the window size changes, while in :ref:`basic.cc <saBasic>` we simply check the window size every time we render. On
certain scaled displays (notably on MacOS) the window size and framebuffer size can be different. So if you are
getting the size with GLFW functions, use glfwGetFramebufferSize rather than glfwGetWindowSize. On the other hand,
mouse coordinates are returned by the operating system in window rather than framebuffer units; thus the
:ref:`mouse interaction functions <viCamera>` should use glfwGetWindowSize to obtain the window height needed to
normalize the mouse displacement data.

mjr_render renders all mjvGeoms from the list mjvScene.geom. The abstract visualization options mjvOption are no
longer relevant here; they are used by mjv_updateScene to determine which geoms to add, and as far as mjr_render is
concerned these options are already baked-in. There is however another set of rendering options that are embedded in
mjvScene, and these affect the OpenGL rendering process. The array mjvScene.flags contains flags indexed by the enum
type :ref:`mjtRndFlag` and include options for enabling and disabling wireframe mode, shadows, reflections, skyboxes
and fog. Shadows and reflections involve additional rendering passes. MuJoCo's renderer is very efficient, but
depending on the model complexity and available GPU, it may be necessary to disable one or both of these effects in
some cases.

The parameter mjvScene.stereo determines the stereo mode. The possible values are given by the enum type
:ref:`mjtStereo` and are as follows:

mjSTEREO_NONE
   Stereo rendering is disabled. The average of the two OpenGL cameras in mjvScene is used. Note that the renderer
   always expects both cameras to be properly defined, even if stereo is not used.
mjSTEREO_QUADBUFFERED
   This mode works only when the active buffer is the window, and the window supports quad-buffered OpenGL. This
   requires a professional video card. The code sample :ref:`simulate.cc <saSimulate>` attempts to open such a window.
   In this mode MuJoCo's renderer uses the GL_BACK_LEFT and GL_BACK_RIGHT buffers to render the two views (as determined
   by the two OpenGL cameras in mjvScene) when the window is double-buffered, and GL_FRONT_LEFT and GL_FRONT_RIGHT
   otherwise. If the window does not support quad-buffered OpenGL or the active buffer is the offscreen buffer, the
   renderer reverts to the side-by-side mode described next.
mjSTEREO_SIDEBYSIDE
   This stereo mode does not require special hardware and is always available. The viewport given to mjr_render is split
   in two equal rectangles side-by-side. The left view is rendered on the left side and the right view on the right
   side. In principle users can cross their eyes and see stereo on a regular monitor, but the goal here is to show it in
   a stereoscopic device. Most head-mounted displays support this stereo mode.

In addition to the main mjr_render function, we provide several functions for "decorating" the image. These are 2D
rendering functions and include :ref:`mjr_overlay`, :ref:`mjr_text`, :ref:`mjr_rectangle`, and :ref:`mjr_figure`. The
user can draw additional decorations with their own OpenGL code. This should be done after mjr_render, because
mjr_render clears the viewport.

We also provide the functions :ref:`mjr_finish` and :ref:`mjr_getError` for explicit synchronization with the GPU and
for OpenGL error checking. They simply call glFinish and glGetError internally. This together with the basic 2d
drawing functions above is meant to provide enough functionality so that most users will not need to write OpenGL
code. Of course we cannot achieve this in all cases, short of providing wrappers for all of OpenGL.
