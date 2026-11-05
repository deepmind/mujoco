// Copyright 2025 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MUJOCO_MJRFILAMENT_H_
#define MUJOCO_MJRFILAMENT_H_

#include <stdint.h>

#include <mujoco/mjmodel.h>
#include <mujoco/mjrender.h>
#include <mujoco/mjtype.h>

#if defined(__cplusplus)
extern "C" {
#endif

//---------------------------------- Filament rendering --------------------------------------------

// Opaque handles for each of our components.
typedef struct mjrfContext_ mjrfContext;
typedef struct mjrfTexture_ mjrfTexture;
typedef struct mjrfMesh_ mjrfMesh;
typedef struct mjrfScene_ mjrfScene;
typedef struct mjrfLight_ mjrfLight;
typedef struct mjrfRenderable_ mjrfRenderable;
typedef struct mjrfRenderTarget_ mjrfRenderTarget;

// Callback function type for rendering operations.
typedef void (*mjrfCallback)(void* user_data);

// Underlying graphics API to use for rendering.
typedef enum mjrGraphicsApi_ {
  mjGRAPHICS_API_DEFAULT = 0,   // default (platform-dependent)
  mjGRAPHICS_API_OPENGL,        // desktop, mobile (GLES), web (WebGL)
  mjGRAPHICS_API_VULKAN,        // vulkan
} mjrGraphicsApi;

// Parameters for creating filament graphics context (mjrfContext).
typedef struct mjrfContextConfig_ {
  int graphics_api;                  // rendering graphics API [mjrGraphicsApi]
  mjtBool software_rendering;        // force backend to use software rendering
  void* native_window;               // platform-dependent window handle (or nullptr for windowless)
} mjrfContextConfig;

// Initializes the mjrfContextConfig to default values.
void mjrf_defaultContextConfig(mjrfContextConfig* config);

// Creates a filament rendering context.
mjrfContext* mjrf_createContext(const mjrfContextConfig* config);

// Destroys the filament rendering context.
void mjrf_destroyContext(mjrfContext* ctx);

// Gets active renderer information for the given filament context.
void mjrf_getRendererInfo(mjrfContext* ctx, mjrRendererInfo* info);

// High-level control for how to draw objects in the scene.
typedef enum mjrDrawMode_ {
  mjDRAW_MODE_DEFAULT,                // default colors and lighting
  mjDRAW_MODE_DEFAULT_NO_TEXTURES,    // default, but without textures
  mjDRAW_MODE_WIREFRAME,              // wireframe rendering
  mjDRAW_MODE_DEPTH,                  // grayscale depth map
  mjDRAW_MODE_ISLANDS,                // color objects based on island and sleep state
  mjDRAW_MODE_SEGMENTATION_BY_ID,     // color objects based on segmentation id
  mjDRAW_MODE_SEGMENTATION_BY_COLOR,  // generate visually distinct colors using segmentation id
} mjrDrawMode;

// A single rendering operation.
typedef struct mjrfRenderRequest_ {
  mjrfScene* scene;                  // scene to render
  mjrCamera camera;                  // camera (viewpoint) from which to render scene
  mjrRect viewport;                  // viewport (rect area) into which to render
  mjrfRenderTarget* target;          // target used for rendering (or nullptr for window rendering)
  int draw_mode;                     // method to use for drawing objects [mjrDrawMode]
  mjtBool post_processing;           // enable post processing, enabled by default
  mjtBool reflections;               // enable reflections, enabled by default
  mjtBool shadows;                   // enable shadows, enabled by default
  mjtBool skybox;                    // enable skybox rendering, enabled by default
  mjtBool haze;                      // enable horizon haze, enabled by default

  // The headlight is a directional light aligned with this request's camera. It
  // is a property of the request rather than of the scene, so that a scene
  // rendered from several cameras is not lit by any one of them.
  mjtBool headlight;                 // enable the headlight, disabled by default
  float headlight_color[3];          // headlight color, RGB
  float headlight_intensity;         // headlight intensity, in lux
} mjrfRenderRequest;

// Initializes the mjrfRenderRequest to default values.
void mjrf_defaultRenderRequest(mjrfRenderRequest* request);

// A single pixel read operation.
typedef struct mjrfReadPixelsRequest_ {
  mjrfRenderTarget* target;              // render target from which to read the image pixels
  void* output;                          // buffer into which the pixels will be stored
  mjtSize num_bytes;                     // size of output buffer
  mjrfCallback read_completed;           // callback when read is complete; can use to free output
  void* user_data;                       // user data for read_completed_callback
} mjrfReadRequest;

// Initializes the mjrfReadRequest to default values.
void mjrf_defaultReadRequest(mjrfReadRequest* request);

// Unique handle assigned to each render request; used to block until request is completed or query
// the status of the request.
typedef uint64_t mjrfFrameHandle;

// Submits the given requests for rendering. Because rendering happens asynchronously, callers have
// to submit both the render and read requests in the same call. Multiple requests and reads can be
// submitted in a single call. These requests will be processed in order, so some care must be
// taken. Firstly, requests should be grouped by target. Next, the combined area of the viewports
// for all requests for a given target must be contained within the dimensions of the target itself.
//
// Callbacks will be invoked from within this function, though there is no guarantee on which
// invocation of this function it will be done.
mjrfFrameHandle mjrf_render(mjrfContext* ctx, const mjrfRenderRequest* req, int nreq,
                            const mjrfReadRequest* read_req, int nread_req);

// Waits for all rendering operations to complete for the given frame handle, triggering any
// callbacks as needed.
void mjrf_waitForFrame(mjrfContext* ctx, mjrfFrameHandle frame);

// Flushes pending operations to the rendering backend and invokes any completed callbacks.
void mjrf_flush(mjrfContext* ctx);

// Sets the clear color for the renderer.
void mjrf_setClearColor(mjrfContext* ctx, const float color[3]);

// Information for a single frame of rendering.
typedef struct mjrfFrameStats_ {
  double frame_rate;              // frame rate, in frames per second
} mjrfFrameStats;

// Initializes the mjrFrameStats to default values.
void mjrf_defaultFrameStats(mjrfFrameStats* stats);

// Returns the stats for the given frame but updating the given `stats_out`.
void mjrf_getFrameStats(mjrfContext* ctx, mjrfFrameHandle frame, mjrfFrameStats* stats_out);

// Parameters for creating a texture (mjrfTexture).
typedef struct mjrfTextureConfig_ {
  int width;                         // texture width, or number of bytes for compressed data (e.g. KTX)
  int height;                        // texture height, or 0 for compressed data (e.g. KTX)
  int format;                        // pixel format (e.g. RGB8, RGBA8, KTX, etc.) [mjrPixelFormat]
  int color_space;                   // color space (e.g. LINEAR, sRGB, etc.) [mjrColorSpace]
  int sampler_type;                  // texture sampler (e.g. 2D, cube, etc.) [mjrSamplerType]
} mjrfTextureConfig;

// Initializes the mjrfTextureConfig to default values.
void mjrf_defaultTextureConfig(mjrfTextureConfig* config);

// Creates a filament texture. Note that the texture will not be created on the GPU until
// `mjrf_uploadTexture()` is called.
mjrfTexture* mjrf_createTexture(mjrfContext* ctx, const mjrfTextureConfig* config);

// Destroys the texture.
void mjrf_destroyTexture(mjrfTexture* texture);

// Binary data payload for an mjrfTexture.
typedef struct mjrfTextureData_ {
  const void* bytes;               // pointer to image data, or nullptr for empty texture
  mjtSize num_bytes;               // number of bytes in the image data
  mjrfCallback release;            // callback when data has finished uploading
  void* user_data;                 // user data for release callback
} mjrfTextureData;

// Initializes the mjrfTextureData to default values.
void mjrf_defaultTextureData(mjrfTextureData* data);

// Uploads the given texture data to the texture.
void mjrf_uploadTexture(mjrfTexture* texture, const mjrfTextureData* data);

// Returns the width of the texture.
int mjrf_getTextureWidth(const mjrfTexture* texture);

// Returns the height of the texture.
int mjrf_getTextureHeight(const mjrfTexture* texture);

// Returns the sampler type (mjrSamplerType) used by the texture.
// [returns: mjrSamplerType]
int mjrf_getTextureSamplerType(const mjrfTexture* texture);

// Maximum number of vertex attributes in a mesh.
enum { mjMAX_VERTEX_ATTRIBUTES = 16 };

// Parameters describing a mesh (mjrfMesh).
typedef struct mjrfMeshConfig_ {
  mjtSize max_vertices;         // maximum number of vertices
  mjtSize max_indices;          // maximum number of indices
  int num_attributes;           // number of defined attributes
  mjrVertexAttribute attributes[mjMAX_VERTEX_ATTRIBUTES];  // per-vertex attribute information
  mjtBool interleaved;          // true if vertex attributes are interleaved
  int index_type;               // index data format (e.g. UINT16 or UINT32) [mjrIndexType]
  int primitive_type;           // index interpretation (e.g. TRIANGLES, etc.) [mjrMeshPrimitiveType]
} mjrfMeshConfig;

// Initializes the mjrfMeshConfig to default values.
void mjrf_defaultMeshConfig(mjrfMeshConfig* config);

// Creates an empty mesh with the given config.
mjrfMesh* mjrf_createMesh(mjrfContext* ctx, const mjrfMeshConfig* config);

// Destroys the mesh.
void mjrf_destroyMesh(mjrfMesh* mesh);

// Binary data used for creating a mesh (mjrfMesh).
typedef struct mjrfMeshData_ {
  mjtSize num_vertices;         // number of vertices
  const void* vertices[mjMAX_VERTEX_ATTRIBUTES];  // per-vertex attribute data arrays
  mjtSize num_indices;          // number of indices
  const void* indices;          // indices data array
  mjtBool compute_bounds;       // if true, compute bounds from vertex positions
  float bounds_min[3];          // min/max bounds; assume unset if bounds_min == bounds_max
  float bounds_max[3];
  mjrfCallback release;         // callback when data has finished uploading
  void* user_data;              // user data for release callback
} mjrfMeshData;

// Initializes the mjrfMeshData to default values.
void mjrf_defaultMeshData(mjrfMeshData* data);

// Uploads the given mesh data to the mesh.
void mjrf_uploadMesh(mjrfMesh* mesh, const mjrfMeshData* data);

// Parameters for creating a scene (mjrfScene).
typedef struct mjrfSceneParams_ {
  char unused;  // ensure min size of 1 for C/C++ compatibility
} mjrfSceneParams;

// Initializes the mjrfSceneParams to default values.
void mjrf_defaultSceneParams(mjrfSceneParams* params);

// Creates a scene with the given parameters.
mjrfScene* mjrf_createScene(mjrfContext* ctx, const mjrfSceneParams* params);

// Destroys the scene.
void mjrf_destroyScene(mjrfScene* scene);

// Adds a light to the scene.
void mjrf_addSceneLight(mjrfScene* scene, mjrfLight* light);

// Removes the light from the scene.
void mjrf_removeSceneLight(mjrfScene* scene, mjrfLight* light);

// Adds a renderable to the scene.
void mjrf_addSceneRenderable(mjrfScene* scene, mjrfRenderable* renderable);

// Removes the renderable from the scene.
void mjrf_removeSceneRenderable(mjrfScene* scene, mjrfRenderable* renderable);

// Sets the skybox (cube texture) for the scene.
void mjrf_setSceneSkybox(mjrfScene* scene, const mjrfTexture* texture);

// Configures the scene based on the parameters in an mjModel.
void mjrf_configureSceneFromModel(mjrfScene* scene, const mjModel* model);

// Sets the number of multi-sample anti-aliasing samples; values <= 1 disable MSAA.
void mjrf_setSceneMsaa(mjrfScene* scene, int samples);

// Parameters for creating a light (mjrfLight).
typedef struct mjrfLightParams_ {
  int type;                        // type of light (e.g. spot, point, image, etc.) [mjrLightType]
  const mjrfTexture* texture;      // texture; only for image lights
  float color[3];                  // RGB color
  float intensity;                 // light intensity, in candela
  mjtBool cast_shadow;             // if true, cast shadows
  float range;                     // effective range of light, in meters
  float cutoff;                    // spot light cone angle, in degrees
  float softness;                  // spot light edge softness, fraction of cone angle in [0, 1]
  int shadow_map_size;             // size of shadow map texture, 0 to use default size
  float bulb_radius;               // bulb radius, used for soft shadows
} mjrfLightParams;

// Initializes the mjrfLightParams to default values.
void mjrf_defaultLightParams(mjrfLightParams* params);

// Creates a light for the filament renderer.
mjrfLight* mjrf_createLight(mjrfContext* ctx, const mjrfLightParams* params);

// Destroys the light.
void mjrf_destroyLight(mjrfLight* light);

// Activates or deactivates the light.
void mjrf_setLightActive(mjrfLight* light, mjtBool active);

// Enables or disables whether or not the light casts shadows.
void mjrf_setLightCastShadow(mjrfLight* light, mjtBool enabled);

// Sets the RGB color of the light.
void mjrf_setLightColor(mjrfLight* light, const float color[3]);

// Sets the intensity of the light, in candela.
void mjrf_setLightIntensity(mjrfLight* light, float intensity);

// Sets the effective range of the light, in meters.
void mjrf_setLightRange(mjrfLight* light, float range);

// Sets the cutoff angle of the light, in degrees. Only used for spot lights.
void mjrf_setLightCutoff(mjrfLight* light, float cutoff);

// Sets the softness of the light, in the range [0, 1]. Only used for spot lights.
void mjrf_setLightSoftness(mjrfLight* light, float softness);

// Sets the radius of the light bulb.
void mjrf_setLightBulbRadius(mjrfLight* light, float radius);

// Sets the resolution of the light's shadow map, in texels.
void mjrf_setLightShadowMapSize(mjrfLight* light, int map_size);

// Sets the position and direction of the light.
void mjrf_setLightPose(mjrfLight* light, const float position[3], const float direction[3]);

// Returns the type of the light (mjrLightType).
int mjrf_getLightType(const mjrfLight* light);

// Material properties for a renderable (mjrfRenderable).
typedef struct mjrfMaterial_ {
  float color[4];               // object color; defaults to white
  int32_t segmentation_id;      // ID for segmentation rendering; maps to RGB8 color (i.e. 24 bits)
  int32_t island_id;            // ID to which the renderable belongs
  int sleep_state;              // sleep state of the renderable [mjtSleepState]
  float uv_scale[3];            // scale applied to UV coordinates; defaults to (1,1,1)
  float uv_offset[3];           // offset applied to UV coordinates; defaults to (0,0,0)
  float scissor[4];             // if non-zero, applies scissor testing when rendering
  float metallic;               // metallic factory [0, 1]; disabled if < 0
  float roughness;              // roughness factor [0, 1]; disabled if < 0
  float specular;               // specular factor [0, 1]; disabled if < 0
  float glossiness;             // glossiness factor [0, 1]; disabled if < 0
  float emissive;               // emissive/glow factor [0, 1]; disabled if < 0
  float reflectance;            // blend factor for reflective surfaces [0, 1]; applies only to planes
  mjtBool decor;                // ux/decor element: unlit, drawn on the decor layer, no shadows
  mjtBool selected;             // for "selected" ux elements, adds additional styling
  const mjrfTexture* color_texture;       // color/albedo texture (RGB8)
  const mjrfTexture* opacity_texture;     // opacity texture (A8)
  const mjrfTexture* normal_texture;      // normal map texture (RGB8)
  const mjrfTexture* metallic_texture;    // metallic map texture (R8)
  const mjrfTexture* roughness_texture;   // roughness map texture (R8)
  const mjrfTexture* occlusion_texture;   // ambient occlusion texture (R8)
  const mjrfTexture* orm_texture;         // occlusion/roughness/metallic texture (RGB8)
  const mjrfTexture* emissive_texture;    // emissive texture (RGB8)
  const mjrfTexture* reflection_texture;  // reflection texture (internal)
  float reflection_normal[3];      // mirror normal, gates reflection to front face (internal)
  float reflection_view_proj[16];  // main camera view-proj for reflection UV mapping (internal)
  float haze_ratio;                // horizon haze ratio [0, 1]; disabled if < 0 (internal)
} mjrfMaterial;

// Initializes the mjrfMaterial to default values.
void mjrf_defaultMaterial(mjrfMaterial* material);

// Parameters for creating a renderable (mjrfRenderable).
typedef struct mjrfRenderableParams_ {
  mjtBool cast_shadow;                  // if true, casts shadows
  mjtBool receive_shadow;               // if true, receives shadows
  uint16_t blend_order;                 // controls draw order for transparent objects [0, 8]
} mjrfRenderableParams;

// Initializes the mjrfRenderableParams to default values.
void mjrf_defaultRenderableParams(mjrfRenderableParams* params);

// Creates a renderable with the given parameters.
mjrfRenderable* mjrf_createRenderable(mjrfContext* ctx, const mjrfRenderableParams* params);

// Destroys the renderable.
void mjrf_destroyRenderable(mjrfRenderable* renderable);

// Sets the mesh of the renderable.
void mjrf_setRenderableMesh(mjrfRenderable* renderable, const mjrfMesh* mesh, int elem_offset,
                            int elem_count);

// Sets the mesh of the renderable to a built-in mesh based on the geom type. Note: using the same
// parameters (nstack, nslice, nquad) will have better performance as the internal mesh data can be
// shared across renderables.
// [type: mjtGeom]
void mjrf_setRenderableGeomMesh(mjrfRenderable* renderable, int type, int nstack, int nslice,
                                int nquad);

// Sets the material properties and textures of the renderable.
void mjrf_setRenderableMaterial(mjrfRenderable* renderable, const mjrfMaterial* material);

// Copies the material properties of the renderable into the given mjrfMaterial.
void mjrf_getRenderableMaterial(mjrfRenderable* renderable, mjrfMaterial* material);

// Sets the position and rotation of the renderable.
void mjrf_setRenderablePose(mjrfRenderable* renderable, const float position[3],
                            const float rotation[9]);

// Sets the size of the renderable. Note that, for most renderables, this is equivalent to setting
// the scale. However, for some geom-based renderables, the size scale is not applied uniformly
// (e.g. the spherical ends of a capsule are scaled such that they always remain spherical).
void mjrf_setRenderableSize(mjrfRenderable* renderable, const float size[3]);

// Enables or disables whether or not the renderable casts shadows.
void mjrf_setRenderableCastShadow(mjrfRenderable* renderable, mjtBool enabled);

// Parameters for creating a render target (mjrfRenderTarget).
typedef struct mjrfRenderTargetConfig_ {
  int width;                              // texture width
  int height;                             // texture height
  int color_format;                       // pixel format for color buffer [mjrPixelFormat]
  int depth_format;                       // pixel format for depth buffer [mjrPixelFormat]
} mjrfRenderTargetConfig;

// Initializes the mjrfRenderTargetConfig to default values.
void mjrf_defaultRenderTargetConfig(mjrfRenderTargetConfig* config);

// Creates a render target for the filament renderer.
mjrfRenderTarget* mjrf_createRenderTarget(mjrfContext* ctx, const mjrfRenderTargetConfig* config);

// Destroys the render target.
void mjrf_destroyRenderTarget(mjrfRenderTarget* render_target);

// Resizes the render target to the given width and height.
void mjrf_resizeRenderTarget(mjrfRenderTarget* render_target, int width, int height);

#if defined(__cplusplus)
}  // extern "C"
#endif

#endif  // MUJOCO_MJRFILAMENT_H_
