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

#include <mujoco/mjrfilament.h>

#include <array>
#include <cstring>

#include <filament/Engine.h>
#include <math/mat3.h>
#include <math/vec3.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/filament_context.h"
#include "render/filament/core/light.h"
#include "render/filament/core/mesh.h"
#include "render/filament/core/render_target.h"
#include "render/filament/core/renderable.h"
#include "render/filament/core/scene_view.h"
#include "render/filament/core/texture.h"

template <int N>
static void setf(float (&arr)[N], const std::array<float, N>& values) {
  for (int i = 0; i < N; ++i) {
    arr[i] = values[i];
  }
}

static const char* BackendName(filament::Engine::Backend backend) {
  switch (backend) {
    case filament::Engine::Backend::OPENGL:
      return "opengl";
    case filament::Engine::Backend::VULKAN:
      return "vulkan";
    default:
      return "unknown";
  }
}

extern "C" {

void mjrf_defaultContextConfig(mjrfContextConfig* config) {
  memset(config, 0, sizeof(mjrfContextConfig));
}

void mjrf_defaultTextureData(mjrfTextureData* data) {
  memset(data, 0, sizeof(mjrfTextureData));
}

void mjrf_defaultTextureConfig(mjrfTextureConfig* config) {
  memset(config, 0, sizeof(mjrfTextureConfig));
}

void mjrf_defaultMeshConfig(mjrfMeshConfig* config) {
  memset(config, 0, sizeof(mjrfMeshConfig));
}

void mjrf_defaultMeshData(mjrfMeshData* data) {
  memset(data, 0, sizeof(mjrfMeshData));
}

void mjrf_defaultSceneParams(mjrfSceneParams* params) {
  memset(params, 0, sizeof(mjrfSceneParams));
}

void mjrf_defaultLightParams(mjrfLightParams* params) {
  memset(params, 0, sizeof(mjrfLightParams));
  params->type = mjLIGHT_POINT;
  params->texture = nullptr;
  params->color[0] = 0;
  params->color[1] = 0;
  params->color[2] = 0;
  params->intensity = 0.0f;
  params->cast_shadow = true;
  params->range = 10.0f;
  params->cutoff = 180.f;
  params->softness = 0.0f;
  params->bulb_radius = 0.0f;
  params->shadow_map_size = 2048;
}

void mjrf_defaultMaterial(mjrfMaterial* material) {
  memset(material, 0, sizeof(mjrfMaterial));
  setf(material->color, {1.f, 1.f, 1.f, 1.f});
  setf(material->uv_scale, {1, 1, 1});
  material->sleep_state = mjS_STATIC;
  material->island_id = -1;
  material->emissive = -1.0f;
  material->specular = -1.0f;
  material->glossiness = -1.0f;
  material->metallic = -1.0f;
  material->roughness = -1.0f;
  material->haze_ratio = -1.0f;
}

void mjrf_defaultRenderableParams(mjrfRenderableParams* params) {
  memset(params, 0, sizeof(mjrfRenderableParams));
  params->cast_shadow = true;
  params->receive_shadow = true;
  params->blend_order = 0;
}

void mjrf_defaultRenderTargetConfig(mjrfRenderTargetConfig* config) {
  memset(config, 0, sizeof(mjrfRenderTargetConfig));
  config->color_format = mjPIXEL_FORMAT_RGBA8;
  config->depth_format = mjPIXEL_FORMAT_DEPTH32F;
}

void mjrf_defaultRenderRequest(mjrfRenderRequest* request) {
  memset(request, 0, sizeof(mjrfRenderRequest));
  request->post_processing = true;
  request->reflections = true;
  request->shadows = true;
  request->skybox = true;
  request->haze = true;
}

void mjrf_defaultReadRequest(mjrfReadRequest* request) {
  memset(request, 0, sizeof(mjrfReadRequest));
}

void mjrf_defaultFrameStats(mjrfFrameStats* stats) {
  memset(stats, 0, sizeof(mjrfFrameStats));
}

mjrfContext* mjrf_createContext(const mjrfContextConfig* config) {
  return new mujoco::FilamentContext(config);
}

void mjrf_destroyContext(mjrfContext* ctx) {
  delete mujoco::FilamentContext::downcast(ctx);
}

void mjrf_getRendererInfo(mjrfContext* ctx, mjrRendererInfo* info) {
  memset(info, 0, sizeof(mjrRendererInfo));
  info->renderer = "filament";
  info->backend = ctx ? BackendName(mujoco::FilamentContext::downcast(ctx)->GetBackend()) : "";
}

mjrfTexture* mjrf_createTexture(mjrfContext* ctx,
                                const mjrfTextureConfig* config) {
  return new mujoco::Texture(
      mujoco::FilamentContext::downcast(ctx)->GetEngine(), *config);
}

void mjrf_destroyTexture(mjrfTexture* texture) {
  delete mujoco::Texture::downcast(texture);
}

mjrfMesh* mjrf_createMesh(mjrfContext* ctx, const mjrfMeshConfig* config) {
  return new mujoco::Mesh(mujoco::FilamentContext::downcast(ctx)->GetEngine(),
                          *config);
}

void mjrf_destroyMesh(mjrfMesh* mesh) { delete mujoco::Mesh::downcast(mesh); }

mjrfScene* mjrf_createScene(mjrfContext* ctx, const mjrfSceneParams* params) {
  return new mujoco::SceneView(
      mujoco::FilamentContext::downcast(ctx)->GetObjectManager(),
      mujoco::FilamentContext::downcast(ctx)->GetMaterialManager(), *params);
}

void mjrf_destroyScene(mjrfScene* scene) {
  delete mujoco::SceneView::downcast(scene);
}

mjrfLight* mjrf_createLight(mjrfContext* ctx, const mjrfLightParams* params) {
  return new mujoco::Light(mujoco::FilamentContext::downcast(ctx)->GetEngine(),
                           *params);
}

void mjrf_destroyLight(mjrfLight* light) {
  delete mujoco::Light::downcast(light);
}

mjrfRenderable* mjrf_createRenderable(mjrfContext* ctx,
                                      const mjrfRenderableParams* params) {
  return new mujoco::Renderable(
      mujoco::FilamentContext::downcast(ctx)->GetEngine(), *params,
      mujoco::FilamentContext::downcast(ctx)->GetMaterialManager());
}

void mjrf_destroyRenderable(mjrfRenderable* renderable) {
  delete mujoco::Renderable::downcast(renderable);
}

mjrfRenderTarget* mjrf_createRenderTarget(
    mjrfContext* ctx, const mjrfRenderTargetConfig* config) {
  return new mujoco::RenderTarget(
      mujoco::FilamentContext::downcast(ctx)->GetEngine(), *config);
}

void mjrf_destroyRenderTarget(mjrfRenderTarget* render_target) {
  delete mujoco::RenderTarget::downcast(render_target);
}

void mjrf_uploadTexture(mjrfTexture* texture, const mjrfTextureData* data) {
  mujoco::Texture::downcast(texture)->Upload(*data);
}

void mjrf_uploadMesh(mjrfMesh* mesh, const mjrfMeshData* data) {
  mujoco::Mesh::downcast(mesh)->Upload(*data);
}

int mjrf_getTextureWidth(const mjrfTexture* texture) {
  return mujoco::Texture::downcast(texture)->GetWidth();
}

int mjrf_getTextureHeight(const mjrfTexture* texture) {
  return mujoco::Texture::downcast(texture)->GetHeight();
}

int mjrf_getTextureSamplerType(const mjrfTexture* texture) {
  return mujoco::Texture::downcast(texture)->GetSamplerType();
}

void mjrf_setLightActive(mjrfLight* light, mjtBool active) {
  if (active) {
    mujoco::Light::downcast(light)->Enable();
  } else {
    mujoco::Light::downcast(light)->Disable();
  }
}

void mjrf_setLightCastShadow(mjrfLight* light, mjtBool enabled) {
  mujoco::Light::downcast(light)->SetShadowsEnabled(enabled);
}

void mjrf_setLightColor(mjrfLight* light, const float color[3]) {
  mujoco::Light::downcast(light)->SetColor({color[0], color[1], color[2]});
}

void mjrf_setLightIntensity(mjrfLight* light, float intensity) {
  mujoco::Light::downcast(light)->SetIntensity(intensity);
}

void mjrf_setLightRange(mjrfLight* light, float range) {
  mujoco::Light::downcast(light)->SetRange(range);
}

void mjrf_setLightCutoff(mjrfLight* light, float cutoff) {
  mujoco::Light::downcast(light)->SetCutoffAngle(cutoff);
}

void mjrf_setLightSoftness(mjrfLight* light, float softness) {
  mujoco::Light::downcast(light)->SetSoftness(softness);
}

void mjrf_setLightBulbRadius(mjrfLight* light, float radius) {
  mujoco::Light::downcast(light)->SetBulbRadius(radius);
}

void mjrf_setLightShadowMapSize(mjrfLight* light, int map_size) {
  mujoco::Light::downcast(light)->SetShadowMapSize(map_size);
}

void mjrf_setLightPose(mjrfLight* light, const float position[3],
                            const float direction[3]) {
  mujoco::Light::downcast(light)->SetTransform(
      {position[0], position[1], position[2]},
      {direction[0], direction[1], direction[2]});
}

int mjrf_getLightType(const mjrfLight* light) {
  return mujoco::Light::downcast(light)->GetType();
}

void mjrf_setRenderableMesh(mjrfRenderable* renderable, const mjrfMesh* mesh,
                            int elem_offset, int elem_count) {
  mujoco::Renderable::downcast(renderable)
      ->SetMesh(mujoco::Mesh::downcast(mesh), elem_offset, elem_count);
}

void mjrf_setRenderableGeomMesh(mjrfRenderable* renderable, int type,
                                int nstack, int nslice, int nquad) {
  mujoco::Renderable::downcast(renderable)
      ->SetGeomMesh(static_cast<mjtGeom>(type), nstack, nslice, nquad);
}

void mjrf_setRenderableMaterial(mjrfRenderable* renderable,
                                const mjrfMaterial* material) {
  mujoco::Renderable::downcast(renderable)->UpdateMaterial(*material);
}

void mjrf_getRenderableMaterial(mjrfRenderable* renderable,
                                mjrfMaterial* material) {
  *material = mujoco::Renderable::downcast(renderable)->GetMaterial();
}

void mjrf_setRenderablePose(mjrfRenderable* renderable,
                            const float position[3],
                            const float rotation[9]) {
  const filament::math::float3 fposition{position[0], position[1], position[2]};
  const filament::math::mat3f frotation{rotation[0], rotation[3], rotation[6],
                                        rotation[1], rotation[4], rotation[7],
                                        rotation[2], rotation[5], rotation[8]};
  mujoco::Renderable::downcast(renderable)->SetTransform(fposition, frotation);
}

void mjrf_setRenderableSize(mjrfRenderable* renderable, const float size[3]) {
  const filament::math::float3 fsize{size[0], size[1], size[2]};
  mujoco::Renderable::downcast(renderable)->SetSize(fsize);
}

void mjrf_setRenderableCastShadow(mjrfRenderable* renderable, mjtBool enabled) {
  mujoco::Renderable::downcast(renderable)->SetCastShadows(enabled);
}

void mjrf_addSceneLight(mjrfScene* scene, mjrfLight* light) {
  mujoco::SceneView::downcast(scene)->AddToScene(
      mujoco::Light::downcast(light));
}

void mjrf_removeSceneLight(mjrfScene* scene, mjrfLight* light) {
  mujoco::SceneView::downcast(scene)->RemoveFromScene(
      mujoco::Light::downcast(light));
}

void mjrf_addSceneRenderable(mjrfScene* scene, mjrfRenderable* renderable) {
  mujoco::SceneView::downcast(scene)->AddToScene(
      mujoco::Renderable::downcast(renderable));
}

void mjrf_removeSceneRenderable(mjrfScene* scene,
                                    mjrfRenderable* renderable) {
  mujoco::SceneView::downcast(scene)->RemoveFromScene(
      mujoco::Renderable::downcast(renderable));
}

void mjrf_setSceneSkybox(mjrfScene* scene, const mjrfTexture* texture) {
  mujoco::SceneView::downcast(scene)->SetSkybox(
      mujoco::Texture::downcast(texture));
}

void mjrf_configureSceneFromModel(mjrfScene* scene, const mjModel* model) {
  mujoco::SceneView::downcast(scene)->Configure(model);
}

void mjrf_setSceneMsaa(mjrfScene* scene, int samples) {
  mujoco::SceneView::downcast(scene)->SetMsaa(samples);
}

void mjrf_resizeRenderTarget(mjrfRenderTarget* render_target, int width,
                             int height) {
  mujoco::RenderTarget::downcast(render_target)->Prepare(width, height);
}

mjrfFrameHandle mjrf_render(mjrfContext* ctx, const mjrfRenderRequest* req,
                            int nreq, const mjrfReadRequest* read_req,
                            int nread_req) {
  return mujoco::FilamentContext::downcast(ctx)->Render(
      {req, static_cast<size_t>(nreq)},
      {read_req, static_cast<size_t>(nread_req)});
}

void mjrf_waitForFrame(mjrfContext* ctx, mjrfFrameHandle frame) {
  mujoco::FilamentContext::downcast(ctx)->WaitForFrame(frame);
}

void mjrf_flush(mjrfContext* ctx) {
  mujoco::FilamentContext::downcast(ctx)->Flush();
}

void mjrf_setClearColor(mjrfContext* ctx, const float color[3]) {
  mujoco::FilamentContext::downcast(ctx)->SetClearColor(
      {color[0], color[1], color[2], 1.0f});
}

void mjrf_getFrameStats(mjrfContext* ctx, mjrfFrameHandle frame,
                        mjrfFrameStats* stats_out) {
  mujoco::FilamentContext::downcast(ctx)->GetFrameStats(frame, stats_out);
}
}  // extern "C"
