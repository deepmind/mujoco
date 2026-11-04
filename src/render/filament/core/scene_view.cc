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

#include "render/filament/core/scene_view.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <filament/Color.h>
#include <filament/ColorGrading.h>
#include <filament/ColorSpace.h>
#include <filament/LightManager.h>
#include <filament/Material.h>
#include <filament/Options.h>
#include <filament/RenderTarget.h>
#include <filament/Renderer.h>
#include <filament/Skybox.h>
#include <filament/TransformManager.h>
#include <filament/View.h>
#include <filament/Viewport.h>
#include <math/TVecHelpers.h>
#include <math/mat3.h>
#include <math/mat4.h>
#include <math/mathfwd.h>
#include <math/scalar.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <utils/EntityManager.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/core/builtins.h"
#include "render/filament/core/color_grading_options.h"
#include "render/filament/core/light.h"
#include "render/filament/core/material_manager.h"
#include "render/filament/core/object_manager.h"
#include "render/filament/core/outliner.h"
#include "render/filament/core/reflection_manager.h"
#include "render/filament/core/render_target.h"
#include "render/filament/core/renderable.h"
#include "render/filament/core/texture.h"
#include "render/filament/support/filament_util.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;
using filament::math::mat3f;
using filament::math::mat4;
using filament::math::mat4f;

static constexpr filament::color::ColorSpace REC709_LINEAR_D65 =
    filament::color::Rec709 - filament::color::Linear -
    filament::color::D65;
static constexpr filament::color::ColorSpace REC709_SRGB_D65 =
    filament::color::Rec709 - filament::color::sRGB -
    filament::color::D65;

static filament::ColorGrading::Builder ToBuilder(
    const ColorGradingOptions& opts) {
  filament::color::ColorSpace output_color_space = REC709_SRGB_D65;
  switch (opts.output_color_space) {
    case OutputColorSpace::kRec709_Linear_D65:
      output_color_space = REC709_LINEAR_D65;
      break;
    case OutputColorSpace::kRec709_sRGB_D65:
      output_color_space = REC709_SRGB_D65;
      break;
  }

  return filament::ColorGrading::Builder()
      .format(opts.format)
      .outputColorSpace(output_color_space)
      .dimensions(opts.dimension)
      .luminanceScaling(opts.luminance_scaling)
      .gamutMapping(opts.gamut_mapping)
      .exposure(opts.exposure)
      .nightAdaptation(opts.night_adaptation)
      .contrast(opts.contrast)
      .vibrance(opts.vibrance)
      .saturation(opts.saturation)
      .whiteBalance(opts.temperature, opts.tint)
      .channelMixer(opts.out_red, opts.out_green, opts.out_blue)
      .shadowsMidtonesHighlights(opts.shadows, opts.midtones, opts.highlights,
                                 opts.tonal_ranges)
      .slopeOffsetPower(opts.slope, opts.offset, opts.power)
      .curves(opts.shadow_gamma, opts.mid_point, opts.highlight_scale);
}

static void SetupCamera(const mjrfRenderRequest& request,
                        filament::Camera* camera) {
  const mjrCamera& cam = request.camera;
  const filament::Camera::Projection type =
      cam.orthographic ? filament::Camera::Projection::ORTHO
                       : filament::Camera::Projection::PERSPECTIVE;
  const float3 cam_pos(cam.pos[0], cam.pos[1], cam.pos[2]);
  const float3 cam_fwd(cam.forward[0], cam.forward[1], cam.forward[2]);
  const float3 cam_up(cam.up[0], cam.up[1], cam.up[2]);
  const float3 cam_at = cam_pos + cam_fwd;
  const float aspect_ratio =
      (float)request.viewport.width / (float)request.viewport.height;
  const float halfwidth =
      cam.frustum_width
          ? cam.frustum_width
          : 0.5f * aspect_ratio * (cam.frustum_top - cam.frustum_bottom);
  camera->lookAt(cam_pos, cam_at, cam_up);
  camera->setProjection(type, cam.frustum_center - halfwidth,
                        cam.frustum_center + halfwidth, cam.frustum_bottom,
                        cam.frustum_top, cam.frustum_near, cam.frustum_far);
  if (request.draw_mode == mjDRAW_MODE_DEPTH) {
    // Filament's setProjection puts the perspective far plane at infinity; use
    // the finite-far culling frustum so depth maps span [near, far].
    const mat4 proj = camera->getCullingProjectionMatrix();
    camera->setCustomProjection(proj, proj, cam.frustum_near, cam.frustum_far);
  }
}

// Sets up the `reflection_camera` as a reflection of `src_camera` across the
// plane defined by `surface_xform`, so the reflection texture can be applied to
// the plane with screen-space uvs.
static void SetupReflectionCamera(const mat4& surface_xform,
                                  const filament::Camera* src_camera,
                                  filament::Camera* reflection_camera) {
  const mat4 src_model_matrix = src_camera->getModelMatrix();
  const mat4 src_projection = src_camera->getCullingProjectionMatrix();

  reflection_camera->setModelMatrix(ToReflectionMatrix(surface_xform) *
                                    src_model_matrix);

  // Use un-skewed projection to preserve froxel lighting grid and shadow cascades.
  reflection_camera->setCustomProjection(src_projection, src_projection,
                                         src_camera->getNear(),
                                         src_camera->getCullingFar());
}

SceneView::SceneView(ObjectManager* object_mgr, MaterialManager* material_mgr,
                     const mjrfSceneParams& params)
    : object_mgr_(object_mgr), material_mgr_(material_mgr) {
  filament::Engine* engine = object_mgr_->GetEngine();
  reflection_mgr_ = std::make_unique<ReflectionManager>(engine, material_mgr);

  scene_ = engine->createScene();
  camera_ = engine->createCamera(utils::EntityManager::get().create());
  reflect_camera_ = engine->createCamera(utils::EntityManager::get().create());

  main_view_ = engine->createView();
  main_view_->setScene(scene_);
  main_view_->setCamera(camera_);
  main_view_->setVisibleLayers(0xff, kLayerMask_All);

  depth_segment_view_ = engine->createView();
  depth_segment_view_->setScene(scene_);
  depth_segment_view_->setCamera(camera_);
  depth_segment_view_->setVisibleLayers(0xff, kLayerMask_Object);
  depth_segment_view_->setShadowingEnabled(false);
  depth_segment_view_->setPostProcessingEnabled(false);

  reflect_view_ = engine->createView();
  reflect_view_->setScene(scene_);
  reflect_view_->setCamera(reflect_camera_);
  reflect_view_->setShadowingEnabled(true);
  reflect_view_->setPostProcessingEnabled(false);
  reflect_view_->setFrontFaceWindingInverted(true);
  reflect_view_->setVisibleLayers(0xff, kLayerMask_Object | kLayerMask_Skybox);
  reflect_view_->setMultiSampleAntiAliasingOptions({.enabled = false});

  // The headlight is aimed at each request's camera in Render(). It is created
  // black and disabled; requests that ask for it supply color and intensity.
  mjrfLightParams headlight_params;
  mjrf_defaultLightParams(&headlight_params);
  headlight_params.type = mjLIGHT_DIRECTIONAL;
  headlight_params.cast_shadow = 0;
  headlight_params.intensity = 0.0f;
  headlight_ = std::make_unique<Light>(engine, headlight_params);
  headlight_->Disable();
  AddToScene(headlight_.get());

  // Rotate the fog to align with mujoco's +Z up space.
  auto fog = main_view_->getFogEntity();
  auto& tm = engine->getTransformManager();
  tm.create(fog);
  tm.setTransform(tm.getInstance(fog),
                  mat4::rotation(filament::math::f::PI / 2, float3{-1, 0, 0}));

  mjrfRenderableParams haze_params;
  mjrf_defaultRenderableParams(&haze_params);
  haze_params.cast_shadow = false;
  haze_params.receive_shadow = false;
  haze_ = std::make_unique<Renderable>(engine, haze_params, material_mgr_);
  haze_->SetPriority(0);
  SetHaze(0.3f, nullptr);
  haze_->SetMesh(
      object_mgr_->GetBuiltins(numstacks_, numslices_, numquads_)->Haze());
  AddToScene(haze_.get());
}

SceneView::~SceneView() {
  filament::Engine* engine = object_mgr_->GetEngine();

  outliner_.reset();
  if (skybox_) {
    scene_->setSkybox(nullptr);
    engine->destroy(skybox_);
  }
  for (auto& light : lights_) {
    light->RemoveFromScene(scene_);
  }
  for (auto& renderable : renderables_) {
    renderable->RemoveFromScene(scene_);
  }
  lights_.clear();
  renderables_.clear();
  headlight_.reset();
  haze_.reset();
  engine->destroyCameraComponent(reflect_camera_->getEntity());
  engine->destroy(reflect_view_);
  engine->destroyCameraComponent(camera_->getEntity());
  if (color_grading_) {
    engine->destroy(color_grading_);
  }
  engine->destroy(scene_);
  engine->destroy(depth_segment_view_);
  engine->destroy(main_view_);
}

void SceneView::AddToScene(Light* light) {
  if (lights_.insert(light).second) {
    light->AddToScene(scene_, [this](Light* light) { RemoveFromScene(light); });
  }
}

void SceneView::RemoveFromScene(Light* light) {
  if (lights_.erase(light)) {
    light->RemoveFromScene(scene_);
  }
}

void SceneView::AddToScene(Renderable* renderable) {
  if (renderables_.insert(renderable).second) {
    renderable->AddToScene(
        scene_, [this](Renderable* renderable) { RemoveFromScene(renderable); });
  }
}

void SceneView::RemoveFromScene(Renderable* renderable) {
  if (renderables_.erase(renderable)) {
    renderable->RemoveFromScene(scene_);
  }
}

void SceneView::SetSkybox(const Texture* skybox_texture) {
  if (skybox_) {
    scene_->setSkybox(nullptr);
    GetEngine()->destroy(skybox_);
    skybox_ = nullptr;
  }
  if (skybox_texture) {
    filament::Skybox::Builder builder;
    builder.environment(skybox_texture->GetFilamentTexture());
    skybox_ = builder.build(*GetEngine());
    scene_->setSkybox(skybox_);
  }
}

void SceneView::PrepareToRender(std::span<const mjrfRenderRequest*> requests) {
  for (const mjrfRenderRequest* request : requests) {
    if (request->scene != this) {
      mju_error("Invalid scene for SceneView::PrepareToRender.");
    }
  }

  bool has_selected = false;
  reflection_mgr_->ClearRenderables();
  for (Renderable* renderable : renderables_) {
    renderable->Prepare(requests, reflection_mgr_.get());
    if (renderable->GetMaterial().selected) {
      has_selected = true;
    }
  }

  if (has_selected) {
    mjrfMaterial material;
    mjrf_defaultMaterial(&material);
    material.selected = true;
    outline_material_key_ = material_mgr_->PrepareMaterialInstance(
        material, mjDRAW_MODE_DEFAULT, mjGEOM_NONE, nullptr);
  }
}

void SceneView::Render(filament::Renderer* renderer,
                       const mjrfRenderRequest& request) {
  if (request.scene != this) {
    mju_error("Invalid scene for SceneView::Render.");
  }

  filament::Viewport viewport(request.viewport.left, request.viewport.bottom,
                              request.viewport.width, request.viewport.height);
  filament::View* view = main_view_;
  if (request.draw_mode == mjDRAW_MODE_DEPTH ||
      request.draw_mode == mjDRAW_MODE_SEGMENTATION_BY_ID ||
      request.draw_mode == mjDRAW_MODE_SEGMENTATION_BY_COLOR) {
    view = depth_segment_view_;
  } else {
    view->setShadowingEnabled(request.shadows);
    view->setPostProcessingEnabled(request.post_processing);
  }
  view->setViewport(viewport);

  filament::MultiSampleAntiAliasingOptions options =
      view->getMultiSampleAntiAliasingOptions();

  RenderTarget* render_target = RenderTarget::downcast(request.target);
  if (render_target) {
    // We need to disable msaa in order to render to texture.
    view->setMultiSampleAntiAliasingOptions({.enabled = false});
  }

  SetupCamera(request, camera_);
  camera_->setFocusDistance(focus_distance_);

  // Aim the headlight along this request's camera. Filament re-gathers the
  // scene's lights for each render() call, so requests batched into one frame
  // each see their own headlight.
  if (request.headlight) {
    const float3 forward = ReadFloat3(request.camera.forward);
    // Sit slightly behind the eye to avoid clipping artifacts.
    const float3 position = ReadFloat3(request.camera.pos) - 0.05f * forward;
    headlight_->SetTransform(position, forward);
    headlight_->SetColor(ReadFloat3(request.headlight_color));
    headlight_->SetIntensity(request.headlight_intensity);
    headlight_->Enable();
  } else {
    headlight_->Disable();
  }

  // with a skybox, center the horizon disks of infinite planes beneath the
  // camera and place the haze on the nearest plane below it
  const bool show_skybox = skybox_ != nullptr && ShowsSkybox(request);
  const float3 cam_pos = ReadFloat3(request.camera.pos);
  const float horizon = kHorizonFarRatio * request.camera.frustum_far;
  float haze_elevation = std::numeric_limits<float>::infinity();
  float3 haze_center;
  mat3f haze_rotation;

  std::vector<Renderable*> selected_renderables;
  for (Renderable* renderable : renderables_) {
    renderable->BindMaterialInstance(request);
    if (renderable->GetMaterial().selected &&
        request.draw_mode == mjDRAW_MODE_DEFAULT) {
      selected_renderables.push_back(renderable);
    }
    if (show_skybox && renderable->IsInfinitePlane()) {
      const mat4f& transform = renderable->GetTransform();
      const float3 normal = normalize(transform[2].xyz);
      const float elevation = dot(cam_pos - transform[3].xyz, normal);
      const float3 center = cam_pos - elevation * normal;
      const mat3f rotation(normalize(transform[0].xyz),
                           normalize(transform[1].xyz), normal);
      // part 1 of an infinite plane is its horizon disk
      renderable->SetPartTransform(
          1, mat4f(rotation, center) *
                 mat4f::scaling(float3{horizon, horizon, 1.0f}));
      if (elevation > 0.0f && elevation < haze_elevation) {
        haze_elevation = elevation;
        haze_center = center;
        haze_rotation = rotation;
      }
    }
  }

  const bool show_haze =
      show_skybox && request.haze &&
      haze_elevation < std::numeric_limits<float>::infinity();
  if (show_haze) {
    haze_->SetTransform(haze_center, haze_rotation);
    haze_->SetSize(float3{horizon, horizon, haze_elevation});
  }

  main_view_->setVisibleLayers(
      kLayerMask_Skybox | kLayerMask_Haze,
      (show_skybox ? kLayerMask_Skybox : 0) |
          (show_haze ? kLayerMask_Haze : 0));
  reflect_view_->setVisibleLayers(
      0xff, kLayerMask_Object | (show_skybox ? kLayerMask_Skybox : 0));

  const int num_reflective = reflection_mgr_->GetNumRenderables();
  for (int i = 0; i < num_reflective; ++i) {
    Renderable* renderable =
        Renderable::downcast(reflection_mgr_->GetRenderable(i));
    const mat4 transform(renderable->GetTransform());
    SetupReflectionCamera(transform, camera_, reflect_camera_);

    // Hide reflective surface from its own reflection pass.
    std::uint8_t prev_layer_mask = renderable->SetLayerMask(kLayerMask_None);

    // Hide geometry completely behind the mirror plane from the reflection pass.
    const float3 mirror_pos = transform[3].xyz;
    const float3 mirror_normal = normalize(transform[2].xyz);
    std::vector<std::pair<Renderable*, std::uint8_t>> hidden_behind;
    for (Renderable* other : renderables_) {
      if (other == renderable) {
        continue;
      }
      const mat4 other_xform(other->GetTransform());
      const float3 center = other_xform[3].xyz;
      // The geom's three scaled axes reach every corner, so their summed
      // lengths conservatively bound it; only hide when the whole bound is
      // behind the plane.
      const float radius = length(other_xform[0].xyz) +
                           length(other_xform[1].xyz) +
                           length(other_xform[2].xyz);
      if (dot(center - mirror_pos, mirror_normal) + radius < 0.0f) {
        hidden_behind.emplace_back(other, other->SetLayerMask(kLayerMask_None));
      }
    }

    // Render the reflection to its render target.
    viewport.left = 0;
    viewport.bottom = 0;
    reflect_view_->setViewport(viewport);
    const RenderTarget* target = reflection_mgr_->GetRenderTarget(i);
    reflect_view_->setRenderTarget(target->GetFilamentRenderTarget());
    renderer->render(reflect_view_);
    reflect_view_->setRenderTarget(nullptr);

    // Restore the behind-mirror geometry and the reflective surface.
    for (const auto& [other, mask] : hidden_behind) {
      other->SetLayerMask(mask);
    }
    renderable->SetLayerMask(prev_layer_mask);
  }
  for (int i = 0; i < num_reflective; ++i) {
    Renderable::downcast(reflection_mgr_->GetRenderable(i))
        ->SetMaterialInstance(reflection_mgr_->GetMaterialKey(i));
  }

  filament::RenderTarget* filament_render_target =
      render_target ? render_target->GetFilamentRenderTarget() : nullptr;
  view->setRenderTarget(filament_render_target);
  renderer->render(view);
  view->setRenderTarget(nullptr);

  if (!selected_renderables.empty()) {
    if (!outliner_) {
      outliner_ = std::make_unique<Outliner>(
          object_mgr_, kLayerMask_Outline, kLayerMask_Object,
          float4{0.9f, 0.9f, 0.2f, 0.7f}, 3.5f, 0.25f);
    }

    for (Renderable* renderable : selected_renderables) {
      renderable->SetMaterialInstance(outline_material_key_);
    }
    outliner_->Render(renderer, view, filament_render_target);
  }

  if (request.target) {
    view->setMultiSampleAntiAliasingOptions(options);
  }
}

void SceneView::SetColorGradingOptions(const ColorGradingOptions& opts) {
  auto tone_mapper = CreateToneMapper(opts.tone_mapper);
  auto color_grading = ToBuilder(opts)
                           .toneMapper(tone_mapper.get())
                           .build(*GetEngine());
  main_view_->setColorGrading(color_grading);
  if (color_grading_) {
    GetEngine()->destroy(color_grading_);
  }
  color_grading_ = color_grading;
  color_grading_options_ = opts;
}

filament::View* SceneView::GetDefaultRenderView() { return main_view_; }

ColorGradingOptions SceneView::GetColorGradingOptions() const {
  return color_grading_options_;
}

static ToneMapperType ToneMapperTypeFromString(std::string_view name) {
  if (name == "aces") {
    return ToneMapperType::kACES;
  } else if (name == "aces_legacy") {
    return ToneMapperType::kACESLegacy;
  } else if (name == "filmic") {
    return ToneMapperType::kFilmic;
  } else if (name == "linear") {
    return ToneMapperType::kLinear;
  } else if (name == "pbr_neutral") {
    return ToneMapperType::kPBRNeutral;
  }
  mju_warning("Unknown tone mapper type: %s", name.data());
  return ToneMapperType::kPBRNeutral;
}

static OutputColorSpace OutputColorSpaceFromString(
    std::string_view name) {
  if (name == "linear") {
    return OutputColorSpace::kRec709_Linear_D65;
  } else if (name == "srgb") {
    return OutputColorSpace::kRec709_sRGB_D65;
  }
  mju_warning("Unknown output color space: %s", name.data());
  return OutputColorSpace::kRec709_sRGB_D65;
}

static filament::RgbaType ColorTypeFromString(std::string_view name) {
  if (name == "linear") {
    return filament::RgbaType::LINEAR;
  } else if (name == "srgb") {
    return filament::RgbaType::sRGB;
  }
  mju_warning("Unknown color type: %s", name.data());
  return filament::RgbaType::sRGB;
}

static filament::QualityLevel QualityLevelFromString(std::string_view name) {
  if (name == "low") {
    return filament::QualityLevel::LOW;
  } else if (name == "medium") {
    return filament::QualityLevel::MEDIUM;
  } else if (name == "high") {
    return filament::QualityLevel::HIGH;
  } else if (name == "ultra") {
    return filament::QualityLevel::ULTRA;
  }
  mju_warning("Unknown quality level: %s", name.data());
  return filament::QualityLevel::MEDIUM;
}

static filament::ShadowType ShadowTypeFromString(std::string_view name) {
  if (name == "pcf") {
    return filament::ShadowType::PCF;
  } else if (name == "vsm") {
    return filament::ShadowType::VSM;
  } else if (name == "pcss") {
    return filament::ShadowType::PCSS;
  }
  mju_warning("Unknown shadow type: %s", name.data());
  return filament::ShadowType::PCF;
}

void SceneView::SetMsaa(int samples) {
  samples = std::clamp(samples, 1, 255);
  auto msaa = main_view_->getMultiSampleAntiAliasingOptions();
  msaa.enabled = samples > 1;
  msaa.sampleCount = static_cast<uint8_t>(samples);
  main_view_->setMultiSampleAntiAliasingOptions(msaa);
}

void SceneView::SetHaze(float ratio, const float rgba[4]) {
  mjrfMaterial material = haze_->GetMaterial();
  material.haze_ratio = std::clamp(ratio, 0.0f, 1.0f);
  if (rgba) {
    std::copy(rgba, rgba + 4, material.color);
  }
  haze_->UpdateMaterial(material);
}

float SceneView::GetHazeRatio() const {
  return haze_->GetMaterial().haze_ratio;
}

float4 SceneView::GetHazeColor() const {
  return ReadFloat4(haze_->GetMaterial().color);
}

void SceneView::Configure(const mjModel* model) {
  SetMsaa(model->vis.quality.offsamples);
  numstacks_ = model->vis.quality.numstacks;
  numslices_ = std::max(3, model->vis.quality.numslices);
  numquads_ = model->vis.quality.numquads;
  haze_->SetMesh(
      object_mgr_->GetBuiltins(numstacks_, numslices_, numquads_)->Haze());
  SetHaze(model->vis.map.haze, model->vis.rgba.haze);

  auto cg = color_grading_options_;
  cg.exposure = ReadElement(model, "filament.cg.exposure", cg.exposure);
  cg.contrast = ReadElement(model, "filament.cg.contrast", cg.contrast);
  cg.vibrance = ReadElement(model, "filament.cg.vibrance", cg.vibrance);
  cg.saturation = ReadElement(model, "filament.cg.saturation", cg.saturation);
  cg.temperature =
      ReadElement(model, "filament.cg.temperature", cg.temperature);
  cg.tint = ReadElement(model, "filament.cg.tint", cg.tint);
  cg.gamut_mapping =
      ReadElement(model, "filament.cg.gamut_mapping", cg.gamut_mapping);
  cg.luminance_scaling =
      ReadElement(model, "filament.cg.luminance_scaling", cg.luminance_scaling);
  cg.slope = ReadElement(model, "filament.cg.slope", cg.slope);
  cg.offset = ReadElement(model, "filament.cg.offset", cg.offset);
  cg.power = ReadElement(model, "filament.cg.power", cg.power);
  cg.shadow_gamma =
      ReadElement(model, "filament.cg.shadow_gamma", cg.shadow_gamma);
  cg.mid_point = ReadElement(model, "filament.cg.mid_point", cg.mid_point);
  cg.highlight_scale =
      ReadElement(model, "filament.cg.highlight_scale", cg.highlight_scale);
  cg.shadows = ReadElement(model, "filament.cg.shadows", cg.shadows);
  cg.midtones = ReadElement(model, "filament.cg.midtones", cg.midtones);
  cg.highlights = ReadElement(model, "filament.cg.highlights", cg.highlights);
  cg.tonal_ranges =
      ReadElement(model, "filament.cg.tonal_ranges", cg.tonal_ranges);
  cg.tone_mapper = ToneMapperTypeFromString(ReadElement<std::string_view>(
      model, "filament.cg.tone_mapping", "pbr_neutral"));
  cg.output_color_space =
      OutputColorSpaceFromString(ReadElement<std::string_view>(
          model, "filament.cg.output_color_space", "srgb"));
  SetColorGradingOptions(cg);

  auto ao = main_view_->getAmbientOcclusionOptions();
  ao.enabled = ReadElement(model, "filament.ao.enabled", true);
  ao.bentNormals = ReadElement(model, "filament.ao.bent_normals", false);
  ao.ssct.enabled = ReadElement(model, "filament.ao.ssct", ao.ssct.enabled);
  ao.quality = QualityLevelFromString(
      ReadElement<std::string_view>(model, "filament.ao.quality", "ultra"));
  ao.lowPassFilter = QualityLevelFromString(ReadElement<std::string_view>(
      model, "filament.ao.low_pass_filter", "ultra"));
  ao.upsampling = QualityLevelFromString(
      ReadElement<std::string_view>(model, "filament.ao.upsampling", "ultra"));
  ao.bilateralThreshold =
      ReadElement(model, "filament.ao.bilateral_threshold", 0.5f);
  main_view_->setAmbientOcclusionOptions(ao);

  auto shadow_type = main_view_->getShadowType();
  shadow_type = ShadowTypeFromString(
      ReadElement<std::string_view>(model, "filament.shadows.type", "pcf"));
  main_view_->setShadowType(shadow_type);

  auto fog_opts = main_view_->getFogOptions();
  fog_opts.enabled =
      ReadElement(model, "filament.fog.enabled", fog_opts.enabled);
  fog_opts.color = ReadElement(model, "filament.fog.color", fog_opts.color);
  fog_opts.distance =
      ReadElement(model, "filament.fog.distance", fog_opts.distance);
  fog_opts.density =
      ReadElement(model, "filament.fog.density", fog_opts.density);
  fog_opts.cutOffDistance = ReadElement(model, "filament.fog.cut_off_distance",
                                        fog_opts.cutOffDistance);
  fog_opts.maximumOpacity = ReadElement(model, "filament.fog.maximum_opacity",
                                        fog_opts.maximumOpacity);
  fog_opts.height = ReadElement(model, "filament.fog.height", fog_opts.height);
  fog_opts.heightFalloff =
      ReadElement(model, "filament.fog.height_falloff", fog_opts.heightFalloff);
  fog_opts.inScatteringStart = ReadElement(
      model, "filament.fog.in_scattering_start", fog_opts.inScatteringStart);
  fog_opts.inScatteringSize = ReadElement(
      model, "filament.fog.in_scattering_size", fog_opts.inScatteringSize);
  main_view_->setFogOptions(fog_opts);

  auto bloom = main_view_->getBloomOptions();
  bloom.enabled = ReadElement(model, "filament.bloom.enabled", bloom.enabled);
  bloom.strength =
      ReadElement(model, "filament.bloom.strength", bloom.strength);
  bloom.dirtStrength =
      ReadElement(model, "filament.bloom.dirt_strength", bloom.dirtStrength);
  bloom.quality = QualityLevelFromString(
      ReadElement<std::string_view>(model, "filament.bloom.quality", "low"));
  bloom.resolution =
      ReadElement(model, "filament.bloom.resolution", bloom.resolution);
  bloom.levels = ReadElement(model, "filament.bloom.levels", bloom.levels);
  main_view_->setBloomOptions(bloom);

  auto vignette = main_view_->getVignetteOptions();
  vignette.enabled =
      ReadElement(model, "filament.vignette.enabled", vignette.enabled);
  vignette.midPoint =
      ReadElement(model, "filament.vignette.midpoint", vignette.midPoint);
  main_view_->setVignetteOptions(vignette);

  material_mgr_->SetBaseColorType(
      ColorTypeFromString(ReadElement<std::string_view>(
          model, "filament.material.base_color_type", "srgb")));
}
}  // namespace mujoco
