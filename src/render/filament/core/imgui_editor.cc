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

#include <algorithm>
#include <any>
#include <cstdint>
#include <numbers>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include <imgui.h>
#include <filament/ColorGrading.h>
#include <filament/IndirectLight.h>
#include <filament/LightManager.h>
#include <filament/Options.h>
#include <filament/View.h>
#include <math/mathfwd.h>
#include <math/scalar.h>
#include <math/vec3.h>
#include <mujoco/mjrfilament.h>
#include "render/filament/core/color_grading_options.h"
#include "render/filament/core/imgui_editor.h"
#include "render/filament/core/scene_view.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;
using filament::math::mat4;

using NameList = std::span<const char*>;

namespace {

using InitValues = std::unordered_map<ImGuiID, std::any>;

// Uses static storage (not heap) for a map that is never destroyed, avoiding
// static destruction order issues. Note this is not causing a problem now but
// may do so in the future.
InitValues& GetInitValues() {
  alignas(InitValues) static char buf[sizeof(InitValues)];
  static InitValues* map = new (buf) InitValues();
  return *map;
}

constexpr ImVec4 kModifiedColor = ImVec4(1.0f, 0.6f, 0.2f, 1.0f);

template <typename T>
bool IsModified(ImGuiID id, const T& current_value) {
  auto it = GetInitValues().find(id);
  if (it == GetInitValues().end()) {
    GetInitValues()[id] = current_value;
    return false;
  }
  const T& a = current_value;
  const T& b = std::any_cast<const T&>(it->second);
  return a != b;
}

}  // namespace

NameList EnumNames(ToneMapperType v) {
  static const char* names[] = {"PBR Neutral", "ACES", "ACES Legacy", "Filmic",
                                "Linear"};
  return names;
}

NameList EnumNames(OutputColorSpace v) {
  static const char* names[] = {"Rec709-Linear-D65", "Rec709-sRGB-D65"};
  return names;
}

NameList EnumNames(filament::QualityLevel v) {
  static const char* names[] = {"LOW", "MEDIUM", "HIGH", "ULTRA"};
  return names;
}

NameList EnumNames(filament::ShadowType v) {
  static const char* names[] = {"PCF", "VSM", "DPCF", "PCSS"};
  return names;
}

NameList EnumNames(filament::TemporalAntiAliasingOptions::BoxType v) {
  static const char* names[] = {"AABB", "VARIANCE", "AABB_VARIANCE"};
  return names;
}

NameList EnumNames(filament::TemporalAntiAliasingOptions::BoxClipping v) {
  static const char* names[] = {"ACCURATE", "CLAMP", "NONE"};
  return names;
}

NameList EnumNames(filament::TemporalAntiAliasingOptions::JitterPattern v) {
  static const char* names[] = {"RGSS_X4", "UNIFORM_HELIX_X4", "HALTON_23_X8",
                                "HALTON_23_X16", "HALTON_23_X32"};
  return names;
}

NameList EnumNames(filament::ColorGrading::LutFormat v) {
  static const char* names[] = {"Integer", "Float"};
  return names;
}

NameList EnumNames(filament::BloomOptions::BlendMode v) {
  static const char* names[] = {"Add", "Interpolate"};
  return names;
}

NameList EnumNames(filament::DepthOfFieldOptions::Filter v) {
  static const char* names[] = {"NONE", "UNUSED", "MEDIAN"};
  return names;
}

template <typename T>
struct UiOpts {
  std::optional<T> min;
  std::optional<T> max;
  std::optional<T> step;
  std::optional<T> fstep;
};

// This is a workaround to fix compilation on gcc <= 12 and clang <= 16
template <typename T>
struct dependent_false : std::false_type {};

template <typename T>
bool Ui(std::string_view label, T* value, UiOpts<T> opts = {}) {
  bool changed = false;
  bool modified = false;
  ImGuiID id = ImGui::GetID(label.data());

  if constexpr (std::is_enum_v<T>) {
    modified = IsModified(id, static_cast<int>(*value));
  } else {
    modified = IsModified(id, *value);
  }

  if (modified) {
    ImGui::PushStyleColor(ImGuiCol_Text, kModifiedColor);
  }

  if constexpr (std::is_enum_v<T>) {
    int idx = static_cast<int>(*value);
    const NameList names = EnumNames(*value);
    changed = ImGui::Combo(label.data(), &idx, names.data(), names.size());
    *value = static_cast<T>(idx);
  } else if constexpr (std::is_same_v<T, bool>) {
    changed = ImGui::Checkbox(label.data(), value);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    changed = ImGui::DragScalar(label.data(), ImGuiDataType_U8, value, 1.0f);
  } else if constexpr (std::is_same_v<T, uint16_t>) {
    changed = ImGui::DragScalar(label.data(), ImGuiDataType_U16, value, 1.0f);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    changed = ImGui::DragScalar(label.data(), ImGuiDataType_U32, value, 1.0f);
  } else if constexpr (std::is_same_v<T, int>) {
    if (opts.min.has_value() && opts.max.has_value()) {
      changed = ImGui::SliderInt(label.data(), value, *opts.min, *opts.max);
    } else {
      float speed = opts.step.value_or(1);
      changed = ImGui::DragInt(label.data(), value, speed);
    }
  } else if constexpr (std::is_same_v<T, float>) {
    if (opts.min.has_value() && opts.max.has_value()) {
      changed =
          ImGui::SliderFloat(label.data(), value, *opts.min, *opts.max, "%.5f");
    } else {
      float speed = opts.step.value_or(0.01f);
      changed =
          ImGui::DragFloat(label.data(), value, speed, 0.0f, 0.0f, "%.5f");
    }
  } else if constexpr (std::is_same_v<T, float3>) {
    changed =
        ImGui::DragFloat3(label.data(), &value->x, 0.01f, 0.0f, 0.0f, "%.5f");
  } else if constexpr (std::is_same_v<T, float4>) {
    changed =
        ImGui::DragFloat4(label.data(), &value->x, 0.01f, 0.0f, 0.0f, "%.5f");
  } else {
    static_assert(dependent_false<T>::value, "Unsupported type");
  }

  // The Drag widgets above do not enforce bounds when only one of min/max is
  // given, and typed input (ctrl+click) bypasses Slider bounds; clamp here.
  if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
    if (changed) {
      if (opts.min.has_value()) {
        *value = std::max(*value, *opts.min);
      }
      if (opts.max.has_value()) {
        *value = std::min(*value, *opts.max);
      }
    }
  }

  if (modified) {
    ImGui::PopStyleColor();
    ImGui::SetItemTooltip("RMB to revert change");

    if (ImGui::IsItemClicked(ImGuiMouseButton_Right)) {
      if constexpr (std::is_enum_v<T>) {
        *value = static_cast<T>(std::any_cast<int>(GetInitValues()[id]));
      } else {
        *value = std::any_cast<T>(GetInitValues()[id]);
      }
      changed = true;
    }
  }

  return changed;
}

void DrawAmbientOcclusionGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getAmbientOcclusionOptions();
  constexpr float PI = std::numbers::pi_v<float>;

  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Radius (m)", &opts.radius, {.min = 0.f, .max = 5.f});
  changed |= Ui("Power Contrrast", &opts.power, {.min = 0.f, .max = 5.f});
  changed |= Ui("Bias (m)", &opts.bias, {.min = 0.f, .max = .001f});
  changed |= Ui("Resolution Scale", &opts.resolution, {.min = .5f, .max = 1.f});
  changed |= Ui("AO Intensity", &opts.intensity, {.min = 0.f, .max = 5.f});
  changed |= Ui("Bend Normals", &opts.bentNormals);
  changed |= Ui("Bilateral Threshold", &opts.bilateralThreshold,
                {.min = 0.f, .max = 10.f});
  changed |= Ui("Min. Horizon Angle (Rad)", &opts.minHorizonAngleRad,
                {.min = 0.f, .max = PI / 2});
  changed |= Ui("Quality", &opts.quality);
  changed |= Ui("Low Pass Filter", &opts.lowPassFilter);
  changed |= Ui("Upsampling", &opts.upsampling);
  changed |= Ui("SSCT Enabled", &opts.ssct.enabled);
  changed |= Ui("SSCT Cone Angle (Rad)", &opts.ssct.lightConeRad,
                {.min = 0.0f, .max = PI / 2});
  changed |= Ui("SSCT Shadow Distance (m)", &opts.ssct.shadowDistance);
  changed |= Ui("SSCT Max Contact (m)", &opts.ssct.contactDistanceMax);
  changed |= Ui("SSCT Intensity", &opts.ssct.intensity);
  changed |= Ui("SSCT Depth Bias", &opts.ssct.depthBias);
  changed |= Ui("SSCT Depth Slope Bias", &opts.ssct.depthSlopeBias);
  changed |=
      Ui("SSCT Sample Count", &opts.ssct.sampleCount, {.min = 1, .max = 255});
  changed |= Ui("SSCT Ray Count", &opts.ssct.rayCount, {.min = 1, .max = 255});
  changed |= Ui("SSCT Light Direction", &opts.ssct.lightDirection);

  if (changed) {
    opts.resolution = (opts.resolution < 0.75f) ? 0.f : 1.0f;
    opts.power = std::max(opts.power, 0.0f);
    opts.radius = std::clamp(opts.radius, 0.0f, 10.0f);
    opts.bias = std::clamp(opts.bias, 0.0f, 0.001f);
    opts.ssct.lightConeRad =
        std::clamp(opts.ssct.lightConeRad, 0.0f, filament::math::f::PI / 2.0f);

    view->setAmbientOcclusionOptions(opts);
  }
}

void DrawScreenSpaceGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto reflection = view->getScreenSpaceReflectionsOptions();
  bool refraction_enabled = view->isScreenSpaceRefractionEnabled();
  auto guard_band = view->getGuardBandOptions();

  bool changed = false;
  changed |= Ui("Reflection Enabled", &reflection.enabled);
  changed |= Ui("Refl. Thickness", &reflection.thickness);
  changed |= Ui("Refl. Bias", &reflection.bias);
  changed |= Ui("Refl. Max Distance", &reflection.maxDistance);
  changed |= Ui("Refl. Stride", &reflection.stride);
  if (changed) {
    view->setScreenSpaceReflectionsOptions(reflection);
  }
  if (Ui("Refraction", &refraction_enabled)) {
    view->setScreenSpaceRefractionEnabled(refraction_enabled);
  }
  if (Ui("Guard Band Enabled", &guard_band.enabled)) {
    view->setGuardBandOptions(guard_band);
  }
}

void DrawShadowingGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto vsm_opts = view->getVsmShadowOptions();
  auto soft_opts = view->getSoftShadowOptions();

  bool enabled = view->isShadowingEnabled();
  if (Ui("Shadowing", &enabled)) {
    view->setShadowingEnabled(enabled);
  }
  auto shadow_type = view->getShadowType();
  if (Ui("Shadow Type", &shadow_type)) {
    view->setShadowType(shadow_type);
  }

  bool changed = false;
  changed |= Ui("VSM Anisotropy", &vsm_opts.anisotropy);
  changed |= Ui("VSM Mipmapping", &vsm_opts.mipmapping);
  changed |= Ui("VSM MSAA Samples", &vsm_opts.msaaSamples);
  changed |= Ui("VSM High Precision", &vsm_opts.highPrecision);
  changed |= Ui("VSM Min Variance Scale", &vsm_opts.minVarianceScale);
  changed |= Ui("VSM Light Bleed Reduction", &vsm_opts.lightBleedReduction);
  if (changed) {
    vsm_opts.minVarianceScale = std::max(vsm_opts.minVarianceScale, 0.0f);
    vsm_opts.lightBleedReduction =
        std::clamp(vsm_opts.lightBleedReduction, 0.0f, 1.0f);

    view->setVsmShadowOptions(vsm_opts);
  }

  changed = false;
  changed |= Ui("Soft Penumbra Scale", &soft_opts.penumbraScale);
  changed |= Ui("Soft Penumbra Ratio Scale", &soft_opts.penumbraRatioScale);
  if (changed) {
    soft_opts.penumbraScale = std::max(soft_opts.penumbraScale, 0.0f);
    soft_opts.penumbraRatioScale = std::max(soft_opts.penumbraRatioScale, 1.0f);

    view->setSoftShadowOptions(soft_opts);
  }
}

void DrawPostProcessingGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  bool enabled = view->isPostProcessingEnabled();
  if (Ui("Enabled", &enabled)) {
    view->setPostProcessingEnabled(enabled);
  }
}

void DrawFxaaGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  bool enabled = view->getAntiAliasing() == filament::AntiAliasing::FXAA;
  if (Ui("Enabled", &enabled)) {
    view->setAntiAliasing(enabled ? filament::AntiAliasing::FXAA
                                  : filament::AntiAliasing::NONE);
  }
}

void DrawMsaaGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getMultiSampleAntiAliasingOptions();

  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Samples", &opts.sampleCount);
  changed |= Ui("Custom Resolve", &opts.customResolve);
  if (changed) {
    view->setMultiSampleAntiAliasingOptions(opts);
  }
}

void DrawTaaGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getTemporalAntiAliasingOptions();

  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Filter Width", &opts.filterWidth);
  changed |= Ui("Feedback", &opts.feedback, {.min = 0.f, .max = 1.f});
  changed |= Ui("Lod Bias", &opts.lodBias);
  changed |= Ui("Sharpness", &opts.sharpness);
  changed |= Ui("Upscaling", &opts.upscaling);
  changed |= Ui("Filter History", &opts.filterHistory);
  changed |= Ui("Filter Input", &opts.filterInput);
  changed |= Ui("Use YCoCg", &opts.useYCoCg);
  changed |= Ui("Box Type", &opts.boxType);
  changed |= Ui("Box Clipping", &opts.boxClipping);
  changed |= Ui("Jitter Pattern", &opts.jitterPattern);
  changed |= Ui("Variance Gamma", &opts.varianceGamma);
  changed |= Ui("Prevent Flickering", &opts.preventFlickering);

  if (changed) {
    opts.feedback = std::clamp(opts.feedback, 0.f, 1.f);
    opts.varianceGamma = std::clamp(opts.varianceGamma, 0.75f, 1.25f);
    view->setTemporalAntiAliasingOptions(opts);
  }
}

void DrawBloomGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getBloomOptions();
  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Strength", &opts.strength);
  changed |= Ui("Quality", &opts.quality);
  changed |= Ui("Resolution", &opts.resolution);
  changed |= Ui("Levels", &opts.levels);
  changed |= Ui("Blend Mode", &opts.blendMode);
  changed |= Ui("Threshold", &opts.threshold);
  changed |= Ui("Highlight", &opts.highlight);
  changed |= Ui("Lens Flare", &opts.lensFlare);
  changed |= Ui("Starburst", &opts.starburst);
  changed |= Ui("Ch. Aberration", &opts.chromaticAberration);
  changed |= Ui("Ghost Count", &opts.ghostCount);
  changed |= Ui("Ghost Spacing", &opts.ghostSpacing);
  changed |= Ui("Ghost Threshold", &opts.ghostThreshold);
  changed |= Ui("Halo Radius", &opts.haloRadius);
  changed |= Ui("Halo Thickness", &opts.haloThickness);
  changed |= Ui("Halo Threshold", &opts.haloThreshold);

  if (changed) {
    opts.strength = std::clamp(opts.strength, 0.0f, 1.0f);
    opts.levels = std::clamp<uint8_t>(opts.levels, 1, 11);
    opts.ghostSpacing = std::clamp(opts.ghostSpacing, 0.0f, 1.0f);
    opts.highlight = std::max(opts.highlight, 10.0f);
    opts.haloThickness = std::clamp(opts.haloThickness, 0.0f, 1.0f);
    opts.haloRadius = std::clamp(opts.haloRadius, 0.0f, 0.5f);
    view->setBloomOptions(opts);
  }
}

void DrawColorGradingGui(SceneView* scene_view) {
  auto opts = scene_view->GetColorGradingOptions();

  bool changed = false;
  changed |= Ui("Tone Mapper", &opts.tone_mapper);
  changed |= Ui("Output Color Space", &opts.output_color_space);
  changed |= Ui("Lut Format", &opts.format);
  changed |= Ui("Lut Dimension", &opts.dimension);
  changed |= Ui("Luminance Scaling", &opts.luminance_scaling);
  changed |= Ui("Gamut Mapping", &opts.gamut_mapping);
  changed |= Ui("Exposure", &opts.exposure, {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Night Adaptation", &opts.night_adaptation,
                {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Contrast", &opts.contrast, {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Vibrance", &opts.vibrance, {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Saturation", &opts.saturation, {.step = 0.01f, .fstep = 0.1f});
  changed |=
      Ui("Temperature", &opts.temperature, {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Tint", &opts.tint, {.step = 0.01f, .fstep = 0.1f});
  changed |= Ui("Out Red", &opts.out_red);
  changed |= Ui("Out Green", &opts.out_green);
  changed |= Ui("Out Blue", &opts.out_blue);
  changed |= Ui("Shadows", &opts.shadows);
  changed |= Ui("Midtones", &opts.midtones);
  changed |= Ui("Highlights", &opts.highlights);
  changed |= Ui("Tonal Ranges", &opts.tonal_ranges);
  changed |= Ui("Slope", &opts.slope);
  changed |= Ui("Offset", &opts.offset);
  changed |= Ui("Power", &opts.power);
  changed |= Ui("Shadow Gamma", &opts.shadow_gamma);
  changed |= Ui("Mid Point", &opts.mid_point);
  changed |= Ui("Highlight Scale", &opts.highlight_scale);

  opts.contrast = std::clamp(opts.contrast, 0.0f, 2.0f);
  opts.vibrance = std::clamp(opts.vibrance, 0.0f, 2.0f);
  opts.saturation = std::clamp(opts.saturation, 0.0f, 2.0f);
  opts.temperature = std::clamp(opts.temperature, -1.0f, 1.0f);
  opts.tint = std::clamp(opts.tint, -1.0f, 1.0f);
  opts.slope.x = std::max(0.000001f, opts.slope.x);
  opts.slope.y = std::max(0.000001f, opts.slope.y);
  opts.slope.z = std::max(0.000001f, opts.slope.z);
  opts.power.x = std::max(0.000001f, opts.power.x);
  opts.power.y = std::max(0.000001f, opts.power.y);
  opts.power.z = std::max(0.000001f, opts.power.z);
  opts.shadow_gamma.x = std::max(0.000001f, opts.shadow_gamma.x);
  opts.shadow_gamma.y = std::max(0.000001f, opts.shadow_gamma.y);
  opts.shadow_gamma.z = std::max(0.000001f, opts.shadow_gamma.z);
  opts.mid_point.x = std::max(0.000001f, opts.mid_point.x);
  opts.mid_point.y = std::max(0.000001f, opts.mid_point.y);
  opts.mid_point.z = std::max(0.000001f, opts.mid_point.z);
  if (opts.highlight_scale.x == 0.0f) opts.highlight_scale.x = 0.000001f;
  if (opts.highlight_scale.y == 0.0f) opts.highlight_scale.y = 0.000001f;
  if (opts.highlight_scale.z == 0.0f) opts.highlight_scale.z = 0.000001f;

  if (changed) {
    scene_view->SetColorGradingOptions(opts);
  }
}

void DrawDepthOfFieldGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getDepthOfFieldOptions();
  float focus_distance = scene_view->GetFocusDistance();
  bool changed = false;

  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Filter", &opts.filter);
  changed |= Ui("Max. Aperture", &opts.maxApertureDiameter);
  changed |= Ui("Resolution", &opts.nativeResolution);
  changed |= Ui("Focal Distance", &focus_distance);
  changed |= Ui("COC Scale", &opts.cocScale);
  changed |= Ui("COC Aspect Ratio", &opts.cocAspectRatio);
  changed |= Ui("Max COC (Foreground)", &opts.maxForegroundCOC);
  changed |= Ui("Max COC (Background)", &opts.maxBackgroundCOC);
  changed |= Ui("Foreground Ring Count", &opts.foregroundRingCount);
  changed |= Ui("Background Ring Count", &opts.backgroundRingCount);
  changed |= Ui("Fast Gather Ring Count", &opts.fastGatherRingCount);
  if (changed) {
    view->setDepthOfFieldOptions(opts);
    scene_view->SetFocusDistance(focus_distance);
  }
}

void DrawDitheringGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  bool enabled = view->getDithering() != filament::Dithering::NONE;
  if (Ui("Enabled", &enabled)) {
    view->setDithering(enabled ? filament::Dithering::TEMPORAL
                               : filament::Dithering::NONE);
  }
}

void DrawFogGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getFogOptions();
  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Color", &opts.color);
  changed |= Ui("Use IBL Color", &opts.fogColorFromIbl);
  changed |= Ui("Density", &opts.density);
  changed |= Ui("Distance", &opts.distance);
  changed |= Ui("Cutoff", &opts.cutOffDistance);
  changed |= Ui("Max. Opacity", &opts.maximumOpacity);
  changed |= Ui("Height", &opts.height);
  changed |= Ui("Height Falloff", &opts.heightFalloff);
  changed |= Ui("InScattering Start", &opts.inScatteringStart);
  changed |= Ui("InScattering Size", &opts.inScatteringSize);
  if (changed) {
    opts.maximumOpacity = std::clamp(opts.maximumOpacity, 0.0f, 1.0f);
    view->setFogOptions(opts);
  }
}

void DrawHazeGui(SceneView* scene_view) {
  float ratio = scene_view->GetHazeRatio();
  float4 color = scene_view->GetHazeColor();
  bool changed = false;
  changed |= Ui("Ratio", &ratio, {.min = 0.0f, .max = 1.0f});
  changed |= ImGui::ColorEdit4("Color", &color.x);
  if (changed) {
    const float rgba[4] = {color.x, color.y, color.z, color.w};
    scene_view->SetHaze(ratio, rgba);
  }
}

void DrawVignetteGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto opts = view->getVignetteOptions();
  bool changed = false;
  changed |= Ui("Enabled", &opts.enabled);
  changed |= Ui("Color", &opts.color);
  changed |= Ui("Midpoint", &opts.midPoint);
  changed |= Ui("Roundness", &opts.roundness);
  changed |= Ui("Feature", &opts.feather);

  if (changed) {
    opts.midPoint = std::clamp(opts.midPoint, 0.0f, 1.0f);
    opts.roundness = std::clamp(opts.roundness, 0.0f, 1.0f);
    opts.feather = std::clamp(opts.feather, 0.0f, 1.0f);
    view->setVignetteOptions(opts);
  }
}

void DrawVisibleLayersGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  uint32_t layers = view->getVisibleLayers();
  bool changed = false;

  ImGui::Columns(2);
  changed |= ImGui::CheckboxFlags("Layer 0", &layers, 1 << 0);
  changed |= ImGui::CheckboxFlags("Layer 1", &layers, 1 << 1);
  changed |= ImGui::CheckboxFlags("Layer 2", &layers, 1 << 2);
  changed |= ImGui::CheckboxFlags("Layer 3", &layers, 1 << 3);
  ImGui::NextColumn();
  changed |= ImGui::CheckboxFlags("Layer 4", &layers, 1 << 4);
  changed |= ImGui::CheckboxFlags("Layer 5", &layers, 1 << 5);
  changed |= ImGui::CheckboxFlags("Layer 6", &layers, 1 << 6);
  changed |= ImGui::CheckboxFlags("Layer 7", &layers, 1 << 7);
  ImGui::Columns(1);
  if (changed) {
    view->setVisibleLayers(0xff, static_cast<uint8_t>(layers));
  }
}

void DrawCameraGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  filament::Camera& camera = view->getCamera();
  auto position = float3(camera.getPosition());
  Ui("Position", &position);
  auto direction = camera.getForwardVector();
  Ui("Direction", &direction);
}

void DrawIndirectLightGui(SceneView* scene_view) {
  filament::View* view = scene_view->GetDefaultRenderView();
  auto ibl = view->getScene()->getIndirectLight();

  float intensity = 10000.0f;
  if (ibl) {
    intensity = ibl->getIntensity();
    if (Ui("Intensity", &intensity, {.min = 0.0f, .step = 1000.0f, .fstep = 10000.0f})) {
      ibl->setIntensity(intensity);
    }
  }
}

void DrawLightGui(filament::LightManager& lm,
                  filament::LightManager::Instance li) {
  auto color = lm.getColor(li);
  if (ImGui::ColorEdit3("Color", &color.x)) {
    lm.setColor(li, color);
  }
  auto position = lm.getPosition(li);
  if (Ui("Position", &position)) {
    lm.setPosition(li, position);
  }
  auto direction = lm.getDirection(li);
  if (Ui("Direction", &direction)) {
    lm.setDirection(li, direction);
  }
  float intensity = lm.getIntensity(li);
  if (Ui("Intensity", &intensity, {.min = 0.0f, .step = 1000.0f, .fstep = 10000.0f})) {
    lm.setIntensityCandela(li, intensity);
  }
  float falloff = lm.getFalloff(li);
  if (Ui("Falloff Radius", &falloff)) {
    lm.setFalloff(li, falloff);
  }
  if (lm.isSpotLight(li)) {
    float spotLightInnerCone = lm.getSpotLightInnerCone(li);
    float spotLightOuterCone = lm.getSpotLightOuterCone(li);
    if (Ui("Spot Light Inner Cone", &spotLightInnerCone)) {
      lm.setSpotLightCone(li, spotLightInnerCone, spotLightOuterCone);
    }
    if (Ui("Spot Light Outer Cone", &spotLightOuterCone)) {
      lm.setSpotLightCone(li, spotLightInnerCone, spotLightOuterCone);
    }
  }
  ImGui::Columns(2);
  for (int i = 0; i < 8; ++i) {
    std::string name = "Channel " + std::to_string(i);
    bool enabled = lm.getLightChannel(li, i);
    if (i == 4) {
      ImGui::NextColumn();
    }
    if (Ui(name, &enabled)) {
      lm.setLightChannel(li, i, enabled);
    }
  }
  ImGui::Columns(1);
  if (ImGui::TreeNodeEx("Shadow Options")) {
    bool enabled = lm.isShadowCaster(li);
    if (Ui("Enabled", &enabled)) {
      lm.setShadowCaster(li, enabled);
    }

    filament::LightManager::ShadowOptions opts = lm.getShadowOptions(li);
    bool changed = false;
    changed |= Ui("Stable", &opts.stable);
    changed |= Ui("LiSPSM", &opts.lispsm);
    changed |= Ui("SS Contact Shadows", &opts.screenSpaceContactShadows);
    changed |= Ui("Map Size", &opts.mapSize);
    changed |= Ui("Bulb Radius", &opts.shadowBulbRadius);
    changed |= Ui("Step Count", &opts.stepCount);
    changed |= Ui("# Cascades", &opts.shadowCascades);
    changed |= Ui("Constant Bias", &opts.constantBias);
    changed |= Ui("Normal Bias", &opts.normalBias);
    changed |= Ui("EL VSM Enabled", &opts.vsm.elvsm);
    changed |= Ui("VSM Blur Width", &opts.vsm.blurWidth);
    if (changed) {
      lm.setShadowOptions(li, opts);
    }
    ImGui::TreePop();
  }
}

void DrawSceneEditor(mjrfScene* scene) {
  SceneView* scene_view = SceneView::downcast(scene);
  filament::View* view = scene_view->GetDefaultRenderView();
  filament::Engine* engine = scene_view->GetEngine();
  filament::LightManager& lm = engine->getLightManager();

  if (ImGui::TreeNodeEx("Ambient Occlusion")) {
    mujoco::DrawAmbientOcclusionGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Screen Space")) {
    mujoco::DrawScreenSpaceGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Shadowing")) {
    mujoco::DrawShadowingGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Haze")) {
    mujoco::DrawHazeGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Post Processing")) {
    mujoco::DrawPostProcessingGui(scene_view);
    if (ImGui::TreeNodeEx("Anti Aliasing (FXAA)")) {
      mujoco::DrawFxaaGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Anti Aliasing (MSAA)")) {
      mujoco::DrawMsaaGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Anti Aliasing (Temporal)")) {
      mujoco::DrawTaaGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Bloom")) {
      mujoco::DrawBloomGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Color Grading")) {
      mujoco::DrawColorGradingGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Depth of Field")) {
      mujoco::DrawDepthOfFieldGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Dithering")) {
      mujoco::DrawDitheringGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Fog")) {
      mujoco::DrawFogGui(scene_view);
      ImGui::TreePop();
    }
    if (ImGui::TreeNodeEx("Vignette")) {
      mujoco::DrawVignetteGui(scene_view);
      ImGui::TreePop();
    }
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Visibility Layers")) {
    mujoco::DrawVisibleLayersGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Camera")) {
    mujoco::DrawCameraGui(scene_view);
    ImGui::TreePop();
  }
  if (ImGui::TreeNodeEx("Lights")) {
    if (ImGui::TreeNodeEx("Indirect (Image-based) Light")) {
      mujoco::DrawIndirectLightGui(scene_view);
      ImGui::TreePop();
    }
    view->getScene()->forEach([&](utils::Entity entity) {
      auto li = lm.getInstance(entity);
      if (!li.isValid()) {
        return;
      }

      const char* type = lm.isDirectional(li)  ? " (D)"
                         : lm.isPointLight(li) ? " (P)"
                                               : " (S)";
      const std::string name = "Light " + std::to_string(entity.getId()) + type;
      if (ImGui::TreeNodeEx(name.c_str())) {
        mujoco::DrawLightGui(lm, li);
        ImGui::TreePop();
      }
    });
    ImGui::TreePop();
  }
}

}  // namespace mujoco
