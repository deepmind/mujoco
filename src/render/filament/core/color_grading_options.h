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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_CORE_COLOR_GRADING_OPTIONS_H_
#define MUJOCO_SRC_RENDER_FILAMENT_CORE_COLOR_GRADING_OPTIONS_H_

#include <cstdint>
#include <memory>

#include <filament/ColorGrading.h>
#include <filament/ToneMapper.h>
#include <math/vec3.h>
#include <math/vec4.h>

namespace mujoco {

// Filament has deprecated its ToneMapper enum. Instead, users can define their
// own ToneMapper objects directly. However, we don't want to expose that
// complexity to users of this library, so we define our own enum which can
// then be used to create predefined ToneMapper objects.
enum class ToneMapperType {
  kPBRNeutral,
  kACES,
  kACESLegacy,
  kFilmic,
  kLinear,
};

// Supported output color spaces for color grading.
enum class OutputColorSpace {
  kRec709_Linear_D65,
  kRec709_sRGB_D65,
};

std::unique_ptr<filament::ToneMapper> CreateToneMapper(ToneMapperType type);

// Structure used to initialize a filament::ColorGrading::Builder object.
// Unlike most other post processing options, the ColorGrading isn't defined
// as a struct, but rather as a write-only class with a Builder. However, we
// want to be able to change these values at runtime, so we define our own
// struct from which we can create new ColorGrading objects.
//
// The default values for this struct are chosen to match the default values
// for the Filament ColorGrading::Builder class.
struct ColorGradingOptions {
  ToneMapperType tone_mapper = ToneMapperType::kPBRNeutral;
  OutputColorSpace output_color_space = OutputColorSpace::kRec709_sRGB_D65;
  filament::ColorGrading::LutFormat format =
      filament::ColorGrading::LutFormat::INTEGER;
  uint8_t dimension = 32;
  bool luminance_scaling = false;
  bool gamut_mapping = false;
  float exposure = 0.0f;
  float night_adaptation = 0.0f;
  float contrast = 1.0f;
  float vibrance = 1.0f;
  float saturation = 1.0f;
  float temperature = 0.0f;
  float tint = 0.0f;
  filament::math::float3 out_red = {1.0f, 0.0f, 0.0f};
  filament::math::float3 out_green = {0.0f, 1.0f, 0.0f};
  filament::math::float3 out_blue = {0.0f, 0.0f, 1.0f};
  // Note: shadows, midtones, and highlights do not actually match the filament
  // defaults (which are [1,1,1,0]), but we do not want to change this now as it
  // will affect the default rendering results for all our users.
  filament::math::float4 shadows = {1.0f, 1.0f, 1.0f, 1.0f};
  filament::math::float4 midtones = {1.0f, 1.0f, 1.0f, 1.0f};
  filament::math::float4 highlights = {1.0f, 1.0f, 1.0f, 1.0f};
  filament::math::float4 tonal_ranges = {0.0f, 0.333f, 0.550f, 1.0f};
  filament::math::float3 slope = {1.0f};
  filament::math::float3 offset = {0.0f};
  filament::math::float3 power = {1.0f};
  filament::math::float3 shadow_gamma = {1.0f};
  filament::math::float3 mid_point = {1.0f};
  filament::math::float3 highlight_scale = {1.0f};
};

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_CORE_COLOR_GRADING_OPTIONS_H_
