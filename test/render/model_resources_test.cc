// Copyright 2026 DeepMind Technologies Limited
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

#include "src/render/filament/support/model_resources.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "src/render/filament/support/model_lights.h"
#include "src/render/filament/support/model_renderables.h"
#include "src/render/filament/support/renderer.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::Return;

mjrfContextConfig MakeOffscreenConfig() {
  mjrfContextConfig config;
  mjrf_defaultContextConfig(&config);
  config.native_window = nullptr;
  config.software_rendering = true;
  config.graphics_api = mjGRAPHICS_API_VULKAN;
  return config;
}

// A model whose texture files are in test/render/testdata, compiled eager or
// with <compiler passthrough="textures"/>.
static MjModelPtr LoadTexturedModel(std::string_view assets,
                                    std::string_view body, bool by_reference) {
  std::string xml =
      absl::Substitute(R"(
  <mujoco>
    <compiler texturedir="$0" passthrough="$1"/>
    <asset>
      $2
    </asset>
    <worldbody>
      $3
    </worldbody>
  </mujoco>
  )",
                       GetTestDataFilePath("render/testdata"),
                       by_reference ? "textures" : "none", assets, body);
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model, NotNull()) << error.data();
  return model;
}

// Renders the model of the given resources into a 64x64 RGB image, from a
// camera on the -y axis looking at the origin.
static std::vector<uint8_t> RenderModel(Renderer& renderer,
                                        ModelResources& resources) {
  const mjModel* model = resources.GetModel();
  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  mjrfScene* scene = renderer.GetScene("scene");
  ModelRenderables renderables(scene, &resources);
  ModelLights lights(scene, &resources);
  renderables.Update(data);
  lights.Update(data);
  mj_deleteData(data);

  Renderer::RenderTargetHandle target =
      renderer.GetRenderTargetHandle("target");
  if (target == Renderer::kInvalidRenderTarget) {
    mjrfRenderTargetConfig config;
    mjrf_defaultRenderTargetConfig(&config);
    config.width = 64;
    config.height = 64;
    config.color_format = mjPIXEL_FORMAT_RGB8;
    config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
    target = renderer.CreateRenderTarget("target", config);
  }

  Renderer::View views[1];
  Renderer::View& view = views[0];
  view.render_target = target;
  view.scene = renderer.GetSceneHandle("scene");
  view.camera.pos[1] = -4.0f;
  view.camera.forward[1] = 1.0f;
  view.camera.up[2] = 1.0f;
  view.camera.frustum_bottom = -0.05f;
  view.camera.frustum_top = 0.05f;
  view.camera.frustum_near = 0.1f;
  view.camera.frustum_far = 50.0f;
  view.post_processing = false;
  view.shadows = false;
  view.reflections = false;
  renderer.Render(views);
  std::span<const std::byte> pixels = renderer.GetPixels(target);
  const uint8_t* bytes = reinterpret_cast<const uint8_t*>(pixels.data());
  return std::vector<uint8_t>(bytes, bytes + pixels.size());
}

using ModelTextureTest = MujocoTest;

static constexpr char kPngAndKtxAssets[] = R"(
      <texture name="pattern" type="2d" file="test-pattern.png"/>
      <texture name="ibl" type="2d" file="ibl.ktx"/>
      <material name="pattern" texture="pattern"/>
)";
static constexpr char kPngAndKtxBody[] = R"(
      <light type="image" texture="ibl" intensity="10000"/>
      <geom type="box" size="1 1 1" material="pattern"/>
)";

// A PNG texture loaded with mj_loadTexture and a KTX read as bytes from the
// resource system render the same image as their eager counterparts.
TEST_F(ModelTextureTest, ByReferenceRendersLikeEager) {
  MjModelPtr eager = LoadTexturedModel(kPngAndKtxAssets, kPngAndKtxBody, false);
  MjModelPtr byref = LoadTexturedModel(kPngAndKtxAssets, kPngAndKtxBody, true);
  ASSERT_THAT(eager, NotNull());
  ASSERT_THAT(byref, NotNull());
  ASSERT_EQ(byref->tex_adr[0], -1);
  ASSERT_EQ(byref->tex_adr[1], -1);

  Renderer renderer(MakeOffscreenConfig());
  ModelResources eager_resources(eager.get(), renderer.GetContext());
  std::vector<uint8_t> expected = RenderModel(renderer, eager_resources);

  ModelResources byref_resources(byref.get(), renderer.GetContext());
  EXPECT_THAT(byref_resources.GetTexture(0), NotNull());
  EXPECT_THAT(byref_resources.GetTexture(1), NotNull());
  std::vector<uint8_t> actual = RenderModel(renderer, byref_resources);

  // the box fills the image with the pattern, which is not a flat color
  EXPECT_NE(std::count(expected.begin(), expected.end(), expected[0]),
            static_cast<std::ptrdiff_t>(expected.size()));
  EXPECT_EQ(actual, expected);
}

// A by-reference PNG authored colorspace="auto" keeps auto in the model; the
// renderer uploads it with the colorspace the file resolves to (sRGB here).
TEST_F(ModelTextureTest, ByReferenceResolvesAutoColorspace) {
  static constexpr char kAuto[] = R"(
      <texture name="pattern" type="2d" file="test-pattern.png"/>
      <material name="pattern" texture="pattern"/>
  )";
  static constexpr char kLinear[] = R"(
      <texture name="pattern" type="2d" file="test-pattern.png"
          colorspace="linear"/>
      <material name="pattern" texture="pattern"/>
  )";
  static constexpr char kBody[] = R"(
      <geom type="box" size="1 1 1" material="pattern"/>
  )";
  MjModelPtr eager = LoadTexturedModel(kAuto, kBody, false);
  MjModelPtr byref = LoadTexturedModel(kAuto, kBody, true);
  MjModelPtr linear = LoadTexturedModel(kLinear, kBody, true);
  ASSERT_THAT(eager, NotNull());
  ASSERT_THAT(byref, NotNull());
  ASSERT_THAT(linear, NotNull());
  ASSERT_EQ(eager->tex_colorspace[0], mjCOLORSPACE_SRGB);
  ASSERT_EQ(byref->tex_colorspace[0], mjCOLORSPACE_AUTO);

  Renderer renderer(MakeOffscreenConfig());
  ModelResources eager_resources(eager.get(), renderer.GetContext());
  ModelResources byref_resources(byref.get(), renderer.GetContext());
  ModelResources linear_resources(linear.get(), renderer.GetContext());
  std::vector<uint8_t> expected = RenderModel(renderer, eager_resources);
  EXPECT_EQ(RenderModel(renderer, byref_resources), expected);
  EXPECT_NE(RenderModel(renderer, linear_resources), expected);
}

// Two by-reference textures with the same file and decode parameters share one
// GPU texture; different parameters or embedded textures do not.
TEST_F(ModelTextureTest, ByReferenceSharesUploads) {
  static constexpr char kAssets[] = R"(
      <texture name="a" type="2d" file="test-pattern.png"/>
      <texture name="b" type="2d" file="test-pattern.png"/>
      <texture name="rgba" type="2d" file="test-pattern.png" nchannel="4"/>
      <texture name="linear" type="2d" file="test-pattern.png"
          colorspace="linear"/>
      <texture name="ktx" type="2d" file="ibl.ktx"/>
      <texture name="ktx2" type="2d" file="ibl.ktx" content_type="image/ktx"/>
  )";
  MjModelPtr eager = LoadTexturedModel(kAssets, "", false);
  MjModelPtr byref = LoadTexturedModel(kAssets, "", true);
  ASSERT_THAT(eager, NotNull());
  ASSERT_THAT(byref, NotNull());

  Renderer renderer(MakeOffscreenConfig());
  ModelResources eager_resources(eager.get(), renderer.GetContext());
  EXPECT_NE(eager_resources.GetTexture(0), eager_resources.GetTexture(1));

  ModelResources byref_resources(byref.get(), renderer.GetContext());
  const mjrfTexture* a = byref_resources.GetTexture(0);
  ASSERT_THAT(a, NotNull());
  EXPECT_EQ(byref_resources.GetTexture(1), a);
  EXPECT_NE(byref_resources.GetTexture(2), a);
  EXPECT_NE(byref_resources.GetTexture(3), a);
  // a KTX file is uploaded as the file's bytes; the content type is part of the
  // key as authored, so an explicit one is a different key
  EXPECT_THAT(byref_resources.GetTexture(4), NotNull());
  EXPECT_THAT(byref_resources.GetTexture(5), NotNull());
  EXPECT_NE(byref_resources.GetTexture(5), byref_resources.GetTexture(4));

  // re-uploading an id reads the file again, for every id which shares it
  byref_resources.UploadTexture(byref.get(), 0);
  const mjrfTexture* reloaded = byref_resources.GetTexture(0);
  EXPECT_THAT(reloaded, NotNull());
  EXPECT_NE(reloaded, a);
  EXPECT_EQ(byref_resources.GetTexture(1), reloaded);
  EXPECT_NE(byref_resources.GetTexture(2), reloaded);

  // an id whose key changed leaves the ids of its old key
  byref->tex_colorspace[1] = mjCOLORSPACE_LINEAR;
  byref_resources.UploadTexture(byref.get(), 1);
  EXPECT_THAT(byref_resources.GetTexture(1), NotNull());
  EXPECT_NE(byref_resources.GetTexture(1), reloaded);
  EXPECT_EQ(byref_resources.GetTexture(0), reloaded);
  byref_resources.UploadTexture(byref.get(), 0);
  EXPECT_NE(byref_resources.GetTexture(0), byref_resources.GetTexture(1));
}

// A by-reference texture whose file is missing warns once and leaves the
// material untextured, rendering as if the material had no texture.
TEST_F(ModelTextureTest, MissingByReferenceTextureWarns) {
  static constexpr char kMissing[] = R"(
      <texture name="missing" type="2d" file="missing.png"/>
      <material name="mat" texture="missing"/>
  )";
  static constexpr char kUntextured[] = R"(
      <material name="mat"/>
  )";
  static constexpr char kBody[] = R"(
      <geom type="box" size="1 1 1" material="mat"/>
  )";
  MjModelPtr missing = LoadTexturedModel(kMissing, kBody, true);
  MjModelPtr untextured = LoadTexturedModel(kUntextured, kBody, false);
  ASSERT_THAT(missing, NotNull());
  ASSERT_THAT(untextured, NotNull());

  Renderer renderer(MakeOffscreenConfig());
  ModelResources untextured_resources(untextured.get(), renderer.GetContext());
  std::vector<uint8_t> expected = RenderModel(renderer, untextured_resources);

  EXPECT_CALL(mock_warning_handler,
              Warn(HasSubstr("texture 0 not loaded, left untextured")))
      .WillOnce(Return());
  ModelResources missing_resources(missing.get(), renderer.GetContext());
  EXPECT_THAT(missing_resources.GetTexture(0), IsNull());
  EXPECT_EQ(RenderModel(renderer, missing_resources), expected);
}

// End-to-end MJCF test verifying mjrf_configureSceneFromModel wires
// vis.map.haze and vis.rgba.haze so an infinite ground plane blends smoothly
// into a gradient skybox under post-processing without a brightness jump at the
// horizon.
TEST_F(ModelTextureTest, ConfigureSceneHazeBlendsIntoGradientSkybox) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <visual>
      <map haze="0.5" zfar="100"/>
      <rgba haze="0.15 0.25 0.35 1"/>
    </visual>
    <asset>
      <texture type="skybox" builtin="gradient" rgb1=".3 .5 .7" rgb2="0 0 0"
          width="32" height="512"/>
      <texture name="grid" type="2d" builtin="checker" width="64" height="64"
          rgb1=".8 .8 .8" rgb2=".6 .6 .6"/>
      <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
    </asset>
    <worldbody>
      <light pos="0 0 10" dir="0 0 -1" directional="true"/>
      <geom type="plane" size="0 0 0.05" material="grid"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(kXml, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  mjData* data = mj_makeData(model.get());
  mj_forward(model.get(), data);

  Renderer renderer(MakeOffscreenConfig());
  ModelResources resources(model.get(), renderer.GetContext());
  mjrfScene* scene = renderer.GetScene("scene");
  mjrf_configureSceneFromModel(scene, model.get());
  ModelRenderables renderables(scene, &resources);
  ModelLights lights(scene, &resources);
  renderables.Update(data);
  lights.Update(data);
  mj_deleteData(data);

  constexpr int kWidth = 64;
  constexpr int kHeight = 256;
  mjrfRenderTargetConfig config;
  mjrf_defaultRenderTargetConfig(&config);
  config.width = kWidth;
  config.height = kHeight;
  config.color_format = mjPIXEL_FORMAT_RGB8;
  config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle target =
      renderer.CreateRenderTarget("haze_target", config);

  Renderer::View views[1];
  Renderer::View& view = views[0];
  view.render_target = target;
  view.scene = renderer.GetSceneHandle("scene");
  view.post_processing = true;
  view.shadows = false;
  view.reflections = false;
  view.camera.orthographic = false;
  view.camera.pos[0] = 15.3f;
  view.camera.pos[1] = -27.8f;
  view.camera.pos[2] = 20.0f;
  view.camera.forward[1] = 1.0f;
  view.camera.up[2] = 1.0f;
  view.camera.frustum_width = 0.1f;
  view.camera.frustum_bottom = -0.1f;
  view.camera.frustum_top = 0.1f;
  view.camera.frustum_near = 0.1f;
  view.camera.frustum_far = 100.0f;

  view.haze = false;
  renderer.Render(views);
  std::span<const std::byte> raw_no_haze = renderer.GetPixels(target);
  std::vector<uint8_t> no_haze(
      reinterpret_cast<const uint8_t*>(raw_no_haze.data()),
      reinterpret_cast<const uint8_t*>(raw_no_haze.data()) +
          raw_no_haze.size());

  view.haze = true;
  renderer.Render(views);
  std::span<const std::byte> raw_with_haze = renderer.GetPixels(target);
  std::vector<uint8_t> with_haze(
      reinterpret_cast<const uint8_t*>(raw_with_haze.data()),
      reinterpret_cast<const uint8_t*>(raw_with_haze.data()) +
          raw_with_haze.size());

  EXPECT_NE(with_haze, no_haze);

  // Without haze, the bright checker plane meets the dark skybox at row ~164
  // with a sharp single-row luminance jump; with haze matching the skybox
  // horizon color, the horizon discontinuity disappears and the lower cone
  // spreads the transition smoothly across ~37 rows.
  auto max_step_in_band = [&](const std::vector<uint8_t>& img) {
    int max_diff = 0;
    for (int y = 145; y < 195; ++y) {
      for (int c = 0; c < 3; ++c) {
        int a = img[(y * kWidth + kWidth / 2) * 3 + c];
        int b = img[((y + 1) * kWidth + kWidth / 2) * 3 + c];
        max_diff = std::max(max_diff, std::abs(a - b));
      }
    }
    return max_diff;
  };
  EXPECT_GT(max_step_in_band(no_haze), 80);
  EXPECT_LT(max_step_in_band(with_haze), 35);

  for (int c = 0; c < 3; ++c) {
    int sky_above_rim = with_haze[(160 * kWidth + kWidth / 2) * 3 + c];
    int haze_at_rim = with_haze[(164 * kWidth + kWidth / 2) * 3 + c];
    EXPECT_NEAR(haze_at_rim, sky_above_rim, 10);
  }
}

}  // namespace
}  // namespace mujoco
