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

#include "src/render/filament/support/renderer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "src/render/filament/support/mjrf_ptr.h"
#include "src/render/filament/support/model_resources.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::HasSubstr;
using RendererTest = MujocoTest;

static constexpr uint8_t kRed[3] = {255, 0, 0};
static constexpr uint8_t kGreen[3] = {0, 255, 0};
static constexpr uint8_t kBlue[3] = {0, 0, 255};
static constexpr uint8_t kWhite[3] = {255, 255, 255};

static UniquePtr<mjrfTexture> MakeTexture(mjrfContext* ctx,
                                          const uint8_t* color) {
  mjrfTextureConfig config;
  mjrf_defaultTextureConfig(&config);
  config.width = 1;
  config.height = 1;
  config.format = mjPIXEL_FORMAT_RGB8;
  UniquePtr<mjrfTexture> texture = CreateTexture(ctx, config);

  mjrfTextureData data;
  mjrf_defaultTextureData(&data);
  data.bytes = color;
  data.num_bytes = 3;
  mjrf_uploadTexture(texture.get(), &data);
  return texture;
}

static UniquePtr<mjrfMesh> MakeQuadMesh(mjrfContext* ctx, bool small = false) {
  static const float vertices[] = {
      -256.0f, -256.0f, 256.0f, -256.0f, -256.0f, 256.0f, 256.0f, 256.0f,
  };
  static const float small_vertices[] = {
      -4.0f, -4.0f, 4.0f, -4.0f, -4.0f, 4.0f, 4.0f, 4.0f,
  };
  static const float uvs[] = {
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
  };
  static const uint8_t colors[] = {
      255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255,
  };
  static const uint16_t indices[] = {0, 1, 2, 1, 3, 2};

  mjrfMeshConfig config;
  mjrf_defaultMeshConfig(&config);
  config.num_attributes = 3;
  config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
  config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
  config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
  config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
  config.attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_COLOR;
  config.attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_UBYTE4;
  config.max_vertices = 4;
  config.max_indices = 6;
  config.interleaved = false;
  config.index_type = mjINDEX_TYPE_U16;
  config.primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;
  UniquePtr<mjrfMesh> mesh = CreateMesh(ctx, config);

  mjrfMeshData data;
  mjrf_defaultMeshData(&data);
  data.num_vertices = 4;
  data.vertices[0] = small ? small_vertices : vertices;
  data.vertices[1] = uvs;
  data.vertices[2] = colors;
  data.num_indices = 6;
  data.indices = indices;
  mjrf_uploadMesh(mesh.get(), &data);

  return mesh;
}

static UniquePtr<mjrfRenderable> MakeRenderable(mjrfContext* ctx,
                                                mjrfMesh* mesh,
                                                mjrfTexture* texture) {
  mjrfMaterial material;
  mjrf_defaultMaterial(&material);
  material.decor = true;
  material.color_texture = texture;

  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);
  params.cast_shadow = false;
  params.receive_shadow = false;
  auto renderable = CreateRenderable(ctx, params);
  mjrf_setRenderableMesh(renderable.get(), mesh, 0, 0);
  mjrf_setRenderableMaterial(renderable.get(), &material);
  return renderable;
}

mjrfContextConfig MakeOffscreenConfig() {
  mjrfContextConfig config;
  mjrf_defaultContextConfig(&config);
  config.native_window = nullptr;
  config.software_rendering = true;
  config.graphics_api = mjGRAPHICS_API_VULKAN;
  return config;
}

void SetupView(Renderer::View* view, Renderer::SceneHandle scene,
               Renderer::RenderTargetHandle target, int width, int height) {
  view->scene = scene;
  view->render_target = target;
  view->post_processing = false;
  view->shadows = false;
  view->reflections = false;
  view->camera.orthographic = true;
  view->camera.pos[0] = 0.0f;
  view->camera.pos[1] = 0.0f;
  view->camera.pos[2] = 1.0f;
  view->camera.forward[0] = 0.0f;
  view->camera.forward[1] = 0.0f;
  view->camera.forward[2] = -1.0f;
  view->camera.up[0] = 0.0f;
  view->camera.up[1] = 1.0f;
  view->camera.up[2] = 0.0f;
  view->camera.frustum_top = 0.0f;
  view->camera.frustum_near = 0.0f;
  view->camera.frustum_far = 1.0f;
  view->camera.frustum_center = width / 2.0f;
  view->camera.frustum_width = width / 2.0f;
  view->camera.frustum_bottom = height;
}

TEST_F(RendererTest, ContextCreationAndDefaults) {
  Renderer renderer(MakeOffscreenConfig());
  ASSERT_NE(renderer.GetContext(), nullptr);
  mjrf_flush(renderer.GetContext());
  EXPECT_FLOAT_EQ(renderer.GetGpuFrameMs(), 0.0f);
}

TEST_F(RendererTest, SceneManagement) {
  Renderer renderer(MakeOffscreenConfig());

  // Unknown scene names and invalid handles return invalid/null results.
  EXPECT_EQ(renderer.GetSceneHandle("main"), Renderer::kInvalidScene);
  EXPECT_EQ(renderer.GetScene(Renderer::kInvalidScene), nullptr);
  EXPECT_EQ(renderer.GetScene(Renderer::SceneHandle{-5}), nullptr);
  EXPECT_EQ(renderer.GetScene(Renderer::SceneHandle{0}), nullptr);
  EXPECT_EQ(renderer.GetScene(Renderer::SceneHandle{100}), nullptr);

  // GetScene lazily creates a scene and assigns a valid handle.
  mjrfScene* scene_a = renderer.GetScene("scene_a");
  ASSERT_NE(scene_a, nullptr);
  Renderer::SceneHandle handle_a = renderer.GetSceneHandle("scene_a");
  EXPECT_NE(handle_a, Renderer::kInvalidScene);
  EXPECT_EQ(renderer.GetScene(handle_a), scene_a);

  // Requesting the same scene name again returns the existing scene instance.
  EXPECT_EQ(renderer.GetScene("scene_a"), scene_a);
  EXPECT_EQ(renderer.GetSceneHandle("scene_a"), handle_a);

  // Creating a second scene produces a distinct handle and pointer.
  mjrfScene* scene_b = renderer.GetScene("scene_b");
  ASSERT_NE(scene_b, nullptr);
  EXPECT_NE(scene_b, scene_a);
  Renderer::SceneHandle handle_b = renderer.GetSceneHandle("scene_b");
  EXPECT_NE(handle_b, Renderer::kInvalidScene);
  EXPECT_NE(handle_b, handle_a);
  EXPECT_EQ(renderer.GetScene(handle_b), scene_b);
  EXPECT_EQ(renderer.GetScene(handle_a), scene_a);
}

TEST_F(RendererTest, RenderTargetLifecycleAndResize) {
  Renderer renderer(MakeOffscreenConfig());

  EXPECT_EQ(renderer.GetRenderTargetHandle("color"),
            Renderer::kInvalidRenderTarget);
  EXPECT_TRUE(renderer.GetPixels(Renderer::kInvalidRenderTarget).empty());
  EXPECT_TRUE(renderer.GetPixels(Renderer::kWindowRenderTarget).empty());
  EXPECT_TRUE(renderer.GetPixels(Renderer::RenderTargetHandle{42}).empty());

  // Resizing invalid or out-of-bounds handles is a safe no-op.
  renderer.ResizeRenderTarget(Renderer::kInvalidRenderTarget, 64, 64);
  renderer.ResizeRenderTarget(Renderer::RenderTargetHandle{42}, 64, 64);
  renderer.ResizeRenderTarget(Renderer::kWindowRenderTarget, 128, 128);
  EXPECT_TRUE(renderer.GetPixels(Renderer::kWindowRenderTarget).empty());

  // Creating a target with positive dimensions immediately allocates buffer.
  mjrfRenderTargetConfig config;
  mjrf_defaultRenderTargetConfig(&config);
  config.width = 64;
  config.height = 32;
  config.color_format = mjPIXEL_FORMAT_RGBA8;
  config.depth_format = mjPIXEL_FORMAT_DEPTH32F;

  Renderer::RenderTargetHandle color_target =
      renderer.CreateRenderTarget("color", config);
  EXPECT_NE(color_target, Renderer::kInvalidRenderTarget);
  EXPECT_EQ(renderer.GetRenderTargetHandle("color"), color_target);
  EXPECT_EQ(renderer.GetPixels(color_target).size(), 64 * 32 * 4);

  // Resizing updates the pixel buffer capacity accordingly.
  renderer.ResizeRenderTarget(color_target, 128, 64);
  EXPECT_EQ(renderer.GetPixels(color_target).size(), 128 * 64 * 4);

  renderer.ResizeRenderTarget(color_target, 16, 8);
  EXPECT_EQ(renderer.GetPixels(color_target).size(), 16 * 8 * 4);

  renderer.ResizeRenderTarget(color_target, 0, 0);
  EXPECT_TRUE(renderer.GetPixels(color_target).empty());

  // Creating a target with zero initial dimensions defers pixel allocation
  // until ResizeRenderTarget is called.
  mjrfRenderTargetConfig deferred_config;
  mjrf_defaultRenderTargetConfig(&deferred_config);
  deferred_config.width = 0;
  deferred_config.height = 0;
  deferred_config.color_format = mjPIXEL_FORMAT_RGB8;

  Renderer::RenderTargetHandle deferred_target =
      renderer.CreateRenderTarget("deferred", deferred_config);
  EXPECT_NE(deferred_target, Renderer::kInvalidRenderTarget);
  EXPECT_NE(deferred_target, color_target);
  EXPECT_TRUE(renderer.GetPixels(deferred_target).empty());

  renderer.ResizeRenderTarget(deferred_target, 32, 24);
  EXPECT_EQ(renderer.GetPixels(deferred_target).size(), 32 * 24 * 3);
}

TEST_F(RendererTest, RenderTargetPixelFormatsAndDuplicateNames) {
  Renderer renderer(MakeOffscreenConfig());

  struct FormatCase {
    const char* name;
    mjrPixelFormat format;
    size_t bytes_per_pixel;
  };
  constexpr FormatCase kFormats[] = {
      {"r8", mjPIXEL_FORMAT_R8, 1},
      {"rgb8", mjPIXEL_FORMAT_RGB8, 3},
      {"rgba8", mjPIXEL_FORMAT_RGBA8, 4},
      {"r32f", mjPIXEL_FORMAT_R32F, 4},
  };

  constexpr int kWidth = 20;
  constexpr int kHeight = 10;
  for (const FormatCase& tc : kFormats) {
    mjrfRenderTargetConfig config;
    mjrf_defaultRenderTargetConfig(&config);
    config.width = kWidth;
    config.height = kHeight;
    config.color_format = tc.format;

    Renderer::RenderTargetHandle handle =
        renderer.CreateRenderTarget(tc.name, config);
    EXPECT_NE(handle, Renderer::kInvalidRenderTarget) << tc.name;
    EXPECT_EQ(renderer.GetRenderTargetHandle(tc.name), handle) << tc.name;
    EXPECT_EQ(renderer.GetPixels(handle).size(),
              kWidth * kHeight * tc.bytes_per_pixel)
        << tc.name;
  }

  // Creating a render target with an already-registered name raises mju_error.
  EXPECT_THAT(MjuErrorMessageFrom(+[](Renderer* r) {
                mjrfRenderTargetConfig cfg;
                mjrf_defaultRenderTargetConfig(&cfg);
                r->CreateRenderTarget("rgb8", cfg);
              })(&renderer),
              HasSubstr("Render target 'rgb8' already exists"));
}

TEST_F(RendererTest, RenderFullscreen) {
  Renderer renderer(MakeOffscreenConfig());

  mjrfContext* ctx = renderer.GetContext();
  UniquePtr<mjrfMesh> mesh = MakeQuadMesh(ctx);
  UniquePtr<mjrfTexture> texture = MakeTexture(ctx, kRed);
  UniquePtr<mjrfRenderable> renderable =
      MakeRenderable(ctx, mesh.get(), texture.get());

  mjrfRenderTargetConfig target_config;
  mjrf_defaultRenderTargetConfig(&target_config);
  target_config.width = 32;
  target_config.height = 32;
  target_config.color_format = mjPIXEL_FORMAT_RGB8;
  target_config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle target =
      renderer.CreateRenderTarget("offscreen", target_config);
  ASSERT_NE(target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene = renderer.GetScene("scene");
  ASSERT_NE(scene, nullptr);
  mjrf_addSceneRenderable(scene, renderable.get());

  Renderer::View views[1];
  SetupView(&views[0], renderer.GetSceneHandle("scene"), target, 32, 32);

  renderer.Render(views);

  std::span<const std::byte> pixels = renderer.GetPixels(target);
  EXPECT_EQ(pixels.size(), target_config.width * target_config.height * 3);
  EXPECT_NE(pixels.data(), nullptr);
  for (int i = 0; i < pixels.size(); i += 3) {
    EXPECT_EQ((uint8_t)pixels[i + 0], kRed[0]);
    EXPECT_EQ((uint8_t)pixels[i + 1], kRed[1]);
    EXPECT_EQ((uint8_t)pixels[i + 2], kRed[2]);
  }
  mjrf_removeSceneRenderable(scene, renderable.get());
}

TEST_F(RendererTest, RenderGrid) {
  Renderer renderer(MakeOffscreenConfig());

  mjrfContext* ctx = renderer.GetContext();

  UniquePtr<mjrfMesh> mesh = MakeQuadMesh(ctx);
  std::vector<UniquePtr<mjrfTexture>> textures;
  textures.push_back(MakeTexture(ctx, kRed));
  textures.push_back(MakeTexture(ctx, kGreen));
  textures.push_back(MakeTexture(ctx, kBlue));
  textures.push_back(MakeTexture(ctx, kWhite));

  std::vector<UniquePtr<mjrfRenderable>> renderables;
  for (int i = 0; i < textures.size(); ++i) {
    renderables.push_back(MakeRenderable(ctx, mesh.get(), textures[i].get()));
  }

  mjrfRenderTargetConfig target_config;
  mjrf_defaultRenderTargetConfig(&target_config);
  target_config.width = 32;
  target_config.height = 32;
  target_config.color_format = mjPIXEL_FORMAT_RGB8;
  target_config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle target =
      renderer.CreateRenderTarget("offscreen", target_config);
  ASSERT_NE(target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene0 = renderer.GetScene("scene0");
  mjrf_addSceneRenderable(scene0, renderables[0].get());
  mjrfScene* scene1 = renderer.GetScene("scene1");
  mjrf_addSceneRenderable(scene1, renderables[1].get());
  mjrfScene* scene2 = renderer.GetScene("scene2");
  mjrf_addSceneRenderable(scene2, renderables[2].get());
  mjrfScene* scene3 = renderer.GetScene("scene3");
  mjrf_addSceneRenderable(scene3, renderables[3].get());

  Renderer::View views[4];

  SetupView(&views[0], renderer.GetSceneHandle("scene0"), target, 32, 32);
  views[0].render_target = target;
  views[0].scene = renderer.GetSceneHandle("scene0");
  views[0].rel_viewport[0] = 0.0f;
  views[0].rel_viewport[1] = 0.0f;
  views[0].rel_viewport[2] = 0.5f;
  views[0].rel_viewport[3] = 0.5f;

  SetupView(&views[1], renderer.GetSceneHandle("scene1"), target, 32, 32);
  views[1].rel_viewport[0] = 0.0f;
  views[1].rel_viewport[1] = 0.5f;
  views[1].rel_viewport[2] = 0.5f;
  views[1].rel_viewport[3] = 0.5f;

  SetupView(&views[2], renderer.GetSceneHandle("scene2"), target, 32, 32);
  views[2].rel_viewport[0] = 0.5f;
  views[2].rel_viewport[1] = 0.0f;
  views[2].rel_viewport[2] = 0.5f;
  views[2].rel_viewport[3] = 0.5f;

  SetupView(&views[3], renderer.GetSceneHandle("scene3"), target, 32, 32);
  views[3].rel_viewport[0] = 0.5f;
  views[3].rel_viewport[1] = 0.5f;
  views[3].rel_viewport[2] = 0.5f;
  views[3].rel_viewport[3] = 0.5f;

  renderer.Render(views);

  std::span<const std::byte> pixels = renderer.GetPixels(target);
  EXPECT_EQ(pixels.size(), target_config.width * target_config.height * 3);
  EXPECT_NE(pixels.data(), nullptr);
  auto get_pixel = [&](int x, int y) {
    int index = (y * target_config.width + x) * 3;
    return std::make_tuple((uint8_t)pixels[index + 0],
                           (uint8_t)pixels[index + 1],
                           (uint8_t)pixels[index + 2]);
  };

  for (int x = 0; x < 16; ++x) {
    for (int y = 0; y < 16; ++y) {
      auto pixel = get_pixel(x, y);
      EXPECT_EQ(std::get<0>(pixel), kGreen[0]);
      EXPECT_EQ(std::get<1>(pixel), kGreen[1]);
      EXPECT_EQ(std::get<2>(pixel), kGreen[2]);
    }
  }
  for (int x = 16; x < 32; ++x) {
    for (int y = 0; y < 16; ++y) {
      auto pixel = get_pixel(x, y);
      EXPECT_EQ(std::get<0>(pixel), kWhite[0]);
      EXPECT_EQ(std::get<1>(pixel), kWhite[1]);
      EXPECT_EQ(std::get<2>(pixel), kWhite[2]);
    }
  }
  for (int x = 0; x < 16; ++x) {
    for (int y = 16; y < 32; ++y) {
      auto pixel = get_pixel(x, y);
      EXPECT_EQ(std::get<0>(pixel), kRed[0]);
      EXPECT_EQ(std::get<1>(pixel), kRed[1]);
      EXPECT_EQ(std::get<2>(pixel), kRed[2]);
    }
  }
  for (int x = 16; x < 32; ++x) {
    for (int y = 16; y < 32; ++y) {
      auto pixel = get_pixel(x, y);
      EXPECT_EQ(std::get<0>(pixel), kBlue[0]);
      EXPECT_EQ(std::get<1>(pixel), kBlue[1]);
      EXPECT_EQ(std::get<2>(pixel), kBlue[2]);
    }
  }
  mjrf_removeSceneRenderable(scene0, renderables[0].get());
  mjrf_removeSceneRenderable(scene1, renderables[1].get());
  mjrf_removeSceneRenderable(scene2, renderables[2].get());
  mjrf_removeSceneRenderable(scene3, renderables[3].get());
}

// a texture with an unsupported number of channels is skipped with a warning,
// rather than an mju_error which would longjmp past the objects being built
TEST_F(RendererTest, UnsupportedTextureChannelsSkipped) {
  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(mj_makeSpec(),
                                                         mj_deleteSpec);
  mjsTexture* texture = mjs_addTexture(spec.get(), nullptr);
  mjs_setName(texture->element, "texture");
  texture->type = mjTEXTURE_2D;
  texture->width = 2;
  texture->height = 2;
  texture->nchannel = 2;
  const std::vector<uint8_t> data(8, 255);
  mjs_setBuffer(texture->data, data.data(), data.size());
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_compile(spec.get(), nullptr), mj_deleteModel);
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->tex_nchannel[0], 2);

  Renderer renderer(MakeOffscreenConfig());
  mock_warning_handler.ExpectWarnings("2-channel");
  ModelResources resources(model.get(), renderer.GetContext());
  EXPECT_EQ(resources.GetTexture(0), nullptr);
}

TEST_F(RendererTest, TextureGeneratesMipmaps) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();

  // 64x64 texture where 1 in every 4x4 block is 240 and all other texels are 0.
  // The box-filtered mip average at level >= 2 is 240 / 16 = 15, whereas
  // sampling level 0 at the 4x4 render target pixel centers (u = 8, 24, 40, 56)
  // would bilinearly sample one 240 texel and three 0 texels (240 / 4 = 60).
  constexpr int kTexSize = 64;
  std::vector<uint8_t> tex_bytes(kTexSize * kTexSize * 3, 0);
  for (int y = 0; y < kTexSize; y += 4) {
    for (int x = 0; x < kTexSize; x += 4) {
      int idx = (y * kTexSize + x) * 3;
      tex_bytes[idx + 0] = 240;
      tex_bytes[idx + 1] = 240;
      tex_bytes[idx + 2] = 240;
    }
  }

  mjrfTextureConfig tex_config;
  mjrf_defaultTextureConfig(&tex_config);
  tex_config.width = kTexSize;
  tex_config.height = kTexSize;
  tex_config.format = mjPIXEL_FORMAT_RGB8;
  tex_config.color_space = mjCOLORSPACE_LINEAR;
  UniquePtr<mjrfTexture> texture = CreateTexture(ctx, tex_config);

  mjrfTextureData tex_data;
  mjrf_defaultTextureData(&tex_data);
  tex_data.bytes = tex_bytes.data();
  tex_data.num_bytes = tex_bytes.size();
  mjrf_uploadTexture(texture.get(), &tex_data);

  UniquePtr<mjrfMesh> mesh = MakeQuadMesh(ctx, /*small=*/true);
  mjrfMaterial material;
  mjrf_defaultMaterial(&material);
  material.metallic = 0.0f;
  material.emissive_texture = texture.get();

  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);
  params.cast_shadow = false;
  params.receive_shadow = false;
  UniquePtr<mjrfRenderable> renderable = CreateRenderable(ctx, params);
  mjrf_setRenderableMesh(renderable.get(), mesh.get(), 0, 0);
  mjrf_setRenderableMaterial(renderable.get(), &material);

  mjrfRenderTargetConfig target_config;
  mjrf_defaultRenderTargetConfig(&target_config);
  target_config.width = 4;
  target_config.height = 4;
  target_config.color_format = mjPIXEL_FORMAT_RGB8;
  target_config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle target =
      renderer.CreateRenderTarget("offscreen", target_config);
  ASSERT_NE(target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene = renderer.GetScene("scene");
  mjrf_addSceneRenderable(scene, renderable.get());

  Renderer::View views[1];
  SetupView(&views[0], renderer.GetSceneHandle("scene"), target, 4, 4);
  views[0].camera.frustum_bottom = 0.0f;
  views[0].camera.frustum_top = 4.0f;
  renderer.Render(views);

  std::span<const std::byte> pixels = renderer.GetPixels(target);
  ASSERT_EQ(pixels.size(), 4 * 4 * 3);
  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_NEAR(static_cast<uint8_t>(pixels[i]), 15, 2);
  }

  // UI (decor) materials sample level 0 directly without mipmapping.
  mjrf_removeSceneRenderable(scene, renderable.get());
  UniquePtr<mjrfRenderable> decor_renderable =
      MakeRenderable(ctx, mesh.get(), texture.get());
  mjrf_addSceneRenderable(scene, decor_renderable.get());
  SetupView(&views[0], renderer.GetSceneHandle("scene"), target, 4, 4);
  renderer.Render(views);

  pixels = renderer.GetPixels(target);
  ASSERT_EQ(pixels.size(), 4 * 4 * 3);
  for (size_t i = 0; i < pixels.size(); ++i) {
    EXPECT_NEAR(static_cast<uint8_t>(pixels[i]), 60, 2);
  }
  mjrf_removeSceneRenderable(scene, decor_renderable.get());
}
// A texture destroyed before Filament has read its data releases the data once,
// after the upload, without touching the destroyed texture.
TEST_F(RendererTest, TextureDestroyedBeforeUploadReleasesData) {
  static int released;
  released = 0;
  {
    Renderer renderer(MakeOffscreenConfig());
    mjrfTextureConfig config;
    mjrf_defaultTextureConfig(&config);
    config.width = 4;
    config.height = 4;
    config.format = mjPIXEL_FORMAT_RGB8;
    UniquePtr<mjrfTexture> texture =
        CreateTexture(renderer.GetContext(), config);

    auto* bytes = new std::vector<uint8_t>(4 * 4 * 3, 255);
    mjrfTextureData data;
    mjrf_defaultTextureData(&data);
    data.bytes = bytes->data();
    data.num_bytes = bytes->size();
    data.user_data = bytes;
    data.release = +[](void* user_data) {
      delete static_cast<std::vector<uint8_t>*>(user_data);
      released++;
    };
    mjrf_uploadTexture(texture.get(), &data);
    texture.reset();
  }
  EXPECT_EQ(released, 1);
}

static UniquePtr<mjrfTexture> MakeSkyboxTexture(mjrfContext* ctx,
                                                const uint8_t color[3]) {
  mjrfTextureConfig config;
  mjrf_defaultTextureConfig(&config);
  config.width = 1;
  config.height = 1;
  config.format = mjPIXEL_FORMAT_RGB8;
  config.color_space = mjCOLORSPACE_LINEAR;
  config.sampler_type = mjTEXTURE_SKYBOX;
  UniquePtr<mjrfTexture> texture = CreateTexture(ctx, config);

  mjrfTextureData data;
  mjrf_defaultTextureData(&data);
  data.bytes = color;
  data.num_bytes = 3;
  mjrf_uploadTexture(texture.get(), &data);
  return texture;
}

TEST_F(RendererTest, HorizonHazeAndSkyboxFlags) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();

  constexpr int kSize = 64;
  mjrfRenderTargetConfig color_cfg;
  mjrf_defaultRenderTargetConfig(&color_cfg);
  color_cfg.width = kSize;
  color_cfg.height = kSize;
  color_cfg.color_format = mjPIXEL_FORMAT_RGB8;
  color_cfg.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle color_target =
      renderer.CreateRenderTarget("color", color_cfg);
  ASSERT_NE(color_target, Renderer::kInvalidRenderTarget);

  mjrfRenderTargetConfig float_cfg;
  mjrf_defaultRenderTargetConfig(&float_cfg);
  float_cfg.width = kSize;
  float_cfg.height = kSize;
  float_cfg.color_format = mjPIXEL_FORMAT_R32F;
  float_cfg.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle float_target =
      renderer.CreateRenderTarget("float", float_cfg);
  ASSERT_NE(float_target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene = renderer.GetScene("scene");
  Renderer::SceneHandle scene_handle = renderer.GetSceneHandle("scene");

  // Blue skybox and pure red haze configured from mjModel.
  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(mj_makeSpec(),
                                                         mj_deleteSpec);
  spec->visual.map.haze = 0.5f;
  spec->visual.rgba.haze[0] = 1.0f;
  spec->visual.rgba.haze[1] = 0.0f;
  spec->visual.rgba.haze[2] = 0.0f;
  spec->visual.rgba.haze[3] = 1.0f;
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_compile(spec.get(), nullptr), mj_deleteModel);
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  mjrf_configureSceneFromModel(scene, model.get());

  UniquePtr<mjrfTexture> skybox = MakeSkyboxTexture(ctx, kBlue);
  mjrf_setSceneSkybox(scene, skybox.get());

  // Infinite green plane at z = 0 (size x=0, y=0).
  mjrfRenderableParams plane_params;
  mjrf_defaultRenderableParams(&plane_params);
  plane_params.cast_shadow = false;
  plane_params.receive_shadow = false;
  UniquePtr<mjrfRenderable> plane = CreateRenderable(ctx, plane_params);
  mjrf_setRenderableGeomMesh(plane.get(), mjGEOM_PLANE, 1, 28, 4);
  const float kInfiniteSize[3] = {0.0f, 0.0f, 1.0f};
  const float kOrigin[3] = {0.0f, 0.0f, 0.0f};
  const float kIdentityRot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjrf_setRenderableSize(plane.get(), kInfiniteSize);
  mjrf_setRenderablePose(plane.get(), kOrigin, kIdentityRot);

  UniquePtr<mjrfTexture> green_tex = MakeTexture(ctx, kGreen);
  mjrfMaterial plane_mat;
  mjrf_defaultMaterial(&plane_mat);
  plane_mat.color[0] = 0.0f;
  plane_mat.color[1] = 1.0f;
  plane_mat.color[2] = 0.0f;
  plane_mat.color[3] = 1.0f;
  plane_mat.emissive = 1.0f;
  plane_mat.emissive_texture = green_tex.get();
  plane_mat.metallic = 0.0f;
  plane_mat.segmentation_id = 42;
  mjrf_setRenderableMaterial(plane.get(), &plane_mat);
  mjrf_addSceneRenderable(scene, plane.get());

  // Perspective camera far from origin in XY (verifies infinite-plane
  // re-centering) at elevation z = 20 looking horizontally along +Y.
  Renderer::View views[1];
  Renderer::View& view = views[0];
  view.scene = scene_handle;
  view.render_target = color_target;
  view.post_processing = false;
  view.shadows = false;
  view.reflections = false;
  view.camera.orthographic = false;
  view.camera.pos[0] = 100.0f;
  view.camera.pos[1] = -200.0f;
  view.camera.pos[2] = 20.0f;
  view.camera.forward[0] = 0.0f;
  view.camera.forward[1] = 1.0f;
  view.camera.forward[2] = 0.0f;
  view.camera.up[0] = 0.0f;
  view.camera.up[1] = 0.0f;
  view.camera.up[2] = 1.0f;
  view.camera.frustum_center = 0.0f;
  view.camera.frustum_width = 0.1f;
  view.camera.frustum_bottom = -0.1f;
  view.camera.frustum_top = 0.1f;
  view.camera.frustum_near = 0.1f;
  view.camera.frustum_far = 100.0f;

  auto pixel_at = [&](int x, int y) {
    std::span<const std::byte> pixels = renderer.GetPixels(color_target);
    int idx = (y * kSize + x) * 3;
    return std::make_tuple(static_cast<uint8_t>(pixels[idx + 0]),
                           static_cast<uint8_t>(pixels[idx + 1]),
                           static_cast<uint8_t>(pixels[idx + 2]));
  };

  // 1. With skybox=true and haze=true:
  //    - y=5: pure blue skybox above the haze cone
  //    - y=36: upper haze stack blending red haze over blue skybox
  //    - y=41: near the plane disk rim (r=70), dominated by red haze
  //    - y=45: lower haze stack blending red haze over green ground plane
  //    - y=60: foreground green plane inside the inner haze ring
  view.skybox = true;
  view.haze = true;
  renderer.Render(views);
  {
    auto [r_sky, g_sky, b_sky] = pixel_at(32, 5);
    EXPECT_EQ(r_sky, 0);
    EXPECT_EQ(g_sky, 0);
    EXPECT_GT(b_sky, 150);

    auto [r_upper, g_upper, b_upper] = pixel_at(32, 36);
    EXPECT_GT(r_upper, 40);
    EXPECT_EQ(g_upper, 0);
    EXPECT_GT(b_upper, 40);

    auto [r_rim, g_rim, b_rim] = pixel_at(32, 41);
    EXPECT_GT(r_rim, 140);
    EXPECT_LT(g_rim, 40);
    EXPECT_LT(b_rim, 40);

    auto [r_lower, g_lower, b_lower] = pixel_at(32, 45);
    EXPECT_GT(r_lower, 40);
    EXPECT_GT(g_lower, 40);
    EXPECT_EQ(b_lower, 0);

    auto [r_fg, g_fg, b_fg] = pixel_at(32, 60);
    EXPECT_EQ(r_fg, 0);
    EXPECT_EQ(g_fg, 255);
    EXPECT_EQ(b_fg, 0);
  }

  // 2. With skybox=true and haze=false: sharp transition at the plane disk rim,
  //    with zero red haze in either skybox or plane regions.
  view.haze = false;
  renderer.Render(views);
  {
    auto [r_upper, g_upper, b_upper] = pixel_at(32, 36);
    EXPECT_EQ(r_upper, 0);
    EXPECT_EQ(g_upper, 0);
    EXPECT_GT(b_upper, 150);

    auto [r_beyond_disk, g_beyond_disk, b_beyond_disk] = pixel_at(32, 39);
    EXPECT_EQ(r_beyond_disk, 0);
    EXPECT_EQ(g_beyond_disk, 0);
    EXPECT_GT(b_beyond_disk, 150);

    auto [r_lower, g_lower, b_lower] = pixel_at(32, 45);
    EXPECT_EQ(r_lower, 0);
    EXPECT_EQ(g_lower, 255);
    EXPECT_EQ(b_lower, 0);
  }

  // 3. With skybox=false (even if haze=true): skybox and haze are both hidden.
  view.skybox = false;
  view.haze = true;
  renderer.Render(views);
  {
    auto [r_sky, g_sky, b_sky] = pixel_at(32, 5);
    EXPECT_EQ(r_sky, 0);
    EXPECT_EQ(g_sky, 0);
    EXPECT_EQ(b_sky, 0);

    auto [r_rim, g_rim, b_rim] = pixel_at(32, 41);
    EXPECT_EQ(r_rim, 0);
    EXPECT_EQ(b_rim, 0);
  }

  // 4. Depth, segmentation, and wireframe passes suppress skybox and haze.
  view.skybox = true;
  view.haze = true;
  view.draw_mode = mjDRAW_MODE_WIREFRAME;
  renderer.Render(views);
  {
    auto [r_sky, g_sky, b_sky] = pixel_at(32, 5);
    EXPECT_EQ(r_sky, 0);
    EXPECT_EQ(g_sky, 0);
    EXPECT_EQ(b_sky, 0);

    auto [r_rim, g_rim, b_rim] = pixel_at(32, 41);
    EXPECT_EQ(r_rim, 0);
    EXPECT_EQ(b_rim, 0);
  }

  view.render_target = float_target;
  for (mjrDrawMode mode : {mjDRAW_MODE_DEPTH, mjDRAW_MODE_SEGMENTATION_BY_ID}) {
    view.draw_mode = mode;
    view.haze = false;
    renderer.Render(views);
    std::span<const std::byte> no_haze_bytes = renderer.GetPixels(float_target);
    std::vector<std::byte> expected(no_haze_bytes.begin(), no_haze_bytes.end());

    view.haze = true;
    renderer.Render(views);
    std::span<const std::byte> with_haze_bytes =
        renderer.GetPixels(float_target);
    EXPECT_EQ(
        std::vector<std::byte>(with_haze_bytes.begin(), with_haze_bytes.end()),
        expected);
  }
  view.draw_mode = mjDRAW_MODE_DEFAULT;
  view.render_target = color_target;

  // 5. Opaque foreground geometry in front of the haze cone occludes haze.
  static constexpr uint8_t kCyan[3] = {0, 255, 255};
  UniquePtr<mjrfTexture> cyan_tex = MakeTexture(ctx, kCyan);
  UniquePtr<mjrfRenderable> blocker = CreateRenderable(ctx, plane_params);
  mjrf_setRenderableGeomMesh(blocker.get(), mjGEOM_BOX, 1, 28, 4);
  const float kBoxSize[3] = {5.0f, 1.0f, 5.0f};
  const float kBoxPos[3] = {100.0f, -180.0f, 14.3f};
  mjrf_setRenderableSize(blocker.get(), kBoxSize);
  mjrf_setRenderablePose(blocker.get(), kBoxPos, kIdentityRot);
  mjrfMaterial box_mat = plane_mat;
  box_mat.color[0] = 0.0f;
  box_mat.color[1] = 1.0f;
  box_mat.color[2] = 1.0f;
  box_mat.emissive_texture = cyan_tex.get();
  mjrf_setRenderableMaterial(blocker.get(), &box_mat);
  mjrf_addSceneRenderable(scene, blocker.get());

  view.skybox = true;
  view.haze = true;
  renderer.Render(views);
  {
    auto [r_box, g_box, b_box] = pixel_at(32, 41);
    EXPECT_EQ(r_box, 0);
    EXPECT_EQ(g_box, 255);
    EXPECT_EQ(b_box, 255);
  }
  mjrf_removeSceneRenderable(scene, blocker.get());

  // 6. Camera below the infinite plane (elevation < 0) does not render haze.
  view.camera.pos[2] = -5.0f;
  renderer.Render(views);
  std::span<const std::byte> below_pixels = renderer.GetPixels(color_target);
  for (size_t i = 0; i < below_pixels.size(); i += 3) {
    EXPECT_EQ(static_cast<uint8_t>(below_pixels[i]), 0);
  }
}

TEST_F(RendererTest, DepthProjectionUsesFiniteFar) {
  constexpr int kSize = 33;
  constexpr float kNear = 0.1f;
  constexpr float kFar = 10.0f;

  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();

  mjrfRenderTargetConfig depth_cfg;
  mjrf_defaultRenderTargetConfig(&depth_cfg);
  depth_cfg.width = kSize;
  depth_cfg.height = kSize;
  depth_cfg.color_format = mjPIXEL_FORMAT_R32F;
  depth_cfg.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle depth_target =
      renderer.CreateRenderTarget("depth", depth_cfg);
  ASSERT_NE(depth_target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene = renderer.GetScene("scene");
  Renderer::SceneHandle scene_handle = renderer.GetSceneHandle("scene");

  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);
  params.cast_shadow = false;
  params.receive_shadow = false;

  UniquePtr<mjrfTexture> white_tex = MakeTexture(ctx, kWhite);
  mjrfMaterial mat;
  mjrf_defaultMaterial(&mat);
  mat.emissive = 1.0f;
  mat.emissive_texture = white_tex.get();

  const float kIdentityRot[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  // Center box whose top face sits at world z = 1.5 (planar depth 1.5 from
  // z=3).
  UniquePtr<mjrfRenderable> near_box = CreateRenderable(ctx, params);
  mjrf_setRenderableGeomMesh(near_box.get(), mjGEOM_BOX, 1, 28, 4);
  const float kNearSize[3] = {0.2f, 0.2f, 0.5f};
  const float kNearPos[3] = {0.0f, 0.0f, 1.0f};
  mjrf_setRenderableSize(near_box.get(), kNearSize);
  mjrf_setRenderablePose(near_box.get(), kNearPos, kIdentityRot);
  mjrf_setRenderableMaterial(near_box.get(), &mat);
  mjrf_addSceneRenderable(scene, near_box.get());

  // Off-axis box spanning camera depth [7.0, 20.0] at x in [6.0, 26.0]: its
  // front face is inside frustum_far (not CPU-culled), while its -X side face
  // crosses frustum_far = 10.0 and must be hardware-clipped past z_far.
  UniquePtr<mjrfRenderable> straddling_box = CreateRenderable(ctx, params);
  mjrf_setRenderableGeomMesh(straddling_box.get(), mjGEOM_BOX, 1, 28, 4);
  const float kStraddleSize[3] = {10.0f, 20.0f, 6.5f};
  const float kStraddlePos[3] = {16.0f, 0.0f, -10.5f};
  mjrf_setRenderableSize(straddling_box.get(), kStraddleSize);
  mjrf_setRenderablePose(straddling_box.get(), kStraddlePos, kIdentityRot);
  mjrf_setRenderableMaterial(straddling_box.get(), &mat);
  mjrf_addSceneRenderable(scene, straddling_box.get());

  // Backdrop box wholly beyond frustum_far (top face at planar depth 11.5).
  UniquePtr<mjrfRenderable> beyond_far = CreateRenderable(ctx, params);
  mjrf_setRenderableGeomMesh(beyond_far.get(), mjGEOM_BOX, 1, 28, 4);
  const float kBeyondSize[3] = {20.0f, 20.0f, 0.5f};
  const float kBeyondPos[3] = {0.0f, 0.0f, -9.0f};
  mjrf_setRenderableSize(beyond_far.get(), kBeyondSize);
  mjrf_setRenderablePose(beyond_far.get(), kBeyondPos, kIdentityRot);
  mjrf_setRenderableMaterial(beyond_far.get(), &mat);
  mjrf_addSceneRenderable(scene, beyond_far.get());

  auto depth_at = [&](int x, int y) {
    std::span<const std::byte> bytes = renderer.GetPixels(depth_target);
    float value = 0.0f;
    std::memcpy(&value, bytes.data() + (y * kSize + x) * sizeof(float),
                sizeof(float));
    return value;
  };

  Renderer::View views[1];
  Renderer::View& view = views[0];
  view.scene = scene_handle;
  view.render_target = depth_target;
  view.draw_mode = mjDRAW_MODE_DEPTH;
  view.post_processing = false;
  view.shadows = false;
  view.reflections = false;
  view.camera.pos[0] = 0.0f;
  view.camera.pos[1] = 0.0f;
  view.camera.pos[2] = 3.0f;
  view.camera.forward[0] = 0.0f;
  view.camera.forward[1] = 0.0f;
  view.camera.forward[2] = -1.0f;
  view.camera.up[0] = 0.0f;
  view.camera.up[1] = 1.0f;
  view.camera.up[2] = 0.0f;
  view.camera.frustum_center = 0.0f;
  view.camera.frustum_near = kNear;
  view.camera.frustum_far = kFar;

  // Perspective camera (90-degree FOV): reversed-Z finite-frustum mapping
  // d = near * (far - z) / (z * (far - near)).
  view.camera.orthographic = false;
  view.camera.frustum_width = 0.1f;
  view.camera.frustum_bottom = -0.1f;
  view.camera.frustum_top = 0.1f;
  renderer.Render(views);

  constexpr float kBoxDepth = 1.5f;
  constexpr float kExpectedPerspCenter =
      kNear * (kFar - kBoxDepth) / (kBoxDepth * (kFar - kNear));
  constexpr float kStraddleFrontDepth = 7.0f;
  constexpr float kExpectedStraddleFront =
      kNear * (kFar - kStraddleFrontDepth) /
      (kStraddleFrontDepth * (kFar - kNear));
  EXPECT_NEAR(depth_at(kSize / 2, kSize / 2), kExpectedPerspCenter, 1e-4f);
  EXPECT_NEAR(depth_at(31, kSize / 2), kExpectedStraddleFront, 1e-4f);
  EXPECT_NEAR(depth_at(24, kSize / 2), 0.0f, 1e-6f);
  EXPECT_NEAR(depth_at(0, 0), 0.0f, 1e-6f);

  // Orthographic camera: linear reversed-Z mapping d = (far - z) / (far -
  // near).
  view.camera.orthographic = true;
  view.camera.frustum_width = 1.0f;
  view.camera.frustum_bottom = -1.0f;
  view.camera.frustum_top = 1.0f;
  renderer.Render(views);

  constexpr float kExpectedOrthoCenter = (kFar - kBoxDepth) / (kFar - kNear);
  EXPECT_NEAR(depth_at(kSize / 2, kSize / 2), kExpectedOrthoCenter, 1e-4f);
  EXPECT_NEAR(depth_at(0, 0), 0.0f, 1e-6f);
}

TEST_F(RendererTest, WaitForFrameFlushesSubmittedFrame) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();
  mjrfScene* scene = renderer.GetScene("scene");

  mjrfTextureConfig tex_config;
  mjrf_defaultTextureConfig(&tex_config);
  tex_config.width = 4;
  tex_config.height = 4;
  tex_config.format = mjPIXEL_FORMAT_RGB8;
  UniquePtr<mjrfTexture> texture = CreateTexture(ctx, tex_config);

  std::vector<uint8_t> tex_bytes(4 * 4 * 3, 255);
  int released = 0;
  mjrfTextureData tex_data;
  mjrf_defaultTextureData(&tex_data);
  tex_data.bytes = tex_bytes.data();
  tex_data.num_bytes = tex_bytes.size();
  tex_data.user_data = &released;
  tex_data.release =
      +[](void* user_data) { (*static_cast<int*>(user_data))++; };
  mjrf_uploadTexture(texture.get(), &tex_data);

  mjrfRenderRequest req;
  mjrf_defaultRenderRequest(&req);
  req.scene = scene;
  req.target = nullptr;
  req.viewport = {0, 0, 16, 16};
  req.post_processing = false;
  req.shadows = false;
  req.reflections = false;
  req.camera.orthographic = true;
  req.camera.pos[2] = 1.0f;
  req.camera.forward[2] = -1.0f;
  req.camera.up[1] = 1.0f;
  req.camera.frustum_near = 0.0f;
  req.camera.frustum_far = 1.0f;
  req.camera.frustum_center = 8.0f;
  req.camera.frustum_width = 8.0f;
  req.camera.frustum_bottom = 16.0f;

  mjrfFrameHandle frame = mjrf_render(ctx, &req, 1, nullptr, 0);
  EXPECT_GT(frame, 0);

  mjrf_waitForFrame(ctx, frame);
  EXPECT_EQ(released, 1);

  // Waiting again on an already-completed or zero handle is a safe no-op.
  mjrf_waitForFrame(ctx, frame);
  mjrf_waitForFrame(ctx, 0);
  EXPECT_EQ(released, 1);
}

TEST_F(RendererTest, MeshReuploadReleasesResources) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();
  UniquePtr<mjrfMesh> mesh = MakeQuadMesh(ctx);

  static const float vertices[] = {
      -256.0f, -256.0f, 256.0f, -256.0f, -256.0f, 256.0f, 256.0f, 256.0f,
  };
  static const float uvs[] = {
      0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
  };
  static const uint8_t colors[] = {
      255, 255, 255, 255, 255, 255, 255, 255,
      255, 255, 255, 255, 255, 255, 255, 255,
  };
  static const uint16_t indices[] = {0, 1, 2, 1, 3, 2};

  UniquePtr<mjrfTexture> texture = MakeTexture(ctx, kRed);
  UniquePtr<mjrfRenderable> renderable =
      MakeRenderable(ctx, mesh.get(), texture.get());

  mjrfRenderTargetConfig target_config;
  mjrf_defaultRenderTargetConfig(&target_config);
  target_config.width = 32;
  target_config.height = 32;
  target_config.color_format = mjPIXEL_FORMAT_RGB8;
  target_config.depth_format = mjPIXEL_FORMAT_DEPTH32F;
  Renderer::RenderTargetHandle target =
      renderer.CreateRenderTarget("offscreen_mesh", target_config);
  ASSERT_NE(target, Renderer::kInvalidRenderTarget);

  mjrfScene* scene = renderer.GetScene("scene");
  ASSERT_NE(scene, nullptr);
  mjrf_addSceneRenderable(scene, renderable.get());

  Renderer::View views[1];
  SetupView(&views[0], renderer.GetSceneHandle("scene"), target, 32, 32);

  int release_count_1 = 0;
  mjrfMeshData data1;
  mjrf_defaultMeshData(&data1);
  data1.num_vertices = 4;
  data1.vertices[0] = vertices;
  data1.vertices[1] = uvs;
  data1.vertices[2] = colors;
  data1.num_indices = 6;
  data1.indices = indices;
  data1.release = +[](void* user) { (*static_cast<int*>(user))++; };
  data1.user_data = &release_count_1;
  mjrf_uploadMesh(mesh.get(), &data1);

  renderer.Render(views);
  EXPECT_EQ(release_count_1, 1);

  int release_count_2 = 0;
  mjrfMeshData data2;
  mjrf_defaultMeshData(&data2);
  data2.num_vertices = 4;
  data2.vertices[0] = vertices;
  data2.vertices[1] = uvs;
  data2.vertices[2] = colors;
  data2.num_indices = 6;
  data2.indices = indices;
  data2.release = +[](void* user) { (*static_cast<int*>(user))++; };
  data2.user_data = &release_count_2;
  mjrf_uploadMesh(mesh.get(), &data2);

  renderer.Render(views);
  EXPECT_EQ(release_count_2, 1);

  mjrf_removeSceneRenderable(scene, renderable.get());
}

// Submits an empty frame and blocks until Filament has executed every queued
// command and dispatched the resulting BufferDescriptor release callbacks.
static void RenderEmptyFrameAndWait(mjrfContext* ctx, mjrfScene* scene) {
  mjrfRenderRequest req;
  mjrf_defaultRenderRequest(&req);
  req.scene = scene;
  req.target = nullptr;
  req.viewport = {0, 0, 16, 16};
  req.post_processing = false;
  req.shadows = false;
  req.reflections = false;
  req.camera.orthographic = true;
  req.camera.pos[2] = 1.0f;
  req.camera.forward[2] = -1.0f;
  req.camera.up[1] = 1.0f;
  req.camera.frustum_near = 0.0f;
  req.camera.frustum_far = 1.0f;
  req.camera.frustum_center = 8.0f;
  req.camera.frustum_width = 8.0f;
  req.camera.frustum_bottom = 16.0f;
  mjrfFrameHandle frame = mjrf_render(ctx, &req, 1, nullptr, 0);
  mjrf_waitForFrame(ctx, frame);
}

// Regression test for a use-after-free: mjrf_uploadMesh only queues the vertex
// and index buffers on Filament's command stream, which references the caller's
// memory (and Mesh-internal index/orientation scratch buffers) until the
// backend copies it. Destroying the mesh before that happens (e.g. when
// ModelResources re-creates meshes) used to run data.release and free the
// internal buffers synchronously in ~Mesh, so the backend later read freed
// memory. The release must be deferred until Filament has consumed the buffers.
TEST_F(RendererTest, MeshDestroyedBeforeFlushDefersRelease) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();

  // Separate normals and implicit indices exercise every internal allocation:
  // orientations, the index sequence and the wireframe indices.
  mjrfMeshConfig config;
  mjrf_defaultMeshConfig(&config);
  config.num_attributes = 2;
  config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
  config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
  config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_NORMAL;
  config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
  config.max_vertices = 3;
  config.max_indices = 3;
  config.interleaved = false;
  config.index_type = mjINDEX_TYPE_U16;
  config.primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;
  UniquePtr<mjrfMesh> mesh = CreateMesh(ctx, config);

  struct HeapPayload {
    std::vector<float> positions = {
        -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f,
    };
    std::vector<float> normals = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
    };
    int* release_count = nullptr;
  };

  int release_count = 0;
  auto* payload = new HeapPayload{.release_count = &release_count};

  mjrfMeshData data;
  mjrf_defaultMeshData(&data);
  data.num_vertices = 3;
  data.vertices[0] = payload->positions.data();
  data.vertices[1] = payload->normals.data();
  data.num_indices = 3;
  data.indices = nullptr;
  data.user_data = payload;
  data.release = +[](void* user) {
    auto* p = static_cast<HeapPayload*>(user);
    (*p->release_count)++;
    delete p;
  };
  EXPECT_EQ(release_count, 0);
  mjrf_uploadMesh(mesh.get(), &data);
  EXPECT_EQ(release_count, 0);
  mesh.reset();

  // Filament has not executed the queued uploads yet, so the payload must
  // still be alive.
  EXPECT_EQ(release_count, 0);

  RenderEmptyFrameAndWait(ctx, renderer.GetScene("scene"));
  EXPECT_EQ(release_count, 1);
}

// Two uploads queued on the same mesh before a flush each own their buffers
// until Filament consumes them; the second upload must not drop or release the
// first upload's data early.
TEST_F(RendererTest, MeshReuploadBeforeFlushReleasesBothUploads) {
  Renderer renderer(MakeOffscreenConfig());
  mjrfContext* ctx = renderer.GetContext();
  UniquePtr<mjrfMesh> mesh = MakeQuadMesh(ctx);

  struct HeapQuad {
    std::vector<float> vertices = {
        -256.0f, -256.0f, 256.0f, -256.0f, -256.0f, 256.0f, 256.0f, 256.0f,
    };
    std::vector<float> uvs = {
        0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
    };
    std::vector<uint8_t> colors = std::vector<uint8_t>(16, 255);
    std::vector<uint16_t> indices = {0, 1, 2, 1, 3, 2};
    int* release_count = nullptr;
  };

  auto upload = [&](int* count) {
    auto* quad = new HeapQuad{.release_count = count};
    mjrfMeshData data;
    mjrf_defaultMeshData(&data);
    data.num_vertices = 4;
    data.vertices[0] = quad->vertices.data();
    data.vertices[1] = quad->uvs.data();
    data.vertices[2] = quad->colors.data();
    data.num_indices = 6;
    data.indices = quad->indices.data();
    data.user_data = quad;
    data.release = +[](void* user) {
      auto* q = static_cast<HeapQuad*>(user);
      (*q->release_count)++;
      delete q;
    };
    mjrf_uploadMesh(mesh.get(), &data);
  };

  int release_count_1 = 0;
  int release_count_2 = 0;
  upload(&release_count_1);
  upload(&release_count_2);
  EXPECT_EQ(release_count_1, 0);
  EXPECT_EQ(release_count_2, 0);

  RenderEmptyFrameAndWait(ctx, renderer.GetScene("scene"));
  EXPECT_EQ(release_count_1, 1);
  EXPECT_EQ(release_count_2, 1);
}
}  // namespace
}  // namespace mujoco
