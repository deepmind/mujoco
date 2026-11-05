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
#include "studio/hal/filament_renderer.h"

#include <cstddef>
#include <memory>
#include <vector>

#include "third_party/mujoco/google/gfx/opengl_dynamic_loader.h"
#include <mujoco/mujoco.h>
#include "studio/hal/graphics_mode.h"
#include "studio/sim/model_holder.h"

#include "testing/base/public/gunit.h"

namespace mujoco::studio {
namespace {

class RendererTest : public ::testing::Test {
 public:
  void SetUp() {
    mjSpec* spec = mj_makeSpec();
    // Set a clear color.
    const double clear_color[] = {1.0, 1.0, 1.0, 1.0};
    mjsNumeric* numeric = mjs_addNumeric(spec);
    mjs_setName(numeric->element, "filament.clear_color");
    numeric->size = 4;
    mjs_setDouble(numeric->data, clear_color, numeric->size);
    holder_ = ModelHolder::FromSpec(spec);
    holder_->model()->vis.global.offwidth = width_;
    holder_->model()->vis.global.offheight = height_;
  }

  void Test();

  int width_ = 2;
  int height_ = 2;
  std::unique_ptr<ModelHolder> holder_;
};

TEST_F(RendererTest, OpengGlSoftware) {
  FilamentRenderer renderer(nullptr, GraphicsMode::FilamentOpenGlSoftware);
  renderer.Init(holder_->model());

  gl::DriverType driver_type = gl::GetLoadedDriverType();
  ASSERT_EQ(driver_type, gl::DriverType::kOsMesa);

  std::vector<std::byte> pixels(width_ * height_ * 3);
  renderer.Sync(holder_->model(), holder_->data(), nullptr, nullptr, nullptr, width_, height_);
  renderer.Submit(width_, height_, pixels);
  // We set the clear color to white, but we don't know the exact color due to
  // post processing, but it should definitely not be black.
  for (int i = 0; i < pixels.size(); i += 3) {
    EXPECT_NE((int)pixels[i + 0], 0);
    EXPECT_NE((int)pixels[i + 1], 0);
    EXPECT_NE((int)pixels[i + 2], 0);
  }
}

TEST_F(RendererTest, OpengGlHeadless) {
  FilamentRenderer renderer(nullptr, GraphicsMode::FilamentOpenGlHeadless);
  renderer.Init(holder_->model());

  gl::DriverType driver_type = gl::GetLoadedDriverType();
#if TEST_HAS_GPU
  ASSERT_EQ(driver_type, gl::DriverType::kEgl);
#else
  ASSERT_EQ(driver_type, gl::DriverType::kOsMesa);
#endif

  std::vector<std::byte> pixels(width_ * height_ * 3);
  renderer.Sync(holder_->model(), holder_->data(), nullptr, nullptr, nullptr, width_, height_);
  renderer.Submit(width_, height_, pixels);
  // We set the clear color to white, but we don't know the exact color due to
  // post processing, but it should definitely not be black.
  for (int i = 0; i < pixels.size(); i += 3) {
    EXPECT_NE((int)pixels[i + 0], 0);
    EXPECT_NE((int)pixels[i + 1], 0);
    EXPECT_NE((int)pixels[i + 2], 0);
  }
}

TEST_F(RendererTest, VulkanSoftware) {
  FilamentRenderer renderer(nullptr, GraphicsMode::FilamentVulkanSoftware);
  renderer.Init(holder_->model());

  std::vector<std::byte> pixels(width_ * height_ * 3);
  renderer.Sync(holder_->model(), holder_->data(), nullptr, nullptr, nullptr, width_, height_);
  renderer.Submit(width_, height_, pixels);
  // We set the clear color to white, but we don't know the exact color due to
  // post processing, but it should definitely not be black.
  for (int i = 0; i < pixels.size(); i += 3) {
    EXPECT_NE((int)pixels[i + 0], 0);
    EXPECT_NE((int)pixels[i + 1], 0);
    EXPECT_NE((int)pixels[i + 2], 0);
  }
}

TEST_F(RendererTest, VulkanHeadless) {
  FilamentRenderer renderer(nullptr, GraphicsMode::FilamentVulkan);
  renderer.Init(holder_->model());

  std::vector<std::byte> pixels(width_ * height_ * 3);
  renderer.Sync(holder_->model(), holder_->data(), nullptr, nullptr, nullptr, width_, height_);
  renderer.Submit(width_, height_, pixels);
  // We set the clear color to white, but we don't know the exact color due to
  // post processing, but it should definitely not be black.
  for (int i = 0; i < pixels.size(); i += 3) {
    EXPECT_NE((int)pixels[i + 0], 0);
    EXPECT_NE((int)pixels[i + 1], 0);
    EXPECT_NE((int)pixels[i + 2], 0);
  }
}

}  // namespace
}  // namespace mujoco::studio
