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

#ifndef MUJOCO_STUDIO_HAL_WINDOW_H_
#define MUJOCO_STUDIO_HAL_WINDOW_H_

#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include <SDL_render.h>
#include <SDL_video.h>
#include "studio/hal/graphics_mode.h"

namespace mujoco::studio {

// Platform-independent window abstraction using SDL2.
//
// Initializes the SDL2 and ImGui libraries, creates/owns the native window, and
// handles events from the window.
class Window {
 public:
  struct Config {
    GraphicsMode gfx_mode = GraphicsMode::FilamentVulkan;
    bool load_fonts = true;
  };

  Window(std::string_view title, int width, int height, Config config);
  ~Window();

  Window(const Window&) = delete;
  Window& operator=(const Window&) = delete;

  // The status of the window.
  enum Status {
    kRunning,
    kQuitting,
  };

  // Processes all pendings window events and prepares ImGui for input handling
  // and GUI rendering. Returns the status of the window.
  Status NewFrame();

  // Finalizes ImGui input handling. Must call NewFrame first.
  void EndFrame();

  // Swaps and presents the window buffer.
  void Present(std::span<const std::byte> pixels = {});

  // Sets the title of the window.
  void SetTitle(std::string_view title);

  // Returns information related to the current size of the window.
  int GetWidth() const { return width_; }
  int GetHeight() const { return height_; }
  float GetScale() const { return scale_; }
  float GetAspectRatio() const {
    return height_ > 0 ? static_cast<float>(width_) / static_cast<float>(height_) : 1.0f;
  }

  // Returns the path to a file that was dropped on the window. Once called,
  // the value will be cleared until the next time a file is dropped.
  std::string GetDropFile();

  // Returns the graphics configuration of the window.
  GraphicsMode GetGraphicsMode() const;

  // Resizes the window to the given width and height.
  void Resize(int width, int height);

  // Enables window resizing.
  void EnableWindowResizing();

  // Disables window resizing.
  void DisableWindowResizing();

  // Returns the handle to the underlying native window.
  void* GetNativeWindowHandle() { return native_window_; }

 private:
  void InitOffscreenEglContext();

  int width_ = 0;
  int height_ = 0;
  float scale_ = 1.0f;
  Config config_;
  void* native_window_ = nullptr;
  SDL_Window* sdl_window_ = nullptr;
  SDL_Renderer* sdl_renderer_ = nullptr;
  bool should_exit_ = false;
  std::string drop_file_;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_HAL_WINDOW_H_
