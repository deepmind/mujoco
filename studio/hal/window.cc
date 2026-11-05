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

#include "studio/hal/window.h"

#include <algorithm>
#include <cstddef>
#include <span>
#include <string>
#include <string_view>

#include <SDL.h>
#include <SDL_error.h>
#include <SDL_events.h>
#include <SDL_hints.h>
#include <SDL_render.h>
#include <SDL_stdinc.h>
#include <SDL_surface.h>
#include <SDL_syswm.h>
#include <SDL_version.h>
#include <SDL_video.h>
#include <backends/imgui_impl_sdl2.h>
#include <imgui.h>
#include <mujoco/mujoco.h>
#include "studio/hal/graphics_mode.h"
#include "studio/ux/fonts.h"

// Because X11/Xlib.h defines Status.
#ifdef Status
  #undef Status
#endif

#if defined(__APPLE__)
extern void* GetNativeWindowOsx(void* window);
#endif

#if defined(__EMSCRIPTEN__)
  #include <emscripten/em_js.h>

// clang-format off
EM_JS(bool, PlatformIsApple, (), {
  if (typeof navigator === "undefined") {
    return false;
  }
  const platform = (navigator.userAgentData && navigator.userAgentData.platform) ||
                   navigator.platform ||
                   "";
  if (/Mac|iPhone|iPod|iPad/i.test(platform)) {
    return true;
  }
  return /Macintosh|Mac OS X/i.test(navigator.userAgent || "");
});
// clang-format on
#endif

namespace mujoco::studio {

static void InitImGui(SDL_Window* window, float content_scale, bool load_fonts) {
  ImGui::CreateContext();

  ImGuiIO& io = ImGui::GetIO();
  io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
  io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
  io.IniFilename = nullptr;
  io.ConfigDpiScaleFonts = true;
  io.ConfigDpiScaleViewports = true;
  // Match SDL2's double-click thresholds so trackpad double-taps register;
  // MouseSingleClickDelay (0.50s default) must exceed MouseDoubleClickTime.
  io.MouseDoubleClickTime = 0.50f;
  io.MouseDoubleClickMaxDist = 32.0f * content_scale;
  io.MouseSingleClickDelay = 0.60f;
#if defined(__EMSCRIPTEN__)
  if (PlatformIsApple()) {
    io.ConfigMacOSXBehaviors = true;
  }
#endif
  ImGui::StyleColorsDark();
  ImGui_ImplSDL2_InitForOther(window);

  ImGuiStyle& style = ImGui::GetStyle();
  style.ScaleAllSizes(content_scale);
  style.FontScaleDpi = content_scale;

  if (load_fonts) {
    AddStudioFonts(LoadFontResource);
  }
}

Window::Window(std::string_view title, int width, int height, Config config)
    : width_(width), height_(height), config_(config) {
  if (IsHeadless(config_.gfx_mode)) {
    SDL_SetHint(SDL_HINT_RENDER_DRIVER, "software");
    SDL_SetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION, "0");
  } else {
    SDL_SetHint(SDL_HINT_FRAMEBUFFER_ACCELERATION, "1");
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 16);
  }

  if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS) != 0) {
    mju_error("Error initializing SDL: %s", SDL_GetError());
  }

  int window_flags = SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
  if (IsHeadless(config_.gfx_mode)) {
    // No additional window flags needed for headless rendering.
  } else if (IsVulkan(config_.gfx_mode)) {
    window_flags |= SDL_WINDOW_VULKAN;
  } else if (IsWebGl(config_.gfx_mode)) {
    window_flags |= SDL_WINDOW_OPENGL;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_ES);
  } else if (IsOpenGl(config_.gfx_mode)) {
    // On Windows, SDL_WINDOW_OPENGL makes SDL call SetPixelFormat on the HWND,
    // which can only be set once and would block Filament's PlatformWGL.
#if !defined(__WIN32__)
    window_flags |= SDL_WINDOW_OPENGL;
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 0);
#endif
  } else {
    mju_error("Unsupported window config: %d", config_.gfx_mode);
  }

  const float content_scale = std::max(1.0f, ImGui_ImplSDL2_GetContentScaleForDisplay(0));
  width_ = width * content_scale;
  height_ = height * content_scale;
  sdl_window_ = SDL_CreateWindow(title.data(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
                                 width_, height_, window_flags);
  if (!sdl_window_) {
    mju_error("Error creating window: %s", SDL_GetError());
  }

  InitImGui(sdl_window_, content_scale, config.load_fonts);

  // Filament (except WebGL) manages its own swap chain including when to swap.
  // In all other cases, we'll use SDL to manage the swap chain.
  if (config_.gfx_mode == GraphicsMode::FilamentWebGl) {
    SDL_GLContext gl_context = SDL_GL_CreateContext(sdl_window_);
    SDL_GL_MakeCurrent(sdl_window_, gl_context);
  }

  // In headless mode, we'll render to a texture and then blit the texture onto
  // the window surface. In this case, we'll use SDL's software renderer to
  // perform the blitting. Since we're in charge of the window, we don't need
  // to get the native window handle for the renderer.
  if (IsHeadless(config_.gfx_mode)) {
    sdl_renderer_ = SDL_CreateRenderer(sdl_window_, -1, SDL_RENDERER_SOFTWARE);
  } else {
    SDL_SysWMinfo wmi;
    SDL_VERSION(&wmi.version);
    SDL_GetWindowWMInfo(sdl_window_, &wmi);

#if defined(__linux__)
    native_window_ = reinterpret_cast<void*>(wmi.info.x11.window);
#elif defined(__WIN32__)
    native_window_ = reinterpret_cast<void*>(wmi.info.win.window);
#elif defined(__APPLE__)
    native_window_ = GetNativeWindowOsx(reinterpret_cast<void*>(wmi.info.cocoa.window));
#endif
  }

  int drawable_width = width_;
  int drawable_height = height_;
  SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
  scale_ = (float)drawable_width / (float)width_;
}

Window::~Window() {
  ImGui_ImplSDL2_Shutdown();
  if (ImGui::GetCurrentContext() != nullptr) {
    ImGui::DestroyContext(nullptr);
  }
  SDL_DestroyWindow(sdl_window_);
  SDL_Quit();
}

void Window::SetTitle(std::string_view title) { SDL_SetWindowTitle(sdl_window_, title.data()); }

void Window::Resize(int width, int height) {
  SDL_SetWindowSize(sdl_window_, width, height);
  SDL_SetWindowPosition(sdl_window_, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);

  SDL_GetWindowSize(sdl_window_, &width_, &height_);
  int drawable_width = width_;
  int drawable_height = height_;
  SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
  scale_ = (float)drawable_width / (float)width_;
}

void Window::DisableWindowResizing() { SDL_SetWindowResizable(sdl_window_, SDL_FALSE); }
void Window::EnableWindowResizing() { SDL_SetWindowResizable(sdl_window_, SDL_TRUE); }

std::string Window::GetDropFile() {
  std::string tmp;
  std::swap(tmp, drop_file_);
  return tmp;
}

Window::Status Window::NewFrame() {
  SDL_Event event;
  while (SDL_PollEvent(&event)) {
    ImGui_ImplSDL2_ProcessEvent(&event);

    if (event.type == SDL_QUIT) {
      should_exit_ = true;
    } else if (event.type == SDL_APP_WILLENTERBACKGROUND) {
      should_exit_ = true;
    } else if (event.type == SDL_WINDOWEVENT) {
      if (event.window.event == SDL_WINDOWEVENT_RESIZED) {
        SDL_GetWindowSize(sdl_window_, &width_, &height_);
        int drawable_width = width_;
        int drawable_height = height_;
        SDL_GL_GetDrawableSize(sdl_window_, &drawable_width, &drawable_height);
        scale_ = (float)drawable_width / (float)width_;
      } else if (event.window.event == SDL_WINDOWEVENT_CLOSE) {
        should_exit_ = true;
      }
    } else if (event.type == SDL_DROPFILE) {
      drop_file_ = event.drop.file;
    }
  }

  ImGui_ImplSDL2_NewFrame();
  ImGui::NewFrame();

  return should_exit_ ? kQuitting : kRunning;
}

void Window::EndFrame() {
  // We use ImGui for input management in addition to GUI rendering so its
  // important to call ImGui::EndFrame even if we don't call ImGui::Render.
  // Note ImGui::Render internally calls ImGui::EndFrame, but so long as
  // ImGui::NewFrame has been called, ImGui::EndFrame may be called multiple
  // times; it will be a no-op.
  ImGui::EndFrame();
}

void Window::Present(std::span<const std::byte> pixels) {
  // In headless mode, we assume the caller has renderered the scene to an RGB
  // buffer of size width_ * height_ * 3.
  if (IsHeadless(config_.gfx_mode)) {
    if (pixels.size() != width_ * height_ * 3) {
      mju_error("Offscreen mode expects RGB buffer of size %d", width_ * height_ * 3);
    }

    SDL_Surface* surface = SDL_GetWindowSurface(sdl_window_);
    const unsigned char* src = reinterpret_cast<const unsigned char*>(pixels.data());
    unsigned char* dst = static_cast<unsigned char*>(surface->pixels);

    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j) {
        // Swap the RGB channels and add an alpha value of 1.0
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 0xff;
        dst += 4;
        src += 3;
      }
    }

    SDL_RenderPresent(sdl_renderer_);
  } else if (config_.gfx_mode != GraphicsMode::FilamentVulkan &&
             config_.gfx_mode != GraphicsMode::FilamentOpenGl) {
    SDL_GL_SwapWindow(sdl_window_);
  }
}

GraphicsMode Window::GetGraphicsMode() const { return config_.gfx_mode; }

}  // namespace mujoco::studio
