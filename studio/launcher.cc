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

#include "studio/launcher.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>

#include <mujoco/mujoco.h>
#include "studio/hal/graphics_mode.h"
#include "studio/io/resources.h"
#include "studio/ux/gui.h"
#include "studio/app.h"

namespace mujoco::studio {

int LaunchStudio(int argc, char** argv, LauncherConfig config) {
  const std::string ini_path = GetDefaultIniPath();

  mujoco::studio::RegisterProviders();

  if (config.gfx_mode.empty()) {
    const char* display = std::getenv("DISPLAY");
    if (display && strcmp(display, ":20") == 0) {
      config.gfx_mode = "opengl_headless";
    }
  }

  const char* session_type = std::getenv("XDG_SESSION_TYPE");
  const char* wayland_display = std::getenv("WAYLAND_DISPLAY");
  if ((session_type && std::string_view(session_type) == "wayland") || wayland_display) {
    if (config.gfx_mode.empty()) {
      config.gfx_mode = "opengl_headless";
    } else if (config.gfx_mode == "opengl") {
      mju_error(
          "Wayland does not support '%s' graphics mode. "
          "Restart with a different graphics mode, or login using X11.",
          config.gfx_mode.c_str());
    }
  }

  mujoco::studio::GraphicsMode gfx_mode = mujoco::studio::GraphicsModeFromString(
      config.gfx_mode, mujoco::studio::GraphicsMode::FilamentOpenGl);

  // Use config values if they are set (non-default), otherwise use flags.
  mujoco::studio::App app({
      .width = config.window_width,
      .height = config.window_height,
      .ini_path = ini_path,
      .gfx_mode = gfx_mode,
      .title = config.title,
  });

  if (config.model_file.empty()) {
    app.InitEmptyModel();
  } else {
    app.LoadModelFromFile(config.model_file);
  }

  while (app.Update()) {
    app.BuildGui();
    app.Render();
  }
  return 0;
}

}  // namespace mujoco::studio
