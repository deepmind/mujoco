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

#ifndef MUJOCO_STUDIO_LAUNCHER_H_
#define MUJOCO_STUDIO_LAUNCHER_H_

#include <string>
#include "studio/app.h"

namespace mujoco::studio {

struct LauncherConfig {
  // Title of the window.
  std::string title = "MuJoCo Studio";

  // Initial window width.
  int window_width = 1400;

  // Initial window height.
  int window_height = 720;

  // Model file to load on startup. If empty, the launcher will try to load
  // from flags or command line arguments.
  std::string model_file = "";

  std::string gfx_mode = "";
};

// Runs the MuJoCo Studio application.
// This function handles resource provider registration, graphics mode
// detection, app creation, and the main loop.
//
// Expects absl::ParseCommandLine to have been called before this.
int LaunchStudio(int argc, char** argv, LauncherConfig config = {});

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_LAUNCHER_H_
