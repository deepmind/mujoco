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

#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "studio/launcher.h"

ABSL_FLAG(int, window_width, 1400, "Window width");
ABSL_FLAG(int, window_height, 720, "Window height");
ABSL_FLAG(std::string, model_file, "", "MuJoCo model file.");
ABSL_FLAG(std::string, gfx, "", "Graphics API");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string model_file = absl::GetFlag(FLAGS_model_file);
  if (model_file.empty() && argc > 1 && argv[1][0] != '-') {
    model_file = argv[1];
  }

  return mujoco::studio::LaunchStudio(argc, argv,
                                      {
                                          .window_width = absl::GetFlag(FLAGS_window_width),
                                          .window_height = absl::GetFlag(FLAGS_window_height),
                                          .model_file = model_file,
                                          .gfx_mode = absl::GetFlag(FLAGS_gfx),
                                      });
}
