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

// A minimal plugin, built with the mujoco_plugin macro for mujoco_plugin_test.

#include <mujoco/mujoco.h>

namespace mujoco {
namespace {

mjPLUGIN_LIB_INIT(mujoco_plugin_test) {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.mujoco_plugin";
  mjp_registerPlugin(&plugin);
}

}  // namespace
}  // namespace mujoco
