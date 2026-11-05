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

// Tests a plugin built with the mujoco_plugin macro: loading it registers the
// plugin with the same engine this test links dynamically (mujoco_shared).
//
// Uses a plain main() rather than a test framework, so that it links nothing
// else that mujoco_shared might also contain.

#include <cstdio>
#include <cstdlib>

#include <mujoco/mujoco.h>

int main() {
  constexpr char kPluginName[] = "mujoco.test.mujoco_plugin";

  const char* path = std::getenv("MUJOCO_PLUGIN_TEST_LIBRARY");
  if (!path) {
    std::fprintf(stderr, "FAIL: MUJOCO_PLUGIN_TEST_LIBRARY isn't set\n");
    return 1;
  }
  if (mjp_getPlugin(kPluginName, nullptr)) {
    std::fprintf(stderr, "FAIL: %s is registered before loading %s\n",
                 kPluginName, path);
    return 1;
  }
  int count = mjp_pluginCount();

  mj_loadPluginLibrary(path);

  if (mjp_pluginCount() != count + 1 || !mjp_getPlugin(kPluginName, nullptr)) {
    std::fprintf(stderr, "FAIL: loading %s didn't register %s\n", path,
                 kPluginName);
    return 1;
  }
  std::printf("PASS: loading %s registered %s\n", path, kPluginName);
  return 0;
}
