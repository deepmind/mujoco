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

#include "studio/ux/plugin.h"

#include <functional>
#include <string_view>

#include <mujoco/mujoco.h>
#include "engine/engine_global_table.h"

namespace mujoco::studio {

template <typename T>
void RegisterPlugin(T plugin) {
  if (plugin.name == nullptr || plugin.name[0] == '\0') {
    mju_error("Plugin name must not be empty or null.");
  }
  GlobalTable<T>::GetSingleton().AppendIfUnique(plugin);
}

template <typename T>
void ForEachPlugin(const std::function<void(T*)>& fn) {
  auto& table = mujoco::GlobalTable<T>::GetSingleton();
  for (int i = 0; i < table.count(); ++i) {
    const T* plugin = table.GetAtSlot(i);
    fn(const_cast<T*>(plugin));
  }
}

}  // namespace mujoco::studio

#define MUJOCO_SPECIALIZE_PLUGIN(PLUGIN, NAME)                                        \
  template <>                                                                         \
  const char* mujoco::GlobalTable<PLUGIN>::HumanReadableTypeName() {                  \
    return NAME;                                                                      \
  }                                                                                   \
  template <>                                                                         \
  std::string_view mujoco::GlobalTable<PLUGIN>::ObjectKey(const PLUGIN& p) {          \
    return std::string_view(p.name);                                                  \
  }                                                                                   \
  template <>                                                                         \
  bool mujoco::GlobalTable<PLUGIN>::ObjectEqual(const PLUGIN& p1, const PLUGIN& p2) { \
    return CaseInsensitiveEqual(p1.name, p2.name);                                    \
  }                                                                                   \
  template <>                                                                         \
  bool mujoco::GlobalTable<PLUGIN>::CopyObject(PLUGIN& dst, const PLUGIN& src,        \
                                               ErrorMessage& err) {                   \
    dst = src;                                                                        \
    return true;                                                                      \
  }                                                                                   \
  namespace mujoco::studio {                                                          \
  template void RegisterPlugin<PLUGIN>(PLUGIN plugin);                                \
  template void ForEachPlugin<PLUGIN>(const std::function<void(PLUGIN*)>& fn);        \
  }

MUJOCO_SPECIALIZE_PLUGIN(mujoco::studio::GuiPlugin, "gui plugin");
MUJOCO_SPECIALIZE_PLUGIN(mujoco::studio::ModelPlugin, "model plugin");
MUJOCO_SPECIALIZE_PLUGIN(mujoco::studio::ScenePlugin, "scene plugin");
MUJOCO_SPECIALIZE_PLUGIN(mujoco::studio::KeyHandlerPlugin, "key handler plugin");
MUJOCO_SPECIALIZE_PLUGIN(mujoco::studio::SpecEditorPlugin, "spec editor plugin");
