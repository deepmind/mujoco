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

#ifndef MUJOCO_STUDIO_UX_THEME_H_
#define MUJOCO_STUDIO_UX_THEME_H_

// The Studio ImGui themes. Kept apart from gui.h so that a client that only
// needs to look like Studio (the web viewer's browser-side UI) can depend on
// this without pulling in the full GUI library.

#include <imgui.h>

namespace mujoco::studio {

// Standard default UX themes for MuJoCo applications.
enum class GuiTheme {
  kLight,
  kDark,
  kClassic,
};

// Updates the ImGui internal style state to match the requested theme. Sizes
// are derived from the current style's FontScaleDpi, so set that first.
void SetupTheme(GuiTheme theme);

// "Light", "Dark" or "Classic".
const char* ThemeName(GuiTheme theme);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_THEME_H_
