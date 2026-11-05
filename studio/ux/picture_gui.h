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

#ifndef MUJOCO_STUDIO_UX_PICTURE_GUI_H_
#define MUJOCO_STUDIO_UX_PICTURE_GUI_H_

#include <functional>
#include <string>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Describes a pre-uploaded texture source available for picture-in-picture.
struct PipSource {
  std::string name;
  ImTextureID texture = ImTextureID_Invalid;
  float aspect_ratio = 1.0f;
};

// Manages the state of a single picture-in-picture widget.
struct PipState {
  enum Mode { Color, Depth, Segmentation };
  int camera = 0;
  ImTextureID texture = ImTextureID_Invalid;
  Mode mode = Color;
};

using PipRenderCallback = std::function<ImTextureID(const PipState&, int, int)>;

// Renders the GUI for a set of picture-in-picture widgets.
void PipGui(const mjModel* model, float aspect_ratio, const PipRenderCallback& render_cb,
            std::vector<PipState>* pips);

// Renders the GUI for a set of picture-in-picture widgets backed by
// pre-uploaded textures.
void PipGui(const std::vector<PipSource>& sources, std::vector<PipState>* pips);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_PICTURE_GUI_H_
