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

#include "studio/ux/picture_gui.h"

#include <algorithm>
#include <string>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>
#include "studio/ux/imgui_widgets.h"

namespace mujoco::studio {

// Returns false if the user requests that this picture-in-picture widget be
// removed from the GUI.
static bool PipGuiImpl(const mjModel* model, float aspect_ratio, const PipRenderCallback& render_cb,
                       PipState* pip) {
  bool result = true;

  auto get_camera_name = [model](int i) -> const char* {
    if (const char* name = mj_id2name(model, mjOBJ_CAMERA, i)) {
      return name;
    } else {
      return "Unnamed";
    }
  };

  const int width = ImGui::GetContentRegionAvail().x;
  const int height = aspect_ratio != 0.f ? width / aspect_ratio : width;
  const int combo_width = (width - 30) / 2;

  ImGui::PushID(pip);
  ImGui::SetNextItemWidth(combo_width);
  if (ImGui::BeginCombo("##PipCamera", get_camera_name(pip->camera))) {
    for (int i = 0; i < model->ncam; i++) {
      if (ImGui::Selectable(get_camera_name(i), (pip->camera == i))) {
        pip->camera = i;
      }
    }
    ImGui::EndCombo();
  }

  const char* mode_names[] = {"Color", "Depth", "Segmentation"};
  ImGui::SameLine();
  ImGui::SetNextItemWidth(combo_width);
  if (ImGui::BeginCombo("##PipMode", mode_names[pip->mode])) {
    for (int i = 0; i < 3; i++) {
      if (ImGui::Selectable(mode_names[i], (pip->mode == i))) {
        pip->mode = static_cast<PipState::Mode>(i);
      }
    }
    ImGui::EndCombo();
  }

  ImGui::SameLine();
  if (ImGui::Button(ICON_FA_TRASH_CAN)) {
    result = false;
  }

  pip->texture = render_cb(*pip, width, height);

  ImGui::Image(pip->texture, {(float)width, (float)height});
  ImGui::PopID();
  return result;
}

void PipGui(const mjModel* model, float aspect_ratio, const PipRenderCallback& render_cb,
            std::vector<PipState>* pips) {
  if (pips->empty()) {
    pips->emplace_back();
  }

  std::vector<int> to_delete;
  for (int i = 0; i < pips->size(); ++i) {
    PipState& pip = pips->at(i);
    if (PipGuiImpl(model, aspect_ratio, render_cb, &pip) == false) {
      to_delete.push_back(i);
    };
    ImGui::Separator();
  }
  for (int i = to_delete.size() - 1; i >= 0; --i) {
    pips->erase(pips->begin() + to_delete[i]);
  }
  if (ImGui::Button("+")) {
    pips->emplace_back();
  };
}

void PipGui(const std::vector<PipSource>& sources, std::vector<PipState>* pips) {
  if (pips == nullptr || sources.empty()) {
    return;
  }
  const int n = static_cast<int>(sources.size());
  if (pips->empty()) {
    pips->emplace_back();
  }

  std::vector<int> to_delete;
  for (int idx = 0; idx < static_cast<int>(pips->size()); ++idx) {
    PipState& pip = (*pips)[idx];
    pip.camera = std::clamp(pip.camera, 0, n - 1);
    const PipSource& src = sources[pip.camera];

    const float width = ImGui::GetContentRegionAvail().x;
    ImGui::PushID(&pip);
    ImGui::SetNextItemWidth(width - 30);
    if (ImGui::BeginCombo("##PipCamera", src.name.c_str())) {
      for (int i = 0; i < n; ++i) {
        if (ImGui::Selectable(sources[i].name.c_str(), pip.camera == i)) {
          pip.camera = i;
        }
      }
      ImGui::EndCombo();
    }

    ImGui::SameLine();
    if (ImGui::Button(ICON_FA_TRASH_CAN)) {
      to_delete.push_back(idx);
    }

    pip.texture = src.texture;
    if (pip.texture != ImTextureID_Invalid) {
      const float height = src.aspect_ratio > 0.0f ? width / src.aspect_ratio : width;
      ImGui::Image(pip.texture, {width, height});
    }
    ImGui::PopID();
    ImGui::Separator();
  }
  for (int i = static_cast<int>(to_delete.size()) - 1; i >= 0; --i) {
    pips->erase(pips->begin() + to_delete[i]);
  }
  if (ImGui::Button("+")) {
    pips->emplace_back();
  }
}

}  // namespace mujoco::studio
