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

#include "studio/ux/gui_spec.h"

#include <memory>

#include "testing/base/public/gunit.h"
#include <imgui.h>
#include <imgui_internal.h>
#include <mujoco/mujoco.h>
#include "studio/ux/imgui_widgets.h"
#include "studio/ux/spec_editor.h"

namespace mujoco::studio {
namespace {

using SpecPtr = std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)>;

TEST(GuiSpecTest, DeleteBodyWhileOpenDoesNotUseAfterFree) {
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.DisplaySize = ImVec2(800, 600);
  io.DeltaTime = 1.0f / 60.0f;
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  unsigned char* pixels = nullptr;
  int width = 0;
  int height = 0;
  io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

  SpecPtr spec(mj_makeSpec(), mj_deleteSpec);
  mjsBody* world = mjs_findBody(spec.get(), "world");
  mjsBody* child_body = mjs_addBody(world, nullptr);
  mjs_setName(child_body->element, "child");
  mjs_addGeom(child_body, nullptr);

  SpecEditor editor;
  editor.Reset(*spec);

  mjsBody* active_world = mjs_findBody(editor.GetActiveSpec(), "world");
  ASSERT_NE(active_world, nullptr);
  mjsBody* active_child = mjs_findChild(active_world, "child");
  ASSERT_NE(active_child, nullptr);

  mjsElement* selected_element = active_child->element;
  editor.SetActiveElement(selected_element);

  ImGui::NewFrame();
  ImGui::Begin("SpecTreeTest");

  ImGuiWindow* window = ImGui::GetCurrentWindow();

  // Compute exact IDs matching the ID stack inside BodyTreeGuiRecursive.
  ImGuiID id_bt = window->GetID("Body Tree");
  ImGui::PushOverrideID(id_bt);

  ImGui::PushID("world");
  ImGuiID id_world = window->GetID("world");
  ImGui::PushOverrideID(id_world);

  ImGui::PushID("child");
  ImGuiID id_child = window->GetID("child");
  ImGui::PushOverrideID(id_child);
  ImGuiID trash_id = window->GetID(ICON_FA_TRASH_CAN);

  ImGui::PopID();  // pop id_child
  ImGui::PopID();  // pop "child"
  ImGui::PopID();  // pop id_world
  ImGui::PopID();  // pop "world"
  ImGui::PopID();  // pop id_bt

  // Force all tree nodes to be open.
  window->StateStorage.SetInt(id_bt, 1);
  window->StateStorage.SetInt(id_world, 1);
  window->StateStorage.SetInt(id_child, 1);

  // Activate trash button.
  ImGuiContext& g = *GImGui;
  g.NavId = trash_id;
  g.NavActivateId = trash_id;
  g.NavActivateDownId = trash_id;
  g.NavActivateFlags = ImGuiActivateFlags_None;

  SpecTreeGui(&selected_element, editor.GetActiveSpec(), &editor);

  ImGui::End();
  ImGui::Render();

  EXPECT_EQ(selected_element, nullptr);

  ImGui::DestroyContext();
}

}  // namespace
}  // namespace mujoco::studio
