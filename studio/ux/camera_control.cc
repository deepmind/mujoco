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

#include "studio/ux/camera_control.h"

#include <algorithm>

#include <imgui.h>
#include <mujoco/mujoco.h>

namespace mujoco::studio {

// How much the fly speed grows each frame the keys are held, and its caps
// without and with Shift.
static constexpr float kFlyAcceleration = 0.001f;
static constexpr float kMaxFlySpeed = 0.01f;
static constexpr float kMaxFastFlySpeed = 0.1f;

// Wheel steps per unit of zoom motion.
static constexpr float kWheelStepsPerZoom = 50.0f;

// Returns `button`'s ButtonState flags this frame.
static int ReadImGuiButton(ImGuiMouseButton button) {
  int state = 0;
  if (ImGui::IsMouseDown(button)) {
    state |= kButtonPressed;
  }
  if (ImGui::IsMouseReleased(button)) {
    state |= kButtonReleased;
  }
  return state;
}

CameraInput ReadImGuiCameraInput() {
  CameraInput input;
  if (ImGui::GetCurrentContext() == nullptr) {
    return input;
  }
  const ImGuiIO& io = ImGui::GetIO();
  const bool has_display = io.DisplaySize.x > 0 && io.DisplaySize.y > 0;
  if (has_display) {
    input.mouse_dx = io.MouseDelta.x / io.DisplaySize.x;
    input.mouse_dy = io.MouseDelta.y / io.DisplaySize.y;
  }
  input.mouse_wheel = io.MouseWheel;
  input.left_button = ReadImGuiButton(ImGuiMouseButton_Left);
  input.right_button = ReadImGuiButton(ImGuiMouseButton_Right);
  input.middle_button = ReadImGuiButton(ImGuiMouseButton_Middle);
  input.ctrl = io.KeyCtrl;
  input.shift = io.KeyShift;
  input.fly_forward = ImGui::IsKeyDown(ImGuiKey_W) ? 1 : (ImGui::IsKeyDown(ImGuiKey_S) ? -1 : 0);
  input.fly_right = ImGui::IsKeyDown(ImGuiKey_A) ? -1 : (ImGui::IsKeyDown(ImGuiKey_D) ? 1 : 0);
  input.fly_up = ImGui::IsKeyDown(ImGuiKey_Q) ? 1 : (ImGui::IsKeyDown(ImGuiKey_E) ? -1 : 0);
  input.mouse_focus = has_display && !io.WantCaptureMouse;
  input.keyboard_focus = !io.WantCaptureKeyboard;
  return input;
}

bool MoveCameraWithMouse(const mjModel* m, mjvCamera* cam, bool fly_camera,
                         const CameraInput& input) {
  if (!input.mouse_focus) {
    return false;
  }

  bool moved = false;
  const float dx = input.mouse_dx;
  const float dy = input.mouse_dy;
  const bool left = input.left_button & kButtonPressed;
  const bool right = input.right_button & kButtonPressed;
  const bool middle = input.middle_button & kButtonPressed;
  const bool dragging = (dx != 0 || dy != 0) && (left || right || middle);
  if (dragging && !input.ctrl) {
    if (left) {
      // Turn (pan/tilt) the fly camera, or orbit around the lookat point.
      mjv_moveCamera(m, fly_camera ? mjMOUSE_TURN_H : mjMOUSE_ROTATE_H, dx, 0, cam);
      mjv_moveCamera(m, fly_camera ? mjMOUSE_TURN_V : mjMOUSE_ROTATE_V, 0, dy, cam);
      moved = true;
    } else if (middle && !fly_camera) {
      mjv_moveCamera(m, mjMOUSE_ZOOM, 0, dy, cam);
      moved = true;
    }

    // Right drags move relative to the horizontal and vertical planes.
    if (right) {
      mjv_moveCamera(m, input.shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V, dx, dy, cam);
      moved = true;
    }
  }

  // The wheel zooms towards/away from the lookat point, which the fly camera
  // does not have.
  if (input.mouse_wheel != 0 && !fly_camera) {
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -input.mouse_wheel / kWheelStepsPerZoom, cam);
    moved = true;
  }
  return moved;
}

bool MoveCameraWithKeys(const mjModel* m, mjvCamera* cam, const CameraInput& input, float* speed) {
  const float s = *speed;
  bool moved = false;
  if (input.keyboard_focus) {
    // Strafe (truck) right/left and move (dolly) forward/backward.
    if (input.fly_right != 0 || input.fly_forward != 0) {
      mjv_moveCamera(m, mjMOUSE_MOVE_H_REL, input.fly_right * s, input.fly_forward * s, cam);
      moved = true;
    }

    // Move (pedestal) up/down.
    if (input.fly_up != 0) {
      mjv_moveCamera(m, mjMOUSE_MOVE_V_REL, 0, input.fly_up * s, cam);
      moved = true;
    }
  }

  if (moved) {
    const float max_speed = input.shift ? kMaxFastFlySpeed : kMaxFlySpeed;
    *speed = std::min(s + kFlyAcceleration, max_speed);
  } else {
    *speed = kMinFlySpeed;
  }
  return moved;
}

}  // namespace mujoco::studio
