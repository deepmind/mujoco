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

#ifndef MUJOCO_STUDIO_UX_CAMERA_CONTROL_H_
#define MUJOCO_STUDIO_UX_CAMERA_CONTROL_H_

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// The speed at which the fly camera starts to fly (see MoveCameraWithKeys).
inline constexpr float kMinFlySpeed = 0.001f;

// Flags for the state of a mouse button this frame.
enum ButtonState : int {
  kButtonPressed = 1 << 0,   // Held down.
  kButtonReleased = 1 << 1,  // Let go this frame.
};

// One frame of the mouse and keyboard input that moves the camera, independent
// of where it came from (e.g. ReadImGuiCameraInput).
struct CameraInput {
  // Mouse motion this frame, as a fraction of the view's width and height.
  float mouse_dx = 0;
  float mouse_dy = 0;
  // Mouse wheel motion this frame.
  float mouse_wheel = 0;

  // Mouse button states, as ButtonState flags.
  int left_button = 0;
  int right_button = 0;
  int middle_button = 0;

  // Modifier keys held down.
  bool ctrl = false;
  bool shift = false;

  // Direction to fly the fly camera in, each -1, 0 or 1: fly_right and
  // fly_forward move along mjMOUSE_MOVE_H_REL's x and y, and fly_up along
  // mjMOUSE_MOVE_V_REL's y.
  int fly_right = 0;
  int fly_forward = 0;
  int fly_up = 0;

  // Whether the 3D view has the mouse and keyboard focus, i.e. ImGui does not
  // want to capture them (!ImGuiIO::WantCaptureMouse and
  // !ImGuiIO::WantCaptureKeyboard, as set by ReadImGuiCameraInput).
  bool mouse_focus = false;
  bool keyboard_focus = false;
};

// Returns this frame's input in the current ImGui context, where the view has
// the mouse/keyboard focus unless ImGui wants to capture it
// (ImGuiIO::WantCaptureMouse/WantCaptureKeyboard), and the W/S, A/D and Q/E
// keys fly forward/backward, left/right and up/down (W, A and Q win over the
// opposite key). Without a context, there is no input.
CameraInput ReadImGuiCameraInput();

// Moves the camera with mouse drags and the wheel as Studio maps them, and
// returns whether it moved:
//   - left drag orbits around the lookat point, or turns the fly camera;
//   - middle drag and the wheel zoom, except in the fly camera;
//   - right drag moves in the vertical plane, or the horizontal one with Shift.
// Ctrl drags are left to perturbation and move nothing.
bool MoveCameraWithMouse(const mjModel* m, mjvCamera* cam, bool fly_camera,
                         const CameraInput& input);

// Flies the camera in `input`'s fly direction as Studio does in the fly camera,
// and returns whether it moved. `speed` accelerates while a direction is held,
// up to a cap that Shift raises, and drops back to kMinFlySpeed when none is.
bool MoveCameraWithKeys(const mjModel* m, mjvCamera* cam, const CameraInput& input, float* speed);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_CAMERA_CONTROL_H_
