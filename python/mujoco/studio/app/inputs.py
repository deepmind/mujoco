# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Temporary input (e.g., keyboard, mouse) handling functions for Studio app."""

# TODO(matijak): These free functions implement the keyboard and mouse event
# handling for Studio. They are separated from the main App class to
# clarify the long-term API and avoid cluttering it with a large amount of
# temporary code. When studio/platform has a proper API for registering key
# bindings and mouse behaviour, the event handling functions will delegate to
# code shared with the C++ studio application.

from typing import Any

import mujoco
from mujoco.studio import sim
from mujoco.studio import ux
import numpy as np

from mujoco.studio.dear_imgui import dear_imgui as imgui


def handle_vis_options_keyboard_events(
    vis_options: mujoco.MjvOption,
    is_freecam_wasd: bool,
) -> bool:
  """Toggles visualization flags based on keyboard shortcuts.

  Args:
    vis_options: The visualization options to modify.
    is_freecam_wasd: If True, keys Q/E/A/D are reserved for camera movement and
      will not toggle visualization flags.

  Returns:
    True if a key was handled, False otherwise.
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False

  pressed = imgui.IsKeyChordPressed

  # Frame and label cycling.
  if pressed(imgui.Key.F6):
    vis_options.frame = (vis_options.frame + 1) % mujoco.mjtFrame.mjNFRAME.value
  elif pressed(imgui.Key.F7):
    vis_options.label = (vis_options.label + 1) % mujoco.mjtLabel.mjNLABEL.value

  # Visualization flag toggles (single-key shortcuts).
  elif pressed(imgui.Key.H):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] ^= 1
  elif pressed(imgui.Key.X):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_TEXTURE] ^= 1
  elif pressed(imgui.Key.J):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_JOINT] ^= 1
  elif not is_freecam_wasd and pressed(imgui.Key.Q):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] ^= 1
  elif pressed(imgui.Key.U):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] ^= 1
  elif pressed(imgui.Key.Comma):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] ^= 1
  elif pressed(imgui.Key.Z):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] ^= 1
  elif pressed(imgui.Key.V):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_TENDON] ^= 1
  elif pressed(imgui.Key.Y):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] ^= 1
  elif not is_freecam_wasd and pressed(imgui.Key.E):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONSTRAINT] ^= 1
  elif pressed(imgui.Key.I):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] ^= 1
  elif pressed(imgui.Key.Apostrophe):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_SCLINERTIA] ^= 1
  elif pressed(imgui.Key.B):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] ^= 1
  elif pressed(imgui.Key.O):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] ^= 1
  elif pressed(imgui.Key.C):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] ^= 1
  elif pressed(imgui.Key.N):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_ISLAND] ^= 1
  elif pressed(imgui.Key.F):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] ^= 1
  elif pressed(imgui.Key.P):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_CONTACTSPLIT] ^= 1
  elif pressed(imgui.Key.T):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] ^= 1
  elif not is_freecam_wasd and pressed(imgui.Key.A):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_AUTOCONNECT] ^= 1
  elif pressed(imgui.Key.M):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_COM] ^= 1
  elif not is_freecam_wasd and pressed(imgui.Key.D):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_STATIC] ^= 1
  elif pressed(imgui.Key.Semicolon):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_SKIN] ^= 1
  elif pressed(imgui.Key.GraveAccent):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] ^= 1
  elif pressed(imgui.Key.Backslash):
    vis_options.flags[mujoco.mjtVisFlag.mjVIS_MESHBVH] ^= 1

  # Site group toggles (Shift + 0-5).
  elif pressed(int(imgui.Key.N0) | int(imgui.Key.Shift)):
    vis_options.sitegroup[0] ^= 1
  elif pressed(int(imgui.Key.N1) | int(imgui.Key.Shift)):
    vis_options.sitegroup[1] ^= 1
  elif pressed(int(imgui.Key.N2) | int(imgui.Key.Shift)):
    vis_options.sitegroup[2] ^= 1
  elif pressed(int(imgui.Key.N3) | int(imgui.Key.Shift)):
    vis_options.sitegroup[3] ^= 1
  elif pressed(int(imgui.Key.N4) | int(imgui.Key.Shift)):
    vis_options.sitegroup[4] ^= 1
  elif pressed(int(imgui.Key.N5) | int(imgui.Key.Shift)):
    vis_options.sitegroup[5] ^= 1

  # Geom group toggles (0-5).
  elif pressed(imgui.Key.N0):
    vis_options.geomgroup[0] ^= 1
  elif pressed(imgui.Key.N1):
    vis_options.geomgroup[1] ^= 1
  elif pressed(imgui.Key.N2):
    vis_options.geomgroup[2] ^= 1
  elif pressed(imgui.Key.N3):
    vis_options.geomgroup[3] ^= 1
  elif pressed(imgui.Key.N4):
    vis_options.geomgroup[4] ^= 1
  elif pressed(imgui.Key.N5):
    vis_options.geomgroup[5] ^= 1

  else:
    return False

  return True


def handle_step_control_keyboard_events(
    step_control: sim.StepControl,
    ux_state: ux.UxState,
) -> bool:
  """Handles keyboard shortcuts for simulation stepping control.

  Args:
    step_control: The simulation step control object.
    ux_state: The UX state object.

  Returns:
    True if a key was handled, False otherwise.
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False

  pressed = imgui.IsKeyChordPressed

  if pressed(imgui.Key.Space):
    if step_control.get_pause_state() == sim.PauseState.UNPAUSED:
      step_control.set_pause_state(sim.PauseState.NORMAL_PAUSED)
    else:
      step_control.set_pause_state(sim.PauseState.UNPAUSED)
    return True
  elif pressed(imgui.Key.Minus):
    ux_state.speed_index = ux.set_speed_index(
        step_control, ux_state.speed_index, ux_state.speed_index + 1
    )
    return True
  elif pressed(imgui.Key.Equal):
    ux_state.speed_index = ux.set_speed_index(
        step_control, ux_state.speed_index, ux_state.speed_index - 1
    )
    return True

  return False


def handle_sim_history_keyboard_events(
    step_control: sim.StepControl,
    ux_state: ux.UxState,
) -> bool:
  """Handles keyboard shortcuts for scrubbing the simulation history.

  Left/Right load the previous/next recorded frame while paused; Right at the
  head of the history requests a single step instead. Matches the C++ Studio
  app.

  Args:
    step_control: The simulation step control object.
    ux_state: The UX state holding the timeline history size and index.

  Returns:
    True if a key was handled, False otherwise.
  """
  if imgui.GetIO().WantCaptureKeyboard or ux_state.history_size <= 0:
    return False

  # Scrubbing only makes sense while paused: a running sim would immediately
  # step away from the loaded frame.
  if step_control.get_pause_state() != sim.PauseState.NORMAL_PAUSED:
    return False

  pressed = imgui.IsKeyChordPressed

  if pressed(imgui.Key.LeftArrow):
    ux_state.history_index -= 1
    return True
  elif pressed(imgui.Key.RightArrow):
    if ux_state.history_index == 0:
      step_control.request_single_step()
    else:
      ux_state.history_index += 1
    return True

  return False


def handle_reset_keyboard_events(
    model: mujoco.MjModel, data: mujoco.MjData
) -> bool:
  """Handles keyboard shortcuts for simulation reset."""
  if imgui.GetIO().WantCaptureKeyboard:
    return False

  if imgui.IsKeyChordPressed(imgui.Key.Backspace):
    if model is not None and data is not None:
      mujoco.mj_resetData(model, data)
      mujoco.mj_forward(model, data)
      return True

  return False


def handle_camera_select_keyboard_events(
    model: mujoco.MjModel,
    camera: mujoco.MjvCamera,
    ux_state: ux.UxState,
) -> bool:
  """Handles keyboard shortcuts for camera selection.

  Args:
    model: The MuJoCo model.
    camera: The MuJoCo camera object.
    ux_state: The UX state object.

  Returns:
    True if a key was handled, False otherwise.
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False

  pressed = imgui.IsKeyChordPressed

  if pressed(imgui.Key.Escape):
    ux_state.camera_index = ux.set_camera(model, camera, ux.TUMBLE_CAMERA_IDX)
    return True
  elif pressed(imgui.Key.LeftBracket):
    ux_state.camera_index = ux.set_camera(
        model, camera, ux_state.camera_index - 1
    )
    return True
  elif pressed(imgui.Key.RightBracket):
    ux_state.camera_index = ux.set_camera(
        model, camera, ux_state.camera_index + 1
    )
    return True

  return False


def handle_miscellaneous_keyboard_events(app: Any) -> bool:
  """Handles miscellaneous keyboard shortcuts for toggling panels and windows.

  Tab/Shift+Tab toggle Options and Inspector panels.
  F2/F3 the Info and Profiler windows.

  Args:
    app: The viewer application instance, whose config holds the UI state flags.

  Returns:
    True if a key was handled, False otherwise.
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False

  # Shift+Tab has to be tested before Tab, else the plain chord consumes it.
  if imgui.IsKeyChordPressed(int(imgui.Key.Tab) | int(imgui.Key.Shift)):
    app.config.show_inspector = not app.config.show_inspector
    return True
  elif imgui.IsKeyChordPressed(imgui.Key.Tab):
    app.config.show_options = not app.config.show_options
    return True
  elif imgui.IsKeyChordPressed(imgui.Key.F2):
    app.config.show_info = not app.config.show_info
    return True
  elif imgui.IsKeyChordPressed(imgui.Key.F3):
    app.config.show_profiler = not app.config.show_profiler
    return True

  return False


def handle_freecam_wasd_keyboard_events(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    cam_speed: float,
) -> tuple[bool, float]:
  """Handles keyboard shortcuts for free camera movement.

  Args:
    model: The MuJoCo model.
    data: The MuJoCo data.
    camera: The MuJoCo camera object.
    cam_speed: The current camera speed.

  Returns:
    A tuple of (handled, updated_cam_speed).
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False, cam_speed

  if (
      imgui.IsKeyDown(imgui.Key.W)
      or imgui.IsKeyDown(imgui.Key.S)
      or imgui.IsKeyDown(imgui.Key.A)
      or imgui.IsKeyDown(imgui.Key.D)
      or imgui.IsKeyDown(imgui.Key.Q)
      or imgui.IsKeyDown(imgui.Key.E)
  ):
    moved = False

    # Move (dolly) forward/backward using W and S keys.
    if imgui.IsKeyDown(imgui.Key.W):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_DOLLY,
          0,
          cam_speed,
      )
      moved = True
    elif imgui.IsKeyDown(imgui.Key.S):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_DOLLY,
          0,
          -cam_speed,
      )
      moved = True

    # Strafe (truck) left/right using A and D keys.
    if imgui.IsKeyDown(imgui.Key.A):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_DOLLY,
          -cam_speed,
          0,
      )
      moved = True
    elif imgui.IsKeyDown(imgui.Key.D):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_DOLLY,
          cam_speed,
          0,
      )
      moved = True

    # Move (pedestal) up/down using Q and E keys.
    if imgui.IsKeyDown(imgui.Key.Q):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_PEDESTAL,
          0,
          cam_speed,
      )
      moved = True
    elif imgui.IsKeyDown(imgui.Key.E):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.TRUCK_PEDESTAL,
          0,
          -cam_speed,
      )
      moved = True

    if moved:
      cam_speed += 0.001
      max_speed = 0.1 if imgui.GetIO().KeyShift else 0.01
      if cam_speed > max_speed:
        cam_speed = max_speed
    else:
      cam_speed = 0.001

    return True, cam_speed

  return False, cam_speed


def handle_keyboard_events(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    vis_options: mujoco.MjvOption,
    step_control: sim.StepControl,
    ux_state: ux.UxState,
    cam_speed: float,
) -> tuple[bool, float]:
  """Handle keyboard events according to Studio's bindings.

  Args:
    model: The MuJoCo model.
    data: The MuJoCo data.
    camera: The MuJoCo camera object.
    vis_options: The MuJoCo visualization options.
    step_control: The simulation step control object.
    ux_state: The UX state object.
    cam_speed: The current camera speed.

  Returns:
    A tuple of (handled, updated_cam_speed).
  """
  if imgui.GetIO().WantCaptureKeyboard:
    return False, cam_speed

  is_freecam_wasd = ux_state.camera_index == ux.FREE_CAMERA_IDX
  if handle_step_control_keyboard_events(step_control, ux_state):
    return True, cam_speed

  if handle_reset_keyboard_events(model, data):
    return True, cam_speed

  if handle_camera_select_keyboard_events(model, camera, ux_state):
    return True, cam_speed

  if handle_vis_options_keyboard_events(vis_options, is_freecam_wasd):
    return True, cam_speed

  if is_freecam_wasd:
    handled, cam_speed = handle_freecam_wasd_keyboard_events(
        model, data, camera, cam_speed
    )
    if handled:
      return True, cam_speed

  return False, cam_speed


def handle_camera_tracking_mouse_events(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    vis_options: mujoco.MjvOption,
    ux_state: ux.UxState,
) -> None:
  """Handles mouse events for camera tracking."""
  io = imgui.GetIO()
  if imgui.GetIO().WantCaptureMouse:
    return

  if io.DisplaySize.x <= 0 or io.DisplaySize.y <= 0:
    return

  mouse_x = io.MousePos.x / io.DisplaySize.x
  mouse_y = io.MousePos.y / io.DisplaySize.y
  aspect_ratio = io.DisplaySize.x / io.DisplaySize.y

  # Right double click.
  if imgui.IsMouseDoubleClicked(imgui.MouseButton.Right):
    picked = ux.Pick(
        model,
        data,
        camera,
        mouse_x,
        mouse_y,
        aspect_ratio,
        vis_options,
    )
    camera.lookat[:] = picked.point
    if picked.body > 0 and io.KeyCtrl:
      # Switch camera to tracking mode and track the selected body.
      camera.type = int(mujoco.mjtCamera.mjCAMERA_TRACKING)
      camera.trackbodyid = picked.body
      camera.fixedcamid = -1
      ux_state.camera_index = ux.TRACKING_CAMERA_IDX


def handle_mouse_events(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
    vis_options: mujoco.MjvOption,
    perturb: mujoco.MjvPerturb,
    ux_state: ux.UxState,
) -> None:
  """Handles mouse events."""
  io = imgui.GetIO()
  if io.WantCaptureMouse:
    return

  if io.DisplaySize.x <= 0 or io.DisplaySize.y <= 0:
    return

  mouse_x = io.MousePos.x / io.DisplaySize.x
  mouse_y = io.MousePos.y / io.DisplaySize.y
  mouse_dx = io.MouseDelta.x / io.DisplaySize.x
  mouse_dy = io.MouseDelta.y / io.DisplaySize.y
  mouse_scroll = io.MouseWheel / 50.0

  is_mouse_moving = mouse_dx != 0.0 or mouse_dy != 0.0
  is_any_mouse_down = (
      imgui.IsMouseDown(imgui.MouseButton.Left)
      or imgui.IsMouseDown(imgui.MouseButton.Right)
      or imgui.IsMouseDown(imgui.MouseButton.Middle)
  )
  is_mouse_dragging = is_mouse_moving and is_any_mouse_down

  # If no mouse buttons are down, end any active perturbations.
  if not is_any_mouse_down:
    perturb.active = 0

  # Handle perturbation mouse actions.
  if is_mouse_dragging and io.KeyCtrl:
    if perturb.select > 0:
      action = int(mujoco.mjtMouse.mjMOUSE_NONE)
      if imgui.IsMouseDown(imgui.MouseButton.Left):
        if io.KeyAlt:
          action = int(
              mujoco.mjtMouse.mjMOUSE_MOVE_H
              if io.KeyShift
              else mujoco.mjtMouse.mjMOUSE_MOVE_V
          )
        else:
          action = int(
              mujoco.mjtMouse.mjMOUSE_ROTATE_H
              if io.KeyShift
              else mujoco.mjtMouse.mjMOUSE_ROTATE_V
          )
      elif imgui.IsMouseDown(imgui.MouseButton.Right):
        action = int(
            mujoco.mjtMouse.mjMOUSE_MOVE_H
            if io.KeyShift
            else mujoco.mjtMouse.mjMOUSE_MOVE_V
        )
      elif imgui.IsMouseDown(imgui.MouseButton.Middle):
        action = int(mujoco.mjtMouse.mjMOUSE_ZOOM)

      move_actions = (
          int(mujoco.mjtMouse.mjMOUSE_MOVE_V),
          int(mujoco.mjtMouse.mjMOUSE_MOVE_H),
      )
      active = int(
          mujoco.mjtPertBit.mjPERT_TRANSLATE
          if action in move_actions
          else mujoco.mjtPertBit.mjPERT_ROTATE
      )
      if active != perturb.active:
        ux.InitPerturb(model, data, camera, perturb, active)
      ux.MovePerturb(
          model,
          data,
          camera,
          perturb,
          action,
          mouse_dx,
          mouse_dy,
      )
  elif is_mouse_dragging:
    if ux_state.camera_index == ux.FREE_CAMERA_IDX:
      if imgui.IsMouseDown(imgui.MouseButton.Left):
        ux.MoveCamera(
            model,
            data,
            camera,
            ux.CameraMotion.PAN_TILT,
            mouse_dx,
            mouse_dy,
        )
    else:
      if imgui.IsMouseDown(imgui.MouseButton.Left):
        ux.MoveCamera(
            model,
            data,
            camera,
            ux.CameraMotion.ORBIT,
            mouse_dx,
            mouse_dy,
        )
      elif imgui.IsMouseDown(imgui.MouseButton.Middle):
        ux.MoveCamera(
            model,
            data,
            camera,
            ux.CameraMotion.ZOOM,
            mouse_dx,
            mouse_dy,
        )

    # Right mouse movement is relative to the horizontal and vertical planes.
    if imgui.IsMouseDown(imgui.MouseButton.Right) and io.KeyShift:
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.PLANAR_MOVE_H,
          mouse_dx,
          mouse_dy,
      )
    elif imgui.IsMouseDown(imgui.MouseButton.Right):
      ux.MoveCamera(
          model,
          data,
          camera,
          ux.CameraMotion.PLANAR_MOVE_V,
          mouse_dx,
          mouse_dy,
      )

  # Mouse scroll.
  if mouse_scroll != 0.0 and ux_state.camera_index != ux.FREE_CAMERA_IDX:
    ux.MoveCamera(
        model,
        data,
        camera,
        ux.CameraMotion.ZOOM,
        0,
        -mouse_scroll,
    )

  aspect_ratio = io.DisplaySize.x / io.DisplaySize.y

  # Left double click.
  if imgui.IsMouseDoubleClicked(imgui.MouseButton.Left):
    picked = ux.Pick(
        model,
        data,
        camera,
        mouse_x,
        mouse_y,
        aspect_ratio,
        vis_options,
    )
    if picked.body >= 0:
      perturb.select = picked.body
      perturb.flexselect = picked.flex
      perturb.skinselect = picked.skin

      # Compute the local position of the selected object in the world.
      tmp = np.array(picked.point, dtype=np.float64) - data.xpos[picked.body]
      xmat = np.array(data.xmat[picked.body], dtype=np.float64).reshape(3, 3)
      perturb.localpos = xmat.T @ tmp
    else:
      perturb.select = 0
      perturb.flexselect = -1
      perturb.skinselect = -1

  handle_camera_tracking_mouse_events(
      model, data, camera, vis_options, ux_state
  )
