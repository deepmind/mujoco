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

#ifndef MUJOCO_STUDIO_UX_INTERACTION_H_
#define MUJOCO_STUDIO_UX_INTERACTION_H_

#include <mujoco/mujoco.h>

namespace mujoco::studio {

// The result of a pick operation.
struct PickResult {
  mjtNum point[3] = {0, 0, 0};  // World coordinates
  mjtNum dist = -1;             // Distance from the camera.
  int body = -1;
  int geom = -1;
  int flex = -1;
  int skin = -1;
};

// Returns information about the object (if any) under the mouse cursor.
PickResult Pick(const mjModel* m, const mjData* d, const mjvCamera* camera, float x, float y,
                float aspect_ratio, const mjvOption* vis_options);

// Indices for cameras that are not defined in the model.
static constexpr int kTumbleCameraIdx = -3;
static constexpr int kFreeCameraIdx = -2;
static constexpr int kTrackingCameraIdx = -1;

// Updates the camera according to the requested index.
//
// The function returns the new index of the camera which may differ from the
// request if the request was invalid (e.g. request was out of range).
int SetCamera(const mjModel* m, mjvCamera* camera, int request_idx);

// Camera motions are either relative to a target or the camera itself.
//
// We use the following camera nomenclature:
//   - Truck: moves the camera left/right along a horizontal plane.
//   - Pedestal: moves the camera up/down along a vertical plane.
//   - Dolly: moves the camera forward/backward along a horizontal plane.
//   - Pan: turns the camera left/right.
//   - Tilt: turns the camera upwards/downwards.
//   - Zoom: moves the camera closer to or away from the target. This is
//           different from dolly in that the movement is relative to the
//           target. (It's also not actually a camera zoom, which is an
//           action of lens of the camera, rather than the camera itself.)
//   - Orbit: moves the camera around the target.
//   - Planer: creates a horizontal or vertical plane based on the cameras
//             position and orientation, then moves the camera along that plane.
enum class CameraMotion {
  ZOOM,
  ORBIT,
  TRUCK_PEDESTAL,
  TRUCK_DOLLY,
  PAN_TILT,
  PLANAR_MOVE_H,
  PLANAR_MOVE_V,
};

void MoveCamera(const mjModel* m, const mjData* d, mjvCamera* cam, CameraMotion motion, mjtNum dx,
                mjtNum dy);

void InitPerturb(const mjModel* m, const mjData* d, const mjvCamera* cam, mjvPerturb* pert,
                 mjtPertBit active);

void MovePerturb(const mjModel* m, const mjData* d, const mjvCamera* cam, mjvPerturb* pert,
                 mjtMouse action, mjtNum reldx, mjtNum reldy);

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_UX_INTERACTION_H_
