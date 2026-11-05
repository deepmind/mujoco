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

#include <initializer_list>

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

constexpr char kXml[] = R"(
<mujoco>
  <worldbody>
    <geom size="1"/>
  </worldbody>
</mujoco>
)";

// A mouse drag's motion, as a fraction of the view size.
constexpr float kDx = 0.1f;
constexpr float kDy = 0.2f;

class CameraControlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    char error[1000] = "";
    mjSpec* spec = mj_parseXMLString(kXml, nullptr, error, sizeof(error));
    ASSERT_NE(spec, nullptr) << error;
    model_ = mj_compile(spec, nullptr);
    mj_deleteSpec(spec);
    ASSERT_NE(model_, nullptr);
    mjv_defaultFreeCamera(model_, &camera_);
  }

  void TearDown() override { mj_deleteModel(model_); }

  // One mjv_moveCamera call.
  struct Motion {
    mjtMouse action;
    mjtNum dx;
    mjtNum dy;
  };

  // Returns the camera as moved by mjv_moveCamera with `motions`, in order.
  mjvCamera Moved(std::initializer_list<Motion> motions) const {
    mjvCamera camera = camera_;
    for (const Motion& motion : motions) {
      mjv_moveCamera(model_, motion.action, motion.dx, motion.dy, &camera);
    }
    return camera;
  }

  static void ExpectSameCamera(const mjvCamera& actual, const mjvCamera& expected) {
    EXPECT_DOUBLE_EQ(actual.lookat[0], expected.lookat[0]);
    EXPECT_DOUBLE_EQ(actual.lookat[1], expected.lookat[1]);
    EXPECT_DOUBLE_EQ(actual.lookat[2], expected.lookat[2]);
    EXPECT_DOUBLE_EQ(actual.distance, expected.distance);
    EXPECT_DOUBLE_EQ(actual.azimuth, expected.azimuth);
    EXPECT_DOUBLE_EQ(actual.elevation, expected.elevation);
  }

  static CameraInput Drag(float dx, float dy) {
    CameraInput input;
    input.mouse_dx = dx;
    input.mouse_dy = dy;
    input.mouse_focus = true;
    return input;
  }

  mjModel* model_ = nullptr;
  mjvCamera camera_;
};

TEST_F(CameraControlTest, LeftDragOrbits) {
  CameraInput input = Drag(kDx, kDy);
  input.left_button = kButtonPressed;
  const mjvCamera expected = Moved({{mjMOUSE_ROTATE_H, kDx, 0}, {mjMOUSE_ROTATE_V, 0, kDy}});

  EXPECT_TRUE(MoveCameraWithMouse(model_, &camera_,
                                  /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, LeftDragTurnsTheFlyCamera) {
  CameraInput input = Drag(kDx, kDy);
  input.left_button = kButtonPressed;
  const mjvCamera expected = Moved({{mjMOUSE_TURN_H, kDx, 0}, {mjMOUSE_TURN_V, 0, kDy}});

  EXPECT_TRUE(MoveCameraWithMouse(model_, &camera_, /*fly_camera=*/true, input));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, MiddleDragZoomsExceptTheFlyCamera) {
  CameraInput input = Drag(kDx, kDy);
  input.middle_button = kButtonPressed;
  const mjvCamera unmoved = camera_;
  const mjvCamera expected = Moved({{mjMOUSE_ZOOM, 0, kDy}});

  EXPECT_FALSE(MoveCameraWithMouse(model_, &camera_,
                                   /*fly_camera=*/true, input));
  ExpectSameCamera(camera_, unmoved);
  EXPECT_TRUE(MoveCameraWithMouse(model_, &camera_,
                                  /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, RightDragMovesInThePlaneShiftPicks) {
  CameraInput input = Drag(kDx, kDy);
  input.right_button = kButtonPressed;
  input.shift = true;
  const mjvCamera expected = Moved({{mjMOUSE_MOVE_H, kDx, kDy}});

  EXPECT_TRUE(MoveCameraWithMouse(model_, &camera_,
                                  /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, CtrlDragMovesNothing) {
  CameraInput input = Drag(kDx, kDy);
  input.left_button = kButtonPressed;
  input.ctrl = true;
  const mjvCamera unmoved = camera_;

  EXPECT_FALSE(MoveCameraWithMouse(model_, &camera_,
                                   /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, unmoved);
}

TEST_F(CameraControlTest, ReleasedButtonMovesNothing) {
  CameraInput input = Drag(kDx, kDy);
  input.left_button = kButtonReleased;
  const mjvCamera unmoved = camera_;

  EXPECT_FALSE(MoveCameraWithMouse(model_, &camera_,
                                   /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, unmoved);
}

TEST_F(CameraControlTest, MouseWithoutFocusMovesNothing) {
  CameraInput input = Drag(kDx, kDy);
  input.left_button = kButtonPressed;
  input.mouse_wheel = 1;
  input.mouse_focus = false;
  const mjvCamera unmoved = camera_;

  EXPECT_FALSE(MoveCameraWithMouse(model_, &camera_,
                                   /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, unmoved);
}

TEST_F(CameraControlTest, WheelZoomsExceptTheFlyCamera) {
  CameraInput input = Drag(0, 0);
  input.mouse_wheel = 2;
  const mjvCamera unmoved = camera_;
  const mjvCamera expected = Moved({{mjMOUSE_ZOOM, 0, -2 / 50.0f}});

  EXPECT_FALSE(MoveCameraWithMouse(model_, &camera_,
                                   /*fly_camera=*/true, input));
  ExpectSameCamera(camera_, unmoved);
  EXPECT_TRUE(MoveCameraWithMouse(model_, &camera_,
                                  /*fly_camera=*/false, input));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, KeysFlyFasterWhileHeldAndResetOnRelease) {
  CameraInput input;
  input.keyboard_focus = true;
  input.fly_forward = 1;
  const mjvCamera expected = Moved({{mjMOUSE_MOVE_H_REL, 0, kMinFlySpeed}});
  float speed = kMinFlySpeed;

  EXPECT_TRUE(MoveCameraWithKeys(model_, &camera_, input, &speed));
  ExpectSameCamera(camera_, expected);
  EXPECT_GT(speed, kMinFlySpeed);

  input.fly_forward = 0;
  EXPECT_FALSE(MoveCameraWithKeys(model_, &camera_, input, &speed));
  EXPECT_EQ(speed, kMinFlySpeed);
}

TEST_F(CameraControlTest, KeysFlyInAllHeldDirections) {
  CameraInput input;
  input.keyboard_focus = true;
  input.fly_right = -1;
  input.fly_forward = 1;
  input.fly_up = -1;
  const mjvCamera expected = Moved(
      {{mjMOUSE_MOVE_H_REL, -kMinFlySpeed, kMinFlySpeed}, {mjMOUSE_MOVE_V_REL, 0, -kMinFlySpeed}});
  float speed = kMinFlySpeed;

  EXPECT_TRUE(MoveCameraWithKeys(model_, &camera_, input, &speed));
  ExpectSameCamera(camera_, expected);
}

TEST_F(CameraControlTest, ShiftRaisesTheFlySpeedCap) {
  CameraInput input;
  input.keyboard_focus = true;
  input.fly_up = 1;
  float speed = kMinFlySpeed;
  for (int i = 0; i < 1000; ++i) {
    MoveCameraWithKeys(model_, &camera_, input, &speed);
  }
  const float max_speed = speed;

  input.shift = true;
  for (int i = 0; i < 1000; ++i) {
    MoveCameraWithKeys(model_, &camera_, input, &speed);
  }
  EXPECT_GT(speed, max_speed);
}

TEST_F(CameraControlTest, KeysWithoutFocusMoveNothing) {
  CameraInput input;
  input.fly_forward = 1;
  const mjvCamera unmoved = camera_;
  float speed = kMinFlySpeed;

  EXPECT_FALSE(MoveCameraWithKeys(model_, &camera_, input, &speed));
  ExpectSameCamera(camera_, unmoved);
}

}  // namespace
}  // namespace mujoco::studio
