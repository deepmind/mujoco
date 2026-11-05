// Copyright 2023 DeepMind Technologies Limited
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

// Tests for sensor plugins.

#include <string>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using SensorPluginTest = MujocoTest;

static const char* const kTOuchGridPath =
    "plugin/sensor/testdata/touch_grid_test.xml";

TEST_F(SensorPluginTest, ValidAttributes) {
  const std::string xml_path = GetTestDataFilePath(kTOuchGridPath);
  char error[1024] = {0};
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, testing::NotNull()) << error;
  int nchannel = 3;
  int size_x = 6;
  int size_y = 2;
  EXPECT_THAT(model->sensor_dim[0], nchannel * size_x * size_y);
  mj_deleteModel(model);
}

// invalid attributes fail plugin init, which is reported as a compile error;
// leaks of the partial mjData or of C++ locals are detected under asan
TEST_F(SensorPluginTest, InvalidAttributeErrorDoesNotLeak) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <worldbody>
      <site name="touch"/>
    </worldbody>
    <sensor>
      <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch">
        <config key="size" value="6 2"/>
        <config key="fov" value="0 45"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
    </sensor>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(model, testing::IsNull());
  EXPECT_THAT(error, testing::HasSubstr("`fov[0]` must be a float"));
}

TEST_F(SensorPluginTest, TouchGridNsensordataInvalidSize) {
  for (const char* bad_size : {"", "6", "-2 2", "50000 50000"}) {
    std::string xml = std::string(R"(
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sensor.touch_grid"/>
      </extension>
      <worldbody>
        <site name="touch"/>
      </worldbody>
      <sensor>
        <framepos objtype="site" objname="touch"/>
        <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch">
          <config key="size" value=")") +
                      bad_size + R"("/>
          <config key="fov" value="45 45"/>
          <config key="gamma" value="0"/>
          <config key="nchannel" value="1"/>
        </plugin>
      </sensor>
    </mujoco>
    )";
    char error[1024] = {0};
    MjModelPtr model = LoadModelFromString(xml.c_str(), error, sizeof(error));
    EXPECT_THAT(model, testing::IsNull()) << "size=\"" << bad_size << "\"";
  }
}

TEST_F(SensorPluginTest, TouchGridWeldedChildContactForceSign) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 0.01"/>
      <body name="sensor_root" pos="0 0 0.09">
        <freejoint/>
        <site name="touch_down" pos="0 0 0"/>
        <site name="touch_up" pos="0 0 0" euler="180 0 0"/>
        <body name="welded_child">
          <geom name="ball" type="sphere" size="0.1" mass="1"/>
        </body>
      </body>
      <body name="top_weight" pos="0 0 0.27">
        <freejoint/>
        <geom name="top_ball" type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <sensor>
      <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_down">
        <config key="size" value="1 1"/>
        <config key="fov" value="45 45"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
      <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_up">
        <config key="size" value="1 1"/>
        <config key="fov" value="45 45"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
    </sensor>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, testing::NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, testing::NotNull());
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 2);
  // floor vs welded_child (weld2 == parent_weld, parent_body < body2)
  EXPECT_GT(data->sensordata[0], 0);
  // welded_child vs top_weight (weld1 == parent_weld)
  EXPECT_GT(data->sensordata[1], 0);
}

// contact pairs are ordered by geom type, not by body id: the sphere on the
// higher-id body is geom1 and the box on the sensor body is geom2
TEST_F(SensorPluginTest, TouchGridGeomTypeOrderContactForceSign) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 0.01"/>
      <body name="sensor_box" pos="0 0 0.09">
        <freejoint/>
        <geom name="box" type="box" size="0.1 0.1 0.1" mass="1"/>
        <site name="touch_up" pos="0 0 0" euler="180 0 0"/>
      </body>
      <body name="top_weight" pos="0 0 0.27">
        <freejoint/>
        <geom name="top_ball" type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <sensor>
      <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch_up">
        <config key="size" value="1 1"/>
        <config key="fov" value="45 45"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
    </sensor>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, testing::NotNull()) << error;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, testing::NotNull());
  mj_forward(model.get(), data.get());
  int ball = mj_name2id(model.get(), mjOBJ_GEOM, "top_ball");
  int box = mj_name2id(model.get(), mjOBJ_GEOM, "box");
  bool found = false;
  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].geom[0] == ball && data->contact[i].geom[1] == box) {
      found = true;
    }
  }
  ASSERT_TRUE(found);
  EXPECT_GT(data->sensordata[0], 0);
}

}  // namespace
}  // namespace mujoco
