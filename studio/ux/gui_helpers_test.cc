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

#include "studio/ux/gui_helpers.h"

#include <string>

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

TEST(GuiHelpersTest, KeyframeToStringBasic) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <joint name="j1" type="hinge"/>
        <geom type="sphere" size="0.1"/>
      </body>
      <body name="mocap1" mocap="true" pos="1 2 3"/>
    </worldbody>
    <actuator>
      <general joint="j1" dyntype="integrator"/>
    </actuator>
  </mujoco>
  )";

  char error[1000] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_NE(spec, nullptr) << error;
  mjModel* model = mj_compile(spec, nullptr);
  mj_deleteSpec(spec);
  ASSERT_NE(model, nullptr);
  mjData* data = mj_makeData(model);
  ASSERT_NE(data, nullptr);

  std::string str = KeyframeToString(model, data, /*full_precision=*/false);
  EXPECT_NE(str.find("<key\n"), std::string::npos);
  EXPECT_NE(str.find("qpos=\"0\""), std::string::npos);
  EXPECT_NE(str.find("act=\"0\""), std::string::npos);
  EXPECT_NE(str.find("ctrl=\"0\""), std::string::npos);
  EXPECT_NE(str.find("mpos=\"1 2 3\""), std::string::npos);
  EXPECT_NE(str.find("mquat=\"1 0 0 0\""), std::string::npos);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(GuiHelpersTest, KeyframeToStringLargeModelDoesNotOverflowBuffer) {
  mjSpec* spec = mj_makeSpec();
  ASSERT_NE(spec, nullptr);
  mjsBody* world = mjs_findBody(spec, "world");
  ASSERT_NE(world, nullptr);

  for (int i = 0; i < 250; ++i) {
    mjsBody* b = mjs_addBody(world, nullptr);
    mjs_addFreeJoint(b);
    mjsGeom* g = mjs_addGeom(b, nullptr);
    g->type = mjGEOM_SPHERE;
    g->size[0] = 0.01;
  }

  mjModel* model = mj_compile(spec, nullptr);
  mj_deleteSpec(spec);
  ASSERT_NE(model, nullptr);
  mjData* data = mj_makeData(model);
  ASSERT_NE(data, nullptr);

  for (int i = 0; i < model->nq; ++i) {
    data->qpos[i] = static_cast<mjtNum>(1.23456);
  }
  for (int i = 0; i < model->nv; ++i) {
    data->qvel[i] = static_cast<mjtNum>(0.98765);
  }

  std::string regular = KeyframeToString(model, data, /*full_precision=*/false);
  EXPECT_GT(regular.size(), 5000);
  EXPECT_EQ(regular.substr(regular.size() - 4), "\"\n/>");

  std::string full = KeyframeToString(model, data, /*full_precision=*/true);
  EXPECT_GT(full.size(), regular.size());
  EXPECT_NE(full.find("<key\n time=\""), std::string::npos);
  EXPECT_NE(full.find("qpos=\""), std::string::npos);
  EXPECT_NE(full.find("qvel=\""), std::string::npos);
  EXPECT_EQ(full.substr(full.size() - 4), "\"\n/>");

  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mujoco::studio
