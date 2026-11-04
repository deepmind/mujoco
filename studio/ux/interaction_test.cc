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

#include "studio/ux/interaction.h"

#include <array>

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>

namespace mujoco::studio {
namespace {

TEST(InteractionTest, PickSkinDoesNotCrashOnUninflatedSkin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b0" pos="0 0 1"/>
    </worldbody>
    <deformable>
      <skin name="skin" vertex="0 0 1 1 0 1 0 1 1" face="0 1 2">
        <bone body="b0" bindpos="0 0 1" bindquat="1 0 0 0" vertid="0 1 2"
          vertweight="1 1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_NE(spec, nullptr) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  mj_deleteSpec(spec);
  ASSERT_NE(m, nullptr);
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  mjvCamera cam;
  mjv_defaultCamera(&cam);
  cam.lookat[0] = 0.33;
  cam.lookat[1] = 0.33;
  cam.lookat[2] = 1.0;
  cam.distance = 2.0;

  mjvOption opt;
  mjv_defaultOption(&opt);

  PickResult res = Pick(m, d, &cam, 0.5f, 0.5f, 1.0f, &opt);
  EXPECT_EQ(res.skin, 0);
  EXPECT_EQ(res.body, 1);

  mj_deleteData(d);
  mj_deleteModel(m);
}

}  // namespace
}  // namespace mujoco::studio
