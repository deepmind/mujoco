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

// Tests for SDF plugins.

#include <cmath>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using SdfPluginTest = MujocoTest;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

TEST_F(SdfPluginTest, GearAndBoltAabbEncloseSurface) {
  const mjpPlugin* gear = mjp_getPlugin("mujoco.sdf.gear", nullptr);
  ASSERT_THAT(gear, NotNull());
  mjtNum gear_attr[5] = {0, 0.1, 4, 0.02, -1};  // D = 0.1, N = 4
  mjtNum gear_aabb[6];
  gear->sdf_aabb(gear_aabb, gear_attr);
  // Tooth tip radius Ro = (D/2) * (1 + 2/N) = 0.05 * 1.5 = 0.075
  EXPECT_GE(gear_aabb[3], 0.075);
  EXPECT_GE(gear_aabb[4], 0.075);

  const mjpPlugin* bolt = mjp_getPlugin("mujoco.sdf.bolt", nullptr);
  ASSERT_THAT(bolt, NotNull());
  mjtNum bolt_attr[1] = {1.0};
  mjtNum bolt_aabb[6];
  bolt->sdf_aabb(bolt_aabb, bolt_attr);
  // Thread outer radius = 1.0 + 1/24 < 1.2
  EXPECT_GE(bolt_aabb[3], 1.2);
  EXPECT_GE(bolt_aabb[4], 1.2);
}

TEST_F(SdfPluginTest, GearSmoothUnionNegativeKDoesNotInvert) {
  const mjpPlugin* gear = mjp_getPlugin("mujoco.sdf.gear", nullptr);
  ASSERT_THAT(gear, NotNull());
  // For N = 120, k = Rb - Ro + h < 0; root interior must remain negative.
  mjtNum gear_attr[5] = {0, 0.1, 120, 0.02, -1};
  mjtNum p_root[3] = {0.048, 0.0, 0.0};
  EXPECT_LT(gear->sdf_staticdistance(p_root, gear_attr), -1e-4);
}

TEST_F(SdfPluginTest, BowlInvalidHeightFailsWithoutNan) {
  const mjpPlugin* bowl = mjp_getPlugin("mujoco.sdf.bowl", nullptr);
  ASSERT_THAT(bowl, NotNull());
  mjtNum bad_attr[3] = {1.5, 1.0, 0.1};  // height > radius
  mjtNum p[3] = {0.2, 0.0, 0.5};
  EXPECT_FALSE(std::isnan(bowl->sdf_staticdistance(p, bad_attr)));

  static constexpr char kBowlInvalidXml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.bowl">
        <instance name="bowl">
          <config key="radius" value="1.0"/>
          <config key="height" value="1.5"/>
          <config key="thickness" value="0.1"/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="bowl">
        <plugin instance="bowl"/>
      </mesh>
    </asset>
    <worldbody>
      <geom type="sdf" mesh="bowl">
        <plugin instance="bowl"/>
      </geom>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr model = LoadModelFromString(kBowlInvalidXml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Bowl plugin"));

  // a negative height is a valid cut plane (a shallow bowl)
  std::string shallow = kBowlInvalidXml;
  shallow.replace(shallow.find("\"1.5\""), 5, "\"-0.5\"");
  MjModelPtr shallow_model =
      LoadModelFromString(shallow.c_str(), error, sizeof(error));
  EXPECT_THAT(shallow_model.get(), NotNull()) << error;
}

TEST_F(SdfPluginTest, TorusZAxisGradientIsFiniteUnitVector) {
  static constexpr char kTorusXml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="torus">
          <config key="radius1" value="1.0"/>
          <config key="radius2" value="0.25"/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="torus">
        <plugin instance="torus"/>
      </mesh>
    </asset>
    <worldbody>
      <geom type="sdf" mesh="torus">
        <plugin instance="torus"/>
      </geom>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kTorusXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  const mjpPlugin* torus = mjp_getPluginAtSlot(m->plugin[0]);
  ASSERT_THAT(torus, NotNull());

  for (mjtNum z : {mjtNum(0.0), mjtNum(0.75)}) {
    mjtNum p[3] = {0.0, 0.0, z};
    mjtNum grad[3] = {0.0, 0.0, 0.0};
    torus->sdf_gradient(grad, p, d.get(), 0);
    EXPECT_TRUE(std::isfinite(grad[0]));
    EXPECT_TRUE(std::isfinite(grad[1]));
    EXPECT_TRUE(std::isfinite(grad[2]));
    EXPECT_THAT(mju_norm3(grad), MjNear(1.0, 1e-6, 1e-5));
  }
}

// the out-of-bounds read of the last trajectory point is only detected under
// ASan/MSan, since the point it reads is unused for the sphere geoms
TEST_F(SdfPluginTest, VisualizerCapacityTailSinglePointDoesNotReadOob) {
  static constexpr char kBoltXml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.bolt">
        <instance name="bolt"/>
      </plugin>
    </extension>
    <asset>
      <mesh name="bolt">
        <plugin instance="bolt"/>
      </mesh>
    </asset>
    <worldbody>
      <geom type="sdf" mesh="bolt">
        <plugin instance="bolt"/>
      </geom>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kBoltXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  const mjpPlugin* bolt = mjp_getPluginAtSlot(m->plugin[0]);
  ASSERT_THAT(bolt, NotNull());
  bolt->reset(m.get(), nullptr, reinterpret_cast<void*>(d->plugin_data[0]), 0);

  constexpr int kCapacity = 10 * (mjMAXCONPAIR + 1) * (mjMAXCONPAIR + 1);
  mjtNum p[3] = {0.1, 0.2, -0.3};
  mjtNum grad[3];

  // Fill kCapacity - 1 points in the first trajectory.
  bolt->compute(m.get(), d.get(), 0, mjPLUGIN_SDF);
  for (int i = 0; i < kCapacity - 1; ++i) {
    bolt->sdf_gradient(grad, p, d.get(), 0);
  }
  // Final 1-point trajectory sits right at buffer capacity.
  bolt->compute(m.get(), d.get(), 0, mjPLUGIN_SDF);
  bolt->sdf_gradient(grad, p, d.get(), 0);

  mjvOption opt;
  mjv_defaultOption(&opt);
  opt.flags[mjVIS_SDFITER] = 1;

  mjvScene scn;
  mjv_defaultScene(&scn);
  mjv_makeScene(m.get(), &scn, kCapacity + 10);
  bolt->visualize(m.get(), d.get(), &opt, &scn, 0);
  EXPECT_EQ(scn.ngeom, (2 + (kCapacity - 2)) + 2);
  mjv_freeScene(&scn);
}

}  // namespace
}  // namespace mujoco
