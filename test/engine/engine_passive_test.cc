// Copyright 2021 DeepMind Technologies Limited
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

// Tests for engine/engine_core_smooth.c.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_core_util.h"
#include "src/engine/engine_metric.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

using PassiveTest = MujocoTest;

TEST_F(PassiveTest, DisableFlags) {
  static constexpr char flex_xml[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body gravcomp="1">
        <joint type="slide" springref="1" stiffness="10" damping="1"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qvel="-1"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(flex_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qacc[0], 11);

  m->opt.disableflags = mjDSBL_DAMPER;
  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qacc[0], 10);

  m->opt.disableflags = mjDSBL_SPRING;
  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qacc[0], 1);

  m->opt.disableflags = mjDSBL_SPRING | mjDSBL_DAMPER;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->qacc[0], -10);
}

TEST_F(PassiveTest, GravcompNestedBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body pos="0 0 2">
        <freejoint/>
        <body gravcomp="1.2">
          <geom size="0.2" mass="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_forward(m.get(), d.get());

  EXPECT_GT(d->qacc[2], 0);
  EXPECT_NEAR(d->qacc[2], 2.0, 0.1);
}

TEST_F(PassiveTest, PolyStiffnessSlide) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" stiffness="10 5 1"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qfrc_spring[0], -48);
}

TEST_F(PassiveTest, PolyStiffnessAntiSymmetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" stiffness="10 5 1"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qpos="-2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qfrc_spring[0], 8);
}

TEST_F(PassiveTest, PolyStiffnessTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" name="j"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <tendon>
      <fixed>
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  m->tendon_stiffness[0] = 10;
  m->tendon_stiffnesspoly[0] = 5;
  m->tendon_stiffnesspoly[1] = 1;

  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->qfrc_spring[0], -48);
}

TEST_F(PassiveTest, PolyStiffnessEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.0001">
      <flag energy="enable"/>
    </option>

    <worldbody>
      <body>
        <joint type="slide" stiffness="10 5 1"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  mjtNum total_energy = d->energy[0] + d->energy[1];

  for (int i = 0; i < 100; i++) {
    mj_step(m.get(), d.get());
    EXPECT_NEAR(d->energy[0] + d->energy[1], total_energy, 0.002);
  }
}

// ------------------------ ellipsoid fluid model ------------------------------

using EllipsoidFluidTest = MujocoTest;

TEST_F(EllipsoidFluidTest, GeomsEquivalentToBodies) {
  static constexpr char two_bodies_xml[] = R"(
  <mujoco>
    <option wind="5 5 0" density="10"/>
    <worldbody>
      <body>
        <freejoint/>
        <body>
          <geom type="box" size=".1 .01 0.01" pos="0.1 0 0" euler="40 0 0" fluidshape="ellipsoid"/>
        </body>
        <body>
          <geom type="box" size=".1 .01 0.01" pos="-.1 0 0" euler="0 20 0" fluidshape="ellipsoid"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m2 = LoadModelFromString(two_bodies_xml, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);
  for (int i = 0; i < 6; i++) {
    d2->qvel[i] = (mjtNum)i + 1;
  }
  d2->qpos[3] = 0.5;
  d2->qpos[4] = 0.5;
  d2->qpos[5] = 0.5;
  d2->qpos[6] = 0.5;

  static constexpr char one_body_xml[] = R"(
  <mujoco>
    <option wind="5 5 0" density="10"/>
    <worldbody>
      <body pos="1 2 3">
        <freejoint align="false"/>
        <geom type="box" size=".1 .01 0.01" pos="0.1 0 0" euler="40 0 0" fluidshape="ellipsoid"/>
        <geom type="box" size=".1 .01 0.01" pos="-.1 0 0" euler="0 20 0" fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr m1 = LoadModelFromString(one_body_xml, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);
  for (int i = 0; i < 6; i++) {
    d1->qvel[i] = (mjtNum)i + 1;
  }
  d1->qpos[3] = 0.5;
  d1->qpos[4] = 0.5;
  d1->qpos[5] = 0.5;
  d1->qpos[6] = 0.5;

  // tolerance for floating point numbers
  const mjtNum tol = MjTol(1e-14, 1e-5);

  EXPECT_EQ(m1->nv, m2->nv);

  mj_forward(m2.get(), d2.get());
  mj_forward(m1.get(), d1.get());
  for (int i = 0; i < m1->nv; i++) {
    EXPECT_NEAR(d2->qfrc_passive[i], d1->qfrc_passive[i], tol);
  }
}

TEST_F(EllipsoidFluidTest, DefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option wind="5 5 0" density="10"/>
    <default>
      <geom fluidshape="ellipsoid" fluidcoef="2 3 4 5 6"/>
      <default class="test_class">
        <geom fluidshape="none" fluidcoef="5 4 3 2 1"/>
      </default>
    </default>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .01 0.01" pos="0.1 0 0" class="test_class"/>
        <geom type="box" size=".1 .01 0.01" pos="-0.1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  EXPECT_THAT(AsVector(model->geom_fluid, 6), ElementsAre(0, 0, 0, 0, 0, 0));
  EXPECT_THAT(AsVector(model->geom_fluid + mjNFLUID, 6),
              ElementsAre(1, 2, 3, 4, 5, 6));
}

TEST_F(EllipsoidFluidTest, KuttaLiftLowSpeedScaling) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option density="1.225"/>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="ellipsoid" size=".025 .01 .001" euler="0 -30 0"
              fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // reference lift/v^2 at v = 10 m/s
  d->qvel[0] = 10.0;
  mj_forward(m.get(), d.get());
  const mjtNum ref_lift_coeff = d->qfrc_fluid[2] / (10.0 * 10.0);
  EXPECT_GT(ref_lift_coeff, 0.0);

  // verify exact v^2 scaling at low speeds (1 m/s, 0.1 m/s, 1 cm/s, 1 mm/s)
  for (mjtNum v : {1.0, 0.1, 0.01, 0.001}) {
    mj_resetData(m.get(), d.get());
    d->qvel[0] = v;
    mj_forward(m.get(), d.get());
    EXPECT_NEAR(d->qfrc_fluid[2] / (v * v), ref_lift_coeff,
                MjTol(1e-12, 1e-5) * ref_lift_coeff);
  }
}

// ------------------------------ tendons --------------------------------------

using TendonTest = MujocoTest;

// check tendon spring deadband using example model
TEST_F(TendonTest, SpringrangeDeadband) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/tendon_springlength.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  // initial state outside deadband: spring is active
  mj_forward(model, data);
  mjtNum expected_force = model->tendon_stiffness[0] *
                          (model->tendon_lengthspring[1] - data->ten_length[0]);
  EXPECT_EQ(expected_force, data->qfrc_passive[0]);

  // put body inside deadband: spring is inactive
  data->qpos[0] = -1;
  mj_forward(model, data);
  EXPECT_EQ(0, data->qfrc_passive[0]);

  mj_deleteData(data);
  mj_deleteModel(model);
}

// -------------------------------- flex ------------------------------------

constexpr int kFlexStiffnessStride2D = 21;
constexpr int kFlexStiffnessStride3D = 24;

using ElasticityTest = MujocoTest;

TEST_F(ElasticityTest, FlexCompatibility) {
  static constexpr char flex_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <flexcomp name="soft" type="grid" count="3 3 3"
                  radius="0.01" dim="3"mass="1">
            <pin id="2"/>
            <elasticity young="5e4" poisson="0.2"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(flex_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;

  MjDataPtr d = MakeData(m);
}

// -------------------------------- shell -----------------------------------
TEST_F(ElasticityTest, ElasticEnergyShell) {
  static constexpr char cantilever_xml[] = R"(
  <mujoco>
  <worldbody>
    <flexcomp type="grid" count="8 8 1" spacing="1 1 1"
              radius=".025" name="test" dim="2">
      <elasticity young="2" poisson="0" thickness="1"/>
    </flexcomp>
  </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(cantilever_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_kinematics(m.get(), d.get());
  mj_flex(m.get(), d.get());

  // check that a plane is in the kernel of the energy
  for (mjtNum scale = 1; scale < 4; scale++) {
    for (int e = 0; e < m->flex_edgenum[0]; e++) {
      int* edge = m->flex_edge + 2 * (m->flex_edgeadr[0] + e);
      int* flap = m->flex_edgeflap + 2 * (m->flex_edgeadr[0] + e);
      int v[4] = {edge[0], edge[1], flap[0], flap[1]};
      if (v[3] == -1) {
        continue;
      }
      mjtNum energy = 0;
      mjtNum volume = 1. / 2.;
      for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
          for (int x = 0; x < 3; x++) {
            mjtNum elongation1 = scale * d->flexvert_xpos[3 * v[i] + x];
            mjtNum elongation2 = scale * d->flexvert_xpos[3 * v[j] + x];
            energy +=
                m->flex_bending[17 * e + 4 * i + j] * elongation1 * elongation2;
          }
        }
      }
      EXPECT_NEAR(4 * energy / volume, 0,
                  std::numeric_limits<float>::epsilon());
    }
  }
}

TEST_F(ElasticityTest, CurvedShell) {
  static constexpr char cantilever_xml[] = R"(
  <mujoco>
  <worldbody>
    <body name="v0" pos="-0.5 -0.5 -0.5">
      <inertial pos="0 0 0" mass="0.125" diaginertia="1e-4 1e-4 1e-4"/>
      <joint axis="1 0 0" type="slide"/>
      <joint axis="0 1 0" type="slide"/>
      <joint axis="0 0 1" type="slide"/>
    </body>
    <body name="v1" pos="-0.5 0.5 -0.5">
      <inertial pos="0 0 0" mass="0.125" diaginertia="1e-4 1e-4 1e-4"/>
      <joint axis="1 0 0" type="slide"/>
      <joint axis="0 1 0" type="slide"/>
      <joint axis="0 0 1" type="slide"/>
    </body>
    <body name="v2" pos="0.5 -0.5 -0.5">
      <inertial pos="0 0 0" mass="0.125" diaginertia="1e-4 1e-4 1e-4"/>
      <joint axis="1 0 0" type="slide"/>
      <joint axis="0 1 0" type="slide"/>
      <joint axis="0 0 1" type="slide"/>
    </body>
    <body name="v3" pos="-0.5 -0.5 0.5">
      <inertial pos="0 0 0" mass="0.125" diaginertia="1e-4 1e-4 1e-4"/>
      <joint axis="1 0 0" type="slide"/>
      <joint axis="0 1 0" type="slide"/>
      <joint axis="0 0 1" type="slide"/>
    </body>
  </worldbody>
  <deformable>
    <flex name="test" radius="0.025" flatskin="true" body="v0 v1 v2 v3"
          element="0 2 3 0 3 1">
      <elasticity young="2" thickness="1" elastic2d="bend"/>
    </flex>
  </deformable>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(cantilever_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_kinematics(m.get(), d.get());
  mj_comPos(m.get(), d.get());
  mj_flex(m.get(), d.get());
  mj_passive(m.get(), d.get());

  // v1 force component is in-plane along v1-v0 edge (y-axis)
  EXPECT_NEAR(d->qfrc_spring[3], 0, 1e-6);
  EXPECT_NEAR(d->qfrc_spring[5], 0, 1e-6);

  // v2 force component is in-plane along v2-v0 edge (x-axis)
  EXPECT_NEAR(d->qfrc_spring[7], 0, 1e-6);
  EXPECT_NEAR(d->qfrc_spring[8], 0, 1e-6);

  // v3 force component is in-plane along v3-v0 edge (z-axis)
  EXPECT_NEAR(d->qfrc_spring[9], 0, 1e-6);
  EXPECT_NEAR(d->qfrc_spring[10], 0, 1e-6);
}

// curved non-isosceles diamond with interior hinge (0, 1), all lengths
// (including the thickness) scaled by s
static MjModelPtr CurvedDiamond(double s, const char* elasticity_extra,
                                bool pin) {
  const double point[12] = {0, 0, 0, 1, 0, 0, 0.3, 0.45, 0.15, 0.65, -0.5, 0.2};
  std::string points;
  for (double x : point) points += " " + std::to_string(s * x);
  std::string xml = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1" point=")" +
                    points + R"(" element="0 1 2  0 3 1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness=")" +
                    std::to_string(0.02 * s) + R"(" elastic2d="bend" )" +
                    elasticity_extra + R"(/>)" +
                    (pin ? R"(<pin id="0 1 2"/>)" : "") + R"(
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
  EXPECT_THAT(m.get(), NotNull()) << error;
  return m;
}

// the bending forces of a curved shell, including its curved reference term,
// scale as s^2 when all lengths are scaled by s
TEST_F(ElasticityTest, CurvedShellBendingScaleInvariance) {
  MjModelPtr m1 = CurvedDiamond(1, "", false);
  ASSERT_THAT(m1.get(), NotNull());

  // the curved reference term of the interior hinge is active
  int interior = -1;
  for (int e = 0; e < m1->flex_edgenum[0]; e++) {
    if (m1->flex_edgeflap[2 * e + 1] != -1) interior = e;
  }
  ASSERT_GE(interior, 0);
  ASSERT_NE(m1->flex_bending[17 * interior + 16], 0);

  MjDataPtr d1 = MakeData(m1);
  mj_forward(m1.get(), d1.get());
  std::vector<mjtNum> frc1(d1->qfrc_spring, d1->qfrc_spring + m1->nv);
  mjtNum frc1_norm = mju_norm(frc1.data(), m1->nv);
  EXPECT_GT(frc1_norm, 0);

  // at the curved rest shape, the quadratic and cubic terms cancel along the
  // out-of-plane triangle normals at the flap vertices v2 and v3 (leaving only
  // in-plane residual forces, as in Garg et al. 2007)
  const mjtNum* x0 = d1->flexvert_xpos;
  mjtNum e0[3], e1[3], e2[3], n0[3], n1[3];
  mju_sub3(e0, x0 + 3, x0);
  mju_sub3(e1, x0 + 6, x0);
  mju_sub3(e2, x0 + 9, x0);
  mju_cross(n0, e0, e1);
  mju_normalize3(n0);
  mju_cross(n1, e2, e0);
  mju_normalize3(n1);
  EXPECT_NEAR(mju_dot3(frc1.data() + 6, n0), 0, MjTol(1e-8, 1e-4) * frc1_norm);
  EXPECT_NEAR(mju_dot3(frc1.data() + 9, n1), 0, MjTol(1e-8, 1e-4) * frc1_norm);

  // without the curved reference term, the out-of-plane forces do not cancel
  mjtNum b16 = m1->flex_bending[17 * interior + 16];
  m1->flex_bending[17 * interior + 16] = 0;
  mj_forward(m1.get(), d1.get());
  EXPECT_GT(std::abs(mju_dot3(d1->qfrc_spring + 6, n0)), 0.1 * frc1_norm);
  EXPECT_GT(std::abs(mju_dot3(d1->qfrc_spring + 9, n1)), 0.1 * frc1_norm);
  m1->flex_bending[17 * interior + 16] = b16;

  for (double s : {0.1, 10.0}) {
    MjModelPtr m = CurvedDiamond(s, "", false);
    ASSERT_THAT(m.get(), NotNull());
    ASSERT_EQ(m->nv, m1->nv);
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i] / (s * s), frc1[i],
                  MjTol(1e-8, 1e-4) * frc1_norm)
          << "scale " << s << ", dof " << i;
    }
  }
}

// the bending return mapping of a curved shell is scale invariant: folding the
// flap by the same angle gives the same plastic state at every length scale
TEST_F(ElasticityTest, CurvedShellBendingPlasticScaleInvariance) {
  auto fold = [](double s) -> std::vector<mjtNum> {
    MjModelPtr m = CurvedDiamond(s, R"(yield="0.002")", true);
    EXPECT_THAT(m.get(), NotNull());
    if (!m) return {};
    EXPECT_EQ(m->nplastic, m->flex_edgenum[0]);
    MjDataPtr d = MakeData(m);

    // rotate the free flap vertex v3 by 0.3 rad around the hinge (x-axis)
    double y = -0.5 * s, z = 0.2 * s, a = 0.3;
    d->qpos[1] = y * std::cos(a) - z * std::sin(a) - y;
    d->qpos[2] = y * std::sin(a) + z * std::cos(a) - z;
    mj_step(m.get(), d.get());
    return std::vector<mjtNum>(d->plastic, d->plastic + m->nplastic);
  };

  std::vector<mjtNum> p1 = fold(1);
  ASSERT_FALSE(p1.empty());
  mjtNum p1_norm = mju_norm(p1.data(), p1.size());
  EXPECT_GT(p1_norm, 0);
  for (double s : {0.1, 10.0}) {
    std::vector<mjtNum> p = fold(s);
    ASSERT_EQ(p.size(), p1.size());
    for (int i = 0; i < (int)p.size(); i++) {
      EXPECT_NEAR(p[i], p1[i], MjTol(1e-8, 1e-4) * p1_norm)
          << "scale " << s << ", entry " << i;
    }
  }
}

// -------------------------------- membrane -----------------------------------
TEST_F(ElasticityTest, ElasticEnergyMembrane) {
  static constexpr char cantilever_xml[] = R"(
  <mujoco>
  <worldbody>
    <flexcomp type="grid" count="8 8 1" spacing="1 1 1"
              radius=".025" name="test" dim="2">
      <elasticity young="2" poisson="0.3" thickness="0.5" elastic2d="stretch"/>
      <edge equality="false"/>
    </flexcomp>
  </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(cantilever_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_kinematics(m.get(), d.get());
  mj_flex(m.get(), d.get());
  mjtNum* metric =
      m->flex_stiffness + kFlexStiffnessStride2D * m->flex_elemadr[0];

  // if the edge elongations are s = scale * L0^2, then sum_e s_e basis_e is
  // the tensor scale * I in the plane of the element, and s' K s equals the
  // plane-stress energy density mu tr(T^2) + lambda/2 tr(T)^2 times the
  // element volume (area times thickness)
  const mjtNum young = 2, poisson = 0.3, thickness = 0.5;
  const mjtNum mu = young / (2 * (1 + poisson));
  const mjtNum lambda = young * poisson / (1 - poisson * poisson);

  for (mjtNum scale = 1; scale < 4; scale++) {
    for (int t = 0; t < m->flex_elemnum[0]; t++) {
      mjtNum energy = 0;
      mjtNum volume = thickness / 2;
      int idx = 0;
      for (int e1 = 0; e1 < 3; e1++) {
        for (int e2 = e1; e2 < 3; e2++) {
          int idx1 = m->flex_elemedge[3 * t + e1 + m->flex_elemedgeadr[0]];
          int idx2 = m->flex_elemedge[3 * t + e2 + m->flex_elemedgeadr[0]];
          mjtNum elong1 =
              scale * m->flexedge_length0[idx1] * m->flexedge_length0[idx1];
          mjtNum elong2 =
              scale * m->flexedge_length0[idx2] * m->flexedge_length0[idx2];
          energy += metric[kFlexStiffnessStride2D * t + idx++] * elong1 *
                    elong2 * (e1 == e2 ? 1. : 2.);
        }
      }
      mjtNum expected = mu * 2 * scale * scale + lambda / 2 * 4 * scale * scale;
      const mjtNum tol = MjTol(std::numeric_limits<float>::epsilon(), 1e-5);
      EXPECT_NEAR(energy / volume / expected, 1, tol);
    }
  }
}

// map flex vertices to their x, y, z slider qpos addresses
static std::vector<int> FlexVertexQpos(const mjModel* m, int f) {
  std::vector<int> qadr(3 * m->flex_vertnum[f], -1);
  for (int v = 0; v < m->flex_vertnum[f]; v++) {
    int b = m->flex_vertbodyid[m->flex_vertadr[f] + v];
    if (m->body_jntnum[b] != 3) continue;
    for (int k = 0; k < 3; k++) {
      int j = m->body_jntadr[b] + k;
      if (m->jnt_type[j] != mjJNT_SLIDE || m->jnt_axis[3 * j + k] != 1) {
        return {};
      }
      qadr[3 * v + k] = m->jnt_qposadr[j];
    }
  }
  return qadr;
}

// elastic energy stored by moving from qpos0 to qpos0 + dq, computed as the
// work of qfrc_spring along a straight path; exact for polynomial energies of
// degree up to 8 in dq
static mjtNum SpringWork(const mjModel* m, mjData* d,
                         const std::vector<mjtNum>& dq) {
  static constexpr mjtNum kNode[4] = {0.0694318442029737, 0.3300094782075719,
                                      0.6699905217924281, 0.9305681557970263};
  static constexpr mjtNum kWeight[4] = {0.1739274225687269, 0.3260725774312731,
                                        0.3260725774312731, 0.1739274225687269};
  mjtNum work = 0;
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < m->nq; j++) {
      d->qpos[j] = m->qpos0[j] + kNode[i] * dq[j];
    }
    mj_forward(m, d);
    work -= kWeight[i] * mju_dot(d->qfrc_spring, dq.data(), m->nv);
  }
  return work;
}

// energy of homogeneous deformations x = F X of a unit square sheet, compared
// to plane-stress Saint Venant-Kirchhoff: thickness * (mu tr(E^2) +
// lambda*/2 tr(E)^2), with lambda* = young poisson / (1 - poisson^2)
TEST_F(ElasticityTest, ShellMembranePlaneStress) {
  for (mjtNum poisson : {0.0, 0.3, 0.45}) {
    char xml[1024];
    std::snprintf(xml, sizeof(xml), R"(
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <flexcomp type="grid" count="5 5 1" spacing=".25 .25 .25" radius=".001"
                  name="sheet" dim="2" mass="1">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="1e3" poisson="%g" thickness="0.01" elastic2d="stretch"/>
          <edge equality="false"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    )",
                  poisson);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    std::vector<int> qadr = FlexVertexQpos(m.get(), 0);
    ASSERT_FALSE(qadr.empty());
    mj_forward(m.get(), d.get());
    std::vector<mjtNum> x0(d->flexvert_xpos,
                           d->flexvert_xpos + 3 * m->flex_vertnum[0]);

    const mjtNum young = 1e3, thickness = 0.01, area = 1;
    const mjtNum mu = young / (2 * (1 + poisson));
    const mjtNum lambda = young * poisson / (1 - poisson * poisson);

    // uniaxial, biaxial, simple shear, and a large mixed deformation
    const mjtNum deformations[4][4] = {{1.01, 0, 0, 1},
                                       {1.01, 0, 0, 1.01},
                                       {1, 0.01, 0, 1},
                                       {1.3, 0.2, -0.1, 0.8}};
    for (const auto& F : deformations) {
      std::vector<mjtNum> dq(m->nq, 0);
      for (int v = 0; v < m->flex_vertnum[0]; v++) {
        const mjtNum* X = x0.data() + 3 * v;
        dq[qadr[3 * v + 0]] = F[0] * X[0] + F[1] * X[1] - X[0];
        dq[qadr[3 * v + 1]] = F[2] * X[0] + F[3] * X[1] - X[1];
      }
      mjtNum energy = SpringWork(m.get(), d.get(), dq);

      // Green strain E = (F'F - I) / 2
      mjtNum e00 = (F[0] * F[0] + F[2] * F[2] - 1) / 2;
      mjtNum e11 = (F[1] * F[1] + F[3] * F[3] - 1) / 2;
      mjtNum e01 = (F[0] * F[1] + F[2] * F[3]) / 2;
      mjtNum tr = e00 + e11;
      mjtNum trsq = e00 * e00 + e11 * e11 + 2 * e01 * e01;
      mjtNum expected = area * thickness * (mu * trsq + lambda / 2 * tr * tr);
      EXPECT_NEAR(energy / expected, 1, MjTol(1e-8, 1e-3))
          << "poisson " << poisson << ", F " << F[0] << " " << F[1] << " "
          << F[2] << " " << F[3];
    }
  }
}

// -------------------------------- solid -----------------------------------
TEST_F(ElasticityTest, ElasticEnergySolid) {
  static constexpr char cantilever_xml[] = R"(
  <mujoco>
  <worldbody>
    <flexcomp type="grid" count="8 8 8" spacing="1 1 1"
              radius=".025" name="test" dim="3">
      <elasticity young="2" poisson="0"/>
      <edge equality="false"/>
    </flexcomp>
  </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(cantilever_xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_kinematics(m.get(), d.get());
  mj_flex(m.get(), d.get());
  mjtNum* metric = m->flex_stiffness + m->flex_stiffnessadr[0];

  // check that if the entire geometry is rescaled by a factor "scale", then
  // trace(strain^2) = 3*scale^2

  for (mjtNum scale = 1; scale < 4; scale++) {
    for (int t = 0; t < m->flex_elemnum[0]; t++) {
      mjtNum energy = 0;
      mjtNum volume = 1. / 6.;
      int idx = 0;
      for (int e1 = 0; e1 < 6; e1++) {
        for (int e2 = e1; e2 < 6; e2++) {
          int idx1 = m->flex_elemedge[6 * t + e1 + m->flex_elemedgeadr[0]];
          int idx2 = m->flex_elemedge[6 * t + e2 + m->flex_elemedgeadr[0]];
          mjtNum elong1 =
              scale * m->flexedge_length0[idx1] * m->flexedge_length0[idx1];
          mjtNum elong2 =
              scale * m->flexedge_length0[idx2] * m->flexedge_length0[idx2];
          energy += metric[kFlexStiffnessStride3D * t + idx++] * elong1 *
                    elong2 * (e1 == e2 ? 1. : 2.);
        }
      }
      const mjtNum tol = MjTol(std::numeric_limits<float>::epsilon(), 1e-4);
      EXPECT_NEAR(energy / volume, 3 * scale * scale, tol);
    }
  }
}

// energy of homogeneous deformations x = F X of a unit cube, compared to Saint
// Venant-Kirchhoff: mu tr(E^2) + lambda/2 tr(E)^2
TEST_F(ElasticityTest, SolidStVKEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="3 3 3" spacing=".5 .5 .5" radius=".001"
                name="cube" dim="3" mass="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1e3" poisson="0.3"/>
        <edge equality="false"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  std::vector<int> qadr = FlexVertexQpos(m.get(), 0);
  ASSERT_FALSE(qadr.empty());
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> x0(d->flexvert_xpos,
                         d->flexvert_xpos + 3 * m->flex_vertnum[0]);

  const mjtNum young = 1e3, poisson = 0.3, volume = 1;
  const mjtNum mu = young / (2 * (1 + poisson));
  const mjtNum lambda = young * poisson / ((1 + poisson) * (1 - 2 * poisson));

  // dilatation, simple shear, and a large mixed deformation
  const mjtNum deformations[3][9] = {{1.01, 0, 0, 0, 1.01, 0, 0, 0, 1.01},
                                     {1, 0.01, 0, 0, 1, 0, 0, 0, 1},
                                     {1.2, 0.1, 0, 0, 0.9, 0.05, 0.02, 0, 1.1}};
  for (const auto& F : deformations) {
    std::vector<mjtNum> dq(m->nq, 0);
    for (int v = 0; v < m->flex_vertnum[0]; v++) {
      mjtNum x[3];
      mju_mulMatVec3(x, F, x0.data() + 3 * v);
      for (int k = 0; k < 3; k++) {
        dq[qadr[3 * v + k]] = x[k] - x0[3 * v + k];
      }
    }
    mjtNum energy = SpringWork(m.get(), d.get(), dq);

    // Green strain E = (F'F - I) / 2
    mjtNum C[9], E[9];
    mju_mulMatTMat(C, F, F, 3, 3, 3);
    for (int i = 0; i < 9; i++) {
      E[i] = (C[i] - (i % 4 == 0)) / 2;
    }
    mjtNum tr = E[0] + E[4] + E[8];
    mjtNum trsq = mju_dot(E, E, 9);
    mjtNum expected = volume * (mu * trsq + lambda / 2 * tr * tr);
    EXPECT_NEAR(energy / expected, 1, MjTol(1e-8, 1e-3));
  }
}

// -------------------------------- shell bending ------------------------------
// bending a flat sheet into a cylinder of curvature k is an isometry with
// Kirchhoff energy density D k^2 / 2, D = E t^3 / (12 (1 - nu^2)), for any nu;
// boundary hinges are not included, so the discrete energy converges as 1/n
TEST_F(ElasticityTest, ShellBendingCylinder) {
  for (mjtNum poisson : {0.0, 0.3}) {
    mjtNum prev_error = 1;
    for (int n : {8, 16, 32}) {
      char xml[1024];
      std::snprintf(xml, sizeof(xml), R"(
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <flexcomp type="grid" count="%d %d 1" spacing="%.17g %.17g 1"
                    radius=".001" name="sheet" dim="2" mass="1">
            <contact contype="0" conaffinity="0" selfcollide="none"/>
            <elasticity young="1e5" poisson="%g" thickness="0.01"
                        elastic2d="bend"/>
            <edge equality="false"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      )",
                    n + 1, n + 1, 1.0 / n, 1.0 / n, poisson);
      char error[1024] = {0};
      MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
      ASSERT_THAT(m.get(), NotNull()) << error;
      MjDataPtr d = MakeData(m);
      std::vector<int> qadr = FlexVertexQpos(m.get(), 0);
      ASSERT_FALSE(qadr.empty());
      mj_forward(m.get(), d.get());
      std::vector<mjtNum> x0(d->flexvert_xpos,
                             d->flexvert_xpos + 3 * m->flex_vertnum[0]);
      mjtNum xmean = 0;
      for (int v = 0; v < m->flex_vertnum[0]; v++) xmean += x0[3 * v];
      xmean /= m->flex_vertnum[0];

      // displace by z = k x^2 / 2; for a flat reference the bending force is
      // linear in the positions, so the energy is -qfrc_spring' dq / 2
      const mjtNum k = 1e-2;
      std::vector<mjtNum> dq(m->nq, 0);
      for (int v = 0; v < m->flex_vertnum[0]; v++) {
        mjtNum x = x0[3 * v] - xmean;
        dq[qadr[3 * v + 2]] = k * x * x / 2;
      }
      mju_copy(d->qpos, dq.data(), m->nq);
      mj_forward(m.get(), d.get());
      mjtNum energy = -mju_dot(d->qfrc_spring, dq.data(), m->nv) / 2;

      const mjtNum young = 1e5, thickness = 0.01, area = 1;
      mjtNum D = young * thickness * thickness * thickness /
                 (12 * (1 - poisson * poisson));
      mjtNum ratio = energy / (D * k * k / 2 * area);
      EXPECT_NEAR(ratio, 1 - 1.0 / n, MjTol(1e-6, 1e-3))
          << "n " << n << ", poisson " << poisson;
      EXPECT_LT(1 - ratio, prev_error);
      prev_error = 1 - ratio;
    }
  }
}

// Evaluate the expanded energy independently of the engine's derivative
// helpers.
static mjtNum SNHElementEnergy(const mjtNum* k, const mjtNum s[6], mjtNum J) {
  mjtNum energy = k[22] * (J - 1) * (J - 1);
  int index = 0;
  for (int i = 0; i < 6; i++) {
    for (int j = i; j < 6; j++) {
      energy += (i == j ? .25 : .5) * k[index++] * s[i] * s[j];
    }
  }
  mjtNum a = s[0], b = s[2], c = s[4];
  mjtNum d = 0.5 * (a + b - s[1]), e = 0.5 * (a + c - s[5]);
  mjtNum f = 0.5 * (b + c - s[3]);
  return energy + k[21] * (a * b * c + 2 * d * e * f - a * f * f - b * e * e -
                           c * d * d);
}

// Every tetrahedral record reproduces the compact SNH energy under signed
// dilation.
TEST_F(ElasticityTest, SNHEnergySolid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp type="grid" count="3 3 3" spacing="1 1 1" name="test" dim="3">
        <elasticity young="2" poisson="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  spec->option.integrator = mjINT_DISCRETE;
  mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "test"))->elastic3d = 1;
  MjModelPtr m(mj_compile(spec, nullptr));
  mj_deleteSpec(spec);
  ASSERT_THAT(m.get(), NotNull());
  EXPECT_EQ(m->nflexstiffness, kFlexStiffnessStride3D * m->nflexelem);
  for (mjtNum scale :
       {mjtNum(-1), mjtNum(0), mjtNum(.3), mjtNum(1), mjtNum(2)}) {
    mjtNum scale2 = scale * scale;
    mjtNum J = scale2 * scale;
    mjtNum expected =
        (1.5 * (scale2 - 1) - (J - 1) + .5 * (J - 1) * (J - 1)) / 6;
    for (int t = 0; t < m->nflexelem; t++) {
      const mjtNum* k = m->flex_stiffness + m->flex_stiffnessadr[0] +
                        kFlexStiffnessStride3D * t;
      mjtNum s[6];
      for (int e = 0; e < 6; e++) {
        int idx = m->flex_elemedge[m->flex_elemedgeadr[0] + 6 * t + e];
        mjtNum length = m->flexedge_length0[m->flex_edgeadr[0] + idx];
        s[e] = length * length * (scale2 - 1);
      }
      EXPECT_NEAR(SNHElementEnergy(k, s, J), expected, MjTol(1e-12, 1e-5));
    }
  }
}

// Independent deformation-gradient evaluation on an irregular reference
// tetrahedron.
static mjtNum SNHEnergy(const mjData* d) {
  const mjtNum* p = d->flexvert_xpos;
  mjtNum F[9];
  for (int x = 0; x < 3; x++) {
    F[3 * x] = p[3 + x] - p[x];
    F[3 * x + 1] = (p[6 + x] - p[x] - .2 * F[3 * x]) / .9;
    F[3 * x + 2] = (p[9 + x] - p[x] + .1 * F[3 * x] - .3 * F[3 * x + 1]) / 1.1;
  }
  mjtNum J = F[0] * (F[4] * F[8] - F[5] * F[7]) -
             F[1] * (F[3] * F[8] - F[5] * F[6]) +
             F[2] * (F[3] * F[7] - F[4] * F[6]);
  mjtNum mu = 1000 / (2 * 1.3), lambda = 1000 * .3 / (1.3 * .4);
  return (.9 * 1.1 / 6) * (.5 * mu * (mju_dot(F, F, 9) - 3) - mu * (J - 1) +
                           .5 * (lambda + mu) * (J - 1) * (J - 1));
}

TEST_F(ElasticityTest, SNHForceThroughInversion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  .2 .9 0  -.1 .3 1.1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".3"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  for (bool reverse : {false, true}) {
    SCOPED_TRACE(reverse);
    std::string source = xml;
    if (reverse) {
      source.replace(source.find("element=\"0 1 2 3\""),
                     std::string("element=\"0 1 2 3\"").size(),
                     "element=\"0 2 1 3\"");
    }
    mjSpec* spec = mj_parseXMLString(source.c_str(), nullptr, nullptr, 0);
    ASSERT_THAT(spec, NotNull());
    spec->option.integrator = mjINT_DISCRETE;
    mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "tet"))->elastic3d = 1;
    MjModelPtr m(mj_compile(spec, nullptr));
    mj_deleteSpec(spec);
    ASSERT_THAT(m.get(), NotNull());
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());
    std::vector<mjtNum> rest(d->flexvert_xpos, d->flexvert_xpos + 12);
    // Full rank, reflected, flat, rank one, and rank zero, with a nontrivial
    // rotation.
    for (mjtNum scale :
         {mjtNum(1), mjtNum(.1), mjtNum(0), mjtNum(-.1), mjtNum(-1)}) {
      for (int rank = 1; rank <= 3; rank++) {
        for (int v = 0; v < 4; v++) {
          int adr = m->body_dofadr[m->flex_vertbodyid[v]];
          mjtNum x = rank > 1 ? rest[3 * v] + .3 * rest[3 * v + 1] : 0;
          mjtNum y = rank > 2 ? rest[3 * v + 1] : 0;
          mjtNum z = scale * rest[3 * v + 2];
          d->qpos[adr] = .8 * x - .6 * y - rest[3 * v];
          d->qpos[adr + 1] = .6 * x + .8 * y - rest[3 * v + 1];
          d->qpos[adr + 2] = z - rest[3 * v + 2];
        }
        mj_forward(m.get(), d.get());
        mjtNum s[6], a[3], b[3], c[3], cross[3];
        for (int e = 0; e < 6; e++) {
          int idx = m->flex_elemedge[e];
          s[e] = d->flexedge_length[idx] * d->flexedge_length[idx] -
                 m->flexedge_length0[idx] * m->flexedge_length0[idx];
        }
        const int* vert = m->flex_elem;
        mju_sub3(a, d->flexvert_xpos + 3 * vert[1],
                 d->flexvert_xpos + 3 * vert[0]);
        mju_sub3(b, d->flexvert_xpos + 3 * vert[2],
                 d->flexvert_xpos + 3 * vert[0]);
        mju_sub3(c, d->flexvert_xpos + 3 * vert[3],
                 d->flexvert_xpos + 3 * vert[0]);
        mju_cross(cross, b, c);
        mjtNum J = m->flex_stiffness[23] * mju_dot3(a, cross);
        EXPECT_NEAR(SNHElementEnergy(m->flex_stiffness, s, J),
                    SNHEnergy(d.get()), MjTol(1e-10, 1e-3));
        std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
        mjtNum eps = MjEps(1e-6, 1e-3);
        for (int i = 0; i < m->nv; i++) {
          mjtNum saved = d->qpos[i];
          d->qpos[i] = saved + eps;
          mj_forward(m.get(), d.get());
          mjtNum plus = SNHEnergy(d.get());
          d->qpos[i] = saved - eps;
          mj_forward(m.get(), d.get());
          mjtNum minus = SNHEnergy(d.get());
          d->qpos[i] = saved;
          mjtNum fd = -(plus - minus) / (2 * eps);
          EXPECT_TRUE(std::isfinite(force[i]));
          EXPECT_NEAR(force[i], fd, MjTol(1e-6, .2));
        }
      }
    }
  }
}

// An inverted tetrahedron must recover without internal contacts, for both the
// assembled (Newton) and matrix-free (CG) discrete integration paths.
TEST_F(ElasticityTest, SNHInversionRecovery) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" tolerance="1e-10"/>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".3" damping=".03"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "tet"))->elastic3d = 1;
  MjModelPtr m(mj_compile(spec, nullptr));
  mj_deleteSpec(spec);
  ASSERT_THAT(m.get(), NotNull());
  ASSERT_EQ(m->nv, 3);
  MjDataPtr d = MakeData(m);
  for (int solver : {mjSOL_CG, mjSOL_NEWTON}) {
    m->opt.solver = solver;
    for (mjtNum timestep : {mjtNum(.001), mjtNum(.01), mjtNum(.03)}) {
      m->opt.timestep = timestep;
      mj_resetData(m.get(), d.get());
      // vertex 3 rest z is 1, so displacement qpos[2] = -1.5 places it at
      // z = -0.5, giving J = V/V0 = -0.5 (inverted)
      d->qpos[2] = -1.5;
      for (int i = 0; i < 2000; i++) {
        mj_step(m.get(), d.get());
        ASSERT_TRUE(std::isfinite(d->qpos[2]));
        ASSERT_LT(mju_abs(d->qpos[2]), 3);
      }
      EXPECT_NEAR(d->qpos[2], 0, MjTol(1e-6, 1e-4));
      for (int warning = 0; warning < mjNWARNING; warning++) {
        EXPECT_EQ(d->warning[warning].number, 0);
      }
      EXPECT_EQ(d->ncon, 0);
    }
  }
}

TEST_F(PassiveTest, PolynomialStiffnessJoint) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" stiffness="2 3 4"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qpos="0.5"/>
    </keyframe>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  mj_forward(m.get(), d.get());

  mjtNum x = 0.5;
  mjtNum a = 2, b = 3, c = 4;
  mjtNum expected = -(a + b * x + c * x * x) * x;
  EXPECT_NEAR(d->qfrc_spring[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolynomialStiffnessNegativeDisplacement) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" stiffness="2 3 4"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qpos="-0.5"/>
    </keyframe>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  mj_forward(m.get(), d.get());

  mjtNum x = -0.5;
  mjtNum a = 2, b = 3, c = 4;
  mjtNum expected = -(a + b * x + c * x * x) * x;
  EXPECT_NEAR(d->qfrc_spring[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolyStiffnessFixedTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" name="j"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <tendon>
      <fixed stiffness="10 5 1">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());

  mjtNum x = d->ten_length[0] - m->tendon_lengthspring[1];
  mjtNum expected = -(10 + 5 * x + 1 * x * x) * x;
  EXPECT_NEAR(d->qfrc_spring[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolyStiffnessSpatialTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="s0"/>
      <body>
        <joint type="slide" name="j"/>
        <geom size="1" mass="1"/>
        <site name="s1"/>
      </body>
    </worldbody>

    <tendon>
      <spatial stiffness="10 5 1">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());

  mjtNum x = d->ten_length[0] - m->tendon_lengthspring[1];
  mjtNum expected = -x * (10 + 5 * x + 1 * x * x);
  EXPECT_NEAR(d->qfrc_spring[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolynomialDampingJoint) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" damping="2 3 4"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="0.5"/>
    </keyframe>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  mj_forward(m.get(), d.get());

  mjtNum v = 0.5;
  mjtNum a = 2, b = 3, c = 4;
  mjtNum expected = -(a * v + b * v * mju_abs(v) + c * v * v * v);
  EXPECT_NEAR(d->qfrc_damper[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolynomialDampingNegativeVelocity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" damping="2 3 4"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="-0.5"/>
    </keyframe>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  mj_forward(m.get(), d.get());

  mjtNum v = -0.5;
  mjtNum a = 2, b = 3, c = 4;
  mjtNum expected = -(a * v + b * v * mju_abs(v) + c * v * v * v);
  EXPECT_NEAR(d->qfrc_damper[0], expected, 1e-12);
}

TEST_F(PassiveTest, PolynomialDampingTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" name="j"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <tendon>
      <fixed damping="10 5 1">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>

    <keyframe>
      <key qvel="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());

  mjtNum v = d->ten_velocity[0];
  mjtNum expected = -(10 * v + 5 * v * mju_abs(v) + 1 * v * v * v);
  EXPECT_NEAR(d->qfrc_damper[0], expected, 1e-12);
}

// shell-mode (elastic2d=stretch) flexcomp must have zero passive spring forces
// at rest (initial configuration); any nonzero force indicates a rotation
// mismatch between compile-time reference positions and runtime corotation.
TEST_F(ElasticityTest, ShellModeZeroForceAtRest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="8 8 8" spacing=".07 .07 .07" pos="0 0 1"
                dim="3" cellcount="1 1 1" radius=".001" rgba="0 .7 .7 1"
                mass="5" name="softbody" dof="trilinear">
        <elasticity young="1e4" poisson="0.1" damping="0.01"
                    elastic2d="stretch" thickness="0.02"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_forward(m.get(), d.get());

  // all spring forces should be zero at rest
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->qfrc_spring[i], 0, 1e-10)
        << "nonzero spring force at DOF " << i;
  }
}

// flex stretch damping must exert exactly no force at zero velocity
TEST_F(ElasticityTest, StretchDampingZeroVelocity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp type="grid" count="3 3 3" spacing=".1 .1 .1" radius=".01" dim="3" mass="1" name="solid">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="10"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // deform at zero velocity
  for (int i = 0; i < m->nv; i++) {
    d->qpos[i] += 1e-3 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> damped(d->qfrc_passive, d->qfrc_passive + m->nv);
  EXPECT_GT(mju_norm(damped.data(), m->nv), 1)
      << "test should exercise a nontrivial stretch force";

  // the same state without damping must give bitwise the same force
  m->flex_damping[0] = 0;
  mj_forward(m.get(), d.get());
  for (int i = 0; i < m->nv; i++) {
    EXPECT_EQ(d->qfrc_passive[i], damped[i]) << "damping force at DOF " << i;
  }
}

// the spring and damper flags each disable their part of the flex stretch
// force, and its Rayleigh damping is booked in qfrc_damper. The solid's
// vertices are simple bodies; the cloth's, under a moving parent and with a
// pinned vertex, are not.
TEST_F(ElasticityTest, StretchDisableFlags) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="3 3 3" spacing=".1 .1 .1" radius=".01" dim="3" mass="1" name="solid">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="10"/>
      </flexcomp>
      <body name="parent" pos="1 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom size=".01" mass="1" contype="0" conaffinity="0"/>
        <flexcomp type="grid" count="3 3 1" spacing=".1 .1 .1" radius=".01" dim="2" mass="1" name="cloth">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.3" thickness="1e-2" elastic2d="stretch" damping="10"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  int nv = m->nv;

  // deform, with nonzero velocity
  for (int i = 0; i < nv; i++) {
    d->qpos[i] += 1e-3 * (mju_Halton(i, 2) - 0.5);
    d->qvel[i] = 1e-3 * (mju_Halton(i, 3) - 0.5);
  }

  // stretch spring force: the same state without damping
  std::vector<mjtNum> damping(m->flex_damping, m->flex_damping + m->nflex);
  for (int f = 0; f < m->nflex; f++) {
    m->flex_damping[f] = 0;
  }
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> spring = AsVector(d->qfrc_spring, nv);
  EXPECT_GT(mju_norm(spring.data(), nv), 1)
      << "test should exercise a nontrivial stretch force";

  // both enabled: the damping is booked in qfrc_damper
  for (int f = 0; f < m->nflex; f++) {
    m->flex_damping[f] = damping[f];
  }
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> damper = AsVector(d->qfrc_damper, nv);
  EXPECT_GT(mju_norm(damper.data(), nv), 1)
      << "test should exercise a nontrivial damping force";
  EXPECT_EQ(AsVector(d->qfrc_spring, nv), spring);

  // spring disabled: only the damping remains
  std::vector<mjtNum> zero(nv, 0);
  m->opt.disableflags = mjDSBL_SPRING;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(AsVector(d->qfrc_spring, nv), zero);
  EXPECT_EQ(AsVector(d->qfrc_damper, nv), damper);

  // damper disabled: only the spring remains
  m->opt.disableflags = mjDSBL_DAMPER;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(AsVector(d->qfrc_spring, nv), spring);
  EXPECT_EQ(AsVector(d->qfrc_damper, nv), zero);
}
// interpolated shell bending must produce zero spring forces at rest
TEST_F(ElasticityTest, InterpBendingZeroForceAtRest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="8 8 8" spacing=".07 .07 .07" pos="0 0 1"
                dim="3" cellcount="2 2 1" radius=".001" rgba="0 .7 .7 1"
                mass="5" name="softbody" dof="trilinear">
        <elasticity young="1e4" poisson="0.1" damping="0"
                    elastic2d="bend" thickness="0.02"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // verify bending data was compiled
  const mjtNum* bdata = m->flex_bending + m->flex_bendingadr[0];
  int nedge = (int)bdata[0];
  EXPECT_GT(nedge, 0) << "no bending edges compiled";

  mj_forward(m.get(), d.get());

  // all spring forces should be zero at rest
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->qfrc_spring[i], 0, 1e-10)
        << "nonzero spring force at DOF " << i;
  }

  // verify per-edge bending data
  int n_flat = 0, n_corner = 0;
  for (int e = 0; e < nedge; e++) {
    const mjtNum* edata = bdata + 1 + e * 10;
    mjtNum stiffness = edata[6];
    mjtNum dn0[3] = {edata[7], edata[8], edata[9]};
    mjtNum dn0_norm = mju_norm3(dn0);

    // stiffness must be positive
    EXPECT_GT(stiffness, 0) << "edge " << e << " has non-positive stiffness";

    if (dn0_norm < 1e-10) {
      // intra-surface edge: coplanar faces, zero normal jump
      n_flat++;
    } else {
      // corner edge: 90° between perpendicular face normals, |dn0| = sqrt(2)
      n_corner++;
      EXPECT_NEAR(dn0_norm, mju_sqrt(2.0), 1e-10)
          << "corner edge " << e << " has unexpected |dn0|=" << dn0_norm;
    }
  }

  // for a 2x2x1 box: 12 intra-surface + 20 corner = 32 edges
  EXPECT_GT(n_flat, 0) << "no intra-surface edges found";
  EXPECT_GT(n_corner, 0) << "no corner edges found";
  EXPECT_EQ(n_flat + n_corner, nedge);
}

// interpolated shell bending must produce zero forces after a rigid rotation
TEST_F(ElasticityTest, InterpBendingRigidRotationInvariance) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="8 2 12" spacing=".025 .05 .025" pos="0 0 1"
                dim="3" cellcount="6 1 6" radius=".001" rgba="0 .7 .7 1"
                mass="5" name="softbody" dof="trilinear">
        <elasticity young="1e5" poisson="0.3" damping="0"
                    elastic2d="bend" thickness="0.03"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // compute geometric center from body positions (skip world body)
  mjtNum center[3] = {0, 0, 0};
  int nnodes = 0;
  for (int b = 1; b < m->nbody; b++) {
    center[0] += m->body_pos[3 * b + 0];
    center[1] += m->body_pos[3 * b + 1];
    center[2] += m->body_pos[3 * b + 2];
    nnodes++;
  }
  ASSERT_GT(nnodes, 0);
  center[0] /= nnodes;
  center[1] /= nnodes;
  center[2] /= nnodes;

  // rotation: 45 degrees about (1,1,1)/sqrt(3)
  mjtNum angle = 45 * 3.14159265358979 / 180.0;
  mjtNum sa = mju_sin(angle / 2), ca = mju_cos(angle / 2);
  mjtNum inv_sqrt3 = 1.0 / mju_sqrt(3.0);
  mjtNum quat[4] = {ca, sa * inv_sqrt3, sa * inv_sqrt3, sa * inv_sqrt3};
  mjtNum neg_quat[4];
  mju_negQuat(neg_quat, quat);

  // apply rigid rotation via slide joint displacements:
  // new_pos = center + R * (body_pos - center)
  // qpos = new_pos - body_pos
  for (int b = 1; b < m->nbody; b++) {
    mjtNum rel[3] = {m->body_pos[3 * b + 0] - center[0],
                     m->body_pos[3 * b + 1] - center[1],
                     m->body_pos[3 * b + 2] - center[2]};
    mjtNum rotated[3];
    mju_rotVecQuat(rotated, rel, neg_quat);

    // each body has 3 slide joints (x, y, z)
    for (int j = 0; j < m->body_jntnum[b] && j < 3; j++) {
      int jid = m->body_jntadr[b] + j;
      int qadr = m->jnt_qposadr[jid];
      int axis = -1;
      for (int a = 0; a < 3; a++) {
        if (m->jnt_axis[3 * jid + a] != 0) {
          axis = a;
          break;
        }
      }
      if (axis >= 0) {
        d->qpos[qadr] =
            (center[axis] + rotated[axis]) - m->body_pos[3 * b + axis];
      }
    }
  }

  mj_forward(m.get(), d.get());

  // spring forces should still be zero after rigid rotation
  const mjtNum tol = MjTol(1e-6, 1e-3);
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->qfrc_spring[i], 0, tol)
        << "nonzero spring force at DOF " << i << " after rigid rotation";
  }
}

// a trilinear flex whose first cage cell is empty (no mesh content, all-zero
// stiffness block) must still get elastic forces from its non-empty cells:
// the flex used to be gated on the stiffness of cell 0 only
TEST_F(ElasticityTest, TrilinearEmptyFirstCell) {
  // a cube of 5 tetrahedra in x=[.7, 1] plus an unattached vertex at x=0 that
  // extends the cage to x=[0, 1]; of the 3x1x1 cells only the last is occupied
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="INTEGRATOR" timestep="1e-3"
            solver="CG"/>
    <worldbody>
      <flexcomp name="test" type="direct" dim="3" dof="trilinear" mass="1"
                cellcount="3 1 1" radius=".001"
                point=".7 0 0  1 0 0  .7 1 0  1 1 0
                       .7 0 1  1 0 1  .7 1 1  1 1 1
                       0 .5 .5"
                element="0 1 2 4  3 2 1 7  5 1 4 7  6 2 4 7  1 2 4 7">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="100" poisson="0.3" damping="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  for (const char* integrator : {"Euler", "discrete"}) {
    std::string model_xml = xml;
    model_xml.replace(model_xml.find("INTEGRATOR"), 10, integrator);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(model_xml.c_str(), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);

    // cell 0 is empty, cell 2 has stiffness
    ASSERT_EQ(m->flex_interp[0], 1);
    ASSERT_EQ(m->flex_edgeequality[0], 0);
    int block = 24 * 24;
    const mjtNum* k = m->flex_stiffness + m->flex_stiffnessadr[0];
    ASSERT_EQ(k[0], 0) << "test requires an empty first cell";
    ASSERT_NE(k[2 * block], 0) << "test requires a non-empty last cell";

    // free node bodies (nodes exclusively in empty cells are pinned)
    std::vector<int> free_bodies;
    for (int n = 0; n < m->flex_nodenum[0]; n++) {
      int b = m->flex_nodebodyid[m->flex_nodeadr[0] + n];
      if (m->body_dofnum[b] == 3) {
        free_bodies.push_back(b);
      }
    }
    ASSERT_EQ(free_bodies.size(), 8u);

    // deformation: summed change of the distances between free nodes
    mj_kinematics(m.get(), d.get());
    std::vector<mjtNum> xpos0 = AsVector(d->xpos, 3 * m->nbody);
    auto deformation = [&]() {
      mjtNum total = 0;
      for (int i : free_bodies) {
        for (int j : free_bodies) {
          if (i < j) {
            mjtNum dist = mju_dist3(d->xpos + 3 * i, d->xpos + 3 * j);
            mjtNum dist0 =
                mju_dist3(xpos0.data() + 3 * i, xpos0.data() + 3 * j);
            total += mju_abs(dist - dist0);
          }
        }
      }
      return total;
    };

    // displace one free node: the non-empty cell must push back
    int dof = m->body_dofadr[free_bodies[0]];
    d->qpos[m->jnt_qposadr[m->dof_jntid[dof]]] += 0.02;
    mj_forward(m.get(), d.get());
    EXPECT_GT(mju_norm(d->qfrc_spring, m->nv), 0)
        << integrator << ": no elastic force with an empty first cell";
    mjtNum initial = deformation();
    ASSERT_GT(initial, 0);

    // simulate: the deformation must decay and the state stay finite
    for (int i = 0; i < 2000; i++) {
      mj_step(m.get(), d.get());
    }
    for (int i = 0; i < m->nq; i++) {
      ASSERT_TRUE(std::isfinite(d->qpos[i])) << integrator << ": qpos " << i;
    }
    mj_kinematics(m.get(), d.get());
    EXPECT_LT(deformation(), 0.5 * initial)
        << integrator << ": deformation did not decay";
  }
}

// shell-mode interpolated flexes apply bending for elastic2d="bend", membrane
// stretch on all 6 faces (including the x=0 and x=max faces whose first
// stiffness entry is zero) for elastic2d="stretch", and their sum for "both"
TEST_F(ElasticityTest, TrilinearShellElastic2d) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp type="grid" count="4 4 4" spacing=".025 .05 .025" pos="0 0 1"
                dim="3" cellcount="1 2 2" radius=".001" mass="5"
                name="softbody" dof="trilinear">
        <elasticity young="1e4" poisson="0.3"
                    elastic2d="ELASTIC2D" thickness="0.03"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  auto load_and_deform = [&](const char* elastic2d, MjModelPtr* m_out) {
    std::string model_xml = xml;
    model_xml.replace(model_xml.find("ELASTIC2D"), 9, elastic2d);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(model_xml.c_str(), error, sizeof(error));
    EXPECT_THAT(m.get(), NotNull()) << error;
    if (!m) return std::vector<mjtNum>();
    EXPECT_LT(m->flex_interp[0], 0) << "test requires a shell-mode flex";
    MjDataPtr d = MakeData(m);
    for (int i = 0; i < m->nq; i++) {
      d->qpos[i] = m->qpos0[i] + 0.01 * mju_sin(1.7 * i + 0.3);
    }
    mj_forward(m.get(), d.get());
    std::vector<mjtNum> frc = AsVector(d->qfrc_spring, m->nv);
    if (m_out) *m_out = std::move(m);
    return frc;
  };

  MjModelPtr m_bend, m_stretch, m_both;
  std::vector<mjtNum> bend = load_and_deform("bend", &m_bend);
  std::vector<mjtNum> stretch = load_and_deform("stretch", &m_stretch);
  std::vector<mjtNum> both = load_and_deform("both", &m_both);
  ASSERT_FALSE(bend.empty());
  ASSERT_EQ(bend.size(), stretch.size());
  ASSERT_EQ(bend.size(), both.size());

  // bend-only does not allocate membrane stiffness; stretch and both do
  EXPECT_EQ(m_bend->flex_stiffnessadr[0], -1);
  EXPECT_GE(m_stretch->flex_stiffnessadr[0], 0);
  EXPECT_GE(m_both->flex_stiffnessadr[0], 0);

  mjtNum bend_norm = 0, stretch_norm = 0;
  for (int i = 0; i < static_cast<int>(bend.size()); i++) {
    bend_norm += bend[i] * bend[i];
    stretch_norm += stretch[i] * stretch[i];
    EXPECT_NEAR(both[i], bend[i] + stretch[i], MjTol(1e-10, 1e-4))
        << "dof " << i;
  }
  EXPECT_GT(bend_norm, 0);
  EXPECT_GT(stretch_norm, 0);

  // With cellcount="1 2 2" (node grid 2 x 3 x 3), node (0, 1, 1) at index 4
  // and node (1, 1, 1) at index 13 lie strictly in the interior of the x=0
  // and x=max faces (faces 0 and 1, where normal_axis=0 and k_elem[0]==0) and
  // do not touch any y- or z-normal faces. Verify both receive nonzero
  // membrane stretch forces.
  for (int node_idx : {4, 13}) {
    int bid = m_stretch->flex_nodebodyid[m_stretch->flex_nodeadr[0] + node_idx];
    int dadr = m_stretch->body_dofadr[bid];
    mjtNum node_frc = mju_norm3(stretch.data() + dadr);
    EXPECT_GT(node_frc, 1e-2)
        << "expected nonzero stretch force on x-face interior node "
        << node_idx;
  }
}

// verify that a pinned vertex (on a static body) gets zero bending force
// while its free neighbors get nonzero bending force
TEST_F(ElasticityTest, PinnedVertexBendingForce) {
  static constexpr char xml[] = R"(
  <mujoco>
  <worldbody>
    <body name="parent">
      <flexcomp name="test" type="grid" count="3 3 1" spacing="1 1 1"
                radius="0.01" dim="2">
        <elasticity young="1" poisson="0" thickness="1" elastic2d="bend"/>
        <pin id="0"/>
      </flexcomp>
    </body>
  </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // displace a free vertex out of plane to create bending
  d->qpos[3] = 0.1;
  mj_forward(m.get(), d.get());

  // vertex 0 is pinned to the world body (body 0), which has 0 dofs,
  // so no bending force is written for it — the reaction is absorbed by the pin

  // verify that at least some free vertices have nonzero spring force
  bool has_nonzero = false;
  for (int i = 0; i < m->nv; i++) {
    if (d->qfrc_spring[i] != 0) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "bending should produce nonzero spring forces";
}

// verify that a flexcomp with dof="trilinear" inside a parent body
// with non-identity quaternion produces:
//   1. restoring forces (not expanding) for small perturbations
//   2. identical qfrc_passive between rotated and non-rotated models
//   3. stable simulation with implicit integration (no NaN, bounded qacc)
TEST_F(ElasticityTest, TrilinearParentBodyRotation) {
  static constexpr char rotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.0005" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0" quat="0.7071 0 0.7071 0">
        <flexcomp type="grid" count="3 3 3" spacing=".05 .05 .05"
                  dim="3" radius=".001" mass=".005" name="soft" dof="trilinear">
          <elasticity young="1e4" poisson="0.1" damping="0.1"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char nonrotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.0005" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0">
        <flexcomp type="grid" count="3 3 3" spacing=".05 .05 .05"
                  dim="3" radius=".001" mass=".005" name="soft" dof="trilinear">
          <elasticity young="1e4" poisson="0.1" damping="0.1"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};

  // --- test 1: force sign and rotational invariance ---
  MjModelPtr m_rot = LoadModelFromString(rotated_xml, error, sizeof(error));
  ASSERT_THAT(m_rot.get(), NotNull()) << error;
  MjDataPtr d_rot = MakeData(m_rot);

  MjModelPtr m_non = LoadModelFromString(nonrotated_xml, error, sizeof(error));
  ASSERT_THAT(m_non.get(), NotNull()) << error;
  MjDataPtr d_non = MakeData(m_non);

  // apply small +X perturbation to DOF 0 (node 0, X slide)
  d_rot->qpos[0] = 1e-6;
  d_non->qpos[0] = 1e-6;

  mj_forward(m_rot.get(), d_rot.get());
  mj_forward(m_non.get(), d_non.get());

  // check force sign on the displaced DOF: positive qpos -> negative qfrc
  // (restoring)
  EXPECT_LT(d_rot->qfrc_passive[0], 0)
      << "rotated model: expected restoring force on displaced DOF 0";
  EXPECT_LT(d_non->qfrc_passive[0], 0)
      << "non-rotated model: expected restoring force on displaced DOF 0";

  // check rotational invariance: forces match within numerical precision
  // (tol ~ 1e-13 due to floating-point differences in polar decomposition path)
  const mjtNum tol = MjTol(1e-12, 1e-5);
  for (int i = 0; i < m_rot->nv; i++) {
    EXPECT_NEAR(d_rot->qfrc_passive[i], d_non->qfrc_passive[i], tol)
        << "rotated/non-rotated qfrc mismatch at DOF " << i;
  }

  // --- test 2: implicit integration stability (200 steps, no NaN) ---
  mjtNum max_qacc = 0;
  for (int step = 0; step < 200; step++) {
    mj_step(m_rot.get(), d_rot.get());

    for (int i = 0; i < m_rot->nv; i++) {
      ASSERT_TRUE(std::isfinite(d_rot->qacc[i]))
          << "NaN/Inf in qacc at DOF " << i << " at step " << step;
      mjtNum a = mju_abs(d_rot->qacc[i]);
      if (a > max_qacc) max_qacc = a;
    }
    if (HasFatalFailure()) return;
  }

  EXPECT_LT(max_qacc, 1e4);
}

// A dim=2 flexcomp with bending elasticity inside a parent body with a
// non-identity quaternion. The implicit metric assembles the bending
// stiffness from world-space vertex positions, but the vertex bodies' slide
// dofs live in the (rotated) parent frame. Without the R^T (.) R change
// of basis the metric stops being the Jacobian of the passive force,
// which shows up as a loss of rotational invariance and, at stiffnesses
// the unrotated model handles comfortably, as divergence.
TEST_F(ElasticityTest, BendParentBodyRotation) {
  static constexpr char rotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.001" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0" quat="0.7071 0.7071 0 0">
        <flexcomp type="grid" count="5 5 1" spacing=".05 .05 .05"
                  dim="2" radius=".001" mass=".01" name="sheet">
          <elasticity young="1e5" poisson="0" thickness="1e-2"
                      elastic2d="bend"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char nonrotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.001" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0">
        <flexcomp type="grid" count="5 5 1" spacing=".05 .05 .05"
                  dim="2" radius=".001" mass=".01" name="sheet">
          <elasticity young="1e5" poisson="0" thickness="1e-2"
                      elastic2d="bend"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};

  MjModelPtr m_rot = LoadModelFromString(rotated_xml, error, sizeof(error));
  ASSERT_THAT(m_rot.get(), NotNull()) << error;
  MjDataPtr d_rot = MakeData(m_rot);

  MjModelPtr m_non = LoadModelFromString(nonrotated_xml, error, sizeof(error));
  ASSERT_THAT(m_non.get(), NotNull()) << error;
  MjDataPtr d_non = MakeData(m_non);

  // lift one corner out of plane: that bends at first order but stretches only
  // at second, so the bending force is not swamped. As in the stretch case the
  // displacement is applied in the parent frame, so the response must not
  // depend on the parent's orientation
  d_rot->qpos[2] = 1e-2;
  d_non->qpos[2] = 1e-2;

  mj_forward(m_rot.get(), d_rot.get());
  mj_forward(m_non.get(), d_non.get());

  EXPECT_LT(d_non->qfrc_passive[2], -1e-6)
      << "expected a restoring force on the lifted dof";

  const mjtNum tol = MjTol(1e-12, 1e-5);
  for (int i = 0; i < m_rot->nv; i++) {
    EXPECT_NEAR(d_rot->qfrc_passive[i], d_non->qfrc_passive[i], tol)
        << "rotated/non-rotated qfrc mismatch at dof " << i;
  }

  // qacc exercises the metric itself, so this also covers mjd_flexBend_mul
  for (int i = 0; i < m_rot->nv; i++) {
    EXPECT_NEAR(d_rot->qacc[i], d_non->qacc[i], MjTol(1e-9, 1e-3))
        << "rotated/non-rotated qacc mismatch at dof " << i;
  }

  // and the rotated model must integrate stably
  for (int step = 0; step < 200; step++) {
    mj_step(m_rot.get(), d_rot.get());
    for (int i = 0; i < m_rot->nv; i++) {
      ASSERT_TRUE(std::isfinite(d_rot->qacc[i]))
          << "NaN/Inf in qacc at dof " << i << " at step " << step;
    }
    if (HasFatalFailure()) return;
  }
}

// A dim=2 flexcomp with stretch elasticity inside a parent body with a
// non-identity quaternion. The implicit metric assembles the stretch
// stiffness from world-space edge vectors, but the vertex bodies' slide
// dofs live in the (rotated) parent frame. Without the R^T (.) R change
// of basis the metric stops being the Jacobian of the passive force,
// which shows up as a loss of rotational invariance and, at stiffnesses
// the unrotated model handles comfortably, as divergence.
TEST_F(ElasticityTest, StretchParentBodyRotation) {
  static constexpr char rotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.001" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0" quat="0.7071 0.7071 0 0">
        <flexcomp type="grid" count="5 5 1" spacing=".05 .05 .05"
                  dim="2" radius=".001" mass=".01" name="sheet">
          <elasticity young="1e5" poisson="0" thickness="1e-3"
                      elastic2d="stretch"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char nonrotated_xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"
            timestep="0.001" solver="CG"/>
    <worldbody>
      <body name="base" pos="0 0 0">
        <flexcomp type="grid" count="5 5 1" spacing=".05 .05 .05"
                  dim="2" radius=".001" mass=".01" name="sheet">
          <elasticity young="1e5" poisson="0" thickness="1e-3"
                      elastic2d="stretch"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};

  MjModelPtr m_rot = LoadModelFromString(rotated_xml, error, sizeof(error));
  ASSERT_THAT(m_rot.get(), NotNull()) << error;
  MjDataPtr d_rot = MakeData(m_rot);

  MjModelPtr m_non = LoadModelFromString(nonrotated_xml, error, sizeof(error));
  ASSERT_THAT(m_non.get(), NotNull()) << error;
  MjDataPtr d_non = MakeData(m_non);

  // stretch one dof; the response is expressed in the parent frame in
  // both models, so the passive force and the implicit acceleration must
  // agree regardless of the parent's orientation
  d_rot->qpos[0] = 1e-4;
  d_non->qpos[0] = 1e-4;

  mj_forward(m_rot.get(), d_rot.get());
  mj_forward(m_non.get(), d_non.get());

  EXPECT_LT(d_rot->qfrc_passive[0], 0)
      << "expected a restoring force on the stretched dof";

  const mjtNum tol = MjTol(1e-12, 1e-5);
  for (int i = 0; i < m_rot->nv; i++) {
    EXPECT_NEAR(d_rot->qfrc_passive[i], d_non->qfrc_passive[i], tol)
        << "rotated/non-rotated qfrc mismatch at dof " << i;
  }

  // qacc exercises the metric itself (the force is only its right-hand
  // side): a metric in the wrong basis breaks this invariance even though
  // the force above is already correct
  for (int i = 0; i < m_rot->nv; i++) {
    EXPECT_NEAR(d_rot->qacc[i], d_non->qacc[i], MjTol(1e-9, 1e-3))
        << "rotated/non-rotated qacc mismatch at dof " << i;
  }

  // and the rotated model must integrate stably
  for (int step = 0; step < 200; step++) {
    mj_step(m_rot.get(), d_rot.get());
    for (int i = 0; i < m_rot->nv; i++) {
      ASSERT_TRUE(std::isfinite(d_rot->qacc[i]))
          << "NaN/Inf in qacc at dof " << i << " at step " << step;
    }
    if (HasFatalFailure()) return;
  }
}

// ------------------------------ 1D rod bending ------------------------------

constexpr int kRodHingeSize = 10;

// build orthonormal frame given xaxis in frame[0..2] (matches mju_makeFrame)
void MakeFrame(mjtNum frame[9]) {
  mju_normalize3(frame);
  if (mju_dot3(frame + 3, frame + 3) < 0.25) {
    mju_zero3(frame + 3);
    if (frame[1] < 0.5 && frame[1] > -0.5) {
      frame[4] = 1;
    } else {
      frame[5] = 1;
    }
  }
  mju_addToScl3(frame + 3, frame, -mju_dot3(frame, frame + 3));
  mju_normalize3(frame + 3);
  mju_cross(frame + 6, frame, frame + 3);
}

// chain order and per-edge reference (t, u, v) and twist angle of a 1D DER rod
struct RodChainFrames {
  std::vector<int> v_chain;     // flex-local vertices in chain order
  std::vector<mjtNum> t, u, v;  // 3 per chain edge
  std::vector<mjtNum> theta;    // 1 per chain edge
};

// discrete elastic rod bending energy of flex f, from the compiled hinge data;
// frames (optional) receives the reference frames of a DER chain rod
mjtNum RodBendingEnergy(const mjModel* m, const mjData* d, int f,
                        RodChainFrames* frames = nullptr) {
  static constexpr mjtNum kSmooth = static_cast<mjtNum>(1e-4);
  const mjtNum gam = mju_sqrt(1 + kSmooth * kSmooth);
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[f];
  const mjtNum* xpos = d->flexvert_xpos + 3 * m->flex_vertadr[f];
  auto binormal = [&](int va, int v, int vb, mjtNum kb[3]) {
    mjtNum e0[3], e1[3], cross[3];
    mju_sub3(e0, xpos + 3 * v, xpos + 3 * va);
    mju_sub3(e1, xpos + 3 * vb, xpos + 3 * v);
    mju_cross(cross, e0, e1);
    mjtNum denom = mju_norm3(e0) * mju_norm3(e1) + mju_dot3(e0, e1);
    mju_scl3(kb, cross, 2 / denom);
  };
  int nhinge = (int)b[0];
  bool is_chain = nhinge > 0;
  bool any_aniso = false;
  bool any_twist_jnt = false;
  for (int h = 0; h < nhinge && is_chain; h++) {
    const mjtNum* hinge = b + 1 + kRodHingeSize * h;
    if (hinge[3] >= 0) is_chain = false;
    if (h > 0) {
      const mjtNum* prev = b + 1 + kRodHingeSize * (h - 1);
      if ((int)hinge[0] != (int)prev[1] || (int)hinge[1] != (int)prev[2]) {
        is_chain = false;
      }
    }
    if (hinge[5] != 0 || hinge[9] != 0) any_aniso = true;
    if (hinge[7] >= 0) any_twist_jnt = true;
  }
  if (is_chain) {
    const mjtNum* last_hinge = b + 1 + kRodHingeSize * (nhinge - 1);
    if ((int)last_hinge[1] == (int)b[1 + 0] ||
        (int)last_hinge[2] == (int)b[1 + 0]) {
      is_chain = false;
    }
  }
  int va = m->flex_vertadr[f];
  bool root_clamped = (m->flex_vertbodyid[va + (int)b[1 + 0]] ==
                       m->flex_vertbodyid[va + (int)b[1 + 1]]);
  if (is_chain && (any_aniso || any_twist_jnt)) {
    int nedge = nhinge + 1;
    int nvert_chain = nhinge + 2;
    std::vector<int> v_chain(nvert_chain);
    for (int h = 0; h < nhinge; h++) {
      v_chain[h] = (int)b[1 + kRodHingeSize * h + 0];
    }
    v_chain[nhinge] = (int)b[1 + kRodHingeSize * (nhinge - 1) + 1];
    v_chain[nhinge + 1] = (int)b[1 + kRodHingeSize * (nhinge - 1) + 2];

    auto body_pose0 = [&](int b_id, mjtNum pos[3], mjtNum quat[4]) {
      mju_zero3(pos);
      quat[0] = 1;
      quat[1] = quat[2] = quat[3] = 0;
      for (int cur = b_id; cur > 0; cur = m->body_parentid[cur]) {
        mjtNum p[3], q[4];
        mju_rotVecQuat(p, pos, m->body_quat + 4 * cur);
        mju_add3(pos, p, m->body_pos + 3 * cur);
        mju_mulQuat(q, m->body_quat + 4 * cur, quat);
        mju_copy4(quat, q);
      }
    };
    auto pt = [](const mjtNum t_from[3], const mjtNum t_to[3],
                 const mjtNum u_in[3], mjtNum u_out[3], mjtNum v_out[3]) {
      mjtNum denom = 1 + mju_dot3(t_from, t_to);
      mjtNum sum[3];
      mju_add3(sum, t_from, t_to);
      mjtNum scale = mju_dot3(t_to, u_in) / denom;
      mju_addScl3(u_out, u_in, sum, -scale);
      mju_normalize3(u_out);
      mju_cross(v_out, t_to, u_out);
    };

    std::vector<mjtNum> xpos0(3 * nvert_chain);
    int b0 = m->flex_vertbodyid[va + v_chain[0]];
    mjtNum p0_b0[3], q0_b0[4], R0_b0[9];
    body_pose0(b0, p0_b0, q0_b0);
    mju_quat2Mat(R0_b0, q0_b0);
    for (int k = 0; k < nvert_chain; k++) {
      int gv = va + v_chain[k];
      int bk = m->flex_vertbodyid[gv];
      mjtNum pk[3], qk[4], off[3];
      body_pose0(bk, pk, qk);
      mju_rotVecQuat(off, m->flex_vert + 3 * gv, qk);
      mju_add3(xpos0.data() + 3 * k, pk, off);
    }

    mjtNum e0_ref[3], frame_loc[9] = {0}, t0_ref[3], u0_ref[3], v0_ref[3];
    mju_sub3(e0_ref, xpos0.data() + 3, xpos0.data());
    mju_mulMatTVec3(frame_loc, R0_b0, e0_ref);
    mju_normalize3(frame_loc);
    MakeFrame(frame_loc);
    mjtNum psi0 = (b[1 + 9] < 0) ? 0.5 * mjPI : 0;
    mjtNum c0 = mju_cos(psi0), s0 = mju_sin(psi0);
    mjtNum u_loc[3], v_loc[3];
    for (int c = 0; c < 3; c++) {
      u_loc[c] = c0 * frame_loc[3 + c] + s0 * frame_loc[6 + c];
      v_loc[c] = -s0 * frame_loc[3 + c] + c0 * frame_loc[6 + c];
    }
    mju_mulMatVec3(t0_ref, d->xmat + 9 * b0, frame_loc);
    mju_mulMatVec3(u0_ref, d->xmat + 9 * b0, u_loc);
    mju_mulMatVec3(v0_ref, d->xmat + 9 * b0, v_loc);

    std::vector<mjtNum> om0(2 * nhinge, 0);
    std::vector<mjtNum> t_ref_world(3 * nedge, 0), u_ref_world(3 * nedge, 0);
    mjtNum v_prev_ref[3];
    mju_mulMatVec3(t_ref_world.data(), R0_b0, frame_loc);
    mju_mulMatVec3(u_ref_world.data(), R0_b0, u_loc);
    mju_mulMatVec3(v_prev_ref, R0_b0, v_loc);
    for (int h = 0; h < nhinge; h++) {
      if (b[1 + kRodHingeSize * h + 6] > 0) {
        mjtNum e0r[3], e1r[3], cr[3], kb_ref[3];
        mju_sub3(e0r, xpos0.data() + 3 * (h + 1), xpos0.data() + 3 * h);
        mju_sub3(e1r, xpos0.data() + 3 * (h + 2), xpos0.data() + 3 * (h + 1));
        mju_cross(cr, e0r, e1r);
        mjtNum den = mju_norm3(e0r) * mju_norm3(e1r) + mju_dot3(e0r, e1r);
        mju_scl3(kb_ref, cr, 2 / den);
        om0[2 * h + 0] = mju_dot3(kb_ref, u_ref_world.data() + 3 * h);
        om0[2 * h + 1] = mju_dot3(kb_ref, v_prev_ref);
      }
      mjtNum* t_next_ref = t_ref_world.data() + 3 * (h + 1);
      mju_sub3(t_next_ref, xpos0.data() + 3 * (h + 2),
               xpos0.data() + 3 * (h + 1));
      mju_normalize3(t_next_ref);
      pt(t_ref_world.data() + 3 * h, t_next_ref, u_ref_world.data() + 3 * h,
         u_ref_world.data() + 3 * (h + 1), v_prev_ref);
    }

    std::vector<mjtNum> t_edge(3 * nedge), u_edge(3 * nedge), v_edge(3 * nedge);
    for (int e = 0; e < nedge; e++) {
      mju_sub3(t_edge.data() + 3 * e, xpos + 3 * v_chain[e + 1],
               xpos + 3 * v_chain[e]);
      mju_normalize3(t_edge.data() + 3 * e);
    }

    const mjtNum* state = (m->flex_plasticadr[f] >= 0 && d->plastic)
                              ? (d->plastic + m->flex_plasticadr[f])
                              : nullptr;
    // clamped tail (at least two hinges): the last edge's qpos0 reference frame
    // rotates with its body
    int bT = m->flex_vertbodyid[va + v_chain[nhinge + 1]];
    bool tail_clamped =
        nhinge >= 2 && m->flex_vertbodyid[va + v_chain[nhinge]] == bT;
    mjtNum tT_ref[3] = {0}, uT_ref[3] = {0};
    if (tail_clamped) {
      mjtNum pT[3], qT[4], R0_T[9], R_rel[9];
      body_pose0(bT, pT, qT);
      mju_quat2Mat(R0_T, qT);
      mju_mulMatMatT(R_rel, d->xmat + 9 * bT, R0_T, 3, 3, 3);
      mju_mulMatVec3(tT_ref, R_rel, t_ref_world.data() + 3 * nhinge);
      mju_mulMatVec3(uT_ref, R_rel, u_ref_world.data() + 3 * nhinge);
    }

    std::vector<bool> anchored(nedge, false);
    for (int e = 0; e < nedge; e++) {
      mjtNum* te = t_edge.data() + 3 * e;
      mjtNum* ue = u_edge.data() + 3 * e;
      mjtNum* ve = v_edge.data() + 3 * e;
      if (e == 0 && root_clamped) {
        pt(t0_ref, te, u0_ref, ue, ve);
        anchored[e] = true;
        continue;
      }
      if (e == nhinge && tail_clamped) {
        pt(tT_ref, te, uT_ref, ue, ve);
        anchored[e] = true;
        continue;
      }
      // time-parallel transport of the state, else spatial transport from the
      // previous edge (edge 0: from its qpos0 reference)
      const mjtNum* state_e = state ? (state + 6 * e) : nullptr;
      bool has_state = state_e && mju_dot3(state_e, state_e) > 0.25 &&
                       mju_dot3(state_e + 3, state_e + 3) > 0.25;
      if (has_state) {
        pt(state_e, te, state_e + 3, ue, ve);
        anchored[e] = true;
      } else if (e > 0) {
        pt(te - 3, te, ue - 3, ue, ve);
      } else {
        pt(t_ref_world.data(), te, u_ref_world.data(), ue, ve);
      }
    }
    std::vector<mjtNum> theta(nedge, 0);
    for (int e = 0; e < nedge; e++) {
      int jnt = mj_rodChainTwistJnt(m, f, e);
      if (jnt >= 0) {
        int padr = m->jnt_qposadr[jnt];
        theta[e] = d->qpos[padr] - m->qpos0[padr];
      }
    }
    if (frames) {
      frames->v_chain = v_chain;
      frames->t = t_edge;
      frames->u = u_edge;
      frames->v = v_edge;
      frames->theta = theta;
    }
    mjtNum chain_energy = 0;
    for (int h = 0; h < nhinge; h++) {
      const mjtNum* hinge = b + 1 + kRodHingeSize * h;
      mjtNum k1 = hinge[4], dk = hinge[5], kt = hinge[8];
      mjtNum k2 = k1 + dk;
      mjtNum kb[3];
      binormal(v_chain[h], v_chain[h + 1], v_chain[h + 2], kb);
      for (int side = 0; side < 2; side++) {
        int e = h + side;
        mjtNum b1 = mju_dot3(kb, u_edge.data() + 3 * e);
        mjtNum b2 = mju_dot3(kb, v_edge.data() + 3 * e);
        mjtNum c = mju_cos(theta[e]), s = mju_sin(theta[e]);
        mjtNum om1 = c * b1 + s * b2;
        mjtNum om2 = -s * b1 + c * b2;
        mjtNum d1 = om1 - om0[2 * h + 0];
        mjtNum d2 = om2 - om0[2 * h + 1];
        chain_energy += 0.5 * (k1 * d1 * d1 + k2 * d2 * d2);
      }
      mjtNum u_trans[3], v_trans[3];
      pt(t_edge.data() + 3 * h, t_edge.data() + 3 * (h + 1),
         u_edge.data() + 3 * h, u_trans, v_trans);
      mjtNum dpsi_h = mju_atan2(mju_dot3(u_edge.data() + 3 * (h + 1), v_trans),
                                mju_dot3(u_edge.data() + 3 * (h + 1), u_trans));
      if (state && anchored[h] && anchored[h + 1]) {
        const mjtNum psi_old = state[6 * nedge + h];
        dpsi_h -= 2 * mjPI * mju_round((dpsi_h - psi_old) / (2 * mjPI));
      }
      mjtNum dtheta = theta[h + 1] - theta[h] + dpsi_h;
      chain_energy += kt * dtheta * dtheta;
    }
    return chain_energy;
  }
  mjtNum energy = 0;
  for (int h = 0; h < (int)b[0]; h++) {
    const mjtNum* hinge = b + 1 + kRodHingeSize * h;
    int v0 = hinge[0], v1 = hinge[1], v2 = hinge[2], v3 = hinge[3];

    // single hinge: k |kb|^2, or k (|kb| - kb0)^2 smoothed at kb = 0
    if (v3 < 0) {
      mjtNum kb[3];
      binormal(v0, v1, v2, kb);
      mjtNum k = hinge[4], kb0 = hinge[6];
      mjtNum r = mju_norm3(kb);
      mjtNum rho = mju_sqrt(r * r + kSmooth * kSmooth * kb0 * kb0);
      energy += k * (rho - gam * kb0) * (rho - gam * kb0);
      continue;
    }

    // hinge pair: binormals along axes orthogonal to the shared edge, chosen
    // differently from the engine since the energy does not depend on them
    mjtNum kba[3], kbb[3], t[3], u[3] = {0, 0, 0}, v[3];
    binormal(v0, v1, v2, kba);
    binormal(v1, v2, v3, kbb);
    mju_sub3(t, xpos + 3 * v2, xpos + 3 * v1);
    mju_normalize3(t);
    int axis = 0;
    for (int i = 1; i < 3; i++) {
      if (std::abs(t[i]) < std::abs(t[axis])) axis = i;
    }
    u[axis] = 1;
    mju_addToScl3(u, t, -t[axis]);
    mju_normalize3(u);
    mju_cross(v, t, u);
    mjtNum om[4] = {mju_dot3(kba, u), mju_dot3(kba, v), mju_dot3(kbb, u),
                    mju_dot3(kbb, v)};

    // radial bending + dihedral torsion across the shared edge
    mjtNum w0 = hinge[4], w1 = hinge[5];
    mjtNum r0a = hinge[6], r0b = hinge[7], tc = hinge[8], ts = hinge[9];
    mjtNum kt = mju_sqrt(tc * tc + ts * ts);
    mjtNum ra = mju_sqrt(om[0] * om[0] + om[1] * om[1]);
    mjtNum rb = mju_sqrt(om[2] * om[2] + om[3] * om[3]);
    mjtNum rhoa = mju_sqrt(ra * ra + kSmooth * kSmooth * r0a * r0a);
    mjtNum rhob = mju_sqrt(rb * rb + kSmooth * kSmooth * r0b * r0b);
    energy += w0 * (rhoa - gam * r0a) * (rhoa - gam * r0a) +
              w1 * (rhob - gam * r0b) * (rhob - gam * r0b);
    if (kt > 0) {
      mjtNum nta[2] = {gam * om[0] / rhoa, gam * om[1] / rhoa};
      mjtNum ntb[2] = {gam * om[2] / rhob, gam * om[3] / rhob};
      mjtNum rna[2] = {(tc * nta[0] - ts * nta[1]) / kt,
                       (ts * nta[0] + tc * nta[1]) / kt};
      mjtNum dx = rna[0] - ntb[0], dy = rna[1] - ntb[1];
      energy += kt * (dx * dx + dy * dy);
    }
  }
  return energy;
}

// set the director state of every 1D DER rod to the current configuration;
// finite differences of positions then see a fixed state (without one, the
// directors are propagated from edge 0 and the forces treat them as frozen)
void FreezeRodState(const mjModel* m, mjData* d) {
  for (int f = 0; f < m->nflex; f++) {
    if (m->flex_dim[f] == 1 && m->flex_plasticadr[f] >= 0) {
      mj_rodChain(m, d, f, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  d->plastic + m->flex_plasticadr[f]);
    }
  }
}

static constexpr char kRodStraightXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
              radius="0.01" dim="1" mass="1">
      <elasticity young="1e7" damping="0.01"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

static constexpr char kRodTurnedXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="20 30 40">
      <flexcomp name="turned_rod" type="grid" count="6 1 1"
                spacing="0.1 0.1 0.1" radius="0.01" dim="1" mass="1">
        <elasticity young="1e7" damping="0.01"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kRodCircleXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="20 30 40">
      <flexcomp name="ring" type="circle" count="8 1 1" spacing="0.15 0.15 0.15"
                radius="0.01" dim="1" mass="1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" damping="0.01"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kRodHelixXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="15 -25 35">
      <flexcomp name="helix" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2  2 3  3 4  4 5  5 6  6 7"
                point="0.15 0.00 0.00   0.106 0.106 0.03   0.00 0.15 0.06
                       -0.106 0.106 0.09   -0.15 0.00 0.12   -0.106 -0.106 0.15
                       0.00 -0.15 0.18   0.106 -0.106 0.21">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" damping="0.01"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kRodVeeXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="vee" type="direct" dim="1" radius="0.01" mass="1"
              element="0 1  1 2" point="0 0 0  0.1 0 0  0.18 0.06 0.02">
      <contact selfcollide="none" contype="0" conaffinity="0"/>
      <elasticity young="1e7" damping="0.01"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

// straight segment ending in a planar arc: 5 single DER hinges (straight at
// vertices 1, 2 and curved at vertices 3, 4, 5)
static constexpr char kRodHookXml[] = R"(
<mujoco>
  <option gravity="0 0 0"/>
  <worldbody>
    <flexcomp name="hook" type="direct" dim="1" radius="0.01" mass="1"
              element="0 1  1 2  2 3  3 4  4 5  5 6"
              point="0 0 0  0.05 0 0  0.1 0 0  0.15 0 0  0.193 0.025 0
                     0.218 0.068 0  0.218 0.118 0">
      <contact selfcollide="none" contype="0" conaffinity="0"/>
      <elasticity young="1e7" damping="0.01"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";

TEST_F(ElasticityTest, RodBendingCompile) {
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kRodStraightXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_GE(m->flex_bendingadr[0], 0);
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];

  // 6 vertices on a line: 4 single straight hinges
  EXPECT_EQ(b[0], 4);
  mjtNum EI = 1e7 * mjPI * std::pow(0.01, 4) / 4;
  for (int h = 0; h < 4; h++) {
    EXPECT_EQ(b[1 + kRodHingeSize * h + 3], -1);
    EXPECT_NEAR(b[1 + kRodHingeSize * h + 4], EI / 0.2, MjTol(1e-12, 1e-6));
  }

  // rotated straight rod
  m = LoadModelFromString(kRodTurnedXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 4);

  // closed circle (count="8 1 1" wraps to 7 vertices): 7 hinge pairs
  m = LoadModelFromString(kRodCircleXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->flex_vertnum[0], 7);
  b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 7);
  for (int h = 0; h < 7; h++) {
    EXPECT_GE(b[1 + kRodHingeSize * h + 3], 0);
  }
}

// references bent by more than 1e-4 rad are kept as the rest shape, as curved
// single hinges or as hinge pairs; smaller bends are treated as straight
TEST_F(ElasticityTest, RodBendingCurvedReference) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2" point="0 0 0  0.1 0 0  0.2 %s 0">
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char xml[2048];
  char error[1024] = {0};

  // bent by 1e-5 rad: straight single hinge
  std::snprintf(xml, sizeof(xml), xml_template, "1e-6");
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1 + 3], -1);
  EXPECT_EQ(b[1 + 6], 0);

  // bent by 1e-3 rad: curved single hinge, storing |kb| = 2 tan(phi / 2)
  std::snprintf(xml, sizeof(xml), xml_template, "1e-4");
  m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[1 + 3], -1);
  EXPECT_NEAR(b[1 + 6], 2 * std::tan(std::atan(1e-3) / 2), MjTol(1e-12, 1e-9));

  // fully folded hinge
  static constexpr char folded[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2" point="0 0 0  0.1 0 0  0 0 0">
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(folded, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("degenerate 1D flex hinge"));

  // open <flexcomp> planar arc of radius 0.2 with 8 vertices: 6 single DER
  // hinges and 7 per-edge twist joints
  static constexpr int kN = 8;
  static constexpr double kRadius = 0.2;
  const double dtheta = mjPI / (kN - 1);
  const double chord = 2 * kRadius * std::sin(dtheta / 2);
  std::string point, element, raw_bodies, raw_body_names;
  for (int i = 0; i < kN; i++) {
    char buf[256];
    double xi = kRadius * std::cos(i * dtheta);
    double yi = kRadius * std::sin(i * dtheta);
    std::snprintf(buf, sizeof(buf), "%.17g %.17g 0  ", xi, yi);
    point += buf;
    if (i) element += std::to_string(i - 1) + " " + std::to_string(i) + "  ";
    std::snprintf(
        buf, sizeof(buf),
        R"(<body name="v%d" pos="%.17g %.17g 0">)"
        R"(<inertial pos="0 0 0" mass="0.125" diaginertia="1e-5 1e-5 1e-5"/>)"
        R"(<joint type="slide" axis="1 0 0"/>)"
        R"(<joint type="slide" axis="0 1 0"/>)"
        R"(<joint type="slide" axis="0 0 1"/></body>)",
        i, xi, yi);
    raw_bodies += buf;
    raw_body_names += "v" + std::to_string(i) + " ";
  }
  std::string arc = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="arc" type="direct" dim="1" radius="0.01" mass="1"
                element=")" +
                    element + R"(" point=")" + point + R"(">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(arc.c_str(), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nv, 3 * kN + (kN - 1));
  b = m->flex_bending + m->flex_bendingadr[0];
  ASSERT_EQ(b[0], kN - 2);

  const double k = 1e7 * mjPI * std::pow(0.01, 4) / 4 / (2 * chord);
  for (int h = 0; h < kN - 2; h++) {
    const mjtNum* hinge = b + 1 + kRodHingeSize * h;
    EXPECT_EQ(hinge[0], h);
    EXPECT_EQ(hinge[1], h + 1);
    EXPECT_EQ(hinge[2], h + 2);
    EXPECT_EQ(hinge[3], -1);
    EXPECT_NEAR(hinge[4], k, MjTol(1e-10, 1e-5) * k) << "hinge " << h;
    EXPECT_GE(hinge[7], 0);
  }
  EXPECT_GE(mj_name2id(m.get(), mjOBJ_JOINT, "arc_twist_0"), 0);
  EXPECT_GE(mj_name2id(m.get(), mjOBJ_JOINT, "arc_twist"), 0);

  // zero energy at rest
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), 0), 0, MjTol(1e-12, 1e-6));

  // in-plane change of curvature at fixed edge lengths: k (|kb| - |kb0|)^2 per
  // hinge
  const double radius = 0.25;
  const double dtheta1 = 2 * std::asin(chord / (2 * radius));
  for (int i = 0; i < kN; i++) {
    d->qpos[3 * i + 0] = radius * std::cos(i * dtheta1) + kRadius - radius -
                         kRadius * std::cos(i * dtheta);
    d->qpos[3 * i + 1] =
        radius * std::sin(i * dtheta1) - kRadius * std::sin(i * dtheta);
  }
  mj_forward(m.get(), d.get());
  double dkb = 2 * std::tan(dtheta1 / 2) - 2 * std::tan(dtheta / 2);
  double energy = (kN - 2) * k * dkb * dkb;
  EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), 0), energy,
              MjTol(1e-6, 1e-2) * energy);

  // the energy decreases along the force
  mjtNum e0 = RodBendingEnergy(m.get(), d.get(), 0);
  std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
  mjtNum fnorm = mju_norm(force.data(), m->nv);
  ASSERT_GT(fnorm, 0);
  for (int i = 0; i < m->nv; i++) {
    d->qpos[i] += 1e-4 / fnorm * force[i];
  }
  mj_fwdPosition(m.get(), d.get());
  EXPECT_LT(RodBendingEnergy(m.get(), d.get(), 0), e0);

  // compute reference edge tangents and Bishop directors for rigid 3D rotations
  auto pt = [](const mjtNum t_from[3], const mjtNum t_to[3],
               const mjtNum u_in[3], mjtNum u_out[3], mjtNum v_out[3]) {
    mjtNum denom = 1 + mju_dot3(t_from, t_to);
    mjtNum sum[3];
    mju_add3(sum, t_from, t_to);
    mjtNum scale = mju_dot3(t_to, u_in) / denom;
    mju_addScl3(u_out, u_in, sum, -scale);
    mju_normalize3(u_out);
    mju_cross(v_out, t_to, u_out);
  };
  std::vector<mjtNum> t_ref(3 * (kN - 1)), u_ref(3 * (kN - 1));
  for (int e = 0; e < kN - 1; e++) {
    mjtNum x0[3] = {static_cast<mjtNum>(kRadius * std::cos(e * dtheta)),
                    static_cast<mjtNum>(kRadius * std::sin(e * dtheta)), 0};
    mjtNum x1[3] = {static_cast<mjtNum>(kRadius * std::cos((e + 1) * dtheta)),
                    static_cast<mjtNum>(kRadius * std::sin((e + 1) * dtheta)),
                    0};
    mju_sub3(t_ref.data() + 3 * e, x1, x0);
    mju_normalize3(t_ref.data() + 3 * e);
  }
  mjtNum frame0[9] = {0};
  mju_copy3(frame0, t_ref.data());
  MakeFrame(frame0);
  mju_copy3(u_ref.data(), frame0 + 3);
  for (int e = 0; e < kN - 2; e++) {
    mjtNum v_tmp[3];
    pt(t_ref.data() + 3 * e, t_ref.data() + 3 * (e + 1), u_ref.data() + 3 * e,
       u_ref.data() + 3 * (e + 1), v_tmp);
  }

  // rigid 3D rotations at rest: setting per-edge twist angles for plastic == 0
  // (and equivalently rotating (t_s, u_s) in d->plastic with theta == 0) yields
  // zero energy and zero qfrc_spring on all m->nv DOFs
  const mjtNum axes[3][3] = {{1, 0, 0}, {0, 1, 1}, {1, -2, 3}};
  const mjtNum angles[3] = {mjPI / 2, 2.0, -1.0};
  int padr = m->flex_plasticadr[0];
  ASSERT_GE(padr, 0);
  for (int r = 0; r < 3; r++) {
    mjtNum quat[4], axis[3];
    mju_copy3(axis, axes[r]);
    mju_normalize3(axis);
    mju_axisAngle2Quat(quat, axis, angles[r]);

    // Case A: d->plastic == 0: the directors start from edge 0's reference
    // transported to R t_ref, PT(t_ref -> R t_ref), and propagate spatially,
    // which is equivariant under R, so one twist angle (that of edge 0) on
    // every edge recovers the rotated material frames
    d = MakeData(m);
    for (int i = 0; i < kN; i++) {
      mjtNum x[3] = {static_cast<mjtNum>(kRadius * std::cos(i * dtheta)),
                     static_cast<mjtNum>(kRadius * std::sin(i * dtheta)), 0};
      mjtNum rx[3];
      mju_rotVecQuat(rx, x, quat);
      mju_sub3(d->qpos + 3 * i, rx, x);
    }
    mjtNum rt0[3], ru0[3], u_pt0[3], v_pt0[3];
    mju_rotVecQuat(rt0, t_ref.data(), quat);
    mju_rotVecQuat(ru0, u_ref.data(), quat);
    pt(t_ref.data(), rt0, u_ref.data(), u_pt0, v_pt0);
    const mjtNum alpha0 = mju_atan2(mju_dot3(ru0, v_pt0), mju_dot3(ru0, u_pt0));
    for (int e = 0; e < kN - 1; e++) {
      int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
      d->qpos[m->jnt_qposadr[jnt]] = alpha0;
    }
    mj_forward(m.get(), d.get());
    EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), 0), 0, MjTol(1e-10, 1e-5));
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i], 0, MjTol(1e-6, 1e-2))
          << "rotation " << r << " (theta), dof " << i;
    }

    // Case B: rotate (t_s, u_s) in d->plastic with theta[e] == 0
    for (int e = 0; e < kN - 1; e++) {
      int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
      d->qpos[m->jnt_qposadr[jnt]] = 0;
      mju_rotVecQuat(d->plastic + padr + 6 * e, t_ref.data() + 3 * e, quat);
      mju_rotVecQuat(d->plastic + padr + 6 * e + 3, u_ref.data() + 3 * e, quat);
    }
    mj_forward(m.get(), d.get());
    EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), 0), 0, MjTol(1e-10, 1e-5));
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i], 0, MjTol(1e-6, 1e-2))
          << "rotation " << r << " (plastic), dof " << i;
    }
  }

  // raw <flex dim="1"> arc without twist joints: compiles to kN - 3 hinge pairs
  std::string raw_arc = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>)" + raw_bodies +
                        R"(</worldbody>
    <deformable>
      <flex name="arc" dim="1" radius="0.01" body=")" +
                        raw_body_names + R"(" element=")" + element + R"(">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  MjModelPtr m_raw = LoadModelFromString(raw_arc.c_str(), error, sizeof(error));
  ASSERT_THAT(m_raw.get(), NotNull()) << error;
  const mjtNum* b_raw = m_raw->flex_bending + m_raw->flex_bendingadr[0];
  ASSERT_EQ(b_raw[0], kN - 3);
  std::vector<double> stiffness(kN, 0);
  for (int h = 0; h < kN - 3; h++) {
    const mjtNum* hinge = b_raw + 1 + kRodHingeSize * h;
    EXPECT_GE(hinge[3], 0);
    stiffness[static_cast<int>(hinge[1])] += hinge[4];
    stiffness[static_cast<int>(hinge[2])] += hinge[5];
  }
  for (int v = 1; v < kN - 1; v++) {
    EXPECT_NEAR(stiffness[v], k, MjTol(1e-10, 1e-5) * k) << "vertex " << v;
  }
  MjDataPtr d_raw = MakeData(m_raw);
  for (int r = 0; r < 3; r++) {
    mjtNum quat[4], axis[3];
    mju_copy3(axis, axes[r]);
    mju_normalize3(axis);
    mju_axisAngle2Quat(quat, axis, angles[r]);
    for (int i = 0; i < kN; i++) {
      mjtNum x[3] = {static_cast<mjtNum>(kRadius * std::cos(i * dtheta)),
                     static_cast<mjtNum>(kRadius * std::sin(i * dtheta)), 0};
      mjtNum rx[3];
      mju_rotVecQuat(rx, x, quat);
      mju_sub3(d_raw->qpos + 3 * i, rx, x);
    }
    mj_forward(m_raw.get(), d_raw.get());
    EXPECT_NEAR(RodBendingEnergy(m_raw.get(), d_raw.get(), 0), 0,
                MjTol(1e-10, 1e-5));
    for (int i = 0; i < m_raw->nv; i++) {
      EXPECT_NEAR(d_raw->qfrc_spring[i], 0, MjTol(1e-6, 1e-2))
          << "raw rotation " << r << ", dof " << i;
    }
  }

  // straight segment ending in an arc: 5 single DER hinges (0, 1 straight and
  // 2, 3, 4 curved)
  m = LoadModelFromString(kRodHookXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  b = m->flex_bending + m->flex_bendingadr[0];
  ASSERT_EQ(b[0], 5);
  for (int h = 0; h < 5; h++) {
    const mjtNum* hinge = b + 1 + kRodHingeSize * h;
    EXPECT_EQ(hinge[0], h);
    EXPECT_EQ(hinge[1], h + 1);
    EXPECT_EQ(hinge[2], h + 2);
    EXPECT_EQ(hinge[3], -1);
    if (h < 2) {
      EXPECT_EQ(hinge[6], 0);
    } else {
      EXPECT_GT(hinge[6], 0);
    }
  }

  // vertex 3 (hinge 2) has the full stiffness EI / (l0 + l1)
  const double EI = 1e7 * mjPI * std::pow(0.01, 4) / 4;
  const double l1 = std::sqrt(0.043 * 0.043 + 0.025 * 0.025);
  const mjtNum* hinge2 = b + 1 + kRodHingeSize * 2;
  EXPECT_NEAR(hinge2[4], EI / (0.05 + l1), MjTol(1e-10, 1e-5) * hinge2[4]);
}

// out-of-plane twist on curved hinges is resisted with torsional stiffness
// kt = GJ / (l0 + l1) on open DER chains and kt = GJ / (2 l) on hinge pairs
TEST_F(ElasticityTest, RodBendingTorsion) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="zrod" type="direct" dim="1" radius="0.01" mass="1"
                point="0 1 0  0 0 0  1 0 0  1 1 0">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" poisson="%s" %s/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char xml[2048];
  char error[1024] = {0};

  // <flexcomp> circle section with poisson = 0.25: 2 single DER hinges
  // (G = E / 2.5, J = pi r^4 / 2, l0 + l1 = 2)
  std::snprintf(xml, sizeof(xml), xml_template, "0.25", "");
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
  ASSERT_EQ(b[0], 2);
  const mjtNum* pair = b + 1;
  EXPECT_EQ(pair[3], -1);

  const double J_circle = mjPI * std::pow(0.01, 4) / 2;
  const double kt = (1e7 / (2 * 1.25)) * J_circle / 2.0;
  EXPECT_NEAR(pair[6], 2.0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(pair[8], kt, MjTol(1e-12, 1e-6) * kt);
  EXPECT_NEAR(b[1 + kRodHingeSize + 6], 2.0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(b[1 + kRodHingeSize + 8], kt, MjTol(1e-12, 1e-6) * kt);

  // rotate vertex 3 around the shared edge (1, 2) by angle psi: verify
  // restoring force along -z and finite-difference gradient consistency
  MjDataPtr d = MakeData(m);
  const double psi = 0.6;
  d->qpos[3 * 3 + 1] = std::cos(psi) - 1.0;
  d->qpos[3 * 3 + 2] = std::sin(psi);
  mj_forward(m.get(), d.get());
  FreezeRodState(m.get(), d.get());
  mj_forward(m.get(), d.get());
  EXPECT_LT(d->qfrc_spring[3 * 3 + 2], 0);
  std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
  mjtNum fmax = 0;
  for (mjtNum fi : force) fmax = std::max(fmax, std::abs(fi));
  ASSERT_GT(fmax, 0);
  const mjtNum eps = MjEps(1e-6, 1e-3);
  for (int i = 0; i < m->nv; i++) {
    mjtNum q = d->qpos[i];
    d->qpos[i] = q + eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum ep = RodBendingEnergy(m.get(), d.get(), 0);
    d->qpos[i] = q - eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum em = RodBendingEnergy(m.get(), d.get(), 0);
    d->qpos[i] = q;
    EXPECT_NEAR(force[i], -(ep - em) / (2 * eps), MjTol(1e-5, 1e-2) * fmax)
        << "dof " << i;
  }

  // raw <flex dim="1"> zrod without twist joints: 1 hinge pair (mj_rodPair)
  static constexpr char kRawZrodXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="v0" pos="0 1 0">
        <inertial pos="0 0 0" mass="0.25" diaginertia="1e-5 1e-5 1e-5"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
      <body name="v1" pos="0 0 0">
        <inertial pos="0 0 0" mass="0.25" diaginertia="1e-5 1e-5 1e-5"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
      <body name="v2" pos="1 0 0">
        <inertial pos="0 0 0" mass="0.25" diaginertia="1e-5 1e-5 1e-5"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
      <body name="v3" pos="1 1 0">
        <inertial pos="0 0 0" mass="0.25" diaginertia="1e-5 1e-5 1e-5"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="zrod" dim="1" radius="0.01" body="v0 v1 v2 v3"
            element="0 1  1 2  2 3">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" poisson="0.25"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  MjModelPtr m_raw = LoadModelFromString(kRawZrodXml, error, sizeof(error));
  ASSERT_THAT(m_raw.get(), NotNull()) << error;
  const mjtNum* b_raw = m_raw->flex_bending + m_raw->flex_bendingadr[0];
  ASSERT_EQ(b_raw[0], 1);
  const mjtNum* raw_pair = b_raw + 1;
  EXPECT_EQ(raw_pair[3], 3);
  EXPECT_NEAR(raw_pair[6], 2.0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(raw_pair[7], 2.0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(raw_pair[8], kt, MjTol(1e-12, 1e-6) * kt);
  EXPECT_NEAR(raw_pair[9], 0.0, MjTol(1e-12, 1e-6) * kt);
  MjDataPtr d_raw = MakeData(m_raw);
  d_raw->qpos[3 * 3 + 1] = std::cos(psi) - 1.0;
  d_raw->qpos[3 * 3 + 2] = std::sin(psi);
  mj_forward(m_raw.get(), d_raw.get());
  double expected_energy = 2 * kt * (1 - std::cos(psi));
  EXPECT_NEAR(RodBendingEnergy(m_raw.get(), d_raw.get(), 0), expected_energy,
              MjTol(1e-12, 1e-5) * expected_energy);
  EXPECT_LT(d_raw->qfrc_spring[3 * 3 + 2], 0);

  // box section: Saint-Venant torsion constant J (single DER hinges)
  std::snprintf(xml, sizeof(xml), xml_template, "0.0",
                R"(section="box" sectionsize="0.02 0.01")");
  m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 2);
  pair = b + 1;
  EXPECT_EQ(pair[3], -1);
  const double a = 0.02, c = 0.01;
  const double J_box =
      a * c * c * c *
      (16.0 / 3.0 - 3.36 * (c / a) * (1.0 - std::pow(c / a, 4) / 12.0));
  const double kt_box = (1e7 / 2.0) * J_box / 2.0;
  EXPECT_NEAR(pair[8], kt_box, MjTol(1e-12, 1e-6) * kt_box);
}

// a straight reference can be posed curved through qpos: the bending force
// straightens it
TEST_F(ElasticityTest, RodBendingCurvedPose) {
  char error[1024] = {0};

  // rod bent at its middle vertex
  static constexpr char rod[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2" point="0 0 0  0.1 0 0  0.2 0 0">
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(rod, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nv, 11);
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->qfrc_spring[i], 0, MjTol(1e-8, 1e-4)) << "dof " << i;
  }
  d->qpos[7] = 0.05;
  mj_forward(m.get(), d.get());
  const mjtNum* f = d->qfrc_spring;
  EXPECT_LT(f[1], 0);  // vertex 0 moves down
  EXPECT_GT(f[4], 0);  // the hinge vertex moves up
  EXPECT_LT(f[7], 0);  // vertex 2 moves down
  EXPECT_NEAR(f[1] + f[4] + f[7], 0, MjTol(1e-8, 1e-3) * std::abs(f[4]));

  // straight rod coiled into an open arc of 300 degrees: the energy decreases
  // along the force
  static constexpr int kArc = 11;
  const double radius = 0.2;
  const double angle = 5 * mjPI / 3 / (kArc - 1);
  const double chord = 2 * radius * std::sin(angle / 2);
  std::string points;
  std::string elements;
  for (int i = 0; i < kArc; i++) {
    points += std::to_string(chord * i) + " 0 0 ";
    if (i > 0)
      elements += std::to_string(i - 1) + " " + std::to_string(i) + " ";
  }
  std::string straight = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="1"
                element=")" +
                         elements + R"(" point=")" + points + R"(">
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(straight.c_str(), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nv, 3 * kArc + (kArc - 1));
  EXPECT_EQ(m->flex_bending[m->flex_bendingadr[0]], kArc - 2);
  d = MakeData(m);
  for (int i = 0; i < kArc; i++) {
    d->qpos[3 * i + 0] = radius * std::sin(angle * i) - chord * i;
    d->qpos[3 * i + 1] = radius * (1 - std::cos(angle * i));
  }
  mj_forward(m.get(), d.get());
  mjtNum e0 = RodBendingEnergy(m.get(), d.get(), 0);
  EXPECT_GT(e0, 0);
  std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
  mjtNum fnorm = mju_norm(force.data(), m->nv);
  ASSERT_GT(fnorm, 0);
  const mjtNum step = 1e-3 / fnorm;
  for (int i = 0; i < m->nv; i++) {
    d->qpos[i] += step * force[i];
  }
  mj_fwdPosition(m.get(), d.get());
  EXPECT_LT(RodBendingEnergy(m.get(), d.get(), 0), e0);
}

TEST_F(ElasticityTest, RodBendingWithEdgeEquality) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="grid" count="4 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="1">
        <elasticity young="1e7"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_GE(m->flex_bendingadr[0], 0);
}

TEST_F(ElasticityTest, RodBendingZeroForceAtRest) {
  for (const char* xml : {kRodStraightXml, kRodTurnedXml, kRodCircleXml,
                          kRodHelixXml, kRodVeeXml, kRodHookXml}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    SCOPED_TRACE(m->names + m->name_flexadr[0]);
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i], 0, MjTol(1e-8, 1e-4)) << "dof " << i;
    }
  }
}

TEST_F(ElasticityTest, RodBendingForceIsEnergyGradient) {
  for (const char* xml : {kRodStraightXml, kRodTurnedXml, kRodCircleXml,
                          kRodHelixXml, kRodVeeXml, kRodHookXml}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    SCOPED_TRACE(m->names + m->name_flexadr[0]);
    MjDataPtr d = MakeData(m);

    // deterministic perturbation of the slide joints
    for (int i = 0; i < m->nq; i++) {
      d->qpos[i] = 0.02 * std::sin(1.7 * i + 0.3);
    }
    mj_forward(m.get(), d.get());
    FreezeRodState(m.get(), d.get());
    mj_forward(m.get(), d.get());
    std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
    mjtNum fmax = 0;
    for (mjtNum fi : force) fmax = std::max(fmax, std::abs(fi));
    ASSERT_GT(fmax, 0);

    // central differences of the energy (slide joints: qpos and qvel spaces
    // coincide)
    const mjtNum eps = MjEps(1e-6, 1e-3);
    for (int i = 0; i < m->nv; i++) {
      mjtNum q = d->qpos[i];
      d->qpos[i] = q + eps;
      mj_fwdPosition(m.get(), d.get());
      mjtNum ep = RodBendingEnergy(m.get(), d.get(), 0);
      d->qpos[i] = q - eps;
      mj_fwdPosition(m.get(), d.get());
      mjtNum em = RodBendingEnergy(m.get(), d.get(), 0);
      d->qpos[i] = q;
      EXPECT_NEAR(force[i], -(ep - em) / (2 * eps), MjTol(1e-5, 1e-2) * fmax)
          << "dof " << i;
    }
  }
}

TEST_F(ElasticityTest, RodBendingDamping) {
  for (const char* xml : {kRodStraightXml, kRodCircleXml, kRodHelixXml,
                          kRodVeeXml, kRodHookXml}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    SCOPED_TRACE(m->names + m->name_flexadr[0]);
    MjDataPtr d = MakeData(m);

    // bent configuration
    for (int i = 0; i < m->nq; i++) {
      d->qpos[i] = 0.02 * std::sin(1.7 * i + 0.3);
    }

    // arbitrary velocity: damping dissipates energy
    for (int i = 0; i < m->nv; i++) {
      d->qvel[i] = std::cos(2.3 * i + 0.1);
    }
    mj_forward(m.get(), d.get());
    EXPECT_LT(mju_dot(d->qfrc_damper, d->qvel, m->nv), 0);

    // rigid rotation about an arbitrary axis: no damping (qvel is in body
    // frame), with per-edge twist hinges spinning at omega . t_e
    mjtNum omega[3] = {0.3, -0.5, 0.7};
    mju_zero(d->qvel, m->nv);
    int nvert = m->flex_vertnum[0];
    for (int v = 0; v < nvert; v++) {
      int bid = m->flex_vertbodyid[m->flex_vertadr[0] + v];
      int dofadr = m->body_dofadr[bid];
      mjtNum vworld[3];
      mju_cross(vworld, omega, d->flexvert_xpos + 3 * v);
      mju_mulMatTVec3(d->qvel + dofadr, d->xmat + 9 * m->body_weldid[bid],
                      vworld);
    }
    if (m->flex_bendingadr[0] >= 0) {
      const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
      int nhinge = static_cast<int>(b[0]);
      if (mj_rodHasTwist(m.get(), 0)) {
        for (int e = 0; e <= nhinge; e++) {
          int jid = mj_rodChainTwistJnt(m.get(), 0, e);
          if (jid >= 0) {
            const mjtNum* p = b + 1 + kRodHingeSize * (e == 0 ? 0 : e - 1);
            int va = static_cast<int>(e == 0 ? p[0] : p[1]);
            int vb = static_cast<int>(e == 0 ? p[1] : p[2]);
            mjtNum te[3];
            mju_sub3(te, d->flexvert_xpos + 3 * vb, d->flexvert_xpos + 3 * va);
            mju_normalize3(te);
            d->qvel[m->jnt_dofadr[jid]] = mju_dot3(omega, te);
          }
        }
      }
    }
    mj_forward(m.get(), d.get());
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_damper[i], 0, MjTol(1e-8, 1e-5)) << "dof " << i;
    }
  }
}

TEST_F(ElasticityTest, RodBendingDisableFlags) {
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kRodStraightXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  for (int i = 0; i < m->nq; i++) {
    d->qpos[i] = 0.02 * std::sin(1.7 * i + 0.3);
    d->qvel[i] = std::cos(2.3 * i + 0.1);
  }
  m->opt.disableflags |= mjDSBL_SPRING | mjDSBL_DAMPER;
  mj_forward(m.get(), d.get());
  for (int i = 0; i < m->nv; i++) {
    EXPECT_EQ(d->qfrc_passive[i], 0);
  }
}

// static tip deflection of a clamped cantilever under a transverse tip load
// matches Euler-Bernoulli beam theory, delta = P L^3 / (3 E I); the discrete
// integrator settles the rod implicitly at a time step far above the explicit
// stability limit of the bending modes
TEST_F(ElasticityTest, RodBendingCantilever) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0" timestep="1e-3"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="22 1 1" spacing="0.05 0.05 0.05"
                radius="0.01" dim="1" mass="1">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <edge equality="true"/>
        <elasticity young="1e9" damping="0.2" %s/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  // circle of the flex radius, and a box whose weak axis is along the second
  // half-size
  struct Section {
    const char* attrib;
    mjtNum I;
  };
  const Section sections[] = {
      {"", static_cast<mjtNum>(mjPI * std::pow(0.01, 4) / 4)},
      {R"(section="box" sectionsize="0.02 0.01")",
       static_cast<mjtNum>(0.04 * std::pow(0.02, 3) / 12)},
  };

  for (const Section& section : sections) {
    char xml[2048];
    std::snprintf(xml, sizeof(xml), xml_template, section.attrib);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    // vertices 0 and 1 are pinned, load at the last vertex. Each hinge is a
    // rotational spring over its Voronoi region, so the hinge at vertex 1 also
    // covers half of the pinned edge and the beam is effectively clamped at the
    // midpoint of that edge; with this length the discretization error is
    // O(1/n^2) (0.06% here) instead of O(1/n)
    int nvert = m->flex_vertnum[0];
    int tip = nvert - 1;
    const mjtNum* x = d->flexvert_xpos + 3 * m->flex_vertadr[0];
    mjtNum L = x[3 * tip] - (x[0] + x[3]) / 2;
    EXPECT_NEAR(L, 1.025, MjTol(1e-12, 1e-6));
    mjtNum z0 = x[3 * tip + 2];

    mjtNum EI = 1e9 * section.I;
    mjtNum delta = 0.01 * L;  // small deflection, linear regime
    mjtNum P = 3 * EI * delta / (L * L * L);
    int tipbody = m->flex_vertbodyid[m->flex_vertadr[0] + tip];
    d->xfrc_applied[6 * tipbody + 2] = -P;

    // settle
    for (int i = 0; i < 5000; i++) {
      mj_step(m.get(), d.get());
    }
    mjtNum vmax = 0;
    for (int i = 0; i < m->nv; i++) vmax = std::max(vmax, std::abs(d->qvel[i]));
    EXPECT_LT(vmax, 1e-6) << "cantilever did not settle: " << section.attrib;

    mj_forward(m.get(), d.get());
    mjtNum deflection = z0 - x[3 * tip + 2];
    EXPECT_NEAR(deflection, delta, 0.005 * delta) << section.attrib;
  }
}

TEST_F(ElasticityTest, RodBendingBoxSection) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.005" dim="1" mass="1">
        <elasticity young="1e7" %s/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char xml[2048];
  char error[1024] = {0};

  // weak axis is the first half-size: I = h w^3 / 12 with w = 0.01, h = 0.04
  std::snprintf(xml, sizeof(xml), xml_template,
                R"(section="box" sectionsize="0.005 0.02")");
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
  EXPECT_EQ(b[0], 4);
  mjtNum EI = 1e7 * 0.04 * std::pow(0.01, 3) / 12;
  for (int h = 0; h < 4; h++) {
    EXPECT_NEAR(b[1 + kRodHingeSize * h + 4], EI / 0.2, MjTol(1e-12, 1e-6));
  }

  // saved XML keeps the section
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, NotNull()) << error;
  mjModel* compiled = mj_compile(spec, nullptr);
  ASSERT_THAT(compiled, NotNull()) << mjs_getError(spec);
  mj_deleteModel(compiled);
  std::string saved = SaveAndReadXml(spec);
  mj_deleteSpec(spec);
  EXPECT_THAT(saved, HasSubstr(R"(section="box")"));
  EXPECT_THAT(saved, HasSubstr(R"(sectionsize="0.005 0.02")"));
  MjModelPtr m2 = LoadModelFromString(saved.c_str(), error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  const mjtNum* b2 = m2->flex_bending + m2->flex_bendingadr[0];
  EXPECT_NEAR(b2[1 + 4], EI / 0.2, MjTol(1e-12, 1e-6));

  // box requires positive sizes
  std::snprintf(xml, sizeof(xml), xml_template, R"(section="box")");
  m = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("box section requires positive sectionsize"));
  // box section requires an open chain
  static constexpr char kCircleBoxXml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="ring" type="circle" count="8 1 1" spacing="0.1 0.1 0.1"
                radius="0.005" dim="1" mass="1">
        <elasticity young="1e7" section="box" sectionsize="0.005 0.02"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(kCircleBoxXml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(
      error,
      HasSubstr("1D flex elasticity with box section requires an open chain"));
}

// verify twist hinge joint, twist-to-writhe coupling, rotating root body b0,
// and anisotropic box tridiagonal quasistatic twist solve
TEST_F(ElasticityTest, RodTwistAndAnisotropicBox) {
  static constexpr char kCircleRodXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="turned" euler="15 -25 35">
        <joint type="hinge" axis="1 0 0"/>
        <joint type="hinge" axis="0 1 0"/>
        <joint type="hinge" axis="0 0 1"/>
        <geom size="0.01"/>
        <flexcomp name="rod" type="grid" count="7 1 1" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="1" mass="1">
          <pin id="0 1"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" poisson="0.25"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char kBoxRodXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="turned" euler="15 -25 35">
        <joint type="hinge" axis="1 0 0"/>
        <joint type="hinge" axis="0 1 0"/>
        <joint type="hinge" axis="0 0 1"/>
        <geom size="0.01"/>
        <body name="mount">
          <flexcomp name="rod" type="grid" count="7 1 1" spacing="0.1 0.1 0.1"
                    radius="0.005" dim="1" mass="1">
            <pin id="0 1"/>
            <contact selfcollide="none" contype="0" conaffinity="0"/>
            <elasticity young="1e7" poisson="0.25"
                        section="box" sectionsize="0.005 0.02"/>
          </flexcomp>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {kCircleRodXml, kBoxRodXml}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    int twist_jnt = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist");
    ASSERT_GE(twist_jnt, 0);
    int twist_dof = m->jnt_dofadr[twist_jnt];
    int twist_qpos = m->jnt_qposadr[twist_jnt];

    MjDataPtr d = MakeData(m);
    for (int i = 0; i < m->nq; i++) {
      d->qpos[i] = (i < 3 ? 0.05 : 0.005) * std::sin(1.7 * i + 0.3);
    }
    d->qpos[twist_qpos] = 0.4;
    mj_forward(m.get(), d.get());
    FreezeRodState(m.get(), d.get());
    mj_forward(m.get(), d.get());

    EXPECT_LT(d->qfrc_spring[twist_dof], 0);
    std::vector<mjtNum> force(d->qfrc_spring, d->qfrc_spring + m->nv);
    mjtNum fmax = 0;
    for (mjtNum fi : force) fmax = std::max(fmax, std::abs(fi));
    ASSERT_GT(fmax, 0);

    const mjtNum eps = MjEps(1e-6, 1e-3);
    for (int i = 0; i < m->nv; i++) {
      mjtNum q = d->qpos[i];
      d->qpos[i] = q + eps;
      mj_fwdPosition(m.get(), d.get());
      mjtNum ep = RodBendingEnergy(m.get(), d.get(), 0);
      d->qpos[i] = q - eps;
      mj_fwdPosition(m.get(), d.get());
      mjtNum em = RodBendingEnergy(m.get(), d.get(), 0);
      d->qpos[i] = q;
      EXPECT_NEAR(force[i], -(ep - em) / (2 * eps), MjTol(1e-5, 1e-2) * fmax)
          << "dof " << i;
    }
  }

  // verify anisotropic stiffness ratio (strong vs weak axis = (0.02/0.005)^2 =
  // 16) and equalized low-mode damping ratio (damping ratio = sqrt(16) = 4)
  static constexpr char kFlatBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                radius="0.005" dim="1" mass="1">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" damping="0.01"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kFlatBoxXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d_weak = MakeData(m);
  MjDataPtr d_strong = MakeData(m);
  // displace vertex 2 in z (weak axis) vs y (strong axis), and give unit
  // velocity along the same axis
  d_weak->qpos[2] = 0.01;
  d_weak->qvel[2] = 1.0;
  d_strong->qpos[1] = 0.01;
  d_strong->qvel[1] = 1.0;
  mj_forward(m.get(), d_weak.get());
  mj_forward(m.get(), d_strong.get());
  mjtNum e_weak = RodBendingEnergy(m.get(), d_weak.get(), 0);
  mjtNum e_strong = RodBendingEnergy(m.get(), d_strong.get(), 0);
  EXPECT_GT(e_weak, 0);
  EXPECT_NEAR(e_strong / e_weak, 16.0, MjTol(1e-10, 1e-4));
  EXPECT_LT(d_weak->qfrc_damper[2], 0);
  EXPECT_NEAR(d_strong->qfrc_damper[1] / d_weak->qfrc_damper[2], 4.0,
              MjTol(1e-10, 1e-4));

  // zigzag strong-axis bending, where the exact Hessian of the twist solve is
  // indefinite: forces stay consistent with the energy
  MjDataPtr d_bent = MakeData(m);
  int twist_qpos =
      m->jnt_qposadr[mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist")];
  d_bent->qpos[1] = 0.05;
  d_bent->qpos[2] = 0.01;
  d_bent->qpos[4] = -0.05;
  d_bent->qpos[7] = 0.05;
  d_bent->qpos[twist_qpos] = 0.3;
  mj_forward(m.get(), d_bent.get());
  FreezeRodState(m.get(), d_bent.get());
  mj_forward(m.get(), d_bent.get());
  std::vector<mjtNum> force(d_bent->qfrc_spring, d_bent->qfrc_spring + m->nv);
  mjtNum fmax = 0;
  for (mjtNum fi : force) fmax = std::max(fmax, std::abs(fi));
  ASSERT_GT(fmax, 0);
  const mjtNum eps = MjEps(1e-6, 1e-3);
  for (int i = 0; i < m->nv; i++) {
    mjtNum q = d_bent->qpos[i];
    d_bent->qpos[i] = q + eps;
    mj_fwdPosition(m.get(), d_bent.get());
    mjtNum ep = RodBendingEnergy(m.get(), d_bent.get(), 0);
    d_bent->qpos[i] = q - eps;
    mj_fwdPosition(m.get(), d_bent.get());
    mjtNum em = RodBendingEnergy(m.get(), d_bent.get(), 0);
    d_bent->qpos[i] = q;
    EXPECT_NEAR(force[i], -(ep - em) / (2 * eps), MjTol(1e-5, 1e-2) * fmax)
        << "dof " << i;
  }

  // free and single-vertex-pinned rods have a free root twist joint and are
  // SO(3) rigid-rotation invariant even for anisotropic box cross-sections
  static constexpr char kFreeBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.005" dim="1" mass="1">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" poisson="0.25"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m_free = LoadModelFromString(kFreeBoxXml, error, sizeof(error));
  ASSERT_THAT(m_free.get(), NotNull()) << error;
  EXPECT_GE(mj_name2id(m_free.get(), mjOBJ_JOINT, "rod_twist_0"), 0);
  EXPECT_GE(mj_name2id(m_free.get(), mjOBJ_JOINT, "rod_twist"), 0);
  // qpos and dof addresses of the 5 twist joints, by chain edge
  int tq[5], td[5];
  for (int e = 0; e < 5; e++) {
    int jnt = mj_rodChainTwistJnt(m_free.get(), 0, e);
    ASSERT_GE(jnt, 0);
    tq[e] = m_free->jnt_qposadr[jnt];
    td[e] = m_free->jnt_dofadr[jnt];
  }
  auto relax_roll = [&](const mjModel* m_in, mjData* d_in) {
    const mjtNum eps_t = 1e-7;
    for (int iter = 0; iter < 10; iter++) {
      mj_forward(m_in, d_in);
      mjtNum f0 = 0;
      for (int e = 0; e < 5; e++) f0 += d_in->qfrc_spring[td[e]];
      if (std::abs(f0) < 1e-13) break;
      for (int e = 0; e < 5; e++) d_in->qpos[tq[e]] += eps_t;
      mj_forward(m_in, d_in);
      mjtNum fp = 0;
      for (int e = 0; e < 5; e++) fp += d_in->qfrc_spring[td[e]];
      for (int e = 0; e < 5; e++) d_in->qpos[tq[e]] -= eps_t;
      mjtNum df = (fp - f0) / eps_t;
      mjtNum step = (std::abs(df) > 1e-12) ? (-f0 / df) : (0.1 * f0);
      for (int e = 0; e < 5; e++) d_in->qpos[tq[e]] += step;
    }
    mj_forward(m_in, d_in);
  };
  MjDataPtr d_free = MakeData(m_free);
  for (int i = 0; i < m_free->nq; i++) {
    d_free->qpos[i] = 0.02 * std::sin(1.7 * i + 0.3);
  }
  relax_roll(m_free.get(), d_free.get());
  int padr = m_free->flex_plasticadr[0];
  ASSERT_GE(padr, 0);
  EXPECT_EQ(m_free->nplastic, 6 * 5 + 4);
  mj_step(m_free.get(), d_free.get());
  mj_forward(m_free.get(), d_free.get());
  mjtNum e_orig = RodBendingEnergy(m_free.get(), d_free.get(), 0);
  std::vector<mjtNum> f_orig(d_free->qfrc_spring,
                             d_free->qfrc_spring + m_free->nv);
  // rotate all free vertices (1..5) and the per-edge time-parallel reference
  // frames around pinned vertex 0 at origin by an arbitrary 3D rotation;
  // per-edge twist angles are unchanged (SO(3) frame-indifferent)
  mjtNum axis[3] = {0.4, -0.7, 0.5};
  mju_normalize3(axis);
  mjtNum qrot[4], rmat[9];
  mju_axisAngle2Quat(qrot, axis, 1.23);
  mju_quat2Mat(rmat, qrot);
  MjDataPtr d_rot = MakeData(m_free);
  for (int v = 1; v < 6; v++) {
    mjtNum rel[3], rot_rel[3], x_rot[3];
    mju_sub3(rel, d_free->flexvert_xpos + 3 * v, d_free->flexvert_xpos);
    mju_mulMatVec3(rot_rel, rmat, rel);
    mju_add3(x_rot, d_free->flexvert_xpos, rot_rel);
    int b_id = m_free->flex_vertbodyid[v];
    int qadr = m_free->jnt_qposadr[m_free->body_jntadr[b_id]];
    mju_sub3(d_rot->qpos + qadr, x_rot, m_free->body_pos + 3 * b_id);
  }
  for (int e = 0; e < 5; e++) {
    d_rot->qpos[tq[e]] = d_free->qpos[tq[e]];
    mju_mulMatVec3(d_rot->plastic + padr + 6 * e, rmat,
                   d_free->plastic + padr + 6 * e);
    mju_mulMatVec3(d_rot->plastic + padr + 6 * e + 3, rmat,
                   d_free->plastic + padr + 6 * e + 3);
  }
  mj_forward(m_free.get(), d_rot.get());
  mjtNum e_rot = RodBendingEnergy(m_free.get(), d_rot.get(), 0);
  EXPECT_NEAR(e_rot, e_orig, MjTol(1e-9, 1e-4) * e_orig);
  for (int v = 1; v < 6; v++) {
    int dadr = m_free->body_dofadr[m_free->flex_vertbodyid[v]];
    mjtNum f_exp[3];
    mju_mulMatVec3(f_exp, rmat, f_orig.data() + dadr);
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(d_rot->qfrc_spring[dadr + j], f_exp[j], MjTol(1e-6, 1e-2));
    }
  }
  for (int e = 0; e < 5; e++) {
    EXPECT_NEAR(d_rot->qfrc_spring[td[e]], f_orig[td[e]], MjTol(1e-6, 1e-2));
  }

  // twist bodies are parented under worldbody with 1e-9 mass so parent-body
  // translation/rotation is not double-counted
  for (int e = 0; e < 5; e++) {
    char jname[32];
    if (e < 4) {
      std::snprintf(jname, sizeof(jname), "rod_twist_%d", e);
    } else {
      std::snprintf(jname, sizeof(jname), "rod_twist");
    }
    int jid = mj_name2id(m_free.get(), mjOBJ_JOINT, jname);
    ASSERT_GE(jid, 0);
    int bid = m_free->jnt_bodyid[jid];
    EXPECT_EQ(m_free->body_parentid[bid], 0);
    EXPECT_NEAR(m_free->body_mass[bid], 1e-9, 1e-15);
  }
  static constexpr char kImplicitBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 -9.81" timestep="0.002" integrator="discrete"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="11 1 1" spacing="0.05 0.05 0.05"
                radius="0.005" dim="1" mass="0.1">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" poisson="0.25" damping="0.01"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m_imp = LoadModelFromString(kImplicitBoxXml, error, sizeof(error));
  ASSERT_THAT(m_imp.get(), NotNull()) << error;
  MjDataPtr d_imp = MakeData(m_imp);
  int j_end = mj_name2id(m_imp.get(), mjOBJ_JOINT, "rod_twist");
  ASSERT_GE(j_end, 0);
  d_imp->qpos[m_imp->jnt_qposadr[j_end]] = 0.5;
  for (int step = 0; step < 50; step++) {
    mj_step(m_imp.get(), d_imp.get());
    ASSERT_TRUE(std::isfinite(d_imp->qpos[m_imp->jnt_qposadr[j_end]]));
  }
  EXPECT_LT(std::abs(d_imp->qpos[m_imp->jnt_qposadr[j_end]]), 0.5);

  // box-section 1D flex rendering generates 12 triangles per element using the
  // oriented DER box cross-section
  static constexpr char kRenderBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="4 1 1" spacing="0.1 0.1 0.1"
                radius="0.005" dim="1" mass="1">
        <pin id="0 1"/>
        <contact selfcollide="none"/>
        <elasticity young="1e7" poisson="0.25"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m_ren = LoadModelFromString(kRenderBoxXml, error, sizeof(error));
  ASSERT_THAT(m_ren.get(), NotNull()) << error;
  MjDataPtr d_ren = MakeData(m_ren);
  mj_forward(m_ren.get(), d_ren.get());
  mjvScene scn;
  mjvOption opt;
  mjvCamera cam;
  mjv_defaultScene(&scn);
  mjv_defaultOption(&opt);
  mjv_defaultCamera(&cam);
  mjv_makeScene(m_ren.get(), &scn, 100);
  mjv_updateScene(m_ren.get(), d_ren.get(), &opt, nullptr, &cam, mjCAT_ALL,
                  &scn);
  EXPECT_EQ(scn.flexfacenum[0], 12 * m_ren->flex_elemnum[0]);
  EXPECT_EQ(scn.flexfaceused[0], 12 * m_ren->flex_elemnum[0]);
  mjv_freeScene(&scn);

  // box-section 1D flex collision uses the oriented DER box cross-section and
  // computes rotational contact Jacobians on per-edge twist DOFs and vertices
  static constexpr char kCollideBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 -9.81" timestep="0.001"/>
    <worldbody>
      <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1"/>
      <geom name="ball" type="sphere" pos="0.05 0.022 0.05" size="0.005"/>
      <geom name="block" type="box" pos="0.15 0 0.059" size="0.005 0.005 0.005"/>
      <flexcomp name="rod" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                pos="0 0 0.05" radius="0.005" dim="1" mass="0.1">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none"/>
        <elasticity young="1e6" poisson="0.25" damping="0.01"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m_col = LoadModelFromString(kCollideBoxXml, error, sizeof(error));
  ASSERT_THAT(m_col.get(), NotNull()) << error;
  MjDataPtr d_col = MakeData(m_col);
  mj_forward(m_col.get(), d_col.get());
  // Even though radius=0.0005 is far smaller than the gaps, the box
  // cross-section has half-sizes 0.02 along y and 0.005 along z, so ball (at
  // y=0.022, r=0.005) penetrates by 0.02 + 0.005 - 0.022 = 0.003 and block (at
  // z=0.059, sz=0.005) penetrates by 0.005 + 0.005 - (0.059 - 0.05) = 0.001.
  int n_ball = 0, n_block = 0;
  int id_ball = mj_name2id(m_col.get(), mjOBJ_GEOM, "ball");
  int id_block = mj_name2id(m_col.get(), mjOBJ_GEOM, "block");
  for (int i = 0; i < d_col->ncon; i++) {
    if (d_col->contact[i].geom[0] == id_ball ||
        d_col->contact[i].geom[1] == id_ball) {
      n_ball++;
      EXPECT_NEAR(d_col->contact[i].dist, -0.003, 1e-6);
    }
    if (d_col->contact[i].geom[0] == id_block ||
        d_col->contact[i].geom[1] == id_block) {
      n_block++;
      EXPECT_NEAR(d_col->contact[i].dist, -0.001, 1e-6);
    }
  }
  EXPECT_GT(n_ball, 0);
  EXPECT_GT(n_block, 0);

  // Verify plane-box collision and finite-difference contact Jacobian wrt
  // both vertex translations and per-edge twist DOFs on a tilted ribbon
  static constexpr char kPlaneTwistBoxXml[] = R"(
  <mujoco>
    <option gravity="0 0 -9.81" timestep="0.001" cone="pyramidal"/>
    <worldbody>
      <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.1" condim="1" friction="0 0 0"/>
      <flexcomp name="rod" type="grid" count="4 1 1" spacing="0.1 0.1 0.1"
                pos="0 0 0.012" radius="0.004" dim="1" mass="0.1">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none" condim="1"/>
        <elasticity young="1e6" shear="1e5" damping="0.05"
                    section="box" sectionsize="0.02 0.004"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m_pt =
      LoadModelFromString(kPlaneTwistBoxXml, error, sizeof(error));
  ASSERT_THAT(m_pt.get(), NotNull()) << error;
  MjDataPtr d_pt = MakeData(m_pt);
  int j_twist1 = mj_name2id(m_pt.get(), mjOBJ_JOINT, "rod_twist_1");
  int j_twist2 = mj_name2id(m_pt.get(), mjOBJ_JOINT, "rod_twist");
  ASSERT_GE(j_twist1, 0);
  ASSERT_GE(j_twist2, 0);
  // Tilt edges 1 and 2 by 0.5 rad (~28.6 deg) and bend slightly out of plane
  d_pt->qpos[1] = 0.003;
  d_pt->qpos[2] = -0.002;
  d_pt->qpos[4] = -0.002;
  d_pt->qpos[5] = 0.001;
  d_pt->qpos[m_pt->jnt_qposadr[j_twist1]] = 0.5;
  d_pt->qpos[m_pt->jnt_qposadr[j_twist2]] = 0.5;
  mj_forward(m_pt.get(), d_pt.get());
  FreezeRodState(m_pt.get(), d_pt.get());
  mj_forward(m_pt.get(), d_pt.get());
  ASSERT_GT(d_pt->ncon, 0);

  // Finite-difference check on contact normal Jacobian J_n = d(dist)/dq
  // Note: for frictionless pyramidal/elliptic with 1 normal row per contact,
  // efc_J for each contact is the normal Jacobian.
  // (the frictional rows: RodContactJacobianMaterialVelocity)
  const mjtNum fd_eps = MjEps(1e-7, 1e-4);
  for (int c = 0; c < d_pt->ncon; c++) {
    int efc_adr = d_pt->contact[c].efc_address;
    ASSERT_GE(efc_adr, 0);
    mjtNum pos0[3];
    mju_copy3(pos0, d_pt->contact[c].pos);
    std::vector<mjtNum> j_ana(m_pt->nv), j_fd(m_pt->nv);
    mjtNum jscale = 0;
    for (int j = 0; j < m_pt->nv; j++) {
      mjtNum J_ana = 0;
      if (mj_isSparse(m_pt.get())) {
        int rownnz = d_pt->efc_J_rownnz[efc_adr];
        int rowadr = d_pt->efc_J_rowadr[efc_adr];
        for (int k = 0; k < rownnz; k++) {
          if (d_pt->efc_J_colind[rowadr + k] == j) {
            J_ana = d_pt->efc_J[rowadr + k];
            break;
          }
        }
      } else {
        J_ana = d_pt->efc_J[efc_adr * m_pt->nv + j];
      }
      MjDataPtr d_plus = MakeData(m_pt);
      mju_copy(d_plus->qpos, d_pt->qpos, m_pt->nq);
      mju_copy(d_plus->plastic, d_pt->plastic, m_pt->nplastic);
      d_plus->qpos[j] += fd_eps;
      mj_forward(m_pt.get(), d_plus.get());
      // Find matching contact by vertex/element and closest contact position
      int best_c = -1;
      mjtNum best_d = 1e10;
      for (int c2 = 0; c2 < d_plus->ncon; c2++) {
        if (d_plus->contact[c2].elem[1] == d_pt->contact[c].elem[1] &&
            d_plus->contact[c2].vert[1] == d_pt->contact[c].vert[1]) {
          mjtNum diff[3];
          mju_sub3(diff, d_plus->contact[c2].pos, pos0);
          mjtNum d2 = mju_norm3(diff);
          if (d2 < best_d) {
            best_d = d2;
            best_c = c2;
          }
        }
      }
      ASSERT_GE(best_c, 0);
      j_ana[j] = J_ana;
      j_fd[j] = (d_plus->contact[best_c].dist - d_pt->contact[c].dist) / fd_eps;
      jscale = std::max(jscale, std::abs(j_fd[j]));
    }
    ASSERT_GT(jscale, 0);
    for (int j = 0; j < m_pt->nv; j++) {
      EXPECT_NEAR(j_ana[j], j_fd[j], MjTol(1e-3, 5e-2) * jscale)
          << "contact " << c << " dof " << j;
    }
  }
  // Verify non-zero twist torque from the floor contact on the tilted edge
  int dof_twist2 = m_pt->jnt_dofadr[j_twist2];
  EXPECT_LT(d_pt->qfrc_constraint[dof_twist2], -1e-4);
}

// contact rows of 1D DER rods (normal, frictional, torsional and rolling): the
// velocity efc_J * qvel of each row matches that of the rod's material point at
// the contact (and the angular velocity of its cross-section), differentiated
// through the vertices, the twist angles and the frozen director state; for
// circular and box sections and for a rod with arbitrary element order
TEST_F(ElasticityTest, RodContactJacobianMaterialVelocity) {
  static constexpr char kCircleShuffled[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" solver="CG" cone="elliptic"/>
    <worldbody>
      <geom type="plane" size="1 1 0.1" condim="6" friction="0.8 0.02 0.01"/>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="0.1"
                element="2 3  0 1  3 4  1 2"
                point="0 0 0.01  0.1 0 0.01  0.2 0 0.01  0.3 0 0.01  0.4 0 0.01">
        <contact selfcollide="none" condim="6" margin="0.003"/>
        <elasticity young="1e7" shear="4e6"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  static constexpr char kBox[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" solver="CG" cone="elliptic"/>
    <worldbody>
      <geom type="plane" size="1 1 0.1" condim="6" friction="0.8 0.02 0.01"/>
      <flexcomp name="rod" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                pos="0 0 0.003" radius="0.003" dim="1" mass="0.1">
        <contact selfcollide="none" condim="6" margin="0.003"/>
        <elasticity young="1e7" shear="4e6" section="box"
                    sectionsize="0.02 0.003"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  static constexpr char kCircleTiltedTip[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" solver="CG" cone="elliptic"/>
    <worldbody>
      <geom type="plane" size="1 1 0.1" condim="6" friction="0.8 0.02 0.01"/>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="0.1"
                element="0 1  1 2  2 3  3 4"
                point="0 0 0.01  0.07 0 0.08  0.14 0 0.15  0.21 0 0.22  0.28 0 0.29">
        <contact selfcollide="none" condim="6" margin="0.003"/>
        <elasticity young="1e7" shear="4e6"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  for (const char* xml : {kCircleShuffled, kBox, kCircleTiltedTip}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    const bool is_box = mj_rodBoxSize(m.get(), 0, nullptr);
    SCOPED_TRACE(xml == kCircleTiltedTip ? "tilted_tip"
                                         : (is_box ? "box" : "circle"));
    MjDataPtr d = MakeData(m);
    for (int j = 0; j < m->njnt; j++) {
      int qadr = m->jnt_qposadr[j];
      int dadr = m->jnt_dofadr[j];
      bool twist = m->jnt_type[j] == mjJNT_HINGE;
      bool vertical = !twist && m->jnt_axis[3 * j + 2] > 0.5;
      if (xml != kCircleTiltedTip || twist) {
        d->qpos[qadr] =
            twist ? 0.3 * std::sin(1.1 * j)
                  : (vertical ? 0.0004 : 0.004) * std::sin(1.7 * j + 0.3);
      }
      d->qvel[dadr] = std::cos(2.3 * j + 0.1);
    }
    mj_forward(m.get(), d.get());
    FreezeRodState(m.get(), d.get());
    mj_forward(m.get(), d.get());
    ASSERT_GT(d->ncon, 0);
    std::vector<mjtNum> jv(d->nefc);
    mj_mulJacVec(m.get(), d.get(), jv.data(), d->qvel);

    // material frames and points of the rod at q -/+ dt qvel, state frozen
    RodChainFrames fr;
    RodBendingEnergy(m.get(), d.get(), 0, &fr);
    ASSERT_FALSE(fr.v_chain.empty());
    const mjtNum dt = MjEps(1e-6, 1e-3);
    auto frame_at = [&](mjtNum sgn, RodChainFrames* out,
                        std::vector<mjtNum>* xpos) {
      MjDataPtr d2 = MakeData(m);
      mju_copy(d2->qpos, d->qpos, m->nq);
      mju_copy(d2->plastic, d->plastic, m->nplastic);
      mj_integratePos(m.get(), d2->qpos, d->qvel, sgn * dt);
      mj_fwdPosition(m.get(), d2.get());
      RodBendingEnergy(m.get(), d2.get(), 0, out);
      xpos->assign(d2->flexvert_xpos, d2->flexvert_xpos + 3 * m->nflexvert);
    };
    RodChainFrames fr_p, fr_m;
    std::vector<mjtNum> x_p, x_m;
    frame_at(1, &fr_p, &x_p);
    frame_at(-1, &fr_m, &x_m);
    auto material = [](const RodChainFrames& f, int k, mjtNum m1[3],
                       mjtNum m2[3]) {
      mjtNum c = mju_cos(f.theta[k]), s = mju_sin(f.theta[k]);
      for (int i = 0; i < 3; i++) {
        m1[i] = c * f.u[3 * k + i] + s * f.v[3 * k + i];
        m2[i] = -s * f.u[3 * k + i] + c * f.v[3 * k + i];
      }
    };

    mjtNum scale = 0, max_err = 0;
    int nrod = 0;
    for (int c = 0; c < d->ncon; c++) {
      const mjContact& con = d->contact[c];
      int side = (con.flex[1] == 0) ? 1 : 0;
      if (con.flex[side] != 0 || con.geom[1 - side] < 0) continue;
      nrod++;
      int elem = con.elem[side] >= 0
                     ? con.elem[side]
                     : mj_rodVertElem(m.get(), 0, con.vert[side]);
      ASSERT_GE(elem, 0);
      const int* ev = m->flex_elem + m->flex_elemdataadr[0] + 2 * elem;
      int k = -1;
      for (int e = 0; e + 1 < static_cast<int>(fr.v_chain.size()); e++) {
        if ((fr.v_chain[e] == ev[0] && fr.v_chain[e + 1] == ev[1]) ||
            (fr.v_chain[e] == ev[1] && fr.v_chain[e + 1] == ev[0])) {
          k = e;
        }
      }
      ASSERT_GE(k, 0);
      if (con.vert[side] >= 0) {
        EXPECT_TRUE(con.vert[side] == ev[0] || con.vert[side] == ev[1]);
      }
      const int a0 = fr.v_chain[k], a1 = fr.v_chain[k + 1];
      const mjtNum* x0 = d->flexvert_xpos + 3 * a0;
      const mjtNum* x1 = d->flexvert_xpos + 3 * a1;
      const mjtNum* t = fr.t.data() + 3 * k;
      mjtNum le = mju_dist3(x0, x1);

      // surface point on the rod (plane contacts lie midway between surfaces)
      mjtNum p[3];
      mju_addScl3(p, con.pos, con.frame, 0.5 * (side ? 1 : -1) * con.dist);
      mjtNum w;
      if (con.vert[side] >= 0) {
        w = (con.vert[side] == a1) ? 1 : 0;
      } else {
        mjtNum ap[3];
        mju_sub3(ap, p, x0);
        w = mju_clip(mju_dot3(ap, t) / le, 0, 1);
      }
      mjtNum r[3], m1[3], m2[3];
      for (int i = 0; i < 3; i++) r[i] = p[i] - (1 - w) * x0[i] - w * x1[i];
      material(fr, k, m1, m2);
      const mjtNum ra = mju_dot3(r, m1), rb = mju_dot3(r, m2),
                   rt = mju_dot3(r, t);

      // velocity of the material point and angular velocity of its section
      mjtNum P[2][3], R[2][9];
      for (int n = 0; n < 2; n++) {
        const RodChainFrames& f = n ? fr_p : fr_m;
        const std::vector<mjtNum>& x = n ? x_p : x_m;
        const mjtNum* y0 = x.data() + 3 * (m->flex_vertadr[0] + a0);
        const mjtNum* y1 = x.data() + 3 * (m->flex_vertadr[0] + a1);
        mjtNum n1[3], n2[3];
        material(f, k, n1, n2);
        for (int i = 0; i < 3; i++) {
          P[n][i] = (1 - w) * y0[i] + w * y1[i] + ra * n1[i] + rb * n2[i] +
                    rt * f.t[3 * k + i];
          R[n][3 * i + 0] = n1[i];
          R[n][3 * i + 1] = n2[i];
          R[n][3 * i + 2] = f.t[3 * k + i];
        }
      }
      mjtNum vel[3], q0[4], q1[4], q0inv[4], dq[4], omega[3];
      for (int i = 0; i < 3; i++) vel[i] = (P[1][i] - P[0][i]) / (2 * dt);
      mju_mat2Quat(q0, R[0]);
      mju_mat2Quat(q1, R[1]);
      mju_negQuat(q0inv, q0);
      mju_mulQuat(dq, q1, q0inv);
      mju_quat2Vel(omega, dq, 2 * dt);

      mjtNum sgn = side ? 1 : -1;
      for (int row = 0; row < con.dim; row++) {
        const mjtNum* axis = con.frame + 3 * (row % 3);
        mjtNum expected = sgn * mju_dot3(axis, row < 3 ? vel : omega);
        mjtNum got = jv[con.efc_address + row];
        scale = std::max(scale, std::abs(expected));
        max_err = std::max(max_err, std::abs(got - expected));
        EXPECT_NEAR(got, expected, MjTol(1e-5, 5e-2) * (1 + std::abs(expected)))
            << "contact " << c << " row " << row;
      }
      if (xml == kCircleTiltedTip && con.vert[side] == 0) {
        // spherical cap at vertex 0 on a tilted edge: dvec is along the contact
        // normal, so n . (omega x dvec) = 0 and the normal row has unit weight
        // on vertex 0 and zero on vertex 1
        std::vector<mjtNum> jrow(m->nv, 0);
        for (int dof = 0; dof < m->nv; dof++) {
          std::vector<mjtNum> e_dof(m->nv, 0), out(d->nefc, 0);
          e_dof[dof] = 1;
          mj_mulJacVec(m.get(), d.get(), out.data(), e_dof.data());
          jrow[dof] = out[con.efc_address];
        }
        int d0 = m->body_dofadr[m->flex_vertbodyid[m->flex_vertadr[0] + 0]];
        int d1 = m->body_dofadr[m->flex_vertbodyid[m->flex_vertadr[0] + 1]];
        for (int i = 0; i < 3; i++) {
          EXPECT_NEAR(jrow[d0 + i], sgn * con.frame[i], MjTol(1e-12, 1e-6));
          EXPECT_NEAR(jrow[d1 + i], 0, MjTol(1e-12, 1e-6));
        }
      }
    }
    EXPECT_GT(nrod, xml == kCircleTiltedTip ? 0 : 2);
    EXPECT_GT(scale, 0.1);
    EXPECT_LT(max_err, MjTol(1e-6, 1e-2) * scale);
  }
}

// a constant torque on the twist hinge of a straight rod twists it by about
// ctrl / (GJ / L), and the rod holds that twist without overshoot
TEST_F(ElasticityTest, RodTwistMotorSteadyTwist) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="21 1 1" spacing="0.05 0.05 0.05"
                radius="0.01" dim="1" mass="0.1">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" damping="0.05"/>
      </flexcomp>
    </worldbody>
    <equality>
      <connect body1="rod_20" anchor="0 0 0"/>
    </equality>
    <actuator>
      <motor joint="rod_twist"/>
    </actuator>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  int twist = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist");
  ASSERT_GE(twist, 0);
  int dof = m->jnt_dofadr[twist];
  int qadr = m->jnt_qposadr[twist];
  int f = mj_name2id(m.get(), mjOBJ_FLEX, "rod");

  // each edge twist hinge has the per-edge polar inertia; Rayleigh torsional
  // damping acts relatively in qfrc_damper rather than m->dof_damping
  EXPECT_NEAR(m->body_inertia[3 * m->jnt_bodyid[twist]],
              (0.1 / 20) * 0.01 * 0.01 / 2, 1e-12);
  EXPECT_EQ(m->dof_damping[dof], 0);
  EXPECT_GE(m->flex_plasticadr[f], 0);
  {
    MjDataPtr d_damp = MakeData(m);
    d_damp->qvel[dof] = 1.0;
    mj_forward(m.get(), d_damp.get());
    int twist_prev = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist_18");
    ASSERT_GE(twist_prev, 0);
    EXPECT_LT(d_damp->qfrc_damper[dof], 0);
    EXPECT_NEAR(d_damp->qfrc_damper[m->jnt_dofadr[twist_prev]],
                -d_damp->qfrc_damper[dof], MjTol(1e-12, 1e-6));
  }

  const mjtNum J = mjPI * 1e-8 / 2, L = 1.0, ctrl = 0.05;
  for (mjtNum sign : {1, -1}) {
    MjDataPtr d = MakeData(m);
    d->ctrl[0] = sign * ctrl;
    mjtNum fmax = 0;
    for (int i = 0; i < 2000; i++) {
      mj_step(m.get(), d.get());
      fmax = std::max(fmax, std::abs(d->qfrc_spring[dof]));
      ASSERT_TRUE(std::isfinite(d->qpos[qadr]));
    }
    EXPECT_NEAR(d->qpos[qadr], sign * ctrl * L / (4e6 * J),
                0.1 * ctrl * L / (4e6 * J));
    EXPECT_NEAR(d->qfrc_spring[dof], -sign * ctrl, 1e-3 * ctrl);
    EXPECT_LT(std::abs(d->qvel[dof]), 1e-3);
    EXPECT_LT(fmax, 1.05 * ctrl);
  }
}

// uniform twist of more than pi per hinge on a coarse straight rod: the torsion
// energy keeps growing as kt * alpha^2 per hinge and the tail torque does not
// flip sign at alpha = pi (the joint difference is not wrapped)
TEST_F(ElasticityTest, RodTwistBeyondPiEachHinge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="0.1">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
  const int nhinge = static_cast<int>(b[0]);
  ASSERT_EQ(nhinge, 4);
  mjtNum kt_sum = 0;
  for (int h = 0; h < nhinge; h++) {
    kt_sum += b[1 + kRodHingeSize * h + 8];
  }
  ASSERT_GT(kt_sum, 0);
  int tail = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist");
  ASSERT_GE(tail, 0);
  const mjtNum kt_tail = b[1 + kRodHingeSize * (nhinge - 1) + 8];

  // edges 1..4: theta_e = (e - 0) * alpha, edge 0 clamped at 0
  mjtNum e_prev = -1;
  for (int i = 1; i <= 16; i++) {
    const mjtNum alpha = i * (2 * mjPI / 8);  // up to 4 pi per hinge
    mj_resetData(m.get(), d.get());
    for (int e = 1; e <= nhinge; e++) {
      int jnt = mj_rodEdgeTwistJnt(m.get(), 0, e, nullptr);
      ASSERT_GE(jnt, 0);
      d->qpos[m->jnt_qposadr[jnt]] = e * alpha;
    }
    mj_forward(m.get(), d.get());
    mjtNum energy = RodBendingEnergy(m.get(), d.get(), 0);
    EXPECT_NEAR(energy, kt_sum * alpha * alpha,
                MjTol(1e-9, 1e-4) * kt_sum * alpha * alpha);
    EXPECT_GT(energy, e_prev) << "alpha " << alpha;
    e_prev = energy;
    EXPECT_NEAR(d->qfrc_spring[m->jnt_dofadr[tail]], -2 * kt_tail * alpha,
                MjTol(1e-9, 1e-4) * 2 * kt_tail * alpha);
  }

  // dynamic: a position servo turns the tail edge through 3 turns while the
  // interior twist joints relax; the reference twist state is stepped along,
  // and the energy and the tail torque grow continuously past pi per hinge
  static constexpr char xml_servo[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="0.002"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="0.1">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" damping="0.05"/>
      </flexcomp>
    </worldbody>
    <actuator>
      <position joint="rod_twist" kp="1" kv="0.02"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr ms = LoadModelFromString(xml_servo, error, sizeof(error));
  ASSERT_THAT(ms.get(), NotNull()) << error;
  MjDataPtr ds = MakeData(ms);
  int tail_s = mj_name2id(ms.get(), mjOBJ_JOINT, "rod_twist");
  ASSERT_GE(tail_s, 0);
  const int dof_s = ms->jnt_dofadr[tail_s];
  const int nstep = 3000;
  mjtNum e_last = 0, f_last = 0, df_max = 0, f_max = 0;
  for (int i = 0; i < nstep; i++) {
    ds->ctrl[0] = 6 * mjPI * std::min<mjtNum>(1, (i + 1.0) / (0.8 * nstep));
    mj_step(ms.get(), ds.get());
    mjtNum energy = RodBendingEnergy(ms.get(), ds.get(), 0);
    mjtNum f = ds->qfrc_spring[dof_s];
    ASSERT_TRUE(std::isfinite(energy));
    if (i > 0) {
      // no 2 pi jumps: per-step changes stay small against the final values
      EXPECT_LT(std::abs(energy - e_last), 0.01 * kt_sum * 2 * mjPI * mjPI)
          << "step " << i;
      df_max = std::max(df_max, std::abs(f - f_last));
    }
    EXPECT_LE(f, MjTol(1e-9, 1e-5)) << "tail torque flipped at step " << i;
    f_max = std::max(f_max, std::abs(f));
    e_last = energy;
    f_last = f;
  }
  // final twist: about 1.5 pi per hinge, torque near -2 kt (1.5 pi)
  const mjtNum alpha_end = ds->qpos[ms->jnt_qposadr[tail_s]] / nhinge;
  EXPECT_GT(alpha_end, 1.2 * mjPI);
  EXPECT_NEAR(ds->qfrc_spring[dof_s], -2 * kt_tail * alpha_end,
              0.05 * 2 * kt_tail * alpha_end);
  EXPECT_LT(df_max, 0.05 * f_max);
}

// without director state (mj_resetData, keyframes without plastic) a rod starts
// twist-free from edge 0's current frame: a clamped root rolled about the rod
// axis carries no spurious reference twist, and a free rod flipped end to end
// (antipodal to its qpos0 tangents) is still handled and initializes its state
TEST_F(ElasticityTest, RodNoStateStartsFromCurrentFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" solver="CG"/>
    <worldbody>
      <body name="root">
        <joint name="roll" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.01" mass="0.01"/>
        <flexcomp name="rod" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="1" mass="0.1">
          <pin id="0 1"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" shear="4e6"/>
        </flexcomp>
      </body>
      <flexcomp name="free" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                pos="0 1 0" radius="0.004" dim="1" mass="0.1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" section="box"
                    sectionsize="0.01 0.004"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  int roll = mj_name2id(m.get(), mjOBJ_JOINT, "roll");
  int rod = mj_name2id(m.get(), mjOBJ_FLEX, "rod");
  int flipped = mj_name2id(m.get(), mjOBJ_FLEX, "free");
  ASSERT_GE(roll, 0);

  // rolled root: zero energy and zero twist torques
  d->qpos[m->jnt_qposadr[roll]] = 1.3;
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), rod), 0, MjTol(1e-12, 1e-6));
  mjtNum energy = -1;
  ASSERT_EQ(mj_rodChain(m.get(), d.get(), rod, &energy, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr),
            1);
  EXPECT_NEAR(energy, 0, MjTol(1e-12, 1e-6));
  for (int e = 1; e < 4; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), rod, e);
    ASSERT_GE(jnt, 0);
    int dof = m->jnt_dofadr[jnt];
    EXPECT_NEAR(d->qfrc_spring[dof], 0, MjTol(1e-9, 1e-4)) << "edge " << e;
  }

  // free rod flipped end to end: every tangent is antipodal to its reference
  mj_resetData(m.get(), d.get());
  for (int i = 0; i < m->flex_vertnum[flipped]; i++) {
    int body = m->flex_vertbodyid[m->flex_vertadr[flipped] + i];
    int qadr = m->jnt_qposadr[m->body_jntadr[body]];
    d->qpos[qadr] = -2 * (0.1 * i - 0.2);
  }
  mj_forward(m.get(), d.get());
  ASSERT_EQ(mj_rodChain(m.get(), d.get(), flipped, &energy, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr),
            1);
  EXPECT_NEAR(energy, 0, MjTol(1e-12, 1e-6));
  mjtNum mat[9], pos[3], size[3];
  EXPECT_EQ(mj_rodBoxElem(m.get(), d.get(), flipped, 2, pos, mat, size), 1);
  mj_step(m.get(), d.get());
  const mjtNum* state = d->plastic + m->flex_plasticadr[flipped];
  for (int e = 0; e < 4; e++) {
    EXPECT_NEAR(mju_norm3(state + 6 * e), 1, MjTol(1e-12, 1e-6));
    EXPECT_NEAR(mju_norm3(state + 6 * e + 3), 1, MjTol(1e-12, 1e-6));
  }
}

// a rod whose last two vertices are on one body is clamped at the tail like at
// the root: no twist joint there, the last edge's frame turns with the body,
// and the body receives the rod's torsional reaction (spring and damper)
TEST_F(ElasticityTest, RodTailClampTransmitsTorsion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" solver="CG"/>
    <worldbody>
      <body name="v2" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="0.02" diaginertia="1e-6 1e-6 1e-6"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
      <body name="tail" pos="0.35 0 0">
        <joint name="roll" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="0.02" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
      <body name="t1">
        <joint name="rod_twist_1" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>
      </body>
      <body name="t2">
        <joint name="rod_twist_2" type="hinge" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-8 1e-8 1e-8"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="rod" dim="1" radius="0.01" body="world world v2 tail tail"
            vertex="0 0 0  0.1 0 0  0 0 0  -0.05 0 0  0.05 0 0"
            element="0 1  1 2  2 3  3 4">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" damping="0.01"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_TRUE(mj_rodHasTwist(m.get(), 0));
  int roll = mj_name2id(m.get(), mjOBJ_JOINT, "roll");
  int tw1 = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist_1");
  int tw2 = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist_2");
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), 0, 0), -1);
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), 0, 1), tw1);
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), 0, 2), tw2);
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), 0, 3), -1);
  const int roll_dof = m->jnt_dofadr[roll], tw2_dof = m->jnt_dofadr[tw2];

  // turning the tail body loads the rod in torsion: restoring torque on it,
  // consistent with the energy, and the opposite torque on the last free edge
  MjDataPtr d = MakeData(m);
  d->qpos[m->jnt_qposadr[roll]] = 0.4;
  mj_forward(m.get(), d.get());
  FreezeRodState(m.get(), d.get());
  mj_forward(m.get(), d.get());
  EXPECT_LT(d->qfrc_spring[roll_dof], 0);
  EXPECT_GT(d->qfrc_spring[tw2_dof], 0);
  const mjtNum eps = MjEps(1e-6, 1e-3);
  for (int jnt : {roll, tw1, tw2}) {
    int qadr = m->jnt_qposadr[jnt];
    mjtNum q = d->qpos[qadr];
    d->qpos[qadr] = q + eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum ep = RodBendingEnergy(m.get(), d.get(), 0);
    d->qpos[qadr] = q - eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum em = RodBendingEnergy(m.get(), d.get(), 0);
    d->qpos[qadr] = q;
    EXPECT_NEAR(d->qfrc_spring[m->jnt_dofadr[jnt]], -(ep - em) / (2 * eps),
                MjTol(1e-6, 1e-2) * std::abs(d->qfrc_spring[roll_dof]))
        << m->names + m->name_jntadr[jnt];
  }

  // spinning the tail body: relative-rate damping between it and the last edge
  mj_resetData(m.get(), d.get());
  d->qvel[roll_dof] = 1;
  mj_forward(m.get(), d.get());
  EXPECT_LT(d->qfrc_damper[roll_dof], 0);
  EXPECT_NEAR(d->qfrc_damper[tw2_dof], -d->qfrc_damper[roll_dof],
              MjTol(1e-12, 1e-6) * std::abs(d->qfrc_damper[roll_dof]));
}

// a box ribbon clamped at both ends (pinned first and last edges) under the
// discrete integrator: stable under gravity, and without gravity a torque on a
// middle twist joint is resisted by both clamps, halving the twist of a ribbon
// with a free tail
TEST_F(ElasticityTest, ClampedBoxRodLastEdgeDiscreteStability) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep="0.002" tolerance="1e-10"
            iterations="200" gravity="0 0 %s"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="8 1 1" spacing="0.05 0.05 0.05"
                radius="0.0015" dim="1" mass="0.05">
        <pin id="%s"/>
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" poisson="0.25" damping="0.05" section="box"
                    sectionsize="0.01 0.0015"/>
      </flexcomp>
    </worldbody>
    <actuator>
      <motor joint="rod_twist_3"/>
    </actuator>
  </mujoco>
  )";
  mjtNum twist[2] = {0, 0};
  for (int run = 0; run < 3; run++) {
    int clamped = run > 0, gravity = run == 2;
    char xml[2048], error[1024] = {0};
    std::snprintf(xml, sizeof(xml), xml_template, gravity ? "-9.81" : "0",
                  clamped ? "0 1 6 7" : "0 1 7");
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    EXPECT_EQ(mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist") >= 0, !clamped);
    int jnt = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist_3");
    ASSERT_GE(jnt, 0);
    MjDataPtr d = MakeData(m);
    d->ctrl[0] = 1e-3;
    for (int i = 0; i < 1500; i++) {
      mj_step(m.get(), d.get());
      ASSERT_EQ(d->warning[mjWARN_BADQACC].number, 0) << "step " << i;
    }
    for (int i = 0; i < m->nv; i++) {
      ASSERT_TRUE(std::isfinite(d->qvel[i]));
      EXPECT_LT(std::abs(d->qvel[i]), 1e-2) << "dof " << i;
    }
    if (!gravity) {
      twist[clamped] = d->qpos[m->jnt_qposadr[jnt]];
    }
  }
  EXPECT_GT(twist[0], 0);
  EXPECT_NEAR(twist[1] / twist[0], 0.5, 0.1);
}

// the time-parallel state (frames and unwrapped twists) is part of
// mjSTATE_PLASTIC and of keyframes: restoring it reproduces the trajectory
TEST_F(ElasticityTest, RodStatePlasticRoundTrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep="0.002"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.05 0.05 0.05"
                radius="0.002" dim="1" mass="0.05">
        <pin id="0 1"/>
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e6" shear="4e5" damping="0.01" section="box"
                    sectionsize="0.01 0.002"/>
      </flexcomp>
    </worldbody>
    <actuator>
      <motor joint="rod_twist"/>
    </actuator>
    <keyframe>
      <key/>
    </keyframe>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->nplastic, 6 * 5 + 4);
  EXPECT_EQ(mj_stateSize(m.get(), mjSTATE_PLASTIC), m->nplastic);
  MjDataPtr d = MakeData(m);
  d->ctrl[0] = 2e-3;
  for (int i = 0; i < 300; i++) mj_step(m.get(), d.get());
  EXPECT_GT(mju_norm(d->plastic, m->nplastic), 1);

  const int spec = mjSTATE_INTEGRATION;
  std::vector<mjtNum> state(mj_stateSize(m.get(), spec));
  mj_getState(m.get(), d.get(), state.data(), spec);
  std::vector<mjtNum> saved_plastic(d->plastic, d->plastic + m->nplastic);
  mj_setKeyframe(m.get(), d.get(), 0);
  auto run = [&](mjData* dd) {
    for (int i = 0; i < 100; i++) mj_step(m.get(), dd);
    return std::vector<mjtNum>(dd->qpos, dd->qpos + m->nq);
  };
  std::vector<mjtNum> qpos_a = run(d.get());

  MjDataPtr d_state = MakeData(m);
  mj_setState(m.get(), d_state.get(), state.data(), spec);
  std::vector<mjtNum> qpos_b = run(d_state.get());

  MjDataPtr d_key = MakeData(m);
  mj_resetDataKeyframe(m.get(), d_key.get(), 0);
  d_key->ctrl[0] = 2e-3;
  for (int i = 0; i < m->nplastic; i++) {
    EXPECT_EQ(d_key->plastic[i], saved_plastic[i]);
  }
  std::vector<mjtNum> qpos_c = run(d_key.get());
  for (int i = 0; i < m->nq; i++) {
    EXPECT_EQ(qpos_b[i], qpos_a[i]) << "state, qpos " << i;
    // keyframes do not store the solver warmstart
    EXPECT_NEAR(qpos_c[i], qpos_a[i], MjTol(1e-6, 1e-3))
        << "keyframe, qpos " << i;
  }
}

// two rods in one model: the second rod's first-edge joint follows the
// "edge 1 - 1" convention within its own joints, and the rods do not interact
TEST_F(ElasticityTest, RodTwistTwoRods) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="a" type="grid" count="4 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="0.1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6"/>
      </flexcomp>
      <flexcomp name="b" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                pos="0 0.5 0" radius="0.004" dim="1" mass="0.1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" section="box"
                    sectionsize="0.01 0.004"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  int fa = mj_name2id(m.get(), mjOBJ_FLEX, "a");
  int fb = mj_name2id(m.get(), mjOBJ_FLEX, "b");
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), fa, 0),
            mj_name2id(m.get(), mjOBJ_JOINT, "a_twist_0"));
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), fb, 0),
            mj_name2id(m.get(), mjOBJ_JOINT, "b_twist_0"));
  EXPECT_EQ(mj_rodChainTwistJnt(m.get(), fb, 3),
            mj_name2id(m.get(), mjOBJ_JOINT, "b_twist"));

  // twist only rod b: rod a feels nothing, rod b's forces follow its energy
  MjDataPtr d = MakeData(m);
  for (int e = 0; e < 4; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), fb, e);
    d->qpos[m->jnt_qposadr[jnt]] = 0.2 * (e + 1) * (e % 2 ? -1 : 1);
  }
  mj_forward(m.get(), d.get());
  for (int e = 0; e < 3; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), fa, e);
    EXPECT_EQ(d->qfrc_spring[m->jnt_dofadr[jnt]], 0) << "a edge " << e;
  }
  const mjtNum eps = MjEps(1e-6, 1e-3);
  mjtNum fmax = 0;
  for (int e = 0; e < 4; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), fb, e);
    fmax = std::max(fmax, std::abs(d->qfrc_spring[m->jnt_dofadr[jnt]]));
  }
  ASSERT_GT(fmax, 0);
  for (int e = 0; e < 4; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), fb, e);
    int qadr = m->jnt_qposadr[jnt];
    mjtNum q = d->qpos[qadr];
    d->qpos[qadr] = q + eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum ep = RodBendingEnergy(m.get(), d.get(), fb);
    d->qpos[qadr] = q - eps;
    mj_fwdPosition(m.get(), d.get());
    mjtNum em = RodBendingEnergy(m.get(), d.get(), fb);
    d->qpos[qadr] = q;
    EXPECT_NEAR(d->qfrc_spring[m->jnt_dofadr[jnt]], -(ep - em) / (2 * eps),
                MjTol(1e-6, 1e-2) * fmax)
        << "b edge " << e;
  }
}

// Euler integrates the twist joints explicitly: with a small timestep a
// motor-twisted clamped rod settles at the static twist
TEST_F(ElasticityTest, RodTwistEuler) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="2e-4"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="0.1">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" damping="0.002"/>
      </flexcomp>
    </worldbody>
    <actuator>
      <motor joint="rod_twist"/>
    </actuator>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  const mjtNum* b = m->flex_bending + m->flex_bendingadr[0];
  mjtNum compliance = 0;  // series compliance of the 4 unclamped hinges
  for (int h = 0; h < 4; h++)
    compliance += 1 / (2 * b[1 + kRodHingeSize * h + 8]);
  int tail = mj_name2id(m.get(), mjOBJ_JOINT, "rod_twist");
  MjDataPtr d = MakeData(m);
  const mjtNum ctrl = 0.01;
  d->ctrl[0] = ctrl;
  for (int i = 0; i < 10000; i++) {
    mj_step(m.get(), d.get());
    ASSERT_TRUE(std::isfinite(d->qpos[m->jnt_qposadr[tail]])) << "step " << i;
  }
  EXPECT_NEAR(d->qpos[m->jnt_qposadr[tail]], ctrl * compliance,
              0.05 * ctrl * compliance);
}

// a resting rod falls asleep with its twist trees, which share its island (all
// the trees of the model are the rod's)
TEST_F(ElasticityTest, RodTwistSleep) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" viscosity="2"
            sleep_tolerance="0.1">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.05 0.05 0.05"
                radius="0.01" dim="1" mass="0.05" pos="0 0 0.01">
        <contact selfcollide="none"/>
        <edge equality="true"/>
        <elasticity young="1e6" shear="4e5" damping="0.1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  for (int e = 0; e < 5; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
    ASSERT_GE(jnt, 0);
    d->qvel[m->jnt_dofadr[jnt]] = 1;  // start spinning
  }
  bool shared_island = false;
  for (int i = 0; i < 3000; i++) {
    mj_step(m.get(), d.get());
    if (i == 0) {
      int vtree = m->body_treeid[m->flex_vertbodyid[m->flex_vertadr[0]]];
      int island = d->tree_island[vtree];
      shared_island = island >= 0;
      for (int e = 0; e < 5; e++) {
        int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
        shared_island &=
            d->tree_island[m->body_treeid[m->jnt_bodyid[jnt]]] == island;
      }
    }
    if (d->ntree_awake == 0) break;
  }
  EXPECT_TRUE(shared_island);
  ASSERT_EQ(d->ntree_awake, 0);
  for (int e = 0; e < 5; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
    EXPECT_EQ(d->tree_awake[m->body_treeid[m->jnt_bodyid[jnt]]], 0)
        << "edge " << e;
  }
}

// a vertex of a rod with arbitrary element order maps to the element of the
// chain edge leaving it (the last edge at the far end), not element "vertex"
TEST_F(ElasticityTest, RodVertElemDirectOrder) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.004" mass="0.1"
                element="2 3  0 1  3 4  1 2"
                point="0 0 0  0.1 0 0  0.2 0 0  0.3 0 0  0.4 0 0">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" section="box" sectionsize="0.01 0.004"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_TRUE(mj_rodHasTwist(m.get(), 0));
  // element="2 3  0 1  3 4  1 2": chain edges (0,1), (1,2), (2,3), (3,4) are
  // elements 1, 3, 0, 2, with the far-end vertex 4 also mapping to element 2
  const int expected[5] = {1, 3, 0, 2, 2};
  for (int v = 0; v < 5; v++) {
    EXPECT_EQ(mj_rodVertElem(m.get(), 0, v), expected[v]) << "vertex " << v;
  }
}

// joint stiffness on a rod twist hinge combines additively with the rod's
// curvature-induced twist bending stiffness in efm_ck and stays idempotent
// across velocity-stage skips
TEST_F(ElasticityTest, RodTwistJointStiffnessMetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep="0.002" gravity="0 0 0"/>
    <default>
      <joint type="hinge" stiffness="5"/>
    </default>
    <worldbody>
      <flexcomp name="arc" type="direct" dim="1" radius="0.003" mass="0.1"
                element="0 1  1 2  2 3"
                point="0 0 0  0.1 0 0  0.18 0.06 0  0.22 0.15 0">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" section="box"
                    sectionsize="0.012 0.003"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  mjtNum ktheta[3] = {0};
  ASSERT_EQ(mj_rodChain(m.get(), d.get(), 0, nullptr, nullptr, nullptr, nullptr,
                        nullptr, ktheta, nullptr),
            1);
  const mjtNum h = m->opt.timestep;
  for (int e = 0; e < 3; e++) {
    int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
    ASSERT_GE(jnt, 0);
    EXPECT_EQ(m->jnt_stiffness[jnt], 5);
    int dof = m->jnt_dofadr[jnt];
    EXPECT_GT(ktheta[e], 0) << "edge " << e;
    EXPECT_NEAR(d->efm_ck[dof], h * (5 + ktheta[e]), MjTol(1e-12, 1e-6))
        << "edge " << e;
  }

  // re-running from mjSTAGE_VEL skips effDiagStiff; efm_ck must stay unchanged
  std::vector<mjtNum> ck_before(d->efm_ck, d->efm_ck + m->nv);
  mj_forwardSkip(m.get(), d.get(), mjSTAGE_POS, 0);
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->efm_ck[i], ck_before[i], MjTol(1e-12, 1e-6)) << "dof " << i;
  }
}

// a twisted belt coils: the geometric twist exceeds pi, and the twist torque
// stays continuous and settles at the motor torque
TEST_F(ElasticityTest, RodTwistCoilingBelt) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete">
      <flag gravity="disable"/>
    </option>
    <worldbody>
      <flexcomp name="belt" type="direct" dim="1" pos="-.5 0 .05" radius=".005"
                mass=".05"
                point="0 0 0  .05 0 .05  .1 0 .1  .15 0 .15  .2 0 .2  .25 0 .25
                       .3 0 .3  .35 0 .35  .4 0 .4  .45 0 .45  .5 0 .5
                       .55 0 .45  .6 0 .4  .65 0 .35  .7 0 .3  .75 0 .25
                       .8 0 .2  .85 0 .15  .9 0 .1  .95 0 .05  1 0 0">
        <contact selfcollide="auto"/>
        <edge equality="true"/>
        <elasticity young="1e7" shear="1e8" damping=".05" section="box"
                    sectionsize="0.02 0.0015"/>
        <pin id="0 1"/>
      </flexcomp>
    </worldbody>
    <equality>
      <connect body1="belt_20" anchor="0 0 0"/>
    </equality>
    <actuator>
      <motor joint="belt_twist"/>
    </actuator>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  int twist = mj_name2id(m.get(), mjOBJ_JOINT, "belt_twist");
  int f = mj_name2id(m.get(), mjOBJ_FLEX, "belt");
  ASSERT_GE(twist, 0);
  ASSERT_GE(m->flex_plasticadr[f], 0);
  int dof = m->jnt_dofadr[twist];

  const mjtNum ctrl = 0.2;
  MjDataPtr d = MakeData(m);
  d->ctrl[0] = ctrl;
  mjtNum fmax = 0, fsum = 0;
  int nsteady = 0;
  for (int i = 0; i < 3000; i++) {
    mj_step(m.get(), d.get());
    for (int j = 0; j < m->nq; j++) {
      ASSERT_TRUE(std::isfinite(d->qpos[j])) << "step " << i;
    }
    if (d->time > 1) {
      fmax = std::max(fmax, std::abs(d->qfrc_spring[dof]));
    }
    if (d->time >= 2) {
      fsum += d->qfrc_spring[dof];
      nsteady++;
    }
  }
  EXPECT_EQ(m->nplastic, 6 * 20 + 19);
  for (int e = 0; e < 20; e++) {
    const mjtNum* p = d->plastic + m->flex_plasticadr[f] + 6 * e;
    EXPECT_NEAR(mju_norm3(p), 1.0, 1e-6);
    EXPECT_NEAR(mju_norm3(p + 3), 1.0, 1e-6);
    EXPECT_NEAR(mju_dot3(p, p + 3), 0.0, 1e-6);
  }
  EXPECT_GT(std::abs(d->qpos[m->jnt_qposadr[twist]]), mjPI);
  EXPECT_LT(fmax, 1.5 * ctrl);
  EXPECT_NEAR(d->qfrc_passive[dof], -ctrl, 0.02 * ctrl);
  EXPECT_NEAR(fsum / nsteady, -ctrl, 0.12 * ctrl);
  // the coiled belt still drifts slowly, so the spring torque at one step
  // wanders around the motor torque (in float, from -0.99 to -0.81 times it)
  EXPECT_NEAR(d->qfrc_spring[dof], -ctrl, MjTol(0.12, 0.25) * ctrl);
}

// rigid 1D flexes carry no rod bending, so their reference need not be straight
TEST_F(ElasticityTest, RodBendingSkipsRigid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="ring" type="circle" count="11 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="1" rigid="true">
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->flex_bendingadr[0], -1);
  MjDataPtr d = MakeData(m);
  mj_step(m.get(), d.get());
}

// interpolated 1D flexes carry no rod bending: their vertices have no bodies
TEST_F(ElasticityTest, RodBendingSkipsInterpolated) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="direct" dim="1" radius="0.01" mass="1"
                dof="trilinear" element="0 1  1 2  2 3"
                point="0 0 0  .1 .05 .02  .2 0 .05  .3 .05 .1">
        <contact selfcollide="none"/>
        <elasticity young="1e7"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->flex_bendingadr[0], -1);
  MjDataPtr d = MakeData(m);
  mj_step(m.get(), d.get());
}

// mocap attachments are allowed for undamped rod bending only, since mocap
// vertices have no velocity
TEST_F(ElasticityTest, RodBendingMocapRequiresZeroDamping) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <worldbody>
      <body name="v0" mocap="true"/>
      <body name="v1" pos="0.1 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
      <body name="v2" pos="0.2 0 0">
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
      </body>
    </worldbody>
    <deformable>
      <flex dim="1" body="v0 v1 v2" element="0 1  1 2" radius="0.01">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" damping="%s"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  char xml[2048];
  char error[1024] = {0};
  std::snprintf(xml, sizeof(xml), xml_template, "0");
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_GE(m->flex_bendingadr[0], 0);

  std::snprintf(xml, sizeof(xml), xml_template, "0.01");
  m = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error,
              HasSubstr("flex elasticity does not support mocap attachments"));
}

// 1D rod bending uses the sparse covered-block Cholesky factor (effChol) in the
// effective metric without requiring IPC, including the 2-hop hinge span pairs
// (v_prev, v_next), so (M + K) is factored exactly with zero fill-in and the
// smooth metric solve converges in a single PCG iteration without
// mjWARN_INERTIA
TEST_F(ElasticityTest, RodBendingMetricCholesky) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002" iterations="1"/>
    <worldbody>
      <flexcomp name="rod" type="grid" count="41 1 1" spacing="0.025 0.025 0.025"
                radius="0.005" dim="1" mass="0.1">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <edge equality="true"/>
        <elasticity young="1e9" damping="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->nefmCvert, 41);

  MjDataPtr d = MakeData(m);
  // bend into a smooth curve so all hinges have state-dependent 3-vertex blocks
  for (int i = 0; i < m->nq; i++) {
    d->qpos[i] = 0.005 * mju_sin(1.3 * i + 0.2);
    d->qvel[i] = 0.1 * mju_cos(0.7 * i + 0.4);
  }

  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->warning[mjWARN_INERTIA].number, 0);
  EXPECT_EQ(d->nefmdof, 39);
  ASSERT_GT(d->nefmL, 0);
  EXPECT_EQ(d->efm_L[0], 1);

  // check that (M + B) qacc_smooth == qfrc_smooth + efm_c to machine precision
  // in 1 PCG iteration: the preconditioner sees the same twist chain as the
  // operator (the twist stiffness itself is checked against finite differences
  // in DerivativeTest.FlexRodTwistHessian)
  auto add_metric = [](const mjModel* m_in, mjData* d_in, mjtNum* res) {
    mj_effMulAdd(m_in, d_in, res, d_in->qacc_smooth, /*flg_contact=*/1);
  };
  std::vector<mjtNum> rhs(m->nv), lhs(m->nv);
  mju_add(rhs.data(), d->qfrc_smooth, d->efm_c, m->nv);
  mj_mulM(m.get(), d.get(), lhs.data(), d->qacc_smooth);
  add_metric(m.get(), d.get(), lhs.data());
  mjtNum err = 0, norm = mju_norm(rhs.data(), m->nv);
  ASSERT_GT(norm, 0);
  for (int i = 0; i < m->nv; i++) {
    mjtNum diff = lhs[i] - rhs[i];
    err += diff * diff;
  }
  EXPECT_LT(mju_sqrt(err) / norm, MjTol(1e-11, 1e-4));

  // restore iterations for the constraint solver (edge equalities) and step
  // with constraint islands active
  m->opt.iterations = 100;
  for (int step = 0; step < 10; step++) {
    mj_step(m.get(), d.get());
    EXPECT_EQ(d->warning[mjWARN_INERTIA].number, 0);
    EXPECT_EQ(d->warning[mjWARN_BADQACC].number, 0);
  }
  EXPECT_GT(d->nisland, 0);

  // 3D curved helix with coupled bending (effChol) and tridiagonal twist:
  // still 1 PCG iteration
  static constexpr char kCurvedCholXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002" iterations="1"/>
    <worldbody>
      <flexcomp name="helix" type="direct" dim="1" radius="0.005" mass="0.1"
                element="0 1  1 2  2 3  3 4  4 5  5 6  6 7  7 8  8 9"
                point="0.10 0.00 0.00   0.08 0.06 0.02   0.03 0.095 0.04
                       -0.03 0.095 0.06   -0.08 0.06 0.08   -0.10 0.00 0.10
                       -0.08 -0.06 0.12   -0.03 -0.095 0.14   0.03 -0.095 0.16
                       0.08 -0.06 0.18">
        <pin id="0 1"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <edge equality="true"/>
        <elasticity young="1e9" damping="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(kCurvedCholXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  d = MakeData(m);
  for (int i = 0; i < m->nq; i++) {
    d->qpos[i] = 0.005 * mju_sin(1.3 * i + 0.2);
    d->qvel[i] = 0.1 * mju_cos(0.7 * i + 0.4);
  }
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->warning[mjWARN_INERTIA].number, 0);
  EXPECT_EQ(d->nefmdof, 8);
  ASSERT_GT(d->nefmL, 0);
  EXPECT_EQ(d->efm_L[0], 1);
  rhs.resize(m->nv);
  lhs.resize(m->nv);
  mju_add(rhs.data(), d->qfrc_smooth, d->efm_c, m->nv);
  mj_mulM(m.get(), d.get(), lhs.data(), d->qacc_smooth);
  add_metric(m.get(), d.get(), lhs.data());
  err = 0;
  norm = mju_norm(rhs.data(), m->nv);
  ASSERT_GT(norm, 0);
  for (int i = 0; i < m->nv; i++) {
    mjtNum diff = lhs[i] - rhs[i];
    err += diff * diff;
  }
  EXPECT_LT(mju_sqrt(err) / norm, MjTol(1e-11, 1e-4));
}

// CG constraint solver with a 1D DER rod uses the O(nedge) tridiagonal twist
// fold (effTwistSolve + tw_diag), both when the twist DOFs touch only a static
// floor and when contacts couple them to a free rigid body, matching Newton
TEST_F(ElasticityTest, RodTwistConstraintMetricFold) {
  static constexpr char kUncoupledXml[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep="0.002"
            tolerance="1e-10" iterations="500"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 0.1" friction="0.6 0.005 0.0001"/>
      <flexcomp name="ribbon" type="grid" count="15 1 1" spacing="0.04 0.04 0.04"
                pos="0 0 0.009" radius="0.003" dim="1" mass="0.05">
        <pin id="0 1"/>
        <contact selfcollide="none"/>
        <edge equality="true"/>
        <elasticity young="1e8" shear="4e7" damping="0.05"
                    section="box" sectionsize="0.02 0.003"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  static constexpr char kCoupledXml[] = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" timestep="0.002"
            tolerance="1e-10" iterations="500"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 0.1" friction="0.6 0.005 0.0001"/>
      <body name="slider" pos="0.24 0 0.027">
        <freejoint/>
        <geom name="slider_geom" type="box" size="0.08 0.08 0.01" mass="0.2"
              friction="0.6 0.005 0.0001"/>
      </body>
      <flexcomp name="ribbon" type="grid" count="15 1 1" spacing="0.04 0.04 0.04"
                pos="0 0 0.009" radius="0.003" dim="1" mass="0.05">
        <pin id="0 1"/>
        <contact selfcollide="none"/>
        <edge equality="true"/>
        <elasticity young="1e8" shear="4e7" damping="0.05"
                    section="box" sectionsize="0.02 0.003"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {kUncoupledXml, kCoupledXml}) {
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d_cg = MakeData(m);
    for (int e = 1; e < 14; e++) {
      std::string jname =
          (e < 13) ? ("ribbon_twist_" + std::to_string(e)) : "ribbon_twist";
      int jid = mj_name2id(m.get(), mjOBJ_JOINT, jname.c_str());
      ASSERT_GE(jid, 0);
      d_cg->qpos[m->jnt_qposadr[jid]] = 0.4;
    }
    for (int step = 0; step < 20; step++) {
      mj_step(m.get(), d_cg.get());
      EXPECT_EQ(d_cg->warning[mjWARN_INERTIA].number, 0);
      EXPECT_EQ(d_cg->warning[mjWARN_BADQACC].number, 0);
    }
    mj_forward(m.get(), d_cg.get());
    ASSERT_GT(d_cg->ncon, 0);
    if (xml == kCoupledXml) {
      // the slider must touch the ribbon, coupling twist and rigid-body trees
      int slider = mj_name2id(m.get(), mjOBJ_GEOM, "slider_geom");
      ASSERT_GE(slider, 0);
      bool slider_ribbon = false;
      for (int c = 0; c < d_cg->ncon; c++) {
        const mjContact& con = d_cg->contact[c];
        slider_ribbon |= (con.geom[0] == slider && con.flex[1] == 0) ||
                         (con.geom[1] == slider && con.flex[0] == 0);
      }
      EXPECT_TRUE(slider_ribbon);
    }

    // compare single-step qacc with Newton at the same state
    MjDataPtr d_nw = MakeData(m);
    mju_copy(d_nw->qpos, d_cg->qpos, m->nq);
    mju_copy(d_nw->qvel, d_cg->qvel, m->nv);
    mju_copy(d_nw->plastic, d_cg->plastic, m->nplastic);
    m->opt.solver = mjSOL_NEWTON;
    mj_forward(m.get(), d_nw.get());

    mjtNum err = 0, norm = mju_norm(d_nw->qacc, m->nv);
    ASSERT_GT(norm, 0);
    for (int i = 0; i < m->nv; i++) {
      mjtNum diff = d_cg->qacc[i] - d_nw->qacc[i];
      err += diff * diff;
    }
    EXPECT_LT(mju_sqrt(err) / norm, MjTol(1e-5, 1e-3));
  }
}

// circular 1D <flexcomp> rod pushed tangentially on a frictional plane rolls
// without slipping: axial twist DOFs spin at -vy / r and the effective rolling
// inertia reduces linear acceleration to 2/3 of frictionless sliding
TEST_F(ElasticityTest, RodCapsuleRollingOnPlane) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 -9.81" timestep="0.001"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 0.1" %s solref="0.004 1"/>
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.08 0.08 0.08"
                pos="0 0 0.02" radius="0.02" dim="1" mass="0.12">
        <edge equality="true"/>
        <contact selfcollide="none" %s solref="0.004 1"/>
        <elasticity young="1e7" shear="1e7" damping="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  auto run_push = [&](const char* floor_attr, mjtNum* vy_out, mjtNum* dy_out,
                      std::vector<mjtNum>* w_twist_out) {
    char xml[2048], error[1024] = {0};
    std::snprintf(xml, sizeof(xml), xml_template, floor_attr, floor_attr);
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);

    // settle onto the floor under gravity
    for (int step = 0; step < 100; step++) {
      mj_step(m.get(), d.get());
    }
    mj_forward(m.get(), d.get());
    mjtNum y0 = 0;
    for (int v = 0; v < 6; v++) {
      y0 += d->flexvert_xpos[3 * v + 1] / 6;
    }

    // apply uniform +Y horizontal push F = 0.24 N (0.04 N per vertex)
    for (int v = 0; v < 6; v++) {
      int bid = m->flex_vertbodyid[v];
      d->xfrc_applied[6 * bid + 1] = 0.04;
    }
    for (int step = 0; step < 200; step++) {
      mj_step(m.get(), d.get());
    }
    mj_forward(m.get(), d.get());

    mjtNum y1 = 0, vy = 0;
    for (int v = 0; v < 6; v++) {
      y1 += d->flexvert_xpos[3 * v + 1] / 6;
      vy += d->qvel[m->body_dofadr[m->flex_vertbodyid[v]] + 1] / 6;
    }
    *dy_out = y1 - y0;
    *vy_out = vy;
    w_twist_out->clear();
    for (int e = 0; e < 5; e++) {
      std::string jname =
          (e < 4) ? ("rod_twist_" + std::to_string(e)) : "rod_twist";
      int jid = mj_name2id(m.get(), mjOBJ_JOINT, jname.c_str());
      ASSERT_GE(jid, 0);
      w_twist_out->push_back(d->qvel[m->jnt_dofadr[jid]]);
    }
  };

  mjtNum vy_slide = 0, dy_slide = 0, vy_roll = 0, dy_roll = 0;
  std::vector<mjtNum> w_slide, w_roll;
  run_push(R"(condim="1" friction="0 0 0")", &vy_slide, &dy_slide, &w_slide);
  run_push(R"(friction="1 0.005 0.0001")", &vy_roll, &dy_roll, &w_roll);

  // frictionless floor: pure translation at a = F / M = 2 m/s^2, zero axial
  // spin
  EXPECT_NEAR(vy_slide, 0.4, 0.01);
  for (mjtNum w : w_slide) {
    EXPECT_NEAR(w, 0.0, 1e-6);
  }

  // frictional floor: solid cylinder M + I / r^2 = 1.5 M gives 2/3 speed and
  // displacement, and rolling without slip gives omega_twist = -vy / r
  EXPECT_NEAR(vy_roll / vy_slide, 2.0 / 3.0, 0.04);
  EXPECT_NEAR(dy_roll / dy_slide, 2.0 / 3.0, 0.04);
  const mjtNum r = 0.02;
  for (int e = 0; e < 5; e++) {
    EXPECT_NEAR(w_roll[e], -vy_roll / r, 0.05 * std::abs(vy_roll / r))
        << "edge " << e;
  }
}

// free 3D tumbling of straight (discrete) and curved (RK4) box-section ribbons
// in zero gravity preserves unit-orthogonal (t_old, u_old) reference frames
// without secular roll drift or spurious elastic bending energy
// rotation R minimizing sum_i |R a_i - b_i|^2 over n centered point pairs
// (Horn's quaternion method: the dominant eigenvector, by power iteration)
void BestFitRotation(const mjtNum* a, const mjtNum* b, int n, mjtNum R[9]) {
  mjtNum S[9] = {0};
  for (int i = 0; i < n; i++) {
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        S[3 * r + c] += a[3 * i + r] * b[3 * i + c];
      }
    }
  }
  const mjtNum xx = S[0], xy = S[1], xz = S[2], yx = S[3], yy = S[4], yz = S[5],
               zx = S[6], zy = S[7], zz = S[8];
  mjtNum N[16] = {xx + yy + zz, yz - zy,      zx - xz,       xy - yx,
                  yz - zy,      xx - yy - zz, xy + yx,       zx + xz,
                  zx - xz,      xy + yx,      -xx + yy - zz, yz + zy,
                  xy - yx,      zx + xz,      yz + zy,       -xx - yy + zz};
  mjtNum shift = 0;
  for (int i = 0; i < 9; i++) shift += std::abs(S[i]);
  for (int i = 0; i < 4; i++) N[5 * i] += shift;
  mjtNum q[4] = {1, 0.1, 0.1, 0.1};
  for (int iter = 0; iter < 2000; iter++) {
    mjtNum qn[4];
    mju_mulMatVec(qn, N, q, 4, 4);
    mju_normalize4(qn);
    mju_copy4(q, qn);
  }
  mju_quat2Mat(R, q);
}

TEST_F(ElasticityTest, RodFreeTumblingParallelTransport) {
  // Case 1: straight box ribbon under integrator="discrete" tumbling > 2.5 rad
  {
    static constexpr char kStraightXml[] = R"(
    <mujoco>
      <option integrator="discrete" gravity="0 0 0" timestep="0.002"/>
      <worldbody>
        <flexcomp name="ribbon" type="grid" count="5 1 1"
                  spacing="0.05 0.05 0.05" dim="1" radius="0.005" mass="0.01">
          <edge equality="true"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" poisson="0.25" damping="0"
                      section="box" sectionsize="0.02 0.005"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    )";
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(kStraightXml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    ASSERT_EQ(m->nplastic, 6 * 4 + 3);

    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    mjtNum com[3] = {0, 0, 0};
    for (int v = 0; v < 5; v++) {
      mju_addToScl3(com, d->flexvert_xpos + 3 * v, 1.0 / 5.0);
    }
    const mjtNum omega[3] = {1.5, -1.2, 2.0};
    for (int v = 0; v < 5; v++) {
      mjtNum rel[3], vel[3];
      mju_sub3(rel, d->flexvert_xpos + 3 * v, com);
      mju_cross(vel, omega, rel);
      mju_copy3(d->qvel + m->body_dofadr[m->flex_vertbodyid[v]], vel);
    }
    for (int e = 0; e < 4; e++) {
      std::string jname =
          (e < 3) ? ("ribbon_twist_" + std::to_string(e)) : "ribbon_twist";
      int jid = mj_name2id(m.get(), mjOBJ_JOINT, jname.c_str());
      ASSERT_GE(jid, 0);
      mjtNum te[3];
      mju_sub3(te, d->flexvert_xpos + 3 * (e + 1), d->flexvert_xpos + 3 * e);
      mju_normalize3(te);
      d->qvel[m->jnt_dofadr[jid]] = mju_dot3(omega, te);
    }

    mjtNum spin;  // axial spin of the straight ribbon, conserved
    {
      mjtNum t0[3];
      mju_sub3(t0, d->flexvert_xpos + 3 * 4, d->flexvert_xpos);
      mju_normalize3(t0);
      spin = mju_dot3(omega, t0);
    }
    for (int step = 0; step < 500; step++) {
      mj_step(m.get(), d.get());
    }
    mj_forward(m.get(), d.get());

    // the vertices cannot tell a roll about the axis of a straight ribbon,
    // which costs no energy either: the time-parallel frames must not spin, so
    // that every edge's material frame rolls by exactly the conserved spin *
    // time
    for (int e = 0; e < 4; e++) {
      int jnt = mj_rodChainTwistJnt(m.get(), 0, e);
      ASSERT_GE(jnt, 0);
      EXPECT_NEAR(d->qpos[m->jnt_qposadr[jnt]], spin * d->time,
                  MjTol(1e-6, 1e-3) * std::abs(spin * d->time))
          << "edge " << e;
    }
    EXPECT_LT(RodBendingEnergy(m.get(), d.get(), 0), MjTol(1e-8, 1e-5));
  }

  // Case 2: curved quarter-arc box ribbon undergoing 3D non-planar Poinsot
  // tumbling (nonzero spherical holonomy across edges)
  static constexpr int kN = 5;
  static constexpr double kR = 0.2;
  std::string point, element;
  for (int i = 0; i < kN; i++) {
    double angle = (mjPI / 2) * i / (kN - 1);
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%.17g %.17g 0  ", kR * std::cos(angle),
                  kR * std::sin(angle));
    point += buf;
    if (i > 0) {
      element += std::to_string(i - 1) + " " + std::to_string(i) + "  ";
    }
  }
  std::string xml = R"(
  <mujoco>
    <option integrator="RK4" gravity="0 0 0" timestep="0.0005"/>
    <worldbody>
      <flexcomp name="ribbon" type="direct" dim="1" radius="0.005" mass="0.01"
                element=")" +
                    element + R"(" point=")" + point + R"(">
        <edge equality="true"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="5e5" poisson="0.25" damping="0"
                    section="box" sectionsize="0.02 0.005"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nplastic, 6 * (kN - 1) + (kN - 2));

  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(RodBendingEnergy(m.get(), d.get(), 0), 0.0, MjTol(1e-12, 1e-6));

  // compute center of mass and impart rigid oblique 3D angular velocity
  mjtNum com[3] = {0, 0, 0};
  for (int v = 0; v < kN; v++) {
    mju_addToScl3(com, d->flexvert_xpos + 3 * v, 1.0 / kN);
  }
  const mjtNum omega[3] = {0.6, -0.8, 1.0};
  for (int v = 0; v < kN; v++) {
    mjtNum rel[3], vel[3];
    mju_sub3(rel, d->flexvert_xpos + 3 * v, com);
    mju_cross(vel, omega, rel);
    mju_copy3(d->qvel + m->body_dofadr[m->flex_vertbodyid[v]], vel);
  }
  for (int e = 0; e < kN - 1; e++) {
    std::string jname =
        (e < kN - 2) ? ("ribbon_twist_" + std::to_string(e)) : "ribbon_twist";
    int jid = mj_name2id(m.get(), mjOBJ_JOINT, jname.c_str());
    ASSERT_GE(jid, 0);
    mjtNum te[3];
    mju_sub3(te, d->flexvert_xpos + 3 * (e + 1), d->flexvert_xpos + 3 * e);
    mju_normalize3(te);
    d->qvel[m->jnt_dofadr[jid]] = mju_dot3(omega, te);
  }

  mj_forward(m.get(), d.get());
  std::vector<mjtNum> Mqvel(m->nv);
  mj_mulM(m.get(), d.get(), Mqvel.data(), d->qvel);
  const mjtNum ke0 = 0.5 * mju_dot(d->qvel, Mqvel.data(), m->nv);
  ASSERT_GT(ke0, 1e-5);

  // material frames and centered vertices before tumbling
  mjtNum box[2];
  std::vector<mjtNum> mat0(9 * (kN - 1)), mat1(9 * (kN - 1));
  ASSERT_TRUE(mj_rodBoxEdge(m.get(), d.get(), 0, mat0.data(), box));
  auto centered = [&](std::vector<mjtNum>* x) {
    x->assign(d->flexvert_xpos, d->flexvert_xpos + 3 * kN);
    mjtNum c[3] = {0, 0, 0};
    for (int v = 0; v < kN; v++) mju_addToScl3(c, x->data() + 3 * v, 1.0 / kN);
    for (int v = 0; v < kN; v++) mju_subFrom3(x->data() + 3 * v, c);
  };
  std::vector<mjtNum> x0, x1;
  centered(&x0);

  for (int step = 0; step < 500; step++) {
    mj_step(m.get(), d.get());
  }
  mj_forward(m.get(), d.get());

  // material frames follow the best-fit rigid rotation of the vertices
  centered(&x1);
  mjtNum R[9];
  BestFitRotation(x0.data(), x1.data(), kN, R);
  mjtNum shape_err = 0;
  for (int v = 0; v < kN; v++) {
    mjtNum rx[3];
    mju_mulMatVec3(rx, R, x0.data() + 3 * v);
    shape_err = std::max(shape_err, mju_dist3(rx, x1.data() + 3 * v));
  }
  EXPECT_LT(shape_err, 1e-3 * kR);  // nearly rigid motion
  mjtNum rot_angle = mju_acos(mju_clip(0.5 * (R[0] + R[4] + R[8] - 1), -1, 1));
  EXPECT_GT(rot_angle, 0.2);  // a substantial rotation
  ASSERT_TRUE(mj_rodBoxEdge(m.get(), d.get(), 0, mat1.data(), box));
  for (int e = 0; e < kN - 1; e++) {
    mjtNum expected[9];
    mju_mulMatMat(expected, R, mat0.data() + 9 * e, 3, 3, 3);
    for (int i = 0; i < 9; i++) {
      EXPECT_NEAR(mat1[9 * e + i], expected[i], MjTol(2e-3, 1e-2))
          << "element " << e << " entry " << i;
    }
  }

  // verify negligible elastic bending/twisting energy (no secular roll drift)
  const mjtNum bend_energy = RodBendingEnergy(m.get(), d.get(), 0);
  EXPECT_LT(bend_energy, MjTol(1e-8, 1e-5));
}

// passive contact on interpolated flexes (trilinear volumetric, trilinear shell
// elastic2d="bend", quadratic, and with pinned cage nodes): contacts are routed
// to d->nefmcon with d->nefc == 0 and produce upward repulsive qfrc_spring on
// the cage nodes matching the basis-interpolated contact force
TEST_F(ElasticityTest, InterpPassiveContactForceAndPin) {
  struct Case {
    const char* name;
    const char* dof;
    const char* elasticity;
    const char* pin;
    int expected_interp;
  };
  const Case cases[] = {
      {"trilinear_vol", "trilinear",
       R"(<elasticity young="1e4" poisson="0.3"/>)", "", 1},
      {"trilinear_shell_bend", "trilinear",
       R"(<elasticity young="1e4" poisson="0.3" thickness="0.01" elastic2d="bend"/>)",
       "", -1},
      {"quadratic_vol", "quadratic",
       R"(<elasticity young="1e4" poisson="0.3"/>)", "", 2},
      {"trilinear_pinned", "trilinear",
       R"(<elasticity young="1e4" poisson="0.3"/>)", R"(<pin id="1 3 5 7"/>)",
       1},
  };

  for (const Case& tc : cases) {
    SCOPED_TRACE(tc.name);
    std::string xml = std::string(R"(
    <mujoco>
      <option integrator="discrete" gravity="0 0 -9.81" timestep="0.002"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <flexcomp name="box" type="grid" count="3 3 3" cellcount="1 1 1"
                  spacing="0.1 0.1 0.1" radius="0.01" dim="3" mass="1.0"
                  pos="0 0 0.095" dof=")") +
                      tc.dof + R"(">
          <contact selfcollide="none" passive="true"/>
          )" + tc.elasticity +
                      "\n" + tc.pin + R"(
        </flexcomp>
      </worldbody>
    </mujoco>
    )";

    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    EXPECT_EQ(m->flex_interp[0], tc.expected_interp);

    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    // At rest qpos = 0: bottom vertices at z = 0.095 - 0.1 - radius(0.01) =
    // -0.015 penetrate the z = 0 plane, while elastic strain is zero so
    // qfrc_spring is purely the passive contact force.
    EXPECT_GT(d->ncon, 0);
    EXPECT_GT(d->nefmcon, 0);
    EXPECT_EQ(d->nefc, 0);

    // Check that total upward vertical force in qfrc_spring is strictly
    // positive and matches the published passive contact rows in
    // d->efm_con_ind/val.
    mjtNum total_fz_nodes = 0;
    for (int n = 0; n < m->flex_nodenum[0]; n++) {
      int b = m->body_weldid[m->flex_nodebodyid[m->flex_nodeadr[0] + n]];
      if (m->body_dofnum[b] == 3) {
        int da = m->body_dofadr[b];
        total_fz_nodes += d->qfrc_spring[da + 2];
        EXPECT_GE(d->qfrc_spring[da + 2], 0);
      }
    }
    EXPECT_GT(total_fz_nodes, 0);

    // Verify qfrc_spring equals the sum of force * Jacobian row over all
    // published passive contacts, and for unpinned trilinear flexes verify
    // total vertical force equals sum of -k * dist over the 9 bottom contacts.
    std::vector<mjtNum> expected_qfrc(m->nv, 0);
    for (int adr = 0; adr < d->nefmcon;) {
      int nnz = d->efm_con_ind[adr];
      mjtNum f = d->efm_con_val[adr + 1];
      EXPECT_GT(f, 0);
      for (int a = 0; a < nnz; a++) {
        expected_qfrc[d->efm_con_ind[adr + 2 + a]] +=
            f * d->efm_con_val[adr + 2 + a];
      }
      adr += 2 + nnz;
    }
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i], expected_qfrc[i],
                  MjTol(1e-11, 1e-4) * mju_max(1.0, mju_abs(expected_qfrc[i])))
          << "dof " << i;
    }
    if (std::string(tc.name) == "trilinear_vol") {
      // 8 unpinned cage nodes of mass m_node = 1/8 kg, nodenum/vertnum = 8/27,
      // 9 bottom vertices with penetration -dist = 0.015 m:
      //   4 corners (sum w^2 = 1), 4 edges (sum w^2 = 1/2), 1 center (sum w^2 =
      //   1/4),
      // so sum(1 / sum(w^2)) = 4*1 + 4*2 + 1*4 = 16.
      mjtNum k_base = 5e7 * (1.0 / 8.0) * (8.0 / 27.0);
      EXPECT_NEAR(total_fz_nodes, 16 * k_base * 0.015,
                  MjTol(1e-8, 1e-2) * (16 * k_base * 0.015));
    }
  }
}

// FCM quadrature scales elastic restoring forces on cut boundary cells by the
// occupied volume fraction, matches between dim=2 and dim=3, and simulates
// stably when outer corner cells (including cell 0) are empty.
TEST_F(ElasticityTest, FcmCutCellPassiveForcesAndSimulation) {
  static constexpr char kShapeXml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="1e-3"/>
    <worldbody>
      <flexcomp name="soft" type="SHAPE" dim="DIM" count="7 7 7"
                spacing="0.05 0.05 0.05" cellcount="2 2 2" dof="trilinear"
                mass="1" radius="0.001">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.25" damping="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  auto compute_stretch_force_norm = [&](const char* shape, const char* dim) {
    std::string xml = kShapeXml;
    xml.replace(xml.find("SHAPE"), 5, shape);
    xml.replace(xml.find("DIM"), 3, dim);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
    EXPECT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    mj_kinematics(m.get(), d.get());

    // Apply a uniform 5% axial stretch along x to all node bodies
    for (int n = 0; n < m->flex_nodenum[0]; n++) {
      int b = m->flex_nodebodyid[m->flex_nodeadr[0] + n];
      int dof = m->body_dofadr[b];
      if (dof >= 0) {
        d->qpos[m->jnt_qposadr[m->dof_jntid[dof]]] = 0.05 * d->xpos[3 * b + 0];
      }
    }
    mj_forward(m.get(), d.get());
    return mju_norm(d->qfrc_spring, m->nv);
  };

  mjtNum f_box3 = compute_stretch_force_norm("box", "3");
  mjtNum f_box2 = compute_stretch_force_norm("box", "2");
  mjtNum f_ell3 = compute_stretch_force_norm("ellipsoid", "3");
  mjtNum f_ell2 = compute_stretch_force_norm("ellipsoid", "2");

  EXPECT_GT(f_box3, 0);
  EXPECT_NEAR(f_box2, f_box3, MjTol(1e-9, 1e-4));
  EXPECT_GT(f_ell3, 0);
  EXPECT_LT(f_ell3, 0.7 * f_box3);
  EXPECT_GT(f_ell3, 0.35 * f_box3);
  EXPECT_NEAR(f_ell2, f_ell3, 0.1 * f_ell3);

  // Simulate a 6x6x6 sphere (where cell 0 at the outer corner is empty,
  // boundary cells are cut, and interior cells are full) for both trilinear and
  // quadratic interpolation.
  static constexpr char kSphereSimXml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="1e-3" solver="CG"/>
    <worldbody>
      <body name="parent">
        <inertial mass="1" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
        <flexcomp name="sphere" type="ellipsoid" dim="2" count="7 7 7"
                  spacing="0.04 0.04 0.04" cellcount="6 6 6" dof="DOF_TYPE"
                  mass="1" radius="0.001">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="500" poisson="0.25" damping="0.05"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* dof : {"trilinear", "quadratic"}) {
    std::string xml = kSphereSimXml;
    xml.replace(xml.find("DOF_TYPE"), 8, dof);
    char error[1024] = {0};
    MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << dof << ": " << error;
    EXPECT_EQ(m->flex_stiffness[0], 0) << dof << ": cell 0 should be empty";

    MjDataPtr d = MakeData(m);
    d->qpos[0] = 0.01;
    mj_forward(m.get(), d.get());
    EXPECT_GT(mju_norm(d->qfrc_spring, m->nv), 0) << dof;

    for (int step = 0; step < 50; step++) {
      mj_step(m.get(), d.get());
    }
    EXPECT_EQ(d->warning[mjWARN_BADQACC].number, 0) << dof;
    for (int i = 0; i < m->nq; i++) {
      ASSERT_TRUE(std::isfinite(d->qpos[i])) << dof << " qpos " << i;
    }
  }
}

// ------------------------------- plasticity ----------------------------------

using PlasticityTest = MujocoTest;

TEST_F(PlasticityTest, NonPlasticBackwardCompatibility) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide" stiffness="10" springref="0.5"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t" stiffness="10" springlength="0.1 0.2">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->nplastic, 0);
  EXPECT_EQ(m->jnt_plasticadr[0], -1);
  EXPECT_EQ(m->jnt_springyield[0], 0);
  EXPECT_EQ(m->tendon_plasticadr[0], -1);
  EXPECT_EQ(m->tendon_springyield[0], 0);
  EXPECT_EQ(mj_stateSize(m.get(), mjSTATE_PLASTIC), 0);
}

TEST_F(PlasticityTest, NegativeSpringYieldError) {
  static constexpr char jnt_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" stiffness="10" springyield="-0.1"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(jnt_xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("springyield cannot be negative"));

  static constexpr char tendon_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="10" springyield="-0.1">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";

  m = LoadModelFromString(tendon_xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("springyield cannot be negative"));

  static constexpr char flex_neg_xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <elasticity young="1000" yield="-0.1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(flex_neg_xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("yield cannot be negative"));

  static constexpr char flex_1d_xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="rod" type="grid" dim="1" count="3 1 1" spacing="0.1 0.1 0.1">
        <elasticity young="1000" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  m = LoadModelFromString(flex_1d_xml, error, sizeof(error));
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("flex plasticity requires a 2d or 3d flex"));
}

TEST_F(PlasticityTest, SlideAndHingeReturnMappingAndState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.01">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0"
               stiffness="100" springref="0.2" springyield="0.1"/>
        <joint name="hinge" type="hinge" axis="0 0 1"
               stiffness="50" springref="30" springyield="15"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nplastic, 2);
  EXPECT_EQ(m->jnt_plasticadr[0], 0);
  EXPECT_EQ(m->jnt_plasticadr[1], 1);
  EXPECT_MJTNUM_EQ(m->jnt_springyield[0], 0.1);
  EXPECT_NEAR(m->jnt_springyield[1], 15.0 * mjPI / 180.0, MjTol(1e-12, 1e-5));

  MjDataPtr d = MakeData(m);
  EXPECT_MJTNUM_EQ(d->plastic[0], 0.2);
  EXPECT_NEAR(d->plastic[1], 30.0 * mjPI / 180.0, MjTol(1e-12, 1e-5));

  // Within elastic range: plastic reference does not change after step
  d->qpos[0] = 0.25;  // |0.25 - 0.2| = 0.05 <= 0.1
  d->qpos[1] = 30.0 * mjPI / 180.0;
  mj_step(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->plastic[0], 0.2);

  // Exceed positive yield on slide joint: qpos = 0.5 -> plastic = 0.5 - 0.1 =
  // 0.4
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.5;
  d->qpos[1] = 30.0 * mjPI / 180.0;
  d->qvel[0] = 0;
  m->opt.timestep = 0;  // pure return mapping at current qpos
  mj_step(m.get(), d.get());
  EXPECT_NEAR(d->plastic[0], 0.4, MjTol(1e-12, 1e-5));

  // Force and potential energy use updated d->plastic: at qpos = 0.4, force and
  // energy are 0
  d->qpos[0] = d->plastic[0];
  d->qpos[1] = d->plastic[1];
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(d->qfrc_spring[0], 0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->qfrc_spring[1], 0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->energy[0], 0, MjTol(1e-12, 1e-5));

  // Exceed negative yield on slide joint: qpos = 0.1 < 0.4 - 0.1 -> plastic =
  // 0.1 + 0.1 = 0.2
  d->qpos[0] = 0.1;
  mj_step(m.get(), d.get());
  EXPECT_NEAR(d->plastic[0], 0.2, MjTol(1e-12, 1e-5));

  // State get/set via mjSTATE_PLASTIC and mjSTATE_PHYSICS
  d->plastic[0] = 0.77;
  d->plastic[1] = -0.33;
  int phys_sz = mj_stateSize(m.get(), mjSTATE_PHYSICS);
  EXPECT_EQ(phys_sz, m->nq + m->nv + m->na + m->nplastic);
  std::vector<mjtNum> state(phys_sz);
  mj_getState(m.get(), d.get(), state.data(), mjSTATE_PHYSICS);

  // Reset restores initial spring references
  mj_resetData(m.get(), d.get());
  EXPECT_MJTNUM_EQ(d->plastic[0], 0.2);

  // Restore saved state
  mj_setState(m.get(), d.get(), state.data(), mjSTATE_PHYSICS);
  EXPECT_MJTNUM_EQ(d->plastic[0], 0.77);
  EXPECT_MJTNUM_EQ(d->plastic[1], -0.33);
}

TEST_F(PlasticityTest, BallAndFreeJointReturnMapping) {
  static constexpr char xml_free_err[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="free" type="free" stiffness="10" springyield="0.2"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m_err = LoadModelFromString(xml_free_err, error, sizeof(error));
  EXPECT_THAT(m_err.get(), IsNull());
  EXPECT_THAT(error,
              HasSubstr("springyield should not be defined in free joint"));

  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>
    <option gravity="0 0 0" timestep="0">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <body>
        <joint name="ball" type="ball" stiffness="10" springyield="0.1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nplastic, 4);  // 4 (ball)
  EXPECT_EQ(m->jnt_plasticadr[0], 0);

  MjDataPtr d = MakeData(m);

  // Rotate ball joint around z by 0.3 rad (> 0.1 yield)
  mjtNum angle = 0.3;
  mjtNum axis[3] = {0, 0, 1};
  mju_axisAngle2Quat(d->qpos, axis, angle);

  mj_step(m.get(), d.get());

  // Ball joint quaternion reference should be unit norm and at angle 0.3 - 0.1
  // = 0.2 rad
  mjtNum qnorm = mju_norm(d->plastic, 4);
  EXPECT_NEAR(qnorm, 1.0, MjTol(1e-12, 1e-5));
  mjtNum expected_quat[4];
  mju_axisAngle2Quat(expected_quat, axis, 0.2);
  for (int k = 0; k < 4; k++) {
    EXPECT_NEAR(d->plastic[k], expected_quat[k], MjTol(1e-12, 1e-5));
  }

  // At the new yielded reference, spring forces and potential energy are zero
  mju_copy(d->qpos, d->plastic, m->nq);
  mj_forward(m.get(), d.get());
  for (int i = 0; i < m->nv; i++) {
    EXPECT_NEAR(d->qfrc_spring[i], 0, MjTol(1e-12, 1e-5));
  }
  EXPECT_NEAR(d->energy[0], 0, MjTol(1e-12, 1e-5));

  // Runtime springyield = 0 when plasticadr >= 0 produces full plastic slip
  m->jnt_springyield[0] = 0;
  mju_axisAngle2Quat(d->qpos, axis, 0.5);
  mj_step(m.get(), d.get());
  for (int k = 0; k < 4; k++) {
    EXPECT_NEAR(d->plastic[k], d->qpos[k], MjTol(1e-12, 1e-5));
  }
}

TEST_F(PlasticityTest, TendonFixedAndSpatialReturnMapping) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.01">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <site name="s0" pos="0 0 0"/>
      <body pos="1 0 0">
        <joint name="j" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
        <site name="s1" pos="0 0 0"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="tfixed" stiffness="100" springlength="0.1 0.3" springyield="0.2">
        <joint joint="j" coef="1"/>
      </fixed>
      <spatial name="tspatial" stiffness="100" springyield="0.15">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nplastic, 4);  // 2 tendons x 2
  EXPECT_EQ(m->tendon_plasticadr[0], 0);
  EXPECT_EQ(m->tendon_plasticadr[1], 2);

  MjDataPtr d = MakeData(m);
  // Fixed tendon initial deadband [0.1, 0.3]
  EXPECT_NEAR(d->plastic[0], 0.1, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.3, MjTol(1e-12, 1e-5));
  // Spatial tendon had automatic springlength [-1, -1], resolved to [1.0, 1.0]
  // at qpos0
  EXPECT_NEAR(d->plastic[2], 1.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[3], 1.0, MjTol(1e-12, 1e-5));

  // Stretch j to 0.7 with 0 velocity and dt=0:
  // tfixed length = 0.7 > 0.3 + 0.2 -> delta = (0.7 - 0.3) - 0.2 = 0.2 -> [0.3,
  // 0.5] tspatial length = 1.7 > 1.0 + 0.15 -> delta = (1.7 - 1.0) - 0.15 =
  // 0.55 -> [1.55, 1.55]
  m->opt.timestep = 0;
  d->qpos[0] = 0.7;
  mj_forward(m.get(), d.get());
  mj_step(m.get(), d.get());

  EXPECT_NEAR(d->plastic[0], 0.3, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.5, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1] - d->plastic[0], 0.2, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 1.55, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[3], 1.55, MjTol(1e-12, 1e-5));

  // Compress j to -0.1 with dt=0:
  // tfixed length = -0.1 < 0.3 - 0.2 -> delta = (-0.1 - 0.3) + 0.2 = -0.2 ->
  // [0.1, 0.3]
  // tspatial length = 0.9 < 1.55 - 0.15 -> delta = (0.9 - 1.55) + 0.15 = -0.50
  // -> [1.05, 1.05]
  d->qpos[0] = -0.1;
  mj_forward(m.get(), d.get());
  mj_step(m.get(), d.get());

  EXPECT_NEAR(d->plastic[0], 0.1, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.3, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 1.05, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[3], 1.05, MjTol(1e-12, 1e-5));

  // Step with non-zero timestep and velocity under both Euler and RK4:
  // Set stiffness=0 so velocity stays constant (qacc=0), dt=0.01, qvel=20.0
  // Starting from qpos=0.05 (tspatial length=1.05, deadband [1.05, 1.05]):
  // After one step, qpos = 0.25, L_next = 1.05 + 0.01 * 20 = 1.25.
  // Yield is 0.15, so new plastic length should be 1.25 - 0.15 = 1.10.
  m->tendon_stiffness[0] = 0;
  m->tendon_stiffness[1] = 0;
  m->opt.timestep = 0.01;
  for (mjtIntegrator integ : {mjINT_EULER, mjINT_RK4}) {
    m->opt.integrator = integ;
    mj_resetData(m.get(), d.get());
    d->qpos[0] = 0.05;
    d->qvel[0] = 20.0;
    d->plastic[2] = 1.05;
    d->plastic[3] = 1.05;
    mj_step(m.get(), d.get());
    EXPECT_NEAR(d->plastic[2], 1.10, MjTol(1e-12, 1e-5));
    EXPECT_NEAR(d->plastic[3], 1.10, MjTol(1e-12, 1e-5));
  }
}

TEST_F(PlasticityTest, XmlAndBinarySaveLoadRoundTrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree"/>
    <worldbody>
      <body>
        <joint name="h" type="hinge" stiffness="10" springyield="30"/>
        <geom size="0.1" mass="1"/>
      </body>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" yield="0.04"/>
      </flexcomp>
    </worldbody>
    <tendon>
      <fixed name="t" stiffness="20" springyield="0.25">
        <joint joint="h" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_NEAR(m->jnt_springyield[0], 30.0 * mjPI / 180.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->tendon_springyield[0], 0.25, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->flex_yield[0], 0.04, MjTol(1e-12, 1e-5));

  // XML round-trip
  std::string saved_xml = SaveAndReadXml(m.get());
  MjModelPtr m_xml = LoadModelFromString(saved_xml, error, sizeof(error));
  ASSERT_THAT(m_xml.get(), NotNull()) << error;
  EXPECT_EQ(m_xml->nplastic, m->nplastic);
  EXPECT_NEAR(m_xml->jnt_springyield[0], m->jnt_springyield[0], 1e-6);
  EXPECT_NEAR(m_xml->tendon_springyield[0], m->tendon_springyield[0], 1e-6);
  EXPECT_NEAR(m_xml->flex_yield[0], m->flex_yield[0], 1e-6);

  // Binary MJB round-trip
  mjtSize buf_sz = mj_sizeModel(m.get());
  std::vector<char> buf(buf_sz);
  mj_saveModel(m.get(), nullptr, buf.data(), buf_sz);
  mjModel* m_bin = mj_loadModelBuffer(buf.data(), buf_sz);
  ASSERT_THAT(m_bin, NotNull());
  EXPECT_EQ(m_bin->nplastic, m->nplastic);
  EXPECT_EQ(m_bin->jnt_plasticadr[0], m->jnt_plasticadr[0]);
  EXPECT_EQ(m_bin->tendon_plasticadr[0], m->tendon_plasticadr[0]);
  EXPECT_EQ(m_bin->flex_plasticadr[0], m->flex_plasticadr[0]);
  EXPECT_MJTNUM_EQ(m_bin->jnt_springyield[0], m->jnt_springyield[0]);
  EXPECT_MJTNUM_EQ(m_bin->tendon_springyield[0], m->tendon_springyield[0]);
  EXPECT_MJTNUM_EQ(m_bin->flex_yield[0], m->flex_yield[0]);
  mj_deleteModel(m_bin);
}

TEST_F(PlasticityTest, ImplicitIntegratorWithPlasticity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="implicit" timestep="0.005"/>
    <worldbody>
      <body>
        <joint name="s" type="slide" stiffness="200" damping="20" springyield="0.1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Pull joint to 0.5 (> 0.1 yield) and simulate until it settles at the new
  // plastic reference
  d->qpos[0] = 0.5;
  for (int step = 0; step < 500; step++) {
    mj_step(m.get(), d.get());
  }
  EXPECT_GT(d->plastic[0], 0.35);
  EXPECT_NEAR(d->qpos[0], d->plastic[0], 1e-3);
  EXPECT_NEAR(d->qvel[0], 0.0, 1e-3);
}

TEST_F(PlasticityTest, DiscreteIntegratorWithPlasticity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="0.005"/>
    <worldbody>
      <body>
        <joint name="s1" type="slide" stiffness="200" damping="20" springyield="0.1"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body pos="0 1 0">
        <joint name="s2" type="slide" damping="20"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t2" stiffness="200" springyield="0.1">
        <joint joint="s2" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Pull both joints to 0.5 (> 0.1 yield) and simulate until they settle at
  // their new plastic references
  d->qpos[0] = 0.5;
  d->qpos[1] = 0.5;
  for (int step = 0; step < 500; step++) {
    mj_step(m.get(), d.get());
  }
  EXPECT_GT(d->plastic[0], 0.35);
  EXPECT_NEAR(d->qpos[0], d->plastic[0], 1e-3);
  EXPECT_NEAR(d->qvel[0], 0.0, 1e-3);

  EXPECT_GT(d->plastic[1], 0.35);
  EXPECT_NEAR(d->qpos[1], d->plastic[1], 1e-3);
  EXPECT_NEAR(d->qvel[1], 0.0, 1e-3);
}

TEST_F(PlasticityTest, RecompilePreservesPlasticState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b1">
        <joint name="j1" type="slide" stiffness="100" springyield="0.1"/>
        <geom size="0.1" mass="1"/>
      </body>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" yield="0.05"/>
      </flexcomp>
    </worldbody>
    <tendon>
      <fixed name="t1" stiffness="100" springyield="0.2">
        <joint joint="j1" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";

  char error[1024];
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, NotNull()) << error;
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());
  ASSERT_EQ(m->nplastic, 9);  // j1 (1) + t1 (2) + tet (6)

  // Set non-initial plastic references
  d->plastic[0] = 0.42;
  d->plastic[1] = 0.15;
  d->plastic[2] = 0.35;
  for (int i = 0; i < 6; i++) {
    d->plastic[3 + i] = 0.01 * (i + 1);
  }

  // Add a new body and joint to spec and recompile
  mjsBody* b2 = mjs_addBody(mjs_findBody(spec, "world"), nullptr);
  mjsJoint* j2 = mjs_addJoint(b2, nullptr);
  j2->type = mjJNT_SLIDE;
  j2->stiffness[0] = 50;
  j2->springyield = 0.05;
  mjsGeom* g2 = mjs_addGeom(b2, nullptr);
  g2->size[0] = 0.1;

  ASSERT_EQ(mj_recompile(spec, nullptr, m, d), 0) << mjs_getError(spec);
  ASSERT_EQ(m->nplastic, 10);  // j1 (1) + j2 (1) + t1 (2) + tet (6)
  EXPECT_NEAR(d->plastic[0], 0.42, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 0.15, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[3], 0.35, MjTol(1e-12, 1e-5));
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(d->plastic[4 + i], 0.01 * (i + 1), MjTol(1e-12, 1e-5));
  }

  mj_deleteData(d);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(PlasticityTest, UnilateralTendonAndKeyframesAndSleep) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body name="b1" sleep="allowed">
        <joint name="j1" type="slide" axis="1 0 0" stiffness="10" springyield="0.1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="tslack" stiffness="100" springlength="0 0.5" springyield="0.2">
        <joint joint="j1" coef="1"/>
      </fixed>
    </tendon>
    <keyframe>
      <key name="k0" qpos="0.3" plastic="0.15 0 0.6"/>
      <key name="k_default" qpos="0.1"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nplastic, 3);  // j1 (1) + tslack (2)
  ASSERT_EQ(m->nkey, 2);

  // Explicit keyframe plastic values
  EXPECT_NEAR(m->key_plastic[0], 0.15, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[1], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[2], 0.6, MjTol(1e-12, 1e-5));

  // Default keyframe plastic values (populated from initial plastic0)
  EXPECT_NEAR(m->key_plastic[3], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[4], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[5], 0.5, MjTol(1e-12, 1e-5));

  MjDataPtr d = MakeData(m);

  // Unilateral slack tendon [0, 0.5]: stretch j1 to 0.9 (> 0.5 + 0.2)
  // lower bound stays pinned at 0, upper bound elongates to 0.9 - 0.2 = 0.7
  d->qpos[0] = 0.9;
  mj_forward(m.get(), d.get());
  mj_step(m.get(), d.get());
  EXPECT_NEAR(d->plastic[0], 0.8, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 0.7, MjTol(1e-12, 1e-5));

  // Save current state into keyframe 1 and reset to keyframe 0
  mj_setKeyframe(m.get(), d.get(), 1);
  EXPECT_NEAR(m->key_plastic[3], 0.8, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[4], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(m->key_plastic[5], 0.7, MjTol(1e-12, 1e-5));

  mj_resetDataKeyframe(m.get(), d.get(), 0);
  EXPECT_NEAR(d->plastic[0], 0.15, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[1], 0.0, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 0.6, MjTol(1e-12, 1e-5));

  // Put body to sleep and displace qpos: sleeping body/tendon should not yield
  d->body_awake[1] = mjS_ASLEEP;
  d->tree_awake[0] = 0;
  d->ntree_awake = 0;
  d->qpos[0] = 2.0;
  mj_Euler(m.get(), d.get());
  EXPECT_NEAR(d->plastic[0], 0.15, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->plastic[2], 0.6, MjTol(1e-12, 1e-5));
}

TEST_F(PlasticityTest, Flex3DStretchPlasticity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" damping="0.001" yield="0.05"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);
  ASSERT_EQ(m->nplastic, 6);
  EXPECT_EQ(m->flex_plasticadr[0], 0);
  EXPECT_NEAR(m->flex_yield[0], 0.05, MjTol(1e-12, 1e-5));

  MjDataPtr d = MakeData(m);
  for (int e = 0; e < 6; e++) {
    EXPECT_EQ(d->plastic[e], 0.0);
  }

  // 1. Pure hydrostatic expansion: scale all vertices by lambda = 1.2 (20% > 5%
  // yield). Because J2 plasticity is purely deviatoric (isochoric), hydrostatic
  // strain must produce zero plastic strain!
  {
    static constexpr char xml_free[] = R"(
    <mujoco>
      <option gravity="0 0 0" integrator="Euler" timestep="0"/>
      <worldbody>
        <flexcomp name="tet" type="direct" dim="3" mass="1"
                  point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="1000" poisson="0.25" damping="0.001" yield="0.05"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    )";
    MjModelPtr mf = LoadModelFromString(xml_free, error, sizeof(error));
    ASSERT_THAT(mf.get(), NotNull()) << error;
    MjDataPtr df = MakeData(mf);
    mj_forward(mf.get(), df.get());
    std::vector<mjtNum> rest(df->flexvert_xpos, df->flexvert_xpos + 12);
    for (int v = 0; v < 4; v++) {
      int adr = mf->body_dofadr[mf->flex_vertbodyid[v]];
      df->qpos[adr + 0] = 0.2 * rest[3 * v + 0];
      df->qpos[adr + 1] = 0.2 * rest[3 * v + 1];
      df->qpos[adr + 2] = 0.2 * rest[3 * v + 2];
    }
    mj_step(mf.get(), df.get());
    for (int e = 0; e < 6; e++) {
      EXPECT_NEAR(df->plastic[e], 0.0, MjTol(1e-10, 1e-5));
    }
  }

  // 2. Uniaxial / deviatoric stretch of vertex 3 along z:
  // First small displacement below yield (dz = 0.02 -> E_zz ~ 0.02 < 0.05): no
  // plastic strain
  d->qpos[2] = 0.02;
  mj_step(m.get(), d.get());
  for (int e = 0; e < 6; e++) {
    EXPECT_NEAR(d->plastic[e], 0.0, MjTol(1e-12, 1e-5));
  }

  // Now displace vertex 3 along z beyond yield (dz = 0.2 -> E_zz = 0.22):
  d->qpos[2] = 0.2;
  mj_forward(m.get(), d.get());
  mjtNum fz_before_yield = d->qfrc_spring[2];
  EXPECT_LT(fz_before_yield, -1.0);

  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());
  mjtNum fz_after_yield = d->qfrc_spring[2];

  // Plastic strain must be non-zero on the z-incident edges, and restoring
  // force magnitude reduced
  mjtNum pnorm = mju_norm(d->plastic, 6);
  EXPECT_GT(pnorm, 0.05);
  EXPECT_LT(mju_abs(fz_after_yield), mju_abs(fz_before_yield));

  // Stepping again at the same configuration produces no additional plastic
  // flow (on yield surface)
  std::vector<mjtNum> p_saved(d->plastic, d->plastic + 6);
  mj_step(m.get(), d.get());
  for (int e = 0; e < 6; e++) {
    EXPECT_NEAR(d->plastic[e], p_saved[e], MjTol(1e-12, 1e-5));
  }

  // When returned to rest (qpos[2] = 0), residual force pushes in the opposite
  // (+z) direction
  d->qpos[2] = 0.0;
  mj_forward(m.get(), d.get());
  EXPECT_GT(d->qfrc_spring[2], 0.1);

  // Resetting data clears plastic strain and restores zero force at rest
  mj_resetData(m.get(), d.get());
  mj_forward(m.get(), d.get());
  for (int e = 0; e < 6; e++) {
    EXPECT_EQ(d->plastic[e], 0.0);
  }
  EXPECT_NEAR(d->qfrc_spring[2], 0.0, MjTol(1e-12, 1e-5));

  // Put flex bodies to sleep and displace beyond yield: sleeping flex should
  // not yield
  for (int b = 1; b < m->nbody; b++) {
    d->body_awake[b] = mjS_ASLEEP;
  }
  for (int t = 0; t < m->ntree; t++) {
    d->tree_awake[t] = 0;
  }
  d->ntree_awake = 0;
  d->qpos[2] = 0.2;
  mj_Euler(m.get(), d.get());
  for (int e = 0; e < 6; e++) {
    EXPECT_EQ(d->plastic[e], 0.0);
  }

  // 3. Test Stable Neo-Hookean (elastic3d = 1) and mjsFlex::stressyield via
  // mjSpec:
  {
    mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
    ASSERT_THAT(spec, NotNull()) << error;
    spec->option.integrator = mjINT_DISCRETE;
    mjsFlex* flex = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "tet"));
    ASSERT_THAT(flex, NotNull());
    EXPECT_NEAR(flex->stressyield[0], 0.05, MjTol(1e-12, 1e-5));
    EXPECT_EQ(flex->stressyield[1], 0);  // single value: stretching only in 3D
    flex->elastic3d = 1;
    flex->stressyield[0] = 0.04;
    MjModelPtr m_snh(mj_compile(spec, nullptr));
    mj_deleteSpec(spec);
    ASSERT_THAT(m_snh.get(), NotNull());
    EXPECT_NEAR(m_snh->flex_yield[0], 0.04, MjTol(1e-12, 1e-5));

    MjDataPtr d_snh = MakeData(m_snh);
    d_snh->qpos[2] = 0.2;
    mj_forward(m_snh.get(), d_snh.get());
    mjtNum fz_snh_before = d_snh->qfrc_spring[2];
    EXPECT_LT(fz_snh_before, -1.0);

    mj_step(m_snh.get(), d_snh.get());
    mj_forward(m_snh.get(), d_snh.get());
    mjtNum fz_snh_after = d_snh->qfrc_spring[2];
    EXPECT_GT(mju_norm(d_snh->plastic, 6), 0.05);
    EXPECT_LT(mju_abs(fz_snh_after), mju_abs(fz_snh_before));

    // Large plastic stretch (dz = 1.0 -> lambda_z = 2.0): verify exact
    // nonlinear volume preservation det(G(g + s_p)) == det(G(g))
    d_snh->qpos[2] = 1.0;
    mj_step(m_snh.get(), d_snh.get());
    auto gram_det = [](const mjtNum q[6]) {
      mjtNum a = q[0], b = q[2], c = q[4];
      mjtNum d = 0.5 * (q[0] + q[2] - q[1]);
      mjtNum e = 0.5 * (q[0] + q[4] - q[5]);
      mjtNum f = 0.5 * (q[2] + q[4] - q[3]);
      return a * (b * c - f * f) - d * (d * c - e * f) + e * (d * f - b * e);
    };
    mjtNum g0[6], gp[6];
    for (int e = 0; e < 6; e++) {
      int idx = m_snh->flex_elemedge[e];
      g0[e] = m_snh->flexedge_length0[idx] * m_snh->flexedge_length0[idx];
      gp[e] = g0[e] + d_snh->plastic[e];
    }
    EXPECT_NEAR(gram_det(gp), gram_det(g0), MjTol(1e-10, 1e-4));
  }
}

TEST_F(PlasticityTest, Flex2DStretchPlasticity) {
  // Two triangles sharing edge (0, 1) along the x-axis:
  // v0=(0,0,0), v1=(1,0,0), v2=(0.5, 0.5, 0), v3=(0.5, -0.5, 0)
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1"
                point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                element="0 1 2  0 3 1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="both" yield="0.01"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);
  // 2 triangles * 3 edges = 6 stretch plastic entries + 5 bend plastic entries
  int stretch_pnum = 3 * m->flex_elemnum[0];
  int bend_pnum = m->flex_edgenum[0];
  ASSERT_EQ(m->nplastic, stretch_pnum + bend_pnum);

  MjDataPtr d = MakeData(m);

  // Small in-plane stretch below yield: plastic strain stays zero
  d->qpos[1] = -0.002;
  mj_step(m.get(), d.get());
  for (int i = 0; i < m->nplastic; i++) {
    EXPECT_NEAR(d->plastic[i], 0.0, MjTol(1e-12, 1e-5));
  }

  // In-plane membrane stretching beyond yield (pull v3 from y=-0.5 to y=-0.7):
  // stretch entries yield while bending entries stay zero
  d->qpos[0] = 0.0;
  d->qpos[1] = -0.2;
  d->qpos[2] = 0.0;
  mj_forward(m.get(), d.get());
  mjtNum fy_before = d->qfrc_spring[1];
  EXPECT_GT(fy_before, 1.0);

  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());
  mjtNum fy_after = d->qfrc_spring[1];
  EXPECT_GT(mju_norm(d->plastic, stretch_pnum), 0.01);
  EXPECT_NEAR(mju_norm(d->plastic + stretch_pnum, bend_pnum), 0.0,
              MjTol(1e-12, 1e-5));
  EXPECT_LT(mju_abs(fy_after), mju_abs(fy_before));

  // Stepping again at the same configuration produces no additional plastic
  // flow
  std::vector<mjtNum> p_saved(d->plastic, d->plastic + m->nplastic);
  mj_step(m.get(), d.get());
  for (int i = 0; i < m->nplastic; i++) {
    EXPECT_NEAR(d->plastic[i], p_saved[i], MjTol(1e-12, 1e-5));
  }

  // Equibiaxial in-plane tension (alpha > 0, s_dev = 0) yields via out-of-plane
  // thinning (alpha_pos), whereas equibiaxial compression (alpha < 0, s_dev =
  // 0) does not yield:
  {
    static constexpr char xml_tri[] = R"(
    <mujoco>
      <option gravity="0 0 0" integrator="Euler" timestep="0"/>
      <worldbody>
        <flexcomp name="tri" type="direct" dim="2" mass="1"
                  point="0 0 0  1 0 0  0 1 0" element="0 1 2">
          <contact contype="0" conaffinity="0" selfcollide="none"/>
          <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="stretch" yield="0.05"/>
        </flexcomp>
      </worldbody>
    </mujoco>
    )";
    MjModelPtr mt = LoadModelFromString(xml_tri, error, sizeof(error));
    ASSERT_THAT(mt.get(), NotNull()) << error;
    MjDataPtr dt = MakeData(mt);
    mj_forward(mt.get(), dt.get());
    std::vector<mjtNum> rest(dt->flexvert_xpos, dt->flexvert_xpos + 9);

    // Equibiaxial compression by 10% (lambda = 0.9 -> alpha < 0, s_dev = 0):
    for (int v = 0; v < 3; v++) {
      int adr = mt->body_dofadr[mt->flex_vertbodyid[v]];
      dt->qpos[adr + 0] = -0.1 * rest[3 * v + 0];
      dt->qpos[adr + 1] = -0.1 * rest[3 * v + 1];
    }
    mj_step(mt.get(), dt.get());
    EXPECT_NEAR(mju_norm(dt->plastic, 3), 0.0, MjTol(1e-10, 1e-5));

    // Equibiaxial tension by 10% (lambda = 1.1 -> alpha = 0.21 > 0.05):
    for (int v = 0; v < 3; v++) {
      int adr = mt->body_dofadr[mt->flex_vertbodyid[v]];
      dt->qpos[adr + 0] = 0.1 * rest[3 * v + 0];
      dt->qpos[adr + 1] = 0.1 * rest[3 * v + 1];
    }
    mj_step(mt.get(), dt.get());
    EXPECT_GT(mju_norm(dt->plastic, 3), 0.05);
  }

  // Also test discrete integrator with 3D plastic flex (exercises
  // mj_flexHessian invalidation)
  static constexpr char xml_discrete[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="0.005"/>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="500" poisson="0.25" damping="0.1" yield="0.05"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr md = LoadModelFromString(xml_discrete, error, sizeof(error));
  ASSERT_THAT(md.get(), NotNull()) << error;
  MjDataPtr dd = MakeData(md);
  dd->qpos[2] = 0.3;
  for (int step = 0; step < 50; step++) {
    mj_step(md.get(), dd.get());
  }
  EXPECT_GT(mju_norm(dd->plastic, 6), 0.05);
}

TEST_F(PlasticityTest, Flex2DBendPlasticity) {
  // Two triangles sharing interior hinge edge (0, 1) along the x-axis:
  // v0=(0,0,0), v1=(1,0,0), v2=(0.5, 0.5, 0), v3=(0.5, -0.5, 0)
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1"
                point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                element="0 1 2  0 3 1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="bend" yield="0.002"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);
  // elastic2d="bend" only allocates bending plastic entries (1 per edge = 5)
  ASSERT_EQ(m->nplastic, m->flex_edgenum[0]);
  ASSERT_EQ(m->nplastic, 5);

  MjDataPtr d = MakeData(m);

  // Small out-of-plane displacement below yield: plastic state stays zero
  d->qpos[2] = 0.01;
  mj_step(m.get(), d.get());
  for (int i = 0; i < m->nplastic; i++) {
    EXPECT_NEAR(d->plastic[i], 0.0, MjTol(1e-12, 1e-5));
  }

  // Fold flap vertex v3 upward along +z beyond yield (dz = 0.2):
  d->qpos[0] = 0.0;
  d->qpos[1] = 0.0;
  d->qpos[2] = 0.2;
  mj_forward(m.get(), d.get());
  mjtNum fz_before = d->qfrc_spring[2];
  EXPECT_LT(fz_before, -5e-4);

  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());
  mjtNum fz_after = d->qfrc_spring[2];
  EXPECT_GT(mju_norm(d->plastic, m->nplastic), 5e-4);
  EXPECT_LT(mju_abs(fz_after), mju_abs(fz_before));

  // Stepping again at the same configuration produces no additional plastic
  // flow
  std::vector<mjtNum> p_saved(d->plastic, d->plastic + m->nplastic);
  mj_step(m.get(), d.get());
  for (int i = 0; i < m->nplastic; i++) {
    EXPECT_NEAR(d->plastic[i], p_saved[i], MjTol(1e-12, 1e-5));
  }

  // When returned to flat (z = 0), residual plastic curvature pushes v3 toward
  // +z
  d->qpos[2] = 0.0;
  mj_forward(m.get(), d.get());
  EXPECT_GT(d->qfrc_spring[2], 3e-4);

  // Near-zero yield threshold with pure out-of-plane rotation of v3 around the
  // hinge edge (y = -0.5 * cos(theta), z = 0.5 * sin(theta), so A_1 == A_2):
  // tangential bending torque around the hinge drops to zero after one step
  m->flex_yield[1] = 1e-12;
  mj_resetData(m.get(), d.get());
  mjtNum theta = 0.2;
  d->qpos[1] = 0.5 * (1.0 - std::cos(theta));
  d->qpos[2] = 0.5 * std::sin(theta);
  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());
  mjtNum tau_hinge =
      d->qfrc_spring[1] * std::sin(theta) + d->qfrc_spring[2] * std::cos(theta);
  EXPECT_NEAR(tau_hinge, 0.0, MjTol(1e-10, 1e-5));

  // Dynamic fold-and-release test (timestep > 0, edge equality="true"):
  // fold v3 to 45 deg, yield, and release dynamically; verify the residual
  // fold angle is retained without reverse yielding on springback
  static constexpr char xml_dyn[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete" timestep="0.002"/>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1"
                point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                element="0 1 2  0 3 1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <edge equality="true" solref="0.002 1"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="bend" damping="0.05" yield="0.005"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr md = LoadModelFromString(xml_dyn, error, sizeof(error));
  ASSERT_THAT(md.get(), NotNull()) << error;
  MjDataPtr dd = MakeData(md);
  mjtNum theta_fold = 0.8;
  dd->qpos[1] = 0.5 * (1.0 - std::cos(theta_fold));
  dd->qpos[2] = 0.5 * std::sin(theta_fold);
  mj_step(md.get(), dd.get());
  std::vector<mjtNum> p_folded(dd->plastic, dd->plastic + md->nplastic);
  for (int step = 0; step < 300; step++) {
    mj_step(md.get(), dd.get());
  }
  for (int i = 0; i < md->nplastic; i++) {
    EXPECT_NEAR(dd->plastic[i], p_folded[i], MjTol(1e-10, 1e-5));
  }
  mjtNum y3 = -0.5 + dd->qpos[1];
  mjtNum z3 = dd->qpos[2];
  mjtNum theta_final = std::atan2(z3, -y3);
  EXPECT_GT(theta_final, 0.65);
}

TEST_F(PlasticityTest, Flex2DSeparateStretchAndBendYield) {
  // Two triangles sharing interior hinge edge (0, 1) along the x-axis, with
  // stretching and bending: yield="stretch bend" yields each mode independently
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1"
                point="0 0 0  1 0 0  0.5 0.5 0  0.5 -0.5 0"
                element="0 1 2  0 3 1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="both" yield="YIELD"/>
        <pin id="0 1 2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  // plastic state after one step of pure folding (rotating v3 by 0.4 rad around
  // the hinge, no stretching) and of pure in-plane stretching (v3 pulled to
  // y=-0.7), with optional runtime yield strains
  struct Flow {
    int mode, nplastic;
    mjtNum fold_stretch, fold_bend, pull_stretch, pull_bend;
  };
  auto flow = [&](const char* yield, mjtNum runtime_stretch = -1,
                  mjtNum runtime_bend = -1) -> Flow {
    std::string model_xml(xml);
    model_xml.replace(model_xml.find("YIELD"), 5, yield);
    char error[1024];
    MjModelPtr m = LoadModelFromString(model_xml.c_str(), error, sizeof(error));
    EXPECT_THAT(m.get(), NotNull()) << error;
    if (!m) return {};
    if (runtime_stretch >= 0) m->flex_yield[0] = runtime_stretch;
    if (runtime_bend >= 0) m->flex_yield[1] = runtime_bend;

    // only the modes with a positive yield strain have plastic state
    int mode = m->flex_plasticmode[0];
    int stretch_pnum = (mode & 2) ? 3 * m->flex_elemnum[0] : 0;
    int bend_pnum = (mode & 1) ? m->flex_edgenum[0] : 0;
    EXPECT_EQ(m->nplastic, stretch_pnum + bend_pnum);

    Flow result = {mode, (int)m->nplastic, 0, 0, 0, 0};
    MjDataPtr d = MakeData(m);
    mjtNum theta = 0.4;
    d->qpos[1] = 0.5 * (1.0 - std::cos(theta));
    d->qpos[2] = 0.5 * std::sin(theta);
    mj_step(m.get(), d.get());
    result.fold_stretch = mju_norm(d->plastic, stretch_pnum);
    result.fold_bend = mju_norm(d->plastic + stretch_pnum, bend_pnum);

    mj_resetData(m.get(), d.get());
    d->qpos[1] = -0.2;
    mj_step(m.get(), d.get());
    result.pull_stretch = mju_norm(d->plastic, stretch_pnum);
    result.pull_bend = mju_norm(d->plastic + stretch_pnum, bend_pnum);
    return result;
  };
  mjtNum tol = MjTol(1e-12, 1e-5);

  // one value: both modes are plastic and yield, same as two equal values
  Flow both = flow("0.002");
  EXPECT_EQ(both.mode, 3);
  EXPECT_EQ(both.nplastic, 11);
  EXPECT_NEAR(both.fold_stretch, 0.0, tol);
  EXPECT_GT(both.fold_bend, 1e-4);
  EXPECT_GT(both.pull_stretch, 0.01);
  EXPECT_NEAR(both.pull_bend, 0.0, tol);
  Flow both2 = flow("0.002 0.002");
  EXPECT_EQ(both2.mode, 3);
  EXPECT_EQ(both2.fold_bend, both.fold_bend);
  EXPECT_EQ(both2.pull_stretch, both.pull_stretch);

  // large stretching and small bending threshold: yields in bending only
  Flow bend_only = flow("1 0.002");
  EXPECT_EQ(bend_only.mode, 3);
  EXPECT_NEAR(bend_only.fold_stretch, 0.0, tol);
  EXPECT_EQ(bend_only.fold_bend, both.fold_bend);
  EXPECT_NEAR(bend_only.pull_stretch, 0.0, tol);
  EXPECT_NEAR(bend_only.pull_bend, 0.0, tol);

  // small stretching and large bending threshold: yields in stretching only
  Flow stretch_only = flow("0.002 1");
  EXPECT_EQ(stretch_only.mode, 3);
  EXPECT_NEAR(stretch_only.fold_stretch, 0.0, tol);
  EXPECT_NEAR(stretch_only.fold_bend, 0.0, tol);
  EXPECT_EQ(stretch_only.pull_stretch, both.pull_stretch);
  EXPECT_NEAR(stretch_only.pull_bend, 0.0, tol);

  // a zero threshold at compile time: the mode has no plastic state and stays
  // elastic, also if its yield strain is changed at runtime
  Flow elastic_stretch = flow("0 0.002", /*runtime_stretch=*/0.001);
  EXPECT_EQ(elastic_stretch.mode, 1);
  EXPECT_EQ(elastic_stretch.nplastic, 5);
  EXPECT_EQ(elastic_stretch.fold_bend, both.fold_bend);
  Flow elastic_bend = flow("0.002 0", /*runtime_stretch=*/-1,
                           /*runtime_bend=*/0.001);
  EXPECT_EQ(elastic_bend.mode, 2);
  EXPECT_EQ(elastic_bend.nplastic, 6);
  EXPECT_EQ(elastic_bend.pull_stretch, both.pull_stretch);

  // a zero threshold at runtime on a plastic mode is full plastic flow
  Flow full_stretch = flow("1 1", /*runtime_stretch=*/0);
  EXPECT_GT(full_stretch.pull_stretch, both.pull_stretch);
  EXPECT_NEAR(full_stretch.fold_bend, 0.0, tol);
  Flow full_bend = flow("1 1", /*runtime_stretch=*/-1, /*runtime_bend=*/0);
  EXPECT_GT(full_bend.fold_bend, both.fold_bend);
  EXPECT_NEAR(full_bend.pull_stretch, 0.0, tol);
}

TEST_F(PlasticityTest, FlexYieldOfAbsentModeAllocatesNoState) {
  // a positive threshold for a mode the flex does not have allocates no plastic
  // state: bending yield of a stretch-only 2D shell; a single value is only the
  // stretching yield strain of a 3D flex
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="shell" type="direct" dim="2" mass="1"
                point="0 0 0  1 0 0  0.5 0.5 0" element="0 1 2">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02" elastic2d="stretch" yield="0 0.1"/>
      </flexcomp>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  0 1 0  0 0 1" element="0 1 2 3">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  EXPECT_EQ(m->nplastic, 6);
  EXPECT_EQ(m->flex_plasticadr[0], -1);
  EXPECT_EQ(m->flex_plasticmode[0], 0);
  EXPECT_EQ(m->flex_plasticadr[1], 0);
  EXPECT_EQ(m->flex_plasticmode[1], 2);
  EXPECT_FLOAT_EQ(m->flex_yield[2], 0.05);
  EXPECT_EQ(m->flex_yield[3], 0);
}

TEST_F(PlasticityTest, FlexInterpStretchPlasticity) {
  // 1. 3D Trilinear (order = 1) single non-cubic cell:
  static constexpr char xml_3d_trilin[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="box1" type="grid" dim="3" dof="trilinear"
                count="2 2 2" cellcount="1 1 1" spacing="2 1 0.5" mass="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" damping="0.001" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml_3d_trilin, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);
  ASSERT_EQ(m->flex_interp[0], 1);
  ASSERT_EQ(m->nplastic, 6);  // 1 cell * 6 strain components

  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> rest(m->flex_node0, m->flex_node0 + 3 * m->nflexnode);

  // Pure hydrostatic expansion (scale all nodes by 1.2 -> 20% > 5% yield):
  // deviatoric strain is zero, so plastic strain remains zero
  for (int i = 0; i < m->nflexnode; i++) {
    int adr = m->body_dofadr[m->flex_nodebodyid[i]];
    d->qpos[adr + 0] = 0.2 * rest[3 * i + 0];
    d->qpos[adr + 1] = 0.2 * rest[3 * i + 1];
    d->qpos[adr + 2] = 0.2 * rest[3 * i + 2];
  }
  mj_step(m.get(), d.get());
  for (int k = 0; k < 6; k++) {
    EXPECT_NEAR(d->plastic[k], 0.0, MjTol(1e-12, 1e-5));
  }

  // Uniaxial stretch along z below yield (eps_zz = 0.02 < 0.05):
  mj_resetData(m.get(), d.get());
  for (int i = 0; i < m->nflexnode; i++) {
    int adr = m->body_dofadr[m->flex_nodebodyid[i]];
    d->qpos[adr + 2] = 0.02 * (rest[3 * i + 2] - rest[2]);
  }
  mj_step(m.get(), d.get());
  for (int k = 0; k < 6; k++) {
    EXPECT_NEAR(d->plastic[k], 0.0, MjTol(1e-12, 1e-5));
  }

  // Combined stretch and shear beyond yield on non-cubic cell:
  mj_resetData(m.get(), d.get());
  for (int i = 0; i < m->nflexnode; i++) {
    int adr = m->body_dofadr[m->flex_nodebodyid[i]];
    mjtNum dx0 = rest[3 * i + 0] - rest[0];
    mjtNum dy0 = rest[3 * i + 1] - rest[1];
    mjtNum dz0 = rest[3 * i + 2] - rest[2];
    d->qpos[adr + 0] = 0.1 * dy0 + 0.08 * dz0;
    d->qpos[adr + 1] = 0.1 * dx0 + 0.06 * dz0;
    d->qpos[adr + 2] = 0.2 * dz0 + 0.08 * dx0 + 0.06 * dy0;
  }
  mj_forward(m.get(), d.get());
  mjtNum f_norm_before = mju_norm(d->qfrc_spring, m->nv);
  EXPECT_GT(f_norm_before, 10.0);

  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());
  mjtNum f_norm_after = mju_norm(d->qfrc_spring, m->nv);
  EXPECT_GT(mju_norm(d->plastic, 6), 0.05);
  EXPECT_GT(mju_abs(d->plastic[3]), 0.01);
  EXPECT_GT(mju_abs(d->plastic[4]), 0.01);
  EXPECT_GT(mju_abs(d->plastic[5]), 0.01);
  // Trace of plastic strain must be zero (volume-preserving J2 flow)
  EXPECT_NEAR(d->plastic[0] + d->plastic[1] + d->plastic[2], 0.0,
              MjTol(1e-12, 1e-5));
  EXPECT_LT(f_norm_after, f_norm_before);

  // Stepping again at the same configuration produces no additional flow
  std::vector<mjtNum> p_saved(d->plastic, d->plastic + 6);
  mj_step(m.get(), d.get());
  for (int k = 0; k < 6; k++) {
    EXPECT_NEAR(d->plastic[k], p_saved[k], MjTol(1e-12, 1e-5));
  }

  // At the exact plastic rest configuration rotated by an arbitrary 3D rigid
  // rotation R, spring forces must be identically zero:
  mjtNum axis[3] = {1.0, 2.0, -1.5};
  mju_normalize3(axis);
  mjtNum quat_rot[4];
  mju_axisAngle2Quat(quat_rot, axis, 0.7);
  for (int i = 0; i < m->nflexnode; i++) {
    mjtNum dx0[3] = {rest[3 * i + 0] - rest[0], rest[3 * i + 1] - rest[1],
                     rest[3 * i + 2] - rest[2]};
    mjtNum x_p[3];
    x_p[0] = rest[3 * i + 0] + d->plastic[0] * dx0[0] +
             0.5 * d->plastic[3] * dx0[1] + 0.5 * d->plastic[4] * dx0[2];
    x_p[1] = rest[3 * i + 1] + 0.5 * d->plastic[3] * dx0[0] +
             d->plastic[1] * dx0[1] + 0.5 * d->plastic[5] * dx0[2];
    x_p[2] = rest[3 * i + 2] + 0.5 * d->plastic[4] * dx0[0] +
             0.5 * d->plastic[5] * dx0[1] + d->plastic[2] * dx0[2];
    mjtNum x_rot[3];
    mju_rotVecQuat(x_rot, x_p, quat_rot);
    int adr = m->body_dofadr[m->flex_nodebodyid[i]];
    d->qpos[adr + 0] = x_rot[0] - rest[3 * i + 0];
    d->qpos[adr + 1] = x_rot[1] - rest[3 * i + 1];
    d->qpos[adr + 2] = x_rot[2] - rest[3 * i + 2];
  }
  mj_forward(m.get(), d.get());
  EXPECT_NEAR(mju_norm(d->qfrc_spring, m->nv), 0.0, MjTol(1e-7, 1e-3));

  // 2. 3D Quadratic (order = 2) single non-cubic cell (27 nodes):
  static constexpr char xml_3d_quad[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="box2" type="grid" dim="3" dof="quadratic"
                count="2 2 2" cellcount="1 1 1" spacing="2 1 0.5" mass="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.25" damping="0.001" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr mq = LoadModelFromString(xml_3d_quad, error, sizeof(error));
  ASSERT_THAT(mq.get(), NotNull()) << error;
  ASSERT_EQ(mq->flex_interp[0], 2);
  ASSERT_EQ(mq->nplastic, 6);
  MjDataPtr dq = MakeData(mq);
  mj_forward(mq.get(), dq.get());
  std::vector<mjtNum> rest_q(mq->flex_node0,
                             mq->flex_node0 + 3 * mq->nflexnode);
  for (int i = 0; i < mq->nflexnode; i++) {
    int adr = mq->body_dofadr[mq->flex_nodebodyid[i]];
    dq->qpos[adr + 0] = 0.25 * (rest_q[3 * i + 0] - rest_q[0]);
  }
  mj_step(mq.get(), dq.get());
  EXPECT_GT(mju_norm(dq->plastic, 6), 0.1);
  EXPECT_NEAR(dq->plastic[0] + dq->plastic[1] + dq->plastic[2], 0.0,
              MjTol(1e-12, 1e-5));
  for (int i = 0; i < mq->nflexnode; i++) {
    mjtNum dx0[3] = {rest_q[3 * i + 0] - rest_q[0],
                     rest_q[3 * i + 1] - rest_q[1],
                     rest_q[3 * i + 2] - rest_q[2]};
    int adr = mq->body_dofadr[mq->flex_nodebodyid[i]];
    dq->qpos[adr + 0] = dq->plastic[0] * dx0[0];
    dq->qpos[adr + 1] = dq->plastic[1] * dx0[1];
    dq->qpos[adr + 2] = dq->plastic[2] * dx0[2];
  }
  mj_forward(mq.get(), dq.get());
  EXPECT_NEAR(mju_norm(dq->qfrc_spring, mq->nv), 0.0, MjTol(1e-9, 1e-4));

  // 3. 2D Shell Mode (interp < 0, elastic2d = "both"):
  static constexpr char xml_2d_shell[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="Euler" timestep="0"/>
    <worldbody>
      <flexcomp name="shell1" type="grid" dim="3" dof="trilinear"
                count="2 2 2" cellcount="1 1 1" spacing="0.5 0.5 0.5" mass="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson="0.3" thickness="0.02"
                    elastic2d="both" damping="0.001" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr ms = LoadModelFromString(xml_2d_shell, error, sizeof(error));
  ASSERT_THAT(ms.get(), NotNull()) << error;
  ASSERT_EQ(ms->flex_interp[0], -1);
  // 6 boundary faces * 3 strain components = 18
  ASSERT_EQ(ms->nplastic, 18);
  MjDataPtr ds = MakeData(ms);
  mj_forward(ms.get(), ds.get());
  std::vector<mjtNum> rest_s(ms->flex_node0,
                             ms->flex_node0 + 3 * ms->nflexnode);

  // Stretch along x by 20% (> 5% yield):
  for (int i = 0; i < ms->nflexnode; i++) {
    int adr = ms->body_dofadr[ms->flex_nodebodyid[i]];
    ds->qpos[adr + 0] = 0.2 * (rest_s[3 * i + 0] - rest_s[0]);
  }
  mj_forward(ms.get(), ds.get());
  mjtNum fs_before = mju_norm(ds->qfrc_spring, ms->nv);
  EXPECT_GT(fs_before, 1.0);

  mj_step(ms.get(), ds.get());
  mj_forward(ms.get(), ds.get());
  mjtNum fs_after = mju_norm(ds->qfrc_spring, ms->nv);
  EXPECT_GT(mju_norm(ds->plastic, ms->nplastic), 0.1);
  EXPECT_LT(fs_after, fs_before);

  // Stepping again at the same configuration produces no additional plastic
  // flow
  std::vector<mjtNum> ps_saved(ds->plastic, ds->plastic + ms->nplastic);
  mj_step(ms.get(), ds.get());
  for (int k = 0; k < ms->nplastic; k++) {
    EXPECT_NEAR(ds->plastic[k], ps_saved[k], MjTol(1e-12, 1e-5));
  }
}

// sleeping flex does not accumulate passive spring forces in qfrc_spring
TEST_F(PassiveTest, SleepingFlexDoesNotAccumulateQfrcSpring) {
  // edge spring
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <option gravity="0 0 -10">
        <flag sleep="enable"/>
      </option>
      <worldbody>
        <body name="v0" pos="0 0 1"/>
        <body name="v1" pos="0 0 0" sleep="allowed">
          <joint name="j1" type="slide" axis="0 0 1"/>
          <geom size="0.1" mass="1"/>
        </body>
        <body name="b2" pos="2 0 0">
          <joint name="j2" type="slide" axis="1 0 0"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <deformable>
        <flex name="f1" dim="1" body="v0 v1" element="0 1">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <edge stiffness="10" damping="0"/>
        </flex>
      </deformable>
    </mujoco>
    )";
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);

    // at qpos=-1 the edge length is 2 (rest length 1): edge spring force (+10)
    // balances gravity (-10) on v1
    d->qpos[0] = -1.0;
    d->qvel[1] = 1.0;  // keep b2 awake
    mj_forward(m.get(), d.get());
    ASSERT_EQ(d->body_awake[2], mjS_AWAKE);
    EXPECT_NEAR(d->qfrc_spring[0], 10.0, MjTol(1e-7, 1e-4));

    // once v1 falls asleep, qfrc_spring must stay at 10 rather than accumulate
    for (int i = 0; i < 20; i++) {
      mj_step(m.get(), d.get());
    }
    ASSERT_EQ(d->body_awake[2], mjS_ASLEEP);
    ASSERT_EQ(d->body_awake[3], mjS_AWAKE);
    EXPECT_NEAR(d->qfrc_spring[0], 10.0, MjTol(1e-7, 1e-4));
  }

  // 1D rod bending elasticity (mj_flexPassiveBend1D)
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <option timestep="0.01" integrator="discrete" solver="CG"
        gravity="0 0 0">
        <flag sleep="enable"/>
      </option>
      <worldbody>
        <body name="v0" pos="0 0 0"/>
        <body name="v1" pos="1 0 0"/>
        <body name="v2" pos="2 0 0" sleep="allowed">
          <joint name="j1" type="slide" axis="0 0 1"/>
          <geom size="0.1" mass="1"/>
        </body>
        <body name="b2" pos="4 0 0">
          <joint name="j2" type="slide" axis="1 0 0"/>
          <geom size="0.1" mass="1"/>
        </body>
      </worldbody>
      <deformable>
        <flex name="rod" dim="1" radius="0.1" body="v0 v1 v2"
          element="0 1 1 2">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e6"/>
        </flex>
      </deformable>
    </mujoco>
    )";
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);

    // bend v2 to z=-0.1 and balance rod bending force with vertical gravity
    d->qpos[0] = -0.1;
    d->qvel[1] = 1.0;  // keep b2 awake
    mj_forward(m.get(), d.get());
    mjtNum f_rod = d->qfrc_spring[0];
    ASSERT_GT(f_rod, 1.0);
    m->opt.gravity[2] = -f_rod;

    // once v2 falls asleep, qfrc_spring must stay at f_rod rather than
    // accumulate
    for (int i = 0; i < 30; i++) {
      mj_step(m.get(), d.get());
    }
    ASSERT_EQ(d->body_awake[3], mjS_ASLEEP);
    ASSERT_EQ(d->body_awake[4], mjS_AWAKE);
    EXPECT_NEAR(d->qfrc_spring[0], f_rod, MjTol(1e-12, 1e-5));
  }
}

}  // namespace
}  // namespace mujoco
