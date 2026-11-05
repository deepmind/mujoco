// Copyright 2024 DeepMind Technologies Limited
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

// Tests for engine/engine_collision_gjk.c.

#include "src/engine/engine_collision_gjk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <ccd/ccd.h>   // IWYU pragma: keep
#include <ccd/vec3.h>  // IWYU pragma: keep
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_collision_convex.h"
#include "test/fixture.h"

// uncomment to run tests with libccd
// #define TEST_WITH_LIBCCD

namespace mujoco {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::Pointwise;

constexpr mjtNum kTolerance = 1e-6;
constexpr int kMaxIterations = 1000;
constexpr char kEllipsoidXml[] = R"(
<mujoco model="Ellipsoid Test">
  <compiler angle="radian"/>
  <size nkey="1"/>
  <worldbody>
    <geom name="geom1" size="0.1 0.1 0.1" pos="0 0 -0.1" type="ellipsoid"/>
    <body pos="0 0 0.1">
      <joint type="free" limited="false" actuatorfrclimited="false"/>
      <geom name="geom2" size="0.01 0.02 0.1" type="ellipsoid"/>
    </body>
  </worldbody>

  <keyframe>
    <key time="1.446"
      qpos="0.000886189 -0.0303047 0.0951303 0.98783 0.155468 0.00454524 1.60038e-06"
      qvel="0.00769267 -0.258656 -0.0775641 2.73712 0.0813998 -0.000166485"/>
  </keyframe>
</mujoco>)";

mjtNum GeomDist(const MjModelPtr& m, const MjDataPtr& d, int g1, int g2,
                mjtNum x1[3], mjtNum x2[3], mjtNum cutoff = mjMAX_LIMIT) {
  mjCCDConfig config;
  mjCCDStatus status;

  // set config
  config.max_iterations = kMaxIterations, config.tolerance = kTolerance,
  config.max_contacts = 0;  // no geom contacts needed
  config.dist_cutoff = cutoff;
  config.buffer = nullptr;

  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m.get(), d.get(), g1, 0);
  mjc_initCCDObj(&obj2, m.get(), d.get(), g2, 0);

  mjtNum dist = mjc_ccd(&config, &status, &obj1, &obj2);
  if (status.nx > 0) {
    if (x1 != nullptr) mju_copy3(x1, status.x1);
    if (x2 != nullptr) mju_copy3(x2, status.x2);
  }
  return dist;
}

int Penetration(mjCCDStatus& status, mjtNum& depth, std::vector<mjtNum>& dir,
                std::vector<mjtNum>& pos, const MjModelPtr& model,
                const MjDataPtr& data, int g1, int g2, mjtNum margin = 0,
                int max_contacts = 1) {
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, model.get(), data.get(), g1, margin);
  mjc_initCCDObj(&obj2, model.get(), data.get(), g2, margin);

#if defined(TEST_WITH_LIBCCD)
  if (max_contacts == 1) {
    ccd_t ccd;
    CCD_INIT(&ccd);
    ccd.mpr_tolerance = kTolerance;
    ccd.epa_tolerance = kTolerance;
    ccd.max_iterations = kMaxIterations;
    ccd.center1 = mjccd_center;
    ccd.center2 = mjccd_center;
    ccd.support1 = mjccd_support;
    ccd.support2 = mjccd_support;

    ccd_real_t ccd_depth;
    ccd_vec3_t ccd_dir, ccd_pos;

    int ret =
        ccdMPRPenetration(&obj1, &obj2, &ccd, &ccd_depth, &ccd_dir, &ccd_pos);
    if (ret) return 0;
    dir.resize(3);
    pos.resize(3);
    depth = -ccd_depth;
    mju_copy3(dir.data(), ccd_dir.v);
    mju_copy3(pos.data(), ccd_pos.v);
    return 1;
  }
#endif

  mjCCDConfig config;

  // set config
  auto buffer = std::vector<std::byte>(
      mjc_ccdSize(model->npolygonmax, model->nmeshdegmax, kMaxIterations));
  config.max_iterations = kMaxIterations;
  config.tolerance = kTolerance;
  config.max_contacts = max_contacts;
  config.dist_cutoff = 0;  // no geom distances needed
  config.npolygonmax = model->npolygonmax;
  config.nmeshdegmax = model->nmeshdegmax;
  config.max_contacts = max_contacts;
  config.buffer = buffer.data();

  mjtNum dist = mjc_ccd(&config, &status, &obj1, &obj2);
  if (dist < 0) {
    dir.resize(3 * status.nx);
    pos.resize(3 * status.nx);
    for (int i = 0; i < status.nx; ++i) {
      // compute direction
      mju_sub3(&dir[3 * i], status.x1 + 3 * i, status.x2 + 3 * i);
      mju_normalize3(&dir[3 * i]);

      // compute position
      pos[3 * i + 0] = 0.5 * (status.x1[0 + 3 * i] + status.x2[0 + 3 * i]);
      pos[3 * i + 1] = 0.5 * (status.x1[1 + 3 * i] + status.x2[1 + 3 * i]);
      pos[3 * i + 2] = 0.5 * (status.x1[2 + 3 * i] + status.x2[2 + 3 * i]);
    }
    depth = dist;
    return status.nx;
  }

  // no contacts
  return 0;
}

using MjGjkTest = MujocoTest;

TEST_F(MjGjkTest, SphereSphereDist) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
      <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum x1[3], x2[3];
  mjtNum dist = GeomDist(model, data, geom1, geom2, x1, x2);

  EXPECT_EQ(dist, 1);
  EXPECT_THAT(x1, ElementsAre(-.5, 0, 0));
  EXPECT_THAT(x2, ElementsAre(.5, 0, 0));
}

TEST_F(MjGjkTest, SphereSphereDistCutoff) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
      <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum dist = GeomDist(model, data, geom1, geom2, nullptr, nullptr, .999999);

  EXPECT_EQ(dist, mjMAX_LIMIT);
}

TEST_F(MjGjkTest, SphereSphereNoDist) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="sphere" pos="-1.5 0 0" size="1"/>
      <geom name="geom2" type="sphere" pos="1.5 0 0" size="1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 0);
}

TEST_F(MjGjkTest, SphereSphereIntersect) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="sphere" pos="-1 0 0" size="3"/>
      <geom name="geom2" type="sphere" pos="3 0 0" size="3"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 1);

  // penetration depth
  EXPECT_NEAR(dist, -2, kTolerance);

  // direction
  EXPECT_NEAR(dir[0], 1, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], 0, kTolerance);

  // position
  EXPECT_NEAR(pos[0], 1, kTolerance);
  EXPECT_NEAR(pos[1], 0, kTolerance);
  EXPECT_NEAR(pos[2], 0, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxDepth) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="-1 0 0" size="2.5 2.5 2.5"/>
      <geom name="geom2" type="box" pos="1.5 0 0" size="1 1 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 1);

  EXPECT_NEAR(dist, -1, kTolerance);
  EXPECT_NEAR(dir[0], 1, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], 0, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxDepth2) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="5 5 .1"/>
      <geom name="geom2" type="box" size="1 1 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat + 9;
  mjtNum* xpos = data->geom_xpos + 3;

  xpos[0] = -0.000171208577507291721461757383;
  xpos[1] = -0.000171208577507290908310128019;
  xpos[2] = 1.067119586248553853025100579544;

  xmat[0] = 0.999999966039443077825410455262;
  xmat[1] = -0.000000033960556969622165789148;
  xmat[2] = -0.000260616790777324182967061850;
  xmat[3] = -0.000000033960556972087627235699;
  xmat[4] = 0.999999966039443077825410455262;
  xmat[5] = -0.000260616790777321797722282382;
  xmat[6] = 0.000260616790777324182967061850;
  xmat[7] = 0.000260616790777321797722282382;
  xmat[8] = 0.999999932078886044628518448008;

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -0.033401579411886845, kTolerance);
  EXPECT_THAT(dir[0], MjNear(0, kTolerance, 1e-5));
  EXPECT_THAT(dir[1], MjNear(0, kTolerance, 1e-5));
  EXPECT_THAT(dir[2], MjNear(1, kTolerance, 1e-5));
}

TEST_F(MjGjkTest, BoxBoxDepth3) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.25 0.25 0.05"/>
      <geom name="geom2" type="box" size="0.25 0.25 0.05"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.965925826289068201191412299522;
  xmat[1] = -0.258819045102520739476403832668;
  xmat[2] = 0.000000000000000006339100926609;
  xmat[3] = 0.258819045102520739476403832668;
  xmat[4] = 0.965925826289068201191412299522;
  xmat[5] = -0.000000000000000214827792362716;
  xmat[6] = 0.000000000000000049478422780336;
  xmat[7] = 0.000000000000000209148392896446;
  xmat[8] = 1.000000000000000000000000000000;

  xpos[0] = -0.015346499999999199323474918799;
  xpos[1] = -0.023505500000000002086553152481;
  xpos[2] = -4.562296442400120888294168253196;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.866025403784438707610604524234;
  xmat[1] = -0.499999999999999944488848768742;
  xmat[2] = 0.000000000000000018716705841316;
  xmat[3] = 0.499999999999999944488848768742;
  xmat[4] = 0.866025403784438707610604524234;
  xmat[5] = -0.000000000000000263161736875730;
  xmat[6] = 0.000000000000000115371725704125;
  xmat[7] = 0.000000000000000237263102359077;
  xmat[8] = 1.000000000000000000000000000000;

  xpos[0] = -0.015346499999999797803074130798;
  xpos[1] = -0.023505499999999998617106200527;
  xpos[2] = -4.659230360891631228525966434972;

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -0.003066, kTolerance);
  EXPECT_NEAR(dir[0], 0, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], -1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxSize05) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.5 0.5 0.5"/>
      <geom name="geom2" type="box" size="0.5 0.5 0.5"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.000000000000000;
  xmat[1] = 0.000000047289880;
  xmat[2] = -0.000000050905665;
  xmat[3] = -0.000000047289880;
  xmat[4] = 1.000000000000000;
  xmat[5] = 0.000000017136196;
  xmat[6] = 0.000000050905665;
  xmat[7] = -0.000000017136193;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000000009724202;
  xpos[1] = -0.000000014139289;
  xpos[2] = 7.369161128997803;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.000000000000000;
  xmat[1] = -0.000000013726950;
  xmat[2] = 0.000000008946020;
  xmat[3] = 0.000000013726950;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000000012039017;
  xmat[6] = -0.000000008946020;
  xmat[7] = 0.000000012039017;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000000013445962;
  xpos[1] = -0.000000019194527;
  xpos[2] = 8.264492034912109;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxSize05b) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.5 0.5 0.5"/>
      <geom name="geom2" type="box" size="0.5 0.5 0.5"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.000000000000000;
  xmat[1] = -0.000000008764291;
  xmat[2] = 0.000000386995168;
  xmat[3] = 0.000000008764733;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000001144207772;
  xmat[6] = -0.000000386995168;
  xmat[7] = 0.000001144207772;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000000962082822;
  xpos[1] = -0.000001747370789;
  xpos[2] = 3.469238519668579;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.000000000000000;
  xmat[1] = 0.000000003313412;
  xmat[2] = -0.000000196321196;
  xmat[3] = -0.000000003313673;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000001329654879;
  xmat[6] = 0.000000196321196;
  xmat[7] = 0.000001329654879;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000002897753802;
  xpos[1] = -0.000004625266229;
  xpos[2] = 4.435211658477783;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dir[0], 0, 1e-5);
  EXPECT_NEAR(dir[1], 0, 1e-5);
  EXPECT_NEAR(dir[2], 1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxSize05c) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.5 0.5 0.5"/>
      <geom name="geom2" type="box" size="0.5 0.5 0.5"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.000000000000000;
  xmat[1] = -0.000000000570266;
  xmat[2] = 0.000000168314983;
  xmat[3] = 0.000000000570290;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000000142656916;
  xmat[6] = -0.000000168314983;
  xmat[7] = 0.000000142656916;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000000188941314;
  xpos[1] = -0.000000195227585;
  xpos[2] = 0.497281551361084;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.000000000000000;
  xmat[1] = -0.000000000058608;
  xmat[2] = 0.000001607574859;
  xmat[3] = 0.000000000060496;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000001174638669;
  xmat[6] = -0.000001607574859;
  xmat[7] = 0.000001174638669;
  xmat[8] = 1.000000000000000;

  xpos[0] = 0.000000953407323;
  xpos[1] = -0.000000923845278;
  xpos[2] = 1.493984460830688;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dir[0], 0, 1e-5);
  EXPECT_NEAR(dir[1], 0, 1e-5);
  EXPECT_NEAR(dir[2], 1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxTouching) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 1.859913200000001376466229885409" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 2 1.859913200000001376466229885409" size="1 1 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2);

  ASSERT_EQ(ncons, 0);
  EXPECT_EQ(status.epa_status, -1);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 1.9" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -.1, kTolerance);

  EXPECT_NEAR(dir[0], 0, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], -1, kTolerance);

  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), {-1.0,  1.0, 0.95,
                                                       1.0,  1.0, 0.95,
                                                       1.0, -1.0, 0.95,
                                                      -1.0, -1.0, 0.95}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxBoxMultiCCD2) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="9.5 9.5 1.9" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -.1, kTolerance);

  EXPECT_NEAR(dir[0], 0, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], -1, kTolerance);

  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), { 8.5, 10.0, 0.95,
                                                      10.0, 10.0, 0.95,
                                                      10.0,  8.5, 0.95,
                                                       8.5,  8.5, 0.95}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxBoxMultiCCD3) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" name="geom1" size="5 5 .1" pos="0 0 0"/>
      <geom type="box" name="geom2" size="1 1 1"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat + 9;
  mjtNum* xpos = data->geom_xpos + 3;

  xmat[0] = 0.999999806540386004805043285160;
  xmat[1] = -0.000014738590672566122784237219;
  xmat[2] = 0.000621853651764864637230267874;
  xmat[3] = -0.000621853434269146370175218586;
  xmat[4] = 0.000014756878555191479777952690;
  xmat[5] = 0.999999806540251667819063641218;
  xmat[6] = -0.000014747764440060310685981504;
  xmat[7] = -0.999999999782504311873765345808;
  xmat[8] = 0.000014747710457105431443303178;

  xpos[0] = -0.941218618591869393696924817050;
  xpos[1] = 2.209729011624415928594089564285;
  xpos[2] = 1.095456702630382306296041861060;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD4) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
     <geom name="geom1" size="0.25 0.25 0.05" type="box"/>
     <geom name="geom2" size="0.25 0.25 0.05" type="box"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.500063246694118501700643264485;
  xmat[1] = -0.865988885078582182330819705385;
  xmat[2] = -0.000015036290463686326402846169;
  xmat[3] = 0.865988885208801795201338791230;
  xmat[4] = 0.500063246603650646271432833601;
  xmat[5] = 0.000009541064416582982810641038;
  xmat[6] = -0.000000743359510433135621196039;
  xmat[7] = -0.000017792396065397684211655677;
  xmat[8] = 0.999999999841438502734547455475;

  xpos[0] = -0.015346718925143524800414063236;
  xpos[1] = -0.023500448793229846561336771060;
  xpos[2] = -4.859382717259980388746498647379;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.999999999448633714038692232862;
  xmat[1] = -0.000033207420761195452995305499;
  xmat[2] = -0.000000044925527333868828730462;
  xmat[3] = 0.000033207420790006526530903364;
  xmat[4] = 0.999999999448428988912951353996;
  xmat[5] = 0.000000641458652741046316968134;
  xmat[6] = 0.000000044904226121706672357864;
  xmat[7] = -0.000000641460144248257277838641;
  xmat[8] = 0.999999999999794386695839421009;

  xpos[0] = -0.015347749710384111718197708285;
  xpos[1] = -0.023500601273213628239489025873;
  xpos[2] = -4.958782854594746325460619118530;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 8);
  EXPECT_NEAR(dist, -0.00060425119242707459, kTolerance);

  EXPECT_NEAR(dir[0], 0, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], -1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD5) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
     <geom name="geom1" size="0.25 0.25 0.05" type="box"/>
     <geom name="geom2" size="0.25 0.25 0.05" type="box"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.965955045562010394810670277366;
  xmat[1] = -0.258709898141739669252814337597;
  xmat[2] = -0.000196358811267032467391333017;
  xmat[3] = 0.258709919231419560592399875532;
  xmat[4] = 0.965955055634174608591990818240;
  xmat[5] = 0.000090476785846218643442895324;
  xmat[6] = 0.000166266546411239724218358860;
  xmat[7] = -0.000138196479997660070767120932;
  xmat[8] = 0.999999976628582865068040064216;

  xpos[0] = -0.015381524498156991936914650410;
  xpos[1] = -0.023527931890396581310342938309;
  xpos[2] = -4.559214004409498421921398403356;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.866076536677693908927722077351;
  xmat[1] = -0.499911388413602053581996642606;
  xmat[2] = -0.000190658753729162216972170540;
  xmat[3] = 0.499911409912061843741071243130;
  xmat[4] = 0.866076540935322825021103199106;
  xmat[5] = 0.000086494189368211055798235654;
  xmat[6] = 0.000121885643632020743577087929;
  xmat[7] = -0.000170223074359586521841353202;
  xmat[8] = 0.999999978083999430111816764111;

  xpos[0] = -0.015358668590921718474784363195;
  xpos[1] = -0.023542070504611382203430380855;
  xpos[2] = -4.659108354876987156956147373421;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 8);

  EXPECT_THAT(dist, MjNear(-0.0001077858631973211, kTolerance, 1e-4));
  EXPECT_THAT(dir[0], MjNear(0.00019065, kTolerance, 1e-4));
  EXPECT_THAT(dir[1], MjNear(-0.00008649, kTolerance, 1e-4));
  EXPECT_NEAR(dir[2], -1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD6) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size=".5 .5 .1" pos="0 0 -.1"/>
      <geom name="geom2" type="box" size=".1 .1 .1" pos="0 0 0"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat + 9;
  mjtNum* xpos = data->geom_xpos + 3;

  xmat[0] = -0.412617528992808124677083014831;
  xmat[1] = -0.910903939143411389700588642881;
  xmat[2] = -0.000887930675351447824816819576;
  xmat[3] = 0.910904370383107120368038067681;
  xmat[4] = -0.412617275794986082537718630192;
  xmat[5] = -0.000460143975736545586020798115;
  xmat[6] = 0.000052771423713213129642884969;
  xmat[7] = -0.000998683403024198425301793947;
  xmat[8] = 0.999999499923193035932911243435;

  xpos[0] = 0.413029898172642018217004533653;
  xpos[1] = 0.190777715293135141649827346555;
  xpos[2] = 0.100006658017411736993906856696;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 2);
  EXPECT_THAT(dist, MjNear(-0.00009843, kTolerance, 1e-4));

  EXPECT_THAT(dir[0], MjNear(-0.0008879, kTolerance, 2e-4));
  EXPECT_THAT(dir[1], MjNear(-0.0004601, kTolerance, 1e-3));
  EXPECT_NEAR(dir[2], 0.9999994, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD7) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size=".25 .25 .05"/>
      <geom name="geom2" type="box" size=".25 .25 .05"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.482851932827058627495375731087;
  xmat[1] = -0.875697459006381406787511423317;
  xmat[2] = 0.002823341095436950488190008812;
  xmat[3] = 0.875701084072774249555948244961;
  xmat[4] = 0.482853601927766051815638093103;
  xmat[5] = -0.000102269990141710693382082198;
  xmat[6] = -0.001273702846902712276094815635;
  xmat[7] = 0.002521784120391480209927292933;
  xmat[8] = 0.999996009134990648803409385437;

  xpos[0] = -0.002020740254618143012105280221;
  xpos[1] = -0.022654384848980465422263463893;
  xpos[2] = -4.858542902144324493463045655517;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.999985133805306514176436394337;
  xmat[1] = -0.005293845271528460454113496070;
  xmat[2] = 0.001306663930443651821383665990;
  xmat[3] = 0.005293871114312041943616993223;
  xmat[4] = 0.999985987232967277194006783247;
  xmat[5] = -0.000016319793417504115210251922;
  xmat[6] = -0.001306559226005186893221354794;
  xmat[7] = 0.000023236861241766870316309210;
  xmat[8] = 0.999999146181155484924829579541;

  xpos[0] = -0.011066235018223425159988870803;
  xpos[1] = -0.023114696036485724711662115283;
  xpos[2] = -4.958375812037025376355359185254;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 5);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD8) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size=".25 .25 .05"/>
      <geom name="geom2" type="box" size=".25 .25 .05"/>
    </worldbody>
</mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.000000000000000000000000000000;
  xmat[1] = 0.000000000000000000000000000000;
  xmat[2] = 0.000000000000000000000000000000;
  xmat[3] = 0.000000000000000000000000000000;
  xmat[4] = 1.000000000000000000000000000000;
  xmat[5] = 0.000000000000000000000000000000;
  xmat[6] = 0.000000000000000000000000000000;
  xmat[7] = 0.000000000000000000000000000000;
  xmat[8] = 1.000000000000000000000000000000;

  xpos[0] = -0.015346500000000000765720820084;
  xpos[1] = -0.023505499999999998617106200527;
  xpos[2] = -4.859662640000005140450412000064;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.000000000000000000000000000000;
  xmat[1] = 0.000000000000000000000000000000;
  xmat[2] = 0.000000000000000000000000000000;
  xmat[3] = 0.000000000000000000000000000000;
  xmat[4] = 1.000000000000000000000000000000;
  xmat[5] = -0.000000000000000015361939765351;
  xmat[6] = 0.000000000000000000000000000000;
  xmat[7] = 0.000000000000000015361939765351;
  xmat[8] = 1.000000000000000000000000000000;

  xpos[0] = -0.015346500000000000765720820084;
  xpos[1] = -0.023505499999999998617106200527;
  xpos[2] = -4.958574289672835533338002278470;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD9) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size=".025 .025 .025"/>
      <geom name="geom2" type="box" size=".025 .025 .025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = -0.0000000000000000000000000000000000050579;
  xmat[2] = 0.0000000000000000001927715439853908006818;
  xmat[3] = 0.0000000000000000000000000000000000056403;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000030208514688407265124010;
  xmat[6] = -0.0000000000000000001927715439853908006818;
  xmat[7] = 0.0000000000000000030208514688407265124010;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1071400000000000268807198722242901567370;
  xpos[1] = -0.1928599999999999758948376893386011943221;
  xpos[2] = 0.1749951524564917204607183975895168259740;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = 0.0000000000000000000000000000000037070001;
  xmat[2] = -0.0000000000000000649578747741268744461630;
  xmat[3] = 0.0000000000000000000000000000000064174485;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = 0.0000000000000001558617582226398399563910;
  xmat[6] = 0.0000000000000000649578747741268744461630;
  xmat[7] = -0.0000000000000001558617582226398399563910;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1071400000000000268807198722242901567370;
  xpos[1] = -0.1928599999999999758948376893386011943221;
  xpos[2] = 0.2156259187793853615566774806211469694972;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD10) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;

  xpos[0] = -0.1034859999999999946584949839234468527138;
  xpos[1] = -0.0765140000000000264357424839545274153352;
  xpos[2] = 0.1257628745456405572333835607423679903150;

  xpos = data->geom_xpos + 3;

  xpos[0] = -0.1034859999999999946584949839234468527138;
  xpos[1] = -0.0765140000000000264357424839545274153352;
  xpos[2] = 0.1751399999999999623767621415026951581240;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD11) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = 0.0000000000000000000000000000000000000000;
  xmat[2] = 0.0000000000000000000000000000000000000000;
  xmat[3] = 0.0000000000000000000000000000000000000000;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000013928437397151766790940;
  xmat[6] = 0.0000000000000000000000000000000000000000;
  xmat[7] = 0.0000000000000000013928437397151766790940;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1036549999999999971400654885655967518687;
  xpos[1] = -0.1963450000000000195132798808117513544858;
  xpos[2] = 0.1247685038468368534658736734854755923152;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = 0.0000000000000000000000000000000000000000;
  xmat[2] = 0.0000000000000000000000000000000000000000;
  xmat[3] = 0.0000000000000000000000000000000000000000;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000018885268354605779111974;
  xmat[6] = 0.0000000000000000000000000000000000000000;
  xmat[7] = 0.0000000000000000018885268354605779111974;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1036549999999999971400654885655967518687;
  xpos[1] = -0.1963450000000000195132798808117513544858;
  xpos[2] = 0.1745248497897437800485676007156143896282;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD12) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = 0.0000000000000000000000000000000000000000;
  xmat[2] = 0.0000000000000000000000000000000000000000;
  xmat[3] = 0.0000000000000000000000000000000000000000;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000032154383478277941584027;
  xmat[6] = 0.0000000000000000000000000000000000000000;
  xmat[7] = 0.0000000000000000032154383478277941584027;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = 0.0164299999999999862820843077315657865256;
  xpos[1] = -0.0764300000000000256950016819246229715645;
  xpos[2] = 0.1252706891962387103500731200256268493831;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = 0.0000000000000000000000000000000000000000;
  xmat[2] = 0.0000000000000000000000000000000000000000;
  xmat[3] = 0.0000000000000000000000000000000000000000;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000018997602302052549055743;
  xmat[6] = 0.0000000000000000000000000000000000000000;
  xmat[7] = 0.0000000000000000018997602302052549055743;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = 0.0164299999999999862820843077315657865256;
  xpos[1] = -0.0764300000000000256950016819246229715645;
  xpos[2] = 0.1748374248948718623353215662064030766487;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD13) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.02 0.02 0.02"/>
      <geom name="geom2" type="box" size="0.02 0.02 0.02"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = -0.0000000000000000000000000000001366192847;
  xmat[2] = -0.0000000000000002451235402041944571528177;
  xmat[3] = 0.0000000000000000000000000000001366707863;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = 0.0000000000000000002101047324941978855126;
  xmat[6] = 0.0000000000000002451235402041944571528177;
  xmat[7] = -0.0000000000000000002101047324941978855126;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1000000000000000055511151231257827021182;
  xpos[1] = -0.2000000000000000111022302462515654042363;
  xpos[2] = -0.0809921810760001470441693527391180396080;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xmat[0] = 1.0000000000000000000000000000000000000000;
  xmat[1] = -0.0000000000000000000000000000000740327228;
  xmat[2] = -0.0000000000000002557259745463766308177658;
  xmat[3] = 0.0000000000000000000000000000000775428823;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = 0.0000000000000000137262533997081760161613;
  xmat[6] = 0.0000000000000002557259745463766308177658;
  xmat[7] = -0.0000000000000000137262533997081760161613;
  xmat[8] = 1.0000000000000000000000000000000000000000;

  xpos[0] = -0.1000000000000000055511151231257827021182;
  xpos[1] = -0.2000000000000000111022302462515654042363;
  xpos[2] = -0.0418396695286432432348000531874276930466;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);

  ASSERT_EQ(ncons, 4);

  EXPECT_NEAR(dir[0], 0, kTolerance);
  EXPECT_NEAR(dir[1], 0, kTolerance);
  EXPECT_NEAR(dir[2], 1, kTolerance);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD14) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.02 0.02 0.02"/>
      <geom name="geom2" type="box" size="0.02 0.02 0.02"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = 0.9999999980312528347070610834634862840176;
  xmat[1] = 0.0000179109150612445166097282805983681442;
  xmat[2] = -0.0000601389470252842008382576644009986921;
  xmat[3] = -0.0000179108686851742081238159781664265324;
  xmat[4] = 0.9999999998393023226128661917755380272865;
  xmat[5] = 0.0000007716871733595438989517368948145570;
  xmat[6] = 0.0000601389608372434404702858157243383630;
  xmat[7] = -0.0000007706100310572527002924239115932981;
  xmat[8] = 0.9999999981913554325529958077822811901569;

  xpos[0] = 0.0002051257133161473724877743585182088282;
  xpos[1] = 0.0000051793157380883478958571650152542531;
  xpos[2] = -0.0800031938952457943869944756443146616220;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xmat[0] = 0.9999999606378873195922096783760935068130;
  xmat[1] = -0.0000186818570733572177707156047876679850;
  xmat[2] = -0.0002799557310143530259108346491814245383;
  xmat[3] = 0.0000186853252997592718994551708178164517;
  xmat[4] = 0.9999999997487241110150080203311517834663;
  xmat[5] = 0.0000123858711158191162315369768243122905;
  xmat[6] = 0.0002799554995529331168427344955773605761;
  xmat[7] = -0.0000123911016921886008170612322731862776;
  xmat[8] = 0.9999999607356884201436741932411678135395;

  xpos[0] = 0.0002145111032389043976328218965576866140;
  xpos[1] = -0.0000051338999751368759734112059978095033;
  xpos[2] = -0.0400059009625639144802633495601185131818;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);
  ASSERT_EQ(ncons, 2);
}

TEST_F(MjGjkTest, BoxBoxMultiCCD15) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.5 0.5 0.1"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 1.0;
  xmat[1] = 0.0;
  xmat[2] = 0.0;
  xmat[3] = 0.0;
  xmat[4] = 1.0;
  xmat[5] = 0.0;
  xmat[6] = 0.0;
  xmat[7] = 0.0;
  xmat[8] = 1.0;

  xpos[0] = 0.0;
  xpos[1] = 0.0;
  xpos[2] = 0.0;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 1.0;
  xmat[1] = 1.62423755001306e-10;
  xmat[2] = -1.73500047822017e-05;
  xmat[3] = 1.44241105171083e-10;
  xmat[4] = 1.0;
  xmat[5] = 1.76752037077677e-05;
  xmat[6] = 1.73500047822017e-05;
  xmat[7] = -1.76752037077677e-05;
  xmat[8] = 1.0;

  xpos[0] = 0.0520339831709862;
  xpos[1] = -0.0520339831709862;
  xpos[2] = 0.124986477196217;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, SmallBoxMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" scale=".5 .5 .1"
        vertex="-1 -1 -1
                 1 -1 -1
                 1  1 -1
                 1  1  1
                 1 -1  1
                -1  1 -1
                -1  1  1
                -1 -1  1"/>
      <mesh name="smallbox" scale=".1 .1 .1"
        vertex="-1 -1 -1
                 1 -1 -1
                 1  1 -1
                 1  1  1
                 1 -1  1
                -1  1 -1
                -1  1  1
                -1 -1  1"/>
    </asset>

    <worldbody>
      <geom name="geom2" pos="0 0 .1" size=".1 .1 .1"  type="mesh" mesh="smallbox"/>
      <geom name="geom1" pos="0 0 -.099999999" size=".5 .5 .1"  type="mesh" mesh="box"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  // contact unrecoverable under single precision
  ASSERT_EQ(ncons, sizeof(mjtNum) == 8 ? 1 : 0);
  if (ncons) {
    EXPECT_NEAR(dist, 0, kTolerance);

    // direction
    EXPECT_NEAR(dir[0], 0, kTolerance);
    EXPECT_NEAR(dir[1], 0, kTolerance);
    EXPECT_NEAR(dir[2], 1, kTolerance);

    // position (any point on the [-0.1, 0.1]^2 overlapping face of  smallbox is
    // valid)
    EXPECT_NEAR(pos[0], 0, 0.1);
    EXPECT_NEAR(pos[1], 0, 0.1);
    EXPECT_NEAR(pos[2], 0, kTolerance);
  }
}

TEST_F(MjGjkTest, BoxMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 -.01" size="3 3 .01"/>
      <geom name="geom2" pos="0 0 0.157" euler="0 -90 0" type="mesh" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g2, g1, 0, 1000);
  EXPECT_EQ(model->nmeshpoly, 7);
  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxMesh2) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 -.01" size="3 3 .01"/>
      <geom name="geom2" pos="0 0 -0.001" type="mesh" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g2, g1, 0, 1000);

  ASSERT_EQ(ncons, 5);
}

TEST_F(MjGjkTest, BoxMeshPrune) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 -.01" size="3 3 .01"/>
      <geom name="geom2" pos="0 0 -0.001" type="mesh" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g2, g1, 0, 4);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, MeshMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"
            scale="1 1 .01"/>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="mesh" pos="0 0 -0.01" mesh="box"/>
      <geom name="geom2" pos="0 0 -0.001" type="mesh" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 5);
}

TEST_F(MjGjkTest, MeshMeshPrune) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"
            scale="1 1 .01"/>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="mesh" pos="0 0 -0.01" mesh="box"/>
      <geom name="geom2" pos="0 0 -0.001" type="mesh" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, BoxEdge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" name="box1" size="1 1 1" pos="0 0 2"/>
      <geom type="box" name="box2" size="1 1 1" pos="0 0 4.4" euler="0 90 40"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "box2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 2);
}

TEST_F(MjGjkTest, BoxEdge2) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" name="box1" size="1 1 1"/>
      <geom type="box" name="box2" size="1 1 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.9999979704374094557906005320546682924032;
  xmat[1] = -0.0017789363449516469497385662279498319549;
  xmat[2] = -0.0009457835609818190025430140188689165370;
  xmat[3] = 0.0017817418144636251123996695255868871755;
  xmat[4] = 0.9999939910254954655854930933855939656496;
  xmat[5] = 0.0029737701675293876438233020564894104609;
  xmat[6] = 0.0009404877299599626429629783963548561587;
  xmat[7] = -0.0029754492741947335607277658198199787876;
  xmat[8] = 0.9999951310803708581786963804916013032198;

  xpos[0] = 0.0005578602979296120537716641152314878127;
  xpos[1] = 0.0098645950089783600300830102014515432529;
  xpos[2] = 1.1037596929447945903746131079969927668571;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.0006737475542006746490053537002040684456;
  xmat[1] = -0.0095603689585630827196816028390458086506;
  xmat[2] = 0.9999540716500983084102927023195661604404;
  xmat[3] = -0.1095658134726250898527410981841967441142;
  xmat[4] = 0.9939334085756179604231874691322445869446;
  xmat[5] = 0.0095766296438734854756802405972848646343;
  xmat[6] = -0.9939793149670246297233688892447389662266;
  xmat[7] = -0.1095672335264066821203243762283818796277;
  xmat[8] = -0.0003778293989772788311065632171903416747;

  xpos[0] = -0.0218119359455731035013492657981259981170;
  xpos[1] = 0.9828851949225971829093850828940048813820;
  xpos[2] = 3.0930077345364814789263618877157568931580;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "box2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 2);
}

TEST_F(MjGjkTest, BoxEdgeEdge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" name="box1" size=".15 1 3"/>
      <geom type="box" name="box2" size=".15 1 3"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.9182779243587342321575306414160877466202;
  xmat[1] = -0.0000000000000000000364268564068890756444;
  xmat[2] = 0.3959364262547898638544552341045346111059;
  xmat[3] = -0.0000000000000000000591321577502441383525;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = 0.0000000000000000002291443915550544102718;
  xmat[6] = -0.3959364262547898638544552341045346111059;
  xmat[7] = -0.0000000000000000002338308114719865891040;
  xmat[8] = 0.9182779243587342321575306414160877466202;

  xpos[0] = -1.3241298058948087756903078116010874509811;
  xpos[1] = 0.0000000000000000007148993364299687318184;
  xpos[2] = 2.8141526153588731773425024584867060184479;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.9182779243587342321575306414160877466202;
  xmat[1] = 0.0000000000000000000728398144756416399722;
  xmat[2] = -0.3959364262547898638544552341045346111059;
  xmat[3] = -0.0000000000000000001674060251593158490713;
  xmat[4] = 1.0000000000000000000000000000000000000000;
  xmat[5] = -0.0000000000000000002042889652712837518138;
  xmat[6] = 0.3959364262547898638544552341045346111059;
  xmat[7] = 0.0000000000000000002538761903338069406631;
  xmat[8] = 0.9182779243587342321575306414160877466202;

  xpos[0] = 1.3241298058948089977349127366323955357075;
  xpos[1] = -0.0000000000000000008679606505055748997840;
  xpos[2] = 2.8141526153588731773425024584867060184479;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "box2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 2);
}

TEST_F(MjGjkTest, MeshEdge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="smallbox"
        vertex="-1 -1 -1  1 -1 -1   1  1 -1
                 1  1  1  1 -1  1  -1  1 -1
                -1  1  1 -1 -1  1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="smallbox" name="box1" size="1 1 1" pos="0 0 2"/>
      <geom type="mesh" mesh="smallbox" name="box2" size="1 1 1" pos="0 0 4.4" euler="0 90 40"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "box2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 2);
}

TEST_F(MjGjkTest, MeshEdge2) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="meshbox"
            vertex="-1 -1 -1
                     1 -1 -1
                     1  1 -1
                     1  1  1
                     1 -1  1
                    -1  1 -1
                    -1  1  1
                    -1 -1  1"/>
    </asset>

    <worldbody>
      <geom type="box" name="floor" size="6 6 2" pos="0 0 -1"/>
      <body pos="2 0 1" euler="0 5 0">
        <freejoint/>
        <geom type="mesh" name="meshbox" mesh="meshbox"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "meshbox");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, EllipsoidEllipsoidPenetrating) {
  MjModelPtr model = LoadModelFromString(kEllipsoidXml);
  MjDataPtr data = MakeData(model);
  mj_resetDataKeyframe(model.get(), data.get(), 0);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -0.00022548856248122027, kTolerance);
}

TEST_F(MjGjkTest, EllipsoidEllipsoid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="ellipsoid" pos="1.5 0 -.5" size=".15 .30 .20"/>
      <geom name="geom2" type="ellipsoid" pos="1.5 .5 .5" size=".10 .10 .15"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum dist = GeomDist(model, data, geom1, geom2, nullptr, nullptr);

  EXPECT_NEAR(dist, 0.7542, .0001);
}

TEST_F(MjGjkTest, EllipsoidEllipsoidSlowConvergence) {
  // This example takes gjkIntersect 30 iterations to converge, thus failing to
  // detect collision if opt.ccd_iterations is set to 35. With gjkIntersect
  // disabled, GJK succeeds in 24 iterations.
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="ellipsoid" size=".05 .07 .15"/>
      <geom name="geom2" type="ellipsoid" size=".05 .07 .15"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xmat = data->geom_xmat;
  mjtNum* xpos = data->geom_xpos;

  xmat[0] = 0.98799245191271645172;
  xmat[1] = -0.13055117805648397411;
  xmat[2] = -0.08262750674904573156;
  xmat[3] = -0.03910798838001148386;
  xmat[4] = 0.30606703783583483203;
  xmat[5] = -0.95120635699897815307;
  xmat[6] = 0.14947066671547387662;
  xmat[7] = 0.94301609650019369013;
  xmat[8] = 0.29728632920665870598;

  xpos[0] = 0.44826351361072502844;
  xpos[1] = -0.10441423315171689812;
  xpos[2] = 0.21216466278453649519;

  xmat = data->geom_xmat + 9;
  xpos = data->geom_xpos + 3;

  xmat[0] = 0.53301656081215953442;
  xmat[1] = -0.59335516633486651283;
  xmat[2] = -0.60317741377119038493;
  xmat[3] = -0.39469414877746378245;
  xmat[4] = 0.45620167535223110633;
  xmat[5] = -0.79755661888461082398;
  xmat[6] = 0.74840488695679441289;
  xmat[7] = 0.66318148194106107951;
  xmat[8] = 0.00896923573374096283;

  xpos[0] = 0.39694137187624300989;
  xpos[1] = 0.00961542646741688108;
  xpos[2] = 0.29832742817753182818;

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  Penetration(status, dist, dir, pos, model, data, geom1, geom2);

  EXPECT_LT(dist, 0.0);
  EXPECT_NEAR(dist, 0.0, kTolerance);
}

TEST_F(MjGjkTest, BoxBox) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="-1.5 .5 0" size="1 1 1"/>
      <geom name="geom2" type="box" pos="1.5 0 0" size="1 1 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum dist = GeomDist(model, data, geom1, geom2, nullptr, nullptr);

  EXPECT_EQ(dist, 1);
}

TEST_F(MjGjkTest, BoxBoxLarge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 0.25" size="50 50 0.25"/>
      <geom name="geom2" type="box" pos="0 0 0.60" size="0.1 0.1 0.1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjtNum* xpos = data->geom_xpos + 3;
  mjtNum* xmat = data->geom_xmat + 9;

  xpos[0] = -0.000000000043537;
  xpos[1] = -0.000000000012973;
  xpos[2] = 0.599245786666870;

  xmat[0] = 1.000000000000000;
  xmat[1] = -0.000000000004566;
  xmat[2] = -0.000000000167641;
  xmat[3] = 0.000000000004566;
  xmat[4] = 1.000000000000000;
  xmat[5] = -0.000000000017877;
  xmat[6] = 0.000000000167641;
  xmat[7] = 0.000000000017877;
  xmat[8] = 1.000000000000000;

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncon = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  EXPECT_NEAR(dist, -0.000754, kTolerance);
  EXPECT_EQ(ncon, 4);
}

TEST_F(MjGjkTest, LongBox) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="long_box"
            vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"
            scale=".6 .03 .03"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="box" size="1 1 .3" pos="0 0 -.3"/>
      <geom name="geom2" type="mesh" mesh="long_box" pos="0 0 .02" euler="0 0 40"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -0.01, kTolerance);

  EXPECT_THAT(dir[0], MjNear(0, kTolerance, 1e-5));
  EXPECT_THAT(dir[1], MjNear(0, kTolerance, 1e-5));
  EXPECT_THAT(dir[2], MjNear(1, kTolerance, kTolerance));

  // multicontact
  ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);
  ASSERT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, EllipsoidEllipsoidIntersect) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="ellipsoid" pos="1.5 0 -.5" size=".15 .30 .20"/>
      <geom name="geom2" type="ellipsoid" pos="1.5 .5 .5" size=".10 .10 .15"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 15);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -14.245732934582151, MjTol(kTolerance, 1e-5));
}

TEST_F(MjGjkTest, CapsuleCapsule) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="capsule" pos="-.3 .2 -.4" size=".15 .30"/>
      <geom name="geom2" type="capsule" pos=".3 .2 .4" size=".10 .10"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int geom1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int geom2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum dist = GeomDist(model, data, geom1, geom2, nullptr, nullptr);

  EXPECT_NEAR(dist, 0.4711, .0001);
}

TEST_F(MjGjkTest, CylinderBoxMargin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <statistic meansize="0.15"/>
    <option>
      <flag gravity="disable" multiccd="disable"/>
    </option>

    <worldbody>
      <body pos="0 0 .265">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" margin="0.1" gap="0.1"/>
      </body>
      <body mocap="true">
        <geom name="geom2" type="cylinder" size=".2 .2"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // margin=0.1 means forces generated when dist<0.1
  // the contact at dist~0.015 is within margin, so forces are generated
  EXPECT_EQ(data->ncon, 1);
  EXPECT_GE(data->contact[0].efc_address, 0);
}

TEST_F(MjGjkTest, BoxEdgeFlipped) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" pos="1.10164554 -0.11389316 0.74"
            quat="-0.348312918 0 0 0.937378318" type="box" size="0.65 0.48 0.04"/>
      <geom name="geom2" type="box" size="0.1 1.2 1.4" pos="1.4 0 1.425"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  ASSERT_EQ(ncons, 2);
  EXPECT_NEAR(status.x1[0], 1.907368, kTolerance);
  EXPECT_NEAR(status.x1[1], -0.052973, kTolerance);
  EXPECT_NEAR(status.x1[2], 0.700000, kTolerance);
  EXPECT_NEAR(status.x2[0], 1.30000, kTolerance);
  EXPECT_NEAR(status.x2[1], -0.052973, kTolerance);
  EXPECT_NEAR(status.x2[2], 0.700000, kTolerance);
}

TEST_F(MjGjkTest, ThinBoxGrazing) {
  // thin boxes separated by +7.6933e-7, below the ccd_tolerance of 1e-6
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.015108062742470629 0.00020768330936692047 0.00033708479398415232"/>
      <geom name="geom2" type="box" size="0.0022889234034685867 0.00039019141130537644 0.00066943745945985227"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = 0.41747699392674587;
  xmat[1] = 0.7445704580736479;
  xmat[2] = -0.5208913442416622;
  xmat[3] = 0.735898261710082;
  xmat[4] = -0.6133129289446175;
  xmat[5] = -0.28688150794614425;
  xmat[6] = -0.5330728917830726;
  xmat[7] = -0.263556605216739;
  xmat[8] = -0.8039721437292835;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xpos[0] = -0.0045224108719272319;
  xpos[1] = -0.0099558526994404459;
  xpos[2] = 0.00749543399075938;

  xmat[0] = -0.11330328891322261;
  xmat[1] = 0.6014637637569688;
  xmat[2] = -0.7908246996703808;
  xmat[3] = 0.9557435215667675;
  xmat[4] = 0.2834865022971088;
  xmat[5] = 0.07867479900517776;
  xmat[6] = 0.2715081687621951;
  xmat[7] = -0.7469114699230777;
  xmat[8] = -0.6069650487406557;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  mjtNum dist = mj_geomDistance(model.get(), data.get(), g1, g2, 1, nullptr);

  // below ccd_tolerance, but still positive distance
  EXPECT_GE(dist, 0);
  EXPECT_LE(dist, model->opt.ccd_tolerance);
}

TEST_F(MjGjkTest, CylinderBoxBackFaceEdge) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="cylinder" size="0.05 0.15"/>
      <geom name="geom2" type="box" size="0.05 0.15 0.07"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos1 = data->geom_xpos;
  mjtNum* xmat1 = data->geom_xmat;

  xpos1[0] = 0.23191021521759686;
  xpos1[1] = -0.43472537698525593;
  xpos1[2] = 0.39661285409646718;

  xmat1[0] = -0.32701292892485478;
  xmat1[1] = -0.12329280998746245;
  xmat1[2] = -0.93694259553154202;
  xmat1[3] = 0.49240015314737545;
  xmat1[4] = -0.8684624433071888;
  xmat1[5] = -0.057576677095404399;
  xmat1[6] = -0.80660066544506925;
  xmat1[7] = -0.48017899534475972;
  xmat1[8] = 0.34470784576691638;

  mjtNum* xpos2 = data->geom_xpos + 3;
  mjtNum* xmat2 = data->geom_xmat + 9;

  xpos2[0] = 0.11445187572345047;
  xpos2[1] = -0.40755363814662998;
  xpos2[2] = 0.33704040755212905;

  xmat2[0] = 0.34942077365670599;
  xmat2[1] = -0.93671954300419125;
  xmat2[2] = -0.021485359926434244;
  xmat2[3] = -0.13775655212039611;
  xmat2[4] = -0.028678169746457732;
  xmat2[5] = -0.99005085471802623;
  xmat2[6] = 0.92678382338334431;
  xmat2[7] = 0.34890408471958706;
  xmat2[8] = -0.13906000280791611;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status1, status2;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status1, dist, dir, pos, model, data, g1, g2, 0, 4);
  ASSERT_EQ(ncons, 2);

  ncons = Penetration(status2, dist, dir, pos, model, data, g1, g2);
  ASSERT_EQ(ncons, 1);

  EXPECT_NEAR(status1.x1[3], status2.x1[0], 2 * kTolerance);
}

}  // namespace

TEST_F(MjGjkTest, CorrectFaceMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="wedge_box"
            vertex="-0.5 -0.1  0.2   0.5 -0.1  0.2   0.5  0.1  0.2  -0.5  0.1  0.2
                    -0.5 -0.1  0.01  0.5 -0.1  0.0   0.5  0.1  0.0  -0.5  0.1  0.01
                     0.4 -0.1  0.0   0.4  0.1  0.0"/>
    </asset>
    <worldbody>
      <geom name="floor" type="box" size="2 2 0.05"/>
      <geom name="wedge" type="mesh" mesh="wedge_box"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "wedge");

  mjtNum* xpos2 = data->geom_xpos + 3;
  mjtNum* xmat2 = data->geom_xmat + 9;

  xpos2[0] = 0.0041063283594611893;
  xpos2[1] = -2.8315597012716811e-17;
  xpos2[2] = 0.14782321875543883;

  xmat2[0] = -0.0054978550657821534;
  xmat2[1] = 1.5171835118936141e-17;
  xmat2[2] = 0.99998488668063767;
  xmat2[3] = -1.297280742604275e-17;
  xmat2[4] = -1.0000000000000062;
  xmat2[5] = 1.5100740726228752e-17;
  xmat2[6] = 0.99998488668063767;
  xmat2[7] = -1.2889589679962258e-17;
  xmat2[8] = 0.0054978550657821534;

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 1000);

  EXPECT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, CapsuleMeshMultiCCDShallow) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh"/>
      <geom name="cap" type="capsule" pos="0 0 0.14" zaxis="1 0 0" size="0.05 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "cap");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 2);

  // depth = 0.01 < radius = 0.05: shallow GJK only
  EXPECT_EQ(ncons, 2);
  EXPECT_NEAR(dist, -0.01, kTolerance);
  EXPECT_EQ(status.epa_iterations, 0);
}

TEST_F(MjGjkTest, CapsuleMeshMultiCCDEpaFallback) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh"/>
      <geom name="cap" type="capsule" pos="0 0 0.08" zaxis="1 0 0" size="0.05 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "cap");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 2);

  // depth = 0.07 > radius = 0.05: EPA fallback
  EXPECT_EQ(ncons, 2);
  EXPECT_NEAR(dist, -0.07, kTolerance);
  EXPECT_GT(status.epa_iterations, 0);
}

TEST_F(MjGjkTest, CapsuleMeshMultiCCDReversed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh"/>
      <geom name="cap" type="capsule" pos="0 0 0.14" zaxis="1 0 0" size="0.05 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "cap");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 2);

  ASSERT_EQ(ncons, 2);
  EXPECT_NEAR(dist, -0.01, kTolerance);
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance),
                             {-0.2, 0.0, 0.095, 0.2, 0.0, 0.095}));
}

TEST_F(MjGjkTest, CapsuleCylinderMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="cyl" type="cylinder" size="1.0 0.1"/>
      <geom name="cap" type="capsule" pos="0 0 0.14" zaxis="1 0 0" size="0.05 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int cyl = mj_name2id(model.get(), mjOBJ_GEOM, "cyl");
  int cap = mj_name2id(model.get(), mjOBJ_GEOM, "cap");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;

  // cylinder as obj1, capsule as obj2
  int ncons1 = Penetration(status, dist, dir, pos, model, data, cyl, cap, 0, 2);
  EXPECT_EQ(ncons1, 2);
  EXPECT_NEAR(dist, -0.01, kTolerance);

  // capsule as obj1, cylinder as obj2
  int ncons2 = Penetration(status, dist, dir, pos, model, data, cap, cyl, 0, 2);
  EXPECT_EQ(ncons2, 2);
  EXPECT_NEAR(dist, -0.01, kTolerance);
}

TEST_F(MjGjkTest, CapsuleMeshMultiCCDClipped) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh"/>
      <geom name="cap" type="capsule" pos="0 0 0.14" zaxis="1 0 0" size="0.05 2.0"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "cap");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 2);

  // capsule [-2, 2] clipped to mesh face boundary [-1, 1]
  ASSERT_EQ(ncons, 2);
  EXPECT_NEAR(dist, -0.01, kTolerance);
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance),
                             {-1.0, 0.0, 0.095, 1.0, 0.0, 0.095}));
}

TEST_F(MjGjkTest, CapsuleMeshMultiCCDTilted) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh"/>
      <geom name="cap" type="capsule" pos="0 0 0.14" zaxis="1 0 0.2" size="0.05 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "cap");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 2);

  // tilted capsule should only generate 1 contact at the lower end
  EXPECT_EQ(ncons, 1);
}

TEST_F(MjGjkTest, Polytope3CoplanarSupport) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cone"
            vertex="1 0 0   0.70710678 0.70710678 0   0 1 0  -0.70710678 0.70710678 0
                   -1 0 0  -0.70710678 -0.70710678 0  0 -1 0  0.70710678 -0.70710678 0
                    0 0 0.5"/>
    </asset>
    <worldbody>
      <geom name="g1" type="mesh" mesh="cone" euler="13 27 41"/>
      <geom name="g2" type="sphere" size="0.1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // get world-space coordinates of base vertices 0, 2, 4 and apex 8
  auto world_vert = [&](int v, mjtNum out[3]) {
    mjtNum local[3] = {model->mesh_vert[3 * v + 0], model->mesh_vert[3 * v + 1],
                       model->mesh_vert[3 * v + 2]};
    mju_mulMatVec3(out, data->geom_xmat, local);
    mju_addTo3(out, data->geom_xpos);
  };
  mjtNum w0[3], w2[3], w4[3], w8[3];
  world_vert(0, w0);
  world_vert(2, w2);
  world_vert(4, w4);
  world_vert(8, w8);

  // shrink sphere to a point on the base face (within tolerance of the plane)
  model->geom_size[3] = 0;
  for (int k = 0; k < 3; k++) {
    mjtNum base = 0.2 * w0[k] + 0.24 * w2[k] + 0.56 * w4[k];
    data->geom_xpos[3 + k] = base + 1e-8 * (w8[k] - base);
  }

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, 0, 1);
  EXPECT_EQ(ncons, 0);
  EXPECT_NE(status.epa_status, mjEPA_SUCCESS);
}

TEST_F(MjGjkTest, GeomDistanceBelowTolerance) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="g1" type="box" size="0.1 0.1 0.1"/>
      <geom name="g2" type="ellipsoid" pos="0 0 0.2000005" size="0.1 0.1 0.1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum fromto[6] = {0};
  mjtNum dist = mj_geomDistance(model.get(), data.get(), 0, 1, 1.0, fromto);
  EXPECT_NEAR(dist, 5e-7, MjTol(1e-9, 5e-6));
  EXPECT_NEAR(fromto[2], 0.1, MjTol(1e-6, 1e-5));
  EXPECT_NEAR(fromto[5], 0.1000005, MjTol(1e-6, 1e-5));
}

TEST_F(MjGjkTest, MultiCCDEdgeClipEmpty) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="box1" type="box" size="0.2 0.2 0.01"
            pos="0.1316 0.7377 -0.0022" euler="-9 -15 15"/>
      <geom name="box2" type="box" size="0.001 0.5 0.5"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "box2");

  mjCCDStatus status1, status2;
  std::vector<mjtNum> dir1, pos1, dir2, pos2;
  mjtNum dist1, dist2;
  int ncons1 = Penetration(status1, dist1, dir1, pos1, model, data, g1, g2);
  ASSERT_EQ(ncons1, 1);

  int ncons2 =
      Penetration(status2, dist2, dir2, pos2, model, data, g1, g2, 0, 4);
  ASSERT_EQ(ncons2, 1);

  EXPECT_MJTNUM_EQ(dist1, dist2);
  EXPECT_MJTNUM_EQ(dir1[0], dir2[0]);
  EXPECT_MJTNUM_EQ(dir1[1], dir2[1]);
  EXPECT_MJTNUM_EQ(dir1[2], dir2[2]);
}

TEST_F(MjGjkTest, TiltedBoxBoxMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="box" pos="0 0 -0.05" size="1 1 0.05"/>
      <geom name="box" type="box" pos="0 0 0.08" euler="5 0 0" size="0.1 0.1 0.1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "box");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "floor");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 8);

  // Box (half-size 0.1) tilted 5 deg about x at z = 0.08:
  // lower edge (y_local = -0.1, z_local = -0.1): y = -0.0909039, z = -0.0283350
  // upper edge (y_local = +0.1, z_local = -0.1): y =  0.1083350, z = -0.0109038
  const double cos5 = mju_cos(5.0 * mjPI / 180.0);
  const double sin5 = mju_sin(5.0 * mjPI / 180.0);
  const double y_lower = -0.1 * cos5 + 0.1 * sin5;
  const double z_lower = 0.08 - 0.1 * sin5 - 0.1 * cos5;
  const double y_upper = 0.1 * cos5 + 0.1 * sin5;
  const double z_upper = 0.08 + 0.1 * sin5 - 0.1 * cos5;

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(status.dist[0], z_upper, kTolerance);
  EXPECT_NEAR(status.dist[1], z_upper, kTolerance);
  EXPECT_NEAR(status.dist[2], z_lower, kTolerance);
  EXPECT_NEAR(status.dist[3], z_lower, kTolerance);

  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), {
      -0.1, y_upper, 0.5 * z_upper,
       0.1, y_upper, 0.5 * z_upper,
       0.1, y_lower, 0.5 * z_lower,
      -0.1, y_lower, 0.5 * z_lower}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxBoxMarginShallowMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 2.05" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  // Shallow margin band: distance = +0.05, margin = 0.1 -> depth = -0.05
  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0.1, 8);
  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.05, kTolerance);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(status.dist[i], -0.05, kTolerance);
    EXPECT_NEAR(dir[3 * i + 0], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 1], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 2], -1, kTolerance);
  }
  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), {-1.0,  1.0, 1.025,
                                                       1.0,  1.0, 1.025,
                                                       1.0, -1.0, 1.025,
                                                      -1.0, -1.0, 1.025}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxBoxMarginTouchingMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 2.0" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  // Touching faces: distance = 0.0, margin = 0.1 -> depth = -0.1
  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0.1, 8);
  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.1, kTolerance);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(status.dist[i], -0.1, kTolerance);
    EXPECT_NEAR(dir[3 * i + 0], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 1], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 2], -1, kTolerance);
  }
  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), {-1.0,  1.0, 1.0,
                                                       1.0,  1.0, 1.0,
                                                       1.0, -1.0, 1.0,
                                                      -1.0, -1.0, 1.0}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxBoxMarginDeepMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" pos="0 0 1.9" size="1 1 1"/>
      <geom name="geom2" type="box" pos="0 0 0" size="10 10 1"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  // Deep penetration: distance = -0.1, margin = 0.1 -> depth = -0.2
  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0.1, 8);
  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.2, kTolerance);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(status.dist[i], -0.2, kTolerance);
    EXPECT_NEAR(dir[3 * i + 0], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 1], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 2], -1, kTolerance);
  }
  // clang-format off
  EXPECT_THAT(pos, Pointwise(DoubleNear(kTolerance), {-1.0,  1.0, 0.95,
                                                       1.0,  1.0, 0.95,
                                                       1.0, -1.0, 0.95,
                                                      -1.0, -1.0, 0.95}));
  // clang-format on
}

TEST_F(MjGjkTest, BoxEdgeMarginMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" pos="1.10164554 -0.11389316 0.74"
            quat="-0.348312918 0 0 0.937378318" type="box" size="0.65 0.48 0.04"/>
      <geom name="geom2" type="box" size="0.1 1.2 1.4" pos="1.45 0 1.425"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0.1, 8);

  // Same geometry as BoxEdgeFlipped (uninflated x1[0] = 1.907368) with geom2
  // shifted +0.05 in x (uninflated x2[0] = 1.35) and margin = 0.1 (0.05 each).
  ASSERT_EQ(ncons, 2);
  EXPECT_NEAR(status.x1[0], 1.907368 + 0.05, kTolerance);
  EXPECT_NEAR(status.x1[1], -0.052973, kTolerance);
  EXPECT_NEAR(status.x1[2], 0.700000, kTolerance);
  EXPECT_NEAR(status.x2[0], 1.35000 - 0.05, kTolerance);
  EXPECT_NEAR(status.x2[1], -0.052973, kTolerance);
  EXPECT_NEAR(status.x2[2], 0.700000, kTolerance);
}

TEST_F(MjGjkTest, CylinderBoxMarginMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <statistic meansize="0.15"/>
    <option>
      <flag gravity="disable"/>
    </option>

    <worldbody>
      <body pos="0 0 .265">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" margin="0.1" gap="0.1"/>
      </body>
      <body mocap="true">
        <geom name="geom2" type="cylinder" size=".2 .2"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  ASSERT_EQ(data->ncon, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(data->contact[i].dist, 0.015, kTolerance);
    EXPECT_NEAR(data->contact[i].pos[2], 0.2075, kTolerance);
    EXPECT_GE(data->contact[i].efc_address, 0);
  }

  // explicit <pair margin=...> override
  static constexpr char xml_pair[] = R"(
  <mujoco>
    <statistic meansize="0.15"/>
    <option>
      <flag gravity="disable"/>
    </option>

    <worldbody>
      <body pos="0 0 .265">
        <freejoint/>
        <geom name="box" type="box" size=".05 .05 .05" margin="0.005"/>
      </body>
      <body mocap="true">
        <geom name="cyl" type="cylinder" size=".2 .2"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="box" geom2="cyl" margin="0.1" gap="0.1"/>
    </contact>
  </mujoco>)";

  MjModelPtr model_pair = LoadModelFromString(xml_pair);
  MjDataPtr data_pair = MakeData(model_pair);
  mj_forward(model_pair.get(), data_pair.get());

  ASSERT_EQ(data_pair->ncon, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(data_pair->contact[i].dist, 0.015, kTolerance);
    EXPECT_NEAR(data_pair->contact[i].pos[2], 0.2075, kTolerance);
    EXPECT_GE(data_pair->contact[i].efc_address, 0);
  }

  // cylinder side-edge on box face with margin (shallow and deep penetration)
  static constexpr char xml_edge[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="box" size="1 1 0.1" margin="0.05"/>
      <body pos="0 0 0.17">
        <freejoint/>
        <geom name="cyl" type="cylinder" zaxis="1 0 0" size="0.05 0.2" margin="0.05"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model_edge = LoadModelFromString(xml_edge);
  MjDataPtr data_edge = MakeData(model_edge);

  // shallow (within margin)
  mj_forward(model_edge.get(), data_edge.get());
  ASSERT_EQ(data_edge->ncon, 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(data_edge->contact[i].dist, 0.02, kTolerance);
    EXPECT_NEAR(data_edge->contact[i].pos[2], 0.11, kTolerance);
  }

  // deep penetration (triggers EPA)
  data_edge->qpos[2] = 0.08;
  mj_forward(model_edge.get(), data_edge.get());
  ASSERT_EQ(data_edge->ncon, 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(data_edge->contact[i].dist, -0.07, kTolerance);
    EXPECT_NEAR(data_edge->contact[i].pos[2], 0.065, kTolerance);
  }
}

TEST_F(MjGjkTest, CapsuleMeshMarginMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="boxmesh"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.1   1 -1  0.1   1  1  0.1  -1  1  0.1"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="boxmesh" margin="0.05"/>
      <body pos="0 0 0.17">
        <freejoint/>
        <geom name="cap" type="capsule" zaxis="1 0 0" size="0.05 0.2" margin="0.05"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // shallow penetration (within margin + capsule radius)
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(data->contact[i].dist, 0.02, kTolerance);
    EXPECT_NEAR(data->contact[i].pos[2], 0.11, kTolerance);
  }

  // deep penetration (capsule centerline penetrates mesh face, triggers EPA)
  data->qpos[2] = 0.08;
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 2);
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(data->contact[i].dist, -0.07, kTolerance);
    EXPECT_NEAR(data->contact[i].pos[2], 0.065, kTolerance);
  }
}

TEST_F(MjGjkTest, MeshMeshMarginMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" vertex="-1 -1 -1 1 -1 -1 1 1 -1 1 1 1 1 -1 1 -1 1 -1 -1 1 1 -1 -1 1"
            scale="1 1 .01"/>
      <mesh name="pentaprism"
            vertex="1 0 0 0.309 0.951 0 -0.809 0.588 0 -0.809 -0.588 0 0.309 -0.951 0
                    1 0 1 0.309 0.951 1 -0.809 0.588 1 -0.809 -0.588 1 0.309 -0.951 1"
            scale=".2 .2 .1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="mesh" pos="0 0 -0.01" mesh="box"/>
      <geom name="geom2" type="mesh" pos="0 0 -0.001" mesh="pentaprism"/>
      <geom name="geom3" type="mesh" pos="0 0 0.19" euler="0 90 0" mesh="pentaprism"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");
  int g3 = mj_name2id(model.get(), mjOBJ_GEOM, "geom3");

  // deep penetration face-face:
  // distance = -0.001, margin = 0.1 -> depth = -0.101
  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0.1, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.101, kTolerance);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(status.dist[i], -0.101, kTolerance);
    EXPECT_NEAR(dir[3 * i + 0], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 1], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 2], 1, kTolerance);
    EXPECT_NEAR(pos[3 * i + 2], -0.0005, kTolerance);
  }

  // deep penetration edge-face: distance = -0.01, margin = 0.1 -> depth = -0.11
  ncons = Penetration(status, dist, dir, pos, model, data, g1, g3, 0.1, 4);
  ASSERT_EQ(ncons, 2);
  EXPECT_NEAR(dist, -0.11, kTolerance);
  for (int i = 0; i < 2; ++i) {
    EXPECT_NEAR(status.dist[i], -0.11, kTolerance);
    EXPECT_NEAR(dir[3 * i + 0], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 1], 0, kTolerance);
    EXPECT_NEAR(dir[3 * i + 2], 1, kTolerance);
    EXPECT_NEAR(pos[3 * i + 2], -0.005, kTolerance);
  }
}

TEST_F(MjGjkTest, TiltedBoxMeshMarginMultiCCD) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="floor"
            vertex="-1 -1 -0.1   1 -1 -0.1   1  1 -0.1  -1  1 -0.1
                    -1 -1  0.0   1 -1  0.0   1  1  0.0  -1  1  0.0"/>
    </asset>
    <worldbody>
      <geom name="floor" type="mesh" mesh="floor" margin="0.05"/>
      <body pos="0 0 0.13" euler="5 0 0">
        <freejoint/>
        <geom name="box" type="box" size="0.1 0.1 0.1" margin="0.05"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // Box (half-size 0.1) tilted 5 deg about x at z = 0.13:
  // lower edge (y_local = -0.1, z_local = -0.1): y = -0.0909039, z = 0.0216650
  // upper edge (y_local = +0.1, z_local = -0.1): y =  0.1083350, z = 0.0390962
  const mjtNum cos5 = mju_cos(5.0 * mjPI / 180.0);
  const mjtNum sin5 = mju_sin(5.0 * mjPI / 180.0);
  const mjtNum y_lower = -0.1 * cos5 + 0.1 * sin5;
  const mjtNum z_lower = 0.13 - 0.1 * sin5 - 0.1 * cos5;
  const mjtNum y_upper = 0.1 * cos5 + 0.1 * sin5;
  const mjtNum z_upper = 0.13 + 0.1 * sin5 - 0.1 * cos5;

  ASSERT_EQ(data->ncon, 4);
  int n_lower = 0, n_upper = 0;
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(mju_abs(data->contact[i].pos[0]), 0.1, kTolerance);
    if (data->contact[i].pos[1] < 0) {
      n_lower++;
      EXPECT_NEAR(data->contact[i].pos[1], y_lower, kTolerance);
      EXPECT_NEAR(data->contact[i].pos[2], 0.5 * z_lower, kTolerance);
      EXPECT_NEAR(data->contact[i].dist, z_lower, kTolerance);
    } else {
      n_upper++;
      EXPECT_NEAR(data->contact[i].pos[1], y_upper, kTolerance);
      EXPECT_NEAR(data->contact[i].pos[2], 0.5 * z_upper, kTolerance);
      EXPECT_NEAR(data->contact[i].dist, z_upper, kTolerance);
    }
  }
  EXPECT_EQ(n_lower, 2);
  EXPECT_EQ(n_upper, 2);
}

TEST_F(MjGjkTest, CapsuleEllipsoidSmallSimplex) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="capsule" size="0.042 0.14"/>
      <geom name="geom2" type="ellipsoid" size="0.095 0.14 0.14"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xmat[0] = -0.07982016;
  xmat[1] = 0.9632121;
  xmat[2] = 0.25661483;
  xmat[3] = -0.9521492;
  xmat[4] = 0.0025194245;
  xmat[5] = -0.30562323;
  xmat[6] = -0.2950265;
  xmat[7] = -0.2687305;
  xmat[8] = 0.91692054;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xpos[0] = -0.11622479;
  xpos[1] = -0.078617394;
  xpos[2] = -0.2301344;

  xmat[0] = -0.96931815;
  xmat[1] = 0.026906298;
  xmat[2] = 0.24433327;
  xmat[3] = 0.05312106;
  xmat[4] = -0.94757426;
  xmat[5] = 0.31508982;
  xmat[6] = 0.24000177;
  xmat[7] = 0.31840146;
  xmat[8] = 0.91707146;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2);
  ASSERT_EQ(ncons, 1);
  EXPECT_LT(dist, 0);
  EXPECT_NEAR(dist, -1.82696e-7, MjTol(1e-9, 1e-7));
}

// Stacked boxes with 2.5e-10 rad tilt where polytope3 requires pole-swap and
// multicontact recovers all 4 face contacts.
TEST_F(MjGjkTest, BoxBoxMultiCCD16) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xpos[0] = -0.0434859991;
  xpos[1] = -0.196513996;
  xpos[2] = 0.12576285;

  xmat[0] = 1.0;
  xmat[1] = 0.0;
  xmat[2] = 2.50575921e-10;
  xmat[3] = 0.0;
  xmat[4] = 1.0;
  xmat[5] = 0.0;
  xmat[6] = -2.50575921e-10;
  xmat[7] = 0.0;
  xmat[8] = 1.0;

  xpos = data->geom_xpos + 3;

  xpos[0] = -0.0434859991;
  xpos[1] = -0.196513996;
  xpos[2] = 0.175140008;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  for (int i = 0; i < 4; ++i) {
    EXPECT_NEAR(status.dist[i], -0.000622842, MjTol(kTolerance, 1e-5));
  }
}

// Shallow box-box edge contact (depth ~6e-7) that previously failed in float32
// due to mjMINDIST2/3 = 1e-10f rejecting the initial polytope face.
TEST_F(MjGjkTest, BoxBoxMultiCCD17) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xpos[0] = -0.424999505;
  xpos[1] = 0.375003099;
  xpos[2] = 0.124994218;

  xmat[0] = 1.0;
  xmat[1] = 1.28095244e-05;
  xmat[2] = -1.73168264e-05;
  xmat[3] = -1.281001e-05;
  xmat[4] = 1.0;
  xmat[5] = -2.80516215e-05;
  xmat[6] = 1.73164663e-05;
  xmat[7] = 2.80518434e-05;
  xmat[8] = 1.0;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xpos[0] = -0.424999416;
  xpos[1] = 0.424999982;
  xpos[2] = 0.174989387;

  xmat[0] = 1.0;
  xmat[1] = -9.61651585e-06;
  xmat[2] = 2.00568993e-05;
  xmat[3] = 9.6184167e-06;
  xmat[4] = 1.0;
  xmat[5] = -9.47636436e-05;
  xmat[6] = -2.00559862e-05;
  xmat[7] = 9.47638327e-05;
  xmat[8] = 1.0;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 2);
  EXPECT_LT(dist, 0);
  EXPECT_NEAR(dist, -6.07466e-7, MjTol(1e-9, 2e-7));
}

// Stacked boxes with tiny opposite tilts where polytope3 falls back to a
// 4-face tetrahedron and multicontact recovers all 4 face contacts.
TEST_F(MjGjkTest, BoxBoxMultiCCD18) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum* xpos = data->geom_xpos;
  mjtNum* xmat = data->geom_xmat;

  xpos[0] = 0.134004995;
  xpos[1] = -0.194004998;
  xpos[2] = 0.124997169;

  xmat[0] = 1.0;
  xmat[1] = 0.0;
  xmat[2] = 3.07510284e-08;
  xmat[3] = 0.0;
  xmat[4] = 1.0;
  xmat[5] = 0.0;
  xmat[6] = -3.07510284e-08;
  xmat[7] = 0.0;
  xmat[8] = 1.0;

  xpos = data->geom_xpos + 3;
  xmat = data->geom_xmat + 9;

  xpos[0] = 0.134004995;
  xpos[1] = -0.194004998;
  xpos[2] = 0.17400974;

  xmat[0] = 1.0;
  xmat[1] = 0.0;
  xmat[2] = -4.7682768e-08;
  xmat[3] = 0.0;
  xmat[4] = 1.0;
  xmat[5] = 0.0;
  xmat[6] = 4.7682768e-08;
  xmat[7] = 0.0;
  xmat[8] = 1.0;

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.000987431, MjTol(kTolerance, 1e-5));
}

TEST_F(MjGjkTest, BoxBoxConcentric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="0.025 0.025 0.025"/>
      <geom name="geom2" type="box" size="0.025 0.025 0.025"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -0.05, MjTol(kTolerance, 1e-5));
}

TEST_F(MjGjkTest, CylinderCylinderNearlyConcentric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="cylinder" size="1 0.5"/>
      <geom name="geom2" type="cylinder" size="1 0.5" pos="1e-7 0 0"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -1.0, MjTol(kTolerance, 1e-5));
}

TEST_F(MjGjkTest, BoxNormalsFaceDiagonal) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="geom1" type="box" size="1 1 1"/>
      <geom name="geom2" type="box" size="1 1 1" pos="0 0 1.9"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, model.get(), data.get(), g1, 0);
  mjc_initCCDObj(&obj2, model.get(), data.get(), g2, 0);

  // obj1 has two distinct support vertices (4 and 7) forming a face diagonal on
  // the +z face
  obj1.nsupp = 3;
  obj1.suppind[0] = 4;
  obj1.suppind[1] = 7;
  obj1.suppind[2] = 4;
  mjtNum s1[9] = {-1, -1, 1, 1, 1, 1, -1, -1, 1};
  for (int i = 0; i < 9; i++) obj1.suppvert[i] = s1[i];

  // obj2 has three distinct support vertices on the -z face
  obj2.nsupp = 3;
  obj2.suppind[0] = 0;
  obj2.suppind[1] = 1;
  obj2.suppind[2] = 2;
  mjtNum s2[9] = {-1, -1, 0.9, 1, -1, 0.9, -1, 1, 0.9};
  for (int i = 0; i < 9; i++) obj2.suppvert[i] = s2[i];

  // witness points with a small lateral tilt (> acos(mjFACE_TOL)) so
  // boxNormals2 would return 0
  std::vector<mjtNum> x1(12, 0), x2(12, 0), dist(4, 0);
  x1[0] = 0.0;
  x1[1] = 0.0;
  x1[2] = 1.0;
  x2[0] = -0.01;
  x2[1] = 0.0;
  x2[2] = 0.9;
  dist[0] = -0.1;

  auto buffer = std::vector<uint8_t>(mjc_ccdSize(4, 3, kMaxIterations));
  int ncons = mjc_multicontact(&obj1, &obj2, x1.data(), x2.data(), dist.data(),
                               4, 3, 4, buffer.data());
  EXPECT_EQ(ncons, 4);
}

TEST_F(MjGjkTest, MeshBoxConcentric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube"
            vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                    -1 -1  1  1 -1  1  1 1  1  -1 1  1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="mesh" mesh="cube"/>
      <geom name="geom2" type="box" size="0.2 0.2 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2, 0, 4);

  ASSERT_EQ(ncons, 4);
  EXPECT_NEAR(dist, -1.2, MjTol(1e-12, 1e-5));
}

TEST_F(MjGjkTest, AnisotropicMeshBoxConcentric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="slab" scale="0.18 0.18 0.06"
            vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                    -1 -1  1  1 -1  1  1 1  1  -1 1  1"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="mesh" mesh="slab"/>
      <geom name="geom2" type="box" size="0.039 0.039 0.14"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int g1 = mj_name2id(model.get(), mjOBJ_GEOM, "geom1");
  int g2 = mj_name2id(model.get(), mjOBJ_GEOM, "geom2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, g1, g2);

  ASSERT_EQ(ncons, 1);
  EXPECT_NEAR(dist, -0.2, MjTol(1e-8, 1e-5));
}

TEST_F(MjGjkTest, TouchingEllipsoids) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="e1" type="ellipsoid" pos="-0.5 0 0" size="0.5 0.3 0.2"/>
      <geom name="e2" type="ellipsoid" pos="0.4999995 0 0" size="0.5 0.3 0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int e1 = mj_name2id(model.get(), mjOBJ_GEOM, "e1");
  int e2 = mj_name2id(model.get(), mjOBJ_GEOM, "e2");

  mjCCDStatus status;
  std::vector<mjtNum> dir, pos;
  mjtNum dist;
  int ncons = Penetration(status, dist, dir, pos, model, data, e1, e2);

  ASSERT_EQ(ncons, 1);
  EXPECT_LT(dist, 0);
  EXPECT_NEAR(dist, -5e-7, MjTol(1e-11, 1e-6));
}

}  // namespace mujoco
