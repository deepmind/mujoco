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

// Tests for engine/{engine_support.c and engine_core_util.c}

#include "src/engine/engine_support.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::vector;
using ::testing::ContainsRegex;  // NOLINT
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::MatchesRegex;
using ::testing::Ne;
using ::testing::NotNull;
using ::testing::Pointwise;

using Name2idTest = MujocoTest;

static constexpr char name2idTestingModel[] = R"(
<mujoco>
    <asset>
      <texture name="texture1" type="2d" builtin="checker" rgb1="1 1 1"
       rgb2="1 1 1" width="300" height="300" mark="none"/>
      <material name="material1" texture="texture1" texrepeat="1 1"
       texuniform="true" reflectance=".2"/>
    </asset>

    <asset>
      <mesh name="mesh1" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>

  <worldbody>
    <light name="light1" pos="0 0 1"/>
    <site name="site1" pos="0 0 .3" size=".01"/>
    <site name="site2" pos="-.1 -.1 -.1" size=".01"/>
    <camera name="camera1" pos="0 -1.3 .5" xyaxes="1 0 0 0 1 2"/>
    <body name="body1">
      <joint axis="0 1 0" name="joint1"/>
      <geom size="1" name="body1_geom1"/>
      <geom size="1" name="body1_geom2"/>
    </body>
    <body name="body2">
      <joint axis="0 1 0" name="joint2"/>
      <geom size="1" name="body2_geom1"/>
      <geom size="1" name="camera1"/>
    </body>
    <body name="">
    </body>
  </worldbody>

  <tendon>
    <spatial name="tendon1" limited="true" range="0 0.35" width="0.003">
      <site site="site1"/>
      <site site="site2"/>
    </spatial>
  </tendon>

  <actuator>
    <motor name="actuator1" joint="joint1" gear="1"/>
  </actuator>

  <sensor>
    <accelerometer name="sensor1" site="site1"/>
  </sensor>
</mujoco>
)";

TEST_F(Name2idTest, FindIds) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(name2idTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_BODY, "world"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_BODY, "body1"), 1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_BODY, "body2"), 2);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_GEOM, "body1_geom1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_GEOM, "body1_geom2"), 1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_JOINT, "joint2"), 1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_MESH, "mesh1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_LIGHT, "light1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_CAMERA, "camera1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_SITE, "site2"), 1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_MATERIAL, "material1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_TEXTURE, "texture1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_TENDON, "tendon1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_ACTUATOR, "actuator1"), 0);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_SENSOR, "sensor1"), 0);
}

TEST_F(Name2idTest, MissingIds) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(name2idTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_BODY, "abody3"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_GEOM, "abody2_geom2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_JOINT, "joint3"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_MESH, "amesh2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_LIGHT, "alight2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_CAMERA, "acamera2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_SITE, "asite3"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_MATERIAL, "amaterial2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_TEXTURE, "atexture2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_TENDON, "atendon2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_ACTUATOR, "aactuator2"), -1);
  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_SENSOR, "asensor2"), -1);
}

TEST_F(Name2idTest, EmptyIds) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(name2idTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_BODY, ""), -1);
}

TEST_F(Name2idTest, Namespaces) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(name2idTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(mj_name2id(model.get(), mjOBJ_GEOM, "camera1"), 3);
}

using VersionTest = MujocoTest;

TEST_F(VersionTest, MjVersion) { EXPECT_EQ(mj_version(), mjVERSION_HEADER); }

TEST_F(VersionTest, MjVersionString) {
#if GTEST_USES_SIMPLE_RE == 1
  auto regex_matcher = ContainsRegex("^\\d+\\.\\d+\\.\\d+");
#else
  auto regex_matcher = MatchesRegex("^[0-9]+\\.[0-9]+\\.[0-9]+(-[0-9a-z]+)?$");
#endif
  EXPECT_THAT(std::string(mj_versionString()), regex_matcher);
}

using SupportTest = MujocoTest;

// utility: generate two random quaternions with a given angle difference
void randomQuatPair(mjtNum qa[4], mjtNum qb[4], mjtNum angle, int seed) {
  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(seed);
  std::normal_distribution<double> dist(0, 1);

  // sample qa = qb
  for (int i = 0; i < 4; i++) {
    qa[i] = qb[i] = dist(rng);
  }
  mju_normalize4(qa);
  mju_normalize4(qb);

  // integrate qb in random direction by angle
  mjtNum dir[3];
  for (int i = 0; i < 3; i++) {
    dir[i] = dist(rng);
  }
  mju_normalize3(dir);
  mju_quatIntegrate(qb, dir, angle);
}

static constexpr char ballJointModel[] = R"(
<mujoco>
  <worldbody>
    <body>
      <joint type="ball"/>
      <geom size="1"/>
    </body>
  </worldbody>
</mujoco>
)";

TEST_F(SupportTest, DifferentiatePosSubQuat) {
  const mjtNum eps = 2e-12;  // epsilon for float comparison

  char error[1024];
  MjModelPtr model = LoadModelFromString(ballJointModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  int seed = 1;
  for (mjtNum angle : {0.0, 1e-5, 1e-2}) {
    for (mjtNum dt : {1e-6, 1e-3, 1e-1}) {
      // random quaternion pair with given angle difference
      mjtNum qpos1[4], qpos2[4];
      randomQuatPair(qpos1, qpos2, angle, seed++);

      // get velocity given timestep
      mjtNum qvel[3];
      mj_differentiatePos(model.get(), qvel, dt, qpos1, qpos2);

      // equivalent computation
      mjtNum qneg[4], qdif[4], qvel_expect[3];
      mju_negQuat(qneg, qpos1);
      mju_mulQuat(qdif, qneg, qpos2);
      mju_quat2Vel(qvel_expect, qdif, dt);

      // expect numerical equality
      EXPECT_THAT(AsVector(qvel, 3), Pointwise(MjNear(eps, 1e-3), qvel_expect));
    }
  }
}

static const char* const kDefaultModel = "testdata/model.xml";

using StateTest = MujocoTest;

TEST_F(StateTest, GetSetStateStepEqual) {
  const std::string xml_path = GetTestDataFilePath(kDefaultModel);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error;
  mjData* data = mj_makeData(model);

  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(3);
  std::normal_distribution<double> dist(0, .01);

  // set controls and applied joint forces to random values
  for (int i = 0; i < model->nu; i++) data->ctrl[i] = dist(rng);
  for (int i = 0; i < model->nv; i++) data->qfrc_applied[i] = dist(rng);
  for (int i = 0; i < model->neq; i++) data->eq_active[i] = dist(rng) > 0;

  // take one step
  mj_step(model, data);

  int signature = mjSTATE_INTEGRATION;
  int size = mj_stateSize(model, signature);

  // save the initial state and step
  vector<mjtNum> state0a(size);
  mj_getState(model, data, state0a.data(), signature);

  // get the initial state, expect equality
  vector<mjtNum> state0b(size);
  mj_getState(model, data, state0b.data(), signature);
  EXPECT_EQ(state0a, state0b);

  // take one step
  mj_step(model, data);

  // save the resulting state
  vector<mjtNum> state1a(size);
  mj_getState(model, data, state1a.data(), signature);

  // expect the state to be different after stepping
  EXPECT_THAT(state0a, testing::Ne(state1a));

  // reset to the saved state, step again, get the resulting state
  mj_setState(model, data, state0a.data(), signature);
  mj_step(model, data);
  vector<mjtNum> state1b(size);
  mj_getState(model, data, state1b.data(), signature);

  // expect the state to be the same after re-stepping
  EXPECT_EQ(state1a, state1b);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(StateTest, GetSetStateDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.05" nsample="5"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // verify history buffer exists: nhistory = 2 + 2*5 = 12
  EXPECT_EQ(model->nhistory, 12);  // [user, cursor, times(5), values(5)]

  // state size should include history buffer
  int size = mj_stateSize(model.get(), mjSTATE_HISTORY);
  EXPECT_EQ(size, model->nhistory);

  // step to populate history buffer
  data->ctrl[0] = 1.0;
  mj_step(model.get(), data.get());
  data->ctrl[0] = 2.0;
  mj_step(model.get(), data.get());

  // get history state
  vector<mjtNum> history_state(size);
  mj_getState(model.get(), data.get(), history_state.data(), mjSTATE_HISTORY);

  // modify the history buffer manually (value at index 7 = 2+5 = after times)
  data->history[7] = 99.0;  // first value

  // set history state back - should restore original
  mj_setState(model.get(), data.get(), history_state.data(), mjSTATE_HISTORY);

  // verify restoration
  EXPECT_NE(data->history[7], 99.0);
}

// history buffers require a positive timestep: the compiler rejects a model
// without one, and a reset after the timestep is changed warns
TEST_F(StateTest, HistoryNonPositiveTimestep) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="%s"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.02" nsample="3"/>
    </actuator>
    <sensor>
      <jointpos joint="slide" nsample="3" interval="0.02"/>
    </sensor>
  </mujoco>
  )";

  char model_xml[1024];
  char error[1024];
  std::snprintf(model_xml, sizeof(model_xml), xml, "0");
  MjModelPtr bad = LoadModelFromString(model_xml, error, sizeof(error));
  EXPECT_EQ(bad.get(), nullptr);
  std::string message(error);
  std::string expected = "history buffers require positive timestep";
  EXPECT_THAT(message, HasSubstr(expected));
  EXPECT_EQ(message.find(expected), message.rfind(expected)) << message;

  std::snprintf(model_xml, sizeof(model_xml), xml, "0.01");
  MjModelPtr model = LoadModelFromString(model_xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  model->opt.timestep = 0;
  mock_warning_handler.ExpectWarnings(
      "history buffers require positive timestep");
  mj_resetData(model.get(), data.get());
  for (int i = 0; i < model->nhistory; i++) {
    EXPECT_TRUE(std::isfinite(data->history[i])) << i;
  }
}

TEST_F(StateTest, CopyState) {
  const std::string xml_path = GetTestDataFilePath(kDefaultModel);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << "Failed to load model: " << error;

  mjData* src = mj_makeData(m);
  mjData* dst = mj_makeData(m);

  // init both datas to default
  mj_resetData(m, src);
  mj_resetData(m, dst);

  // modify d_src
  src->time = 1.23;
  for (int i = 0; i < m->nq; ++i) src->qpos[i] = i * 0.1;
  for (int i = 0; i < m->nv; ++i) src->qvel[i] = i * 0.2;
  for (int i = 0; i < m->na; ++i) src->act[i] = i * 0.3;
  for (int i = 0; i < m->nu; ++i) src->ctrl[i] = i * 0.4;
  for (int i = 0; i < m->nhistory; ++i) src->history[i] = i * 0.5;

  for (int i = 0; i < m->neq; ++i) src->eq_active[i] = 1 - m->eq_active0[i];

  // check that states differ
  EXPECT_NE(src->time, dst->time);
  EXPECT_THAT(AsVector(src->qpos, m->nq), Ne(AsVector(dst->qpos, m->nq)));
  EXPECT_THAT(AsVector(src->ctrl, m->nu), Ne(AsVector(dst->ctrl, m->nu)));

  // copy state with signature
  int signature = mjSTATE_FULLPHYSICS | mjSTATE_EQ_ACTIVE;
  mj_copyState(m, src, dst, signature);

  // check copied components
  EXPECT_EQ(dst->time, src->time);
  EXPECT_EQ(AsVector(dst->qpos, m->nq), AsVector(src->qpos, m->nq));
  EXPECT_EQ(AsVector(dst->qvel, m->nv), AsVector(src->qvel, m->nv));
  EXPECT_EQ(AsVector(dst->act, m->na), AsVector(src->act, m->na));
  EXPECT_EQ(AsVector(dst->history, m->nhistory),
            AsVector(src->history, m->nhistory));
  EXPECT_EQ(AsVector(dst->eq_active, m->neq), AsVector(src->eq_active, m->neq));

  // check non-copied components (CTRL not in signature)
  EXPECT_THAT(AsVector(dst->ctrl, m->nu), Ne(AsVector(src->ctrl, m->nu)));
  EXPECT_EQ(AsVector(dst->ctrl, m->nu), vector<mjtNum>(m->nu, 0.0));

  mj_deleteData(src);
  mj_deleteData(dst);
  mj_deleteModel(m);
}

TEST_F(StateTest, ExtractState) {
  const std::string xml_path = GetTestDataFilePath(kDefaultModel);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error;
  mjData* data = mj_makeData(model);

  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(3);
  std::normal_distribution<double> dist(0, .01);

  // set controls and applied joint forces to random values
  for (int i = 0; i < model->nu; i++) data->ctrl[i] = dist(rng);
  for (int i = 0; i < model->nv; i++) data->qfrc_applied[i] = dist(rng);
  for (int i = 0; i < model->neq; i++) data->eq_active[i] = dist(rng) > 0;

  // take one step
  mj_step(model, data);

  // take a state that will be used as src
  int srcsig = mjSTATE_TIME | mjSTATE_QPOS | mjSTATE_QVEL | mjSTATE_CTRL |
               mjSTATE_HISTORY;
  int srcsize = mj_stateSize(model, srcsig);
  vector<mjtNum> srcstate(srcsize);
  mj_getState(model, data, srcstate.data(), srcsig);

  // extract a subset consisting of only a single bit in srcsig
  int dstsig1 = mjSTATE_CTRL;
  int dstsize1 = mj_stateSize(model, dstsig1);
  EXPECT_LT(dstsize1, srcsize);
  EXPECT_EQ(dstsize1, model->nu);
  vector<mjtNum> dststate1(dstsize1);
  mj_extractState(model, srcstate.data(), srcsig, dststate1.data(), dstsig1);
  EXPECT_EQ(dststate1, AsVector(data->ctrl, model->nu));

  // extract a subset consisting of multiple non-consecutive bits in srcsig
  int dstsig2 = mjSTATE_QPOS | mjSTATE_CTRL;
  int dstsize2 = mj_stateSize(model, dstsig2);
  EXPECT_LT(dstsize2, srcsize);
  EXPECT_EQ(dstsize2, model->nq + model->nu);
  vector<mjtNum> dststate2(dstsize2);
  mj_extractState(model, srcstate.data(), srcsig, dststate2.data(), dstsig2);
  EXPECT_EQ(AsVector(dststate2.data(), model->nq),
            AsVector(data->qpos, model->nq));
  EXPECT_EQ(AsVector(dststate2.data() + model->nq, model->nu),
            AsVector(data->ctrl, model->nu));

  // extract history state
  int dstsig3 = mjSTATE_HISTORY;
  int dstsize3 = mj_stateSize(model, dstsig3);
  EXPECT_EQ(dstsize3, model->nhistory);
  vector<mjtNum> dststate3(dstsize3);
  mj_extractState(model, srcstate.data(), srcsig, dststate3.data(), dstsig3);
  EXPECT_EQ(dststate3, AsVector(data->history, model->nhistory));

  // test that an error is correctly raised if dstsig is not a subset of srcsig
  static int error_count;
  static char last_error_msg[128];
  error_count = 0;
  last_error_msg[0] = '\0';
  auto* error_handler = +[](const char* msg) {
    std::strncpy(last_error_msg, msg, sizeof(last_error_msg));
    ++error_count;
  };

  auto* old_mju_user_error = mju_user_error;
  mju_user_error = error_handler;

  mj_extractState(model, nullptr, srcsig, nullptr, mjSTATE_QFRC_APPLIED);
  EXPECT_EQ(error_count, 1);
  EXPECT_EQ(std::string_view(last_error_msg),
            "mj_extractState: dstsig is not a subset of srcsig");

  mj_extractState(model, nullptr, -1, nullptr, mjSTATE_QFRC_APPLIED);
  EXPECT_EQ(error_count, 2);
  EXPECT_EQ(std::string_view(last_error_msg),
            "mj_extractState: invalid srcsig -1 < 0");

  mju_user_error = old_mju_user_error;

  mj_deleteData(data);
  mj_deleteModel(model);
}

using InertiaTest = MujocoTest;

static const char* const kInertiaPath = "engine/testdata/inertia.xml";

TEST_F(InertiaTest, AddMdenseSameAsSparse) {
  const std::string xml_path = GetTestDataFilePath(kInertiaPath);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << "Failed to load model: " << error;
  int nv = m->nv;

  mjData* d = mj_makeData(m);

  mj_step(m, d);

  // dense matrix, all values are 3.0
  vector<mjtNum> dst_dense(nv * nv, 3.0);

  // sparse matrix, all values are 3.0
  vector<mjtNum> dst_sparse(nv * nv, 3.0);
  vector<int> rownnz(nv, nv);
  vector<int> rowadr(nv, 0);
  vector<int> colind(nv * nv, 0);

  // set sparse structure
  for (int i = 0; i < nv; i++) {
    rowadr[i] = i * nv;
    for (int j = 0; j < nv; j++) {
      colind[rowadr[i] + j] = j;
    }
  }

  // sparse addM
  mj_addM(m, d, dst_sparse.data(), rownnz.data(), rowadr.data(), colind.data());

  // dense addM
  mj_addM(m, d, dst_dense.data(), nullptr, nullptr, nullptr);

  // dense comparison (lower triangle)
  for (int i = 0; i < nv; i++) {
    for (int j = 0; j < nv; j++) {
      EXPECT_EQ(dst_dense[i * nv + j], dst_sparse[i * nv + j]);
    }
  }

  // clean up
  mj_deleteData(d);
  mj_deleteModel(m);
}

TEST_F(InertiaTest, mulM) {
  const std::string xml_path = GetTestDataFilePath(kInertiaPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error;
  int nv = model->nv;

  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  // dense M matrix
  vector<mjtNum> Mdense(nv * nv);
  mju_sym2dense(Mdense.data(), data->M, nv, model->M_rownnz, model->M_rowadr,
                model->M_colind);

  // arbitrary RHS vector
  vector<mjtNum> vec(nv);
  for (int i = 0; i < nv; i++) vec[i] = vec[i] = 20 + 30 * i;

  // multiply directly
  vector<mjtNum> res1(nv, 0);
  mju_mulMatVec(res1.data(), Mdense.data(), vec.data(), nv, nv);

  // multiply with mj_mulM
  vector<mjtNum> res2(nv, 0);
  mj_mulM(model, data, res2.data(), vec.data());

  // expect vectors to match to floating point precision
  EXPECT_THAT(res1, Pointwise(MjNear(1e-10, 0.1), res2));

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(InertiaTest, mulM2) {
  const std::string xml_path = GetTestDataFilePath(kInertiaPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error;
  int nv = model->nv;

  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  // arbitrary RHS vector
  vector<mjtNum> vec(nv);
  for (int i = 0; i < nv; i++) vec[i] = .2 + .3 * i;

  // multiply sqrtMvec = M^1/2 * vec
  vector<mjtNum> sqrtMvec(nv);
  mj_mulM2(model, data, sqrtMvec.data(), vec.data());

  // multiply Mvec = M * vec
  vector<mjtNum> Mvec(nv);
  mj_mulM(model, data, Mvec.data(), vec.data());

  // compute vec' * M * vec in two different ways, expect them to match
  mjtNum sqrtMvec2 = mju_dot(sqrtMvec.data(), sqrtMvec.data(), nv);
  mjtNum vecMvec = mju_dot(vec.data(), Mvec.data(), nv);
  EXPECT_MJTNUM_EQ(sqrtMvec2, vecMvec);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(InertiaTest, FullM) {
  const std::string xml_path = GetTestDataFilePath(kInertiaPath);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << "Failed to load model: " << error;
  int nv = m->nv;

  // forward dynamics, populate M and qLD
  mjData* d = mj_makeData(m);
  mj_forward(m, d);

  // get dense mass matrix from M using mj_fullM
  vector<mjtNum> M(nv * nv);
  mj_fullM(m, d, M.data());

  // get dense mass matrix from M using mju_sparse2dense
  vector<mjtNum> M_CSR(nv * nv);
  mju_sparse2dense(M_CSR.data(), d->M, nv, nv, m->M_rownnz, m->M_rowadr,
                   m->M_colind);

  // expect lower triangles to match exactly
  for (int i = 0; i < nv; ++i) {
    for (int j = 0; j <= i; ++j) {
      EXPECT_EQ(M[i * nv + j], M_CSR[i * nv + j]);
    }
  }

  // get dense LTDL factor (D on the diagonal)
  vector<mjtNum> LD(nv * nv);
  mju_sparse2dense(LD.data(), d->qLD, nv, nv, m->M_rownnz, m->M_rowadr,
                   m->M_colind);

  // extract L and D from LD
  vector<mjtNum> L = LD;
  vector<mjtNum> D(nv * nv, 0.0);
  for (int i = 0; i < nv; i++) {
    D[i * nv + i] = LD[i * nv + i];
    L[i * nv + i] = 1.0;
  }

  // compute DL = D * L
  vector<mjtNum> DL(nv * nv, 0.0);
  mju_mulMatMat(DL.data(), D.data(), L.data(), nv, nv, nv);

  // compute the triple product P = L^T * D * L
  vector<mjtNum> P(nv * nv, 0.0);
  mju_mulMatTMat(P.data(), L.data(), DL.data(), nv, nv, nv);

  // expect M and P to match to high precision
  EXPECT_THAT(M, Pointwise(MjNear(1e-10, 1e-4), P));

  mj_deleteData(d);
  mj_deleteModel(m);
}

static constexpr char GeomDistanceTestingModel1[] = R"(
<mujoco>
  <option>
    <flag nativeccd="enable"/>
  </option>
  <asset>
    <mesh name="box" scale=".1 .1 .1" vertex="0 0 0  1 0 0  0 1 0  1 1 0
                                              0 0 1  1 0 1  0 1 1  1 1 1"/>
  </asset>

  <worldbody>
    <geom type="plane" size="1 1 1"/>
    <geom pos="0 0 1" size="0.2"/>
    <geom pos="1 0 1" size="0.3"/>
    <geom type="mesh" mesh="box"/>
  </worldbody>
</mujoco>
)";

static constexpr char GeomDistanceTestingModel2[] = R"(
<mujoco>
  <worldbody>
    <geom type="sphere" size=".1"/>
    <geom type="ellipsoid" size=".1 .1 .1" pos="0 0 1"/>
  </worldbody>
</mujoco>
)";

TEST_F(SupportTest, GeomDistance) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(GeomDistanceTestingModel1, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_kinematics(model.get(), data.get());

  // plane-sphere, distmax too small
  mjtNum distmax = 0.5;
  EXPECT_EQ(mj_geomDistance(model.get(), data.get(), 0, 1, distmax, nullptr),
            0.5);
  mjtNum fromto[6];
  EXPECT_EQ(mj_geomDistance(model.get(), data.get(), 0, 1, distmax, fromto),
            0.5);
  EXPECT_THAT(fromto, Pointwise(Eq(), vector<mjtNum>{0, 0, 0, 0, 0, 0}));

  // plane-sphere
  distmax = 1.0;
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 0, 1, 1.0, fromto),
              MjNear(0.8, 1e-12, 1e-5));
  mjtNum eps = 1e-12;
  EXPECT_THAT(fromto,
              Pointwise(MjNear(eps, 1e-5), vector<mjtNum>{0, 0, 0, 0, 0, 0.8}));

  // sphere-plane
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 1, 0, 1.0, fromto),
              MjNear(0.8, 1e-12, 1e-5));
  EXPECT_THAT(fromto,
              Pointwise(MjNear(eps, 1e-5), vector<mjtNum>{0, 0, 0.8, 0, 0, 0}));

  // sphere-sphere
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 1, 2, 1.0, fromto),
              MjNear(0.5, 1e-12, 1e-5));
  EXPECT_THAT(fromto,
              Pointwise(MjNear(eps, 1e-5), vector<mjtNum>{.2, 0, 1, .7, 0, 1}));

  // sphere-sphere, flipped order
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 2, 1, 1.0, fromto),
              MjNear(0.5, 1e-12, 1e-5));
  EXPECT_THAT(fromto,
              Pointwise(MjNear(eps, 1e-5), vector<mjtNum>{.7, 0, 1, .2, 0, 1}));

  // mesh-sphere (close distmax)
  distmax = 0.701;
  eps = model->opt.ccd_tolerance;
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 3, 1, distmax, fromto),
              MjNear(0.7, eps, eps * 100));
  EXPECT_THAT(fromto, Pointwise(MjNear(eps, eps * 100),
                                vector<mjtNum>{0, 0, .1, 0, 0, .8}));

  // mesh-sphere (far distmax)
  distmax = 1.0;
  EXPECT_THAT(mj_geomDistance(model.get(), data.get(), 3, 1, distmax, fromto),
              MjNear(0.7, eps, eps * 100));
  EXPECT_THAT(fromto, Pointwise(MjNear(eps, eps * 100),
                                vector<mjtNum>{0, 0, .1, 0, 0, .8}));
}

TEST_F(SupportTest, GeomDistanceFromToFlipped) {
  mjtNum distmax = 10.0;
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(GeomDistanceTestingModel2, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_kinematics(model.get(), data.get());

  mjtNum fromto01[6];
  mjtNum fromto10[6];

  for (int flag : {0, (int)mjDSBL_NATIVECCD}) {
    model->opt.disableflags = flag;
    mj_geomDistance(model.get(), data.get(), 0, 1, distmax, fromto01);
    mj_geomDistance(model.get(), data.get(), 1, 0, distmax, fromto10);
    mjtNum fromto10flipped[6] = {fromto10[3], fromto10[4], fromto10[5],
                                 fromto10[0], fromto10[1], fromto10[2]};

    EXPECT_THAT(AsVector(fromto10flipped, 6),
                Pointwise(MjNear(1.0e-12, 1e-5), fromto01));
  }
}

static constexpr char kSetKeyframeTestingModel[] = R"(
<mujoco>
  <size nkey="2"/>

  <worldbody>
    <body>
      <joint name="joint" axis="0 1 0"/>
      <geom size=".1" pos="1 0 0"/>
    </body>
  </worldbody>

  <actuator>
    <intvelocity joint="joint" actrange="-1 1" kp="100" dampratio="1"/>
  </actuator>
</mujoco>
)";

TEST_F(SupportTest, SetKeyframe) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kSetKeyframeTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  data->ctrl[0] = 1;
  while (data->time < 1) {
    mj_step(model.get(), data.get());
  }

  mj_setKeyframe(model.get(), data.get(), 1);
  EXPECT_EQ(data->time, model->key_time[1]);
  EXPECT_EQ(data->ctrl[0], model->key_ctrl[model->nu * 1]);
  EXPECT_EQ(data->qpos[0], model->key_qpos[model->nq * 1]);
  EXPECT_EQ(data->qvel[0], model->key_qvel[model->nv * 1]);
  EXPECT_EQ(data->act[0], model->key_act[model->na * 1]);

  mj_step(model.get(), data.get());
  mj_setKeyframe(model.get(), data.get(), 0);
  EXPECT_EQ(data->time, model->key_time[0]);
  EXPECT_EQ(data->ctrl[0], model->key_ctrl[model->nu * 0]);
  EXPECT_EQ(data->qpos[0], model->key_qpos[model->nq * 0]);
  EXPECT_EQ(data->qvel[0], model->key_qvel[model->nv * 0]);
  EXPECT_EQ(data->act[0], model->key_act[model->na * 0]);
}

TEST_F(SupportTest, ContactSensorDim) {
  int dataSpec = 1 << mjCONDATA_FOUND | 1 << mjCONDATA_FORCE |
                 1 << mjCONDATA_DIST | 1 << mjCONDATA_POS |
                 1 << mjCONDATA_TANGENT | 1 << mjCONDATA_LINVEL;

  EXPECT_EQ(mju_condataSize(dataSpec), 1 + 3 + 1 + 3 + 3 + 3);
}

// ------------------------------ ctrl delays --------------------------------

TEST_F(SupportTest, ReadCtrlNoDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);

  // no delay: should return pointer to current ctrl value
  data->ctrl[0] = 42.0;
  mjtNum result = 0;
  const mjtNum* ptr = mj_readCtrl(model.get(), data.get(), 0, data->time,
                                  &result, /*interp=*/0);
  ASSERT_THAT(ptr, NotNull());
  EXPECT_EQ(*ptr, 42.0);
}

TEST_F(SupportTest, ReadCtrlWithDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.03" nsample="3"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // model should have delay configured
  // delay = 0.03 seconds, timestep = 0.01, so ndelay = ceil(0.03/0.01) = 3
  EXPECT_EQ(model->actuator_history[0], 3);
  EXPECT_NEAR(model->actuator_delay[0], 0.03, 1e-7);
  EXPECT_GE(model->actuator_historyadr[0], 0);

  // initially, buffer should be filled with constant value (from init)
  // reading at current time should return the init value
  mjtNum result = 0;
  const mjtNum* ptr = mj_readCtrl(model.get(), data.get(), 0, data->time,
                                  &result, /*interp=*/0);
  mjtNum val = ptr ? *ptr : result;
  EXPECT_EQ(val, data->ctrl[0]);
}

TEST_F(SupportTest, InitCtrlDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.02" nsample="3"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // verify nhistory
  EXPECT_EQ(model->actuator_history[0], 3);

  // initialize with custom times and values
  // buffer stores: time 0.0 -> value 1.0, time 0.01 -> value 2.0, time 0.02 ->
  // value 3.0
  mjtNum times[3] = {0.0, 0.01, 0.02};
  mjtNum values[3] = {1.0, 2.0, 3.0};
  mj_initCtrlHistory(model.get(), data.get(), 0, times, values);

  // mj_readCtrl now auto-subtracts delay: lookup_time = time - delay
  // delay = 0.02, so:
  //   time=0.04 -> lookup at 0.02 -> value 3.0
  //   time=0.03 -> lookup at 0.01 -> value 2.0
  //   time=0.02 -> lookup at 0.00 -> value 1.0
  mjtNum result = 0;
  const mjtNum* ptr =
      mj_readCtrl(model.get(), data.get(), 0, 0.04, &result, /*interp=*/0);
  mjtNum val = ptr ? *ptr : result;
  EXPECT_EQ(val, 3.0);

  ptr = mj_readCtrl(model.get(), data.get(), 0, 0.03, &result, /*interp=*/0);
  val = ptr ? *ptr : result;
  EXPECT_EQ(val, 2.0);

  ptr = mj_readCtrl(model.get(), data.get(), 0, 0.02, &result, /*interp=*/0);
  val = ptr ? *ptr : result;
  EXPECT_EQ(val, 1.0);
}

TEST_F(SupportTest, InitCtrlDelayMultiInput) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid joint="slide" kp="10" kv="2" input="pos vel ff"
           delay="0.02" nsample="3"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // nhistory = 2 + nsample + nsample*ctrlnum = 2 + 3 + 3*3 = 14
  EXPECT_EQ(model->actuator_ctrlnum[0], 3);
  EXPECT_EQ(model->actuator_history[0], 3);
  EXPECT_EQ(model->nhistory, 14);

  mjtNum times[3] = {0.0, 0.01, 0.02};
  mjtNum values[9] = {
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
  };
  mj_initCtrlHistory(model.get(), data.get(), 0, times, values);

  // ZOH at time=0.03 -> lookup at 0.01 -> [4.0, 5.0, 6.0]
  mjtNum result[3] = {0, 0, 0};
  const mjtNum* ptr =
      mj_readCtrl(model.get(), data.get(), 0, 0.03, result, /*interp=*/0);
  ASSERT_THAT(ptr, NotNull());
  EXPECT_EQ(ptr[0], 4.0);
  EXPECT_EQ(ptr[1], 5.0);
  EXPECT_EQ(ptr[2], 6.0);

  // linear interp at time=0.035 -> lookup at 0.015 -> [5.5, 6.5, 7.5]
  ptr = mj_readCtrl(model.get(), data.get(), 0, 0.035, result, /*interp=*/1);
  EXPECT_EQ(ptr, nullptr);
  EXPECT_NEAR(result[0], 5.5, 1e-6);
  EXPECT_NEAR(result[1], 6.5, 1e-6);
  EXPECT_NEAR(result[2], 7.5, 1e-6);
}

TEST_F(SupportTest, InitCtrlDelayNullTimes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.02" nsample="3"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // get existing times from buffer
  int adr = model->actuator_historyadr[0];
  mjtNum* buf = data->history + adr;
  mjtNum existing_times[3] = {buf[2], buf[3], buf[4]};

  // initialize with NULL times (use existing) and new values
  mjtNum values[3] = {10.0, 20.0, 30.0};
  mj_initCtrlHistory(model.get(), data.get(), 0, nullptr, values);

  // verify times are unchanged
  EXPECT_EQ(buf[2], existing_times[0]);
  EXPECT_EQ(buf[3], existing_times[1]);
  EXPECT_EQ(buf[4], existing_times[2]);

  // verify values are updated
  EXPECT_EQ(buf[5], 10.0);
  EXPECT_EQ(buf[6], 20.0);
  EXPECT_EQ(buf[7], 30.0);
}

TEST_F(SupportTest, InitCtrlAndSensorHistoryNullTimesAfterStepWrap) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.02" nsample="3"/>
    </actuator>
    <sensor>
      <jointpos joint="slide" delay="0.02" nsample="3"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Step twice so the circular history cursor wraps from 2 to 0.
  mj_step(model.get(), data.get());
  mj_step(model.get(), data.get());

  mjtNum ctrl_values[3] = {10.0, 20.0, 30.0};
  mj_initCtrlHistory(model.get(), data.get(), 0, nullptr, ctrl_values);

  mjtNum sensor_values[3] = {100.0, 200.0, 300.0};
  mj_initSensorHistory(model.get(), data.get(), 0, nullptr, sensor_values, 0.0);

  mjtNum res = 0;
  const mjtNum* p0 = mj_readCtrl(model.get(), data.get(), 0, 0.01, &res, 0);
  const mjtNum* p1 = mj_readCtrl(model.get(), data.get(), 0, 0.02, &res, 0);
  const mjtNum* p2 = mj_readCtrl(model.get(), data.get(), 0, 0.03, &res, 0);
  ASSERT_THAT(p0, NotNull());
  ASSERT_THAT(p1, NotNull());
  ASSERT_THAT(p2, NotNull());
  EXPECT_MJTNUM_EQ(*p0, 10.0);
  EXPECT_MJTNUM_EQ(*p1, 20.0);
  EXPECT_MJTNUM_EQ(*p2, 30.0);

  const mjtNum* s0 = mj_readSensor(model.get(), data.get(), 0, 0.01, &res, 0);
  const mjtNum* s1 = mj_readSensor(model.get(), data.get(), 0, 0.02, &res, 0);
  const mjtNum* s2 = mj_readSensor(model.get(), data.get(), 0, 0.03, &res, 0);
  ASSERT_THAT(s0, NotNull());
  ASSERT_THAT(s1, NotNull());
  ASSERT_THAT(s2, NotNull());
  EXPECT_MJTNUM_EQ(*s0, 100.0);
  EXPECT_MJTNUM_EQ(*s1, 200.0);
  EXPECT_MJTNUM_EQ(*s2, 300.0);
}

TEST_F(SupportTest, InitSensorDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" delay="0.02" nsample="3"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // verify nsample for sensor
  EXPECT_EQ(model->sensor_history[0], 3);

  // initialize with custom times and values, phase=0
  // buffer stores: time 0.0 -> value 0.5, time 0.01 -> value 0.6, time 0.02 ->
  // value 0.7
  mjtNum times[3] = {0.0, 0.01, 0.02};
  mjtNum values[3] = {0.5, 0.6, 0.7};
  mj_initSensorHistory(model.get(), data.get(), 0, times, values,
                       /*phase=*/0.0);

  // mj_readSensor now auto-subtracts delay: lookup_time = time - delay
  // delay = 0.02, so:
  //   time=0.04 -> lookup at 0.02 -> value 0.7
  //   time=0.03 -> lookup at 0.01 -> value 0.6
  mjtNum result = 0;
  const mjtNum* ptr =
      mj_readSensor(model.get(), data.get(), 0, 0.04, &result, /*order=*/0);
  mjtNum val = ptr ? *ptr : result;
  EXPECT_NEAR(val, 0.7, 1e-6);

  ptr = mj_readSensor(model.get(), data.get(), 0, 0.03, &result, /*order=*/0);
  val = ptr ? *ptr : result;
  EXPECT_NEAR(val, 0.6, 1e-6);
}

TEST_F(SupportTest, InsideSite) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box_mesh"
            vertex="-0.2 -0.3 -0.4
                     0.2 -0.3 -0.4
                     0.2  0.3 -0.4
                    -0.2  0.3 -0.4
                    -0.2 -0.3  0.4
                     0.2 -0.3  0.4
                     0.2  0.3  0.4
                    -0.2  0.3  0.4"/>
      <mesh name="octa_mesh"
            vertex=" 1  0  0
                    -1  0  0
                     0  1  0
                     0 -1  0
                     0  0  1
                     0  0 -1"/>
    </asset>
    <worldbody>
      <site name="sph" type="sphere" size="0.5" pos="1 0 0"/>
      <site name="box" type="box" size="0.5 0.5 0.5" pos="0 1 0"/>
      <site name="msh" type="mesh" mesh="box_mesh" pos="0 0 1"/>
      <site name="oct" type="mesh" mesh="octa_mesh" pos="0 0 0"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  mjData* data = mj_makeData(model.get());
  mj_forward(model.get(), data);

  int sph_id = mj_name2id(model.get(), mjOBJ_SITE, "sph");
  int box_id = mj_name2id(model.get(), mjOBJ_SITE, "box");
  int msh_id = mj_name2id(model.get(), mjOBJ_SITE, "msh");
  int oct_id = mj_name2id(model.get(), mjOBJ_SITE, "oct");

  // sphere tests
  mjtNum pt_inside_sph[3] = {1.2, 0, 0};
  mjtNum pt_outside_sph[3] = {1.6, 0, 0};
  EXPECT_EQ(mj_insideSite(model.get(), data, sph_id, pt_inside_sph), 1);
  EXPECT_EQ(mj_insideSite(model.get(), data, sph_id, pt_outside_sph), 0);

  // box tests
  mjtNum pt_inside_box[3] = {0.3, 1.3, -0.3};
  mjtNum pt_outside_box[3] = {0.6, 1.0, 0};
  EXPECT_EQ(mj_insideSite(model.get(), data, box_id, pt_inside_box), 1);
  EXPECT_EQ(mj_insideSite(model.get(), data, box_id, pt_outside_box), 0);

  // box mesh tests
  mjtNum pt_inside_msh[3] = {0.1, -0.2, 1.3};
  mjtNum pt_outside_msh[3] = {0.3, 0, 1.0};
  mjtNum pt_far_msh[3] = {10.0, 10.0, 10.0};
  EXPECT_EQ(mj_insideSite(model.get(), data, msh_id, pt_inside_msh), 1);
  EXPECT_EQ(mj_insideSite(model.get(), data, msh_id, pt_outside_msh), 0);
  EXPECT_EQ(mj_insideSite(model.get(), data, msh_id, pt_far_msh), 0);

  // octahedron mesh tests: inside hull, inside AABB but outside hull, outside
  // AABB
  mjtNum pt_inside_oct[3] = {0.2, 0.2, 0.2};
  mjtNum pt_in_aabb_out_hull[3] = {0.6, 0.6, 0.0};
  mjtNum pt_out_aabb_x[3] = {1.1, 0.0, 0.0};
  mjtNum pt_out_aabb_y[3] = {0.0, 1.1, 0.0};
  mjtNum pt_out_aabb_z[3] = {0.0, 0.0, 1.1};
  EXPECT_EQ(mj_insideSite(model.get(), data, oct_id, pt_inside_oct), 1);
  EXPECT_EQ(mj_insideSite(model.get(), data, oct_id, pt_in_aabb_out_hull), 0);
  EXPECT_EQ(mj_insideSite(model.get(), data, oct_id, pt_out_aabb_x), 0);
  EXPECT_EQ(mj_insideSite(model.get(), data, oct_id, pt_out_aabb_y), 0);
  EXPECT_EQ(mj_insideSite(model.get(), data, oct_id, pt_out_aabb_z), 0);

  mj_deleteData(data);
}

TEST_F(SupportTest, GeomDistanceWitnessAtExactDistmax) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="g1" type="sphere" size="0.5" pos="-0.5 0 0"/>
      <geom name="g2" type="sphere" size="0.5" pos="1.5 0 0"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum fromto[6] = {-1, -1, -1, -1, -1, -1};
  mjtNum dist =
      mj_geomDistance(model.get(), data.get(), 0, 1, /*distmax=*/1.0, fromto);
  EXPECT_MJTNUM_EQ(dist, 1.0);
  EXPECT_THAT(fromto, Pointwise(MjNear(1e-12, 1e-5), {0, 0, 0, 1, 0, 0}));
}

}  // namespace
}  // namespace mujoco
