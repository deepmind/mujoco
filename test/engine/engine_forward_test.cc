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

// Tests for engine/engine_forward.c.

#include "src/engine/engine_forward.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_callback.h"
#include "src/engine/engine_core_util.h"
#include "src/engine/engine_derivative.h"
#include "src/engine/engine_io.h"
#include "src/engine/engine_memory.h"
#include "test/fixture.h"

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

namespace mujoco {
namespace {

static const char* const kEnergyConservingPendulumPath =
    "engine/testdata/derivative/energy_conserving_pendulum.xml";

// helper for precision-aware checks in macros (e.g. MJDATA_POINTERS)
template <typename T>
void ExpectNear(T a, T b) {
  EXPECT_EQ(a, b);
}

template <>
void ExpectNear<mjtNum>(mjtNum a, mjtNum b) {
  EXPECT_EQ(a, b);
}

static const char* const kDampedActuatorsPath =
    "engine/testdata/derivative/damped_actuators.xml";
static const char* const kJointForceClamp =
    "engine/testdata/actuation/joint_force_clamp.xml";
static const char* const kTendonForceClamp =
    "engine/testdata/actuation/tendon_force_clamp.xml";

using ::testing::Pointwise;

using ::testing::_;
using ::testing::AllOf;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Lt;
using ::testing::Ne;
using ::testing::NotNull;

// --------------------------- activation limits -------------------------------

struct ActLimitedTestCase {
  std::string test_name;
  mjtIntegrator integrator;
};

class ParametrizedForwardTest
    : public MujocoTest,
      public ::testing::WithParamInterface<ActLimitedTestCase> {};

TEST_P(ParametrizedForwardTest, ActLimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="slide" gainprm="100" biasprm="0 -100" biastype="affine"
      dynprm="10" dyntype="integrator"
      actlimited="true" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  model->opt.integrator = GetParam().integrator;

  data->ctrl[0] = 1.0;
  // integrating up from 0, we will hit the clamp after 99 steps
  for (int i = 0; i < 200; i++) {
    mj_step(model.get(), data.get());
    // always greater than lower bound
    EXPECT_GT(data->act[0], -1);
    // after 99 steps we hit the upper bound
    if (i < 99) EXPECT_LT(data->act[0], 1);
    if (i >= 99) EXPECT_NEAR(data->act[0], 1, MjTol(0, 5e-6));
  }

  data->ctrl[0] = -1.0;
  // integrating down from 1, we will hit the clamp after 199 steps
  for (int i = 0; i < 300; i++) {
    mj_step(model.get(), data.get());
    // always smaller than upper bound
    EXPECT_LT(data->act[0], model->actuator_actrange[1]);
    // after 199 steps we hit the lower bound
    if (i < 199) EXPECT_GT(data->act[0], model->actuator_actrange[0]);
    if (i >= 199) {
      EXPECT_NEAR(data->act[0], model->actuator_actrange[0], MjTol(0.0, 5e-6));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ParametrizedForwardTest, ParametrizedForwardTest,
    testing::ValuesIn<ActLimitedTestCase>({
        {"Euler", mjINT_EULER},
        {"Implicit", mjINT_IMPLICIT},
        {"Implicitfast", mjINT_IMPLICITFAST},
        {"RK4", mjINT_RK4},
        {"Discrete", mjINT_DISCRETE},
    }),
    [](const testing::TestParamInfo<ParametrizedForwardTest::ParamType>& info) {
      return info.param.test_name;
    });

// --------------------------- damping actuator --------------------------------

using ForwardTest = MujocoTest;

TEST_F(ForwardTest, DamperDampens) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt"/>
      <damper joint="jnt" kv="1000" ctrlrange="0 100"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // move the joint
  data->ctrl[0] = 100.0;
  data->ctrl[1] = 0.0;
  for (int i = 0; i < 100; i++) mj_step(model.get(), data.get());

  // stop the joint with damping
  data->ctrl[0] = 0.0;
  data->ctrl[1] = 100.0;
  for (int i = 0; i < 1000; i++) mj_step(model.get(), data.get());

  EXPECT_LE(data->qvel[0], std::numeric_limits<double>::epsilon());
}

static const char* const kArmatureEquivalencePath =
    "engine/testdata/armature_equivalence.xml";

// test that adding joint armature is equivalent to a coupled rotating mass with
// a gear ratio enforced by an equality
TEST_F(ForwardTest, ArmatureEquivalence) {
  const std::string xml_path = GetTestDataFilePath(kArmatureEquivalencePath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  // with actuators
  mjtNum qpos_mse = 0;
  int nstep = 0;
  while (data->time < 4) {
    data->ctrl[0] = data->ctrl[1] = mju_sin(2 * data->time);
    mj_step(model, data);
    nstep++;
    mjtNum err = data->qpos[0] - data->qpos[2];
    qpos_mse += err * err;
  }
  EXPECT_LT(mju_sqrt(qpos_mse / nstep), 1e-3);

  // no actuators
  model->opt.disableflags |= mjDSBL_ACTUATION;
  qpos_mse = 0;
  nstep = 0;
  mj_resetData(model, data);
  while (data->time < 4) {
    mj_step(model, data);
    nstep++;
    mjtNum err = data->qpos[0] - data->qpos[2];
    qpos_mse += err * err;
  }
  EXPECT_LT(mju_sqrt(qpos_mse / nstep), 1e-3);

  mj_deleteData(data);
  mj_deleteModel(model);
}

// --------------------------- implicit integrator -----------------------------

using ImplicitIntegratorTest = MujocoTest;

// Disabling implicit joint damping works as expected
TEST_F(ImplicitIntegratorTest, EulerDampDisable) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag eulerdamp="disable"/>
    </option>

    <worldbody>
      <body>
        <joint axis="1 0 0" damping="2"/>
        <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
        <body pos="0 .1 0">
          <joint axis="0 1 0" damping="1"/>
          <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // step once, call mj_forward, save qvel and qacc
  mj_step(model.get(), data.get());
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> qvel = AsVector(data->qvel, model->nv);
  std::vector<mjtNum> qacc = AsVector(data->qacc, model->nv);

  // second step
  mj_step(model.get(), data.get());

  // compute finite-difference acceleration
  std::vector<mjtNum> qacc_fd(model->nv);
  for (int i = 0; i < model->nv; i++) {
    qacc_fd[i] = (data->qvel[i] - qvel[i]) / model->opt.timestep;
  }
  // expect finite-differenced qacc to match to high precision
  EXPECT_THAT(qacc_fd, Pointwise(MjNear(1e-14, 1e-6), qacc));

  // reach the same initial state
  mj_resetData(model.get(), data.get());
  mj_step(model.get(), data.get());

  // second step again, but with implicit integration of joint damping
  model->opt.disableflags &= ~mjDSBL_EULERDAMP;
  mj_step(model.get(), data.get());

  // compute finite-difference acceleration difference
  std::vector<mjtNum> dqacc(model->nv);
  for (int i = 0; i < model->nv; i++) {
    dqacc[i] = (data->qvel[i] - qvel[i]) / model->opt.timestep;
  }
  // expect finite-differenced qacc to not match
  EXPECT_GT(mju_norm(dqacc.data(), model->nv), 1);
}

// Reducing timesteps reduces the difference between implicit/explicit
TEST_F(ImplicitIntegratorTest, EulerDampLimit) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint axis="1 0 0" damping="2"/>
        <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
        <body pos="0 .1 0">
          <joint axis="0 1 0" damping="1"/>
          <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mjtNum diff_norm_prev = -1;
  for (const mjtNum dt : {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8}) {
    // set timestep
    model->opt.timestep = dt;

    // step twice with implicit damping, save qvel
    model->opt.disableflags &= ~mjDSBL_EULERDAMP;
    mj_resetData(model.get(), data.get());
    mj_step(model.get(), data.get());
    mj_step(model.get(), data.get());
    std::vector<mjtNum> qvel_imp = AsVector(data->qvel, model->nv);

    // step once, step again without implicit damping, save qvel
    mj_resetData(model.get(), data.get());
    mj_step(model.get(), data.get());
    model->opt.disableflags |= mjDSBL_EULERDAMP;
    mj_step(model.get(), data.get());
    std::vector<mjtNum> qvel_exp = AsVector(data->qvel, model->nv);

    mjtNum diff_norm = 0;
    for (int i = 0; i < model->nv; i++) {
      diff_norm += (qvel_imp[i] - qvel_exp[i]) * (qvel_imp[i] - qvel_exp[i]);
    }
    diff_norm = mju_sqrt(diff_norm);

    if (diff_norm_prev != -1) {
      EXPECT_LT(diff_norm, diff_norm_prev);
    }

    diff_norm_prev = diff_norm;
  }
}

// Euler and implicit should be equivalent if there is only joint damping
TEST_F(ImplicitIntegratorTest, EulerImplicitEquivalent) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint axis="1 0 0" damping="2"/>
        <geom type="capsule" size=".01" fromto="0 0 0 0 .1 0"/>
        <body pos="0 .1 0">
          <joint axis="0 1 0" damping="1"/>
          <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // step 10 times with Euler, save copy of qpos as vector
  for (int i = 0; i < 10; i++) {
    mj_step(model.get(), data.get());
  }
  std::vector<mjtNum> qposEuler = AsVector(data->qpos, model->nq);

  // reset, step 10 times with implicit
  mj_resetData(model.get(), data.get());
  model->opt.integrator = mjINT_IMPLICIT;
  for (int i = 0; i < 10; i++) {
    mj_step(model.get(), data.get());
  }

  // expect qpos vectors to be numerically different
#ifndef mjUSESINGLE
  EXPECT_THAT(AsVector(data->qpos, model->nq), Pointwise(Ne(), qposEuler));
#endif

  // expect qpos vectors to be similar to high precision
  EXPECT_THAT(AsVector(data->qpos, model->nq),
              Pointwise(MjNear(1e-14, 1e-6), qposEuler));
}

// Joint and actuator damping should integrate identically under implicit
TEST_F(ImplicitIntegratorTest, JointActuatorEquivalent) {
  const std::string xml_path = GetTestDataFilePath(kDampedActuatorsPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  // take 1000 steps with Euler
  for (int i = 0; i < 1000; i++) {
    mj_step(model, data);
  }
  // expect corresponding joint values to be significantly different
#ifndef mjUSESINGLE
  EXPECT_GT(fabs(data->qpos[0] - data->qpos[2]), 1e-4);
  EXPECT_GT(fabs(data->qpos[1] - data->qpos[3]), 1e-4);
#endif

  // reset, take 10 steps with implicit
  mj_resetData(model, data);
  model->opt.integrator = mjINT_IMPLICIT;
  for (int i = 0; i < 10; i++) {
    mj_step(model, data);
  }

  // expect corresponding joint values to be insignificantly different
  EXPECT_LT(fabs(data->qpos[0] - data->qpos[2]), MjTol(1e-16, 1e-6));
  EXPECT_LT(fabs(data->qpos[1] - data->qpos[3]), MjTol(1e-16, 1e-6));

  mj_deleteData(data);
  mj_deleteModel(model);
}

// Energy conservation: RungeKutta > implicit > Euler
TEST_F(ImplicitIntegratorTest, EnergyConservation) {
  const std::string xml_path =
      GetTestDataFilePath(kEnergyConservingPendulumPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  const int nstep = 500;  // number of steps to take

  // take nstep steps with Euler, measure energy (potential + kinetic)
  model->opt.integrator = mjINT_EULER;
  for (int i = 0; i < nstep; i++) {
    mj_step(model, data);
  }
  mjtNum energyEuler = data->energy[0] + data->energy[1];

  // take nstep steps with implicit, measure energy
  model->opt.integrator = mjINT_IMPLICIT;
  mj_resetData(model, data);
  for (int i = 0; i < nstep; i++) {
    mj_step(model, data);
  }
  mjtNum energyImplicit = data->energy[0] + data->energy[1];

  // take nstep steps with 4th order Runge-Kutta, measure energy
  model->opt.integrator = mjINT_RK4;
  mj_resetData(model, data);
  for (int i = 0; i < nstep; i++) {
    mj_step(model, data);
  }
  mjtNum energyRK4 = data->energy[0] + data->energy[1];

  // energy was measured: expect all energies to be nonzero
  EXPECT_NE(energyEuler, 0);
  EXPECT_NE(energyImplicit, 0);
  EXPECT_NE(energyRK4, 0);

  // test conservation: perfectly conserved energy would remain 0.0
  // expect RK4 to be better than implicit
  EXPECT_LT(fabs(energyRK4), fabs(energyImplicit));
  // expect implicit to be better than Euler
  EXPECT_LT(fabs(energyImplicit), fabs(energyEuler));

  mj_deleteData(data);
  mj_deleteModel(model);
}

// free-body local solve: implicitfast matches implicit exactly for a standalone
// free body
TEST_F(ImplicitIntegratorTest, FreeBodyMatchesImplicit) {
  // damped free body in vacuum
  static constexpr char xml1[] = R"(
  <mujoco>
    <option timestep="0.005"/>
    <worldbody>
      <body pos="0.1 -0.2 0.5" euler="20 -30 40">
        <joint type="free" damping="0.1"/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // free body in fluid with wind, ellipsoid fluid model (asymmetric lift
  // derivatives)
  static constexpr char xml2[] = R"(
  <mujoco>
    <option timestep="0.005" density="1.2" viscosity="0.002" wind="1 2 3"/>
    <worldbody>
      <body pos="0.1 -0.2 0.5" euler="20 -30 40">
        <joint type="free"/>
        <geom type="ellipsoid" size=".1 .2 .3" mass="2" pos=".04 -.02 .03"
              fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // free rigid subtree with rotated, offset inertias and multiple levels
  static constexpr char xml3[] = R"(
  <mujoco>
    <option timestep="0.005"/>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody>
      <body pos="0.1 -0.2 0.5" euler="20 -30 40">
        <joint type="free" damping="0.1"/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
        <body pos=".2 -.1 .3" euler="30 20 -10">
          <geom type="box" size=".2 .1 .1" mass="1" pos=".03 .01 -.02"/>
          <body pos="-.1 .2 .1" euler="10 -20 30">
            <geom type="box" size=".1 .1 .2" mass=".5"/>
          </body>
        </body>
      </body>
      <body pos="3 0 1">
        <freejoint/>
        <geom type="box" size=".1 .2 .3"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // fluid forces on a fixed child must retain asymmetric lift derivatives
  static constexpr char xml4[] = R"(
  <mujoco>
    <option timestep="0.005" density="1.2" viscosity="0.002" wind="1 2 3"/>
    <worldbody>
      <body pos="0.1 -0.2 0.5" euler="20 -30 40">
        <freejoint/>
        <body pos=".04 -.02 .03" euler="10 20 30">
          <geom type="ellipsoid" size=".1 .2 .3" mass="2" fluidshape="ellipsoid"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  int xml_idx = 1;
  for (auto xml : {xml1, xml2, xml3, xml4}) {
    SCOPED_TRACE(testing::Message() << "XML case " << xml_idx++);
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr d1 = MakeData(model);
    MjDataPtr d2 = MakeData(model);
    mjModel* m = model.get();

    // tumbling initial velocity
    mj_resetData(m, d1.get());
    d1->qvel[3] = 5;
    d1->qvel[4] = -3;
    d1->qvel[5] = 2;

    // step both integrators from identical states, re-synchronizing each step
    // to avoid chaotic divergence of tumbling trajectories
    int nstate = mj_stateSize(m, mjSTATE_INTEGRATION);
    std::vector<mjtNum> state(nstate);
    for (int i = 0; i < 50; i++) {
      mj_getState(m, d1.get(), state.data(), mjSTATE_INTEGRATION);
      mj_setState(m, d2.get(), state.data(), mjSTATE_INTEGRATION);

      m->opt.integrator = mjINT_IMPLICITFAST;
      mj_step(m, d1.get());
      m->opt.integrator = mjINT_IMPLICIT;
      mj_step(m, d2.get());

      for (int k = 0; k < m->nv; k++) {
        EXPECT_NEAR(d1->qvel[k], d2->qvel[k], MjTol(1e-14, 1e-6))
            << "step " << i << " dof " << k;
      }
    }
  }
}

// free-body local solve: spinning free bodies do not gain energy in vacuum
TEST_F(ImplicitIntegratorTest, FreeBodyGyroStable) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.005">
      <flag energy="enable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  // middle-axis tumble and fast principal-axis spin
  static constexpr mjtNum qvel0[2][3] = {{0.05, 5, 0.05}, {20, 0.05, 0.05}};

  for (int c = 0; c < 2; c++) {
    SCOPED_TRACE(testing::Message() << "velocity case " << c);
    mj_resetData(m, d);
    mju_copy3(d->qvel + 3, qvel0[c]);
    mj_forward(m, d);
    mjtNum initial_energy = d->energy[1];

    // 100 simulated seconds
    for (int i = 0; i < 20000; i++) {
      mj_step(m, d);
      ASSERT_LT(d->energy[1], 1.01 * initial_energy)
          << "energy gain at step " << i;
    }
  }
}

// a massless free root with a fixed inertial child must not disable gyroscopic
// stabilization
TEST_F(ImplicitIntegratorTest, FreeRigidSubtreeGyroStable) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.001" gravity="0 0 0">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <body>
          <inertial pos="0 0 0" mass="0.12"
                    diaginertia="0.00017231 0.00000658 0.00017243"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nbody, 3);
  MjDataPtr data = MakeData(model);
  for (mjtIntegrator integrator : {mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    SCOPED_TRACE(testing::Message() << "integrator " << integrator);
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qvel[3] = 100;
    data->qvel[4] = 30;
    data->qvel[5] = 100;
    mj_forward(model.get(), data.get());
    mjtNum initial_energy = data->energy[1];

    for (int i = 0; i < 10000; i++) {
      mj_step(model.get(), data.get());
      ASSERT_EQ(data->warning[mjWARN_BADQACC].number, 0) << "step " << i;
      ASSERT_LT(data->energy[1], 1.01 * initial_energy) << "step " << i;
    }
  }
}

// free-body local solve: applies to bodies in contact
TEST_F(ImplicitIntegratorTest, FreeBodyGyroStableContact) {
  // spinning ellipsoid on an inclined plane, as in gyroscopic.xml
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.002"/>
    <worldbody>
      <geom type="plane" size="5 5 .1" euler="0 15 0"/>
      <body pos="0 0 .2">
        <freejoint/>
        <geom type="ellipsoid" size=".05 .1 .15" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  mj_resetData(m, d);
  d->qvel[3] = 30;
  mjtNum initial_speed = mju_norm(d->qvel, m->nv);

  int ncon_total = 0;
  for (int i = 0; i < 5000; i++) {
    mj_step(m, d);
    ncon_total += d->ncon;
    ASSERT_LT(mju_norm(d->qvel, m->nv), 2 * initial_speed)
        << "speed gain at step " << i;
  }

  // the body was in contact while spinning
  EXPECT_GT(ncon_total, 1000);
}

// free-body local solve: energy of a tumbling free body never increases and is
// only mildly damped; angular momentum drift is bounded
TEST_F(ImplicitIntegratorTest, FreeBodyConservation) {
  // aligned: CoM at joint origin
  static constexpr char xml1[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.01">
      <flag energy="enable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // non-aligned: CoM offset from joint origin
  static constexpr char xml2[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.01">
      <flag energy="enable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1" euler="10 20 30" pos=".03 .02 .01"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  int xml_idx = 1;
  for (auto xml : {xml1, xml2}) {
    SCOPED_TRACE(testing::Message() << "XML case " << xml_idx++);
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    mjModel* m = model.get();
    mjData* d = data.get();

    mj_resetData(m, d);
    d->qvel[3] = 1.0;
    d->qvel[4] = 2.0;
    d->qvel[5] = 3.0;
    mj_forward(m, d);
    mjtNum initial_energy = d->energy[1];
    mjtNum initial_angmom[3];
    mj_subtreeVel(m, d);
    mju_copy3(initial_angmom, d->subtree_angmom);

    for (int i = 0; i < 500; i++) {
      mj_step(m, d);

      // energy never increases (small tolerance for rounding)
      ASSERT_LT(d->energy[1], initial_energy * (1 + MjTol(1e-9, 1e-4)))
          << "energy gain at step " << i;
    }

    // implicit damping of tumbling is mild: measured E_end/E0 = 0.93
    EXPECT_GT(d->energy[1], 0.7 * initial_energy);

    // angular momentum drift is bounded: measured 5e-3
    mj_subtreeVel(m, d);
    mjtNum angmom_err[3];
    mju_sub3(angmom_err, d->subtree_angmom, initial_angmom);
    EXPECT_LT(mju_norm3(angmom_err), 0.05);
  }
}

// gyroscopic instability: Euler gains energy where implicitfast and discrete do
// not
TEST_F(ImplicitIntegratorTest, FreeBodyEulerGainsImplicitfastDissipates) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01">
      <flag energy="enable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  mjtNum energy_end[3];
  int index = 0;
  for (int integrator : {mjINT_EULER, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    m->opt.integrator = integrator;
    mj_resetData(m, d);
    d->qvel[3] = 1.0;
    d->qvel[4] = 2.0;
    d->qvel[5] = 3.0;
    mj_forward(m, d);
    mjtNum initial_energy = d->energy[1];
    for (int i = 0; i < 500; i++) {
      mj_step(m, d);
    }
    energy_end[index++] = d->energy[1] / initial_energy;
  }

  // Euler gains energy (measured: 1.09), implicitfast and discrete do not
  EXPECT_GT(energy_end[0], 1.01);
  EXPECT_LT(energy_end[1], 1.0);
  EXPECT_LT(energy_end[2], 1.0);
}

// the invdiscrete flag has no effect on forward dynamics
TEST_F(ImplicitIntegratorTest, InvdiscreteForwardNoop) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.005"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .3">
        <joint type="free" damping="0.1"/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".03 .02 .01"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(model);
  MjDataPtr d2 = MakeData(model);
  mjModel* m = model.get();

  for (int integrator : {mjINT_IMPLICITFAST, mjINT_IMPLICIT, mjINT_DISCRETE}) {
    m->opt.integrator = integrator;

    mj_resetData(m, d1.get());
    d1->qvel[3] = 5;
    d1->qvel[5] = 2;
    mj_resetData(m, d2.get());
    d2->qvel[3] = 5;
    d2->qvel[5] = 2;

    for (int i = 0; i < 200; i++) {
      m->opt.enableflags &= ~mjENBL_INVDISCRETE;
      mj_step(m, d1.get());
      m->opt.enableflags |= mjENBL_INVDISCRETE;
      mj_step(m, d2.get());
    }
    m->opt.enableflags &= ~mjENBL_INVDISCRETE;

    // trajectories are bit-identical
    for (int k = 0; k < m->nq; k++) {
      EXPECT_EQ(d1->qpos[k], d2->qpos[k]) << "qpos " << k;
    }
    for (int k = 0; k < m->nv; k++) {
      EXPECT_EQ(d1->qvel[k], d2->qvel[k]) << "qvel " << k;
    }
  }
}

// model with degenerate translational inertia
TEST_F(ForwardTest, DegenerateInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" cone="elliptic">
      <flag gravity="disable" diagexact="enable"/>
    </option>
    <worldbody>
      <body name="1" pos="0.05 0.3 0">
        <joint name="1" axis="0 1 0"/>
        <geom type="capsule" size="0.1 0.5"/>
      </body>
      <body name="2">
        <joint name="2" axis="1 0 0" stiffness="1" springref="90"/>
        <geom type="capsule" size="0.1 0.5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  for (int i = 0; i < 1000; i++) {
    mj_step(model.get(), data.get());
    EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0)
        << "divergence at timestep " << i;
    if (data->warning[mjWARN_BADQACC].number != 0) {
      break;
    }
  }
}

// a singular modified inertia M - h*qDeriv is a warning under both implicit
// integrators: implicitfast clamps the LDL pivot, implicit clamps the LU pivot
TEST_F(ForwardTest, SingularModifiedInertiaWarns) {
  mock_warning_handler.ExpectWarnings(
      "Inertia matrix is too close to singular");

  // the affine velocity gain of 1 makes M - h*qDeriv exactly singular at h=1,
  // once a unit control brings the early activation to 1
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1" integrator="implicit"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="slide" dyntype="integrator" gaintype="affine" gainprm="1 0 1"
               actearly="true"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  for (int integrator : {mjINT_IMPLICIT, mjINT_IMPLICITFAST}) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->ctrl[0] = 1;
    mj_step(model.get(), data.get());
    EXPECT_EQ(data->warning[mjWARN_INERTIA].number, 1) << integrator;
    EXPECT_EQ(data->warning[mjWARN_INERTIA].lastinfo, 0);
    EXPECT_TRUE(std::isfinite(data->qvel[0]));
  }
}

TEST_F(ForwardTest, ControlClamping) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="slide" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="unclamped" joint="slide"/>
      <motor name="clamped" joint="slide" ctrllimited="true" ctrlrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // for the unclamped actuator, ctrl={1, 2} produce different accelerations
  data->ctrl[0] = 1;
  mj_forward(model.get(), data.get());
  mjtNum qacc1 = data->qacc[0];
  data->ctrl[0] = 2;
  mj_forward(model.get(), data.get());
  mjtNum qacc2 = data->qacc[0];
  EXPECT_NE(qacc1, qacc2);

  // for the clamped actuator, ctrl={1, 2} produce identical accelerations
  data->ctrl[1] = 1;
  mj_forward(model.get(), data.get());
  qacc1 = data->qacc[0];
  data->ctrl[1] = 2;
  mj_forward(model.get(), data.get());
  qacc2 = data->qacc[0];
  EXPECT_EQ(qacc1, qacc2);

  // data->ctrl[1] remains pristine
  EXPECT_EQ(data->ctrl[1], 2);

  MockWarningHandler warning_handler;

  // for the unclamped actuator, huge raises warning
  warning_handler.ExpectWarnings(
      "Nan, Inf or huge value in CTRL at ACTUATOR 0");
  data->ctrl[0] = 10 * mjMAXVAL;
  mj_forward(model.get(), data.get());
  testing::Mock::VerifyAndClearExpectations(&warning_handler);

  // for the clamped actuator, huge does not raise warning
  EXPECT_CALL(warning_handler, Warn(_)).Times(0);
  mj_resetData(model.get(), data.get());
  data->ctrl[1] = 10 * mjMAXVAL;
  mj_forward(model.get(), data.get());
  testing::Mock::VerifyAndClearExpectations(&warning_handler);

  // for the clamped actuator, NaN raises warning
  warning_handler.ExpectWarnings(
      "Nan, Inf or huge value in CTRL at ACTUATOR 1");
  mj_resetData(model.get(), data.get());
  data->ctrl[1] = std::numeric_limits<double>::quiet_NaN();
  mj_forward(model.get(), data.get());
}

void control_callback(const mjModel* m, mjData* d) { d->ctrl[0] = 2; }

TEST_F(ForwardTest, MjcbControlDisabled) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // install global control callback
  mjcb_control = control_callback;

  // call forward
  mj_forward(model.get(), data.get());
  // expect that callback was used
  EXPECT_EQ(data->ctrl[0], 2.0);

  // reset, disable actuation, call forward
  mj_resetData(model.get(), data.get());
  model->opt.disableflags |= mjDSBL_ACTUATION;
  mj_forward(model.get(), data.get());
  // expect that callback was not used
  EXPECT_EQ(data->ctrl[0], 0.0);

  // same for mj_step1
  mj_resetData(model.get(), data.get());
  mj_step1(model.get(), data.get());
  EXPECT_EQ(data->ctrl[0], 0.0);
  model->opt.disableflags &= ~mjDSBL_ACTUATION;
  mj_step1(model.get(), data.get());
  EXPECT_EQ(data->ctrl[0], 2.0);

  // remove global control callback
  mjcb_control = nullptr;
}

// filtered position servos on a ball joint track a target wound through pi
TEST_F(ForwardTest, FilteredPositionBallTracksWindingTarget) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast">
      <flag contact="disable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <geom type="box" size=".05 .07 .03"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="ball" gear="1 0 0" kp="1" dampratio="1"
        timeconst="0.01"/>
      <position joint="ball" gear="0 1 0" kp="1" dampratio="1"
        timeconst="0.01"/>
      <position joint="ball" gear="0 0 1" kp="1" dampratio="1"
        timeconst="0.01"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  for (mjtDyn dyntype : {mjDYN_FILTEREXACT, mjDYN_FILTER}) {
    for (int i = 0; i < m->nu; i++) {
      m->actuator_dyntype[i] = dyntype;
    }
    mj_resetData(m, d);

    // ramp the z target from 0 to 2*pi, expect tracking on the circle
    const mjtNum rate = 0.5;  // rad/s
    mjtNum max_error = 0;
    while (d->time < 2 * mjPI / rate) {
      d->ctrl[2] = rate * d->time;
      mj_step(m, d);
      mjtNum err = d->ctrl[2] - d->actuator_length[2];
      err -= 2 * mjPI * mju_round(err / (2 * mjPI));
      max_error = mju_max(max_error, mju_abs(err));
    }
    EXPECT_LT(max_error, 0.5) << "dyntype " << dyntype;

    // the filter state is kept as the representative nearest the length
    EXPECT_LE(mju_abs(d->act[2] - d->actuator_length[2]), mjPI);
  }
}

TEST_F(ForwardTest, gravcomp) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -10" />
    <worldbody>
      <body>
        <joint type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
      <body pos="3 0 0" gravcomp="1">
        <joint type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
      <body pos="6 0 0" gravcomp="2">
        <joint type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data = MakeData(model);
  while (data->time < 1) {
    mj_step(model.get(), data.get());
  }

  mjtNum dist = 0.5 * mju_norm3(model->opt.gravity) * (data->time * data->time);

  // expect that body 1 moved down, allowing some slack from our estimate
  EXPECT_NEAR(data->qpos[0], -dist, 0.011);

  // expect that body 2 does not move
  EXPECT_EQ(data->qpos[1], 0.0);

  // expect that body 3 moves up the same distance that body 0 moved down
  EXPECT_EQ(data->qpos[0], -data->qpos[2]);
}

// test disabling of equality constraints
TEST_F(ForwardTest, eq_active) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="vertical" type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="vertical"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data = MakeData(model);

  // simulate for 1 second
  while (data->time < 1) {
    mj_step(model.get(), data.get());
  }

  // expect that the body has barely moved
  EXPECT_LT(mju_abs(data->qpos[0]), 0.001);

  // turn the equality off, simulate for another second
  data->eq_active[0] = 0;
  while (data->time < 2) {
    mj_step(model.get(), data.get());
  }

  // expect that the body has fallen about 5m
  EXPECT_LT(data->qpos[0], -4.5);
  EXPECT_GT(data->qpos[0], -5.5);

  // turn the equality back on, simulate for another second
  data->eq_active[0] = 1;
  while (data->time < 3) {
    mj_step(model.get(), data.get());
  }

  // expect that the body has snapped back
  EXPECT_LT(mju_abs(data->qpos[0]), 0.001);
}

// test that normalized and denormalized quats give the same result
TEST_F(ForwardTest, NormalizeQuats) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "Skipping in float32: exact mjData comparison infeasible.";
#endif
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicit">
      <flag warmstart="disable" energy="enable"/>
    </option>
    <worldbody>
      <body name="free">
        <freejoint/>
        <geom size="1" pos=".1 .2 .3"/>
      </body>
      <body pos="3 0 0">
        <joint name="ball" type="ball" stiffness="100" range="0 10"/>
        <geom size="1" pos=".1 .2 .3"/>
      </body>
    </worldbody>
    <sensor>
      <ballquat joint="ball"/>
      <framequat objtype="body" objname="free"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data_u = MakeData(model);

// we'll compare all the memory, so unpoison it first
#ifdef MEMORY_SANITIZER
  __msan_unpoison(data_u->buffer, data_u->nbuffer);
  __msan_unpoison(data_u->arena, data_u->narena);
#endif

  // set quats to denormalized values, non-zero velocities
  for (int i = 3; i < model->nq; i++) data_u->qpos[i] = i;
  for (int i = 0; i < model->nv; i++) data_u->qvel[i] = 0.1 * i;

  // copy data and normalize quats
  mjData* data_n = mj_copyData(nullptr, model.get(), data_u.get());
  mj_normalizeQuat(model.get(), data_n->qpos);

  // call forward, expect quats to be untouched
  mj_forward(model.get(), data_u.get());
  for (int i = 3; i < model->nq; i++) {
    EXPECT_EQ(data_u->qpos[i], (mjtNum)i);
  }

  // expect that the ball joint limit is active
  EXPECT_EQ(data_u->nl, 1);

  // step both models
  mj_step(model.get(), data_u.get());
  mj_step(model.get(), data_n);

// expect everything to match
#define X(type, name, nr, nc)         \
  for (int i = 0; i < model->nr; i++) \
    for (int j = 0; j < nc; j++)      \
      ExpectNear(data_n->name[i * nc + j], data_u->name[i * nc + j]);
  MJDATA_POINTERS;
#undef X

  // repeat the above with RK4 integrator
  model->opt.integrator = mjINT_RK4;

  // reset data, unpoison
  mj_resetData(model.get(), data_u.get());
#ifdef MEMORY_SANITIZER
  __msan_unpoison(data_u->buffer, data_u->nbuffer);
  __msan_unpoison(data_u->arena, data_u->narena);
#endif

  // set quats to un-normalized values, non-zero velocities
  for (int i = 3; i < model->nq; i++) data_u->qpos[i] = i;
  for (int i = 0; i < model->nv; i++) data_u->qvel[i] = 0.1 * i;

  // copy data and normalize quats
  mj_copyData(data_n, model.get(), data_u.get());
  mj_normalizeQuat(model.get(), data_n->qpos);

  // step both models
  mj_step(model.get(), data_u.get());
  mj_step(model.get(), data_n);

// expect everything to match
#define X(type, name, nr, nc)         \
  for (int i = 0; i < model->nr; i++) \
    for (int j = 0; j < nc; j++)      \
      ExpectNear(data_n->name[i * nc + j], data_u->name[i * nc + j]);
  MJDATA_POINTERS;
#undef X

  mj_deleteData(data_n);
}

// test that normalized and denormalized quats give the same result
TEST_F(ForwardTest, MocapQuats) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="mocap" mocap="true" quat="1 1 1 1">
        <geom size="1"/>
      </body>
    </worldbody>
    <sensor>
      <framequat objtype="body" objname="mocap"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // expect mocap_quat to be normalized (by the compiler)
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(data->mocap_quat[i], 0.5, MjTol(0, 1e-6));
    EXPECT_NEAR(data->xquat[4 + i], 0.5, MjTol(0, 1e-6));
  }

  // write denormalized quats to mocap_quat, call forward again
  for (int i = 0; i < 4; i++) {
    data->mocap_quat[i] = 1;
  }
  mj_forward(model.get(), data.get());

  // expect mocap_quat to remain denormalized, but xquat to be normalized
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(data->mocap_quat[i], 1, MjTol(0, 1e-6));
    EXPECT_NEAR(data->xquat[4 + i], 0.5, MjTol(0, 1e-6));
  }
}

// user defined 2nd-order activation dynamics: frequency-controlled oscillator
//  note that scalar mjcb_act_dyn callbacks are expected to return act_dot, but
//  since we have a vector output we write into act_dot directly
mjtNum oscillator(const mjModel* m, const mjData* d, int id) {
  // check that actnum == 2
  if (m->actuator_actnum[id] != 2) {
    mju_error("callback expected actnum == 2");
  }

  // get pointers to activations (inputs) and their derivatives (outputs)
  mjtNum* act = d->act + m->actuator_actadr[id];
  mjtNum* act_dot = d->act_dot + m->actuator_actadr[id];

  // harmonic oscillator with controlled frequency
  mjtNum frequency = 2 * mjPI * d->ctrl[id];
  act_dot[0] = -act[1] * frequency;
  act_dot[1] = act[0] * frequency;

  return 0;  // ignored by caller
}

TEST_F(ForwardTest, MjcbActDynSecondOrderExpectsActnum) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1e-4"/>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" dyntype="user" actdim="2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // install global dynamics callback
  mjcb_act_dyn = oscillator;

  // for two arbitrary frequencies, compare actuator force as output by the
  // user-defined oscillator and analytical sine function
  for (mjtNum frequency : {1.5, 0.7}) {
    mj_resetData(model.get(), data.get());
    data->ctrl[0] = frequency;  // set desired oscillation frequency
    data->act[0] = 1;           // initialise activation

    // simulate and compare to sine function
    while (data->time < 1) {
      mjtNum expected_force = mju_sin(2 * mjPI * data->time * frequency);
      mj_step(model.get(), data.get());
      EXPECT_NEAR(data->actuator_force[0], expected_force, .01);
    }
  }

  // uninstall global dynamics callback
  mjcb_act_dyn = nullptr;
}

// ------------------------------ actuators -----------------------------------

using ActuatorTest = MujocoTest;

TEST_F(ActuatorTest, ExpectedAdhesionForce) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -1"/>

    <worldbody>
      <body name="static">
        <!-- small increase to size to ensure contact -->
        <geom size=".02001" pos=" .01  .01 .07"/>
        <geom size=".02001" pos="-.01  .01 .07"/>
        <geom size=".02001" pos=" .01 -.01 .07"/>
        <geom size=".02001" pos="-.01 -.01 .07"/>
      </body>
      <body name="free">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <adhesion body="static" ctrlrange="0 2"/>
      <adhesion body="free" ctrlrange="0 2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // iterate over cone type
  for (mjtCone cone : {mjCONE_ELLIPTIC, mjCONE_PYRAMIDAL}) {
    // set cone
    model->opt.cone = cone;
    // iterate over condim
    for (int condim : {1, 3, 4, 6}) {
      // set condim
      for (int id = 0; id < model->ngeom; id++) {
        model->geom_condim[id] = condim;
      }
      // iterate over actuators
      for (int id = 0; id < 2; id++) {
        // set ctrl > 1, expect free body to not fall
        mj_resetData(model.get(), data.get());
        data->ctrl[id] = 1.01;
        for (int i = 0; i < 100; i++) {
          mj_step(model.get(), data.get());
        }
        // moved down at most 10 microns
        EXPECT_GT(data->qpos[2], -1e-5);

        // set ctrl < 1, expect free body to fall below 1cm
        mj_resetData(model.get(), data.get());
        data->ctrl[id] = 0.99;
        for (int i = 0; i < 100; i++) {
          mj_step(model.get(), data.get());
        }
        // fell lower than 1cm
        EXPECT_LT(data->qpos[2], -0.01);
      }
    }
  }
}

// Actuator force clamping at joints
TEST_F(ActuatorTest, ActuatorForceClamping) {
  const std::string xml_path = GetTestDataFilePath(kJointForceClamp);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  data->ctrl[0] = 10;
  mj_forward(model, data);

  // expect clamping as specified in the model
  EXPECT_NEAR(data->actuator_force[0], 1, MjTol(0, 1e-6));
  EXPECT_NEAR(data->qfrc_actuator[0], 0.4, MjTol(0, 1e-6));

  // simulate for 2 seconds to gain velocity
  while (data->time < 2) {
    mj_step(model, data);
  }

  // activate damper, expect force to be clamped at lower bound
  data->ctrl[1] = 1;
  mj_forward(model, data);
  EXPECT_NEAR(data->qfrc_actuator[0], -0.4, MjTol(0, 1e-6));

  mj_deleteData(data);
  mj_deleteModel(model);
}

// Apply gravity compensation via actuators
TEST_F(ActuatorTest, ActuatorGravcomp) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body gravcomp="1">
        <joint name="joint" type="slide" axis="0 0 1"
               actuatorfrcrange="-2 2" actuatorgravcomp="true"/>
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <motor name="actuator" joint="joint"/>
    </actuator>

    <sensor>
      <actuatorfrc actuator="actuator"/>
      <jointactuatorfrc joint="joint"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());

  // expect force clamping as specified in the model
  EXPECT_EQ(data->actuator_force[0], 0);
  EXPECT_EQ(data->qfrc_actuator[0], 2);
  EXPECT_EQ(data->qfrc_passive[0], 0);
  EXPECT_EQ(data->sensordata[0], 0);
  EXPECT_EQ(data->sensordata[1], 2);

  // reduce gravity so gravcomp is not clamped
  model->opt.gravity[2] = -1;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->actuator_force[0], 0);
  EXPECT_EQ(data->qfrc_actuator[0], 1);
  EXPECT_EQ(data->qfrc_passive[0], 0);
  EXPECT_EQ(data->sensordata[0], 0);
  EXPECT_EQ(data->sensordata[1], 1);

  // add control, see that it adds up
  data->ctrl[0] = 0.5;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->actuator_force[0], 0.5);
  EXPECT_EQ(data->qfrc_actuator[0], 1.5);
  EXPECT_EQ(data->qfrc_passive[0], 0);
  EXPECT_EQ(data->sensordata[0], 0.5);
  EXPECT_EQ(data->sensordata[1], 1.5);

  // add larger control, expect clamping
  data->ctrl[0] = 1.5;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->actuator_force[0], 1.5);
  EXPECT_EQ(data->qfrc_actuator[0], 2);
  EXPECT_EQ(data->qfrc_passive[0], 0);
  EXPECT_EQ(data->sensordata[0], 1.5);
  EXPECT_EQ(data->sensordata[1], 2);

  // disable actgravcomp, expect gravcomp as a passive force
  model->jnt_actgravcomp[0] = 0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->actuator_force[0], 1.5);
  EXPECT_EQ(data->qfrc_actuator[0], 1.5);
  EXPECT_EQ(data->qfrc_passive[0], 1);
  EXPECT_EQ(data->sensordata[0], 1.5);
  EXPECT_EQ(data->sensordata[1], 1.5);
}

// Check that dampratio works as expected
TEST_F(ActuatorTest, DampRatio) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast"/>

    <worldbody>
      <body>
        <joint name="slide1" axis="1 0 0" type="slide"/>
        <geom size=".05"/>
      </body>

      <body pos="0 0 -.15">
        <joint name="slide2" axis="1 0 0" type="slide"/>
        <geom size=".05"/>
      </body>
    </worldbody>

    <actuator>
      <position name="slightly underdamped" joint="slide1" kp="10" dampratio="0.99"/>
      <position name="slightly overdamped"  joint="slide2" kp="10" dampratio="1.01"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  data->qpos[0] = data->qpos[1] = -0.1;

  mjtNum under_damped = data->qpos[0];
  mjtNum over_damped = data->qpos[1];
  while (data->time < 10) {
    mj_step(model.get(), data.get());
    under_damped = mju_max(under_damped, data->qpos[0]);
    over_damped = mju_max(over_damped, data->qpos[1]);
  }

  // expect slightly underdamped to slightly overshoot
  EXPECT_GT(under_damped, 0);
  EXPECT_LT(under_damped, 1e-6);

  // expect slightly overdamped to slightly undershoot
  EXPECT_LT(over_damped, 0);
  EXPECT_GT(over_damped, -1e-6);
}

// Check dampratio for actuators with nontrivial transmission
TEST_F(ActuatorTest, DampRatioTendon) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/actuation/tendon_dampratio.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  data->ctrl[0] = 1;
  data->ctrl[1] = 4;

  while (data->time < 1) {
    mj_step(model, data);
  }

  // expect first and second fingers to move together
  double tol = 1e-10;
  EXPECT_THAT(AsVector(data->qpos, 4),
              Pointwise(MjNear(tol, tol), AsVector(data->qpos + 4, 4)));
  EXPECT_THAT(AsVector(data->qvel, 4),
              Pointwise(MjNear(tol, tol), AsVector(data->qvel + 4, 4)));

  mj_deleteData(data);
  mj_deleteModel(model);
}

// ----------------------- DC motor actuators ----------------------------------

using DCMotorTest = MujocoTest;

// A stateless dcmotor with setpoint inputs matches <pid> exactly, for any
// motor constant and resistance: the torque-space controller commands
// kp*(qref - l) + kd*(vref - ldot) and the tau->V map compensates back-EMF.
TEST_F(DCMotorTest, SetpointMatchesPid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001" integrator="implicitfast">
      <flag contact="disable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="slide1" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="slide2" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid name="pid" joint="slide1" kp="10" kv="5"/>
      <dcmotor name="dcmotor" joint="slide2" motorconst="0.05" resistance="2.0"
               input="pos vel" controller="10 0 5"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  // both actuators own [pos, vel] blocks
  ASSERT_EQ(model->nu, 4);
  MjDataPtr data = MakeData(model);

  // time-varying position and velocity commands, identical for both
  while (data->time < 1.0) {
    mjtNum qref = mju_sin(5 * data->time);
    mjtNum vref = mju_cos(3 * data->time);
    data->ctrl[0] = data->ctrl[2] = qref;
    data->ctrl[1] = data->ctrl[3] = vref;
    mj_step(model.get(), data.get());

    EXPECT_NEAR(data->qpos[0], data->qpos[1], MjTol(1e-14, 1e-6));
    EXPECT_NEAR(data->qvel[0], data->qvel[1], MjTol(1e-14, 1e-6));

    // recompute forces at the post-step state before comparing: identical, the
    // tau->V map cancels back-EMF so there is no extra damping to account for
    mj_forward(model.get(), data.get());
    EXPECT_NEAR(data->actuator_force[0], data->actuator_force[1],
                MjTol(1e-13, 1e-5));
  }
}
TEST_F(DCMotorTest, StatelessSteadyState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // the dcmotor ff input is the terminal voltage, named accordingly
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 0, 0), "voltage");

  double K = 0.05;
  double R = 2.0;
  double V = 12.0;
  double omega = 3.0;

  data->ctrl[0] = V;
  data->qvel[0] = omega;
  mj_forward(model.get(), data.get());

  double expected_force = K / R * (V - K * omega);
  EXPECT_NEAR(data->actuator_force[0], expected_force, MjTol(1e-12, 1e-5));
  EXPECT_EQ(model->actuator_actnum[0], 0);
}

TEST_F(DCMotorTest, CurrentFilterConverges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.0001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="1000"/>
        <geom size="1" mass="100"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               inductance="0.01 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  ASSERT_EQ(model->actuator_actnum[0], 1);

  double K = 0.05;
  double R = 2.0;
  double V = 12.0;

  data->ctrl[0] = V;
  for (int i = 0; i < 10000; i++) {
    mj_step(model.get(), data.get());
  }

  double omega = data->qvel[0];
  double i_ss = V / R - K / R * omega;
  double expected_force = K * i_ss;

  EXPECT_NEAR(data->act[0], i_ss, MjTol(1e-6, 1e-4));
  EXPECT_NEAR(data->actuator_force[0], expected_force, MjTol(1e-6, 1e-4));
}

TEST_F(DCMotorTest, CurrentFilterExactIntegration) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="10000"/>
        <geom size="1" mass="10000"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               inductance="0.01 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  double R = 2.0;
  double te = 0.01 / R;
  double V = 12.0;

  data->ctrl[0] = V;
  mj_step(model.get(), data.get());

  double h = model->opt.timestep;
  double exact_current = V / R * (1 - mju_exp(-h / te));
  EXPECT_NEAR(data->act[0], exact_current, MjTol(1e-10, 1e-4));

  double euler_current = V / R * h / te;
  EXPECT_GT(std::abs(data->act[0] - euler_current),
            std::abs(data->act[0] - exact_current));
}

TEST_F(DCMotorTest, CoggingTorque) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               cogging="0.1 6 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  double A = 0.1, Np = 6, phi = 0;
  double K = 0.05, R = 2.0;
  double V = 5.0;
  double pos = 1.0;

  data->ctrl[0] = V;
  data->qpos[0] = pos;
  mj_forward(model.get(), data.get());

  double electrical_force = K / R * V;
  double cogging = A * mju_sin(Np * pos + phi);
  EXPECT_NEAR(data->actuator_force[0], electrical_force + cogging,
              MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, CoggingBypassesSaturation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               saturation="0.001 0" cogging="0.1 6 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  double A = 0.1, Np = 6, phi = 0;
  double pos = 1.0;

  data->ctrl[0] = 100.0;
  data->qpos[0] = pos;
  mj_forward(model.get(), data.get());

  double cogging = A * mju_sin(Np * pos + phi);
  EXPECT_NEAR(model->actuator_forcerange[1], 0.001, MjTol(1e-12, 1e-5));
  EXPECT_GT(mju_abs(data->actuator_force[0]), 0.001);
  EXPECT_NEAR(data->actuator_force[0], 0.001 + cogging, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, LuGreViscousFriction) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  ASSERT_EQ(model->actuator_actnum[0], 1);

  double sigma1 = 1, sigma2 = 0.01;
  double K = 0.05, R = 2.0;
  double omega = 2.0;

  data->ctrl[0] = 0;
  data->qvel[0] = omega;
  mj_forward(model.get(), data.get());

  EXPECT_MJTNUM_EQ(model->actuator_damping[0], sigma2);
  double electrical_force = K / R * (0 - K * omega);
  double z = data->act[model->actuator_actadr[0]];
  double z_dot = data->act_dot[model->actuator_actadr[0]];
  double lugre_force = 100 * z + sigma1 * z_dot;
  EXPECT_NEAR(data->actuator_force[0], electrical_force - lugre_force,
              MjTol(1e-12, 1e-5));
}

// the LuGre bristle must integrate the velocity of its own transmission:
// placing a multi-output (SO3) actuator before the DC motor, so that the
// motor's actuator id and output address diverge, must not change the
// bristle dynamics
TEST_F(DCMotorTest, LuGreBristleVelocityOrderInvariance) {
  static constexpr char xml_dc_first[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
      <body pos="3 0 0">
        <joint name="ball" type="ball"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor name="dc" joint="hinge" motorconst="0.05" resistance="2.0"
               lugre="100 0.5 0.5 0.8 0.5"/>
      <orientation name="so3" joint="ball" kp="1"/>
    </actuator>
  </mujoco>
  )";
  static constexpr char xml_so3_first[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
      <body pos="3 0 0">
        <joint name="ball" type="ball"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <orientation name="so3" joint="ball" kp="1"/>
      <dcmotor name="dc" joint="hinge" motorconst="0.05" resistance="2.0"
               lugre="100 0.5 0.5 0.8 0.5"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];

  // reference: DC motor first, actuator id == output address
  MjModelPtr model = LoadModelFromString(xml_dc_first, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int dc = mj_name2id(model.get(), mjOBJ_ACTUATOR, "dc");
  data->qvel[model->jnt_dofadr[mj_name2id(model.get(), mjOBJ_JOINT, "hinge")]] =
      1;
  mj_step(model.get(), data.get());
  double z_dc_first = data->act[model->actuator_actadr[dc]];

  // reordered: the SO3 actuator has 3 outputs, so the DC motor now has
  // actuator id 1 but output address 3
  model = LoadModelFromString(xml_so3_first, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  data = MakeData(model);
  dc = mj_name2id(model.get(), mjOBJ_ACTUATOR, "dc");
  ASSERT_EQ(model->actuator_outadr[dc], 3);
  data->qvel[model->jnt_dofadr[mj_name2id(model.get(), mjOBJ_JOINT, "hinge")]] =
      1;
  mj_step(model.get(), data.get());
  double z_so3_first = data->act[model->actuator_actadr[dc]];

  // the bristle state saw the same spinning hinge in both models
  EXPECT_NE(z_dc_first, 0);
  EXPECT_MJTNUM_EQ(z_so3_first, z_dc_first);
}

// A dcmotor with input="none" has no controls and acts as a passive device:
// LuGre friction, cogging and back-EMF braking with the terminals shorted.
TEST_F(DCMotorTest, PassiveNoInputs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0" input="none"
               damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // no controls at all, one actuator, one bristle state
  EXPECT_EQ(model->nu, 0);
  EXPECT_EQ(model->nactuator, 1);
  EXPECT_EQ(model->actuator_ctrlnum[0], 0);
  ASSERT_EQ(model->actuator_actnum[0], 1);

  // same force as a voltage-commanded motor with u = 0: shorted terminals
  double sigma1 = 1;
  double K = 0.05, R = 2.0;
  double omega = 2.0;
  data->qvel[0] = omega;
  mj_forward(model.get(), data.get());

  double electrical_force = K / R * (0 - K * omega);
  double z = data->act[model->actuator_actadr[0]];
  double z_dot = data->act_dot[model->actuator_actadr[0]];
  double lugre_force = 100 * z + sigma1 * z_dot;
  EXPECT_NEAR(data->actuator_force[0], electrical_force - lugre_force,
              MjTol(1e-12, 1e-5));

  // the model steps with an empty ctrl vector: friction brakes the joint
  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }
  EXPECT_GT(data->act[model->actuator_actadr[0]], 0);
  EXPECT_LT(data->qvel[0], omega);
}

// input="none" validation: dcmotor-only, incompatible with slew and ki.
TEST_F(DCMotorTest, NoInputsCompileErrors) {
  char error[1024];

  // pid does not accept input="none"
  static constexpr char pid_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid joint="joint" kp="1" input="none"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(pid_xml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("subset"));

  // slew rate limiting requires an input
  static constexpr char slew_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0" input="none"
               controller="0 0 0 4.0 0 0"/>
    </actuator>
  </mujoco>
  )";
  model = LoadModelFromString(slew_xml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("slew"));

  // the "voltage" keyword is dcmotor-only
  static constexpr char voltage_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid joint="joint" kp="1" input="voltage"/>
    </actuator>
  </mujoco>
  )";
  model = LoadModelFromString(voltage_xml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull());

  // integral gain requires the pos input
  static constexpr char ki_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0" input="vel ff"
               controller="0 2.0 0 0 0 0"/>
    </actuator>
  </mujoco>
  )";
  model = LoadModelFromString(ki_xml, error, sizeof(error));
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("pos input"));
}

TEST_F(DCMotorTest, ThermalRiseAndFall) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="10000"/>
        <geom size="1" mass="10000"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               thermal="10 5 0 0 25 25"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  int adr = model->actuator_actadr[0];
  ASSERT_EQ(model->actuator_actnum[0], 1);
  EXPECT_EQ(data->act[adr], 0);

  double R = 2.0, V = 10.0;
  double RT = 10.0, C = 5.0;
  double h = model->opt.timestep;
  double P = V * V / R;

  data->ctrl[0] = V;

  mj_step(model.get(), data.get());
  double dT1 = h * P / C;
  EXPECT_NEAR(data->act[adr], dT1, MjTol(1e-11, 1e-4));

  mj_step(model.get(), data.get());
  double dT2 = dT1 + h * (P - dT1 / RT) / C;
  EXPECT_NEAR(data->act[adr], dT2, MjTol(1e-11, 1e-4));

  data->ctrl[0] = 0;
  mj_step(model.get(), data.get());
  double dT3 = dT2 + h * (0 - dT2 / RT) / C;
  EXPECT_NEAR(data->act[adr], dT3, MjTol(1e-11, 1e-4));
  EXPECT_LT(data->act[adr], dT2);
}

TEST_F(DCMotorTest, ThermalSteadyState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="10000"/>
        <geom size="1" mass="10000"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               thermal="0.1 0.1 0 0 25 25"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  double R = 2.0, V = 10.0;
  double RT = 0.1;
  double dT_ss = RT * V * V / R;

  data->ctrl[0] = V;
  for (int i = 0; i < 10000; i++) {
    mj_step(model.get(), data.get());
  }

  int adr = model->actuator_actadr[0];
  EXPECT_NEAR(data->act[adr], dT_ss, 1e-4);
}

TEST_F(DCMotorTest, ThermalAffectsForce) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               thermal="0.1 0.1 0 0.004 25 25"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  double K = 0.05, R = 2.0, V = 10.0;
  double alpha = 0.004;
  int adr = model->actuator_actadr[0];

  data->ctrl[0] = V;
  data->act[adr] = 0;
  mj_forward(model.get(), data.get());
  double force_cold = data->actuator_force[0];
  EXPECT_NEAR(force_cold, K / R * V, MjTol(1e-12, 1e-5));

  double dT = 50;
  data->act[adr] = dT;
  mj_forward(model.get(), data.get());
  double R_hot = R * (1 + alpha * dT);
  double force_hot = data->actuator_force[0];
  EXPECT_NEAR(force_hot, K / R_hot * V, MjTol(1e-12, 1e-5));
  EXPECT_LT(force_hot, force_cold);
}

// Temperature slot must be correctly offset past slew and integral states.
TEST_F(DCMotorTest, ThermalAffectsForceWithController) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               input="pos vel" controller="1.0 1.0 0 5.0 0"
               thermal="0.1 0.1 0 0.004 25 25"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // slot order: slew(0), integral(1), temperature(2)
  ASSERT_EQ(model->actuator_actnum[0], 3);
  int adr = model->actuator_actadr[0];
  int temp_adr = adr + 2;  // temperature is slot 2

  double R = 2.0, alpha = 0.004;
  double dT = 50;
  data->act[adr] = 1.0;      // slew state = ctrl: no rate-limiting applied
  data->act[adr + 1] = 0.0;  // integral state x_I = 0
  data->act[temp_adr] = dT;  // temperature rise above ambient
  data->ctrl[0] = 1.0;       // position setpoint = 1.0, qpos = 0, error = 1.0
  mj_forward(model.get(), data.get());

  // u_eff = ctrl = 1.0 (no slew applied since act[slew] == ctrl)
  // torque command tau = kp*(u_eff - length) = 1.0; the tau->V map uses the
  // nameplate resistance, so the hot motor under-delivers by R/R(T):
  // V = R/K * tau = 40, force = K/R(T) * V = (R/R(T)) * tau
  double R_hot = R * (1 + alpha * dT);
  EXPECT_NEAR(data->actuator_force[0], R / R_hot * 1.0, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, StatelessPositionMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="2.0 0 0.5 0 0"
               motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // setpoint inputs are named
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 0, 0), "pos");
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 0, 1), "vel");

  // Position target 5.0, current pos 0.0, current vel 0.0
  data->ctrl[0] = 5.0;
  mj_forward(model.get(), data.get());

  // torque-space controller with back-EMF compensation: force = kp*(qref - l)
  // force = 2.0 * 5.0 = 10.0, exactly
  EXPECT_NEAR(data->actuator_force[0], 10.0, MjTol(1e-12, 1e-5));

  // Velocity penalty
  data->qvel[0] = 2.0;
  mj_forward(model.get(), data.get());
  // force = kp*(qref - l) - kd*omega = 10.0 - 0.5*2.0 = 9.0: no back-EMF droop
  EXPECT_NEAR(data->actuator_force[0], 9.0, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, StatelessVelocityMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="0 0 3.0 0 0"
               motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Velocity target 4.0 (second input), current vel 1.0
  data->ctrl[1] = 4.0;
  data->qvel[0] = 1.0;
  mj_forward(model.get(), data.get());

  // force = kd * (vref - omega) = 3.0 * (4.0 - 1.0) = 9.0, exactly
  EXPECT_NEAR(data->actuator_force[0], 9.0, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, StatefulPositionMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="2.0 0.5 0.1 10.0 5.0"
               motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Controller states: 1 for slew, 1 for ki -> actnum = 2
  ASSERT_EQ(model->actuator_actnum[0], 2);
  int adr = model->actuator_actadr[0];

  // Current states
  double u_prev = 1.0;
  double x_I = 2.0;
  data->act[adr] = u_prev;
  data->act[adr + 1] = x_I;

  // target 5.0 position, current 0.0
  data->ctrl[0] = 5.0;
  data->qvel[0] = 0.5;
  mj_forward(model.get(), data.get());

  // slew bounding: s = 10.0, dt = 0.001. max_change = 0.01
  // Target = 5.0. It is upper bounded by u_prev + 0.01 = 1.01
  EXPECT_NEAR(data->act_dot[adr], 10.0, MjTol(1e-12, 1e-5));

  // PI error: error = u_eff - length = 1.01 - 0.0 = 1.01
  EXPECT_NEAR(data->act_dot[adr + 1], 1.01, MjTol(1e-12, 1e-5));

  // force = kp * (u_eff - length) + ki * x_I - kd * omega
  //       = 2.0 * 1.01 + 0.5 * 2.0 - 0.1 * 0.5 = 2.97, exactly
  EXPECT_NEAR(data->actuator_force[0], 2.97, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, StatefulPositionWithCurrentMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="2.0 0.5 0.1 10.0 5.0"
               motorconst="0.05" resistance="2.0" inductance="1.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Controller states: slew (0), ki (1), current (2). actnum = 3
  ASSERT_EQ(model->actuator_actnum[0], 3);
  int adr = model->actuator_actadr[0];

  double u_prev = 1.0;
  double x_I = 2.0;
  double current = 0.5;
  data->act[adr] = u_prev;
  data->act[adr + 1] = x_I;
  data->act[adr + 2] = current;

  // Target 5.0 position, velocity 0.5
  data->ctrl[0] = 5.0;
  data->qvel[0] = 0.5;
  mj_forward(model.get(), data.get());

  // Slew bounding: max_change = 0.01, u_eff = 1.01
  EXPECT_NEAR(data->act_dot[adr], 10.0, MjTol(1e-12, 1e-5));

  // PI error: error = u_eff - length = 1.01
  EXPECT_NEAR(data->act_dot[adr + 1], 1.01, MjTol(1e-12, 1e-5));

  // torque command: tau = kp * (u_eff - length) + ki * x_I - kd * omega
  //                      = 2.0 * 1.01 + 0.5 * 2.0 - 0.1 * 0.5 = 2.97
  // tau->V map with back-EMF compensation:
  // V = R/K * tau + K * omega = 2.0/0.05 * 2.97 + 0.05 * 0.5 = 118.825

  // Current filter:
  // t_e = L / R = 1.0 / 2.0 = 0.5
  // di/dt = (V/R - K/R * omega - i) / t_e
  // di/dt = (118.825/2.0 - 0.05/2.0 * 0.5 - 0.5) / 0.5 = 117.8
  EXPECT_NEAR(data->act_dot[adr + 2], 117.8, MjTol(1e-12, 1e-4));

  // Force is K * next_activation (actearly is always on for DC motors)
  // Inline mj_nextActivation for te = 0.5
  mjtNum te = 0.5;
  mjtNum h = model->opt.timestep;
  mjtNum next_i = 0.5 + data->act_dot[adr + 2] * te * (1 - mju_exp(-h / te));
  EXPECT_NEAR(data->actuator_force[0], 0.05 * next_i, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, StatefulVelocityMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="3.0 1.0 0 0 2.0"
               motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Controller states: 1 for ki (no slew)
  ASSERT_EQ(model->actuator_actnum[0], 1);
  int adr = model->actuator_actadr[0];

  double x_I = 2.0;  // Exactly at Imax limit (Imax = 2.0)
  data->act[adr] = x_I;

  // position target 4.0, current pos 0, current vel 1.0 (kd = 0)
  data->ctrl[0] = 4.0;
  data->qvel[0] = 1.0;
  mj_forward(model.get(), data.get());

  // error = 4.0; since x_I == Imax (2.0) and error > 0, act_dot clamps to 0
  EXPECT_NEAR(data->act_dot[adr], 0.0, MjTol(1e-12, 1e-5));

  // force = kp * (qref - l) + ki * x_I = 3.0 * (4.0 - 0.0) + 1.0 * 2.0 = 14.0
  EXPECT_NEAR(data->actuator_force[0], 14.0, MjTol(1e-12, 1e-5));

  // repeat with non-zero joint position
  data->qpos[0] = 1.5;
  mj_forward(model.get(), data.get());

  // force = 3.0 * (4.0 - 1.5) + 1.0 * 2.0 = 9.5
  EXPECT_NEAR(data->actuator_force[0], 9.5, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, CurrentPlusThermal) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="10000"/>
        <geom size="1" mass="10000"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               inductance="0.01 0" thermal="10 5 0 0.004 25 25"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  ASSERT_EQ(model->actuator_actnum[0], 2);
  int adr = model->actuator_actadr[0];

  double K = 0.05, R = 2.0, V = 12.0;
  double te = 0.01 / R;
  double RT = 10.0, C = 5.0;

  double current = 3.0;
  double dT = 10.0;
  data->act[adr] = dT;
  data->act[adr + 1] = current;
  data->ctrl[0] = V;
  mj_forward(model.get(), data.get());

  // Force uses next_activation (actearly is always on for DC motors)
  // Inline mj_nextActivation for te = 0.01 / R = 0.005
  mjtNum h = model->opt.timestep;
  mjtNum next_i =
      current + data->act_dot[adr + 1] * te * (1 - mju_exp(-h / te));
  EXPECT_NEAR(data->actuator_force[0], K * next_i, MjTol(1e-12, 1e-5));

  double R_hot = R * (1 + 0.004 * dT);
  double T_dot = (R_hot * current * current - dT / RT) / C;
  EXPECT_NEAR(data->act_dot[adr], T_dot, MjTol(1e-10, 1e-4));

  double omega = data->qvel[0];
  double i_dot = (V / R_hot - K / R_hot * omega - current) / te;
  EXPECT_NEAR(data->act_dot[adr + 1], i_dot, MjTol(1e-10, 1e-3));
}

TEST_F(DCMotorTest, CurrentRateLimit) {
  // Verifies that saturation:current_rate clamps di/dt.
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint" damping="10000"/>
        <geom size="1" mass="10000"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               inductance="0.01 0" saturation="0 0 100"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  ASSERT_EQ(model->actuator_actnum[0], 1);
  int adr = model->actuator_actadr[0];

  double V = 12.0;
  double dimax = 100.0;  // A/s rate limit

  // unclamped: i_dot = (V/R - 0 - 0) / te = 6 / 0.005 = 1200 A/s >> dimax
  data->act[adr] = 0;  // current = 0
  data->ctrl[0] = V;
  mj_forward(model.get(), data.get());

  // i_dot should be clipped to +dimax
  EXPECT_NEAR(data->act_dot[adr], dimax, MjTol(1e-12, 1e-5));

  // reverse: large negative drive
  data->ctrl[0] = -V;
  mj_forward(model.get(), data.get());

  // i_dot should be clipped to -dimax
  EXPECT_NEAR(data->act_dot[adr], -dimax, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, VoltageLimit) {
  // verifies that saturation:voltage clamps voltage
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               input="pos vel" controller="1 0 0 0 0 10.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Vmax = 10.0, ctrl = 20.0
  // force = K/R * Vmax = 0.05 / 2.0 * 10.0 = 0.25
  data->ctrl[0] = 20.0;
  mj_forward(model.get(), data.get());

  EXPECT_NEAR(data->actuator_force[0], 0.25, MjTol(1e-12, 1e-5));

  // negative drive
  data->ctrl[0] = -20.0;
  mj_forward(model.get(), data.get());

  EXPECT_NEAR(data->actuator_force[0], -0.25, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, IntegralClamp) {
  // verifies that controller Imax clamps integral state
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" input="pos vel" controller="2.0 0.5 0 0 5.0"
               motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Imax = 5.0
  ASSERT_EQ(model->actuator_actnum[0], 1);  // only ki is stateful
  int adr = model->actuator_actadr[0];

  // set integral state to Imax
  data->act[adr] = 5.0;

  // set target to generate positive error (ctrl - length)
  data->ctrl[0] = 1.0;  // target
  data->qpos[0] = 0.0;  // length = 0

  mj_forward(model.get(), data.get());

  // act_dot should be clamped to 0 because act >= Imax and error > 0
  EXPECT_NEAR(data->act_dot[adr], 0.0, MjTol(1e-12, 1e-5));

  // set target to generate negative error
  data->ctrl[0] = -1.0;
  mj_forward(model.get(), data.get());

  // act_dot should be negative (not clamped)
  EXPECT_NEAR(data->act_dot[adr], -1.0, MjTol(1e-12, 1e-5));

  // set integral state to -Imax
  data->act[adr] = -5.0;

  // set target to generate negative error
  data->ctrl[0] = -1.0;
  data->qpos[0] = 0.0;
  mj_forward(model.get(), data.get());

  // act_dot should be clamped to 0 because act <= -Imax and error < 0
  EXPECT_NEAR(data->act_dot[adr], 0.0, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, LuGreExactIntegration) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1" mass="1e6"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  ASSERT_EQ(model->actuator_actnum[0], 1);
  int adr = model->actuator_actadr[0];

  double sigma0 = 100, F_C = 0.5, F_S = 0.7, v_S = 10;
  double z0 = 0.002;
  double v = 0.5;
  double h = model->opt.timestep;

  data->act[adr] = z0;
  data->qvel[0] = v;

  double ratio = v / v_S;
  double g_v = F_C + (F_S - F_C) * mju_exp(-ratio * ratio);
  double a = -sigma0 * std::abs(v) / g_v;
  double exp_ah = mju_exp(a * h);
  double int_h = (exp_ah - 1) / a;
  double z_new = exp_ah * z0 + int_h * v;

  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->act[adr], z_new, MjTol(1e-12, 1e-5));
}

TEST_F(DCMotorTest, LuGreSteadyState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1" mass="1e6"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  int adr = model->actuator_actadr[0];

  double sigma0 = 100, sigma2 = 0.01;
  double F_C = 0.5, F_S = 0.7, v_S = 10;
  double K = 0.05, R = 2.0;
  double v = 0.5;

  data->qvel[0] = v;
  data->ctrl[0] = 0;
  for (int i = 0; i < 10000; i++) {
    mj_step(model.get(), data.get());
  }

  double ratio = v / v_S;
  double g_v = F_C + (F_S - F_C) * mju_exp(-ratio * ratio);
  double z_ss = g_v / sigma0;
  EXPECT_NEAR(data->act[adr], z_ss, 1e-4);

  EXPECT_MJTNUM_EQ(model->actuator_damping[0], sigma2);
  double back_emf = K * K / R * data->qvel[0];
  double lugre_ss = g_v;
  EXPECT_NEAR(data->actuator_force[0], -back_emf - lugre_ss, 1e-3);
}

TEST_F(DCMotorTest, LuGreBristleSpring) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="joint" motorconst="0.05" resistance="2.0"
               damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  int adr = model->actuator_actadr[0];
  double sigma0 = 100;
  double X = 0.01;

  data->act[adr] = X;
  data->ctrl[0] = 0;
  mj_forward(model.get(), data.get());

  EXPECT_NEAR(data->actuator_force[0], -sigma0 * X, MjTol(1e-12, 1e-5));
}

// ----------------------- filterexact actuators -------------------------------

using FilterExactTest = MujocoTest;

TEST_F(FilterExactTest, ApproximatesContinuousTime) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true"/>
    <worldbody>
      <body name="box">
        <joint name="slide" type="slide" axis="1 0 0" />
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="slide" dyntype="filter" gainprm="1.1" />
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum kSimulationTime = 1.0;

  // compute act with a small timestep to approximate continuous integration
  model->opt.timestep = 0.001;
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 1.0;
  data->act[0] = 0.0;
  for (int i = 0; i < std::round(kSimulationTime / model->opt.timestep); i++) {
    mj_step(model.get(), data.get());
  }
  mjtNum continuous_act = data->act[0];

  // compute again with a larger timestep, introducing integration error
  model->opt.timestep = 0.01;
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 1.0;
  data->act[0] = 0.0;
  for (int i = 0; i < std::round(kSimulationTime / model->opt.timestep); i++) {
    mj_step(model.get(), data.get());
  }
  mjtNum discrete_act = data->act[0];

  // compute a third time with exact integration
  model->actuator_dyntype[0] = mjDYN_FILTEREXACT;
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 1.0;
  data->act[0] = 0.0;
  for (int i = 0; i < std::round(kSimulationTime / model->opt.timestep); i++) {
    mj_step(model.get(), data.get());
  }
  mjtNum exactfilter_act = data->act[0];

  // expect exact integration to be closer to the small-timestep result
  EXPECT_THAT(std::abs(continuous_act - discrete_act),
              Gt(5 * std::abs(continuous_act - exactfilter_act)))
      << "Using filterexact should make the error at least 5 times smaller";
}

TEST_F(FilterExactTest, TimestepIndependent) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true"/>
    <worldbody>
      <body name="box">
        <joint name="slide" type="slide" axis="1 0 0" />
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="slide" dyntype="filterexact" dynprm="0.9" gainprm="1.1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum kSimulationTime = 1.0;

  // first, compute act based on a small timestep and exact integration
  model->opt.timestep = 0.01;
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 1.0;
  data->act[0] = 0.0;
  for (int i = 0; i < std::round(kSimulationTime / model->opt.timestep); i++) {
    mj_step(model.get(), data.get());
  }
  mjtNum small_timestep_act = data->act[0];

  // now change the timestep to a much larger timestep
  model->opt.timestep = 0.1;
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 1.0;
  data->act[0] = 0.0;
  for (int i = 0; i < std::round(kSimulationTime / model->opt.timestep); i++) {
    mj_step(model.get(), data.get());
  }
  mjtNum large_timestep_act = data->act[0];

  EXPECT_NEAR(small_timestep_act, large_timestep_act, MjTol(1e-14, 1e-6))
      << "exact integration should be independent of timestep to machine "
         "precision.";
}

TEST_F(FilterExactTest, ActEqualsCtrlWhenTauIsZero) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true"/>
    <worldbody>
      <body name="box">
        <joint name="slide" type="slide" axis="1 0 0" />
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="slide" dyntype="filterexact" dynprm="0" gainprm="1.1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  data->ctrl[0] = 0.5;
  data->act[0] = 0.0;
  mj_step(model.get(), data.get());
  EXPECT_EQ(data->act[0], data->ctrl[0]);
}

// ----------------------- actearly actuator attribute -------------------------

using ActEarlyTest = MujocoTest;

TEST_F(ActEarlyTest, RemovesOneStepDelay) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/actuation/actearly.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  ASSERT_EQ(model->nu % 2, 0) << "number of actuators should be even";
  ASSERT_EQ(model->nu, model->na) << "all actuators should be stateful";
  ASSERT_EQ(model->nq, model->nu);
  EXPECT_GT(model->nu, 0);

  // actuators are ordered in pairs with actearly=true and actearly=false
  for (int i = 0; i < model->na / 2; i++) {
    EXPECT_TRUE(model->actuator_actearly[2 * i]);
    EXPECT_FALSE(model->actuator_actearly[2 * i + 1]);
  }

  mjData* data = mj_makeData(model);

  // set all controls to the same value and make one step
  mju_fill(data->ctrl, 0.5, model->nu);
  mj_step(model, data);

  for (int i = 0; i < model->na / 2; i++) {
    EXPECT_EQ(data->act[2 * i], data->act[2 * i + 1])
        << "act should be the same after first step for "
        << mj_id2name(model, mjOBJ_ACTUATOR, 2 * i);

    EXPECT_EQ(data->act_dot[2 * i], data->act_dot[2 * i + 1])
        << "act_dot should be the same after first step for "
        << mj_id2name(model, mjOBJ_ACTUATOR, 2 * i);
  }

  for (int i = 0; i < 100; i++) {
    std::vector<mjtNum> last_qfrc(data->qfrc_actuator,
                                  data->qfrc_actuator + model->nu);
    mj_step(model, data);
    for (int j = 0; j < model->nu / 2; j++) {
      // this is true for torque actuators
      EXPECT_NEAR(last_qfrc[2 * j], data->qfrc_actuator[2 * j + 1],
                  MjTol(1e-3, 1e-1))
          << "there should be a 1 step delay between qfrc for "
          << mj_id2name(model, mjOBJ_ACTUATOR, 2 * j);
    }
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(ActEarlyTest, DoesntChangeStateInMjForward) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/actuation/actearly.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  mjData* data = mj_makeData(model);

  // set all controls to the same value and make one step
  mju_fill(data->ctrl, 0.5, model->nu);
  mj_forward(model, data);

  for (int i = 0; i < model->na; i++) {
    EXPECT_EQ(data->act[i], 0) << "act should not change with mj_forward."
                               << mj_id2name(model, mjOBJ_ACTUATOR, i);
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(ActuatorTest, DisableActuator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <motor joint="slide" gear="2"  group="0"/>
      <position joint="slide" kp="1" group="1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  data->ctrl[0] = 1.0;
  data->ctrl[1] = 1.0;

  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->qfrc_actuator[0], 3.0);

  model->opt.disableactuator = 1 << 0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->qfrc_actuator[0], 1.0);

  model->opt.disableactuator = 1 << 1;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->qfrc_actuator[0], 2.0);
}

TEST_F(ActuatorTest, DisableActuatorOutOfRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <motor joint="slide" gear="-1" group="-1"/>
      <motor joint="slide" gear="5"  group="0"/>
      <motor joint="slide" gear="31" group="31"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  data->ctrl[0] = 1.0;
  data->ctrl[1] = 1.0;
  data->ctrl[2] = 1.0;

  // all actuators active
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->qfrc_actuator[0], 35.0);

  // set all bits of disableactuator, only group 1 is disabled
  model->opt.disableactuator = ~0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->qfrc_actuator[0], 30.0);
}

TEST_F(ActuatorTest, TendonActuatorForceRange) {
  const std::string xml_path = GetTestDataFilePath(kTendonForceClamp);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  EXPECT_EQ(model->tendon_actfrclimited[0], 0);
  EXPECT_EQ(model->tendon_actfrcrange[0], 0);
  EXPECT_EQ(model->tendon_actfrcrange[1], 0);

  EXPECT_EQ(model->tendon_actfrclimited[1], 1);
  EXPECT_EQ(model->tendon_actfrcrange[2], -1);
  EXPECT_EQ(model->tendon_actfrcrange[3], 1);

  EXPECT_EQ(model->tendon_actfrclimited[2], 1);
  EXPECT_EQ(model->tendon_actfrcrange[4], -10);
  EXPECT_EQ(model->tendon_actfrcrange[5], 10);

  EXPECT_EQ(model->tendon_actfrclimited[3], 1);
  EXPECT_EQ(model->tendon_actfrcrange[6], 0);
  EXPECT_EQ(model->tendon_actfrcrange[7], 1);

  data->ctrl[0] = 1;
  data->ctrl[1] = 1;
  data->ctrl[2] = 1;

  data->ctrl[3] = -1;
  data->ctrl[4] = 1;

  data->ctrl[5] = -20;
  data->ctrl[6] = 5;
  data->ctrl[7] = -5;

  mj_forward(model, data);

  EXPECT_NEAR(data->actuator_force[0], 1, 1e-6);
  EXPECT_NEAR(data->actuator_force[1], 1, 1e-6);
  EXPECT_NEAR(data->actuator_force[2], 1, 1e-6);
  EXPECT_NEAR(data->actuator_force[3], -1, 1e-6);
  EXPECT_NEAR(data->actuator_force[4], 1, 1e-6);
  EXPECT_NEAR(data->actuator_force[5], -10, 1e-6);
  EXPECT_NEAR(data->actuator_force[6], 5, 1e-6);
  EXPECT_NEAR(data->actuator_force[7], -5, 1e-6);

  EXPECT_EQ(data->sensordata[0], 3);
  EXPECT_EQ(data->sensordata[1], 0);
  EXPECT_EQ(data->sensordata[2], -10);
  EXPECT_EQ(data->sensordata[3], 0);

  mj_deleteData(data);
  mj_deleteModel(model);
}

// ----------------------------- actuator delays -------------------------------

TEST_F(ForwardTest, ActuatorDelay) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.02" nsample="2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // delay = 0.02 seconds, timestep = 0.01, so ndelay = ceil(0.02/0.01) = 2
  EXPECT_EQ(model->actuator_history[0], 2);

  // set ctrl to a nonzero value
  data->ctrl[0] = 10.0;

  // step once: the new ctrl is appended but won't be read for 2 timesteps
  mj_step(model.get(), data.get());
  // actuator_force should still be 0 (delayed value from buffer init)
  EXPECT_NEAR(data->actuator_force[0], 0.0, 1e-10);

  // step again
  mj_step(model.get(), data.get());
  // still reading old values
  EXPECT_NEAR(data->actuator_force[0], 0.0, 1e-10);

  // step a third time - now the delayed ctrl should arrive
  mj_step(model.get(), data.get());
  // actuator_force should now be 10.0
  EXPECT_NEAR(data->actuator_force[0], 10.0, 1e-10);
}

// Test actuator delay with linear interpolation (interp=1)
// Uses delay = 1.5*timestep so interpolation is meaningful
TEST_F(ForwardTest, ActuatorDelayLinearInterp) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.015" nsample="3" interp="linear"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // delay = 0.015 seconds = 1.5*timestep, nsample=3, interp=1 (linear)
  EXPECT_EQ(model->actuator_history[0], 3);
  EXPECT_EQ(model->actuator_history[1], 1);  // interp=1 (linear)
  EXPECT_NEAR(model->actuator_delay[0], 0.015, MjTol(1e-10, 5e-6));

  // Set increasing ctrl values
  // Buffer has samples at times: -0.02, -0.01, 0 with values 0, 0, 0
  // After step 0 at time=0.01: buffer has times -0.01, 0, 0.01 with values 0,
  // 0, ctrl[0] Read at time 0.01 - 0.015 = -0.005: interpolate between t=-0.01
  // and t=0 Since both values are 0, expected actuator_force = 0

  data->ctrl[0] = 10.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 0.0, MjTol(1e-10, 5e-6)) << "step 0";

  // After step 1 at time=0.02: buffer has times 0, 0.01, 0.02 with values 0,
  // 10, 20 Read at time 0.02 - 0.015 = 0.005: interpolate between t=0 (val=0)
  // and t=0.01 (val=10) Expected: 0 * 0.5 + 10 * 0.5 = 5

  data->ctrl[0] = 20.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 5.0, MjTol(1e-10, 5e-6)) << "step 1";

  // After step 2 at time=0.03: buffer has times 0.01, 0.02, 0.03 with values
  // 10, 20, 30 Read at 0.03 - 0.015 = 0.015: interpolate between t=0.01
  // (val=10) and t=0.02 (val=20) Expected: 10 * 0.5 + 20 * 0.5 = 15

  data->ctrl[0] = 30.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 15.0, MjTol(1e-10, 5e-6)) << "step 2";
}

// Test actuator delay with multi-input PID actuator (input="pos vel ff")
TEST_F(ForwardTest, ActuatorDelayMultiInputPID) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid joint="slide" kp="10" kv="2" input="pos vel ff"
           delay="0.02" nsample="2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  EXPECT_EQ(model->actuator_ctrlnum[0], 3);
  EXPECT_EQ(model->actuator_history[0], 2);
  EXPECT_EQ(model->nhistory, 2 + 2 + 2 * 3);

  // Set pos=1.0, vel=2.0, ff=5.0
  // Undelayed force at qpos=0, qvel=0 would be kp*1 + kv*2 + ff = 10 + 4 + 5 =
  // 19
  data->ctrl[0] = 1.0;
  data->ctrl[1] = 2.0;
  data->ctrl[2] = 5.0;

  // Step 1 & 2: all three inputs (pos, vel, ff) must be delayed (force = 0)
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 0.0, 1e-10);

  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 0.0, 1e-10);

  // Step 3: delayed inputs [1.0, 2.0, 5.0] arrive while body is still at rest
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 19.0, 1e-10);
}

// Test actuator delay with SO(3) orientation actuators (expmap and quat inputs)
TEST_F(ForwardTest, ActuatorDelaySO3) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="b1" type="ball"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="b2" type="ball"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <orientation joint="b1" kp="10" input="expmap" delay="0.02" nsample="2"/>
      <orientation joint="b2" kp="10" input="quat" delay="0.02" nsample="2"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  EXPECT_EQ(model->actuator_ctrlnum[0], 3);
  EXPECT_EQ(model->actuator_ctrlnum[1], 4);
  // nhistory = (2 + 2 + 2*3) + (2 + 2 + 2*4) = 10 + 12 = 22
  EXPECT_EQ(model->nhistory, 22);
  EXPECT_EQ(model->actuator_historyadr[0], 0);
  EXPECT_EQ(model->actuator_historyadr[1], 10);

  // Command a 0.2 rad rotation around Y for b1 (expmap) and Z for b2 (quat)
  data->ctrl[0] = 0.0;
  data->ctrl[1] = 0.2;
  data->ctrl[2] = 0.0;

  mjtNum half_angle = 0.15;
  data->ctrl[3] = mju_cos(half_angle);
  data->ctrl[4] = 0.0;
  data->ctrl[5] = 0.0;
  data->ctrl[6] = mju_sin(half_angle);

  // Steps 1 & 2: both SO3 actuators read neutral history, producing zero torque
  mj_step(model.get(), data.get());
  for (int k = 0; k < 6; k++) {
    EXPECT_NEAR(data->actuator_force[k], 0.0, 1e-10);
  }

  mj_step(model.get(), data.get());
  for (int k = 0; k < 6; k++) {
    EXPECT_NEAR(data->actuator_force[k], 0.0, 1e-10);
  }

  // Step 3: delayed commands arrive; b1 gets torque along Y (kp * 0.2 = 2.0),
  // b2 gets torque along Z (kp * 0.3 = 3.0)
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->actuator_force[0], 0.0, 1e-10);
  EXPECT_NEAR(data->actuator_force[1], 2.0, MjTol(1e-10, 1e-5));
  EXPECT_NEAR(data->actuator_force[2], 0.0, 1e-10);
  EXPECT_NEAR(data->actuator_force[3], 0.0, 1e-10);
  EXPECT_NEAR(data->actuator_force[4], 0.0, 1e-10);
  EXPECT_NEAR(data->actuator_force[5], 3.0, MjTol(1e-10, 1e-5));
}

TEST_F(ForwardTest, FlexTrilinearInstability) {
  // model parameters matches user's trilinear.xml
  constexpr char xml[] = R"(
  <mujoco model="stability_test">
      <option gravity="0 0 -9.81" iterations="100" solver="CG" tolerance="1e-10"
              timestep="0.002" integrator="discrete">
          <flag warmstart="disable" island="disable"/>
      </option>
      <worldbody>
          <geom name="floor" size="0 0 .05" type="plane" condim="3"/>
          <flexcomp name="bed" type="grid" count="17 17 3" spacing="0.05 0.05 0.05"
                    pos="0 0 0.05" radius="0.0005" dim="3" mass="10" dof="trilinear">
              <contact condim="3" solref="0.005 1" solimp=".99 .99 .001" selfcollide="none"/>
              <elasticity young="865067.00" poisson="0.1" damping="1"/>
          </flexcomp>
          <body name="box" pos="0.05 0.05 0.5">
              <freejoint/>
              <geom name="box_geom" type="box" size="0.04 0.04 0.04" mass="0.5"
                    solref="0.001 1" solimp="0.99 0.99 0.01"/>
          </body>
      </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data = MakeData(model);

  // flex stiffness sign checks
  // verify correct sign of flex stiffness derivatives before simulation
  int nv = model->nv;
  mjtNum h = model->opt.timestep;

  // create a test vector
  std::vector<mjtNum> v(nv), Mv(nv), flex_Kv(nv);
  for (int i = 0; i < nv; i++) v[i] = mju_Halton(i, 2) - 0.5;
  mjtNum vnorm = mju_norm(v.data(), nv);
  for (int i = 0; i < nv; i++) v[i] /= vnorm;

  mj_forward(model.get(), data.get());

  // compute M*v and stiffness contributions
  mj_mulM(model.get(), data.get(), Mv.data(), v.data());

  // note: we use mjd_flexInterp_mulK here (unscaled by h^2) to check raw
  // stiffness logic similar to what we expect in the solver now
  mjtNum* v_copy = (mjtNum*)mju_malloc(nv * sizeof(mjtNum));
  mju_copy(v_copy, v.data(), nv);
  mju_zero(flex_Kv.data(), nv);

  // using mulKD for legacy check consistency, but we know it applies h^2+h*d
  // scaling; actually, let's stick to the high-level property checks from
  // FlexStiffnessSign which used mulKD
  mjd_flexInterp_mul(model.get(), data.get(), flex_Kv.data(), v.data(), h * h,
                     h, NULL);

  // compute v^T*M*v and v^T*scale*K*v
  mjtNum vMv = mju_dot(v.data(), Mv.data(), nv);
  // mulKD returns -scale*K*v, so -flex_Kv = +scale*K*v
  mjtNum vKv = -mju_dot(v.data(), flex_Kv.data(), nv);

  // assertions from FlexStiffnessSign
  EXPECT_GT(vKv, 0) << "Stiffness contribution should be positive";
  EXPECT_GT(vMv + vKv, vMv) << "Full Hessian should exceed M alone";

  mju_free(v_copy);

  // stability simulation
  // run for steps to catch instability
  for (int i = 0; i < 500; ++i) {
    mj_step(model.get(), data.get());

    for (int j = 0; j < model->nq; ++j) {
      if (mju_abs(data->qpos[j]) > 1000.0) {
        ADD_FAILURE() << "Instability detected at step " << i << " dof " << j
                      << " val " << data->qpos[j];
        return;  // Exit early
      }
    }
  }
}

// Verify that flex damping does not affect rigid body motion
TEST_F(ForwardTest, FlexDampingRigidMotion) {
  constexpr char xml[] = R"(
  <mujoco>
      <option gravity="0 0 0" timestep="0.01" integrator="discrete" solver="CG"/>
      <worldbody>
          <flexcomp name="flex" type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                    pos="0 0 0" euler="45 45 45" radius="0.01" dim="3" mass="1" dof="trilinear">
              <contact selfcollide="none"/>
              <elasticity young="1e5" poisson="0.3" damping="10"/>
          </flexcomp>
      </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Set initial rigid rotation velocity about Z axis
  // Center of mass is roughly at 0 0 0 because pos="0 0 0" and symmetric grid.
  // v = w x r. Let w = (1, 1, 1).
  mjtNum w[3] = {10.0, 10.0, 10.0};
  for (int i = 0; i < model->nv / 3; ++i) {
    int qpos_adr = model->jnt_qposadr[i];
    int qvel_adr = model->jnt_dofadr[i];
    mjtNum* pos = data->qpos + qpos_adr;
    mjtNum* vel = data->qvel + qvel_adr;

    mjtNum r[3] = {pos[0], pos[1], pos[2]};
    mju_cross(vel, w, r);
  }

  mj_forward(model.get(), data.get());
  mjtNum initial_energy = data->energy[0] + data->energy[1];

  // Run a few steps
  for (int i = 0; i < 10; ++i) {
    mj_step(model.get(), data.get());
  }

  mj_forward(model.get(), data.get());
  mjtNum final_energy = data->energy[0] + data->energy[1];

  // Expect energy conservation.
  // With the bug, damping force acts on rigid rotation, dissipating energy.
  EXPECT_NEAR(final_energy, initial_energy, 1e-6 * initial_energy)
      << "Energy decayed significantly (" << initial_energy << " -> "
      << final_energy << ")";
}

// verify that implicit integrator respects parent-flex coupling
TEST_F(ForwardTest, FlexParentCoupling) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.01" solver="CG"/>
    <worldbody>
      <body name="parent" pos="0 0 0">
        <freejoint/>
        <geom size=".1" mass="10"/>
        <flexcomp name="flex" type="grid" count="3 3 3" cellcount="1 1 1" spacing="1 1 1"
                  radius=".01" dim="3" mass="100" dof="trilinear" pos="1 1 1">
          <contact selfcollide="none"/>
          <elasticity young="1e4" poisson="0.3" damping="50"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // set state: parent moving, flex deformed
  // this ensures both H_fp (coupling) and qacc_parent are non-trivial
  // Run with Euler (timestep 1e-6)
  model->opt.timestep = 1e-6;
  model->opt.integrator = mjINT_EULER;
  mj_resetData(model.get(), data.get());
  data->qvel[0] = 1.0;
  data->qpos[7] += 0.01;
  data->qfrc_applied[0] = 10000.0;   // Apply large force to parent
  mj_step(model.get(), data.get());  // Step integrates
  std::vector<mjtNum> qvel_euler(model->nv);
  mju_copy(qvel_euler.data(), data->qvel, model->nv);

  // Run with Discrete (timestep 1e-6), the metric carrier for flex elasticity
  model->opt.integrator = mjINT_DISCRETE;
  mj_resetData(model.get(), data.get());
  data->qvel[0] = 1.0;
  data->qpos[7] += 0.01;
  data->qfrc_applied[0] = 10000.0;
  mj_step(model.get(), data.get());  // Step integrates
  std::vector<mjtNum> qvel_implicit(model->nv);
  mju_copy(qvel_implicit.data(), data->qvel, model->nv);

  // Check agreement
  double max_diff = 0;
  for (int i = 0; i < model->nv; ++i) {
    double diff = mju_abs(qvel_euler[i] - qvel_implicit[i]);
    if (diff > max_diff) max_diff = diff;
  }

  // implicit and explicit flex damping legitimately differ at
  // O(h*damping*K/M) in this comparison
  EXPECT_LT(max_diff, MjTol(1e-4, 1.5e-2))
      << "Implicit integrator should match Euler at small timestep";
}

TEST_F(ForwardTest, TrilinearPinnedParentWithFreejoint) {
  static constexpr char xml[] = R"(
  <mujoco>
  <option integrator="discrete" solver="CG"/>
  <worldbody>
    <body>
      <joint type="free"/>
      <geom type="box" size="0.13 0.18 0.036" pos="0 0 0.036"/>
      <body name="parent">
        <flexcomp name="test" type="grid"
                  count="3 3 3" spacing=".1 .02 .1" radius="0.001"
                  pos="0 0 0.1" dof="trilinear" xyaxes="0 1 0 0 0 1" mass="10" dim="3">
          <contact selfcollide="none"/>
          <elasticity young="1e5" poisson="0.3" damping="0.1"/>
          <pin id="0 2 4 6"/>
        </flexcomp>
      </body>
    </body>
  </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  MjDataPtr d = MakeData(m);

  int parent_id = mj_name2id(m.get(), mjOBJ_BODY, "parent");
  ASSERT_GT(parent_id, 0);

  EXPECT_EQ(m->nflexnode, 8);
  EXPECT_EQ(m->body_dofnum[parent_id], 0) << "parent body should have 0 DOFs";

  int freejoint_body = m->body_parentid[parent_id];
  EXPECT_EQ(m->body_dofnum[freejoint_body], 6) << "freejoint body has 6 DOFs";

  mj_resetData(m.get(), d.get());
  mj_forward(m.get(), d.get());

  for (int i = 0; i < 500; i++) {
    mj_step(m.get(), d.get());

    ASSERT_FALSE(mju_isBad(d->qpos[0]))
        << "Simulation became unstable at step " << i;
    ASSERT_FALSE(mju_isBad(d->qvel[0]))
        << "Velocity became unstable at step " << i;

    for (int j = 0; j < m->nq; j++) {
      ASSERT_LT(mju_abs(d->qpos[j]), 100.0)
          << "Position exploded at step " << i << ", qpos[" << j
          << "]=" << d->qpos[j];
    }
    for (int j = 0; j < m->nv; j++) {
      ASSERT_LT(mju_abs(d->qvel[j]), 1000.0)
          << "Velocity exploded at step " << i << ", qvel[" << j
          << "]=" << d->qvel[j];
    }
  }
}

// -------------------- actuator damping and armature --------------------------

using ActuatorDampingTest = MujocoTest;

TEST_F(ActuatorDampingTest, SingleActuatorJointDamping) {
  // actuator damping=3 with gear=2 should produce same force as
  // joint damping=12 (3*2^2=12)
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="2" damping="3"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_joint[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"
               damping="12"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_joint, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_forward(m1.get(), d1.get());

  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  mj_forward(m2.get(), d2.get());

  EXPECT_EQ(d1->qfrc_passive[0], d2->qfrc_passive[0]);
}

TEST_F(ActuatorDampingTest, SingleActuatorTendonDamping) {
  // actuator damping through tendon transmission
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="ten">
        <joint joint="jnt" coef="1"/>
      </fixed>
    </tendon>
    <actuator>
      <motor tendon="ten" gear="2" damping="3"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_tendon[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="ten" damping="12">
        <joint joint="jnt" coef="1"/>
      </fixed>
    </tendon>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_tendon, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_forward(m1.get(), d1.get());

  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  mj_forward(m2.get(), d2.get());

  EXPECT_EQ(d1->qfrc_passive[0], d2->qfrc_passive[0]);
}

TEST_F(ActuatorDampingTest, SingleActuatorArmature) {
  // actuator armature=0.5 with gear=3 should equal
  // joint armature=4.5 (0.5*3^2=4.5)
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="3" armature="0.5"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_joint[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"
               armature="4.5"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_joint, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_forward(m1.get(), d1.get());

  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  mj_forward(m2.get(), d2.get());

  EXPECT_EQ(d1->qacc[0], d2->qacc[0]);
}

TEST_F(ActuatorDampingTest, MultipleActuatorsAccumulate) {
  // two actuators: damping=2 gear=3, damping=1 gear=4
  // equivalent joint damping: 2*9 + 1*16 = 34
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="3" damping="2"/>
      <motor joint="jnt" gear="4" damping="1"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_joint[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"
               damping="34"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_joint, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_forward(m1.get(), d1.get());

  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  mj_forward(m2.get(), d2.get());

  EXPECT_EQ(d1->qfrc_passive[0], d2->qfrc_passive[0]);
}

TEST_F(ActuatorDampingTest, DampingSimulationEquivalence) {
  // actuator damping=5 gear=2 should match joint damping=20 over time
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="2" damping="5"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_joint[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="0 0 1"
               damping="20"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_joint, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  for (int i = 0; i < 100; i++) {
    mj_step(m1.get(), d1.get());
    mj_step(m2.get(), d2.get());
  }

  EXPECT_MJTNUM_EQ(d1->qpos[0], d2->qpos[0]);
  EXPECT_MJTNUM_EQ(d1->qvel[0], d2->qvel[0]);
}

TEST_F(ActuatorDampingTest, ArmatureSimulationEquivalence) {
  // actuator armature=2 gear=3 should match joint armature=18 over time
  static constexpr char xml_actuator[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="0 0 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="3" armature="2"/>
    </actuator>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_joint[] = R"(
  <mujoco>
    <option gravity="0 0 -10"/>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="0 0 1"
               armature="18"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key qvel="1"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml_actuator, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  MjModelPtr m2 = LoadModelFromString(xml_joint, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;
  MjDataPtr d2 = MakeData(m2);

  mj_resetDataKeyframe(m1.get(), d1.get(), 0);
  mj_resetDataKeyframe(m2.get(), d2.get(), 0);
  for (int i = 0; i < 100; i++) {
    mj_step(m1.get(), d1.get());
    mj_step(m2.get(), d2.get());
  }

  EXPECT_MJTNUM_EQ(d1->qpos[0], d2->qpos[0]);
  EXPECT_MJTNUM_EQ(d1->qvel[0], d2->qvel[0]);
}

TEST_F(ActuatorDampingTest, UtilityFunctionValues) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="5" damping="7" armature="3"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  mjtNum poly[mjNPOLY] = {0};
  EXPECT_EQ(mj_actuatorDamping(m.get(), mjOBJ_JOINT, 0, poly), 175);
  EXPECT_EQ(mj_actuatorArmature(m.get(), mjOBJ_JOINT, 0), 75);
}

TEST_F(ActuatorDampingTest, NonlinearDamping) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="jnt" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" gear="3" damping="2 0.5 0.1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  // linear damping: 2 * gear^2 = 18
  mjtNum poly0[mjNPOLY] = {0};
  EXPECT_EQ(mj_actuatorDamping(m.get(), mjOBJ_JOINT, 0, poly0), 18);

  // poly coefficients scaled by gear^2
  mjtNum poly[mjNPOLY] = {0};
  mj_actuatorDamping(m.get(), mjOBJ_JOINT, 0, poly);
  EXPECT_MJTNUM_EQ(poly[0], 0.5 * 9);  // 4.5
  EXPECT_MJTNUM_EQ(poly[1], 0.1 * 9);  // 0.9
}

TEST_F(ActuatorDampingTest, DampingVsKvGearScaling) {
  // Single model with two parallel bodies: one using kv, one using damping.
  // Both produce the same joint-space damping force:
  //   kv:      qfrc_actuator contribution = -kv * gear^2 * qvel
  //   damping: qfrc_passive  contribution = -damping * gear^2 * qvel
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="implicitfast"/>
    <worldbody>
      <body name="kv_body">
        <joint name="jnt_kv" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
      <body name="damp_body" pos="5 0 0">
        <joint name="jnt_damp" type="slide" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt_kv" kp="0" kv="5" gear="3"/>
      <position joint="jnt_damp" kp="0" damping="5" gear="3"/>
    </actuator>
    <keyframe>
      <key qvel="1 1"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // check forces at initial state
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  mj_forward(m.get(), d.get());

  // kv force arrives via qfrc_actuator, damping via qfrc_passive
  mjtNum frc_kv = d->qfrc_actuator[0];
  mjtNum frc_damp = d->qfrc_passive[1];
  EXPECT_NEAR(frc_kv, frc_damp, MjTol(1e-12, 1e-5));

  // expected force = -5 * 3^2 * 1 = -45
  EXPECT_NEAR(frc_damp, -45, MjTol(1e-12, 1e-5));

  // simulate and check trajectory equivalence
  mj_resetDataKeyframe(m.get(), d.get(), 0);
  for (int i = 0; i < 100; i++) {
    mj_step(m.get(), d.get());
  }

  EXPECT_NEAR(d->qpos[0], d->qpos[1], MjTol(1e-12, 1e-5))
      << "position trajectory mismatch";
  EXPECT_NEAR(d->qvel[0], d->qvel[1], MjTol(1e-12, 1e-5))
      << "velocity trajectory mismatch";
}

// Passive flex contact stiffness is far beyond the explicit limit (~50x)
// because its curvature is carried by the metric. Both curvature and shift are
// needed; without the shift it rings apart.
TEST_F(ImplicitIntegratorTest, PassiveFlexContactInMetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" iterations="400"/>
    <worldbody>
      <flexcomp name="lower" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
                radius=".004" mass=".3" pos="0 0 .2">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
        <pin id="0 8 72 80"/>
      </flexcomp>
      <flexcomp name="upper" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".004" mass=".1" pos="0 0 .27">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  MjDataPtr d = MakeData(m);
  const mjModel* model = m.get();
  mjData* data = d.get();

  // Physical peak speed is ~2 m/s; without the shift scene reaches 143 m/s.
  mjtNum vmax = 0;
  for (int i = 0; i < 1000; i++) {
    mj_step(model, data);
    for (int j = 0; j < model->nv; j++) {
      vmax = mju_max(vmax, mju_abs(data->qvel[j]));
    }
    ASSERT_FALSE(data->warning[mjWARN_BADQACC].number)
        << "diverged at step " << i;
  }
  EXPECT_LT(vmax, 4.0) << "peak speed " << vmax;

  // Upper sheet must not pass through the lower one: check that its lowest
  // vertex stays above the lower sheet's lowest point.
  mjtNum lo[2] = {1e30, 1e30};
  for (int k = 0; k < 2; k++) {
    int f = mj_name2id(model, mjOBJ_FLEX, k ? "upper" : "lower");
    for (int i = 0; i < model->flex_vertnum[f]; i++) {
      lo[k] = mju_min(
          lo[k], data->flexvert_xpos[3 * (model->flex_vertadr[f] + i) + 2]);
    }
  }
  EXPECT_GT(lo[1], lo[0] - 0.01)
      << "upper sheet passed through: lowest z " << lo[1]
      << " against the lower sheet's " << lo[0];
}

// A dense Jacobian gets no chain from mj_contactJacobian: it returns every dof
// in order and leaves the chain untouched, so the published rows are compacted
// on their nonzeros instead. Reading the chain regardless indexed uninitialized
// memory and crashed in the velocity shift.
TEST_F(ImplicitIntegratorTest, PassiveFlexContactDenseJacobian) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" iterations="400"
            jacobian="dense"/>
    <worldbody>
      <flexcomp name="lower" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
                radius=".004" mass=".3" pos="0 0 .2">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
        <pin id="0 8 72 80"/>
      </flexcomp>
      <flexcomp name="upper" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".004" mass=".1" pos="0 0 .27">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  MjDataPtr d = MakeData(m);
  const mjModel* model = m.get();
  mjData* data = d.get();
  ASSERT_FALSE(mj_isSparse(model)) << "the scene must exercise the dense path";

  for (int i = 0; i < 300; i++) {
    mj_step(model, data);
    ASSERT_FALSE(data->warning[mjWARN_BADQACC].number)
        << "diverged at step " << i;
    // every published column index addresses a real dof
    for (int adr = 0; adr < data->nefmcon;) {
      int nnz = data->efm_con_ind[adr];
      ASSERT_GE(nnz, 0);
      ASSERT_LE(adr + 2 + nnz, data->nefmcon);
      for (int j = 0; j < nnz; j++) {
        int c = data->efm_con_ind[adr + 2 + j];
        ASSERT_GE(c, 0);
        ASSERT_LT(c, model->nv);
      }
      adr += 2 + nnz;
    }
  }
}

// The contact rows carry the full Jacobian chain: with the lower sheet pinned
// to a hinged body, the upper sheet's contacts couple to the hinge inside the
// metric, not only in the shift. The drop must stay stable, stay bounded, and
// reach the base. The pinned sheet is stretch-only because pins on a jointed
// body are refused with bending, and soft because its elastic coupling to the
// hinge is still explicit. With contact restricted to flex dofs this scene
// diverged within 100 steps of contact onset.
TEST_F(ImplicitIntegratorTest, PassiveFlexContactMovingBase) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" iterations="400"/>
    <worldbody>
      <body name="base" pos="0 0 .2">
        <joint name="lift" type="slide" axis="0 0 1" stiffness="1000" damping="100"/>
        <joint name="tilt" type="hinge" axis="0 1 0" stiffness="1" damping=".05"/>
        <geom type="box" size=".05 .05 .005" pos="0 0 -.1" mass="2"/>
        <flexcomp name="lower" type="grid" dim="2" count="9 9 1" spacing=".04 .04 1"
                  radius=".008" mass=".3">
          <contact selfcollide="auto" passive="true"/>
          <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="stretch" damping="1e-4"/>
          <pin id="0 8 72 80"/>
        </flexcomp>
      </body>
      <flexcomp name="upper" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".008" mass=".1" pos=".05 0 .23">
        <contact selfcollide="auto" passive="true"/>
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both" damping="1e-4"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  MjDataPtr d = MakeData(m);
  const mjModel* model = m.get();
  mjData* data = d.get();

  mjtNum vmax = 0;
  for (int i = 0; i < 300; i++) {
    mj_step(model, data);
    for (int j = 0; j < model->nv; j++) {
      vmax = mju_max(vmax, mju_abs(data->qvel[j]));
    }
    ASSERT_FALSE(data->warning[mjWARN_BADQACC].number)
        << "diverged at step " << i;
  }
  EXPECT_LT(vmax, 4.0) << "peak speed " << vmax;

  // the load reaches the base through the pinned corners: the vertical spring
  // carries base + lower sheet + upper sheet (2 + 0.3 + 0.1 kg) wherever the
  // upper sheet has slid to. (The base tilt is not a load indicator: on the
  // fine-timestep trajectory the sheet slides to the centre of the lower sheet
  // and produces no torque.)
  int lift = mj_name2id(model, mjOBJ_JOINT, "lift");
  mjtNum supported = -data->qpos[model->jnt_qposadr[lift]] * 1000 / 9.81;
  EXPECT_GT(supported, 2.33) << "load never reached the base";

  // upper sheet must not pass through the lower one
  mjtNum lo[2] = {1e30, 1e30};
  for (int k = 0; k < 2; k++) {
    int f = mj_name2id(model, mjOBJ_FLEX, k ? "upper" : "lower");
    for (int i = 0; i < model->flex_vertnum[f]; i++) {
      lo[k] = mju_min(
          lo[k], data->flexvert_xpos[3 * (model->flex_vertadr[f] + i) + 2]);
    }
  }
  EXPECT_GT(lo[1], lo[0] - 0.01)
      << "upper sheet passed through: lowest z " << lo[1]
      << " against the lower sheet's " << lo[0];
}

// passive contact on unsupported flex shapes (rigid, 1D): a
// compile-time warning fires, the attribute is ignored, and the contact stays
// on the constraint solver
TEST_F(ImplicitIntegratorTest, PassiveFlexContactUnsupportedShape) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="implicitfast"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <flexcomp name="cable" type="grid" dim="1" count="8 1 1" spacing=".04 1 1"
                radius=".008" mass=".1" pos="0 0 .2">
        <contact passive="true"/>
        <edge equality="true"/>
        <pin id="0 7"/>
      </flexcomp>
      <body pos="0 0 .3">
        <freejoint/>
        <geom type="sphere" size=".03" mass=".02"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings("not supported for rigid or 1D flexes");
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  testing::Mock::VerifyAndClearExpectations(&warning_handler);
  MjDataPtr d = MakeData(m);
  mjModel* model = m.get();
  EXPECT_FALSE(mj_effFlexContactPossible(model, 0));

  // the sphere lands on the cable, resting on an ordinary solver contact
  for (int i = 0; i < 400; i++) {
    mj_step(model, d.get());
  }
  EXPECT_FALSE(d->warning[mjWARN_BADQACC].number);
  ASSERT_GT(d->ncon, 0);
  EXPECT_GT(d->qpos[model->nq - 5], 0.1) << "sphere fell through the cable";
}

// passive contact with no elastic stiffness still contributes to the metric:
// the flex predicate must see it, and island discovery unions the contacting
// body with the flex trees into one island while unrelated bodies form their
// own islands
TEST_F(ImplicitIntegratorTest, PassiveFlexContactOnlyInMetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG"/>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="net" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".004" mass=".1" pos="0 0 .2">
        <contact passive="true"/>
        <edge equality="true"/>
        <pin id="0 4 20 24"/>
      </flexcomp>
      <body pos="0.02 0.02 0.25">
        <freejoint/>
        <geom type="sphere" size=".02" mass=".05"/>
      </body>
      <body pos="2 0 .1">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass=".2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjModel* model = m.get();
  EXPECT_TRUE(mj_effFlexPossible(model, 0));

  MjDataPtr d1 = MakeData(m);
  MjDataPtr d2 = MakeData(m);
  for (int i = 0; i < 150; i++) {
    model->opt.disableflags &= ~mjDSBL_ISLAND;
    mj_step(model, d1.get());
    model->opt.disableflags |= mjDSBL_ISLAND;
    mj_step(model, d2.get());
  }
  model->opt.disableflags &= ~mjDSBL_ISLAND;

  // the sphere landed on the net, deforming it, and no divergence occurred
  ASSERT_GT(d1->ncon, 0);
  EXPECT_EQ(d1->nisland, 2);
  EXPECT_FALSE(d1->warning[mjWARN_BADQACC].number);
  EXPECT_GT(d1->qpos[model->nq - 12], 0.1) << "sphere fell through the net";
  mjtNum sag = 0;
  for (int k = 0; k < 63; k++) {  // 21 unpinned vertices x 3 dofs
    sag = mju_max(sag, mju_abs(d1->qpos[k]));
  }
  EXPECT_GT(sag, 0.002);

  // island and monolithic solves agree (within CG stopping tolerance across
  // separate per-island vs. combined monolithic CG trajectories)
  EXPECT_THAT(AsVector(d1->qpos, model->nq),
              Pointwise(MjNear(2e-6, 1e-5), AsVector(d2->qpos, model->nq)));
}

// Passive contact on interpolated flexes (trilinear volumetric, trilinear shell
// elastic2d="bend", and quadratic) against a plane/box geom and against another
// flex with pinned cage nodes: contacts are carried in d->nefmcon with
// d->nefc == 0 and step stably under integrator="discrete" without falling
// through.
TEST_F(ImplicitIntegratorTest, PassiveFlexContactInterpolated) {
  struct GeomCase {
    const char* name;
    const char* dof;
    const char* elasticity;
    const char* pin;
  };
  const GeomCase geom_cases[] = {
      {"trilinear_vol", "trilinear",
       R"(<elasticity young="5e4" poisson="0.3" damping="0.01"/>)", ""},
      {"trilinear_shell_bend", "trilinear",
       R"(<elasticity young="5e2" poisson="0.3" thickness="0.015" elastic2d="bend" damping="0"/>)",
       R"(<pin id="1 3 5 7"/>)"},
      {"trilinear_shell_stretch", "trilinear",
       R"(<elasticity young="5e4" poisson="0.3" thickness="0.015" elastic2d="stretch" damping="0.01"/>)",
       ""},
      {"quadratic_vol", "quadratic",
       R"(<elasticity young="5e4" poisson="0.3" damping="0.01"/>)", ""},
      {"trilinear_pinned_corner", "trilinear",
       R"(<elasticity young="5e4" poisson="0.3" damping="0.01"/>)",
       R"(<pin id="1"/>)"},
  };

  for (const GeomCase& tc : geom_cases) {
    SCOPED_TRACE(tc.name);
    std::string xml = std::string(R"(
    <mujoco>
      <option timestep="0.002" integrator="discrete" solver="CG" iterations="200"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <geom name="press" type="box" size="0.3 0.3 0.02" pos="0 0 0.25"/>
        <flexcomp name="block" type="grid" count="4 4 4" cellcount="1 1 1"
                  spacing="0.06 0.06 0.06" radius="0.008" dim="3" mass="0.5"
                  pos="0 0 0.10" dof=")") +
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
    ASSERT_THAT(m, NotNull()) << error;
    MjDataPtr d = MakeData(m);

    bool saw_passive_con = false;
    mjtNum vmax = 0;
    for (int step = 0; step < 300; step++) {
      mj_step(m.get(), d.get());
      EXPECT_EQ(d->nefc, 0);
      if (d->nefmcon > 0) {
        saw_passive_con = true;
      }
      for (int i = 0; i < m->nv; i++) {
        vmax = mju_max(vmax, mju_abs(d->qvel[i]));
      }
      ASSERT_EQ(d->warning[mjWARN_BADQACC].number, 0) << "step " << step;
      ASSERT_EQ(d->warning[mjWARN_INERTIA].number, 0) << "step " << step;
    }
    EXPECT_TRUE(saw_passive_con);
    EXPECT_LT(vmax, 5.0);

    // Verify no vertex fell through the z = 0 floor.
    mjtNum min_z = 1e30;
    for (int v = 0; v < m->flex_vertnum[0]; v++) {
      min_z = mju_min(min_z, d->flexvert_xpos[3 * v + 2]);
    }
    EXPECT_GT(min_z, -0.005);
  }

  // Flex-vs-flex passive contact between two trilinear flexes, with the lower
  // flex pinned at its bottom cage nodes.
  static constexpr char kFlexFlexXml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" iterations="200"/>
    <worldbody>
      <flexcomp name="lower" type="grid" count="4 4 3" cellcount="1 1 1"
                spacing="0.08 0.08 0.04" radius="0.008" dim="3" mass="0.4"
                pos="0 0 0.1" dof="trilinear">
        <contact selfcollide="none" passive="true"/>
        <elasticity young="1e5" poisson="0.3" damping="0.01"/>
        <pin id="0 2 4 6"/>
      </flexcomp>
      <flexcomp name="upper" type="grid" count="3 3 3" cellcount="1 1 1"
                spacing="0.06 0.06 0.04" radius="0.008" dim="3" mass="0.2"
                pos="0 0 0.21" dof="trilinear">
        <contact selfcollide="none" passive="true"/>
        <elasticity young="1e5" poisson="0.3" damping="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(kFlexFlexXml, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  MjDataPtr d = MakeData(m);

  bool saw_passive_con = false;
  for (int step = 0; step < 300; step++) {
    mj_step(m.get(), d.get());
    EXPECT_EQ(d->nefc, 0);
    if (d->nefmcon > 0) {
      saw_passive_con = true;
    }
    ASSERT_EQ(d->warning[mjWARN_BADQACC].number, 0) << "step " << step;
    ASSERT_EQ(d->warning[mjWARN_INERTIA].number, 0) << "step " << step;
  }
  EXPECT_TRUE(saw_passive_con);

  int f_lower = mj_name2id(m.get(), mjOBJ_FLEX, "lower");
  int f_upper = mj_name2id(m.get(), mjOBJ_FLEX, "upper");
  mjtNum min_z_lower = 1e30, min_z_upper = 1e30;
  for (int v = 0; v < m->flex_vertnum[f_lower]; v++) {
    min_z_lower = mju_min(
        min_z_lower, d->flexvert_xpos[3 * (m->flex_vertadr[f_lower] + v) + 2]);
  }
  for (int v = 0; v < m->flex_vertnum[f_upper]; v++) {
    min_z_upper = mju_min(
        min_z_upper, d->flexvert_xpos[3 * (m->flex_vertadr[f_upper] + v) + 2]);
  }
  EXPECT_GT(min_z_upper, min_z_lower);

  // Refining the surface mesh (count) at a fixed cage (cellcount) should keep
  // passive contact penetration similar rather than stiffening with vertex
  // count, for both volumetric and shell flexes (where interior cage nodes are
  // pinned).
  for (int shell : {0, 1}) {
    SCOPED_TRACE(shell ? "shell_refinement" : "volumetric_refinement");
    mjtNum max_pen_coarse = 0, max_pen_fine = 0;
    for (int fine : {0, 1}) {
      std::string ref_xml =
          std::string(R"(
      <mujoco>
        <option timestep="0.002" integrator="discrete" solver="CG" iterations="200"/>
        <worldbody>
          <geom name="floor" type="plane" size="2 2 0.1"/>
          <flexcomp name="block" type=")") +
          (shell ? "box" : "grid") + R"(" count=")" +
          (fine ? "7 7 7" : "3 3 3") + R"(" cellcount="2 2 2" spacing=")" +
          (fine ? "0.03 0.03 0.03" : "0.09 0.09 0.09") +
          R"(" radius="0.008" dim=")" + (shell ? "2" : "3") + R"(" mass="1.0"
                    pos="0 0 0.098" dof="trilinear">
            <contact selfcollide="none" passive="true"/>
            )" +
          (shell
               ? R"(<elasticity young="1e5" poisson="0.3" damping="0.05" elastic2d="stretch" thickness="0.015"/>)"
               : R"(<elasticity young="1e5" poisson="0.3" damping="0.05"/>)") +
          R"(
          </flexcomp>
        </worldbody>
      </mujoco>
      )";
      char rerr[1024] = {0};
      MjModelPtr rm = LoadModelFromString(ref_xml.c_str(), rerr, sizeof(rerr));
      ASSERT_THAT(rm, NotNull()) << rerr;
      MjDataPtr rd = MakeData(rm);
      for (int step = 0; step < 200; step++) {
        mj_step(rm.get(), rd.get());
      }
      mjtNum max_pen = 0;
      for (int c = 0; c < rd->ncon; c++) {
        max_pen = mju_max(max_pen, -rd->contact[c].dist);
      }
      if (fine) {
        max_pen_fine = max_pen;
      } else {
        max_pen_coarse = max_pen;
      }
    }
    EXPECT_GT(max_pen_coarse, 0);
    EXPECT_GT(max_pen_fine, 0);
    // 7x7 = 49 bottom vertices vs 3x3 = 9 bottom vertices (5.4x more contacts
    // per face): penetration remains within a factor of 2.5 rather than
    // dropping >5x.
    EXPECT_LT(max_pen_coarse / max_pen_fine, 2.5);
    EXPECT_LT(max_pen_fine / max_pen_coarse, 2.5);
  }
}

TEST_F(ImplicitIntegratorTest, FlexContactEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -10" timestep="0.001" integrator="discrete"
            solver="CG" tolerance="1e-6">
      <flag energy="enable"/>
    </option>
    <default>
      <geom solref="0.003 1"/>
    </default>
    <worldbody>
      <geom type="plane" size="5 5 0.1"/>
      <flexcomp type="grid" count="8 8 1" spacing=".04 .04 .04"
                radius=".01" name="sheet" dim="2" pos="0 0 0.02" mass="0.1">
        <edge equality="true"/>
        <elasticity young="3e6" poisson="0" thickness="2e-2"
                    elastic2d="bend" damping="0"/>
        <contact solref="0.003 1" selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);

  MjDataPtr d = MakeData(m);

  // compute initial energy
  mj_forward(m.get(), d.get());
  mjtNum initial_energy = d->energy[0] + d->energy[1];
  ASSERT_GT(initial_energy, 0);

  // simulate
  mjtNum max_energy = initial_energy;
  int max_energy_step = 0;
  int nsteps = 500;
  for (int i = 0; i < nsteps; i++) {
    mj_step(m.get(), d.get());
    mjtNum total_energy = d->energy[0] + d->energy[1];
    if (total_energy > max_energy) {
      max_energy = total_energy;
      max_energy_step = i + 1;
    }
  }

  mjtNum energy_ratio = max_energy / initial_energy;

  EXPECT_LE(energy_ratio, 1.01)
      << "contact solver injected energy: max_energy/initial_energy = "
      << energy_ratio << " (max at step " << max_energy_step << ")"
      << "\n  initial_energy = " << initial_energy
      << "\n  max_energy     = " << max_energy;
}

// bending damping on a flat flex must dissipate energy with implicit integrator
TEST_F(ImplicitIntegratorTest, BendingDampingDecaysEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.001" integrator="discrete" solver="CG">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <flexcomp type="grid" count="6 6 1" spacing=".1 .1 .1"
                radius=".005" name="sheet" dim="2" mass="0.1">
        <edge equality="false" damping="0" stiffness="0"/>
        <elasticity young="1e6" poisson="0" thickness="0.02"
                    elastic2d="bend" damping="0.1"/>
        <contact solref="0.01" selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 1);
  ASSERT_GT(m->flex_damping[0], 0) << "flex_damping not set";

  MjDataPtr d = MakeData(m);

  // perturb a central vertex with upward velocity
  // vertex layout is 6x6 grid; pick a central vertex (row=3, col=3 -> id=21)
  int center_vert = 21;
  int bid = m->flex_vertbodyid[m->flex_vertadr[0] + center_vert];
  int dofadr = m->body_dofadr[bid];
  d->qvel[dofadr + 2] = 1.0;  // z-velocity

  // initial forward to compute energy
  mj_forward(m.get(), d.get());
  mjtNum initial_energy = d->energy[0] + d->energy[1];
  ASSERT_GT(initial_energy, 0) << "initial energy should be nonzero";

  // step forward and check energy decay
  mjtNum max_energy = initial_energy;
  int nsteps = 100;
  for (int i = 0; i < nsteps; i++) {
    mj_step(m.get(), d.get());
    mjtNum total_energy = d->energy[0] + d->energy[1];
    max_energy = mju_max(max_energy, total_energy);
  }

  // energy must never exceed initial (system must not go unstable)
  EXPECT_LE(max_energy, initial_energy * 1.01)
      << "energy exceeded initial by more than 1%: max=" << max_energy
      << ", initial=" << initial_energy;

  // after 100 steps (0.1 seconds), energy should have decayed significantly
  mjtNum final_energy = d->energy[0] + d->energy[1];
  EXPECT_LT(final_energy, 0.5 * initial_energy)
      << "energy did not decay by at least 50% after " << nsteps << " steps"
      << " (initial=" << initial_energy << ", final=" << final_energy << ")";
}

// interp stretch stiffness with implicitfast must preserve energy stability
TEST_F(ImplicitIntegratorTest, InterpStretchEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.001" integrator="discrete" solver="CG">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <flexcomp type="grid" count="4 4 4" cellcount="3 3 3"
                spacing=".05 .05 .05" radius=".005" name="cube"
                dim="3" mass="10" dof="trilinear">
        <elasticity young="1e6" poisson="0.3" damping="0"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  MjDataPtr d = MakeData(m);

  // perturb a central vertex with velocity
  int center_body = m->nbody / 2;
  int dofadr = m->body_dofadr[center_body];
  ASSERT_GT(m->body_dofnum[center_body], 0);
  d->qvel[dofadr + 2] = 1.0;  // z-velocity

  mj_forward(m.get(), d.get());
  mjtNum initial_energy = d->energy[0] + d->energy[1];
  ASSERT_GT(initial_energy, 0) << "initial energy should be nonzero";

  // step and track max energy
  mjtNum max_energy = initial_energy;
  int nsteps = 50;
  for (int i = 0; i < nsteps; i++) {
    mj_step(m.get(), d.get());
    mjtNum total_energy = d->energy[0] + d->energy[1];
    max_energy = mju_max(max_energy, total_energy);
  }

  // energy must not blow up
  EXPECT_LE(max_energy, initial_energy * 1.01)
      << "energy exceeded initial by more than 1%: max=" << max_energy
      << ", initial=" << initial_energy;
}

// with the implicit effective metric active, inverse dynamics must recover the
// applied force (zero here): the forward solve is (M+B)*qacc = qfrc_smooth + c
// + J'*f and the inverse adds the same B*qacc - c terms. This is the fwd/inv
// consistency fence for the discrete metric.
TEST_F(ForwardTest, DiscreteFlexInverseConsistency) {
  static const char* const kXml = R"(
  <mujoco>
    <option solver="CG" integrator="discrete" tolerance="1e-14"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.1 0.1 0.1"
                radius=".01" dim="2" mass="1" pos="0 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01"
                    elastic2d="both" damping="0.5"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  // deform and settle a few steps under gravity
  for (int i = 0; i < nv; i++) {
    data->qvel[i] = 0.1 * (mju_Halton(i, 3) - 0.5);
  }
  for (int step = 0; step < 50; step++) {
    mj_step(model.get(), data.get());
  }

  // forward then inverse at the same state
  mj_forward(model.get(), data.get());
  mj_inverse(model.get(), data.get());

  // no applied forces: the inverse must return ~zero, at the scale of the
  // passive forces
  mjtNum scale = 1 + mju_norm(data->qfrc_passive, nv);
  EXPECT_LT(mju_norm(data->qfrc_inverse, nv), MjTol(1e-6, 1e-4) * scale);
}

// the fluid drag and passive flex contact terms of the discrete integrator's
// effective metric follow the spring and damper disable flags, as the forces
// do: mj_passive skips both, like every passive force, only when both flags
// are disabled. Neither model has springs, dampers, other metric terms or
// constraints, so disabling either flag alone must leave the step unchanged,
// and disabling both must reduce the smooth acceleration to M^-1 * qfrc_smooth.
TEST_F(ForwardTest, DiscretePassiveDisableFlags) {
  static constexpr char fluid[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" density="1000" viscosity="1"/>
    <worldbody>
      <body>
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <geom type="box" size=".1 .05 .01" mass=".2"/>
      </body>
      <body pos="1 0 0">
        <joint type="slide" axis="1 0 0"/>
        <joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <geom type="box" size=".1 .05 .01" mass=".2" fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char contact[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG"/>
    <worldbody>
      <geom type="box" size=".5 .5 .05" pos="0 0 -.05"/>
      <flexcomp type="grid" count="4 4 1" spacing=".05 .05 1" radius=".005" dim="2" mass=".1" pos="0 0 .004" name="sheet">
        <contact passive="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {fluid, contact}) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    int nv = m->nv;

    // qacc at a moving state, with the given disable flags
    auto qacc = [&](int flags) {
      m->opt.disableflags = flags;
      mj_resetData(m.get(), d.get());
      for (int i = 0; i < nv; i++) {
        d->qvel[i] = mju_Halton(i, 3) - 0.5;
      }
      mj_forward(m.get(), d.get());
      return AsVector(d->qacc, nv);
    };

    qacc(0);
    EXPECT_GT(mju_norm(d->qfrc_passive, nv), 0.1)
        << "test should exercise a nontrivial passive force";
    EXPECT_EQ(qacc(mjDSBL_SPRING), qacc(0));
    EXPECT_EQ(qacc(mjDSBL_DAMPER), qacc(0));

    // both flags disabled: no passive force, and the metric reduces to M
    qacc(mjDSBL_SPRING | mjDSBL_DAMPER);
    ASSERT_EQ(d->nefc, 0);
    std::vector<mjtNum> expected(nv), diff(nv);
    mj_solveM(m.get(), d.get(), expected.data(), d->qfrc_smooth, 1);
    mju_sub(diff.data(), d->qacc_smooth, expected.data(), nv);
    EXPECT_LT(mju_norm(diff.data(), nv),
              MjTol(1e-12, 1e-5) * mju_norm(expected.data(), nv));
  }
}

// the flex terms of the discrete integrator's effective metric follow the
// spring and damper disable flags, as the flex forces do. Scaling the flex
// stiffness K and damping d by powers of 2 is exact, so each flag setting has a
// scaling which leaves every enabled force, and therefore the step, bitwise
// unchanged: any scaling with both flags disabled, (4K, d/4) with springs
// disabled (the damping force depends on d*K), and (K, 4d) with dampers
// disabled. The first model assembles its flex terms into the metric CSR; in
// the second, bending and an interpolated flex under a moving body are applied
// matrix-free.
TEST_F(ForwardTest, DiscreteFlexDisableFlags) {
  static constexpr char assembled[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG"/>
    <worldbody>
      <flexcomp type="grid" count="4 4 1" spacing=".1 .1 .1" radius=".01" dim="2"
                mass="1" name="cloth">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="1e-2" elastic2d="both" damping="0.05"/>
        <pin id="0"/>
      </flexcomp>
      <flexcomp type="grid" count="3 3 3" spacing=".1 .1 .1" radius=".01" dim="3" mass="1"
                name="solid" pos="1 0 0" dof="trilinear">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  static constexpr char matrix_free[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG"/>
    <worldbody>
      <flexcomp type="grid" count="4 4 1" spacing=".1 .1 .1" radius=".01" dim="2"
                mass="1" name="cloth">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="1e-2" elastic2d="bend" damping="0.05"/>
        <pin id="0"/>
      </flexcomp>
      <body name="parent" pos="1 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom size=".01" mass="1" contype="0" conaffinity="0"/>
        <flexcomp type="grid" count="3 3 3" spacing=".1 .1 .1" radius=".01" dim="3" mass="1"
                  name="solid" dof="trilinear">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.3" damping="0.05"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {assembled, matrix_free}) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    int nv = m->nv;
    std::vector<mjtNum> stiffness =
        AsVector(m->flex_stiffness, m->nflexstiffness);
    std::vector<mjtNum> bending = AsVector(m->flex_bending, m->nflexbending);
    std::vector<mjtNum> damping = AsVector(m->flex_damping, m->nflex);

    // qacc at a deformed, moving state, with K and d scaled
    auto qacc = [&](int flags, mjtNum kscale, mjtNum dscale) {
      for (int i = 0; i < m->nflexstiffness; i++) {
        m->flex_stiffness[i] = kscale * stiffness[i];
      }
      for (int i = 0; i < m->nflexbending; i++) {
        m->flex_bending[i] = kscale * bending[i];
      }
      for (int f = 0; f < m->nflex; f++) {
        m->flex_damping[f] = dscale * damping[f];
      }
      m->opt.disableflags = flags;
      mj_resetData(m.get(), d.get());
      for (int i = 0; i < nv; i++) {
        d->qpos[i] += 5e-3 * (mju_Halton(i, 2) - 0.5);
        d->qvel[i] = 0.2 * (mju_Halton(i, 3) - 0.5);
      }
      mj_forward(m.get(), d.get());
      return AsVector(d->qacc, nv);
    };

    qacc(0, 1, 1);
    EXPECT_GT(mju_norm(d->qfrc_spring, nv), 0.1)
        << "test should exercise a nontrivial spring force";
    EXPECT_GT(mju_norm(d->qfrc_damper, nv), 0.1)
        << "test should exercise a nontrivial damping force";

    EXPECT_EQ(qacc(mjDSBL_SPRING | mjDSBL_DAMPER, 4, 4),
              qacc(mjDSBL_SPRING | mjDSBL_DAMPER, 1, 1));
    EXPECT_EQ(qacc(mjDSBL_SPRING, 4, 0.25), qacc(mjDSBL_SPRING, 1, 1));
    EXPECT_EQ(qacc(mjDSBL_DAMPER, 1, 4), qacc(mjDSBL_DAMPER, 1, 1));
  }
}

// ------------------------------ discrete integrator --------------------------

// with no position stiffness and no constraints, Euler (eulerdamp),
// implicitfast and discrete all compute the same update
// (M + h*B) v+ = M v + h*qfrc: velocities must match
TEST_F(ForwardTest, DiscreteMatchesImplicitDampingOnly) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="3"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
        <body pos=".5 0 0">
          <joint name="ball" type="ball" damping="1.5"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;
  const mjtNum qvel0[4] = {2, .5, -.3, .7};

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    mju_copy(data->qvel, qvel0, nv);
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_euler, qvel_fast, qvel_discrete;
  step_with(mjINT_EULER, qvel_euler);
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);

  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_euler[i], 1e-15, 1e-6))
        << "vs Euler, dof " << i;
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-15, 1e-6))
        << "vs implicitfast, dof " << i;
  }
}

// joint stiffness far beyond the explicit stability bound (omega*h >> 2): the
// discrete integrator is stable and dissipates toward the spring reference
TEST_F(ForwardTest, DiscreteStiffSpringStable) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" gravity="0 0 0"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0" stiffness="1e5"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_resetData(model.get(), data.get());
  data->qpos[0] = 0.5;
  mjtNum bound = 0.5 * 1.01;
  for (int i = 0; i < 1000; i++) {
    mj_step(model.get(), data.get());
    ASSERT_LT(mju_abs(data->qpos[0]), bound) << "unstable at step " << i;
  }

  // backward-Euler numerical dissipation has decayed the oscillation
  EXPECT_LT(mju_abs(data->qpos[0]), 0.05);
  EXPECT_LT(mju_abs(data->qvel[0]), 10);
}

// under discrete, qacc reported by forward is the step map: stepping advances
// the velocity by exactly h*qacc, including through contact
TEST_F(ForwardTest, DiscreteQaccIsStepMap) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  mj_resetData(model.get(), data.get());
  for (int i = 0; i < 10; i++) {
    std::vector<mjtNum> qvel_before(data->qvel, data->qvel + nv);
    mj_step(model.get(), data.get());
    ASSERT_GT(data->ncon, 0);
    for (int j = 0; j < nv; j++) {
      EXPECT_EQ(data->qvel[j],
                qvel_before[j] + model->opt.timestep * data->qacc[j])
          << "step " << i << " dof " << j;
    }
  }
}

// split stepping is the same trajectory: mj_step1 + ctrl + mj_step2 must match
// mj_step bit-for-bit under every single-step integrator, including those with
// the effective metric
TEST_F(ForwardTest, Step1Step2MatchesStep) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" solver="CG"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .3">
        <joint type="free" damping="0.5"/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".03 .02 .01"/>
      </body>
      <body pos="1 0 .1">
        <joint name="hinge" axis="0 1 0" damping="2"/>
        <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(model);
  MjDataPtr d2 = MakeData(model);
  mjModel* m = model.get();
  int nq = m->nq, nv = m->nv;

  for (int integrator :
       {mjINT_EULER, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    m->opt.integrator = integrator;
    mj_resetData(m, d1.get());
    mj_resetData(m, d2.get());
    for (int i = 0; i < 50; i++) {
      mjtNum ctrl = mju_sin(0.1 * i);
      d1->ctrl[0] = ctrl;
      mj_step(m, d1.get());
      mj_step1(m, d2.get());
      d2->ctrl[0] = ctrl;
      mj_step2(m, d2.get());
    }
    for (int k = 0; k < nq; k++) {
      EXPECT_EQ(d1->qpos[k], d2->qpos[k])
          << "integrator " << integrator << " qpos " << k;
    }
    for (int k = 0; k < nv; k++) {
      EXPECT_EQ(d1->qvel[k], d2->qvel[k])
          << "integrator " << integrator << " qvel " << k;
    }
  }
}

// diagonal classes (joint damping and stiffness) in contact: native discrete
// inverse dynamics recovers the (zero) applied force
TEST_F(ForwardTest, DiscreteJointInverseConsistency) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="200"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="2" stiffness="50" springref="10"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  // settle onto the floor so a contact is active, with nonzero velocity state
  mj_resetData(model.get(), data.get());
  for (int i = 0; i < 50; i++) {
    mj_step(model.get(), data.get());
  }
  data->qvel[0] = 0.1;

  mj_forward(model.get(), data.get());
  ASSERT_GT(data->ncon, 0);
  mj_inverse(model.get(), data.get());

  mjtNum scale = mju_norm(data->qfrc_passive, nv) +
                 mju_norm(data->qfrc_constraint, nv) +
                 mju_norm(data->qfrc_bias, nv);
  EXPECT_LT(mju_norm(data->qfrc_inverse, nv), MjTol(1e-6, 5e-5) * scale);

  // repeat under elliptic cones
  model->opt.cone = mjCONE_ELLIPTIC;
  mj_forward(model.get(), data.get());
  ASSERT_GT(data->ncon, 0);
  mj_inverse(model.get(), data.get());
  scale = mju_norm(data->qfrc_passive, nv) +
          mju_norm(data->qfrc_constraint, nv) + mju_norm(data->qfrc_bias, nv);
  EXPECT_LT(mju_norm(data->qfrc_inverse, nv), MjTol(1e-6, 5e-5) * scale);
}

TEST_F(ForwardTest, SNHRequiresDiscreteIntegrator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="test" dim="3" count="2 2 2">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" damping=".01"/>
      </flexcomp>
    </worldbody>
  </mujoco>)";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "test"))->elastic3d = 1;
  MjModelPtr m(mj_compile(spec, nullptr));
  mj_deleteSpec(spec);
  ASSERT_THAT(m.get(), NotNull());
  MjDataPtr d = MakeData(m);
  auto forward_error = MjuErrorMessageFrom(mj_forward);
  auto step_error = MjuErrorMessageFrom(mj_step);
  auto step1_error = MjuErrorMessageFrom(mj_step1);
  for (int integrator :
       {mjINT_EULER, mjINT_RK4, mjINT_IMPLICIT, mjINT_IMPLICITFAST}) {
    SCOPED_TRACE(integrator);
    m->opt.integrator = integrator;
    for (auto error :
         {forward_error(m.get(), d.get()), step_error(m.get(), d.get()),
          step1_error(m.get(), d.get())}) {
      EXPECT_THAT(
          error,
          HasSubstr(
              "stable Neo-Hookean elasticity requires integrator='discrete'"));
    }
  }
  m->opt.integrator = mjINT_DISCRETE;
  for (int solver : {mjSOL_CG, mjSOL_NEWTON}) {
    m->opt.solver = solver;
    EXPECT_EQ(forward_error(m.get(), d.get()), "");
    EXPECT_EQ(step_error(m.get(), d.get()), "");
  }
}

// flex elasticity lost its implicit treatment under implicit*: the migration is
// loud
TEST_F(ForwardTest, ImplicitFlexElasticityRequiresMetric) {
  static const char* const kXmls[] = {
      R"(
      <mujoco>
        <option integrator="discrete" solver="CG"/>
        <worldbody>
          <flexcomp name="sheet" type="grid" count="4 4 1" spacing=".05 .05 .05" radius=".01"
                    dim="2" mass=".1" pos="0 0 1">
            <elasticity young="1e4" poisson="0" thickness="1e-2" elastic2d="bend"/>
            <pin id="0 3"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      )",
      R"(
      <mujoco>
        <option integrator="discrete" solver="CG"/>
        <worldbody>
          <flexcomp name="rod" type="grid" count="4 1 1" spacing=".05 .05 .05" radius=".01"
                    dim="1" mass=".1" pos="0 0 1">
            <elasticity young="1e4"/>
            <pin id="0"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      )",
  };
  char error[1024];
  auto forward_error = MjuErrorMessageFrom(mj_forward);

  for (const char* xml : kXmls) {
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);

    // the model loads and runs under discrete; flipping to implicit or
    // implicitfast is a loud error
    EXPECT_EQ(forward_error(model.get(), data.get()), "");
    for (int integrator : {mjINT_IMPLICIT, mjINT_IMPLICITFAST}) {
      model->opt.integrator = integrator;
      EXPECT_THAT(forward_error(model.get(), data.get()),
                  testing::HasSubstr("flex elasticity is no longer integrated "
                                     "implicitly under integrator='implicit' "
                                     "and 'implicitfast': set "
                                     "integrator='discrete', or 'Euler' with "
                                     "a small timestep"));
    }

    // Euler always integrated elasticity explicitly and still does
    model->opt.integrator = mjINT_EULER;
    EXPECT_EQ(forward_error(model.get(), data.get()), "");
  }
}

// flex edge damping is integrated explicitly under implicitfast, so uniform
// translation of a free flex has zero relative edge velocity and is not damped,
// while relative vertex motion is damped
TEST_F(ForwardTest, FlexEdgeDampingTranslationalInvariance) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="implicitfast" gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="rope" type="grid" count="3 1 1" spacing=".1 .1 .1"
                radius=".01" dim="1" mass="1">
        <edge damping="10"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // give all 3 vertices the same uniform velocity (1, -2, 3)
  for (int v = 0; v < 3; v++) {
    data->qvel[3 * v + 0] = 1.0;
    data->qvel[3 * v + 1] = -2.0;
    data->qvel[3 * v + 2] = 3.0;
  }
  mj_step(model.get(), data.get());

  // uniform translation must not be damped by edge damping
  for (int v = 0; v < 3; v++) {
    EXPECT_NEAR(data->qvel[3 * v + 0], 1.0, MjTol(1e-12, 1e-6));
    EXPECT_NEAR(data->qvel[3 * v + 1], -2.0, MjTol(1e-12, 1e-6));
    EXPECT_NEAR(data->qvel[3 * v + 2], 3.0, MjTol(1e-12, 1e-6));
  }

  // give vertices differing velocities along the edge direction (x)
  mj_resetData(model.get(), data.get());
  data->qvel[0] = 1.0;
  mj_step(model.get(), data.get());

  // relative motion between vertices 0 and 1 is damped
  EXPECT_LT(data->qvel[0], 1.0);
  EXPECT_GT(data->qvel[3], 0.0);
  EXPECT_LT(data->qvel[0] - data->qvel[3], 1.0);
}

// flex edge stiffness and damping are not supported under discrete
TEST_F(ForwardTest, FlexEdgeStiffnessDampingDisallowedInDiscrete) {
  static const char* const kXmls[] = {
      R"(
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="3 1 1" spacing=".1 .1 .1"
                    radius=".01" dim="1" mass="1">
            <edge stiffness="10"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      )",
      R"(
      <mujoco>
        <worldbody>
          <flexcomp name="rope" type="grid" count="3 1 1" spacing=".1 .1 .1"
                    radius=".01" dim="1" mass="1">
            <edge damping="1"/>
          </flexcomp>
        </worldbody>
      </mujoco>
      )",
  };
  char error[1024];
  auto forward_error = MjuErrorMessageFrom(mj_forward);

  for (const char* xml : kXmls) {
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);

    for (int integrator :
         {mjINT_EULER, mjINT_RK4, mjINT_IMPLICIT, mjINT_IMPLICITFAST}) {
      model->opt.integrator = integrator;
      EXPECT_EQ(forward_error(model.get(), data.get()), "");
    }
    model->opt.integrator = mjINT_DISCRETE;
    EXPECT_THAT(forward_error(model.get(), data.get()),
                testing::HasSubstr("discrete integrator: flex edge stiffness "
                                   "and damping are not supported; use "
                                   "<elasticity> or <edge equality> instead"));
  }
}

// the strain equality mode stores its constraint eigenmodes in flex_stiffness:
// they are not elasticity, so the migration error must not fire
TEST_F(ForwardTest, StrainEqualityIsNotElasticity) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="implicitfast"/>
    <worldbody>
      <flexcomp name="beam" type="box" spacing=".1 .1 .1" radius=".001" mass="1"
                dim="3" dof="trilinear">
        <contact selfcollide="none"/>
        <edge equality="strain"/>
        <pin id="0 1 2 3"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  // the stiffness block is allocated and nonzero (it holds the eigenmodes), yet
  // the flex contributes nothing to the metric
  ASSERT_EQ(model->flex_edgeequality[0], 3);
  ASSERT_GE(model->flex_stiffnessadr[0], 0);
  ASSERT_NE(model->flex_stiffness[model->flex_stiffnessadr[0]], 0);
  EXPECT_FALSE(mj_effFlexPossible(model.get(), 0));

  MjDataPtr data = MakeData(model);
  auto forward_error = MjuErrorMessageFrom(mj_forward);
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  model->opt.integrator = mjINT_IMPLICIT;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
}

// unsupported option combinations are runtime errors
TEST_F(ForwardTest, DiscreteUnsupportedOptions) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="h" type="hinge" damping="1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="h" kp="10"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  auto forward_error = MjuErrorMessageFrom(mj_forward);

  // PGS and noslip are supported: joint damping is a backbone term, and the
  // servo is excluded from the metric under PGS (mj_effCouplings), integrating
  // explicitly
  model->opt.solver = mjSOL_PGS;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  EXPECT_EQ(data->nefmA, 0);

  model->opt.solver = mjSOL_CG;
  model->opt.noslip_iterations = 5;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");

  // tendon terms enter the metric under primal solvers; under PGS the class is
  // excluded and the tendon force integrates explicitly
  static const char* const kXmlTendon = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body pos="0 0 1">
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass="1"/>
        <site name="a" pos="0 0 .1"/>
      </body>
      <site name="b" pos="0 0 2"/>
    </worldbody>
    <tendon>
      <spatial stiffness="100" range="0 2">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model_t = LoadModelFromString(kXmlTendon, error, sizeof(error));
  ASSERT_THAT(model_t.get(), NotNull()) << error;
  MjDataPtr data_t = MakeData(model_t);
  mj_forward(model_t.get(), data_t.get());
  EXPECT_EQ(data_t->nefmT, 1);
  model_t->opt.solver = mjSOL_PGS;
  EXPECT_EQ(forward_error(model_t.get(), data_t.get()), "");
  EXPECT_EQ(data_t->nefmT, 0);

  // same exclusion for a servo on a multi-dof transmission: its rank-1 metric
  // term has off-diagonal content the backbone factor does not carry
  static const char* const kXmlServo = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="s1" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
      <body pos=".2 0 1">
        <joint name="s2" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="T">
        <joint joint="s1" coef="1"/>
        <joint joint="s2" coef="1"/>
      </fixed>
    </tendon>
    <actuator>
      <position tendon="T" kp="10"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model_s = LoadModelFromString(kXmlServo, error, sizeof(error));
  ASSERT_THAT(model_s.get(), NotNull()) << error;
  MjDataPtr data_s = MakeData(model_s);
  EXPECT_EQ(forward_error(model_s.get(), data_s.get()), "");
  EXPECT_EQ(data_s->nefmA, 1);
  model_s->opt.solver = mjSOL_PGS;
  EXPECT_EQ(forward_error(model_s.get(), data_s.get()), "");
  EXPECT_EQ(data_s->nefmA, 0);

  // flex cannot be excluded from the metric: explicit flex elasticity diverges
  static const char* const kXmlFlex = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="sheet" type="grid" count="4 4 1" spacing=".05 .05 .05" radius=".01"
                dim="2" mass=".1" pos="0 0 1">
        <elasticity young="1e4" poisson="0" thickness="1e-2" elastic2d="bend"/>
        <pin id="0 3"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model_f = LoadModelFromString(kXmlFlex, error, sizeof(error));
  ASSERT_THAT(model_f.get(), NotNull()) << error;
  MjDataPtr data_f = MakeData(model_f);
  EXPECT_EQ(forward_error(model_f.get(), data_f.get()), "");
  model_f->opt.solver = mjSOL_PGS;
  EXPECT_THAT(forward_error(model_f.get(), data_f.get()),
              testing::HasSubstr("not yet supported with flex"));

  // interpolated flex anchored to a moving body: supported under both CG and
  // Newton via general DOF-level CSR assembly
  static const char* const kXmlInterp = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body name="arm" pos="0 0 1">
        <joint type="hinge"/>
        <geom type="sphere" size=".05" mass="1"/>
        <flexcomp name="soft" type="grid" count="3 3 3" spacing=".1 .1 .1" dof="trilinear"
                  dim="3" radius=".01" mass=".1" pos="0 0 -.5">
          <elasticity young="1e4" poisson="0.2"/>
          <contact selfcollide="none"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model_i = LoadModelFromString(kXmlInterp, error, sizeof(error));
  ASSERT_THAT(model_i.get(), NotNull()) << error;
  MjDataPtr data_i = MakeData(model_i);
  model_i->opt.solver = mjSOL_CG;
  EXPECT_EQ(forward_error(model_i.get(), data_i.get()), "");
  model_i->opt.solver = mjSOL_NEWTON;
  EXPECT_EQ(forward_error(model_i.get(), data_i.get()), "");

  model->opt.noslip_iterations = 0;
  model->opt.enableflags |= mjENBL_SLEEP;
  model->opt.disableflags |= mjDSBL_ISLAND;
  EXPECT_THAT(forward_error(model.get(), data.get()),
              testing::HasSubstr("sleep without islands"));

  // sleep with islands is supported, including with flex
  model->opt.disableflags &= ~mjDSBL_ISLAND;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  model_f->opt.solver = mjSOL_CG;
  model_f->opt.enableflags |= mjENBL_SLEEP;
  EXPECT_EQ(forward_error(model_f.get(), data_f.get()), "");
  model_f->opt.disableflags |= mjDSBL_ISLAND;
  EXPECT_THAT(forward_error(model_f.get(), data_f.get()),
              testing::HasSubstr("sleep without islands"));

  // valid options again: no error, under both supported solvers and both cones
  model->opt.enableflags &= ~mjENBL_SLEEP;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  model->opt.solver = mjSOL_NEWTON;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  model->opt.cone = mjCONE_ELLIPTIC;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
  model->opt.solver = mjSOL_CG;
  EXPECT_EQ(forward_error(model.get(), data.get()), "");
}

// Newton and CG solve the same discrete problem: qacc must match, for the
// diagonal classes, an assembled flex, and a bending-only flex (whose CSR
// assembly Newton forces)
TEST_F(ForwardTest, DiscreteNewtonMatchesCG) {
  static const char* const kXmls[] = {
      // ellipsoid-fluid paddle on a hinge, in contact
      R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500" density="1000" viscosity="1"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .08">
        <joint name="hinge" type="hinge" axis="0 1 0" damping=".1"/>
        <geom type="box" size=".2 .15 .005" pos=".2 0 0" mass=".3" fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )",
      // joint spring-damper in contact
      R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="2" stiffness="500"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )",
      // stretch + bending cloth: CSR assembled under both solvers
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.05 0.05 0.05"
                radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )",
      // bending-only cloth: CG uses the matrix-free stencil, Newton forces
      // assembly
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="5 5 1" spacing="0.05 0.05 0.05"
                radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="bend" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )",
      // cross-tree stiff damped tendon in contact: tendon terms in the metric
      R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="-.2 0 .05"><joint name="s1" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body pos=".2 0 .05"><joint name="s2" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
    </worldbody>
    <tendon><spatial stiffness="3e3" springlength="0.3" damping="10"><site site="a"/><site site="b"/></spatial></tendon>
  </mujoco>
  )",
      // stiff position servo pressing into contact: actuator terms in the
      // metric
      R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
    <actuator><position joint="hinge" kp="2e4" kv="50"/></actuator>
  </mujoco>
  )",
  };

  for (const char* xml : kXmls) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    int nv = model->nv;

    for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
      model->opt.cone = cone;
      model->opt.solver = mjSOL_CG;

      // develop a nontrivial state under CG
      mj_resetData(model.get(), data.get());
      for (int i = 0; i < 20; i++) {
        mj_step(model.get(), data.get());
      }

      mj_forward(model.get(), data.get());
      std::vector<mjtNum> qacc_cg(data->qacc, data->qacc + nv);

      model->opt.solver = mjSOL_NEWTON;
      mj_forward(model.get(), data.get());

      mjtNum scale = mju_norm(qacc_cg.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
      for (int i = 0; i < nv; i++) {
        EXPECT_THAT(data->qacc[i],
                    MjNear(qacc_cg[i], 1e-13 * scale, 1e-4 * scale))
            << "cone " << cone << " dof " << i;
      }

      // Newton remains stable when stepping
      for (int i = 0; i < 100; i++) {
        mj_step(model.get(), data.get());
      }
      EXPECT_LT(mju_norm(data->qvel, nv), 1e4);
    }
  }
}

TEST_F(ForwardTest, DiscreteArticulatedFlexNewtonMatchesCG) {
  static const char* const kXmls[] = {
      // articulated 1D elastic rod pinned to a hinged arm in contact
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .06">
        <joint name="hinge" type="hinge" axis="0 1 0" damping=".2"/>
        <geom type="sphere" size=".03" mass=".5"/>
        <flexcomp name="rod" type="grid" count="5 1 1" spacing=".05 .05 .05"
                  radius=".01" dim="1" mass=".5" pos="0 0 -.02">
          <pin id="0"/>
          <contact selfcollide="none"/>
          <edge equality="true"/>
          <elasticity young="5e4" damping=".05"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )",
      // articulated clamped rod (per-edge twist dofs) on a ball-jointed arm in
      // contact
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .06">
        <joint type="ball" damping=".2"/>
        <geom type="sphere" size=".03" mass=".5"/>
        <flexcomp name="rod" type="grid" count="6 1 1" spacing=".05 .05 .05"
                  radius=".01" dim="1" mass=".5" pos="0 0 -.05">
          <pin id="0 1"/>
          <contact selfcollide="none"/>
          <elasticity young="5e4" poisson=".25" damping=".05"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )",
      // articulated 2D cloth pinned to a 2-link arm in contact, with a second
      // separate island
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="link1" pos="0 0 .12">
        <joint type="hinge" axis="0 1 0" damping=".1"/>
        <geom type="sphere" size=".03" mass=".5"/>
        <body name="link2" pos=".1 0 0">
          <joint type="ball" damping=".1"/>
          <geom type="sphere" size=".03" mass=".5"/>
          <flexcomp name="cloth" type="grid" count="4 4 1" spacing=".05 .05 .05"
                    radius=".008" dim="2" mass="1" pos="0 0 -.08" dof="full">
            <pin id="0 3"/>
            <contact selfcollide="none"/>
            <elasticity young="2e3" poisson=".2" damping=".1" elastic2d="both" thickness=".01"/>
          </flexcomp>
        </body>
      </body>
      <body name="other_island" pos="1 0 .04">
        <freejoint/>
        <geom type="sphere" size=".05" mass=".5"/>
      </body>
    </worldbody>
  </mujoco>
  )",
      // articulated 3D solid pinned to a hinged body in contact
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .082">
        <joint type="hinge" axis="0 1 0" damping=".2"/>
        <geom type="sphere" pos=".02 0 .03" size=".02" mass=".5"/>
        <flexcomp name="solid" type="grid" count="2 2 2" spacing=".05 .05 .05"
                  radius=".008" dim="3" mass="1" pos="0 0 -.05" dof="full">
          <pin id="1 3"/>
          <contact selfcollide="none"/>
          <elasticity young="5e3" poisson=".25" damping=".1"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )",
      // articulated trilinear and shell interpolated flexes pinned to a moving
      // body in contact
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .13">
        <joint type="hinge" axis="0 1 0" damping=".2"/>
        <geom type="sphere" pos=".02 0 .04" size=".02" mass=".5"/>
        <flexcomp name="tri" type="grid" count="3 3 3" spacing=".05 .05 .05"
                  radius=".008" dim="3" mass="1" pos="0 0 -.074" dof="trilinear">
          <pin id="2"/>
          <contact selfcollide="none"/>
          <elasticity young="5e3" poisson=".2" damping=".1"/>
        </flexcomp>
        <flexcomp name="shell" type="grid" count="4 4 4" spacing=".04 .04 .04"
                  cellcount="1 2 2" radius=".008" dim="3" mass="1"
                  pos=".25 0 -.064" dof="trilinear">
          <pin id="2"/>
          <contact selfcollide="none"/>
          <elasticity young="5e3" poisson=".2" damping=".1"
                      elastic2d="stretch" thickness=".01"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )",
      // articulated quadratic interpolated flex pinned to a moving body in
      // contact
      R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .13">
        <joint type="hinge" axis="0 1 0" damping=".2"/>
        <geom type="sphere" pos=".02 0 .04" size=".02" mass=".5"/>
        <flexcomp name="quad" type="grid" count="3 3 3" spacing=".05 .05 .05"
                  cellcount="1 1 1" radius=".008" dim="3" mass="1"
                  pos="0 0 -.074" dof="quadratic">
          <pin id="2"/>
          <contact selfcollide="none"/>
          <elasticity young="5e3" poisson=".2" damping=".1"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )",
  };

  for (size_t xml_idx = 0; xml_idx < std::size(kXmls); ++xml_idx) {
#ifdef mjUSESINGLE
    // in single precision, CG diverges on a rod clamped to a moving body with
    // per-edge twist dofs, independently of the articulated assembly
    if (xml_idx == 1) continue;
#endif
    const char* xml = kXmls[xml_idx];
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    int nv = model->nv;

    // rods carry per-edge twist dofs, coupled to the articulated vertices
    if (mj_name2id(model.get(), mjOBJ_FLEX, "rod") >= 0) {
      EXPECT_GE(mj_name2id(model.get(), mjOBJ_JOINT, "rod_twist"), 0)
          << "xml " << xml_idx;
    }

    for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
      model->opt.cone = cone;
      model->opt.solver = mjSOL_CG;

      mj_resetData(model.get(), data.get());
      for (int i = 0; i < 20; i++) {
        mj_step(model.get(), data.get());
      }

      mj_forward(model.get(), data.get());
      EXPECT_GT(data->ncon, 0) << "xml " << xml_idx;
      EXPECT_GT(data->nefc, 0) << "xml " << xml_idx;
      int nisl = data->nisland > 0 ? data->nisland : 1;
#ifndef mjUSESINGLE
      // the 1e-14 CG tolerance is only reachable in double precision
      for (int isl = 0; isl < nisl; isl++) {
        EXPECT_LT(data->solver_niter[isl], model->opt.iterations);
      }
#endif
      std::vector<mjtNum> qacc_cg(data->qacc, data->qacc + nv);
      std::vector<mjtNum> smooth_cg(data->qacc_smooth, data->qacc_smooth + nv);

      model->opt.solver = mjSOL_NEWTON;
      mj_forward(model.get(), data.get());
      EXPECT_GT(data->nefmK, 0);
      EXPECT_EQ(data->efm_active, 2);
      // dof 0 is the joint of the arm carrying the flex: the articulated
      // pull-back is assembled
      EXPECT_GT(data->efm_K_rownnz[0], 0) << "xml " << xml_idx;
      for (int isl = 0; isl < nisl; isl++) {
        EXPECT_LT(data->solver_niter[isl], 25);
      }

      mjtNum scale = mju_norm(qacc_cg.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
      for (int i = 0; i < nv; i++) {
        EXPECT_THAT(data->qacc_smooth[i],
                    MjNear(smooth_cg[i], 1e-8 * scale, 4e-3 * scale))
            << "xml " << xml_idx << " cone " << cone << " dof " << i;
        EXPECT_THAT(data->qacc[i],
                    MjNear(qacc_cg[i], 1e-6 * scale, 1e-4 * scale))
            << "xml " << xml_idx << " cone " << cone << " dof " << i;
      }

      for (int i = 0; i < 100; i++) {
        mj_step(model.get(), data.get());
      }
      EXPECT_LT(mju_norm(data->qvel, nv), 1e4);
    }
  }
}

// Newton with an articulated cloth under the IPC contact flag and under sleep:
// both paths now reach the general flex assembly, so it must step stably and
// keep the articulated rows. IPC accepts pins on slide chains only; a single
// oblique slide is still a general (non-simple) attachment.
TEST_F(ForwardTest, DiscreteArticulatedFlexNewtonIpcAndSleep) {
  // sleep: a cloth pinned to a hinged arm
  static const char* const kSleepXml = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="arm" pos="0 0 .12">
        <joint type="hinge" axis="0 1 0" damping=".2"/>
        <geom type="sphere" size=".02" pos=".02 0 .03" mass=".3"/>
        <flexcomp name="cloth" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                  radius=".004" mass=".1" pos="0 0 -.05">
          <pin id="0 4"/>
          <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both"
                      damping="1e-4"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  // IPC: free vertices must not be under jointed bodies and pins must ride
  // slide chains, so the patch has three free vertices and one on a jointless
  // child of a body on an oblique slide. There is no plane: the slide would
  // drive the pinned vertex into it, which the IPC barrier cannot resolve.
  static const char* const kIpcXml = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="Newton" iterations="400">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <body name="v0" pos="0 0 .1">
        <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass=".05" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
      <body name="v1" pos=".1 0 .1">
        <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass=".05" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
      <body name="v2" pos="0 .1 .1">
        <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/>
        <joint type="slide" axis="0 0 1"/>
        <inertial pos="0 0 0" mass=".05" diaginertia="1e-5 1e-5 1e-5"/>
      </body>
      <body name="carrier" pos=".1 .1 .1">
        <joint type="slide" axis="1 0 1" damping=".2"/>
        <inertial pos="0 0 0" mass=".1" diaginertia="1e-4 1e-4 1e-4"/>
        <body name="v3"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="patch" dim="2" body="v0 v1 v2 v3" element="0 1 2 1 3 2" radius=".004">
        <elasticity young="1e5" poisson=".2" thickness="2e-3" elastic2d="both"
                    damping="1e-4"/>
      </flex>
    </deformable>
  </mujoco>
  )";

  for (const char* xml : {kSleepXml, kIpcXml}) {
    const char* name = xml == kIpcXml ? "ipc" : "sleep";
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << name << ": " << error;
    MjDataPtr data = MakeData(model);
    int nv = model->nv;

    for (int i = 0; i < 300; i++) {
      mj_step(model.get(), data.get());
      ASSERT_FALSE(data->warning[mjWARN_BADQACC].number)
          << name << ": diverged at step " << i;
    }
    mj_forward(model.get(), data.get());
    EXPECT_EQ(data->efm_active, 2) << name;
    EXPECT_GT(data->nefmK, 0) << name;
    EXPECT_LT(mju_norm(data->qvel, nv), 1e2) << name;
  }
}

TEST_F(ForwardTest, DiscreteArticulatedFlexSolverFallbackAndCopy) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.001" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="arm" pos="0 0 .08">
        <joint type="ball" damping=".1"/>
        <geom type="sphere" size=".03" mass=".5"/>
        <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                  radius=".008" dim="2" mass=".2" pos="0 0 -.06" dof="full">
          <pin id="0 2"/>
          <contact selfcollide="none"/>
          <elasticity young="3e3" poisson=".2" damping=".1" elastic2d="both" thickness=".01"/>
        </flexcomp>
      </body>
      <body name="other" pos="1 0 .04">
        <freejoint/>
        <geom type="sphere" size=".05" mass=".4"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  for (int i = 0; i < 15; i++) {
    mj_step(model.get(), data.get());
  }

  std::vector<mjtNum> qacc_cg_noisland;
  for (int use_island : {0, 1}) {
    if (use_island) {
      model->opt.disableflags &= ~mjDSBL_ISLAND;
    } else {
      model->opt.disableflags |= mjDSBL_ISLAND;
    }

    model->opt.solver = mjSOL_CG;
    mj_forward(model.get(), data.get());
    EXPECT_GT(data->nefc, 0) << "use_island=" << use_island;
    if (use_island) {
      // the island CG path applies the matrix-free operators per island
      // (flexMatrixFree)
      EXPECT_GT(data->nisland, 0);
    }
    std::vector<mjtNum> qacc_cg(data->qacc, data->qacc + nv);
    mjtNum scale = mju_norm(qacc_cg.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    if (use_island) {
      // island and monolithic CG solve the same problem, to CG's
      // sqrt(tolerance) accuracy
      EXPECT_THAT(qacc_cg, Pointwise(MjNear(1e-6 * scale, 1e-3 * scale),
                                     qacc_cg_noisland));
    } else {
      qacc_cg_noisland = qacc_cg;
    }

    // Build general CSR metric under Newton, copy mjData, then run constraint
    // solve under CG (as happens when Newton falls back to CG).
    model->opt.solver = mjSOL_NEWTON;
    mj_forward(model.get(), data.get());
    EXPECT_GT(data->nefmK, 0);
    EXPECT_EQ(data->efm_active, 2);

    // the general CSR has rows for the ball joint of the arm
    int arm = mj_name2id(model.get(), mjOBJ_BODY, "arm");
    EXPECT_GT(data->efm_K_rownnz[model->body_dofadr[arm]], 0);

    mjData* copy = mj_copyData(nullptr, model.get(), data.get());
    model->opt.solver = mjSOL_CG;
    mj_fwdConstraint(model.get(), copy);

    EXPECT_THAT(AsVector(copy->qacc, nv),
                Pointwise(MjNear(1e-10 * scale, 1e-4 * scale), qacc_cg))
        << "use_island=" << use_island;
    mj_deleteData(copy);
  }
}

// implicit constraint rows: the one-step map of an equality row is exactly the
// backward-Euler row map. With constant impedance d, weight R/f and the
// transported, divided reference give the applied gain W = d/(d*f + 1 - d),
// f = 1 + h*b + h^2*k*d, and the map T = [1 - h^2*W*k*d, h*s; -h*W*k*d, s],
// s = 1 - W*(f - 1). The smooth spring pins the metric side of the identity
TEST_F(ForwardTest, DiscreteImplicitRowMap) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.02" integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <body name="eq1"><joint name="eq1" type="slide" axis="1 0 0"/><geom size=".1" mass="1"/></body>
      <body name="eq2" pos="0 1 0"><joint name="eq2" type="slide" axis="1 0 0"/><geom size=".1" mass="1"/></body>
      <body name="spring" pos="0 2 0">
        <joint name="spring" type="slide" axis="1 0 0" stiffness="625" damping="5"/>
        <geom size=".1" mass="1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="eq1" solref="0.4 0.1" solimp="0.9 0.9 0.001"/>
      <joint joint1="eq2" solref="0.02 0.3" solimp="0.95 0.95 0.001"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjtNum h = model->opt.timestep;

  // measure the engine's one-step map of dof j by stepping the unit states
  auto column = [&](int j, int col, mjtNum out[2]) {
    mj_resetData(model.get(), data.get());
    data->qpos[j] = col == 0 ? 1 : 0;
    data->qvel[j] = col == 0 ? 0 : 1;
    mj_step(model.get(), data.get());
    out[0] = data->qpos[j];
    out[1] = data->qvel[j];
  };

  // equality rows: solref (timeconst, dampratio), solimp d = d_width, d0 == d1
  const mjtNum tau[2] = {0.4, 0.02}, zeta[2] = {0.1, 0.3}, d[2] = {0.9, 0.95};
  mjtNum tol = MjTol(1e-12, 1e-5);
  for (int j = 0; j < 2; j++) {
    mjtNum b = 2 / (d[j] * tau[j]);
    mjtNum kd = d[j] / (d[j] * d[j] * tau[j] * tau[j] * zeta[j] * zeta[j]);
    mjtNum f = 1 + h * b + h * h * kd;
    mjtNum W = d[j] / (d[j] * f + 1 - d[j]);
    mjtNum s = 1 - W * (f - 1);
    mjtNum T[2][2] = {{1 - h * h * W * kd, h * s}, {-h * W * kd, s}};
    for (int col = 0; col < 2; col++) {
      mjtNum out[2];
      column(j, col, out);
      EXPECT_NEAR(out[0], T[0][col], tol) << "row " << j << " col " << col;
      EXPECT_NEAR(out[1], T[1][col], tol) << "row " << j << " col " << col;
    }
  }

  // smooth spring: (m + h*b + h^2*k) a = -k*q - (b + h*k) v, then the same
  // update
  mjtNum k = 625, b = 5, Mhat = 1 + h * b + h * h * k;
  mjtNum aq = -k / Mhat, av = -(b + h * k) / Mhat;
  mjtNum Ts[2][2] = {{1 + h * h * aq, h * (1 + h * av)}, {h * aq, 1 + h * av}};
  for (int col = 0; col < 2; col++) {
    mjtNum out[2];
    column(2, col, out);
    EXPECT_NEAR(out[0], Ts[0][col], tol) << "spring col " << col;
    EXPECT_NEAR(out[1], Ts[1][col], tol) << "spring col " << col;
  }
}

// implicit contact rows: a sphere resting on a plane with timeconst = h/20 is
// stable and quiet under discrete (refsafe does not clamp under this
// integrator), and the elliptic cone's regularized friction coefficient is
// derived from the implicit row impedances
TEST_F(ForwardTest, DiscreteImplicitRowContactStiff) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" cone="elliptic">
      <flag refsafe="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.1">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" solref="0.0005 1" condim="3"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // drop, settle: no warnings, resting in contact with tiny penetration and
  // velocity
  for (int i = 0; i < 200; i++) {
    mj_step(model.get(), data.get());
  }
  int nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_EQ(nwarning, 0);
  ASSERT_GT(data->ncon, 0);
  EXPECT_LT(mju_abs(data->contact[0].dist), MjTol(1e-3, 1e-3));
  EXPECT_LT(mju_norm(data->qvel, 6), MjTol(1e-8, 1e-5));

  // the contact carries the weight, and mu_reg = mu * sqrt(R[1]/R[0]) on
  // implicit R
  int adr = data->contact[0].efc_address;
  ASSERT_GE(adr, 0);
  EXPECT_GT(data->efc_force[adr], 9);
  mjtNum mu_expected = data->contact[0].friction[0] *
                       mju_sqrt(data->efc_R[adr + 1] / data->efc_R[adr]);
  EXPECT_NEAR(data->contact[0].mu, mu_expected, MjTol(1e-12, 1e-6));
}

// implicit rows at the impedance ceiling (refsafe disabled, so the authored
// rows are honored): an absurdly stiff solref pushes the row's effective
// impedance past mjMAXIMP, where the weight is floored (the reference keeps the
// full factor). The weights then equal those of a max-impedance model, a sphere
// rests rigidly and quietly, and a loose three-body pile at timeconst = 1e-9 --
// which blew up in single precision with unfloored weights -- survives
TEST_F(ForwardTest, DiscreteImplicitRowImpedanceCeiling) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete">
      <flag refsafe="disable"/>
    </option>
    <default>
      <geom %s/>
    </default>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.1">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" priority="1"/>
      </body>
      %s
    </worldbody>
  </mujoco>
  )";
  static constexpr char pile[] = R"(
      <body pos="0 0 0.25">
        <freejoint/>
        <geom type="box" size=".04 .04 .04" mass=".5" priority="2"/>
      </body>
      <body pos=".03 0 0.4">
        <freejoint/>
        <geom type="capsule" size=".03 .04" mass=".3" priority="3"/>
      </body>)";
  // three boxes at rest, yawed so the box-box faces are not aligned
  static constexpr char stack[] = R"(
      <body pos=".5 0 .05">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass="1" priority="2"/>
      </body>
      <body pos=".5 0 .14" euler="0 0 20">
        <freejoint/>
        <geom type="box" size=".04 .04 .04" mass=".5" priority="3"/>
      </body>
      <body pos=".5 0 .21" euler="0 0 40">
        <freejoint/>
        <geom type="box" size=".03 .03 .03" mass=".3" priority="4"/>
      </body>)";
  const char* extra[3] = {"", pile, stack};
  char error[1024];
  char xml[4096];

  for (int variant = 0; variant < 3; variant++) {
    std::snprintf(xml, sizeof(xml), xml_template, "solref=\"1e-9 1\"",
                  extra[variant]);
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    mjtNum maxspeed = 0;  // of the extra bodies, once settled
    for (int i = 0; i < 300; i++) {
      mj_step(model.get(), data.get());
      if (i >= 200) {
        maxspeed = mju_max(maxspeed, mju_norm(data->qvel + 6, model->nv - 6));
      }
    }
    int nwarning = 0;
    for (int w = 0; w < mjNWARNING; w++) {
      nwarning += data->warning[w].number;
    }
    EXPECT_EQ(nwarning, 0) << "variant " << variant;
    ASSERT_GT(data->ncon, 0);

    // the weights are those of a max-impedance model in the same configuration
    mj_forward(model.get(), data.get());
    std::snprintf(xml, sizeof(xml), xml_template,
                  "solimp=\"0.9999 0.9999 0.001\"", extra[variant]);
    MjModelPtr maximp = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(maximp.get(), NotNull()) << error;
    MjDataPtr data_maximp = MakeData(maximp);
    mju_copy(data_maximp->qpos, data->qpos, model->nq);
    mju_copy(data_maximp->qvel, data->qvel, model->nv);
    mj_forward(maximp.get(), data_maximp.get());
    ASSERT_EQ(data_maximp->nefc, data->nefc);
    for (int i = 0; i < data->nefc; i++) {
      EXPECT_NEAR(data->efc_D[i] / data_maximp->efc_D[i], 1, MjTol(1e-12, 1e-6))
          << "row " << i;
    }

    if (variant == 0) {
      // the sphere rests rigidly and quietly
      EXPECT_NEAR(data->qpos[2], 0.05, MjTol(1e-5, 1e-4));
      EXPECT_LT(mju_norm(data->qvel, 6), MjTol(1e-6, 1e-3));
    } else if (variant == 1) {
      // the pile survives: bodies bounded, nothing launched
      for (int b = 0; b < 3; b++) {
        EXPECT_GT(data->qpos[7 * b + 2], 0.0) << "body " << b;
        EXPECT_LT(data->qpos[7 * b + 2], 0.5) << "body " << b;
      }
      EXPECT_LT(mju_norm(data->qvel, model->nv), 50);
    } else {
      // the stack stands still: boxes where they started, no settled motion
      mjtNum z0[3] = {0.05, 0.14, 0.21};
      for (int b = 0; b < 3; b++) {
        EXPECT_NEAR(data->qpos[7 * (b + 1) + 2], z0[b], MjTol(1e-6, 1e-5))
            << "box " << b;
      }
      EXPECT_LT(maxspeed, MjTol(1e-10, 1e-4));
    }
  }
}

// refsafe under the discrete integrator: a contact or limit row stiffer than
// the step can resolve (h^2*K*I > 1) would rebound on impact, so it is replaced
// by the resolved row: zero restitution (h^2*K*I = 1) at the authored damping
// ratio, i.e. row factor f = 2 + 2*dampratio/sqrt(I). Rows with
// timeconst*dampratio >= h and equality rows are untouched; disabling refsafe
// restores the authored spring-damper and its rebound
TEST_F(ForwardTest, DiscreteRefsafeResolvedRow) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" gravity="0 0 0">
      <flag refsafe="%s"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .05">
        <freejoint/>
        <geom name="stiff" type="sphere" size=".05" mass="1" solref="0.004 1" priority="1"/>
      </body>
      <body pos="1 0 .0499">
        <freejoint/>
        <geom name="soft" type="sphere" size=".05" mass="1" priority="1"/>
      </body>
      <body pos="2 0 1">
        <joint name="hinge" axis="0 1 0" range="-1 0" solreflimit="0.004 2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 0 0 -.3" mass="1"/>
      </body>
      <body pos="3 0 1">
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="slide" solref="0.004 1"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  char xml[4096];
  const mjtNum v0 = 1;
  std::vector<mjtNum> kbip_authored;

  // authored rows first (refsafe disabled), then the resolved rows
  for (int enabled = 0; enabled < 2; enabled++) {
    std::snprintf(xml, sizeof(xml), xml_template,
                  enabled ? "enable" : "disable");
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    mjtNum h = model->opt.timestep;
    int stiff = mj_name2id(model.get(), mjOBJ_GEOM, "stiff");
    int hinge = mj_name2id(model.get(), mjOBJ_JOINT, "hinge");
    int slide = mj_name2id(model.get(), mjOBJ_JOINT, "slide");

    // stiff sphere: one step of penetration at speed v0; hinge pushed into its
    // limit; slide displaced from its equality
    data->qpos[2] = 0.05 - v0 * h;
    data->qvel[2] = -v0;
    data->qpos[model->jnt_qposadr[hinge]] = 0.01;
    data->qvel[model->jnt_dofadr[hinge]] = 1;
    data->qpos[model->jnt_qposadr[slide]] = 0.01;
    mj_forward(model.get(), data.get());
    ASSERT_EQ(data->ncon, 2);
    if (!enabled) {
      kbip_authored.assign(data->efc_KBIP, data->efc_KBIP + 4 * data->nefc);
      continue;
    }
    ASSERT_EQ(4 * data->nefc, kbip_authored.size());

    int nresolved = 0;
    for (int i = 0; i < data->nefc; i++) {
      const mjtNum* kbip = data->efc_KBIP + 4 * i;
      int type = data->efc_type[i];
      bool stiff_contact = (type == mjCNSTR_CONTACT_PYRAMIDAL) &&
                           (data->contact[data->efc_id[i]].geom[1] == stiff);
      if (stiff_contact || type == mjCNSTR_LIMIT_JOINT) {
        // resolved: zero restitution at the authored damping ratio (1 for the
        // contact, 2 for the limit)
        mjtNum zeta = stiff_contact ? 1 : 2;
        EXPECT_NEAR(h * h * kbip[0] * kbip[2], 1, MjTol(1e-12, 1e-6))
            << "row " << i;
        EXPECT_NEAR(1 + h * kbip[1] + h * h * kbip[0] * kbip[2],
                    2 + 2 * zeta / mju_sqrt(kbip[2]), MjTol(1e-10, 1e-5))
            << "row " << i;
        nresolved++;
      } else {
        // soft contact (timeconst*dampratio > h) and equality: as authored
        for (int k = 0; k < 4; k++) {
          EXPECT_EQ(kbip[k], kbip_authored[4 * i + k]) << "row " << i;
        }
      }
    }
    EXPECT_EQ(nresolved, 5);  // 4 pyramidal rows + 1 limit
  }

  // impact: authored row rebounds, resolved row does not
  for (int enabled = 0; enabled < 2; enabled++) {
    std::snprintf(xml, sizeof(xml), xml_template,
                  enabled ? "enable" : "disable");
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    mjtNum h = model->opt.timestep;
    data->qpos[2] = 0.05 - v0 * h;
    data->qvel[2] = -v0;
    mj_forward(model.get(), data.get());

    // one-step map of the normal row: v+ = (1-W) v + W (v - h K I pos)/f
    int row = data->contact[0].efc_address;
    const mjtNum* kbip = data->efc_KBIP + 4 * row;
    mjtNum f = 1 + h * kbip[1] + h * h * kbip[0] * kbip[2];
    mjtNum W = kbip[2] * f / (kbip[2] * f + 1 - kbip[2]);
    mjtNum pos = data->efc_pos[row];
    mjtNum v_expected =
        (1 - W) * (-v0) + W * (-v0 - h * kbip[0] * kbip[2] * pos) / f;
    mj_step(model.get(), data.get());
    EXPECT_NEAR(data->qvel[2], v_expected, MjTol(1e-9, 1e-4))
        << "refsafe " << enabled;
    if (enabled) {
      EXPECT_LT(mju_abs(data->qvel[2]), 0.05 * v0);  // no rebound
    } else {
      EXPECT_GT(data->qvel[2], 0.3 * v0);  // rebound
    }
  }
}

// implicit equality rows: a stiff connect (timeconst = 0.4 h) is stable and
// closes the loop tighter than the refsafe-clamped classic row; the folded
// Jdot*v correction keeps the loop stable at 0.6 rad of rotation per step
TEST_F(ForwardTest, DiscreteImplicitRowEquality) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete"/>
    <worldbody>
      <body name="crank" pos="0 0 1">
        <joint name="j1" axis="0 1 0" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.015" density="2000"/>
        <body name="coupler" pos="0.1 0 0">
          <joint name="j2" axis="0 1 0" damping="0.01"/>
          <geom type="capsule" fromto="0 0 0 0.25 0 -0.05" size="0.012" density="2000"/>
          <body name="rocker" pos="0.25 0 -0.05">
            <joint name="j3" axis="0 1 0" damping="0.01"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.15" size="0.012" density="2000"/>
            <site name="tip" pos="0 0 -0.15"/>
          </body>
        </body>
      </body>
      <site name="anchor" pos="0.35 0 0.8"/>
    </worldbody>
    <equality>
      <connect site1="tip" site2="anchor" solref="0.004 1"/>
    </equality>
    <keyframe>
      <key name="spin" qvel="10 0 0"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // mean loop-closure violation over 2 seconds from the spinning keyframe
  auto violation = [&](mjtIntegrator integrator, mjtNum h, mjtNum* maxv) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetDataKeyframe(model.get(), data.get(), 0);
    int nstep = (int)mju_round(2.0 / h);
    mjtNum sum = 0;
    *maxv = 0;
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
      sum += mju_norm(data->efc_pos, data->ne);
      for (int j = 0; j < 3; j++) {
        *maxv = mju_max(*maxv, mju_abs(data->qvel[j]));
      }
    }
    return sum / nstep;
  };

  // at h = 10 ms (timeconst = 0.4 h) the implicit row is stable and tighter
  // than the classic row, which refsafe softens to timeconst = 2 h
  mjtNum maxv_discrete, maxv_classic;
  mjtNum viol_discrete = violation(mjINT_DISCRETE, 0.01, &maxv_discrete);
  mjtNum viol_classic = violation(mjINT_IMPLICITFAST, 0.01, &maxv_classic);
  EXPECT_LT(maxv_discrete, 100);
  EXPECT_LT(viol_discrete, 0.5 * viol_classic);

  // 0.6 rad per step: stable with the Jdot*v correction folded into the
  // reference
  mjtNum viol_coarse = violation(mjINT_DISCRETE, 0.06, &maxv_discrete);
  EXPECT_LT(maxv_discrete, 100);
  EXPECT_LT(viol_coarse, 0.05);
}

// implicit rows with the noslip post-pass: a stiff elliptic contact sliding to
// rest under noslip stays bounded (the cone relations are derived from
// implicit R)
TEST_F(ForwardTest, DiscreteImplicitRowNoslip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.016" integrator="discrete" cone="elliptic" impratio="10"
            noslip_iterations="2">
      <flag refsafe="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1" solref="0.002 1"/>
      <body pos="0 0 0.05">
        <freejoint/>
        <geom size=".05" mass="1" solref="0.002 1" condim="6"
          friction="1 0.05 0.05"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="slide" qvel="2 0 0 0 0 5"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_resetDataKeyframe(model.get(), data.get(), 0);
  mjtNum max_qvel = 0;
  for (int i = 0; i < 200; i++) {
    mj_step(model.get(), data.get());
    if (i >= 190) {
      max_qvel = mju_max(max_qvel, mju_norm(data->qvel, 6));
    }
  }
  int nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_EQ(nwarning, 0);
  EXPECT_LT(max_qvel, MjTol(3e-9, 1e-6));
  EXPECT_NEAR(data->qpos[2], 0.05, MjTol(5e-5, 5e-5));
}

// stiff ">" leg pressed onto the floor: joint stiffness far beyond the explicit
// stability limit interacting with a contact. discrete converges first-order to
// the fine-timestep limit; implicitfast diverges at h*omega = 3.6; at the
// settled equilibrium the active set does not flicker
TEST_F(ForwardTest, DiscreteStiffLegPress) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete"/>
    <worldbody>
      <geom name="floor" type="plane" size="2 2 .1"/>
      <body name="upper" pos="0 0 0.5532">
        <joint name="slide" type="slide" axis="0 0 1" stiffness="10000" springref="-0.05" damping="50"/>
        <joint name="hip" type="hinge" axis="0 -1 0" stiffness="4000" damping="12"/>
        <geom name="upper" type="capsule" size="0.04" fromto="0 0 0 0.3064 0 -0.2571"/>
        <body name="lower" pos="0.3064 0 -0.2571">
          <joint name="knee" type="hinge" axis="0 -1 0" stiffness="4000" damping="12"/>
          <geom name="lower" type="capsule" size="0.04" fromto="0 0 0 -0.3064 0 -0.2571"/>
        </body>
      </body>
    </worldbody>
    <keyframe>
      <key name="impact" qvel="-2 0 0"/>
    </keyframe>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // step from the impact keyframe to t = 0.8, return the endpoint state
  auto endpoint = [&](mjtIntegrator integrator, mjtNum h, mjtNum out[6]) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetDataKeyframe(model.get(), data.get(), 0);
    int nstep = (int)mju_round(0.8 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    mju_copy(out, data->qpos, 3);
    mju_copy(out + 3, data->qvel, 3);
  };
  auto err = [](const mjtNum a[6], const mjtNum b[6]) {
    mjtNum e = 0;
    for (int i = 0; i < 6; i++) {
      e += mju_abs(a[i] - b[i]);
    }
    return e;
  };

  // discrete converges to the fine-timestep implicitfast reference, at first
  // order
  mjtNum ref[6], coarse[6], fine[6];
  endpoint(mjINT_IMPLICITFAST, 1e-4, ref);
  endpoint(mjINT_DISCRETE, 0.01, coarse);
  endpoint(mjINT_DISCRETE, 0.005, fine);
  EXPECT_LT(err(coarse, ref), MjTol(0.07, 0.07));
  EXPECT_LT(err(fine, ref), 0.7 * err(coarse, ref));

  // at h = 0.02 the joint springs are beyond the explicit limit (h*omega
  // = 3.6): implicitfast diverges, discrete settles quietly
  mjtNum big[6];
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings("The simulation is unstable");
  endpoint(mjINT_IMPLICITFAST, 0.02, big);
  int nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_GT(nwarning, 0);
  testing::Mock::VerifyAndClearExpectations(&warning_handler);
  endpoint(mjINT_DISCRETE, 0.02, big);
  nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_EQ(nwarning, 0);

  // zero-flicker: at the settled equilibrium, contact count, constraint states
  // and force signs are constant across steps
  endpoint(mjINT_DISCRETE, 0.01, coarse);
  mjtNum tiny = MjTol(1e-10, 1e-4);
  auto force_sign = [&](mjtNum f) {
    return f > tiny ? 1 : (f < -tiny ? -1 : 0);
  };
  int ncon = data->ncon, nefc = data->nefc;
  std::vector<int> state(data->efc_state, data->efc_state + nefc);
  std::vector<int> sign(nefc);
  for (int i = 0; i < nefc; i++) {
    sign[i] = force_sign(data->efc_force[i]);
  }
  for (int k = 0; k < 50; k++) {
    mj_step(model.get(), data.get());
    ASSERT_EQ(data->ncon, ncon);
    ASSERT_EQ(data->nefc, nefc);
    for (int i = 0; i < nefc; i++) {
      EXPECT_EQ(data->efc_state[i], state[i]) << "row " << i << " step " << k;
      EXPECT_EQ(force_sign(data->efc_force[i]), sign[i])
          << "row " << i << " step " << k;
    }
  }
}

// under discrete, the dual and noslip AR pattern is laid down at the position
// stage while the numeric phases are deferred to the actuation stage, so the Y
// column indices must be filled by the symbolic phase. The sparse-Jacobian path
// read uninitialized indices before the fix; sparse and dense must agree
TEST_F(ForwardTest, DiscreteSparseDualMatchesDense) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" cone="elliptic" jacobian="%s"
            noslip_iterations="2" solver="%s"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 0.3">
        <freejoint/>
        <geom type="box" size=".05 .07 .09" mass="1"/>
      </body>
      <body pos=".3 0 0.2">
        <freejoint/>
        <geom type="box" size=".06 .05 .04" mass="0.5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  char xml[2048];

  // Newton + noslip: run the sparse and dense Jacobian paths through impact
  std::vector<mjtNum> qpos[2];
  for (int s = 0; s < 2; s++) {
    std::snprintf(xml, sizeof(xml), xml_template, s ? "sparse" : "dense",
                  "Newton");
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    for (int i = 0; i < 100; i++) {
      mj_step(model.get(), data.get());
    }
    qpos[s] = std::vector<mjtNum>(data->qpos, data->qpos + model->nq);
    int nwarning = 0;
    for (int w = 0; w < mjNWARNING; w++) {
      nwarning += data->warning[w].number;
    }
    EXPECT_EQ(nwarning, 0);
  }
  EXPECT_THAT(qpos[1], Pointwise(MjNear(2e-8, 1e-3), qpos[0]));

  // PGS consumes the same symbolic AR: the sparse path steps cleanly
  std::snprintf(xml, sizeof(xml), xml_template, "sparse", "PGS");
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }
  int nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_EQ(nwarning, 0);
  for (int i = 0; i < model->nv; i++) {
    EXPECT_TRUE(std::isfinite(data->qacc[i]));
  }
}

// muscle force-length stiffness enters the discrete metric: the actuator metric
// scalar matches a finite difference of the actuator force, clamped to the
// stabilizing sign
TEST_F(ForwardTest, DiscreteMuscleStiffnessInMetric) {
  // near-zero fpmax: essentially only the active FL curve, so the descending
  // limb is net destabilizing (the compiler requires fpmax strictly positive)
  static constexpr char active_xml[] = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <muscle name="active" joint="slide" ctrlrange="0 1" fpmax="0.001"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(active_xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjtNum h = model->opt.timestep;

  // actuator force at (q, v, act); qpos maps to actuator length with unit gear
  auto force = [&](mjtNum q, mjtNum v, mjtNum act) {
    data->qpos[0] = q;
    data->qvel[0] = v;
    data->act[0] = act;
    mj_forward(model.get(), data.get());
    return data->actuator_force[0];
  };

  // centered finite difference of the force w.r.t. length (curves are piecewise
  // quadratic, so the FD is exact up to roundoff)
  mjtNum eps = MjEps(1e-6, 1e-3);
  auto fd_deriv = [&](mjtNum q, mjtNum v, mjtNum act) {
    return (force(q + eps, v, act) - force(q - eps, v, act)) / (2 * eps);
  };

  // the joint range maps to normalized length [0.75, 1.05]: check the metric
  // scalar against the FD on the ascending limb interior, at the curve knots,
  // and at nonzero velocities (FV scaling)
  mjtNum tol = MjTol(5e-9, 1e-3);
  mjtNum points[][3] = {{-0.8, 0, 0.7},    {-0.545, 0, 0.7}, {-0.3, 0, 0.7},
                        {-0.0909, 0, 0.7}, {0.2, 0, 0.7},    {-0.3, -0.5, 0.7},
                        {-0.3, 0.5, 0.7}};
  for (auto& p : points) {
    mjtNum expected = h * mju_max(0, -fd_deriv(p[0], p[1], p[2]));
    force(p[0], p[1], p[2]);
    ASSERT_EQ(data->nefmA, 1);
    EXPECT_NEAR(data->efm_ak[0], expected, tol)
        << "q " << p[0] << " v " << p[1];
  }

  // descending limb with no passive term: the slope is destabilizing (FD is
  // positive) and the stable-sign clamp zeroes the metric term
  EXPECT_GT(fd_deriv(0.9, 0, 0.7), 1);
  force(0.9, 0, 0.7);
  ASSERT_EQ(data->nefmA, 1);
  EXPECT_EQ(data->efm_ak[0], 0);

  // default muscle, zero activation: beyond L=1 the passive FP curve alone
  // contributes
  static constexpr char passive_xml[] = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <muscle name="passive" joint="slide" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  model = LoadModelFromString(passive_xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  data = MakeData(model);
  mjtNum expected = h * mju_max(0, -fd_deriv(0.9, 0, 0));
  EXPECT_GT(expected, 0);
  force(0.9, 0, 0);
  ASSERT_EQ(data->nefmA, 1);
  EXPECT_NEAR(data->efm_ak[0], expected, tol);
}

// the DC-motor controller position loop enters the discrete metric: kp exactly
// when the current is stateless, scaled by the one-step filter factor when
// stateful
TEST_F(ForwardTest, DiscreteDCMotorStiffnessInMetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="slide" motorconst="0.05" resistance="2.0" input="pos" controller="10 0 0"/>
      <dcmotor joint="slide" motorconst="0.05" resistance="2.0" input="pos" controller="10 0 0"
               inductance="0.002"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjtNum h = model->opt.timestep;
  mjtNum kp = 10;

  data->qpos[0] = 0.3;
  data->ctrl[0] = data->ctrl[1] = 0.5;
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->nefmA, 2);

  // stateless: df/dl = -kp exactly, the motor parameters cancel; a
  // setpoint-only controller has no net velocity term (back-EMF compensation is
  // exact), so the combined scalar is the pure stiffness term
  mjtNum tol = MjTol(1e-12, 1e-7);
  EXPECT_NEAR(data->efm_ak[0], h * kp, tol);
  EXPECT_NEAR(data->efm_as[0], h * h * kp, tol);

  // stateful current: the same one-step filter factor as the velocity
  // derivative
  mjtNum te = model->actuator_dynprm[mjNDYN];
  ASSERT_GT(te, 0);
  mjtNum s = 1 - mju_exp(-h / te);
  EXPECT_NEAR(data->efm_ak[1], h * kp * s, tol);
}

// a stateless setpoint dcmotor is the same servo as <pid>: under discrete both
// receive the same metric stiffness, and their trajectories match at a timestep
// far beyond the explicit stability limit of the shared kp
TEST_F(ForwardTest, DiscreteDCMotorMatchesPid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete">
      <flag contact="disable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="slide1" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="slide2" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <pid name="pid" joint="slide1" kp="90000" kv="600"/>
      <dcmotor name="dcmotor" joint="slide2" motorconst="0.05" resistance="2.0"
               input="pos vel" controller="90000 0 600"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nu, 4);
  MjDataPtr data = MakeData(model);

  // h*omega = 3: beyond the explicit limit, the trajectories are stable only
  // because kp is in the metric -- for the dcmotor, through the controller path
  while (data->time < 1.0) {
    mjtNum qref = mju_sin(5 * data->time);
    mjtNum vref = mju_cos(3 * data->time);
    data->ctrl[0] = data->ctrl[2] = qref;
    data->ctrl[1] = data->ctrl[3] = vref;
    mj_step(model.get(), data.get());
    ASSERT_NEAR(data->qpos[0], data->qpos[1], MjTol(1e-12, 1e-4));
    ASSERT_NEAR(data->qvel[0], data->qvel[1], MjTol(1e-10, 1e-2));
  }

  // the servos track: the comparison is not between two frozen joints
  EXPECT_GT(mju_abs(data->qpos[0]), 0.1);
}

// passive muscle stiffness far beyond the explicit stability limit:
// implicitfast diverges (the passive curve has no matching muscle velocity
// damping), discrete settles quietly onto the statics solution
TEST_F(ForwardTest, DiscreteStiffMuscleHang) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="0 0 1" range="-0.1 0.1" damping="1"/>
        <geom size=".05" mass="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <muscle joint="slide" gear="-1" force="2e5" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // hang for 2 seconds, return the endpoint and the peak tail speed
  auto hang = [&](mjtIntegrator integrator, mjtNum h, mjtNum* maxv_tail) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    int nstep = (int)mju_round(2.0 / h);
    *maxv_tail = 0;
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
      if (i >= nstep - 50) {
        *maxv_tail = mju_max(*maxv_tail, mju_abs(data->qvel[0]));
      }
    }
    return data->qpos[0];
  };

  // the mass stretches the muscle onto the passive curve; the equilibrium is a
  // statics problem, so discrete lands on the fine-timestep reference at any
  // timestep
  mjtNum maxv;
  mjtNum ref = hang(mjINT_IMPLICITFAST, 1e-4, &maxv);
  for (double h : {0.01, 0.02}) {
    mjtNum q = hang(mjINT_DISCRETE, h, &maxv);
    EXPECT_NEAR(q, ref, MjTol(1e-5, 1e-4));
    EXPECT_LT(maxv, MjTol(1e-6, 1e-4));
  }

  // h*omega = 2.4: implicitfast survives but rings hard on the explicit
  // stiffness
  hang(mjINT_IMPLICITFAST, 0.01, &maxv);
  EXPECT_GT(maxv, 0.05);

  // h*omega = 4.8: implicitfast diverges
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings("The simulation is unstable");
  hang(mjINT_IMPLICITFAST, 0.02, &maxv);
  int nwarning = 0;
  for (int w = 0; w < mjNWARNING; w++) {
    nwarning += data->warning[w].number;
  }
  EXPECT_GT(nwarning, 0);
  testing::Mock::VerifyAndClearExpectations(&warning_handler);
}

// cross-tree tendon stiffness far beyond the explicit stability bound: stable
// and dissipating
TEST_F(ForwardTest, DiscreteTendonStiffStable) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" gravity="0 0 0"/>
    <worldbody>
      <body pos="-.2 0 0"><joint name="s1" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body pos=".2 0 0"><joint name="s2" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
    </worldbody>
    <tendon><spatial stiffness="1e6" springlength="0.3"><site site="a"/><site site="b"/></spatial></tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // omega*h = h*sqrt(2k/m) = 14: far beyond the explicit bound of 2
  mj_resetData(model.get(), data.get());
  for (int i = 0; i < 500; i++) {
    mj_step(model.get(), data.get());
    ASSERT_LT(mju_abs(data->qpos[0]), 0.2) << "unstable at step " << i;
  }

  // backward-Euler dissipation has damped the oscillation into the deadband
  EXPECT_LT(mju_abs(data->qvel[0]) + mju_abs(data->qvel[1]), 0.1);
}

// single-chain fixed-tendon damping: discrete matches implicitfast (which
// carries this coupling in qDeriv since both dofs are ancestor-related)
TEST_F(ForwardTest, DiscreteTendonDampingMatchesImplicitfast) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="h1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .4 0 0" mass="1"/>
        <body pos=".4 0 0">
          <joint name="h2" type="hinge" axis="0 1 0"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass=".5"/>
        </body>
      </body>
    </worldbody>
    <tendon><fixed damping="4"><joint joint="h1" coef="1"/><joint joint="h2" coef="-.7"/></fixed></tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;
  const mjtNum qvel0[2] = {1.5, -2};

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    mju_copy(data->qvel, qvel0, nv);
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);

  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// islands and the monolithic solve agree under the discrete metric, including a
// cross-tree tendon whose trees the union-find merges into one island
TEST_F(ForwardTest, DiscreteIslandsMatchMonolithic) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="-.2 0 .04"><joint name="s1" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body pos=".2 0 .04"><joint name="s2" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
      <body pos="0 1 .04"><joint name="far" type="slide" axis="1 0 0" damping="2" stiffness="50"/><geom type="box" size=".05 .05 .05" mass="1"/></body>
    </worldbody>
    <tendon><spatial stiffness="2e3" springlength="0.3" damping="8"><site site="a"/><site site="b"/></spatial></tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    model->opt.solver = solver;

    // develop a nontrivial contact-rich state with islands enabled
    model->opt.disableflags &= ~mjDSBL_ISLAND;
    mj_resetData(model.get(), data.get());
    data->qvel[0] = 0.5;
    data->qvel[2] = -0.5;
    for (int i = 0; i < 20; i++) {
      mj_step(model.get(), data.get());
    }
    mj_forward(model.get(), data.get());
    ASSERT_GT(data->nisland, 1) << "expected multiple islands";
    std::vector<mjtNum> qacc_island(data->qacc, data->qacc + nv);

    // same state, monolithic
    model->opt.disableflags |= mjDSBL_ISLAND;
    mj_forward(model.get(), data.get());

    mjtNum scale = mju_norm(qacc_island.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    for (int i = 0; i < nv; i++) {
      EXPECT_THAT(data->qacc[i],
                  MjNear(qacc_island[i], 1e-7 * scale, 1e-4 * scale))
          << "solver " << solver << " dof " << i;
    }
    model->opt.disableflags &= ~mjDSBL_ISLAND;
  }
}

// a stiff tendon routed through three trees: island discovery unions the full
// ten_J footprint (not just two trees), and the island solve matches the
// monolithic solve
TEST_F(ForwardTest, DiscreteTendonThreeTreeIsland) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <geom type="plane" size="2 2 .1" condim="1"/>
      <body pos="0 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s1"/>
      </body>
      <body pos=".3 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s2"/>
      </body>
      <body pos=".6 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s3"/>
      </body>
    </worldbody>
    <tendon>
      <spatial stiffness="100" damping="20" springlength="0 0">
        <site site="s1"/>
        <site site="s2"/>
        <site site="s3"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->tendon_treenum[0], 3);
  MjDataPtr d1 = MakeData(model);
  MjDataPtr d2 = MakeData(model);
  mjModel* m = model.get();
  int nq = m->nq;

  // all three trees share one island through the tendon coupling
  mj_forward(m, d1.get());
  ASSERT_GT(d1->ncon, 0);
  EXPECT_EQ(d1->nisland, 1);

  // island and monolithic solves agree (islands only reorder the constraint
  // rows)
  mj_resetData(m, d1.get());
  for (int i = 0; i < 100; i++) {
    m->opt.disableflags &= ~mjDSBL_ISLAND;
    mj_step(m, d1.get());
    m->opt.disableflags |= mjDSBL_ISLAND;
    mj_step(m, d2.get());
  }
  m->opt.disableflags &= ~mjDSBL_ISLAND;
  EXPECT_THAT(AsVector(d1->qpos, nq),
              Pointwise(MjNear(1e-8, 1e-4), AsVector(d2->qpos, nq)));
}

// islands and the monolithic solve agree under the discrete metric with flexes
// (CSR-assembled, bending-only matrix-free, and trilinear interpolated), while
// an unconstrained flex stays out of islands
TEST_F(ForwardTest, DiscreteFlexIslandsMatchMonolithic) {
  static const char* const kXmls[] = {
      // CSR-assembled 2D cloth in contact + separate rigid box + free-falling
      // elastic flex
      R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                radius=".01" dim="2" mass=".2" pos="0 0 .008" dof="full">
        <contact selfcollide="none"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
      </flexcomp>
      <body pos="1 0 .04">
        <joint name="far" type="slide" axis="1 0 0" damping="2" stiffness="50"/>
        <geom type="box" size=".05 .05 .05" mass="1"/>
      </body>
      <flexcomp name="free_flex" type="grid" count="2 2 1" spacing=".05 .05 .05"
                radius=".005" dim="2" mass=".1" pos="-1 0 1" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )",
      // bending-only 2D cloth (matrix-free under CG, assembled under Newton) +
      // trilinear 3D solid in separate contact islands
      R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="bend_cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                radius=".01" dim="2" mass=".2" pos="0 0 .008" dof="full">
        <contact selfcollide="none"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="bend" thickness="0.01"/>
      </flexcomp>
      <flexcomp name="solid" type="grid" count="2 2 2" spacing=".06 .06 .06"
                radius=".01" dim="3" mass=".3" pos="1 0 .035" dof="trilinear">
        <contact selfcollide="none"/>
        <elasticity young="2e3" poisson="0.2" damping="0.1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )",
  };

  for (const char* xml : kXmls) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    MjDataPtr data = MakeData(model);
    int nv = model->nv;

    for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
      model->opt.solver = solver;
      model->opt.disableflags &= ~mjDSBL_ISLAND;
      mj_resetData(model.get(), data.get());
      for (int i = 0; i < 15; i++) {
        mj_step(model.get(), data.get());
      }
      mj_forward(model.get(), data.get());
      ASSERT_GE(data->nisland, 2);
      std::vector<mjtNum> qacc_island(data->qacc, data->qacc + nv);

      model->opt.disableflags |= mjDSBL_ISLAND;
      mj_forward(model.get(), data.get());

      mjtNum scale =
          mju_norm(qacc_island.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
      for (int i = 0; i < nv; i++) {
        EXPECT_THAT(data->qacc[i],
                    MjNear(qacc_island[i], 1e-7 * scale, 1e-4 * scale))
            << "solver " << solver << " dof " << i;
      }
      model->opt.disableflags &= ~mjDSBL_ISLAND;
    }
  }
}

// sleep composes with the discrete metric: unrelated trees sleep while the
// metric-coupled tendon pair stays awake (the pre-existing tendon sleep policy)
// and keeps solving correctly
TEST_F(ForwardTest, DiscreteSleep) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" sleep_tolerance="0.01">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body name="b1" pos="-.2 0 .04"><joint name="s1" type="slide" axis="1 0 0" damping="1"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body name="b2" pos=".2 0 .04"><joint name="s2" type="slide" axis="1 0 0" damping="1"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
      <body name="lone" pos="0 1 .04"><joint name="s3" type="slide" axis="1 0 0" damping="1" stiffness="20"/><geom type="sphere" size=".05" mass="1"/></body>
    </worldbody>
    <tendon><spatial stiffness="100" springlength="0.3 0.5" damping="5"><site site="a"/><site site="b"/></spatial></tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int lone_tree =
      model->body_treeid[mj_name2id(model.get(), mjOBJ_BODY, "lone")];
  int b1_tree = model->body_treeid[mj_name2id(model.get(), mjOBJ_BODY, "b1")];
  int b2_tree = model->body_treeid[mj_name2id(model.get(), mjOBJ_BODY, "b2")];

  // settle: the lone tree sleeps; the tendon-coupled pair stays awake by policy
  mj_resetData(model.get(), data.get());
  int asleep_at = -1;
  for (int i = 0; i < 3000; i++) {
    mj_step(model.get(), data.get());
    if (!data->tree_awake[lone_tree]) {
      asleep_at = i;
      break;
    }
  }
  ASSERT_GE(asleep_at, 0) << "lone tree never slept";
  EXPECT_TRUE(data->tree_awake[b1_tree]);
  EXPECT_TRUE(data->tree_awake[b2_tree]);

  // the metric solve keeps running correctly with a sleeping tree present
  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }
  EXPECT_FALSE(data->tree_awake[lone_tree]);
  EXPECT_LT(mju_norm(data->qvel, model->nv), 1e-6);

  // waking the lone tree with an applied force works as usual
  data->xfrc_applied[6 * mj_name2id(model.get(), mjOBJ_BODY, "lone")] = 5;
  mj_step(model.get(), data.get());
  EXPECT_TRUE(data->tree_awake[lone_tree]);
}

// position servo with kp far beyond the explicit stability bound (the use case
// of issue #3443): stable and converging to the setpoint at a 2ms timestep
TEST_F(ForwardTest, DiscreteActuatorKpStable) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" gravity="0 0 -9.81"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
    <actuator><position name="servo" joint="hinge" kp="1e5" kv="200"/></actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // omega*h = h*sqrt(kp/I) = 2*sqrt(1e5/0.083)/1000 = 2.2: beyond the explicit
  // bound
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 0.5;
  for (int i = 0; i < 1000; i++) {
    mj_step(model.get(), data.get());
    ASSERT_LT(mju_abs(data->qpos[0]), 2.0) << "unstable at step " << i;
  }
  EXPECT_NEAR(data->qpos[0], 0.5, 1e-3);
  EXPECT_LT(mju_abs(data->qvel[0]), 0.01);
}

// velocity-gain actuator with no stiffness: discrete matches implicitfast,
// which carries the same kv in qDeriv
TEST_F(ForwardTest, DiscreteActuatorKvMatchesImplicitfast) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="1"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
    <actuator><velocity joint="hinge" kv="20"/></actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qvel[0] = 2;
    data->ctrl[0] = 0.7;
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);
  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// destabilizing (wrong-sign) position gain is clamped out of the metric: the
// force integrates explicitly under both integrators, and one step matches
// implicitfast
TEST_F(ForwardTest, DiscreteWrongSignGainClamped) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="2"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
    <actuator><general joint="hinge" gaintype="fixed" gainprm="1" biastype="affine" biasprm="0 50 0"/></actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qpos[0] = 0.3;
    data->qvel[0] = -1;
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);
  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// cross-tree refsite actuator: the union-find merges the two trees, islands
// match the monolithic solve
TEST_F(ForwardTest, DiscreteRefsiteIslandsMatchMonolithic) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="-.2 0 .04"><joint name="s1" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body pos=".2 0 .04"><joint name="s2" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
      <body pos="0 1 .04"><joint name="far" type="slide" axis="1 0 0" damping="2" stiffness="50"/><geom type="box" size=".05 .05 .05" mass="1"/></body>
    </worldbody>
    <actuator><position site="a" refsite="b" kp="500" kv="20"/></actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    model->opt.solver = solver;
    model->opt.disableflags &= ~mjDSBL_ISLAND;
    mj_resetData(model.get(), data.get());
    data->ctrl[0] = 0.1;
    data->qvel[2] = -0.5;
    for (int i = 0; i < 20; i++) {
      mj_step(model.get(), data.get());
    }
    mj_forward(model.get(), data.get());
    ASSERT_GT(data->nisland, 1) << "expected multiple islands";
    std::vector<mjtNum> qacc_island(data->qacc, data->qacc + nv);

    model->opt.disableflags |= mjDSBL_ISLAND;
    mj_forward(model.get(), data.get());

    mjtNum scale = mju_norm(qacc_island.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    for (int i = 0; i < nv; i++) {
      EXPECT_THAT(data->qacc[i],
                  MjNear(qacc_island[i], 1e-7 * scale, 1e-4 * scale))
          << "solver " << solver << " dof " << i;
    }
    model->opt.disableflags &= ~mjDSBL_ISLAND;
  }
}

// a position servo on a three-tree tendon, with the damper class disabled so
// only the actuator-transmission edge remains: island discovery still unions
// the full footprint
TEST_F(ForwardTest, DiscreteActuatorTendonThreeTreeIsland) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG" tolerance="1e-14">
      <flag damper="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1" condim="1"/>
      <body pos="0 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s1"/>
      </body>
      <body pos=".3 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s2"/>
      </body>
      <body pos=".6 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
        <site name="s3"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="t">
        <site site="s1"/>
        <site site="s2"/>
        <site site="s3"/>
      </spatial>
    </tendon>
    <actuator>
      <position tendon="t" kp="50"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(model);
  MjDataPtr d2 = MakeData(model);
  mjModel* m = model.get();
  int nq = m->nq;

  // the tendon itself is metric-inactive (damper disabled, no stiffness): the
  // actuator gain supplies the coupling and all three trees share one island
  ASSERT_FALSE(mj_effTendonPossible(m, 0));
  d1->ctrl[0] = 0.3;
  mj_forward(m, d1.get());
  ASSERT_GT(d1->ncon, 0);
  EXPECT_EQ(d1->nisland, 1);

  // island and monolithic solves agree
  mj_resetData(m, d1.get());
  d1->ctrl[0] = 0.3;
  d2->ctrl[0] = 0.3;
  for (int i = 0; i < 100; i++) {
    m->opt.disableflags &= ~mjDSBL_ISLAND;
    mj_step(m, d1.get());
    m->opt.disableflags |= mjDSBL_ISLAND;
    mj_step(m, d2.get());
  }
  m->opt.disableflags &= ~mjDSBL_ISLAND;
  EXPECT_THAT(AsVector(d1->qpos, nq),
              Pointwise(MjNear(1e-8, 1e-4), AsVector(d2->qpos, nq)));
}

// servo in contact: native discrete inverse dynamics recovers the actuator
// force
TEST_F(ForwardTest, DiscreteActuatorInverseConsistency) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .5 0 0" mass="1"/>
      </body>
    </worldbody>
    <actuator><position joint="hinge" kp="2e4" kv="50"/></actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  // press the arm into the floor with the servo (positive rotation about +y
  // swings down)
  mj_resetData(model.get(), data.get());
  data->ctrl[0] = 0.5;
  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }
  data->qvel[0] = 0.1;

  for (int diagexact = 0; diagexact < 2; diagexact++) {
    if (diagexact) {
      model->opt.enableflags |= mjENBL_DIAGEXACT;
    } else {
      model->opt.enableflags &= ~mjENBL_DIAGEXACT;
    }
    mj_forward(model.get(), data.get());
    ASSERT_GT(data->ncon, 0);

    // the inverse only multiplies by the metric: it factors the backbone only
    // when the exact constraint diagonal asks for it
    mjtNum factored = data->qHDiagInv[0];
    data->qHDiagInv[0] = -1;
    mj_inverse(model.get(), data.get());
    EXPECT_EQ(data->qHDiagInv[0], diagexact ? factored : -1);

    // qfrc_inverse is the force that must be applied: here, the actuator's
    mjtNum scale = mju_norm(data->qfrc_actuator, nv) +
                   mju_norm(data->qfrc_constraint, nv) +
                   mju_norm(data->qfrc_bias, nv);
    std::vector<mjtNum> diff(nv);
    mju_sub(diff.data(), data->qfrc_inverse, data->qfrc_actuator, nv);
    EXPECT_LT(mju_norm(diff.data(), nv), 1e-6 * scale)
        << "diagexact " << diagexact;
  }
}

// tendon spring-damper in contact: native discrete inverse dynamics recovers
// the (zero) applied force
TEST_F(ForwardTest, DiscreteTendonInverseConsistency) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="500"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="-.2 0 .04"><joint name="s1" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="a"/></body>
      <body pos=".2 0 .04"><joint name="s2" type="slide" axis="1 0 0"/><geom type="sphere" size=".05" mass="1"/><site name="b"/></body>
    </worldbody>
    <tendon><spatial stiffness="2e3" springlength="0.3" damping="8"><site site="a"/><site site="b"/></spatial></tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  mj_resetData(model.get(), data.get());
  for (int i = 0; i < 30; i++) {
    mj_step(model.get(), data.get());
  }
  data->qvel[0] = 0.2;

  mj_forward(model.get(), data.get());
  ASSERT_GT(data->ncon, 0);
  mj_inverse(model.get(), data.get());

  mjtNum scale = mju_norm(data->qfrc_passive, nv) +
                 mju_norm(data->qfrc_constraint, nv) +
                 mju_norm(data->qfrc_bias, nv);
  EXPECT_LT(mju_norm(data->qfrc_inverse, nv), 1e-6 * scale);
}

// under integrator="discrete" the constraint regularizer must live in the
// effective metric: for a single active row with constant impedance d, the
// realized constraint acceleration interpolates the metric-consistent free
// acceleration and aref with weight exactly d:  J*qacc == (1-d)*J*qacc_smooth +
// d*aref
TEST_F(ForwardTest, DiscreteImpedanceInEffectiveMetric) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" jacobian="dense"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="0 0 1" damping="100" limited="true" range="-1 0" solimplimit="0.9 0.9 0.001"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  for (mjtSolver solver : {mjSOL_NEWTON, mjSOL_CG}) {
    model->opt.solver = solver;
    mj_resetData(model.get(), data.get());
    data->qpos[0] = 0.05;  // beyond the upper limit: exactly one active row
    mj_forward(model.get(), data.get());
    ASSERT_EQ(data->nefc, 1);

    // implicit row: the weight is R/f with f = 1 + h*B + h^2*K*I, so the
    // mixing weight is d*f/(d*f + 1 - d)
    const mjtNum d_imp = 0.9;  // constant solimp
    const mjtNum* KBIP = data->efc_KBIP;
    mjtNum h = model->opt.timestep;
    mjtNum f = 1 + h * KBIP[1] + h * h * KBIP[0] * KBIP[2];
    mjtNum d_eff = d_imp * f / (d_imp * f + 1 - d_imp);
    mjtNum lhs = data->efc_J[0] * data->qacc[0];
    mjtNum rhs = (1 - d_eff) * data->efc_J[0] * data->qacc_smooth[0] +
                 d_eff * data->efc_aref[0];
    EXPECT_NEAR(lhs, rhs, MjTol(1e-6, 1e-4)) << "solver=" << solver;
  }

  // same identity with actuator gains providing the metric terms: rho = M/(M +
  // h*kv + h^2*kp)
  static constexpr char xml_servo[] = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" jacobian="dense"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="0 0 1" limited="true" range="-1 0" solimplimit="0.9 0.9 0.001"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="slide" kp="1000" kv="100"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model2 = LoadModelFromString(xml_servo, error, sizeof(error));
  ASSERT_THAT(model2.get(), NotNull()) << error;
  MjDataPtr data2 = MakeData(model2);
  for (mjtSolver solver : {mjSOL_NEWTON, mjSOL_CG}) {
    model2->opt.solver = solver;
    mj_resetData(model2.get(), data2.get());
    data2->qpos[0] = 0.05;
    mj_forward(model2.get(), data2.get());
    ASSERT_EQ(data2->nefc, 1);

    const mjtNum d_imp = 0.9;
    const mjtNum* KBIP = data2->efc_KBIP;
    mjtNum h = model2->opt.timestep;
    mjtNum f = 1 + h * KBIP[1] + h * h * KBIP[0] * KBIP[2];
    mjtNum d_eff = d_imp * f / (d_imp * f + 1 - d_imp);
    mjtNum lhs = data2->efc_J[0] * data2->qacc[0];
    mjtNum rhs = (1 - d_eff) * data2->efc_J[0] * data2->qacc_smooth[0] +
                 d_eff * data2->efc_aref[0];
    EXPECT_NEAR(lhs, rhs, MjTol(1e-6, 1e-4)) << "servo, solver=" << solver;
  }

  // unit-level: with DIAGEXACT the exact diagonal is diag(J*Mhat^-1*J'), here
  // 1/(M + h*b); efc_diagA stores it in the impedance relation R =
  // (1-d)/d*diagA, which on an implicit row carries the factor 1/f
  model->opt.solver = mjSOL_NEWTON;
  model->opt.enableflags |= mjENBL_DIAGEXACT;
  mj_resetData(model.get(), data.get());
  data->qpos[0] = 0.05;
  mj_forward(model.get(), data.get());
  {
    const mjtNum* KBIP = data->efc_KBIP;
    mjtNum h = model->opt.timestep;
    mjtNum f = 1 + h * KBIP[1] + h * h * KBIP[0] * KBIP[2];
    EXPECT_NEAR(data->efc_diagA[0] * f, 1.0 / (1.0 + 0.005 * 100),
                MjTol(1e-10, 1e-6));
  }
}

// with the box fluid model (pure drag, no lift terms) and no constraints,
// discrete and implicitfast compute the same symmetrized fluid derivative: one
// step must match tightly
TEST_F(ForwardTest, DiscreteFluidMatchesImplicitfast) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" solver="CG" tolerance="1e-14" density="500" viscosity="2" wind="1 0 0"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size=".2 .15 .01" pos=".2 0 0" mass=".5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qvel[0] = 3;
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);
  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// strong ellipsoid-model drag at a large timestep: the drag-only metric keeps
// the paddle stable and monotonically dissipates its kinetic energy
TEST_F(ForwardTest, DiscreteEllipsoidDragStable) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.02" integrator="discrete" solver="CG" gravity="0 0 0" density="5000" viscosity="5"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="box" size=".3 .2 .005" pos=".3 0 0" mass=".2" fluidshape="ellipsoid"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_resetData(model.get(), data.get());
  data->qvel[0] = 10;
  for (int i = 0; i < 200; i++) {
    mj_step(model.get(), data.get());
    ASSERT_LT(mju_abs(data->qvel[0]), 20) << "unstable at step " << i;
  }
  EXPECT_LT(mju_abs(data->qvel[0]), 1.0);
}

// PGS solves the same discrete dual problem: qacc must match CG on a model
// whose metric is fully carried by the backbone (diagonal classes + fluid)
// re-evaluating from the velocity stage on (mj_forwardSkip, finite-difference
// loops) must not allocate: the Y/AR patterns are laid down at the position
// stage by mj_projectConstraint, later stages only fill values
TEST_F(ForwardTest, DiscreteSkipEvaluationAllocationFree) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <geom type="plane" size="1 1 1"/>
      <body pos="0 0 .095">
        <joint type="slide" axis="0 0 1" damping="1"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // plain primal, primal with the exact diagonal, and the dual (Y and AR)
  for (int config = 0; config < 3; config++) {
    model->opt.solver = config == 2 ? mjSOL_PGS : mjSOL_NEWTON;
    if (config == 1) {
      model->opt.enableflags |= mjENBL_DIAGEXACT;
    } else {
      model->opt.enableflags &= ~mjENBL_DIAGEXACT;
    }
    mj_resetData(model.get(), data.get());
    mj_forward(model.get(), data.get());
    ASSERT_GT(data->nefc, 0);
    size_t parena = data->parena;
    for (int i = 0; i < 100; i++) {
      mj_forwardSkip(model.get(), data.get(), mjSTAGE_POS, 0);
    }
    EXPECT_EQ(data->parena, parena) << "config " << config;
  }
}

TEST_F(ForwardTest, DiscretePGSMatchesCG) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14" iterations="1000"
            density="1.2" viscosity="0.02"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.049">
        <joint name="slidex" type="slide" axis="1 0 0"/>
        <joint name="slidez" type="slide" axis="0 0 1" stiffness="500" damping="2" springref="-0.1"/>
        <geom type="sphere" size="0.05" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  // pressed sliding contact: the spring drives the sphere into the plane while
  // it slides
  auto press = [&]() {
    mj_resetData(model.get(), data.get());
    data->qvel[0] = 0.5;
    mj_forward(model.get(), data.get());
    ASSERT_GT(data->ncon, 0);
    // the contact must carry force: a separating contact would make this test
    // vacuous
    ASSERT_GT(mju_norm(data->efc_force, data->nefc), 1);
    // the fluid backbone blocks must be exercised
    ASSERT_NE(data->efm_fluid, nullptr);
  };
  press();
  std::vector<mjtNum> qacc_cg(data->qacc, data->qacc + nv);

  model->opt.solver = mjSOL_PGS;
  model->opt.iterations = 2000;
  press();

  mjtNum scale = mju_norm(qacc_cg.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(data->qacc[i], MjNear(qacc_cg[i], 1e-7 * scale, 1e-6 * scale))
        << "dof " << i;
  }

  // noslip post-processing runs and keeps qacc finite
  model->opt.solver = mjSOL_CG;
  model->opt.noslip_iterations = 10;
  press();
  for (int i = 0; i < nv; i++) {
    EXPECT_FALSE(mju_isBad(data->qacc[i])) << "dof " << i;
  }
}

// inverse dynamics under PGS: the coupling classes are excluded from the metric
// (mj_effCouplings) and every consumer sees the same backbone, so forward and
// inverse remain consistent
TEST_F(ForwardTest, DiscretePGSInverseConsistency) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="PGS" tolerance="1e-14" iterations="50000"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.049">
        <joint name="slidex" type="slide" axis="1 0 0"/>
        <joint name="slidez" type="slide" axis="0 0 1" stiffness="500" damping="2" springref="-0.1"/>
        <geom type="sphere" size="0.05" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_resetData(model.get(), data.get());
  data->qvel[0] = 0.5;
  mj_forward(model.get(), data.get());
  ASSERT_GT(data->ncon, 0);
  ASSERT_GT(mju_norm(data->efc_force, data->nefc), 1);
  mj_compareFwdInv(model.get(), data.get());

  // measured residual: 1.1e-6 double (the PGS convergence floor); slots ~1
  // click above
  mjtNum epsilon = MjTol(1e-5, 5e-4);
  EXPECT_LT(data->solver_fwdinv[0], epsilon);
  EXPECT_LT(data->solver_fwdinv[1], epsilon);
}

// standalone free bodies receive the local gyroscopic treatment under discrete
// (mj_discreteGyro): a tumbling body must match implicitfast, which reinstates
// the same derivative -- the two integrators coincide on this model by design
TEST_F(ForwardTest, DiscreteGyroMatchesImplicitfast) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" solver="CG" tolerance="1e-14" gravity="0 0 0"/>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".3 .1 .02" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qvel[3] = 1;  // tumbling spin about the unstable middle axis
    data->qvel[4] = 6;
    data->qvel[5] = 0.5;
    mj_step(model.get(), data.get());
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);
  for (int i = 0; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// the gyroscopic composition also applies on the constrained exit: a tumbling
// free body is unaffected by unrelated contacts elsewhere in the scene
TEST_F(ForwardTest, DiscreteGyroWithUnrelatedContacts) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" solver="CG" tolerance="1e-14" gravity="0 0 0"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="1 0 .049">
        <freejoint/>
        <geom type="sphere" size=".05" mass="1" condim="1"/>
      </body>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="box" size=".3 .1 .02" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv;

  auto step_with = [&](mjtIntegrator integrator, std::vector<mjtNum>& qvel) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qvel[9] = 1;  // tumbling spin about the unstable middle axis
    data->qvel[10] = 6;
    data->qvel[11] = 0.5;
    mj_step(model.get(), data.get());
    ASSERT_GT(data->ncon,
              0);  // the sphere's contacts keep the solve constrained
    qvel.assign(data->qvel, data->qvel + nv);
  };

  std::vector<mjtNum> qvel_fast, qvel_discrete;
  step_with(mjINT_IMPLICITFAST, qvel_fast);
  step_with(mjINT_DISCRETE, qvel_discrete);
  for (int i = 6; i < nv; i++) {
    EXPECT_THAT(qvel_discrete[i], MjNear(qvel_fast[i], 1e-12, 2e-6))
        << "dof " << i;
  }
}

// the diagonal-ratio impedance correction is exact for backbone classes but
// sees only the diagonal of the tendon block: with a cross-branch tendon the
// realized interpolation weight deviates from solimp's d by the
// analytically-known ratio rho = Ahat_true/Ahat_diag
TEST_F(ForwardTest, DiscreteImpedanceTendonResidual) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG" tolerance="1e-14" jacobian="dense" gravity="0 0 0"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="s1" type="slide" axis="0 0 1" limited="true" range="-1 0.02" solimplimit="0.9 0.9 0.001"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body pos="1 0 1">
        <joint name="s2" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="40000" springlength="0.0499 0.0499">
        <joint joint="s1" coef="1"/>
        <joint joint="s2" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_resetData(model.get(), data.get());
  data->qpos[0] = 0.05;  // beyond the upper limit: one active row on dof 0
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->nefc, 1);
  ASSERT_GT(mju_abs(data->efc_force[0]), 0);  // the row is active

  // measured interpolation weight
  mjtNum Ja = data->efc_J[0] * data->qacc[0] + data->efc_J[1] * data->qacc[1];
  mjtNum Ja0 = data->efc_J[0] * data->qacc_smooth[0] +
               data->efc_J[1] * data->qacc_smooth[1];
  mjtNum w = (Ja - Ja0) / (data->efc_aref[0] - Ja0);

  // analytic: Mhat = I + c*[1 1;1 1] with c = h^2*k; true Ahat_11 =
  // (1+c)/(1+2c), diagonal surrogate = 1/(1+c); rho = their ratio; w_pred =
  // rho*d/(rho*d + 1-d)
  mjtNum h = model->opt.timestep, c = h * h * 40000, d_imp = 0.9;
  mjtNum rho = (1 + c) * (1 + c) / (1 + 2 * c);

  // the implicit row's weight is R/f: d enters the mix as d*f
  const mjtNum* KBIP = data->efc_KBIP;
  mjtNum f = 1 + h * KBIP[1] + h * h * KBIP[0] * KBIP[2];
  mjtNum w_pred = rho * d_imp * f / (rho * d_imp * f + 1 - d_imp);
  EXPECT_THAT(w, MjNear(w_pred, 1e-14, 1e-6));

  // and the deviation from the diagonal-class weight is real: the pinned
  // deficit this test documents
  mjtNum w_diag = d_imp * f / (d_imp * f + 1 - d_imp);
  EXPECT_GT(mju_abs(w - w_diag), 0.01);
}

// a spring-and-damper tendon pressing a body into the floor.
// The constraint solve in the effective metric couples the contact to the
// damper; implicitfast computes contact forces against bare M, and its
// trajectory error grows with h*b/m while discrete's stays near the
// h->0 reference
TEST_F(ForwardTest, DiscreteContactCouplingTendon) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <geom type="plane" size="1 1 .1" condim="1"/>
      <body pos="0 0 0.099">
        <joint name="press" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1" condim="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="400" springlength="-0.05 -0.05" damping="300">
        <joint joint="press" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // simulate 0.2s from a perturbed contacting state, return endpoint (qpos,
  // qvel)
  auto endpoint = [&](mjtIntegrator integrator, mjtNum h, mjtNum out[2]) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    data->qvel[0] = -0.2;
    int nstep = (int)mju_round(0.2 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    ASSERT_GT(data->ncon, 0);  // the spring keeps the contact engaged
    out[0] = data->qpos[0];
    out[1] = data->qvel[0];
  };

  // h->0 reference: the integrators approach agreement
  mjtNum ref[2], ref2[2], disc[2], fast[2];
  endpoint(mjINT_IMPLICITFAST, 1e-4, ref);
  endpoint(mjINT_DISCRETE, 1e-4, ref2);
  mjtNum dref = mju_abs(ref2[0] - ref[0]) + mju_abs(ref2[1] - ref[1]);
  EXPECT_LT(dref, 1e-3);

  // large timestep, h*b/m = 3: implicitfast's error dominates both discrete's
  // error and the residual small-h disagreement
  endpoint(mjINT_DISCRETE, 0.01, disc);
  endpoint(mjINT_IMPLICITFAST, 0.01, fast);
  mjtNum err_disc = mju_abs(disc[0] - ref[0]) + mju_abs(disc[1] - ref[1]);
  mjtNum err_fast = mju_abs(fast[0] - ref[0]) + mju_abs(fast[1] - ref[1]);
  EXPECT_GT(err_fast, 5 * mju_max(err_disc, dref));
}

// design 5.6, the sensor contract: acceleration-stage sensors consume the
// step-map qacc through mj_rnePostConstraint --
// the accelerometer reads (v+ - v)/h plus gravity
TEST_F(ForwardTest, DiscreteAccelerometerStepMap) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.09">
        <joint name="fall" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="imu"/>
      </body>
    </worldbody>
    <sensor>
      <accelerometer site="imu"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // impact state: penetrating with downward velocity, one step arrests it
  mj_resetData(model.get(), data.get());
  data->qvel[0] = -2;
  mj_forward(model.get(), data.get());
  ASSERT_GT(data->ncon, 0);

  // site frame is world-aligned and motion is pure z:
  // proper acceleration = qacc - g
  mjtNum expected = data->qacc[0] - model->opt.gravity[2];
  EXPECT_THAT(data->sensordata[2], MjNear(expected, 1e-10, 1e-5));
}

// constraint-buffer exhaustion recovery must deactivate the metric: mj_clearEfc
// NULLs the efm arena pointers, so the activity flag and counts consumers gate
// on must clear with them
TEST_F(ForwardTest, DiscreteClearEfcDeactivatesMetric) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="s1" type="slide" axis="0 0 1" damping="1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
      <body pos=".2 0 1">
        <joint name="s2" type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".05" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="T" stiffness="10">
        <joint joint="s1" coef="1"/>
        <joint joint="s2" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->efm_active, 1);
  ASSERT_EQ(data->nefmT, 1);

  mj_clearEfc(data.get());
  EXPECT_EQ(data->efm_active, 0);
  EXPECT_EQ(data->nefmT, 0);
  EXPECT_EQ(data->nefmA, 0);
  EXPECT_EQ(data->nefmK, 0);
  EXPECT_EQ(data->nefmL, 0);
  EXPECT_EQ(data->nefmdof, 0);

  // the pipeline recovers on the next full forward
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->efm_active, 1);
  EXPECT_EQ(data->nefmT, 1);
}

// all single-step integrators share the continuous limit and are first-order
// accurate: halving the timestep halves the endpoint error against an RK4
// reference
TEST_F(ForwardTest, IntegratorConvergenceOrder) {
  static const char* const kXml = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 1">
        <joint name="shoulder" type="hinge" axis="0 1 0" damping="0.1"/>
        <geom type="capsule" size=".02" fromto="0 0 0 .3 0 0" mass="1"/>
        <body pos=".3 0 0">
          <joint name="elbow" type="hinge" axis="0 1 0" stiffness="5" damping="0.05"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .25 0 0" mass=".5"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // simulate 0.5s of passive swing from rest, return endpoint qpos
  auto endpoint = [&](mjtIntegrator integrator, mjtNum h, mjtNum out[2]) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    int nstep = (int)mju_round(0.5 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    out[0] = data->qpos[0];
    out[1] = data->qpos[1];
  };

  // reference: RK4 at small timestep
  mjtNum ref[2];
  endpoint(mjINT_RK4, 2.5e-4, ref);

  auto err = [&](const mjtNum e[2]) {
    return mju_abs(e[0] - ref[0]) + mju_abs(e[1] - ref[1]);
  };

  mjtNum err_min = 1;  // smallest first-order error, tracked for the RK4 check
  for (mjtIntegrator integrator :
       {mjINT_EULER, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    mjtNum e1[2], e2[2], e4[2];
    endpoint(integrator, 4e-3, e1);
    endpoint(integrator, 2e-3, e2);
    endpoint(integrator, 1e-3, e4);
    EXPECT_THAT(err(e1) / err(e2), AllOf(Gt(1.5), Lt(2.7))) << integrator;
    EXPECT_THAT(err(e2) / err(e4), AllOf(Gt(1.5), Lt(2.7))) << integrator;
    err_min = mju_min(err_min, err(e4));
  }

  // RK4 is higher-order: at the largest timestep it still beats them all
  mjtNum rk4[2];
  endpoint(mjINT_RK4, 4e-3, rk4);
  EXPECT_LT(err(rk4), 0.05 * err_min);
}

// the shared continuous limit holds through a force-carrying contact: a
// pendulum swings on a base that a spring presses onto the floor, shaking the
// contact inside the friction cone, and every first-order integrator converges
// to the same RK4 reference trajectory. Under discrete this pins consistency:
// the metric modifies the step map at O(h), not the limit. (Gross sliding is
// deliberately avoided: at coarse timesteps the coupled cone converts fast slip
// into normal-force boost and the contact chatters.)
TEST_F(ForwardTest, IntegratorConvergenceOrderContact) {
  static const char* const kXml = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 1" friction="0.2"/>
      <body pos="0 0 0.1">
        <joint name="lift" type="slide" axis="0 0 1" stiffness="50" springref="-0.1" damping="1"/>
        <joint name="slidex" type="slide" axis="1 0 0" stiffness="20" damping="0.5"/>
        <geom type="sphere" size="0.1" mass="1" friction="0.2"/>
        <body pos="0 0 0.1">
          <joint name="swing" type="hinge" axis="0 1 0" damping="0.02"/>
          <geom type="capsule" size=".02" fromto="0 0 0 .15 0 0" mass="0.3"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // simulate 0.5s of swinging on the pressed contact, return endpoint qpos
  auto endpoint = [&](mjtIntegrator integrator, mjtNum h, mjtNum out[2]) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    // start pressed into the cushion with the pendulum raised: the pendulum's
    // initial reaction unloads a surface-level start, separating the contact
    // for one step
    data->qpos[0] = -0.00048;
    data->qpos[2] = 1;
    int nstep = (int)mju_round(0.5 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    out[0] = data->qpos[0];
    out[1] = data->qpos[2];
  };

  // the contact carries force throughout: never airborne, pressed at the
  // endpoint
  {
    model->opt.integrator = mjINT_DISCRETE;
    model->opt.timestep = 4e-3;
    mj_resetData(model.get(), data.get());
    data->qpos[0] = -0.00048;
    data->qpos[2] = 1;
    for (int i = 0; i < 125; i++) {
      mj_step(model.get(), data.get());
      ASSERT_GT(data->ncon, 0);
    }
    mj_forward(model.get(), data.get());
    ASSERT_GT(mju_norm(data->efc_force, data->nefc), 1);
  }

  // reference: RK4 at small timestep
  mjtNum ref[2];
  endpoint(mjINT_RK4, 2.5e-4, ref);

  auto err = [&](const mjtNum e[2]) {
    return mju_abs(e[0] - ref[0]) + mju_abs(e[1] - ref[1]);
  };

  for (mjtIntegrator integrator :
       {mjINT_EULER, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    mjtNum e1[2], e2[2], e4[2];
    endpoint(integrator, 4e-3, e1);
    endpoint(integrator, 2e-3, e2);
    endpoint(integrator, 1e-3, e4);
    EXPECT_THAT(err(e1) / err(e2), AllOf(Gt(1.5), Lt(2.7))) << integrator;
    EXPECT_THAT(err(e2) / err(e4), AllOf(Gt(1.5), Lt(2.7))) << integrator;
  }
}

// RK4 is fourth-order accurate for ball and free joint orientations: the
// observed order log2(err(h) / err(h/2)) is ~4, i.e. halving the timestep
// reduces the orientation error ~16x (order ~2 without the RKMK correction).
// The free body has a rotational spring and the ball joint is a heavy top under
// gravity, so the torques depend on the stage orientations
TEST_F(ForwardTest, RungeKuttaOrientationConvergenceOrder) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "Fourth-order errors are below float32 roundoff";
#endif
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="RK4">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="free" type="free" stiffness="5"/>
        <inertial pos="0 0 0" mass="1" diaginertia=".1 .2 .3"/>
      </body>
      <body pos="1 0 0">
        <joint name="ball" type="ball"/>
        <inertial pos=".02 .01 .1" mass="1" diaginertia=".01 .01 .02"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // addresses of the joint quaternions and angular velocities
  const char* const kJoints[2] = {"free", "ball"};
  int qadr[2], vadr[2];
  for (int j = 0; j < 2; j++) {
    int id = mj_name2id(model.get(), mjOBJ_JOINT, kJoints[j]);
    ASSERT_GE(id, 0);
    int offset = model->jnt_type[id] == mjJNT_FREE ? 3 : 0;
    qadr[j] = model->jnt_qposadr[id] + offset;
    vadr[j] = model->jnt_dofadr[id] + offset;
  }

  // simulate 1s, return the free and ball joint quaternions
  auto endpoint = [&](mjtNum h, mjtNum quat[8]) {
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    const mjtNum kOmegaFree[3] = {1, 2, 0.5};
    mju_copy3(data->qvel + vadr[0], kOmegaFree);
    data->qvel[vadr[1] + 2] = 20;  // spin the top about its axis
    int nstep = (int)mju_round(1 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    for (int j = 0; j < 2; j++) {
      mju_copy4(quat + 4 * j, data->qpos + qadr[j]);
    }
  };

  // reference: timestep 4x smaller than the finest tested one, so its own error
  // is ~1/256 of the finest error
  mjtNum ref[8];
  endpoint(6.25e-4, ref);

  // orientation error of joint j: the angle (rad) of the rotation from the
  // reference orientation; at h = 1e-2 it is ~1e-7 (free) and ~3e-6 (ball)
  auto err = [&](const mjtNum quat[8], int j) {
    mjtNum dif[3];
    mju_subQuat(dif, quat + 4 * j, ref + 4 * j);
    return mju_norm3(dif);
  };

  // observed convergence order between timesteps h and h/2
  auto order = [](mjtNum err_h, mjtNum err_h2) {
    return std::log2(err_h / err_h2);
  };

  mjtNum q1[8], q2[8], q4[8];
  endpoint(1e-2, q1);
  endpoint(5e-3, q2);
  endpoint(2.5e-3, q4);
  for (int j = 0; j < 2; j++) {
    mjtNum e1 = err(q1, j), e2 = err(q2, j), e4 = err(q4, j);

    // errors are far above roundoff, so the ratios measure truncation error
    EXPECT_GT(e4, 1e-12) << kJoints[j];

    // the fast top (h*|omega| = 0.2 at h = 1e-2) is slightly pre-asymptotic:
    // its order approaches 4 from below as h decreases
    EXPECT_NEAR(order(e1, e2), 4, 0.25)
        << kJoints[j] << ", errors " << e1 << ", " << e2;
    EXPECT_NEAR(order(e2, e4), 4, 0.25)
        << kJoints[j] << ", errors " << e2 << ", " << e4;
  }
}

// a torque-free isotropic body spinning exactly n turns per step (n = 1, 2)
// must return to its initial orientation. RK4 stages 2 and 3 have rotation
// vectors of angle n*pi and stage 4 of angle 2*pi*n, where the Bortz
// coefficient of dexp^{-1} is singular (e.g. ~1e14 near 2*pi). Mathematically
// rot is parallel to the angular velocity, so rot x vel = 0, but rounding makes
// the cross product ~1e-13 and the singular coefficient amplifies it into an
// O(1) rad orientation error, unless the angle is reduced to (-pi, pi] first
TEST_F(ForwardTest, RungeKuttaFullTurnStage) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="RK4" timestep="0.01" gravity="0 0 0">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint name="free"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // addresses of the free joint quaternion and angular velocity
  int id = mj_name2id(model.get(), mjOBJ_JOINT, "free");
  ASSERT_GE(id, 0);
  int qadr = model->jnt_qposadr[id] + 3;
  int vadr = model->jnt_dofadr[id] + 3;

  // oblique axes, so rot and vel differ by rounding in all components
  const mjtNum kAxes[2][3] = {{1, 2, 3}, {-0.3, 0.7, 0.2}};
  for (int turns = 1; turns <= 2; turns++) {
    for (int a = 0; a < 2; a++) {
      // angular velocity: n turns per step about the axis
      mj_resetData(model.get(), data.get());
      mjtNum omega[3];
      mju_copy3(omega, kAxes[a]);
      mju_normalize3(omega);
      mju_scl3(omega, omega, 2 * mjPI * turns / model->opt.timestep);
      mju_copy3(data->qvel + vadr, omega);
      mjtNum quat0[4];
      mju_copy4(quat0, data->qpos + qadr);

      mj_step(model.get(), data.get());

      // precondition: torque-free, so the angular velocity is unchanged
      mjtNum dvel[3];
      mju_sub3(dvel, data->qvel + vadr, omega);
      EXPECT_LT(mju_norm3(dvel) / mju_norm3(omega), MjTol(4e-15, 2e-6))
          << "turns " << turns << ", axis " << a;

      // the body is back at its initial orientation (without the angle
      // reduction the error is ~1 rad)
      mjtNum dif[3];
      mju_subQuat(dif, data->qpos + qadr, quat0);
      EXPECT_LT(mju_norm3(dif), MjTol(2e-14, 1e-5))
          << "turns " << turns << ", axis " << a << ": angle error (rad)";
    }
  }
}

// on an undamped oscillator below the explicit stability bound, RK4 and the
// semi-implicit integrators conserve energy while discrete dissipates at the
// analytic backward-Euler rate: per-step energy factor 1/(1 + (h*omega)^2)
TEST_F(ForwardTest, DiscreteDissipationAnalytic) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0">
      <flag energy="enable"/>
    </option>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0" stiffness="100"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // h*omega = 0.1, well below the explicit stability bound h*omega = 2
  const int kNstep = 250;
  mjtNum homega2 =
      model->opt.timestep * model->opt.timestep * 100;  // (h*omega)^2

  // simulate N steps from unit displacement, return total energy ratio E_N /
  // E_0
  auto energy_ratio = [&](mjtIntegrator integrator) {
    model->opt.integrator = integrator;
    mj_resetData(model.get(), data.get());
    data->qpos[0] = 1;
    mj_forward(model.get(), data.get());
    mjtNum energy_0 = data->energy[0] + data->energy[1];
    for (int i = 0; i < kNstep; i++) {
      mj_step(model.get(), data.get());
    }
    mj_forward(model.get(), data.get());
    return (data->energy[0] + data->energy[1]) / energy_0;
  };

  // RK4 conserves; Euler and implicitfast are semi-implicit: |eigenvalue| = 1,
  // energy oscillates within an O(h*omega) band
  EXPECT_THAT(energy_ratio(mjINT_RK4), AllOf(Gt(0.999), Lt(1.001)));
  EXPECT_THAT(energy_ratio(mjINT_EULER), AllOf(Gt(0.9), Lt(1.1)));
  EXPECT_THAT(energy_ratio(mjINT_IMPLICITFAST), AllOf(Gt(0.9), Lt(1.1)));

  // discrete: the step map eigenvalues satisfy |lambda|^2 = 1/(1 + (h*omega)^2)
  mjtNum predicted = mju_pow(1 + homega2, -kNstep);
  mjtNum measured = energy_ratio(mjINT_DISCRETE);
  EXPECT_LT(measured, 0.15);  // the oscillation really has decayed (8x)
  EXPECT_THAT(measured / predicted, AllOf(Gt(0.8), Lt(1.25)));
}

// beyond the explicit stability bound (h*omega = 3 > 2) the
// explicit-on-stiffness integrators diverge, tripping the auto-reset warnings;
// discrete decays monotonically, and remains stable even at h*omega = 30
TEST_F(ForwardTest, IntegratorStabilityEnvelope) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="1 0 0" stiffness="9e4"/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // simulate 200 steps from displacement 0.5, return total warning count
  auto warnings = [&](mjtIntegrator integrator, mjtNum h) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    data->qpos[0] = 0.5;
    for (int i = 0; i < 200; i++) {
      mj_step(model.get(), data.get());
    }
    int nwarning = 0;
    for (int w = 0; w < mjNWARNING; w++) {
      nwarning += data->warning[w].number;
    }
    return nwarning;
  };

  // omega = 300: at h = 0.01 all explicit-on-stiffness integrators blow up
  MockWarningHandler warning_handler;
  for (mjtIntegrator integrator :
       {mjINT_EULER, mjINT_IMPLICITFAST, mjINT_RK4}) {
    warning_handler.ExpectWarnings("The simulation is unstable");
    EXPECT_GT(warnings(integrator, 0.01), 0) << integrator;
    testing::Mock::VerifyAndClearExpectations(&warning_handler);
  }

  // discrete is stable at h*omega = 3 and decays: |lambda| = 1/sqrt(10) per
  // step
  EXPECT_CALL(warning_handler, Warn(_)).Times(0);
  EXPECT_EQ(warnings(mjINT_DISCRETE, 0.01), 0);
  EXPECT_LT(mju_abs(data->qpos[0]), 1e-3);

  // and remains stable at h*omega = 30
  EXPECT_EQ(warnings(mjINT_DISCRETE, 0.1), 0);
  EXPECT_LT(mju_abs(data->qpos[0]), 1e-3);
}

// design 5.4 for the diagonal metric class: a joint damper coupled to a
// contact. under implicitfast the contact solve is blind to the damper (split
// scheme, error growing with h*b/m); under discrete they share the solve and
// the error stays flat
TEST_F(ForwardTest, DiscreteContactCouplingJointDamping) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete" solver="CG" tolerance="1e-14"/>
    <worldbody>
      <geom type="plane" size="1 1 .1" condim="1"/>
      <body pos="0 0 0.099">
        <joint name="press" type="slide" axis="0 0 1" stiffness="400" springref="-0.05"
               damping="300"/>
        <geom type="sphere" size="0.1" mass="1" condim="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // simulate 0.2s from a perturbed contacting state, return endpoint (qpos,
  // qvel)
  auto endpoint = [&](mjtIntegrator integrator, mjtNum h, mjtNum out[2]) {
    model->opt.integrator = integrator;
    model->opt.timestep = h;
    mj_resetData(model.get(), data.get());
    data->qvel[0] = -0.2;
    int nstep = (int)mju_round(0.2 / h);
    for (int i = 0; i < nstep; i++) {
      mj_step(model.get(), data.get());
    }
    ASSERT_GT(data->ncon, 0);  // the spring keeps the contact engaged
    out[0] = data->qpos[0];
    out[1] = data->qvel[0];
  };

  // h->0 reference: the integrators approach agreement
  mjtNum ref[2], ref2[2], disc[2], fast[2];
  endpoint(mjINT_IMPLICITFAST, 1e-4, ref);
  endpoint(mjINT_DISCRETE, 1e-4, ref2);
  mjtNum dref = mju_abs(ref2[0] - ref[0]) + mju_abs(ref2[1] - ref[1]);
  EXPECT_LT(dref, 1e-3);

  // large timestep, h*b/m = 3: implicitfast's error dominates both discrete's
  // error and the residual small-h disagreement
  endpoint(mjINT_DISCRETE, 0.01, disc);
  endpoint(mjINT_IMPLICITFAST, 0.01, fast);
  mjtNum err_disc = mju_abs(disc[0] - ref[0]) + mju_abs(disc[1] - ref[1]);
  mjtNum err_fast = mju_abs(fast[0] - ref[0]) + mju_abs(fast[1] - ref[1]);
  EXPECT_GT(err_fast, 5 * mju_max(err_disc, dref));
}

// joint and tendon limit velocity sensors read efc_vel at the velocity stage:
// under integrator=discrete, efc_vel must be computed before mj_sensorVel runs
TEST_F(ForwardTest, DiscreteLimitVelocitySensors) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <body pos="0 0 1">
        <joint name="joint" type="slide" axis="0 0 1" range="-0.5 0.5"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="a"/>
      </body>
      <body pos="0 0 3">
        <joint name="joint2" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="b"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="tendon" range="0 1.5">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>
    <sensor>
      <jointlimitvel joint="joint"/>
      <tendonlimitvel tendon="tendon"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // set position violating upper limits
  data->qpos[0] = 0.6;  // joint limit violated (0.6 > 0.5)
  data->qpos[1] = 0.5;  // tendon length = (3 + 0.5) - (1 + 0.6) = 1.9 > 1.5
  data->qvel[0] = 2.0;
  data->qvel[1] = 3.0;

  // reference values from implicitfast
  model->opt.integrator = mjINT_IMPLICITFAST;
  mj_forward(model.get(), data.get());
  ASSERT_GT(data->nefc, 0);
  mjtNum expected_joint_vel = data->sensordata[0];
  mjtNum expected_tendon_vel = data->sensordata[1];
  EXPECT_NE(expected_joint_vel, 0);
  EXPECT_NE(expected_tendon_vel, 0);

  // discrete integrator under mj_step1 on fresh data: sensors must be valid
  // before actuation/step2
  model->opt.integrator = mjINT_DISCRETE;
  MjDataPtr data_disc = MakeData(model);
  data_disc->qpos[0] = 0.6;
  data_disc->qpos[1] = 0.5;
  data_disc->qvel[0] = 2.0;
  data_disc->qvel[1] = 3.0;
  mj_step1(model.get(), data_disc.get());
  EXPECT_EQ(data_disc->sensordata[0], expected_joint_vel);
  EXPECT_EQ(data_disc->sensordata[1], expected_tendon_vel);

  // discrete integrator under mj_forward on fresh data
  MjDataPtr data_fwd = MakeData(model);
  data_fwd->qpos[0] = 0.6;
  data_fwd->qpos[1] = 0.5;
  data_fwd->qvel[0] = 2.0;
  data_fwd->qvel[1] = 3.0;
  mj_forward(model.get(), data_fwd.get());
  EXPECT_EQ(data_fwd->sensordata[0], expected_joint_vel);
  EXPECT_EQ(data_fwd->sensordata[1], expected_tendon_vel);
}

TEST_F(ForwardTest, CheckWarningCount) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide"/>
        <joint type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjtNum nan = std::numeric_limits<mjtNum>::quiet_NaN();
  mock_warning_handler.ExpectWarnings();

  for (int disable_autoreset : {0, 1}) {
    if (disable_autoreset) {
      model->opt.disableflags |= mjDSBL_AUTORESET;
    } else {
      model->opt.disableflags &= ~mjDSBL_AUTORESET;
    }

    mj_resetData(model.get(), data.get());
    data->qpos[1] = nan;
    mj_checkPos(model.get(), data.get());
    EXPECT_EQ(data->warning[mjWARN_BADQPOS].number, 1);
    EXPECT_EQ(data->warning[mjWARN_BADQPOS].lastinfo, 1);

    mj_resetData(model.get(), data.get());
    data->qvel[1] = nan;
    mj_checkVel(model.get(), data.get());
    EXPECT_EQ(data->warning[mjWARN_BADQVEL].number, 1);
    EXPECT_EQ(data->warning[mjWARN_BADQVEL].lastinfo, 1);

    mj_resetData(model.get(), data.get());
    data->qacc[1] = nan;
    mj_checkAcc(model.get(), data.get());
    EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 1);
    EXPECT_EQ(data->warning[mjWARN_BADQACC].lastinfo, 1);
  }
}

// after an RK4 step, qacc is the effective acceleration of the step,
// (qvel_next - qvel) / h, and the warm start is the last stage's acceleration
TEST_F(ForwardTest, RungeKuttaQaccIsEffectiveAcceleration) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="RK4" timestep="0.1" gravity="0 0 0">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <joint type="slide" axis="1 0 0" stiffness="4" damping="0.5"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();
  d->qpos[0] = 1.0;

  // reference RK4 on the linear spring-damper: a(x, v) = -4 x - 0.5 v
  auto acc = [](mjtNum x, mjtNum v) { return -4 * x - 0.5 * v; };
  const mjtNum h = m->opt.timestep;

  for (int i = 0; i < 3; i++) {
    mj_forward(m, d);
    mjtNum x0 = d->qpos[0], v0 = d->qvel[0];
    mjtNum a0 = acc(x0, v0);
    mjtNum a1 = acc(x0 + h / 2 * v0, v0 + h / 2 * a0);
    mjtNum v1 = v0 + h / 2 * a0;
    mjtNum a2 = acc(x0 + h / 2 * v1, v0 + h / 2 * a1);
    mjtNum v2 = v0 + h / 2 * a1;
    mjtNum a3 = acc(x0 + h * v2, v0 + h * a2);
    mjtNum qacc_rk = (a0 + 2 * a1 + 2 * a2 + a3) / 6;

    mj_RungeKutta(m, d, 4);
    mjtNum qacc_eff = (d->qvel[0] - v0) / h;
    EXPECT_NEAR(qacc_eff, qacc_rk, MjTol(1e-12, 1e-4));
    EXPECT_NEAR(d->qacc[0], qacc_rk, MjTol(1e-12, 1e-4));
    EXPECT_NEAR(d->qacc_warmstart[0], a3, MjTol(1e-12, 1e-4));
    EXPECT_GT(mju_abs(qacc_rk - a3), 0.01);
  }
}

// under RK4, delayed sensors record stage-0 values (a pure time shift of a
// non-delayed sensor, independent of whether flg_rnepost was set at stage 0)
TEST_F(ForwardTest, RungeKuttaDelayedAccelerometer) {
  static const std::string kXmlBoth = R"(
  <mujoco>
    <option integrator="RK4" timestep="0.015625">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="j" type="hinge" axis="0 1 0"/>
        <geom size="0.1" pos="0.3 0 0" mass="1"/>
        <site name="s" pos="0.3 0 0" euler="20 35 10"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j"/>
    </actuator>
    <sensor>
      <accelerometer site="s"/>
      <accelerometer site="s" delay="0.046875" nsample="10" interp="zoh"/>
      <accelerometer site="s" delay="0.0625" interval="0.03125" nsample="10"
        interp="zoh"/>
    </sensor>
  </mujoco>
  )";
  std::string xml_delayed_only = kXmlBoth;
  const std::string target = "      <accelerometer site=\"s\"/>\n";
  size_t pos = xml_delayed_only.find(target);
  ASSERT_NE(pos, std::string::npos);
  xml_delayed_only.erase(pos, target.size());

  char error[1024];
  MjModelPtr m1 = LoadModelFromString(kXmlBoth.c_str(), error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjModelPtr m2 =
      LoadModelFromString(xml_delayed_only.c_str(), error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << error;

  MjDataPtr d1 = MakeData(m1);
  MjDataPtr d2 = MakeData(m2);
  d1->qvel[0] = 2.5;
  d2->qvel[0] = 2.5;

  constexpr int kSteps = 10;
  std::vector<std::array<mjtNum, 3>> nodelay(kSteps);
  for (int step = 0; step < kSteps; ++step) {
    mjtNum ctrl = 4.0 * mju_sin(0.7 * step);
    d1->ctrl[0] = ctrl;
    d2->ctrl[0] = ctrl;
    mj_step(m1.get(), d1.get());
    mj_step(m2.get(), d2.get());

    for (int k = 0; k < 3; ++k) {
      nodelay[step][k] = d1->sensordata[k];
    }

    // continuous delay = 3*h: exact shift by 3 steps regardless of flg_rnepost
    if (step >= 3) {
      for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(d1->sensordata[3 + k], nodelay[step - 3][k],
                    MjTol(1e-12, 1e-5))
            << "step " << step << " axis " << k;
        EXPECT_NEAR(d2->sensordata[k], nodelay[step - 3][k], MjTol(1e-12, 1e-5))
            << "step " << step << " axis " << k;
      }
    }

    // interval = 2*h, delay = 4*h: samples even steps and holds on odd steps
    if (step >= 4) {
      int sample_step = ((step - 4) / 2) * 2;
      for (int k = 0; k < 3; ++k) {
        EXPECT_NEAR(d1->sensordata[6 + k], nodelay[sample_step][k],
                    MjTol(1e-12, 1e-5))
            << "step " << step << " axis " << k;
        EXPECT_NEAR(d2->sensordata[3 + k], nodelay[sample_step][k],
                    MjTol(1e-12, 1e-5))
            << "step " << step << " axis " << k;
      }
    }
  }
}

// under RK4, rotational setpoint wrapping and DC-motor LuGre bristle updates
// read stage-0 actuator_length and actuator_velocity rather than stage-3 values
TEST_F(ForwardTest, RungeKuttaStage0ActuatorState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="RK4" timestep="0.02" gravity="0 0 0">
      <flag contact="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
      <body pos="1 0 0">
        <joint name="dc_jnt" type="hinge" axis="0 0 1"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body>
    </worldbody>
    <actuator>
      <general name="integ" joint="ball" dyntype="integrator" gaintype="fixed"
               biastype="affine" gainprm="1" biasprm="0 -1 0" gear="1 0 0"/>
      <position name="filt" joint="ball" kp="1" timeconst="0.1" gear="1 0 0"/>
      <dcmotor name="dc" joint="dc_jnt" motorconst="1" resistance="1"
               lugre="100 1 0.5 1.5 0.1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  // ball spins fast around x: stage-0 length is 0, stage-3 length is ~0.4 rad,
  // so act = pi + 0.1 must wrap to -pi + 0.1 against stage-0 length = 0
  d->qvel[0] = 20.0;
  d->act[0] = mjPI + 0.1;
  d->act[1] = mjPI + 0.1;
  d->ctrl[1] = mjPI + 0.1;

  // dc_jnt accelerates strongly from v0 = 0.2 so stage-3 velocity differs a lot
  d->qvel[3] = 0.2;
  d->qfrc_applied[3] = 50.0;
  int dc_id = mj_name2id(m, mjOBJ_ACTUATOR, "dc");
  ASSERT_GE(dc_id, 0);
  int bristle_adr = m->actuator_actadr[dc_id];
  d->act[bristle_adr] = 0.01;

  mj_forward(m, d);

  // mj_Euler advances bristle state from stage-0 actuator_velocity
  MjDataPtr data_ref = MakeData(model);
  mj_copyData(data_ref.get(), m, d);
  mj_Euler(m, data_ref.get());
  mjtNum expected_bristle = data_ref->act[bristle_adr];

  mj_RungeKutta(m, d, 4);

  EXPECT_NEAR(d->act[0], -mjPI + 0.1, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->act[1], -mjPI + 0.1, MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d->act[bristle_adr], expected_bristle, MjTol(1e-12, 1e-5));
}

}  // namespace
}  // namespace mujoco
