// Copyright 2022 DeepMind Technologies Limited
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

// Tests for engine/engine_derivative.c.

#include "src/engine/engine_derivative.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_core_smooth.h"
#include "src/engine/engine_core_util.h"
#include "src/engine/engine_derivative_fd.h"
#include "src/engine/engine_forward.h"
#include "src/engine/engine_io.h"
#include "src/engine/engine_metric.h"
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_sparse.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::vector;
using ::testing::DoubleNear;
using ::testing::Each;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::NotNull;
using ::testing::Pointwise;
using DerivativeTest = MujocoTest;

// errors smaller than this are ignored
static const mjtNum absolute_tolerance = MjTol(1e-9, 1e-3);

// corrected relative error
static mjtNum RelativeError(mjtNum a, mjtNum b) {
  mjtNum nominator = mjMAX(0, mju_abs(a - b) - absolute_tolerance);
  mjtNum denominator = (mju_abs(a) + mju_abs(b) + absolute_tolerance);
  return nominator / denominator;
}

// expect two 2D arrays to have elementwise relative error smaller than eps
// return maximum absolute error
static mjtNum CompareMatrices(mjtNum* Actual, mjtNum* Expected, int nrow,
                              int ncol, mjtNum eps) {
  mjtNum max_error = 0;
  for (int i = 0; i < nrow; i++) {
    for (int j = 0; j < ncol; j++) {
      mjtNum actual = Actual[i * ncol + j];
      mjtNum expected = Expected[i * ncol + j];
      EXPECT_LT(RelativeError(actual, expected), eps)
          << "error at position (" << i << ", " << j << ")"
          << "\nexpected = " << expected << "\nactual   = " << actual
          << "\ndiff     = " << expected - actual;
      max_error = mjMAX(mju_abs(actual - expected), max_error);
    }
  }
  return max_error;
}

static const char* const kEnergyConservingPendulumPath =
    "engine/testdata/derivative/energy_conserving_pendulum.xml";
static const char* const kTumblingThinObjectPath =
    "engine/testdata/derivative/tumbling_thin_object.xml";
static const char* const kTumblingThinObjectEllipsoidPath =
    "engine/testdata/derivative/tumbling_thin_object_ellipsoid.xml";
static const char* const kDampedActuatorsPath =
    "engine/testdata/derivative/damped_actuators.xml";
static const char* const kDamperActuatorsPath =
    "engine/testdata/actuation/damper.xml";
static const char* const kDampedPendulumPath =
    "engine/testdata/derivative/damped_pendulum.xml";
static const char* const kLinearPath = "engine/testdata/derivative/linear.xml";
static const char* const kDCMotorPath =
    "engine/testdata/derivative/dcmotor.xml";
static const char* const kModelPath = "testdata/model.xml";
static const char* const kSleepEqualityPath =
    "engine/testdata/sleep/equality.xml";

// compare analytic and finite-difference d_smooth/d_qvel
TEST_F(DerivativeTest, SmoothDvel) {
  // run test on all models
  for (const char* local_path :
       {kEnergyConservingPendulumPath, kTumblingThinObjectPath,
        kDampedActuatorsPath, kDamperActuatorsPath, kDCMotorPath}) {
    const std::string xml_path = GetTestDataFilePath(local_path);
    char error[1024] = "";
    mjModel* model =
        mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(model, testing::NotNull()) << "Failed to load model: " << error;
    int nD = model->nD;
    mjData* data = mj_makeData(model);

    for (mjtJacobian sparsity : {mjJAC_DENSE, mjJAC_SPARSE}) {
      // set sparsity
      model->opt.jacobian = sparsity;

      // take 100 steps so we have some velocities, then call forward
      mj_resetData(model, data);
      if (model->nu) {
        data->ctrl[0] = 0.1;
      }
      for (int i = 0; i < 100; i++) {
        mj_step(model, data);
      }
      mj_forward(model, data);

      // construct sparse structure in d->D_xxx, compute analytical qDeriv
      mju_zero(data->qDeriv, nD);
      mjd_smooth_vel(model, data, /*flg_bias=*/true);

      // expect derivatives to be non-zero, make copy of qDeriv as a vector
      EXPECT_GT(mju_norm(data->qDeriv, nD), 0);
      vector<mjtNum> qDerivAnalytic = AsVector(data->qDeriv, nD);

      // compute finite-difference derivatives
      mjtNum eps = MjEps(1e-7, 1e-3);
      mju_zero(data->qDeriv, nD);
      mjd_smooth_velFD(model, data, eps);

      // expect FD and analytic derivatives to be numerically different
      EXPECT_NE(mju_norm(data->qDeriv, nD),
                mju_norm(qDerivAnalytic.data(), nD));

      // expect FD and analytic derivatives to be similar to eps precision
      EXPECT_THAT(AsVector(data->qDeriv, nD),
                  Pointwise(MjNear(1e-7, 3e-3), qDerivAnalytic));
    }
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

// mjd_freeBias_vel: 6x6 bias-derivative block for a standalone free body
//   validated against mjd_rne_vel and against finite-differenced mj_rne
TEST_F(DerivativeTest, FreeBiasVel) {
  // free body with offset CoM, rotated inertia, non-identity orientation
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0.1 -0.2 0.3" euler="20 -30 40">
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
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

  // set fast, fully populated velocity
  mjtNum qvel[6] = {0.4, -0.3, 0.2, 5, -3, 2};
  mju_copy(d->qvel, qvel, 6);
  mj_forward(m, d);

  // analytic block
  mjtNum B[36];
  mjd_freeBias_vel(m, d, /*jnt=*/0, B);

  // linear columns are zero by construction
  for (int r = 0; r < 6; r++) {
    for (int c = 0; c < 3; c++) {
      EXPECT_EQ(B[6 * r + c], 0);
    }
  }

  // compare with mjd_rne_vel: B == -(qDeriv(flg_bias=1) - qDeriv(flg_bias=0))
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m, d, /*flg_bias=*/1);
  vector<mjtNum> qDeriv_bias = AsVector(d->qDeriv, m->nD);
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m, d, /*flg_bias=*/0);
  for (int r = 0; r < 6; r++) {
    int rowadr = m->D_rowadr[r];
    ASSERT_EQ(m->D_rownnz[r], 6);
    for (int k = 0; k < 6; k++) {
      int c = m->D_colind[rowadr + k];
      mjtNum rne_val = -(qDeriv_bias[rowadr + k] - d->qDeriv[rowadr + k]);
      EXPECT_NEAR(B[6 * r + c], rne_val, MjTol(1e-14, 1e-6))
          << "mismatch at (" << r << ", " << c << ")";
    }
  }

  // compare with central finite differences of mj_rne
  mjtNum eps = MjEps(1e-6, 1e-3);
  for (int c = 0; c < 6; c++) {
    mjtNum bias_plus[6], bias_minus[6];

    d->qvel[c] = qvel[c] + eps;
    mj_comVel(m, d);
    mj_rne(m, d, /*flg_acc=*/0, bias_plus);

    d->qvel[c] = qvel[c] - eps;
    mj_comVel(m, d);
    mj_rne(m, d, /*flg_acc=*/0, bias_minus);

    d->qvel[c] = qvel[c];

    for (int r = 0; r < 6; r++) {
      mjtNum fd = (bias_plus[r] - bias_minus[r]) / (2 * eps);
      EXPECT_NEAR(B[6 * r + c], fd, MjTol(1e-7, 1e-2))
          << "FD mismatch at (" << r << ", " << c << ")";
    }
  }
}

// fixed descendants contribute inertia about the free joint, not their own body
// frames
TEST_F(DerivativeTest, FreeBiasVelFixedDescendants) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40">
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03" euler="10 20 30"/>
        <body pos=".2 -.1 .3" euler="30 20 -10">
          <geom type="box" size=".2 .1 .1" mass="1" pos=".03 .01 -.02"/>
          <body pos="-.1 .2 .1" euler="10 -20 30">
            <geom type="box" size=".1 .1 .2" mass=".5"/>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nbody, 4);
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();
  mjtNum qvel[6] = {0.4, -0.3, 0.2, 5, -3, 2};
  mju_copy(d->qvel, qvel, 6);
  mj_forward(m, d);

  mjtNum B[36];
  mjd_freeBias_vel(m, d, /*jnt=*/0, B);

  // compare the aggregate derivative with the full recursive Newton-Euler
  // derivative
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m, d, /*flg_bias=*/1);
  for (int r = 0; r < 6; r++) {
    ASSERT_EQ(m->D_rownnz[r], 6);
    for (int k = 0; k < 6; k++) {
      int adr = m->D_rowadr[r] + k;
      EXPECT_NEAR(B[6 * r + m->D_colind[adr]], -d->qDeriv[adr],
                  MjTol(1e-14, 1e-5));
    }
  }

  // independent central finite differences, including the zero linear-velocity
  // columns
  mjtNum eps = MjEps(1e-6, 1e-3);
  for (int c = 0; c < 6; c++) {
    mjtNum plus[6], minus[6];
    d->qvel[c] = qvel[c] + eps;
    mj_comVel(m, d);
    mj_rne(m, d, /*flg_acc=*/0, plus);
    d->qvel[c] = qvel[c] - eps;
    mj_comVel(m, d);
    mj_rne(m, d, /*flg_acc=*/0, minus);
    d->qvel[c] = qvel[c];
    for (int r = 0; r < 6; r++) {
      EXPECT_NEAR(B[6 * r + c], (plus[r] - minus[r]) / (2 * eps),
                  MjTol(1e-7, 1e-2));
    }
  }
}

// fixed-body fusion preserves the bias derivative of a massless free root
TEST_F(DerivativeTest, FreeBiasVelFusedInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="false" alignfree="false"/>
    <default><geom contype="0" conaffinity="0"/></default>
    <worldbody>
      <body pos=".1 -.2 .3" euler="20 -30 40">
        <freejoint/>
        <body pos=".2 -.1 .3" euler="30 20 -10">
          <geom type="box" size=".2 .1 .1" mass="1" pos=".03 .01 -.02"/>
          <body pos="-.1 .2 .1" euler="10 -20 30">
            <geom type="box" size=".1 .1 .2" mass=".5"/>
          </body>
        </body>
        <body pos="-.1 .3 -.2" euler="-20 10 40">
          <geom type="box" size=".1 .2 .3" mass="2" pos=".04 -.02 .03"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::string fused_xml = xml;
  fused_xml.replace(fused_xml.find("false"), 5, "true");
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjModelPtr fused =
      LoadModelFromString(fused_xml.c_str(), error, sizeof(error));
  ASSERT_THAT(fused.get(), NotNull()) << error;
  ASSERT_EQ(model->nbody, 5);
  ASSERT_EQ(fused->nbody, 2);
  ASSERT_EQ(model->body_mass[1], 0);
  MjDataPtr data = MakeData(model);
  MjDataPtr fused_data = MakeData(fused);

  mjtNum qvel[6] = {0.4, -0.3, 0.2, 5, -3, 2};
  mju_copy(data->qvel, qvel, 6);
  mju_copy(fused_data->qvel, qvel, 6);
  mj_forward(model.get(), data.get());
  mj_forward(fused.get(), fused_data.get());

  mjtNum B[36], B_fused[36];
  mjd_freeBias_vel(model.get(), data.get(), /*jnt=*/0, B);
  mjd_freeBias_vel(fused.get(), fused_data.get(), /*jnt=*/0, B_fused);
  EXPECT_GT(mju_norm(B, 36), 1);

  // fusing diagonalizes the inertia at compiler precision
  EXPECT_THAT(AsVector(B, 36),
              Pointwise(MjNear(1e-5, 1e-5), AsVector(B_fused, 36)));
}

// a jointed descendant must still exclude the free root from the local six-DOF
// solve
TEST_F(DerivativeTest, FreeMhatRejectsArticulatedSubtree) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1"/>
        <body pos="0 0 1">
          <joint/>
          <geom size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  mjtNum A[36];
  for (int discrete : {0, 1}) {
    EXPECT_EQ(mjd_freeMhat(model.get(), data.get(), /*jnt=*/0,
                           model->opt.timestep, A, discrete),
              0);
  }
}

// disabled actuators do not contribute to d_qfrc_actuator/d_qvel
TEST_F(DerivativeTest, DisabledActuators) {
  // model with only a position actuator
  static constexpr char xml1[] = R"(
  <mujoco>
    <option integrator="implicitfast"/>

    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <position joint="joint" group="1" kp="2000" kv="200"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml1, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << error;
  MjDataPtr d1 = MakeData(m1);

  d1->ctrl[0] = 6;
  while (d1->time < 1) mj_step(m1.get(), d1.get());

  // model with a position actuator and an intvelocity actuator
  static constexpr char xml2[] = R"(
  <mujoco>
    <option integrator="implicitfast" actuatorgroupdisable="2"/>

    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <position joint="joint" group="1" kp="2000" kv="200"/>
      <intvelocity joint="joint" group="2" kp="2000" kv="200" actrange="-6 6"/>
    </actuator>
  </mujoco>
  )";

  MjModelPtr m2 = LoadModelFromString(xml2);
  MjDataPtr d2 = MakeData(m2);

  d2->ctrl[0] = 6;
  d2->ctrl[1] = 6;

  while (d2->time < 1) mj_step(m2.get(), d2.get());

  // expect same qvel in both models
  EXPECT_EQ(d1->qvel[0], d2->qvel[0]);
}

// actuator order has no effect
TEST_F(DerivativeTest, ActuatorOrder) {
  // model with stateful actuator first
  static constexpr char xml1[] = R"(
  <mujoco>
    <option integrator="implicitfast"/>

    <worldbody>
      <body>
        <joint name="0" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
      <body pos="1 0 0">
        <joint name="1" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <muscle joint="0" ctrlrange="0 6"/>
      <damper joint="1" kv="200" ctrlrange="0 6"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m1 = LoadModelFromString(xml1, error, sizeof(error));
  ASSERT_THAT(m1.get(), NotNull()) << "Failed to load model: " << error;
  MjDataPtr d1 = MakeData(m1);

  d1->ctrl[0] = 6;
  d1->ctrl[1] = 6;

  while (d1->time < 1) mj_step(m1.get(), d1.get());

  // model with stateful actuator second
  static constexpr char xml2[] = R"(
  <mujoco>
    <option integrator="implicitfast"/>

    <worldbody>
      <body>
        <joint name="0" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
      <body pos="1 0 0">
        <joint name="1" type="slide" range="-1 1"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <damper joint="1" kv="200" ctrlrange="0 6"/>
      <muscle joint="0" ctrlrange="0 6"/>
    </actuator>
  </mujoco>
  )";

  MjModelPtr m2 = LoadModelFromString(xml2, error, sizeof(error));
  ASSERT_THAT(m2.get(), NotNull()) << "Failed to load model: " << error;
  MjDataPtr d2 = MakeData(m2);

  d2->ctrl[0] = 6;
  d2->ctrl[1] = 6;

  while (d2->time < 1) mj_step(m2.get(), d2.get());

  // expect same qvel in both models
  EXPECT_EQ(d1->qvel[0], d2->qvel[0]);
  EXPECT_EQ(d1->qvel[1], d2->qvel[1]);
}

// compare analytic and fin-diff d_qfrc_passive/d_qvel
TEST_F(DerivativeTest, PassiveDvel) {
  for (const char* local_path :
       {kTumblingThinObjectPath, kTumblingThinObjectEllipsoidPath}) {
    // load model
    const std::string xml_path = GetTestDataFilePath(local_path);
    char error[1024];
    mjModel* model =
        mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(model, NotNull()) << error;
    int nD = model->nD;
    mjData* data = mj_makeData(model);
    // allocate Jacobians
    mjtNum* qDerivAnalytic = (mjtNum*)mju_malloc(sizeof(mjtNum) * nD);
    mjtNum* qDerivFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * nD);

    for (mjtJacobian sparsity : {mjJAC_DENSE, mjJAC_SPARSE}) {
      // set sparsity
      model->opt.jacobian = sparsity;

      // take 100 steps so we have some velocities, then call forward
      mj_resetData(model, data);
      for (int i = 0; i < 100; i++) {
        mj_step(model, data);
      }
      mj_forward(model, data);

      // get analytic derivatives
      mju_zero(data->qDeriv, model->nD);
      mjd_passive_vel(model, data);
      mju_copy(qDerivAnalytic, data->qDeriv, nD);

      // clear qDeriv, get finite-difference derivatives
      mju_zero(data->qDeriv, nD);
      mju_zero(qDerivFD, nD);
      mjtNum eps = MjEps(1e-6, 1e-4);
      mjd_passive_velFD(model, data, eps);

      // expect FD and analytic derivatives to be similar to tol precision
      EXPECT_THAT(AsVector(data->qDeriv, nD),
                  Pointwise(MjNear(1e-6, 1e-4), AsVector(qDerivAnalytic, nD)));
    }

    mju_free(qDerivFD);
    mju_free(qDerivAnalytic);
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

// ----------------------- derivatives of mj_step() ----------------------------

// mj_stepSkip computes the same next state as mj_step
TEST_F(DerivativeTest, StepSkip) {
  const std::string xml_path = GetTestDataFilePath(kDampedPendulumPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);
  int nq = model->nq;
  int nv = model->nv;

  // disable warm-starts so we don't need to save qacc_warmstart
  model->opt.disableflags |= mjDSBL_WARMSTART;

  for (const mjtIntegrator integrator :
       {mjINT_EULER, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    model->opt.integrator = integrator;

    // reset, take 20 steps
    mj_resetData(model, data);
    for (int i = 0; i < 20; i++) {
      mj_step(model, data);
    }

    // denormalize the quat, just to see that it doesn't make a difference
    for (int j = 0; j < model->njnt; j++) {
      if (model->jnt_type[j] == mjJNT_BALL) {
        int adr = model->jnt_qposadr[j];
        for (int k = 0; k < 4; k++) {
          data->qpos[adr + k] *= 8;
        }
      }
    }

    // save state
    vector<mjtNum> qpos = AsVector(data->qpos, nq);
    vector<mjtNum> qvel = AsVector(data->qvel, nv);

    // take one more step, save next state
    mj_step(model, data);
    vector<mjtNum> qpos_next = AsVector(data->qpos, nq);
    vector<mjtNum> qvel_next = AsVector(data->qvel, nv);

    // reset state, take step again, compare (assert mj_step is deterministic)
    mju_copy(data->qpos, qpos.data(), nq);
    mju_copy(data->qvel, qvel.data(), nv);
    mj_step(model, data);
    EXPECT_THAT(AsVector(data->qpos, nq), Pointwise(Eq(), qpos_next));
    EXPECT_THAT(AsVector(data->qvel, nv), Pointwise(Eq(), qvel_next));

    // reset state, change ctrl, call mj_stepSkip, save next state
    mju_copy(data->qpos, qpos.data(), nq);
    mju_copy(data->qvel, qvel.data(), nv);
    data->ctrl[0] = 1;
    mj_stepSkip(model, data, mjSTAGE_VEL, 0);  // skipping both POS and VEL
    vector<mjtNum> qpos_next_dctrl = AsVector(data->qpos, nq);
    vector<mjtNum> qvel_next_dctrl = AsVector(data->qvel, nv);

    // reset state (ctrl remains unchanged), call full mj_step, compare
    mju_copy(data->qpos, qpos.data(), nq);
    mju_copy(data->qvel, qvel.data(), nv);
    mj_step(model, data);
    EXPECT_THAT(AsVector(data->qpos, nq), Pointwise(Eq(), qpos_next_dctrl));
    EXPECT_THAT(AsVector(data->qvel, nv), Pointwise(Eq(), qvel_next_dctrl));

    // reset state, change velocity, call mj_stepSkip, save next state
    mju_copy(data->qpos, qpos.data(), nq);
    mju_copy(data->qvel, qvel.data(), nv);
    data->qvel[0] += 1;
    mj_stepSkip(model, data, mjSTAGE_POS, 0);  // skipping POS
    vector<mjtNum> qpos_next_dvel = AsVector(data->qpos, nq);
    vector<mjtNum> qvel_next_dvel = AsVector(data->qvel, nv);

    // reset state, change velocity, call full mj_step, compare
    mju_copy(data->qpos, qpos.data(), nq);
    mju_copy(data->qvel, qvel.data(), nv);
    data->qvel[0] += 1;
    mj_step(model, data);
    EXPECT_THAT(AsVector(data->qpos, nq), Pointwise(Eq(), qpos_next_dvel));
    EXPECT_THAT(AsVector(data->qvel, nv), Pointwise(Eq(), qvel_next_dvel));
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// mj_stepSkip dispatches to mj_ipc when mjENBL_IPC is enabled
TEST_F(DerivativeTest, StepSkipIPC) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.005">
      <flag ipc="enable" warmstart="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 0.1"/>
      <flexcomp name="f" type="grid" count="2 2 1" spacing="0.1 0.1 0.1"
                pos="0 0 0.02" dim="2" radius="0.01" mass="0.2">
        <contact selfcollide="none" margin="0.03"/>
        <elasticity young="1e3" poisson="0.2" damping="1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d_step = MakeData(m);
  MjDataPtr d_skip = MakeData(m);

  // initial velocity toward the ground plane so IPC barrier is active
  for (int i = 2; i < m->nv; i += 3) {
    d_step->qvel[i] = -0.5;
  }

  for (mjtStage stage : {mjSTAGE_NONE, mjSTAGE_POS, mjSTAGE_VEL}) {
    mj_copyData(d_skip.get(), m.get(), d_step.get());
    if (stage >= mjSTAGE_POS) {
      mj_forward(m.get(), d_skip.get());
    }
    if (stage == mjSTAGE_POS) {
      d_skip->qvel[0] += 0.1;
    }

    MjDataPtr d_ref = MakeData(m);
    mj_copyData(d_ref.get(), m.get(), d_skip.get());

    mj_stepSkip(m.get(), d_skip.get(), stage, 0);
    mj_step(m.get(), d_ref.get());

    EXPECT_THAT(AsVector(d_skip->qpos, m->nq),
                Pointwise(MjNear(1e-12, 1e-6), AsVector(d_ref->qpos, m->nq)));
    EXPECT_THAT(AsVector(d_skip->qvel, m->nv),
                Pointwise(MjNear(1e-12, 1e-6), AsVector(d_ref->qvel, m->nv)));
  }
}

// mj_stepSkip refactorizes implicit integrators when qDeriv depends on ctrl/act
TEST_F(DerivativeTest, StepSkipImplicitCtrlActDeriv) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.05">
      <flag warmstart="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="j0" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="j1" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="j2" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="j0" kv="10" ctrlrange="0 5"/>
      <position joint="j1" kp="20" dampratio="1" forcerange="-1 1"/>
      <general joint="j2" dyntype="filter" dynprm="0.1"
               gaintype="affine" gainprm="0 0 -8"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  int nv = m->nv, na = m->na, nu = m->nu;
  int ndx = 2 * nv + na;
  mjtNum eps = MjTol(1e-6, 1e-3);

  for (mjtIntegrator integrator : {mjINT_IMPLICIT, mjINT_IMPLICITFAST}) {
    m->opt.integrator = integrator;
    mj_resetData(m.get(), d.get());
    d->qvel[0] = 1.2;
    d->qvel[1] = 0.8;
    d->qvel[2] = -0.9;
    d->ctrl[0] = 1.0;
    d->ctrl[1] = 0.02;  // unsaturated; large nudge pushes into forcerange
    d->ctrl[2] = 0.5;
    d->act[0] = 1.5;

    // verify mj_stepSkip at mjSTAGE_VEL matches mj_step after ctrl and act
    // nudges
    mj_forward(m.get(), d.get());
    MjDataPtr d_skip = MakeData(m);
    MjDataPtr d_full = MakeData(m);
    mj_copyData(d_skip.get(), m.get(), d.get());
    d_skip->ctrl[0] = 3.0;
    d_skip->ctrl[1] = 2.0;  // crosses into saturation
    d_skip->act[0] = 2.5;
    mj_copyData(d_full.get(), m.get(), d_skip.get());

    mj_stepSkip(m.get(), d_skip.get(), mjSTAGE_VEL, 0);
    mj_step(m.get(), d_full.get());
    EXPECT_THAT(AsVector(d_skip->qvel, nv),
                Pointwise(MjNear(1e-12, 1e-6), AsVector(d_full->qvel, nv)));

    // verify mjd_transitionFD matches full mj_step centered differences
    mj_resetData(m.get(), d.get());
    d->qvel[0] = 1.2;
    d->qvel[1] = 0.8;
    d->qvel[2] = -0.9;
    d->ctrl[0] = 1.0;
    d->ctrl[1] = 0.01;
    d->ctrl[2] = 0.5;
    d->act[0] = 1.5;

    vector<mjtNum> A(ndx * ndx), B(ndx * nu);
    mjd_transitionFD(m.get(), d.get(), eps, /*flg_centered=*/1, A.data(),
                     B.data(), nullptr, nullptr);

    // check B columns (ctrl) and A column 2*nv (filter act) against mj_step
    for (int col_u = 0; col_u < 2; col_u++) {
      MjDataPtr dp = MakeData(m), dm = MakeData(m);
      mj_copyData(dp.get(), m.get(), d.get());
      mj_copyData(dm.get(), m.get(), d.get());
      dp->ctrl[col_u] += eps;
      dm->ctrl[col_u] -= eps;
      mj_step(m.get(), dp.get());
      mj_step(m.get(), dm.get());
      for (int r = 0; r < nv; r++) {
        mjtNum dq = (dp->qpos[r] - dm->qpos[r]) / (2 * eps);
        mjtNum dv = (dp->qvel[r] - dm->qvel[r]) / (2 * eps);
        EXPECT_NEAR(B[r * nu + col_u], dq, MjTol(1e-8, 1e-3));
        EXPECT_NEAR(B[(nv + r) * nu + col_u], dv, MjTol(1e-8, 1e-3));
      }
    }

    MjDataPtr dp = MakeData(m), dm = MakeData(m);
    mj_copyData(dp.get(), m.get(), d.get());
    mj_copyData(dm.get(), m.get(), d.get());
    dp->act[0] += eps;
    dm->act[0] -= eps;
    mj_step(m.get(), dp.get());
    mj_step(m.get(), dm.get());
    for (int r = 0; r < nv; r++) {
      mjtNum dv = (dp->qvel[r] - dm->qvel[r]) / (2 * eps);
      EXPECT_NEAR(A[(nv + r) * ndx + 2 * nv], dv, MjTol(1e-8, 1e-3));
    }
  }
}

// mj_stepSkip refactorizes Euler integrator when nonlinear dampingpoly is
// present
TEST_F(DerivativeTest, StepSkipEulerPolyDamping) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="Euler" timestep="0.05">
      <flag warmstart="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="j0" type="slide" damping="1 2 3"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="j1" type="slide" damping="0.5"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="j1" damping="0 1.5"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  d->qvel[0] = 1.3;
  d->qvel[1] = -0.9;

  int nv = m->nv;
  int ndx = 2 * nv;
  mjtNum eps = MjTol(1e-6, 1e-3);
  vector<mjtNum> A(ndx * ndx);
  mjd_transitionFD(m.get(), d.get(), eps, /*flg_centered=*/1, A.data(), nullptr,
                   nullptr, nullptr);

  for (int j = 0; j < nv; j++) {
    MjDataPtr dp = MakeData(m), dm = MakeData(m);
    mj_copyData(dp.get(), m.get(), d.get());
    mj_copyData(dm.get(), m.get(), d.get());
    dp->qvel[j] += eps;
    dm->qvel[j] -= eps;
    mj_step(m.get(), dp.get());
    mj_step(m.get(), dm.get());
    for (int r = 0; r < nv; r++) {
      mjtNum dq = (dp->qpos[r] - dm->qpos[r]) / (2 * eps);
      mjtNum dv = (dp->qvel[r] - dm->qvel[r]) / (2 * eps);
      EXPECT_NEAR(A[r * ndx + nv + j], dq, MjTol(1e-8, 1e-3));
      EXPECT_NEAR(A[(nv + r) * ndx + nv + j], dv, MjTol(1e-8, 1e-3));
    }
  }
}

// Analytic transition matrices for linear dynamical system xn = A*x + B*u
//   given modified mass matrix H (`data->qH`) and
//   Ac = H^-1 [diag(-stiffness) diag(-damping)]
//   we have
//   A  = eye(2*nv) + dt [dt*Ac + [zeros(3) eye(3)]; Ac]
//   given the moment arm matrix K (`data->actuator_moment`) and Bc = H^-1 K
//   B  = dt*[Bc*dt; Bc]
static void LinearSystem(const mjModel* m, mjData* d, mjtNum* A, mjtNum* B) {
  int nv = m->nv, nu = m->nu;
  mjtNum dt = m->opt.timestep;
  mj_markStack(d);

  // === state-transition matrix A
  if (A) {
    mjtNum* Ac = mj_stackAllocNum(d, 2 * nv * nv);
    // Ac = H^-1 [diag(-stiffness) diag(-damping)]
    mju_zero(Ac, 2 * nv * nv);
    for (int i = 0; i < nv; i++) {
      Ac[i * nv + i] = -m->jnt_stiffness[i];
      Ac[nv * nv + i * nv + i] = -m->dof_damping[i];
    }
    mj_solveLD(Ac, d->qH, d->qHDiagInv, nv, 2 * nv, m->M_rownnz, m->M_rowadr,
               m->M_colind, nullptr);

    // A = [dt*Ac; Ac]
    mju_transpose(A, Ac, 2 * nv, nv);
    mju_scl(A, A, dt, nv * 2 * nv);
    mju_transpose(A + 2 * nv * nv, Ac, 2 * nv, nv);

    // Add eye(nv) to top right quadrant of A
    for (int i = 0; i < nv; i++) {
      A[i * 2 * nv + nv + i] += 1;
    }

    // A *= dt
    mju_scl(A, A, dt, 2 * nv * 2 * nv);

    // A += eye(2*nv)
    for (int i = 0; i < 2 * nv; i++) {
      A[i * 2 * nv + i] += 1;
    }
  }

  // === control-transition matrix B
  if (B) {
    mjtNum* Bc = mj_stackAllocNum(d, nu * nv);
    mjtNum* BcT = mj_stackAllocNum(d, nv * nu);
    mju_sparse2dense(Bc, d->actuator_moment, nu, nv, d->moment_rownnz,
                     d->moment_rowadr, d->moment_colind);
    mj_solveLD(Bc, d->qH, d->qHDiagInv, nv, nu, m->M_rownnz, m->M_rowadr,
               m->M_colind, nullptr);
    mju_transpose(BcT, Bc, nu, nv);
    mju_scl(B, BcT, dt * dt, nu * nv);
    mju_scl(B + nu * nv, BcT, dt, nu * nv);
  }

  mj_freeStack(d);
}

// compare FD derivatives to analytic derivatives of linear dynamical system
TEST_F(DerivativeTest, LinearSystem) {
  const std::string xml_path = GetTestDataFilePath(kLinearPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);
  int nv = model->nv, nu = model->nu;

  // set ctrl, integrate for 20 steps
  data->ctrl[0] = .1;
  data->ctrl[1] = -.1;
  for (int i = 0; i < 20; i++) {
    mj_step(model, data);
  }

  // analytic A and B
  mjtNum* A = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  LinearSystem(model, data, A, B);

  // uncomment for debugging:
  // PrintMatrix(A, 2*nv, 2*nv);
  // PrintMatrix(B, 2*nv, nu);

  // forward differenced A and B
  mjtNum eps = MjEps(1e-6, 1e-3);
  mjtNum* AFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum* BFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  mjd_transitionFD(model, data, eps, /*centered=*/0, AFD, BFD, nullptr,
                   nullptr);

  // uncomment for debugging:
  // PrintMatrix(AFD, 2*nv, 2*nv);
  // PrintMatrix(BFD, 2*nv, nu);

  // expect FD and analytic derivatives to be similar to eps precision
  CompareMatrices(A, AFD, 2 * nv, 2 * nv, eps);
  CompareMatrices(B, BFD, 2 * nv, nu, eps);

  // central differenced A and B
  mjtNum* AFDc = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * 2 * nv);
  mjtNum* BFDc = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjd_transitionFD(model, data, eps, /*centered=*/1, AFDc, BFDc, nullptr,
                   nullptr);

  // expect central derivatives to be equal to forward differences
  CompareMatrices(AFD, AFDc, 2 * nv, 2 * nv, eps);
  CompareMatrices(BFD, BFDc, 2 * nv, nu, eps);

  mju_free(BFDc);
  mju_free(AFDc);
  mju_free(BFD);
  mju_free(AFD);
  mju_free(B);
  mju_free(A);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// check ctrl derivatives at the range limit
TEST_F(DerivativeTest, ClampedCtrlDerivatives) {
  const std::string xml_path = GetTestDataFilePath(kLinearPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);
  int nv = model->nv, nu = model->nu;

  // set ctrl, integrate for 20 steps
  data->ctrl[0] = .1;
  data->ctrl[1] = -.1;
  for (int i = 0; i < 20; i++) {
    mj_step(model, data);
  }

  // analytic B
  mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  LinearSystem(model, data, nullptr, B);

  // forward differenced A and B
  mjtNum eps = MjEps(1e-6, 1e-3);
  mjtNum* BFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);

  // set ctrl to the limits, request forward differences
  data->ctrl[0] = 1;
  data->ctrl[1] = -1;
  mjd_transitionFD(model, data, eps, /*centered=*/0, nullptr, BFD, nullptr,
                   nullptr);
  // expect FD and analytic derivatives to be similar to eps precision
  CompareMatrices(B, BFD, 2 * nv, nu, eps);

  // ctrl remains at limits, request central differences
  mjd_transitionFD(model, data, eps, /*centered=*/1, nullptr, BFD, nullptr,
                   nullptr);
  // expect FD and analytic derivatives to be similar to eps precision
  CompareMatrices(B, BFD, 2 * nv, nu, eps);

  // set ctrl beyond limits, request forward differences
  data->ctrl[0] = 2;
  data->ctrl[1] = -2;
  mjd_transitionFD(model, data, eps, /*centered=*/0, nullptr, BFD, nullptr,
                   nullptr);
  // expect derivatives to be 0
  EXPECT_THAT(AsVector(BFD, 2 * nv * nu), Each(Eq(0.0)));

  // expect ctrl to remain unchanged (despite internal clamping)
  EXPECT_EQ(data->ctrl[0], 2.0);
  EXPECT_EQ(data->ctrl[1], -2.0);

  // ctrl remains beyond limits, request centered differences
  mjd_transitionFD(model, data, eps, /*centered=*/1, nullptr, BFD, nullptr,
                   nullptr);
  // expect derivatives to be 0
  EXPECT_THAT(AsVector(BFD, 2 * nv * nu), Each(Eq(0.0)));

  mju_free(BFD);
  mju_free(B);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// compare FD sensor derivatives to analytic derivatives
TEST_F(DerivativeTest, SensorDerivatives) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <general name="actuator" joint="joint" gainprm="3"/>
    </actuator>

    <sensor>
      <jointpos joint="joint"/>
      <jointvel joint="joint"/>
      <actuatorfrc actuator="actuator"/>
    </sensor>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  int nv = model->nv, nu = model->nu, ns = model->nsensordata;
  MjDataPtr data = MakeData(model);

  // finite differenced C and D
  mjtNum eps = 1e-6;
  mjtNum* CFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * ns * 2 * nv);
  mjtNum* DFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * ns * nu);
  mjd_transitionFD(model.get(), data.get(), eps, /*centered=*/0, nullptr,
                   nullptr, CFD, DFD);

  // expected analytic C and D
  mjtNum C[6] = {1, 0, 0, 1, 0, 0};

  mjtNum D[3] = {
      0,
      0,
      3,
  };

  // compare expected and actual values
  CompareMatrices(CFD, C, ns, 2 * nv, eps);
  CompareMatrices(DFD, D, ns, nu, eps);

  mju_free(DFD);
  mju_free(CFD);
}

// centered control-to-sensor FD derivatives preserve sign and ctrlrange
// clamping
TEST_F(DerivativeTest, CenteredSensorControlDerivatives) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option gravity="0 0 0">
      <flag warmstart="disable"/>
    </option>
    <default>
      <geom contype="0" conaffinity="0"/>
    </default>
    <worldbody>
      <body>
        <joint name="j0" type="slide" axis="1 0 0"/>
        <geom size=".1" mass="1"/>
        <site name="s0"/>
      </body>
      <body>
        <joint name="j1" type="slide" axis="1 0 0"/>
        <geom size=".1" mass="2"/>
      </body>
    </worldbody>

    <actuator>
      <general name="a0" joint="j0" gainprm="3" ctrlrange="-1 1"/>
      <general name="a1" joint="j1" gainprm="-4"/>
    </actuator>

    <sensor>
      <actuatorfrc actuator="a0"/>
      <actuatorfrc actuator="a1"/>
      <accelerometer site="s0"/>
    </sensor>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  int nu = m->nu, ns = m->nsensordata;
  ASSERT_EQ(nu, 2);
  ASSERT_EQ(ns, 5);
  mjtNum eps = MjEps(1e-6, 1e-3);

  // interior and boundary controls: centered and fallback one-sided
  // differencing
  d->ctrl[1] = -0.3;
  vector<mjtNum> D(ns * nu);
  const vector<mjtNum> expected_D = {
      3, 0,   // actuatorfrc a0
      0, -4,  // actuatorfrc a1
      3, 0,   // accelerometer x
      0, 0,   // accelerometer y
      0, 0,   // accelerometer z
  };
  for (mjtNum ctrl0 : {mjtNum(0.2), mjtNum(1.0), mjtNum(-1.0)}) {
    d->ctrl[0] = ctrl0;
    mjd_transitionFD(m.get(), d.get(), eps, /*flg_centered=*/1, nullptr,
                     nullptr, nullptr, D.data());
    EXPECT_THAT(D, Pointwise(MjNear(1e-9, 5e-4), expected_D));
  }

  // beyond ctrlrange: clamped actuator column is zero
  d->ctrl[0] = 2.0;
  mjd_transitionFD(m.get(), d.get(), eps, /*flg_centered=*/1, nullptr, nullptr,
                   nullptr, D.data());
  const vector<mjtNum> expected_clamped_D = {0, 0, 0, -4, 0, 0, 0, 0, 0, 0};
  EXPECT_THAT(D, Pointwise(MjNear(1e-9, 5e-4), expected_clamped_D));
}

// if sensor derivatives aren't requested, don't compute sensors
TEST_F(DerivativeTest, SensorSkip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>

    <actuator>
      <general name="actuator" joint="joint" gainprm="3"/>
    </actuator>

    <sensor>
      <jointpos joint="joint"/>
    </sensor>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  int nv = model->nv, nu = model->nu;
  MjDataPtr data = MakeData(model);

  // set a sentinel value in the sensor
  data->sensordata[0] = 1337;

  // finite differenced B
  mjtNum eps = 1e-6;
  mjtNum* BFD = (mjtNum*)mju_malloc(sizeof(mjtNum) * 2 * nv * nu);
  mjd_transitionFD(model.get(), data.get(), eps, /*centered=*/0, nullptr, BFD,
                   nullptr, nullptr);

  EXPECT_EQ(data->sensordata[0], 1337) << "sensors should not be recomputed";

  mju_free(BFD);
}

// derivatives don't mutate the state
TEST_F(DerivativeTest, NoStateMutation) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data0 = mj_makeData(model);
  mjData* data = mj_makeData(model);
  int nv = model->nv, nu = model->nu, na = model->na, ns = model->nsensordata;

  // set time
  data->time = data0->time = 0.5;

  for (int i = 0; i < nv; i++) {
    data->qpos[i] = data0->qpos[i] = (mjtNum)i + 1;
    data->qvel[i] = data0->qvel[i] = (mjtNum)i + 2;
  }

  // set ctrl
  for (int i = 0; i < nu; i++) {
    data->ctrl[i] = data0->ctrl[i] = (mjtNum)i + 1;
  }

  // set act
  for (int i = 0; i < na; i++) {
    data->act[i] = data0->act[i] = (mjtNum)i + 1;
  }

  // allocate Jacobians, call derivatives
  int ndx = nv + nv + na;
  mjtNum* A = (mjtNum*)mju_malloc(sizeof(mjtNum) * ndx * ndx);
  mjtNum* B = (mjtNum*)mju_malloc(sizeof(mjtNum) * ndx * nu);
  mjtNum* C = (mjtNum*)mju_malloc(sizeof(mjtNum) * ns * ndx);
  mjtNum* D = (mjtNum*)mju_malloc(sizeof(mjtNum) * ns * nu);
  mjtNum eps = 1e-6;
  mjd_transitionFD(model, data, eps, /*centered=*/0, A, B, C, D);

  // compare states in data and data0
  EXPECT_EQ(data->time, data0->time);
  EXPECT_EQ(AsVector(data->qpos, model->nq), AsVector(data0->qpos, model->nq));
  EXPECT_EQ(AsVector(data->qvel, nv), AsVector(data0->qvel, nv));
  EXPECT_EQ(AsVector(data->act, na), AsVector(data0->act, na));
  EXPECT_EQ(AsVector(data->ctrl, nu), AsVector(data0->ctrl, nu));

  mju_free(D);
  mju_free(C);
  mju_free(B);
  mju_free(A);
  mj_deleteData(data0);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// finite differencing is not supported with sleeping enabled
TEST_F(DerivativeTest, FiniteDifferenceRejectsSleep) {
  const std::string xml_path = GetTestDataFilePath(kSleepEqualityPath);
  char error[1024];
  MjModelPtr model(mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error)));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  int nv = model->nv, ndx = 2 * nv + model->na;
  vector<mjtNum> A(ndx * ndx);
  vector<mjtNum> DfDq(nv * nv);
  mjtNum eps = 1e-6;

  // all trees are awake, but the evaluations would change the sleep state
  auto transition_error = MjuErrorMessageFrom(mjd_transitionFD);
  EXPECT_THAT(transition_error(model.get(), data.get(), eps, /*centered=*/1,
                               A.data(), nullptr, nullptr, nullptr),
              HasSubstr("sleeping is not supported"));
  auto inverse_error = MjuErrorMessageFrom(mjd_inverseFD);
  EXPECT_THAT(inverse_error(model.get(), data.get(), eps, /*flg_actuation=*/0,
                            DfDq.data(), nullptr, nullptr, nullptr, nullptr,
                            nullptr, nullptr),
              HasSubstr("sleeping is not supported"));

  // disabling sleep allows finite differencing
  model->opt.enableflags &= ~mjENBL_SLEEP;
  EXPECT_EQ(transition_error(model.get(), data.get(), eps, /*centered=*/1,
                             A.data(), nullptr, nullptr, nullptr),
            "");
  EXPECT_EQ(inverse_error(model.get(), data.get(), eps, /*flg_actuation=*/0,
                          DfDq.data(), nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr),
            "");
}

// compare dense and sparse derivatives of qfrc_bias (RNE)
TEST_F(DerivativeTest, DenseSparseRneEquivalent) {
  // run test on all models
  for (const char* local_path :
       {kEnergyConservingPendulumPath, kTumblingThinObjectPath,
        kDampedActuatorsPath, kDamperActuatorsPath, kDCMotorPath}) {
    const std::string xml_path = GetTestDataFilePath(local_path);
    char error[1024] = "";
    mjModel* model =
        mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(model, testing::NotNull()) << "Failed to load model: " << error;
    int nD = model->nD;
    mjtNum* qDeriv = (mjtNum*)mju_malloc(sizeof(mjtNum) * nD);
    mjData* data = mj_makeData(model);

    // take 100 steps so we have some velocities, then call forward
    mj_resetData(model, data);
    if (model->nu) {
      data->ctrl[0] = 0.1;
    }
    for (int i = 0; i < 100; i++) {
      mj_step(model, data);
    }
    mj_forward(model, data);

    // compute qDeriv with sparse function, make local copy
    mjd_smooth_vel(model, data, /*flg_bias=*/1);
    mju_copy(qDeriv, data->qDeriv, nD);

    // re-compute with dense function
    mju_zero(data->qDeriv, model->nD);
    mjd_actuator_vel(model, data);
    mjd_passive_vel(model, data);
    mjd_rne_vel_dense(model, data);

    // expect dense and sparse derivatives to be similar to precision
    EXPECT_THAT(AsVector(data->qDeriv, nD),
                Pointwise(MjNear(1e-12, 5e-5), AsVector(qDeriv, nD)));

    mju_free(qDeriv);
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

// compare FD inverse derivatives to analytic derivatives of linear system
TEST_F(DerivativeTest, LinearSystemInverse) {
  const std::string xml_path = GetTestDataFilePath(kLinearPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  int nv = model->nv;
  int ns = model->nsensordata;
  int nC = model->nC;

  vector<mjtNum> DfDq(nv * nv);
  vector<mjtNum> DfDv(nv * nv);
  vector<mjtNum> DfDa(nv * nv);
  vector<mjtNum> DsDq(nv * ns);
  vector<mjtNum> DsDv(nv * ns);
  vector<mjtNum> DsDa(nv * ns);
  vector<mjtNum> DmDq(nv * nC);

  // call mj_forward to get accelerations at initial state
  mj_forward(model, data);

  // get derivatives
  mjtNum eps = 1e-6;
  mjtByte flg_actuation = 0;
  mjd_inverseFD(model, data, eps, flg_actuation, DfDq.data(), DfDv.data(),
                DfDa.data(), DsDq.data(), DsDv.data(), DsDa.data(),
                DmDq.data());

  // expect that position derivatives are the stiffnesses
  vector<mjtNum> DfDq_expect = {model->jnt_stiffness[0], 0, 0, 0,
                                model->jnt_stiffness[1], 0, 0, 0,
                                model->jnt_stiffness[2]};
  EXPECT_THAT(DfDq, Pointwise(DoubleNear(eps), DfDq_expect));

  // expect that velocity derivatives are the dampings
  vector<mjtNum> DfDv_expect = {model->dof_damping[0], 0, 0, 0,
                                model->dof_damping[1], 0, 0, 0,
                                model->dof_damping[2]};
  EXPECT_THAT(DfDv, Pointwise(DoubleNear(eps), DfDv_expect));

  // expect that acceleration derivatives are the mass matrix
  vector<mjtNum> DfDa_expect(nv * nv, 0);
  mj_fullM(model, data, DfDa_expect.data());
  EXPECT_THAT(DfDa, Pointwise(DoubleNear(eps), DfDa_expect));

  // expect that sensor derivatives w.r.t position only see sensor 1 at dof 0
  vector<mjtNum> DsDq_expect(nv * ns, 0);
  int dof_index = 0;
  int sensordata_index = model->sensor_adr[1];
  DsDq_expect[dof_index * ns + sensordata_index] = 1;
  EXPECT_THAT(DsDq, Pointwise(DoubleNear(eps), DsDq_expect));

  // expect that sensor derivatives w.r.t velocity only see sensor 0 at dof 1
  vector<mjtNum> DsDv_expect(nv * ns, 0);
  dof_index = 1;
  sensordata_index = model->sensor_adr[0];
  DsDv_expect[dof_index * ns + sensordata_index] = 1;
  EXPECT_THAT(DsDv, Pointwise(DoubleNear(eps), DsDv_expect));

  // expect that sensor derivatives w.r.t acceleration see the accelerometer
  // in the y-axis, affected by both dof 0 and dof 1
  vector<mjtNum> DsDa_expect(nv * ns, 0);
  dof_index = 0;
  sensordata_index = model->sensor_adr[2] + 1;
  DsDa_expect[dof_index * ns + sensordata_index] = 1;
  dof_index = 1;
  DsDa_expect[dof_index * ns + sensordata_index] = 1;
  EXPECT_THAT(DsDa, Pointwise(DoubleNear(eps), DsDa_expect));

  // expect that mass matrix derivatives are zero
  vector<mjtNum> DmDq_expect(nv * nC, 0);
  EXPECT_THAT(DmDq, Pointwise(DoubleNear(eps), DmDq_expect));

  mj_deleteData(data);
  mj_deleteModel(model);
}

// mjd_inverseFD updates actuation before acceleration sensors when
// flg_actuation=1
TEST_F(DerivativeTest, InverseFDActuationSensor) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="2"/>
      </body>
    </worldbody>
    <actuator>
      <general name="act" joint="j" gaintype="fixed" gainprm="1"
               biastype="affine" biasprm="0 -10 -4"/>
    </actuator>
    <sensor>
      <actuatorfrc actuator="act"/>
      <jointactuatorfrc joint="j"/>
    </sensor>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  d->qpos[0] = 0.3;
  d->qvel[0] = -0.7;
  d->qacc[0] = 1.2;
  d->ctrl[0] = 2.0;

  int nv = m->nv, ns = m->nsensordata;
  vector<mjtNum> DfDq(nv * nv), DfDv(nv * nv), DfDa(nv * nv);
  vector<mjtNum> DsDq(nv * ns), DsDv(nv * ns), DsDa(nv * ns);
  mjtNum eps = MjTol(1e-6, 1e-3);

  mjd_inverseFD(m.get(), d.get(), eps, /*flg_actuation=*/1, DfDq.data(),
                DfDv.data(), DfDa.data(), DsDq.data(), DsDv.data(), DsDa.data(),
                nullptr);

  // force = m*qacc - (ctrl - 10*qpos - 4*qvel)
  // dF/dq = 10, dF/dv = 4, dF/da = 2
  EXPECT_NEAR(DfDq[0], 10.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DfDv[0], 4.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DfDa[0], 2.0, MjTol(1e-6, 1e-2));

  // actuatorfrc = jointactuatorfrc = ctrl - 10*qpos - 4*qvel
  // ds/dq = -10, ds/dv = -4, ds/da = 0
  EXPECT_NEAR(DsDq[0], -10.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DsDq[1], -10.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DsDv[0], -4.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DsDv[1], -4.0, MjTol(1e-6, 1e-2));
  EXPECT_NEAR(DsDa[0], 0.0, MjTol(1e-8, 1e-4));
  EXPECT_NEAR(DsDa[1], 0.0, MjTol(1e-8, 1e-4));
}

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

// utility: finite-difference Jacobians of mju_subQuat
static void subQuatFD(mjtNum Da[9], mjtNum Db[9], const mjtNum qa[4],
                      const mjtNum qb[4], mjtNum eps) {
  // subQuat
  mjtNum y[3];
  mju_subQuat(y, qa, qb);

  mjtNum dq[3];   // nudge input direction
  mjtNum dqa[4];  // nudged qa input
  mjtNum dqb[4];  // nudged qb input
  mjtNum dy[3];   // nudged output
  mjtNum DaT[9];  // Da transposed
  mjtNum DbT[9];  // Db transposed

  for (int i = 0; i < 3; i++) {
    // perturbation
    mju_zero3(dq);
    dq[i] = 1.0;

    // Jacobian: d_y / d_qa
    mju_copy4(dqa, qa);
    mju_quatIntegrate(dqa, dq, eps);
    mju_subQuat(dy, dqa, qb);

    mju_sub3(DaT + i * 3, dy, y);
    mju_scl3(DaT + i * 3, DaT + i * 3, 1.0 / eps);

    // Jacobian: d_y / d_qb
    mju_copy4(dqb, qb);
    mju_quatIntegrate(dqb, dq, eps);
    mju_subQuat(dy, qa, dqb);

    mju_sub3(DbT + i * 3, dy, y);
    mju_scl3(DbT + i * 3, DbT + i * 3, 1.0 / eps);
  }

  // transpose result
  mju_transpose(Da, DaT, 3, 3);
  mju_transpose(Db, DbT, 3, 3);
}

TEST_F(DerivativeTest, SubQuat) {
  const int nrepeats = 10;  // number of repeats
  const mjtNum eps =
      MjTol(1e-7, 1e-3);  // epsilon for finite-differencing and comparison

  int seed = 1;
  for (int i = 0; i < nrepeats; i++) {
    for (mjtNum angle : {0.0, 1e-9, 1e-5, 1e-2, 1.0, 4.0}) {
      // random quaternions
      mjtNum qa[4];
      mjtNum qb[4];

      // make random quaternion pair with given relative angle
      randomQuatPair(qa, qb, angle, seed++);

      // analytic Jacobians
      mjtNum Da[9];  // d_subQuat(qa, qb) / d_qa
      mjtNum Db[9];  // d_subQuat(qa, qb) / d_qb
      mjd_subQuat(qa, qb, Da, Db);

      // finite-differenced Jacobians
      mjtNum DaFD[9];
      mjtNum DbFD[9];
      subQuatFD(DaFD, DbFD, qa, qb, eps);

      // expect numerical equality
      EXPECT_THAT(AsVector(DaFD, 9),
                  Pointwise(MjNear(1e-7, 1e-3), AsVector(Da, 9)));
      EXPECT_THAT(AsVector(DbFD, 9),
                  Pointwise(MjNear(1e-7, 1e-3), AsVector(Db, 9)));
    }
  }
}

// utility: random quaternion, 3D velocity
static void randomQuatVel(mjtNum quat[4], mjtNum vel[3], int seed) {
  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(seed);
  std::normal_distribution<double> dist(0, 1);

  // sample quat
  for (int i = 0; i < 4; i++) {
    quat[i] = dist(rng);
  }
  mju_normalize4(quat);

  // sample vel
  for (int i = 0; i < 3; i++) {
    vel[i] = dist(rng);
  }
}

// utility: finite-difference Jacobians of mju_quatIntegrate
void mjd_quatIntegrateFD(mjtNum Dquat[9], mjtNum Ds[9], mjtNum Dvel[9],
                         mjtNum Dh[3], const mjtNum quat[4],
                         const mjtNum vel[3], mjtNum h, mjtNum eps) {
  // compute y, output of mju_quatIntegrate(quat, vel, h)
  mjtNum y[4] = {quat[0], quat[1], quat[2], quat[3]};
  mju_quatIntegrate(y, vel, h);

  mjtNum dx[3];      // nudged tangent-space input
  mjtNum dq[4];      // quat output
  mjtNum dy[3];      // nudged tangent-space output
  mjtNum DquatT[9];  // Dquat transposed
  mjtNum DsT[9];     // Ds transposed
  mjtNum DvelT[9];   // Dvel transposed

  for (int i = 0; i < 3; i++) {
    // perturbation
    mju_zero3(dx);
    dx[i] = 1.0;

    // d_y / d_quat
    mju_copy4(dq, quat);
    mju_quatIntegrate(dq, dx, eps);  // nudge dq
    mju_quatIntegrate(dq, vel, h);   // compute nudged
    mju_subQuat(dy, dq, y);          // subtract
    mju_scl3(DquatT + i * 3, dy, 1.0 / eps);

    // d_y / d_sv (scaled velocity)
    mju_copy4(dq, quat);
    mjtNum dsv[3] = {vel[0] * h, vel[1] * h, vel[2] * h};
    mju_addToScl3(dsv, dx, eps);      // nudge dsv
    mju_quatIntegrate(dq, dsv, 1.0);  // compute nudged
    mju_subQuat(dy, dq, y);           // subtract
    mju_scl3(DsT + i * 3, dy, 1.0 / eps);

    // d_y / d_v (unscaled velocity)
    mju_copy4(dq, quat);
    mjtNum dv[3] = {vel[0], vel[1], vel[2]};
    mju_addToScl3(dv, dx, eps);    // nudge dv
    mju_quatIntegrate(dq, dv, h);  // compute nudged
    mju_subQuat(dy, dq, y);        // subtract
    mju_scl3(DvelT + i * 3, dy, 1.0 / eps);
  }

  // d_y / d_h (unscaled velocity)
  mju_copy4(dq, quat);
  mju_quatIntegrate(dq, vel, h + eps);  // compute nudged
  mju_subQuat(dy, dq, y);               // subtract
  mju_scl3(Dh, dy, 1.0 / eps);

  // transpose
  mju_transpose(Dquat, DquatT, 3, 3);
  mju_transpose(Ds, DsT, 3, 3);
  mju_transpose(Dvel, DsT, 3, 3);
}

TEST_F(DerivativeTest, quatIntegrate) {
  const int nrepeats = 10;  // number of repeats
  const mjtNum eps =
      MjTol(1e-7, 1e-3);  // epsilon for finite-differencing and comparison

  int seed = 1;
  for (int i = 0; i < nrepeats; i++) {
    for (mjtNum h : {0.0, 1e-9, 1e-5, 1e-2, 1.0, 4.0}) {
      // make random quaternion and velocity
      mjtNum quat[4];
      mjtNum vel[3];
      randomQuatVel(quat, vel, seed++);

      // analytic Jacobians
      mjtNum Dquat[9];  // d_quatIntegrate(quat, vel, h) / d_quat
      mjtNum Dvel[9];   // d_quatIntegrate(quat, vel, h) / d_vel
      mjtNum Dh[3];     // d_quatIntegrate(quat, vel, h) / d_h
      mjd_quatIntegrate(vel, h, Dquat, Dvel, Dh);

      // finite-differenced Jacobians
      mjtNum DquatFD[9];
      mjtNum DsFD[9];
      mjtNum DvelFD[9];
      mjtNum DhFD[3];
      mjd_quatIntegrateFD(DquatFD, DsFD, DvelFD, DhFD, quat, vel, h, eps);

      // expect numerical equality of un/scaled velocity derivatives
      EXPECT_THAT(AsVector(DvelFD, 9), Pointwise(MjNear(1e-7, 1e-2), DsFD));

      // expect numerical equality of analytic and FD derivatives
      EXPECT_THAT(AsVector(DquatFD, 9), Pointwise(MjNear(1e-7, 1e-2), Dquat));
      EXPECT_THAT(AsVector(DvelFD, 9), Pointwise(MjNear(1e-7, 1e-2), Dvel));
      EXPECT_THAT(AsVector(DhFD, 3), Pointwise(MjNear(1e-7, 1e-2), Dh));
    }
  }
}

// implicit integration is better than Euler with active forcerange clamping
TEST_F(DerivativeTest, ForcerangeClampedDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="implicitfast"/>

    <worldbody>
      <geom name="plane" type="plane" size="2 2 0.1"/>
      <light pos="0 0 3"/>
      <body name="1" pos="0 0 1">
        <joint name="1" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <position joint="1" kp="10000" kv="1000" forcerange="-10 10"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  mjtNum dt_small = 1e-4;
  mjtNum dt_large = 1e-2;
  mjtNum duration = 1.0;

  MjDataPtr d_gt = MakeData(m);
  MjDataPtr d_implicit = MakeData(m);
  MjDataPtr d_euler = MakeData(m);
  MjDataPtr d_discrete = MakeData(m);

  mj_resetData(m.get(), d_gt.get());
  mj_resetData(m.get(), d_implicit.get());
  mj_resetData(m.get(), d_euler.get());
  mj_resetData(m.get(), d_discrete.get());

  d_gt.get()->ctrl[0] = 0.5;
  d_implicit->ctrl[0] = 0.5;
  d_euler->ctrl[0] = 0.5;
  d_discrete->ctrl[0] = 0.5;

  mjtNum error_implicit = 0;
  mjtNum error_discrete = 0;
  mjtNum error_euler = 0;
  int nsteps_large = static_cast<int>(duration / dt_large);
  int substeps = static_cast<int>(dt_large / dt_small);

  m->opt.timestep = dt_large;

  m->opt.integrator = mjINT_IMPLICITFAST;
  mj_resetData(m.get(), d_gt.get());
  d_gt.get()->ctrl[0] = 0.5;
  m->opt.timestep = dt_small;
  m->opt.integrator = mjINT_EULER;

  for (int i = 0; i < nsteps_large; i++) {
    // ground truth: small steps with Euler
    m->opt.integrator = mjINT_EULER;
    m->opt.timestep = dt_small;
    for (int j = 0; j < substeps; j++) {
      mj_step(m.get(), d_gt.get());
    }

    // euler at large timestep
    m->opt.timestep = dt_large;
    mj_step(m.get(), d_euler.get());

    // implicitfast at large timestep
    m->opt.integrator = mjINT_IMPLICITFAST;
    mj_step(m.get(), d_implicit.get());

    // discrete at large timestep
    m->opt.integrator = mjINT_DISCRETE;
    mj_step(m.get(), d_discrete.get());

    // accumulate errors
    mjtNum diff_implicit = d_gt.get()->qpos[0] - d_implicit->qpos[0];
    mjtNum diff_euler = d_gt.get()->qpos[0] - d_euler->qpos[0];
    mjtNum diff_discrete = d_gt.get()->qpos[0] - d_discrete->qpos[0];
    error_implicit += diff_implicit * diff_implicit;
    error_euler += diff_euler * diff_euler;
    error_discrete += diff_discrete * diff_discrete;
  }

  // expect implicitfast to be more accurate than Euler
  EXPECT_LT(error_implicit, error_euler)
      << "implicitfast should be more accurate than Euler at large timestep "
      << "when forcerange derivatives are correctly handled";

  // the discrete arm: while the force is clamped, actuatorDerivSkip keeps the
  // gains out of the metric and discrete coincides with Euler; unclamped
  // stretches trade Euler's explicit overshoot for the metric's implicit
  // damping, so the errors are near-equal with a platform-dependent sign.
  // Assert comparable accuracy; the stability advantage at large h*omega is
  // pinned by DiscreteStiffLegPress
  EXPECT_LT(error_discrete, 1.5 * error_euler)
      << "discrete should be comparable to Euler under a clamped servo";
}

// under discrete, qacc is the step map: finite differences of mj_step through
// mjd_transitionFD (which reuses stages via mj_stepSkip) must match naive
// finite differences with a full reset and a fresh step per perturbation
TEST_F(DerivativeTest, DiscreteStepMapDerivatives) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete"/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body pos="0 0 0.079">
        <joint name="slide" type="slide" axis="0 0 1" stiffness="2000" damping="10" springref="-0.02"/>
        <joint name="hinge" type="hinge" axis="0 1 0" stiffness="500" damping="2"/>
        <geom type="capsule" size="0.04" fromto="0 0 0 0.2 0 -0.04"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  int nv = m->nv, nu = m->nu, ns = 2 * nv;

  // state with an active contact and nonzero velocity
  const mjtNum qpos0[2] = {-0.01, 0.05};
  const mjtNum qvel0[2] = {-0.2, 0.3};
  const mjtNum ctrl0[1] = {0.5};
  mju_copy(d->qpos, qpos0, nv);
  mju_copy(d->qvel, qvel0, nv);
  mju_copy(d->ctrl, ctrl0, nu);
  mj_forward(m.get(), d.get());
  ASSERT_GT(d->ncon, 0);

  // transition derivatives via stage-reusing finite differences
  // (eps is a perturbation size, not a tolerance: independent of MJTOL_SCALE)
  mjtNum eps = sizeof(mjtNum) == 8 ? 1e-6 : 1e-3;
  std::vector<mjtNum> A(ns * ns), B(ns * nu);
  mjd_transitionFD(m.get(), d.get(), eps, /*flg_centered=*/1, A.data(),
                   B.data(), nullptr, nullptr);

  // naive centered differences: full reset and one fresh mj_step per
  // perturbation
  auto naive = [&](int what, int idx, mjtNum* dx) {
    mjtNum x[2][4];
    for (int sgn = 0; sgn < 2; sgn++) {
      mj_resetData(m.get(), d.get());
      mju_copy(d->qpos, qpos0, nv);
      mju_copy(d->qvel, qvel0, nv);
      mju_copy(d->ctrl, ctrl0, nu);
      mjtNum e = sgn ? eps : -eps;
      if (what == 0) d->qpos[idx] += e;
      if (what == 1) d->qvel[idx] += e;
      if (what == 2) d->ctrl[idx] += e;
      mj_step(m.get(), d.get());
      mju_copy(x[sgn], d->qpos, nv);
      mju_copy(x[sgn] + nv, d->qvel, nv);
    }
    for (int i = 0; i < ns; i++) {
      dx[i] = (x[1][i] - x[0][i]) / (2 * eps);
    }
  };

  // guard against a vacuous pass: A contains the identity block and h-scale
  // dynamics
  EXPECT_GT(mju_norm(A.data(), ns * ns), 1);

  mjtNum maxdiff = 0;
  mjtNum dx[4];
  for (int j = 0; j < ns; j++) {
    naive(j < nv ? 0 : 1, j < nv ? j : j - nv, dx);
    for (int i = 0; i < ns; i++) {
      maxdiff = mju_max(maxdiff, mju_abs(A[i * ns + j] - dx[i]));
    }
  }
  for (int j = 0; j < nu; j++) {
    naive(2, j, dx);
    for (int i = 0; i < ns; i++) {
      maxdiff = mju_max(maxdiff, mju_abs(B[i * nu + j] - dx[i]));
    }
  }
  EXPECT_LT(maxdiff, MjTol(1e-8, 2e-5));
}

// forcelimited actuator following a multi-output SO3 actuator: the derivative
// skip for saturated actuators must index forcerange per actuator, not per
// output.
TEST_F(DerivativeTest, ForcerangeClampedAfterSO3) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag contact="disable" gravity="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <geom type="box" size=".05 .07 .03"/>
      </body>
      <body pos="0 0 .3">
        <joint name="hinge"/>
        <geom size=".05"/>
      </body>
    </worldbody>
    <actuator>
      <orientation joint="ball" kp="1" kv="1"/>
      <velocity joint="hinge" kv="10" forcerange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();

  // spin the hinge so the velocity actuator saturates: force -50, clamped -1
  mjtNum qvel[4] = {0.1, 0.2, 0.3, 5};
  mju_copy(d->qvel, qvel, 4);
  mj_forward(m, d);
  ASSERT_EQ(d->actuator_force[3], -1);

  // analytic qDeriv
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m, d, /*flg_bias=*/1);
  vector<mjtNum> qDerivAnalytic = AsVector(d->qDeriv, m->nD);
  EXPECT_GT(mju_norm(qDerivAnalytic.data(), m->nD), 0);

  // expect match with finite differences: the saturated actuator contributes
  // nothing, the SO3 actuator's damping is unaffected by its neighbor
  mjtNum eps = MjEps(1e-7, 1e-3);
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m, d, eps);
  EXPECT_THAT(AsVector(d->qDeriv, m->nD),
              Pointwise(MjNear(1e-7, 3e-3), qDerivAnalytic));
}

// Validate that mjd_actuatorDerivSkip skips joint and jointinparent actuators
// saturated by joint actuatorfrcrange (both without and with actuatorgravcomp)
// across implicitfast, implicit, and discrete integrators.
TEST_F(DerivativeTest, JointActfrcrangeSaturation) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 -10"/>
    <worldbody>
      <body>
        <joint name="j0" type="hinge" axis="0 1 0" damping="1.5"/>
        <geom type="sphere" pos="0.5 0 0" size="0.05" mass="1"/>
        <body name="b1" pos="0.5 0 0" gravcomp="1">
          <joint name="j1" type="hinge" axis="0 1 0" damping="2"
                 actuatorgravcomp="true"
                 actuatorfrclimited="true" actuatorfrcrange="-12 12"/>
          <geom type="sphere" pos="1 0 0" size="0.05" mass="1"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <position joint="j0" kp="20" kv="3"/>
      <position joint="j1" kp="500" kv="50" forcerange="-20 20"/>
      <general jointinparent="j1" biastype="affine" gainprm="100"
        biasprm="0 -100 -10"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  mjModel* m = model.get();

  const int j1_id = mj_name2id(m, mjOBJ_JOINT, "j1");
  const int b1_id = mj_name2id(m, mjOBJ_BODY, "b1");
  ASSERT_GT(j1_id, 0);
  const int dof1 = m->jnt_dofadr[j1_id];
  const int diag1 = m->D_rowadr[dof1] + m->D_diag[dof1];

  for (bool use_actgravcomp : {false, true}) {
    SCOPED_TRACE(use_actgravcomp ? "with_actgravcomp" : "without_actgravcomp");
    m->body_gravcomp[b1_id] = use_actgravcomp ? 1.0 : 0.0;
    m->jnt_actgravcomp[j1_id] = use_actgravcomp ? 1 : 0;

    // At qpos = 0, qvel = {0.2, -0.1}:
    // - without actgravcomp: qfrc_gravcomp[1] = 0
    // - with actgravcomp:    qfrc_gravcomp[1] = -10
    // Actuators 1 & 2 contribute 500*u1 + 5 + 100*u2 + 1 = 500*u1 + 100*u2 + 6.
    struct Case {
      const char* name;
      mjtNum ctrl1;
      mjtNum ctrl2;
      bool clamped;
      mjtNum expected_jnt_frc;
    };
    const Case cases[] = {
        // Unsaturated:
        // without gravcomp: 0 + 6 = 6 in (-12, 12)
        // with gravcomp:    5 + 1 + 6 - 10 = 2 in (-12, 12)
        {"unsaturated", use_actgravcomp ? mjtNum(0.01) : mjtNum(0.0),
         use_actgravcomp ? mjtNum(0.01) : mjtNum(0.0), false,
         use_actgravcomp ? mjtNum(2.0) : mjtNum(6.0)},
        // Upper-saturated (individual actuator 1 force <= 19, inside [-20,
        // 20]):
        // without gravcomp: 10 + 3 + 6 = 19 -> clamped to +12
        // with gravcomp:    14 + 5 = 19 on act1, 5 + 1 = 6 on act2 => 25 - 10 =
        // 15 -> +12
        {"upper_clamped", use_actgravcomp ? mjtNum(0.028) : mjtNum(0.02),
         use_actgravcomp ? mjtNum(0.05) : mjtNum(0.03), true, mjtNum(12.0)},
        // Lower-saturated (individual actuator 1 force >= -10, inside [-20,
        // 20]):
        // without gravcomp: -15 - 5 + 6 = -14 -> clamped to -12
        // with gravcomp:    -10 + 5 = -5 on act1, -1 + 1 = 0 on act2 => -5 - 10
        // = -15 -> -12
        {"lower_clamped", use_actgravcomp ? mjtNum(-0.02) : mjtNum(-0.03),
         use_actgravcomp ? mjtNum(-0.01) : mjtNum(-0.05), true, mjtNum(-12.0)},
    };

    for (const Case& c : cases) {
      SCOPED_TRACE(c.name);
      for (mjtIntegrator integ :
           {mjINT_IMPLICITFAST, mjINT_IMPLICIT, mjINT_DISCRETE}) {
        SCOPED_TRACE(integ);
        m->opt.integrator = integ;

        MjDataPtr data = MakeData(model);
        mjData* d = data.get();
        d->qvel[0] = 0.2;
        d->qvel[1] = -0.1;
        d->ctrl[0] = 0.1;
        d->ctrl[1] = c.ctrl1;
        d->ctrl[2] = c.ctrl2;
        mj_forward(m, d);

        // Individual actuator 1 must NOT be clamped by its own [-20, 20] range
        ASSERT_GT(d->actuator_force[1], -19.5);
        ASSERT_LT(d->actuator_force[1], 19.5);
        EXPECT_NEAR(d->qfrc_actuator[dof1], c.expected_jnt_frc,
                    MjTol(1e-10, 1e-4));

        if (integ == mjINT_DISCRETE) {
          EXPECT_EQ(d->nefmA, c.clamped ? 1 : 3);
        }

        // Compare mjd_smooth_vel against mjd_smooth_velFD and mjd_passive_velFD
        mjtNum eps = MjEps(1e-7, 1e-3);
        int flg_bias = (integ == mjINT_IMPLICIT) ? 1 : 0;
        mju_zero(d->qDeriv, m->nD);
        mjd_smooth_vel(m, d, flg_bias);
        vector<mjtNum> qDerivAnalytic = AsVector(d->qDeriv, m->nD);

        if (flg_bias) {
          mju_zero(d->qDeriv, m->nD);
          mjd_smooth_velFD(m, d, eps);
          EXPECT_THAT(AsVector(d->qDeriv, m->nD),
                      Pointwise(MjNear(1e-6, 1e-2), qDerivAnalytic));
        } else {
          // Compare passive-only FD on dof1 diagonal: when clamped, actuator
          // velocity derivative (-60) is dropped so qDeriv[diag1] equals
          // passive
          mju_zero(d->qDeriv, m->nD);
          mjd_passive_velFD(m, d, eps);
          mjtNum expected_diag1 =
              d->qDeriv[diag1] + (c.clamped ? mjtNum(0) : mjtNum(-60));
          EXPECT_NEAR(qDerivAnalytic[diag1], expected_diag1, MjTol(1e-6, 1e-2));
        }

        // When saturated, transition Jacobians A and B across all 3 integrators
        // must match a constant-torque reference model (no phantom kp/kv).
        if (c.clamped) {
          mj_forward(m, d);
          const int ndx = 2 * m->nv;
          vector<mjtNum> A(ndx * ndx), B(ndx * m->nu);
          mjd_transitionFD(m, d, MjEps(1e-6, 1e-3), /*flg_centered=*/1,
                           A.data(), B.data(), nullptr, nullptr);

          // Control columns for saturated joint actuators must vanish
          for (int row = 0; row < ndx; ++row) {
            EXPECT_NEAR(B[row * m->nu + 1], 0.0, MjTol(1e-9, 1e-4));
            EXPECT_NEAR(B[row * m->nu + 2], 0.0, MjTol(1e-9, 1e-4));
          }

          // Build constant-torque reference model with j1 actuators zeroed
          MjModelPtr ref_model =
              LoadModelFromString(kXml, error, sizeof(error));
          mjModel* m_ref = ref_model.get();
          m_ref->opt.integrator = integ;
          m_ref->body_gravcomp[b1_id] = m->body_gravcomp[b1_id];
          m_ref->jnt_actgravcomp[j1_id] = m->jnt_actgravcomp[j1_id];
          m_ref->jnt_actfrcrange[2 * j1_id] = c.expected_jnt_frc;
          m_ref->jnt_actfrcrange[2 * j1_id + 1] = c.expected_jnt_frc;
          mju_zero(m_ref->actuator_gainprm + mjNGAIN * 1, mjNGAIN * 2);
          mju_zero(m_ref->actuator_biasprm + mjNBIAS * 1, mjNBIAS * 2);

          MjDataPtr ref_data = MakeData(ref_model);
          mjData* d_ref = ref_data.get();
          d_ref->qvel[0] = 0.2;
          d_ref->qvel[1] = -0.1;
          d_ref->ctrl[0] = 0.1;
          mj_forward(m_ref, d_ref);

          vector<mjtNum> A_ref(ndx * ndx);
          mjd_transitionFD(m_ref, d_ref, MjEps(1e-6, 1e-3), /*flg_centered=*/1,
                           A_ref.data(), nullptr, nullptr, nullptr);
          EXPECT_THAT(A, Pointwise(MjNear(1e-6, 5e-3), A_ref));
        }
      }
    }
  }
}

TEST_F(DerivativeTest, NonlinearDampingDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="slide" damping="2 3 4"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qvel="3"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  mjtNum dt_small = 1e-4;
  mjtNum dt_large = 1e-2;
  mjtNum duration = 1.0;

  MjDataPtr d_gt = MakeData(m);
  MjDataPtr d_enabled = MakeData(m);
  MjDataPtr d_disabled = MakeData(m);

  mj_resetDataKeyframe(m.get(), d_gt.get(), 0);
  mj_resetDataKeyframe(m.get(), d_enabled.get(), 0);
  mj_resetDataKeyframe(m.get(), d_disabled.get(), 0);

  m->opt.integrator = mjINT_EULER;
  mjtNum error_enabled = 0;
  mjtNum error_disabled = 0;
  int nsteps_large = static_cast<int>(duration / dt_large);
  int substeps = static_cast<int>(dt_large / dt_small);

  for (int i = 0; i < nsteps_large; i++) {
    m->opt.timestep = dt_small;
    m->opt.disableflags |= mjDSBL_EULERDAMP;  // disable implicit damping
    for (int j = 0; j < substeps; j++) {
      mj_step(m.get(), d_gt.get());
    }

    m->opt.timestep = dt_large;
    mj_step(m.get(), d_disabled.get());

    m->opt.disableflags &= ~mjDSBL_EULERDAMP;  // enable implicit damping
    mj_step(m.get(), d_enabled.get());

    mjtNum diff_enabled = d_gt.get()->qvel[0] - d_enabled.get()->qvel[0];
    mjtNum diff_disabled = d_gt.get()->qvel[0] - d_disabled.get()->qvel[0];
    error_enabled += diff_enabled * diff_enabled;
    error_disabled += diff_disabled * diff_disabled;
  }

  EXPECT_LT(error_enabled, error_disabled)
      << "Euler with implicit damping should be more accurate than without "
      << "when nonlinear damping derivatives are correctly handled";
}

// implicit derivatives should use next activation when actearly is set
TEST_F(DerivativeTest, ActearlyDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1" integrator="implicitfast"/>

    <worldbody>
      <body>
        <joint name="early" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body pos="1 0 0">
        <joint name="late" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="early" dyntype="integrator" gaintype="affine"
               gainprm="1 0 1" actearly="true"/>
      <general joint="late" dyntype="integrator" gaintype="affine"
               gainprm="1 0 1" actearly="false"/>
    </actuator>
  </mujoco>
  )";

  // the affine velocity gain of 1 makes M - h*qDeriv exactly singular at h=1:
  // expect a clamped-pivot warning from the implicitfast qH factorization
  mock_warning_handler.ExpectWarnings(
      "Inertia matrix is too close to singular");

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // set identical ctrl with zero initial activation
  d->ctrl[0] = 1.0;
  d->ctrl[1] = 1.0;
  d->act[0] = 0.0;
  d->act[1] = 0.0;

  // step computes derivatives during implicit integration
  mj_step(m.get(), d.get());

  // both should have same act_dot
  EXPECT_EQ(d->act_dot[0], d->act_dot[1]);

  // with actearly=true and nonzero act_dot, derivative should differ
  // because actearly uses next activation: act + act_dot*dt
  // for our model: next_act = 0 + 1*1 = 1, current_act = 0
  // derivative adds gain_vel * act to qDeriv diagonal
  // for independent bodies, D is diagonal, so diag[i] is at D_rowadr[i]
  int diag0 = m->D_rowadr[0];  // first joint's diagonal
  int diag1 = m->D_rowadr[1];  // second joint's diagonal
  EXPECT_NE(d->qDeriv[diag0], d->qDeriv[diag1])
      << "actearly=true should use next activation in derivative";

  // verify specific values: gain_vel=1, next_act=1, current_act=0
  EXPECT_NEAR(d->qDeriv[diag0], 1.0, 1e-10)
      << "actearly=true should use next_act=1";
  EXPECT_NEAR(d->qDeriv[diag1], 0.0, 1e-10)
      << "actearly=false should use current_act=0";
}

// verify stateful DC motor derivative matches analytical formula
TEST_F(DerivativeTest, DCMotorStatefulDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor name="dc" joint="j" motorconst="2.0" resistance="0.5"
               inductance="0 0.001" input="pos vel" controller="10 0 5"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // set nonzero velocity and ctrl
  d->qvel[0] = 1.0;
  d->ctrl[0] = 0.5;

  // forward to compute act_dot, etc.
  mj_forward(m.get(), d.get());

  // compute analytical derivatives
  mjd_smooth_vel(m.get(), d.get(), /* flg_bias = */ 1);

  // extract diagonal of qDeriv
  mjtNum qDeriv_diag = d->qDeriv[m->D_rowadr[0] + m->D_rownnz[0] - 1];

  // expected: K*(dVdw - K)*(1 - exp(-h/te))/R with the torque-space map
  // dVdw = -kd*R/K + K, so the expression reduces to -kd*(1 - exp(-h/te))
  mjtNum te = 0.001, h = 0.002, kd = 5.0;
  mjtNum expected = -kd * (1 - mju_exp(-h / te));
  EXPECT_NEAR(qDeriv_diag, expected, 1e-10)
      << "stateful DC motor derivative should match analytical formula";
}

// verify that stateful DC motor derivative converges to stateless as te -> 0
TEST_F(DerivativeTest, DCMotorStatefulConvergesToStateless) {
  // stateless DC motor with position controller
  static constexpr char xml_stateless[] = R"(
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor name="dc" joint="j" motorconst="1.0" resistance="1.0"
               input="pos vel" controller="10 0 5"/>
    </actuator>
  </mujoco>
  )";

  // stateful DC motor with very small te
  static constexpr char xml_stateful[] = R"(
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor name="dc" joint="j" motorconst="1.0" resistance="1.0"
               inductance="0 1e-8" input="pos vel" controller="10 0 5"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m_sl = LoadModelFromString(xml_stateless, error, sizeof(error));
  ASSERT_THAT(m_sl.get(), NotNull()) << error;
  MjDataPtr d_sl = MakeData(m_sl);

  MjModelPtr m_sf = LoadModelFromString(xml_stateful, error, sizeof(error));
  ASSERT_THAT(m_sf.get(), NotNull()) << error;
  MjDataPtr d_sf = MakeData(m_sf);

  // set identical state
  d_sl.get()->qvel[0] = d_sf.get()->qvel[0] = 1.0;
  d_sl.get()->ctrl[0] = d_sf.get()->ctrl[0] = 0.5;

  // forward and compute derivatives
  mj_forward(m_sl.get(), d_sl.get());
  mj_forward(m_sf.get(), d_sf.get());
  mjd_smooth_vel(m_sl.get(), d_sl.get(), 1);
  mjd_smooth_vel(m_sf.get(), d_sf.get(), 1);

  // extract diagonals
  mjtNum diag_sl =
      d_sl.get()->qDeriv[m_sl->D_rowadr[0] + m_sl->D_rownnz[0] - 1];
  mjtNum diag_sf =
      d_sf.get()->qDeriv[m_sf->D_rowadr[0] + m_sf->D_rownnz[0] - 1];

  EXPECT_NEAR(diag_sf, diag_sl, 1e-6)
      << "stateful derivative should converge to stateless as te -> 0";
}

// verify hot winding resistance is used for stateless and stateful derivatives
TEST_F(DerivativeTest, DCMotorThermalDerivative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="stateless" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="stateful" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="stateless" motorconst="1" resistance="1"
               thermal="1 1 0 0.004 25 25"/>
      <dcmotor joint="stateful" motorconst="1" resistance="1"
               inductance="0 0.01" thermal="1 1 0 0.004 25 25"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // A 250-degree rise doubles both winding resistances.
  d->act[m->actuator_actadr[0]] = 250;
  d->act[m->actuator_actadr[1]] = 250;
  d->qvel[0] = 0.5;
  d->qvel[1] = 0.5;
  mj_forward(m.get(), d.get());

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  vector<mjtNum> analytic = AsVector(d->qDeriv, m->nD);

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_THAT(AsVector(d->qDeriv, m->nD),
              Pointwise(MjNear(1e-7, 3e-3), analytic));
}

// verify hot resistance and cogging torque in DC motor length derivatives
TEST_F(DerivativeTest, DCMotorLenDerivThermalAndCogging) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="stateless" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="stateful" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="stateless" motorconst="1" resistance="1"
               input="pos" controller="30 0 0" cogging="1.5 2 0.3"
               thermal="1 1 0 0.004 25 25"/>
      <dcmotor joint="stateful" motorconst="1" resistance="1"
               inductance="0 0.002" input="pos" controller="30 0 0"
               cogging="1.5 2 0.3" thermal="1 1 0 0.004 25 25"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // 250-degree rise doubles winding resistance (R = 2 * R0)
  d->act[m->actuator_actadr[0]] = 250;
  d->act[m->actuator_actadr[1]] = 250;
  d->qpos[0] = 0.2;
  d->qpos[1] = 0.2;
  d->ctrl[0] = 0.5;
  d->ctrl[1] = 0.5;
  mj_forward(m.get(), d.get());

  // under discrete integrator, efm_ak = -h * mjd_actuatorLenDeriv when
  // stabilizing (< 0)
  ASSERT_EQ(d->nefmA, 2);
  mjtNum eps = MjTol(1e-6, 1e-3);
  for (int t = 0; t < 2; t++) {
    int i = d->efm_aid[t];
    mjtNum len_deriv_analytic = -d->efm_ak[t] / m->opt.timestep;

    MjDataPtr dp = MakeData(m), dm = MakeData(m);
    mj_copyData(dp.get(), m.get(), d.get());
    mj_copyData(dm.get(), m.get(), d.get());
    dp->qpos[i] += eps;
    dm->qpos[i] -= eps;
    mj_forward(m.get(), dp.get());
    mj_forward(m.get(), dm.get());
    mjtNum len_deriv_fd =
        (dp->actuator_force[i] - dm->actuator_force[i]) / (2 * eps);

    EXPECT_NEAR(len_deriv_analytic, len_deriv_fd, MjTol(1e-6, 1e-2));
  }
}

// verify DC motor force clamping checks pre-mechanical electrical force
TEST_F(DerivativeTest, DCMotorForceClampVsLuGreAndCogging) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="j" motorconst="1" resistance="1"
               input="pos vel" controller="20 0 4" saturation="2 0"
               cogging="0.5 3 0" lugre="100 2 5 6 0.1"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Case 1: electrical force unsaturated (~1.0), LuGre pushes total force > 2.0
  mj_resetData(m.get(), d.get());
  d->ctrl[0] = 0.05;   // kp*0.05 = 1.0 inside [-2, 2]
  d->act[0] = -0.025;  // -sigma0*z = +2.5 => total actuator_force > 2.0
  mj_forward(m.get(), d.get());
  ASSERT_GT(d->actuator_force[0], 2.0);

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic_1 = d->qDeriv[0];
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic_1, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // Case 2: electrical force saturated (> 2.0), LuGre pulls total force inside
  // (-2, 2)
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.9;  // cos(3*0.9) < 0 so cogging stiffness is stabilizing
  d->ctrl[0] = 1.1;  // kp*(1.1 - 0.9) = 4.0 > 2.0 (clamped at 2.0)
  d->act[0] =
      0.015;  // -sigma0*z = -1.5 => total actuator_force ~ 0.5 inside [-2, 2]
  mj_forward(m.get(), d.get());
  ASSERT_LT(mju_abs(d->actuator_force[0]), 2.0);

  // electrical velocity derivative is 0, only LuGre -sigma1 = -2.0 remains
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic_2 = d->qDeriv[0];
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic_2, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // electrical length derivative is 0, only cogging derivative remains in
  // efm_ak
  ASSERT_EQ(d->nefmA, 1);
  mjtNum eps = MjTol(1e-6, 1e-3);
  MjDataPtr dp = MakeData(m), dm = MakeData(m);
  mj_copyData(dp.get(), m.get(), d.get());
  mj_copyData(dm.get(), m.get(), d.get());
  dp->qpos[0] += eps;
  dm->qpos[0] -= eps;
  mj_forward(m.get(), dp.get());
  mj_forward(m.get(), dm.get());
  mjtNum len_fd = (dp->actuator_force[0] - dm->actuator_force[0]) / (2 * eps);
  EXPECT_NEAR(-d->efm_ak[0] / m->opt.timestep, len_fd, MjTol(1e-6, 1e-2));
}

// verify DC motor + co-acting tendon actuator under tendon actuatorfrcrange
TEST_F(DerivativeTest, DCMotorTendonActFrcClampVsLuGreAndCogging) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t" actuatorfrclimited="true" actuatorfrcrange="-2 2">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>
    <actuator>
      <dcmotor tendon="t" motorconst="1" resistance="1"
               input="pos vel" controller="20 0 4"
               cogging="0.5 3 0" lugre="100 2 5 6 0.1"/>
      <position tendon="t" kp="4" kv="1"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Case 1: pre-mechanical sum (1.0 + 0.2 = 1.2) inside [-2, 2], LuGre pushes
  // raw actuator_force sum > 2.0
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.9;
  d->ctrl[0] = 0.95;   // dcmotor pos target -> f_elec = 1.0
  d->ctrl[2] = 0.95;   // position target -> f_pos = 0.2
  d->act[0] = -0.025;  // -sigma0*z = +2.5 => raw sum > 3.7 > 2.0
  mj_forward(m.get(), d.get());
  ASSERT_GT(d->actuator_force[0] + d->actuator_force[1], 2.0);
  EXPECT_EQ(d->nefmA, 2);

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic_1 = d->qDeriv[0];
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic_1, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // Case 2: pre-mechanical sum (4.0 + 0.4 = 4.4) clamped to 2.0, LuGre pulls
  // raw actuator_force sum inside (-2, 2)
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.9;
  d->ctrl[0] = 1.1;   // dcmotor pos target -> f_elec = 4.0
  d->ctrl[2] = 1.0;   // position target -> f_pos = 0.4
  d->act[0] = 0.015;  // -sigma0*z = -1.5 => raw sum ~ 0.71 inside (-2, 2)
  mj_forward(m.get(), d.get());
  ASSERT_LT(mju_abs(d->actuator_force[0] + d->actuator_force[1]), 2.0);

  // electrical and position derivatives vanish; only LuGre -sigma1 remains
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic_2 = d->qDeriv[0];
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic_2, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // only dcmotor cogging derivative remains in efm_ak
  ASSERT_EQ(d->nefmA, 1);
  EXPECT_EQ(d->efm_aid[0], 0);
  mjtNum eps = MjTol(1e-6, 1e-3);
  MjDataPtr dp = MakeData(m), dm = MakeData(m);
  mj_copyData(dp.get(), m.get(), d.get());
  mj_copyData(dm.get(), m.get(), d.get());
  dp->qpos[0] += eps;
  dm->qpos[0] -= eps;
  mj_forward(m.get(), dp.get());
  mj_forward(m.get(), dm.get());
  mjtNum len_fd = (dp->qfrc_actuator[0] - dm->qfrc_actuator[0]) / (2 * eps);
  EXPECT_NEAR(-d->efm_ak[0] / m->opt.timestep, len_fd, MjTol(1e-6, 1e-2));
}

// verify DC motor with LuGre and cogging under joint actuatorfrcrange
TEST_F(DerivativeTest, DCMotorJointActFrcClampVsLuGreAndCogging) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="j" type="slide" actuatorfrclimited="true" actuatorfrcrange="-2 2"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="j" motorconst="1" resistance="1"
               input="pos vel" controller="20 0 4"
               cogging="0.5 3 0" lugre="100 2 5 6 0.1"/>
      <position joint="j" kp="4" kv="1"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Case 1: Unclamped - actuator forces inside [-2, 2], derivatives active and
  // match FD
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.9;
  d->ctrl[0] = 0.95;  // dcmotor pos target -> f_elec = 1.0
  d->ctrl[2] = 0.95;  // position target -> f_pos = 0.2
  d->act[0] = 0.005;  // small LuGre bristle deflection
  mj_forward(m.get(), d.get());
  ASSERT_LT(mju_abs(d->qfrc_actuator[0]), 2.0);
  EXPECT_EQ(d->nefmA, 2);

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic_1 = d->qDeriv[0];
  EXPECT_NE(vel_analytic_1, 0.0);
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic_1, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // Case 2: Clamped - total joint actuator force clamped to 2.0.
  // Joint clamping zeroes ALL derivatives (electrical, position, LuGre, and
  // cogging).
  mj_resetData(m.get(), d.get());
  d->qpos[0] = 0.9;
  d->ctrl[0] = 1.1;  // dcmotor pos target -> f_elec = 4.0
  d->ctrl[2] = 1.0;  // position target -> f_pos = 0.4
  d->act[0] = 0.005;
  mj_forward(m.get(), d.get());
  EXPECT_DOUBLE_EQ(d->qfrc_actuator[0], 2.0);

  // Analytical velocity derivative vanishes completely
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  EXPECT_DOUBLE_EQ(d->qDeriv[0], 0.0);

  // Finite difference derivative also vanishes because joint remains clamped
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(d->qDeriv[0], 0.0, MjTol(1e-6, 1e-2));

  // No active actuators in efm_aid under discrete integrator (all clamped)
  EXPECT_EQ(d->nefmA, 0);

  // verify control sensitivity B vanishes across integrators
  for (mjtIntegrator integ :
       {mjINT_IMPLICITFAST, mjINT_IMPLICIT, mjINT_DISCRETE}) {
    m->opt.integrator = integ;
    mj_forward(m.get(), d.get());
    const int ndx = 2 * m->nv + m->na;
    std::vector<mjtNum> A(ndx * ndx), B(ndx * m->nu);
    mjd_transitionFD(m.get(), d.get(), MjTol(1e-6, 1e-3), /*flg_centered=*/1,
                     A.data(), B.data(), nullptr, nullptr);
    // Control columns for saturated joint actuators must vanish
    for (int row = 0; row < ndx; ++row) {
      for (int u = 0; u < m->nu; ++u) {
        EXPECT_NEAR(B[row * m->nu + u], 0.0, MjTol(1e-8, 1e-4));
      }
    }
  }
}

// verify DC motor derivatives with delayed ctrl and slew rate limiting in
// dcmotorInputs
TEST_F(DerivativeTest, DCMotorInputsDelayedCtrlAndSlew) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.005"/>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="j" delay="0.01" nsample="3" interp="linear"
               input="pos vel" controller="20 0 5 10 0 0"
               motorconst="1" resistance="1"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // step forward so history buffer is populated
  d->ctrl[0] = 0.5;
  for (int step = 0; step < 5; ++step) {
    mj_step(m.get(), d.get());
  }

  // verify smooth_vel matches FD
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  mjtNum vel_analytic = d->qDeriv[0];

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-7, 1e-3));
  EXPECT_NEAR(vel_analytic, d->qDeriv[0], MjTol(1e-6, 1e-2));

  // verify length derivative in efm_ak matches FD
  ASSERT_EQ(d->nefmA, 1);
  mjtNum eps = MjTol(1e-6, 1e-3);
  MjDataPtr dp = MakeData(m), dm = MakeData(m);
  mj_copyData(dp.get(), m.get(), d.get());
  mj_copyData(dm.get(), m.get(), d.get());
  dp->qpos[0] += eps;
  dm->qpos[0] -= eps;
  mj_forward(m.get(), dp.get());
  mj_forward(m.get(), dm.get());
  mjtNum len_fd = (dp->qfrc_actuator[0] - dm->qfrc_actuator[0]) / (2 * eps);
  EXPECT_NEAR(-d->efm_ak[0] / m->opt.timestep, len_fd, MjTol(1e-6, 1e-2));
}

// verify DC motor derivatives under Vmax controller saturation and dimax
// current-rate limit
TEST_F(DerivativeTest, DCMotorVoltageAndCurrentRateSaturation) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.002"/>
    <worldbody>
      <body>
        <joint name="stateless_vmax" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="stateful_vmax" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="stateful_dimax" type="slide"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="stateless_vmax" motorconst="2" resistance="1"
               input="pos vel" controller="20 0 6 0 0 3"/>
      <dcmotor joint="stateful_vmax" motorconst="2" resistance="1"
               inductance="0 0.005" input="pos vel" controller="20 0 6 0 0 3"/>
      <dcmotor joint="stateful_dimax" motorconst="2" resistance="1"
               inductance="0 0.005" saturation="0 0 5"
               input="pos vel" controller="20 0 6"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // large position error saturates Vmax on actuators 0 & 1 and dimax on
  // actuator 2
  d->ctrl[0] = 2.0;
  d->ctrl[2] = 2.0;
  d->ctrl[4] = 2.0;
  mj_forward(m.get(), d.get());

  // compare smooth_vel against FD across all 3 DOFs
  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_vel(m.get(), d.get(), /*flg_bias=*/1);
  vector<mjtNum> analytic = AsVector(d->qDeriv, m->nD);

  mju_zero(d->qDeriv, m->nD);
  mjd_smooth_velFD(m.get(), d.get(), MjTol(1e-6, 1e-3));
  EXPECT_THAT(AsVector(d->qDeriv, m->nD),
              Pointwise(MjNear(1e-6, 1e-2), analytic));

  // all 3 length derivatives are zero under Vmax / dimax saturation (efm_ak ==
  // 0)
  for (int t = 0; t < d->nefmA; t++) {
    EXPECT_MJTNUM_EQ(d->efm_ak[t], 0);
  }
}

// Utility: Rotate flex grid
void RotateFlexGrid(mjModel* model, mjData* data, const char* flex_name,
                    double angle) {
  int flex_id = mj_name2id(model, mjOBJ_FLEX, flex_name);
  ASSERT_NE(flex_id, -1);
  int node_adr = model->flex_nodeadr[flex_id];
  int* node_bodies = model->flex_nodebodyid + node_adr;
  int nodenum = model->flex_nodenum[flex_id];

  // Make deterministic quaternion for rotation inside helper
  mjtNum quat[4] = {1, 0, 0, 0};
  if (angle != 0) {
    mjtNum vel[3] = {1, 1, 1};
    mju_normalize3(vel);
    mju_quatIntegrate(quat, vel, angle);
  }

  // reset first to get initial positions
  mj_resetData(model, data);
  mj_forward(model, data);  // Compute initial xpos

  // Update qpos
  for (int i = 0; i < nodenum; i++) {
    int bodyid = node_bodies[i];

    // Only process nodes with valid bodies (FlexInterpDamping assumes this)
    if (bodyid >= 0) {
      mjtNum xpos0[3];
      mju_copy3(xpos0, data->xpos + 3 * bodyid);  // Initial absolute position

      mjtNum xpos_new[3];
      mju_rotVecQuat(xpos_new, xpos0, quat);  // Rotate absolute position

      mjtNum delta[3];
      mju_sub3(delta, xpos_new, xpos0);

      // Find the qpos address for this node/body
      int jnt = model->body_jntadr[bodyid];
      if (jnt >= 0) {
        int qadr = model->jnt_qposadr[jnt];
        mju_addTo3(data->qpos + qadr, delta);
      }
    }
  }
}

// Helper: assemble flex stiffness into dense matrix via matrix-vector products.
// Builds K column-by-column using mjd_flexInterp_mul.
// Result is -(h^2 + h*damping) * J'KJ (negative sign matches the old addH
// convention where stiffness is subtracted from the system matrix).
static void mulKD_dense(mjModel* m, mjData* d, mjtNum* H_dense, int nv,
                        mjtNum h) {
  std::vector<mjtNum> e_i(nv, 0);
  std::vector<mjtNum> col(nv, 0);
  for (int i = 0; i < nv; i++) {
    mju_zero(e_i.data(), nv);
    mju_zero(col.data(), nv);
    e_i[i] = 1.0;
    mjd_flexInterp_mul(m, d, col.data(), e_i.data(), h * h, h, NULL);
    // col = +(h^2 + h*damp)*K*e_i, negate to match addH convention (H -= K)
    for (int j = 0; j < nv; j++) {
      H_dense[j * nv + i] = -col[j];
    }
  }
}

// compare analytic and fin-diff d_qfrc_passive/d_qvel for flex interp
// Combined test for verify mjd_flexInterp_mulK (stiffness) and damping
TEST_F(DerivativeTest, FlexInterpDerivatives) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="flex" type="grid" count="3 3 3" spacing="0.1 0.2 0.3"
                radius=".01" dim="3" mass="1" dof="trilinear">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson="0.3" damping="50"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nD = model->nD;
  int nv = model->nv;
  ASSERT_EQ(model->nq, 24);  // 8 corners * 3 dofs

  MjDataPtr data = MakeData(model);

  // iterate over rotations
  for (mjtNum angle : {0.0, 0.5, 1.0, mjPI / 2, mjPI, 2.0 * mjPI}) {
    RotateFlexGrid(model.get(), data.get(), "flex", angle);
    mj_forward(model.get(), data.get());

    // part 1: stiffness verification
    {
      std::vector<mjtNum> vec(nv);
      std::vector<mjtNum> res(nv);
      mju_zero(vec.data(), nv);
      // use deterministic random perturbation to verify full stiffness matrix
      // behavior
      for (int i = 0; i < nv; i++) {
        vec[i] = mju_Halton(i, 2) - 0.5;
      }

      // use mulKD to compute K * vec
      // mulKD adds (h^2*K + h*D)*vec to res
      // if we set h=1, damping=0, we get K*vec
      mjtNum save_damping = model->flex_damping[0];
      model->flex_damping[0] = 0;

      std::vector<mjtNum> H(nv * nv, 0);

      // assemble K into H column-by-column
      mulKD_dense(model.get(), data.get(), H.data(), nv, 1.0);

      // restore damping
      model->flex_damping[0] = save_damping;

      // compute res = K * vec
      mju_mulMatVec(res.data(), H.data(), vec.data(), nv, nv);

      // finite difference of mj_passive for stiffness
      mjtNum eps = MjEps(1e-6, 1e-3);
      mjData* data_perturbed = mj_copyData(NULL, model.get(), data.get());

      // apply perturbation
      mju_addToScl(data_perturbed->qpos, vec.data(), eps, nv);

      // recompute geometry/passive
      mj_forward(model.get(), data_perturbed);

      // compute FD estimate of K * vec
      // qfrc_passive = -dV/dq => d(qfrc)/dq = -K
      // (qfrc_new - qfrc)/eps ~= -K * vec
      std::vector<mjtNum> fd_res(nv);
      for (int i = 0; i < nv; ++i) {
        fd_res[i] =
            -(data_perturbed->qfrc_passive[i] - data->qfrc_passive[i]) / eps;
      }

      // compare analytical result (H*vec) with FD result
      for (int i = 0; i < nv; ++i) {
        EXPECT_THAT(res[i], MjNear(fd_res[i], 5e-3, 5.0))
            << "Stiffness Mismatch at DOF " << i;
      }

      mj_deleteData(data_perturbed);

      // check symmetry: K[i,j] == K[j,i]
      std::vector<mjtNum>& K_full = H;
      mjtNum max_asymmetry = 0;
      for (int i = 0; i < nv; i++) {
        for (int j = 0; j < i; j++) {
          mjtNum diff = mju_abs(K_full[i * nv + j] - K_full[j * nv + i]);
          max_asymmetry = mju_max(max_asymmetry, diff);
        }
      }
      EXPECT_THAT(max_asymmetry, MjNear(0, 1e-10, 5e-4))
          << "K matrix is not symmetric at angle " << angle;

      // check positive semi-definiteness: v^T * K * v >= 0
      for (int trial = 0; trial < 5; trial++) {
        std::vector<mjtNum> v(nv);
        for (int i = 0; i < nv; i++) {
          v[i] = mju_Halton(i + trial * nv, 3) - 0.5;
        }
        mjtNum vKv = 0;
        for (int i = 0; i < nv; i++) {
          for (int j = 0; j < nv; j++) {
            vKv += v[i] * K_full[i * nv + j] * v[j];
          }
        }
        EXPECT_GE(vKv, MjTol(-1e-8, -1e-5))
            << "K matrix is not PSD at angle " << angle;
      }
    }

    // part 2: damping verification
    {
      // set velocity non-zero to test damping
      data->qvel[0] = 1.0;

      mj_forward(model.get(), data.get());

      // get analytic derivatives (without Flex Damping currently)
      std::vector<mjtNum> qDerivAnalytic(nD);
      mju_zero(data->qDeriv, nD);
      mjd_passive_vel(model.get(), data.get());
      mju_copy(qDerivAnalytic.data(), data->qDeriv, nD);

      // finite-difference derivatives
      std::vector<mjtNum> qDerivFD(nD);
      mju_zero(data->qDeriv, nD);
      mjtNum eps = MjEps(1e-6, 1e-3);

      mjd_passive_velFD(model.get(), data.get(), eps);
      mju_copy(qDerivFD.data(), data->qDeriv, nD);

      // check that we have non-zero damping (FD should find it)
      EXPECT_GT(mju_norm(qDerivFD.data(), nD), 1e-3);

      // compute expected flex damping using mulKD_dense
      // D = 4*H(0.5) - H(1)
      vector<mjtNum> H1(nv * nv, 0);
      mulKD_dense(model.get(), data.get(), H1.data(), nv, 1.0);

      vector<mjtNum> H2(nv * nv, 0);
      mulKD_dense(model.get(), data.get(), H2.data(), nv, 0.5);

      vector<mjtNum> D(nv * nv);
      for (int i = 0; i < nv * nv; i++) {
        D[i] = 4.0 * H2[i] - H1[i];
      }

      // subtract D from qDerivAnalytic using sparse indexing
      // d(force)/d(vel) = -D
      for (int i = 0; i < nv; i++) {
        int rownnz = model->D_rownnz[i];
        int rowadr = model->D_rowadr[i];
        for (int k = 0; k < rownnz; k++) {
          int index = rowadr + k;
          int j = model->D_colind[index];
          qDerivAnalytic[index] -= D[i * nv + j];
        }
      }

      // expect FD and corrected analytic derivatives to match
      EXPECT_THAT(qDerivAnalytic, Pointwise(MjNear(1e-4, 1e4), qDerivFD))
          << "Damping Mismatch at angle: " << angle;
    }
  }
}

// Test Jacobian under deformation to highlight approximation error
TEST_F(DerivativeTest, FlexInterpDerivativesDeformed) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="flex" type="grid" count="3 3 3" spacing="0.1 0.2 0.3"
                radius=".01" dim="3" mass="1" dof="trilinear">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson="0.3" damping="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;

  MjDataPtr data = MakeData(model);

  // Apply rotation
  RotateFlexGrid(model.get(), data.get(), "flex", 1.0);  // 1 radian rotation

  // Apply deformation (stretch along X)
  // qpos is initialized by RotateFlexGrid.
  // Add a random perturbation to qpos that represents deformation.
  // We use a deterministic sequence to ensure reproducibility.
  std::vector<mjtNum> deformation(nv);
  for (int i = 0; i < nv; i++) {
    // Large deformation to make sure terms are significant
    deformation[i] = (mju_Halton(i, 3) - 0.5) * 0.2;
  }
  mju_addTo(data->qpos, deformation.data(), nv);

  mj_forward(model.get(), data.get());

  // 1. Compute Analytic Jacobian (Approximate)
  // We use mulKD_dense to get K_approx
  std::vector<mjtNum> H_approx(nv * nv, 0);

  // h=1, damping=0 => gives K
  mulKD_dense(model.get(), data.get(), H_approx.data(), nv, 1.0);

  // 2. Compute Finite Difference Jacobian (Ground Truth)
  // qfrc_passive = -dV/dq
  // d(qfrc)/dq = -K_true
  std::vector<mjtNum> K_true(nv * nv, 0);
  mjtNum eps = 1e-6;

  for (int i = 0; i < nv; i++) {
    mjData* data_p = mj_copyData(NULL, model.get(), data.get());
    data_p->qpos[i] += eps;
    mj_forward(model.get(), data_p);

    for (int j = 0; j < nv; j++) {
      // d(force_j)/d(q_i)
      mjtNum df = data_p->qfrc_passive[j] - data->qfrc_passive[j];
      // K_true[j, i] = -df/eps
      K_true[j * nv + i] = -df / eps;
    }
    mj_deleteData(data_p);
  }

  // 3. Compare and check for significant mismatch
  mjtNum max_error = 0;
  for (int i = 0; i < nv * nv; i++) {
    max_error = mju_max(max_error, mju_abs(H_approx[i] - K_true[i]));
  }

  // We expect significant error because of deformation + rotation.
  // The missing term (geometric stiffness) is proportional to stress.
  // We assert that the error is relatively large to confirm the approximation
  // exists.
  EXPECT_GT(max_error, 1e-3)
      << "Jacobian approximation should differ from FD when deformed";
}

// Helper: assemble the standard-flex stretch stiffness into a dense matrix,
// column-by-column using mjd_flexStretch_mul with scale (s1 + s2*damping).
static void stretchK_dense(mjModel* m, mjData* d, mjtNum* K, int nv, mjtNum s1,
                           mjtNum s2) {
  std::vector<mjtNum> e_i(nv, 0);
  std::vector<mjtNum> col(nv, 0);
  for (int i = 0; i < nv; i++) {
    mju_zero(e_i.data(), nv);
    mju_zero(col.data(), nv);
    e_i[i] = 1.0;
    mjd_flexStretch_mul(m, d, col.data(), e_i.data(), s1, s2);
    for (int j = 0; j < nv; j++) {
      K[j * nv + i] = col[j];
    }
  }
}

// The same frame mismatch seen through the derivative: mjd_flexBend_mul is the
// Jacobian of the bending force only if operator and force agree on the frame.
// An unrotated flex cannot catch it, because the stencil is one scalar per
// vertex pair applied coordinate-wise and so commutes with a rotation shared by
// every vertex. The pin also covers the zero-dof vertex path, where body_dofadr
// is negative and an unguarded stencil indexes out of bounds.
TEST_F(DerivativeTest, FlexBendDerivativesRotated) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body name="turned" euler="90 35 20">
        <flexcomp name="rot" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                  radius=".01" dim="2" mass="1">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.3" thickness="0.01"
                      elastic2d="bend" damping="0"/>
        </flexcomp>
      </body>
      <flexcomp name="flat" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                radius=".01" dim="2" mass="1" pos="1 0 0">
        <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01"
                    elastic2d="bend" damping="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  ASSERT_EQ(model->nq, nv);  // all slide dofs
  ASSERT_EQ(model->nflex, 2);
  MjDataPtr data = MakeData(model);

  // deform both flexes out of plane so the bending stencil carries real force
  for (int i = 0; i < nv; i++) {
    data->qpos[i] += 2e-3 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());

  std::vector<mjtNum> vec(nv), res(nv, 0);
  for (int i = 0; i < nv; i++) {
    vec[i] = mju_Halton(i, 3) - 0.5;
  }
  mjd_flexBend_mul(model.get(), data.get(), res.data(), vec.data(), 1, 0);

  mjtNum eps = MjEps(1e-7, 1e-4);
  mjData* perturbed = mj_copyData(NULL, model.get(), data.get());
  mju_addToScl(perturbed->qpos, vec.data(), eps, nv);
  mj_forward(model.get(), perturbed);

  // check each flex on its own: the unrotated one is the control that isolates
  // the rotation
  for (int f = 0; f < model->nflex; f++) {
    SCOPED_TRACE(model->names + model->name_flexadr[f]);
    mjtNum max_err = 0, scale = 0;
    for (int k = 0; k < model->flex_vertnum[f]; k++) {
      int body = model->flex_vertbodyid[model->flex_vertadr[f] + k];
      int adr = model->body_dofadr[body];
      for (int x = 0; x < model->body_dofnum[body]; x++) {
        mjtNum fd =
            -(perturbed->qfrc_passive[adr + x] - data->qfrc_passive[adr + x]) /
            eps;
        max_err = mju_max(max_err, mju_abs(res[adr + x] - fd));
        scale = mju_max(scale, mju_abs(fd));
      }
    }
    // bending is linear in position, so the difference is exact up to roundoff
    EXPECT_GT(scale, 1e-3)
        << "test should exercise nontrivial bending stiffness";
    EXPECT_LT(max_err, MjTol(1e-4, 1e-2) * scale)
        << "mjd_flexBend_mul is not the Jacobian of the flex bending force";
  }
  mj_deleteData(perturbed);
}

// A rest crease sharper than 90 degrees gives a negative bending stencil (cos
// theta < 0 in the Cubic Shells energy). The solver metric keeps only the
// convex stencils, so the bending operator and its assembly stay PSD, while the
// passive force keeps every stencil.
TEST_F(DerivativeTest, FlexBendMetricDropsConcaveStencils) {
  static const char* const kXml = R"(
  <mujoco>
    <option timestep="0.005" integrator="discrete" solver="CG"/>
    <worldbody>
      <flexcomp name="crease" type="direct" dim="2" mass="1e-4" radius=".01"
                point="0 0 0  1 0 0  .5 1 0  .5 .7071 .7071" element="0 1 2  1 0 3">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01" elastic2d="bend" damping="0"/>
      </flexcomp>
      <flexcomp name="flat" type="direct" dim="2" mass="1" radius=".01"
                point="3 0 0  4 0 0  3.5 1 0  3.5 -1 0" element="0 1 2  1 0 3">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01" elastic2d="bend" damping="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nflex, 2);
  int nv = model->nv;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // the interior edge's stencil: negative on the crease, positive on the flat
  // flex
  for (int f = 0; f < 2; f++) {
    int interior = 0;
    for (int e = 0; e < model->flex_edgenum[f]; e++) {
      int adr = model->flex_edgeadr[f] + e;
      if (model->flex_edgeflap[2 * adr + 1] < 0) continue;
      const mjtNum* b =
          model->flex_bending + model->flex_bendingadr[f] + 17 * e;
      mjtNum trace = b[0] + b[5] + b[10] + b[15];
      if (f == 0) {
        EXPECT_LT(trace, 0);
      } else {
        EXPECT_GT(trace, 0);
      }
      interior++;
    }
    EXPECT_EQ(interior, 1);
  }

  // dense bending operator
  std::vector<mjtNum> B(nv * nv, 0), col(nv), e_i(nv);
  for (int i = 0; i < nv; i++) {
    mju_zero(e_i.data(), nv);
    e_i[i] = 1;
    mju_zero(col.data(), nv);
    mjd_flexBend_mul(model.get(), data.get(), col.data(), e_i.data(), 1, 0);
    for (int j = 0; j < nv; j++) {
      B[j * nv + i] = col[j];
    }
  }

  // the crease's dofs carry no metric bending, the flat flex's do
  auto dofs = [&](int f, int* first, int* last) {
    int b0 = model->flex_vertbodyid[model->flex_vertadr[f]];
    int b1 = model->flex_vertbodyid[model->flex_vertadr[f] +
                                    model->flex_vertnum[f] - 1];
    *first = model->body_dofadr[b0];
    *last = model->body_dofadr[b1] + model->body_dofnum[b1];
  };
  int c0, c1, f0, f1;
  dofs(0, &c0, &c1);
  dofs(1, &f0, &f1);
  mjtNum crease = 0, flat = 0;
  for (int i = 0; i < nv; i++) {
    for (int j = 0; j < nv; j++) {
      if (i >= c0 && i < c1) crease = mju_max(crease, mju_abs(B[i * nv + j]));
      if (i >= f0 && i < f1) flat = mju_max(flat, mju_abs(B[i * nv + j]));
    }
  }
  EXPECT_EQ(crease, 0);
  EXPECT_GT(flat, 0);

  // PSD along any direction, and the assembly agrees
  std::vector<int> rownnz(nv), rowadr(nv);
  int nnz = mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                                   rowadr.data(), nullptr, nullptr, 1, 0, 1, 0,
                                   nullptr);
  std::vector<int> colind(nnz);
  std::vector<mjtNum> val(nnz);
  mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(), rowadr.data(),
                         colind.data(), val.data(), 1, 0, 1, 0, nullptr);
  for (int k = 0; k < 20; k++) {
    std::vector<mjtNum> vec(nv), Bv(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(k * nv + i + 1, 3) - 0.5;
    }
    mjd_flexBend_mul(model.get(), data.get(), Bv.data(), vec.data(), 1, 0);
    EXPECT_GE(mju_dot(vec.data(), Bv.data(), nv), -1e-12 * flat);
    for (int i = 0; i < nv; i++) {
      mjtNum assembled = mju_dotSparse(val.data() + rowadr[i], vec.data(),
                                       rownnz[i], colind.data() + rowadr[i]);
      EXPECT_THAT(assembled, MjNear(Bv[i], 1e-10, 1e-8)) << i;
    }
  }

  // the passive force keeps the crease's stencil
  for (int i = c0; i < c1; i++) {
    data->qpos[i] += 1e-3 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());
  mjtNum force = 0;
  for (int i = c0; i < c1; i++) {
    force = mju_max(force, mju_abs(data->qfrc_spring[i]));
  }
  EXPECT_GT(force, 0);

  // stepping stays finite for both bending-only (efm0_L) and bending+stretching
  for (int step = 0; step < 10; step++) {
    mj_step(model.get(), data.get());
  }
  EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0);
  for (int i = 0; i < nv; i++) {
    EXPECT_FALSE(mju_isBad(data->qacc[i])) << i;
    EXPECT_FALSE(mju_isBad(data->qpos[i])) << i;
  }

  std::string both_xml = kXml;
  for (size_t pos = 0;
       (pos = both_xml.find("elastic2d=\"bend\"", pos)) != std::string::npos;) {
    both_xml.replace(pos, 16, "elastic2d=\"both\"");
  }
  MjModelPtr both_model =
      LoadModelFromString(both_xml.c_str(), error, sizeof(error));
  ASSERT_THAT(both_model.get(), NotNull()) << error;
  MjDataPtr both_data = MakeData(both_model);
  for (int i = c0; i < c1; i++) {
    both_data->qpos[i] += 1e-3 * (mju_Halton(i, 2) - 0.5);
  }
  for (int step = 0; step < 10; step++) {
    mj_step(both_model.get(), both_data.get());
  }
  EXPECT_EQ(both_data->warning[mjWARN_BADQACC].number, 0);
  for (int i = 0; i < nv; i++) {
    EXPECT_FALSE(mju_isBad(both_data->qacc[i])) << i;
    EXPECT_FALSE(mju_isBad(both_data->qpos[i])) << i;
  }
}

struct FlexFrameCase {
  const char* name;
  bool welded;
  bool rotated_siblings;
  int nflex;
};

class FlexFrameTest : public MujocoTest,
                      public testing::WithParamInterface<FlexFrameCase> {
 protected:
  void SetUp() override { LoadPatch(); }

  // Two equilateral triangles; only the body frames and attachment topology
  // vary.
  void LoadPatch(const char* elastic2d = "both") {
    std::string xml = R"(
      <mujoco>
        <option timestep="0.002" integrator="discrete" solver="CG"/>
        <worldbody><body name="parent" euler="90 35 20">
    )";
    const char* positions[] = {"0 0 0", "0.1 0 0", "0.05 0.0866025404 0",
                               "0.15 0.0866025404 0"};
    const char* rotations[] = {"0 0 0", "30 -15 20", "-20 40 10", "45 10 -35"};
    for (int v = 0; v < 4; v++) {
      const char* rotation =
          GetParam().rotated_siblings ? rotations[v] : "0 0 0";
      if (GetParam().welded && v == 3) {
        rotation = "25 -15 40";
      }
      xml += "<body name='v" + std::to_string(v) + "' pos='" + positions[v] +
             "' euler='" + rotation + R"('>
          <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
          <joint type="slide" axis="1 0 0" armature="0.01"/>
          <joint type="slide" axis="0 1 0" armature="0.02"/>
          <joint type="slide" axis="0 0 1" armature="0.03"/>
      )";
      if (GetParam().welded && v == 3) {
        xml += R"(<body name="welded" euler="20 10 30">
          <inertial pos="0 0 0" mass="0.1" diaginertia="1e-4 1e-4 1e-4"/>
        </body>)";
      }
      xml += "</body>";
    }
    xml += "</body></worldbody><deformable>";
    for (int f = 0; f < GetParam().nflex; f++) {
      xml += std::string("<flex dim='2' body='v0 v1 v2 ") +
             (GetParam().welded ? "welded" : "v3") + R"(' element="0 1 2 1 3 2">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.3" thickness="0.01" damping="0.02"
      )";
      xml += std::string("elastic2d='") + elastic2d + "'/></flex>";
    }
    xml += "</deformable></mujoco>";
    char error[1024];
    data.reset();
    model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    ASSERT_EQ(model->nv, 12);
    if (!GetParam().welded) {
      for (int v = 0; v < 4; v++) {
        EXPECT_EQ(model->body_simple[model->flex_vertbodyid[v]], 2);
      }
    }
    data = MakeData(model);
    mj_forward(model.get(), data.get());
  }

  std::vector<mjtNum> Direction() const {
    std::vector<mjtNum> vec(model->nv);
    for (int i = 0; i < model->nv; i++) {
      vec[i] = mju_Halton(i, 3) - 0.5;
    }
    return vec;
  }

  // Dilate in world coordinates, then add a small out-of-plane perturbation.
  // All edges stay in tension, where the stretch Hessian is exact.
  void Deform() {
    for (int v = 0; v < 4; v++) {
      int body = model->body_weldid[model->flex_vertbodyid[v]];
      mjtNum delta[3];
      mju_scl3(delta, data->flexvert_xpos + 3 * v, 0.05);
      delta[2] += 1e-4 * (mju_Halton(v, 2) - 0.5);
      mju_mulMatTVec3(data->qpos + model->body_dofadr[body],
                      data->xmat + 9 * body, delta);
    }
    mj_forward(model.get(), data.get());
  }

  std::vector<mjtNum> Apply(const std::vector<mjtNum>& vec, mjtNum s1,
                            mjtNum s2) {
    std::vector<mjtNum> result(model->nv, 0);
    mjd_flexBend_mul(model.get(), data.get(), result.data(), vec.data(), s1,
                     s2);
    mjd_flexStretch_mul(model.get(), data.get(), result.data(), vec.data(), s1,
                        s2);
    return result;
  }

  void ExpectDerivative(bool velocity) {
    // The stretch operator includes geometric stiffness; its velocity part
    // equals the force Jacobian at the unstretched reference, where that extra
    // term vanishes.
    if (!velocity) {
      Deform();
    }
    auto vec = Direction();
    auto result = Apply(vec, !velocity, velocity);
    MjDataPtr perturbed(mj_copyData(nullptr, model.get(), data.get()));
    mjtNum eps = MjEps(1e-7, 1e-4);
    mju_addToScl(velocity ? perturbed->qvel : perturbed->qpos, vec.data(), eps,
                 model->nv);
    mj_forward(model.get(), perturbed.get());
    const mjtNum* force = velocity ? data->qfrc_damper : data->qfrc_spring;
    const mjtNum* pert_force =
        velocity ? perturbed->qfrc_damper : perturbed->qfrc_spring;
    for (int i = 0; i < model->nv; i++) {
      mjtNum fd = -(pert_force[i] - force[i]) / eps;
      EXPECT_NEAR(result[i], fd, MjTol(1e-3, 5e-2) * mju_max(1, mju_abs(fd)))
          << i;
    }
  }

  void ExpectAssembly(mjtNum s1, mjtNum s2) {
    Deform();
    auto vec = Direction();
    auto result = Apply(vec, s1, s2);
    std::vector<int> rownnz(model->nv), rowadr(model->nv);
    int nnz = mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                                     rowadr.data(), nullptr, nullptr, s1, s2, 1,
                                     1, nullptr);
    std::vector<int> colind(nnz);
    std::vector<mjtNum> val(nnz);
    mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                           rowadr.data(), colind.data(), val.data(), s1, s2, 1,
                           1, nullptr);
    for (int i = 0; i < model->nv; i++) {
      mjtNum assembled = mju_dotSparse(val.data() + rowadr[i], vec.data(),
                                       rownnz[i], colind.data() + rowadr[i]);
      EXPECT_THAT(assembled, MjNear(result[i], 1e-10, 2e-5)) << i;
    }
  }

  void ExpectBendingFactorInverse() {
    // Bending-only models use the constant factor, without per-step block
    // preconditioning.
    ASSERT_EQ(data->nefmdof, 0);
    ASSERT_EQ(model->nefm0dof, model->nv);
    int nv = model->nv;
    std::vector<mjtNum> rhs(nv), solution(nv), product(nv);
    mjtNum h = model->opt.timestep;
    for (int col = 0; col < nv; col++) {
      mju_zero(rhs.data(), nv);
      rhs[col] = 1;
      mj_effPrec(model.get(), data.get(), solution.data(), rhs.data());
      mj_mulM(model.get(), data.get(), product.data(), solution.data());
      mjd_flexBend_mul(model.get(), data.get(), product.data(), solution.data(),
                       h * h, h);
      for (int row = 0; row < nv; row++) {
        EXPECT_THAT(product[row], MjNear(rhs[row], 1e-12, 2e-6))
            << row << ", " << col;
      }
    }
  }

  MjModelPtr model;
  MjDataPtr data;
};

TEST_P(FlexFrameTest, RigidTranslationHasNoDamping) {
  const mjtNum velocity[3] = {0.7, -0.4, 1.2};
  for (int v = 0; v < 4; v++) {
    int body = model->body_weldid[model->flex_vertbodyid[v]];
    mju_mulMatTVec3(data->qvel + model->body_dofadr[body],
                    data->xmat + 9 * body, velocity);
  }
  mj_forward(model.get(), data.get());
  EXPECT_THAT(AsVector(data->qfrc_damper, model->nv),
              Each(MjNear(0, 1e-10, 1e-5)));
}

TEST_P(FlexFrameTest, PositionDerivative) { ExpectDerivative(false); }

TEST_P(FlexFrameTest, VelocityDerivativeAtReference) { ExpectDerivative(true); }

TEST_P(FlexFrameTest, PositionAssembly) { ExpectAssembly(1, 0); }

TEST_P(FlexFrameTest, VelocityAssembly) { ExpectAssembly(0, 1); }

TEST_P(FlexFrameTest, ConstantBendingFactor) {
  ASSERT_NO_FATAL_FAILURE(LoadPatch("bend"));
  ExpectBendingFactorInverse();
}

TEST_P(FlexFrameTest, ConstantBendingFactorAfterFrameChange) {
  ASSERT_NO_FATAL_FAILURE(LoadPatch("bend"));
  int body = mj_name2id(model.get(), mjOBJ_BODY, "v1");
  const mjtNum rotation[4] = {0.5, 0.5, 0.5, 0.5};
  mju_copy4(model->body_quat + 4 * body, rotation);
  mj_setConst(model.get(), data.get());
  mj_forward(model.get(), data.get());
  ExpectBendingFactorInverse();
}

INSTANTIATE_TEST_SUITE_P(
    Frames, FlexFrameTest,
    testing::Values(FlexFrameCase{"Aligned", false, false, 1},
                    FlexFrameCase{"Welded", true, false, 1},
                    FlexFrameCase{"RotatedSiblings", false, true, 1},
                    FlexFrameCase{"SharedFlexBodies", true, true, 2}),
    [](const testing::TestParamInfo<FlexFrameCase>& info) {
      return info.param.name;
    });

// The cache follows position updates, survives velocity-only evaluations and
// mj_copyData, and does not change spring forces or either material's damping.
TEST_F(DerivativeTest, FlexHessianCacheLifetime) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <default>
      <joint type="slide"/>
      <geom type="sphere" size=".01" mass=".2" contype="0" conaffinity="0"/>
    </default>
    <worldbody>
      <body name="unused" pos="2 2 2">
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom/>
      </body>
      <body name="a">
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom/>
      </body>
      <body name="b" pos="1 0 0">
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom/>
      </body>
      <body name="c" pos=".2 .9 0">
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom/>
      </body>
      <body name="d" pos="-.1 .3 1.1">
        <joint axis="1 0 0"/><joint axis="0 1 0"/><joint axis="0 0 1"/><geom/>
      </body>
    </worldbody>
    <deformable>
      <flex name="tet" dim="3" body="unused a b c d" element="1 2 3 4">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".3" damping=".02"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  struct {
    int dim;
    int elastic3d;
  } cases[] = {{2, 0}, {3, 0}, {3, 1}};
  for (auto [dim, elastic3d] : cases) {
    SCOPED_TRACE(dim);
    SCOPED_TRACE(elastic3d);
    mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
    ASSERT_THAT(spec, NotNull());
    mjsFlex* flex = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "tet"));
    flex->elastic3d = elastic3d;
    if (elastic3d) {
      spec->option.integrator = mjINT_DISCRETE;
    }
    if (dim == 2) {
      flex->dim = 2;
      flex->elastic2d = 2;  // stretching
      flex->thickness = .02;
      int triangle[] = {1, 2, 3};
      mjs_setInt(flex->elem, triangle, 3);
    }
    MjModelPtr m(mj_compile(spec, nullptr));
    mj_deleteSpec(spec);
    ASSERT_THAT(m.get(), NotNull());
    m->opt.disableflags |= mjDSBL_DAMPER;
    MjDataPtr d = MakeData(m);
    MjDataPtr fresh = MakeData(m);
    std::vector<mjtNum> vec(m->nv), result(m->nv), expected(m->nv);
    for (int i = 0; i < m->nv; i++) {
      vec[i] = mju_Halton(i, 2) - .5;
      d->qpos[i] = .1 * vec[i];
    }

    auto cache_values = [&m](const mjData* data) {
      std::vector<mjtNum> values =
          AsVector(data->flexvert_hessian, 6 * m->nflexvert);
      values.insert(values.end(), data->flexedge_hessian,
                    data->flexedge_hessian + 9 * m->nflexedge);
      return values;
    };

    // SNH's discrete solver prepares the cache; StVK with Euler does not need
    // it.
    mj_forward(m.get(), d.get());
    EXPECT_EQ(d->flex_hessian_valid[0], elastic3d != 0);
    std::vector<mjtNum> spring(d->qfrc_spring, d->qfrc_spring + m->nv);
    mjd_flexStretch_mul(m.get(), d.get(), result.data(), vec.data(), 1, 0);
    ASSERT_TRUE(d->flex_hessian_valid[0]);
    std::vector<mjtNum> cache = cache_values(d.get());
    mj_passive(m.get(), d.get());
    EXPECT_THAT(spring, Pointwise(Eq(), AsVector(d->qfrc_spring, m->nv)));

    // Changing velocity, damping, or timestep does not change the cached
    // Hessian.
    mju_copy(d->qvel, vec.data(), m->nv);
    m->opt.disableflags &= ~mjDSBL_DAMPER;
    m->flex_damping[0] = .04;
    m->opt.timestep *= 2;
    mj_forwardSkip(m.get(), d.get(), mjSTAGE_POS, 1);
    EXPECT_TRUE(d->flex_hessian_valid[0]);
    EXPECT_THAT(cache, Pointwise(Eq(), cache_values(d.get())));
    mju_copy(fresh->qpos, d->qpos, m->nq);
    mju_copy(fresh->qvel, d->qvel, m->nv);
    mj_forward(m.get(), fresh.get());
    EXPECT_EQ(fresh->flex_hessian_valid[0], elastic3d != 0);
    EXPECT_THAT(AsVector(d->qfrc_spring, m->nv),
                Pointwise(Eq(), AsVector(fresh->qfrc_spring, m->nv)));
    EXPECT_THAT(
        AsVector(d->qfrc_damper, m->nv),
        Pointwise(MjNear(1e-12, 1e-5), AsVector(fresh->qfrc_damper, m->nv)));

    // Copying mjData preserves a usable cache; subsequent geometry updates
    // invalidate only the updated data, including standalone kinematics calls.
    mj_copyData(fresh.get(), m.get(), d.get());
    ASSERT_TRUE(fresh->flex_hessian_valid[0]);
    mjd_flexStretch_mul(m.get(), fresh.get(), expected.data(), vec.data(), 1,
                        0);
    EXPECT_THAT(result, Pointwise(Eq(), expected));
    d->qpos[3] += .17;
    mj_fwdKinematics(m.get(), d.get());
    EXPECT_FALSE(d->flex_hessian_valid[0]);
    EXPECT_THAT(cache, Pointwise(Eq(), cache_values(fresh.get())));
    mju_zero(result.data(), m->nv);
    mjd_flexStretch_mul(m.get(), d.get(), result.data(), vec.data(), 1, 0);
    EXPECT_TRUE(d->flex_hessian_valid[0]);
    EXPECT_NE(cache, cache_values(d.get()));
    mju_copy(fresh->qpos, d->qpos, m->nq);
    mj_forward(m.get(), fresh.get());
    mju_zero(expected.data(), m->nv);
    mjd_flexStretch_mul(m.get(), fresh.get(), expected.data(), vec.data(), 1,
                        0);
    EXPECT_THAT(result, Pointwise(Eq(), expected));

    // StVK still supports explicit integration without constructing the cache.
    if (!elastic3d) {
      for (int integrator : {mjINT_EULER, mjINT_RK4}) {
        m->opt.integrator = integrator;
        mj_resetData(m.get(), d.get());
        mju_copy(d->qvel, vec.data(), m->nv);
        mj_step(m.get(), d.get());
        EXPECT_FALSE(d->flex_hessian_valid[0]);
      }
    }

    // Reset invalidates even when the remaining derived data is debug-filled.
    // Discrete assembly also handles the unused vertex's empty sparse rows.
    mj_resetDataDebug(m.get(), d.get(), 255);
    EXPECT_FALSE(d->flex_hessian_valid[0]);
    m->opt.integrator = mjINT_DISCRETE;
    mj_forward(m.get(), d.get());
    EXPECT_TRUE(d->flex_hessian_valid[0]);
  }
}

// Independent dense material-Hessian reference. Assemble d2E/dF2 using the
// Levi-Civita formula for d2(det F), optionally project with a general 9x9
// eigensolver, then pull back through F = Ds Dm^-1. Rest coordinates are the
// original Cartesian positions, independent of the engine's normalized ones.
static std::vector<double> SNHHessianReference(const mjModel* m,
                                               const mjData* d,
                                               const std::vector<mjtNum>& rest,
                                               bool project) {
  std::vector<double> result(m->nv * m->nv, 0);
  auto epsilon = [](int a, int b, int c) {
    return (a - b) * (b - c) * (c - a) / 2;
  };
  for (int t = 0; t < m->flex_elemnum[0]; t++) {
    const int* vert = m->flex_elem + 4 * t;
    const mjtNum* k = m->flex_stiffness + 24 * t;
    double Dm[3][3], Ds[3][3], inverse[3][3] = {}, F[3][3] = {};
    for (int v = 0; v < 3; v++) {
      for (int x = 0; x < 3; x++) {
        Dm[x][v] = rest[3 * vert[v + 1] + x] - rest[3 * vert[0] + x];
        Ds[x][v] = d->flexvert_xpos[3 * vert[v + 1] + x] -
                   d->flexvert_xpos[3 * vert[0] + x];
      }
    }
    // Invert the actual Cartesian reference tetrahedron.
    double det = 0;
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        inverse[i][j] =
            Dm[(j + 1) % 3][(i + 1) % 3] * Dm[(j + 2) % 3][(i + 2) % 3] -
            Dm[(j + 1) % 3][(i + 2) % 3] * Dm[(j + 2) % 3][(i + 1) % 3];
      }
      det += Dm[0][i] * inverse[i][0];
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        inverse[i][j] /= det;
      }
    }
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        for (int l = 0; l < 3; l++) {
          F[i][j] += Ds[i][l] * inverse[l][j];
        }
      }
    }
    double cof[9] = {}, H[9][9] = {};
    for (int i = 0; i < 3; i++) {
      for (int a = 0; a < 3; a++) {
        for (int j = 0; j < 3; j++) {
          for (int b = 0; b < 3; b++) {
            for (int l = 0; l < 3; l++) {
              for (int c = 0; c < 3; c++) {
                double term = epsilon(i, j, l) * epsilon(a, b, c) * F[l][c];
                H[3 * i + a][3 * j + b] += term;
                cof[3 * i + a] += .5 * term * F[j][b];
              }
            }
          }
        }
      }
    }
    double J = F[0][0] * cof[0] + F[0][1] * cof[1] + F[0][2] * cof[2];
    double volume = std::abs(1 / k[23]) / 6;
    double muV = -72 * k[21] * volume * volume;
    double lambdaV = 2 * k[22] - muV;
    double scale = 0;
    for (int i = 0; i < 9; i++) {
      for (int j = 0; j < 9; j++) {
        H[i][j] = (i == j ? muV : 0) + lambdaV * cof[i] * cof[j] +
                  (lambdaV * (J - 1) - muV) * H[i][j];
        scale = std::max(scale, std::abs(H[i][j]));
      }
    }
    if (project) {
      // Dense cyclic Jacobi reference: no SVD, analytic modes, or engine
      // eigensolver.
      double Q[9][9] = {};
      for (int i = 0; i < 9; i++) Q[i][i] = 1;
      bool converged = false;
      for (int sweep = 0; sweep < 100 && !converged; sweep++) {
        converged = true;
        for (int p = 0; p < 9; p++) {
          for (int q = p + 1; q < 9; q++) {
            if (std::abs(H[p][q]) <= 1e-14 * scale) continue;
            converged = false;
            double angle = .5 * std::atan2(2 * H[p][q], H[q][q] - H[p][p]);
            double c = std::cos(angle), s = std::sin(angle);
            double pp = H[p][p], pq = H[p][q], qq = H[q][q];
            H[p][p] = c * c * pp - 2 * c * s * pq + s * s * qq;
            H[q][q] = s * s * pp + 2 * c * s * pq + c * c * qq;
            H[p][q] = H[q][p] = 0;
            for (int r = 0; r < 9; r++) {
              if (r != p && r != q) {
                double rp = H[r][p], rq = H[r][q];
                H[r][p] = H[p][r] = c * rp - s * rq;
                H[r][q] = H[q][r] = s * rp + c * rq;
              }
              double rp = Q[r][p], rq = Q[r][q];
              Q[r][p] = c * rp - s * rq;
              Q[r][q] = s * rp + c * rq;
            }
          }
        }
      }
      EXPECT_TRUE(converged);
      double value[9];
      for (int i = 0; i < 9; i++) value[i] = std::max(0.0, H[i][i]);
      for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
          H[i][j] = 0;
          for (int l = 0; l < 9; l++) H[i][j] += value[l] * Q[i][l] * Q[j][l];
        }
      }
    }
    double gradient[4][3] = {};
    for (int v = 1; v < 4; v++) {
      for (int x = 0; x < 3; x++) {
        gradient[v][x] = inverse[v - 1][x];
        gradient[0][x] -= gradient[v][x];
      }
    }
    for (int i = 0; i < 4; i++) {
      int adr_i = m->body_dofadr[m->flex_vertbodyid[vert[i]]];
      for (int j = 0; j < 4; j++) {
        int adr_j = m->body_dofadr[m->flex_vertbodyid[vert[j]]];
        for (int x = 0; x < 3; x++) {
          for (int y = 0; y < 3; y++) {
            for (int a = 0; a < 3; a++) {
              for (int b = 0; b < 3; b++) {
                result[(adr_i + x) * m->nv + adr_j + y] +=
                    gradient[i][a] * H[3 * x + a][3 * y + b] * gradient[j][b];
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// Projection precedes assembly across shared edges, remains PSD through
// degeneracy and inversion, and makes cached damping dissipative.
TEST_F(DerivativeTest, SNHStiffnessThroughInversion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" integrator="discrete"/>
    <worldbody>
      <flexcomp name="tet" type="direct" dim="3" mass="1"
                point="0 0 0  1 0 0  .2 .9 0  -.1 .3 1.1  .1 .2 -.8"
                element="0 1 2 3  0 2 1 4">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1000" poisson=".3" damping=".02"/>
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
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  // Two tetrahedra share three edges with different local orientations.
  std::vector<mjtNum> rest(d->flexvert_xpos, d->flexvert_xpos + 15);
  int nv = m->nv;
  ASSERT_EQ(nv, 15);
  bool negative_curvature = false;
  for (int rank = 0; rank <= 3; rank++) {
    for (mjtNum scale :
         {mjtNum(-1), mjtNum(-.01), mjtNum(0), mjtNum(.01), mjtNum(1.05)}) {
      for (int v = 0; v < 5; v++) {
        int adr = m->body_dofadr[m->flex_vertbodyid[v]];
        mjtNum x = rank > 0 ? 1.05 * rest[3 * v] : 0;
        mjtNum y = rank > 1 ? 1.05 * rest[3 * v + 1] : 0;
        mjtNum z = rank > 2 ? scale * rest[3 * v + 2] : 0;
        if (rank == 3 && scale > 1) {
          // Distinct singular values and nontrivial left/right singular
          // vectors.
          x = 1.06 * (.8 * rest[3 * v] - .6 * rest[3 * v + 1]);
          y = 1.09 * (.6 * rest[3 * v] + .8 * rest[3 * v + 1]);
          z = 1.12 * rest[3 * v + 2];
        }
        d->qpos[adr] = .8 * x - .6 * y - rest[3 * v];
        d->qpos[adr + 1] = .6 * x + .8 * y - rest[3 * v + 1];
        d->qpos[adr + 2] = z - rest[3 * v + 2];
      }
      for (int i = 0; i < nv; i++) d->qvel[i] = mju_Halton(i, 2) - .5;
      mj_forward(m.get(), d.get());
      std::vector<mjtNum> K(nv * nv, 0);
      stretchK_dense(m.get(), d.get(), K.data(), nv, 1, 0);
      std::vector<int> rownnz(nv), rowadr(nv);
      int nnz =
          mjd_flexStiff_assemble(m.get(), d.get(), rownnz.data(), rowadr.data(),
                                 nullptr, nullptr, 1, 0, 0, 1, nullptr);
      std::vector<int> colind(nnz);
      std::vector<mjtNum> values(nnz);
      mjd_flexStiff_assemble(m.get(), d.get(), rownnz.data(), rowadr.data(),
                             colind.data(), values.data(), 1, 0, 0, 1, nullptr);
      for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nv; j++) {
          EXPECT_TRUE(std::isfinite(K[i * nv + j]));
          EXPECT_NEAR(K[i * nv + j], K[j * nv + i], MjTol(1e-10, 1e-3));
        }
        for (int j = rowadr[i]; j < rowadr[i] + rownnz[i]; j++) {
          EXPECT_NEAR(values[j], K[i * nv + colind[j]], MjTol(1e-10, 1e-3));
        }
      }
      std::vector<double> expected =
          SNHHessianReference(m.get(), d.get(), rest, true);
      std::vector<double> exact =
          SNHHessianReference(m.get(), d.get(), rest, false);
      double error = 0, norm = 0;
      for (int i = 0; i < nv * nv; i++) {
        error += (K[i] - expected[i]) * (K[i] - expected[i]);
        norm += expected[i] * expected[i];
      }
      EXPECT_LT(std::sqrt(error / norm), MjTol(1e-13, 5e-6));
      std::vector<mjtNum> unprojected(exact.begin(), exact.end());
      for (int i = 0; i < nv; i++) {
        unprojected[i * nv + i] += MjEps(1e-8, .01);
      }
      negative_curvature |= mju_cholFactor(unprojected.data(), nv, 0) < nv;
      std::vector<mjtNum> L = K;
      for (int i = 0; i < nv; i++) {
        L[i * nv + i] += MjEps(1e-8, .01);
      }
      EXPECT_EQ(mju_cholFactor(L.data(), nv, 0), nv);
      std::vector<mjtNum> damp(nv, 0);
      mjd_flexStretch_mul(m.get(), d.get(), damp.data(), d->qvel, 0, 1);
      for (int i = 0; i < nv; i++) {
        EXPECT_NEAR(damp[i], -d->qfrc_damper[i], MjTol(1e-10, 1e-3));
      }
      EXPECT_LE(mju_dot(d->qvel, d->qfrc_damper, nv), MjTol(1e-10, 1e-3));
      // Central differences still match the unprojected energy Hessian: the
      // spring force is unchanged, even where the solver metric is projected.
      m->flex_damping[0] = 0;
      mjtNum eps = MjEps(1e-6, 1e-3);
      for (int j = 0; j < nv; j++) {
        mjtNum saved = d->qpos[j];
        d->qpos[j] = saved + eps;
        mj_forward(m.get(), d.get());
        std::vector<mjtNum> plus(d->qfrc_spring, d->qfrc_spring + nv);
        d->qpos[j] = saved - eps;
        mj_forward(m.get(), d.get());
        for (int i = 0; i < nv; i++) {
          mjtNum fd = -(plus[i] - d->qfrc_spring[i]) / (2 * eps);
          EXPECT_NEAR(exact[i * nv + j], fd, MjTol(1e-6, .2));
        }
        d->qpos[j] = saved;
      }
      m->flex_damping[0] = .02;
    }
  }
  EXPECT_TRUE(negative_curvature)
      << "the test must exercise negative modes that require projection";
}

// Rotated irregular reference tetrahedra, both windings, material ratios, and
// length scales. The dense oracle checks the full projection, not only PSD.
TEST_F(DerivativeTest, SNHProjectionSpectralReference) {
  struct SpectralCase {
    mjtNum quat_u[4];
    mjtNum quat_v[4];
    mjtNum sigma[3];
  };
  const SpectralCase kCases[] = {
      // Standard positive stretches
      {{1, 0, 0, 0}, {1, 0, 0, 0}, {1.5, 1.2, 0.8}},
      {{0.5, 0.5, 0.5, 0.5}, {0.5, -0.5, 0.5, -0.5}, {1.0, 1.0, 1.0}},
      {{0.6, 0.8, 0, 0}, {0, 0.6, 0.8, 0}, {0.7, 0.7, 0.7}},
      {{0.5, -0.5, -0.5, 0.5}, {0.6, 0, 0, 0.8}, {2.5, 1.0, 0.4}},

      // Repeated singular values
      {{0.5, 0.5, 0.5, 0.5}, {1, 0, 0, 0}, {1.2, 1.2, 0.6}},
      {{0.6, 0, 0.8, 0}, {0.5, 0.5, 0.5, 0.5}, {1.4, 0.8, 0.8}},

      // Inverted / negative determinant
      {{1, 0, 0, 0}, {0.5, 0.5, 0.5, 0.5}, {1.2, 0.9, -0.5}},
      {{0.5, -0.5, 0.5, -0.5}, {0.6, 0, 0.8, 0}, {1.0, 1.0, -1.0}},
      {{0.6, 0.8, 0, 0}, {1, 0, 0, 0}, {2.0, 0.8, -0.6}},

      // Near-zero / transition singular values
      {{0.6, 0, 0, 0.8},
       {0.5, -0.5, 0.5, -0.5},
       {1.0, 0.8, -MjEps(1e-8, 1e-4)}},
      {{0.5, 0.5, 0.5, 0.5}, {0.6, 0.8, 0, 0}, {1.0, 0.8, MjEps(1e-8, 1e-4)}},

      // Rank deficient / collapsed
      {{1, 0, 0, 0}, {0.6, 0.8, 0, 0}, {1.2, 0.8, 0.0}},
      {{0.5, 0.5, 0.5, 0.5}, {1, 0, 0, 0}, {1.5, 0.0, 0.0}},
      {{1, 0, 0, 0}, {1, 0, 0, 0}, {0.0, 0.0, 0.0}},
  };

  const mjtNum reference[4][3] = {
      {0, 0, 0}, {1, 0, 0}, {.2, .9, 0}, {-.1, .3, 1.1}};
  const mjtNum rotation[9] = {.36, -.48, .8, .8, .6, 0, -.48, .64, .6};
  for (mjtNum size : {mjtNum(.01), mjtNum(1), mjtNum(100)}) {
    for (bool reverse : {false, true}) {
      for (mjtNum poisson : {mjtNum(0), mjtNum(.3), mjtNum(.499)}) {
        SCOPED_TRACE(size);
        SCOPED_TRACE(reverse);
        SCOPED_TRACE(poisson);
        std::string points;
        for (int v = 0; v < 4; v++) {
          mjtNum point[3];
          mju_mulMatVec3(point, rotation, reference[v]);
          for (int x = 0; x < 3; x++) {
            points += std::to_string(size * point[x]) + " ";
          }
        }
        std::string xml =
            "<mujoco><option gravity='0 0 0' integrator='discrete'/><worldbody>"
            "<flexcomp name='tet' type='direct' dim='3' mass='1' point='" +
            points + "' element='" + (reverse ? "0 2 1 3" : "0 1 2 3") +
            "'>"
            "<contact contype='0' conaffinity='0' selfcollide='none'/>"
            "<elasticity young='1000' poisson='" +
            std::to_string(poisson) + "'/></flexcomp></worldbody></mujoco>";
        mjSpec* spec = mj_parseXMLString(xml.c_str(), nullptr, nullptr, 0);
        ASSERT_THAT(spec, NotNull());
        mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "tet"))->elastic3d = 1;
        MjModelPtr m(mj_compile(spec, nullptr));
        mj_deleteSpec(spec);
        ASSERT_THAT(m.get(), NotNull());
        MjDataPtr d = MakeData(m);
        mj_forward(m.get(), d.get());
        ASSERT_EQ(m->nv, 12);
        std::vector<mjtNum> rest(d->flexvert_xpos, d->flexvert_xpos + 12);
        int case_idx = 0;
        for (const auto& c : kCases) {
          SCOPED_TRACE(case_idx++);
          mjtNum U[9], V[9];
          mju_quat2Mat(U, c.quat_u);
          mju_quat2Mat(V, c.quat_v);
          mjtNum F[9] = {0};
          for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
              for (int k = 0; k < 3; k++) {
                F[3 * i + j] += U[3 * i + k] * c.sigma[k] * V[3 * j + k];
              }
            }
          }
          for (int v = 0; v < 4; v++) {
            mjtNum point[3];
            mju_mulMatVec3(point, F, rest.data() + 3 * v);
            int adr = m->body_dofadr[m->flex_vertbodyid[v]];
            for (int x = 0; x < 3; x++) {
              d->qpos[adr + x] = point[x] - rest[3 * v + x];
            }
          }
          mj_forward(m.get(), d.get());
          std::vector<mjtNum> K(144);
          stretchK_dense(m.get(), d.get(), K.data(), 12, 1, 0);
          std::vector<double> expected =
              SNHHessianReference(m.get(), d.get(), rest, true);
          double error = 0, norm = 0;
          for (int i = 0; i < 144; i++) {
            ASSERT_TRUE(std::isfinite(K[i]));
            error += (K[i] - expected[i]) * (K[i] - expected[i]);
            norm += expected[i] * expected[i];
          }
          EXPECT_LT(std::sqrt(error / norm), MjTol(2e-13, 1e-4));
          for (int row = 0; row < 12; row++) {
            for (int axis = 0; axis < 3; axis++) {
              mjtNum translation = 0;
              for (int v = 0; v < 4; v++) {
                translation += K[12 * row + 3 * v + axis];
              }
              EXPECT_LE(mju_abs(translation) / std::sqrt(norm),
                        MjTol(2e-15, 1e-6));
            }
          }
        }
      }
    }
  }
}

// K_stretch must be the full Hessian of the stretch force, not just its
// Gauss-Newton part: the geometric (stress-proportional) term is what makes it
// the Jacobian at finite strain. Uniformly dilating the mesh puts every edge in
// tension, so the tensile clamp is inactive and the operator is exact -- with
// only the Gauss-Newton term the finite-difference error is a large fraction of
// the force. FlexStretchDerivatives covers the near-rest limit, where the two
// agree anyway.
TEST_F(DerivativeTest, FlexStretchDerivativesTensile) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.1 0.1 0.1"
                radius=".01" dim="2" mass="1" pos="0 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01"
                    elastic2d="stretch" damping="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  ASSERT_EQ(model->nq, nv);  // all slide dofs
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // dilate about the flex centroid: every edge stretches by 5%, so every Me is
  // strictly positive
  mjtNum centroid[3] = {0, 0, 0};
  int nvert = model->flex_vertnum[0];
  for (int v = 0; v < nvert; v++) {
    mju_addTo3(centroid, data->flexvert_xpos + 3 * v);
  }
  mju_scl3(centroid, centroid, 1.0 / nvert);
  for (int v = 0; v < nvert; v++) {
    int body = model->flex_vertbodyid[model->flex_vertadr[0] + v];
    int adr = model->body_dofadr[body];
    for (int x = 0; x < 3; x++) {
      data->qpos[adr + x] +=
          0.05 * (data->flexvert_xpos[3 * v + x] - centroid[x]);
    }
  }
  mj_forward(model.get(), data.get());

  std::vector<mjtNum> vec(nv), res(nv, 0);
  for (int i = 0; i < nv; i++) {
    vec[i] = mju_Halton(i, 2) - 0.5;
  }
  mjd_flexStretch_mul(model.get(), data.get(), res.data(), vec.data(), 1, 0);

  mjtNum eps = MjEps(1e-7, 1e-4);
  mjData* perturbed = mj_copyData(NULL, model.get(), data.get());
  mju_addToScl(perturbed->qpos, vec.data(), eps, nv);
  mj_forward(model.get(), perturbed);

  mjtNum max_err = 0, scale = 0;
  for (int i = 0; i < nv; ++i) {
    mjtNum fd = -(perturbed->qfrc_passive[i] - data->qfrc_passive[i]) / eps;
    max_err = mju_max(max_err, mju_abs(res[i] - fd));
    scale = mju_max(scale, mju_abs(fd));
  }
  mj_deleteData(perturbed);

  EXPECT_GT(scale, 1e-3) << "test should exercise nontrivial stretch stiffness";
  EXPECT_LT(max_err, MjTol(1e-4, 1e-3) * scale)
      << "mjd_flexStretch_mul is not the Jacobian of the flex stretch force";
}

// verify mjd_flexStretch_mul (stiffness of the standard-flex stretch force)
// against finite differences of qfrc_passive, plus symmetry, positive
// semi-definiteness and (s1, s2) scale linearity. The model covers both
// element edge tables (dim=2 triangles and dim=3 tets) and a pinned vertex
// (zero-dof body guard).
TEST_F(DerivativeTest, FlexStretchDerivatives) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.1 0.1 0.1"
                radius=".01" dim="2" mass="1" pos="0 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01"
                    elastic2d="stretch" damping="50"/>
        <pin id="0"/>
      </flexcomp>
      <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                radius=".01" dim="3" mass="1" pos="1 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="10"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  ASSERT_EQ(model->nq, nv);  // all slide dofs

  MjDataPtr data = MakeData(model);

  // deform both flexes deterministically, at small strain.
  // FlexStretchDerivativesTensile covers finite strain, where the geometric
  // term of K_stretch is what carries the accuracy; here some edges are
  // compressed and the operator drops their geometric term to stay PSD, so the
  // tolerance stays loose.
  for (int i = 0; i < nv; i++) {
    data->qpos[i] += 5e-4 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());

  // part 1: FD verification of K*vec against qfrc_passive (qvel = 0, so the
  // kD elongation term vanishes and qfrc_passive is the pure stretch spring)
  {
    std::vector<mjtNum> vec(nv), res(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i, 2) - 0.5;
    }
    mjd_flexStretch_mul(model.get(), data.get(), res.data(), vec.data(), 1, 0);

    mjtNum eps = MjEps(1e-7, 1e-4);
    mjData* data_perturbed = mj_copyData(NULL, model.get(), data.get());
    mju_addToScl(data_perturbed->qpos, vec.data(), eps, nv);
    mj_forward(model.get(), data_perturbed);

    // qfrc_passive = -dV/dq  =>  -(qfrc_new - qfrc)/eps ~= K * vec.
    // Compare max error against the force scale rather than entrywise:
    // individual near-zero entries are not meaningful, and the dropped
    // compressive geometric term leaves a residual in both flexes.
    mjtNum max_err = 0, scale = 0;
    for (int i = 0; i < nv; ++i) {
      mjtNum fd =
          -(data_perturbed->qfrc_passive[i] - data->qfrc_passive[i]) / eps;
      max_err = mju_max(max_err, mju_abs(res[i] - fd));
      scale = mju_max(scale, mju_abs(fd));
    }
    EXPECT_GT(scale, 1.0) << "test should exercise nontrivial stiffness";
    EXPECT_LT(max_err, MjTol(5e-3, 5e-2) * scale)
        << "stretch stiffness mismatch: max_err " << max_err
        << " at force scale " << scale;
    mj_deleteData(data_perturbed);
  }

  // part 2: symmetry and positive semi-definiteness of the assembled K
  {
    std::vector<mjtNum> K(nv * nv, 0);
    stretchK_dense(model.get(), data.get(), K.data(), nv, 1, 0);

    mjtNum max_asymmetry = 0;
    for (int i = 0; i < nv; i++) {
      for (int j = 0; j < i; j++) {
        max_asymmetry =
            mju_max(max_asymmetry, mju_abs(K[i * nv + j] - K[j * nv + i]));
      }
    }
    EXPECT_THAT(max_asymmetry, MjNear(0, 1e-10, 5e-4))
        << "K_stretch is not symmetric";

    for (int trial = 0; trial < 5; trial++) {
      std::vector<mjtNum> v(nv);
      for (int i = 0; i < nv; i++) {
        v[i] = mju_Halton(i + trial * nv, 3) - 0.5;
      }
      mjtNum vKv = 0;
      for (int i = 0; i < nv; i++) {
        for (int j = 0; j < nv; j++) {
          vKv += v[i] * K[i * nv + j] * v[j];
        }
      }
      EXPECT_GE(vKv, MjTol(-1e-8, -1e-5)) << "K_stretch is not PSD";
    }
  }

  // part 3: (s1, s2) scale linearity across flexes with different damping:
  // mul(s1, s2) == s1*mul(1, 0) + s2*mul(0, 1)
  {
    std::vector<mjtNum> vec(nv), a(nv, 0), b(nv, 0), c(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i, 5) - 0.5;
    }
    mjtNum h = 1e-3;
    mjd_flexStretch_mul(model.get(), data.get(), a.data(), vec.data(), h * h,
                        h);
    mjd_flexStretch_mul(model.get(), data.get(), b.data(), vec.data(), 1, 0);
    mjd_flexStretch_mul(model.get(), data.get(), c.data(), vec.data(), 0, 1);
    for (int i = 0; i < nv; i++) {
      EXPECT_THAT(a[i], MjNear(h * h * b[i] + h * c[i], 1e-12, 1e-5))
          << "scale linearity mismatch at DOF " << i;
    }
  }
}

// verify mjd_flexStiff_assemble against the matrix-free operators: the
// assembled CSR applied to test vectors must reproduce mjd_flexBend_mul +
// mjd_flexStretch_mul at the same state
TEST_F(DerivativeTest, FlexStiffAssemble) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.1 0.1 0.1"
                radius=".01" dim="2" mass="1" pos="0 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" thickness="0.01"
                    elastic2d="both" damping="7"/>
      </flexcomp>
      <flexcomp name="solid" type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                radius=".01" dim="3" mass="1" pos="1 0 1">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="10"/>
        <pin id="0"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  MjDataPtr data = MakeData(model);

  // deform deterministically
  for (int i = 0; i < nv; i++) {
    data->qpos[i] += 2e-3 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());

  // assemble both terms with a mixed (s1, s2) scale
  mjtNum s1 = 4e-6, s2 = 2e-3;
  std::vector<int> rownnz(nv), rowadr(nv);
  int nnz = mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                                   rowadr.data(), NULL, NULL, s1, s2,
                                   /*flg_bend=*/1, /*flg_stretch=*/1, NULL);
  ASSERT_GT(nnz, 0);
  std::vector<int> colind(nnz);
  std::vector<mjtNum> val(nnz);
  mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(), rowadr.data(),
                         colind.data(), val.data(), s1, s2, /*flg_bend=*/1,
                         /*flg_stretch=*/1, NULL);

  // compare CSR apply vs operators on test vectors
  for (int trial = 0; trial < 3; trial++) {
    std::vector<mjtNum> vec(nv), res_op(nv, 0), res_csr(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i + trial * nv, 3) - 0.5;
    }
    mjd_flexBend_mul(model.get(), data.get(), res_op.data(), vec.data(), s1,
                     s2);
    mjd_flexStretch_mul(model.get(), data.get(), res_op.data(), vec.data(), s1,
                        s2);
    for (int i = 0; i < nv; i++) {
      mjtNum sum = 0;
      for (int k = 0; k < rownnz[i]; k++) {
        sum += val[rowadr[i] + k] * vec[colind[rowadr[i] + k]];
      }
      res_csr[i] = sum;
    }
    for (int i = 0; i < nv; i++) {
      EXPECT_THAT(res_csr[i], MjNear(res_op[i], 1e-12, 2e-5))
          << "assembly/operator mismatch at DOF " << i << " trial " << trial;
    }
  }
}

// 1D rods pinned at their first vertex (free first edge, per-edge twist joints)
static const char* const kRodStraightPinned = R"(
<mujoco>
  <option integrator="discrete" gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="20 30 40">
      <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="1">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";
static const char* const kRodHelixPinned = R"(
<mujoco>
  <option integrator="discrete" gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="20 30 40">
      <flexcomp name="helix" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2  2 3  3 4  4 5"
                point="0.15 0.00 0.00   0.106 0.106 0.04   0.00 0.15 0.08
                       -0.106 0.106 0.12   -0.15 0.00 0.16
                       -0.106 -0.106 0.20">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";
static const char* const kRodVeePinned = R"(
<mujoco>
  <option integrator="discrete" gravity="0 0 0"/>
  <worldbody>
    <body name="turned" euler="20 30 40">
      <flexcomp name="vee" type="direct" dim="1" radius="0.01" mass="1"
                element="0 1  1 2" point="0 0 0  0.1 0 0  0.18 0.06 0.02">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

// 1D rod bending stiffness: at rest (straight, hinge pairs, and a curved single
// hinge) the energy gradient vanishes, so the PSD Gauss-Newton Hessian used by
// the metric is exact. Checked against central differences of the spring force
TEST_F(DerivativeTest, FlexRodHessianAtRest) {
  for (const char* xml : {kRodStraightPinned, kRodHelixPinned, kRodVeePinned}) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    SCOPED_TRACE(model->names + model->name_flexadr[0]);
    int nv = model->nv;
    MjDataPtr data = MakeData(model);
    mj_forward(model.get(), data.get());

    // freeze the director state at rest, so finite differences see a fixed one
    for (int f = 0; f < model->nflex; f++) {
      if (model->flex_plasticadr[f] >= 0) {
        mj_rodChain(model.get(), data.get(), f, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr,
                    data->plastic + model->flex_plasticadr[f]);
      }
    }

    vector<mjtNum> vec(nv), res(nv, 0);
    for (int i = 0; i < nv; i++) {
      int jnt = model->dof_jntid[i];
      vec[i] =
          (model->jnt_type[jnt] == mjJNT_SLIDE) ? mju_Halton(i, 3) - 0.5 : 0;
    }
    mjd_flexStretch_mul(model.get(), data.get(), res.data(), vec.data(), 1, 0);

    // central differences of the spring force
    mjtNum eps = MjEps(1e-6, 1e-3);
    mjData* plus = mj_copyData(NULL, model.get(), data.get());
    mjData* minus = mj_copyData(NULL, model.get(), data.get());
    mju_addToScl(plus->qpos, vec.data(), eps, nv);
    mju_addToScl(minus->qpos, vec.data(), -eps, nv);
    mj_forward(model.get(), plus);
    mj_forward(model.get(), minus);

    mjtNum max_err = 0, scale = 0;
    for (int i = 0; i < nv; i++) {
      if (model->jnt_type[model->dof_jntid[i]] != mjJNT_SLIDE) continue;
      mjtNum fd = -(plus->qfrc_spring[i] - minus->qfrc_spring[i]) / (2 * eps);
      max_err = mju_max(max_err, mju_abs(res[i] - fd));
      scale = mju_max(scale, mju_abs(fd));
    }
    EXPECT_GT(scale, 1e-3)
        << "test should exercise nontrivial bending stiffness";
    EXPECT_LT(max_err, MjTol(1e-5, 1e-2) * scale)
        << "rod bending Hessian is not the Jacobian of the force at rest";
    mj_deleteData(plus);
    mj_deleteData(minus);
  }
}

// twist-twist block of the metric (1D DER rods with per-edge twist joints): the
// torsion tridiagonal 2 kt plus the bending stiffness of each twist angle
// (mj_rodChain), against central differences of the spring force in the
// twist angles. Exact at rest (straight, helix, vee and a curved box ribbon,
// whose material curvatures give the twist angles bending stiffness); away from
// rest (a bent box ribbon) the off-diagonals stay exact and the diagonal only
// gains where a hinge side's second derivative is negative and clamped to 0
TEST_F(DerivativeTest, FlexRodTwistHessian) {
  static const char* const kBoxArc = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="arc" type="direct" dim="1" radius="0.005" mass="0.1"
                element="0 1  1 2  2 3  3 4  4 5"
                point="0.2 0 0  0.190211 0.0618034 0  0.161803 0.117557 0
                       0.117557 0.161803 0  0.0618034 0.190211 0  0 0.2 0">
        <pin id="0"/>
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" shear="4e6" section="box"
                    sectionsize="0.012 0.003"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  struct Case {
    const char* xml;
    bool bend;  // deform the vertices away from rest
  };
  for (Case c : {Case{kRodStraightPinned, false}, Case{kRodHelixPinned, false},
                 Case{kRodVeePinned, false}, Case{kBoxArc, false},
                 Case{kBoxArc, true}}) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(c.xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    SCOPED_TRACE(std::string(model->names + model->name_flexadr[0]) +
                 (c.bend ? " bent" : " rest"));
    model->flex_damping[0] = 0;  // the stiffness part only: B = h^2 K
    const mjtNum h = model->opt.timestep;
    int nv = model->nv;
    MjDataPtr data = MakeData(model);
    if (c.bend) {
      for (int j = 0; j < model->njnt; j++) {
        if (model->jnt_type[j] == mjJNT_SLIDE) {
          int qadr = model->jnt_qposadr[j];
          data->qpos[qadr] = 0.004 * std::sin(1.3 * qadr + 0.4);
        }
      }
    }
    mj_forward(model.get(), data.get());
    for (int f = 0; f < model->nflex; f++) {
      if (model->flex_plasticadr[f] >= 0) {
        mj_rodChain(model.get(), data.get(), f, nullptr, nullptr, nullptr,
                    nullptr, nullptr, nullptr,
                    data->plastic + model->flex_plasticadr[f]);
      }
    }
    mj_forward(model.get(), data.get());

    int nedge =
        static_cast<int>(model->flex_bending[model->flex_bendingadr[0]]) + 1;
    vector<int> dofs;
    for (int e = 0; e < nedge; e++) {
      int jnt = mj_rodChainTwistJnt(model.get(), 0, e);
      ASSERT_GE(jnt, 0) << "edge " << e;
      dofs.push_back(model->jnt_dofadr[jnt]);
    }
    int n = dofs.size();

    // metric columns and central differences of the spring force
    mjtNum eps = MjEps(1e-6, 1e-3), scale = 0, max_err = 0, kbend = 0;
    vector<mjtNum> vec(nv), res(nv);
    for (int j = 0; j < n; j++) {
      mju_zero(vec.data(), nv);
      mju_zero(res.data(), nv);
      vec[dofs[j]] = 1;
      mj_effMulAdd(model.get(), data.get(), res.data(), vec.data(), 0);
      int qadr = model->jnt_qposadr[model->dof_jntid[dofs[j]]];
      mjData* plus = mj_copyData(NULL, model.get(), data.get());
      mjData* minus = mj_copyData(NULL, model.get(), data.get());
      plus->qpos[qadr] += eps;
      minus->qpos[qadr] -= eps;
      mj_forward(model.get(), plus);
      mj_forward(model.get(), minus);
      for (int i = 0; i < n; i++) {
        mjtNum fd =
            -(plus->qfrc_spring[dofs[i]] - minus->qfrc_spring[dofs[i]]) /
            (2 * eps);
        mjtNum ana = res[dofs[i]] / (h * h);
        mjtNum err = ana - fd;
        if (c.bend && i == j) {
          err = mju_min(0, err);  // PSD clamp: the diagonal may only gain
        }
        max_err = mju_max(max_err, mju_abs(err));
        scale = mju_max(scale, mju_abs(fd));
      }
      kbend = mju_max(kbend, data->efm_ck[dofs[j]] / h);
      mj_deleteData(plus);
      mj_deleteData(minus);
    }
    EXPECT_GT(scale, 0);
    EXPECT_LT(max_err, MjTol(1e-6, 1e-2) * scale);

    // curved and bent rods exercise the bending stiffness of the twist angles
    bool straight = (c.xml == kRodStraightPinned) && !c.bend;
    if (straight) {
      EXPECT_LT(kbend, MjTol(1e-9, 1e-4) * scale);
    } else {
      EXPECT_GT(kbend, 1e-3 * scale);
    }
  }
}

// 1D rod bending stiffness away from rest: symmetric, positive semi-definite,
// assembled CSR matches the matrix-free operator
TEST_F(DerivativeTest, FlexRodStiffAssemble) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" gravity="0 0 0"/>
    <worldbody>
      <body name="turned" euler="20 30 40">
        <flexcomp name="rod" type="grid" count="6 1 1" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="1" mass="1">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" damping="0.1"/>
        </flexcomp>
        <flexcomp name="ring" type="circle" count="7 1 1"
                  spacing="0.12 0.12 0.12"
                  pos="0 0.5 0" radius="0.01" dim="1" mass="1">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" damping="0.1"/>
        </flexcomp>
        <flexcomp name="vee" type="direct" dim="1" radius="0.01" mass="1"
                  element="0 1  1 2" point="0 0 0  0.1 0 0  0.18 0.06 0.02"
                  pos="0 -0.5 0">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e7" damping="0.1"/>
        </flexcomp>
      </body>
      <flexcomp name="free" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                radius="0.01" dim="1" mass="1" pos="1 0 0">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e7" damping="0.1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  MjDataPtr data = MakeData(model);
  for (int i = 0; i < nv; i++) {
    data->qpos[i] += 2e-2 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());

  mjtNum s1 = 4e-6, s2 = 2e-3;
  vector<int> rownnz(nv), rowadr(nv);
  int nnz = mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                                   rowadr.data(), NULL, NULL, s1, s2,
                                   /*flg_bend=*/1, /*flg_stretch=*/1, NULL);
  ASSERT_GT(nnz, 0);
  vector<int> colind(nnz);
  vector<mjtNum> val(nnz);
  mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(), rowadr.data(),
                         colind.data(), val.data(), s1, s2, /*flg_bend=*/1,
                         /*flg_stretch=*/1, NULL);

  // the constant-stencil (setconst) assembly excludes state-dependent rod
  // bending
  vector<int> rownnz0(nv), rowadr0(nv);
  EXPECT_EQ(mjd_flexStiff_assemble(model.get(), data.get(), rownnz0.data(),
                                   rowadr0.data(), NULL, NULL, s1, s2,
                                   /*flg_bend=*/1, /*flg_stretch=*/0, NULL),
            0);

  // dense copy of the operator, column by column
  vector<mjtNum> K(nv * nv, 0), csr(nv * nv, 0);
  for (int j = 0; j < nv; j++) {
    vector<mjtNum> e(nv, 0), col(nv, 0);
    e[j] = 1;
    mjd_flexBend_mul(model.get(), data.get(), col.data(), e.data(), s1, s2);
    mjd_flexStretch_mul(model.get(), data.get(), col.data(), e.data(), s1, s2);
    for (int i = 0; i < nv; i++) {
      K[i * nv + j] = col[i];
    }
  }
  for (int i = 0; i < nv; i++) {
    for (int k = 0; k < rownnz[i]; k++) {
      csr[i * nv + colind[rowadr[i] + k]] += val[rowadr[i] + k];
    }
  }

  mjtNum kmax = 0;
  for (mjtNum k : K) kmax = mju_max(kmax, mju_abs(k));
  ASSERT_GT(kmax, 0);
  for (int i = 0; i < nv; i++) {
    for (int j = 0; j < nv; j++) {
      mjtNum tol = MjTol(1e-12, 1e-5) * kmax;
      EXPECT_NEAR(K[i * nv + j], K[j * nv + i], tol) << i << ", " << j;
      EXPECT_NEAR(csr[i * nv + j], K[i * nv + j], tol) << i << ", " << j;
    }
  }

  // positive semi-definite: x'Kx >= 0 for a set of test vectors
  for (int trial = 0; trial < 10; trial++) {
    vector<mjtNum> x(nv), Kx(nv, 0);
    for (int i = 0; i < nv; i++) {
      x[i] = mju_Halton(i + trial * nv, 5) - 0.5;
    }
    mju_mulMatVec(Kx.data(), K.data(), x.data(), nv, nv);
    EXPECT_GE(mju_dot(x.data(), Kx.data(), nv), -1e-12 * kmax * nv);
  }
}

// verify the interp assembly mode: with the K_rot cache supplied, the assembled
// CSR applied to test vectors must reproduce mjd_flexInterp_mul, whose sign
// convention is negated
TEST_F(DerivativeTest, FlexStiffAssembleInterp) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="soft" type="grid" count="4 4 4" spacing="0.1 0.1 0.1"
                radius=".01" dim="3" mass="1" pos="0 0 1" dof="trilinear">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  MjDataPtr data = MakeData(model);
  ASSERT_EQ(mjd_flexInterpAssemblable(model.get()), 1);

  // deform deterministically, refresh kinematics, cache the corotated stiffness
  for (int i = 0; i < nv; i++) {
    data->qpos[i] += 2e-3 * (mju_Halton(i, 2) - 0.5);
  }
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> krot(model->nflexstiffness, 0);
  mjd_flexInterp_cacheKrot(model.get(), data.get(), krot.data());

  // assemble interp only
  mjtNum s1 = 4e-6, s2 = 2e-3;
  std::vector<int> rownnz(nv), rowadr(nv);
  int nnz = mjd_flexStiff_assemble(
      model.get(), data.get(), rownnz.data(), rowadr.data(), NULL, NULL, s1, s2,
      /*flg_bend=*/0, /*flg_stretch=*/0, krot.data());
  ASSERT_GT(nnz, 0);
  std::vector<int> colind(nnz);
  std::vector<mjtNum> val(nnz);
  mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(), rowadr.data(),
                         colind.data(), val.data(), s1, s2, /*flg_bend=*/0,
                         /*flg_stretch=*/0, krot.data());

  // compare CSR apply vs the operator called with negated scales (its
  // convention)
  for (int trial = 0; trial < 3; trial++) {
    std::vector<mjtNum> vec(nv), res_op(nv, 0), res_csr(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i + trial * nv, 3) - 0.5;
    }
    mjd_flexInterp_mul(model.get(), data.get(), res_op.data(), vec.data(), -s1,
                       -s2, krot.data());
    for (int i = 0; i < nv; i++) {
      mjtNum sum = 0;
      for (int k = 0; k < rownnz[i]; k++) {
        sum += val[rowadr[i] + k] * vec[colind[rowadr[i] + k]];
      }
      res_csr[i] = sum;
    }
    for (int i = 0; i < nv; i++) {
      EXPECT_THAT(res_csr[i], MjNear(res_op[i], 1e-12, 2e-5))
          << "interp assembly/operator mismatch at DOF " << i << " trial "
          << trial;
    }
  }
}

// verify shell-mode interpolated flex (flex_interp < 0, elastic2d="stretch")
// derivatives and CSR assembly: mjd_flexInterp_mul matches finite differences
// of qfrc_passive across rotations, is symmetric and PSD, includes the x-normal
// faces (where k_elem[0] == 0), and matches mjd_flexStiff_assemble
TEST_F(DerivativeTest, FlexInterpShellDerivatives) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete"/>
    <worldbody>
      <flexcomp name="flex" type="grid" count="4 4 4" spacing="0.1 0.2 0.3"
                cellcount="1 2 2" radius=".01" dim="3" mass="1" dof="trilinear">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e4" poisson="0.3" damping="5"
                    elastic2d="stretch" thickness="0.02"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  EXPECT_EQ(model->flex_interp[0], -1);
  ASSERT_EQ(mjd_flexInterpAssemblable(model.get()), 1);
  int nv = model->nv;
  MjDataPtr data = MakeData(model);

  for (mjtNum angle : {0.0, 0.5, mjPI / 2}) {
    RotateFlexGrid(model.get(), data.get(), "flex", angle);
    mj_forward(model.get(), data.get());

    // assemble dense K via mjd_flexInterp_mul (with damping=0, h=1)
    mjtNum save_damping = model->flex_damping[0];
    model->flex_damping[0] = 0;
    std::vector<mjtNum> H(nv * nv, 0);
    mulKD_dense(model.get(), data.get(), H.data(), nv, 1.0);
    model->flex_damping[0] = save_damping;

    std::vector<mjtNum> vec(nv), res(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i, 2) - 0.5;
    }
    mju_mulMatVec(res.data(), H.data(), vec.data(), nv, nv);

    // finite difference of mj_passive at zero velocity
    mjtNum eps = MjEps(1e-6, 1e-3);
    mjData* data_perturbed = mj_copyData(NULL, model.get(), data.get());
    mju_addToScl(data_perturbed->qpos, vec.data(), eps, nv);
    mj_forward(model.get(), data_perturbed);
    for (int i = 0; i < nv; ++i) {
      mjtNum fd =
          -(data_perturbed->qfrc_passive[i] - data->qfrc_passive[i]) / eps;
      EXPECT_THAT(res[i], MjNear(fd, 5e-3, 5.0))
          << "shell stiffness mismatch at DOF " << i << " angle " << angle;
    }
    mj_deleteData(data_perturbed);

    // interior nodes on x=0 (node 4) and x=max (node 13) only belong to faces 0
    // and 1 (where k_elem[0] == 0); verify their diagonal blocks are nonzero
    for (int node_idx : {4, 13}) {
      int bid = model->flex_nodebodyid[model->flex_nodeadr[0] + node_idx];
      int dadr = model->body_dofadr[bid];
      mjtNum trace = H[(dadr + 0) * nv + (dadr + 0)] +
                     H[(dadr + 1) * nv + (dadr + 1)] +
                     H[(dadr + 2) * nv + (dadr + 2)];
      EXPECT_GT(trace, 1.0) << "expected positive stiffness trace on x-face "
                               "interior node "
                            << node_idx;
    }

    // verify CSR assembly matches mjd_flexInterp_mul
    std::vector<mjtNum> krot(model->nflexstiffness, 0);
    mjd_flexInterp_cacheKrot(model.get(), data.get(), krot.data());
    mjtNum s1 = 4e-6, s2 = 2e-3;
    std::vector<int> rownnz(nv), rowadr(nv);
    int nnz = mjd_flexStiff_assemble(
        model.get(), data.get(), rownnz.data(), rowadr.data(), NULL, NULL, s1,
        s2, /*flg_bend=*/0, /*flg_stretch=*/0, krot.data());
    ASSERT_GT(nnz, 0);
    std::vector<int> colind(nnz);
    std::vector<mjtNum> val(nnz);
    mjd_flexStiff_assemble(model.get(), data.get(), rownnz.data(),
                           rowadr.data(), colind.data(), val.data(), s1, s2,
                           /*flg_bend=*/0, /*flg_stretch=*/0, krot.data());

    std::vector<mjtNum> res_op(nv, 0), res_csr(nv, 0);
    mjd_flexInterp_mul(model.get(), data.get(), res_op.data(), vec.data(), -s1,
                       -s2, krot.data());
    for (int i = 0; i < nv; i++) {
      mjtNum sum = 0;
      for (int k = 0; k < rownnz[i]; k++) {
        sum += val[rowadr[i] + k] * vec[colind[rowadr[i] + k]];
      }
      res_csr[i] = sum;
      EXPECT_THAT(res_csr[i], MjNear(res_op[i], 1e-12, 2e-5))
          << "shell CSR/operator mismatch at DOF " << i;
    }
  }
}

// verify passive contact metric and velocity shift on interpolated flexes
// (trilinear volumetric, trilinear shell elastic2d="bend", quadratic, and with
// pinned cage nodes) against central finite differences of qfrc_spring
TEST_F(DerivativeTest, FlexInterpPassiveContactDerivatives) {
  struct Case {
    const char* name;
    const char* dof;
    const char* elasticity;
    const char* pin;
  };
  const Case cases[] = {
      {"trilinear_vol", "trilinear",
       R"(<elasticity young="1e3" poisson="0.3" damping="0"/>)", ""},
      {"trilinear_shell_bend", "trilinear",
       R"(<elasticity young="1e3" poisson="0.3" damping="0" thickness="0.01" elastic2d="bend"/>)",
       ""},
      {"quadratic_vol", "quadratic",
       R"(<elasticity young="1e3" poisson="0.3" damping="0"/>)", ""},
      {"trilinear_pinned", "trilinear",
       R"(<elasticity young="1e3" poisson="0.3" damping="0"/>)",
       R"(<pin id="1 3 5 7"/>)"},
  };

  for (const Case& tc : cases) {
    SCOPED_TRACE(tc.name);
    std::string xml = std::string(R"(
    <mujoco>
      <option integrator="discrete" gravity="0 0 0" timestep="0.002"/>
      <worldbody>
        <geom name="floor" type="plane" size="2 2 0.1"/>
        <flexcomp name="box" type="grid" count="3 3 3" cellcount="1 1 1"
                  spacing="0.1 0.1 0.1" radius="0.01" dim="3" mass="1.0"
                  pos="0 0 0.09" dof=")") +
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
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    ASSERT_GT(d->nefmcon, 0);
    ASSERT_EQ(d->nefc, 0);

    // At rest qpos = 0, elastic force is zero so qfrc_spring equals the
    // contact force interpolated onto cage nodes via mj_contactBodyWeight.
    std::vector<mjtNum> expected_qfrc(m->nv, 0);
    for (int i = 0; i < d->ncon; i++) {
      const mjContact* con = d->contact + i;
      if (con->exclude != 4 || con->dist >= 0) continue;
      mjtNum k = mjd_flexContactStiffness(m.get(), d.get(), con, nullptr);
      ASSERT_GT(k, 0);
      mjtNum fn = -k * con->dist;
      mjtNum force[3] = {fn * con->frame[0], fn * con->frame[1],
                         fn * con->frame[2]};
      for (int side = 0; side < 2; side++) {
        if (con->flex[side] < 0) continue;
        int body[128];
        mjtNum weight[128];
        int nb =
            mj_contactBodyWeight(m.get(), d.get(), con, side, body, weight);
        mjtNum sign = (side == 0) ? -1.0 : 1.0;
        for (int j = 0; j < nb; j++) {
          int b = m->body_weldid[body[j]];
          if (m->body_dofnum[b] == 3) {
            int da = m->body_dofadr[b];
            expected_qfrc[da + 0] += sign * weight[j] * force[0];
            expected_qfrc[da + 1] += sign * weight[j] * force[1];
            expected_qfrc[da + 2] += sign * weight[j] * force[2];
          }
        }
      }
    }
    for (int i = 0; i < m->nv; i++) {
      EXPECT_NEAR(d->qfrc_spring[i], expected_qfrc[i],
                  MjTol(1e-9, 1e-3) * mju_max(1.0, mju_abs(expected_qfrc[i])))
          << "dof " << i;
    }

    int nv = m->nv;
    mjtNum h = m->opt.timestep;
    std::vector<mjtNum> vec(nv), res_with_con(nv, 0), res_no_con(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i, 3) - 0.5;
    }
    mj_effMulAdd(m.get(), d.get(), res_with_con.data(), vec.data(),
                 /*flg_contact=*/1);
    mj_effMulAdd(m.get(), d.get(), res_no_con.data(), vec.data(),
                 /*flg_contact=*/0);

    // Isolate the contact-only spring force by comparing against a copy with
    // contact disabled, perturbed by +/- eps*vec (small enough that active
    // contacts remain penetrating with constant plane normal).
    mjtNum eps = MjEps(1e-7, 1e-5);
    MjDataPtr d_plus(mj_copyData(nullptr, m.get(), d.get()));
    MjDataPtr d_minus(mj_copyData(nullptr, m.get(), d.get()));
    mju_addToScl(d_plus->qpos, vec.data(), eps, nv);
    mju_addToScl(d_minus->qpos, vec.data(), -eps, nv);
    mj_forward(m.get(), d_plus.get());
    mj_forward(m.get(), d_minus.get());

    m->opt.disableflags |= mjDSBL_CONTACT;
    MjDataPtr d_plus_nocon(mj_copyData(nullptr, m.get(), d_plus.get()));
    MjDataPtr d_minus_nocon(mj_copyData(nullptr, m.get(), d_minus.get()));
    mj_forward(m.get(), d_plus_nocon.get());
    mj_forward(m.get(), d_minus_nocon.get());
    m->opt.disableflags &= ~mjDSBL_CONTACT;

    mjtNum max_err = 0, scale = 0;
    for (int i = 0; i < nv; i++) {
      mjtNum f_plus = d_plus->qfrc_spring[i] - d_plus_nocon->qfrc_spring[i];
      mjtNum f_minus = d_minus->qfrc_spring[i] - d_minus_nocon->qfrc_spring[i];
      mjtNum fd_h2K = -h * h * (f_plus - f_minus) / (2 * eps);
      mjtNum analytic_h2K = res_with_con[i] - res_no_con[i];
      max_err = mju_max(max_err, mju_abs(analytic_h2K - fd_h2K));
      scale = mju_max(scale, mju_abs(fd_h2K));
    }
    EXPECT_GT(scale, 1e-2);
    EXPECT_LT(max_err, MjTol(1e-6, 1e-3) * scale);

    // Also verify velocity shift d->efm_c: setting qvel = vec must shift efm_c
    // by -(1/h) * (h^2 K_total * vec) = -h * K_total * vec.
    mju_copy(d->qvel, vec.data(), nv);
    mj_forward(m.get(), d.get());
    for (int i = 0; i < nv; i++) {
      EXPECT_NEAR(
          d->efm_c[i], -res_with_con[i] / h,
          MjTol(1e-9, 1e-3) * mju_max(1.0, mju_abs(res_with_con[i] / h)))
          << "shift mismatch at DOF " << i;
    }
  }
}

struct FlexAttachmentCase {
  const char* name;
  const char* joints;
  bool sliding_child = false;
  bool shared_body = false;
};

class FlexAttachmentTest
    : public MujocoTest,
      public testing::WithParamInterface<FlexAttachmentCase> {
 protected:
  void SetUp() override { Load(); }

  std::string Xml(bool independent, const char* elasticity,
                  bool mixed = false) {
    const char* inertia =
        R"(<inertial pos="0 0 0" mass="0.1" diaginertia=".01 .02 .03"/>)";
    const char* slides = R"(<joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>)";
    const char* positions[] = {"0 0 0", ".1 0 0", "0 .1 0", ".1 .1 0"};
    std::string xml = R"(<mujoco><option integrator="discrete" solver="CG"
      gravity="0 0 0" iterations="500"/><worldbody>)";
    int count = independent ? 4 : (GetParam().shared_body ? 2 : 3);
    for (int v = 0; v < count; v++) {
      xml += "<body name='v" + std::to_string(v) + "' pos='" + positions[v] +
             "'>" + inertia + slides + "</body>";
    }
    if (!independent) {
      xml += std::string("<body name='carrier' pos='.08 .08 -.02'>") + inertia +
             GetParam().joints;
      if (GetParam().shared_body) xml += "<body name='v2' pos='-.08 .02 .02'/>";
      xml += "<body name='v3' pos='.02 .02 .02'>";
      if (GetParam().sliding_child) xml += std::string(inertia) + slides;
      xml += "</body></body>";
    }
    if (mixed) {
      xml += R"(<flexcomp name="ordinary" dim="2" type="grid" count="3 3 1"
        spacing=".1 .1 .1" pos="2 0 0" mass="1">
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="100" thickness=".02" damping=".02" elastic2d="stretch"/>
      </flexcomp>)";
    }
    xml +=
        R"(</worldbody><deformable><flex name="patch" dim="2" body="v0 v1 v2 v3"
      element="0 1 2 1 3 2"><contact contype="0" conaffinity="0" selfcollide="none"/>
      <elasticity young="100" poisson=".3" thickness=".02" damping=".02" elastic2d=")";
    return xml + elasticity + R"("/></flex></deformable></mujoco>)";
  }

  void Load(const char* elasticity = "both", bool mixed = false) {
    char error[1024];
    data.reset();
    model = LoadModelFromString(Xml(false, elasticity, mixed), error,
                                sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    data = MakeData(model);
    mj_forward(model.get(), data.get());
    flex = mj_name2id(model.get(), mjOBJ_FLEX, "patch");
    ASSERT_FALSE(mj_flexSimple(model.get(), flex));
  }

  std::vector<mjtNum> Direction(int base = 2) {
    std::vector<mjtNum> result(model->nv);
    for (int i = 0; i < model->nv; i++) result[i] = mju_Halton(i, base) - .5;
    return result;
  }

  void Deform() {
    auto direction = Direction();
    mj_integratePos(model.get(), data->qpos, direction.data(), .05);
    mju_copy(data->qvel, direction.data(), model->nv);
    mj_forward(model.get(), data.get());
  }

  // Independent oracle: public point Jacobians and force application, not the
  // flex helpers.
  std::vector<mjtNum> Gather(const std::vector<mjtNum>& vec) {
    std::vector<mjtNum> result(12), jac(3 * model->nv);
    for (int v = 0; v < 4; v++) {
      int gv = model->flex_vertadr[flex] + v;
      mj_jac(model.get(), data.get(), jac.data(), nullptr,
             data->flexvert_xpos + 3 * gv, model->flex_vertbodyid[gv]);
      mju_mulMatVec(result.data() + 3 * v, jac.data(), vec.data(), 3,
                    model->nv);
    }
    return result;
  }

  std::vector<mjtNum> Scatter(const mjtNum* force) {
    std::vector<mjtNum> result(model->nv, 0);
    for (int v = 0; v < 4; v++) {
      int gv = model->flex_vertadr[flex] + v;
      mj_applyFT(model.get(), data.get(), force + 3 * v, nullptr,
                 data->flexvert_xpos + 3 * gv, model->flex_vertbodyid[gv],
                 result.data());
    }
    return result;
  }

  void Reference(const char* elasticity) {
    char error[1024];
    reference_data.reset();
    reference =
        LoadModelFromString(Xml(true, elasticity), error, sizeof(error));
    ASSERT_THAT(reference.get(), NotNull()) << error;
    reference_data = MakeData(reference);
    auto velocity = Gather(AsVector(data->qvel, model->nv));
    mju_copy(reference_data->qvel, velocity.data(), 12);
    for (int v = 0; v < 4; v++) {
      int body = reference->flex_vertbodyid[v];
      mju_sub3(reference_data->qpos + 3 * v,
               data->flexvert_xpos + 3 * (model->flex_vertadr[flex] + v),
               reference->body_pos + 3 * body);
    }
    mj_forward(reference.get(), reference_data.get());
  }

  void ExpectForce(const char* elasticity, bool damping) {
    ASSERT_NO_FATAL_FAILURE(Load(elasticity));
    Deform();
    ASSERT_NO_FATAL_FAILURE(Reference(elasticity));
    auto expected = Scatter(damping ? reference_data->qfrc_damper
                                    : reference_data->qfrc_spring);
    const mjtNum* actual = damping ? data->qfrc_damper : data->qfrc_spring;
    EXPECT_GT(mju_norm(expected.data(), model->nv), MjTol(1e-7, 1e-4));
    EXPECT_THAT(AsVector(actual, model->nv),
                Pointwise(MjNear(1e-10, 1e-5), expected));
  }

  void ExpectOperator(bool bend) {
    Deform();
    ASSERT_NO_FATAL_FAILURE(Reference("both"));
    auto vec = Direction(3), world = Gather(vec);
    auto apply = bend ? mjd_flexBend_mul : mjd_flexStretch_mul;
    std::vector<mjtNum> world_result(12, 0), result(model->nv + 3, 0);
    apply(reference.get(), reference_data.get(), world_result.data(),
          world.data(), 1, 0);
    result[model->nv] = result[model->nv + 1] = result[model->nv + 2] = 123;
    apply(model.get(), data.get(), result.data(), vec.data(), 1, 0);
    EXPECT_THAT(AsVector(result.data(), model->nv),
                Pointwise(MjNear(1e-10, 2e-5), Scatter(world_result.data())));
    EXPECT_THAT(AsVector(result.data() + model->nv, 3), Each(123));
  }

  void ExpectMetric(bool mixed) {
    ASSERT_NO_FATAL_FAILURE(Load("both", mixed));
    Deform();
    EXPECT_EQ(data->nefmK > 0, mixed);
    auto vec = Direction(3);
    std::vector<mjtNum> expected(model->nv, 0), actual(model->nv, 0);
    mjtNum h = model->opt.timestep;
    mjd_flexBend_mul(model.get(), data.get(), expected.data(), vec.data(),
                     h * h, h);
    mjd_flexStretch_mul(model.get(), data.get(), expected.data(), vec.data(),
                        h * h, h);
    mj_effMulAdd(model.get(), data.get(), actual.data(), vec.data(), 1);
    EXPECT_THAT(actual, Pointwise(MjNear(1e-12, 1e-7), expected));
  }

  int flex;
  MjModelPtr model, reference;
  MjDataPtr data, reference_data;
};

TEST_P(FlexAttachmentTest, GatherMatchesPointJacobian) {
  Deform();
  auto vec = Direction();
  std::vector<mjtNum> actual(12);
  mj_flexGather(model.get(), data.get(), flex, actual.data(), vec.data());
  EXPECT_THAT(actual, Pointwise(MjNear(1e-12, 1e-6), Gather(vec)));
}

TEST_P(FlexAttachmentTest, ScatterPreservesVirtualWork) {
  Deform();
  auto vec = Direction(), world = Gather(vec);
  std::vector<mjtNum> force(12), actual(model->nv, 0);
  for (int i = 0; i < 12; i++) force[i] = mju_Halton(i, 5) - .5;
  mj_flexScatter(model.get(), data.get(), flex, actual.data(), force.data(), 1);
  EXPECT_THAT(actual, Pointwise(MjNear(1e-12, 1e-6), Scatter(force.data())));
  EXPECT_THAT(mju_dot(vec.data(), actual.data(), model->nv),
              MjNear(mju_dot(world.data(), force.data(), 12), 1e-12, 1e-6));
}

TEST_P(FlexAttachmentTest, BendingSpringIncludesAttachmentReaction) {
  ExpectForce("bend", false);
}
TEST_P(FlexAttachmentTest, BendingDampingIncludesAttachmentReaction) {
  ExpectForce("bend", true);
}
TEST_P(FlexAttachmentTest, StretchSpringIncludesAttachmentReaction) {
  ExpectForce("stretch", false);
}
TEST_P(FlexAttachmentTest, StretchDampingIncludesAttachmentReaction) {
  ExpectForce("stretch", true);
}
TEST_P(FlexAttachmentTest, BendingOperatorMatchesWorldPatch) {
  ExpectOperator(true);
}
TEST_P(FlexAttachmentTest, StretchOperatorMatchesWorldPatch) {
  ExpectOperator(false);
}
TEST_P(FlexAttachmentTest, MatrixFreeMetric) { ExpectMetric(false); }
TEST_P(FlexAttachmentTest, MixedAssembledAndMatrixFreeMetric) {
  ExpectMetric(true);
}

TEST_P(FlexAttachmentTest, BendingDampingDerivative) {
  ASSERT_NO_FATAL_FAILURE(Load("bend"));
  Deform();
  auto vec = Direction(3);
  std::vector<mjtNum> expected(model->nv, 0);
  mjd_flexBend_mul(model.get(), data.get(), expected.data(), vec.data(), 0, 1);
  auto initial = AsVector(data->qfrc_damper, model->nv);
  mjtNum eps = MjEps(1e-6, 1e-3);
  mju_addToScl(data->qvel, vec.data(), eps, model->nv);
  mj_forward(model.get(), data.get());
  for (int i = 0; i < model->nv; i++) {
    EXPECT_THAT(-(data->qfrc_damper[i] - initial[i]) / eps,
                MjNear(expected[i], 1e-9, 1e-5));
  }
}

TEST_P(FlexAttachmentTest, EffectiveSolveResidual) {
  ASSERT_NO_FATAL_FAILURE(Load("both", true));
  Deform();
  auto rhs = Direction(3);
  std::vector<mjtNum> solution(model->nv), product(model->nv);
  mj_effSolve(model.get(), data.get(), solution.data(), rhs.data());
  mj_mulM(model.get(), data.get(), product.data(), solution.data());
  mj_effMulAdd(model.get(), data.get(), product.data(), solution.data(), 1);
  EXPECT_THAT(product, Pointwise(MjNear(2e-8, 2e-5), rhs));
}

TEST_P(FlexAttachmentTest, MetricIsSymmetricAndPositive) {
  Deform();
  auto u = Direction(3), v = Direction(5);
  std::vector<mjtNum> Ku(model->nv, 0), Kv(model->nv, 0);
  mj_effMulAdd(model.get(), data.get(), Ku.data(), u.data(), 1);
  mj_effMulAdd(model.get(), data.get(), Kv.data(), v.data(), 1);
  EXPECT_THAT(mju_dot(u.data(), Kv.data(), model->nv),
              MjNear(mju_dot(v.data(), Ku.data(), model->nv), 1e-12, 1e-7));
  EXPECT_GE(mju_dot(u.data(), Ku.data(), model->nv), -MjTol(1e-12, 1e-7));
}

TEST_P(FlexAttachmentTest, AttachmentCannotUseFreeGyroShortcut) {
  int body = mj_name2id(model.get(), mjOBJ_BODY, "carrier");
  EXPECT_FALSE(
      mjd_freeGyroPossible(model.get(), data.get(), model->body_jntadr[body]));
}

TEST_P(FlexAttachmentTest, GeneralAssembleMatchesMatrixFree) {
  for (bool mixed : {false, true}) {
    ASSERT_NO_FATAL_FAILURE(Load("both", mixed));
    Deform();
    int nv = model->nv;
    mjtNum s1 = 4e-4, s2 = 2e-2;

    for (auto [flg_bend, flg_stretch] :
         {std::pair<int, int>{1, 0}, {0, 1}, {1, 1}}) {
      std::vector<int> rownnz(nv), rowadr(nv);
      int nnz = mjd_flexStiff_assembleGen(
          model.get(), data.get(), rownnz.data(), rowadr.data(), nullptr,
          nullptr, s1, s2, flg_bend, flg_stretch, nullptr, /*flg_general=*/1);
      ASSERT_GT(nnz, 0);
      EXPECT_EQ(rowadr[nv - 1] + rownnz[nv - 1], nnz);

      std::vector<int> colind(nnz);
      std::vector<mjtNum> val(nnz);
      int nnz2 = mjd_flexStiff_assembleGen(
          model.get(), data.get(), rownnz.data(), rowadr.data(), colind.data(),
          val.data(), s1, s2, flg_bend, flg_stretch, nullptr,
          /*flg_general=*/1);
      EXPECT_EQ(nnz2, nnz);

      // verify strictly sorted columns and symmetry
      std::vector<mjtNum> dense(nv * nv, 0);
      for (int r = 0; r < nv; r++) {
        for (int k = 0; k < rownnz[r]; k++) {
          int c = colind[rowadr[r] + k];
          ASSERT_GE(c, 0);
          ASSERT_LT(c, nv);
          if (k > 0) {
            EXPECT_GT(c, colind[rowadr[r] + k - 1]);
          }
          dense[r * nv + c] = val[rowadr[r] + k];
        }
      }
      for (int r = 0; r < nv; r++) {
        for (int c = r + 1; c < nv; c++) {
          EXPECT_THAT(dense[r * nv + c],
                      MjNear(dense[c * nv + r], 1e-12, 1e-6));
        }
      }

      // verify CSR * vec matches matrix-free operators
      auto vec = Direction(3);
      std::vector<mjtNum> expected(nv, 0), actual(nv, 0);
      if (flg_bend) {
        mjd_flexBend_mul(model.get(), data.get(), expected.data(), vec.data(),
                         s1, s2);
      }
      if (flg_stretch) {
        mjd_flexStretch_mul(model.get(), data.get(), expected.data(),
                            vec.data(), s1, s2);
      }
      for (int r = 0; r < nv; r++) {
        mjtNum sum = 0;
        for (int k = 0; k < rownnz[r]; k++) {
          sum += val[rowadr[r] + k] * vec[colind[rowadr[r] + k]];
        }
        actual[r] = sum;
      }
      EXPECT_THAT(actual, Pointwise(MjNear(1e-12, 1e-6), expected));
    }
  }
}

TEST_P(FlexAttachmentTest, SupportsNewtonForDiscreteIntegrator) {
  for (bool mixed : {false, true}) {
    std::string xml = Xml(false, "both", mixed);
    const std::string solver_str = "solver=\"CG\"";
    xml.replace(xml.find(solver_str), solver_str.size(),
                "solver=\"CG\" tolerance=\"1e-14\"");
    const std::string wb_str = "<worldbody>";
    xml.replace(xml.find(wb_str), wb_str.size(),
                "<worldbody><geom type='plane' pos='0 0 -.05' size='2 2 .1'/>");
    const std::string nocontact = "contype=\"0\" conaffinity=\"0\" ";
    for (size_t pos = 0;
         (pos = xml.find(nocontact, pos)) != std::string::npos;) {
      xml.erase(pos, nocontact.size());
    }

    char error[1024];
    auto m_test = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m_test.get(), NotNull()) << error;
    auto d_test = MakeData(m_test);
    int nv = m_test->nv;

    std::vector<mjtNum> dir(nv);
    for (int i = 0; i < nv; i++) dir[i] = mju_Halton(i, 2) - .5;
    mj_integratePos(m_test.get(), d_test->qpos, dir.data(), .05);
    mju_copy(d_test->qvel, dir.data(), nv);

    m_test->opt.solver = mjSOL_CG;
    mj_forward(m_test.get(), d_test.get());
    std::vector<mjtNum> qacc_cg(d_test->qacc, d_test->qacc + nv);

    m_test->opt.solver = mjSOL_NEWTON;
    mj_forward(m_test.get(), d_test.get());
    EXPECT_GT(d_test->nefmK, 0);
    EXPECT_EQ(d_test->efm_active, 2);

    // the carrier's joint dofs have rows: the articulated pull-back is in the
    // CSR
    int carrier = mj_name2id(m_test.get(), mjOBJ_BODY, "carrier");
    EXPECT_GT(d_test->efm_K_rownnz[m_test->body_dofadr[carrier]], 0);

    std::vector<mjtNum> rhs(nv), sol(nv), prod(nv);
    for (int i = 0; i < nv; i++) rhs[i] = mju_Halton(i, 3) - .5;
    mj_effSolve(m_test.get(), d_test.get(), sol.data(), rhs.data());
    mj_mulM(m_test.get(), d_test.get(), prod.data(), sol.data());
    mj_effMulAdd(m_test.get(), d_test.get(), prod.data(), sol.data(), 1);
    EXPECT_THAT(prod, Pointwise(MjNear(2e-8, 2e-5), rhs));

    mjtNum scale = mju_norm(qacc_cg.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    EXPECT_THAT(AsVector(d_test->qacc, nv),
                Pointwise(MjNear(1e-12 * scale, 1e-4 * scale), qacc_cg));
  }
}

TEST_F(DerivativeTest, GeneralFlexStiffAssembleAllDimsAndInterp) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <worldbody>
      <body name="arm1" pos="0 0 1">
        <joint type="ball"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.02 0.03"/>
        <flexcomp name="rod" type="grid" count="5 1 1" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="1" mass="0.4">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e5" damping="0.1"/>
        </flexcomp>
        <flexcomp name="tri" type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0.4 0" radius="0.01" dim="3" mass="0.5"
                  dof="trilinear" cellcount="2 1 1">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.25" damping="0.5"/>
        </flexcomp>
      </body>
      <body name="arm2" pos="1 0 1">
        <joint type="hinge" axis="0 1 0"/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.02 0.03"/>
        <flexcomp name="solid" type="grid" count="2 2 2" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="3" mass="0.5" dof="full">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.2" damping="0.2"/>
        </flexcomp>
        <flexcomp name="quad" type="grid" count="3 3 3" spacing="0.1 0.1 0.1"
                  pos="0 0.4 0" radius="0.01" dim="3" mass="0.5"
                  dof="quadratic" cellcount="1 1 1">
          <pin id="0"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.2" damping="0.3"/>
        </flexcomp>
      </body>
      <body name="carrier" pos="2 0 1">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.5" diaginertia="0.01 0.02 0.03"/>
        <flexcomp name="shell" type="grid" count="4 4 4" spacing="0.1 0.1 0.1"
                  radius="0.01" dim="3" mass="0.5" dof="trilinear"
                  cellcount="1 2 2">
          <pin id="0 2"/>
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e4" poisson="0.25" damping="0.4"
                      elastic2d="stretch" thickness="0.01"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  EXPECT_EQ(mjd_flexInterpAssemblable(model.get()), 0);
  int nv = model->nv;
  MjDataPtr data = MakeData(model);

  std::vector<mjtNum> dir(nv);
  for (int i = 0; i < nv; i++) dir[i] = mju_Halton(i, 2) - 0.5;
  mj_integratePos(model.get(), data->qpos, dir.data(), 0.03);
  mju_copy(data->qvel, dir.data(), nv);
  mj_forward(model.get(), data.get());

  std::vector<mjtNum> krot(model->nflexstiffness, 0);
  mjd_flexInterp_cacheKrot(model.get(), data.get(), krot.data());

  mjtNum s1 = 4e-6, s2 = 2e-3;
  std::vector<int> rownnz(nv), rowadr(nv);
  int nnz = mjd_flexStiff_assembleGen(
      model.get(), data.get(), rownnz.data(), rowadr.data(), nullptr, nullptr,
      s1, s2, /*flg_bend=*/1, /*flg_stretch=*/1, krot.data(),
      /*flg_general=*/1);
  ASSERT_GT(nnz, 0);

  std::vector<int> colind(nnz);
  std::vector<mjtNum> val(nnz);
  int nnz2 = mjd_flexStiff_assembleGen(
      model.get(), data.get(), rownnz.data(), rowadr.data(), colind.data(),
      val.data(), s1, s2, /*flg_bend=*/1, /*flg_stretch=*/1, krot.data(),
      /*flg_general=*/1);
  EXPECT_EQ(nnz2, nnz);

  for (int r = 0; r < nv; r++) {
    for (int k = 1; k < rownnz[r]; k++) {
      EXPECT_GT(colind[rowadr[r] + k], colind[rowadr[r] + k - 1]);
    }
  }

  for (int trial = 0; trial < 3; trial++) {
    std::vector<mjtNum> vec(nv), res_op(nv, 0), res_csr(nv, 0);
    for (int i = 0; i < nv; i++) {
      vec[i] = mju_Halton(i + trial * nv, 3) - 0.5;
    }
    mjd_flexBend_mul(model.get(), data.get(), res_op.data(), vec.data(), s1,
                     s2);
    mjd_flexStretch_mul(model.get(), data.get(), res_op.data(), vec.data(), s1,
                        s2);
    mjd_flexInterp_mul(model.get(), data.get(), res_op.data(), vec.data(), -s1,
                       -s2, krot.data());

    for (int r = 0; r < nv; r++) {
      mjtNum sum = 0;
      for (int k = 0; k < rownnz[r]; k++) {
        sum += val[rowadr[r] + k] * vec[colind[rowadr[r] + k]];
      }
      res_csr[r] = sum;
    }
    mjtNum scale = mju_norm(res_op.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    EXPECT_THAT(res_csr,
                Pointwise(MjNear(1e-12 * scale, 5e-5 * scale), res_op));
    EXPECT_GE(mju_dot(vec.data(), res_csr.data(), nv), -1e-11 * scale);
  }
}

// an interpolated node on a jointless child of a moving body moves with that
// body: it is not a fixed node, so the stiffness must reach the body's dofs
// under both solvers
TEST_F(DerivativeTest, InterpNodeOnWeldedChild) {
  static const char* const kXml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG" tolerance="1e-14" iterations="2000"/>
    <worldbody>
      <body name="arm">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".02" pos="-.1 0 0" mass=".3"/>
        <body name="welded"/>
      </body>
      <flexcomp name="pad" type="grid" count="3 3 3" spacing=".05 .05 .05"
                pos=".1 0 0" radius=".005" dim="3" mass=".5" dof="trilinear"
                cellcount="1 1 1">
        <pin id="0"/>
        <contact contype="0" conaffinity="0" selfcollide="none"/>
        <elasticity young="1e4" poisson=".2" damping=".1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  // move the pinned node from the world to the welded child, whose frame is the
  // world frame at qpos0, so its local coordinates are unchanged
  mjSpec* spec = mj_parseXMLString(kXml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjsFlex* flex = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "pad"));
  ASSERT_THAT(flex, NotNull());
  ASSERT_EQ((*flex->nodebody)[0], "world");
  (*flex->nodebody)[0] = "welded";
  MjModelPtr model(mj_compile(spec, nullptr));
  mj_deleteSpec(spec);
  ASSERT_THAT(model.get(), NotNull());
  int welded = mj_name2id(model.get(), mjOBJ_BODY, "welded");
  ASSERT_EQ(model->flex_nodebodyid[model->flex_nodeadr[0]], welded);
  EXPECT_EQ(mjd_flexInterpAssemblable(model.get()), 0);

  int nv = model->nv;
  int hinge = model->jnt_dofadr[mj_name2id(model.get(), mjOBJ_JOINT, "hinge")];
  MjDataPtr data = MakeData(model);
  std::vector<mjtNum> dir(nv);
  for (int i = 0; i < nv; i++) dir[i] = mju_Halton(i, 2) - .5;
  mj_integratePos(model.get(), data->qpos, dir.data(), .02);
  mju_copy(data->qvel, dir.data(), nv);

  // reference: the matrix-free interp operator, which follows the node's body
  // chain
  mj_forward(model.get(), data.get());
  std::vector<mjtNum> vec(nv), expected(nv, 0);
  for (int i = 0; i < nv; i++) vec[i] = mju_Halton(i, 3) - .5;
  mjtNum h = model->opt.timestep;
  mjd_flexInterp_mul(model.get(), data.get(), expected.data(), vec.data(),
                     -h * h, -h, data->flexelem_krot);
  EXPECT_NE(expected[hinge], 0);

  // CG applies the operator matrix-free, Newton assembles the general CSR
  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    model->opt.solver = solver;
    mj_forward(model.get(), data.get());
    std::vector<mjtNum> actual(nv, 0);
    mj_effMulAdd(model.get(), data.get(), actual.data(), vec.data(), 1);
    mjtNum scale = mju_norm(expected.data(), nv) / mju_sqrt((mjtNum)nv) + 1;
    EXPECT_THAT(actual,
                Pointwise(MjNear(1e-12 * scale, 1e-5 * scale), expected))
        << "solver " << solver;
    if (solver == mjSOL_NEWTON) {
      EXPECT_EQ(data->efm_active, 2);
      EXPECT_GT(data->efm_K_rownnz[hinge], 0);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Attachments, FlexAttachmentTest,
    testing::Values(
        FlexAttachmentCase{"Hinge", "<joint axis='0 1 0'/>"},
        FlexAttachmentCase{"Ball", "<joint type='ball'/>"},
        FlexAttachmentCase{"Free", "<freejoint/>"},
        FlexAttachmentCase{"ReorderedSlides", R"(
                      <joint type="slide" axis="0 1 0"/>
                      <joint type="slide" axis="1 0 0"/>
                      <joint type="slide" axis="0 0 -1"/>)"},
        FlexAttachmentCase{"MovingAncestor", "<joint type='ball'/>", true},
        FlexAttachmentCase{"SharedBody", "<joint type='ball'/>", false, true}),
    [](const testing::TestParamInfo<FlexAttachmentCase>& info) {
      return info.param.name;
    });

TEST_F(DerivativeTest, EffSolve) {
  // relative residual of (M+K)x - b after mj_effSolve
  auto solve_residual = [](const mjModel* m, mjData* d) {
    int nv = m->nv;
    std::vector<mjtNum> b(nv), x(nv), r(nv);
    for (int i = 0; i < nv; i++) {
      b[i] = mju_Halton(i, 3) - 0.5;
    }
    mj_effSolve(m, d, x.data(), b.data());
    mju_mulSymVecSparse(r.data(), d->M, x.data(), nv, m->M_rownnz, m->M_rowadr,
                        m->M_colind);
    mj_effMulAdd(m, d, r.data(), x.data(), /*flg_contact=*/1);
    mju_subFrom(r.data(), b.data(), nv);
    return mju_norm(r.data(), nv) / mju_norm(b.data(), nv);
  };

  // stretch + bending cloth on world: per-step factor, exact
  static const char* const kXmlBoth = R"(
  <mujoco>
    <option solver="CG" integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.05 0.05 0.05"
                radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXmlBoth, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_GE(data->efm_active, 1);
  EXPECT_GT(data->nefmK, 0);
  EXPECT_GT(data->nefmdof, 0);
  EXPECT_LT(solve_residual(model.get(), data.get()), MjTol(1e-8, 1e-4));

  // bending-only cloth: no CSR or per-step factor, constant factor covers,
  // exact
  static const char* const kXmlBend = R"(
  <mujoco>
    <option solver="CG" integrator="discrete"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.05 0.05 0.05"
                radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
        <contact selfcollide="none" contype="0" conaffinity="0"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="bend" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(kXmlBend, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_GE(data->efm_active, 1);
  EXPECT_EQ(data->nefmK, 0);
  EXPECT_EQ(data->nefmdof, 0);
  EXPECT_GT(model->nefm0dof, 0);
  EXPECT_LT(solve_residual(model.get(), data.get()), MjTol(1e-8, 1e-4));

  // A cloth under a jointed parent uses the matrix-free attachment mapping;
  // the iterative solve must still meet its tolerance.
  static const char* const kXmlMoving = R"(
  <mujoco>
    <option solver="CG" integrator="discrete"/>
    <worldbody>
      <body name="base" pos="0 0 1">
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
        <flexcomp name="cloth" type="grid" count="6 6 1" spacing="0.05 0.05 0.05"
                  radius=".005" dim="2" mass="0.5" pos="0 0 0" dof="full">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="stretch" thickness="0.01"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(kXmlMoving, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_GE(data->efm_active, 1);
  EXPECT_LT(solve_residual(model.get(), data.get()), 1e-4);
}

// A cloth with per-step stretch stiffness, used by the two tests below.
// Independent flex slides with an articulated descendant: the mass matrix
// couples the covered vertex triple to an uncovered hinge in the same kinematic
// tree.
static const char* const kStretchCloth = R"(
<mujoco>
  <option solver="CG" integrator="discrete"/>
  <default>
    <geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/>
  </default>
  <worldbody>
    <body name="v0">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom/>
      <body pos=".02 0 0">
        <joint axis="0 1 0"/>
        <geom pos=".1 0 0"/>
      </body>
    </body>
    <body name="v1" pos=".1 0 0">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom/>
    </body>
    <body name="v2" pos="0 .1 0">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom/>
    </body>
    <body name="v3" pos=".1 .1 0">
      <joint type="slide" axis="1 0 0"/>
      <joint type="slide" axis="0 1 0"/>
      <joint type="slide" axis="0 0 1"/>
      <geom/>
    </body>
  </worldbody>
  <deformable>
    <flex dim="2" body="v0 v1 v2 v3" element="0 1 2 1 3 2">
      <contact selfcollide="none" contype="0" conaffinity="0"/>
      <elasticity young="1e3" poisson=".2" damping=".1" elastic2d="stretch" thickness=".01"/>
    </flex>
  </deformable>
</mujoco>
)";

// A metric too ill-conditioned for the 3x3 blocks exhausts mj_effSolve's
// iteration budget; it must report that rather than return an under-converged
// qacc_smooth silently. Forced by conditioning rather than by an unreachable
// opt.tolerance, which cannot be expressed in single precision: there the
// residual reaches exactly zero and CG breaks down on a converged solve.
TEST_F(DerivativeTest, EffSolveCapWarns) {
  static const char* const kStiffCloth = R"(
  <mujoco>
    <option solver="CG" integrator="discrete"/>
    <worldbody>
      <body name="base" pos="0 0 1">
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
        <flexcomp name="cloth" type="grid" count="24 24 1" dim="2"
                  spacing="0.02 0.02 0.02" radius=".002" mass="0.02" dof="full">
          <contact selfcollide="none" contype="0" conaffinity="0"/>
          <elasticity young="1e9" poisson="0.45" damping="0" elastic2d="stretch"
                      thickness="0.02"/>
          <pin id="0"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kStiffCloth, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  // Excite in-plane stretch modes; gravity alone translates this flat cloth.
  for (int i = 0; i < model->nv; i++) {
    data->qfrc_applied[i] = mju_Halton(i, 2) - 0.5;
  }

  MockWarningHandler warning_handler;
  // both: the specific cause, then the mjWARN_INERTIA it is reported through
  warning_handler.ExpectWarnings("Flex stiffness is too ill-conditioned");
  warning_handler.ExpectWarnings("Inertia matrix is too close to singular");
  mj_forward(model.get(), data.get());
  testing::Mock::VerifyAndClearExpectations(&warning_handler);
}

// PCG requires a symmetric preconditioner. mj_effPrec must satisfy
// u.P(v) == v.P(u); it did not when the covered and uncovered dofs shared a
// kinematic tree, which is what the articulated vertex body here exercises.
TEST_F(DerivativeTest, EffPrecIsSymmetric) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kStretchCloth, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_GE(data->efm_active, 1);
  ASSERT_EQ(data->nefmdof, 4);
  ASSERT_EQ(model->nv, 13);

  int nv = model->nv;
  std::vector<mjtNum> u(nv), v(nv), Pu(nv), Pv(nv);
  for (int trial = 0; trial < 5; trial++) {
    for (int i = 0; i < nv; i++) {
      u[i] = mju_Halton(i + trial * nv, 2) - 0.5;
      v[i] = mju_Halton(i + trial * nv, 5) - 0.5;
    }
    mj_effPrec(model.get(), data.get(), Pu.data(), u.data());
    mj_effPrec(model.get(), data.get(), Pv.data(), v.data());
    mjtNum a = mju_dot(v.data(), Pu.data(), nv);
    mjtNum b = mju_dot(u.data(), Pv.data(), nv);
    EXPECT_THAT(a, MjNear(b, 1e-10, 1e-4))
        << "preconditioner is not symmetric, trial " << trial;
  }
}

// mj_effPrecFold adds the 3x3 diagonal blocks of J'*D*J to the step's blocks of
// M + K: with no rows it reproduces them, and with every row A1 = A0 + Delta,
// checked through the factors as A1^-1 (A0 y + Delta y) = y
TEST_F(DerivativeTest, EffPrecFoldAddsRowBlocks) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option solver="CG" integrator="discrete" jacobian="dense">
      <flag island="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .01"/>
      <flexcomp name="sheet" type="grid" dim="2" count="5 5 1" spacing=".04 .04 1"
                radius=".005" mass=".1" pos="0 0 .004">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson=".3" thickness="2e-3" elastic2d="stretch"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  const mjModel* m = model.get();
  mjData* d = data.get();
  mj_forward(m, d);
  int nb = d->nefmdof, nv = m->nv, nefc = d->nefc;
  ASSERT_GT(nb, 0);
  ASSERT_GT(nefc, 0);

  // no rows: the step's own blocks
  vector<mjtNum> L0(9 * nb), L1(9 * nb);
  ASSERT_TRUE(mj_effPrecFold(m, d, L0.data(), -1, 0, d->efc_D, 0, d->efc_J,
                             nullptr, nullptr, nullptr));
  for (int i = 0; i < 9 * nb; i++) {
    EXPECT_EQ(L0[i], d->efm_L[i]) << i;
  }

  // every row
  ASSERT_TRUE(mj_effPrecFold(m, d, L1.data(), -1, nefc, d->efc_D, 0, d->efc_J,
                             nullptr, nullptr, nullptr));
  int folded = 0;
  for (int k = 0; k < nb; k++) {
    int base = d->efm_dofid[k];
    mjtNum delta[9] = {0};
    for (int r = 0; r < nefc; r++) {
      const mjtNum* Jk = d->efc_J + r * nv + base;
      for (int a = 0; a < 3; a++) {
        for (int b = 0; b < 3; b++) {
          delta[3 * a + b] += d->efc_D[r] * Jk[a] * Jk[b];
        }
      }
    }
    folded += mju_norm(delta, 9) > 0;
    mjtNum v[3] = {static_cast<mjtNum>(mju_Halton(k, 2) - .5),
                   static_cast<mjtNum>(mju_Halton(k, 3) - .5),
                   static_cast<mjtNum>(mju_Halton(k, 5) - .5)};
    mjtNum y[3], dy[3], rhs[3], x[3];
    mju_cholSolve(y, L0.data() + 9 * k, v, 3);
    mju_mulMatVec3(dy, delta, y);
    mju_add3(rhs, v, dy);
    mju_cholSolve(x, L1.data() + 9 * k, rhs, 3);
    for (int a = 0; a < 3; a++) {
      EXPECT_THAT(x[a], MjNear(y[a], 1e-8, 1e-3)) << "block " << k;
    }
  }
  EXPECT_GT(folded, 0) << "the sheet must touch the plane";
}

// Per-island mj_effPrecFold writes compact blocks matching the island slice of
// monolithic mj_effPrecFold (0 blocks on rigid islands), and island vs
// monolithic CG agree for both 3D flexes (effIslandStep) and 2D flexes + sleep
// (effCholReserved fallback).
TEST_F(DerivativeTest, EffPrecFoldMultiIsland) {
  static constexpr char kXml3D[] = R"(
  <mujoco>
    <option solver="CG" integrator="discrete" tolerance="1e-12" iterations="300"/>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <flexcomp name="cube0" type="grid" dim="3" count="2 2 2" spacing=".05 .05 .05"
                radius=".008" mass=".2" pos="-.3 0 .025">
        <contact selfcollide="none"/>
        <elasticity young="2e4" poisson=".3" damping=".01"/>
      </flexcomp>
      <flexcomp name="cube1" type="grid" dim="3" count="2 2 2" spacing=".05 .05 .05"
                radius=".008" mass=".3" pos=".3 0 .025">
        <contact selfcollide="none"/>
        <elasticity young="5e4" poisson=".25" damping=".01"/>
      </flexcomp>
      <body name="box" pos="0 .4 .04">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass=".4"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(kXml3D, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  for (mjtJacobian jac : {mjJAC_SPARSE, mjJAC_DENSE}) {
    model->opt.jacobian = jac;
    model->opt.disableflags &= ~mjDSBL_ISLAND;
    MjDataPtr data = MakeData(model);
    mjModel* m = model.get();
    mjData* d = data.get();
    mj_forward(m, d);

    int is_sparse = mj_isSparse(m);
    ASSERT_EQ(d->nisland, 3);
    ASSERT_EQ(d->nefmdof, 16);

    vector<mjtNum> L_mono(9 * d->nefmdof);
    ASSERT_EQ(mj_effPrecFold(m, d, L_mono.data(), -1, d->nefc, d->efc_D,
                             is_sparse, d->efc_J, d->efc_J_rownnz,
                             d->efc_J_rowadr, d->efc_J_colind),
              d->nefmdof);

    int flex_islands = 0, rigid_islands = 0;
    for (int isl = 0; isl < d->nisland; isl++) {
      int inv = d->island_nv[isl];
      int inefc = d->island_nefc[isl];
      int idofadr = d->island_idofadr[isl];
      int iefcadr = d->island_iefcadr[isl];

      vector<int> isl_triples;
      for (int k = 0; k < d->nefmdof; k++) {
        if (d->dof_island[d->efm_dofid[k]] == isl) {
          isl_triples.push_back(k);
        }
      }
      int expected = static_cast<int>(isl_triples.size());
      EXPECT_EQ(mj_effFoldBlocks(d, isl), expected);
      if (expected == 0) {
        rigid_islands++;
      } else {
        EXPECT_EQ(expected, 8);
        flex_islands++;
      }

      vector<mjtNum> L_isl(std::max(1, 9 * expected), -123);
      int nfold = 0;
      if (is_sparse) {
        int nJ = 0;
        for (int r = 0; r < inefc; r++) {
          nJ += d->efc_J_rownnz[d->map_iefc2efc[iefcadr + r]];
        }
        vector<mjtNum> iJ(std::max(1, nJ));
        vector<int> iJ_rownnz(inefc), iJ_rowadr(inefc),
            iJ_colind(std::max(1, nJ));
        mju_blockSparse(iJ.data(), iJ_rownnz.data(), iJ_rowadr.data(),
                        iJ_colind.data(), d->efc_J, d->efc_J_rownnz,
                        d->efc_J_rowadr, d->efc_J_colind, inefc,
                        d->map_iefc2efc + iefcadr, d->map_dof2idof, idofadr, 0,
                        nullptr, nullptr);
        nfold = mj_effPrecFold(
            m, d, L_isl.data(), isl, inefc, d->iefc_D + iefcadr, 1, iJ.data(),
            iJ_rownnz.data(), iJ_rowadr.data(), iJ_colind.data());
      } else {
        vector<mjtNum> iJ(std::max(1, inefc * inv));
        mju_block(iJ.data(), d->efc_J, m->nv, inv, inefc,
                  d->map_iefc2efc + iefcadr, d->map_idof2dof + idofadr);
        nfold =
            mj_effPrecFold(m, d, L_isl.data(), isl, inefc, d->iefc_D + iefcadr,
                           0, iJ.data(), nullptr, nullptr, nullptr);
      }
      ASSERT_EQ(nfold, expected);
      for (int p = 0; p < nfold; p++) {
        int k = isl_triples[p];
        for (int a = 0; a < 9; a++) {
          EXPECT_THAT(L_isl[9 * p + a], MjNear(L_mono[9 * k + a], 1e-12, 1e-5))
              << "jac=" << jac << " isl=" << isl << " p=" << p << " a=" << a;
        }
      }
    }
    EXPECT_EQ(flex_islands, 2);
    EXPECT_EQ(rigid_islands, 1);

    model->opt.disableflags |= mjDSBL_ISLAND;
    MjDataPtr data_mono = MakeData(model);
    mj_forward(m, data_mono.get());
    for (int i = 0; i < m->nv; i++) {
      EXPECT_THAT(d->qacc[i], MjNear(data_mono->qacc[i], 1e-4, 1e-2))
          << "jac=" << jac << " dof=" << i;
    }
  }

  // 2D flexes with sleep="enable": effCholReserved holds so per-island CG
  // exercises the !has_isl fallback in mj_effPrecFold and mj_effPrecIsland.
  static constexpr char kXml2DSleep[] = R"(
  <mujoco>
    <option solver="CG" integrator="discrete" tolerance="1e-12" iterations="300">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <flexcomp name="s0" type="grid" dim="2" count="3 3 1" spacing=".05 .05 1"
                radius=".005" mass=".1" pos="-.3 0 .004">
        <contact selfcollide="none"/>
        <elasticity young="1e4" poisson=".3" thickness="2e-3" elastic2d="stretch"/>
      </flexcomp>
      <flexcomp name="s1" type="grid" dim="2" count="3 3 1" spacing=".05 .05 1"
                radius=".005" mass=".15" pos=".3 0 .004">
        <contact selfcollide="none"/>
        <elasticity young="2e4" poisson=".3" thickness="2e-3" elastic2d="stretch"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model2d = LoadModelFromString(kXml2DSleep, error, sizeof(error));
  ASSERT_THAT(model2d.get(), NotNull()) << error;
  MjDataPtr data2d_isl = MakeData(model2d);
  mj_forward(model2d.get(), data2d_isl.get());
  ASSERT_EQ(data2d_isl->nisland, 2);

  model2d->opt.enableflags &= ~mjENBL_SLEEP;
  model2d->opt.disableflags |= mjDSBL_ISLAND;
  MjDataPtr data2d_mono = MakeData(model2d);
  mj_forward(model2d.get(), data2d_mono.get());
  for (int i = 0; i < model2d->nv; i++) {
    EXPECT_THAT(data2d_isl->qacc[i], MjNear(data2d_mono->qacc[i], 1e-4, 1e-2))
        << "2d sleep dof=" << i;
  }
}

// Validate that mjd_actuator_vel (d->qDeriv) and mj_effActuation (d->nefmA)
// skip tendon actuators whose total tendon force saturates tendon_actfrcrange,
// including when proportional scaling across multiple actuators rounds.
TEST_F(DerivativeTest, TendonActfrcrangeSaturation) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="discrete"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.05" mass="1"/>
        <site name="s1"/>
      </body>
      <site name="s0" pos="0 0 0.3"/>
    </worldbody>
    <tendon>
      <spatial name="ten" actuatorfrclimited="true" actuatorfrcrange="-19 0">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
    <actuator>
      <general tendon="ten" biastype="affine" gainprm="3" biasprm="0 -3 -3"/>
      <general tendon="ten" biastype="affine" gainprm="5" biasprm="0 -5 -5"/>
      <general tendon="ten" biastype="affine" gainprm="7" biasprm="0 -7 -7"/>
      <general tendon="ten" biastype="affine" gainprm="11" biasprm="0 -11 -11"/>
      <general tendon="ten" biastype="affine" gainprm="13" biasprm="0 -13 -13"/>
      <general tendon="ten" biastype="affine" gainprm="17" biasprm="0 -17 -17"/>
    </actuator>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), testing::NotNull()) << error;
  MjDataPtr d = MakeData(m);

  auto eval_with_ctrl = [&](mjtNum ctrl) {
    for (int i = 0; i < m->nu; ++i) {
      d->ctrl[i] = ctrl;
    }
    mj_forward(m.get(), d.get());
    mju_zero(d->qDeriv, m->nD);
    mjd_actuator_vel(m.get(), d.get());
    mjtNum sum_force = 0;
    for (int i = 0; i < m->nu; ++i) {
      sum_force += d->actuator_force[i];
    }
    return sum_force;
  };

  // 1. Unsaturated (-11.2 N in (-19, 0)): both qDeriv and nefmA are populated
  EXPECT_NEAR(eval_with_ctrl(0.1), -11.2, MjTol(1e-10, 1e-4));
  EXPECT_LT(d->qDeriv[0], -1.0);
  EXPECT_EQ(d->nefmA, m->nu);

  // 2. Saturated at lower bound (-19 N): qDeriv and nefmA are both skipped
  EXPECT_NEAR(eval_with_ctrl(-1.4), -19.0, MjTol(1e-10, 1e-4));
  EXPECT_EQ(d->qDeriv[0], 0.0);
  EXPECT_EQ(d->nefmA, 0);

  // 3. Saturated at upper bound (0 N): qDeriv and nefmA are both skipped
  EXPECT_EQ(eval_with_ctrl(1.0), 0.0);
  EXPECT_EQ(d->qDeriv[0], 0.0);
  EXPECT_EQ(d->nefmA, 0);
}

}  // namespace
}  // namespace mujoco
