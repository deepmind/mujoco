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

// Tests for engine/engine_inverse.c.

#include "src/engine/engine_inverse.h"

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using InverseTest = MujocoTest;

const int kSteps = 70;
static const char* const kModelPath = "testdata/model.xml";

// test standard continuous-time inverse dynamics
TEST_F(InverseTest, ForwardInverseMatch) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  // set small tolerance and enough iterations for all solvers to converge
  model->opt.iterations = 500;
  model->opt.tolerance = 0;

  // solver names for diagnostics
  const char* solver_name[] = {"PGS", "CG", "Newton"};

  for (int diagexact = 0; diagexact < 2; diagexact++) {
    if (diagexact) {
      model->opt.enableflags |= mjENBL_DIAGEXACT;
    } else {
      model->opt.enableflags &= ~mjENBL_DIAGEXACT;
    }

    for (mjtSolver solver : {mjSOL_PGS, mjSOL_CG, mjSOL_NEWTON}) {
      model->opt.solver = solver;
      mj_resetData(model, data);

      // simulate, call mj_forward
      for (int i = 0; i < kSteps; ++i) {
        mj_step(model, data);
      }
      mj_forward(model, data);

      // call built-in testing function
      mj_compareFwdInv(model, data);

      // per-solver tolerances
      mjtNum epsilon;
      switch (solver) {
        case mjSOL_PGS:
          epsilon = MjTol(1e-6, 1e-2);
          break;
        case mjSOL_CG:
          epsilon = MjTol(1e-9, 1e-1);
          break;
        case mjSOL_NEWTON:
          epsilon = MjTol(1e-10, 2e-2);
          break;
      }
      EXPECT_LT(data->solver_fwdinv[0], epsilon)
          << solver_name[solver] << " diagexact=" << diagexact;
      EXPECT_LT(data->solver_fwdinv[1], epsilon)
          << solver_name[solver] << " diagexact=" << diagexact;
    }
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// test discrete-time inverse dynamics
TEST_F(InverseTest, DiscreteInverseMatch) {
  // load and allocate
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  int nv = model->nv;
  mjData* data = mj_makeData(model);
  int nstate = mj_stateSize(model, mjSTATE_INTEGRATION);
  mjtNum* state = (mjtNum*)mju_malloc(nstate * sizeof(mjtNum));
  mjtNum* qvel_next = (mjtNum*)mju_malloc(nv * sizeof(mjtNum));
  mjtNum* qacc_fd = (mjtNum*)mju_malloc(nv * sizeof(mjtNum));

  for (auto integrator :
       {mjINT_EULER, mjINT_IMPLICIT, mjINT_IMPLICITFAST, mjINT_DISCRETE}) {
    model->opt.integrator = integrator;
    for (bool invdiscrete : {false, true}) {
      // set/unset mjENBL_INVDISCRETE flag (affects both forward and inverse)
      if (invdiscrete) {
        model->opt.enableflags |= mjENBL_INVDISCRETE;
      } else {
        model->opt.enableflags &= ~mjENBL_INVDISCRETE;
      }

      // simulate
      mj_resetData(model, data);
      for (int i = 0; i < kSteps; ++i) {
        mj_step(model, data);
      }

      // save state
      mj_getState(model, data, state, mjSTATE_INTEGRATION);

      // call step, save new qvel
      mj_step(model, data);
      mju_copy(qvel_next, data->qvel, nv);

      // reset the state, compute discrete-time (finite-differenced) qacc
      mj_setState(model, data, state, mjSTATE_INTEGRATION);
      mju_sub(qacc_fd, qvel_next, data->qvel, nv);
      mju_scl(qacc_fd, qacc_fd, 1 / model->opt.timestep, nv);

      // call mj_forward, overwrite qacc with qacc_fd
      mj_forward(model, data);
      mju_copy(data->qacc, qacc_fd, nv);

      // call built-in testing function
      mj_compareFwdInv(model, data);

      // under the discrete integrator qacc is natively the step map and the
      // inverse consumes it directly: invdiscrete is implied and has no effect;
      // the roundtrip through the effective metric accumulates more roundoff
      if (invdiscrete || integrator == mjINT_DISCRETE) {
        mjtNum epsilon = integrator == mjINT_DISCRETE ? MjTol(1e-8, 0.05)
                                                      : MjTol(1e-9, 0.05);
        EXPECT_LT(data->solver_fwdinv[0], epsilon);
        EXPECT_LT(data->solver_fwdinv[1], epsilon);
      } else {
        EXPECT_GT(data->solver_fwdinv[0], 1.0);
        EXPECT_GT(data->solver_fwdinv[1], 1.0);
      }
    }
  }

  // deallocate
  mju_free(qacc_fd);
  mju_free(qvel_next);
  mju_free(state);
  mj_deleteData(data);
  mj_deleteModel(model);
}

// discrete-time inverse dynamics for a spinning free body under implicitfast:
// exercises the local unsymmetric block (bias derivative) in mj_discreteAcc
TEST_F(InverseTest, DiscreteInverseFreeBody) {
  // spinning box resting on a plane: standalone free body with active contacts
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" timestep="0.002">
      <flag invdiscrete="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1" friction="0.2"/>
      <body pos="0 0 .1">
        <joint type="free" damping="0.01"/>
        <geom type="box" size=".2 .15 .1" mass="2" pos=".02 -.01 .03" friction="0.2"/>
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
  int nv = m->nv;

  // spin about the vertical, small tumble components
  mj_resetData(m, d);
  d->qvel[3] = 0.5;
  d->qvel[4] = -0.3;
  d->qvel[5] = 20;

  // settle into persistent contact while still spinning
  for (int i = 0; i < kSteps; i++) {
    mj_step(m, d);
  }

  // save state, step, compute finite-differenced acceleration
  int nstate = mj_stateSize(m, mjSTATE_INTEGRATION);
  std::vector<mjtNum> state(nstate), qvel_next(nv), qacc_fd(nv);
  mj_getState(m, d, state.data(), mjSTATE_INTEGRATION);
  mj_step(m, d);
  mju_copy(qvel_next.data(), d->qvel, nv);
  mj_setState(m, d, state.data(), mjSTATE_INTEGRATION);
  mju_sub(qacc_fd.data(), qvel_next.data(), d->qvel, nv);
  mju_scl(qacc_fd.data(), qacc_fd.data(), 1 / m->opt.timestep, nv);

  // forward, overwrite qacc with finite-differenced acceleration, compare
  mj_forward(m, d);
  ASSERT_GT(d->ncon, 0) << "body should be in contact";
  ASSERT_GT(mju_abs(d->qvel[5]), 1) << "body should still be spinning";
  mju_copy(d->qacc, qacc_fd.data(), nv);
  mj_compareFwdInv(m, d);

  // measured residuals: ~6e-12 double, ~1.5e-2 single (float solver
  // convergence)
  mjtNum epsilon = MjTol(5e-7, 0.05);
  EXPECT_LT(d->solver_fwdinv[0], epsilon);
  EXPECT_LT(d->solver_fwdinv[1], epsilon);
}

// forward/inverse consistency for a free joint with stiffness and damping in
// the metric: covers the diagonal classes on free-joint dofs and the local
// gyroscopic treatment (mj_discreteGyro) together with its inverse mirror
TEST_F(InverseTest, DiscreteFreeJointInverseConsistency) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.005"/>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="0 0 .5">
        <joint type="free" stiffness="30" damping="2"/>
        <geom type="box" size=".2 .15 .1" mass="2" pos=".02 -.01 .03"/>
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
  int nv = m->nv;

  // phase 1, in flight: tumbling spin, no constraints. The local gyroscopic
  // treatment is active (mj_discreteGyro) and mj_inverse mirrors it: the
  // trajectory is passive, so the recovered applied force must vanish
  mj_resetData(m, d);
  d->qvel[0] = 0.3;
  d->qvel[2] = -0.2;
  d->qvel[3] = 0.5;
  d->qvel[4] = -0.3;
  d->qvel[5] = 8;
  for (int i = 0; i < 20; i++) {
    mj_step(m, d);
  }
  mj_forward(m, d);
  ASSERT_EQ(d->ncon, 0);
  mj_inverse(m, d);
  // measured: 3.6e-15 double, 1.9e-6 single
  mjtNum scale = 1 + mju_norm(d->qfrc_passive, nv);
  EXPECT_LT(mju_norm(d->qfrc_inverse, nv), MjTol(5e-14, 5e-6) * scale);

  // phase 2, resting on the floor: force-carrying contact plus the free-joint
  // metric
  for (int i = 0; i < 400; i++) {
    mj_step(m, d);
  }
  mj_forward(m, d);
  ASSERT_GT(d->ncon, 0);
  ASSERT_GT(mju_norm(d->efc_force, d->nefc), 1);
  mj_compareFwdInv(m, d);

  // measured residuals: 3.6e-8 double (solver convergence), 1.5e-7 single
  mjtNum epsilon = MjTol(5e-7, 2e-6);
  EXPECT_LT(d->solver_fwdinv[0], epsilon);
  EXPECT_LT(d->solver_fwdinv[1], epsilon);
}

// discrete inverse dynamics must not overwrite the forward factorizations
TEST_F(InverseTest, CompareFwdInvPreservesFactoredMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" integrator="implicit">
      <flag invdiscrete="enable"/>
    </option>
    <worldbody>
      <body pos="0 0 1">
        <joint name="j1" type="hinge" axis="1 0 0" damping="0.5"/>
        <geom type="capsule" size="0.05 0.2" mass="1"/>
        <body pos="0 0.2 0">
          <joint name="j2" type="hinge" axis="1 0 0" damping="0.5"/>
          <geom type="capsule" size="0.05 0.2" mass="1"/>
        </body>
      </body>
    </worldbody>
    <equality>
      <joint joint1="j1" joint2="j2" polycoef="0 1 0 0 0"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();
  d->qvel[0] = 1.0;

  // implicit: factored qLU
  mj_step(m, d);
  ASSERT_GT(d->nefc, 0);
  std::vector<mjtNum> qLU(d->qLU, d->qLU + m->nD);
  mj_compareFwdInv(m, d);
  for (int i = 0; i < m->nD; i++) {
    EXPECT_EQ(d->qLU[i], qLU[i]) << "qLU[" << i << "]";
  }

  // implicitfast: factored qH
  m->opt.integrator = mjINT_IMPLICITFAST;
  mj_step(m, d);
  ASSERT_GT(d->nefc, 0);
  std::vector<mjtNum> qH(d->qH, d->qH + m->nC);
  mj_compareFwdInv(m, d);
  for (int i = 0; i < m->nC; i++) {
    EXPECT_EQ(d->qH[i], qH[i]) << "qH[" << i << "]";
  }
}

// with damping disabled, Euler integrates without the implicit damping term, so
// discrete inverse dynamics must not add it either
TEST_F(InverseTest, DiscreteInverseEulerDamperDisabled) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="Euler" timestep="0.01">
      <flag invdiscrete="enable" damper="disable"/>
    </option>
    <worldbody>
      <body>
        <joint name="slide" type="slide" damping="5"/>
        <geom type="sphere" size="0.1" mass="1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="slide"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mjModel* m = model.get();
  mjData* d = data.get();
  int nv = m->nv;

  d->qvel[0] = 1.0;
  d->qfrc_applied[0] = 3.0;

  // finite-differenced discrete acceleration of one step
  int nstate = mj_stateSize(m, mjSTATE_INTEGRATION);
  std::vector<mjtNum> state(nstate), qvel_next(nv), qacc_fd(nv);
  mj_getState(m, d, state.data(), mjSTATE_INTEGRATION);
  mj_step(m, d);
  mju_copy(qvel_next.data(), d->qvel, nv);
  mj_setState(m, d, state.data(), mjSTATE_INTEGRATION);
  mju_sub(qacc_fd.data(), qvel_next.data(), d->qvel, nv);
  mju_scl(qacc_fd.data(), qacc_fd.data(), 1 / m->opt.timestep, nv);

  mj_forward(m, d);
  ASSERT_GT(d->nefc, 0);
  mju_copy(d->qacc, qacc_fd.data(), nv);
  mj_compareFwdInv(m, d);

  mjtNum epsilon = MjTol(1e-9, 1e-4);
  EXPECT_LT(d->solver_fwdinv[0], epsilon);
  EXPECT_LT(d->solver_fwdinv[1], epsilon);
}

}  // namespace
}  // namespace mujoco
