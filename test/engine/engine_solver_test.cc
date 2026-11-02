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

// Tests for engine/engine_solver.c

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::max;
using ::testing::NotNull;

using SolverTest = MujocoTest;

static const char* const kModelPath = "engine/testdata/solver/model.xml";
static const char* const kHumanoidPath = "engine/testdata/solver/humanoid.xml";

// compare accelerations produced by CG solver with and without islands
TEST_F(SolverTest, IslandsEquivalent) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  model->opt.solver = mjSOL_CG;                // use CG solver
  model->opt.jacobian = mjJAC_SPARSE;          // use sparse
  model->opt.tolerance = 0;                    // set tolerance to 0
  model->opt.ls_tolerance = 0;                 // set ls_tolerance to 0
  model->opt.ccd_tolerance = 0;                // set ccd_tolerance to 0
  model->opt.disableflags |= mjDSBL_MULTICCD;  // disable multiccd

  int nv = model->nv;

  int state_size = mj_stateSize(model, mjSTATE_INTEGRATION);
  mjtNum* state = (mjtNum*)mju_malloc(sizeof(mjtNum) * state_size);

  mjData* data_island = mj_makeData(model);
  mjData* data_noisland = mj_makeData(model);

  constexpr int kNumTol = 3;
  mjtNum maxiter[kNumTol] = {30, 40, 60};
  // Below are 3 tolerances associated with 3 different iteration counts.
  // Tolerances are set to be ~12x higher than failure thresholds.
  // The point of this test is to show that CG convergence is actually not very
  // precise, simply changing whether islands are used changes the solution by
  // quite a lot, even at high iteration count and zero {ls_}tolerance.
  // Increasing the iteration count higher than 60 does not improve convergence.
  // clang-format off
  mjtNum rtol[kNumTol] = {
      MjTol(1e-1, 1.2e-1),
      MjTol(3e-2, 2e-2),
      MjTol(1.3e-5, 2.8e-3)
  };
  // clang-format on

  for (int i = 0; i < kNumTol; ++i) {
    model->opt.iterations = maxiter[i];
    model->opt.ls_iterations = maxiter[i];

    for (bool coldstart : {true, false}) {
      mj_resetDataKeyframe(model, data_noisland, 0);

      if (coldstart) {
        model->opt.disableflags |= mjDSBL_WARMSTART;
      } else {
        model->opt.disableflags &= ~mjDSBL_WARMSTART;
      }

      mjtNum max_ratio = 0;
      mjtNum worst_diff = 0;
      mjtNum worst_scale = 1.0;
      mjtNum worst_expected = 0;
      mjtNum worst_actual = 0;
      std::string worst_time = "";
      int worst_dof = -1;

      while (data_noisland->time < .1) {
        mj_getState(model, data_noisland, state, mjSTATE_INTEGRATION);
        mj_setState(model, data_island, state, mjSTATE_INTEGRATION);

        model->opt.disableflags &= ~mjDSBL_ISLAND;  // enable islands
        mj_forward(model, data_island);

        model->opt.disableflags |= mjDSBL_ISLAND;  // disable islands
        mj_forward(model, data_noisland);

        for (int j = 0; j < nv; j++) {
          mjtNum diff = std::abs(data_noisland->qacc[j] - data_island->qacc[j]);
          mjtNum scale = 0.5 * max(static_cast<mjtNum>(2.0),
                                   std::abs(data_noisland->qacc[j]) +
                                       std::abs(data_island->qacc[j]));
          mjtNum ratio = diff / scale;
          if (ratio > max_ratio) {
            max_ratio = ratio;
            worst_diff = diff;
            worst_scale = scale;
            worst_expected = data_island->qacc[j];
            worst_actual = data_noisland->qacc[j];
            worst_time = std::to_string(data_noisland->time);
            worst_dof = j;
          }
        }

        mj_step(model, data_noisland);
      }

      // Assert once per condition with the worst offender.
      // rtol[i] is already scaled by MjTolScale() at initialization.
      mjtNum allowed_tol = worst_scale * rtol[i];
      EXPECT_NEAR(worst_actual, worst_expected, allowed_tol)
          << "Worst offender info:\n"
          << "time: " << worst_time << '\n'
          << "dof: " << worst_dof << '\n'
          << "maxiter: " << maxiter[i] << '\n'
          << "coldstart: " << coldstart << '\n'
          << "rtol: " << worst_scale * rtol[i] << '\n'
          << "actual diff: " << worst_diff << " (allowed: " << allowed_tol
          << ")";
    }
  }

  mj_deleteData(data_noisland);
  mj_deleteData(data_island);
  mju_free(state);
  mj_deleteModel(model);
}

// compare accelerations produced by CG/Newton solver with and without islands
TEST_F(SolverTest, IslandsEquivalentForward) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  int nv = model->nv;

  // set tolerance to 0 so opt.iterations are always run
  model->opt.tolerance = 0;

  mjData* data_island = mj_makeData(model);
  mjData* data_noisland = mj_makeData(model);

  for (bool warmstart : {false, true}) {
    for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
      for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
        for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
          if (warmstart) {
            model->opt.disableflags &= ~mjDSBL_WARMSTART;
          } else {
            model->opt.disableflags |= mjDSBL_WARMSTART;
          }
          model->opt.jacobian = jacobian;
          model->opt.solver = solver;
          model->opt.cone = cone;

          // disable islands, reset and step both datas to populate warmstart
          model->opt.disableflags |= mjDSBL_ISLAND;
          mj_resetDataKeyframe(model, data_island, 0);
          mj_resetDataKeyframe(model, data_noisland, 0);
          mj_step(model, data_island);
          mj_step(model, data_noisland);

          // forward with islands disabled
          mj_forward(model, data_noisland);

          // forward with islands enabled
          model->opt.disableflags &= ~mjDSBL_ISLAND;  // enable islands
          mj_forward(model, data_island);

          mjtNum max_diff = 0;
          mjtNum worst_expected = 0;
          mjtNum worst_actual = 0;
          int worst_idx = -1;
          mjtNum scale = 0.5 * (mju_norm(data_noisland->qacc, nv) +
                                mju_norm(data_island->qacc, nv));
          mjtNum rtol =
              solver == mjSOL_CG ? MjTol(1e-8, 1e-4) : MjTol(1e-13, 1e-3);
          mjtNum worst_allowed = scale * rtol;

          for (int j = 0; j < nv; j++) {
            mjtNum diff =
                std::abs(data_island->qacc[j] - data_noisland->qacc[j]);
            if (diff > max_diff) {
              max_diff = diff;
              worst_expected = data_noisland->qacc[j];
              worst_actual = data_island->qacc[j];
              worst_idx = j;
            }
          }

          EXPECT_NEAR(worst_actual, worst_expected, worst_allowed)
              << "Worst offender in IslandsEquivalentForward:\n"
              << "idx: " << worst_idx << '\n'
              << "warmstart: " << warmstart << '\n'
              << "jacobian: " << (jacobian ? "sparse" : "dense") << '\n'
              << "solver: " << (solver == mjSOL_CG ? "CG" : "Newton") << '\n'
              << "cone: " << (cone == 1 ? "elliptic" : "pyramidal") << '\n'
              << "actual diff: " << max_diff << " (allowed: " << worst_allowed
              << ")";
        }
      }
    }
  }

  mj_deleteData(data_noisland);
  mj_deleteData(data_island);
  mj_deleteModel(model);
}

TEST_F(SolverTest, SolversEquivalent) {
  struct SolverTolerances {
    mjtNum newton;
    mjtNum cg;
    mjtNum pgs_pyramidal;
    mjtNum pgs_elliptic;
  };

  // Relative tolerances: 10x above failure thresholds on Linux, clang, x86-64.
  // MjTol(f64, f32) selects the appropriate tolerance for the current build.
  const struct {
    const char* path;
    SolverTolerances tolerances;
  } kConfigs[] = {
      {.path = kModelPath,
       .tolerances =
           {
               .newton = MjTol(1e-13, 1e-5),
               .cg = MjTol(1e-13, 1e-5),
               .pgs_pyramidal = MjTol(1e-13, 1e-5),
               .pgs_elliptic = MjTol(1e-3, 1e-3),
           }},
      {.path = kHumanoidPath,
       .tolerances =
           {
               .newton = MjTol(1e-13, 1e-5),
               .cg = MjTol(1e-12, 1e-5),
               .pgs_pyramidal = MjTol(1e-12, 1e-5),
               .pgs_elliptic = MjTol(1e-8, 1e-4),
           }},
  };

  for (const auto& config : kConfigs) {
    const std::string xml_path = GetTestDataFilePath(config.path);
    char error[1024];
    mjModel* model =
        mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(model, NotNull()) << error;

    model->opt.tolerance = 0;                     // set tolerance to 0
    model->opt.iterations = 500;                  // set iterations to 500
    model->opt.disableflags |= mjDSBL_WARMSTART;  // disable warmstart
    int nv = model->nv;

    mjData* data = mj_makeData(model);
    mjData* data_truth = mj_makeData(model);

    for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
      model->opt.cone = cone;

      // use Newton Dense as ground truth
      model->opt.solver = mjSOL_NEWTON;
      model->opt.jacobian = mjJAC_DENSE;
      mj_resetDataKeyframe(model, data_truth, 0);
      mj_forward(model, data_truth);

      mjtNum scale = mju_norm(data_truth->qfrc_constraint, nv);

      for (mjtSolver solver : {mjSOL_NEWTON, mjSOL_CG, mjSOL_PGS}) {
        mjtNum rtol;
        switch (solver) {
          case mjSOL_NEWTON:
            rtol = config.tolerances.newton;
            break;
          case mjSOL_CG:
            rtol = config.tolerances.cg;
            break;
          case mjSOL_PGS:
            rtol = cone == mjCONE_PYRAMIDAL ? config.tolerances.pgs_pyramidal
                                            : config.tolerances.pgs_elliptic;
            break;
        }

        mjtNum tolerance = scale * rtol;

        for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
          model->opt.solver = solver;
          model->opt.jacobian = jacobian;

          mj_resetDataKeyframe(model, data, 0);
          mj_forward(model, data);

          const char* cone_str =
              (cone == mjCONE_PYRAMIDAL ? "pyramidal" : "elliptic");
          const char* solver_str =
              (solver == mjSOL_NEWTON ? "Newton"
                                      : (solver == mjSOL_CG ? "CG" : "PGS"));
          const char* jacobian_str =
              (jacobian == mjJAC_DENSE ? "dense" : "sparse");

          mjtNum max_diff = 0;
          mjtNum worst_expected = 0;
          mjtNum worst_actual = 0;
          int worst_idx = -1;

          for (int j = 0; j < nv; j++) {
            mjtNum diff = std::abs(data->qfrc_constraint[j] -
                                   data_truth->qfrc_constraint[j]);
            if (diff > max_diff) {
              max_diff = diff;
              worst_expected = data_truth->qfrc_constraint[j];
              worst_actual = data->qfrc_constraint[j];
              worst_idx = j;
            }
          }

          EXPECT_NEAR(worst_actual, worst_expected, tolerance)
              << "Worst offender in SolversEquivalent:\n"
              << "idx: " << worst_idx << '\n'
              << "model: " << config.path << "\n"
              << "cone: " << cone_str << "\n"
              << "solver: " << solver_str << "\n"
              << "jacobian: " << jacobian_str << "\n"
              << "actual diff: " << max_diff << " (allowed: " << tolerance
              << ")";
        }
      }
    }

    mj_deleteData(data_truth);
    mj_deleteData(data);
    mj_deleteModel(model);
  }
}

TEST_F(SolverTest, EllipticLineSearchPrecisionDiagnostics) {
  std::string xml = R"(
  <mujoco>
    <option cone="elliptic" solver="Newton"/>
    <worldbody>
      <geom name="floor" type="plane" size="10 10 1"/>
      <body name="box" pos="0 0 0.499">
        <joint type="free"/>
        <geom type="box" size="0.5 0.5 0.5" mass="1" friction="0.5"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Set gravity to 0
  model->opt.gravity[0] = 0;
  model->opt.gravity[1] = 0;
  model->opt.gravity[2] = 0;

  for (double fn : {1e2, 1e4, 1e6, 1e8}) {
    mj_resetData(model.get(), data.get());

    // Apply large downward force
    data->qfrc_applied[2] = -fn;

    // Apply large lateral force (dynamic friction limit is 0.5 * fn)
    double ft = fn * 1.5;
    data->qfrc_applied[0] = ft;

    mj_forward(model.get(), data.get());

    int niter = std::min(data->solver_niter[0], mjNSOLVER);
    for (int i = 0; i < niter; ++i) {
      const mjSolverStat& stat = data->solver[i];
      EXPECT_GE(stat.improvement, -MjTol(1e-5, 100.0));
    }
  }
}

TEST_F(SolverTest, EllipticLineSearchCancellation) {
  std::string xml = R"(
  <mujoco>
    <option timestep="0.016" integrator="discrete" cone="elliptic" impratio="10"
            noslip_iterations="2">
      <flag refsafe="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .1" solref="0.002 1"/>
      <body pos="0 0 0.05">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" mass="1" solref="0.002 1" condim="6"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Sliding box decelerating to rest across slightly perturbed initial
  // velocities: when the Newton step cancels the tangential velocity in the
  // elliptic cone middle zone, the linesearch second derivative must remain
  // positive without triggering a non-convexity warning.
  for (double v0 : {1.95, 1.97, 1.99, 2.01, 2.03, 2.05}) {
    mj_resetData(model.get(), data.get());
    data->qvel[0] = v0;
    data->qvel[1] = 1e-6;
    for (int i = 0; i < 120; ++i) {
      mj_step(model.get(), data.get());
    }
  }
}

// Newton terminates early when the decrement predicts sub-tolerance improvement
TEST_F(SolverTest, NewtonDecrementTermination) {
  const std::string xml_path = GetTestDataFilePath(kHumanoidPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  model->opt.solver = mjSOL_NEWTON;
  model->opt.disableflags |= mjDSBL_ISLAND;  // monolithic solve
  model->opt.iterations = 100;
  const mjtNum tolerance = MjTol(1e-6, 1e-4);

  int state_size = mj_stateSize(model, mjSTATE_FULLPHYSICS);
  mjtNum* state = (mjtNum*)mju_malloc(sizeof(mjtNum) * state_size);

  mjData* data = mj_makeData(model);
  mjData* data_test = mj_makeData(model);
  mjData* data_deep = mj_makeData(model);
  mj_resetDataKeyframe(model, data, 0);

  int nfired = 0;
  mjtNum max_leftover = 0;
  for (int step = 0; step < 200; step++) {
    model->opt.tolerance = tolerance;
    mj_step(model, data);
    mj_getState(model, data, state, mjSTATE_FULLPHYSICS);

    // solve with the test tolerance
    mj_setState(model, data_test, state, mjSTATE_FULLPHYSICS);
    mj_forward(model, data_test);

    // reference: tolerance 0 runs until the line search finds no improvement
    model->opt.tolerance = 0;
    mj_setState(model, data_deep, state, mjSTATE_FULLPHYSICS);
    mj_forward(model, data_deep);

    // accuracy: both runs produce identical iterates up to the test run's
    // stopping point, so the cost improvement forgone by early termination is
    // the sum of the deep run's remaining (scaled) improvements
    int niter = data_test->solver_niter[0];
    int niter_deep = std::min(data_deep->solver_niter[0], mjNSOLVER);
    mjtNum leftover = 0;
    for (int i = niter; i < niter_deep; i++) {
      leftover += max(static_cast<mjtNum>(0), data_deep->solver[i].improvement);
    }
    max_leftover = max(max_leftover, leftover);

    // count decrement terminations: the test run stopped while both existing
    // criteria were above tolerance, and the deep run shows that the next
    // iteration would have improved the cost by less than tolerance
    if (niter > 0 && niter < std::min(model->opt.iterations, mjNSOLVER) &&
        data_deep->solver_niter[0] > niter) {
      const mjSolverStat& last = data_test->solver[niter - 1];
      const mjSolverStat& next = data_deep->solver[niter];
      if (last.improvement >= tolerance && last.gradient >= tolerance &&
          next.improvement < tolerance) {
        nfired++;
      }
    }
  }

  EXPECT_LT(max_leftover, 10 * tolerance)
      << "early termination forgoes more than a small multiple of tolerance";
  EXPECT_GT(nfired, 0)
      << "no state exercised the Newton decrement termination criterion";

  mj_deleteData(data_deep);
  mj_deleteData(data_test);
  mj_deleteData(data);
  mju_free(state);
  mj_deleteModel(model);
}

// a settled, warmstarted scene certifies convergence and solves in zero
// iterations
TEST_F(SolverTest, WarmstartZeroIterations) {
  std::string xml = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 0.1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  MjDataPtr data = MakeData(model);
  model->opt.disableflags |=
      mjDSBL_ISLAND;  // monolithic solve: stats in slot 0
  model->opt.enableflags |= mjENBL_FWDINV;

  int nv = model->nv;
  int state_size = mj_stateSize(model.get(), mjSTATE_FULLPHYSICS);
  std::vector<mjtNum> state(state_size);
  std::vector<mjtNum> qacc(nv), qfrc(nv);

  // float32 cannot resolve the default tolerance: use a resolvable one
  const mjtNum tolerance = MjTol(1e-8, 1e-6);

  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
      for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
        std::string config =
            std::string(solver == mjSOL_CG ? "CG" : "Newton") +
            (cone == mjCONE_ELLIPTIC ? "/elliptic" : "/pyramidal") +
            (jacobian == mjJAC_SPARSE ? "/sparse" : "/dense");
        model->opt.solver = solver;
        model->opt.cone = cone;
        model->opt.jacobian = jacobian;
        model->opt.tolerance = tolerance;

        // settle the box on the plane
        mj_resetData(model.get(), data.get());
        for (int i = 0; i < 500; i++) {
          mj_step(model.get(), data.get());
        }
        mj_getState(model.get(), data.get(), state.data(), mjSTATE_FULLPHYSICS);

        // solve once more: certificate fires, forward/inverse stay consistent
        mj_forward(model.get(), data.get());
        EXPECT_EQ(data->solver_niter[0], 0) << config;

        // thresholds here and below are ~10x above measured, per precision
        EXPECT_LT(data->solver_fwdinv[0], MjTol(1e-12, 1e-4)) << config;
        EXPECT_LT(data->solver_fwdinv[1], MjTol(1e-2, 2e-1)) << config;
        mju_copy(qacc.data(), data->qacc, nv);
        mju_copy(qfrc.data(), data->qfrc_constraint, nv);

        // control arm: tolerance = 0 disables the certificate, full solve from
        // the same state must agree with the skipped solve
        model->opt.tolerance = 0;
        mj_setState(model.get(), data.get(), state.data(), mjSTATE_FULLPHYSICS);
        mj_forward(model.get(), data.get());
        mjtNum dqacc = 0, dqfrc = 0;
        for (int j = 0; j < nv; j++) {
          dqacc = max(dqacc, std::abs(qacc[j] - data->qacc[j]));
          dqfrc = max(dqfrc, std::abs(qfrc[j] - data->qfrc_constraint[j]));
        }
        EXPECT_LT(dqacc, MjTol(2e-4, 1.5e-3)) << config;
        EXPECT_LT(dqfrc, MjTol(2e-2, 4e-1)) << config;

        // guard: a perturbed scene does not certify
        model->opt.tolerance = tolerance;
        mj_setState(model.get(), data.get(), state.data(), mjSTATE_FULLPHYSICS);
        data->qfrc_applied[0] = 5;
        mj_forward(model.get(), data.get());
        EXPECT_GT(data->solver_niter[0], 0) << config;
        data->qfrc_applied[0] = 0;
      }
    }
  }
}

// per-island certificates: settled islands solve in zero iterations while
// islands with new loads solve normally
TEST_F(SolverTest, WarmstartZeroIterationsIslands) {
  std::string xml = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="2 2 .1"/>
      <body pos="-0.5 0 0.1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <body pos="0.5 0 0.1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // float32 cannot resolve the default tolerance: use a resolvable one
  model->opt.tolerance = MjTol(1e-8, 2e-6);

  // settle both boxes, islands enabled (default)
  for (int i = 0; i < 500; i++) {
    mj_step(model.get(), data.get());
  }

  // both islands certify: zero iterations everywhere
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->nisland, 2);
  EXPECT_EQ(data->solver_niter[0], 0);
  EXPECT_EQ(data->solver_niter[1], 0);

  // kick the second box: its island solves, the settled island still certifies
  data->qfrc_applied[6] = 5;
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->nisland, 2);
  int island1 = data->dof_island[0];
  int island2 = data->dof_island[6];
  ASSERT_GE(island1, 0);
  ASSERT_GE(island2, 0);
  ASSERT_NE(island1, island2);
  EXPECT_EQ(data->solver_niter[island1], 0);
  EXPECT_GT(data->solver_niter[island2], 0);
}

// tolerance == 0 disables early termination, including the Newton decrement
TEST_F(SolverTest, ZeroToleranceDisablesTermination) {
  const std::string xml_path = GetTestDataFilePath(kHumanoidPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  model->opt.solver = mjSOL_NEWTON;
  model->opt.disableflags |= mjDSBL_ISLAND | mjDSBL_WARMSTART;
  model->opt.tolerance = 0;
  model->opt.iterations = 3;

  mjData* data = mj_makeData(model);
  for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
    model->opt.cone = cone;
    mj_resetDataKeyframe(model, data, 0);
    mj_forward(model, data);
    EXPECT_EQ(data->solver_niter[0], 3)
        << "cone: " << (cone == mjCONE_ELLIPTIC ? "elliptic" : "pyramidal");
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// With condim 6 and the default friction (1, 0.005, 0.0001) the local
// elliptic-cone Hessian spans the friction ratios squared, a condition number
// around 1e9, which exhausts the single-precision mantissa. Newton factorizes
// it per contact and folds the factor into the full Hessian with rank-1
// updates, so a Cholesky that responds to a vanishing pivot by clamping the
// diagonal and then dividing the rest of the column by it -- scaling that
// column by 1/sqrt(mindiag) -- injects enormous coupling where there is no
// curvature. This pose reached rank 4 of 6 one step before qacc went to NaN.
TEST_F(SolverTest, EllipticConeHessianSurvivesFrictionRatios) {
  constexpr char xml[] = R"(
  <mujoco>
    <default>
      <geom condim="6"/>
    </default>
    <worldbody>
      <geom name="floor" type="plane" size=".5 1 .01"/>
      <body name="b1" pos="0 0 .4" euler="5 4 3">
        <freejoint/>
        <geom type="box" pos="0 0 .06" size=".15 .15 .03"/>
        <geom type="box" pos="-.05 0 .005" size=".04 .04 .025" euler="1 1 1"/>
        <geom size=".05" pos=".1 -.1 .04"/>
      </body>
      <body name="b2" pos="0 0 .2">
        <joint type="ball" springdamper="0.1 1"/>
        <geom type="box" size=".2 .2 .05"/>
        <geom size=".05" pos=".1 .1 .05"/>
        <geom type="box" size=".05 .05 .01" pos=".1 -.1 .06" euler="2 2 2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.cone = mjCONE_ELLIPTIC;
  model->opt.solver = mjSOL_NEWTON;

  // both factorizations reach the same pivot, at different steps
  for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
    model->opt.jacobian = jacobian;
    MjDataPtr data = MakeData(model);

    // bounded by step count, not by data->time: a divergence resets mjData and
    // rewinds the clock, so a time-based loop would never terminate
    for (int step = 0; step < 200; step++) {
      mj_step(model.get(), data.get());
      for (int i = 0; i < mjNWARNING; i++) {
        ASSERT_EQ(data->warning[i].number, 0)
            << "warning " << i << " at step " << step << ", jacobian "
            << jacobian;
      }
    }
  }
}

// With many sliding contacts, the sparse elliptic Newton solver rebuilds the
// cone-augmented factor with one refactorization instead of per-contact rank-1
// updates (cone folding). The folded factor must match the incremental one, so
// sparse and dense (which always uses rank-1 updates) must agree. The scene is
// a walled corner where a pile of 29 boxes clumps under diagonal gravity and
// slides steadily, holding most contacts in cone state; mixed condim exercises
// contact dim 3, 4 and 6.
TEST_F(SolverTest, ConeFoldEquivalent) {
  constexpr char xml[] = R"(
  <mujoco>
    <option cone="elliptic" solver="Newton" gravity="-8 -8 -10">
      <flag island="disable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .01"/>
      <geom name="wallx" type="box" size=".02 .5 .1" pos="-.5 0 .1"/>
      <geom name="wally" type="box" size=".5 .02 .1" pos="0 -.5 .1"/>
      <replicate count="2" offset="0 .11 0">
        <replicate count="2" offset=".11 0 0">
          <replicate count="2" offset="0 0 .11">
            <body pos="-.35 -.35 .06">
              <freejoint/>
              <geom type="box" size=".05 .05 .05"/>
            </body>
          </replicate>
        </replicate>
      </replicate>
      <body pos="-.1 -.1 .06">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" condim="4" friction="1 .01 .01"/>
      </body>
      <body pos="-.1 -.25 .06">
        <freejoint/>
        <geom type="box" size=".05 .05 .05" condim="6" friction="1 .01 .0001"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.tolerance = 0;  // run all iterations for tight convergence
  int nv = model->nv;

  model->opt.jacobian = mjJAC_SPARSE;
  MjDataPtr data_sparse = MakeData(model);
  MjDataPtr data_dense = MakeData(model);

  // let the pile fall and clump against the walls (sparse drives the state)
  for (int step = 0; step < 40; step++) {
    mj_step(model.get(), data_sparse.get());
  }

  int state_size = mj_stateSize(model.get(), mjSTATE_PHYSICS);
  std::vector<mjtNum> state(state_size);

  // compare sparse (folding) and dense (rank-1 updates) accelerations while
  // the clump slides
  for (int step = 0; step < 15; step++) {
    mj_getState(model.get(), data_sparse.get(), state.data(), mjSTATE_PHYSICS);

    model->opt.jacobian = mjJAC_SPARSE;
    mj_forward(model.get(), data_sparse.get());

    model->opt.jacobian = mjJAC_DENSE;
    mj_setState(model.get(), data_dense.get(), state.data(), mjSTATE_PHYSICS);
    mj_forward(model.get(), data_dense.get());

    for (int j = 0; j < nv; j++) {
      mjtNum scale = max(static_cast<mjtNum>(1), std::abs(data_dense->qacc[j]));
      EXPECT_NEAR(data_sparse->qacc[j] / scale, data_dense->qacc[j] / scale,
                  MjTol(1e-6, 1e-3))
          << "dof " << j << " at step " << step;
    }
    for (int i = 0; i < mjNWARNING; i++) {
      ASSERT_EQ(data_sparse->warning[i].number, 0)
          << "warning " << i << " at step " << step;
    }

    model->opt.jacobian = mjJAC_SPARSE;
    mj_step(model.get(), data_sparse.get());
  }
}

// A chain of thin capsules whose tip is connected to a slider: the connect
// fills H = M + J'*D*J, and once the chain sags, the twist pivots of the
// reverse factorization (the capsules' tiny axial inertia) are five orders of
// magnitude below their diagonals. In single precision rounding loses one:
// Newton must clamp and decouple it and continue, not abort or diverge.
TEST_F(SolverTest, NewtonHessianSurvivesLostPivot) {
  std::string xml = "<mujoco><worldbody>";
  for (int i = 0; i < 40; i++) {
    xml += (i == 0)    ? "<body pos='0 0 0'>"
           : (i == 39) ? "<body name='B_last' pos='.025 0 0'>"
                       : "<body pos='.025 0 0'>";
    xml += "<joint type='ball' damping='.015'/>";
    xml +=
        "<geom type='capsule' size='.005' fromto='0 0 0 .025 0 0' "
        "contype='0' conaffinity='0'/>";
  }
  for (int i = 0; i < 40; i++) {
    xml += "</body>";
  }
  xml += R"(
      <body name="slider" pos="1 0 0">
        <joint type="slide" axis="1 0 0" damping=".1"/>
        <geom size=".01" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <equality>
      <connect body1="B_last" body2="slider" anchor=".025 0 0"/>
    </equality>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml.c_str(), error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  model->opt.solver = mjSOL_NEWTON;

  for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
    for (int disableflags : {0, (int)mjDSBL_ISLAND}) {
      model->opt.jacobian = jacobian;
      model->opt.disableflags = disableflags;
      MjDataPtr data = MakeData(model);

      // bounded by step count: a divergence resets mjData and rewinds the clock
      for (int step = 0; step < 200; step++) {
        mj_step(model.get(), data.get());
      }
      EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0)
          << "jacobian " << jacobian << ", disableflags " << disableflags;
    }
  }
}

// When the arena has enough room for PrimalAllocate and CG but not for Newton's
// sparse Hessian/Cholesky factor, MakeHessian warns (mjWARN_CNSTRFULL) and
// mj_solPrimal falls back to CG instead of aborting in mj_stackAlloc.
TEST_F(SolverTest, NewtonFallsBackToCGOnArenaExhaustion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" solver="Newton"/>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <flexcomp name="cloth" type="grid" dim="2" count="20 20 1"
                spacing=".01 .01 .01" pos="0 0 .001" radius=".002" mass="1">
        <contact selfcollide="none"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  // first run with ample arena to measure full Newton vs CG arena usage
  MjDataPtr data_full = MakeData(model);
  mj_step(model.get(), data_full.get());
  ASSERT_EQ(data_full->warning[mjWARN_CNSTRFULL].number, 0);

  model->opt.solver = mjSOL_CG;
  MjDataPtr data_cg = MakeData(model);
  mj_step(model.get(), data_cg.get());
  ASSERT_LT(data_cg->maxuse_arena, data_full->maxuse_arena);

  // set narena large enough for PrimalAllocate + CG, but too small for Newton's
  // L factor
  model->opt.solver = mjSOL_NEWTON;
  model->narena = (data_cg->maxuse_arena + data_full->maxuse_arena) / 2;
  MjDataPtr data_tight = MakeData(model);
  mock_warning_handler.ExpectWarnings("Insufficient arena memory");
  mj_step(model.get(), data_tight.get());
  EXPECT_GT(data_tight->warning[mjWARN_CNSTRFULL].number, 0);
  EXPECT_EQ(data_tight->warning[mjWARN_BADQACC].number, 0);

  // also test dense Newton with discrete flex metric (flg_metric && !is_sparse)
  static constexpr char xml_dense[] = R"(
  <mujoco>
    <option integrator="discrete" solver="Newton" jacobian="dense">
      <flag island="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="2 2 .01"/>
      <flexcomp name="rod" type="grid" dim="1" count="12 1 1"
                spacing=".02 .02 .02" pos="0 0 .001" radius=".005" mass="1">
        <contact selfcollide="none"/>
        <elasticity young="1e3" damping="1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model_dense = LoadModelFromString(xml_dense, error, sizeof(error));
  ASSERT_THAT(model_dense.get(), NotNull()) << error;

  MjDataPtr dense_full = MakeData(model_dense);
  mj_step(model_dense.get(), dense_full.get());
  ASSERT_EQ(dense_full->warning[mjWARN_CNSTRFULL].number, 0);

  // set narena just above dense_full->maxuse_arena so PrimalAllocate + cg_buf
  // fit, but MakeHessian's loop_bytes reserve check fails and falls back to CG
  model_dense->narena = dense_full->maxuse_arena + 1024;
  MjDataPtr dense_tight = MakeData(model_dense);
  mock_warning_handler.ExpectWarnings("Insufficient arena memory");
  mj_step(model_dense.get(), dense_tight.get());
  EXPECT_GT(dense_tight->warning[mjWARN_CNSTRFULL].number, 0);
  EXPECT_EQ(dense_tight->warning[mjWARN_BADQACC].number, 0);
}

// Elliptic cone Hessian and PrimalEval remain finite when the initial iterate
// has a tangential residual T > 0 small enough that T^3 underflows to zero.
TEST_F(SolverTest, EllipticConeSmallTangentNoNaN) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option cone="elliptic"/>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .099">
        <freejoint/>
        <geom size=".1" condim="6" friction="1 .01 .001" solref="-0 -1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  // zero contact stiffness at rest so efc_aref == 0: a tiny x warmstart gives
  // N == 0 and T^2 normal but T^3 == 0, i.e. the middle zone of the cone at
  // alpha = 0 (CG reaches it only via PrimalEval, Newton also via the Hessian)
  const mjtNum tiny_ax = MjEps(1e-130, 1e-18);
  for (mjtSolver solver : {mjSOL_NEWTON, mjSOL_CG}) {
    for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
      model->opt.solver = solver;
      model->opt.jacobian = jacobian;
      MjDataPtr data = MakeData(model);
      data->qacc_warmstart[0] = tiny_ax;

      mj_step(model.get(), data.get());
      EXPECT_EQ(data->warning[mjWARN_BADQACC].number, 0)
          << "solver " << solver << ", jacobian " << jacobian;
      for (int j = 0; j < model->nv; j++) {
        EXPECT_FALSE(mju_isBad(data->qacc[j]))
            << "dof " << j << ", solver " << solver << ", jacobian "
            << jacobian;
      }
    }
  }
}

TEST_F(SolverTest, UnconstrainedMetricIslandBeyondMaxStat) {
  std::string xml = R"(
  <mujoco>
    <option integrator="discrete" solver="CG"/>
    <default><geom size=".1" contype="0" conaffinity="0"/></default>
    <worldbody>)";
  for (int i = 0; i < mjNISLAND; i++) {
    xml += "<body><joint type='slide' frictionloss='1'/><geom/></body>";
  }
  xml += R"(
      <body><joint name="ja" type="slide"/><geom/></body>
      <body><joint name="jb" type="slide"/><geom/></body>
    </worldbody>
    <tendon>
      <fixed damping="1"><joint joint="ja" coef="1"/><joint joint="jb" coef="1"/></fixed>
    </tendon>
  </mujoco>)";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml.c_str(), error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Island 0 sets solver_nnz[0] = 1 (nJ). Because solver_nnz immediately
  // follows solver_niter[mjNISLAND] in mjData, an out-of-range write to
  // solver_niter[mjNISLAND] for island 20 (nefc == 0) would land on and
  // clobber solver_nnz[0].
  mj_forward(model.get(), data.get());

  ASSERT_EQ(data->nisland, mjNISLAND + 1);
  ASSERT_EQ(data->island_nefc[mjNISLAND], 0);
  EXPECT_EQ(data->solver_nnz[0], 1);
}

// under integrator="discrete", CG does not false-certify at 0 iterations when
// a stiff coupled tendon suppresses P^-1*grad along a zero-strain mode
TEST_F(SolverTest, DiscreteCGCoupledTendonConverges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete" timestep="0.01" gravity="0 0 0"
      iterations="50"/>
    <worldbody>
      <body>
        <joint name="j1" type="slide"/>
        <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body>
        <joint name="j2" type="slide"/>
        <geom size="0.1" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="1e9">
        <joint joint="j1" coef="1"/>
        <joint joint="j2" coef="-1"/>
      </fixed>
    </tendon>
    <equality>
      <joint joint1="j1" solimp="0.01 0.01 0.01" solref="0 -1e-6"/>
      <joint joint1="j2" solimp="0.01 0.01 0.01" solref="0 -1e-6"/>
    </equality>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  const mjtNum tolerance = MjTol(1e-8, 1e-5);

  // reference solution via Newton with tolerance = 0
  model->opt.solver = mjSOL_NEWTON;
  model->opt.tolerance = 0;
  MjDataPtr data_ref = MakeData(model);
  data_ref->qfrc_applied[1] = 0.2;
  mj_forward(model.get(), data_ref.get());

  for (bool disable_island : {true, false}) {
    for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
      if (disable_island) {
        model->opt.disableflags |= mjDSBL_ISLAND;
      } else {
        model->opt.disableflags &= ~mjDSBL_ISLAND;
      }
      model->opt.solver = mjSOL_CG;
      model->opt.jacobian = jacobian;
      model->opt.tolerance = tolerance;

      MjDataPtr data = MakeData(model);
      data->qfrc_applied[1] = 0.2;
      mj_forward(model.get(), data.get());

      EXPECT_GT(data->solver_niter[0], 0);
      EXPECT_NEAR(data->qacc[0], data_ref->qacc[0], MjTol(1e-7, 1e-5));
      EXPECT_NEAR(data->qacc[1], data_ref->qacc[1], MjTol(1e-7, 1e-5));

#ifndef mjUSESINGLE
      // exact warmstart still certifies in 0 iterations
      mju_copy(data->qacc_warmstart, data_ref->qacc, model->nv);
      mj_forward(model.get(), data.get());
      EXPECT_EQ(data->solver_niter[0], 0);
#endif
    }
  }
}

}  // namespace
}  // namespace mujoco
