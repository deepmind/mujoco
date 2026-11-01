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

// Tests of the entire pipeline that are not easily associated with one file.

#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_io.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

static const char* const kDefaultModel = "testdata/model.xml";

using ::testing::NotNull;
using ::testing::Pointwise;
using PipelineTest = MujocoTest;

// sparse and dense pipelines should produce the same results, for all solvers
TEST_F(PipelineTest, SparseDenseEquivalent) {
  const std::string xml_path = GetTestDataFilePath(kDefaultModel);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  const mjtNum tol = MjTol(1e-11, 5e-4);

  const char* sname[4] = {"NEWTON", "PGS", "CG", "NOSLIP"};
  mjtSolver solver[4] = {mjSOL_NEWTON, mjSOL_PGS, mjSOL_CG, mjSOL_NEWTON};

#ifdef mjUSESINGLE
  // CG and NOSLIP sparse-dense equivalence breaks at float32 precision.
  int nsolvers = 2;

  // the tolerance is below float32 precision, where sparse and dense can stop
  // at different iterations: run all of them
  model->opt.tolerance = 0;
#else
  int nsolvers = 4;
#endif

  for (int i = 0; i < nsolvers; i++) {
    model->opt.solver = solver[i];
    if (i == 3) {
      model->opt.noslip_iterations = 2;
    }

    // set dense jacobian, call mj_step, save qacc and new qpos
    model->opt.jacobian = mjJAC_DENSE;
    mj_resetDataKeyframe(model, data, 0);
    mj_step(model, data);
    std::vector<mjtNum> qacc_dense = AsVector(data->qacc, model->nv);
    std::vector<mjtNum> qpos_dense = AsVector(data->qpos, model->nq);

    // set sparse jacobian, call mj_step, save qacc and new qpos
    model->opt.jacobian = mjJAC_SPARSE;
    mj_resetDataKeyframe(model, data, 0);
    mj_step(model, data);
    std::vector<mjtNum> qacc_sparse = AsVector(data->qacc, model->nv);
    std::vector<mjtNum> qpos_sparse = AsVector(data->qpos, model->nq);

    // expect accelerations to be insignificantly different
    EXPECT_THAT(qacc_dense, Pointwise(MjNear(tol, tol), qacc_sparse))
        << "failed qacc equivalence for solver=" << sname[i];
    // expect positions to be insignificantly different
    EXPECT_THAT(qpos_dense, Pointwise(MjNear(tol, tol), qpos_sparse))
        << "failed qpos equivalence for solver=" << sname[i];
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// mj_forward should be idempotent when warm starts are disabled
TEST_F(PipelineTest, DeterministicNoWarmstart) {
  const std::string xml_path = GetTestDataFilePath(kDefaultModel);
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, nullptr, 0);
  mjData* data = mj_makeData(model);
  mjData* data2 = mj_makeData(model);

  // disable warmstarts
  model->opt.disableflags |= mjDSBL_WARMSTART;

  int nv = model->nv;

  int kNumSteps = 50;

  for (mjtSolver solver : {mjSOL_NEWTON, mjSOL_PGS, mjSOL_CG}) {
    model->opt.solver = solver;
    mj_resetData(model, data);
    mj_resetData(model, data2);

    for (int step = 0; step < kNumSteps; step++) {
      mj_step(model, data);
      mj_forward(model, data);

      mj_step(model, data2);
      mj_forward(model, data2);

      // test determinism: both models steps did the same thing
      EXPECT_EQ(AsVector(data->qacc, nv), AsVector(data2->qacc, nv));

      // one more mj_forward call on data2
      mj_forward(model, data2);

      // expect that the extra mj_forward call didn't change anything
      EXPECT_EQ(AsVector(data->qacc, nv), AsVector(data2->qacc, nv));
    }
  }

  mj_deleteData(data2);
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mujoco
