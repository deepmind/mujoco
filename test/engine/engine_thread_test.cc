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

// Tests for engine/engine_island.c.

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_thread.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using ThreadTest = MujocoTest;

TEST_F(ThreadTest, SingleAndMultiThreadedMatch) {
  auto model_path = GetModelPath("humanoid/22_humanoids.xml");
  std::array<char, 1024> error;
  mjModel* model =
      mj_loadXML(model_path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  model->opt.solver = mjSOL_CG;  // use CG solver

  mjModel* model_threaded =
      mj_loadXML(model_path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model_threaded, NotNull()) << error.data();
  model_threaded->opt.solver = mjSOL_CG;  // use CG solver

  mjData* data = mj_makeData(model);
  mjData* data_threaded = mj_makeData(model_threaded);

  // warm-up rollout to get a steady state
  static constexpr int kNumWarmupSteps = 100;
  for (int i = 0; i < kNumWarmupSteps; i++) {
    mj_step(model, data);
  }

  // sync the models and data
  int spec = mjSTATE_INTEGRATION;
  int size = mj_stateSize(model, spec);
  std::vector<mjtNum> initial_state(size);
  mj_getState(model, data, initial_state.data(), spec);
  mj_setState(model_threaded, data_threaded, initial_state.data(), spec);

  // bind a threadpool to the data_threaded
  mju_threadpool(data_threaded, 10);

  for (int i = 0; i < 10; ++i) {
    mj_step(model, data);
    mj_step(model_threaded, data_threaded);
  }

  // compare the mjData's.
  {
#define X(type, name, nr, nc)                               \
  EXPECT_EQ(std::memcmp(data->name, data_threaded->name,    \
                        sizeof(type) * (model->nr) * (nc)), \
            0)                                              \
      << "mjData::" #name " differs";
    MJDATA_POINTERS
#undef X
  }

  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteData(data_threaded);
  mj_deleteModel(model_threaded);
}

TEST_F(ThreadTest, IslandSingleAndMultiThreadedMatch) {
  auto model_path = GetModelPath("humanoid/22_humanoids.xml");
  std::array<char, 1024> error;
  mjModel* model =
      mj_loadXML(model_path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  model->opt.solver = mjSOL_CG;               // use CG solver
  model->opt.disableflags &= ~mjDSBL_ISLAND;  // enable islands

  mjModel* model_threaded =
      mj_loadXML(model_path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model_threaded, NotNull()) << error.data();
  model_threaded->opt.solver = mjSOL_CG;               // use CG solver
  model_threaded->opt.disableflags &= ~mjDSBL_ISLAND;  // enable islands

  mjData* data = mj_makeData(model);
  mjData* data_threaded = mj_makeData(model_threaded);

  // warm-up rollout to get a steady state
  static constexpr int kNumWarmupSteps = 100;
  for (int i = 0; i < kNumWarmupSteps; i++) {
    mj_step(model, data);
  }

  // sync the models and data
  int spec = mjSTATE_INTEGRATION;
  int size = mj_stateSize(model, spec);
  std::vector<mjtNum> initial_state(size);
  mj_getState(model, data, initial_state.data(), spec);
  mj_setState(model_threaded, data_threaded, initial_state.data(), spec);

  // bind a threadpool to the data_threaded
  mju_threadpool(data_threaded, 10);

  for (int i = 0; i < 10; ++i) {
    mj_step(model, data);
    mj_step(model_threaded, data_threaded);
  }

  // compare the mjData's.
  {
#define X(type, name, nr, nc)                               \
  EXPECT_EQ(std::memcmp(data->name, data_threaded->name,    \
                        sizeof(type) * (model->nr) * (nc)), \
            0)                                              \
      << "mjData::" #name " differs";
    MJDATA_POINTERS
#undef X
  }

  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteData(data_threaded);
  mj_deleteModel(model_threaded);
}

TEST_F(ThreadTest, ResetAndCopyPreserveThreadPoolAndClearStack) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024] = "";
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  mjData* d = mj_makeData(m.get());
  mjData* src = mj_makeData(m.get());

  mju_threadpool(d, 4);
  ASSERT_NE(d->threadpool, 0u);
  EXPECT_EQ(mju_numThread(d), 5);

  d->pstack = 128;
  mj_resetData(m.get(), d);
  EXPECT_EQ(d->pstack, 0u);
  EXPECT_EQ(mju_numThread(d), 5);

  mj_step(m.get(), src);
  mj_copyData(d, m.get(), src);
  EXPECT_NE(d->threadpool, 0u);
  EXPECT_EQ(mju_numThread(d), 5);

  mj_deleteData(src);
  mj_deleteData(d);
}

}  // namespace
}  // namespace mujoco
