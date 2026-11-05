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

#include "studio/sim/step_control.h"

#include <chrono>
#include <memory>

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>
#include "studio/sim/model_holder.h"

namespace mujoco::studio {
namespace {

class StepControlTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mjSpec* spec = mj_makeSpec();
    holder_ = ModelHolder::FromSpec(spec);
  }

  Clock::time_point now_{std::chrono::seconds(10)};
  std::unique_ptr<ModelHolder> holder_;
};

TEST_F(StepControlTest, ForceSyncAdvancesWhenClockDoesNotAdvance) {
  StepControl step_control([this]() { return now_; });
  mjModel* m = holder_->model();
  mjData* d = holder_->data();

  int steps = 0;
  step_control.SetPostStepCallback([&](mjModel*, mjData*) { ++steps; });

  // Initial Advance() has force_sync_ = true and must step once even when
  // adjacent clock readings are equal.
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 1);
  EXPECT_DOUBLE_EQ(d->time, m->opt.timestep);

  // Without ForceSync() and without clock advancement, simulation is ahead of
  // CPU and should not step.
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 1);

  // With ForceSync(), Advance() must resync and step once even when the clock
  // has not advanced.
  step_control.ForceSync();
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 2);
  EXPECT_DOUBLE_EQ(d->time, 2 * m->opt.timestep);
}

TEST_F(StepControlTest, SingleStepWhilePausedAdvancesWhenClockDoesNotAdvance) {
  StepControl step_control([this]() { return now_; });
  mjModel* m = holder_->model();
  mjData* d = holder_->data();

  int steps = 0;
  step_control.SetPostStepCallback([&](mjModel*, mjData*) { ++steps; });
  step_control.SetPauseState(StepControl::PauseState::kNormalPaused);

  // Without a single-step request, Advance() returns kPaused without stepping.
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kPaused);
  EXPECT_EQ(steps, 0);
  EXPECT_DOUBLE_EQ(d->time, 0.0);

  // With RequestSingleStep(), Advance() must consume the request and step once
  // even when adjacent clock readings are equal.
  for (int i = 1; i <= 5; ++i) {
    step_control.RequestSingleStep();
    EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
    EXPECT_EQ(steps, i);
  }

  // Subsequent Advance() without RequestSingleStep() remains paused.
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kPaused);
  EXPECT_EQ(steps, 5);
}

TEST_F(StepControlTest, InSyncSteppingCatchesUpToClock) {
  StepControl step_control([this]() { return now_; });
  mjModel* m = holder_->model();
  mjData* d = holder_->data();

  int steps = 0;
  step_control.SetPostStepCallback([&](mjModel*, mjData*) { ++steps; });

  // Initial sync at t = 10s steps once (d->time becomes 0.002s).
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 1);

  // Advance clock by 5ms (2.5 timesteps at default 2ms timestep).
  // d->time should catch up from 2ms -> 4ms -> 6ms (2 additional steps).
  // Measured speed on first in-sync step: elapsed_sim = 2ms, elapsed_cpu = 5ms
  // => slowdown = 2.5 => 40% real-time.
  now_ += std::chrono::milliseconds(5);
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 3);
  EXPECT_NEAR(d->time, 3 * m->opt.timestep, 1e-9);
  EXPECT_FLOAT_EQ(step_control.GetSpeedMeasured(), 40.0f);
}

TEST_F(StepControlTest, StopsSteppingWhenCpuBudgetExceeded) {
  StepControl step_control([this]() { return now_; });
  mjModel* m = holder_->model();
  mjData* d = holder_->data();

  int steps = 0;
  // Initial sync at t = 10s steps once (d->time becomes 2ms).
  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);

  // Advance clock by 50ms (within the 100ms sync_misalign_ window, so no
  // resync, requiring 24 steps to catch up), but each step takes 6ms of CPU
  // time. Stepping should stop after 2 steps when now_cpu - start_cpu reaches
  // the 12ms kMaxCpuTimeForSim budget.
  now_ += std::chrono::milliseconds(50);
  step_control.SetPostStepCallback([&](mjModel*, mjData*) {
    ++steps;
    now_ += std::chrono::milliseconds(6);
  });

  EXPECT_EQ(step_control.Advance(m, d), StepControl::Status::kOk);
  EXPECT_EQ(steps, 2);
}

}  // namespace
}  // namespace mujoco::studio
