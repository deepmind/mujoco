// Copyright 2025 DeepMind Technologies Limited
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

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <ratio>
#include <utility>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

static mjtNum Timer() {
  using Clock = std::chrono::steady_clock;
  using Milliseconds = std::chrono::duration<double, std::milli>;
  static Clock::time_point start = Clock::now();
  return Milliseconds(Clock::now() - start).count();
}


StepControl::StepControl(ClockFn clock_fn) : clock_fn_(std::move(clock_fn)) { mjcb_time = Timer; }

float StepControl::GetSpeedMeasured() const { return speed_measured_; }

float StepControl::GetSpeed() const { return speed_; }

void StepControl::SetSpeed(float speed_percent_real_time) {
  float prev_speed = speed_;
  speed_ = std::clamp(speed_percent_real_time, .1f, 100.f);
  if (speed_ != prev_speed) {
    ForceSync();
  }
}

void StepControl::ForceSync() { force_sync_ = true; }

void StepControl::GetNoiseParameters(float& ctrl_noise_scale, float& ctrl_noise_rate) const {
  ctrl_noise_scale = ctrl_noise_std_;
  ctrl_noise_rate = ctrl_noise_rate_;
}

void StepControl::SetNoiseParameters(float ctrl_noise_scale, float ctrl_noise_rate) {
  ctrl_noise_std_ = ctrl_noise_scale;
  ctrl_noise_rate_ = ctrl_noise_rate;
}

void StepControl::SetPauseState(PauseState state) { pause_state_ = state; }

StepControl::PauseState StepControl::GetPauseState() const { return pause_state_; }

bool StepControl::ConsumeSingleStepRequest() {
  const bool requested = single_step_;
  single_step_ = false;
  return requested;
}

StepControl::Status StepControl::Advance(mjModel* m, mjData* d) {
  if (!m) {
    return Status::kOk;
  }


  if (pause_state_ == PauseState::kNormalPaused) {
    // When we eventually unpause, we need to make sure we sync to immediately
    // and step once. Without this we could step many times before rendering
    // resulting in a noticeable delay before the simulation restarts
    // (especially for large slowdowns).
    force_sync_ = true;

    if (!single_step_) {
      // Run mj_forward to update rendering and joint sliders.
      mj_forward(m, d);
      if (pause_update_) {
        mju_copy(d->qacc_warmstart, d->qacc, m->nv);
      }
      return Status::kPaused;
    }
    single_step_ = false;
  }

  const Clock::time_point start_cpu = clock_fn_();
  const double slowdown = 100. / std::clamp<double>(speed_, 0.001, 100.);
  double elapsed_cpu = Seconds(start_cpu - sync_cpu_).count();
  double elapsed_sim = d->time - sync_sim_;

  bool resync = false;

  // Resync if we're forced to.
  if (force_sync_) {
    force_sync_ = false;
    resync = true;
  }

  // Resync if we've never synced.
  if (sync_cpu_.time_since_epoch().count() == 0) {
    resync = true;
  }

  // Resync if any elapsed time is negative.
  if (elapsed_cpu < 0 || elapsed_sim < 0) {
    resync = true;
  }

  // Resync if the distance from the target simulation time is bigger than
  // sync_misalign_ (misalignment condition).
  if (std::abs(elapsed_cpu / slowdown - elapsed_sim) > sync_misalign_) {
    resync = true;
  }

  if (resync) {
    // Reset sync times.
    sync_cpu_ = start_cpu;
    sync_sim_ = d->time;
  }

  // Stepping loop.
  bool measured = false;
  while (true) {
    if (!resync) {
      const Clock::time_point now_cpu = clock_fn_();
      elapsed_cpu = Seconds(now_cpu - sync_cpu_).count();
      elapsed_sim = d->time - sync_sim_;

      // Stop stepping if simulation no longer lags cpu.
      if (elapsed_sim * slowdown >= elapsed_cpu) {
        return Status::kOk;
      }

      // Stop stepping if simulation is taking too long to catch up.
      // Note: 12ms == 70% of 1/60 seconds/frame.
      constexpr Clock::duration kMaxCpuTimeForSim = std::chrono::milliseconds(12);
      if (now_cpu - start_cpu >= kMaxCpuTimeForSim) {
        // Note: GetSpeed() and GetSpeedMeasured() will be different in this
        // case.
        return Status::kOk;
      }

      // Measure slowdown here in first viable in-sync step. This update
      // location is chosen to minimize visual noise caused by changing
      // measurements.
      if (!measured && elapsed_sim > 0) {
        double measured_slowdown = elapsed_cpu / elapsed_sim;
        speed_measured_ = 100. / measured_slowdown;
        measured = true;
      }
    }

    mjtNum prev_time = d->time;
    InjectNoise(m, d);
    if (pre_step_) {
      pre_step_(m, d);
    }
    mj_step(m, d);
    if (post_step_) {
      post_step_(m, d);
    }

    if (mjDISABLED(mjDSBL_AUTORESET)) {
      for (mjtWarning w : kDivergedWarnings) {
        if (d->warning[w].number > 0) {
          // Stop stepping if the simulation diverged.
          SetPauseState(PauseState::kNormalPaused);
          return Status::kDiverged;
        }
      }
    } else {
      // Stop stepping if we auto reset.
      if (d->time < prev_time) {
        return Status::kAutoReset;
      }
    }

    // Stop after one step if we resynced; next iteration will deal with timing.
    if (resync) {
      return Status::kOk;
    }
  }

  return Status::kDiverged;  // Unreachable
}

void StepControl::InjectNoise(const mjModel* m, mjData* d) {
  // no noise, return
  if (ctrl_noise_std_ <= 0) {
    return;
  }

  // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
  mjtNum rate = mju_exp(-m->opt.timestep / ctrl_noise_rate_);
  mjtNum scale = ctrl_noise_std_ * mju_sqrt(1 - rate * rate);

  for (int i = 0; i < m->nu; i++) {
    mjtNum bottom = 0;
    mjtNum top = 0;
    mjtNum midpoint = 0;
    mjtNum halfrange = 1;
    if (m->actuator_ctrllimited[i]) {
      bottom = m->actuator_ctrlrange[2 * i];
      top = m->actuator_ctrlrange[2 * i + 1];
      midpoint = 0.5 * (top + bottom);   // target of exponential decay
      halfrange = 0.5 * (top - bottom);  // scales noise
    }

    // exponential convergence to midpoint at ctrl_noise_rate
    d->ctrl[i] = rate * d->ctrl[i] + (1 - rate) * midpoint;

    // add noise
    d->ctrl[i] += scale * halfrange * mju_standardNormal(nullptr);

    // clip to range if limited
    if (m->actuator_ctrllimited[i]) {
      d->ctrl[i] = mju_clip(d->ctrl[i], bottom, top);
    }
  }
}

void StepControl::SetPreStepCallback(StepFn step_fn) { pre_step_ = step_fn; }

void StepControl::SetPostStepCallback(StepFn step_fn) { post_step_ = step_fn; }

}  // namespace mujoco::studio
