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

#ifndef MUJOCO_STUDIO_SIM_STEP_CONTROL_H_
#define MUJOCO_STUDIO_SIM_STEP_CONTROL_H_

#include <chrono>
#include <functional>
#include <string>

#include <mujoco/mujoco.h>

namespace mujoco::studio {

using Seconds = std::chrono::duration<double>;
using Clock = std::chrono::steady_clock;
using ClockFn = std::function<Clock::time_point()>;
using StepFn = std::function<void(mjModel*, mjData*)>;

// State and logic for physics synchronization and stepping.
class StepControl {
 public:
  explicit StepControl(ClockFn clock_fn = Clock::now);

  enum class Status {
    kOk,

    // Simulation was not stepped because it is paused.
    kPaused,


    // Simulation diverged with autoreset enabled.
    kAutoReset,

    // Simulation diverged with autoreset disabled.
    // Note: Consider reporting mjData warning diagnostics in kDivergedWarnings.
    kDiverged,
  };

  // List of warnings that are checked to determine simulation divergence.
  static constexpr mjtWarning kDivergedWarnings[] = {mjWARN_BADQACC, mjWARN_BADQVEL,
                                                     mjWARN_BADQPOS};

  // Steps physics forward, respecting speed settings and refresh budget.
  Status Advance(mjModel* m, mjData* d);

  // Ensures the next call to Advance() will synchronize time and step once.
  void ForceSync();

  // Gets/sets the desired simulation speed as a percentage of real time.
  float GetSpeed() const;
  float GetSpeedMeasured() const;
  void SetSpeed(float speed);  // speed is clamped to [0.1%, 100%]

  // Gets/sets the control noise parameters applied before stepping.
  void GetNoiseParameters(float& noise_scale, float& noise_rate) const;
  void SetNoiseParameters(float noise_scale, float noise_rate);

  // Callbacks that will be invoked before/after each call to mj_step.
  void SetPreStepCallback(StepFn step_fn);
  void SetPostStepCallback(StepFn step_fn);

  enum class PauseState { kUnpaused, kNormalPaused };

  // Sets the pause state of the simulation.
  void SetPauseState(PauseState state);

  // Gets the current pause state of the simulation.
  PauseState GetPauseState() const;

  // If the simulation is paused, will perform a single step on the next
  // Advance() call.
  void RequestSingleStep() { single_step_ = true; }

  // Returns whether a single step is pending, clearing the request.
  // This function is useful for clients that do not step themselves but rather
  // forward the request to whoever does (e.g., a viewer thread may drive the
  // step control UI but the actual stepping occurs in the sim thread).
  bool ConsumeSingleStepRequest();

 private:
  std::string AdvanceOneStep(const mjModel* m, mjData* d);

  void InjectNoise(const mjModel* m, mjData* d);

  // Control noise standard deviation
  double ctrl_noise_std_ = 0;

  // Control noise correlation rate
  double ctrl_noise_rate_ = 0;

  // Desired simulation speed as a percentage of real time
  float speed_ = 100;

  // Measured simulation speed as a percentage of real time
  float speed_measured_ = -1;

  // If true, the next call to Advance() will synchronize time step once.
  bool force_sync_ = true;

  // CPU time (aka wall time) of the last synchronization event
  std::chrono::time_point<Clock> sync_cpu_;

  // Simulation time of the last synchronization event
  mjtNum sync_sim_ = 0;

  // Maximum mis-alignment before re-sync (simulation seconds)
  double sync_misalign_ = .1;

  PauseState pause_state_ = PauseState::kUnpaused;

  // Perform only a single step on the next call to Advance() if the simulation
  // is paused.
  bool single_step_ = false;

  // If true and paused, d->qacc_warmstart is set to d->qacc after mj_forward
  // which has the effect of making the constraint solver eventually converge
  // while the simulation is paused.
  bool pause_update_ = false;

  // Callbacks that can be invoked before/after physics is stepped.
  StepFn pre_step_ = nullptr;
  StepFn post_step_ = nullptr;

  ClockFn clock_fn_;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_SIM_STEP_CONTROL_H_
