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

#ifndef MUJOCO_STUDIO_SIM_SIM_PROFILER_H_
#define MUJOCO_STUDIO_SIM_SIM_PROFILER_H_

#include <array>
#include <vector>

#include <imgui.h>
#include <mujoco/mujoco.h>

namespace mujoco::studio {

// Collects and displays profiling data for MuJoCo simulations.
class SimProfiler {
 public:
  SimProfiler();

  // Clears all captured profiling data.
  void Clear();

  // Resets the timer reference used to compute per-update deltas.
  void ResetTimers();

  // Updates the profiling data with the latest simulation data.
  void Update(const mjModel* model, const mjData* data);

  // Displays the profiling data using ImPlot.
  void CpuTimeGraph(ImVec2 plot_size = ImVec2(-1, 0));
  void DimensionsGraph(ImVec2 plot_size = ImVec2(-1, 0));

 private:
  std::array<mjTimerStat, mjNTIMER> timer_prev_ = {};
  std::vector<float> cpu_total_;
  std::vector<float> cpu_collision_;
  std::vector<float> cpu_prepare_;
  std::vector<float> cpu_solve_;
  std::vector<float> cpu_other_;
  std::vector<float> dim_dof_;
  std::vector<float> dim_body_;
  std::vector<float> dim_constraint_;
  std::vector<float> dim_nnz_;
  std::vector<float> dim_contact_;
  std::vector<float> dim_iteration_;
};

}  // namespace mujoco::studio

#endif  // MUJOCO_STUDIO_SIM_SIM_PROFILER_H_
