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

#include "studio/sim/sim_profiler.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <imgui.h>
#include <implot.h>
#include <mujoco/mujoco.h>
#include "studio/ux/imgui_widgets.h"

namespace mujoco::studio {

SimProfiler::SimProfiler() { Clear(); }

void SimProfiler::ResetTimers() { timer_prev_ = {}; }

void SimProfiler::Clear() {
  constexpr int kProfilerMaxFrames = 200;
  ResetTimers();
  cpu_total_.clear();
  cpu_collision_.clear();
  cpu_prepare_.clear();
  cpu_solve_.clear();
  cpu_other_.clear();
  dim_dof_.clear();
  dim_body_.clear();
  dim_constraint_.clear();
  dim_nnz_.clear();
  dim_contact_.clear();
  dim_iteration_.clear();

  cpu_total_.resize(kProfilerMaxFrames, 0);
  cpu_collision_.resize(kProfilerMaxFrames, 0);
  cpu_prepare_.resize(kProfilerMaxFrames, 0);
  cpu_solve_.resize(kProfilerMaxFrames, 0);
  cpu_other_.resize(kProfilerMaxFrames, 0);
  dim_dof_.resize(kProfilerMaxFrames, 0);
  dim_body_.resize(kProfilerMaxFrames, 0);
  dim_constraint_.resize(kProfilerMaxFrames, 0);
  dim_nnz_.resize(kProfilerMaxFrames, 0);
  dim_contact_.resize(kProfilerMaxFrames, 0);
  dim_iteration_.resize(kProfilerMaxFrames, 0);
}

void SimProfiler::Update(const mjModel* model, const mjData* data) {
  // CPU timers, relative to the previous update.
  const mjTimerStat* timer = data->timer;
  if (timer[mjTIMER_STEP].number < timer_prev_[mjTIMER_STEP].number ||
      timer[mjTIMER_FORWARD].number < timer_prev_[mjTIMER_FORWARD].number) {
    timer_prev_ = {};
  }
  const std::array<mjTimerStat, mjNTIMER> prev = timer_prev_;
  std::copy_n(timer, mjNTIMER, timer_prev_.begin());

  mjtNum total = timer[mjTIMER_STEP].duration - prev[mjTIMER_STEP].duration;
  mjtNum number = timer[mjTIMER_STEP].number - prev[mjTIMER_STEP].number;
  if (number == 0.0) {
    total = timer[mjTIMER_FORWARD].duration - prev[mjTIMER_FORWARD].duration;
    number = timer[mjTIMER_FORWARD].number - prev[mjTIMER_FORWARD].number;
  }
  if (number == 0.0) {
    // This can happen if the simulation is paused.
    return;
  }

  mjtNum avg_total = 1000.0 * total / number;
  cpu_total_.erase(cpu_total_.begin());
  cpu_total_.push_back(avg_total);

  auto get_duration = [&](int type) { return timer[type].duration - prev[type].duration; };

  mjtNum collision = 1000.0 * get_duration(mjTIMER_POS_COLLISION) / number;
  cpu_collision_.erase(cpu_collision_.begin());
  cpu_collision_.push_back(collision);

  mjtNum prepare = 1000.0 * ((get_duration(mjTIMER_POS_MAKE) / number) +
                             (get_duration(mjTIMER_POS_PROJECT) / number));
  cpu_prepare_.erase(cpu_prepare_.begin());
  cpu_prepare_.push_back(prepare);

  mjtNum solve = 1000.0 * get_duration(mjTIMER_CONSTRAINT) / number;
  cpu_solve_.erase(cpu_solve_.begin());
  cpu_solve_.push_back(solve);

  mjtNum other = avg_total - collision - prepare - solve;
  cpu_other_.erase(cpu_other_.begin());
  cpu_other_.push_back(other);

  // Solver diagnostics.
  mjtNum nnz = 0;
  int solver_niter = 0;
  const int nisland = data->nefc ? mjMAX(1, mjMIN(data->nisland, mjNISLAND)) : 0;
  for (int island = 0; island < nisland; island++) {
    nnz += data->solver_nnz[island];
    solver_niter += data->solver_niter[island];
  }

  dim_dof_.erase(dim_dof_.begin());
  int nv = (model->opt.enableflags & mjENBL_SLEEP) ? data->nv_awake : model->nv;
  dim_dof_.push_back(nv);

  dim_body_.erase(dim_body_.begin());
  int nbody = (model->opt.enableflags & mjENBL_SLEEP) ? data->nbody_awake : model->nbody;
  dim_body_.push_back(nbody);

  dim_constraint_.erase(dim_constraint_.begin());
  dim_constraint_.push_back(data->nefc);

  dim_nnz_.erase(dim_nnz_.begin());
  dim_nnz_.push_back(nnz);

  dim_contact_.erase(dim_contact_.begin());
  dim_contact_.push_back(data->ncon);

  dim_iteration_.erase(dim_iteration_.begin());
  dim_iteration_.push_back(static_cast<float>(solver_niter) / mjMAX(1, nisland));
}

// Returns a dynamically padded legend label for ImPlot.
static std::string GetLegendLabel(const char* name, const std::vector<float>& values) {
  float sum = 0.f;
  int count = 0;
  for (float val : values) {
    if (val != 0.f) {
      sum += val;
      count++;
    }
  }
  float avg = count > 0 ? (sum / count) : 0.f;

  char buf[128];
  std::snprintf(buf, sizeof(buf), "%-9s (%4.0f)###%s", name, avg, name);
  return std::string(buf);
}

// Returns a dynamically padded legend label for the Dimensions plot.
static std::string GetDimensionLabel(const char* name, float value) {
  char buf[128];
  std::snprintf(buf, sizeof(buf), "%-10s: %5.0f###%s", name, value, name);
  return std::string(buf);
}

void SimProfiler::CpuTimeGraph(ImVec2 plot_size) {
  ScopedStyle style;
  style.Font(ScopedFont::kMono);

  ImPlotFlags flags = ImPlot_SetupPlotFlags(plot_size) | ImPlotFlags_NoMouseText;
  if (ImPlot::BeginPlot("CPU microseconds (avg) vs frame", plot_size, flags)) {
    ImPlot_SetupTimeAxis(plot_size, "");
    ImPlot_SetupValueAxis(plot_size, "", "%.0f");
    float max_val = 0.f;
    for (float val : cpu_total_) {
      if (val > max_val) {
        max_val = val;
      }
    }
    float rounded_max = std::max(100.0f, std::ceil(max_val / 100.0f) * 100.0f);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 0.0, rounded_max, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    ImPlot::SetupFinish();
    ImPlotSpec spec(ImPlotProp_LineWeight, 2.0f);

    ImPlot::PlotLine(GetLegendLabel("total", cpu_total_).c_str(), cpu_total_.data(),
                     cpu_total_.size(), 1, -(int)cpu_total_.size(), spec);
    ImPlot::PlotLine(GetLegendLabel("prepare", cpu_prepare_).c_str(), cpu_prepare_.data(),
                     cpu_prepare_.size(), 1, -(int)cpu_prepare_.size(), spec);
    ImPlot::PlotLine(GetLegendLabel("solve", cpu_solve_).c_str(), cpu_solve_.data(),
                     cpu_solve_.size(), 1, -(int)cpu_solve_.size(), spec);
    ImPlot::PlotLine(GetLegendLabel("collision", cpu_collision_).c_str(), cpu_collision_.data(),
                     cpu_collision_.size(), 1, -(int)cpu_collision_.size(), spec);
    ImPlot::PlotLine(GetLegendLabel("other", cpu_other_).c_str(), cpu_other_.data(),
                     cpu_other_.size(), 1, -(int)cpu_other_.size(), spec);
    ImPlot::EndPlot();
  }
}

void SimProfiler::DimensionsGraph(ImVec2 plot_size) {
  ScopedStyle style;
  style.Font(ScopedFont::kMono);

  ImPlotFlags flags = ImPlot_SetupPlotFlags(plot_size) | ImPlotFlags_NoMouseText;
  if (ImPlot::BeginPlot("Dimensions: current vs frame", plot_size, flags)) {
    ImPlot_SetupTimeAxis(plot_size, "");
    ImPlot_SetupValueAxis(plot_size, "", "%.0f");
    ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
    float max_val = 10.f;
    for (float val : dim_dof_) {
      max_val = std::max(max_val, val);
    }
    for (float val : dim_body_) {
      max_val = std::max(max_val, val);
    }
    for (float val : dim_constraint_) {
      max_val = std::max(max_val, val);
    }
    for (float val : dim_nnz_) {
      max_val = std::max(max_val, val);
    }
    for (float val : dim_contact_) {
      max_val = std::max(max_val, val);
    }
    for (float val : dim_iteration_) {
      max_val = std::max(max_val, val);
    }
    float log_max = std::log10(max_val);
    float y_max = std::pow(10.f, log_max + 0.1f);
    ImPlot::SetupAxisLimits(ImAxis_Y1, 1.0, y_max, ImPlotCond_Always);
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    ImPlot::SetupFinish();
    ImPlotSpec spec(ImPlotProp_LineWeight, 2.0f);

    const int count = static_cast<int>(dim_dof_.size());
    struct GetterData {
      const std::vector<float>* vec;
      int count;
    };
    auto getter = +[](int idx, void* user_data) -> ImPlotPoint {
      const auto* gd = static_cast<const GetterData*>(user_data);
      float val = (*gd->vec)[idx];
      double x = -gd->count + idx;
      double y = (val <= 0.0f) ? std::nan("") : static_cast<double>(val);
      return ImPlotPoint{x, y};
    };

    GetterData nnz_data{&dim_nnz_, count};
    ImPlot::PlotLineG(GetDimensionLabel("nnz", dim_nnz_.back()).c_str(), getter, &nnz_data, count,
                      spec);
    GetterData constraint_data{&dim_constraint_, count};
    ImPlot::PlotLineG(GetDimensionLabel("constraint", dim_constraint_.back()).c_str(), getter,
                      &constraint_data, count, spec);
    GetterData dof_data{&dim_dof_, count};
    ImPlot::PlotLineG(GetDimensionLabel("dof", dim_dof_.back()).c_str(), getter, &dof_data, count,
                      spec);
    GetterData contact_data{&dim_contact_, count};
    ImPlot::PlotLineG(GetDimensionLabel("contact", dim_contact_.back()).c_str(), getter,
                      &contact_data, count, spec);
    GetterData body_data{&dim_body_, count};
    ImPlot::PlotLineG(GetDimensionLabel("body", dim_body_.back()).c_str(), getter, &body_data,
                      count, spec);
    GetterData iteration_data{&dim_iteration_, count};
    ImPlot::PlotLineG(GetDimensionLabel("iteration", dim_iteration_.back()).c_str(), getter,
                      &iteration_data, count, spec);
    ImPlot::EndPlot();
  }
}

}  // namespace mujoco::studio
