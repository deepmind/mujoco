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

#include "pid.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>

namespace mujoco::plugin::actuator {
namespace {

constexpr char kAttrPGain[] = "kp";
constexpr char kAttrIGain[] = "ki";
constexpr char kAttrDGain[] = "kd";
constexpr char kAttrIMax[] = "imax";
constexpr char kAttrSlewMax[] = "slewmax";

std::optional<mjtNum> ReadOptionalDoubleAttr(const mjModel* m, int instance,
                                             const char* attr) {
  const char* value = mj_getPluginConfig(m, instance, attr);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::strtod(value, nullptr);
}

// returns the next act given the current act_dot, after clamping, for native
// mujoco dyntypes.
// copied from engine_forward.
mjtNum NextActivation(const mjModel* m, const mjData* d, int actuator_id,
                      int act_adr, mjtNum act_dot) {
  mjtNum act = d->act[act_adr];

  if (m->actuator_dyntype[actuator_id] == mjDYN_FILTEREXACT) {
    // exact filter integration
    // act_dot(0) = (ctrl-act(0)) / tau
    // act(h) = act(0) + (ctrl-act(0)) (1 - exp(-h / tau))
    //        = act(0) + act_dot(0) * tau * (1 - exp(-h / tau))
    mjtNum tau = mju_max(mjMINVAL, m->actuator_dynprm[actuator_id * mjNDYN]);
    act = act + act_dot * tau * (1 - mju_exp(-m->opt.timestep / tau));
  } else {
    // Euler integration
    act = act + act_dot * m->opt.timestep;
  }

  // clamp to actrange
  if (m->actuator_actlimited[actuator_id]) {
    mjtNum* actrange = m->actuator_actrange + 2 * actuator_id;
    act = mju_clip(act, actrange[0], actrange[1]);
  }

  return act;
}

bool HasSlew(const mjModel* m, int instance) {
  return ReadOptionalDoubleAttr(m, instance, kAttrSlewMax).has_value();
}

}  // namespace

PidConfig PidConfig::FromModel(const mjModel* m, int instance) {
  PidConfig config;
  config.p_gain = ReadOptionalDoubleAttr(m, instance, kAttrPGain).value_or(0);
  config.i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
  config.d_gain = ReadOptionalDoubleAttr(m, instance, kAttrDGain).value_or(0);

  // Clamps in the XML are specified in terms of maximum forces. Scale by i_gain
  // to get the limits on the value of the error integral.
  std::optional<double> i_clamp_max_force =
      ReadOptionalDoubleAttr(m, instance, kAttrIMax);
  if (i_clamp_max_force.has_value() && config.i_gain) {
    config.i_max = *i_clamp_max_force / config.i_gain;
  }

  config.slew_max = ReadOptionalDoubleAttr(m, instance, kAttrSlewMax);

  return config;
}

std::unique_ptr<Pid> Pid::Create(const mjModel* m, int instance) {
  PidConfig config = PidConfig::FromModel(m, instance);

  if (config.i_max.has_value() && *config.i_max < 0) {
    mju_warning("negative imax");
    return nullptr;
  }

  if (config.slew_max.value_or(0.0) < 0) {
    mju_warning("slewmax must be non-negative");
    return nullptr;
  }

  std::vector<int> actuators;
  for (int i = 0; i < m->nactuator; i++) {
    if (m->actuator_plugin[i] == instance) {
      actuators.push_back(i);
    }
  }
  if (actuators.empty()) {
    mju_warning("actuator not found for plugin instance %d", instance);
    return nullptr;
  }
  // Validate actnum values for all actuators:
  for (int actuator_id : actuators) {
    int actnum = m->actuator_actnum[actuator_id];
    int expected_actnum = Pid::ActDim(m, instance, actuator_id);
    int dyntype = m->actuator_dyntype[actuator_id];
    if (dyntype == mjDYN_FILTER || dyntype == mjDYN_FILTEREXACT ||
        dyntype == mjDYN_INTEGRATOR) {
      expected_actnum++;
    }
    if (actnum != expected_actnum) {
      mju_warning(
          "actuator %d has actdim %d, expected %d. Add actdim=\"%d\" to the "
          "actuator plugin element.",
          actuator_id, actnum, expected_actnum, expected_actnum);
      return nullptr;
    }
  }
  return std::unique_ptr<Pid>(new Pid(config, std::move(actuators)));
}

void Pid::Reset(mjtNum* plugin_state) {}

mjtNum Pid::GetCtrl(const mjModel* m, const mjData* d, int actuator_idx,
                    const State& state, bool actearly) const {
  mjtNum ctrl = 0;
  if (m->actuator_dyntype[actuator_idx] == mjDYN_NONE) {
    if (m->actuator_ctrlnum[actuator_idx] > 0) {
      int ctrladr = m->actuator_ctrladr[actuator_idx];
      ctrl = d->ctrl[ctrladr];
      // clamp ctrl
      if (m->actuator_ctrllimited[ctrladr]) {
        ctrl = mju_clip(ctrl, m->actuator_ctrlrange[2 * ctrladr],
                        m->actuator_ctrlrange[2 * ctrladr + 1]);
      }
    }
  } else {
    // Use of act instead of ctrl, to create integrated-velocity controllers or
    // to filter the controls.
    int actadr =
        m->actuator_actadr[actuator_idx] + m->actuator_actnum[actuator_idx] - 1;
    if (actearly) {
      ctrl = NextActivation(m, d, actuator_idx, actadr, d->act_dot[actadr]);
    } else {
      ctrl = d->act[actadr];
    }
  }
  if (config_.slew_max.has_value() && state.previous_ctrl_exists) {
    mjtNum ctrl_min = state.previous_ctrl - *config_.slew_max * m->opt.timestep;
    mjtNum ctrl_max = state.previous_ctrl + *config_.slew_max * m->opt.timestep;
    ctrl = mju_clip(ctrl, ctrl_min, ctrl_max);
  }
  return ctrl;
}

void Pid::ActDot(const mjModel* m, mjData* d, int instance) const {
  for (int actuator_idx : actuators_) {
    int outadr = m->actuator_outadr[actuator_idx];
    State state = GetState(m, d, actuator_idx);
    mjtNum ctrl = GetCtrl(m, d, actuator_idx, state, /*actearly=*/false);
    mjtNum error = ctrl - d->actuator_length[outadr];

    int state_idx = m->actuator_actadr[actuator_idx];
    if (config_.i_gain) {
      mjtNum integral = state.integral + error * m->opt.timestep;
      if (config_.i_max.has_value()) {
        integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
      }
      d->act_dot[state_idx] = (integral - d->act[state_idx]) / m->opt.timestep;
      ++state_idx;
    }
    if (config_.slew_max.has_value()) {
      d->act_dot[state_idx] = (ctrl - d->act[state_idx]) / m->opt.timestep;
      ++state_idx;
    }
  }
}

void Pid::Compute(const mjModel* m, mjData* d, int instance) {
  for (int i = 0; i < actuators_.size(); i++) {
    int actuator_idx = actuators_[i];
    int outadr = m->actuator_outadr[actuator_idx];
    State state = GetState(m, d, actuator_idx);
    mjtNum ctrl =
        GetCtrl(m, d, actuator_idx, state, m->actuator_actearly[actuator_idx]);

    mjtNum error = ctrl - d->actuator_length[outadr];

    mjtNum ctrl_dot = m->actuator_dyntype[actuator_idx] == mjDYN_NONE
                          ? 0
                          : d->act_dot[m->actuator_actadr[actuator_idx] +
                                       m->actuator_actnum[actuator_idx] - 1];
    mjtNum error_dot = ctrl_dot - d->actuator_velocity[outadr];

    mjtNum integral = 0;
    if (config_.i_gain) {
      integral = state.integral + error * m->opt.timestep;
      if (config_.i_max.has_value()) {
        integral = mju_clip(integral, -*config_.i_max, *config_.i_max);
      }
    }

    d->actuator_force[outadr] = config_.p_gain * error +
                                config_.d_gain * error_dot +
                                config_.i_gain * integral;
  }
}

void Pid::Advance(const mjModel* m, mjData* d, int instance) const {
  // act variables already updated by MuJoCo integrating act_dot
}

int Pid::StateSize(const mjModel* m, int instance) { return 0; }

int Pid::ActDim(const mjModel* m, int instance, int actuator_id) {
  double i_gain = ReadOptionalDoubleAttr(m, instance, kAttrIGain).value_or(0);
  return (i_gain ? 1 : 0) + (HasSlew(m, instance) ? 1 : 0);
}

Pid::State Pid::GetState(const mjModel* m, mjData* d, int actuator_idx) const {
  State state;
  int state_idx = m->actuator_actadr[actuator_idx];
  if (config_.i_gain) {
    state.integral = d->act[state_idx++];
  }
  if (config_.slew_max.has_value()) {
    state.previous_ctrl = d->act[state_idx++];
    state.previous_ctrl_exists = d->time > 0;
  }
  return state;
}

void Pid::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.pid";
  plugin.capabilityflags |= mjPLUGIN_ACTUATOR;

  std::vector<const char*> attributes = {kAttrPGain, kAttrIGain, kAttrDGain,
                                         kAttrIMax, kAttrSlewMax};
  plugin.nattribute = attributes.size();
  plugin.attributes = attributes.data();
  plugin.nstate = Pid::StateSize;

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    std::unique_ptr<Pid> pid = Pid::Create(m, instance);
    if (pid == nullptr) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(pid.release());
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Pid*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto* pid = reinterpret_cast<Pid*>(plugin_data);
    pid->Reset(plugin_state);
  };
  plugin.actuator_act_dot = +[](const mjModel* m, mjData* d, int instance) {
    auto* pid = reinterpret_cast<Pid*>(d->plugin_data[instance]);
    pid->ActDot(m, d, instance);
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* pid = reinterpret_cast<Pid*>(d->plugin_data[instance]);
        pid->Compute(m, d, instance);
      };
  plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    auto* pid = reinterpret_cast<Pid*>(d->plugin_data[instance]);
    pid->Advance(m, d, instance);
  };
  // TODO: b/303823996 - allow actuator plugins to compute their derivatives wrt
  // qvel, for implicit integration
  mjp_registerPlugin(&plugin);
}

Pid::Pid(PidConfig config, std::vector<int> actuators)
    : config_(std::move(config)), actuators_(std::move(actuators)) {}

}  // namespace mujoco::plugin::actuator
