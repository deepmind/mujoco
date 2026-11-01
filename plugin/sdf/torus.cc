// Copyright 2022 DeepMind Technologies Limited
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

#include <cstdint>
#include <optional>
#include <utility>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "torus.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum p[3], const mjtNum radius[2]) {
  mjtNum q = mju_sqrt(p[0] * p[0] + p[1] * p[1]) - radius[0];
  return mju_sqrt(q * q + p[2] * p[2]) - radius[1];
}

}  // namespace

// factory function
std::optional<Torus> Torus::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("radius1", m, instance) && CheckAttr("radius2", m, instance)) {
    return Torus(m, d, instance);
  } else {
    mju_warning("Invalid radius1 or radius2 parameters in Torus plugin");
    return std::nullopt;
  }
}

// plugin constructor
Torus::Torus(const mjModel* m, mjData* d, int instance) {
  SdfDefault<TorusAttribute> defattribute;

  for (int i = 0; i < TorusAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        TorusAttribute::names[i],
        mj_getPluginConfig(m, instance, TorusAttribute::names[i]));
  }
}

// sdf
mjtNum Torus::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void Torus::Gradient(mjtNum grad[3], const mjtNum p[3]) const {
  mjtNum len_xy = mju_sqrt(p[0] * p[0] + p[1] * p[1]);
  mjtNum q = len_xy - attribute[0];
  mjtNum grad_q[2] = {len_xy >= mjMINVAL ? p[0] / len_xy : 1,
                      len_xy >= mjMINVAL ? p[1] / len_xy : 0};
  mjtNum len_qz = mju_sqrt(q * q + p[2] * p[2]);
  grad[0] = q * grad_q[0] / mjMAX(len_qz, mjMINVAL);
  grad[1] = q * grad_q[1] / mjMAX(len_qz, mjMINVAL);
  grad[2] = p[2] / mjMAX(len_qz, mjMINVAL);
}

// plugin registration
void Torus::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.torus";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = TorusAttribute::nattribute;
  plugin.attributes = TorusAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Torus::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(new Torus(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Torus*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    // do nothing
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        // do nothing;
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<Torus*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                            const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Torus*>(d->plugin_data[instance]);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
    aabb[0] = aabb[1] = aabb[2] = 0;
    aabb[3] = aabb[4] = attributes[0] + attributes[1];
    aabb[5] = attributes[1];
  };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<TorusAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
