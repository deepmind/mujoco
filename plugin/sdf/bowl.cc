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
#include "bowl.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum point[3], const mjtNum attributes[3]) {
  mjtNum height = attributes[0];
  mjtNum radius = attributes[1];
  mjtNum thick = attributes[2];
  mjtNum width = mju_sqrt(mju_max(0.0, radius * radius - height * height));
  // see https://iquilezles.org/articles/distfunctions/
  mjtNum q[2] = {mju_norm(point, 2), point[2]};
  mjtNum qdiff[2] = {q[0] - width, q[1] - height};
  return ((height * q[0] < width * q[1]) ? mju_norm(qdiff, 2)
                                         : mju_abs(mju_norm(q, 2) - radius)) -
         thick;
}

}  // namespace

// factory function
std::optional<Bowl> Bowl::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("radius", m, instance) && CheckAttr("height", m, instance) &&
      CheckAttr("thickness", m, instance)) {
    SdfDefault<BowlAttribute> defattribute;
    mjtNum height = defattribute.GetDefault(
        BowlAttribute::names[0],
        mj_getPluginConfig(m, instance, BowlAttribute::names[0]));
    mjtNum radius = defattribute.GetDefault(
        BowlAttribute::names[1],
        mj_getPluginConfig(m, instance, BowlAttribute::names[1]));
    if (!(mju_abs(height) <= radius) || !(radius > 0)) {
      mju_warning("Invalid height or radius in Bowl plugin");
      return std::nullopt;
    }
    return Bowl(m, d, instance);
  } else {
    mju_warning("Invalid parameter specification in Bowl plugin");
    return std::nullopt;
  }
}

// plugin constructor
Bowl::Bowl(const mjModel* m, mjData* d, int instance) {
  SdfDefault<BowlAttribute> defattribute;

  for (int i = 0; i < BowlAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        BowlAttribute::names[i],
        mj_getPluginConfig(m, instance, BowlAttribute::names[i]));
  }

  mjtNum height = attribute[0];
  mjtNum radius = attribute[1];
  width = mju_sqrt(mju_max(0.0, radius * radius - height * height));
}

// add new element in the vector storing iteration counts
void Bowl::Compute(const mjModel* m, mjData* d, int instance) {
  visualizer_.Next();
}

// reset visualization counter
void Bowl::Reset() { visualizer_.Reset(); }

// plugin visualization
void Bowl::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                     mjvScene* scn, int instance) {
  visualizer_.Visualize(m, d, opt, scn, instance);
}

// sdf
mjtNum Bowl::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void Bowl::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
  // mjtNum q[2] = { mju_norm(point, 2), point[2] };
  // if (height*q[0] < width*q[1]) {
  //   mjtNum qdiff[2] = { q[0] - width, q[1] - height };
  //   mjtNum qdiffnorm = mju_norm(qdiff, 2);
  //   mjtNum grad_qdiff[3] = {qdiff[0] * point[0] / q[0],
  //                           qdiff[0] * point[1] / q[0],
  //                           qdiff[1]};
  //   grad[0] = - grad_qdiff[0] / qdiffnorm;
  //   grad[1] = - grad_qdiff[1] / qdiffnorm;
  //   grad[2] = - grad_qdiff[2] / qdiffnorm;
  // } else {
  //   mjtNum pnorm = mju_norm3(point);
  //   mjtNum grad_dist = (pnorm - radius) / mju_abs(pnorm - radius);
  //   grad[0] = - grad_dist * point[0] / pnorm;
  //   grad[1] = - grad_dist * point[1] / pnorm;
  //   grad[2] = - grad_dist * point[2] / pnorm;
  // }
  mjtNum eps = 1e-8;
  mjtNum dist0 = distance(point, attribute);

  mjtNum pointX[3] = {point[0] + eps, point[1], point[2]};
  mjtNum distX = distance(pointX, attribute);
  mjtNum pointY[3] = {point[0], point[1] + eps, point[2]};
  mjtNum distY = distance(pointY, attribute);
  mjtNum pointZ[3] = {point[0], point[1], point[2] + eps};
  mjtNum distZ = distance(pointZ, attribute);

  grad[0] = (distX - dist0) / eps;
  grad[1] = (distY - dist0) / eps;
  grad[2] = (distZ - dist0) / eps;
}

// plugin registration
void Bowl::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.bowl";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = BowlAttribute::nattribute;
  plugin.attributes = BowlAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Bowl::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(new Bowl(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Bowl*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto sdf = reinterpret_cast<Bowl*>(plugin_data);
    sdf->Reset();
  };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* sdf = reinterpret_cast<Bowl*>(d->plugin_data[instance]);
    sdf->Visualize(m, d, opt, scn, instance);
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* sdf = reinterpret_cast<Bowl*>(d->plugin_data[instance]);
        sdf->Compute(m, d, instance);
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<Bowl*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                            const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Bowl*>(d->plugin_data[instance]);
    sdf->visualizer_.AddPoint(point);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
    mjtNum radius = attributes[1];
    mjtNum thick = attributes[2];
    aabb[0] = aabb[1] = aabb[2] = 0;
    aabb[3] = aabb[4] = aabb[5] = radius + thick;
  };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<BowlAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
