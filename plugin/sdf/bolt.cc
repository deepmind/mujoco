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

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <optional>
#include <utility>

#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "sdf.h"
#include "bolt.h"

namespace mujoco::plugin::sdf {
namespace {

static mjtNum distance(const mjtNum p[3], const mjtNum attributes[1]) {
  // see https://www.shadertoy.com/view/XtffzX
  mjtNum screw = 12;
  mjtNum radius = mju_sqrt(p[0] * p[0] + p[1] * p[1]) - attributes[0];
  mjtNum sqrt12 = mju_sqrt(2.) / 2.;

  // a triangle wave spun around Oy, offset by the angle between x and z
  mjtNum azimuth = mju_atan2(p[1], p[0]);
  mjtNum triangle = abs(Fract(p[2] * screw - azimuth / mjPI / 2.) - .5);
  mjtNum thread = (radius - triangle / screw) * sqrt12;

  // clip the top and bottom
  mjtNum bolt = Subtraction(thread, .5 - abs(p[2] + .5));
  mjtNum cone = (p[2] - radius) * sqrt12;

  // add a diagonal clipping for more realism
  bolt = Subtraction(bolt, cone + 1. * sqrt12);

  // create the hexagonal geometry for the head
  mjtNum point2D[2] = {p[0], p[1]};
  mjtNum res[2];
  mjtNum k = 6. / mjPI / 2.;
  mjtNum angle = -floor((mju_atan2(point2D[1], point2D[0])) * k + .5) / k;
  mjtNum s[2] = {mju_sin(angle), mju_sin(angle + mjPI * .5)};
  mjtNum mat[4] = {s[1], -s[0], s[0], s[1]};
  mju_mulMatVec(res, mat, point2D, 2, 2);
  mjtNum point3D[3] = {res[0], res[1], p[2]};
  mjtNum head = point3D[0] - .5;

  // the top is also rounded down with a cone
  head = Intersection(head, abs(point3D[2] + .25) - .25);
  head = Intersection(head, (point3D[2] + radius - .22) * sqrt12);
  return Union(bolt, head);
}

}  // namespace

// factory function
std::optional<Bolt> Bolt::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr("radius", m, instance)) {
    return Bolt(m, d, instance);
  } else {
    mju_warning("Invalid parameter specification in Bolt plugin");
    return std::nullopt;
  }
}

// plugin constructor
Bolt::Bolt(const mjModel* m, mjData* d, int instance) {
  SdfDefault<BoltAttribute> defattribute;

  for (int i = 0; i < BoltAttribute::nattribute; i++) {
    attribute[i] = defattribute.GetDefault(
        BoltAttribute::names[i],
        mj_getPluginConfig(m, instance, BoltAttribute::names[i]));
  }
}

// add new element in the vector storing iteration counts
void Bolt::Compute(const mjModel* m, mjData* d, int instance) {
  visualizer_.Next();
}

// reset visualization counter
void Bolt::Reset() { visualizer_.Reset(); }

// plugin visualization
void Bolt::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                     mjvScene* scn, int instance) {
  visualizer_.Visualize(m, d, opt, scn, instance);
}

// sdf
mjtNum Bolt::Distance(const mjtNum point[3]) const {
  return distance(point, attribute);
}

// gradient of sdf
void Bolt::Gradient(mjtNum grad[3], const mjtNum point[3]) const {
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
void Bolt::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sdf.bolt";
  plugin.capabilityflags |= mjPLUGIN_SDF;

  plugin.nattribute = BoltAttribute::nattribute;
  plugin.attributes = BoltAttribute::names;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto sdf_or_null = Bolt::Create(m, d, instance);
    if (!sdf_or_null.has_value()) {
      return -1;
    }
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(new Bolt(std::move(*sdf_or_null)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<Bolt*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto sdf = reinterpret_cast<Bolt*>(plugin_data);
    sdf->Reset();
  };
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* sdf = reinterpret_cast<Bolt*>(d->plugin_data[instance]);
    sdf->Visualize(m, d, opt, scn, instance);
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* sdf = reinterpret_cast<Bolt*>(d->plugin_data[instance]);
        sdf->Compute(m, d, instance);
      };
  plugin.sdf_distance =
      +[](const mjtNum point[3], const mjData* d, int instance) {
        auto* sdf = reinterpret_cast<Bolt*>(d->plugin_data[instance]);
        return sdf->Distance(point);
      };
  plugin.sdf_gradient = +[](mjtNum gradient[3], const mjtNum point[3],
                            const mjData* d, int instance) {
    auto* sdf = reinterpret_cast<Bolt*>(d->plugin_data[instance]);
    sdf->visualizer_.AddPoint(point);
    sdf->Gradient(gradient, point);
  };
  plugin.sdf_staticdistance =
      +[](const mjtNum point[3], const mjtNum* attributes) {
        return distance(point, attributes);
      };
  plugin.sdf_aabb = +[](mjtNum aabb[6], const mjtNum* attributes) {
    aabb[0] = aabb[1] = aabb[2] = 0;
    aabb[3] = aabb[4] = mju_max(0.6, attributes[0] * 1.2);
    aabb[5] = 1;
  };
  plugin.sdf_attribute =
      +[](mjtNum attribute[], const char* name[], const char* value[]) {
        SdfDefault<BoltAttribute> defattribute;
        defattribute.GetDefaults(attribute, name, value);
      };

  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sdf
