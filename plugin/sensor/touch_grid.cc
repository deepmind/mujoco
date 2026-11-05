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

#include "touch_grid.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>

namespace mujoco::plugin::sensor {

namespace {

// Checks that a plugin config attribute exists.
bool CheckAttr(const std::string& input) {
  char* end;
  std::string value = input;
  value.erase(std::remove_if(value.begin(), value.end(), isspace), value.end());
  strtod(value.c_str(), &end);
  return end == value.data() + value.size();
}

// Converts a string into a numeric vector
template <typename T>
void ReadVector(std::vector<T>& output, const std::string& input) {
  std::stringstream ss(input);
  std::string item;
  char delim = ' ';
  while (getline(ss, item, delim)) {
    CheckAttr(item);
    output.push_back(strtod(item.c_str(), nullptr));
  }
}

// Returns the index of the first value in `a` that x is less than or n if no
// such value exists. See: https://stackoverflow.com/a/39100135.
int LowerBound(const mjtNum a[], int n, mjtNum x) {
  int l = 0;
  int h = n;
  while (l < h) {
    int mid = (l + h) / 2;
    if (x <= a[mid]) {
      h = mid;
    } else {
      l = mid + 1;
    }
  }
  return l;
}

// Two dimensional histogram function.
void Histogram2D(const mjtNum x_data[], const mjtNum y_data[],
                 const mjtNum weights[], int n_data, const mjtNum x_edges[],
                 int n_x_edges, const mjtNum y_edges[], int n_y_edges,
                 mjtNum* histogram, int* counts) {
  for (int i = 0; i < n_data; ++i) {
    mjtNum x = x_data[i];
    mjtNum y = y_data[i];
    int x_idx = LowerBound(x_edges, n_x_edges, x);
    if (x_idx == 0 || x_idx == n_x_edges) {
      continue;
    }
    int y_idx = LowerBound(y_edges, n_y_edges, y);
    if (y_idx == 0 || y_idx == n_y_edges) {
      continue;
    }
    int index = (y_idx - 1) * (n_x_edges - 1) + (x_idx - 1);
    histogram[index] += weights[i];
    if (counts) {
      counts[index]++;
    }
  }
}

// Evenly spaced numbers over a specified interval.
void LinSpace(mjtNum lower, mjtNum upper, int n, mjtNum array[]) {
  mjtNum increment = n > 1 ? (upper - lower) / (n - 1) : 0;
  for (int i = 0; i < n; ++i) {
    *array = lower;
    ++array;
    lower += increment;
  }
}

// Parametrized linear/quintic interpolated nonlinearity.
mjtNum Fovea(mjtNum x, mjtNum gamma) {
  // Quick return.
  if (!gamma) return x;

  // Foveal deformation.
  mjtNum g = mjMAX(0, mjMIN(1, gamma));
  return g * mju_pow(x, 5) + (1 - g) * x;
}

// Make bin edges.
void BinEdges(mjtNum* x_edges, mjtNum* y_edges, int size[2], mjtNum fov[2],
              mjtNum gamma) {
  // Make unit bin edges.
  LinSpace(-1, 1, size[0] + 1, x_edges);
  LinSpace(-1, 1, size[1] + 1, y_edges);

  // Apply foveal deformation.
  for (int i = 0; i < size[0] + 1; i++) {
    x_edges[i] = Fovea(x_edges[i], gamma);
  }
  for (int i = 0; i < size[1] + 1; i++) {
    y_edges[i] = Fovea(y_edges[i], gamma);
  }

  // Scale by field-of-view.
  mju_scl(x_edges, x_edges, fov[0] * mjPI / 180, size[0] + 1);
  mju_scl(y_edges, y_edges, fov[1] * mjPI / 180, size[1] + 1);
}

// Permute 3-vector from 0,1,2 to 2,0,1.
static void xyz2zxy(mjtNum* x) {
  mjtNum z = x[2];
  x[2] = x[1];
  x[1] = x[0];
  x[0] = z;
}

// In the functions below transforming Cartesian <-> spherical:
//  - The frame points down the z-axis, so a=e=0 corresponds to (0, 0, -r).
//  - azimuth (a) corresponds to positive rotation around -y (towards +x).
//  - elevation (e) corresponds to positive rotation around +x (towards +y).

// Transform Cartesian (x,y,z) to spherical (azimuth, elevation, radius).
void CartesianToSpherical(const mjtNum xyz[3], mjtNum aer[3]) {
  mjtNum x = xyz[0], y = xyz[1], z = xyz[2];
  aer[0] = mju_atan2(x, -z);
  aer[1] = mju_atan2(y, mju_sqrt(x * x + z * z));
  aer[2] = mju_sqrt(x * x + z * z + y * y);
}

// Transform spherical (azimuth, elevation, radius) to Cartesian (x,y,z).
void SphericalToCartesian(const mjtNum aer[3], mjtNum xyz[3]) {
  mjtNum a = aer[0], e = aer[1], r = aer[2];
  xyz[0] = r * mju_cos(e) * mju_sin(a);
  xyz[1] = r * mju_sin(e);
  xyz[2] = -r * mju_cos(e) * mju_cos(a);
}

}  // namespace

// Creates a TouchGrid instance if all config attributes are defined and
// within their allowed bounds.
TouchGrid* TouchGrid::Create(const mjModel* m, mjData* d, int instance) {
  if (CheckAttr(std::string(mj_getPluginConfig(m, instance, "gamma"))) &&
      CheckAttr(std::string(mj_getPluginConfig(m, instance, "nchannel")))) {
    // nchannel
    double nchannel_d =
        strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
    if (!nchannel_d) nchannel_d = 1;
    if (!(nchannel_d >= 1 && nchannel_d <= 6)) {
      mju_warning("nchannel must be between 1 and 6");
      return nullptr;
    }
    int nchannel = static_cast<int>(nchannel_d);

    // size
    std::vector<mjtNum> size_num;
    std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
    ReadVector(size_num, size_str.c_str());
    if (size_num.size() != 2) {
      mju_warning("Both horizontal and vertical resolutions must be specified");
      return nullptr;
    }
    if (!(size_num[0] >= 1 && size_num[1] >= 1)) {
      mju_warning("Horizontal and vertical resolutions must be positive");
      return nullptr;
    }
    constexpr int kMaxInt = std::numeric_limits<int>::max();
    if (!(size_num[0] <= kMaxInt && size_num[1] <= kMaxInt)) {
      mju_warning("Horizontal and vertical resolutions are too large");
      return nullptr;
    }
    int size[2] = {static_cast<int>(size_num[0]),
                   static_cast<int>(size_num[1])};
    if (size[0] > kMaxInt / size[1] || size[0] * size[1] > kMaxInt / nchannel) {
      mju_warning("Horizontal and vertical resolutions are too large");
      return nullptr;
    }

    // field of view
    std::vector<mjtNum> fov;
    std::string fov_str = std::string(mj_getPluginConfig(m, instance, "fov"));
    ReadVector(fov, fov_str.c_str());
    if (fov.size() != 2) {
      mju_warning(
          "Both horizontal and vertical fields of view must be specified");
      return nullptr;
    }
    if (fov[0] <= 0 || fov[0] > 180) {
      mju_warning("`fov[0]` must be a float between (0, 180] degrees");
      return nullptr;
    }
    if (fov[1] <= 0 || fov[1] > 90) {
      mju_warning("`fov[1]` must be a float between (0, 90] degrees");
      return nullptr;
    }

    // gamma
    mjtNum gamma = strtod(mj_getPluginConfig(m, instance, "gamma"), nullptr);
    if (gamma < 0 || gamma > 1) {
      mju_warning("`gamma` must be a nonnegative float between [0, 1]");
      return nullptr;
    }

    // Make sure sensor is attached to a site.
    for (int i = 0; i < m->nsensor; ++i) {
      if (m->sensor_type[i] == mjSENS_PLUGIN &&
          m->sensor_plugin[i] == instance &&
          m->sensor_objtype[i] != mjOBJ_SITE) {
        mju_warning("Touch Grid sensor must be attached to a site");
        return nullptr;
      }
    }

    return new TouchGrid(m, d, instance, nchannel, size, fov.data(), gamma);
  } else {
    mju_warning("Invalid or missing parameters in touch_grid sensor plugin");
    return nullptr;
  }
}

TouchGrid::TouchGrid(const mjModel* m, mjData* d, int instance, int nchannel,
                     int size[2], mjtNum fov[2], mjtNum gamma)
    : nchannel_(nchannel),
      size_{size[0], size[1]},
      fov_{fov[0], fov[1]},
      gamma_(gamma) {
  // Allocate distance array.
  distance_.resize(size[0] * size[1], 0);
}

void TouchGrid::Reset(const mjModel* m, int instance) {}

void TouchGrid::Compute(const mjModel* m, mjData* d, int instance) {
  mj_markStack(d);

  // Get sensor id.
  int id;
  for (id = 0; id < m->nsensor; ++id) {
    if (m->sensor_type[id] == mjSENS_PLUGIN &&
        m->sensor_plugin[id] == instance) {
      break;
    }
  }

  // Clear sensordata and distance matrix.
  mjtNum* sensordata = d->sensordata + m->sensor_adr[id];
  mju_zero(sensordata, m->sensor_dim[id]);
  int frame = size_[0] * size_[1];
  mju_zero(distance_.data(), frame);

  // Get site id.
  int site_id = m->sensor_objid[id];

  // Count contacts.
  int ncon = 0;
  int parent_body = m->body_weldid[m->site_bodyid[site_id]];
  int parent_weld = m->body_weldid[parent_body];
  for (int i = 0; i < d->ncon; i++) {
    // skip flex contacts (no geom)
    if (d->contact[i].geom1 < 0 || d->contact[i].geom2 < 0) continue;
    int body1 = m->body_weldid[m->geom_bodyid[d->contact[i].geom1]];
    int body2 = m->body_weldid[m->geom_bodyid[d->contact[i].geom2]];
    if (body1 == parent_weld || body2 == parent_weld) {
      ncon++;
    }
  }

  // No contacts, return.
  if (!ncon) {
    mj_freeStack(d);
    return;
  }

  // Get site frame.
  mjtNum* site_pos = d->site_xpos + 3 * site_id;
  mjtNum* site_mat = d->site_xmat + 9 * site_id;

  // allocate contact forces and positions
  mjtNum* forces = mj_stackAllocNum(d, ncon * 6);
  mjtNum* positions = mj_stackAllocNum(d, ncon * 3);

  // Get forces and positions in spherical coordinates.
  int contact = 0;
  for (int i = 0; i < d->ncon; i++) {
    if (d->contact[i].geom1 < 0 || d->contact[i].geom2 < 0) continue;
    int body1 = m->geom_bodyid[d->contact[i].geom1];
    int weld1 = m->body_weldid[body1];
    int body2 = m->geom_bodyid[d->contact[i].geom2];
    int weld2 = m->body_weldid[body2];

    if (weld1 == parent_weld || weld2 == parent_weld) {
      // Get contact force/torque, rotate into world frame, then site frame.
      // Note that contact.frame is column major.
      mjtNum tmp_force[6], tmp1[3];
      mj_contactForce(m, d, i, tmp_force);
      mju_mulMatTVec3(tmp1, d->contact[i].frame, tmp_force);
      mju_mulMatTVec3(forces + 6 * contact, site_mat, tmp1);
      mju_mulMatTVec3(tmp1, d->contact[i].frame, tmp_force + 3);
      mju_mulMatTVec3(forces + 6 * contact + 3, site_mat, tmp1);

      // Contact force points from geom1 (weld1) to geom2 (weld2), so flip sign
      // if the sensorized weld body is weld1.
      if (weld1 == parent_weld) {
        mju_scl(forces + 6 * contact, forces + 6 * contact, -1, 6);
      }

      // Permute forces from x,y,z to z,x,y (normal, tangent, tangent)
      xyz2zxy(forces + 6 * contact);
      xyz2zxy(forces + 6 * contact + 3);

      // Get position, rotate into contact frame.
      mjtNum tmp2[3];
      mju_sub3(tmp1, d->contact[i].pos, site_pos);
      mju_mulMatTVec3(tmp2, site_mat, tmp1);

      // Transform to spherical coordinates, copy into positions array.
      CartesianToSpherical(tmp2, tmp1);
      for (int k = 0; k < 3; k++) {
        positions[k * ncon + contact] = tmp1[k];
      }
      contact++;
    }
  }

  // Transpose forces.
  mjtNum* forcesT = mj_stackAllocNum(d, ncon * 6);
  mju_transpose(forcesT, forces, ncon, 6);

  // Allocate bin edges.
  mjtNum* x_edges = mj_stackAllocNum(d, size_[0] + 1);
  mjtNum* y_edges = mj_stackAllocNum(d, size_[1] + 1);

  // Make bin edges.
  BinEdges(x_edges, y_edges, size_, fov_, gamma_);

  // Compute sensor output.
  for (int i = 0; i < nchannel_; i++) {
    if (!mju_isZero(forcesT + i * ncon, ncon)) {
      Histogram2D(positions, positions + ncon, forcesT + i * ncon, ncon,
                  x_edges, size_[0] + 1, y_edges, size_[1] + 1,
                  sensordata + i * frame, nullptr);
    }
  }

  // Allocate count matrix.
  int* counts = mj_stackAllocInt(d, frame);
  for (int i = 0; i < frame; i++) counts[i] = 0;

  // Compute distance matrix (unnormalized).
  Histogram2D(positions, positions + ncon, positions + 2 * ncon, ncon, x_edges,
              size_[0] + 1, y_edges, size_[1] + 1, distance_.data(), counts);

  // Normalize distances
  for (int i = 0; i < frame; i++) {
    if (counts[i]) {
      distance_.data()[i] /= counts[i];
    }
  }

  mj_freeStack(d);
}

// Thickness of taxel-visualization boxes relative to contact distance.
static const mjtNum kRelativeThickness = 0.02;

void TouchGrid::Visualize(const mjModel* m, mjData* d, const mjvOption* opt,
                          mjvScene* scn, int instance) {
  mj_markStack(d);

  // Get sensor id.
  int id;
  for (id = 0; id < m->nsensor; ++id) {
    if (m->sensor_type[id] == mjSENS_PLUGIN &&
        m->sensor_plugin[id] == instance) {
      break;
    }
  }

  // Get sensor data.
  mjtNum* sensordata = d->sensordata + m->sensor_adr[id];

  // Get maximum absolute normal force.
  mjtNum maxval = 0;
  int frame = size_[0] * size_[1];
  for (int j = 0; j < frame; j++) {
    maxval = mju_max(maxval, mju_abs(sensordata[j]));
  }

  // If no normal force readings, quick return.
  if (!maxval) {
    mj_freeStack(d);
    return;
  }

  // Get site id and frame.
  int site_id = m->sensor_objid[id];
  mjtNum* site_pos = d->site_xpos + 3 * site_id;
  mjtNum* site_mat = d->site_xmat + 9 * site_id;
  mjtNum site_quat[4];
  mju_mat2Quat(site_quat, site_mat);

  // Allocate bin edges.
  mjtNum* x_edges = mj_stackAllocNum(d, size_[0] + 1);
  mjtNum* y_edges = mj_stackAllocNum(d, size_[1] + 1);

  // Make bin edges.
  BinEdges(x_edges, y_edges, size_, fov_, gamma_);

  // Draw geoms.
  for (int i = 0; i < size_[0]; i++) {
    for (int j = 0; j < size_[1]; j++) {
      mjtNum dist = distance_.data()[j * size_[0] + i];
      if (!dist) {
        continue;
      }
      if (scn->ngeom >= scn->maxgeom) {
        if (!scn->status) {
          mju_warning(
              "Pre-allocated visual geom buffer is full. "
              "Increase maxgeom above %d.",
              scn->maxgeom);
          scn->status = 1;
        }
        mj_freeStack(d);
        return;
      } else {
        // size
        mjtNum size[3];
        size[0] = dist * 0.5 * (x_edges[i + 1] - x_edges[i]);
        size[1] = dist * 0.5 * (y_edges[j + 1] - y_edges[j]);
        size[2] = dist * kRelativeThickness;

        // position
        mjtNum pos[3];
        mjtNum aer[3];
        aer[0] = 0.5 * (x_edges[i + 1] + x_edges[i]);
        aer[1] = 0.5 * (y_edges[j + 1] + y_edges[j]);
        aer[2] = dist * (1 - kRelativeThickness);
        SphericalToCartesian(aer, pos);
        mju_mulMatVec3(pos, site_mat, pos);
        mju_addTo3(pos, site_pos);

        // orientation
        mjtNum a_quat[4];
        mjtNum site_y[3] = {-site_mat[1], -site_mat[4], -site_mat[7]};
        mju_axisAngle2Quat(a_quat, site_y, aer[0]);
        mjtNum e_quat[4];
        mjtNum site_x[3] = {site_mat[0], site_mat[3], site_mat[6]};
        mju_axisAngle2Quat(e_quat, site_x, aer[1]);
        mjtNum quat[4];
        mju_mulQuat(quat, e_quat, site_quat);
        mju_mulQuat(quat, a_quat, quat);
        mjtNum mat[9];
        mju_quat2Mat(mat, quat);

        // color
        float rgba[4] = {1, 1, 1, 1.0};
        for (int k = 0; k < mjMIN(nchannel_, 3); k++) {
          rgba[k] = mju_abs(sensordata[k * frame + j * size_[0] + i]) / maxval;
        }

        // draw box geom
        mjvGeom* thisgeom = scn->geoms + scn->ngeom;
        mjv_initGeom(thisgeom, mjGEOM_BOX, size, pos, mat, rgba);
        thisgeom->objtype = mjOBJ_UNKNOWN;
        thisgeom->objid = id;
        thisgeom->category = mjCAT_DECOR;
        thisgeom->segid = scn->ngeom;
        scn->ngeom++;
      }
    }
  }

  mj_freeStack(d);
}


void TouchGrid::RegisterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);

  plugin.name = "mujoco.sensor.touch_grid";
  plugin.capabilityflags |= mjPLUGIN_SENSOR;

  // Parameterized by 4 attributes.
  const char* attributes[] = {"nchannel", "size", "fov", "gamma"};
  plugin.nattribute = sizeof(attributes) / sizeof(attributes[0]);
  plugin.attributes = attributes;

  // Stateless.
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };

  // Sensor dimension = nchannel * size[0] * size[1]
  plugin.nsensordata = +[](const mjModel* m, int instance, int sensor_id) {
    double nchannel_d =
        strtod(mj_getPluginConfig(m, instance, "nchannel"), nullptr);
    if (!nchannel_d) nchannel_d = 1;
    if (!(nchannel_d >= 1 && nchannel_d <= 6)) {
      return 0;
    }
    int nchannel = static_cast<int>(nchannel_d);
    std::vector<mjtNum> size;
    std::string size_str = std::string(mj_getPluginConfig(m, instance, "size"));
    ReadVector(size, size_str.c_str());
    constexpr int kMaxInt = std::numeric_limits<int>::max();
    if (size.size() != 2 || !(size[0] >= 1 && size[1] >= 1 &&
                              size[0] <= kMaxInt && size[1] <= kMaxInt)) {
      return 0;
    }
    int sx = static_cast<int>(size[0]);
    int sy = static_cast<int>(size[1]);
    if (sx > kMaxInt / sy || sx * sy > kMaxInt / nchannel) {
      return 0;
    }
    return nchannel * sx * sy;
  };

  // Can only run after forces have been computed.
  plugin.needstage = mjSTAGE_ACC;

  // Initialization callback.
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    auto* TouchGrid = TouchGrid::Create(m, d, instance);
    if (!TouchGrid) {
      return -1;
    }
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(TouchGrid);
    return 0;
  };

  // Destruction callback.
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<TouchGrid*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };

  // Reset callback.
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {
    auto* TouchGrid = reinterpret_cast<class TouchGrid*>(plugin_data);
    TouchGrid->Reset(m, instance);
  };

  // Compute callback.
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {
        auto* TouchGrid =
            reinterpret_cast<class TouchGrid*>(d->plugin_data[instance]);
        TouchGrid->Compute(m, d, instance);
      };

  // Visualization callback.
  plugin.visualize = +[](const mjModel* m, mjData* d, const mjvOption* opt,
                         mjvScene* scn, int instance) {
    auto* TouchGrid =
        reinterpret_cast<class TouchGrid*>(d->plugin_data[instance]);
    TouchGrid->Visualize(m, d, opt, scn, instance);
  };

  // Register the plugin.
  mjp_registerPlugin(&plugin);
}

}  // namespace mujoco::plugin::sensor
