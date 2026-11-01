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

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <mujoco/mujoco.h>
#include "sdf.h"

namespace mujoco::plugin::sdf {

bool CheckAttr(const char* name, const mjModel* m, int instance) {
  // empty or whitespace is valid (default), otherwise a number with optional
  // surrounding whitespace, as in SdfDefault::GetDefault
  const char* ws = " \t\n\v\f\r";
  const char* value = mj_getPluginConfig(m, instance, name);
  if (!value[std::strspn(value, ws)]) return true;
  char* end;
  strtod(value, &end);
  return end != value && !end[std::strspn(end, ws)];
}

SdfVisualizer::SdfVisualizer() {
  points_.assign(10 * (mjMAXCONPAIR + 1) * (mjMAXCONPAIR + 1) * 3, 0);
  npoints_.clear();
}

void SdfVisualizer::AddPoint(const mjtNum point[3]) {
  if (!npoints_.empty() && npoints_.back() < points_.size() / 3) {
    points_[3 * npoints_.back() + 0] = point[0];
    points_[3 * npoints_.back() + 1] = point[1];
    points_[3 * npoints_.back() + 2] = point[2];
    npoints_.back()++;
  }
}

void SdfVisualizer::Next() {
  npoints_.push_back(npoints_.empty() ? 0 : npoints_.back());
}

void SdfVisualizer::Reset() { npoints_.clear(); }

void SdfVisualizer::Visualize(const mjModel* m, const mjData* d,
                              const mjvOption* opt, mjvScene* scn,
                              int instance) {
  if (!opt->flags[mjVIS_SDFITER]) {
    return;
  }
  if (npoints_.empty()) {
    return;
  }
  int tot = 0, n = 0, g = 0;
  for (int i = 0; i < m->ngeom; i++) {
    if (m->geom_plugin[i] == instance) {
      g = i;
      break;
    }
  }
  mjtNum* points = points_.data();
  int* npoints = npoints_.data();
  int niter = npoints_.size();
  mjtNum geom_mat[9], offset[3], rotation[9], from[3], to[3];
  mjtNum* geom_xpos = d->geom_xpos + 3 * g;
  mjtNum* geom_xmat = d->geom_xmat + 9 * g;
  mjtNum* geom_pos = m->geom_pos + 3 * g;
  mjtNum* geom_quat = m->geom_quat + 4 * g;
  mju_quat2Mat(geom_mat, geom_quat);
  mju_mulMatMatT(rotation, geom_xmat, geom_mat, 3, 3, 3);
  mju_mulMatVec3(offset, rotation, geom_pos);
  mju_sub3(offset, geom_xpos, offset);

  for (int i = 0; i < niter; i++) {
    n = npoints[i] - tot;
    if (!n) {
      continue;
    }
    for (int k = 0; k < 2; k++) {
      for (int j = 0; j < (k == 0 ? 2 : n - 1); j++) {
        if (scn->ngeom >= scn->maxgeom) {
          if (!scn->status) {
            mju_warning(
                "Pre-allocated visual geom buffer is full. "
                "Increase maxgeom above %d.",
                scn->maxgeom);
            scn->status = 1;
          }
          return;
        }
        mjvGeom* thisgeom = scn->geoms + scn->ngeom;
        mjtNum* p1 = points + (tot + (k == 0 ? (n - 1) * j : j)) * 3;
        mju_mulMatVec3(from, rotation, p1);
        mju_addTo3(from, offset);
        if (k == 0) {
          float rgba[4] = {static_cast<float>(j > 0), 0,
                           static_cast<float>(j == 0), 1};
          mjtNum size = 0.2 * m->stat.meansize;
          mjv_initGeom(thisgeom, mjGEOM_SPHERE, &size, from, geom_xmat, rgba);
        } else {
          mjtNum* p2 = points + (tot + j + 1) * 3;
          mju_mulMatVec3(to, rotation, p2);
          mju_addTo3(to, offset);
          mjv_initGeom(thisgeom, mjGEOM_NONE, NULL, NULL, NULL, NULL);
          thisgeom->objtype = mjOBJ_UNKNOWN;
          thisgeom->objid = i;
          thisgeom->category = mjCAT_DECOR;
          thisgeom->segid = scn->ngeom;
          to[0] = from[0] + .95 * (to[0] - from[0]);
          to[1] = from[1] + .95 * (to[1] - from[1]);
          to[2] = from[2] + .95 * (to[2] - from[2]);
          mjv_connector(thisgeom, mjGEOM_LINE, 2, from, to);
          thisgeom->rgba[0] = (j + 1.) / (n - 1.);
          thisgeom->rgba[1] = 0;
          thisgeom->rgba[2] = 1 - (j + 1.) / (n - 1.);
          thisgeom->rgba[3] = 1;
        }
        scn->ngeom++;
      }
    }
    tot += n;
  }
}

}  // namespace mujoco::plugin::sdf
