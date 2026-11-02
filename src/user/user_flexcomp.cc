// Copyright 2021 DeepMind Technologies Limited
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mjtype.h>
#include "cc/array_safety.h"
#include "engine/engine_core_util.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "user/user_api.h"
#include "user/user_cache.h"
#include "user/user_flexcomp.h"
#include "user/user_model.h"
#include "user/user_objects.h"
#include "user/user_resource.h"
#include "user/user_util.h"

namespace {
namespace mju = ::mujoco::util;
using mujoco::user::PairHash;
using std::max;
using std::min;
using std::pow;
using std::stringstream;
using std::vector;
}  // namespace

// strncpy with 0, return false
static bool comperr(char* error, const char* msg, int error_sz) {
  mju_strncpy(error, msg, error_sz);
  return false;
}


// constructor: set defaults outside mjCDef
mjCFlexcomp::mjCFlexcomp(void) {
  type     = mjFCOMPTYPE_GRID;
  count[0] = count[1] = count[2] = 10;
  cellcount[0] = cellcount[1] = cellcount[2] = -1;
  mjuu_setvec(spacing, 0.02, 0.02, 0.02);
  mjuu_setvec(scale, 1, 1, 1);
  mjuu_setvec(origin, 0, 0, 0);
  mass       = 1;
  inertiabox = 0.005;
  equality   = 0;
  mjuu_setvec(pos, 0, 0, 0);
  mjuu_setvec(quat, 1, 0, 0, 0);
  frame        = nullptr;
  rigid        = false;
  centered     = false;
  doftype      = mjFCOMPDOF_FULL;
  has_dim      = false;
  has_origin   = false;
  has_flatskin = false;
  file_or_mesh = false;

  mjs_defaultPlugin(&plugin);
  mjs_defaultOrientation(&alt);
  plugin_name          = "";
  plugin_instance_name = "";
  plugin.plugin_name   = (mjString*)&plugin_name;
  plugin.name          = (mjString*)&plugin_instance_name;
}


// identify empty cells and pin nodes exclusively in empty cells
void mjCFlexcomp::MarkEmptyCells(
    mjCFlex* flex, const double* points, const double minmax[6], int nx, int ny, int nz) {
  // delegate cell_empty computation to mjCFlex
  int nelem = element.size() / (flex->spec.dim + 1);
  flex->ComputeCellEmpty(points, element.data(), nelem, flex->spec.dim, minmax);

  // pin nodes that belong exclusively to empty cells
  for (int gi = 0; gi < nx; gi++) {
    for (int gj = 0; gj < ny; gj++) {
      for (int gk = 0; gk < nz; gk++) {
        if (flex->NodeInEmptyCells(gi, gj, gk)) {
          int idx     = gi * ny * nz + gj * nz + gk;
          pinned[idx] = true;
        }
      }
    }
  }
}


// make flexcomp object
bool mjCFlexcomp::Make(mjsBody* body, char* error, int error_sz, const mjVFS* vfs) {
  // a mesh asset can only be used by mesh and gmsh flexcomps
  if (!mesh.empty() && type != mjFCOMPTYPE_MESH && type != mjFCOMPTYPE_GMSH) {
    return comperr(error, "mesh asset can only be used with flexcomp type mesh or gmsh", error_sz);
  }

  mjCModel*    model    = static_cast<mjCBody*>(body->element)->model;
  mjsCompiler* compiler = static_cast<mjCBody*>(body->element)->compiler;
  mjsFlex*     dflex    = def.spec.flex;
  bool         direct =
      (type == mjFCOMPTYPE_DIRECT || type == mjFCOMPTYPE_MESH || type == mjFCOMPTYPE_GMSH);

  // check parent body name
  if (mjs_getName(body->element)->empty()) {
    return comperr(error, "Parent body must have name", error_sz);
  }

  // check dim
  if (dflex->dim < 1 || dflex->dim > 3) {
    return comperr(error, "Invalid dim, must be between 1 and 3", error_sz);
  }

  // check counts
  for (int i = 0; i < 3; i++) {
    if (count[i] < 1 || ((doftype == mjFCOMPDOF_RADIAL && count[i] < 2) && dflex->dim == 3)) {
      return comperr(error, "Count too small", error_sz);
    }
  }

  // check spacing
  double minspace = 2 * dflex->radius + dflex->margin;
  if (!direct) {
    if (spacing[0] < minspace || spacing[1] < minspace || spacing[2] < minspace) {
      return comperr(error, "Spacing must be larger than geometry size", error_sz);
    }
  }

  // check scale
  if (scale[0] < mjMINVAL || scale[1] < mjMINVAL || scale[2] < mjMINVAL) {
    return comperr(error, "Scale must be larger than mjMINVAL", error_sz);
  }

  // check mass and inertia
  if (mass < mjMINVAL || inertiabox < mjMINVAL) {
    return comperr(error, "Mass and inertiabox must be larger than mjMINVAL", error_sz);
  }

  // check edge equality: 0=none, 1=edge, 3=strain (2=vert was removed)
  if (equality != 0 && equality != 1 && equality != 3) {
    return comperr(error,
                   "Invalid edge equality, must be 0 (none), 1 (edge) or 3 (strain)",
                   error_sz);
  }

  // compute orientation
  const char* alterr = mjs_resolveOrientation(quat, compiler->degree, compiler->eulerseq, &alt);
  if (alterr) { return comperr(error, alterr, error_sz); }

  // accumulate enclosing frame chain if present
  for (const mjCFrame* f = frame ? static_cast<const mjCFrame*>(frame->element) : nullptr; f;
       f                 = f->frame) {
    double framequat[4];
    mjuu_copyvec(framequat, f->spec.quat, 4);
    const char* ferr =
        mjs_resolveOrientation(framequat, compiler->degree, compiler->eulerseq, &f->spec.alt);
    if (ferr) { return comperr(error, ferr, error_sz); }
    mjuu_normvec(framequat, 4);
    mjuu_frameaccumChild(f->spec.pos, framequat, pos, quat);
  }

  // type-specific constructor: populate point and element, possibly set dim
  bool res;
  switch (type) {
    case mjFCOMPTYPE_GRID:
    case mjFCOMPTYPE_CIRCLE:
      res = MakeGrid(error, error_sz);
      break;

    case mjFCOMPTYPE_BOX:
    case mjFCOMPTYPE_CYLINDER:
    case mjFCOMPTYPE_ELLIPSOID:
      res = MakeBox(error, error_sz, dflex->dim);
      break;

    case mjFCOMPTYPE_SQUARE:
    case mjFCOMPTYPE_DISC:
      res = MakeSquare(error, error_sz);
      break;

    case mjFCOMPTYPE_MESH:
      res = MakeMesh(model, compiler, error, error_sz, vfs);
      break;

    case mjFCOMPTYPE_GMSH:
      res = MakeGMSH(model, compiler, error, error_sz, vfs);
      break;

    case mjFCOMPTYPE_DIRECT:
      if (dflex->dim == 1 && element.empty() && point.size() >= 6 && point.size() % 3 == 0) {
        int n = point.size() / 3;
        for (int i = 0; i < n - 1; i++) {
          element.push_back(i);
          element.push_back(i + 1);
        }
      }
      res = true;
      break;

    default:
      return comperr(error, "Unknown flexcomp type", error_sz);
  }
  if (!res) { return false; }

  // default flatskin shading for box, cylinder and 3D grid unless explicitly set
  if (!has_flatskin && (type == mjFCOMPTYPE_BOX ||
                        type == mjFCOMPTYPE_CYLINDER ||
                        (type == mjFCOMPTYPE_GRID && dflex->dim == 3))) {
    dflex->flatskin = true;
  }

  // check pin sizes
  if (pinrange.size() % 2) {
    return comperr(error, "Pin range number must be multiple of 2", error_sz);
  }
  if (pingrid.size() % dflex->dim) {
    return comperr(error, "Pin grid number must be multiple of dim", error_sz);
  }
  if (pingridrange.size() % (2 * dflex->dim)) {
    return comperr(error, "Pin grid range number of must be multiple of 2*dim", error_sz);
  }
  if (type != mjFCOMPTYPE_GRID &&
      !(pingrid.empty() && pingridrange.empty()) &&
      doftype != mjFCOMPDOF_TRILINEAR &&
      doftype != mjFCOMPDOF_QUADRATIC) {
    return comperr(error, "Pin grid(range) can only be used with grid or interpolated", error_sz);
  }
  if (dflex->dim == 1 && !(pingrid.empty() && pingridrange.empty())) {
    return comperr(error, "Pin grid(range) cannot be used with dim=1", error_sz);
  }

  // require element and point
  if (point.empty() || element.empty()) {
    return comperr(error, "Point and element required", error_sz);
  }

  // check point size
  if (point.size() % 3) { return comperr(error, "Point size must be a multiple of 3", error_sz); }

  // check element size
  if (element.size() % (dflex->dim + 1)) {
    return comperr(error, "Element size must be a multiple of dim+1", error_sz);
  }

  // get number of points
  int npnt = point.size() / 3;

  // check elem vertex ids
  for (int i = 0; i < (int)element.size(); i++) {
    if (element[i] < 0 || element[i] >= npnt) {
      char msg[100];
      snprintf(msg,
               sizeof(msg),
               "element %d has point id %d, number of points is %d",
               i,
               element[i],
               npnt);
      return comperr(error, msg, error_sz);
    }
  }

  // check per-vertex texcoord size before any vertex compaction
  if (!texcoord.empty() && elemtexcoord.empty() && (int)texcoord.size() != 2 * npnt) {
    return comperr(error, "two texture coordinates per vertex expected", error_sz);
  }

  // apply scaling for direct types
  if (direct && (scale[0] != 1 || scale[1] != 1 || scale[2] != 1)) {
    for (int i = 0; i < npnt; i++) {
      point[3 * i]     *= scale[0];
      point[3 * i + 1] *= scale[1];
      point[3 * i + 2] *= scale[2];
    }
  }

  // apply pose transform to points
  for (int i = 0; i < npnt; i++) {
    double newp[3], oldp[3] = {point[3 * i], point[3 * i + 1], point[3 * i + 2]};
    mjuu_trnVecPose(newp, pos, quat, oldp);
    point[3 * i]     = newp[0];
    point[3 * i + 1] = newp[1];
    point[3 * i + 2] = newp[2];
  }

  // compute bounding box of points
  double minmax[6] = {mjMAXVAL, mjMAXVAL, mjMAXVAL, -mjMAXVAL, -mjMAXVAL, -mjMAXVAL};
  for (int i = 0; i < npnt; i++) {
    for (int j = 0; j < 3; j++) {
      minmax[j + 0] = std::min(minmax[j + 0], point[3 * i + j]);
      minmax[j + 3] = std::max(minmax[j + 3], point[3 * i + j]);
    }
  }

  // construct pinned array
  int nnode          = 0;
  int count_check[3] = {count[0], count[1], count[2]};
  if (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) {
    int order = doftype == mjFCOMPDOF_TRILINEAR ? 1 : 2;
    // cellcount applies to all types (see the nodal mesh below), else single cell
    int cx = 1, cy = 1, cz = 1;
    if (cellcount[0] >= 0) {
      cx = cellcount[0];
      cy = cellcount[1];
      cz = cellcount[2];
    }
    count_check[0] = cx * order + 1;
    count_check[1] = cy * order + 1;
    count_check[2] = cz * order + 1;
    nnode          = count_check[0] * count_check[1] * count_check[2];
  }
  int npin = nnode ? nnode : npnt;
  pinned   = vector<bool>(std::max(npnt, nnode), rigid);

  // handle pins if user did not specify rigid
  if (!rigid) {
    // process pinid
    for (int i = 0; i < (int)pinid.size(); i++) {
      // check range
      if (pinid[i] < 0 || pinid[i] >= npin) {
        return comperr(error, "pinid out of range", error_sz);
      }

      // set
      pinned[pinid[i]] = true;
    }

    // process pinrange
    for (int i = 0; i < (int)pinrange.size(); i += 2) {
      // check range
      if (pinrange[i] < 0 ||
          pinrange[i] >= npin ||
          pinrange[i + 1] < 0 ||
          pinrange[i + 1] >= npin) {
        return comperr(error, "pinrange out of range", error_sz);
      }

      // set
      for (int k = pinrange[i]; k <= pinrange[i + 1]; k++) { pinned[k] = true; }
    }

    // process pingrid
    for (int i = 0; i < (int)pingrid.size(); i += dflex->dim) {
      // check range
      for (int k = 0; k < dflex->dim; k++) {
        if (pingrid[i + k] < 0 || pingrid[i + k] >= count_check[k]) {
          return comperr(error, "pingrid out of range", error_sz);
        }
      }

      // set
      if (dflex->dim == 2) {
        pinned[GridID(pingrid[i], pingrid[i + 1])] = true;
      } else if (dflex->dim == 3) {
        pinned[GridID(pingrid[i], pingrid[i + 1], pingrid[i + 2])] = true;
      }
    }

    // process pingridrange
    for (int i = 0; i < (int)pingridrange.size(); i += 2 * dflex->dim) {
      // check range
      for (int k = 0; k < 2 * dflex->dim; k++) {
        if (pingridrange[i + k] < 0 || pingridrange[i + k] >= count_check[k % dflex->dim]) {
          return comperr(error, "pingridrange out of range", error_sz);
        }
      }

      // set
      if (dflex->dim == 2) {
        for (int ix = pingridrange[i]; ix <= pingridrange[i + 2]; ix++) {
          for (int iy = pingridrange[i + 1]; iy <= pingridrange[i + 3]; iy++) {
            pinned[GridID(ix, iy)] = true;
          }
        }
      } else if (dflex->dim == 3) {
        for (int ix = pingridrange[i]; ix <= pingridrange[i + 3]; ix++) {
          for (int iy = pingridrange[i + 1]; iy <= pingridrange[i + 4]; iy++) {
            for (int iz = pingridrange[i + 2]; iz <= pingridrange[i + 5]; iz++) {
              pinned[GridID(ix, iy, iz)] = true;
            }
          }
        }
      }
    }

    // center of radial body is always pinned
    if (doftype == mjFCOMPDOF_RADIAL) { pinned[0] = true; }

    // check if all or none are pinned
    bool allpin = true, nopin = true;
    for (int i = 0; i < npin; i++) {
      if (pinned[i]) {
        nopin = false;
      } else {
        allpin = false;
      }
    }

    // adjust rigid and centered
    if (allpin) {
      rigid = true;
    } else if (nopin) {
      centered = true;
    }
  }

  // remove unreferenced for direct, mesh, gmsh
  if (direct) {
    // find used
    used = std::vector<bool>(npnt, false);
    for (int i = 0; i < (int)element.size(); i++) { used[element[i]] = true; }

    // construct reindex
    bool             hasunused = false;
    std::vector<int> reindex(npnt, 0);
    for (int i = 0; i < npnt; i++) {
      if (!used[i]) {
        hasunused = true;
        for (int k = i + 1; k < npnt; k++) { reindex[k]--; }
      }
    }

    // reindex elements if unused present
    if (hasunused) {
      for (int i = 0; i < (int)element.size(); i++) { element[i] += reindex[element[i]]; }

      // compact point, texcoord, pinned arrays
      int new_npnt = 0;
      for (int i = 0; i < npnt; i++) {
        if (used[i]) {
          point[3 * new_npnt + 0] = point[3 * i + 0];
          point[3 * new_npnt + 1] = point[3 * i + 1];
          point[3 * new_npnt + 2] = point[3 * i + 2];

          if (!texcoord.empty() && elemtexcoord.empty()) {
            texcoord[2 * new_npnt + 0] = texcoord[2 * i + 0];
            texcoord[2 * new_npnt + 1] = texcoord[2 * i + 1];
          }

          if (!nnode) { pinned[new_npnt] = pinned[i]; }
          new_npnt++;
        }
      }

      // resize arrays
      point.resize(3 * new_npnt);
      if (!texcoord.empty() && elemtexcoord.empty()) { texcoord.resize(2 * new_npnt); }
      if (!nnode) { pinned.resize(new_npnt); }
      used.assign(new_npnt, true);

      // update count
      npnt = new_npnt;
    }
  }

  // nothing to remove for auto-generated types
  else {
    used = std::vector<bool>(npnt, true);
  }

  // create flex, copy parameters
  mjCFlex* flex = model->AddFlex();
  mjsFlex* pf   = &flex->spec;
  int      id   = flex->id;

  *flex = def.Flex();
  flex->PointToLocal();

  flex->model = model;
  flex->id    = id;
  mjs_setName(pf->element, name.c_str());
  mjs_setInt(pf->elem, element.data(), element.size());
  mjs_setFloat(pf->texcoord, texcoord.data(), texcoord.size());
  mjs_setInt(pf->elemtexcoord, elemtexcoord.data(), elemtexcoord.size());
  if (!centered) { mjs_setDouble(pf->vert, point.data(), point.size()); }

  // rigid: set parent name, nothing else to do
  if (rigid) {
    mjs_appendString(pf->vertbody, mjs_getName(body->element)->c_str());
    return true;
  }

  // compute body mass and inertia matching specs
  double bodymass    = mass / npnt;
  double bodyinertia = bodymass * (2.0 * inertiabox * inertiabox) / 3.0;

  // overwrite plugin name
  if (plugin.active && plugin_instance_name.empty()) {
    plugin_instance_name                          = "flexcomp_" + name;
    static_cast<mjCPlugin*>(plugin.element)->name = plugin_instance_name;
  }

  // create bodies, construct flex vert and vertbody
  for (int i = 0; i < npnt; i++) {
    // not used: skip
    if (!used[i]) { continue; }

    // pinned or trilinear or quadratic: parent body
    if (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC || pinned[i]) {
      mjs_appendString(pf->vertbody, mjs_getName(body->element)->c_str());

      // add plugin
      if (plugin.active) {
        mjsPlugin* pplugin = &body->plugin;
        pplugin->active    = true;
        pplugin->element   = static_cast<mjsElement*>(plugin.element);
        mjs_setString(pplugin->plugin_name, mjs_getString(plugin.plugin_name));
        mjs_setString(pplugin->name, plugin_instance_name.c_str());
      }
    }

    // not pinned and not trilinear: new body
    else {
      // add new body at vertex coordinates
      mjsBody* pb = mjs_addBody(body, 0);

      // set frame and inertial
      pb->pos[0] = point[3 * i];
      pb->pos[1] = point[3 * i + 1];
      pb->pos[2] = point[3 * i + 2];
      mjuu_zerovec(pb->ipos, 3);
      pb->mass             = bodymass;
      pb->inertia[0]       = bodyinertia;
      pb->inertia[1]       = bodyinertia;
      pb->inertia[2]       = bodyinertia;
      pb->explicitinertial = true;

      // add radial slider
      if (doftype == mjFCOMPDOF_RADIAL) {
        mjsJoint* jnt = mjs_addJoint(pb, 0);

        // set properties
        jnt->type = mjJNT_SLIDE;
        mjuu_setvec(jnt->pos, 0, 0, 0);
        mjuu_copyvec(jnt->axis, pb->pos, 3);
        mjuu_normvec(jnt->axis, 3);
      }

      // add three orthogonal sliders
      else if (doftype == mjFCOMPDOF_FULL) {
        for (int j = 0; j < 3; j++) {
          // add joint to body
          mjsJoint* jnt = mjs_addJoint(pb, 0);

          // set properties
          jnt->type = mjJNT_SLIDE;
          mjuu_setvec(jnt->pos, 0, 0, 0);
          mjuu_setvec(jnt->axis, 0, 0, 0);
          jnt->axis[j] = 1;
        }
      }

      // add two orthogonal sliders (x and y only)
      else if (doftype == mjFCOMPDOF_2D) {
        for (int j = 0; j < 2; j++) {
          mjsJoint* jnt = mjs_addJoint(pb, 0);
          jnt->type     = mjJNT_SLIDE;
          mjuu_setvec(jnt->pos, 0, 0, 0);
          mjuu_setvec(jnt->axis, 0, 0, 0);
          jnt->axis[j] = 1;
        }
      }

      // construct body name, add to vertbody
      char txt[100];
      mju::sprintf_arr(txt, "%s_%d", name.c_str(), i);
      mjs_setName(pb->element, txt);
      mjs_appendString(pf->vertbody, mjs_getName(pb->element)->c_str());

      // clear flex vertex coordinates if allocated
      if (!centered) {
        point[3 * i]     = 0;
        point[3 * i + 1] = 0;
        point[3 * i + 2] = 0;
      }

      // add plugin
      if (plugin.active) {
        mjsPlugin* pplugin = &pb->plugin;
        pplugin->active    = true;
        pplugin->element   = static_cast<mjsElement*>(plugin.element);
        mjs_setString(pplugin->plugin_name, mjs_getString(plugin.plugin_name));
        mjs_setString(pplugin->name, plugin_instance_name.c_str());
      }
    }
  }

  // add twist hinge body and joint for 1D elastic open rods with a clamped root edge
  if (pf->dim == 1 &&
      !rigid &&
      doftype == mjFCOMPDOF_FULL &&
      pf->young > 0 &&
      type != mjFCOMPTYPE_CIRCLE &&
      pinned.size() >= 2 &&
      pinned[0] &&
      pinned[1]) {
    mjsBody* pb = mjs_addBody(body, 0);
    mjuu_setvec(pb->pos, 0, 0, 0);
    mjuu_zerovec(pb->ipos, 3);

    // rotational inertia of the twist hinge: polar mass moment of the rod about its axis
    double polar =
        pf->section == mjFLEXSECTION_BOX
            ? (pf->sectionsize[0] * pf->sectionsize[0] + pf->sectionsize[1] * pf->sectionsize[1]) /
                  3
            : pf->radius * pf->radius / 2;
    double twistinertia  = mass * polar > mjMINVAL ? mass * polar : bodymass;
    pb->mass             = bodymass;
    pb->inertia[0]       = twistinertia;
    pb->inertia[1]       = twistinertia;
    pb->inertia[2]       = twistinertia;
    pb->explicitinertial = true;

    mjsJoint* jnt = mjs_addJoint(pb, 0);
    jnt->type     = mjJNT_HINGE;
    mjuu_setvec(jnt->pos, 0, 0, 0);
    mjuu_setvec(jnt->axis, 1, 0, 0);

    char txt[100];
    mju::sprintf_arr(txt, "%s_twist", name.c_str());
    mjs_setName(pb->element, txt);
    mjs_setName(jnt->element, txt);
  }

  // create nodal mesh for trilinear/quadratic interpolation
  if (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) {
    flex->spec.order = doftype == mjFCOMPDOF_TRILINEAR ? 1 : 2;

    if (cellcount[0] >= 0) {
      flex->spec.cellcount[0] = cellcount[0];
      flex->spec.cellcount[1] = cellcount[1];
      flex->spec.cellcount[2] = cellcount[2];
    }

    // total number of nodes with shared boundaries
    int nx    = flex->spec.cellcount[0] * flex->spec.order + 1;
    int ny    = flex->spec.cellcount[1] * flex->spec.order + 1;
    int nz    = flex->spec.cellcount[2] * flex->spec.order + 1;
    int nnode = nx * ny * nz;

    // mark empty cells and pin nodes exclusively in empty cells (volume mode only)
    if (!dflex->elastic2d) { MarkEmptyCells(flex, point.data(), minmax, nx, ny, nz); }

    // shell mode: pin all interior (non-boundary) nodes
    if (dflex->elastic2d) {
      for (int gi = 0; gi < nx; gi++) {
        for (int gj = 0; gj < ny; gj++) {
          for (int gk = 0; gk < nz; gk++) {
            bool is_boundary =
                (gi == 0 || gi == nx - 1 || gj == 0 || gj == ny - 1 || gk == 0 || gk == nz - 1);
            if (!is_boundary) { pinned[gi * ny * nz + gj * nz + gk] = true; }
          }
        }
      }
    }

    // if MarkEmptyCells pinned any nodes, force centered=false
    // so that pf->node (local positions) is saved to the model
    if (centered) {
      for (int i = 0; i < nnode; i++) {
        if (pinned[i]) {
          centered = false;
          break;
        }
      }
    }

    std::vector<double> node(3 * nnode, 0);
    int                 idx = 0;

    // Simpson's rule weights for quadratic mass distribution
    double massP2[3] = {1. / 6., 2. / 3., 1. / 6.};


    // collect created bodies for mass normalization
    std::vector<mjsBody*> node_bodies;

    for (int gi = 0; gi < nx; gi++) {
      for (int gj = 0; gj < ny; gj++) {
        for (int gk = 0; gk < nz; gk++) {
          // parametric position in [0, 1]^3
          double s = (double)gi / (flex->spec.cellcount[0] * flex->spec.order);
          double t = (double)gj / (flex->spec.cellcount[1] * flex->spec.order);
          double u = (double)gk / (flex->spec.cellcount[2] * flex->spec.order);

          // physical position
          double px = minmax[0] + s * (minmax[3] - minmax[0]);
          double py = minmax[1] + t * (minmax[4] - minmax[1]);
          double pz = minmax[2] + u * (minmax[5] - minmax[2]);

          if (pinned[idx]) {
            node[3 * idx + 0] = px;
            node[3 * idx + 1] = py;
            node[3 * idx + 2] = pz;
            mjs_appendString(pf->nodebody, mjs_getName(body->element)->c_str());
            idx++;
            continue;
          }

          mjsBody* pb = mjs_addBody(body, 0);
          pb->pos[0]  = px;
          pb->pos[1]  = py;
          pb->pos[2]  = pz;
          mjuu_zerovec(pb->ipos, 3);

          // mass distribution
          if (!dflex->elastic2d) {
            pb->mass = flex->NodeMassWeight(gi, gj, gk);
          } else if (doftype == mjFCOMPDOF_TRILINEAR) {
            pb->mass = 1.0;
          } else {
            // local index within the cell for mass computation
            int li = gi % flex->spec.order;
            int lj = gj % flex->spec.order;
            int lk = gk % flex->spec.order;
            // boundary nodes: average mass contribution
            int ncells_i = (gi > 0 && gi < nx - 1 && li == 0) ? 2 : 1;
            int ncells_j = (gj > 0 && gj < ny - 1 && lj == 0) ? 2 : 1;
            int ncells_k = (gk > 0 && gk < nz - 1 && lk == 0) ? 2 : 1;
            // use Simpson weights
            double wi = massP2[li == 0 ? 0 : li];
            double wj = massP2[lj == 0 ? 0 : lj];
            double wk = massP2[lk == 0 ? 0 : lk];
            pb->mass  = wi * wj * wk * ncells_i * ncells_j * ncells_k;
          }

          node_bodies.push_back(pb);

          pb->inertia[0]       = pb->mass * (2.0 * inertiabox * inertiabox) / 3.0;
          pb->inertia[1]       = pb->mass * (2.0 * inertiabox * inertiabox) / 3.0;
          pb->inertia[2]       = pb->mass * (2.0 * inertiabox * inertiabox) / 3.0;
          pb->explicitinertial = true;

          for (int d = 0; d < 3; d++) {
            mjsJoint* jnt = mjs_addJoint(pb, 0);
            jnt->type     = mjJNT_SLIDE;
            mjuu_setvec(jnt->pos, 0, 0, 0);
            mjuu_setvec(jnt->axis, 0, 0, 0);
            jnt->axis[d] = 1;
          }

          // construct node name, add to nodebody
          char txt[100];
          mju::sprintf_arr(txt, "%s_%d_%d_%d", name.c_str(), gi, gj, gk);
          mjs_setName(pb->element, txt);
          mjs_appendString(pf->nodebody, mjs_getName(pb->element)->c_str());

          idx++;
        }
      }
    }

    // normalize masses so total equals prescribed mass
    double total_mass = 0;
    for (mjsBody* pb : node_bodies) { total_mass += pb->mass; }
    if (total_mass > 0) {
      double scale = mass / total_mass;
      for (mjsBody* pb : node_bodies) {
        pb->mass       *= scale;
        pb->inertia[0] *= scale;
        pb->inertia[1] *= scale;
        pb->inertia[2] *= scale;
      }
    }

    if (!centered) { mjs_setDouble(pf->node, node.data(), node.size()); }
  }

  if (!centered || doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) {
    mjs_setDouble(pf->vert, point.data(), point.size());
  }

  // create equality constraints
  if (equality) {
    // equality 1=edge(mjEQ_FLEX), 3=strain(mjEQ_FLEXSTRAIN); 2 (vert) was removed
    if (equality == 1) {
      mjsEquality* pe = mjs_addEquality(&model->spec, &def.spec);
      mjs_setDefault(pe->element, &model->Default()->spec);
      pe->type   = mjEQ_FLEX;
      pe->active = true;
      mjs_setString(pe->name1, name.c_str());
    } else if (equality == 3) {
      // create one strain constraint per finite element, storing element index
      int  cell_cx = flex->spec.cellcount[0];
      int  cell_cy = flex->spec.cellcount[1];
      int  cell_cz = flex->spec.cellcount[2];
      bool shell   = (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) &&
                     flex->spec.elastic2d;

      if (shell) {
        // shell mode: one constraint per boundary face element
        int nelem_fe = 2 * (cell_cy * cell_cz + cell_cx * cell_cz + cell_cx * cell_cy);
        for (int fe = 0; fe < nelem_fe; fe++) {
          mjsEquality* pe = mjs_addEquality(&model->spec, &def.spec);
          mjs_setDefault(pe->element, &model->Default()->spec);
          pe->type   = mjEQ_FLEXSTRAIN;
          pe->active = true;
          mjs_setString(pe->name1, name.c_str());
          pe->data[0] = fe;
          pe->data[1] = -1;  // sentinel: shell mode
          pe->data[2] = -1;
        }
      } else {
        // volume mode: one constraint per 3D cell
        for (int ci = 0; ci < cell_cx; ci++) {
          for (int cj = 0; cj < cell_cy; cj++) {
            for (int ck = 0; ck < cell_cz; ck++) {
              // skip empty cells
              if (!flex->cell_empty.empty() &&
                  flex->cell_empty[ci * cell_cy * cell_cz + cj * cell_cz + ck]) {
                continue;
              }
              mjsEquality* pe = mjs_addEquality(&model->spec, &def.spec);
              mjs_setDefault(pe->element, &model->Default()->spec);
              pe->type   = mjEQ_FLEXSTRAIN;
              pe->active = true;
              mjs_setString(pe->name1, name.c_str());
              pe->data[0] = ci;
              pe->data[1] = cj;
              pe->data[2] = ck;
            }
          }
        }
      }
    }
  }

  return true;
}


// get point id from grid coordinates
int mjCFlexcomp::GridID(int ix, int iy) {
  if (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) {
    int order = doftype == mjFCOMPDOF_TRILINEAR ? 1 : 2;
    int cy    = cellcount[0] >= 0 ? cellcount[1] : 1;
    return ix * (cy * order + 1) + iy;
  }
  return ix * count[1] + iy;
}
int mjCFlexcomp::GridID(int ix, int iy, int iz) {
  if (doftype == mjFCOMPDOF_TRILINEAR || doftype == mjFCOMPDOF_QUADRATIC) {
    int order = doftype == mjFCOMPDOF_TRILINEAR ? 1 : 2;
    int cy    = cellcount[0] >= 0 ? cellcount[1] : 1;
    int cz    = cellcount[0] >= 0 ? cellcount[2] : 1;
    int ny    = cy * order + 1;
    int nz    = cz * order + 1;
    return ix * ny * nz + iy * nz + iz;
  }
  return ix * count[1] * count[2] + iy * count[2] + iz;
}


// append points, elements and texcoords of a procedural mesh spec; 1D meshes
// only have nodes, consecutive nodes are connected and closed loops wrap around
static void AppendMeshSpec(const mjsMesh&       mesh,
                           int                  dim,
                           bool                 closed,
                           std::vector<double>& point,
                           std::vector<int>&    element,
                           std::vector<float>&  texcoord) {
  point.insert(point.end(), mesh.usernode->begin(), mesh.usernode->end());
  texcoord.insert(texcoord.end(), mesh.usertexcoord->begin(), mesh.usertexcoord->end());
  if (dim == 1) {
    int n     = mesh.usernode->size() / 3;
    int nedge = closed ? n : n - 1;
    for (int i = 0; i < nedge; i++) {
      element.push_back(i);
      element.push_back((i + 1) % n);
    }
  } else {
    const std::vector<int>& elem = dim == 3 ? *mesh.usertet : *mesh.userface;
    element.insert(element.end(), elem.begin(), elem.end());
  }
}


// make grid
bool mjCFlexcomp::MakeGrid(char* error, int error_sz) {
  int  dim     = def.Flex().spec.dim;
  bool needtex = texcoord.empty() && !def.spec.flex->material->empty();
  bool circle  = dim == 1 && type == mjFCOMPTYPE_CIRCLE;

  mjCMesh mesh;
  if (circle) {
    mesh.MakeCircle(count, spacing);
  } else {
    mesh.MakeGrid(count, spacing, dim, needtex);
  }
  AppendMeshSpec(mesh.spec, dim, circle, point, element, texcoord);

  // check elements
  if (element.empty()) { return comperr(error, "No elements were created in grid", error_sz); }

  return true;
}


// make 2d square or disc
bool mjCFlexcomp::MakeSquare(char* error, int error_sz) {
  // set 2D
  def.spec.flex->dim = 2;
  bool needtex       = texcoord.empty() && !def.spec.flex->material->empty();

  mjCMesh mesh;
  if (type == mjFCOMPTYPE_DISC) {
    mesh.MakeDisc(count, spacing, needtex);
  } else {
    mesh.MakeGrid(count, spacing, 2, needtex);
  }
  AppendMeshSpec(mesh.spec, 2, /*closed=*/false, point, element, texcoord);

  // check elements
  if (element.empty()) { return comperr(error, "No elements were created in grid", error_sz); }

  return true;
}


// make 3d box, ellipsoid or cylinder
bool mjCFlexcomp::MakeBox(char* error, int error_sz, int dim, bool open) {
  bool needtex = texcoord.empty() && !def.spec.flex->material->empty();

  // set dimension
  def.spec.flex->dim = dim;

  mjCMesh mesh;
  if (type == mjFCOMPTYPE_CYLINDER) {
    mesh.MakeCylinder(count, spacing, dim, needtex, open);
  } else if (type == mjFCOMPTYPE_ELLIPSOID) {
    mesh.MakeEllipsoid(count, spacing, dim, needtex, open);
  } else {
    mesh.MakeBox(count, spacing, dim, needtex, open);
  }
  AppendMeshSpec(mesh.spec, dim, /*closed=*/false, point, element, texcoord);

  // check elements
  if (element.empty()) { return comperr(error, "No elements were created in box", error_sz); }

  return true;
}


// copied from user_mesh.cc
template <typename T>
static T* VecToArray(std::vector<T>& vector, bool clear = true) {
  if (vector.empty())
    return nullptr;
  else {
    int n    = (int)vector.size();
    T*  cvec = (T*)mju_malloc(n * sizeof(T));
    memcpy(cvec, vector.data(), n * sizeof(T));
    if (clear) { vector.clear(); }
    return cvec;
  }
}


// map points of a compiled mesh from its inertial frame back to the asset frame, i.e. undo the
// translation to the center of mass and the rotation to the principal axes; the user
// transformations (scale, refpos, refquat) remain applied, as for geoms
static void ToAssetFrame(std::vector<double>& point, const double pos[3], const double quat[4]) {
  double neg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  double mat[9];
  mjuu_quat2mat(mat, neg);
  for (size_t i = 0; i + 2 < point.size(); i += 3) {
    mjuu_mulvecmatT(&point[i], &point[i], mat);
    for (int j = 0; j < 3; j++) { point[i + j] += pos[j]; }
  }
}


// make mesh
bool mjCFlexcomp::MakeMesh(
    mjCModel* model, mjsCompiler* compiler, char* error, int error_sz, const mjVFS* vfs) {
  mjCMesh                  local_mesh;
  mjCMesh*                 cmesh = nullptr;
  std::unique_ptr<mjCMesh> unscaled;

  if (!mesh.empty()) {
    // use mesh asset
    cmesh = static_cast<mjCMesh*>(model->FindObject(mjOBJ_MESH, mesh));
    if (!cmesh) {
      snprintf(error, error_sz, "mesh '%s' not found", mesh.c_str());
      return false;
    }

    if (cmesh->instancemap_ != decltype(cmesh->instancemap_){1, 0, 0, 0, 1, 0, 0, 0, 1}) {
      unscaled               = std::make_unique<mjCMesh>(*cmesh);
      unscaled->model        = cmesh->model;
      unscaled->compiler     = cmesh->compiler;
      unscaled->instancemap_ = {1, 0, 0, 0, 1, 0, 0, 0, 1};
      cmesh                  = unscaled.get();
    }

    try {
      cmesh->Compile(vfs);
    } catch (mjCError err) { return comperr(error, err.message, error_sz); }
  } else {
    // strip path
    if (!file.empty() && model->spec.strippath) { file = mjuu_strippath(file); }

    // file is required
    if (file.empty()) { return comperr(error, "File is required", error_sz); }

    // load resource
    std::string filename = mjuu_combinePaths(mjs_getString(compiler->meshdir), file);
    mjResource* resource = nullptr;

    if (type == mjFCOMPTYPE_GMSH) {
      mjs_setString(local_mesh.spec.content_type, "model/vnd.gmsh");
      local_mesh.CopyFromSpec();
    } else if (mjCMesh::IsMSH(filename)) {
      return comperr(error, "legacy MSH files are not supported in flexcomp", error_sz);
    }

    try {
      resource = mjCBase::LoadResource(mjs_getString(model->spec.modelfiledir), filename, vfs);
    } catch (mjCError err) {
      if (file_or_mesh) {
        std::string msg = "'" + file + "' is neither a mesh asset nor a file: " + err.message;
        return comperr(error, msg.c_str(), error_sz);
      }
      return comperr(error, err.message, error_sz);
    }

    // load mesh
    try {
      local_mesh.LoadFromResource(resource, true);
      mju_closeResource(resource);
    } catch (mjCError err) {
      mju_closeResource(resource);
      return comperr(error, err.message, error_sz);
    }
    cmesh = &local_mesh;
  }

  // volumetric mesh (e.g. from GMSH)
  if (!cmesh->Tet().empty()) {
    if (has_dim && def.spec.flex->dim != 3) {
      return comperr(error,
                     type == mjFCOMPTYPE_GMSH
                         ? "flexcomp dim does not match GMSH mesh dimensionality"
                         : "flexcomp dim does not match mesh dimensionality",
                     error_sz);
    }
    def.spec.flex->dim = 3;

    if (cmesh->Node().empty()) {
      return comperr(error,
                     type == mjFCOMPTYPE_GMSH ? "GMSH mesh has no nodes" : "mesh has no nodes",
                     error_sz);
    }
    point.assign(cmesh->Node().begin(), cmesh->Node().end());
    if (!mesh.empty()) { ToAssetFrame(point, cmesh->GetPosPtr(), cmesh->GetQuatPtr()); }
    element.assign(cmesh->Tet().begin(), cmesh->Tet().end());

    // face texcoord indices refer to surface faces, not to tetrahedra: only keep
    // texcoords given per node
    if (cmesh->HasTexcoord() && cmesh->Texcoord().size() == 2 * (point.size() / 3)) {
      texcoord = cmesh->Texcoord();
    }
    return true;
  }

  if (type == mjFCOMPTYPE_GMSH) {
    if (has_dim && def.spec.flex->dim != 2) {
      return comperr(error, "flexcomp dim does not match GMSH mesh dimensionality", error_sz);
    }
    def.spec.flex->dim = 2;
  }

  // check dim
  if (def.spec.flex->dim < 1) {
    return comperr(error, "Flex dim must be at least 1 for mesh", error_sz);
  }

  // check sizes
  if (cmesh->Vert().empty() || cmesh->Face().empty()) {
    if (type == mjFCOMPTYPE_GMSH) {
      return comperr(error, "unsupported GMSH mesh dimensionality", error_sz);
    }
    return comperr(error, "Vertex and face data required", error_sz);
  }

  // copy vertices; faces index the vertices, which may have been deduplicated, not the nodes
  point.assign(cmesh->Vert().begin(), cmesh->Vert().end());
  if (!mesh.empty()) { ToAssetFrame(point, cmesh->GetPosPtr(), cmesh->GetQuatPtr()); }

  if (cmesh->HasTexcoord()) { texcoord = cmesh->Texcoord(); }

  // copy faces or create 3D mesh
  if (def.spec.flex->dim == 2) {
    element = cmesh->Face();
    if (cmesh->HasTexcoord()) { elemtexcoord = cmesh->FaceTexcoord(); }
  } else if (def.spec.flex->dim == 1) {
    // extract edge pairs from degenerate triangles (i1, i2, i2)
    const std::vector<int>& face = cmesh->Face();
    const std::vector<int>& ftc  = cmesh->FaceTexcoord();
    element.clear();
    element.reserve(face.size() * 2 / 3);
    if (cmesh->HasTexcoord() && !ftc.empty()) {
      elemtexcoord.clear();
      elemtexcoord.reserve(ftc.size() * 2 / 3);
    }
    for (size_t i = 0; i < face.size(); i += 3) {
      element.push_back(face[i]);
      element.push_back(face[i + 1]);
      if (cmesh->HasTexcoord() && i + 1 < ftc.size()) {
        elemtexcoord.push_back(ftc[i]);
        elemtexcoord.push_back(ftc[i + 1]);
      }
    }
  } else {
    if (!has_origin) {
      return comperr(error, "origin must be specified for mesh flexcomps if dim=3", error_sz);
    }
    point.insert(point.begin() + 0, origin[0]);
    point.insert(point.begin() + 1, origin[1]);
    point.insert(point.begin() + 2, origin[2]);
    const std::vector<int>& face = cmesh->Face();
    const std::vector<int>& ftc  = cmesh->FaceTexcoord();
    if (cmesh->HasTexcoord()) {
      texcoord.insert(texcoord.begin(), {0.0f, 0.0f});
      elemtexcoord.clear();
    }
    for (int i = 0; i < (int)face.size(); i += 3) {
      // only add tetrahedra with positive volume
      int    tet[3] = {face[i + 0] + 1, face[i + 1] + 1, face[i + 2] + 1};
      double edge1[3], edge2[3], edge3[3];
      for (int i = 0; i < 3; i++) {
        edge1[i] = point[3 * tet[0] + i] - origin[i];
        edge2[i] = point[3 * tet[1] + i] - origin[i];
        edge3[i] = point[3 * tet[2] + i] - origin[i];
      }
      double normal[3];
      mjuu_crossvec(normal, edge1, edge2);
      if (mjuu_dot3(normal, edge3) < mjMINVAL) { continue; }
      element.push_back(0);
      element.push_back(tet[0]);
      element.push_back(tet[1]);
      element.push_back(tet[2]);
      if (cmesh->HasTexcoord() && !ftc.empty()) {
        elemtexcoord.push_back(0);
        elemtexcoord.push_back(ftc[i + 0] + 1);
        elemtexcoord.push_back(ftc[i + 1] + 1);
        elemtexcoord.push_back(ftc[i + 2] + 1);
      }
    }
  }

  return true;
}


// load points and elements from GMSH file
bool mjCFlexcomp::MakeGMSH(
    mjCModel* model, mjsCompiler* compiler, char* error, int error_sz, const mjVFS* vfs) {
  type = mjFCOMPTYPE_GMSH;
  return MakeMesh(model, compiler, error, error_sz, vfs);
}


//-------------------------- nonlinear elasticity --------------------------------------------------

// simplex connectivity
constexpr int eledge[3][6][2] = {
    {{0, 1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}, {-1, -1}},
    {{1, 2}, {2, 0},   {0, 1},   {-1, -1}, {-1, -1}, {-1, -1}},
    {{0, 1}, {1, 2},   {2, 0},   {2, 3},   {0, 3},   {1, 3}  }
};

struct Stencil2D {
  static constexpr int kNumEdges          = 3;
  static constexpr int kNumVerts          = 3;
  static constexpr int kNumFaces          = 2;
  static constexpr int edge[kNumEdges][2] = {
      {1, 2},
      {2, 0},
      {0, 1}
  };
  static constexpr int face[kNumVerts][2] = {
      {1, 2},
      {2, 0},
      {0, 1}
  };
  static constexpr int edge2face[kNumEdges][2] = {
      {1, 2},
      {2, 0},
      {0, 1}
  };
  int vertices[kNumVerts];
  int edges[kNumEdges];
};

struct Stencil3D {
  static constexpr int kNumEdges          = 6;
  static constexpr int kNumVerts          = 4;
  static constexpr int kNumFaces          = 3;
  static constexpr int edge[kNumEdges][2] = {
      {0, 1},
      {1, 2},
      {2, 0},
      {2, 3},
      {0, 3},
      {1, 3}
  };
  static constexpr int face[kNumVerts][3] = {
      {2, 1, 0},
      {0, 1, 3},
      {1, 2, 3},
      {2, 0, 3}
  };
  static constexpr int edge2face[kNumEdges][2] = {
      {2, 3},
      {1, 3},
      {2, 1},
      {1, 0},
      {0, 2},
      {0, 3}
  };
  int vertices[kNumVerts];
  int edges[kNumEdges];
};

template <typename T>
inline double ComputeVolume(const double* x, const int v[T::kNumVerts]);

template <>
inline double ComputeVolume<Stencil2D>(const double* x, const int v[Stencil2D::kNumVerts]) {
  double        normal[3];
  const double* x0       = x + 3 * v[0];
  const double* x1       = x + 3 * v[1];
  const double* x2       = x + 3 * v[2];
  double        edge1[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
  double        edge2[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
  mjuu_crossvec(normal, edge1, edge2);
  return mjuu_normvec(normal, 3) / 2;
}

template <>
inline double ComputeVolume<Stencil3D>(const double* x, const int v[Stencil3D::kNumVerts]) {
  double        normal[3];
  const double* x0       = x + 3 * v[0];
  const double* x1       = x + 3 * v[1];
  const double* x2       = x + 3 * v[2];
  const double* x3       = x + 3 * v[3];
  double        edge1[3] = {x1[0] - x0[0], x1[1] - x0[1], x1[2] - x0[2]};
  double        edge2[3] = {x2[0] - x0[0], x2[1] - x0[1], x2[2] - x0[2]};
  double        edge3[3] = {x3[0] - x0[0], x3[1] - x0[1], x3[2] - x0[2]};
  mjuu_crossvec(normal, edge1, edge2);
  return mjuu_dot3(normal, edge3) / 6;
}

// compute metric tensor of edge lengths inner product
template <typename T>
void inline MetricTensor(double*      metric,
                         int          idx,
                         double       mu,
                         double       la,
                         const double basis[T::kNumEdges][9],
                         int          stride = 21) {
  double trE[T::kNumEdges]                 = {0};
  double trEE[T::kNumEdges * T::kNumEdges] = {0};
  double k[T::kNumEdges * T::kNumEdges];

  // compute first invariant i.e. trace(strain)
  for (int e = 0; e < T::kNumEdges; e++) {
    for (int i = 0; i < 3; i++) { trE[e] += basis[e][4 * i]; }
  }

  // compute second invariant i.e. trace(strain^2)
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = 0; ed2 < T::kNumEdges; ed2++) {
      for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
          trEE[T::kNumEdges * ed1 + ed2] += basis[ed1][3 * i + j] * basis[ed2][3 * j + i];
        }
      }
    }
  }

  // assembly of strain metric tensor
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = 0; ed2 < T::kNumEdges; ed2++) {
      k[T::kNumEdges * ed1 + ed2] = mu * trEE[T::kNumEdges * ed1 + ed2] + la * trE[ed2] * trE[ed1];
    }
  }

  // copy to triangular representation
  int id = 0;
  for (int ed1 = 0; ed1 < T::kNumEdges; ed1++) {
    for (int ed2 = ed1; ed2 < T::kNumEdges; ed2++) {
      metric[stride * idx + id++] = k[T::kNumEdges * ed1 + ed2];
    }
  }

  if (id != T::kNumEdges * (T::kNumEdges + 1) / 2) { mju_error("incorrect stiffness matrix size"); }
}

// compute local basis
template <typename T>
void inline ComputeBasis(double        basis[9],
                         const double* x,
                         const int     v[T::kNumVerts],
                         const int     faceL[T::kNumFaces],
                         const int     faceR[T::kNumFaces],
                         double        volume);

template <>
void inline ComputeBasis<Stencil2D>(double        basis[9],
                                    const double* x,
                                    const int     v[Stencil2D::kNumVerts],
                                    const int     faceL[Stencil2D::kNumFaces],
                                    const int     faceR[Stencil2D::kNumFaces],
                                    double        volume) {
  double basisL[3], basisR[3];
  double normal[3];

  const double* xL0       = x + 3 * v[faceL[0]];
  const double* xL1       = x + 3 * v[faceL[1]];
  const double* xR0       = x + 3 * v[faceR[0]];
  const double* xR1       = x + 3 * v[faceR[1]];
  double        edgesL[3] = {xL0[0] - xL1[0], xL0[1] - xL1[1], xL0[2] - xL1[2]};
  double        edgesR[3] = {xR1[0] - xR0[0], xR1[1] - xR0[1], xR1[2] - xR0[2]};

  mjuu_crossvec(normal, edgesR, edgesL);
  mjuu_normvec(normal, 3);
  mjuu_crossvec(basisL, normal, edgesL);
  mjuu_crossvec(basisR, edgesR, normal);

  // we use as basis the symmetrized tensor products of the edge normals of the
  // other two edges; this is shown in Weischedel "A discrete geometric view on
  // shear-deformable shell models" in the remark at the end of section 4.1;
  // equivalent to linear finite elements but in a coordinate-free formulation.

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      basis[3 * i + j] = (basisL[i] * basisR[j] + basisR[i] * basisL[j]) / (8 * volume * volume);
    }
  }
}

// compute local basis
template <>
void inline ComputeBasis<Stencil3D>(double        basis[9],
                                    const double* x,
                                    const int     v[Stencil3D::kNumVerts],
                                    const int     faceL[Stencil3D::kNumFaces],
                                    const int     faceR[Stencil3D::kNumFaces],
                                    double        volume) {
  const double* xL0       = x + 3 * v[faceL[0]];
  const double* xL1       = x + 3 * v[faceL[1]];
  const double* xL2       = x + 3 * v[faceL[2]];
  const double* xR0       = x + 3 * v[faceR[0]];
  const double* xR1       = x + 3 * v[faceR[1]];
  const double* xR2       = x + 3 * v[faceR[2]];
  double        edgesL[6] = {xL1[0] - xL0[0],
                             xL1[1] - xL0[1],
                             xL1[2] - xL0[2],
                             xL2[0] - xL0[0],
                             xL2[1] - xL0[1],
                             xL2[2] - xL0[2]};
  double        edgesR[6] = {xR1[0] - xR0[0],
                             xR1[1] - xR0[1],
                             xR1[2] - xR0[2],
                             xR2[0] - xR0[0],
                             xR2[1] - xR0[1],
                             xR2[2] - xR0[2]};

  double normalL[3], normalR[3];
  mjuu_crossvec(normalL, edgesL, edgesL + 3);
  mjuu_crossvec(normalR, edgesR, edgesR + 3);

  // we use as basis the symmetrized tensor products of the area normals of the
  // two faces not adjacent to the edge; this is the 3D equivalent to the basis
  // proposed in Weischedel "A discrete geometric view on shear-deformable shell
  // models" in the remark at the end of section 4.1. This is also equivalent to
  // linear finite elements but in a coordinate-free formulation.

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      basis[3 * i + j] =
          (normalL[i] * normalR[j] + normalR[i] * normalL[j]) / (36 * 2 * volume * volume);
    }
  }
}

// compute stiffness for a single element
//   thickness: shell thickness for triangles, unused for tetrahedra
template <typename T>
void inline ComputeStiffness(std::vector<double>&       stiffness,
                             const std::vector<double>& body_pos,
                             const int*                 v,
                             int                        t,
                             double                     E,
                             double                     nu,
                             double                     thickness = 1) {
  // element measure: triangle area times thickness, or tetrahedron volume
  double volume  = ComputeVolume<T>(body_pos.data(), v);
  double measure = std::abs(volume) * (T::kNumVerts == 3 ? thickness : 1);

  // Lamé parameters, plane stress for triangles (thin shells)
  double mu     = E / (2 * (1 + nu));
  double lambda = T::kNumVerts == 3 ? E * nu / (1 - nu * nu) : E * nu / ((1 + nu) * (1 - 2 * nu));

  // with s = L^2 - L0^2, sum_e s_e basis_e = 2 * (Green strain), so the energy s' K s / 4 is
  // measure * (mu * tr(strain^2) + la * tr(strain)^2): Saint Venant-Kirchhoff needs la = lambda / 2
  double mu_k = mu * measure;
  double la_k = lambda / 2 * measure;

  // local geometric quantities
  double basis[T::kNumEdges][9] = {{0}};

  // compute edge basis
  for (int e = 0; e < T::kNumEdges; e++) {
    ComputeBasis<T>(basis[e],
                    body_pos.data(),
                    v,
                    T::face[T::edge2face[e][0]],
                    T::face[T::edge2face[e][1]],
                    volume);
  }

  // compute metric tensor
  MetricTensor<T>(stiffness.data(), t, mu_k, la_k, basis, T::kNumVerts == 4 ? 24 : 21);
}

// stable Neo-Hookean quadratic metric and three signed-volume/cubic coefficients
static void ComputeSNH(std::vector<double>&       stiffness,
                       const std::vector<double>& body_pos,
                       const int*                 v,
                       int                        t,
                       double                     young,
                       double                     poisson) {
  double  volume  = ComputeVolume<Stencil3D>(body_pos.data(), v);
  double  mu      = young / (2 * (1 + poisson));
  double  lambda  = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
  double  volume0 = std::abs(volume);
  double* k       = stiffness.data() + 24 * t;

  // retain the first-fundamental-form edge basis used by the StVK formulation
  double basis[6][9];
  for (int e = 0; e < 6; e++) {
    ComputeBasis<Stencil3D>(basis[e],
                            body_pos.data(),
                            v,
                            Stencil3D::face[Stencil3D::edge2face[e][0]],
                            Stencil3D::face[Stencil3D::edge2face[e][1]],
                            volume);
  }

  // E = s' K s / 4 + gamma det(D(s)) + beta (J-1)^2, s = L^2 - L0^2.
  // K uses the same trace contractions as StVK: mu*V0*(tr(B_e B_f)-tr(B_e)tr(B_f)).
  MetricTensor<Stencil3D>(stiffness.data(), t, mu * volume0, -mu * volume0, basis, 24);
  k[21] = -mu / (72 * volume0);
  k[22] = volume0 * (lambda + 2 * mu) / 2;
  k[23] = 1 / (6 * volume);
}

// local tetrahedron numbering
constexpr int kNumEdges          = Stencil2D::kNumEdges;
constexpr int kNumVerts          = Stencil2D::kNumVerts;
constexpr int edge[kNumEdges][2] = {
    {1, 2},
    {2, 0},
    {0, 1}
};

// create map from triangles to vertices and edges and from edges to vertices
static void CreateFlapStencil(std::vector<StencilFlap>& flaps,
                              const std::vector<int>&   simplex,
                              const std::vector<int>&   edgeidx) {
  // populate stencil
  int                    ne = 0;
  int                    nt = simplex.size() / kNumVerts;
  std::vector<Stencil2D> elements(nt);
  for (int t = 0; t < nt; t++) {
    for (int v = 0; v < kNumVerts; v++) { elements[t].vertices[v] = simplex[kNumVerts * t + v]; }
  }

  // map from edge vertices to their index in `edges` vector
  std::unordered_map<std::pair<int, int>, int, PairHash> edge_indices;

  // loop over all triangles
  for (int t = 0; t < nt; t++) {
    int* v = elements[t].vertices;

    // compute edges to vertices map for fast computations
    for (int e = 0; e < kNumEdges; e++) {
      auto pair =
          std::pair(std::min(v[edge[e][0]], v[edge[e][1]]), std::max(v[edge[e][0]], v[edge[e][1]]));

      // if edge is already present in the vector only store its index
      auto [it, inserted] = edge_indices.insert({pair, ne});

      if (inserted) {
        StencilFlap flap;
        // store the edge vertices in the same (min, max) order as the edge
        // pair: the engine applies the bending coefficients, which depend on
        // this order, to the vertices listed in flex_edge
        flap.vertices[0] = pair.first;
        flap.vertices[1] = pair.second;
        flap.vertices[2] = v[(edge[e][1] + 1) % 3];
        flap.vertices[3] = -1;
        flaps.push_back(flap);
        elements[t].edges[e] = ne++;
      } else {
        elements[t].edges[e]          = it->second;
        flaps[it->second].vertices[3] = v[(edge[e][1] + 1) % 3];
      }

      // double check that the edge indices are consistent
      if (!edgeidx.empty()) {
        if (elements[t].edges[e] != edgeidx[kNumEdges * t + e]) {
          mju_error("edge indices do not match in CreateFlapStencil");
        }
      }
    }
  }
}

// cotangent between two edges
double inline cot(const double* x, int v0, int v1, int v2) {
  double normal[3];
  double edge1[3] = {x[3 * v1] - x[3 * v0],
                     x[3 * v1 + 1] - x[3 * v0 + 1],
                     x[3 * v1 + 2] - x[3 * v0 + 2]};
  double edge2[3] = {x[3 * v2] - x[3 * v0],
                     x[3 * v2 + 1] - x[3 * v0 + 1],
                     x[3 * v2 + 2] - x[3 * v0 + 2]};

  mjuu_crossvec(normal, edge1, edge2);
  return mjuu_dot3(edge1, edge2) / sqrt(mjuu_dot3(normal, normal));
}

// area of a triangle
double inline ComputeVolume(const double* x, const int v[Stencil2D::kNumVerts]) {
  double normal[3];
  double edge1[3] = {x[3 * v[1]] - x[3 * v[0]],
                     x[3 * v[1] + 1] - x[3 * v[0] + 1],
                     x[3 * v[1] + 2] - x[3 * v[0] + 2]};
  double edge2[3] = {x[3 * v[2]] - x[3 * v[0]],
                     x[3 * v[2] + 1] - x[3 * v[0] + 1],
                     x[3 * v[2] + 2] - x[3 * v[0] + 2]};

  mjuu_crossvec(normal, edge1, edge2);
  return sqrt(mjuu_dot3(normal, normal)) / 2;
}

// compute bending stiffness for a single edge
//   rigidity: plate bending rigidity D = E t^3 / (12 (1 - nu^2))
template <typename T>
void inline ComputeBending(double* bending, double* pos, const int v[4], double rigidity) {
  int vadj[3] = {v[1], v[0], v[3]};

  if (v[3] == -1) {
    // skip boundary edges
    return;
  }

  // cotangent operator from Wardetzky at al., "Discrete Quadratic Curvature
  // Energies", https://cims.nyu.edu/gcl/papers/wardetzky2007dqb.pdf

  double a01    = cot(pos, v[0], v[1], v[2]);
  double a02    = cot(pos, v[0], v[3], v[1]);
  double a03    = cot(pos, v[1], v[2], v[0]);
  double a04    = cot(pos, v[1], v[0], v[3]);
  double c[4]   = {a03 + a04, a01 + a02, -(a01 + a03), -(a02 + a04)};
  double volume = ComputeVolume(pos, v) + ComputeVolume(pos, vadj);

  // normalized so that bending a flat sheet into a cylinder of curvature k (an isometry) has
  // energy density D k^2 / 2, as in Kirchhoff plate theory
  double stiffness = rigidity / volume;

  // Garg et al., "Cubic Shells", https://cims.nyu.edu/gcl/papers/garg2007cs.pdf
  const double* v0        = pos + 3 * v[0];
  const double* v1        = pos + 3 * v[1];
  const double* v2        = pos + 3 * v[2];
  const double* v3        = pos + 3 * v[3];
  double        e0[3]     = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
  double        e1[3]     = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
  double        e2[3]     = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};
  double        e3[3]     = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double        e4[3]     = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};
  double        t0[3]     = {-(a03 * e1[0] + a01 * e3[0]),
                             -(a03 * e1[1] + a01 * e3[1]),
                             -(a03 * e1[2] + a01 * e3[2])};
  double        t1[3]     = {-(a04 * e2[0] + a02 * e4[0]),
                             -(a04 * e2[1] + a02 * e4[1]),
                             -(a04 * e2[2] + a02 * e4[2])};
  double        sqr       = mjuu_dot3(e0, e0);
  double        cos_theta = -mjuu_dot3(t0, t1) / sqr;

  for (int v1 = 0; v1 < T::kNumVerts; v1++) {
    for (int v2 = 0; v2 < T::kNumVerts; v2++) {
      bending[4 * v1 + v2] += c[v1] * c[v2] * cos_theta * stiffness;
    }
  }

  double n[3];
  mjuu_crossvec(n, e0, e1);
  bending[16] = mjuu_dot3(n, e2) * (a01 - a03) * (a04 - a02) * stiffness / (sqr * sqrt(sqr));
}

// bend angle tolerance of a straight 1D flex vertex in the reference configuration
static constexpr double kRodStraightTol = 1e-4;

// curvature binormal kb = 2 e0 x e1 / (|e0||e1| + e0.e1) of hinge (xa, xv, xb), with e0 = xv - xa
// and e1 = xb - xv, and optionally the edge lengths; returns false if the hinge is degenerate
static bool RodCurvatureBinormal(const double* xa,
                                 const double* xv,
                                 const double* xb,
                                 double        kb[3],
                                 double*       l0_out,
                                 double*       l1_out) {
  double e0[3] = {xv[0] - xa[0], xv[1] - xa[1], xv[2] - xa[2]};
  double e1[3] = {xb[0] - xv[0], xb[1] - xv[1], xb[2] - xv[2]};
  double l0    = sqrt(mjuu_dot3(e0, e0));
  double l1    = sqrt(mjuu_dot3(e1, e1));
  double denom = l0 * l1 + mjuu_dot3(e0, e1);
  if (l0 < mjMINVAL || l1 < mjMINVAL || denom < mjMINVAL) { return false; }
  double cross[3];
  mjuu_crossvec(cross, e0, e1);
  double scl = 2.0 / denom;
  kb[0]      = scl * cross[0];
  kb[1]      = scl * cross[1];
  kb[2]      = scl * cross[2];
  if (l0_out) { *l0_out = l0; }
  if (l1_out) { *l1_out = l1; }
  return true;
}

// compute isotropic rod bending and torsion data for a 1D flex, following Bergou et al., "Discrete
// Elastic Rods", https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf. Every vertex shared by exactly
// two edges is a hinge with bending stiffness k = EI / (|e_prev| + |e_next|); vertices with other
// valences carry no bending energy.
//   EI: bending rigidity, Young's modulus times the second moment of area of the cross-section
//   GJ: torsional rigidity, shear modulus times the torsion constant of the cross-section
// Hinges bent by more than kRodStraightTol in the reference configuration are curved. Adjacent
// curved hinges are stored as a pair across every edge they share, splitting the bending stiffness
// of each hinge evenly among its pairs and adding dihedral torsion stiffness kt = GJ / (2 |e|)
// across the shared edge e; all other hinges are stored singly.
//   layout: bending[0] = number of records, then per record
//           single hinge: [v_prev, v, v_next, -1, k, 0, |kb0|, 0, 0, 0], |kb0| = 0 if straight
//           hinge pair:   [v0, v1, v2, v3, k1 / n1, k2 / n2, |kb0_1|, |kb0_2|, kt cos(dphi0),
//                         kt sin(dphi0)] for hinges v1 and v2, with n the number of pairs of a
//                         hinge and dphi0 the reference dihedral angle between kb0_1 and kb0_2
//                         about the shared edge (v1, v2)
// returns 1 on success, or 0 with the id of the first degenerate vertex in *bad
static int ComputeRodBending(std::vector<double>&                    bending,
                             const double*                           pos,
                             int                                     nvert,
                             const std::vector<std::pair<int, int>>& edges,
                             double                                  EI1,
                             double                                  EI2,
                             double                                  GJ,
                             double                                  psi0,
                             int                                     twist_jnt,
                             int*                                    bad) {
  *bad = -1;

  // vertex adjacency
  std::vector<std::vector<int>> neighbors(nvert);
  for (const auto& e : edges) {
    neighbors[e.first].push_back(e.second);
    neighbors[e.second].push_back(e.first);
  }

  // hinge stiffness and reference curvature
  std::vector<double> stiffness(nvert, 0);
  std::vector<double> dstiffness(nvert, 0);
  std::vector<double> torsion(nvert, 0);
  std::vector<double> curvature(nvert, 0);
  for (int v = 0; v < nvert; v++) {
    if (neighbors[v].size() != 2) { continue; }
    const double* xa = pos + 3 * neighbors[v][0];
    const double* xv = pos + 3 * v;
    const double* xb = pos + 3 * neighbors[v][1];
    double        kb[3], l0, l1;
    if (!RodCurvatureBinormal(xa, xv, xb, kb, &l0, &l1)) {
      *bad = v;
      return 0;
    }
    stiffness[v]  = EI1 / (l0 + l1);
    dstiffness[v] = (EI2 - EI1) / (l0 + l1);
    torsion[v]    = GJ / (l0 + l1);

    // reference curvature binormal magnitude 2*tan(phi/2), approximately the bend angle phi
    double norm = sqrt(mjuu_dot3(kb, kb));
    if (norm > kRodStraightTol) { curvature[v] = norm; }
  }

  // number of pairs of each hinge (only for isotropic cross-sections; anisotropic chains stay as
  // single hinges for the Bishop-frame tridiagonal twist solve)
  std::vector<int> npair(nvert, 0);
  if (EI1 == EI2) {
    for (const auto& e : edges) {
      if (curvature[e.first] > 0 && curvature[e.second] > 0) {
        npair[e.first]++;
        npair[e.second]++;
      }
    }
  }

  // hinge pairs across edges shared by curved hinges
  bending.assign(1, 0);
  for (const auto& e : edges) {
    if (EI1 != EI2) { break; }
    int v1 = e.first;
    int v2 = e.second;
    if (curvature[v1] == 0 || curvature[v2] == 0) { continue; }
    int    v0 = neighbors[v1][0] == v2 ? neighbors[v1][1] : neighbors[v1][0];
    int    v3 = neighbors[v2][0] == v1 ? neighbors[v2][1] : neighbors[v2][0];
    double kb1[3], kb2[3], l1;
    RodCurvatureBinormal(pos + 3 * v0, pos + 3 * v1, pos + 3 * v2, kb1, nullptr, &l1);
    RodCurvatureBinormal(pos + 3 * v1, pos + 3 * v2, pos + 3 * v3, kb2, nullptr, nullptr);
    double t[3] = {pos[3 * v2 + 0] - pos[3 * v1 + 0],
                   pos[3 * v2 + 1] - pos[3 * v1 + 1],
                   pos[3 * v2 + 2] - pos[3 * v1 + 2]};
    mjuu_normvec(t, 3);
    double r0a = curvature[v1];
    double r0b = curvature[v2];
    double cross[3];
    mjuu_crossvec(cross, kb1, kb2);
    double cos_dphi0 = mjuu_dot3(kb1, kb2) / (r0a * r0b);
    double sin_dphi0 = mjuu_dot3(cross, t) / (r0a * r0b);
    double kt        = GJ / (2 * l1);

    bending.push_back(v0);
    bending.push_back(v1);
    bending.push_back(v2);
    bending.push_back(v3);
    bending.push_back(stiffness[v1] / npair[v1]);
    bending.push_back(stiffness[v2] / npair[v2]);
    bending.push_back(r0a);
    bending.push_back(r0b);
    bending.push_back(kt * cos_dphi0);
    bending.push_back(kt * sin_dphi0);
    bending[0]++;
  }

  // single hinges: straight, or curved without curved neighbors
  for (int v = 0; v < nvert; v++) {
    if (neighbors[v].size() != 2 || npair[v] > 0) { continue; }
    bending.push_back(neighbors[v][0]);
    bending.push_back(v);
    bending.push_back(neighbors[v][1]);
    bending.push_back(-1);
    bending.push_back(stiffness[v]);
    bending.push_back(dstiffness[v]);
    bending.push_back(curvature[v]);
    bending.push_back(twist_jnt);
    bending.push_back(torsion[v]);
    bending.push_back(psi0);
    bending[0]++;
  }
  return 1;
}

//----------------------------- linear elasticity --------------------------------------------------

// Gauss Legendre quadrature points in 1 dimension on the interval [a, b]
void quadratureGaussLegendre(
    double* points, double* weights, const int order, const double a, const double b) {
  // x is on [-1, 1], p on [a, b]
  double p0   = (a + b) / 2.;
  double dpdx = (b - a) / 2;

  if (order == 1) {
    points[0]  = p0;
    weights[0] = 2.0 * dpdx;
  } else if (order == 2) {
    points[0]  = -dpdx / sqrt(3) + p0;
    points[1]  = dpdx / sqrt(3) + p0;
    weights[0] = dpdx;
    weights[1] = dpdx;
  } else if (order == 3) {
    points[0]  = p0;
    points[1]  = -dpdx * sqrt(3. / 5.) + p0;
    points[2]  = dpdx * sqrt(3. / 5.) + p0;
    weights[0] = 8. / 9. * dpdx;
    weights[1] = 5. / 9. * dpdx;
    weights[2] = 5. / 9. * dpdx;
  } else if (order == 4) {
    double s30 = sqrt(30.0);
    double x1  = sqrt((3.0 - 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    double x2  = sqrt((3.0 + 2.0 * sqrt(6.0 / 5.0)) / 7.0);
    double w1  = (18.0 + s30) / 36.0 * dpdx;
    double w2  = (18.0 - s30) / 36.0 * dpdx;
    points[0]  = -dpdx * x2 + p0;
    points[1]  = -dpdx * x1 + p0;
    points[2]  = dpdx * x1 + p0;
    points[3]  = dpdx * x2 + p0;
    weights[0] = w2;
    weights[1] = w1;
    weights[2] = w1;
    weights[3] = w2;
  } else if (order == 5) {
    double s70 = sqrt(70.0);
    double x1  = sqrt(5.0 - 2.0 * sqrt(10.0 / 7.0)) / 3.0;
    double x2  = sqrt(5.0 + 2.0 * sqrt(10.0 / 7.0)) / 3.0;
    double w0  = (128.0 / 225.0) * dpdx;
    double w1  = (322.0 + 13.0 * s70) / 900.0 * dpdx;
    double w2  = (322.0 - 13.0 * s70) / 900.0 * dpdx;
    points[0]  = p0;
    points[1]  = -dpdx * x1 + p0;
    points[2]  = dpdx * x1 + p0;
    points[3]  = -dpdx * x2 + p0;
    points[4]  = dpdx * x2 + p0;
    weights[0] = w0;
    weights[1] = w1;
    weights[2] = w1;
    weights[3] = w2;
    weights[4] = w2;
  } else if (order == 8) {
    static constexpr double kX[4] = {
        0.183434642495649805,
        0.525532409916328986,
        0.796666477413626740,
        0.960289856497536232,
    };
    static constexpr double kW[4] = {
        0.362683783378361983,
        0.313706645877887287,
        0.222381034453374471,
        0.101228536290376259,
    };
    for (int i = 0; i < 4; i++) {
      points[2 * i + 0]  = -dpdx * kX[i] + p0;
      points[2 * i + 1]  = dpdx * kX[i] + p0;
      weights[2 * i + 0] = kW[i] * dpdx;
      weights[2 * i + 1] = kW[i] * dpdx;
    }
  } else {
    mju_error("Integration order > 3 not yet supported.");
  }
}

// evaluate 1-dimensional basis function
double phi(const double s, const int i, const int order) {
  if (order == 1) {
    return i == 0 ? 1 - s : s;
  } else if (order == 2) {
    switch (i) {
      case 0:
        return 2 * s * s - 3 * s + 1;
      case 1:
        return 4 * (s - s * s);
      case 2:
        return 2 * s * s - s;
      default:
        mjERROR("invalid index %d", i);
        return 0;
    }
  } else {
    mju_error("Order must be 1 or 2.");
    return 0;
  }
}

// evaluate gradient of 1-dimensional basis function
double dphi(const double s, const int i, const int order) {
  if (order == 1) {
    return i == 0 ? -1 : 1;
  } else if (order == 2) {
    switch (i) {
      case 0:
        return 4 * s - 3;
      case 1:
        return 4 * (1 - 2 * s);
      case 2:
        return 4 * s - 1;
      default:
        mjERROR("invalid index %d, must be 0, 1, or 2", i);
        return 0;
    }
  } else {
    mju_error("Order must be 1 or 2.");
    return 0;
  }
}

typedef std::array<std::array<double, 3>, 3> Matrix;

// symmetrize a tensor
Matrix inline sym(const Matrix& tensor) {
  Matrix eps;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) { eps[i][j] = (tensor[i][j] + tensor[j][i]) / 2; }
  }
  return eps;
}

// compute tensor inner product
Matrix inline inner(const Matrix& tensor1, const Matrix& tensor2) {
  Matrix inner;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      inner[i][j] = tensor1[i][0] * tensor2[0][j] +
                    tensor1[i][1] * tensor2[1][j] +
                    tensor1[i][2] * tensor2[2][j];
    }
  }
  return inner;
}

// compute trace of a tensor
double inline trace(const Matrix& tensor) {
  return tensor[0][0] + tensor[1][1] + tensor[2][2];
}


// Finite Cell Method (FCM) subcell quadrature parameters
constexpr int    kFcmSubdiv   = 2;
constexpr double kFcmAlphaMin = 1e-6;
// minimum fraction of the full-cell nodal mass weight (w_full) assigned to
// each node of an active cut cell: prevents near-zero nodal mass and inertia
// and bounds the generalized eigenvalue ratio K_ii / m_i on sliver cells
constexpr double kFcmMassMin = 2e-2;


// compute the linear stiffness matrix for a 3D hexahedral element
void inline ComputeLinearStiffness(std::vector<double>&       K,
                                   const double*              pos,
                                   double                     E,
                                   double                     nu,
                                   int                        order,
                                   const std::vector<double>* alphas = nullptr) {
  int nbasis = order + 1;
  int n      = pow(nbasis, 3);
  int ndof   = 3 * n;

  // compute element transformation
  double dx      = (pos + 3 * (n - 1))[0] - pos[0];
  double dy      = (pos + 3 * (n - 1))[1] - pos[1];
  double dz      = (pos + 3 * (n - 1))[2] - pos[2];
  double detJ    = dx * dy * dz;
  double invJ[3] = {1.0 / dx, 1.0 / dy, 1.0 / dz};

  // compute stiffness matrix
  std::vector<std::array<double, 3>> F(n);
  double                             la = E * nu / (1 + nu) / (1 - 2 * nu);
  double                             mu = E / (2 * (1 + nu));

  if (alphas && !alphas->empty()) {
    int                 nq_axis = kFcmSubdiv * nbasis;
    std::vector<double> points(nq_axis);
    std::vector<double> weight(nq_axis);
    for (int sub = 0; sub < kFcmSubdiv; sub++) {
      quadratureGaussLegendre(points.data() + sub * nbasis,
                              weight.data() + sub * nbasis,
                              nbasis,
                              static_cast<double>(sub) / kFcmSubdiv,
                              static_cast<double>(sub + 1) / kFcmSubdiv);
    }

    // precompute 1D basis function values and derivatives at quadrature points
    std::vector<double> phi_tab(nbasis * nq_axis);
    std::vector<double> dphi_tab(nbasis * nq_axis);
    for (int b = 0; b < nbasis; b++) {
      for (int p = 0; p < nq_axis; p++) {
        phi_tab[b * nq_axis + p]  = phi(points[p], b, order);
        dphi_tab[b * nq_axis + p] = dphi(points[p], b, order);
      }
    }

    // accumulate integrated gradient outer products G[i * n + j][a][b] for j >=
    // i
    std::vector<Matrix> G(n * n);
    for (auto& m : G) { m.fill({0, 0, 0}); }

    int q_idx = 0;
    for (int ps = 0; ps < nq_axis; ps++) {
      for (int pt = 0; pt < nq_axis; pt++) {
        for (int pu = 0; pu < nq_axis; pu++) {
          double alpha = (*alphas)[q_idx++];
          double dvol  = weight[ps] * weight[pt] * weight[pu] * detJ * alpha;
          int    dof   = 0;

          for (int bx = 0; bx < nbasis; bx++) {
            double ph_s  = phi_tab[bx * nq_axis + ps];
            double dph_s = invJ[0] * dphi_tab[bx * nq_axis + ps];
            for (int by = 0; by < nbasis; by++) {
              double ph_t  = phi_tab[by * nq_axis + pt];
              double dph_t = invJ[1] * dphi_tab[by * nq_axis + pt];
              double st_x  = dph_s * ph_t;
              double st_y  = ph_s * dph_t;
              double st_z  = ph_s * ph_t * invJ[2];
              for (int bz = 0; bz < nbasis; bz++) {
                double ph_u  = phi_tab[bz * nq_axis + pu];
                double dph_u = dphi_tab[bz * nq_axis + pu];
                F[dof++]     = {st_x * ph_u, st_y * ph_u, st_z * dph_u};
              }
            }
          }

          for (int i = 0; i < n; i++) {
            double f0 = dvol * F[i][0];
            double f1 = dvol * F[i][1];
            double f2 = dvol * F[i][2];
            for (int j = i; j < n; j++) {
              Matrix& gij  = G[i * n + j];
              gij[0][0]   += f0 * F[j][0];
              gij[0][1]   += f0 * F[j][1];
              gij[0][2]   += f0 * F[j][2];
              gij[1][0]   += f1 * F[j][0];
              gij[1][1]   += f1 * F[j][1];
              gij[1][2]   += f1 * F[j][2];
              gij[2][0]   += f2 * F[j][0];
              gij[2][1]   += f2 * F[j][1];
              gij[2][2]   += f2 * F[j][2];
            }
          }
        }
      }
    }

    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) {
        const Matrix& gij = G[i * n + j];
        Matrix&       gji = G[j * n + i];
        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) { gji[a][b] = gij[b][a]; }
        }
      }
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        const Matrix& gij   = G[i * n + j];
        double        dot_g = mu * (gij[0][0] + gij[1][1] + gij[2][2]);
        for (int k = 0; k < 3; k++) {
          for (int l = 0; l < 3; l++) {
            double val = la * gij[k][l] + mu * gij[l][k] + (k == l ? dot_g : 0.0);
            K[ndof * (3 * i + k) + 3 * j + l] -= val;
          }
        }
      }
    }
    return;
  }

  // compute quadrature points for full cell
  std::vector<double> points(nbasis);  // quadrature points
  std::vector<double> weight(nbasis);  // quadrature weights
  quadratureGaussLegendre(points.data(), weight.data(), nbasis, 0, 1);

  // loop over quadrature points
  for (int ps = 0; ps < nbasis; ps++) {
    for (int pt = 0; pt < nbasis; pt++) {
      for (int pu = 0; pu < nbasis; pu++) {
        double s    = points[ps];
        double t    = points[pt];
        double u    = points[pu];
        double dvol = weight[ps] * weight[pt] * weight[pu] * detJ;
        int    dof  = 0;

        // cartesian product of basis functions
        for (int bx = 0; bx < nbasis; bx++) {
          for (int by = 0; by < nbasis; by++) {
            for (int bz = 0; bz < nbasis; bz++) {
              std::array<double, 3> gradient;
              gradient[0] = dphi(s, bx, order) * phi(t, by, order) * phi(u, bz, order);
              gradient[1] = phi(s, bx, order) * dphi(t, by, order) * phi(u, bz, order);
              gradient[2] = phi(s, bx, order) * phi(t, by, order) * dphi(u, bz, order);
              F[dof++]    = gradient;
            }
          }
        }

        if (dof != n) {  // SHOULD NOT OCCUR
          throw mjCError(nullptr, "incorrect number of basis functions");
        }

        // tensor contraction of the gradients of elastic strains
        // lambda * div(u) * div(v) + 2*mu * sym(grad(u)) : sym(grad(v))
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            Matrix du;
            Matrix dv;
            du.fill({0, 0, 0});
            dv.fill({0, 0, 0});
            for (int k = 0; k < 3; k++) {
              for (int l = 0; l < 3; l++) {
                du[k][0]                           = invJ[0] * F[i][0];
                du[k][1]                           = invJ[1] * F[i][1];
                du[k][2]                           = invJ[2] * F[i][2];
                dv[l][0]                           = invJ[0] * F[j][0];
                dv[l][1]                           = invJ[1] * F[j][1];
                dv[l][2]                           = invJ[2] * F[j][2];
                K[ndof * (3 * i + k) + 3 * j + l] -= la * trace(du) * trace(dv) * dvol;
                K[ndof * (3 * i + k) + 3 * j + l] -= 2 * mu * trace(inner(sym(du), sym(dv))) * dvol;
                mjuu_zerovec(du[k].data(), 3);
                mjuu_zerovec(dv[l].data(), 3);
              }
            }
          }
        }
      }
    }
  }
}


// compute the linear stiffness matrix for a flat 2D quad face element (membrane)
//   K:      output stiffness matrix, size 3*npe x 3*npe, npe = (order+1)^2
//   pos:    node positions (3*npe doubles), ordered row-major in 2D parametric domain
//   E, nu:  Young's modulus and Poisson's ratio
//   order:  interpolation order (1 or 2)
//   thickness: shell thickness
//   normal_axis: axis perpendicular to the face (0=x, 1=y, 2=z)
void inline ComputeLinearStiffness2D(std::vector<double>& K,
                                     const double*        pos,
                                     double               E,
                                     double               nu,
                                     int                  order,
                                     double               thickness,
                                     int                  normal_axis) {
  int nbasis = order + 1;
  int npe    = nbasis * nbasis;  // nodes per face element
  int ndof   = 3 * npe;

  // in-plane axes
  int axis0 = (normal_axis + 1) % 3;  // slow-varying
  int axis1 = (normal_axis + 2) % 3;  // fast-varying

  // compute quadrature points
  std::vector<double> points(nbasis);
  std::vector<double> weight(nbasis);
  quadratureGaussLegendre(points.data(), weight.data(), nbasis, 0, 1);

  // compute element transformation (diagonal Jacobian on flat face)
  double d0 = (pos + 3 * (npe - 1))[axis0] - pos[axis0];  // extent along axis0
  double d1 = (pos + 3 * (npe - 1))[axis1] - pos[axis1];  // extent along axis1
  if (d0 == 0 || d1 == 0) { throw mjCError(nullptr, "degenerate 2D element with zero extent"); }
  double detJ  = d0 * d1;
  double invJ0 = 1.0 / d0;
  double invJ1 = 1.0 / d1;

  // plane-stress Lamé parameter: lambda* = E*nu/(1 - nu^2)
  double la = E * nu / (1.0 - nu * nu);
  double mu = E / (2.0 * (1.0 + nu));

  // basis function gradients (2-component)
  std::vector<std::array<double, 2>> F(npe);

  // loop over quadrature points (2D)
  for (int ps = 0; ps < nbasis; ps++) {
    for (int pt = 0; pt < nbasis; pt++) {
      double s    = points[ps];
      double t    = points[pt];
      double dvol = weight[ps] * weight[pt] * detJ * thickness;
      int    dof  = 0;

      // cartesian product of 2D basis functions
      for (int b0 = 0; b0 < nbasis; b0++) {
        for (int b1 = 0; b1 < nbasis; b1++) {
          F[dof][0] = dphi(s, b0, order) * phi(t, b1, order);
          F[dof][1] = phi(s, b0, order) * dphi(t, b1, order);
          dof++;
        }
      }

      if (dof != npe) { throw mjCError(nullptr, "incorrect number of 2D basis functions"); }

      // tensor contraction: pure membrane (in-plane strain only)
      // only loop over in-plane displacement directions to avoid transverse
      // shear strains (ε_{normal,α}) which are spurious for thin shells
      int inplane[2] = {axis0, axis1};
      for (int i = 0; i < npe; i++) {
        for (int j = 0; j < npe; j++) {
          Matrix du;
          Matrix dv;
          du.fill({0, 0, 0});
          dv.fill({0, 0, 0});
          for (int ki = 0; ki < 2; ki++) {
            int k = inplane[ki];
            for (int li = 0; li < 2; li++) {
              int l        = inplane[li];
              du[k][axis0] = invJ0 * F[i][0];
              du[k][axis1] = invJ1 * F[i][1];
              dv[l][axis0] = invJ0 * F[j][0];
              dv[l][axis1] = invJ1 * F[j][1];

              K[ndof * (3 * i + k) + 3 * j + l] -= la * trace(du) * trace(dv) * dvol;
              K[ndof * (3 * i + k) + 3 * j + l] -= 2 * mu * trace(inner(sym(du), sym(dv))) * dvol;
              mjuu_zerovec(du[k].data(), 3);
              mjuu_zerovec(dv[l].data(), 3);
            }
          }
        }
      }
    }
  }
}


// compute the bilinear warp mode for a 2D face element
//   warp:        output mode vector (ndof doubles), normalized to unit length
//   pos:         node positions (3*npe doubles)
//   npe:         nodes per element ((order+1)^2)
//   order:       interpolation order (1 or 2)
//   normal_axis: axis perpendicular to the face (0=x, 1=y, 2=z)
static void ComputeWarpMode(double* warp, const double* pos, int npe, int order, int normal_axis) {
  int ndof   = 3 * npe;
  int nbasis = order + 1;

  // zero out
  std::fill(warp, warp + ndof, 0.0);

  // evaluate warp pattern (1-2s)(1-2t) at each node
  for (int b0 = 0; b0 < nbasis; b0++) {
    for (int b1 = 0; b1 < nbasis; b1++) {
      int    node = b0 * nbasis + b1;
      double s    = static_cast<double>(b0) / (nbasis - 1);
      double t    = static_cast<double>(b1) / (nbasis - 1);

      warp[3 * node + normal_axis] = (1 - 2 * s) * (1 - 2 * t);
    }
  }

  // orthogonalize against rigid body modes (6 modes: 3 translations + 3 rotations)
  // this is a no-op for rectangular elements (warp is already orthogonal)
  // but keeps the code robust for non-square elements
  double centroid[3] = {0, 0, 0};
  for (int n = 0; n < npe; n++) {
    for (int k = 0; k < 3; k++) { centroid[k] += pos[3 * n + k]; }
  }
  for (int k = 0; k < 3; k++) { centroid[k] /= npe; }

  // build and orthonormalize rigid body modes inline
  std::vector<double> rigid(6 * ndof, 0.0);

  // translations
  for (int n = 0; n < npe; n++) {
    rigid[0 * ndof + 3 * n + 0] = 1;
    rigid[1 * ndof + 3 * n + 1] = 1;
    rigid[2 * ndof + 3 * n + 2] = 1;
  }

  // rotations about centroid
  for (int n = 0; n < npe; n++) {
    double rx = pos[3 * n + 0] - centroid[0];
    double ry = pos[3 * n + 1] - centroid[1];
    double rz = pos[3 * n + 2] - centroid[2];

    rigid[3 * ndof + 3 * n + 1] = -rz;
    rigid[3 * ndof + 3 * n + 2] = ry;
    rigid[4 * ndof + 3 * n + 0] = rz;
    rigid[4 * ndof + 3 * n + 2] = -rx;
    rigid[5 * ndof + 3 * n + 0] = -ry;
    rigid[5 * ndof + 3 * n + 1] = rx;
  }

  // orthonormalize rigid modes via modified Gram-Schmidt
  for (int i = 0; i < 6; i++) {
    double* ri = rigid.data() + i * ndof;
    for (int j = 0; j < i; j++) {
      const double* rj  = rigid.data() + j * ndof;
      double        dot = 0;
      for (int k = 0; k < ndof; k++) dot += ri[k] * rj[k];
      for (int k = 0; k < ndof; k++) ri[k] -= dot * rj[k];
    }
    double norm2 = 0;
    for (int k = 0; k < ndof; k++) norm2 += ri[k] * ri[k];
    if (norm2 > 1e-20) {
      double inv_norm = 1.0 / std::sqrt(norm2);
      for (int k = 0; k < ndof; k++) ri[k] *= inv_norm;
    }
  }

  // project warp against rigid modes
  for (int i = 0; i < 6; i++) {
    const double* ri  = rigid.data() + i * ndof;
    double        dot = 0;
    for (int k = 0; k < ndof; k++) dot += warp[k] * ri[k];
    for (int k = 0; k < ndof; k++) warp[k] -= dot * ri[k];
  }

  // normalize
  double norm2 = 0;
  for (int k = 0; k < ndof; k++) norm2 += warp[k] * warp[k];
  if (norm2 > 1e-20) {
    double inv_norm = 1.0 / std::sqrt(norm2);
    for (int k = 0; k < ndof; k++) warp[k] *= inv_norm;
  }
}


// compute the warp bending stiffness for a 2D face element
// uses plate bending theory: the warp mode is a pure twist (κ_xy),
// with bending stiffness proportional to t³ (no shear locking)
//   pos:          node positions (3*npe doubles)
//   npe:          nodes per element
//   normal_axis:  axis perpendicular to the face
//   E, nu:        Young's modulus and Poisson's ratio
//   thickness:    shell thickness
static double ComputeWarpStiffness(
    const double* pos, int npe, int normal_axis, double E, double nu, double thickness) {
  int    axis0 = (normal_axis + 1) % 3;
  int    axis1 = (normal_axis + 2) % 3;
  double d0    = std::abs(pos[3 * (npe - 1) + axis0] - pos[axis0]);
  double d1    = std::abs(pos[3 * (npe - 1) + axis1] - pos[axis1]);

  if (d0 < 1e-30 || d1 < 1e-30) return 0;

  // plate bending rigidity: D = E*t³ / (12*(1-ν²))
  double D = E * thickness * thickness * thickness / (12.0 * (1.0 - nu * nu));

  // warp stiffness from twist curvature Rayleigh quotient:
  //   w^T K_bend w / |w|^2 = D*(1-ν)*4 / (d0*d1)
  return D * (1.0 - nu) * 4.0 / (d0 * d1);
}


// Eigendecompose cell stiffness matrix and store scaled eigenvectors.
// K_cell is n×n stored (negative convention: K_stored = -K_physical).
// Output layout in `out`:
//   [0]: neig (as double)
//   [1 .. neig*n]: sqrt(λ_phys_i) * v_i, row-major
// Modes with eigenvalue below a relative threshold are discarded (rigid body
// modes and numerical zeros).
// Returns number of retained eigenmodes.
static int EigendecomposeStiffness(const double* K_cell_data, double* out, int ndof) {
  // copy K_cell for in-place decomposition
  std::vector<double> mat(K_cell_data, K_cell_data + ndof * ndof);
  std::vector<double> eigval(ndof);
  std::vector<double> eigvec(ndof * ndof);

  mjuu_eigendecompose(mat.data(), eigval.data(), eigvec.data(), ndof);

  // K_stored = -K_physical, so physical eigenvalue = -eigval[i]
  // retain modes where physical eigenvalue > threshold
  double max_eigval = 0;
  for (int i = 0; i < ndof; i++) { max_eigval = std::max(max_eigval, std::abs(eigval[i])); }

  double threshold = max_eigval * 1e-8;
  int    neig      = 0;
  for (int i = 0; i < ndof; i++) {
    double lambda_phys = -eigval[i];  // negate to get physical eigenvalue
    if (lambda_phys > threshold) {
      // store sqrt(λ) * eigenvector (column i of eigvec matrix)
      double  scale = std::sqrt(lambda_phys);
      double* w     = out + 1 + neig * ndof;
      for (int j = 0; j < ndof; j++) { w[j] = scale * eigvec[j * ndof + i]; }
      neig++;
    }
  }

  out[0] = static_cast<double>(neig);
  return neig;
}


//------------------ class mjCFlex implementation --------------------------------------------------

// constructor
mjCFlex::mjCFlex(mjCModel* _model) {
  mjs_defaultFlex(&spec);
  elemtype = mjOBJ_FLEX;

  // set model
  model = _model;
  if (_model) compiler = &_model->spec.compiler;

  // clear internal variables
  nvert       = 0;
  nnode       = 0;
  nedge       = 0;
  nelem       = 0;
  plasticadr_ = -1;
  plasticnum_ = 0;
  rigid       = false;
  centered    = false;

  PointToLocal();
  CopyFromSpec();
}


mjCFlex::mjCFlex(const mjCFlex& other) {
  *this = other;
}


mjCFlex& mjCFlex::operator=(const mjCFlex& other) {
  if (this != &other) {
    this->spec                    = other.spec;
    *static_cast<mjCFlex_*>(this) = static_cast<const mjCFlex_&>(other);
    *static_cast<mjsFlex*>(this)  = static_cast<const mjsFlex&>(other);
    plasticadr_                   = -1;
    plasticnum_                   = 0;
  }
  PointToLocal();
  return *this;
}


void mjCFlex::ResetId() {
  id          = -1;
  plasticadr_ = -1;
  plasticnum_ = 0;
}


void mjCFlex::ForgetKeyframes(const std::vector<std::string>& names, bool keep) {
  for (auto it = plastic_.begin(); it != plastic_.end();) {
    bool found = std::find(names.begin(), names.end(), it->first) != names.end();
    if (found != keep) {
      it = plastic_.erase(it);
    } else {
      ++it;
    }
  }
}

int mjCFlex::PlasticNum() const {
  bool compiled  = !stiffness.empty() || !bending.empty();
  bool is_interp = compiled ? interpolated : !spec_nodebody_.empty();
  if (is_interp) {
    double yld  = compiled ? stressyield : spec.stressyield;
    double yng  = compiled ? young : spec.young;
    int    el2d = compiled ? elastic2d : spec.elastic2d;
    double thk  = compiled ? thickness : spec.thickness;
    bool   rgd  = compiled ? rigid : (!spec_vert_.empty() && spec_vertbody_.size() == 1);
    if (rgd || yld <= 0 || yng <= 0 || has_strain_eq) { return 0; }
    if (model) {
      for (auto* equality : model->Equalities()) {
        if (equality->spec.type == mjEQ_FLEXSTRAIN && *equality->spec.name1 == name) { return 0; }
      }
    }
    int cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
    if (cx <= 0 || cy <= 0 || cz <= 0) { return 0; }
    if (el2d == 0) { return 6 * cx * cy * cz; }
    if ((el2d & 2) && thk > 0) { return 3 * 2 * (cy * cz + cx * cz + cx * cy); }
    return 0;
  }
  if (compiled) {
    // twist reference of a 1D rod chain with a twist hinge on its first (single) hinge
    if (dim == 1 &&
        bending.size() >= 1 + mjROD_HINGE_SIZE &&
        bending[0] > 0 &&
        bending[1 + 3] < 0 &&
        bending[1 + 7] >= 0) {
      return mjROD_STATE_SIZE;
    }
    int pnum = 0;
    if (stressyield > 0 && !interpolated && !rigid && dim >= 2) {
      if (!stiffness.empty() && stiffness[0] != 0) { pnum += (dim == 2 ? 3 : 6) * nelem; }
      if (dim == 2 && !bending.empty() && thickness > 0) { pnum += nedge; }
    }
    return pnum;
  }
  bool rgd = !spec_vert_.empty() && spec_vertbody_.size() == 1;
  if (!rgd && spec.dim == 1 && spec.young > 0 && spec_nodebody_.empty() && model) {
    const mjCBase* twist = model->FindObject(mjOBJ_JOINT, name + "_twist");
    if (twist && static_cast<const mjCJoint*>(twist)->spec.type == mjJNT_HINGE) {
      return mjROD_STATE_SIZE;
    }
  }
  if (rgd || spec.stressyield <= 0 || !spec_nodebody_.empty() || spec.dim < 2 || spec.young <= 0) {
    return 0;
  }
  int num_elem = (int)spec_elem_.size() / (spec.dim + 1);
  if (num_elem <= 0) { return 0; }
  int pnum = 0;
  if ((spec.dim == 2 && spec.elastic2d >= 2 && spec.thickness > 0) || spec.dim == 3) {
    pnum += (spec.dim == 2 ? 3 : 6) * num_elem;
  }
  if (spec.dim == 2 && (spec.elastic2d == 1 || spec.elastic2d == 3) && spec.thickness > 0) {
    std::unordered_set<std::pair<int, int>, PairHash> unique_edges;
    for (int f = 0; f < num_elem; f++) {
      const int* v = spec_elem_.data() + 3 * f;
      for (int e = 0; e < 3; e++) {
        int v0 = v[eledge[1][e][0]];
        int v1 = v[eledge[1][e][1]];
        unique_edges.insert({std::min(v0, v1), std::max(v0, v1)});
      }
    }
    pnum += (int)unique_edges.size();
  }
  return pnum;
}

void mjCFlex::PointToLocal() {
  spec.element      = static_cast<mjsElement*>(this);
  spec.material     = &spec_material_;
  spec.vertbody     = &spec_vertbody_;
  spec.nodebody     = &spec_nodebody_;
  spec.vert         = &spec_vert_;
  spec.node         = &spec_node_;
  spec.texcoord     = &spec_texcoord_;
  spec.elemtexcoord = &spec_elemtexcoord_;
  spec.elem         = &spec_elem_;
  spec.info         = &info;
  material          = nullptr;
  vertbody          = nullptr;
  nodebody          = nullptr;
  vert              = nullptr;
  node              = nullptr;
  texcoord          = nullptr;
  elemtexcoord      = nullptr;
  elem              = nullptr;
}


void mjCFlex::NameSpace(const mjCModel* m) {
  mjCBase::NameSpace(m);
  for (auto& name : spec_vertbody_) {
    if (!name.empty() && name != "world") { name = m->prefix + name + m->suffix; }
  }
  for (auto& name : spec_nodebody_) {
    if (!name.empty() && name != "world") { name = m->prefix + name + m->suffix; }
  }
  if (model != m) {
    for (auto& mat : spec_material_) {
      if (!mat.empty()) { mat = m->prefix + mat + m->suffix; }
    }
  }
}


void mjCFlex::CopyFromSpec() {
  *static_cast<mjsFlex*>(this) = spec;

  spec.info     = &info;
  material_     = spec_material_;
  vertbody_     = spec_vertbody_;
  nodebody_     = spec_nodebody_;
  vert_         = spec_vert_;
  node_         = spec_node_;
  texcoord_     = spec_texcoord_;
  elemtexcoord_ = spec_elemtexcoord_;
  elem_         = spec_elem_;

  // clear precompiled asset. TODO: use asset cache
  nedge = 0;
  edge.clear();
  shell.clear();

  // clear state derived by a previous compile: the spec may have changed since
  // (e.g. elastic2d, young, vertices, elements or its equality constraints)
  flaps.clear();
  stiffness.clear();
  bending.clear();
  cell_empty.clear();
  cell_full.clear();
  cell_vol.clear();
  cell_alphas.clear();
  has_strain_eq = false;
  rigid         = false;
  centered      = false;
}


bool mjCFlex::HasTexcoord() const {
  return !texcoord_.empty();
}


void mjCFlex::ResolveReferences(const mjCModel* m) {
  interpolated = !nodebody_.empty();
  vertbodyid.clear();
  nodebodyid.clear();
  for (const auto& vertbody : vertbody_) {
    mjCBody* pbody = static_cast<mjCBody*>(m->FindObject(mjOBJ_BODY, vertbody));
    if (pbody) {
      vertbodyid.push_back(pbody->id);
    } else {
      throw mjCError(this, "unknown body '%s' in flex", vertbody.c_str());
    }
  }
  for (const auto& nodebody : nodebody_) {
    mjCBase* pbody = m->FindObject(mjOBJ_BODY, nodebody);
    if (pbody) {
      nodebodyid.push_back(pbody->id);
    } else {
      throw mjCError(this, "unknown body '%s' in flex", nodebody.c_str());
    }
  }
}


// gauge of the lowest frame or body which contains the bodies of the vertices and nodes
mjCGauge mjCFlex::BodyGauge() const {
  std::vector<const mjCBase*> bodies;
  for (const std::vector<std::string>* names : {&spec_vertbody_, &spec_nodebody_}) {
    for (const std::string& name : *names) {
      if (const mjCBase* body = model->FindObject(mjOBJ_BODY, name)) { bodies.push_back(body); }
    }
  }
  return LowestContainerGauge(bodies);
}


// scale the compiled copy: vertex and node positions, given in the coordinates of their bodies, by
// the gauges of those bodies, and the properties of the flex by the gauge of the lowest frame or
// body which contains them all
void mjCFlex::ApplyGauge() {
  auto points = [&](std::vector<double>& pos, const std::vector<int>& bodyid) {
    for (int i = 0; i < pos.size() / 3 && !bodyid.empty(); i++) {
      int b = bodyid[bodyid.size() == 1 ? 0 : i];
      model->Bodies()[b]->ContentGauge().Point(pos.data() + 3 * i);
    }
  };
  points(vert_, vertbodyid);
  points(node_, nodebodyid);

  const mjCGauge g = BodyGauge();
  if (g.IsIdentity()) { return; }
  const double fl  = g.Shape(1);
  radius          *= fl;
  thickness       *= fl;
  sectionsize[0]  *= fl;
  sectionsize[1]  *= fl;
  margin          *= fl;
  gap             *= fl;
  solimp[2]       *= fl;
  friction[1]     *= fl;
  friction[2]     *= fl;
  young           *= g.Force(-1, 1);
  edgestiffness   *= g.Force(0, 1);
  edgedamping     *= g.Force(0, 1);
}


// Mirrors mj_flexSimple: only fixed-frame XYZ translations use the cached bending factor.
bool mjCFlex::IsSimple() const {
  for (int bid : vertbodyid) {
    const mjCBody* weld = model->Bodies()[model->Bodies()[bid]->weldid];
    if (!weld->joints.empty()) {
      if (weld->joints.size() != 3) return false;
      for (int j = 0; j < 3; j++) {
        const mjCJoint* joint = weld->joints[j];
        if (joint->type != mjJNT_SLIDE) return false;
        for (int k = 0; k < 3; k++) {
          // Match the precision of the engine's compiled axes, including in float builds.
          if (std::abs(static_cast<mjtNum>(joint->axis[k]) - (j == k)) > mjEPS) return false;
        }
      }
    }
    for (const mjCBody* ancestor = weld->parent; ancestor; ancestor = ancestor->parent) {
      if (!ancestor->joints.empty()) return false;
    }
  }
  return true;
}


std::string mjCFlex::ComputeStiffnessCacheKey(const std::vector<double>& nodexpos_local) const {
  std::size_t hash = 0;
  auto combine     = [&hash](std::size_t v) { hash ^= v + 0x9e3779b9 + (hash << 6) + (hash >> 2); };

  combine(std::hash<double>{}(young));
  combine(std::hash<double>{}(poisson));
  combine(std::hash<int>{}(dim));
  combine(std::hash<int>{}(spec.order));
  combine(std::hash<int>{}(spec.cellcount[0]));
  combine(std::hash<int>{}(spec.cellcount[1]));
  combine(std::hash<int>{}(spec.cellcount[2]));

  // element type and stored representation (raw stiffness or strain eigenmodes)
  combine(std::hash<int>{}(elastic2d != 0));
  combine(std::hash<double>{}(thickness));
  combine(std::hash<bool>{}(has_strain_eq));

  // compute bounding box from vertex positions
  if (!vert_.empty()) {
    double minx = vert_[0], maxx = vert_[0];
    double miny = vert_[1], maxy = vert_[1];
    double minz = vert_[2], maxz = vert_[2];
    for (std::size_t i = 3; i < vert_.size(); i += 3) {
      minx = std::min(minx, vert_[i]);
      maxx = std::max(maxx, vert_[i]);
      miny = std::min(miny, vert_[i + 1]);
      maxy = std::max(maxy, vert_[i + 1]);
      minz = std::min(minz, vert_[i + 2]);
      maxz = std::max(maxz, vert_[i + 2]);
    }
    combine(std::hash<double>{}(maxx - minx));
    combine(std::hash<double>{}(maxy - miny));
    combine(std::hash<double>{}(maxz - minz));
  }

  for (double x : nodexpos_local) { combine(std::hash<double>{}(x)); }

  for (double v : vert_) { combine(std::hash<double>{}(v)); }

  for (int e : elem_) { combine(std::hash<int>{}(e)); }

  for (bool empty : cell_empty) { combine(std::hash<bool>{}(empty)); }

  for (double vol : cell_vol) { combine(std::hash<double>{}(vol)); }

  for (const auto& alphas : cell_alphas) {
    for (double a : alphas) { combine(std::hash<double>{}(a)); }
  }

  for (int s : shell) { combine(std::hash<int>{}(s)); }

  return "flex_stiffness:" + std::to_string(hash);
}


bool mjCFlex::LoadCachedStiffness(const std::vector<double>& nodexpos_local) {
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);
  if (!cache) return false;

  std::string key = ComputeStiffnessCacheKey(nodexpos_local);

  auto load_fn = [this](const void* data) {
    const auto* cached = static_cast<const std::vector<double>*>(data);
    stiffness          = *cached;
    return true;
  };

  mjpResourceProvider dummy_provider{};
  dummy_provider.modified = [](const mjResource*, const char*) { return 0; };
  mjResource dummy_resource{};
  dummy_resource.name         = const_cast<char*>(key.c_str());
  dummy_resource.timestamp[0] = '\0';
  dummy_resource.provider     = &dummy_provider;

  return cache->PopulateData(key, &dummy_resource, load_fn);
}


void mjCFlex::CacheStiffness(const std::vector<double>& nodexpos_local) {
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);
  if (!cache || stiffness.empty()) return;

  std::string key = ComputeStiffnessCacheKey(nodexpos_local);

  auto* cached = new std::vector<double>(stiffness);

  std::size_t size = sizeof(*cached) + sizeof(double) * stiffness.size();

  std::shared_ptr<const void> cached_data(cached, [](const void* data) {
    delete static_cast<const std::vector<double>*>(data);
  });

  mjResource dummy_resource{};
  dummy_resource.name         = const_cast<char*>(key.c_str());
  dummy_resource.timestamp[0] = '\0';

  cache->Insert("", key, &dummy_resource, cached_data, size);
}


struct CachedFcmGeometry {
  std::vector<bool>                cell_empty;
  std::vector<bool>                cell_full;
  std::vector<double>              cell_vol;
  std::vector<std::vector<double>> cell_alphas;
};


std::string mjCFlex::ComputeFcmCacheKey(
    const double* vpos, const int* elems, int ne, int fdim, const double minmax[6]) const {
  std::size_t hash = 0;
  auto combine     = [&hash](std::size_t v) { hash ^= v + 0x9e3779b9 + (hash << 6) + (hash >> 2); };

  combine(std::hash<int>{}(fdim));
  combine(std::hash<int>{}(spec.order));
  combine(std::hash<int>{}(spec.cellcount[0]));
  combine(std::hash<int>{}(spec.cellcount[1]));
  combine(std::hash<int>{}(spec.cellcount[2]));
  combine(std::hash<int>{}(ne));

  double inv_d[3] = {
      (minmax[3] - minmax[0] > mjMINVAL) ? 1.0 / (minmax[3] - minmax[0]) : 0.0,
      (minmax[4] - minmax[1] > mjMINVAL) ? 1.0 / (minmax[4] - minmax[1]) : 0.0,
      (minmax[5] - minmax[2] > mjMINVAL) ? 1.0 / (minmax[5] - minmax[2]) : 0.0,
  };
  auto qcoord = [](double u) { return static_cast<std::int64_t>(std::llround(u * 1e9)); };

  int nvpe = fdim + 1;
  for (int e = 0; e < ne; e++) {
    int vids[4];
    for (int v = 0; v < nvpe; v++) { vids[v] = elems[e * nvpe + v]; }
    if (fdim == 3) { std::sort(vids, vids + 4); }
    for (int v = 0; v < nvpe; v++) {
      int vid = vids[v];
      combine(std::hash<int>{}(vid));
      for (int d = 0; d < 3; d++) {
        double u = (vpos[3 * vid + d] - minmax[d]) * inv_d[d];
        combine(std::hash<std::int64_t>{}(qcoord(u)));
      }
    }
  }

  return "flex_fcm:" + std::to_string(hash);
}


bool mjCFlex::LoadCachedFcmGeometry(const std::string& key) {
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);
  if (!cache) return false;

  auto load_fn = [this](const void* data) {
    const auto* cached = static_cast<const CachedFcmGeometry*>(data);
    cell_empty         = cached->cell_empty;
    cell_full          = cached->cell_full;
    cell_vol           = cached->cell_vol;
    cell_alphas        = cached->cell_alphas;
    return true;
  };

  mjpResourceProvider dummy_provider{};
  dummy_provider.modified = [](const mjResource*, const char*) { return 0; };
  mjResource dummy_resource{};
  dummy_resource.name         = const_cast<char*>(key.c_str());
  dummy_resource.timestamp[0] = '\0';
  dummy_resource.provider     = &dummy_provider;

  return cache->PopulateData(key, &dummy_resource, load_fn);
}


void mjCFlex::CacheFcmGeometry(const std::string& key) {
  mjCCache* cache = reinterpret_cast<mjCCache*>(mj_getCache()->impl_);
  if (!cache || cell_empty.empty()) return;

  auto* cached        = new CachedFcmGeometry();
  cached->cell_empty  = cell_empty;
  cached->cell_full   = cell_full;
  cached->cell_vol    = cell_vol;
  cached->cell_alphas = cell_alphas;

  std::size_t size = sizeof(*cached) +
                     cell_empty.size() / 8 +
                     cell_full.size() / 8 +
                     sizeof(double) * cell_vol.size() +
                     sizeof(std::vector<double>) * cell_alphas.size();
  for (const auto& alphas : cell_alphas) { size += sizeof(double) * alphas.size(); }

  std::shared_ptr<const void> cached_data(cached, [](const void* data) {
    delete static_cast<const CachedFcmGeometry*>(data);
  });

  mjResource dummy_resource{};
  dummy_resource.name         = const_cast<char*>(key.c_str());
  dummy_resource.timestamp[0] = '\0';

  cache->Insert("", key, &dummy_resource, cached_data, size);
}


// compute interpolated shell bending edge data
// enumerates intra-surface and corner edges, stores per-edge metadata:
//   [fe_A, fe_B, local_A[2], local_B[2], stiffness, dn0[3]]
static void ComputeInterpBending(std::vector<double>&       bending,
                                 const std::vector<double>& nodexpos_local,
                                 int                        order,
                                 const int                  cellcount[3],
                                 double                     young,
                                 double                     poisson,
                                 double                     thickness) {
  // bending modulus D = E * t^3 / (12 * (1 - nu^2))
  double D_bend = young * thickness * thickness * thickness / (12.0 * (1.0 - poisson * poisson));

  int cx = cellcount[0], cy = cellcount[1], cz = cellcount[2];
  int ny_global = cy * order + 1;
  int nz_global = cz * order + 1;
  int npe       = (order + 1) * (order + 1);  // nodes per 2D face element

  // face layout: 6 surfaces of the box
  //   face 0: x=0, face 1: x=max, face 2: y=0, face 3: y=max,
  //   face 4: z=0, face 5: z=max
  int face_sizes[6]  = {cy * cz, cy * cz, cx * cz, cx * cz, cx * cy, cx * cy};
  int face_normal[6] = {0, 0, 1, 1, 2, 2};
  int face_count1[6] = {cz, cz, cx, cx, cy, cy};
  int face_fixed[6]  = {0, cx * order, 0, cy * order, 0, cz * order};

  // gather node positions for one face element
  auto gather_face_nodes = [&](int face_id, int within_face, std::vector<double>& fpos) {
    int nax = face_normal[face_id];
    int a0  = (nax + 1) % 3;
    int a1  = (nax + 2) % 3;
    int c1  = face_count1[face_id];
    int gf  = face_fixed[face_id];
    int q0  = within_face / c1;
    int q1  = within_face % c1;
    fpos.resize(3 * npe);
    int loc = 0;
    for (int l0 = 0; l0 <= order; l0++) {
      for (int l1 = 0; l1 <= order; l1++) {
        int g[3];
        g[nax]   = gf;
        g[a0]    = q0 * order + l0;
        g[a1]    = q1 * order + l1;
        int gidx = g[0] * ny_global * nz_global + g[1] * nz_global + g[2];
        mjuu_copyvec(fpos.data() + 3 * loc, &nodexpos_local[3 * gidx], 3);
        loc++;
      }
    }
  };

  // compute unnormalized normal and tangents at a parametric point
  auto compute_normal = [&](const std::vector<double>& fpos,
                            const double               local[2],
                            double                     normal[3],
                            double                     t1[3],
                            double                     t2[3]) {
    mjuu_zerovec(t1, 3);
    mjuu_zerovec(t2, 3);
    int idx = 0;
    for (int l0 = 0; l0 <= order; l0++) {
      for (int l1 = 0; l1 <= order; l1++) {
        double g0 = dphi(local[0], l0, order) * phi(local[1], l1, order);
        double g1 = phi(local[0], l0, order) * dphi(local[1], l1, order);
        for (int d = 0; d < 3; d++) {
          t1[d] += fpos[3 * idx + d] * g0;
          t2[d] += fpos[3 * idx + d] * g1;
        }
        idx++;
      }
    }
    mjuu_crossvec(normal, t1, t2);
  };

  // face cumulative offsets
  int face_cumul[6];
  face_cumul[0] = 0;
  for (int f = 1; f < 6; f++) { face_cumul[f] = face_cumul[f - 1] + face_sizes[f - 1]; }

  int face_count0[6] = {cy, cy, cz, cz, cx, cx};

  int cells[3] = {cx, cy, cz};

  // find the neighbor of face element (fid, q0, q1) across the edge in
  // direction dir (0=a0, 1=a1) at side (+1 or -1).
  // returns (fid_B, within_B) and fills local_A, local_B with parametric
  // midpoint coordinates on each side of the shared edge.
  auto get_neighbor =
      [&](int fid, int q0, int q1, int dir, int side, double local_A[2], double local_B[2])
      -> std::pair<int, int> {
    int nax = fid / 2, sign_f = fid % 2;
    int a0 = (nax + 1) % 3, a1 = (nax + 2) % 3;
    int nc1 = face_count1[fid];

    // parametric coordinates on face A at the shared edge
    local_A[0] = (dir == 0) ? (side > 0 ? 1.0 : 0.0) : 0.5;
    local_A[1] = (dir == 1) ? (side > 0 ? 1.0 : 0.0) : 0.5;

    // check if neighbor is on the same face (internal)
    int q_nb  = (dir == 0 ? q0 : q1) + side;
    int q_max = (dir == 0) ? face_count0[fid] : nc1;
    if (q_nb >= 0 && q_nb < q_max) {
      // internal neighbor
      int q0_B   = (dir == 0) ? q_nb : q0;
      int q1_B   = (dir == 0) ? q1 : q_nb;
      local_B[0] = (dir == 0) ? (side > 0 ? 0.0 : 1.0) : 0.5;
      local_B[1] = (dir == 1) ? (side > 0 ? 0.0 : 1.0) : 0.5;
      return {fid, q0_B * nc1 + q1_B};
    }

    // boundary neighbor: cross to adjacent face on the box
    int ax    = (dir == 0) ? a0 : a1;         // axis being crossed
    int fid_B = 2 * ax + (side > 0 ? 1 : 0);  // neighboring face
    int nc1_B = face_count1[fid_B];

    // the running coordinate along the shared edge maps to the neighbor face:
    //   dir=0: edge runs along a1, maps to a0_B = (ax+1)%3 = a1 → q0_B
    //   dir=1: edge runs along a0, maps to a1_B = (ax+2)%3 = a0 → q1_B
    // the boundary position maps to the other axis on face B (= nax of face A):
    //   q_boundary = sign_f ? cells[nax]-1 : 0
    int q_run      = (dir == 0) ? q1 : q0;
    int q_boundary = sign_f ? (cells[nax] - 1) : 0;
    int q0_B, q1_B;
    if (dir == 0) {
      q0_B       = q_run;
      q1_B       = q_boundary;
      local_B[0] = 0.5;
      local_B[1] = sign_f ? 1.0 : 0.0;
    } else {
      q0_B       = q_boundary;
      q1_B       = q_run;
      local_B[0] = sign_f ? 1.0 : 0.0;
      local_B[1] = 0.5;
    }
    return {fid_B, q0_B * nc1_B + q1_B};
  };

  struct BendEdge {
    int    fe_A, fe_B;          // global face element indices (for runtime)
    int    fid_A, fid_B;        // face id (0-5)
    int    within_A, within_B;  // within-face element index
    double local_A[2];
    double local_B[2];
  };
  std::vector<BendEdge> edges;

  // enumerate all edges: for each face element, check 4 neighbors
  // (2 directions × 2 sides). Add each edge once via fe_A < fe_B.
  for (int f = 0; f < 6; f++) {
    int nc0 = face_count0[f];
    int nc1 = face_count1[f];
    for (int q0 = 0; q0 < nc0; q0++) {
      for (int q1 = 0; q1 < nc1; q1++) {
        int within_A = q0 * nc1 + q1;
        int fe_A     = face_cumul[f] + within_A;

        for (int dir = 0; dir < 2; dir++) {
          for (int side = -1; side <= 1; side += 2) {
            double lA[2], lB[2];
            auto [fid_B, within_B] = get_neighbor(f, q0, q1, dir, side, lA, lB);
            int fe_B               = face_cumul[fid_B] + within_B;
            if (fe_A < fe_B) {
              BendEdge e;
              e.fe_A     = fe_A;
              e.fid_A    = f;
              e.within_A = within_A;
              e.fe_B     = fe_B;
              e.fid_B    = fid_B;
              e.within_B = within_B;
              mjuu_copyvec(e.local_A, lA, 2);
              mjuu_copyvec(e.local_B, lB, 2);
              edges.push_back(e);
            }
          }
        }
      }
    }
  }

  // compute per-edge bending data
  const int BEND_EDGE_SIZE = 10;  // should match engine_passive.c
  bending.resize(1 + edges.size() * BEND_EDGE_SIZE, 0);
  bending[0] = static_cast<double>(edges.size());

  for (int e = 0; e < (int)edges.size(); e++) {
    const BendEdge&     edge = edges[e];
    std::vector<double> fpos_A, fpos_B;
    gather_face_nodes(edge.fid_A, edge.within_A, fpos_A);
    gather_face_nodes(edge.fid_B, edge.within_B, fpos_B);

    // compute rest normals at edge midpoint
    double n_A[3], t1_A[3], t2_A[3];
    double n_B[3], t1_B[3], t2_B[3];
    compute_normal(fpos_A, edge.local_A, n_A, t1_A, t2_A);
    compute_normal(fpos_B, edge.local_B, n_B, t1_B, t2_B);

    // normalize
    double len_A = mjuu_normvec(n_A, 3);
    double len_B = mjuu_normvec(n_B, 3);
    if (len_A < 1e-12 || len_B < 1e-12) continue;

    // rest normal jump
    double dn0[3] = {n_A[0] - n_B[0], n_A[1] - n_B[1], n_A[2] - n_B[2]};

    // stiffness coefficient: D * l_e / h_e
    // determine which tangent is along vs across the edge for each face:
    //   local[k] == 0.5 means parametric direction k runs along the edge
    double h_A, l_A, h_B, l_B;
    if (edge.local_A[0] == 0.5) {
      // edge runs along ξ on face A: t1 is along edge, t2 is across
      l_A = mjuu_normvec(t1_A, 3);
      h_A = mjuu_normvec(t2_A, 3);
    } else {
      // edge runs along η on face A: t2 is along edge, t1 is across
      h_A = mjuu_normvec(t1_A, 3);
      l_A = mjuu_normvec(t2_A, 3);
    }
    if (edge.local_B[0] == 0.5) {
      l_B = mjuu_normvec(t1_B, 3);
      h_B = mjuu_normvec(t2_B, 3);
    } else {
      h_B = mjuu_normvec(t1_B, 3);
      l_B = mjuu_normvec(t2_B, 3);
    }
    double h_avg           = (h_A + h_B) / 2;
    double l_avg           = (l_A + l_B) / 2;
    double stiffness_coeff = D_bend * l_avg / mjMAX(h_avg, 1e-12);

    // pack into bending array
    double* edata = bending.data() + 1 + e * BEND_EDGE_SIZE;
    edata[0]      = static_cast<double>(edge.fe_A);
    edata[1]      = static_cast<double>(edge.fe_B);
    edata[2]      = edge.local_A[0];
    edata[3]      = edge.local_A[1];
    edata[4]      = edge.local_B[0];
    edata[5]      = edge.local_B[1];
    edata[6]      = stiffness_coeff;
    edata[7]      = dn0[0];
    edata[8]      = dn0[1];
    edata[9]      = dn0[2];
  }
}


// compiler
void mjCFlex::Compile(const mjVFS* vfs) {
  CopyFromSpec();
  interpolated = !nodebody_.empty();

  // set nelem; check sizes
  if (dim < 1 || dim > 3) { throw mjCError(this, "dim must be 1, 2 or 3"); }
  if (elem_.empty()) { throw mjCError(this, "elem is empty"); }
  if (elem_.size() % (dim + 1)) { throw mjCError(this, "elem size must be multiple of (dim+1)"); }
  if (vertbody_.empty() && !interpolated) {
    throw mjCError(this, "vertbody and nodebody are both empty");
  }
  if (vert_.size() % 3) { throw mjCError(this, "vert size must be a multiple of 3"); }
  if (edgestiffness > 0 && dim > 1) {
    throw mjCError(this, "edge stiffness only available for dim=1, please use <elasticity>");
  }
  nelem = (int)elem_.size() / (dim + 1);

  // elastic2d checks
  if (elastic2d) {
    if (thickness <= 0) { throw mjCError(this, "2d elasticity requires positive thickness"); }
    if (poisson < 0.0 || poisson >= 0.5) {
      throw mjCError(this, "Poisson ratio must be in [0, 0.5)");
    }
    if (dim != 2 && !interpolated) { throw mjCError(this, "2d elasticity requires 2d flex"); }
  }

  // elastic3d checks
  if (elastic3d < 0 || elastic3d > 1) {
    throw mjCError(this, "elastic3d must be 0 (StVK) or 1 (SNH)");
  }
  if (elastic3d == 1 && (dim != 3 || interpolated)) {
    throw mjCError(this, "stable Neo-Hookean elasticity requires a non-interpolated 3d flex");
  }
  if (elastic3d == 1 && model->option.integrator != mjINT_DISCRETE) {
    throw mjCError(this, "stable Neo-Hookean elasticity requires integrator='discrete'");
  }

  // yield checks
  if (stressyield < 0) { throw mjCError(this, "yield cannot be negative"); }
  if (stressyield > 0 && dim < 2 && !interpolated) {
    throw mjCError(this, "flex plasticity requires a 2d or 3d flex");
  }

  // set nvert, rigid, centered; check size
  if (vert_.empty()) {
    centered = true;
    nvert    = (int)vertbody_.size();
  } else {
    nvert = (int)vert_.size() / 3;
    if (vertbody_.size() == 1) {
      rigid = true;
    } else if (vertbody_.size() != nvert) {
      throw mjCError(this, "vertbody size must be 1 or nvert");
    }
  }
  if (nvert < dim + 1) { throw mjCError(this, "not enough vertices"); }

  // set nnode
  nnode = static_cast<int>(nodebody_.size());
  if (nnode && !spec.order) {
    throw mjCError(this, "Interpolation order must be explicitly specified (dof is missing)");
  }

  // check node compatibility with count and dof
  if (spec.order > 0) {
    if (spec.cellcount[0] == 0 || spec.cellcount[1] == 0 || spec.cellcount[2] == 0) {
      throw mjCError(this, "cellcount cannot be 0 in any dimension when interpolation order > 0");
    }

    int expected_nodes = (spec.cellcount[0] * spec.order + 1) *
                         (spec.cellcount[1] * spec.order + 1) *
                         (spec.cellcount[2] * spec.order + 1);
    if (nnode != expected_nodes) {
      std::string msg = "number of nodes (" +
                        std::to_string(nnode) +
                        ") does not match cellcount and dof expected (" +
                        std::to_string(expected_nodes) +
                        ")";
      throw mjCError(this, msg.c_str());
    }
  }

  // check elem vertex ids
  for (const auto& elem : elem_) {
    if (elem < 0 || elem >= nvert) { throw mjCError(this, "elem vertex id out of range"); }
  }

  // check texcoord
  if (texcoord_.size() % 2) { throw mjCError(this, "texcoord size must be even"); }
  if (!texcoord_.empty() && texcoord_.size() != 2 * nvert && elemtexcoord_.empty()) {
    throw mjCError(this, "two texture coordinates per vertex expected");
  }
  if (!elemtexcoord_.empty()) {
    if (texcoord_.empty()) { throw mjCError(this, "elemtexcoord requires texcoord"); }
    if ((int)elemtexcoord_.size() != (dim + 1) * nelem) {
      throw mjCError(this, "elemtexcoord size must be (dim+1)*nelem");
    }
    int ntexcoord = static_cast<int>(texcoord_.size() / 2);
    for (int tc : elemtexcoord_) {
      if (tc < 0 || tc >= ntexcoord) { throw mjCError(this, "elemtexcoord index out of range"); }
    }
  }

  // no elemtexcoord: copy from faces
  if (elemtexcoord_.empty() && !texcoord_.empty()) {
    elemtexcoord_.assign((dim + 1) * nelem, 0);
    memcpy(elemtexcoord_.data(), elem_.data(), (dim + 1) * nelem * sizeof(int));
  }

  // resolve material name(s)
  if (!material_.empty() && material_.size() != 1 && (int)material_.size() != nelem) {
    throw mjCError(this, "material size must be 0, 1 or nelem");
  }
  elemmatid_.assign(nelem, -1);
  if (!material_.empty()) {
    std::vector<int> resolved(material_.size(), -1);
    for (size_t i = 0; i < material_.size(); i++) {
      if (!material_[i].empty()) {
        mjCBase* pmat = model->FindObject(mjOBJ_MATERIAL, material_[i]);
        if (!pmat) { throw mjCError(this, "unknown material '%s' in flex", material_[i].c_str()); }
        resolved[i] = pmat->id;
      }
    }
    for (int e = 0; e < nelem; e++) {
      elemmatid_[e] = (resolved.size() == 1) ? resolved[0] : resolved[e];
    }
  }

  // resolve body ids
  ResolveReferences(model);
  ApplyGauge();

  // process elements
  for (int e = 0; e < (int)elem_.size() / (dim + 1); e++) {
    // make sorted copy of element
    std::vector<int> el;
    el.assign(elem_.begin() + e * (dim + 1), elem_.begin() + (e + 1) * (dim + 1));
    std::sort(el.begin(), el.end());

    // check for repeated vertices
    for (int k = 0; k < dim; k++) {
      if (el[k] == el[k + 1]) { throw mjCError(this, "repeated vertex in element"); }
    }
  }

  // determine rigid if not already set
  if (!rigid && !interpolated) {
    rigid = true;
    for (unsigned i = 1; i < vertbodyid.size(); i++) {
      if (vertbodyid[i] != vertbodyid[0]) {
        rigid = false;
        break;
      }
    }
  }

  // determine centered if not already set
  if (!centered && !interpolated) {
    centered = true;
    for (const auto& vert : vert_) {
      if (vert != 0) {
        centered = false;
        break;
      }
    }
  }

  if (!centered && interpolated) {
    centered = true;
    for (const auto& node : node_) {
      if (node != 0) {
        centered = false;
        break;
      }
    }
  }

  // disallow joint damping on free flex vertex/node bodies
  if (!rigid) {
    const std::vector<int>&      bodyids = interpolated ? nodebodyid : vertbodyid;
    std::unordered_map<int, int> body_count;
    for (int bid : bodyids) { body_count[bid]++; }
    for (int bid : bodyids) {
      if (body_count[bid] != 1) continue;
      mjCBody* pbody = model->Bodies()[bid];
      if (pbody->joints.empty() || !pbody->geoms.empty() || !pbody->bodies.empty()) { continue; }
      bool all_slide = true;
      for (const mjCJoint* jnt : pbody->joints) {
        if (jnt->spec.type != mjJNT_SLIDE) {
          all_slide = false;
          break;
        }
      }
      if (!all_slide) continue;
      for (const mjCJoint* jnt : pbody->joints) {
        bool has_damping = (jnt->spec.springdamper[0] > 0 && jnt->spec.springdamper[1] > 0);
        for (int p = 0; p <= mjNPOLY; p++) {
          if (jnt->spec.damping[p] != 0) {
            has_damping = true;
            break;
          }
        }
        if (has_damping) {
          throw mjCError(this,
                         "flex vertex/node body '%s' cannot have joint damping; "
                         "use flex elasticity or edge damping instead",
                         pbody->name.c_str());
        }
      }
    }
  }

  // compute global vertex positions
  vertxpos = std::vector<double>(3 * nvert);
  for (int i = 0; i < nvert; i++) {
    // get body id, set vertxpos = body.xpos0
    int b = rigid ? vertbodyid[0] : vertbodyid[i];
    mjuu_copyvec(vertxpos.data() + 3 * i, model->Bodies()[b]->xpos0, 3);

    // add vertex offset within body if not centered
    if (!centered || interpolated) {
      double offset[3];
      mjuu_rotVecQuat(offset, vert_.data() + 3 * i, model->Bodies()[b]->xquat0);
      mjuu_addtovec(vertxpos.data() + 3 * i, offset, 3);
    }

    if (interpolated) {
      // this should happen in ResolveReferences but we need a body id in this loop to compute
      // the global vertex position, this is a hack since it is the id of the parent body
      vertbodyid[i] = -1;
    }
  }

  // compute global node positions
  std::vector<double> nodexpos = std::vector<double>(3 * nnode);
  for (int i = 0; i < nnode; i++) {
    // get body id, set nodexpos = body.xpos0
    int b = nodebodyid[i];
    mjuu_copyvec(nodexpos.data() + 3 * i, model->Bodies()[b]->xpos0, 3);

    // add node offset within body if not centered
    if (!centered) {
      double offset[3];
      mjuu_rotVecQuat(offset, node_.data() + 3 * i, model->Bodies()[b]->xquat0);
      mjuu_addtovec(nodexpos.data() + 3 * i, offset, 3);
    }
  }

  // compute unrotated node positions for stiffness computation
  double              R0[9]          = {1, 0, 0, 0, 1, 0, 0, 0, 1};  // identity by default
  std::vector<double> nodexpos_local = ComputeUnrotatedNodePositions(nodexpos, R0);

  // reorder tetrahedra so right-handed face orientation is outside
  // faces are (0,1,2); (0,2,3); (0,3,1); (1,3,2)
  if (dim == 3) {
    for (int e = 0; e < nelem; e++) {
      const int* edata  = elem_.data() + e * (dim + 1);
      double*    v0     = vertxpos.data() + 3 * edata[0];
      double*    v1     = vertxpos.data() + 3 * edata[1];
      double*    v2     = vertxpos.data() + 3 * edata[2];
      double*    v3     = vertxpos.data() + 3 * edata[3];
      double     v01[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
      double     v02[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
      double     v03[3] = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};

      // detect wrong orientation
      double nrm[3];
      mjuu_crossvec(nrm, v01, v02);
      if (mjuu_dot3(nrm, v03) > 0) {
        // flip orientation
        int tmp                  = elem_[e * (dim + 1) + 1];
        elem_[e * (dim + 1) + 1] = elem_[e * (dim + 1) + 2];
        elem_[e * (dim + 1) + 2] = tmp;
        if (!elemtexcoord_.empty()) {
          int ttmp                         = elemtexcoord_[e * (dim + 1) + 1];
          elemtexcoord_[e * (dim + 1) + 1] = elemtexcoord_[e * (dim + 1) + 2];
          elemtexcoord_[e * (dim + 1) + 2] = ttmp;
        }
      }
    }
  }

  // create edges
  edgeidx_.assign(elem_.size() * kNumEdges[dim - 1] / (dim + 1), 0);

  // map from edge vertices to their index in `edges` vector
  std::unordered_map<std::pair<int, int>, int, PairHash> edge_indices;

  // insert local edges into global vector
  for (unsigned f = 0; f < elem_.size() / (dim + 1); f++) {
    int* v = elem_.data() + f * (dim + 1);
    for (int e = 0; e < kNumEdges[dim - 1]; e++) {
      auto pair = std::pair(min(v[eledge[dim - 1][e][0]], v[eledge[dim - 1][e][1]]),
                            max(v[eledge[dim - 1][e][0]], v[eledge[dim - 1][e][1]]));

      // if edge is already present in the vector only store its index
      auto [it, inserted] = edge_indices.insert({pair, nedge});

      if (inserted) {
        edge.push_back(pair);
        edgeidx_[f * kNumEdges[dim - 1] + e] = nedge++;
      } else {
        edgeidx_[f * kNumEdges[dim - 1] + e] = it->second;
      }
    }
  }

  // set size
  nedge = (int)edge.size();

  // create flap stencil
  if (dim == 2) { CreateFlapStencil(flaps, elem_, edgeidx_); }

  // compute elasticity
  if (shear < 0) { throw mjCError(this, "shear modulus must be non-negative"); }
  if (young > 0) {
    if (poisson < 0 || poisson >= 0.5) {
      throw mjCError(this, "Poisson ratio must be in [0, 0.5)");
    }

    // Mocap poses do not supply velocities, so they cannot define elastic damping or the
    // implicit shift consistently. Dynamic articulated attachments have ordinary Jacobians.
    // Undamped 1D rod bending allows mocap attachments: its forces depend only on positions,
    // and its implicit stiffness shift treats mocap vertices as fixed over the step, like
    // tendon springs attached to mocap bodies.
    if (!rigid && !interpolated && (dim > 1 || damping > 0)) {
      for (int bid : vertbodyid) {
        for (const mjCBody* body = model->Bodies()[bid]; body; body = body->parent) {
          if (body->mocap) {
            throw mjCError(this,
                           "flex elasticity does not support mocap attachments, body '%s'",
                           body->name.c_str());
          }
        }
      }
    }

    // linear elasticity
    if (!interpolated && dim > 1) { stiffness.assign((dim == 3 ? 24 : 21) * nelem, 0); }

    // geometrically nonlinear elasticity
    for (unsigned int t = 0; t < nelem; t++) {
      if (interpolated) { continue; }
      if (dim == 2 && elastic2d >= 2 && thickness > 0) {
        ComputeStiffness<Stencil2D>(stiffness,
                                    vertxpos,
                                    elem_.data() + (dim + 1) * t,
                                    t,
                                    young,
                                    poisson,
                                    thickness);
      } else if (dim == 3) {
        if (elastic3d == 1) {
          ComputeSNH(stiffness, vertxpos, elem_.data() + 4 * t, t, young, poisson);
        } else {
          ComputeStiffness<Stencil3D>(stiffness, vertxpos, elem_.data() + 4 * t, t, young, poisson);
        }
      }
    }

    // rod bending and torsion stiffness (1D, not interpolated or rigid)
    if (dim == 1 && !interpolated && !rigid) {
      double I1 = 0, I2 = 0, J = 0, psi0 = 0;
      if (section == mjFLEXSECTION_CIRCLE) {
        if (radius <= 0) {
          throw mjCError(this, "1D flex elasticity with circle section requires positive radius");
        }
        I1 = mjPI * pow(radius, 4) / 4;
        I2 = I1;
        J  = 2 * I1;
      } else if (section == mjFLEXSECTION_BOX) {
        if (sectionsize[0] <= 0 || sectionsize[1] <= 0) {
          throw mjCError(this, "1D flex elasticity with box section requires positive sectionsize");
        }
        // half-sizes a >= b: I1 is the weak axis, I2 is the strong axis
        double a = std::max(sectionsize[0], sectionsize[1]);
        double b = std::min(sectionsize[0], sectionsize[1]);
        I1       = 4.0 * a * b * b * b / 3.0;
        I2       = 4.0 * b * a * a * a / 3.0;
        J        = a * b * b * b * (16.0 / 3.0 - 3.36 * (b / a) * (1.0 - pow(b / a, 4) / 12.0));
        if (sectionsize[0] < sectionsize[1]) { psi0 = 0.5 * mjPI; }
      } else {
        throw mjCError(this, "unknown 1D flex section %d", nullptr, section);
      }
      int       twist_jnt = -1;
      mjCJoint* twist     = nullptr;
      if (!name.empty() && vertbodyid.size() >= 2 && vertbodyid[0] == vertbodyid[1]) {
        std::string twist_name = name + "_twist";
        mjCBase*    jobj       = model->FindObject(mjOBJ_JOINT, twist_name);
        if (jobj) {
          mjCJoint* jnt = static_cast<mjCJoint*>(jobj);
          if (jnt->type == mjJNT_HINGE) {
            twist_jnt = jnt->id;
            twist     = jnt;
          }
        }
      }
      double G = shear > 0 ? shear : young / (2 * (1 + poisson));
      int    bad;
      if (!ComputeRodBending(bending,
                             vertxpos.data(),
                             nvert,
                             edge,
                             young * I1,
                             young * I2,
                             G * J,
                             psi0,
                             twist_jnt,
                             &bad)) {
        throw mjCError(this, "degenerate 1D flex hinge at vertex %d", nullptr, bad);
      }
      if (bending[0] == 0) { bending.clear(); }

      // anisotropic box cross-sections require a single open chain of single hinges
      if (section == mjFLEXSECTION_BOX && !bending.empty()) {
        int  nhinge     = (int)bending[0];
        bool open_chain = (edge.size() == static_cast<size_t>(nhinge + 1));
        for (int h = 0; h < nhinge && open_chain; h++) {
          const double* hinge = bending.data() + 1 + mjROD_HINGE_SIZE * h;
          if (hinge[3] >= 0) { open_chain = false; }
          if (h > 0) {
            const double* prev = bending.data() + 1 + mjROD_HINGE_SIZE * (h - 1);
            if ((int)hinge[0] != (int)prev[1] || (int)hinge[1] != (int)prev[2]) {
              open_chain = false;
            }
          }
        }
        if (open_chain) {
          const double* first = bending.data() + 1;
          const double* last  = bending.data() + 1 + mjROD_HINGE_SIZE * (nhinge - 1);
          if ((int)last[1] == (int)first[0] || (int)last[2] == (int)first[0]) {
            open_chain = false;
          }
        }
        if (!open_chain) {
          throw mjCError(this, "1D flex elasticity with box section requires an open chain");
        }
      }

      // Rayleigh damping of the twist hinge, with the series torsional stiffness 2 kt_eff of a
      // chain of single hinges (see mj_rodTwistStiffness), unless the joint sets its own damping
      if (twist && !bending.empty()) {
        int    nhinge = (int)bending[0];
        double inv_kt = 0;
        for (int h = 0; h < nhinge && inv_kt >= 0; h++) {
          const double* hinge      = bending.data() + 1 + mjROD_HINGE_SIZE * h;
          bool          contiguous = h == 0 || ((int)hinge[0] == (int)hinge[1 - mjROD_HINGE_SIZE] &&
                                                (int)hinge[1] == (int)hinge[2 - mjROD_HINGE_SIZE]);
          if (hinge[3] >= 0 || (hinge[6] > 0 && hinge[5] == 0) || !contiguous) {
            inv_kt = -1;
          } else if (hinge[8] > 0) {
            inv_kt += 1 / hinge[8];
          }
        }
        if (inv_kt > 0 && twist->spec.damping[0] == 0) { twist->damping[0] = damping * 2 / inv_kt; }
      }
    }

    // bending stiffness (2D only)
    if (dim == 2 && (elastic2d == 1 || elastic2d == 3) && !interpolated) {
      bending.assign(nedge * 17, 0);

      // plate bending rigidity
      double rigidity = young * thickness * thickness * thickness / (12 * (1 - poisson * poisson));
      for (unsigned int e = 0; e < nedge; e++) {
        ComputeBending<StencilFlap>(bending.data() + 17 * e,
                                    vertxpos.data(),
                                    flaps[e].vertices,
                                    rigidity);
      }
      // store 2D shell thickness in unused slot stiffness[6] (stiffness[0:6] holds 2D stretch
      // metric)
      if (stressyield > 0 && !stiffness.empty()) { stiffness[6] = thickness; }
    }
  }

  // placeholder for setting plugins parameters, currently not used
  for (const auto& vbodyid : vertbodyid) {
    if (vbodyid < 0) { continue; }
    if (model->Bodies()[vbodyid]->plugin.element) {
      mjCPlugin* plugin_instance =
          static_cast<mjCPlugin*>(model->Bodies()[vbodyid]->plugin.element);
      if (!plugin_instance) { throw mjCError(this, "plugin instance not found"); }
    }
  }

  // create shell fragments
  CreateShell();

  // recompute cell_empty and FCM quadrature weights from vertex/element
  // geometry (volume mode only) (survives XML round-trips where flexcomp data
  // is lost)
  if (interpolated && !elastic2d && cell_empty.empty()) {
    int cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
    if (cx * cy * cz >= 1) {
      // as in flexcomp: vertices in the frame of the grid, cells dividing the box of the nodes
      std::vector<double> vertxpos_local(3 * nvert);
      for (int j = 0; j < nvert; j++) {
        mjuu_mulvecmat(vertxpos_local.data() + 3 * j, vertxpos.data() + 3 * j, R0);
      }
      double minmax[6] = {mjMAXVAL, mjMAXVAL, mjMAXVAL, -mjMAXVAL, -mjMAXVAL, -mjMAXVAL};
      for (int i = 0; i < nnode; i++) {
        for (int k = 0; k < 3; k++) {
          minmax[k]     = std::min(minmax[k], nodexpos_local[3 * i + k]);
          minmax[k + 3] = std::max(minmax[k + 3], nodexpos_local[3 * i + k]);
        }
      }
      ComputeCellEmpty(vertxpos_local.data(), elem_.data(), nelem, dim, minmax);

      // the grid rotation is measured on the first non-empty cell
      nodexpos_local = ComputeUnrotatedNodePositions(nodexpos, R0);
    }
  }

  // disallow free flex node bodies on interior shell nodes or empty-cell-only
  // volume nodes (checked once all flexes have resolved their node bodies and
  // empty cells, since multiple flexes may share a cage)
  if (!model->Flexes().empty() && this == model->Flexes().back()) {
    std::unordered_set<int> supported_bodies;
    for (const mjCFlex* f : model->Flexes()) {
      if (!f->interpolated || f->rigid) {
        for (int bid : f->vertbodyid) { supported_bodies.insert(bid); }
        for (int bid : f->nodebodyid) { supported_bodies.insert(bid); }
        continue;
      }
      int cx = f->spec.cellcount[0], cy = f->spec.cellcount[1], cz = f->spec.cellcount[2];
      int nx = cx * f->spec.order + 1;
      int ny = cy * f->spec.order + 1;
      int nz = cz * f->spec.order + 1;

      std::unordered_map<int, int> body_count;
      for (int bid : f->nodebodyid) { body_count[bid]++; }
      for (const auto& [bid, count] : body_count) {
        if (count > 1) { supported_bodies.insert(bid); }
      }

      for (int gi = 0; gi < nx; gi++) {
        for (int gj = 0; gj < ny; gj++) {
          for (int gk = 0; gk < nz; gk++) {
            int  idx       = gi * ny * nz + gj * nz + gk;
            bool supported = f->elastic2d ? (gi == 0 ||
                                             gi == nx - 1 ||
                                             gj == 0 ||
                                             gj == ny - 1 ||
                                             gk == 0 ||
                                             gk == nz - 1)
                                          : !f->NodeInEmptyCells(gi, gj, gk);
            if (supported) { supported_bodies.insert(f->nodebodyid[idx]); }
          }
        }
      }
    }

    auto has_free_node_body = [&](int bid) {
      if (bid <= 0 || supported_bodies.count(bid)) { return false; }
      const mjCBody* pbody = model->Bodies()[bid];
      if (pbody->joints.empty() || !pbody->geoms.empty() || !pbody->bodies.empty()) {
        return false;
      }
      for (const mjCJoint* jnt : pbody->joints) {
        if (jnt->spec.type != mjJNT_SLIDE) { return false; }
      }
      return true;
    };

    for (mjCFlex* f : model->Flexes()) {
      if (!f->interpolated || f->rigid) { continue; }
      int cx = f->spec.cellcount[0], cy = f->spec.cellcount[1], cz = f->spec.cellcount[2];
      int nx = cx * f->spec.order + 1;
      int ny = cy * f->spec.order + 1;
      int nz = cz * f->spec.order + 1;

      for (int gi = 0; gi < nx; gi++) {
        for (int gj = 0; gj < ny; gj++) {
          for (int gk = 0; gk < nz; gk++) {
            int idx = gi * ny * nz + gj * nz + gk;
            if (!has_free_node_body(f->nodebodyid[idx])) { continue; }
            if (f->elastic2d) {
              throw mjCError(f,
                             "shell flex '%s' has unpinned interior node %d "
                             "with free DOFs",
                             f->name.c_str(),
                             idx);
            } else {
              throw mjCError(f,
                             "volumetric flex '%s' has unpinned node %d in "
                             "empty cell with free DOFs",
                             f->name.c_str(),
                             idx);
            }
          }
        }
      }
    }
  }

  // disallow joints on the parent/ancestor body of an unpinned 3-slider flex
  if (!rigid) {
    auto free_slider_parent = [&](int bid) -> const mjCBody* {
      if (bid <= 0) { return nullptr; }
      const mjCBody* pbody = model->Bodies()[bid];
      if (pbody->joints.size() != 3 || !pbody->geoms.empty() || !pbody->bodies.empty()) {
        return nullptr;
      }
      for (const mjCJoint* jnt : pbody->joints) {
        if (jnt->spec.type != mjJNT_SLIDE) { return nullptr; }
      }
      return pbody->parent;
    };

    const mjCBody* common_parent    = nullptr;
    bool           all_free_sliders = true;
    int            num_active       = 0;

    if (!interpolated) {
      for (int bid : vertbodyid) {
        const mjCBody* parent = free_slider_parent(bid);
        if (!parent || (common_parent && parent != common_parent)) {
          all_free_sliders = false;
          break;
        }
        common_parent = parent;
        num_active++;
      }
    } else {
      int cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
      int nx = cx * spec.order + 1;
      int ny = cy * spec.order + 1;
      int nz = cz * spec.order + 1;
      for (int gi = 0; gi < nx && all_free_sliders; gi++) {
        for (int gj = 0; gj < ny && all_free_sliders; gj++) {
          for (int gk = 0; gk < nz && all_free_sliders; gk++) {
            bool supported = elastic2d ? (gi == 0 ||
                                          gi == nx - 1 ||
                                          gj == 0 ||
                                          gj == ny - 1 ||
                                          gk == 0 ||
                                          gk == nz - 1)
                                       : !NodeInEmptyCells(gi, gj, gk);
            if (!supported) { continue; }
            int            idx    = gi * ny * nz + gj * nz + gk;
            const mjCBody* parent = free_slider_parent(nodebodyid[idx]);
            if (!parent || (common_parent && parent != common_parent)) {
              all_free_sliders = false;
              break;
            }
            common_parent = parent;
            num_active++;
          }
        }
      }
    }

    if (all_free_sliders && num_active > 0 && common_parent && common_parent->id > 0) {
      for (const mjCBody* anc = common_parent; anc && anc->id > 0; anc = anc->parent) {
        if (!anc->joints.empty()) {
          throw mjCError(this,
                         "body '%s' has joints but no flex vertices/nodes are pinned to it",
                         anc->name.c_str());
        }
      }
    }
  }

  // check if any strain equality references this flex (the equalities may
  // have been added or removed since the flex was made)
  for (auto* equality : model->Equalities()) {
    if (equality->spec.type == mjEQ_FLEXSTRAIN && *equality->spec.name1 == name) {
      has_strain_eq = true;
      break;
    }
  }

  // compute linear stiffness for interpolated elements (cached)
  bool need_stiffness =
      interpolated && (has_strain_eq || (young > 0 && (elastic2d == 0 || elastic2d >= 2)));
  bool stiffness_cached = false;
  if (need_stiffness) { stiffness_cached = LoadCachedStiffness(nodexpos_local); }

  if (!stiffness_cached && need_stiffness) {
    // use young=1 for strain constraints (eigenvectors are geometry-only)
    double K_young   = has_strain_eq ? 1e1 : young;
    double K_poisson = has_strain_eq ? 0.3 : poisson;

    int cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
    int ny_global = cy * spec.order + 1;
    int nz_global = cz * spec.order + 1;

    // determine element type: 2D boundary quads (shell) or 3D cells (volume)
    bool shell_mode = elastic2d != 0;
    int  npe;       // nodes per element
    int  nelem_fe;  // total finite elements

    if (shell_mode) {
      npe      = pow(spec.order + 1, 2);  // (order+1)^2 for 2D quads
      nelem_fe = 2 * (cy * cz + cx * cz + cx * cy);
    } else {
      npe      = pow(spec.order + 1, 3);  // (order+1)^3 for 3D cells
      nelem_fe = cx * cy * cz;
    }
    int ndof_elem = 3 * npe;

    // total stiffness = nelem_fe * ndof_elem^2 (empty cells stay zero)
    stiffness.assign(nelem_fe * ndof_elem * ndof_elem, 0);

    // face layout for shell mode:
    //   face 0: x=0     (cy*cz quads, normal=0, in-plane=(1,2))
    //   face 1: x=max   (cy*cz quads, normal=0, in-plane=(1,2))
    //   face 2: y=0     (cx*cz quads, normal=1, in-plane=(0,2))
    //   face 3: y=max   (cx*cz quads, normal=1, in-plane=(0,2))
    //   face 4: z=0     (cx*cy quads, normal=2, in-plane=(0,1))
    //   face 5: z=max   (cx*cy quads, normal=2, in-plane=(0,1))
    // face_sizes = {cy*cz, cy*cz, cx*cz, cx*cz, cx*cy, cx*cy}
    int face_sizes[6]  = {cy * cz, cy * cz, cx * cz, cx * cz, cx * cy, cx * cy};
    int face_normal[6] = {0, 0, 1, 1, 2, 2};
    // cell counts along each in-plane axis for each face
    int face_count1[6] = {cz, cz, cx, cx, cy, cy};  // fast axis count
    // fixed axis value (in grid node units, 0 or max)
    int face_fixed[6] = {0, cx * spec.order, 0, cy * spec.order, 0, cz * spec.order};

    // compute stiffness per element
    std::vector<double> K_full_cell;
    bool                has_K_full_cell = false;
    std::vector<double> out_full_cell;
    bool                has_out_full_cell = false;

    for (int fe = 0; fe < nelem_fe; fe++) {
      // gather element node positions
      std::vector<double> elem_pos(3 * npe);
      int                 normal_axis = -1;

      if (shell_mode) {
        // determine which face and quad within face
        int face_id = 0, within_face = fe;
        int cumul = 0;
        for (int f = 0; f < 6; f++) {
          if (fe < cumul + face_sizes[f]) {
            face_id     = f;
            within_face = fe - cumul;
            break;
          }
          cumul += face_sizes[f];
        }

        normal_axis = face_normal[face_id];
        int na0     = (normal_axis + 1) % 3;  // slow in-plane axis
        int na1     = (normal_axis + 2) % 3;  // fast in-plane axis
        int c1      = face_count1[face_id];   // cell count along fast axis
        int g_fixed = face_fixed[face_id];    // grid index along normal axis
        int q0      = within_face / c1;       // quad index along slow in-plane axis
        int q1      = within_face % c1;       // quad index along fast in-plane axis

        // gather 2D face element nodes
        int local = 0;
        for (int l0 = 0; l0 <= spec.order; l0++) {
          for (int l1 = 0; l1 <= spec.order; l1++) {
            // build global node index from 3 axis values
            int g[3];
            g[normal_axis] = g_fixed;
            g[na0]         = q0 * spec.order + l0;
            g[na1]         = q1 * spec.order + l1;
            int global     = g[0] * ny_global * nz_global + g[1] * nz_global + g[2];
            mjuu_copyvec(elem_pos.data() + 3 * local, nodexpos_local.data() + 3 * global, 3);
            local++;
          }
        }
      } else {
        // 3D cell: convert flat index to (ci, cj, ck)
        int ci = fe / (cy * cz);
        int cj = (fe / cz) % cy;
        int ck = fe % cz;

        // skip stiffness computation for empty cells (no mesh content)
        if (!cell_empty.empty() && cell_empty[fe]) { continue; }

        // gather cell's local node positions
        int local = 0;
        for (int li = 0; li <= spec.order; li++) {
          for (int lj = 0; lj <= spec.order; lj++) {
            for (int lk = 0; lk <= spec.order; lk++) {
              int gi     = ci * spec.order + li;
              int gj     = cj * spec.order + lj;
              int gk     = ck * spec.order + lk;
              int global = gi * ny_global * nz_global + gj * nz_global + gk;
              mjuu_copyvec(elem_pos.data() + 3 * local, nodexpos_local.data() + 3 * global, 3);
              local++;
            }
          }
        }
      }

      // compute per-element stiffness
      std::vector<double>        K_elem(ndof_elem * ndof_elem, 0);
      const std::vector<double>* alphas = nullptr;
      if (shell_mode) {
        ComputeLinearStiffness2D(K_elem,
                                 elem_pos.data(),
                                 K_young,
                                 K_poisson,
                                 spec.order,
                                 thickness,
                                 normal_axis);
      } else {
        alphas = (fe < static_cast<int>(cell_alphas.size()) && !cell_alphas[fe].empty())
                     ? &cell_alphas[fe]
                     : nullptr;
        if (!alphas) {
          if (!has_K_full_cell) {
            K_full_cell.assign(ndof_elem * ndof_elem, 0);
            ComputeLinearStiffness(K_full_cell,
                                   elem_pos.data(),
                                   K_young,
                                   K_poisson,
                                   spec.order,
                                   nullptr);
            has_K_full_cell = true;
          }
          K_elem = K_full_cell;
        } else {
          ComputeLinearStiffness(K_elem, elem_pos.data(), K_young, K_poisson, spec.order, alphas);
        }
      }
      double* out = stiffness.data() + fe * ndof_elem * ndof_elem;

      if (has_strain_eq) {
        // eigendecompose: store [neig, sqrt(λ)*v_1, sqrt(λ)*v_2, ...]
        std::fill(out, out + ndof_elem * ndof_elem, 0.0);

        if (shell_mode) {
          // pure membrane K: eigendecompose gives 5 membrane modes (Q1),
          // then we add 1 explicit warp mode with bending stiffness (∝ t³)
          int neig = EigendecomposeStiffness(K_elem.data(), out, ndof_elem);

          // add explicit warp mode with plate bending stiffness
          double warp_stiffness = ComputeWarpStiffness(elem_pos.data(),
                                                       npe,
                                                       normal_axis,
                                                       K_young,
                                                       K_poisson,
                                                       thickness);
          if (warp_stiffness > 0) {
            double* warp_out = out + 1 + neig * ndof_elem;
            ComputeWarpMode(warp_out, elem_pos.data(), npe, spec.order, normal_axis);
            // scale by sqrt(stiffness) to match eigenmode convention
            double scale = std::sqrt(warp_stiffness);
            for (int j = 0; j < ndof_elem; j++) { warp_out[j] *= scale; }
            out[0] = static_cast<double>(neig + 1);
          }
        } else if (!alphas) {
          if (!has_out_full_cell) {
            out_full_cell.assign(ndof_elem * ndof_elem, 0.0);
            EigendecomposeStiffness(K_elem.data(), out_full_cell.data(), ndof_elem);
            has_out_full_cell = true;
          }
          std::copy(out_full_cell.begin(), out_full_cell.end(), out);
        } else {
          EigendecomposeStiffness(K_elem.data(), out, ndof_elem);
        }
      } else {
        // store raw K for passive forces
        std::copy(K_elem.begin(), K_elem.end(), out);
      }
    }
    CacheStiffness(nodexpos_local);
  }

  // compute interpolated shell bending edge data (independent of stiffness cache)
  if (interpolated && (elastic2d == 1 || elastic2d == 3) && thickness > 0 && young > 0) {
    ComputeInterpBending(bending,
                         nodexpos_local,
                         spec.order,
                         spec.cellcount,
                         young,
                         poisson,
                         thickness);
  }

  // create bounding volume hierarchy
  CreateBVH();

  // compute bounding box coordinates
  vert0_.assign(3 * nvert, 0);
  vertcell_.assign(nvert, -1);

  if (interpolated && nnode > 0) {
    // for interpolated flex, compute vert0_ in the unrotated local frame
    // to make parametric coordinates rotation-invariant
    std::vector<double> vertxpos_local(3 * nvert);
    for (int j = 0; j < nvert; j++) {
      mjuu_mulvecmat(vertxpos_local.data() + 3 * j, vertxpos.data() + 3 * j, R0);
    }

    // compute local-frame bounding box from unrotated node positions
    double lo[3] = {1e30, 1e30, 1e30};
    double hi[3] = {-1e30, -1e30, -1e30};
    for (int i = 0; i < nnode; i++) {
      for (int k = 0; k < 3; k++) {
        lo[k] = std::min(lo[k], nodexpos_local[3 * i + k]);
        hi[k] = std::max(hi[k], nodexpos_local[3 * i + k]);
      }
    }

    // set size from local bounding box
    for (int k = 0; k < 3; k++) { size[k] = (hi[k] - lo[k]) / 2; }

    // normalize vertex positions within local bounding box
    for (int j = 0; j < nvert; j++) {
      for (int k = 0; k < 3; k++) {
        double extent = hi[k] - lo[k];
        if (extent > mjMINVAL) {
          vert0_[3 * j + k] = (vertxpos_local[3 * j + k] - lo[k]) / extent;
        } else {
          vert0_[3 * j + k] = 0.5;
        }
      }
    }

    // assign each vertex to its containing active cell (or nearest active cell if pruned)
    int cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
    for (int j = 0; j < nvert; j++) {
      const double* coord = vert0_.data() + 3 * j;
      int           ci    = std::clamp(static_cast<int>(std::floor(coord[0] * cx)), 0, cx - 1);
      int           cj    = std::clamp(static_cast<int>(std::floor(coord[1] * cy)), 0, cy - 1);
      int           ck    = std::clamp(static_cast<int>(std::floor(coord[2] * cz)), 0, cz - 1);
      int           c     = ci * cy * cz + cj * cz + ck;
      if (!cell_empty.empty() && cell_empty[c]) {
        double u[3]       = {coord[0] * cx, coord[1] * cy, coord[2] * cz};
        double best_dist2 = mjMAXVAL;
        for (int ni = 0; ni < cx; ni++) {
          double dx = u[0] - std::clamp(u[0], static_cast<double>(ni), static_cast<double>(ni + 1));
          for (int nj = 0; nj < cy; nj++) {
            double dy =
                u[1] - std::clamp(u[1], static_cast<double>(nj), static_cast<double>(nj + 1));
            for (int nk = 0; nk < cz; nk++) {
              int nc = ni * cy * cz + nj * cz + nk;
              if (cell_empty[nc]) { continue; }
              double dz =
                  u[2] - std::clamp(u[2], static_cast<double>(nk), static_cast<double>(nk + 1));
              double dist2 = dx * dx + dy * dy + dz * dz;
              if (dist2 < best_dist2) {
                best_dist2 = dist2;
                c          = nc;
              }
            }
          }
        }
      }
      vertcell_[j] = c;
    }
  } else {
    // non-interpolated: use BVH bounding box (original behavior)
    const mjtNum* bvh = tree.Bvh().data();
    size[0]           = bvh[3] - radius;
    size[1]           = bvh[4] - radius;
    size[2]           = bvh[5] - radius;
    for (int j = 0; j < nvert; j++) {
      for (int k = 0; k < 3; k++) {
        if (size[k] > mjMINVAL) {
          vert0_[3 * j + k] = (vertxpos[3 * j + k] - bvh[k]) / (2 * size[k]) + 0.5;
        } else {
          vert0_[3 * j + k] = 0.5;
        }
      }
    }
  }

  // store node positions in unrotated (body-local) frame
  // this ensures the runtime displacement refpos - R^{-1}*x is zero at rest
  node0_.assign(3 * nnode, 0);
  for (int i = 0; i < nnode; i++) {
    mjuu_copyvec(node0_.data() + 3 * i, nodexpos_local.data() + 3 * i, 3);
  }
}


// compute unrotated node positions for stiffness computation and node0_
//
// the runtime corotational code extracts rotation R from the deformation
// gradient and computes displacement as R^{-1}*x - refpos; at rest R = R0
// (the total grid rotation), so refpos must equal R0^{-1}*nodexpos to get
// zero displacement at rest; additionally, the stiffness eigenvectors must
// be computed from axis-aligned positions to preserve the diagonal Jacobian
// assumption in ComputeLinearStiffness.
std::vector<double> mjCFlex::ComputeUnrotatedNodePositions(const std::vector<double>& nodexpos,
                                                           double* R0_out) const {
  std::vector<double> nodexpos_local(3 * nnode);
  if (interpolated && nnode > 0) {
    int ny_global = spec.cellcount[1] * spec.order + 1;
    int nz_global = spec.cellcount[2] * spec.order + 1;

    // find first non-empty cell
    int  cx = spec.cellcount[0], cy = spec.cellcount[1], cz = spec.cellcount[2];
    int  ref_ci = 0, ref_cj = 0, ref_ck = 0;
    bool found = false;
    for (int ci = 0; ci < cx && !found; ci++) {
      for (int cj = 0; cj < cy && !found; cj++) {
        for (int ck = 0; ck < cz && !found; ck++) {
          int cell_idx = ci * cy * cz + cj * cz + ck;
          if (cell_empty.empty() || !cell_empty[cell_idx]) {
            ref_ci = ci;
            ref_cj = cj;
            ref_ck = ck;
            found  = true;
          }
        }
      }
    }

    // corner indices of the reference cell (order=1 corners at local 0,0,0
    // and at offsets along each parametric axis)
    int g000 = (ref_ci * spec.order) * ny_global * nz_global +
               (ref_cj * spec.order) * nz_global +
               (ref_ck * spec.order);
    int g100 = ((ref_ci * spec.order) + spec.order) * ny_global * nz_global +
               (ref_cj * spec.order) * nz_global +
               (ref_ck * spec.order);
    int g010 = (ref_ci * spec.order) * ny_global * nz_global +
               ((ref_cj * spec.order) + spec.order) * nz_global +
               (ref_ck * spec.order);
    int g001 = (ref_ci * spec.order) * ny_global * nz_global +
               (ref_cj * spec.order) * nz_global +
               ((ref_ck * spec.order) + spec.order);

    // edge vectors (columns of the deformation gradient F = R * S)
    // we store them as rows in R0 to use mjuu_mulvecmat for applying R0^{-1}
    double R0[9];
    for (int d = 0; d < 3; d++) {
      R0[0 + d] = nodexpos[3 * g100 + d] - nodexpos[3 * g000 + d];
      R0[3 + d] = nodexpos[3 * g010 + d] - nodexpos[3 * g000 + d];
      R0[6 + d] = nodexpos[3 * g001 + d] - nodexpos[3 * g000 + d];
    }

    // normalize to get rotation matrix columns (valid for regular grids)
    double li = mjuu_normvec(R0 + 0, 3);
    double lj = mjuu_normvec(R0 + 3, 3);
    double lk = mjuu_normvec(R0 + 6, 3);
    (void)li;
    (void)lj;
    (void)lk;

    // assert R0 is orthonormal (rows are the normalized edge vectors)
    for (int a = 0; a < 3; a++) {
      for (int b = a; b < 3; b++) {
        double dot      = mjuu_dot3(R0 + 3 * a, R0 + 3 * b);
        double expected = (a == b) ? 1.0 : 0.0;
        if (std::abs(dot - expected) > 1e-8) {
          throw mjCError(this, "flex grid rotation R0 is not orthonormal");
        }
      }
    }

    // output R0 if requested
    if (R0_out) { mjuu_copyvec(R0_out, R0, 9); }

    // apply inverse rotation to each nodexpos to get local-frame positions
    for (int i = 0; i < nnode; i++) {
      const double* p = nodexpos.data() + 3 * i;
      double*       q = nodexpos_local.data() + 3 * i;
      mjuu_mulvecmat(q, p, R0);
    }
  } else {
    nodexpos_local = nodexpos;
  }
  return nodexpos_local;
}

// relative cell core shrink margin for pruning tiny-cut sliver cells
// (must match kMaxExtrap in mju_cellLookup in engine_util_misc.c)
constexpr double kSliverMargin = 0.02;

// test if a 3D triangle (v0, v1, v2) intersects an axis-aligned box [bmin,
// bmax]
static bool TriangleIntersectsBox(const double v0[3],
                                  const double v1[3],
                                  const double v2[3],
                                  const double bmin[3],
                                  const double bmax[3]) {
  double c[3] = {
      0.5 * (bmin[0] + bmax[0]),
      0.5 * (bmin[1] + bmax[1]),
      0.5 * (bmin[2] + bmax[2]),
  };
  double e[3] = {
      0.5 * (bmax[0] - bmin[0]),
      0.5 * (bmax[1] - bmin[1]),
      0.5 * (bmax[2] - bmin[2]),
  };
  double eps  = 1e-9 * std::max({e[0], e[1], e[2], 1e-6});
  e[0]       += eps;
  e[1]       += eps;
  e[2]       += eps;

  double u0[3] = {v0[0] - c[0], v0[1] - c[1], v0[2] - c[2]};
  double u1[3] = {v1[0] - c[0], v1[1] - c[1], v1[2] - c[2]};
  double u2[3] = {v2[0] - c[0], v2[1] - c[1], v2[2] - c[2]};

  // 1. box face normals (AABB overlap)
  for (int d = 0; d < 3; d++) {
    double lo = std::min({u0[d], u1[d], u2[d]});
    double hi = std::max({u0[d], u1[d], u2[d]});
    if (lo > e[d] || hi < -e[d]) { return false; }
  }

  double f[3][3] = {
      {u1[0] - u0[0], u1[1] - u0[1], u1[2] - u0[2]},
      {u2[0] - u1[0], u2[1] - u1[1], u2[2] - u1[2]},
      {u0[0] - u2[0], u0[1] - u2[1], u0[2] - u2[2]},
  };

  // 2. 9 cross-product axes a_ij = e_i x f_j
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double a[3] = {0, 0, 0};
      int    i1   = (i + 1) % 3;
      int    i2   = (i + 2) % 3;
      a[i1]       = -f[j][i2];
      a[i2]       = f[j][i1];
      double p0   = mjuu_dot3(a, u0);
      double p1   = mjuu_dot3(a, u1);
      double p2   = mjuu_dot3(a, u2);
      double r    = e[i1] * std::abs(a[i1]) + e[i2] * std::abs(a[i2]);
      if (std::min({p0, p1, p2}) > r || std::max({p0, p1, p2}) < -r) { return false; }
    }
  }

  // 3. triangle normal n = f[0] x f[1]
  double n[3];
  mjuu_crossvec(n, f[0], f[1]);
  double d0 = mjuu_dot3(n, u0);
  double r  = e[0] * std::abs(n[0]) + e[1] * std::abs(n[1]) + e[2] * std::abs(n[2]);
  if (std::abs(d0) > r) { return false; }

  return true;
}

// test if a 3D segment (v0, v1) intersects an axis-aligned box [bmin, bmax]
static bool SegmentIntersectsBox(const double v0[3],
                                 const double v1[3],
                                 const double bmin[3],
                                 const double bmax[3]) {
  return TriangleIntersectsBox(v0, v1, v1, bmin, bmax);
}

struct TetCellData {
  double v[4][3];
  double v0[3];
  double invM[3][3];
  double lo[3];
  double hi[3];
  int    boundary_mask = 0xF;
};

static bool PointInTet(const double x[3], const TetCellData& tet) {
  for (int d = 0; d < 3; d++) {
    if (x[d] < tet.lo[d] || x[d] > tet.hi[d]) { return false; }
  }
  double           dx[3]    = {x[0] - tet.v0[0], x[1] - tet.v0[1], x[2] - tet.v0[2]};
  double           l1       = mjuu_dot3(tet.invM[0], dx);
  double           l2       = mjuu_dot3(tet.invM[1], dx);
  double           l3       = mjuu_dot3(tet.invM[2], dx);
  double           l0       = 1.0 - l1 - l2 - l3;
  constexpr double kBaryTol = -1e-7;
  return l0 >= kBaryTol && l1 >= kBaryTol && l2 >= kBaryTol && l3 >= kBaryTol;
}

static bool TetIntersectsBox(const double       v0[3],
                             const double       v1[3],
                             const double       v2[3],
                             const double       v3[3],
                             const TetCellData* tet,
                             const double       bmin[3],
                             const double       bmax[3]) {
  if (TriangleIntersectsBox(v0, v1, v2, bmin, bmax) ||
      TriangleIntersectsBox(v0, v1, v3, bmin, bmax) ||
      TriangleIntersectsBox(v0, v2, v3, bmin, bmax) ||
      TriangleIntersectsBox(v1, v2, v3, bmin, bmax)) {
    return true;
  }
  if (tet) {
    double c[3] = {
        0.5 * (bmin[0] + bmax[0]),
        0.5 * (bmin[1] + bmax[1]),
        0.5 * (bmin[2] + bmax[2]),
    };
    if (PointInTet(c, *tet)) { return true; }
  }
  return false;
}


// evaluate shifted Legendre polynomials P_r(2s - 1) and their exact
// antiderivatives int_0^s P_r(2*sigma - 1) d(sigma) on [0, 1] for r = 0..4
static inline void EvalShiftedLegendre1D(double s, double P[5], double intP[5]) {
  double x  = 2.0 * s - 1.0;
  double x2 = x * x;
  P[0]      = 1.0;
  P[1]      = x;
  P[2]      = 0.5 * (3.0 * x2 - 1.0);
  P[3]      = 0.5 * x * (5.0 * x2 - 3.0);
  P[4]      = 0.125 * (35.0 * x2 * x2 - 30.0 * x2 + 3.0);
  double P5 = 0.125 * x * (63.0 * x2 * x2 - 70.0 * x2 + 15.0);

  intP[0] = s;
  intP[1] = (P[2] - P[0]) / 6.0;
  intP[2] = (P[3] - P[1]) / 10.0;
  intP[3] = (P[4] - P[2]) / 14.0;
  intP[4] = (P5 - P[3]) / 18.0;
}

// build the 1D weighted minimum-norm moment-fitting projector H = V^T (V W V^T)^{-1}
// mapping exact shifted Legendre moments mu_r (r = 0..2*order) on [0, 1] to
// quadrature multipliers alpha_k at the nq_axis composite Gauss-Legendre nodes
static void BuildLegendreMomentProjector(int           order,
                                         const double* qpts,
                                         const double* qwts,
                                         double        H[6][5]) {
  int    nbasis  = order + 1;
  int    nq_axis = kFcmSubdiv * nbasis;
  int    nmom    = 2 * order + 1;
  double V[5][6] = {};
  for (int k = 0; k < nq_axis; k++) {
    double P[5], intP[5];
    EvalShiftedLegendre1D(qpts[k], P, intP);
    for (int r = 0; r < nmom; r++) { V[r][k] = P[r]; }
  }

  // Gram matrix G = V * W * V^T and augmented identity for Gauss-Jordan inverse
  double G[5][10] = {};
  for (int r = 0; r < nmom; r++) {
    for (int c = 0; c < nmom; c++) {
      double sum = 0.0;
      for (int k = 0; k < nq_axis; k++) { sum += qwts[k] * V[r][k] * V[c][k]; }
      G[r][c] = sum;
    }
    G[r][nmom + r] = 1.0;
  }

  for (int i = 0; i < nmom; i++) {
    int piv = i;
    for (int r = i + 1; r < nmom; r++) {
      if (std::abs(G[r][i]) > std::abs(G[piv][i])) { piv = r; }
    }
    if (piv != i) {
      for (int c = 0; c < 2 * nmom; c++) { std::swap(G[i][c], G[piv][c]); }
    }
    double inv_diag = 1.0 / G[i][i];
    for (int c = 0; c < 2 * nmom; c++) { G[i][c] *= inv_diag; }
    for (int r = 0; r < nmom; r++) {
      if (r == i) continue;
      double factor = G[r][i];
      for (int c = 0; c < 2 * nmom; c++) { G[r][c] -= factor * G[i][c]; }
    }
  }

  for (int k = 0; k < nq_axis; k++) {
    for (int r = 0; r < nmom; r++) {
      double sum = 0.0;
      for (int c = 0; c < nmom; c++) { sum += V[c][k] * G[c][nmom + r]; }
      H[k][r] = sum;
    }
  }
}

// clip a 3D convex polygon against an axis-aligned half-space
// sign * p[axis] <= sign * bound (sign = +1 for p[axis] <= bound, -1 for p[axis] >= bound)
static int ClipPolygonAxis(
    const double in_pts[][3], int n_in, int axis, double bound, double sign, double out_pts[][3]) {
  if (n_in < 3) return 0;
  int    n_out  = 0;
  double d_prev = sign * (in_pts[n_in - 1][axis] - bound);
  for (int i = 0; i < n_in; i++) {
    const double* p_prev = in_pts[(i + n_in - 1) % n_in];
    const double* p_curr = in_pts[i];
    double        d_curr = sign * (p_curr[axis] - bound);
    if (d_curr <= 0.0) {
      if (d_prev > 0.0) {
        double t = d_prev / (d_prev - d_curr);
        for (int d = 0; d < 3; d++) { out_pts[n_out][d] = p_prev[d] + t * (p_curr[d] - p_prev[d]); }
        out_pts[n_out][axis] = bound;
        n_out++;
      }
      for (int d = 0; d < 3; d++) { out_pts[n_out][d] = p_curr[d]; }
      n_out++;
    } else if (d_prev <= 0.0) {
      double t = d_prev / (d_prev - d_curr);
      for (int d = 0; d < 3; d++) { out_pts[n_out][d] = p_prev[d] + t * (p_curr[d] - p_prev[d]); }
      out_pts[n_out][axis] = bound;
      n_out++;
    }
    d_prev = d_curr;
  }
  return n_out;
}

// accumulate exact shifted Legendre moments M[a][b][c] over a positively-oriented
// tetrahedron (v0, v1, v2, v3) (in local cell coordinates [0, 1]^3) clipped to
// the unit cube [0, 1]^3 via 3D Divergence Theorem + Duffy Gauss-Legendre quadrature
static void AccumulateClippedTetMoments(const double  v0[3],
                                        const double  v1[3],
                                        const double  v2[3],
                                        const double  v3[3],
                                        int           boundary_mask,
                                        double        sign_weight,
                                        int           nmom,
                                        int           n_duffy,
                                        const double* duffy_pts,
                                        const double* duffy_wts,
                                        int           n_duffy_cap,
                                        const double* duffy_cap_pts,
                                        const double* duffy_cap_wts,
                                        double        M[5][5][5]) {
  constexpr double kClipEps = 1e-12;
  double           min_s    = std::min({v0[0], v1[0], v2[0], v3[0]});
  double           max_s    = std::max({v0[0], v1[0], v2[0], v3[0]});
  double           min_t    = std::min({v0[1], v1[1], v2[1], v3[1]});
  double           max_t    = std::max({v0[1], v1[1], v2[1], v3[1]});
  double           min_u    = std::min({v0[2], v1[2], v2[2], v3[2]});
  double           max_u    = std::max({v0[2], v1[2], v2[2], v3[2]});

  if (max_s <= kClipEps ||
      min_s >= 1.0 - kClipEps ||
      max_t <= kClipEps ||
      min_t >= 1.0 - kClipEps ||
      max_u <= kClipEps ||
      min_u >= 1.0 - kClipEps) {
    return;
  }

  auto integrate_clipped_poly = [&](const double poly[][3], int npoly, bool on_s1_cap) {
    if (npoly < 3) return;
    int           nq    = on_s1_cap ? n_duffy_cap : n_duffy;
    const double* q_pts = on_s1_cap ? duffy_cap_pts : duffy_pts;
    const double* q_wts = on_s1_cap ? duffy_cap_wts : duffy_wts;
    const double* p0    = poly[0];
    for (int m = 1; m + 1 < npoly; m++) {
      const double* p1 = poly[m];
      const double* p2 = poly[m + 1];
      double        nx = (p1[1] - p0[1]) * (p2[2] - p0[2]) - (p1[2] - p0[2]) * (p2[1] - p0[1]);
      if (std::abs(nx) <= 1e-16) continue;
      double flux_scale = sign_weight * nx;

      for (int ia = 0; ia < nq; ia++) {
        double alpha = q_pts[ia];
        double wa    = flux_scale * alpha * q_wts[ia];
        double c0    = 1.0 - alpha;
        for (int ib = 0; ib < nq; ib++) {
          double beta = q_pts[ib];
          double w    = wa * q_wts[ib];
          double c1   = alpha * (1.0 - beta);
          double c2   = alpha * beta;

          double t = c0 * p0[1] + c1 * p1[1] + c2 * p2[1];
          double u = c0 * p0[2] + c1 * p1[2] + c2 * p2[2];
          double Pt[5], Pu[5], dummy[5];
          EvalShiftedLegendre1D(t, Pt, dummy);
          EvalShiftedLegendre1D(u, Pu, dummy);

          if (on_s1_cap) {
            // At s = 1, intP[0](1) = 1 and intP[a >= 1](1) = 0 identically.
            for (int b = 0; b < nmom; b++) {
              double wb = w * Pt[b];
              for (int c = 0; c < nmom; c++) { M[0][b][c] += wb * Pu[c]; }
            }
          } else {
            double s = c0 * p0[0] + c1 * p1[0] + c2 * p2[0];
            double Ps[5], intPs[5];
            EvalShiftedLegendre1D(s, Ps, intPs);
            for (int a = 0; a < nmom; a++) {
              double wa_s = w * intPs[a];
              for (int b = 0; b < nmom; b++) {
                double wab = wa_s * Pt[b];
                for (int c = 0; c < nmom; c++) { M[a][b][c] += wab * Pu[c]; }
              }
            }
          }
        }
      }
    }
  };

  // 1. Outward-oriented exterior boundary faces of positively-oriented tetrahedron (v0, v1, v2,
  // v3):
  //    (v0, v2, v1), (v0, v1, v3), (v0, v3, v2), (v1, v2, v3)
  const double* faces[4][3] = {
      {v0, v2, v1},
      {v0, v1, v3},
      {v0, v3, v2},
      {v1, v2, v3},
  };

  for (int f = 0; f < 4; f++) {
    const double* a      = faces[f][0];
    const double* b      = faces[f][1];
    const double* c      = faces[f][2];
    double        min_fs = std::min({a[0], b[0], c[0]});
    if (!(boundary_mask & (1 << f)) && min_fs < 1.0 - kClipEps) continue;
    if (std::max({a[0], b[0], c[0]}) <= kClipEps ||
        min_fs >= 1.0 + kClipEps ||
        std::max({a[1], b[1], c[1]}) <= kClipEps ||
        std::min({a[1], b[1], c[1]}) >= 1.0 - kClipEps ||
        std::max({a[2], b[2], c[2]}) <= kClipEps ||
        std::min({a[2], b[2], c[2]}) >= 1.0 - kClipEps) {
      continue;
    }
    double nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    if (std::abs(nx) <= 1e-16) continue;

    double buf0[16][3], buf1[16][3];
    for (int d = 0; d < 3; d++) {
      buf0[0][d] = a[d];
      buf0[1][d] = b[d];
      buf0[2][d] = c[d];
    }
    int n = 3;
    // Faces lying in the s = 1 plane are integrated as part of the cap: their
    // s-coordinate is not used, and clipping them against s <= 1 with zero
    // tolerance would drop the whole face when a vertex rounds to 1 + eps.
    // Faces that merely touch the plane and extend beyond it are clipped as
    // usual (their in-cell area is zero).
    double max_fs    = std::max({a[0], b[0], c[0]});
    bool   on_s1_cap = min_fs >= 1.0 - kClipEps && max_fs <= 1.0 + kClipEps;
    if (!on_s1_cap) {
      n = ClipPolygonAxis(buf0, n, 0, 0.0, -1.0, buf1);
      n = ClipPolygonAxis(buf1, n, 0, 1.0, +1.0, buf0);
    }
    n = ClipPolygonAxis(buf0, n, 1, 0.0, -1.0, buf1);
    n = ClipPolygonAxis(buf1, n, 1, 1.0, +1.0, buf0);
    n = ClipPolygonAxis(buf0, n, 2, 0.0, -1.0, buf1);
    n = ClipPolygonAxis(buf1, n, 2, 1.0, +1.0, buf0);
    integrate_clipped_poly(buf0, n, on_s1_cap);
  }

  // 2. Cap polygon on the plane s = 1 when the tetrahedron strictly straddles s = 1
  if (min_s < 1.0 - kClipEps && max_s > 1.0 + kClipEps) {
    const double* verts[4] = {v0, v1, v2, v3};
    int           idx_le[4], idx_gt[4];
    int           n_le = 0, n_gt = 0;
    for (int i = 0; i < 4; i++) {
      if (verts[i][0] <= 1.0) {
        idx_le[n_le++] = i;
      } else {
        idx_gt[n_gt++] = i;
      }
    }
    if (n_le > 0 && n_gt > 0) {
      auto edge_point = [&](int i, int j, double out[3]) {
        const double* pi = verts[i];
        const double* pj = verts[j];
        double        t  = (1.0 - pi[0]) / (pj[0] - pi[0]);
        out[0]           = 1.0;
        out[1]           = pi[1] + t * (pj[1] - pi[1]);
        out[2]           = pi[2] + t * (pj[2] - pi[2]);
      };

      double buf0[16][3], buf1[16][3];
      int    n = 0;
      if (n_le == 1) {
        for (int k = 0; k < 3; k++) { edge_point(idx_le[0], idx_gt[k], buf0[n++]); }
      } else if (n_gt == 1) {
        for (int k = 0; k < 3; k++) { edge_point(idx_le[k], idx_gt[0], buf0[n++]); }
      } else {
        // n_le == 2, n_gt == 2: cyclic perimeter order around K_{2,2}
        edge_point(idx_le[0], idx_gt[0], buf0[0]);
        edge_point(idx_le[0], idx_gt[1], buf0[1]);
        edge_point(idx_le[1], idx_gt[1], buf0[2]);
        edge_point(idx_le[1], idx_gt[0], buf0[3]);
        n = 4;
      }

      // orient cap polygon with outward normal +e_s (positive 2D area in (t, u))
      double area2 = 0.0;
      for (int m = 1; m + 1 < n; m++) {
        area2 += (buf0[m][1] - buf0[0][1]) * (buf0[m + 1][2] - buf0[0][2]) -
                 (buf0[m][2] - buf0[0][2]) * (buf0[m + 1][1] - buf0[0][1]);
      }
      if (area2 < 0.0) {
        for (int i = 1, j = n - 1; i < j; i++, j--) {
          std::swap(buf0[i][0], buf0[j][0]);
          std::swap(buf0[i][1], buf0[j][1]);
          std::swap(buf0[i][2], buf0[j][2]);
        }
      }

      n = ClipPolygonAxis(buf0, n, 1, 0.0, -1.0, buf1);
      n = ClipPolygonAxis(buf1, n, 1, 1.0, +1.0, buf0);
      n = ClipPolygonAxis(buf0, n, 2, 0.0, -1.0, buf1);
      n = ClipPolygonAxis(buf1, n, 2, 1.0, +1.0, buf0);
      integrate_clipped_poly(buf0, n, true);
    }
  }
}

// test if a triangle mesh (fdim == 2) is a closed, consistently-oriented surface
// after welding coincident 3D vertex positions
static bool IsClosedSurfaceMesh(const double* vpos,
                                const int*    elems,
                                int           ne,
                                const double  minmax[6]) {
  if (ne < 4) return false;
  double max_ext = std::max({
      minmax[3] - minmax[0],
      minmax[4] - minmax[1],
      minmax[5] - minmax[2],
      1e-6,
  });
  double inv_tol = 1e9 / max_ext;

  struct QuantVert {
    std::int64_t x, y, z;
    bool         operator==(const QuantVert& o) const { return x == o.x && y == o.y && z == o.z; }
  };
  struct QuantVertHash {
    std::size_t operator()(const QuantVert& v) const {
      std::size_t h  = std::hash<std::int64_t>{}(v.x);
      h             ^= std::hash<std::int64_t>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
      h             ^= std::hash<std::int64_t>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
      return h;
    }
  };

  std::unordered_map<QuantVert, int, QuantVertHash> welded_id;
  welded_id.reserve(ne * 2);
  auto get_welded_id = [&](int vid) {
    QuantVert q{
        static_cast<std::int64_t>(std::llround((vpos[3 * vid + 0] - minmax[0]) * inv_tol)),
        static_cast<std::int64_t>(std::llround((vpos[3 * vid + 1] - minmax[1]) * inv_tol)),
        static_cast<std::int64_t>(std::llround((vpos[3 * vid + 2] - minmax[2]) * inv_tol)),
    };
    auto [it, inserted] = welded_id.emplace(q, static_cast<int>(welded_id.size()));
    return it->second;
  };

  std::unordered_map<std::pair<int, int>, std::pair<int, int>, PairHash> edge_counts;
  edge_counts.reserve(ne * 2);
  int n_nondegen = 0;
  for (int e = 0; e < ne; e++) {
    int w[3] = {
        get_welded_id(elems[3 * e + 0]),
        get_welded_id(elems[3 * e + 1]),
        get_welded_id(elems[3 * e + 2]),
    };
    if (w[0] == w[1] || w[1] == w[2] || w[2] == w[0]) continue;
    n_nondegen++;
    for (int k = 0; k < 3; k++) {
      int   u       = w[k];
      int   v       = w[(k + 1) % 3];
      auto  key     = std::minmax(u, v);
      int   sign    = (u < v) ? 1 : -1;
      auto& entry   = edge_counts[key];
      entry.first  += 1;
      entry.second += sign;
    }
  }
  if (n_nondegen < 4) return false;
  for (const auto& [key, counts] : edge_counts) {
    if (counts.first < 2 || (counts.first % 2) != 0 || counts.second != 0) { return false; }
  }
  return true;
}


// identify cells with no mesh content and compute FCM subcell quadrature
// weights
void mjCFlex::ComputeCellEmpty(
    const double* vpos, const int* elems, int ne, int fdim, const double minmax[6]) {
  int cx     = spec.cellcount[0];
  int cy     = spec.cellcount[1];
  int cz     = spec.cellcount[2];
  int ncells = cx * cy * cz;

  double dx = minmax[3] - minmax[0];
  double dy = minmax[4] - minmax[1];
  double dz = minmax[5] - minmax[2];

  cell_empty.assign(ncells, false);
  cell_full.assign(ncells, true);
  cell_vol.assign(ncells, 1.0);
  cell_alphas.assign(ncells, {});

  int nvpe = fdim + 1;
  if (nvpe <= 0 || ne <= 0 || ncells <= 0) { return; }

  std::string fcm_cache_key = ComputeFcmCacheKey(vpos, elems, ne, fdim, minmax);
  if (LoadCachedFcmGeometry(fcm_cache_key)) { return; }

  auto cellIdxLo = [](double coord, double lo, double d, int nc) {
    if (d <= 0) return 0;
    double eps = 1e-9 * d;
    int    c   = static_cast<int>(std::floor((coord - eps - lo) / d * nc));
    return std::max(0, std::min(nc - 1, c));
  };
  auto cellIdxHi = [](double coord, double lo, double d, int nc) {
    if (d <= 0) return 0;
    double eps = 1e-9 * d;
    int    c   = static_cast<int>(std::floor((coord + eps - lo) / d * nc));
    return std::max(0, std::min(nc - 1, c));
  };

  // fallback to AABB overlap if any grid box dimension is degenerate
  if (dx <= mjMINVAL || dy <= mjMINVAL || dz <= mjMINVAL) {
    std::vector<bool> has_element(ncells, false);
    for (int e = 0; e < ne; e++) {
      double elo[3] = {1e30, 1e30, 1e30};
      double ehi[3] = {-1e30, -1e30, -1e30};
      for (int v = 0; v < nvpe; v++) {
        int vid = elems[nvpe * e + v];
        for (int j = 0; j < 3; j++) {
          elo[j] = std::min(elo[j], vpos[3 * vid + j]);
          ehi[j] = std::max(ehi[j], vpos[3 * vid + j]);
        }
      }
      int ci0 = cellIdxLo(elo[0], minmax[0], dx, cx);
      int ci1 = cellIdxHi(ehi[0], minmax[0], dx, cx);
      int cj0 = cellIdxLo(elo[1], minmax[1], dy, cy);
      int cj1 = cellIdxHi(ehi[1], minmax[1], dy, cy);
      int ck0 = cellIdxLo(elo[2], minmax[2], dz, cz);
      int ck1 = cellIdxHi(ehi[2], minmax[2], dz, cz);
      for (int ci = ci0; ci <= ci1; ci++) {
        for (int cj = cj0; cj <= cj1; cj++) {
          for (int ck = ck0; ck <= ck1; ck++) { has_element[ci * cy * cz + cj * cz + ck] = true; }
        }
      }
    }
    if (ncells > 1) {
      for (int c = 0; c < ncells; c++) {
        cell_empty[c] = !has_element[c];
        cell_full[c]  = has_element[c];
        cell_vol[c]   = has_element[c] ? 1.0 : 0.0;
      }
    }
    return;
  }

  double hx = dx / cx;
  double hy = dy / cy;
  double hz = dz / cz;
  double mx = (cx > 1) ? kSliverMargin * hx : 0.0;
  double my = (cy > 1) ? kSliverMargin * hy : 0.0;
  double mz = (cz > 1) ? kSliverMargin * hz : 0.0;

  int                 order   = std::max(1, spec.order);
  int                 nbasis  = order + 1;
  int                 nq_axis = kFcmSubdiv * nbasis;
  int                 nq_cell = nq_axis * nq_axis * nq_axis;
  int                 nmom    = 2 * order + 1;
  std::vector<double> qpts(nq_axis);
  std::vector<double> qwts(nq_axis);
  for (int sub = 0; sub < kFcmSubdiv; sub++) {
    quadratureGaussLegendre(qpts.data() + sub * nbasis,
                            qwts.data() + sub * nbasis,
                            nbasis,
                            static_cast<double>(sub) / kFcmSubdiv,
                            static_cast<double>(sub + 1) / kFcmSubdiv);
  }

  double H[6][5] = {};
  BuildLegendreMomentProjector(order, qpts.data(), qwts.data(), H);

  int    n_duffy          = (order == 1) ? 5 : 8;
  double duffy_pts[8]     = {};
  double duffy_wts[8]     = {};
  int    n_duffy_cap      = 2 * order + 1;
  double duffy_cap_pts[5] = {};
  double duffy_cap_wts[5] = {};
  quadratureGaussLegendre(duffy_pts, duffy_wts, n_duffy, 0.0, 1.0);
  quadratureGaussLegendre(duffy_cap_pts, duffy_cap_wts, n_duffy_cap, 0.0, 1.0);

  double inv_hx = 1.0 / hx;
  double inv_hy = 1.0 / hy;
  double inv_hz = 1.0 / hz;

  auto moments_to_cell_quad = [&](const double M[5][5][5], int c) {
    double vol_exact = M[0][0][0];
    cell_empty[c]    = false;
    if (vol_exact >= 1.0 - 1e-7) {
      cell_full[c] = true;
      cell_vol[c]  = 1.0;
      cell_alphas[c].clear();
    } else if (vol_exact <= 1e-12) {
      cell_full[c] = false;
      cell_vol[c]  = kFcmAlphaMin;
      cell_alphas[c].assign(nq_cell, kFcmAlphaMin);
    } else {
      double T1[5][5][6] = {};
      for (int a = 0; a < nmom; a++) {
        for (int b = 0; b < nmom; b++) {
          for (int pu = 0; pu < nq_axis; pu++) {
            double s = 0.0;
            for (int mc = 0; mc < nmom; mc++) { s += M[a][b][mc] * H[pu][mc]; }
            T1[a][b][pu] = s;
          }
        }
      }
      double T2[5][6][6] = {};
      for (int a = 0; a < nmom; a++) {
        for (int pt = 0; pt < nq_axis; pt++) {
          for (int pu = 0; pu < nq_axis; pu++) {
            double s = 0.0;
            for (int mb = 0; mb < nmom; mb++) { s += T1[a][mb][pu] * H[pt][mb]; }
            T2[a][pt][pu] = s;
          }
        }
      }
      std::vector<double> alphas(nq_cell);
      int                 q_idx = 0;
      for (int ps = 0; ps < nq_axis; ps++) {
        for (int pt = 0; pt < nq_axis; pt++) {
          for (int pu = 0; pu < nq_axis; pu++) {
            double s = 0.0;
            for (int ma = 0; ma < nmom; ma++) { s += T2[ma][pt][pu] * H[ps][ma]; }
            alphas[q_idx++] = kFcmAlphaMin + (1.0 - kFcmAlphaMin) * s;
          }
        }
      }
      cell_full[c]   = false;
      cell_vol[c]    = kFcmAlphaMin + (1.0 - kFcmAlphaMin) * std::min(1.0, vol_exact);
      cell_alphas[c] = std::move(alphas);
    }
  };

  std::vector<bool> has_element(ncells, false);
  std::vector<bool> has_core(ncells, false);

  auto integrate_tet_collection = [&](auto get_tet_verts) {
    std::vector<TetCellData>      valid_tets;
    std::vector<std::vector<int>> cell_tets(ncells);
    double                        vol_scale = hx * hy * hz;
    double                        bbox_eps  = 1e-7 * std::max({hx, hy, hz});

    for (int e = 0; e < ne; e++) {
      const double *v0, *v1, *v2, *v3;
      get_tet_verts(e, v0, v1, v2, v3);

      double elo[3], ehi[3];
      for (int j = 0; j < 3; j++) {
        elo[j] = std::min({v0[j], v1[j], v2[j], v3[j]});
        ehi[j] = std::max({v0[j], v1[j], v2[j], v3[j]});
      }

      double d1[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
      double d2[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
      double d3[3] = {v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]};
      double c1[3], c2[3], c3[3];
      mjuu_crossvec(c1, d2, d3);
      mjuu_crossvec(c2, d3, d1);
      mjuu_crossvec(c3, d1, d2);
      double det = mjuu_dot3(d1, c1);

      int                tet_id  = -1;
      const TetCellData* tet_ptr = nullptr;
      if (std::abs(det) > 1e-14 * vol_scale) {
        TetCellData td;
        mjuu_copyvec(td.v[0], v0, 3);
        if (det > 0.0) {
          mjuu_copyvec(td.v[1], v1, 3);
          mjuu_copyvec(td.v[2], v2, 3);
        } else {
          mjuu_copyvec(td.v[1], v2, 3);
          mjuu_copyvec(td.v[2], v1, 3);
        }
        mjuu_copyvec(td.v[3], v3, 3);
        mjuu_copyvec(td.v0, v0, 3);
        double inv_det = 1.0 / det;
        for (int j = 0; j < 3; j++) {
          td.invM[0][j] = c1[j] * inv_det;
          td.invM[1][j] = c2[j] * inv_det;
          td.invM[2][j] = c3[j] * inv_det;
          td.lo[j]      = elo[j] - bbox_eps;
          td.hi[j]      = ehi[j] + bbox_eps;
        }
        tet_id = static_cast<int>(valid_tets.size());
        valid_tets.push_back(td);
        tet_ptr = &valid_tets.back();
      }

      int ci0 = cellIdxLo(elo[0], minmax[0], dx, cx);
      int ci1 = cellIdxHi(ehi[0], minmax[0], dx, cx);
      int cj0 = cellIdxLo(elo[1], minmax[1], dy, cy);
      int cj1 = cellIdxHi(ehi[1], minmax[1], dy, cy);
      int ck0 = cellIdxLo(elo[2], minmax[2], dz, cz);
      int ck1 = cellIdxHi(ehi[2], minmax[2], dz, cz);

      for (int ci = ci0; ci <= ci1; ci++) {
        double bmin_x = minmax[0] + ci * hx;
        double bmax_x = bmin_x + hx;
        for (int cj = cj0; cj <= cj1; cj++) {
          double bmin_y = minmax[1] + cj * hy;
          double bmax_y = bmin_y + hy;
          for (int ck = ck0; ck <= ck1; ck++) {
            double bmin_z  = minmax[2] + ck * hz;
            double bmax_z  = bmin_z + hz;
            double bmin[3] = {bmin_x, bmin_y, bmin_z};
            double bmax[3] = {bmax_x, bmax_y, bmax_z};
            if (TetIntersectsBox(v0, v1, v2, v3, tet_ptr, bmin, bmax)) {
              int c          = ci * cy * cz + cj * cz + ck;
              has_element[c] = true;
              double cmin[3] = {bmin_x + mx, bmin_y + my, bmin_z + mz};
              double cmax[3] = {bmax_x - mx, bmax_y - my, bmax_z - mz};
              if (TetIntersectsBox(v0, v1, v2, v3, tet_ptr, cmin, cmax)) { has_core[c] = true; }
              if (tet_id >= 0) { cell_tets[c].push_back(tet_id); }
            }
          }
        }
      }
    }

    bool any_core = false;
    for (bool core : has_core) {
      if (core) {
        any_core = true;
        break;
      }
    }

    if (valid_tets.empty()) {
      for (int c = 0; c < ncells; c++) {
        bool active   = any_core ? has_core[c] : has_element[c];
        bool empty    = (ncells > 1) && !active;
        cell_empty[c] = empty;
        cell_full[c]  = !empty;
        cell_vol[c]   = empty ? 0.0 : 1.0;
      }
      return;
    }

    // Cancel shared interior faces with opposite orientation across valid_tets
    double max_ext = std::max({dx, dy, dz, 1e-6});
    double inv_tol = 1e9 / max_ext;
    struct QuantVert {
      std::int64_t x, y, z;
      bool         operator==(const QuantVert& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct QuantVertHash {
      std::size_t operator()(const QuantVert& v) const {
        std::size_t h  = std::hash<std::int64_t>{}(v.x);
        h             ^= std::hash<std::int64_t>{}(v.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h             ^= std::hash<std::int64_t>{}(v.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
      }
    };
    struct FaceKey {
      int  a, b, c;
      bool operator==(const FaceKey& o) const { return a == o.a && b == o.b && c == o.c; }
    };
    struct FaceKeyHash {
      std::size_t operator()(const FaceKey& f) const {
        std::size_t h  = std::hash<int>{}(f.a);
        h             ^= std::hash<int>{}(f.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h             ^= std::hash<int>{}(f.c) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
      }
    };
    struct FaceMatch {
      int count    = 0;
      int sign_sum = 0;
      int tid0     = -1;
      int f0       = -1;
      int tid1     = -1;
      int f1       = -1;
    };

    std::unordered_map<QuantVert, int, QuantVertHash> welded_id;
    welded_id.reserve(valid_tets.size() * 2);
    auto get_welded_id = [&](const double p[3]) {
      QuantVert q{
          static_cast<std::int64_t>(std::llround((p[0] - minmax[0]) * inv_tol)),
          static_cast<std::int64_t>(std::llround((p[1] - minmax[1]) * inv_tol)),
          static_cast<std::int64_t>(std::llround((p[2] - minmax[2]) * inv_tol)),
      };
      auto [it, inserted] = welded_id.emplace(q, static_cast<int>(welded_id.size()));
      return it->second;
    };

    static constexpr int kTetFaces[4][3] = {
        {0, 2, 1},
        {0, 1, 3},
        {0, 3, 2},
        {1, 2, 3},
    };
    std::unordered_map<FaceKey, FaceMatch, FaceKeyHash> face_map;
    face_map.reserve(valid_tets.size() * 4);
    for (int tid = 0; tid < static_cast<int>(valid_tets.size()); tid++) {
      const TetCellData& td   = valid_tets[tid];
      int                w[4] = {
          get_welded_id(td.v[0]),
          get_welded_id(td.v[1]),
          get_welded_id(td.v[2]),
          get_welded_id(td.v[3]),
      };
      for (int f = 0; f < 4; f++) {
        int u0 = w[kTetFaces[f][0]];
        int u1 = w[kTetFaces[f][1]];
        int u2 = w[kTetFaces[f][2]];
        int s0 = u0, s1 = u1, s2 = u2, sign = 1;
        if (s0 > s1) {
          std::swap(s0, s1);
          sign = -sign;
        }
        if (s1 > s2) {
          std::swap(s1, s2);
          sign = -sign;
        }
        if (s0 > s1) {
          std::swap(s0, s1);
          sign = -sign;
        }
        auto& entry = face_map[FaceKey{s0, s1, s2}];
        if (entry.count == 0) {
          entry.tid0 = tid;
          entry.f0   = f;
        } else if (entry.count == 1) {
          entry.tid1 = tid;
          entry.f1   = f;
        }
        entry.count    += 1;
        entry.sign_sum += sign;
      }
    }
    for (const auto& [key, entry] : face_map) {
      if (entry.count == 2 && entry.sign_sum == 0) {
        valid_tets[entry.tid0].boundary_mask &= ~(1 << entry.f0);
        valid_tets[entry.tid1].boundary_mask &= ~(1 << entry.f1);
      }
    }

    for (int ci = 0; ci < cx; ci++) {
      double bmin_x = minmax[0] + ci * hx;
      for (int cj = 0; cj < cy; cj++) {
        double bmin_y = minmax[1] + cj * hy;
        for (int ck = 0; ck < cz; ck++) {
          double bmin_z = minmax[2] + ck * hz;
          int    c      = ci * cy * cz + cj * cz + ck;
          bool   active = any_core ? has_core[c] : has_element[c];
          if (!active) {
            if (ncells > 1) {
              cell_empty[c] = true;
              cell_full[c]  = false;
              cell_vol[c]   = 0.0;
            } else {
              cell_empty[c] = false;
              cell_full[c]  = false;
              cell_vol[c]   = kFcmAlphaMin;
              cell_alphas[c].assign(nq_cell, kFcmAlphaMin);
            }
            continue;
          }

          const auto& tets_c            = cell_tets[c];
          double      bmin[3]           = {bmin_x, bmin_y, bmin_z};
          double      bmax[3]           = {bmin_x + hx, bmin_y + hy, bmin_z + hz};
          bool        has_boundary_face = false;
          for (int tid : tets_c) {
            const TetCellData& td = valid_tets[tid];
            if (td.boundary_mask == 0) continue;
            for (int f = 0; f < 4; f++) {
              if (!(td.boundary_mask & (1 << f))) continue;
              if (TriangleIntersectsBox(td.v[kTetFaces[f][0]],
                                        td.v[kTetFaces[f][1]],
                                        td.v[kTetFaces[f][2]],
                                        bmin,
                                        bmax)) {
                has_boundary_face = true;
                break;
              }
            }
            if (has_boundary_face) break;
          }
          if (!has_boundary_face) {
            cell_empty[c] = false;
            cell_full[c]  = true;
            cell_vol[c]   = 1.0;
            cell_alphas[c].clear();
            continue;
          }

          double M[5][5][5] = {};
          for (int tid : tets_c) {
            const TetCellData& td = valid_tets[tid];
            double             lv[4][3];
            for (int k = 0; k < 4; k++) {
              lv[k][0] = (td.v[k][0] - bmin_x) * inv_hx;
              lv[k][1] = (td.v[k][1] - bmin_y) * inv_hy;
              lv[k][2] = (td.v[k][2] - bmin_z) * inv_hz;
            }
            AccumulateClippedTetMoments(lv[0],
                                        lv[1],
                                        lv[2],
                                        lv[3],
                                        td.boundary_mask,
                                        1.0,
                                        nmom,
                                        n_duffy,
                                        duffy_pts,
                                        duffy_wts,
                                        n_duffy_cap,
                                        duffy_cap_pts,
                                        duffy_cap_wts,
                                        M);
          }

          moments_to_cell_quad(M, c);
        }
      }
    }
  };

  if (fdim == 3) {
    integrate_tet_collection(
        [&](int e, const double*& v0, const double*& v1, const double*& v2, const double*& v3) {
          v0 = vpos + 3 * elems[4 * e + 0];
          v1 = vpos + 3 * elems[4 * e + 1];
          v2 = vpos + 3 * elems[4 * e + 2];
          v3 = vpos + 3 * elems[4 * e + 3];
        });
  } else if (fdim == 2) {
    double origin[3] = {
        0.5 * (minmax[0] + minmax[3]),
        0.5 * (minmax[1] + minmax[4]),
        0.5 * (minmax[2] + minmax[5]),
    };
    double vol_scale = hx * hy * hz;
    double total_det = 0.0;
    bool   has_pos   = false;
    bool   has_neg   = false;
    bool   is_closed = IsClosedSurfaceMesh(vpos, elems, ne, minmax);
    if (is_closed) {
      for (int e = 0; e < ne; e++) {
        const double* a     = vpos + 3 * elems[3 * e + 0];
        const double* b     = vpos + 3 * elems[3 * e + 1];
        const double* c     = vpos + 3 * elems[3 * e + 2];
        double        d1[3] = {a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]};
        double        d2[3] = {b[0] - origin[0], b[1] - origin[1], b[2] - origin[2]};
        double        d3[3] = {c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]};
        double        c1[3];
        mjuu_crossvec(c1, d2, d3);
        double det  = mjuu_dot3(d1, c1);
        total_det  += det;
        if (det > 1e-14 * vol_scale) has_pos = true;
        if (det < -1e-14 * vol_scale) has_neg = true;
      }
      if (std::abs(total_det) <= 1e-12 * vol_scale) { is_closed = false; }
    }

    // If the closed surface is star-shaped with respect to the bounding-box
    // center (all non-degenerate cones have the same orientation sign), the
    // cones (origin, a, b, c) form an exact tetrahedral partition of the
    // interior, identical to fdim == 3.
    if (is_closed && !(has_pos && has_neg)) {
      integrate_tet_collection(
          [&](int e, const double*& v0, const double*& v1, const double*& v2, const double*& v3) {
            v0 = origin;
            v1 = vpos + 3 * elems[3 * e + 0];
            v2 = vpos + 3 * elems[3 * e + 1];
            v3 = vpos + 3 * elems[3 * e + 2];
          });
      CacheFcmGeometry(fcm_cache_key);
      return;
    }

    std::vector<TriCellData>      tris(ne);
    std::vector<std::vector<int>> cell_tris(ncells);
    for (int e = 0; e < ne; e++) {
      const double* v0 = vpos + 3 * elems[3 * e + 0];
      const double* v1 = vpos + 3 * elems[3 * e + 1];
      const double* v2 = vpos + 3 * elems[3 * e + 2];
      mjuu_copyvec(tris[e].a, v0, 3);
      mjuu_copyvec(tris[e].b, v1, 3);
      mjuu_copyvec(tris[e].c, v2, 3);

      double elo[3], ehi[3];
      for (int j = 0; j < 3; j++) {
        elo[j] = std::min({v0[j], v1[j], v2[j]});
        ehi[j] = std::max({v0[j], v1[j], v2[j]});
      }

      int ci0 = cellIdxLo(elo[0], minmax[0], dx, cx);
      int ci1 = cellIdxHi(ehi[0], minmax[0], dx, cx);
      int cj0 = cellIdxLo(elo[1], minmax[1], dy, cy);
      int cj1 = cellIdxHi(ehi[1], minmax[1], dy, cy);
      int ck0 = cellIdxLo(elo[2], minmax[2], dz, cz);
      int ck1 = cellIdxHi(ehi[2], minmax[2], dz, cz);

      for (int ci = ci0; ci <= ci1; ci++) {
        double bmin_x = minmax[0] + ci * hx;
        double bmax_x = bmin_x + hx;
        for (int cj = cj0; cj <= cj1; cj++) {
          double bmin_y = minmax[1] + cj * hy;
          double bmax_y = bmin_y + hy;
          for (int ck = ck0; ck <= ck1; ck++) {
            double bmin_z  = minmax[2] + ck * hz;
            double bmax_z  = bmin_z + hz;
            double bmin[3] = {bmin_x, bmin_y, bmin_z};
            double bmax[3] = {bmax_x, bmax_y, bmax_z};
            if (TriangleIntersectsBox(v0, v1, v2, bmin, bmax)) {
              int c          = ci * cy * cz + cj * cz + ck;
              has_element[c] = true;
              double cmin[3] = {bmin_x + mx, bmin_y + my, bmin_z + mz};
              double cmax[3] = {bmax_x - mx, bmax_y - my, bmax_z - mz};
              if (TriangleIntersectsBox(v0, v1, v2, cmin, cmax)) { has_core[c] = true; }
              cell_tris[c].push_back(e);
            }
          }
        }
      }
    }

    // Build Fast Winding Number BVH (Barill et al. 2018) once for the mesh
    FastWindingBVH winding_bvh;
    winding_bvh.Build(tris);

    // Cells intersecting the shrunken core (has_core) are always active
    // (including open sheets). WindingNumber is only evaluated once per
    // connected component of non-intersected cells (propagated from adjacent
    // !has_element neighbors) to keep hollow interiors of closed shells, plus
    // on pure sliver cells (has_element && !has_core).
    constexpr double  kWindingThreshold = 0.5 - 1e-5;
    bool              any_inside        = is_closed;
    std::vector<bool> is_cut_cell(ncells, false);

    for (int ci = 0; ci < cx; ci++) {
      double bmin_x = minmax[0] + ci * hx;
      for (int cj = 0; cj < cy; cj++) {
        double bmin_y = minmax[1] + cj * hy;
        for (int ck = 0; ck < cz; ck++) {
          double bmin_z = minmax[2] + ck * hz;
          int    c      = ci * cy * cz + cj * cz + ck;

          if (!has_element[c] || (!has_core[c] && ncells > 1)) {
            bool inside = false;
            if (!has_element[c] && ck > 0 && !has_element[c - 1]) {
              inside = cell_full[c - 1];
            } else if (!has_element[c] && cj > 0 && !has_element[c - cz]) {
              inside = cell_full[c - cz];
            } else if (!has_element[c] && ci > 0 && !has_element[c - cy * cz]) {
              inside = cell_full[c - cy * cz];
            } else {
              double center[3] = {bmin_x + 0.5 * hx, bmin_y + 0.5 * hy, bmin_z + 0.5 * hz};
              inside           = std::abs(winding_bvh.Evaluate(center)) >= kWindingThreshold;
            }
            if (!has_element[c]) {
              if (inside) {
                any_inside    = true;
                cell_empty[c] = false;
                cell_full[c]  = true;
                cell_vol[c]   = 1.0;
              } else {
                cell_empty[c] = (ncells > 1);
                cell_full[c]  = false;
                cell_vol[c]   = (ncells > 1) ? 0.0 : kFcmAlphaMin;
                if (ncells == 1) { cell_alphas[c].assign(nq_cell, kFcmAlphaMin); }
              }
              continue;
            } else if (!inside) {
              // surface only grazes outer sliver margin from outside: prune
              // cell
              cell_empty[c] = true;
              cell_full[c]  = false;
              cell_vol[c]   = 0.0;
              continue;
            }
          }

          if (is_closed) {
            is_cut_cell[c] = true;
            continue;
          }

          // Fallback for open/non-watertight triangle soups, which have no
          // closed boundary for the divergence theorem to integrate over:
          // classify each composite Gauss point with the FastWindingBVH
          // indicator (|w| >= 0.5) and give it full or kFcmAlphaMin weight.
          // This is the only path that still uses point sampling, and it is
          // kept deliberately for such meshes.
          const auto& tris_c                                      = cell_tris[c];
          bool        sub_cut[kFcmSubdiv][kFcmSubdiv][kFcmSubdiv] = {};
          bool        sub_in[kFcmSubdiv][kFcmSubdiv][kFcmSubdiv]  = {};
          for (int si = 0; si < kFcmSubdiv; si++) {
            double smin_x = bmin_x + (si * hx) / kFcmSubdiv;
            double smax_x = bmin_x + ((si + 1) * hx) / kFcmSubdiv;
            for (int sj = 0; sj < kFcmSubdiv; sj++) {
              double smin_y = bmin_y + (sj * hy) / kFcmSubdiv;
              double smax_y = bmin_y + ((sj + 1) * hy) / kFcmSubdiv;
              for (int sk = 0; sk < kFcmSubdiv; sk++) {
                double smin[3] = {smin_x, smin_y, bmin_z + (sk * hz) / kFcmSubdiv};
                double smax[3] = {smax_x, smax_y, bmin_z + ((sk + 1) * hz) / kFcmSubdiv};
                bool   cut     = false;
                for (int tid : tris_c) {
                  if (TriangleIntersectsBox(tris[tid].a, tris[tid].b, tris[tid].c, smin, smax)) {
                    cut = true;
                    break;
                  }
                }
                sub_cut[si][sj][sk] = cut;
                if (!cut) {
                  double sc[3]       = {0.5 * (smin[0] + smax[0]),
                                        0.5 * (smin[1] + smax[1]),
                                        0.5 * (smin[2] + smax[2])};
                  sub_in[si][sj][sk] = std::abs(winding_bvh.Evaluate(sc)) >= kWindingThreshold;
                }
              }
            }
          }

          std::vector<double> alphas(nq_cell, kFcmAlphaMin);
          int                 n_inside = 0;
          double              vol      = 0.0;
          int                 q_idx    = 0;

          for (int ps = 0; ps < nq_axis; ps++) {
            int    si      = ps / nbasis;
            double x_coord = bmin_x + qpts[ps] * hx;
            double ws      = qwts[ps];
            for (int pt = 0; pt < nq_axis; pt++) {
              int    sj      = pt / nbasis;
              double y_coord = bmin_y + qpts[pt] * hy;
              double wst     = ws * qwts[pt];
              for (int pu = 0; pu < nq_axis; pu++) {
                int    sk     = pu / nbasis;
                double w_q    = wst * qwts[pu];
                bool   inside = false;
                if (!sub_cut[si][sj][sk]) {
                  inside = sub_in[si][sj][sk];
                } else {
                  double x_q[3] = {x_coord, y_coord, bmin_z + qpts[pu] * hz};
                  inside        = std::abs(winding_bvh.Evaluate(x_q)) >= kWindingThreshold;
                }
                if (inside) {
                  alphas[q_idx] = 1.0;
                  n_inside++;
                  any_inside = true;
                }
                vol += w_q * alphas[q_idx];
                q_idx++;
              }
            }
          }

          cell_empty[c] = false;
          if (n_inside == nq_cell) {
            cell_full[c] = true;
            cell_vol[c]  = 1.0;
          } else {
            cell_full[c]   = false;
            cell_vol[c]    = vol;
            cell_alphas[c] = std::move(alphas);
          }
        }
      }
    }

    if (is_closed) {
      struct MomentGrid {
        double M[5][5][5] = {};
      };
      std::vector<MomentGrid> cell_moments(ncells);
      double                  orient = (total_det >= 0.0) ? 1.0 : -1.0;

      for (int e = 0; e < ne; e++) {
        const double* a = tris[e].a;
        const double* b = tris[e].b;
        const double* c = tris[e].c;

        double d1[3] = {a[0] - origin[0], a[1] - origin[1], a[2] - origin[2]};
        double d2[3] = {b[0] - origin[0], b[1] - origin[1], b[2] - origin[2]};
        double d3[3] = {c[0] - origin[0], c[1] - origin[1], c[2] - origin[2]};
        double c1[3];
        mjuu_crossvec(c1, d2, d3);
        double det = mjuu_dot3(d1, c1);

        // A triangle coplanar with origin forms a flat cone with no volume, but
        // its surface face (a, b, c) still carries boundary flux: the three
        // radial faces skipped under boundary_mask = 0x8 sum to exactly this
        // face, so dropping the triangle leaves the neighbors' skipped radial
        // faces uncancelled. Keep it with the mesh orientation: for both
        // det > 0 and det < 0 the signed flux of face (v1, v2, v3) equals
        // orient times the outward flux of (a, b, c).
        bool          positive = (det > 0.0) || std::abs(det) <= 1e-14 * vol_scale;
        double        sign_w   = positive ? orient : -orient;
        const double* tv[4]    = {
            origin,
            positive ? a : b,
            positive ? b : a,
            c,
        };

        double elo[3], ehi[3];
        for (int j = 0; j < 3; j++) {
          elo[j] = std::min({origin[j], a[j], b[j], c[j]});
          ehi[j] = std::max({origin[j], a[j], b[j], c[j]});
        }

        int ci0 = cellIdxLo(elo[0], minmax[0], dx, cx);
        int ci1 = cellIdxHi(ehi[0], minmax[0], dx, cx);
        int cj0 = cellIdxLo(elo[1], minmax[1], dy, cy);
        int cj1 = cellIdxHi(ehi[1], minmax[1], dy, cy);
        int ck0 = cellIdxLo(elo[2], minmax[2], dz, cz);
        int ck1 = cellIdxHi(ehi[2], minmax[2], dz, cz);

        for (int ci = ci0; ci <= ci1; ci++) {
          double bmin_x = minmax[0] + ci * hx;
          for (int cj = cj0; cj <= cj1; cj++) {
            double bmin_y = minmax[1] + cj * hy;
            for (int ck = ck0; ck <= ck1; ck++) {
              int cell_idx = ci * cy * cz + cj * cz + ck;
              if (!is_cut_cell[cell_idx]) continue;
              double bmin_z = minmax[2] + ck * hz;
              double lv[4][3];
              for (int k = 0; k < 4; k++) {
                lv[k][0] = (tv[k][0] - bmin_x) * inv_hx;
                lv[k][1] = (tv[k][1] - bmin_y) * inv_hy;
                lv[k][2] = (tv[k][2] - bmin_z) * inv_hz;
              }
              AccumulateClippedTetMoments(lv[0],
                                          lv[1],
                                          lv[2],
                                          lv[3],
                                          0x8,
                                          sign_w,
                                          nmom,
                                          n_duffy,
                                          duffy_pts,
                                          duffy_wts,
                                          n_duffy_cap,
                                          duffy_cap_pts,
                                          duffy_cap_wts,
                                          cell_moments[cell_idx].M);
            }
          }
        }
      }

      for (int c = 0; c < ncells; c++) {
        if (!is_cut_cell[c]) continue;
        moments_to_cell_quad(cell_moments[c].M, c);
      }
    }

    // if surface is open and encloses no volume, fallback to has_core /
    // has_element with full occupancy
    if (!any_inside) {
      bool any_core = false;
      for (bool core : has_core) {
        if (core) {
          any_core = true;
          break;
        }
      }
      for (int c = 0; c < ncells; c++) {
        bool active   = any_core ? has_core[c] : has_element[c];
        bool empty    = (ncells > 1) && !active;
        cell_empty[c] = empty;
        cell_full[c]  = !empty;
        cell_vol[c]   = empty ? 0.0 : 1.0;
        cell_alphas[c].clear();
      }
    }
  } else {
    for (int e = 0; e < ne; e++) {
      const double* v0 = vpos + 3 * elems[2 * e + 0];
      const double* v1 = vpos + 3 * elems[2 * e + 1];
      double        elo[3], ehi[3];
      for (int j = 0; j < 3; j++) {
        elo[j] = std::min(v0[j], v1[j]);
        ehi[j] = std::max(v0[j], v1[j]);
      }
      int ci0 = cellIdxLo(elo[0], minmax[0], dx, cx);
      int ci1 = cellIdxHi(ehi[0], minmax[0], dx, cx);
      int cj0 = cellIdxLo(elo[1], minmax[1], dy, cy);
      int cj1 = cellIdxHi(ehi[1], minmax[1], dy, cy);
      int ck0 = cellIdxLo(elo[2], minmax[2], dz, cz);
      int ck1 = cellIdxHi(ehi[2], minmax[2], dz, cz);
      for (int ci = ci0; ci <= ci1; ci++) {
        double bmin_x = minmax[0] + ci * hx;
        double bmax_x = bmin_x + hx;
        for (int cj = cj0; cj <= cj1; cj++) {
          double bmin_y = minmax[1] + cj * hy;
          double bmax_y = bmin_y + hy;
          for (int ck = ck0; ck <= ck1; ck++) {
            double bmin_z  = minmax[2] + ck * hz;
            double bmax_z  = bmin_z + hz;
            double bmin[3] = {bmin_x, bmin_y, bmin_z};
            double bmax[3] = {bmax_x, bmax_y, bmax_z};
            if (SegmentIntersectsBox(v0, v1, bmin, bmax)) {
              int c          = ci * cy * cz + cj * cz + ck;
              has_element[c] = true;
              double cmin[3] = {bmin_x + mx, bmin_y + my, bmin_z + mz};
              double cmax[3] = {bmax_x - mx, bmax_y - my, bmax_z - mz};
              if (SegmentIntersectsBox(v0, v1, cmin, cmax)) { has_core[c] = true; }
            }
          }
        }
      }
    }
    bool any_core = false;
    for (bool core : has_core) {
      if (core) {
        any_core = true;
        break;
      }
    }
    for (int c = 0; c < ncells; c++) {
      bool active   = any_core ? has_core[c] : has_element[c];
      bool empty    = (ncells > 1) && !active;
      cell_empty[c] = empty;
      cell_full[c]  = !empty;
      cell_vol[c]   = empty ? 0.0 : 1.0;
    }
  }
  CacheFcmGeometry(fcm_cache_key);
}


// return true if all cells adjacent to node (gi, gj, gk) are empty
bool mjCFlex::NodeInEmptyCells(int gi, int gj, int gk) const {
  if (cell_empty.empty() || spec.order <= 0) { return false; }
  int cx     = spec.cellcount[0];
  int cy     = spec.cellcount[1];
  int cz     = spec.cellcount[2];
  int order  = spec.order;
  int ci_min = std::max(0, gi == 0 ? 0 : (gi - 1) / order);
  int ci_max = std::min(cx - 1, gi / order);
  int cj_min = std::max(0, gj == 0 ? 0 : (gj - 1) / order);
  int cj_max = std::min(cy - 1, gj / order);
  int ck_min = std::max(0, gk == 0 ? 0 : (gk - 1) / order);
  int ck_max = std::min(cz - 1, gk / order);

  for (int ci = ci_min; ci <= ci_max; ci++) {
    for (int cj = cj_min; cj <= cj_max; cj++) {
      for (int ck = ck_min; ck <= ck_max; ck++) {
        if (!cell_empty[ci * cy * cz + cj * cz + ck]) { return false; }
      }
    }
  }
  return true;
}


// compute positive mass-lumping basis function on [0, 1] with exact
// Simpson/trilinear integrals
static inline double MassBasis1D(double s, int l, int order) {
  if (order == 1) { return (l == 0) ? (1.0 - s) : s; }
  if (l == 0) { return 0.5 * (1.0 - s) * (1.0 - s); }
  if (l == 1) { return 0.5 + s * (1.0 - s); }
  return 0.5 * s * s;
}


// compute FCM volume-weighted nodal mass weight for node (gi, gj, gk)
double mjCFlex::NodeMassWeight(int gi, int gj, int gk) const {
  int cx      = spec.cellcount[0];
  int cy      = spec.cellcount[1];
  int cz      = spec.cellcount[2];
  int order   = std::max(1, spec.order);
  int nbasis  = order + 1;
  int nq_axis = kFcmSubdiv * nbasis;

  double qpts[6];
  double qwts[6];
  for (int sub = 0; sub < kFcmSubdiv; sub++) {
    quadratureGaussLegendre(qpts + sub * nbasis,
                            qwts + sub * nbasis,
                            nbasis,
                            static_cast<double>(sub) / kFcmSubdiv,
                            static_cast<double>(sub + 1) / kFcmSubdiv);
  }
  double m1d[3][6] = {};
  for (int l = 0; l < nbasis; l++) {
    for (int p = 0; p < nq_axis; p++) { m1d[l][p] = qwts[p] * MassBasis1D(qpts[p], l, order); }
  }

  const double  massP1[2] = {0.5, 0.5};
  const double  massP2[3] = {1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0};
  const double* full_w    = (order == 1) ? massP1 : massP2;

  int ci_min = std::max(0, gi == 0 ? 0 : (gi - 1) / order);
  int ci_max = std::min(cx - 1, gi / order);
  int cj_min = std::max(0, gj == 0 ? 0 : (gj - 1) / order);
  int cj_max = std::min(cy - 1, gj / order);
  int ck_min = std::max(0, gk == 0 ? 0 : (gk - 1) / order);
  int ck_max = std::min(cz - 1, gk / order);

  double weight = 0.0;
  for (int ci = ci_min; ci <= ci_max; ci++) {
    for (int cj = cj_min; cj <= cj_max; cj++) {
      for (int ck = ck_min; ck <= ck_max; ck++) {
        int c = ci * cy * cz + cj * cz + ck;
        if (!cell_empty.empty() && cell_empty[c]) { continue; }
        int    li     = gi - ci * order;
        int    lj     = gj - cj * order;
        int    lk     = gk - ck * order;
        double w_full = full_w[li] * full_w[lj] * full_w[lk];
        if (c >= static_cast<int>(cell_alphas.size()) || cell_alphas[c].empty()) {
          weight += w_full;
        } else {
          const std::vector<double>& alphas   = cell_alphas[c];
          const double*              ms       = m1d[li];
          const double*              mt       = m1d[lj];
          const double*              mu       = m1d[lk];
          double                     c_weight = 0.0;
          int                        q_idx    = 0;
          for (int ps = 0; ps < nq_axis; ps++) {
            double ws = ms[ps];
            for (int pt = 0; pt < nq_axis; pt++) {
              double wst = ws * mt[pt];
              for (int pu = 0; pu < nq_axis; pu++) { c_weight += wst * mu[pu] * alphas[q_idx++]; }
            }
          }
          weight += std::max(c_weight, kFcmMassMin * w_full);
        }
      }
    }
  }
  return weight;
}


// create flex BVH
void mjCFlex::CreateBVH() {
  int nbvh = 0;

  // allocate element bounding boxes
  elemaabb_.resize(6 * nelem);
  tree.AllocateBoundingVolumes(nelem);

  // construct element bounding boxes, add to hierarchy
  for (int e = 0; e < nelem; e++) {
    const int* edata = elem_.data() + e * (dim + 1);

    // skip inactive in 3D
    if (dim == 3 && elemlayer[e] >= activelayers) { continue; }

    // compute min and max along each global axis
    double xmin[3], xmax[3];
    mjuu_copyvec(xmin, vertxpos.data() + 3 * edata[0], 3);
    mjuu_copyvec(xmax, vertxpos.data() + 3 * edata[0], 3);
    for (int i = 1; i <= dim; i++) {
      for (int j = 0; j < 3; j++) {
        xmin[j] = std::min(xmin[j], vertxpos[3 * edata[i] + j]);
        xmax[j] = std::max(xmax[j], vertxpos[3 * edata[i] + j]);
      }
    }

    // compute aabb (center, size)
    elemaabb_[6 * e + 0] = 0.5 * (xmax[0] + xmin[0]);
    elemaabb_[6 * e + 1] = 0.5 * (xmax[1] + xmin[1]);
    elemaabb_[6 * e + 2] = 0.5 * (xmax[2] + xmin[2]);
    elemaabb_[6 * e + 3] = 0.5 * (xmax[0] - xmin[0]) + radius;
    elemaabb_[6 * e + 4] = 0.5 * (xmax[1] - xmin[1]) + radius;
    elemaabb_[6 * e + 5] = 0.5 * (xmax[2] - xmin[2]) + radius;

    // add bounding volume for this element
    // contype and conaffinity are set to nonzero to force bvh generation
    const double* aabb = elemaabb_.data() + 6 * e;
    tree.AddBoundingVolume(e, 1, 1, aabb, nullptr, aabb);
    nbvh++;
  }

  // create hierarchy
  tree.RemoveInactiveVolumes(nbvh);
  tree.CreateBVH(model, this);
}


// create shells
void mjCFlex::CreateShell(void) {
  std::vector<std::vector<int>> fragspec(
      nelem * (dim + 1));  // [sorted frag vertices, elem, original frag vertices]
  std::vector<std::vector<int>> connectspec;  // [elem1, elem2, common sorted frag vertices]

  std::vector<bool> border(nelem, false);                  // is element on the border
  std::vector<bool> borderfrag(nelem * (dim + 1), false);  // is fragment on the border

  // make fragspec
  for (int e = 0; e < nelem; e++) {
    int n = e * (dim + 1);

    // element vertices in original (unsorted) order
    std::vector<int> el;
    el.assign(elem_.begin() + n, elem_.begin() + n + dim + 1);

    // line: 2 vertex fragments
    if (dim == 1) {
      fragspec[n].push_back(el[0]);
      fragspec[n].push_back(e);
      fragspec[n].push_back(el[0]);

      fragspec[n + 1].push_back(el[1]);
      fragspec[n + 1].push_back(e);
      fragspec[n + 1].push_back(el[1]);
    }

    // triangle: 3 edge fragments
    else if (dim == 2) {
      fragspec[n].push_back(el[0]);
      fragspec[n].push_back(el[1]);
      fragspec[n].push_back(e);
      fragspec[n].push_back(el[0]);
      fragspec[n].push_back(el[1]);

      fragspec[n + 2].push_back(el[1]);
      fragspec[n + 2].push_back(el[2]);
      fragspec[n + 2].push_back(e);
      fragspec[n + 2].push_back(el[1]);
      fragspec[n + 2].push_back(el[2]);

      fragspec[n + 1].push_back(el[2]);
      fragspec[n + 1].push_back(el[0]);
      fragspec[n + 1].push_back(e);
      fragspec[n + 1].push_back(el[2]);
      fragspec[n + 1].push_back(el[0]);
    }

    // tetrahedron: 4 face fragments
    else {
      fragspec[n].push_back(el[0]);
      fragspec[n].push_back(el[1]);
      fragspec[n].push_back(el[2]);
      fragspec[n].push_back(e);
      fragspec[n].push_back(el[0]);
      fragspec[n].push_back(el[1]);
      fragspec[n].push_back(el[2]);

      fragspec[n + 2].push_back(el[0]);
      fragspec[n + 2].push_back(el[2]);
      fragspec[n + 2].push_back(el[3]);
      fragspec[n + 2].push_back(e);
      fragspec[n + 2].push_back(el[0]);
      fragspec[n + 2].push_back(el[2]);
      fragspec[n + 2].push_back(el[3]);

      fragspec[n + 1].push_back(el[0]);
      fragspec[n + 1].push_back(el[3]);
      fragspec[n + 1].push_back(el[1]);
      fragspec[n + 1].push_back(e);
      fragspec[n + 1].push_back(el[0]);
      fragspec[n + 1].push_back(el[3]);
      fragspec[n + 1].push_back(el[1]);

      fragspec[n + 3].push_back(el[1]);
      fragspec[n + 3].push_back(el[3]);
      fragspec[n + 3].push_back(el[2]);
      fragspec[n + 3].push_back(e);
      fragspec[n + 3].push_back(el[1]);
      fragspec[n + 3].push_back(el[3]);
      fragspec[n + 3].push_back(el[2]);
    }
  }

  // sort first segment of each fragspec
  if (dim > 1) {
    for (int n = 0; n < nelem * (dim + 1); n++) {
      std::sort(fragspec[n].begin(), fragspec[n].begin() + dim);
    }
  }

  // sort fragspec
  std::sort(fragspec.begin(), fragspec.end());

  // make border and connectspec, record borderfrag
  int cnt = 1;
  for (int n = 1; n < nelem * (dim + 1); n++) {
    // extract frag vertices, without elem
    std::vector<int> previous = {fragspec[n - 1].begin(), fragspec[n - 1].begin() + dim};
    std::vector<int> current  = {fragspec[n].begin(), fragspec[n].begin() + dim};

    // same sequential fragments
    if (previous == current) {
      // found pair of elements connected by common fragment
      std::vector<int> connect;
      connect.insert(connect.end(), fragspec[n - 1][dim]);
      connect.insert(connect.end(), fragspec[n][dim]);
      connect.insert(connect.end(), fragspec[n].begin(), fragspec[n].begin() + dim);
      connectspec.push_back(connect);

      // count same sequential fragments
      cnt++;
    }

    // different sequential fragments
    else {
      // found border fragment
      if (cnt == 1) {
        border[fragspec[n - 1][dim]] = true;
        borderfrag[n - 1]            = true;
      }

      // reset count
      cnt = 1;
    }
  }

  // last fragment is border
  if (cnt == 1) {
    int n                        = nelem * (dim + 1);
    border[fragspec[n - 1][dim]] = true;
    borderfrag[n - 1]            = true;
  }

  // create shell
  for (unsigned i = 0; i < borderfrag.size(); i++) {
    if (borderfrag[i]) {
      // add fragment vertices, in original order
      shell.insert(shell.end(), fragspec[i].begin() + dim + 1, fragspec[i].end());
    }
  }

  // compute elemlayer (distance from border) via value iteration in 3D
  if (dim < 3) {
    elemlayer = std::vector<int>(nelem, 0);
  } else {
    elemlayer = std::vector<int>(nelem, nelem + 1);  // init with greater than max value
    for (int e = 0; e < nelem; e++) {
      if (border[e]) {
        elemlayer[e] = 0;  // set border elements to 0
      }
    }

    bool change = true;
    while (change) {  // repeat while changes are happening
      change = false;

      // process edges of element connectivity graph
      for (const auto& connect : connectspec) {
        int e1 = connect[0];  // get element pair for this edge
        int e2 = connect[1];
        if (elemlayer[e1] > elemlayer[e2] + 1) {
          elemlayer[e1] = elemlayer[e2] + 1;  // better value found for e1: update
          change        = true;
        } else if (elemlayer[e2] > elemlayer[e1] + 1) {
          elemlayer[e2] = elemlayer[e1] + 1;  // better value found for e2: update
          change        = true;
        }
      }
    }
  }
}
