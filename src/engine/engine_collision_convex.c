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

#include "engine/engine_collision_convex.h"

#include <float.h>
#include <stddef.h>

#include <ccd/ccd.h>
#include <ccd/vec3.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_collision_gjk.h"
#include "engine/engine_macro.h"
#include "engine/engine_memory.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"

#define mjMINVAL2 (mjMINVAL * mjMINVAL)

// CCD internal buffer used for batched processing; if NULL, stack memory allocated on each
// mjc_penetration call
static mjTHREADLOCAL void* ccd_buffer = NULL;

// set CCD internal buffer
void mjc_setCCDBuffer(void* buffer) {
  ccd_buffer = buffer;
}

// ccd prism first dir
static void prism_firstdir(const void* o1, const void* o2, ccd_vec3_t *vec) {
  ccdVec3Set(vec, 0, 0, 1);
}

// wrapper around libccd; returns number of collisions found
static inline int _libccd_wrapper(const mjModel* m, mjCCDObj* obj1, mjCCDObj* obj2,
                                  mjPreContact* con, mjtNum margin) {
  ccd_t ccd;
  CCD_INIT(&ccd);
  ccd.mpr_tolerance = m->opt.ccd_tolerance;
  ccd.epa_tolerance = m->opt.ccd_tolerance;  // use MPR tolerance for EPA
  ccd.max_iterations = m->opt.ccd_iterations;
  ccd.support1 = mjccd_support;
  ccd.support2 = mjccd_support;
  ccd.center1 = mjccd_center;
  ccd.center2 = mjccd_center;
  if (obj1->geom_type == mjGEOM_HFIELD || obj2->geom_type == mjGEOM_HFIELD) {
    ccd.first_dir = prism_firstdir;
  } else {
    ccd.first_dir = ccdFirstDirDefault;
  }

  ccd_real_t ccd_depth;
  ccd_vec3_t ccd_dir, ccd_pos;
  int ret = ccdMPRPenetration(obj1, obj2, &ccd, &ccd_depth, &ccd_dir, &ccd_pos);
  if (ret == 0) {
    if (ccdVec3Eq(&ccd_dir, ccd_vec3_origin)) {
      return 0;
    }
    con[0].dist = margin - ccd_depth;
    mji_copy3(con[0].normal, ccd_dir.v);
    mji_copy3(con[0].pos, ccd_pos.v);
    mji_zero3(con[0].tangent);
    return 1;
  }
  return 0;
}


// find penetration info between two geoms; returns number of collisions found
static int mjc_penetration(const mjModel* m, mjData* d, mjCCDObj* obj1, mjCCDObj* obj2,
                           mjPreContact* con, int nconmax, mjtNum margin) {
  if (mjDISABLED(mjDSBL_NATIVECCD)) {
    return _libccd_wrapper(m, obj1, obj2, con, margin);
  }

  // nativeccd
  mjCCDConfig config;
  mjCCDStatus status;
  mjtNum dist;
  void* buffer = ccd_buffer;

  // set config
  config.max_iterations = m->opt.ccd_iterations;
  config.tolerance = m->opt.ccd_tolerance;
  config.max_contacts = nconmax;
  config.dist_cutoff = 0;  // no geom distances needed
  config.npolygonmax = m->npolygonmax;
  config.nmeshdegmax = m->nmeshdegmax;
  if (buffer) {
    config.buffer = buffer;
  } else {
    mj_markStack(d);
    int npolygonmax = mjDISABLED(mjDSBL_MULTICCD) ? 0 : m->npolygonmax;
    int nmeshdegmax = mjDISABLED(mjDSBL_MULTICCD) ? 0 : m->nmeshdegmax;
    config.buffer = mj_stackAllocByte(d, mjc_ccdSize(npolygonmax, nmeshdegmax,
                                                     config.max_iterations), sizeof(mjtNum));
  }

  int ncon = 0;
  if ((dist = mjc_ccd(&config, &status, obj1, obj2)) < 0) {
    int nwitness = status.nx;
    for (int i = 0; i < nwitness; i++) {
      con[i].dist = margin + status.dist[i];
      mji_add3(con[i].pos, status.x1 + 3*i, status.x2 + 3*i);
      mju_scl3(con[i].pos, con[i].pos, 0.5);
      mji_sub3(con[i].normal, status.x1 + 3*i, status.x2 + 3*i);
      mju_normalize3(con[i].normal);
      mji_zero3(con[i].tangent);
    }
    ncon = nwitness;
  }
  if (!buffer) {
    mj_freeStack(d);
  }
  return ncon;
}


// ccd center function
void mjccd_center(const void *obj, ccd_vec3_t *center) {
  mjc_center(center->v, (const mjCCDObj*) obj);
}

// center function for convex collision algorithms
void mjc_center(mjtNum res[3], const mjCCDObj *obj) {
  int g = obj->geom;
  int f = obj->flex;
  int e = obj->elem;
  int v = obj->vert;

  if (obj->geom_type == mjGEOM_HFIELD) {
    mju_zero3(res);
    for (int i=0; i < 6; i++) {
      mji_addTo3(res, obj->data.hfield.prism[i]);
     }
    mju_scl3(res, res, 1.0/6.0);
    return;
  }

  // return geom position
  if (g >= 0) {
    mji_copy3(res, obj->pos);
    return;
  }

  // return flex element position
  if (e >= 0) {
    mji_copy3(res, obj->data.flex.aabb);
    return;
  }

  // return flex vertex position
  if (f >= 0) {
    mji_copy3(res, obj->data.flex.vert_xpos + 3*v);
    return;
  }
}

// ------------------------------------ Support functions -----------------------------------------

// transform a vector from global to local frame
static inline void mulMatTVec3(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3]) {
  // perform matT * dir
  res[0] = mat[0]*dir[0] + mat[3]*dir[1] + mat[6]*dir[2];
  res[1] = mat[1]*dir[0] + mat[4]*dir[1] + mat[7]*dir[2];
  res[2] = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
}


// transform a vector from local to global frame
static inline void localToGlobal(mjtNum res[3], const mjtNum mat[9], const mjtNum dir[3],
                                 const mjtNum pos[3]) {
  // perform mat * dir + pos
  res[0] = mat[0]*dir[0] + mat[1]*dir[1] + mat[2]*dir[2];
  res[1] = mat[3]*dir[0] + mat[4]*dir[1] + mat[5]*dir[2];
  res[2] = mat[6]*dir[0] + mat[7]*dir[1] + mat[8]*dir[2];
  res[0] += pos[0];
  res[1] += pos[1];
  res[2] += pos[2];
}


// point support function
void mjc_pointSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  mji_copy3(res, obj->pos);
}


// sphere support function
static void mjc_sphereSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // sphere data
  const mjtNum* pos = obj->pos;
  mjtNum radius = obj->size[0];

  res[0] = radius*dir[0] + pos[0];
  res[1] = radius*dir[1] + pos[1];
  res[2] = radius*dir[2] + pos[2];
}


// line support function (capsule)
void mjc_lineSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // capsule data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  mjtNum length = obj->size[1];

  mjtNum dot = mat[2]*dir[0] + mat[5]*dir[1] + mat[8]*dir[2];
  mjtNum scl = dot >= 0 ? length : -length;

  // transform result to global frame
  res[0] = mat[2]*scl + pos[0];
  res[1] = mat[5]*scl + pos[1];
  res[2] = mat[8]*scl + pos[2];
}


// capsule support function
static void mjc_capsuleSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // capsule data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  mjtNum radius = obj->size[0];
  mjtNum length = obj->size[1];

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // start with sphere
  local_supp[0] = local_dir[0] * radius;
  local_supp[1] = local_dir[1] * radius;
  local_supp[2] = local_dir[2] * radius;

  // add cylinder contribution
  local_supp[2] += (local_dir[2] >= 0 ? length : -length);

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// ellipsoid support function
static void mjc_ellipsoidSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // ellipsoid data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point on unit sphere: scale dir by ellipsoid sizes
  local_supp[0] = local_dir[0] * size[0];
  local_supp[1] = local_dir[1] * size[1];
  local_supp[2] = local_dir[2] * size[2];

  mjtNum norm2 = local_supp[0]*local_supp[0] + local_supp[1]*local_supp[1] + local_supp[2]*local_supp[2];

  // too small to normalize
  if (norm2 < mjMINVAL2) {
    res[0] = mat[0]*size[0] + pos[0];
    res[1] = mat[3]*size[0] + pos[1];
    res[2] = mat[6]*size[0] + pos[2];
    return;
  }

  // normalize and transform to ellipsoid
  mjtNum norm_inv = 1/mju_sqrt(norm2);
  local_supp[0] *= norm_inv * size[0];
  local_supp[1] *= norm_inv * size[1];
  local_supp[2] *= norm_inv * size[2];

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// cylinder support function
static void mjc_cylinderSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // cylinder data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum n2 = local_dir[0]*local_dir[0] + local_dir[1]*local_dir[1];
  mjtNum scl = n2 >= mjMINVAL2 ? size[0] / mju_sqrt(n2) : 0;
  local_supp[0] = scl * local_dir[0];
  local_supp[1] = scl * local_dir[1];

  // set result in Z direction
  local_supp[2] = local_dir[2] >= 0 ? size[1] : -size[1];
  obj->vertindex = local_dir[2] >= 0 ? 0 : 1;

  // transform result to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// box support function
static void mjc_boxSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  // box data
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const mjtNum* size = obj->size;

  // rotate dir to geom local frame
  mjtNum local_dir[3], local_supp[3];
  mulMatTVec3(local_dir, mat, dir);

  // find support point in local frame
  local_supp[0] = local_dir[0] >= 0 ? size[0] : -size[0];
  local_supp[1] = local_dir[1] >= 0 ? size[1] : -size[1];
  local_supp[2] = local_dir[2] >= 0 ? size[2] : -size[2];

  // mark the index of the corner of the box for fast lookup
  obj->vertindex  = (local_supp[0] > 0) ? 1 : 0;
  obj->vertindex |= (local_supp[1] > 0) ? 2 : 0;
  obj->vertindex |= (local_supp[2] > 0) ? 4 : 0;

  // transform support point to global frame
  localToGlobal(res, mat, local_supp, pos);
}


// dot product between mjtNum and float
static inline mjtNum dot3f(const mjtNum a[3], const float b[3]) {
  return a[0]*(mjtNum)b[0] + a[1]*(mjtNum)b[1] + a[2]*(mjtNum)b[2];
}


// mesh support function via exhaustive search
static void mjc_meshSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  const mjtNum* mat = obj->mat;
  const mjtNum* pos = obj->pos;
  const float* verts = obj->data.mesh.vert;
  int nverts = obj->data.mesh.nvert;

  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  mjtNum max = -FLT_MAX;
  int imax = 0;

  // used cached results from previous search
  if (obj->vertindex >= 0) {
    imax = obj->vertindex;
    max = dot3f(local_dir, verts + 3*imax);
  }

  // search all vertices, find maximum dot product
  for (int i=0; i < nverts; i++) {
    mjtNum vdot = dot3f(local_dir, verts + 3*i);

    // update max
    if (vdot > max) {
      max = vdot;
      imax = i;
    }
  }

  // record vertex index of maximum
  obj->vertindex = imax;

  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}


// mesh support function via hill climbing
static void mjc_hillclimbSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  int numvert = obj->data.mesh.graph[0];
  const int* vert_edgeadr = obj->data.mesh.graph + 2;
  const int* vert_globalid = obj->data.mesh.graph  + 2 + numvert;
  const int* edge_localid = obj->data.mesh.graph + 2 + 2*numvert;
  const float* verts = obj->data.mesh.vert;
  const mjtNum* pos = obj->pos;
  const mjtNum* mat = obj->mat;

  // rotate dir to geom local frame
  mjtNum local_dir[3];
  mulMatTVec3(local_dir, mat, dir);

  int prev = -1;
  int imax;

  // map continuous direction to discrete 3x3x3 grid (-1, 0, 1) indices
  int cx = (local_dir[0] > 0.4) - (local_dir[0] < -0.4) + 1;
  int cy = (local_dir[1] > 0.4) - (local_dir[1] < -0.4) + 1;
  int cz = (local_dir[2] > 0.4) - (local_dir[2] < -0.4) + 1;
  int grid_idx = obj->data.mesh.extrema[cx*9 + cy*3 + cz];

  if (obj->meshindex >= 0) {
    // warm start: pick the better of cached vertex vs grid seed
    mjtNum cached_dot = dot3f(local_dir, verts + 3*vert_globalid[obj->meshindex]);
    mjtNum seed_dot = dot3f(local_dir, verts + 3*vert_globalid[grid_idx]);
    imax = (seed_dot > cached_dot) ? grid_idx : obj->meshindex;
  } else {
    // cold start: use grid seed
    imax = grid_idx;
  }

  mjtNum max = dot3f(local_dir, verts + 3*vert_globalid[imax]);

  // hillclimb until no change
  while (imax != prev) {
    prev = imax;
    int subidx;
    for (int i = vert_edgeadr[imax]; (subidx = edge_localid[i]) >= 0; i++) {
      mjtNum vdot = dot3f(local_dir, verts + 3*vert_globalid[subidx]);
      if (vdot > max) {
        max = vdot;
        imax = subidx;  // update maximum vertex index
      }
    }
  }

  // record vertex index of maximum (local id)
  obj->meshindex = imax;

  // get resulting support vertex
  obj->vertindex = imax = vert_globalid[imax];
  local_dir[0] = (mjtNum)verts[3*imax + 0];
  local_dir[1] = (mjtNum)verts[3*imax + 1];
  local_dir[2] = (mjtNum)verts[3*imax + 2];

  // transform result to global frame
  localToGlobal(res, mat, local_dir, pos);
}


// prism support function
static void mjc_prism_support(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  int istart, ibest;
  mjtNum best, tmp;
  mjtNum (*prism)[3] = obj->data.hfield.prism;

  // find best vertex in halfspace determined by dir.z
  istart = dir[2] < 0 ? 0 : 3;
  ibest = istart;
  best = mju_dot3(prism[istart], dir);
  for (int i=1; i < 3; i++) {
    if ((tmp = mju_dot3(prism[istart + i], dir)) > best) {
      ibest = istart + i;
      best = tmp;
    }
  }

  // copy best point
  mji_copy3(res, prism[ibest]);
}


// flex support function
static void mjc_flexSupport(mjtNum res[3], mjCCDObj* obj, const mjtNum dir[3]) {
  int f = obj->flex;
  int dim = obj->data.flex.dim[f];

  // flex element
  if (obj->elem >= 0) {
    const int* edata = obj->data.flex.elem;
    const mjtNum* vert = obj->data.flex.vert_xpos;

    // find element vertex with largest projection along dir
    mji_copy3(res, vert+3*edata[0]);
    mjtNum best = mju_dot3(res, dir);
    for (int i=1; i <= dim; i++) {
      mjtNum dot = mju_dot3(vert+3*edata[i], dir);

      // better vertex found: assign
      if (dot > best) {
        best = dot;
        mji_copy3(res, vert+3*edata[i]);
      }
    }

    // add radius and margin/2
    mji_addToScl3(res, dir, obj->data.flex.xradius[f] + 0.5*obj->margin);
    return;
  }

  // flex vertex
  else {
    const mjtNum* vert = obj->data.flex.vert_xpos + 3*obj->vert;
    mji_addScl3(res, vert, dir, obj->data.flex.xradius[f] + 0.5*obj->margin);
    return;
  }
}


// libccd support function
void mjccd_support(const void *_obj, const ccd_vec3_t *_dir, ccd_vec3_t *vec) {
  mjCCDObj *obj = (mjCCDObj *)_obj;
  mjtNum *res = vec->v;
  const mjtNum *dir = _dir->v;
  int g = obj->geom;

  if (g < 0) {
    int f = obj->flex;
    int dim = obj->data.flex.dim[f];

    // flex element
    if (obj->elem >= 0) {
      const int* edata = obj->data.flex.elem;
      const mjtNum* vert = obj->data.flex.vert_xpos;

      // find element vertex with largest projection along dir
      mji_copy3(res, vert+3*edata[0]);
      mjtNum best = mju_dot3(res, dir);
      for (int i=1; i <= dim; i++) {
        mjtNum dot = mju_dot3(vert+3*edata[i], dir);

        // better vertex found: assign
        if (dot > best) {
          best = dot;
          mji_copy3(res, vert+3*edata[i]);
        }
      }

      // add radius and margin/2
      mji_addToScl3(res, dir, obj->data.flex.xradius[f] + 0.5*obj->margin);
      return;
    }

    // flex vertex
    else {
      const mjtNum* vert = obj->data.flex.vert_xpos + 3*obj->vert;
      mji_addScl3(res, vert, dir, obj->data.flex.xradius[f] + 0.5*obj->margin);
      return;
    }
  }

  const float* vertdata;
  int ibest, numvert, change, locid;
  mjtNum tmp, vdot;

  const mjtNum* size = obj->size;          // geom sizes
  mjtNum local_dir[3];                    // direction in geom local frame

  // rotate dir to geom local frame
  mju_mulMatTVec3(local_dir, obj->mat, dir);

  // compute result according to geom type
  switch ((mjtGeom) obj->geom_type) {
  case mjGEOM_SPHERE:
    mji_scl3(res, local_dir, size[0]);
    break;

  case mjGEOM_CAPSULE:
    // start with sphere
    mji_scl3(res, local_dir, size[0]);

    // add cylinder contribution
    res[2] += mju_sign(local_dir[2]) * size[1];
    break;

  case mjGEOM_ELLIPSOID:
    // find support point on unit sphere: scale dir by ellipsoid sizes and renormalize
    for (int i=0; i < 3; i++) {
      res[i] = local_dir[i] * size[i];
    }
    mju_normalize3(res);

    // transform to ellipsoid
    for (int i=0; i < 3; i++) {
      res[i] *= size[i];
    }
    break;

  case mjGEOM_CYLINDER:
    // set result in XY plane: support on circle
    tmp = mju_sqrt(local_dir[0]*local_dir[0] + local_dir[1]*local_dir[1]);
    if (tmp > mjMINVAL) {
      res[0] = local_dir[0]/tmp*size[0];
      res[1] = local_dir[1]/tmp*size[0];
    } else {
      res[0] = res[1] = 0;
    }

    // set result in Z direction
    res[2] = mju_sign(local_dir[2]) * size[1];
    break;

  case mjGEOM_BOX:
    for (int i=0; i < 3; i++) {
      res[i] = mju_sign(local_dir[i]) * size[i];
    }
    break;

  case mjGEOM_MESH:
  case mjGEOM_SDF:
    // init search
    vertdata = obj->data.mesh.vert;
    tmp = -1E+10;
    ibest = -1;

    // no graph data: exhaustive search
    if (obj->data.mesh.graph == NULL) {
      // search all vertices, find best
      for (int i=0; i < obj->data.mesh.nvert; i++) {
        // vdot = dot(vertex, dir)
        vdot = local_dir[0] * (mjtNum)vertdata[3*i] +
               local_dir[1] * (mjtNum)vertdata[3*i+1] +
               local_dir[2] * (mjtNum)vertdata[3*i+2];

        // update best
        if (vdot > tmp) {
          tmp = vdot;
          ibest = i;
        }
      }

      // record best vertex index, in globalid format
      obj->meshindex = ibest;
    }

    // hill-climb using graph data
    else {
      // get info
      numvert = obj->data.mesh.graph[0];
      const int* vert_edgeadr = obj->data.mesh.graph + 2;
      const int* vert_globalid = obj->data.mesh.graph + 2 + numvert;
      const int* edge_localid = obj->data.mesh.graph + 2 + 2*numvert;

      // init with first vertex in convex hull or warmstart
      ibest = obj->meshindex < 0 ? 0 : obj->meshindex;
      tmp = local_dir[0] * (mjtNum)vertdata[3*vert_globalid[ibest]+0] +
            local_dir[1] * (mjtNum)vertdata[3*vert_globalid[ibest]+1] +
            local_dir[2] * (mjtNum)vertdata[3*vert_globalid[ibest]+2];

      // hill-climb until no change
      change = 1;
      while (change) {
        // look for improvement in ibest neighborhood
        change = 0;
        int i = vert_edgeadr[ibest];
        while ((locid=edge_localid[i]) >= 0) {
          // vdot = dot(vertex, local_dir)
          vdot = local_dir[0] * (mjtNum)vertdata[3*vert_globalid[locid]] +
                 local_dir[1] * (mjtNum)vertdata[3*vert_globalid[locid]+1] +
                 local_dir[2] * (mjtNum)vertdata[3*vert_globalid[locid]+2];

          // update best
          if (vdot > tmp) {
            tmp = vdot;
            ibest = locid;
            change = 1;
          }

          // advance to next edge
          i++;
        }
      }

      // record best vertex index, in locid format
      obj->meshindex = ibest;

      // map best index to globalid
      ibest = vert_globalid[ibest];
    }

    // sanity check, SHOULD NOT OCCUR
    if (ibest < 0) {
      mju_warning("mesh_support could not find support vertex");
      mju_zero3(res);
    }

    // copy best vertex
    else {
      for (int i=0; i < 3; i++) {
        res[i] = (mjtNum)vertdata[3*ibest + i];
      }
    }
    break;

  case mjGEOM_HFIELD:
    mjc_prism_support(res, obj, dir);
    mji_addToScl3(res, dir, 0.5*obj->margin);
    return;

  default:
    mjERROR("ccd support function is undefined for geom type %d", obj->geom_type);
  }

  // add local_dir*margin/2 to result
  for (int i=0; i < 3; i++) {
    res[i] += local_dir[i] * obj->margin/2;
  }

  // rotate result to global frame
  mju_mulMatVec3(res, obj->mat, res);

  // add geom position
  mji_addTo3(res, obj->pos);
}

// ------------------------------------------------------------------------------------------------

// initialize support function and mesh/hfield data for a CCD object
static void mjc_initCCDObjSupport(mjCCDObj* obj, const mjModel* m, int dataid) {
  switch ((mjtGeom) obj->geom_type) {
  case mjGEOM_ELLIPSOID:
    obj->support = mjc_ellipsoidSupport;
    break;
  case mjGEOM_MESH:
  case mjGEOM_SDF:
    if (dataid >= 0) {
      int graphadr = m->mesh_graphadr[dataid];
      int vertadr = m->mesh_vertadr[dataid];
      int polyadr = m->mesh_polyadr[dataid];
      if (graphadr < 0 || m->mesh_vertnum[dataid] < mjMESH_HILLCLIMB_MIN) {
        obj->data.mesh.graph = NULL;
        obj->data.mesh.extrema = NULL;
        obj->support = mjc_meshSupport;
      } else {
        obj->data.mesh.graph = m->mesh_graph + graphadr;
        obj->data.mesh.extrema = m->mesh_extrema + 27 * dataid;
        obj->support = mjc_hillclimbSupport;
      }
      obj->data.mesh.vert = m->mesh_vert + 3*vertadr;
      obj->data.mesh.nvert = m->mesh_vertnum[dataid];
      obj->data.mesh.mpolymapadr = m->mesh_polymapadr + vertadr;
      obj->data.mesh.mpolymapnum = m->mesh_polymapnum + vertadr;
      obj->data.mesh.polymap = m->mesh_polymap;
      obj->data.mesh.polynormal = m->mesh_polynormal + 3*polyadr;
      obj->data.mesh.polyvertadr = m->mesh_polyvertadr + polyadr;
      obj->data.mesh.polyvertnum = m->mesh_polyvertnum + polyadr;
      obj->data.mesh.polyvert = m->mesh_polyvert;
      obj->data.mesh.mesh_polynum = m->mesh_polynum[dataid];
    } else {
      obj->support = NULL;
    }
    break;
  case mjGEOM_SPHERE:
    obj->support = mjc_sphereSupport;
    break;
  case mjGEOM_CAPSULE:
    obj->support = mjc_capsuleSupport;
    break;
  case mjGEOM_CYLINDER:
    obj->support = mjc_cylinderSupport;
    break;
  case mjGEOM_BOX:
    obj->support = mjc_boxSupport;
    break;
  case mjGEOM_HFIELD:
    obj->center = mjc_center;
    obj->support = mjc_prism_support;
    if (dataid >= 0) {
      obj->data.hfield.hfield_nrow = m->hfield_nrow[dataid];
      obj->data.hfield.hfield_ncol = m->hfield_ncol[dataid];
      mju_copy(obj->size, m->hfield_size + 4*dataid, 4);
      obj->data.hfield.hfield_data = m->hfield_data + m->hfield_adr[dataid];
    }
    break;
  default:
    obj->support = NULL;
    break;
  }
}


// initialize a CCD object
void mjc_initCCDObj(mjCCDObj* obj, const mjModel* m, const mjData* d, int g, mjtNum margin) {
  obj->geom = g;
  obj->margin = margin;
  obj->center = mjc_center;
  obj->vertindex = -1;
  obj->meshindex = -1;
  obj->nsupp = 0;
  obj->flex = -1;
  obj->elem = -1;
  obj->vert = -1;
  mju_zero4(obj->rotate);
  obj->rotate[0] = 1;
  if (g >= 0) {
    mju_copy(obj->size, m->geom_size+3*g, 3);
    if (d) {
      mju_copy(obj->pos, d->geom_xpos+3*g, 3);
      mju_copy(obj->mat, d->geom_xmat+9*g, 9);
    }
    obj->geom_type = m->geom_type[g];
    mjc_initCCDObjSupport(obj, m, m->geom_dataid[g]);
  } else {
    obj->geom_type = mjGEOM_FLEX;
    obj->data.flex.dim = m->flex_dim;
    obj->support = mjc_flexSupport;
    obj->data.flex.aabb = d ? d->flexelem_aabb : NULL;
    obj->data.flex.elemadr = m->flex_elemadr;
    obj->data.flex.vert_xpos = d ? d->flexvert_xpos : NULL;
    obj->data.flex.vertadr = m->flex_vertadr;
    obj->data.flex.xradius = m->flex_radius;
    obj->data.flex.elemdataadr = m->flex_elemdataadr;
    obj->data.flex.elem = m->flex_elem;
  }
}


// initialize a CCD object from a site
void mjc_initCCDObjSite(mjCCDObj* obj, const mjModel* m, const mjData* d, int s, mjtNum margin) {
  obj->geom = -1;
  obj->margin = margin;
  obj->center = mjc_center;
  obj->vertindex = -1;
  obj->meshindex = -1;
  obj->nsupp = 0;
  obj->flex = -1;
  obj->elem = -1;
  obj->vert = -1;
  mju_zero4(obj->rotate);
  obj->rotate[0] = 1;
  if (s >= 0 && s < m->nsite) {
    int dataid = m->site_dataid[s];
    mji_copy3(obj->size, m->site_size+3*s);
    obj->size[3] = 0;
    obj->geom_type = m->site_type[s];
    if (obj->geom_type == mjGEOM_MESH && dataid >= 0) {
      mjtNum xquat[4];
      mji_mulQuat(xquat, d->site_xquat+4*s, m->mesh_quat+4*dataid);
      mji_quat2Mat(obj->mat, xquat);
      mji_mulMatVec3(obj->pos, d->site_xmat+9*s, m->mesh_pos+3*dataid);
      mji_addTo3(obj->pos, d->site_xpos+3*s);
    } else {
      mji_copy3(obj->pos, d->site_xpos+3*s);
      mji_copy9(obj->mat, d->site_xmat+9*s);
    }
    mjc_initCCDObjSupport(obj, m, dataid);
  }
}


// set flex data for CCD object
static void mjc_setCCDObjFlex(mjCCDObj* obj, int flex, int elem, int vert) {
  obj->flex = flex;
  obj->elem = elem;
  obj->vert = vert;
  if (flex >= 0) {
    obj->data.flex.vert_xpos += 3 * obj->data.flex.vertadr[flex];
    if (elem >= 0) {
      int dim = obj->data.flex.dim[flex];
      obj->data.flex.aabb += 6 * (obj->data.flex.elemadr[flex] + elem);
      obj->data.flex.elem += obj->data.flex.elemdataadr[flex] + elem * (dim + 1);
    }
  }
}


// compare new contact to previous contacts, return 1 if it is far from all of them
static int mjc_isDistinctContact(const mjPreContact* con, int ncon, mjtNum tolerance) {
  const mjtNum* last_pos = con[ncon - 1].pos;
  for (int i=0; i < ncon-1; i++) {
    if (mju_dist3(con[i].pos, last_pos) <= tolerance) {
      return 0;
    }
  }
  return 1;
}


// in-place rotation of spatial frame around given point of origin
static void mju_rotateFrame(const mjtNum origin[3], const mjtNum rot[9],
                            mjtNum xmat[9], mjtNum xpos[3]) {
  mjtNum mat[9], vec[3], rel[3];

  // rotate frame: xmat = rot*xmat
  mju_mulMatMat3(mat, rot, xmat);
  mju_copy(xmat, mat, 9);

  // vector to rotation origin: rel = origin - xpos
  mji_sub3(rel, origin, xpos);

  // displacement of origin due to rotation: vec = rot*rel - rel
  mju_mulMatVec3(vec, rot, rel);
  mju_subFrom3(vec, rel);

  // correct xpos by subtracting displacement: xpos = xpos - vec
  mji_subFrom3(xpos, vec);
}


// return number of contacts supported by a single pass of narrowphase
static int maxContacts(const mjModel* m, const mjCCDObj* obj1, const mjCCDObj* obj2) {
  // single pass not supported for libccd
  if (mjDISABLED(mjDSBL_NATIVECCD)) {
    return 1;
  }

  // always return up to 8 contacts for box-box collision in one pass
  int type1 = obj1->geom_type;
  int type2 = obj2->geom_type;
  if (type1 == mjGEOM_BOX && type2 == mjGEOM_BOX) {
    return 8;
  }

  // multicontact isn't available
  if (mjDISABLED(mjDSBL_MULTICCD)) {
    return 1;
  }

  // geoms with flat faces
  int hasface1 = (type1 == mjGEOM_BOX || type1 == mjGEOM_MESH || type1 == mjGEOM_CYLINDER);
  int hasface2 = (type2 == mjGEOM_BOX || type2 == mjGEOM_MESH || type2 == mjGEOM_CYLINDER);

  // geoms with only edges
  int hasedge1 = (type1 == mjGEOM_CAPSULE);
  int hasedge2 = (type2 == mjGEOM_CAPSULE);

  // colliding geoms with flat faces
  if (hasface1 && hasface2) {
    return 4;
  }

  // colliding geoms with edges
  if ((hasedge1 && hasface2) || (hasface1 && hasedge2) || (hasedge1 && hasedge2)) {
    return 2;
  }

  // not supported for other geom types
  return 1;
}


// multi-point convex-convex collision, using libccd
int mjc_Convex(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  // init ccd objects
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, margin);
  mjc_initCCDObj(&obj2, m, d, g2, margin);
  int max_contacts = maxContacts(m, &obj1, &obj2);

  // ellipsoid (including sphere) geoms don't require multiple contacts
  int isellipsoid1 = (obj1.geom_type == mjGEOM_ELLIPSOID || obj1.geom_type == mjGEOM_SPHERE);
  int isellipsoid2 = (obj2.geom_type == mjGEOM_ELLIPSOID || obj2.geom_type == mjGEOM_SPHERE);

  // find initial contact
  int ncon = mjc_penetration(m, d, &obj1, &obj2, con, max_contacts, margin);

  // fix normal for libccd
  if (mjDISABLED(mjDSBL_NATIVECCD) && ncon && g1 >= 0 && g2 >= 0) {
    mjc_fixNormal(m, d, con, g1, g2);
  }

  // no additional contacts needed
  if (max_contacts > 1 || isellipsoid1 || isellipsoid2) {
    return ncon;
  }

  // look for additional contacts
  if (ncon == 1 && !mjDISABLED(mjDSBL_MULTICCD) && !isellipsoid1 && !isellipsoid2) {
    // multiCCD parameters
    const mjtNum relative_tolerance = 1e-3;
    const mjtNum perturbation_angle = 1e-3;

    // complete frame of initial contact
    mjtNum frame[9];
    mji_copy3(frame, con[0].normal);
    mju_zero(frame+3, 6);
    mju_makeFrame(frame);

    // tolerance for determining if newly found contacts are distinct
    const mjtNum tolerance = relative_tolerance * mju_min(m->geom_rbound[g1], m->geom_rbound[g2]);

    // axes and rotation angles for perturbation test
    mjtNum* axes[2] = {frame+3, frame+6};
    mjtNum angles[2] = {-perturbation_angle, perturbation_angle};

    // rotate both geoms, search for new contacts
    for (int axis_id = 0; axis_id < 2; ++axis_id) {
      for (int angle_id = 0; angle_id < 2; ++angle_id) {
        mjtNum* axis = axes[axis_id];
        mjtNum angle = angles[angle_id];

        // make rotation matrix rot
        mjtNum quat[4], rot[9];
        mji_axisAngle2Quat(quat, axis, angle);
        mju_quat2Mat(rot, quat);

        // rotate g1 around initial contact point
        mju_rotateFrame(con[0].pos, rot, obj1.mat, obj1.pos);

        // inversely rotate g2 around initial contact point
        mjtNum invrot[9];
        mju_transpose(invrot, rot, 3, 3);
        mju_rotateFrame(con[0].pos, invrot, obj2.mat, obj2.pos);

        // search for new contact
        int n = mjc_penetration(m, d, &obj1, &obj2, con + ncon, 1, margin);
        if (mjDISABLED(mjDSBL_NATIVECCD) && n && g1 >= 0 && g2 >= 0) {
          mjc_fixNormal(m, d, con + ncon, g1, g2);
        }

        // check new contact
        if (n && mjc_isDistinctContact(con, ncon + 1, tolerance)) {
          // set penetration of new point to equal that of initial point
          con[ncon].dist = con[0].dist;
          // add new point
          ncon += 1;
        }

        // reset positions and orientations of g1 and g2
        mji_copy3(obj1.pos, d->geom_xpos+3*g1);
        mji_copy9(obj1.mat, d->geom_xmat+9*g1);
        mji_copy3(obj2.pos, d->geom_xpos+3*g2);
        mji_copy9(obj2.mat, d->geom_xmat+9*g2);
      }
    }
  }
  return ncon;
}


static inline void sub3f(mjtNum res[3], const float* a, const float* b) {
  res[0] = (mjtNum)a[0] - (mjtNum)b[0];
  res[1] = (mjtNum)a[1] - (mjtNum)b[1];
  res[2] = (mjtNum)a[2] - (mjtNum)b[2];
}

// compute area of a quadrilateral (helper for hull4f)
static inline mjtNum area4f(const float* hull, const int* idx, int a, int b, int c, int d) {
  mjtNum ca[3], db[3], cross[3];
  sub3f(ca, hull + 3*idx[a], hull + 3*idx[c]);
  sub3f(db, hull + 3*idx[b], hull + 3*idx[d]);
  mji_cross(cross, ca, db);
  return 0.5 * mju_norm3(cross);
}


// prune a convex hull to a maximum area quadrilateral using an anchor vertex
// returns number of vertices in the pruned hull
static inline int hull4f(int res[4], const float* hull, const int* idx, int nhull, int a) {
  int b = (a + 1) % nhull, c = (a + 2) % nhull, d = (a + 3) % nhull;
  res[0] = a, res[1] = b, res[2] = c, res[3] = d;
  if (nhull <= 4) {
    return nhull;
  }
  mjtNum m = area4f(hull, idx, a, b, c, d), m_next;
  while (1) {
    int d_next = (d + 1) % nhull;
    m_next = area4f(hull, idx, a, b, c, d_next);
    if (m_next <= m) {
      break;
    }
    d = d_next, m = m_next;
    res[0] = a, res[1] = b, res[2] = c, res[3] = d;
    while (1) {
      int c_next = (c + 1) % nhull;
      m_next = area4f(hull, idx, a, b, c_next, d);
      if (m_next <= m) {
        break;
      }
      c = c_next, m = m_next;
      res[0] = a, res[1] = b, res[2] = c, res[3] = d;
    }
    while (1) {
      int b_next = (b + 1) % nhull;
      m_next = area4f(hull, idx, a, b_next, c, d);
      if (m_next <= m) {
        break;
      }
      b = b_next, m = m_next;
      res[0] = a, res[1] = b, res[2] = c, res[3] = d;
    }
  }
  return 4;
}

// plane-convex collision
int mjc_PlaneConvex(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2, mjtNum margin) {
  const mjtNum* pos1 = d->geom_xpos + 3*g1;
  const mjtNum* mat1 = d->geom_xmat + 9*g1;
  const mjtNum* pos2 = d->geom_xpos + 3*g2;
  const mjtNum* mat2 = d->geom_xmat + 9*g2;

  mjtNum diff[3], normal[3] = {mat1[2], mat1[5], mat1[8]};
  mjCCDObj obj;
  mjc_initCCDObj(&obj, m, d, g2, 0);

  // get support point in negative plane normal direction
  mjtNum v[3], dir[3] = {-mat1[2], -mat1[5], -mat1[8]};
  obj.support(v, &obj, dir);

  // compute normal distance, return if too far
  mji_sub3(diff, v, pos1);
  mjtNum dist = mju_dot3(normal, diff);
  if (dist > margin) {
    return 0;
  }

  // always include the support point (deepest contact) as the first contact
  con[0].dist = dist;
  mji_copy3(con[0].pos, v);
  mji_addToScl3(con[0].pos, normal, -0.5*dist);
  mji_copy3(con[0].normal, normal);
  mji_zero3(con[0].tangent);

  // geom without mesh data (e.g. ellipsoid): return single contact only
  if (m->geom_dataid[g2] == -1 || !obj.data.mesh.mesh_polynum || obj.vertindex < 0) {
    return 1;
  }

  // mesh polygon data
  int vertadr = m->mesh_vertadr[m->geom_dataid[g2]];
  int polyadr = m->mesh_polyadr[m->geom_dataid[g2]];
  int polymapadr = m->mesh_polymapadr[vertadr + obj.vertindex];
  int polymapnum = m->mesh_polymapnum[vertadr + obj.vertindex];

  // find mesh face with normal most anti-aligned with plane normal
  int best_poly = -1;
  mjtNum best_dot = 1;
  mjtNum local_normal[3];
  mju_mulMatTVec3(local_normal, mat2, normal);

  for (int i = 0; i < polymapnum; i++) {
    int idx = m->mesh_polymap[polymapadr + i];
    const mjtNum* pn = m->mesh_polynormal + 3*(polyadr + idx);
    mjtNum ndot = pn[0]*local_normal[0] + pn[1]*local_normal[1] + pn[2]*local_normal[2];
    if (ndot < best_dot) {
      best_dot = ndot;
      best_poly = idx;
    }
  }

  // no suitable face found
  if (best_poly < 0) {
    return 1;
  }

  // get all vertices of the best polygon face
  int face_vertadr = m->mesh_polyvertadr[polyadr + best_poly];
  int face_vertnum = m->mesh_polyvertnum[polyadr + best_poly];
  const float* vertdata = m->mesh_vert + 3*vertadr;
  const int* face_polyvert = m->mesh_polyvert + face_vertadr;

  // find obj.vertindex within the face to use as the fixed anchor for pruning
  int a = 0;
  for (int i = 0; i < face_vertnum; i++) {
    if (face_polyvert[i] == obj.vertindex) {
      a = i;
      break;
    }
  }

  // prune to 4 supporting vertices if face has more than 4
  int ncon = 1, indices[4];
  int n = hull4f(indices, vertdata, face_polyvert, face_vertnum, a);

  // add supporting face vertices (skip the anchor vertex)
  for (int i = 1; i < n; i++) {
    mjtNum local_vert[3] = {
      (mjtNum)vertdata[3*face_polyvert[indices[i]] + 0],
      (mjtNum)vertdata[3*face_polyvert[indices[i]] + 1],
      (mjtNum)vertdata[3*face_polyvert[indices[i]] + 2]
    };
    mjtNum pnt[3];
    localToGlobal(pnt, mat2, local_vert, pos2);

    mji_sub3(diff, pnt, pos1);
    mjtNum vdist = mju_dot3(normal, diff);

    // skip vertex if above margin or above mesh center
    mji_sub3(diff, pnt, pos2);
    if (vdist > margin || mju_dot3(normal, diff) > 0) {
      continue;
    }
    con[ncon].dist = vdist;
    mji_copy3(con[ncon].pos, pnt);
    mji_addToScl3(con[ncon].pos, normal, -0.5*vdist);
    mji_copy3(con[ncon].normal, normal);
    mji_zero3(con[ncon].tangent);
    ncon++;
  }

  return ncon;
}


//----------------------------  heightfield collisions ---------------------------------------------


// add vertex to prism
static inline void addVert(mjCCDObj* obj, mjtNum x, mjtNum y, mjtNum z) {
  // move old data
  mji_copy3(obj->data.hfield.prism[0], obj->data.hfield.prism[1]);
  mji_copy3(obj->data.hfield.prism[1], obj->data.hfield.prism[2]);
  mji_copy3(obj->data.hfield.prism[3], obj->data.hfield.prism[4]);
  mji_copy3(obj->data.hfield.prism[4], obj->data.hfield.prism[5]);

  // add new vertex at last position
  obj->data.hfield.prism[2][0] = obj->data.hfield.prism[5][0] = x;
  obj->data.hfield.prism[2][1] = obj->data.hfield.prism[5][1] = y;
  obj->data.hfield.prism[5][2] = z;
}


// add vertex to prism
static inline void addPrismVert(mjCCDObj* obj, int r, int c, int i, mjtNum dx, mjtNum dy) {
  // move old data
  mji_copy3(obj->data.hfield.prism[0], obj->data.hfield.prism[1]);
  mji_copy3(obj->data.hfield.prism[1], obj->data.hfield.prism[2]);
  mji_copy3(obj->data.hfield.prism[3], obj->data.hfield.prism[4]);
  mji_copy3(obj->data.hfield.prism[4], obj->data.hfield.prism[5]);

  int dr = 1 - i;

  // add new vertex at last position
  obj->data.hfield.prism[2][0] = obj->data.hfield.prism[5][0] = dx*c - obj->size[0];
  obj->data.hfield.prism[2][1] = obj->data.hfield.prism[5][1] = dy*(r + dr) - obj->size[1];
  obj->data.hfield.prism[5][2] = obj->data.hfield.hfield_data[(r + dr)*obj->data.hfield.hfield_ncol + c]*obj->size[2];
}


// entry point for heightfield collisions
int mjc_ConvexHField(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                     mjtNum margin) {
  // hfield frame
  const mjtNum* pos1 = d->geom_xpos + 3*g1;
  const mjtNum* mat1 = d->geom_xmat + 9*g1;

  // geom2 frame
  mjtNum* pos2 = d->geom_xpos + 3*g2;
  mjtNum* mat2 = d->geom_xmat + 9*g2;

  // hfield data
  int hid = m->geom_dataid[g1];
  int nrow = m->hfield_nrow[hid];
  int ncol = m->hfield_ncol[hid];
  mjtNum size0 = m->hfield_size[4*hid + 0], size1 = m->hfield_size[4*hid + 1];
  mjtNum size2 = m->hfield_size[4*hid + 2], size3 = m->hfield_size[4*hid + 3];

  // try early return using box-sphere test

  // express geom2 pos in hfield frame
  mjtNum local_pos[3] = {pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2]};
  mju_mulMatTVec3(local_pos, mat1, local_pos);

  // sphere radius is geom2 rbound + margin
  mjtNum radius = m->geom_rbound[g2] + margin;

  // box-sphere test
  if ((size0 < local_pos[0] - radius) || (-size0 > local_pos[0] + radius) ||
      (size1 < local_pos[1] - radius) || (-size1 > local_pos[1] + radius) ||
      (size2 < local_pos[2] - radius) || (-size3 > local_pos[2] + radius)) {
    return 0;
  }

  // ccd set up
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, margin);
  mjc_initCCDObj(&obj2, m, d, g2, 0);


  // try early return using AABB box-box test

  // express geom2 mat in hfield frame
  mjtNum mat[9];
  mji_mulMatTMat3(mat, mat1, mat2);

  mji_copy9(obj2.mat, mat);
  mji_copy3(obj2.pos, local_pos);

  mjtNum local_dir[3] = {0, 0, 0}, res[3];

  // get support point in +X
  local_dir[0] = 1;
  obj2.support(res, &obj2, local_dir);
  mjtNum xmax = res[0];

  // get support point in -X
  local_dir[0] = -1;
  obj2.support(res, &obj2, local_dir);
  mjtNum xmin = res[0];

  local_dir[0] = 0;

  // get support point in +Y
  local_dir[1] = 1;
  obj2.support(res, &obj2, local_dir);
  mjtNum ymax = res[1];

  // get support point in -Y
  local_dir[1] = -1;
  obj2.support(res, &obj2, local_dir);
  mjtNum ymin = res[1];

  local_dir[1] = 0;

  // get support point in +Z
  local_dir[2] = 1;
  obj2.support(res, &obj2, local_dir);
  mjtNum zmax = res[2];

  // get support point in -Z
  local_dir[2] = -1;
  obj2.support(res, &obj2, local_dir);
  mjtNum zmin = res[2];

  xmin -= margin;
  xmax += margin;
  ymin -= margin;
  ymax += margin;
  zmin -= margin;
  zmax += margin;

  // AABB box-box test
  if ((xmin > size0) || (xmax < -size0) ||
      (ymin > size1) || (ymax < -size1) ||
      (zmin > size2) || (zmax < -size3)) {
    return 0;
  }

  // compute sub-grid bounds
  int cmin = (int) mju_floor((xmin + size0) / (2.0*size0) * (ncol-1));
  int cmax = (int) mju_ceil ((xmax + size0) / (2.0*size0) * (ncol-1));
  int rmin = (int) mju_floor((ymin + size1) / (2.0*size1) * (nrow-1));
  int rmax = (int) mju_ceil ((ymax + size1) / (2.0*size1) * (nrow-1));
  cmin = mjMAX(0, cmin);
  cmax = mjMIN(ncol-1, cmax);
  rmin = mjMAX(0, rmin);
  rmax = mjMIN(nrow-1, rmax);

  // geom margin needed for actual collision test
  obj2.margin = margin;

  // compute real-valued grid step
  mjtNum dx = (2.0*size0) / (ncol-1);
  mjtNum dy = (2.0*size1) / (nrow-1);

  // set zbottom value using base size
  mjtNum (*prism)[3] = obj1.data.hfield.prism;
  prism[0][2] = prism[1][2] = prism[2][2] = -size3;

  // process all prisms in subgrid
  int ncon = 0;
  for (int r=rmin; r < rmax; r++) {
    addPrismVert(&obj1, r, cmin, 0, dx, dy);
    addPrismVert(&obj1, r, cmin, 1, dx, dy);
    for (int c=cmin + 1; c <= cmax; c++) {
      for (int i=0; i < 2; i++) {
        // send vertex to prism constructor
        addPrismVert(&obj1, r, c, i, dx, dy);

        // prism height test
        if (prism[3][2] < zmin && prism[4][2] < zmin && prism[5][2] < zmin) {
          continue;
        }

        // run penetration function, save contact
        if (mjc_penetration(m, d, &obj1, &obj2, con + ncon, 1, margin)) {
          // transform to global coordinates
          mji_copy3(local_dir, con[ncon].normal);
          mji_copy3(local_pos, con[ncon].pos);
          mji_mulMatVec3(con[ncon].normal, mat1, local_dir);
          mji_mulMatVec3(con[ncon].pos, mat1, local_pos);
          mji_addTo3(con[ncon].pos, pos1);

          // force out of all loops if max contacts reached
          if (++ncon >= mjMAXCONPAIR) {
            r = rmax+1;
            c = cmax+1;
            i = 3;
            break;
          }
        }
      }
    }
  }

  if (mjDISABLED(mjDSBL_NATIVECCD)) {
    // fix contact normals
    for (int i=0; i < ncon; i++) {
      mjc_fixNormal(m, d, con + i, g1, g2);
    }
  }

  return ncon;
}


//--------------------------- fix contact frame normal ---------------------------------------------

// compute normal for point outside ellipsoid, using ray-projection SQP
static int mjc_ellipsoidInside(mjtNum nrm[3], const mjtNum pos[3], const mjtNum size[3]) {
  // algorithm constants
  const int maxiter = 30;
  const mjtNum tolerance = 1e-6;

  // precompute quantities
  mjtNum S2inv[3] = {1/(size[0]*size[0]), 1/(size[1]*size[1]), 1/(size[2]*size[2])};
  mjtNum C = pos[0]*pos[0]*S2inv[0] + pos[1]*pos[1]*S2inv[1] + pos[2]*pos[2]*S2inv[2] - 1;
  if (C > 0) {
    return 0;
  }

  // normalize initial normal (just in case)
  mju_normalize3(nrm);

  // main iteration
  int iter;
  for (iter=0; iter < maxiter; iter++) {
    // coefficients and determinant of quadratic
    mjtNum A = nrm[0]*nrm[0]*S2inv[0] + nrm[1]*nrm[1]*S2inv[1] + nrm[2]*nrm[2]*S2inv[2];
    mjtNum B = pos[0]*nrm[0]*S2inv[0] + pos[1]*nrm[1]*S2inv[1] + pos[2]*nrm[2]*S2inv[2];
    mjtNum det = B*B - A*C;
    if (det < mjMINVAL || A < mjMINVAL) {
      return (iter > 0);
    }

    // ray intersection with ellipse: pos + x*nrm, x>=0
    mjtNum x = (-B + mju_sqrt(det))/A;
    if (x < 0) {
      return (iter > 0);
    }

    // new point on ellipsoid
    mjtNum pnt[3];
    mji_addScl3(pnt, pos, nrm, x);

    // normal at new point
    mjtNum newnrm[3] = {pnt[0]*S2inv[0], pnt[1]*S2inv[1], pnt[2]*S2inv[2]};
    mju_normalize3(newnrm);

    // save change and assign
    mjtNum change = mju_dist3(nrm, newnrm);
    mji_copy3(nrm, newnrm);

    // terminate if converged
    if (change < tolerance) {
      break;
    }
  }

  return 1;
}


// compute normal for point inside ellipsoid, using diagonal QCQP
static int mjc_ellipsoidOutside(mjtNum nrm[3], const mjtNum pos[3], const mjtNum size[3]) {
  // algorithm constants
  const int maxiter = 30;
  const mjtNum tolerance = 1e-6;

  // precompute quantities
  mjtNum S2[3] = {size[0]*size[0], size[1]*size[1], size[2]*size[2]};
  mjtNum PS2[3] = {pos[0]*pos[0]*S2[0], pos[1]*pos[1]*S2[1], pos[2]*pos[2]*S2[2]};

  // main iteration
  mjtNum la = 0;
  int iter;
  for (iter=0; iter < maxiter; iter++) {
    // precompute 1/(s^2+la)
    mjtNum R[3] = {1/(S2[0]+la), 1/(S2[1]+la), 1/(S2[2]+la)};

    // value
    mjtNum val = PS2[0]*R[0]*R[0] + PS2[1]*R[1]*R[1] + PS2[2]*R[2]*R[2] - 1;
    if (val < tolerance) {
      break;
    }

    // derivative
    mjtNum deriv = -2*(PS2[0]*R[0]*R[0]*R[0] + PS2[1]*R[1]*R[1]*R[1] + PS2[2]*R[2]*R[2]*R[2]);
    if (deriv > -mjMINVAL) {
      break;
    }

    // delta
    mjtNum delta = -val/deriv;
    if (delta < tolerance) {
      break;
    }

    // update
    la += delta;
  }

  // compute normal given lambda
  nrm[0] = pos[0]/(S2[0]+la);
  nrm[1] = pos[1]/(S2[1]+la);
  nrm[2] = pos[2]/(S2[2]+la);
  mju_normalize3(nrm);

  return 1;
}


// fix normals if required
void mjc_fixNormal(const mjModel* m, const mjData* d, mjPreContact* con, int g1, int g2) {
  mjtNum dst1, dst2;

  // get geom ids and types
  int gid[2] = {g1, g2};
  mjtGeom type[2];
  for (int i=0; i < 2; i++) {
    if (gid[i] < 0) {
      type[i] = mjGEOM_NONE;
    } else {
      type[i] = m->geom_type[gid[i]];
    }

    // set to mjGEOM_NONE if type cannot be processed
    if (type[i] != mjGEOM_SPHERE && type[i] != mjGEOM_CAPSULE &&
        type[i] != mjGEOM_ELLIPSOID && type[i] != mjGEOM_CYLINDER) {
      type[i] = mjGEOM_NONE;
    }
  }

  // neither type can be processed: nothing to do
  if (type[0] == mjGEOM_NONE && type[1] == mjGEOM_NONE) {
    return;
  }

  // init normals
  mjtNum normal[2][3] = {
    {con->normal[0], con->normal[1], con->normal[2]},
    {-con->normal[0], -con->normal[1], -con->normal[2]}
  };

  // process geoms in type range
  int processed[2] = {0, 0};
  for (int i=0; i < 2; i++) {
    if (type[i] != mjGEOM_NONE) {
      // get geom mat and size
      mjtNum* mat = d->geom_xmat + 9*gid[i];
      mjtNum* size = m->geom_size + 3*gid[i];

      // map contact point and normal to local frame
      mjtNum dif[3], pos1[3], nrm[3];
      mju_sub3(dif, con->pos, d->geom_xpos+3*gid[i]);
      mju_mulMatTVec3(pos1, mat, dif);
      mju_mulMatTVec3(nrm, mat, normal[i]);

      // process according to type
      switch (type[i]) {
      case mjGEOM_SPHERE:
        mji_copy3(nrm, pos1);
        processed[i] = 1;
        break;

      case mjGEOM_CAPSULE:
        // Z: bottom cap
        if (pos1[2] < -size[1]) {
          nrm[2] = pos1[2]+size[1];
        }

        // Z: top cap
        else if (pos1[2] > size[1]) {
          nrm[2] = pos1[2]-size[1];
        }

        // Z: cylinder
        else {
          nrm[2] = 0;
        }

        // copy XY
        nrm[0] = pos1[0];
        nrm[1] = pos1[1];
        processed[i] = 1;
        break;

      case mjGEOM_ELLIPSOID:
        // guard against invalid ellipsoid size (just in case)
        if (size[0] < mjMINVAL || size[1] < mjMINVAL || size[2] < mjMINVAL) {
          break;
        }

        // compute elliptic distance^2
        dst1 = pos1[0]*pos1[0]/(size[0]*size[0]) +
               pos1[1]*pos1[1]/(size[1]*size[1]) +
               pos1[2]*pos1[2]/(size[2]*size[2]);

        // dispatch to inside or outside solver
        if (dst1 <= 1) {
          processed[i] = mjc_ellipsoidInside(nrm, pos1, size);
        } else {
          processed[i] = mjc_ellipsoidOutside(nrm, pos1, size);
        }
        break;

      case mjGEOM_CYLINDER:
        // skip if within 5% length of flat wall
        if (mju_abs(pos1[2]) > 0.95*size[1]) {
          break;
        }

        // compute distances to flat and round wall
        dst1 = mju_abs(size[1]-mju_abs(pos1[2]));
        dst2 = mju_abs(size[0]-mju_norm(pos1, 2));

        // require 4x closer to round than flat wall
        if (dst1 < 0.25*dst2) {
          break;
        }

        // set normal for round wall
        nrm[0] = pos1[0];
        nrm[1] = pos1[1];
        nrm[2] = 0;
        processed[i] = 1;
        break;

      default:
        // do nothing: only sphere, capsule, ellipsoid and cylinder are processed
        break;
      }

      // normalize and map normal to global frame
      if (processed[i]) {
        mju_normalize3(nrm);
        mji_mulMatVec3(normal[i], mat, nrm);
      }
    }
  }

  // both processed: average
  if (processed[0] && processed[1]) {
    mji_sub3(con->normal, normal[0], normal[1]);
    mju_normalize3(con->normal);
  }

  // first processed: copy
  else if (processed[0]) {
    mji_copy3(con->normal, normal[0]);
  }

  // second processed: copy reverse
  else if (processed[1]) {
    mji_scl3(con->normal, normal[1], -1);
  }
}


//----------------------------  flex collisions ---------------------------------------------

// geom-elem or elem-elem or vert-elem convex collision using ccd
int mjc_ConvexElem(const mjModel* m, mjData* d, mjPreContact* con, int g1, int f1, int e1, int v1,
                   int f2, int e2, mjtNum margin) {
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m, d, g1, margin);
  mjc_initCCDObj(&obj2, m, d, -1, margin);
  mjc_setCCDObjFlex(&obj1, f1, e1, v1);
  mjc_setCCDObjFlex(&obj2, f2, e2, -1);

  // find contacts
  int ncon = mjc_penetration(m, d, &obj1, &obj2, con, 1, margin);

  // fix normals for 2D flex
  if (ncon && !mjDISABLED(mjDSBL_NATIVECCD)) {
    // check if either object is a 2D flex
    int isflex2d = 0;
    if (f1 >= 0 && m->flex_dim[f1] == 2) isflex2d = 1;
    if (f2 >= 0 && m->flex_dim[f2] == 2) isflex2d = 1;

    if (isflex2d) {
      for (int i = 0; i < ncon; i++) {
        mjc_fixNormal(m, d, con + i, g1, -1);
      }
    }
  }
  return ncon;
}


// test a height field and a flex element for collision
int mjc_HFieldElem(const mjModel* m, mjData* d, mjPreContact* con, int g, int f, int e,
                   mjtNum margin) {
  mjtNum vec[3], dx, dy;
  mjtNum xmin, xmax, ymin, ymax, zmin, zmax;
  int dr[2], cnt, rmin, rmax, cmin, cmax;
  mjCCDObj obj1;
  mjc_initCCDObj(&obj1, m, d, g, margin);

  // get hfield info
  int hid = m->geom_dataid[g];
  int nrow = m->hfield_nrow[hid];
  int ncol = m->hfield_ncol[hid];
  const mjtNum* hpos = d->geom_xpos + 3*g;
  const mjtNum* hmat = d->geom_xmat + 9*g;
  const mjtNum* hsize = m->hfield_size + 4*hid;
  const float* hdata = m->hfield_data + m->hfield_adr[hid];

  // get elem info
  int dim = m->flex_dim[f];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
  static const int local_edata[4] = {0, 1, 2, 3};
  mjtNum evert[4][3], ecenter[3];
  for (int i=0; i <= dim; i++) {
    const mjtNum* v = d->flexvert_xpos + 3*(m->flex_vertadr[f] + edata[i]);
    mji_sub3(vec, v, hpos);
    mji_mulMatTVec3(evert[i], hmat, vec);
  }
  mji_sub3(vec, d->flexelem_aabb + 6*(m->flex_elemadr[f] + e), hpos);
  mji_mulMatTVec3(ecenter, hmat, vec);

  // ccd-related
  mjCCDObj obj2;
  mjc_initCCDObj(&obj2, m, d, -1, margin);
  mjc_setCCDObjFlex(&obj2, f, e, -1);
  obj2.data.flex.elem = local_edata;
  obj2.data.flex.vert_xpos = (const mjtNum*)evert;
  obj2.data.flex.aabb = ecenter;
  //------------------------------------- AABB computation, box-box test

  // compute elem bounding box (in hfield frame)
  xmin = xmax = evert[0][0];
  ymin = ymax = evert[0][1];
  zmin = zmax = evert[0][2];
  for (int i=1; i <= dim; i++) {
    xmin = mju_min(xmin, evert[i][0]);
    xmax = mju_max(xmax, evert[i][0]);
    ymin = mju_min(ymin, evert[i][1]);
    ymax = mju_max(ymax, evert[i][1]);
    zmin = mju_min(zmin, evert[i][2]);
    zmax = mju_max(zmax, evert[i][2]);
  }

  mjtNum expand = m->flex_radius[f] + margin;
  xmin -= expand;
  xmax += expand;
  ymin -= expand;
  ymax += expand;
  zmin -= expand;
  zmax += expand;

  // box-box test
  if ((xmin > hsize[0]) || (xmax < -hsize[0]) ||
      (ymin > hsize[1]) || (ymax < -hsize[1]) ||
      (zmin > hsize[2]) || (zmax < -hsize[3])) {
    return 0;
  }

  // compute sub-grid bounds
  cmin = (int) mju_floor((xmin + hsize[0]) / (2*hsize[0]) * (ncol-1));
  cmax = (int) mju_ceil ((xmax + hsize[0]) / (2*hsize[0]) * (ncol-1));
  rmin = (int) mju_floor((ymin + hsize[1]) / (2*hsize[1]) * (nrow-1));
  rmax = (int) mju_ceil ((ymax + hsize[1]) / (2*hsize[1]) * (nrow-1));
  cmin = mjMAX(0, cmin);
  cmax = mjMIN(ncol-1, cmax);
  rmin = mjMAX(0, rmin);
  rmax = mjMIN(nrow-1, rmax);

  //------------------------------------- collision testing

  // compute real-valued grid step, and triangulation direction
  dx = (2.0*hsize[0]) / (ncol-1);
  dy = (2.0*hsize[1]) / (nrow-1);
  dr[0] = 1;
  dr[1] = 0;

  // set zbottom value using base size
  mjtNum (*prism)[3] = obj1.data.hfield.prism;
  prism[0][2] = prism[1][2] = prism[2][2] = -hsize[3];

  // process all prisms in sub-grid
  cnt = 0;
  for (int r=rmin; r < rmax; r++) {
    int nvert = 0;
    for (int c=cmin; c <= cmax; c++) {
      for (int k=0; k < 2; k++) {
        // send vertex to prism constructor
        addVert(&obj1, dx*c-hsize[0], dy*(r+dr[k])-hsize[1],
                hdata[(r+dr[k])*ncol+c]*hsize[2]);

        // check for enough vertices
        if (++nvert > 2) {
          // prism height test
          if (prism[3][2] < zmin && prism[4][2] < zmin && prism[5][2] < zmin) {
            continue;
          }

          // run ccd, save contact
          if (mjc_penetration(m, d, &obj1, &obj2, con + cnt, 1, margin)) {
            // transform to global coordinates
            mji_zero3(con[cnt].tangent);
            mju_mulMatVec3(con[cnt].normal, hmat, con[cnt].normal);
            mju_mulMatVec3(con[cnt].pos, hmat, con[cnt].pos);
            mji_addTo3(con[cnt].pos, hpos);

            // count, stop if max number reached
            cnt++;
            if (cnt >= mjMAXCONPAIR) {
              r = rmax+1;
              c = cmax+1;
              k = 3;
              break;
            }
          }
        }
      }
    }
  }

  return cnt;
}


// returns approximation (lower bound) of directed Hausdorff distance between two
// compact convex geoms; if distance is positive then obj1 is guaranteed to not be enclosed in obj2
mjtNum mjc_hausdorff(mjCCDObj* obj1, mjCCDObj* obj2, int nitermax,
                     mjtNum stepsize, mjtNum tolerance) {
  if (!obj1 || !obj2 || !obj1->support || !obj2->support) {
    mjERROR("invalid ccd object or null support function");
  }
  mjtGeom type1 = (mjtGeom)obj1->geom_type;
  mjtGeom type2 = (mjtGeom)obj2->geom_type;
  if (type1 < mjGEOM_SPHERE || type1 > mjGEOM_MESH ||
      type2 < mjGEOM_SPHERE || type2 > mjGEOM_MESH) {
    mjERROR("only compact convex shapes are supported, got types %d and %d", type1, type2);
  }

  mjtNum x_k[6][3], best_x[6][3], best_grad[6][3], best_val[6], step[6];
  int active[6] = {1, 1, 1, 1, 1, 1};

  // seed with obj2's local axes
  for (int s = 0; s < 6; s++) {
    int axis = s / 2;
    mjtNum sgn = (s % 2) ? -1.0 : 1.0;
    x_k[s][0] = sgn * obj2->mat[0 + axis];
    x_k[s][1] = sgn * obj2->mat[3 + axis];
    x_k[s][2] = sgn * obj2->mat[6 + axis];
    best_val[s] = -mjMAXVAL;
    step[s] = stepsize;
  }

  for (int k = 0; k < nitermax; k++) {
    int any_active = 0;
    for (int s = 0; s < 6; s++) {
      if (!active[s]) {
        continue;
      }

      mjtNum v1[3], v2[3], vert[3];
      obj1->support(v1, obj1, x_k[s]);
      obj2->support(v2, obj2, x_k[s]);
      mji_sub3(vert, v1, v2);
      mjtNum val = mju_dot3(vert, x_k[s]);

      // compute tangent gradient on S^2: grad = vert - (val * x_k[s])
      mjtNum grad[3], scaled_x[3];
      mju_scl3(scaled_x, x_k[s], val);
      mju_sub3(grad, vert, scaled_x);
      mjtNum grad_norm = mju_norm3(grad);

      if (val > best_val[s]) {
        best_val[s] = val;
        mju_copy3(best_x[s], x_k[s]);

        // scale-invariant angular convergence check
        if (grad_norm <= mjMINVAL || grad_norm <= tolerance * mju_abs(val)) {
          active[s] = 0;
          continue;
        }
        mju_scl3(best_grad[s], grad, 1.0 / grad_norm);
      } else {
        // overshot a normal-cone ridge/peak: halve step and average subgradients
        step[s] *= 0.5;
        if (step[s] < tolerance) {
          active[s] = 0;
          continue;
        }
        if (grad_norm > mjMINVAL) {
          mju_addToScl3(best_grad[s], grad, 1.0 / grad_norm);
        }
        mjtNum proj = mju_dot3(best_grad[s], best_x[s]);
        mju_addToScl3(best_grad[s], best_x[s], -proj);
        mjtNum avg_norm = mju_norm3(best_grad[s]);
        if (avg_norm <= tolerance) {
          active[s] = 0;
          continue;
        }
        mju_scl3(best_grad[s], best_grad[s], 1.0 / avg_norm);
      }

      // step from best_x[s] in the unit tangent direction and normalize
      mju_copy3(x_k[s], best_x[s]);
      mju_addToScl3(x_k[s], best_grad[s], step[s]);
      mju_normalize3(x_k[s]);
      any_active = 1;
    }
    if (!any_active) {
      break;
    }
  }

  mjtNum max_val = best_val[0];
  for (int s = 1; s < 6; s++) {
    max_val = mju_max(max_val, best_val[s]);
  }
  return max_val;
}
