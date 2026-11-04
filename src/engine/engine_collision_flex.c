// Copyright 2026 DeepMind Technologies Limited
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

#include "engine/engine_collision_flex.h"

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_core_util.h"
#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_spatial.h"

// sphere : triangle with radius
static int sphereTriangle(mjPreContact* con, mjtNum margin, const mjtNum* s,
                          mjtNum rs, const mjtNum* t1, const mjtNum* t2,
                          const mjtNum* t3, mjtNum rt) {
  mjtNum rbound = margin + rs + rt;
  mjtNum w[3], X[3], nrm[3];
  mjc_triWeights(w, s, t1, t2, t3);
  for (int k = 0; k < 3; k++) {
    X[k] = w[0] * t1[k] + w[1] * t2[k] + w[2] * t3[k];
  }

  // compute contact normal and distance
  mji_sub3(nrm, X, s);
  mjtNum dst = mju_normalize3(nrm);
  if (dst > rbound) {
    return 0;
  }

  // construct contact
  con[0].dist = dst - rs - rt;
  mji_addScl3(con[0].pos, s, nrm, rs + con[0].dist / 2);
  mji_copy3(con[0].normal, nrm);
  mju_zero3(con[0].tangent);

  return 1;
}


// box : triangle with radius
static int boxTriangle(mjPreContact* con, mjtNum margin, const mjtNum* pos,
                       const mjtNum* mat, const mjtNum* size, const mjtNum* t1,
                       const mjtNum* t2, const mjtNum* t3, mjtNum rt) {
  int cnt = 0;
  const mjtNum* vert[3] = {t1, t2, t3};

  for (int i = 0; i < 3; i++) {
    // map vertex to box local frame
    mjtNum diff[3], local[3];
    mju_sub3(diff, vert[i], pos);
    mju_mulMatTVec3(local, mat, diff);

    // find max penetration / closest face
    int maxaxis = 0;
    mjtNum maxval = mju_abs(local[0]) - size[0];
    for (int j = 1; j < 3; j++) {
      mjtNum val = mju_abs(local[j]) - size[j];
      if (val > maxval) {
        maxval = val;
        maxaxis = j;
      }
    }

    // contact distance: dist = maxval - rt
    // strictly, we only care if dist < margin
    if (maxval - rt > margin) {
      continue;
    }

    // check if within other dimensions (with margin/radius)
    int inside = 1;
    for (int j = 0; j < 3; j++) {
      if (mju_abs(local[j]) > size[j] + margin + rt) {
        inside = 0;
        break;
      }
    }
    if (!inside) {
      continue;
    }

    // create contact
    if (cnt < mjMAXCONPAIR) {
      // normal in local frame
      mjtNum nrm_local[3] = {0, 0, 0};
      nrm_local[maxaxis] = (local[maxaxis] > 0 ? 1 : -1);

      // normal in global frame (from Box to Triangle)
      mju_mulMatVec3(con[cnt].normal, mat, nrm_local);

      // distance
      con[cnt].dist = maxval - rt;

      // position: v - nrm * (rt + dist/2)
      mjtNum offset = rt + con[cnt].dist * 0.5;
      mji_addScl3(con[cnt].pos, vert[i], con[cnt].normal, -offset);

      // frame details
      mju_zero3(con[cnt].tangent);

      cnt++;
    }
  }

  // check box corners against triangle
  for (int i = 0; i < 8; i++) {
    if (cnt >= mjMAXCONPAIR) {
      break;
    }

    // get corner in local coordinates
    mjtNum vec[3];
    vec[0] = (i & 1 ? size[0] : -size[0]);
    vec[1] = (i & 2 ? size[1] : -size[1]);
    vec[2] = (i & 4 ? size[2] : -size[2]);

    // get corner in global coordinates relative to box center
    mjtNum corner[3];
    mju_mulMatVec3(corner, mat, vec);
    mju_addTo3(corner, pos);

    // check collision with triangle (radius 0 for corner)
    if (sphereTriangle(con + cnt, margin, corner, 0, t1, t2, t3, rt)) {
      // mjraw_SphereTriangle normal points from Sphere (Corner) to Triangle.
      cnt++;
    }
  }

  return cnt;
}


// capsule : triangle with radius
static int capsuleTriangle(mjPreContact* con, mjtNum margin, const mjtNum* pos,
                           const mjtNum* mat, const mjtNum* size,
                           const mjtNum* t1, const mjtNum* t2, const mjtNum* t3,
                           mjtNum rt) {
  int cnt = 0;
  mjtNum radius = size[0];
  mjtNum len = size[1];
  mjtNum axis[3] = {mat[2], mat[5], mat[8]};
  mjtNum p1[3], p2[3];

  // capsule endpoints
  mju_addScl3(p1, pos, axis, -len);
  mju_addScl3(p2, pos, axis, len);

  // Check endpoints against triangle
  cnt += sphereTriangle(con + cnt, margin, p1, radius, t1, t2, t3, rt);
  if (cnt >= mjMAXCONPAIR) return cnt;
  cnt += sphereTriangle(con + cnt, margin, p2, radius, t1, t2, t3, rt);
  if (cnt >= mjMAXCONPAIR) return cnt;

  // Check triangle vertices against capsule axis (Point-Segment)
  const mjtNum* vert[3] = {t1, t2, t3};
  for (int i = 0; i < 3; i++) {
    // point-segment distance
    mjtNum vec[3], ab[3];
    mju_sub3(vec, vert[i], p1);
    mju_sub3(ab, p2, p1);
    mjtNum t = mju_dot3(vec, ab) / (4 * len * len);  // ab length is 2*len

    // clamp t to [0, 1] segment (only process interior)
    if (t <= mjMINVAL || t >= 1 - mjMINVAL) {
      continue;
    }

    // closest point on segment
    mjtNum closest[3];
    mji_addScl3(closest, p1, ab, t);

    // distance vector
    mju_sub3(vec, vert[i], closest);
    mjtNum dist = mju_normalize3(vec);

    if (dist > radius + rt + margin) {
      continue;
    }

    // con->dist
    con[cnt].dist = dist - radius - rt;

    // Frame: normal from Capsule to Triangle. 'vec' points Closest->Vert.
    mji_copy3(con[cnt].normal, vec);
    mju_zero3(con[cnt].tangent);

    // Position: midway between surfaces
    mji_add3(con[cnt].pos, closest, vert[i]);
    mji_addToScl3(con[cnt].pos, vec, radius - rt);
    mju_scl3(con[cnt].pos, con[cnt].pos, 0.5);

    cnt++;
    if (cnt >= mjMAXCONPAIR) return cnt;
  }

  return cnt;
}

// test a plane geom and a flex for collision, return number of contacts
int mjc_PlaneFlex(const mjModel* m, mjData* d, mjPreContact* con, int* vert, int g, int f,
                  mjtNum margin) {
  mjtNum radius = m->flex_radius[f];
  mjtNum* pos = d->geom_xpos + 3*g;
  mjtNum* mat = d->geom_xmat + 9*g;
  mjtNum normal[3] = {mat[2], mat[5], mat[8]};

  // prepare contact parameters (same for all vertices)
  int flex_vertnum = m->flex_vertnum[f];
  int elemnum = m->flex_elemnum[f];
  mjtNum box[2];
  int is_box = (elemnum > 0 && mj_rodBoxSize(m, f, box));
  mjtNum bound = is_box ? mju_norm(box, 2) : radius;

  // collide all flex vertices with plane
  int ncon = 0;
  for (int i = 0; i < flex_vertnum; i++) {
    mjtNum* v = d->flexvert_xpos + 3*(m->flex_vertadr[f] + i);

    // distance from plane to vertex
    mjtNum dif[3] = {v[0] - pos[0], v[1] - pos[1], v[2] - pos[2]};
    mjtNum dist = mju_dot3(dif, normal);

    // quick reject by bounding radius
    if (dist > margin + bound) {
      continue;
    }

    if (is_box) {
      int e = mj_rodVertElem(m, f, i);
      mjtNum bpos[3], em[9], bsize[3];
      if (e >= 0 && mj_rodBoxElem(m, d, f, e, bpos, em, bsize)) {
        mjtNum m1[3] = {em[0], em[3], em[6]};
        mjtNum m2[3] = {em[1], em[4], em[7]};
        mjtNum n1 = mju_dot3(normal, m1);
        mjtNum n2 = mju_dot3(normal, m2);
        mjtNum s1 = (n1 >= 0) ? -1 : 1;
        mjtNum s2 = (n2 >= 0) ? -1 : 1;
        mjtNum c1[2] = {s1 * box[0], s1 * box[0]};
        mjtNum c2[2] = {s2 * box[1], s2 * box[1]};
        if (box[0] * mju_abs(n1) <= box[1] * mju_abs(n2)) {
          c1[1] = -c1[0];
        } else {
          c2[1] = -c2[0];
        }
        for (int k = 0; k < 2; k++) {
          mjtNum corner[3];
          for (int r = 0; r < 3; r++) {
            corner[r] = c1[k] * m1[r] + c2[k] * m2[r];
          }
          mjtNum ldist = mju_dot3(corner, normal);
          if (dist + ldist > margin || (k == 1 && ldist > 0)) {
            continue;
          }
          con[ncon].dist = dist + ldist;
          mju_add3(con[ncon].pos, v, corner);
          mji_addToScl3(con[ncon].pos, normal, -0.5 * con[ncon].dist);
          mju_copy3(con[ncon].normal, normal);
          mju_zero3(con[ncon].tangent);
          vert[ncon] = i;
          ncon++;
        }
        continue;
      }
    }

    // no contact
    if (dist > margin + radius) {
      continue;
    }

    // create contact
    con[ncon].dist = dist - radius;
    mju_addScl3(con[ncon].pos, v, normal, -con[ncon].dist * 0.5 - radius);
    mju_copy3(con[ncon].normal, normal);
    mju_zero3(con[ncon].tangent);
    vert[ncon] = i;
    ncon++;
  }
  return ncon;
}


// filter contact based sphere-box test, treating sphere as box
static int filterSphereBox(const mjtNum s[3], mjtNum bound, const mjtNum aabb[6]) {
  if (s[0] + bound < aabb[0] - aabb[3]) return 1;
  if (s[1] + bound < aabb[1] - aabb[4]) return 1;
  if (s[2] + bound < aabb[2] - aabb[5]) return 1;
  if (s[0] - bound > aabb[0] + aabb[3]) return 1;
  if (s[1] - bound > aabb[1] + aabb[4]) return 1;
  if (s[2] - bound > aabb[2] + aabb[5]) return 1;
  return 0;
}


// filter contact based on global AABBs
static int filterBox(const mjtNum aabb1[6], const mjtNum aabb2[6], mjtNum margin) {
  if (aabb1[0] + aabb1[3] + margin < aabb2[0] - aabb2[3]) return 1;
  if (aabb1[1] + aabb1[4] + margin < aabb2[1] - aabb2[4]) return 1;
  if (aabb1[2] + aabb1[5] + margin < aabb2[2] - aabb2[5]) return 1;
  if (aabb2[0] + aabb2[3] + margin < aabb1[0] - aabb1[3]) return 1;
  if (aabb2[1] + aabb2[4] + margin < aabb1[1] - aabb1[4]) return 1;
  if (aabb2[2] + aabb2[5] + margin < aabb1[2] - aabb1[5]) return 1;
  return 0;
}


// make capsule from two flex vertices
static void makeCapsule(const mjModel* m, mjData* d, int f, const int vid[2],
                        mjtNum pos[3], mjtNum mat[9], mjtNum size[2]) {
  // get vertex positions
  mjtNum* v1 = d->flexvert_xpos + 3*(m->flex_vertadr[f] + vid[0]);
  mjtNum* v2 = d->flexvert_xpos + 3*(m->flex_vertadr[f] + vid[1]);

  // construct capsule from vertices
  mjtNum dif[3] = {v1[0]-v2[0], v1[1]-v2[1], v1[2]-v2[2]};
  size[0] = m->flex_radius[f];
  size[1] = 0.5*mju_normalize3(dif);

  mju_add3(pos, v1, v2);
  mju_scl3(pos, pos, 0.5);

  mjtNum quat[4];
  mju_quatZ2Vec(quat, dif);
  mju_quat2Mat(mat, quat);
}


// test a geom and an elem for collision, return number of contacts
int mjc_GeomElem(const mjModel* m, mjData* d, mjPreContact* con, int g, int f, int e,
                 mjtNum margin) {
  int dim = m->flex_dim[f], type = m->geom_type[g], ncon;

  // bounding sphere test: only if midphase is disabled
  if (mjDISABLED(mjDSBL_MIDPHASE)) {
    int eglobal = m->flex_elemadr[f] + e;
    if (filterSphereBox(d->geom_xpos + 3*g, m->geom_rbound[g] + margin,
                        d->flexelem_aabb + 6*eglobal)) {
      return 0;
    }
  }

  // skip if element has vertices on the same body as geom
  int b = m->geom_bodyid[g];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim + 1);
  const int* bdata = m->flex_vertbodyid + m->flex_vertadr[f];
  for (int i = 0; i <= dim; i++) {
    if (b >= 0 && b == bdata[edata[i]]) {
      return 0;
    }
  }

  // sphere/capsule/box : 1D element (box or capsule)
  if (dim == 1 && (type == mjGEOM_SPHERE || type == mjGEOM_CAPSULE || type == mjGEOM_BOX)) {
    mjtNum bpos[3], bmat[9], bsize[3];
    if (mj_rodBoxElem(m, d, f, e, bpos, bmat, bsize)) {
      switch (type) {
        case mjGEOM_SPHERE:
          ncon = mjraw_SphereBox(con, margin, d->geom_xpos + 3*g, d->geom_xmat + 9*g,
                                 m->geom_size + 3*g, bpos, bmat, bsize);
          break;
        case mjGEOM_CAPSULE:
          ncon = mjraw_CapsuleBox(con, margin, d->geom_xpos + 3*g, d->geom_xmat + 9*g,
                                  m->geom_size + 3*g, bpos, bmat, bsize);
          break;
        case mjGEOM_BOX:
          ncon = mjraw_BoxBox(con, margin, d->geom_xpos + 3*g, d->geom_xmat + 9*g,
                              m->geom_size + 3*g, bpos, bmat, bsize);
          break;
        default:
          ncon = 0;
      }
      return ncon;
    }

    // make capsule from vertices
    mjtNum pos[3], mat[9], size[2];
    makeCapsule(m, d, f, m->flex_elem + m->flex_elemdataadr[f] + e*2, pos, mat, size);

    // call raw primitive for corresponding geom type
    switch (type) {
      case mjGEOM_SPHERE:
        ncon = mjraw_SphereCapsule(con, margin, d->geom_xpos + 3*g, d->geom_xmat+9*g,
                                   m->geom_size + 3*g, pos, mat, size);
        break;
      case mjGEOM_CAPSULE:
        ncon = mjraw_CapsuleCapsule(con, margin, d->geom_xpos + 3*g, d->geom_xmat + 9*g,
                                    m->geom_size + 3*g, pos, mat, size);
        break;
      case mjGEOM_BOX:
        ncon = mjraw_CapsuleBox(con, margin, pos, mat, size, d->geom_xpos + 3*g,
                                d->geom_xmat + 9*g, m->geom_size + 3*g);
        for (int i=0; i < ncon; i++) {
          mju_scl3(con[i].normal, con[i].normal, -1);
        }
        break;
      default:
        ncon = 0;
    }
  }

  // heightfield : elem
  else if (type == mjGEOM_HFIELD) {
    ncon = mjc_HFieldElem(m, d, con, g, f, e, margin);
  }

  // sphere : triangle
  else if (type == mjGEOM_SPHERE && dim == 2) {
    const mjtNum* vertxpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
    ncon = sphereTriangle(con, margin,
                               d->geom_xpos + 3*g, m->geom_size[3*g],
                               vertxpos + 3*edata[0], vertxpos + 3*edata[1],
                               vertxpos + 3*edata[2], m->flex_radius[f]);
  }

  // box : triangle
  else if (type == mjGEOM_BOX && dim == 2) {
    const mjtNum* vertxpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
    ncon = boxTriangle(con, margin, d->geom_xpos + 3*g,
                             d->geom_xmat + 9*g, m->geom_size + 3*g,
                             vertxpos + 3*edata[0], vertxpos + 3*edata[1],
                             vertxpos + 3*edata[2], m->flex_radius[f]);
  }

  // capsule : triangle
  else if (type == mjGEOM_CAPSULE && dim == 2) {
    const mjtNum* vertxpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
    ncon = capsuleTriangle(con, margin, d->geom_xpos + 3*g, d->geom_xmat + 9*g,
                                 m->geom_size + 3*g, vertxpos + 3*edata[0], vertxpos + 3*edata[1],
                                 vertxpos + 3*edata[2], m->flex_radius[f]);
  }

  // general geom : elem
  else {
    ncon = mjc_ConvexElem(m, d, con, g, -1, -1, -1, f, e, margin);
  }

  return ncon;
}


// range [lo, hi] of the interpolation grid cells containing vertex v of flex f, per axis: a vertex
//   on an internal cell plane belongs to the cells on both sides, so that the result does not
//   depend on the side the vertex is rounded to; vertices within eps of a plane are treated as on
//   the plane, where eps is relative to the cell size and covers the rounding of flex_vert0 in
//   [0, 1] in single precision; if v falls in a pruned/empty cell, returns the active cell that
//   extrapolates v so mjc_ElemSharedNode reflects the nodes actually driving v
void mjc_vertCellRange(const mjModel* m, int f, int v, int lo[3], int hi[3]) {
  int gv = m->flex_vertadr[f] + v;
  const mjtNum* coord = m->flex_vert0 + 3*gv;
  const int* cellnum = m->flex_cellnum + 3*f;
  int cell = m->flex_vertcell[gv];
  if (cell >= 0) {
    int cx = cellnum[0], cy = cellnum[1], cz = cellnum[2];
    int ci = mjMAX(0, mjMIN(cx - 1, (int)mju_floor(coord[0] * cx)));
    int cj = mjMAX(0, mjMIN(cy - 1, (int)mju_floor(coord[1] * cy)));
    int ck = mjMAX(0, mjMIN(cz - 1, (int)mju_floor(coord[2] * cz)));
    if (cell != ci * cy * cz + cj * cz + ck) {
      lo[0] = hi[0] = cell / (cy * cz);
      lo[1] = hi[1] = (cell / cz) % cy;
      lo[2] = hi[2] = cell % cz;
      return;
    }
  }
  for (int i = 0; i < 3; i++) {
    mjtNum u = coord[i] * cellnum[i];
    mjtNum eps = 1e-6 * cellnum[i];
    lo[i] = mjMAX(0, mjMIN(cellnum[i] - 1, (int)mju_floor(u - eps)));
    hi[i] = mjMAX(0, mjMIN(cellnum[i] - 1, (int)mju_floor(u + eps)));
  }
}


// return 1 if cell ranges [lo1, hi1] and [lo2, hi2] are within distance 1 along every axis
int mjc_cellRangeShared(const int lo1[3], const int hi1[3], const int lo2[3], const int hi2[3]) {
  return lo2[0] - hi1[0] <= 1 && lo1[0] - hi2[0] <= 1 &&
         lo2[1] - hi1[1] <= 1 && lo1[1] - hi2[1] <= 1 &&
         lo2[2] - hi1[2] <= 1 && lo1[2] - hi2[2] <= 1;
}


// return 1 if elements e1 and e2 of flex f have vertices in the same or in neighboring cells of
//   its interpolation grid, 0 for non-interpolated flexes. Such elements are driven by a common
//   grid node (neighboring cells share a face, edge or corner of nodes, for both trilinear and
//   quadratic interpolation) and can only intersect if the grid inverts locally, which is resisted
//   by elasticity; their vertices have no body for the shared-body filter
int mjc_ElemSharedNode(const mjModel* m, int f, int e1, int e2) {
  if (!m->flex_interp[f]) {
    return 0;
  }

  int dim = m->flex_dim[f];
  const int* edata1 = m->flex_elem + m->flex_elemdataadr[f] + e1*(dim + 1);
  const int* edata2 = m->flex_elem + m->flex_elemdataadr[f] + e2*(dim + 1);
  int lo1[4][3], hi1[4][3], lo2[4][3], hi2[4][3];
  for (int i = 0; i <= dim; i++) {
    mjc_vertCellRange(m, f, edata1[i], lo1[i], hi1[i]);
    mjc_vertCellRange(m, f, edata2[i], lo2[i], hi2[i]);
  }

  // cell ranges within distance 1 along every axis
  for (int i1 = 0; i1 <= dim; i1++) {
    for (int i2 = 0; i2 <= dim; i2++) {
      if (mjc_cellRangeShared(lo1[i1], hi1[i1], lo2[i2], hi2[i2])) {
        return 1;
      }
    }
  }
  return 0;
}


// test two elems for collision, return number of contacts
int mjc_ElemElem(const mjModel* m, mjData* d, mjPreContact* con, int f1, int e1,
                 int f2, int e2, mjtNum margin) {
  int dim1 = m->flex_dim[f1], dim2 = m->flex_dim[f2];
  int ncon = 0;

  // ignore margin and gap in self-collisions
  if (f1 == f2) {
    margin = 0;
  }

  // bounding box filter (not applied in midphase)
  if (filterBox(d->flexelem_aabb + 6*(m->flex_elemadr[f1] + e1),
                d->flexelem_aabb + 6*(m->flex_elemadr[f2] + e2), margin)) {
    return 0;
  }

  // skip if elements have vertices on the same body
  const int* edata1 = m->flex_elem + m->flex_elemdataadr[f1] + e1*(dim1 + 1);
  const int* edata2 = m->flex_elem + m->flex_elemdataadr[f2] + e2*(dim2 + 1);
  const int* bdata1 = m->flex_vertbodyid + m->flex_vertadr[f1];
  const int* bdata2 = m->flex_vertbodyid + m->flex_vertadr[f2];
  for (int i1 = 0; i1 <= dim1; i1++) {
    int b1 = bdata1[edata1[i1]];
    for (int i2 = 0; i2 <= dim2; i2++) {
      if (b1 >= 0 && b1 == bdata2[edata2[i2]]) {
        return 0;
      }
    }
  }

  // 1D : 1D (box or capsule)
  if (dim1 == 1 && dim2 == 1) {
    mjtNum bpos1[3], bmat1[9], bsize1[3];
    mjtNum bpos2[3], bmat2[9], bsize2[3];
    int box1 = mj_rodBoxElem(m, d, f1, e1, bpos1, bmat1, bsize1);
    int box2 = mj_rodBoxElem(m, d, f2, e2, bpos2, bmat2, bsize2);
    if (box1 && box2) {
      return mjraw_BoxBox(con, margin, bpos1, bmat1, bsize1, bpos2, bmat2, bsize2);
    }

    mjtNum pos1[3], mat1[9], size1[2];
    mjtNum pos2[3], mat2[9], size2[2];
    if (box1) {
      makeCapsule(m, d, f2, m->flex_elem + m->flex_elemdataadr[f2] + e2*2, pos2, mat2, size2);
      ncon = mjraw_CapsuleBox(con, margin, pos2, mat2, size2, bpos1, bmat1, bsize1);
      for (int i = 0; i < ncon; i++) {
        mju_scl3(con[i].normal, con[i].normal, -1);
      }
      return ncon;
    }
    if (box2) {
      makeCapsule(m, d, f1, m->flex_elem + m->flex_elemdataadr[f1] + e1*2, pos1, mat1, size1);
      return mjraw_CapsuleBox(con, margin, pos1, mat1, size1, bpos2, bmat2, bsize2);
    }

    // make capsules from vertices
    makeCapsule(m, d, f1, m->flex_elem + m->flex_elemdataadr[f1] + e1*2,
                pos1, mat1, size1);
    makeCapsule(m, d, f2, m->flex_elem + m->flex_elemdataadr[f2] + e2*2,
                pos2, mat2, size2);

    // raw primitive
    ncon = mjraw_CapsuleCapsule(con, margin, pos1, mat1, size1, pos2, mat2, size2);
  }

  // general convex collision
  else {
    ncon = mjc_ConvexElem(m, d, con, -1, f1, e1, -1, f2, e2, margin);
  }
  return ncon;
}
