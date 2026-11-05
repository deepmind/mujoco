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

#include "engine/engine_util_misc.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mjmacro.h>
#include <mujoco/mjtype.h>
#include "engine/engine_array_safety.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_spatial.h"

//------------------------------ tendon wrapping ---------------------------------------------------

// check for intersection of two 2D line segments
static mjtBool is_intersect(const mjtNum* p1, const mjtNum* p2,
                            const mjtNum* p3, const mjtNum* p4) {
  mjtNum u[2] = {p2[0] - p1[0], p2[1] - p1[1]};
  mjtNum v[2] = {p4[0] - p3[0], p4[1] - p3[1]};

  // compute determinant, check
  mjtNum det = v[1]*u[0] - v[0]*u[1];
  if (mju_abs(det) < mjMINVAL) {
    return false;
  }

  // compute intersection point on each line
  mjtNum s = (det > 0 ? 1 : -1);
  mjtNum w[2] = {p1[0] - p3[0], p1[1] - p3[1]};
  mjtNum d[2] = {p2[0] - p4[0], p2[1] - p4[1]};
  mjtNum a0 = (v[0]*w[1] - v[1]*w[0]) * s;
  mjtNum a1 = (v[0]*d[1] - v[1]*d[0]) * s;
  mjtNum b0 = (u[0]*w[1] - u[1]*w[0]) * s;
  mjtNum b1 = (u[0]*d[1] - u[1]*d[0]) * s;

  return (a0 >= 0 && a1 < 0 && b0 >= 0 && b1 < 0);
}


// curve length along circle
static mjtNum length_circle(const mjtNum* p0, const mjtNum* p1, int ind, mjtNum radius) {
  // compute angle between 0 and pi
  mjtNum cross = p0[1]*p1[0] - p0[0]*p1[1];
  mjtNum angle = mju_atan2(mju_abs(cross), mju_dot(p0, p1, 2));

  // flip if necessary
  if (angle > 1e-6 && ((cross > 0 && ind) || (cross < 0 && !ind))) {
    angle = 2*mjPI - angle;
  }

  return radius*angle;
}


// signed angle swept from p to q around the origin, in (-pi, pi]
static mjtNum winding_angle(const mjtNum p[2], const mjtNum q[2]) {
  return mju_atan2(p[0]*q[1] - p[1]*q[0], p[0]*q[0] + p[1]*q[1]);
}


// 2D circle wrap
//  input:  pair of 2D endpoints in end[4], optional 2D side point in side[2], radius
//  output: return length of circular wrap or -1
//          pair of 2D points in pnt[4]
static mjtNum wrap_circle(mjtNum pnt[4], const mjtNum end[4], const mjtNum* side, mjtNum radius) {
  mjtNum sqlen0 = end[0]*end[0] + end[1]*end[1];
  mjtNum sqlen1 = end[2]*end[2] + end[3]*end[3];
  mjtNum sqrad = radius*radius;

  // either point inside circle or circle too small: no wrap
  if (sqlen0 < sqrad || sqlen1 < sqrad || radius < mjMINVAL) {
    return -1;
  }

  // points too close: no wrap
  mjtNum dif[2] = {end[2]-end[0], end[3]-end[1]};
  mjtNum dd = dif[0]*dif[0] + dif[1]*dif[1];
  if (dd < mjMINVAL) {
    return -1;
  }

  // find nearest point on line segment to origin: a*dif + d0
  mjtNum a = -(dif[0]*end[0]+dif[1]*end[1])/dd;
  if (a < 0) {
    a = 0;
  } else if (a > 1) {
    a = 1;
  }

  // check for intersection
  mjtNum tmp[2] = {a*dif[0] + end[0], a*dif[1] + end[1]};
  int clear = tmp[0]*tmp[0] + tmp[1]*tmp[1] > sqrad;

  // side: the path is homotopic to the polyline end0 -> side -> end1, whose winding angle around
  // the center determines the wrap orientation; the segment end0 -> end1 is in the same homotopy
  // class iff its winding angle is the same (they differ by multiples of 2*pi)
  int ccw = 0;
  mjtNum wside = 0;
  if (side) {
    wside = winding_angle(end, side) + winding_angle(side, end+2);
    if (clear && mju_abs(wside - winding_angle(end, end+2)) < mjPI) {
      return -1;
    }
    ccw = (wside > 0);
  } else if (clear) {
    return -1;
  }

  mjtNum sqrt0 = mju_sqrt(sqlen0 - sqrad);
  mjtNum sqrt1 = mju_sqrt(sqlen1 - sqrad);

  // construct the two solutions, compute goodness
  mjtNum sol[2][2][2], good[2];
  for (int i=0; i < 2; i++) {
    int sgn = (i == 0 ? 1 : -1);

    sol[i][0][0] = (end[0]*sqrad + sgn*radius*end[1]*sqrt0)/sqlen0;
    sol[i][0][1] = (end[1]*sqrad - sgn*radius*end[0]*sqrt0)/sqlen0;
    sol[i][1][0] = (end[2]*sqrad - sgn*radius*end[3]*sqrt1)/sqlen1;
    sol[i][1][1] = (end[3]*sqrad + sgn*radius*end[2]*sqrt1)/sqlen1;

    // goodness: shorter path
    mju_sub(tmp, sol[i][0], sol[i][1], 2);
    good[i] = -mju_dot(tmp, tmp, 2);

    // penalize for intersection
    if (is_intersect(end, sol[i][0], end+2, sol[i][1])) {
      good[i] = -10000;
    }
  }

  // select the solution: with side, by orientation (i=1 is counterclockwise), otherwise the better
  int i = side ? ccw : (good[0] > good[1] ? 0 : 1);
  pnt[0] = sol[i][0][0];
  pnt[1] = sol[i][0][1];
  pnt[2] = sol[i][1][0];
  pnt[3] = sol[i][1][1];

  // check for intersection
  if (is_intersect(end, pnt, end+2, pnt+2)) {
    return -1;
  }

  // side: the path winds like the polyline, so the arc angle is its winding minus that of the
  // tangent segments; unlike length_circle, this is accurate for wraps close to 2*pi
  if (side) {
    mjtNum arc = wside - winding_angle(end, pnt) - winding_angle(pnt+2, end+2);
    return radius*mju_abs(arc);
  }

  // return curve length
  return length_circle(sol[i][0], sol[i][1], i, radius);
}


// 2D inside wrap
//  input: pair of 2D endpoints in end[4], radius
//  output: pair of 2D points in pnt[4]; return 0 if wrap, -1 if no wrap
static mjtNum wrap_inside(mjtNum pnt[4], const mjtNum end[4], mjtNum radius) {
  // algorithm parameters
  const int maxiter = 20;
  const mjtNum tolerance = 1e-6;

  // constants
  mjtNum len0 = mju_norm(end, 2);
  mjtNum len1 = mju_norm(end+2, 2);
  mjtNum dif[2] = {end[2]-end[0], end[3]-end[1]};
  mjtNum dd = dif[0]*dif[0] + dif[1]*dif[1];

  // either point inside circle or circle too small: no wrap
  if (len0 <= radius || len1 <= radius || radius < mjMINVAL || len0 < mjMINVAL || len1 < mjMINVAL) {
    return -1;
  }

  // segment-circle intersection: no wrap
  if (dd > mjMINVAL) {
    // find nearest point on line segment to origin: d0 + a*dif
    mjtNum a = -(dif[0]*end[0] + dif[1]*end[1]) / dd;

    // in segment
    if (a > 0 && a < 1) {
      mjtNum tmp[2];
      mju_addScl(tmp, end, dif, a, 2);
      if (mju_norm(tmp, 2) <= radius) {
        return -1;
      }
    }
  }

  // prepare default in case of numerical failure: average
  pnt[0] = 0.5*(end[0] + end[2]);
  pnt[1] = 0.5*(end[1] + end[3]);
  mju_normalize(pnt, 2);
  mju_scl(pnt, pnt, radius, 2);
  pnt[2] = pnt[0];
  pnt[3] = pnt[1];

  // compute function parameters: asin(A*z) + asin(B*z) - 2*asin(z) + G = 0
  mjtNum A = radius/len0;
  mjtNum B = radius/len1;
  mjtNum cosG = (len0*len0 + len1*len1 - dd) / (2*len0*len1);
  if (cosG < -1+mjMINVAL) {
    return -1;
  } else if (cosG > 1-mjMINVAL) {
    return 0;
  }
  mjtNum G = mju_acos(cosG);

  // init
  mjtNum phi = mjPI/2;
  mjtNum z = mju_sin(phi);
  mjtNum f = mju_asin(A*z) + mju_asin(B*z) - 2*phi + G;

  // make sure init is not on the other side
  if (f > 0) {
    return 0;
  }

  // Newton method
  int iter;
  for (iter=0; iter < maxiter && mju_abs(f) > tolerance; iter++) {
    // derivative
    mjtNum df = mju_cos(phi)*(A/mju_max(mjMINVAL, mju_sqrt(1-z*z*A*A)) +
                              B/mju_max(mjMINVAL, mju_sqrt(1-z*z*B*B))) - 2;

    // check sign; SHOULD NOT OCCUR
    if (df > -mjMINVAL) {
      return 0;
    }

    // new point
    mjtNum phi1 = phi - f/df;

    // make sure we are moving to the left; SHOULD NOT OCCUR
    if (phi1 > phi) {
      return 0;
    }

    // update solution
    phi = phi1;
    z = mju_sin(phi);
    f = mju_asin(A*z) + mju_asin(B*z) - 2*phi + G;

    // exit if positive; SHOULD NOT OCCUR
    if (f > tolerance) {
      return 0;
    }
  }

  // check convergence
  if (iter >= maxiter) {
    return 0;
  }

  // finalize: rotation by ang from vec = a or b, depending on cross(a,b) sign
  mjtNum vec[2];
  mjtNum ang;
  if (end[0]*end[3] - end[1]*end[2] > 0) {
    mju_copy(vec, end, 2);
    ang = phi - mju_asin(A*z);
  } else {
    mju_copy(vec, end+2, 2);
    ang = phi - mju_asin(B*z);
  }
  mju_normalize(vec, 2);
  pnt[0] = radius*(mju_cos(ang)*vec[0] - mju_sin(ang)*vec[1]);
  pnt[1] = radius*(mju_sin(ang)*vec[0] + mju_cos(ang)*vec[1]);
  pnt[2] = pnt[0];
  pnt[3] = pnt[1];

  return 0;
}


// wrap tendons around spheres and cylinders
//  input:  x0, x1: pair of 3D endpoints
//          xpos, xmat, radius: position, orientation and radius of geom
//          type: wrap type (mjtWrap)
//          side: 3D position of sidesite
//  output: return wrap length, -1 if no wrap
//          wpnt: pair of 3D wrap points
mjtNum mju_wrap(mjtNum wpnt[6], const mjtNum x0[3], const mjtNum x1[3],
                const mjtNum xpos[3], const mjtNum xmat[9], mjtNum radius,
                int type, const mjtNum side[3]) {
  // check object type;  SHOULD NOT OCCUR
  if (type != mjWRAP_SPHERE && type != mjWRAP_CYLINDER) {
    mjERROR("unknown wrapping object type %d", type);
  }

  // map sites to wrap object's local frame
  mjtNum tmp[3];
  mju_sub3(tmp, x0, xpos);
  mjtNum p[2][3];
  mju_mulMatTVec3(p[0], xmat, tmp);
  mju_sub3(tmp, x1, xpos);
  mju_mulMatTVec3(p[1], xmat, tmp);

  // too close to origin: return
  if (mju_norm3(p[0]) < mjMINVAL || mju_norm3(p[1]) < mjMINVAL) {
    return -1;
  }

  // construct 2D frame for circle wrap
  mjtNum axis[2][3];
  if (type == mjWRAP_SPHERE) {
    // 1st axis = p0
    mju_copy3(axis[0], p[0]);
    mju_normalize3(axis[0]);

    // normal to p0-0-p1 plane = cross(p0, p1)
    mjtNum normal[3];
    mju_cross(normal, p[0], p[1]);
    mjtNum nrm = mju_normalize3(normal);

    // if (p0, p1) parallel: different normal
    if (nrm < mjMINVAL) {
      // find max component of axis0
      int i = 0;
      if (mju_abs(axis[0][1]) > mju_abs(axis[0][0]) &&
          mju_abs(axis[0][1]) > mju_abs(axis[0][2])) {
        i = 1;
      }
      if (mju_abs(axis[0][2]) > mju_abs(axis[0][0]) &&
          mju_abs(axis[0][2]) > mju_abs(axis[0][1])) {
        i = 2;
      }

      // init second axis: 0 at i; 1 elsewhere
      axis[1][0] = 1;
      axis[1][1] = 1;
      axis[1][2] = 1;
      axis[1][i] = 0;

      // recompute normal
      mju_cross(normal, axis[0], axis[1]);
      mju_normalize3(normal);
    }

    // 2nd axis = cross(normal, p0)
    mju_cross(axis[1], normal, axis[0]);
    mju_normalize3(axis[1]);
  } else {
    // 1st axis = x
    axis[0][0] = 1;
    axis[0][1] = axis[0][2] = 0;

    // 2nd axis = y
    axis[1][1] = 1;
    axis[1][0] = axis[1][2] = 0;
  }

  // project points in 2D frame: p => d
  mjtNum s[3], d[4], sd[2];
  d[0] = mju_dot3(p[0], axis[0]);
  d[1] = mju_dot3(p[0], axis[1]);
  d[2] = mju_dot3(p[1], axis[0]);
  d[3] = mju_dot3(p[1], axis[1]);

  // handle sidesite
  if (side) {
    // side point: apply same projection as x0, x1
    mju_sub3(tmp, side, xpos);
    mju_mulMatTVec3(s, xmat, tmp);

    // side point: project and rescale
    sd[0] = mju_dot3(s, axis[0]);
    sd[1] = mju_dot3(s, axis[1]);
    mju_normalize(sd, 2);
    mju_scl(sd, sd, radius, 2);
  }

  // apply inside wrap
  mjtNum wlen;
  mjtNum pnt[4];
  if (side && mju_norm3(s) < radius) {
    wlen = wrap_inside(pnt, d, radius);
  }

  // apply circle wrap
  else {
    wlen = wrap_circle(pnt, d, (side ? sd : NULL), radius);
  }

  // no wrap: return
  if (wlen < 0) {
    return -1;
  }

  // reconstruct 3D points in local frame: res
  mjtNum res[6];
  for (int i=0; i < 2; i++) {
    // res = axis0*d0 + axis1*d1
    mju_scl3(res+3*i, axis[0], pnt[2*i]);
    mju_scl3(tmp, axis[1], pnt[2*i+1]);
    mju_addTo3(res+3*i, tmp);
  }

  // cylinder: correct along z
  if (type == mjWRAP_CYLINDER) {
    // set vertical coordinates
    mjtNum L0 = mju_sqrt((p[0][0]-res[0])*(p[0][0]-res[0]) + (p[0][1]-res[1])*(p[0][1]-res[1]));
    mjtNum L1 = mju_sqrt((p[1][0]-res[3])*(p[1][0]-res[3]) + (p[1][1]-res[4])*(p[1][1]-res[4]));
    res[2] = p[0][2] + (p[1][2] - p[0][2])*L0 / (L0+wlen+L1);
    res[5] = p[0][2] + (p[1][2] - p[0][2])*(L0+wlen) / (L0+wlen+L1);

    // correct wlen for height
    mjtNum height = mju_abs(res[5] - res[2]);
    wlen = mju_sqrt(wlen*wlen + height*height);
  }

  // map back to global frame: wpnt
  mju_mulMatVec3(wpnt, xmat, res);
  mju_mulMatVec3(wpnt+3, xmat, res+3);
  mju_addTo3(wpnt, xpos);
  mju_addTo3(wpnt+3, xpos);

  return wlen;
}


//------------------------------ misc geometry -----------------------------------------------------

// all 3 semi-axes of a geom
void mju_geomSemiAxes(mjtNum semiaxes[3], const mjtNum size[3], mjtGeom type) {
  switch (type) {
  case mjGEOM_SPHERE:
    semiaxes[0] = size[0];
    semiaxes[1] = size[0];
    semiaxes[2] = size[0];
    break;

  case mjGEOM_CAPSULE:
    semiaxes[0] = size[0];
    semiaxes[1] = size[0];
    semiaxes[2] = size[1] + size[0];
    break;

  case mjGEOM_CYLINDER:
    semiaxes[0] = size[0];
    semiaxes[1] = size[0];
    semiaxes[2] = size[1];
    break;

  default:
    semiaxes[0] = size[0];
    semiaxes[1] = size[1];
    semiaxes[2] = size[2];
  }
}


// return 1 if point is inside a primitive geom, 0 otherwise
int mju_insidePrimitive(const mjtNum pos[3], const mjtNum mat[9], const mjtNum size[3], mjtGeom type,
                        const mjtNum point[3]) {
  // vector from geom to point
  mjtNum vec[3];
  mju_sub3(vec, point, pos);

  // quick return for spheres, frame rotation not required
  if (type == mjGEOM_SPHERE) {
    return mju_dot3(vec, vec) < size[0]*size[0];
  }

  // rotate into local frame
  mjtNum plocal[3];
  mju_mulMatTVec3(plocal, mat, vec);

  // handle other geom types
  switch (type) {
  case mjGEOM_CAPSULE: {
    mjtNum z = plocal[2];
    mjtNum z_clamped = mju_clip(z, -size[1], size[1]);
    mjtNum z_dist_sq = (z - z_clamped) * (z - z_clamped);
    return (plocal[0]*plocal[0] + plocal[1]*plocal[1] + z_dist_sq < size[0]*size[0]);
  }

  case mjGEOM_ELLIPSOID:
    return (plocal[0]*plocal[0]/(size[0]*size[0]) +
            plocal[1]*plocal[1]/(size[1]*size[1]) +
            plocal[2]*plocal[2]/(size[2]*size[2]) < 1);

  case mjGEOM_CYLINDER:
    return (mju_abs(plocal[2]) < size[1] &&
            plocal[0]*plocal[0] + plocal[1]*plocal[1] < size[0]*size[0]);

  case mjGEOM_BOX:
    return (mju_abs(plocal[0]) < size[0] &&
            mju_abs(plocal[1]) < size[1] &&
            mju_abs(plocal[2]) < size[2]);

  case mjGEOM_PLANE:
    return plocal[2] < 0;

  default:
    return 0;
  }
}


// compute ray origin and direction for pixel (col, row) in camera image
// for perspective: origin is unchanged, direction is computed
// for orthographic: direction is -Z in camera frame, origin is offset from camera center
void mju_camPixelRay(mjtNum origin[3], mjtNum direction[3],
                     const mjtNum cam_xpos[3], const mjtNum cam_xmat[9],
                     int col, int row, mjtNum fx, mjtNum fy, mjtNum cx, mjtNum cy,
                     int projection, mjtNum ortho_extent) {
  // pixel center (row 0 = top of image)
  mjtNum px = col + 0.5 - cx;
  mjtNum py = row + 0.5 - cy;

  if (projection == mjPROJ_PERSPECTIVE) {
    // origin is camera position
    mju_copy3(origin, cam_xpos);

    // direction in camera frame: (x/fx, -y/fy, -1), then normalized
    mjtNum dir_cam[3] = {px / fx, -py / fy, -1.0};
    mju_mulMatVec3(direction, cam_xmat, dir_cam);
    mju_normalize3(direction);
  } else {
    // orthographic: parallel rays, direction is -Z in camera frame
    direction[0] = -cam_xmat[2];
    direction[1] = -cam_xmat[5];
    direction[2] = -cam_xmat[8];

    // origin offset in camera frame (ortho_extent is full height, use half for each side)
    mjtNum half_extent = ortho_extent / 2;
    mjtNum offset_cam[3] = {px / fx * half_extent, -py / fy * half_extent, 0};
    mjtNum offset_world[3];
    mju_mulMatVec3(offset_world, cam_xmat, offset_cam);
    mju_add3(origin, cam_xpos, offset_world);
  }
}


// ----------------------------- flex interpolation ------------------------------------------------

// use shared shape functions from engine_util_misc.h
#define phi mju_flexPhi
#define dphi mju_flexDphi

// evaluate the deformation gradient at p using the nodal dof values
void mju_defGradient(mjtNum res[9], const mjtNum p[3], const mjtNum* dof, int order) {
  int idx = 0;
  mjtNum gradient[3];
  mju_zero(res, 9);
  for (int i = 0; i <= order; i++) {
    for (int j = 0; j <= order; j++) {
      for (int k = 0; k <= order; k++) {
        gradient[0] = dphi(p[0], i, order) *  phi(p[1], j, order) *  phi(p[2], k, order);
        gradient[1] =  phi(p[0], i, order) * dphi(p[1], j, order) *  phi(p[2], k, order);
        gradient[2] =  phi(p[0], i, order) *  phi(p[1], j, order) * dphi(p[2], k, order);
        res[0] += dof[3*idx+0] * gradient[0];
        res[1] += dof[3*idx+0] * gradient[1];
        res[2] += dof[3*idx+0] * gradient[2];
        res[3] += dof[3*idx+1] * gradient[0];
        res[4] += dof[3*idx+1] * gradient[1];
        res[5] += dof[3*idx+1] * gradient[2];
        res[6] += dof[3*idx+2] * gradient[0];
        res[7] += dof[3*idx+2] * gradient[1];
        res[8] += dof[3*idx+2] * gradient[2];
        idx++;
      }
    }
  }
}

// evaluate the basis function at x for the i-th node
mjtNum mju_evalBasis(const mjtNum x[3], int i, int order) {
  if (order == 1) {
    return phi(x[2], i&1, order) * phi(x[1], i&2, order) * phi(x[0], i&4, order);
  } else if (order == 2) {
    return phi(x[2], i % 3, order) * phi(x[1], (i / 3) % 3, order) * phi(x[0], i / 9, order);
  } else {
    return -1;
  }
}

// evaluate the basis functions at x for all nodes in the cell
void mju_evalBasisArray(mjtNum* basis, const mjtNum x[3], int order) {
  if (order == 1) {
    mjtNum p[3][2] = {
      {1 - x[0], x[0]},
      {1 - x[1], x[1]},
      {1 - x[2], x[2]}
    };
    int j = 0;
    for (int i0=0; i0<2; i0++) {
      mjtNum w0 = p[0][i0];
      for (int i1=0; i1<2; i1++) {
        mjtNum w01 = w0 * p[1][i1];
        for (int i2=0; i2<2; i2++) {
          basis[j++] = w01 * p[2][i2];
        }
      }
    }
  } else if (order == 2) {
    mjtNum p[3][3];
    for (int d=0; d<3; d++) {
      for (int i=0; i<3; i++) {
        p[d][i] = phi(x[d], i, 2);
      }
    }
    int j = 0;
    for (int i0=0; i0<3; i0++) {
      mjtNum w0 = p[0][i0];
      for (int i1=0; i1<3; i1++) {
        mjtNum w01 = w0 * p[1][i1];
        for (int i2=0; i2<3; i2++) {
          basis[j++] = w01 * p[2][i2];
        }
      }
    }
  } else {
    int npoint = (order + 1) * (order + 1) * (order + 1);
    for (int j=0; j < npoint; j++) {
      basis[j] = mju_evalBasis(x, j, order);
    }
  }
}

// map global parametric coord to cell-local coord and build node indices
//   coord: [0,1]^3 parametric coordinates
//   cellnum: cell counts (cx, cy, cz)
//   order: interpolation order (1=trilinear, 2=triquadratic)
//   cell: flat cell index (ci*cy*cz + cj*cz + ck), or -1 to compute from coord
//   local: output local parametric coordinates within cell (extrapolated within [-0.02, 1.02]^3 if
//          cell differs from the natural cell of coord)
//   nodeindices: output array of global node indices for the cell (size (order+1)^3, may be NULL)
//   returns: number of nodes per cell (order+1)^3
int mju_cellLookup(const mjtNum coord[3], const int cellnum[3], int order, int cell,
                   mjtNum local[3], int* nodeindices) {
  int cx = cellnum[0], cy = cellnum[1], cz = cellnum[2];

  // find containing cell
  int ci = (int)mju_floor(coord[0] * cx);
  int cj = (int)mju_floor(coord[1] * cy);
  int ck = (int)mju_floor(coord[2] * cz);
  ci = mjMIN(ci, cx - 1); ci = mjMAX(ci, 0);
  cj = mjMIN(cj, cy - 1); cj = mjMAX(cj, 0);
  ck = mjMIN(ck, cz - 1); ck = mjMAX(ck, 0);

  // if precomputed cell differs from natural cell, redirect and extrapolate within [-0.02, 1.02]
  if (cell >= 0 && cell != ci * cy * cz + cj * cz + ck) {
    ck = cell % cz;
    cj = (cell / cz) % cy;
    ci = cell / (cy * cz);
    const mjtNum kMaxExtrap = 0.02;  // must match kSliverMargin in user_flexcomp.cc
    local[0] = mju_clip(coord[0] * cx - ci, -kMaxExtrap, 1 + kMaxExtrap);
    local[1] = mju_clip(coord[1] * cy - cj, -kMaxExtrap, 1 + kMaxExtrap);
    local[2] = mju_clip(coord[2] * cz - ck, -kMaxExtrap, 1 + kMaxExtrap);
  } else {
    local[0] = mju_clip(coord[0] * cx - ci, 0, 1);
    local[1] = mju_clip(coord[1] * cy - cj, 0, 1);
    local[2] = mju_clip(coord[2] * cz - ck, 0, 1);
  }

  // build node indices for this cell
  if (nodeindices) {
    int gi_base = ci * order;
    int gj_base = cj * order;
    int gk_base = ck * order;
    int ny_g = cy * order + 1;
    int nz_g = cz * order + 1;
    int ni = 0;
    for (int li = 0; li <= order; li++) {
      int gi = gi_base + li;
      int gi_stride = gi * ny_g * nz_g;
      for (int lj = 0; lj <= order; lj++) {
        int gj = gj_base + lj;
        int gj_stride = gi_stride + gj * nz_g;
        for (int lk = 0; lk <= order; lk++) {
          int gk = gk_base + lk;
          nodeindices[ni++] = gj_stride + gk;
        }
      }
    }
  }

  int npc = (order + 1) * (order + 1) * (order + 1);
  return npc;
}


// interpolate a function at x with given interpolation coefficients and order n
void mju_interpolate3D(mjtNum res[3], const mjtNum x[3], const mjtNum* coeff, int order,
                       const int* nodeindices) {
  int npoint = (order + 1) * (order + 1) * (order + 1);

  if (npoint > 27) {
    for (int j=0; j < npoint; j++) {
      int idx = nodeindices ? nodeindices[j] : j;
      mju_addToScl3(res, coeff+3*idx, mju_evalBasis(x, j, order));
    }
    return;
  }

  mjtNum basis[27];
  mju_evalBasisArray(basis, x, order);

  for (int j=0; j < npoint; j++) {
    int idx = nodeindices ? nodeindices[j] : j;
    mju_addToScl3(res, coeff+3*idx, basis[j]);
  }
}


static void flexInterpRotation(int order, const mjtNum* xpos_c,
                               const mjtNum dref[3], const mjtNum local[3],
                               mjtNum* quat) {
  mjtNum mat[9] = {0};

  if (order > 0) {
    mju_defGradient(mat, local, xpos_c, order);
    for (int col = 0; col < 3; col++) {
      if (dref[col] > mjMINVAL) {
        mjtNum inv = 1 / dref[col];
        mat[0*3 + col] *= inv;
        mat[1*3 + col] *= inv;
        mat[2*3 + col] *= inv;
      }
    }
  } else {
    // order 0: fallback to identity matrix
    mat[0] = 1;
    mat[4] = 1;
    mat[8] = 1;
  }

  // find rotation
  mju_unit4(quat);
  mju_mat2Rot(quat, mat);
  mju_negQuat(quat, quat);
}


// gather cell-local quantities and optionally compute rotation
void mju_flexGatherCellState(int order, int cy, int cz, int ci, int cj, int ck,
                             const mjtNum* xpos_g, const mjtNum* vel_g, const mjtNum* xpos0_g,
                             mjtNum* xpos_c, mjtNum* vel_c, mjtNum* xpos0_c,
                             int* nodeindices, mjtNum* quat) {
  int ny_g = cy * order + 1;
  int nz_g = cz * order + 1;

  int local = 0;
  for (int li = 0; li <= order; li++) {
    for (int lj = 0; lj <= order; lj++) {
      for (int lk = 0; lk <= order; lk++) {
        int gi = ci*order + li;
        int gj = cj*order + lj;
        int gk = ck*order + lk;
        int gidx = gi*ny_g*nz_g + gj*nz_g + gk;

        if (xpos_c && xpos_g) mju_copy3(xpos_c + 3*local, xpos_g + 3*gidx);
        if (vel_c && vel_g) mju_copy3(vel_c + 3*local, vel_g + 3*gidx);
        if (xpos0_c && xpos0_g) mju_copy3(xpos0_c + 3*local, xpos0_g + 3*gidx);
        if (nodeindices) nodeindices[local] = gidx;

        local++;
      }
    }
  }

  if (quat && xpos_c) {
    mjtNum p[3] = {.5, .5, .5};
    mjtNum dref[3] = {1, 1, 1};
    if (xpos0_g && order > 0) {
      int gidx0 = (ci*order)*ny_g*nz_g + (cj*order)*nz_g + (ck*order);
      int gidx1 = ((ci+1)*order)*ny_g*nz_g + ((cj+1)*order)*nz_g + ((ck+1)*order);
      mju_sub3(dref, xpos0_g + 3*gidx1, xpos0_g + 3*gidx0);
    }
    flexInterpRotation(order, xpos_c, dref, p, quat);
  }
}


// compute corotational rotation from 2D deformation gradient on a flat face
void mju_flexInterpRotation2D(int order, const mjtNum* xpos_f, int npe,
                              int axis0, int axis1, int normal_axis,
                              const mjtNum local[2], mjtNum* quat) {
  // compute 3x2 deformation gradient F at parametric point local
  mjtNum t1[3] = {0, 0, 0};  // tangent along axis0
  mjtNum t2[3] = {0, 0, 0};  // tangent along axis1
  int idx = 0;
  for (int l0 = 0; l0 <= order; l0++) {
    for (int l1 = 0; l1 <= order; l1++) {
      mjtNum grad0 = dphi(local[0], l0, order) * phi(local[1], l1, order);
      mjtNum grad1 = phi(local[0], l0, order) * dphi(local[1], l1, order);
      for (int d = 0; d < 3; d++) {
        t1[d] += xpos_f[3*idx + d] * grad0;
        t2[d] += xpos_f[3*idx + d] * grad1;
      }
      idx++;
    }
  }

  // normal = t1 x t2
  mjtNum normal[3];
  mju_cross(normal, t1, t2);

  // build 3x3 matrix with columns assigned to canonical axes (row-major)
  // axis0 → t1, axis1 → t2, normal_axis → normal
  // this ensures identity rotation for axis-aligned grids
  mjtNum mat[9] = {0};
  mjtNum* vecs[3];
  vecs[axis0] = t1;
  vecs[axis1] = t2;
  vecs[normal_axis] = normal;

  for (int col = 0; col < 3; col++) {
    mat[0*3 + col] = vecs[col][0];
    mat[1*3 + col] = vecs[col][1];
    mat[2*3 + col] = vecs[col][2];
  }

  // extract rotation via polar decomposition
  mju_unit4(quat);
  mju_mat2Rot(quat, mat);
  mju_negQuat(quat, quat);
}


// gather face-element-local quantities and optionally compute rotation (shell mode)
//
// face element enumeration for a grid with cell counts (cx, cy, cz):
//   face 0: x=0     cy*cz quads (normal=0)
//   face 1: x=max   cy*cz quads (normal=0)
//   face 2: y=0     cx*cz quads (normal=1)
//   face 3: y=max   cx*cz quads (normal=1)
//   face 4: z=0     cx*cy quads (normal=2)
//   face 5: z=max   cx*cy quads (normal=2)
void mju_flexGatherFaceState(int order, int cx, int cy, int cz,
                             int face_elem_idx,
                             const mjtNum* xpos_g, const mjtNum* vel_g,
                             const mjtNum* xpos0_g,
                             mjtNum* xpos_f, mjtNum* vel_f, mjtNum* xpos0_f,
                             int* nodeindices, mjtNum* quat) {
  int ny_g = cy * order + 1;
  int nz_g = cz * order + 1;
  int npe = (order + 1) * (order + 1);

  // face sizes and properties
  int face_sizes[6] = {cy*cz, cy*cz, cx*cz, cx*cz, cx*cy, cx*cy};
  int face_normal[6] = {0, 0, 1, 1, 2, 2};
  int face_count1[6] = {cz, cz, cx, cx, cy, cy};
  int face_fixed_vals[6];
  face_fixed_vals[0] = 0;
  face_fixed_vals[1] = cx * order;
  face_fixed_vals[2] = 0;
  face_fixed_vals[3] = cy * order;
  face_fixed_vals[4] = 0;
  face_fixed_vals[5] = cz * order;

  // determine which face and quad within face
  int face_id = 0;
  int within_face = face_elem_idx;
  int cumul = 0;
  for (int f = 0; f < 6; f++) {
    if (face_elem_idx < cumul + face_sizes[f]) {
      face_id = f;
      within_face = face_elem_idx - cumul;
      break;
    }
    cumul += face_sizes[f];
  }

  int normal_axis = face_normal[face_id];
  int na0 = (normal_axis + 1) % 3;  // slow in-plane axis
  int na1 = (normal_axis + 2) % 3;  // fast in-plane axis
  int c1 = face_count1[face_id];
  int g_fixed = face_fixed_vals[face_id];
  int q0 = within_face / c1;
  int q1 = within_face % c1;

  // gather nodes
  int local = 0;
  int gidx0 = 0, gidx1 = 0;
  for (int l0 = 0; l0 <= order; l0++) {
    for (int l1 = 0; l1 <= order; l1++) {
      int g[3];
      g[normal_axis] = g_fixed;
      g[na0] = q0 * order + l0;
      g[na1] = q1 * order + l1;
      int gidx = g[0] * ny_g * nz_g + g[1] * nz_g + g[2];
      if (local == 0) gidx0 = gidx;
      gidx1 = gidx;

      if (xpos_f && xpos_g) mju_copy3(xpos_f + 3*local, xpos_g + 3*gidx);
      if (vel_f && vel_g) mju_copy3(vel_f + 3*local, vel_g + 3*gidx);
      if (xpos0_f && xpos0_g) mju_copy3(xpos0_f + 3*local, xpos0_g + 3*gidx);
      if (nodeindices) nodeindices[local] = gidx;

      local++;
    }
  }

  if (quat && xpos_f) {
    if (xpos0_g && order > 0) {
      mjtNum d0 = xpos0_g[3*gidx1 + na0] - xpos0_g[3*gidx0 + na0];
      mjtNum d1 = xpos0_g[3*gidx1 + na1] - xpos0_g[3*gidx0 + na1];
      mjtNum t1[3] = {0, 0, 0};
      mjtNum t2[3] = {0, 0, 0};
      int idx = 0;
      for (int l0 = 0; l0 <= order; l0++) {
        for (int l1 = 0; l1 <= order; l1++) {
          mjtNum grad0 = dphi(0.5, l0, order) * phi(0.5, l1, order);
          mjtNum grad1 = phi(0.5, l0, order) * dphi(0.5, l1, order);
          for (int d = 0; d < 3; d++) {
            t1[d] += xpos_f[3*idx + d] * grad0;
            t2[d] += xpos_f[3*idx + d] * grad1;
          }
          idx++;
        }
      }
      if (d0 > mjMINVAL) mju_scl3(t1, t1, 1 / d0);
      if (d1 > mjMINVAL) mju_scl3(t2, t2, 1 / d1);
      mjtNum normal[3];
      mju_cross(normal, t1, t2);
      mjtNum mat[9] = {0};
      mjtNum* vecs[3];
      vecs[na0] = t1;
      vecs[na1] = t2;
      vecs[normal_axis] = normal;
      for (int col = 0; col < 3; col++) {
        mat[0*3 + col] = vecs[col][0];
        mat[1*3 + col] = vecs[col][1];
        mat[2*3 + col] = vecs[col][2];
      }
      mju_unit4(quat);
      mju_mat2Rot(quat, mat);
      mju_negQuat(quat, quat);
    } else {
      mjtNum p[2] = {.5, .5};
      mju_flexInterpRotation2D(order, xpos_f, npe, na0, na1, normal_axis, p, quat);
    }
  }
}


// compute unnormalized surface normal and tangent vectors at a parametric point
// on a 2D face element; normal = t1 x t2 (unnormalized)
void mju_flexFaceNormal2D(mjtNum normal[3], mjtNum t1[3], mjtNum t2[3],
                          int order, const mjtNum* xpos_f,
                          const mjtNum local[2]) {
  mju_zero3(t1);
  mju_zero3(t2);
  int idx = 0;
  for (int l0 = 0; l0 <= order; l0++) {
    for (int l1 = 0; l1 <= order; l1++) {
      mjtNum grad0 = dphi(local[0], l0, order) *  phi(local[1], l1, order);
      mjtNum grad1 =  phi(local[0], l0, order) * dphi(local[1], l1, order);
      for (int d = 0; d < 3; d++) {
        t1[d] += xpos_f[3*idx + d] * grad0;
        t2[d] += xpos_f[3*idx + d] * grad1;
      }
      idx++;
    }
  }
  mju_cross(normal, t1, t2);
}


// helper: get nodexpos value for node (i,j,k) in an nx*ny*nz grid
static inline const mjtNum* nodeAt(const mjtNum* nodexpos, int ny, int nz, int i, int j, int k) {
  return nodexpos + 3*(i*ny*nz + j*nz + k);
}

static inline mjtNum* nodeAtMut(mjtNum* nodexpos, int ny, int nz, int i, int j, int k) {
  return nodexpos + 3*(i*ny*nz + j*nz + k);
}

// reconstruct interior node positions from boundary nodes via Transfinite Interpolation
void mju_shellTrackInterior(mjtNum* nodexpos, int nx, int ny, int nz) {
  // need at least 3 nodes in each direction to have interior nodes
  if (nx < 3 || ny < 3 || nz < 3) {
    return;
  }

  for (int i = 1; i < nx-1; i++) {
    for (int j = 1; j < ny-1; j++) {
      for (int k = 1; k < nz-1; k++) {
        // parametric coordinates in [0, 1]
        mjtNum s = (mjtNum)i / (nx-1);
        mjtNum t = (mjtNum)j / (ny-1);
        mjtNum u = (mjtNum)k / (nz-1);

        mjtNum result[3] = {0, 0, 0};

        // --- face contributions (bilinear interpolation on each face pair) ---
        // x-faces: i=0 and i=nx-1
        for (int d = 0; d < 3; d++) {
          result[d] += (1-s) * nodeAt(nodexpos, ny, nz, 0, j, k)[d]
                     +    s  * nodeAt(nodexpos, ny, nz, nx-1, j, k)[d];
        }
        // y-faces: j=0 and j=ny-1
        for (int d = 0; d < 3; d++) {
          result[d] += (1-t) * nodeAt(nodexpos, ny, nz, i, 0, k)[d]
                     +    t  * nodeAt(nodexpos, ny, nz, i, ny-1, k)[d];
        }
        // z-faces: k=0 and k=nz-1
        for (int d = 0; d < 3; d++) {
          result[d] += (1-u) * nodeAt(nodexpos, ny, nz, i, j, 0)[d]
                     +    u  * nodeAt(nodexpos, ny, nz, i, j, nz-1)[d];
        }

        // --- edge corrections (subtract 12 edges, each linearly interpolated) ---
        // edges along x (4 edges: (j,k) at corners of y-z face)
        for (int d = 0; d < 3; d++) {
          result[d] -= (1-t)*(1-u) * nodeAt(nodexpos, ny, nz, i, 0, 0)[d];
          result[d] -= (1-t)*   u  * nodeAt(nodexpos, ny, nz, i, 0, nz-1)[d];
          result[d] -=    t *(1-u) * nodeAt(nodexpos, ny, nz, i, ny-1, 0)[d];
          result[d] -=    t *   u  * nodeAt(nodexpos, ny, nz, i, ny-1, nz-1)[d];
        }
        // edges along y (4 edges: (i,k) at corners of x-z face)
        for (int d = 0; d < 3; d++) {
          result[d] -= (1-s)*(1-u) * nodeAt(nodexpos, ny, nz, 0, j, 0)[d];
          result[d] -= (1-s)*   u  * nodeAt(nodexpos, ny, nz, 0, j, nz-1)[d];
          result[d] -=    s *(1-u) * nodeAt(nodexpos, ny, nz, nx-1, j, 0)[d];
          result[d] -=    s *   u  * nodeAt(nodexpos, ny, nz, nx-1, j, nz-1)[d];
        }
        // edges along z (4 edges: (i,j) at corners of x-y face)
        for (int d = 0; d < 3; d++) {
          result[d] -= (1-s)*(1-t) * nodeAt(nodexpos, ny, nz, 0, 0, k)[d];
          result[d] -= (1-s)*   t  * nodeAt(nodexpos, ny, nz, 0, ny-1, k)[d];
          result[d] -=    s *(1-t) * nodeAt(nodexpos, ny, nz, nx-1, 0, k)[d];
          result[d] -=    s *   t  * nodeAt(nodexpos, ny, nz, nx-1, ny-1, k)[d];
        }

        // --- corner corrections (add 8 corners back) ---
        for (int d = 0; d < 3; d++) {
          result[d] += (1-s)*(1-t)*(1-u) * nodeAt(nodexpos, ny, nz, 0, 0, 0)[d];
          result[d] += (1-s)*(1-t)*   u  * nodeAt(nodexpos, ny, nz, 0, 0, nz-1)[d];
          result[d] += (1-s)*   t *(1-u) * nodeAt(nodexpos, ny, nz, 0, ny-1, 0)[d];
          result[d] += (1-s)*   t *   u  * nodeAt(nodexpos, ny, nz, 0, ny-1, nz-1)[d];
          result[d] +=    s *(1-t)*(1-u) * nodeAt(nodexpos, ny, nz, nx-1, 0, 0)[d];
          result[d] +=    s *(1-t)*   u  * nodeAt(nodexpos, ny, nz, nx-1, 0, nz-1)[d];
          result[d] +=    s *   t *(1-u) * nodeAt(nodexpos, ny, nz, nx-1, ny-1, 0)[d];
          result[d] +=    s *   t *   u  * nodeAt(nodexpos, ny, nz, nx-1, ny-1, nz-1)[d];
        }

        // write result to interior node
        mju_copy3(nodexpos + 3*(i*ny*nz + j*nz + k), result);
      }
    }
  }
}


// distribute interior node forces to boundary nodes via transpose of Transfinite Interpolation
void mju_shellScatterInterior(mjtNum* nodefrc, int nx, int ny, int nz) {
  if (nx < 3 || ny < 3 || nz < 3) {
    return;
  }

  for (int i = 1; i < nx-1; i++) {
    for (int j = 1; j < ny-1; j++) {
      for (int k = 1; k < nz-1; k++) {
        mjtNum* f_ptr = nodeAtMut(nodefrc, ny, nz, i, j, k);
        mjtNum f[3] = {f_ptr[0], f_ptr[1], f_ptr[2]};
        mju_zero3(f_ptr);
        if (!f[0] && !f[1] && !f[2]) {
          continue;
        }

        mjtNum s = (mjtNum)i / (nx-1);
        mjtNum t = (mjtNum)j / (ny-1);
        mjtNum u = (mjtNum)k / (nz-1);

        // face contributions
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, j, k), f, 1-s);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, j, k), f, s);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, 0, k), f, 1-t);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, ny-1, k), f, t);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, j, 0), f, 1-u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, j, nz-1), f, u);

        // edge corrections
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, 0, 0), f, -(1-t)*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, 0, nz-1), f, -(1-t)*u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, ny-1, 0), f, -t*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, i, ny-1, nz-1), f, -t*u);

        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, j, 0), f, -(1-s)*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, j, nz-1), f, -(1-s)*u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, j, 0), f, -s*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, j, nz-1), f, -s*u);

        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, 0, k), f, -(1-s)*(1-t));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, ny-1, k), f, -(1-s)*t);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, 0, k), f, -s*(1-t));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, ny-1, k), f, -s*t);

        // corner corrections
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, 0, 0), f, (1-s)*(1-t)*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, 0, nz-1), f, (1-s)*(1-t)*u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, ny-1, 0), f, (1-s)*t*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, 0, ny-1, nz-1), f, (1-s)*t*u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, 0, 0), f, s*(1-t)*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, 0, nz-1), f, s*(1-t)*u);
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, ny-1, 0), f, s*t*(1-u));
        mju_addToScl3(nodeAtMut(nodefrc, ny, nz, nx-1, ny-1, nz-1), f, s*t*u);
      }
    }
  }
}


// helper to accumulate weights in a sparse list
static void addWeight(int* nb, int* body, mjtNum* bweight, int b, mjtNum w) {
  for (int i = 0; i < *nb; i++) {
    if (body[i] == b) {
      if (bweight) {
        bweight[i] += w;
      }
      return;
    }
  }
  body[*nb] = b;
  if (bweight) {
    bweight[*nb] = w;
  }
  (*nb)++;
}

// compute TFI weights for an interior node (i,j,k) and distribute to boundary nodes
void mju_shellTFIWeights(int nx, int ny, int nz, int i, int j, int k,
                         mjtNum w, int* nb, int* body, mjtNum* bweight,
                         const int* nodebodyid, int nstart) {
  mjtNum s = (mjtNum)i / (nx-1);
  mjtNum t = (mjtNum)j / (ny-1);
  mjtNum u = (mjtNum)k / (nz-1);

  #define NODE_ID(idx) (nodebodyid ? nodebodyid[nstart + (idx)] : nstart + (idx))

  // face contributions
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + j*nz + k), w * (1-s));
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + j*nz + k), w * s);

  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + 0*nz + k), w * (1-t));
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + (ny-1)*nz + k), w * t);

  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + j*nz + 0), w * (1-u));
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + j*nz + (nz-1)), w * u);

  // edge corrections
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + 0*nz + 0), -w * (1-t)*(1-u));
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + 0*nz + (nz-1)), -w * (1-t)*u);
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + (ny-1)*nz + 0), -w * t*(1-u));
  addWeight(nb, body, bweight, NODE_ID(i*ny*nz + (ny-1)*nz + (nz-1)), -w * t*u);

  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + j*nz + 0), -w * (1-s)*(1-u));
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + j*nz + (nz-1)), -w * (1-s)*u);
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + j*nz + 0), -w * s*(1-u));
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + j*nz + (nz-1)), -w * s*u);

  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + 0*nz + k), -w * (1-s)*(1-t));
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + (ny-1)*nz + k), -w * (1-s)*t);
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + 0*nz + k), -w * s*(1-t));
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + (ny-1)*nz + k), -w * s*t);

  // corner corrections
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + 0*nz + 0), w * (1-s)*(1-t)*(1-u));
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + 0*nz + (nz-1)), w * (1-s)*(1-t)*u);
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + (ny-1)*nz + 0), w * (1-s)*t*(1-u));
  addWeight(nb, body, bweight, NODE_ID(0*ny*nz + (ny-1)*nz + (nz-1)), w * (1-s)*t*u);
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + 0*nz + 0), w * s*(1-t)*(1-u));
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + 0*nz + (nz-1)), w * s*(1-t)*u);
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + (ny-1)*nz + 0), w * s*t*(1-u));
  addWeight(nb, body, bweight, NODE_ID((nx-1)*ny*nz + (ny-1)*nz + (nz-1)), w * s*t*u);

  #undef NODE_ID
}


//------------------------------ actuator models ---------------------------------------------------

// normalized muscle length-gain curve
mjtNum mju_muscleGainLength(mjtNum length, mjtNum lmin, mjtNum lmax) {
  if (lmin <= length && length <= lmax) {
    // mid-ranges (maximum is at 1.0)
    mjtNum a = 0.5*(lmin+1);
    mjtNum b = 0.5*(1+lmax);

    if (length <= a) {
      mjtNum x = (length-lmin) / mjMAX(mjMINVAL, a-lmin);
      return 0.5*x*x;
    } else if (length <= 1) {
      mjtNum x = (1-length) / mjMAX(mjMINVAL, 1-a);
      return 1 - 0.5*x*x;
    } else if (length <= b) {
      mjtNum x = (length-1) / mjMAX(mjMINVAL, b-1);
      return 1 - 0.5*x*x;
    } else {
      mjtNum x = (lmax-length) / mjMAX(mjMINVAL, lmax-b);
      return 0.5*x*x;
    }
  }

  return 0.0;
}


// muscle active force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
mjtNum mju_muscleGain(mjtNum len, mjtNum vel, const mjtNum lengthrange[2],
                      mjtNum acc0, const mjtNum prm[9]) {
  // unpack parameters
  mjtNum range[2] = {prm[0], prm[1]};
  mjtNum force    = prm[2];
  mjtNum scale    = prm[3];
  mjtNum lmin     = prm[4];
  mjtNum lmax     = prm[5];
  mjtNum vmax     = prm[6];
  mjtNum fvmax    = prm[8];

  // scale force if negative
  if (force < 0) {
    force = scale / mjMAX(mjMINVAL, acc0);
  }

  // optimum length
  mjtNum L0 = (lengthrange[1]-lengthrange[0]) / mjMAX(mjMINVAL, range[1]-range[0]);

  // normalized length and velocity
  mjtNum L = range[0] + (len-lengthrange[0]) / mjMAX(mjMINVAL, L0);
  mjtNum V = vel / mjMAX(mjMINVAL, L0*vmax);

  // length curve
  mjtNum FL = mju_muscleGainLength(L, lmin, lmax);

  // velocity curve
  mjtNum FV;
  mjtNum y = fvmax-1;
  if (V <= -1) {
    FV = 0;
  } else if (V <= 0) {
    FV = (V+1)*(V+1);
  } else if (V <= y) {
    FV = fvmax - (y-V)*(y-V) / mjMAX(mjMINVAL, y);
  } else {
    FV = fvmax;
  }

  // compute FVL and scale, make it negative
  return -force*FL*FV;
}


// muscle passive force, prm = (range[2], force, scale, lmin, lmax, vmax, fpmax, fvmax)
mjtNum mju_muscleBias(mjtNum len, const mjtNum lengthrange[2],
                      mjtNum acc0, const mjtNum prm[9]) {
  // unpack parameters
  mjtNum range[2] = {prm[0], prm[1]};
  mjtNum force    = prm[2];
  mjtNum scale    = prm[3];
  mjtNum lmax     = prm[5];
  mjtNum fpmax    = prm[7];

  // scale force if negative
  if (force < 0) {
    force = scale / mjMAX(mjMINVAL, acc0);
  }

  // optimum length
  mjtNum L0 = (lengthrange[1]-lengthrange[0]) / mjMAX(mjMINVAL, range[1]-range[0]);

  // normalized length
  mjtNum L = range[0] + (len-lengthrange[0]) / mjMAX(mjMINVAL, L0);

  // half-quadratic to (L0+lmax)/2, linear beyond
  mjtNum b = 0.5*(1+lmax);
  if (L <= 1) {
    return 0;
  } else if (L <= b) {
    mjtNum x = (L-1) / mjMAX(mjMINVAL, b-1);
    return -force*fpmax*0.5*x*x;
  } else {
    mjtNum x = (L-b) / mjMAX(mjMINVAL, b-1);
    return -force*fpmax*(0.5 + x);
  }
}


// muscle time constant with optional smoothing
mjtNum mju_muscleDynamicsTimescale(mjtNum dctrl, mjtNum tau_act, mjtNum tau_deact,
                                   mjtNum smoothing_width) {
  mjtNum tau;

  // hard switching
  if (smoothing_width < mjMINVAL) {
    tau = dctrl > 0 ? tau_act : tau_deact;
  }

  // smooth switching
  else {
    // scale by width, center around 0.5 midpoint, rescale to bounds
    tau = tau_deact + (tau_act-tau_deact)*mju_sigmoid(dctrl/smoothing_width + 0.5);
  }
  return tau;
}


// muscle activation dynamics, prm = (tau_act, tau_deact, smoothing_width)
mjtNum mju_muscleDynamics(mjtNum ctrl, mjtNum act, const mjtNum prm[3]) {
  // clamp control
  mjtNum ctrlclamp = mju_clip(ctrl, 0, 1);

  // clamp activation
  mjtNum actclamp = mju_clip(act, 0, 1);

  // compute timescales as in Millard et al. (2013) https://doi.org/10.1115/1.4023390
  mjtNum tau_act = prm[0] * (0.5 + 1.5*actclamp);    // activation timescale
  mjtNum tau_deact = prm[1] / (0.5 + 1.5*actclamp);  // deactivation timescale
  mjtNum smoothing_width = prm[2];                   // width of smoothing sigmoid
  mjtNum dctrl = ctrlclamp - act;                    // excess excitation

  mjtNum tau = mju_muscleDynamicsTimescale(dctrl, tau_act, tau_deact, smoothing_width);

  // filter output
  return dctrl / mjMAX(mjMINVAL, tau);
}


// LuGre Stribeck function: g(v) = F_C + (F_S - F_C) * exp(-(v/v_S)^2)
mjtNum mj_lugreStribeck(mjtNum velocity, mjtNum F_C, mjtNum F_S, mjtNum v_S) {
  mjtNum ratio = velocity / mju_max(mjMINVAL, v_S);
  return F_C + (F_S - F_C) * mju_exp(-ratio*ratio);
}


// compute DC motor activation slot indices from parameter arrays
mjDCMotorSlots mj_dcmotorSlots(const mjtNum* dynprm, const mjtNum* gainprm) {
  mjDCMotorSlots s = {-1, -1, -1, -1, -1, 0};
  if (dynprm[7] > 0)  s.slew        = s.num_slots++;  // slew rate limiting
  if (gainprm[5] > 0) s.integral    = s.num_slots++;  // PI integral
  if (dynprm[2] > 0)  s.temperature = s.num_slots++;  // thermal model
  if (dynprm[5] > 0)  s.bristle     = s.num_slots++;  // LuGre bristle
  if (dynprm[0] > 0)  s.current     = s.num_slots++;  // current filter

  return s;
}


//---------------------------------------- Base64 --------------------------------------------------

// decoding function for Base64
static uint32_t _decode(char ch) {
  if (ch >= 'A' && ch <= 'Z') {
    return ch - 'A';
  }

  if (ch >= 'a' && ch <= 'z') {
    return (ch - 'a') + 26;
  }

  if (ch >= '0' && ch <= '9') {
    return (ch - '0') + 52;
  }

  if (ch == '+') {
    return 62;
  }

  if (ch == '/') {
    return 63;
  }

  return 0;
}


// encode data as Base64 into buf (including padding and null char)
// returns number of chars written in buf: 4 * [(ndata + 2) / 3] + 1
size_t mju_encodeBase64(char* buf, const uint8_t* data, size_t ndata) {
  static const char *table =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  int i = 0, j = 0;

  // loop over 24 bit chunks
  while (i + 3 <= ndata) {
    // take next 24 bit chunk (3 bytes)
    uint32_t byte_1 = data[i++];
    uint32_t byte_2 = data[i++];
    uint32_t byte_3 = data[i++];

    // merge bytes into one 32 bit int
    uint32_t k = (byte_1 << 16) | (byte_2 << 8) | byte_3;

    // encode 6 bit chucks into four chars
    buf[j++] = table[(k >> 18) & 63];
    buf[j++] = table[(k >> 12) & 63];
    buf[j++] = table[(k >>  6) & 63];
    buf[j++] = table[(k >>  0) & 63];
  }

  // one byte left
  if (i + 1 == ndata) {
    uint32_t byte_1 = data[i];
    uint32_t k = byte_1 << 16;
    buf[j++] = table[(k >> 18) & 63];
    buf[j++] = table[(k >> 12) & 63];
    buf[j++] = '=';  // padding
    buf[j++] = '=';  // padding
  }

  // two bytes left
  if (i + 2 == ndata) {
    uint32_t byte_1 = data[i++];
    uint32_t byte_2 = data[i];

    uint32_t k = (byte_1 << 16) + (byte_2 << 8);

    buf[j++] = table[(k >> 18) & 63];
    buf[j++] = table[(k >> 12) & 63];
    buf[j++] = table[(k >>  6) & 63];
    buf[j++] = '=';  // padding
  }

  buf[j] = '\0';
  return 4 * ((ndata + 2) / 3) + 1;
}


// return size in decoded bytes if s is a valid Base64 encoding
// return 0 if s is empty or invalid Base64 encoding
size_t mju_isValidBase64(const char* s) {
  size_t i = 0;
  int pad = 0;  // 0, 1, or 2 zero padding at the end of s

  // validate chars
  for (; s[i] && s[i] != '='; i++) {
    if (!isalnum(s[i]) && s[i] != '/' && s[i] != '+') {
      return 0;
    }
  }

  // padding at end
  if (s[i] == '=') {
    if (!s[i + 1]) {
      pad = 1;  // one '=' padding at end
    } else if (s[i + 1] == '=' && !s[i + 2]) {
      pad = 2;  // two '=' padding at end
    } else {
      return 0;
    }
  }

  // strlen(s) must be a multiple of 4
  int len = i + pad;
  return len % 4 ? 0 : 3 * (len / 4) - pad;
}


// decode valid Base64 in string s into buf, undefined behavior if s is not valid Base64
// returns number of bytes decoded (upper limit of 3 * (strlen(s) / 4))
size_t mju_decodeBase64(uint8_t* buf, const char* s) {
  size_t i = 0, j = 0;

  // loop over 24 bit chunks
  while (s[i] != '\0') {
    // take next 24 bit chuck (4 chars; 6 bits each)
    uint32_t char_1 = _decode(s[i++]);
    uint32_t char_2 = _decode(s[i++]);
    uint32_t char_3 = _decode(s[i++]);
    uint32_t char_4 = _decode(s[i++]);

    // merge into 32 bit int
    uint32_t k = (char_1 << 18) | (char_2 << 12) | (char_3 << 6) | char_4;


    // write up to three bytes (exclude padding at end)
    buf[j++] = (k >> 16) & 0xFF;
    if (s[i - 2] != '=') {
      buf[j++] = (k >> 8) & 0xFF;
    }
    if (s[i - 1] != '=') {
      buf[j++] =  k & 0xFF;
    }
  }
  return j;
}


//------------------------------ history buffers ---------------------------------------------------

// convert logical index (0=oldest, n-1=newest) to physical index
// cursor points to the newest element (logical index n-1)
static inline int historyPhysicalIndex(int cursor, int n, int logical) {
  return (cursor + 1 + logical) % n;
}


// find logical index i such that times[i-1] < t <= times[i], using circular binary search
// returns 0 if t <= times[oldest], n if t > times[newest]
// cursor points to the newest element (logical index n-1)
static int historyFindIndex(const mjtNum* times, int n, int cursor, mjtNum t) {
  // get oldest and newest timestamps
  int oldest_phys = historyPhysicalIndex(cursor, n, 0);
  int newest_phys = historyPhysicalIndex(cursor, n, n-1);
  mjtNum t_oldest = times[oldest_phys];
  mjtNum t_newest = times[newest_phys];

  // before or at first element
  if (t <= t_oldest) {
    return 0;
  }

  // after last element
  if (t > t_newest) {
    return n;
  }

  // circular binary search: find smallest logical i such that times[phys(i)] >= t
  int lo = 0;
  int hi = n - 1;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    int mid_phys = historyPhysicalIndex(cursor, n, mid);
    if (times[mid_phys] < t) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  return hi;
}


// initialize history buffer with given times and values; times must be strictly increasing
// buffer layout: [user(1), cursor(1), times(n), values(n*dim)]
void mju_historyInit(mjtNum* buf, int n, int dim, const mjtNum* times, const mjtNum* values,
                     mjtNum user) {
  mjtNum* buf_times = buf + 2;
  mjtNum* buf_values = buf + 2 + n;
  int cursor = (times == buf_times) ? (int)buf[1] : n - 1;
  if (cursor < 0 || cursor >= n) {
    cursor = n - 1;
  }

  // check strict monotonicity of times
  for (int i=0; i < n-1; i++) {
    int idx0 = (times == buf_times) ? historyPhysicalIndex(cursor, n, i) : i;
    int idx1 = (times == buf_times) ? historyPhysicalIndex(cursor, n, i+1) : i+1;
    if (times[idx1] - times[idx0] < mjMINVAL) {
      mjERROR("times must be strictly increasing, got times[%d]=%g >= times[%d]=%g",
              i, times[idx0], i+1, times[idx1]);
    }
  }

  // buf layout: [user(1), cursor(1), times(n), values(n*dim)]
  buf[0] = user;            // user value
  buf[1] = (mjtNum)cursor;  // cursor points to newest (logical index n-1)

  if (times != buf_times) {
    mju_copy(buf_times, times, n);
    if (values) {
      mju_copy(buf_values, values, n*dim);
    }
  } else if (values) {
    for (int i=0; i < n; i++) {
      int phys = historyPhysicalIndex(cursor, n, i);
      mju_copy(buf_values + phys*dim, values + i*dim, dim);
    }
  }
}


// find insertion slot for time t, maintaining sorted order
// if t matches an existing timestamp, returns pointer to that slot
// if a new sample is inserted, the oldest sample is dropped
// returns pointer to value slot where caller should write dim values
mjtNum* mju_historyInsert(mjtNum* buf, int n, int dim, mjtNum t) {
  int cursor = (int)buf[1];
  mjtNum* times = buf + 2;
  mjtNum* values = buf + 2 + n;

  // find logical insertion index: times[i-1] < t <= times[i]
  int i = historyFindIndex(times, n, cursor, t);

  // exact match at logical i: return pointer to existing slot
  if (i < n) {
    int phys_i = historyPhysicalIndex(cursor, n, i);
    if (mju_abs(t - times[phys_i]) < mjMINVAL) {
      return values + phys_i*dim;
    }
  }

  // logical i == 0: new sample is older than oldest, replace oldest slot
  if (i == 0) {
    int oldest_phys = historyPhysicalIndex(cursor, n, 0);
    times[oldest_phys] = t;
    return values + oldest_phys*dim;
  }

  // logical i == n: new sample is newer than newest, advance cursor and write
  if (i == n) {
    cursor = (cursor + 1) % n;
    buf[1] = (mjtNum)cursor;

    // cursor now points to the new newest slot (which was the old oldest)
    times[cursor] = t;
    return values + cursor*dim;
  }

  // 0 < i < n: out-of-order insertion, shift [1, i-1] left (dropping 0), insert at i-1
  for (int j = 0; j < i-1; j++) {
    int src_phys = historyPhysicalIndex(cursor, n, j+1);
    int dst_phys = historyPhysicalIndex(cursor, n, j);
    times[dst_phys] = times[src_phys];
    mju_copy(values + dst_phys*dim, values + src_phys*dim, dim);
  }
  int insert_phys = historyPhysicalIndex(cursor, n, i-1);
  times[insert_phys] = t;
  return values + insert_phys*dim;
}


// read vector value at time t; interp: 0=zero-order-hold, 1=linear, 2=cubic spline
// returns pointer to sample in buffer on exact match or ZOH (res untouched)
// returns NULL and writes interpolated result to res on interpolation
const mjtNum* mju_historyRead(const mjtNum* buf, int n, int dim, mjtNum* res, mjtNum t, int interp) {
  int cursor = (int)buf[1];
  const mjtNum* times = buf + 2;
  const mjtNum* values = buf + 2 + n;

  int oldest_phys = historyPhysicalIndex(cursor, n, 0);
  int newest_phys = historyPhysicalIndex(cursor, n, n-1);
  mjtNum t_oldest = times[oldest_phys];
  mjtNum t_newest = times[newest_phys];

  // extrapolate before oldest: return pointer to oldest value
  if (t <= t_oldest + mjMINVAL) {
    return values + oldest_phys*dim;
  }

  // extrapolate after newest: return pointer to newest value
  if (t >= t_newest - mjMINVAL) {
    return values + newest_phys*dim;
  }

  // find bracketing logical index: times[i-1] < t <= times[i]
  int i = historyFindIndex(times, n, cursor, t);
  int phys_i = historyPhysicalIndex(cursor, n, i);

  // check for exact match at i
  if (mju_abs(t - times[phys_i]) < mjMINVAL) {
    return values + phys_i*dim;
  }

  // lo = i-1, hi = i (we know i > 0 because t > t_oldest)
  int phys_lo = historyPhysicalIndex(cursor, n, i-1);
  int phys_hi = phys_i;

  // zero-order hold: return pointer to lo (most recent sample <= t)
  if (interp == 0) {
    return values + phys_lo*dim;
  }

  mjtNum dt = times[phys_hi] - times[phys_lo];
  mjtNum alpha = (t - times[phys_lo]) / dt;

  // piecewise linear interpolation
  if (interp == 1) {
    for (int d = 0; d < dim; d++) {
      res[d] = values[phys_lo*dim+d] + alpha * (values[phys_hi*dim+d] - values[phys_lo*dim+d]);
    }
  }

  // cubic spline interpolation
  else {
    // Hermite basis functions
    mjtNum alpha2 = alpha * alpha;
    mjtNum alpha3 = alpha2 * alpha;
    mjtNum h00 = 2*alpha3 - 3*alpha2 + 1;
    mjtNum h10 = alpha3 - 2*alpha2 + alpha;
    mjtNum h01 = -2*alpha3 + 3*alpha2;
    mjtNum h11 = alpha3 - alpha2;

    for (int d = 0; d < dim; d++) {
      // finite differenced catmull-rom slopes, 0 at endpoints (constant extrapolation)

      mjtNum m_lo = 0;
      if (i > 1) {
        int phys_lo_prev = historyPhysicalIndex(cursor, n, i-2);
        mjtNum dt_lo = times[phys_hi] - times[phys_lo_prev];
        m_lo = (values[phys_hi*dim+d] - values[phys_lo_prev*dim+d]) / dt_lo;
      }

      mjtNum m_hi = 0;
      if (i < n - 1) {
        int phys_hi_next = historyPhysicalIndex(cursor, n, i+1);
        mjtNum dt_hi = times[phys_hi_next] - times[phys_lo];
        m_hi = (values[phys_hi_next*dim+d] - values[phys_lo*dim+d]) / dt_hi;
      }

      res[d] = h00 * values[phys_lo*dim+d] +
               h10 * dt * m_lo +
               h01 * values[phys_hi*dim+d] +
               h11 * dt * m_hi;
    }
  }

  return NULL;
}


//------------------------------ miscellaneous -----------------------------------------------------

// convert contact force to pyramid representation
// the pyramid frame is: V0_i = N + mu_i*T_i
//                       V1_i = N - mu_i*T_i
void mju_encodePyramid(mjtNum* pyramid, const mjtNum* force, const mjtNum* mu, int dim) {
  // special handling of frictionless contacts
  if (dim == 1) {
    pyramid[0] = force[0];
    return;
  }

  mjtNum a = force[0]/(dim-1), b;

  // arbitrary redundancy resolution:
  //  pyramid0_i + pyramid1_i = force_normal/(dim-1) = a
  //  pyramid0_i - pyramid1_i = force_tangent_i/mu_i = b
  for (int i=0; i < dim-1; i++) {
    b = mju_min(a, force[i+1]/mu[i]);
    pyramid[2*i] = 0.5*(a+b);
    pyramid[2*i+1] = 0.5*(a-b);
  }
}


// convert pyramid representation to contact force
void mju_decodePyramid(mjtNum* force, const mjtNum* pyramid, const mjtNum* mu, int dim) {
  // special handling of frictionless contacts
  if (dim == 1) {
    force[0] = pyramid[0];
    return;
  }

  // force_normal = sum(pyramid0_i + pyramid1_i)
  force[0] = 0;
  for (int i=0; i < 2*(dim-1); i++) {
    force[0] += pyramid[i];
  }

  // force_tangent_i = (pyramid0_i - pyramid1_i) * mu_i
  for (int i=0; i < dim-1; i++) {
    force[i+1] = (pyramid[2*i] - pyramid[2*i+1]) * mu[i];
  }
}


// integrate spring-damper analytically, return pos(t)
mjtNum mju_springDamper(mjtNum pos0, mjtNum vel0, mjtNum k, mjtNum b, mjtNum t) {
  mjtNum det, det_tol, c1, c2, r1, r2, w;

  // determinant of characteristic equation
  det = b*b - 4*k;

  // scale tolerance with determinant terms to make regime selection invariant to time units
  det_tol = mjMINVAL * mju_max(b*b, 4*mju_abs(k));

  // overdamping
  //  pos(t) = c1*exp(r1*t) + c2*exp(r2*t);  r12 = (-b +- sqrt(det))/2
  if (det > det_tol) {
    // compute w = sqrt(det)/2
    w = mju_sqrt(det)/2;

    // avoid cancellation: compute the root of larger magnitude, then use r1*r2 = k
    if (b >= 0) {
      r2 = -b/2 - w;
      r1 = k/r2;
    } else {
      r1 = -b/2 + w;
      r2 = k/r1;
    }

    // compute coefficients
    c1 = (pos0*r2-vel0) / (r2-r1);
    c2 = (pos0*r1-vel0) / (r1-r2);

    // evaluate result
    return c1*mju_exp(r1*t) + c2*mju_exp(r2*t);
  }

  // critical damping
  //  pos(t) = exp(-b*t/2) * (c1 + c2*t)
  else if (det <= det_tol && det >= -det_tol) {
    // compute coefficients
    c1 = pos0;
    c2 = vel0 + b*c1/2;

    // evaluate result
    return mju_exp(-b*t/2) * (c1 + c2*t);
  }

  // underdamping
  //  pos(t) = exp(-b*t/2) * (c1*cos(w*t) + c2*sin(w*t));  w = sqrt(abs(det))/2
  else {
    // compute w
    w = mju_sqrt(mju_abs(det))/2;

    // compute coefficients
    c1 = pos0;
    c2 = (vel0 + b*c1/2)/w;

    // evaluate result
    return mju_exp(-b*t/2) * (c1*mju_cos(w*t) + c2*mju_sin(w*t));
  }
}


// return 1 if point is outside box given by pos, mat, size * inflate
// return -1 if point is inside box given by pos, mat, size / inflate
// return 0 if point is between the inflated and deflated boxes
int mju_outsideBox(const mjtNum point[3], const mjtNum pos[3], const mjtNum mat[9],
                   const mjtNum size[3], mjtNum inflate) {
  // check inflation coefficient
  if (inflate < 1) {
    mjERROR("inflation coefficient must be >= 1")
  }

  // vector from pos to point, projected to box frame
  mjtNum vec[3] = {point[0]-pos[0], point[1]-pos[1], point[2]-pos[2]};
  mju_mulMatTVec3(vec, mat, vec);

  // big: inflated box
  mjtNum big[3] = {size[0], size[1], size[2]};
  if (inflate > 1) {
    mju_scl3(big, big, inflate);
  }

  // check if outside big box
  if (vec[0] > big[0] || vec[0] < -big[0] ||
      vec[1] > big[1] || vec[1] < -big[1] ||
      vec[2] > big[2] || vec[2] < -big[2]) {
    return 1;
  }

  // quick return if no inflation
  if (inflate == 1) {
    return -1;
  }

  // check if inside small (deflated) box
  mjtNum small[3] = {size[0]/inflate, size[1]/inflate, size[2]/inflate};
  if (vec[0] < small[0] && vec[0] > -small[0] &&
      vec[1] < small[1] && vec[1] > -small[1] &&
      vec[2] < small[2] && vec[2] > -small[2]) {
    return -1;
  }

  // within margin between small and big box
  return 0;
}


// print matrix to screen
void mju_printMat(const mjtNum* mat, int nr, int nc) {
  for (int r=0; r < nr; r++) {
    for (int c=0; c < nc; c++) {
      printf("%.8f ", mat[r*nc+c]);
    }
    printf("\n");
  }
  printf("\n");
}


// print sparse matrix to screen
void mju_printMatSparse(const mjtNum* mat, int nr,
                        const int* rownnz, const int* rowadr,
                        const int* colind) {
  for (int r=0; r < nr; r++) {
    for (int adr=rowadr[r]; adr < rowadr[r]+rownnz[r]; adr++) {
      printf("(%d %d): %9.6f  ", r, colind[adr], mat[adr]);
    }
    printf("\n");
  }
  printf("\n");
}


// min function, avoid re-evaluation
mjtNum mju_min(mjtNum a, mjtNum b) {
  return a <= b ? a : b;
}


// max function, avoid re-evaluation
mjtNum mju_max(mjtNum a, mjtNum b) {
  return a >= b ? a : b;
}


// clip x to the range [min, max]
mjtNum mju_clip(mjtNum x, mjtNum min, mjtNum max) {
  return x < min ? min : (x > max ? max : x);
}


// sign function
mjtNum mju_sign(mjtNum x) {
  return (x > 0.0) - (x < 0.0);
}


// round to nearest integer
int mju_round(mjtNum x) {
  double d = (double)x;
  if (d > INT_MAX) return INT_MAX;
  if (d < INT_MIN) return INT_MIN;
  return (int)round(d);
}


// convert type id to type name
const char* mju_type2Str(int type) {
  switch ((mjtObj) type) {
  case mjOBJ_BODY:
    return "body";

  case mjOBJ_XBODY:
    return "xbody";

  case mjOBJ_JOINT:
    return "joint";

  case mjOBJ_DOF:
    return "dof";

  case mjOBJ_GEOM:
    return "geom";

  case mjOBJ_SITE:
    return "site";

  case mjOBJ_CAMERA:
    return "camera";

  case mjOBJ_LIGHT:
    return "light";

  case mjOBJ_FLEX:
    return "flex";

  case mjOBJ_MESH:
    return "mesh";

  case mjOBJ_SKIN:
    return "skin";

  case mjOBJ_HFIELD:
    return "hfield";

  case mjOBJ_TEXTURE:
    return "texture";

  case mjOBJ_MATERIAL:
    return "material";

  case mjOBJ_PAIR:
    return "pair";

  case mjOBJ_EXCLUDE:
    return "exclude";

  case mjOBJ_EQUALITY:
    return "equality";

  case mjOBJ_TENDON:
    return "tendon";

  case mjOBJ_ACTUATOR:
    return "actuator";

  case mjOBJ_SENSOR:
    return "sensor";

  case mjOBJ_NUMERIC:
    return "numeric";

  case mjOBJ_TEXT:
    return "text";

  case mjOBJ_TUPLE:
    return "tuple";

  case mjOBJ_KEY:
    return "key";

  case mjOBJ_PLUGIN:
    return "plugin";

  case mjOBJ_FRAME:
    return "frame";

  default:
    return 0;
  }
}


// convert type id to type name
int mju_str2Type(const char* str) {
  if (!strcmp(str, "body")) {
    return mjOBJ_BODY;
  }

  else if (!strcmp(str, "xbody")) {
    return mjOBJ_XBODY;
  }

  else if (!strcmp(str, "joint")) {
    return mjOBJ_JOINT;
  }

  else if (!strcmp(str, "dof")) {
    return mjOBJ_DOF;
  }

  else if (!strcmp(str, "geom")) {
    return mjOBJ_GEOM;
  }

  else if (!strcmp(str, "site")) {
    return mjOBJ_SITE;
  }

  else if (!strcmp(str, "camera")) {
    return mjOBJ_CAMERA;
  }

  else if (!strcmp(str, "light")) {
    return mjOBJ_LIGHT;
  }

  else if (!strcmp(str, "flex")) {
    return mjOBJ_FLEX;
  }

  else if (!strcmp(str, "mesh")) {
    return mjOBJ_MESH;
  }

  else if (!strcmp(str, "skin")) {
    return mjOBJ_SKIN;
  }

  else if (!strcmp(str, "hfield")) {
    return mjOBJ_HFIELD;
  }

  else if (!strcmp(str, "texture")) {
    return mjOBJ_TEXTURE;
  }

  else if (!strcmp(str, "material")) {
    return mjOBJ_MATERIAL;
  }

  else if (!strcmp(str, "pair")) {
    return mjOBJ_PAIR;
  }

  else if (!strcmp(str, "exclude")) {
    return mjOBJ_EXCLUDE;
  }

  else if (!strcmp(str, "equality")) {
    return mjOBJ_EQUALITY;
  }

  else if (!strcmp(str, "tendon")) {
    return mjOBJ_TENDON;
  }

  else if (!strcmp(str, "actuator")) {
    return mjOBJ_ACTUATOR;
  }

  else if (!strcmp(str, "sensor")) {
    return mjOBJ_SENSOR;
  }

  else if (!strcmp(str, "numeric")) {
    return mjOBJ_NUMERIC;
  }

  else if (!strcmp(str, "text")) {
    return mjOBJ_TEXT;
  }

  else if (!strcmp(str, "tuple")) {
    return mjOBJ_TUPLE;
  }

  else if (!strcmp(str, "key")) {
    return mjOBJ_KEY;
  }

  else if (!strcmp(str, "plugin")) {
    return mjOBJ_PLUGIN;
  }

  else if (!strcmp(str, "frame")) {
    return mjOBJ_FRAME;
  }

  else {
    return mjOBJ_UNKNOWN;
  }
}


// return human readable number of bytes using standard letter suffix
const char* mju_writeNumBytes(size_t nbytes) {
  int i;
  static mjTHREADLOCAL char message[20];
  static const char suffix[] = " KMGTPE";
  for (i=0; i < 6; i++) {
    const size_t bits = (size_t)(1) << (10*(6-i));
    if (nbytes >= bits && !(nbytes & (bits - 1))) {
      break;
    }
  }
  if (i < 6) {
    mjSNPRINTF(message, "%zu%c", nbytes >> (10*(6-i)), suffix[6-i]);
  } else {
    mjSNPRINTF(message, "%zu", nbytes >> (10*(6-i)));
  }
  return message;
}


// warning text
const char* mju_warningText(int warning, size_t info) {
  static mjTHREADLOCAL char str[1000];

  switch ((mjtWarning) warning) {
  case mjWARN_INERTIA:
    mjSNPRINTF(str,
               "Inertia matrix is too close to singular, or the effective-metric solve failed to "
               "converge, at DOF %zu. Check model.",
               info);
    break;

  case mjWARN_CONTACTFULL:
    mjSNPRINTF(str,
               "Too many contacts. The arena memory is full, increase arena memory allocation."
               "(ncon = %zu)", info);
    break;

  case mjWARN_CNSTRFULL:
    mjSNPRINTF(str,
               "Insufficient arena memory for the number of constraints generated. "
               "Increase arena memory allocation above %s bytes.", mju_writeNumBytes(info));
    break;

  case mjWARN_BADQPOS:
    mjSNPRINTF(str, "Nan, Inf or huge value in QPOS at DOF %zu. The simulation is unstable.", info);
    break;

  case mjWARN_BADQVEL:
    mjSNPRINTF(str, "Nan, Inf or huge value in QVEL at DOF %zu. The simulation is unstable.", info);
    break;

  case mjWARN_BADQACC:
    mjSNPRINTF(str, "Nan, Inf or huge value in QACC at DOF %zu. The simulation is unstable.", info);
    break;

  case mjWARN_BADCTRL:
    mjSNPRINTF(str, "Nan, Inf or huge value in CTRL at ACTUATOR %zu. The simulation is unstable.",
               info);
    break;

  default:
    mjSNPRINTF(str, "Unknown warning type %d.", warning);
  }

  return str;
}


// return 1 if nan or abs(x)>mjMAXVAL, 0 otherwise
int mju_isBad(mjtNum x) {
  return (x != x || x > mjMAXVAL || x < -mjMAXVAL);
}


// return 1 if all elements are 0
int mju_isZero(const mjtNum* vec, int n) {
  for (int i=0; i < n; i++) {
    if (vec[i] != 0) {
      return 0;
    }
  }

  return 1;
}


// return 1 if all elements are 0
int mju_isZeroByte(const unsigned char* vec, int n) {
  if (!n || *vec) return !n;
  return memcmp(vec, vec + 1, n - 1) == 0;
}


// set integer vector to 0
void mju_zeroInt(int* res, int n) {
  memset(res, 0, n*sizeof(int));
}


// copy int vector vec into res
void mju_copyInt(int* res, const int* vec, int n) {
  memcpy(res, vec, n*sizeof(int));
}

// fill int vector with val
void mju_fillInt(int* res, int val, int n) {
  for (int i = 0; i < n; i++) {
    res[i] = val;
  }
}

// standard normal random number generator (optional second number)
mjtNum mju_standardNormal(mjtNum* num2) {
  const mjtNum scale = 2.0/((mjtNum)RAND_MAX);
  mjtNum x1, x2, w;

  do {
    x1 = scale * (mjtNum)rand() - 1.0;
    x2 = scale * (mjtNum)rand() - 1.0;
    w = x1 * x1 + x2 * x2;
  } while (w >= 1.0 || w == 0);

  w = mju_sqrt((-2.0 * mju_log(w)) / w);
  if (num2) {
    *num2 = x2 * w;
  }

  return (x1 * w);
}


// convert from float to mjtNum
void mju_f2n(mjtNum* res, const float* vec, int n) {
  for (int i=0; i < n; i++) {
    res[i] = (mjtNum) vec[i];
  }
}


// convert from mjtNum to float
void mju_n2f(float* res, const mjtNum* vec, int n) {
  for (int i=0; i < n; i++) {
    res[i] = (float) vec[i];
  }
}


// convert from double to mjtNum
void mju_d2n(mjtNum* res, const double* vec, int n) {
  for (int i=0; i < n; i++) {
    res[i] = (mjtNum) vec[i];
  }
}


// convert from mjtNum to double
void mju_n2d(double* res, const mjtNum* vec, int n) {
  for (int i=0; i < n; i++) {
    res[i] = (double) vec[i];
  }
}


// gather: res[i] = vec[ind[i]], or copy if ind is NULL
void mju_gather(mjtNum* restrict res, const mjtNum* restrict vec, const int* restrict ind, int n) {
  if (!ind) {
    mju_copy(res, vec, n);
    return;
  }

  for (int i=0; i < n; i++) {
    res[i] = vec[ind[i]];
  }
}


// masked gather (set to 0 at negative indices)
void mju_gatherMasked(mjtNum* restrict res, const mjtNum* restrict vec,
                      const int* restrict ind, int n) {
  for (int i=0; i < n; i++) {
    res[i] = ind[i] >= 0 ? vec[ind[i]] : 0;
  }
}


// scatter: res[ind[i]] = vec[i], or copy if ind is NULL
void mju_scatter(mjtNum* restrict res, const mjtNum* restrict vec, const int* restrict ind, int n) {
  if (!ind) {
    mju_copy(res, vec, n);
    return;
  }

  for (int i=0; i < n; i++) {
    res[ind[i]] = vec[i];
  }
}


// gather integers
void mju_gatherInt(int* restrict res, const int* restrict vec, const int* restrict ind, int n) {
  for (int i=0; i < n; i++) {
    res[i] = vec[ind[i]];
  }
}


// scatter integers
void mju_scatterInt(int* restrict res, const int* restrict vec, const int* restrict ind, int n) {
  for (int i=0; i < n; i++) {
    res[ind[i]] = vec[i];
  }
}


// build gather indices mapping src to res, assumes pattern(res) \subseteq pattern(src)
void mju_sparseMap(int* map, int nr,
                   const int* res_rowadr, const int* res_rownnz, const int* res_colind,
                   const int* src_rowadr, const int* src_rownnz, const int* src_colind) {
  for (int i = 0; i < nr; i++) {
    int res_cursor = res_rowadr[i];
    int res_end    = res_cursor + res_rownnz[i];
    int src_cursor = src_rowadr[i];
    int src_end    = src_cursor + src_rownnz[i];

    while (res_cursor < res_end) {
      int res_col = res_colind[res_cursor];
      while (src_cursor < src_end && src_colind[src_cursor] < res_col) {
        src_cursor++;
      }

      // found match, set index and advance cursors
      map[res_cursor++] = src_cursor++;
    }
  }
}


// build masked-gather map to copy a lower-triangular src into symmetric res
//  `cursor` is a preallocated buffer of size `nr`
void mju_lower2SymMap(int* map, int nr,
                      const int* res_rowadr, const int* res_rownnz, const int* res_colind,
                      const int* src_rowadr, const int* src_rownnz, const int* src_colind,
                      int* cursor) {
  if (!nr) return;

  // default all map entries to "no source"
  int nnz = res_rowadr[nr-1] + res_rownnz[nr-1];
  mju_fillInt(map, -1, nnz);

  // initialize per-row cursor
  for (int i = 0; i < nr; i++) {
    cursor[i] = res_rowadr[i];
  }

  // sweep src rows; for each lower (i,j) set res(i,j) and res(j,i)
  for (int i = 0; i < nr; i++) {
    int src_start = src_rowadr[i];
    int src_end   = src_start + src_rownnz[i];

    // sweep src row
    for (int k = src_start; k < src_end; k++) {
      int j = src_colind[k];
      if (j > i) break;  // use only lower triangle of src

      // --- lower triangle: res(i, j)
      int res_start = res_rowadr[i];
      int res_end   = res_start + res_rownnz[i];
      int c         = cursor[i];

      // increment c until there is a match
      while (c < res_end && res_colind[c] < j) c++;

      // found match, set index, advance and save cursor
      if (c < res_end && res_colind[c] == j) {
        map[c] = k;
        c++;
      }
      cursor[i] = c;


      // --- upper mirror: res(j, i)
      if (j != i) {
        res_start = res_rowadr[j];
        res_end   = res_start + res_rownnz[j];
        c         = cursor[j];

        // increment c until there is a match
        while (c < res_end && res_colind[c] < i) c++;

        // found match, set index and advance and save cursor
        if (c < res_end && res_colind[c] == i) {
          map[c] = k;
          c++;
        }
        cursor[j] = c;
      }
    }
  }
}


// insertion sort, increasing order
void mju_insertionSort(mjtNum* list, int n) {
  for (int i=1; i < n; i++) {
    mjtNum x = list[i];
    int j = i-1;
    while (j >= 0 && list[j] > x) {
      list[j+1] = list[j];
      j--;
    }
    list[j+1] = x;
  }
}


// integer insertion sort, increasing order
void mju_insertionSortInt(int* list, int n) {
  for (int i=1; i < n; i++) {
    int x = list[i];
    int j = i-1;
    while (j >= 0 && list[j] > x) {
      list[j+1] = list[j];
      j--;
    }
    list[j+1] = x;
  }
}


// Halton sequence
mjtNum mju_Halton(int index, int base) {
  int n0 = index;
  mjtNum b = (mjtNum)base;
  mjtNum f = 1/b, hn = 0;

  while (n0 > 0) {
    int n1 = n0/base;
    int r = n0 - n1*base;
    hn += f*r;
    f /= b;
    n0 = n1;
  }

  return hn;
}


// Call strncpy, then set dst[n-1] = 0.
char* mju_strncpy(char *dst, const char *src, int n) {
  if (dst && src && n > 0) {
    strncpy(dst, src, n);
    dst[n-1] = 0;
  }

  return dst;
}


// polynomial force coefficient: force = -x * mju_polyForce(...)
//   flg_odd=0: linear + poly[0]*x   + poly[1]*x^2 + ...
//   flg_odd=1: linear + poly[0]*|x| + poly[1]*x^2 + ...  (p is even, p*x is odd)
mjtNum mju_polyForce(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd) {
  x = flg_odd ? mju_abs(x) : x;
  mjtNum res = linear;

  mjtNum xpow = 1;
  for (int i=0; i < n; i++) {
    xpow *= x;
    res += poly[i] * xpow;
  }

  return res;
}


// derivative of (x * mju_polyForce) w.r.t. x
mjtNum mjd_xPolyForce(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd) {
  x = flg_odd ? mju_abs(x) : x;
  mjtNum res = linear;

  mjtNum xpow = 1;
  for (int i=0; i < n; i++) {
    xpow *= x;
    res += (i+2) * poly[i] * xpow;
  }

  return res;
}


// potential energy: integral from 0 to x of mju_polyForce(t) * t dt
mjtNum mju_polyPotential(mjtNum linear, const mjtNum* poly, mjtNum x, int n, int flg_odd) {
  x = flg_odd ? mju_abs(x) : x;
  mjtNum res = 0.5 * linear * (x * x);

  mjtNum xpow = x;
  for (int i=0; i < n; i++) {
    xpow *= x;
    res += poly[i] / (i+3) * (xpow * x);
  }

  return res;
}


// sigmoid function over 0<=x<=1 using quintic polynomial
mjtNum mju_sigmoid(mjtNum x) {
  // fast return
  if (x <= 0) {
    return 0;
  }
  if (x >= 1) {
    return 1;
  }

  // sigmoid: f(x) = 6*x^5 - 15*x^4 + 10*x^3
  // solution of f(0) = f'(0) = f''(0) = 0, f(1) = 1, f'(1) = f''(1) = 0
  return x*x*x * (3*x * (2*x - 5) + 10);
}
