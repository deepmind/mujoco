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

#include "engine/engine_setconst.h"

#include <math.h>
#include <stdio.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_derivative.h"
#include "engine/engine_forward.h"
#include "engine/engine_io.h"
#include "engine/engine_memory.h"
#include "engine/engine_metric.h"
#include "engine/engine_sleep.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_solve.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"


// compute dof_M0 via composite rigid body algorithm
static void mj_setM0(mjModel* m, mjData* d) {
  mjtNum buf[6];
  mjtNum* crb = d->crb;
  int last_body = m->nbody - 1, nv = m->nv;

  // copy cinert into crb
  mju_copy(crb, d->cinert, 10*m->nbody);

  // backward pass over bodies, accumulate composite inertias
  for (int i=last_body; i > 0; i--) {
    if (m->body_parentid[i] > 0) {
      mju_addTo(crb+10*m->body_parentid[i], crb+10*i, 10);
    }
  }

  for (int i=0; i < nv; i++) {
    // precomute buf = crb_body_i * cdof_i
    mju_mulInertVec(buf, crb+10*m->dof_bodyid[i], d->cdof+6*i);

    // dof_M0(i) = armature inertia + cdof_i * (crb_body_i * cdof_i)
    mjtNum armature = m->dof_armature[i] + mj_actuatorArmature(m, mjOBJ_JOINT, m->dof_jntid[i]);
    m->dof_M0[i] = armature + mju_dot(d->cdof+6*i, buf, 6);
  }
}


// helper function to get the tree id of a wrap object
static int GetWrapBodyTreeId(const mjModel* m, int wrap_index) {
  int bodyid = -1;
  int objid = m->wrap_objid[wrap_index];
  switch ((mjtWrap)m->wrap_type[wrap_index]) {
  case mjWRAP_JOINT:
    bodyid = m->jnt_bodyid[objid];
    break;
  case mjWRAP_SITE:
    bodyid = m->site_bodyid[objid];
    break;
  case mjWRAP_SPHERE:
  case mjWRAP_CYLINDER:
    bodyid = m->geom_bodyid[objid];
    break;
  case mjWRAP_PULLEY:
  case mjWRAP_NONE:
    break;
  }
  return (bodyid != -1) ? m->body_treeid[bodyid] : -1;
}


// tolerance for frame comparison, must match compiler's kFrameEps
static const mjtNum kSameFrameEps = 1e-6;


// return 1 if all 3 elements are near zero
static int isNullVec3(const mjtNum v[3]) {
  return mju_abs(v[0]) < kSameFrameEps &&
         mju_abs(v[1]) < kSameFrameEps &&
         mju_abs(v[2]) < kSameFrameEps;
}


// return 1 if quaternion is near identity (1,0,0,0), accounting for double-cover
static int isNullQuat(const mjtNum q[4]) {
  int plus  = mju_abs(q[0] - 1) < kSameFrameEps &&
              mju_abs(q[1]) < kSameFrameEps &&
              mju_abs(q[2]) < kSameFrameEps &&
              mju_abs(q[3]) < kSameFrameEps;
  int minus = mju_abs(q[0] + 1) < kSameFrameEps &&
              mju_abs(q[1]) < kSameFrameEps &&
              mju_abs(q[2]) < kSameFrameEps &&
              mju_abs(q[3]) < kSameFrameEps;
  return plus || minus;
}


// return 1 if two quaternions are near equal, accounting for double-cover
static int isSameQuat(const mjtNum q1[4], const mjtNum q2[4]) {
  int plus  = mju_abs(q1[0] - q2[0]) < kSameFrameEps &&
              mju_abs(q1[1] - q2[1]) < kSameFrameEps &&
              mju_abs(q1[2] - q2[2]) < kSameFrameEps &&
              mju_abs(q1[3] - q2[3]) < kSameFrameEps;
  int minus = mju_abs(q1[0] + q2[0]) < kSameFrameEps &&
              mju_abs(q1[1] + q2[1]) < kSameFrameEps &&
              mju_abs(q1[2] + q2[2]) < kSameFrameEps &&
              mju_abs(q1[3] + q2[3]) < kSameFrameEps;
  return plus || minus;
}


// return 1 if two 3-vectors are near equal
static int isSameVec3(const mjtNum v1[3], const mjtNum v2[3]) {
  return mju_abs(v1[0] - v2[0]) < kSameFrameEps &&
         mju_abs(v1[1] - v2[1]) < kSameFrameEps &&
         mju_abs(v1[2] - v2[2]) < kSameFrameEps;
}


// recompute body_sameframe, geom_sameframe, site_sameframe from model geometry
static void setSameframe(mjModel* m) {
  // body_sameframe: compare body inertial frame to body frame
  for (int i=1; i < m->nbody; i++) {
    mjtNum* ipos = m->body_ipos+3*i;
    mjtNum* iquat = m->body_iquat+4*i;
    if (isNullVec3(ipos) && isNullQuat(iquat)) {
      m->body_sameframe[i] = mjSAMEFRAME_BODY;
    } else if (isNullQuat(iquat)) {
      m->body_sameframe[i] = mjSAMEFRAME_BODYROT;
    } else {
      m->body_sameframe[i] = mjSAMEFRAME_NONE;
    }
  }

  // geom_sameframe: compare geom frame to body and inertial frames
  for (int i=0; i < m->ngeom; i++) {
    int b = m->geom_bodyid[i];
    mjtNum* gpos = m->geom_pos+3*i;
    mjtNum* gquat = m->geom_quat+4*i;
    mjtNum* ipos = m->body_ipos+3*b;
    mjtNum* iquat = m->body_iquat+4*b;
    if (isNullVec3(gpos) && isNullQuat(gquat)) {
      m->geom_sameframe[i] = mjSAMEFRAME_BODY;
    } else if (isNullQuat(gquat)) {
      m->geom_sameframe[i] = mjSAMEFRAME_BODYROT;
    } else if (isSameVec3(gpos, ipos) && isSameQuat(gquat, iquat)) {
      m->geom_sameframe[i] = mjSAMEFRAME_INERTIA;
    } else if (isSameQuat(gquat, iquat)) {
      m->geom_sameframe[i] = mjSAMEFRAME_INERTIAROT;
    } else {
      m->geom_sameframe[i] = mjSAMEFRAME_NONE;
    }
  }

  // site_sameframe: compare site frame to body and inertial frames
  for (int i=0; i < m->nsite; i++) {
    int b = m->site_bodyid[i];
    mjtNum* spos = m->site_pos+3*i;
    mjtNum* squat = m->site_quat+4*i;
    mjtNum* ipos = m->body_ipos+3*b;
    mjtNum* iquat = m->body_iquat+4*b;
    if (isNullVec3(spos) && isNullQuat(squat)) {
      m->site_sameframe[i] = mjSAMEFRAME_BODY;
    } else if (isNullQuat(squat)) {
      m->site_sameframe[i] = mjSAMEFRAME_BODYROT;
    } else if (isSameVec3(spos, ipos) && isSameQuat(squat, iquat)) {
      m->site_sameframe[i] = mjSAMEFRAME_INERTIA;
    } else if (isSameQuat(squat, iquat)) {
      m->site_sameframe[i] = mjSAMEFRAME_INERTIAROT;
    } else {
      m->site_sameframe[i] = mjSAMEFRAME_NONE;
    }
  }
}


// minimum value in bound and fluid computations, must match the compiler's mjEPS
static const double kBoundEps = 1e-14;


// bounding box in (min, max) format and bounding sphere radius of a geom in its frame, return 0 for
// mesh and SDF geoms, whose bounds are those of their mesh
static int geomBound(const mjModel* m, int g, mjtNum aamm[6], mjtNum* rbound) {
  const mjtNum* size = m->geom_size + 3*g;
  switch ((mjtGeom) m->geom_type[g]) {
  case mjGEOM_MESH:
  case mjGEOM_SDF:
    return 0;

  case mjGEOM_HFIELD: {
    const mjtNum* hsize = m->hfield_size + 4*m->geom_dataid[g];
    aamm[0] = -hsize[0];
    aamm[1] = -hsize[1];
    aamm[2] = -hsize[3];
    aamm[3] = hsize[0];
    aamm[4] = hsize[1];
    aamm[5] = hsize[2];
    *rbound = mju_sqrt(hsize[0]*hsize[0] + hsize[1]*hsize[1] +
                       mju_max(hsize[2]*hsize[2], hsize[3]*hsize[3]));
    return 1;
  }

  case mjGEOM_PLANE:
    aamm[0] = aamm[1] = aamm[2] = -mjMAXVAL;
    aamm[3] = aamm[4] = mjMAXVAL;
    aamm[5] = 0;
    *rbound = 0;
    return 1;

  case mjGEOM_SPHERE:
    aamm[3] = aamm[4] = aamm[5] = size[0];
    *rbound = size[0];
    break;

  case mjGEOM_CAPSULE:
    aamm[3] = aamm[4] = size[0];
    aamm[5] = size[0] + size[1];
    *rbound = size[0] + size[1];
    break;

  case mjGEOM_CYLINDER:
    aamm[3] = aamm[4] = size[0];
    aamm[5] = size[1];
    *rbound = mju_sqrt(size[0]*size[0] + size[1]*size[1]);
    break;

  case mjGEOM_ELLIPSOID:
    mju_copy3(aamm+3, size);
    *rbound = mju_max(mju_max(size[0], size[1]), size[2]);
    break;

  case mjGEOM_BOX:
    mju_copy3(aamm+3, size);
    *rbound = mju_sqrt(size[0]*size[0] + size[1]*size[1] + size[2]*size[2]);
    break;

  default:
    mju_copy3(aamm+3, size);
    *rbound = 0;
  }

  // symmetric box
  aamm[0] = -aamm[3];
  aamm[1] = -aamm[4];
  aamm[2] = -aamm[5];
  return 1;
}


// set geom bounds, body collision masks and margins, and the bounding volumes of bodies
static void setBound(mjModel* m, mjData* d) {
  mj_markStack(d);

  // bounding boxes and spheres of geoms whose bounds follow from their size
  for (int i=0; i < m->ngeom; i++) {
    if (m->geom_type[i] == mjGEOM_HFIELD) {
      const mjtNum* hsize = m->hfield_size + 4*m->geom_dataid[i];
      mjtNum* size = m->geom_size + 3*i;
      size[0] = hsize[0];
      size[1] = hsize[1];
      size[2] = 0.25*hsize[2] + 0.5*hsize[3];
    }

    mjtNum aamm[6], rbound;
    if (geomBound(m, i, aamm, &rbound)) {
      mjtNum* aabb = m->geom_aabb + 6*i;
      for (int k=0; k < 3; k++) {
        aabb[k] = (aamm[k+3] + aamm[k]) / 2;
        aabb[k+3] = (aamm[k+3] - aamm[k]) / 2;
      }
      m->geom_rbound[i] = rbound;
    }
  }

  // body collision masks and margins, accumulated over geoms
  for (int i=0; i < m->nbody; i++) {
    int contype = 0, conaffinity = 0;
    mjtNum margin = 0;
    for (int g=m->body_geomadr[i]; g < m->body_geomadr[i]+m->body_geomnum[i]; g++) {
      contype |= m->geom_contype[g];
      conaffinity |= m->geom_conaffinity[g];
      margin = mju_max(margin, m->geom_margin[g] + m->geom_gap[g]);
    }
    m->body_contype[i] = contype;
    m->body_conaffinity[i] = conaffinity;
    m->body_margin[i] = margin;
  }

  // bounding volume hierarchies of bodies, in their inertial frames: refit the hierarchy made by
  // the compiler, whose children follow their parents
  int nbvhmax = 0;
  for (int i=0; i < m->nbody; i++) {
    nbvhmax = mju_max(nbvhmax, m->body_bvhnum[i]);
  }
  mjtNum* node = mjSTACKALLOC(d, 6*mju_max(1, nbvhmax), mjtNum);
  int* inbvh = mjSTACKALLOC(d, mju_max(1, m->ngeom), int);
  mju_zeroInt(inbvh, m->ngeom);
  for (int i=0; i < m->nbody; i++) {
    int adr = m->body_bvhadr[i];
    int num = m->body_bvhnum[i];
    if (adr < 0) {
      continue;
    }
    const mjtNum* ipos = m->body_ipos + 3*i;
    const mjtNum* iquat = m->body_iquat + 4*i;
    mjtNum qinv[4];
    mju_negQuat(qinv, iquat);

    // bounds of nodes in (min, max) format, children before parents
    for (int j=num-1; j >= 0; j--) {
      mjtNum* aamm = node + 6*j;
      int g = m->bvh_nodeid[adr+j];

      // leaf: corners of the geom box, in the inertial frame
      if (g >= 0) {
        inbvh[g] = 1;
        const mjtNum* aabb = m->geom_aabb + 6*g;
        const mjtNum* gpos = m->geom_pos + 3*g;
        const mjtNum* gquat = m->geom_quat + 4*g;
        aamm[0] = aamm[1] = aamm[2] = mjMAXVAL;
        aamm[3] = aamm[4] = aamm[5] = -mjMAXVAL;
        for (int v=0; v < 8; v++) {
          mjtNum vert[3], box[3];
          vert[0] = aabb[0] + (v & 1 ? aabb[3] : -aabb[3]);
          vert[1] = aabb[1] + (v & 2 ? aabb[4] : -aabb[4]);
          vert[2] = aabb[2] + (v & 4 ? aabb[5] : -aabb[5]);
          mju_rotVecQuat(box, vert, gquat);
          box[0] += gpos[0] - ipos[0];
          box[1] += gpos[1] - ipos[1];
          box[2] += gpos[2] - ipos[2];
          mju_rotVecQuat(vert, box, qinv);
          for (int k=0; k < 3; k++) {
            aamm[k] = mju_min(aamm[k], vert[k]);
            aamm[k+3] = mju_max(aamm[k+3], vert[k]);
          }
        }
      }

      // internal node: union of its children
      else {
        const mjtNum* child1 = node + 6*m->bvh_child[2*(adr+j)];
        const mjtNum* child2 = node + 6*m->bvh_child[2*(adr+j)+1];
        for (int k=0; k < 3; k++) {
          aamm[k] = mju_min(child1[k], child2[k]);
          aamm[k+3] = mju_max(child1[k+3], child2[k+3]);
        }
      }
    }

    // store in (center, size) format, inflate flat boxes
    for (int j=0; j < num; j++) {
      const mjtNum* aamm = node + 6*j;
      mjtNum* aabb = m->bvh_aabb + 6*(adr+j);
      for (int k=0; k < 3; k++) {
        mjtNum lo = aamm[k], hi = aamm[k+3];
        if (mju_abs(lo - hi) < kBoundEps) {
          lo -= kBoundEps;
          hi += kBoundEps;
        }
        aabb[k] = (hi + lo) / 2;
        aabb[k+3] = (hi - lo) / 2;
      }
    }
  }

  // colliding geoms must be in the hierarchy of their body, if it has one
  for (int i=0; i < m->ngeom; i++) {
    int b = m->geom_bodyid[i];
    if (m->body_bvhadr[b] >= 0 && !inbvh[i] && (m->geom_contype[i] || m->geom_conaffinity[i])) {
      mj_freeStack(d);
      mjERROR("geom %d collides but was not compiled to collide, recompile the model", i);
      return;
    }
  }

  mj_freeStack(d);
}


// 15-point Gauss-Kronrod quadrature of the added-mass coefficient of an ellipsoid with semi-axes
// (dx, dy, dz) along dx: the integral over l in [0, inf) of dx*dy*dz / sqrt((dx^2 + l)^3 *
// (dy^2 + l) * (dz^2 + l)), with the change of variables l = x^3 / (1-x)^2
static double addedMassKappa(double dx, double dy, double dz) {
  // K15 points x in [0, 1], unused, left in comment for completeness
  // static const double kronrod_x[15] = {
  //   0.00427231, 0.02544604, 0.06756779, 0.12923441, 0.20695638,
  //   0.29707742, 0.39610752, 0.50000000, 0.60389248, 0.70292258,
  //   0.79304362, 0.87076559, 0.93243221, 0.97455396, 0.99572769
  // };

  // K15 weights
  static const double kronrod_w[15] = {
    0.01146766, 0.03154605, 0.05239501, 0.07032663, 0.08450236,
    0.09517529, 0.10221647, 0.10474107, 0.10221647, 0.09517529,
    0.08450236, 0.07032663, 0.05239501, 0.03154605, 0.01146766
  };

  // l(x) at the K15 points x in [0, 1]
  static const double kronrod_l[15] = {
    7.865151709349917e-08, 1.7347976913907274e-05, 0.0003548008144506193,
    0.002846636252924549, 0.014094260903596077, 0.053063261727396636,
    0.17041978741317773, 0.5, 1.4036301548686991,
    3.9353484827022642, 11.644841677041734, 39.53187807410903,
    177.5711362220801, 1429.4772912937397, 54087.416549217705
  };

  // dl/dx at the K15 points
  static const double kronrod_d[15] = {
    5.538677720489877e-05, 0.002080868285293228, 0.016514126520723166,
    0.07261900344370877, 0.23985243401862602, 0.6868318249020725,
    1.8551129519182894, 5.0, 14.060031152313941,
    43.28941239611009, 156.58546376397112, 747.9826085305024,
    5827.4042950027115, 116754.0197944512, 25482945.327264845
  };

  double invdx2 = 1.0 / (dx*dx);
  double invdy2 = 1.0 / (dy*dy);
  double invdz2 = 1.0 / (dz*dz);

  // l is non-dimensionalized by scale, since it enters as 1 + l/d^2
  double scale = pow(dx*dx*dx*dy*dz, 0.4);
  double kappa = 0;
  for (int i=0; i < 15; i++) {
    double lambda = scale * kronrod_l[i];
    double denom = (1 + lambda*invdx2) *
                   sqrt((1 + lambda*invdx2) * (1 + lambda*invdy2) * (1 + lambda*invdz2));
    kappa += scale * kronrod_d[i] / denom * kronrod_w[i];
  }
  return kappa * invdx2;
}


// set the added mass and inertia of geoms in the ellipsoid fluid model, from their size
static void setFluid(mjModel* m) {
  for (int i=0; i < m->ngeom; i++) {
    mjtNum* fluid = m->geom_fluid + mjNFLUID*i;
    if (fluid[0] <= 0) {
      continue;
    }

    // semi-axes of the equivalent ellipsoid
    mjtNum semiaxes[3];
    mju_geomSemiAxes(semiaxes, m->geom_size + 3*i, (mjtGeom) m->geom_type[i]);
    double dx = semiaxes[0], dy = semiaxes[1], dz = semiaxes[2];

    // volume of the equivalent ellipsoid
    double volume = 4.0 / 3.0 * mjPI * dx * dy * dz;

    // kappa is invariant to permutation of its last two arguments
    double kx = addedMassKappa(dx, dy, dz);
    double ky = addedMassKappa(dy, dz, dx);
    double kz = addedMassKappa(dz, dx, dy);

    // virtual moment of inertia factors: if (kz-ky) in the numerator is negative, so is the
    // denominator, take absolute values of both and clip to kBoundEps
    double yz = dy*dy - dz*dz;
    double zx = dz*dz - dx*dx;
    double xy = dx*dx - dy*dy;
    double Ixfac = yz*yz * fabs(kz - ky) /
                   mju_max(kBoundEps, fabs(2*yz + (dy*dy + dz*dz)*(ky - kz)));
    double Iyfac = zx*zx * fabs(kx - kz) /
                   mju_max(kBoundEps, fabs(2*zx + (dz*dz + dx*dx)*(kz - kx)));
    double Izfac = xy*xy * fabs(ky - kx) /
                   mju_max(kBoundEps, fabs(2*xy + (dx*dx + dy*dy)*(kx - ky)));

    // virtual mass and inertia, entries 6-11 of the fluid coefficients
    fluid[6] = volume * kx / mju_max(kBoundEps, 2 - kx);
    fluid[7] = volume * ky / mju_max(kBoundEps, 2 - ky);
    fluid[8] = volume * kz / mju_max(kBoundEps, 2 - kz);
    fluid[9] = volume * Ixfac / 5;
    fluid[10] = volume * Iyfac / 5;
    fluid[11] = volume * Izfac / 5;
  }
}


// set fixed quantities (do not depend on qpos0)
static void setFixed(mjModel* m, mjData* d) {
  mj_markStack(d);

  // ----- general

  // compute subtreemass
  for (int i=0; i < m->nbody; i++) {
    m->body_subtreemass[i] = m->body_mass[i];
  }
  for (int i=m->nbody-1; i > 0; i--) {
    m->body_subtreemass[m->body_parentid[i]] += m->body_subtreemass[i];
  }

  // compute ngravcomp: number of bodies with gravity compensation
  int ngravcomp = 0;
  for (int i=0; i < m->nbody; i++) {
    ngravcomp += (m->body_gravcomp[i] > 0);
  }
  m->ngravcomp = ngravcomp;
  m->flg_gravcomp = (ngravcomp > 0);

  // compute flg_surfacevel: whether any geom has nonzero surfacevel
  mjtBool flg_surfacevel = 0;
  for (int i=0; i < m->ngeom; i++) {
    const mjtNum* sv = m->geom_surfacevel + 6*i;
    if (sv[0] || sv[1] || sv[2] || sv[3] || sv[4] || sv[5]) {
      flg_surfacevel = 1;
      break;
    }
  }
  m->flg_surfacevel = flg_surfacevel;

  // compute flg_adhesion: whether any geom or pair has nonzero adhesion
  mjtBool flg_adhesion = 0;
  for (int i=0; i < m->ngeom; i++) {
    if (m->geom_adhesion[i]) {
      flg_adhesion = 1;
      break;
    }
  }
  for (int i=0; i < m->npair && !flg_adhesion; i++) {
    if (m->pair_adhesion[i]) {
      flg_adhesion = 1;
    }
  }
  m->flg_adhesion = flg_adhesion;

  // set jnt_actuatorid and tendon_actuatorid
  mju_fillInt(m->jnt_actuatorid, -1, m->njnt);
  mju_fillInt(m->tendon_actuatorid, -1, m->ntendon);
  for (int i=0; i < m->nactuator; i++) {
    // skip actuator with no damping and no armature
    if (m->actuator_damping[i] == 0 &&
        mju_isZero(m->actuator_dampingpoly+mjNPOLY*i, mjNPOLY) &&
        m->actuator_armature[i] == 0) {
      continue;
    }

    // joint or jointinparent transmission
    if (m->actuator_trntype[i] == mjTRN_JOINT ||
        m->actuator_trntype[i] == mjTRN_JOINTINPARENT) {
      int jntid = m->actuator_trnid[2*i];

      // first actuator: set id to i
      if (m->jnt_actuatorid[jntid] == -1) {
        m->jnt_actuatorid[jntid] = i;
      }

      // multiple actuators acting on single transmission: use -2 sentinel
      else {
        m->jnt_actuatorid[jntid] = -2;
      }
    }

    // tendon transmission
    else if (m->actuator_trntype[i] == mjTRN_TENDON) {
      int tenid = m->actuator_trnid[2*i];

      // first actuator: set id to i
      if (m->tendon_actuatorid[tenid] == -1) {
        m->tendon_actuatorid[tenid] = i;
      }

      // multiple actuators acting on single transmission: use -2 sentinel
      else {
        m->tendon_actuatorid[tenid] = -2;
      }
    }
  }

  // ----- tree related (body_treeid and dof_treeid already computed)

  // compute body_treeid
  for (int i=0; i < m->nbody; i++) {
    int weldid = m->body_weldid[i];
    if (m->body_dofnum[weldid]) {
      m->body_treeid[i] = m->dof_treeid[m->body_dofadr[weldid]];
    } else {
      m->body_treeid[i] = -1;
    }
  }

  // compute tree_bodyadr, tree_bodynum
  mju_zeroInt(m->tree_bodynum, m->ntree);
  int tree_current = -1;
  for (int i=1; i < m->nbody; i++) {
    int treeid = m->body_treeid[i];
    if (treeid != -1) {
      if (treeid > tree_current) {
        m->tree_bodyadr[++tree_current] = i;
      }
      m->tree_bodynum[tree_current]++;
    }
  }

  // compute tree_dofadr, tree_dofnum
  mju_zeroInt(m->tree_dofnum, m->ntree);
  tree_current = -1;
  for (int i=0; i < m->nv; i++) {
    if (m->dof_treeid[i] > tree_current) {
      m->tree_dofadr[++tree_current] = i;
    }
    m->tree_dofnum[tree_current]++;
  }

  // compute tendon_treeid, tendon_treenum
  int* tree_marker = mjSTACKALLOC(d, m->ntree, int);  // 1 if tree has been visited, 0 otherwise
  for (int i = 0; i < m->ntendon; i++) {
    mju_zeroInt(tree_marker, m->ntree);
    m->tendon_treenum[i] = 0;
    m->tendon_treeid[2*i] = -1;
    m->tendon_treeid[2*i+1] = -1;

    for (int j = m->tendon_adr[i]; j < m->tendon_adr[i] + m->tendon_num[i]; j++) {
      int wrap_treeid = GetWrapBodyTreeId(m, j);
      if (wrap_treeid != -1 && !tree_marker[wrap_treeid]) {
        tree_marker[wrap_treeid] = 1;
        if (m->tendon_treenum[i] == 0) {
          m->tendon_treeid[2*i] = wrap_treeid;
        } else if (m->tendon_treenum[i] == 1) {
          m->tendon_treeid[2*i+1] = wrap_treeid;
        }
        m->tendon_treenum[i]++;
      }
    }
  }

  // ----- apply compiler AUTO tree sleep policy

  // resolve the AUTO policy anew: trees which a previous call resolved return to AUTO
  for (int i=0; i < m->ntree; i++) {
    int policy = m->tree_sleep_policy[i];
    if (policy == mjSLEEP_AUTO_NEVER || policy == mjSLEEP_AUTO_ALLOWED) {
      m->tree_sleep_policy[i] = mjSLEEP_AUTO;
    }
  }

  // actuators: trees with any actuated joint, site, body, or tendon do not auto-sleep
  for (int i=0; i < m->nactuator; i++) {
    int bodyid = -1, bodyid2 = -1;
    int tid = m->actuator_trnid[2*i];
    int tid2 = m->actuator_trnid[2*i+1];
    switch ((mjtTrn)m->actuator_trntype[i]) {
    case mjTRN_JOINT:
    case mjTRN_JOINTINPARENT:
      bodyid = m->jnt_bodyid[tid];
      break;
    case mjTRN_SO3:
      bodyid = tid2 >= 0 ? m->site_bodyid[tid] : m->jnt_bodyid[tid];
      bodyid2 = tid2 >= 0 ? m->site_bodyid[tid2] : -1;
      break;
    case mjTRN_SITE:
    case mjTRN_SLIDERCRANK:
      bodyid = m->site_bodyid[tid];
      bodyid2 = tid2 >= 0 ? m->site_bodyid[tid2] : -1;
      break;
    case mjTRN_BODY:
      bodyid = tid;
      break;
    case mjTRN_TENDON:
      // wake all trees connected by this actuated tendon
      for (int j = m->tendon_adr[tid]; j < m->tendon_adr[tid] + m->tendon_num[tid]; j++) {
        int treeid = GetWrapBodyTreeId(m, j);
        if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
          m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
        }
      }
      continue;  // next actuator
    case mjTRN_UNDEFINED:
      continue;  // next actuator
    }

    // wake tree containing bodyid, if any
    if (bodyid != -1) {
      int treeid = m->body_treeid[bodyid];
      if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
        m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
      }
    }
    if (bodyid2 != -1) {
      int treeid = m->body_treeid[bodyid2];
      if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
        m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
      }
    }
  }

  // trees with inter-tree tendons that have non-zero stiffness or damping do not auto-sleep
  // if the tendon spans more than 2 trees.
  for (int i=0; i < m->ntendon; i++) {
    int treenum = m->tendon_treenum[i];

    // tendon spans 1 or 0 trees: skip
    if (treenum < 2) {
      continue;
    }

    // tendon spans 2 trees and has no stiffness or damping: skip
    if (treenum == 2 && !mj_tendonHasStiffness(m, i) && !mj_tendonHasDamping(m, i)) {
      continue;
    }

    // tendon spans two trees with stiffness or damping or more than two trees: wake all trees
    mju_zeroInt(tree_marker, m->ntree);
    for (int j = m->tendon_adr[i]; j < m->tendon_adr[i] + m->tendon_num[i]; j++) {
      int treeid = GetWrapBodyTreeId(m, j);

      // if the tree is not yet marked, mark it and wake it up
      if (treeid != -1 && !tree_marker[treeid]) {
        tree_marker[treeid] = 1;
        int policy = m->tree_sleep_policy[treeid];

        // mark tree as never sleeping
        if (policy == mjSLEEP_AUTO) {
          m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
        }

        // if the user marked it as sleepable, throw an error
        else if (policy == mjSLEEP_ALLOWED || policy == mjSLEEP_INIT) {
          mj_freeStack(d);
          if (treenum > 2) {
            mjERROR("tree %d connected to tendon %d which spans more than 2 trees, "
                    "sleeping not allowed", treeid, i);
          } else {
            mjERROR("tree %d connected to tendon %d with non-zero stiffness or damping, "
                    "sleeping not allowed", treeid, i);
          }
          return;
        }
      }
    }
  }

  // flexes: constraint-free trees are not allowed to sleep unless metric-coupled
  for (int i = 0; i < m->nflex; ++i) {
    // constrained or metric-coupled flexes are allowed to sleep
    if (m->flex_edgeequality[i] || mj_effFlexPossible(m, i)) {
      continue;
    }

    // node-based flex
    if (m->flex_interp[i]) {
      int nodenum = m->flex_nodenum[i];
      int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[i];
      for (int j = 0; j < nodenum; ++j) {
        int treeid = m->body_treeid[bodyid[j]];
        if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
          m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
        }
      }
    }

    // vertex-based flex
    else {
      int vertnum = m->flex_vertnum[i];
      int* bodyid = m->flex_vertbodyid + m->flex_vertadr[i];
      for (int j = 0; j < vertnum; ++j) {
        int treeid = m->body_treeid[bodyid[j]];
        if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
          m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
        }
      }
      if (m->flex_dim[i] == 1 && m->flex_bendingadr[i] >= 0) {
        const mjtNum* b = m->flex_bending + m->flex_bendingadr[i];
        int nhinge = (int)b[0];
        for (int e = 0; e <= nhinge; ++e) {
          int jnt = mj_rodChainTwistJnt(m, i, e);
          if (jnt >= 0) {
            int treeid = m->dof_treeid[m->jnt_dofadr[jnt]];
            if (treeid != -1 && m->tree_sleep_policy[treeid] == mjSLEEP_AUTO) {
              m->tree_sleep_policy[treeid] = mjSLEEP_AUTO_NEVER;
            }
          }
        }
      }
    }
  }

  // set remaining trees with mjSLEEP_AUTO policy to mjSLEEP_AUTO_ALLOWED
  for (int i = 0; i < m->ntree; i++) {
    if (m->tree_sleep_policy[i] == mjSLEEP_AUTO) {
      m->tree_sleep_policy[i] = mjSLEEP_AUTO_ALLOWED;
    }
  }

  mj_freeStack(d);
}

// compute tendon Jacobian sparsity
static void makeTendonSparse(mjModel* m) {
  int ntendon = m->ntendon;
  int* rownnz = m->ten_J_rownnz;
  int* rowadr = m->ten_J_rowadr;
  int* colind = m->ten_J_colind;

  if (!ntendon) {
    return;
  }

  // clear
  mju_zeroInt(rownnz, ntendon);
  mju_zeroInt(rowadr, ntendon);

  // compute rownnz, rowadr, and colind for each tendon
  for (int i = 0; i < ntendon; i++) {
    rowadr[i] = (i > 0 ? rowadr[i-1] + rownnz[i-1] : 0);
    int adr = m->tendon_adr[i];
    int num = m->tendon_num[i];

    // joint tendon: each wrap object is a joint, colind is its dofadr
    if (m->tendon_type[i] == mjTENDON_FIXED) {
      for (int j = 0; j < num; j++) {
        colind[rowadr[i] + j] = m->jnt_dofadr[m->wrap_objid[adr + j]];
      }
      rownnz[i] = num;
    } else {
      // spatial tendon: collect used dofs from wrap object bodies
      int nnz = 0;
      for (int j = 0; j < num; j++) {
        int type = m->wrap_type[adr + j];

        // get body id from site or geom wrap object
        int bodyid = -1;
        if (type == mjWRAP_SITE) {
          bodyid = m->site_bodyid[m->wrap_objid[adr + j]];
        } else if (type == mjWRAP_SPHERE || type == mjWRAP_CYLINDER) {
          bodyid = m->geom_bodyid[m->wrap_objid[adr + j]];
        }

        // walk up the body tree, collecting used dofs
        if (bodyid > 0) {
          int bid = bodyid;
          while (bid > 0) {
            int bdofadr = m->body_dofadr[bid];
            int bdofnum = m->body_dofnum[bid];
            for (int k = 0; k < bdofnum; k++) {
              int dof = bdofadr + k;

              // check if dof already in colind
              int found = 0;
              for (int l = 0; l < nnz; l++) {
                if (colind[rowadr[i] + l] == dof) {
                  found = 1;
                  break;
                }
              }

              // append new dof
              if (!found) {
                colind[rowadr[i] + nnz] = dof;
                nnz++;
              }
            }
            bid = m->body_parentid[bid];
          }
        }
      }
      rownnz[i] = nnz;
    }

    // sort colind for this tendon
    int nnz = rownnz[i];
    for (int j = 0; j < nnz - 1; j++) {
      for (int k = j + 1; k < nnz; k++) {
        // swap out-of-order entries
        if (colind[rowadr[i] + k] < colind[rowadr[i] + j]) {
          int tmp = colind[rowadr[i] + j];
          colind[rowadr[i] + j] = colind[rowadr[i] + k];
          colind[rowadr[i] + k] = tmp;
        }
      }
    }
  }
}

// compute flex sparsity: flexedge_J_{rowadr,rownnz,colind}
static void makeFlexSparse(mjModel* m, mjData* d) {
  int nv = m->nv;
  int* rowadr = m->flexedge_J_rowadr;
  int* rownnz = m->flexedge_J_rownnz;
  int* colind = m->flexedge_J_colind;

  if (!m->nflex) {
    return;
  }

  mj_markStack(d);
  int* chain = mjSTACKALLOC(d, nv, int);
  mjtNum* dummy_pos = mjSTACKALLOC(d, 3, mjtNum);
  mju_zero(dummy_pos, 3);

  // clear
  mju_zeroInt(rowadr, m->nflexedge);
  mju_zeroInt(rownnz, m->nflexedge);
  mju_zeroInt(m->flexedge_J_colind, m->nJfe);

  // compute lengths and Jacobians of edges
  int total_nnz = 0;
  for (int f = 0; f < m->nflex; f++) {
    int ebase = m->flex_edgeadr[f];

    // skip if edges cannot generate forces
    if (m->flex_rigid[f] || m->flex_interp[f]) {
      for (int e = 0; e < m->flex_edgenum[f]; e++) {
        rowadr[ebase + e] = total_nnz;
      }
      continue;
    }

    // skip Jacobian if no built-in passive force is needed
    int skipjacobian = !m->flex_edgeequality[f] && !m->flex_edgedamping[f] &&
                       !m->flex_edgestiffness[f] && !m->flex_damping[f];

    // process edges of this flex
    int vbase = m->flex_vertadr[f];
    for (int e = 0; e < m->flex_edgenum[f]; e++) {
      // set rowadr
      rowadr[ebase + e] = total_nnz;
      if (skipjacobian) {
        continue;
      }

      int v1 = m->flex_edge[2 * (ebase + e)];
      int v2 = m->flex_edge[2 * (ebase + e) + 1];
      int b1 = m->flex_vertbodyid[vbase + v1];
      int b2 = m->flex_vertbodyid[vbase + v2];

      // get sparsity
      int NV = mj_jacDifPair(m, d, chain, b1, b2, dummy_pos, dummy_pos, NULL,
                             NULL, NULL, NULL, NULL, NULL, /*issparse=*/1, /*skipcommon=*/0);

      // copy sparsity info
      rownnz[ebase + e] = NV;
      mju_copyInt(colind + total_nnz, chain, NV);
      total_nnz += NV;
    }
  }

  mj_freeStack(d);
}


// run kinematics in qpos0, which the quantities that depend on qpos0 use
static void kinematics0(mjModel* m, mjData* d) {
  // run computations in qpos0
  mju_copy(d->qpos, m->qpos0, m->nq);
  mj_kinematics(m, d);
  mj_comPos(m, d);

  // run remaining computations
  mj_tendon(m, d);
  mj_flexSkip(m, d, 0);
  mj_transmission(m, d);
}


// set kinematic quantities that depend on qpos0; assume kinematics0 already executed
static void setKinematics0(mjModel* m, mjData* d) {
  // copy fields
  mju_copy(m->flexedge_length0, d->flexedge_length, m->nflexedge);
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_interp[f]) {
      mju_zero(m->flexedge_length0 + m->flex_edgeadr[f], m->flex_edgenum[f]);
    }
  }
  mju_copy(m->tendon_length0, d->ten_length, m->ntendon);
  mju_copy(m->actuator_length0, d->actuator_length, m->nout);

  // equality constraints between bodies: the anchor of connects in the second body, and the
  // relative pose of welds which are marked in data[11] for it to be computed
  for (int i=0; i < m->neq; i++) {
    if (m->eq_objtype[i] != mjOBJ_BODY) {
      continue;
    }
    int id1 = m->eq_obj1id[i];
    int id2 = m->eq_obj2id[i];
    mjtNum* data = m->eq_data + mjNEQDATA*i;

    // connect: anchor position is in body1 local frame
    if (m->eq_type[i] == mjEQ_CONNECT) {
      mjtNum pos[3];
      mj_local2Global(d, pos, 0, data, 0, id1, 0);

      // data[3-5] = anchor position in body2 local frame
      mju_subFrom3(pos, d->xpos+3*id2);
      mju_mulMatTVec3(data+3, d->xmat+9*id2, pos);
    }

    // weld: anchor position is in body2 local frame
    else if (m->eq_type[i] == mjEQ_WELD && data[11]) {
      mjtNum pos[3], quat[4];
      mj_local2Global(d, pos, 0, data, 0, id2, 0);

      // data[3-5] = anchor position in body1 local frame
      mju_subFrom3(pos, d->xpos+3*id1);
      mju_mulMatTVec3(data+3, d->xmat+9*id1, pos);

      // data[6-9] = neg(xquat1)*xquat2 = "xquat2-xquat1" in body1 local frame
      mju_negQuat(quat, d->xquat+4*id1);
      mju_mulQuat(data+6, quat, d->xquat+4*id2);
    }
  }

  // camera compos0, pos0, mat0
  for (int i=0; i < m->ncam; i++) {
    // get body ids
    int id = m->cam_bodyid[i];              // camera body
    int id1 = m->cam_targetbodyid[i];       // target body

    // pose in fixed mode
    mjtNum pos[3], mat[9];
    mj_local2Global(d, pos, mat, m->cam_pos+3*i, m->cam_quat+4*i, id, 0);

    // compute positional offsets
    mju_sub3(m->cam_pos0+3*i, pos, d->xpos+3*id);
    mju_sub3(m->cam_poscom0+3*i, pos, d->subtree_com + (id1 >= 0 ? 3*id1 : 3*id));

    // copy mat
    mju_copy9(m->cam_mat0+9*i, mat);
  }

  // light compos0, pos0, dir0
  for (int i=0; i < m->nlight; i++) {
    // get body ids
    int id = m->light_bodyid[i];            // light body
    int id1 = m->light_targetbodyid[i];     // target body

    // pose in fixed mode
    mjtNum pos[3], dir[3];
    mj_local2Global(d, pos, 0, m->light_pos+3*i, 0, id, 0);
    mju_rotVecQuat(dir, m->light_dir+3*i, d->xquat+4*id);
    mju_normalize3(dir);

    // compute positional offsets
    mju_sub3(m->light_pos0+3*i, pos, d->xpos+3*id);
    mju_sub3(m->light_poscom0+3*i, pos, d->subtree_com + (id1 >= 0 ? 3*id1 : 3*id));

    // copy dir
    mju_copy3(m->light_dir0+3*i, dir);
  }
}


// set inertial quantities that depend on qpos0; assume kinematics0 already executed
static void setInertia0(mjModel* m, mjData* d) {
  int nv = m->nv;
  mjtNum A[36] = {0};
  mj_markStack(d);
  mjtNum* jac = mjSTACKALLOC(d, 6*nv, mjtNum);
  mjtNum* tmp = mjSTACKALLOC(d, 6*nv, mjtNum);
  mjtNum* moment = mjSTACKALLOC(d, nv, mjtNum);

  // compute dof_M0 for CRB algorithm
  mj_setM0(m, d);

  // factorize the inertia matrix
  mj_makeM(m, d);
  mj_factorM(m, d);

  // compute body_invweight0
  m->body_invweight0[0] = m->body_invweight0[1] = 0.0;
  for (int i=1; i < m->nbody; i++) {
    // bodies with no dofs (static and mocap): zero invweight0
    if (m->body_dofnum[m->body_weldid[i]] == 0) {
      m->body_invweight0[2*i] = m->body_invweight0[2*i+1] = 0;
    }

    // accelerate simple bodies with no rotations
    else if (m->body_simple[i] == 2) {
      mjtNum mass = m->body_mass[i];
      if (!mass) {  // SHOULD NOT OCCUR
        mjERROR("moving body %d has 0 mass", i);
      }
      m->body_invweight0[2*i+0] = 1/mju_max(mjMINVAL, mass);
      m->body_invweight0[2*i+1] = 0;
    }

    // general body: full inertia
    else {
      if (nv) {
        // inverse spatial inertia: A = J*inv(M)*J'
        mj_jacBodyCom(m, d, jac, jac+3*nv, i);
        mj_solveM(m, d, tmp, jac, 6);
        mju_mulMatMatT(A, jac, tmp, 6, nv, 6);
      }

      // average diagonal and assign
      m->body_invweight0[2*i] = (A[0] + A[7] + A[14])/3;
      m->body_invweight0[2*i+1] = (A[21] + A[28] + A[35])/3;
    }
  }

  // compute dof_invweight0
  for (int i=0; i < m->njnt; i++) {
    // simple body with no rotations: no off-diagonal inertia
    if (m->body_simple[m->jnt_bodyid[i]] == 2) {
      int id = m->jnt_dofadr[i];
      int bi = m->jnt_bodyid[i];
      mjtNum mass = m->body_mass[bi];
      if (!mass) {  // SHOULD NOT OCCUR
        mjERROR("moving body %d has 0 mass", bi);
      }
      m->dof_invweight0[id] = 1/mju_max(mjMINVAL, mass);
    }

    // general joint: full inertia
    else {
      int dnum, id = m->jnt_dofadr[i];

      // get number of components
      if (m->jnt_type[i] == mjJNT_FREE) {
        dnum = 6;
      } else if (m->jnt_type[i] == mjJNT_BALL) {
        dnum = 3;
      } else {
        dnum = 1;
      }

      // inverse joint inertia:  A = J*inv(M)*J'
      if (nv) {
        mju_zero(jac, dnum*nv);
        for (int j=0; j < dnum; j++) {
          jac[j*(nv+1) + id] = 1;
        }
        mj_solveM(m, d, tmp, jac, dnum);
        mju_mulMatMatT(A, jac, tmp, dnum, nv, dnum);
      }

      // average diagonal and assign
      if (dnum == 6) {
        m->dof_invweight0[id] = m->dof_invweight0[id+1] = m->dof_invweight0[id+2] =
          (A[0] + A[7] + A[14])/3;
        m->dof_invweight0[id+3] = m->dof_invweight0[id+4] = m->dof_invweight0[id+5] =
          (A[21] + A[28] + A[35])/3;
      } else if (dnum == 3) {
        m->dof_invweight0[id] = m->dof_invweight0[id+1] = m->dof_invweight0[id+2] =
          (A[0] + A[4] + A[8])/3;
      } else {
        m->dof_invweight0[id] = A[0];
      }
    }
  }

  // compute flexedge_invweight0, tendon_invweight0, actuator_acc0
  mju_zero(m->flexedge_invweight0, m->nflexedge);
  if (nv) {
    // compute flexedge_invweight0
    for (int f=0; f < m->nflex; f++) {
      if (m->flex_interp[f]) {
        continue;
      }

      for (int i=m->flex_edgeadr[f]; i < m->flex_edgeadr[f]+m->flex_edgenum[f]; i++) {
        // bodies connected by edge
        int b1 = m->flex_vertbodyid[m->flex_vertadr[f] + m->flex_edge[2*i]];
        int b2 = m->flex_vertbodyid[m->flex_vertadr[f] + m->flex_edge[2*i+1]];

        // rigid edge: set to 0
        if (m->flexedge_rigid[i]) {
          m->flexedge_invweight0[i] = 0;
        }

        // accelerate edges that connect simple bodies with no rotations
        else if (m->body_simple[b1] == 2 && m->body_simple[b2] == 2) {
          m->flexedge_invweight0[i] = (1/m->body_mass[b1] + 1/m->body_mass[b2])/2;
        }

        // handle general edge
        else {
          // make dense vector into tmp
          mju_zero(tmp, nv);
          int end = m->flexedge_J_rowadr[i] + m->flexedge_J_rownnz[i];
          for (int j=m->flexedge_J_rowadr[i]; j < end; j++) {
            tmp[m->flexedge_J_colind[j]] = d->flexedge_J[j];
          }

          // solve into tmp+nv
          mj_solveM(m, d, tmp+nv, tmp, 1);
          m->flexedge_invweight0[i] = mju_dot(tmp, tmp+nv, nv);
        }
      }
    }

    // compute tendon_invweight0
    for (int i=0; i < m->ntendon; i++) {
      mju_sparse2dense(tmp, d->ten_J, 1, nv, m->ten_J_rownnz+i, m->ten_J_rowadr+i, m->ten_J_colind);

      // solve into tmp+nv
      mj_solveM(m, d, tmp+nv, tmp, 1);
      m->tendon_invweight0[i] = mju_dot(tmp, tmp+nv, nv);
    }

    // compute actuator_acc0, one per force output (moment row)
    for (int i=0; i < m->nout; i++) {
      mju_sparse2dense(moment, d->actuator_moment, 1, nv, d->moment_rownnz + i,
                       d->moment_rowadr + i, d->moment_colind);
      mj_solveM(m, d, tmp, moment, 1);
      m->actuator_acc0[i] = mju_norm(tmp, nv);
    }
  } else {
    mju_zero(m->tendon_invweight0, m->ntendon);
    mju_zero(m->actuator_acc0, m->nout);
  }

  mj_freeStack(d);
}


// accumulate bounding box
static void updateBox(mjtNum* xmin, mjtNum* xmax, mjtNum* pos, mjtNum radius) {
  for (int i=0; i < 3; i++) {
    xmin[i] = mjMIN(xmin[i], pos[i] - radius);
    xmax[i] = mjMAX(xmax[i], pos[i] + radius);
  }
}


// approximate length associated with each body: the largest distance from its center of mass to a
// joint of it or of a child, to the far side of a geom, or along a flex edge; at least 1E-5 for all
// bodies but the world; assume computations already executed in qpos0
static void bodySize(const mjModel* m, const mjData* d, mjtNum* body) {
  // set body size to max com-joint distance
  mju_zero(body, m->nbody);
  for (int i=0; i < m->njnt; i++) {
    // handle this body
    int id = m->jnt_bodyid[i];
    body[id] = mju_max(body[id], mju_dist3(d->xipos+3*id, d->xanchor+3*i));

    // handle parent body
    id = m->body_parentid[id];
    body[id] = mju_max(body[id], mju_dist3(d->xipos+3*id, d->xanchor+3*i));
  }
  body[0] = 0;

  // set body size to max of old value, and geom rbound + com-geom dist
  for (int i=1; i < m->nbody; i++) {
    for (int id=m->body_geomadr[i]; id < m->body_geomadr[i]+m->body_geomnum[i]; id++) {
      if (m->geom_rbound[id] > 0) {
        body[i] = mju_max(body[i], m->geom_rbound[id] + mju_dist3(d->xipos+3*i, d->geom_xpos+3*id));
      }
    }
  }

  // adjust body size for flex edges involving body
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_interp[f]) {
      for (int v1=m->flex_nodeadr[f]; v1 < m->flex_nodeadr[f]+m->flex_nodenum[f]; v1++) {
        for (int v2=m->flex_nodeadr[f]; v2 < m->flex_nodeadr[f]+m->flex_nodenum[f]; v2++) {
          mjtNum edge = mju_dist3(d->xpos+3*m->flex_nodebodyid[v1],
                                  d->xpos+3*m->flex_nodebodyid[v2]);
          body[m->flex_nodebodyid[v1]] = mju_max(body[m->flex_nodebodyid[v1]], edge);
        }
      }
      continue;
    }
    for (int e=m->flex_edgeadr[f]; e < m->flex_edgeadr[f]+m->flex_edgenum[f]; e++) {
      int b1 = m->flex_vertbodyid[m->flex_vertadr[f]+m->flex_edge[2*e]];
      int b2 = m->flex_vertbodyid[m->flex_vertadr[f]+m->flex_edge[2*e+1]];

      body[b1] = mju_max(body[b1], m->flexedge_length0[e]);
      body[b2] = mju_max(body[b2], m->flexedge_length0[e]);
    }
  }

  // make sure all sizes are above min
  for (int i=1; i < m->nbody; i++) {
    body[i] = mju_max(body[i], 1E-5);
  }
}


// set dof_length: 1 for translations, the size of the body for rotations; assume computations
// already executed in qpos0
static void setDofLength(mjModel* m, mjData* d) {
  mj_markStack(d);
  mjtNum* body = mjSTACKALLOC(d, m->nbody, mjtNum);
  bodySize(m, d, body);

  // inherit dof length from parent body
  for (int i=0; i < m->nv; i++) {
    // default to linear dof, already has length units
    m->dof_length[i] = 1;

    // if rotational dof, inherit from body
    int jnt = m->dof_jntid[i];
    mjtJoint type = m->jnt_type[jnt];
    int offset = i - m->jnt_dofadr[jnt];
    if (type == mjJNT_BALL  ||
        type == mjJNT_HINGE ||
        (type == mjJNT_FREE && offset >= 3)) {
      m->dof_length[i] = body[m->dof_bodyid[i]];
    }
  }

  mj_freeStack(d);
}


// run computations in qpos_spring, whose frames setEfm0Factor uses
static void setSpring(mjModel* m, mjData* d) {
  mju_copy(d->qpos, m->qpos_spring, m->nq);
  mj_kinematics(m, d);
  mj_comPos(m, d);
  mj_tendon(m, d);
  mj_transmission(m, d);
}


// constant part of the implicit effective metric factor (currently dim-2 bending): sparse
// reverse-Cholesky of M + (h^2 + h*damping)*K_bend over
// the dofs of unpinned vertices of standard dim-2 flexes with bending. The matrix is constant
// (flat-rest bending stiffness, independent point masses), so it is computed here once and reused
// by the implicit-flex constraint solve every step. Bending couples full 3x3 blocks when vertex
// frames differ. Row order (body order, coordinate fastest) and the fill count must match the
// compiler's symbolic sizing (checked below).
static void setEfm0Factor(mjModel* m, mjData* d) {
  int nbd = m->nefm0dof;
  if (!nbd) {
    return;
  }
  mj_markStack(d);
  mjtNum h = m->opt.timestep;

  // enumerate covered vertices: compact slot per unpinned vertex of qualifying flexes
  // (filter matches the compiler's sizing and, for bending, mjd_flexStiff_active in
  // engine_derivative.c: constant bending stencils exist only for dim-2 flexes; 1D rod
  // bending is state-dependent and assembled per step)
  int* bodyslot = mjSTACKALLOC(d, m->nbody, int);
  for (int b=0; b < m->nbody; b++) {
    bodyslot[b] = -1;
  }
  for (int f=0; f < m->nflex; f++) {
    if (m->flex_interp[f] || m->flex_rigid[f] || m->flex_dim[f] != 2 ||
        m->flex_bendingadr[f] < 0 || !mj_flexSimple(m, f)) {
      continue;
    }
    for (int lv=0; lv < m->flex_vertnum[f]; lv++) {
      int gv = m->flex_vertadr[f] + lv;
      int wid = m->body_weldid[m->flex_vertbodyid[gv]];
      if (m->body_dofnum[wid] == 3) {
        bodyslot[wid] = 1;
      }
    }
  }
  int nfree = 0;
  for (int b=0; b < m->nbody; b++) {
    if (bodyslot[b] > 0) {
      int s = nfree++;
      bodyslot[b] = s;
      int da = m->body_dofadr[b];
      for (int k=0; k < 3; k++) {
        m->efm0_dofid[3*s + k] = da + k;
      }
    }
  }
  if (3*nfree != nbd) {
    mj_freeStack(d);
    mjERROR("constant metric factor dof count mismatch: compiler sized %d, engine found %d",
            nbd, 3*nfree);
    return;
  }

  // assemble the bending-only stiffness K = (h^2 + h*damping)*K_bend over all dofs with the
  // shared stencil walker from engine_derivative. Only fixed-frame attachments participate;
  // their orientations were computed by setSpring. As in the per-step metric, the h^2 and
  // h*damping parts enter only when the spring and damper forces are enabled.
  mjtNum s1 = mjDISABLED(mjDSBL_SPRING) ? 0 : h*h;
  mjtNum s2 = mjDISABLED(mjDSBL_DAMPER) ? 0 : h;
  int nv = m->nv;
  int* K_rownnz = mjSTACKALLOC(d, nv, int);
  int* K_rowadr = mjSTACKALLOC(d, nv, int);
  int nK = mjd_flexStiff_assemble(m, d, K_rownnz, K_rowadr, NULL, NULL, s1, s2,
                                  /*flg_bend=*/1, /*flg_stretch=*/0, NULL);
  int* K_colind = mjSTACKALLOC(d, nK > 0 ? nK : 1, int);
  mjtNum* K_val = mjSTACKALLOC(d, nK > 0 ? nK : 1, mjtNum);
  mjd_flexStiff_assemble(m, d, K_rownnz, K_rowadr, K_colind, K_val, s1, s2, 1, 0, NULL);

  // inverse map: dof address -> compact factor row (monotone: slots follow dof order)
  int* dofrow = mjSTACKALLOC(d, nv, int);
  for (int i=0; i < nv; i++) {
    dofrow[i] = -1;
  }
  for (int r=0; r < nbd; r++) {
    dofrow[m->efm0_dofid[r]] = r;
  }

  // Compact K_bend to covered rows, retaining full off-diagonal blocks Q_ij * R_i^T * R_j.
  // Only same-body blocks are diagonal, even when several vertices share a weld parent.
  // Keep structural zeros between bodies so the pattern is independent of their orientations.
  // H = M + (h^2+h*d)*K_bend in compact dof indices: lower CSR (values) + upper CSR (pattern)
  int nHl = 0, nHu = 0;
  for (int r=0; r < nbd; r++) {
    int dof = m->efm0_dofid[r];
    for (int c=0; c < K_rownnz[dof]; c++) {
      int rc = dofrow[K_colind[K_rowadr[dof] + c]];
      if (rc < 0 || (rc != r && rc/3 == r/3)) continue;
      if (rc < r) nHl++;
      else if (rc > r) nHu++;
    }
  }
  nHl += nbd;  // diagonals: always present, also for rows without bending entries
  int* Hl_rownnz = mjSTACKALLOC(d, nbd, int);
  int* Hl_rowadr = mjSTACKALLOC(d, nbd, int);
  int* Hl_colind = mjSTACKALLOC(d, nHl, int);
  mjtNum* Hl_val  = mjSTACKALLOC(d, nHl, mjtNum);
  int* Hu_rownnz = mjSTACKALLOC(d, nbd, int);
  int* Hu_rowadr = mjSTACKALLOC(d, nbd, int);
  int* Hu_colind = mjSTACKALLOC(d, nHu > 0 ? nHu : 1, int);

  int ladr = 0, uadr = 0;
  for (int r=0; r < nbd; r++) {
    int dof = m->efm0_dofid[r];
    Hl_rowadr[r] = ladr;
    Hu_rowadr[r] = uadr;
    mjtNum diag = 0;
    // B columns are dof-ascending, so filtered lower/upper columns stay ascending
    for (int c=0; c < K_rownnz[dof]; c++) {
      int adr = K_rowadr[dof] + c;
      int rc = dofrow[K_colind[adr]];
      if (rc < 0 || (rc != r && rc/3 == r/3)) continue;
      if (rc < r) {
        Hl_colind[ladr] = rc;
        Hl_val[ladr++] = K_val[adr];
      } else if (rc > r) {
        Hu_colind[uadr++] = rc;
      } else {
        diag = K_val[adr];
      }
    }
    // diagonal last: composite point mass + armature + bending diagonal
    Hl_colind[ladr] = r;
    Hl_val[ladr++] = m->dof_M0[dof] + diag;
    Hl_rownnz[r] = ladr - Hl_rowadr[r];
    Hu_rownnz[r] = uadr - Hu_rowadr[r];
  }

  // symbolic factorization: counting phase (from the upper-triangle pattern)
  int* LT_rownnz = mjSTACKALLOC(d, nbd, int);
  int* LT_rowadr = mjSTACKALLOC(d, nbd, int);
  int nnz = mju_cholFactorSymbolic(NULL, m->efm0_L_rownnz, m->efm0_L_rowadr,
                                   NULL, LT_rownnz, LT_rowadr, NULL,
                                   Hu_rownnz, Hu_rowadr, Hu_colind, nbd, d);
  if (nnz != m->nefm0L) {
    mj_freeStack(d);
    mjERROR("constant metric factor size mismatch: compiler sized %d, symbolic found %d",
            (int)m->nefm0L, nnz);
    return;
  }

  // symbolic factorization: filling phase
  int* LT_colind = mjSTACKALLOC(d, nnz, int);
  int* LT_map = mjSTACKALLOC(d, nnz, int);
  mju_cholFactorSymbolic(m->efm0_L_colind, m->efm0_L_rownnz, m->efm0_L_rowadr,
                         LT_colind, LT_rownnz, LT_rowadr, LT_map,
                         Hu_rownnz, Hu_rowadr, Hu_colind, nbd, d);

  // numeric factorization
  mjtNum* scratch = mjSTACKALLOC(d, nbd, mjtNum);
  int rank = mju_cholFactorNumeric(m->efm0_L, nbd, mjMINVAL,
                                   m->efm0_L_rownnz, m->efm0_L_rowadr, m->efm0_L_colind,
                                   LT_rownnz, LT_rowadr, LT_colind, LT_map,
                                   Hl_val, Hl_rownnz, Hl_rowadr, Hl_colind, scratch);
  if (rank != nbd) {
    // The factor is consumed only when no per-step flex stiffness is assembled (bending-only
    // models): with stretch (or assemblable interp) present, mj_effBuild assembles the full
    // metric every step and the per-step blocks replace this factor, so a singular
    // M + K_bend (e.g. vanishing vertex masses) is harmless there.
    if (mjd_flexStiff_any(m, mjd_flexInterpAssemblable(m))) {
      mj_freeStack(d);
      return;
    }
    mj_freeStack(d);
    mjERROR("constant metric factor is rank-deficient (%d of %d)", rank, nbd);
    return;
  }

  mj_freeStack(d);
}


//----------------------------- constants set once by compilation ----------------------------------

// compute the model statistics; assume computations already executed in qpos0
static void setStat(mjModel* m, mjData* d) {
  mjtNum xmin[3] = {1E+10, 1E+10, 1E+10};
  mjtNum xmax[3] = {-1E+10, -1E+10, -1E+10};
  mjtNum rbound;
  mj_markStack(d);

  // compute bounding box of bodies, joint centers, geoms and sites
  for (int i=1; i < m->nbody; i++) {
    updateBox(xmin, xmax, d->xpos+3*i, 0);
    updateBox(xmin, xmax, d->xipos+3*i, 0);
  }
  for (int i=0; i < m->njnt; i++) {
    updateBox(xmin, xmax, d->xanchor+3*i, 0);
  }
  for (int i=0; i < m->nsite; i++) {
    updateBox(xmin, xmax, d->site_xpos+3*i, 0);
  }
  for (int i=0; i < m->ngeom; i++) {
    // set rbound: regular geom rbound, or 0.1 of plane or hfield max size
    rbound = 0;
    if (m->geom_rbound[i] > 0) {
      rbound = m->geom_rbound[i];
    } else if (m->geom_type[i] == mjGEOM_PLANE) {
      // finite in at least one direction
      if (m->geom_size[3*i] || m->geom_size[3*i+1]) {
        rbound = mjMAX(m->geom_size[3*i], m->geom_size[3*i+1]) * 0.1;
      }

      // infinite in both directions
      else {
        rbound = 0.01;
      }
    } else if (m->geom_type[i] == mjGEOM_HFIELD) {
      int j = m->geom_dataid[i];
      rbound = mjMAX(m->hfield_size[4*j],
                     mjMAX(m->hfield_size[4*j+1],
                           mjMAX(m->hfield_size[4*j+2], m->hfield_size[4*j+3]))) * 0.1;
    }

    updateBox(xmin, xmax, d->geom_xpos+3*i, rbound);
  }

  // compute center
  mju_add3(m->stat.center, xmin, xmax);
  mju_scl3(m->stat.center, m->stat.center, 0.5);

  // compute bounding box size
  if (xmax[0] > xmin[0])
    m->stat.extent = mju_max(1E-5,
                             mju_max(xmax[0]-xmin[0], mju_max(xmax[1]-xmin[1], xmax[2]-xmin[2])));

  // compute meansize
  if (m->nbody > 1) {
    mjtNum* body = mjSTACKALLOC(d, m->nbody, mjtNum);
    bodySize(m, d, body);
    m->stat.meansize = 0;
    for (int i=1; i < m->nbody; i++) {
      m->stat.meansize += body[i]/(m->nbody-1);
    }
  }

  // fix extent if too small compared to meanbody
  m->stat.extent = mju_max(m->stat.extent, 2 * m->stat.meansize);

  // compute meanmass
  if (m->nbody > 1) {
    m->stat.meanmass = 0;
    for (int i=1; i < m->nbody; i++) {
      m->stat.meanmass += m->body_mass[i];
    }
    m->stat.meanmass /= (m->nbody-1);
  }

  // compute meaninertia
  if (m->nv) {
    m->stat.meaninertia = 0;
    for (int i=0; i < m->nv; i++) {
      m->stat.meaninertia += d->M[m->M_rowadr[i] + m->M_rownnz[i] - 1];
    }
    m->stat.meaninertia /= m->nv;
  }

  mj_freeStack(d);
}


// compute the damping of position-like actuators which have a damping ratio, given as a positive
// biasprm[2]; assume computations already executed in qpos0
static void setDampRatio(mjModel* m, mjData* d) {
  int nv = m->nv;
  mj_markStack(d);
  mjtNum* tmp = mjSTACKALLOC(d, nv, mjtNum);
  mjtNum* moment = mjSTACKALLOC(d, nv, mjtNum);

  for (int i=0; i < m->nactuator; i++) {
    // get bias, gain parameters
    mjtNum* biasprm = m->actuator_biasprm + i*mjNBIAS;
    mjtNum* gainprm = m->actuator_gainprm + i*mjNGAIN;

    // not a position-like actuator: skip (PID single-sources kp in biasprm[1])
    int is_pid = m->actuator_gaintype[i] == mjGAIN_PID;
    if (!is_pid && gainprm[0] != -biasprm[1]) {
      continue;
    }

    // damping is 0 or negative (interpreted as regular "kv"): skip
    if (biasprm[2] <= 0) {
      continue;
    }

    // "reflected" inertia: 1 / (J * inv(M) * J'), averaged over force outputs
    mjtNum mass = 0;
    if (nv) {
      int outadr = m->actuator_outadr[i];
      int outnum = m->actuator_outnum[i];
      mjtNum w = 0;
      for (int k=0; k < outnum; k++) {
        int j = outadr + k;
        mju_sparse2dense(moment, d->actuator_moment, 1, nv, d->moment_rownnz + j,
                         d->moment_rowadr + j, d->moment_colind);
        mj_solveM(m, d, tmp, moment, 1);
        w += mju_dot(moment, tmp, nv);
      }
      w /= outnum;
      if (w > mjMINVAL) {
        mass = 1 / w;
      }
    }

    // damping = dampratio * 2 * sqrt(kp * mass)
    mjtNum kp = is_pid ? -biasprm[1] : gainprm[0];
    mjtNum damping = biasprm[2] * 2 * mju_sqrt(kp * mass);

    // set biasprm[2] to negative damping
    biasprm[2] = -damping;
  }

  mj_freeStack(d);
}


// set the spring length range of tendons which have none, given as (-1, -1), to their length;
// assume computations already executed in qpos_spring
static void setSpringLength(mjModel* m, mjData* d) {
  for (int i=0; i < m->ntendon; i++) {
    if (m->tendon_lengthspring[2*i] == -1 && m->tendon_lengthspring[2*i+1] == -1) {
      m->tendon_lengthspring[2*i] = m->tendon_lengthspring[2*i+1] = d->ten_length[i];
    }
  }
}


//----------------------------- entry points -------------------------------------------------------

// run the groups of mj_setConst computations selected by the bits of groups, in order
void mj_setConstGroups(mjModel* m, mjData* d, int groups) {
  // quantities independent of the configuration
  if (groups & mjSCGROUP_FIXED) {
    // compute npolygonmax and nmeshdegmax
    m->npolygonmax = 0;
    for (int i=0; i < m->nmeshpoly; i++) {
      m->npolygonmax = mjMAX(m->npolygonmax, m->mesh_polyvertnum[i]);
    }

    m->nmeshdegmax = 0;
    for (int i=0; i < m->nmeshvert; i++) {
      m->nmeshdegmax = mjMAX(m->nmeshdegmax, m->mesh_polymapnum[i]);
    }

    // recompute sameframe flags from current model geometry
    setSameframe(m);

    // error if simple body lost sameframe (user must set simple="false")
    for (int i = 1; i < m->nbody; i++) {
      if (m->body_simple[i] > 0 && m->body_sameframe[i] != mjSAMEFRAME_BODY) {
        mjERROR("body %d is compiled as simple but sameframe no longer holds, "
                "use body/simple='false'", i);
      }
    }

    // set geom and body bounds
    setBound(m, d);

    // set fluid added mass
    setFluid(m);

    // set fixed quantities
    setFixed(m, d);

    // set Jacobian sparsity of tendons and flexes
    makeTendonSparse(m);
    makeFlexSparse(m, d);
  }

  // computations below cover all bodies, sleeping or not (not restored if an error is raised)
  int enableflags = m->opt.enableflags;
  m->opt.enableflags &= ~mjENBL_SLEEP;

  // the constants do not depend on the contacts of d, which mj_transmission averages for adhesion
  // actuators: clear them and the constraints, as collision detection does
  if (groups & (mjSCGROUP_KIN0 | mjSCGROUP_INERT0 | mjSCGROUP_SPRING)) {
    d->ncon = 0;
    mj_clearEfc(d);
  }

  // set mocap bodies to their model pose (restored after setSpring)
  int mocap = m->nmocap && (groups & (mjSCGROUP_KIN0 | mjSCGROUP_INERT0 | mjSCGROUP_SPRING));
  mjtNum* mocap_pos = NULL;
  mjtNum* mocap_quat = NULL;
  if (mocap) {
    mj_markStack(d);
    mocap_pos = mjSTACKALLOC(d, 3*m->nmocap, mjtNum);
    mocap_quat = mjSTACKALLOC(d, 4*m->nmocap, mjtNum);
    mju_copy(mocap_pos, d->mocap_pos, 3*m->nmocap);
    mju_copy(mocap_quat, d->mocap_quat, 4*m->nmocap);
    for (int i=0; i < m->nbody; i++) {
      int id = m->body_mocapid[i];
      if (id >= 0) {
        mju_copy3(d->mocap_pos+3*id, m->body_pos+3*i);
        mju_copy4(d->mocap_quat+4*id, m->body_quat+4*i);
      }
    }
  }

  // set quantities that depend on qpos0
  if (groups & (mjSCGROUP_KIN0 | mjSCGROUP_INERT0)) {
    kinematics0(m, d);
    if (groups & mjSCGROUP_KIN0) {
      setKinematics0(m, d);
      setDofLength(m, d);
    }
    if (groups & mjSCGROUP_INERT0) {
      setInertia0(m, d);
    }
  }

  // set quantities that depend on qpos_spring
  if (groups & mjSCGROUP_SPRING) {
    setSpring(m, d);
  }

  if (mocap) {
    mju_copy(d->mocap_pos, mocap_pos, 3*m->nmocap);
    mju_copy(d->mocap_quat, mocap_quat, 4*m->nmocap);
    mj_freeStack(d);
  }

  if (groups & mjSCGROUP_SPRING) {
    // precompute the constant part of the implicit effective metric factor
    setEfm0Factor(m, d);

    // the vertex ordering of the metric's sparse factor
    mj_effCholSetConst(m, d);
  }

  m->opt.enableflags = enableflags;

  // the kinematics above overwrote the cached state of sleeping trees: wake all trees
  if (groups & (mjSCGROUP_KIN0 | mjSCGROUP_INERT0 | mjSCGROUP_SPRING)) {
    mju_fillInt(d->tree_asleep, -(1+mjMINAWAKE), m->ntree);
    mj_updateSleep(m, d);
  }
}


// set all remaining constant fields of mjModel, except for lengthrange
static void setConst(mjModel* m, mjData* d, int compiled) {
  mj_setConstGroups(m, d, mjSCGROUP_FIXED | mjSCGROUP_KIN0 | mjSCGROUP_INERT0);

  // in compilation, the other qpos0 quantities
  if (compiled) {
    setDampRatio(m, d);
    setStat(m, d);
  }

  mj_setConstGroups(m, d, mjSCGROUP_SPRING);

  // in compilation, the other qpos_spring quantities
  if (compiled) {
    setSpringLength(m, d);
  }
}


void mj_setConst(mjModel* m, mjData* d) {
  setConst(m, d, 0);
}


void mj_setCompiledConst(mjModel* m, mjData* d) {
  setConst(m, d, 1);
}


//----------------------------- actuator length range computation ----------------------------------

// evaluate actuator length, advance special dynamics
static mjtNum evalAct(const mjModel* m, mjData* d, int index, int side,
                      const mjLROpt* opt) {
  int nv = m->nv;
  int out = m->actuator_outadr[index];

  // reduce velocity
  mju_scl(d->qvel, d->qvel, mju_exp(-m->opt.timestep/mjMAX(0.01, opt->timeconst)), nv);

  // step1: compute inertia and actuator moments
  mj_step1(m, d);

  // dense actuator_moment row
  mj_markStack(d);
  mjtNum* moment = mjSTACKALLOC(d, nv, mjtNum);
  mju_sparse2dense(moment, d->actuator_moment, 1, nv, d->moment_rownnz + out,
                   d->moment_rowadr + out, d->moment_colind);

  // set force to generate desired acceleration
  mj_solveM(m, d, d->qfrc_applied, moment, 1);
  mjtNum nrm = mju_norm(d->qfrc_applied, nv);
  mju_scl(d->qfrc_applied, moment, (2*side-1)*opt->accel/mjMAX(mjMINVAL, nrm), nv);

  // impose maxforce
  nrm = mju_norm(d->qfrc_applied, nv);
  if (opt->maxforce > 0 && nrm > opt->maxforce) {
    mju_scl(d->qfrc_applied, d->qfrc_applied, opt->maxforce/mjMAX(mjMINVAL, nrm), nv);
  }

  // step2: apply force
  mj_step2(m, d);

  mj_freeStack(d);

  // return actuator length
  return d->actuator_length[out];
}


// Set length range for specified actuator, return 1 if ok, 0 if error.
int mj_setLengthRange(mjModel* m, mjData* d, int index,
                      const mjLROpt* opt, char* error, int error_sz) {
  // check index
  if (index < 0 || index >= m->nactuator) {
    mjERROR("invalid actuator index");
  }

  // first output row (remaining rows of multi-output actuators keep compiled values)
  int out = m->actuator_outadr[index];

  // skip depending on mode and type
  int ismuscle = (m->actuator_gaintype[index] == mjGAIN_MUSCLE ||
                  m->actuator_biastype[index] == mjBIAS_MUSCLE);
  int isuser = (m->actuator_gaintype[index] == mjGAIN_USER ||
                m->actuator_biastype[index] == mjBIAS_USER);
  if ((opt->mode == mjLRMODE_NONE) ||
      (opt->mode == mjLRMODE_MUSCLE && !ismuscle) ||
      (opt->mode == mjLRMODE_MUSCLEUSER && !ismuscle && !isuser)) {
    return 1;
  }

  // use existing length range if available
  if (opt->useexisting && (m->actuator_lengthrange[2*out] < m->actuator_lengthrange[2*out+1])) {
    return 1;
  }

  // get transmission id
  int threadid = m->actuator_trnid[2*index];

  // use joint and tendon limits if available
  if (opt->uselimit) {
    // joint or jointinparent
    if (m->actuator_trntype[index] == mjTRN_JOINT ||
        m->actuator_trntype[index] == mjTRN_JOINTINPARENT) {
      // make sure joint is limited
      if (m->jnt_limited[threadid]) {
        // copy range
        m->actuator_lengthrange[2*out] = m->jnt_range[2*threadid];
        m->actuator_lengthrange[2*out+1] = m->jnt_range[2*threadid+1];

        // skip optimization
        return 1;
      }
    }

    // tendon
    if (m->actuator_trntype[index] == mjTRN_TENDON) {
      // make sure tendon is limited
      if (m->tendon_limited[threadid]) {
        // copy range
        m->actuator_lengthrange[2*out] = m->tendon_range[2*threadid];
        m->actuator_lengthrange[2*out+1] = m->tendon_range[2*threadid+1];

        // skip optimization
        return 1;
      }
    }
  }

  // optimize in both directions
  mjtNum lmin[2] = {0, 0}, lmax[2] = {0, 0};
  int side;
  for (side=0; side < 2; side++) {
    // init at qpos0
    mj_resetData(m, d);

    // simulate
    int updated = 0;
    while (d->time < opt->inttotal) {
      // advance and get length
      mjtNum len = evalAct(m, d, index, side, opt);

      // reset: cannot proceed
      if (d->time == 0) {
        snprintf(error, error_sz, "Unstable lengthrange simulation in actuator %d", index);
        return 0;
      }

      // update limits
      if (d->time > opt->inttotal-opt->interval) {
        if (len < lmin[side] || !updated) {
          lmin[side] = len;
        }
        if (len > lmax[side] || !updated) {
          lmax[side] = len;
        }

        updated = 1;
      }
    }

    // assign
    m->actuator_lengthrange[2*out+side] = (side == 0 ? lmin[side] : lmax[side]);
  }

  // check range
  mjtNum dif = m->actuator_lengthrange[2*out+1] - m->actuator_lengthrange[2*out];
  if (dif <= 0) {
    snprintf(error, error_sz,
             "Invalid lengthrange (%g, %g) in actuator %d",
             m->actuator_lengthrange[2*out],
             m->actuator_lengthrange[2*out+1], index);
    return 0;
  }

  // check convergence, side 0
  if (lmax[0]-lmin[0] > opt->tolrange*dif) {
    snprintf(error, error_sz,
             "Lengthrange computation did not converge in actuator %d:\n"
             "  eval (%g, %g)\n  range (%g, %g)",
             index, lmin[0], lmax[0],
             m->actuator_lengthrange[2*out],
             m->actuator_lengthrange[2*out+1]);
    return 0;
  }

  // check convergence, side 1
  if (lmax[1]-lmin[1] > opt->tolrange*dif) {
    snprintf(error, error_sz,
             "Lengthrange computation did not converge in actuator %d:\n"
             "  eval (%g, %g)\n range (%g, %g)",
             index, lmin[1], lmax[1],
             m->actuator_lengthrange[2*out],
             m->actuator_lengthrange[2*out+1]);
    return 0;
  }

  return 1;
}
