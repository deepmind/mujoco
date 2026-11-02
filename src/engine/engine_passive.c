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

#include "engine/engine_passive.h"

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_callback.h"
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_inline.h"
#include "engine/engine_memory.h"
#include "engine/engine_metric.h"
#include "engine/engine_plugin.h"
#include "engine/engine_sleep.h"
#include "engine/engine_support.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"


//----------------------------- passive forces -----------------------------------------------------


// passive forces for interpolated flex (stretch + bending)
static void mj_flexPassiveInterp(const mjModel* m, mjData* d, int f,
                                 int enbl_spring, int enbl_damper) {
  // check if we have any work to do (element 0 may be an empty cell with zero stiffness)
  if (m->flex_edgeequality[f] == 3 || !mj_flexHasStiffness(m, f)) {
    return;
  }

  mjtNum* k = m->flex_stiffness + m->flex_stiffnessadr[f];
  int nodenum = m->flex_nodenum[f];

  int order = m->flex_interp[f];
  int shell_mode = order < 0;
  order = order < 0 ? -order : order;
  int cx = m->flex_cellnum[3*f+0];
  int cy = m->flex_cellnum[3*f+1];
  int cz = m->flex_cellnum[3*f+2];

  // determine element type: 2D boundary quads (shell) or 3D cells (volume)
  int npe;       // nodes per element
  int nelem_fe;  // total finite elements

  if (shell_mode) {
    npe = (order+1)*(order+1);
    nelem_fe = 2*(cy*cz + cx*cz + cx*cy);
  } else {
    npe = (order+1)*(order+1)*(order+1);
    nelem_fe = cx * cy * cz;
  }

  mj_markStack(d);

  // allocate global arrays
  mjtNum* xpos_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* vel_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* frc_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* dmp_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* xpos0 = m->flex_node0 + 3*m->flex_nodeadr[f];
  int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];

  // gather global node positions and velocities (unrotated)
  mju_flexGatherState(m, d, f, xpos_g, vel_g);

  // zero global force accumulators
  mju_zero(frc_g, 3*nodenum);
  mju_zero(dmp_g, 3*nodenum);

  // per-element arrays (sized for npe)
  mjtNum* xpos_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* vel_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* xpos0_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* displ_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* frc_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* dmp_e = mjSTACKALLOC(d, 3*npe, mjtNum);
  int* gindices = mjSTACKALLOC(d, npe, int);

  // -------------------- stretch forces --------------------
  for (int fe = 0; fe < nelem_fe; fe++) {
    // get element stiffness matrix
    mjtNum* k_elem = k + fe * 3*npe * 3*npe;

    // skip empty elements (zero stiffness) in volume mode; shells have no empty cells and
    // their x-normal faces (faces 0 and 1) have k_elem[0] == 0
    if (!shell_mode && k_elem[0] == 0) {
      continue;
    }

    // gather element-local node data and compute corotational rotation
    mjtNum quat[4];
    if (shell_mode) {
      mju_flexGatherFaceState(order, cx, cy, cz, fe, xpos_g, vel_g, xpos0,
                              xpos_e, vel_e, xpos0_e, gindices, quat);
    } else {
      int ci = fe / (cy * cz);
      int cj = (fe / cz) % cy;
      int ck = fe % cz;
      mju_flexGatherCellState(order, cy, cz, ci, cj, ck, xpos_g, vel_g,
                              xpos0, xpos_e, vel_e, xpos0_e, gindices,
                              quat);
    }

    // rotate to corotational frame
    for (int n = 0; n < npe; n++) {
      mju_rotVecQuat(xpos_e+3*n, xpos_e+3*n, quat);
      mju_rotVecQuat(vel_e+3*n, vel_e+3*n, quat);
    }

    // compute displacement
    for (int n = 0; n < npe; n++) {
      mji_addScl3(displ_e+3*n, xpos_e+3*n, xpos0_e+3*n, -1);
    }

    // subtract cell-center plastic displacement in corotational frame
    if (m->flex_plasticadr[f] >= 0) {
      if (shell_mode) {
        int normal_axis = (fe < 2*cy*cz) ? 0 : (fe < 2*(cy*cz + cx*cz)) ? 1 : 2;
        int axis0 = (normal_axis + 1) % 3;
        int axis1 = (normal_axis + 2) % 3;
        const mjtNum* sp = d->plastic + m->flex_plasticadr[f] + 3 * fe;
        for (int n = 0; n < npe; n++) {
          mjtNum dx0_0 = xpos0_e[3*n + axis0] - xpos0_e[axis0];
          mjtNum dx0_1 = xpos0_e[3*n + axis1] - xpos0_e[axis1];
          displ_e[3*n + axis0] -= sp[0]*dx0_0 + 0.5*sp[2]*dx0_1;
          displ_e[3*n + axis1] -= 0.5*sp[2]*dx0_0 + sp[1]*dx0_1;
        }
      } else {
        const mjtNum* sp = d->plastic + m->flex_plasticadr[f] + 6 * fe;
        for (int n = 0; n < npe; n++) {
          mjtNum dx0[3];
          mji_sub3(dx0, xpos0_e + 3*n, xpos0_e);
          displ_e[3*n + 0] -= sp[0]*dx0[0] + 0.5*sp[3]*dx0[1] + 0.5*sp[4]*dx0[2];
          displ_e[3*n + 1] -= 0.5*sp[3]*dx0[0] + sp[1]*dx0[1] + 0.5*sp[5]*dx0[2];
          displ_e[3*n + 2] -= 0.5*sp[4]*dx0[0] + 0.5*sp[5]*dx0[1] + sp[2]*dx0[2];
        }
      }
    }

    // compute force in corotational frame
    if (enbl_spring) {
      mju_mulMatVec(frc_e, k_elem, displ_e, 3*npe, 3*npe);
    }
    if (enbl_damper) {
      mju_mulMatVec(dmp_e, k_elem, vel_e, 3*npe, 3*npe);
    }

    // rotate back to global frame and scatter using node indices
    mju_negQuat(quat, quat);
    for (int n = 0; n < npe; n++) {
      mjtNum qfrc[3], qdmp[3];
      mji_rotVecQuat(qfrc, frc_e+3*n, quat);
      mji_rotVecQuat(qdmp, dmp_e+3*n, quat);
      int gidx = gindices[n];
      if (enbl_spring) {
        mji_addTo3(frc_g + 3*gidx, qfrc);
      }
      if (enbl_damper) {
        mji_addTo3(dmp_g + 3*gidx, qdmp);
      }
    }
  }

  // apply accumulated forces to bodies
  for (int i = 0; i < nodenum; i++) {
    mju_scl3(dmp_g+3*i, dmp_g+3*i, m->flex_damping[f]);
    int bid = bodyid[i];
    int nidx = i + m->flex_nodeadr[f];

    // fast path: node at body origin (not pinned), direct DOF write
    if (m->body_dofnum[bid] > 0 &&
        (m->flex_centered[f] ||
         (m->flex_node[3*nidx+0] == 0 &&
          m->flex_node[3*nidx+1] == 0 &&
          m->flex_node[3*nidx+2] == 0))) {
      if (enbl_spring) {
        mjtNum qfrc_loc[3];
        mju_mulMatTVec3(qfrc_loc, d->xmat+9*bid, frc_g+3*i);
        mji_addTo3(d->qfrc_spring + m->body_dofadr[bid], qfrc_loc);
      }
      if (enbl_damper) {
        mjtNum qdmp_loc[3];
        mju_mulMatTVec3(qdmp_loc, d->xmat+9*bid, dmp_g+3*i);
        mji_addTo3(d->qfrc_damper + m->body_dofadr[bid], qdmp_loc);
      }
    } else {
      if (enbl_spring) mj_applyFT(m, d, frc_g+3*i, 0, xpos_g+3*i, bid, d->qfrc_spring);
      if (enbl_damper) mj_applyFT(m, d, dmp_g+3*i, 0, xpos_g+3*i, bid, d->qfrc_damper);
    }
  }

  mj_freeStack(d);
}


// 2D shape function gradient: dir=0 returns dphi(s0,l0)*phi(s1,l1),
//                             dir=1 returns phi(s0,l0)*dphi(s1,l1)
static inline mjtNum mju_dphi2D(mjtNum s0, int l0, mjtNum s1, int l1,
                                int order, int dir) {
  if (dir == 0) {
    return mju_flexDphi(s0, l0, order) * mju_flexPhi(s1, l1, order);
  } else {
    return mju_flexPhi(s0, l0, order) * mju_flexDphi(s1, l1, order);
  }
}


// per-edge data layout in flex_bending for interpolated shell bending
#define BEND_EDGE_SIZE 10


// passive bending forces for interpolated flex shell
//
// Current approach: discrete Crouzeix-Raviart — point evaluation of the
// normal jump at each edge midpoint, with energy E = D/(2h) * |Δn - Δn₀|² * l.
//
// TODO(quaglino): upgrade to a Galerkin formulation by precomputing a bending
// stiffness matrix K_bend in the corotated frame using edge normal-jump residuals
// as DOFs and Gauss integration over each face. At runtime, rotate the residual
// vector into the corotated frame, multiply by K_bend, and rotate back. This
// would give implicit derivatives for free (K_bend is constant) and better
// accuracy for elements with varying curvature.
static void mj_flexPassiveBendInterp(const mjModel* m, mjData* d, int f,
                                      int enbl_spring, int enbl_damper) {
  if (m->flex_interp[f] >= 0) {
    return;
  }

  int bendingadr = m->flex_bendingadr[f];
  if (bendingadr < 0) {
    return;
  }

  // read bending edge data
  const mjtNum* bdata = m->flex_bending + bendingadr;
  int nedge = (int)bdata[0];
  if (nedge == 0) return;

  int order = -m->flex_interp[f];  // shell_mode: interp < 0
  int cx = m->flex_cellnum[3*f+0];
  int cy = m->flex_cellnum[3*f+1];
  int cz = m->flex_cellnum[3*f+2];
  int npe = (order+1)*(order+1);
  int nodenum = m->flex_nodenum[f];

  mj_markStack(d);

  // gather global state
  mjtNum* xpos_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* vel_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* frc_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mjtNum* dmp_g = mjSTACKALLOC(d, 3*nodenum, mjtNum);
  mju_flexGatherState(m, d, f, xpos_g, vel_g);
  mju_zero(frc_g, 3*nodenum);
  mju_zero(dmp_g, 3*nodenum);

  // precompute per-face cache: each face appears in multiple bending edges,
  // so caching avoids redundant position gathering and quaternion computation
  const mjtNum* xpos0 = m->flex_node0 + 3*m->flex_nodeadr[f];
  int nfaces = 2*(cy*cz + cx*cz + cx*cy);
  mjtNum* face_xpos = mjSTACKALLOC(d, nfaces * 3*npe, mjtNum);
  int* face_gidx = mjSTACKALLOC(d, nfaces * npe, int);
  mjtNum* face_quat = mjSTACKALLOC(d, nfaces * 4, mjtNum);
  for (int fi = 0; fi < nfaces; fi++) {
    mju_flexGatherFaceState(order, cx, cy, cz, fi, xpos_g,
                            NULL, xpos0,
                            face_xpos + fi * 3*npe, NULL, NULL,
                            face_gidx + fi * npe, face_quat + fi * 4);
  }

  // check cached node indices are in bounds
  for (int i = 0; i < nfaces * npe; i++) {
    if (face_gidx[i] < 0 || face_gidx[i] >= nodenum) {
      mjERROR("cached node index out of range: face_gidx[%d]=%d, nodenum=%d",
              i, face_gidx[i], nodenum);
    }
  }

  // per-edge temporaries
  mjtNum* xpos_A = mjSTACKALLOC(d, 3*npe, mjtNum);
  mjtNum* xpos_B = mjSTACKALLOC(d, 3*npe, mjtNum);
  int* gidx_A = mjSTACKALLOC(d, npe, int);
  int* gidx_B = mjSTACKALLOC(d, npe, int);

  mjtNum kD = m->opt.timestep > 0 ? m->flex_damping[f] / m->opt.timestep : 0;
  if (enbl_damper && kD > 0) {
    mju_warning("Bending damping is not yet supported for interpolated flex shells.");
  }

  for (int e = 0; e < nedge; e++) {
    const mjtNum* edata = bdata + 1 + e * BEND_EDGE_SIZE;
    int fe_A = (int)edata[0];
    int fe_B = (int)edata[1];
    mjtNum local_A[2] = {edata[2], edata[3]};
    mjtNum local_B[2] = {edata[4], edata[5]};
    mjtNum stiffness = edata[6];
    mjtNum dn0[3] = {edata[7], edata[8], edata[9]};

    if (stiffness == 0) continue;

    // look up cached face data instead of recomputing
    mju_copy(xpos_A, face_xpos + fe_A * 3*npe, 3*npe);
    mju_copy(xpos_B, face_xpos + fe_B * 3*npe, 3*npe);
    mju_copyInt(gidx_A, face_gidx + fe_A * npe, npe);
    mju_copyInt(gidx_B, face_gidx + fe_B * npe, npe);
    mjtNum quat_A[4], quat_B[4];
    mju_copy(quat_A, face_quat + fe_A * 4, 4);
    mju_copy(quat_B, face_quat + fe_B * 4, 4);


    // compute deformed normals at edge midpoint
    mjtNum n_A[3], t1_A[3], t2_A[3];
    mjtNum n_B[3], t1_B[3], t2_B[3];
    mju_flexFaceNormal2D(n_A, t1_A, t2_A, order, xpos_A, local_A);
    mju_flexFaceNormal2D(n_B, t1_B, t2_B, order, xpos_B, local_B);

    // normalize normals
    mjtNum len_A = mju_norm3(n_A);
    mjtNum len_B = mju_norm3(n_B);
    if (len_A < mjMINVAL || len_B < mjMINVAL) continue;
    mjtNum inv_A = 1.0 / len_A;
    mjtNum inv_B = 1.0 / len_B;
    n_A[0] *= inv_A; n_A[1] *= inv_A; n_A[2] *= inv_A;
    n_B[0] *= inv_B; n_B[1] *= inv_B; n_B[2] *= inv_B;

    // average corotational frame: symmetric under face swap
    // quat_A and quat_B encode R^{-1}; average them, then negate to get R_avg
    // ensure quaternions are in the same hemisphere before averaging
    if (mju_dot(quat_A, quat_B, 4) < 0) {
      mju_scl(quat_B, quat_B, -1, 4);
    }
    mjtNum quat_avg[4];
    mju_add(quat_avg, quat_A, quat_B, 4);
    mju_normalize(quat_avg, 4);  // NLERP = SLERP at t=0.5 for two quaternions
    // negate to get R_avg (from rest frame to current frame)
    mju_negQuat(quat_avg, quat_avg);

    // rotate dn0 from rest frame to current frame using average corotational R
    mjtNum dn0_rot[3];
    mju_rotVecQuat(dn0_rot, dn0, quat_avg);

    // normal jump residual: r = (n_A - n_B) - R_avg * dn0
    mjtNum r[3];
    mji_sub3(r, n_A, n_B);
    r[0] -= dn0_rot[0]; r[1] -= dn0_rot[1]; r[2] -= dn0_rot[2];

    // --- spring force ---
    if (enbl_spring) {
      // w_A = P_A * r = (r - n_A*(n_A.r)) / |c_A|
      mjtNum dot_A = mju_dot3(n_A, r);
      mjtNum w_A[3];
      w_A[0] = (r[0] - n_A[0]*dot_A) * inv_A;
      w_A[1] = (r[1] - n_A[1]*dot_A) * inv_A;
      w_A[2] = (r[2] - n_A[2]*dot_A) * inv_A;

      mjtNum dot_B = mju_dot3(n_B, r);
      mjtNum w_B[3];
      w_B[0] = (r[0] - n_B[0]*dot_B) * inv_B;
      w_B[1] = (r[1] - n_B[1]*dot_B) * inv_B;
      w_B[2] = (r[2] - n_B[2]*dot_B) * inv_B;

      // precompute cross products: wA x t2_A, wA x t1_A
      mjtNum wAt2[3], wAt1[3], wBt2[3], wBt1[3];
      mji_cross(wAt2, w_A, t2_A);
      mji_cross(wAt1, w_A, t1_A);
      mji_cross(wBt2, w_B, t2_B);
      mji_cross(wBt1, w_B, t1_B);

      // force on face A nodes: f_k = stiffness * [g0_k * (wA x t2_A) -
      //                                           g1_k * (wA x t1_A)]
      int idx = 0;
      for (int l0 = 0; l0 <= order; l0++) {
        for (int l1 = 0; l1 <= order; l1++) {
          mjtNum g0 = mju_dphi2D(local_A[0], l0, local_A[1], l1, order, 0);
          mjtNum g1 = mju_dphi2D(local_A[0], l0, local_A[1], l1, order, 1);
          int gi = gidx_A[idx];
          for (int j = 0; j < 3; j++) {
            frc_g[3*gi + j] += stiffness * (g0 * wAt2[j] - g1 * wAt1[j]);
          }
          idx++;
        }
      }

      // force on face B nodes (negative sign: ∂Δn/∂x = -∂n_B/∂x)
      idx = 0;
      for (int l0 = 0; l0 <= order; l0++) {
        for (int l1 = 0; l1 <= order; l1++) {
          mjtNum g0 = mju_dphi2D(local_B[0], l0, local_B[1], l1, order, 0);
          mjtNum g1 = mju_dphi2D(local_B[0], l0, local_B[1], l1, order, 1);
          int gi = gidx_B[idx];
          for (int j = 0; j < 3; j++) {
            frc_g[3*gi + j] -= stiffness * (g0 * wBt2[j] - g1 * wBt1[j]);
          }
          idx++;
        }
      }
    }

    // --- damping force ---
    // TODO(quaglino): bending damping is disabled because at corner edges
    // with nonzero rest normal jump, dΔn/dt = ω × Δn₀ ≠ 0 under rigid
    // rotation, producing anti-conservative forces. A correct implementation
    // would subtract the rigid-body velocity component before computing the
    // damping residual.
  }

  // apply accumulated forces to bodies
  int* bodyid = m->flex_nodebodyid + m->flex_nodeadr[f];
  for (int i = 0; i < nodenum; i++) {
    int bid = bodyid[i];
    int nidx = i + m->flex_nodeadr[f];

    // fast path: node at body origin, direct DOF write
    if (m->body_dofnum[bid] > 0 &&
        (m->flex_centered[f] ||
         (m->flex_node[3*nidx+0] == 0 &&
          m->flex_node[3*nidx+1] == 0 &&
          m->flex_node[3*nidx+2] == 0))) {
      if (enbl_spring) {
        mjtNum qfrc_loc[3];
        mju_mulMatTVec3(qfrc_loc, d->xmat+9*bid, frc_g+3*i);
        mji_addTo3(d->qfrc_spring + m->body_dofadr[bid], qfrc_loc);
      }
      if (enbl_damper) {
        mjtNum qdmp_loc[3];
        mju_mulMatTVec3(qdmp_loc, d->xmat+9*bid, dmp_g+3*i);
        mji_addTo3(d->qfrc_damper + m->body_dofadr[bid], qdmp_loc);
      }
    } else {
      if (enbl_spring) mj_applyFT(m, d, frc_g+3*i, 0, xpos_g+3*i, bid, d->qfrc_spring);
      if (enbl_damper) mj_applyFT(m, d, dmp_g+3*i, 0, xpos_g+3*i, bid, d->qfrc_damper);
    }
  }

  mj_freeStack(d);
}


// passive forces for flex bending
static void mj_flexPassiveBend(const mjModel* m, mjData* d, int f,
                               int enbl_spring, int enbl_damper) {
  if (m->flex_dim[f] != 2) {
    return;
  }

  int bendingadr = m->flex_bendingadr[f];
  if (bendingadr < 0) {
    return;
  }
  mjtNum damp = enbl_damper ? m->flex_damping[f] : 0;
  enbl_damper = (damp != 0);
  if (!enbl_spring && !enbl_damper) {
    return;
  }

  int vertnum = m->flex_vertnum[f];
  int edgenum = m->flex_edgenum[f];
  mjtNum* xpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
  mjtNum* b = m->flex_bending + bendingadr;
  const mjtNum* bend_plastic = NULL;
  if (m->flex_plasticadr[f] >= 0) {
    int stiffnessadr = m->flex_stiffnessadr[f];
    int stretch_pnum = (stiffnessadr >= 0 && m->flex_stiffness[stiffnessadr] != 0)
                       ? 3 * m->flex_elemnum[f] : 0;
    bend_plastic = d->plastic + m->flex_plasticadr[f] + stretch_pnum;
  }

  mj_markStack(d);
  mjtNum* vvel = enbl_damper ? mjSTACKALLOC(d, 3*vertnum, mjtNum) : NULL;
  mjtNum* spring = enbl_spring ? mjSTACKALLOC(d, 3*vertnum, mjtNum) : NULL;
  mjtNum* damper = enbl_damper ? mjSTACKALLOC(d, 3*vertnum, mjtNum) : NULL;

  if (enbl_spring) {
    mju_zero(spring, 3*vertnum);
  }
  if (enbl_damper) {
    mju_zero(damper, 3*vertnum);
    mj_flexGather(m, d, f, vvel, d->qvel);
  }

  for (int e = 0; e < edgenum; e++) {
    const int* edge = m->flex_edge + 2*(e+m->flex_edgeadr[f]);
    const int* flap = m->flex_edgeflap + 2*(e+m->flex_edgeadr[f]);
    int v[4] = {edge[0], edge[1], flap[0], flap[1]};
    if (v[3] == -1) {
      // skip boundary edges
      continue;
    }

    // flap edges
    mjtNum ed[3][3];
    mji_sub3(ed[0], xpos + 3*v[1], xpos + 3*v[0]);
    mji_sub3(ed[1], xpos + 3*v[2], xpos + 3*v[0]);
    mji_sub3(ed[2], xpos + 3*v[3], xpos + 3*v[0]);

    // forces at the vertices due to curved reference
    mjtNum frc[4][3];
    mji_cross(frc[1], ed[1], ed[2]);
    mji_cross(frc[2], ed[2], ed[0]);
    mji_cross(frc[3], ed[0], ed[1]);
    frc[0][0] = -(frc[1][0] + frc[2][0] + frc[3][0]);
    frc[0][1] = -(frc[1][1] + frc[2][1] + frc[3][1]);
    frc[0][2] = -(frc[1][2] + frc[2][2] + frc[3][2]);

    // accumulate world-space bending forces per vertex
    mjtNum b16 = b[17*e+16] + (bend_plastic ? bend_plastic[e] : 0);
    for (int i = 0; i < 4; i++) {
      for (int x = 0; x < 3; x++) {
        for (int j = 0; j < 4; j++) {
          // thin plate bending force
          if (enbl_spring) spring[3*v[i]+x] += b[17*e+4*i+j] * xpos[3*v[j]+x];

          // thin plate damping force
          if (enbl_damper) damper[3*v[i]+x] += b[17*e+4*i+j] * vvel[3*v[j]+x];
        }

        // curved reference contribution
        if (enbl_spring) spring[3*v[i]+x] += b16 * frc[i][x];
      }
    }
  }

  if (enbl_spring) mj_flexScatter(m, d, f, d->qfrc_spring, spring, -1);
  if (enbl_damper) mj_flexScatter(m, d, f, d->qfrc_damper, damper, -damp);

  mj_freeStack(d);
}


// passive forces for 1D flex bending: isotropic discrete elastic rod without twist, see
// Bergou et al., "Discrete Elastic Rods", https://www.cs.columbia.edu/cg/pdfs/143-rods.pdf
//   energy per hinge: k |kb|^2, with k = EI / (|e0| + |e1|) in the reference; curved references
//   compare kb with its reference value up to rotations about the edges, see mjROD_HINGE_SIZE
//   damping: Rayleigh, restricted so that rigid rotations are not damped
static void mj_flexPassiveBend1D(const mjModel* m, mjData* d, int f,
                                 int enbl_spring, int enbl_damper) {
  int bendingadr = m->flex_bendingadr[f];
  if (bendingadr < 0) {
    return;
  }
  mjtNum damp = enbl_damper ? m->flex_damping[f] : 0;
  enbl_damper = (damp != 0);
  if (!enbl_spring && !enbl_damper) {
    return;
  }

  const mjtNum* b = m->flex_bending + bendingadr;
  int nhinge = (int)b[0];
  int vertnum = m->flex_vertnum[f];
  const mjtNum* xpos = d->flexvert_xpos + 3*m->flex_vertadr[f];

  mj_markStack(d);
  mjtNum* vvel = enbl_damper ? mjSTACKALLOC(d, 3*vertnum, mjtNum) : NULL;
  mjtNum* frc = mjSTACKALLOC(d, 3*vertnum, mjtNum);
  mjtNum* dmp = mjSTACKALLOC(d, 3*vertnum, mjtNum);
  mju_zero(frc, 3*vertnum);
  mju_zero(dmp, 3*vertnum);
  if (enbl_damper) {
    mj_flexGather(m, d, f, vvel, d->qvel);
  }

  mjtNum* Wd_chain = enbl_damper ? mjSTACKALLOC(d, 9*nhinge, mjtNum) : NULL;
  if (mj_rodChain(m, d, f, NULL, enbl_spring ? frc : NULL,
                  enbl_spring ? d->qfrc_spring : NULL, NULL, Wd_chain, NULL)) {
    if (enbl_damper) {
      for (int h = 0; h < nhinge; h++) {
        const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
        int v[3] = {(int)hinge[0], (int)hinge[1], (int)hinge[2]};
        mjtNum kb[3], J[27], udot[3] = {0}, wd[3] = {0};
        if (!mj_rodHinge(xpos + 3*v[0], xpos + 3*v[1], xpos + 3*v[2], kb, J)) {
          continue;
        }
        for (int r = 0; r < 3; r++) {
          for (int i = 0; i < 3; i++) {
            udot[r] += mju_dot3(J + 9*r + 3*i, vvel + 3*v[i]);
          }
        }
        mju_mulMatVec3(wd, Wd_chain + 9*h, udot);
        for (int i = 0; i < 3; i++) {
          for (int c = 0; c < 3; c++) {
            for (int r = 0; r < 3; r++) {
              dmp[3*v[i]+c] -= J[9*r + 3*i + c] * wd[r];
            }
          }
        }
      }
    }
    if (enbl_spring) {
      mj_flexScatter(m, d, f, d->qfrc_spring, frc, 1);
    }
    if (enbl_damper) {
      mj_flexScatter(m, d, f, d->qfrc_damper, dmp, damp);
    }
    mj_freeStack(d);
    return;
  }

  for (int h = 0; h < nhinge; h++) {
    const mjtNum* hinge = b + 1 + mjROD_HINGE_SIZE*h;
    int nvrt = mj_rodHingeVert(hinge);
    int v[4] = {(int)hinge[0], (int)hinge[1], (int)hinge[2], (int)hinge[3]};

    // w: weight in kb (single hinge) or om (hinge pair) space, mapped to vertex forces as -J'w
    mjtNum J[48], w[4] = {0}, Wd[16];
    if (nvrt == 3) {
      mjtNum kb[3];
      if (!mj_rodHinge(xpos + 3*v[0], xpos + 3*v[1], xpos + 3*v[2], kb, J)) {
        continue;
      }
      mj_rodHingeWeights(enbl_spring ? w : NULL, NULL, enbl_damper ? Wd : NULL,
                         kb, hinge[4], hinge[6]);
    } else {
      mjtNum om[4];
      if (!mj_rodPair(xpos + 3*v[0], xpos + 3*v[1], xpos + 3*v[2], xpos + 3*v[3], om, J)) {
        continue;
      }
      mj_rodPairWeights(enbl_spring ? w : NULL, NULL, enbl_damper ? Wd : NULL,
                        om, hinge[4], hinge[5], hinge + 6);
    }

    // damper weight: Wd J vel
    mjtNum wd[4] = {0};
    if (enbl_damper) {
      mjtNum udot[4] = {0};
      for (int r = 0; r < nvrt; r++) {
        for (int i = 0; i < nvrt; i++) {
          udot[r] += mju_dot3(J + 3*nvrt*r + 3*i, vvel + 3*v[i]);
        }
      }
      mju_mulMatVec(wd, Wd, udot, nvrt, nvrt);
    }

    for (int i = 0; i < nvrt; i++) {
      for (int c = 0; c < 3; c++) {
        for (int r = 0; r < nvrt; r++) {
          frc[3*v[i]+c] -= J[3*nvrt*r + 3*i + c] * w[r];
          dmp[3*v[i]+c] -= J[3*nvrt*r + 3*i + c] * wd[r];
        }
      }
    }
  }

  if (enbl_spring) mj_flexScatter(m, d, f, d->qfrc_spring, frc, 1);
  if (enbl_damper) mj_flexScatter(m, d, f, d->qfrc_damper, dmp, damp);

  mj_freeStack(d);
}


// passive forces for flex stretch
static void mj_flexPassiveStretch(const mjModel* m, mjData* d, int f,
                                  int enbl_spring, int enbl_damper) {
  int stiffnessadr = m->flex_stiffnessadr[f];
  if (stiffnessadr < 0) {
    return;
  }
  mjtNum* k = m->flex_stiffness + stiffnessadr;
  if (k[0] == 0) {
    return;
  }

  // Rayleigh damping coefficient, 0 if disabled
  mjtNum kD = enbl_damper && m->opt.timestep > 0 ? m->flex_damping[f] / m->opt.timestep : 0;
  if (!enbl_spring && !kD) {
    return;
  }

  int dim = m->flex_dim[f];
  int nedge = (dim == 2) ? 3 : 6;
  int snh = dim == 3 && k[21] != 0;
  const int* elem = m->flex_elem + m->flex_elemdataadr[f];
  const int* edgeelem = m->flex_elemedge + m->flex_elemedgeadr[f];
  mjtNum* xpos = d->flexvert_xpos + 3*m->flex_vertadr[f];
  mjtNum* vel = d->flexedge_velocity + m->flex_edgeadr[f];
  mjtNum* deformed = d->flexedge_length + m->flex_edgeadr[f];
  mjtNum* reference = m->flexedge_length0 + m->flex_edgeadr[f];
  const mjtNum* plastic = m->flex_plasticadr[f] >= 0 ? d->plastic + m->flex_plasticadr[f] : NULL;

  mj_markStack(d);
  mjtNum* frc = mjSTACKALLOC(d, 3*m->flex_vertnum[f], mjtNum);
  mjtNum* dmp = mjSTACKALLOC(d, 3*m->flex_vertnum[f], mjtNum);
  mju_zero(frc, 3*m->flex_vertnum[f]);
  mju_zero(dmp, 3*m->flex_vertnum[f]);

  // compute forces element-by-element
  int elemnum = m->flex_elemnum[f];
  for (int t = 0; t < elemnum; t++)  {
    const int* vert = elem + (dim+1) * t;

    mjtNum edgevec[6][3], metric[36], tension[6];
    mj_stretchEdgeVectors(edgevec, xpos, vert, dim);

    // shared quadratic energy in squared-edge-length differences
    const mjtNum* packed = k + (dim == 3 ? 24 : 21)*t;
    mjtNum elongation[6];
    mj_stretchElongation(elongation, edgeelem + t*nedge, deformed, reference,
                         plastic ? plastic + nedge*t : NULL, nedge);
    mj_stretchElasticity(metric, tension, packed, elongation, nedge);

    mjtNum grad[4][3], pressure = 0;
    if (snh) {
      mj_snhCubic(tension, elongation, packed[21]);
      pressure = 2*packed[22]*(mj_snhVolume(grad, edgevec, packed)-1);
    }
    if (enbl_spring) {
      mj_stretchForce(frc, vert, tension, edgevec, dim);
      if (snh) {
        for (int v = 0; v < 4; v++) {
          mju_addToScl3(frc + 3*vert[v], grad[v], -pressure);
        }
      }
    }

    if (snh) {
      continue;
    }

    // damper force: generalized Rayleigh damping as described in Section 5.2 of
    // Kharevych et al., "Geometric, Variational Integrators for Computer
    // Animation" http://multires.caltech.edu/pubs/DiscreteLagrangian.pdf; its
    // elongation L^2 - Lprev^2 is factored as dL*(2*L - dL), dL = L - Lprev = vel*timestep,
    // so it has no cancellation and vanishes exactly at zero velocity
    if (kD) {
      for (int e = 0; e < nedge; e++) {
        int idx = edgeelem[t * nedge + e];
        mjtNum dL = vel[idx] * m->opt.timestep;
        elongation[e] = dL*(2*deformed[idx] - dL) * kD;
      }
      mj_stretchTension(tension, metric, elongation, nedge);
      mj_stretchForce(dmp, vert, tension, edgevec, dim);
    }
  }

  if (snh && kD) {
    // SNH Rayleigh damping uses the cached PSD tangent, so its instantaneous power is non-positive
    mjtNum* worldvel = mjSTACKALLOC(d, 3*m->flex_vertnum[f], mjtNum);
    mj_flexGather(m, d, f, worldvel, d->qvel);
    mj_flexHessian(m, d, f);
    mj_flexHessianMul(m, d, f, dmp, worldvel, -m->flex_damping[f]);
  }

  // insert forces into qfrc_spring and qfrc_damper
  if (enbl_spring) {
    mj_flexScatter(m, d, f, d->qfrc_spring, frc, 1);
  }
  if (kD) {
    mj_flexScatter(m, d, f, d->qfrc_damper, dmp, 1);
  }

  mj_freeStack(d);
}


// spring and damper forces
static void mj_springdamper(const mjModel* m, mjData* d) {
  int nv = m->nv, ntendon = m->ntendon;
  int enbl_spring = !mjDISABLED(mjDSBL_SPRING);
  int enbl_damper = !mjDISABLED(mjDSBL_DAMPER);
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  // joint-level springs
  if (enbl_spring) {
    for (int b=0; b < nbody; b++) {
      int i = sleep_filter ? d->body_awake_ind[b] : b;
      int jnt_start = m->body_jntadr[i];
      int jnt_end = jnt_start + m->body_jntnum[i];
      for (int j=jnt_start; j < jnt_end; j++) {
        mjtNum stiffness = m->jnt_stiffness[j];
        const mjtNum* spoly = m->jnt_stiffnesspoly + mjNPOLY*j;

        if (stiffness == 0 && mju_isZero(spoly, mjNPOLY)) {
          continue;
        }

        int padr = m->jnt_qposadr[j];
        int dadr = m->jnt_dofadr[j];
        const mjtNum* qspring = m->jnt_plasticadr[j] >= 0 ?
                                d->plastic + m->jnt_plasticadr[j] :
                                m->qpos_spring + padr;

        switch ((mjtJoint) m->jnt_type[j]) {
        case mjJNT_FREE:
          // apply force
          {
            mjtNum dif[3];
            mji_sub3(dif, d->qpos+padr, qspring);
            mjtNum r = mju_norm3(dif);
            mjtNum k = mju_polyForce(stiffness, spoly, r, mjNPOLY, 0);
            mji_addToScl3(d->qfrc_spring + dadr, dif, -k);
          }

          // continue with rotations
          dadr += 3;
          padr += 3;
          qspring += 3;
          mjFALLTHROUGH;

        case mjJNT_BALL:
          {
            // convert quaternion difference into angular "velocity"
            mjtNum dif[3], quat[4];
            mji_copy4(quat, d->qpos+padr);
            mju_normalize4(quat);
            mji_subQuat(dif, quat, qspring);
            mjtNum r = mju_norm3(dif);
            mjtNum k = mju_polyForce(stiffness, spoly, r, mjNPOLY, 0);

            // apply torque
            mji_addToScl3(d->qfrc_spring + dadr, dif, -k);
          }
          break;

        case mjJNT_SLIDE:
        case mjJNT_HINGE:
          {
            // apply force or torque
            mjtNum x = d->qpos[padr] - qspring[0];
            d->qfrc_spring[dadr] = -x * mju_polyForce(stiffness, spoly, x, mjNPOLY, 0);
          }
          break;
        }
      }
    }
  }

  // dof-level dampers
  if (enbl_damper) {
    int nv_awake = sleep_filter ? d->nv_awake : nv;
    for (int j = 0; j < nv_awake; j++) {
      int i = sleep_filter ? d->dof_awake_ind[j] : j;
      mjtNum poly[mjNPOLY];
      mju_copy(poly, m->dof_dampingpoly + mjNPOLY*i, mjNPOLY);
      mjtNum damping = m->dof_damping[i]
                       + mj_actuatorDamping(m, mjOBJ_JOINT, m->dof_jntid[i], poly);
      if (damping != 0 || !mju_isZero(poly, mjNPOLY)) {
        mjtNum v = d->qvel[i];
        d->qfrc_damper[i] = -v * mju_polyForce(damping, poly, v, mjNPOLY, 1);
      }
    }
  }

  // flex elasticity
  for (int f=0; f < m->nflex; f++) {
    // skip rigid, sleeping or static flex
    if (m->flex_rigid[f] || (sleep_filter && mj_sleepState(m, d, mjOBJ_FLEX, f) != mjS_AWAKE)) {
      continue;
    }

    if (m->flex_dim[f] == 1) {
      // rod bending forces
      mj_flexPassiveBend1D(m, d, f, enbl_spring, enbl_damper);
      continue;
    }

    if (m->flex_interp[f]) {
      // interpolated flex: stretch forces
      mj_flexPassiveInterp(m, d, f, enbl_spring, enbl_damper);

      // interpolated shell bending forces
      mj_flexPassiveBendInterp(m, d, f, enbl_spring, enbl_damper);
    } else {
      // add bending forces
      mj_flexPassiveBend(m, d, f, enbl_spring, enbl_damper);

      // stretch forces
      mj_flexPassiveStretch(m, d, f, enbl_spring, enbl_damper);
    }
  }

  // flexedge-level spring-dampers
  for (int f=0; f < m->nflex; f++) {
    mjtNum stiffness = enbl_spring ? m->flex_edgestiffness[f] : 0;
    mjtNum damping = enbl_damper ? m->flex_edgedamping[f] : 0;

    // disabled or rigid: nothing to do
    if (m->flex_rigid[f] || (stiffness == 0 && damping == 0)) {
      continue;
    }

    // skip sleeping or static flex
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_FLEX, f) != mjS_AWAKE) {
      continue;
    }

    // process non-rigid edges of this flex (global edge index)
    int edgeend = m->flex_edgeadr[f] + m->flex_edgenum[f];
    for (int e=m->flex_edgeadr[f]; e < edgeend; e++) {
      // skip rigid
      if (m->flexedge_rigid[e]) {
        continue;
      }

      // compute spring-damper force along edge
      mjtNum frc_spring = stiffness * (m->flexedge_length0[e] - d->flexedge_length[e]);
      mjtNum frc_damper = -damping * d->flexedge_velocity[e];

      // transform to joint torque, add to qfrc_{spring, damper}: always sparse
      int end = m->flexedge_J_rowadr[e] + m->flexedge_J_rownnz[e];
      for (int j=m->flexedge_J_rowadr[e]; j < end; j++) {
        int colind = m->flexedge_J_colind[j];
        mjtNum J = d->flexedge_J[j];
        d->qfrc_spring[colind] += J * frc_spring;
        d->qfrc_damper[colind] += J * frc_damper;
      }
    }
  }

  // tendon-level spring-dampers
  for (int i=0; i < ntendon; i++) {
    // skip sleeping or static tendon
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) != mjS_AWAKE) {
      continue;
    }

    // compute spring and damper forces along tendon
    mjtNum frc_spring, frc_damper;
    mj_tendonSpringDamper(m, d, i, &frc_spring, &frc_damper);

    // transform to joint torque, add to qfrc_{spring, damper}
    if (frc_spring || frc_damper) {
      int end = m->ten_J_rowadr[i] + m->ten_J_rownnz[i];
      for (int j=m->ten_J_rowadr[i]; j < end; j++) {
        int k = m->ten_J_colind[j];
        mjtNum J = d->ten_J[j];
        d->qfrc_spring[k] += J * frc_spring;
        d->qfrc_damper[k] += J * frc_damper;
      }
    }
  }
}


// body-level gravity compensation, return 1 if any, 0 otherwise
static int mj_gravcomp(const mjModel* m, mjData* d) {
  if (!m->flg_gravcomp || mjDISABLED(mjDSBL_GRAVITY) || mju_norm3(m->opt.gravity) == 0) {
    return 0;
  }

  int has_gravcomp = 0;
  mjtNum force[3], torque[3]={0};
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  // apply per-body gravity compensation
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;
    if (m->body_gravcomp[i]) {
      has_gravcomp = 1;
      mji_scl3(force, m->opt.gravity, -(m->body_mass[i]*m->body_gravcomp[i]));
      mj_applyFT(m, d, force, torque, d->xipos+3*i, i, d->qfrc_gravcomp);
    }
  }

  return has_gravcomp;
}


// fluid forces
static int mj_fluid(const mjModel* m, mjData* d) {
  // no fluid forces: early return
  if (!m->opt.viscosity && !m->opt.density) {
    return 0;
  }

  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  for (int b=0; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    if (m->body_mass[i] < mjMINVAL) {
      continue;
    }

    // if any child geom uses the ellipsoid model, inertia-box model is disabled for parent body
    int use_ellipsoid_model = 0;
    int geomnum = m->body_geomnum[i];
    for (int j=0; j < geomnum && use_ellipsoid_model == 0; j++) {
      const int geomid = m->body_geomadr[i] + j;
      use_ellipsoid_model += (m->geom_fluid[mjNFLUID*geomid] > 0);
    }

    if (use_ellipsoid_model) {
      mj_ellipsoidFluidModel(m, d, i);
    } else {
      mj_inertiaBoxFluidModel(m, d, i);
    }
  }

  return 1;
}


// passive contact forces
int mj_contactPassive(const mjModel* m, mjData* d) {
  if (mjDISABLED(mjDSBL_CONTACT) || d->ncon == 0 || m->nv == 0) {
    return 0;
  }

  // the force rides on the rows the metric build published at the position stage, so the force,
  // the operator and the velocity shift see one contact set. Without an active metric there are
  // no rows: a model that wants passive flex contact gets an error rather than a silent zero
  if (!d->efm_active) {
    for (int f=0; f < m->nflex; f++) {
      if (mj_effFlexContactPossible(m, f)) {
        mjERROR("passive flex contact needs the effective metric: run the position stage "
                "(mj_forward, or mj_fwdPosition) before mj_passive");
      }
    }
    return 0;
  }
  if (!d->nefmcon) {
    return 0;
  }

  mj_effContactForce(d, d->qfrc_spring);
  return 1;
}


// adhesion forces: constant attraction along the normals of adhesive contacts
int mj_adhesion(const mjModel* m, mjData* d) {
  int ncon = d->ncon, issparse = mj_isSparse(m);
  int NV, nv = m->nv, *chain = NULL;
  mjtNum *jac, *jacdif, *jac1p, *jac2p, *qfrc;
  mjContact* con;
  int has_adhesion = 0;

  if (!m->flg_adhesion || mjDISABLED(mjDSBL_CONTACT) || ncon == 0 || nv == 0) {
    return 0;
  }

  // early return if no adhesive contact
  for (int i=0; i < ncon; i++) {
    if (d->contact[i].adhesion && d->contact[i].exclude <= 1) {
      has_adhesion = 1;
      break;
    }
  }
  if (!has_adhesion) {
    return 0;
  }

  // allocate Jacobian; contacts are normal-only (dim 1): no rotational buffers
  mj_markStack(d);
  jac    = mjSTACKALLOC(d, nv, mjtNum);
  jacdif = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac1p  = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac2p  = mjSTACKALLOC(d, 3*nv, mjtNum);
  qfrc   = mjSTACKALLOC(d, nv, mjtNum);
  if (issparse) {
    chain = mjSTACKALLOC(d, nv, int);
  }

  // pull adhesive contacts together along the contact normal
  for (int i=0; i < ncon; i++) {
    con = d->contact + i;
    if (!con->adhesion || con->exclude > 1) {
      continue;
    }

    // normal Jacobian
    NV = mj_contactJacobian(m, d, con, 1, jacdif, NULL,
                            jac1p, jac2p, NULL, NULL, chain);
    if (NV == 0) {
      continue;
    }
    mju_mulMatMat(jac, con->frame, jacdif, 1, 3, NV);

    // accumulate qfrc_adhesion += jacN' * (-adhesion)
    if (!issparse) {
      mju_addToScl(d->qfrc_adhesion, jac, -con->adhesion, nv);
    } else {
      mju_scl(qfrc, jac, -con->adhesion, NV);
      for (int j=0; j < NV; j++) {
        d->qfrc_adhesion[chain[j]] += qfrc[j];
      }
    }
  }

  mj_freeStack(d);
  return 1;
}


// all passive forces
void mj_passive(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nv = sleep_filter ? d->nv_awake : m->nv;
  const int* dof_awake_ind = sleep_filter ? d->dof_awake_ind : NULL;

  // clear passive force vectors for awake dofs
  if (sleep_filter) {
    mju_zeroInd(d->qfrc_spring,   nv, dof_awake_ind);
    mju_zeroInd(d->qfrc_damper,   nv, dof_awake_ind);
    mju_zeroInd(d->qfrc_gravcomp, nv, dof_awake_ind);
    mju_zeroInd(d->qfrc_fluid,    nv, dof_awake_ind);
    mju_zeroInd(d->qfrc_adhesion, nv, dof_awake_ind);
    mju_zeroInd(d->qfrc_passive,  nv, dof_awake_ind);
  } else {
    mju_zero(d->qfrc_spring,   nv);
    mju_zero(d->qfrc_damper,   nv);
    mju_zero(d->qfrc_gravcomp, nv);
    mju_zero(d->qfrc_fluid,    nv);
    mju_zero(d->qfrc_adhesion, nv);
    mju_zero(d->qfrc_passive,  nv);
  }

  // both spring and damping disabled: skip all passive forces
  if (mjDISABLED(mjDSBL_SPRING) && mjDISABLED(mjDSBL_DAMPER)) {
    return;
  }

  // springs and dampers
  mj_springdamper(m, d);

  // gravity compensation
  int has_gravcomp = mj_gravcomp(m, d);

  // fluid forces
  int has_fluid = mj_fluid(m, d);

  // contact forces
  mj_contactPassive(m, d);

  // adhesion forces
  int has_adhesion = mj_adhesion(m, d);

  // add passive forces into qfrc_passive
  if (sleep_filter) {
    mju_addInd(d->qfrc_passive, d->qfrc_spring, d->qfrc_damper, dof_awake_ind, nv);
  } else {
    mju_add(d->qfrc_passive, d->qfrc_spring, d->qfrc_damper, nv);
  }

  if (has_fluid) {
    if (sleep_filter) {
      mju_addToInd(d->qfrc_passive, d->qfrc_fluid, dof_awake_ind, nv);
    } else {
      mju_addTo(d->qfrc_passive, d->qfrc_fluid, nv);
    }
  }

  if (has_adhesion) {
    if (sleep_filter) {
      mju_addToInd(d->qfrc_passive, d->qfrc_adhesion, dof_awake_ind, nv);
    } else {
      mju_addTo(d->qfrc_passive, d->qfrc_adhesion, nv);
    }
  }

  if (has_gravcomp) {
    int ndof = sleep_filter ? d->nv_awake : nv;
    for (int v=0; v < ndof; v++) {
      int dof = sleep_filter ? d->dof_awake_ind[v] : v;

      // add gravity compensation force unless added via actuators
      if (!m->jnt_actgravcomp[m->dof_jntid[dof]]) {
        d->qfrc_passive[dof] += d->qfrc_gravcomp[dof];
      }
    }
  }

  // user callback: add custom passive forces
  if (mjcb_passive) {
    mjcb_passive(m, d);
  }

  // plugin: add custom passive forces
  if (m->nplugin) {
    const int nslot = mjp_pluginCount();

    // iterate over plugins, call compute if type is mjPLUGIN_PASSIVE
    for (int i=0; i < m->nplugin; i++) {
      const int slot = m->plugin[i];
      const mjpPlugin* plugin = mjp_getPluginAtSlotUnsafe(slot, nslot);
      if (!plugin) {
        mjERROR("invalid plugin slot: %d", slot);
      }
      if (plugin->capabilityflags & mjPLUGIN_PASSIVE) {
        if (!plugin->compute) {
          mjERROR("`compute` is a null function pointer for plugin at slot %d", slot);
        }
        plugin->compute(m, d, i, mjPLUGIN_PASSIVE);
      }
    }
  }
}


//---------------------------------- fluid models --------------------------------------------------

// fluid forces based on inertia-box approximation
void mj_inertiaBoxFluidModel(const mjModel* m, mjData* d, int i) {
  mjtNum lvel[6], wind[6], lwind[6], lfrc[6], bfrc[6], box[3], diam, *inertia;
  inertia = m->body_inertia + 3*i;
  box[0] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[1] + inertia[2] - inertia[0])) / m->body_mass[i] * 6.0);
  box[1] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[0] + inertia[2] - inertia[1])) / m->body_mass[i] * 6.0);
  box[2] = mju_sqrt(mju_max(mjMINVAL,
                            (inertia[0] + inertia[1] - inertia[2])) / m->body_mass[i] * 6.0);

  // map from CoM-centered to local body-centered 6D velocity
  mj_objectVelocity(m, d, mjOBJ_BODY, i, lvel, 1);

  // compute wind in local coordinates
  mju_zero(wind, 6);
  mji_copy3(wind+3, m->opt.wind);
  mju_transformSpatial(lwind, wind, 0, d->xipos+3*i,
                       d->subtree_com+3*m->body_rootid[i], d->ximat+9*i);

  // subtract translational component from body velocity
  mji_subFrom3(lvel+3, lwind+3);
  mju_zero(lfrc, 6);

  // set viscous force and torque
  if (m->opt.viscosity > 0) {
    // diameter of sphere approximation
    diam = (box[0] + box[1] + box[2])/3.0;

    // angular viscosity
    mji_scl3(lfrc, lvel, -mjPI*diam*diam*diam*m->opt.viscosity);

    // linear viscosity
    mji_scl3(lfrc+3, lvel+3, -3.0*mjPI*diam*m->opt.viscosity);
  }

  // add lift and drag force and torque
  if (m->opt.density > 0) {
    // force
    lfrc[3] -= 0.5*m->opt.density*box[1]*box[2]*mju_abs(lvel[3])*lvel[3];
    lfrc[4] -= 0.5*m->opt.density*box[0]*box[2]*mju_abs(lvel[4])*lvel[4];
    lfrc[5] -= 0.5*m->opt.density*box[0]*box[1]*mju_abs(lvel[5])*lvel[5];

    // torque
    lfrc[0] -= m->opt.density*box[0]*(box[1]*box[1]*box[1]*box[1]+box[2]*box[2]*box[2]*box[2])*
               mju_abs(lvel[0])*lvel[0]/64.0;
    lfrc[1] -= m->opt.density*box[1]*(box[0]*box[0]*box[0]*box[0]+box[2]*box[2]*box[2]*box[2])*
               mju_abs(lvel[1])*lvel[1]/64.0;
    lfrc[2] -= m->opt.density*box[2]*(box[0]*box[0]*box[0]*box[0]+box[1]*box[1]*box[1]*box[1])*
               mju_abs(lvel[2])*lvel[2]/64.0;
  }
  // rotate to global orientation: lfrc -> bfrc
  mji_mulMatVec3(bfrc, d->ximat+9*i, lfrc);
  mji_mulMatVec3(bfrc+3, d->ximat+9*i, lfrc+3);

  // apply force and torque to body com
  mj_applyFT(m, d, bfrc+3, bfrc, d->xipos+3*i, i, d->qfrc_fluid);
}


// fluid forces based on ellipsoid approximation
void mj_ellipsoidFluidModel(const mjModel* m, mjData* d, int bodyid) {
  mjtNum lvel[6], wind[6], lwind[6], lfrc[6], bfrc[6];
  mjtNum geom_interaction_coef, magnus_lift_coef, kutta_lift_coef;
  mjtNum semiaxes[3], virtual_mass[3], virtual_inertia[3];
  mjtNum blunt_drag_coef, slender_drag_coef, ang_drag_coef;

  for (int j=0; j < m->body_geomnum[bodyid]; j++) {
    const int geomid = m->body_geomadr[bodyid] + j;

    mju_geomSemiAxes(semiaxes, m->geom_size + 3*geomid, m->geom_type[geomid]);

    readFluidGeomInteraction(
      m->geom_fluid + mjNFLUID*geomid, &geom_interaction_coef,
      &blunt_drag_coef, &slender_drag_coef, &ang_drag_coef,
      &kutta_lift_coef, &magnus_lift_coef,
      virtual_mass, virtual_inertia);

    // scales all forces, read from MJCF as boolean (0.0 or 1.0)
    if (geom_interaction_coef == 0.0) {
      continue;
    }

    // map from CoM-centered to local body-centered 6D velocity
    mj_objectVelocity(m, d, mjOBJ_GEOM, geomid, lvel, 1);

    // compute wind in local coordinates
    mju_zero(wind, 6);
    mji_copy3(wind+3, m->opt.wind);
    mju_transformSpatial(lwind, wind, 0,
                         d->geom_xpos + 3*geomid,  // Frame of ref's origin.
                         d->subtree_com + 3*m->body_rootid[bodyid],
                         d->geom_xmat + 9*geomid);  // Frame of ref's orientation.

    // subtract translational component from grom velocity
    mji_subFrom3(lvel+3, lwind+3);

    // initialize viscous force and torque
    mju_zero(lfrc, 6);

    // added-mass forces and torques
    mj_addedMassForces(lvel, NULL, m->opt.density, virtual_mass, virtual_inertia, lfrc);

    // lift force orthogonal to lvel from Kutta-Joukowski theorem
    mj_viscousForces(lvel, m->opt.density, m->opt.viscosity, semiaxes, magnus_lift_coef,
                     kutta_lift_coef, blunt_drag_coef, slender_drag_coef, ang_drag_coef, lfrc);

    // scale by geom_interaction_coef (1.0 by default)
    mju_scl(lfrc, lfrc, geom_interaction_coef, 6);

    // rotate to global orientation: lfrc -> bfrc
    mji_mulMatVec3(bfrc, d->geom_xmat + 9*geomid, lfrc);
    mji_mulMatVec3(bfrc+3, d->geom_xmat + 9*geomid, lfrc+3);

    // apply force and torque to body com
    mj_applyFT(m, d, bfrc+3, bfrc,
               d->geom_xpos + 3*geomid,  // point where FT is generated
               bodyid, d->qfrc_fluid);
  }
}


// compute forces due to fluid mass moving with the body
void mj_addedMassForces(const mjtNum local_vels[6], const mjtNum local_accels[6],
                        const mjtNum fluid_density, const mjtNum virtual_mass[3],
                        const mjtNum virtual_inertia[3], mjtNum local_force[6])
{
  const mjtNum lin_vel[3] = {local_vels[3], local_vels[4], local_vels[5]};
  const mjtNum ang_vel[3] = {local_vels[0], local_vels[1], local_vels[2]};
  const mjtNum virtual_lin_mom[3] = {
    fluid_density * virtual_mass[0] * lin_vel[0],
    fluid_density * virtual_mass[1] * lin_vel[1],
    fluid_density * virtual_mass[2] * lin_vel[2]
  };
  const mjtNum virtual_ang_mom[3] = {
    fluid_density * virtual_inertia[0] * ang_vel[0],
    fluid_density * virtual_inertia[1] * ang_vel[1],
    fluid_density * virtual_inertia[2] * ang_vel[2]
  };

  // disabled due to dependency on qacc but included for completeness
  if (local_accels) {
    local_force[0] -= fluid_density * virtual_inertia[0] * local_accels[0];
    local_force[1] -= fluid_density * virtual_inertia[1] * local_accels[1];
    local_force[2] -= fluid_density * virtual_inertia[2] * local_accels[2];
    local_force[3] -= fluid_density * virtual_mass[0] * local_accels[3];
    local_force[4] -= fluid_density * virtual_mass[1] * local_accels[4];
    local_force[5] -= fluid_density * virtual_mass[2] * local_accels[5];
  }

  mjtNum added_mass_force[3], added_mass_torque1[3], added_mass_torque2[3];
  mji_cross(added_mass_force, virtual_lin_mom, ang_vel);
  mji_cross(added_mass_torque1, virtual_lin_mom, lin_vel);
  mji_cross(added_mass_torque2, virtual_ang_mom, ang_vel);

  mji_addTo3(local_force, added_mass_torque1);
  mji_addTo3(local_force, added_mass_torque2);
  mji_addTo3(local_force+3, added_mass_force);
}


// inlined helper functions
static inline mjtNum mji_pow4(const mjtNum val) {
  return (val*val)*(val*val);
}

static inline mjtNum mji_pow2(const mjtNum val) {
  return val*val;
}

static inline mjtNum mji_ellipsoid_max_moment(const mjtNum size[3], const int dir) {
  const mjtNum d0 = size[dir], d1 = size[(dir+1) % 3], d2 = size[(dir+2) % 3];
  return 8.0/15.0 * mjPI * d0 * mji_pow4(mju_max(d1, d2));
}


// lift and drag forces due to motion in the fluid
void mj_viscousForces(
  const mjtNum local_vels[6], const mjtNum fluid_density,
  const mjtNum fluid_viscosity, const mjtNum size[3],
  const mjtNum magnus_lift_coef, const mjtNum kutta_lift_coef,
  const mjtNum blunt_drag_coef, const mjtNum slender_drag_coef,
  const mjtNum ang_drag_coef, mjtNum local_force[6])
{
  const mjtNum lin_vel[3] = {local_vels[3], local_vels[4], local_vels[5]};
  const mjtNum ang_vel[3] = {local_vels[0], local_vels[1], local_vels[2]};
  const mjtNum volume = 4.0/3.0 * mjPI * size[0] * size[1] * size[2];
  const mjtNum d_max = mju_max(mju_max(size[0], size[1]), size[2]);
  const mjtNum d_min = mju_min(mju_min(size[0], size[1]), size[2]);
  const mjtNum d_mid = size[0] + size[1] + size[2] - d_max - d_min;
  const mjtNum A_max = mjPI * d_max * d_mid;

  mjtNum magnus_force[3];
  mji_cross(magnus_force, ang_vel, lin_vel);
  magnus_force[0] *= magnus_lift_coef * fluid_density * volume;
  magnus_force[1] *= magnus_lift_coef * fluid_density * volume;
  magnus_force[2] *= magnus_lift_coef * fluid_density * volume;

  // unit velocity and normalized semi-axes
  const mjtNum speed = mju_norm3(lin_vel);
  const mjtNum inv_speed = speed > mjMINVAL ? 1.0 / speed : 0.0;
  const mjtNum v[3] = {lin_vel[0]*inv_speed, lin_vel[1]*inv_speed, lin_vel[2]*inv_speed};
  const mjtNum inv_dmax = d_max > mjMINVAL ? 1.0 / d_max : 0.0;
  const mjtNum s[3] = {size[0]*inv_dmax, size[1]*inv_dmax, size[2]*inv_dmax};

  // the dot product between velocity and the normal to the cross-section that
  // defines the body's projection along velocity is proj_num/sqrt(proj_denom)
  const mjtNum a_raw = mji_pow2(s[1] * s[2]);
  const mjtNum b_raw = mji_pow2(s[2] * s[0]);
  const mjtNum c_raw = mji_pow2(s[0] * s[1]);
  const mjtNum proj_num = a_raw * v[0]*v[0] + b_raw * v[1]*v[1] + c_raw * v[2]*v[2];
  const mjtNum inv_proj_num = proj_num > mjMINVAL ? 1.0 / proj_num : 0.0;
  const mjtNum a = a_raw * inv_proj_num;
  const mjtNum b = b_raw * inv_proj_num;
  const mjtNum c = c_raw * inv_proj_num;
  const mjtNum proj_denom = a*a * v[0]*v[0] + b*b * v[1]*v[1] + c*c * v[2]*v[2];

  // projected surface in the direction of the velocity
  const mjtNum A_proj = mjPI * d_max * d_max * mju_sqrt(proj_num * proj_denom);

  // not-unit normal to ellipsoid's projected area in the direction of velocity
  const mjtNum norm[3] = {a * v[0], b * v[1], c * v[2]};

  // cosine between velocity and normal to the surface
  // divided by proj_denom instead of sqrt(proj_denom) to account for skipped normalization in norm
  const mjtNum cos_alpha = proj_denom > mjMINVAL ? 1.0 / proj_denom : 0.0;
  mjtNum kutta_circ[3];
  mji_cross(kutta_circ, norm, lin_vel);
  kutta_circ[0] *= kutta_lift_coef * fluid_density * cos_alpha * A_proj;
  kutta_circ[1] *= kutta_lift_coef * fluid_density * cos_alpha * A_proj;
  kutta_circ[2] *= kutta_lift_coef * fluid_density * cos_alpha * A_proj;
  mjtNum kutta_force[3];
  mji_cross(kutta_force, kutta_circ, lin_vel);

  // viscous force and torque in Stokes flow, analytical for spherical bodies
  const mjtNum eq_sphere_D = 2.0/3.0 * (size[0] + size[1] + size[2]);
  const mjtNum lin_visc_force_coef = 3.0 * mjPI * eq_sphere_D;
  const mjtNum lin_visc_torq_coef = mjPI * eq_sphere_D*eq_sphere_D*eq_sphere_D;

  // moments of inertia used to compute angular quadratic drag
  const mjtNum I_max = 8.0/15.0 * mjPI * d_mid * mji_pow4(d_max);
  const mjtNum II[3] = {
    mji_ellipsoid_max_moment(size, 0),
    mji_ellipsoid_max_moment(size, 1),
    mji_ellipsoid_max_moment(size, 2)
  };
  const mjtNum mom_visc[3] = {
    ang_vel[0] * (ang_drag_coef*II[0] + slender_drag_coef*(I_max - II[0])),
    ang_vel[1] * (ang_drag_coef*II[1] + slender_drag_coef*(I_max - II[1])),
    ang_vel[2] * (ang_drag_coef*II[2] + slender_drag_coef*(I_max - II[2]))
  };

  const mjtNum drag_lin_coef =  // linear plus quadratic
                               fluid_viscosity*lin_visc_force_coef + fluid_density*speed*(
    A_proj*blunt_drag_coef + slender_drag_coef*(A_max - A_proj));
  const mjtNum drag_ang_coef =  // linear plus quadratic
                               fluid_viscosity * lin_visc_torq_coef +
                               fluid_density * mju_norm3(mom_visc);

  local_force[0] -= drag_ang_coef * ang_vel[0];
  local_force[1] -= drag_ang_coef * ang_vel[1];
  local_force[2] -= drag_ang_coef * ang_vel[2];
  local_force[3] += magnus_force[0] + kutta_force[0] - drag_lin_coef*lin_vel[0];
  local_force[4] += magnus_force[1] + kutta_force[1] - drag_lin_coef*lin_vel[1];
  local_force[5] += magnus_force[2] + kutta_force[2] - drag_lin_coef*lin_vel[2];
}


// read the geom_fluid_coefs array into its constituent parts
void readFluidGeomInteraction(const mjtNum* geom_fluid_coefs,
                              mjtNum* geom_fluid_coef,
                              mjtNum* blunt_drag_coef,
                              mjtNum* slender_drag_coef,
                              mjtNum* ang_drag_coef,
                              mjtNum* kutta_lift_coef,
                              mjtNum* magnus_lift_coef,
                              mjtNum virtual_mass[3],
                              mjtNum virtual_inertia[3]) {
  int i = 0;
  geom_fluid_coef[0]   = geom_fluid_coefs[i++];
  blunt_drag_coef[0]   = geom_fluid_coefs[i++];
  slender_drag_coef[0] = geom_fluid_coefs[i++];
  ang_drag_coef[0]     = geom_fluid_coefs[i++];
  kutta_lift_coef[0]   = geom_fluid_coefs[i++];
  magnus_lift_coef[0]  = geom_fluid_coefs[i++];
  virtual_mass[0]      = geom_fluid_coefs[i++];
  virtual_mass[1]      = geom_fluid_coefs[i++];
  virtual_mass[2]      = geom_fluid_coefs[i++];
  virtual_inertia[0]   = geom_fluid_coefs[i++];
  virtual_inertia[1]   = geom_fluid_coefs[i++];
  virtual_inertia[2]   = geom_fluid_coefs[i++];
  if (i != mjNFLUID) {
    mjERROR("wrong number of entries.");
  }
}
