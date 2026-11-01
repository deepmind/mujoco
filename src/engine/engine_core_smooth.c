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

#include "engine/engine_core_smooth.h"

#include <stddef.h>
#include <string.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include "engine/engine_core_constraint.h"
#include "engine/engine_core_util.h"
#include "engine/engine_crossplatform.h"
#include "engine/engine_inline.h"
#include "engine/engine_macro.h"
#include "engine/engine_memory.h"
#include "engine/engine_sleep.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_sparse.h"
#include "engine/engine_util_spatial.h"


//--------------------------- position -------------------------------------------------------------

// forward kinematics part 1: bodies
void mj_kinematics1(const mjModel* m, mjData* d) {
  int nbody = m->nbody;

  // set world position and orientation
  mju_zero3(d->xpos);
  mju_unit4(d->xquat);
  mju_zero3(d->xipos);
  mju_unit4(d->xiquat);
  mju_zero(d->xmat, 9);
  mju_zero(d->ximat, 9);
  d->xmat[0] = d->xmat[4] = d->xmat[8] = 1;
  d->ximat[0] = d->ximat[4] = d->ximat[8] = 1;

  int sleep_filter = mjENABLED(mjENBL_SLEEP);

  // compute global cartesian positions and orientations of all bodies
  for (int i=1; i < nbody; i++) {
    // skip static bodies
    if (sleep_filter) {
      if (d->body_awake[i] == mjS_STATIC) continue;
    }

    mjtNum xpos[3], xquat[4];
    int jntadr = m->body_jntadr[i];
    int jntnum = m->body_jntnum[i];

    // free joint
    if (jntnum == 1 && m->jnt_type[jntadr] == mjJNT_FREE) {
      // get qpos address
      int qadr = m->jnt_qposadr[jntadr];

      // copy pos and quat from qpos
      mji_copy3(xpos, d->qpos+qadr);
      mji_copy4(xquat, d->qpos+qadr+3);
      mju_normalize4(xquat);

      // assign xanchor and xaxis
      mji_copy3(d->xanchor+3*jntadr, xpos);
      mji_copy3(d->xaxis+3*jntadr, m->jnt_axis+3*jntadr);
    }

    // regular or no joint
    else {
      int pid = m->body_parentid[i];

      // get body pos and quat: from model or mocap
      mjtNum *bodypos, *bodyquat, quat[4];
      if (m->body_mocapid[i] >= 0) {
        bodypos = d->mocap_pos + 3*m->body_mocapid[i];
        mji_copy4(quat, d->mocap_quat + 4*m->body_mocapid[i]);
        mju_normalize4(quat);
        bodyquat = quat;
      } else {
        bodypos = m->body_pos+3*i;
        bodyquat = m->body_quat+4*i;
      }

      // apply fixed translation and rotation relative to parent
      if (pid) {
        mji_mulMatVec3(xpos, d->xmat+9*pid, bodypos);
        mji_addTo3(xpos, d->xpos+3*pid);
        mji_mulQuat(xquat, d->xquat+4*pid, bodyquat);
      } else {
        // parent is the world
        mji_copy3(xpos, bodypos);
        mji_copy4(xquat, bodyquat);
      }

      // accumulate joints, compute xpos and xquat for this body
      mjtNum xanchor[3], xaxis[3];
      for (int j=0; j < jntnum; j++) {
        // get joint id, qpos address, joint type
        int jid = jntadr + j;
        int qadr = m->jnt_qposadr[jid];
        mjtJoint jtype = m->jnt_type[jid];

        // compute axis in global frame; ball jnt_axis is (0,0,1), set by compiler
        mji_rotVecQuat(xaxis, m->jnt_axis+3*jid, xquat);

        // compute anchor in global frame
        mji_rotVecQuat(xanchor, m->jnt_pos+3*jid, xquat);
        mji_addTo3(xanchor, xpos);

        // apply joint transformation
        switch (jtype) {
        case mjJNT_SLIDE:
          mji_addToScl3(xpos, xaxis, d->qpos[qadr] - m->qpos0[qadr]);
          break;

        case mjJNT_BALL:
        case mjJNT_HINGE:
          {
            // compute local quaternion rotation
            mjtNum qloc[4];
            if (jtype == mjJNT_BALL) {
              mji_copy4(qloc, d->qpos+qadr);
              mju_normalize4(qloc);
            } else {
              mji_axisAngle2Quat(qloc, m->jnt_axis+3*jid, d->qpos[qadr] - m->qpos0[qadr]);
            }

            // apply rotation
            mju_mulQuat(xquat, xquat, qloc);

            // correct for off-center rotation
            mjtNum vec[3];
            mji_rotVecQuat(vec, m->jnt_pos+3*jid, xquat);
            mji_sub3(xpos, xanchor, vec);
          }
          break;

        default:
          mjERROR("unknown joint type %d", jtype);  // SHOULD NOT OCCUR
        }

        // assign xanchor and xaxis
        mji_copy3(d->xanchor+3*jid, xanchor);
        mji_copy3(d->xaxis+3*jid, xaxis);
      }
    }

    // normalize quaternion
    mju_normalize4(xquat);

    // sleeping body, check for mismatch
    if (sleep_filter && jntnum && d->body_awake[i] == mjS_ASLEEP) {
      // compare new and existing xpos and xquat
      const mjtNum* pos = d->xpos+3*i;
      const mjtNum* xq = d->xquat+4*i;
      int match = xpos[0] == pos[0] && xpos[1] == pos[1] && xpos[2] == pos[2] &&
                  xquat[0] == xq[0] && xquat[1] == xq[1] && xquat[2] == xq[2] && xquat[3] == xq[3];

      // match: continue to next body
      if (match) {
        continue;
      }

      // mismatch: mark the tree for waking later (in mj_wake)
      else {
        d->tree_awake[m->body_treeid[i]] = 1;
      }
    }

    // assign xquat and xpos, construct xmat
    mji_copy4(d->xquat+4*i, xquat);
    mji_copy3(d->xpos+3*i, xpos);
    mji_quat2Mat(d->xmat+9*i, xquat);
  }
}


// forward kinematics part 2: body inertias, geoms and sites
void mj_kinematics2(const mjModel* m, mjData* d) {
  int sleep_enabled = mjENABLED(mjENBL_SLEEP);
  int sleep_filter = sleep_enabled && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  // compute/copy Cartesian positions and orientations of body inertial frames
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // skip static body
    if (sleep_enabled && d->body_awake[i] == mjS_STATIC) continue;

    mj_local2GlobalQuat(d, d->xipos+3*i, d->xiquat+4*i, d->ximat+9*i,
                        m->body_ipos+3*i, m->body_iquat+4*i,
                        i, m->body_sameframe[i]);
  }

  // compute/copy Cartesian positions and orientations of geoms
  for (int b=0; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // skip geoms in static body
    if (sleep_enabled && d->body_awake[i] == mjS_STATIC) continue;

    int start = m->body_geomadr[i];
    int end = start + m->body_geomnum[i];
    for (int g=start; g < end; g++) {
      mj_local2GlobalQuat(d, d->geom_xpos+3*g, d->geom_xquat+4*g, d->geom_xmat+9*g,
                          m->geom_pos+3*g, m->geom_quat+4*g,
                          m->geom_bodyid[g], m->geom_sameframe[g]);
    }
  }

  // compute/copy Cartesian positions and orientations of sites
  int nsite = m->nsite;
  for (int i=0; i < nsite; i++) {
    int bodyid = m->site_bodyid[i];

    // skip site in sleeping or static body
    if (sleep_enabled && d->body_awake[bodyid] != mjS_AWAKE) continue;

    mj_local2GlobalQuat(d, d->site_xpos+3*i, d->site_xquat+4*i, d->site_xmat+9*i,
                        m->site_pos+3*i, m->site_quat+4*i,
                        bodyid, m->site_sameframe[i]);
  }
}


// forward kinematics
void mj_kinematics(const mjModel* m, mjData* d) {
  mj_kinematics1(m, d);
  if (mj_wake(m, d)) {
    mj_updateSleep(m, d);
  }
  mj_kinematics2(m, d);
}


// map inertias and motion dofs to global frame centered at subtree-CoM
void mj_comPos(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  int nparent = sleep_filter ? d->nparent_awake : m->nbody;

  // subtree_com: initialize with body moment
  for (int b=0; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    mji_scl3(d->subtree_com+3*i, d->xipos+3*i, m->body_mass[i]);
  }

  // subtree_com: accumulate to parent in backward pass
  for (int b=nparent-1; b >= 0; b--) {
    int i = sleep_filter ? d->parent_awake_ind[b] : b;
    if (!i) continue;

    // accumulate moment to parent, rescale if sleeping
    int parent = m->body_parentid[i];
    if (sleep_filter && d->body_awake[i] == mjS_ASLEEP) {
      mjtNum child_moment[3];
      mji_scl3(child_moment, d->subtree_com+3*i, m->body_subtreemass[i]);
      mji_addTo3(d->subtree_com+3*parent, child_moment);
    } else {
      mji_addTo3(d->subtree_com+3*parent, d->subtree_com+3*i);
    }
  }

  // subtree_com: normalize
  for (int b=0; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    if (m->body_subtreemass[i] < mjMINVAL) {
      mji_copy3(d->subtree_com+3*i, d->xipos+3*i);
    } else {
      mju_scl3(d->subtree_com + 3 * i, d->subtree_com + 3 * i,
               1.0 / m->body_subtreemass[i]);
    }
  }

  // zero out CoM frame inertia for the world body
  mju_zero(d->cinert, 10);

  // map inertias to frame centered at subtree_com
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    mjtNum offset[3];
    mji_sub3(offset, d->xipos+3*i, d->subtree_com+3*m->body_rootid[i]);
    mju_inertCom(d->cinert+10*i, m->body_inertia+3*i, d->ximat+9*i, offset, m->body_mass[i]);
  }

  // map motion dofs to global frame centered at subtree_com
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    int jntnum = m->body_jntnum[i];
    if (!jntnum) continue;

    int start = m->body_jntadr[i];
    int end = start + jntnum;
    for (int j=start; j < end; j++) {
      // get cdof address
      int da = 6*m->jnt_dofadr[j];

      // compute com-anchor vector
      mjtNum offset[3], axis[3];
      mji_sub3(offset, d->subtree_com+3*m->body_rootid[i], d->xanchor+3*j);

      // create motion dof
      int skip = 0;
      switch ((mjtJoint) m->jnt_type[j]) {
      case mjJNT_FREE:
        // translation components: x, y, z in global frame
        mju_zero(d->cdof+da, 18);
        d->cdof[da+3+7*0] = 1;
        d->cdof[da+3+7*1] = 1;
        d->cdof[da+3+7*2] = 1;

        // rotation components: same as ball
        skip = 18;
        mjFALLTHROUGH;

      case mjJNT_BALL:
        for (int k=0; k < 3; k++) {
          // I_3 rotation in child frame (assume no subsequent rotations)
          axis[0] = d->xmat[9*i + k + 0];
          axis[1] = d->xmat[9*i + k + 3];
          axis[2] = d->xmat[9*i + k + 6];

          mju_dofCom(d->cdof+da+skip+6*k, axis, offset);
        }
        break;

      case mjJNT_SLIDE:
        mju_dofCom(d->cdof+da, d->xaxis+3*j, 0);
        break;

      case mjJNT_HINGE:
        mju_dofCom(d->cdof+da, d->xaxis+3*j, offset);
        break;
      }
    }
  }
}


// compute camera and light positions and orientations
void mj_camlight(const mjModel* m, mjData* d) {
  int ncam = m->ncam, nlight = m->nlight;
  int sleep_filter = mjENABLED(mjENBL_SLEEP);

  // compute Cartesian positions and orientations of cameras
  for (int i=0; i < ncam; i++) {
    // get camera body id and target body id
    int id = m->cam_bodyid[i];
    int id1 = m->cam_targetbodyid[i];

    // skip camera if both body and target body are asleep or static
    if (sleep_filter && d->body_awake[id] != mjS_AWAKE) {
      if (id1 < 0 || d->body_awake[id1] != mjS_AWAKE) {
        continue;
      }
    }

    // default processing for fixed mode
    mj_local2Global(d, d->cam_xpos+3*i, d->cam_xmat+9*i,
                    m->cam_pos+3*i, m->cam_quat+4*i, id, 0);

    // adjust for mode
    switch ((mjtCamLight) m->cam_mode[i]) {
    case mjCAMLIGHT_FIXED:
      break;
    case mjCAMLIGHT_TRACK:
    case mjCAMLIGHT_TRACKCOM:
      // fixed global orientation
      mji_copy9(d->cam_xmat+9*i, m->cam_mat0+9*i);

      // position: track camera body
      if (m->cam_mode[i] == mjCAMLIGHT_TRACK) {
        mji_add3(d->cam_xpos+3*i, d->xpos+3*id, m->cam_pos0+3*i);
      }

      // position: track subtree com
      else {
        mji_add3(d->cam_xpos+3*i, d->subtree_com+3*id, m->cam_poscom0+3*i);
      }
      break;

    case mjCAMLIGHT_TARGETBODY:
    case mjCAMLIGHT_TARGETBODYCOM:
      // only if target body is specified
      if (id1 >= 0) {
        mjtNum pos[3];
        // get position to look at
        if (m->cam_mode[i] == mjCAMLIGHT_TARGETBODY) {
          mji_copy3(pos, d->xpos+3*id1);
        } else {
          mji_copy3(pos, d->subtree_com+3*id1);
        }

        // zaxis = -desired camera direction, in global frame
        mjtNum matT[9];
        mji_sub3(matT+6, d->cam_xpos+3*i, pos);
        mju_normalize3(matT+6);

        // xaxis: orthogonal to zaxis and to (0,0,1)
        matT[3] = 0;
        matT[4] = 0;
        matT[5] = 1;
        mji_cross(matT, matT+3, matT+6);
        mju_normalize3(matT);

        // yaxis: orthogonal to xaxis and zaxis
        mji_cross(matT+3, matT+6, matT);
        mju_normalize3(matT+3);

        // set camera frame
        mju_transpose(d->cam_xmat+9*i, matT, 3, 3);
      }
    }
  }

  // compute Cartesian positions and directions of lights
  for (int i=0; i < nlight; i++) {
    // get light body id and target body id
    int id = m->light_bodyid[i];
    int id1 = m->light_targetbodyid[i];

    // skip light if both body and target body are asleep or static
    if (sleep_filter && d->body_awake[id] != mjS_AWAKE) {
      if (id1 < 0 || d->body_awake[id1] != mjS_AWAKE) {
        continue;
      }
    }

    // default processing for fixed mode
    mj_local2Global(d, d->light_xpos+3*i, 0, m->light_pos+3*i, 0, id, 0);
    mji_rotVecQuat(d->light_xdir+3*i, m->light_dir+3*i, d->xquat+4*id);

    // adjust for mode
    switch ((mjtCamLight) m->light_mode[i]) {
    case mjCAMLIGHT_FIXED:
      break;
    case mjCAMLIGHT_TRACK:
    case mjCAMLIGHT_TRACKCOM:
      // fixed global orientation
      mji_copy3(d->light_xdir+3*i, m->light_dir0+3*i);

      // position: track light body
      if (m->light_mode[i] == mjCAMLIGHT_TRACK) {
        mji_add3(d->light_xpos+3*i, d->xpos+3*id, m->light_pos0+3*i);
      }

      // position: track subtree com
      else {
        mji_add3(d->light_xpos+3*i, d->subtree_com+3*id, m->light_poscom0+3*i);
      }
      break;

    case mjCAMLIGHT_TARGETBODY:
    case mjCAMLIGHT_TARGETBODYCOM:
      // only if target body is specified
      if (id1 >= 0) {
        // get position to look at
        mjtNum lookat[3];
        if (m->light_mode[i] == mjCAMLIGHT_TARGETBODY) {
          mji_copy3(lookat, d->xpos+3*id1);
        } else {
          mji_copy3(lookat, d->subtree_com+3*id1);
        }

        // set dir
        mji_sub3(d->light_xdir+3*i, lookat, d->light_xpos+3*i);
      }
    }

    // normalize dir
    mju_normalize3(d->light_xdir+3*i);
  }
}


// update dynamic BVH; leaf aabbs must be updated before call
void mj_updateDynamicBVH(const mjModel* m, mjData* d, int bvhadr, int bvhnum) {
  mj_markStack(d);
  int* modified = mjSTACKALLOC(d, bvhnum, int);
  mju_zeroInt(modified, bvhnum);

  // mark leafs as modified
  for (int i=0; i < bvhnum; i++) {
    if (m->bvh_nodeid[bvhadr+i] >= 0) {
      modified[i] = 1;
    }
  }

  // update non-leafs in backward pass (parents come before children)
  for (int i=bvhnum-1; i >= 0; i--) {
    if (m->bvh_nodeid[bvhadr+i] < 0) {
      int child1 = m->bvh_child[2*(bvhadr+i)];
      int child2 = m->bvh_child[2*(bvhadr+i)+1];

      // update if either child is modified
      if (modified[child1] || modified[child2]) {
        mjtNum* aabb = d->bvh_aabb_dyn + 6*(bvhadr - m->nbvhstatic + i);
        const mjtNum* aabb1 = d->bvh_aabb_dyn + 6*(bvhadr - m->nbvhstatic + child1);
        const mjtNum* aabb2 = d->bvh_aabb_dyn + 6*(bvhadr - m->nbvhstatic + child2);

        // compute new (min, max)
        mjtNum xmin[3], xmax[3];
        for (int k=0; k < 3; k++) {
          xmin[k] = mju_min(aabb1[k] - aabb1[k+3], aabb2[k] - aabb2[k+3]);
          xmax[k] = mju_max(aabb1[k] + aabb1[k+3], aabb2[k] + aabb2[k+3]);
        }

        // convert to (center, size)
        for (int k=0; k < 3; k++) {
          aabb[k]   = 0.5*(xmax[k]+xmin[k]);
          aabb[k+3] = 0.5*(xmax[k]-xmin[k]);
        }

        modified[i] = 1;
      }
    }
  }

  mj_freeStack(d);
}


// compute flex-related quantities, optionally skipping rigid flexes
void mj_flexSkip(const mjModel* m, mjData* d, int skiprigid) {
  int nv = m->nv;
  int* rowadr = m->flexedge_J_rowadr;

  // skip if no flexes
  if (!m->nflex) {
    return;
  }

  // geometry updates invalidate the lazily prepared Cartesian stretch Hessian
  memset(d->flex_hessian_valid, 0, m->nflex * sizeof(mjtBool));

  // compute Cartesian positions of flex vertices
  for (int f=0; f < m->nflex; f++) {
    int vstart = m->flex_vertadr[f];
    int vend = m->flex_vertadr[f] + m->flex_vertnum[f];
    int nstart = m->flex_nodeadr[f];
    int nend = m->flex_nodeadr[f] + m->flex_nodenum[f];

    // 0: vertices are the mesh vertices, 1: vertices are interpolated from nodal dofs
    if (m->flex_interp[f] == 0) {
      for (int i=vstart; i < vend; i++) {
        if (m->flex_centered[f] ||
            (m->flex_vert[3*i+0] == 0 &&
             m->flex_vert[3*i+1] == 0 &&
             m->flex_vert[3*i+2] == 0)) {
          mji_copy3(d->flexvert_xpos+3*i, d->xpos+3*m->flex_vertbodyid[i]);
        } else {
          mji_mulMatVec3(d->flexvert_xpos+3*i, d->xmat+9*m->flex_vertbodyid[i], m->flex_vert+3*i);
          mji_addTo3(d->flexvert_xpos+3*i, d->xpos+3*m->flex_vertbodyid[i]);
        }
      }
    }

    // trilinear/quadratic interpolation
    else {
      int nodenum = nend - nstart;
      mj_markStack(d);
      mjtNum* nodexpos = mjSTACKALLOC(d, 3*nodenum, mjtNum);
      for (int i=nstart; i < nend; i++) {
        int j = i - nstart;
        if (m->flex_centered[f] ||
            (m->flex_node[3*i+0] == 0 &&
             m->flex_node[3*i+1] == 0 &&
             m->flex_node[3*i+2] == 0)) {
          mji_copy3(nodexpos + 3*j, d->xpos + 3*m->flex_nodebodyid[i]);
        } else {
          mji_mulMatVec3(nodexpos + 3*j, d->xmat + 9*m->flex_nodebodyid[i], m->flex_node + 3*i);
          mji_addTo3(nodexpos + 3*j, d->xpos + 3*m->flex_nodebodyid[i]);
        }
      }

      int interp = m->flex_interp[f];
      int order = interp < 0 ? -interp : interp;
      int cx = m->flex_cellnum[3*f+0];
      int cy = m->flex_cellnum[3*f+1];
      int cz = m->flex_cellnum[3*f+2];
      int nx_g = cx * order + 1;
      int ny_g = cy * order + 1;
      int nz_g = cz * order + 1;
      if (nend - nstart != nx_g * ny_g * nz_g) {
        mjERROR("flex_interp_order mismatch");
      }

      // shell mode: reconstruct interior node positions from boundary via TFI
      if (interp < 0) {
        mju_shellTrackInterior(nodexpos, nx_g, ny_g, nz_g);
      }

      for (int i=vstart; i < vend; i++) {
        mju_zero3(d->flexvert_xpos+3*i);

        // cell lookup: get local coords and node indices
        mjtNum local[3];
        int nodeindices[27];  // max npc for quadratic: 3^3 = 27
        mju_cellLookup(m->flex_vert0 + 3*i, m->flex_cellnum+3*f, order, m->flex_vertcell[i], local, nodeindices);
        mju_interpolate3D(d->flexvert_xpos+3*i, local, nodexpos, order, nodeindices);
      }
      mj_freeStack(d);
    }
  }

  // compute flex element aabb
  for (int f=0; f < m->nflex; f++) {
    int dim = m->flex_dim[f];

    // process elements of this flex
    int elemnum = m->flex_elemnum[f];
    for (int e=0; e < elemnum; e++) {
      const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e*(dim+1);
      const mjtNum* vert = d->flexvert_xpos + 3*m->flex_vertadr[f];

      // compute min and max along each global axis
      mjtNum xmin[3], xmax[3];
      mji_copy3(xmin, vert+3*edata[0]);
      mji_copy3(xmax, vert+3*edata[0]);
      for (int i=1; i <= dim; i++) {
        for (int j=0; j < 3; j++) {
          mjtNum value = vert[3*edata[i]+j];
          xmin[j] = mju_min(xmin[j], value);
          xmax[j] = mju_max(xmax[j], value);
        }
      }

      // compute aabb (center, size)
      int base = m->flex_elemadr[f] + e;
      d->flexelem_aabb[6*base+0] = 0.5*(xmax[0]+xmin[0]);
      d->flexelem_aabb[6*base+1] = 0.5*(xmax[1]+xmin[1]);
      d->flexelem_aabb[6*base+2] = 0.5*(xmax[2]+xmin[2]);
      d->flexelem_aabb[6*base+3] = 0.5*(xmax[0]-xmin[0]) + m->flex_radius[f];
      d->flexelem_aabb[6*base+4] = 0.5*(xmax[1]-xmin[1]) + m->flex_radius[f];
      d->flexelem_aabb[6*base+5] = 0.5*(xmax[2]-xmin[2]) + m->flex_radius[f];
    }
  }

  // update flex bhv_aabb_dyn if needed
  if (!mjDISABLED(mjDSBL_MIDPHASE)) {
    for (int f=0; f < m->nflex; f++) {
      if (m->flex_bvhadr[f] >= 0) {
        int flex_bvhadr = m->flex_bvhadr[f];
        int flex_bvhnum = m->flex_bvhnum[f];

        // copy element aabbs to bhv leaf aabbs
        for (int i=flex_bvhadr; i < flex_bvhadr+flex_bvhnum; i++) {
          if (m->bvh_nodeid[i] >= 0) {
            mji_copy6(d->bvh_aabb_dyn + 6*(i - m->nbvhstatic),
                      d->flexelem_aabb + 6*(m->flex_elemadr[f] + m->bvh_nodeid[i]));
          }
        }

        // update dynamic BVH
        mj_updateDynamicBVH(m, d, m->flex_bvhadr[f], m->flex_bvhnum[f]);
      }
    }
  }

  // allocate space
  mj_markStack(d);
  mjtNum* jac1 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jac2 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacdif = mjSTACKALLOC(d, 3*nv, mjtNum);
  int* chain = mjSTACKALLOC(d, nv, int);

  // clear Jacobian
  mju_zero(d->flexedge_J, m->nJfe);

  // compute lengths and Jacobians of edges
  for (int f=0; f < m->nflex; f++) {
    // skip if edges cannot generate forces
    if ((skiprigid && m->flex_rigid[f]) || m->flex_interp[f]) {
      continue;
    }

    // skip edge Jacobian if no built-in passive force is needed
    int skipjacobian = m->flex_edgeequality[f] != 1 &&
                       !m->flex_edgedamping[f] &&
                       !m->flex_edgestiffness[f] &&
                       !m->flex_damping[f];

    // process edges of this flex
    int vbase = m->flex_vertadr[f];
    int ebase = m->flex_edgeadr[f];
    int edgenum = m->flex_edgenum[f];
    for (int e=0; e < edgenum; e++) {
      int v1 = m->flex_edge[2*(ebase+e)];
      int v2 = m->flex_edge[2*(ebase+e)+1];
      int b1 = m->flex_vertbodyid[vbase+v1];
      int b2 = m->flex_vertbodyid[vbase+v2];
      mjtNum* pos1 = d->flexvert_xpos + 3*(vbase+v1);
      mjtNum* pos2 = d->flexvert_xpos + 3*(vbase+v2);

      // vec = unit vector from v1 to v2, compute edge length
      mjtNum vec[3];
      mji_sub3(vec, pos2, pos1);
      d->flexedge_length[ebase+e] = mju_normalize3(vec);

      // skip Jacobian if not needed
      if (skipjacobian) {
        continue;
      }

      // get endpoint Jacobians, subtract
      int NV = mj_jacDifPair(m, d, chain, b1, b2, pos1, pos2,
                              jac1, jac2, jacdif, NULL, NULL, NULL, /*issparse=*/1,
                              /*skipcommon=*/0);

      // no dofs: skip
      if (!NV) {
        continue;
      }
      if (NV > m->flexedge_J_rownnz[ebase+e]) {
        mjERROR("flexedge_J_rownnz too small");
      }

      // apply chain rule to compute edge Jacobian
      mju_mulMatTVec(d->flexedge_J + rowadr[ebase+e], jacdif, vec, 3, NV);
    }
  }

  mj_freeStack(d);
}


// compute flex-related quantities
void mj_flex(const mjModel* m, mjData* d) {
  mj_flexSkip(m, d, 1);
}


// compute tendon lengths and moments
void mj_tendon(const mjModel* m, mjData* d) {
  int nv = m->nv, nten = m->ntendon;
  const int *rownnz = m->ten_J_rownnz, *rowadr = m->ten_J_rowadr, *colind = m->ten_J_colind;
  mjtNum *L = d->ten_length, *J = d->ten_J;

  if (!nten) {
    return;
  }

  // allocate stack arrays
  int *chain;
  mjtNum *jac1, *jac2, *jacdif, *tmp;
  mj_markStack(d);
  jac1 = mjSTACKALLOC(d, 3*nv, mjtNum);
  jac2 = mjSTACKALLOC(d, 3*nv, mjtNum);
  jacdif = mjSTACKALLOC(d, 3*nv, mjtNum);
  tmp = mjSTACKALLOC(d, nv, mjtNum);
  chain = mjSTACKALLOC(d, nv, int);

  // clear results
  mju_zero(L, nten);

  // clear Jacobian
  mju_zero(J, m->nJten);

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;

  // loop over tendons
  int wrapcount = 0;
  for (int i=0; i < nten; i++) {
    // skip sleeping tendon
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) == mjS_ASLEEP) {
      continue;
    }

    // initialize tendon path
    int adr = m->tendon_adr[i];
    d->ten_wrapadr[i] = wrapcount;
    d->ten_wrapnum[i] = 0;
    int tendon_num = m->tendon_num[i];

    // process fixed tendon
    if (m->tendon_type[i] == mjTENDON_FIXED) {
      // process all defined joints
      for (int j=0; j < tendon_num; j++) {
        // get joint id
        int k = m->wrap_objid[adr+j];

        // add to length
        L[i] += m->wrap_prm[adr+j] * d->qpos[m->jnt_qposadr[k]];

        mjtNum coef = 1;
        int dofadr = m->jnt_dofadr[k];
        mju_combineSparseInc(J + rowadr[i], &coef, m->nv, 1, m->wrap_prm[adr+j],
                             rownnz[i], 1, colind + rowadr[i], &dofadr);
      }

      continue;
    }

    // process spatial tendon
    mjtNum divisor = 1;
    int wraptype, j = 0;
    while (j < tendon_num-1) {
      // get 1st and 2nd object
      int type0 = m->wrap_type[adr+j+0];
      int type1 = m->wrap_type[adr+j+1];
      int id0 = m->wrap_objid[adr+j+0];
      int id1 = m->wrap_objid[adr+j+1];

      // pulley
      if (type0 == mjWRAP_PULLEY || type1 == mjWRAP_PULLEY) {
        // get divisor, insert obj=-2
        if (type0 == mjWRAP_PULLEY) {
          divisor = m->wrap_prm[adr+j];
          mju_zero3(d->wrap_xpos+wrapcount*3);
          d->wrap_obj[wrapcount] = -2;
          d->ten_wrapnum[i]++;
          wrapcount++;
        }

        // move to next
        j++;
        continue;
      }

      // init sequence; assume it starts with site
      mjtNum wlen = -1;
      int wrapid = -1;
      mjtNum wpnt[12];
      mji_copy3(wpnt, d->site_xpos+3*id0);
      int wbody[4];
      wbody[0] = m->site_bodyid[id0];

      // second object is geom: process site-geom-site
      if (type1 == mjWRAP_SPHERE || type1 == mjWRAP_CYLINDER) {
        // reassign, get 2nd site info
        wraptype = type1;
        wrapid = id1;
        type1 = m->wrap_type[adr+j+2];
        id1 = m->wrap_objid[adr+j+2];

        // do wrapping, possibly get 2 extra points (wlen>=0)
        int sideid = mju_round(m->wrap_prm[adr+j+1]);
        if (sideid < -1 || sideid >= m->nsite) {
          mjERROR("invalid sideid %d in wrap_prm", sideid);  // SHOULD NOT OCCUR
        }

        wlen = mju_wrap(wpnt+3, d->site_xpos+3*id0, d->site_xpos+3*id1,
                        d->geom_xpos+3*wrapid, d->geom_xmat+9*wrapid, m->geom_size[3*wrapid],
                        wraptype, (sideid >= 0 ? d->site_xpos+3*sideid : 0));
      } else {
        wraptype = mjWRAP_NONE;
      }

      // complete sequence, accumulate lengths
      if (wlen < 0) {
        mji_copy3(wpnt+3, d->site_xpos+3*id1);
        wbody[1] = m->site_bodyid[id1];
        L[i] += mju_dist3(wpnt, wpnt+3) / divisor;
      } else {
        mji_copy3(wpnt+9, d->site_xpos+3*id1);
        wbody[1] = wbody[2] = m->geom_bodyid[wrapid];
        wbody[3] = m->site_bodyid[id1];
        L[i] += (mju_dist3(wpnt, wpnt+3) + wlen + mju_dist3(wpnt+6, wpnt+9)) / divisor;
      }

      // accumulate moments if consecutive points are in different bodies
      for (int k=0; k < (wlen < 0 ? 1 : 3); k++) {
        if (wbody[k] != wbody[k+1]) {
          // get 3D position difference, normalize
          mjtNum dif[3];
          mji_sub3(dif, wpnt+3*k+3, wpnt+3*k);
          mju_normalize3(dif);

          // get endpoint Jacobians, subtract
          int NV = mj_jacDifPair(m, d, chain,
                                 wbody[k], wbody[k+1], wpnt+3*k, wpnt+3*k+3,
                                 jac1, jac2, jacdif, NULL, NULL, NULL, /*issparse=*/1,
                                 /*skipcommon=*/0);

          // no dofs: skip
          if (!NV) {
            continue;
          }

          // apply chain rule to compute tendon Jacobian
          mju_mulMatTVec(tmp, jacdif, dif, 3, NV);

          // add to existing
          mju_combineSparseInc(J+rowadr[i], tmp, nv, 1, 1/divisor,
                              rownnz[i], NV, colind+rowadr[i],
                              chain);
        }
      }

      // assign to wrap
      if (wlen < 0) {
        mji_copy3(d->wrap_xpos+wrapcount*3, wpnt);
      } else {
        mji_copy9(d->wrap_xpos+wrapcount*3, wpnt);
      }
      d->wrap_obj[wrapcount] = -1;
      if (wlen >= 0) {
        d->wrap_obj[wrapcount+1] = d->wrap_obj[wrapcount+2] = wrapid;
      }
      d->ten_wrapnum[i] += (wlen < 0 ? 1 : 3);
      wrapcount += (wlen < 0 ? 1 : 3);

      // advance
      j += (wraptype != mjWRAP_NONE ? 2 : 1);

      // assign last site before pulley or tendon end
      if (j == tendon_num-1 || m->wrap_type[adr+j+1] == mjWRAP_PULLEY) {
        mji_copy3(d->wrap_xpos+wrapcount*3, d->site_xpos+3*id1);
        d->wrap_obj[wrapcount] = -1;
        d->ten_wrapnum[i]++;
        wrapcount++;
      }
    }
  }

  mj_freeStack(d);
}


// return dot product of tendon Jacobian time derivative with vector
mjtNum mj_tendonDot(const mjModel* m, mjData* d, int id, const mjtNum* vec) {
  int nv = m->nv;
  mjtNum res = 0;

  // tendon id is invalid: return
  if (id < 0 || id >= m->ntendon) {
    return 0;
  }

  // fixed tendon has zero Jdot: return
  if (m->tendon_type[id] == mjTENDON_FIXED) {
    return 0;
  }

  // allocate stack arrays
  mj_markStack(d);
  int issparse = mj_isSparse(m);
  int* chain = issparse ? mjSTACKALLOC(d, nv, int) : NULL;
  mjtNum* jac1 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jac2 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacdif = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* tmp = mjSTACKALLOC(d, nv, mjtNum);

  // process spatial tendon
  mjtNum divisor = 1;
  int wraptype, j = 0;
  int adr = m->tendon_adr[id];
  int num = m->tendon_num[id];
  while (j < num-1) {
    // get 1st and 2nd object
    int type0 = m->wrap_type[adr+j+0];
    int type1 = m->wrap_type[adr+j+1];
    int id0 = m->wrap_objid[adr+j+0];
    int id1 = m->wrap_objid[adr+j+1];

    // pulley
    if (type0 == mjWRAP_PULLEY || type1 == mjWRAP_PULLEY) {
      // get divisor, insert obj=-2
      if (type0 == mjWRAP_PULLEY) {
        divisor = m->wrap_prm[adr+j];
      }

      // move to next
      j++;
      continue;
    }

    // init sequence; assume it starts with site
    mjtNum wpnt[6];
    mji_copy3(wpnt, d->site_xpos+3*id0);
    mjtNum vel[6];
    mj_objectVelocity(m, d, mjOBJ_SITE, id0, vel, /*flg_local=*/0);
    mjtNum wvel[6] = {vel[3], vel[4], vel[5], 0, 0, 0};
    int wbody[2];
    wbody[0] = m->site_bodyid[id0];

    // second object is geom: process site-geom-site
    if (type1 == mjWRAP_SPHERE || type1 == mjWRAP_CYLINDER) {
      // TODO(tassa) support geom wrapping (requires derivatives of mju_wrap)
      mjERROR("geom wrapping not supported");
    } else {
      wraptype = mjWRAP_NONE;
    }

    // complete sequence
    wbody[1] = m->site_bodyid[id1];
    mji_copy3(wpnt+3, d->site_xpos+3*id1);
    mj_objectVelocity(m, d, mjOBJ_SITE, id1, vel, /*flg_local=*/0);
    mji_copy3(wvel+3, vel+3);

    // accumulate moments if consecutive points are in different bodies
    if (wbody[0] != wbody[1]) {
      // dpnt = 3D position difference, normalize
      mjtNum dpnt[3];
      mju_sub3(dpnt, wpnt+3, wpnt);
      mjtNum norm = mju_normalize3(dpnt);

      // dvel = d / dt (dpnt)
      mjtNum dvel[3];
      mju_sub3(dvel, wvel+3, wvel);
      mjtNum dot = mju_dot3(dpnt, dvel);
      mju_addToScl3(dvel, dpnt, -dot);
      mju_scl3(dvel, dvel, norm > mjMINVAL ? 1/norm : 0);

      // sparse
      if (issparse) {
        // construct merged chain
        int NV = mj_mergeChain(m, chain, wbody[0], wbody[1], /*flg_skipcommon=*/0);

        if (NV) {
          // get endpoint JacobianDots, subtract
          mj_jacDotSparse(m, d, jac1, 0, wpnt, wbody[0], NV, chain);
          mj_jacDotSparse(m, d, jac2, 0, wpnt+3, wbody[1], NV, chain);
          mju_sub(jacdif, jac2, jac1, 3*NV);

          // chain rule, first term: Jdot += d/dt(jac2 - jac1) * dpnt
          mju_mulMatTVec(tmp, jacdif, dpnt, 3, NV);

          for (int k=0; k < NV; k++) {
            res += (tmp[k] / divisor) * vec[chain[k]];
          }

          // get endpoint Jacobians, subtract
          mj_jacSparse(m, d, jac1, 0, wpnt, wbody[0], NV, chain, /*flg_skipcommon=*/0);
          mj_jacSparse(m, d, jac2, 0, wpnt+3, wbody[1], NV, chain, /*flg_skipcommon=*/0);
          mju_sub(jacdif, jac2, jac1, 3*NV);

          // chain rule, second term: Jdot += (jac2 - jac1) * d/dt(dpnt)
          mju_mulMatTVec(tmp, jacdif, dvel, 3, NV);

          for (int k=0; k < NV; k++) {
            res += (tmp[k] / divisor) * vec[chain[k]];
          }
        }
      }

      // dense
      else {
        // get endpoint JacobianDots, subtract
        mj_jacDot(m, d, jac1, 0, wpnt, wbody[0]);
        mj_jacDot(m, d, jac2, 0, wpnt+3, wbody[1]);
        mju_sub(jacdif, jac2, jac1, 3*nv);

        // chain rule, first term: Jdot += d/dt(jac2 - jac1) * dpnt
        mju_mulMatTVec(tmp, jacdif, dpnt, 3, nv);

        res += mju_dot(tmp, vec, nv) / divisor;

        // get endpoint Jacobians, subtract
        mj_jac(m, d, jac1, 0, wpnt, wbody[0]);
        mj_jac(m, d, jac2, 0, wpnt+3, wbody[1]);
        mju_sub(jacdif, jac2, jac1, 3*nv);

        // chain rule, second term: Jdot += (jac2 - jac1) * d/dt(dpnt)
        mju_mulMatTVec(tmp, jacdif, dvel, 3, nv);

        res += mju_dot(tmp, vec, nv) / divisor;
      }
    }

    // advance
    j += (wraptype != mjWRAP_NONE ? 2 : 1);
  }

  mj_freeStack(d);
  return res;
}


// compute actuator/transmission lengths and moments
void mj_transmission(const mjModel* m, mjData* d) {
  int nv = m->nv, nactuator = m->nactuator;

  // nothing to do
  if (!nactuator) {
    return;
  }

  // outputs
  mjtNum* length = d->actuator_length;
  mjtNum* moment = d->actuator_moment;
  int *rownnz = d->moment_rownnz;
  int *rowadr = d->moment_rowadr;
  int *colind = d->moment_colind;

  // allocate Jacbians
  mj_markStack(d);
  mjtNum* jac  = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacA = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacS = mjSTACKALLOC(d, 3*nv, mjtNum);

  // define stack variables required for body transmission, don't allocate
  int issparse = mj_isSparse(m);
  mjtNum* efc_force = NULL;  // used as marker for allocation requirement
  mjtNum *moment_exclude, *jacdifp, *jac1p, *jac2p;
  int *chain;

  // define stack variables required for site transmission, don't allocate
  mjtNum *jacref = NULL, *moment_row = NULL;

  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < nv;

  // compute lengths and moments
  for (int i=0; i < nactuator; i++) {
    // address of the actuator's output block (single row for all current types)
    int out = m->actuator_outadr[i];
    rowadr[out] = out == 0 ? 0 : rowadr[out-1] + rownnz[out-1];
    int nnz, adr = rowadr[out];

    // skip sleeping actuator: zero all rows of its output block
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_ACTUATOR, i) == mjS_ASLEEP) {
      for (int k=0; k < m->actuator_outnum[i]; k++) {
        rowadr[out+k] = out+k == 0 ? 0 : rowadr[out+k-1] + rownnz[out+k-1];
        rownnz[out+k] = 0;
      }
      continue;
    }

    // extract info
    int id = m->actuator_trnid[2*i];
    mjtNum* gear = m->actuator_gear+6*out;

    // process according to transmission type
    switch ((mjtTrn) m->actuator_trntype[i]) {
    case mjTRN_JOINT:                   // joint
    case mjTRN_JOINTINPARENT:           // joint, force in parent frame
      // slide and hinge joint: scalar gear
      if (m->jnt_type[id] == mjJNT_SLIDE || m->jnt_type[id] == mjJNT_HINGE) {
        // sparsity
        rownnz[out] = 1;
        colind[adr] = m->jnt_dofadr[id];

        length[out] = d->qpos[m->jnt_qposadr[id]]*gear[0];
        moment[adr] = gear[0];
      }

      // ball joint: 3D wrench gear
      else if (m->jnt_type[id] == mjJNT_BALL) {
        // axis: expmap representation of quaternion
        mjtNum axis[3], quat[4];
        mji_copy4(quat, d->qpos+m->jnt_qposadr[id]);
        mju_normalize4(quat);
        mji_quat2Vel(axis, quat, 1);

        // gearAxis: rotate to parent frame if necessary
        mjtNum gearAxis[3];
        if (m->actuator_trntype[i] == mjTRN_JOINT) {
          mji_copy3(gearAxis, gear);
        } else {
          mju_negQuat(quat, quat);
          mji_rotVecQuat(gearAxis, gear, quat);
        }

        // length: axis*gearAxis
        length[out] = mju_dot3(axis, gearAxis);

        // dof start address
        int jnt_dofadr = m->jnt_dofadr[id];

        // sparsity
        for (int j = 0; j < 3; j++) {
          colind[adr+j] = jnt_dofadr + j;
        }
        rownnz[out] = 3;

        // moment: gearAxis
        mji_copy3(moment+adr, gearAxis);
      }

      // free joint: 6D wrench gear
      else {
        // cannot compute meaningful length, set to 0
        length[out] = 0;

        // gearAxis: rotate to world frame if necessary
        mjtNum gearAxis[3];
        if (m->actuator_trntype[i] == mjTRN_JOINT) {
          mji_copy3(gearAxis, gear+3);
        } else {
          mjtNum quat[4];
          mji_copy4(quat, d->qpos+m->jnt_qposadr[id]+3);
          mju_normalize4(quat);
          mju_negQuat(quat, quat);
          mji_rotVecQuat(gearAxis, gear+3, quat);
        }

        // dof start address
        int jnt_dofadr = m->jnt_dofadr[id];

        // sparsity
        for (int j = 0; j < 6; j++) {
          colind[adr+j] = jnt_dofadr + j;
        }
        rownnz[out] = 6;

        // moment: gear(tran), gearAxis
        mji_copy3(moment+adr, gear);
        mji_copy3(moment+adr+3, gearAxis);
      }
      break;

    case mjTRN_SLIDERCRANK:             // slider-crank
      {
        // get data
        int idslider = m->actuator_trnid[2*i+1];
        mjtNum rod = m->actuator_cranklength[i];
        mjtNum axis[3] = {d->site_xmat[9 * idslider + 2],
                          d->site_xmat[9 * idslider + 5],
                          d->site_xmat[9 * idslider + 8]};
        mjtNum vec[3];
        mju_sub3(vec, d->site_xpos+3*id, d->site_xpos+3*idslider);

        // compute length and determinant
        //  length = a'*v - sqrt(det);  det = (a'*v)^2 + r^2 - v'*v)
        mjtNum av = mju_dot3(vec, axis);
        mjtNum sdet, det = av*av + rod*rod - mju_dot3(vec, vec);
        int ok = 1;
        if (det <= 0) {
          ok = 0;
          sdet = 0;
          length[out] = av;
        } else {
          sdet = mju_sqrt(det);
          length[out] = av - sdet;
        }

        // compute derivatives of length w.r.t. vec and axis
        mjtNum dlda[3], dldv[3];
        if (ok) {
          mju_scl3(dldv, axis, 1-av/sdet);
          mju_scl3(dlda, vec, 1/sdet);        // use dlda as temp
          mji_addTo3(dldv, dlda);

          mju_scl3(dlda, vec, 1-av/sdet);
        } else {
          mji_copy3(dlda, vec);
          mji_copy3(dldv, axis);
        }

        // get Jacobians of axis(jacA) and vec(jac)
        mj_jacPointAxis(m, d, jacS, jacA, d->site_xpos+3*idslider,
                        axis, m->site_bodyid[idslider]);
        mj_jacSite(m, d, jac, 0, id);
        mju_subFrom(jac, jacS, 3*nv);

        moment_row = mjSTACKALLOC(d, nv, mjtNum);

        // clear moment
        mju_zero(moment_row, nv);

        // apply chain rule
        for (int j=0; j < nv; j++) {
          for (int k=0; k < 3; k++) {
            moment_row[j] += dlda[k]*jacA[k*nv+j] + dldv[k]*jac[k*nv+j];
          }
        }

        // scale by gear ratio
        length[out] *= gear[0];

        // sparsity (compress)
        nnz = 0;
        for (int j = 0; j < nv; j++) {
          if (moment_row[j]) {
            moment[adr+nnz] = moment_row[j] * gear[0];
            colind[adr+nnz] = j;
            nnz++;
          }
        }
        rownnz[out] = nnz;
      }
      break;

    case mjTRN_TENDON:                  // tendon
      length[out] = d->ten_length[id]*gear[0];

      // moment
      {
        int ten_J_rownnz = m->ten_J_rownnz[id];
        int ten_J_rowadr = m->ten_J_rowadr[id];
        rownnz[out] = ten_J_rownnz;
        mju_copyInt(colind + adr, m->ten_J_colind + ten_J_rowadr, ten_J_rownnz);

        mju_scl(moment + adr, d->ten_J + ten_J_rowadr, gear[0], ten_J_rownnz);
      }
      break;

    case mjTRN_SO3:                     // relative orientation: 3 lengths, 3 moment rows
      // ball joint: lengths = expmap of the joint quaternion, rows = identity at the joint dofs
      if (m->actuator_trnid[2*i+1] == -1) {
        mjtNum axis[3], quat[4];
        mji_copy4(quat, d->qpos+m->jnt_qposadr[id]);
        mju_normalize4(quat);
        mji_quat2Vel(axis, quat, 1);
        for (int k=0; k < 3; k++) {
          int outk = out + k;
          rowadr[outk] = k == 0 ? adr : rowadr[outk-1] + rownnz[outk-1];
          length[outk] = axis[k];
          rownnz[outk] = 1;
          colind[rowadr[outk]] = m->jnt_dofadr[id] + k;
          moment[rowadr[outk]] = 1;
        }
      }

      // site+refsite: lengths = relative expmap, rows = relative rotational Jacobian
      else {
        int refid = m->actuator_trnid[2*i+1];
        if (!jacref) jacref = mjSTACKALLOC(d, 3*nv, mjtNum);
        if (!moment_row) moment_row = mjSTACKALLOC(d, nv, mjtNum);

        // relative rotation as expmap in the refsite frame
        mjtNum vec[3];
        mji_subQuat(vec, d->site_xquat+4*id, d->site_xquat+4*refid);

        // relative rotational Jacobian in global frame
        mj_jacSite(m, d, NULL, jacS, id);
        mj_jacSite(m, d, NULL, jacref, refid);
        mju_subFrom(jacS, jacref, 3*nv);

        // if common ancestral dof exists, clear the columns of its parental chain
        {
          int b0 = m->body_weldid[m->site_bodyid[id]];
          int b1 = m->body_weldid[m->site_bodyid[refid]];
          int dofadr0 = m->body_dofadr[b0] + m->body_dofnum[b0] - 1;
          int dofadr1 = m->body_dofadr[b1] + m->body_dofnum[b1] - 1;
          int dofadr_common = -1;
          if (dofadr0 >= 0 && dofadr1 >= 0) {
            while (dofadr0 != dofadr1) {
              if (dofadr0 < dofadr1) {
                dofadr1 = m->dof_parentid[dofadr1];
              } else {
                dofadr0 = m->dof_parentid[dofadr0];
              }
              if (dofadr0 == -1 || dofadr1 == -1) {
                break;
              }
            }
            if (dofadr0 == dofadr1) {
              dofadr_common = dofadr0;
            }
          }
          int da = dofadr_common;
          while (da >= 0) {
            jacS[nv*0 + da] = 0;
            jacS[nv*1 + da] = 0;
            jacS[nv*2 + da] = 0;
            da = m->dof_parentid[da];
          }
        }

        // row k: site axis k in global frame, projected on the relative Jacobian; torque and
        // velocity components are in the site frame, matching the ball branch's child frame
        // (the expmap length components are frame-invariant: exp fixes its own axis)
        for (int k=0; k < 3; k++) {
          int outk = out + k;
          rowadr[outk] = k == 0 ? adr : rowadr[outk-1] + rownnz[outk-1];
          length[outk] = vec[k];

          mjtNum wrench[3] = {d->site_xmat[9*id+k], d->site_xmat[9*id+k+3],
                              d->site_xmat[9*id+k+6]};
          mju_mulMatTVec(moment_row, jacS, wrench, 3, nv);

          // sparsity (compress)
          nnz = 0;
          for (int j=0; j < nv; j++) {
            if (moment_row[j]) {
              moment[rowadr[outk]+nnz] = moment_row[j];
              colind[rowadr[outk]+nnz] = j;
              nnz++;
            }
          }
          rownnz[outk] = nnz;
        }
      }
      break;

    case mjTRN_SITE:                    // site
      // get site translation (jac) and rotation (jacS) Jacobians in global frame
      mj_jacSite(m, d, jac, jacS, id);

      // clear length
      length[out] = 0;

      if (!moment_row) moment_row = mjSTACKALLOC(d, nv, mjtNum);

      // reference site undefined
      if (m->actuator_trnid[2*i+1] == -1) {
        // wrench: gear expressed in global frame
        mjtNum wrench[6];
        mji_mulMatVec3(wrench, d->site_xmat+9*id, gear);      // translation
        mji_mulMatVec3(wrench+3, d->site_xmat+9*id, gear+3);  // rotation

        // moment: global Jacobian projected on wrench
        mju_mulMatTVec(moment_row, jac, wrench, 3, nv);       // translation
        mju_mulMatTVec(jac, jacS, wrench+3, 3, nv);           // rotation
        mju_addTo(moment_row, jac, nv);                       // add the two
      }

      // reference site defined
      else {
        int refid = m->actuator_trnid[2*i+1];
        if (!jacref) jacref = mjSTACKALLOC(d, 3*nv, mjtNum);

        // initialize last dof address for each body
        int b0 = m->body_weldid[m->site_bodyid[id]];
        int b1 = m->body_weldid[m->site_bodyid[refid]];
        int dofadr0 = m->body_dofadr[b0] + m->body_dofnum[b0] - 1;
        int dofadr1 = m->body_dofadr[b1] + m->body_dofnum[b1] - 1;

        // find common ancestral dof, if any
        int dofadr_common = -1;
        if (dofadr0 >= 0 && dofadr1 >= 0) {
          // traverse up the tree until common ancestral dof is found
          while (dofadr0 != dofadr1) {
            if (dofadr0 < dofadr1) {
              dofadr1 = m->dof_parentid[dofadr1];
            } else {
              dofadr0 = m->dof_parentid[dofadr0];
            }
            if (dofadr0 == -1 || dofadr1 == -1) {
              // reached tree root, no common ancestral dof
              break;
            }
          }

          // found common ancestral dof
          if (dofadr0 == dofadr1) {
            dofadr_common = dofadr0;
          }
        }

        // clear moment
        mju_zero(moment_row, nv);

        // translational transmission
        if (!mju_isZero(gear, 3)) {
          // vec: site position in reference site frame
          mjtNum vec[3];
          mju_sub3(vec, d->site_xpos+3*id, d->site_xpos+3*refid);
          mju_mulMatTVec3(vec, d->site_xmat+9*refid, vec);

          // length: dot product with gear
          length[out] += mju_dot3(vec, gear);

          // jacref: global Jacobian of reference site
          mj_jacSite(m, d, jacref, NULL, refid);

          // subtract jacref from jac
          mju_subFrom(jac, jacref, 3*nv);

          // if common ancestral dof was found, clear the columns of its parental chain
          int da = dofadr_common;
          while (da >= 0) {
            jac[nv*0 + da] = 0;
            jac[nv*1 + da] = 0;
            jac[nv*2 + da] = 0;
            da = m->dof_parentid[da];
          }

          // wrench: translational gear expressed in global frame
          mjtNum wrench[6];
          mji_mulMatVec3(wrench, d->site_xmat+9*refid, gear);

          // moment: global Jacobian projected on wrench
          mju_mulMatTVec(moment_row, jac, wrench, 3, nv);
        }

        // rotational transmission
        if (!mju_isZero(gear+3, 3)) {
          // convert difference to expmap (axis-angle)
          mjtNum vec[3];
          mji_subQuat(vec, d->site_xquat+4*id, d->site_xquat+4*refid);

          // add length: dot product with gear
          length[out] += mju_dot3(vec, gear+3);

          // jacref: global rotational Jacobian of reference site
          mj_jacSite(m, d, NULL, jacref, refid);

          // subtract jacref from jacS
          mju_subFrom(jacS, jacref, 3*nv);

          // if common ancestral dof was found, clear the columns of its parental chain
          int da = dofadr_common;
          while (da >= 0) {
            jacS[nv*0 + da] = 0;
            jacS[nv*1 + da] = 0;
            jacS[nv*2 + da] = 0;
            da = m->dof_parentid[da];
          }

          // wrench: rotational gear expressed in global frame
          mjtNum wrench[6];
          mji_mulMatVec3(wrench, d->site_xmat+9*refid, gear+3);

          // global Jacobian projected on wrench, add to moment
          // reuse jac as temporary storage
          mju_mulMatTVec(jac, jacS, wrench, 3, nv);
          mju_addTo(moment_row, jac, nv);
        }
      }

      // sparsity (compress)
      nnz = 0;
      for (int j = 0; j < nv; j++) {
        if (moment_row[j]) {
          moment[adr+nnz] = moment_row[j];
          colind[adr+nnz] = j;
          nnz++;
        }
      }
      rownnz[out] = nnz;

      break;

    case mjTRN_BODY:                  // body (adhesive contacts)
      // cannot compute meaningful length, set to 0
      length[out] = 0;

      // clear moment
      if (!moment_row) moment_row = mjSTACKALLOC(d, nv, mjtNum);
      mju_zero(moment_row, nv);

      // moment is average of all contact normal Jacobians
      {
        // allocate stack variables for the first mjTRN_BODY
        if (!efc_force) {
          efc_force = mjSTACKALLOC(d, d->nefc, mjtNum);
          moment_exclude = mjSTACKALLOC(d, nv, mjtNum);
          jacdifp = mjSTACKALLOC(d, 3*nv, mjtNum);
          jac1p = mjSTACKALLOC(d, 3*nv, mjtNum);
          jac2p = mjSTACKALLOC(d, 3*nv, mjtNum);
          chain = issparse ? mjSTACKALLOC(d, nv, int) : NULL;
        }

        // clear efc_force and moment_exclude
        mju_zero(efc_force, d->nefc);
        mju_zero(moment_exclude, nv);

        // count all relevant contacts, accumulate Jacobians
        int counter = 0, ncon = d->ncon;
        for (int j=0; j < ncon; j++) {
          const mjContact* con = d->contact+j;

          // get geom ids
          int g1 = con->geom[0];
          int g2 = con->geom[1];

          // contact involving flex, continue
          if (g1 < 0 || g2 < 0) {
            continue;
          }

          // get body ids
          int b1 = m->geom_bodyid[g1];
          int b2 = m->geom_bodyid[g2];

          // irrelevant contact, continue
          if (b1 != id && b2 != id) {
            continue;
          }

          // mark contact normals in efc_force
          if (!con->exclude) {
            if (con->efc_address < 0) {
              continue;
            }
            counter++;

            // condim 1 or elliptic cones: normal is in the first row
            if (con->dim == 1 || m->opt.cone == mjCONE_ELLIPTIC) {
              efc_force[con->efc_address] = 1;
            }

            // pyramidal cones: average all pyramid directions
            else {
              int npyramid = con->dim-1;  // number of frictional directions
              for (int k=0; k < 2*npyramid; k++) {
                efc_force[con->efc_address+k] = 0.5/npyramid;
              }
            }
          }

          // excluded contact in gap: get sparse or dense Jacobian, accumulate
          else if (con->exclude == 1) {
            counter++;

            // get Jacobian difference
            int NV = mj_jacDifPair(m, d, chain, b1, b2, con->pos, con->pos,
                                   jac1p, jac2p, jacdifp, NULL, NULL, NULL, issparse,
                                   /*flg_skipcommon=*/0);

            // project Jacobian along the normal of the contact frame
            mju_mulMatMat(jac, con->frame, jacdifp, 1, 3, NV);

            // accumulate in moment_exclude
            if (issparse) {
              for (int k=0; k < NV; k++) {
                moment_exclude[chain[k]] += jac[k];
              }
            } else {
              mju_addTo(moment_exclude, jac, nv);
            }
          }
        }

        // moment is average over contact normal Jacobians, make negative for adhesion
        if (counter) {
          // accumulate active contact Jacobians into moment
          mj_mulJacTVec(m, d, moment_row, efc_force);

          // add Jacobians from excluded contacts
          mju_addTo(moment_row, moment_exclude, nv);

          // normalize by total contacts, flip sign
          mju_scl(moment_row, moment_row, -1.0/counter, nv);
        }
      }

      // sparsity (compress)
      nnz = 0;
      for (int j = 0; j < nv; j++) {
        if (moment_row[j]) {
          moment[adr+nnz] = moment_row[j];
          colind[adr+nnz] = j;
          nnz++;
        }
      }
      rownnz[out] = nnz;

      break;

    default:
      mjERROR("unknown transmission type %d", m->actuator_trntype[i]);  // SHOULD NOT OCCUR
    }
  }

  mj_freeStack(d);
}


//-------------------------- inertia ---------------------------------------------------------------

// add tendon armature to M
void mj_tendonArmature(const mjModel* m, mjData* d) {
  int nv = m->nv, ntendon = m->ntendon;
  const int* M_rownnz = m->M_rownnz;
  const int* M_rowadr = m->M_rowadr;
  const int* M_colind = m->M_colind;

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < nv;

  for (int k=0; k < ntendon; k++) {
    // skip sleeping tendon
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, k) == mjS_ASLEEP) {
      continue;
    }

    mjtNum armature = m->tendon_armature[k] + mj_actuatorArmature(m, mjOBJ_TENDON, k);
    if (!armature) {
      continue;
    }

    // get sparse info for tendon k
    int J_rowadr = m->ten_J_rowadr[k];
    int J_rownnz = m->ten_J_rownnz[k];
    const int* J_colind = m->ten_J_colind + J_rowadr;
    mjtNum* ten_J = d->ten_J + J_rowadr;

    // M += armature * ten_J' * ten_J
    for (int j=0; j < J_rownnz; j++) {
      mjtNum ten_J_i = ten_J[j];
      if (!ten_J_i) {
        continue;
      }

      // M[i,:] += armature * ten_J[i] * ten_J
      int i = J_colind[j];
      int M_adr = M_rowadr[i];
      mju_addToSclSparseInc(d->M + M_adr, ten_J,
                            M_rownnz[i], M_colind + M_adr,
                            J_rownnz, J_colind, armature * ten_J_i);
    }
  }
}


// composite rigid body inertia algorithm
void mj_crb(const mjModel* m, mjData* d) {
  // outputs
  mjtNum* crb = d->crb;
  mjtNum* M   = d->M;

  // inputs
  const mjtNum* cinert        = d->cinert;
  const mjtNum* cdof          = d->cdof;
  const mjtNum* dof_M0        = m->dof_M0;
  const mjtNum* dof_armature  = m->dof_armature;
  const int* body_awake_ind   = d->body_awake_ind;
  const int* parent_awake_ind = d->parent_awake_ind;
  const int* dof_awake_ind    = d->dof_awake_ind;
  const int* rownnz           = m->M_rownnz;
  const int* rowadr           = m->M_rowadr;
  const int* body_parentid    = m->body_parentid;
  const int* dof_parentid     = m->dof_parentid;
  const int* dof_simplenum    = m->dof_simplenum;
  const int* dof_bodyid       = m->dof_bodyid;

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  int nparent = sleep_filter ? d->nparent_awake : m->nbody;
  int nv = sleep_filter ? d->nv_awake : m->nv;

  // crb = cinert
  if (!sleep_filter) {
    mju_copy(crb, cinert, 10*nbody);
  } else {
    mju_copyRows(crb, cinert, body_awake_ind, nbody, 10);
  }

  // backward pass over bodies, accumulate composite inertias
  for (int b = nparent - 1; b >= 0; b--) {
    int i = sleep_filter ? parent_awake_ind[b] : b;
    if (body_parentid[i] > 0) {
      mju_addTo(crb + 10*body_parentid[i], crb + 10*i, 10);
    }
  }

  // clear M
  if (!sleep_filter) {
    mju_zero(M, m->nC);
  } else {
    mju_zeroSparse(M, rownnz, rowadr, dof_awake_ind, nv);
  }

  // dense forward pass over dofs
  for (int v=0; v < nv; v++) {
    int i = sleep_filter ? dof_awake_ind[v] : v;

    // simple dof: fixed diagonal inertia
    int adr = rowadr[i];
    if (dof_simplenum[i]) {
      M[adr] = dof_M0[i];
      continue;
    }

    // init M(i,i) with armature inertia
    int Madr_ij = adr + rownnz[i] - 1;
    M[Madr_ij] = dof_armature[i] + mj_actuatorArmature(m, mjOBJ_JOINT, m->dof_jntid[i]);

    // precompute buf = crb_body_i * cdof_i
    mjtNum buf[6];
    mju_mulInertVec(buf, crb+10*dof_bodyid[i], cdof+6*i);

    // sparse backward pass over ancestors
    for (int j=i; j >= 0; j = dof_parentid[j]) {
      // M(i,j) += cdof_j * (crb_body_i * cdof_i)
      M[Madr_ij--] += mji_dot6(cdof+6*j, buf);
    }
  }
}


void mj_makeM(const mjModel* m, mjData* d) {
  TM_START;
  mj_crb(m, d);
  mj_tendonArmature(m, d);
  TM_END(mjTIMER_POS_INERTIA);
}


// sparse L'*D*L factorizaton of the inertia matrix M, assumed spd
void mj_factorM(const mjModel* m, mjData* d) {
  TM_START;

  // sleep filtering
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nv_awake < m->nv;
  const int* index;
  int nv;

  // no sleep filtering: copy everything
  if (!sleep_filter) {
    index = NULL;
    nv = m->nv;
    mju_copy(d->qLD, d->M, m->nC);
  }

  // sleep filtering: copy only awake dofs
  else {
    index = d->dof_awake_ind;
    nv = d->nv_awake;
    mju_copySparse(d->qLD, d->M, m->M_rownnz, m->M_rowadr, d->dof_awake_ind, d->nv_awake);
  }

  // factorize
  int clamped = mj_factorI(d->qLD, d->qLDiagInv, nv, m->M_rownnz, m->M_rowadr, m->M_colind,
                           index);

  // near-singular inertia: a pivot was clamped
  if (clamped >= 0) {
    mj_warning(d, mjWARN_INERTIA, clamped);
  }

  TM_ADD(mjTIMER_POS_INERTIA);
}


// sparse L'*D*L factorizaton of inertia-like matrix M, assumed spd (with dof skipping);
// clamp non-positive pivots up to mjMINVAL, return first clamped dof index or -1 if none
int mj_factorI(mjtNum* mat, mjtNum* diaginv, int nv,
               const int* rownnz, const int* rowadr, const int* colind,
               const int* index) {
  int clamped = -1;

  // backward loop over rows
  for (int j=nv-1; j >= 0; j--) {
    int k = index ? index[j] : j;

    // get row k's address, diagonal index
    int start = rowadr[k];
    int diag = rownnz[k] - 1;
    int end = start + diag;

    // clamp small or non-positive pivot from below, save first clamped dof
    if (mat[end] < mjMINVAL) {
      mat[end] = mjMINVAL;
      if (clamped < 0) {
        clamped = k;
      }
    }

    // inverse diagonal value
    mjtNum invD = 1 / mat[end];
    if (diaginv) diaginv[k] = invD;

    // update triangle above row k
    for (int adr=end - 1; adr >= start; adr--) {
      // update row i < k:  L(i, 0..i) -= L(i, 0..i) * L(k, i) / L(k, k)
      int i = colind[adr];
      mju_addToScl(mat + rowadr[i], mat + start, -mat[adr] * invD, rownnz[i]);
    }

    // update row k:  L(k, :) /= L(k, k)
    mju_scl(mat + start, mat + start, invD, diag);
  }

  return clamped;
}


// in-place sparse backsubstitution:  x = inv(L'*D*L)*x  (with dof skipping)
void mj_solveLD(mjtNum* restrict x, const mjtNum* qLD, const mjtNum* qLDiagInv, int nv, int n,
                const int* rownnz, const int* rowadr, const int* colind, const int* index) {
  // x <- L^-T x
  for (int k = nv - 1; k >= 0; k--) {
    int i = index ? index[k] : k;

    // skip diagonal rows
    if (rownnz[i] == 1) {
      continue;
    }

    // one vector
    if (n == 1) {
      mjtNum x_i;
      if ((x_i = x[i])) {
        int start = rowadr[i];
        int end = start + rownnz[i] - 1;
        for (int adr=start; adr < end; adr++) {
          x[colind[adr]] -= qLD[adr] * x_i;
        }
      }
    }

    // multiple vectors
    else {
      int start = rowadr[i];
      int end = start + rownnz[i] - 1;
      for (int offset=0; offset < n*nv; offset+=nv) {
        mjtNum x_i;
        if ((x_i = x[i+offset])) {
          for (int adr=start; adr < end; adr++) {
            x[offset + colind[adr]] -= qLD[adr] * x_i;
          }
        }
      }
    }
  }

  // x <- D^-1 x
  for (int k = 0; k < nv; k++) {
    int i = index ? index[k] : k;

    mjtNum invD_i = qLDiagInv[i];

    // one vector
    if (n == 1) {
      x[i] *= invD_i;
    }

    // multiple vectors
    else {
      for (int offset=0; offset < n*nv; offset+=nv) {
        x[i+offset] *= invD_i;
      }
    }
  }

  // x <- L^-1 x
  for (int k = 0; k < nv; k++) {
    int i = index ? index[k] : k;

    // skip diagonal rows
    if (rownnz[i] == 1) {
      continue;
    }

    int d;
    if ((d = rownnz[i] - 1) > 0) {
      int adr = rowadr[i];

      // one vector
      if (n == 1) {
        x[i] -= mju_dotSparse(qLD+adr, x, d, colind+adr);
      }

      // multiple vectors
      else {
        for (int offset=0; offset < n*nv; offset+=nv) {
          x[i+offset] -= mju_dotSparse(qLD+adr, x+offset, d, colind+adr);
        }
      }
    }
  }
}


// sparse backsubstitution:  x = inv(L'*D*L)*y
//  use factorization in d
void mj_solveM(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y, int n) {
  if (x != y) {
    mju_copy(x, y, n*m->nv);
  }
  mj_solveLD(x, d->qLD, d->qLDiagInv, m->nv, n, m->M_rownnz, m->M_rowadr, m->M_colind, NULL);
}


// half of sparse backsubstitution:  x = sqrt(inv(D))*inv(L')*y
void mj_solveM2(const mjModel* m, mjData* d, mjtNum* x, const mjtNum* y,
                const mjtNum* sqrtInvD, int n) {
  mj_solveM2_impl(x, y, sqrtInvD, d->qLD, m->nv, n,
                  m->M_rownnz, m->M_rowadr, m->M_colind, m->dof_simplenum);
}


// implementation of mj_solveM2: backsubstitute against a caller-selected factor (qLD or qH)
void mj_solveM2_impl(mjtNum* x, const mjtNum* y, const mjtNum* sqrtInvD, const mjtNum* factor,
                     int nv, int n, const int* rownnz, const int* rowadr, const int* colind,
                     const int* diagnum) {
  const mjtNum* qLD = factor;

  // x = y
  mju_copy(x, y, n * nv);

  // x <- L^-T x
  for (int i=nv-1; i > 0; i--) {
    // skip diagonal rows
    if (diagnum[i]) {
      continue;
    }

    // prepare row i column address range
    int start = rowadr[i];
    int end = start + rownnz[i] - 1;

    // process all vectors
    for (int offset=0; offset < n*nv; offset+=nv) {
      mjtNum x_i;
      if ((x_i = x[i+offset])) {
        for (int adr=start; adr < end; adr++) {
          x[offset + colind[adr]] -= qLD[adr] * x_i;
        }
      }
    }
  }

  // x <- D^-1/2 x
  for (int i=0; i < nv; i++) {
    mjtNum invD_i = sqrtInvD[i];
    for (int offset=0; offset < n*nv; offset+=nv) {
      x[i+offset] *= invD_i;
    }
  }
}


//---------------------------------- velocity ------------------------------------------------------

// compute cvel, cdof_dot
void mj_comVel(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  // set world vel to 0
  mju_zero(d->cvel, 6);

  // forward pass over bodies
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // cvel = cvel_parent
    mjtNum cvel[6];
    mji_copy6(cvel, d->cvel+6*m->body_parentid[i]);

    // cvel = cvel_parent + cdof * qvel,  cdofdot = cvel x cdof
    int dofnum = m->body_dofnum[i];
    int bda = m->body_dofadr[i];
    mjtNum cdofdot[36];
    for (int j=0; j < dofnum; j++) {
      mjtNum tmp[6];

      // compute cvel and cdofdot
      switch ((mjtJoint) m->jnt_type[m->dof_jntid[bda+j]]) {
      case mjJNT_FREE:
        // cdofdot = 0
        mju_zero(cdofdot, 18);

        // update velocity
        mju_mulDofVec(tmp, d->cdof+6*bda, d->qvel+bda, 3);
        mju_addTo(cvel, tmp, 6);

        // continue with rotations
        j += 3;
        mjFALLTHROUGH;

      case mjJNT_BALL:
        // compute all 3 cdofdots using parent velocity
        mji_crossMotion(cdofdot+6*(j+0), cvel, d->cdof+6*(bda+j+0));
        mji_crossMotion(cdofdot+6*(j+1), cvel, d->cdof+6*(bda+j+1));
        mji_crossMotion(cdofdot+6*(j+2), cvel, d->cdof+6*(bda+j+2));

        // update velocity
        mju_mulDofVec(tmp, d->cdof+6*(bda+j), d->qvel+bda+j, 3);
        mju_addTo(cvel, tmp, 6);

        // adjust for 3-dof joint
        j += 2;
        break;

      default:
        // in principle we should use the new velocity to compute cdofdot,
        // but it makes no difference because crossMotion(cdof, cdof) = 0,
        // and using the old velocity may be more accurate numerically
        mji_crossMotion(cdofdot+6*j, cvel, d->cdof+6*(bda+j));

        // update velocity
        mju_mulDofVec(tmp, d->cdof+6*(bda+j), d->qvel+bda+j, 1);
        mju_addTo(cvel, tmp, 6);
      }
    }

    // assign cvel, cdofdot
    mji_copy6(d->cvel+6*i, cvel);
    mju_copy(d->cdof_dot+6*bda, cdofdot, 6*dofnum);
  }
}


// subtree linear velocity and angular momentum
void mj_subtreeVel(const mjModel* m, mjData* d) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;

  mj_markStack(d);
  mjtNum* body_vel = mjSTACKALLOC(d, 6*m->nbody, mjtNum);

  // bodywise quantities
  for (int b=0; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // compute and save body velocity
    mj_objectVelocity(m, d, mjOBJ_BODY, i, body_vel+6*i, 0);

    // body linear momentum
    mju_scl3(d->subtree_linvel+3*i, body_vel+6*i+3, m->body_mass[i]);

    // body angular momentum
    mjtNum dv[3];
    mju_mulMatTVec3(dv, d->ximat+9*i, body_vel+6*i);
    dv[0] *= m->body_inertia[3*i];
    dv[1] *= m->body_inertia[3*i+1];
    dv[2] *= m->body_inertia[3*i+2];
    mji_mulMatVec3(d->subtree_angmom+3*i, d->ximat+9*i, dv);
  }

  // subtree linear velocity
  for (int b=nbody-1; b >= 0; b--) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // non-world: add linear momentum to parent
    if (i) {
      mji_addTo3(d->subtree_linvel+3*m->body_parentid[i], d->subtree_linvel+3*i);
    }

    // convert linear momentum to linear velocity
    mju_scl3(d->subtree_linvel+3*i, d->subtree_linvel+3*i,
             1/mju_max(mjMINVAL, m->body_subtreemass[i]));
  }

  // subtree angular momentum
  for (int b=nbody-1; b > 0; b--) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    int parent = m->body_parentid[i];

    // momentum wrt body i
    mjtNum dx[3], dv[3], dp[3], dL[3];
    mju_sub3(dx, d->xipos+3*i, d->subtree_com+3*i);
    mju_sub3(dv, body_vel+6*i+3, d->subtree_linvel+3*i);
    mju_scl3(dp, dv, m->body_mass[i]);
    mji_cross(dL, dx, dp);

    // add to subtree i
    mji_addTo3(d->subtree_angmom+3*i, dL);

    // add to parent
    mji_addTo3(d->subtree_angmom+3*parent, d->subtree_angmom+3*i);

    // momentum wrt parent
    mju_sub3(dx, d->subtree_com+3*i, d->subtree_com+3*parent);
    mju_sub3(dv, d->subtree_linvel+3*i, d->subtree_linvel+3*parent);
    mju_scl3(dv, dv, m->body_subtreemass[i]);
    mji_cross(dL, dx, dv);

    // add to parent
    mji_addTo3(d->subtree_angmom+3*parent, dL);
  }

  mj_freeStack(d);

  // mark as computed
  d->flg_subtreevel = 1;
}


//---------------------------------- RNE -----------------------------------------------------------

// RNE: compute M(qpos)*qacc + C(qpos,qvel); flg_acc=0 removes inertial term
void mj_rne(const mjModel* m, mjData* d, int flg_acc, mjtNum* result) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->nbody_awake < m->nbody;
  int nbody = sleep_filter ? d->nbody_awake : m->nbody;
  int nparent = sleep_filter ? d->nparent_awake : m->nbody;
  int nv = sleep_filter ? d->nv_awake : m->nv;

  mj_markStack(d);
  mjtNum* loc_cacc = mjSTACKALLOC(d, m->nbody*6, mjtNum);
  mjtNum* loc_cfrc_body = mjSTACKALLOC(d, m->nbody*6, mjtNum);

  // set world acceleration to -gravity
  mju_zero(loc_cacc, 6);
  if (!mjDISABLED(mjDSBL_GRAVITY)) {
    mju_scl3(loc_cacc+3, m->opt.gravity, -1);
  }

  // forward pass over bodies: accumulate cacc, set cfrc_body
  for (int b=1; b < nbody; b++) {
    int i = sleep_filter ? d->body_awake_ind[b] : b;

    // get body's first dof address
    int bda = m->body_dofadr[i];

    // cacc = cacc_parent + cdofdot * qvel
    mjtNum tmp[6];
    mju_mulDofVec(tmp, d->cdof_dot+6*bda, d->qvel+bda, m->body_dofnum[i]);
    mju_add(loc_cacc+6*i, loc_cacc+6*m->body_parentid[i], tmp, 6);

    // cacc += cdof * qacc
    if (flg_acc) {
      mju_mulDofVec(tmp, d->cdof+6*bda, d->qacc+bda, m->body_dofnum[i]);
      mju_addTo(loc_cacc+6*i, tmp, 6);
    }

    // cfrc_body = cinert * cacc + cvel x (cinert * cvel)
    mju_mulInertVec(loc_cfrc_body+6*i, d->cinert+10*i, loc_cacc+6*i);
    mju_mulInertVec(tmp, d->cinert+10*i, d->cvel+6*i);
    mjtNum tmp1[6];
    mji_crossForce(tmp1, d->cvel+6*i, tmp);
    mju_addTo(loc_cfrc_body+6*i, tmp1, 6);
  }

  // clear world cfrc_body
  mju_zero(loc_cfrc_body, 6);

  // backward pass over bodies: accumulate cfrc_body from children
  for (int b=nparent-1; b > 0; b--) {
    int i = sleep_filter ? d->parent_awake_ind[b] : b;
    int j = m->body_parentid[i];

    if (j) {
      mju_addTo(loc_cfrc_body+6*j, loc_cfrc_body+6*i, 6);
    }
  }

  // result = cdof * cfrc_body
  for (int v=0; v < nv; v++) {
    int i = sleep_filter ? d->dof_awake_ind[v] : v;
    result[i] = mji_dot6(d->cdof+6*i, loc_cfrc_body+6*m->dof_bodyid[i]);
  }

  mj_freeStack(d);
}


// add world-oriented wrench (torque:force) applied at point to cfrc_ext of body, scaled
static void addExternal(const mjModel* m, mjData* d, int body, const mjtNum point[3],
                        const mjtNum cfrc[6], mjtNum scale) {
  // nothing to accumulate on the world
  if (!body) {
    return;
  }

  // map to the subtree com of the body's root, accumulate
  mjtNum cfrc_com[6];
  mju_transformSpatial(cfrc_com, cfrc, 1, d->subtree_com+3*m->body_rootid[body], point, 0);
  mju_addToScl(d->cfrc_ext+6*body, cfrc_com, scale, 6);
}


// add the force along spatial tendon t to cfrc_ext of the bodies on its path
static void addTendon(const mjModel* m, mjData* d, int t, mjtNum frc) {
  int adr = m->tendon_adr[t], num = m->tendon_num[t];
  int p = d->ten_wrapadr[t], pend = p + d->ten_wrapnum[t];
  mjtNum divisor = 1;

  // previous point of the branch, none at the start of a branch
  int prevbody = -1;
  const mjtNum* prevpnt = NULL;

  // walk the wrap objects in lockstep with the path computed by mj_tendon
  for (int j=adr; j < adr+num; j++) {
    int type = m->wrap_type[j], objid = m->wrap_objid[j];

    // pulley: divides the force in the next branch, skip its marker in the path
    if (type == mjWRAP_PULLEY) {
      divisor = m->wrap_prm[j];
      prevbody = -1;
      p++;
      continue;
    }

    // site: one point; geom: two points if the tendon wraps around it
    int npnt, body;
    if (type == mjWRAP_SITE) {
      npnt = 1;
      body = m->site_bodyid[objid];
    } else {
      npnt = (p < pend && d->wrap_obj[p] == objid) ? 2 : 0;
      body = m->geom_bodyid[objid];
    }

    // straight segments between points on different bodies carry the force
    for (int k=0; k < npnt; k++) {
      const mjtNum* pnt = d->wrap_xpos + 3*p;
      if (prevbody >= 0 && body != prevbody) {
        mjtNum cfrc[6] = {0, 0, 0, 0, 0, 0};
        mji_sub3(cfrc+3, pnt, prevpnt);
        if (mju_normalize3(cfrc+3) >= mjMINVAL) {
          mju_scl3(cfrc+3, cfrc+3, frc/divisor);
          addExternal(m, d, body, pnt, cfrc, 1);
          addExternal(m, d, prevbody, prevpnt, cfrc, -1);
        }
      }
      prevbody = body;
      prevpnt = pnt;
      p++;
    }
  }
}


// RNE with complete data: compute cacc, cfrc_ext, cfrc_int
void mj_rnePostConstraint(const mjModel* m, mjData* d) {
  int nbody = m->nbody, ntendon = m->ntendon, nactuator = m->nactuator;
  mjtNum cfrc[6], lfrc[6], *ten_frc = NULL;
  int* ten_nlimit = NULL;
  mjContact* con;

  // clear cacc, set world acceleration to -gravity
  mju_zero(d->cacc, 6);
  if (!mjDISABLED(mjDSBL_GRAVITY)) {
    mju_scl3(d->cacc+3, m->opt.gravity, -1);
  }

  // total force along each tendon: constraint, actuator, passive and armature
  if (ntendon) {
    mj_markStack(d);
    ten_frc = mjSTACKALLOC(d, ntendon, mjtNum);
    ten_nlimit = mjSTACKALLOC(d, ntendon, int);
    mju_zero(ten_frc, ntendon);
    mju_zeroInt(ten_nlimit, ntendon);
  }

  // cfrc_ext = perturb
  mju_zero(d->cfrc_ext, 6*nbody);
  for (int i=1; i < nbody; i++) {
    if (!mju_isZero(d->xfrc_applied+6*i, 6)) {
      // rearrange as torque:force, apply at body com
      mji_copy3(cfrc, d->xfrc_applied+6*i+3);
      mji_copy3(cfrc+3, d->xfrc_applied+6*i);
      addExternal(m, d, i, d->xipos+3*i, cfrc, 1);
    }
  }

  // cfrc_ext += contacts
  int ncon = d->ncon;
  for (int i=0; i < ncon; i++) {
    // get contact pointer
    con = d->contact+i;

    // skip excluded contacts
    if (con->efc_address < 0) {
      continue;
    }

    // skip contact involving flex
    if (con->geom[0] < 0 || con->geom[1] < 0) {
      continue;
    }

    // tmp = contact-local force:torque vector
    mj_contactForce(m, d, i, lfrc);

    // cfrc = world-oriented torque:force vector (swap in the process)
    mju_mulMatTVec3(cfrc, con->frame, lfrc+3);
    mju_mulMatTVec3(cfrc+3, con->frame, lfrc);

    // apply at the contact point, opposite for body 1
    addExternal(m, d, m->geom_bodyid[con->geom[0]], con->pos, cfrc, -1);
    addExternal(m, d, m->geom_bodyid[con->geom[1]], con->pos, cfrc, 1);
  }

  // cfrc_ext += connect, weld, flex constraints
  int i = 0, ne = d->ne;
  while (i < ne) {
    if (d->efc_type[i] != mjCNSTR_EQUALITY) {
      mjERROR("row %d of efc is not an equality constraint", i);  // SHOULD NOT OCCUR
    }

    int id = d->efc_id[i];
    mjtNum* eq_data = m->eq_data + mjNEQDATA*id;
    mjtNum pos[3], *offset;
    int k, obj1, obj2, body_semantic;
    switch ((mjtEq) m->eq_type[id]) {
    case mjEQ_CONNECT:
    case mjEQ_WELD:
      body_semantic = m->eq_objtype[id] == mjOBJ_BODY;
      obj1 = m->eq_obj1id[id];
      obj2 = m->eq_obj2id[id];

      // cfrc = world-oriented torque:force vector
      mji_copy3(cfrc + 3, d->efc_force + i);
      if (m->eq_type[id] == mjEQ_CONNECT) {
        mju_zero3(cfrc);  // no torque from connect
      } else {
        // the rotational rows are 0.5*torquescale * neg(q1)*(jac0-jac1)*q0*relpose (see
        // mj_instantiateEquality), so the torque is the adjoint applied to the multiplier:
        // 0.5*torquescale * q1*force*neg(q0*relpose), with site frames as q1 and q0*relpose
        mjtNum q0[4], q1[4], quat[4];
        if (body_semantic) {
          mju_mulQuat(q0, d->xquat+4*obj1, eq_data+6);
          mji_copy4(q1, d->xquat+4*obj2);
        } else {
          mju_mulQuat(q0, d->xquat+4*m->site_bodyid[obj1], m->site_quat+4*obj1);
          mju_mulQuat(q1, d->xquat+4*m->site_bodyid[obj2], m->site_quat+4*obj2);
        }
        mju_negQuat(q0, q0);
        mju_mulQuatAxis(quat, q1, d->efc_force+i+3);
        mju_mulQuat(quat, quat, q0);
        mju_scl3(cfrc, quat+1, 0.5*eq_data[10]);
      }

      // body 1
      k = body_semantic ? obj1 : m->site_bodyid[obj1];
      if (k) {
        offset = body_semantic ? eq_data + 3 * (m->eq_type[id] == mjEQ_WELD) :
                                 m->site_pos + 3 * obj1;

        // transform point on body1: local -> global, apply
        mj_local2Global(d, pos, 0, offset, 0, k, 0);
        addExternal(m, d, k, pos, cfrc, 1);
      }

      // body 2
      k = body_semantic ? obj2 : m->site_bodyid[obj2];
      if (k) {
        offset = body_semantic ? eq_data + 3 * (m->eq_type[id] == mjEQ_CONNECT) :
                                 m->site_pos + 3 * obj2;

        // transform point on body2: local -> global, apply (opposite for body 2)
        mj_local2Global(d, pos, 0, offset, 0, k, 0);
        addExternal(m, d, k, pos, cfrc, -1);
      }

      // increment rows
      i += m->eq_type[id] == mjEQ_WELD ? 6 : 3;
      break;

    case mjEQ_JOINT:
      // increment 1 row
      i++;
      break;

    case mjEQ_TENDON:
      // force along tendon 1, and along tendon 2 through the derivative of the coupling
      obj1 = m->eq_obj1id[id];
      obj2 = m->eq_obj2id[id];
      ten_frc[obj1] += d->efc_force[i];
      if (obj2 >= 0) {
        mjtNum dif = d->ten_length[obj2] - m->tendon_length0[obj2];
        mjtNum deriv = eq_data[1] + 2*eq_data[2]*dif + 3*eq_data[3]*dif*dif +
                       4*eq_data[4]*dif*dif*dif;
        ten_frc[obj2] -= deriv * d->efc_force[i];
      }

      // increment 1 row
      i++;
      break;

    case mjEQ_FLEX:
      // increment with number of non-rigid edges
      k = m->eq_obj1id[id];
      int flex_edgeadr = m->flex_edgeadr[k];
      int flex_edgenum = m->flex_edgenum[k];

      for (int e=flex_edgeadr; e < flex_edgeadr+flex_edgenum; e++) {
        if (!m->flexedge_rigid[e]) {
          i++;
        }
      }
      break;

    case mjEQ_FLEXSTRAIN:
      // advance past all rows instantiated for this strain constraint
      while (i < ne && d->efc_id[i] == id) {
        i++;
      }
      break;

    default:
      mjERROR("unknown constraint type type %d", m->eq_type[id]);    // SHOULD NOT OCCUR
    }
  }

  if (ntendon) {
    // ten_frc += friction and limit rows; the lower limit row precedes the upper
    int nf = d->nf, nl = d->nl;
    for (int r=ne; r < ne+nf+nl; r++) {
      int t = d->efc_id[r];
      if (d->efc_type[r] == mjCNSTR_FRICTION_TENDON) {
        ten_frc[t] += d->efc_force[r];
      } else if (d->efc_type[r] == mjCNSTR_LIMIT_TENDON) {
        // Jacobian is -side*ten_J, the lower limit (side -1) comes first if active
        int lower = ten_nlimit[t]++ == 0 &&
                    -(m->tendon_range[2*t] - d->ten_length[t]) < m->tendon_margin[t];
        ten_frc[t] += lower ? d->efc_force[r] : -d->efc_force[r];
      }
    }

    // ten_frc += actuators
    for (int j=0; j < nactuator; j++) {
      if (m->actuator_trntype[j] == mjTRN_TENDON) {
        int out = m->actuator_outadr[j];
        ten_frc[m->actuator_trnid[2*j]] += d->actuator_force[out] * m->actuator_gear[6*out];
      }
    }

    // cfrc_ext += spatial tendons
    int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
    for (int t=0; t < ntendon; t++) {
      // fixed tendon: acts through the joints
      if (m->tendon_type[t] == mjTENDON_FIXED) {
        continue;
      }

      // sleeping tendon: no path
      if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, t) == mjS_ASLEEP) {
        continue;
      }

      // ten_frc += spring and damper
      mjtNum frc_spring, frc_damper;
      mj_tendonSpringDamper(m, d, t, &frc_spring, &frc_damper);
      ten_frc[t] += frc_spring + frc_damper;

      // ten_frc -= armature * tendon acceleration: reaction of the armature inertia
      mjtNum armature = m->tendon_armature[t] + mj_actuatorArmature(m, mjOBJ_TENDON, t);
      if (armature) {
        mjtNum acc = mj_tendonDot(m, d, t, d->qvel);
        int end = m->ten_J_rowadr[t] + m->ten_J_rownnz[t];
        for (int j=m->ten_J_rowadr[t]; j < end; j++) {
          acc += d->ten_J[j] * d->qacc[m->ten_J_colind[j]];
        }
        ten_frc[t] -= armature * acc;
      }

      // apply along the path
      if (ten_frc[t]) {
        addTendon(m, d, t, ten_frc[t]);
      }
    }

    mj_freeStack(d);
  }

  // forward pass over bodies: compute cacc, cfrc_int
  mjtNum cacc[6], cfrc_body[6], cfrc_corr[6];
  mju_zero(d->cfrc_int, 6);
  for (int j=1; j < nbody; j++) {
    // get body's first dof address
    int bda = m->body_dofadr[j];

    // cacc = cacc_parent + cdofdot * qvel + cdof * qacc
    mju_mulDofVec(cacc, d->cdof_dot+6*bda, d->qvel+bda, m->body_dofnum[j]);
    mju_add(d->cacc+6*j, d->cacc+6*m->body_parentid[j], cacc, 6);
    mju_mulDofVec(cacc, d->cdof+6*bda, d->qacc+bda, m->body_dofnum[j]);
    mju_addTo(d->cacc+6*j, cacc, 6);

    // cfrc_body = cinert * cacc + cvel x (cinert * cvel)
    mju_mulInertVec(cfrc_body, d->cinert+10*j, d->cacc+6*j);
    mju_mulInertVec(cfrc_corr, d->cinert+10*j, d->cvel+6*j);
    mji_crossForce(cfrc, d->cvel+6*j, cfrc_corr);
    mju_addTo(cfrc_body, cfrc, 6);

    // set cfrc_int = cfrc_body - cfrc_ext
    mju_sub(d->cfrc_int+6*j, cfrc_body, d->cfrc_ext+6*j, 6);
  }

  // backward pass over bodies: accumulate cfrc_int from children
  for (int j=nbody-1; j > 0; j--) {
    mju_addTo(d->cfrc_int+6*m->body_parentid[j], d->cfrc_int+6*j, 6);
  }

  // mark as computed
  d->flg_rnepost = 1;
}


// add bias force due to tendon armature
void mj_tendonBias(const mjModel* m, mjData* d, mjtNum* qfrc) {
  int sleep_filter = mjENABLED(mjENBL_SLEEP) && d->ntree_awake < m->ntree;
  int ntendon = m->ntendon;

  // add bias term due to tendon armature
  for (int i=0; i < ntendon; i++) {
    // skip sleeping tendon
    if (sleep_filter && mj_sleepState(m, d, mjOBJ_TENDON, i) == mjS_ASLEEP) {
      continue;
    }

    mjtNum armature = m->tendon_armature[i] + mj_actuatorArmature(m, mjOBJ_TENDON, i);

    // no armature: skip
    if (!armature) {
      continue;
    }

    // get d/dt(tendon Jacobian) dotted with qvel for tendon i
    mjtNum dot = mj_tendonDot(m, d, i, d->qvel);

    // add bias term:  qfrc += ten_J * armature * dot(ten_Jdot, qvel)
    mjtNum coef = armature * dot;

    if (coef) {
      // sparse
      int nnz = m->ten_J_rownnz[i];
      int adr = m->ten_J_rowadr[i];
      const int* colind = m->ten_J_colind + adr;
      const mjtNum* ten_J = d->ten_J + adr;
      for (int j=0; j < nnz; j++) {
        qfrc[colind[j]] += coef * ten_J[j];
      }
    }
  }
}
