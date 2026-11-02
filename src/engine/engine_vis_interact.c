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

#include "engine/engine_vis_interact.h"

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjsan.h>  // IWYU pragma: keep
#include <mujoco/mjvisualize.h>
#include "engine/engine_core_smooth.h"
#include "engine/engine_core_util.h"
#include "engine/engine_memory.h"
#include "engine/engine_ray.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"
#include "engine/engine_vis_visualize.h"

// transform pose from room to model space
void mjv_room2model(mjtNum* modelpos, mjtNum* modelquat, const mjtNum* roompos,
                    const mjtNum* roomquat, const mjvScene* scn) {
  mjtNum translate[3], rotate[4], invpos[3], invquat[4];

  // check scale
  if (scn->scale < mjMINVAL) {
    mjERROR("mjvScene scale too small");
  }

  // enabled: transform
  if (scn->enabletransform) {
    // convert translate, rotate to mjtNum
    mju_f2n(translate, scn->translate, 3);
    mju_f2n(rotate, scn->rotate, 4);

    // invert model pose (without scale)
    mju_negPose(invpos, invquat, translate, rotate);

    // map from room to model space
    mju_mulPose(modelpos, modelquat, invpos, invquat, roompos, roomquat);

    // divide position by scale
    mju_scl3(modelpos, modelpos, 1.0/scn->scale);
  }

  // disabled: copy
  else {
    mju_copy3(modelpos, roompos);
    mju_copy4(modelquat, roomquat);
  }
}


// transform pose from model to room space
void mjv_model2room(mjtNum* roompos, mjtNum* roomquat, const mjtNum* modelpos,
                    const mjtNum* modelquat, const mjvScene* scn) {
  mjtNum translate[3], rotate[4];

  // check scale
  if (scn->scale < mjMINVAL) {
    mjERROR("mjvScene scale too small");
  }

  // enabled: transform
  if (scn->enabletransform) {
    // convert translate, rotate to mjtNum
    mju_f2n(translate, scn->translate, 3);
    mju_f2n(rotate, scn->rotate, 4);

    // scale position and map from model to room space
    mjtNum scaledpos[3];
    mju_scl3(scaledpos, modelpos, scn->scale);
    mju_mulPose(roompos, roomquat, translate, rotate, scaledpos, modelquat);
  }

  // disabled: copy
  else {
    mju_copy3(roompos, modelpos);
    mju_copy4(roomquat, modelquat);
  }
}


// get camera info in model space: average left and right OpenGL cameras
void mjv_cameraInModel(mjtNum* headpos, mjtNum* forward, mjtNum* up, const mjvScene* scn) {
  mjtNum pos[3], fwd[3], u[3], quat[4];
  mjtNum modelpos[3], modelquat[4], modelmat[9];

  // check znear
  if (scn->camera[0].frustum_near < mjMINVAL || scn->camera[1].frustum_near < mjMINVAL) {
    mjERROR("mjvScene frustum_near too small");
  }

  // clear results
  if (headpos) {
    mju_zero3(headpos);
  }
  if (forward) {
    mju_zero3(forward);
  }
  if (up) {
    mju_zero3(up);
  }

  // average over cameras
  for (int n=0; n < 2; n++) {
    // convert pos, fwd, u
    mju_f2n(pos, scn->camera[n].pos, 3);
    mju_f2n(fwd, scn->camera[n].forward, 3);
    mju_f2n(u, scn->camera[n].up, 3);

    // normalize just in case
    mju_normalize3(fwd);
    mju_normalize3(u);

    // make orientation matrix: x = left, y = up, z = forward
    mjtNum left[3];
    mju_cross(left, u, fwd);
    mju_normalize3(left);
    mjtNum mat[9] = {
      left[0], u[0], fwd[0],
      left[1], u[1], fwd[1],
      left[2], u[2], fwd[2]
    };
    mju_mat2Quat(quat, mat);

    // convert to model space, make orientation matrix
    mjv_room2model(modelpos, modelquat, pos, quat, scn);
    mju_quat2Mat(modelmat, modelquat);

    // finalize results
    if (headpos) {
      mju_addToScl3(headpos, modelpos, 0.5);
    }
    if (forward) {
      forward[0] += 0.5*modelmat[2];
      forward[1] += 0.5*modelmat[5];
      forward[2] += 0.5*modelmat[8];
    }
    if (up) {
      up[0] += 0.5*modelmat[1];
      up[1] += 0.5*modelmat[4];
      up[2] += 0.5*modelmat[7];
    }
  }

  // normalize forward and up
  if (forward) {
    mju_normalize3(forward);
  }
  if (up) {
    mju_normalize3(up);
  }
}


// get camera info in room space: average left and right OpenGL cameras
void mjv_cameraInRoom(mjtNum* headpos, mjtNum* forward, mjtNum* up, const mjvScene* scn) {
  mjtNum pos[3], fwd[3], u[3];

  // check znear
  if (scn->camera[0].frustum_near < mjMINVAL || scn->camera[1].frustum_near < mjMINVAL) {
    mjERROR("mjvScene frustum_near too small");
  }

  // clear results
  if (headpos) {
    mju_zero3(headpos);
  }
  if (forward) {
    mju_zero3(forward);
  }
  if (up) {
    mju_zero3(up);
  }

  // average over cameras
  for (int n=0; n < 2; n++) {
    // convert pos, fwd, u
    mju_f2n(pos, scn->camera[n].pos, 3);
    mju_f2n(fwd, scn->camera[n].forward, 3);
    mju_f2n(u, scn->camera[n].up, 3);

    // finalize results
    if (headpos) {
      mju_addToScl3(headpos, pos, 0.5);
    }
    if (forward) {
      mju_addToScl3(forward, fwd, 0.5);
    }
    if (up) {
      mju_addToScl3(up, u, 0.5);
    }
  }

  // normalize
  if (forward) {
    mju_normalize3(forward);
  }
  if (up) {
    mju_normalize3(up);
  }
}


// get frustum height at unit distance from camera; average left and right OpenGL cameras
mjtNum mjv_frustumHeight(const mjvScene* scn) {
  const mjvGLCamera* cam1 = scn->camera;
  const mjvGLCamera* cam2 = scn->camera + 1;

  if (cam1->orthographic != cam2->orthographic) {
    mjERROR("cannot average frustums of perspective and orthographic cameras");
  }

  // get height
  mjtNum height;
  if (!cam1->orthographic) {
    // check znear
    if (cam1->frustum_near < mjMINVAL || cam2->frustum_near < mjMINVAL) {
      mjERROR("mjvScene frustum_near too small");
    }

    // add normalized height for left and right cameras
    height = (cam1->frustum_top - cam1->frustum_bottom) / cam1->frustum_near +
             (cam2->frustum_top - cam2->frustum_bottom) / cam2->frustum_near;
  } else {
    // add height for left and right cameras
    height = (cam1->frustum_top - cam1->frustum_bottom) +
             (cam2->frustum_top - cam2->frustum_bottom);
  }

  // average
  return 0.5*height;
}


static mjtNum cameraFrustumHeight(const mjModel* m, const mjvCamera* cam) {
  float zclip[2] = {0, 0}, zver[2] = {0, 0};
  mjv_cameraFrustum(zver, NULL, zclip, m, cam);
  if (cam->orthographic) {
    return (zver[1] + zver[0]);
  } else {
    if (zclip[0] < mjMINVAL) {
      mjERROR("mjvScene frustum_near too small");
    }
    return (zver[1] + zver[0]) / zclip[0];
  }
}


// rotate 3D vec in horizontal plane by angle between (0,1) and (forward_x,forward_y)
void mjv_alignToCamera(mjtNum* res, const mjtNum* vec, const mjtNum* forward) {
  mjtNum xaxis[2], yaxis[2];

  // forward-aligned y-axis
  mju_copy(yaxis, forward, 2);
  mju_normalize(yaxis, 2);

  // corresponding x-axis
  xaxis[0] = yaxis[1];
  xaxis[1] = -yaxis[0];

  // apply horizontal rotation
  res[0] = vec[0]*xaxis[0] + vec[1]*yaxis[0];
  res[1] = vec[0]*xaxis[1] + vec[1]*yaxis[1];
  res[2] = vec[2];
}


// convert 2D mouse motion to z-aligned 3D world coordinates
static void convert2D(mjtNum* res, int action, mjtNum dx, mjtNum dy, const mjtNum* forward) {
  mjtNum vec[3];

  // construct 3D vector
  switch ((mjtMouse) action) {
  case mjMOUSE_ROTATE_V:
    vec[0] = dy;
    vec[1] = 0;
    vec[2] = dx;
    break;

  case mjMOUSE_ROTATE_H:
    vec[0] = dy;
    vec[1] = dx;
    vec[2] = 0;
    break;

  case mjMOUSE_MOVE_V:
  case mjMOUSE_MOVE_V_REL:
    vec[0] = dx;
    vec[1] = 0;
    vec[2] = -dy;
    break;

  case mjMOUSE_MOVE_H:
  case mjMOUSE_MOVE_H_REL:
    vec[0] = dx;
    vec[1] = -dy;
    vec[2] = 0;
    break;

  case mjMOUSE_ZOOM:
    break;

  default:
    mjERROR("unexpected mouse action %d in convert2D", action);
  }

  // call 3D converter
  mjv_alignToCamera(res, vec, forward);
}


// move camera with mouse; action is mjtMouse
void mjv_moveCamera(const mjModel* m, int action, mjtNum reldx, mjtNum reldy, mjvCamera* cam) {
  mjtNum headpos[3], forward[3], up[3], right[3];
  mjtNum vec[3], dif[3], scl;

  // fixed camera: nothing to do
  // note: mjv_cameraFrame, which we use below, requires mjData for non-fixed camera
  if (cam->type == mjCAMERA_FIXED) {
    return;
  }

  // process action
  switch ((mjtMouse) action) {
  case mjMOUSE_ROTATE_V:
  case mjMOUSE_ROTATE_H:
    cam->azimuth -= reldx * 180.0;
    cam->elevation -= reldy * 180.0;
    break;

  case mjMOUSE_MOVE_V:
  case mjMOUSE_MOVE_H:
    // do not move lookat point of tracking camera
    if (cam->type == mjCAMERA_TRACKING) {
      return;
    }

    // get camera info and align
    mjv_cameraFrame(headpos, forward, NULL, NULL, NULL, cam);
    convert2D(vec, action, reldx, reldy, forward);

    // compute scaling: rendered lookat displacement = mouse displacement
    mju_sub3(dif, cam->lookat, headpos);
    scl = cameraFrustumHeight(m, cam) * mju_dot3(dif, forward);

    // multiply by mystery coefficient TODO: b/346130949
    if (cam->orthographic) scl *= 0.15;

    // move lookat point in opposite direction
    mju_addToScl3(cam->lookat, vec, -scl);
    break;

  case mjMOUSE_TURN_V:
  case mjMOUSE_TURN_H:
    if (cam->type == mjCAMERA_TRACKING) {
      return;
    }

    mjv_cameraFrame(headpos, forward, NULL, NULL, NULL, cam);
    if (action == mjMOUSE_TURN_H) {
      cam->azimuth -= reldx * 180.0;
    }
    if (action == mjMOUSE_TURN_V) {
      cam->elevation = mju_clip(cam->elevation - reldy * 180.0, -89, 89);
    }
    mjv_cameraFrame(NULL, forward, NULL, NULL, NULL, cam);

    // move lookat point so that camera faces new direction
    mju_addScl3(cam->lookat, headpos, forward, cam->distance);
    break;

  case mjMOUSE_ZOOM:
    cam->distance -= mju_log(1 + cam->distance/m->stat.extent/3) * reldy * 9 * m->stat.extent;
    break;

  case mjMOUSE_MOVE_V_REL:
  case mjMOUSE_MOVE_H_REL:
    // do not move lookat point of tracking camera
    if (cam->type == mjCAMERA_TRACKING) {
      return;
    }

    mjv_cameraFrame(headpos, forward, up, NULL, NULL, cam);
    mju_cross(right, forward, up);

    // y-axis movement moves forward/backward (ie. camera dolly) on horizontal plane or up/down
    // (ie. camera pedestal) on vertical plane
    mju_addToScl3(cam->lookat, (action == mjMOUSE_MOVE_V_REL) ? up : forward, reldy);

    // x-axis movement strafes left/right (ie. camera truck)
    mju_addToScl3(cam->lookat, right, reldx);

    break;

  default:
    mjERROR("unexpected action %d", action);
  }

  // clamp camera parameters
  if (cam->azimuth > 180) {
    cam->azimuth -= 360;
  }
  if (cam->azimuth < -180) {
    cam->azimuth += 360;
  }
  if (cam->elevation > 89) {
    cam->elevation = 89;
  }
  if (cam->elevation < -89) {
    cam->elevation = -89;
  }
  if (cam->distance < 0.01*m->stat.extent) {
    cam->distance = 0.01*m->stat.extent;
  }
  if (cam->distance > 100*m->stat.extent) {
    cam->distance = 100*m->stat.extent;
  }
}


// move perturb object with mouse; action is mjtMouse
void mjv_movePerturb(const mjModel* m, const mjData* d, int action, mjtNum reldx,
                     mjtNum reldy, const mjvScene* scn, mjvPerturb* pert) {
  const mjtNum xaxis[3] = {1, 0, 0};
  const mjtNum yaxis[3] = {0, 1, 0};
  const mjtNum zaxis[3] = {0, 0, 1};

  int sel = pert->select;
  if (sel <= 0 || sel >= m->nbody) {
    return;
  }
  const mjtNum* xmat = d->xmat+9*sel;
  mjtNum forward[3], vec[3], scl, q1[4], xiquat[4];

  // get camera info and align
  mjv_cameraInModel(NULL, forward, NULL, scn);
  convert2D(vec, action, reldx, reldy, forward);

  // process action
  switch ((mjtMouse) action) {
  case mjMOUSE_MOVE_V:
  case mjMOUSE_MOVE_H:
    // move along world-space horizontal/vertical planes relative to camera
    mju_addToScl3(pert->refpos, vec, pert->scale);
    mju_addToScl3(pert->refselpos, vec, pert->scale);
    break;

  case mjMOUSE_MOVE_V_REL:
  case mjMOUSE_MOVE_H_REL:
    // move along object's local coordinate frame
    if (action == mjMOUSE_MOVE_H_REL) {
      mju_mulMatVec3(vec, xmat, xaxis);
      mju_addToScl3(pert->refpos, vec, pert->scale * reldy);
      mju_addToScl3(pert->refselpos, vec, pert->scale * reldy);
    } else {
      mju_mulMatVec3(vec, xmat, zaxis);
      mju_addToScl3(pert->refpos, vec, pert->scale * reldy);
      mju_addToScl3(pert->refselpos, vec, pert->scale * reldy);
    }

    mju_mulMatVec3(vec, xmat, yaxis);
    mju_addToScl3(pert->refpos, vec, pert->scale * reldx);
    mju_addToScl3(pert->refselpos, vec, pert->scale * reldx);
    break;

  case mjMOUSE_ROTATE_V:
  case mjMOUSE_ROTATE_H:
    // normalize vector, get length
    scl = mju_normalize3(vec);

    // make quaternion and apply
    mju_axisAngle2Quat(q1, vec, scl*mjPI*2);
    mju_mulQuat(pert->refquat, q1, pert->refquat);
    mju_normalize4(pert->refquat);

    // compute xiquat
    mju_mulQuat(xiquat, d->xquat+4*sel, m->body_iquat+4*sel);

    // limit rotation relative to selected body
    if (sel > 0 && sel < m->nbody) {
      // q2 = neg(selbody) * refquat
      mjtNum q2[4];
      mju_negQuat(q1, xiquat);
      mju_mulQuat(q2, q1, pert->refquat);

      // convert q2 to axis-angle
      mjtNum dif[3];
      mju_quat2Vel(dif, q2, 1);
      scl = mju_normalize3(dif);

      // check limit: +/- 90 deg allowed
      if (scl < -mjPI*0.5 || scl > mjPI*0.5) {
        // clamp angle
        scl = mju_max(-mjPI*0.5, mju_min(mjPI*0.5, scl));

        // reconstruct q2
        mju_axisAngle2Quat(q2, dif, scl);

        // set refquat = selbody * q2_new
        mju_mulQuat(pert->refquat, xiquat, q2);
      }
    }
    break;

  case mjMOUSE_ZOOM:
    break;

  default:
    mjERROR("unexpected mouse action %d", action);
  }
}


// move model with mouse; action is mjtMouse
void mjv_moveModel(const mjModel* m, int action, mjtNum reldx, mjtNum reldy,
                   const mjtNum roomup[3], mjvScene* scn) {
  mjtNum roomforward[3], roomright[3], camforward[3];
  mjtNum vec[3], scl, quat[4], rotate[4], result[4];

  // transformation disabled: nothing to do
  if (!scn->enabletransform) {
    return;
  }

  // get camera forward in room space
  mjv_cameraInRoom(NULL, camforward, NULL, scn);

  // make orthogonal to roomright
  mju_addScl3(roomforward, camforward, roomup, -mju_dot3(camforward, roomup));
  mju_normalize3(roomforward);

  // compute roomright
  mju_cross(roomright, roomforward, roomup);
  mju_normalize3(roomright);

  // process action
  switch ((mjtMouse) action) {
  case mjMOUSE_ROTATE_V:
  case mjMOUSE_ROTATE_H:
    // construct rotation vector
    for (int i=0; i < 3; i++) {
      if (action == mjMOUSE_ROTATE_V) {
        vec[i] = roomup[i]*reldx + roomright[i]*reldy;
      } else {
        vec[i] = roomforward[i]*reldx + roomright[i]*reldy;
      }
    }

    // make quaternion from angle-axis
    scl = mju_normalize3(vec);
    mju_axisAngle2Quat(quat, vec, scl*mjPI);

    // get current model rotation
    mju_f2n(rotate, scn->rotate, 4);

    // compose rotation, normalize and set
    mju_mulQuat(result, quat, rotate);
    mju_normalize4(result);
    mju_n2f(scn->rotate, result, 4);
    break;

  case mjMOUSE_MOVE_V:
  case mjMOUSE_MOVE_V_REL:
    for (int i=0; i < 3; i++) {
      scn->translate[i] += (float)(roomright[i]*reldx - roomup[i]*reldy) * m->stat.extent;
    }
    break;

  case mjMOUSE_MOVE_H:
  case mjMOUSE_MOVE_H_REL:
    for (int i=0; i < 3; i++) {
      scn->translate[i] += (float)(roomright[i]*reldx - roomforward[i]*reldy) * m->stat.extent;
    }
    break;

  case mjMOUSE_ZOOM:
    scn->scale += (float)(mju_log(1 + scn->scale/3) * reldy * 3);
    if (scn->scale < 0.01f) {
      scn->scale = 0.01f;
    } else if (scn->scale > 100.0f) {
      scn->scale = 100.0f;
    }
    break;

  default:
    mjERROR("unexpected action %d", action);
  }
}


// copy perturb pos,quat from selected body; set scale for perturbation
void mjv_initPerturb(const mjModel* m, mjData* d, const mjvScene* scn, mjvPerturb* pert) {
  mj_markStack(d);

  int nv  = m->nv;
  int sel = pert->select;
  mjtNum headpos[3], forward[3], dif[3];

  mjtNum* jac = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* jacM2 = mjSTACKALLOC(d, 3*nv, mjtNum);
  mjtNum* sqrtInvD = mjSTACKALLOC(d, nv, mjtNum);

  // invalid selected body: return
  if (sel <= 0 || sel >= m->nbody) {
    mj_freeStack(d);
    return;
  }

  // compute selection point in world coordinates
  mjtNum selpos[3];
  mju_mulMatVec3(selpos, d->xmat+9*sel, pert->localpos);
  mju_addTo3(selpos, d->xpos+3*sel);

  // compute average spatial inertia at selection point
  for (int i=0; i < nv; i++) {
    sqrtInvD[i] = mju_sqrt(d->qLDiagInv[i]);
  }
  mj_jac(m, d, jac, NULL, selpos, sel);
  mj_solveM2(m, d, jacM2, jac, sqrtInvD, 3);
  mjtNum invmass = mju_dot(jacM2+0*nv, jacM2+0*nv, nv) +
                   mju_dot(jacM2+1*nv, jacM2+1*nv, nv) +
                   mju_dot(jacM2+2*nv, jacM2+2*nv, nv);
  pert->localmass = (invmass == 0) ? 1 : 3 / mju_max(invmass, mjMINVAL);

  // scale localmass with flex average number of edges per vertex
  if (pert->flexselect >= 0 && !m->flex_rigid[pert->flexselect]) {
    pert->localmass *= (2.0*m->flex_edgenum[pert->flexselect]) /
                       (mjtNum)m->flex_vertnum[pert->flexselect];
  }

  // copy
  mju_copy3(pert->refpos, d->xipos+3*sel);
  mju_mulQuat(pert->refquat, d->xquat+4*sel, m->body_iquat+4*sel);
  mju_copy3(pert->refselpos, selpos);

  // get camera info
  mjv_cameraInModel(headpos, forward, NULL, scn);

  // compute scaling: rendered pert->refselpos displacement = mouse displacement
  mju_sub3(dif, pert->refselpos, headpos);
  pert->scale = mjv_frustumHeight(scn) * mju_dot3(dif, forward);

  // multiply by mystery coefficient TODO: b/346130949
  if (scn->camera[0].orthographic) pert->scale *= 0.15;

  mj_freeStack(d);
}


// set perturb pos,quat in d->mocap when selected body is mocap, and in d->qpos otherwise
//  d->qpos written only if flg_paused and subtree root for selected body has free joint
void mjv_applyPerturbPose(const mjModel* m, mjData* d, const mjvPerturb* pert, int flg_paused) {
  int rootid = 0, sel = pert->select;
  mjtNum pos1[3], quat1[4], pos2[3], quat2[4], refpos[3], refquat[4];
  mjtNum *Rpos, *Rquat, *Cpos, *Cquat;

  // exit if nothing to do
  if (sel <= 0 || sel >= m->nbody || !(pert->active | pert->active2)) {
    return;
  }

  // get rootid above selected body
  rootid = m->body_rootid[sel];

  // transform refpos,refquat from I-frame to X-frame of body[sel]
  mju_negPose(pos1, quat1, m->body_ipos+3*sel, m->body_iquat+4*sel);
  mju_mulPose(refpos, refquat, pert->refpos, pert->refquat, pos1, quat1);

  // mocap body
  if (m->body_mocapid[sel] >= 0) {
    // copy ref pose into mocap pose
    mju_copy3(d->mocap_pos + 3*m->body_mocapid[sel], refpos);
    mju_copy4(d->mocap_quat + 4*m->body_mocapid[sel], refquat);
  }

  // floating body, paused
  else if (flg_paused && m->body_jntnum[sel] == 1 &&
           m->jnt_type[m->body_jntadr[sel]] == mjJNT_FREE) {
    // copy ref pose into qpos
    mju_copy3(d->qpos + m->jnt_qposadr[m->body_jntadr[sel]], refpos);
    mju_copy4(d->qpos + m->jnt_qposadr[m->body_jntadr[sel]] + 3, refquat);
  }

  // child of floating body, paused
  else if (flg_paused && m->body_jntnum[rootid] == 1 &&
           m->jnt_type[m->body_jntadr[rootid]] == mjJNT_FREE) {
    // get pointers to root
    Rpos = d->qpos + m->jnt_qposadr[m->body_jntadr[rootid]];
    Rquat = Rpos + 3;

    // get pointers to child
    Cpos = d->xpos + 3*sel;
    Cquat = d->xquat + 4*sel;

    // set root <- ref*neg(child)*root
    mju_negPose(pos1, quat1, Cpos, Cquat);                      // neg(child)
    mju_mulPose(pos2, quat2, pos1, quat1, Rpos, Rquat);         // neg(child)*root
    mju_mulPose(Rpos, Rquat, refpos, refquat, pos2, quat2);     // ref*neg(child)*root
  }
}


// set perturb force,torque in d->xfrc_applied, if selected body is dynamic
void mjv_applyPerturbForce(const mjModel* m, mjData* d, const mjvPerturb* pert) {
  int sel = pert->select;

  // exit if nothing to do
  if (sel <= 0 || sel >= m->nbody || !(pert->active | pert->active2)) {
    return;
  }

  // pointers to body xfrc_applied, force and torque
  mjtNum *force = d->xfrc_applied + 6*sel;
  mjtNum *torque = d->xfrc_applied + 6*sel + 3;

  // pointers to global selbody velocity, linear and rotational
  mjtNum bvel[6];
  mj_objectVelocity(m, d, mjOBJ_BODY, sel, bvel, 0);
  mjtNum *body_linvel = bvel + 3;
  mjtNum *body_rotvel = bvel;

  // body rotational inertia
  mjtNum invweight = m->body_invweight0[2*sel+1];
  mjtNum inertia = invweight ? 1.0/mju_max(mjMINVAL, invweight) : 1;

  if (((pert->active | pert->active2) & mjPERT_TRANSLATE)) {
    // compute selection point in world coordinates
    mjtNum selpos[3];
    mju_mulMatVec3(selpos, d->xmat+9*sel, pert->localpos);
    mju_addTo3(selpos, d->xpos+3*sel);

    // displacement of selection point from reference point
    mjtNum diff[3];
    mju_sub3(diff, selpos, pert->refselpos);

    // spring perturbation force
    mjtNum stiffness = m->vis.map.stiffness;
    mju_copy3(force, diff);
    mju_scl3(force, force, -stiffness*pert->localmass);

    // moment arm w.r.t body com
    mjtNum moment_arm[3];
    mju_sub3(moment_arm, selpos, d->xipos+3*sel);

    // translational velocity of selection point
    mjtNum svel[3];
    mju_cross(svel, body_rotvel, moment_arm);
    mju_addTo3(svel, body_linvel);

    // add critical damping force of selection point
    mju_addToScl3(force, svel, -mju_sqrt(stiffness)*pert->localmass);

    // torque on body com due to force
    mju_cross(torque, moment_arm, force);

    // add critically damped torsional torque along displacement axis
    stiffness = m->vis.map.stiffnessrot;
    mju_normalize3(diff);
    mju_addToScl3(torque, diff, -mju_sqrt(stiffness)*inertia*mju_dot3(diff, body_rotvel));
  }

  if (((pert->active | pert->active2) & mjPERT_ROTATE)) {
    // spring perturbation torque, with critical damping
    mjtNum stiffness = m->vis.map.stiffnessrot;
    mjtNum xiquat[4], difquat[4], rottorque[3];
    mju_mulQuat(xiquat, d->xquat+4*sel, m->body_iquat+4*sel);
    mju_negQuat(xiquat, xiquat);
    mju_mulQuat(difquat, pert->refquat, xiquat);
    mju_quat2Vel(rottorque, difquat, 1.0/(stiffness*inertia));
    mju_addToScl3(rottorque, body_rotvel, -mju_sqrt(stiffness)*inertia);
    if ((pert->active | pert->active2) & mjPERT_TRANSLATE) {
      mju_addTo3(torque, rottorque);
    } else {
      mju_copy3(torque, rottorque);
    }
  }
}


// return the average of two OpenGL cameras
mjvGLCamera mjv_averageCamera(const mjvGLCamera* cam1, const mjvGLCamera* cam2) {
  mjtNum pos[3], forward[3], up[3], projection, tmp1[3], tmp2[3];
  mjvGLCamera cam;

  // compute pos
  mju_f2n(tmp1, cam1->pos, 3);
  mju_f2n(tmp2, cam2->pos, 3);
  mju_add3(pos, tmp1, tmp2);
  mju_scl3(pos, pos, 0.5);

  // compute forward
  mju_f2n(tmp1, cam1->forward, 3);
  mju_f2n(tmp2, cam2->forward, 3);
  mju_add3(forward, tmp1, tmp2);
  mju_normalize3(forward);

  // compute up, make it orthogonal to forward
  mju_f2n(tmp1, cam1->up, 3);
  mju_f2n(tmp2, cam2->up, 3);
  mju_add3(up, tmp1, tmp2);
  projection = mju_dot3(up, forward);
  mju_addToScl3(up, forward, -projection);
  mju_normalize3(up);

  // assign 3d quantities
  mju_n2f(cam.pos, pos, 3);
  mju_n2f(cam.forward, forward, 3);
  mju_n2f(cam.up, up, 3);

  // average frustum
  cam.frustum_bottom = 0.5f * (cam1->frustum_bottom + cam2->frustum_bottom);
  cam.frustum_top    = 0.5f * (cam1->frustum_top + cam2->frustum_top);
  cam.frustum_center = 0.5f * (cam1->frustum_center + cam2->frustum_center);
  cam.frustum_width  = 0.5f * (cam1->frustum_width + cam2->frustum_width);
  cam.frustum_near   = 0.5f * (cam1->frustum_near + cam2->frustum_near);
  cam.frustum_far    = 0.5f * (cam1->frustum_far + cam2->frustum_far);

  if (cam1->orthographic != cam2->orthographic) {
    mjERROR("cannot average perspective and orthographic cameras");
  } else {
    cam.orthographic = cam1->orthographic;
  }

  return cam;
}


// converts a mjvCamera to a mjvGLCamera
mjvGLCamera mjv_camera2GLCamera(const mjModel* model, const mjData* data,
                                const mjvCamera* mjv_camera) {
  mjvGLCamera gl_camera;
  gl_camera.orthographic = mjv_camera->orthographic;
  if (mjv_camera->type == mjCAMERA_FIXED) {
    const int cid = mjv_camera->fixedcamid;
    if (cid < 0 || cid >= model->ncam) {
      mjERROR("fixed camera id is outside valid range");
    }
    gl_camera.orthographic = model->cam_projection[cid] == mjPROJ_ORTHOGRAPHIC;
  }

  mjtNum headpos[3], forward[3], up[3];
  mjv_cameraFrame(headpos, forward, up, NULL, data, mjv_camera);
  for (int i = 0; i < 3; i++) {
    gl_camera.pos[i] = (float)(headpos[i]);
    gl_camera.forward[i] = (float)forward[i];
    gl_camera.up[i] = (float)up[i];
  }

  float zver[2], zhor[2], zclip[2];
  mjv_cameraFrustum(zver, zhor, zclip, model, mjv_camera);
  gl_camera.frustum_top = zver[0];
  gl_camera.frustum_bottom = -zver[1];
  gl_camera.frustum_center = (zhor[1] - zhor[0]) / 2;
  gl_camera.frustum_width = (zhor[1] + zhor[0]) / 2;
  gl_camera.frustum_near = zclip[0];
  gl_camera.frustum_far = zclip[1];
  return gl_camera;
}


// return body id, compute position of a vertex in a flex
int mjv_flexBodyId(const mjModel* m, const mjData* d, int flexid, int vertid, mjtNum flexpnt[3]) {
  int flexbodyid = -1;
  if (m->flex_interp[flexid]) {
    mjtNum* coord = m->flex_vert0 + 3*(m->flex_vertadr[flexid] + vertid);
    int order = m->flex_interp[flexid];
    order = order < 0 ? -order : order;
    int npc = (order+1)*(order+1)*(order+1);

    // cell lookup: get local coords and node indices
    int gv = m->flex_vertadr[flexid] + vertid;
    int cell = m->flex_vertcell ? m->flex_vertcell[gv] : -1;
    mjtNum loc[3];
    int nodeindices[27];  // max npc for quadratic: 3^3 = 27
    mju_cellLookup(coord, m->flex_cellnum+3*flexid, order, cell, loc, nodeindices);

    // find node with largest weight in this cell
    // in shell mode, skip interior nodes (pinned to worldbody)
    int nodeid = -1;
    int nstart = m->flex_nodeadr[flexid];
    mjtNum w = 0;
    int shell_mode = m->flex_interp[flexid] < 0;
    for (int j = 0; j < npc; j++) {
      mjtNum ww = mju_evalBasis(loc, j, order);
      int nid = nodeindices[j];
      // skip interior nodes in shell mode (they map to worldbody)
      if (shell_mode && m->body_dofnum[m->flex_nodebodyid[nstart + nid]] == 0) {
        continue;
      }
      if (ww > w) {
        w = ww;
        nodeid = nid;
      }
    }
    flexbodyid = m->flex_nodebodyid[nstart + nodeid];
    if (m->flex_centered[flexid]) {
      mju_copy3(flexpnt, d->xpos + 3*flexbodyid);
    } else {
      mju_mulMatVec3(flexpnt, d->xmat + 9*flexbodyid, m->flex_node + 3*(nstart + nodeid));
      mju_addTo3(flexpnt, d->xpos + 3*flexbodyid);
    }
  } else {
    flexbodyid = m->flex_vertbodyid[m->flex_vertadr[flexid] + vertid];
    mju_copy3(flexpnt, d->flexvert_xpos + 3*(m->flex_vertadr[flexid] + vertid));
  }
  return flexbodyid;
}

// Select geom, flex or skin with mouse, return bodyid; -1: none selected.
int mjv_select(const mjModel* m, const mjData* d, const mjvOption* vopt,
               mjtNum aspectratio, mjtNum relx, mjtNum rely,
               const mjvScene* scn, mjtNum selpnt[3],
               int geomid[1], int flexid[1], int skinid[1]) {
  // get average camera
  mjvGLCamera cam = mjv_averageCamera(scn->camera, scn->camera+1);

  // get camera pose in model space
  mjtNum pos[3], forward[3], up[3], left[3];
  mjv_cameraInModel(pos, forward, up, scn);
  mju_cross(left, up, forward);
  mju_normalize3(left);

  // compute frustum halfwidth so as to match viewport aspect ratio
  mjtNum halfwidth = cam.frustum_width ? cam.frustum_width
                     : 0.5*aspectratio*(cam.frustum_top - cam.frustum_bottom);

  // compute up and left offsets from normalized cursor
  mjtNum d_up = cam.frustum_bottom + rely*(cam.frustum_top-cam.frustum_bottom);
  mjtNum d_left = -(cam.frustum_center + (2*relx-1)*halfwidth);

  // define ray
  mjtNum ray[3];

  // construct ray for orthographic camera: fixed direction, modify pos
  if (cam.orthographic) {
    mju_copy3(ray, forward);
    mju_addToScl3(pos, up, d_up);
    mju_addToScl3(pos, left, d_left);
  }

  // construct ray for perspective camera: fixed pos, modify direction
  else {
    mju_scl3(ray, forward, cam.frustum_near);
    mju_addToScl3(ray, up, d_up);
    mju_addToScl3(ray, left, d_left);
    mju_normalize3(ray);
  }

  // find intersection with geoms
  *geomid = -1;
  mjtNum geomdist = mj_ray(m, d, pos, ray,
                           vopt->geomgroup, vopt->flags[mjVIS_STATIC], -1, geomid, NULL);

  // find intersection with flexes
  int flexbodyid = -1;
  mjtNum flexdist = -1;
  mjtNum flexpnt[3] = {0, 0, 0};
  *flexid = -1;
  if (vopt->flags[mjVIS_FLEXVERT] || vopt->flags[mjVIS_FLEXEDGE] ||
      vopt->flags[mjVIS_FLEXFACE] || vopt->flags[mjVIS_FLEXSKIN]) {
    for (int i=0; i < m->nflex; i++) {
      if (!vopt->flexgroup[mjMAX(0, mjMIN(mjNGROUP-1, m->flex_group[i]))]) {
        continue;
      }

      // process one flex
      int vertid;
      mjtNum newdist =
          mj_rayFlex(m, d, vopt->flex_layer, vopt->flags[mjVIS_FLEXVERT],
                     vopt->flags[mjVIS_FLEXEDGE], vopt->flags[mjVIS_FLEXFACE],
                     vopt->flags[mjVIS_FLEXSKIN], i, pos, ray, &vertid, NULL);

      // update if closer intersection found
      if (newdist >= 0 && (newdist < flexdist || flexdist < 0)) {
        flexdist = newdist;
        *flexid = i;
        flexbodyid = mjv_flexBodyId(m, d, *flexid, vertid, flexpnt);
      }
    }
  }

  // find intersection with skins
  int skinbodyid = -1;
  mjtNum skindist = -1;
  mjtNum skinpnt[3] = {0, 0, 0};
  *skinid = -1;
  if (vopt->flags[mjVIS_SKIN]) {
    for (int i=0; i < m->nskin; i++) {
      if (!vopt->skingroup[mjMAX(0, mjMIN(mjNGROUP-1, m->skin_group[i]))]) {
        continue;
      }

      // process one skin
      int vertid;
      mjtNum newdist = mju_raySkin(m->skin_facenum[i], m->skin_vertnum[i],
                                   m->skin_face + 3*m->skin_faceadr[i],
                                   scn->skinvert + 3*m->skin_vertadr[i],
                                   pos, ray, &vertid);

      // update if closer intersection found
      if (newdist >= 0 && (newdist < skindist || skindist < 0)) {
        skindist = newdist;

        // find body with largest weight for this vertex
        float bestweight = -1;
        for (int j=m->skin_boneadr[i];
             j < m->skin_boneadr[i]+m->skin_bonenum[i];
             j++) {
          for (int k=m->skin_bonevertadr[j];
               k < m->skin_bonevertadr[j]+m->skin_bonevertnum[j];
               k++) {
            // get vertex id and weight
            int vid = m->skin_bonevertid[k];
            float vweight = m->skin_bonevertweight[k];

            // update if matching id and bigger weight
            if (vid == vertid && vweight > bestweight) {
              bestweight = vweight;
              skinbodyid = m->skin_bonebodyid[j];
              *skinid = i;
              mju_f2n(skinpnt, scn->skinvert + 3*(m->skin_vertadr[i] + vertid), 3);
            }
          }
        }
      }
    }
  }

  // no intersection
  if (geomdist < 0 && flexdist < 0 && skindist < 0) {
    return -1;
  }

  // find smallest non-negative distance
  mjtNum raydist[3] = {geomdist, flexdist, skindist};
  int best = -1;
  for (int i=0; i < 3; i++) {
    if (raydist[i] >= 0) {
      if (best < 0 || raydist[best] > raydist[i]) {
        best = i;
      }
    }
  }

  // geom
  if (best == 0) {
    *flexid = -1;
    *skinid = -1;
    mju_addScl3(selpnt, pos, ray, raydist[best]);
    return m->geom_bodyid[*geomid];
  }

  // flex
  else if (best == 1) {
    *geomid = -1;
    *skinid = -1;
    mju_copy3(selpnt, flexpnt);
    return flexbodyid;
  }

  // skin
  else {
    *geomid = -1;
    *flexid = -1;
    mju_copy3(selpnt, skinpnt);
    return skinbodyid;
  }
}
