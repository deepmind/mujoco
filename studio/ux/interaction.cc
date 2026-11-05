// Copyright 2025 DeepMind Technologies Limited
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

#include "studio/ux/interaction.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include <mujoco/mujoco.h>
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_vis_interact.h"
#include "engine/engine_vis_visualize.h"

namespace mujoco::studio {

static mjtNum CalculateMovementScale(const mjModel* m, const mjvCamera* cam) {
  float zclip[2] = {0, 0}, zver[2] = {0, 0};
  mjv_cameraFrustum(zver, nullptr, zclip, m, cam);
  if (cam->orthographic) {
    // TODO(b/346130949): multiply by mystery coefficient
    return (zver[1] + zver[0]) * 0.15;
  } else if (zclip[0] >= mjMINVAL) {
    return (zver[1] + zver[0]) / zclip[0];
  } else {
    mjERROR("mjvScene frustum_near too small");
    return 0;
  }
}

static void AlignToCamera(mjtNum res[3], mjtMouse action, mjtNum dx, mjtNum dy,
                          const mjtNum forward[3]) {
  mjtNum vec[3];
  switch (action) {
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
      mjERROR("unexpected mouse action %d in AlignToCamera", action);
  }

  // call 3D converter
  mjv_alignToCamera(res, vec, forward);
}

void InitPerturb(const mjModel* m, const mjData* d, const mjvCamera* cam, mjvPerturb* pert,
                 mjtPertBit active) {
  // compute selection point in world coordinates
  const int sel = pert->select;
  mjtNum selpos[3];
  mju_mulMatVec3(selpos, d->xmat + 9 * sel, pert->localpos);
  mju_addTo3(selpos, d->xpos + 3 * sel);

  // compute average spatial inertia at selection point
  const int nv = m->nv;
  std::vector<mjtNum> sqrtInvD(nv);
  for (int i = 0; i < nv; i++) {
    sqrtInvD[i] = mju_sqrt(d->qLDiagInv[i]);
  }

  std::vector<mjtNum> jac(3 * nv);
  mj_jac(m, d, jac.data(), nullptr, selpos, sel);

  std::vector<mjtNum> jacM2(3 * nv);
  mj_solveM2(m, const_cast<mjData*>(d), jacM2.data(), jac.data(), sqrtInvD.data(), 3);
  mjtNum invmass = mju_dot(jacM2.data() + 0 * nv, jacM2.data() + 0 * nv, nv) +
                   mju_dot(jacM2.data() + 1 * nv, jacM2.data() + 1 * nv, nv) +
                   mju_dot(jacM2.data() + 2 * nv, jacM2.data() + 2 * nv, nv);
  pert->localmass = (invmass == 0) ? 1 : 3 / mju_max(invmass, mjMINVAL);

  // scale localmass with flex average number of edges per vertex
  if (pert->flexselect >= 0 && !m->flex_rigid[pert->flexselect]) {
    pert->localmass *=
        (2.0 * m->flex_edgenum[pert->flexselect]) / (mjtNum)m->flex_vertnum[pert->flexselect];
  }

  // copy
  mju_copy3(pert->refpos, d->xipos + 3 * sel);
  mju_mulQuat(pert->refquat, d->xquat + 4 * sel, m->body_iquat + 4 * sel);
  mju_copy3(pert->refselpos, selpos);

  // get camera info
  mjtNum headpos[3], forward[3];
  mjv_cameraFrame(headpos, forward, nullptr, nullptr, d, cam);

  // compute scaling: rendered pert->refselpos displacement = mouse displacement
  mjtNum dif[3];
  mju_sub3(dif, pert->refselpos, headpos);
  pert->scale = CalculateMovementScale(m, cam) * mju_dot3(dif, forward);
  pert->active = active;
}

void MovePerturb(const mjModel* m, const mjData* d, const mjvCamera* cam, mjvPerturb* pert,
                 mjtMouse action, mjtNum reldx, mjtNum reldy) {
  const mjtNum xaxis[3] = {1, 0, 0};
  const mjtNum yaxis[3] = {0, 1, 0};
  const mjtNum zaxis[3] = {0, 0, 1};

  int sel = pert->select;
  const mjtNum* xmat = d->xmat + 9 * sel;
  mjtNum forward[3], vec[3], scl, q1[4], xiquat[4];

  // get camera info and align
  mjv_cameraFrame(nullptr, forward, nullptr, nullptr, d, cam);
  AlignToCamera(vec, action, reldx, reldy, forward);

  // process action
  switch ((mjtMouse)action) {
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
      mju_axisAngle2Quat(q1, vec, scl * mjPI * 2);
      mju_mulQuat(pert->refquat, q1, pert->refquat);
      mju_normalize4(pert->refquat);

      // compute xiquat
      mju_mulQuat(xiquat, d->xquat + 4 * sel, m->body_iquat + 4 * sel);

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
        if (scl < -mjPI * 0.5 || scl > mjPI * 0.5) {
          // clamp angle
          scl = mju_max(-mjPI * 0.5, mju_min(mjPI * 0.5, scl));

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

void MoveCamera(const mjModel* m, const mjData* d, mjvCamera* cam, CameraMotion motion, mjtNum dx,
                mjtNum dy) {
  switch (motion) {
    case CameraMotion::ZOOM:
      mjv_moveCamera(m, mjMOUSE_ZOOM, 0, dy, cam);
      break;
    case CameraMotion::ORBIT:
      mjv_moveCamera(m, mjMOUSE_ROTATE_H, dx, 0, cam);
      mjv_moveCamera(m, mjMOUSE_ROTATE_V, 0, dy, cam);
      break;
    case CameraMotion::TRUCK_PEDESTAL:
      mjv_moveCamera(m, mjMOUSE_MOVE_V_REL, dx, dy, cam);
      break;
    case CameraMotion::TRUCK_DOLLY:
      mjv_moveCamera(m, mjMOUSE_MOVE_H_REL, dx, dy, cam);
      break;
    case CameraMotion::PAN_TILT:
      mjv_moveCamera(m, mjMOUSE_TURN_H, dx, 0, cam);
      mjv_moveCamera(m, mjMOUSE_TURN_V, 0, dy, cam);
      break;
    case CameraMotion::PLANAR_MOVE_V:
      mjv_moveCamera(m, mjMOUSE_MOVE_V, dx, dy, cam);
      break;
    case CameraMotion::PLANAR_MOVE_H:
      mjv_moveCamera(m, mjMOUSE_MOVE_H, dx, dy, cam);
      break;
  }
}

static void MakePickRay(mjtNum pos[3], mjtNum ray[3], const mjModel* m, const mjData* d,
                        const mjvCamera* camera, float relx, float rely, float aspect_ratio) {
  mjtNum forward[3], up[3], right[3];
  mjv_cameraFrame(pos, forward, up, right, d, camera);

  float zver[2], zhor[2], zclip[2] = {0, 0};
  mjv_cameraFrustum(zver, zhor, zclip, m, camera);

  // compute frustum halfwidth to match viewport aspect ratio
  mjtNum half_width = 0.5 * aspect_ratio * (zver[0] + zver[1]);
  mjtNum frustum_center = (zhor[1] - zhor[0]) / 2;

  // compute up and right offsets from normalized cursor
  mjtNum d_up = -zver[0] + rely * (zver[0] + zver[1]);
  mjtNum d_right = frustum_center + (2 * relx - 1) * half_width;

  if (camera->orthographic) {
    mju_copy3(ray, forward);
    mju_addToScl3(pos, up, d_up);
    mju_addToScl3(pos, right, d_right);
  } else {
    mju_scl3(ray, forward, zclip[0]);
    mju_addToScl3(ray, up, d_up);
    mju_addToScl3(ray, right, d_right);
    mju_normalize3(ray);
  }
}

static PickResult PickGeom(const mjModel* m, const mjData* d, const mjtNum ray_pos[3],
                           const mjtNum ray_dir[3], const mjvOption* vis_options) {
  PickResult result;
  result.dist = mj_ray(m, d, ray_pos, ray_dir, vis_options->geomgroup,
                       vis_options->flags[mjVIS_STATIC], -1, &result.geom, nullptr);
  if (result.dist >= 0 && result.geom >= 0) {
    mju_addScl3(result.point, ray_pos, ray_dir, result.dist);
    result.body = m->geom_bodyid[result.geom];
  }
  return result;
}

static PickResult PickFlex(const mjModel* m, const mjData* d, const mjtNum ray_pos[3],
                           const mjtNum ray_dir[3], const mjvOption* vis_options) {
  const mjtByte flag_vert = vis_options->flags[mjVIS_FLEXVERT];
  const mjtByte flag_edge = vis_options->flags[mjVIS_FLEXEDGE];
  const mjtByte flag_face = vis_options->flags[mjVIS_FLEXFACE];
  const mjtByte flag_skin = vis_options->flags[mjVIS_FLEXSKIN];

  PickResult result;
  if (!flag_vert && !flag_edge && !flag_face && !flag_skin) {
    return result;
  }

  for (int i = 0; i < m->nflex; i++) {
    int vertid;
    const mjtNum test_dist =
        mj_rayFlex(m, d, vis_options->flex_layer, flag_vert, flag_edge, flag_face, flag_skin, i,
                   ray_pos, ray_dir, &vertid, nullptr);

    if (test_dist < 0) {
      continue;
    } else if (result.dist >= 0 && test_dist >= result.dist) {
      continue;
    }

    result.dist = test_dist;
    result.flex = i;
    result.body = mjv_flexBodyId(m, d, i, vertid, result.point);
  }
  return result;
}

static void MakeSkin(const mjModel* m, const mjData* d, const mjvOption* opt, int i,
                     float* skinnormal, float* skinvert) {
  int vertadr = m->skin_vertadr[i];
  int vertnum = m->skin_vertnum[i];
  int faceadr = m->skin_faceadr[i];
  int facenum = m->skin_facenum[i];

  // accumulate positions from all bones
  for (int j = m->skin_boneadr[i]; j < m->skin_boneadr[i] + m->skin_bonenum[i]; j++) {
    // get bind pose
    mjtNum bindpos[3] = {(mjtNum)m->skin_bonebindpos[3 * j + 0],
                         (mjtNum)m->skin_bonebindpos[3 * j + 1],
                         (mjtNum)m->skin_bonebindpos[3 * j + 2]};
    mjtNum bindquat[4] = {
        (mjtNum)m->skin_bonebindquat[4 * j + 0], (mjtNum)m->skin_bonebindquat[4 * j + 1],
        (mjtNum)m->skin_bonebindquat[4 * j + 2], (mjtNum)m->skin_bonebindquat[4 * j + 3]};

    // compute rotation
    int bodyid = m->skin_bonebodyid[j];
    mjtNum quat[4], quatneg[4], rotate[9];
    mju_negQuat(quatneg, bindquat);
    mju_mulQuat(quat, d->xquat + 4 * bodyid, quatneg);
    mju_quat2Mat(rotate, quat);

    // compute translation
    mjtNum translate[3];
    mju_mulMatVec3(translate, rotate, bindpos);
    mju_sub3(translate, d->xpos + 3 * bodyid, translate);

    // process all bone vertices
    for (int k = m->skin_bonevertadr[j]; k < m->skin_bonevertadr[j] + m->skin_bonevertnum[j]; k++) {
      // vertex id and weight
      int vid = m->skin_bonevertid[k];
      float vweight = m->skin_bonevertweight[k];

      // get original position
      mjtNum pos[3] = {
          (mjtNum)m->skin_vert[3 * (vertadr + vid)],
          (mjtNum)m->skin_vert[3 * (vertadr + vid) + 1],
          (mjtNum)m->skin_vert[3 * (vertadr + vid) + 2],
      };

      // transform
      mjtNum pos1[3];
      mju_mulMatVec3(pos1, rotate, pos);
      mju_addTo3(pos1, translate);

      // accumulate position
      skinvert[(3 * vid)] += vweight * (float)pos1[0];
      skinvert[(3 * vid) + 1] += vweight * (float)pos1[1];
      skinvert[(3 * vid) + 2] += vweight * (float)pos1[2];
    }
  }

  // inflate
  if (m->skin_inflate[i] && skinnormal != nullptr) {
    // compute vertex normals from face normals
    for (int k = faceadr; k < faceadr + facenum; k++) {
      // get face vertex indices
      int vid[3] = {m->skin_face[3 * k], m->skin_face[3 * k + 1], m->skin_face[3 * k + 2]};

      // get triangle edges
      mjtNum vec01[3], vec02[3];
      for (int r = 0; r < 3; r++) {
        vec01[r] = skinvert[3 * (vid[1]) + r] - skinvert[3 * (vid[0]) + r];
        vec02[r] = skinvert[3 * (vid[2]) + r] - skinvert[3 * (vid[0]) + r];
      }

      // compute face normal
      mjtNum nrm[3];
      mju_cross(nrm, vec01, vec02);

      // add normal to each vertex with weight = area
      for (int r = 0; r < 3; r++) {
        for (int t = 0; t < 3; t++) {
          skinnormal[3 * (vid[r]) + t] += nrm[t];
        }
      }
    }

    // normalize normals
    for (int k = 0; k < vertnum; k++) {
      float s = sqrtf(skinnormal[3 * (k) + 0] * skinnormal[3 * k + 0] +
                      skinnormal[3 * (k) + 1] * skinnormal[3 * k + 1] +
                      skinnormal[3 * (k) + 2] * skinnormal[3 * k + 2]);

      float scl = 1 / mjMAX(mjMINVAL, s);
      skinnormal[3 * k] *= scl;
      skinnormal[3 * k + 1] *= scl;
      skinnormal[3 * k + 2] *= scl;
    }

    float inflate = m->skin_inflate[i];
    for (int k = 0; k < vertnum; k++) {
      skinvert[3 * k] += inflate * skinnormal[3 * k];
      skinvert[3 * k + 1] += inflate * skinnormal[3 * k + 1];
      skinvert[3 * k + 2] += inflate * skinnormal[3 * k + 2];
    }
  }
}

static PickResult PickSkin(const mjModel* m, const mjData* d, const mjtNum ray_pos[3],
                           const mjtNum ray_dir[3], const mjvOption* vis_options) {
  PickResult result;
  if (!vis_options->flags[mjVIS_SKIN]) {
    return result;
  }

  std::vector<float> vertex_buffer;
  std::vector<float> normal_buffer;

  for (int i = 0; i < m->nskin; i++) {
    const int skin_group = mjMAX(0, mjMIN(mjNGROUP - 1, m->skin_group[i]));
    if (!vis_options->skingroup[skin_group]) {
      continue;
    }

    vertex_buffer.resize(3 * m->skin_vertnum[i]);
    if (m->skin_inflate[i]) {
      normal_buffer.resize(3 * m->skin_vertnum[i]);
    }

    float* skinvert = vertex_buffer.data();
    float* skinnormal = m->skin_inflate[i] ? normal_buffer.data() : nullptr;
    MakeSkin(m, d, vis_options, i, skinnormal, skinvert);

    int vertid;
    mjtNum test_dist =
        mju_raySkin(m->skin_facenum[i], m->skin_vertnum[i], m->skin_face + 3 * m->skin_faceadr[i],
                    skinvert, ray_pos, ray_dir, &vertid);
    if (test_dist < 0) {
      continue;
    } else if (result.dist >= 0 && test_dist >= result.dist) {
      continue;
    }

    result.dist = test_dist;

    // find body with largest weight for this vertex
    float best_weight = -1;
    for (int j = m->skin_boneadr[i]; j < m->skin_boneadr[i] + m->skin_bonenum[i]; j++) {
      for (int k = m->skin_bonevertadr[j]; k < m->skin_bonevertadr[j] + m->skin_bonevertnum[j];
           k++) {
        // get vertex id and weight
        const int vertex_id = m->skin_bonevertid[k];
        const float vertex_weight = m->skin_bonevertweight[k];

        // update if matching id and bigger weight
        if (vertex_id == vertid && vertex_weight > best_weight) {
          best_weight = vertex_weight;
          result.body = m->skin_bonebodyid[j];
          result.skin = i;
          mju_f2n(result.point, skinvert + 3 * vertid, 3);
        }
      }
    }
  }

  return result;
}

PickResult Pick(const mjModel* m, const mjData* d, const mjvCamera* camera, float x, float y,
                float aspect_ratio, const mjvOption* vis_options) {
  mjtNum ray_pos[3];
  mjtNum ray_dir[3];
  MakePickRay(ray_pos, ray_dir, m, d, camera, x, 1.0 - y, aspect_ratio);

  PickResult results[3];
  results[0] = PickGeom(m, d, ray_pos, ray_dir, vis_options);
  results[1] = PickFlex(m, d, ray_pos, ray_dir, vis_options);
  results[2] = PickSkin(m, d, ray_pos, ray_dir, vis_options);

  PickResult best_result;
  for (int i = 0; i < 3; i++) {
    if (results[i].dist < 0) {
      continue;
    }
    if (best_result.dist < 0 || results[i].dist < best_result.dist) {
      best_result = results[i];
    }
  }
  return best_result;
}

int SetCamera(const mjModel* m, mjvCamera* camera, int request_idx) {
  const int ncam = m ? m->ncam : 0;
  const int camera_idx = std::clamp(request_idx, kTumbleCameraIdx, ncam - 1);

  if (camera_idx == kTumbleCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->fixedcamid = -1;
  } else if (camera_idx == kFreeCameraIdx) {
    camera->type = mjCAMERA_FREE;
    camera->distance = 2.0f;
    camera->fixedcamid = -1;
  } else if (camera_idx == kTrackingCameraIdx) {
    if (camera->trackbodyid >= 0) {
      camera->type = mjCAMERA_TRACKING;
    } else {
      camera->type = mjCAMERA_FREE;
    }
    camera->fixedcamid = -1;
  } else {
    camera->type = mjCAMERA_FIXED;
    camera->fixedcamid = camera_idx;
  }

  return camera_idx;
}
}  // namespace mujoco::studio
