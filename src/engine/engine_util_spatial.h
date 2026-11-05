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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_H_
#define MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_H_

#include <mujoco/mjexport.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif
//------------------------------ quaternion operations ---------------------------------------------

// rotate vector by quaternion
MJAPI void mju_rotVecQuat(mjtNum res[3], const mjtNum vec[3], const mjtNum quat[4]);

// compute conjugate quaternion, corresponding to opposite rotation
MJAPI void mju_negQuat(mjtNum res[4], const mjtNum quat[4]);

// multiply quaternions
MJAPI void mju_mulQuat(mjtNum res[4], const mjtNum quat1[4], const mjtNum quat2[4]);

// multiply quaternion and axis
MJAPI void mju_mulQuatAxis(mjtNum res[4], const mjtNum quat[4], const mjtNum axis[3]);

// convert axisAngle to quaternion
MJAPI void mju_axisAngle2Quat(mjtNum res[4], const mjtNum axis[3], mjtNum angle);

// convert quaternion (corresponding to orientation difference) to 3D velocity
MJAPI void mju_quat2Vel(mjtNum res[3], const mjtNum quat[4], mjtNum dt);

// subtract quaternions, convert to 3D velocity: qb*quat(res) = qa
MJAPI void mju_subQuat(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]);

// convert quaternion to 3D rotation matrix
MJAPI void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]);

// convert 3D rotation matrix to quaternion
MJAPI void mju_mat2Quat(mjtNum quat[4], const mjtNum mat[9]);

// time-derivative of quaternion, given 3D rotational velocity
MJAPI void mju_derivQuat(mjtNum res[4], const mjtNum quat[4], const mjtNum vel[3]);

// time-derivative of expmap in quat = quat0 * exp(expmap), given 3D rotational velocity in the
//  rotated frame; expmap is first reduced to angle in (-pi, pi]; res and vel can alias
MJAPI void mju_derivExpmap(mjtNum res[3], const mjtNum expmap[3], const mjtNum vel[3]);

// integrate quaternion given 3D angular velocity
MJAPI void mju_quatIntegrate(mjtNum quat[4], const mjtNum vel[3], mjtNum scale);

// compute quaternion performing rotation from z-axis to given vector
MJAPI void mju_quatZ2Vec(mjtNum quat[4], const mjtNum vec[3]);

// eigenvalue decomposition of symmetric 3x3 matrix
MJAPI int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9]);

// same as mju_eig3, stop when off-diagonal elements are below reltol times the largest element
MJAPI int mju_eig3Tol(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9],
                      mjtNum reltol);

// extract 3D rotation from an arbitrary 3x3 matrix by refining the input quaternion
// returns the number of iterations required to converge
MJAPI int mju_mat2Rot(mjtNum quat[4], const mjtNum mat[9]);


//------------------------------ pose operations (pos, quat) ---------------------------------------

// multiply two poses
MJAPI void mju_mulPose(mjtNum posres[3], mjtNum quatres[4],
                       const mjtNum pos1[3], const mjtNum quat1[4],
                       const mjtNum pos2[3], const mjtNum quat2[4]);

// compute conjugate pose, corresponding to the opposite spatial transformation
MJAPI void mju_negPose(mjtNum posres[3], mjtNum quatres[4],
                       const mjtNum pos[3], const mjtNum quat[4]);

// transform vector by pose
MJAPI void mju_trnVecPose(mjtNum res[3], const mjtNum pos[3], const mjtNum quat[4],
                          const mjtNum vec[3]);

// convert sequence of Euler angles (radians) to quaternion
// seq[0,1,2] must be in 'xyzXYZ', lower/upper-case mean intrinsic/extrinsic rotations
MJAPI void mju_euler2Quat(mjtNum quat[4], const mjtNum euler[3], const char* seq);

//------------------------------ spatial algebra ---------------------------------------------------

// vector cross-product, 3D
MJAPI void mju_cross(mjtNum res[3], const mjtNum a[3], const mjtNum b[3]);

// cross-product for motion vector
void mju_crossMotion(mjtNum res[6], const mjtNum vel[6], const mjtNum v[6]);

// cross-product for force vectors
void mju_crossForce(mjtNum res[6], const mjtNum vel[6], const mjtNum f[6]);

// express inertia in com-based frame
void mju_inertCom(mjtNum* res, const mjtNum inert[3], const mjtNum mat[9],
                  const mjtNum dif[3], mjtNum mass);

// express motion axis in com-based frame
void mju_dofCom(mjtNum* res, const mjtNum axis[3], const mjtNum offset[3]);

// multiply 6D vector (rotation, translation) by 6D inertia matrix
MJAPI void mju_mulInertVec(mjtNum* res, const mjtNum inert[10], const mjtNum vec[6]);

// multiply dof matrix by vector
void mju_mulDofVec(mjtNum* res, const mjtNum* mat, const mjtNum* vec, int n);

// coordinate transform of 6D motion or force vector in rotation:translation format
//  rotnew2old is 3-by-3, NULL means no rotation; flg_force specifies force or motion type
MJAPI void mju_transformSpatial(mjtNum res[6], const mjtNum vec[6], int flg_force,
                                const mjtNum newpos[3], const mjtNum oldpos[3],
                                const mjtNum rotnew2old[9]);

// make 3D frame given X axis (and possibly Y axis)
void mju_makeFrame(mjtNum frame[9]);

#ifdef __cplusplus
}
#endif
#endif  // MUJOCO_SRC_ENGINE_ENGINE_UTIL_SPATIAL_H_
