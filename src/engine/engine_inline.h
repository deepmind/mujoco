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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_INLINE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_INLINE_H_

#include <assert.h>

#include <mujoco/mjtype.h>
#include "engine/engine_util_blas.h"
#include "engine/engine_util_blas_avx.h"  // IWYU pragma: keep
#include "engine/engine_util_spatial_avx.h"  // IWYU pragma: keep

#ifdef __cplusplus
  #if defined(_MSC_VER)
    #define restrict __restrict
  #else
    #define restrict __restrict__
  #endif
  extern "C" {
#endif

/* =================================================================================================

Private, high efficiency inlined functions for internal engine use.
Inlining on its own does nothing due to LTO, so criteria for mji_ functions are:
- More efficient assembly output due to `restrict`.
- Skipped copies due to non-alias guarantee (e.g., compare mju_cross and mji_cross).

mji_ functions:
- Should be modified as little as possible from their mju_ counterparts.
- Should be used only on the hotpath to prevent future bugs due to aliasing.

================================================================================================= */


//------------------------------ 3D vector and matrix-vector operations ----------------------------

// res = 0
static inline
void mji_zero3(mjtNum* restrict res) {
  res[0] = 0;
  res[1] = 0;
  res[2] = 0;
}


// res = vec
static inline
void mji_copy3(mjtNum* restrict res, const mjtNum *vec) {
  res[0] = vec[0];
  res[1] = vec[1];
  res[2] = vec[2];
}


// res = vec*scl
static inline
void mji_scl3(mjtNum* restrict res, const mjtNum vec[3], mjtNum scl) {
  res[0] = vec[0] * scl;
  res[1] = vec[1] * scl;
  res[2] = vec[2] * scl;
}


// res = vec1 + vec2
static inline
void mji_add3(mjtNum* restrict res, const mjtNum vec1[3], const mjtNum vec2[3]) {
  res[0] = vec1[0] + vec2[0];
  res[1] = vec1[1] + vec2[1];
  res[2] = vec1[2] + vec2[2];
}


// res = vec1 - vec2
static inline
void mji_sub3(mjtNum* restrict res, const mjtNum vec1[3], const mjtNum vec2[3]) {
  res[0] = vec1[0] - vec2[0];
  res[1] = vec1[1] - vec2[1];
  res[2] = vec1[2] - vec2[2];
}


// res += vec
static inline
void mji_addTo3(mjtNum* restrict res, const mjtNum vec[3]) {
  res[0] += vec[0];
  res[1] += vec[1];
  res[2] += vec[2];
}


// res -= vec
static inline
void mji_subFrom3(mjtNum* restrict res, const mjtNum vec[3]) {
  res[0] -= vec[0];
  res[1] -= vec[1];
  res[2] -= vec[2];
}


// res += vec*scl
static inline
void mji_addToScl3(mjtNum* restrict res, const mjtNum vec[3], mjtNum scl) {
  res[0] += vec[0] * scl;
  res[1] += vec[1] * scl;
  res[2] += vec[2] * scl;
}


// res = vec1 + vec2*scl
static inline
void mji_addScl3(mjtNum* restrict res, const mjtNum* vec1, const mjtNum* vec2, mjtNum scl) {
  res[0] = vec1[0] + scl*vec2[0];
  res[1] = vec1[1] + scl*vec2[1];
  res[2] = vec1[2] + scl*vec2[2];
}


// normalize vector, return length before normalization
// (for use in this file only)
static inline
mjtNum mji__normalize3(mjtNum vec[3]) {
  mjtNum norm = mju_sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

  if (norm < mjMINVAL) {
    vec[0] = 1;
    vec[1] = 0;
    vec[2] = 0;
  } else {
    mjtNum normInv = 1/norm;
    vec[0] *= normInv;
    vec[1] *= normInv;
    vec[2] *= normInv;
  }

  return norm;
}


// multiply vector by 3D rotation matrix
static inline
void mji_mulMatVec3(mjtNum* restrict res, const mjtNum mat[9], const mjtNum vec[3]) {
  assert(res != vec);
  res[0] = mat[0]*vec[0] + mat[1]*vec[1] + mat[2]*vec[2];
  res[1] = mat[3]*vec[0] + mat[4]*vec[1] + mat[5]*vec[2];
  res[2] = mat[6]*vec[0] + mat[7]*vec[1] + mat[8]*vec[2];
}


// multiply vector by transposed 3D rotation matrix
static inline
void mji_mulMatTVec3(mjtNum* restrict res, const mjtNum mat[9], const mjtNum vec[3]) {
  assert(res != vec);
  res[0] = mat[0]*vec[0] + mat[3]*vec[1] + mat[6]*vec[2];
  res[1] = mat[1]*vec[0] + mat[4]*vec[1] + mat[7]*vec[2];
  res[2] = mat[2]*vec[0] + mat[5]*vec[1] + mat[8]*vec[2];
}


// multiply 3x3 matrices,
static inline
void mji_mulMatMat3(mjtNum* restrict res, const mjtNum mat1[9], const mjtNum mat2[9]) {
  assert(res != mat1 && res != mat2);
  res[0] = mat1[0]*mat2[0] + mat1[1]*mat2[3] + mat1[2]*mat2[6];
  res[1] = mat1[0]*mat2[1] + mat1[1]*mat2[4] + mat1[2]*mat2[7];
  res[2] = mat1[0]*mat2[2] + mat1[1]*mat2[5] + mat1[2]*mat2[8];
  res[3] = mat1[3]*mat2[0] + mat1[4]*mat2[3] + mat1[5]*mat2[6];
  res[4] = mat1[3]*mat2[1] + mat1[4]*mat2[4] + mat1[5]*mat2[7];
  res[5] = mat1[3]*mat2[2] + mat1[4]*mat2[5] + mat1[5]*mat2[8];
  res[6] = mat1[6]*mat2[0] + mat1[7]*mat2[3] + mat1[8]*mat2[6];
  res[7] = mat1[6]*mat2[1] + mat1[7]*mat2[4] + mat1[8]*mat2[7];
  res[8] = mat1[6]*mat2[2] + mat1[7]*mat2[5] + mat1[8]*mat2[8];
}


// multiply 3x3 matrices, first argument transposed
static inline
void mji_mulMatTMat3(mjtNum* restrict res, const mjtNum mat1[9], const mjtNum mat2[9]) {
  assert(res != mat1 && res != mat2);
  res[0] = mat1[0]*mat2[0] + mat1[3]*mat2[3] + mat1[6]*mat2[6];
  res[1] = mat1[0]*mat2[1] + mat1[3]*mat2[4] + mat1[6]*mat2[7];
  res[2] = mat1[0]*mat2[2] + mat1[3]*mat2[5] + mat1[6]*mat2[8];
  res[3] = mat1[1]*mat2[0] + mat1[4]*mat2[3] + mat1[7]*mat2[6];
  res[4] = mat1[1]*mat2[1] + mat1[4]*mat2[4] + mat1[7]*mat2[7];
  res[5] = mat1[1]*mat2[2] + mat1[4]*mat2[5] + mat1[7]*mat2[8];
  res[6] = mat1[2]*mat2[0] + mat1[5]*mat2[3] + mat1[8]*mat2[6];
  res[7] = mat1[2]*mat2[1] + mat1[5]*mat2[4] + mat1[8]*mat2[7];
  res[8] = mat1[2]*mat2[2] + mat1[5]*mat2[5] + mat1[8]*mat2[8];
}


// transpose 3x3 matrix
static inline
void mji_transpose3(mjtNum* restrict res, const mjtNum mat[9]) {
  assert(res != mat);
  res[0] = mat[0];
  res[1] = mat[3];
  res[2] = mat[6];
  res[3] = mat[1];
  res[4] = mat[4];
  res[5] = mat[7];
  res[6] = mat[2];
  res[7] = mat[5];
  res[8] = mat[8];
}


//------------------------------ 4D vector and matrix-vector operations ----------------------------

// res = vec
static inline
void mji_copy4(mjtNum* restrict res, const mjtNum data[4]) {
#ifdef mjUSEAVX
  mju_copy4_avx(res, data);
#else
  res[0] = data[0];
  res[1] = data[1];
  res[2] = data[2];
  res[3] = data[3];
#endif
}


// normalize vector, return length before normalization
// (for use in this file only)
static inline
mjtNum mji__normalize4(mjtNum vec[4]) {
#ifdef mjUSEAVX
  return mju_normalize4_avx(vec);
#else
  mjtNum norm = mju_sqrt((vec[0]*vec[0] + vec[1]*vec[1]) + (vec[2]*vec[2] + vec[3]*vec[3]));

  if (norm < mjMINVAL) {
    vec[0] = 1;
    vec[1] = 0;
    vec[2] = 0;
    vec[3] = 0;
  } else if (mju_abs(norm - 1) > mjMINVAL) {
    mjtNum normInv = 1/norm;
    vec[0] *= normInv;
    vec[1] *= normInv;
    vec[2] *= normInv;
    vec[3] *= normInv;
  }

  return norm;
#endif
}


//------------------------------ quaternion operations ---------------------------------------------

// rotate vector by quaternion
static inline
void mji_rotVecQuat(mjtNum* restrict res, const mjtNum vec[3], const mjtNum quat[4]) {
  assert(res != quat);
  // null quat: copy vec
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    mji_copy3(res, vec);
  }

  // regular processing
  else {
    mjtNum tmp[3];
    // tmp = q_w * v + cross(q_xyz, v)
    tmp[0] = quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1];
    tmp[1] = quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2];
    tmp[2] = quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0];
    // res = v + 2 * cross(q_xyz, t)
    res[0] = vec[0] + 2 * (quat[2]*tmp[2] - quat[3]*tmp[1]);
    res[1] = vec[1] + 2 * (quat[3]*tmp[0] - quat[1]*tmp[2]);
    res[2] = vec[2] + 2 * (quat[1]*tmp[1] - quat[2]*tmp[0]);
  }
}


// negate quaternion
static inline
void mji_negQuat(mjtNum* restrict res, const mjtNum quat[4]) {
#ifdef mjUSEAVX
  mju_negQuat_avx(res, quat);
#else
  res[0] = quat[0];
  res[1] = -quat[1];
  res[2] = -quat[2];
  res[3] = -quat[3];
#endif
}


// multiply quaternions
static inline
void mji_mulQuat(mjtNum* restrict res, const mjtNum qa[4], const mjtNum qb[4]) {
  assert(res != qa && res != qb);
#ifdef mjUSEAVX
  mju_mulQuat_avx(res, qa, qb);
#else
  res[0] = (qa[0]*qb[0] - qa[1]*qb[1]) + (-qa[2]*qb[2] - qa[3]*qb[3]);
  res[1] = (qa[0]*qb[1] + qa[1]*qb[0]) + ( qa[2]*qb[3] - qa[3]*qb[2]);
  res[2] = (qa[0]*qb[2] - qa[1]*qb[3]) + ( qa[2]*qb[0] + qa[3]*qb[1]);
  res[3] = (qa[0]*qb[3] + qa[1]*qb[2]) + (-qa[2]*qb[1] + qa[3]*qb[0]);
#endif
}


// multiply quaternion and axis
static inline
void mji_mulQuatAxis(mjtNum* restrict res, const mjtNum quat[4], const mjtNum axis[3]) {
  assert(res != quat && res != axis);
#ifdef mjUSEAVX
  mju_mulQuatAxis_avx(res, quat, axis);
#else
  res[0] = -quat[1]*axis[0] - quat[2]*axis[1] - quat[3]*axis[2];
  res[1] =  quat[0]*axis[0] - quat[3]*axis[1] + quat[2]*axis[2];
  res[2] =  quat[3]*axis[0] + quat[0]*axis[1] - quat[1]*axis[2];
  res[3] = -quat[2]*axis[0] + quat[1]*axis[1] + quat[0]*axis[2];
#endif
}


// convert axisAngle to quaternion
static inline
void mji_axisAngle2Quat(mjtNum* restrict res, const mjtNum axis[3], mjtNum angle) {
  // zero angle: null quat
  if (angle == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
  }

  // regular processing
  else {
    mjtNum s = mju_sin(angle*0.5);
    res[0] = mju_cos(angle*0.5);
    res[1] = axis[0]*s;
    res[2] = axis[1]*s;
    res[3] = axis[2]*s;
  }
}


// convert quaternion (corresponding to orientation difference) to 3D velocity
static inline
void mji_quat2Vel(mjtNum* restrict res, const mjtNum quat[4], mjtNum dt) {
  mjtNum axis[3] = {quat[1], quat[2], quat[3]};
  mjtNum sin_a_2 = mji__normalize3(axis);
  mjtNum speed = 2 * mju_atan2(sin_a_2, quat[0]);

  // when axis-angle is larger than pi, rotation is in the opposite direction
  if (speed > mjPI) {
    speed -= 2*mjPI;
  }
  speed /= dt;

  mji_scl3(res, axis, speed);
}


// Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
static inline
void mji_subQuat(mjtNum* restrict res, const mjtNum qa[4], const mjtNum qb[4]) {
  // qdif = neg(qb)*qa
  mjtNum qneg[4], qdif[4];
  mji_negQuat(qneg, qb);
  mji_mulQuat(qdif, qneg, qa);

  // convert to 3D velocity
  mji_quat2Vel(res, qdif, 1);
}


// convert quaternion to 3D rotation matrix
static inline
void mji_quat2Mat(mjtNum* restrict res, const mjtNum quat[4]) {
  assert(res != quat);
#ifdef mjUSEAVX
  mju_quat2Mat_avx(res, quat);
#else
  // null quat: identity
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
    res[4] = 1;
    res[5] = 0;
    res[6] = 0;
    res[7] = 0;
    res[8] = 1;
  }

  // regular processing
  else {
    mjtNum q00 = quat[0]*quat[0];
    mjtNum q01 = quat[0]*quat[1];
    mjtNum q02 = quat[0]*quat[2];
    mjtNum q03 = quat[0]*quat[3];
    mjtNum q11 = quat[1]*quat[1];
    mjtNum q12 = quat[1]*quat[2];
    mjtNum q13 = quat[1]*quat[3];
    mjtNum q22 = quat[2]*quat[2];
    mjtNum q23 = quat[2]*quat[3];
    mjtNum q33 = quat[3]*quat[3];

    res[0] = (q00 - q22) + (q11 - q33);
    res[4] = (q00 + q22) - (q11 + q33);
    res[8] = (q00 - q22) - (q11 - q33);

    res[1] = 2*(q12 - q03);
    res[2] = 2*(q13 + q02);
    res[3] = 2*(q12 + q03);
    res[5] = 2*(q23 - q01);
    res[6] = 2*(q13 - q02);
    res[7] = 2*(q23 + q01);
  }
#endif
}


// convert 3D rotation matrix to quaternion
static inline
void mji_mat2Quat(mjtNum* restrict quat, const mjtNum mat[9]) {
  assert(quat != mat);
  // q0 largest
  if (mat[0]+mat[4]+mat[8]>0) {
    quat[0] = 0.5 * mju_sqrt(1 + mat[0] + mat[4] + mat[8]);
    quat[1] = 0.25 * (mat[7] - mat[5]) / quat[0];
    quat[2] = 0.25 * (mat[2] - mat[6]) / quat[0];
    quat[3] = 0.25 * (mat[3] - mat[1]) / quat[0];
  }

  // q1 largest
  else if (mat[0]>mat[4] && mat[0]>mat[8]) {
    quat[1] = 0.5 * mju_sqrt(1 + mat[0] - mat[4] - mat[8]);
    quat[0] = 0.25 * (mat[7] - mat[5]) / quat[1];
    quat[2] = 0.25 * (mat[1] + mat[3]) / quat[1];
    quat[3] = 0.25 * (mat[2] + mat[6]) / quat[1];
  }

  // q2 largest
  else if (mat[4]>mat[8]) {
    quat[2] = 0.5 * mju_sqrt(1 - mat[0] + mat[4] - mat[8]);
    quat[0] = 0.25 * (mat[2] - mat[6]) / quat[2];
    quat[1] = 0.25 * (mat[1] + mat[3]) / quat[2];
    quat[3] = 0.25 * (mat[5] + mat[7]) / quat[2];
  }

  // q3 largest
  else {
    quat[3] = 0.5 * mju_sqrt(1 - mat[0] - mat[4] + mat[8]);
    quat[0] = 0.25 * (mat[3] - mat[1]) / quat[3];
    quat[1] = 0.25 * (mat[2] + mat[6]) / quat[3];
    quat[2] = 0.25 * (mat[5] + mat[7]) / quat[3];
  }

  mji__normalize4(quat);
}


// integrate quaternion given 3D angular velocity
static inline
void mji_quatIntegrate(mjtNum* restrict quat, const mjtNum vel[3], mjtNum scale) {
  assert(quat != vel);
  mjtNum angle, tmp[4], qrot[4];

  // form local rotation quaternion, apply
  mji_copy3(tmp, vel);
  angle = scale * mji__normalize3(tmp);
  mji_axisAngle2Quat(qrot, tmp, angle);
  mji__normalize4(quat);
  mji_copy4(tmp, quat);
  mji_mulQuat(quat, tmp, qrot);
}


//------------------------------ spatial algebra ---------------------------------------------------

// vector cross-product, 3D
static inline
void mji_cross(mjtNum* restrict res, const mjtNum a[3], const mjtNum b[3]) {
  assert(res != a && res != b);
  res[0] = a[1]*b[2] - a[2]*b[1];
  res[1] = a[2]*b[0] - a[0]*b[2];
  res[2] = a[0]*b[1] - a[1]*b[0];
}


// cross-product for motion vector
static inline
void mji_crossMotion(mjtNum* restrict res, const mjtNum vel[6], const mjtNum v[6]) {
  assert(res != vel && res != v);
  res[0] = -vel[2]*v[1] + vel[1]*v[2];
  res[1] =  vel[2]*v[0] - vel[0]*v[2];
  res[2] = -vel[1]*v[0] + vel[0]*v[1];
  res[3] = -vel[2]*v[4] + vel[1]*v[5];
  res[4] =  vel[2]*v[3] - vel[0]*v[5];
  res[5] = -vel[1]*v[3] + vel[0]*v[4];

  res[3] += -vel[5]*v[1] + vel[4]*v[2];
  res[4] +=  vel[5]*v[0] - vel[3]*v[2];
  res[5] += -vel[4]*v[0] + vel[3]*v[1];
}


// cross-product for force vectors
static inline
void mji_crossForce(mjtNum* restrict res, const mjtNum vel[6], const mjtNum f[6]) {
  assert(res != vel && res != f);
  res[0] = -vel[2]*f[1] + vel[1]*f[2];
  res[1] =  vel[2]*f[0] - vel[0]*f[2];
  res[2] = -vel[1]*f[0] + vel[0]*f[1];
  res[3] = -vel[2]*f[4] + vel[1]*f[5];
  res[4] =  vel[2]*f[3] - vel[0]*f[5];
  res[5] = -vel[1]*f[3] + vel[0]*f[4];

  res[0] += -vel[5]*f[4] + vel[4]*f[5];
  res[1] +=  vel[5]*f[3] - vel[3]*f[5];
  res[2] += -vel[4]*f[3] + vel[3]*f[4];
}


// 6D vector dot-product
static inline
mjtNum mji_dot6(const mjtNum vec1[6], const mjtNum vec2[6]) {
  // match order of operations to mju_dot
  return ((vec1[0] * vec2[0] + vec1[2] * vec2[2]) +
          (vec1[1] * vec2[1] + vec1[3] * vec2[3])) +
          (vec1[4] * vec2[4] + vec1[5] * vec2[5]);
}


// res = vec
static inline
void mji_copy6(mjtNum* restrict res, const mjtNum *vec) {
  res[0] = vec[0];
  res[1] = vec[1];
  res[2] = vec[2];
  res[3] = vec[3];
  res[4] = vec[4];
  res[5] = vec[5];
}


// res = vec
static inline
void mji_copy9(mjtNum* restrict res, const mjtNum data[9]) {
  res[0] = data[0];
  res[1] = data[1];
  res[2] = data[2];
  res[3] = data[3];
  res[4] = data[4];
  res[5] = data[5];
  res[6] = data[6];
  res[7] = data[7];
  res[8] = data[8];
}


#ifdef __cplusplus
  }  // extern "C"
  #undef restrict
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_INLINE_H_
