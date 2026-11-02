// Copyright 2022 DeepMind Technologies Limited
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

// Tests for engine/engine_util_spatial.c

#include <cmath>
#include <cstdlib>
#include <cstring>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_inline.h"  // IWYU pragma: keep
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_spatial.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointwise;

using Quat2MatTest = MujocoTest;

TEST_F(Quat2MatTest, NoRotation) {
  mjtNum result[9] = {0};
  mjtNum quat[] = {1, 0, 0, 0};
  mju_quat2Mat(result, quat);
  EXPECT_THAT(AsVector(result, 9), ElementsAre(1, 0, 0, 0, 1, 0, 0, 0, 1));
}

TEST_F(Quat2MatTest, TinyRotation) {
  mjtNum result[9] = {0};
  // An angle so small that cos(angle) == 1.0 to double accuracy
  mjtNum angle = 1e-8;
  mjtNum quat[] = {cos(angle / 2), sin(angle / 2), 0, 0};
  mju_quat2Mat(result, quat);
  EXPECT_THAT(AsVector(result, 9),
              ElementsAre(1, 0, 0, 0, cos(angle), -sin(angle), 0, sin(angle),
                          cos(angle)));
}

using MulQuatTest = MujocoTest;

TEST_F(MulQuatTest, TinyRotation) {
  mjtNum null_quat[4] = {1, 0, 0, 0};
  mjtNum result[4];
  // An angle so small that cos(angle) == 1.0 to double accuracy
  mjtNum angle = 1e-8;
  mjtNum quat[] = {cos(angle / 2), sin(angle / 2), 0, 0};
  mju_mulQuat(result, null_quat, quat);
  EXPECT_THAT(AsVector(result, 4),
              ElementsAre(cos(angle / 2), sin(angle / 2), 0, 0));
}

using RotVecQuatTest = MujocoTest;

TEST_F(RotVecQuatTest, NoRotation) {
  mjtNum result[3];
  mjtNum vec[] = {1, 2, 3};
  mjtNum quat[] = {1, 0, 0, 0};
  mju_rotVecQuat(result, vec, quat);
  EXPECT_THAT(AsVector(result, 3), ElementsAre(1, 2, 3));
}

TEST_F(RotVecQuatTest, TinyRotation) {
  mjtNum result[3];
  mjtNum vec[] = {0, 1, 0};
  // An angle so small that cos(angle) == 1.0 to double accuracy
  mjtNum angle = 1e-8;
  mjtNum quat[] = {cos(angle / 2), sin(angle / 2), 0, 0};
  mju_rotVecQuat(result, vec, quat);
  EXPECT_THAT(AsVector(result, 3), ElementsAre(0, cos(angle), sin(angle)));
}

// Rotate a vector by explicitly converting the quaternion to a 3x3 matrix
void RotVecQuatWithMatrix(mjtNum res[3], const mjtNum vec[3],
                          const mjtNum quat[4]) {
  if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    mju_copy3(res, vec);
  } else {
    mjtNum mat[9];
    mju_quat2Mat(mat, quat);
    mju_mulMatVec3(res, mat, vec);
  }
}

TEST_F(RotVecQuatTest, TestEquivalence) {
  mjtNum resultActual[3], resultExpected[3], quat[4];
  // List of rotation axes
  mjtNum vecs[5][3] = {
      {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-0.5, 1, -0.5}, {1.22, -2.33, 3.44}};
  // List of angles to rotate by, in degrees
  mjtNum angles[6] = {0.0, 1e-8, 31, 47, 181, 271};
  static const mjtNum eps = MjTol(1e-15, 1e-5);
  for (auto vec : vecs) {
    // Unit-normalize the vector
    mju_normalize3(vec);
    for (auto angleDegree : angles) {
      // Convert the axis-angle to a quaternion
      auto angleRad = angleDegree * mjPI / 180;
      mju_axisAngle2Quat(quat, vec, angleRad);
      // Rotate
      mju_rotVecQuat(resultActual, vec, quat);
      RotVecQuatWithMatrix(resultExpected, vec, quat);
      // Compare
      EXPECT_NEAR(resultExpected[0], resultActual[0], eps);
      EXPECT_NEAR(resultExpected[1], resultActual[1], eps);
      EXPECT_NEAR(resultExpected[2], resultActual[2], eps);
    }
  }
}

using Euler2QuatTest = MujocoTest;

TEST_F(Euler2QuatTest, BadSeq) {
  EXPECT_FATAL_FAILURE(
      {
        mjtNum quat[4];
        mjtNum euler[3] = {0};
        char seq[] = "xiz";
        mju_euler2Quat(quat, euler, seq);
      },
      "mju_euler2Quat: seq[1] is 'i', should be one of x, y, z, X, Y, Z");
}

TEST_F(Euler2QuatTest, BadSeqLength) {
  EXPECT_FATAL_FAILURE(
      {
        mjtNum quat[4];
        mjtNum euler[3] = {0};
        char seq[] = "xyzy";
        mju_euler2Quat(quat, euler, seq);
      },
      "mju_euler2Quat: seq must contain exactly 3 characters");
}

TEST_F(Euler2QuatTest, Euler2Quat) {
  mjtNum quat[4] = {0};
  mjtNum tol = MjTol(1e-14, 1e-6);

  char seq[] = "xyz";
  mjtNum euler[3] = {mjPI, 0, 0};
  mjtNum expected[4] = {0, 1, 0, 0};
  mju_euler2Quat(quat, euler, seq);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected));

  euler[1] = mjPI;
  mjtNum expected2[4] = {0, 0, 0, 1};
  mju_euler2Quat(quat, euler, seq);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected2));

  char seq2[] = "XYZ";
  mjtNum expected3[4] = {0, 0, 0, -1};
  mju_euler2Quat(quat, euler, seq2);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected3));

  mjtNum euler2[3] = {2 * mjPI, 2 * mjPI, 2 * mjPI};
  mjtNum expected4[4] = {-1, 0, 0, 0};
  mju_euler2Quat(quat, euler2, seq);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected4));
  mju_euler2Quat(quat, euler2, seq2);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected4));

  mjtNum euler3[3] = {mjPI / 2, mjPI / 2, mjPI / 2};
  mjtNum expected5[4] = {0, mju_sqrt(.5), 0, mju_sqrt(.5)};
  mju_euler2Quat(quat, euler3, seq);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected5));
  mju_euler2Quat(quat, euler3, seq2);
  mjtNum expected6[4] = {mju_sqrt(.5), 0, mju_sqrt(.5), 0};
  EXPECT_THAT(quat, Pointwise(MjNear(tol, tol), expected6));
}

using Mat2RotTest = MujocoTest;

TEST_F(Mat2RotTest, RotationFromArbitraryMatrix) {
  // create arbitrary target rotation matrix
  mjtNum target[4], rot[9];
  mjtNum axis[3] = {1, 1, 1};
  mju_axisAngle2Quat(target, axis, mjPI / 6);
  mju_normalize4(target);
  mju_quat2Mat(rot, target);

  // combine rotation with arbitrary stretch
  mjtNum mat[9];
  // clang-format off
  mjtNum deformation_gradient[9] = {0.5, 0.25, 0.125,
                                    0.3, 0.66, 0.999,
                                    0.4, 0.22, 0.111};
  // clang-format on
  mjtNum stretch[9];
  mju_mulMatTMat3(stretch, deformation_gradient, deformation_gradient);
  mju_mulMatMat3(mat, rot, stretch);

  // calculate rotational part of the matrix
  mjtNum quat[4] = {1, 0, 0, 0};
  int niter = mju_mat2Rot(quat, mat);
  EXPECT_THAT(quat, Pointwise(MjNear(1e-8, 1e-5), target));
  EXPECT_LE(niter, 150);
}

TEST_F(Mat2RotTest, IdentityFromRandomRotation) {
  // This test is based on the following paper:
  // Müller, Matthias, Jan Bender, Nuttapong Chentanez, and Miles Macklin. "A
  // robust method to extract the rotational part of deformations." In
  // Proceedings of the 9th International Conference on Motion in Games, pp.
  // 55-60. 2016.
  mjtNum mat[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  srand(123);

  for (int i = 0; i < 100; ++i) {
    // random quaternion
    mjtNum quat[4];
    for (int j = 0; j < 4; ++j) {
      quat[j] = rand() / (float)RAND_MAX;  // NOLINT
    }

    // calculate rotational part of the matrix
    mjtNum res[9];
    mju_normalize4(quat);
    EXPECT_LE(mju_mat2Rot(quat, mat), 40);
    mju_quat2Mat(res, quat);
    EXPECT_THAT(res, Pointwise(MjNear(1e-6, 1e-5), mat));
  }
}

TEST_F(Mat2RotTest, SpecialCases) {
  mjtNum eye[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjtNum quat[4] = {1, 0, 0, 0};
  EXPECT_EQ(mju_mat2Rot(quat, eye), 0);
  EXPECT_THAT(quat, Pointwise(MjNear(1e-8, 1e-8), {1, 0, 0, 0}));
  mjtNum unnorm_quat[4] = {2, 0, 0, 0};
  EXPECT_EQ(mju_mat2Rot(unnorm_quat, eye), 0);
  EXPECT_THAT(unnorm_quat, Pointwise(MjNear(1e-8, 1e-8), {1, 0, 0, 0}));
  EXPECT_NEAR(mju_norm(unnorm_quat, 4), 1, MjTol(1e-12, 1e-6));
  mjtNum zero[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(mju_mat2Rot(quat, zero), 0);
  EXPECT_THAT(quat, Pointwise(MjNear(1e-8, 1e-4), {1, 0, 0, 0}));
  mjtNum ones[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
  EXPECT_EQ(mju_mat2Rot(quat, ones), 0);
  EXPECT_THAT(quat, Pointwise(MjNear(1e-8, 1e-4), {1, 0, 0, 0}));
}

TEST_F(Mat2RotTest, StretchedHalfTurn) {
  // mat = diag(2.5, -1, -1): trace is 0.5 > 0 from identity, but polar rotation
  // is 180 deg about x
  mjtNum mat[9] = {2.5, 0, 0, 0, -1, 0, 0, 0, -1};
  mjtNum quat[4] = {1, 0, 0, 0};
  EXPECT_LT(mju_mat2Rot(quat, mat), 100);
  mjtNum target[4] = {0, 1, 0, 0};
  mjtNum res[9], exp_mat[9];
  mju_quat2Mat(res, quat);
  mju_quat2Mat(exp_mat, target);
  EXPECT_THAT(res, Pointwise(MjNear(1e-8, 1e-5), exp_mat));
  EXPECT_NEAR(mju_norm(quat, 4), 1, MjTol(1e-12, 1e-6));

  // rotated off-axis: R * diag(2.5, -1, -1) * R^T
  mjtNum rot[9], rot_T[9], tmp[9], rot_mat[9];
  mjtNum qrot[4] = {0.5, 0.5, 0.5, 0.5};
  mju_normalize4(qrot);
  mju_quat2Mat(rot, qrot);
  mji_mulMatMat3(tmp, rot, mat);
  mji_transpose3(rot_T, rot);
  mji_mulMatMat3(rot_mat, tmp, rot_T);

  mjtNum exp_rot_mat[9];
  mji_mulMatMat3(tmp, rot, exp_mat);
  mji_mulMatMat3(exp_rot_mat, tmp, rot_T);

  mjtNum quat_off[4] = {1, 0, 0, 0};
  EXPECT_LT(mju_mat2Rot(quat_off, rot_mat), 100);
  mjtNum res_off[9];
  mju_quat2Mat(res_off, quat_off);
  EXPECT_THAT(res_off, Pointwise(MjNear(1e-8, 1e-5), exp_rot_mat));
  EXPECT_NEAR(mju_norm(quat_off, 4), 1, MjTol(1e-12, 1e-6));
}

TEST_F(Mat2RotTest, HalfTurnFromIdentity) {
  // 180-degree rotations about x, y, z and a diagonal axis: the initial
  // identity is a stationary point with negative trace
  const mjtNum s = 1 / mju_sqrt(3);
  const mjtNum axes[4][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {s, s, s}};
  for (const mjtNum* axis : axes) {
    mjtNum target[4] = {0, axis[0], axis[1], axis[2]};
    mjtNum mat[9];
    mju_quat2Mat(mat, target);

    // unnormalized initial quaternion
    mjtNum quat[4] = {2, 0, 0, 0};
    EXPECT_LT(mju_mat2Rot(quat, mat), 100);
    mjtNum res[9];
    mju_quat2Mat(res, quat);
    EXPECT_THAT(res, Pointwise(MjNear(1e-8, 1e-5), mat));
    EXPECT_NEAR(mju_norm(quat, 4), 1, MjTol(1e-12, 1e-6));
  }
}

using DerivExpmapTest = MujocoTest;

// mju_derivExpmap is the rate of the rotation vector log(exp(expmap) * exp(t *
// vel)) at t = 0: compare against central finite differences, for angles below
// and above the polynomial cutoff (1.75), near pi, and above pi (where the rate
// is that of the equivalent rotation vector with angle in (-pi, pi])
TEST_F(DerivExpmapTest, MatchesFiniteDifference) {
#ifdef mjUSESINGLE
  const mjtNum h = 1e-2;
#else
  const mjtNum h = 1e-5;
#endif
  const mjtNum kVel[3] = {0.3, -1.2, 0.7};
  for (mjtNum angle : {0.0, 1e-3, 0.5, 1.7, 1.8, 3.0, 4.0, 7.0}) {
    mjtNum axis[3] = {1, 2, -2};
    mju_normalize3(axis);
    mjtNum expmap[3];
    mju_scl3(expmap, axis, angle);

    mjtNum res[3];
    mju_derivExpmap(res, expmap, kVel);

    // finite difference of the rotation vector
    mjtNum quat_p[4], quat_m[4], rot_p[3], rot_m[3], expected[3];
    mju_axisAngle2Quat(quat_p, axis, angle);
    mju_copy4(quat_m, quat_p);
    mju_quatIntegrate(quat_p, kVel, h);
    mju_quatIntegrate(quat_m, kVel, -h);
    mju_quat2Vel(rot_p, quat_p, 1);
    mju_quat2Vel(rot_m, quat_m, 1);
    mju_sub3(expected, rot_p, rot_m);
    mju_scl3(expected, expected, 1 / (2 * h));
    EXPECT_THAT(res, Pointwise(MjNear(1e-9, 1e-4), expected))
        << "angle " << angle;

    // in place
    mjtNum vel[3] = {kVel[0], kVel[1], kVel[2]};
    mju_derivExpmap(vel, expmap, vel);
    EXPECT_THAT(vel, Pointwise(MjNear(0, 0), res)) << "angle " << angle;
  }
}

using QuatSimdParityTest = MujocoTest;

// Reference scalar implementations matching CL 2 evaluation order bit-for-bit.
mjtNum ScalarNormalize4(mjtNum vec[4]) {
  mjtNum norm = mju_sqrt((vec[0] * vec[0] + vec[1] * vec[1]) +
                         (vec[2] * vec[2] + vec[3] * vec[3]));
  if (norm < mjMINVAL) {
    vec[0] = 1;
    vec[1] = 0;
    vec[2] = 0;
    vec[3] = 0;
  } else if (mju_abs(norm - 1) > mjMINVAL) {
    mjtNum norm_inv = 1 / norm;
    vec[0] *= norm_inv;
    vec[1] *= norm_inv;
    vec[2] *= norm_inv;
    vec[3] *= norm_inv;
  }
  return norm;
}

void ScalarNegQuat(mjtNum res[4], const mjtNum quat[4]) {
  res[0] = quat[0];
  res[1] = -quat[1];
  res[2] = -quat[2];
  res[3] = -quat[3];
}

void ScalarMulQuat(mjtNum res[4], const mjtNum qa[4], const mjtNum qb[4]) {
  mjtNum tmp[4] = {
      (qa[0] * qb[0] - qa[1] * qb[1]) + (-qa[2] * qb[2] - qa[3] * qb[3]),
      (qa[0] * qb[1] + qa[1] * qb[0]) + (qa[2] * qb[3] - qa[3] * qb[2]),
      (qa[0] * qb[2] - qa[1] * qb[3]) + (qa[2] * qb[0] + qa[3] * qb[1]),
      (qa[0] * qb[3] + qa[1] * qb[2]) + (-qa[2] * qb[1] + qa[3] * qb[0]),
  };
  std::memcpy(res, tmp, sizeof(tmp));
}

void ScalarMulQuatAxis(mjtNum res[4], const mjtNum quat[4],
                       const mjtNum axis[3]) {
  mjtNum tmp[4] = {
      -quat[1] * axis[0] - quat[2] * axis[1] - quat[3] * axis[2],
      quat[0] * axis[0] - quat[3] * axis[1] + quat[2] * axis[2],
      quat[3] * axis[0] + quat[0] * axis[1] - quat[1] * axis[2],
      -quat[2] * axis[0] + quat[1] * axis[1] + quat[0] * axis[2],
  };
  std::memcpy(res, tmp, sizeof(tmp));
}

void ScalarQuat2Mat(mjtNum res[9], const mjtNum quat[4]) {
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
    return;
  }
  mjtNum q00 = quat[0] * quat[0];
  mjtNum q01 = quat[0] * quat[1];
  mjtNum q02 = quat[0] * quat[2];
  mjtNum q03 = quat[0] * quat[3];
  mjtNum q11 = quat[1] * quat[1];
  mjtNum q12 = quat[1] * quat[2];
  mjtNum q13 = quat[1] * quat[3];
  mjtNum q22 = quat[2] * quat[2];
  mjtNum q23 = quat[2] * quat[3];
  mjtNum q33 = quat[3] * quat[3];

  res[0] = (q00 - q22) + (q11 - q33);
  res[4] = (q00 + q22) - (q11 + q33);
  res[8] = (q00 - q22) - (q11 - q33);

  res[1] = 2 * (q12 - q03);
  res[2] = 2 * (q13 + q02);
  res[3] = 2 * (q12 + q03);
  res[5] = 2 * (q23 - q01);
  res[6] = 2 * (q13 - q02);
  res[7] = 2 * (q23 + q01);
}

#if defined(mjUSEPLATFORMSIMD) && defined(__AVX__) && !defined(mjUSESINGLE)
  #ifndef mjUSEAVX
    #error \
        "mjUSEAVX must be defined when mjUSEPLATFORMSIMD and __AVX__ are active"
  #endif
#endif

TEST_F(QuatSimdParityTest, BitExactWithScalarAndInPlace) {
  const mjtNum kQuats[][4] = {
      {1, 0, 0, 0},
      {0, 0, 0, 0},
      {-0.0, 0.0, -0.0, 0.0},
      {1e-16, -2e-16, 3e-16, -1e-16},
      {0, 1, 0, 0},
      {0, 0, 0, 1},
      {0.5, -0.5, 0.5, -0.5},
      {0.7312, -0.4189, 0.2951, 0.4493},
      {-0.3104, 0.8127, -0.1945, 0.4521},
      {1.723, -2.419, 0.881, -0.337},
  };

  for (const auto& qa : kQuats) {
    // copy4 & normalize4
    mjtNum ref_n[4], mju_n[4], mji_n[4], cpy_u[4], cpy_i[4];
    std::memcpy(ref_n, qa, sizeof(ref_n));
    std::memcpy(mju_n, qa, sizeof(mju_n));
    std::memcpy(mji_n, qa, sizeof(mji_n));
    mju_copy4(cpy_u, qa);
    mji_copy4(cpy_i, qa);
    EXPECT_EQ(std::memcmp(cpy_u, qa, sizeof(cpy_u)), 0);
    EXPECT_EQ(std::memcmp(cpy_i, qa, sizeof(cpy_i)), 0);

    mjtNum nr = ScalarNormalize4(ref_n);
    mjtNum nu = mju_normalize4(mju_n);
    mjtNum ni = mji__normalize4(mji_n);
    EXPECT_EQ(std::memcmp(&nu, &nr, sizeof(mjtNum)), 0);
    EXPECT_EQ(std::memcmp(&ni, &nr, sizeof(mjtNum)), 0);
    EXPECT_EQ(std::memcmp(mju_n, ref_n, sizeof(ref_n)), 0);
    EXPECT_EQ(std::memcmp(mji_n, ref_n, sizeof(ref_n)), 0);

    // negQuat (out-of-place and in-place)
    mjtNum neg_r[4], neg_u[4], neg_i[4], neg_ip[4];
    ScalarNegQuat(neg_r, qa);
    mju_negQuat(neg_u, qa);
    mji_negQuat(neg_i, qa);
    std::memcpy(neg_ip, qa, sizeof(neg_ip));
    mju_negQuat(neg_ip, neg_ip);
    EXPECT_EQ(std::memcmp(neg_u, neg_r, sizeof(neg_r)), 0);
    EXPECT_EQ(std::memcmp(neg_i, neg_r, sizeof(neg_r)), 0);
    EXPECT_EQ(std::memcmp(neg_ip, neg_r, sizeof(neg_r)), 0);

    // quat2Mat (out-of-place and in-place)
    mjtNum mat_r[9], mat_u[9], mat_i[9], mat_ip[9] = {0};
    ScalarQuat2Mat(mat_r, qa);
    mju_quat2Mat(mat_u, qa);
    mji_quat2Mat(mat_i, qa);
    std::memcpy(mat_ip, qa, sizeof(qa));
    mju_quat2Mat(mat_ip, mat_ip);
    EXPECT_EQ(std::memcmp(mat_u, mat_r, sizeof(mat_r)), 0);
    EXPECT_EQ(std::memcmp(mat_i, mat_r, sizeof(mat_r)), 0);
    EXPECT_EQ(std::memcmp(mat_ip, mat_r, sizeof(mat_r)), 0);

    for (const auto& qb : kQuats) {
      // mulQuat (out-of-place and both in-place aliasings)
      mjtNum mq_r[4], mq_u[4], mq_i[4], mq_ipa[4], mq_ipb[4];
      ScalarMulQuat(mq_r, qa, qb);
      mju_mulQuat(mq_u, qa, qb);
      mji_mulQuat(mq_i, qa, qb);
      std::memcpy(mq_ipa, qa, sizeof(mq_ipa));
      mju_mulQuat(mq_ipa, mq_ipa, qb);
      std::memcpy(mq_ipb, qb, sizeof(mq_ipb));
      mju_mulQuat(mq_ipb, qa, mq_ipb);
      EXPECT_EQ(std::memcmp(mq_u, mq_r, sizeof(mq_r)), 0);
      EXPECT_EQ(std::memcmp(mq_i, mq_r, sizeof(mq_r)), 0);
      EXPECT_EQ(std::memcmp(mq_ipa, mq_r, sizeof(mq_r)), 0);
      EXPECT_EQ(std::memcmp(mq_ipb, mq_r, sizeof(mq_r)), 0);

      // mulQuatAxis (out-of-place and in-place)
      const mjtNum axis[3] = {qb[1], qb[2], qb[3]};
      mjtNum ma_r[4], ma_u[4], ma_i[4], ma_ip[4];
      ScalarMulQuatAxis(ma_r, qa, axis);
      mju_mulQuatAxis(ma_u, qa, axis);
      mji_mulQuatAxis(ma_i, qa, axis);
      std::memcpy(ma_ip, qa, sizeof(ma_ip));
      mju_mulQuatAxis(ma_ip, ma_ip, axis);
      EXPECT_EQ(std::memcmp(ma_u, ma_r, sizeof(ma_r)), 0);
      EXPECT_EQ(std::memcmp(ma_i, ma_r, sizeof(ma_r)), 0);
      EXPECT_EQ(std::memcmp(ma_ip, ma_r, sizeof(ma_r)), 0);
    }
  }
}

TEST_F(MulQuatTest, PoseOperationsInPlace) {
  const mjtNum pos1[3] = {10, 20, 30};
  const mjtNum pos2[3] = {1, 0, 0};
  const mjtNum s = mju_sqrt(0.5);
  const mjtNum quat1[4] = {s, 0, 0, s};
  const mjtNum quat2[4] = {s, 0, 0, s};

  mjtNum expected_pos[3], expected_quat[4];
  mju_mulPose(expected_pos, expected_quat, pos1, quat1, pos2, quat2);

  // mju_mulPose in-place with first pose
  mjtNum inplace_pos1[3] = {pos1[0], pos1[1], pos1[2]};
  mjtNum inplace_quat1[4] = {quat1[0], quat1[1], quat1[2], quat1[3]};
  mju_mulPose(inplace_pos1, inplace_quat1, inplace_pos1, inplace_quat1, pos2,
              quat2);
  EXPECT_THAT(inplace_pos1, Pointwise(MjNear(1e-12, 1e-5), expected_pos));
  EXPECT_THAT(inplace_quat1, Pointwise(MjNear(1e-12, 1e-5), expected_quat));

  // mju_mulPose in-place with second pose
  mjtNum inplace_pos2[3] = {pos2[0], pos2[1], pos2[2]};
  mjtNum inplace_quat2[4] = {quat2[0], quat2[1], quat2[2], quat2[3]};
  mju_mulPose(inplace_pos2, inplace_quat2, pos1, quat1, inplace_pos2,
              inplace_quat2);
  EXPECT_THAT(inplace_pos2, Pointwise(MjNear(1e-12, 1e-5), expected_pos));
  EXPECT_THAT(inplace_quat2, Pointwise(MjNear(1e-12, 1e-5), expected_quat));

  // mju_negPose in-place
  mjtNum expected_neg_pos[3], expected_neg_quat[4];
  mju_negPose(expected_neg_pos, expected_neg_quat, pos1, quat1);
  mjtNum inplace_neg_pos[3] = {pos1[0], pos1[1], pos1[2]};
  mjtNum inplace_neg_quat[4] = {quat1[0], quat1[1], quat1[2], quat1[3]};
  mju_negPose(inplace_neg_pos, inplace_neg_quat, inplace_neg_pos,
              inplace_neg_quat);
  EXPECT_THAT(inplace_neg_pos,
              Pointwise(MjNear(1e-12, 1e-5), expected_neg_pos));
  EXPECT_THAT(inplace_neg_quat,
              Pointwise(MjNear(1e-12, 1e-5), expected_neg_quat));

  // mju_trnVecPose in-place with pos and vec
  mjtNum expected_trn[3];
  mju_trnVecPose(expected_trn, pos1, quat1, pos2);
  mjtNum inplace_trn_pos[3] = {pos1[0], pos1[1], pos1[2]};
  mju_trnVecPose(inplace_trn_pos, inplace_trn_pos, quat1, pos2);
  EXPECT_THAT(inplace_trn_pos, Pointwise(MjNear(1e-12, 1e-5), expected_trn));
}

}  // namespace
}  // namespace mujoco
