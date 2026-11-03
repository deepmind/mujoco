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

#include "user/user_util.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "engine/engine_crossplatform.h"  // IWYU pragma: keep
#include "engine/engine_util_errmem.h"


// workaround with locale bug on some MacOS machines
#if defined(__APPLE__) && defined(__MACH__)
  #include <xlocale.h>
  #include <locale.h>

  #define strtof(X, Y) strtof_l((X), (Y), _c_locale)
  #define strtod(X, Y) strtod_l((X), (Y), _c_locale)
#endif

// check if numeric variable is defined
bool mjuu_defined(double num) {
  return !std::isnan(num);
}


// compute address of M[g1][g2] where M is triangular n-by-n
int mjuu_matadr(int g1, int g2, int n) {
  if (g1 < 0 || g2 < 0 || g1 >= n || g2 >= n) { return -1; }

  if (g1 > g2) {
    int tmp = g1;
    g1      = g2;
    g2      = tmp;
  }

  return g1 * n + g2;
}


// set 4D vector
void mjuu_setvec(double* dest, double x, double y, double z, double w) {
  dest[0] = x;
  dest[1] = y;
  dest[2] = z;
  dest[3] = w;
}
void mjuu_setvec(float* dest, double x, double y, double z, double w) {
  dest[0] = (float)x;
  dest[1] = (float)y;
  dest[2] = (float)z;
  dest[3] = (float)w;
}


// set 3D vector
void mjuu_setvec(double* dest, double x, double y, double z) {
  dest[0] = x;
  dest[1] = y;
  dest[2] = z;
}
void mjuu_setvec(float* dest, double x, double y, double z) {
  dest[0] = (float)x;
  dest[1] = (float)y;
  dest[2] = (float)z;
}


// set 2D vector
void mjuu_setvec(double* dest, double x, double y) {
  dest[0] = x;
  dest[1] = y;
}


// add to double array
void mjuu_addtovec(double* dest, const double* src, int n) {
  for (int i = 0; i < n; i++) { dest[i] += src[i]; }
}

// zero double array
void mjuu_zerovec(double* dest, int n) {
  for (int i = 0; i < n; i++) { dest[i] = 0; }
}

// zero float array
void mjuu_zerovec(float* dest, int n) {
  for (int i = 0; i < n; i++) { dest[i] = 0; }
}

// dot-product in 3D
double mjuu_dot3(const double* a, const double* b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}


// distance between 3D points
double mjuu_dist3(const double* a, const double* b) {
  return sqrt((a[0] - b[0]) * (a[0] - b[0]) +
              (a[1] - b[1]) * (a[1] - b[1]) +
              (a[2] - b[2]) * (a[2] - b[2]));
}


// L1 norm between vectors
double mjuu_L1(const double* a, const double* b, int n) {
  double res = 0;
  for (int i = 0; i < n; i++) { res += std::abs(a[i] - b[i]); }

  return res;
}


// normalize vector to unit length, return previous length
double mjuu_normvec(double* vec, const int n) {
  double nrm = 0;

  for (int i = 0; i < n; i++) { nrm += vec[i] * vec[i]; }
  if (nrm < mjEPS) { return 0; }

  nrm = sqrt(nrm);

  // don't normalize if nrm is within mjEPS of 1
  if (std::abs(nrm - 1) > mjEPS) {
    for (int i = 0; i < n; i++) { vec[i] /= nrm; }
  }

  return nrm;
}


// normalize float vector to unit length, return previous length
float mjuu_normvec(float* vec, const int n) {
  float nrm = 0;

  for (int i = 0; i < n; i++) { nrm += vec[i] * vec[i]; }
  if (nrm < mjEPS) { return 0; }

  nrm = sqrt(nrm);

  // don't normalize if nrm is within mjEPS of 1
  if (std::abs(nrm - 1) > mjEPS) {
    for (int i = 0; i < n; i++) { vec[i] /= nrm; }
  }

  return nrm;
}

// scale vector by scalar
void mjuu_scalevec(double* res, const double* vec, double s, int n) {
  for (int i = 0; i < n; i++) { res[i] = s * vec[i]; }
}

// convert quaternion to rotation matrix
void mjuu_quat2mat(double* res, const double* quat) {
  // identity quat: identity mat
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

  // regular processing
  double q00 = quat[0] * quat[0];
  double q01 = quat[0] * quat[1];
  double q02 = quat[0] * quat[2];
  double q03 = quat[0] * quat[3];
  double q11 = quat[1] * quat[1];
  double q12 = quat[1] * quat[2];
  double q13 = quat[1] * quat[3];
  double q22 = quat[2] * quat[2];
  double q23 = quat[2] * quat[3];
  double q33 = quat[3] * quat[3];

  res[0] = q00 + q11 - q22 - q33;
  res[4] = q00 - q11 + q22 - q33;
  res[8] = q00 - q11 - q22 + q33;

  res[1] = 2 * (q12 - q03);
  res[2] = 2 * (q13 + q02);
  res[3] = 2 * (q12 + q03);
  res[5] = 2 * (q23 - q01);
  res[6] = 2 * (q13 - q02);
  res[7] = 2 * (q23 + q01);
}


// multiply two unit quaternions
void mjuu_mulquat(double* res, const double* qa, const double* qb) {
  double tmp[4];
  tmp[0] = qa[0] * qb[0] - qa[1] * qb[1] - qa[2] * qb[2] - qa[3] * qb[3];
  tmp[1] = qa[0] * qb[1] + qa[1] * qb[0] + qa[2] * qb[3] - qa[3] * qb[2];
  tmp[2] = qa[0] * qb[2] - qa[1] * qb[3] + qa[2] * qb[0] + qa[3] * qb[1];
  tmp[3] = qa[0] * qb[3] + qa[1] * qb[2] - qa[2] * qb[1] + qa[3] * qb[0];
  mjuu_normvec(tmp, 4);
  mjuu_copyvec(res, tmp, 4);
}


// multiply matrix by vector, 3-by-3
void mjuu_mulvecmat(double* res, const double* vec, const double* mat) {
  double tmp[3] = {mat[0] * vec[0] + mat[1] * vec[1] + mat[2] * vec[2],
                   mat[3] * vec[0] + mat[4] * vec[1] + mat[5] * vec[2],
                   mat[6] * vec[0] + mat[7] * vec[1] + mat[8] * vec[2]};

  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
}


// multiply transposed matrix by vector, 3-by-3
void mjuu_mulvecmatT(double* res, const double* vec, const double* mat) {
  double tmp[3] = {mat[0] * vec[0] + mat[3] * vec[1] + mat[6] * vec[2],
                   mat[1] * vec[0] + mat[4] * vec[1] + mat[7] * vec[2],
                   mat[2] * vec[0] + mat[5] * vec[1] + mat[8] * vec[2]};

  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
}

// clang-format off
// compute res = R * M * R'
void mjuu_mulRMRT(double* res, const double* R, const double* M) {
  double tmp[9];

  // tmp = R*M
  tmp[0] = R[0]*M[0] + R[1]*M[3] + R[2]*M[6];
  tmp[1] = R[0]*M[1] + R[1]*M[4] + R[2]*M[7];
  tmp[2] = R[0]*M[2] + R[1]*M[5] + R[2]*M[8];
  tmp[3] = R[3]*M[0] + R[4]*M[3] + R[5]*M[6];
  tmp[4] = R[3]*M[1] + R[4]*M[4] + R[5]*M[7];
  tmp[5] = R[3]*M[2] + R[4]*M[5] + R[5]*M[8];
  tmp[6] = R[6]*M[0] + R[7]*M[3] + R[8]*M[6];
  tmp[7] = R[6]*M[1] + R[7]*M[4] + R[8]*M[7];
  tmp[8] = R[6]*M[2] + R[7]*M[5] + R[8]*M[8];

  // res = tmp*R'
  res[0] = tmp[0]*R[0] + tmp[1]*R[1] + tmp[2]*R[2];
  res[1] = tmp[0]*R[3] + tmp[1]*R[4] + tmp[2]*R[5];
  res[2] = tmp[0]*R[6] + tmp[1]*R[7] + tmp[2]*R[8];
  res[3] = tmp[3]*R[0] + tmp[4]*R[1] + tmp[5]*R[2];
  res[4] = tmp[3]*R[3] + tmp[4]*R[4] + tmp[5]*R[5];
  res[5] = tmp[3]*R[6] + tmp[4]*R[7] + tmp[5]*R[8];
  res[6] = tmp[6]*R[0] + tmp[7]*R[1] + tmp[8]*R[2];
  res[7] = tmp[6]*R[3] + tmp[7]*R[4] + tmp[8]*R[5];
  res[8] = tmp[6]*R[6] + tmp[7]*R[7] + tmp[8]*R[8];
}


// multiply two matrices, all 3-by-3
void mjuu_mulmat(double* res, const double* A, const double* B) {
  double tmp[9];
  tmp[0] = A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
  tmp[1] = A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
  tmp[2] = A[0]*B[2] + A[1]*B[5] + A[2]*B[8];

  tmp[3] = A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
  tmp[4] = A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
  tmp[5] = A[3]*B[2] + A[4]*B[5] + A[5]*B[8];

  tmp[6] = A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
  tmp[7] = A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
  tmp[8] = A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
  mjuu_copyvec(res, tmp, 9);
}
// clang-format on

// transpose 3-by-3 matrix
void mjuu_transposemat(double* res, const double* mat) {
  double tmp[9] = {mat[0], mat[3], mat[6], mat[1], mat[4], mat[7], mat[2], mat[5], mat[8]};
  mjuu_copyvec(res, tmp, 9);
}


// convert global to local axis relative to given frame
void mjuu_localaxis(double* al, const double* ag, const double* quat) {
  double mat[9];
  double qneg[4] = {quat[0], -quat[1], -quat[2], -quat[3]};
  mjuu_quat2mat(mat, qneg);
  mjuu_mulvecmat(al, ag, mat);
}


// convert global to local position relative to given frame
void mjuu_localpos(double* pl, const double* pg, const double* pos, const double* quat) {
  double a[3] = {pg[0] - pos[0], pg[1] - pos[1], pg[2] - pos[2]};
  mjuu_localaxis(pl, a, quat);
}


// compute quaternion rotation from parent to child
void mjuu_localquat(double* local, const double* child, const double* parent) {
  double pneg[4] = {parent[0], -parent[1], -parent[2], -parent[3]};
  mjuu_mulquat(local, pneg, child);
}


// clang-format off
// compute vector cross-product a = b x c
void mjuu_crossvec(double* a, const double* b, const double* c) {
  a[0] = b[1]*c[2] - b[2]*c[1];
  a[1] = b[2]*c[0] - b[0]*c[2];
  a[2] = b[0]*c[1] - b[1]*c[0];
}
// clang-format on


// compute normal vector to given triangle, return length
template <typename T>
double mjuu_makenormal(double* normal, const T a[3], const T b[3], const T c[3]) {
  double v1[3]     = {a[0], a[1], a[2]};
  double v2[3]     = {b[0], b[1], b[2]};
  double v3[3]     = {c[0], c[1], c[2]};
  double diffAB[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
  double diffAC[3] = {v3[0] - v1[0], v3[1] - v1[1], v3[2] - v1[2]};

  mjuu_crossvec(normal, diffAB, diffAC);
  double nrm = std::sqrt(mjuu_dot3(normal, normal));
  if (nrm < mjEPS) {
    normal[0] = 1;
    normal[1] = 0;
    normal[2] = 0;
    return 1;
  }
  normal[0] /= nrm;
  normal[1] /= nrm;
  normal[2] /= nrm;
  return nrm;
}

template double mjuu_makenormal(double*      normal,
                                const double a[3],
                                const double b[3],
                                const double c[3]);
template double mjuu_makenormal(double*     normal,
                                const float a[3],
                                const float b[3],
                                const float c[3]);

// compute quaternion as minimal rotation from [0;0;1] to vec
void mjuu_z2quat(double* quat, const double* vec) {
  double z[3] = {0, 0, 1};
  mjuu_crossvec(quat + 1, z, vec);
  double s = mjuu_normvec(quat + 1, 3);
  if (s < 1E-10) {
    quat[1] = 1;
    quat[2] = quat[3] = 0;
  }
  double ang  = atan2(s, vec[2]);
  quat[0]     = cos(ang / 2);
  quat[1]    *= sin(ang / 2);
  quat[2]    *= sin(ang / 2);
  quat[3]    *= sin(ang / 2);
}


// compute quaternion given frame (axes are in matrix columns)
void mjuu_frame2quat(double* quat, const double* x, const double* y, const double* z) {
  const double* mat[3] = {x, y, z};  // mat[c][r] indexing

  // q0 largest
  if (mat[0][0] + mat[1][1] + mat[2][2] > 0) {
    quat[0] = 0.5 * sqrt(1 + mat[0][0] + mat[1][1] + mat[2][2]);
    quat[1] = 0.25 * (mat[1][2] - mat[2][1]) / quat[0];
    quat[2] = 0.25 * (mat[2][0] - mat[0][2]) / quat[0];
    quat[3] = 0.25 * (mat[0][1] - mat[1][0]) / quat[0];
  }

  // q1 largest
  else if (mat[0][0] > mat[1][1] && mat[0][0] > mat[2][2]) {
    quat[1] = 0.5 * sqrt(1 + mat[0][0] - mat[1][1] - mat[2][2]);
    quat[0] = 0.25 * (mat[1][2] - mat[2][1]) / quat[1];
    quat[2] = 0.25 * (mat[1][0] + mat[0][1]) / quat[1];
    quat[3] = 0.25 * (mat[2][0] + mat[0][2]) / quat[1];
  }

  // q2 largest
  else if (mat[1][1] > mat[2][2]) {
    quat[2] = 0.5 * sqrt(1 - mat[0][0] + mat[1][1] - mat[2][2]);
    quat[0] = 0.25 * (mat[2][0] - mat[0][2]) / quat[2];
    quat[1] = 0.25 * (mat[1][0] + mat[0][1]) / quat[2];
    quat[3] = 0.25 * (mat[2][1] + mat[1][2]) / quat[2];
  }

  // q3 largest
  else {
    quat[3] = 0.5 * sqrt(1 - mat[0][0] - mat[1][1] + mat[2][2]);
    quat[0] = 0.25 * (mat[0][1] - mat[1][0]) / quat[3];
    quat[1] = 0.25 * (mat[2][0] + mat[0][2]) / quat[3];
    quat[2] = 0.25 * (mat[2][1] + mat[1][2]) / quat[3];
  }

  mjuu_normvec(quat, 4);
}


// invert frame transformation
void mjuu_frameinvert(double       newpos[3],
                      double       newquat[4],
                      const double oldpos[3],
                      const double oldquat[4]) {
  // position
  mjuu_localaxis(newpos, oldpos, oldquat);
  newpos[0] = -newpos[0];
  newpos[1] = -newpos[1];
  newpos[2] = -newpos[2];

  // orientation
  newquat[0] = oldquat[0];
  newquat[1] = -oldquat[1];
  newquat[2] = -oldquat[2];
  newquat[3] = -oldquat[3];
}


// accumulate frame transformations (forward kinematics)
void mjuu_frameaccum(double       pos[3],
                     double       quat[4],
                     const double childpos[3],
                     const double childquat[4]) {
  double mat[9], vec[3], qtmp[4];
  mjuu_quat2mat(mat, quat);
  mjuu_mulvecmat(vec, childpos, mat);
  pos[0] += vec[0];
  pos[1] += vec[1];
  pos[2] += vec[2];
  mjuu_mulquat(qtmp, quat, childquat);
  mjuu_copyvec(quat, qtmp, 4);
}


// accumulate frame transformation in second frame
void mjuu_frameaccumChild(const double pos[3],
                          const double quat[4],
                          double       childpos[3],
                          double       childquat[4]) {
  double p[] = {pos[0], pos[1], pos[2]};
  double q[] = {quat[0], quat[1], quat[2], quat[3]};
  mjuu_frameaccum(p, q, childpos, childquat);
  mjuu_copyvec(childpos, p, 3);
  mjuu_copyvec(childquat, q, 4);
}


// invert frame accumulation
void mjuu_frameaccuminv(double       pos[3],
                        double       quat[4],
                        const double childpos[3],
                        const double childquat[4]) {
  double mat[9], vec[3], qtmp[4];
  double qneg[4] = {childquat[0], -childquat[1], -childquat[2], -childquat[3]};
  mjuu_mulquat(qtmp, quat, qneg);
  mjuu_copyvec(quat, qtmp, 4);
  mjuu_quat2mat(mat, quat);
  mjuu_mulvecmat(vec, childpos, mat);
  pos[0] -= vec[0];
  pos[1] -= vec[1];
  pos[2] -= vec[2];
}


// clang-format off
// convert local_inertia[3] to global_inertia[6]
void mjuu_globalinertia(double* global, const double* local, const double* quat) {
  double mat[9];
  mjuu_quat2mat(mat, quat);

  double tmp[9] = {
    mat[0]*local[0], mat[3]*local[0], mat[6]*local[0],
    mat[1]*local[1], mat[4]*local[1], mat[7]*local[1],
    mat[2]*local[2], mat[5]*local[2], mat[8]*local[2]
  };

  global[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6];
  global[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7];
  global[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8];
  global[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7];
  global[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8];
  global[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8];
}
// clang-format on


// compute off-center correction to inertia matrix
//  mass * [y^2+z^2, -x*y, -x*z;  -x*y, x^2+z^2, -y*z;  -x*z, -y*z, x^2+y^2]
void mjuu_offcenter(double* res, const double mass, const double* vec) {
  res[0] = mass * (vec[1] * vec[1] + vec[2] * vec[2]);
  res[1] = mass * (vec[0] * vec[0] + vec[2] * vec[2]);
  res[2] = mass * (vec[0] * vec[0] + vec[1] * vec[1]);
  res[3] = -mass * vec[0] * vec[1];
  res[4] = -mass * vec[0] * vec[2];
  res[5] = -mass * vec[1] * vec[2];
}


// compute viscosity coefficients from mass and inertia
void mjuu_visccoef(double* visccoef, double mass, const double* inertia, double scl) {
  // compute equivalent box
  double ebox[3];
  ebox[0] = sqrt(std::max(mjEPS, (inertia[1] + inertia[2] - inertia[0])) / mass * 6.0);
  ebox[1] = sqrt(std::max(mjEPS, (inertia[0] + inertia[2] - inertia[1])) / mass * 6.0);
  ebox[2] = sqrt(std::max(mjEPS, (inertia[0] + inertia[1] - inertia[2])) / mass * 6.0);

  // apply formula for box (or rather cross) viscosity

  //  torque components
  visccoef[0] =
      scl * 4.0 / 3.0 * ebox[0] * (ebox[1] * ebox[1] * ebox[1] + ebox[2] * ebox[2] * ebox[2]);
  visccoef[1] =
      scl * 4.0 / 3.0 * ebox[1] * (ebox[0] * ebox[0] * ebox[0] + ebox[2] * ebox[2] * ebox[2]);
  visccoef[2] =
      scl * 4.0 / 3.0 * ebox[2] * (ebox[0] * ebox[0] * ebox[0] + ebox[1] * ebox[1] * ebox[1]);

  //  force components
  visccoef[3] = scl * 4 * ebox[1] * ebox[2];
  visccoef[4] = scl * 4 * ebox[0] * ebox[2];
  visccoef[5] = scl * 4 * ebox[0] * ebox[1];
}


// convert axisAngle to quaternion
static void mjuu_axisAngle2Quat(double res[4], const double axis[3], double angle) {
  // zero angle: identity quat
  if (angle == 0) {
    res[0] = 1;
    res[1] = 0;
    res[2] = 0;
    res[3] = 0;
  }

  // regular processing
  else {
    double s = sin(angle * 0.5);
    res[0]   = cos(angle * 0.5);
    res[1]   = axis[0] * s;
    res[2]   = axis[1] * s;
    res[3]   = axis[2] * s;
  }
}

// rotate vector by quaternion
void mjuu_rotVecQuat(double res[3], const double vec[3], const double quat[4]) {
  // zero vec: zero res
  if (vec[0] == 0 && vec[1] == 0 && vec[2] == 0) {
    res[0] = res[1] = res[2] = 0;
  }

  // null quat: copy vec
  else if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    mjuu_copyvec(res, vec, 3);
  }

  // regular processing
  else {
    // tmp = q_w * v + cross(q_xyz, v)
    double tmp[3] = {quat[0] * vec[0] + quat[2] * vec[2] - quat[3] * vec[1],
                     quat[0] * vec[1] + quat[3] * vec[0] - quat[1] * vec[2],
                     quat[0] * vec[2] + quat[1] * vec[1] - quat[2] * vec[0]};

    // res = v + 2 * cross(q_xyz, t)
    res[0] = vec[0] + 2 * (quat[2] * tmp[2] - quat[3] * tmp[1]);
    res[1] = vec[1] + 2 * (quat[3] * tmp[0] - quat[1] * tmp[2]);
    res[2] = vec[2] + 2 * (quat[1] * tmp[1] - quat[2] * tmp[0]);
  }
}

// update moving frame along a curve or initialize it, returns edge length
//   inputs:
//     normal    - normal vector computed by a previous call to the function
//     edge      - edge vector (non-unit tangent vector)
//     tprv      - unit tangent vector of previous body
//     tnxt      - unit tangent vector of next body
//     first     - 1 if the frame requires initialization
//   outputs:
//     quat      - frame orientation
//     normal    - unit normal vector
double mjuu_updateFrame(double       quat[4],
                        double       normal[3],
                        const double edge[3],
                        const double tprv[3],
                        const double tnxt[3],
                        int          first) {
  double tangent[3], binormal[3];

  // normalize tangent
  mjuu_copyvec(tangent, edge, 3);
  mjuu_normvec(tangent, 3);

  // compute moving frame
  if (first) {
    // use the first vertex binormal for the first edge
    mjuu_crossvec(binormal, tangent, tnxt);
    mjuu_normvec(binormal, 3);

    // compute edge normal given tangent and binormal
    mjuu_crossvec(normal, binormal, tangent);
    mjuu_normvec(normal, 3);
  } else {
    double darboux[4];

    // rotate edge normal about the vertex binormal
    mjuu_crossvec(binormal, tprv, tangent);
    double angle = atan2(mjuu_normvec(binormal, 3), mjuu_dot3(tprv, tangent));
    mjuu_axisAngle2Quat(darboux, binormal, angle);
    mjuu_rotVecQuat(normal, normal, darboux);
    mjuu_normvec(normal, 3);

    // compute edge binormal given tangent and normal
    mjuu_crossvec(binormal, tangent, normal);
    mjuu_normvec(binormal, 3);
  }
  // global orientation of the frame
  mjuu_frame2quat(quat, tangent, normal, binormal);

  // return edge length
  return sqrt(mjuu_dot3(edge, edge));
}


// eigenvalue decomposition of symmetric 3x3 matrix
static const double kEigTOL = 4E-15;  // off-diagonal tolerance, relative to the largest element
static const double kEigEPS = 1E-12;  // eigenvalue swap threshold, relative to the largest element

int mjuu_eig3(
    double eigval[3], double eigvec[9], double quat[4], const double mat[9], double reltol) {
  double D[9], tmp[9], tmp2[9];
  double tau, t;
  int    iter, rk, ck, rotk;

  // off-diagonal tolerance: no smaller than roundoff level of D, about 16 epsilons
  double scale = 0;
  for (int i = 0; i < 9; i++) { scale = std::max(scale, std::abs(mat[i])); }
  double tol = scale * std::max(reltol, kEigTOL);

  // initialize with unit quaternion
  quat[0] = 1;
  quat[1] = quat[2] = quat[3] = 0;

  // Jacobi iteration
  for (iter = 0; iter < 500; iter++) {
    // make quaternion matrix eigvec, compute D = eigvec'*mat*eigvec
    mjuu_quat2mat(eigvec, quat);
    mjuu_transposemat(tmp2, eigvec);
    mjuu_mulmat(tmp, tmp2, mat);
    mjuu_mulmat(D, tmp, eigvec);

    // assign eigenvalues
    eigval[0] = D[0];
    eigval[1] = D[4];
    eigval[2] = D[8];

    // find max off-diagonal element, set indices
    if (std::abs(D[1]) > std::abs(D[2]) && std::abs(D[1]) > std::abs(D[5])) {
      rk   = 0;  // row
      ck   = 1;  // column
      rotk = 2;  // rotation axis
    } else if (std::abs(D[2]) > std::abs(D[5])) {
      rk   = 0;
      ck   = 2;
      rotk = 1;
    } else {
      rk   = 1;
      ck   = 2;
      rotk = 0;
    }

    // terminate if max off-diagonal element too small
    if (std::abs(D[3 * rk + ck]) <= tol) { break; }

    // 2x2 symmetric Schur decomposition: t = tan(angle)
    tau = (D[4 * ck] - D[4 * rk]) / (2 * D[3 * rk + ck]);
    if (tau >= 0) {
      t = 1.0 / (tau + sqrt(1 + tau * tau));
    } else {
      t = -1.0 / (-tau + sqrt(1 + tau * tau));
    }

    // express rotation as quaternion, using h = tan(angle/2): accurate for small angles
    double h      = t / (1 + sqrt(1 + t * t));
    tmp[0]        = 1 / sqrt(1 + h * h);
    tmp[1]        = 0;
    tmp[2]        = 0;
    tmp[3]        = 0;
    tmp[rotk + 1] = (rotk == 1 ? h : -h) * tmp[0];

    // accumulate quaternion rotation
    mjuu_mulquat(quat, quat, tmp);
    mjuu_normvec(quat, 4);
  }

  // sort eigenvalues in decreasing order (bubblesort: 0, 1, 0)
  double eps = scale * std::max(reltol, kEigEPS);
  for (int j = 0; j < 3; j++) {
    int j1 = j % 2;  // lead index

    // only swap if the eigenvalues are different
    if (eigval[j1] + eps < eigval[j1 + 1]) {
      // swap eigenvalues
      t              = eigval[j1];
      eigval[j1]     = eigval[j1 + 1];
      eigval[j1 + 1] = t;

      // rotate quaternion
      tmp[0] = 0.707106781186548;  // cos(pi/4) = sin(pi/4)
      tmp[1] = tmp[2] = tmp[3] = 0;
      tmp[(j1 + 2) % 3 + 1]    = tmp[0];
      mjuu_mulquat(quat, quat, tmp);
      mjuu_normvec(quat, 4);
    }
  }

  // recompute eigvec
  mjuu_quat2mat(eigvec, quat);

  return iter;
}


// Jacobi eigenvalue decomposition of symmetric n×n matrix.
// On output, eigenvalues are in eigval and eigenvectors are columns of eigvec.
// Both arrays must be pre-allocated: eigval[n], eigvec[n*n].
// The input matrix mat is destroyed.
int mjuu_eigendecompose(double* mat, double* eigval, double* eigvec, int n) {
  // initialize eigvec to identity
  std::fill(eigvec, eigvec + n * n, 0.0);
  for (int i = 0; i < n; i++) { eigvec[i * n + i] = 1.0; }

  const int    max_sweeps = 200;
  const double tol        = 1e-12;

  int sweep;
  for (sweep = 0; sweep < max_sweeps; sweep++) {
    // check convergence: sum of squared off-diagonal elements
    double off_diag = 0;
    for (int i = 0; i < n; i++) {
      for (int j = i + 1; j < n; j++) { off_diag += mat[i * n + j] * mat[i * n + j]; }
    }
    if (off_diag < tol * tol) break;

    // sweep over all off-diagonal pairs
    for (int p = 0; p < n; p++) {
      for (int q = p + 1; q < n; q++) {
        double apq = mat[p * n + q];
        if (std::abs(apq) < tol * 1e-3) continue;

        // compute rotation angle
        double app = mat[p * n + p];
        double aqq = mat[q * n + q];
        double tau = (aqq - app) / (2.0 * apq);
        double t   = (tau >= 0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        double c   = 1.0 / std::sqrt(1.0 + t * t);
        double s   = t * c;

        // update matrix (Jacobi rotation)
        mat[p * n + p] -= t * apq;
        mat[q * n + q] += t * apq;
        mat[p * n + q]  = 0;
        mat[q * n + p]  = 0;

        for (int r = 0; r < n; r++) {
          if (r == p || r == q) continue;
          double mrp     = mat[r * n + p];
          double mrq     = mat[r * n + q];
          mat[r * n + p] = mat[p * n + r] = c * mrp - s * mrq;
          mat[r * n + q] = mat[q * n + r] = s * mrp + c * mrq;
        }

        // accumulate eigenvectors
        for (int r = 0; r < n; r++) {
          double vrp        = eigvec[r * n + p];
          double vrq        = eigvec[r * n + q];
          eigvec[r * n + p] = c * vrp - s * vrq;
          eigvec[r * n + q] = s * vrp + c * vrq;
        }
      }
    }
  }

  // extract eigenvalues from diagonal
  for (int i = 0; i < n; i++) { eigval[i] = mat[i * n + i]; }

  return sweep;
}


// transform vector by pose
void mjuu_trnVecPose(double       res[3],
                     const double pos[3],
                     const double quat[4],
                     const double vec[3]) {
  // res = quat*vec + pos
  mjuu_rotVecQuat(res, vec, quat);
  res[0] += pos[0];
  res[1] += pos[1];
  res[2] += pos[2];
}

// strip directory from filename
std::string mjuu_strippath(std::string filename) {
  // find last pathsymbol
  size_t start = filename.find_last_of("/\\");

  // no path found: return original
  if (start == std::string::npos) {
    return filename;
  }

  // return name without path
  else {
    return filename.substr(start + 1, filename.size() - start - 1);
  }
}


// compute frame quat and diagonal inertia from full inertia matrix, return error if any
const char* mjuu_fullInertia(double quat[4], double inertia[3], const double fullinertia[6]) {
  if (!mjuu_defined(fullinertia[0])) { return nullptr; }

  double eigval[3], eigvec[9], quattmp[4];
  double full[9] = {fullinertia[0],
                    fullinertia[3],
                    fullinertia[4],
                    fullinertia[3],
                    fullinertia[1],
                    fullinertia[5],
                    fullinertia[4],
                    fullinertia[5],
                    fullinertia[2]};

  mjuu_eig3(eigval, eigvec, quattmp, full);

  // copy, also on error so that callers can bound the inertia
  if (quat) { mjuu_copyvec(quat, quattmp, 4); }

  if (inertia) { mjuu_copyvec(inertia, eigval, 3); }

  // check minimal eigenvalue
  if (eigval[2] < mjEPS) { return "inertia must have positive eigenvalues"; }

  return nullptr;
}


// strip extension
std::string mjuu_stripext(std::string filename) {
  // find last dot
  size_t end = filename.find_last_of('.');

  // no path found: return original
  if (end == std::string::npos) { return filename; }

  // return name without extension
  return filename.substr(0, end);
}

std::string mjuu_getext(std::string_view filename) {
  size_t dot = filename.find_last_of('.');

  if (dot == std::string::npos) { return ""; }
  return std::string(filename.substr(dot, filename.size() - dot));
}


// is directory path absolute
bool mjuu_isabspath(std::string path) {
  // empty: not absolute
  if (path.empty()) { return false; }

  // path is scheme:filename which we consider an absolute path
  // e.g. file URI's are always absolute paths
  if (mjp_findProvider(path.c_str()) != nullptr) { return true; }

  // check first char
  const char* str = path.c_str();
  if (str[0] == '\\' || str[0] == '/') { return true; }

  // find ":/" or ":\"
  if (path.find(":/") != std::string::npos || path.find(":\\") != std::string::npos) {  // NOLINT
    return true;
  }

  return false;
}


// assemble two file paths
std::string mjuu_combinePaths(const std::string& path1, const std::string& path2) {
  // path2 has absolute path
  if (mjuu_isabspath(path2)) { return path2; }

  std::size_t n = path1.size();
  if (n > 0 && path1[n - 1] != '\\' && path1[n - 1] != '/') { return path1 + "/" + path2; }
  return path1 + path2;
}


// assemble three file paths
std::string mjuu_combinePaths(const std::string& path1,
                              const std::string& path2,
                              const std::string& path3) {
  return mjuu_combinePaths(path1, mjuu_combinePaths(path2, path3));
}


// return true if the text is in a valid content type format:
// {type}/{subtype}[;{parameter}={value}]
static bool mjuu_isValidContentType(std::string_view text) {
  // find a forward slash that's not the last character
  size_t n = text.find('/');
  if (n == std::string::npos || n == text.size() - 1) { return false; }

  size_t m = text.find(';');
  if (m == std::string::npos) { return true; }

  if (m + 1 <= n) { return false; }

  // just check if there's an equal sign; this isn't robust enough for general
  // validation, but works for our scope, hence this is a private helper
  // function
  size_t s = text.find('=');
  if (s == std::string::npos || s + 1 <= m) { return false; }

  return true;
}


// return type from content_type format {type}/{subtype}[;{parameter}={value}]
// return empty string on invalid format
std::optional<std::string_view> mjuu_parseContentTypeAttrType(std::string_view text) {
  if (!mjuu_isValidContentType(text)) { return std::nullopt; }

  return {text.substr(0, text.find('/'))};
}


// return subtype from content_type format {type}/{subtype}[;{parameter}={value}]
// return empty string on invalid format
std::optional<std::string_view> mjuu_parseContentTypeAttrSubtype(std::string_view text) {
  if (!mjuu_isValidContentType(text)) { return std::nullopt; }

  size_t n = text.find('/');
  size_t m = text.find(';', n + 1);
  if (m == std::string::npos) { return {text.substr(n + 1)}; }

  return {text.substr(n + 1, m - n - 1)};
}


// convert filename extension to content type; return empty string if not found
std::string mjuu_extToContentType(std::string_view filename) {
  std::string ext = mjuu_getext(filename);

  if (!strcasecmp(ext.c_str(), ".stl")) {
    return "model/stl";
  } else if (!strcasecmp(ext.c_str(), ".obj")) {
    return "model/obj";
  } else if (!strcasecmp(ext.c_str(), ".ply")) {
    return "model/ply";
  } else if (!strcasecmp(ext.c_str(), ".msh")) {
    return "model/vnd.mujoco.msh";
  } else if (!strcasecmp(ext.c_str(), ".png")) {
    return "image/png";
  } else if (!strcasecmp(ext.c_str(), ".ktx")) {
    return "image/ktx";
  } else {
    return "";
  }
}

// get the length of the dirname portion of a given path
int mjuu_dirnamelen(const char* path) {
  if (!path) { return 0; }

  int pos = -1;
  for (int i = 0; path[i]; ++i) {
    if (path[i] == '/' || path[i] == '\\') { pos = i; }
  }

  return pos + 1;
}

namespace mujoco::user {

std::string FilePath::Combine(const std::string& s1, const std::string& s2) {
  // str2 has absolute path
  if (!AbsPrefix(s2).empty()) { return s2; }

  std::size_t n = s1.size();
  if (n > 0 && s1[n - 1] != '\\' && s1[n - 1] != '/') { return s1 + "/" + s2; }
  return s1 + s2;
}

std::string FilePath::PathReduce(const std::string& str) {
  std::vector<std::string> dirs;
  std::string              abs_prefix = AbsPrefix(str);

  int j = abs_prefix.size();

  for (int i = j; i < str.size(); ++i) {
    if (IsSeparator(str[i])) {
      std::string temp = str.substr(j, i - j);
      j                = i + 1;
      if (temp == ".." && !dirs.empty() && dirs.back() != "..") {
        dirs.pop_back();
      } else if (temp != "." && !temp.empty()) {
        dirs.push_back(std::move(temp));
      }
    }
  }

  // push the rest of the string
  dirs.push_back(str.substr(j, str.size() - j));

  // join the path
  std::stringstream path;
  auto              it = dirs.begin();
  path << abs_prefix << *it++;
  for (; it != dirs.end(); ++it) { path << "/" << *it; }
  return path.str();
}

FilePath FilePath::operator+(const FilePath& path) const {
  return FilePath(path_, path.path_);
}

std::string FilePath::Ext() const {
  std::size_t n = path_.find_last_of('.');

  if (n == std::string::npos) { return ""; }
  return path_.substr(n, path_.size() - n);
}

FilePath FilePath::StripExt() const {
  size_t n = path_.find_last_of('.');

  // no extension
  if (n == std::string::npos) { return FilePathFast(path_); }

  // return path without extension
  return FilePathFast(path_.substr(0, n));
}

// is directory absolute path
std::string FilePath::AbsPrefix(const std::string& str) {
  // empty: not absolute
  if (str.empty()) { return ""; }

  // path is scheme:filename which we consider an absolute path
  // e.g. file URI's are always absolute paths
  const mjpResourceProvider* provider = mjp_findProvider(str.c_str());
  if (provider != nullptr) {
    std::size_t n = std::strlen(provider->prefix);
    std::size_t j = n + 1;
    while (j < str.size() && IsSeparator(str[j])) { ++j; }
    return str.substr(0, j);
  }

  // check first char
  if (str[0] == '\\' || str[0] == '/') { return str.substr(0, 1); }

  // find ":/" or ":\"
  std::size_t pos = str.find(":/");
  if (pos != std::string::npos) { return str.substr(0, pos + 2); }

  pos = str.find(":\\");
  if (pos != std::string::npos) { return str.substr(0, pos + 2); }

  return "";
}

FilePath FilePath::StripPath() const {
  // find last path symbol
  std::size_t n = path_.find_last_of("/\\");

  // no path
  if (n == std::string::npos) { return FilePathFast(path_); }

  return FilePathFast(path_.substr(n + 1, path_.size() - (n + 1)));
}

std::string FilePath::StrLower() const {
  std::string str = path_;
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return str;
}

// read file into memory buffer
std::vector<uint8_t> FileToMemory(const char* filename) {
  FILE* fp = mju_fopen(filename, "rb");
  if (!fp) { return {}; }

  // find size
  if (fseek(fp, 0, SEEK_END) != 0) {
    fclose(fp);
    mju_warning("Failed to calculate size for '%s'", filename);
    return {};
  }

  // ensure file size fits in int
  long long_filesize = ftell(fp);  // NOLINT(runtime/int)
  if (long_filesize > INT_MAX) {
    fclose(fp);
    mju_warning("File size over 2GB is not supported. File: '%s'", filename);
    return {};
  } else if (long_filesize < 0) {
    fclose(fp);
    mju_warning("Failed to calculate size for '%s'", filename);
    return {};
  }

  std::vector<uint8_t> buffer;
  try {
    buffer.resize(long_filesize);
  } catch (...) {
    fclose(fp);
    return {};
  }

  // go back to start of file
  if (fseek(fp, 0, SEEK_SET) != 0) {
    fclose(fp);
    mju_warning("Read error while reading '%s'", filename);
    return {};
  }

  // allocate and read
  std::size_t bytes_read = fread(buffer.data(), 1, buffer.size(), fp);

  // check that read data matches file size
  if (bytes_read != buffer.size()) {  // SHOULD NOT OCCUR
    if (ferror(fp)) {
      fclose(fp);
      mju_warning("Read error while reading '%s'", filename);
      return {};
    } else if (feof(fp)) {
      buffer.resize(bytes_read);
    }
  }

  // close file, return contents
  fclose(fp);
  return buffer;
}

// convert vector to string separating elements by whitespace
template <typename T>
std::string VectorToString(const std::vector<T>& v) {
  std::stringstream ss;

  for (const T& t : v) { ss << t << " "; }

  std::string s = ss.str();
  if (!s.empty()) s.pop_back();  // remove trailing space
  return s;
}

template std::string VectorToString(const std::vector<int>& v);
template std::string VectorToString(const std::vector<float>& v);
template std::string VectorToString(const std::vector<double>& v);
template std::string VectorToString(const std::vector<std::string>& v);

namespace {

template <typename T>
T StrToNum(char* str, char** c);

template <>
int StrToNum(char* str, char** c) {
  long n = std::strtol(str, c, 10);
  if (n < INT_MIN || n > INT_MAX) errno = ERANGE;
  return n;
}

template <>
float StrToNum(char* str, char** c) {
  float f = strtof(str, c);
  if (errno == ERANGE && std::isfinite(f)) errno = 0;
  if (std::isnan(f)) errno = EDOM;
  return f;
}

template <>
double StrToNum(char* str, char** c) {
  double d = strtod(str, c);
  if (errno == ERANGE && std::isfinite(d)) errno = 0;
  if (std::isnan(d)) errno = EDOM;
  return d;
}

template <>
unsigned char StrToNum(char* str, char** c) {
  long n = std::strtol(str, c, 10);
  if (n < 0 || n > UCHAR_MAX) errno = ERANGE;
  return n;
}

inline bool IsNullOrSpace(char* c) {
  return std::isspace(static_cast<unsigned char>(*c)) || *c == '\0';
}

inline char* SkipSpace(char* c) {
  for (; *c != '\0'; c++) {
    if (!IsNullOrSpace(c)) { break; }
  }
  return c;
}
}  // namespace

template <typename T>
std::vector<T> StringToVector(char* cs) {
  std::vector<T> v;
  char*          ch      = cs;
  bool           has_nan = false;

  errno = 0;
  // reserve worst case
  v.reserve((std::strlen(cs) >> 1) + 1);

  for (;;) {
    cs = SkipSpace(ch);      // skip leading spaces
    if (*cs == '\0') break;  // end of string
    errno = 0;
    T num = StrToNum<T>(cs, &ch);            // parse number
    if (!IsNullOrSpace(ch)) errno = EINVAL;  // invalid separator
    if (cs == ch) errno = EINVAL;            // failed to parse number
    if (errno == EDOM) {
      has_nan = true;
      errno   = 0;
    }
    if (errno) break;  // NaNs are quietly ignored
    v.push_back(num);
  }
  if (!errno && has_nan) errno = EDOM;

  v.shrink_to_fit();
  return v;
}

template <>
std::vector<std::string> StringToVector(char* cs) {
  return StringToVector<std::string>(std::string(cs));
}

template <>
std::vector<std::string> StringToVector(const std::string& s) {
  std::vector<std::string> v;
  std::stringstream        ss(s);
  std::string              word;
  while (ss >> word) { v.push_back(word); }
  return v;
}

template std::vector<int>           StringToVector(char* cs);
template std::vector<float>         StringToVector(char* cs);
template std::vector<double>        StringToVector(char* cs);
template std::vector<unsigned char> StringToVector(char* cs);


template <typename T>
std::vector<T> StringToVector(const std::string& s) {
  return StringToVector<T>(const_cast<char*>(s.c_str()));
}
template std::vector<int>           StringToVector(const std::string& s);
template std::vector<float>         StringToVector(const std::string& s);
template std::vector<double>        StringToVector(const std::string& s);
template std::vector<unsigned char> StringToVector(const std::string& s);

}  // namespace mujoco::user
