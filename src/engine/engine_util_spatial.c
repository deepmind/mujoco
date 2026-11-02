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

#include "engine/engine_util_spatial.h"

#include <math.h>

#include "engine/engine_inline.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"
#include "engine/engine_util_spatial_avx.h"  // IWYU pragma: keep


//------------------------------ quaternion operations ---------------------------------------------

// rotate vector by quaternion
void mju_rotVecQuat(mjtNum res[3], const mjtNum vec[3], const mjtNum quat[4]) {
  // zero vec: zero res
  if (vec[0] == 0 && vec[1] == 0 && vec[2] == 0) {
    mju_zero3(res);
  }

  // null quat: copy vec
  else if (quat[0] == 1 && quat[1] == 0 && quat[2] == 0 && quat[3] == 0) {
    mji_copy3(res, vec);
  }

  // regular processing
  else {
    // tmp = q_w * v + cross(q_xyz, v)
    mjtNum tmp[3] = {
      quat[0]*vec[0] + quat[2]*vec[2] - quat[3]*vec[1],
      quat[0]*vec[1] + quat[3]*vec[0] - quat[1]*vec[2],
      quat[0]*vec[2] + quat[1]*vec[1] - quat[2]*vec[0]
    };

    // res = v + 2 * cross(q_xyz, t)
    res[0] = vec[0] + 2 * (quat[2]*tmp[2] - quat[3]*tmp[1]);
    res[1] = vec[1] + 2 * (quat[3]*tmp[0] - quat[1]*tmp[2]);
    res[2] = vec[2] + 2 * (quat[1]*tmp[1] - quat[2]*tmp[0]);
  }
}


// negate quaternion
void mju_negQuat(mjtNum res[4], const mjtNum quat[4]) {
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
void mju_mulQuat(mjtNum res[4], const mjtNum qa[4], const mjtNum qb[4]) {
#ifdef mjUSEAVX
  mju_mulQuat_avx(res, qa, qb);
#else
  mjtNum tmp[4] = {
    (qa[0]*qb[0] - qa[1]*qb[1]) + (-qa[2]*qb[2] - qa[3]*qb[3]),
    (qa[0]*qb[1] + qa[1]*qb[0]) + ( qa[2]*qb[3] - qa[3]*qb[2]),
    (qa[0]*qb[2] - qa[1]*qb[3]) + ( qa[2]*qb[0] + qa[3]*qb[1]),
    (qa[0]*qb[3] + qa[1]*qb[2]) + (-qa[2]*qb[1] + qa[3]*qb[0])
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
  res[3] = tmp[3];
#endif
}


// multiply quaternion and axis
void mju_mulQuatAxis(mjtNum res[4], const mjtNum quat[4], const mjtNum axis[3]) {
#ifdef mjUSEAVX
  mju_mulQuatAxis_avx(res, quat, axis);
#else
  mjtNum tmp[4] = {
    -quat[1]*axis[0] - quat[2]*axis[1] - quat[3]*axis[2],
     quat[0]*axis[0] - quat[3]*axis[1] + quat[2]*axis[2],
     quat[3]*axis[0] + quat[0]*axis[1] - quat[1]*axis[2],
    -quat[2]*axis[0] + quat[1]*axis[1] + quat[0]*axis[2]
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
  res[3] = tmp[3];
#endif
}


// convert axisAngle to quaternion
void mju_axisAngle2Quat(mjtNum res[4], const mjtNum axis[3], mjtNum angle) {
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
void mju_quat2Vel(mjtNum res[3], const mjtNum quat[4], mjtNum dt) {
  mjtNum axis[3] = {quat[1], quat[2], quat[3]};
  mjtNum sin_a_2 = mju_normalize3(axis);
  mjtNum speed = 2 * mju_atan2(sin_a_2, quat[0]);

  // when axis-angle is larger than pi, rotation is in the opposite direction
  if (speed > mjPI) {
    speed -= 2*mjPI;
  }
  speed /= dt;

  mji_scl3(res, axis, speed);
}


// Subtract quaternions, express as 3D velocity: qb*quat(res) = qa.
void mju_subQuat(mjtNum res[3], const mjtNum qa[4], const mjtNum qb[4]) {
  // qdif = neg(qb)*qa
  mjtNum qneg[4], qdif[4];
  mji_negQuat(qneg, qb);
  mji_mulQuat(qdif, qneg, qa);

  // convert to 3D velocity
  mji_quat2Vel(res, qdif, 1);
}


// convert quaternion to 3D rotation matrix
void mju_quat2Mat(mjtNum res[9], const mjtNum quat[4]) {
  if (res == quat) {
    mjtNum tmp[4];
    mji_copy4(tmp, quat);
    mji_quat2Mat(res, tmp);
  } else {
    mji_quat2Mat(res, quat);
  }
}


// convert 3D rotation matrix to quaternion
void mju_mat2Quat(mjtNum quat[4], const mjtNum mat[9]) {
  // q0 largest
  if (mat[0]+mat[4]+mat[8] > 0) {
    quat[0] = 0.5 * mju_sqrt(1 + mat[0] + mat[4] + mat[8]);
    quat[1] = 0.25 * (mat[7] - mat[5]) / quat[0];
    quat[2] = 0.25 * (mat[2] - mat[6]) / quat[0];
    quat[3] = 0.25 * (mat[3] - mat[1]) / quat[0];
  }

  // q1 largest
  else if (mat[0] > mat[4] && mat[0] > mat[8]) {
    quat[1] = 0.5 * mju_sqrt(1 + mat[0] - mat[4] - mat[8]);
    quat[0] = 0.25 * (mat[7] - mat[5]) / quat[1];
    quat[2] = 0.25 * (mat[1] + mat[3]) / quat[1];
    quat[3] = 0.25 * (mat[2] + mat[6]) / quat[1];
  }

  // q2 largest
  else if (mat[4] > mat[8]) {
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

  mju_normalize4(quat);
}


// time-derivative of quaternion, given 3D rotational velocity
void mju_derivQuat(mjtNum res[4], const mjtNum quat[4], const mjtNum vel[3]) {
  res[0] = 0.5*(-vel[0]*quat[1] - vel[1]*quat[2] - vel[2]*quat[3]);
  res[1] = 0.5*( vel[0]*quat[0] + vel[1]*quat[3] - vel[2]*quat[2]);
  res[2] = 0.5*(-vel[0]*quat[3] + vel[1]*quat[0] + vel[2]*quat[1]);
  res[3] = 0.5*( vel[0]*quat[2] - vel[1]*quat[1] + vel[2]*quat[0]);
}


// coefficient of expmap x (expmap x vel) in the Bortz rotation vector equation, given xx = x^2:
//  (1 - (x/2) cot(x/2)) / x^2; below |x| = 1.75 the closed form loses accuracy to cancellation,
//  so a minimax polynomial in xx is used there, with coefficients for the floating-point type of
//  mjtNum; together with the closed form above, the error is below 6 ulp on [-pi, pi];
//  singular at x = 2*pi*n, n != 0
static mjtNum bortzCoef(mjtNum xx) {
#ifdef mjUSESINGLE
  static const mjtNum k[9] = {
    0.0833333358f, 0.00138884981f, 3.32480922e-05f, 4.57548566e-07f, 4.23758962e-07f,
    -2.48791736e-07f, 8.78585737e-08f, -1.64186869e-08f, 1.26339661e-09f
  };
#else
  static const mjtNum k[9] = {
    0.083333333333333384, 0.0013888888888859755, 3.3068783094433201e-05,
    8.2671949071520734e-07, 2.0876900642039142e-08, 5.2828494303844588e-10,
    1.3455485951057888e-11, 3.1615515500778219e-13, 1.230313397310169e-14
  };
#endif

  // minimax polynomial (Horner form)
  if (xx < 1.75*1.75) {
    mjtNum res = k[8];
    for (int i=7; i >= 0; i--) {
      res = k[i] + xx*res;
    }
    return res;
  }

  // closed form
  mjtNum x = mju_sqrt(xx);
  return (1 - 0.5*x / mju_tan(0.5*x)) / xx;
}


// time-derivative of expmap in quat = quat0 * exp(expmap), given 3D rotational velocity in the
//  rotated frame: res = vel + 1/2 expmap x vel + bortzCoef(|expmap|^2) expmap x (expmap x vel),
//  i.e. dexp^{-1}_{-expmap}(vel); expmap is first replaced with the equivalent rotation vector of
//  angle in (-pi, pi], since the coefficient is singular at multiples of 2*pi;
//  res and vel can alias
void mju_derivExpmap(mjtNum res[3], const mjtNum expmap[3], const mjtNum vel[3]) {
  mjtNum r[3] = {expmap[0], expmap[1], expmap[2]};
  mjtNum xx = mju_dot3(r, r);

  // reduce the angle to (-pi, pi], like the SO3 setpoint re-anchoring in mj_advance
  if (xx > mjPI*mjPI) {
    mjtNum x = mju_sqrt(xx);
    mjtNum scale = (x - 2*mjPI*mju_floor(x/(2*mjPI) + 0.5)) / x;
    for (int k=0; k < 3; k++) {
      r[k] *= scale;
    }
    xx = mju_dot3(r, r);
  }

  mjtNum c = bortzCoef(xx);

  mjtNum rxv[3], rxrxv[3];
  mji_cross(rxv, r, vel);
  mji_cross(rxrxv, r, rxv);
  for (int k=0; k < 3; k++) {
    res[k] = vel[k] + 0.5*rxv[k] + c*rxrxv[k];
  }
}


// integrate quaternion given 3D angular velocity
void mju_quatIntegrate(mjtNum quat[4], const mjtNum vel[3], mjtNum scale) {
  mjtNum angle, tmp[4], qrot[4];

  // form local rotation quaternion, apply
  mji_copy3(tmp, vel);
  angle = scale * mju_normalize3(tmp);
  mji_axisAngle2Quat(qrot, tmp, angle);
  mju_normalize4(quat);
  mju_mulQuat(quat, quat, qrot);
}


// compute quaternion performing rotation from z-axis to given vector
void mju_quatZ2Vec(mjtNum quat[4], const mjtNum vec[3]) {
  mjtNum axis[3], a, vn[3] = {vec[0], vec[1], vec[2]}, z[3] = {0, 0, 1};

  // set default result to no-rotation quaternion
  quat[0] = 1;
  mju_zero3(quat+1);

  // normalize vector; if too small, no rotation
  if (mju_normalize3(vn) < mjMINVAL) {
    return;
  }

  // compute angle and axis
  mji_cross(axis, z, vn);
  a = mju_normalize3(axis);

  // almost parallel
  if (mju_abs(a) < mjMINVAL) {
    // opposite: 180 deg rotation around x axis
    if (mju_dot3(vn, z) < 0) {
      quat[0] = 0;
      quat[1] = 1;
    }

    return;
  }

  // make quaternion from angle and axis
  a = mju_atan2(a, mju_dot3(vn, z));
  mji_axisAngle2Quat(quat, axis, a);
}


// off-diagonal tolerance and eigenvalue swap threshold, relative to max element
#ifdef mjUSESINGLE
static const mjtNum eigTOL = 2e-6f;
static const mjtNum eigEPS = 1e-5f;
#else
static const mjtNum eigTOL = 4e-15;
static const mjtNum eigEPS = 1e-12;
#endif


// eigenvalue decomposition of symmetric 3x3 matrix, using eigTOL; returns number of iterations
int mju_eig3(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9]) {
  return mju_eig3Tol(eigval, eigvec, quat, mat, eigTOL);
}


// same as mju_eig3, stop when off-diagonal elements are below reltol times the largest element
int mju_eig3Tol(mjtNum eigval[3], mjtNum eigvec[9], mjtNum quat[4], const mjtNum mat[9],
                mjtNum reltol) {
  mjtNum D[9], tmp[9];
  mjtNum tau, t;
  int iter, rk, ck, rotk;

  // off-diagonal tolerance: no smaller than the roundoff level of D, about 16 epsilons
  mjtNum scale = 0;
  for (int i=0; i < 9; i++) {
    mjtNum abs_elem = mju_abs(mat[i]);
    if (abs_elem > scale) {
      scale = abs_elem;
    }
  }
  mjtNum eff_reltol = reltol > eigTOL ? reltol : eigTOL;
  mjtNum tol = scale * eff_reltol;

  // initialize with unit quaternion
  quat[0] = 1;
  quat[1] = quat[2] = quat[3] = 0;

  // Jacobi iteration
  for (iter=0; iter < 500; iter++) {
    // make quaternion matrix eigvec, compute D = eigvec'*mat*eigvec
    mju_quat2Mat(eigvec, quat);
    mji_mulMatTMat3(tmp, eigvec, mat);
    mji_mulMatMat3(D, tmp, eigvec);

    // assign eigenvalues
    eigval[0] = D[0];
    eigval[1] = D[4];
    eigval[2] = D[8];

    // find max off-diagonal element, set indices
    if (mju_abs(D[1]) > mju_abs(D[2]) && mju_abs(D[1]) > mju_abs(D[5])) {
      rk = 0;     // row
      ck = 1;     // column
      rotk = 2;   // rotation axis
    } else if (mju_abs(D[2]) > mju_abs(D[5])) {
      rk = 0;
      ck = 2;
      rotk = 1;
    } else {
      rk = 1;
      ck = 2;
      rotk = 0;
    }

    // terminate if max off-diagonal element too small
    if (mju_abs(D[3*rk+ck]) <= tol) {
      break;
    }

    // 2x2 symmetric Schur decomposition: t = tan(angle)
    tau = (D[4*ck]-D[4*rk])/(2*D[3*rk+ck]);
    if (tau >= 0) {
      t = 1.0/(tau + mju_sqrt(1 + tau*tau));
    } else {
      t = -1.0/(-tau + mju_sqrt(1 + tau*tau));
    }

    // express rotation as quaternion, using h = tan(angle/2): accurate for small angles
    mjtNum h = t/(1 + mju_sqrt(1 + t*t));
    tmp[0] = 1/mju_sqrt(1 + h*h);
    tmp[1] = tmp[2] = tmp[3] = 0;
    tmp[rotk+1] = (rotk == 1 ? h : -h) * tmp[0];

    // accumulate quaternion rotation
    mju_mulQuat(quat, quat, tmp);
    mju_normalize4(quat);
  }

  // sort eigenvalues in decreasing order (bubble sort: 0, 1, 0)
  mjtNum eps = scale * eigEPS;
  for (int j=0; j < 3; j++) {
    int j1 = j%2;       // lead index

    // only swap if the eigenvalues are different
    if (eigval[j1]+eps < eigval[j1+1]) {
      // swap eigenvalues
      t = eigval[j1];
      eigval[j1] = eigval[j1+1];
      eigval[j1+1] = t;

      // rotate quaternion
      tmp[0] = 0.707106781186548;  // = cos(pi/4) = sin(pi/4)
      tmp[1] = tmp[2] = tmp[3] = 0;
      tmp[(j1+2)%3+1] = tmp[0];
      mju_mulQuat(quat, quat, tmp);
      mju_normalize4(quat);
    }
  }

  // recompute eigvec
  mju_quat2Mat(eigvec, quat);

  return iter;
}


// extract 3D rotation from an arbitrary 3x3 matrix
#ifdef mjUSESINGLE
static const mjtNum rotEPS = 1e-6f;
#else
static const mjtNum rotEPS = 1e-9;
#endif
int mju_mat2Rot(mjtNum quat[4], const mjtNum mat[9]) {
  // Müller, Matthias, Jan Bender, Nuttapong Chentanez, and Miles Macklin.
  // "A robust method to extract the rotational part of deformations."
  // In Proceedings of the 9th International Conference on Motion in Games,
  // pp. 55-60. 2016.

  int iter;
  mju_normalize4(quat);
  mjtNum col1_mat[3] = {mat[0], mat[3], mat[6]};
  mjtNum col2_mat[3] = {mat[1], mat[4], mat[7]};
  mjtNum col3_mat[3] = {mat[2], mat[5], mat[8]};
  for (iter = 0; iter < 500; iter++) {
    mjtNum rot[9];
    mju_quat2Mat(rot, quat);
    mjtNum col1_rot[3] = {rot[0], rot[3], rot[6]};
    mjtNum col2_rot[3] = {rot[1], rot[4], rot[7]};
    mjtNum col3_rot[3] = {rot[2], rot[5], rot[8]};
    mjtNum omega[3], vec1[3], vec2[3], vec3[3];
    mji_cross(vec1, col1_rot, col1_mat);
    mji_cross(vec2, col2_rot, col2_mat);
    mji_cross(vec3, col3_rot, col3_mat);
    mji_add3(omega, vec1, vec2);
    mji_addTo3(omega, vec3);
    mjtNum dot[3] = {mju_dot3(col1_rot, col1_mat),
                     mju_dot3(col2_rot, col2_mat),
                     mju_dot3(col3_rot, col3_mat)};
    mjtNum trace = dot[0] + dot[1] + dot[2];
    mju_scl3(omega, omega, 1.0 / (mju_abs(trace) + mjMINVAL));
    mjtNum w = mju_normalize3(omega);
    if (w < rotEPS) {
      mjtNum A[9], sym[9];
      mji_mulMatTMat3(A, rot, mat);
      sym[0] = A[0];
      sym[1] = 0.5 * (A[1] + A[3]);
      sym[2] = 0.5 * (A[2] + A[6]);
      sym[3] = sym[1];
      sym[4] = A[4];
      sym[5] = 0.5 * (A[5] + A[7]);
      sym[6] = sym[2];
      sym[7] = sym[5];
      sym[8] = A[8];

      // fast path: T = tr(S)*I - S is strictly positive definite iff rotation maximizes tr(R'*M)
      mjtNum T00 = sym[4] + sym[8], T11 = sym[0] + sym[8], T22 = sym[0] + sym[4];
      mjtNum detT = T00 * (T11 * T22 - sym[5] * sym[5]) - sym[1] * sym[1] * T22 -
                    sym[2] * sym[2] * T11 - 2 * sym[1] * sym[2] * sym[5];
      mjtNum scale = mju_abs(sym[0]) + mju_abs(sym[4]) + mju_abs(sym[8]) + mjMINVAL;
      mjtNum tol = rotEPS * scale, tol2 = tol * scale, tol3 = tol2 * scale;
      if (T00 > tol && (T00 * T11 - sym[1] * sym[1]) > tol2 && detT > tol3) {
        break;
      }

      mjtNum eigval[3], eigvec[9], qtmp[4];
      mju_eig3(eigval, eigvec, qtmp, sym);
      if (eigval[0] <= trace + rotEPS * (mju_abs(eigval[0]) + mju_abs(trace) + mjMINVAL)) {
        break;
      }

      // stationary point (minimum or saddle): flip 180 degrees about rot * top eigenvector, which
      // strictly increases the trace
      mjtNum v_max[3] = {eigvec[0], eigvec[3], eigvec[6]};
      mjtNum axis[3];
      mji_mulMatVec3(axis, rot, v_max);
      mjtNum qflip[4] = {0, axis[0], axis[1], axis[2]};
      mju_mulQuat(quat, qflip, quat);
      mju_normalize4(quat);
      continue;
    }
    mjtNum qrot[4];
    mji_axisAngle2Quat(qrot, omega, w);
    mju_mulQuat(quat, qrot, quat);
    mju_normalize4(quat);
  }
  return iter;
}


//------------------------------ pose operations (quat, pos) ---------------------------------------

// multiply two poses
void mju_mulPose(mjtNum posres[3], mjtNum quatres[4],
                 const mjtNum pos1[3], const mjtNum quat1[4],
                 const mjtNum pos2[3], const mjtNum quat2[4]) {
  mjtNum ptmp[3], qtmp[4];

  // ptmp = quat1*pos2 + pos1
  mji_rotVecQuat(ptmp, pos2, quat1);
  mji_addTo3(ptmp, pos1);

  // qtmp = quat1*quat2
  mji_mulQuat(qtmp, quat1, quat2);
  mju_normalize4(qtmp);

  mji_copy3(posres, ptmp);
  mji_copy4(quatres, qtmp);
}


// negate pose
void mju_negPose(mjtNum posres[3], mjtNum quatres[4], const mjtNum pos[3], const mjtNum quat[4]) {
  mjtNum ptmp[3], qtmp[4];

  // qtmp = neg(quat)
  mji_negQuat(qtmp, quat);

  // posres = -neg(quat)*pos
  mji_rotVecQuat(ptmp, pos, qtmp);
  mji_scl3(posres, ptmp, -1);

  mji_copy4(quatres, qtmp);
}


// transform vector by pose
void mju_trnVecPose(mjtNum res[3], const mjtNum pos[3], const mjtNum quat[4], const mjtNum vec[3]) {
  mjtNum tmp[3];

  // res = quat*vec + pos
  mji_rotVecQuat(tmp, vec, quat);
  mji_addTo3(tmp, pos);
  mji_copy3(res, tmp);
}


//------------------------------ spatial algebra ---------------------------------------------------

// vector cross-product, 3D
void mju_cross(mjtNum res[3], const mjtNum a[3], const mjtNum b[3]) {
  mjtNum tmp[3] = {
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
  };
  res[0] = tmp[0];
  res[1] = tmp[1];
  res[2] = tmp[2];
}


// cross-product for motion vector
void mju_crossMotion(mjtNum res[6], const mjtNum vel[6], const mjtNum v[6]) {
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
void mju_crossForce(mjtNum res[6], const mjtNum vel[6], const mjtNum f[6]) {
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


// express inertia in com-based frame
void mju_inertCom(mjtNum* restrict res, const mjtNum inert[3], const mjtNum mat[9],
                  const mjtNum dif[3], mjtNum mass) {
  // tmp = diag(inert) * mat'  (mat is local-to-global rotation)
  mjtNum tmp[9] = {mat[0]*inert[0], mat[3]*inert[0], mat[6]*inert[0],
                   mat[1]*inert[1], mat[4]*inert[1], mat[7]*inert[1],
                   mat[2]*inert[2], mat[5]*inert[2], mat[8]*inert[2]};

  // res_rot = mat * diag(inert) * mat'
  res[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6];
  res[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7];
  res[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8];
  res[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7];
  res[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8];
  res[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8];

  // res_rot -= mass * dif_cross * dif_cross
  res[0] += mass*(dif[1]*dif[1] + dif[2]*dif[2]);
  res[1] += mass*(dif[0]*dif[0] + dif[2]*dif[2]);
  res[2] += mass*(dif[0]*dif[0] + dif[1]*dif[1]);
  res[3] -= mass*dif[0]*dif[1];
  res[4] -= mass*dif[0]*dif[2];
  res[5] -= mass*dif[1]*dif[2];

  // res_tran = mass * dif
  res[6] = mass*dif[0];
  res[7] = mass*dif[1];
  res[8] = mass*dif[2];

  // res_mass = mass
  res[9] = mass;
}


// multiply 6D vector (rotation, translation) by 6D inertia matrix
void mju_mulInertVec(mjtNum* restrict res, const mjtNum i[10], const mjtNum v[6]) {
  res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
  res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
  res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
  res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
  res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
  res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}


// express motion axis in com-based frame
void mju_dofCom(mjtNum* restrict res, const mjtNum axis[3], const mjtNum offset[3]) {
  // hinge
  if (offset) {
    mji_copy3(res, axis);
    mji_cross(res+3, axis, offset);
  }

  // slide
  else {
    mju_zero3(res);
    mji_copy3(res+3, axis);
  }
}


// multiply dof matrix (6-by-n, transposed) by vector (n-by-1)
void mju_mulDofVec(mjtNum* res, const mjtNum* dof, const mjtNum* vec, int n) {
  if (n == 1) {
    mju_scl(res, dof, vec[0], 6);
  } else if (n <= 0) {
    mju_zero(res, 6);
  } else {
    mju_mulMatTVec(res, dof, vec, n, 6);
  }
}


// transform 6D motion or force vector between frames
//   flg_force: determines vector type (motion or force)
//   rotnew2old: rotation that maps vectors from new to old frame,
//               its columns are the new frame's axes, expressed in the old frame
//   oldpos and newpos are expressed in old frame
void mju_transformSpatial(mjtNum res[6], const mjtNum vec[6], int flg_force,
                          const mjtNum newpos[3], const mjtNum oldpos[3],
                          const mjtNum rotnew2old[9]) {
  mjtNum cros[3], dif[3], tran[6];

  // apply translation
  mju_copy(tran, vec, 6);
  mji_sub3(dif, newpos, oldpos);
  if (flg_force) {
    mji_cross(cros, dif, vec+3);
    mji_sub3(tran, vec, cros);
  } else {
    mji_cross(cros, dif, vec);
    mji_sub3(tran+3, vec+3, cros);
  }

  // if provided, apply old -> new rotation
  if (rotnew2old) {
    mji_mulMatTVec3(res, rotnew2old, tran);
    mji_mulMatTVec3(res+3, rotnew2old, tran+3);
  }

  // otherwise copy
  else {
    mji_copy6(res, tran);
  }
}


// make 3D frame given x-axis (normal) and possibly y-axis (tangent 1)
void mju_makeFrame(mjtNum frame[9]) {
  mjtNum tmp[3];

  // normalize xaxis
  if (mju_normalize3(frame) < 0.5) {
    mjERROR("xaxis of contact frame undefined");
  }

  // if yaxis undefined, set yaxis to (0,1,0) if possible, otherwise (0,0,1)
  if (mju_dot3(frame+3, frame+3) < 0.25) {
    mju_zero3(frame+3);

    if (frame[1] < 0.5 && frame[1] > -0.5) {
      frame[4] = 1;
    } else {
      frame[5] = 1;
    }
  }

  // make yaxis orthogonal to xaxis
  mji_scl3(tmp, frame, mju_dot3(frame, frame+3));
  mji_subFrom3(frame+3, tmp);
  mju_normalize3(frame+3);

  // zaxis = cross(xaxis, yaxis)
  mji_cross(frame+6, frame, frame+3);
}


// convert sequence of Euler angles (radians) to quaternion
// seq[0,1,2] must be in 'xyzXYZ', lower/upper-case mean intrinsic/extrinsic rotations
void mju_euler2Quat(mjtNum quat[4], const mjtNum euler[3], const char* seq) {
  if (strnlen(seq, 4) != 3) {
    mjERROR("seq must contain exactly 3 characters");
  }

  // init
  mjtNum tmp[4] = {1, 0, 0, 0};

  // loop over euler angles, accumulate rotations
  for (int i=0; i<3; i++) {
    // construct quaternion rotation
    mjtNum rot[4] = {cos(euler[i]/2), 0, 0, 0};
    mjtNum sa = sin(euler[i]/2);
    if (seq[i]=='x' || seq[i]=='X') {
      rot[1] = sa;
    } else if (seq[i]=='y' || seq[i]=='Y') {
      rot[2] = sa;
    } else if (seq[i]=='z' || seq[i]=='Z') {
      rot[3] = sa;
    } else {
      mjERROR("seq[%d] is '%c', should be one of x, y, z, X, Y, Z", i, seq[i]);
    }

    // accumulate rotation
    if (seq[i]=='x' || seq[i]=='y' || seq[i]=='z') {
      mju_mulQuat(tmp, tmp, rot);  // moving axes: post-multiply
    } else {
      mju_mulQuat(tmp, rot, tmp);  // fixed axes: pre-multiply
    }
  }

  mji_copy4(quat, tmp);
}
