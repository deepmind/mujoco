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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_CORE_UTIL_H_
#define MUJOCO_SRC_ENGINE_ENGINE_CORE_UTIL_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"

#ifdef __cplusplus
extern "C" {
#endif


//-------------------------- model properties ------------------------------------------------------

// determine type of friction cone
MJAPI int mj_isPyramidal(const mjModel* m);

// determine type of constraint Jacobian
MJAPI int mj_isSparse(const mjModel* m);

// determine type of solver
MJAPI int mj_isDual(const mjModel* m);


//-------------------------- sparse chains ---------------------------------------------------------

// merge dof chains for two bodies
int mj_mergeChain(const mjModel* m, int* chain, int b1, int b2, int flg_skipcommon);

// merge dof chains for two different simple bodies
int mj_mergeChainSimple(const mjModel* m, int* chain, int b1, int b2);

// get body chain
int mj_bodyChain(const mjModel* m, int body, int* chain);


//-------------------------- Jacobians -------------------------------------------------------------

// compute 3/6-by-nv Jacobian of global point attached to given body
MJAPI void mj_jac(const mjModel* m, const mjData* d,
                  mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body);

// compute body frame Jacobian
MJAPI void mj_jacBody(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int body);

// compute body center-of-mass Jacobian
MJAPI void mj_jacBodyCom(const mjModel* m, const mjData* d,
                         mjtNum* jacp, mjtNum* jacr, int body);

// compute subtree center-of-mass Jacobian
MJAPI void mj_jacSubtreeCom(const mjModel* m, mjData* d, mjtNum* jacp, int body);

// compute geom Jacobian
MJAPI void mj_jacGeom(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int geom);

// compute site Jacobian
MJAPI void mj_jacSite(const mjModel* m, const mjData* d,
                      mjtNum* jacp, mjtNum* jacr, int site);

// compute translation Jacobian of point, and rotation Jacobian of axis
MJAPI void mj_jacPointAxis(const mjModel* m, mjData* d,
                           mjtNum* jacPoint, mjtNum* jacAxis,
                           const mjtNum point[3], const mjtNum axis[3], int body);

// compute 3/6-by-nv sparse Jacobian of global point attached to given body
void mj_jacSparse(const mjModel* m, const mjData* d,
                  mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                  int NV, const int* chain, int flg_skipcommon);

// sparse Jacobian difference for simple body contacts
void mj_jacSparseSimple(const mjModel* m, const mjData* d,
                        mjtNum* jacdifp, mjtNum* jacdifr, const mjtNum* point,
                        int body, int flg_second, int NV, int start);

// compute 3/6-by-NV sparse Jacobian time derivative of global point attached to given body
MJAPI void mj_jacDotSparse(const mjModel* m, const mjData* d,
                           mjtNum* jacp, mjtNum* jacr, const mjtNum* point, int body,
                           int NV, const int* chain);

// dense or sparse Jacobian difference for two body points: pos2 - pos1, global
MJAPI int mj_jacDifPair(const mjModel* m, const mjData* d, int* chain,
                        int b1, int b2, const mjtNum pos1[3], const mjtNum pos2[3],
                        mjtNum* jac1p, mjtNum* jac2p, mjtNum* jacdifp,
                        mjtNum* jac1r, mjtNum* jac2r, mjtNum* jacdifr,
                        int issparse, int flg_skipcommon);

// dense or sparse weighted sum of multiple body Jacobians at same point
int mj_jacSum(const mjModel* m, mjData* d, int* chain,
              int n, const int* body, const mjtNum* weight,
              const mjtNum point[3], mjtNum* jacp, mjtNum* jacr, int flg_rot);

// compute bodies and weights of one side of a contact, return number of bodies
//   the weights interpolate the side at the contact point; weight can be NULL
MJAPI int mj_contactBodyWeight(const mjModel* m, const mjData* d,
                               const mjContact* con, int side, int* body,
                               mjtNum* weight);

// compute 3/6-by-nv Jacobian time derivative of global point attached to given body
MJAPI void mj_jacDot(const mjModel* m, const mjData* d,
                     mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body);

// compute subtree angular momentum matrix
MJAPI void mj_angmomMat(const mjModel* m, mjData* d, mjtNum* mat, int body);


//-------------------------- coordinate transformation ---------------------------------------------

// compute object 6D velocity in object-centered frame, world/local orientation
MJAPI void mj_objectVelocity(const mjModel* m, const mjData* d,
                             int objtype, int objid, mjtNum res[6], int flg_local);

// compute material surface velocity of a geom at a point, in world frame
void mj_geomSurfaceVelocity(const mjModel* m, const mjData* d, int geomid,
                            const mjtNum point[3], mjtNum linear[3], mjtNum angular[3]);

// compute object 6D acceleration in object-centered frame, world/local orientation
MJAPI void mj_objectAcceleration(const mjModel* m, const mjData* d,
                                 int objtype, int objid, mjtNum res[6], int flg_local);

// map from body local to global Cartesian coordinates
MJAPI void mj_local2Global(mjData* d, mjtNum xpos[3], mjtNum xmat[9],
                           const mjtNum pos[3], const mjtNum quat[4],
                           int body, mjtByte sameframe);

// map from body local to global Cartesian position, quaternion and matrix
void mj_local2GlobalQuat(mjData* d, mjtNum* xpos, mjtNum* xquat, mjtNum* xmat,
                         const mjtNum pos[3], const mjtNum quat[4],
                         int body, mjtByte sameframe);


//-------------------------- miscellaneous ---------------------------------------------------------

// Fast-path eligibility for standard flex attachment mappings.
MJAPI int mj_flexBodySimple(const mjModel* m, int body);
MJAPI int mj_flexSimple(const mjModel* m, int f);

// For standard flexes, gather J*vec in world coordinates; scatter adds scale*J'*vec to forces.
// Requires kinematics and comPos, and current flexvert_xpos for general attachments.
MJAPI void mj_flexGather(const mjModel* m, const mjData* d, int f, mjtNum* res, const mjtNum* vec);
MJAPI void mj_flexScatter(const mjModel* m, const mjData* d, int f, mjtNum* res,
                          const mjtNum* vec, mjtNum scale);

// gather global node positions and velocities
MJAPI void mju_flexGatherState(const mjModel* m, const mjData* d, int f, mjtNum* xpos, mjtNum* vel);

// extract 6D force:torque for one contact, in contact frame
MJAPI void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6]);

// count the number of length limit violations for tendon i (0, 1 or 2)
int tendonLimit(const mjModel* m, const mjtNum* ten_length, int i);

// compute spring and damper forces along tendon i, zero when disabled
void mj_tendonSpringDamper(const mjModel* m, const mjData* d, int i,
                           mjtNum* frc_spring, mjtNum* frc_damper);

// return actuator damping contribution to joint or tendon
MJAPI mjtNum mj_actuatorDamping(const mjModel* m, mjtObj type, int id, mjtNum poly[mjNPOLY]);

// return actuator armature contribution to joint or tendon
MJAPI mjtNum mj_actuatorArmature(const mjModel* m, mjtObj type, int id);

// return DC motor winding resistance at the current temperature
mjtNum mj_dcmotorResistance(const mjModel* m, const mjData* d, int id);

// high-level warning function: count warnings in mjData, print only the first time
MJAPI void mj_warning(mjData* d, int warning, int info);


//-------------------------- effective-metric predicates ------------------------------------------

// the selected integrator performs the constraint solve in the effective metric
int mj_isMetric(const mjModel* m);

// do the tendon and actuator classes enter the metric (excluded under solver=PGS only;
// noslip atop a primal solver keeps them)
MJAPI int mj_effCouplings(const mjModel* m);

// tendon i has a spring: nonzero stiffness or stiffness polynomial
int mj_tendonHasStiffness(const mjModel* m, int i);

// tendon i has a damper: nonzero damping, damping polynomial, or an attached actuator
int mj_tendonHasDamping(const mjModel* m, int i);

// flex f has a nonzero stretch stiffness record; for volume-mode interpolated flexes,
// any cell counts (empty cells carry an all-zero block)
int mj_flexHasStiffness(const mjModel* m, int f);

// does flex f use the passive contact path (metric-carried contacts)
MJAPI int mj_effFlexContactPossible(const mjModel* m, int f);

// does flex f contribute elastic stiffness to the metric
int mj_effFlexStiffPossible(const mjModel* m, int f);

// does flex f need the implicit metric treatment: elastic stiffness or passive contact
MJAPI int mj_effFlexPossible(const mjModel* m, int f);

// does any flex need the implicit metric treatment: elastic stiffness or passive contact
int mj_effFlexAny(const mjModel* m);

// can this tendon contribute to the metric (model-level; mirrored by island discovery
// and the sleep wake rule)
MJAPI int mj_effTendonPossible(const mjModel* m, int i);

// can this actuator contribute to the metric (model-level type check)
int mj_effActuatorPossible(const mjModel* m, int i);


//-------------------------- flex elasticity -------------------------------------------------------

// lazily assemble the unscaled Cartesian stretch stiffness for a standard 2D or 3D flex
// SNH projects each material Hessian to PSD before pulling back and assembling
// stores symmetric diagonal blocks per vertex and oriented off-diagonal blocks per edge
void mj_flexHessian(const mjModel* m, mjData* d, int f);

// add scale * cached Hessian * vec to res; vectors use flex-local Cartesian vertex order
void mj_flexHessianMul(const mjModel* m, const mjData* d, int f, mjtNum* res,
                       const mjtNum* vec, mjtNum scale);

//-------------------------- 1D rod bending --------------------------------------------------------

// per-record data layout in flex_bending for 1D flexes, after a leading record count; a record is a
// single hinge (v0, v1, v2) or a pair of adjacent curved hinges (v0, v1, v2) and (v1, v2, v3):
//   single: [v0, v1, v2, -1, k, 0, kb0, 0, 0, 0], energy k (|kb| - kb0)^2, with kb0 = |kb| in the
//           reference (0 if straight, then the energy is k |kb|^2)
//   pair:   [v0, v1, v2, v3, w0, w1, om0[4]], energy min_a sum_i w_i |om_i - R(a) om0_i|^2, with
//           om the curvature binormals of both hinges in a frame orthogonal to the shared edge
//           (v1, v2), om0 their reference values and R(a) a rotation about the edge
// k = EI / (l0 + l1) is the stiffness of each hinge; adjacent curved hinges are paired across every
// shared edge, splitting the stiffness of each hinge evenly among its pairs (w0, w1). There are no
// material frames, so the rod has no torsional stiffness: the minimization over the edge rotation
// lets the bending planes of adjacent hinges rotate relative to each other at no cost in the limit
// of fine discretization, and only the curvature magnitude is resisted (as for a single hinge).
#define mjROD_HINGE_SIZE 10

// number of vertices of a 1D rod bending record: 3 (single hinge) or 4 (hinge pair)
static inline int mj_rodHingeVert(const mjtNum* hinge) {
  return hinge[3] < 0 ? 3 : 4;
}

// curvature binormal kb = 2 e0 x e1 / (|e0||e1| + e0.e1) of hinge (xa, xv, xb), with
// e0 = xv - xa and e1 = xb - xv, and its Jacobian J (3x9, columns xa, xv, xb);
// returns 0 if the hinge is degenerate or fully folded
int mj_rodHinge(const mjtNum xa[3], const mjtNum xv[3], const mjtNum xb[3],
                mjtNum kb[3], mjtNum J[27]);

// weights of the single hinge energy in kb space, all including the factor 2k:
//   g:  gradient dE/d(kb)
//   Wk: Hessian d2E/d(kb)2, projected to PSD (exact at the reference)
//   Wd: Rayleigh damping weight, restricted to the rate of |kb| (rotation invariant)
// a curved reference (kb0 > 0) is smoothed at kb = 0; each output is optional
void mj_rodHingeWeights(mjtNum g[3], mjtNum Wk[9], mjtNum Wd[9],
                        const mjtNum kb[3], mjtNum k, mjtNum kb0);

// curvature binormals om[4] of hinges (x0, x1, x2) and (x1, x2, x3) in a frame orthogonal to the
// edge (x1, x2), and their Jacobian J (4x12, columns x0..x3, frame held fixed: the energy does not
// depend on the frame); returns 0 if either hinge or the shared edge is degenerate
int mj_rodPair(const mjtNum x0[3], const mjtNum x1[3], const mjtNum x2[3], const mjtNum x3[3],
               mjtNum om[4], mjtNum J[48]);

// weights of the hinge pair energy in om space:
//   g:  gradient dE/d(om)
//   Wk: Hessian d2E/d(om)2, projected to PSD (exact at the reference)
//   Wd: Rayleigh damping weight, orthogonal to rotations about the shared edge
// the energy is smoothed at om = 0; each output is optional
void mj_rodPairWeights(mjtNum g[4], mjtNum Wk[16], mjtNum Wd[16],
                       const mjtNum om[4], mjtNum w0, mjtNum w1, const mjtNum om0[4]);

// element-local geometry, quadratic edge response, and stiffness contractions shared by
// passive forces and both solver paths; keep the helpers visible to the compiler so the
// small edge loops can be optimized together with their callers

// local edges for triangles and tetrahedra, in the compiler's stiffness ordering
static const int mj_stretchEdges[2][6][2] = {
  {{1, 2}, {2, 0}, {0, 1}, {0, 0}, {0, 0}, {0, 0}},
  {{0, 1}, {1, 2}, {2, 0}, {2, 3}, {0, 3}, {1, 3}}
};


// x_i - x_j: half the squared-length gradient at vertex i
static inline void mj_stretchEdgeVectors(mjtNum edgevec[6][3], const mjtNum* xpos,
                                         const int* vert, int dim) {
  int nedge = dim == 2 ? 3 : 6;
  const int (*edge)[2] = mj_stretchEdges[dim-2];
  for (int e = 0; e < nedge; e++) {
    for (int x = 0; x < 3; x++) {
      edgevec[e][x] = xpos[3*vert[edge[e][0]]+x] - xpos[3*vert[edge[e][1]]+x];
    }
  }
}


// unpack the symmetric quadratic metric (the first 21 numbers for tetrahedra)
static inline void mj_stretchMetric(mjtNum metric[36], const mjtNum* packed, int nedge) {
  int id = 0;
  for (int a = 0; a < nedge; a++) {
    for (int b = a; b < nedge; b++) {
      metric[nedge*a + b] = packed[id];
      metric[nedge*b + a] = packed[id++];
    }
  }
}


// gather squared-length differences in local edge order
static inline void mj_stretchElongation(mjtNum elongation[6], const int* edge,
                                        const mjtNum* length, const mjtNum* reference,
                                        const mjtNum* plastic, int nedge) {
  for (int e = 0; e < nedge; e++) {
    int idx = edge[e];
    elongation[e] = length[idx]*length[idx] - reference[idx]*reference[idx] -
                    (plastic ? plastic[e] : 0);
  }
}


// multiply the symmetric edge metric, also used for damping and stiffness-vector products
static inline void mj_stretchTension(mjtNum tension[6], const mjtNum metric[36],
                                     const mjtNum elongation[6], int nedge) {
  for (int e = 0; e < nedge; e++) {
    tension[e] = 0;
    for (int a = 0; a < nedge; a++) {
      tension[e] += metric[nedge*e + a]*elongation[a];
    }
  }
}


// quadratic energy s' K s / 4: its edge tension is K s, for both StVK and SNH
static inline void mj_stretchElasticity(mjtNum metric[36], mjtNum tension[6],
                                        const mjtNum* packed, const mjtNum elongation[6],
                                        int nedge) {
  mj_stretchMetric(metric, packed, nedge);
  mj_stretchTension(tension, metric, elongation, nedge);
}


// accumulate world-frame vertex forces, independently of the body Jacobians
static inline void mj_stretchForce(mjtNum* force, const int* vert, const mjtNum tension[6],
                                   mjtNum edgevec[6][3], int dim) {
  int nedge = dim == 2 ? 3 : 6;
  const int (*edge)[2] = mj_stretchEdges[dim-2];
  for (int e = 0; e < nedge; e++) {
    for (int i = 0; i < 2; i++) {
      for (int x = 0; x < 3; x++) {
        mjtNum gradient = i == 0 ? edgevec[e][x] : -edgevec[e][x];
        force[3*vert[edge[e][i]]+x] -= tension[e]*gradient;
      }
    }
  }
}


// prepare the material and geometric coefficients shared by both solver paths
static inline void mj_stretchStiffness(mjtNum metric[36], mjtNum tension[6], const mjtNum* packed,
                                       const int* edge, const mjtNum* length,
                                       const mjtNum* reference, const mjtNum* plastic, int nedge) {
  mjtNum elongation[6];
  mj_stretchElongation(elongation, edge, length, reference, plastic, nedge);
  mj_stretchElasticity(metric, tension, packed, elongation, nedge);

  // a compressed edge's geometric block is negative semidefinite; keep only the tensile
  // part for the SPD solver metric, exactly as in the original StVK operator and assembler
  // passive forces retain the full tension, and the material metric is not modified
  for (int e = 0; e < nedge; e++) {
    tension[e] = mju_max(tension[e], 0);
  }
}


// evaluate the corresponding 3x3 world-frame vertex-pair block for sparse assembly
static inline void mj_stretchStiffnessBlock(mjtNum block[9], const mjtNum metric[36],
                                            const mjtNum tension[6], mjtNum edgevec[6][3],
                                            int dim, int i, int j, mjtNum scale) {
  int nedge = dim == 2 ? 3 : 6;
  const int (*edge)[2] = mj_stretchEdges[dim-2];
  mju_zero(block, 9);
  for (int a = 0; a < nedge; a++) {
    mjtNum sa = (i == edge[a][0]) ? 1 : ((i == edge[a][1]) ? -1 : 0);
    if (!sa) continue;
    for (int b = 0; b < nedge; b++) {
      mjtNum sb = (j == edge[b][0]) ? 1 : ((j == edge[b][1]) ? -1 : 0);
      if (!sb) continue;
      mjtNum w = 2*scale*metric[nedge*a + b]*sa*sb;
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          block[3*r+c] += w*edgevec[a][r]*edgevec[b][c];
        }
      }
    }
  }
  mjtNum geo = 0;
  for (int a = 0; a < nedge; a++) {
    mjtNum sa = (i == edge[a][0]) ? 1 : ((i == edge[a][1]) ? -1 : 0);
    mjtNum sb = (j == edge[a][0]) ? 1 : ((j == edge[a][1]) ? -1 : 0);
    if (!sa || !sb) continue;
    geo += tension[a]*sa*sb;
  }
  geo *= scale;
  block[0] += geo;
  block[4] += geo;
  block[8] += geo;
}

//-------------------------- Stable Neo-Hookean tetrahedra ---------------------

// Standard 3D elements use a 24-number flex_stiffness record. StVK stores its
// packed quadratic metric in [0:21] and zeros in [21:24]. For SNH the record
// stores:
//   [0:21] symmetric K, [21] gamma = -mu / (72 * V0), [22] beta = V0*(lambda+2
//   mu)/2, [23] 1/det(Dm), where V0 = abs(det(Dm))/6 and mu,lambda are the Lame
//   parameters.
// Energy: s' K s / 4 + gamma P(s) + beta (J-1)^2, with s_e = L_e^2 - L0_e^2.
// P(s) is the determinant of the Gram-matrix difference D(s), defined in
// mj_snhCubic. This is exactly V0*[mu/2*(tr(F'F)-3) - mu*(J-1) +
// (lambda+mu)/2*(J-1)^2]. No division by current volume is needed, even at
// collapse or inversion. The nonzero cubic coefficient in [21] identifies SNH
// without an extra mjModel field.

// add twice the gradient of gamma P(s) to the edge tension
void mj_snhCubic(mjtNum tension[6], const mjtNum s[6], mjtNum gamma);

// signed volume ratio and its four world-space vertex gradients
static inline mjtNum mj_snhVolume(mjtNum grad[4][3], mjtNum edgevec[6][3],
                                  const mjtNum k[24]) {
  // edges from vertex 0 to vertices 1,2,3 in mj_stretchEdges ordering
  mjtNum a[3], b[3], c[3];
  for (int x = 0; x < 3; x++) {
    a[x] = -edgevec[0][x];
    b[x] =  edgevec[2][x];
    c[x] = -edgevec[4][x];
  }
  mju_cross(grad[1], b, c);
  mju_cross(grad[2], c, a);
  mju_cross(grad[3], a, b);
  mjtNum J = k[23]*mju_dot3(a, grad[1]);
  for (int x = 0; x < 3; x++) {
    grad[0][x] = 0;
    for (int v = 1; v < 4; v++) {
      grad[v][x] *= k[23];
      grad[0][x] -= grad[v][x];
    }
  }
  return J;
}


#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_CORE_UTIL_H_
