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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_H_

#include <stddef.h>

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// derivatives of mju_subQuat w.r.t inputs
MJAPI void mjd_subQuat(const mjtNum qa[4], const mjtNum qb[4], mjtNum Da[9], mjtNum Db[9]);

// derivatives of mju_quatIntegrate w.r.t inputs
MJAPI void mjd_quatIntegrate(const mjtNum vel[3], mjtNum scale,
                             mjtNum Dquat[9], mjtNum Dvel[9], mjtNum Dscale[3]);

// analytical derivative of smooth forces w.r.t velocities:
//   d->qDeriv = d (qfrc_actuator + qfrc_passive - [qfrc_bias]) / d qvel
MJAPI void mjd_smooth_vel(const mjModel* m, mjData* d, int flg_bias);

// add (d qfrc_actuator / d qvel) to qDeriv
MJAPI void mjd_actuator_vel(const mjModel* m, mjData* d);

// add (d qfrc_passive / d qvel) to qDeriv
MJAPI void mjd_passive_vel(const mjModel* m, mjData* d);

// subtract (d qfrc_bias / d qvel) from qDeriv (dense version)
MJAPI void mjd_rne_vel_dense(const mjModel* m, mjData* d);

// return 1 if body is the root of a free rigid subtree: a free joint with fixed descendants
mjtBool mj_isFreeBody(const mjModel* m, int body);

// 6x6 block B = d qfrc_bias / d qvel for the free joint of a rigid subtree
//   requires valid d->crb, computed by mj_crb
MJAPI void mjd_freeBias_vel(const mjModel* m, const mjData* d, int jnt, mjtNum B[36]);

// 6x6 block A = M - h * (d qfrc_smooth / d qvel) for the free joint of a rigid subtree
//   returns 1 and writes A if jnt is the free joint of an awake rigid subtree, 0 otherwise
//   requires valid d->qDeriv rows for the block, computed with flg_bias = 0
MJAPI int mjd_freeMhat(const mjModel* m, const mjData* d, int jnt, mjtNum h, mjtNum A[36],
                       int flg_discrete);

// can this free rigid subtree take the local gyroscopic treatment under discrete
MJAPI int mjd_freeGyroPossible(const mjModel* m, const mjData* d, int jnt);

// compute res += (s1 + s2*damping) * J'*K*J * vec, for all interpolated flexes
//   K_rot_cache: if non-NULL, use pre-cached K_rot (same layout as m->flex_stiffness)
MJAPI void mjd_flexInterp_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                              mjtNum s1, mjtNum s2, const mjtNum* K_rot_cache);

// stack bytes mjd_flexInterp_mul takes, at most
size_t mjd_flexInterp_mulBytes(const mjModel* m);

// precompute unscaled K_rot for all elements into cache (same layout as m->flex_stiffness)
MJAPI void mjd_flexInterp_cacheKrot(const mjModel* m, mjData* d, mjtNum* K_rot_out);

// compute res += scale * K_bend * vec for standard (non-interp) flex bending, over the convex
// stencils
//   scale = s1 + s2 * flex_damping[f]  per flex
MJAPI void mjd_flexBend_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                            mjtNum s1, mjtNum s2);

// compute res += scale * K_stretch * vec for standard (non-interp) flex stretch,
// K_stretch is the PSD world-space stretch Hessian projected through the vertex Jacobians
//   scale = s1 + s2 * flex_damping[f]  per flex
MJAPI void mjd_flexStretch_mul(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                               mjtNum s1, mjtNum s2);

// assemble the standard-flex implicit stiffness (s1 + s2*damping)*(K_bend + K_stretch) into
// dof-level CSR; phase 1 (colind==NULL) fills rownnz/rowadr and returns total nnz, phase 2
// fills colind and, unless val is NULL (structure only), val. Interp flexes are assembled iff
// Krot (mjd_flexInterp_cacheKrot cache) is non-NULL and the centered fast path applies (check
// mjd_flexInterpAssemblable first).
// The flex-contact law, shared by the passive penalty and the IPC contact mode. A pair with
// normal row `row` and stiffness `scale` costs
//
//   0.5 * scale * d^2,   d = gap - s - lam/k,   s = max(0, gap - lam/k)
//
// giving a force -scale*d along the row and a curvature scale*row'row in the metric. `gap` is the
// pair's signed distance in the producer's own convention (surface distance for the penalty,
// midsurface distance less the standoff for IPC), `lam` the augmented-Lagrangian multiplier and
// `k` the stiffness it is defined against. The passive penalty is the case lam = 0, scale = k.
// The slack is frozen across an inner solve: computed once per outer iteration, passed back in.
// Convention: scale*row'row must equal h^2 * k * J'J, however the producer splits the factors.
MJAPI mjtNum mjd_flexContactSlack(mjtNum k, mjtNum gap, mjtNum lam);
MJAPI mjtNum mjd_flexContactResidual(mjtNum k, mjtNum gap, mjtNum s, mjtNum lam);

// natural frequency of the law: pair stiffness = mjFLEXCONTACT_OMEGA2 * min nonzero vertex mass
#define mjFLEXCONTACT_OMEGA2 5e7

// point mass of global flex vertex gv, 0 when it has no 3-dof body of its own (pinned)
MJAPI mjtNum mjd_flexVertMass(const mjModel* m, const mjData* d, int gv);

// count dynamic (3-dof) cage nodes of interpolated flex f
int mjd_flexInterpDynNodes(const mjModel* m, int f);

// passive contact stiffness of a pair; force and Hessian must use the same value
//   flex_ndyn (optional, size nflex) provides precomputed mjd_flexInterpDynNodes(m, f)
MJAPI mjtNum mjd_flexContactStiffness(const mjModel* m, const mjData* d, const mjContact* con,
                                      const int* flex_ndyn);

MJAPI int mjd_flexStiff_assemble(const mjModel* m, mjData* d, int* rownnz, int* rowadr,
                                 int* colind, mjtNum* val, mjtNum s1, mjtNum s2,
                                 int flg_bend, int flg_stretch, const mjtNum* Krot);

// flex stiffness (s1 + s2*damping)*K as a dof-level CSR in two passes: pass 1 (colind == NULL)
// fills rownnz (and rowadr, if given) and returns nnz; pass 2 fills colind (and val, if given)
// with sorted columns. With flg_general == 0 this is mjd_flexStiff_assemble: simple flexes only,
// as 3x3 vertex blocks. With flg_general == 1 every active flex (standard and, given Krot,
// interpolated) is pulled back through the sparse point Jacobians of its attachment bodies,
// J_u'*K_uw*J_w, so vertices and nodes may sit on arbitrary kinematic chains.
MJAPI int mjd_flexStiff_assembleGen(const mjModel* m, mjData* d, int* rownnz, int* rowadr,
                                    int* colind, mjtNum* val, mjtNum s1, mjtNum s2,
                                    int flg_bend, int flg_stretch, const mjtNum* Krot,
                                    int flg_general);

// can all interp flexes be assembled to dof-level CSR? (centered fast path everywhere)
MJAPI mjtBool mjd_flexInterpAssemblable(const mjModel* m);

// does any flex contribute assemblable implicit stiffness? (existence check)
MJAPI mjtBool mjd_flexStiff_any(const mjModel* m, int flg_interp);

//------------------------- helpers shared with engine_metric.c ------------------------------------

// per-actuator skip conditions shared by the qDeriv and discrete-metric assemblers:
// disabled, sleeping, or force-clamped actuators contribute no derivative
int mjd_actuatorDerivSkip(const mjModel* m, const mjData* d, int i, int sleep_filter);

// d(force)/d(length) of actuator i: all gain and bias types
mjtNum mjd_actuatorLenDeriv(const mjModel* m, const mjData* d, int i);

// d(force)/d(velocity) of actuator i: all gain and bias types
mjtNum mjd_actuatorVelDeriv(const mjModel* m, const mjData* d, int i);

// does any flex use the passive contact path? Distinct from elasticity: an empty CSR is valid for
// elastic models (matrix-free operators) but means "nothing" for a contact-only flex.
mjtBool mjd_flexPassiveContact_any(const mjModel* m);

// does this standard flex contribute implicit stiffness under the given term flags?
mjtBool mjd_flexStiff_active(const mjModel* m, int f, int flg_bend, int flg_stretch);
mjtBool mjd_flexStiff_activeGen(const mjModel* m, int f, int flg_bend, int flg_stretch,
                                int flg_general);

// mjd_flexStiff_assembleGen with flg_general == 1 in one pass: fills rownnz and rowadr (nv),
// allocates colind and val (nnz) from the arena and fills them; returns nnz
int mjd_flexStiff_assembleGenArena(const mjModel* m, mjData* d, int* rownnz, int* rowadr,
                                   int** colind, mjtNum** val, mjtNum s1, mjtNum s2,
                                   int flg_bend, int flg_stretch, const mjtNum* Krot);

// upper bound on the stack bytes of the pattern passes of the general assembly (no values)
size_t mjd_flexStiff_assembleGenStack(const mjModel* m, int flg_bend, int flg_stretch,
                                      int flg_interp);

// is this interpolated flex processed by mjd_flexInterp_mul?
mjtBool mjd_flexInterp_processed(const mjModel* m, int f);

// compute res += (s1 + s2*damping) * J'*K*J * vec for interpolated flexes in [first, last)
void mjd_flexInterp_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                             mjtNum s1, mjtNum s2, const mjtNum* K_rot_cache,
                             int first, int last);

// compute res += scale * K_bend * vec for standard (non-interp) flex bending, over the convex
// stencils
//   scale = s1 + s2 * flex_damping[f]  per flex
//   for stiffness+damping: s1=h^2, s2=h  =>  scale = h^2 + h*damping
//   for stiffness only:    s1=h,   s2=0  =>  scale = h
void mjd_flexBend_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                           mjtNum s1, mjtNum s2, int first, int last);

// compute res += (s1 + s2*flex_damping) * K_stretch * vec for standard flexes
// SNH uses its PSD-projected material Hessian; StVK keeps the material term
// and tensile geometric stiffness. For articulated attachments the pullback J'KJ
// omits derivatives of the attachment Jacobian.
void mjd_flexStretch_mulRange(const mjModel* m, mjData* d, mjtNum* res, const mjtNum* vec,
                              mjtNum s1, mjtNum s2, int first, int last);

// add d(fluid force)/d(velocity) of body bodyid to res (qDeriv's sparse layout), ellipsoid
// model; if flg_dragonly, only the dissipative drag terms (the variant the metric uses)
void mjd_ellipsoidFluid(const mjModel* m, mjData* d, mjtNum* res, int bodyid,
                        int flg_dragonly);

// add d(fluid force)/d(velocity) of body i to res (qDeriv's sparse layout), inertia-box model
void mjd_inertiaBoxFluid(const mjModel* m, mjData* d, mjtNum* res, int i);


#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_DERIVATIVE_H_
