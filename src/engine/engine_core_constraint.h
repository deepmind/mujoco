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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_CORE_CONSTRAINT_H_
#define MUJOCO_SRC_ENGINE_ENGINE_CORE_CONSTRAINT_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif


//-------------------------- Jacobian-related ------------------------------------------------------

// multiply Jacobian by vector
MJAPI void mj_mulJacVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// multiply JacobianT by vector
MJAPI void mj_mulJacTVec(const mjModel* m, const mjData* d, mjtNum* res, const mjtNum* vec);

// subtract Jdot*v correction from result vector
MJAPI void mj_Jdotv(const mjModel* m, mjData* d, mjtNum* result);


//-------------------------- utility functions -----------------------------------------------------

// assign/override solver reference parameters
void mj_assignRef(const mjModel* m, mjtNum* target, const mjtNum* source);

// assign/override solver impedance parameters
void mj_assignImp(const mjModel* m, mjtNum* target, const mjtNum* source);

// assign/clamp contact friction parameters
void mj_assignFriction(const mjModel* m, mjtNum* target, const mjtNum* source);

// assign/override geom margin
mjtNum mj_assignMargin(const mjModel* m, mjtNum source);

// compute contact condim, solref, solimp, friction, adhesion
void mj_contactParam(const mjModel* m, int* condim, mjtNum* solref, mjtNum* solimp,
                     mjtNum* friction, mjtNum* adhesion, int g1, int g2, int f1, int f2);

// add contact to d->contact list; return 0 if success; 1 if buffer full
MJAPI int mj_addContact(const mjModel* m, mjData* d, const mjContact* con);

//-------------------------- constraint instantiation ----------------------------------------------

// equality constraints
void mj_instantiateEquality(const mjModel* m, mjData* d);

// frictionless and frictional contacts
void mj_instantiateContact(const mjModel* m, mjData* d);

// compute Jacobian for contact, return number of DOFs affected
//
// Arguments:
//   m:       model structure (mjModel)
//   d:       data structure (mjData)
//   con:     contact information
//   dim:     contact dimension (1 to 6)
//   jacdifp: translational Jacobian difference [3 * (NV or nv)]
//   jacdifr: rotational Jacobian difference    [3 * (NV or nv)], nullable, unused if dim <= 3
//   jac1p:   translational Jacobian for body 1 [3 * (NV or nv)], nullable
//   jac2p:   translational Jacobian for body 2 [3 * (NV or nv)], nullable
//   jac1r:   rotational Jacobian for body 1    [3 * (NV or nv)], nullable, unused if dim <= 3
//   jac2r:   rotational Jacobian for body 2    [3 * (NV or nv)], nullable, unused if dim <= 3
//   chain:   list of DOF indices affecting contact [NV], unused if dense
//
// Returns:
//   number of DOFs affected (NV for sparse, nv for dense), 0 if contact affects no DOFs
MJAPI int mj_contactJacobian(const mjModel* m, mjData* d, const mjContact* con, int dim,
                             mjtNum* jacdifp, mjtNum* jacdifr,
                             mjtNum* jac1p, mjtNum* jac2p,
                             mjtNum* jac1r, mjtNum* jac2r, int* chain);


//------------------------ parameter computation/extraction ----------------------------------------

// compute efc_diagA
void mj_diagApprox(const mjModel* m, mjData* d);

// compute efc_R, efc_D, efc_KDIP, adjust diagA
void mj_makeImpedance(const mjModel* m, mjData* d);


//---------------------------- top-level API for constraint construction ---------------------------

// main driver: call all functions above
MJAPI void mj_makeConstraint(const mjModel* m, mjData* d);

// compute efc_AR
MJAPI void mj_projectConstraint(const mjModel* m, mjData* d);

// compute efc_vel
void mj_velocityConstraint(const mjModel* m, mjData* d);

// compute efc_vel, efc_aref
MJAPI void mj_referenceConstraint(const mjModel* m, mjData* d);

// compute efc_diagA (approximate or exact), efc_R, efc_D, efc_KBIP in the solve metric
void mj_regularizeConstraint(const mjModel* m, mjData* d, int flg_AR);

// compute efc_state, efc_force
//  optional: cost(qacc) = s_hat(jar); cone Hessians
MJAPI void mj_constraintUpdate_impl(int ne, int nf, int nefc,
                                    const mjtNum* D, const mjtNum* R, const mjtNum* floss,
                                    const mjtNum* jar, const int* type, const int* id,
                                    mjContact* contact, int* state, mjtNum* force, mjtNum cost[1],
                                    int flg_coneHessian);

// compute efc_state, efc_force, qfrc_constraint
// optional: cost(qacc) = s_hat(jar) where jar = Jac*qacc-aref; cone Hessians
MJAPI void mj_constraintUpdate(const mjModel* m, mjData* d, const mjtNum* jar,
                               mjtNum cost[1], int flg_coneHessian);


#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_CORE_CONSTRAINT_H_
