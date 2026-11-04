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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_FORWARD_H_
#define MUJOCO_SRC_ENGINE_ENGINE_FORWARD_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif
// check positions, velocities, accelerations; reset if bad
MJAPI void mj_checkPos(const mjModel* m, mjData* d);
MJAPI void mj_checkVel(const mjModel* m, mjData* d);
MJAPI void mj_checkAcc(const mjModel* m, mjData* d);


//-------------------------------- top-level API ---------------------------------------------------

// advance simulation: use control callback, no external force, RK4 available
MJAPI void mj_step(const mjModel* m, mjData* d);

// advance simulation in two steps: before external force/control is set by user
MJAPI void mj_step1(const mjModel* m, mjData* d);

// advance simulation in two steps: after external force/control is set by user
MJAPI void mj_step2(const mjModel* m, mjData* d);

// forward dynamics
MJAPI void mj_forward(const mjModel* m, mjData* d);

// forward dynamics with skip; skipstage is mjtStage
MJAPI void mj_forwardSkip(const mjModel* m, mjData* d, int skipstage, int skipsensor);


//-------------------------------- integrators -----------------------------------------------------

// shared tail of every velocity-level integrator: history buffers, activations, sleep,
// velocity/position integration, time, plugin states, qacc warmstart save (see engine_forward.c
// for the arguments)
void mj_advance(const mjModel* m, mjData* d,
                const mjtNum* act_dot, const mjtNum* qacc, const mjtNum* qvel);

// the same tail for an integrator that solved the step at the position level: the effective
// acceleration, then history buffers, activations and sleep at the pre-step state, then the
// verified endpoint and its velocity replace the state, then time and plugin states. The warm
// start is left to the caller (see engine_forward.c)
void mj_commit(const mjModel* m, mjData* d, const mjtNum* act_dot,
               const mjtNum* qpos, const mjtNum* qvel, const mjtNum* qacc);

// Runge Kutta explicit order-N integrator
MJAPI void mj_RungeKutta(const mjModel* m, mjData* d, int N);

// Euler integrator, semi-implicit in velocity
MJAPI void mj_Euler(const mjModel* m, mjData* d);

// Euler integrator, semi-implicit in velocity, possibly skipping factorisation
MJAPI void mj_EulerSkip(const mjModel* m, mjData* d, int skipfactor);

// fully implicit in velocity
MJAPI void mj_implicit(const mjModel *m, mjData *d);

// fully implicit in velocity, possibly skipping factorization
MJAPI void mj_implicitSkip(const mjModel *m, mjData *d, int skipfactor);

// discrete integrator: the solver's qacc is the step map, advance directly
void mj_discrete(const mjModel* m, mjData* d);

// runtime option validation for the discrete integrator: mjERROR on unsupported combinations
void mj_checkDiscrete(const mjModel* m);

//-------------------------------- solver components -----------------------------------------------

// all kinematics-like computations
MJAPI void mj_fwdKinematics(const mjModel* m, mjData* d);

// computations that depend only on qpos
MJAPI void mj_fwdPosition(const mjModel* m, mjData* d);

// computations that depend only on qpos and qvel
MJAPI void mj_fwdVelocity(const mjModel* m, mjData* d);

// compute actuator force
MJAPI void mj_fwdActuation(const mjModel* m, mjData* d);

// add up all non-constraint forces, compute qacc_smooth
MJAPI void mj_fwdAcceleration(const mjModel* m, mjData* d);

// forward constraint
MJAPI void mj_fwdConstraint(const mjModel* m, mjData* d);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_FORWARD_H_
