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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
#define MUJOCO_SRC_ENGINE_ENGINE_IPC_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include "engine/engine_collision_continuous.h"

#ifdef __cplusplus
extern "C" {
#endif

// IPC contact mode of the discrete integrator (flag ipc): owns the full step. Minimizes the
// incremental potential with penetration-free contact by a barrier-free augmented-Lagrangian
// method; flex self-contact (vertex-triangle, edge-edge) and flex-vs-geom. Elasticity rides the
// native efc rows (edge equality) and the effective metric (elastic2d).
void mj_ipc(const mjModel* m, mjData* d);

// number of distinct pairs among cand and an optional old active set (nold pairs in old, or
// nold=0/old=NULL for empty old set); allocates from the arena (for testing only)
MJAPI int mj_ipcCountDistinct(mjData* d, const mjcFlexPair* cand, int ncand,
                              const mjcFlexPair* old, int nold);


#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_IPC_H_
