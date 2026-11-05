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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_SETCONST_H_
#define MUJOCO_SRC_ENGINE_ENGINE_SETCONST_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

// Set constant fields of mjModel, corresponding to qpos0 configuration.
MJAPI void mj_setConst(mjModel* m, mjData* d);

// Groups of mj_setConst computations, in the order of the columns of MJSETCONST_ARRAYS.
#define mjSCGROUP_FIXED  (1<<0)  // independent of the configuration
#define mjSCGROUP_KIN0   (1<<1)  // kinematics in qpos0
#define mjSCGROUP_INERT0 (1<<2)  // inertia matrix in qpos0
#define mjSCGROUP_SPRING (1<<3)  // kinematics in qpos_spring
#define mjSCGROUP_ALL    (mjSCGROUP_FIXED | mjSCGROUP_KIN0 | mjSCGROUP_INERT0 | mjSCGROUP_SPRING)

// Run the groups of mj_setConst computations selected by the bits of groups, in order.
MJAPI void mj_setConstGroups(mjModel* m, mjData* d, int groups);

// Set constant fields of mjModel, corresponding to qpos0 configuration, and those which
// compilation derives once from the conventions of the spec: the damping of position-like
// actuators given a damping ratio as a positive biasprm[2], the spring length of tendons given
// (-1, -1), and the model statistics.
void mj_setCompiledConst(mjModel* m, mjData* d);

// Set actuator_lengthrange for specified actuator; return 1 if ok, 0 if error.
MJAPI int mj_setLengthRange(mjModel* m, mjData* d, int index,
                            const mjLROpt* opt, char* error, int error_sz);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_SETCONST_H_
