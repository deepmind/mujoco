// Copyright 2026 DeepMind Technologies Limited
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

#ifndef MUJOCO_MJBATCH_H_
#define MUJOCO_MJBATCH_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

//---------------------------------- Batched simulation --------------------------------------------

// many simulations of one model on a thread pool
typedef struct mjBatch_ mjBatch;

// function run on one simulation by mjb_apply
typedef void (*mjfBatchFunc)(const mjModel* m, mjData* d, int sim, void* arg);

typedef struct mjBatchConfig_ {  // parameters of mjb_makeBatch
  int nthread;                   // threads including the caller's; <= 0: one per logical CPU
  mjtBool persistent;            // one mjData per simulation instead of one per thread
} mjBatchConfig;

typedef struct mjBatchRecord_ {  // mjData field or state recorded by mjb_rollout
  const char* field;             // mjData field (as for mjb_output), NULL: record a state
  int spec;                      // mjtState bits of the recorded state
  void* buf;                     // output (n x nstep x size), of the field's type
} mjBatchRecord;

// Set default batch parameters: one thread per logical CPU, one mjData per thread.
MJAPI void mjb_defaultConfig(mjBatchConfig* config);

// Allocate a batch of nsim simulations of a copy of m; return NULL and write error on failure.
// Nullable: config, error
MJAPI mjBatch* mjb_makeBatch(const mjModel* m, int nsim, const mjBatchConfig* config,
                             char* error, int error_sz);

// Free a batch.
MJAPI void mjb_deleteBatch(mjBatch* b);

// Return size of a state row: mj_stateSize(m, mjSTATE_INTEGRATION).
MJAPI int mjb_nstate(const mjBatch* b);

// Return state rows (nsim x nstate), in mj_getState order.
MJAPI mjtNum* mjb_state(mjBatch* b);

// Return address of a state component (one mjtState bit) in the first row; rows are nstate apart.
// Nullable: size
MJAPI mjtNum* mjb_stateComponent(mjBatch* b, int component, int* size);

// Return warning counters (nsim x mjNWARNING).
MJAPI mjWarningStat* mjb_warning(mjBatch* b);

// Advance simulations by nstep calls to mj_step, then compute stages up to stage (mjtStage);
// return number of failed simulations.
// Nullable: ids
MJAPI int mjb_step(mjBatch* b, const int* ids, int nid, int nstep, int stage);

// Compute stages up to stage (mjtStage) on simulations; return number of failed simulations.
// Nullable: ids
MJAPI int mjb_forward(mjBatch* b, const int* ids, int nid, int stage);

// Reset simulations to keyframes (key, per simulation; -1 or NULL: defaults), then compute stages;
// return number failed.
// Nullable: ids, key
MJAPI int mjb_reset(mjBatch* b, const int* ids, int nid, const int* key, int stage);

// Return status of each simulation's last call (nsim): nonzero if it raised an error.
MJAPI const int* mjb_status(const mjBatch* b);

// Return error message of simulation's last call, or "" if none.
MJAPI const char* mjb_error(const mjBatch* b, int sim);

// Copy mjData field out after every call; return its storage (nsim x size), NULL if invalid.
// Nullable: size, elemsize
MJAPI void* mjb_output(mjBatch* b, const char* name, int* size, int* elemsize);

// Give mjModel field per-simulation values; return its storage (nsim x size), NULL if invalid.
// Nullable: size, elemsize
MJAPI void* mjb_expand(mjBatch* b, const char* name, int* size, int* elemsize);

// Return storage (nsim x size) of mjModel field if expanded, without expanding it; NULL if not.
// Nullable: size, elemsize
MJAPI const void* mjb_expanded(mjBatch* b, const char* name, int* size, int* elemsize);

// Run mj_setConst on each simulation's model; return number of failed simulations.
// Nullable: ids
MJAPI int mjb_setConst(mjBatch* b, const int* ids, int nid);

// Run simulation i on models[i], the caller's, instead of the batch's model; NULL: the batch's.
// Nullable: models
MJAPI void mjb_setModels(mjBatch* b, const mjModel* const* models);

// Advance simulations with per-substep controls and records; return number failed.
// Nullable: ids, control, record
MJAPI int mjb_rollout(mjBatch* b, const int* ids, int nid, int nstep, const mjtNum* control,
                      int control_spec, const mjBatchRecord* record, int nrecord);

// Run func on each simulation, saving the state if save is nonzero; return number failed.
// Nullable: ids, arg
MJAPI int mjb_apply(mjBatch* b, const int* ids, int nid, mjfBatchFunc func, void* arg,
                    int save);

// Write simulation sim's model into m, which has the batch's sizes, such as a copy of mjb_model.
MJAPI void mjb_getModel(mjBatch* b, int sim, mjModel* m);

// Write simulation sim's state into d, which has the batch's sizes; persistent: its whole mjData.
MJAPI void mjb_getData(mjBatch* b, int sim, mjData* d);

// Return number of simulations.
MJAPI int mjb_nsim(const mjBatch* b);

// Return number of threads, including the caller's.
MJAPI int mjb_nthread(const mjBatch* b);

// Return 1 if the batch keeps one mjData per simulation, 0 otherwise.
MJAPI int mjb_persistent(const mjBatch* b);

// Return the batch's copy of the model.
MJAPI const mjModel* mjb_model(const mjBatch* b);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MJBATCH_H_
