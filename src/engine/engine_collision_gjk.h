// Copyright 2024 DeepMind Technologies Limited
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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_GJK_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_GJK_H_

#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#include "engine/engine_collision_convex.h"

#ifdef __cplusplus
extern "C" {
#endif

// numerical limits
#ifdef mjUSESINGLE
  #define mjMAX_LIMIT FLT_MAX
  // tolerance for normal alignment of two faces (~5.1 deg)
  #define mjFACE_TOL 0.996f
  // tolerance for edge-face alignment (~5.1 deg)
  #define mjEDGE_TOL 0.0888f
#else
  #define mjMAX_LIMIT DBL_MAX
  // tolerance for normal alignment of two faces (~5.1 deg)
  #define mjFACE_TOL 0.996
  // tolerance for edge-face alignment (~5.1 deg)
  #define mjEDGE_TOL 0.0888
#endif


// Status of an EPA run
typedef enum {
  mjEPA_NOCONTACT           = -1,
  mjEPA_SUCCESS             = 0,
  mjEPA_P2_INVALID_FACES,
  mjEPA_P2_NONCONVEX,
  mjEPA_P2_ORIGIN_ON_FACE,
  mjEPA_P3_BAD_NORMAL,
  mjEPA_P3_INVALID_V4,
  mjEPA_P3_INVALID_V5,
  mjEPA_P3_MISSING_ORIGIN,
  mjEPA_P3_ORIGIN_ON_FACE,
  mjEPA_P4_MISSING_ORIGIN,
} mjEPAStatus;

// configuration for convex collision detection
typedef struct {
  int max_iterations;  // the maximum number of iterations for GJK and EPA
  mjtNum tolerance;    // tolerance used by GJK and EPA
  int max_contacts;    // set to max number of contact points to recover
  mjtNum dist_cutoff;  // set to max geom distance to recover
  void* buffer;        // buffer memory for polytope (should be sized given by mjc_ccdSize)
  int npolygonmax;     // max number of vertices in a mesh polygon
  int nmeshdegmax;     // max number of edges adjacent to a mesh vertex
} mjCCDConfig;

// data produced from running GJK and EPA
typedef struct {
  // geom distance information
  int separated;                // set to true if geoms are verified to be separated
  mjtNum dist[mjMAXCONPAIR];    // distance between witness points
  mjtNum x1[3 * mjMAXCONPAIR];  // witness points for geom 1
  mjtNum x2[3 * mjMAXCONPAIR];  // witness points for geom 2
  int nx;                       // number of witness points

  // configurations used
  int max_iterations;           // the maximum number of iterations for GJK and EPA
  mjtNum tolerance;             // tolerance used by GJK and EPA
  int max_contacts;             // set to max number of contact points to recover
  mjtNum dist_cutoff;           // set to max geom distance to recover

  // statistics for debugging purposes
  int gjk_iterations;           // number of iterations that GJK ran
  int epa_iterations;           // number of iterations that EPA ran (zero if EPA did not run)
  mjEPAStatus epa_status;       // status of the EPA run
} mjCCDStatus;

// return size in bytes of the buffer needed for mjc_ccd for a given number of iterations
MJAPI size_t mjc_ccdSize(int npolygonmax, int nmeshdegmax, int iterations);

// run general convex collision detection, returns positive for distance, negative for penetration
MJAPI mjtNum mjc_ccd(const mjCCDConfig* config, mjCCDStatus* status, mjCCDObj* obj1, mjCCDObj* obj2);

// run warm-started GJK distance query (no EPA), returns positive distance or 0 if penetrating
MJAPI mjtNum mjc_gjk(mjCCDObj* obj1, mjCCDObj* obj2, mjtNum x1[3], mjtNum x2[3],
                     mjtNum dist_cutoff, mjtNum tolerance, int max_iterations);

// compute multiple contacts from GJK/EPA support vertices, assuming 1 initial contact in x1, x2
// return number of contacts
MJAPI int mjc_multicontact(mjCCDObj* obj1, mjCCDObj* obj2, mjtNum* x1, mjtNum* x2, mjtNum* dist,
                           int nconmax, int nmeshdegmax, int npolygonmax, uint8_t* buffer);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_GJK_H_
