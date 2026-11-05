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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_FLEX_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_FLEX_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// test a plane geom and a flex for collision, return number of contacts
int mjc_PlaneFlex(const mjModel* m, mjData* d, mjPreContact* con, int* vert, int g, int f,
                  mjtNum margin);

// test a geom and an elem for collision, return number of contacts
int mjc_GeomElem(const mjModel* m, mjData* d, mjPreContact* con, int g, int f, int e,
                 mjtNum margin);

// range [lo, hi] of the interpolation grid cells containing vertex v of flex f,
// per axis
void mjc_vertCellRange(const mjModel* m, int f, int v, int lo[3], int hi[3]);

// return 1 if cell ranges [lo1, hi1] and [lo2, hi2] are within distance 1 along
// every axis
int mjc_cellRangeShared(const int lo1[3], const int hi1[3], const int lo2[3],
                        const int hi2[3]);

// return 1 if elements e1 and e2 of flex f are driven by a common node of its
// interpolation grid
int mjc_ElemSharedNode(const mjModel* m, int f, int e1, int e2);

// test two elems for collision, return number of contacts
int mjc_ElemElem(const mjModel* m, mjData* d, mjPreContact* con, int f1, int e1,
                 int f2, int e2, mjtNum margin);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_FLEX_H_
