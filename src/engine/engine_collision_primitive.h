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

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_PRIMITIVE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_PRIMITIVE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>

#ifdef __cplusplus
extern "C" {
#endif

// compute the barycentric weights w of the closest point on triangle ABC to p
void mjc_triWeights(mjtNum w[3], const mjtNum p[3], const mjtNum a[3], const mjtNum b[3],
                    const mjtNum c[3]);

// compute the barycentric weights w = {s1, t1, s2, t2} in [0, 1] of the closest points
// A + s*(B - A) on segment AB and C + t*(D - C) on segment CD;
// return the number of closest point pairs (1, or 2 if parallel)
int mjc_segWeights(mjtNum w[4], const mjtNum a[3], const mjtNum b[3], const mjtNum c[3],
                   const mjtNum d[3]);


// raw collision functions (called by mjc_XXX)
int mjraw_SphereCapsule(mjPreContact* con, mjtNum margin, const mjtNum* pos1, const mjtNum* mat1,
                        const mjtNum* size1, const mjtNum* pos2, const mjtNum* mat2,
                        const mjtNum* size2);
int mjraw_CapsuleCapsule(mjPreContact* con, mjtNum margin, const mjtNum* pos1, const mjtNum* mat1,
                         const mjtNum* size1, const mjtNum* pos2, const mjtNum* mat2,
                         const mjtNum* size2);
int mjraw_CapsuleBox(mjPreContact* con, mjtNum margin, const mjtNum* pos1, const mjtNum* mat1,
                     const mjtNum* size1, const mjtNum* pos2, const mjtNum* mat2,
                     const mjtNum* size2);
int mjraw_SphereBox(mjPreContact* con, mjtNum margin, const mjtNum* pos1, const mjtNum* mat1,
                    const mjtNum* size1, const mjtNum* pos2, const mjtNum* mat2,
                    const mjtNum* size2);
int mjraw_BoxBox(mjPreContact* con, mjtNum margin, const mjtNum* pos1, const mjtNum* mat1,
                 const mjtNum* size1, const mjtNum* pos2, const mjtNum* mat2,
                 const mjtNum* size2);

// plane collisions
MJAPI int mjc_PlaneSphere(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                          mjtNum margin);
MJAPI int mjc_PlaneCapsule(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                           mjtNum margin);
MJAPI int mjc_PlaneCylinder(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                            mjtNum margin);
MJAPI int mjc_PlaneBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                       mjtNum margin);

// sphere and capsule collisions
MJAPI int mjc_SphereSphere(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                            mjtNum margin);
MJAPI int mjc_SphereCapsule(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                            mjtNum margin);
MJAPI int mjc_SphereCylinder(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                             mjtNum margin);
MJAPI int mjc_CapsuleCapsule(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                             mjtNum margin);

// box collisions: from engine_collision_box.c
MJAPI int mjc_CapsuleBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                         mjtNum margin);
MJAPI int mjc_SphereBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                        mjtNum margin);
MJAPI int mjc_BoxBox(const mjModel* m, mjData* d, mjPreContact* con, int g1, int g2,
                     mjtNum margin);

#ifdef __cplusplus
}
#endif
#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_PRIMITIVE_H_
