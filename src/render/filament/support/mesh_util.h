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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MESH_UTIL_H_
#define MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MESH_UTIL_H_

#include <vector>
#include <math/vec4.h>
#include <mujoco/mujoco.h>
#include <mujoco/mjrfilament.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

struct FlexMeshSlice {
  int matid;
  int elem_offset;
  int elem_count;
};

// Calculates the per-material vertex buffer slices for a flex mesh.
std::vector<FlexMeshSlice> CalculateFlexMeshSlices(const mjModel* model,
                                                   int flex_id);

// Creates a mjrfMesh for the given flex object.
UniquePtr<mjrfMesh> CreateFlexMesh(mjrfContext* ctx, const mjModel* model,
                                   const mjData* data, int flex_id);

// Creates a mjrfMesh for the given skin object.
UniquePtr<mjrfMesh> CreateSkinMesh(mjrfContext* ctx, const mjModel* model,
                                   const mjData* data, int skin_id);

// Populates the set of points that define the given tendon. Points are
// added in pairs, representing the start and end of a segment of the tendon.
// The w-component of the point stores the width/radius of the tendon.
void GatherSpatialTendonPoints(const mjModel* model, const mjData* data,
                               int tendon_id,
                               std::vector<filament::math::float4>& points);

}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MESH_UTIL_H_
