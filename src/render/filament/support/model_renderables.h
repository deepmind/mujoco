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

#ifndef MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_RENDERABLES_H_
#define MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_RENDERABLES_H_

#include <array>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <math/vec4.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mesh_util.h"
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_resources.h"

namespace mujoco {

// Manages Renderable entities in an mjrfScene for an mjModel.
class ModelRenderables {
 public:
  // Populates the mjrScene with the renderables from the model.
  ModelRenderables(mjrfScene* scene, ModelResources* model_resources);
  ~ModelRenderables();

  // Updates the state of the renderables in the scene.
  void Update(const mjData* data);

  // Returns the renderable corresponding to the given model object.
  // For most objects, the sub_index is ignored. For slider-crank actuators,
  // sub_index=0 returns the slider, sub_index=1 returns the crank. For tendons,
  // sub_index is the tendon segment index. The number of segments can vary
  // between frames so is undefined. If obj_type == mjOBJ_BODY, returns the
  // geom or site associated with the body.
  mjrfRenderable* GetRenderable(mjtObj obj_type, int obj_index,
                                int sub_index = -1);

  // Returns the object type and index corresponding to the given
  // segmentation ID.
  std::pair<mjtObj, int> GetObjectFromSegmentationId(int segmentation_id) const;

  // Marks the given object as "selected", unmarking any previously selected
  // object.
  void MarkAsSelected(mjtObj obj_type, int obj_index);

  // Sets the visualization options on the renderables. The following options
  // are supported:
  // - shows/hides renderables based on the group visibility
  // - shows/hides renderables based on mjVIS_STATIC, mjVIS_SKIN,
  //   mjVIS_FLEXSKIN, mjVIS_TENDON, and mjVIS_ACTUATOR flags
  // - switches meshes to convex hull mode if mjVIS_CONVEXHULL is enabled
  void SetOptions(const mjvOption& opt);

  ModelRenderables(const ModelRenderables&) = delete;
  ModelRenderables& operator=(const ModelRenderables&) = delete;

 private:
  void AddGeomGeoms();
  void AddSiteGeoms();
  void AddFlexGeoms();
  void AddSkinGeoms();
  void AddSliderCrankGeoms();

  void UpdateQuality();
  void UpdateSpatialTendons(const mjData* data, int tendon_id);
  void AppendSegmentToTendon(int tendon_id);
  void RemoveSegmentFromTendon(int tendon_id);
  void UpdateSliderCranks(const mjData* data, int actuator_id);

  int GetSegmentationId(mjtObj obj_type, int obj_index);

  void SetVisibility(mjtObj obj_type, int idx, bool visible);

  mjrfMaterial GetMaterial(mjtObj obj_type, int obj_index,
                           const mjData* data = nullptr,
                           std::optional<int> matid_override = std::nullopt);

  mjrfScene* scene_;
  ModelResources* model_resources_;

  mjvOption vopts_;
  int numstacks_ = 0;
  int numslices_ = 0;
  int numquads_ = 0;
  mjtObj selected_obj_type_ = mjOBJ_UNKNOWN;
  int selected_obj_index_ = -1;

  std::vector<UniquePtr<mjrfRenderable>> geoms_;
  std::vector<UniquePtr<mjrfRenderable>> sites_;
  std::vector<UniquePtr<mjrfRenderable>> skins_;
  std::vector<std::vector<UniquePtr<mjrfRenderable>>> flexes_;
  std::vector<std::vector<FlexMeshSlice>> flex_slices_;
  // 1D flexes whose edge renderables currently hold box meshes (box-section rods)
  std::vector<bool> flex_box_mesh_;
  std::vector<std::vector<UniquePtr<mjrfRenderable>>> tendons_;
  std::unordered_map<int, std::array<UniquePtr<mjrfRenderable>, 2>>
      slider_cranks_;

  std::vector<UniquePtr<mjrfMesh>> flex_meshes_;
  std::vector<UniquePtr<mjrfMesh>> skin_meshes_;
  std::vector<filament::math::float4> point_cache_;
};
}  // namespace mujoco

#endif  // MUJOCO_SRC_RENDER_FILAMENT_SUPPORT_MODEL_RENDERABLES_H_
