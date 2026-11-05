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

#include "render/filament/support/model_renderables.h"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include <math/TVecHelpers.h>
#include <math/mat3.h>
#include <math/quat.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "engine/engine_core_util.h"
#include "engine/engine_sleep.h"
#include "render/filament/support/filament_util.h"
#include "render/filament/support/mesh_util.h"
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_resources.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;
using filament::math::mat3f;
using filament::math::quatf;

// Converts an array of numbers into an array of floats.
template <typename T>
void xtof(float* dst, const T* src, int n) {
  for (int i = 0; i < n; ++i) {
    dst[i] = static_cast<float>(src[i]);
  }
}

static mjtCatBit GetBodyCategory(const mjModel* m, int bodyid) {
  // mocap subtrees are their own weld, hence not static
  if (m->body_weldid[bodyid] == 0) {
    return mjCAT_STATIC;
  } else {
    return mjCAT_DYNAMIC;
  }
}

static void Connect(mjrfRenderable* renderable, const float3 p0,
                    const float3 p1, const float width) {
  const float3 vec = p1 - p0;
  const float len = length(vec);
  const float3 pos = (p0 + p1) * 0.5f;
  const float3 size = {width, width, 0.5f * len};
  const quatf rot =
      quatf::fromDirectedRotation(normalize(vec), float3{0, 0, 1});
  const mat3f mat = mat3f(rot);
  mjrf_setRenderablePose(renderable, pos.v, mat.asArray());
  mjrf_setRenderableSize(renderable, size.v);
}

static bool GetSize(const mjModel* model, mjtGeom type, const mjtNum* size,
                    float out[3]) {
  out[0] = size[0];
  out[1] = size[1];
  out[2] = size[2];
  switch (type) {
    case mjGEOM_SPHERE:
      out[2] = size[0];
      out[1] = size[0];
      return true;
    case mjGEOM_CAPSULE:
    case mjGEOM_CYLINDER:
      out[2] = size[1];
      out[1] = size[0];
      return true;
    case mjGEOM_TRIANGLE:
    case mjGEOM_PLANE:
      out[2] = 1.0f;
      return true;
    case mjGEOM_ELLIPSOID:
    case mjGEOM_BOX:
    case mjGEOM_ARROW:
    case mjGEOM_ARROW1:
    case mjGEOM_ARROW2:
    case mjGEOM_LINE:
    case mjGEOM_LINEBOX:
      return true;
    case mjGEOM_MESH:
    case mjGEOM_SDF:
    case mjGEOM_HFIELD:
    case mjGEOM_FLEX:
    case mjGEOM_SKIN:
    case mjGEOM_NONE:
    case mjGEOM_LABEL:
      return false;
    case mjNGEOMTYPES:
      mju_warning("Unsupported geom type: %d", type);
      return false;
  }
}

static void SetGeomMesh(mjrfRenderable* renderable, ModelResources* model_objs,
                        mjtGeom type, int data_id = -1) {
  const mjModel* model = model_objs->GetModel();
  const int nstack = model->vis.quality.numstacks;
  const int nslice = model->vis.quality.numslices;
  const int nquad = model->vis.quality.numquads;

  switch (type) {
    case mjGEOM_MESH:
    case mjGEOM_SDF: {
      const int mesh_id = data_id * 2;
      mjrf_setRenderableMesh(renderable, model_objs->GetMesh(mesh_id), 0, 0);
      break;
    }
    case mjGEOM_HFIELD: {
      mjrf_setRenderableMesh(renderable, model_objs->GetHeightField(data_id), 0,
                             0);
      break;
    }
    case mjGEOM_FLEX:
      mju_error("Flex meshes should be handled separately.");
      break;
    case mjGEOM_SKIN:
      mju_error("Skin meshes should be handled separately.");
      break;
    case mjGEOM_PLANE:
    case mjGEOM_SPHERE:
    case mjGEOM_ELLIPSOID:
    case mjGEOM_BOX:
    case mjGEOM_CAPSULE:
    case mjGEOM_CYLINDER:
    case mjGEOM_ARROW:
    case mjGEOM_ARROW1:
    case mjGEOM_ARROW2:
    case mjGEOM_LINE:
    case mjGEOM_LINEBOX:
    case mjGEOM_TRIANGLE:
      mjrf_setRenderableGeomMesh(renderable, type, nstack, nslice, nquad);
      break;
    case mjGEOM_NONE:
    case mjGEOM_LABEL:
      // Do nothing.
      break;
    case mjNGEOMTYPES:
      mju_warning("Unsupported geom type: %d", type);
      break;
  }
}

ModelRenderables::ModelRenderables(mjrfScene* scene,
                                   ModelResources* model_resources)
    : scene_(scene), model_resources_(model_resources) {
  mjv_defaultOption(&vopts_);

  const mjModel* model = model_resources_->GetModel();
  numstacks_ = model->vis.quality.numstacks;
  numslices_ = model->vis.quality.numslices;
  numquads_ = model->vis.quality.numquads;

  AddGeomGeoms();
  AddSiteGeoms();
  AddFlexGeoms();
  AddSkinGeoms();
  AddSliderCrankGeoms();
  // A tendon is composed of a collection of renderables which are managed
  // every frame. For now, all we do is reserve space for the collections.
  tendons_.resize(model->ntendon);
}

ModelRenderables::~ModelRenderables() {
  for (auto& renderable : geoms_) {
    mjrf_removeSceneRenderable(scene_, renderable.get());
  }
  for (auto& renderable : sites_) {
    mjrf_removeSceneRenderable(scene_, renderable.get());
  }
  for (auto& parts : flexes_) {
    for (auto& renderable : parts) {
      mjrf_removeSceneRenderable(scene_, renderable.get());
    }
  }
  for (auto& renderable : skins_) {
    mjrf_removeSceneRenderable(scene_, renderable.get());
  }
  for (auto& segments : tendons_) {
    for (auto& renderable : segments) {
      mjrf_removeSceneRenderable(scene_, renderable.get());
    }
  }
  for (auto& [_, renderables] : slider_cranks_) {
    for (auto& renderable : renderables) {
      mjrf_removeSceneRenderable(scene_, renderable.get());
    }
  }
}

void ModelRenderables::UpdateQuality() {
  const mjModel* model = model_resources_->GetModel();
  const int nstack = model->vis.quality.numstacks;
  const int nslice = model->vis.quality.numslices;
  const int nquad = model->vis.quality.numquads;
  if (nstack == numstacks_ && nslice == numslices_ && nquad == numquads_) {
    return;
  }

  if (nquad != numquads_) {
    for (int i = 0; i < model->nhfield; ++i) {
      model_resources_->UploadHeightField(model, i);
    }
  }

  numstacks_ = nstack;
  numslices_ = nslice;
  numquads_ = nquad;

  for (int i = 0; i < model->ngeom; ++i) {
    const mjtGeom type = (mjtGeom)model->geom_type[i];
    if (type != mjGEOM_MESH && type != mjGEOM_SDF) {
      SetGeomMesh(geoms_[i].get(), model_resources_, type, model->geom_dataid[i]);
    }
  }
  for (int i = 0; i < model->nsite; ++i) {
    const mjtGeom type = (mjtGeom)model->site_type[i];
    if (type != mjGEOM_MESH && type != mjGEOM_SDF) {
      SetGeomMesh(sites_[i].get(), model_resources_, type, model->site_dataid[i]);
    }
  }
  for (auto& [_, pair] : slider_cranks_) {
    mjrf_setRenderableGeomMesh(pair[0].get(), mjGEOM_CYLINDER, nstack, nslice,
                               nquad);
    mjrf_setRenderableGeomMesh(pair[1].get(), mjGEOM_CAPSULE, nstack, nslice,
                               nquad);
  }
}

void ModelRenderables::Update(const mjData* data) {
  UpdateQuality();

  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  for (int i = 0; i < model->ngeom; ++i) {
    const float3 pos = ReadFloat3(data->geom_xpos, i);
    const mat3f mat = ReadMat3(data->geom_xmat, i);
    mjrf_setRenderablePose(geoms_[i].get(), pos.v, mat.asArray());

    mjrfMaterial material = GetMaterial(mjOBJ_GEOM, i, data);
    mjrf_setRenderableMaterial(geoms_[i].get(), &material);
    mjrf_setRenderableCastShadow(geoms_[i].get(), model->geom_castshadow[i]);
  }

  for (int i = 0; i < model->nsite; ++i) {
    float3 pos = ReadFloat3(data->site_xpos, i);
    mat3f mat = ReadMat3(data->site_xmat, i);
    if (model->site_type[i] == mjGEOM_MESH && model->site_dataid[i] >= 0) {
      int meshid = model->site_dataid[i];
      mjtNum meshpos[3], meshmat[9], meshrot[9];
      mju_mulMatVec3(meshpos, data->site_xmat + 9 * i, model->mesh_pos + 3 * meshid);
      mju_addTo3(meshpos, data->site_xpos + 3 * i);
      mju_quat2Mat(meshrot, model->mesh_quat + 4 * meshid);
      mju_mulMatMat(meshmat, data->site_xmat + 9 * i, meshrot, 3, 3, 3);
      pos = ReadFloat3(meshpos);
      mat = ReadMat3(meshmat);
    }
    mjrf_setRenderablePose(sites_[i].get(), pos.v, mat.asArray());

    mjrfMaterial material = GetMaterial(mjOBJ_SITE, i, data);
    mjrf_setRenderableMaterial(sites_[i].get(), &material);
  }

  if (vopts_.flags[mjVIS_FLEXSKIN]) {
    for (int i = 0; i < model->nflex; ++i) {
      const int dim = model->flex_dim[i];
      if (dim > 1) {
        auto mesh = CreateFlexMesh(ctx, model, data, i);
        const auto& slices = flex_slices_[i];
        for (size_t k = 0; k < slices.size() && k < flexes_[i].size(); ++k) {
          if (slices[k].elem_count > 0) {
            mjrf_setRenderableMesh(flexes_[i][k].get(), mesh.get(),
                                   slices[k].elem_offset, slices[k].elem_count);
          }
          mjrfMaterial material =
              GetMaterial(mjOBJ_FLEX, i, data, slices[k].matid);
          mjrf_setRenderableMaterial(flexes_[i][k].get(), &material);
        }
        flex_meshes_[i] = std::move(mesh);
      } else {
        const int vertadr = model->flex_vertadr[i];
        const int edgeadr = model->flex_edgeadr[i];
        const int edgenum = model->flex_edgenum[i];
        const int elemadr = model->flex_elemadr[i];
        const int elemnum = model->flex_elemnum[i];
        float radius = model->flex_radius[i];
        mjtNum box[2];
        std::vector<mjtNum> elem_mat;
        const bool box_rod = mj_rodBoxSize(model, i, box) && elemnum > 0;
        bool is_box = false;
        if (box_rod) {
          elem_mat.resize(9 * elemnum);
          is_box = mj_rodBoxEdge(model, data, i, elem_mat.data(), box);
          // fallback capsule of a box rod: the minor half-size, as in collisions
          radius = static_cast<float>(box[1]);
        }
        auto& edges = flexes_[i];

        // switch the edge meshes when the box frames become (un)available
        if (is_box != flex_box_mesh_[i]) {
          for (auto& edge : edges) {
            mjrf_setRenderableGeomMesh(edge.get(),
                                       is_box ? mjGEOM_BOX : mjGEOM_CAPSULE,
                                       is_box ? 1 : 4, is_box ? 1 : 8, 1);
          }
          flex_box_mesh_[i] = is_box;
        }

        // element of each edge: flex_edge need not follow the element order
        const int* elem = model->flex_elem + model->flex_elemdataadr[i];
        auto edge_elem = [&](int j, int idx0, int idx1) {
          auto match = [&](int k) {
            return k >= 0 && k < elemnum &&
                   ((elem[2 * k] == idx0 && elem[2 * k + 1] == idx1) ||
                    (elem[2 * k] == idx1 && elem[2 * k + 1] == idx0));
          };
          if (match(j)) return j;
          for (int k = 0; k < elemnum; ++k) {
            if (match(k)) return k;
          }
          return -1;
        };

        for (int j = 0; j < edgenum; ++j) {
          const int e = j + edgeadr;
          const int idx0 = model->flex_edge[2 * e];
          const int idx1 = model->flex_edge[2 * e + 1];
          const float3 v0 = ReadFloat3(data->flexvert_xpos, vertadr + idx0);
          const float3 v1 = ReadFloat3(data->flexvert_xpos, vertadr + idx1);
          const int k = edge_elem(j, idx0, idx1);
          if (is_box && k >= 0) {
            const float len = length(v1 - v0);
            const float3 pos = (v0 + v1) * 0.5f;
            const float3 size = {static_cast<float>(box[0]),
                                 static_cast<float>(box[1]), 0.5f * len};
            const mat3f mat = ReadMat3(elem_mat.data(), k);
            mjrf_setRenderablePose(edges[j].get(), pos.v, mat.asArray());
            mjrf_setRenderableSize(edges[j].get(), size.v);
          } else {
            Connect(edges[j].get(), v0, v1, radius);
          }
          const int matid =
              (k >= 0) ? model->flex_matid[elemadr + k]
                       : (elemnum > 0 ? model->flex_matid[elemadr] : -1);
          mjrfMaterial material = GetMaterial(mjOBJ_FLEX, i, data, matid);
          mjrf_setRenderableMaterial(flexes_[i][j].get(), &material);
        }
      }
    }
  }

  for (int i = 0; i < model->nskin; ++i) {
    auto mesh = CreateSkinMesh(ctx, model, data, i);
    mjrf_setRenderableMesh(skins_[i].get(), mesh.get(), 0, 0);
    skin_meshes_[i] = std::move(mesh);

    mjrfMaterial material = GetMaterial(mjOBJ_SKIN, i, data);
    mjrf_setRenderableMaterial(skins_[i].get(), &material);
  }

  for (int i = 0; i < model->ntendon; i++) {
    UpdateSpatialTendons(data, i);
  }

  for (int i = 0; i < model->nactuator; i++) {
    if (model->actuator_trntype[i] == mjTRN_SLIDERCRANK) {
      UpdateSliderCranks(data, i);
    }
  }
}

mjrfRenderable* ModelRenderables::GetRenderable(mjtObj obj_type, int obj_index,
                                                int sub_index) {
  switch (obj_type) {
    case mjOBJ_GEOM:
      if (obj_index >= 0 && obj_index < geoms_.size()) {
        return geoms_[obj_index].get();
      }
      break;
    case mjOBJ_SITE:
      if (obj_index >= 0 && obj_index < sites_.size()) {
        return sites_[obj_index].get();
      }
      break;
    case mjOBJ_FLEX:
      if (obj_index >= 0 && obj_index < flexes_.size()) {
        auto& parts = flexes_[obj_index];
        if (sub_index >= 0 && sub_index < parts.size()) {
          return parts[sub_index].get();
        }
      }
      break;
    case mjOBJ_SKIN:
      if (obj_index >= 0 && obj_index < skins_.size()) {
        return skins_[obj_index].get();
      }
      break;
    case mjOBJ_TENDON:
      if (obj_index >= 0 && obj_index < tendons_.size()) {
        auto& segments = tendons_[obj_index];
        if (sub_index >= 0 && sub_index < segments.size()) {
          return segments[sub_index].get();
        }
      }
      break;
    case mjOBJ_ACTUATOR:
      if (auto it = slider_cranks_.find(obj_index);
          it != slider_cranks_.end()) {
        if (sub_index >= 0 && sub_index < it->second.size()) {
          return it->second[sub_index].get();
        }
      }
      break;
    case mjOBJ_BODY: {
      const mjModel* model = model_resources_->GetModel();
      for (int i = 0; i < model->ngeom; ++i) {
        if (model->geom_bodyid[i] == obj_index) {
          return geoms_[i].get();
        }
      }
      for (int i = 0; i < model->nsite; ++i) {
        if (model->site_bodyid[i] == obj_index) {
          return sites_[i].get();
        }
      }
      break;
    }
    default:
      break;
  }
  return nullptr;
}

void ModelRenderables::AddGeomGeoms() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  geoms_.reserve(model->ngeom);
  for (int i = 0; i < model->ngeom; ++i) {
    const mjtGeom type = (mjtGeom)model->geom_type[i];

    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    params.cast_shadow = model->geom_castshadow[i];
    auto renderable = CreateRenderable(ctx, params);

    SetGeomMesh(renderable.get(), model_resources_, type,
                model->geom_dataid[i]);

    mjrfMaterial material = GetMaterial(mjOBJ_GEOM, i);
    mjrf_setRenderableMaterial(renderable.get(), &material);

    float size[3];
    if (GetSize(model, type, model->geom_size + (3 * i), size)) {
      mjrf_setRenderableSize(renderable.get(), size);
    }

    const mjtCatBit category = GetBodyCategory(model, model->geom_bodyid[i]);
    if (vopts_.geomgroup[model->geom_group[i]] &&
        (category != mjCAT_STATIC || vopts_.flags[mjVIS_STATIC])) {
      mjrf_addSceneRenderable(scene_, renderable.get());
    }
    geoms_.emplace_back(std::move(renderable));
  }
}

void ModelRenderables::AddSiteGeoms() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  sites_.reserve(model->nsite);
  for (int i = 0; i < model->nsite; ++i) {
    const mjtGeom type = (mjtGeom)model->site_type[i];

    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    auto renderable = CreateRenderable(ctx, params);

    SetGeomMesh(renderable.get(), model_resources_, type,
                model->site_dataid[i]);

    mjrfMaterial material = GetMaterial(mjOBJ_SITE, i);
    mjrf_setRenderableMaterial(renderable.get(), &material);

    float size[3];
    if (GetSize(model, type, model->site_size + (3 * i), size)) {
      mjrf_setRenderableSize(renderable.get(), size);
    }

    const mjtCatBit category = GetBodyCategory(model, model->site_bodyid[i]);
    if (vopts_.sitegroup[model->site_group[i]] &&
        (category != mjCAT_STATIC || vopts_.flags[mjVIS_STATIC])) {
      mjrf_addSceneRenderable(scene_, renderable.get());
    }
    sites_.push_back(std::move(renderable));
  }
}

void ModelRenderables::AddFlexGeoms() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  flexes_.reserve(model->nflex);
  flex_slices_.reserve(model->nflex);
  flex_box_mesh_.reserve(model->nflex);
  flex_meshes_.reserve(model->nflex);
  for (int i = 0; i < model->nflex; ++i) {
    flex_meshes_.emplace_back(nullptr, nullptr);

    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    mjrfMaterial material = GetMaterial(mjOBJ_FLEX, i);

    auto& parts = flexes_.emplace_back();
    auto& slices = flex_slices_.emplace_back();

    const int dim = model->flex_dim[i];
    if (dim == 1) {
      mjtNum box[2];
      const bool is_box = mj_rodBoxSize(model, i, box) && model->flex_elemnum[i] > 0;
      const float radius = model->flex_radius[i];
      const float size[3] = {is_box ? static_cast<float>(box[0]) : radius,
                             is_box ? static_cast<float>(box[1]) : radius,
                             radius};
      parts.reserve(model->flex_edgenum[i]);
      for (int j = 0; j < model->flex_edgenum[i]; ++j) {
        auto renderable = CreateRenderable(ctx, params);
        mjrf_setRenderableGeomMesh(renderable.get(),
                                   is_box ? mjGEOM_BOX : mjGEOM_CAPSULE,
                                   is_box ? 1 : 4, is_box ? 1 : 8, 1);
        mjrf_setRenderableSize(renderable.get(), size);
        mjrf_setRenderableMaterial(renderable.get(), &material);
        parts.push_back(std::move(renderable));
      }
      flex_box_mesh_.push_back(is_box);
    } else {
      flex_box_mesh_.push_back(false);
      slices = CalculateFlexMeshSlices(model, i);
      parts.reserve(slices.size());
      for (size_t k = 0; k < slices.size(); ++k) {
        auto renderable = CreateRenderable(ctx, params);
        mjrf_setRenderableMaterial(renderable.get(), &material);
        parts.push_back(std::move(renderable));
      }
    }

    if (vopts_.flexgroup[model->flex_group[i]]) {
      for (auto& renderable : parts) {
        mjrf_addSceneRenderable(scene_, renderable.get());
      }
    }
  }
}

void ModelRenderables::AddSkinGeoms() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  skins_.reserve(model->nskin);
  skin_meshes_.reserve(model->nskin);
  for (int i = 0; i < model->nskin; i++) {
    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    auto renderable = CreateRenderable(ctx, params);

    mjrfMaterial material = GetMaterial(mjOBJ_SKIN, i);
    mjrf_setRenderableMaterial(renderable.get(), &material);

    if (vopts_.skingroup[model->skin_group[i]]) {
      mjrf_addSceneRenderable(scene_, renderable.get());
    }
    skins_.emplace_back(std::move(renderable));
    skin_meshes_.emplace_back(nullptr, nullptr);
  }
}

void ModelRenderables::AddSliderCrankGeoms() {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();
  const int nstack = model->vis.quality.numstacks;
  const int nslice = model->vis.quality.numslices;
  const int nquad = model->vis.quality.numquads;

  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);

  for (int i = 0; i < model->nactuator; i++) {
    if (model->actuator_trntype[i] != mjTRN_SLIDERCRANK) {
      continue;
    }

    // Create two renderables, one for the slider and the other for the crank.
    auto slider = CreateRenderable(ctx, params);
    mjrf_setRenderableGeomMesh(slider.get(), mjGEOM_CYLINDER, nstack, nslice,
                               nquad);
    mjrf_addSceneRenderable(scene_, slider.get());

    auto crank = CreateRenderable(ctx, params);
    mjrf_setRenderableGeomMesh(crank.get(), mjGEOM_CAPSULE, nstack, nslice,
                               nquad);
    mjrf_addSceneRenderable(scene_, crank.get());

    std::array<UniquePtr<mjrfRenderable>, 2> pair{std::move(slider),
                                                  std::move(crank)};
    slider_cranks_.emplace(i, std::move(pair));
  }
}

void ModelRenderables::UpdateSliderCranks(const mjData* data, int actuator_id) {
  auto it = slider_cranks_.find(actuator_id);
  if (it == slider_cranks_.end()) {
    mju_error("Slider/crank not found: %d", actuator_id);
    return;
  }

  const mjModel* model = model_resources_->GetModel();
  const float scale = model->stat.meansize;
  const float slider_width = scale * model->vis.scale.slidercrank;
  const float crank_width = scale * model->vis.scale.slidercrank / 2.0;

  const int crank_id = model->actuator_trnid[2 * actuator_id];
  const int slider_id = model->actuator_trnid[2 * actuator_id + 1];
  const float3 crank_pos = ReadFloat3(data->site_xpos, crank_id);
  const float3 slider_pos = ReadFloat3(data->site_xpos, slider_id);

  const float3 vec = crank_pos - slider_pos;

  float3 axis;
  axis[0] = data->site_xmat[9 * slider_id + 2];
  axis[1] = data->site_xmat[9 * slider_id + 5];
  axis[2] = data->site_xmat[9 * slider_id + 8];
  float len = dot(vec, axis);

  const float rod = model->actuator_cranklength[actuator_id];
  const float det = (len * len) + (rod * rod) - dot(vec, vec);
  const bool broken = (det < 0);
  if (det > 0) {
    len -= std::sqrt(det);
  }

  const float3 end = slider_pos + (axis * len);

  mjrfRenderable* slider = it->second[0].get();
  mjrfRenderable* crank = it->second[1].get();
  Connect(slider, slider_pos, end, slider_width);
  Connect(crank, end, crank_pos, crank_width);

  mjrfMaterial material = GetMaterial(mjOBJ_ACTUATOR, actuator_id, data);

  xtof(material.color, model->vis.rgba.slidercrank, 4);
  mjrf_setRenderableMaterial(slider, &material);

  if (broken) {
    xtof(material.color, model->vis.rgba.crankbroken, 4);
  }
  mjrf_setRenderableMaterial(crank, &material);
}

void ModelRenderables::AppendSegmentToTendon(int tendon_id) {
  mjrfContext* ctx = model_resources_->GetContext();
  const mjModel* model = model_resources_->GetModel();

  const int nstack = model->vis.quality.numstacks;
  const int nslice = model->vis.quality.numslices;
  const int nquad = model->vis.quality.numquads;

  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);
  auto renderable = CreateRenderable(ctx, params);
  mjrf_setRenderableGeomMesh(renderable.get(), mjGEOM_CAPSULE, nstack, nslice,
                             nquad);
  if (vopts_.tendongroup[model->tendon_group[tendon_id]]) {
    mjrf_addSceneRenderable(scene_, renderable.get());
  }

  auto& segments = tendons_[tendon_id];
  segments.emplace_back(std::move(renderable));
}

void ModelRenderables::RemoveSegmentFromTendon(int tendon_id) {
  auto& segments = tendons_[tendon_id];
  mjrfRenderable* renderable = segments.back().get();
  mjrf_removeSceneRenderable(scene_, renderable);
  segments.pop_back();
}

void ModelRenderables::UpdateSpatialTendons(const mjData* data, int tendon_id) {
  const mjModel* model = model_resources_->GetModel();

  // Gather the points that define the tendon. We'll use a simple cache to avoid
  // excessive reallocations.
  point_cache_.clear();
  GatherSpatialTendonPoints(model, data, tendon_id, point_cache_);

  // Tendons are composed on N "segments" represented by cylinder geoms.
  // We adjust the number of segments we need to match the number of points.
  const int nsegments = static_cast<int>(point_cache_.size()) / 2;
  std::vector<UniquePtr<mjrfRenderable>>& segments = tendons_[tendon_id];
  while (tendons_[tendon_id].size() < nsegments) {
    AppendSegmentToTendon(tendon_id);
  }
  while (tendons_[tendon_id].size() > nsegments) {
    RemoveSegmentFromTendon(tendon_id);
  }

  mjrfMaterial material = GetMaterial(mjOBJ_TENDON, tendon_id, data);

  // If tendon has no explicit color then color it using limit impedance. We
  // use the model color not the material color since the material color may
  // have been overridden.
  const float* rgba = model->tendon_rgba + (4 * tendon_id);
  if (model->tendon_matid[tendon_id] == -1 && rgba[0] == 0.5 &&
      rgba[1] == 0.5 && rgba[2] == 0.5 && rgba[3] == 1) {
    mjtNum imp = 0;
    int efc_start = data->ne + data->nf;
    int efc_end = efc_start + data->nl;
    for (int k = efc_start; k < efc_end; k++) {
      if (data->efc_type[k] == mjCNSTR_LIMIT_TENDON &&
          data->efc_id[k] == tendon_id) {
        imp = data->efc_KBIP[4 * k + 2];
      }
    }
    const float scale = (1 - imp);
    const float* constraint = model->vis.rgba.constraint;
    material.color[0] = scale * material.color[0] + imp * constraint[0];
    material.color[1] = scale * material.color[1] + imp * constraint[1];
    material.color[2] = scale * material.color[2] + imp * constraint[2];
  }

  if (vopts_.flags[mjVIS_ISLAND]) {
    const int ecf = data->nisland ? data->tendon_efcadr[tendon_id] : -1;
    if (ecf >= 0) {
      material.island_id = data->island_dofadr[data->efc_island[ecf]];
      material.sleep_state = mjS_AWAKE;
    }
  }

  for (int i = 0; i < point_cache_.size(); i += 2) {
    mjrfRenderable* renderable = segments[i / 2].get();
    const float width = point_cache_[i].w;
    const float3 p0 = point_cache_[i + 0].xyz;
    const float3 p1 = point_cache_[i + 1].xyz;
    Connect(renderable, p0, p1, width);
    mjrf_setRenderableMaterial(renderable, &material);
  }
}

std::pair<mjtObj, int> ModelRenderables::GetObjectFromSegmentationId(
    int segmentation_id) const {
  if (segmentation_id <= 0) {
    return std::make_pair(mjOBJ_UNKNOWN, -1);
  }

  const mjModel* model = model_resources_->GetModel();
  segmentation_id -= 1;

  if (segmentation_id < model->ngeom) {
    return std::make_pair(mjOBJ_GEOM, segmentation_id);
  } else {
    segmentation_id -= model->ngeom;
  }
  if (segmentation_id < model->nsite) {
    return std::make_pair(mjOBJ_SITE, segmentation_id);
  } else {
    segmentation_id -= model->nsite;
  }
  if (segmentation_id < model->nflex) {
    return std::make_pair(mjOBJ_FLEX, segmentation_id);
  } else {
    segmentation_id -= model->nflex;
  }
  if (segmentation_id < model->nskin) {
    return std::make_pair(mjOBJ_SKIN, segmentation_id);
  } else {
    segmentation_id -= model->nskin;
  }
  if (segmentation_id < model->ntendon) {
    return std::make_pair(mjOBJ_TENDON, segmentation_id);
  } else {
    segmentation_id -= model->ntendon;
  }
  if (segmentation_id < model->nactuator) {
    return std::make_pair(mjOBJ_ACTUATOR, segmentation_id);
  } else {
    segmentation_id -= model->nactuator;
  }

  return std::make_pair(mjOBJ_UNKNOWN, -1);
}

int ModelRenderables::GetSegmentationId(mjtObj obj_type, int obj_index) {
  const mjModel* model = model_resources_->GetModel();

  int id = 1;
  if (obj_type == mjOBJ_GEOM) {
    return id + obj_index;
  } else {
    id += model->ngeom;
  }
  if (obj_type == mjOBJ_SITE) {
    return id + obj_index;
  } else {
    id += model->nsite;
  }
  if (obj_type == mjOBJ_FLEX) {
    return id + obj_index;
  } else {
    id += model->nflex;
  }
  if (obj_type == mjOBJ_SKIN) {
    return id + obj_index;
  } else {
    id += model->nskin;
  }
  if (obj_type == mjOBJ_TENDON) {
    return id + obj_index;
  } else {
    id += model->ntendon;
  }
  if (obj_type == mjOBJ_ACTUATOR) {
    return id + obj_index;
  } else {
    id += model->nactuator;
  }
  mju_error("Unsupported object type: %d", obj_type);
  return -1;
}

mjrfMaterial ModelRenderables::GetMaterial(mjtObj obj_type, int obj_index,
                                           const mjData* data,
                                           std::optional<int> matid_override) {
  const mjModel* model = model_resources_->GetModel();

  mjrfMaterial material;
  mjrf_defaultMaterial(&material);

  int matid = -1;
  int dataid = -1;
  mjtGeom geom_type = mjGEOM_NONE;
  mjtCatBit category = mjCAT_STATIC;
  const float* rgba = nullptr;
  const mjtNum* size = nullptr;
  const mjtNum* pos = nullptr;
  const mjtNum* quat = nullptr;

  switch (obj_type) {
    case mjOBJ_GEOM:
      geom_type = (mjtGeom)model->geom_type[obj_index];
      category = GetBodyCategory(model, model->geom_bodyid[obj_index]);
      rgba = model->geom_rgba + (4 * obj_index);
      size = model->geom_size + (3 * obj_index);
      pos = model->geom_pos + (3 * obj_index);
      quat = model->geom_quat + (4 * obj_index);
      matid = model->geom_matid[obj_index];
      dataid = model->geom_dataid[obj_index];
      break;
    case mjOBJ_SITE:
      geom_type = (mjtGeom)model->site_type[obj_index];
      category = GetBodyCategory(model, model->site_bodyid[obj_index]);
      rgba = model->site_rgba + (4 * obj_index);
      size = model->site_size + (3 * obj_index);
      matid = model->site_matid[obj_index];
      dataid = model->site_dataid[obj_index];
      break;
    case mjOBJ_FLEX:
      geom_type = mjGEOM_FLEX;
      category = mjCAT_DYNAMIC;
      rgba = model->flex_rgba + (4 * obj_index);
      size = model->flex_size + (3 * obj_index);
      if (model->flex_elemnum[obj_index] > 0) {
        matid = model->flex_matid[model->flex_elemadr[obj_index]];
      }
      break;
    case mjOBJ_SKIN:
      geom_type = mjGEOM_SKIN;
      category = mjCAT_DYNAMIC;
      rgba = model->skin_rgba + (4 * obj_index);
      matid = model->skin_matid[obj_index];
      break;
    case mjOBJ_TENDON:
      geom_type = mjGEOM_CAPSULE;
      category = mjCAT_DYNAMIC;
      rgba = model->tendon_rgba + (4 * obj_index);
      size = model->tendon_width + obj_index;
      matid = model->tendon_matid[obj_index];
      break;
    case mjOBJ_ACTUATOR:
      geom_type = mjGEOM_CYLINDER;
      rgba = model->vis.rgba.slidercrank;
      // size = model->tendon_width + obj_index;
      break;
    default:
      mju_error("Unsupported object type: %d", obj_type);
      break;
  }

  if (matid_override.has_value()) {
    matid = *matid_override;
  }

  xtof(material.color, rgba, 4);

  if (matid >= 0 && matid < model->nmat) {
    auto get_texture = [&](int role) -> const mjrfTexture* {
      const int tex_id = model->mat_texid[matid * mjNTEXROLE + role];
      return tex_id >= 0 ? model_resources_->GetTexture(tex_id) : nullptr;
    };
    material.color_texture = get_texture(mjTEXROLE_RGB);
    material.normal_texture = get_texture(mjTEXROLE_NORMAL);
    material.emissive_texture = get_texture(mjTEXROLE_EMISSIVE);
    material.orm_texture = get_texture(mjTEXROLE_ORM);
    material.metallic_texture = get_texture(mjTEXROLE_METALLIC);
    material.roughness_texture = get_texture(mjTEXROLE_ROUGHNESS);
    material.occlusion_texture = get_texture(mjTEXROLE_OCCLUSION);
    material.opacity_texture = get_texture(mjTEXROLE_OPACITY);

    material.emissive = model->mat_emission[matid];
    material.specular = model->mat_specular[matid];
    material.glossiness = model->mat_shininess[matid];
    material.reflectance = model->mat_reflectance[matid];
    material.metallic = model->mat_metallic[matid];
    material.roughness = model->mat_roughness[matid];

    if (rgba[0] == 0.5f && rgba[1] == 0.5f && rgba[2] == 0.5f &&
        rgba[3] == 1.0f) {
      const float* mat_rgba = model->mat_rgba + 4 * matid;
      xtof(material.color, mat_rgba, 4);
    }
  } else {
    material.emissive = 0;
    material.specular = 0.5;
    material.glossiness = 0.5;
    material.reflectance = 0;
  }

  // UvScale only applies to objects that don't have explicit UV coordinates
  // in their vertex buffer. Instead, we set the UV coordinate to be the same
  // as the vertex position.
  //
  // The material's `texuniform` and `texrepeat` parameters allow us to scale
  // the programmatic UVs.

  if (material.color_texture) {
    float fsize[3] = {1.0f, 1.0f, 1.0f};
    if (size) {
      xtof(fsize, size, 3);
      if (geom_type != mjGEOM_NONE) {
        switch (geom_type) {
          case mjGEOM_SPHERE:
            fsize[2] = size[0];
            fsize[1] = size[0];
            break;
          case mjGEOM_CAPSULE:
          case mjGEOM_CYLINDER:
            fsize[2] = size[1];
            fsize[1] = size[0];
            break;
          case mjGEOM_TRIANGLE:
          case mjGEOM_PLANE:
            fsize[2] = 1.0f;
            break;
          default:
            break;
        }
      }
    }

    const bool tex_uniform = model->mat_texuniform[matid];
    if (mjrf_getTextureSamplerType(material.color_texture) == mjTEXTURE_2D) {
      // For 2D textures, `tex_repeat` specifies how many times the texture
      // image is repeated. The `tex_uniform` flag determines if the repetition
      // is applied at in object space (false) or in world space (true).
      float tex_repeat[2];
      tex_repeat[0] = model->mat_texrepeat[(matid * 2) + 0];
      tex_repeat[1] = model->mat_texrepeat[(matid * 2) + 1];
      material.uv_scale[0] = tex_repeat[0];
      material.uv_scale[1] = tex_repeat[1];

      const bool is_infinite_plane =
          geom_type == mjGEOM_PLANE && (fsize[0] <= 0 || fsize[1] <= 0);
      if (is_infinite_plane) {
        // With infinite planes, we want to use the UvOffset to account for
        // the plane's position in world space due to re-centering.
        //
        // Infinite planes will use world-space UVs, with a tile size of 1x1
        // world units by default (texrepeat=(1, 1)).
        //
        // The shader computes: uv = uv0 * UvScale + UvOffset
        // We want:             uv = dot(worldPos, planeAxis) * texFrequency
        //
        // Since worldPos = rotation * (objectPos * planeScale) + geomPos:
        //
        // dot(worldPos, axisK) = objectPos.k * planeScale + dot(geomPos, axisK)
        //
        // Meaning:
        //   UvScale  = planeScale * texrepeat
        //   UvOffset = dot(geomPos, planeAxis) * texrepeat
        const float plane_scale = static_cast<float>(mjMAXPLANEGRID) / 2.0f;

        float dot_pos_axis_x = 0.0f;
        float dot_pos_axis_y = 0.0f;
        if (pos && quat) {
          mjtNum mat[9];
          mju_quat2Mat(mat, quat);
          dot_pos_axis_x = pos[0] * mat[0] + pos[1] * mat[3] + pos[2] * mat[6];
          dot_pos_axis_y = pos[0] * mat[1] + pos[1] * mat[4] + pos[2] * mat[7];
        }

        material.uv_scale[0] *= plane_scale;
        material.uv_scale[1] *= plane_scale;
        // The vertex UVs in PlaneBuilder are
        //     u0 =  (0.5 * x) + 0.5
        //     v0 = -(0.5 * y) + 0.5
        // To keep the world-space texture coordinate
        //     u = 0.5 * worldX * texrepeat - 0.5
        // independent of the snapped geomPos, uv_offset must compensate by
        // 0.5 * dot_pos.
        material.uv_offset[0] =
            (0.5f * dot_pos_axis_x - 0.5f * plane_scale) * tex_repeat[0] - 0.5f;
        material.uv_offset[1] =
            (-0.5f * dot_pos_axis_y - 0.5f * plane_scale) * tex_repeat[1] - 0.5f;
      } else {
        if (geom_type == mjGEOM_HFIELD ||
            ((geom_type == mjGEOM_MESH || geom_type == mjGEOM_SDF) &&
             dataid >= 0 && model->mesh_texcoordadr[dataid] < 0)) {
          if (fsize[0] > 0) {
            material.uv_scale[0] /= mju_max(mjMINVAL, fsize[0]);
          }
          if (fsize[1] > 0) {
            material.uv_scale[1] /= mju_max(mjMINVAL, fsize[1]);
          }
          // Adjust the offset to the origin to match OpenGL's glTexGen.
          material.uv_offset[0] = -0.5f;
          material.uv_offset[1] = -0.5f;
        }
        if (tex_uniform && geom_type != mjGEOM_FLEX &&
            geom_type != mjGEOM_SKIN) {
          if (fsize[0] > 0) {
            material.uv_scale[0] *= fsize[0];
          }
          if (fsize[1] > 0) {
            material.uv_scale[1] *= fsize[1];
          }
        }
      }
    } else {
      // For cube maps, if `tex_uniform` is true, then scale the texture so that
      // it covers a 1x1 area of world space rather than the area of the object.
      if (tex_uniform) {
        material.uv_scale[0] = 1.0f / (fsize[0] ? fsize[0] : 1.0f);
        material.uv_scale[1] = 1.0f / (fsize[1] ? fsize[1] : 1.0f);
        material.uv_scale[2] = 1.0f / (fsize[2] ? fsize[2] : 1.0f);
      }
    }
  }

  // Apply material multipliers from the model.
  material.emissive *= model_resources_->GetEmissiveMultiplier();
  material.specular *= model_resources_->GetSpecularMultiplier();
  material.glossiness *= model_resources_->GetShininessMultiplier();

  material.segmentation_id = GetSegmentationId(obj_type, obj_index);

  // Apply transparency to dynamic objects.
  if (vopts_.flags[mjVIS_TRANSPARENT] && category == mjCAT_DYNAMIC) {
    material.color[3] *= model->vis.map.alpha;
  }

  // Highlight selected objects.
  if (selected_obj_type_ == mjOBJ_BODY) {
    if (obj_type == mjOBJ_GEOM &&
        model->geom_bodyid[obj_index] == selected_obj_index_) {
      material.selected = true;
    }
    if (obj_type == mjOBJ_SITE &&
        model->site_bodyid[obj_index] == selected_obj_index_) {
      material.selected = true;
    }
  } else if (selected_obj_type_ == obj_type &&
             selected_obj_index_ == obj_index) {
    material.selected = true;
  }

  // Set island IDs based on sleep state.
  if (data && vopts_.flags[mjVIS_ISLAND]) {
    const bool sleep_enabled = model->opt.enableflags & mjENBL_SLEEP;

    if (obj_type == mjOBJ_GEOM) {
      const int body_id = model->geom_bodyid[obj_index];
      const int weld_id = model->body_weldid[body_id];
      if (model->body_dofnum[weld_id]) {
        const int awake = data->body_awake[body_id];
        const int dof = model->body_dofadr[weld_id];
        const int island = data->nisland ? data->dof_island[dof] : -1;
        int island_id = island >= 0 ? data->island_dofadr[island] : -1;
        if (island_id == -1 && sleep_enabled) {
          int tree = model->dof_treeid[dof];
          if (!awake) {
            tree = mj_sleepCycle(data->tree_asleep, model->ntree, tree);
          }
          island_id = model->tree_dofadr[tree];
        }

        material.island_id = island_id;
        material.sleep_state = awake ? mjS_AWAKE : mjS_ASLEEP;
      }
    }

    if (obj_type == mjOBJ_FLEX) {
      int bodyid = -1;
      if (model->flex_interp[obj_index]) {
        int nodeadr = model->flex_nodeadr[obj_index];
        for (int j = 0; j < model->flex_nodenum[obj_index] && bodyid < 0; j++) {
          int b = model->flex_nodebodyid[nodeadr + j];
          if (model->body_treeid[b] >= 0) bodyid = b;
        }
      } else {
        int vertadr = model->flex_vertadr[obj_index];
        for (int j = 0; j < model->flex_vertnum[obj_index] && bodyid < 0; j++) {
          int b = model->flex_vertbodyid[vertadr + j];
          if (model->body_treeid[b] >= 0) bodyid = b;
        }
      }
      if (bodyid >= 0) {
        int weld_id = model->body_weldid[bodyid];
        int dof = model->body_dofadr[weld_id];
        int island = data->nisland ? data->dof_island[dof] : -1;
        int island_id = island >= 0 ? data->island_dofadr[island] : -1;
        int awake = data->body_awake[bodyid];
        if (island_id == -1 && sleep_enabled) {
          int tree = model->dof_treeid[dof];
          if (!awake) {
            tree = mj_sleepCycle(data->tree_asleep, model->ntree, tree);
          }
          island_id = model->tree_dofadr[tree];
        }
        material.island_id = island_id;
        material.sleep_state = awake ? mjS_AWAKE : mjS_ASLEEP;
      }
    }
  }
  return material;
}

namespace {
enum VisibilityOp { kNoop, kShow, kHide };
}

static void GetVisibilityOps(VisibilityOp* ops, const mjtByte* vis_by_group,
                             const mjtByte* vis_by_flag, int idx,
                             bool visible) {
  const bool prev_global_vis = vis_by_flag ? *vis_by_flag : true;
  const bool curr_global_vis =
      (vis_by_flag && idx == mjNGROUP) ? visible : prev_global_vis;

  for (int i = 0; i < mjNGROUP; ++i) {
    const bool prev_group_vis = vis_by_group[i];
    const bool curr_group_vis = idx == i ? visible : prev_group_vis;

    const bool previously_visible = prev_global_vis && prev_group_vis;
    const bool currently_visible = curr_global_vis && curr_group_vis;

    if (previously_visible == currently_visible) {
      ops[i] = kNoop;
    } else if (currently_visible) {
      ops[i] = kShow;
    } else {
      ops[i] = kHide;
    }
  }
}

static void ApplyVisibility(VisibilityOp* ops, int group, mjrfScene* scene,
                            std::span<UniquePtr<mjrfRenderable>> renderables) {
  const VisibilityOp op = ops[group];
  if (op == kShow) {
    for (auto& renderable : renderables) {
      mjrf_addSceneRenderable(scene, renderable.get());
    }
  } else if (op == kHide) {
    for (auto& renderable : renderables) {
      mjrf_removeSceneRenderable(scene, renderable.get());
    }
  }
}

void ModelRenderables::SetVisibility(mjtObj obj_type, int idx, bool visible) {
  const mjModel* model = model_resources_->GetModel();
  const mjtByte* static_flag = &vopts_.flags[mjVIS_STATIC];

  VisibilityOp ops[mjNGROUP] = {kNoop};
  VisibilityOp static_ops[mjNGROUP] = {kNoop};
  switch (obj_type) {
    case mjOBJ_GEOM:
      GetVisibilityOps(ops, vopts_.geomgroup, nullptr, idx, visible);
      GetVisibilityOps(static_ops, vopts_.geomgroup, static_flag, idx, visible);
      for (int i = 0; i < model->ngeom; ++i) {
        const int category = GetBodyCategory(model, model->geom_bodyid[i]);
        VisibilityOp* vis_ops = (category == mjCAT_STATIC) ? static_ops : ops;
        ApplyVisibility(vis_ops, model->geom_group[i], scene_, {&geoms_[i], 1});
      }
      break;
    case mjOBJ_SITE:
      GetVisibilityOps(ops, vopts_.sitegroup, nullptr, idx, visible);
      GetVisibilityOps(static_ops, vopts_.sitegroup, static_flag, idx, visible);
      for (int i = 0; i < model->nsite; ++i) {
        const int category = GetBodyCategory(model, model->site_bodyid[i]);
        VisibilityOp* vis_ops = (category == mjCAT_STATIC) ? static_ops : ops;
        ApplyVisibility(vis_ops, model->site_group[i], scene_, {&sites_[i], 1});
      }
      break;
    case mjOBJ_FLEX:
      GetVisibilityOps(ops, vopts_.flexgroup, &vopts_.flags[mjVIS_FLEXSKIN],
                       idx, visible);
      for (int i = 0; i < model->nflex; ++i) {
        ApplyVisibility(ops, model->flex_group[i], scene_, flexes_[i]);
      }
      break;
    case mjOBJ_SKIN:
      GetVisibilityOps(ops, vopts_.skingroup, &vopts_.flags[mjVIS_SKIN], idx,
                       visible);
      for (int i = 0; i < model->nskin; ++i) {
        ApplyVisibility(ops, model->skin_group[i], scene_, {&skins_[i], 1});
      }
      break;
    case mjOBJ_TENDON:
      GetVisibilityOps(ops, vopts_.tendongroup, &vopts_.flags[mjVIS_TENDON],
                       idx, visible);
      for (int i = 0; i < model->ntendon; ++i) {
        ApplyVisibility(ops, model->tendon_group[i], scene_, tendons_[i]);
      }
      break;
    case mjOBJ_ACTUATOR:
      GetVisibilityOps(ops, vopts_.actuatorgroup, &vopts_.flags[mjVIS_ACTUATOR],
                       idx, visible);
      for (int i = 0; i < model->nactuator; ++i) {
        auto it = slider_cranks_.find(i);
        if (it == slider_cranks_.end()) {
          continue;
        }
        ApplyVisibility(ops, model->actuator_group[i], scene_, it->second);
      }
      break;
    default:
      mju_error("Unsupported object type: %d", obj_type);
      break;
  }
}

void ModelRenderables::SetOptions(const mjvOption& opt) {
  const mjModel* model = model_resources_->GetModel();

  if (vopts_.flags[mjVIS_STATIC] != opt.flags[mjVIS_STATIC]) {
    SetVisibility(mjOBJ_GEOM, mjNGROUP, opt.flags[mjVIS_STATIC]);
    SetVisibility(mjOBJ_SITE, mjNGROUP, opt.flags[mjVIS_STATIC]);
    vopts_.flags[mjVIS_STATIC] = opt.flags[mjVIS_STATIC];
  }
  for (int i = 0; i < mjNGROUP; ++i) {
    if (vopts_.geomgroup[i] != opt.geomgroup[i]) {
      SetVisibility(mjOBJ_GEOM, i, opt.geomgroup[i]);
      vopts_.geomgroup[i] = opt.geomgroup[i];
    }
    if (vopts_.sitegroup[i] != opt.sitegroup[i]) {
      SetVisibility(mjOBJ_SITE, i, opt.sitegroup[i]);
      vopts_.sitegroup[i] = opt.sitegroup[i];
    }
    if (vopts_.skingroup[i] != opt.skingroup[i]) {
      SetVisibility(mjOBJ_SKIN, i, opt.skingroup[i]);
      vopts_.skingroup[i] = opt.skingroup[i];
    }
    if (vopts_.flexgroup[i] != opt.flexgroup[i]) {
      SetVisibility(mjOBJ_FLEX, i, opt.flexgroup[i]);
      vopts_.flexgroup[i] = opt.flexgroup[i];
    }
    if (vopts_.tendongroup[i] != opt.tendongroup[i]) {
      SetVisibility(mjOBJ_TENDON, i, opt.tendongroup[i]);
      vopts_.tendongroup[i] = opt.tendongroup[i];
    }
    if (vopts_.actuatorgroup[i] != opt.actuatorgroup[i]) {
      SetVisibility(mjOBJ_ACTUATOR, i, opt.actuatorgroup[i]);
      vopts_.actuatorgroup[i] = opt.actuatorgroup[i];
    }
  }
  if (vopts_.flags[mjVIS_SKIN] != opt.flags[mjVIS_SKIN]) {
    SetVisibility(mjOBJ_SKIN, mjNGROUP, opt.flags[mjVIS_SKIN]);
    vopts_.flags[mjVIS_SKIN] = opt.flags[mjVIS_SKIN];
  }
  if (vopts_.flags[mjVIS_FLEXSKIN] != opt.flags[mjVIS_FLEXSKIN]) {
    SetVisibility(mjOBJ_FLEX, mjNGROUP, opt.flags[mjVIS_FLEXSKIN]);
    vopts_.flags[mjVIS_FLEXSKIN] = opt.flags[mjVIS_FLEXSKIN];
  }
  if (vopts_.flags[mjVIS_TENDON] != opt.flags[mjVIS_TENDON]) {
    SetVisibility(mjOBJ_TENDON, mjNGROUP, opt.flags[mjVIS_TENDON]);
    vopts_.flags[mjVIS_TENDON] = opt.flags[mjVIS_TENDON];
  }
  if (vopts_.flags[mjVIS_ACTUATOR] != opt.flags[mjVIS_ACTUATOR]) {
    SetVisibility(mjOBJ_ACTUATOR, mjNGROUP, opt.flags[mjVIS_ACTUATOR]);
    vopts_.flags[mjVIS_ACTUATOR] = opt.flags[mjVIS_ACTUATOR];
  }

  if (vopts_.flags[mjVIS_CONVEXHULL] != opt.flags[mjVIS_CONVEXHULL]) {
    for (int i = 0; i < model->ngeom; ++i) {
      const mjtGeom geom_type = (mjtGeom)model->geom_type[i];
      if (geom_type == mjGEOM_MESH || geom_type == mjGEOM_SDF) {
        mjrfRenderable* renderable = geoms_[i].get();
        const int mesh_id =
            model->geom_dataid[i] * 2 + (opt.flags[mjVIS_CONVEXHULL] ? 1 : 0);
        const mjrfMesh* mesh = model_resources_->GetMesh(mesh_id);
        if (mesh) {
          mjrf_setRenderableMesh(renderable, mesh, 0, 0);
        }
      }
    }
  }
  vopts_ = opt;
}

void ModelRenderables::MarkAsSelected(mjtObj obj_type, int obj_index) {
  selected_obj_type_ = obj_type;
  selected_obj_index_ = obj_index;
}

}  // namespace mujoco
