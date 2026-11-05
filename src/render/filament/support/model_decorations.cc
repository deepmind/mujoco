// Copyright 2025 DeepMind Technologies Limited
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

#include "render/filament/support/model_decorations.h"

#include <algorithm>
#include <cstring>
#include <span>
#include <utility>

#include <math/TMatHelpers.h>
#include <math/mat4.h>
#include <math/vec3.h>
#include <math/vec4.h>
#include <mujoco/mjrfilament.h>
#include <mujoco/mujoco.h>
#include "render/filament/support/mjrf_ptr.h"

namespace mujoco {

using filament::math::float3;
using filament::math::float4;
using filament::math::mat4;

ModelDecorations::ModelDecorations(mjrfContext* ctx, mjrfScene* scene,
                                   const mjModel* model, int num_geoms)
    : ctx_(ctx), scene_(scene), model_(model) {
  std::memset(&mjv_scene_, 0, sizeof(mjvScene));
  mjv_makeScene(model_, &mjv_scene_, num_geoms);
}

ModelDecorations::~ModelDecorations() {
  Clear();
  mjv_freeScene(&mjv_scene_);
}

void ModelDecorations::Clear() {
  for (auto& iter : decorations_) {
    mjrf_removeSceneRenderable(scene_, iter.get());
  }
  decorations_.clear();
  meshes_.clear();
}

static mat4 CalcClipFromWorld(const mjModel* model, const mjData* data,
                              mjvCamera mjv_camera, const mjrRect& viewport) {
  if (mjv_camera.type == mjCAMERA_TRACKING) {
    int bid = mjv_camera.trackbodyid;
    if (bid < 0 || bid >= model->nbody) {
      mju_error("track body id is outside valid range");
    }
    mju_copy3(mjv_camera.lookat, data->subtree_com + 3 * bid);
  }

  mjtNum headpos[3], forward[3], up[3];
  mjv_cameraFrame(headpos, forward, up, nullptr, data, &mjv_camera);
  float zver[2], zhor[2], zclip[2] = {0, 0};
  mjv_cameraFrustum(zver, zhor, zclip, model, &mjv_camera);

  const float frustum_top = zver[0];
  const float frustum_bottom = -zver[1];
  const float frustum_center = (zhor[1] - zhor[0]) / 2;
  const float frustum_width = (zhor[1] + zhor[0]) / 2;
  const float frustum_near = zclip[0];
  const float frustum_far = zclip[1];

  bool orthographic = mjv_camera.orthographic;
  if (mjv_camera.type == mjCAMERA_FIXED) {
    const int cid = mjv_camera.fixedcamid;
    if (cid < 0 || cid >= model->ncam) {
      mju_error("fixed camera id is outside valid range");
    }
    orthographic = model->cam_projection[cid] == mjPROJ_ORTHOGRAPHIC;
  }

  const float3 cam_pos{headpos[0], headpos[1], headpos[2]};
  const float3 cam_fwd{forward[0], forward[1], forward[2]};
  const float3 cam_up{up[0], up[1], up[2]};
  const float3 cam_at = cam_pos + cam_fwd;
  const float aspect_ratio = (float)viewport.width / (float)viewport.height;
  const float halfwidth =
      frustum_width ? frustum_width
                    : 0.5f * aspect_ratio * (frustum_top - frustum_bottom);
  const float left = frustum_center - halfwidth;
  const float right = frustum_center + halfwidth;

  mat4 projection;
  if (orthographic) {
    projection = mat4::ortho(left, right, frustum_bottom, frustum_top,
                             frustum_near, frustum_far);
  } else {
    projection = mat4::frustum(left, right, frustum_bottom, frustum_top,
                               frustum_near, frustum_far);
    projection[2][2] = -1.0f;
    projection[3][2] = -2.0f * frustum_near;
  }
  const mat4 look_at = mat4::lookAt(cam_pos, cam_at, cam_up);
  return projection * inverse(look_at);
}

void ModelDecorations::Update(mjData* data, const mjvOption* vis_option,
                              const mjvPerturb* perturb, mjvCamera* camera,
                              const mjrRect& viewport,
                              DrawTextAtFn draw_text_at_fn,
                              std::span<const mjvGeom> extra_geoms) {
  Clear();

  const int nstack = model_->vis.quality.numstacks;
  const int nslice = model_->vis.quality.numslices;
  const int nquad = model_->vis.quality.numquads;

  // For decor rendering, we go through the mjvScene. Ideally, we would only
  // update the mjvGeoms in the scene, but the API is not capable of that.
  mjv_updateScene(model_, data, vis_option, perturb, camera, mjCAT_ALL,
                  &mjv_scene_);
  const int num_extra_geoms =
      std::min<int>(extra_geoms.size(), mjv_scene_.maxgeom - mjv_scene_.ngeom);
  for (int i = 0; i < num_extra_geoms; ++i) {
    mjv_scene_.geoms[mjv_scene_.ngeom] = extra_geoms[i];
    mjv_scene_.geoms[mjv_scene_.ngeom].category = mjCAT_DECOR;
    ++mjv_scene_.ngeom;
  }

  const mat4 clip_from_world =
      CalcClipFromWorld(model_, data, *camera, viewport);
  for (int i = 0; i < mjv_scene_.ngeom; ++i) {
    const mjvGeom& geom = mjv_scene_.geoms[i];
    if (draw_text_at_fn && geom.label[0] != 0) {
      const float4 clip_pos =
          clip_from_world * float4(geom.pos[0], geom.pos[1], geom.pos[2], 1.0f);
      if (clip_pos.w != 0.0f) {
        const float3 pos = clip_pos.xyz / clip_pos.w;
        draw_text_at_fn(geom.label, pos.x, pos.y, pos.z);
      }
    }

    const mjtGeom geom_type = (mjtGeom)geom.type;

    if (geom_type == mjGEOM_NONE || geom_type == mjGEOM_LABEL) {
      continue;
    }
    if (geom.category != mjCAT_DECOR && geom.type != mjGEOM_FLEX) {
      continue;
    }

    if (geom.type == mjGEOM_FLEX) {
      // Draw flexes "normally" with skins; no decor/debug rendering needed.
      if (mjv_scene_.flexskinopt) {
        continue;
      }

      mjrfMaterial material;
      mjrf_defaultMaterial(&material);
      material.decor = true;
      material.color[0] = model_->flex_rgba[4 * geom.objid + 0];
      material.color[1] = model_->flex_rgba[4 * geom.objid + 1];
      material.color[2] = model_->flex_rgba[4 * geom.objid + 2];
      material.color[3] = model_->flex_rgba[4 * geom.objid + 3];

      mjrfRenderableParams params;
      mjrf_defaultRenderableParams(&params);

      const int vertadr = mjv_scene_.flexvertadr[geom.objid];
      const int vertnum = mjv_scene_.flexvertnum[geom.objid];
      const float radius = model_->flex_radius[geom.objid];

      if (geom.dataid <= 0 && mjv_scene_.flexvertopt) {
        // Use small spheres to represent vertices.
        const float rot[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f};
        const float size[] = {radius, radius, radius};
        for (int v = vertadr; v < vertadr + vertnum; ++v) {
          auto vertex = CreateRenderable(ctx_, params);
          mjrf_setRenderableGeomMesh(vertex.get(), mjGEOM_SPHERE, 8, 8, 1);
          mjrf_setRenderableMaterial(vertex.get(), &material);
          mjrf_setRenderableSize(vertex.get(), size);
          mjrf_setRenderablePose(vertex.get(), mjv_scene_.flexvert + 3 * v,
                                      rot);
          mjrf_addSceneRenderable(scene_, vertex.get());
          decorations_.push_back(std::move(vertex));
        }
      }

      if (geom.dataid <= 0 && mjv_scene_.flexedgeopt) {
        const int edgeadr = mjv_scene_.flexedgeadr[geom.objid];
        const int edgenum = mjv_scene_.flexedgenum[geom.objid];

        // Use small thin cylinders to represent the edges.
        for (int e = edgeadr; e < edgeadr + edgenum; ++e) {
          const float* v1 =
              mjv_scene_.flexvert + 3 * (vertadr + mjv_scene_.flexedge[2 * e]);
          const float* v2 = mjv_scene_.flexvert +
                            3 * (vertadr + mjv_scene_.flexedge[2 * e + 1]);
          const mjtNum vec[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};

          const float pos[3]{
              (v1[0] + v2[0]) * 0.5f,
              (v1[1] + v2[1]) * 0.5f,
              (v1[2] + v2[2]) * 0.5f,
          };

          mjtNum quat[4];
          mju_quatZ2Vec(quat, vec);
          mjtNum edgemat[9];
          mju_quat2Mat(edgemat, quat);
          const float rot[9] = {
              static_cast<float>(edgemat[0]), static_cast<float>(edgemat[1]),
              static_cast<float>(edgemat[2]), static_cast<float>(edgemat[3]),
              static_cast<float>(edgemat[4]), static_cast<float>(edgemat[5]),
              static_cast<float>(edgemat[6]), static_cast<float>(edgemat[7]),
              static_cast<float>(edgemat[8]),
          };

          const float len = static_cast<float>(mju_norm3(vec));
          const float size[3] = {radius, radius, len * 0.5f};

          auto vertex = CreateRenderable(ctx_, params);
          mjrf_setRenderableGeomMesh(vertex.get(), mjGEOM_CYLINDER, 1, 8, 1);
          mjrf_setRenderableMaterial(vertex.get(), &material);
          mjrf_setRenderableSize(vertex.get(), size);
          mjrf_setRenderablePose(vertex.get(), pos, rot);
          mjrf_addSceneRenderable(scene_, vertex.get());
          decorations_.push_back(std::move(vertex));
        }
      }

      const int facenum = geom.dataid >= 0
                              ? static_cast<int>(geom.size[2])
                              : mjv_scene_.flexfaceused[geom.objid];
      if (mjv_scene_.flexfaceopt && facenum > 0) {
        const bool has_uvs = geom.texcoord && geom.matid >= 0;
        const int addr =
            mjv_scene_.flexfaceadr[geom.objid] +
            (geom.dataid >= 0 ? static_cast<int>(geom.size[1]) : 0);
        const float* positions = mjv_scene_.flexface + (9 * addr);
        const float* normals = mjv_scene_.flexnormal + (9 * addr);
        const float* uvs =
            has_uvs ? mjv_scene_.flextexcoord + (6 * addr) : nullptr;

        mjrfMeshConfig config;
        mjrf_defaultMeshConfig(&config);
        config.num_attributes = has_uvs ? 3 : 2;
        config.max_vertices = 3 * facenum;
        config.max_indices = 3 * facenum;
        config.attributes[0].usage = mjVERTEX_ATTRIBUTE_USAGE_POSITION;
        config.attributes[0].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
        config.attributes[1].usage = mjVERTEX_ATTRIBUTE_USAGE_NORMAL;
        config.attributes[1].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT3;
        config.attributes[2].usage = mjVERTEX_ATTRIBUTE_USAGE_UV;
        config.attributes[2].type = mjVERTEX_ATTRIBUTE_TYPE_FLOAT2;
        config.index_type = mjINDEX_TYPE_U32;
        config.primitive_type = mjMESH_PRIMITIVE_TYPE_TRIANGLES;

        mjrfMeshData data;
        mjrf_defaultMeshData(&data);
        data.vertices[0] = positions;
        data.vertices[1] = normals;
        data.vertices[2] = uvs;
        data.num_vertices = 3 * facenum;
        data.num_indices = 3 * facenum;
        data.indices = nullptr;
        data.compute_bounds = true;
        data.release = nullptr;
        data.user_data = nullptr;

        auto mesh = CreateMesh(ctx_, config);
        mjrf_uploadMesh(mesh.get(), &data);
        meshes_.push_back(std::move(mesh));

        auto renderable = CreateRenderable(ctx_, params);
        mjrf_setRenderableMesh(renderable.get(), meshes_.back().get(), 0, 0);
        mjrf_setRenderableMaterial(renderable.get(), &material);
        mjrf_addSceneRenderable(scene_, renderable.get());
        decorations_.push_back(std::move(renderable));
      }
      continue;
    }

    mjrfRenderableParams params;
    mjrf_defaultRenderableParams(&params);
    auto renderable = CreateRenderable(ctx_, params);

    mjrf_setRenderableGeomMesh(renderable.get(), geom_type, nstack, nslice,
                               nquad);
    mjrf_setRenderablePose(renderable.get(), geom.pos, geom.mat);
    if (geom_type == mjGEOM_PLANE) {
      // Planes only define an xy size, so set the z-dimension to 1.0f.
      const float size[3] = {geom.size[0], geom.size[1], 1.0f};
      mjrf_setRenderableSize(renderable.get(), size);
    } else {
      mjrf_setRenderableSize(renderable.get(), geom.size);
    }

    mjrfMaterial material;
    mjrf_defaultMaterial(&material);
    material.decor = true;
    material.color[0] = geom.rgba[0];
    material.color[1] = geom.rgba[1];
    material.color[2] = geom.rgba[2];
    material.color[3] = geom.rgba[3];
    mjrf_setRenderableMaterial(renderable.get(), &material);

    mjrf_addSceneRenderable(scene_, renderable.get());
    decorations_.push_back(std::move(renderable));
  }
}

}  // namespace mujoco
