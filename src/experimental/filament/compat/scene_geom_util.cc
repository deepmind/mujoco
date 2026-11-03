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

#include "experimental/filament/compat/scene_geom_util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>

#include <mujoco/mjrfilament.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "experimental/filament/compat/scene_objects.h"
#include "render/filament/support/mjrf_ptr.h"
#include "render/filament/support/model_resources.h"

namespace mujoco {

static void PrepareGeomMeshes(mjrfRenderable* renderable, const mjvGeom& geom,
                              ModelResources* model_objs,
                              SceneObjects* scene_objs) {
  const mjModel* model = model_objs->GetModel();
  const int nstack = model->vis.quality.numstacks;
  const int nslice = model->vis.quality.numslices;
  const int nquad = model->vis.quality.numquads;

  float position[3];
  std::memcpy(position, &geom.pos, 3 * sizeof(float));
  float rotation[9];
  std::memcpy(rotation, &geom.mat, 9 * sizeof(float));

  const mjtGeom geom_type = (mjtGeom)geom.type;
  switch (geom_type) {
    case mjGEOM_MESH:
    case mjGEOM_SDF:
      mjrf_setRenderableMesh(renderable, model_objs->GetMesh(geom.dataid), 0, 0);
      break;
    case mjGEOM_HFIELD:
      mjrf_setRenderableMesh(renderable, model_objs->GetHeightField(geom.dataid), 0, 0);
      break;
    case mjGEOM_PLANE: {
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      // Planes only define an xy size, so set the z-dimension to 1.0f.
      const float size[3] = {geom.size[0], geom.size[1], 1.0f};
      mjrf_setRenderableSize(renderable, size);
      break;
    }
    case mjGEOM_SPHERE:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_ELLIPSOID:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_BOX:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_CAPSULE:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_CYLINDER:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_ARROW:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_ARROW1:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_ARROW2:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_LINE:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_LINEBOX:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_TRIANGLE:
      mjrf_setRenderableGeomMesh(renderable, geom_type, nstack, nslice, nquad);
      mjrf_setRenderableSize(renderable, geom.size);
      break;
    case mjGEOM_FLEX:
      mjrf_setRenderableMesh(
          renderable, scene_objs->GetFlexMesh(geom.objid, geom.dataid), 0, 0);
      // Flexes are defined in global space.
      std::memset(position, 0, sizeof(position));
      std::memset(rotation, 0, sizeof(rotation));
      rotation[0] = 1.f;
      rotation[4] = 1.f;
      rotation[8] = 1.f;
      break;
    case mjGEOM_SKIN:
      mjrf_setRenderableMesh(renderable, scene_objs->GetSkinMesh(geom.objid), 0, 0);
      // Skins are defined in global space.
      std::memset(position, 0, sizeof(position));
      std::memset(rotation, 0, sizeof(rotation));
      rotation[0] = 1.f;
      rotation[4] = 1.f;
      rotation[8] = 1.f;
      break;
    case mjGEOM_NONE:
    case mjGEOM_LABEL:
      // Do nothing.
      break;
    case mjNGEOMTYPES:
      mju_warning("Unsupported geom type: %d", geom.type);
      break;
  }

  mjrf_setRenderablePose(renderable, position, rotation);
}

static void UpdateGeomMaterial(mjrfRenderable* renderable, const mjvGeom& geom,
                               ModelResources* model_objs) {
  const mjModel* model = model_objs->GetModel();

  mjrfMaterial material;
  mjrf_defaultMaterial(&material);

  if (geom.category == mjCAT_DECOR) {
    material.decor = true;
  }

  material.color[0] = geom.rgba[0];
  material.color[1] = geom.rgba[1];
  material.color[2] = geom.rgba[2];
  material.color[3] = geom.rgba[3];

  if (geom.matid >= 0 && geom.matid < model->nmat) {
    auto get_texture = [&](int role) -> const mjrfTexture* {
      const int tex_id = model->mat_texid[geom.matid * mjNTEXROLE + role];
      return tex_id >= 0 ? model_objs->GetTexture(tex_id) : nullptr;
    };
    material.color_texture = get_texture(mjTEXROLE_RGB);
    material.normal_texture = get_texture(mjTEXROLE_NORMAL);
    material.opacity_texture = get_texture(mjTEXROLE_OPACITY);
    material.emissive_texture = get_texture(mjTEXROLE_EMISSIVE);
    material.orm_texture = get_texture(mjTEXROLE_ORM);
    material.metallic_texture = get_texture(mjTEXROLE_METALLIC);
    material.roughness_texture = get_texture(mjTEXROLE_ROUGHNESS);
    material.occlusion_texture = get_texture(mjTEXROLE_OCCLUSION);
  }
  if (geom.texid >= 0) {
    material.color_texture = model_objs->GetTexture(geom.texid);
  }

  material.reflectance = geom.reflectance;
  material.emissive = geom.emission;
  material.specular = geom.specular;
  material.glossiness = geom.shininess;
  if (geom.matid >= 0) {
    material.metallic = model->mat_metallic[geom.matid];
    material.roughness = model->mat_roughness[geom.matid];
  }

  // Filament segmentation ids are 1-based (0 is the background), matching the
  // classic renderer's mjRND_IDCOLOR, which colors each geom by segid + 1.
  material.segmentation_id = geom.segid + 1;

  // Assume an emissive object is a selected object.
  if (geom.emission > 0 && geom.emission == model->vis.global.glow) {
    material.selected = true;
    material.emissive = 0.0f;
  }

  // UvScale only applies to objects that don't have explicit UV coordinates
  // in their vertex buffer. Instead, we set the UV coordinate to be the same
  // as the vertex position.
  //
  // The material's `texuniform` and `texrepeat` parameters allow us to scale
  // the programmatic UVs.

  if (material.color_texture) {
    bool tex_uniform;
    float tex_repeat[2];
    if (geom.texid >= 0) {
      tex_uniform = geom.texuniform;
      tex_repeat[0] = geom.texrepeat[0];
      tex_repeat[1] = geom.texrepeat[1];
    } else {
      tex_uniform = model->mat_texuniform[geom.matid];
      tex_repeat[0] = model->mat_texrepeat[(geom.matid * 2) + 0];
      tex_repeat[1] = model->mat_texrepeat[(geom.matid * 2) + 1];
    }
    if (mjrf_getTextureSamplerType(material.color_texture) == mjTEXTURE_2D) {
      // For 2D textures, `tex_repeat` specifies how many times the texture
      // image is repeated. The `tex_uniform` flag determines if the repetition
      // is applied at in object space (false) or in world space (true).
      material.uv_scale[0] = tex_repeat[0];
      material.uv_scale[1] = tex_repeat[1];

      const bool is_infinite_plane =
          geom.type == mjGEOM_PLANE && (geom.size[0] <= 0 || geom.size[1] <= 0);
      if (is_infinite_plane) {
        const float plane_scale = static_cast<float>(mjMAXPLANEGRID) / 2.0f;

        const float dot_pos_axis_x = geom.pos[0] * geom.mat[0] +
                                     geom.pos[1] * geom.mat[3] +
                                     geom.pos[2] * geom.mat[6];
        const float dot_pos_axis_y = geom.pos[0] * geom.mat[1] +
                                     geom.pos[1] * geom.mat[4] +
                                     geom.pos[2] * geom.mat[7];

        material.uv_scale[0] *= plane_scale;
        material.uv_scale[1] *= plane_scale;
        // The vertex UVs in PlaneBuilder are u0 = 0.5*x + 0.5, v0 = -0.5*y + 0.5.
        // To keep the world-space texture coordinate u = 0.5*worldX*texrepeat - 0.5
        // independent of the snapped geomPos, uv_offset must compensate by 0.5*dot_pos.
        material.uv_offset[0] =
            (0.5f * dot_pos_axis_x - 0.5f * plane_scale) * tex_repeat[0] - 0.5f;
        material.uv_offset[1] =
            (-0.5f * dot_pos_axis_y - 0.5f * plane_scale) * tex_repeat[1] - 0.5f;
      } else {
        if ((geom.type == mjGEOM_HFIELD || geom.type == mjGEOM_MESH ||
             geom.type == mjGEOM_SDF) &&
            geom.dataid >= 0 && !geom.texcoord) {
          if (geom.size[0] > 0) {
            material.uv_scale[0] /= mju_max(mjMINVAL, geom.size[0]);
          }
          if (geom.size[1] > 0) {
            material.uv_scale[1] /= mju_max(mjMINVAL, geom.size[1]);
          }
          // Adjust the offset to the origin to match OpenGL's glTexGen.
          material.uv_offset[0] = -0.5f;
          material.uv_offset[1] = -0.5f;
        }
        if (tex_uniform && geom.type != mjGEOM_FLEX &&
            geom.type != mjGEOM_SKIN) {
          if (geom.size[0] > 0) {
            material.uv_scale[0] *= geom.size[0];
          }
          if (geom.size[1] > 0) {
            material.uv_scale[1] *= geom.size[1];
          }
        }
      }
    } else {
      // For cube maps, if `tex_uniform` is true, then scale the texture so that
      // it covers a 1x1 area of world space rather than the area of the object.
      if (tex_uniform) {
        material.uv_scale[0] = 1.0f / (geom.size[0] ? geom.size[0] : 1.0f);
        material.uv_scale[1] = 1.0f / (geom.size[1] ? geom.size[1] : 1.0f);
        material.uv_scale[2] = 1.0f / (geom.size[2] ? geom.size[2] : 1.0f);
      }
    }
  }

  // Apply material multipliers from the model.
  material.emissive *= model_objs->GetEmissiveMultiplier();
  material.specular *= model_objs->GetSpecularMultiplier();
  material.glossiness *= model_objs->GetShininessMultiplier();

  mjrf_setRenderableMaterial(renderable, &material);
}

UniquePtr<mjrfRenderable> CreateGeomRenderable(const mjvGeom& geom,
                                               mjrfContext* ctx,
                                               ModelResources* model_objs,
                                               SceneObjects* scene_objs) {
  mjrfRenderableParams params;
  mjrf_defaultRenderableParams(&params);
  params.cast_shadow = geom.castshadow;
  auto renderable = CreateRenderable(ctx, params);
  PrepareGeomMeshes(renderable.get(), geom, model_objs, scene_objs);
  UpdateGeomMaterial(renderable.get(), geom, model_objs);
  return renderable;
}
}  // namespace mujoco
