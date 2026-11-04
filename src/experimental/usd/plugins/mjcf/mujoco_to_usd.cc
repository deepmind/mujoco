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

#include "mjcf/mujoco_to_usd.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <mujoco/experimental/usd/mjcPhysics/tokens.h>
#include <mujoco/experimental/usd/utils.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "mjcf/utils.h"
#include <pxr/base/arch/attributes.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/callContext.h>
#include <pxr/base/tf/diagnostic.h>
#include <pxr/base/tf/diagnosticHelper.h>
#include <pxr/base/tf/enum.h>
#include <pxr/base/tf/registryManager.h>
#include <pxr/base/tf/staticData.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/dictionary.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/kind/registry.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/declareHandles.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/sdf/valueTypeName.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdLux/tokens.h>
#include <pxr/usd/usdPhysics/fixedJoint.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/prismaticJoint.h>
#include <pxr/usd/usdPhysics/revoluteJoint.h>
#include <pxr/usd/usdPhysics/sphericalJoint.h>
#include <pxr/usd/usdPhysics/tokens.h>
#include <pxr/usd/usdShade/tokens.h>

namespace {

// The ID of the World in mjModel and mjData.
static constexpr int kWorldIndex = 0;

// Using to satisfy TF_DEFINE_PRIVATE_TOKENS macro below and avoid operating in
// PXR_NS.
using pxr::TfToken;
template <typename T>
using TfStaticData = pxr::TfStaticData<T>;

// clang-format off
TF_DEFINE_PRIVATE_TOKENS(kTokens,
                         ((body, "Body"))
                         ((body_name, "mujoco:body_name"))
                         ((geom, "Geom"))
                         ((light, "Light"))
                         ((meshScope, "MeshSources"))
                         ((materialsScope, "Materials"))
                         ((physicsMaterialsScope, "PhysicsMaterials"))
                         ((previewSurface, "PreviewSurface"))
                         ((keyframesScope, "Keyframes"))
                         ((actuatorsScope, "Actuators"))
                         ((keyframe, "Keyframe"))
                         ((surface, "PreviewSurface"))
                         ((world, "World"))
                         ((xformOpTransform, "xformOp:transform"))
                         ((xformOpScale, "xformOp:scale"))
                         (st)
                         ((primvarsSt, "primvars:st"))
                         ((outputsSt, "outputs:st"))
                         ((inputsSt, "inputs:st"))
                         ((inputsVarname, "inputs:varname"))
                         ((inputsFile, "inputs:file"))
                         ((inputsWrapS, "inputs:wrapS"))
                         ((inputsWrapT, "inputs:wrapT"))
                         ((inputsDiffuseColor, "inputs:diffuseColor"))
                         ((inputsEmissiveColor, "inputs:emissiveColor"))
                         ((outputsRgb, "outputs:rgb"))
                         ((outputsR, "outputs:r"))
                         ((outputsG, "outputs:g"))
                         ((outputsB, "outputs:b"))
                         ((inputsMetallic, "inputs:metallic"))
                         ((inputsOcclusion, "inputs:occlusion"))
                         ((inputsRoughness, "inputs:roughness"))
                         (repeat)
                         ((sourceMesh, pxr::UsdGeomTokens->Mesh))
                         ((inputsNormal, "inputs:normal"))
                         ((joint, "Joint"))
                         (UsdPrimvarReader_float2)
                         (UsdUVTexture)
                         (UsdPreviewSurface)
                         ((NewtonMaterialAPI, "NewtonMaterialAPI"))
                         ((NewtonMeshCollisionAPI, "NewtonMeshCollisionAPI"))
                         ((NewtonJointAPI, "NewtonJointAPI"))
                         ((NewtonMassAPI, "NewtonMassAPI"))
                         ((newtonTorsionalFriction, "newton:torsionalFriction"))
                         ((newtonRollingFriction, "newton:rollingFriction"))
                         ((newtonArmature, "newton:armature"))
                         ((newtonDamping, "newton:damping"))
                         ((newtonFriction, "newton:friction"))
                         ((newtonMassModel, "newton:massModel"))
                         ((newtonMaxHullVertices, "newton:maxHullVertices"))
                         ((newtonMaxSolverIterations, "newton:maxSolverIterations"))
                         ((newtonTimeStepsPerSecond, "newton:timeStepsPerSecond"))
                         ((newtonGravityEnabled, "newton:gravityEnabled"))
                         ((newtonContactMargin, "newton:contactMargin"))
                         ((newtonContactGap, "newton:contactGap"))
                         ((newtonContactAdhesion, "newton:contactAdhesion"))
                        );

// Using to satisfy TF_REGISTRY_FUNCTION macro below and avoid operating in PXR_NS.
using pxr::TfEnum;
using pxr::Tf_RegistryStaticInit;
using pxr::Tf_RegistryInit;
using pxr::TfEnum;
template <typename T>
using Arch_PerLibInit = pxr::Arch_PerLibInit<T>;
#if defined(ARCH_OS_DARWIN)
using Arch_ConstructorEntry = pxr::Arch_ConstructorEntry;
#endif
enum ErrorCodes { UnsupportedActuatorTypeError, UnsupportedGeomTypeError, MujocoCompilationError };

TF_REGISTRY_FUNCTION(pxr::TfEnum) {
  TF_ADD_ENUM_NAME(UnsupportedGeomTypeError, "UsdGeom type is unsupported.")
  TF_ADD_ENUM_NAME(MujocoCompilationError, "Mujoco spec failed to compile.")
}

// Usings to satisfy TF_ERROR macro.
using pxr::TfCallContext;
using pxr::Tf_PostErrorHelper;
// clang-format on

using pxr::MjcPhysicsTokens;

using mujoco::usd::AddAttributeConnection;
using mujoco::usd::AddPrimInherit;
using mujoco::usd::AddPrimReference;
using mujoco::usd::ApplyApiSchema;
using mujoco::usd::CreateAttributeSpec;
using mujoco::usd::CreateClassSpec;
using mujoco::usd::CreatePrimSpec;
using mujoco::usd::CreateRelationshipSpec;
using mujoco::usd::SetAttributeDefault;
using mujoco::usd::SetAttributeTimeSample;
using mujoco::usd::SetLayerMetadata;
using mujoco::usd::SetPrimKind;
using mujoco::usd::SetPrimPurpose;

pxr::GfMatrix4d MujocoPosQuatToTransform(mjtNum *pos, mjtNum *quat) {
  pxr::GfQuatd quaternion = pxr::GfQuatd::GetIdentity();
  quaternion.SetReal(quat[0]);
  quaternion.SetImaginary(quat[1], quat[2], quat[3]);
  pxr::GfRotation rotation(quaternion);

  pxr::GfVec3d translation(0.0, 0.0, 0.0);
  translation.Set(pos[0], pos[1], pos[2]);

  pxr::GfMatrix4d transform;
  transform.SetTransform(rotation, translation);
  return transform;
}

}  // namespace

class ModelWriter {
 public:
  ModelWriter(mjSpec* spec, mjModel* model, pxr::SdfLayerRefPtr layer,
              bool skip_elems_from_usd)
      : spec_(spec),
        model_(model),
        layer_(layer),
        skip_elems_from_usd_(skip_elems_from_usd),
        class_path_("/Bad_Path") {
    body_paths_ = std::vector<pxr::SdfPath>(model->nbody);
    site_paths_ = std::vector<pxr::SdfPath>(model->nsite);
    joint_paths_ = std::vector<pxr::SdfPath>(model->njnt);
  }
  ~ModelWriter() { mj_deleteModel(model_); }

  bool ShouldWrite(mjsElement* element) {
    // If we've been told to skip over spec elements that originated from USD,
    // check the user value and if it has a primpath then it came from decoding
    // a USD stage into an MjSpec.
    return !skip_elems_from_usd_ || mujoco::usd::GetUsdPrimPathUserValue(element).IsEmpty();
  }

  void Write() {
    // Create top level class holder.
    class_path_ = CreateClassSpec(layer_, pxr::SdfPath::AbsoluteRootPath(),
                                  pxr::TfToken("__class__"))->GetPath();

    // Create the world body.
    body_paths_[kWorldIndex] = WriteWorldBody(kWorldIndex)->GetPath();

    layer_->SetDocumentation("Generated by mujoco model writer.");
    // Set the world body to be the default prim for referencing/payloads.
    layer_->SetDefaultPrim(body_paths_[kWorldIndex].GetNameToken());
    // Mujoco is Z up by default.
    SetLayerMetadata(layer_, pxr::UsdGeomTokens->upAxis, pxr::UsdGeomTokens->z);
    // Mujoco is authored in meters by default.
    SetLayerMetadata(layer_, pxr::UsdGeomTokens->metersPerUnit,
                     pxr::UsdGeomLinearUnits::meters);

    WritePhysicsScene();

    // Author mesh scope + mesh prims to be referenced.
    WriteMeshes();
    WriteMaterials();
    WriteBodies();
    WriteActuators();
    WriteKeyframes();
  }

 private:
  mjSpec *spec_;
  mjModel *model_;
  pxr::SdfLayerRefPtr layer_;

  bool skip_elems_from_usd_ = false;
  // Path to top level class spec that all classes should be children of.
  pxr::SdfPath class_path_;
  // Mapping from Mujoco body id to SdfPath.
  std::vector<pxr::SdfPath> body_paths_;
  // Mapping from Mujoco site id to SdfPath.
  std::vector<pxr::SdfPath> site_paths_;
  // Mapping from Mujoco joint id to SdfPath.
  std::vector<pxr::SdfPath> joint_paths_;
  // Mapping from mesh names to Mesh prim path.
  std::unordered_map<std::string, pxr::SdfPath> mesh_paths_;
  // Set of body ids that have had the articulation root API applied.
  std::unordered_set<int> articulation_roots_;

  // Given a name index and a parent prim path this returns a
  // token such that appending it to the parent prim path does not
  // identify an existing prim spec.
  //
  // This is necessary since mujoco does not require names for elements
  // so we must differentiate between elements of the same type.
  //
  // For example:
  // <mujoco model="model with two planes">
  //    <worldbody>
  //      <geom type="plane"/>
  //      <geom type="plane"/>
  //    </worldbody>
  //  </mujoco>
  //
  //  We expect that the occurrence of this happens little enough that linear
  //  searching is plenty efficient.
  pxr::TfToken GetAvailablePrimName(const std::string base_name,
                                    const pxr::TfToken fallback_name,
                                    const pxr::SdfPath &parent_path) {
    const auto valid_base_name = pxr::TfMakeValidIdentifier(
        base_name.empty() ? fallback_name : base_name);
    std::string name = valid_base_name;
    pxr::SdfPath test_path = parent_path.AppendChild(pxr::TfToken(name));
    int count = 1;
    while (layer_->HasSpec(test_path) &&
           layer_->GetSpecType(test_path) == pxr::SdfSpecType::SdfSpecTypePrim) {
      name = pxr::TfStringPrintf("%s_%d", valid_base_name.c_str(), count++);
      test_path = parent_path.AppendChild(pxr::TfToken(name));
    }
    return pxr::TfToken(name);
  }

  // This function, conversely to GetAvailablePrimName will not handle
  // collisions. This is useful when looking up a prim path that might exist or
  // a path that you know must be unique.
  pxr::TfToken GetValidPrimName(const std::string name) {
    return pxr::TfToken(pxr::TfMakeValidIdentifier(name));
  }

  struct BodyPathComponents {
    pxr::SdfPath parent_path;
    pxr::TfToken body_name;
  };

  void WriteScaleXformOp(const pxr::SdfPrimSpecHandle prim_spec,
                         const pxr::GfVec3f &scale) {
    auto scale_attr_spec =
        CreateAttributeSpec(layer_, prim_spec, kTokens->xformOpScale,
                            pxr::SdfValueTypeNames->Float3);
    SetAttributeDefault(layer_, scale_attr_spec, scale);
  }

  void WriteTransformXformOp(const pxr::SdfPrimSpecHandle prim_spec,
                             const pxr::GfMatrix4d &transform) {
    auto transform_op_spec =
        CreateAttributeSpec(layer_, prim_spec, kTokens->xformOpTransform,
                            pxr::SdfValueTypeNames->Matrix4d);
    SetAttributeDefault(layer_, transform_op_spec, transform);
  }

  void WriteXformOpOrder(const pxr::SdfPrimSpecHandle prim_spec,
                         const pxr::VtArray<pxr::TfToken> &order) {
    auto xform_op_order_spec =
        CreateAttributeSpec(layer_, prim_spec, pxr::UsdGeomTokens->xformOpOrder,
                            pxr::SdfValueTypeNames->TokenArray);
    SetAttributeDefault(layer_, xform_op_order_spec, order);
  }

  template <typename T>
  void WriteUniformAttribute(const pxr::SdfPrimSpecHandle prim_spec,
                             const pxr::SdfValueTypeName &value_type_name,
                             const pxr::TfToken &token, const T &value) {
    auto attr_spec = CreateAttributeSpec(
        layer_, prim_spec, token, value_type_name, pxr::SdfVariabilityUniform);
    SetAttributeDefault(layer_, attr_spec, value);
  }

  template <typename T>
  void WriteColorAndOpacityAttributes(const pxr::SdfPrimSpecHandle prim_spec,
                                      const T &element) {
    // If rgba is not the default (0.5, 0.5, 0.5, 1), then set the
    // displayColor attribute.
    // No effort is made to properly handle the interaction between rgba
    // and the material if both are specified.
    if (element->rgba[0] != 0.5f || element->rgba[1] != 0.5f ||
        element->rgba[2] != 0.5f || element->rgba[3] != 1.0f) {
      // Set the displayColor attribute.
      auto display_color_attr = CreateAttributeSpec(
          layer_, prim_spec, pxr::UsdGeomTokens->primvarsDisplayColor,
          pxr::SdfValueTypeNames->Color3fArray);
      SetAttributeDefault(
          layer_, display_color_attr,
          pxr::VtArray<pxr::GfVec3f>{
              {element->rgba[0], element->rgba[1], element->rgba[2]}});
      // Set the displayOpacity attribute, only if the opacity is not 1.
      if (element->rgba[3] != 1.0f) {
        auto display_opacity_attr = CreateAttributeSpec(
            layer_, prim_spec, pxr::UsdGeomTokens->primvarsDisplayOpacity,
            pxr::SdfValueTypeNames->FloatArray);
        SetAttributeDefault(layer_, display_opacity_attr,
                            pxr::VtArray<float>{element->rgba[3]});
      }
    }
  }

  void PrependToXformOpOrder(const pxr::SdfPrimSpecHandle prim_spec,
                             const pxr::VtArray<pxr::TfToken> &order) {
    auto xform_op_order_path =
        prim_spec->GetPath().AppendProperty(pxr::UsdGeomTokens->xformOpOrder);
    auto attr_spec = layer_->GetAttributeAtPath(xform_op_order_path);
    if (!attr_spec || !attr_spec->HasDefaultValue()) {
      WriteXformOpOrder(prim_spec, order);
      return;
    }

    auto existing_order =
        attr_spec->GetDefaultValue().Get<pxr::VtArray<pxr::TfToken>>();

    pxr::VtArray<pxr::TfToken> new_order(order.size() + existing_order.size());
    std::copy(order.begin(), order.end(), new_order.begin());
    std::copy(existing_order.begin(), existing_order.end(),
              new_order.begin() + order.size());

    SetAttributeDefault(layer_, attr_spec, new_order);
  }

  void WriteMesh(const mjsMesh *mesh, const pxr::SdfPath &parent_path) {
    auto name = GetAvailablePrimName(*mjs_getName(mesh->element),
                                     pxr::UsdGeomTokens->Mesh, parent_path);
    auto subcomponent_spec =
        CreatePrimSpec(layer_, parent_path, name, pxr::UsdGeomTokens->Xform);
    auto mesh_spec =
        CreatePrimSpec(layer_, subcomponent_spec->GetPath(), kTokens->sourceMesh,
                       pxr::UsdGeomTokens->Mesh);
    mesh_paths_[*mjs_getName(mesh->element)] = subcomponent_spec->GetPath();

    ApplyApiSchema(layer_, mesh_spec, MjcPhysicsTokens->MjcMeshCollisionAPI);

    pxr::TfToken inertia = MjcPhysicsTokens->legacy;
    if (mesh->inertia == mjtMeshInertia::mjMESH_INERTIA_EXACT) {
      inertia = MjcPhysicsTokens->exact;
    } else if (mesh->inertia == mjtMeshInertia::mjMESH_INERTIA_CONVEX) {
      inertia = MjcPhysicsTokens->convex;
    } else if (mesh->inertia == mjtMeshInertia::mjMESH_INERTIA_SHELL) {
      inertia = MjcPhysicsTokens->shell;
    }

    WriteUniformAttribute(mesh_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcInertia, inertia);

    // Newton mass model attribute (partially replaces deprecated mjc:inertia)
    if (mesh->inertia == mjtMeshInertia::mjMESH_INERTIA_SHELL) {
      ApplyApiSchema(layer_, mesh_spec, kTokens->NewtonMassAPI);
      WriteUniformAttribute(mesh_spec, pxr::SdfValueTypeNames->Token,
                            kTokens->newtonMassModel, MjcPhysicsTokens->shell);
    }

    // Newton mesh attribute (replaces deprecated mjc:maxhullvert)
    if (mesh->maxhullvert != -1) {
      WriteUniformAttribute(mesh_spec, pxr::SdfValueTypeNames->Int,
                            kTokens->newtonMaxHullVertices, mesh->maxhullvert);
    }

    // NOTE: The geometry data taken from the spec is the post-compilation
    // data after it has been mjCMesh::Compile'd. So don't be surprised if
    // things like user defined vertices have moved due to re-centering to
    // CoM and other modifications (see mjCMesh::Process for other xforms).
    int mesh_id = mjs_getId(mesh->element);
    int vert_start_offset = model_->mesh_vertadr[mesh_id] * 3;
    int nvert = model_->mesh_vertnum[mesh_id];
    pxr::VtArray<pxr::GfVec3f> points;
    points.reserve(nvert);
    for (int i = vert_start_offset; i < vert_start_offset + nvert * 3; i += 3) {
      points.emplace_back(&model_->mesh_vert[i]);
    }

    auto points_attr =
        CreateAttributeSpec(layer_, mesh_spec, pxr::UsdGeomTokens->points,
                            pxr::SdfValueTypeNames->Vector3fArray);
    SetAttributeDefault(layer_, points_attr, points);

    // NOTE: nface is never 0.
    int nface = model_->mesh_facenum[mesh_id];
    pxr::VtArray<int> faces;
    faces.reserve(nface * 3);
    int face_start_offset = model_->mesh_faceadr[mesh_id] * 3;
    for (int i = face_start_offset; i < face_start_offset + nface * 3; i += 3) {
      faces.push_back(model_->mesh_face[i]);
      faces.push_back(model_->mesh_face[i + 1]);
      faces.push_back(model_->mesh_face[i + 2]);
    }
    auto face_vertex_idx_attr= CreateAttributeSpec(
        layer_, mesh_spec, pxr::UsdGeomTokens->faceVertexIndices,
        pxr::SdfValueTypeNames->IntArray);
    SetAttributeDefault(layer_, face_vertex_idx_attr, faces);

    pxr::VtArray<int> vertex_counts;
    for (int i = 0; i < nface; ++i) {
      // Mujoco is always triangles.
      vertex_counts.push_back(3);
    }
    auto face_vertex_counts_attr= CreateAttributeSpec(
        layer_, mesh_spec, pxr::UsdGeomTokens->faceVertexCounts,
        pxr::SdfValueTypeNames->IntArray);
    SetAttributeDefault(layer_, face_vertex_counts_attr, vertex_counts);

    if (model_->mesh_normalnum[mesh_id]) {
      // We have to convert from Mujoco's indexed normals to USD's faceVarying
      // normals.
      pxr::VtArray<pxr::GfVec3f> normals;
      normals.reserve(nface * 3);
      int normal_start_adr = model_->mesh_normaladr[mesh_id];
      int face_start_offset = model_->mesh_faceadr[mesh_id] * 3;
      for (int i = face_start_offset; i < face_start_offset + nface * 3; ++i) {
        int normal_adr = normal_start_adr + model_->mesh_facenormal[i];
        normals.emplace_back(&model_->mesh_normal[normal_adr * 3]);
      }
      auto normals_attr=
          CreateAttributeSpec(layer_, mesh_spec, pxr::UsdGeomTokens->normals,
                              pxr::SdfValueTypeNames->Vector3fArray);
      SetAttributeDefault(layer_, normals_attr, normals);
      normals_attr->SetField(pxr::UsdGeomTokens->interpolation,
                           pxr::UsdGeomTokens->faceVarying);
    }

    if (model_->mesh_texcoordnum[mesh_id]) {
      // We have to convert from Mujoco's indexed texcoords to USD's faceVarying
      // texcoords.
      pxr::VtArray<pxr::GfVec2f> texcoords;
      texcoords.reserve(nface * 3);
      int texcoord_start_adr = model_->mesh_texcoordadr[mesh_id];
      int face_start_offset = model_->mesh_faceadr[mesh_id] * 3;
      for (int i = face_start_offset; i < face_start_offset + nface * 3; ++i) {
        int texcoord_adr = texcoord_start_adr + model_->mesh_facetexcoord[i];
        // Invert the V coordinate, Mujoco assumes OpenGL 0,0 is top left.
        // But USD UVs use image bottom left 0,0 convention.
        pxr::GfVec2f uv(&model_->mesh_texcoord[texcoord_adr * 2]);
        uv[1] = 1.0f - uv[1];
        texcoords.push_back(uv);
      }

      auto texcoords_attr=
          CreateAttributeSpec(layer_, mesh_spec, kTokens->primvarsSt,
                              pxr::SdfValueTypeNames->TexCoord2fArray);
      SetAttributeDefault(layer_, texcoords_attr, texcoords);
      texcoords_attr->SetField(pxr::UsdGeomTokens->interpolation,
                           pxr::UsdGeomTokens->faceVarying);
    }

    // Default subdivision scheme is catmull clark so explicitly set it
    // to none here.
    auto subdivision_scheme= CreateAttributeSpec(
        layer_, mesh_spec, pxr::UsdGeomTokens->subdivisionScheme,
        pxr::SdfValueTypeNames->Token);
    SetAttributeDefault(layer_, subdivision_scheme,
                        pxr::UsdGeomTokens->none);
  }

  void WritePhysicsScene() {
    auto physics_scene_spec = CreatePrimSpec(
        layer_, body_paths_[kWorldIndex], pxr::UsdPhysicsTokens->PhysicsScene,
        pxr::UsdPhysicsTokens->PhysicsScene);

    ApplyApiSchema(layer_, physics_scene_spec, MjcPhysicsTokens->MjcSceneAPI);

    const std::vector<std::pair<pxr::TfToken, double>>
        option_double_attributes = {
            // mjc:option:timestep deprecated in favor of newton:timeStepsPerSecond
            {MjcPhysicsTokens->mjcOptionTolerance, spec_->option.tolerance},
            {MjcPhysicsTokens->mjcOptionLs_tolerance,
             spec_->option.ls_tolerance},
            {MjcPhysicsTokens->mjcOptionNoslip_tolerance,
             spec_->option.noslip_tolerance},
            {MjcPhysicsTokens->mjcOptionCcd_tolerance,
             spec_->option.ccd_tolerance},
            {MjcPhysicsTokens->mjcOptionSleep_tolerance,
             spec_->option.sleep_tolerance},
            {MjcPhysicsTokens->mjcOptionImpratio, spec_->option.impratio},
            {MjcPhysicsTokens->mjcOptionDensity, spec_->option.density},
            {MjcPhysicsTokens->mjcOptionViscosity, spec_->option.viscosity},
            {MjcPhysicsTokens->mjcOptionO_margin, spec_->option.o_margin},
        };
    for (const auto &[token, value] : option_double_attributes) {
      WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double,
                            token, value);
    }

    const std::vector<std::pair<pxr::TfToken, int>> option_int_attributes = {
        // mjc:option:iterations deprecated in favor of newton:maxSolverIterations
        {MjcPhysicsTokens->mjcOptionLs_iterations, spec_->option.ls_iterations},
        {MjcPhysicsTokens->mjcOptionNoslip_iterations,
         spec_->option.noslip_iterations},
        {MjcPhysicsTokens->mjcOptionCcd_iterations,
         spec_->option.ccd_iterations},
        {MjcPhysicsTokens->mjcOptionSdf_iterations,
         spec_->option.sdf_iterations},
        {MjcPhysicsTokens->mjcOptionSdf_initpoints,
         spec_->option.sdf_initpoints},
    };
    for (const auto &[token, value] : option_int_attributes) {
      WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Int,
                            token, value);
    }

    auto cone_attr = CreateAttributeSpec(
        layer_, physics_scene_spec, MjcPhysicsTokens->mjcOptionCone,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);

    switch (spec_->option.cone) {
      case mjCONE_PYRAMIDAL:
        SetAttributeDefault(layer_, cone_attr, MjcPhysicsTokens->pyramidal);
        break;
      case mjCONE_ELLIPTIC:
        SetAttributeDefault(layer_, cone_attr, MjcPhysicsTokens->elliptic);
        break;
      default:
        break;
    }

    auto jacobian_attr = CreateAttributeSpec(
        layer_, physics_scene_spec, MjcPhysicsTokens->mjcOptionJacobian,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);

    switch (spec_->option.jacobian) {
      case mjJAC_AUTO:
        SetAttributeDefault(layer_, jacobian_attr, MjcPhysicsTokens->auto_);
        break;
      case mjJAC_DENSE:
        SetAttributeDefault(layer_, jacobian_attr, MjcPhysicsTokens->dense);
        break;
      case mjJAC_SPARSE:
        SetAttributeDefault(layer_, jacobian_attr, MjcPhysicsTokens->sparse);
        break;
      default:
        break;
    }

    auto solver_attr = CreateAttributeSpec(
        layer_, physics_scene_spec, MjcPhysicsTokens->mjcOptionSolver,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
    switch (spec_->option.solver) {
      case mjSOL_NEWTON:
        SetAttributeDefault(layer_, solver_attr, MjcPhysicsTokens->newton);
        break;
      case mjSOL_PGS:
        SetAttributeDefault(layer_, solver_attr, MjcPhysicsTokens->pgs);
        break;
      case mjSOL_CG:
        SetAttributeDefault(layer_, solver_attr, MjcPhysicsTokens->cg);
        break;
      default:
        break;
    }

    pxr::GfVec3f gravity(spec_->option.gravity[0], spec_->option.gravity[1],
                         spec_->option.gravity[2]);
    // Normalize will normalize gravity in place and return the magnitude before
    // normalization.
    float gravity_magnitude = gravity.Normalize();

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Float,
                          pxr::UsdPhysicsTokens->physicsGravityMagnitude,
                          gravity_magnitude);
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Vector3f,
                          pxr::UsdPhysicsTokens->physicsGravityDirection,
                          gravity);

    pxr::GfVec3d wind(spec_->option.wind[0], spec_->option.wind[1],
                      spec_->option.wind[2]);
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double3,
                          MjcPhysicsTokens->mjcOptionWind, wind);

    pxr::GfVec3d magnetic(spec_->option.magnetic[0], spec_->option.magnetic[1],
                          spec_->option.magnetic[2]);
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double3,
                          MjcPhysicsTokens->mjcOptionMagnetic, magnetic);

    pxr::VtArray<double> o_solref(spec_->option.o_solref,
                                  spec_->option.o_solref + mjNREF);
    WriteUniformAttribute(physics_scene_spec,
                          pxr::SdfValueTypeNames->DoubleArray,
                          MjcPhysicsTokens->mjcOptionO_solref, o_solref);

    pxr::VtArray<double> o_solimp(spec_->option.o_solimp,
                                  spec_->option.o_solimp + mjNIMP);
    WriteUniformAttribute(physics_scene_spec,
                          pxr::SdfValueTypeNames->DoubleArray,
                          MjcPhysicsTokens->mjcOptionO_solimp, o_solimp);

    pxr::VtArray<double> o_friction(spec_->option.o_friction,
                                    spec_->option.o_friction + 5);
    WriteUniformAttribute(physics_scene_spec,
                          pxr::SdfValueTypeNames->DoubleArray,
                          MjcPhysicsTokens->mjcOptionO_friction, o_friction);

    pxr::VtArray<int> actuatorgroupdisable;
    for (int i = 0; i < mjNGROUP; ++i) {
      if (spec_->option.disableactuator & (1 << i)) {
        actuatorgroupdisable.push_back(i);
      }
    }
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->IntArray,
                          MjcPhysicsTokens->mjcOptionActuatorgroupdisable,
                          actuatorgroupdisable);

    auto integrator_attr = CreateAttributeSpec(
        layer_, physics_scene_spec, MjcPhysicsTokens->mjcOptionIntegrator,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
    switch (spec_->option.integrator) {
      case mjINT_EULER:
        SetAttributeDefault(layer_, integrator_attr, MjcPhysicsTokens->euler);
        break;
      case mjINT_RK4:
        SetAttributeDefault(layer_, integrator_attr, MjcPhysicsTokens->rk4);
        break;
      default:
        break;
    }

    auto create_flag_attr = [&](pxr::TfToken token, int flag, bool enable) {
      int flags =
          enable ? spec_->option.enableflags : spec_->option.disableflags;
      bool value = enable ? (flags & flag) : !(flags & flag);
      WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                            token, value);
    };

    const std::vector<std::pair<pxr::TfToken, int>> enable_flags = {
        {MjcPhysicsTokens->mjcFlagFwdinv, mjENBL_FWDINV},
        {MjcPhysicsTokens->mjcFlagEnergy, mjENBL_ENERGY},
        {MjcPhysicsTokens->mjcFlagOverride, mjENBL_OVERRIDE},
        {MjcPhysicsTokens->mjcFlagInvdiscrete, mjENBL_INVDISCRETE},
        {MjcPhysicsTokens->mjcFlagSleep, mjENBL_SLEEP}};
    for (const auto &[token, flag] : enable_flags) {
      create_flag_attr(token, flag, true);
    }

    const std::vector<std::pair<pxr::TfToken, int>> disable_flags = {
        {MjcPhysicsTokens->mjcFlagConstraint, mjDSBL_CONSTRAINT},
        {MjcPhysicsTokens->mjcFlagEquality, mjDSBL_EQUALITY},
        {MjcPhysicsTokens->mjcFlagFrictionloss, mjDSBL_FRICTIONLOSS},
        {MjcPhysicsTokens->mjcFlagLimit, mjDSBL_LIMIT},
        {MjcPhysicsTokens->mjcFlagContact, mjDSBL_CONTACT},
        {MjcPhysicsTokens->mjcFlagSpring, mjDSBL_SPRING},
        {MjcPhysicsTokens->mjcFlagDamper, mjDSBL_DAMPER},
        // mjc:flag:gravity deprecated in favor of newton:gravityEnabled
        {MjcPhysicsTokens->mjcFlagClampctrl, mjDSBL_CLAMPCTRL},
        {MjcPhysicsTokens->mjcFlagWarmstart, mjDSBL_WARMSTART},
        {MjcPhysicsTokens->mjcFlagFilterparent, mjDSBL_FILTERPARENT},
        {MjcPhysicsTokens->mjcFlagActuation, mjDSBL_ACTUATION},
        {MjcPhysicsTokens->mjcFlagRefsafe, mjDSBL_REFSAFE},
        {MjcPhysicsTokens->mjcFlagSensor, mjDSBL_SENSOR},
        {MjcPhysicsTokens->mjcFlagMidphase, mjDSBL_MIDPHASE},
        {MjcPhysicsTokens->mjcFlagEulerdamp, mjDSBL_EULERDAMP},
        {MjcPhysicsTokens->mjcFlagAutoreset, mjDSBL_AUTORESET},
        {MjcPhysicsTokens->mjcFlagNativeccd, mjDSBL_NATIVECCD},
        {MjcPhysicsTokens->mjcFlagIsland, mjDSBL_ISLAND},
        {MjcPhysicsTokens->mjcFlagMulticcd, mjDSBL_MULTICCD}};
    for (const auto &[token, flag] : disable_flags) {
      create_flag_attr(token, flag, false);
    }

    // Compiler attributes
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerAutoLimits,
                          (bool)spec_->compiler.autolimits);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double,
                          MjcPhysicsTokens->mjcCompilerBoundMass,
                          spec_->compiler.boundmass);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double,
                          MjcPhysicsTokens->mjcCompilerBoundInertia,
                          spec_->compiler.boundinertia);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Double,
                          MjcPhysicsTokens->mjcCompilerSetTotalMass,
                          spec_->compiler.settotalmass);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerUseThread,
                          (bool)spec_->compiler.usethread);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerBalanceInertia,
                          (bool)spec_->compiler.balanceinertia);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcCompilerAngle,
                          spec_->compiler.degree ? MjcPhysicsTokens->degree
                                                 : MjcPhysicsTokens->radian);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerFitAABB,
                          (bool)spec_->compiler.fitaabb);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerFuseStatic,
                          (bool)spec_->compiler.fusestatic);

    pxr::TfToken inertiafromgeom_token = MjcPhysicsTokens->auto_;
    if (spec_->compiler.inertiafromgeom ==
        mjINERTIAFROMGEOM_TRUE) {  // mjINERTIA_TRUE
      inertiafromgeom_token = MjcPhysicsTokens->true_;
    } else if (spec_->compiler.inertiafromgeom ==
               mjINERTIAFROMGEOM_FALSE) {  // mjINERTIA_FALSE
      inertiafromgeom_token = MjcPhysicsTokens->false_;
    }
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcCompilerInertiaFromGeom,
                          inertiafromgeom_token);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerAlignFree,
                          (bool)spec_->compiler.alignfree);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcCompilerInertiaGroupRangeMin,
                          spec_->compiler.inertiagrouprange[0]);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcCompilerInertiaGroupRangeMax,
                          spec_->compiler.inertiagrouprange[1]);

    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcCompilerSaveInertial,
                          (bool)spec_->compiler.saveinertial);

    // Newton scene attributes (auto-applied via MjcSceneAPI -> NewtonSceneAPI)
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Int,
                          kTokens->newtonMaxSolverIterations,
                          spec_->option.iterations);
    if (spec_->option.timestep > 0) {
      WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Int,
                            kTokens->newtonTimeStepsPerSecond,
                            static_cast<int>(std::round(1.0 / spec_->option.timestep)));
    }
    bool gravity_disabled = spec_->option.disableflags & mjDSBL_GRAVITY;
    WriteUniformAttribute(physics_scene_spec, pxr::SdfValueTypeNames->Bool,
                          kTokens->newtonGravityEnabled, !gravity_disabled);
  }

  void WriteMeshes() {
    // Create a scope for the meshes to keep things organized
    auto scope =
        CreatePrimSpec(layer_, body_paths_[kWorldIndex], kTokens->meshScope,
                       pxr::UsdGeomTokens->Scope);

    // Make the mesh scope invisible since they will be referenced by the bits
    // that should be visible.
    scope->SetField(pxr::SdfFieldKeys->Active, false);

    mjsMesh *mesh = mjs_asMesh(mjs_firstElement(spec_, mjOBJ_MESH));
    while (mesh) {
      if (ShouldWrite(mesh->element)) {
        WriteMesh(mesh, scope->GetPath());
      }
      mesh = mjs_asMesh(mjs_nextElement(spec_, mesh->element));
    }
  }

  pxr::SdfAttributeSpecHandle AddUVTextureShader(
      const pxr::SdfPrimSpecHandle material_spec, const pxr::TfToken &name) {
    auto uvmap_shader_spec =
        CreatePrimSpec(layer_, material_spec->GetPath(), name, pxr::UsdShadeTokens->Shader);

    auto uvmap_info_id_attr = CreateAttributeSpec(
        layer_, uvmap_shader_spec, pxr::UsdShadeTokens->infoId,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
    SetAttributeDefault(layer_, uvmap_info_id_attr,
                        kTokens->UsdPrimvarReader_float2);

    auto uvmap_varname_attr =
        CreateAttributeSpec(layer_, uvmap_shader_spec, kTokens->inputsVarname,
                            pxr::SdfValueTypeNames->Token);
    SetAttributeDefault(layer_, uvmap_varname_attr, kTokens->st);

    auto uvmap_st_output_attr =
        CreateAttributeSpec(layer_, uvmap_shader_spec, kTokens->outputsSt,
                            pxr::SdfValueTypeNames->Float2);

    return uvmap_st_output_attr;
  }

  std::vector<pxr::SdfAttributeSpecHandle> AddTextureShader(
      const pxr::SdfPrimSpecHandle material_spec, const char *texture_file,
      const pxr::TfToken &name,
      const pxr::SdfAttributeSpecHandle uvmap_st_output_attr,
      const std::vector<pxr::TfToken> &output_channels) {
    auto texture_shader_spec =
        CreatePrimSpec(layer_, material_spec->GetPath(), name, pxr::UsdShadeTokens->Shader);
    auto texture_info_id_attr = CreateAttributeSpec(
        layer_, texture_shader_spec, pxr::UsdShadeTokens->infoId,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
    SetAttributeDefault(layer_, texture_info_id_attr,
                        kTokens->UsdUVTexture);

    auto texture_file_attr =
        CreateAttributeSpec(layer_, texture_shader_spec, kTokens->inputsFile,
                            pxr::SdfValueTypeNames->Asset);
    SetAttributeDefault(layer_, texture_file_attr,
                        pxr::SdfAssetPath(texture_file));

    auto texture_st_input_attr =
        CreateAttributeSpec(layer_, texture_shader_spec, kTokens->inputsSt,
                            pxr::SdfValueTypeNames->Float2);
    AddAttributeConnection(layer_, texture_st_input_attr, uvmap_st_output_attr);

    auto texture_wrap_s_attr =
        CreateAttributeSpec(layer_, texture_shader_spec, kTokens->inputsWrapS,
                            pxr::SdfValueTypeNames->Token);
    SetAttributeDefault(layer_, texture_wrap_s_attr, kTokens->repeat);

    auto texture_wrap_t_attr =
        CreateAttributeSpec(layer_, texture_shader_spec, kTokens->inputsWrapT,
                            pxr::SdfValueTypeNames->Token);
    SetAttributeDefault(layer_, texture_wrap_t_attr, kTokens->repeat);

    std::vector<pxr::SdfAttributeSpecHandle> texture_output_attrs;
    for (const auto &output_channel : output_channels) {
      pxr::SdfValueTypeName value_type;
      if (output_channel == kTokens->outputsRgb) {
        value_type = pxr::SdfValueTypeNames->Float3;
      } else {
        // Assume the other specified channels are outputR, outputG, outputB.
        value_type = pxr::SdfValueTypeNames->Float;
      }
      texture_output_attrs.push_back(CreateAttributeSpec(
          layer_, texture_shader_spec, output_channel, value_type));
    }
    return texture_output_attrs;
  }

  pxr::SdfPrimSpecHandle WritePhysicsMaterial(mjsGeom *geom) {
    pxr::SdfPath scope_path =
        body_paths_[kWorldIndex].AppendChild(kTokens->physicsMaterialsScope);

    pxr::SdfPrimSpecHandle scope_spec = layer_->GetPrimAtPath(scope_path);
    if (!scope_spec) {
      scope_spec = CreatePrimSpec(layer_, body_paths_[kWorldIndex],
                                  kTokens->physicsMaterialsScope,
                                  pxr::UsdGeomTokens->Scope);
    }

    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdShadeTokens->Material, scope_path);
    auto material_spec =
        CreatePrimSpec(layer_, scope_spec->GetPath(), name, pxr::UsdShadeTokens->Material);

    ApplyApiSchema(layer_, material_spec,
                   pxr::UsdPhysicsTokens->PhysicsMaterialAPI);
    ApplyApiSchema(layer_, material_spec, MjcPhysicsTokens->MjcMaterialAPI);

    mjsGeom *geom_default = mjs_getDefault(geom->element)->geom;
    if (geom->friction[0] != geom_default->friction[0]) {
      // Since MuJoCo has no concept of static friction, only write dynamic
      // friction to remain truthful to how MuJoCo perceives the data.
      WriteUniformAttribute(material_spec, pxr::SdfValueTypeNames->Float,
                            pxr::UsdPhysicsTokens->physicsDynamicFriction,
                            (float)geom->friction[0]);
    }
    // Newton material attributes (replaces deprecated mjc:torsionalfriction / mjc:rollingfriction)
    WriteUniformAttribute(material_spec, pxr::SdfValueTypeNames->Float,
                          kTokens->newtonTorsionalFriction,
                          (float)geom->friction[1]);
    WriteUniformAttribute(material_spec, pxr::SdfValueTypeNames->Float,
                          kTokens->newtonRollingFriction,
                          (float)geom->friction[2]);
    if (geom->adhesion != 0.0f) {
      WriteUniformAttribute(material_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonContactAdhesion,
                            (float)geom->adhesion);
    }

    return material_spec;
  }

  void WriteMaterial(mjsMaterial *material, const pxr::SdfPath &parent_path) {
    // Create a Material prim.
    auto name =
        GetAvailablePrimName(*mjs_getName(material->element),
                             pxr::UsdShadeTokens->Material, parent_path);
    auto material_spec =
        CreatePrimSpec(layer_, parent_path, name, pxr::UsdShadeTokens->Material);

    // Create a Shader prim "PreviewSurface" under the Material prim.
    auto preview_surface_shader_spec =
        CreatePrimSpec(layer_, material_spec->GetPath(), kTokens->previewSurface,
                       pxr::UsdShadeTokens->Shader);

    // Set the Shader'sinfoId attribute to UsdPreviewSurface, a standard surface
    // shader.
    auto info_id_attr = CreateAttributeSpec(
        layer_, preview_surface_shader_spec, pxr::UsdShadeTokens->infoId,
        pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
    SetAttributeDefault(layer_, info_id_attr,
                        kTokens->UsdPreviewSurface);

    // Connect material's surface output to the preview surface's surface
    // output.
    auto surface_output_attr = CreateAttributeSpec(
        layer_, preview_surface_shader_spec, pxr::UsdShadeTokens->outputsSurface,
        pxr::SdfValueTypeNames->Token);
    auto material_surface_output_attr = CreateAttributeSpec(
        layer_, material_spec, pxr::UsdShadeTokens->outputsSurface,
        pxr::SdfValueTypeNames->Token);
    AddAttributeConnection(layer_, material_surface_output_attr,
                           surface_output_attr);

    // Connect material's displacement output to the preview surface's
    // displacement output.
    auto displacement_output_attr =
        CreateAttributeSpec(layer_, preview_surface_shader_spec,
                            pxr::UsdShadeTokens->outputsDisplacement,
                            pxr::SdfValueTypeNames->Token);
    auto material_displacement_output_attr = CreateAttributeSpec(
        layer_, material_spec, pxr::UsdShadeTokens->outputsDisplacement,
        pxr::SdfValueTypeNames->Token);
    AddAttributeConnection(layer_, material_displacement_output_attr,
                           displacement_output_attr);

    // Add an st (uv) Shader, a prim var reader for the UV coordinates.
    const pxr::SdfAttributeSpecHandle &uvmap_st_output_attr =
        AddUVTextureShader(material_spec, pxr::TfToken("uvmap"));
    const mjStringVec &textures = *(material->textures);

    // Set the values of metallic, roughness and occlusion. These can come from
    // an ORM packed texture, as individual textures, or as a values defined in
    // mjsMaterial_ (with the exception of occlusion).
    auto metallic_attr = CreateAttributeSpec(
        layer_, preview_surface_shader_spec, kTokens->inputsMetallic,
        pxr::SdfValueTypeNames->Float);
    auto roughness_attr = CreateAttributeSpec(
        layer_, preview_surface_shader_spec, kTokens->inputsRoughness,
        pxr::SdfValueTypeNames->Float);
    // Find the occlusion, roughness, and metallic textures.
    if (mjTEXROLE_ORM < textures.size()) {
      std::string orm_texture_name = textures[mjTEXROLE_ORM];
      mjsTexture *orm_texture = mjs_asTexture(
          mjs_findElement(spec_, mjOBJ_TEXTURE, orm_texture_name.c_str()));
      std::string occlusion_texture_name = textures[mjTEXROLE_OCCLUSION];
      mjsTexture *occlusion_texture = mjs_asTexture(mjs_findElement(
          spec_, mjOBJ_TEXTURE, occlusion_texture_name.c_str()));
      std::string roughness_texture_name = textures[mjTEXROLE_ROUGHNESS];
      mjsTexture *roughness_texture = mjs_asTexture(mjs_findElement(
          spec_, mjOBJ_TEXTURE, roughness_texture_name.c_str()));
      std::string metallic_texture_name = textures[mjTEXROLE_METALLIC];
      mjsTexture *metallic_texture = mjs_asTexture(
          mjs_findElement(spec_, mjOBJ_TEXTURE, metallic_texture_name.c_str()));
      if (orm_texture) {
        // Create the ORM shader and connect its output to the preview
        // surface ORM attrs.
        const std::vector<pxr::SdfAttributeSpecHandle> orm_output_attrs =
            AddTextureShader(
                material_spec, orm_texture->file->c_str(),
                pxr::TfToken("orm_packed"), uvmap_st_output_attr,
                {kTokens->outputsR, kTokens->outputsG, kTokens->outputsB});
        if (orm_output_attrs.size() == 3) {
          auto occlusion_attr = CreateAttributeSpec(
              layer_, preview_surface_shader_spec, kTokens->inputsOcclusion,
              pxr::SdfValueTypeNames->Float);
          AddAttributeConnection(layer_, occlusion_attr, orm_output_attrs[0]);
          AddAttributeConnection(layer_, roughness_attr, orm_output_attrs[1]);
          AddAttributeConnection(layer_, metallic_attr, orm_output_attrs[2]);
        }
      } else {
        if (metallic_texture) {
          const std::vector<pxr::SdfAttributeSpecHandle> metallic_output_attrs =
              AddTextureShader(material_spec, metallic_texture->file->c_str(),
                               pxr::TfToken("metallic"), uvmap_st_output_attr,
                               {kTokens->outputsRgb});
          if (metallic_output_attrs.size() == 1) {
            AddAttributeConnection(layer_, metallic_attr,
                                   metallic_output_attrs[0]);
          }
        } else {
          SetAttributeDefault(layer_, metallic_attr, material->metallic);
        }
        if (roughness_texture) {
          const std::vector<pxr::SdfAttributeSpecHandle> roughness_output_attrs =
              AddTextureShader(material_spec, roughness_texture->file->c_str(),
                               pxr::TfToken("roughness"), uvmap_st_output_attr,
                               {kTokens->outputsRgb});
          if (roughness_output_attrs.size() == 1) {
            AddAttributeConnection(layer_, roughness_attr,
                                   roughness_output_attrs[0]);
          }
        } else {
          SetAttributeDefault(layer_, roughness_attr, material->roughness);
        }
        if (occlusion_texture) {
          auto occlusion_attr = CreateAttributeSpec(
              layer_, preview_surface_shader_spec, kTokens->inputsOcclusion,
              pxr::SdfValueTypeNames->Float);
          const std::vector<pxr::SdfAttributeSpecHandle> occlusion_output_attrs =
              AddTextureShader(material_spec, occlusion_texture->file->c_str(),
                               pxr::TfToken("occlusion"), uvmap_st_output_attr,
                               {kTokens->outputsRgb});
          if (occlusion_output_attrs.size() == 1) {
            AddAttributeConnection(layer_, occlusion_attr,
                                   occlusion_output_attrs[0]);
          }
        }
      }
    }
    // Find the normal texture if specified.
    if (mjTEXROLE_NORMAL < textures.size()) {
      std::string normal_texture_name = textures[mjTEXROLE_NORMAL];
      mjsTexture *normal_texture = mjs_asTexture(
          mjs_findElement(spec_, mjOBJ_TEXTURE, normal_texture_name.c_str()));
      if (normal_texture) {
        auto normal_attr = CreateAttributeSpec(
            layer_, preview_surface_shader_spec, kTokens->inputsNormal,
            pxr::SdfValueTypeNames->Normal3f);
        // Create the normal map shader and connect its output to the preview
        // surface normal attr.
        const std::vector<pxr::SdfAttributeSpecHandle> normal_map_output_attrs =
            AddTextureShader(material_spec, normal_texture->file->c_str(),
                             pxr::TfToken("normal"), uvmap_st_output_attr,
                             {kTokens->outputsRgb});
        if (normal_map_output_attrs.size() == 1) {
          AddAttributeConnection(layer_, normal_attr,
                                 normal_map_output_attrs[0]);
        }
      }
    }

    // Connect an emissive texture if specified.
    if (mjTEXROLE_EMISSIVE < textures.size()) {
      std::string emissive_texture_name = textures[mjTEXROLE_EMISSIVE];
      mjsTexture *emissive_texture = mjs_asTexture(
          mjs_findElement(spec_, mjOBJ_TEXTURE, emissive_texture_name.c_str()));
      if (emissive_texture) {
        auto emissive_attr = CreateAttributeSpec(
            layer_, preview_surface_shader_spec, kTokens->inputsEmissiveColor,
            pxr::SdfValueTypeNames->Color3f);
        const std::vector<pxr::SdfAttributeSpecHandle> emissive_map_output_attrs =
            AddTextureShader(material_spec, emissive_texture->file->c_str(),
                             pxr::TfToken("emissive"), uvmap_st_output_attr,
                             {kTokens->outputsRgb});
        if (emissive_map_output_attrs.size() == 1) {
          AddAttributeConnection(layer_, emissive_attr,
                                 emissive_map_output_attrs[0]);
        }
      }
    }

    // Set the value of diffuse color. This can come from a diffuse texture
    // or as a value defined in mjsMaterial_.
    auto diffuse_color_attr = CreateAttributeSpec(
        layer_, preview_surface_shader_spec, kTokens->inputsDiffuseColor,
        pxr::SdfValueTypeNames->Color3f);

    // Find the main texture if specified.
    std::string main_texture_name = textures[mjTEXROLE_RGB];
    mjsTexture *main_texture = mjs_asTexture(
        mjs_findElement(spec_, mjOBJ_TEXTURE, main_texture_name.c_str()));
    if (main_texture) {
      // Create the texture shader and connect it to the diffuse color
      // attribute.
      const std::vector<pxr::SdfAttributeSpecHandle>
          texture_diffuse_output_attrs =
              AddTextureShader(material_spec, main_texture->file->c_str(),
                               pxr::TfToken("diffuse"), uvmap_st_output_attr,
                               {kTokens->outputsRgb});
      if (texture_diffuse_output_attrs.size() == 1) {
        AddAttributeConnection(layer_, diffuse_color_attr,
                               texture_diffuse_output_attrs[0]);
      }
    } else {
      // If no texture is specified, use the rgba diffuse color.
      SetAttributeDefault(layer_, diffuse_color_attr,
                          pxr::GfVec3f(material->rgba[0], material->rgba[1],
                                       material->rgba[2]));
    }
  }

  void WriteMaterials() {
    // Create a scope for the meshes to keep things organized
    auto scope =
        CreatePrimSpec(layer_, body_paths_[kWorldIndex], kTokens->materialsScope,
                       pxr::UsdGeomTokens->Scope);

    mjsMaterial *material =
        mjs_asMaterial(mjs_firstElement(spec_, mjOBJ_MATERIAL));
    while (material) {
      WriteMaterial(material, scope->GetPath());
      material = mjs_asMaterial(mjs_nextElement(spec_, material->element));
    }
  }

  void WriteKeyframesWithName(const std::string &name,
                              const std::vector<mjsKey *> &keyframes,
                              const pxr::SdfPath &parent_path) {
    if (keyframes.empty()) {
      return;
    }
    const auto keyframe_name = pxr::TfToken(pxr::TfMakeValidIdentifier(
        name.empty() ? MjcPhysicsTokens->MjcKeyframe : name));
    pxr::SdfPath keyframe_path = parent_path.AppendChild(keyframe_name);
    auto keyframe_spec =
        layer_->HasSpec(keyframe_path)
            ? layer_->GetPrimAtPath(keyframe_path)
            : CreatePrimSpec(layer_, parent_path, keyframe_name,
                             pxr::MjcPhysicsTokens->MjcKeyframe);

    auto set_attribute_data = [&](const pxr::SdfAttributeSpecHandle &attr_spec,
                                  const pxr::VtDoubleArray &value,
                                  mjsKey *keyframe) {
      // If the keyframe time is the default, and there are no other keyframes
      // set the attribute at the default time code.
      if (keyframe->time == 0 && keyframes.size() == 1) {
        SetAttributeDefault(layer_, attr_spec, value);
      } else {
        SetAttributeTimeSample(layer_, attr_spec, keyframe->time, value);
      }
    };

    for (auto *keyframe : keyframes) {
      auto qpos_attr =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcQpos,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          qpos_attr,
          pxr::VtDoubleArray(keyframe->qpos->begin(), keyframe->qpos->end()),
          keyframe);

      auto qvel_attr =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcQvel,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          qvel_attr,
          pxr::VtDoubleArray(keyframe->qvel->begin(), keyframe->qvel->end()),
          keyframe);

      auto act_attr =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcAct,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          act_attr,
          pxr::VtDoubleArray(keyframe->act->begin(), keyframe->act->end()),
          keyframe);

      auto ctrl_attr =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcCtrl,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          ctrl_attr,
          pxr::VtDoubleArray(keyframe->ctrl->begin(), keyframe->ctrl->end()),
          keyframe);

      auto mpos_attr =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcMpos,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          mpos_attr,
          pxr::VtDoubleArray(keyframe->mpos->begin(), keyframe->mpos->end()),
          keyframe);

      auto mquat_attr_path =
          CreateAttributeSpec(layer_, keyframe_spec, MjcPhysicsTokens->mjcMquat,
                              pxr::SdfValueTypeNames->DoubleArray);
      set_attribute_data(
          mquat_attr_path,
          pxr::VtDoubleArray(keyframe->mquat->begin(), keyframe->mquat->end()),
          keyframe);
    }
  }

  void WriteKeyframes() {
    std::unordered_map<std::string, std::vector<mjsKey *>> keyframes_map;
    mjsKey *keyframe = mjs_asKey(mjs_firstElement(spec_, mjOBJ_KEY));
    while (keyframe) {
      if (ShouldWrite(keyframe->element)) {
        std::string keyframe_name = mjs_getName(keyframe->element)->empty()
                                        ? kTokens->keyframe
                                        : *mjs_getName(keyframe->element);
        keyframes_map[keyframe_name].push_back(keyframe);
      }
      keyframe = mjs_asKey(mjs_nextElement(spec_, keyframe->element));
    }

    auto scope =
        CreatePrimSpec(layer_, body_paths_[kWorldIndex], kTokens->keyframesScope,
                       pxr::UsdGeomTokens->Scope);
    for (const auto &[keyframe_name, keyframes] : keyframes_map) {
      WriteKeyframesWithName(keyframe_name, keyframes, scope->GetPath());
    }
  }

  void WriteActuator(mjsActuator *actuator, const pxr::SdfPath &parent_path) {
    const pxr::TfToken valid_name = GetValidPrimName(*mjs_getName(actuator->element));
    const pxr::SdfPath actuator_path = parent_path.AppendChild(valid_name);
    pxr::SdfPrimSpecHandle actuator_spec;
    if (layer_->HasSpec(actuator_path)) {
      actuator_spec = layer_->GetPrimAtPath(actuator_path);
    } else {
      actuator_spec = CreatePrimSpec(layer_, parent_path, valid_name,
                     pxr::MjcPhysicsTokens->MjcActuator);
    }

    pxr::SdfPath target_path;
    if (actuator->trntype == mjtTrn::mjTRN_BODY) {
      int body_id = mj_name2id(model_, mjOBJ_BODY, actuator->target->c_str());
      target_path = body_paths_[body_id];
    } else if (actuator->trntype == mjtTrn::mjTRN_SITE ||
               actuator->trntype == mjtTrn::mjTRN_SLIDERCRANK) {
      int site_id = mj_name2id(model_, mjOBJ_SITE, actuator->target->c_str());
      target_path = site_paths_[site_id];
    } else if (actuator->trntype == mjtTrn::mjTRN_JOINT) {
      int joint_id = mj_name2id(model_, mjOBJ_JOINT, actuator->target->c_str());
      target_path = joint_paths_[joint_id];
    } else {
      TF_WARN(UnsupportedActuatorTypeError,
              "Unsupported transmission type for actuator %d",
              mjs_getId(actuator->element));
      return;
    }

    CreateRelationshipSpec(layer_, actuator_spec, MjcPhysicsTokens->mjcTarget,
                           target_path, pxr::SdfVariabilityUniform);

    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcGroup, actuator->group);

    if (!actuator->refsite->empty()) {
      int refsite_id =
          mj_name2id(model_, mjOBJ_SITE, actuator->refsite->c_str());
      pxr::SdfPath refsite_path = site_paths_[refsite_id];
      CreateRelationshipSpec(layer_, actuator_spec, MjcPhysicsTokens->mjcRefSite,
                             refsite_path, pxr::SdfVariabilityUniform);
    }

    if (!actuator->slidersite->empty()) {
      int slidersite_id =
          mj_name2id(model_, mjOBJ_SITE, actuator->slidersite->c_str());
      pxr::SdfPath slidersite_path = site_paths_[slidersite_id];
      CreateRelationshipSpec(layer_, actuator_spec,
                             MjcPhysicsTokens->mjcSliderSite, slidersite_path,
                             pxr::SdfVariabilityUniform);
    }

    const std::vector<std::pair<pxr::TfToken, int>> limited_attributes = {
        {MjcPhysicsTokens->mjcCtrlLimited, actuator->ctrllimited},
        {MjcPhysicsTokens->mjcForceLimited, actuator->forcelimited},
        {MjcPhysicsTokens->mjcActLimited, actuator->actlimited},
    };
    for (const auto &[token, value] : limited_attributes) {
      pxr::TfToken limited_token = pxr::MjcPhysicsTokens->auto_;
      if (value == mjLIMITED_TRUE) {
        limited_token = pxr::MjcPhysicsTokens->true_;
      } else if (value == mjLIMITED_FALSE) {
        limited_token = pxr::MjcPhysicsTokens->false_;
      }
      WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Token, token,
                            limited_token);
    }

    const std::vector<std::pair<pxr::TfToken, double>>
        actuator_double_attributes = {
            {MjcPhysicsTokens->mjcCtrlRangeMin, actuator->ctrlrange[0]},
            {MjcPhysicsTokens->mjcCtrlRangeMax, actuator->ctrlrange[1]},
            {MjcPhysicsTokens->mjcForceRangeMin, actuator->forcerange[0]},
            {MjcPhysicsTokens->mjcForceRangeMax, actuator->forcerange[1]},
            {MjcPhysicsTokens->mjcActRangeMin, actuator->actrange[0]},
            {MjcPhysicsTokens->mjcActRangeMax, actuator->actrange[1]},
            {MjcPhysicsTokens->mjcLengthRangeMin, actuator->lengthrange[0]},
            {MjcPhysicsTokens->mjcLengthRangeMax, actuator->lengthrange[1]},
            {MjcPhysicsTokens->mjcCrankLength, actuator->cranklength},
        };
    for (const auto &[token, value] : actuator_double_attributes) {
      WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Double,
                            token, value);
    }

    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcActDim, actuator->actdim);
    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Bool,
                          MjcPhysicsTokens->mjcActEarly,
                          (bool)actuator->actearly);
    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Double,
                          MjcPhysicsTokens->mjcInheritRange,
                          actuator->inheritrange);

    WriteUniformAttribute(
        actuator_spec, pxr::SdfValueTypeNames->DoubleArray,
        MjcPhysicsTokens->mjcGear,
        pxr::VtDoubleArray(actuator->gear, actuator->gear + 6));

    pxr::TfToken dyn_type;
    if (actuator->dyntype == mjtDyn::mjDYN_NONE) {
      dyn_type = MjcPhysicsTokens->none;
    } else if (actuator->dyntype == mjtDyn::mjDYN_INTEGRATOR) {
      dyn_type = MjcPhysicsTokens->integrator;
    } else if (actuator->dyntype == mjtDyn::mjDYN_FILTER) {
      dyn_type = MjcPhysicsTokens->filter;
    } else if (actuator->dyntype == mjtDyn::mjDYN_FILTEREXACT) {
      dyn_type = MjcPhysicsTokens->filterexact;
    } else if (actuator->dyntype == mjtDyn::mjDYN_MUSCLE) {
      dyn_type = MjcPhysicsTokens->muscle;
    } else if (actuator->dyntype == mjtDyn::mjDYN_USER) {
      dyn_type = MjcPhysicsTokens->user;
    }
    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcDynType, dyn_type);
    WriteUniformAttribute(
        actuator_spec, pxr::SdfValueTypeNames->DoubleArray,
        MjcPhysicsTokens->mjcDynPrm,
        pxr::VtDoubleArray(actuator->dynprm, actuator->dynprm + 10));

    pxr::TfToken gain_type;
    if (actuator->gaintype == mjtGain::mjGAIN_FIXED) {
      gain_type = MjcPhysicsTokens->fixed;
    } else if (actuator->gaintype == mjtGain::mjGAIN_AFFINE) {
      gain_type = MjcPhysicsTokens->affine;
    } else if (actuator->gaintype == mjtGain::mjGAIN_MUSCLE) {
      gain_type = MjcPhysicsTokens->muscle;
    } else if (actuator->gaintype == mjtGain::mjGAIN_USER) {
      gain_type = MjcPhysicsTokens->user;
    }
    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcGainType, gain_type);
    WriteUniformAttribute(
        actuator_spec, pxr::SdfValueTypeNames->DoubleArray,
        MjcPhysicsTokens->mjcGainPrm,
        pxr::VtDoubleArray(actuator->gainprm, actuator->gainprm + 10));

    pxr::TfToken bias_type;
    if (actuator->biastype == mjtBias::mjBIAS_NONE) {
      bias_type = MjcPhysicsTokens->fixed;
    } else if (actuator->biastype == mjtBias::mjBIAS_AFFINE) {
      bias_type = MjcPhysicsTokens->affine;
    } else if (actuator->biastype == mjtBias::mjBIAS_MUSCLE) {
      bias_type = MjcPhysicsTokens->muscle;
    } else if (actuator->biastype == mjtBias::mjBIAS_USER) {
      bias_type = MjcPhysicsTokens->user;
    }
    WriteUniformAttribute(actuator_spec, pxr::SdfValueTypeNames->Token,
                          MjcPhysicsTokens->mjcBiasType, bias_type);
    WriteUniformAttribute(
        actuator_spec, pxr::SdfValueTypeNames->DoubleArray,
        MjcPhysicsTokens->mjcBiasPrm,
        pxr::VtDoubleArray(actuator->biasprm, actuator->biasprm + 10));
  }

  void WriteActuators() {
    auto scope =
        CreatePrimSpec(layer_, body_paths_[kWorldIndex], kTokens->actuatorsScope,
                       pxr::UsdGeomTokens->Scope);
    mjsActuator *actuator =
        mjs_asActuator(mjs_firstElement(spec_, mjOBJ_ACTUATOR));
    while (actuator) {
      if (ShouldWrite(actuator->element)) {
        WriteActuator(actuator, scope->GetPath());
      }
      actuator = mjs_asActuator(mjs_nextElement(spec_, actuator->element));
    }
  }

  pxr::SdfPrimSpecHandle WriteMeshGeom(const mjsGeom *geom,
                             const pxr::SdfPath &body_path) {
    std::string mj_name = mjs_getName(geom->element)->empty()
                              ? *geom->meshname
                              : *mjs_getName(geom->element);
    auto name =
        GetAvailablePrimName(mj_name, pxr::UsdGeomTokens->Mesh, body_path);
    auto subcomponent_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Xform);

    // Reference the mesh asset written in WriteMeshes.
    AddPrimReference(layer_, subcomponent_spec, mesh_paths_[*geom->meshname]);

    // We want to use instancing with meshes, and it requires creating a parent
    // scope to be referenced, with the Mesh prim as a child.
    // To be able to actually manipulate the Mesh prim, we need to create and
    // return the corresponding `over` prim as a child of the referencing prim.
    auto over_mesh_spec =
        CreatePrimSpec(layer_, subcomponent_spec->GetPath(), kTokens->sourceMesh,
                       pxr::UsdGeomTokens->Mesh, pxr::SdfSpecifierOver);

    return over_mesh_spec;
  }

  pxr::SdfPrimSpecHandle WriteSiteGeom(const mjsSite *site,
                             const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(site->element),
                                     pxr::UsdGeomTokens->Cube, body_path);

    int site_idx = mjs_getId(site->element);
    const mjtNum *size = &model_->site_size[site_idx * 3];
    pxr::SdfPrimSpecHandle site_spec;
    switch (site->type) {
      case mjGEOM_BOX:
        site_spec = WriteBox(name, size, body_path);
        break;
      case mjGEOM_SPHERE:
        site_spec = WriteSphere(name, size, body_path);
        break;
      case mjGEOM_CAPSULE:
        site_spec = WriteCapsule(name, size, body_path);
        break;
      case mjGEOM_CYLINDER:
        site_spec = WriteCylinder(name, size, body_path);
        break;
      case mjGEOM_ELLIPSOID:
        site_spec = WriteEllipsoid(name, size, body_path);
        break;
      default:
        break;
    }

    return site_spec;
  }

  pxr::SdfPrimSpecHandle WriteBox(const pxr::TfToken &name, const mjtNum *size,
                        const pxr::SdfPath &body_path) {
    pxr::SdfPrimSpecHandle box_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Cube);
    // MuJoCo uses half sizes. Always set size to 2 (and correspondingly extent
    // from -1 to 1), and let scale determine the actual size.
    auto size_attr_spec =
        CreateAttributeSpec(layer_, box_spec, pxr::UsdGeomTokens->size,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, size_attr_spec, 2.0);

    auto extent_attr_spec =
        CreateAttributeSpec(layer_, box_spec, pxr::UsdGeomTokens->extent,
                            pxr::SdfValueTypeNames->Float3Array);
    SetAttributeDefault(layer_, extent_attr_spec,
                        pxr::VtArray<pxr::GfVec3f>(
                            {pxr::GfVec3f(-1, -1, -1), pxr::GfVec3f(1, 1, 1)}));

    pxr::GfVec3f scale(static_cast<float>(size[0]), static_cast<float>(size[1]),
                       static_cast<float>(size[2]));
    WriteScaleXformOp(box_spec, scale);
    WriteXformOpOrder(box_spec,
                      pxr::VtArray<pxr::TfToken>{kTokens->xformOpScale});
    return box_spec;
  }

  pxr::SdfPrimSpecHandle WriteBoxGeom(const mjsGeom *geom,
                            const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Cube, body_path);

    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];
    return WriteBox(name, geom_size, body_path);
  }

  pxr::SdfPrimSpecHandle WriteCapsule(const pxr::TfToken name, const mjtNum *size,
                            const pxr::SdfPath &body_path) {
    auto capsule_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Capsule);

    auto radius_attr_spec =
        CreateAttributeSpec(layer_, capsule_spec, pxr::UsdGeomTokens->radius,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, radius_attr_spec, (double)size[0]);

    auto height_attr_spec =
        CreateAttributeSpec(layer_, capsule_spec, pxr::UsdGeomTokens->height,
                            pxr::SdfValueTypeNames->Double);
    // MuJoCo uses half sizes.
    SetAttributeDefault(layer_, height_attr_spec, (double)(size[1] * 2));
    return capsule_spec;
  }

  pxr::SdfPrimSpecHandle WriteCapsuleGeom(const mjsGeom *geom,
                                const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Capsule, body_path);
    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];

    return WriteCapsule(name, geom_size, body_path);
  }

  pxr::SdfPrimSpecHandle WriteCylinder(const pxr::TfToken name, const mjtNum *size,
                             const pxr::SdfPath &body_path) {
    auto cylinder_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Cylinder);

    auto radius_attr_spec =
        CreateAttributeSpec(layer_, cylinder_spec, pxr::UsdGeomTokens->radius,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, radius_attr_spec, (double)size[0]);

    auto height_attr_spec =
        CreateAttributeSpec(layer_, cylinder_spec, pxr::UsdGeomTokens->height,
                            pxr::SdfValueTypeNames->Double);
    // MuJoCo uses half sizes.
    SetAttributeDefault(layer_, height_attr_spec, (double)(size[1] * 2));
    return cylinder_spec;
  }

  pxr::SdfPrimSpecHandle WriteCylinderGeom(const mjsGeom *geom,
                                 const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Cylinder, body_path);

    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];
    return WriteCylinder(name, geom_size, body_path);
  }

  pxr::SdfPrimSpecHandle WriteEllipsoid(const pxr::TfToken name, const mjtNum *size,
                              const pxr::SdfPath &body_path) {
    auto ellipsoid_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Sphere);

    pxr::GfVec3f scale = {static_cast<float>(size[0]),
                          static_cast<float>(size[1]),
                          static_cast<float>(size[2])};

    auto radius_attr_spec =
        CreateAttributeSpec(layer_, ellipsoid_spec, pxr::UsdGeomTokens->radius,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, radius_attr_spec, 1.0);

    WriteScaleXformOp(ellipsoid_spec, scale);
    WriteXformOpOrder(ellipsoid_spec,
                      pxr::VtArray<pxr::TfToken>{kTokens->xformOpScale});
    return ellipsoid_spec;
  }

  pxr::SdfPrimSpecHandle WriteEllipsoidGeom(const mjsGeom *geom,
                                  const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Sphere, body_path);
    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];

    return WriteEllipsoid(name, geom_size, body_path);
  }

  pxr::SdfPrimSpecHandle WriteSphere(const pxr::TfToken name, const mjtNum *size,
                           const pxr::SdfPath &body_path) {
    auto sphere_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Sphere);

    auto radius_attr_spec =
        CreateAttributeSpec(layer_, sphere_spec, pxr::UsdGeomTokens->radius,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, radius_attr_spec, (double)size[0]);
    return sphere_spec;
  }

  pxr::SdfPrimSpecHandle WriteSphereGeom(const mjsGeom *geom,
                               const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Sphere, body_path);
    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];
    return WriteSphere(name, geom_size, body_path);
  }

  pxr::SdfPrimSpecHandle WritePlane(const pxr::TfToken &name, const mjtNum *size,
                          const pxr::SdfPath &body_path) {
    auto plane_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Plane);

    // MuJoCo uses half sizes.
    // Note that UsdGeomPlane is infinite for simulation purposes but can have
    // width/length for visualization, same as MuJoCo.
    double width = size[0] * 2.0;
    double length = size[1] * 2.0;

    auto width_attr_spec =
        CreateAttributeSpec(layer_, plane_spec, pxr::UsdGeomTokens->width,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, width_attr_spec, width);

    auto length_attr_spec =
        CreateAttributeSpec(layer_, plane_spec, pxr::UsdGeomTokens->length,
                            pxr::SdfValueTypeNames->Double);
    SetAttributeDefault(layer_, length_attr_spec, length);

    // MuJoCo plane is always a XY plane with +Z up.
    // UsdGeomPlane is also a XY plane if axis is 'Z', which is default.
    // So no need to set axis attribute explicitly.

    return plane_spec;
  }

  pxr::SdfPrimSpecHandle WritePlaneGeom(const mjsGeom *geom,
                              const pxr::SdfPath &body_path) {
    auto name = GetAvailablePrimName(*mjs_getName(geom->element),
                                     pxr::UsdGeomTokens->Plane, body_path);
    int geom_idx = mjs_getId(geom->element);
    mjtNum *geom_size = &model_->geom_size[geom_idx * 3];
    return WritePlane(name, geom_size, body_path);
  }

  void WriteSite(mjsSite *site, const mjsBody *body) {
    const int body_id = mjs_getId(body->element);
    const auto &body_path = body_paths_[body_id];
    auto name = GetAvailablePrimName(*mjs_getName(site->element),
                                     pxr::UsdGeomTokens->Xform, body_path);

    // Create a geom primitive and set its purpose to guide so it won't be
    // rendered.
    auto site_spec = WriteSiteGeom(site, body_path);
    SetPrimPurpose(layer_, site_spec, pxr::UsdGeomTokens->guide);

    ApplyApiSchema(layer_, site_spec, MjcPhysicsTokens->MjcSiteAPI);

    WriteUniformAttribute(site_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcGroup, site->group);

    WriteColorAndOpacityAttributes(site_spec, site);

    int site_id = mjs_getId(site->element);
    auto transform = MujocoPosQuatToTransform(&model_->site_pos[3 * site_id],
                                              &model_->site_quat[4 * site_id]);
    WriteTransformXformOp(site_spec, transform);

    PrependToXformOpOrder(
        site_spec, pxr::VtArray<pxr::TfToken>{kTokens->xformOpTransform});

    site_paths_[site_id] = site_spec->GetPath();
  }

  void WriteGeom(mjsGeom *geom, const mjsBody *body) {
    const int body_id = mjs_getId(body->element);
    const auto &body_path = body_paths_[body_id];

    pxr::SdfPrimSpecHandle geom_spec;
    int geom_id = mjs_getId(geom->element);
    switch (geom->type) {
      case mjGEOM_PLANE:
        geom_spec = WritePlaneGeom(geom, body_path);
        break;
      case mjGEOM_MESH:
        geom_spec = WriteMeshGeom(geom, body_path);
        break;
      case mjGEOM_BOX:
        geom_spec = WriteBoxGeom(geom, body_path);
        break;
      case mjGEOM_CAPSULE:
        geom_spec = WriteCapsuleGeom(geom, body_path);
        break;
      case mjGEOM_CYLINDER:
        geom_spec = WriteCylinderGeom(geom, body_path);
        break;
      case mjGEOM_ELLIPSOID:
        geom_spec = WriteEllipsoidGeom(geom, body_path);
        break;
      case mjGEOM_SPHERE:
        geom_spec = WriteSphereGeom(geom, body_path);
        break;
      default:
        TF_WARN(UnsupportedGeomTypeError, "Unsupported geom type for geom %d",
                geom_id);
        return;
    }

    WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Int,
                          MjcPhysicsTokens->mjcGroup, geom->group);

    if (model_->geom_contype[geom_id] == 0 &&
        model_->geom_conaffinity[geom_id] == 0) {
      // If the geom is purely visual, apply the imageable API.
      ApplyApiSchema(layer_, geom_spec, MjcPhysicsTokens->MjcImageableAPI);
    }

    if (geom->typeinertia == mjtGeomInertia::mjINERTIA_SHELL) {
      ApplyApiSchema(layer_, geom_spec, kTokens->NewtonMassAPI);
      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Token,
                            kTokens->newtonMassModel,
                            MjcPhysicsTokens->shell);
    }

    const double density = (geom->mass == 0) ? 0.0 : geom->density;
    if (geom->mass >= mjMINVAL || density >= 0) {
      ApplyApiSchema(layer_, geom_spec, pxr::UsdPhysicsTokens->PhysicsMassAPI);
    }

    if (geom->mass >= mjMINVAL) {
      auto mass_attr_spec = CreateAttributeSpec(
          layer_, geom_spec, pxr::UsdPhysicsTokens->physicsMass,
          pxr::SdfValueTypeNames->Float, pxr::SdfVariabilityUniform);

      // Make sure to cast to float here since mjtNum might be a double.
      SetAttributeDefault(layer_, mass_attr_spec, (float)geom->mass);
    }

    // Even though density is not used for mass computation when mass exists
    // we want to retain the information anyways.
    if (density >= 0) {
      auto density_attr_spec = CreateAttributeSpec(
          layer_, geom_spec, pxr::UsdPhysicsTokens->physicsDensity,
          pxr::SdfValueTypeNames->Float, pxr::SdfVariabilityUniform);

      // Make sure to cast to float here since mjtNum might be a double.
      SetAttributeDefault(layer_, density_attr_spec, (float)density);
    }

    // Apply the physics schemas if we are writing physics and the
    // geom participates in collisions.
    if (model_->geom_contype[geom_id] != 0 ||
        model_->geom_conaffinity[geom_id] != 0) {
      ApplyApiSchema(layer_, geom_spec,
                     pxr::UsdPhysicsTokens->PhysicsCollisionAPI);
      ApplyApiSchema(layer_, geom_spec, MjcPhysicsTokens->MjcCollisionAPI);

      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Int,
                            MjcPhysicsTokens->mjcPriority, geom->priority);

      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Int,
                            MjcPhysicsTokens->mjcCondim, geom->condim);

      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcSolmix, geom->solmix);

      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcSolmix, geom->solmix);

      WriteUniformAttribute(
          geom_spec, pxr::SdfValueTypeNames->DoubleArray,
          MjcPhysicsTokens->mjcSolref,
          pxr::VtArray<double>(geom->solref, geom->solref + mjNREF));

      WriteUniformAttribute(
          geom_spec, pxr::SdfValueTypeNames->DoubleArray,
          MjcPhysicsTokens->mjcSolimp,
          pxr::VtArray<double>(geom->solimp, geom->solimp + mjNIMP));

      // Newton collision attributes (replaces deprecated mjc:margin / mjc:gap)
      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonContactMargin,
                            static_cast<float>(geom->margin));
      WriteUniformAttribute(geom_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonContactGap,
                            static_cast<float>(geom->gap));

      mjsDefault *geom_default = mjs_getDefault(geom->element);

      if (geom->friction[0] != geom_default->geom->friction[0] ||
          geom->friction[1] != geom_default->geom->friction[1] ||
          geom->friction[2] != geom_default->geom->friction[2]) {
        auto physics_material_spec = WritePhysicsMaterial(geom);
        ApplyApiSchema(layer_, geom_spec, pxr::UsdShadeTokens->MaterialBindingAPI);
        // Bind the material to this geom.
        CreateRelationshipSpec(
            layer_, geom_spec,
            pxr::UsdShadeTokens->materialBinding, physics_material_spec->GetPath(),
            pxr::SdfVariabilityUniform);
      }

      // For meshes, also apply PhysicsMeshCollisionAPI and set the
      // approximation attribute.
      if (geom->type == mjGEOM_MESH) {
        ApplyApiSchema(layer_, geom_spec,
                       pxr::UsdPhysicsTokens->PhysicsMeshCollisionAPI);

        // Note: MuJoCo documentation states that for collision purposes, meshes
        // are always replaced with their convex hulls. Therefore, we set the
        // approximation attribute to convexHull explicitly.
        auto approximation_attr_spec = CreateAttributeSpec(
            layer_, geom_spec, pxr::UsdPhysicsTokens->physicsApproximation,
            pxr::SdfValueTypeNames->Token, pxr::SdfVariabilityUniform);
        SetAttributeDefault(layer_, approximation_attr_spec,
                            pxr::UsdPhysicsTokens->convexHull);
      }
    } else {
      // Currently imageable only has a group API. But since it's the same
      // naming in MjcCollisionsAPI we've already set it earlier in this
      // function.
      ApplyApiSchema(layer_, geom_spec, MjcPhysicsTokens->MjcImageableAPI);
    }

    mjsDefault *spec_default = mjs_getDefault(geom->element);
    pxr::TfToken valid_class_name =
        GetValidPrimName(*mjs_getName(spec_default->element));
    pxr::SdfPath geom_class_path = class_path_.AppendChild(valid_class_name);
    if (!layer_->HasSpec(geom_class_path)) {
      auto class_spec =
          CreateClassSpec(layer_, class_path_, valid_class_name);
      auto visibility_attr_spec =
          CreateAttributeSpec(layer_, class_spec, pxr::UsdGeomTokens->visibility,
                              pxr::SdfValueTypeNames->Token);
      SetAttributeDefault(layer_, visibility_attr_spec,
                          pxr::UsdGeomTokens->inherited);
    }

    // Bind material if it exists.
    if (!geom->material->empty()) {
      pxr::SdfPath material_path =
          body_paths_[kWorldIndex]
              .AppendChild(kTokens->materialsScope)
              .AppendChild(GetValidPrimName(*geom->material));
      if (layer_->HasSpec(material_path)) {
        ApplyApiSchema(layer_, geom_spec,
                       pxr::UsdShadeTokens->MaterialBindingAPI);
        // Bind the material to this geom.
        CreateRelationshipSpec(layer_, geom_spec,
                               pxr::UsdShadeTokens->materialBinding,
                               material_path, pxr::SdfVariabilityUniform);
      }
    }

    WriteColorAndOpacityAttributes(geom_spec, geom);

    if (body_id == kWorldIndex) {
      SetPrimKind(layer_, geom_spec, pxr::KindTokens->component);
    }
    // Inherit from class.
    AddPrimInherit(layer_, geom_spec, geom_class_path);

    auto transform = MujocoPosQuatToTransform(&model_->geom_pos[3 * geom_id],
                                              &model_->geom_quat[4 * geom_id]);
    WriteTransformXformOp(geom_spec, transform);

    PrependToXformOpOrder(
        geom_spec, pxr::VtArray<pxr::TfToken>{kTokens->xformOpTransform});
  }

  void WriteSites(mjsBody *body) {
    mjsSite *site = mjs_asSite(mjs_firstChild(body, mjOBJ_SITE, false));
    while (site) {
      if (ShouldWrite(site->element)) {
        WriteSite(site, body);
      }
      site = mjs_asSite(mjs_nextChild(body, site->element, false));
    }
  }

  void WriteGeoms(mjsBody *body) {
    mjsGeom *geom = mjs_asGeom(mjs_firstChild(body, mjOBJ_GEOM, false));
    while (geom) {
      if (ShouldWrite(geom->element)) {
        WriteGeom(geom, body);
      }
      geom = mjs_asGeom(mjs_nextChild(body, geom->element, false));
    }
  }

  void WriteJoints(mjsBody *body) {
    int body_id = mjs_getId(body->element);
    if (body_id == kWorldIndex) return;

    mjsJoint *joint = mjs_asJoint(mjs_firstChild(body, mjOBJ_JOINT, false));

    if (!joint) {
      // If no joint is found, then we pass nullptr to create a FixedJoint.
      // WriteJoint properly handles the case where the parent is the worldbody.
      WriteJoint(nullptr, body);
    } else {
      if (ShouldWrite(joint->element)) {
        WriteJoint(joint, body);
      }
      if (mjs_asJoint(mjs_nextChild(body, joint->element, false))) {
        TF_WARN(
            "Multiple joints found for body %d. Only writing the first one.",
            body_id);
      }
    }
  }

  // Write the joint. If null, then a FixedJoint is created.
  void WriteJoint(mjsJoint *joint, const mjsBody *parent_mj_body) {
    // Default to fixed joint if joint is null.
    pxr::TfToken joint_prim_type = pxr::UsdPhysicsTokens->PhysicsFixedJoint;

    int joint_id = -1;
    if (joint) {
      joint_id = mjs_getId(joint->element);
      mjtJoint type = (mjtJoint)model_->jnt_type[joint_id];
      switch (type) {
        case mjJNT_FREE:
          // Free joints are guaranteed to only ever be on the top-level body so
          // we just write no joint. As a top-level body with no joint it will
          // be considered a floating-base body.
          return;
        case mjJNT_HINGE:
          joint_prim_type = pxr::UsdPhysicsTokens->PhysicsRevoluteJoint;
          break;
        case mjJNT_SLIDE:
          joint_prim_type = pxr::UsdPhysicsTokens->PhysicsPrismaticJoint;
          break;
        default:
          TF_WARN("Unsupported joint type '%d' for joint '%s'. Skipping.",
                  (int)type, mjs_getName(joint->element)->c_str());
          return;
      }
    }

    int body_id = mjs_getId(parent_mj_body->element);

    // the joint connects the current body as body1, to its parent body as
    // body0.
    int body1_id_usd = body_id;
    int body0_id_usd = model_->body_parentid[body_id];

    const pxr::SdfPath &body1_path_usd = body_paths_[body1_id_usd];
    auto joint_name = joint ? *mjs_getName(joint->element) : "FixedJoint";
    pxr::TfToken joint_name_token =
        GetAvailablePrimName(joint_name, kTokens->joint, body1_path_usd);
    auto joint_spec = CreatePrimSpec(layer_, body1_path_usd,
                                             joint_name_token, joint_prim_type);

    // Set body0 and body1 relationships
    // For the initial joints that connect to the world, we signal this by
    // keeping the body0 relationship empty.
    if (body0_id_usd != kWorldIndex) {
      const pxr::SdfPath &body0_path_usd = body_paths_[body0_id_usd];
      CreateRelationshipSpec(layer_, joint_spec,
                             pxr::UsdPhysicsTokens->physicsBody0,
                             body0_path_usd, pxr::SdfVariabilityUniform);
    }
    CreateRelationshipSpec(layer_, joint_spec,
                           pxr::UsdPhysicsTokens->physicsBody1, body1_path_usd,
                           pxr::SdfVariabilityUniform);

    // Joint frame in MuJoCo is defined by jnt_pos and jnt_axis in body1's frame
    // For FixedJoint, these are both unity.
    pxr::GfVec3d mj_jnt_pos = pxr::GfVec3d(0.0);
    pxr::GfVec3d mj_jnt_axis = pxr::GfVec3d(0.0, 0.0, 1.0);
    if (joint) {
      mj_jnt_pos = pxr::GfVec3d(&model_->jnt_pos[joint_id * 3]);
      mj_jnt_axis = pxr::GfVec3d(&model_->jnt_axis[joint_id * 3]);
    }

    // Local joint frame for body1
    pxr::GfVec3f local_pos1(mj_jnt_pos);
    pxr::GfQuatf local_rot1(
        pxr::GfRotation(pxr::GfVec3f::ZAxis(), mj_jnt_axis).GetQuat());

    SetAttributeDefault(
        layer_,
        CreateAttributeSpec(layer_, joint_spec,
                            pxr::UsdPhysicsTokens->physicsLocalPos1,
                            pxr::SdfValueTypeNames->Float3),
        local_pos1);
    if (joint_prim_type == pxr::UsdPhysicsTokens->PhysicsRevoluteJoint ||
        joint_prim_type == pxr::UsdPhysicsTokens->PhysicsPrismaticJoint) {
      SetAttributeDefault(
          layer_,
          CreateAttributeSpec(layer_, joint_spec,
                              pxr::UsdPhysicsTokens->physicsLocalRot1,
                              pxr::SdfValueTypeNames->Quatf),
          local_rot1);
    } else {
      SetAttributeDefault(
          layer_,
          CreateAttributeSpec(layer_, joint_spec,
                              pxr::UsdPhysicsTokens->physicsLocalRot1,
                              pxr::SdfValueTypeNames->Quatf),
          pxr::GfQuatf::GetIdentity());
    }

    // Calculate local joint frame for body0
    pxr::GfMatrix4d body1_transform_local =
        MujocoPosQuatToTransform(&model_->body_pos[body1_id_usd * 3],
                                 &model_->body_quat[body1_id_usd * 4]);
    pxr::GfVec3d jnt_pos_parent_local =
        body1_transform_local.Transform(mj_jnt_pos);
    pxr::GfVec3d jnt_axis_parent_local =
        body1_transform_local.TransformDir(mj_jnt_axis);

    pxr::GfVec3f local_pos0(jnt_pos_parent_local);

    SetAttributeDefault(
        layer_,
        CreateAttributeSpec(layer_, joint_spec,
                            pxr::UsdPhysicsTokens->physicsLocalPos0,
                            pxr::SdfValueTypeNames->Float3),
        local_pos0);

    if (joint_prim_type == pxr::UsdPhysicsTokens->PhysicsRevoluteJoint ||
        joint_prim_type == pxr::UsdPhysicsTokens->PhysicsPrismaticJoint) {
      pxr::GfQuatf other_rot0(
          pxr::GfRotation(pxr::GfVec3f::ZAxis(), jnt_axis_parent_local)
              .GetQuat());

      SetAttributeDefault(
          layer_,
          CreateAttributeSpec(layer_, joint_spec,
                              pxr::UsdPhysicsTokens->physicsLocalRot0,
                              pxr::SdfValueTypeNames->Quatf),
          other_rot0);
    } else {
      // Fixed joints have no frame and no axis per se. We simply need the
      // rotation quaternion of the body its on.
      SetAttributeDefault(
          layer_,
          CreateAttributeSpec(layer_, joint_spec,
                              pxr::UsdPhysicsTokens->physicsLocalRot0,
                              pxr::SdfValueTypeNames->Quatf),
          body1_transform_local.ExtractRotationQuat());
    }

    if (joint) {
      mjtJoint type = (mjtJoint)model_->jnt_type[joint_id];

      // Joint-specific attributes
      if (type == mjJNT_HINGE || type == mjJNT_SLIDE) {
        // The joint motion occurs around/along the Z-axis of the joint frame
        // established by localRot0/1.
        SetAttributeDefault(
            layer_,
            CreateAttributeSpec(layer_, joint_spec,
                                pxr::UsdPhysicsTokens->physicsAxis,
                                pxr::SdfValueTypeNames->Token),
            pxr::UsdPhysicsTokens->z);  // "Z" axis
      }

      if (model_->jnt_limited[joint_id]) {
        float lower_limit = model_->jnt_range[joint_id * 2];
        float upper_limit = model_->jnt_range[joint_id * 2 + 1];

        if (type == mjJNT_HINGE) {
          // Convert radians to degrees for USD
          // As per the XML Reference, "mjModel always uses radians"
          lower_limit *= (180.0 / mjPI);
          upper_limit *= (180.0 / mjPI);
          SetAttributeDefault(
              layer_,
              CreateAttributeSpec(layer_, joint_spec,
                                  pxr::UsdPhysicsTokens->physicsLowerLimit,
                                  pxr::SdfValueTypeNames->Float),
              lower_limit);
          SetAttributeDefault(
              layer_,
              CreateAttributeSpec(layer_, joint_spec,
                                  pxr::UsdPhysicsTokens->physicsUpperLimit,
                                  pxr::SdfValueTypeNames->Float),
              upper_limit);
        } else if (type == mjJNT_SLIDE) {
          SetAttributeDefault(
              layer_,
              CreateAttributeSpec(layer_, joint_spec,
                                  pxr::UsdPhysicsTokens->physicsLowerLimit,
                                  pxr::SdfValueTypeNames->Float),
              lower_limit);
          SetAttributeDefault(
              layer_,
              CreateAttributeSpec(layer_, joint_spec,
                                  pxr::UsdPhysicsTokens->physicsUpperLimit,
                                  pxr::SdfValueTypeNames->Float),
              upper_limit);
        }
      }

      // Finally write the mjcPhysicsJointAPI attributes.
      ApplyApiSchema(layer_, joint_spec, MjcPhysicsTokens->MjcJointAPI);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Int,
                            MjcPhysicsTokens->mjcGroup, joint->group);

      WriteUniformAttribute(
          joint_spec, pxr::SdfValueTypeNames->DoubleArray,
          MjcPhysicsTokens->mjcSpringdamper,
          pxr::VtArray<double>(joint->springdamper, joint->springdamper + 2));

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->DoubleArray,
                            MjcPhysicsTokens->mjcSolreflimit,
                            pxr::VtArray<double>(joint->solref_limit,
                                                 joint->solref_limit + mjNREF));

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->DoubleArray,
                            MjcPhysicsTokens->mjcSolimplimit,
                            pxr::VtArray<double>(joint->solimp_limit,
                                                 joint->solimp_limit + mjNIMP));

      WriteUniformAttribute(
          joint_spec, pxr::SdfValueTypeNames->DoubleArray,
          MjcPhysicsTokens->mjcSolreffriction,
          pxr::VtArray<double>(joint->solref_friction,
                               joint->solref_friction + mjNREF));

      WriteUniformAttribute(
          joint_spec, pxr::SdfValueTypeNames->DoubleArray,
          MjcPhysicsTokens->mjcSolimpfriction,
          pxr::VtArray<double>(joint->solimp_friction,
                               joint->solimp_friction + mjNIMP));

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcStiffness, joint->stiffness[0]);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcActuatorfrcrangeMin,
                            joint->actfrcrange[0]);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcActuatorfrcrangeMax,
                            joint->actfrcrange[1]);

      pxr::TfToken actuatorfrclimited_token = MjcPhysicsTokens->auto_;
      if (joint->actfrclimited == mjLIMITED_TRUE) {
        actuatorfrclimited_token = MjcPhysicsTokens->true_;
      } else if (joint->actfrclimited == mjLIMITED_FALSE) {
        actuatorfrclimited_token = MjcPhysicsTokens->false_;
      }
      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Token,
                            MjcPhysicsTokens->mjcActuatorfrclimited,
                            actuatorfrclimited_token);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Bool,
                            MjcPhysicsTokens->mjcActuatorgravcomp,
                            static_cast<bool>(joint->actgravcomp));

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcMargin, joint->margin);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcRef, joint->ref);

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Double,
                            MjcPhysicsTokens->mjcSpringref, joint->springref);

      // Newton joint attributes (replaces deprecated mjc:armature / mjc:damping
      // / mjc:frictionloss)
      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonArmature,
                            static_cast<float>(joint->armature));

      // MuJoCo angular damping is per radian; Newton expects per degree.
      double damping = joint->damping[0];
      if (joint->type == mjJNT_HINGE) {
        damping *= std::numbers::pi / 180.0;
      }
      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonDamping,
                            static_cast<float>(damping));

      WriteUniformAttribute(joint_spec, pxr::SdfValueTypeNames->Float,
                            kTokens->newtonFriction,
                            static_cast<float>(joint->frictionloss));
    }
    if (joint_id >= 0) {
      joint_paths_[joint_id] = joint_spec->GetPath();
    }
  }

  void WriteCamera(mjsCamera *spec_cam, const mjsBody *body) {
    const auto &body_path = body_paths_[mjs_getId(body->element)];
    auto name = GetAvailablePrimName(*mjs_getName(spec_cam->element),
                                     pxr::UsdGeomTokens->Camera, body_path);
    // Create a root Xform for the world body with the model name if it exists
    // otherwise called 'World'.
    auto camera_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdGeomTokens->Camera);

    int cam_id = mjs_getId(spec_cam->element);
    auto transform = MujocoPosQuatToTransform(&model_->cam_pos[3 * cam_id],
                                              &model_->cam_quat[4 * cam_id]);
    WriteTransformXformOp(camera_spec, transform);
    WriteXformOpOrder(camera_spec,
                      pxr::VtArray<pxr::TfToken>{kTokens->xformOpTransform});

    // If the camera intrinsics are specified, then it is important that we
    // reproduce the code in mujoco/src/engine/engine_vis_visualize.c
    const float *cam_sensorsize = &model_->cam_sensorsize[cam_id * 2];
    bool use_intrinsic = cam_sensorsize[1] > 0.0f;
    float znear = spec_->visual.map.znear * model_->stat.extent * 100;
    float zfar = spec_->visual.map.zfar * model_->stat.extent * 100;
    mjtNum fovy = model_->cam_fovy[cam_id];
    const float *cam_intrinsic = &model_->cam_intrinsic[cam_id * 4];

    const float aspect_ratio =
        use_intrinsic ? cam_sensorsize[0] / cam_sensorsize[1] : 4.0f / 3;
    float vertical_apperture =
        2 * znear *
        (use_intrinsic ? 1.0f / cam_intrinsic[1] *
                             (cam_sensorsize[1] / 2.f - cam_intrinsic[3])
                       : mju_tan((fovy / 2) * (std::numbers::pi / 180.0)));
    float horizontal_aperture =
        use_intrinsic ? 2 * znear / cam_intrinsic[0] *
                            (cam_sensorsize[0] / 2.f - cam_intrinsic[2])
                      : vertical_apperture * aspect_ratio;

    WriteUniformAttribute(camera_spec, pxr::SdfValueTypeNames->Float2,
                          pxr::UsdGeomTokens->clippingRange,
                          pxr::GfVec2f(znear, zfar));
    WriteUniformAttribute(camera_spec, pxr::SdfValueTypeNames->Float,
                          pxr::UsdGeomTokens->focalLength, znear);
    WriteUniformAttribute(camera_spec, pxr::SdfValueTypeNames->Float,
                          pxr::UsdGeomTokens->verticalAperture,
                          vertical_apperture);
    WriteUniformAttribute(camera_spec, pxr::SdfValueTypeNames->Float,
                          pxr::UsdGeomTokens->horizontalAperture,
                          horizontal_aperture);
  }

  void WriteCameras(mjsBody *body) {
    mjsCamera *cam = mjs_asCamera(mjs_firstChild(body, mjOBJ_CAMERA, false));
    while (cam) {
      if (ShouldWrite(cam->element)) {
        WriteCamera(cam, body);
      }
      cam = mjs_asCamera(mjs_nextChild(body, cam->element, false));
    }
  }

  void WriteLight(mjsLight *light, const mjsBody *body) {
    const auto &body_path = body_paths_[mjs_getId(body->element)];
    auto name = GetAvailablePrimName(*mjs_getName(light->element),
                                     kTokens->light, body_path);
    // Create a root Xform for the world body with the model name if it exists
    // otherwise called 'World'.
    auto light_spec =
        CreatePrimSpec(layer_, body_path, name, pxr::UsdLuxTokens->SphereLight);

    int light_id = mjs_getId(light->element);
    pxr::GfVec3d dir(model_->light_dir[3 * light_id],
                     model_->light_dir[3 * light_id + 1],
                     model_->light_dir[3 * light_id + 2]);
    pxr::GfRotation rotation;
    rotation.SetIdentity();
    if (dir.GetLengthSq() > 1e-12) {
      rotation = pxr::GfRotation(pxr::GfVec3d(0, 0, -1), dir);
    }
    pxr::GfVec3d translation(model_->light_pos[3 * light_id],
                             model_->light_pos[3 * light_id + 1],
                             model_->light_pos[3 * light_id + 2]);
    pxr::GfMatrix4d transform;
    transform.SetTransform(rotation, translation);
    WriteTransformXformOp(light_spec, transform);
    WriteXformOpOrder(light_spec,
                      pxr::VtArray<pxr::TfToken>{kTokens->xformOpTransform});
  }

  void WriteLights(mjsBody *body) {
    mjsLight *light = mjs_asLight(mjs_firstChild(body, mjOBJ_LIGHT, false));
    while (light) {
      if (ShouldWrite(light->element)) {
        WriteLight(light, body);
      }
      light = mjs_asLight(mjs_nextChild(body, light->element, false));
    }
  }

  void WriteBody(mjsBody *body) {
    int body_id = mjs_getId(body->element);
    // This should be safe as we process parent bodies before children.
    mjsBody *parent = mjs_getParent(body->element);
    int parent_id = mjs_getId(parent->element);
    pxr::SdfPath parent_path = body_paths_[parent_id];
    pxr::TfToken body_name = GetValidPrimName(*mjs_getName(body->element));

    // Create Xform prim for body.
    auto body_spec = CreatePrimSpec(layer_, parent_path, body_name,
                                            pxr::UsdGeomTokens->Xform);
    // The parent_path will be a component which makes the actual articulated
    // bodies subcomponents.
    auto kind = parent_id == kWorldIndex ? pxr::KindTokens->component
                                         : pxr::KindTokens->subcomponent;
    SetPrimKind(layer_, body_spec, kind);

    // If the parent is not the world body, but is child of the world body
    // then we need to apply the articulation root API.
    if (parent_id != kWorldIndex) {
      int parent_parent_id = mjs_getId(mjs_getParent(parent->element)->element);
      // We guard against applying the API more than once, which can happen when
      // there are multiple children.
      if (parent_parent_id == kWorldIndex &&
          articulation_roots_.find(parent_id) == articulation_roots_.end()) {
        ApplyApiSchema(layer_, layer_->GetPrimAtPath(parent_path),
                       pxr::UsdPhysicsTokens->PhysicsArticulationRootAPI);
        articulation_roots_.insert(parent_id);
      }
    }

    // If the body had a mass specified then it must have either inertia or
    // fullinertia specified per inertia element XML documentation.
    // Therefore it is sufficient to check if the mass is non-zero to see if
    // we should set inertial attributes on the body.
    //
    // Note that if the user has NOT specified any inertial properties then
    // we don't want to pull values from the compiled model since coming back
    // into Mujoco would take those values instead of computing them
    // automatically from the subtree.
    if (body->mass > 0) {
      // User might have specified the inertia via fullinertia and the
      // compiler has extracted all values properly. So leverage those
      // instead of doing the computation ourselves here.
      ApplyApiSchema(layer_, body_spec, pxr::UsdPhysicsTokens->PhysicsMassAPI);
      WriteUniformAttribute(body_spec, pxr::SdfValueTypeNames->Float,
                            pxr::UsdPhysicsTokens->physicsMass,
                            (float)model_->body_mass[body_id]);

      mjtNum *body_ipos = &model_->body_ipos[body_id * 3];
      pxr::GfVec3f inertial_pos(body_ipos[0], body_ipos[1], body_ipos[2]);
      WriteUniformAttribute(body_spec, pxr::SdfValueTypeNames->Point3f,
                            pxr::UsdPhysicsTokens->physicsCenterOfMass,
                            inertial_pos);

      mjtNum *body_iquat = &model_->body_iquat[body_id * 4];
      pxr::GfQuatf inertial_frame(body_iquat[0], body_iquat[1], body_iquat[2],
                                  body_iquat[3]);
      WriteUniformAttribute(body_spec, pxr::SdfValueTypeNames->Quatf,
                            pxr::UsdPhysicsTokens->physicsPrincipalAxes,
                            inertial_frame);

      mjtNum *inertia = &model_->body_inertia[body_id * 3];
      pxr::GfVec3f diag_inertia(inertia[0], inertia[1], inertia[2]);
      WriteUniformAttribute(body_spec, pxr::SdfValueTypeNames->Float3,
                            pxr::UsdPhysicsTokens->physicsDiagonalInertia,
                            diag_inertia);
    }

    ApplyApiSchema(layer_, body_spec,
                   pxr::UsdPhysicsTokens->PhysicsRigidBodyAPI);

    // Create classes if necessary
    mjsDefault *spec_default = mjs_getDefault(body->element);

    pxr::TfToken body_class_name =
        GetValidPrimName(*mjs_getName(spec_default->element));
    pxr::SdfPath body_class_path = class_path_.AppendChild(body_class_name);
    if (!layer_->HasSpec(body_class_path)) {
      CreateClassSpec(layer_, class_path_, body_class_name);
    }

    // Create XformOp attribute for body transform.

    auto xform_op_spec =
        CreateAttributeSpec(layer_, body_spec, kTokens->xformOpTransform,
                            pxr::SdfValueTypeNames->Matrix4d);
    // mjModel will have all frames already accounted for so no need to worry
    // about them here.
    auto body_xform = MujocoPosQuatToTransform(&model_->body_pos[body_id * 3],
                                               &model_->body_quat[body_id * 4]);
    SetAttributeDefault(layer_, xform_op_spec, body_xform);

    // Create XformOpOrder attribute for body transform order.
    // For us this is simply the transform we authored above.
    WriteXformOpOrder(body_spec,
                      pxr::VtArray<pxr::TfToken>{kTokens->xformOpTransform});

    pxr::VtDictionary customData;
    customData[kTokens->body_name] = *mjs_getName(body->element);
    body_spec->SetField(pxr::SdfFieldKeys->CustomData, customData);

    body_paths_[body_id] = body_spec->GetPath();
    mujoco::usd::SetUsdPrimPathUserValue(body->element, body_spec->GetPath());
  }

  void WriteBodies() {
    mjsBody *body = mjs_asBody(mjs_firstElement(spec_, mjOBJ_BODY));
    while (body) {
      // If this body originally came from USD, don't try and write it
      // again.
      if (ShouldWrite(body->element)) {
        // Only write a rigidbody if we are not the world body.
        // We fall through since the world body might have static
        // geom children.
        if (mjs_getId(body->element) != kWorldIndex) {
          WriteBody(body);
        }
        WriteSites(body);
        WriteGeoms(body);
        WriteJoints(body);
        WriteCameras(body);
        WriteLights(body);
      }
      body = mjs_asBody(mjs_nextElement(spec_, body->element));
    }
  }

  pxr::SdfPrimSpecHandle WriteWorldBody(const size_t body_index) {
    // Create a root Xform for the world body with the model name if it exists
    // otherwise called 'World'.
    auto name = GetAvailablePrimName(*spec_->modelname, kTokens->world,
                                     pxr::SdfPath::AbsoluteRootPath());
    auto world_group_spec =
        CreatePrimSpec(layer_, pxr::SdfPath::AbsoluteRootPath(), name,
                       pxr::UsdGeomTokens->Xform);
    SetPrimKind(layer_, world_group_spec, pxr::KindTokens->group);

    return world_group_spec;
  }
};

namespace mujoco {
namespace usd {

bool WriteSpecToData(mjSpec *spec, pxr::SdfLayerRefPtr layer, bool skip_elems_from_usd) {
  mjModel *model = mj_compile(spec, nullptr);
  if (model == nullptr) {
    TF_ERROR(MujocoCompilationError, "%s", mjs_getError(spec));
    return false;
  }

  ModelWriter(spec, model, layer, skip_elems_from_usd).Write();

  return true;
}

}  // namespace usd
}  // namespace mujoco
