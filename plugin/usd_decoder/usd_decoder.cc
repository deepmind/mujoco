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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <mujoco/experimental/usd/mjcPhysics/actuator.h>
#include <mujoco/experimental/usd/mjcPhysics/collisionAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/equalityAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/equalityConnectAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/equalityJointAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/equalityWeldAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/imageableAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/jointAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/keyframe.h>
#include <mujoco/experimental/usd/mjcPhysics/materialAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/meshCollisionAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/sceneAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/siteAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/tendon.h>
#include <mujoco/experimental/usd/mjcPhysics/tokens.h>
#include <mujoco/experimental/usd/utils.h>
#include <mujoco/mujoco.h>
#include "kinematic_tree.h"
#include "material_parsing.h"
#include "newton_tokens.h"
#include <pxr/base/gf/declare.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/matrix4f.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/tf/staticData.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primFlags.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/plane.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
#include <pxr/usd/usdPhysics/filteredPairsAPI.h>
#include <pxr/usd/usdPhysics/fixedJoint.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/massAPI.h>
#include <pxr/usd/usdPhysics/materialAPI.h>
#include <pxr/usd/usdPhysics/meshCollisionAPI.h>
#include <pxr/usd/usdPhysics/prismaticJoint.h>
#include <pxr/usd/usdPhysics/revoluteJoint.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#include <pxr/usd/usdPhysics/scene.h>
#include <pxr/usd/usdPhysics/sphericalJoint.h>
#include <pxr/usd/usdPhysics/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

using mujoco::NewtonTokens;
using mujoco::usd::SetUsdPrimPathUserValue;
using pxr::MjcPhysicsTokens;
using pxr::TfToken;
template <typename T>
using TfStaticData = pxr::TfStaticData<T>;

struct UsdCaches {
  pxr::UsdGeomXformCache xform_cache;
  pxr::UsdShadeMaterialBindingAPI::BindingsCache bindings_cache;
  pxr::UsdShadeMaterialBindingAPI::CollectionQueryCache collection_query_cache;
  std::map<pxr::SdfPath, mjsMaterial*> parsed_materials;
};

std::string GetStageModelName(const pxr::UsdStageRefPtr& stage) {
  pxr::UsdPrim default_prim = stage->GetDefaultPrim();
  if (default_prim) {
    std::string display_name = default_prim.GetDisplayName();
    if (!display_name.empty()) {
      return display_name;
    }
    std::string prim_name = default_prim.GetName().GetString();
    if (!prim_name.empty()) {
      return prim_name;
    }
  }

  pxr::SdfLayerHandle root_layer = stage->GetRootLayer();
  if (root_layer && !root_layer->IsAnonymous()) {
    std::string layer_stem =
        pxr::TfStringGetBeforeSuffix(root_layer->GetDisplayName());
    if (!layer_stem.empty()) {
      return layer_stem;
    }
  }

  return "";
}

void SetDoubleArrFromGfVec3d(double* to, const pxr::GfVec3d& from) {
  to[0] = from[0];
  to[1] = from[1];
  to[2] = from[2];
}

void SetMjtNumArrFromGfVec3d(mjtNum* to, const pxr::GfVec3d& from) {
  to[0] = from[0];
  to[1] = from[1];
  to[2] = from[2];
}

void SetDoubleArrFromGfQuatd(double* to, const pxr::GfQuatd& from) {
  // pxr::GfQuatd uses wxyz stored as a real and an imaginary component.
  to[0] = from.GetReal();
  to[1] = from.GetImaginary()[0];
  to[2] = from.GetImaginary()[1];
  to[3] = from.GetImaginary()[2];
}

template <typename T>
void SetLocalPoseFromPrim(const pxr::UsdPrim& prim,
                          const pxr::UsdPrim& parent_prim, T* element,
                          pxr::UsdGeomXformCache& xform_cache) {
  pxr::GfMatrix4d xform = xform_cache.GetLocalToWorldTransform(prim);
  pxr::GfMatrix4d parent_xform =
      xform_cache.GetLocalToWorldTransform(parent_prim);
  pxr::GfMatrix4d relative_xform = xform * parent_xform.GetInverse();

  pxr::GfVec3d translation = relative_xform.ExtractTranslation();
  pxr::GfQuatd rotation =
      relative_xform.RemoveScaleShear().ExtractRotationQuat();

  SetDoubleArrFromGfVec3d(element->pos, translation);
  SetDoubleArrFromGfQuatd(element->quat, rotation);
}

pxr::GfVec3d GetScale(const pxr::GfMatrix4d& matrix) {
  pxr::GfMatrix4d rotation;
  pxr::GfVec3d scale;
  pxr::GfMatrix4d u;
  pxr::GfVec3d translation;
  pxr::GfMatrix4d p;
  if (!matrix.Factor(&rotation, &scale, &u, &translation, &p)) {
    // In the event that we could not factorize, return the identity.
    mju_error("Could not factorise matrix.");
    return pxr::GfVec3d(1, 1, 1);
  }
  return scale;
}

bool IsUniformScale(const pxr::GfVec3d& scale) {
  static const double epsilon = 1e-6;
  return fabs(scale[0] - scale[1]) < epsilon &&
         fabs(scale[1] - scale[2]) < epsilon;
}

template <typename Shape, typename Element>
bool ParseAxialPrimitive(const Shape& shape, const char* name,
                         mjtGeom geom_type, const pxr::GfVec3d& scale,
                         Element* element) {
  element->type = geom_type;
  double radius;
  if (!shape.GetRadiusAttr().template Get<double>(&radius)) {
    mju_error("Could not get %s radius attr.", name);
    return false;
  }

  double height;
  if (!shape.GetHeightAttr().template Get<double>(&height)) {
    mju_error("Could not get %s height attr.", name);
    return false;
  }

  TfToken axis = pxr::UsdGeomTokens->z;
  shape.GetAxisAttr().Get(&axis);

  // Scale radius and half-height using the radial and axial components for the
  // authored USD axis, and rotate local +Z to align with X or Y when needed.
  double radial_scale_a = std::abs(scale[0]);
  double radial_scale_b = std::abs(scale[1]);
  double axial_scale = std::abs(scale[2]);
  pxr::GfQuatd axis_rot = pxr::GfQuatd::GetIdentity();
  if (axis == pxr::UsdGeomTokens->x) {
    radial_scale_a = std::abs(scale[1]);
    radial_scale_b = std::abs(scale[2]);
    axial_scale = std::abs(scale[0]);
    axis_rot =
        pxr::GfRotation(pxr::GfVec3d::ZAxis(), pxr::GfVec3d::XAxis()).GetQuat();
  } else if (axis == pxr::UsdGeomTokens->y) {
    radial_scale_a = std::abs(scale[0]);
    radial_scale_b = std::abs(scale[2]);
    axial_scale = std::abs(scale[1]);
    axis_rot =
        pxr::GfRotation(pxr::GfVec3d::ZAxis(), pxr::GfVec3d::YAxis()).GetQuat();
  }
  if (std::abs(radial_scale_a - radial_scale_b) >= 1e-6) {
    mju_warning(
        "Non-uniform radial scale on %s prim %s; using max radial scale.", name,
        shape.GetPath().GetText());
  }
  double radial_scale = std::max(radial_scale_a, radial_scale_b);

  element->size[0] = radial_scale * radius;
  element->size[1] = axial_scale * height / 2.0;
  element->size[2] = 0;

  pxr::GfQuatd current_rot(element->quat[0], element->quat[1], element->quat[2],
                           element->quat[3]);
  SetDoubleArrFromGfQuatd(element->quat, current_rot * axis_rot);
  return true;
}

template <typename T>
bool MaybeParseGeomPrimitive(const pxr::UsdPrim& prim, T* element,
                             pxr::UsdGeomXformCache& xform_cache) {
  auto world_xform = xform_cache.GetLocalToWorldTransform(prim);
  auto scale = GetScale(world_xform);
  if (prim.IsA<pxr::UsdGeomSphere>()) {
    double radius;
    if (!pxr::UsdGeomSphere(prim).GetRadiusAttr().Get<double>(&radius)) {
      mju_error("Could not get sphere radius attr.");
      return false;
    }
    // If scale is uniform (or *very close*) then create a sphere, otherwise
    // this is an ellipsoid.
    if (IsUniformScale(scale)) {
      element->type = mjGEOM_SPHERE;
      element->size[0] = scale[0] * radius;
      element->size[1] = scale[0] * radius;
      element->size[2] = scale[0] * radius;
    } else {
      element->type = mjGEOM_ELLIPSOID;
      element->size[0] = scale[0] * radius;
      element->size[1] = scale[1] * radius;
      element->size[2] = scale[2] * radius;
    }
  } else if (prim.IsA<pxr::UsdGeomCylinder>()) {
    return ParseAxialPrimitive(pxr::UsdGeomCylinder(prim), "cylinder",
                               mjGEOM_CYLINDER, scale, element);
  } else if (prim.IsA<pxr::UsdGeomCapsule>()) {
    return ParseAxialPrimitive(pxr::UsdGeomCapsule(prim), "capsule",
                               mjGEOM_CAPSULE, scale, element);
  } else if (prim.IsA<pxr::UsdGeomCube>()) {
    element->type = mjGEOM_BOX;
    auto cube = pxr::UsdGeomCube(prim);
    double size;
    if (!cube.GetSizeAttr().Get<double>(&size)) {
      mju_error("Could not get cube size attr.");
      return false;
    }
    // MuJoCo uses half-length for box size.
    size = size / 2;
    element->size[0] = scale[0] * size;
    element->size[1] = scale[1] * size;
    element->size[2] = scale[2] * size;
  } else if (prim.IsA<pxr::UsdGeomPlane>()) {
    element->type = mjGEOM_PLANE;

    pxr::UsdGeomPlane plane(prim);
    TfToken axis;
    if (!plane.GetAxisAttr().Get(&axis)) {
      mju_error("Could not get plane axis attr.");
      return false;
    }
    if (axis != pxr::UsdGeomTokens->z) {
      mju_error("Only z-axis planes are supported.");
      return false;
    }

    double length;
    if (!plane.GetLengthAttr().Get(&length)) {
      mju_error("Could not get plane length attr.");
      return false;
    }
    double width;
    if (!plane.GetWidthAttr().Get(&width)) {
      mju_error("Could not get plane width attr.");
      return false;
    }
    // MuJoCo uses half-length for plane size.
    width = width / 2;
    length = length / 2;
    // Plane geoms in mjc are always infinite. Scale is used for visualization.
    element->size[0] = scale[0] * width;
    element->size[1] = scale[1] * length;
    element->size[2] = scale[2];
  } else {
    return false;
  }

  return true;
}

mjsMesh* ParseUsdMesh(mjSpec* spec, const pxr::UsdPrim& prim, mjsGeom* geom,
                      pxr::UsdGeomXformCache& xform_cache) {
  if (!prim.IsA<pxr::UsdGeomMesh>()) {
    return nullptr;
  }
  mjsMesh* mesh = mjs_addMesh(spec, nullptr);

  SetUsdPrimPathUserValue(mesh->element, prim.GetPath());

  geom->type = mjGEOM_MESH;
  pxr::UsdGeomMesh usd_mesh(prim);
  std::vector<float> uservert;
  std::vector<int> userface;

  pxr::VtVec3fArray points;
  usd_mesh.GetPointsAttr().Get(&points);

  pxr::VtVec2fArray uvs;
  pxr::VtIntArray uv_indices;
  std::vector<float> texcoord;
  std::vector<int> userfacetexcoord;

  pxr::UsdGeomPrimvarsAPI primvarsAPI(prim);
  pxr::UsdGeomPrimvar st_primvar =
      primvarsAPI.FindPrimvarWithInheritance(pxr::TfToken("st"));

  uservert.reserve(points.size() * 3);
  for (const auto& pt : points) {
    uservert.push_back(pt[0]);
    uservert.push_back(pt[1]);
    uservert.push_back(pt[2]);
  }

  bool has_uvs = st_primvar.HasAuthoredValue();
  bool face_varying_uvs = has_uvs && st_primvar.GetInterpolation() ==
                                         pxr::UsdGeomTokens->faceVarying;
  bool st_indexed = st_primvar.IsIndexed();
  if (has_uvs) {
    pxr::VtVec2fArray uvs;
    st_primvar.Get(&uvs);

    if (st_indexed) {
      st_primvar.GetIndices(&uv_indices);
    }

    // If the UVs do not vary per point per face, then we need to compute the
    // effective uv array taking indexing into account as we do not support UV
    // indexing in mujoco.
    if (!face_varying_uvs && st_primvar.IsIndexed()) {
      texcoord.reserve(uv_indices.size() * 2);
      for (const auto& idx : uv_indices) {
        auto uv = uvs[idx];
        texcoord.push_back(uv[0]);
        // USD origin is bottom left, MuJoCo is top left.
        texcoord.push_back(1.0f - uv[1]);
      }
    } else {
      texcoord.reserve(uvs.size() * 2);
      for (const auto& uv : uvs) {
        texcoord.push_back(uv[0]);
        // USD origin is bottom left, MuJoCo is top left.
        texcoord.push_back(1.0f - uv[1]);
      }
    }
    mjs_setFloat(mesh->usertexcoord, texcoord.data(), texcoord.size());
  }

  pxr::VtIntArray indices;
  usd_mesh.GetFaceVertexIndicesAttr().Get(&indices);
  pxr::VtIntArray counts;
  usd_mesh.GetFaceVertexCountsAttr().Get(&counts);

  userface.reserve(indices.size());
  int vtx_idx = 0;
  for (int count : counts) {
    int k = 1;
    // If the prim is a triangle create a triangle fan rooted
    // at the first index.
    while (k < count - 1) {
      userface.push_back(indices[vtx_idx]);
      userface.push_back(indices[vtx_idx + k]);
      userface.push_back(indices[vtx_idx + k + 1]);

      // If the UVs vary per face, then we need to compute the effective
      // index array after we've created the triangle fan for non triangular
      // faces.
      if (face_varying_uvs) {
        userfacetexcoord.push_back(st_indexed ? uv_indices[vtx_idx] : vtx_idx);
        userfacetexcoord.push_back(st_indexed ? uv_indices[vtx_idx + k]
                                              : vtx_idx + k);
        userfacetexcoord.push_back(st_indexed ? uv_indices[vtx_idx + k + 1]
                                              : vtx_idx + k + 1);
      }
      k++;
    }
    vtx_idx += count;
  }

  auto world_xform = xform_cache.GetLocalToWorldTransform(prim);
  auto scale = GetScale(world_xform);
  mesh->scale[0] = scale[0];
  mesh->scale[1] = scale[1];
  mesh->scale[2] = scale[2];

  std::string mesh_name = usd_mesh.GetPath().GetAsString();
  mjs_setName(mesh->element, mesh_name.c_str());
  mjs_setFloat(mesh->uservert, uservert.data(), uservert.size());
  mjs_setInt(mesh->userface, userface.data(), userface.size());

  if (face_varying_uvs) {
    mjs_setInt(mesh->userfacetexcoord, userfacetexcoord.data(),
               userfacetexcoord.size());
  }

  mjs_setString(geom->meshname, mesh_name.c_str());
  return mesh;
}

void SetGravityAttributes(
    mjSpec* spec, const pxr::UsdStageRefPtr stage,
    std::optional<pxr::GfVec3f> gravity_direction = std::nullopt,
    std::optional<float> gravity_magnitude = std::nullopt) {
  // Parse gravity and gravity direction.
  if (!gravity_direction.has_value()) {
    TfToken up_axis = pxr::UsdGeomGetStageUpAxis(stage);
    if (up_axis == pxr::UsdGeomTokens->y) {
      gravity_direction = pxr::GfVec3f(0, -1, 0);
    } else if (up_axis == pxr::UsdGeomTokens->z) {
      gravity_direction = pxr::GfVec3f(0, 0, -1);
    } else {
      mju_error("Invalid stage up axis token %s", up_axis.GetString().c_str());
    }
  }

  if (!gravity_magnitude.has_value()) {
    gravity_magnitude = 9.81f / pxr::UsdGeomGetStageMetersPerUnit(stage);
  }

  pxr::GfVec3f gravity = gravity_direction.value() * gravity_magnitude.value();

  spec->option.gravity[0] = gravity[0];
  spec->option.gravity[1] = gravity[1];
  spec->option.gravity[2] = gravity[2];
}

void ParseUsdPhysicsScene(mjSpec* spec,
                          const pxr::UsdPhysicsScene& physics_scene) {
  std::optional<pxr::GfVec3f> gravity_direction = std::nullopt;
  std::optional<float> gravity_magnitude = std::nullopt;
  auto stage = physics_scene.GetPrim().GetStage();

  // Parse gravity and gravity direction.
  auto gravity_direction_attr = physics_scene.GetGravityDirectionAttr();
  if (gravity_direction_attr.HasAuthoredValue()) {
    pxr::GfVec3f authored_gravity_dir;
    gravity_direction_attr.Get(&authored_gravity_dir);
    gravity_direction = authored_gravity_dir;
  }

  auto gravity_magnitude_attr = physics_scene.GetGravityMagnitudeAttr();
  if (gravity_magnitude_attr.HasAuthoredValue()) {
    float authored_gravity_magnitude;
    gravity_magnitude_attr.Get(&authored_gravity_magnitude);
    gravity_magnitude = authored_gravity_magnitude;
  }

  SetGravityAttributes(spec, stage, gravity_direction, gravity_magnitude);

  // Parse Newton scene attributes if present (works for Newton-only files)
  pxr::UsdPrim scene_prim = physics_scene.GetPrim();
  auto newton_iterations =
      scene_prim.GetAttribute(NewtonTokens->newtonMaxSolverIterations);
  if (newton_iterations && newton_iterations.HasAuthoredValue()) {
    int val;
    newton_iterations.Get(&val);
    if (val >= 0) spec->option.iterations = val;
  }
  auto newton_timesteps =
      scene_prim.GetAttribute(NewtonTokens->newtonTimeStepsPerSecond);
  if (newton_timesteps && newton_timesteps.HasAuthoredValue()) {
    int val;
    newton_timesteps.Get(&val);
    if (val > 0) spec->option.timestep = 1.0 / val;
  }
  auto newton_gravity =
      scene_prim.GetAttribute(NewtonTokens->newtonGravityEnabled);
  if (newton_gravity && newton_gravity.HasAuthoredValue()) {
    bool enabled;
    newton_gravity.Get(&enabled);
    if (!enabled) {
      spec->option.disableflags |= mjDSBL_GRAVITY;
    }
  }

  // Early exit if there's no MjcPhysicsSceneAPI applied.
  if (!scene_prim.HasAPI<pxr::MjcPhysicsSceneAPI>()) {
    return;
  }
  auto mjc_physics_scene = pxr::MjcPhysicsSceneAPI(scene_prim);

  // MJC values override Newton values only when explicitly authored.
  auto timestep_attr = mjc_physics_scene.GetTimestepAttr();
  if (timestep_attr.HasAuthoredValue()) {
    double timestep;
    timestep_attr.Get(&timestep);
    spec->option.timestep = timestep;
    mju_warning(
        "Scene '%s' uses deprecated mjc:option:timestep. "
        "Please migrate to newton:timeStepsPerSecond.",
        scene_prim.GetPath().GetText());
  }

  double impratio;
  mjc_physics_scene.GetImpRatioAttr().Get(&impratio);
  spec->option.impratio = impratio;

  double tolerance;
  mjc_physics_scene.GetToleranceAttr().Get(&tolerance);
  spec->option.tolerance = tolerance;

  double ls_tolerance;
  mjc_physics_scene.GetLSToleranceAttr().Get(&ls_tolerance);
  spec->option.ls_tolerance = ls_tolerance;

  double noslip_tolerance;
  mjc_physics_scene.GetNoslipToleranceAttr().Get(&noslip_tolerance);
  spec->option.noslip_tolerance = noslip_tolerance;

  double ccd_tolerance;
  mjc_physics_scene.GetCCDToleranceAttr().Get(&ccd_tolerance);
  spec->option.ccd_tolerance = ccd_tolerance;

  double sleep_tolerance;
  mjc_physics_scene.GetSleepToleranceAttr().Get(&sleep_tolerance);
  spec->option.sleep_tolerance = sleep_tolerance;

  pxr::GfVec3d wind;
  mjc_physics_scene.GetWindAttr().Get(&wind);
  SetMjtNumArrFromGfVec3d(spec->option.wind, wind);

  pxr::GfVec3d magnetic;
  mjc_physics_scene.GetMagneticAttr().Get(&magnetic);
  SetMjtNumArrFromGfVec3d(spec->option.magnetic, magnetic);

  double density;
  mjc_physics_scene.GetDensityAttr().Get(&density);
  spec->option.density = density;

  double viscosity;
  mjc_physics_scene.GetViscosityAttr().Get(&viscosity);
  spec->option.viscosity = viscosity;

  double o_margin;
  mjc_physics_scene.GetOMarginAttr().Get(&o_margin);
  spec->option.o_margin = o_margin;

  pxr::VtDoubleArray o_solref;
  mjc_physics_scene.GetOSolRefAttr().Get(&o_solref);
  if (o_solref.size() != mjNREF) {
    mju_error("Invalid size for o_solref attribute: %zu expected %d",
              o_solref.size(), mjNREF);
    return;
  }
  for (int i = 0; i < mjNREF; ++i) {
    spec->option.o_solref[i] = o_solref[i];
  }

  pxr::VtDoubleArray o_solimp;
  mjc_physics_scene.GetOSolImpAttr().Get(&o_solimp);
  if (o_solimp.size() != mjNIMP) {
    mju_error("Invalid size for o_solimp attribute: %zu expected %d",
              o_solimp.size(), mjNIMP);
    return;
  }
  for (int i = 0; i < mjNIMP; ++i) {
    spec->option.o_solimp[i] = o_solimp[i];
  }

  pxr::VtDoubleArray o_friction;
  mjc_physics_scene.GetOFrictionAttr().Get(&o_friction);
  if (o_friction.size() != 5) {
    mju_error("Invalid size for o_friction attribute: %zu expected %d",
              o_friction.size(), 5);
    return;
  }
  for (int i = 0; i < 5; ++i) {
    spec->option.o_friction[i] = o_friction[i];
  }

  TfToken integrator;
  mjc_physics_scene.GetIntegratorAttr().Get(&integrator);
  if (integrator == MjcPhysicsTokens->euler) {
    spec->option.integrator = mjINT_EULER;
  } else if (integrator == MjcPhysicsTokens->rk4) {
    spec->option.integrator = mjINT_RK4;
  } else if (integrator == MjcPhysicsTokens->implicit) {
    spec->option.integrator = mjINT_IMPLICIT;
  } else if (integrator == MjcPhysicsTokens->implicitfast) {
    spec->option.integrator = mjINT_IMPLICITFAST;
  }

  TfToken cone;
  mjc_physics_scene.GetConeAttr().Get(&cone);
  if (cone == MjcPhysicsTokens->elliptic) {
    spec->option.cone = mjCONE_ELLIPTIC;
  } else if (cone == MjcPhysicsTokens->pyramidal) {
    spec->option.cone = mjCONE_PYRAMIDAL;
  }

  TfToken jacobian;
  mjc_physics_scene.GetJacobianAttr().Get(&jacobian);
  if (jacobian == MjcPhysicsTokens->auto_) {
    spec->option.jacobian = mjJAC_AUTO;
  } else if (jacobian == MjcPhysicsTokens->dense) {
    spec->option.jacobian = mjJAC_DENSE;
  } else if (jacobian == MjcPhysicsTokens->sparse) {
    spec->option.jacobian = mjJAC_SPARSE;
  }

  TfToken solver;
  mjc_physics_scene.GetSolverAttr().Get(&solver);
  if (solver == MjcPhysicsTokens->newton) {
    spec->option.solver = mjSOL_NEWTON;
  } else if (solver == MjcPhysicsTokens->cg) {
    spec->option.solver = mjSOL_CG;
  } else if (solver == MjcPhysicsTokens->pgs) {
    spec->option.solver = mjSOL_PGS;
  }

  auto iterations_attr = mjc_physics_scene.GetIterationsAttr();
  if (iterations_attr.HasAuthoredValue()) {
    int iterations;
    iterations_attr.Get(&iterations);
    spec->option.iterations = iterations;
    mju_warning(
        "Scene '%s' uses deprecated mjc:option:iterations. "
        "Please migrate to newton:maxSolverIterations.",
        scene_prim.GetPath().GetText());
  }

  int ls_iterations;
  mjc_physics_scene.GetLSIterationsAttr().Get(&ls_iterations);
  spec->option.ls_iterations = ls_iterations;

  int noslip_iterations;
  mjc_physics_scene.GetNoslipIterationsAttr().Get(&noslip_iterations);
  spec->option.noslip_iterations = noslip_iterations;

  int ccd_iterations;
  mjc_physics_scene.GetCCDIterationsAttr().Get(&ccd_iterations);
  spec->option.ccd_iterations = ccd_iterations;

  int sdf_initpoints;
  mjc_physics_scene.GetSDFInitPointsAttr().Get(&sdf_initpoints);
  spec->option.sdf_initpoints = sdf_initpoints;

  int sdf_iterations;
  mjc_physics_scene.GetSDFIterationsAttr().Get(&sdf_iterations);
  spec->option.sdf_iterations = sdf_iterations;

  bool constraint_flag;
  mjc_physics_scene.GetConstraintFlagAttr().Get(&constraint_flag);
  spec->option.disableflags |= (!constraint_flag ? mjDSBL_CONSTRAINT : 0);

  bool equality_flag;
  mjc_physics_scene.GetEqualityFlagAttr().Get(&equality_flag);
  spec->option.disableflags |= (!equality_flag ? mjDSBL_EQUALITY : 0);

  bool frictionloss_flag;
  mjc_physics_scene.GetFrictionLossFlagAttr().Get(&frictionloss_flag);
  spec->option.disableflags |= (!frictionloss_flag ? mjDSBL_FRICTIONLOSS : 0);

  bool limit_flag;
  mjc_physics_scene.GetLimitFlagAttr().Get(&limit_flag);
  spec->option.disableflags |= (!limit_flag ? mjDSBL_LIMIT : 0);

  bool contact_flag;
  mjc_physics_scene.GetContactFlagAttr().Get(&contact_flag);
  spec->option.disableflags |= (!contact_flag ? mjDSBL_CONTACT : 0);

  bool spring_flag;
  mjc_physics_scene.GetSpringFlagAttr().Get(&spring_flag);
  spec->option.disableflags |= (!spring_flag ? mjDSBL_SPRING : 0);

  bool damper_flag;
  mjc_physics_scene.GetDamperFlagAttr().Get(&damper_flag);
  spec->option.disableflags |= (!damper_flag ? mjDSBL_DAMPER : 0);

  auto gravity_flag_attr = mjc_physics_scene.GetGravityFlagAttr();
  if (gravity_flag_attr.HasAuthoredValue()) {
    bool gravity_flag;
    gravity_flag_attr.Get(&gravity_flag);
    if (!gravity_flag) {
      spec->option.disableflags |= mjDSBL_GRAVITY;
    } else {
      spec->option.disableflags &= ~mjDSBL_GRAVITY;
    }
    mju_warning(
        "Scene '%s' uses deprecated mjc:flag:gravity. "
        "Please migrate to newton:gravityEnabled.",
        scene_prim.GetPath().GetText());
  }

  bool clampctrl_flag;
  mjc_physics_scene.GetClampCtrlFlagAttr().Get(&clampctrl_flag);
  spec->option.disableflags |= (!clampctrl_flag ? mjDSBL_CLAMPCTRL : 0);

  bool warmstart_flag;
  mjc_physics_scene.GetWarmStartFlagAttr().Get(&warmstart_flag);
  spec->option.disableflags |= (!warmstart_flag ? mjDSBL_WARMSTART : 0);

  bool filterparent_flag;
  mjc_physics_scene.GetFilterParentFlagAttr().Get(&filterparent_flag);
  spec->option.disableflags |= (!filterparent_flag ? mjDSBL_FILTERPARENT : 0);

  bool actuation_flag;
  mjc_physics_scene.GetActuationFlagAttr().Get(&actuation_flag);
  spec->option.disableflags |= (!actuation_flag ? mjDSBL_ACTUATION : 0);

  bool refsafe_flag;
  mjc_physics_scene.GetRefSafeFlagAttr().Get(&refsafe_flag);
  spec->option.disableflags |= (!refsafe_flag ? mjDSBL_REFSAFE : 0);

  bool sensor_flag;
  mjc_physics_scene.GetSensorFlagAttr().Get(&sensor_flag);
  spec->option.disableflags |= (!sensor_flag ? mjDSBL_SENSOR : 0);

  bool midphase_flag;
  mjc_physics_scene.GetMidPhaseFlagAttr().Get(&midphase_flag);
  spec->option.disableflags |= (!midphase_flag ? mjDSBL_MIDPHASE : 0);

  bool nativeccd_flag;
  mjc_physics_scene.GetNativeCCDFlagAttr().Get(&nativeccd_flag);
  spec->option.disableflags |= (!nativeccd_flag ? mjDSBL_NATIVECCD : 0);

  bool eulerdamp_flag;
  mjc_physics_scene.GetEulerDampFlagAttr().Get(&eulerdamp_flag);
  spec->option.disableflags |= (!eulerdamp_flag ? mjDSBL_EULERDAMP : 0);

  bool autoreset_flag;
  mjc_physics_scene.GetAutoResetFlagAttr().Get(&autoreset_flag);
  spec->option.disableflags |= (!autoreset_flag ? mjDSBL_AUTORESET : 0);

  bool island_flag;
  mjc_physics_scene.GetIslandFlagAttr().Get(&island_flag);
  spec->option.disableflags |= (!island_flag ? mjDSBL_ISLAND : 0);

  bool override_flag;
  mjc_physics_scene.GetOverrideFlagAttr().Get(&override_flag);
  spec->option.enableflags |= (override_flag ? mjENBL_OVERRIDE : 0);

  bool energy_flag;
  mjc_physics_scene.GetEnergyFlagAttr().Get(&energy_flag);
  spec->option.enableflags |= (energy_flag ? mjENBL_ENERGY : 0);

  bool fwdinv_flag;
  mjc_physics_scene.GetFwdinvFlagAttr().Get(&fwdinv_flag);
  spec->option.enableflags |= (fwdinv_flag ? mjENBL_FWDINV : 0);

  bool invdiscrete_flag;
  mjc_physics_scene.GetInvDiscreteFlagAttr().Get(&invdiscrete_flag);
  spec->option.enableflags |= (invdiscrete_flag ? mjENBL_INVDISCRETE : 0);

  bool multiccd_flag;
  mjc_physics_scene.GetMultiCCDFlagAttr().Get(&multiccd_flag);
  spec->option.disableflags |= (!multiccd_flag ? mjDSBL_MULTICCD : 0);

  pxr::VtArray<int> actuatorgroupdisable;
  mjc_physics_scene.GetActuatorGroupDisableAttr().Get(&actuatorgroupdisable);
  for (int group : actuatorgroupdisable) {
    if (group >= 0 && group < mjNGROUP) {
      spec->option.disableactuator |= (1 << group);
    }
  }

  bool sleep_flag;
  mjc_physics_scene.GetSleepFlagAttr().Get(&sleep_flag);
  spec->option.enableflags |= (sleep_flag ? mjENBL_SLEEP : 0);

  // Compiler attributes
  auto auto_limits_attr = mjc_physics_scene.GetAutoLimitsAttr();
  if (auto_limits_attr.HasAuthoredValue()) {
    bool autolimits;
    auto_limits_attr.Get(&autolimits);
    spec->compiler.autolimits = autolimits;
  }

  auto use_thread_attr = mjc_physics_scene.GetUseThreadAttr();
  if (use_thread_attr.HasAuthoredValue()) {
    bool use_thread;
    use_thread_attr.Get(&use_thread);
    spec->compiler.usethread = use_thread;
  }

  auto balance_inertia_attr = mjc_physics_scene.GetBalanceInertiaAttr();
  if (balance_inertia_attr.HasAuthoredValue()) {
    bool balanceinertia;
    balance_inertia_attr.Get(&balanceinertia);
    spec->compiler.balanceinertia = balanceinertia;
  }

  auto angle_attr = mjc_physics_scene.GetAngleAttr();
  if (angle_attr.HasAuthoredValue()) {
    pxr::TfToken angle;
    angle_attr.Get(&angle);
    spec->compiler.degree = angle == MjcPhysicsTokens->degree;
  }

  auto fit_aabb_attr = mjc_physics_scene.GetFitAABBAttr();
  if (fit_aabb_attr.HasAuthoredValue()) {
    bool fitaabb;
    fit_aabb_attr.Get(&fitaabb);
    spec->compiler.fitaabb = fitaabb;
  }

  auto fuse_static_attr = mjc_physics_scene.GetFuseStaticAttr();
  if (fuse_static_attr.HasAuthoredValue()) {
    bool fusestatic;
    fuse_static_attr.Get(&fusestatic);
    spec->compiler.fusestatic = fusestatic;
  }

  auto inertia_from_geom_attr = mjc_physics_scene.GetInertiaFromGeomAttr();
  if (inertia_from_geom_attr.HasAuthoredValue()) {
    pxr::TfToken inertiafromgeom;
    inertia_from_geom_attr.Get(&inertiafromgeom);
    if (inertiafromgeom == MjcPhysicsTokens->auto_) {
      spec->compiler.inertiafromgeom = mjINERTIAFROMGEOM_AUTO;
    } else if (inertiafromgeom == MjcPhysicsTokens->false_) {
      spec->compiler.inertiafromgeom = mjINERTIAFROMGEOM_FALSE;
    } else if (inertiafromgeom == MjcPhysicsTokens->true_) {
      spec->compiler.inertiafromgeom = mjINERTIAFROMGEOM_TRUE;
    } else {
      mju_warning("Invalid inertiafromgeom token: %s",
                  inertiafromgeom.GetText());
    }
  }

  auto align_free_attr = mjc_physics_scene.GetAlignFreeAttr();
  if (align_free_attr.HasAuthoredValue()) {
    bool alignfree;
    align_free_attr.Get(&alignfree);
    spec->compiler.alignfree = alignfree;
  }

  auto inertia_group_range_min_attr =
      mjc_physics_scene.GetInertiaGroupRangeMinAttr();
  auto inertia_group_range_max_attr =
      mjc_physics_scene.GetInertiaGroupRangeMaxAttr();
  if (inertia_group_range_min_attr.HasAuthoredValue() &&
      inertia_group_range_max_attr.HasAuthoredValue()) {
    int inertiagrouprangemin;
    inertia_group_range_min_attr.Get(&inertiagrouprangemin);
    int inertiagrouprangemax;
    inertia_group_range_max_attr.Get(&inertiagrouprangemax);
    spec->compiler.inertiagrouprange[0] = inertiagrouprangemin;
    spec->compiler.inertiagrouprange[1] = inertiagrouprangemax;
  } else if (inertia_group_range_min_attr.HasAuthoredValue() ||
             inertia_group_range_max_attr.HasAuthoredValue()) {
    mju_warning(
        "Only one of inertiaGroupRangeMin and inertiaGroupRangeMax was "
        "authored, ignoring both.");
  }

  auto save_inertial_attr = mjc_physics_scene.GetSaveInertialAttr();
  if (save_inertial_attr.HasAuthoredValue()) {
    bool saveinertial;
    save_inertial_attr.Get(&saveinertial);
    spec->compiler.saveinertial = saveinertial;
  }

  mjc_physics_scene.GetBoundMassAttr().Get(&spec->compiler.boundmass);
  mjc_physics_scene.GetBoundInertiaAttr().Get(&spec->compiler.boundinertia);
  mjc_physics_scene.GetSetTotalMassAttr().Get(&spec->compiler.settotalmass);
}

void ParseUsdPhysicsMassAPIForBody(mjsBody* body,
                                   const pxr::UsdPhysicsMassAPI& mass_api) {
  bool has_mass = false;
  bool has_com = false;
  bool has_diagonal_inertia = false;

  auto mass_attr = mass_api.GetMassAttr();
  if (mass_attr.HasAuthoredValue()) {
    float mass;
    mass_attr.Get(&mass);
    body->mass = mass;
    has_mass = mass > 0;
  }

  auto com_attr = mass_api.GetCenterOfMassAttr();
  if (com_attr.HasAuthoredValue()) {
    pxr::GfVec3f com;
    com_attr.Get(&com);
    if (std::isfinite(com[0]) && std::isfinite(com[1]) &&
        std::isfinite(com[2])) {
      SetDoubleArrFromGfVec3d(body->ipos, com);
      has_com = true;
    }
  }

  auto principle_axes_attr = mass_api.GetPrincipalAxesAttr();
  if (principle_axes_attr.HasAuthoredValue()) {
    pxr::GfQuatf principle_axes;
    principle_axes_attr.Get(&principle_axes);
    SetDoubleArrFromGfQuatd(body->iquat, principle_axes);
  }

  auto diag_inertia_attr = mass_api.GetDiagonalInertiaAttr();
  if (diag_inertia_attr.HasAuthoredValue()) {
    pxr::GfVec3f diag_inertia;
    diag_inertia_attr.Get(&diag_inertia);
    SetDoubleArrFromGfVec3d(body->inertia, diag_inertia);
    has_diagonal_inertia =
        diag_inertia[0] > 0 && diag_inertia[1] > 0 && diag_inertia[2] > 0;
  }

  if (has_mass && has_diagonal_inertia) {
    body->explicitinertial = true;
    if (!has_com) {
      SetDoubleArrFromGfVec3d(body->ipos, pxr::GfVec3d(0, 0, 0));
    }
  }

  auto newton_inertia_attr =
      mass_api.GetPrim().GetAttribute(NewtonTokens->newtonInertia);
  if (newton_inertia_attr && newton_inertia_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray inertia;
    newton_inertia_attr.Get(&inertia);
    if (inertia.size() == 6) {
      std::copy(inertia.begin(), inertia.end(), body->fullinertia);
    }
  }
}

void ParseUsdPhysicsMassAPIForGeom(mjsGeom* geom,
                                   const pxr::UsdPhysicsMassAPI& mass_api) {
  auto mass_attr = mass_api.GetMassAttr();
  if (mass_attr.HasAuthoredValue()) {
    float mass;
    mass_attr.Get(&mass);
    geom->mass = mass;
  }

  auto density_attr = mass_api.GetDensityAttr();
  if (density_attr.HasAuthoredValue()) {
    float density;
    density_attr.Get(&density);
    geom->density = density;
  }
}

void ParseNewtonMassAPIForGeom(mjsGeom* geom, const pxr::UsdPrim& prim) {
  // Mesh inertia belongs to the mesh asset. Setting shell inertia on its geom
  // is invalid in MuJoCo; ParseMjcPhysicsMeshCollisionAPI handles the mass
  // model.
  if (prim.IsA<pxr::UsdGeomMesh>()) {
    return;
  }

  auto shell_inertia_attr =
      prim.GetAttribute(MjcPhysicsTokens->mjcShellinertia);
  auto newton_mass_model_attr =
      prim.GetAttribute(NewtonTokens->newtonMassModel);

  bool shell_inertia_authored =
      shell_inertia_attr && shell_inertia_attr.HasAuthoredValue();
  bool newton_mass_model_authored =
      newton_mass_model_attr && newton_mass_model_attr.HasAuthoredValue();

  if (shell_inertia_authored) {
    bool shell_inertia;
    shell_inertia_attr.Get(&shell_inertia);
    geom->typeinertia = shell_inertia ? mjtGeomInertia::mjINERTIA_SHELL
                                      : mjtGeomInertia::mjINERTIA_VOLUME;
    mju_warning(
        "Prim '%s' uses deprecated mjc:shellinertia. "
        "Please migrate to NewtonMassAPI's newton:massModel.",
        prim.GetPath().GetText());
  } else if (newton_mass_model_authored) {
    pxr::TfToken mass_model;
    newton_mass_model_attr.Get(&mass_model);
    if (mass_model == MjcPhysicsTokens->shell) {
      geom->typeinertia = mjtGeomInertia::mjINERTIA_SHELL;
    } else {
      geom->typeinertia = mjtGeomInertia::mjINERTIA_VOLUME;
    }
  }
}

void ParseMjcPhysicsCollisionAPI(
    mjsGeom* geom, const pxr::MjcPhysicsCollisionAPI& collision_api) {
  ParseNewtonMassAPIForGeom(geom, collision_api.GetPrim());

  auto group_attr = collision_api.GetGroupAttr();
  if (group_attr.HasAuthoredValue()) {
    group_attr.Get(&geom->group);
  }

  auto priority_attr = collision_api.GetPriorityAttr();
  if (priority_attr.HasAuthoredValue()) {
    priority_attr.Get(&geom->priority);
  }

  auto condim_attr = collision_api.GetConDimAttr();
  if (condim_attr.HasAuthoredValue()) {
    condim_attr.Get(&geom->condim);
  }

  auto solmix_attr = collision_api.GetSolMixAttr();
  if (solmix_attr.HasAuthoredValue()) {
    solmix_attr.Get(&geom->solmix);
  }

  auto solref_attr = collision_api.GetSolRefAttr();
  if (solref_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solref;
    solref_attr.Get(&solref);
    if (solref.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        geom->solref[i] = solref[i];
      }
    } else {
      mju_warning(
          "solref attribute for geom %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(geom->element)->c_str(), solref.size(), mjNREF);
    }
  }

  auto solimp_attr = collision_api.GetSolImpAttr();
  if (solimp_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimp;
    solimp_attr.Get(&solimp);
    if (solimp.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        geom->solimp[i] = solimp[i];
      }
    } else {
      mju_warning(
          "solimp attribute for geom %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(geom->element)->c_str(), solimp.size(), mjNIMP);
    }
  }

  auto margin_attr = collision_api.GetMarginAttr();
  auto gap_attr = collision_api.GetGapAttr();
  bool mjc_margin_authored = margin_attr.HasAuthoredValue();
  bool mjc_gap_authored = gap_attr.HasAuthoredValue();

  if (mjc_margin_authored) {
    margin_attr.Get(&geom->margin);
    mju_warning(
        "Prim '%s' uses deprecated mjc:margin. "
        "Please migrate to newton:contactMargin and newton:contactGap.",
        collision_api.GetPrim().GetPath().GetText());
  }
  if (mjc_gap_authored) {
    gap_attr.Get(&geom->gap);
    mju_warning(
        "Prim '%s' uses deprecated mjc:gap. "
        "Please migrate to newton:contactGap.",
        collision_api.GetPrim().GetPath().GetText());
  }

  // Newton collision fallback: newton:contactMargin + newton:contactGap ->
  // margin, gap
  if (!mjc_margin_authored || !mjc_gap_authored) {
    pxr::UsdPrim prim = collision_api.GetPrim();
    auto newton_margin = prim.GetAttribute(NewtonTokens->newtonContactMargin);
    auto newton_gap = prim.GetAttribute(NewtonTokens->newtonContactGap);
    float n_margin = 0, n_gap = 0;
    bool has_newton_margin = newton_margin && newton_margin.HasAuthoredValue();
    bool has_newton_gap = newton_gap && newton_gap.HasAuthoredValue();
    if (has_newton_margin) newton_margin.Get(&n_margin);
    if (has_newton_gap) newton_gap.Get(&n_gap);
    if (!mjc_gap_authored && has_newton_gap) {
      geom->gap = n_gap;
    }
    if (!mjc_margin_authored && has_newton_margin) {
      geom->margin = n_margin;
    }
  }
}

void ParseMjcPhysicsMeshCollisionAPI(
    mjsMesh* mesh, const pxr::MjcPhysicsMeshCollisionAPI& mesh_collision_api) {
  auto inertia_attr = mesh_collision_api.GetInertiaAttr();
  auto newton_mass_model_attr =
      mesh_collision_api.GetPrim().GetAttribute(NewtonTokens->newtonMassModel);

  bool inertia_authored = inertia_attr.HasAuthoredValue();
  bool newton_mass_model_authored =
      newton_mass_model_attr && newton_mass_model_attr.HasAuthoredValue();

  if (inertia_authored) {
    pxr::TfToken inertia;
    inertia_attr.Get(&inertia);
    if (inertia == MjcPhysicsTokens->shell) {
      mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_SHELL;
    } else if (inertia == MjcPhysicsTokens->exact) {
      mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_EXACT;
    } else if (inertia == MjcPhysicsTokens->convex) {
      mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_CONVEX;
    } else {
      mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_LEGACY;
    }
    mju_warning(
        "Prim '%s' uses deprecated mjc:inertia. "
        "Please migrate to NewtonMassAPI's newton:massModel.",
        mesh_collision_api.GetPrim().GetPath().GetText());
  } else {
    pxr::TfToken mass_model;
    bool is_shell = false;
    if (newton_mass_model_authored) {
      newton_mass_model_attr.Get(&mass_model);
      if (mass_model == MjcPhysicsTokens->shell) {
        mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_SHELL;
        is_shell = true;
      } else {
        mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_EXACT;
      }
    }

    // MuJoCo does not support shell inertia computation of convex
    // representation.
    if (!is_shell) {
      pxr::UsdPhysicsMeshCollisionAPI usd_mesh_api(
          mesh_collision_api.GetPrim());
      pxr::TfToken approx;
      usd_mesh_api.GetApproximationAttr().Get(&approx);
      if (approx == pxr::UsdPhysicsTokens->convexHull) {
        mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_CONVEX;
      } else {
        mesh->inertia = mjtMeshInertia::mjMESH_INERTIA_EXACT;
      }
    }
  }

  auto maxhullvert_attr = mesh_collision_api.GetMaxHullVertAttr();
  auto newton_maxhull = mesh_collision_api.GetPrim().GetAttribute(
      NewtonTokens->newtonMaxHullVertices);
  if (maxhullvert_attr.HasAuthoredValue()) {
    maxhullvert_attr.Get(&mesh->maxhullvert);
    if (!newton_maxhull || !newton_maxhull.HasAuthoredValue()) {
      mju_warning(
          "Prim '%s' uses deprecated mjc:maxhullvert. "
          "Please migrate to newton:maxHullVertices.",
          mesh_collision_api.GetPrim().GetPath().GetText());
    }
  } else if (newton_maxhull && newton_maxhull.HasAuthoredValue()) {
    int val;
    newton_maxhull.Get(&val);
    mesh->maxhullvert = val;
  }
}

void ParseMjcPhysicsTendon(mjSpec* spec, const pxr::MjcPhysicsTendon& tendon) {
  pxr::UsdPrim prim = tendon.GetPrim();
  pxr::UsdStageRefPtr stage = prim.GetStage();

  pxr::TfToken type;
  tendon.GetTypeAttr().Get(&type);

  pxr::SdfPathVector wrap_targets;
  tendon.GetMjcPathRel().GetTargets(&wrap_targets);

  pxr::VtIntArray wrap_path_indices;
  tendon.GetMjcPathIndicesAttr().Get(&wrap_path_indices);

  auto n_targets = wrap_path_indices.empty() ? wrap_targets.size()
                                             : wrap_path_indices.size();

  pxr::SdfPathVector side_site_paths;
  tendon.GetMjcSideSitesRel().GetTargets(&side_site_paths);

  pxr::VtIntArray side_site_indices;
  tendon.GetMjcSideSitesIndicesAttr().Get(&side_site_indices);

  pxr::VtIntArray segments;
  tendon.GetMjcPathSegmentsAttr().Get(&segments);

  pxr::VtDoubleArray divisors;
  tendon.GetMjcPathDivisorsAttr().Get(&divisors);

  pxr::VtDoubleArray coefs;
  tendon.GetMjcPathCoefAttr().Get(&coefs);

  if (type == MjcPhysicsTokens->spatial) {
    // Check that for N targets we have 0 or N elements in segments.
    if (!segments.empty() && segments.size() != n_targets) {
      mju_warning(
          "Spatial tendon %s has %lu segments but %lu wrap targets, skipping.",
          prim.GetPath().GetAsString().c_str(), segments.size(), n_targets);
      return;
    }
    // Check that if we have >1 segments that the user has specified how much
    // each segment contributes to the total segment length.
    if (!segments.empty() && divisors.empty()) {
      mju_warning(
          "Spatial tendon %s has >1 segments (%d) but does not specify "
          "divisors, skipping.",
          prim.GetPath().GetAsString().c_str(),
          *std::max_element(segments.begin(), segments.end()) + 1);
      return;
    }
    // Check that if we side site indices that we have N of them.
    if (!side_site_indices.empty() && side_site_indices.size() != n_targets) {
      mju_warning(
          "Spatial tendon %s has %lu sideSite indices but %lu wrap targets, "
          "skipping.",
          prim.GetPath().GetAsString().c_str(), side_site_indices.size(),
          n_targets);
      return;
    }

    if (!side_site_indices.empty() && side_site_paths.empty()) {
      mju_warning(
          "Spatial tendon %s has %lu sideSite indices but no side sites, "
          "skipping.",
          prim.GetPath().GetAsString().c_str(), side_site_indices.size());
      return;
    }
  } else {  // Fixed tendon.
    // Check that for N targets we have 0 or N elements in coef:
    if (!coefs.empty() && coefs.size() != n_targets) {
      mju_warning(
          "Fixed tendon %s has %lu coefs but %lu wrap targets, skipping.",
          prim.GetPath().GetAsString().c_str(), coefs.size(), n_targets);
      return;
    }
  }

  mjsTendon* mj_tendon = mjs_addTendon(spec, nullptr);
  mjs_setName(mj_tendon->element, prim.GetPath().GetAsString().c_str());
  SetUsdPrimPathUserValue(mj_tendon->element, prim.GetPath());

  int last_segment = 0;
  for (int i = 0; i < n_targets; ++i) {
    auto index = wrap_path_indices.empty() ? i : wrap_path_indices[i];
    auto wrap_target = wrap_targets[index];
    auto wrap_prim = stage->GetPrimAtPath(wrap_target);
    // Important to check site before Imageable here because some Imageable
    // prims are sites.

    if (!segments.empty()) {
      int segment = segments[i];
      if (segment >= divisors.size()) {
        mju_warning(
            "Tendon %s has at least %d segments but only %lu divisors, "
            "skipping.",
            prim.GetPath().GetAsString().c_str(), segment + 1, divisors.size());
        mjs_delete(spec, mj_tendon->element);
        return;
      }

      if (segment > last_segment || (i == 0 && divisors[segment] != 1.0)) {
        mjsWrap* pulley_wrap = mjs_wrapPulley(mj_tendon, divisors[segment]);
        mjs_setString(pulley_wrap->info, ("Pulley between segments: " +
                                          std::to_string(last_segment) +
                                          " and " + std::to_string(segment))
                                             .c_str());
      }
      last_segment = segment;
    }

    mjsWrap* wrap = nullptr;
    if (wrap_prim.HasAPI<pxr::MjcPhysicsSiteAPI>()) {
      wrap = mjs_wrapSite(mj_tendon, wrap_target.GetAsString().c_str());
    } else if (wrap_prim.IsA<pxr::UsdPhysicsJoint>()) {
      double coef = 1.0;
      if (!coefs.empty()) {
        coef = coefs[i];
      }
      wrap = mjs_wrapJoint(mj_tendon, wrap_target.GetAsString().c_str(), coef);
    } else if (wrap_prim.IsA<pxr::UsdGeomImageable>()) {
      std::string side_site_name = "";
      if (!side_site_indices.empty()) {
        int side_site_index = side_site_indices[i];
        if (side_site_index > 0 && side_site_index >= side_site_paths.size()) {
          mju_warning(
              "Tendon %s has side site index %d but only %lu side sites, "
              "skipping.",
              prim.GetPath().GetAsString().c_str(), side_site_index,
              side_site_paths.size());
          mjs_delete(spec, mj_tendon->element);
          return;
        }
        if (side_site_index >= 0) {
          side_site_name = side_site_paths[side_site_index].GetAsString();
        }
      }
      wrap = mjs_wrapGeom(mj_tendon, wrap_target.GetAsString().c_str(),
                          side_site_name.c_str());
    } else {
      mju_warning("Tendon %s has an invalid wrap target type, skipping.",
                  prim.GetPath().GetAsString().c_str());
      mjs_delete(spec, mj_tendon->element);
      return;
    }
    mjs_setString(wrap->info, ("Prim: " + wrap_target.GetAsString()).c_str());
  }

  auto group_attr = tendon.GetGroupAttr();
  if (group_attr.HasAuthoredValue()) {
    group_attr.Get(&mj_tendon->group);
  }

  auto limited_attr = tendon.GetLimitedAttr();
  if (limited_attr.HasAuthoredValue()) {
    pxr::TfToken limited;
    limited_attr.Get(&limited);
    if (limited == MjcPhysicsTokens->true_) {
      mj_tendon->limited = mjLIMITED_TRUE;
    } else if (limited == MjcPhysicsTokens->false_) {
      mj_tendon->limited = mjLIMITED_FALSE;
    } else {
      mj_tendon->limited = mjLIMITED_AUTO;
    }
  }

  auto actuatorfrclimited_attr = tendon.GetActuatorFrcLimitedAttr();
  if (actuatorfrclimited_attr.HasAuthoredValue()) {
    pxr::TfToken actuatorfrclimited;
    actuatorfrclimited_attr.Get(&actuatorfrclimited);
    if (actuatorfrclimited == MjcPhysicsTokens->true_) {
      mj_tendon->actfrclimited = mjLIMITED_TRUE;
    } else if (actuatorfrclimited == MjcPhysicsTokens->false_) {
      mj_tendon->actfrclimited = mjLIMITED_FALSE;
    } else {
      mj_tendon->actfrclimited = mjLIMITED_AUTO;
    }
  }

  auto range_min_attr = tendon.GetRangeMinAttr();
  if (range_min_attr.HasAuthoredValue()) {
    range_min_attr.Get(&mj_tendon->range[0]);
  }

  auto range_max_attr = tendon.GetRangeMaxAttr();
  if (range_max_attr.HasAuthoredValue()) {
    range_max_attr.Get(&mj_tendon->range[1]);
  }

  auto actuatorfrcrange_min_attr = tendon.GetActuatorFrcRangeMinAttr();
  if (actuatorfrcrange_min_attr.HasAuthoredValue()) {
    actuatorfrcrange_min_attr.Get(&mj_tendon->actfrcrange[0]);
  }

  auto actuatorfrcrange_max_attr = tendon.GetActuatorFrcRangeMaxAttr();
  if (actuatorfrcrange_max_attr.HasAuthoredValue()) {
    actuatorfrcrange_max_attr.Get(&mj_tendon->actfrcrange[1]);
  }

  auto solreflimit_attr = tendon.GetSolRefLimitAttr();
  if (solreflimit_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solreflimit;
    solreflimit_attr.Get(&solreflimit);
    if (solreflimit.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        mj_tendon->solref_limit[i] = solreflimit[i];
      }
    } else {
      mju_warning(
          "solreflimit attribute for tendon %s has incorrect size %zu, "
          "expected %d.",
          prim.GetPath().GetAsString().c_str(), solreflimit.size(), mjNREF);
    }
  }

  auto solimplimit_attr = tendon.GetSolImpLimitAttr();
  if (solimplimit_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimplimit;
    solimplimit_attr.Get(&solimplimit);
    if (solimplimit.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        mj_tendon->solimp_limit[i] = solimplimit[i];
      }
    } else {
      mju_warning(
          "solimplimit attribute for tendon %s has incorrect size %zu, "
          "expected %d.",
          prim.GetPath().GetAsString().c_str(), solimplimit.size(), mjNIMP);
    }
  }

  auto solreffriction_attr = tendon.GetSolRefFrictionAttr();
  if (solreffriction_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solreffriction;
    solreffriction_attr.Get(&solreffriction);
    if (solreffriction.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        mj_tendon->solref_friction[i] = solreffriction[i];
      }
    } else {
      mju_warning(
          "solreffriction attribute for tendon %s has incorrect size %zu, "
          "expected %d.",
          prim.GetPath().GetAsString().c_str(), solreffriction.size(), mjNREF);
    }
  }

  auto solimpfriction_attr = tendon.GetSolImpFrictionAttr();
  if (solimpfriction_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimpfriction;
    solimpfriction_attr.Get(&solimpfriction);
    if (solimpfriction.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        mj_tendon->solimp_friction[i] = solimpfriction[i];
      }
    } else {
      mju_warning(
          "solimpfriction attribute for tendon %s has incorrect size %zu, "
          "expected %d.",
          prim.GetPath().GetAsString().c_str(), solimpfriction.size(), mjNIMP);
    }
  }

  auto margin_attr = tendon.GetMarginAttr();
  if (margin_attr.HasAuthoredValue()) {
    margin_attr.Get(&mj_tendon->margin);
  }

  auto frictionloss_attr = tendon.GetFrictionLossAttr();
  if (frictionloss_attr.HasAuthoredValue()) {
    frictionloss_attr.Get(&mj_tendon->frictionloss);
  }

  auto width_attr = tendon.GetWidthAttr();
  if (width_attr.HasAuthoredValue()) {
    width_attr.Get(&mj_tendon->width);
  }

  auto rgba_attr = tendon.GetRgbaAttr();
  if (rgba_attr.HasAuthoredValue()) {
    pxr::GfVec4f rgba;
    rgba_attr.Get(&rgba);
    mj_tendon->rgba[0] = rgba[0];
    mj_tendon->rgba[1] = rgba[1];
    mj_tendon->rgba[2] = rgba[2];
    mj_tendon->rgba[3] = rgba[3];
  }

  auto springlength_attr = tendon.GetSpringLengthAttr();
  if (springlength_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray springlength;
    springlength_attr.Get(&springlength);
    if (springlength.size() == 1) {
      mj_tendon->springlength[0] = springlength[0];
      mj_tendon->springlength[1] = springlength[0];
    } else if (springlength.size() == 2) {
      mj_tendon->springlength[0] = springlength[0];
      mj_tendon->springlength[1] = springlength[1];
    } else {
      mju_warning(
          "springlength attribute for tendon %s has incorrect size %zu, "
          "expected 1 or 2.",
          prim.GetPath().GetAsString().c_str(), springlength.size());
    }
  }

  auto stiffness_attr = tendon.GetStiffnessAttr();
  if (stiffness_attr.HasAuthoredValue()) {
    stiffness_attr.Get(&mj_tendon->stiffness[0]);
  }

  auto damping_attr = tendon.GetDampingAttr();
  if (damping_attr.HasAuthoredValue()) {
    damping_attr.Get(&mj_tendon->damping[0]);
  }

  auto armature_attr = tendon.GetArmatureAttr();
  if (armature_attr.HasAuthoredValue()) {
    armature_attr.Get(&mj_tendon->armature);
  }
}

void ParseMjcPhysicsActuator(mjSpec* spec,
                             const pxr::MjcPhysicsActuator& tran) {
  pxr::UsdPrim prim = tran.GetPrim();
  mjsActuator* mj_act = mjs_addActuator(spec, nullptr);
  mjs_setName(mj_act->element, prim.GetPath().GetAsString().c_str());

  SetUsdPrimPathUserValue(mj_act->element, prim.GetPath());

  auto group_attr = tran.GetGroupAttr();
  if (group_attr.HasAuthoredValue()) {
    group_attr.Get(&mj_act->group);
  }

  pxr::SdfPathVector targets;
  tran.GetMjcTargetRel().GetTargets(&targets);
  if (targets.empty()) {
    mju_warning("Actuator %s has no target, skipping.",
                prim.GetPath().GetAsString().c_str());
    return;
  }
  if (targets.size() > 1) {
    mju_warning("Actuator has more than one target, using the first.");
  }
  mjs_setString(mj_act->target, targets[0].GetAsString().c_str());

  auto target_prim = prim.GetStage()->GetPrimAtPath(targets[0]);
  bool slider_crank = tran.GetMjcSliderSiteRel().HasAuthoredTargets();
  if (target_prim.IsA<pxr::UsdPhysicsJoint>()) {
    mj_act->trntype = mjTRN_JOINT;
  } else if (target_prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>()) {
    mj_act->trntype = mjTRN_BODY;
  } else if (target_prim.HasAPI<pxr::MjcPhysicsSiteAPI>()) {
    mj_act->trntype = slider_crank ? mjTRN_SLIDERCRANK : mjTRN_SITE;
  } else if (target_prim.IsA<pxr::MjcPhysicsTendon>()) {
    mj_act->trntype = mjTRN_TENDON;
  } else {
    mju_warning("Actuator %s has an invalid target type, skipping.",
                prim.GetPath().GetAsString().c_str());
    return;
  }

  if (slider_crank) {
    pxr::SdfPathVector slider_sites;
    tran.GetMjcSliderSiteRel().GetTargets(&slider_sites);
    if (slider_sites.size() > 1) {
      mju_warning(
          "Slider crank slider site relationship has more than one target, "
          "using the first.");
    }
    mjs_setString(mj_act->slidersite, slider_sites[0].GetAsString().c_str());
  }

  auto setLimitedField = [](mjsActuator* mj_act,
                            const pxr::UsdAttribute& usd_attribute,
                            mjtLimited* mj_limited_field) {
    if (usd_attribute.HasAuthoredValue()) {
      pxr::TfToken limited;
      usd_attribute.Get(&limited);
      if (limited == MjcPhysicsTokens->true_) {
        *mj_limited_field = mjLIMITED_TRUE;
      } else if (limited == MjcPhysicsTokens->false_) {
        *mj_limited_field = mjLIMITED_FALSE;
      } else if (limited == MjcPhysicsTokens->auto_) {
        *mj_limited_field = mjLIMITED_AUTO;
      }
    }
  };

  setLimitedField(mj_act, tran.GetMjcCtrlLimitedAttr(), &mj_act->ctrllimited);
  setLimitedField(mj_act, tran.GetMjcForceLimitedAttr(), &mj_act->forcelimited);
  setLimitedField(mj_act, tran.GetMjcActLimitedAttr(), &mj_act->actlimited);

  auto setRangeField =
      [](mjsActuator* mj_act, const pxr::UsdAttribute& usd_min_attribute,
         const pxr::UsdAttribute& usd_max_attribute, double* range) {
        if (usd_min_attribute.HasAuthoredValue()) {
          double range_min;
          usd_min_attribute.Get(&range_min);
          range[0] = range_min;
        }
        if (usd_max_attribute.HasAuthoredValue()) {
          double range_max;
          usd_max_attribute.Get(&range_max);
          range[1] = range_max;
        }
      };

  setRangeField(mj_act, tran.GetMjcCtrlRangeMinAttr(),
                tran.GetMjcCtrlRangeMaxAttr(), mj_act->ctrlrange);
  setRangeField(mj_act, tran.GetMjcForceRangeMinAttr(),
                tran.GetMjcForceRangeMaxAttr(), mj_act->forcerange);
  setRangeField(mj_act, tran.GetMjcActRangeMinAttr(),
                tran.GetMjcActRangeMaxAttr(), mj_act->actrange);
  setRangeField(mj_act, tran.GetMjcLengthRangeMinAttr(),
                tran.GetMjcLengthRangeMaxAttr(), mj_act->lengthrange);

  auto gear_attr = tran.GetMjcGearAttr();
  if (gear_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray gear;
    gear_attr.Get(&gear);
    for (int i = 0; i < 6; ++i) {
      mj_act->gear[i] = gear[i];
    }
  }

  auto crank_length_attr = tran.GetMjcCrankLengthAttr();
  if (crank_length_attr.HasAuthoredValue()) {
    double crank_length;
    crank_length_attr.Get(&crank_length);
    mj_act->cranklength = crank_length;
  }

  auto dyn_type_attr = tran.GetMjcDynTypeAttr();
  if (dyn_type_attr.HasAuthoredValue()) {
    pxr::TfToken dyn_type;
    dyn_type_attr.Get(&dyn_type);
    if (dyn_type == MjcPhysicsTokens->none) {
      mj_act->dyntype = mjtDyn::mjDYN_NONE;
    } else if (dyn_type == MjcPhysicsTokens->integrator) {
      mj_act->dyntype = mjtDyn::mjDYN_INTEGRATOR;
    } else if (dyn_type == MjcPhysicsTokens->filter) {
      mj_act->dyntype = mjtDyn::mjDYN_FILTER;
    } else if (dyn_type == MjcPhysicsTokens->filterexact) {
      mj_act->dyntype = mjtDyn::mjDYN_FILTEREXACT;
    } else if (dyn_type == MjcPhysicsTokens->muscle) {
      mj_act->dyntype = mjtDyn::mjDYN_MUSCLE;
    } else if (dyn_type == MjcPhysicsTokens->user) {
      mj_act->dyntype = mjtDyn::mjDYN_USER;
    }
  }

  auto gain_type_attr = tran.GetMjcGainTypeAttr();
  if (gain_type_attr.HasAuthoredValue()) {
    pxr::TfToken gain_type;
    gain_type_attr.Get(&gain_type);
    if (gain_type == MjcPhysicsTokens->fixed) {
      mj_act->gaintype = mjtGain::mjGAIN_FIXED;
    } else if (gain_type == MjcPhysicsTokens->affine) {
      mj_act->gaintype = mjtGain::mjGAIN_AFFINE;
    } else if (gain_type == MjcPhysicsTokens->muscle) {
      mj_act->gaintype = mjtGain::mjGAIN_MUSCLE;
    } else if (gain_type == MjcPhysicsTokens->user) {
      mj_act->gaintype = mjtGain::mjGAIN_USER;
    }
  }

  auto biastype_attr = tran.GetMjcBiasTypeAttr();
  if (biastype_attr.HasAuthoredValue()) {
    pxr::TfToken biastype;
    biastype_attr.Get(&biastype);
    if (biastype == MjcPhysicsTokens->none) {
      mj_act->biastype = mjtBias::mjBIAS_NONE;
    } else if (biastype == MjcPhysicsTokens->affine) {
      mj_act->biastype = mjtBias::mjBIAS_AFFINE;
    } else if (biastype == MjcPhysicsTokens->muscle) {
      mj_act->biastype = mjtBias::mjBIAS_MUSCLE;
    } else if (biastype == MjcPhysicsTokens->user) {
      mj_act->biastype = mjtBias::mjBIAS_USER;
    }
  }

  auto setPrmField = [](mjsActuator* mj_act,
                        const pxr::UsdAttribute& usd_attribute, double* prm) {
    if (usd_attribute.HasAuthoredValue()) {
      pxr::VtDoubleArray usd_prm;
      usd_attribute.Get(&usd_prm);
      int n_elems = usd_prm.size() < 10 ? usd_prm.size() : 10;
      for (int i = 0; i < n_elems; ++i) {
        prm[i] = usd_prm[i];
      }
    }
  };

  setPrmField(mj_act, tran.GetMjcDynPrmAttr(), mj_act->dynprm);
  setPrmField(mj_act, tran.GetMjcBiasPrmAttr(), mj_act->biasprm);
  setPrmField(mj_act, tran.GetMjcGainPrmAttr(), mj_act->gainprm);

  auto act_dim_attr = tran.GetMjcActDimAttr();
  if (act_dim_attr.HasAuthoredValue()) {
    int act_dim;
    act_dim_attr.Get(&act_dim);
    mj_act->actdim = act_dim;
  }

  auto act_early_attr = tran.GetMjcActEarlyAttr();
  if (act_early_attr.HasAuthoredValue()) {
    bool act_early;
    act_early_attr.Get(&act_early);
    mj_act->actearly = (int)act_early;
  }

  auto inherit_range_attr = tran.GetMjcInheritRangeAttr();
  if (inherit_range_attr.HasAuthoredValue()) {
    inherit_range_attr.Get(&mj_act->inheritrange);
  }

  auto ref_site_rel = tran.GetMjcRefSiteRel();
  if (ref_site_rel.HasAuthoredTargets()) {
    pxr::SdfPathVector targets;
    ref_site_rel.GetTargets(&targets);
    pxr::SdfPath first_path = targets[0];
    mjs_setString(mj_act->refsite, first_path.GetString().c_str());
  }
}

void ParseMjcPhysicsJointAPI(mjsJoint* mj_joint,
                             const pxr::MjcPhysicsJointAPI& joint_api) {
  auto group_attr = joint_api.GetGroupAttr();
  if (group_attr.HasAuthoredValue()) {
    group_attr.Get(&mj_joint->group);
  }

  auto springdamper_attr = joint_api.GetMjcSpringdamperAttr();
  if (springdamper_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray springdamper;
    springdamper_attr.Get(&springdamper);
    if (springdamper.size() == 2) {
      mj_joint->springdamper[0] = springdamper[0];
      mj_joint->springdamper[1] = springdamper[1];
    } else {
      mju_warning(
          "springdamper attribute for joint %s has incorrect size %zu, "
          "expected 2.",
          mjs_getName(mj_joint->element)->c_str(), springdamper.size());
    }
  }

  auto solreflimit_attr = joint_api.GetMjcSolreflimitAttr();
  if (solreflimit_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solreflimit;
    solreflimit_attr.Get(&solreflimit);
    if (solreflimit.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        mj_joint->solref_limit[i] = solreflimit[i];
      }
    } else {
      mju_warning(
          "solreflimit attribute for joint %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(mj_joint->element)->c_str(), solreflimit.size(), mjNREF);
    }
  }

  auto solimplimit_attr = joint_api.GetMjcSolimplimitAttr();
  if (solimplimit_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimplimit;
    solimplimit_attr.Get(&solimplimit);
    if (solimplimit.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        mj_joint->solimp_limit[i] = solimplimit[i];
      }
    } else {
      mju_warning(
          "solimplimit attribute for joint %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(mj_joint->element)->c_str(), solimplimit.size(), mjNIMP);
    }
  }

  auto solreffriction_attr = joint_api.GetMjcSolreffrictionAttr();
  if (solreffriction_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solreffriction;
    solreffriction_attr.Get(&solreffriction);
    if (solreffriction.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        mj_joint->solref_friction[i] = solreffriction[i];
      }
    } else {
      mju_warning(
          "solreffriction attribute for joint %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(mj_joint->element)->c_str(), solreffriction.size(),
          mjNREF);
    }
  }

  auto solimpfriction_attr = joint_api.GetMjcSolimpfrictionAttr();
  if (solimpfriction_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimpfriction;
    solimpfriction_attr.Get(&solimpfriction);
    if (solimpfriction.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        mj_joint->solimp_friction[i] = solimpfriction[i];
      }
    } else {
      mju_warning(
          "solimpfriction attribute for joint %s has incorrect size %zu, "
          "expected %d.",
          mjs_getName(mj_joint->element)->c_str(), solimpfriction.size(),
          mjNIMP);
    }
  }

  auto stiffness_attr = joint_api.GetMjcStiffnessAttr();
  if (stiffness_attr.HasAuthoredValue()) {
    double stiffness;
    stiffness_attr.Get(&stiffness);
    mj_joint->stiffness[0] = stiffness;
  }

  auto actuatorfrcrange_min_attr = joint_api.GetMjcActuatorfrcrangeMinAttr();
  if (actuatorfrcrange_min_attr.HasAuthoredValue()) {
    double min_val;
    actuatorfrcrange_min_attr.Get(&min_val);
    mj_joint->actfrcrange[0] = min_val;
  }
  auto actuatorfrcrange_max_attr = joint_api.GetMjcActuatorfrcrangeMaxAttr();
  if (actuatorfrcrange_max_attr.HasAuthoredValue()) {
    double max_val;
    actuatorfrcrange_max_attr.Get(&max_val);
    mj_joint->actfrcrange[1] = max_val;
  }

  auto actuatorfrclimited_attr = joint_api.GetMjcActuatorfrclimitedAttr();
  if (actuatorfrclimited_attr.HasAuthoredValue()) {
    pxr::TfToken limited;
    actuatorfrclimited_attr.Get(&limited);
    if (limited == MjcPhysicsTokens->true_) {
      mj_joint->actfrclimited = mjLIMITED_TRUE;
    } else if (limited == MjcPhysicsTokens->false_) {
      mj_joint->actfrclimited = mjLIMITED_FALSE;
    } else if (limited == MjcPhysicsTokens->auto_) {
      mj_joint->actfrclimited = mjLIMITED_AUTO;
    }
  }

  auto actuatorgravcomp_attr = joint_api.GetMjcActuatorgravcompAttr();
  if (actuatorgravcomp_attr.HasAuthoredValue()) {
    bool gravcomp;
    actuatorgravcomp_attr.Get(&gravcomp);
    mj_joint->actgravcomp = gravcomp;
  }

  auto margin_attr = joint_api.GetMjcMarginAttr();
  if (margin_attr.HasAuthoredValue()) {
    double margin;
    margin_attr.Get(&margin);
    mj_joint->margin = margin;
  }

  auto ref_attr = joint_api.GetMjcRefAttr();
  if (ref_attr.HasAuthoredValue()) {
    double ref;
    ref_attr.Get(&ref);
    mj_joint->ref = ref;
  }

  auto springref_attr = joint_api.GetMjcSpringrefAttr();
  if (springref_attr.HasAuthoredValue()) {
    double springref;
    springref_attr.Get(&springref);
    mj_joint->springref = springref;
  }

  auto mjc_armature_attr =
      joint_api.GetPrim().GetAttribute(MjcPhysicsTokens->mjcArmature);
  auto mjc_damping_attr =
      joint_api.GetPrim().GetAttribute(MjcPhysicsTokens->mjcDamping);
  auto mjc_frictionloss_attr =
      joint_api.GetPrim().GetAttribute(MjcPhysicsTokens->mjcFrictionloss);

  auto newton_armature_attr =
      joint_api.GetPrim().GetAttribute(NewtonTokens->newtonArmature);
  auto newton_damping_attr =
      joint_api.GetPrim().GetAttribute(NewtonTokens->newtonDamping);
  auto newton_friction_attr =
      joint_api.GetPrim().GetAttribute(NewtonTokens->newtonFriction);

  bool mjc_armature_authored =
      mjc_armature_attr && mjc_armature_attr.HasAuthoredValue();
  bool mjc_damping_authored =
      mjc_damping_attr && mjc_damping_attr.HasAuthoredValue();
  bool mjc_frictionloss_authored =
      mjc_frictionloss_attr && mjc_frictionloss_attr.HasAuthoredValue();

  bool newton_armature_authored =
      newton_armature_attr && newton_armature_attr.HasAuthoredValue();
  bool newton_damping_authored =
      newton_damping_attr && newton_damping_attr.HasAuthoredValue();
  bool newton_friction_authored =
      newton_friction_attr && newton_friction_attr.HasAuthoredValue();

  if (mjc_armature_authored) {
    mjc_armature_attr.Get(&mj_joint->armature);
    mju_warning(
        "Prim '%s' uses deprecated mjc:armature. "
        "Please migrate to newton:armature.",
        joint_api.GetPrim().GetPath().GetText());
  } else if (newton_armature_authored) {
    float armature;
    newton_armature_attr.Get(&armature);
    mj_joint->armature = armature;
  }

  if (mjc_damping_authored) {
    double damping;
    mjc_damping_attr.Get(&damping);
    mj_joint->damping[0] = damping;
    mju_warning(
        "Prim '%s' uses deprecated mjc:damping. "
        "Please migrate to newton:damping.",
        joint_api.GetPrim().GetPath().GetText());
  } else if (newton_damping_authored) {
    float damping;
    newton_damping_attr.Get(&damping);
    // Newton angular damping is per degree; MuJoCo damping is per radian,
    // regardless of the compiler's angle setting.
    mj_joint->damping[0] =
        mj_joint->type == mjJNT_HINGE ? damping * 180.0 / mjPI : damping;
  }

  if (mjc_frictionloss_authored) {
    double frictionloss;
    mjc_frictionloss_attr.Get(&frictionloss);
    mj_joint->frictionloss = frictionloss;
    mju_warning(
        "Prim '%s' uses deprecated mjc:frictionloss. "
        "Please migrate to newton:friction.",
        joint_api.GetPrim().GetPath().GetText());
  } else if (newton_friction_authored) {
    float friction;
    newton_friction_attr.Get(&friction);
    mj_joint->frictionloss = friction;
  }
}

void ParseUsdPhysicsMaterialAPI(
    mjsGeom* geom, const pxr::UsdPhysicsMaterialAPI& material_api) {
  auto dynamic_friction_attr = material_api.GetDynamicFrictionAttr();
  if (dynamic_friction_attr.HasAuthoredValue()) {
    float dynamic_friction;
    dynamic_friction_attr.Get(&dynamic_friction);
    geom->friction[0] = dynamic_friction;
  }

  auto restitution_attr = material_api.GetRestitutionAttr();
  if (restitution_attr.HasAuthoredValue()) {
    mju_warning(
        "Material %s is trying to set the resitution coefficient, to control "
        "restitution in MuJoCo use the direct method of setting solref to "
        "(-stiffness, -damping). See "
        "https://mujoco.readthedocs.io/en/latest/modeling.html#restitution for "
        "examples.",
        material_api.GetPath().GetString().c_str());
  }

  auto density_attr = material_api.GetDensityAttr();
  if (density_attr.HasAuthoredValue()) {
    float density;
    density_attr.Get(&density);
    geom->density = density;
  }
}

void ParseMjcPhysicsMaterialAPI(
    mjsGeom* geom, const pxr::UsdPrim& material_prim,
    const pxr::MjcPhysicsMaterialAPI& material_api) {
  // Torsional friction: prefer newton:torsionalFriction, fall back to
  // mjc:torsionalfriction with deprecation warning. If both are authored,
  // mjc takes precedence for backwards compatibility.
  auto mjc_torsional = material_api.GetTorsionalFrictionAttr();
  auto newton_torsional =
      material_prim.GetAttribute(NewtonTokens->newtonTorsionalFriction);
  if (mjc_torsional.HasAuthoredValue()) {
    mjc_torsional.Get(&geom->friction[1]);
    if (!newton_torsional || !newton_torsional.HasAuthoredValue()) {
      mju_warning(
          "Prim '%s' uses deprecated mjc:torsionalfriction. "
          "Please migrate to newton:torsionalFriction.",
          material_prim.GetPath().GetText());
    }
  } else if (newton_torsional && newton_torsional.HasAuthoredValue()) {
    float val;
    newton_torsional.Get(&val);
    geom->friction[1] = val;
  }

  // Rolling friction: same deprecation/fallback pattern.
  auto mjc_rolling = material_api.GetRollingFrictionAttr();
  auto newton_rolling =
      material_prim.GetAttribute(NewtonTokens->newtonRollingFriction);
  if (mjc_rolling.HasAuthoredValue()) {
    mjc_rolling.Get(&geom->friction[2]);
    if (!newton_rolling || !newton_rolling.HasAuthoredValue()) {
      mju_warning(
          "Prim '%s' uses deprecated mjc:rollingfriction. "
          "Please migrate to newton:rollingFriction.",
          material_prim.GetPath().GetText());
    }
  } else if (newton_rolling && newton_rolling.HasAuthoredValue()) {
    float val;
    newton_rolling.Get(&val);
    geom->friction[2] = val;
  }

  // Contact adhesion: newton:contactAdhesion -> geom->adhesion
  auto newton_adhesion =
      material_prim.GetAttribute(NewtonTokens->newtonContactAdhesion);
  if (newton_adhesion && newton_adhesion.HasAuthoredValue()) {
    float val;
    newton_adhesion.Get(&val);
    if (val >= 0.0f) {
      geom->adhesion = val;
    }
  }
}

void ParseDisplayColorAndOpacity(const pxr::UsdPrim& prim, mjsGeom* geom) {
  // Convert displayColor and displayOpacity to rgba.
  // We want to support primvar inheritance, hence FindPrimvarWithInheritance.
  pxr::UsdGeomPrimvarsAPI primvarsAPI(prim);
  pxr::UsdGeomPrimvar displayColorPrimvar =
      primvarsAPI.FindPrimvarWithInheritance(
          pxr::UsdGeomTokens->primvarsDisplayColor);
  pxr::UsdGeomPrimvar displayOpacityPrimvar =
      primvarsAPI.FindPrimvarWithInheritance(
          pxr::UsdGeomTokens->primvarsDisplayOpacity);
  if (displayColorPrimvar.HasAuthoredValue()) {
    pxr::VtArray<pxr::GfVec3f> display_color;
    displayColorPrimvar.Get(&display_color);
    if (!display_color.empty()) {
      geom->rgba[0] = display_color[0][0];
      geom->rgba[1] = display_color[0][1];
      geom->rgba[2] = display_color[0][2];
    }
  }
  if (displayOpacityPrimvar.HasAuthoredValue()) {
    pxr::VtArray<float> display_opacity;
    displayOpacityPrimvar.Get(&display_opacity);
    if (!display_opacity.empty()) {
      geom->rgba[3] = display_opacity[0];
    }
  }
}

mjsGeom* ParseUsdGeomGprim(mjSpec* spec, const pxr::UsdPrim& gprim,
                           const pxr::UsdPrim& body_prim, mjsBody* parent,
                           UsdCaches& caches) {
  mjsGeom* geom = mjs_addGeom(parent, nullptr);
  mjs_setName(geom->element, gprim.GetPath().GetAsString().c_str());
  geom->contype = 0;
  geom->conaffinity = 0;
  if (!body_prim.IsPseudoRoot() ||
      gprim.HasAPI<pxr::MjcPhysicsImageableAPI>()) {
    geom->density = 0;
  }

  SetUsdPrimPathUserValue(geom->element, gprim.GetPath());

  ParseDisplayColorAndOpacity(gprim, geom);
  SetLocalPoseFromPrim(gprim, body_prim, geom, caches.xform_cache);
  if (!MaybeParseGeomPrimitive(gprim, geom, caches.xform_cache)) {
    mjsMesh* mesh = ParseUsdMesh(spec, gprim, geom, caches.xform_cache);
    if (mesh != nullptr &&
        (gprim.HasAPI<pxr::MjcPhysicsMeshCollisionAPI>() ||
         gprim.HasAPI(NewtonTokens->NewtonMeshCollisionAPI) ||
         gprim.HasAPI(NewtonTokens->NewtonMassAPI))) {
      ParseMjcPhysicsMeshCollisionAPI(mesh,
                                      pxr::MjcPhysicsMeshCollisionAPI(gprim));
    }
  }

  ParseNewtonMassAPIForGeom(geom, gprim);

  pxr::UsdShadeMaterial bound_material =
      pxr::UsdShadeMaterialBindingAPI(gprim).ComputeBoundMaterial(
          &caches.bindings_cache, &caches.collection_query_cache);
  if (bound_material) {
    pxr::SdfPath material_path = bound_material.GetPrim().GetPath();
    mjsMaterial* material = nullptr;
    if (auto iter = caches.parsed_materials.find(material_path);
        iter != caches.parsed_materials.end()) {
      material = iter->second;
    } else {
      material = ParseMaterial(spec, bound_material);
      // ParseMaterial may return a nullptr if the material is not supported.
      if (material) {
        caches.parsed_materials[material_path] = material;
      }
    }
    if (material) {
      mjs_setString(geom->material, mjs_getName(material->element)->c_str());
    }
  }

  if (gprim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
    ParseUsdPhysicsMassAPIForGeom(geom, pxr::UsdPhysicsMassAPI(gprim));
  }

  if (gprim.HasAPI<pxr::MjcPhysicsCollisionAPI>()) {
    ParseMjcPhysicsCollisionAPI(geom, pxr::MjcPhysicsCollisionAPI(gprim));
  }
  if (gprim.HasAPI<pxr::MjcPhysicsImageableAPI>()) {
    auto imageable_api = pxr::MjcPhysicsImageableAPI(gprim);
    auto group_attr = imageable_api.GetGroupAttr();
    if (group_attr.HasAuthoredValue()) {
      group_attr.Get(&geom->group);
    }
  }

  return geom;
}

void ParseUsdPhysicsCollider(mjSpec* spec,
                             const pxr::UsdPhysicsCollisionAPI& collision_api,
                             const pxr::UsdPrim& body_prim, mjsBody* parent,
                             UsdCaches& caches) {
  pxr::UsdPrim prim = collision_api.GetPrim();
  // UsdPhysicsCollisionAPI can only be applied to gprim primitives.
  if (!prim.IsA<pxr::UsdGeomGprim>()) {
    mju_warning(
        "UsdPhysicsCollisionAPI applied to a non-UsdGeomGprim prim: %s. "
        "Skipping.",
        prim.GetPath().GetAsString().c_str());
    return;
  }

  mjsGeom* geom = mjs_addGeom(parent, nullptr);
  mjs_setName(geom->element, prim.GetPath().GetAsString().c_str());
  geom->contype = 1;
  geom->conaffinity = 1;

  SetUsdPrimPathUserValue(geom->element, prim.GetPath());

  ParseNewtonMassAPIForGeom(geom, prim);
  if (prim.HasAPI<pxr::MjcPhysicsCollisionAPI>()) {
    ParseMjcPhysicsCollisionAPI(geom, pxr::MjcPhysicsCollisionAPI(prim));
  }

  pxr::UsdShadeMaterial bound_material =
      pxr::UsdShadeMaterialBindingAPI(prim).ComputeBoundMaterial(
          &caches.bindings_cache, &caches.collection_query_cache);
  if (bound_material) {
    pxr::UsdPrim bound_material_prim = bound_material.GetPrim();
    if (bound_material_prim.HasAPI<pxr::UsdPhysicsMaterialAPI>() ||
        bound_material_prim.HasAPI<pxr::MjcPhysicsMaterialAPI>() ||
        bound_material_prim.HasAPI(NewtonTokens->NewtonMaterialAPI)) {
      ParseUsdPhysicsMaterialAPI(
          geom, pxr::UsdPhysicsMaterialAPI(bound_material_prim));
      ParseMjcPhysicsMaterialAPI(
          geom, bound_material_prim,
          pxr::MjcPhysicsMaterialAPI(bound_material_prim));
    }
    pxr::SdfPath material_path = bound_material_prim.GetPath();
    mjsMaterial* material = nullptr;
    if (auto iter = caches.parsed_materials.find(material_path);
        iter != caches.parsed_materials.end()) {
      material = iter->second;
    } else {
      material = ParseMaterial(spec, bound_material);
      // ParseMaterial may return a nullptr if the material is not supported.
      if (material) {
        caches.parsed_materials[material_path] = material;
      }
    }
    if (material) {
      mjs_setString(geom->material, mjs_getName(material->element)->c_str());
    }
  }

  // Parse the Mass API after the physics material APIs since the density
  // attribute from the Mass API is supposed to override the Material API
  // density attribute. See
  // https://openusd.org/dev/api/usd_physics_page_front.html
  if (prim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
    ParseUsdPhysicsMassAPIForGeom(geom, pxr::UsdPhysicsMassAPI(prim));
  }

  ParseDisplayColorAndOpacity(prim, geom);

  SetLocalPoseFromPrim(prim, body_prim, geom, caches.xform_cache);

  if (!MaybeParseGeomPrimitive(prim, geom, caches.xform_cache)) {
    mjsMesh* mesh = ParseUsdMesh(spec, prim, geom, caches.xform_cache);
    if (mesh != nullptr && (prim.HasAPI<pxr::MjcPhysicsMeshCollisionAPI>() ||
                            prim.HasAPI(NewtonTokens->NewtonMeshCollisionAPI) ||
                            prim.HasAPI(NewtonTokens->NewtonMassAPI))) {
      ParseMjcPhysicsMeshCollisionAPI(mesh,
                                      pxr::MjcPhysicsMeshCollisionAPI(prim));
    }
  }
}

void ParseJointEnabled(mjsEquality* eq, const pxr::UsdPhysicsJoint& joint) {
  bool jointEnabled = true;
  if (eq->type == mjEQ_JOINT) {
    // Mimic activation is independent of the follower joint's enabled state.
    // An unauthored mimic flag uses the schema default of true.
    joint.GetPrim()
        .GetAttribute(NewtonTokens->newtonMimicEnabled)
        .Get(&jointEnabled);
  } else {
    joint.GetJointEnabledAttr().Get(&jointEnabled);
  }
  eq->active = jointEnabled ? 1 : 0;
}

void ParseMjcEqualityAPISolverParams(
    mjsEquality* eq, const pxr::MjcPhysicsEqualityAPI& equality_api,
    const pxr::UsdPrim& prim) {
  auto solref_attr = equality_api.GetSolRefAttr();
  if (solref_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solref;
    solref_attr.Get(&solref);
    if (solref.size() == mjNREF) {
      for (int i = 0; i < mjNREF; ++i) {
        eq->solref[i] = solref[i];
      }
    } else {
      mju_warning(
          "solref attribute for equality %s has incorrect size "
          "%zu, expected %d.",
          prim.GetPath().GetAsString().c_str(), solref.size(), mjNREF);
    }
  }

  auto solimp_attr = equality_api.GetSolImpAttr();
  if (solimp_attr.HasAuthoredValue()) {
    pxr::VtDoubleArray solimp;
    solimp_attr.Get(&solimp);
    if (solimp.size() == mjNIMP) {
      for (int i = 0; i < mjNIMP; ++i) {
        eq->solimp[i] = solimp[i];
      }
    } else {
      mju_warning(
          "solimp attribute for equality %s has incorrect size "
          "%zu, expected %d.",
          prim.GetPath().GetAsString().c_str(), solimp.size(), mjNIMP);
    }
  }
}

void ParseConstraint(mjSpec* spec, const pxr::UsdPrim& prim, mjsBody* body,
                     pxr::UsdGeomXformCache& xform_cache) {
  if (prim.HasAPI(NewtonTokens->NewtonMimicAPI) ||
      prim.HasAPI<pxr::MjcPhysicsEqualityJointAPI>()) {
    pxr::MjcPhysicsEqualityJointAPI eq_joint_api(prim);
    mjsEquality* eq = mjs_addEquality(spec, nullptr);
    eq->type = mjEQ_JOINT;
    mjs_setName(eq->element, prim.GetPath().GetAsString().c_str());
    SetUsdPrimPathUserValue(eq->element, prim.GetPath());

    // The prim this API is applied to is the constrained joint (joint1).
    eq->objtype = mjOBJ_JOINT;
    mjs_setString(eq->name1, prim.GetPath().GetAsString().c_str());

    // Authored MJC aliases take precedence over Newton properties, as for
    // joint parameters. Schema defaults are not authored overrides.
    auto mjc_target = prim.GetRelationship(MjcPhysicsTokens->mjcTarget);
    auto newton_target = prim.GetRelationship(NewtonTokens->newtonMimicJoint);
    pxr::SdfPathVector targets;
    if (mjc_target && mjc_target.HasAuthoredTargets()) {
      // An explicitly empty target also wins: it constrains joint1 to a
      // constant rather than introducing a coupling to the Newton target.
      mjc_target.GetTargets(&targets);
      mju_warning(
          "Prim '%s' uses deprecated mjc:target. "
          "Please migrate to newton:mimicJoint.",
          prim.GetPath().GetText());
    } else if (newton_target) {
      newton_target.GetTargets(&targets);
    }
    if (!targets.empty()) {
      mjs_setString(eq->name2, targets[0].GetAsString().c_str());
    }

    auto mjc_coef0 = eq_joint_api.GetCoef0Attr();
    auto mjc_coef1 = eq_joint_api.GetCoef1Attr();
    auto newton_coef0 = prim.GetAttribute(NewtonTokens->newtonMimicCoef0);
    auto newton_coef1 = prim.GetAttribute(NewtonTokens->newtonMimicCoef1);
    if (mjc_coef0 && mjc_coef0.HasAuthoredValue()) {
      mjc_coef0.Get(&eq->data[0]);
      mju_warning(
          "Prim '%s' uses deprecated mjc:coef0. "
          "Please migrate to newton:mimicCoef0.",
          prim.GetPath().GetText());
    } else if (newton_coef0 && newton_coef0.HasAuthoredValue()) {
      float val;
      newton_coef0.Get(&val);
      // USD revolute offsets are degrees. MuJoCo equality coefficients use
      // radians regardless of compiler.angle; legacy MJC values already do.
      eq->data[0] =
          prim.IsA<pxr::UsdPhysicsRevoluteJoint>() ? val * mjPI / 180.0 : val;
    }
    if (mjc_coef1 && mjc_coef1.HasAuthoredValue()) {
      mjc_coef1.Get(&eq->data[1]);
      mju_warning(
          "Prim '%s' uses deprecated mjc:coef1. "
          "Please migrate to newton:mimicCoef1.",
          prim.GetPath().GetText());
    } else if (newton_coef1 && newton_coef1.HasAuthoredValue()) {
      float val;
      newton_coef1.Get(&val);
      eq->data[1] = val;
    }
    eq_joint_api.GetCoef2Attr().Get(&eq->data[2]);
    eq_joint_api.GetCoef3Attr().Get(&eq->data[3]);
    eq_joint_api.GetCoef4Attr().Get(&eq->data[4]);

    pxr::UsdPhysicsJoint joint(prim);
    ParseJointEnabled(eq, joint);

    // Solver params are now inline on MjcEqualityJointAPI
    auto solref_attr = prim.GetAttribute(MjcPhysicsTokens->mjcSolref);
    if (solref_attr.HasAuthoredValue()) {
      pxr::VtDoubleArray solref;
      solref_attr.Get(&solref);
      if (solref.size() == mjNREF) {
        for (int i = 0; i < mjNREF; ++i) {
          eq->solref[i] = solref[i];
        }
      }
    }
    auto solimp_attr = prim.GetAttribute(MjcPhysicsTokens->mjcSolimp);
    if (solimp_attr.HasAuthoredValue()) {
      pxr::VtDoubleArray solimp;
      solimp_attr.Get(&solimp);
      if (solimp.size() == mjNIMP) {
        for (int i = 0; i < mjNIMP; ++i) {
          eq->solimp[i] = solimp[i];
        }
      }
    }
  } else if (prim.IsA<pxr::UsdPhysicsFixedJoint>() ||
             prim.IsA<pxr::UsdPhysicsSphericalJoint>()) {
    // Handle fixed joints as weld constraints, spherical joints as connect
    // constraints
    pxr::UsdPhysicsJoint joint(prim);
    // A fixed joint means the bodies are welded.
    pxr::UsdRelationship body0_rel = joint.GetBody0Rel();
    pxr::UsdRelationship body1_rel = joint.GetBody1Rel();
    pxr::SdfPathVector targets0, targets1;
    body0_rel.GetTargets(&targets0);
    body1_rel.GetTargets(&targets1);

    pxr::SdfPath body0_path;
    if (!targets0.empty()) body0_path = targets0[0];
    pxr::SdfPath body1_path;
    if (!targets1.empty()) body1_path = targets1[0];

    auto stage = prim.GetStage();

    // Get the default prim path to identify the world body.
    pxr::SdfPath default_prim_path;
    if (stage->GetDefaultPrim().IsValid()) {
      default_prim_path = stage->GetDefaultPrim().GetPath();
    }

    // Map default prim to world body (empty path means world in MuJoCo).
    bool body0_is_world =
        body0_path.IsEmpty() || body0_path == default_prim_path;
    bool body1_is_world =
        body1_path.IsEmpty() || body1_path == default_prim_path;

    auto body0_prim = stage->GetPrimAtPath(body0_path);
    auto body1_prim = stage->GetPrimAtPath(body1_path);

    bool body0_is_site = false;
    if (!body0_path.IsEmpty()) {
      body0_is_site = body0_prim.HasAPI<pxr::MjcPhysicsSiteAPI>();
    }
    bool body1_is_site = false;
    if (!body1_path.IsEmpty()) {
      body1_is_site = body1_prim.HasAPI<pxr::MjcPhysicsSiteAPI>();
    }

    if (body0_is_site != body1_is_site) {
      mju_warning(
          "Weld constraint %s has mismatch between site and body targets, "
          "skipping",
          prim.GetPath().GetAsString().c_str());
      return;
    }

    mjsEquality* eq = mjs_addEquality(spec, nullptr);
    eq->type = mjEQ_WELD;
    mjs_setName(eq->element, prim.GetPath().GetAsString().c_str());
    SetUsdPrimPathUserValue(eq->element, prim.GetPath());

    if (body0_is_site) {
      mjs_setString(eq->name1, body0_path.GetAsString().c_str());
      mjs_setString(eq->name2, body1_path.GetAsString().c_str());
      eq->objtype = mjOBJ_SITE;
    } else {
      // For body welds, use "world" for the world body, otherwise use the USD
      // path.
      mjs_setString(eq->name1, body0_is_world
                                   ? "world"
                                   : body0_path.GetAsString().c_str());
      mjs_setString(eq->name2, body1_is_world
                                   ? "world"
                                   : body1_path.GetAsString().c_str());
      eq->objtype = mjOBJ_BODY;
    }

    // In USD, joints have a reference frame that is shared between the two
    // connecting bodies. This reference frame is defined relative to both
    // bodies, in localPos/Rot 0 and 1.
    // In MuJoCo this is specified as an equality constraint, with the
    // anchor point defined relative to body2 (USD body1) and
    // the relpose attribute defines the position of the anchor relative to
    // body1 (USD body0) and the orientation of body2 relative to body1.

    // Concretely:
    // anchor = localPos1
    // relpose = (localPos0, localRot0 * localRot1.GetConjugate())

    // relpose is float[7], pos(3) + quat(4)
    // anchor is float[3], pos(3)
    // in mjsEquality data: anchor 0-2, relpose 3-9.

    auto body0_xform = xform_cache.GetLocalToWorldTransform(body0_prim);
    auto body1_xform = xform_cache.GetLocalToWorldTransform(body1_prim);

    pxr::GfVec3d body0_scale, body1_scale;

    {
      pxr::GfMatrix4d scale_orient, rot, persp;
      pxr::GfVec3d translation;
      if (!body0_xform.Factor(&scale_orient, &body0_scale, &rot, &translation,
                              &persp)) {
        // unable to decompose, emit warning and set scale to identity
        mju_warning("Unable to decompose matrix for body 0: %s.",
                    body0_path.GetAsString().c_str());
        body0_scale = pxr::GfVec3f(1, 1, 1);
      }

      if (!body1_xform.Factor(&scale_orient, &body1_scale, &rot, &translation,
                              &persp)) {
        // unable to decompose, emit warning and set scale to identity
        mju_warning("Unable to decompose matrix for body 1: %s.",
                    body1_path.GetAsString().c_str());
        body1_scale = pxr::GfVec3f(1, 1, 1);
      }
    }

    pxr::GfVec3f localPos0;
    joint.GetLocalPos0Attr().Get(&localPos0);
    localPos0[0] *= body0_scale[0];
    localPos0[1] *= body0_scale[1];
    localPos0[2] *= body0_scale[2];

    pxr::GfVec3f localPos1;
    joint.GetLocalPos1Attr().Get(&localPos1);
    localPos1[0] *= body1_scale[0];
    localPos1[1] *= body1_scale[1];
    localPos1[2] *= body1_scale[2];

    pxr::GfQuatf localRot0, localRot1;
    joint.GetLocalRot0Attr().Get(&localRot0);
    joint.GetLocalRot1Attr().Get(&localRot1);

    auto relpose_quat = localRot0 * localRot1.GetConjugate();
    relpose_quat.Normalize();

    if (prim.IsA<pxr::UsdPhysicsSphericalJoint>() ||
        prim.HasAPI<pxr::MjcPhysicsEqualityConnectAPI>()) {
      eq->type = mjEQ_CONNECT;
      eq->data[0] = localPos0[0];
      eq->data[1] = localPos0[1];
      eq->data[2] = localPos0[2];

      if (prim.HasAPI<pxr::MjcPhysicsEqualityConnectAPI>()) {
        pxr::MjcPhysicsEqualityAPI equality_api(prim);
        ParseMjcEqualityAPISolverParams(eq, equality_api, prim);
      }
    } else {
      eq->data[0] = localPos1[0];
      eq->data[1] = localPos1[1];
      eq->data[2] = localPos1[2];

      bool has_relpose = joint.GetLocalPos0Attr().HasAuthoredValue() ||
                         joint.GetLocalRot0Attr().HasAuthoredValue() ||
                         joint.GetLocalRot1Attr().HasAuthoredValue();

      if (has_relpose) {
        eq->data[3] = localPos0[0];
        eq->data[4] = localPos0[1];
        eq->data[5] = localPos0[2];
        eq->data[6] = relpose_quat.GetReal();
        eq->data[7] = relpose_quat.GetImaginary()[0];
        eq->data[8] = relpose_quat.GetImaginary()[1];
        eq->data[9] = relpose_quat.GetImaginary()[2];
      }

      if (prim.HasAPI<pxr::MjcPhysicsEqualityWeldAPI>()) {
        pxr::MjcPhysicsEqualityAPI equality_api(prim);
        ParseMjcEqualityAPISolverParams(eq, equality_api, prim);

        pxr::MjcPhysicsEqualityWeldAPI weld_api(prim);
        float torque_scale;
        if (weld_api.GetTorqueScaleAttr().Get(&torque_scale)) {
          eq->data[10] = torque_scale;
        }
      }
    }

    ParseJointEnabled(eq, joint);
  }
}

void ParseUsdPhysicsJoint(mjSpec* spec, const pxr::UsdPrim& prim, mjsBody* body,
                          pxr::UsdGeomXformCache& xform_cache) {
  // A fixed joint means the bodies are welded.
  if (prim.IsA<pxr::UsdPhysicsFixedJoint>()) {
    // No joint needed for welded bodies.
    return;
  }

  pxr::UsdPhysicsJoint joint(prim);

  mjtJoint type;
  if (prim.IsA<pxr::UsdPhysicsRevoluteJoint>()) {
    type = mjJNT_HINGE;
  } else if (prim.IsA<pxr::UsdPhysicsPrismaticJoint>()) {
    type = mjJNT_SLIDE;
  } else {
    // Unsupported joint type.
    mju_warning("Unsupported joint type for %s",
                prim.GetPath().GetAsString().c_str());
    return;
  }

  mjsJoint* mj_joint = mjs_addJoint(body, nullptr);
  mj_joint->type = type;
  mjs_setName(mj_joint->element, prim.GetPath().GetAsString().c_str());

  SetUsdPrimPathUserValue(mj_joint->element, prim.GetPath());

  if (prim.IsA<pxr::UsdPhysicsRevoluteJoint>()) {
    pxr::UsdPhysicsRevoluteJoint revolute(prim);
    TfToken axis;
    revolute.GetAxisAttr().Get(&axis);
    if (axis == pxr::UsdGeomTokens->x) {
      mj_joint->axis[0] = 1;
      mj_joint->axis[1] = 0;
      mj_joint->axis[2] = 0;
    } else if (axis == pxr::UsdGeomTokens->y) {
      mj_joint->axis[0] = 0;
      mj_joint->axis[1] = 1;
      mj_joint->axis[2] = 0;
    } else {  // Z is default
      mj_joint->axis[0] = 0;
      mj_joint->axis[1] = 0;
      mj_joint->axis[2] = 1;
    }

    float lower, upper;
    if (revolute.GetLowerLimitAttr().Get(&lower) &&
        revolute.GetUpperLimitAttr().Get(&upper) &&
        !(std::isinf(lower) && lower < 0 && std::isinf(upper) && upper > 0)) {
      mj_joint->limited = mjLIMITED_TRUE;
      if (spec->compiler.degree) {
        mj_joint->range[0] = lower;
        mj_joint->range[1] = upper;
      } else {
        mj_joint->range[0] = lower * mjPI / 180.0;
        mj_joint->range[1] = upper * mjPI / 180.0;
      }
    }
  } else if (prim.IsA<pxr::UsdPhysicsPrismaticJoint>()) {
    pxr::UsdPhysicsPrismaticJoint prismatic(prim);
    TfToken axis;
    prismatic.GetAxisAttr().Get(&axis);
    if (axis == pxr::UsdGeomTokens->x) {
      mj_joint->axis[0] = 1;
      mj_joint->axis[1] = 0;
      mj_joint->axis[2] = 0;
    } else if (axis == pxr::UsdGeomTokens->y) {
      mj_joint->axis[0] = 0;
      mj_joint->axis[1] = 1;
      mj_joint->axis[2] = 0;
    } else {  // Z is default
      mj_joint->axis[0] = 0;
      mj_joint->axis[1] = 0;
      mj_joint->axis[2] = 1;
    }
    float lower, upper;
    if (prismatic.GetLowerLimitAttr().Get(&lower) &&
        prismatic.GetUpperLimitAttr().Get(&upper) &&
        !(std::isinf(lower) && lower < 0 && std::isinf(upper) && upper > 0)) {
      mj_joint->limited = mjLIMITED_TRUE;
      mj_joint->range[0] = lower;
      mj_joint->range[1] = upper;
    }
  }

  // localPose1 is joint frame in child body (body1) coordinates.
  pxr::GfVec3f localPos1;
  joint.GetLocalPos1Attr().Get(&localPos1);
  pxr::GfVec3d pos(localPos1);

  pxr::GfQuatf localRot1;
  joint.GetLocalRot1Attr().Get(&localRot1);
  pxr::GfQuatd rot(localRot1);

  SetDoubleArrFromGfVec3d(mj_joint->pos, pos);

  // Transform joint axis by localPose1 rotation.
  if (mj_joint->type == mjJNT_HINGE || mj_joint->type == mjJNT_SLIDE) {
    pxr::GfVec3d axis_vec(mj_joint->axis[0], mj_joint->axis[1],
                          mj_joint->axis[2]);
    pxr::GfVec3d rotated_axis = rot.Transform(axis_vec);
    SetDoubleArrFromGfVec3d(mj_joint->axis, rotated_axis);
  }

  if (prim.HasAPI<pxr::MjcPhysicsJointAPI>() ||
      prim.HasAPI(NewtonTokens->NewtonJointAPI)) {
    ParseMjcPhysicsJointAPI(mj_joint, pxr::MjcPhysicsJointAPI(prim));
  }
}

void ParseSite(mjSpec* spec, const pxr::UsdPrim& prim,
               const pxr::UsdPrim& parent_prim, mjsBody* parent,
               pxr::UsdGeomXformCache& xform_cache) {
  mjsSite* site = mjs_addSite(parent, 0);
  mjs_setName(site->element, prim.GetPath().GetAsString().c_str());
  SetLocalPoseFromPrim(prim, parent_prim, site, xform_cache);

  SetUsdPrimPathUserValue(site->element, prim.GetPath());

  if (prim.HasAPI<pxr::MjcPhysicsSiteAPI>()) {
    auto site_api = pxr::MjcPhysicsSiteAPI(prim);
    auto group_attr = site_api.GetGroupAttr();
    if (group_attr.HasAuthoredValue()) {
      group_attr.Get(&site->group);
    }
  }

  // Convert USD type to MuJoCo type.
  if (!MaybeParseGeomPrimitive(prim, site, xform_cache)) {
    mju_error("Prim with SiteAPI has unsupported type %s",
              prim.GetTypeName().GetString().c_str());
    return;
  }
}

void ParseMjcPhysicsKeyframe(mjSpec* spec,
                             const pxr::MjcPhysicsKeyframe& keyframe) {
  auto prim = keyframe.GetPrim();
  auto qpos_attr = keyframe.GetMjcQposAttr();
  auto qvel_attr = keyframe.GetMjcQvelAttr();
  auto act_attr = keyframe.GetMjcActAttr();
  auto ctrl_attr = keyframe.GetMjcCtrlAttr();
  auto mpos_attr = keyframe.GetMjcMposAttr();
  auto mquat_attr = keyframe.GetMjcMquatAttr();

  auto setKeyframeData = [](mjsKey* key, const pxr::UsdAttribute& attr,
                            std::vector<double>** key_data,
                            double* time = nullptr) {
    if (attr.HasAuthoredValue()) {
      pxr::VtDoubleArray data;
      if (time == nullptr) {
        attr.Get(&data);
      } else {
        attr.Get(&data, *time);
      }
      *key_data = new std::vector<double>(data.begin(), data.end());
    }
  };

  size_t n_time_samples = 0;
  if (qpos_attr.HasAuthoredValue()) {
    n_time_samples = qpos_attr.GetNumTimeSamples();
  }

  if (n_time_samples == 0) {
    // If no time samples, we create a single keyframe.
    mjsKey* key = mjs_addKey(spec);

    SetUsdPrimPathUserValue(key->element, prim.GetPath());

    mjs_setName(key->element, prim.GetName().GetString().c_str());
    setKeyframeData(key, qpos_attr, &key->qpos);
    setKeyframeData(key, qvel_attr, &key->qvel);
    setKeyframeData(key, act_attr, &key->act);
    setKeyframeData(key, ctrl_attr, &key->ctrl);
    setKeyframeData(key, mpos_attr, &key->mpos);
    setKeyframeData(key, mquat_attr, &key->mquat);
  } else {
    // If time samples, we create a keyframe for each time sample.
    std::vector<double> times;
    qpos_attr.GetTimeSamples(&times);
    int keyframe_id = 0;
    for (double time : times) {
      mjsKey* key = mjs_addKey(spec);

      SetUsdPrimPathUserValue(key->element, prim.GetPath());

      std::string key_name =
          prim.GetName().GetString() + "_" + std::to_string(keyframe_id++);
      mjs_setName(key->element, key_name.c_str());
      key->time = time;
      setKeyframeData(key, qpos_attr, &key->qpos, &time);
      setKeyframeData(key, qvel_attr, &key->qvel, &time);
      setKeyframeData(key, act_attr, &key->act, &time);
      setKeyframeData(key, ctrl_attr, &key->ctrl, &time);
      setKeyframeData(key, mpos_attr, &key->mpos, &time);
      setKeyframeData(key, mquat_attr, &key->mquat, &time);
    }
  }
}

void ParseUsdFilteredPairsAPI(mjSpec* spec, const pxr::UsdPrim& prim) {
  pxr::SdfPathVector filtered_bodies;
  pxr::UsdPhysicsFilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets(
      &filtered_bodies);
  for (const auto& filtered_body : filtered_bodies) {
    mjsExclude* exclude = mjs_addExclude(spec);
    mjs_setString(exclude->bodyname1, prim.GetPath().GetAsString().c_str());
    mjs_setString(exclude->bodyname2, filtered_body.GetAsString().c_str());
  }
}

void ParseNewtonArticulationRootAPI(mjSpec* spec, const pxr::UsdPrim& prim) {
  auto add_mobility_attr =
      prim.GetAttribute(NewtonTokens->newtonJointsAddMobility);
  if (add_mobility_attr && add_mobility_attr.HasAuthoredValue()) {
    bool add_mobility;
    add_mobility_attr.Get(&add_mobility);
    if (!add_mobility) {
      mju_warning(
          "Prim '%s' has newton:jointsAddMobility set to false, but MuJoCo "
          "only supports reduced-coordinate formulations where joints add "
          "mobility. Joints will be treated as adding mobility.",
          prim.GetPath().GetText());
    }
  }
}

mjsBody* ParseUsdPhysicsRigidbody(
    mjSpec* spec, const pxr::UsdPhysicsRigidBodyAPI& rigidbody_api,
    const pxr::UsdPrim& parent_prim, mjsBody* parent,
    pxr::UsdGeomXformCache& xform_cache) {
  pxr::UsdPrim prim = rigidbody_api.GetPrim();
  mjsBody* body = mjs_addBody(parent, nullptr);
  mjs_setName(body->element, prim.GetPath().GetAsString().c_str());
  SetLocalPoseFromPrim(prim, parent_prim, body, xform_cache);

  if (prim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
    ParseUsdPhysicsMassAPIForBody(body, pxr::UsdPhysicsMassAPI(prim));
  }

  if (prim.HasAPI(NewtonTokens->NewtonArticulationRootAPI)) {
    ParseNewtonArticulationRootAPI(spec, prim);
  }

  SetUsdPrimPathUserValue(body->element, prim.GetPath());

  if (prim.HasAPI<pxr::UsdPhysicsFilteredPairsAPI>()) {
    ParseUsdFilteredPairsAPI(spec, prim);
  }

  return body;
}

// There is no common base class for UsdPhysicsRigidBodyAPI and
// UsdPhysicsCollisionAPI so we need a templated function.
template <class T>
bool IsObjectInPhysicsScene(const T& object,
                            const pxr::UsdPhysicsScene& physics_scene) {
  pxr::SdfPathVector sim_owners;
  object.GetSimulationOwnerRel().GetTargets(&sim_owners);
  for (const auto& sim_owner : sim_owners) {
    if (physics_scene.GetPath() == sim_owner) {
      return true;
    }
  }
  return false;
};

// Helper type to store all the prims that belong to a body.
using BodyPrimMap = std::map<pxr::SdfPath, std::vector<pxr::SdfPath>>;

bool HasAuthoredMassOrDensity(const pxr::UsdPrim& prim) {
  if (!prim.HasAPI<pxr::UsdPhysicsMassAPI>()) {
    return false;
  }
  pxr::UsdPhysicsMassAPI mass_api(prim);
  return mass_api.GetMassAttr().HasAuthoredValue() ||
         mass_api.GetDensityAttr().HasAuthoredValue();
}

bool BodySubtreeHasMass(mjsBody* body) {
  if (body->explicitinertial || body->mass > 0) {
    return true;
  }
  for (mjsElement* el = mjs_firstChild(body, mjOBJ_GEOM, 0); el != nullptr;
       el = mjs_nextChild(body, el, 0)) {
    mjsGeom* geom = mjs_asGeom(el);
    if (geom->mass > 0 || geom->density > 0) {
      return true;
    }
  }
  for (mjsElement* el = mjs_firstChild(body, mjOBJ_BODY, 0); el != nullptr;
       el = mjs_nextChild(body, el, 0)) {
    mjsBody* child = mjs_asBody(el);
    if (mjs_firstChild(child, mjOBJ_JOINT, 0) == nullptr &&
        BodySubtreeHasMass(child)) {
      return true;
    }
  }
  return false;
}

// Recursively traverses the kinematic tree, creating bodies, joints, and geoms
// in the mjSpec.
void PopulateSpecFromTree(pxr::UsdStageRefPtr stage, mjSpec* spec,
                          mjsBody* parent_mj_body, const Node* parent_node,
                          const Node* current_node, UsdCaches& caches) {
  mjsBody* current_mj_body = nullptr;

  if (!current_node->body_path.IsEmpty()) {
    // This is *not* the world body.
    pxr::SdfPath parent_body_path =
        parent_node ? parent_node->body_path : pxr::SdfPath();
    pxr::UsdPrim parent_prim_for_xform =
        parent_body_path.IsEmpty() ? stage->GetPseudoRoot()
                                   : stage->GetPrimAtPath(parent_body_path);

    current_mj_body = ParseUsdPhysicsRigidbody(
        spec, pxr::UsdPhysicsRigidBodyAPI::Get(stage, current_node->body_path),
        parent_prim_for_xform, parent_mj_body, caches.xform_cache);
  } else {
    current_mj_body = mjs_findBody(spec, "world");
  }

  if (!current_node->joints.empty()) {
    for (const auto& joint_path : current_node->joints) {
      ParseUsdPhysicsJoint(spec, stage->GetPrimAtPath(joint_path),
                           current_mj_body, caches.xform_cache);
    }
  } else if (parent_mj_body == mjs_findBody(spec, "world")) {
    // No joint to parent, and parent is world: this is a floating body.
    mjsJoint* free_joint = mjs_addJoint(current_mj_body, nullptr);
    free_joint->type = mjJNT_FREE;
  }

  for (const auto& constraint_path : current_node->constraints) {
    ParseConstraint(spec, stage->GetPrimAtPath(constraint_path),
                    current_mj_body, caches.xform_cache);
  }

  pxr::UsdPrim body_prim_for_xform =
      current_node->body_path.IsEmpty()
          ? stage->GetPseudoRoot()
          : stage->GetPrimAtPath(current_node->body_path);

  std::vector<mjsGeom*> unauthored_visual_geoms;
  for (const auto& gprim_path : current_node->visual_gprims) {
    auto gprim = stage->GetPrimAtPath(gprim_path);
    mjsGeom* geom = ParseUsdGeomGprim(spec, gprim, body_prim_for_xform,
                                      current_mj_body, caches);
    if (!HasAuthoredMassOrDensity(gprim)) {
      unauthored_visual_geoms.push_back(geom);
    }
  }

  for (const auto& collider_path : current_node->colliders) {
    ParseUsdPhysicsCollider(
        spec, pxr::UsdPhysicsCollisionAPI(stage->GetPrimAtPath(collider_path)),
        body_prim_for_xform, current_mj_body, caches);
  }

  for (const auto& site_path : current_node->sites) {
    ParseSite(spec, stage->GetPrimAtPath(site_path), body_prim_for_xform,
              current_mj_body, caches.xform_cache);
  }

  // Recurse through children.
  for (const auto& child_node : current_node->children) {
    PopulateSpecFromTree(stage, spec, current_mj_body, current_node,
                         child_node.get(), caches);
  }

  if (!unauthored_visual_geoms.empty() &&
      mjs_firstChild(current_mj_body, mjOBJ_JOINT, 0) != nullptr &&
      !BodySubtreeHasMass(current_mj_body)) {
    mju_warning(
        "Body '%s' has only visual gprims and no colliders or explicit "
        "inertia; falling back to default density on visual geoms.",
        current_node->body_path.GetText());
    mjsGeom default_geom;
    mjs_defaultGeom(&default_geom);
    for (mjsGeom* geom : unauthored_visual_geoms) {
      geom->density = default_geom.density;
    }
  }
}

mjSpec* ParseStage(const pxr::UsdStageRefPtr stage) {
  mjSpec* spec = mj_makeSpec();
  std::string model_name = GetStageModelName(stage);
  if (!model_name.empty()) {
    mjs_setString(spec->modelname, model_name.c_str());
  }

  std::unique_ptr<Node> root = BuildKinematicTree(stage);
  if (!root) {
    mj_deleteSpec(spec);
    return nullptr;
  }

  // First parse the physics scene and other root elements such as keyframes
  // and actuators.
  if (!root->physics_scene.IsEmpty()) {
    ParseUsdPhysicsScene(spec,
                         pxr::UsdPhysicsScene::Get(stage, root->physics_scene));
  } else {
    // If there is no physics scene we still need to infer the gravity vector
    // from the stage up axis and units per meter metadata.
    SetGravityAttributes(spec, stage);
  }

  if (!root->keyframes.empty()) {
    for (const auto& keyframe : root->keyframes) {
      ParseMjcPhysicsKeyframe(spec,
                              pxr::MjcPhysicsKeyframe::Get(stage, keyframe));
    }
  }

  if (!root->actuators.empty()) {
    for (const auto& actuator : root->actuators) {
      ParseMjcPhysicsActuator(spec,
                              pxr::MjcPhysicsActuator::Get(stage, actuator));
    }
  }

  if (!root->tendons.empty()) {
    for (const auto& tendon : root->tendons) {
      ParseMjcPhysicsTendon(spec, pxr::MjcPhysicsTendon::Get(stage, tendon));
    }
  }

  // Set of caches to use for all queries when parsing.
  UsdCaches caches;
  // Then populate the kinematic tree.
  PopulateSpecFromTree(stage, spec, /*parent_mj_body=*/nullptr,
                       /*parent_node=*/nullptr, root.get(), caches);

  return spec;
}

namespace {
// load 2D
mjSpec* Decode(mjResource* resource, const mjVFS* vfs, char* error,
               int error_sz) {
  auto stage = pxr::UsdStage::Open(resource->name);
  if (!stage) {
    if (error && error_sz > 0) {
      snprintf(error, error_sz, "usd_decoder: could not open USD stage '%s'",
               resource->name);
    }
    return nullptr;
  }
  return ParseStage(stage);
}

int CanDecode(const mjResource* resource) {
  std::string_view name(resource->name);
  auto ends_with = [name](std::string_view suffix) {
    return name.size() >= suffix.size() &&
           name.substr(name.size() - suffix.size()) == suffix;
  };
  return ends_with(".usd") || ends_with(".usda") || ends_with(".usdc") ||
         ends_with(".usdz");
}
}  // namespace

// clang-format off
mjPLUGIN_LIB_INIT(usd_decoder) {
  mjpDecoder decoder;
  mjp_defaultDecoder(&decoder);
  decoder.content_type = "model/usd";
  decoder.extension = ".usd|.usda|.usdc|.usdz";
  decoder.decode = Decode;
  decoder.can_decode = CanDecode;
  mjp_registerDecoder(&decoder);
}
