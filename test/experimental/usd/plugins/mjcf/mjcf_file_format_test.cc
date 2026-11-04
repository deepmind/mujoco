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
#include <numbers>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/experimental/usd/mjcPhysics/actuator.h>
#include <mujoco/experimental/usd/mjcPhysics/collisionAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/imageableAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/jointAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/meshCollisionAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/sceneAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/siteAPI.h>
#include <mujoco/experimental/usd/mjcPhysics/tokens.h>
#include "src/experimental/usd/plugins/mjcf/utils.h"
#include "test/experimental/usd/test_utils.h"
#include "test/fixture.h"
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/tf/staticData.h>
#include <pxr/base/tf/staticTokens.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/pxr.h>
#include <pxr/usd/kind/registry.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/declareHandles.h>
#include <pxr/usd/sdf/fileFormat.h>
#include <pxr/usd/sdf/listOp.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/primSpec.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/modelAPI.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>  // IWYU pragma: keep, used for TraverseAll
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/timeCode.h>
#include <pxr/usd/usd/tokens.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/plane.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
#include <pxr/usd/usdPhysics/fixedJoint.h>
#include <pxr/usd/usdPhysics/joint.h>
#include <pxr/usd/usdPhysics/massAPI.h>
#include <pxr/usd/usdPhysics/meshCollisionAPI.h>
#include <pxr/usd/usdPhysics/prismaticJoint.h>
#include <pxr/usd/usdPhysics/revoluteJoint.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#include <pxr/usd/usdPhysics/scene.h>
#include <pxr/usd/usdPhysics/tokens.h>

PXR_NAMESPACE_OPEN_SCOPE
// clang-format off
TF_DEFINE_PRIVATE_TOKENS(_tokens,
                         (st)
                         ((newtonTimeStepsPerSecond, "newton:timeStepsPerSecond"))
                         ((newtonMaxSolverIterations, "newton:maxSolverIterations"))
                         ((newtonGravityEnabled, "newton:gravityEnabled"))
                         );
// clang-format on
PXR_NAMESPACE_CLOSE_SCOPE

namespace mujoco {
namespace usd {

using pxr::MjcPhysicsSiteAPI;
using pxr::MjcPhysicsTokens;
using pxr::SdfPath;
using MjcfSdfFileFormatPluginTest = MujocoTest;

static const char* kMaterialsPath =
    "experimental/usd/plugins/mjcf/testdata/materials.xml";
static const char* kMeshObjPath =
    "experimental/usd/plugins/mjcf/testdata/mesh_obj.xml";

TEST_F(MjcfSdfFileFormatPluginTest, TestClassAuthored) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <default>
        <default class="test">
        </default>
      </default>
      <worldbody>
        <body name="test_body" pos="0 0 0">
          <geom class="test" type="sphere" size="2 2 2"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  EXPECT_PRIM_VALID(stage, "/__class__");
  EXPECT_PRIM_VALID(stage, "/__class__/test");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestBasicMeshSources) {
  static constexpr char kXml[] = R"(
    <mujoco model="mesh test">
      <asset>
        <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <body name="test_body">
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_PRIM_VALID(stage, "/mesh_test");
  EXPECT_PRIM_VALID(stage, "/mesh_test/test_body/tetrahedron");
  EXPECT_PRIM_VALID(stage, "/mesh_test/test_body/tetrahedron/Mesh");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsMaterials) {
  static constexpr char kXml[] = R"(
    <mujoco model="physics materials test">
      <worldbody>
        <body name="test_body">
          <geom name="geom_with_friction" type="sphere" size="1" friction="4 5 6"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  auto stage = OpenStage(kXml);
  EXPECT_PRIM_VALID(
      stage, "/physics_materials_test/PhysicsMaterials/geom_with_friction");
  EXPECT_REL_HAS_TARGET(
      stage,
      "/physics_materials_test/test_body/geom_with_friction.material:binding",
      "/physics_materials_test/PhysicsMaterials/geom_with_friction");
  ExpectAttributeEqual(stage,
                       "/physics_materials_test/PhysicsMaterials/"
                       "geom_with_friction.physics:dynamicFriction",
                       4.0f);
  ExpectAttributeEqual(stage,
                       "/physics_materials_test/PhysicsMaterials/"
                       "geom_with_friction.newton:torsionalFriction",
                       5.0f);
  ExpectAttributeEqual(stage,
                       "/physics_materials_test/PhysicsMaterials/"
                       "geom_with_friction.newton:rollingFriction",
                       6.0f);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage,
      "/physics_materials_test/PhysicsMaterials/"
      "geom_with_friction.mjc:torsionalfriction");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage,
      "/physics_materials_test/PhysicsMaterials/"
      "geom_with_friction.mjc:rollingfriction");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMaterials) {
  const std::string xml_path = GetTestDataFilePath(kMaterialsPath);

  auto stage = pxr::UsdStage::Open(xml_path);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_VALID(stage, "/mesh_test");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials");

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_red");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_red/PreviewSurface");
  ExpectAttributeEqual(
      stage,
      "/mesh_test/Materials/material_red/PreviewSurface.inputs:diffuseColor",
      pxr::GfVec3f(0.8, 0, 0));
  ExpectAttributeHasConnection(
      stage, "/mesh_test/Materials/material_red.outputs:surface",
      "/mesh_test/Materials/material_red/PreviewSurface.outputs:surface");
  ExpectAttributeHasConnection(
      stage, "/mesh_test/Materials/material_red.outputs:displacement",
      "/mesh_test/Materials/material_red/PreviewSurface.outputs:displacement");

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_texture");
  EXPECT_PRIM_VALID(stage,
                    "/mesh_test/Materials/material_texture/PreviewSurface");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_texture/uvmap");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_texture/diffuse");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_texture/"
      "PreviewSurface.inputs:diffuseColor",
      "/mesh_test/Materials/material_texture/diffuse.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_texture/diffuse.inputs:file",
      pxr::SdfAssetPath("textures/cube.png"));

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_metallic");
  EXPECT_PRIM_VALID(stage,
                    "/mesh_test/Materials/material_metallic/PreviewSurface");
  ExpectAttributeEqual(
      stage,
      "/mesh_test/Materials/material_metallic/PreviewSurface.inputs:metallic",
      0.6f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMaterialLayers) {
  const std::string xml_path = GetTestDataFilePath(kMaterialsPath);
  auto stage = pxr::UsdStage::Open(xml_path);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered/uvmap");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered/diffuse");

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered/normal");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_layered/"
      "PreviewSurface.inputs:normal",
      "/mesh_test/Materials/material_layered/normal.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_layered/normal.inputs:file",
      pxr::SdfAssetPath("textures/normal.png"));

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered/orm_packed");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_layered/"
      "PreviewSurface.inputs:occlusion",
      "/mesh_test/Materials/material_layered/orm_packed.outputs:r");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_layered/"
      "PreviewSurface.inputs:roughness",
      "/mesh_test/Materials/material_layered/orm_packed.outputs:g");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_layered/"
      "PreviewSurface.inputs:metallic",
      "/mesh_test/Materials/material_layered/orm_packed.outputs:b");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_layered/orm_packed.inputs:file",
      pxr::SdfAssetPath("textures/orm.png"));

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_layered/emissive");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_layered/"
      "PreviewSurface.inputs:emissiveColor",
      "/mesh_test/Materials/material_layered/emissive.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_layered/emissive.inputs:file",
      pxr::SdfAssetPath("textures/emissive.png"));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMaterialPBRSeparate) {
  const std::string xml_path = GetTestDataFilePath(kMaterialsPath);
  auto stage = pxr::UsdStage::Open(xml_path);

  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_pbr_separate");
  EXPECT_PRIM_VALID(stage, "/mesh_test/Materials/material_pbr_separate/uvmap");
  EXPECT_PRIM_VALID(stage,
                    "/mesh_test/Materials/material_pbr_separate/occlusion");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_pbr_separate/"
      "PreviewSurface.inputs:occlusion",
      "/mesh_test/Materials/material_pbr_separate/occlusion.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_pbr_separate/occlusion.inputs:file",
      pxr::SdfAssetPath("textures/occlusion.png"));
  EXPECT_PRIM_VALID(stage,
                    "/mesh_test/Materials/material_pbr_separate/roughness");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_pbr_separate/"
      "PreviewSurface.inputs:roughness",
      "/mesh_test/Materials/material_pbr_separate/roughness.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_pbr_separate/roughness.inputs:file",
      pxr::SdfAssetPath("textures/roughness.png"));
  EXPECT_PRIM_VALID(stage,
                    "/mesh_test/Materials/material_pbr_separate/metallic");
  ExpectAttributeHasConnection(
      stage,
      "/mesh_test/Materials/material_pbr_separate/"
      "PreviewSurface.inputs:metallic",
      "/mesh_test/Materials/material_pbr_separate/metallic.outputs:rgb");
  ExpectAttributeEqual(
      stage, "/mesh_test/Materials/material_pbr_separate/metallic.inputs:file",
      pxr::SdfAssetPath("textures/metallic.png"));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestGeomRgba) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <geom type="sphere" name="sphere_red" size="1" rgba="1 0 0 1"/>
        <geom type="sphere" name="sphere_default" size="1"/>
        <geom type="sphere" name="sphere_also_default" size="1" rgba="0.5 0.5 0.5 1"/>
        <geom type="sphere" name="sphere_almost_default" size="1" rgba="0.5 0.5 0.5 0.9"/>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  EXPECT_PRIM_VALID(stage, "/test/sphere_red");
  ExpectAttributeEqual(stage, "/test/sphere_red.primvars:displayColor",
                       pxr::VtArray<pxr::GfVec3f>{{1, 0, 0}});
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(stage,
                                "/test/sphere_red.primvars:displayOpacity");

  // There's no mechanism in Mujoco to specify whether an attribute was set
  // explicitly or not. We do the same as Mujoco does, which is to compare with
  // the default value.
  // Which explains why not setting rgba is the same as setting it to the
  // default value of (0.5, 0.5, 0.5, 1).
  EXPECT_PRIM_VALID(stage, "/test/sphere_default");
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(stage,
                                "/test/sphere_default.primvars:displayColor");
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(stage,
                                "/test/sphere_default.primvars:displayOpacity");

  EXPECT_PRIM_VALID(stage, "/test/sphere_also_default");
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(
      stage, "/test/sphere_also_default.primvars:displayColor");
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(
      stage, "/test/sphere_also_default.primvars:displayOpacity");

  EXPECT_PRIM_VALID(stage, "/test/sphere_almost_default");
  ExpectAttributeEqual(stage,
                       "/test/sphere_almost_default.primvars:displayColor",
                       pxr::VtArray<pxr::GfVec3f>{{0.5, 0.5, 0.5}});
  ExpectAttributeEqual(stage,
                       "/test/sphere_almost_default.primvars:displayOpacity",
                       pxr::VtArray<float>{0.9});
}

TEST_F(MjcfSdfFileFormatPluginTest, TestSiteRgba) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <site type="sphere" name="sphere_red" size="1" rgba="1 0 0 1"/>
      </worldbody>
    </mujoco>
  )";

  pxr::SdfLayerRefPtr layer = LoadLayer(kXml);
  auto stage = pxr::UsdStage::Open(layer);

  EXPECT_PRIM_VALID(stage, "/test/sphere_red");
  ExpectAttributeEqual(stage, "/test/sphere_red.primvars:displayColor",
                       pxr::VtArray<pxr::GfVec3f>{{1, 0, 0}});
  EXPECT_ATTRIBUTE_HAS_NO_VALUE(stage,
                                "/test/sphere_red.primvars:displayOpacity");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestFaceVaryingMeshSourcesSimpleMjcfMesh) {
  static constexpr char kXml[] = R"(
    <mujoco model="mesh test">
      <asset>
        <mesh
          name="tetrahedron"
          face="0 3 2  0 1 3  0 2 1  1 2 3"
          vertex="0 1 0  0 0 0  1 0 1  1 0 -1"
          normal="1 0 0  0 1 0  0 0 1  -1 0 0"
          texcoord="0.5 0.5  0 0.5  1 1  1 0"/>
      </asset>
      <worldbody>
        <body name="test_body">
          <geom type="mesh" mesh="tetrahedron"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  auto mesh = pxr::UsdGeomMesh::Get(
      stage, SdfPath("/mesh_test/test_body/tetrahedron/Mesh"));
  ASSERT_TRUE(mesh);
  pxr::VtArray<int> face_vertex_counts;
  mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
  EXPECT_EQ(face_vertex_counts.size(), 4);
  EXPECT_EQ(face_vertex_counts, pxr::VtArray<int>({3, 3, 3, 3}));

  pxr::VtArray<int> face_vertex_indices;
  mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);
  EXPECT_EQ(face_vertex_indices.size(), 12);
  EXPECT_EQ(face_vertex_indices,
            pxr::VtArray<int>({0, 3, 2, 0, 1, 3, 0, 2, 1, 1, 2, 3}));

  pxr::VtArray<pxr::GfVec3f> normals;
  mesh.GetNormalsAttr().Get(&normals);
  EXPECT_EQ(normals.size(), face_vertex_indices.size());
  // We can't directly check the normals values because they are altered by
  // Mujoco's compiling step. So we at least check that normals with the same
  // original index are the same.
  for (int i = 0; i < face_vertex_indices.size(); ++i) {
    for (int j = i + 1; j < face_vertex_indices.size(); ++j) {
      if (face_vertex_indices[i] == face_vertex_indices[j]) {
        EXPECT_EQ(normals[i], normals[j]);
      }
    }
  }

  auto primvars_api = pxr::UsdGeomPrimvarsAPI(mesh.GetPrim());

  pxr::VtArray<pxr::GfVec2f> texcoords;
  EXPECT_TRUE(primvars_api.HasPrimvar(pxr::_tokens->st));
  auto primvar_st = primvars_api.GetPrimvar(pxr::_tokens->st);
  primvar_st.Get(&texcoords);
  EXPECT_EQ(texcoords.size(), face_vertex_indices.size());

  // Check the faceVarying texcoords against the manually indexed source
  // texcoords.
  pxr::VtArray<pxr::GfVec2f> source_texcoords{
      {0.5, 0.5}, {0, 0.5}, {1, 0}, {1, 1}};
  for (int i = 0; i < face_vertex_indices.size(); ++i) {
    EXPECT_EQ(texcoords[i], source_texcoords[face_vertex_indices[i]]);
  }
}

TEST_F(MjcfSdfFileFormatPluginTest,
       TestFaceVaryingMeshSourcesObjWithIndexedNormals) {
  const std::string xml_path = GetTestDataFilePath(kMeshObjPath);

  auto stage = pxr::UsdStage::Open(xml_path);
  EXPECT_THAT(stage, testing::NotNull());

  auto mesh =
      pxr::UsdGeomMesh::Get(stage, SdfPath("/mesh_test/test_body/mesh/Mesh"));
  ASSERT_TRUE(mesh);
  pxr::VtArray<int> face_vertex_counts;
  mesh.GetFaceVertexCountsAttr().Get(&face_vertex_counts);
  EXPECT_EQ(face_vertex_counts.size(), 4);
  EXPECT_EQ(face_vertex_counts, pxr::VtArray<int>({3, 3, 3, 3}));

  pxr::VtArray<int> face_vertex_indices;
  mesh.GetFaceVertexIndicesAttr().Get(&face_vertex_indices);
  EXPECT_EQ(face_vertex_indices.size(), 12);
  EXPECT_EQ(face_vertex_indices,
            pxr::VtArray<int>({0, 3, 2, 0, 1, 3, 0, 2, 1, 1, 2, 3}));

  pxr::VtArray<pxr::GfVec3f> normals;
  mesh.GetNormalsAttr().Get(&normals);
  EXPECT_EQ(normals.size(), face_vertex_indices.size());
  // We can't directly check the normals values because they are altered by
  // Mujoco's compiling step.
  // We also can't access the normals indexing data, and can't use the vertex
  // indexing data here because they are separate.
  // So we check that the first half of the normals are the same, then the
  // second half, as set in the OBJ file.
  pxr::GfVec3f first_half_normal = normals[0];
  pxr::GfVec3f second_half_normal = normals[face_vertex_indices.size() / 2];
  EXPECT_NE(first_half_normal, second_half_normal);
  int i = 0;
  for (; i < face_vertex_indices.size() / 2; ++i) {
    EXPECT_EQ(normals[i], first_half_normal);
  }
  for (; i < face_vertex_indices.size(); ++i) {
    EXPECT_EQ(normals[i], second_half_normal);
  }

  auto primvars_api = pxr::UsdGeomPrimvarsAPI(mesh.GetPrim());

  pxr::VtArray<pxr::GfVec2f> texcoords;
  EXPECT_TRUE(primvars_api.HasPrimvar(pxr::_tokens->st));
  auto primvar_st = primvars_api.GetPrimvar(pxr::_tokens->st);
  primvar_st.Get(&texcoords);
  EXPECT_EQ(texcoords.size(), face_vertex_indices.size());

  // Check the faceVarying texcoords against the manually indexed source
  // texcoords.
  // NOTE: For OBJ we must use different indices for the texcoords than for the
  // vertices!
  std::vector<int> source_face_texcoord_indices{0, 1, 2, 1, 2, 3,
                                                2, 3, 0, 3, 0, 1};
  pxr::VtArray<pxr::GfVec2f> source_texcoords{
      {0.5, 0.5}, {0, 0.5}, {1, 0}, {1, 1}};
  // NOTE: The v component of the texcoords is flipped when Mujoco loads the
  // OBJ.
  for (auto& uv : source_texcoords) {
    uv[1] = 1 - uv[1];
  }
  for (int i = 0; i < source_face_texcoord_indices.size(); ++i) {
    EXPECT_EQ(texcoords[i], source_texcoords[source_face_texcoord_indices[i]]);
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, TestBody) {
  static constexpr char kXml[] = R"(
    <mujoco model="body test">
      <asset>
        <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <body name="test_body" pos="0 1 0">
          <joint type="free" />
          <frame pos="0 0 1">
            <frame pos="0 0 1">
              <body name="test_body_2" pos="1 0 0">
                <geom type="mesh" mesh="tetrahedron"/>
              </body>
            </frame>
          </frame>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  EXPECT_PRIM_VALID(stage, "/body_test");
  EXPECT_PRIM_VALID(stage, "/body_test/test_body");
  EXPECT_PRIM_VALID(stage, "/body_test/test_body/test_body_2");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestBasicParenting) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="root" pos="0 1 0">
              <body name="root/body_1" pos="1 0 0" />
              <body name="root/body_2" pos="1 0 0">
                <body name="root/body_3" pos="1 0 0" />
              </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_PRIM_VALID(stage, "/test/root");
  EXPECT_PRIM_VALID(stage, "/test/root/root_body_1");
  EXPECT_PRIM_VALID(stage, "/test/root/root_body_2");
  EXPECT_PRIM_VALID(stage, "/test/root/root_body_2/root_body_3");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestJointsDoNotAffectParenting) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <asset>
        <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <body name="root" pos="0 1 0">
          <joint type="free" />
          <geom type="mesh" mesh="tetrahedron"/>
          <body name="middle">
            <body name="tet">
              <joint type="hinge" />
              <geom type="mesh" mesh="tetrahedron"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_PRIM_VALID(stage, "/test/root");
  EXPECT_PRIM_VALID(stage, "/test/root/middle");
  EXPECT_PRIM_VALID(stage, "/test/root/middle/tet");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestKindAuthoring) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <asset>
        <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <body name="root" pos="0 1 0">
          <joint type="free" />
          <geom type="mesh" mesh="tetrahedron"/>
          <body name="middle">
            <body name="tet">
              <joint type="hinge" />
              <geom type="mesh" mesh="tetrahedron"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_PRIM_KIND(stage, "/test", pxr::KindTokens->group);
  EXPECT_PRIM_KIND(stage, "/test/root", pxr::KindTokens->component);
  EXPECT_PRIM_KIND(stage, "/test/root/middle", pxr::KindTokens->subcomponent);
  EXPECT_PRIM_KIND(stage, "/test/root/middle/tet",
                   pxr::KindTokens->subcomponent);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestAttributesMatchSchemaTypes) {
  // TODO(robinalazard): Make the scene much more comprehensive. We ideally want
  // to test all the prims that the plugin can generate.
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <geom type="plane" name="plane_geom" size="10 20 0.1"/>
        <geom type="box" name="box_geom" size="10 20 30"/>
        <geom type="sphere" name="sphere_geom" size="10 20 30"/>
        <geom type="capsule" name="capsule_geom" size="10 20 30"/>
        <geom type="cylinder" name="cylinder_geom" size="10 20 30"/>
        <geom type="ellipsoid" name="ellipsoid_geom" size="10 20 30"/>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  for (const auto& prim : stage->TraverseAll()) {
    ExpectAllAuthoredAttributesMatchSchemaTypes(prim);
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, TestGeomsPrims) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <geom type="plane" name="plane_geom" size="10 20 0.1"/>
        <geom type="box" name="box_geom" size="10 20 30"/>
        <geom type="sphere" name="sphere_geom" size="10 20 30"/>
        <geom type="capsule" name="capsule_geom" size="10 20 30"/>
        <geom type="cylinder" name="cylinder_geom" size="10 20 30"/>
        <geom type="ellipsoid" name="ellipsoid_geom" size="10 20 30"/>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  // Note that all sizes are multiplied by 2 because Mujoco uses half sizes.

  // Plane
  EXPECT_PRIM_VALID(stage, "/test/plane_geom");
  EXPECT_PRIM_IS_A(stage, "/test/plane_geom", pxr::UsdGeomPlane);
  ExpectAttributeEqual(stage, "/test/plane_geom.width", 2 * 10.0);
  ExpectAttributeEqual(stage, "/test/plane_geom.length", 2 * 20.0);
  // Box
  EXPECT_PRIM_VALID(stage, "/test/box_geom");
  EXPECT_PRIM_IS_A(stage, "/test/box_geom", pxr::UsdGeomCube);
  // Box is a special case, it uses a UsdGeomCube and scales it with
  // xformOp:scale. The radius is always set to 2 and the extent from -1 to 1.
  ExpectAttributeEqual(stage, "/test/box_geom.size", 2.0);
  ExpectAttributeEqual(stage, "/test/box_geom.extent",
                       pxr::VtArray<pxr::GfVec3f>({{-1, -1, -1}, {1, 1, 1}}));
  ExpectAttributeEqual(stage, "/test/box_geom.xformOp:scale",
                       pxr::GfVec3f(10.0, 20.0, 30.0));
  // Sphere
  EXPECT_PRIM_VALID(stage, "/test/sphere_geom");
  EXPECT_PRIM_IS_A(stage, "/test/sphere_geom", pxr::UsdGeomSphere);
  ExpectAttributeEqual(stage, "/test/sphere_geom.radius", 10.0);
  // Capsule
  EXPECT_PRIM_VALID(stage, "/test/capsule_geom");
  EXPECT_PRIM_IS_A(stage, "/test/capsule_geom", pxr::UsdGeomCapsule);
  ExpectAttributeEqual(stage, "/test/capsule_geom.radius", 10.0);
  ExpectAttributeEqual(stage, "/test/capsule_geom.height", 2 * 20.0);
  // Cylinder
  EXPECT_PRIM_VALID(stage, "/test/cylinder_geom");
  EXPECT_PRIM_IS_A(stage, "/test/cylinder_geom", pxr::UsdGeomCylinder);
  ExpectAttributeEqual(stage, "/test/cylinder_geom.radius", 10.0);
  ExpectAttributeEqual(stage, "/test/cylinder_geom.height", 2 * 20.0);
  // Ellipsoid
  EXPECT_PRIM_VALID(stage, "/test/ellipsoid_geom");
  // Ellipsoid is a special case, it uses a UsdGeomSphere and scales it with
  // xformOp:scale. The radius is always set to 1.
  EXPECT_PRIM_IS_A(stage, "/test/ellipsoid_geom", pxr::UsdGeomSphere);
  ExpectAttributeEqual(stage, "/test/ellipsoid_geom.radius", 1.0);
  ExpectAttributeEqual(stage, "/test/ellipsoid_geom.xformOp:scale",
                       pxr::GfVec3f(10.0, 20.0, 30.0));
}

static const pxr::SdfPath kPhysicsScenePrimPath("/test/PhysicsScene");

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimTimestep) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option timestep="0.005"> </option>
    </mujoco>
  )");

  // newton:timeStepsPerSecond = round(1/0.005) = 200
  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::_tokens->newtonTimeStepsPerSecond),
                       200);
  // deprecated mjc:option:timestep should not be authored
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, kPhysicsScenePrimPath.AppendProperty(
                 MjcPhysicsTokens->mjcOptionTimestep));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimCone) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option cone="elliptic"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionCone),
      MjcPhysicsTokens->elliptic);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimWind) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option wind="1 2 3"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionWind),
      pxr::GfVec3d(1, 2, 3));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimImpratio) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option impratio="0.8"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionImpratio),
      0.8);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimMagnetic) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option magnetic="1 2 3"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionMagnetic),
      pxr::GfVec3d(1, 2, 3));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimDensity) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option density="1.2"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionDensity),
      1.2);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimViscosity) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option viscosity="0.8"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionViscosity),
                       0.8);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimO_margin) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option o_margin="0.001"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionO_margin),
      0.001);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimO_solref) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option o_solref="0.1 0.2"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionO_solref),
      pxr::VtArray<double>({0.1, 0.2}));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimO_solimp) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option o_solimp="0.1 0.2 0.3 0.4 0.5"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionO_solimp),
      pxr::VtArray<double>({0.1, 0.2, 0.3, 0.4, 0.5}));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimTolerance) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option tolerance="0.0012"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionTolerance),
                       0.0012);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimLSTolerance) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option ls_tolerance="0.0034"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionLs_tolerance),
                       0.0034);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimNoslipTolerance) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option noslip_tolerance="0.0056"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionNoslip_tolerance),
                       0.0056);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimCCDTolerance) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option ccd_tolerance="0.0078"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionCcd_tolerance),
                       0.0078);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimSleepTolerance) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option sleep_tolerance="0.0091"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionSleep_tolerance),
                       0.0091);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimOFriction) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option o_friction="0.1 0.2 0.3 0.4 0.5"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionO_friction),
                       pxr::VtArray<double>({0.1, 0.2, 0.3, 0.4, 0.5}));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimIntegrator) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option integrator="RK4"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionIntegrator),
                       MjcPhysicsTokens->rk4);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimJacobian) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option jacobian="sparse"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionJacobian),
      MjcPhysicsTokens->sparse);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimSolver) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option solver="CG"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(MjcPhysicsTokens->mjcOptionSolver),
      MjcPhysicsTokens->cg);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimIterations) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option iterations="10"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::_tokens->newtonMaxSolverIterations),
                       10);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, kPhysicsScenePrimPath.AppendProperty(
                 MjcPhysicsTokens->mjcOptionIterations));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimLSIterations) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option ls_iterations="20"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionLs_iterations),
                       20);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimNoslipIterations) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option noslip_iterations="30"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionNoslip_iterations),
                       30);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimCCDIterations) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option ccd_iterations="40"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionCcd_iterations),
                       40);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimSDFInitPoints) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option sdf_initpoints="50"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionSdf_initpoints),
                       50);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimSDFIterations) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option sdf_iterations="60"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           MjcPhysicsTokens->mjcOptionSdf_iterations),
                       60);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimGravity) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option gravity="-123 0 0"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::UsdPhysicsTokens->physicsGravityMagnitude),
                       123.0f);
  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::UsdPhysicsTokens->physicsGravityDirection),
                       pxr::GfVec3f(-1.0f, 0.0f, 0.0f));

  stage = OpenStage(R"(
    <mujoco model="test">
      <option gravity="2 3 6"> </option>
    </mujoco>
  )");

  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::UsdPhysicsTokens->physicsGravityMagnitude),
                       7.0f);
  ExpectAttributeEqual(stage,
                       kPhysicsScenePrimPath.AppendProperty(
                           pxr::UsdPhysicsTokens->physicsGravityDirection),
                       pxr::GfVec3f(0.2857143f, 0.42857143f, 0.85714287f));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimDisableFlags) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option>
        <flag
          constraint="disable"
          equality="disable"
          frictionloss="disable"
          limit="disable"
          contact="disable"
          spring="disable"
          damper="disable"
          gravity="disable"
          clampctrl="disable"
          warmstart="disable"
          filterparent="disable"
          actuation="disable"
          refsafe="disable"
          sensor="disable"
          midphase="disable"
          nativeccd="disable"
          eulerdamp="disable"
          autoreset="disable"
          island="disable"
        />
      </option>
    </mujoco>
  )");

  const std::vector<pxr::TfToken> kFlags = {
      MjcPhysicsTokens->mjcFlagConstraint,
      MjcPhysicsTokens->mjcFlagEquality,
      MjcPhysicsTokens->mjcFlagFrictionloss,
      MjcPhysicsTokens->mjcFlagLimit,
      MjcPhysicsTokens->mjcFlagContact,
      MjcPhysicsTokens->mjcFlagSpring,
      MjcPhysicsTokens->mjcFlagDamper,
      // mjc:flag:gravity is deprecated, now exported as newton:gravityEnabled
      MjcPhysicsTokens->mjcFlagClampctrl,
      MjcPhysicsTokens->mjcFlagWarmstart,
      MjcPhysicsTokens->mjcFlagFilterparent,
      MjcPhysicsTokens->mjcFlagActuation,
      MjcPhysicsTokens->mjcFlagRefsafe,
      MjcPhysicsTokens->mjcFlagSensor,
      MjcPhysicsTokens->mjcFlagMidphase,
      MjcPhysicsTokens->mjcFlagNativeccd,
      MjcPhysicsTokens->mjcFlagEulerdamp,
      MjcPhysicsTokens->mjcFlagAutoreset,
      MjcPhysicsTokens->mjcFlagIsland,
  };
  for (const auto& flag : kFlags) {
    ExpectAttributeEqual(stage, kPhysicsScenePrimPath.AppendProperty(flag),
                         false);
  }
  ExpectAttributeEqual(
      stage,
      kPhysicsScenePrimPath.AppendProperty(pxr::_tokens->newtonGravityEnabled),
      false);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsScenePrimEnableFlags) {
  auto stage = OpenStage(R"(
    <mujoco model="test">
      <option>
        <flag
          override="enable"
          energy="enable"
          fwdinv="enable"
          invdiscrete="enable"
          multiccd="enable"
          sleep="enable"
        />
      </option>
    </mujoco>
  )");

  // clang-format off
  const std::vector<pxr::TfToken> kFlags = {
      MjcPhysicsTokens->mjcFlagOverride,
      MjcPhysicsTokens->mjcFlagEnergy,
      MjcPhysicsTokens->mjcFlagFwdinv,
      MjcPhysicsTokens->mjcFlagInvdiscrete,
      MjcPhysicsTokens->mjcFlagMulticcd,
      MjcPhysicsTokens->mjcFlagSleep,
  };
  // clang-format on
  for (const auto& flag : kFlags) {
    ExpectAttributeEqual(stage, kPhysicsScenePrimPath.AppendProperty(flag),
                         true);
  }
}

static constexpr char kSiteXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <site type="box" name="box_site"/>
        <body name="ball">
          <site type="sphere" name="sphere_site" group="1"/>
          <site type="capsule" name="capsule_site" group="2"/>
          <site type="cylinder" name="cylinder_site" group="3"/>
          <site type="ellipsoid" name="ellipsoid_site" group="4"/>
          <geom type="sphere" size="1 1 1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

TEST_F(MjcfSdfFileFormatPluginTest, TestSitePrimsAuthored) {
  auto stage = OpenStage(kSiteXml);
  EXPECT_PRIM_VALID(stage, "/test/box_site");
  EXPECT_PRIM_IS_A(stage, "/test/box_site", pxr::UsdGeomCube);
  EXPECT_PRIM_API_APPLIED(stage, "/test/box_site", pxr::MjcPhysicsSiteAPI);
  EXPECT_PRIM_VALID(stage, "/test/ball/sphere_site");
  EXPECT_PRIM_IS_A(stage, "/test/ball/sphere_site", pxr::UsdGeomSphere);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ball/sphere_site",
                          pxr::MjcPhysicsSiteAPI);
  EXPECT_PRIM_VALID(stage, "/test/ball/capsule_site");
  EXPECT_PRIM_IS_A(stage, "/test/ball/capsule_site", pxr::UsdGeomCapsule);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ball/capsule_site",
                          pxr::MjcPhysicsSiteAPI);
  EXPECT_PRIM_VALID(stage, "/test/ball/cylinder_site");
  EXPECT_PRIM_IS_A(stage, "/test/ball/cylinder_site", pxr::UsdGeomCylinder);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ball/cylinder_site",
                          pxr::MjcPhysicsSiteAPI);
  EXPECT_PRIM_VALID(stage, "/test/ball/ellipsoid_site");
  EXPECT_PRIM_IS_A(stage, "/test/ball/ellipsoid_site", pxr::UsdGeomSphere);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ball/ellipsoid_site",
                          pxr::MjcPhysicsSiteAPI);

  ExpectAttributeEqual(stage, "/test/box_site.mjc:group", 0);
  ExpectAttributeEqual(stage, "/test/ball/sphere_site.mjc:group", 1);
  ExpectAttributeEqual(stage, "/test/ball/capsule_site.mjc:group", 2);
  ExpectAttributeEqual(stage, "/test/ball/cylinder_site.mjc:group", 3);
  ExpectAttributeEqual(stage, "/test/ball/ellipsoid_site.mjc:group", 4);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestSitePrimsPurpose) {
  auto stage = OpenStage(kSiteXml);
  EXPECT_PRIM_PURPOSE(stage, "/test/box_site", pxr::UsdGeomTokens->guide);
  EXPECT_PRIM_PURPOSE(stage, "/test/ball/sphere_site",
                      pxr::UsdGeomTokens->guide);
  EXPECT_PRIM_PURPOSE(stage, "/test/ball/capsule_site",
                      pxr::UsdGeomTokens->guide);
  EXPECT_PRIM_PURPOSE(stage, "/test/ball/cylinder_site",
                      pxr::UsdGeomTokens->guide);
  EXPECT_PRIM_PURPOSE(stage, "/test/ball/ellipsoid_site",
                      pxr::UsdGeomTokens->guide);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestArticulationRootAppliedOnce) {
  static constexpr char kXml[] = R"(
    <mujoco model="physics_test">
      <worldbody>
        <body name="parent" pos="0 0 0">
          <geom name="parent_geom" type="sphere" size="1"/>
          <body name="child_1" pos="1 0 0">
            <geom name="child_1_geom" type="sphere" size="1"/>
          </body>
          <body name="child_2" pos="2 0 0">
            <geom name="child_2_geom" type="sphere" size="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  pxr::SdfLayerRefPtr layer = LoadLayer(kXml);
  ;

  // This test is particular in the sense that the authoring mistake, which is
  // made on the SdfLayer level, would disappear when we access the COMPOSED
  // stage because duplicates are removed. So we need to check the SdfLayer
  // directly to see the problem.
  auto primSpec = layer->GetPrimAtPath(pxr::SdfPath("/physics_test/parent"));
  EXPECT_TRUE(primSpec);

  pxr::VtValue apiSchemasValue = primSpec->GetInfo(pxr::UsdTokens->apiSchemas);
  const pxr::SdfTokenListOp& listOp =
      apiSchemasValue.UncheckedGet<pxr::SdfTokenListOp>();
  const pxr::SdfTokenListOp::ItemVector& prependedItems =
      listOp.GetPrependedItems();

  int count = std::count(prependedItems.begin(), prependedItems.end(),
                         pxr::UsdPhysicsTokens->PhysicsArticulationRootAPI);
  EXPECT_EQ(count, 1);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsRigidBody) {
  static constexpr char kXml[] = R"(
    <mujoco model="physics_test">
      <worldbody>
        <body name="test_body" pos="0 0 0">
          <geom name="test_geom" type="sphere" size="1"/>
          <body name="test_body_2" pos="2 0 0">
            <geom name="test_geom_2" type="sphere" size="1"/>
          </body>
        </body>
        <body name="test_body_3" pos="0 0 0">
          <geom name="test_geom_3" type="sphere" size="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  EXPECT_THAT(stage, testing::NotNull());
  EXPECT_PRIM_VALID(stage, "/physics_test");
  EXPECT_PRIM_VALID(stage, "/physics_test/test_body");
  EXPECT_PRIM_VALID(stage, "/physics_test/test_body/test_body_2");

  EXPECT_PRIM_API_APPLIED(stage, "/physics_test/test_body",
                          pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/physics_test/test_body/test_body_2",
                          pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/physics_test/test_body_3",
                          pxr::UsdPhysicsRigidBodyAPI);

  // Articulation root is applied to the children of the world body.
  EXPECT_PRIM_API_APPLIED(stage, "/physics_test/test_body",
                          pxr::UsdPhysicsArticulationRootAPI);
  // test_body_3 is a child of the world but has no children so should not be
  // an articulation root.
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/physics_test/test_body_3",
                              pxr::UsdPhysicsArticulationRootAPI);

  // Articulation root is not applied to other bodies or world body.
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/physics_test",
                              pxr::UsdPhysicsArticulationRootAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/physics_test/test_body/test_body_2",
                              pxr::UsdPhysicsArticulationRootAPI);

  // Geoms should not have RigidBodyAPI applied either.
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/physics_test/test_body/test_geom",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage,
                              "/physics_test/test_body/test_body_2/test_geom_2",
                              pxr::UsdPhysicsRigidBodyAPI);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsColliders) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <asset>
        <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      </asset>
      <worldbody>
        <geom name="ground" type="plane" size="5 5 0.1"
              contype="1" conaffinity="1"/>
        <body name="body_0" pos="-3 0 2">
          <joint type="free"/>
          <geom name="body_0_col" type="sphere" size="1"
                contype="1" conaffinity="1"/>
          <body name="body_0_0" pos="-1 0 2">
            <geom name="body_0_0_col" type="sphere" size="1"
                  contype="1" conaffinity="1"/>
          </body>
        </body>
        <body name="body_1" pos="0 0 3">
          <joint type="free"/>
          <geom name="body_1_col_0" type="sphere" size="1"
                contype="1" conaffinity="1"/>
          <geom name="body_1_col_1" type="sphere" size="1" pos="-1 0 2"
                contype="1" conaffinity="1"/>
        </body>
        <body name="body_2" pos="3 0 3">
          <joint type="free"/>
          <geom name="body_2_nocol" type="sphere" size="1"
                contype="0" conaffinity="0"/>
        </body>
        <body name="body_3" pos="0 3 3">
          <joint type="free"/>
          <geom name="body_3_col" type="mesh" mesh="tetrahedron"
                contype="1" conaffinity="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);

  EXPECT_THAT(stage, testing::NotNull());
  EXPECT_PRIM_VALID(stage, "/test");

  // Expected hierarchy under /test:
  //
  // ground [collider]
  //
  // body_0 [rigidbody]
  //   body_0/body_0_col [collider]
  //
  // body_0/body_0_0 [rigidbody]
  //     body_0/body_0_0/body_0_0_col [collider]
  //
  // body_1 [rigidbody]
  //   body_1/body_1_col_0 [collider]
  //   body_1/body_1_col_1 [collider]
  //
  // body_2 [rigidbody]
  //   body_2/body_2_nocol []
  //
  // body_3 [rigidbody]
  //   body_3/body_3_col []  <-- Intermediate prim for mesh instancing
  //     body_3/body_3_col/Mesh [collider, mesh collider]

  // ground [collider] (Static collider)
  EXPECT_PRIM_VALID(stage, "/test/ground");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/ground",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ground", pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/ground", pxr::MjcPhysicsCollisionAPI);

  // body_0/body_0_0 [rigidbody] (Nested body - reparented)
  EXPECT_PRIM_VALID(stage, "/test/body_0/body_0_0");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_0/body_0_0",
                          pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_0/body_0_0",
                              pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_0/body_0_0",
                              pxr::MjcPhysicsCollisionAPI);
  //   body_0/body_0_0/body_0_0_col [collider]
  EXPECT_PRIM_VALID(stage, "/test/body_0/body_0_0/body_0_0_col");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_0/body_0_0/body_0_0_col",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_0/body_0_0/body_0_0_col",
                          pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_0/body_0_0/body_0_0_col",
                          pxr::MjcPhysicsCollisionAPI);

  // body_1 [rigidbody]
  EXPECT_PRIM_VALID(stage, "/test/body_1");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_1", pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_1",
                              pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_1",
                              pxr::MjcPhysicsCollisionAPI);
  //   body_1/body_1_col_0 [collider]
  EXPECT_PRIM_VALID(stage, "/test/body_1/body_1_col_0");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_1/body_1_col_0",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_1/body_1_col_0",
                          pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_1/body_1_col_0",
                          pxr::MjcPhysicsCollisionAPI);
  //   body_1/body_1_col_1 [collider]
  EXPECT_PRIM_VALID(stage, "/test/body_1/body_1_col_1");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_1/body_1_col_1",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_1/body_1_col_1",
                          pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_1/body_1_col_1",
                          pxr::MjcPhysicsCollisionAPI);

  // body_2 [rigidbody]
  EXPECT_PRIM_VALID(stage, "/test/body_2");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_2", pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_2",
                              pxr::UsdPhysicsCollisionAPI);
  //   body_2/body_2_nocol [] (No physics APIs applied)
  EXPECT_PRIM_VALID(stage, "/test/body_2/body_2_nocol");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_2/body_2_nocol",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_2/body_2_nocol",
                              pxr::UsdPhysicsCollisionAPI);

  // body_3 [rigidbody]
  EXPECT_PRIM_VALID(stage, "/test/body_3");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_3", pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_3",
                              pxr::UsdPhysicsCollisionAPI);
  //   body_3/body_3_col [] (Intermediate prim for mesh instancing)
  EXPECT_PRIM_VALID(stage, "/test/body_3/body_3_col");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_3/body_3_col",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_3/body_3_col",
                              pxr::UsdPhysicsCollisionAPI);
  //     body_3/body_3_col/Mesh [collider, mesh collider]
  EXPECT_PRIM_VALID(stage, "/test/body_3/body_3_col/Mesh");
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body_3/body_3_col/Mesh",
                              pxr::UsdPhysicsRigidBodyAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_3/body_3_col/Mesh",
                          pxr::UsdPhysicsCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_3/body_3_col/Mesh",
                          pxr::UsdPhysicsMeshCollisionAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body_3/body_3_col/Mesh",
                          pxr::MjcPhysicsMeshCollisionAPI);
  ExpectAttributeEqual(stage,
                       "/test/body_3/body_3_col/Mesh.physics:approximation",
                       pxr::UsdPhysicsTokens->convexHull);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsImageableAPI) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <asset>
      <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom
          name="mesh"
          type="mesh"
          mesh="tetrahedron"
          group="4"
          contype="0"
          conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_API_APPLIED(stage, "/test/body/mesh/Mesh",
                          pxr::MjcPhysicsImageableAPI);
  ExpectAttributeEqual(stage, "/test/body/mesh/Mesh.mjc:group", 4);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsCollisionAPI) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom
          name="box"
          type="box"
          size=".05 .05 .05"
          mass="0.1"
          group="4"
          priority="2"
          condim="4"
          solmix="0.5"
          solref="0.1 0.2"
          solimp="0.3 0.4 0.5 0.6 0.7"
          margin="0.8"
          gap="0.9"
          shellinertia="true"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  ExpectAttributeEqual(stage, "/test/body/box.mjc:group", 4);
  ExpectAttributeEqual(stage, "/test/body/box.mjc:priority", 2);
  ExpectAttributeEqual(stage, "/test/body/box.mjc:condim", 4);
  ExpectAttributeEqual(stage, "/test/body/box.mjc:solmix", 0.5);
  ExpectAttributeEqual(stage, "/test/body/box.mjc:solref",
                       pxr::VtArray<double>({0.1, 0.2}));
  ExpectAttributeEqual(stage, "/test/body/box.mjc:solimp",
                       pxr::VtArray<double>({0.3, 0.4, 0.5, 0.6, 0.7}));
  // margin and gap are now exported as Newton attributes
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(stage, "/test/body/box.mjc:margin");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(stage, "/test/body/box.mjc:gap");
  ExpectAttributeEqual(stage, "/test/body/box.newton:contactMargin", 0.8f);
  ExpectAttributeEqual(stage, "/test/body/box.newton:contactGap", 0.9f);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(stage,
                                         "/test/body/box.mjc:shellinertia");
  ExpectAttributeEqual(stage, "/test/body/box.newton:massModel",
                       pxr::TfToken("shell"));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsMeshCollisionAPI) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <asset>
      <mesh name="tet_legacy" inertia="legacy" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="tet_exact" inertia="exact" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="tet_convex" inertia="convex" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="tet_shell" inertia="shell" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="tet_max_vert" inertia="shell" maxhullvert="12" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom name="tet_legacy" type="mesh" mesh="tet_legacy"/>
        <geom name="tet_exact" type="mesh" mesh="tet_exact"/>
        <geom name="tet_convex" type="mesh" mesh="tet_convex"/>
        <geom name="tet_shell" type="mesh" mesh="tet_shell"/>
        <geom name="tet_max_vert" type="mesh" mesh="tet_max_vert"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  ExpectAttributeEqual(stage, "/test/body/tet_legacy/Mesh.mjc:inertia",
                       MjcPhysicsTokens->legacy);
  ExpectAttributeEqual(stage, "/test/body/tet_exact/Mesh.mjc:inertia",
                       MjcPhysicsTokens->exact);
  ExpectAttributeEqual(stage, "/test/body/tet_convex/Mesh.mjc:inertia",
                       MjcPhysicsTokens->convex);
  ExpectAttributeEqual(stage, "/test/body/tet_shell/Mesh.mjc:inertia",
                       MjcPhysicsTokens->shell);
  // mjc:maxhullvert deprecated, newton:maxHullVertices used instead
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/body/tet_max_vert/Mesh.mjc:maxhullvert");
  ExpectAttributeEqual(
      stage, "/test/body/tet_max_vert/Mesh.newton:maxHullVertices", 12);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMassAPIApplied) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" mass="0.1"/>
        <geom
          name="massless_collider"
          type="box"
          size=".05 .05 .05"
          mass="0"/>
        <geom
          name="zero_density_collider"
          type="box"
          size=".05 .05 .05"
          density="0"/>
        <geom
          name="visual"
          type="box"
          size=".05 .05 .05"
          mass="0.2"
          contype="0"
          conaffinity="0"/>
        <geom
          name="massless_visual"
          type="box"
          size=".05 .05 .05"
          mass="0"
          contype="0"
          conaffinity="0"/>
        <geom
          name="zero_density_visual"
          type="box"
          size=".05 .05 .05"
          density="0"
          contype="0"
          conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/body");
  EXPECT_PRIM_VALID(stage, "/test/body/box");
  EXPECT_PRIM_VALID(stage, "/test/body/massless_collider");
  EXPECT_PRIM_VALID(stage, "/test/body/zero_density_collider");
  EXPECT_PRIM_VALID(stage, "/test/body/visual");
  EXPECT_PRIM_VALID(stage, "/test/body/massless_visual");
  EXPECT_PRIM_VALID(stage, "/test/body/zero_density_visual");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/box", pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/massless_collider",
                          pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/zero_density_collider",
                          pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/visual", pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/massless_visual",
                          pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/zero_density_visual",
                          pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_NOT_APPLIED(stage, "/test/body", pxr::UsdPhysicsMassAPI);
  ExpectAttributeEqual(stage, "/test/body/box.physics:mass", 0.1f);
  ExpectAttributeEqual(stage, "/test/body/massless_collider.physics:density",
                       0.0f);
  ExpectAttributeEqual(
      stage, "/test/body/zero_density_collider.physics:density", 0.0f);
  ExpectAttributeEqual(stage, "/test/body/visual.physics:mass", 0.2f);
  ExpectAttributeEqual(stage, "/test/body/massless_visual.physics:density",
                       0.0f);
  ExpectAttributeEqual(stage, "/test/body/zero_density_visual.physics:density",
                       0.0f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMassAPIAppliedToBody) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <inertial pos="1 2 3" mass="3" diaginertia="1 1 1"/>
        <geom name="box" type="box" size=".05 .05 .05" mass="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/body");
  EXPECT_PRIM_VALID(stage, "/test/body/box");
  EXPECT_PRIM_API_APPLIED(stage, "/test/body/box", pxr::UsdPhysicsMassAPI);
  EXPECT_PRIM_API_APPLIED(stage, "/test/body", pxr::UsdPhysicsMassAPI);
  // Make sure that body gets it's inertial elements from the inertial element
  // and not from the subtree.
  ExpectAttributeEqual(stage, "/test/body.physics:mass", 3.0f);
  ExpectAttributeEqual(stage, "/test/body.physics:centerOfMass",
                       pxr::GfVec3f(1, 2, 3));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMassAPIDensity) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  ExpectAttributeEqual(stage, "/test/body/box.physics:density", 1234.0f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsActuator) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
        <site name="site"/>
        <site name="ref"/>
      </body>
    </worldbody>
    <actuator>
      <general
        name="general"
        group="123"
        site="site"
        refsite="ref"
        ctrllimited="true"
        ctrlrange="0 1"
        forcelimited="true"
        forcerange="2 3"
        actlimited="false"
        actrange="4 5"
        lengthrange="6 7"
        actdim="1"
        actearly="true"
        dyntype="filter"
        gaintype="user"
        biastype="user"
        gear="1 2 3 4 5 6"
        dynprm="0 1 2 3 4 5 6 7 8 9"
        gainprm="0 1 2 3 4 5 6 7 8 9"
        biasprm="0 1 2 3 4 5 6 7 8 9"
      />
    </actuator>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Actuators/general");
  EXPECT_PRIM_IS_A(stage, "/test/Actuators/general", pxr::MjcPhysicsActuator);
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:target",
                        "/test/body/site");
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:refSite",
                        "/test/body/ref");
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:group", 123);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:ctrlLimited",
                       pxr::MjcPhysicsTokens->true_);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:ctrlRange:min", 0.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:ctrlRange:max", 1.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:forceLimited",
                       pxr::MjcPhysicsTokens->true_);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:forceRange:min",
                       2.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:forceRange:max",
                       3.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:actLimited",
                       pxr::MjcPhysicsTokens->false_);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:actRange:min", 4.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:actRange:max", 5.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:lengthRange:min",
                       6.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:lengthRange:max",
                       7.0);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:actDim", 1);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:dynType",
                       MjcPhysicsTokens->filter);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:gainType",
                       MjcPhysicsTokens->user);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:biasType",
                       MjcPhysicsTokens->user);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:actEarly", true);
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:gear",
                       pxr::VtDoubleArray{{1, 2, 3, 4, 5, 6}});
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:dynPrm",
                       pxr::VtDoubleArray{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:gainPrm",
                       pxr::VtDoubleArray{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:biasPrm",
                       pxr::VtDoubleArray{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}});
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsPositionActuator) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
        <joint name="hinge" range="12 34"/>
      </body>
    </worldbody>
    <actuator>
      <position
        name="position"
        joint="hinge"
        inheritrange="1"
      />
    </actuator>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Actuators/position");
  EXPECT_PRIM_IS_A(stage, "/test/Actuators/position", pxr::MjcPhysicsActuator);
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/position.mjc:target",
                        "/test/body/hinge");
  ExpectAttributeEqual(stage, "/test/Actuators/position.mjc:inheritRange", 1.0);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsJointActuator) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="axle">
        <body name="rod">
          <joint name="rod_hinge" type="hinge"/>
          <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <general
        name="general"
        joint="rod_hinge"
      />
    </actuator>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Actuators/general");
  EXPECT_PRIM_IS_A(stage, "/test/Actuators/general", pxr::MjcPhysicsActuator);
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:target",
                        "/test/axle/rod/rod_hinge");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsBodyActuator) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
      </body>
    </worldbody>
    <actuator>
      <general
        name="general"
        body="body"
      />
    </actuator>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Actuators/general");
  EXPECT_PRIM_IS_A(stage, "/test/Actuators/general", pxr::MjcPhysicsActuator);
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:target",
                        "/test/body");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsSliderCrankActuator) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="body">
        <geom name="box" type="box" size=".05 .05 .05" density="1234"/>
        <site name="crank"/>
        <site name="slider"/>
      </body>
    </worldbody>
    <actuator>
      <general
        name="general"
        cranksite="crank"
        slidersite="slider"
        cranklength="1.23"
      />
    </actuator>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Actuators/general");
  EXPECT_PRIM_IS_A(stage, "/test/Actuators/general", pxr::MjcPhysicsActuator);
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:target",
                        "/test/body/crank");
  EXPECT_REL_HAS_TARGET(stage, "/test/Actuators/general.mjc:sliderSite",
                        "/test/body/slider");
  ExpectAttributeEqual(stage, "/test/Actuators/general.mjc:crankLength", 1.23);
}

TEST_F(MjcfSdfFileFormatPluginTest, NewtonDampingUnits) {
  for (const char* angle : {"degree", "radian"}) {
    for (const char* type : {"hinge", "slide"}) {
      SCOPED_TRACE(angle);
      SCOPED_TRACE(type);
      const std::string xml =
          std::string("<mujoco model='test'><compiler angle='") + angle +
          "'/><worldbody><body name='body'><joint name='joint' type='" + type +
          "' damping='0.7'/><geom type='sphere' size='1'/></body>"
          "</worldbody></mujoco>";
      auto stage = OpenStage(xml);
      const float expected =
          std::string(type) == "hinge" ? 0.7 * std::numbers::pi / 180.0 : 0.7;
      ExpectAttributeEqual(stage, "/test/body/joint.newton:damping", expected);
      EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(stage,
                                             "/test/body/joint.mjc:damping");
    }
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsJointAPI) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <worldbody>
      <body name="parent">
        <body name="child">
          <joint name="my_joint" type="hinge"
            group="4"
            springdamper="1 2"
            solreflimit="0.1 0.2"
            solimplimit="0.3 0.4 0.5 0.6 0.7"
            solreffriction="0.8 0.9"
            solimpfriction="1.0 1.1 1.2 1.3 1.4"
            stiffness="1.5"
            actuatorfrcrange="-1.6 1.7"
            actuatorfrclimited="true"
            actuatorgravcomp="true"
            margin="1.8"
            ref="1.9"
            springref="2.0"
            armature="2.1"
            damping="2.2"
            frictionloss="2.3"
          />
          <geom type="sphere" size="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto stage = OpenStage(xml);

  const SdfPath joint_path("/test/parent/child/my_joint");
  EXPECT_PRIM_API_APPLIED(stage, joint_path, pxr::MjcPhysicsJointAPI);

  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:group", 4);
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:springdamper",
                       pxr::VtArray<double>({1, 2}));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:solreflimit",
                       pxr::VtArray<double>({0.1, 0.2}));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:solimplimit",
                       pxr::VtArray<double>({0.3, 0.4, 0.5, 0.6, 0.7}));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:solreffriction",
                       pxr::VtArray<double>({0.8, 0.9}));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:solimpfriction",
                       pxr::VtArray<double>({1.0, 1.1, 1.2, 1.3, 1.4}));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:stiffness", 1.5);
  ExpectAttributeEqual(
      stage, "/test/parent/child/my_joint.mjc:actuatorfrcrange:min", -1.6);
  ExpectAttributeEqual(
      stage, "/test/parent/child/my_joint.mjc:actuatorfrcrange:max", 1.7);
  ExpectAttributeEqual(stage,
                       "/test/parent/child/my_joint.mjc:actuatorfrclimited",
                       MjcPhysicsTokens->true_);
  ExpectAttributeEqual(
      stage, "/test/parent/child/my_joint.mjc:actuatorgravcomp", true);
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:margin", 1.8);
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:ref", 1.9);
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.mjc:springref", 2.0);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child/my_joint.mjc:armature");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child/my_joint.mjc:damping");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child/my_joint.mjc:frictionloss");

  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.newton:armature",
                       2.1f);
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.newton:damping",
                       static_cast<float>(2.2 * std::numbers::pi / 180.0));
  ExpectAttributeEqual(stage, "/test/parent/child/my_joint.newton:friction",
                       2.3f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsFloatingAndFixedBaseBody) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="fixed_base">
          <geom type="sphere" size="1"/>
        </body>
        <body name="floating_base">
          <joint type="free"/>
          <geom type="sphere" size="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  // Test that the fixed_base body has a UsdPhysicsJoint child connected to the
  // worldbody.
  EXPECT_PRIM_VALID(stage, "/test/fixed_base/FixedJoint");
  auto joint = pxr::UsdPhysicsFixedJoint::Get(
      stage, SdfPath("/test/fixed_base/FixedJoint"));
  ASSERT_TRUE(joint);

  // Initial joint to the worldbody does't set a body0 rel.
  EXPECT_REL_TARGET_COUNT(stage, "/test/fixed_base/FixedJoint.physics:body0",
                          0);
  EXPECT_REL_HAS_TARGET(stage, "/test/fixed_base/FixedJoint.physics:body1",
                        "/test/fixed_base");

  // Test that the floating_base body has no UsdPhysicsJoint children.
  auto floating_base = stage->GetPrimAtPath(SdfPath("/test/floating_base"));
  ASSERT_TRUE(floating_base);
  for (const auto& child : floating_base.GetChildren()) {
    EXPECT_FALSE(child.IsA<pxr::UsdPhysicsJoint>());
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsFixedJoint) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="parent">
          <geom type="sphere" size="1"/>
          <body name="child" pos="1 0 0">
            <geom type="sphere" size="1"/>
            <body name="grandchild" pos="1 0 0">
              <geom type="sphere" size="1"/>
            </body>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_IS_A(stage, "/test/parent/FixedJoint", pxr::UsdPhysicsFixedJoint);
  // Initial joint to the worldbody does't set a body0 rel.
  EXPECT_REL_TARGET_COUNT(stage, "/test/parent/FixedJoint.physics:body0", 0);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/FixedJoint.physics:body1",
                        "/test/parent");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child/FixedJoint",
                   pxr::UsdPhysicsFixedJoint);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child/FixedJoint.physics:body0",
                        "/test/parent");
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child/FixedJoint.physics:body1",
                        "/test/parent/child");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child/grandchild/FixedJoint",
                   pxr::UsdPhysicsFixedJoint);
  EXPECT_REL_HAS_TARGET(
      stage, "/test/parent/child/grandchild/FixedJoint.physics:body0",
      "/test/parent/child");
  EXPECT_REL_HAS_TARGET(
      stage, "/test/parent/child/grandchild/FixedJoint.physics:body1",
      "/test/parent/child/grandchild");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsRevoluteJoint) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="parent">
          <joint name="hinge_root"/>
          <geom type="sphere" size="1"/>
          <body name="child0" pos="1 0 0">
            <joint name="hinge_normal" type="hinge" axis="0 0 1"/>
            <geom type="sphere" size="1"/>
          </body>
          <body name="child1" pos="1 0 0">
            <joint name="hinge_limited" type="hinge" axis="0 0 1" limited="true" range="-30 45"/>
            <geom type="sphere" size="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  // hinge_root doesn't set a type so it's the default: a revolute joint.
  EXPECT_PRIM_IS_A(stage, "/test/parent/hinge_root",
                   pxr::UsdPhysicsRevoluteJoint);
  // Initial joint to the worldbody does't set a body0 rel.
  EXPECT_REL_TARGET_COUNT(stage, "/test/parent/hinge_root.physics:body0", 0);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/hinge_root.physics:body1",
                        "/test/parent");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child0/hinge_normal",
                   pxr::UsdPhysicsRevoluteJoint);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child0/hinge_normal.physics:body0",
                        "/test/parent");
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child0/hinge_normal.physics:body1",
                        "/test/parent/child0");
  ExpectAttributeEqual(stage, "/test/parent/child0/hinge_normal.physics:axis",
                       pxr::UsdPhysicsTokens->z);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child0/hinge_normal.physics:lowerLimit");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child0/hinge_normal.physics:upperLimit");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child1/hinge_limited",
                   pxr::UsdPhysicsRevoluteJoint);
  EXPECT_REL_HAS_TARGET(
      stage, "/test/parent/child1/hinge_limited.physics:body0", "/test/parent");
  EXPECT_REL_HAS_TARGET(stage,
                        "/test/parent/child1/hinge_limited.physics:body1",
                        "/test/parent/child1");
  ExpectAttributeEqual(stage, "/test/parent/child1/hinge_limited.physics:axis",
                       pxr::UsdPhysicsTokens->z);
  ExpectAttributeEqual(
      stage, "/test/parent/child1/hinge_limited.physics:lowerLimit", -30.0f);
  ExpectAttributeEqual(
      stage, "/test/parent/child1/hinge_limited.physics:upperLimit", 45.0f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsPrismaticJoint) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="parent">
          <joint name="slide_root" type="slide"/>
          <geom type="sphere" size="1"/>
          <body name="child0" pos="1 0 0">
            <joint name="slide_normal" type="slide" axis="1 0 0"/>
            <geom type="sphere" size="1"/>
          </body>
          <body name="child1" pos="1 0 0">
            <joint name="slide_limited" type="slide" axis="1 0 0" limited="true" range="-2.5 2.5"/>
            <geom type="sphere" size="1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_IS_A(stage, "/test/parent/slide_root",
                   pxr::UsdPhysicsPrismaticJoint);
  // Initial joint to the worldbody does't set a body0 rel.
  EXPECT_REL_TARGET_COUNT(stage, "/test/parent/slide_root.physics:body0", 0);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/slide_root.physics:body1",
                        "/test/parent");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child0/slide_normal",
                   pxr::UsdPhysicsPrismaticJoint);
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child0/slide_normal.physics:body0",
                        "/test/parent");
  EXPECT_REL_HAS_TARGET(stage, "/test/parent/child0/slide_normal.physics:body1",
                        "/test/parent/child0");
  ExpectAttributeEqual(stage, "/test/parent/child0/slide_normal.physics:axis",
                       pxr::UsdPhysicsTokens->z);
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child0/slide_normal.physics:lowerLimit");
  EXPECT_ATTRIBUTE_HAS_NO_AUTHORED_VALUE(
      stage, "/test/parent/child0/slide_normal.physics:upperLimit");

  EXPECT_PRIM_IS_A(stage, "/test/parent/child1/slide_limited",
                   pxr::UsdPhysicsPrismaticJoint);
  EXPECT_REL_HAS_TARGET(
      stage, "/test/parent/child1/slide_limited.physics:body0", "/test/parent");
  EXPECT_REL_HAS_TARGET(stage,
                        "/test/parent/child1/slide_limited.physics:body1",
                        "/test/parent/child1");
  ExpectAttributeEqual(stage, "/test/parent/child1/slide_limited.physics:axis",
                       pxr::UsdPhysicsTokens->z);
  ExpectAttributeEqual(
      stage, "/test/parent/child1/slide_limited.physics:lowerLimit", -2.5f);
  ExpectAttributeEqual(
      stage, "/test/parent/child1/slide_limited.physics:upperLimit", 2.5f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestRadianAnglesAreConvertedToDegrees) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <compiler angle="radian"/>

      <worldbody>
        <body name="parent">
          <joint name="hinge" type="hinge" axis="0 0 1" limited="true" range="-3.14159265359 0.78539816339"/>
          <geom type="sphere" size="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_VALID(stage, "/test/parent/hinge");
  ExpectAttributeEqual(stage, "/test/parent/hinge.physics:lowerLimit", -180.0f);
  ExpectAttributeEqual(stage, "/test/parent/hinge.physics:upperLimit", 45.0f);
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsJointFrames) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="parent" pos="0 1 0">
          <body name="child0" pos="1 0 0">
            <joint name="hinge" type="hinge" pos="0.1 0.2 0.3" axis="0 1 0"/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body name="child1" pos="2 3 4">
            <joint name="slide" type="slide" pos="0.4 0.5 0.6" axis="-1 0 0"/>
            <geom type="sphere" size="0.1"/>
          </body>
          <body name="child2" pos="5 6 7">
            <joint name="slide_nonaxis" type="slide" pos="0.7 0.8 0.9" axis="1 1 1"/>
            <geom type="sphere" size="0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  // Test the hinge joint.
  EXPECT_PRIM_VALID(stage, "/test/parent/child0/hinge");
  auto hinge_joint = pxr::UsdPhysicsRevoluteJoint::Get(
      stage, SdfPath("/test/parent/child0/hinge"));
  ASSERT_TRUE(hinge_joint);

  ExpectAttributeEqual(stage, "/test/parent/child0/hinge.physics:localPos0",
                       pxr::GfVec3f(1.1, 0.2, 0.3));

  pxr::GfRotation hinge_rot;
  hinge_rot.SetRotateInto({0, 0, 1}, {0, 1, 0});
  pxr::GfQuatf expected_hinge_rot(hinge_rot.GetQuat());

  pxr::GfQuatf hinge_local_rot0;
  hinge_joint.GetLocalRot0Attr().Get(&hinge_local_rot0);
  EXPECT_TRUE(AreQuatsSameRotation(expected_hinge_rot, hinge_local_rot0));

  ExpectAttributeEqual(stage, "/test/parent/child0/hinge.physics:localPos1",
                       pxr::GfVec3f(0.1, 0.2, 0.3));

  pxr::GfQuatf hinge_local_rot1;
  hinge_joint.GetLocalRot1Attr().Get(&hinge_local_rot1);
  EXPECT_TRUE(AreQuatsSameRotation(expected_hinge_rot, hinge_local_rot1));

  // Test the slide joint.
  EXPECT_PRIM_VALID(stage, "/test/parent/child1/slide");
  auto slide_joint = pxr::UsdPhysicsPrismaticJoint::Get(
      stage, SdfPath("/test/parent/child1/slide"));
  ASSERT_TRUE(slide_joint);

  ExpectAttributeEqual(stage, "/test/parent/child1/slide.physics:localPos0",
                       pxr::GfVec3f(2.4, 3.5, 4.6));

  pxr::GfRotation slide_rot;
  slide_rot.SetRotateInto({0, 0, 1}, {-1, 0, 0});
  pxr::GfQuatf expected_slide_rot(slide_rot.GetQuat());

  pxr::GfQuatf slide_local_rot0;
  slide_joint.GetLocalRot0Attr().Get(&slide_local_rot0);
  EXPECT_TRUE(AreQuatsSameRotation(expected_slide_rot, slide_local_rot0));

  ExpectAttributeEqual(stage, "/test/parent/child1/slide.physics:localPos1",
                       pxr::GfVec3f(0.4, 0.5, 0.6));

  pxr::GfQuatf slide_local_rot1;
  slide_joint.GetLocalRot1Attr().Get(&slide_local_rot1);
  EXPECT_TRUE(AreQuatsSameRotation(expected_slide_rot, slide_local_rot1));

  // Test the slide_nonaxis joint.
  EXPECT_PRIM_VALID(stage, "/test/parent/child2/slide_nonaxis");
  auto slide_nonaxis_joint = pxr::UsdPhysicsPrismaticJoint::Get(
      stage, SdfPath("/test/parent/child2/slide_nonaxis"));
  ASSERT_TRUE(slide_nonaxis_joint);

  ExpectAttributeEqual(stage,
                       "/test/parent/child2/slide_nonaxis.physics:localPos0",
                       pxr::GfVec3f(5.7, 6.8, 7.9));

  pxr::GfRotation slide_nonaxis_rot;
  slide_nonaxis_rot.SetRotateInto({0, 0, 1}, {1, 1, 1});
  pxr::GfQuatf expected_slide_nonaxis_rot(slide_nonaxis_rot.GetQuat());

  pxr::GfQuatf slide_nonaxis_local_rot0;
  slide_nonaxis_joint.GetLocalRot0Attr().Get(&slide_nonaxis_local_rot0);
  EXPECT_TRUE(AreQuatsSameRotation(expected_slide_nonaxis_rot,
                                   slide_nonaxis_local_rot0));

  ExpectAttributeEqual(stage,
                       "/test/parent/child2/slide_nonaxis.physics:localPos1",
                       pxr::GfVec3f(0.7, 0.8, 0.9));

  pxr::GfQuatf slide_nonaxis_local_rot1;
  slide_nonaxis_joint.GetLocalRot1Attr().Get(&slide_nonaxis_local_rot1);
  EXPECT_TRUE(AreQuatsSameRotation(expected_slide_nonaxis_rot,
                                   slide_nonaxis_local_rot1));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestPhysicsUnsupportedJoint) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <worldbody>
        <body name="parent">
          <joint type="ball" name="ball_joint"/>
          <geom type="sphere" size="1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  EXPECT_THAT(stage, testing::NotNull());

  EXPECT_PRIM_INVALID(stage, "/test/parent/ball_joint");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestMjcPhysicsKeyframe) {
  static constexpr char xml[] = R"(
  <mujoco model="test">
      <worldbody>
        <frame name="frame"/>
        <body name="body">
          <joint/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="home" qpos="1"/>
        <key time="1" qpos="2"/>
        <key time="2" qpos="3"/>
      </keyframe>
    </mujoco>)";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/Keyframes/home");
  EXPECT_PRIM_VALID(stage, "/test/Keyframes/Keyframe");
  ExpectAttributeEqual(stage, "/test/Keyframes/home.mjc:qpos",
                       pxr::VtDoubleArray({1}));

  // Check time samples are correctly authored.
  ExpectAttributeEqual(stage, "/test/Keyframes/Keyframe.mjc:qpos",
                       pxr::VtDoubleArray({2}), pxr::UsdTimeCode(1.0));

  ExpectAttributeEqual(stage, "/test/Keyframes/Keyframe.mjc:qpos",
                       pxr::VtDoubleArray({3}), pxr::UsdTimeCode(2.0));
}

TEST_F(MjcfSdfFileFormatPluginTest, TestCompilerOptions) {
  // settotalmass is deprecated
  mock_warning_handler.ExpectWarnings("settotalmass");

  static constexpr char xml[] = R"(
<mujoco model="test">
  <compiler
    autolimits="true"
    boundmass="1.2"
    boundinertia="3.4"
    settotalmass="5.6"
    usethread="false"
    balanceinertia="true"
    angle="radian"
    fitaabb="true"
    fusestatic="true"
    inertiafromgeom="true"
    alignfree="true"
    inertiagrouprange="1 6"
    saveinertial="true"
  />
</mujoco>
)";
  auto stage = OpenStage(xml);

  EXPECT_PRIM_VALID(stage, "/test/PhysicsScene");

  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:autoLimits",
                       true);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:boundMass", 1.2);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:boundInertia",
                       3.4);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:setTotalMass",
                       5.6);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:useThread",
                       false);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:balanceInertia",
                       true);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:angle",
                       MjcPhysicsTokens->radian);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:fitAABB", true);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:fuseStatic",
                       true);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:inertiaFromGeom",
                       MjcPhysicsTokens->true_);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:alignFree",
                       true);
  ExpectAttributeEqual(
      stage, "/test/PhysicsScene.mjc:compiler:inertiaGroupRange:min", 1);
  ExpectAttributeEqual(
      stage, "/test/PhysicsScene.mjc:compiler:inertiaGroupRange:max", 6);
  ExpectAttributeEqual(stage, "/test/PhysicsScene.mjc:compiler:saveInertial",
                       true);
}

TEST_F(MjcfSdfFileFormatPluginTest, LightDirectionAndTransform) {
  static constexpr char kXml[] = R"(
    <mujoco model="light_test">
      <worldbody>
        <light name="light_x" pos="1 2 3" dir="1 0 0"/>
        <light name="light_y" pos="4 5 6" dir="0 1 0"/>
        <light name="light_neg_z" pos="7 8 9" dir="0 0 -1"/>
        <light name="light_pos_z" pos="-1 -2 -3" dir="0 0 1"/>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  ASSERT_THAT(stage, testing::NotNull());

  struct LightExpectation {
    const char* prim_path;
    pxr::GfVec3d pos;
    pxr::GfVec3d dir;
  };
  const LightExpectation kExpected[] = {
      {"/light_test/light_x", {1, 2, 3}, {1, 0, 0}},
      {"/light_test/light_y", {4, 5, 6}, {0, 1, 0}},
      {"/light_test/light_neg_z", {7, 8, 9}, {0, 0, -1}},
      {"/light_test/light_pos_z", {-1, -2, -3}, {0, 0, 1}},
  };

  for (const auto& expected : kExpected) {
    SCOPED_TRACE(expected.prim_path);
    EXPECT_PRIM_VALID(stage, expected.prim_path);
    auto attr = stage->GetAttributeAtPath(
        pxr::SdfPath(expected.prim_path)
            .AppendProperty(pxr::TfToken("xformOp:transform")));
    ASSERT_TRUE(attr.IsValid());
    pxr::GfMatrix4d xf;
    ASSERT_TRUE(attr.Get(&xf));

    pxr::GfVec3d trans = xf.ExtractTranslation();
    EXPECT_NEAR(trans[0], expected.pos[0], 1e-5);
    EXPECT_NEAR(trans[1], expected.pos[1], 1e-5);
    EXPECT_NEAR(trans[2], expected.pos[2], 1e-5);

    pxr::GfVec3d world_dir = xf.TransformDir(pxr::GfVec3d(0, 0, -1));
    EXPECT_NEAR(world_dir[0], expected.dir[0], 1e-5);
    EXPECT_NEAR(world_dir[1], expected.dir[1], 1e-5);
    EXPECT_NEAR(world_dir[2], expected.dir[2], 1e-5);
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, BoxAndEllipsoidPreserveScaleXformOpOrder) {
  static constexpr char kXml[] = R"(
    <mujoco model="scale_test">
      <compiler angle="degree"/>
      <worldbody>
        <geom name="box_geom" type="box" pos="1 2 3" euler="0 0 90"
              size="2 3 4"/>
        <geom name="ellipsoid_geom" type="ellipsoid" pos="4 5 6" euler="0 0 90"
              size="0.2 0.3 0.4"/>
        <body name="b">
          <site name="box_site" type="box" pos="-1 0 2" euler="0 0 90"
                size="0.5 1.5 2.5"/>
          <site name="ellipsoid_site" type="ellipsoid" pos="0 -2 1"
                euler="0 0 90" size="0.1 0.2 0.3"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  ASSERT_THAT(stage, testing::NotNull());

  const pxr::VtTokenArray kExpectedOpOrder(
      {pxr::TfToken("xformOp:transform"), pxr::TfToken("xformOp:scale")});

  struct ScaledPrimExpectation {
    const char* prim_path;
    pxr::GfVec3d world_corner;
  };
  const ScaledPrimExpectation kExpected[] = {
      {"/scale_test/box_geom", {-2.0, 4.0, 7.0}},
      {"/scale_test/ellipsoid_geom", {3.7, 5.2, 6.4}},
      {"/scale_test/b/box_site", {-2.5, 0.5, 4.5}},
      {"/scale_test/b/ellipsoid_site", {-0.2, -1.9, 1.3}},
  };

  pxr::UsdGeomXformCache xform_cache;
  for (const auto& expected : kExpected) {
    SCOPED_TRACE(expected.prim_path);
    pxr::SdfPath prim_path(expected.prim_path);
    ExpectAttributeEqual(
        stage, prim_path.AppendProperty(pxr::UsdGeomTokens->xformOpOrder),
        kExpectedOpOrder);

    pxr::UsdPrim prim = stage->GetPrimAtPath(prim_path);
    ASSERT_TRUE(prim.IsValid());
    pxr::GfMatrix4d world_xf = xform_cache.GetLocalToWorldTransform(prim);
    pxr::GfVec3d pt = world_xf.Transform(pxr::GfVec3d(1.0, 1.0, 1.0));
    EXPECT_NEAR(pt[0], expected.world_corner[0], 1e-5);
    EXPECT_NEAR(pt[1], expected.world_corner[1], 1e-5);
    EXPECT_NEAR(pt[2], expected.world_corner[2], 1e-5);
  }
}

TEST_F(MjcfSdfFileFormatPluginTest, CreateRelationshipSpecDuplicate) {
  auto layer = pxr::SdfLayer::CreateAnonymous();
  auto prim_spec = pxr::SdfCreatePrimInLayer(layer, pxr::SdfPath("/World"));
  auto rel_name = pxr::TfToken("my_relationship");
  auto target1 = pxr::SdfPath("/World/Target1");
  auto target2 = pxr::SdfPath("/World/Target2");

  auto rel1 = CreateRelationshipSpec(layer, prim_spec, rel_name, target1);
  ASSERT_TRUE(rel1);
  EXPECT_THAT(rel1->GetTargetPathList().GetAppendedItems(),
              testing::ElementsAre(target1));

  auto rel2 = CreateRelationshipSpec(layer, prim_spec, rel_name, target2);
  ASSERT_TRUE(rel2);
  EXPECT_THAT(rel2->GetTargetPathList().GetAppendedItems(),
              testing::ElementsAre(target1, target2));
}

TEST_F(MjcfSdfFileFormatPluginTest,
       GeomWithCustomFrictionAndMaterialAppendsMaterialBindings) {
  static constexpr char kXml[] = R"(
    <mujoco model="mat_friction_test">
      <asset>
        <material name="mat" rgba="1 0 0 1"/>
      </asset>
      <worldbody>
        <geom name="g" type="sphere" size="0.1" friction="0.8 0.01 0.001"
              material="mat"/>
      </worldbody>
    </mujoco>
  )";

  auto stage = OpenStage(kXml);
  ASSERT_THAT(stage, testing::NotNull());
  EXPECT_PRIM_VALID(stage, "/mat_friction_test/g");
  EXPECT_REL_TARGET_COUNT(stage, "/mat_friction_test/g.material:binding", 2);
  EXPECT_REL_HAS_TARGET(stage, "/mat_friction_test/g.material:binding",
                        "/mat_friction_test/Materials/mat");
}

TEST_F(MjcfSdfFileFormatPluginTest, TestActuatorGroupDisableOption) {
  static constexpr char kXml[] = R"(
    <mujoco model="test">
      <option actuatorgroupdisable="0 2"/>
    </mujoco>
  )";
  auto stage = OpenStage(kXml);
  EXPECT_PRIM_VALID(stage, "/test/PhysicsScene");
  ExpectAttributeEqual(stage,
                       "/test/PhysicsScene.mjc:option:actuatorgroupdisable",
                       pxr::VtIntArray({0, 2}));
}

}  // namespace usd
}  // namespace mujoco
