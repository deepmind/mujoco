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

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <ios>
#include <memory>
#include <numbers>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <mujoco/experimental/usd/utils.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"
#include <pxr/base/arch/fileSystem.h>
#include <pxr/base/tf/token.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/common.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usd/stage.h>

namespace mujoco {
namespace {

// Ensure mujoco_usd is not dead-stripped by the linker so its static
// initializer registers USD plugins (mjcPhysics, newton, usdMjcf) at load time.
[[maybe_unused]] const pxr::SdfPath kForceLoadMujocoUsd =
    mujoco::usd::GetUsdPrimPathUserValue(nullptr);

// Joint kind, compiler angle units, and damping source (none/Newton/MJC/both).
class UsdJointDampingTest
    : public MujocoTest,
      public testing::WithParamInterface<std::tuple<bool, bool, int>> {
 public:
  static void SetUpTestSuite() {
#ifdef USD_DECODER_PLUGIN_PATH
    mj_loadPluginLibrary(USD_DECODER_PLUGIN_PATH);
#endif
  }

 protected:
  void SetUp() override {
    directory_ = pxr::ArchMakeTmpSubdir(
        std::filesystem::temp_directory_path().string(), "usd_damping");
    ASSERT_FALSE(directory_.empty());
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  std::string directory_;
};

TEST_P(UsdJointDampingTest, DampingUnitsAndLegacyPrecedence) {
  const auto [angular, degrees, source] = GetParam();
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Z"
)
def PhysicsScene "scene" (
    prepend apiSchemas = ["MjcSceneAPI"]
) {
    uniform token mjc:compiler:angle = "degree"
}
def Xform "body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
) {
    def Sphere "geom" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    ) {
        double radius = 1
    }
    def PhysicsRevoluteJoint "joint" (
        prepend apiSchemas = ["MjcJointAPI"]
    ) {
        rel physics:body1 = </body>
        float newton:damping
        double mjc:damping
    }
}
)"));
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/body/joint"));
  if (!angular) {
    ASSERT_TRUE(joint.SetTypeName(pxr::TfToken("PhysicsPrismaticJoint")));
  }
  ASSERT_TRUE(
      stage->GetAttributeAtPath(pxr::SdfPath("/scene.mjc:compiler:angle"))
          .Set(pxr::TfToken(degrees ? "degree" : "radian")));
  if (source == 1 || source == 3) {
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("newton:damping")).Set(0.25f));
  }
  if (source == 2 || source == 3) {
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("mjc:damping")).Set(0.7));
    mock_warning_handler.ExpectWarnings("deprecated mjc:damping");
  }
  const std::string path = directory_ + "/joint.usda";
  ASSERT_TRUE(stage->GetRootLayer()->Export(path));
  char error[1024] = {};
  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
      mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
      mj_deleteSpec);
  ASSERT_NE(spec, nullptr) << error;
  EXPECT_EQ(spec->compiler.degree, degrees);
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_compile(spec.get(), nullptr), mj_deleteModel);
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->njnt, 1);
  EXPECT_EQ(model->jnt_type[0], angular ? mjJNT_HINGE : mjJNT_SLIDE);
  double expected = 0;
  if (source == 1) {
    expected = angular ? 0.25 * 180.0 / std::numbers::pi : 0.25;
  } else if (source >= 2) {
    expected = 0.7;
  }
  EXPECT_NEAR(model->dof_damping[0], expected, 1e-6);
}

INSTANTIATE_TEST_SUITE_P(JointDamping, UsdJointDampingTest,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Values(0, 1, 2, 3)));

class UsdMassModelTest : public MujocoTest {
 public:
  static void SetUpTestSuite() {
#ifdef USD_DECODER_PLUGIN_PATH
    mj_loadPluginLibrary(USD_DECODER_PLUGIN_PATH);
#endif
  }

 protected:
  void SetUp() override {
    directory_ = pxr::ArchMakeTmpSubdir(
        std::filesystem::temp_directory_path().string(), "usd_inertia");
    ASSERT_FALSE(directory_.empty());
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  // Unit density makes the compiled mass equal volume or surface area.
  pxr::UsdStageRefPtr MakeStage(bool mesh, bool collider, bool mjc_api,
                                bool shell) {
    auto stage = pxr::UsdStage::CreateInMemory();
    EXPECT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Mesh "geom" (
        prepend apiSchemas = ["NewtonMassAPI", "PhysicsMassAPI", "PhysicsMeshCollisionAPI"]
    ) {
        point3f[] points = [(0,0,0),(1,0,0),(0,1,0),(0,0,1)]
        int[] faceVertexCounts = [3,3,3,3]
        int[] faceVertexIndices = [0,2,1,0,1,3,0,3,2,1,2,3]
        uniform token physics:approximation = "convexHull"
        uniform token newton:massModel = "solid"
        float physics:density = 1
        double radius = 1
        token mjc:inertia
    }
}
)"));
    auto geom = stage->GetPrimAtPath(pxr::SdfPath("/body/geom"));
    if (!mesh) {
      EXPECT_TRUE(geom.SetTypeName(pxr::TfToken("Sphere")));
    }
    if (collider) {
      EXPECT_TRUE(geom.ApplyAPI(pxr::TfToken("PhysicsCollisionAPI")));
    }
    if (mjc_api) {
      EXPECT_TRUE(geom.ApplyAPI(
          pxr::TfToken(mesh ? "MjcMeshCollisionAPI" : "MjcCollisionAPI")));
    }
    EXPECT_TRUE(geom.GetAttribute(pxr::TfToken("newton:massModel"))
                    .Set(pxr::TfToken(shell ? "shell" : "solid")));
    return stage;
  }

  void CheckStage(const pxr::UsdStageRefPtr& stage, bool mesh, bool shell,
                  mjtMeshInertia expected_inertia) {
    const std::string path = directory_ + "/geom.usda";
    ASSERT_TRUE(stage->GetRootLayer()->Export(path));
    char error[1024] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
        mj_deleteSpec);
    ASSERT_NE(spec, nullptr) << error;
    auto* geom = mjs_asGeom(mjs_firstElement(spec.get(), mjOBJ_GEOM));
    ASSERT_NE(geom, nullptr);
    // Visual density parsing is separate from mass-model parsing; make the
    // reference density explicit here for both visual and collision geoms.
    geom->density = 1;
    EXPECT_EQ(geom->typeinertia,
              !mesh && shell ? mjINERTIA_SHELL : mjINERTIA_VOLUME);
    if (mesh) {
      auto* asset = mjs_asMesh(mjs_firstElement(spec.get(), mjOBJ_MESH));
      ASSERT_NE(asset, nullptr);
      EXPECT_EQ(asset->inertia, expected_inertia);
    }
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_compile(spec.get(), nullptr), mj_deleteModel);
    ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
    ASSERT_EQ(model->nbody, 2);
    const double expected_mass =
        mesh ? (shell ? (3 + std::sqrt(3.0)) / 2 : 1.0 / 6)
             : (shell ? 4 * std::numbers::pi : 4 * std::numbers::pi / 3);
    EXPECT_NEAR(model->body_mass[1], expected_mass, 1e-6);
  }

  std::string directory_;
};

// Mesh/primitive, collider/visual, MJC API present/absent, shell/solid.
class UsdNewtonMassModelTest
    : public UsdMassModelTest,
      public testing::WithParamInterface<std::tuple<bool, bool, bool, bool>> {};

TEST_P(UsdNewtonMassModelTest, InertiaLivesOnMeshAssetOrPrimitiveGeom) {
  const auto [mesh, collider, mjc_api, shell] = GetParam();
  auto stage = MakeStage(mesh, collider, mjc_api, shell);
  CheckStage(stage, mesh, shell,
             shell ? mjMESH_INERTIA_SHELL : mjMESH_INERTIA_CONVEX);
}

INSTANTIATE_TEST_SUITE_P(MassModel, UsdNewtonMassModelTest,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Bool(), testing::Bool()));

TEST_F(UsdMassModelTest, DeprecatedMeshInertiaOverridesConvexApproximation) {
  for (bool collider : {false, true}) {
    for (const char* inertia : {"exact", "convex", "legacy", "shell"}) {
      SCOPED_TRACE(collider);
      SCOPED_TRACE(inertia);
      auto stage = MakeStage(true, collider, true, false);
      ASSERT_TRUE(
          stage->GetAttributeAtPath(pxr::SdfPath("/body/geom.mjc:inertia"))
              .Set(pxr::TfToken(inertia)));
      mock_warning_handler.ExpectWarnings("deprecated mjc:inertia");
      mjtMeshInertia expected_inertia = mjMESH_INERTIA_LEGACY;
      if (std::string(inertia) == "exact") {
        expected_inertia = mjMESH_INERTIA_EXACT;
      } else if (std::string(inertia) == "convex") {
        expected_inertia = mjMESH_INERTIA_CONVEX;
      } else if (std::string(inertia) == "shell") {
        expected_inertia = mjMESH_INERTIA_SHELL;
      }
      CheckStage(stage, true, std::string(inertia) == "shell",
                 expected_inertia);
    }
  }
}

class UsdMimicTest : public MujocoTest {
 public:
  static void SetUpTestSuite() {
#ifdef USD_DECODER_PLUGIN_PATH
    mj_loadPluginLibrary(USD_DECODER_PLUGIN_PATH);
#endif
  }

 protected:
  void SetUp() override {
    directory_ = pxr::ArchMakeTmpSubdir(
        std::filesystem::temp_directory_path().string(), "usd_mimic");
    ASSERT_FALSE(directory_.empty());
  }

  void TearDown() override {
    if (!directory_.empty()) std::filesystem::remove_all(directory_);
  }

  pxr::UsdStageRefPtr MakeStage(bool angular = true, bool mjc_api = true) {
    auto stage = pxr::UsdStage::CreateInMemory();
    EXPECT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def PhysicsScene "scene" (prepend apiSchemas = ["MjcSceneAPI"]) {
    uniform token mjc:compiler:angle = "degree"
}
def Xform "leader" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 1
    }
    def PhysicsRevoluteJoint "joint" {
        rel physics:body1 = </leader>
    }
}
def Xform "other" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 1
    }
    def PhysicsRevoluteJoint "joint" {
        rel physics:body1 = </other>
    }
}
def Xform "follower" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 1
    }
    def PhysicsRevoluteJoint "joint" {
        rel physics:body1 = </follower>
        rel newton:mimicJoint
        rel mjc:target
        float newton:mimicCoef0
        float newton:mimicCoef1
        double mjc:coef0
        double mjc:coef1
        bool newton:mimicEnabled
        bool physics:jointEnabled
    }
}
)"));
    auto follower = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
    EXPECT_TRUE(follower.ApplyAPI(
        pxr::TfToken(mjc_api ? "MjcEqualityJointAPI" : "NewtonMimicAPI")));
    if (!angular) {
      for (const char* path :
           {"/leader/joint", "/other/joint", "/follower/joint"}) {
        EXPECT_TRUE(stage->GetPrimAtPath(pxr::SdfPath(path))
                        .SetTypeName(pxr::TfToken("PhysicsPrismaticJoint")));
      }
    }
    return stage;
  }

  MjModelPtr Compile(const pxr::UsdStageRefPtr& stage) {
    const std::string path = directory_ + "/mimic.usda";
    EXPECT_TRUE(stage->GetRootLayer()->Export(path));
    char error[1024] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
        mj_deleteSpec);
    if (!spec) {
      ADD_FAILURE() << error;
      return nullptr;
    }
    MjModelPtr model(mj_compile(spec.get(), nullptr));
    EXPECT_NE(model, nullptr) << mjs_getError(spec.get());
    return model;
  }

  std::string directory_;
};

// MJC schema defaults must not override an authored Newton value. Test both
// direct NewtonMimicAPI and inheritance through MjcEqualityJointAPI.
class UsdMimicCoefficientsTest
    : public UsdMimicTest,
      public testing::WithParamInterface<std::tuple<bool, bool, bool, int>> {};

TEST_P(UsdMimicCoefficientsTest, UnitsDefaultsAndAuthoredPrecedence) {
  const auto [angular, degrees, mjc_api, source] = GetParam();
  auto stage = MakeStage(angular, mjc_api);
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
  ASSERT_TRUE(joint.GetRelationship(pxr::TfToken("newton:mimicJoint"))
                  .SetTargets({pxr::SdfPath("/leader/joint")}));
  ASSERT_TRUE(
      stage->GetAttributeAtPath(pxr::SdfPath("/scene.mjc:compiler:angle"))
          .Set(pxr::TfToken(degrees ? "degree" : "radian")));
  if (source == 1 || source == 3) {
    ASSERT_TRUE(
        joint.GetAttribute(pxr::TfToken("newton:mimicCoef0")).Set(18.0f));
    ASSERT_TRUE(
        joint.GetAttribute(pxr::TfToken("newton:mimicCoef1")).Set(2.0f));
  }
  if (source >= 2) {
    // An explicitly authored default still overrides the Newton value.
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("mjc:coef0")).Set(0.0));
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("mjc:coef1")).Set(1.0));
    mock_warning_handler.ExpectWarnings("deprecated mjc:");
  }
  auto model = Compile(stage);
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->neq, 1);
  const double expected =
      source == 1 ? (angular ? std::numbers::pi / 10 : 18) : 0;
  EXPECT_NEAR(model->eq_data[0], expected, MjTol(1e-15, 1e-7));
  EXPECT_EQ(model->eq_data[1], source == 1 ? 2 : 1);
}

INSTANTIATE_TEST_SUITE_P(Mimic, UsdMimicCoefficientsTest,
                         testing::Combine(testing::Bool(), testing::Bool(),
                                          testing::Bool(),
                                          testing::Values(0, 1, 2, 3)));

TEST_F(UsdMimicTest, LegacyPolynomialKeepsNativeUnits) {
  auto stage = MakeStage();
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
  ASSERT_TRUE(joint.GetRelationship(pxr::TfToken("mjc:target"))
                  .SetTargets({pxr::SdfPath("/leader/joint")}));
  for (int i = 0; i < 5; ++i) {
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("mjc:coef" + std::to_string(i)))
                    .Set(0.25 * (i + 1)));
  }
  mock_warning_handler.ExpectWarnings("deprecated mjc:");
  auto model = Compile(stage);
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->neq, 1);
  for (int i = 0; i < 5; ++i) EXPECT_EQ(model->eq_data[i], 0.25 * (i + 1));
}

// -1 means unauthored, 0 false, 1 true. Mimic activation is independent of
// the joint's enabled state, including when the mimic flag defaults to true.
class UsdMimicEnabledTest
    : public UsdMimicTest,
      public testing::WithParamInterface<std::tuple<bool, int, int>> {};

TEST_P(UsdMimicEnabledTest, MimicFlagAndDefaultIgnoreJointEnabled) {
  const auto [mjc_api, mimic_enabled, joint_enabled] = GetParam();
  auto stage = MakeStage(true, mjc_api);
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
  ASSERT_TRUE(joint.GetRelationship(pxr::TfToken("newton:mimicJoint"))
                  .SetTargets({pxr::SdfPath("/leader/joint")}));
  if (mimic_enabled >= 0) {
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("newton:mimicEnabled"))
                    .Set(mimic_enabled != 0));
  }
  if (joint_enabled >= 0) {
    ASSERT_TRUE(joint.GetAttribute(pxr::TfToken("physics:jointEnabled"))
                    .Set(joint_enabled != 0));
  }
  auto model = Compile(stage);
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->neq, 1);
  const bool expected = mimic_enabled != 0;
  EXPECT_EQ(model->eq_active0[0], expected);
  // Disabling the equality must preserve its follower joint and DOF.
  EXPECT_EQ(model->njnt, 3);
  EXPECT_EQ(model->nv, 3);
  auto data = MakeData(model);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data->eq_active[0], expected);
}

INSTANTIATE_TEST_SUITE_P(Mimic, UsdMimicEnabledTest,
                         testing::Combine(testing::Bool(),
                                          testing::Values(-1, 0, 1),
                                          testing::Values(-1, 0, 1)));

TEST_F(UsdMimicTest, CoefficientPrecedenceIsPerAttribute) {
  mock_warning_handler.ExpectWarnings("deprecated mjc:");
  for (int legacy_coef : {0, 1}) {
    SCOPED_TRACE(legacy_coef);
    auto stage = MakeStage();
    auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
    ASSERT_TRUE(joint.GetRelationship(pxr::TfToken("newton:mimicJoint"))
                    .SetTargets({pxr::SdfPath("/leader/joint")}));
    ASSERT_TRUE(
        joint.GetAttribute(pxr::TfToken("newton:mimicCoef0")).Set(18.0f));
    ASSERT_TRUE(
        joint.GetAttribute(pxr::TfToken("newton:mimicCoef1")).Set(2.0f));
    ASSERT_TRUE(joint
                    .GetAttribute(
                        pxr::TfToken("mjc:coef" + std::to_string(legacy_coef)))
                    .Set(0.5));
    auto model = Compile(stage);
    ASSERT_NE(model, nullptr);
    ASSERT_EQ(model->neq, 1);
    EXPECT_NEAR(model->eq_data[0],
                legacy_coef == 0 ? 0.5 : std::numbers::pi / 10,
                MjTol(1e-15, 1e-7));
    EXPECT_EQ(model->eq_data[1], legacy_coef == 1 ? 0.5 : 2.0);
  }
}

class UsdLoopJointEnabledTest
    : public UsdMimicTest,
      public testing::WithParamInterface<std::tuple<bool, bool>> {};

TEST_P(UsdLoopJointEnabledTest, WeldAndConnectKeepJointEnabledSemantics) {
  const auto [weld, enabled] = GetParam();
  auto stage = MakeStage();
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
  ASSERT_TRUE(joint.RemoveAPI(pxr::TfToken("MjcEqualityJointAPI")));
  ASSERT_TRUE(joint.SetTypeName(
      pxr::TfToken(weld ? "PhysicsFixedJoint" : "PhysicsSphericalJoint")));
  ASSERT_TRUE(joint.ApplyAPI(
      pxr::TfToken(weld ? "MjcEqualityWeldAPI" : "MjcEqualityConnectAPI")));
  ASSERT_TRUE(joint.CreateRelationship(pxr::TfToken("physics:body0"))
                  .SetTargets({pxr::SdfPath("/leader")}));
  ASSERT_TRUE(
      joint
          .CreateAttribute(pxr::TfToken("physics:excludeFromArticulation"),
                           pxr::SdfValueTypeNames->Bool)
          .Set(true));
  ASSERT_TRUE(
      joint.GetAttribute(pxr::TfToken("physics:jointEnabled")).Set(enabled));
  ASSERT_TRUE(
      joint.GetAttribute(pxr::TfToken("newton:mimicEnabled")).Set(!enabled));
  auto model = Compile(stage);
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->neq, 1);
  EXPECT_EQ(model->eq_type[0], weld ? mjEQ_WELD : mjEQ_CONNECT);
  EXPECT_EQ(model->eq_active0[0], enabled);
}

INSTANTIATE_TEST_SUITE_P(Mimic, UsdLoopJointEnabledTest,
                         testing::Combine(testing::Bool(), testing::Bool()));

class UsdMimicTargetTest : public UsdMimicTest,
                           public testing::WithParamInterface<int> {};

TEST_P(UsdMimicTargetTest, AuthoredMJCPrecedesNewtonIncludingEmptyTargets) {
  const int source = GetParam();
  auto stage = MakeStage();
  auto joint = stage->GetPrimAtPath(pxr::SdfPath("/follower/joint"));
  if (source == 1 || source >= 3) {
    ASSERT_TRUE(joint.GetRelationship(pxr::TfToken("newton:mimicJoint"))
                    .SetTargets({pxr::SdfPath("/leader/joint")}));
  }
  if (source >= 2) {
    pxr::SdfPathVector targets;
    if (source != 4) targets.push_back(pxr::SdfPath("/other/joint"));
    ASSERT_TRUE(
        joint.GetRelationship(pxr::TfToken("mjc:target")).SetTargets(targets));
    mock_warning_handler.ExpectWarnings("deprecated mjc:target");
  }
  auto model = Compile(stage);
  ASSERT_NE(model, nullptr);
  ASSERT_EQ(model->neq, 1);
  int expected = -1;
  if (source == 1)
    expected = mj_name2id(model.get(), mjOBJ_JOINT, "/leader/joint");
  if (source == 2 || source == 3)
    expected = mj_name2id(model.get(), mjOBJ_JOINT, "/other/joint");
  EXPECT_EQ(model->eq_obj2id[0], expected);
}

INSTANTIATE_TEST_SUITE_P(Mimic, UsdMimicTargetTest, testing::Range(0, 5));

class UsdInertialSemanticsTest : public MujocoTest {
 public:
  static void SetUpTestSuite() {
#ifdef USD_DECODER_PLUGIN_PATH
    mj_loadPluginLibrary(USD_DECODER_PLUGIN_PATH);
#endif
  }

 protected:
  void SetUp() override {
    directory_ = pxr::ArchMakeTmpSubdir(
        std::filesystem::temp_directory_path().string(), "usd_inertial");
    ASSERT_FALSE(directory_.empty());
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> Parse(
      const pxr::UsdStageRefPtr& stage) {
    const std::string path = directory_ + "/model.usda";
    EXPECT_TRUE(stage->GetRootLayer()->Export(path));
    char error[1024] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
        mj_deleteSpec);
    EXPECT_NE(spec, nullptr) << error;
    return spec;
  }

  std::string directory_;
};

TEST_F(UsdInertialSemanticsTest, ExplicitBodyInertialPreserved) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 1.2
    float3 physics:diagonalInertia = (0.02, 0.03, 0.04)
    quatf physics:principalAxes = (0.70710678, 0.70710678, 0, 0)
    def Sphere "collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 1
    }
}
def Xform "zero_body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 1.0
    float3 physics:diagonalInertia = (0, 0, 0)
    def Sphere "collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
    ) {
        double radius = 0.1
        float physics:mass = 3.4
    }
}
def Xform "zero_mass_body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
) {
    float physics:mass = 0.0
    float3 physics:diagonalInertia = (0.02, 0.03, 0.04)
    def Sphere "collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
    ) {
        double radius = 0.1
        float physics:mass = 5.6
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsBody* body = mjs_findBody(spec.get(), "/body");
  ASSERT_NE(body, nullptr);
  EXPECT_TRUE(body->explicitinertial);

  mjsBody* zero_body = mjs_findBody(spec.get(), "/zero_body");
  ASSERT_NE(zero_body, nullptr);
  EXPECT_FALSE(zero_body->explicitinertial);

  mjsBody* zero_mass_body = mjs_findBody(spec.get(), "/zero_mass_body");
  ASSERT_NE(zero_mass_body, nullptr);
  EXPECT_FALSE(zero_mass_body->explicitinertial);

  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->nbody, 4);
  EXPECT_NEAR(model->body_mass[1], 1.2, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_ipos[3], 0.0, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_ipos[4], 0.0, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_ipos[5], 0.0, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_inertia[3], 0.02, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_inertia[4], 0.03, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_inertia[5], 0.04, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_iquat[4], 0.70710678, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_iquat[5], 0.70710678, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_iquat[6], 0.0, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_iquat[7], 0.0, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_mass[2], 3.4, MjTol(1e-7, 1e-6));
  EXPECT_NEAR(model->body_mass[3], 5.6, MjTol(1e-7, 1e-6));
}

TEST_F(UsdInertialSemanticsTest, VisualGeomMassAPIDecoded) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "visual" (
        prepend apiSchemas = ["PhysicsMassAPI", "MjcImageableAPI"]
    ) {
        double radius = 1
        float physics:mass = 2.5
        float physics:density = 500
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsGeom* geom =
      mjs_asGeom(mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/visual"));
  ASSERT_NE(geom, nullptr);
  EXPECT_NEAR(geom->mass, 2.5, 1e-6);
  EXPECT_NEAR(geom->density, 500.0, 1e-6);

  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->nbody, 2);
  EXPECT_NEAR(model->body_mass[1], 2.5, MjTol(1e-7, 1e-6));
}

TEST_F(UsdInertialSemanticsTest, MasslessVisualGeomsHaveZeroDensity) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
    ) {
        double radius = 0.1
        float physics:mass = 1.0
    }
    def Sphere "zero_density_collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMassAPI"]
    ) {
        double radius = 1.0
        float physics:density = 0.0
    }
    def Sphere "visual" (prepend apiSchemas = ["MjcImageableAPI"]) {
        double radius = 1.0
        uniform int mjc:group = 0
    }
    def Sphere "plain_visual" {
        double radius = 1.0
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsGeom* zero_collider = mjs_asGeom(
      mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/zero_density_collider"));
  ASSERT_NE(zero_collider, nullptr);
  EXPECT_EQ(zero_collider->density, 0.0);

  mjsGeom* visual =
      mjs_asGeom(mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/visual"));
  ASSERT_NE(visual, nullptr);
  EXPECT_EQ(visual->density, 0.0);

  mjsGeom* plain_visual =
      mjs_asGeom(mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/plain_visual"));
  ASSERT_NE(plain_visual, nullptr);
  EXPECT_EQ(plain_visual->density, 0.0);

  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->nbody, 2);
  EXPECT_NEAR(model->body_mass[1], 1.0, MjTol(1e-7, 1e-6));
}

TEST_F(UsdInertialSemanticsTest, VisualOnlyBodyFallsBackToDefaultDensity) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "visual" {
        double radius = 0.1
    }
}
)"));
  mock_warning_handler.ExpectWarnings("has only visual gprims");
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsGeom* visual =
      mjs_asGeom(mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/visual"));
  ASSERT_NE(visual, nullptr);
  EXPECT_EQ(visual->density, 1000.0);

  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  EXPECT_GT(model->body_mass[1], 0.0);
}

TEST_F(UsdInertialSemanticsTest, FixedAndAuthoredZeroVisualsDoNotFallBack) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "fixed_body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "visual" {
        double radius = 0.1
    }
}
def PhysicsFixedJoint "weld" {
    prepend rel physics:body1 = </fixed_body>
}
def Xform "authored_zero_body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "zero_visual" (prepend apiSchemas = ["PhysicsMassAPI"]) {
        double radius = 0.1
        float physics:density = 0.0
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsGeom* fixed_visual =
      mjs_asGeom(mjs_findElement(spec.get(), mjOBJ_GEOM, "/fixed_body/visual"));
  ASSERT_NE(fixed_visual, nullptr);
  EXPECT_EQ(fixed_visual->density, 0.0);

  mjsGeom* zero_visual = mjs_asGeom(mjs_findElement(
      spec.get(), mjOBJ_GEOM, "/authored_zero_body/zero_visual"));
  ASSERT_NE(zero_visual, nullptr);
  EXPECT_EQ(zero_visual->density, 0.0);
}

class UsdPreservationSemanticsTest : public MujocoTest {
 public:
  static void SetUpTestSuite() {
#ifdef USD_DECODER_PLUGIN_PATH
    mj_loadPluginLibrary(USD_DECODER_PLUGIN_PATH);
#endif
  }

 protected:
  void SetUp() override {
    directory_ = pxr::ArchMakeTmpSubdir(
        std::filesystem::temp_directory_path().string(), "usd_preservation");
    ASSERT_FALSE(directory_.empty());
  }

  void TearDown() override {
    if (!directory_.empty()) {
      std::filesystem::remove_all(directory_);
    }
  }

  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> Parse(
      const pxr::UsdStageRefPtr& stage,
      const std::string& filename = "model.usda") {
    const std::string path = directory_ + "/" + filename;
    EXPECT_TRUE(stage->GetRootLayer()->Export(path));
    char error[1024] = {};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
        mj_deleteSpec);
    EXPECT_NE(spec, nullptr) << error;
    return spec;
  }

  std::string directory_;
};

TEST_F(UsdPreservationSemanticsTest, SiteGprimNotDecodedAsGeom) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "mjc_site" (prepend apiSchemas = ["MjcSiteAPI"]) {
        double radius = 0.05
    }
    def Sphere "newton_site" (prepend apiSchemas = ["NewtonSiteAPI"]) {
        double radius = 0.05
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(mjs_firstElement(spec.get(), mjOBJ_GEOM), nullptr);
  EXPECT_NE(
      mjs_asSite(mjs_findElement(spec.get(), mjOBJ_SITE, "/body/mjc_site")),
      nullptr);
  EXPECT_NE(
      mjs_asSite(mjs_findElement(spec.get(), mjOBJ_SITE, "/body/newton_site")),
      nullptr);
}

TEST_F(UsdPreservationSemanticsTest, StageModelNamePrecedence) {
  auto stage_display = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage_display->GetRootLayer()->ImportFromString(R"(#usda 1.0
(defaultPrim = "MyRobot")
def Xform "MyRobot" (displayName = "CustomDisplayName") {
}
)"));
  auto spec_display = Parse(stage_display, "layer_stem.usda");
  ASSERT_NE(spec_display, nullptr);
  EXPECT_STREQ(mjs_getString(spec_display->modelname), "CustomDisplayName");

  auto stage_prim = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage_prim->GetRootLayer()->ImportFromString(R"(#usda 1.0
(defaultPrim = "MyRobot")
def Xform "MyRobot" {
}
)"));
  auto spec_prim = Parse(stage_prim, "layer_stem.usda");
  ASSERT_NE(spec_prim, nullptr);
  EXPECT_STREQ(mjs_getString(spec_prim->modelname), "MyRobot");

  auto stage_layer = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage_layer->GetRootLayer()->ImportFromString(R"(#usda 1.0
def Xform "MyRobot" {
}
)"));
  auto spec_layer = Parse(stage_layer, "layer_stem.usda");
  ASSERT_NE(spec_layer, nullptr);
  EXPECT_STREQ(mjs_getString(spec_layer->modelname), "layer_stem");

  char error[1024] = {};
  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec_anon(
      mj_parse(stage_layer->GetRootLayer()->GetIdentifier().c_str(),
               "model/usd", nullptr, error, sizeof(error)),
      mj_deleteSpec);
  ASSERT_NE(spec_anon, nullptr) << error;
  EXPECT_STREQ(mjs_getString(spec_anon->modelname), "MuJoCo Model");
}

TEST_F(UsdPreservationSemanticsTest, UnlimitedJointsRemainUnlimited) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "unlimited_hinge_body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def PhysicsRevoluteJoint "unlimited_hinge" {
        rel physics:body1 = </unlimited_hinge_body>
        uniform token physics:axis = "Z"
    }
}
def Xform "unlimited_slide_body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def PhysicsPrismaticJoint "unlimited_slide" {
        rel physics:body1 = </unlimited_slide_body>
        uniform token physics:axis = "X"
    }
}
def Xform "limited_hinge_body" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsArticulationRootAPI"]
) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def PhysicsRevoluteJoint "limited_hinge" {
        rel physics:body1 = </limited_hinge_body>
        uniform token physics:axis = "Z"
        float physics:lowerLimit = -45
        float physics:upperLimit = 45
    }
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());

  int unlimited_hinge_id = mj_name2id(model.get(), mjOBJ_JOINT,
                                      "/unlimited_hinge_body/unlimited_hinge");
  int unlimited_slide_id = mj_name2id(model.get(), mjOBJ_JOINT,
                                      "/unlimited_slide_body/unlimited_slide");
  int limited_hinge_id =
      mj_name2id(model.get(), mjOBJ_JOINT, "/limited_hinge_body/limited_hinge");
  ASSERT_GE(unlimited_hinge_id, 0);
  ASSERT_GE(unlimited_slide_id, 0);
  ASSERT_GE(limited_hinge_id, 0);
  EXPECT_EQ(model->jnt_limited[unlimited_hinge_id], mjLIMITED_FALSE);
  EXPECT_EQ(model->jnt_limited[unlimited_slide_id], mjLIMITED_FALSE);
  EXPECT_EQ(model->jnt_limited[limited_hinge_id], mjLIMITED_TRUE);
}

TEST_F(UsdPreservationSemanticsTest,
       DisabledColliderPreservesMjcCollisionMetadata) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cube "disabled_collider" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "MjcCollisionAPI"]
    ) {
        double size = 0.1
        bool physics:collisionEnabled = false
        uniform int mjc:group = 2
        uniform int mjc:priority = 3
        uniform int mjc:condim = 4
    }
}
)"));
  mock_warning_handler.ExpectWarnings("has only visual gprims");
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  mjsGeom* geom = mjs_asGeom(
      mjs_findElement(spec.get(), mjOBJ_GEOM, "/body/disabled_collider"));
  ASSERT_NE(geom, nullptr);
  EXPECT_EQ(geom->contype, 0);
  EXPECT_EQ(geom->conaffinity, 0);
  EXPECT_EQ(geom->group, 2);
  EXPECT_EQ(geom->priority, 3);
  EXPECT_EQ(geom->condim, 4);
}

TEST_F(UsdPreservationSemanticsTest, DiamondGraphKinematicTree) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def PhysicsScene "scene" {}
def Xform "body2" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
}
def Xform "body3" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "collider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
}
def PhysicsFixedJoint "joint_world_body2" {
    rel physics:body1 = </body2>
}
def PhysicsFixedJoint "joint_world_body3" {
    rel physics:body1 = </body3>
}
def PhysicsFixedJoint "joint_body2_body3" {
    rel physics:body0 = </body2>
    rel physics:body1 = </body3>
}
)"));
  const std::string path = directory_ + "/diamond.usda";
  ASSERT_TRUE(stage->GetRootLayer()->Export(path));

  static thread_local std::string captured_error;
  captured_error.clear();
  auto old_handler = _mjPRIVATE_setTlsLogHandler(+[](const mjLogMessage* msg) {
    if (msg->level == mjLOG_ERROR) {
      captured_error = msg->subject;
    }
  });
  char error[1024] = {};
  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
      mj_parse(path.c_str(), nullptr, nullptr, error, sizeof(error)),
      mj_deleteSpec);
  _mjPRIVATE_setTlsLogHandler(old_handler);

  EXPECT_EQ(spec, nullptr);
  EXPECT_NE(captured_error.find("Cycle detected in the kinematic tree"),
            std::string::npos)
      << captured_error;

  ASSERT_TRUE(
      stage->GetPrimAtPath(pxr::SdfPath("/joint_body2_body3"))
          .CreateAttribute(pxr::TfToken("physics:excludeFromArticulation"),
                           pxr::SdfValueTypeNames->Bool)
          .Set(true));
  auto valid_spec = Parse(stage, "diamond_excluded.usda");
  ASSERT_NE(valid_spec, nullptr);
  MjModelPtr model(mj_compile(valid_spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(valid_spec.get());
  EXPECT_EQ(model->neq, 1);
}

// tiny RGB 3 x 2 PNG file
constexpr uint8_t kTinyPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0x12, 0x16, 0xf1, 0x4d,
    0x00, 0x00, 0x00, 0x1c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
    0x78, 0xc1, 0xc0, 0xc0, 0xc0, 0xf0, 0xbf, 0xb8, 0xb8, 0x98, 0x81,
    0xe1, 0x3f, 0xc3, 0xff, 0xff, 0xff, 0xc5, 0xc4, 0xc4, 0x00, 0x46,
    0xd7, 0x07, 0x7f, 0xd2, 0x52, 0xa1, 0x41, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

TEST_F(UsdPreservationSemanticsTest,
       MissingTextureFileDoesNotCreateOrphanTexture) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
        rel material:binding = </material>
    }
}
def Material "material" {
    token outputs:surface.connect = </material/surface.outputs:surface>
    def Shader "surface" {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </material/texture.outputs:rgb>
        token outputs:surface
    }
    def Shader "texture" {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @nonexistent.png@
        float3 outputs:rgb
    }
}
)"));
  mock_warning_handler.ExpectWarnings("Could not open");
  mock_warning_handler.ExpectWarnings("No value or texture for shader input");
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(mjs_findElement(spec.get(), mjOBJ_TEXTURE, "/material/texture"),
            nullptr);
  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
}

TEST_F(UsdPreservationSemanticsTest, SharedTextureDoesNotCollide) {
  const std::string png_path = directory_ + "/test.png";
  std::ofstream(png_path, std::ios::binary)
      .write(reinterpret_cast<const char*>(kTinyPng), sizeof(kTinyPng));

  auto stage = pxr::UsdStage::CreateInMemory();
  const std::string usda = std::string(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
        rel material:binding = </material>
    }
}
def Material "material" {
    token outputs:surface.connect = </material/surface.outputs:surface>
    def Shader "surface" {
        uniform token info:id = "UsdPreviewSurface"
        float inputs:metallic.connect = </material/texture.outputs:r>
        float inputs:roughness.connect = </material/texture.outputs:r>
        color3f inputs:diffuseColor.connect = </material/texture.outputs:rgb>
        token outputs:surface
    }
    def Shader "texture" {
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @)") +
                           png_path + R"(@
        float outputs:r
        float3 outputs:rgb
    }
}
)";
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(usda));

  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  EXPECT_STREQ(mjs_getError(spec.get()), "");
  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->ntex, 1);
  EXPECT_EQ(model->tex_nchannel[0], 3);
}

TEST_F(UsdPreservationSemanticsTest,
       FixedTendonCoefMismatchSkipsAndSpatialSegmentZeroPulleyPreserved) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def Sphere "site1" (prepend apiSchemas = ["MjcSiteAPI"]) {
        double radius = 0.01
    }
    def Sphere "site2" (prepend apiSchemas = ["MjcSiteAPI"]) {
        double radius = 0.01
    }
    def PhysicsRevoluteJoint "j1" {
        rel physics:body1 = </body>
        uniform token physics:axis = "Z"
    }
    def PhysicsRevoluteJoint "j2" {
        rel physics:body1 = </body>
        uniform token physics:axis = "Y"
    }
}
def MjcTendon "bad_fixed" {
    uniform token mjc:type = "fixed"
    rel mjc:path = [</body/j1>, </body/j2>]
    uniform double[] mjc:path:coef = [1.5]
}
def MjcTendon "spatial_pulley0" {
    uniform token mjc:type = "spatial"
    rel mjc:path = [</body/site1>, </body/site2>]
    uniform int[] mjc:path:segments = [0, 0]
    uniform double[] mjc:path:divisors = [2.0]
}
)"));
  mock_warning_handler.ExpectWarnings("has 1 coefs but 2 wrap targets");
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(mjs_findElement(spec.get(), mjOBJ_TENDON, "/bad_fixed"), nullptr);

  mjsTendon* spatial = mjs_asTendon(
      mjs_findElement(spec.get(), mjOBJ_TENDON, "/spatial_pulley0"));
  ASSERT_NE(spatial, nullptr);
  ASSERT_EQ(mjs_getWrapNum(spatial), 3);
  EXPECT_EQ(mjs_getWrap(spatial, 0)->type, mjWRAP_PULLEY);
  EXPECT_DOUBLE_EQ(mjs_getWrapDivisor(mjs_getWrap(spatial, 0)), 2.0);
}

TEST_F(UsdPreservationSemanticsTest,
       SphericalLoopJointWithoutMjcEqualityConnectAPIIsConnect) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "leader" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def PhysicsRevoluteJoint "joint" {
        rel physics:body1 = </leader>
    }
}
def Xform "follower" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Sphere "geom" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        double radius = 0.1
    }
    def PhysicsRevoluteJoint "joint" {
        rel physics:body1 = </follower>
    }
}
def PhysicsSphericalJoint "loop_joint" {
    rel physics:body0 = </leader>
    rel physics:body1 = </follower>
    bool physics:excludeFromArticulation = true
    point3f physics:localPos0 = (1, 2, 3)
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  ASSERT_EQ(model->neq, 1);
  EXPECT_EQ(model->eq_type[0], mjEQ_CONNECT);
  EXPECT_MJTNUM_EQ(model->eq_data[0], 1.0);
  EXPECT_MJTNUM_EQ(model->eq_data[1], 2.0);
  EXPECT_MJTNUM_EQ(model->eq_data[2], 3.0);
}

TEST_F(UsdPreservationSemanticsTest, CylinderAndCapsuleAxisAndScale) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI"]) {
    def Cylinder "cyl_z" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "Z"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0.70710678, 0, 0)
        float3 xformOp:scale = (2, 2, 3)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Cylinder "cyl_x" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "X"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)
        float3 xformOp:scale = (3, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Cylinder "cyl_y" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "Y"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)
        float3 xformOp:scale = (2, 3, 2)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Capsule "cap_z" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "Z"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0.70710678, 0, 0)
        float3 xformOp:scale = (1, 2.5, 3)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Capsule "cap_x" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "X"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)
        float3 xformOp:scale = (3, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Capsule "cap_y" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {
        uniform token axis = "Y"
        double radius = 0.4
        double height = 1.6
        quatf xformOp:orient = (0.70710678, 0, 0, 0.70710678)
        float3 xformOp:scale = (2, 3, 2)
        uniform token[] xformOpOrder = ["xformOp:orient", "xformOp:scale"]
    }
    def Cylinder "site_cyl_x" (prepend apiSchemas = ["MjcSiteAPI"]) {
        uniform token axis = "X"
        double radius = 0.25
        double height = 1.2
        float3 xformOp:scale = (4, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:scale"]
    }
}
)"));

  mock_warning_handler.ExpectWarnings("Non-uniform radial scale");
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  MjModelPtr model(mj_compile(spec.get(), nullptr));
  ASSERT_NE(model, nullptr) << mjs_getError(spec.get());
  auto data = MakeData(model);
  ASSERT_NE(data, nullptr);
  mj_forward(model.get(), data.get());

  struct GeomExpectation {
    const char* name;
    mjtGeom type;
    double radius;
    double half_height;
    double world_axis[3];
  };
  const GeomExpectation kExpected[] = {
      {"/body/cyl_z", mjGEOM_CYLINDER, 0.8, 2.4, {0.0, -1.0, 0.0}},
      {"/body/cyl_x", mjGEOM_CYLINDER, 0.8, 2.4, {0.0, 1.0, 0.0}},
      {"/body/cyl_y", mjGEOM_CYLINDER, 0.8, 2.4, {-1.0, 0.0, 0.0}},
      {"/body/cap_z", mjGEOM_CAPSULE, 1.0, 2.4, {0.0, -1.0, 0.0}},
      {"/body/cap_x", mjGEOM_CAPSULE, 0.8, 2.4, {0.0, 1.0, 0.0}},
      {"/body/cap_y", mjGEOM_CAPSULE, 0.8, 2.4, {-1.0, 0.0, 0.0}},
  };

  for (const auto& expected : kExpected) {
    SCOPED_TRACE(expected.name);
    int id = mj_name2id(model.get(), mjOBJ_GEOM, expected.name);
    ASSERT_GE(id, 0);
    EXPECT_EQ(model->geom_type[id], expected.type);
    EXPECT_NEAR(model->geom_size[3 * id + 0], expected.radius, 1e-6);
    EXPECT_NEAR(model->geom_size[3 * id + 1], expected.half_height, 1e-6);
    // MuJoCo cylinders and capsules are aligned with local +Z (column 2 of
    // geom_xmat).
    EXPECT_NEAR(data->geom_xmat[9 * id + 2], expected.world_axis[0], 1e-5);
    EXPECT_NEAR(data->geom_xmat[9 * id + 5], expected.world_axis[1], 1e-5);
    EXPECT_NEAR(data->geom_xmat[9 * id + 8], expected.world_axis[2], 1e-5);
  }

  int site_id = mj_name2id(model.get(), mjOBJ_SITE, "/body/site_cyl_x");
  ASSERT_GE(site_id, 0);
  EXPECT_EQ(model->site_type[site_id], mjGEOM_CYLINDER);
  EXPECT_NEAR(model->site_size[3 * site_id + 0], 0.5, 1e-6);
  EXPECT_NEAR(model->site_size[3 * site_id + 1], 2.4, 1e-6);
  EXPECT_NEAR(data->site_xmat[9 * site_id + 2], 1.0, 1e-5);
  EXPECT_NEAR(data->site_xmat[9 * site_id + 5], 0.0, 1e-5);
  EXPECT_NEAR(data->site_xmat[9 * site_id + 8], 0.0, 1e-5);
}

TEST_F(UsdPreservationSemanticsTest, ActuatorGroupDisableOptionParsed) {
  auto stage = pxr::UsdStage::CreateInMemory();
  ASSERT_TRUE(stage->GetRootLayer()->ImportFromString(R"(#usda 1.0
(metersPerUnit = 1)
def PhysicsScene "PhysicsScene" (prepend apiSchemas = ["MjcSceneAPI"]) {
    uniform int[] mjc:option:actuatorgroupdisable = [0, 2]
}
)"));
  auto spec = Parse(stage);
  ASSERT_NE(spec, nullptr);
  EXPECT_EQ(spec->option.disableactuator, (1 << 0) | (1 << 2));
}

}  // namespace
}  // namespace mujoco
