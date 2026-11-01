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

// Tests for user/user_objects.cc.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using MjCMeshTest = MujocoTest;

static const char* const kMeshPath = "user/testdata/mesh.xml";
static const char* const kDuplicateVerticesPath =
    "user/testdata/duplicate_vertices.xml";
static const char* const kCubePath = "user/testdata/cube.xml";
static const char* const kCubeCompletePath = "user/testdata/cube_complete.obj";
static const char* const kTorusPath = "user/testdata/torus.xml";
static const char* const kTorusMaxhullVertPath =
    "user/testdata/torus_maxhullvert.xml";
static const char* const kTorusDefaultMaxhullVertPath =
    "user/testdata/torus_maxhullvert_default.xml";
static const char* const kCompareInertiaPath =
    "user/testdata/inertia_compare.xml";
static const char* const kConvexInertiaPath =
    "user/testdata/inertia_convex.xml";
static const char* const kConcaveInertiaPath =
    "user/testdata/inertia_concave.xml";
static const char* const kShellInertiaPath = "user/testdata/inertia_shell.xml";
static const char* const kTorusQuadsPath = "user/testdata/torus_quads.xml";
static const char* const kTexturedTorusPath =
    "user/testdata/textured_torus.xml";
static const char* const kDuplicateOBJPath = "user/testdata/duplicate.xml";
static const char* const kMalformedFaceOBJPath =
    "user/testdata/malformed_face.xml";
static const char* const kCubeSkinPath = "user/testdata/cube_skin.xml";

static const char* const kNoDecoderForMeshErrorMsh =
    "no decoder found for mesh";

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

static constexpr mjtNum kMaxAbsErr = std::numeric_limits<float>::epsilon();

// ------------- test invalid filenames ----------------------------------------

TEST_F(MjCMeshTest, UnknownMeshFormat) {
  static constexpr char xml_format[] = R"(
    <mujoco>
      <asset>
        <mesh name="m" file="%s"/>
      </asset>
      <worldbody>
        <geom type="mesh" mesh="m"/>
      </worldbody>
    </mujoco>
  )";

  std::vector<std::string> invalid_names = {"noextension", "anobj", "f",
                                            "mesh.exe", "file%s"};
  mjVFS vfs;
  mj_defaultVFS(&vfs);

  for (const auto& name : invalid_names) {
    mj_addBufferVFS(&vfs, name.c_str(), nullptr, 0);
    std::string xml = absl::StrFormat(xml_format, name);
    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml.c_str(), error.data(), error.size(), &vfs);
    ASSERT_THAT(model.get(), testing::IsNull())
        << "Should fail to load a mesh named: " << name;
    EXPECT_THAT(error.data(), HasSubstr(kNoDecoderForMeshErrorMsh));
    EXPECT_THAT(error.data(), HasSubstr(name));
  }

  mj_deleteVFS(&vfs);
}

// -------------------- test OS filesystem fallback ----------------------------

TEST_F(MjCMeshTest, LoadMSHWithVFS) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" file="unknown_file.msh"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should fallback to OS filesystem
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

TEST_F(MjCMeshTest, LoadOBJWithVFS) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" file="unknown_file.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should fallback to OS filesystem
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

TEST_F(MjCMeshTest, LoadSTLWithVFS) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" file="unknown_file.stl"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should fallback to OS filesystem
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

// ------------- test content_type attributes ----------------------------------

TEST_F(MjCMeshTest, LoadMSHWithContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model/vnd.mujoco.msh" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should try opening the file (not found obviously)
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

TEST_F(MjCMeshTest, LoadOBJWithContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model/obj" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should try opening the file (not found obviously)
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

TEST_F(MjCMeshTest, LoadSTLWithContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model/stl" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should try opening the file (not found obviously)
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

TEST_F(MjCMeshTest, LoadMSHWithContentTypeError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model/unknown" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "some_file", nullptr, 0);

  // should error with unknown file type
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, &vfs);
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr(kNoDecoderForMeshErrorMsh));
  mj_deleteVFS(&vfs);
}

TEST_F(MjCMeshTest, LoadMSHWithInvalidContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "some_file", nullptr, 0);

  // should error with unknown file type
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, &vfs);
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr(kNoDecoderForMeshErrorMsh));
  mj_deleteVFS(&vfs);
}

TEST_F(MjCMeshTest, LoadMSHWithContentTypeParam) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh1" content_type="model/vnd.mujoco.msh;parameter=value" file="some_file"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="mesh1"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  size_t error_sz = 1024;

  // load VFS on the heap
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  // should try opening the file (not found obviously)
  MjModelPtr model = LoadModelFromString(xml, error, error_sz, vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("Error opening file"));
  mj_deleteVFS(vfs.get());
}

// ------------- test vertex deduplication (STL) ------------------------------

TEST_F(MjCMeshTest, DeduplicateSTLVertices) {
  const std::string xml_path = GetTestDataFilePath(kDuplicateVerticesPath);
  char error[1024];
  size_t error_sz = 1024;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error, error_sz);
  ASSERT_THAT(model, NotNull()) << error;
  ASSERT_EQ(model->nmeshvert, 4);
  mj_deleteModel(model);
}

// -------------------- test Mesh loading (MSH) --------------------------------

TEST_F(MjCMeshTest, LoadMSH) {
  const std::string xml_path = GetTestDataFilePath(kMeshPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  ASSERT_EQ(model->nmeshvert, 36);
  mj_deleteModel(model);
}

// ------------- test OBJ loading ----------------------------------------------

using MjCMeshTest = MujocoTest;

TEST_F(MjCMeshTest, LoadMSHMirroredScale) {
  const std::string msh_path = GetTestDataFilePath("user/testdata/cube.msh");
  std::string xml = absl::StrFormat(
      R"(
  <mujoco>
    <asset>
      <mesh name="unmirrored" file="%s" scale="1 1 1" inertia="exact"/>
      <mesh name="mirrored" file="%s" scale="-1 1 1" inertia="exact"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="unmirrored"/>
      <geom type="mesh" mesh="mirrored"/>
    </worldbody>
  </mujoco>
  )",
      msh_path, msh_path);
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml.c_str(), error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nmesh, 2);
  ASSERT_GT(model->mesh_facenum[0], 0);
  ASSERT_EQ(model->mesh_facenum[0], model->mesh_facenum[1]);
  ASSERT_GT(model->mesh_texcoordnum[1], 0);

  int fadr0 = model->mesh_faceadr[0];
  int fadr1 = model->mesh_faceadr[1];
  for (int f = 0; f < model->mesh_facenum[1]; ++f) {
    // mirrored face winding is flipped once relative to unmirrored
    EXPECT_EQ(model->mesh_face[3 * (fadr1 + f) + 0],
              model->mesh_face[3 * (fadr0 + f) + 0]);
    EXPECT_EQ(model->mesh_face[3 * (fadr1 + f) + 1],
              model->mesh_face[3 * (fadr0 + f) + 2]);
    EXPECT_EQ(model->mesh_face[3 * (fadr1 + f) + 2],
              model->mesh_face[3 * (fadr0 + f) + 1]);
    // mesh_facetexcoord keeps the same winding as mesh_face
    for (int k = 0; k < 3; ++k) {
      EXPECT_EQ(model->mesh_facetexcoord[3 * (fadr1 + f) + k],
                model->mesh_face[3 * (fadr1 + f) + k]);
    }
  }
}

TEST_F(MjCMeshTest, LoadMSHWithUserNormals) {
  const std::string msh_path = GetTestDataFilePath("user/testdata/cube.msh");
  std::unique_ptr<mjSpec, void (*)(mjSpec*)> spec(mj_makeSpec(), mj_deleteSpec);
  mjsMesh* mesh = mjs_addMesh(spec.get(), nullptr);
  mjs_setName(mesh->element, "cube");
  *mesh->file = msh_path;
  mesh->usernormal->assign(36 * 3, 0.0f);
  for (int i = 0; i < 36; ++i) {
    (*mesh->usernormal)[3 * i + 2] = 1.0f;
  }
  mjModel* model = mj_compile(spec.get(), nullptr);
  std::unique_ptr<mjModel, void (*)(mjModel*)> model_ptr(model, mj_deleteModel);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec.get());
  EXPECT_EQ(model->nmeshvert, 36);
}

TEST_F(MjCMeshTest, RepeatedFaceNormalSpecification) {
  const std::string obj_path =
      GetTestDataFilePath("user/testdata/cube_complete.obj");
  std::unique_ptr<mjSpec, void (*)(mjSpec*)> spec(mj_makeSpec(), mj_deleteSpec);
  mjsMesh* mesh = mjs_addMesh(spec.get(), nullptr);
  mjs_setName(mesh->element, "cube");
  *mesh->file = obj_path;
  mesh->userfacenormal->assign(36, 0);
  mjModel* model = mj_compile(spec.get(), nullptr);
  std::unique_ptr<mjModel, void (*)(mjModel*)> model_ptr(model, mj_deleteModel);
  EXPECT_THAT(model, IsNull());
  EXPECT_THAT(mjs_getError(spec.get()),
              HasSubstr("repeated facenormal specification"));
}

TEST_F(MjCMeshTest, LoadCube) {
  const std::string xml_path = GetTestDataFilePath(kCubePath);
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, nullptr, 0);
  ASSERT_GT(model->ngeom, 0);
  ASSERT_EQ(model->nmeshvert, 8);
  ASSERT_EQ(model->nmeshface, 12);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, LoadTorus) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_GT(model->ngeom, 0);
  ASSERT_GT(model->nmeshvert, 0);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, SavedContentTypeIndependentOfCache) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;

  // the second compilation loads the mesh from the asset cache
  std::array<std::string, 2> saved;
  for (std::string& xml : saved) {
    mjSpec* spec =
        mj_parseXML(xml_path.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << error.data();
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    std::array<char, 8192> str;
    ASSERT_EQ(mj_saveXMLString(spec, str.data(), str.size(), error.data(),
                               error.size()),
              0)
        << error.data();
    xml = str.data();
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }

  EXPECT_THAT(saved[0], HasSubstr("content_type=\"model/obj\""));
  EXPECT_EQ(saved[0], saved[1]);
}

TEST_F(MjCMeshTest, LoadTorusQuads) {
  const std::string xml_path = GetTestDataFilePath(kTorusQuadsPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_GT(model->ngeom, 0);
  ASSERT_GT(model->nmeshvert, 0);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, LoadTexturedTorus) {
  const std::string xml_path = GetTestDataFilePath(kTexturedTorusPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_GT(model->ngeom, 0);
  ASSERT_GT(model->nmeshvert, 0);
  ASSERT_GT(model->ntex, 0);
  ASSERT_GT(model->ntexdata, 0);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, KeepDuplicateOBJVertices) {
  const std::string xml_path = GetTestDataFilePath(kDuplicateOBJPath);
  char error[1024];
  size_t error_sz = 1024;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error, error_sz);
  ASSERT_EQ(model->nmeshvert, 16);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, SaveMeshOnce) {
  const std::string xml_path = GetTestDataFilePath(kCubePath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  // Confirm the mesh file is loaded and its resolved path stored in paths
  EXPECT_EQ(std::filesystem::path(&model->paths[model->mesh_pathadr[0]])
                .lexically_normal(),
            std::filesystem::path(GetTestDataFilePath("user/testdata/cube.obj"))
                .lexically_normal());
  std::string saved_xml = SaveAndReadXml(model);
  EXPECT_THAT(saved_xml, Not(testing::HasSubstr("vertex")));
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, TinyMeshLoads) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tiny" vertex="0 0 0  1e-4 0 0  0 1e-4 0  0 0 1e-4"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="tiny"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nmeshpoly, 4);

  // Polygon areas are ~5e-9 m^2 (squared Newell norm ~1e-16 < mjEPS);
  // normals must still be distinct rather than falling back to (1, 0, 0).
  for (int i = 0; i < 4; ++i) {
    for (int j = i + 1; j < 4; ++j) {
      EXPECT_LT(mju_dot3(model->mesh_polynormal + 3 * i,
                         model->mesh_polynormal + 3 * j),
                1e-7);
    }
  }
}

// ------------- test max hull vert -------------------------------------------
TEST_F(MjCMeshTest, MaxHullVert) {
  const std::string xml_path = GetTestDataFilePath(kTorusMaxhullVertPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_GT(model->ngeom, 0);
  ASSERT_EQ(model->mesh_graph[0], 4);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, MaxHullVertDefault) {
  const std::string xml_path =
      GetTestDataFilePath(kTorusDefaultMaxhullVertPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_GT(model->ngeom, 0);
  ASSERT_EQ(model->mesh_graph[0], 64);
  mj_deleteModel(model);
}

// ------------- test inline loading ------------------------------------------
TEST_F(MjCMeshTest, FaceNormalAutogenerated) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="1 0 0  0 1 0  0 0 1  0.707 0 0.707"
        face="0 2 1  0 3 2" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

// ------------- test inertia -------------------------------------------------

TEST_F(MjCMeshTest, SmallInertiaLoads) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tiny" vertex="0 0 0  1e-4 0 0  0 1e-4 0  0 0 1e-4"/>
    </asset>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="mesh" mesh="tiny"/>
        <geom name="small" size=".001"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

TEST_F(MjCMeshTest, TinyInertiaFails) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tiny" vertex="0 0 0  1e-4 0 0  0 1e-4 0  0 0 1e-4"/>
    </asset>
    <worldbody>
      <body name="tiny_body">
        <freejoint/>
        <geom type="mesh" mesh="tiny"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(
      error.data(),
      HasSubstr(
          "mass and inertia of moving bodies must be larger than mjMINVAL"));
  EXPECT_THAT(error.data(), HasSubstr("Element name 'tiny_body'"));
}

TEST_F(MjCMeshTest, FlippedFaceAllowedLegacyInertia) {
  const std::string xml_path = GetTestDataFilePath(kMalformedFaceOBJPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  EXPECT_THAT(model->nmeshface, 4);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, MissingFaceAllowedConvexInertia) {
  const std::string xml_path = GetTestDataFilePath(kCompareInertiaPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  EXPECT_THAT(model->nmeshface, 7);
  EXPECT_NEAR(model->body_inertia[3], model->body_inertia[6], kMaxAbsErr);
  EXPECT_NEAR(model->body_inertia[4], model->body_inertia[7], kMaxAbsErr);
  EXPECT_NEAR(model->body_inertia[5], model->body_inertia[8], kMaxAbsErr);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, FlippedFaceFailsExactInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh" inertia="exact"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="2 0 3  0 1 3  1 2 3  0 1 2" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr(
          "Error: faces of mesh 'example_mesh' have inconsistent orientation. "
          "Please check the faces containing the vertices 1 and 2."));
}

void CheckTetrahedronWasRescaled(mjModel* model) {
  // The rotated and rescaled positions of the tetrahedron
  // with vertices (0, 0, 0), (1, 0, 0), (0, 2, 0), (0, 0, 3)
  // after mesh preprocessing is performed
  std::vector<mjtNum> vert = {
      -0.51610732078552246, -0.57402682304382324, -0.52832412719726562,
      0.42337465286254883,  -0.90627527236938477, -0.61189794540405273,
      0.065528050065040588, 1.2306685447692871,   -1.1645431518554688,
      0.027204651385545731, 0.24963347613811493,  2.3047652244567871};
  mjtNum tolerance = std::numeric_limits<float>::epsilon();
  for (int i = 0; i < 12; ++i) {
    EXPECT_NEAR(model->mesh_vert[i], vert[i], tolerance);
  }
}

TEST_F(MjCMeshTest, FlippedFaceAllowedWorld) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 2 0  0 0 3"
        face="2 0 3  0 1 3  1 2 3  0 1 2" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  CheckTetrahedronWasRescaled(model.get());
}

TEST_F(MjCMeshTest, FlippedFaceAllowedNoMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 2 0  0 0 3"
        face="2 0 3  0 1 3  1 2 3  0 1 2" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh" mass="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  CheckTetrahedronWasRescaled(model.get());
}

TEST_F(MjCMeshTest, FlippedFaceAllowedInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 2 0  0 0 3"
        face="2 0 3  0 1 3  1 2 3  0 1 2" />
    </asset>
    <worldbody>
      <body>
        <inertial pos="0 0 0" mass="1"/>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  CheckTetrahedronWasRescaled(model.get());
}

TEST_F(MjCMeshTest, FlippedFaceAllowedNegligibleArea) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 2 0  0 0 3  0 0 3"
        face="2 0 3  0 1 3  1 2 3  0 2 1  0 3 4" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  CheckTetrahedronWasRescaled(model.get());
}

TEST_F(MjCMeshTest, AreaTooSmall) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1e-8 0 0  0 1e-8 0  0 0 1e-8"
        face="2 0 3  0 1 3  1 2 3  0 2 1" inertia="shell"/>
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("mesh surface area is too small"));
}

TEST_F(MjCMeshTest, VolumeTooSmall) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(), HasSubstr("mesh volume is too small"));
}

TEST_F(MjCMeshTest, VisualVolumeTooSmall) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" mass="0"/>
      </default>
    </default>
    <asset>
      <mesh name="example_mesh"
        vertex="0 -4e-16 0  1 0 4e-16  0 1 0  0 0 1"
        face="0 2 1" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh" class="visual"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(), HasSubstr("mesh volume is too small"));
}

TEST_F(MjCMeshTest, VisualVolumeSmallAllowedShell) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" mass="0"/>
      </default>
    </default>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  1 1 1e-6"
        face="0 1 2  2 1 3" />
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh" class="visual"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  EXPECT_LE(mju_abs(model->geom_size[0]), 1);
  EXPECT_LE(mju_abs(model->geom_size[1]), 1);
  EXPECT_LE(mju_abs(model->geom_size[2]), 1);
}

TEST_F(MjCMeshTest, VolumeSmallAllowedShell) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  1 1 1e-6"
        face="0 1 2  2 1 3" inertia="shell"/>
    </asset>
    <worldbody>
      <body>
        <geom type="mesh" mesh="example_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  EXPECT_LE(mju_abs(model->geom_size[0]), 1);
  EXPECT_LE(mju_abs(model->geom_size[1]), 1);
  EXPECT_LE(mju_abs(model->geom_size[2]), 1);
}

// Check the compiled stiffness against continuum strain energies on a unit
// cube. Affine fields test the Lamé coefficients; quadratic fields also test
// quadrature.
TEST_F(MjCMeshTest, InterpolatedFlexStrainEnergy) {
  for (bool membrane : {false, true}) {
    for (bool quadratic : {false, true}) {
      for (double poisson : {0.0, 0.25}) {
        SCOPED_TRACE(absl::StrFormat("membrane=%d quadratic=%d poisson=%g",
                                     membrane, quadratic, poisson));
        std::string xml =
            absl::StrFormat(R"(
          <mujoco>
            <worldbody>
              <flexcomp name="f" type="grid" count="3 3 3" spacing=".5 .5 .5" dim="3" dof="%s">
                <elasticity young="2" poisson="%g" thickness=".1" %s/>
                <contact selfcollide="none"/>
              </flexcomp>
            </worldbody>
          </mujoco>)",
                            quadratic ? "quadratic" : "trilinear", poisson,
                            membrane ? "elastic2d=\"stretch\"" : "");
        char error[1024];
        MjModelPtr model =
            LoadModelFromString(xml.c_str(), error, sizeof(error));
        ASSERT_THAT(model.get(), testing::NotNull()) << error;
        int order = quadratic ? 2 : 1;
        int nbasis = order + 1;
        int nnode = membrane ? nbasis * nbasis : nbasis * nbasis * nbasis;
        int ndof = 3 * nnode;
        double mu = 1 / (1 + poisson);
        double lambda = membrane
                            ? 2 * poisson / (1 - poisson * poisson)
                            : 2 * poisson / ((1 + poisson) * (1 - 2 * poisson));
        for (int field = 0; field < (quadratic ? 3 : 2); ++field) {
          SCOPED_TRACE(absl::StrFormat("field=%d", field));
          // u_x=x, u_y=x, or u_x=x^2/2. For membranes use each face's in-plane
          // axes.
          for (int face = 0; face < (membrane ? 6 : 1); ++face) {
            int axis0 = membrane ? (face / 2 + 1) % 3 : 0;
            int axis1 = membrane ? (face / 2 + 2) % 3 : 1;
            std::vector<double> u(ndof, 0);
            for (int i = 0; i < nnode; ++i) {
              int ix = i / (membrane ? nbasis : nbasis * nbasis);
              double x = static_cast<double>(ix) / order - 0.5;
              u[3 * i + (field == 1 ? axis1 : axis0)] =
                  field == 2 ? x * x / 2 : x;
            }
            const mjtNum* stiffness = model->flex_stiffness +
                                      model->flex_stiffnessadr[0] +
                                      face * ndof * ndof;
            double energy = 0;
            for (int i = 0; i < ndof; ++i) {
              for (int j = 0; j < ndof; ++j) {
                energy -= 0.5 * u[i] * stiffness[i * ndof + j] * u[j];
              }
            }
            double expected = 0.5 * (field == 1 ? mu : lambda + 2 * mu);
            if (field == 2) expected /= 12;  // integral of x^2 on [-1/2, 1/2]
            if (membrane) expected *= 0.1;   // thickness
            EXPECT_NEAR(energy, expected, MjTol(1e-12, 2e-6));
          }
        }
      }
    }
  }
}

TEST_F(MjCMeshTest, Flex2DElasticityRequiresPositiveThickness) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 1" spacing="1 1 1" dim="2" dof="2d">
        <elasticity young="1" thickness="0" elastic2d="bend"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("2d elasticity requires positive thickness"));
}

TEST_F(MjCMeshTest, InterpolatedFlexSupportsBendElasticityWithWarning) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 2" spacing="1 1 1" dim="3" dof="trilinear">
        <contact selfcollide="none"/>
        <elasticity young="1" thickness="1" elastic2d="bend"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::NotNull()) << error.data();
}

TEST_F(MjCMeshTest, InterpolatedFlexSupportsBothElasticityWithWarning) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 2" spacing="1 1 1" dim="3" dof="trilinear">
        <contact selfcollide="none"/>
        <elasticity young="1" thickness="1" elastic2d="both"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::NotNull()) << error.data();
}

TEST_F(MjCMeshTest, Flex2DElasticityRequires2DFlex) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 3" spacing="1 1 1" dim="3" dof="2d">
        <elasticity young="1" thickness="1" elastic2d="bend"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(), HasSubstr("2d elasticity requires 2d flex"));
}

TEST_F(MjCMeshTest, VolumeNegativeThrowsError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      MESH_DEFINITIONS
    </asset>
    <worldbody>
      <body>
        GEOM_DEFINITIONS
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char bad_mesh[] = R"(
      <mesh name="bad_mesh%d" inertia="exact"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="3 0 2  0 3 1  1 3 2  0 1 2"/>\n"
  )";
  static constexpr char geom[] = R"(<geom type="mesh" mesh="bad_mesh%d"/>\n)";

  for (int nmesh : {3, 16, 17, 50}) {
    std::string mesh_definitions = "";
    for (int i = 1; i < nmesh + 1; i++) {
      mesh_definitions += absl::StrFormat(bad_mesh, i);
    }

    std::string geom_definitions = "";
    for (int i = 1; i < nmesh + 1; i++) {
      geom_definitions += absl::StrFormat(geom, i);
    }

    std::string xml_str = xml;
    absl::StrReplaceAll({{"MESH_DEFINITIONS", mesh_definitions},
                         {"GEOM_DEFINITIONS", geom_definitions}},
                        &xml_str);

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml_str.c_str(), error.data(), error.size());
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(error.data(), HasSubstr("mesh volume is negative"));
  }
}

TEST_F(MjCMeshTest, MeshIgnoresDefaultDensity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <geom density="0" />
    </default>
    <asset>
      <mesh name="a" vertex="0 0 0 1 0 0 0 1 0 0 0 1" scale="10 10 10" />
    </asset>
    <worldbody/>
  </mujoco>)";
  char error[1024];
  mjSpec* spec = mj_parseXMLString(xml, 0, error, sizeof(error));
  EXPECT_THAT(spec, NotNull()) << error;
  mjModel* m1 = mj_compile(spec, nullptr);
  EXPECT_THAT(m1, NotNull());
  mj_deleteModel(m1);
  mjModel* m2 = mj_compile(spec, nullptr);
  EXPECT_THAT(m2, NotNull());
  mj_deleteModel(m2);
  mj_deleteSpec(spec);
}

// ------------- test concave and shell inertia --------------------------------

TEST_F(MjCMeshTest, ExactConcaveInertia) {
  const std::string xml_path = GetTestDataFilePath(kConcaveInertiaPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  // analytic computation of 1x1x1 cube with a .8x.8x.9 hole
  // see https://en.wikipedia.org/wiki/List_of_moments_of_inertia
  mjtNum density = 2.;
  mjtNum m_hole = .9 * .8 * .8 * density;
  mjtNum m_cube = 1. * density;
  mjtNum m_concave_cube = m_cube - m_hole;
  mjtNum I_cube = m_cube / 6.;
  // due to the asymmetric hole, the com position has changed
  // so we need to use https://en.wikipedia.org/wiki/Parallel_axis_theorem
  mjtNum d_cube = .5 - model->body_ipos[5];
  mjtNum d_hole = .55 - model->body_ipos[5];
  mjtNum I1 = I_cube - m_hole * (.8 * .8 + .8 * .8) / 12;
  mjtNum I2 = I_cube - m_hole * (.8 * .8 + .9 * .9) / 12 +
              m_cube * d_cube * d_cube - m_hole * d_hole * d_hole;
  EXPECT_NEAR(model->body_mass[1], m_concave_cube, kMaxAbsErr);
  EXPECT_NEAR(model->body_mass[2], m_concave_cube, kMaxAbsErr);
  EXPECT_NEAR(model->body_mass[3], m_concave_cube, kMaxAbsErr);
  EXPECT_NEAR(model->body_mass[4], m_concave_cube, kMaxAbsErr);
  for (int i = 3; i < 15; i += 3) {
    EXPECT_NEAR(model->body_inertia[i], I1, kMaxAbsErr);
    EXPECT_NEAR(model->body_inertia[i + 1], I2, kMaxAbsErr);
    EXPECT_NEAR(model->body_inertia[i + 2], I2, kMaxAbsErr);
  }
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, ExactConvexInertia) {
  const std::string xml_path = GetTestDataFilePath(kConvexInertiaPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  // https://en.wikipedia.org/wiki/List_of_moments_of_inertia
  mjtNum m_solid_cube = 1.;
  mjtNum I_solid_cube = 1. / 6. * m_solid_cube;
  EXPECT_LE(mju_abs(model->body_mass[1] - m_solid_cube), kMaxAbsErr);
  EXPECT_LE(mju_abs(model->body_mass[2] - m_solid_cube), kMaxAbsErr);
  for (int i = 3; i < 9; i++) {
    EXPECT_LE(mju_abs(model->body_inertia[i] - I_solid_cube), kMaxAbsErr);
  }
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, UnreferencedConvexInertiaMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="orphan" inertia="convex"
            vertex="0 0 1   1 0 0   0 1 0   -1 0 0   0 -1 0"
            face="0 1 2   0 2 3   0 3 4   0 4 1   1 4 3   1 3 2"/>
    </asset>
    <worldbody/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
}

TEST_F(MjCMeshTest, ExactShellInertia) {
  const std::string xml_path = GetTestDataFilePath(kShellInertiaPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  // see https://en.wikipedia.org/wiki/List_of_moments_of_inertia
  mjtNum m_hollow_cube = 6.;
  mjtNum I_hollow_cube = 5. / 18. * m_hollow_cube;
  EXPECT_LE(mju_abs(model->body_mass[1] - m_hollow_cube), kMaxAbsErr);
  EXPECT_LE(mju_abs(model->body_inertia[3] - I_hollow_cube), kMaxAbsErr);
  EXPECT_LE(mju_abs(model->body_inertia[4] - I_hollow_cube), kMaxAbsErr);
  EXPECT_LE(mju_abs(model->body_inertia[5] - I_hollow_cube), kMaxAbsErr);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, MeshPosQuat) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pyramid" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" name="geom1" mesh="pyramid"/>
      <geom type="mesh" name="geom2" pos="1 2 3" quat="0.5 0.5 0.5 0.5" mesh="pyramid"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  // loading the mesh results in an offset of the geom's pos and quat due to the
  // fact that the geom's center is not the volumetric center of the mesh. To
  // recover the geom's originally specified pose, the offset used is stored in
  // mesh_pos and mesh_quat. In order to recover the originally specified pose
  // and orientation, first invert the specified mesh_pos and mesh_quat
  mjtNum inverse_mesh_pos[3];
  mjtNum inverse_mesh_quat[4];
  mju_negPose(inverse_mesh_pos, inverse_mesh_quat, &model->mesh_pos[0],
              &model->mesh_quat[0]);

  // apply the inverted mesh_pos and inverted mesh_quat to the geom's pos and
  // quat. It should match the originally specified values
  mjtNum recovered_pos[3];
  mjtNum recovered_quat[4];
  mju_mulPose(recovered_pos, recovered_quat, &model->geom_pos[0],
              &model->geom_quat[0], inverse_mesh_pos, inverse_mesh_quat);
  EXPECT_NEAR(recovered_pos[0], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[1], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[2], 0, MjTol(1e-12, 1e-6));

  EXPECT_NEAR(recovered_quat[0], 1, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[1], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[2], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[3], 0, MjTol(1e-12, 1e-6));

  // same test on the other geom
  mju_negPose(inverse_mesh_pos, inverse_mesh_quat, &model->mesh_pos[0],
              &model->mesh_quat[0]);
  mju_mulPose(recovered_pos, recovered_quat, &model->geom_pos[3],
              &model->geom_quat[4], inverse_mesh_pos, inverse_mesh_quat);
  EXPECT_NEAR(recovered_pos[0], 1, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[1], 2, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[2], 3, MjTol(1e-12, 1e-6));

  EXPECT_NEAR(recovered_quat[0], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[1], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[2], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[3], 0.5, MjTol(1e-12, 1e-6));
}

TEST_F(MjCMeshTest, MeshPosQuatShellInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pyramid" vertex="0 0 0  1 0 0  0 1 0  0 0 1" inertia="shell"/>
    </asset>
    <worldbody>
      <geom type="mesh" name="geom1" mesh="pyramid"/>
      <geom type="mesh" name="geom2" pos="1 2 3" quat="0.5 0.5 0.5 0.5" mesh="pyramid"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  // loading the mesh results in an offset of the geom's pos and quat due to the
  // fact that the geom's center is not the volumetric center of the mesh. To
  // recover the geom's originally specified pose, the offset used is stored in
  // mesh_pos and mesh_quat. In order to recover the originally specified pose
  // and orientation, first invert the specified mesh_pos and mesh_quat
  mjtNum inverse_mesh_pos[3];
  mjtNum inverse_mesh_quat[4];
  mju_negPose(inverse_mesh_pos, inverse_mesh_quat, &model->mesh_pos[0],
              &model->mesh_quat[0]);

  // apply the inverted mesh_pos and inverted mesh_quat to the geom's pos and
  // quat. It should match the originally specified values
  mjtNum recovered_pos[3];
  mjtNum recovered_quat[4];
  mju_mulPose(recovered_pos, recovered_quat, &model->geom_pos[0],
              &model->geom_quat[0], inverse_mesh_pos, inverse_mesh_quat);
  EXPECT_NEAR(recovered_pos[0], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[1], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[2], 0, MjTol(1e-12, 1e-6));

  EXPECT_NEAR(recovered_quat[0], 1, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[1], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[2], 0, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[3], 0, MjTol(1e-12, 1e-6));

  // same test on the other geom
  mju_negPose(inverse_mesh_pos, inverse_mesh_quat, &model->mesh_pos[0],
              &model->mesh_quat[0]);
  mju_mulPose(recovered_pos, recovered_quat, &model->geom_pos[3],
              &model->geom_quat[4], inverse_mesh_pos, inverse_mesh_quat);
  EXPECT_NEAR(recovered_pos[0], 1, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[1], 2, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_pos[2], 3, MjTol(1e-12, 1e-6));

  EXPECT_NEAR(recovered_quat[0], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[1], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[2], 0.5, MjTol(1e-12, 1e-6));
  EXPECT_NEAR(recovered_quat[3], 0.5, MjTol(1e-12, 1e-6));
}

TEST_F(MjCMeshTest, MeshScale) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pyramid" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="pyramid_scaled" vertex="0 0 0  1 0 0  0 1 0  0 0 1" scale="0.9 1 -1"/>
    </asset>
    <worldbody>
      <geom type="mesh" name="geom1" mesh="pyramid"/>
      <geom type="mesh" name="geom2" mesh="pyramid_scaled"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(AsVector(model->mesh_scale + 0, 3), ElementsAre(1, 1, 1));
  EXPECT_THAT(AsVector(model->mesh_scale + 3, 3), ElementsAre(0.9, 1, -1));
}

TEST_F(MjCMeshTest, UserNormalsAnisotropicScale) {
  // normals are covectors: under mesh scale s they must transform by 1/s.
  // cube with per-vertex normals along the vertex directions, anisotropic
  // scale; expected compiled normals follow (+-1/sx, +-1/sy, +-1/sz)
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" scale=".2 .1 .1"
        vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1  -1 -1 1  1 -1 1  1 1 1  -1 1 1"
        normal="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1  -1 -1 1  1 -1 1  1 1 1  -1 1 1"
        face="0 2 1  0 3 2  4 5 6  4 6 7  0 1 5  0 5 4  1 2 6  1 6 5
              2 3 7  2 7 6  3 0 4  3 4 7"/>
    </asset>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="mesh" mesh="cube"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  const mjModel* m = model.get();
  int mid = 0;
  const mjtNum* mq = m->mesh_quat + 4 * mid;
  const mjtNum* mp = m->mesh_pos + 3 * mid;
  double scale[3] = {.2, .1, .1};
  int va = m->mesh_vertadr[mid];
  int na = m->mesh_normaladr[mid];
  int fa = m->mesh_faceadr[mid];
  for (int i = 0; i < 3 * m->mesh_facenum[mid]; i++) {
    int vid = m->mesh_face[3 * fa + i];
    int nid = m->mesh_facenormal[3 * fa + i];

    // rotate vertex and normal back to the authored frame
    mjtNum vm[3] = {m->mesh_vert[3 * (va + vid) + 0],
                    m->mesh_vert[3 * (va + vid) + 1],
                    m->mesh_vert[3 * (va + vid) + 2]};
    mjtNum nm[3] = {m->mesh_normal[3 * (na + nid) + 0],
                    m->mesh_normal[3 * (na + nid) + 1],
                    m->mesh_normal[3 * (na + nid) + 2]};
    mjtNum v[3], n[3];
    mju_rotVecQuat(v, vm, mq);
    mju_addTo3(v, mp);
    mju_rotVecQuat(n, nm, mq);

    // expected: sign pattern of the vertex, divided by scale, normalized
    mjtNum expected[3];
    for (int k = 0; k < 3; k++) {
      expected[k] = (v[k] > 0 ? 1.0 : -1.0) / scale[k];
    }
    mju_normalize3(expected);
    EXPECT_GT(mju_dot3(n, expected), 0.999);
  }
}

TEST_F(MjCMeshTest, NegativeScaleConvexPolygons) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="normal"
            vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                    -1 -1 1  1 -1 1  1 1 1  -1 1 1"/>
      <mesh name="mirrored" scale="1 -1 1" inertia="convex"
            vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1
                    -1 -1 1  1 -1 1  1 1 1  -1 1 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="normal"/>
      <geom type="mesh" mesh="mirrored"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();

  auto check_normals = [&](const char* name) {
    int mesh_id = mj_name2id(model.get(), mjOBJ_MESH, name);
    int start = model->mesh_vertadr[mesh_id];
    int poly_start = model->mesh_polyadr[mesh_id];
    int poly_num = model->mesh_polynum[mesh_id];
    ASSERT_EQ(poly_num, 6);

    for (int p = 0; p < poly_num; p++) {
      int a = model->mesh_polyvertadr[poly_start + p];
      int num_polyvert = model->mesh_polyvertnum[poly_start + p];
      mjtNum outward[3] = {0, 0, 0};

      for (int i = 0; i < num_polyvert; i++) {
        int v_id = start + model->mesh_polyvert[a + i];
        outward[0] += model->mesh_vert[3 * v_id + 0];
        outward[1] += model->mesh_vert[3 * v_id + 1];
        outward[2] += model->mesh_vert[3 * v_id + 2];
      }

      int normal_idx = 3 * (poly_start + p);
      mjtNum dot = outward[0] * model->mesh_polynormal[normal_idx + 0] +
                   outward[1] * model->mesh_polynormal[normal_idx + 1] +
                   outward[2] * model->mesh_polynormal[normal_idx + 2];
      EXPECT_GT(dot, 0.0) << "Normal is inward for mesh " << name;
    }
  };

  check_normals("normal");
  check_normals("mirrored");
}

TEST_F(MjCMeshTest, PolygonNormalCollinearOutline) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="plate"
            vertex="-0.4 -0.4 -0.01   0.4 -0.4 -0.01
                     0.4  0.4 -0.01  -0.4  0.4 -0.01
                    -0.4 -0.4  0.01   0.4 -0.4  0.01
                     0.4  0.4  0.01   0.0  0.4  0.0101
                    -0.4  0.4  0.01"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="plate"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  ASSERT_EQ(model->nmeshpoly, 6);

  // Rotate polygon normals from mesh inertial frame back to user frame.
  // Specifically check the top (+z) polygon, which has 3 nearly collinear
  // outline vertices at y = 0.4: (0.4, 0.4, 0.01), (0, 0.4, 0.0101), (-0.4,
  // 0.4, 0.01).
  mjtNum max_z = -1;
  for (int p = 0; p < model->nmeshpoly; ++p) {
    mjtNum n_user[3];
    mju_rotVecQuat(n_user, model->mesh_polynormal + 3 * p, model->mesh_quat);
    max_z = mju_max(max_z, n_user[2]);
  }
  EXPECT_NEAR(max_z, 1.0, MjTol(1e-4, 1e-4));
}

TEST_F(MjCMeshTest, PolygonNormalTriangularOutlineOfSeveralFaces) {
  // A triangular prism whose top vertex (0.5, 0.3, 0.10001) is slightly
  // raised: the top polygon is a triangle merged from three faces.
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="prism"
            vertex="0 0 0.1   1 0 0.1   0.5 0.9 0.1   0.5 0.3 0.10001
                    0 0 0     1 0 0     0.5 0.9 0"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="prism"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  ASSERT_EQ(model->nmeshpoly, 5);

  // Rotate polygon normals from mesh inertial frame back to user frame; the
  // top (+z) and bottom (-z) polygons must both point outward.
  mjtNum max_z = -1, min_z = 1;
  for (int p = 0; p < model->nmeshpoly; ++p) {
    mjtNum n_user[3];
    mju_rotVecQuat(n_user, model->mesh_polynormal + 3 * p, model->mesh_quat);
    max_z = mju_max(max_z, n_user[2]);
    min_z = mju_min(min_z, n_user[2]);
  }
  EXPECT_NEAR(max_z, 1.0, MjTol(1e-4, 1e-4));
  EXPECT_NEAR(min_z, -1.0, MjTol(1e-4, 1e-4));
}

TEST_F(MjCMeshTest, NegativeScaleUserMeshCompiles) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh" scale="-1 1 1" inertia="exact"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1  0 3 2  1 3 0  1 2 3" />
      <mesh name="convex_mesh" scale="-1 1 1" inertia="convex"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1  0 3 2  1 3 0  1 2 3" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
      <geom type="mesh" mesh="convex_mesh"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

TEST_F(MjCMeshTest, NegativeScaleUserMeshMatchesPositiveScale) {
  static constexpr char pos_xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pos_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1  0 3 2  1 3 0  1 2 3" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="pos_mesh"/>
    </worldbody>
  </mujoco>
  )";

  static constexpr char neg_xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="neg_mesh" scale="-1 1 1"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1  0 3 2  1 3 0  1 2 3" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="neg_mesh"/>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr pos_model = LoadModelFromString(pos_xml, error, sizeof(error));
  ASSERT_THAT(pos_model.get(), NotNull()) << error;

  MjModelPtr neg_model = LoadModelFromString(neg_xml, error, sizeof(error));
  ASSERT_THAT(neg_model.get(), NotNull()) << error;

  ASSERT_EQ(pos_model->nmeshface, neg_model->nmeshface);

  for (int i = 0; i < pos_model->nmeshface; i++) {
    EXPECT_EQ(pos_model->mesh_face[3 * i + 0], neg_model->mesh_face[3 * i + 0]);
    EXPECT_EQ(pos_model->mesh_face[3 * i + 1], neg_model->mesh_face[3 * i + 2]);
    EXPECT_EQ(pos_model->mesh_face[3 * i + 2], neg_model->mesh_face[3 * i + 1]);
  }
}

TEST_F(MjCMeshTest, ShellInertiaTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="pyramid" vertex="0 0 0  1 0 0  0 1 0  0 0 1" inertia="shell"/>
      <mesh name="pyramid_scaled" vertex="0 0 0  1 0 0  0 1 0  0 0 1" scale="0.9 1 -1"/>
    </asset>
    <worldbody>
      <geom type="mesh" name="geom1" mesh="pyramid"/>
      <geom type="mesh" name="geom2" mesh="pyramid_scaled"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  EXPECT_THAT(AsVector(model->mesh_scale + 0, 3), ElementsAre(1, 1, 1));
  EXPECT_THAT(AsVector(model->mesh_scale + 3, 3), ElementsAre(0.9, 1, -1));
}

// ----------------------------- texcoord -------------------------------------

TEST_F(MjCMeshTest, CreateFaceTexCoord) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            texcoord="0 0  0 0  0 0  0 0"/>
      </asset>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

TEST_F(MjCMeshTest, UseFaceTexCoord) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  2 0 3  1 2 3"
            texcoord="0 0  .1 .1  .2 .2  .3 .3"/>
      </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[0]], .0);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[1]], .2);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[2]], .1);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[3]], .0);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[4]], .1);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[5]], .3);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[6]], .2);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[7]], .0);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[8]], .3);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[9]], .1);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[10]], .2);
  EXPECT_FLOAT_EQ(model->mesh_texcoord[2 * model->mesh_facetexcoord[11]], .3);
}

TEST_F(MjCMeshTest, MissingTexCoord) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            texcoord="0 0"/>
      </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(), HasSubstr("texcoord must be 2*nv"));
}

TEST_F(MjCMeshTest, ObjIncompleteFaceTexCoord) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" file="mesh.obj"/>
    </asset>
  </mujoco>
  )";
  static constexpr char obj[] =
      "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
      "vt 0 0\n"
      "f 1 3/1 2/1\nf 1/1 2/1 4/1\nf 3/1 1/1 4/1\nf 2/1 3/1 4/1\n";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "mesh.obj", obj, sizeof(obj) - 1);
  std::array<char, 1024> error;
  mock_warning_handler.ExpectWarnings("missing face texture coordinate");
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size(), &vfs);
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->mesh_texcoordadr[0], -1);
  EXPECT_EQ(model->nmeshtexcoord, 0);
  mj_deleteVFS(&vfs);
}

TEST_F(MjCMeshTest, ObjOutOfRangeFaceTexCoord) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" file="mesh.obj"/>
    </asset>
  </mujoco>
  )";
  static constexpr char obj[] =
      "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
      "vt \n"
      "f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 3/3 1/1 4/4\nf 2/2 3/3 4/4\n";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "mesh.obj", obj, sizeof(obj) - 1);
  std::array<char, 1024> error;
  mock_warning_handler.ExpectWarnings("invalid or missing face texture");
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size(), &vfs);
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->mesh_texcoordadr[0], -1);
  EXPECT_EQ(model->nmeshtexcoord, 0);
  mj_deleteVFS(&vfs);
}

// ----------------------------- qhull ----------------------------------------

TEST_F(MjCMeshTest, NaNConvexHullDisallowed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh"vertex="nan 0 0  0 1 0  0 1 0  0 0 1"/>
    </asset>
  </mujoco>
    )";
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings();
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), testing::IsNull());
  EXPECT_THAT(error.data(), HasSubstr("vertex coordinate 0 is not finite"));
}

TEST_F(MjCMeshTest, InvalidIndexInFace) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="1 0 0  0 1 0  0 0 1  0.707 0 0.707"
        face="0 2 6  0 3 2" />
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("in face 0, vertex index 6 does not exist"));
}

TEST_F(MjCMeshTest, CacheRespectsSmoothNormal) {
  constexpr char xml[] = R"(
    <mujoco>
      <asset><mesh name="cube" file="%s" smoothnormal="%s"/></asset>
      <worldbody><geom type="mesh" mesh="cube"/></worldbody>
    </mujoco>)";
  std::string path = GetTestDataFilePath("user/testdata/cube.stl");
  MjModelPtr hard = LoadModelFromString(absl::StrFormat(xml, path, "false"));
  MjModelPtr smooth = LoadModelFromString(absl::StrFormat(xml, path, "true"));
  ASSERT_THAT(hard.get(), NotNull());
  ASSERT_THAT(smooth.get(), NotNull());
  EXPECT_GT(mj_getCacheSize(mj_getCache()), 0);
  EXPECT_NE(AsVector(hard->mesh_normal, 3 * hard->nmeshnormal),
            AsVector(smooth->mesh_normal, 3 * smooth->nmeshnormal));
}

TEST_F(MjCMeshTest, QhullCache) {
  static constexpr char xml1[] = R"(
    <mujoco>
      <asset>
        <mesh name="box" file="cube_complete.obj"/>
      </asset>
      <worldbody>
        <geom type="mesh" pos="0 0 2" mesh="box" contype="0" conaffinity="0"/>
        <geom type="mesh" pos="0 0 0" mesh="box" contype="0" conaffinity="0"/>
      </worldbody>
    </mujoco>)";

  static constexpr char xml2[] = R"(
      <mujoco>
        <asset>
          <mesh name="box" file="cube_complete.obj"/>
        </asset>
        <worldbody>
          <geom type="mesh" pos="0 0 2" mesh="box"/>
          <geom type="mesh" pos="0 0 0" mesh="box"/>
        </worldbody>
      </mujoco>)";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addFileVFS(&vfs, "", GetTestDataFilePath(kCubeCompletePath).c_str());

  std::array<char, 1000> error;
  MjModelPtr model =
      LoadModelFromString(xml1, error.data(), error.size(), &vfs);
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  EXPECT_THAT(model->mesh_graphadr[0], -1);

  model = LoadModelFromString(xml2, error.data(), error.size(), &vfs);
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  EXPECT_GT(model->mesh_graphadr[0], -1);
  mj_deleteVFS(&vfs);
}

TEST_F(MjCMeshTest, ColocatedMeshError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
            vertex="0 0 0  0 0 0  0 0 0  0 0 0"
            face="0 1 2   1 3 2"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("colocated"));
}

TEST_F(MjCMeshTest, CollinearMeshError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
            vertex="0 0 0  1 0 0  2 0 0  3 0 0"
            face="0 1 2   1 3 2"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("collinear"));
}

TEST_F(MjCMeshTest, CoplanarMeshError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="flat_quad"
            vertex="-5 -5 0   5 -5 0   -5 5 0   5 5 0"
            face="0 1 2   1 3 2"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="flat_quad"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("coplanar"));
}

TEST_F(MjCMeshTest, LoadSkin) {
  const std::string xml_path = GetTestDataFilePath(kCubeSkinPath);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(xml_path.c_str(), 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  mjModel* m1 = mj_compile(spec, 0);
  EXPECT_THAT(m1, NotNull());
  mj_deleteModel(m1);
  mjModel* m2 = mj_compile(spec, 0);
  EXPECT_THAT(m2, NotNull());
  mj_deleteModel(m2);
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, LoadSKNMalformed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <skin name="skin" file="test.skn"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom type="box" size="1 1 1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // Test 1: Integer overflow in ntexcoord (b/511889379)
  {
    uint32_t evil_skn[7] = {1, 0x20000000, 0, 0, 0, 0, 0};
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "test.skn", evil_skn, sizeof(evil_skn));

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml, error.data(), error.size(), &vfs);
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(error.data(),
                HasSubstr("too large sizes in SKN file 'test.skn'"));
    mj_deleteVFS(&vfs);
  }

  // Test 2: Integer overflow in nvert
  {
    uint32_t evil_skn[4] = {0x20000000, 0, 0, 0};
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "test.skn", evil_skn, sizeof(evil_skn));

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml, error.data(), error.size(), &vfs);
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(error.data(),
                HasSubstr("too large sizes in SKN file 'test.skn'"));
    mj_deleteVFS(&vfs);
  }

  // Test 3: Insufficient data in buffer
  {
    uint32_t evil_skn[7] = {10, 0, 0, 0, 0, 0, 0};
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "test.skn", evil_skn, sizeof(evil_skn));

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml, error.data(), error.size(), &vfs);
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(error.data(),
                HasSubstr("insufficient data in SKN file 'test.skn'"));
    mj_deleteVFS(&vfs);
  }

  // Test 4: Negative size in header
  {
    int evil_skn[4] = {-1, 0, 0, 0};
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "test.skn", evil_skn, sizeof(evil_skn));

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml, error.data(), error.size(), &vfs);
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(error.data(),
                HasSubstr("negative size in header of SKN file 'test.skn'"));
    mj_deleteVFS(&vfs);
  }

  // Test 5: Integer overflow in bone vertex count (vcount)
  {
    // Header: nvert=0, ntexcoord=0, nface=0, nbone=1 (16 bytes)
    // Bone: name[40] (40B), bindpos[3] (12B), bindquat[4] (16B),
    // vcount=0x40000000 (4B) Total size = 16 + 72 = 88 bytes
    std::vector<uint8_t> evil_skn(88, 0);
    uint32_t header[4] = {0, 0, 0, 1};
    std::memcpy(evil_skn.data(), header, sizeof(header));
    uint32_t vcount = 0x40000000;
    std::memcpy(evil_skn.data() + 16 + 40 + 12 + 16, &vcount, sizeof(vcount));

    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "test.skn", evil_skn.data(), evil_skn.size());

    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml, error.data(), error.size(), &vfs);
    EXPECT_THAT(model.get(), IsNull());
    EXPECT_THAT(
        error.data(),
        HasSubstr("insufficient vertex data in SKN file 'test.skn', bone 0"));
    mj_deleteVFS(&vfs);
  }
}

TEST_F(MjCMeshTest, SkinFaceIndexOutOfRange) {
  static constexpr char xml_high[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom type="box" size="1 1 1"/>
      </body>
    </worldbody>
    <deformable>
      <skin name="skin" vertex="0 0 0 1 0 0 0 1 0" face="0 1 99">
        <bone body="body" bindpos="0 0 0" bindquat="1 0 0 0"
              vertid="0 1 2" vertweight="1 1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model_high =
      LoadModelFromString(xml_high, error.data(), error.size());
  EXPECT_THAT(model_high.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("face vertex index"));

  static constexpr char xml_neg[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom type="box" size="1 1 1"/>
      </body>
    </worldbody>
    <deformable>
      <skin name="skin" vertex="0 0 0 1 0 0 0 1 0" face="0 -1 2">
        <bone body="body" bindpos="0 0 0" bindquat="1 0 0 0"
              vertid="0 1 2" vertweight="1 1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  MjModelPtr model_neg =
      LoadModelFromString(xml_neg, error.data(), error.size());
  EXPECT_THAT(model_neg.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("face vertex index"));
}

// ------------- test octree ---------------------------------------------------

TEST_F(MjCMeshTest, Octree) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(xml_path.c_str(), 0, error.data(), error.size());
  mjsGeom* geom = mjs_asGeom(mjs_firstElement(spec, mjOBJ_GEOM));
  geom->type = mjGEOM_SDF;
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_GT(model->mesh_octnum[0], 0);
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

namespace {
bool AreAabbsAdjacent(const mjtNum* aabb1, const mjtNum* aabb2) {
  const double kEps = 1e-6;
  int touching_dims = 0;
  int overlapping_dims = 0;

  for (int dim = 0; dim < 3; ++dim) {
    const mjtNum center1 = aabb1[dim];
    const mjtNum half_size1 = aabb1[dim + 3];
    const mjtNum center2 = aabb2[dim];
    const mjtNum half_size2 = aabb2[dim + 3];
    const mjtNum gap = std::abs(center1 - center2) - (half_size1 + half_size2);
    if (std::abs(gap) < kEps) {
      touching_dims++;
    } else if (gap < -kEps) {
      overlapping_dims++;
    }
  }

  return touching_dims == 1 && overlapping_dims == 2;
}
}  // namespace

TEST_F(MjCMeshTest, OctreeIsBalanced) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(xml_path.c_str(), 0, error.data(), error.size());
  mjsGeom* geom = mjs_asGeom(mjs_firstElement(spec, mjOBJ_GEOM));
  geom->type = mjGEOM_SDF;
  mjsMesh* mesh = mjs_asMesh(mjs_firstElement(spec, mjOBJ_MESH));
  mesh->octree_maxdepth = 5;
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_GT(model->mesh_octnum[0], 0);

  const int octree_adr = model->mesh_octadr[0];
  const int noct = model->mesh_octnum[0];

  std::vector<int> leaves;
  for (int i = 0; i < noct; ++i) {
    bool is_leaf = true;
    for (int j = 0; j < 8; ++j) {
      if (model->oct_child[(octree_adr + i) * 8 + j] != -1) {
        is_leaf = false;
        break;
      }
    }
    if (is_leaf) {
      leaves.push_back(i);
    }
  }

  int unbalanced_pairs = 0;
  for (int i = 0; i < leaves.size(); ++i) {
    for (int j = i + 1; j < leaves.size(); ++j) {
      const int node1_idx = leaves[i];
      const int node2_idx = leaves[j];
      const mjtNum* aabb1 = &model->oct_aabb[(octree_adr + node1_idx) * 6];
      const mjtNum* aabb2 = &model->oct_aabb[(octree_adr + node2_idx) * 6];

      if (AreAabbsAdjacent(aabb1, aabb2)) {
        const int level1 = model->oct_depth[octree_adr + node1_idx];
        const int level2 = model->oct_depth[octree_adr + node2_idx];
        if (std::abs(level1 - level2) > 1) {
          if (unbalanced_pairs < 10) {
            ADD_FAILURE() << "Nodes " << node1_idx << " (level " << level1
                          << ") and " << node2_idx << " (level " << level2
                          << ") are not balanced."
                          << "\nAABB1: center=(" << aabb1[0] << ", " << aabb1[1]
                          << ", " << aabb1[2] << "), half_size=(" << aabb1[3]
                          << ", " << aabb1[4] << ", " << aabb1[5] << ")"
                          << "\nAABB2: center=(" << aabb2[0] << ", " << aabb2[1]
                          << ", " << aabb2[2] << "), half_size=(" << aabb2[3]
                          << ", " << aabb2[4] << ", " << aabb2[5] << ")";
          }
          unbalanced_pairs++;
        }
      }
    }
  }

  EXPECT_EQ(unbalanced_pairs, 0)
      << "Found " << unbalanced_pairs << " unbalanced adjacent leaf pairs.";

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, OctreeHangingNodeInterpolation) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(xml_path.c_str(), 0, error.data(), error.size());
  mjsGeom* geom = mjs_asGeom(mjs_firstElement(spec, mjOBJ_GEOM));
  geom->type = mjGEOM_SDF;
  mjsMesh* mesh = mjs_asMesh(mjs_firstElement(spec, mjOBJ_MESH));
  mesh->octree_maxdepth = 5;
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_GT(model->mesh_octnum[0], 0);
  const mjtNum kEps = MjTol(1e-6, 1e-4);

  const int octree_adr = model->mesh_octadr[0];
  const int noct = model->mesh_octnum[0];
  const mjtNum* sdf = model->oct_coeff + octree_adr * 8;

  // find all leaves in the octree
  std::vector<int> leaves;
  for (int i = 0; i < noct; ++i) {
    bool is_leaf = true;
    for (int j = 0; j < 8; ++j) {
      if (model->oct_child[(octree_adr + i) * 8 + j] != -1) {
        is_leaf = false;
        break;
      }
    }
    if (is_leaf) {
      leaves.push_back(i);
    }
  }

  // do a n^2 check of all pairs of leaves in the octree
  // for each pair, check if they are adjacent and if so, check that all hanging
  // nodes within the octree can be interpolated from their parent nodes
  int hanging_nodes_checked = 0;
  int interpolation_failures = 0;
  for (int i = 0; i < leaves.size(); ++i) {
    for (int j = i + 1; j < leaves.size(); ++j) {
      const int node1_idx = leaves[i];
      const int node2_idx = leaves[j];
      const mjtNum* aabb1 = &model->oct_aabb[(octree_adr + node1_idx) * 6];
      const mjtNum* aabb2 = &model->oct_aabb[(octree_adr + node2_idx) * 6];

      if (AreAabbsAdjacent(aabb1, aabb2)) {
        const int level1 = model->oct_depth[octree_adr + node1_idx];
        const int level2 = model->oct_depth[octree_adr + node2_idx];

        if (level1 == level2) {
          continue;
        }

        // decide which node is finer and which is coarser
        const int finer_node_idx = (level1 > level2) ? node1_idx : node2_idx;
        const int coarser_node_idx = (level1 > level2) ? node2_idx : node1_idx;
        const mjtNum* coarser_aabb =
            &model->oct_aabb[(octree_adr + coarser_node_idx) * 6];
        const mjtNum* finer_aabb =
            &model->oct_aabb[(octree_adr + finer_node_idx) * 6];

        mjtNum coarser_corners[8][3];
        for (int c = 0; c < 8; ++c) {
          int sx = (c & 1) ? 1 : -1;
          int sy = (c & 2) ? 1 : -1;
          int sz = (c & 4) ? 1 : -1;
          coarser_corners[c][0] = coarser_aabb[0] + sx * coarser_aabb[3];
          coarser_corners[c][1] = coarser_aabb[1] + sy * coarser_aabb[4];
          coarser_corners[c][2] = coarser_aabb[2] + sz * coarser_aabb[5];
        }

        // for all vertices in the finer node, check if they are hanging and
        // can be interpolated
        for (int v_idx = 0; v_idx < 8; ++v_idx) {
          mjtNum v_pos[3];
          int sx = (v_idx & 1) ? 1 : -1;
          int sy = (v_idx & 2) ? 1 : -1;
          int sz = (v_idx & 4) ? 1 : -1;
          v_pos[0] = finer_aabb[0] + sx * finer_aabb[3];
          v_pos[1] = finer_aabb[1] + sy * finer_aabb[4];
          v_pos[2] = finer_aabb[2] + sz * finer_aabb[5];

          // skip finer vertices that are also coarse corners
          bool is_coarse_corner = false;
          for (int c = 0; c < 8; ++c) {
            if (mju_dist3(v_pos, coarser_corners[c]) < 1e-6) {
              is_coarse_corner = true;
              break;
            }
          }
          if (is_coarse_corner) {
            continue;
          }

          // skip vertices that are not on the boundary
          mjtNum p_local[3];
          bool outside = false;
          for (int d = 0; d < 3; ++d) {
            p_local[d] = (v_pos[d] - coarser_aabb[d]) / coarser_aabb[d + 3];
            if (std::abs(p_local[d]) > 1.0 + kEps) {
              outside = true;
              break;
            }
          }
          if (outside) {
            continue;
          }

          // count the number of dimensions that are on the boundary
          int num_dim = 0;
          for (int d = 0; d < 3; ++d) {
            if (std::abs(p_local[d] - 1.0) < kEps) {
              num_dim++;
            } else if (std::abs(p_local[d] + 1.0) < kEps) {
              num_dim++;
            }
          }

          double interpolated_sdf = 0;
          const mjtNum* coarser_sdf = sdf + coarser_node_idx * 8;

          // for edge or face nodes, try to interpolate the hanging nodes
          if (num_dim == 1 || num_dim == 2) {
            for (int k = 0; k < 8; ++k) {
              int sx = (k & 1) ? 1 : -1;
              int sy = (k & 2) ? 1 : -1;
              int sz = (k & 4) ? 1 : -1;
              double weight = (1 + p_local[0] * sx) / 2.0 *
                              (1 + p_local[1] * sy) / 2.0 *
                              (1 + p_local[2] * sz) / 2.0;
              if (weight > kEps && num_dim == 1) {
                ASSERT_NEAR(weight, 0.25, kEps);
              } else if (weight > kEps && num_dim == 2) {
                ASSERT_NEAR(weight, 0.5, kEps);
              }
              interpolated_sdf += weight * coarser_sdf[k];
            }
          } else {
            continue;
          }

          // if the values do not match, log an error
          const mjtNum* finer_sdf = sdf + finer_node_idx * 8;
          if (std::abs(finer_sdf[v_idx] - interpolated_sdf) > kEps) {
            if (interpolation_failures < 10) {
              EXPECT_NEAR(finer_sdf[v_idx], interpolated_sdf, kEps);
            }
            interpolation_failures++;
          }
          hanging_nodes_checked++;
        }
      }
    }
  }
  EXPECT_GT(hanging_nodes_checked, 0);
  EXPECT_EQ(interpolation_failures, 0)
      << "Found " << interpolation_failures
      << " hanging node interpolation failures.";

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MjCMeshTest, OctreeNotComputedForNonSDF) {
  const std::string xml_path = GetTestDataFilePath(kTorusPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_EQ(model->noct, 0);
  mj_deleteModel(model);
}

mjtNum CubeSDF(mjtNum p[3], mjtNum b[3]) {
  mjtNum q[3] = {mju_abs(p[0]) - b[0], mju_abs(p[1]) - b[1],
                 mju_abs(p[2]) - b[2]};
  return mju_sqrt(std::pow(std::max(q[0], (mjtNum)0), 2) +
                  std::pow(std::max(q[1], (mjtNum)0), 2) +
                  std::pow(std::max(q[2], (mjtNum)0), 2)) +
         std::min(std::max(q[0], std::max(q[1], q[2])), (mjtNum)0);
}

TEST_F(MjCMeshTest, OctreeCube) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" vertex="-1 -1 -1  1 -1 -1  -1 1 -1  1 1 -1
                                -1 -1  1  1 -1  1  -1 1  1  1 1  1"/>
    </asset>
    <worldbody>
      <geom mesh="mesh" type="sdf"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->noct, 63497);
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m.get(), d.get());
  mjSDF sdf;
  int instance = 0;
  mjtGeom geomtype = mjGEOM_SDF;
  const mjpPlugin* sdf_ptr = NULL;
  sdf.id = &instance;
  sdf.type = mjSDFTYPE_SINGLE;
  sdf.plugin = &sdf_ptr;
  sdf.geomtype = &geomtype;
  mjtNum pnt[3][3] = {{1, 1, 1}, {.5, .5, .5}, {.5, 0, 0}};
  mjtNum size[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    EXPECT_NEAR(mjc_distance(m.get(), d.get(), &sdf, pnt[i]),
                CubeSDF(pnt[i], size), 1e-1)
        << "i = " << i;
  }
}

TEST_F(MjCMeshTest, HemisphereSizes) {
  static constexpr char xml[] = R"(
  <mujoco model="makemesh">
    <asset>
      <mesh name="h0" builtin="hemisphere" params="0"/>
      <mesh name="h1" builtin="hemisphere" params="1"/>
      <mesh name="h2" builtin="hemisphere" params="2"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->mesh_vertnum[0], 2 * (0 + 1) * (0 + 2) + 2);
  EXPECT_EQ(model->mesh_vertnum[1], 2 * (1 + 1) * (1 + 2) + 2);
  EXPECT_EQ(model->mesh_vertnum[2], 2 * (2 + 1) * (2 + 2) + 2);
  EXPECT_EQ(model->mesh_facenum[0], 4 * (0 + 1) * (0 + 2));
  EXPECT_EQ(model->mesh_facenum[1], 4 * (1 + 1) * (1 + 2));
  EXPECT_EQ(model->mesh_facenum[2], 4 * (2 + 1) * (2 + 2));
}

TEST_F(MjCMeshTest, SphereSizes) {
  static constexpr char xml[] = R"(
  <mujoco model="makemesh">
    <asset>
      <mesh name="h0" builtin="sphere" params="0"/>
      <mesh name="h1" builtin="sphere" params="1"/>
      <mesh name="h2" builtin="sphere" params="2"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->mesh_vertnum[0], 2 + 10 * std::pow(4, 0));
  EXPECT_EQ(model->mesh_vertnum[1], 2 + 10 * std::pow(4, 1));
  EXPECT_EQ(model->mesh_vertnum[2], 2 + 10 * std::pow(4, 2));
  EXPECT_EQ(model->mesh_facenum[0], 20 * std::pow(4, 0));
  EXPECT_EQ(model->mesh_facenum[1], 20 * std::pow(4, 1));
  EXPECT_EQ(model->mesh_facenum[2], 20 * std::pow(4, 2));
}

TEST_F(MjCMeshTest, MeshMaterial) {
  static constexpr char xml[] = R"(
  <mujoco model="mesh_material">
    <asset>
      <mesh name="h0" builtin="sphere" params="0" material="m1"/>
      <material name="m0" rgba="1 0 0 1"/>
      <material name="m1" rgba="1 1 0 1"/>
    </asset>

    <worldbody>
      <geom name="g0" type="mesh" mesh="h0"/>
      <geom name="g1" type="mesh" mesh="h0" material="m0"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->geom_matid[0], 1);
  EXPECT_EQ(model->geom_matid[1], 0);
}

// generate a procedural mesh in a new spec using the public API
static mjsMesh* MakeBuiltinMesh(mjSpec* spec, mjtMeshBuiltin builtin,
                                std::vector<double> params) {
  mjsMesh* mesh = mjs_addMesh(spec, nullptr);
  if (mjs_makeMesh(mesh, builtin, params.data(), params.size())) {
    return nullptr;
  }
  return mesh;
}

TEST_F(MjCMeshTest, MakeGrid2D) {
  mjSpec* spec = mj_makeSpec();
  mjsMesh* mesh =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_GRID, {3, 4, 1, 0.1, 0.2, 0, 2});
  ASSERT_THAT(mesh, NotNull());
  const std::vector<double>& node = *mesh->usernode;
  const std::vector<int>& face = *mesh->userface;
  ASSERT_EQ(node.size(), 3 * 4 * 3);
  ASSERT_EQ(face.size(), 2 * 3 * 2 * 3);  // 2*3 quads * 2 tris * 3 verts
  EXPECT_TRUE(mesh->usertet->empty());

  // nodes are stored in double precision, ordered with x outermost
  EXPECT_EQ(node[0], 0.1 * -1.0);
  EXPECT_EQ(node[1], 0.2 * -1.5);
  EXPECT_EQ(node[2], 0.0);
  EXPECT_EQ(node[3 * 11 + 0], 0.1 * 1.0);
  EXPECT_EQ(node[3 * 11 + 1], 0.2 * 1.5);

  // first quad (ix=0, iy=0) has vertices {0, 4, 5, 1}
  EXPECT_THAT(std::vector<int>(face.begin(), face.begin() + 6),
              ElementsAre(0, 4, 5, 0, 5, 1));
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, MakeGrid3D) {
  mjSpec* spec = mj_makeSpec();
  mjsMesh* mesh =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_GRID, {2, 2, 2, 1, 1, 1, 3});
  ASSERT_THAT(mesh, NotNull());
  const std::vector<double>& node = *mesh->usernode;
  const std::vector<int>& tet = *mesh->usertet;
  ASSERT_EQ(node.size(), 8 * 3);
  ASSERT_EQ(tet.size(), 6 * 4);  // 1 cube = 6 tets * 4 indices
  EXPECT_TRUE(mesh->userface->empty());
  EXPECT_THAT(std::vector<double>(node.begin(), node.begin() + 6),
              ElementsAre(-0.5, -0.5, -0.5, -0.5, -0.5, 0.5));
  EXPECT_THAT(std::vector<int>(tet.begin(), tet.begin() + 8),
              ElementsAre(0, 2, 4, 3, 0, 4, 1, 3));
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, MakeDisc) {
  mjSpec* spec = mj_makeSpec();
  mjsMesh* mesh =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_DISC, {3, 3, 1, 1, 1, 0});
  ASSERT_THAT(mesh, NotNull());
  const std::vector<double>& node = *mesh->usernode;
  const std::vector<int>& face = *mesh->userface;
  ASSERT_EQ(node.size(), 9 * 3);
  ASSERT_EQ(face.size(), 2 * 2 * 2 * 3);

  // corner (-1, -1) is projected radially onto the unit circle
  EXPECT_DOUBLE_EQ(node[0], -std::sqrt(0.5));
  EXPECT_DOUBLE_EQ(node[1], -std::sqrt(0.5));
  // edge midpoint (-1, 0) is unchanged
  EXPECT_DOUBLE_EQ(node[3], -1.0);
  EXPECT_DOUBLE_EQ(node[4], 0.0);
  // center stays at the origin
  EXPECT_DOUBLE_EQ(node[12], 0.0);
  EXPECT_DOUBLE_EQ(node[13], 0.0);

  // quad (0, 0) is not flipped, quad (0, 1) is flipped
  EXPECT_THAT(std::vector<int>(face.begin(), face.begin() + 12),
              ElementsAre(0, 3, 4, 0, 4, 1, 1, 4, 2, 4, 5, 2));
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, MakeBoxCylinderEllipsoid) {
  mjSpec* spec = mj_makeSpec();
  std::vector<double> params3d = {3, 3, 3, 0.5, 0.5, 0.5, 3};

  mjsMesh* box2d =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_BOX, {3, 3, 3, 0.5, 0.5, 0.5, 2});
  ASSERT_THAT(box2d, NotNull());
  EXPECT_FALSE(box2d->userface->empty());
  EXPECT_TRUE(box2d->usertet->empty());
  // no center node for surfaces
  EXPECT_THAT(std::vector<double>(box2d->usernode->begin(),
                                  box2d->usernode->begin() + 3),
              ElementsAre(-0.5, -0.5, -0.5));

  mjsMesh* box3d = MakeBuiltinMesh(spec, mjMESH_BUILTIN_BOX, params3d);
  ASSERT_THAT(box3d, NotNull());
  const std::vector<double>& node = *box3d->usernode;
  const std::vector<int>& tet = *box3d->usertet;
  ASSERT_FALSE(tet.empty());
  EXPECT_TRUE(box3d->userface->empty());
  // center node followed by the first corner
  EXPECT_THAT(std::vector<double>(node.begin(), node.begin() + 6),
              ElementsAre(0, 0, 0, -0.5, -0.5, -0.5));
  // first tet connects the center to the first triangle of side iz=0
  EXPECT_THAT(std::vector<int>(tet.begin(), tet.begin() + 4),
              ElementsAre(0, 1, 4, 5));

  mjsMesh* cyl = MakeBuiltinMesh(spec, mjMESH_BUILTIN_CYLINDER, params3d);
  ASSERT_THAT(cyl, NotNull());
  EXPECT_EQ(*cyl->usertet, tet);
  EXPECT_DOUBLE_EQ((*cyl->usernode)[3], -0.5 * std::sqrt(0.5));
  EXPECT_DOUBLE_EQ((*cyl->usernode)[4], -0.5 * std::sqrt(0.5));
  EXPECT_DOUBLE_EQ((*cyl->usernode)[5], -0.5);

  mjsMesh* ell = MakeBuiltinMesh(spec, mjMESH_BUILTIN_ELLIPSOID, params3d);
  ASSERT_THAT(ell, NotNull());
  EXPECT_EQ(*ell->usertet, tet);
  EXPECT_DOUBLE_EQ((*ell->usernode)[3], -0.5 / std::sqrt(3.0));
  EXPECT_DOUBLE_EQ((*ell->usernode)[4], -0.5 / std::sqrt(3.0));
  EXPECT_DOUBLE_EQ((*ell->usernode)[5], -0.5 / std::sqrt(3.0));
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, MakeBuiltinInvalidParams) {
  struct Case {
    mjtMeshBuiltin builtin;
    std::vector<double> params;
    const char* error;
  };
  std::vector<Case> cases = {
      // wrong number of parameters
      {mjMESH_BUILTIN_GRID, {3, 3, 3, 1, 1}, "requires 6 or 7 parameters"},
      {mjMESH_BUILTIN_BOX, {3, 3, 3, 1, 1}, "requires 6 or 7 parameters"},
      {mjMESH_BUILTIN_SQUARE, {3, 3, 1, 1, 1, 0, 2}, "requires 6 parameters"},
      // invalid dim
      {mjMESH_BUILTIN_GRID, {3, 3, 3, 1, 1, 1, 4}, "dim must be 2 or 3"},
      {mjMESH_BUILTIN_GRID, {5, 1, 1, 1, 0, 0, 1}, "dim must be 2 or 3"},
      {mjMESH_BUILTIN_BOX, {3, 3, 3, 1, 1, 1, 1}, "dim must be 2 or 3"},
      // grid counts must be positive and 1 beyond dim
      {mjMESH_BUILTIN_GRID, {0, 3, 3, 1, 1, 1, 3}, "count must be positive"},
      {mjMESH_BUILTIN_GRID, {3, 3, 2, 1, 1, 1, 2}, "beyond dim"},
      // square and disc are 2D
      {mjMESH_BUILTIN_SQUARE, {3, 3, 0, 1, 1, 1}, "count[2] must be 1"},
      {mjMESH_BUILTIN_DISC, {3, 3, 2, 1, 1, 1}, "count[2] must be 1"},
      // boxes need at least two points per side
      {mjMESH_BUILTIN_BOX, {3, 1, 3, 1, 1, 1, 3}, "at least 2"},
      {mjMESH_BUILTIN_CYLINDER, {3, 3, 1, 1, 1, 1, 2}, "at least 2"},
      // cone nvert >= 3 and radius in [0, 1]
      {mjMESH_BUILTIN_CONE, {2, 0.5}, "at least 3"},
      {mjMESH_BUILTIN_CONE, {-2, 0.5}, "at least 3"},
      {mjMESH_BUILTIN_CONE, {4, -0.1}, "must be in [0, 1]"},
      {mjMESH_BUILTIN_CONE, {4, 1.5}, "must be in [0, 1]"},
      {mjMESH_BUILTIN_CONE,
       {4, std::numeric_limits<double>::quiet_NaN()},
       "must be in [0, 1]"},
  };
  for (const Case& c : cases) {
    mjSpec* spec = mj_makeSpec();
    EXPECT_THAT(MakeBuiltinMesh(spec, c.builtin, c.params), IsNull())
        << c.error;
    EXPECT_THAT(mjs_getError(spec), HasSubstr(c.error));
    mj_deleteSpec(spec);
  }
}

TEST_F(MjCMeshTest, MakeBuiltinShellInertia) {
  mjSpec* spec = mj_makeSpec();
  // surfaces have no volume, legacy inertia is changed to shell
  mjsMesh* square =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_SQUARE, {3, 3, 1, 1, 1, 0});
  ASSERT_THAT(square, NotNull());
  EXPECT_EQ(square->inertia, mjMESH_INERTIA_SHELL);
  // volumes keep the default inertia
  mjsMesh* box =
      MakeBuiltinMesh(spec, mjMESH_BUILTIN_BOX, {3, 3, 3, 1, 1, 1, 3});
  ASSERT_THAT(box, NotNull());
  EXPECT_EQ(box->inertia, mjMESH_INERTIA_LEGACY);
  mj_deleteSpec(spec);
}

TEST_F(MjCMeshTest, BuiltinProceduralXML) {
  static constexpr char xml[] = R"(
  <mujoco model="procedural_mesh">
    <asset>
      <mesh name="m_grid3d" builtin="grid" params="3 3 3 0.1 0.1 0.1 3"/>
      <mesh name="m_grid2d" builtin="grid" params="3 3 1 0.1 0.1 0 2"/>
      <mesh name="m_box" builtin="box" params="3 3 3 0.1 0.1 0.1 3"/>
      <mesh name="m_cylinder" builtin="cylinder" params="3 3 3 0.1 0.1 0.1 3"/>
      <mesh name="m_ellipsoid" builtin="ellipsoid" params="3 3 3 0.1 0.1 0.1 3"/>
      <mesh name="m_square" builtin="square" params="3 3 1 0.1 0.1 0"/>
      <mesh name="m_disc" builtin="disc" params="3 3 1 0.1 0.1 0"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->nmesh, 7);
}

}  // namespace
}  // namespace mujoco
