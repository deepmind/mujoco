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

// Tests for the MJZ encoder plugin.

#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/match.h"

// Disable unused function warnings for miniz.
// NOLINTBEGIN(misc-include-cleaner) - Disable clang-tidy for miniz.
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "miniz.h"
#if defined(__GNUC__) || defined(__clang__)
  #pragma GCC diagnostic pop
#endif
// NOLINTEND(misc-include-cleaner) - End clang-tidy disable for miniz.
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_numeric_format.h"
#include "test/compare_model.h"
#include "test/fixture.h"

namespace mujoco {

namespace fs = std::filesystem;
using ::testing::NotNull;

static mjSpec* MakeSimpleSpec() {
  mjSpec* s = mj_makeSpec();
  mjsBody* world = mjs_findBody(s, "world");
  mjsBody* body = mjs_addBody(world, nullptr);
  mjsGeom* geom = mjs_addGeom(body, nullptr);
  geom->size[0] = 1.0;
  geom->size[1] = 1.0;
  geom->size[2] = 1.0;
  return s;
}

static std::string SanitizePathForTestName(const std::string& path) {
  fs::path p(path);
  std::string name = p.stem().string();
  std::string full_name = p.string();
  for (std::string_view marker : {"third_party/mujoco/", "_main/"}) {
    size_t pos = full_name.find(marker);
    if (pos != std::string::npos) {
      full_name = full_name.substr(pos + marker.size());
      break;
    }
  }
  std::string sanitized;
  for (char c : full_name) {
    if (std::isalnum(c)) {
      sanitized += c;
    } else {
      sanitized += '_';
    }
  }
  return sanitized;
}

std::vector<std::string> GetWriteReadTestModels() {
  std::vector<std::string> models;
  std::string ext(".xml");
  for (const auto& path : {GetTestDataFilePath("."), GetModelPath(".")}) {
    for (const auto& p : std::filesystem::recursive_directory_iterator(path)) {
      if (p.path().extension() == ext) {
        // generic format, so patterns containing '/' also match on Windows
        std::string xml = p.path().generic_string();
        if (  // intentional parse or compile failure tests
            absl::StrContains(xml, "malformed_") ||
            absl::StrContains(xml, "_fail") ||
            absl::StrContains(xml, "xml/testdata/parent_") ||
            // large SDF, tetrahedral flex, or benchmark models too slow under
            // sanitizers
            absl::StrContains(xml, "cow") || absl::StrContains(xml, "shark_") ||
            absl::StrContains(xml, "perf") ||
            absl::StrContains(xml, "100_humanoids") ||
            // USD assets (.usda) are not MJCF files
            absl::StrContains(xml, "usd") ||
            // references in-memory VFS files created only in specific unit
            // tests
            absl::StrContains(xml, "mixed_test")) {
          continue;
        }
        models.push_back(xml);
      }
    }
  }
  return models;
}

using MjzEncoderTest = MujocoTest;

TEST_F(MjzEncoderTest, EncoderIsRegistered) {
  const mjpEncoder* enc = mjp_findEncoder("model.mjz", nullptr);
  ASSERT_THAT(enc, testing::NotNull());
  ASSERT_THAT(enc->encode, testing::NotNull());
}

TEST_F(MjzEncoderTest, EncodeReturnsPositiveByteCount) {
  mjSpec* spec = MakeSimpleSpec();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  const mjpEncoder* enc = mjp_findEncoder("model.mjz", nullptr);
  ASSERT_THAT(enc, testing::NotNull());

  mjResource resource = {};
  resource.name = const_cast<char*>("model.mjz");

  int nbytes = enc->encode(spec, model, nullptr, &resource);
  EXPECT_GT(nbytes, 0);
  EXPECT_THAT(resource.data, testing::NotNull());

  std::free(resource.data);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RootFileInArchiveIsNamedModelXml) {
  mjSpec* spec = MakeSimpleSpec();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  const mjpEncoder* enc = mjp_findEncoder("custom_model.mjz", nullptr);
  ASSERT_THAT(enc, testing::NotNull());

  mjResource resource = {};
  resource.name = const_cast<char*>("custom_model.mjz");

  int nbytes = enc->encode(spec, model, nullptr, &resource);
  ASSERT_GT(nbytes, 0);
  ASSERT_THAT(resource.data, testing::NotNull());

  // NOLINTBEGIN(misc-include-cleaner) - Disable clang-tidy for miniz.
  mz_zip_archive zip;
  std::memset(&zip, 0, sizeof(zip));
  ASSERT_TRUE(mz_zip_reader_init_mem(&zip, resource.data, nbytes, 0));

  int file_index = mz_zip_reader_locate_file(&zip, "model.xml", nullptr, 0);
  EXPECT_GE(file_index, 0) << "Expected 'model.xml' to exist in the archive";

  mz_zip_archive_file_stat stat;
  ASSERT_TRUE(mz_zip_reader_file_stat(&zip, file_index, &stat));
  EXPECT_STREQ(stat.m_filename, "model.xml");
  EXPECT_GT(stat.m_uncomp_size, 0);

  size_t xml_size = 0;
  char* xml_data = static_cast<char*>(
      mz_zip_reader_extract_to_heap(&zip, file_index, &xml_size, 0));
  ASSERT_THAT(xml_data, testing::NotNull());
  std::string xml_str(xml_data, xml_size);
  EXPECT_THAT(xml_str, testing::HasSubstr("<mujoco"));

  std::free(xml_data);
  mz_zip_reader_end(&zip);
  // NOLINTEND(misc-include-cleaner) - End clang-tidy disable for miniz.
  std::free(resource.data);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, ArchiveEntryNamesUseForwardSlashes) {
  // one asset path joined with meshdir, one with a subdirectory in the file
  const std::pair<std::string, std::string> cases[] = {
      {"meshdir_test", "assets/box.obj"},
      {"subdir_mesh", "meshes/box.obj"},
  };
  for (const auto& [dir, entry] : cases) {
    SCOPED_TRACE(dir);
    std::string xml_path =
        GetTestDataFilePath("xml/mjz/testdata/" + dir + "/model.xml");

    char error[1024] = {0};
    mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(spec, testing::NotNull()) << error;
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

    const mjpEncoder* enc = mjp_findEncoder("model.mjz", nullptr);
    ASSERT_THAT(enc, testing::NotNull());

    mjResource resource = {};
    resource.name = const_cast<char*>("model.mjz");
    int nbytes = enc->encode(spec, model, nullptr, &resource);
    ASSERT_GT(nbytes, 0);

    mz_zip_archive zip;
    std::memset(&zip, 0, sizeof(zip));
    ASSERT_TRUE(mz_zip_reader_init_mem(&zip, resource.data, nbytes, 0));
    EXPECT_GE(mz_zip_reader_locate_file(&zip, entry.c_str(), nullptr, 0), 0)
        << "Expected '" << entry << "' to exist in the archive";

    mz_zip_reader_end(&zip);
    std::free(resource.data);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

TEST_F(MjzEncoderTest, RoundTrip) {
  mjSpec* spec = MakeSimpleSpec();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  const std::string tmp_path =
      testing::TempDir() + "/mjz_encoder_roundtrip.mjz";

  char error[1024] = {0};
  int nbytes = mj_encode(spec, model, tmp_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mjSpec* decoded =
      mj_parse(tmp_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull());

  EXPECT_EQ(decoded_model->nbody, model->nbody);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);
  EXPECT_EQ(decoded_model->njnt, model->njnt);

  mj_deleteVFS(&vfs);
  std::remove(tmp_path.c_str());
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripFromXmlString) {
  const char* xml = R"(
    <mujoco model="test">
      <worldbody>
        <body name="body1" pos="0 0 1">
          <joint type="hinge"/>
          <geom type="sphere" size="0.1"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  const std::string tmp_path = testing::TempDir() + "/xml_string.mjz";
  int nbytes = mj_encode(spec, model, tmp_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mjSpec* decoded =
      mj_parse(tmp_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull());

  EXPECT_EQ(decoded_model->nbody, model->nbody);
  EXPECT_EQ(decoded_model->njnt, model->njnt);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&vfs);
  std::remove(tmp_path.c_str());
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripPreservesModelCounts) {
  const char* xml = R"(
    <mujoco model="counts_test">
      <worldbody>
        <body name="b1" pos="0 0 1">
          <joint type="hinge"/>
          <geom type="sphere" size="0.1"/>
          <body name="b2" pos="0 0 2">
            <joint type="slide"/>
            <geom type="box" size="0.1 0.1 0.1"/>
          </body>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  const std::string tmp_path = testing::TempDir() + "/counts.mjz";
  int nbytes = mj_encode(spec, model, tmp_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mjSpec* decoded =
      mj_parse(tmp_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull());

  EXPECT_EQ(decoded_model->nbody, model->nbody);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);
  EXPECT_EQ(decoded_model->njnt, model->njnt);
  EXPECT_EQ(decoded_model->nq, model->nq);
  EXPECT_EQ(decoded_model->nv, model->nv);

  mj_deleteVFS(&vfs);
  std::remove(tmp_path.c_str());
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithMeshFromVfs) {
  const char* xml = R"(
    <mujoco model="vfs_mesh">
      <asset>
        <mesh name="box" file="box.obj"/>
      </asset>
      <worldbody>
        <body name="b1">
          <geom type="mesh" mesh="box"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addFileVFS(
      &vfs, nullptr,
      GetTestDataFilePath("xml/mjz/testdata/disk_mesh/box.obj").c_str());

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, &vfs, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;
  auto* mesh = mjs_asMesh(mjs_firstElement(spec, mjOBJ_MESH));
  ASSERT_THAT(mesh, testing::NotNull());
  ASSERT_THAT(mesh->file, testing::NotNull());

  mjModel* model = mj_compile(spec, &vfs);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

  const std::string tmp_path = testing::TempDir() + "/vfs_mesh.mjz";
  int nbytes = mj_encode(spec, model, tmp_path.c_str(), nullptr, &vfs, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Reset VFS
  mj_deleteVFS(&vfs);
  mj_defaultVFS(&vfs);

  mjSpec* decoded =
      mj_parse(tmp_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  EXPECT_EQ(decoded_model->nmesh, model->nmesh);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&vfs);
  std::remove(tmp_path.c_str());
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithMeshFromDisk) {
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/disk_mesh/model.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  fs::path tmpdir = fs::path(testing::TempDir()) / "disk_mesh_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  EXPECT_EQ(decoded_model->nmesh, model->nmesh);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithMeshInSubdirectory) {
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/subdir_mesh/model.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  fs::path tmpdir = fs::path(testing::TempDir()) / "subdir_mesh_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  EXPECT_EQ(decoded_model->nmesh, model->nmesh);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithMeshdir) {
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/meshdir_test/model.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  fs::path tmpdir = fs::path(testing::TempDir()) / "meshdir_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  EXPECT_EQ(decoded_model->nmesh, model->nmesh);

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripRewrittenFilesUnderMeshdir) {
  // under meshdir="assets", the file of "outside" and "outside_again" is
  // rewritten, and the file of "inside" collides with it and is renamed
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/meshdir_rewrite_test/model.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

  fs::path tmpdir = fs::path(testing::TempDir()) / "meshdir_rewrite_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  // mesh.obj and assets/mesh.obj have different vertex counts
  ASSERT_EQ(decoded_model->nmesh, model->nmesh);
  for (int i = 0; i < model->nmesh; ++i) {
    EXPECT_EQ(decoded_model->mesh_vertnum[i], model->mesh_vertnum[i]);
  }

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithInclude) {
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/include_test/parent.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  fs::path tmpdir = fs::path(testing::TempDir()) / "include_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull());

  EXPECT_EQ(decoded_model->nbody, model->nbody);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripTransitiveInclude) {
  std::string xml_path =
      GetTestDataFilePath("xml/mjz/testdata/transitive_test/parent.xml");

  char error[1024] = {0};
  mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  fs::path tmpdir = fs::path(testing::TempDir()) / "transitive_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &vfs);
  ASSERT_THAT(decoded_model, testing::NotNull());

  EXPECT_EQ(decoded_model->nbody, model->nbody);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripMixedDiskAndVfs) {
  static const char* xml = R"(
    <mujoco model="mixed_test">
      <asset>
        <mesh name="disk_box" />
        <mesh name="vfs_box" file="xml/mjz/testdata/disk_mesh/box.obj"/>
      </asset>
      <worldbody>
        <body name="b1">
          <geom type="mesh" mesh="disk_box"/>
        </body>
        <body name="b2">
          <geom type="mesh" mesh="vfs_box"/>
        </body>
      </worldbody>
    </mujoco>

  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addFileVFS(
      &vfs, nullptr,
      GetTestDataFilePath("xml/mjz/testdata/disk_mesh/box.obj").c_str());

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, &vfs, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  // Fix the on disk mesh path
  mjsMesh* mesh = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "disk_box"));
  ASSERT_THAT(mesh, testing::NotNull());
  mjs_setString(
      mesh->file,
      GetTestDataFilePath("xml/mjz/testdata/mixed_test/disk_box.obj").c_str());

  mjModel* model = mj_compile(spec, &vfs);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

  fs::path tmpdir = fs::path(testing::TempDir()) / "mixed_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();

  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, &vfs, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  EXPECT_EQ(decoded_model->nmesh, model->nmesh);
  EXPECT_EQ(decoded_model->ngeom, model->ngeom);

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
  mj_deleteVFS(&vfs);
}

TEST_F(MjzEncoderTest, AttachedSpecsWithCollidingAssetNames) {
  std::string child_a_xml = GetTestDataFilePath(
      "xml/mjz/testdata/asset_collision_test/child_a/child.xml");
  std::string child_b_xml = GetTestDataFilePath(
      "xml/mjz/testdata/asset_collision_test/child_b/child.xml");

  char error[1024] = {0};

  // Parse both children from their respective directories.
  mjSpec* child_a =
      mj_parse(child_a_xml.c_str(), nullptr, nullptr, error, sizeof(error));
  ASSERT_THAT(child_a, testing::NotNull()) << error;
  mjSpec* child_b =
      mj_parse(child_b_xml.c_str(), nullptr, nullptr, error, sizeof(error));
  ASSERT_THAT(child_b, testing::NotNull()) << error;

  // Create parent spec and attach both children.
  mjSpec* parent = mj_makeSpec();
  ASSERT_THAT(parent, testing::NotNull());
  mjsBody* world = mjs_findBody(parent, "world");

  // Attach child_a with prefix "a_".
  mjsFrame* frame_a = mjs_addFrame(world, nullptr);
  mjs_attach(frame_a->element, mjs_findBody(child_a, "body")->element, "a_",
             "");

  // Attach child_b with prefix "b_".
  mjsFrame* frame_b = mjs_addFrame(world, nullptr);
  mjs_attach(frame_b->element, mjs_findBody(child_b, "body")->element, "b_",
             "");

  // Compile the combined model.
  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(parent);
  EXPECT_EQ(model->nmesh, 2) << "Expected two distinct meshes";

  // Record the original vertex counts for both meshes.
  int mesh0_nvert = model->mesh_vertnum[0];
  int mesh1_nvert = model->mesh_vertnum[1];
  EXPECT_NE(mesh0_nvert, mesh1_nvert)
      << "The two meshes should have different vertex counts";

  // Encode to MJZ.
  fs::path tmpdir = fs::path(testing::TempDir()) / "asset_collision_test_out";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();
  int nbytes = mj_encode(parent, model, out_path.c_str(), nullptr, nullptr,
                         error, sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Decode from MJZ.
  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  // Verify both meshes survived with correct, distinct vertex counts.
  EXPECT_EQ(decoded_model->nmesh, 2) << "Decoded model should have two meshes";
  EXPECT_EQ(decoded_model->mesh_vertnum[0], mesh0_nvert)
      << "First mesh vertex count mismatch after roundtrip";
  EXPECT_EQ(decoded_model->mesh_vertnum[1], mesh1_nvert)
      << "Second mesh vertex count mismatch after roundtrip";

  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(parent);
  mj_deleteSpec(child_a);
  mj_deleteSpec(child_b);
}

// --- Fake resource provider for URI stripping tests ---

const char* kBoxObj = R"(
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 2 3
f 1 3 4
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
)";

// Static STL data served by the fake provider.
static std::vector<char> g_fake_data;

int FakeUriOpen(mjResource* resource) {
  if (g_fake_data.empty()) return 0;
  resource->data = g_fake_data.data();
  return 1;
}

int FakeUriRead(mjResource* resource, const void** buffer) {
  if (!resource->data) return -1;
  *buffer = resource->data;
  return static_cast<int>(g_fake_data.size());
}

void FakeUriClose(mjResource* resource) { resource->data = nullptr; }

// Register the fake provider once (idempotent due to AppendIfUnique).
static void EnsureFakeProviderRegistered() {
  static bool registered = false;
  if (registered) return;
  mjpResourceProvider provider = {};
  mjp_defaultProvider(&provider);
  provider.prefix = "testmjzenc";
  provider.open = FakeUriOpen;
  provider.read = FakeUriRead;
  provider.close = FakeUriClose;
  mjp_registerProvider(&provider);
  registered = true;
}

TEST_F(MjzEncoderTest, RoundTripStripsUriPrefix) {
  EnsureFakeProviderRegistered();

  // Set up the fake STL data that the provider will serve.
  g_fake_data.assign(kBoxObj, kBoxObj + std::strlen(kBoxObj));

  const char* xml = R"(
    <mujoco model="uri_test">
      <asset>
        <mesh name="uri_box" file="testmjzenc:meshes/box.obj"/>
      </asset>
      <worldbody>
        <body name="body">
          <geom type="mesh" mesh="uri_box"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  // Compile.
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nmesh, 1);

  // Encode to MJZ.
  fs::path tmpdir = fs::path(testing::TempDir()) / "uri_strip_test";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();
  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Decode from MJZ.
  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  // Verify the decoded mesh file attribute does NOT contain the URI prefix.
  auto* decoded_mesh = mjs_asMesh(mjs_firstElement(decoded, mjOBJ_MESH));
  ASSERT_THAT(decoded_mesh, testing::NotNull());
  const char* decoded_file = mjs_getString(decoded_mesh->file);
  ASSERT_THAT(decoded_file, testing::NotNull());
  EXPECT_FALSE(absl::StrContains(decoded_file, ":"))
      << "Decoded mesh file should not contain ':': " << decoded_file;
  EXPECT_THAT(decoded_file, testing::HasSubstr("box.obj"));

  // Verify the model compiles correctly from the decoded spec.
  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
  EXPECT_EQ(decoded_model->nmesh, model->nmesh);

  // Cleanup.
  g_fake_data.clear();
  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripStripsUriWithSpecialChars) {
  EnsureFakeProviderRegistered();

  // Set up the fake STL data.
  g_fake_data.assign(kBoxObj, kBoxObj + std::strlen(kBoxObj));

  const char* xml = R"(
    <mujoco model="uri_special_test">
      <asset>
        <mesh name="uri_box" file="testmjzenc://store/abc+123%20v2/box.obj"/>
      </asset>
      <worldbody>
        <body name="body">
          <geom type="mesh" mesh="uri_box"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull());

  // Encode to MJZ.
  fs::path tmpdir = fs::path(testing::TempDir()) / "uri_special_test";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();
  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Decode and verify no URI prefix and no special chars in the file attr.
  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  // Check the decoded mesh file attribute.
  auto* decoded_mesh = mjs_asMesh(mjs_firstElement(decoded, mjOBJ_MESH));
  ASSERT_THAT(decoded_mesh, testing::NotNull());
  const char* decoded_file = mjs_getString(decoded_mesh->file);
  ASSERT_THAT(decoded_file, testing::NotNull());
  EXPECT_FALSE(absl::StrContains(decoded_file, "testmjzenc://"))
      << "Decoded mesh file should not contain the URI prefix: "
      << decoded_file;
  EXPECT_FALSE(absl::StrContains(decoded_file, "+"))
      << "Decoded mesh file should not contain '+': " << decoded_file;
  EXPECT_FALSE(absl::StrContains(decoded_file, "%"))
      << "Decoded mesh file should not contain '%': " << decoded_file;
  EXPECT_THAT(decoded_file, testing::HasSubstr(".obj"));

  // Verify roundtrip compiles.
  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
  EXPECT_EQ(decoded_model->nmesh, model->nmesh);

  g_fake_data.clear();
  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripWithMeshdirOutsideArchive) {
  EnsureFakeProviderRegistered();

  // meshdirs that leave the model directory, are absolute, or use a resource
  // provider; the archive is decoded after these sources are removed
  const fs::path srcdir = fs::path(testing::TempDir()) / "meshdir_outside_src";
  const fs::path tmpdir = fs::path(testing::TempDir()) / "meshdir_outside_out";
  const std::string meshdirs[] = {
      "../meshdir_outside_src/assets",
      fs::absolute(srcdir / "assets").string(),
      "testmjzenc:assets",
  };
  for (const std::string& meshdir : meshdirs) {
    SCOPED_TRACE(meshdir);
    fs::remove_all(srcdir);
    fs::create_directories(srcdir / "assets");
    fs::copy_file(
        GetTestDataFilePath("xml/mjz/testdata/meshdir_test/model.xml"),
        srcdir / "model.xml");
    fs::copy_file(
        GetTestDataFilePath("xml/mjz/testdata/meshdir_test/assets/box.obj"),
        srcdir / "assets" / "box.obj");
    g_fake_data.assign(kBoxObj, kBoxObj + std::strlen(kBoxObj));

    char error[1024] = {0};
    const std::string xml_path = (srcdir / "model.xml").string();
    mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(spec, testing::NotNull()) << error;
    mjs_setString(spec->compiler.meshdir, meshdir.c_str());

    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

    fs::create_directories(tmpdir);
    const std::string out_path = (tmpdir / "model.mjz").string();
    int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr,
                           error, sizeof(error));
    ASSERT_GT(nbytes, 0) << error;
    fs::remove_all(srcdir);
    g_fake_data.clear();

    mjVFS decode_vfs;
    mj_defaultVFS(&decode_vfs);
    mjSpec* decoded =
        mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
    ASSERT_THAT(decoded, testing::NotNull()) << error;

    mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
    ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
    EXPECT_EQ(decoded_model->nmesh, model->nmesh);

    mj_deleteVFS(&decode_vfs);
    fs::remove_all(tmpdir);
    mj_deleteModel(decoded_model);
    mj_deleteSpec(decoded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

TEST_F(MjzEncoderTest, RoundTripWithRepeatedSeparators) {
  // compiler attributes and assets that refer to sub/box.obj and sub/tex.png
  // with a repeated separator, which archive entry names drop
  const std::pair<std::string, std::string> cases[] = {
      {"", R"(<mesh name="box" file="sub//box.obj"/>)"},
      // a later reference to an archived file
      {"", R"(<mesh name="box" file="sub/box.obj"/>)"
           R"(<mesh name="box_again" file="sub//box.obj"/>)"},
      {R"(meshdir="sub//")", R"(<mesh name="box" file="box.obj"/>)"},
      {R"(texturedir="sub//")",
       R"(<texture name="tex" type="2d" file="tex.png"/>)"},
  };
  const fs::path srcdir =
      fs::path(testing::TempDir()) / "repeated_separators_src";
  const fs::path tmpdir =
      fs::path(testing::TempDir()) / "repeated_separators_out";
  fs::remove_all(srcdir);
  fs::create_directories(srcdir / "sub");
  fs::copy_file(GetTestDataFilePath("xml/mjz/testdata/disk_mesh/box.obj"),
                srcdir / "sub" / "box.obj");
  fs::copy_file(GetTestDataFilePath("xml/testdata/hfield.png"),
                srcdir / "sub" / "tex.png");
  const std::string xml_path = (srcdir / "model.xml").string();
  for (const auto& [compiler, assets] : cases) {
    const std::string xml = "<mujoco><compiler " + compiler + "/><asset>" +
                            assets + "</asset></mujoco>";
    SCOPED_TRACE(xml);
    std::ofstream(xml_path) << xml;

    char error[1024] = {0};
    mjSpec* spec = mj_parseXML(xml_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(spec, testing::NotNull()) << error;

    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

    fs::create_directories(tmpdir);
    const std::string out_path = (tmpdir / "model.mjz").string();
    int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr,
                           error, sizeof(error));
    ASSERT_GT(nbytes, 0) << error;

    mjVFS decode_vfs;
    mj_defaultVFS(&decode_vfs);
    mjSpec* decoded =
        mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
    ASSERT_THAT(decoded, testing::NotNull()) << error;

    mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
    ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
    EXPECT_EQ(decoded_model->nmesh, model->nmesh);
    EXPECT_EQ(decoded_model->ntex, model->ntex);

    mj_deleteVFS(&decode_vfs);
    fs::remove_all(tmpdir);
    mj_deleteModel(decoded_model);
    mj_deleteSpec(decoded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
  fs::remove_all(srcdir);
}

class MjzEncoderParameterizedTest
    : public MujocoTest,
      public ::testing::WithParamInterface<std::string> {};

TEST_P(MjzEncoderParameterizedTest, WriteReadCompare) {
  mujoco::FullFloatPrecision increase_precision;

  std::string xml = GetParam();
  std::array<char, 1000> error;

  mjSpec* s = mj_parseXML(xml.c_str(), nullptr, error.data(), error.size());
  if (!s) {
    GTEST_SKIP() << "Failed to load " << xml.c_str() << ": " << error.data();
  }

  mjModel* m = mj_compile(s, nullptr);
  if (!m) {
    std::string error_message = mjs_getError(s);
    mj_deleteSpec(s);
    GTEST_SKIP() << "Failed to compile " << xml.c_str() << ": "
                 << error_message;
  }

  // one archive per case: on Linux, ctest runs the cases as parallel processes
  const std::string tmp_path = testing::TempDir() + "/mjz_roundtrip_" +
                               SanitizePathForTestName(xml) + ".mjz";
  int nbytes = mj_encode(s, m, tmp_path.c_str(), nullptr, nullptr, error.data(),
                         error.size());
  ASSERT_GT(nbytes, 0) << error.data();
  mj_deleteSpec(s);

  mjVFS vfs;
  mj_defaultVFS(&vfs);

  mjSpec* stemp =
      mj_parse(tmp_path.c_str(), nullptr, &vfs, error.data(), error.size());
  ASSERT_THAT(stemp, NotNull()) << error.data();

  mjModel* mtemp = mj_compile(stemp, &vfs);
  ASSERT_THAT(mtemp, NotNull()) << mjs_getError(stemp);
  mj_deleteSpec(stemp);

  mjtNum tol = 0;
  if (absl::StrContains(xml, "belt.xml") ||
      absl::StrContains(xml, "cable.xml") ||
      absl::StrContains(xml, "helix.xml") ||
      absl::StrContains(xml, "fromto_body_body.xml")) {
    tol = 1e-13;
  }

  // Make paths identical to avoid failure in CompareModel due to localization.
  // For example, we might localize "../../y" to "y", which changes the paths
  // and the addresses of the paths that follow it.
  char* old_m_paths = m->paths;
  char* old_mtemp_paths = mtemp->paths;
  int old_m_npaths = m->npaths;
  int old_mtemp_npaths = mtemp->npaths;

  std::string dummy = "ignored_path";
  int dummy_len = dummy.length() + 1;

  m->paths = (char*)mju_malloc(dummy_len);
  mtemp->paths = (char*)mju_malloc(dummy_len);

  std::memcpy(m->paths, dummy.c_str(), dummy_len);
  std::memcpy(mtemp->paths, dummy.c_str(), dummy_len);

  m->npaths = dummy_len;
  mtemp->npaths = dummy_len;

  // point asset paths at the dummy, keeping -1 (no path)
  auto reset_pathadr = [](int* pathadr, mjtSize n) {
    for (int i = 0; i < n; ++i) {
      if (pathadr[i] > 0) pathadr[i] = 0;
    }
  };
  for (mjModel* model : {m, mtemp}) {
    reset_pathadr(model->mesh_pathadr, model->nmesh);
    reset_pathadr(model->skin_pathadr, model->nskin);
    reset_pathadr(model->hfield_pathadr, model->nhfield);
    reset_pathadr(model->tex_pathadr, model->ntex);
    reset_pathadr(model->tex_contenttypeadr, model->ntex);
  }

  std::string field = "";
  mjtNum result = CompareModel(m, mtemp, field);

  // Restore old pointers and sizes so mj_deleteModel can free them correctly!
  mju_free(m->paths);
  mju_free(mtemp->paths);
  m->paths = old_m_paths;
  mtemp->paths = old_mtemp_paths;
  m->npaths = old_m_npaths;
  mtemp->npaths = old_mtemp_npaths;

  EXPECT_LE(result, tol) << "Loaded and saved models are different!\n"
                         << "Affected file: " << xml << '\n'
                         << "Different field: " << field << '\n';

  mj_deleteVFS(&vfs);
  mj_deleteModel(mtemp);
  mj_deleteModel(m);
  std::remove(tmp_path.c_str());
}

INSTANTIATE_TEST_SUITE_P(MjzEncoderParameterizedTests,
                         MjzEncoderParameterizedTest,
                         ::testing::ValuesIn(GetWriteReadTestModels()),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           return SanitizePathForTestName(info.param);
                         });

TEST_F(MjzEncoderTest, DuplicateFileReferencesRewrite) {
  EnsureFakeProviderRegistered();

  // Set up the fake data that the provider will serve.
  g_fake_data.assign(kBoxObj, kBoxObj + std::strlen(kBoxObj));

  // Two meshes referencing the same URI-prefixed file.
  const char* xml = R"(
    <mujoco model="dup_file_test">
      <asset>
        <mesh name="box_a" file="testmjzenc:meshes/box.obj"/>
        <mesh name="box_b" file="testmjzenc:meshes/box.obj"/>
      </asset>
      <worldbody>
        <body name="a">
          <geom type="mesh" mesh="box_a"/>
        </body>
        <body name="b">
          <geom type="mesh" mesh="box_b"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

  // Encode to MJZ.
  fs::path tmpdir = fs::path(testing::TempDir()) / "dup_file_test";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();
  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Decode from MJZ.
  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  // Both decoded meshes should have sanitized file paths (no URI prefix).
  auto* mesh_a = mjs_asMesh(mjs_firstElement(decoded, mjOBJ_MESH));
  ASSERT_THAT(mesh_a, testing::NotNull());
  const char* file_a = mjs_getString(mesh_a->file);
  ASSERT_THAT(file_a, testing::NotNull());
  EXPECT_FALSE(absl::StrContains(file_a, ":"))
      << "First mesh file should not contain ':': " << file_a;

  auto* mesh_b = mjs_asMesh(mjs_nextElement(decoded, mesh_a->element));
  ASSERT_THAT(mesh_b, testing::NotNull());
  const char* file_b = mjs_getString(mesh_b->file);
  ASSERT_THAT(file_b, testing::NotNull());
  EXPECT_FALSE(absl::StrContains(file_b, ":"))
      << "Second mesh file should not contain ':': " << file_b;

  // Both should compile correctly.
  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
  EXPECT_EQ(decoded_model->nmesh, model->nmesh);

  // Cleanup.
  g_fake_data.clear();
  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, CubefileUriRewrite) {
  EnsureFakeProviderRegistered();

  // The fake provider serves arbitrary data; cubefiles need valid PNG data to
  // compile textures, but the encoder just packs bytes. We verify archive
  // contents and XML rewriting without requiring valid image data.
  // Use a minimal valid 1x1 white PNG for each face.
  static const unsigned char kTinyPng[] = {
      0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d,
      0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
      0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, 0xde, 0x00, 0x00, 0x00,
      0x0c, 0x49, 0x44, 0x41, 0x54, 0x78, 0x9c, 0x63, 0xf8, 0xff, 0xff, 0x3f,
      0x00, 0x05, 0xfe, 0x02, 0xfe, 0x0d, 0xef, 0x46, 0xb8, 0x00, 0x00, 0x00,
      0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
  };
  g_fake_data.assign(
      reinterpret_cast<const char*>(kTinyPng),
      reinterpret_cast<const char*>(kTinyPng) + sizeof(kTinyPng));

  const char* xml = R"(
    <mujoco model="cubefile_test">
      <asset>
        <texture name="sky" type="cube"
                 fileright="testmjzenc:faces/right.png"
                 fileleft="testmjzenc:faces/left.png"
                 fileup="testmjzenc:faces/up.png"
                 filedown="testmjzenc:faces/down.png"
                 filefront="testmjzenc:faces/front.png"
                 fileback="testmjzenc:faces/back.png"/>
        <material name="sky_mat" texture="sky"/>
      </asset>
      <worldbody>
        <body name="body">
          <geom type="sphere" size="0.1" material="sky_mat"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  char error[1024] = {0};
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error, sizeof(error));
  ASSERT_THAT(spec, testing::NotNull()) << error;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);

  // Encode to MJZ.
  fs::path tmpdir = fs::path(testing::TempDir()) / "cubefile_test";
  fs::create_directories(tmpdir);
  const std::string out_path = (tmpdir / "model.mjz").string();
  int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, nullptr, error,
                         sizeof(error));
  ASSERT_GT(nbytes, 0) << error;

  // Decode from MJZ.
  mjVFS decode_vfs;
  mj_defaultVFS(&decode_vfs);
  mjSpec* decoded =
      mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
  ASSERT_THAT(decoded, testing::NotNull()) << error;

  // Verify the decoded texture's cubefile paths don't contain URI prefix.
  auto* decoded_tex = mjs_asTexture(mjs_firstElement(decoded, mjOBJ_TEXTURE));
  ASSERT_THAT(decoded_tex, testing::NotNull());
  ASSERT_THAT(decoded_tex->cubefiles, testing::NotNull());
  for (const mjString& cubefile : *decoded_tex->cubefiles) {
    EXPECT_FALSE(absl::StrContains(cubefile, ":"))
        << "Cubefile should not contain ':': " << cubefile;
    EXPECT_THAT(std::string(cubefile), testing::HasSubstr(".png"));
  }

  // Verify the model compiles correctly from the decoded spec.
  mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
  ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);

  // Cleanup.
  g_fake_data.clear();
  mj_deleteVFS(&decode_vfs);
  fs::remove_all(tmpdir);
  mj_deleteModel(decoded_model);
  mj_deleteSpec(decoded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MjzEncoderTest, RoundTripStrippathTrue) {
  for (bool savecompiled : {false, true}) {
    SCOPED_TRACE(savecompiled);
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    mj_addBufferVFS(&vfs, "box.obj", kBoxObj, std::strlen(kBoxObj));
    mj_addBufferVFS(&vfs, "box$1.obj", kBoxObj, std::strlen(kBoxObj));

    const char* xml = R"(
      <mujoco model="strippath_test">
        <compiler strippath="true"/>
        <asset>
          <mesh name="box" file="sub/box.obj"/>
          <mesh name="sanitized_box" file="other/sub/box$1.obj"/>
        </asset>
        <worldbody>
          <body name="body">
            <geom type="mesh" mesh="box"/>
            <geom type="mesh" mesh="sanitized_box"/>
          </body>
        </worldbody>
      </mujoco>
    )";

    char error[1024] = {0};
    mjSpec* spec = mj_parseXMLString(xml, &vfs, error, sizeof(error));
    ASSERT_THAT(spec, testing::NotNull()) << error;
    spec->compiler.savecompiled = savecompiled;

    mjModel* model = mj_compile(spec, &vfs);
    ASSERT_THAT(model, testing::NotNull()) << mjs_getError(spec);
    EXPECT_EQ(model->nmesh, 2);

    fs::path tmpdir = fs::path(testing::TempDir()) / "strippath_test";
    fs::create_directories(tmpdir);
    const std::string out_path = (tmpdir / "model.mjz").string();
    int nbytes = mj_encode(spec, model, out_path.c_str(), nullptr, &vfs, error,
                           sizeof(error));
    ASSERT_GT(nbytes, 0) << error;

    mjVFS decode_vfs;
    mj_defaultVFS(&decode_vfs);
    mjSpec* decoded =
        mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
    ASSERT_THAT(decoded, testing::NotNull()) << error;

    auto* san_mesh =
        mjs_asMesh(mjs_findElement(decoded, mjOBJ_MESH, "sanitized_box"));
    ASSERT_THAT(san_mesh, testing::NotNull());
    EXPECT_STREQ(mjs_getString(san_mesh->file), "box_1.obj");

    mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
    ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
    EXPECT_EQ(decoded_model->nmesh, 2);

    mj_deleteVFS(&decode_vfs);
    fs::remove_all(tmpdir);
    mj_deleteModel(decoded_model);
    mj_deleteSpec(decoded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
    mj_deleteVFS(&vfs);
  }
}

TEST_F(MjzEncoderTest, RoundTripStrippathCollidingSubdirAssets) {
  static constexpr char kTetObj[] = R"(
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 3 2
f 1 2 4
f 2 3 4
f 3 1 4
)";
  static constexpr char kChildAXml[] = R"(
    <mujoco model="child_a">
      <compiler strippath="true"/>
      <asset>
        <mesh name="part" file="dir_a/part.obj"/>
      </asset>
      <worldbody>
        <body name="body">
          <geom type="mesh" mesh="part"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  static constexpr char kChildBXml[] = R"(
    <mujoco model="child_b">
      <compiler strippath="true"/>
      <asset>
        <mesh name="part" file="dir_b/part.obj"/>
      </asset>
      <worldbody>
        <body name="body">
          <geom type="mesh" mesh="part"/>
        </body>
      </worldbody>
    </mujoco>
  )";

  for (bool savecompiled : {false, true}) {
    SCOPED_TRACE(savecompiled);
    const fs::path srcdir =
        fs::path(testing::TempDir()) / "strippath_collision_src";
    const fs::path tmpdir =
        fs::path(testing::TempDir()) / "strippath_collision_out";
    fs::remove_all(srcdir);
    fs::create_directories(srcdir / "a");
    fs::create_directories(srcdir / "b");
    std::ofstream(srcdir / "a" / "child.xml") << kChildAXml;
    std::ofstream(srcdir / "a" / "part.obj") << kBoxObj;
    std::ofstream(srcdir / "b" / "child.xml") << kChildBXml;
    std::ofstream(srcdir / "b" / "part.obj") << kTetObj;

    char error[1024] = {0};
    const std::string child_a_path = (srcdir / "a" / "child.xml").string();
    const std::string child_b_path = (srcdir / "b" / "child.xml").string();
    mjSpec* child_a =
        mj_parseXML(child_a_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(child_a, testing::NotNull()) << error;
    mjSpec* child_b =
        mj_parseXML(child_b_path.c_str(), nullptr, error, sizeof(error));
    ASSERT_THAT(child_b, testing::NotNull()) << error;

    mjSpec* parent = mj_makeSpec();
    parent->strippath = true;
    parent->compiler.savecompiled = savecompiled;
    mjsBody* world = mjs_findBody(parent, "world");

    mjsFrame* frame_a = mjs_addFrame(world, nullptr);
    mjs_attach(frame_a->element, mjs_findBody(child_a, "body")->element, "a_",
               "");
    mjsFrame* frame_b = mjs_addFrame(world, nullptr);
    mjs_attach(frame_b->element, mjs_findBody(child_b, "body")->element, "b_",
               "");

    mjModel* model = mj_compile(parent, nullptr);
    ASSERT_THAT(model, testing::NotNull()) << mjs_getError(parent);
    ASSERT_EQ(model->nmesh, 2);
    EXPECT_EQ(model->mesh_vertnum[0], 8);
    EXPECT_EQ(model->mesh_vertnum[1], 4);

    fs::create_directories(tmpdir);
    const std::string out_path = (tmpdir / "model.mjz").string();
    int nbytes = mj_encode(parent, model, out_path.c_str(), nullptr, nullptr,
                           error, sizeof(error));
    ASSERT_GT(nbytes, 0) << error;
    fs::remove_all(srcdir);

    mjVFS decode_vfs;
    mj_defaultVFS(&decode_vfs);
    mjSpec* decoded =
        mj_parse(out_path.c_str(), nullptr, &decode_vfs, error, sizeof(error));
    ASSERT_THAT(decoded, testing::NotNull()) << error;

    mjModel* decoded_model = mj_compile(decoded, &decode_vfs);
    ASSERT_THAT(decoded_model, testing::NotNull()) << mjs_getError(decoded);
    ASSERT_EQ(decoded_model->nmesh, 2);
    EXPECT_EQ(decoded_model->mesh_vertnum[0], 8);
    EXPECT_EQ(decoded_model->mesh_vertnum[1], 4);

    mj_deleteVFS(&decode_vfs);
    fs::remove_all(tmpdir);
    mj_deleteModel(decoded_model);
    mj_deleteSpec(decoded);
    mj_deleteModel(model);
    mj_deleteSpec(parent);
    mj_deleteSpec(child_a);
    mj_deleteSpec(child_b);
  }
}

}  // namespace mujoco
