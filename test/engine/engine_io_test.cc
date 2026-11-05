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

// Tests for engine/{engine_io.c and engine_memory.c}.

#include "src/engine/engine_io.h"

#include <array>
#include <atomic>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>  // NOLINT
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_memory.h"
#include "src/engine/engine_thread.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ContainsRegex;  // NOLINT(misc-unused-using-decls) asan only
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

using EngineIoTest = MujocoTest;

TEST_F(EngineIoTest, VerifySizeModel) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  std::filesystem::path temp_file =
      (std::filesystem::temp_directory_path() / "model.mjb");

  mj_saveModel(model.get(), temp_file.string().c_str(), NULL, 0);

  std::uintmax_t file_size = std::filesystem::file_size(temp_file);
  mjtSize model_size = mj_sizeModel(model.get());

  std::filesystem::remove(temp_file);

  EXPECT_EQ(file_size, model_size);
}

TEST_F(EngineIoTest, MakeDataLoadsQpos0) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  model->qpos0[0] = 1;
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());
  EXPECT_EQ(data->qpos[0], 1);
}

TEST_F(EngineIoTest, MakeDataLoadsMocapBodies) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body mocap="true" pos="42 0 42">
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());
  EXPECT_EQ(data->mocap_pos[0], 42);
}

TEST_F(EngineIoTest, MakeDataReturnsNullOnFailure) {
  constexpr char xml[] = "<mujoco/>";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  // fail mj_makeData intentionally with a bad size
  model->nbody = -1;
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings();
  MjDataPtr data = MakeData(model);
  EXPECT_THAT(data, IsNull());
}

TEST_F(EngineIoTest, SubTimestepSensorIntervalStrictlyMonotonic) {
  for (int nsample : {5, 7, 10, 16}) {
    std::string xml = absl::StrFormat(
        R"(
        <mujoco>
          <option timestep="0.01" gravity="0 0 0"/>
          <worldbody>
            <body>
              <joint name="slide" type="slide"/>
              <geom size="0.1"/>
            </body>
          </worldbody>
          <sensor>
            <jointpos joint="slide" delay="0.02" interval="0.002 0" nsample="%d"/>
            <jointpos joint="slide" delay="0.02" interval="0.002 -0.001" nsample="%d"/>
            <jointpos joint="slide" delay="0.02" interval="0.01 0" nsample="%d"/>
            <jointpos joint="slide" delay="0.02" interval="0.01 -0.005" nsample="%d"/>
            <jointpos joint="slide" delay="0.02" interval="0.015 0" nsample="%d"/>
          </sensor>
        </mujoco>
        )",
        nsample, nsample, nsample, nsample, nsample);
    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml.c_str(), error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    MjDataPtr data = MakeData(model);

    for (int s = 0; s < model->nsensor; s++) {
      int n = model->sensor_history[2 * s];
      ASSERT_EQ(n, nsample);
      const mjtNum* buf = data->history + model->sensor_historyadr[s];
      const mjtNum* times = buf + 2;
      for (int j = 0; j < n - 1; j++) {
        EXPECT_LT(times[j], times[j + 1])
            << "nsample=" << nsample << " sensor=" << s << " index=" << j;
      }
      if (s < 4) {
        int offset = (s % 2 == 0) ? n : n - 1;
        for (int j = 0; j < n; j++) {
          EXPECT_NEAR(times[j], -(offset - j) * model->opt.timestep,
                      MjTol(1e-12, 1e-6))
              << "nsample=" << nsample << " sensor=" << s << " index=" << j;
        }
      }
    }

    // Verify t=0 sample on sensor 0 (phase=0) is recorded and read after delay.
    data->qpos[0] = 5.0;
    mj_step(model.get(), data.get());
    EXPECT_NEAR(data->sensordata[0], 0.0, MjTol(1e-10, 1e-6));
    mj_step(model.get(), data.get());
    EXPECT_NEAR(data->sensordata[0], 0.0, MjTol(1e-10, 1e-6));
    mj_step(model.get(), data.get());
    EXPECT_NEAR(data->sensordata[0], 5.0, MjTol(1e-10, 1e-6));
  }
}

TEST_F(EngineIoTest, ResetVariableSizes) {
  constexpr char xml[] = "<mujoco/>";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  MjDataPtr data = MakeData(model);
  ASSERT_THAT(model.get(), NotNull()) << "Failed to create mjData";

  // don't call mj_forward, vars should be reset
  EXPECT_EQ(data->ne, 0);
  EXPECT_EQ(data->nf, 0);
  EXPECT_EQ(data->nefc, 0);
  EXPECT_EQ(data->nJ, 0);
  EXPECT_EQ(data->nA, 0);
  EXPECT_EQ(data->ncon, 0);
}

TEST_F(EngineIoTest, MakeDataResetsAllArenaPointerSizes) {
  constexpr char xml[] = "<mujoco />";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  // without calling mj_forward, all array sizes should be zero.
  EXPECT_EQ(data->parena, 0) << "expecting empty arena";

#define X(type, name, nr, nc) \
  EXPECT_EQ(nr* nc, 0) << "expecting (" #nr " x " #nc ") to be zero";

#undef MJ_D
#undef MJ_M
#define MJ_D(name) data->name
#define MJ_M(name) model->name
  MJDATA_ARENA_POINTERS;
#undef MJ_D
#undef MJ_M
#define MJ_D(n) n
#define MJ_M(n) n
#undef X
}

TEST_F(EngineIoTest, MjvCopyModel) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tet" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="tet"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model1 = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model1.get(), NotNull()) << error;

  mjModel* model2 = mj_copyModel(nullptr, model1.get());
  ASSERT_THAT(model2, NotNull()) << error;

  model1->mesh_vert[0] = 0.1;
  model1->geom_rgba[0] = 0.2;
  mj_copyModel(model2, model1.get());

  EXPECT_FLOAT_EQ(model2->mesh_vert[0], 0.1);
  EXPECT_FLOAT_EQ(model2->geom_rgba[0], 0.2);

  model1->mesh_vert[0] = 0.3;
  model1->geom_rgba[0] = 0.4;
  mjv_copyModel(model2, model1.get());

  EXPECT_FLOAT_EQ(model2->mesh_vert[0], 0.1);  // unchanged
  EXPECT_FLOAT_EQ(model2->geom_rgba[0], 0.4);

  mj_deleteModel(model2);
}

TEST_F(EngineIoTest, MjvCopyData) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0.8">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
      <geom type="plane" size="1 1 1"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  MjDataPtr data1 = MakeData(model);
  mj_forward(model.get(), data1.get());
  EXPECT_THAT(data1->efc_J, NotNull());

  mjData* data2 = mj_copyData(nullptr, model.get(), data1.get());
  EXPECT_THAT(data2->efc_J, NotNull());

  mj_deleteData(data2);
  data2 = mjv_copyData(nullptr, model.get(), data1.get());
  EXPECT_THAT(data2->efc_J, IsNull());

  mj_deleteData(data2);
}

// number of blocks allocated by MisalignedMalloc and not yet freed
std::atomic<int> misaligned_nblock{0};

// user allocator with blocks that are 16-byte aligned, like those of malloc,
// but never 64-byte aligned
void* MisalignedMalloc(std::size_t size) {
  char* raw = static_cast<char*>(std::malloc(size + 128));
  if (!raw) {
    return nullptr;
  }

  // start the block 16 bytes past a 64-byte boundary, keep raw just before it
  std::uintptr_t boundary =
      (reinterpret_cast<std::uintptr_t>(raw) + 63) & ~std::uintptr_t{63};
  void** block = reinterpret_cast<void**>(boundary + 16);
  block[-1] = raw;
  misaligned_nblock++;
  return block;
}

void MisalignedFree(void* block) {
  misaligned_nblock--;
  std::free(static_cast<void**>(block)[-1]);
}

// installs the misaligned allocator, restores the previous one on destruction
class MisalignedAllocator {
 public:
  MisalignedAllocator() {
    mju_user_malloc = MisalignedMalloc;
    mju_user_free = MisalignedFree;
  }
  ~MisalignedAllocator() {
    mju_user_malloc = prev_malloc_;
    mju_user_free = prev_free_;
  }

 private:
  void* (*prev_malloc_)(std::size_t) = mju_user_malloc;
  void (*prev_free_)(void*) = mju_user_free;
};

// expect the arrays in the buffers of two mjData to be bitwise equal
void ExpectBuffersEqual(const mjModel* m, const mjData* d1, const mjData* d2) {
#define X(type, name, nr, nc)                                                  \
  EXPECT_EQ(std::memcmp(d1->name, d2->name, sizeof(type) * (m->nr) * (nc)), 0) \
      << "mjData::" #name " differs";
  MJDATA_POINTERS
#undef X
}

TEST_F(EngineIoTest, MisalignedUserAllocator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 1"/>
      <body pos="0 0 0.1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  constexpr int kSteps = 10;
  char error[1024];

  // reference model and data, made with the default allocator
  MjModelPtr model_ref = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model_ref.get(), NotNull()) << error;
  mj_freeLastXML();
  MjDataPtr data_ref = MakeData(model_ref);
  for (int i = 0; i < kSteps; i++) {
    mj_step(model_ref.get(), data_ref.get());
  }
  ASSERT_GT(data_ref->nefc, 0);

  {
    // objects declared after the allocator are freed before it is uninstalled
    MisalignedAllocator allocator;

    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    mj_freeLastXML();
    MjDataPtr data = MakeData(model);
    ASSERT_THAT(data.get(), NotNull());
    for (int i = 0; i < kSteps; i++) {
      mj_step(model.get(), data.get());
    }

    // the buffers are misaligned, their arrays are 64-byte aligned
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(model->buffer) % 64, 16);
    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(data->buffer) % 64, 16);
#define X(type, name, nr, nc) \
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(model->name) % 64, 0) << #name;
    MJMODEL_POINTERS
#undef X
#define X(type, name, nr, nc) \
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(data->name) % 64, 0) << #name;
    MJDATA_POINTERS
#undef X

    // same trajectory as with the default allocator
    EXPECT_EQ(data->nefc, data_ref->nefc);
    ExpectBuffersEqual(model.get(), data.get(), data_ref.get());

    // copy and load across allocators
    mj_copyModel(model_ref.get(), model.get());
    MjDataPtr copy(mj_copyData(nullptr, model.get(), data_ref.get()));
    ExpectBuffersEqual(model.get(), copy.get(), data_ref.get());
    EXPECT_EQ(std::memcmp(copy->efc_force, data_ref->efc_force,
                          sizeof(mjtNum) * data_ref->nefc),
              0);
    std::vector<char> mjb(mj_sizeModel(model_ref.get()));
    mj_saveModel(model_ref.get(), nullptr, mjb.data(), mjb.size());
    MjModelPtr loaded(mj_loadModelBuffer(mjb.data(), mjb.size()));
    EXPECT_THAT(loaded.get(), NotNull());
  }

  // all blocks were freed before the allocator was uninstalled
  EXPECT_EQ(misaligned_nblock.load(), 0);
}

using ValidateReferencesTest = MujocoTest;

TEST_F(ValidateReferencesTest, BodyReferences) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  model->jnt_bodyid[0] = 2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("jnt_bodyid"));
}

TEST_F(ValidateReferencesTest, AddressRange) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <joint axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  model->body_jntnum[1] = 3;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("body_jntadr"));
  model->body_jntnum[1] = 2;
  // Could be more strict and test for -1, but at the moment the code is a bit
  // lenient.
  model->body_jntadr[1] = -2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("body_jntadr"));
}

TEST_F(ValidateReferencesTest, AddressRangeNegativeNum) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <joint axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // jntadr + jntnum is within safe range, but jntnum is negative.
  model->body_jntadr[1] += 5;
  model->body_jntnum[1] -= 5;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("body_jntnum"));
}

TEST_F(ValidateReferencesTest, GeomCondim) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size="1" condim="6"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
  model->geom_condim[0] = 7;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("geom_condim"));
  model->geom_condim[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("geom_condim"));
}

TEST_F(ValidateReferencesTest, HField) {
  static const char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="h" nrow="2" ncol="3" size="1 1 1 1" />
    </asset>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  model->hfield_adr[0] = -2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("hfield_adr"));
  model->hfield_adr[0] = 0;
  model->hfield_ncol[0] = 4;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("hfield_adr"));
}

TEST_F(ValidateReferencesTest, Texture) {
  static const char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="t" type="2d" width="2" height="3" builtin="flat" />
    </asset>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  model->tex_adr[0] = -2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("tex_adr"));
  model->tex_adr[0] = 0;
  model->tex_height[0] = 4;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("tex_adr"));
}

TEST_F(ValidateReferencesTest, Mesh) {
  static const char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="m" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
                      face="0 1 2  0 1 3  0 2 3  1 2 3"
                      texcoord="0 0  1 0  0 1  1 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="m"/>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // A mesh face vertex index is a mesh-local index into [0, mesh_vertnum).
  // These are read verbatim from a binary (MJB) model, so an out-of-range value
  // must be rejected here (it would otherwise cause an out-of-bounds read when
  // the mesh is ray-cast, rendered or collided).
  int saved = model->mesh_face[0];
  model->mesh_face[0] = model->mesh_vertnum[0];  // one past the end
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("mesh_face"));
  model->mesh_face[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("mesh_face"));
  model->mesh_face[0] = saved;
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // Face normal indices are mesh-local indices into [0, mesh_normalnum) and
  // are used the same way (mesh_normal + 3*(mesh_facenormal[...] + normaladr)).
  saved = model->mesh_facenormal[0];
  model->mesh_facenormal[0] = model->mesh_normalnum[0];  // one past the end
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("mesh_facenormal"));
  model->mesh_facenormal[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("mesh_facenormal"));
  model->mesh_facenormal[0] = saved;
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // Face texcoord indices are mesh-local indices into [0, mesh_texcoordnum),
  // used only when the mesh has texcoords (mesh_texcoordadr != -1).
  ASSERT_GE(model->mesh_texcoordadr[0], 0);
  saved = model->mesh_facetexcoord[0];
  model->mesh_facetexcoord[0] = model->mesh_texcoordnum[0];  // one past the end
  EXPECT_THAT(mj_validateReferences(model.get()),
              HasSubstr("mesh_facetexcoord"));
  model->mesh_facetexcoord[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()),
              HasSubstr("mesh_facetexcoord"));

  // A mesh without texcoords (mesh_texcoordadr == -1) never reads
  // mesh_facetexcoord, so its contents are not validated in that case.
  int saved_adr = model->mesh_texcoordadr[0];
  int saved_num = model->mesh_texcoordnum[0];
  model->mesh_texcoordadr[0] = -1;
  model->mesh_texcoordnum[0] = 0;
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
  model->mesh_texcoordadr[0] = saved_adr;
  model->mesh_texcoordnum[0] = saved_num;
  model->mesh_facetexcoord[0] = saved;
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // adr == -1 is only valid for empty elements
  saved = model->mesh_faceadr[0];
  model->mesh_faceadr[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("mesh_faceadr"));
  model->mesh_faceadr[0] = saved;
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
}

TEST_F(ValidateReferencesTest, GeomPairs) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom name="geom0" size="1"/>
      </body>
      <body>
        <geom name="geom1" size="1"/>
        <geom name="geom2" size="1"/>
        <geom name="geom3" size="1"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="geom0" geom2="geom3"/>
    </contact>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // Invalid geomid=4
  model->pair_signature[0] = (1 << 16) | 5;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("pair_body1"));

  model->pair_signature[0] = (5 << 16) | 1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("pair_body2"));
}

TEST_F(ValidateReferencesTest, SensorsAddress) {
  // The test will likely only catch sensor size errors for the last sensor
  // in the model, so iterate over possible last sensors, instead of adding
  // them all into the same model.
  static const char xml_template[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <joint name="slider" type="slide" axis="0 0 1"
            limited="true" range="-.2 .5"/>
        <geom name="geom1" size="1"/>
        <site name="site1"/>
      </body>
    </worldbody>
    <sensor>
      %s
    </sensor>
  </mujoco>
  )";
  std::vector<std::string> sensor_strings{
      "<framepos objtype='site' objname='site1'/>",
      "<rangefinder site='site1'/>",
      "<gyro site='site1'/>",
      "<touch site='site1'/>",
      "<force site='site1'/>",
      "<torque site='site1'/>",
      "<jointlimitfrc joint='slider'/>",
      "<accelerometer site='site1'/>",
      "<subtreeangmom body='body1'/>",
  };

  for (const std::string& sensor_string : sensor_strings) {
    std::string xml = absl::StrFormat(xml_template, sensor_string);
    std::array<char, 1024> error;
    MjModelPtr model =
        LoadModelFromString(xml.c_str(), error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull())
        << "Failed to load model: " << error.data();

    EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
  }
}

TEST_F(ValidateReferencesTest, SensorsAddressUser) {
  static const char xml[] = R"(
  <mujoco>
    <sensor>
      <user dim="3" user="1 2 3 4 5" />
      <user dim="2" />
      <user dim="1" />
    </sensor>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
}

TEST_F(ValidateReferencesTest, SensorsObj) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <joint/>
        <geom size="1"/>
        <site name="site1"/>
      </body>
    </worldbody>
    <sensor>
      <framepos objtype="site" objname="site1" reftype="body" refname="body1"/>
    </sensor>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  model->sensor_objtype[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("sensor_objtype"));
  model->sensor_objtype[0] = mjOBJ_SITE;

  model->sensor_objid[0] = model->nsite;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("sensor_objid"));
  model->sensor_objid[0] = 0;

  model->sensor_reftype[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("sensor_reftype"));
  model->sensor_reftype[0] = mjOBJ_BODY;

  model->sensor_refid[0] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("sensor_refid"));
}

TEST_F(ValidateReferencesTest, MoreBodiesThanGeoms) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom name="geom0" size="1"/>
      </body>
      <body>
        <geom name="geom1" size="1"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="geom1" geom2="geom0"/>
    </contact>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
}

TEST_F(ValidateReferencesTest, BodyExcludes) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1" />
      <body name="body2" />
    </worldbody>
    <contact>
      <exclude body1="body1" body2="body2"/>
    </contact>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // Invalid bodyid=3
  model->exclude_signature[0] = (1 << 16) | 4;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("exclude_body1"));
  model->exclude_signature[0] = (4 << 16) | 2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("exclude_body2"));
}

TEST_F(ValidateReferencesTest, FlexPlasticMode) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="shell" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" elastic2d="both" yield="0 0.01"/>
      </flexcomp>
      <flexcomp name="rope" type="grid" count="4 1 1" spacing=".1 .1 .1" dim="1"/>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  int shell = mj_name2id(model.get(), mjOBJ_FLEX, "shell");
  int rope = mj_name2id(model.get(), mjOBJ_FLEX, "rope");
  ASSERT_EQ(model->flex_plasticmode[shell], 1);
  ASSERT_EQ(model->flex_plasticadr[rope], -1);
  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // unknown mode bits
  model->flex_plasticmode[shell] = 4;
  EXPECT_THAT(mj_validateReferences(model.get()),
              HasSubstr("flex_plasticmode"));
  model->flex_plasticmode[shell] = 1;

  // plastic modes without plastic state
  model->flex_plasticadr[shell] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()),
              HasSubstr("flex_plasticmode"));
  model->flex_plasticadr[shell] = 0;

  // stretching plasticity of a 1D flex
  model->flex_plasticmode[rope] = 2;
  model->flex_plasticadr[rope] = 0;
  EXPECT_THAT(mj_validateReferences(model.get()),
              HasSubstr("flex_plasticmode"));
}

TEST_F(ValidateReferencesTest, EqualityConstraints) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <joint name="joint1"/>
        <geom name="geom1" size="1"/>
      </body>
      <body name="body2">
        <joint/>
        <joint name="joint2" axis="1 0 0"/>
        <geom size="1"/>
        <geom size="1"/>
        <geom name="geom2" size="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="tendon1"><joint joint="joint1" coef="1"/></fixed>
      <fixed><joint joint="joint1" coef="1"/></fixed>
      <fixed><joint joint="joint1" coef="1"/></fixed>
      <fixed><joint joint="joint1" coef="1"/></fixed>
      <fixed name="tendon2"><joint joint="joint1" coef="1"/></fixed>
    </tendon>
    <equality>
      <connect anchor="0 0 0" body1="body1" />
      <weld body1="body1" body2="body2" />
      <joint joint1="joint1"/>
      <joint joint1="joint1" joint2="joint2"/>
      <tendon tendon1="tendon1"/>
      <tendon tendon1="tendon1" tendon2="tendon2"/>
    </equality>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  // connect constraint
  model->eq_obj1id[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj1id"));
  model->eq_obj1id[0] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj1id"));
  model->eq_obj1id[0] = 1;

  model->eq_obj2id[0] = -2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj2id"));
  model->eq_obj2id[0] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj2id"));
  model->eq_obj2id[0] = 0;

  // weld constraint
  model->eq_obj1id[1] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj1id"));
  model->eq_obj1id[1] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj1id"));
  model->eq_obj1id[1] = 1;

  model->eq_obj2id[1] = -2;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj2id"));
  model->eq_obj2id[1] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("eq_obj2id"));
  model->eq_obj2id[1] = model->nbody - 1;
}

TEST_F(ValidateReferencesTest, Tuples) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <joint name="joint1"/>
        <geom size="1"/>
      </body>
      <body name="body2">
        <joint/>
        <joint name="joint2" axis="1 0 0"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <custom>
      <tuple name="tuple">
        <element objtype="body" objname="body1"/>
        <element objtype="joint" objname="joint2"/>
      </tuple>
    </custom>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  model->tuple_objtype[0] = -1;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("tuple_objtype"));
  model->tuple_objtype[0] = mjOBJ_BODY;

  model->tuple_objid[0] = model->nbody;
  EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr("tuple_objid"));
  model->tuple_objid[0] = 1;
}

TEST_F(ValidateReferencesTest, SparseMatrices) {
  static const char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b1">
        <joint name="j1" type="hinge"/>
        <geom size="1"/>
        <site name="s1"/>
        <body name="b2" pos="0 0 1">
          <joint name="j2" type="hinge"/>
          <geom size="1"/>
          <site name="s2"/>
        </body>
      </body>
    </worldbody>
    <tendon>
      <spatial>
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  EXPECT_THAT(mj_validateReferences(model.get()), IsNull());

  auto check_field = [&](int* field, int bad_val, const char* substr) {
    int saved = *field;
    *field = bad_val;
    EXPECT_THAT(mj_validateReferences(model.get()), HasSubstr(substr));
    *field = saved;
    EXPECT_THAT(mj_validateReferences(model.get()), IsNull());
  };

  check_field(&model->M_rowadr[0], 0x1000000, "M_rowadr");
  check_field(&model->M_rowadr[0], -1, "M_rowadr");
  check_field(&model->M_rownnz[1], 1, "M_rownnz");
  check_field(&model->M_rownnz[1], 3, "M_rownnz");
  check_field(&model->M_colind[0], 1, "M_colind");
  check_field(&model->dof_Madr[1], 0, "dof_Madr");

  check_field(&model->D_rowadr[0], model->nD, "D_rowadr");
  check_field(&model->D_rownnz[0], 1, "D_rownnz");
  check_field(&model->D_colind[0], 1, "D_colind");
  check_field(&model->D_diag[1], 0, "D_diag");

  check_field(&model->B_rowadr[1], model->nB, "B_rowadr");
  check_field(&model->B_rownnz[0], 1, "B_rownnz");
  check_field(&model->B_colind[model->B_rowadr[2]], 1, "B_colind");

  check_field(&model->mapM2M[0], -1, "mapM2M");
  check_field(&model->mapM2M[0], model->nM, "mapM2M");
  check_field(&model->mapM2D[0], -2, "mapM2D");
  check_field(&model->mapM2D[0], model->nC, "mapM2D");
  check_field(&model->mapD2M[0], -1, "mapD2M");
  check_field(&model->mapD2M[0], model->nD, "mapD2M");

  check_field(&model->ten_J_rowadr[0], model->nJten, "ten_J_rowadr");
  check_field(&model->ten_J_rownnz[0], model->nv + 1, "ten_J_rownnz");
  check_field(&model->ten_J_colind[0], -1, "ten_J_colind");
  check_field(&model->ten_J_colind[0], model->nv, "ten_J_colind");
}

TEST_F(EngineIoTest, CanMarkAndFreeStack) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  auto pstack_before = data->pstack;
  mj_markStack(data.get());
  EXPECT_GT(data->pstack, pstack_before);
  mj_freeStack(data.get());
  EXPECT_EQ(data->pstack, pstack_before);
}

TEST_F(EngineIoTest, LargeMemory) {
  constexpr char xml[] = R"(
  <mujoco>
    <size memory="2400M"/>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  // allocate 2.3G of mjtNums
  mj_markStack(data.get());
  size_t num = 2300000000 / sizeof(mjtNum);
  mjtNum* testNum = mj_stackAllocNum(data.get(), num);
  testNum[num - 1] = 1;
  mj_freeStack(data.get());

  // allocate 2.3G of bytes
  mj_markStack(data.get());
  num = 2300000000;
  char* testByte = (char*)mj_stackAllocByte(data.get(), num, alignof(char));
  testByte[num - 1] = 1;
  mj_freeStack(data.get());
}

TEST_F(EngineIoTest, VeryLargeMemory) {
  constexpr char xml[] = R"(
  <mujoco>
    <size memory="8G"/>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  if (!model.get()) {
    // in some test environments, 8GB is too large
    EXPECT_THAT(error.data(), HasSubstr("Could not allocate memory"));
  } else {
    ASSERT_THAT(model.get(), NotNull())
        << "Failed to load model: " << error.data();
    MjDataPtr data = MakeData(model);
    ASSERT_THAT(data, NotNull());

    // allocate 7G of mjtNums
    mj_markStack(data.get());
    size_t num = 7516192768ull / sizeof(mjtNum);
    mjtNum* testNum = mj_stackAllocNum(data.get(), num);
    testNum[num - 1] = 1;
    mj_freeStack(data.get());

    // allocate 7G of bytes
    mj_markStack(data.get());
    num = 7516192768ull;
    char* testByte = (char*)mj_stackAllocByte(data.get(), num, alignof(char));
    testByte[num - 1] = 1;
    mj_freeStack(data.get());
  }
}

struct TestFunctionArgs {
  int stack_output[1000];
  int arena_output[1000];
};

void TestFunction(const mjModel* m, mjData* d, void* args, int i, int j) {
  TestFunctionArgs* test_args = static_cast<TestFunctionArgs*>(args);
  mj_markStack(d);
  int* test_ints = mj_stackAllocInt(d, 10);
  test_ints[0] = j;
  test_args->stack_output[j] = test_ints[0];
  mj_freeStack(d);
}

TEST_F(EngineIoTest, TestStackShardingForThreads) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());
  mju_threadpool(data.get(), 10);

  constexpr int kTasks = 1000;
  TestFunctionArgs test_function_args;
  mju_dispatch(model.get(), data.get(), TestFunction, &test_function_args,
               kTasks);

  for (int i = 0; i < kTasks; ++i) {
    EXPECT_EQ(i, test_function_args.stack_output[i]);
  }
}

#ifdef mjUSEASAN
void MarkFreeStack(mjData* d, bool free) {
  mj_markStack(d);
  if (free) {
    mj_freeStack(d);
  }
}

TEST_F(EngineIoTest, CanDetectStackFrameLeakage) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  // MarkFreeStack correctly calls mj_freeStack, should not error.
  MarkFreeStack(data.get(), /* free= */ true);

  // MarkFreeStack calls mj_markStack without mj_freeStack, the next call to
  // mj_freeStack should detect the stack frame leakage.
  mj_markStack(data.get());
  MarkFreeStack(data.get(), /* free= */ false);
  EXPECT_THAT(
      MjuErrorMessageFrom(mj_freeStack)(data.get()),
      ContainsRegex(
          "mj_markStack in .*MarkFreeStack.* at .*engine_io_test\\.cc.* has no "
          "corresponding mj_freeStack"));

  // Dangling stack frames should be detected in mj_deleteData.
  mj_resetData(model.get(), data.get());
  mj_markStack(data.get());
  EXPECT_THAT(
      MjuErrorMessageFrom(mj_deleteData)(data.get()),
      ContainsRegex(
          "mj_markStack in .+EngineIoTest_CanDetectStackFrameLeakage.+ has no "
          "corresponding mj_freeStack"));

  mj_resetData(model.get(), data.get());
}

TEST_F(EngineIoTest, RedZoneAlignmentTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  mj_markStack(data.get());
  mj_stackAllocByte(data.get(), 1, 1);
  mj_stackAllocByte(data.get(), 1, 1);
  mj_freeStack(data.get());
}
#endif

// Regression test: crafted binary model with overflow-inducing sizes must be
// safely rejected (not cause heap overflow). This exercises the overflow checks
// in safeAddToBufferSize on all compilers including MSVC.
TEST_F(EngineIoTest, LoadModelBufferRejectsOverflowingSizes) {
  // construct a minimal valid-looking .mjb header
  int header[5];
  header[0] = 20;              // ID
  header[1] = sizeof(mjtNum);  // floating point size
  // We need the correct nsize and nptr from the current build.
  // Rather than hardcoding, we create and save a trivial model, then mutate
  // the sizes to trigger overflow.

  constexpr char xml[] = "<mujoco />";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  // save model to a buffer
  mjtSize bufsize = mj_sizeModel(model.get());
  ASSERT_GT(bufsize, 0);
  std::vector<char> buffer(bufsize);
  mj_saveModel(model.get(), nullptr, buffer.data(), bufsize);

  // locate the size fields in the buffer (after 5-int header)
  const int header_bytes = 5 * sizeof(int);
  ASSERT_GT(bufsize, header_bytes + 77 * (int)sizeof(mjtSize));

  // mutate a size field to an extremely large value that would overflow
  // when multiplied by sizeof(type). ntexdata is a good candidate since it
  // is a byte count field and gets multiplied by sizeof(mjtByte)==1, but other
  // fields multiply by sizeof(int) or sizeof(mjtNum), making overflow easier.
  // Use memcpy to avoid undefined behavior from unaligned access.
  const int size_offset = header_bytes + 49 * sizeof(mjtSize);

  // set to overflow-inducing value
  mjtSize overflow_val = static_cast<mjtSize>(SIZE_MAX / 2);
  std::memcpy(buffer.data() + size_offset, &overflow_val, sizeof(mjtSize));

  // also need to update the nbuffer field (last size) to avoid the early
  // nbuffer mismatch check — but the overflow should be caught earlier
  // in safeAddToBufferSize/mj_makeModel before we reach that check.

  // Intercept warnings to prevent them from failing the test.
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings();

  // attempt to load — should return NULL, not crash
  mjModel* bad_model = mj_loadModelBuffer(buffer.data(), bufsize);
  EXPECT_THAT(bad_model, IsNull())
      << "Expected mj_loadModelBuffer to reject overflow-inducing sizes";

  // clean up if somehow it succeeded
  if (bad_model) {
    mj_deleteModel(bad_model);
  }
}

TEST_F(EngineIoTest, SaveModelAcceptsLargeBufferSize) {
  constexpr char xml[] = "<mujoco />";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  mjtSize bufsize = mj_sizeModel(model.get());
  std::vector<char> buffer(bufsize);

  mjtSize large_bufsize = static_cast<mjtSize>(INT_MAX) + 1024LL;
  mj_saveModel(model.get(), nullptr, buffer.data(), large_bufsize);
}

TEST_F(EngineIoTest, ArenaAllocMisalignsPointerWhenBaseNotMultiple) {
  mj_freeLastXML();
  MisalignedAllocator allocator;

  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b0"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  mj_freeLastXML();
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data.get(), NotNull());

  // Arena base is 16-byte aligned, not 64-byte aligned.
  ASSERT_NE(reinterpret_cast<std::uintptr_t>(data->arena) % 64, 0);

  void* ptr = mj_arenaAllocByte(data.get(), 64, 64);
  ASSERT_THAT(ptr, NotNull());
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % 64, 0);
}

TEST_F(EngineIoTest, LoadModelBufferTruncatedStructsDoesNotLeak) {
  constexpr char xml[] = R"(
  <mujoco/>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();

  mjtSize bufsize = mj_sizeModel(model.get());
  std::vector<char> buffer(bufsize);
  mj_saveModel(model.get(), nullptr, buffer.data(), bufsize);

  int nsize = 0;
  std::memcpy(&nsize, buffer.data() + 2 * sizeof(int), sizeof(int));
  ASSERT_GT(nsize, 0);

  // Truncate buffer after header and sizes, in the middle of structs.
  const mjtSize header_and_sizes = 5 * sizeof(int) + nsize * sizeof(mjtSize);
  const mjtSize truncated_size = header_and_sizes + 16;
  ASSERT_LT(truncated_size, bufsize);

  static std::atomic<int> live_allocs{0};
  live_allocs = 0;
  void* (*prev_malloc)(std::size_t) = mju_user_malloc;
  void (*prev_free)(void*) = mju_user_free;

  mju_user_malloc = +[](std::size_t size) -> void* {
    live_allocs++;
    return std::malloc(size);
  };
  mju_user_free = +[](void* block) {
    if (block) {
      live_allocs--;
      std::free(block);
    }
  };

  {
    MockWarningHandler warning_handler;
    warning_handler.ExpectWarnings();

    mjModel* bad_model = mj_loadModelBuffer(buffer.data(), truncated_size);
    EXPECT_THAT(bad_model, IsNull());
    if (bad_model) {
      mj_deleteModel(bad_model);
    }
  }

  mju_user_malloc = prev_malloc;
  mju_user_free = prev_free;

  EXPECT_EQ(live_allocs.load(), 0)
      << "mj_loadModelBuffer leaked model memory on truncated struct read";
}

TEST_F(EngineIoTest, RecompilePreservesThreadPool) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b" pos="0 0 1">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());

  mju_threadpool(d, 4);
  const std::uintptr_t pool = d->threadpool;
  ASSERT_NE(pool, 0u);
  EXPECT_EQ(mju_numThread(d), 5);

  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* b2 = mjs_addBody(world, nullptr);
  mjs_addGeom(b2, nullptr)->size[0] = 0.2;

  ASSERT_EQ(mj_recompile(spec, nullptr, m, d), 0) << mjs_getError(spec);
  EXPECT_EQ(d->threadpool, pool);
  EXPECT_EQ(mju_numThread(d), 5);

  mj_step(m, d);

  mj_deleteData(d);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

std::uintptr_t g_observed_plugin_data = 1;
int g_init_zero_calls = 0;

// Register at static initialization time (before main) so GlobalTable's
// intentional process-lifetime allocation does not trip HeapLeakChecker.
const int kInitZeroPluginSlot = []() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.io_init_zero";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    g_observed_plugin_data = d->plugin_data[instance];
    g_init_zero_calls++;
    return 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) {};
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int type) {};
  return mjp_registerPlugin(&plugin);
}();

TEST_F(EngineIoTest, InitPluginZeroesPluginDataBeforeInit) {
  ASSERT_GE(kInitZeroPluginSlot, 0);

  constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.test.io_init_zero">
        <instance name="inst"/>
      </plugin>
    </extension>
    <worldbody>
      <body>
        <freejoint/>
        <geom size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  g_observed_plugin_data = 0xDEADBEEFu;
  g_init_zero_calls = 0;

  void* (*prev_malloc)(std::size_t) = mju_user_malloc;
  void (*prev_free)(void*) = mju_user_free;
  mju_user_malloc = +[](std::size_t size) -> void* {
    void* ptr = std::malloc(size);
    if (ptr) {
      std::memset(ptr, 0xAB, size);
    }
    return ptr;
  };
  mju_user_free = +[](void* ptr) { std::free(ptr); };

  // allocate and free data under the same hooks
  mjData* data = mj_makeData(model.get());
  const bool made_data = data != nullptr;
  const std::uintptr_t plugin_data = made_data ? data->plugin_data[0] : 1;
  mj_deleteData(data);

  mju_user_malloc = prev_malloc;
  mju_user_free = prev_free;

  ASSERT_TRUE(made_data);
  EXPECT_EQ(g_init_zero_calls, 1);
  EXPECT_EQ(g_observed_plugin_data, 0u);
  EXPECT_EQ(plugin_data, 0u);
}

// Note: in float32 (mjUSESINGLE) builds, 14 lines of ~105 characters stay
// under the 2048-byte buffer and cannot overflow it; only float64 (~310
// characters per line at 1e302%) exercises truncation at sizeof(buf) - 1.
TEST_F(EngineIoTest, LogTimingDiagnosticsBufferOverflow) {
  constexpr char xml[] = R"(
  <mujoco/>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  MjDataPtr data = MakeData(model);

  data->timer[mjTIMER_STEP].number = 1;
#ifdef mjUSESINGLE
  data->timer[mjTIMER_STEP].duration = 1.0f;
  const mjtNum large_dur = 1e35f;
#else
  data->timer[mjTIMER_STEP].duration = 1e-300;
  const mjtNum large_dur = 1.0;
#endif

  for (int i = mjTIMER_POSITION; i <= mjTIMER_ADVANCE; i++) {
    data->timer[i].number = 1;
    data->timer[i].duration = large_dur;
  }
  for (int p = mjTIMER_POS_KINEMATICS; p <= mjTIMER_POS_PROJECT; p++) {
    data->timer[p].number = 1;
    data->timer[p].duration = large_dur;
  }
  for (int c = mjTIMER_COL_BROAD; c <= mjTIMER_COL_NARROW; c++) {
    data->timer[c].number = 1;
    data->timer[c].duration = large_dur;
  }

  static int captured_count;
  static std::string* captured_subject;
  static std::string* captured_body;
  std::string subject;
  std::string body;
  captured_count = 0;
  captured_subject = &subject;
  captured_body = &body;

  mjfLogHandler prev_handler =
      _mjPRIVATE_setTlsLogHandler(+[](const mjLogMessage* msg) {
        if (msg->topic == mjTOPIC_TIME_STP) {
          captured_count++;
          *captured_subject = msg->subject;
          *captured_body = msg->body ? msg->body : "";
        }
      });
  mj_resetData(model.get(), data.get());
  _mjPRIVATE_setTlsLogHandler(prev_handler);

  EXPECT_EQ(captured_count, 1);
  EXPECT_THAT(subject, HasSubstr("average time per step"));
#ifdef mjUSESINGLE
  EXPECT_GT(body.size(), 0u);
  EXPECT_LT(body.size(), 2048u);
  EXPECT_THAT(body, HasSubstr("total"));
#else
  EXPECT_EQ(body.size(), 2047u);
#endif
  EXPECT_EQ(data->timer[mjTIMER_STEP].number, 0);
}

}  // namespace
}  // namespace mujoco
