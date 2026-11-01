// Copyright 2022 DeepMind Technologies Limited
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

#include "src/engine/engine_print.h"

#include <array>
#include <fstream>
#include <sstream>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

#ifdef MEMORY_SANITIZER
  #include <sanitizer/msan_interface.h>
#endif

namespace mujoco {
namespace {

#ifdef MEMORY_SANITIZER
using ::testing::Eq;
using ::testing::Not;
#endif
using ::testing::HasSubstr;
using ::testing::NotNull;

using EnginePrintTest = MujocoTest;

constexpr const char* NullFile() {
#ifdef _WIN32
  return "NUL";
#else
  return "/dev/null";
#endif
}

TEST_F(EnginePrintTest, PrintDataWorksWithMsan) {
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
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data, NotNull());

  // Mark qacc_smooth[0] as initialized.
  data->qacc_smooth[0] = 0;

  // This will read "uninitialized" values from mjData, but shouldn't fail.
  mj_printData(model.get(), data.get(), NullFile());

  // After mj_printData, poisoned status should be restored correctly, so
  // qacc_smooth[0] should be marked initialzed.
  EXPECT_EQ(data->qacc_smooth[0], 0);

#ifdef MEMORY_SANITIZER
  EXPECT_THAT(__msan_test_shadow(data->buffer, data->nbuffer), Not(Eq(-1)))
      << "Expecting some of data->buffer to be marked uninitialized";
#endif
}

TEST_F(EnginePrintTest, RejectsTruncatedFloatFormatStrings) {
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

  mock_warning_handler.ExpectWarnings("Format string specifier must be one of");
  mock_warning_handler.ExpectWarnings("Received invalid float_format");

  const char truncated_len_fmt[4] = {'%', 'L', '\0', '\0'};
  mj_printFormattedModel(model.get(), NullFile(), truncated_len_fmt);

  const char truncated_pct_fmt[4] = {'%', '\0', 'g', '\0'};
  mj_printFormattedModel(model.get(), NullFile(), truncated_pct_fmt);
}

TEST_F(EnginePrintTest,
       PrintModelExcludesWorldMassAndPrintsFreeJointPlasticKeyframe) {
  constexpr char xml[] = R"(
  <mujoco>
    <size nkey="1"/>
    <worldbody>
      <body>
        <freejoint/>
        <geom size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="j1" axis="1 0 0" springyield="0.1" stiffness="1"/>
        <joint name="j2" axis="0 1 0" springyield="0.1" stiffness="1"/>
        <joint name="j3" axis="0 0 1" springyield="0.1" stiffness="1"/>
        <joint name="j4" type="slide" axis="1 0 0"
          springyield="0.1" stiffness="1"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body>
        <joint name="j5" axis="1 0 0" springyield="0.1" stiffness="1"/>
        <joint name="j6" axis="0 1 0" springyield="0.1" stiffness="1"/>
        <joint name="j7" axis="0 0 1" springyield="0.1" stiffness="1"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull())
      << "Failed to load model: " << error.data();
  ASSERT_EQ(model->nplastic, 7);

  model->body_mass[0] = 5.0;
  EXPECT_MJTNUM_EQ(mj_getTotalmass(model.get()), 3.0);

  // mjCJoint::Compile rejects springyield > 0 on type="free", so assign the 7
  // compiled plastic slots to the free joint post-compile to cover mjJNT_FREE.
  model->jnt_plasticadr[0] = 0;
  for (int j = 1; j < model->njnt; ++j) {
    model->jnt_plasticadr[j] = -1;
  }
  for (int d = 0; d < 7; ++d) {
    model->key_plastic[d] = model->qpos_spring[d];
  }
  model->key_plastic[1] = model->qpos_spring[1] + static_cast<mjtNum>(2.5);

  std::string filepath = testing::TempDir() + "/printed_model.txt";
  mj_printFormattedModel(model.get(), filepath.c_str(), "%.1f");

  std::ifstream ifs(filepath);
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  std::string content = buffer.str();

  EXPECT_THAT(content, HasSubstr("totalmass            3.0"));
  EXPECT_THAT(content, HasSubstr("key_plastic0"));
}

}  // namespace
}  // namespace mujoco
