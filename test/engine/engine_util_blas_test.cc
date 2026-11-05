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

// Tests for engine/engine_util_blas.c

#include "src/engine/engine_util_blas.h"

#include <random>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjtype.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;

using EngineUtilBlasTest = MujocoTest;

TEST_F(EngineUtilBlasTest, MjuDot) {
  mjtNum a[] = {1, 2, 3, 4, 5, 6, 7};
  mjtNum b[] = {7, 6, 5, 4, 3, 2, 1};

  // test various vector lengths because mju_dot adds numbers in groups of four
  EXPECT_EQ(mju_dot(a, b, 0), 0);
  EXPECT_EQ(mju_dot(a, b, 1), 7);
  EXPECT_EQ(mju_dot(a, b, 2), 7 + 2 * 6);
  EXPECT_EQ(mju_dot(a, b, 3), 7 + 2 * 6 + 3 * 5);
  EXPECT_EQ(mju_dot(a, b, 4), 7 + 2 * 6 + 3 * 5 + 4 * 4);
  EXPECT_EQ(mju_dot(a, b, 5), 7 + 2 * 6 + 3 * 5 + 4 * 4 + 5 * 3);
  EXPECT_EQ(mju_dot(a, b, 6), 7 + 2 * 6 + 3 * 5 + 4 * 4 + 5 * 3 + 6 * 2);
  EXPECT_EQ(mju_dot(a, b, 7), 7 + 2 * 6 + 3 * 5 + 4 * 4 + 5 * 3 + 6 * 2 + 7);
}

TEST_F(EngineUtilBlasTest, MjuMulVecMatVec) {
  mjtNum vec1[] = {1, 2, 3};
  mjtNum vec2[] = {3, 2, 1};
  mjtNum mat[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};

  EXPECT_EQ(mju_mulVecMatVec(vec1, mat, vec2, 3), 204);
}

TEST_F(EngineUtilBlasTest, MjuFill) {
  mjtNum vec[] = {0, 1, 4};
  mju_fill(vec, 4.5, 3);

  EXPECT_EQ(vec[0], 4.5);
  EXPECT_EQ(vec[1], 4.5);
  EXPECT_EQ(vec[2], 4.5);
}

TEST_F(EngineUtilBlasTest, MjuEye) {
  mjtNum mat1[1];
  mju_eye(mat1, 1);
  EXPECT_EQ(mat1[0], 1);

  mjtNum mat3[9];
  mju_eye(mat3, 3);
  EXPECT_THAT(mat3, ElementsAre(1, 0, 0, 0, 1, 0, 0, 0, 1));
}

TEST_F(EngineUtilBlasTest, MjuSymmetrize) {
  mjtNum mat[] = {1, 2.5, 3.5, 1.5, 2, 4, 2.5, 3, 3};
  mjtNum res[9] = {0};
  mju_symmetrize(res, mat, 3);
  EXPECT_THAT(res, ElementsAre(1, 2, 3, 2, 2, 3.5, 3, 3.5, 3));

  // test for the case res==mat
  mju_symmetrize(mat, mat, 3);
  EXPECT_THAT(mat, ElementsAre(1, 2, 3, 2, 2, 3.5, 3, 3.5, 3));
}

TEST_F(EngineUtilBlasTest, MjuMulMat3) {
  const mjtNum mat1[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  const mjtNum mat2[9] = {2, 3, 4, 5, 6, 7, 8, 9, 10};
  mjtNum res1[9] = {0};
  mjtNum res2[9] = {0};

  mju_mulMatMat3(res1, mat1, mat2);
  mju_mulMatMat(res2, mat1, mat2, 3, 3, 3);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(res1[i], res2[i]);
  }

  mju_mulMatTMat3(res1, mat1, mat2);
  mju_mulMatTMat(res2, mat1, mat2, 3, 3, 3);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(res1[i], res2[i]);
  }

  mju_mulMatMatT3(res1, mat1, mat2);
  mju_mulMatMatT(res2, mat1, mat2, 3, 3, 3);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(res1[i], res2[i]);
  }
}

// utility: random quaternion, normally distributed
void RandomQuat(mjtNum quat[4], int seed) {
  // make distribution using seed
  std::mt19937_64 rng;
  rng.seed(seed);
  std::normal_distribution<mjtNum> dist(0, 1);

  // sample
  for (int i = 0; i < 4; i++) {
    quat[i] = dist(rng);
  }
}

TEST_F(EngineUtilBlasTest, Normalize4IsIdempotent) {
  for (int i = 0; i < 10000; ++i) {
    // make random quaternion and normalize it
    mjtNum quat[4];
    RandomQuat(quat, i);
    mju_normalize4(quat);

    // normalize again
    mjtNum quat_renormalized[4] = {quat[0], quat[1], quat[2], quat[3]};
    mju_normalize4(quat_renormalized);

    // expect equality
    EXPECT_MJTNUM_EQ(quat[0], quat_renormalized[0]);
    EXPECT_MJTNUM_EQ(quat[1], quat_renormalized[1]);
    EXPECT_MJTNUM_EQ(quat[2], quat_renormalized[2]);
    EXPECT_MJTNUM_EQ(quat[3], quat_renormalized[3]);
  }
}

TEST_F(EngineUtilBlasTest, Normalize4EdgeCases) {
  // zero quat normalizes to unit quat
  mjtNum quat[4] = {0};
  mjtNum norm = mju_normalize4(quat);
  EXPECT_EQ(norm, 0);
  EXPECT_EQ(quat[0], 1);
  EXPECT_EQ(quat[1], 0);
  EXPECT_EQ(quat[2], 0);
  EXPECT_EQ(quat[3], 0);

  // small quat normalizes regularly
  quat[0] = 0;
  quat[1] = mjMINVAL;
  norm = mju_normalize4(quat);
  EXPECT_EQ(norm, mjMINVAL);
  EXPECT_EQ(quat[0], 0);
  EXPECT_EQ(quat[1], 1);
  EXPECT_EQ(quat[2], 0);
  EXPECT_EQ(quat[3], 0);

  // tiny quat normalizes to unit quat
  quat[0] = 0;
  quat[1] = mjMINVAL / 2;
  norm = mju_normalize4(quat);
  EXPECT_EQ(norm, mjMINVAL / 2);
  EXPECT_EQ(quat[0], 1);
  EXPECT_EQ(quat[1], 0);
  EXPECT_EQ(quat[2], 0);
  EXPECT_EQ(quat[3], 0);

  // near-unit quat is normalized
  quat[0] = 1 + mjMINVAL;
  quat[1] = 0;
  norm = mju_normalize4(quat);
  EXPECT_EQ(norm, 1 + mjMINVAL);
  EXPECT_EQ(quat[0], 1);
  EXPECT_EQ(quat[1], 0);
  EXPECT_EQ(quat[2], 0);
  EXPECT_EQ(quat[3], 0);

  // very-near-unit quat is untouched
  quat[0] = 1 + mjMINVAL / 2;
  quat[1] = 0;
  norm = mju_normalize4(quat);
  EXPECT_EQ(norm, 1 + mjMINVAL / 2);
  EXPECT_EQ(quat[0], 1 + mjMINVAL / 2);
  EXPECT_EQ(quat[1], 0);
  EXPECT_EQ(quat[2], 0);
  EXPECT_EQ(quat[3], 0);
}

TEST_F(EngineUtilBlasTest, MjuNormalizeZeroLength) {
  mjtNum vec[1] = {42.0};
  EXPECT_MJTNUM_EQ(mju_normalize(vec, 0), 0.0);
  EXPECT_MJTNUM_EQ(vec[0], 42.0);
}

TEST_F(EngineUtilBlasTest, MjuNormalize) {
  // standard normalization
  mjtNum vec[] = {3, 4, 0};
  EXPECT_MJTNUM_EQ(mju_normalize(vec, 3), 5);
  EXPECT_MJTNUM_EQ(vec[0], 0.6);
  EXPECT_MJTNUM_EQ(vec[1], 0.8);
  EXPECT_MJTNUM_EQ(vec[2], 0);

  // zero vector normalizes to x-axis
  mjtNum zero[] = {0, 0, 0};
  EXPECT_MJTNUM_EQ(mju_normalize(zero, 3), 0);
  EXPECT_THAT(zero, ElementsAre(1, 0, 0));

  // tiny vector normalizes to x-axis
  mjtNum tiny[] = {0, mjMINVAL / 2, 0};
  EXPECT_MJTNUM_EQ(mju_normalize(tiny, 3), mjMINVAL / 2);
  EXPECT_THAT(tiny, ElementsAre(1, 0, 0));
}

}  // namespace
}  // namespace mujoco
