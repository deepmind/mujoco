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

// Tests for engine/engine_util_sparse.c

#include "src/engine/engine_util_sparse.h"

#include <array>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

// permute the rows and columns of a dense matrix
inline void PermuteMat(mjtNum* res, const mjtNum* mat, int nr, int nc,
                       const int* perm_r, const int* perm_c, bool scatter_r,
                       bool scatter_c) {
  for (int r = 0; r < nr; r++) {
    for (int c = 0; c < nc; c++) {
      if (scatter_r && scatter_c) {
        // scatter both
        res[perm_r[r] * nc + perm_c[c]] = mat[r * nc + c];
      } else if (scatter_r && !scatter_c) {
        // scatter rows, gather columns
        res[perm_r[r] * nc + c] = mat[r * nc + perm_c[c]];
      } else if (!scatter_r && scatter_c) {
        // gather rows, scatter columns
        res[r * nc + perm_c[c]] = mat[perm_r[r] * nc + c];
      } else {
        // gather both
        res[r * nc + c] = mat[perm_r[r] * nc + perm_c[c]];
      }
    }
  }
}

using ::testing::ElementsAre;
using EngineUtilSparseTest = MujocoTest;

TEST_F(EngineUtilSparseTest, MjuDot) {
  mjtNum a[] = {2, 3, 4, 5, 6, 7, 8};
  mjtNum b[] = {8, 1, 7, 1, 1, 6, 1, 1, 1, 5, 1, 1, 1, 4, 1, 1, 3, 1, 2};
  int i[] = {0, 2, 5, 9, 13, 16, 18};

  // test various vector lengths as mju_dotSparse adds numbers in groups of four

  EXPECT_EQ(mju_dotSparse(a, b, 0, i), 0);
  EXPECT_EQ(mju_dotSparse(a, b, 1, i), 2 * 8);
  EXPECT_EQ(mju_dotSparse(a, b, 2, i), 2 * 8 + 3 * 7);
  EXPECT_EQ(mju_dotSparse(a, b, 3, i), 2 * 8 + 3 * 7 + 4 * 6);
  EXPECT_EQ(mju_dotSparse(a, b, 4, i), 2 * 8 + 3 * 7 + 4 * 6 + 5 * 5);
  EXPECT_EQ(mju_dotSparse(a, b, 5, i), 2 * 8 + 3 * 7 + 4 * 6 + 5 * 5 + 6 * 4);
  EXPECT_EQ(mju_dotSparse(a, b, 6, i),
            2 * 8 + 3 * 7 + 4 * 6 + 5 * 5 + 6 * 4 + 7 * 3);
  EXPECT_EQ(mju_dotSparse(a, b, 7, i),
            2 * 8 + 3 * 7 + 4 * 6 + 5 * 5 + 6 * 4 + 7 * 3 + 8 * 2);
}

TEST_F(EngineUtilSparseTest, MjuDot2) {
  constexpr int annz = 6;
  constexpr int bnnz = 5;

  // values
  mjtNum a[annz] = {2, 3, 4, 5, 6};
  mjtNum b[bnnz] = {8, 7, 6, 5, 4};

  // indices
  int ia[annz] = {0, 2, 5, 6, 7};
  int ib[bnnz] = {1, 2, 3, 5, 7};

  EXPECT_EQ(mju_dotSparse2(a, ia, annz, b, ib, bnnz), 3 * 7 + 4 * 5 + 6 * 4);
}

TEST_F(EngineUtilSparseTest, CombineSparseCount) {
  {
    std::array a_ind{0, 1};
    std::array b_ind{2};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              3);
  }

  {
    std::array a_ind{2};
    std::array b_ind{0, 1};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              3);
  }

  {
    std::array a_ind{0, 1};
    std::array b_ind{2, 3, 4};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              5);
  }

  {
    std::array a_ind{5, 6};
    std::array b_ind{1, 3, 8};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              5);
  }

  {
    std::array a_ind{1, 2, 3};
    std::array b_ind{0, 4};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              5);
  }

  {
    std::array a_ind{1, 4};
    std::array b_ind{2, 3};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              4);
  }

  {
    std::array a_ind{0, 1, 3};
    std::array b_ind{0, 3, 4};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              4);
  }

  {
    std::array a_ind{1, 3, 5, 6};
    std::array b_ind{1, 3, 5, 6};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), b_ind.size(), a_ind.data(),
                                     b_ind.data()),
              4);
  }

  EXPECT_EQ(mju_combineSparseCount(0, 0, nullptr, nullptr), 0);

  {
    std::array b_ind{1, 2};
    EXPECT_EQ(mju_combineSparseCount(0, b_ind.size(), nullptr, b_ind.data()),
              2);
  }

  {
    std::array a_ind{0};
    EXPECT_EQ(mju_combineSparseCount(a_ind.size(), 0, a_ind.data(), nullptr),
              1);
  }
}

TEST_F(EngineUtilSparseTest, MjuTranspose3by3) {
  // 1 2 0      1 0 0
  // 0 1 0 -->  2 1 3
  // 0 3 0      0 0 0

  mjtNum mat[] = {1, 2, 1, 3};
  int colind[] = {0, 1, 1, 1};
  int rownnz[] = {2, 1, 1};
  int rowadr[] = {0, 2, 3};

  mjtNum matT[] = {0, 0, 0, 0};
  int colindT[] = {0, 0, 0, 0};
  int rownnzT[] = {0, 0, 0};
  int rowadrT[] = {0, 0, 0};

  mju_transposeSparse(matT, mat, 3, 3, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1, 2, 1, 3));
  EXPECT_THAT(colindT, ElementsAre(0, 0, 1, 2));
  EXPECT_THAT(rownnzT, ElementsAre(1, 3, 0));
  EXPECT_THAT(rowadrT, ElementsAre(0, 1, 4));
}

TEST_F(EngineUtilSparseTest, MjuTranspose1by3) {
  // 1 0 3      1
  //       -->  0
  //            3

  mjtNum mat[] = {1, 3};
  int colind[] = {0, 2};
  int rownnz[] = {2};
  int rowadr[] = {0};

  mjtNum matT[] = {0, 0};
  int colindT[] = {0, 0};
  int rownnzT[] = {0, 0, 0};
  int rowadrT[] = {0, 0, 0};

  mju_transposeSparse(matT, mat, 1, 3, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1, 3));
  EXPECT_THAT(colindT, ElementsAre(0, 0));
  EXPECT_THAT(rownnzT, ElementsAre(1, 0, 1));
  EXPECT_THAT(rowadrT, ElementsAre(0, 1, 1));
}

TEST_F(EngineUtilSparseTest, MjuTranspose3by1) {
  // 1       1 0 3
  // 0  -->
  // 3

  mjtNum mat[] = {1, 3};
  int colind[] = {0, 0};
  int rownnz[] = {1, 0, 1};
  int rowadr[] = {0, 1, 1};

  mjtNum matT[] = {0, 0};
  int colindT[] = {0, 0};
  int rownnzT[] = {0};
  int rowadrT[] = {0};

  mju_transposeSparse(matT, mat, 3, 1, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1, 3));
  EXPECT_THAT(colindT, ElementsAre(0, 2));
  EXPECT_THAT(rownnzT, ElementsAre(2));
  EXPECT_THAT(rowadrT, ElementsAre(0));
}

TEST_F(EngineUtilSparseTest, MjuTransposeDense) {
  // 1 2 3      1 4 7
  // 4 5 6 -->  2 5 8
  // 7 8 9      3 6 9

  mjtNum mat[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int colindT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rownnzT[] = {0, 0, 0};
  int rowadrT[] = {0, 0, 0};
  int rowsuperT[] = {0, 0, 0};

  mju_transposeSparse(matT, mat, 3, 3, rownnzT, rowadrT, colindT, rowsuperT,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1, 4, 7, 2, 5, 8, 3, 6, 9));
  EXPECT_THAT(colindT, ElementsAre(0, 1, 2, 0, 1, 2, 0, 1, 2));
  EXPECT_THAT(rownnzT, ElementsAre(3, 3, 3));
  EXPECT_THAT(rowadrT, ElementsAre(0, 3, 6));
  EXPECT_THAT(rowsuperT, ElementsAre(2, 1, 0));
}

TEST_F(EngineUtilSparseTest, MjuTransposeSuper) {
  //  mat:   0,  1,  2,  0,  3,  4,  5,  0,  0,  0,  0,  0,  0
  //         0,  0,  0,  6,  7,  8,  9, 10, 11, 12,  0,  0,  0
  //
  //  super: 0,  1,  0,  0,  2,  1,  0,  2,  1,  0,  2,  1,  0

  mjtNum mat[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int colind[] = {1, 2, 4, 5, 6, 3, 4, 5, 6, 7, 8, 9};
  int rownnz[] = {5, 7};
  int rowadr[] = {0, 5};

  mjtNum matT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int colindT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rownnzT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rowadrT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rowsuperT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  mju_transposeSparse(matT, mat, 2, 13, rownnzT, rowadrT, colindT, rowsuperT,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1, 2, 6, 3, 7, 4, 8, 5, 9, 10, 11, 12));
  EXPECT_THAT(colindT, ElementsAre(0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1));
  EXPECT_THAT(rownnzT, ElementsAre(0, 1, 1, 1, 2, 2, 2, 1, 1, 1, 0, 0, 0));
  EXPECT_THAT(rowadrT, ElementsAre(0, 0, 1, 2, 3, 5, 7, 9, 10, 11, 12, 12, 12));
  EXPECT_THAT(rowsuperT, ElementsAre(0, 1, 0, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0));
}

TEST_F(EngineUtilSparseTest, MjuTranspose1by1) {
  // 1 -> 1

  mjtNum mat[] = {1};
  int colind[] = {0};
  int rownnz[] = {1};
  int rowadr[] = {0};

  mjtNum matT[] = {0};
  int colindT[] = {0};
  int rownnzT[] = {0};
  int rowadrT[] = {0};

  mju_transposeSparse(matT, mat, 1, 1, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  EXPECT_THAT(matT, ElementsAre(1));
  EXPECT_THAT(colindT, ElementsAre(0));
  EXPECT_THAT(rownnzT, ElementsAre(1));
  EXPECT_THAT(rowadrT, ElementsAre(0));
}

TEST_F(EngineUtilSparseTest, MjuTransposeNullMatrix) {
  // 0 -> 0

  mjtNum* mat = nullptr;
  int* colind = nullptr;
  int rownnz[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int rowadr[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  mjtNum* matT = nullptr;
  int* colindT = nullptr;
  int rownnzT[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int rowadrT[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  mju_transposeSparse(matT, mat, 10, 10, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  EXPECT_THAT(rownnzT, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_THAT(rowadrT, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, MjuCompressSparse) {
  // sparse matrix (uncompressed with spurious values between the rows):
  // [[1, 0, 2]
  //  [0, O, 3]  (second zero represented)
  mjtNum mat[] = {1, 2, 9, 0, 3};   // spurious 9 value
  int colind[] = {0, 2, -1, 1, 2};  // spurious -1 index
  int rownnz[] = {2, 2};
  int rowadr[] = {0, 3};

  mjtNum dense_expected[] = {1, 0, 2, 0, 0, 3};
  mjtNum dense[6];
  mju_sparse2dense(dense, mat, 2, 3, rownnz, rowadr, colind);
  EXPECT_EQ(AsVector(dense, 6), AsVector(dense_expected, 6));

  // check that spurious values are removed
  int nnz = mju_compressSparse(mat, 2, 3, rownnz, rowadr, colind,
                               /*minval=*/-1);
  EXPECT_EQ(nnz, 4);
  mju_sparse2dense(dense, mat, 2, 3, rownnz, rowadr, colind);
  EXPECT_EQ(AsVector(dense, 6), AsVector(dense_expected, 6));

  // check that represented zero gets compressed aways with minval=0
  nnz = mju_compressSparse(mat, 2, 3, rownnz, rowadr, colind, /*minval=*/0);
  EXPECT_EQ(nnz, 3);
  mju_sparse2dense(dense, mat, 2, 3, rownnz, rowadr, colind);
  EXPECT_EQ(AsVector(dense, 6), AsVector(dense_expected, 6));

  // check that 1 gets compressed aways with minval=1
  nnz = mju_compressSparse(mat, 2, 3, rownnz, rowadr, colind, /*minval=*/1);
  EXPECT_EQ(nnz, 2);
  mju_sparse2dense(dense, mat, 2, 3, rownnz, rowadr, colind);
  mjtNum dense_expected_minval1[] = {0, 0, 2, 0, 0, 3};
  EXPECT_EQ(AsVector(dense, 6), AsVector(dense_expected_minval1, 6));
}

TEST_F(EngineUtilSparseTest, MjuSym2Dense) {
  // lower-triangular CSR for a 3x3 symmetric matrix:
  //     1 2 0
  //     2 3 4
  //     0 4 5
  // stored as lower triangle:
  //   row 0: [1]           (col 0)
  //   row 1: [2, 3]        (cols 0, 1)
  //   row 2: [4, 5]        (cols 1, 2)
  mjtNum mat[] = {1, 2, 3, 4, 5};
  int rownnz[] = {1, 2, 2};
  int rowadr[] = {0, 1, 3};
  int colind[] = {0, 0, 1, 1, 2};

  mjtNum dense[9];
  mju_sym2dense(dense, mat, 3, rownnz, rowadr, colind);

  mjtNum expected[] = {1, 2, 0, 2, 3, 4, 0, 4, 5};
  EXPECT_EQ(AsVector(dense, 9), AsVector(expected, 9));
}

TEST_F(EngineUtilSparseTest, MjuSym2DenseWithUpper) {
  mjtNum mat[] = {1, 999, 2, 3, 4, 5};
  int rownnz[] = {2, 2, 2};
  int rowadr[] = {0, 2, 4};
  int colind[] = {0, 1, 0, 1, 1, 2};

  mjtNum dense[9];
  mju_sym2dense(dense, mat, 3, rownnz, rowadr, colind);

  mjtNum expected[] = {1, 2, 0, 2, 3, 4, 0, 4, 5};
  EXPECT_EQ(AsVector(dense, 9), AsVector(expected, 9));
}

// helper: run split-col approach and return dense result
static void SqrMatTDSplitCol(std::vector<mjtNum>& dense_result, int nr, int nc,
                             const mjtNum* mat, const int* rownnz,
                             const int* rowadr, const int* colind,
                             const mjtNum* matT, const int* rownnzT,
                             const int* rowadrT, const int* colindT,
                             const int* rowsuperT, const mjtNum* diag,
                             int* out_diagind, mjData* d) {
  // count mode
  std::vector<int> H_rownnz(nc, 0);
  std::vector<int> H_rowadr(nc, 0);
  int nnz = mju_sqrMatTDSparseSymbolic(
      H_rownnz.data(), H_rowadr.data(), nullptr, out_diagind, nr, nc, rownnz,
      rowadr, colind, rownnzT, rowadrT, colindT, rowsuperT, d);

  // fill mode
  std::vector<int> H_colind(nnz);
  mju_sqrMatTDSparseSymbolic(H_rownnz.data(), H_rowadr.data(), H_colind.data(),
                             out_diagind, nr, nc, rownnz, rowadr, colind,
                             rownnzT, rowadrT, colindT, rowsuperT, d);

  // numeric phase
  std::vector<mjtNum> H(nnz, 0);
  mju_sqrMatTDSparseNumeric(H.data(), nc, H_rownnz.data(), H_rowadr.data(),
                            H_colind.data(), out_diagind, mat, rownnz, rowadr,
                            colind, matT, rownnzT, rowadrT, colindT, rowsuperT,
                            diag, d);

  // densify
  dense_result.assign(nc * nc, 0);
  for (int r = 0; r < nc; r++) {
    for (int j = 0; j < H_rownnz[r]; j++) {
      int c = H_colind[H_rowadr[r] + j];
      dense_result[r * nc + c] = H[H_rowadr[r] + j];
    }
  }
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse1) {
  //       0 0 0
  // M  =  0 0 0
  //       0 0 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  int colindT[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 3};
  int rowadrT[] = {0, 3, 6};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, nullptr, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseLower) {
  //     2 -1  1
  // M = 2 -1  2
  //     2  2  3

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {2, -1, 1, 2, -1, 2, 2, 2, 3};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {2, 2, 2, -1, -1, 2, 1, 2, 3};
  int colindT[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 3};
  int rowadrT[] = {0, 3, 6};

  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, nullptr, nullptr, data.get());

  EXPECT_THAT(dense, ElementsAre(12, 0, 0, 0, 6, 0, 12, 3, 14));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse2) {
  //     2 -1  1
  // M = 2 -1  2
  //     2  2  3

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {2, -1, 1, 2, -1, 2, 2, 2, 3};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {2, 2, 2, -1, -1, 2, 1, 2, 3};
  int colindT[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 3};
  int rowadrT[] = {0, 3, 6};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, nullptr, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(12, 0, 12, 0, 6, 3, 12, 3, 14));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse3) {
  //     1 2 0
  // M = 0 3 0
  //     4 0 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 3, 4};
  int colind[] = {0, 1, 1, 0};
  int rownnz[] = {2, 1, 1};
  int rowadr[] = {0, 2, 3};

  mjtNum matT[] = {1, 4, 2, 3};
  int colindT[] = {0, 2, 0, 1};
  int rownnzT[] = {2, 2, 0};
  int rowadrT[] = {0, 2, 4};

  mjtNum diag[] = {2, 3, 4};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(66, 4, 0, 4, 35, 0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse3b) {
  //     1 2 0
  // M = 0 3 4
  //     5 0 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 3, 4, 5};
  int colind[] = {0, 1, 1, 2, 0};
  int rownnz[] = {2, 2, 1};
  int rowadr[] = {0, 2, 4};

  mjtNum matT[] = {1, 5, 2, 3, 4};
  int colindT[] = {0, 2, 0, 1, 1};
  int rownnzT[] = {2, 2, 1};
  int rowadrT[] = {0, 2, 4};

  mjtNum diag[] = {1, 1, 1};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(26, 2, 0, 2, 13, 12, 0, 12, 16));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse4) {
  //     1 0 2
  // M = 0 0 3
  //     4 0 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 3, 4};
  int colind[] = {0, 2, 2, 0};
  int rownnz[] = {2, 1, 1};
  int rowadr[] = {0, 2, 3};

  mjtNum matT[] = {1, 4, 2, 3};
  int colindT[] = {0, 2, 0, 1};
  int rownnzT[] = {2, 0, 2};
  int rowadrT[] = {0, 2, 2};

  mjtNum diag[] = {2, 3, 4};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(66, 0, 4, 0, 0, 0, 4, 0, 35));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse5) {
  //     1 0 4
  // M = 0 0 0
  //     2 3 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 4, 2, 3};
  int colind[] = {0, 2, 0, 1};
  int rownnz[] = {2, 0, 2};
  int rowadr[] = {0, 2, 2};

  mjtNum matT[] = {1, 2, 3, 4};
  int colindT[] = {0, 2, 2, 0};
  int rownnzT[] = {2, 1, 1};
  int rowadrT[] = {0, 2, 3};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, nullptr, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(5, 6, 4, 6, 9, 0, 4, 0, 16));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse6) {
  //     1 0 2
  // M = 0 2 0
  //     0 0 3

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 2, 3};
  int colind[] = {0, 2, 1, 2};
  int rownnz[] = {2, 1, 1};
  int rowadr[] = {0, 2, 3};

  mjtNum matT[] = {1, 2, 2, 3};
  int colindT[] = {0, 1, 0, 2};
  int rownnzT[] = {1, 1, 2};
  int rowadrT[] = {0, 1, 2};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, nullptr, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(1, 0, 2, 0, 4, 0, 2, 0, 13));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse7) {
  //     1 2
  // M = 0 3
  //     4 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 3, 4};
  int colind[] = {0, 1, 1, 0};
  int rownnz[] = {2, 1, 1};
  int rowadr[] = {0, 2, 3};

  mjtNum matT[] = {1, 4, 2, 3};
  int colindT[] = {0, 2, 0, 1};
  int rownnzT[] = {2, 2};
  int rowadrT[] = {0, 2};

  mjtNum diag[] = {2, 3, 4};

  int diagindH[2];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 2, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(66, 4, 4, 35));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse8) {
  // M = 1 0 4
  //     2 3 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 4, 2, 3};
  int colind[] = {0, 2, 0, 1};
  int rownnz[] = {2, 2};
  int rowadr[] = {0, 2};

  mjtNum matT[] = {1, 2, 3, 4};
  int colindT[] = {0, 1, 1, 0};
  int rownnzT[] = {2, 1, 1};
  int rowadrT[] = {0, 2, 3};

  mjtNum diag[] = {2, 3};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 2, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(14, 18, 8, 18, 27, 0, 8, 0, 32));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse9) {
  //     1 2 2
  // M = 1 3 4
  //     4 4 4

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 2, 1, 3, 4, 4, 4, 4};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {1, 1, 4, 2, 3, 4, 2, 4, 4};
  int colindT[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 3};
  int rowadrT[] = {0, 3, 6};

  mjtNum diag[] = {2, 3, 4};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, nullptr, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(69, 77, 80, 77, 99, 108, 80, 108, 120));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse10) {
  //     1 2 3
  // M = 2 3 2
  //     3 1 1

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 2, 3, 2, 3, 2, 3, 1, 1};
  int colind[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnz[] = {3, 3, 3};
  int rowadr[] = {0, 3, 6};

  mjtNum matT[] = {1, 2, 3, 2, 3, 1, 3, 2, 1};
  int colindT[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 3};
  int rowadrT[] = {0, 3, 6};
  int rowsuperT[] = {2, 1, 0};

  mjtNum diag[] = {1, 2, 1};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, rowsuperT, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(18, 17, 14, 17, 23, 19, 14, 19, 18));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse11) {
  //     1 1 1
  // M = 0 0 0
  //     0 3 3

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 1, 1, 3, 3};
  int colind[] = {0, 1, 2, 1, 2};
  int rownnz[] = {3, 0, 2};
  int rowadr[] = {0, 3, 3};

  mjtNum matT[] = {1, 1, 3, 1, 3};
  int colindT[] = {0, 0, 2, 0, 2};
  int rownnzT[] = {1, 2, 2};
  int rowadrT[] = {0, 1, 3};
  int rowsuperT[] = {0, 1, 0};

  mjtNum diag[] = {1, 1, 1};

  int diagindH[3];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 3, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, rowsuperT, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(1, 1, 1, 1, 10, 10, 1, 10, 10));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse12) {
  //     1 1 1 1
  // M = 0 0 0 0
  //     0 0 3 3

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 1, 1, 1, 3, 3};
  int colind[] = {0, 1, 2, 3, 2, 3};
  int rownnz[] = {4, 0, 2};
  int rowadr[] = {0, 4, 4};

  mjtNum matT[] = {1, 1, 1, 3, 1, 3};
  int colindT[] = {0, 0, 0, 2, 0, 2};
  int rownnzT[] = {1, 1, 2, 2};
  int rowadrT[] = {0, 1, 2, 4};
  int rowsuperT[] = {1, 0, 1, 0};

  mjtNum diag[] = {1, 1, 1};

  int diagindH[4];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 4, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, rowsuperT, diag, diagindH, data.get());

  EXPECT_THAT(dense,
              ElementsAre(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 10, 10, 1, 1, 10, 10));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse13) {
  //     1 1 0 0 0
  // M = 1 1 0 0 0
  //     1 1 0 0 0

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 1, 1, 1, 1, 1};
  int colind[] = {0, 1, 0, 1, 0, 1};
  int rownnz[] = {2, 2, 2};
  int rowadr[] = {0, 2, 4};

  mjtNum matT[] = {1, 1, 1, 1, 1, 1};
  int colindT[] = {0, 1, 2, 0, 1, 2};
  int rownnzT[] = {3, 3, 0, 0, 0};
  int rowadrT[] = {0, 3, 6, 6, 6};
  int rowsuperT[] = {1, 0, 2, 1, 0};

  mjtNum diag[] = {1, 1, 1};

  int diagindH[5];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 3, 5, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, rowsuperT, diag, diagindH, data.get());

  EXPECT_THAT(dense, ElementsAre(3, 3, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparse14) {
  // M = 1 1 1 1 2 2 2

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  mjtNum mat[] = {1, 1, 1, 1, 2, 2, 2};
  int colind[] = {0, 1, 2, 3, 4, 5, 6};
  int rownnz[] = {7};
  int rowadr[] = {0};

  mjtNum matT[] = {1, 1, 1, 1, 2, 2, 2};
  int colindT[] = {0, 0, 0, 0, 0, 0, 0};
  int rownnzT[] = {1, 1, 1, 1, 1, 1, 1};
  int rowadrT[] = {0, 1, 2, 3, 4, 5, 6};
  int rowsuperT[] = {3, 2, 1, 0, 2, 1, 0};

  int diagindH[7];
  std::vector<mjtNum> dense;
  SqrMatTDSplitCol(dense, 1, 7, mat, rownnz, rowadr, colind, matT, rownnzT,
                   rowadrT, colindT, rowsuperT, nullptr, diagindH, data.get());

  EXPECT_THAT(dense,
              ElementsAre(1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1,
                          2, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2,
                          2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseSymbolic) {
  // Simple dense 2x2 matrix:
  //     1 2
  // M = 3 4
  //
  // M'M (lower triangle) should have 3 elements: (0,0), (1,0), (1,1)

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  // M in CSR: row 0 has cols 0,1; row 1 has cols 0,1
  int colind[] = {0, 1, 0, 1};
  int rownnz[] = {2, 2};
  int rowadr[] = {0, 2};

  // compute transpose using mju_transposeSparse
  mjtNum mat[] = {1, 2, 3, 4};
  mjtNum matT[4];
  int colindT[4];
  int rownnzT[2];
  int rowadrT[2];
  mju_transposeSparse(matT, mat, 2, 2, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  // use old function as ground truth
  int rownnzH_expected[] = {0, 0};
  int rowadrH_expected[] = {0, 0};
  int nnz_expected = mju_sqrMatTDSparseCount(
      rownnzH_expected, rowadrH_expected, 2, rownnz, rowadr, colind, rownnzT,
      rowadrT, colindT, nullptr, data.get(), /*flg_upper=*/0);

  // verify: lower triangle should have 3 elements: (0,0), (1,0), (1,1)
  EXPECT_EQ(nnz_expected, 3);
  EXPECT_THAT(rownnzH_expected, ElementsAre(1, 2));
  EXPECT_THAT(rowadrH_expected, ElementsAre(0, 1));

  // test count mode of new function
  int rownnzH[] = {0, 0};
  int rowadrH[] = {0, 0};

  int nnz = mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, nullptr, 2, 2,
                                       rownnz, rowadr, colind, rownnzT, rowadrT,
                                       colindT, nullptr, data.get());

  EXPECT_EQ(nnz, nnz_expected);
  EXPECT_THAT(rownnzH, ElementsAre(rownnzH_expected[0], rownnzH_expected[1]));
  EXPECT_THAT(rowadrH, ElementsAre(rowadrH_expected[0], rowadrH_expected[1]));

  // test fill mode
  std::vector<int> colindH(nnz, -1);

  mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, colindH.data(), nullptr, 2, 2,
                             rownnz, rowadr, colind, rownnzT, rowadrT, colindT,
                             nullptr, data.get());

  // verify: row 0 should have {0}, row 1 should have {0, 1}
  EXPECT_THAT(colindH, ElementsAre(0, 0, 1));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseSymbolicUpper) {
  // Test flg_upper=1: count both lower and upper triangle
  // Same matrix as previous test

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  int colind[] = {0, 1, 0, 1};
  int rownnz[] = {2, 2};
  int rowadr[] = {0, 2};

  mjtNum mat[] = {1, 2, 3, 4};
  mjtNum matT[4];
  int colindT[4];
  int rownnzT[2];
  int rowadrT[2];
  mju_transposeSparse(matT, mat, 2, 2, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  // use old function as ground truth with flg_upper=1
  int rownnzH_expected[] = {0, 0};
  int rowadrH_expected[] = {0, 0};
  int nnz_expected = mju_sqrMatTDSparseCount(
      rownnzH_expected, rowadrH_expected, 2, rownnz, rowadr, colind, rownnzT,
      rowadrT, colindT, nullptr, data.get(), /*flg_upper=*/1);

  // test new function with diagind (upper triangle)
  int rownnzH[] = {0, 0};
  int rowadrH[] = {0, 0};
  int diagindH[] = {0, 0};
  int nnz = mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, diagindH, 2,
                                       2, rownnz, rowadr, colind, rownnzT,
                                       rowadrT, colindT, nullptr, data.get());

  EXPECT_EQ(nnz, nnz_expected);
  EXPECT_THAT(rownnzH, ElementsAre(rownnzH_expected[0], rownnzH_expected[1]));
  EXPECT_THAT(rowadrH, ElementsAre(rowadrH_expected[0], rowadrH_expected[1]));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseSymbolicSupernode) {
  // Test supernode exploitation with a matrix that has supernodes
  // M has two rows with identical sparsity pattern

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  // 3x2 matrix where rows 1 and 2 have same pattern
  //     1 0
  // M = 2 3
  //     4 5
  int colind[] = {0, 0, 1, 0, 1};
  int rownnz[] = {1, 2, 2};
  int rowadr[] = {0, 1, 3};

  mjtNum mat[] = {1, 2, 3, 4, 5};
  mjtNum matT[5];
  int colindT[5];
  int rownnzT[2];
  int rowadrT[2];
  mju_transposeSparse(matT, mat, 3, 2, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  // compute rowsuperT
  int rowsuperT[2];
  mju_superSparse(2, rowsuperT, rownnzT, rowadrT, colindT);

  // use old function as ground truth
  int rownnzH_expected[] = {0, 0};
  int rowadrH_expected[] = {0, 0};
  int nnz_expected = mju_sqrMatTDSparseCount(
      rownnzH_expected, rowadrH_expected, 2, rownnz, rowadr, colind, rownnzT,
      rowadrT, colindT, rowsuperT, data.get(), /*flg_upper=*/0);

  // test new function with supernodes
  int rownnzH[] = {0, 0};
  int rowadrH[] = {0, 0};
  int nnz = mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, nullptr, 3, 2,
                                       rownnz, rowadr, colind, rownnzT, rowadrT,
                                       colindT, rowsuperT, data.get());

  EXPECT_EQ(nnz, nnz_expected);
  EXPECT_THAT(rownnzH, ElementsAre(rownnzH_expected[0], rownnzH_expected[1]));
  EXPECT_THAT(rowadrH, ElementsAre(rowadrH_expected[0], rowadrH_expected[1]));

  // test fill mode with supernodes
  std::vector<int> colindH(nnz, -1);
  mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, colindH.data(), nullptr, 3, 2,
                             rownnz, rowadr, colind, rownnzT, rowadrT, colindT,
                             rowsuperT, data.get());

  // verify all filled
  for (int i = 0; i < nnz; i++) {
    EXPECT_GE(colindH[i], 0) << "colindH[" << i << "] not filled";
  }

  // verify numeric phase with supernodes
  std::vector<mjtNum> resH(nnz);
  mjtNum diag[] = {1, 1, 1, 1, 1};  // dummy diagonal
  mju_sqrMatTDSparseNumeric(resH.data(), 2, rownnzH, rowadrH, colindH.data(),
                            nullptr, mat, rownnz, rowadr, colind, matT, rownnzT,
                            rowadrT, colindT, rowsuperT, diag, data.get());

  // ground truth numeric
  std::vector<mjtNum> res_expected(4);
  std::vector<int> colindH_expected(4);
  int rownnzH_exp[] = {0, 0};
  int rowadrH_exp[] = {0, 2};
  mju_sqrMatTDSparse(res_expected.data(), mat, matT, diag, 3, 2, rownnzH_exp,
                     rowadrH_exp, colindH_expected.data(), rownnz, rowadr,
                     colind, nullptr, rownnzT, rowadrT, colindT, rowsuperT,
                     data.get(), nullptr);

  // compare values (sparse result vs sparse ground truth)
  for (int r = 0; r < 2; r++) {
    for (int i = 0; i < rownnzH[r]; i++) {
      // find matching col in ground truth
      int c = colindH[rowadrH[r] + i];
      mjtNum val = resH[rowadrH[r] + i];

      bool found = false;
      for (int j = 0; j < rownnzH_exp[r]; j++) {
        if (colindH_expected[rowadrH_exp[r] + j] == c) {
          EXPECT_NEAR(val, res_expected[rowadrH_exp[r] + j], 1e-14);
          found = true;
          break;
        }
      }
      EXPECT_TRUE(found) << "Column " << c
                         << " not found in ground truth for row " << r;
    }
  }
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseNumeric) {
  // Test numeric phase using symbolic phase + existing function as ground truth
  //     1 2
  // M = 3 4

  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  int colind[] = {0, 1, 0, 1};
  int rownnz[] = {2, 2};
  int rowadr[] = {0, 2};
  mjtNum mat[] = {1, 2, 3, 4};

  // compute transpose
  mjtNum matT[4];
  int colindT[4];
  int rownnzT[2];
  int rowadrT[2];
  mju_transposeSparse(matT, mat, 2, 2, rownnzT, rowadrT, colindT, nullptr,
                      rownnz, rowadr, colind);

  // compute supernodes
  int rowsuperT[2];
  mju_superSparse(2, rowsuperT, rownnzT, rowadrT, colindT);

  mjtNum diag[] = {2, 3};  // diagonal weighting matrix

  // test both diagind cases: lower-only (diagind=NULL) and both triangles
  // (diagind!=NULL)
  for (int use_diagind = 0; use_diagind <= 1; use_diagind++) {
    // compute sparsity pattern using symbolic phase
    int rownnzH[] = {0, 0};
    int rowadrH[] = {0, 0};
    int diagindH[] = {0, 0};
    int nnz = mju_sqrMatTDSparseSymbolic(
        rownnzH, rowadrH, nullptr, use_diagind ? diagindH : nullptr, 2, 2,
        rownnz, rowadr, colind, rownnzT, rowadrT, colindT, nullptr, data.get());

    std::vector<int> colindH(nnz);
    mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, colindH.data(),
                               use_diagind ? diagindH : nullptr, 2, 2, rownnz,
                               rowadr, colind, rownnzT, rowadrT, colindT,
                               nullptr, data.get());

    // compute values using numeric phase
    std::vector<mjtNum> resH(nnz);
    mju_sqrMatTDSparseNumeric(resH.data(), 2, rownnzH, rowadrH, colindH.data(),
                              use_diagind ? diagindH : nullptr, mat, rownnz,
                              rowadr, colind, matT, rownnzT, rowadrT, colindT,
                              rowsuperT, diag, data.get());

    // compute ground truth using existing mju_sqrMatTDSparse
    // use uncompressed storage to give the old function enough room
    std::vector<mjtNum> res_expected(4);  // 2x2 uncompressed
    std::vector<int> colindH_expected(4);
    int rownnzH_exp[] = {0, 0};
    int rowadrH_exp[] = {0, 2};
    int diagind_exp[] = {0, 0};
    mju_sqrMatTDSparse(res_expected.data(), mat, matT, diag, 2, 2, rownnzH_exp,
                       rowadrH_exp, colindH_expected.data(), rownnz, rowadr,
                       colind, nullptr, rownnzT, rowadrT, colindT, nullptr,
                       data.get(), use_diagind ? diagind_exp : nullptr);

    // check that rownnz matches (nnz may differ due to compressed vs
    // uncompressed storage)
    EXPECT_EQ(rownnzH[0], rownnzH_exp[0])
        << "rownnz[0] mismatch for use_diagind=" << use_diagind;
    EXPECT_EQ(rownnzH[1], rownnzH_exp[1])
        << "rownnz[1] mismatch for use_diagind=" << use_diagind;

    // compare column indices and values for each row
    for (int r = 0; r < 2; r++) {
      for (int j = 0; j < rownnzH[r]; j++) {
        int idx = rowadrH[r] + j;
        int idx_exp = rowadrH_exp[r] + j;
        EXPECT_EQ(colindH[idx], colindH_expected[idx_exp])
            << "colind mismatch at row " << r << " pos " << j
            << " for use_diagind=" << use_diagind;
        EXPECT_NEAR(resH[idx], res_expected[idx_exp], 1e-10)
            << "value mismatch at row " << r << " pos " << j
            << " for use_diagind=" << use_diagind;
      }
    }
  }

  // Test superset sparsity pattern (as in HessianConeFolded where M is merged
  // into H_colind):
  // - col 0 is unconstrained (rownnzT[0] == 0) but present in H_colind
  // - cols 1, 2 form a JT supernode, with an extra off-diagonal (2, 0) in H
  int J_colind[] = {1, 2};
  int J_rownnz[] = {2};
  int J_rowadr[] = {0};
  mjtNum J[] = {2, 3};
  mjtNum D[] = {5};

  mjtNum JT[2];
  int JT_colind[2], JT_rownnz[3], JT_rowadr[3], JT_rowsuper[3];
  mju_transposeSparse(JT, J, 1, 3, JT_rownnz, JT_rowadr, JT_colind, JT_rowsuper,
                      J_rownnz, J_rowadr, J_colind);

  int H_rownnz[] = {1, 1, 3};
  int H_rowadr[] = {0, 1, 2};
  int H_colind[] = {0, 1, 0, 1, 2};
  mjtNum H[] = {-99, -99, -99, -99, -99};

  mju_sqrMatTDSparseNumeric(H, 3, H_rownnz, H_rowadr, H_colind, nullptr, J,
                            J_rownnz, J_rowadr, J_colind, JT, JT_rownnz,
                            JT_rowadr, JT_colind, JT_rowsuper, D, data.get());
  EXPECT_THAT(AsVector(H, 5), ElementsAre(0, 20, 0, 30, 45));
}

TEST_F(EngineUtilSparseTest, MjuMulMatTVec) {
  int nr = 2;
  int nc = 3;
  mjtNum mat[] = {1, 2, 0, 0, 3, 4};

  mjtNum mat_sparse[4];
  int rownnz[2];
  int rowadr[2];
  int colind[4];
  mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, 4);

  // multiply: res = mat' * vec
  mjtNum vec[] = {5, 6};
  mjtNum res[3];
  mju_mulMatTVecSparse(res, mat_sparse, vec, nr, nc, rownnz, rowadr, colind);

  EXPECT_THAT(AsVector(res, 3), ElementsAre(5, 28, 24));
}

TEST_F(EngineUtilSparseTest, MjuAddToSymSparse) {
  //     1 2 4
  // M = 2 3 0
  //     4 0 5

  // only lower triangle represented
  mjtNum mat[] = {1, 2, 3, 4, 5};
  int colind[] = {0, 0, 1, 0, 2};
  int rownnz[] = {1, 2, 2};
  int rowadr[] = {0, 1, 3};

  //     0 0 0
  // A = 5 4 2
  //     4 3 2
  mjtNum A[] = {0, 0, 0, 5, 4, 2, 4, 3, 2};

  mju_addToSymSparse(A, mat, 3, rownnz, rowadr, colind, /*flg_upper=*/1);
  EXPECT_THAT(AsVector(A, 9), ElementsAre(1, 2, 4, 7, 7, 2, 8, 3, 7));

  // same as A
  mjtNum B[] = {0, 0, 0, 5, 4, 2, 4, 3, 2};

  mju_addToSymSparse(B, mat, 3, rownnz, rowadr, colind, /*flg_upper=*/0);
  EXPECT_THAT(AsVector(B, 9), ElementsAre(1, 0, 0, 7, 7, 2, 8, 3, 7));
}

TEST_F(EngineUtilSparseTest, MjuMulSymVecSparse) {
  constexpr int n = 4;
  constexpr int nnz = 9;

  mjtNum mat[n * n] = {1, 0, 0, 0, 0, 2, 0, 0, 3, 0, 4, 0, 5, 6, 7, 8};

  // dense, full matrix
  mjtNum sym[n * n] = {1, 0, 3, 5, 0, 2, 0, 6, 3, 0, 4, 7, 5, 6, 7, 8};

  mjtNum mat_sparse[nnz];
  int rownnz[n];
  int rowadr[n];
  int colind[nnz];
  mju_dense2sparse(mat_sparse, mat, n, n, rownnz, rowadr, colind, nnz);

  // multiply: res = (mat + strict_upper(mat')) * vec
  mjtNum vec[n] = {4, 3, 2, 1};
  mjtNum res[n];
  mju_mulSymVecSparse(res, mat_sparse, vec, n, rownnz, rowadr, colind);

  // dense multiply
  mjtNum res2[n];
  mju_mulMatVec(res2, sym, vec, n, n);

  for (int i = 0; i < n; i++) {
    EXPECT_EQ(res[i], res2[i]);
  }
}

TEST_F(EngineUtilSparseTest, MjuDenseToSparse) {
  int nr = 2;
  int nc = 2;
  mjtNum mat[] = {1, 2, 0, 3};

  mjtNum mat_sparse[4];
  int rownnz[2];
  int rowadr[2];
  int colind[4];

  // nnz == number of non-zeros
  int status3 =
      mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, 3);

  EXPECT_EQ(status3, 0);

  // nnz > number of non-zeros
  int status4 =
      mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, 4);

  EXPECT_EQ(status4, 0);

  // nnz < number of non-zeros
  int status2 =
      mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, 2);

  EXPECT_EQ(status2, 1);

  // nnz == 0
  int status0 =
      mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, 0);

  EXPECT_EQ(status0, 1);
}

TEST_F(EngineUtilSparseTest, MergeSorted) {
  const int chain1_a[] = {1, 2, 3};
  const int* chain2_a = nullptr;
  int merged_a[3];
  int n1 = 3;
  int n2 = 0;
  EXPECT_EQ(mj_mergeSorted(merged_a, chain1_a, n1, chain2_a, n2), 3);
  EXPECT_THAT(merged_a, ElementsAre(1, 2, 3));

  const int chain1_b[] = {1, 3, 5, 7, 8};
  const int chain2_b[] = {2, 4, 5, 6, 8};
  int merged_b[8];
  n1 = 5;
  n2 = 5;
  EXPECT_EQ(mj_mergeSorted(merged_b, chain1_b, n1, chain2_b, n2), 8);
  EXPECT_THAT(merged_b, ElementsAre(1, 2, 3, 4, 5, 6, 7, 8));
}

TEST_F(EngineUtilSparseTest, BlockDiag) {
  // 4x5 matrix with 3 blocks
  constexpr int nr = 4;
  constexpr int nc = 5;
  const mjtNum mat[nr * nc] = {1, 2, 0, 0, 0, 0, 0, 3, 4, 0,
                               0, 0, 5, 6, 0, 0, 0, 0, 0, 7};

  // block structure
  constexpr int nb = 3;
  const int block_nr[nb] = {1, 2, 1};
  const int block_nc[nb] = {2, 2, 1};
  const int block_r[nb] = {0, 1, 3};
  const int block_c[nb] = {0, 2, 4};

  // test with identity permutations
  const int perm_r[nr] = {0, 1, 2, 3};
  const int perm_c[nc] = {0, 1, 2, 3, 4};
  mjtNum res[nr * nc] = {0};
  mju_blockDiag(res, mat, nc, nc, nb, perm_r, perm_c, block_nr, block_nc,
                block_r, block_c);
  EXPECT_THAT(res, ElementsAre(1, 2, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7,
                               0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, BlockDiagPerm) {
  // 4x5 matrix with 3 blocks
  constexpr int nr = 4;
  constexpr int nc = 5;
  const mjtNum mat[nr * nc] = {1, 2, 0, 0, 0, 0, 0, 3, 4, 0,
                               0, 0, 5, 6, 0, 0, 0, 0, 0, 7};

  // block structure
  constexpr int nb = 3;
  const int block_nr[nb] = {1, 2, 1};
  const int block_nc[nb] = {2, 2, 1};
  const int block_r[nb] = {0, 1, 3};
  const int block_c[nb] = {0, 2, 4};

  // scatter mat into mat_p
  const int perm_r[nr] = {1, 3, 2, 0};
  const int perm_c[nc] = {2, 0, 4, 3, 1};
  mjtNum mat_p[nr * nc];
  PermuteMat(mat_p, mat, nr, nc, perm_r, perm_c, true, true);

  // test with permutation
  mjtNum res[nr * nc] = {0};
  mju_blockDiag(res, mat_p, nc, nc, nb, perm_r, perm_c, block_nr, block_nc,
                block_r, block_c);
  EXPECT_THAT(res, ElementsAre(1, 2, 0, 0, 0, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0, 7,
                               0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, BlockDiagLessCols) {
  // 4x5 matrix with 3 blocks
  constexpr int nr = 4;
  constexpr int nc = 5;
  const mjtNum mat[nr * nc] = {1, 2, 0, 0, 0, 0, 0, 3, 4, 0,
                               0, 0, 5, 6, 0, 0, 0, 0, 0, 7};

  // block structure (ignore middle block)
  constexpr int nb = 2;
  const int block_nr[nb] = {1, 1};
  const int block_nc[nb] = {2, 1};
  const int block_r[nb] = {0, 3};
  const int block_c[nb] = {0, 4};

  // scatter mat into mat_p
  const int perm_r[nr] = {1, 3, 2, 0};
  const int perm_c[nc] = {2, 0, 4, 3, 1};
  mjtNum mat_p[nr * nc];
  PermuteMat(mat_p, mat, nr, nc, perm_r, perm_c, true, true);

  // test with permutation and less columns (ignore middle block)
  constexpr int nc_res = 3;
  mjtNum res2[nr * nc_res] = {0};
  mju_blockDiag(res2, mat_p, nc, nc_res, nb, perm_r, perm_c, block_nr, block_nc,
                block_r, block_c);
  EXPECT_THAT(res2, ElementsAre(1, 2, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0));
}

TEST_F(EngineUtilSparseTest, BlockDiagSparse) {
  // 4x5 matrix with 3 blocks
  constexpr int nr = 4;
  constexpr int nc = 5;
  const mjtNum mat[nr * nc] = {1, 2, 0, 0, 0, 0, 0, 3, 4, 0,
                               0, 0, 5, 6, 0, 0, 0, 0, 0, 7};
  constexpr int nnz = 7;

  // block structure
  constexpr int nb = 3;
  const int block_r[nb] = {0, 1, 3};
  const int block_c[nb] = {0, 2, 4};

  // convert to sparse
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum mat_sparse[nnz];
  mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, nnz);

  // test with identity permutations
  const int perm_r[nr] = {0, 1, 2, 3};
  const int perm_c[nc] = {0, 1, 2, 3, 4};
  int res_rownnz[nr];
  int res_rowadr[nr];
  int res_colind[nnz];
  mjtNum res[nnz];
  mju_blockDiagSparse(res, res_rownnz, res_rowadr, res_colind, mat_sparse,
                      rownnz, rowadr, colind, nr, nb, perm_r, perm_c, block_r,
                      block_c, nullptr, nullptr);
  mjtNum dense_res[nr * nc];
  mju_sparse2dense(dense_res, res, nr, nc, res_rownnz, res_rowadr, res_colind);
  EXPECT_THAT(dense_res, ElementsAre(1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 5, 6, 0, 0,
                                     0, 7, 0, 0, 0, 0));

  // permute mat into mat_p (scatter rows, gather columns)
  const int perm_r2[nr] = {3, 1, 0, 2};
  const int perm_c2[nc] = {4, 0, 2, 1, 3};
  mjtNum mat_p[nr * nc];
  PermuteMat(mat_p, mat, nr, nc, perm_r2, perm_c2, true, false);
  mju_dense2sparse(mat_sparse, mat_p, nr, nc, rownnz, rowadr, colind, nnz);

  // test with permutation
  mju_blockDiagSparse(res, res_rownnz, res_rowadr, res_colind, mat_sparse,
                      rownnz, rowadr, colind, nr, nb, perm_r2, perm_c2, block_r,
                      block_c, nullptr, nullptr);
  mju_sparse2dense(dense_res, res, nr, nc, res_rownnz, res_rowadr, res_colind);
  EXPECT_THAT(dense_res, ElementsAre(1, 2, 0, 0, 0, 3, 4, 0, 0, 0, 5, 6, 0, 0,
                                     0, 7, 0, 0, 0, 0));
}

TEST_F(EngineUtilSparseTest, BlockDiagSparseTranspose) {
  // 4x5 matrix with 3 blocks
  constexpr int nr = 4;
  constexpr int nc = 5;
  const mjtNum mat[nr * nc] = {1, 2, 0, 0, 0, 0, 0, 3, 4, 0,
                               0, 0, 5, 6, 0, 0, 0, 0, 0, 7};
  constexpr int nnz = 7;

  // block structure
  constexpr int nb = 3;
  const int block_nr[nb] = {1, 2, 1};
  const int block_nc[nb] = {2, 2, 1};
  const int block_r[nb] = {0, 1, 3};
  const int block_c[nb] = {0, 2, 4};

  // convert to sparse
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum mat_sparse[nnz];
  mju_dense2sparse(mat_sparse, mat, nr, nc, rownnz, rowadr, colind, nnz);

  // block diagonalize
  const int perm_r[nr] = {0, 1, 2, 3};
  const int perm_c[nc] = {0, 1, 2, 3, 4};
  int res_rownnz[nr];
  int res_rowadr[nr];
  int res_colind[nnz];
  mjtNum res[nnz];
  mju_blockDiagSparse(res, res_rownnz, res_rowadr, res_colind, mat_sparse,
                      rownnz, rowadr, colind, nr, nb, perm_r, perm_c, block_r,
                      block_c, nullptr, nullptr);

  // transpose each block
  mjtNum matT[nnz];
  int colindT[nnz];
  int rownnzT[nc];  // Max possible size for rownnzT is nc
  int rowadrT[nc];

  mjtNum denseT[nr * nc];
  mjtNum dense_block[nr * nc];
  mjtNum dense_block_T[nr * nc];

  for (int b = 0; b < nb; ++b) {
    int bnr = block_nr[b];
    int bnc = block_nc[b];
    int r_offset = block_r[b];
    int c_offset = block_c[b];

    if (bnr == 0 || bnc == 0) continue;

    int block_start_adr = res_rowadr[r_offset];

    // pointers to the start of the current block
    mjtNum* block_res_vals = res + block_start_adr;
    int* block_res_rownnz = res_rownnz + r_offset;
    int* block_res_rowadr = res_rowadr + r_offset;
    int* block_res_colind = res_colind + block_start_adr;

    mju_transposeSparse(matT, block_res_vals, bnr, bnc, rownnzT, rowadrT,
                        colindT, nullptr, block_res_rownnz, block_res_rowadr,
                        block_res_colind);

    // verification:
    // 1. convert transposed sparse block to dense
    mju_zero(denseT, bnc * bnr);
    mju_sparse2dense(denseT, matT, bnc, bnr, rownnzT, rowadrT, colindT);

    // 2. extract original block to dense
    for (int i = 0; i < bnr; ++i) {
      for (int j = 0; j < bnc; ++j) {
        dense_block[i * bnc + j] = mat[(r_offset + i) * nc + (c_offset + j)];
      }
    }
    // 3. manually transpose the original dense block
    for (int i = 0; i < bnr; ++i) {
      for (int j = 0; j < bnc; ++j) {
        dense_block_T[j * bnr + i] = dense_block[i * bnc + j];
      }
    }

    // 4. Compare
    for (int i = 0; i < bnc * bnr; ++i) {
      EXPECT_EQ(denseT[i], dense_block_T[i])
          << "block " << b << " element " << i;
    }
  }
}

TEST_F(EngineUtilSparseTest, PermuteMat) {
  const mjtNum mat[] = {1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6};
  const int perm_r[] = {2, 0, 1};
  const int perm_c[] = {3, 2, 0, 1};
  mjtNum gather[3 * 4];
  PermuteMat(gather, mat, 3, 4, perm_r, perm_c, false, false);
  EXPECT_THAT(gather, ElementsAre(6, 5, 0, 0, 0, 0, 1, 2, 4, 3, 0, 0));
  mjtNum scatter[3 * 4];
  PermuteMat(scatter, gather, 3, 4, perm_r, perm_c, true, true);
  EXPECT_THAT(scatter, ElementsAre(1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6));
  mjtNum mixed[3 * 4];
  PermuteMat(mixed, mat, 3, 4, perm_r, perm_c, true, false);
  EXPECT_THAT(mixed, ElementsAre(4, 3, 0, 0, 6, 5, 0, 0, 0, 0, 1, 2));
  mjtNum mixed_back[3 * 4];
  PermuteMat(mixed_back, mixed, 3, 4, perm_r, perm_c, false, true);
  EXPECT_THAT(mixed_back, ElementsAre(1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 5, 6));
}

TEST_F(EngineUtilSparseTest, MjuSqrMatTDSparseSymbolicEmptyRowAndGuards) {
  MjModelPtr model = LoadModelFromString("<mujoco/>");
  MjDataPtr data = MakeData(model);

  // 2x3 matrix M with empty column 1: [[1, 0, 2], [0, 0, 3]]
  int colind[] = {0, 2, 2};
  int rownnz[] = {2, 1};
  int rowadr[] = {0, 2};
  int colindT[] = {0, 0, 1};
  int rownnzT[] = {1, 0, 2};
  int rowadrT[] = {0, 1, 1};

  int rownnzH[] = {0, 0, 0};
  int rowadrH[] = {0, 0, 0};
  int diagindH[] = {0, 0, 0};

  int nnz = mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, diagindH, 2,
                                       3, rownnz, rowadr, colind, rownnzT,
                                       rowadrT, colindT, nullptr, data.get());
  EXPECT_EQ(nnz, 4);
  EXPECT_THAT(rownnzH, ElementsAre(2, 0, 2));
  EXPECT_THAT(rowadrH, ElementsAre(0, 2, 2));

  std::vector<int> colindH(nnz, -1);
  mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, colindH.data(), diagindH, 2, 3,
                             rownnz, rowadr, colind, rownnzT, rowadrT, colindT,
                             nullptr, data.get());
  EXPECT_EQ(diagindH[0], 0);
  EXPECT_EQ(diagindH[1], -1);
  EXPECT_EQ(diagindH[2], 3);

  EXPECT_EQ(mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, diagindH, 0,
                                       3, rownnz, rowadr, colind, rownnzT,
                                       rowadrT, colindT, nullptr, data.get()),
            0);
  EXPECT_THAT(rownnzH, ElementsAre(0, 0, 0));
  EXPECT_THAT(rowadrH, ElementsAre(0, 0, 0));
  EXPECT_THAT(diagindH, ElementsAre(-1, -1, -1));

  // Fill mode with nr == 0, nc > 0 must preserve caller-provided res_rowadr
  int shifted_rowadrH[] = {0, 2, 5};
  rownnzH[0] = rownnzH[1] = rownnzH[2] = 7;
  diagindH[0] = diagindH[1] = diagindH[2] = 7;
  EXPECT_EQ(mju_sqrMatTDSparseSymbolic(rownnzH, shifted_rowadrH, colindH.data(),
                                       diagindH, 0, 3, rownnz, rowadr, colind,
                                       rownnzT, rowadrT, colindT, nullptr,
                                       data.get()),
            0);
  EXPECT_THAT(rownnzH, ElementsAre(0, 0, 0));
  EXPECT_THAT(shifted_rowadrH, ElementsAre(0, 2, 5));
  EXPECT_THAT(diagindH, ElementsAre(-1, -1, -1));

  EXPECT_EQ(mju_sqrMatTDSparseSymbolic(rownnzH, rowadrH, nullptr, diagindH, 2,
                                       0, rownnz, rowadr, colind, rownnzT,
                                       rowadrT, colindT, nullptr, data.get()),
            0);
}

}  // namespace
}  // namespace mujoco
