// Copyright 2024 DeepMind Technologies Limited
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

// Tests for user/user_util.cc

#include "src/user/user_util.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;
using ::testing::IsNan;
using user::FilePath;
using user::StringToVector;
using user::VectorToString;

using UserUtilTest = MujocoTest;

TEST_F(UserUtilTest, PathReduce) {
  FilePath path = FilePath("/hello/.././world/");
  EXPECT_EQ(path.Str(), "/world/");
}

TEST_F(UserUtilTest, PathReduce2) {
  FilePath path = FilePath("../hello/./world/");
  EXPECT_EQ(path.Str(), "../hello/world/");
}

TEST_F(UserUtilTest, PathReduce3) {
  FilePath path = FilePath("../../hello/world.txt");
  EXPECT_EQ(path.Str(), "../../hello/world.txt");
}

TEST_F(UserUtilTest, PathReduceWin) {
  FilePath path = FilePath("C:\\hello\\..\\world");
  EXPECT_EQ(path.Str(), "C:\\world");
}

TEST_F(UserUtilTest, PathReduceRepeatedSeparators) {
  EXPECT_EQ(FilePath("sub//box.obj").Str(), "sub/box.obj");
  EXPECT_EQ(FilePath("sub///box.obj").Str(), "sub/box.obj");
  EXPECT_EQ(FilePath("/hello//world/").Str(), "/hello/world/");
  EXPECT_EQ(FilePath("//hello///world//").Str(), "/hello/world/");
  EXPECT_EQ(FilePath("C:\\\\hello\\\\..\\\\world").Str(), "C:\\world");
}

TEST_F(UserUtilTest, PathReduceProvider) {
  mjpResourceProvider provider = {
      .prefix = "testuserutil",
      .open = +[](mjResource*) { return 0; },
      .read = +[](mjResource*, const void**) { return 0; },
      .close = +[](mjResource*) {},
  };
  mjp_registerProvider(&provider);

  EXPECT_EQ(FilePath("testuserutil:faces/right.png").Str(),
            "testuserutil:faces/right.png");
  EXPECT_EQ(FilePath("testuserutil:/scene.xml").Str(),
            "testuserutil:/scene.xml");
  EXPECT_EQ(FilePath("testuserutil://store/box.obj").Str(),
            "testuserutil://store/box.obj");
  EXPECT_EQ(FilePath("testuserutil:/a//b/../c.xml").Str(),
            "testuserutil:/a/c.xml");
  EXPECT_TRUE(FilePath("testuserutil:/scene.xml").IsAbs());
  EXPECT_TRUE(FilePath("testuserutil:faces/right.png").IsAbs());
}

TEST_F(UserUtilTest, IsAbs) {
  EXPECT_TRUE(FilePath("/hello").IsAbs());
  EXPECT_TRUE(FilePath("C:\\hello").IsAbs());
  EXPECT_FALSE(FilePath("hello").IsAbs());
}

TEST_F(UserUtilTest, Combine) {
  FilePath path1 = FilePath("/hello");
  FilePath path2 = FilePath("world");
  EXPECT_EQ((path1 + path2).Str(), "/hello/world");
}

TEST_F(UserUtilTest, Combine2) {
  FilePath path1 = FilePath("hello/");
  FilePath path2 = FilePath("world");
  EXPECT_EQ((path1 + path2).Str(), "hello/world");
}

TEST_F(UserUtilTest, Combine3) {
  FilePath path1 = FilePath("/hello");
  FilePath path2 = FilePath("../world");
  EXPECT_EQ((path1 + path2).Str(), "/world");
}

TEST_F(UserUtilTest, CombineAbs) {
  FilePath path1 = FilePath("/hello");
  FilePath path2 = FilePath("/world");
  EXPECT_EQ((path1 + path2).Str(), "/world");
}

TEST_F(UserUtilTest, Ext) {
  FilePath path = FilePath("/hello/world.txt");
  EXPECT_EQ(path.Ext(), ".txt");
}

TEST_F(UserUtilTest, ExtEmpty) {
  FilePath path = FilePath("/hello/world");
  EXPECT_EQ(path.Ext(), "");
}

TEST_F(UserUtilTest, StripExt) {
  FilePath path = FilePath("/hello/world.txt");
  EXPECT_EQ(path.StripExt().Str(), "/hello/world");
}

TEST_F(UserUtilTest, StripPath) {
  FilePath path = FilePath("/hello/world.txt");
  EXPECT_EQ(path.StripPath().Str(), "world.txt");
}

TEST_F(UserUtilTest, StripPathEmpty) {
  FilePath path = FilePath("world.txt");
  EXPECT_EQ(path.StripPath().Str(), "world.txt");
}

TEST_F(UserUtilTest, StripPathWin) {
  FilePath path = FilePath("\\world.txt");
  EXPECT_EQ(path.StripPath().Str(), "world.txt");
}

TEST_F(UserUtilTest, StrLower) {
  FilePath path = FilePath("/HELLO/worlD.txt");
  EXPECT_EQ(path.StrLower(), "/hello/world.txt");
}

TEST_F(UserUtilTest, StringToVectorFloat) {
  std::vector<float> v = StringToVector<float>(" 1.2 3.2     5.3 6 ");
  EXPECT_THAT(v, ElementsAre(1.2, 3.2, 5.3, 6));
  EXPECT_EQ(errno, 0);
}

TEST_F(UserUtilTest, StringToVectorEmpty) {
  std::vector<float> v = StringToVector<float>("   ");
  EXPECT_THAT(v, ElementsAre());
  EXPECT_EQ(errno, 0);
}

TEST_F(UserUtilTest, StringToVectorError) {
  std::vector<float> v = StringToVector<float>("2.1 3ABCD. /123/122/113");
  EXPECT_THAT(v, ElementsAre(2.1));
  EXPECT_EQ(errno, EINVAL);
}

TEST_F(UserUtilTest, StringToVectorInt) {
  std::vector<int> v = StringToVector<int>("  -1 3  5 6  ");
  EXPECT_THAT(v, ElementsAre(-1, 3, 5, 6));
  EXPECT_EQ(errno, 0);
}

TEST_F(UserUtilTest, StringToVectorString) {
  auto v = StringToVector<std::string>(" abc  def ");
  EXPECT_THAT(v, ElementsAre("abc", "def"));
}

TEST_F(UserUtilTest, StringToVectorInvalidNumber) {
  auto v = StringToVector<double>("1 0.1.2.3");
  EXPECT_THAT(v, ElementsAre(1));
  EXPECT_EQ(errno, EINVAL);
}

TEST_F(UserUtilTest, StringToVectorNan) {
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings();
  auto v = StringToVector<double>("1 2 nan 3.21");
  EXPECT_THAT(v[2], IsNan());
  EXPECT_EQ(v[3], 3.21);
  EXPECT_EQ(errno, EDOM);
}

TEST_F(UserUtilTest, StringToVectorRange) {
  auto v = StringToVector<unsigned char>("-10");
  EXPECT_EQ(errno, ERANGE);
}

TEST_F(UserUtilTest, StringToVectorSubnormalAndUnderflow) {
  auto vf = StringToVector<float>("1e-40 -1e-45 0 1e-50 2.5");
  ASSERT_EQ(vf.size(), 5);
  EXPECT_GT(vf[0], 0.0f);
  EXPECT_LT(vf[1], 0.0f);
  EXPECT_EQ(vf[2], 0.0f);
  EXPECT_EQ(vf[3], 0.0f);
  EXPECT_EQ(vf[4], 2.5f);
  EXPECT_EQ(errno, 0);

  auto vd = StringToVector<double>("1e-315 -1e-320 1e-400 3.0");
  ASSERT_EQ(vd.size(), 4);
  EXPECT_GT(vd[0], 0.0);
  EXPECT_LT(vd[1], 0.0);
  EXPECT_EQ(vd[2], 0.0);
  EXPECT_EQ(vd[3], 3.0);
  EXPECT_EQ(errno, 0);

  auto vnan = StringToVector<float>("nan 1e-40");
  ASSERT_EQ(vnan.size(), 2);
  EXPECT_THAT(vnan[0], IsNan());
  EXPECT_GT(vnan[1], 0.0f);
  EXPECT_EQ(errno, EDOM);

  auto vovf = StringToVector<float>("1.0 1e40");
  EXPECT_EQ(errno, ERANGE);
}

TEST_F(UserUtilTest, VectorToString) {
  std::vector<double> v = {1.2, 3.2, 5.3, 6};
  EXPECT_EQ(VectorToString(v), "1.2 3.2 5.3 6");
}

TEST_F(UserUtilTest, VectorToStringEmpty) {
  std::vector<double> v;
  EXPECT_EQ(VectorToString(v), "");
}

// utility: modified Gram-Schmidt to orthogonalize columns of Q (n x n)
static void gramSchmidt(double* Q, int n) {
  for (int j = 0; j < n; j++) {
    // subtract projections onto previous columns
    for (int k = 0; k < j; k++) {
      double dot = 0;
      for (int i = 0; i < n; i++) {
        dot += Q[i * n + j] * Q[i * n + k];
      }
      for (int i = 0; i < n; i++) {
        Q[i * n + j] -= dot * Q[i * n + k];
      }
    }
    // normalize
    double norm = 0;
    for (int i = 0; i < n; i++) {
      norm += Q[i * n + j] * Q[i * n + j];
    }
    norm = std::sqrt(norm);
    for (int i = 0; i < n; i++) {
      Q[i * n + j] /= norm;
    }
  }
}

// utility: compose SPD matrix A = Q * diag(eigvals) * Q^T
static void composeMatrix(double* A, const double* Q, const double* eigvals,
                          int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      double sum = 0;
      for (int k = 0; k < n; k++) {
        sum += Q[i * n + k] * eigvals[k] * Q[j * n + k];
      }
      A[i * n + j] = sum;
      A[j * n + i] = sum;
    }
  }
}

TEST_F(UserUtilTest, EigendecomposeConvergence) {
  // seeded RNG for reproducibility
  std::mt19937_64 rng;
  rng.seed(42);
  std::normal_distribution<double> dist(0, 1);

  // sweep over matrix sizes used by flex stiffness
  //   order=1: 8 nodes * 3 dof = 24
  //   order=2: 27 nodes * 3 dof = 81
  for (int n : {24, 81}) {
    int total_sweeps = 0;
    int max_sweeps = 0;
    int count = 0;

    // generate random orthogonal matrix Q via Gram-Schmidt
    std::vector<double> Q(n * n);
    for (int i = 0; i < n * n; i++) {
      Q[i] = dist(rng);
    }
    gramSchmidt(Q.data(), n);

    // sweep eigenvalue spectra of varying difficulty
    //   well-separated, clustered, wide condition number
    for (double condition : {1e1, 1e3, 1e6}) {
      for (double cluster : {0.0, 0.5, 0.9}) {
        // construct eigenvalues
        std::vector<double> eigvals(n);
        for (int i = 0; i < n; i++) {
          // base: logarithmically spaced from 1 to condition
          double t = (double)i / (n - 1);
          double base = std::exp(t * std::log(condition));

          // cluster: push eigenvalues toward geometric mean
          double mean = std::sqrt(condition);
          eigvals[i] = (1 - cluster) * base + cluster * mean;
        }

        // compose A = Q * diag(eigvals) * Q^T
        std::vector<double> A(n * n);
        composeMatrix(A.data(), Q.data(), eigvals.data(), n);

        // save copy for verification
        std::vector<double> A_copy(A);

        // decompose
        std::vector<double> found_eigval(n);
        std::vector<double> found_eigvec(n * n);
        int sweeps = mjuu_eigendecompose(A.data(), found_eigval.data(),
                                         found_eigvec.data(), n);

        total_sweeps += sweeps;
        if (sweeps > max_sweeps) max_sweeps = sweeps;
        count++;

        // verify convergence
        EXPECT_LT(sweeps, 200) << "n=" << n << " condition=" << condition
                               << " cluster=" << cluster;

        // verify A*v = lambda*v for each eigenpair
        for (int i = 0; i < n; i++) {
          for (int r = 0; r < n; r++) {
            double Av = 0;
            for (int c = 0; c < n; c++) {
              Av += A_copy[r * n + c] * found_eigvec[c * n + i];
            }
            double lv = found_eigval[i] * found_eigvec[r * n + i];
            EXPECT_NEAR(Av, lv, 1e-6 * std::abs(found_eigval[i]))
                << "n=" << n << " condition=" << condition
                << " cluster=" << cluster << " eigpair=" << i << " row=" << r;
          }
        }

        // verify all eigenvalues are positive
        for (int i = 0; i < n; i++) {
          EXPECT_GT(found_eigval[i], 0) << "n=" << n << " eigenvalue " << i;
        }
      }
    }

    double mean_sweeps = (double)total_sweeps / count;

    // assert reasonable average convergence
    EXPECT_LE(mean_sweeps, 20.0) << "n=" << n << ": mean sweeps too high";

    // assert max sweeps within budget
    EXPECT_LT(max_sweeps, 200) << "n=" << n << ": max sweeps exceeded 200";
  }
}

// residuals of the decomposition mat = eigvec * diag(eigval) * eigvec':
// reconstruction error relative to the largest element of mat, and deviation of
// eigvec from orthonormality
static void Eig3Residual(double* recon, double* orth, const double mat[9],
                         const double eigval[3], const double eigvec[9]) {
  double scale = 0;
  for (int i = 0; i < 9; i++) scale = std::max(scale, std::abs(mat[i]));

  *recon = *orth = 0;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      double A = 0, I = 0;
      for (int k = 0; k < 3; k++) {
        A += eigvec[3 * i + k] * eigval[k] * eigvec[3 * j + k];
        I += eigvec[3 * k + i] * eigvec[3 * k + j];
      }
      *recon = std::max(*recon, std::abs(A - mat[3 * i + j]) / scale);
      *orth = std::max(*orth, std::abs(I - (i == j)));
    }
  }
}

// the decomposition converges to roundoff, whatever the scale of the matrix;
// mjuu_eig3 is double precision in both builds
TEST_F(UserUtilTest, Eig3ConvergesAtAnyScale) {
  for (double scale : {1e-15, 1.0, 1e15}) {
    double mat[9] = {2, .1, .2, .1, 3, .3, .2, .3, 4};
    for (int i = 0; i < 9; i++) mat[i] *= scale;

    double eigval[3], eigvec[9], quat[4];
    EXPECT_LT(mjuu_eig3(eigval, eigvec, quat, mat), 20);

    // eigenvalues are decreasing
    EXPECT_GT(eigval[0], eigval[1]);
    EXPECT_GT(eigval[1], eigval[2]);

    double recon, orth;
    Eig3Residual(&recon, &orth, mat, eigval, eigvec);
    EXPECT_LE(recon, MjTol(1e-13, 1e-13)) << "scale " << scale;
    EXPECT_LE(orth, MjTol(1e-13, 1e-13)) << "scale " << scale;
  }
}

// repeated and nearly repeated eigenvalues, where the Jacobi rotation is
// ill-defined, converge in a few iterations
TEST_F(UserUtilTest, Eig3RepeatedEigenvalues) {
  std::mt19937_64 rng;
  rng.seed(3);
  std::normal_distribution<double> dist(0, 1);

  const double spectrum[4][3] = {
      {3, 1, 1}, {3, 3, 1}, {1, 1, 1}, {1 + 1e-14, 1, 1 - 1e-14}};

  for (const double* w : spectrum) {
    int max_iter = 0;
    double max_recon = 0;
    for (int n = 0; n < 1000; n++) {
      // random rotation: orthonormalized Gaussian matrix
      double R[9];
      for (int i = 0; i < 9; i++) R[i] = dist(rng);
      gramSchmidt(R, 3);

      // mat = R * diag(w) * R', exactly symmetric
      double mat[9];
      for (int i = 0; i < 3; i++) {
        for (int j = i; j < 3; j++) {
          mat[3 * i + j] = mat[3 * j + i] = R[3 * i] * w[0] * R[3 * j] +
                                            R[3 * i + 1] * w[1] * R[3 * j + 1] +
                                            R[3 * i + 2] * w[2] * R[3 * j + 2];
        }
      }

      double eigval[3], eigvec[9], quat[4], recon, orth;
      max_iter = std::max(max_iter, mjuu_eig3(eigval, eigvec, quat, mat));
      Eig3Residual(&recon, &orth, mat, eigval, eigvec);
      max_recon = std::max(max_recon, recon);
    }
    EXPECT_LT(max_iter, 20)
        << "spectrum " << w[0] << " " << w[1] << " " << w[2];
    EXPECT_LE(max_recon, MjTol(1e-13, 1e-13))
        << "spectrum " << w[0] << " " << w[1] << " " << w[2];
  }
}

// a loose tolerance stops early: off-diagonal elements are below the tolerance
// and eigvec is orthonormal, avoiding spurious rotations from float32 mesh
// noise
TEST_F(UserUtilTest, Eig3Tolerance) {
  const double mat[9] = {2, .1, .2, .1, 3, .3, .2, .3, 4};
  const double reltol = 1e-7;

  double eigval[3], eigvec[9], quat[4];
  int tight = mjuu_eig3(eigval, eigvec, quat, mat);
  int loose = mjuu_eig3(eigval, eigvec, quat, mat, reltol);
  EXPECT_LT(loose, tight);

  double recon, orth;
  Eig3Residual(&recon, &orth, mat, eigval, eigvec);
  EXPECT_LE(recon, reltol * 2);
  EXPECT_LE(orth, MjTol(1e-13, 1e-13));
}

}  // namespace
}  // namespace mujoco
