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

// Tests for engine/engine_core_util.c.

#include "src/engine/engine_core_util.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjdata.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using ::testing::Pointwise;

using FlexGatherStateTest = MujocoTest;

TEST_F(FlexGatherStateTest, mju_flexGatherState_Grid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="flex0" type="grid" count="3 3 3" spacing=".1 .1 .1"
                dim="3" mass="1" radius="0.01" dof="trilinear">
        <elasticity young="5e4" poisson="0.2"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  ASSERT_EQ(model->nflex, 1);
  int f = 0;
  int nodenum = model->flex_nodenum[f];
  int nstart = model->flex_nodeadr[f];

  // Simulate a rotated state (90 degrees around Z axis)
  ASSERT_TRUE(model->flex_centered[f]);
  for (int i = 0; i < nodenum; i++) {
    int b = model->flex_nodebodyid[nstart + i];
    mjtNum x = data->xpos[3 * b + 0];
    mjtNum y = data->xpos[3 * b + 1];
    mjtNum z = data->xpos[3 * b + 2];

    // Rotate 90 degrees around Z: (x, y, z) -> (-y, x, z)
    data->xpos[3 * b + 0] = -y;
    data->xpos[3 * b + 1] = x;
    data->xpos[3 * b + 2] = z;
  }

  std::vector<mjtNum> xpos(3 * nodenum);
  mju_flexGatherState(model.get(), data.get(), f, xpos.data(), NULL);

  // Verify that gathered xpos matches the rotated data->xpos
  for (int i = 0; i < nodenum; i++) {
    int b = model->flex_nodebodyid[nstart + i];
    EXPECT_NEAR(xpos[3 * i + 0], data->xpos[3 * b + 0], 1e-5);
    EXPECT_NEAR(xpos[3 * i + 1], data->xpos[3 * b + 1], 1e-5);
    EXPECT_NEAR(xpos[3 * i + 2], data->xpos[3 * b + 2], 1e-5);
  }
}

TEST_F(FlexGatherStateTest, mju_flexGatherState_ShellMode) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="flex0" type="grid" count="3 3 3" spacing=".1 .1 .1"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 2 2">
        <elasticity young="5e4" poisson="0.2" elastic2d="stretch" thickness="0.02"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  ASSERT_EQ(model->nflex, 1);
  int f = 0;
  model->flex_interp[f] = -1;

  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int nodenum = model->flex_nodenum[f];
  int nstart = model->flex_nodeadr[f];

  // Move boundary nodes, keep interior node stuck (it is pinned)
  mjtNum shift[3] = {0.1, 0.2, 0.3};
  for (int i = 0; i < nodenum; i++) {
    if (i == 13) continue;  // Skip center node
    int b = model->flex_nodebodyid[nstart + i];
    data->xpos[3 * b + 0] += shift[0];
    data->xpos[3 * b + 1] += shift[1];
    data->xpos[3 * b + 2] += shift[2];
  }

  std::vector<mjtNum> xpos(3 * nodenum);
  mju_flexGatherState(model.get(), data.get(), f, xpos.data(), NULL);

  // Verify that gathered xpos for center node (13) is the TFI reconstructed
  // position
  EXPECT_NEAR(xpos[3 * 13 + 0], shift[0], 1e-5);
  EXPECT_NEAR(xpos[3 * 13 + 1], shift[1], 1e-5);
  EXPECT_NEAR(xpos[3 * 13 + 2], shift[2], 1e-5);
}

// mj_flexInterpWeights evaluated at vertex reference coordinates reproduces
// d->flexvert_xpos in a perturbed configuration, and at arbitrary interior
// points in [0, 1]^3 reproduces affine fields (using only boundary nodes in
// shell mode), for trilinear and quadratic interpolation.
TEST_F(FlexGatherStateTest, InterpWeightsMatchVertexPositions) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <flexcomp name="vol1" type="grid" count="4 4 3" spacing=".1 .1 .1"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 2 1">
        <elasticity young="1e4" poisson="0.2"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="vol2" type="grid" count="4 4 3" spacing=".1 .1 .1" pos="1 0 0"
                dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="1 1 1">
        <elasticity young="1e4" poisson="0.2"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="shell1" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="2 0 0"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="3 3 3">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="shell2" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="3 0 0"
                dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="2 2 2">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->nflex, 4);
  EXPECT_EQ(m->flex_interp[0], 1);
  EXPECT_EQ(m->flex_interp[1], 2);
  EXPECT_EQ(m->flex_interp[2], -1);
  EXPECT_EQ(m->flex_interp[3], -2);

  // perturbed configuration
  MjDataPtr d = MakeData(m);
  for (int i = 0; i < m->nq; i++) {
    d->qpos[i] = m->qpos0[i] + 0.02 * (mju_Halton(i + 1, 2) - 0.5);
  }
  mj_forward(m.get(), d.get());

  const mjtNum A[9] = {1.2, 0.3, -0.1, 0.05, 0.9, 0.2, -0.15, 0.1, 1.1};
  const mjtNum b[3] = {0.02, -0.03, 0.04};
  int nsample = 0;
  for (int f = 0; f < m->nflex; f++) {
    int vadr = m->flex_vertadr[f];
    int order = m->flex_interp[f] < 0 ? -m->flex_interp[f] : m->flex_interp[f];
    int nx = m->flex_cellnum[3 * f + 0] * order + 1;
    int ny = m->flex_cellnum[3 * f + 1] * order + 1;
    int nz = m->flex_cellnum[3 * f + 2] * order + 1;

    // at every vertex, mj_flexInterpWeights reproduces d->flexvert_xpos
    for (int v = 0; v < m->flex_vertnum[f]; v++) {
      int gv = vadr + v;
      int node[mjMAXFLEXINTERPW];
      mjtNum weight[mjMAXFLEXINTERPW];
      int nn = mj_flexInterpWeights(m.get(), f, m->flex_vert0 + 3 * gv, node,
                                    weight);
      ASSERT_GT(nn, 0);
      mjtNum recon[3] = {0, 0, 0};
      for (int q = 0; q < nn; q++) {
        mjtNum pos[3];
        mj_flexNodePos(m.get(), d.get(), f, m->flex_nodeadr[f] + node[q], pos);
        mju_addToScl3(recon, pos, weight[q]);
        if (m->flex_interp[f] < 0) {
          int k = node[q] % nz, j = (node[q] / nz) % ny,
              i = node[q] / (ny * nz);
          EXPECT_TRUE(i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ||
                      k == 0 || k == nz - 1)
              << "interior node in shell-mode weights, flex " << f;
        }
      }
      for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(recon[c], d->flexvert_xpos[3 * gv + c], MjTol(1e-10, 1e-5))
            << "flex " << f << " vert " << v;
      }
      nsample++;
    }

    // at interior points in [0, 1]^3, weights use only boundary nodes in shell
    // mode and reproduce affine fields
    for (int s = 0; s < 20; s++) {
      mjtNum coord[3] = {
          mju_Halton(s + 1, 2),
          mju_Halton(s + 1, 3),
          mju_Halton(s + 1, 5),
      };
      int node[mjMAXFLEXINTERPW];
      mjtNum weight[mjMAXFLEXINTERPW];
      int nn = mj_flexInterpWeights(m.get(), f, coord, node, weight);
      ASSERT_GT(nn, 0);
      mjtNum wsum = 0;
      mjtNum recon[3] = {0, 0, 0};
      for (int q = 0; q < nn; q++) {
        wsum += weight[q];
        int k = node[q] % nz, j = (node[q] / nz) % ny, i = node[q] / (ny * nz);
        if (m->flex_interp[f] < 0) {
          EXPECT_TRUE(i == 0 || i == nx - 1 || j == 0 || j == ny - 1 ||
                      k == 0 || k == nz - 1)
              << "interior node in shell-mode weights, flex " << f;
        }
        mjtNum ncoord[3] = {
            (mjtNum)i / (nx - 1),
            (mjtNum)j / (ny - 1),
            (mjtNum)k / (nz - 1),
        };
        mjtNum y[3];
        mju_mulMatVec3(y, A, ncoord);
        mju_addTo3(y, b);
        mju_addToScl3(recon, y, weight[q]);
      }
      EXPECT_NEAR(wsum, 1.0, MjTol(1e-12, 1e-5));
      mjtNum expected[3];
      mju_mulMatVec3(expected, A, coord);
      mju_addTo3(expected, b);
      for (int c = 0; c < 3; c++) {
        EXPECT_NEAR(recon[c], expected[c], MjTol(1e-12, 1e-5))
            << "flex " << f << " sample " << s;
      }
    }
  }
  EXPECT_GT(nsample, 0);
}

// Coons TFI of shell-mode flexes reproduces affine motions of the boundary:
// with the boundary nodes moved by an affine map, the (pinned) interior lattice
// nodes and all vertices follow the same map, for trilinear and quadratic
// interpolation, and the boundary nodes are left unchanged.
TEST_F(FlexGatherStateTest, ShellTFIReproducesAffineMotion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="shell1" type="grid" count="4 4 4" spacing=".1 .1 .1"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="3 3 3">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="shell2" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="1 0 0"
                dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="2 2 2">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());

  // rest positions of nodes and vertices
  std::vector<mjtNum> node0(3 * m->nflexnode), vert0(3 * m->nflexvert);
  for (int f = 0; f < m->nflex; f++) {
    ASSERT_LT(m->flex_interp[f], 0);
    for (int n = 0; n < m->flex_nodenum[f]; n++) {
      int g = m->flex_nodeadr[f] + n;
      mj_flexNodePos(m.get(), d.get(), f, g, node0.data() + 3 * g);
    }
  }
  mju_copy(vert0.data(), d->flexvert_xpos, 3 * m->nflexvert);

  // affine map y = A x + b
  const mjtNum A[9] = {1.2, 0.3, -0.1, 0.05, 0.9, 0.2, -0.15, 0.1, 1.1};
  const mjtNum b[3] = {0.02, -0.03, 0.04};
  auto affine = [&](const mjtNum* x, mjtNum* y) {
    mju_mulMatVec3(y, A, x);
    mju_addTo3(y, b);
  };

  // move the boundary node bodies (the only ones with dofs) by the affine map
  for (int g = 0; g < m->nflexnode; g++) {
    int body = m->flex_nodebodyid[g];
    if (m->body_dofnum[body] != 3) {
      continue;
    }
    mjtNum y[3];
    affine(node0.data() + 3 * g, y);
    for (int k = 0; k < 3; k++) {
      int jnt = m->dof_jntid[m->body_dofadr[body] + k];
      d->qpos[m->jnt_qposadr[jnt]] = y[k] - node0[3 * g + k];
    }
  }
  mj_forward(m.get(), d.get());

  for (int f = 0; f < m->nflex; f++) {
    // nodes: interior reconstructed by TFI, boundary unchanged
    std::vector<mjtNum> xpos(3 * m->flex_nodenum[f]);
    mju_flexGatherState(m.get(), d.get(), f, xpos.data(), nullptr);
    for (int n = 0; n < m->flex_nodenum[f]; n++) {
      int g = m->flex_nodeadr[f] + n;
      mjtNum y[3];
      affine(node0.data() + 3 * g, y);
      for (int k = 0; k < 3; k++) {
        EXPECT_NEAR(xpos[3 * n + k], y[k], MjTol(1e-12, 1e-5))
            << "flex " << f << " node " << n;
      }
    }

    // vertices: Lagrange interpolation of an affine node field is exact
    for (int v = 0; v < m->flex_vertnum[f]; v++) {
      int g = m->flex_vertadr[f] + v;
      mjtNum y[3];
      affine(vert0.data() + 3 * g, y);
      for (int k = 0; k < 3; k++) {
        EXPECT_NEAR(d->flexvert_xpos[3 * g + k], y[k], MjTol(1e-12, 1e-5))
            << "flex " << f << " vertex " << v;
      }
    }
  }
}

using AngMomMatTest = MujocoTest;

static constexpr char AngMomTestingModel[] = R"(
<mujoco>
  <option>
    <flag gravity="disable"/>
  </option>
  <worldbody>
    <body name="link1" pos="0 0 0.5">
      <freejoint/>
      <geom type="ellipsoid" size="0.15 0.17 0.19" quat="1 .2 .3 .4"/>
      <body name="link2" >
        <joint type="hinge" axis="1 0 0" />
        <geom type="capsule" size="0.05" fromto="0 0 0  0 0.5 0"/>
        <body pos="0 0.6 0">
          <joint type="slide" axis="1 0 0"/>
          <geom type="capsule" size="0.05 0.2" quat="0.707 0 0.707 0"/>
          <body name="link3">
            <joint type="ball" pos="0.2 0 0"/>
            <geom type="capsule" pos="0.2 0 0" size="0.03 0.4"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key qvel="0 0 0 .1 .2 .3 .4 .5 .4 .3 .2"/>
  </keyframe>
</mujoco>
)";

// compare subtree angular momentum computed in two ways
TEST_F(AngMomMatTest, CompareAngMom) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(AngMomTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "link1");

  MjDataPtr data = MakeData(model);

  // reset to the keyframe with some angular velocities
  mj_resetDataKeyframe(model.get(), data.get(), 0);
  mj_forward(model.get(), data.get());

  // get the reference value of angular momentum
  mj_subtreeVel(model.get(), data.get());
  mjtNum angmom_ref[3];
  mju_copy3(angmom_ref, data->subtree_angmom + 3 * bodyid);

  // compute angular momentum using the angular momentum matrix
  mjtNum* angmom_mat = (mjtNum*)mju_malloc(sizeof(mjtNum) * 3 * nv);
  mj_angmomMat(model.get(), data.get(), angmom_mat, bodyid);
  mjtNum angmom_test[3];
  mju_mulMatVec(angmom_test, angmom_mat, data->qvel, 3, nv);

  // compare the two angular momentum values
  for (int i = 0; i < 3; i++) {
    EXPECT_THAT(angmom_ref[i], MjNear(angmom_test[i], 1e-8, 1e-4));
  }

  mju_free(angmom_mat);
}

// compare subtree angular momentum matrix: analytical and findiff
TEST_F(AngMomMatTest, CompareAngMomMats) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(AngMomTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "link1");
  MjDataPtr data = MakeData(model);
  mjtNum* angmom_mat = (mjtNum*)mju_malloc(sizeof(mjtNum) * 3 * nv);
  mjtNum* angmom_mat_fd = (mjtNum*)mju_malloc(sizeof(mjtNum) * 3 * nv);

  // reset to the keyframe with some angular velocities
  mj_resetDataKeyframe(model.get(), data.get(), 0);
  mj_forward(model.get(), data.get());

  // compute the angular momentum matrix using the analytical method
  mj_angmomMat(model.get(), data.get(), angmom_mat, bodyid);

  // compute the angular momentum matrix using finite differences
  static const mjtNum eps = MjEps(1e-6, 1e-3);
  for (int i = 0; i < nv; i++) {
    // reset vel, forward nudge i-th dof, get angmom
    mju_copy(data->qvel, model->key_qvel, model->nv);
    data->qvel[i] += eps;
    mj_forward(model.get(), data.get());
    mj_subtreeVel(model.get(), data.get());
    mjtNum agmf[3];
    mju_copy3(agmf, data->subtree_angmom + 3 * bodyid);

    // reset vel, backward nudge i-th dof, get angmom
    mju_copy(data->qvel, model->key_qvel, model->nv);
    data->qvel[i] -= eps;
    mj_forward(model.get(), data.get());
    mj_subtreeVel(model.get(), data.get());
    mjtNum agmb[3];
    mju_copy3(agmb, data->subtree_angmom + 3 * bodyid);

    // finite-difference the angmom matrix
    for (int j = 0; j < 3; j++) {
      angmom_mat_fd[nv * j + i] = (agmf[j] - agmb[j]) / (2 * eps);
    }
  }

  // compare the two matrices
  for (int i = 0; i < 3 * nv; i++) {
    EXPECT_THAT(angmom_mat_fd[i], MjNear(angmom_mat[i], 1e-8, 2e-4));
  }

  mju_free(angmom_mat_fd);
  mju_free(angmom_mat);
}

using JacobianTest = MujocoTest;
static const mjtNum max_abs_err = std::numeric_limits<float>::epsilon();

static constexpr char kJacobianTestingModel[] = R"(
<mujoco>
  <worldbody>
    <body name="distractor1" pos="0 0 .3">
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body name="main">
      <freejoint/>
      <geom size=".1"/>
      <body pos=".1 0 0">
        <joint axis="0 1 0"/>
        <geom type="capsule" size=".03" fromto="0 0 0 .2 0 0"/>
      </body>
      <body pos="0 .1 0">
        <joint type="ball"/>
        <geom type="capsule" size=".03" fromto="0 0 0 0 .2 0"/>
        <body pos="0 .2 0">
          <joint type="slide" axis="1 1 1"/>
          <geom size=".05"/>
        </body>
      </body>
    </body>
    <body name="distractor2" pos="0 0 -.3">
      <freejoint/>
      <geom size=".1"/>
    </body>
  </worldbody>
</mujoco>
)";

// compare analytic and finite-differenced subtree-com Jacobian
TEST_F(JacobianTest, SubtreeJac) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kJacobianTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "main");
  MjDataPtr data = MakeData(model);
  mjtNum* jac_subtree = (mjtNum*)mju_malloc(sizeof(mjtNum) * 3 * nv);
  mjtNum* qpos = (mjtNum*)mju_malloc(sizeof(mjtNum) * model->nq);
  mjtNum* nudge = (mjtNum*)mju_malloc(sizeof(mjtNum) * nv);

  // all we need for Jacobians are kinematics and CoM-related quantities
  mj_kinematics(model.get(), data.get());
  mj_comPos(model.get(), data.get());

  // get subtree CoM Jacobian of free body
  mj_jacSubtreeCom(model.get(), data.get(), jac_subtree, bodyid);

  // save current subtree-com and qpos, clear nudge
  mjtNum subtree_com[3];
  mju_copy3(subtree_com, data->subtree_com + 3 * bodyid);
  mju_copy(qpos, data->qpos, model->nq);
  mju_zero(nudge, nv);

  // compare analytic Jacobian to finite-difference approximation
  static const mjtNum eps = 1e-6;
  for (int i = 0; i < nv; i++) {
    // reset qpos, nudge i-th dof, update data->qpos, reset nudge
    mju_copy(data->qpos, qpos, model->nq);
    nudge[i] = 1;
    mj_integratePos(model.get(), data->qpos, nudge, eps);
    nudge[i] = 0;

    // kinematics and comPos to get nudged com
    mj_kinematics(model.get(), data.get());
    mj_comPos(model.get(), data.get());

    // compare finite-differenced and analytic Jacobian
    for (int j = 0; j < 3; j++) {
      mjtNum findiff =
          (data->subtree_com[3 * bodyid + j] - subtree_com[j]) / eps;
      EXPECT_THAT(jac_subtree[nv * j + i], MjNear(findiff, eps, 1e-2));
    }
  }

  mju_free(nudge);
  mju_free(qpos);
  mju_free(jac_subtree);
}

// confirm that applying linear forces via the subtree-com Jacobian only creates
// the expected linear accelerations (no accelerations of internal joints)
TEST_F(JacobianTest, SubtreeJacNoInternalAcc) {
  char error[1024];
  MjModelPtr model =
      LoadModelFromString(kJacobianTestingModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "main");
  MjDataPtr data = MakeData(model);
  mjtNum* jac_subtree = (mjtNum*)mju_malloc(sizeof(mjtNum) * 3 * nv);

  // all we need for Jacobians are kinematics and CoM-related quantities
  mj_kinematics(model.get(), data.get());
  mj_comPos(model.get(), data.get());

  // get subtree CoM Jacobian of free body
  mj_jacSubtreeCom(model.get(), data.get(), jac_subtree, bodyid);

  // uncomment for debugging
  // mju_printMat(jac_subtree, 3, nv);

  // call fwdPosition since we'll need the factorised mass matrix in the test
  mj_fwdPosition(model.get(), data.get());

  // treating the subtree Jacobian as the projection of 3 axis-aligned unit
  // forces into joint space, solve for the resulting accelerations in-place
  mj_solveM(model.get(), data.get(), jac_subtree, jac_subtree, 3);

  // expect to find accelerations of magnitude 1/subtreemass in the first 3
  // coordinates of the free joint and 0s elsewhere, since applying forces to
  // the CoM should accelerate the whole mechanism without any internal motion
  int body_dofadr = model->body_dofadr[bodyid];
  mjtNum invtreemass = 1.0 / model->body_subtreemass[bodyid];
  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < nv; c++) {
      mjtNum expected = c - body_dofadr == r ? invtreemass : 0.0;
      EXPECT_THAT(jac_subtree[nv * r + c], MjNear(expected, max_abs_err, 1e-4));
    }
  }

  mju_free(jac_subtree);
}

static constexpr char kQuat[] = R"(
<mujoco>
  <worldbody>
    <body name="query">
      <joint type="ball"/>
      <geom size="1"/>
      <site name="query" pos=".1 .2 .3"/>
    </body>
  </worldbody>
  <keyframe>
    <key qvel="2 3 5"/>
  </keyframe>
</mujoco>
)";

static constexpr char kFreeBall[] = R"(
<mujoco>
  <worldbody>
    <body name="distractor1" pos="0 0 .3">
      <freejoint/>
      <geom size=".1"/>
    </body>
    <body name="main">
      <freejoint/>
      <geom size=".1"/>
      <body pos=".1 0 0">
        <joint axis="0 1 0"/>
        <geom type="capsule" size=".03" fromto="0 0 0 .2 0 0"/>
        <body pos=".2 0 0">
          <joint type="ball" stiffness="20"/>
          <geom type="capsule" size=".03" fromto="0 0 0 0 .2 0"/>
          <body name="query" pos="0 .2 0">
            <joint type="slide" axis="1 1 1"/>
            <geom size=".05"/>
            <site name="query" pos=".1 .2 .3"/>
          </body>
        </body>
      </body>
    </body>
    <body name="distractor2" pos="0 0 -.3">
      <freejoint/>
      <geom size=".1"/>
    </body>
  </worldbody>
  <keyframe>
    <key qvel="1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"/>
  </keyframe>
</mujoco>
)";

static constexpr char kQuatlessPendulum[] = R"(
<mujoco>
  <option integrator="implicit">
    <flag constraint="disable"/>
  </option>
  <worldbody>
    <body pos="0.15 0 0">
      <joint type="hinge" axis="0 1 0"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
      <body pos="0.1 0 0">
        <joint type="slide" axis="1 0 0" stiffness="200"/>
        <geom type="capsule" size="0.015" fromto="-.1 0 0 .1 0 0"/>
        <body pos=".1 0 0">
          <joint axis="1 0 0"/>
          <joint axis="0 1 0"/>
          <joint axis="0 0 1"/>
          <geom type="box" size=".02" fromto="0 0 0 0 .1 0"/>
          <body name="query" pos="0 .1 0">
            <joint axis="1 0 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 0 .1 0"/>
            <site name="query" pos=".1 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr char kTelescope[] = R"(
<mujoco>
  <worldbody>
    <body>
      <joint type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 .02 0 .1 .02 0"/>
      <body pos=".1 .02 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
        <body pos=".1 .02 0">
          <joint type="slide" axis="1 0 0"/>
          <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
          <body pos=".1 .02 0" name="query">
            <joint type="slide" axis="1 0 0"/>
            <geom type="capsule" size="0.02" fromto="0 0 0 .1 0 0"/>
            <site name="query" pos=".1 0 0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key qvel="1 1 1 1 1 1"/>
  </keyframe>
</mujoco>
)";

static constexpr char kHinge[] = R"(
<mujoco>
  <worldbody>
    <body name="query">
      <joint name="link1" axis="0 1 0"/>
      <geom type="capsule" size=".02" fromto="0 0 0 0 0 -1"/>
      <site name="query" pos="0 0 -1"/>
    </body>
  </worldbody>

  <keyframe>
    <key qpos="1" qvel="1"/>
  </keyframe>
</mujoco>
)";

// compare mj_jacDot with finite-differenced mj_jac
TEST_F(JacobianTest, JacDot) {
  for (auto xml : {kHinge, kQuat, kTelescope, kFreeBall, kQuatlessPendulum}) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    int nv = model->nv;
    MjDataPtr data = MakeData(model);

    // load keyframe if present, step for a bit
    if (model->nkey) mj_resetDataKeyframe(model.get(), data.get(), 0);
    while (data->time < 0.1) {
      mj_step(model.get(), data.get());
    }

    // minimal call required for mj_jacDot outputs to be valid
    mj_kinematics(model.get(), data.get());
    mj_comPos(model.get(), data.get());
    mj_comVel(model.get(), data.get());

    // get bodyid
    int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "query");
    EXPECT_GT(bodyid, 0);

    // get site position
    int siteid = mj_name2id(model.get(), mjOBJ_SITE, "query");
    EXPECT_GT(siteid, -1);
    mjtNum point[3];
    mju_copy3(point, data->site_xpos + 3 * siteid);

    // jac, jac_dot
    std::vector<mjtNum> jacp(3 * nv);
    std::vector<mjtNum> jacr(3 * nv);
    mj_jac(model.get(), data.get(), jacp.data(), jacr.data(), point, bodyid);
    std::vector<mjtNum> jacp_dot(3 * nv);
    std::vector<mjtNum> jacr_dot(3 * nv);
    mj_jacDot(model.get(), data.get(), jacp_dot.data(), jacr_dot.data(), point,
              bodyid);

    // jac_h: jacobian after integrating qpos with a timestep of h
    const mjtNum h = MjTol(1e-7, 5e-4);
    mj_integratePos(model.get(), data->qpos, data->qvel, h);
    mj_kinematics(model.get(), data.get());
    mj_comPos(model.get(), data.get());
    std::vector<mjtNum> jacp_h(3 * nv);
    std::vector<mjtNum> jacr_h(3 * nv);
    mju_copy3(point,
              data->site_xpos + 3 * siteid);  // get updated site position
    mj_jac(model.get(), data.get(), jacp_h.data(), jacr_h.data(), point,
           bodyid);

    // jac_dot_h finite-difference approximation
    std::vector<mjtNum> jacp_dot_h(3 * nv);
    mju_sub(jacp_dot_h.data(), jacp_h.data(), jacp.data(), 3 * nv);
    mju_scl(jacp_dot_h.data(), jacp_dot_h.data(), 1 / h, 3 * nv);
    std::vector<mjtNum> jacr_dot_h(3 * nv);
    mju_sub(jacr_dot_h.data(), jacr_h.data(), jacr.data(), 3 * nv);
    mju_scl(jacr_dot_h.data(), jacr_dot_h.data(), 1 / h, 3 * nv);

    // compare finite-differenced and analytic
    mjtNum tol = 1e-5;
    EXPECT_THAT(jacp_dot, Pointwise(MjNear(tol, 5e-2), jacp_dot_h));
    EXPECT_THAT(jacr_dot, Pointwise(MjNear(tol, 5e-2), jacr_dot_h));
  }
}

// compare mj_jacDotSparse with dense mj_jacDot
TEST_F(JacobianTest, JacDotSparse) {
  for (auto xml : {kHinge, kQuat, kTelescope, kFreeBall, kQuatlessPendulum}) {
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    int nv = model->nv;
    MjDataPtr data = MakeData(model);

    // load keyframe if present, step for a bit
    if (model->nkey) mj_resetDataKeyframe(model.get(), data.get(), 0);
    while (data->time < 0.1) {
      mj_step(model.get(), data.get());
    }

    // minimal call required for mj_jacDot outputs to be valid
    mj_kinematics(model.get(), data.get());
    mj_comPos(model.get(), data.get());
    mj_comVel(model.get(), data.get());

    // get bodyid and site position
    int bodyid = mj_name2id(model.get(), mjOBJ_BODY, "query");
    EXPECT_GT(bodyid, 0);
    int siteid = mj_name2id(model.get(), mjOBJ_SITE, "query");
    EXPECT_GT(siteid, -1);
    mjtNum point[3];
    mju_copy3(point, data->site_xpos + 3 * siteid);

    // dense jacDot
    std::vector<mjtNum> jacp_dense(3 * nv);
    std::vector<mjtNum> jacr_dense(3 * nv);
    mj_jacDot(model.get(), data.get(), jacp_dense.data(), jacr_dense.data(),
              point, bodyid);

    // compute body chain using public mjModel fields
    std::vector<int> chain(nv);
    int NV = 0;
    int weldbody = model->body_weldid[bodyid];
    if (weldbody) {
      int da = model->body_dofadr[weldbody] + model->body_dofnum[weldbody] - 1;
      while (da >= 0) {
        chain[NV++] = da;
        da = model->dof_parentid[da];
      }
      std::reverse(chain.begin(), chain.begin() + NV);
    }
    EXPECT_GT(NV, 0);

    // sparse jacDot
    std::vector<mjtNum> jacp_sparse(3 * NV);
    std::vector<mjtNum> jacr_sparse(3 * NV);
    mj_jacDotSparse(model.get(), data.get(), jacp_sparse.data(),
                    jacr_sparse.data(), point, bodyid, NV, chain.data());

    // expand sparse to dense and compare
    std::vector<mjtNum> jacp_expanded(3 * nv, 0);
    std::vector<mjtNum> jacr_expanded(3 * nv, 0);
    for (int ci = 0; ci < NV; ci++) {
      int di = chain[ci];
      for (int r = 0; r < 3; r++) {
        jacp_expanded[di + r * nv] = jacp_sparse[ci + r * NV];
        jacr_expanded[di + r * nv] = jacr_sparse[ci + r * NV];
      }
    }

    // expect bitwise equality
    EXPECT_EQ(jacp_expanded, jacp_dense);
    EXPECT_EQ(jacr_expanded, jacr_dense);
  }
}

// validate rotational Jacobian used in welds
TEST_F(JacobianTest, WeldRotJacobian) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "FD Jacobian with eps=1e-6 below float32 precision";
#endif
  constexpr char xml[] = R"(
  <mujoco>
    <option jacobian="dense"/>
    <worldbody>
      <body>
        <joint type="ball"/>
        <geom size=".1"/>
      </body>
      <body pos=".5 0 0">
        <joint axis="1 0 0" pos="0 0 .01"/>
        <joint axis="0 1 0" pos=".02 0 0"/>
        <joint axis="0 0 1" pos="0 .03 0"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), testing::NotNull()) << error;
  ASSERT_EQ(model->nq, 7);
  ASSERT_EQ(model->nv, 6);
  static const int nv = 6;  // for increased readability
  MjDataPtr data = MakeData(model);

  // arbitrary initial values for the ball and hinge joints
  mjtNum qpos0[7] = {.5, .5, .5, .5, .7, .8, .9};

  // compute required quantities using mj_step1
  mj_step1(model.get(), data.get());

  // get orientation error
  mjtNum res[3];
  // compute rotation residual following formula in mj_instantiateEquality
  auto RotationResidual = [](const mjModel* model, mjData* data,
                             const mjtNum qpos[7], const mjtNum dqpos[6],
                             mjtNum res[3]) {
    // copy configuration, compute required quantities with mj_step1
    mju_copy(data->qpos, qpos, 7);

    // perturb configuration if given
    if (dqpos) {
      mj_integratePos(model, data->qpos, dqpos, 1);
    }

    // update relevant quantities
    mj_step1(model, data);

    // compute orientation residual
    mjtNum quat1[4], quat2[4], quat3[4];
    mju_copy4(quat1, data->xquat + 4 * 1);
    mju_negQuat(quat2, data->xquat + 4 * 2);
    mju_mulQuat(quat3, quat2, quat1);
    mju_copy3(res, quat3 + 1);
  };

  RotationResidual(model.get(), data.get(), qpos0, NULL, res);

  // compute Jacobian with finite-differencing
  mjtNum jacFD[3 * nv];
  mjtNum dqpos[nv] = {0};
  mjtNum dres[3];
  const mjtNum eps = 1e-6;
  for (int i = 0; i < nv; i++) {
    // nudge i-th dof
    dqpos[i] = eps;

    // get nudged residual
    RotationResidual(model.get(), data.get(), qpos0, dqpos, dres);

    // remove nudge
    dqpos[i] = 0.0;

    // compute Jacobian column
    for (int j = 0; j < 3; j++) {
      jacFD[nv * j + i] = (dres[j] - res[j]) / eps;
    }
  }

  // reset mjData to qpos0
  mju_copy(data->qpos, qpos0, 7);
  mj_step1(model.get(), data.get());

  // intermediate quaternions quat1 and quat2
  mjtNum quat1[4], negQuat2[4];
  mju_copy4(quat1, data->xquat + 4 * 1);
  mju_negQuat(negQuat2, data->xquat + 4 * 2);

  // get analytical Jacobian following formula in mj_instantiateEquality
  mjtNum jacdif[3 * nv], jac0[3 * nv], jac1[3 * nv];
  mjtNum point[3] = {0};

  // rotational Jacobian difference
  mj_jacDifPair(model.get(), data.get(), NULL, 2, 1, point, point, NULL, NULL,
                NULL, jac0, jac1, jacdif, mj_isSparse(model.get()),
                /*flg_skipcommon=*/0);

  // formula: 0.5 * neg(quat2) * (jac1-jac2) * quat1
  mjtNum axis[3], quat3[4], quat4[4];
  for (int j = 0; j < nv; j++) {
    // axis = [jac1-jac2]_col(j)
    axis[0] = jacdif[0 * nv + j];
    axis[1] = jacdif[1 * nv + j];
    axis[2] = jacdif[2 * nv + j];

    // apply formula
    mju_mulQuatAxis(quat3, negQuat2, axis);
    mju_mulQuat(quat4, quat3, quat1);

    // correct Jacobian
    jacdif[0 * nv + j] = 0.5 * quat4[1];
    jacdif[1 * nv + j] = 0.5 * quat4[2];
    jacdif[2 * nv + j] = 0.5 * quat4[3];
  }

  // test that analytical and finite-differenced Jacobians match
  EXPECT_THAT(AsVector(jacFD, 3 * nv),
              Pointwise(MjNear(eps, 1e-3), AsVector(jacdif, 3 * nv)));
}

// Jacobian difference of two points on one simple body
TEST_F(JacobianTest, JacDifPairSameSimpleBody) {
  // flex edge 0 joins two points of body A, edge 1 joins A to the world
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="A" pos="0 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="tri" dim="1" radius=".01" body="A A world" vertex="0 0 0  .2 0 0  .5 0 1" element="0 1 1 2">
        <edge stiffness="10"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  int nv = model->nv;
  int body = mj_name2id(model.get(), mjOBJ_BODY, "A");
  ASSERT_TRUE(model->body_simple[body]);
  MjDataPtr data = MakeData(model);

  for (int i = 0; i < 100; i++) {
    mj_step(model.get(), data.get());
  }

  // edge 0 has each dof once, zero Jacobian since its length is constant
  int rowadr = model->flexedge_J_rowadr[0];
  ASSERT_EQ(model->flexedge_J_rownnz[0], nv);
  for (int i = 0; i < nv; i++) {
    EXPECT_EQ(model->flexedge_J_colind[rowadr + i], i);
    EXPECT_THAT(data->flexedge_J[rowadr + i], MjNear(0, 1e-15, 1e-6));
  }

  // sparse Jacobian difference of the vertices matches dense, is nonzero
  const mjtNum* pos = data->flexvert_xpos;
  std::vector<int> chain(nv);
  std::vector<mjtNum> jac1(3 * nv), jac2(3 * nv);
  std::vector<mjtNum> jacdif(3 * nv), jacdif_dense(3 * nv);
  ASSERT_EQ(mj_jacDifPair(model.get(), data.get(), chain.data(), body, body,
                          pos, pos + 3, jac1.data(), jac2.data(), jacdif.data(),
                          nullptr, nullptr, nullptr, /*issparse=*/1,
                          /*flg_skipcommon=*/0),
            nv);
  mj_jacDifPair(model.get(), data.get(), nullptr, body, body, pos, pos + 3,
                jac1.data(), jac2.data(), jacdif_dense.data(), nullptr, nullptr,
                nullptr, /*issparse=*/0, /*flg_skipcommon=*/0);
  EXPECT_EQ(jacdif, jacdif_dense);
  EXPECT_GT(mju_norm(jacdif_dense.data(), 3 * nv), 0);

  // skipping common dofs leaves none
  EXPECT_EQ(mj_jacDifPair(model.get(), data.get(), chain.data(), body, body,
                          pos, pos + 3, nullptr, nullptr, nullptr, nullptr,
                          nullptr, nullptr, /*issparse=*/1,
                          /*flg_skipcommon=*/1),
            0);
}

TEST_F(JacobianTest, QuadraticFlexContactPreservesNegativeBasisWeights) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="parent">
        <flexcomp name="quad" type="grid" count="5 5 2" spacing=".1 .1 .1"
                  dim="3" mass="1" radius="0.02" dof="quadratic" cellcount="1 1 1">
          <elasticity young="1e4" poisson="0.2"/>
          <contact condim="1"/>
        </flexcomp>
      </body>
      <body name="probe_body">
        <joint name="probe_z" type="slide" axis="0 0 1"/>
        <geom name="probe" type="sphere" size="0.02"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Find interior vertex (1, 1, 0) on the bottom face, where local coordinates
  // are (0.25, 0.25, 0) and quadratic basis functions along x and y evaluate to
  // {0.375, 0.75, -0.125}.
  int v_target = -1;
  for (int v = 0; v < m->nflexvert; v++) {
    const mjtNum* c = m->flex_vert0 + 3 * v;
    if (std::abs(c[0] - 0.25) < 1e-6 && std::abs(c[1] - 0.25) < 1e-6 &&
        std::abs(c[2] - 0.0) < 1e-6) {
      v_target = v;
      break;
    }
  }
  ASSERT_GE(v_target, 0);

  // Place probe sphere directly beneath v_target so the contact point is at
  // (0.25, 0.25, 0) with normal +z (mj_forward computes flexvert_xpos).
  mj_forward(m.get(), d.get());
  int probe = mj_name2id(m.get(), mjOBJ_GEOM, "probe");
  ASSERT_GE(probe, 0);
  // (move the body: the geom shares its frame, so geom_pos is not used)
  int probe_body = m->geom_bodyid[probe];
  m->body_pos[3 * probe_body + 0] = d->flexvert_xpos[3 * v_target + 0];
  m->body_pos[3 * probe_body + 1] = d->flexvert_xpos[3 * v_target + 1];
  m->body_pos[3 * probe_body + 2] = d->flexvert_xpos[3 * v_target + 2] - 0.03;

  // In the 3x3x3 quadratic node grid (order=2, cellcount=1x1x1), node (2, 1, 0)
  // at index 2*9 + 1*3 + 0 = 21 has basis weight (-0.125) * (0.75) * (1.0) =
  // -0.09375 at local coordinate (0.25, 0.25, 0).
  ASSERT_EQ(m->flex_nodenum[0], 27);
  int b_neg = m->flex_nodebodyid[m->flex_nodeadr[0] + 21];
  int node_neg_z_dof = m->body_dofadr[b_neg] + 2;
  ASSERT_GE(node_neg_z_dof, 0);
  int probe_dof = m->jnt_dofadr[mj_name2id(m.get(), mjOBJ_JOINT, "probe_z")];

  for (mjtJacobian sparsity : {mjJAC_DENSE, mjJAC_SPARSE}) {
    m->opt.jacobian = sparsity;
    mj_forward(m.get(), d.get());
    ASSERT_GT(d->ncon, 0);
    ASSERT_GT(d->nefc, 0);

    // Extract the efc_J row of the contact closest to v_target.
    int ic = 0;
    for (int i = 1; i < d->ncon; i++) {
      if (mju_dist3(d->contact[i].pos, d->flexvert_xpos + 3 * v_target) <
          mju_dist3(d->contact[ic].pos, d->flexvert_xpos + 3 * v_target)) {
        ic = i;
      }
    }
    ASSERT_LT(mju_dist3(d->contact[ic].pos, d->flexvert_xpos + 3 * v_target),
              0.03);
    int r = d->contact[ic].efc_address;
    ASSERT_GE(r, 0);
    std::vector<mjtNum> row(m->nv, 0);
    if (mj_isSparse(m.get())) {
      int adr = d->efc_J_rowadr[r];
      int nnz = d->efc_J_rownnz[r];
      for (int k = 0; k < nnz; k++) {
        row[d->efc_J_colind[adr + k]] = d->efc_J[adr + k];
      }
    } else {
      mju_copy(row.data(), d->efc_J + r * m->nv, m->nv);
    }

    // By partition of unity, moving all flex z-DOFs and the probe z-DOF by +1
    // produces zero relative normal velocity, and node (2, 1, 0) has a negative
    // basis weight (so its Jacobian entry has the same sign as the probe's).
    mjtNum sum_z = row[probe_dof];
    for (int n = 0; n < m->flex_nodenum[0]; n++) {
      int b = m->flex_nodebodyid[m->flex_nodeadr[0] + n];
      sum_z += row[m->body_dofadr[b] + 2];
    }
    EXPECT_THAT(sum_z, MjNear(0.0, 1e-6, 1e-5));
    EXPECT_GT(row[node_neg_z_dof] / row[probe_dof], 1e-2);
  }
}

TEST_F(FlexGatherStateTest, GatherScatterCellInterpolated) {
  static constexpr char xml_simple[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f1" type="grid" count="4 4 3" spacing=".1 .1 .1"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 2 1">
        <elasticity young="1e4" poisson="0.2"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="f2" type="grid" count="4 4 3" spacing=".1 .1 .1" pos="1 0 0"
                dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="1 1 1">
        <elasticity young="1e4" poisson="0.2"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="f3" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="2 0 0"
                dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 2 2">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
      <flexcomp name="f4" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="3 0 0"
                dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="1 1 1">
        <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
        <contact selfcollide="none"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_parent_pinned[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent" pos="0.2 -0.1 0.3" euler="15 25 -10">
        <freejoint/>
        <inertial mass="0.5" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
        <flexcomp name="f1" type="grid" count="3 3 3" spacing=".1 .1 .1"
                  dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 1 1">
          <pin id="0"/>
          <elasticity young="1e4" poisson="0.2"/>
          <contact selfcollide="none"/>
        </flexcomp>
        <flexcomp name="f2" type="grid" count="3 3 3" spacing=".1 .1 .1" pos="0.5 0 0"
                  dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="1 1 1">
          <pin id="0"/>
          <elasticity young="1e4" poisson="0.2"/>
          <contact selfcollide="none"/>
        </flexcomp>
        <flexcomp name="f3" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="1.0 0 0"
                  dim="3" mass="1" radius="0.01" dof="trilinear" cellcount="2 2 2">
          <pin id="0"/>
          <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
          <contact selfcollide="none"/>
        </flexcomp>
        <flexcomp name="f4" type="grid" count="4 4 4" spacing=".1 .1 .1" pos="1.5 0 0"
                  dim="3" mass="1" radius="0.01" dof="quadratic" cellcount="1 1 1">
          <pin id="0"/>
          <elasticity young="1e4" poisson="0.2" elastic2d="both" thickness="0.01"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml : {xml_simple, xml_parent_pinned}) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    int nv = m->nv;
    std::vector<mjtNum> u(nv), f_vert, v_gather, qfrc_scatter(nv);
    for (int i = 0; i < nv; i++) {
      u[i] = 0.1 * ((i % 7) - 3);
    }

    static const mjtNum eps = MjEps(1e-6, 1e-3);
    std::vector<mjtNum> qpos0(m->nq, 0);
    mju_copy(qpos0.data(), d->qpos, m->nq);

    for (int f = 0; f < m->nflex; f++) {
      ASSERT_NE(m->flex_interp[f], 0);
      int nvert = m->flex_vertnum[f];
      int vadr = m->flex_vertadr[f];
      v_gather.assign(3 * nvert, 0);
      mj_flexGather(m.get(), d.get(), f, v_gather.data(), u.data());

      // Compare gathered vertex velocity against central finite differences of
      // flexvert_xpos under qpos perturbation along u.
      mju_copy(d->qpos, qpos0.data(), m->nq);
      mj_integratePos(m.get(), d->qpos, u.data(), eps);
      mj_kinematics(m.get(), d.get());
      mj_flex(m.get(), d.get());
      std::vector<mjtNum> xpos_plus(d->flexvert_xpos + 3 * vadr,
                                    d->flexvert_xpos + 3 * (vadr + nvert));

      mju_copy(d->qpos, qpos0.data(), m->nq);
      mj_integratePos(m.get(), d->qpos, u.data(), -eps);
      mj_kinematics(m.get(), d.get());
      mj_flex(m.get(), d.get());
      std::vector<mjtNum> xpos_minus(d->flexvert_xpos + 3 * vadr,
                                     d->flexvert_xpos + 3 * (vadr + nvert));

      mju_copy(d->qpos, qpos0.data(), m->nq);
      mj_forward(m.get(), d.get());

      for (int k = 0; k < 3 * nvert; k++) {
        mjtNum fd = (xpos_plus[k] - xpos_minus[k]) / (2 * eps);
        EXPECT_THAT(v_gather[k], MjNear(fd, 1e-6, 2e-3));
      }

      // Check exact transpose duality <f_vert, J u> == <J^T f_vert, u>.
      f_vert.resize(3 * nvert);
      for (int k = 0; k < 3 * nvert; k++) {
        f_vert[k] = 0.2 * ((k % 11) - 5);
      }
      std::fill(qfrc_scatter.begin(), qfrc_scatter.end(), 0);
      mjtNum scale = 1.5;
      mj_flexScatter(m.get(), d.get(), f, qfrc_scatter.data(), f_vert.data(),
                     scale);

      mjtNum dot_vert =
          scale * mju_dot(f_vert.data(), v_gather.data(), 3 * nvert);
      mjtNum dot_dof = mju_dot(qfrc_scatter.data(), u.data(), nv);
      EXPECT_THAT(dot_dof, MjNear(dot_vert, 1e-10, 1e-4));
    }
  }

  // Verify that quadratic shell mode (flex_interp == -2, f4 in xml_simple) uses
  // 27-node quadratic cell interpolation after the TFI of the interior nodes,
  // rather than piecewise-trilinear sub-cell interpolation: a quadratic
  // displacement u_x(X) = X^2 on boundary nodes is reproduced exactly at face
  // vertices X = 1/3 and 2/3.
  {
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml_simple, error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    mj_forward(m.get(), d.get());

    int f = 3;
    ASSERT_EQ(m->flex_interp[f], -2);
    int nadr = m->flex_nodeadr[f];
    int nodenum = m->flex_nodenum[f];
    ASSERT_EQ(nodenum, 27);

    std::vector<mjtNum> u(m->nv, 0);
    for (int i = 0; i < nodenum; i++) {
      int b = m->flex_nodebodyid[nadr + i];
      if (b > 0 && m->body_dofnum[b] == 3) {
        mjtNum X = 0.5 * (i / 9);
        u[m->body_dofadr[b] + 0] = X * X;
      }
    }

    int vadr = m->flex_vertadr[f];
    int nvert = m->flex_vertnum[f];
    std::vector<mjtNum> v_gather(3 * nvert, 0);
    mj_flexGather(m.get(), d.get(), f, v_gather.data(), u.data());

    for (int v = 0; v < nvert; v++) {
      const mjtNum* c = m->flex_vert0 + 3 * (vadr + v);
      // On boundary faces z == 0 or z == 1, u_x depends only on X^2 and is
      // independent of the interior node, so 27-node quadratic interpolation
      // reproduces X^2 exactly at X = 1/3 and 2/3.
      if (std::abs(c[2]) < 1e-6 || std::abs(c[2] - 1.0) < 1e-6) {
        EXPECT_THAT(v_gather[3 * v + 0], MjNear(c[0] * c[0], 1e-12, 1e-5));
      }
    }
  }
}

}  // namespace
}  // namespace mujoco
