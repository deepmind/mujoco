// Copyright 2023 DeepMind Technologies Limited
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

// Tests for engine/engine_island.c.

#include "src/engine/engine_island.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_util_sparse.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::NotNull;
using ::testing::Pointwise;
using IslandTest = MujocoTest;

TEST_F(IslandTest, DsuRootReturnsCanonicalRootAndCompressesPath) {
  int parent[] = {0, 0, 1, 2, 3};

  EXPECT_EQ(mj_dsuRoot(parent, 0), 0);
  EXPECT_EQ(mj_dsuRoot(parent, 4), 0);
  EXPECT_THAT(parent, ElementsAre(0, 0, 0, 0, 0));
}

TEST_F(IslandTest, DsuMergeActivatesEndpointsAndUsesMinimumRoot) {
  int parent[] = {-1, -1, -1, -1, -1, -1};

  mj_dsuMerge(parent, -1, 4);
  mj_dsuMerge(parent, 3, -1);
  mj_dsuMerge(parent, 5, 2);
  mj_dsuMerge(parent, 4, 5);
  mj_dsuMerge(parent, 3, 4);

  EXPECT_THAT(parent, ElementsAre(-1, -1, 2, 2, 2, 2));
  for (int tree = 2; tree < 6; ++tree) {
    EXPECT_EQ(mj_dsuRoot(parent, tree), 2);
  }
  EXPECT_THAT(parent, ElementsAre(-1, -1, 2, 2, 2, 2));
}

TEST_F(IslandTest, DsuMergeRedundantAndReversedEdgesAreIdempotent) {
  int parent[] = {-1, -1, -1, -1};
  mj_dsuMerge(parent, 3, 1);
  mj_dsuMerge(parent, 2, 1);
  EXPECT_THAT(parent, ElementsAre(-1, 1, 1, 1));

  mj_dsuMerge(parent, 1, 3);
  mj_dsuMerge(parent, 3, 1);
  mj_dsuMerge(parent, 2, 2);
  mj_dsuMerge(parent, -1, 2);
  EXPECT_THAT(parent, ElementsAre(-1, 1, 1, 1));
}

TEST_F(IslandTest, DsuMergeFastPathActivatesBeforeTestingParents) {
  int self_parent[] = {-1, -1, -1};
  mj_dsuMerge(self_parent, 1, 1);
  EXPECT_THAT(self_parent, ElementsAre(-1, 1, -1));

  int static_first[] = {-1, -1, -1};
  mj_dsuMerge(static_first, -1, 2);
  EXPECT_THAT(static_first, ElementsAre(-1, -1, 2));

  int static_second[] = {-1, -1, -1};
  mj_dsuMerge(static_second, 0, -1);
  EXPECT_THAT(static_second, ElementsAre(0, -1, -1));
}

TEST_F(IslandTest, DsuMergeFastPathDistinguishesParentsFromRoots) {
  int distinct_parent[] = {0, 0, 2, 2};
  mj_dsuMerge(distinct_parent, 1, 3);
  EXPECT_THAT(distinct_parent, ElementsAre(0, 0, 0, 2));

  int shared_parent[] = {0, 0, 0, 3};
  mj_dsuMerge(shared_parent, 1, 2);
  EXPECT_THAT(shared_parent, ElementsAre(0, 0, 0, 3));

  int long_paths[] = {0, 0, 1, 3, 3, 4};
  mj_dsuMerge(long_paths, 2, 5);
  EXPECT_THAT(long_paths, ElementsAre(0, 0, 0, 0, 3, 3));
}

TEST_F(IslandTest, DsuMergeFastPathPreservesCyclesDuplicatesAndForest) {
  int parent[] = {-1, -1, -1, -1, -1, -1};
  mj_dsuMerge(parent, 0, 1);
  mj_dsuMerge(parent, 1, 2);
  mj_dsuMerge(parent, 2, 0);
  mj_dsuMerge(parent, 0, 2);
  mj_dsuMerge(parent, 3, 4);
  mj_dsuMerge(parent, 4, 5);
  EXPECT_THAT(parent, ElementsAre(0, 0, 0, 3, 3, 3));

  mj_dsuMerge(parent, 5, 0);
  EXPECT_THAT(parent, ElementsAre(0, 0, 0, 0, 3, 3));
}

TEST_F(IslandTest, DsuMergeRejectsStaticSelfIncidence) {
  int parent[] = {-1, 1, 1, 3};

  EXPECT_EQ(MjuErrorMessageFrom(mj_dsuMerge)(parent, -1, -1),
            "self-incidence of the static tree");
  EXPECT_THAT(parent, ElementsAre(-1, 1, 1, 3));
}

TEST_F(IslandTest, DsuAssignHandlesEmptyAndInactiveInputs) {
  int island[] = {71};
  int parent[] = {72};
  const int tree_dofnum[] = {73};
  int nidof = -1;

  EXPECT_EQ(mj_dsuAssign(island, parent, tree_dofnum, 0, &nidof), 0);
  EXPECT_EQ(nidof, 0);
  EXPECT_THAT(island, ElementsAre(71));
  EXPECT_THAT(parent, ElementsAre(72));

  parent[0] = -1;
  EXPECT_EQ(mj_dsuAssign(island, parent, tree_dofnum, 1, &nidof), 0);
  EXPECT_EQ(nidof, 0);
  EXPECT_THAT(island, ElementsAre(-1));
  EXPECT_THAT(parent, ElementsAre(-1));
}

TEST_F(IslandTest, DsuAssignLabelsComponentsAndCountsOnlyActiveDofs) {
  int parent[] = {-1, 1, 1, 2, 4, 4, 6};
  const int tree_dofnum[] = {1000, 0, 3, 5, 7, 11, 13};
  int island[] = {9, 9, 9, 9, 9, 9, 9};
  int nidof = -1;

  EXPECT_EQ(mj_dsuAssign(island, parent, tree_dofnum, 7, &nidof), 3);
  EXPECT_EQ(nidof, 39);
  EXPECT_THAT(island, ElementsAre(-1, 0, 0, 0, 1, 1, 2));
  EXPECT_THAT(parent, ElementsAre(-1, 1, 1, 1, 4, 4, 6));
}

TEST_F(IslandTest, DsuAssignCompressesAscendingMultiHopForest) {
  int parent[] = {-1, 1, 1, 2, 4, 4, 5, 7, 7, 8};
  const int tree_dofnum[] = {99, 0, 2, 3, 0, 5, 7, 11, 0, 13};
  int island[] = {9, 9, 9, 9, 9, 9, 9, 9, 9, 9};
  int nidof = -1;

  EXPECT_EQ(mj_dsuAssign(island, parent, tree_dofnum, 10, &nidof), 3);
  EXPECT_EQ(nidof, 41);
  EXPECT_THAT(island, ElementsAre(-1, 0, 0, 0, 1, 1, 1, 2, 2, 2));
  EXPECT_THAT(parent, ElementsAre(-1, 1, 1, 1, 4, 4, 4, 7, 7, 7));
}

TEST_F(IslandTest, DsuAssignCompresses4096NodeAdversarialChain) {
  constexpr int kTreeCount = 4096;
  std::vector<int> parent(kTreeCount);
  std::vector<int> island(kTreeCount, -2);
  std::vector<int> tree_dofnum(kTreeCount);
  parent[0] = 0;
  int expected_nidof = 0;
  for (int tree = 1; tree < kTreeCount; ++tree) {
    parent[tree] = tree - 1;
    tree_dofnum[tree] = tree % 5;
    expected_nidof += tree_dofnum[tree];
  }

  int nidof = -1;
  EXPECT_EQ(mj_dsuAssign(island.data(), parent.data(), tree_dofnum.data(),
                         kTreeCount, &nidof),
            1);
  EXPECT_EQ(nidof, expected_nidof);
  for (int tree = 0; tree < kTreeCount; ++tree) {
    EXPECT_EQ(island[tree], 0);
    EXPECT_EQ(parent[tree], 0);
  }
}

TEST_F(IslandTest, DsuHandlesLongConnectedBoundaryCase) {
  constexpr int kTreeCount = 4096;
  std::vector<int> parent(kTreeCount, -1);
  std::vector<int> island(kTreeCount, -2);
  std::vector<int> tree_dofnum(kTreeCount);

  int expected_nidof = 0;
  for (int tree = kTreeCount - 1; tree > 0; --tree) {
    mj_dsuMerge(parent.data(), tree, tree - 1);
  }
  for (int tree = 0; tree < kTreeCount; ++tree) {
    tree_dofnum[tree] = tree % 7;
    expected_nidof += tree_dofnum[tree];
  }

  int nidof = -1;
  EXPECT_EQ(mj_dsuAssign(island.data(), parent.data(), tree_dofnum.data(),
                         kTreeCount, &nidof),
            1);
  EXPECT_EQ(nidof, expected_nidof);
  for (int tree = 0; tree < kTreeCount; ++tree) {
    EXPECT_EQ(island[tree], 0);
    EXPECT_EQ(parent[tree], 0);
  }
}

TEST_F(IslandTest, DsuRandomizedDifferentialAgainstGraphTraversal) {
  constexpr uint32_t kSeed = 0x5eed3396u;
  constexpr int kTrials = 2000;
  uint32_t state = kSeed;
  auto next = [&state]() {
    state = state * 1664525u + 1013904223u;
    return state;
  };

  for (int trial = 0; trial < kTrials; ++trial) {
    const int ntree = 1 + next() % 64;
    const int nedge = next() % 192;
    std::vector<std::array<int, 2>> edges;
    edges.reserve(nedge);
    for (int edge = 0; edge < nedge; ++edge) {
      int tree1;
      int tree2;
      switch (next() % 8) {
        case 0:
          tree1 = -1;
          tree2 = next() % ntree;
          break;
        case 1:
          tree1 = next() % ntree;
          tree2 = -1;
          break;
        case 2:
          tree1 = next() % ntree;
          tree2 = tree1;
          break;
        case 3:
          if (!edges.empty()) {
            const auto& previous = edges[next() % edges.size()];
            tree1 = previous[0];
            tree2 = previous[1];
            break;
          }
          [[fallthrough]];
        case 4:
          if (!edges.empty()) {
            const auto& previous = edges[next() % edges.size()];
            tree1 = previous[1];
            tree2 = previous[0];
            break;
          }
          [[fallthrough]];
        default:
          tree1 = next() % ntree;
          tree2 = next() % ntree;
          break;
      }
      edges.push_back({tree1, tree2});
    }

    std::vector<int> parent(ntree, -1);
    for (const auto& edge : edges) {
      mj_dsuMerge(parent.data(), edge[0], edge[1]);
    }

    std::vector<int> active(ntree);
    std::vector<std::vector<int>> adjacent(ntree);
    for (const auto& edge : edges) {
      if (edge[0] >= 0) active[edge[0]] = 1;
      if (edge[1] >= 0) active[edge[1]] = 1;
      if (edge[0] >= 0 && edge[1] >= 0) {
        adjacent[edge[0]].push_back(edge[1]);
        adjacent[edge[1]].push_back(edge[0]);
      }
    }

    std::vector<int> expected_island(ntree, -1);
    std::vector<int> expected_parent(ntree, -1);
    int expected_nisland = 0;
    for (int start = 0; start < ntree; ++start) {
      if (!active[start] || expected_island[start] != -1) continue;
      std::vector<int> pending = {start};
      std::vector<int> component;
      expected_island[start] = expected_nisland;
      while (!pending.empty()) {
        const int tree = pending.back();
        pending.pop_back();
        component.push_back(tree);
        for (int neighbor : adjacent[tree]) {
          if (expected_island[neighbor] == -1) {
            expected_island[neighbor] = expected_nisland;
            pending.push_back(neighbor);
          }
        }
      }
      for (int tree : component) expected_parent[tree] = start;
      ++expected_nisland;
    }

    std::vector<int> tree_dofnum(ntree);
    int expected_nidof = 0;
    for (int tree = 0; tree < ntree; ++tree) {
      tree_dofnum[tree] = next() % 8;
      if (active[tree]) expected_nidof += tree_dofnum[tree];
    }
    std::vector<int> island(ntree, -2);
    int nidof = -1;
    const int nisland = mj_dsuAssign(island.data(), parent.data(),
                                     tree_dofnum.data(), ntree, &nidof);

    SCOPED_TRACE(::testing::Message()
                 << "seed=" << kSeed << " trial=" << trial << " ntree=" << ntree
                 << " nedge=" << nedge);
    EXPECT_EQ(nisland, expected_nisland);
    EXPECT_EQ(nidof, expected_nidof);
    EXPECT_EQ(island, expected_island);
    EXPECT_EQ(parent, expected_parent);
  }
}

TEST_F(IslandTest, FloodFillSingleton) {
  // adjacency matrix for the graph  0   1   2
  //                                 U       U
  // (3 singletons, 0 and 2 have self-edges)
  mjtNum mat[9] = {1, 0, 0, 0, 0, 0, 0, 0, 1};
  constexpr int nr = 3;
  constexpr int nnz = 2;
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum res[nnz];  // unused
  mju_dense2sparse(res, mat, nr, nr, rownnz, rowadr, colind, nnz);

  // outputs / scratch
  int island[nr];
  int scratch[2 * nr];

  // flood fill
  int nisland = mj_floodFill(island, nr, rownnz, rowadr, colind, scratch);

  EXPECT_EQ(nisland, 2);
  EXPECT_THAT(island, ElementsAre(0, -1, 1));
}

TEST_F(IslandTest, FloodFill1) {
  // adjacency matrix for the graph  0 - 1 - 2
  mjtNum mat[9] = {0, 1, 0, 1, 0, 1, 0, 1, 0};
  constexpr int nr = 3;
  constexpr int nnz = 4;
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum res[nnz];  // unused
  mju_dense2sparse(res, mat, nr, nr, rownnz, rowadr, colind, nnz);

  // outputs / stack
  int island[nr];
  int stack[nnz];

  int nisland = mj_floodFill(island, nr, rownnz, rowadr, colind, stack);

  EXPECT_EQ(nisland, 1);
  EXPECT_THAT(island, ElementsAre(0, 0, 0));
}

TEST_F(IslandTest, FloodFill2) {
  //  adjacency matrix for the graph   6 – 1 – 4   0 – 3 – 5 – 2
  mjtNum mat[49] = {
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
      0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
  };
  constexpr int nr = 7;
  constexpr int nnz = 10;
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum res[nnz];  // unused
  mju_dense2sparse(res, mat, nr, nr, rownnz, rowadr, colind, nnz);

  // outputs / stack
  int island[nr];
  int stack[nnz];

  int nisland = mj_floodFill(island, nr, rownnz, rowadr, colind, stack);

  EXPECT_EQ(nisland, 2);
  EXPECT_THAT(island, ElementsAre(0, 1, 0, 0, 1, 0, 1));
}

TEST_F(IslandTest, FloodFill3a) {
  //  adjacency matrix for the graph   0   2   1 – 3
  //                                       U
  mjtNum mat[16] = {
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,
  };
  constexpr int nr = 4;
  constexpr int nnz = 3;
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum res[nnz];  // unused
  mju_dense2sparse(res, mat, nr, nr, rownnz, rowadr, colind, nnz);

  // outputs / stack
  int island[nr];
  int stack[nnz];

  int nisland = mj_floodFill(island, nr, rownnz, rowadr, colind, stack);

  EXPECT_EQ(nisland, 2);
  EXPECT_THAT(island, ElementsAre(-1, 0, 1, 0));
}

TEST_F(IslandTest, FloodFill3b) {
  /*
    adjacency matrix for the graph   1 – 2   3   4 – 5
                                     U           | \ |
                                                 0 – 6
  */
  mjtNum mat[49] = {
      0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0,
  };
  constexpr int nr = 7;
  constexpr int nnz = 13;
  int rownnz[nr];
  int rowadr[nr];
  int colind[nnz];
  mjtNum res[nnz];  // unused
  mju_dense2sparse(res, mat, nr, nr, rownnz, rowadr, colind, nnz);

  // outputs / stack
  int island[nr];
  int stack[nnz];

  int nisland = mj_floodFill(island, nr, rownnz, rowadr, colind, stack);

  EXPECT_EQ(nisland, 2);
  EXPECT_THAT(island, ElementsAre(0, 1, 1, -1, 0, 0, 0));
}

TEST_F(IslandTest, ProductionStaticFirstAndRepeatedRows) {
  static constexpr char xml[] = R"(
<mujoco>
  <option jacobian="sparse"><flag contact="disable" gravity="disable"/></option>
  <worldbody>
    <site name="world"/>
    <body name="b0">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint type="slide"/><site name="s0"/>
    </body>
    <body name="b1">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint type="slide"/><site name="s1"/>
    </body>
    <body name="b2">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint type="slide"/><site name="s2"/>
    </body>
  </worldbody>
  <equality>
    <connect site1="world" site2="s0"/>
    <connect site1="s1" site2="s2"/>
  </equality>
</mujoco>
)";
  char error[1024] = {};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->ntree, 3);
  ASSERT_EQ(model->nv, 3);

  // The first equality incidence is static first, then dynamic tree 0.
  ASSERT_EQ(model->eq_objtype[0], mjOBJ_SITE);
  int body1 = model->site_bodyid[model->eq_obj1id[0]];
  int body2 = model->site_bodyid[model->eq_obj2id[0]];
  EXPECT_EQ(model->body_treeid[body1], -1);
  EXPECT_EQ(model->body_treeid[body2], 0);

  MjDataPtr data = MakeData(model);
  mj_fwdPosition(model.get(), data.get());

  ASSERT_EQ(data->nefc, 6);
  EXPECT_EQ(data->nisland, 2);
  EXPECT_EQ(data->nidof, 3);
  EXPECT_EQ(data->ne, 6);
  EXPECT_EQ(data->nf, 0);
  EXPECT_THAT(
      AsVector(data->efc_type, data->nefc),
      ElementsAre(mjCNSTR_EQUALITY, mjCNSTR_EQUALITY, mjCNSTR_EQUALITY,
                  mjCNSTR_EQUALITY, mjCNSTR_EQUALITY, mjCNSTR_EQUALITY));
  EXPECT_THAT(AsVector(data->efc_id, data->nefc),
              ElementsAre(0, 0, 0, 1, 1, 1));
  EXPECT_THAT(AsVector(data->tree_island, model->ntree), ElementsAre(0, 1, 1));
  EXPECT_THAT(AsVector(data->island_ntree, data->nisland), ElementsAre(1, 2));
  EXPECT_THAT(AsVector(data->island_itreeadr, data->nisland),
              ElementsAre(0, 1));
  EXPECT_THAT(AsVector(data->map_itree2tree, model->ntree),
              ElementsAre(0, 1, 2));
  EXPECT_THAT(AsVector(data->dof_island, model->nv), ElementsAre(0, 1, 1));
  EXPECT_THAT(AsVector(data->island_nv, data->nisland), ElementsAre(1, 2));
  EXPECT_THAT(AsVector(data->island_idofadr, data->nisland), ElementsAre(0, 1));
  EXPECT_THAT(AsVector(data->island_dofadr, data->nisland), ElementsAre(0, 1));
  EXPECT_THAT(AsVector(data->map_dof2idof, model->nv), ElementsAre(0, 1, 2));
  EXPECT_THAT(AsVector(data->map_idof2dof, model->nv), ElementsAre(0, 1, 2));
  EXPECT_THAT(AsVector(data->efc_island, data->nefc),
              ElementsAre(0, 0, 0, 1, 1, 1));
  EXPECT_THAT(AsVector(data->island_ne, data->nisland), ElementsAre(3, 3));
  EXPECT_THAT(AsVector(data->island_nf, data->nisland), ElementsAre(0, 0));
  EXPECT_THAT(AsVector(data->island_nefc, data->nisland), ElementsAre(3, 3));
  EXPECT_THAT(AsVector(data->island_iefcadr, data->nisland), ElementsAre(0, 3));
  EXPECT_THAT(AsVector(data->map_efc2iefc, data->nefc),
              ElementsAre(0, 1, 2, 3, 4, 5));
  EXPECT_THAT(AsVector(data->map_iefc2efc, data->nefc),
              ElementsAre(0, 1, 2, 3, 4, 5));
}

TEST_F(IslandTest, ReportsConstraintBetweenTwoStaticBodies) {
  static constexpr char xml[] = R"(
<mujoco>
  <option jacobian="sparse"><flag contact="disable" gravity="disable"/></option>
  <worldbody>
    <body>
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint type="slide" frictionloss="1"/>
    </body>
  </worldbody>
</mujoco>
)";
  char error[1024] = {};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_fwdPosition(model.get(), data.get());
  ASSERT_GT(data->nefc, 0);
  ASSERT_EQ(data->efc_type[0], mjCNSTR_FRICTION_DOF);

  model->dof_treeid[data->efc_id[0]] = -1;

  EXPECT_EQ(MjuErrorMessageFrom(mj_island)(model.get(), data.get()),
            "constraint 0 is between two static bodies");
}

TEST_F(IslandTest, ProductionFlexEqualityRescansRows) {
  static constexpr char xml[] = R"(
<mujoco>
  <option jacobian="sparse"><flag contact="disable" gravity="disable"/></option>
  <worldbody>
    <flexcomp name="f" type="grid" dim="1" count="3 1 1"
              spacing=".05 .05 .05" radius=".01" mass="1">
      <edge equality="true"/>
      <contact selfcollide="none"/>
    </flexcomp>
  </worldbody>
</mujoco>
)";
  char error[1024] = {};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->ntree, 3);
  ASSERT_EQ(model->nv, 9);
  ASSERT_EQ(model->neq, 1);
  ASSERT_EQ(model->eq_type[0], mjEQ_FLEX);
  ASSERT_TRUE(mj_isSparse(model.get()));

  MjDataPtr data = MakeData(model);
  mj_fwdPosition(model.get(), data.get());

  ASSERT_EQ(data->nefc, 2);
  auto row_trees = [&](int row) {
    std::vector<int> trees;
    for (int j = 0; j < data->efc_J_rownnz[row]; j++) {
      int dof = data->efc_J_colind[data->efc_J_rowadr[row] + j];
      int tree = model->dof_treeid[dof];
      if (trees.empty() || trees.back() != tree) {
        trees.push_back(tree);
      }
    }
    return trees;
  };

  // Rows share one flex equality id but have different tree incidence.
  EXPECT_THAT(row_trees(0), ElementsAre(0, 1));
  EXPECT_THAT(row_trees(1), ElementsAre(1, 2));
  EXPECT_THAT(AsVector(data->efc_type, data->nefc),
              ElementsAre(mjCNSTR_EQUALITY, mjCNSTR_EQUALITY));
  EXPECT_THAT(AsVector(data->efc_id, data->nefc), ElementsAre(0, 0));
  EXPECT_EQ(data->nisland, 1);
  EXPECT_EQ(data->nidof, 9);
  EXPECT_EQ(data->ne, 2);
  EXPECT_EQ(data->nf, 0);
  EXPECT_THAT(AsVector(data->tree_island, model->ntree), ElementsAre(0, 0, 0));
  EXPECT_THAT(AsVector(data->island_ntree, data->nisland), ElementsAre(3));
  EXPECT_THAT(AsVector(data->island_itreeadr, data->nisland), ElementsAre(0));
  EXPECT_THAT(AsVector(data->map_itree2tree, model->ntree),
              ElementsAre(0, 1, 2));
  EXPECT_THAT(AsVector(data->dof_island, model->nv),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_THAT(AsVector(data->island_nv, data->nisland), ElementsAre(9));
  EXPECT_THAT(AsVector(data->island_idofadr, data->nisland), ElementsAre(0));
  EXPECT_THAT(AsVector(data->island_dofadr, data->nisland), ElementsAre(0));
  EXPECT_THAT(AsVector(data->map_dof2idof, model->nv),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(AsVector(data->map_idof2dof, model->nv),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8));
  EXPECT_THAT(AsVector(data->efc_island, data->nefc), ElementsAre(0, 0));
  EXPECT_THAT(AsVector(data->island_ne, data->nisland), ElementsAre(2));
  EXPECT_THAT(AsVector(data->island_nf, data->nisland), ElementsAre(0));
  EXPECT_THAT(AsVector(data->island_nefc, data->nisland), ElementsAre(2));
  EXPECT_THAT(AsVector(data->island_iefcadr, data->nisland), ElementsAre(0));
  EXPECT_THAT(AsVector(data->map_efc2iefc, data->nefc), ElementsAre(0, 1));
  EXPECT_THAT(AsVector(data->map_iefc2efc, data->nefc), ElementsAre(0, 1));
}

TEST_F(IslandTest, BoundedArenaSupports1024Trees) {
  constexpr int kTreeCount = 1024;
  constexpr size_t kArenaBytes = 2 * 1024 * 1024;
  std::string xml = R"(
<mujoco>
  <size memory="2M"/>
  <option jacobian="sparse">
    <flag contact="disable"/>
  </option>
  <worldbody>
)";
  xml.reserve(160 * kTreeCount);
  for (int i = 0; i < kTreeCount; i++) {
    std::string name = std::to_string(i);
    xml += "<body name=\"b" + name + "\">";
    xml += "<inertial pos=\"0 0 0\" mass=\"1\" diaginertia=\"1 1 1\"/>";
    xml += "<joint name=\"j" + name + "\" type=\"slide\"/>";
    if (i < 2) {
      xml += "<site name=\"s" + name + "\"/>";
    }
    xml += "</body>";
  }
  xml += R"(
  </worldbody>
  <equality>
    <connect site1="s0" site2="s1"/>
  </equality>
</mujoco>
)";

  char error[1024] = {};
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->ntree, kTreeCount);
  ASSERT_EQ(model->narena, kArenaBytes);
  MjDataPtr data = MakeData(model);
  ASSERT_THAT(data.get(), NotNull());

  mj_fwdPosition(model.get(), data.get());

  ASSERT_EQ(data->nefc, 3);
  ASSERT_EQ(data->nisland, 1);
  ASSERT_THAT(data->tree_island, NotNull());
  EXPECT_EQ(data->tree_island[0], 0);
  EXPECT_EQ(data->tree_island[1], 0);
  for (int tree = 2; tree < kTreeCount; tree++) {
    EXPECT_EQ(data->tree_island[tree], -1);
  }
  EXPECT_LE(data->maxuse_arena, kArenaBytes);
}

static const char* const kAbacusPath = "engine/testdata/island/abacus.xml";

TEST_F(IslandTest, Abacus) {
  const std::string xml_path = GetTestDataFilePath(kAbacusPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  // disable gravity
  model->opt.disableflags |= mjDSBL_GRAVITY;

  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  // no islands at qpos0
  EXPECT_EQ(data->nisland, 0);

  // push bead 0 to the left and bead 2 to the right until there are 3 contacts
  data->qfrc_applied[0] = -1;
  data->qfrc_applied[2] = 1;
  while (data->ncon != 3) {
    mj_step(model, data);
  }

  // sizes
  int nv = model->nv;
  int nefc = data->nefc;
  int nisland = data->nisland;
  int nidof = data->nidof;

  // 4 dofs, 12 constraints, 2 islands
  EXPECT_EQ(nv, 4);
  EXPECT_EQ(nidof, 3);
  EXPECT_EQ(nefc, 12);  // 3 pyramidal contacts
  EXPECT_EQ(nisland, 2);

  // the islands begin at dofs 0 and 1
  EXPECT_THAT(AsVector(data->island_idofadr, nisland), ElementsAre(0, 1));

  // number of dofs in the 2 islands
  EXPECT_THAT(AsVector(data->island_nv, nisland), ElementsAre(1, 2));

  // dof 0 in    island 0
  // dof 1 in no island
  // dofs 2,3 in island 1
  EXPECT_THAT(AsVector(data->dof_island, nv), ElementsAre(0, -1, 1, 1));

  // dof 0 constitutes first island
  // dofs 2, 3 are the second island
  // last index is unassigned since dof 1 is unconstrained
  EXPECT_THAT(AsVector(data->map_idof2dof, nv), ElementsAre(0, 2, 3, 1));

  // dof 0 constitutes first island
  // dofs 1 is unassigned
  // dofs 2, 3 are second island
  EXPECT_THAT(AsVector(data->map_dof2idof, nv), ElementsAre(0, 3, 1, 2));

  // island 0 starts at constraint 0
  // island 1 starts at constraint 4
  EXPECT_THAT(AsVector(data->island_iefcadr, nisland), ElementsAre(0, 4));

  // number of constraints in the 2 islands
  EXPECT_THAT(AsVector(data->island_nefc, nisland), ElementsAre(4, 8));

  // first contact (4 constraints) is in island 0
  // second contact (8 constraints) is in island 1
  EXPECT_THAT(AsVector(data->efc_island, nefc),
              ElementsAre(0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1));

  // index lists for islands 0 and 1
  EXPECT_THAT(AsVector(data->map_iefc2efc, nefc),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));

  // reset, push 0 to the left, 3 to the right, 1,2 to the middle
  mj_resetData(model, data);
  data->qfrc_applied[0] = -1;
  data->qfrc_applied[1] = 1;
  data->qfrc_applied[2] = -1;
  data->qfrc_applied[3] = 1;

  // simulate until there are 3 contacts
  while (data->ncon != 3) {
    mj_step(model, data);
  }

  // local variables
  nefc = data->nefc;
  nisland = data->nisland;
  nidof = data->nidof;

  EXPECT_EQ(nisland, 3);
  EXPECT_EQ(nidof, 4);
  EXPECT_THAT(AsVector(data->island_idofadr, nisland), ElementsAre(0, 1, 3));
  EXPECT_THAT(AsVector(data->island_nv, nisland), ElementsAre(1, 2, 1));
  EXPECT_THAT(AsVector(data->dof_island, nv), ElementsAre(0, 1, 1, 2));
  EXPECT_THAT(AsVector(data->map_idof2dof, nv), ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(AsVector(data->map_dof2idof, nv), ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(AsVector(data->island_iefcadr, nisland), ElementsAre(0, 4, 8));
  EXPECT_THAT(AsVector(data->island_nefc, nisland), ElementsAre(4, 4, 4));
  EXPECT_THAT(AsVector(data->efc_island, nefc),
              ElementsAre(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2));
  EXPECT_THAT(AsVector(data->map_iefc2efc, nefc),
              ElementsAre(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11));

  mj_deleteData(data);
  mj_deleteModel(model);
}

static const char* const kTendonWrapPath =
    "engine/testdata/island/tendon_wrap.xml";

TEST_F(IslandTest, DenseSparse) {
  const std::string xml_path = GetTestDataFilePath(kTendonWrapPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data1 = mj_makeData(model);
  mjData* data2 = mj_makeData(model);

  // dense
  model->opt.jacobian = mjJAC_DENSE;
  while (!data1->nefc) {
    mj_step(model, data1);
  }

  // sparse
  model->opt.jacobian = mjJAC_SPARSE;
  while (!data2->nefc) {
    mj_step(model, data2);
  }

  // sizes
  int nv = model->nv;
  int nefc = data1->nefc;
  int nisland = data1->nisland;

  // expect sparse and dense to be identical
  EXPECT_EQ(data1->nidof, data2->nidof);
  EXPECT_EQ(data1->nefc, data2->nefc);
  EXPECT_EQ(data1->nisland, data2->nisland);
  EXPECT_EQ(data1->nefc, data2->nefc);
  EXPECT_EQ(AsVector(data1->island_idofadr, nisland),
            AsVector(data2->island_idofadr, nisland));
  EXPECT_EQ(AsVector(data1->island_nv, nisland),
            AsVector(data2->island_nv, nisland));
  EXPECT_EQ(AsVector(data1->dof_island, nv), AsVector(data2->dof_island, nv));
  EXPECT_EQ(AsVector(data1->map_idof2dof, nv),
            AsVector(data2->map_idof2dof, nv));
  EXPECT_EQ(AsVector(data1->map_dof2idof, nv),
            AsVector(data2->map_dof2idof, nv));
  EXPECT_EQ(AsVector(data1->island_iefcadr, nisland),
            AsVector(data2->island_iefcadr, nisland));
  EXPECT_EQ(AsVector(data1->island_nefc, nisland),
            AsVector(data2->island_nefc, nisland));
  EXPECT_EQ(AsVector(data1->efc_island, nefc),
            AsVector(data2->efc_island, nefc));
  EXPECT_EQ(AsVector(data1->map_iefc2efc, nefc),
            AsVector(data2->map_iefc2efc, nefc));
  EXPECT_EQ(AsVector(data1->map_efc2iefc, nefc),
            AsVector(data2->map_efc2iefc, nefc));

  mj_deleteData(data2);
  mj_deleteData(data1);
  mj_deleteModel(model);
}

static const char* const kIlslandEfcPath =
    "engine/testdata/island/island_efc.xml";

TEST_F(IslandTest, IslandEfc) {
  const std::string xml_path = GetTestDataFilePath(kIlslandEfcPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  while (data->time < 0.2) {
    mj_step(model, data);
  }

  // expect island structure to correspond to comment at top of xml
  EXPECT_EQ(data->nisland, 4);
  EXPECT_EQ(data->ne, 7);
  EXPECT_EQ(data->nf, 2);
  EXPECT_EQ(data->nl, 1);
  EXPECT_EQ(data->nefc, 30);
  EXPECT_THAT(AsVector(data->island_ne, data->nisland),
              ElementsAre(1, 0, 0, 6));
  EXPECT_THAT(AsVector(data->island_nf, data->nisland),
              ElementsAre(0, 1, 1, 0));
  EXPECT_THAT(AsVector(data->island_nefc, data->nisland),
              ElementsAre(6, 17, 1, 6));
  EXPECT_THAT(AsVector(data->efc_island, data->nefc),
              ElementsAre(0, 3, 3, 3, 3, 3, 3, 1, 2, 0, 0, 0, 0, 0, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1));

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(IslandTest, IslandFlex) {
  const std::string xml_path = GetTestDataFilePath("testdata/flex.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data1 = mj_makeData(model);
  mjData* data2 = mj_makeData(model);

  model->opt.disableflags &= ~mjDSBL_ISLAND;
  while (data1->time < 0.045) {
    mj_step(model, data1);
  }

  model->opt.disableflags |= mjDSBL_ISLAND;
  while (data2->time < 0.045) {
    mj_step(model, data2);
  }

  EXPECT_THAT(AsVector(data1->qpos, model->nq),
              Pointwise(MjNear(1e-6, 1e-5), AsVector(data2->qpos, model->nq)));

  mj_deleteData(data2);
  mj_deleteData(data1);
  mj_deleteModel(model);
}

// stiffness couples all vertices of a flex: one contact anywhere on the flex
// must pull every vertex tree (and the contacting body) into a single island
TEST_F(IslandTest, FlexStiffnessUnionsTrees) {
  static const char xml[] = R"(
  <mujoco>
    <option solver="Newton"/>
    <worldbody>
      <flexcomp name="cloth" type="grid" count="4 4 1" spacing="0.1 0.1 0.1"
                radius=".005" dim="2" mass="0.5" pos="0 0 1" dof="full">
        <contact selfcollide="none"/>
        <elasticity young="1e3" poisson="0.2" damping="0.1" elastic2d="both" thickness="0.01"/>
      </flexcomp>
      <body pos="0.1 0.1 0.96">
        <freejoint/>
        <geom type="sphere" size="0.05"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // the sphere penetrates the cloth at one corner
  ASSERT_GT(data->ncon, 0);

  // one island containing every dof: 16 vertices and the free sphere
  EXPECT_EQ(data->nisland, 1);
  EXPECT_EQ(data->nidof, model->nv);
}

// rigid flex hinged about its middle: vertices on the axis touch the plane
static constexpr char kRigidFlap[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="flap">
      <joint name="hinge" axis="0 1 0"/>
      <geom name="mass" size=".01" pos=".3 0 .1"/>
      <flexcomp name="flap" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" rigid="true"/>
    </body>
  </worldbody>
</mujoco>
)";

// rigid flex sliding along x on a box: the normal row is zero, a tangent is not
static constexpr char kRigidSlider[] = R"(
<mujoco>
  <option cone="elliptic"/>
  <worldbody>
    <geom name="box" type="box" size=".05 .05 .05" pos="0 0 -.05"/>
    <body name="slider">
      <joint type="slide" axis="1 0 0"/>
      <geom name="mass" size=".02" pos="0 0 .2"/>
      <flexcomp name="f1" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" rigid="true"
                radius=".005"/>
    </body>
  </worldbody>
</mujoco>
)";

// cloth pinned to a hinged body: pins on the hinge axis touch the plane
static constexpr char kPinnedCloth[] = R"(
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 .1"/>
    <body name="flap">
      <joint name="hinge" axis="0 1 0"/>
      <geom name="mass" size=".01" pos=".3 0 .1"/>
      <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" mass=".1">
        <pin id="3 4 5"/>
        <edge equality="true"/>
      </flexcomp>
    </body>
  </worldbody>
</mujoco>
)";

// triangle with vertices in three trees, hinges orthogonal to the box: element
// contacts span the three trees and their normal row is zero
static constexpr char kHingedTriangle[] = R"(
<mujoco>
  <option cone="elliptic"/>
  <worldbody>
    <geom name="box" type="box" size=".2 .2 .05" pos="0 0 -.05"/>
    <body name="b0" pos=".1 0 0">
      <joint/>
      <geom size=".01" pos=".02 0 .1"/>
    </body>
    <body name="b1" pos="-.05 .1 0">
      <joint/>
      <geom size=".01" pos=".02 0 .1"/>
    </body>
    <body name="b2" pos="-.05 -.1 0">
      <joint/>
      <geom size=".01" pos=".02 0 .1"/>
    </body>
  </worldbody>
  <deformable>
    <flex name="triangle" dim="2" body="b0 b1 b2" vertex="0 0 0 0 0 0 0 0 0" element="0 1 2"
          radius=".005">
      <edge damping="1"/>
    </flex>
  </deformable>
</mujoco>
)";

// a flex contact is in the trees of its bodies even if its first row is zero:
// dense and sparse Jacobians give the same contacts, in one island
TEST_F(IslandTest, FlexContactWithZeroRow) {
  for (auto xml : {kRigidFlap, kRigidSlider, kPinnedCloth, kHingedTriangle}) {
    SCOPED_TRACE(xml);
    char error[1024];
    MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
    ASSERT_THAT(model.get(), NotNull()) << error;
    int nv = model->nv;

    // dense: some contact has a zero first row
    model->opt.jacobian = mjJAC_DENSE;
    MjDataPtr dense = MakeData(model);
    mj_forward(model.get(), dense.get());
    int nzero = 0;
    for (int i = 0; i < dense->ncon; i++) {
      int adr = dense->contact[i].efc_address;
      nzero += adr >= 0 && mju_isZero(dense->efc_J + adr * nv, nv);
    }
    EXPECT_GT(nzero, 0);

    // sparse: same contacts
    model->opt.jacobian = mjJAC_SPARSE;
    MjDataPtr sparse = MakeData(model);
    mj_forward(model.get(), sparse.get());
    ASSERT_EQ(dense->ncon, sparse->ncon);
    for (int i = 0; i < dense->ncon; i++) {
      const mjContact& con1 = dense->contact[i];
      const mjContact& con2 = sparse->contact[i];
      EXPECT_THAT(con1.geom, ElementsAre(con2.geom[0], con2.geom[1]));
      EXPECT_THAT(con1.flex, ElementsAre(con2.flex[0], con2.flex[1]));
      EXPECT_THAT(con1.elem, ElementsAre(con2.elem[0], con2.elem[1]));
      EXPECT_THAT(con1.vert, ElementsAre(con2.vert[0], con2.vert[1]));
      EXPECT_EQ(con1.exclude, con2.exclude);
      EXPECT_EQ(con1.efc_address, con2.efc_address);
    }

    // one island with every dof
    for (const mjData* data : {dense.get(), sparse.get()}) {
      EXPECT_EQ(data->nisland, 1);
      EXPECT_EQ(data->nidof, nv);
    }
  }
}

TEST_F(IslandTest, IslandEfcElliptic) {
  const std::string xml_path = GetTestDataFilePath(kIlslandEfcPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  model->opt.cone = mjCONE_ELLIPTIC;
  while (data->time < 0.2) {
    mj_step(model, data);
  }
  mj_forward(model, data);

  EXPECT_EQ(data->nisland, 4);
  EXPECT_EQ(data->ne, 7);
  EXPECT_EQ(data->nf, 2);
  EXPECT_EQ(data->nl, 1);
  EXPECT_EQ(data->nefc, 25);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(IslandTest, EqualityConstraintOfTendons) {
  static const char xml[] = R"(
<mujoco>
  <worldbody>
    <body name="b1">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="j1" type="slide" axis="1 0 0"/>
    </body>
    <body name="b2">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="j2" type="slide" axis="1 0 0"/>
    </body>
    <body name="b3">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="j3" type="slide" axis="1 0 0"/>
    </body>
    <body name="b4">
      <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      <joint name="j4" type="slide" axis="1 0 0"/>
    </body>
  </worldbody>

  <tendon>
    <fixed name="t12">
      <joint joint="j1" coef="1"/>
      <joint joint="j2" coef="1"/>
    </fixed>
    <fixed name="t34">
      <joint joint="j3" coef="1"/>
      <joint joint="j4" coef="1"/>
    </fixed>
  </tendon>

  <equality>
    <tendon name="eq" tendon1="t12" tendon2="t34"/>
  </equality>
</mujoco>
)";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
}

TEST_F(IslandTest, PGSIsland) {
  const std::string xml_path = GetTestDataFilePath(kIlslandEfcPath);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  // simulate to get a non-trivial state
  while (d->time < 0.5) {
    mj_step(m, d);
  }

  // switch to PGS, disable early termination
  m->opt.solver = mjSOL_PGS;
  m->opt.tolerance = 0;

  // solve with islands
  m->opt.disableflags &= ~mjDSBL_ISLAND;
  mj_forward(m, d);
  ASSERT_GT(d->nisland, 1);
  std::vector<mjtNum> qfrc_island(d->qfrc_constraint,
                                  d->qfrc_constraint + m->nv);

  // solve without islands
  m->opt.disableflags |= mjDSBL_ISLAND;
  mj_forward(m, d);
  std::vector<mjtNum> qfrc_mono(d->qfrc_constraint, d->qfrc_constraint + m->nv);

  // expect close match (inexact due to randomized constraint visitation order)
  EXPECT_THAT(qfrc_island, Pointwise(MjNear(1e-3, 1e-3), qfrc_mono));

  mj_deleteData(d);
  mj_deleteModel(m);
}

}  // namespace
}  // namespace mujoco
