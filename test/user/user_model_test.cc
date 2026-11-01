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

// Tests for user/user_model.cc.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_numeric_format.h"
#include "test/compare_model.h"
#include "test/compare_spec.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pointwise;

static std::vector<mjtNum> GetRow(const mjtNum* array, int ncolumn, int row) {
  return std::vector<mjtNum>(array + ncolumn * row,
                             array + ncolumn * (row + 1));
}

// ----------------------------- test mjCModel  --------------------------------

using UserModelTest = MujocoTest;

// clarify the semantic of body_rootid and body_weldid
TEST_F(UserModelTest, WeldRootID) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="1">
        <joint/>
        <geom size="1"/>
        <body name="2">
          <joint/>
          <geom size="1"/>
        </body>
      </body>

      <body name="3">
        <joint/>
        <geom size="1"/>
        <body name="4">
          <geom size="1"/>
        </body>
      </body>

      <body name="5">
        <geom size="1"/>
        <body name="6">
          <geom size="1"/>
        </body>
      </body>

      <body name="7" mocap="true">
        <geom size="1"/>
        <body name="8">
          <geom size="1"/>
        </body>
      </body>

      <body name="9" mocap="true">
        <geom size="1"/>
        <body name="10">
          <joint/>
          <geom size="1"/>
        </body>
      </body>
     </worldbody>
   </mujoco>
   )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull());

  EXPECT_THAT(AsVector(model->body_rootid, model->nbody),
              ElementsAre(0, 1, 1, 3, 3, 5, 5, 7, 7, 9, 9));
  // mocap bodies (7, 9) are their own weld roots,
  // inherited by static children (8)
  EXPECT_THAT(AsVector(model->body_weldid, model->nbody),
              ElementsAre(0, 1, 2, 3, 3, 0, 0, 7, 7, 9, 10));
}

TEST_F(UserModelTest, RepeatedNames) {
  static constexpr char xml[] = R"(
   <mujoco>
     <worldbody>
       <body name="body1">
         <joint axis="0 1 0" name="joint1"/>
         <geom size="1" name="geom1"/>
         <geom size="1" name="geom1"/>
        </body>
      </worldbody>
    </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("repeated name 'geom1' in geom"));
}

TEST_F(UserModelTest, SameFrame) {
  static constexpr char xml[] = R"(
   <mujoco>
     <default>
      <geom type="box" size="1 2 3"/>
     </default>

     <worldbody>
       <body name="body1">
         <geom name="none"       mass="0" pos="1 1 1" euler="10 10 10"/>
         <geom name="body"       mass="0"/>
         <geom name="inertia"    mass="1" pos="3 2 1" euler="20 30 40"/>
         <geom name="bodyrot"    mass="0" pos="1 1 1"/>
         <geom name="inertiarot" mass="0" euler="20 30 40"/>
        </body>
      </worldbody>
    </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->geom_sameframe[0], mjSAMEFRAME_NONE);
  EXPECT_EQ(model->geom_sameframe[1], mjSAMEFRAME_BODY);
  EXPECT_EQ(model->geom_sameframe[2], mjSAMEFRAME_INERTIA);
  EXPECT_EQ(model->geom_sameframe[3], mjSAMEFRAME_BODYROT);
  EXPECT_EQ(model->geom_sameframe[4], mjSAMEFRAME_INERTIAROT);

  // make data, get geom_xpos
  MjDataPtr data = MakeData(model);
  mj_kinematics(model.get(), data.get());
  auto geom_xpos = AsVector(data->geom_xpos, model->ngeom * 3);
  auto geom_xmat = AsVector(data->geom_xmat, model->ngeom * 9);

  // set all geom_sameframe to 0, call kinematics again
  for (int i = 0; i < model->ngeom; i++) {
    model->geom_sameframe[i] = mjSAMEFRAME_NONE;
  }
  mj_resetData(model.get(), data.get());
  mj_kinematics(model.get(), data.get());
  auto geom_xpos2 = AsVector(data->geom_xpos, model->ngeom * 3);
  auto geom_xmat2 = AsVector(data->geom_xmat, model->ngeom * 9);

  // expect them to be equal
  constexpr double eps = 1e-6;
  EXPECT_THAT(geom_xpos, Pointwise(MjNear(eps, eps), geom_xpos2));
  EXPECT_THAT(geom_xmat, Pointwise(MjNear(eps, eps), geom_xmat2));
}

TEST_F(UserModelTest, ActuatorSparsity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="a"/>
        <body>
          <geom size="1"/>
          <joint name="b"/>
        </body>
      </body>
    </worldbody>
    <actuator>
      <motor joint="a"/>
      <motor joint="b"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nJmom, 2);
}

TEST_F(UserModelTest, FixedTendonSparsity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size=".1"/>
        <joint name="0"/>
      </body>
      <body pos="1 0 0">
        <geom size=".1"/>
        <joint name="1"/>
      </body>
      <body pos="2 0 0">
        <geom size=".1"/>
        <joint name="2"/>
      </body>
    </worldbody>

    <tendon>
      <fixed>
        <joint coef="3" joint="2"/>
        <joint coef="2" joint="1"/>
        <joint coef="1" joint="0"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());

  EXPECT_EQ(m->nJten, 3);
  EXPECT_EQ(m->ten_J_rownnz[0], 3);
  EXPECT_EQ(m->ten_J_rowadr[0], 0);
  EXPECT_EQ(m->wrap_type[m->tendon_adr[0]], mjWRAP_JOINT);

  int rowadr = m->ten_J_rowadr[0];
  int* colind = m->ten_J_colind + rowadr;
  EXPECT_THAT(std::vector<int>(colind, colind + 3), ElementsAre(0, 1, 2));
}

TEST_F(UserModelTest, TendonType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="s0"/>
      <body pos="1 0 0">
        <joint name="j"/>
        <geom size=".1"/>
        <site name="s1"/>
      </body>
    </worldbody>

    <tendon>
      <fixed>
        <joint joint="j" coef="1"/>
      </fixed>
      <spatial>
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // a tendon added through the spec API is typed by its path
  mjsTendon* tendon = mjs_addTendon(spec, nullptr);
  mjs_wrapJoint(tendon, "j", 2);

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->tendon_type[0], mjTENDON_FIXED);
  EXPECT_EQ(m->tendon_type[1], mjTENDON_SPATIAL);
  EXPECT_EQ(m->tendon_type[2], mjTENDON_FIXED);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(UserModelTest, NestedZeroMassBodiesOK) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <body>
          <body>
            <body>
              <geom size="1"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

TEST_F(UserModelTest, NestedZeroMassBodiesWithJointOK) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <body>
          <body>
            <body>
              <joint/>
              <geom size="1"/>
            </body>
            <body>
              <geom size="1"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
}

TEST_F(UserModelTest, NestedZeroMassBodiesFail) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <body name="bad">
          <freejoint/>
          <body>
            <body>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(
      error,
      HasSubstr(
          "mass and inertia of moving bodies must be larger than mjMINVAL"));
  EXPECT_THAT(error, HasSubstr("Element name 'bad'"));
}

// the statistic attribute meaninertia is deprecated: it does not affect
// simulation, and compilation warns
TEST_F(UserModelTest, MeanInertiaDeprecated) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag island="disable"/>
    </option>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mjs_numWarnings(spec), 0);

  mock_warning_handler.ExpectWarnings("meaninertia");
  spec->stat.meaninertia = 1000;
  mjModel* given = mj_compile(spec, nullptr);
  ASSERT_THAT(given, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(mjs_numWarnings(spec), 1);
  EXPECT_THAT(mjs_getWarning(spec, 0), HasSubstr("deprecated"));
  EXPECT_EQ(given->stat.meaninertia, 1000);

  // the same trajectory, for every solver
  for (int solver : {mjSOL_PGS, mjSOL_CG, mjSOL_NEWTON}) {
    m->opt.solver = given->opt.solver = solver;
    mjData* d = mj_makeData(m);
    mjData* d_given = mj_makeData(given);
    for (int i = 0; i < 10; i++) {
      mj_step(m, d);
      mj_step(given, d_given);
    }
    EXPECT_EQ(AsVector(d->qpos, m->nq), AsVector(d_given->qpos, m->nq))
        << "solver " << solver;
    mj_deleteData(d_given);
    mj_deleteData(d);
  }

  mj_deleteModel(given);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// settotalmass is deprecated: it still scales the masses, with a warning which
// says what to do instead
TEST_F(UserModelTest, SetTotalMassDeprecated) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size=".1" pos=".2 0 0"/>
        <body pos=".4 0 0">
          <joint axis="0 1 0"/>
          <inertial pos=".1 0 0" mass="2" diaginertia=".1 .2 .3"/>
          <geom type="capsule" size=".05" fromto="0 0 0 .3 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // without the attribute there is no warning
  mjModel* unscaled = mj_compile(spec, nullptr);
  ASSERT_THAT(unscaled, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mjs_numWarnings(spec), 0);

  // with it the masses are scaled as before, and compilation warns
  mock_warning_handler.ExpectWarnings("settotalmass");
  spec->compiler.settotalmass = 8;
  mjModel* scaled = mj_compile(spec, nullptr);
  ASSERT_THAT(scaled, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(mj_getTotalmass(scaled), 8, MjTol(1e-12, 1e-5));
  ASSERT_EQ(mjs_numWarnings(spec), 1);
  EXPECT_THAT(mjs_getWarning(spec, 0), HasSubstr("deprecated"));
  EXPECT_THAT(mjs_getWarning(spec, 0), HasSubstr("mj_setTotalmass"));

  // what the warning says to do instead gives the same model
  mj_setTotalmass(unscaled, 8);
  mjData* data = mj_makeData(unscaled);
  mj_setConst(unscaled, data);
  std::string field;
  EXPECT_LE(CompareModel(scaled, unscaled, field), MjTol(1e-12, 1e-5)) << field;

  mj_deleteData(data);
  mj_deleteModel(scaled);
  mj_deleteModel(unscaled);
  mj_deleteSpec(spec);
}

TEST_F(UserModelTest, ConvexHullForCollisionMeshes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh_no_hull" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
      <mesh name="mesh_with_hull_contype" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
      <mesh name="mesh_with_hull_conaffinity" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    </asset>
    <worldbody>
      <geom name="geom_no_hull" type="mesh" mesh="mesh_no_hull" contype="0" conaffinity="0"/>
      <geom name="geom_with_hull_contype" type="mesh" mesh="mesh_with_hull_contype" contype="1"/>
      <geom name="geom_with_hull_conaffinity" type="mesh" mesh="mesh_with_hull_conaffinity" conaffinity="1"/>
    </worldbody>
  </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  int no_hull_id = mj_name2id(model.get(), mjOBJ_MESH, "mesh_no_hull");
  int with_hull_contype_id =
      mj_name2id(model.get(), mjOBJ_MESH, "mesh_with_hull_contype");
  int with_hull_conaffinity_id =
      mj_name2id(model.get(), mjOBJ_MESH, "mesh_with_hull_conaffinity");

  EXPECT_NE(no_hull_id, -1);
  EXPECT_NE(with_hull_contype_id, -1);
  EXPECT_NE(with_hull_conaffinity_id, -1);

  // mesh_no_hull should not have a convex hull.
  EXPECT_EQ(model->mesh_graphadr[no_hull_id], -1);

  // mesh_with_hull_contype should have a convex hull.
  EXPECT_NE(model->mesh_graphadr[with_hull_contype_id], -1);

  // mesh_with_hull_conaffinity should have a convex hull.
  EXPECT_NE(model->mesh_graphadr[with_hull_conaffinity_id], -1);
}

TEST_F(UserModelTest, MeshExtremaValidIndices) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="test_mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    </asset>
    <worldbody>
      <geom name="test_geom" type="mesh" mesh="test_mesh"/>
    </worldbody>
  </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  int mesh_id = mj_name2id(model.get(), mjOBJ_MESH, "test_mesh");
  EXPECT_NE(mesh_id, -1);
  EXPECT_NE(model->mesh_graphadr[mesh_id], -1);

  const float* verts = model->mesh_vert + 3 * model->mesh_vertadr[mesh_id];
  const int* graph = model->mesh_graph + model->mesh_graphadr[mesh_id];
  const int* vert_globalid = graph + 2 + graph[0];

  int k = 0;
  for (int cx = -1; cx <= 1; cx++) {
    for (int cy = -1; cy <= 1; cy++) {
      for (int cz = -1; cz <= 1; cz++) {
        int v_idx = model->mesh_extrema[mesh_id * 27 + k];
        EXPECT_GE(v_idx, 0);
        EXPECT_LT(v_idx, graph[0]);

        int global_id = vert_globalid[v_idx];
        float max_dot = verts[3 * global_id + 0] * cx +
                        verts[3 * global_id + 1] * cy +
                        verts[3 * global_id + 2] * cz;

        // verify no other vertex yields a strictly greater dot product.
        for (int i = 0; i < graph[0]; i++) {
          int other_gid = vert_globalid[i];
          float dot = verts[3 * other_gid + 0] * cx +
                      verts[3 * other_gid + 1] * cy +
                      verts[3 * other_gid + 2] * cz;
          EXPECT_LE(dot, max_dot);
        }
        k++;
      }
    }
  }
}

TEST_F(UserModelTest, ConvexHullForPairCollisionMeshes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    </asset>
    <worldbody>
      <geom name="geom1" type="sphere" size="1"/>
      <geom name="geom_mesh" type="mesh" mesh="mesh" contype="0" conaffinity="0"/>
    </worldbody>
    <contact>
      <pair name="hello" geom1="geom1" geom2="geom_mesh"/>
    </contact>
  </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  int mesh_in_pair_id = mj_name2id(model.get(), mjOBJ_MESH, "mesh");

  EXPECT_NE(mesh_in_pair_id, -1);

  // mesh_in_pair should have a convex hull because it is in a collision pair.
  EXPECT_NE(model->mesh_graphadr[mesh_in_pair_id], -1);
}

// ------------- test automatic inference of nuser_xxx -------------------------

using UserDataTest = MujocoTest;

TEST_F(UserDataTest, AutoNUserBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body user="1 2 3"/>
      <body user="2 3"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_body, 3);
  EXPECT_THAT(GetRow(m->body_user, m->nuser_body, 1), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->body_user, m->nuser_body, 2), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserJoint) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint user="1 2 3"/>
        <joint user="2 3" axis="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_jnt, 3);
  EXPECT_THAT(GetRow(m->jnt_user, m->nuser_jnt, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->jnt_user, m->nuser_jnt, 1), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserGeom) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom size="1" user="1 2 3"/>
      <geom size="1" user="2 3"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_geom, 3);
  EXPECT_THAT(GetRow(m->geom_user, m->nuser_geom, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->geom_user, m->nuser_geom, 1), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserSite) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site user="1 2 3"/>
      <site user="2 3"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_site, 3);
  EXPECT_THAT(GetRow(m->site_user, m->nuser_site, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->site_user, m->nuser_site, 1), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserCamera) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera user="1 2 3"/>
      <camera user="2 3"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_cam, 3);
  EXPECT_THAT(GetRow(m->cam_user, m->nuser_cam, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->cam_user, m->nuser_cam, 1), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="a"/>
      <site name="b"/>
    </worldbody>
    <tendon>
      <spatial user="1 2 3">
        <site site="a"/>
        <site site="b"/>
      </spatial>
      <spatial user="2 3">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_tendon, 3);
  EXPECT_THAT(GetRow(m->tendon_user, m->nuser_tendon, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->tendon_user, m->nuser_tendon, 1), ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserActuator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="a"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="a" user="1 2 3"/>
      <motor joint="a" user="2 3"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_actuator, 3);
  EXPECT_THAT(GetRow(m->actuator_user, m->nuser_actuator, 0),
              ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->actuator_user, m->nuser_actuator, 1),
              ElementsAre(2, 3, 0));
}

TEST_F(UserDataTest, AutoNUserSensor) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="a"/>
    </worldbody>
    <sensor>
      <accelerometer site="a" user="1 2 3"/>
      <gyro site="a" user="2 3"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_EQ(m->nuser_sensor, 3);
  EXPECT_THAT(GetRow(m->sensor_user, m->nuser_sensor, 0), ElementsAre(1, 2, 3));
  EXPECT_THAT(GetRow(m->sensor_user, m->nuser_sensor, 1), ElementsAre(2, 3, 0));
}

// ------------- test duplicate names ------------------------------------------
TEST_F(UserDataTest, DuplicateNames) {
  static const char* const kFilePath = "user/testdata/malformed_duplicated.xml";
  const std::string xml_path = GetTestDataFilePath(kFilePath);

  std::array<char, 1024> error;
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());

  EXPECT_THAT(m, IsNull());
  EXPECT_THAT(error.data(), HasSubstr("repeated name 'cube' in mesh"));
}

// ------------- test fusestatic -----------------------------------------------

using FuseStaticTest = MujocoTest;
TEST_F(FuseStaticTest, FuseStaticEquivalent) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5" pos="1 0 0" contype="0" conaffinity="0"/>
        <body>
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::string fuse = absl::StrFormat(xml_template, "true");
  std::string no_fuse = absl::StrFormat(xml_template, "false");

  MjModelPtr m_fuse = LoadModelFromString(fuse.c_str());
  MjModelPtr m_no_fuse = LoadModelFromString(no_fuse.c_str());
  ASSERT_THAT(m_fuse.get(), NotNull());
  ASSERT_THAT(m_no_fuse.get(), NotNull());

  EXPECT_EQ(m_fuse->nbody, 2) << "Expecting a world body and one other body";
  EXPECT_EQ(m_no_fuse->nbody, 3) << "Expecting a world body and two others";

  EXPECT_EQ(m_no_fuse->body_contype[2], 1);
  EXPECT_EQ(m_no_fuse->body_conaffinity[2], 1);
  EXPECT_EQ(m_fuse->body_contype[1], 1);
  EXPECT_EQ(m_fuse->body_conaffinity[1], 1);

  EXPECT_EQ(m_no_fuse->body_bvhnum[2], 3);
  EXPECT_EQ(m_fuse->body_bvhnum[1], 3);

  MjDataPtr d_fuse = MakeData(m_fuse);
  MjDataPtr d_no_fuse = MakeData(m_no_fuse);

  mj_step(m_fuse.get(), d_fuse.get());
  mj_step(m_no_fuse.get(), d_no_fuse.get());

  EXPECT_NEAR(d_fuse.get()->qvel[0], d_no_fuse.get()->qvel[0],
              MjTol(2e-16, 1e-8))
      << "Velocity should be the same after 1 step";
  EXPECT_NE(d_fuse.get()->qvel[0], 0);
}

TEST_F(FuseStaticTest, FuseStaticActuatorReferencedBody) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>

    <worldbody>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5" pos="1 0 0" contype="0" conaffinity="0"/>
        <body name="not_referenced">
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
        <body name="referenced">
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
      </body>
    </worldbody>

    <actuator>
      <adhesion body="referenced" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml_template, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 3) << "Expecting a world body and two others";
}

TEST_F(FuseStaticTest, FuseStaticLightReferencedBody) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>

    <worldbody>
      <light mode="targetbody" target="referenced"/>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5" pos="1 0 0" contype="0" conaffinity="0"/>
        <body name="not_referenced">
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
        <body name="referenced">
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml_template, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 3) << "Expecting a world body and two others";
}

TEST_F(FuseStaticTest, FuseStaticForceSensorReferencedBody) {
  static constexpr char xml_template[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>

    <worldbody>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5" pos="1 0 0" contype="0" conaffinity="0"/>
        <body name="not_referenced">
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
        <body name="referenced">
          <site name="force"/>
          <geom size="0.5" pos="0 1 0" contype="1" conaffinity="1"/>
          <geom size="0.5" pos="0 -2 0" contype="1" conaffinity="1"/>
        </body>
      </body>
    </worldbody>

    <sensor>
      <force site="force"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml_template, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 3) << "Expecting a world body and two others";
}

// fusing a body changes the ids of the bodies which follow it
TEST_F(FuseStaticTest, FuseStaticBodyIdsAfterFuse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>
    <worldbody>
      <body>
        <geom size=".1"/>
      </body>
      <body name="B">
        <geom size=".1"/>
      </body>
      <body name="C">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <sensor>
      <framepos objtype="body" objname="C"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 2) << "Expecting a world body and the moving body";
  EXPECT_EQ(m->sensor_objid[0], mj_name2id(m.get(), mjOBJ_BODY, "C"));
}

// fusing a body which follows a sibling moves its elements ahead of the
// sibling's; names keep their objects, in the model and in the spec
TEST_F(FuseStaticTest, FuseStaticElementIdsAfterFuse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>
    <worldbody>
      <body name="moving">
        <freejoint/>
        <geom name="moving" size=".1"/>
        <site name="moving"/>
        <camera name="moving"/>
        <light name="moving"/>
      </body>
      <body pos="1 0 0">
        <geom name="static" size=".2"/>
        <site name="static"/>
        <camera name="static"/>
        <light name="static"/>
      </body>
    </worldbody>
    <sensor>
      <framepos objtype="geom" objname="moving"/>
      <framepos objtype="site" objname="moving"/>
      <framepos objtype="camera" objname="moving"/>
    </sensor>
    <custom>
      <tuple name="tuple">
        <element objtype="light" objname="moving"/>
      </tuple>
    </custom>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nbody, 2) << "Static body should be fused";

  // the fused elements precede the elements of the moving body
  EXPECT_EQ(m->sensor_objid[0], 1);
  EXPECT_EQ(m->sensor_objid[1], 1);
  EXPECT_EQ(m->sensor_objid[2], 1);
  EXPECT_EQ(m->tuple_objid[0], 1);

  for (mjtObj type : {mjOBJ_GEOM, mjOBJ_SITE, mjOBJ_CAMERA, mjOBJ_LIGHT}) {
    EXPECT_EQ(mj_name2id(m, type, "moving"), 1);
    for (const char* name : {"moving", "static"}) {
      mjsElement* element = mjs_findElement(spec, type, name);
      ASSERT_THAT(element, NotNull());
      EXPECT_STREQ(mjs_getString(mjs_getName(element)), name);
      EXPECT_EQ(mjs_getId(element), mj_name2id(m, type, name));
    }
  }

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// the fuse is an operation on the spec, which the compiler option applies
TEST_F(FuseStaticTest, FuseStaticOperation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="base" pos="0 0 1">
        <joint name="base"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <body name="static" pos="1 0 0" euler="0 0 90">
          <geom name="static" size=".1" pos="0 0 .5"/>
          <body name="first" pos="0 1 0">
            <joint name="first"/>
            <geom size=".1"/>
          </body>
        </body>
        <body name="second" pos="0 0 1">
          <joint name="second"/>
          <geom size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  std::string field;

  // compiled with the option
  mjSpec* spec_option = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec_option, NotNull()) << error.data();
  spec_option->compiler.fusestatic = 1;
  mjModel* m_option = mj_compile(spec_option, nullptr);
  ASSERT_THAT(m_option, NotNull()) << mjs_getError(spec_option);
  ASSERT_EQ(m_option->nbody, 4);

  // the operation followed by a compilation without the option
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_fuseStatic(spec, nullptr), 1) << mjs_getError(spec);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(CompareModel(m_option, m, field), 0) << field;

  // the body is a frame in its parent, which holds its geom as it was written
  EXPECT_THAT(mjs_findBody(spec, "static"), IsNull());
  mjsElement* geom = mjs_findElement(spec, mjOBJ_GEOM, "static");
  ASSERT_THAT(geom, NotNull());
  mjsFrame* frame = mjs_getFrame(geom);
  ASSERT_THAT(frame, NotNull());
  EXPECT_EQ(frame->pos[0], 1);
  EXPECT_EQ(mjs_asGeom(geom)->pos[2], 0.5);
  EXPECT_EQ(mjs_getParent(geom), mjs_findBody(spec, "base"));

  // its child body takes its place, so the joints keep their order
  EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "first"), 1);
  EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "second"), 2);

  // the parent, whose inertial is given, has the mass of the fused geom too
  const double sphere = 4.0 / 3.0 * mjPI * 1e-3 * 1000;
  EXPECT_NEAR(m->body_mass[1], 1 + sphere, MjTol(1e-12, 1e-6));

  // applying it again changes nothing
  mjSpec* fused = mj_copySpec(spec);
  EXPECT_EQ(mjs_fuseStatic(spec, nullptr), 0);
  EXPECT_THAT(CompareSpec(fused, spec), IsEmpty());

  // if the compilation fails after the fuse, the fuse stays and the error says
  // so
  mjSpec* spec_fail = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec_fail, NotNull()) << error.data();
  spec_fail->compiler.fusestatic = 1;
  mjsKey* key = mjs_addKey(spec_fail);
  const std::vector<double> qpos(5, 0);
  mjs_setDouble(key->qpos, qpos.data(), qpos.size());
  EXPECT_THAT(mj_compile(spec_fail, nullptr), IsNull());
  EXPECT_THAT(mjs_getError(spec_fail), HasSubstr("were fused"));
  EXPECT_THAT(mjs_findBody(spec_fail, "static"), IsNull());
  EXPECT_EQ(mjs_delete(spec_fail, key->element), 0);
  mjModel* m_fixed = mj_compile(spec_fail, nullptr);
  ASSERT_THAT(m_fixed, NotNull()) << mjs_getError(spec_fail);
  EXPECT_EQ(CompareModel(m_option, m_fixed, field), 0) << field;

  mj_deleteModel(m_option);
  mj_deleteModel(m);
  mj_deleteModel(m_fixed);
  mj_deleteSpec(fused);
  mj_deleteSpec(spec_fail);
  mj_deleteSpec(spec);
  mj_deleteSpec(spec_option);
}

// a body whose fuse attribute is "false" is kept
TEST_F(FuseStaticTest, FuseStaticKeepsBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>
    <worldbody>
      <body name="moving">
        <joint/>
        <geom size=".1"/>
        <body name="kept" pos="1 0 0" fuse="false">
          <geom size=".1"/>
          <body name="inner" pos="0 1 0">
            <geom size=".1"/>
          </body>
        </body>
        <body name="fused" pos="0 0 1">
          <geom size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_findBody(spec, "kept")->fuse, 0);
  EXPECT_EQ(mjs_findBody(spec, "fused")->fuse, 1);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  // the body is kept, the static bodies in it and beside it are fused
  EXPECT_EQ(m->nbody, 3);
  EXPECT_EQ(mj_name2id(m, mjOBJ_BODY, "kept"), 2);
  EXPECT_THAT(mjs_findBody(spec, "inner"), IsNull());
  EXPECT_THAT(mjs_findBody(spec, "fused"), IsNull());

  // the attribute is saved, only where it is not the default
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved,
              HasSubstr(R"(<body name="kept" pos="1 0 0" fuse="false">)"));
  EXPECT_EQ(saved.find("fuse="), saved.rfind("fuse="));

  // without the attribute, the operation fuses the body
  mjs_findBody(spec, "kept")->fuse = 1;
  EXPECT_EQ(mjs_fuseStatic(spec, nullptr), 1) << mjs_getError(spec);
  EXPECT_THAT(mjs_findBody(spec, "kept"), IsNull());

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// inertia which compilation adjusted is kept: the mass of each body is raised
// to boundmass, which the geoms of both in one body would not reproduce
TEST_F(FuseStaticTest, FuseStaticBoundMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler boundmass="1" inertiafromgeom="%s"/>
    <worldbody>
      <body name="moving">
        <joint name="slide" type="slide"/>
        <geom name="moving" size=".1" mass=".1"/>
        <body name="static" pos="1 0 0">
          <geom name="static" size=".1" mass=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  for (const char* inertiafromgeom : {"auto", "true"}) {
    SCOPED_TRACE(inertiafromgeom);
    std::string model = absl::StrFormat(xml, inertiafromgeom);
    MjModelPtr no_fuse = LoadModelFromString(model, error.data(), error.size());
    ASSERT_THAT(no_fuse.get(), NotNull()) << error.data();
    MjDataPtr d_no_fuse = MakeData(no_fuse);
    mj_forward(no_fuse.get(), d_no_fuse.get());
    EXPECT_EQ(d_no_fuse->M[0], 2);

    mjSpec* spec =
        mj_parseXMLString(model.c_str(), 0, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << error.data();
    EXPECT_EQ(mjs_fuseStatic(spec, nullptr),
              std::string(inertiafromgeom) == "true" ? 0 : 1)
        << mjs_getError(spec);
    MjModelPtr fuse(mj_compile(spec, nullptr));
    ASSERT_THAT(fuse.get(), NotNull()) << mjs_getError(spec);

    // the sum cannot be written when the inertia of every body is inferred: the
    // body is kept
    EXPECT_EQ(fuse->nbody, std::string(inertiafromgeom) == "true" ? 3 : 2);
    MjDataPtr d_fuse = MakeData(fuse);
    mj_forward(fuse.get(), d_fuse.get());
    EXPECT_NEAR(d_fuse->M[0], 2, MjTol(1e-12, 1e-6));
    mj_deleteSpec(spec);
  }
}

// a body which is referenced by a skin is not fused, also if the skin is read
// from a file
TEST_F(FuseStaticTest, FuseStaticSkinReferencedBody) {
  const std::string path = GetTestDataFilePath("user/testdata/cube_skin.xml");
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  spec->compiler.fusestatic = 1;
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nbody, 2);
  EXPECT_EQ(m->skin_bonebodyid[0], mj_name2id(m, mjOBJ_BODY, "box"));
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a body which is referenced by a flex is not fused
TEST_F(FuseStaticTest, FuseStaticFlexReferencedBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>

    <worldbody>
      <body>
        <freejoint/>
        <geom size="0.5"/>
        <body name="not_referenced">
          <geom size="0.1"/>
        </body>
        <body name="v0">
          <geom size="0.1"/>
        </body>
        <body name="v1" pos="1 0 0">
          <geom size="0.1"/>
        </body>
        <body name="v2" pos="0 1 0">
          <geom size="0.1"/>
        </body>
      </body>
    </worldbody>

    <deformable>
      <flex name="flex" dim="2" body="v0 v1 v2" element="0 1 2"/>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 5) << "Expecting world, a moving body and 3 flex bodies";
  EXPECT_EQ(mj_name2id(m.get(), mjOBJ_BODY, "not_referenced"), -1);
  for (int i = 0; i < 3; i++) {
    std::string name = absl::StrFormat("v%d", i);
    EXPECT_EQ(m->flex_vertbodyid[i],
              mj_name2id(m.get(), mjOBJ_BODY, name.c_str()));
  }
}

TEST_F(FuseStaticTest, FuseStaticCameraInBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>
    <worldbody>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5"/>
        <body pos="1 0 0">
          <site name="site1"/>
          <camera name="cam1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 2) << "Static body should be fused";
  EXPECT_EQ(m->ncam, 1);
}

TEST_F(FuseStaticTest, FuseStaticLightInBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="true"/>
    <worldbody>
      <body>
        <joint axis="1 0 0"/>
        <geom size="0.5"/>
        <body pos="1 0 0">
          <light name="light1" dir="0 0 -1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->nbody, 2) << "Static body should be fused";
  EXPECT_EQ(m->nlight, 1);
}

// expect equal global poses of same-named elements, equal joint-space inertia
// and equal unconstrained acceleration of a moving model
static void ExpectEquivalent(const MjModelPtr& m1, const MjModelPtr& m2) {
  ASSERT_EQ(m1->nv, m2->nv);
  ASSERT_EQ(m1->nC, m2->nC);
  MjDataPtr d1 = MakeData(m1);
  MjDataPtr d2 = MakeData(m2);
  mju_fill(d1->qvel, 1, m1->nv);
  mju_fill(d2->qvel, 1, m2->nv);
  mj_forward(m1.get(), d1.get());
  mj_forward(m2.get(), d2.get());

  struct Field {
    const char* name;
    mjtObj type;
    mjtSize num;
    int dim;
    const mjtNum* x1;
    const mjtNum* x2;
  };
  const Field fields[] = {
      {"xpos", mjOBJ_BODY, m1->nbody, 3, d1->xpos, d2->xpos},
      {"xmat", mjOBJ_BODY, m1->nbody, 9, d1->xmat, d2->xmat},
      {"xanchor", mjOBJ_JOINT, m1->njnt, 3, d1->xanchor, d2->xanchor},
      {"xaxis", mjOBJ_JOINT, m1->njnt, 3, d1->xaxis, d2->xaxis},
      {"geom_xpos", mjOBJ_GEOM, m1->ngeom, 3, d1->geom_xpos, d2->geom_xpos},
      {"geom_xmat", mjOBJ_GEOM, m1->ngeom, 9, d1->geom_xmat, d2->geom_xmat},
      {"site_xpos", mjOBJ_SITE, m1->nsite, 3, d1->site_xpos, d2->site_xpos},
      {"site_xmat", mjOBJ_SITE, m1->nsite, 9, d1->site_xmat, d2->site_xmat},
      {"cam_xpos", mjOBJ_CAMERA, m1->ncam, 3, d1->cam_xpos, d2->cam_xpos},
      {"cam_xmat", mjOBJ_CAMERA, m1->ncam, 9, d1->cam_xmat, d2->cam_xmat},
      {"light_xpos", mjOBJ_LIGHT, m1->nlight, 3, d1->light_xpos,
       d2->light_xpos},
      {"light_xdir", mjOBJ_LIGHT, m1->nlight, 3, d1->light_xdir,
       d2->light_xdir},
  };
  for (const Field& f : fields) {
    for (int i = 0; i < f.num; i++) {
      const char* name = mj_id2name(m1.get(), f.type, i);
      ASSERT_THAT(name, NotNull()) << f.name << " " << i;
      int j = mj_name2id(m2.get(), f.type, name);
      ASSERT_GE(j, 0) << name;
      EXPECT_THAT(
          AsVector(f.x1 + f.dim * i, f.dim),
          Pointwise(MjNear(1e-15, 2e-6), AsVector(f.x2 + f.dim * j, f.dim)))
          << f.name << " of " << name;
    }
  }

  // fused inertia is as accurate as its principal axes
  mjtNum scale = 1 / mju_norm(d2->M, m2->nC);
  mju_scl(d1->M, d1->M, scale, m1->nC);
  mju_scl(d2->M, d2->M, scale, m2->nC);
  EXPECT_THAT(AsVector(d1->M, m1->nC),
              Pointwise(MjNear(1e-6, 1e-6), AsVector(d2->M, m2->nC)));

  // so are the bias and passive forces
  scale = 1 / mju_norm(d2->qacc_smooth, m2->nv);
  mju_scl(d1->qacc_smooth, d1->qacc_smooth, scale, m1->nv);
  mju_scl(d2->qacc_smooth, d2->qacc_smooth, scale, m2->nv);
  EXPECT_THAT(AsVector(d1->qacc_smooth, m1->nv),
              Pointwise(MjNear(2e-6, 1e-5), AsVector(d2->qacc_smooth, m2->nv)));
}

// expect the fused model to be equivalent to the model which is not fused,
// and to be reproduced by recompiling, copying and saving the fused spec
static void ExpectCoherentFuse(const std::string& fuse_xml,
                               const std::string& no_fuse_xml, mjtNum tol) {
  std::array<char, 1024> error;
  MjModelPtr no_fuse =
      LoadModelFromString(no_fuse_xml, error.data(), error.size());
  ASSERT_THAT(no_fuse.get(), NotNull()) << error.data();

  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
      mj_parseXMLString(fuse_xml.c_str(), nullptr, error.data(), error.size()),
      mj_deleteSpec);
  ASSERT_THAT(spec.get(), NotNull()) << error.data();
  MjModelPtr fuse(mj_compile(spec.get(), nullptr));
  ASSERT_THAT(fuse.get(), NotNull()) << mjs_getError(spec.get());
  EXPECT_LT(fuse->nbody, no_fuse->nbody);
  ExpectEquivalent(fuse, no_fuse);

  std::string field;
  MjModelPtr recompiled(mj_compile(spec.get(), nullptr));
  ASSERT_THAT(recompiled.get(), NotNull()) << mjs_getError(spec.get());
  EXPECT_LE(CompareModel(fuse.get(), recompiled.get(), field), tol)
      << "recompiled model is different: " << field;

  std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> copy(
      mj_copySpec(spec.get()), mj_deleteSpec);
  MjModelPtr copied(mj_compile(copy.get(), nullptr));
  ASSERT_THAT(copied.get(), NotNull()) << mjs_getError(copy.get());
  EXPECT_LE(CompareModel(fuse.get(), copied.get(), field), tol)
      << "copied model is different: " << field;

  FullFloatPrecision increase_precision;
  MjModelPtr saved = LoadModelFromString(SaveAndReadXml(spec.get()),
                                         error.data(), error.size());
  ASSERT_THAT(saved.get(), NotNull()) << error.data();
  EXPECT_LE(CompareModel(fuse.get(), saved.get(), field), tol)
      << "saved model is different: " << field;
}

TEST_F(FuseStaticTest, FuseStaticFrames) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body name="fixed" pos="1 2 3" euler="10 20 30">
        <frame pos="0 0 1" euler="0 0 30">
          <geom name="fixed" type="box" size=".1 .2 .3" pos=".1 0 0"/>
        </frame>
      </body>
      <body name="moving" pos="0 0 1">
        <freejoint name="free"/>
        <inertial pos=".1 .2 .3" mass="2" diaginertia="1 2 3"/>
        <geom name="moving" size=".1"/>
        <frame name="wrapper" pos="0 0 .5" euler="0 90 0">
          <body name="static" pos="1 0 0" euler="0 0 90">
            <geom name="static" type="box" size=".1 .2 .3" pos="0 0 1"/>
            <frame name="outer" pos="0 1 0" euler="90 0 0">
              <geom name="outer" type="box" size=".3 .2 .1" pos=".1 .2 .3"/>
              <geom name="fromto" type="box" size=".1" fromto="0 0 0 .1 .2 .3"/>
              <frame name="inner" pos="0 0 1" euler="0 45 0">
                <site name="inner" pos=".2 .3 .4" euler="0 30 0"/>
                <camera name="inner" pos=".3 .4 .5" euler="10 20 30"/>
                <light name="inner" pos=".4 .5 .6" dir="1 2 3"/>
                <body name="child" pos=".5 .6 .7" euler="30 20 10">
                  <joint name="hinge" axis="1 2 3"/>
                  <geom name="child" type="box" size=".1 .2 .3" pos=".1 0 0"/>
                </body>
                <body name="nested" pos=".7 .6 .5" euler="10 30 20">
                  <geom name="nested" type="box" size=".2 .3 .1" pos="0 .1 0"/>
                </body>
              </frame>
            </frame>
          </body>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";
  ExpectCoherentFuse(absl::StrFormat(xml, "true"),
                     absl::StrFormat(xml, "false"), 0);
}

TEST_F(FuseStaticTest, FuseStaticInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body name="moving">
        <freejoint name="free"/>
        %s
        <geom name="moving" type="box" size=".1 .2 .3" pos=".1 0 0"/>
        <body name="static" pos="1 0 0" euler="10 20 30">
          %s
          <geom name="static" type="box" size=".3 .1 .2" pos="0 .2 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char parent[] =
      R"(<inertial pos=".1 .2 .3" mass="2" diaginertia="1 2 3"/>)";
  static constexpr char parent_full[] =
      R"(<inertial pos=".1 .2 .3" mass="2" fullinertia="4 3 2 .3 .2 .1"/>)";
  static constexpr char parent_framed[] = R"(
      <frame pos="0 0 1" euler="0 0 30">
        <inertial pos=".1 .2 .3" mass="2" diaginertia="1 2 3"/>
      </frame>)";
  static constexpr char child[] =
      R"(<inertial pos=".3 .2 .1" mass="3" diaginertia="3 2 4"/>)";
  static constexpr char child_framed[] = R"(
      <frame pos="0 1 0" euler="30 0 0">
        <inertial pos=".3 .2 .1" mass="3" diaginertia="3 2 4"/>
      </frame>)";

  for (const char* parent_inertial : {"", parent, parent_full, parent_framed}) {
    for (const char* child_inertial : {"", child, child_framed}) {
      SCOPED_TRACE(absl::StrFormat("parent '%s' child '%s'", parent_inertial,
                                   child_inertial));

      // inertia inferred from the fused geoms is as accurate as principal axes
      bool inferred = !*parent_inertial && !*child_inertial;
      ExpectCoherentFuse(
          absl::StrFormat(xml, "true", parent_inertial, child_inertial),
          absl::StrFormat(xml, "false", parent_inertial, child_inertial),
          inferred ? 1e-5 : 0);
    }
  }
}

// the geoms of a body which gives a massless inertial do not add their mass to
// a parent which infers its inertia when the body is fused
TEST_F(FuseStaticTest, FuseStaticMasslessInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body name="moving">
        <freejoint name="free"/>
        <geom name="moving" type="box" size=".1 .2 .3" pos=".1 0 0"/>
        <body name="static" pos="1 0 0" euler="10 20 30">
          <inertial pos="%s" mass="%s" diaginertia="0 0 0"/>
          <geom name="static" type="box" size=".3 .1 .2" pos="0 .2 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  struct Case {
    const char* pos;
    const char* mass;
  };
  const Case cases[] = {{"0 0 0", "0"}, {".1 .2 .3", "1e-20"}};
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrFormat("pos '%s' mass '%s'", c.pos, c.mass));
    std::string fuse_xml = absl::StrFormat(xml, "true", c.pos, c.mass);
    std::string no_fuse_xml = absl::StrFormat(xml, "false", c.pos, c.mass);
    MjModelPtr fuse = LoadModelFromString(fuse_xml);
    MjModelPtr no_fuse = LoadModelFromString(no_fuse_xml);
    ASSERT_THAT(fuse.get(), NotNull());
    ASSERT_THAT(no_fuse.get(), NotNull());
    ASSERT_EQ(fuse->nbody, 2);
    ASSERT_EQ(no_fuse->nbody, 3);

    int i = mj_name2id(fuse.get(), mjOBJ_BODY, "moving");
    int j = mj_name2id(no_fuse.get(), mjOBJ_BODY, "moving");
    ASSERT_GE(i, 0);
    ASSERT_GE(j, 0);
    EXPECT_NEAR(fuse->body_mass[i], no_fuse->body_mass[j], 1e-12);
    EXPECT_NEAR(fuse->body_subtreemass[i], no_fuse->body_subtreemass[j], 1e-12);
    EXPECT_THAT(AsVector(fuse->body_inertia + 3 * i, 3),
                Pointwise(DoubleNear(1e-12),
                          AsVector(no_fuse->body_inertia + 3 * j, 3)));
    EXPECT_THAT(
        AsVector(fuse->body_ipos + 3 * i, 3),
        Pointwise(DoubleNear(1e-12), AsVector(no_fuse->body_ipos + 3 * j, 3)));
    ExpectCoherentFuse(fuse_xml, no_fuse_xml, 0);
  }
}

// fusestatic does not discard the free joint alignment of cameras and lights
TEST_F(FuseStaticTest, FuseStaticAlignFree) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s" alignfree="true"/>
    <worldbody>
      <body name="moving" pos="0 0 1">
        <freejoint name="free"/>
        <geom name="moving" type="box" size=".1 .2 .3" pos="1 2 3" euler="10 20 30"/>
        <camera name="moving" pos="0 1 0" euler="30 20 10"/>
        <light name="moving" pos="0 1 0" dir="1 2 3"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr fuse = LoadModelFromString(absl::StrFormat(xml, "true"));
  MjModelPtr no_fuse = LoadModelFromString(absl::StrFormat(xml, "false"));
  ASSERT_THAT(fuse.get(), NotNull());
  ASSERT_THAT(no_fuse.get(), NotNull());
  ExpectEquivalent(fuse, no_fuse);
}

// gravcomp applies to the mass of each body, fused masses have equal gravcomp
TEST_F(FuseStaticTest, FuseStaticGravcomp) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body name="fixed" gravcomp="1">
        <geom name="fixed" size=".1"/>
      </body>
      <body name="moving" pos="0 0 1" gravcomp="%s">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="moving" size=".1"/>
        <body name="massless" gravcomp="2">
          <site name="massless"/>
        </body>
        <body name="static" pos="1 0 0" gravcomp="%s">
          <geom name="static" size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  struct Case {
    const char* parent;
    const char* child;
    int nbody;
  };

  // bodies which are fixed to the world or massless are always fused
  const Case cases[] = {
      {"0", ".5", 3}, {".5", "0", 3}, {".5", "2", 3}, {".5", ".5", 2}};
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrFormat("parent %s child %s", c.parent, c.child));
    std::string fuse_xml = absl::StrFormat(xml, "true", c.parent, c.child);
    MjModelPtr fuse = LoadModelFromString(fuse_xml);
    ASSERT_THAT(fuse.get(), NotNull());
    EXPECT_EQ(fuse->nbody, c.nbody);
    ExpectCoherentFuse(fuse_xml,
                       absl::StrFormat(xml, "false", c.parent, c.child),
                       c.nbody == 2 ? 1e-5 : 0);
  }
}

// fluid forces apply to the inertia and ellipsoid geoms of each body, which are
// not fused in a fluid
TEST_F(FuseStaticTest, FuseStaticFluid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <option %s/>
    <worldbody>
      <body name="fixed">
        <geom name="fixed" size=".1"/>
      </body>
      <body name="moving" pos="0 0 1">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="moving" size=".1"/>
        <body name="massless" pos="0 0 1">
          <site name="massless"/>
        </body>
        <body name="static" pos="1 0 0">
          <geom name="static" size=".1" %s/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  struct Case {
    const char* option;
    const char* geom;
    int nbody;
  };

  // bodies which are fixed to the world or have no mass and no ellipsoid geoms
  // are always fused
  const Case cases[] = {
      {"", "", 2},
      {R"(density="1000")", "", 3},
      {R"(viscosity="1")", "", 3},
      {R"(density="1000")", R"(mass="0")", 2},
      {R"(density="1000")", R"(mass="0" fluidshape="ellipsoid")", 3}};
  for (const Case& c : cases) {
    SCOPED_TRACE(absl::StrFormat("option '%s' geom '%s'", c.option, c.geom));
    std::string fuse_xml = absl::StrFormat(xml, "true", c.option, c.geom);
    MjModelPtr fuse = LoadModelFromString(fuse_xml);
    ASSERT_THAT(fuse.get(), NotNull());
    EXPECT_EQ(fuse->nbody, c.nbody);
    ExpectCoherentFuse(fuse_xml,
                       absl::StrFormat(xml, "false", c.option, c.geom),
                       c.nbody == 2 ? 1e-5 : 0);
  }
}

// a body with a plugin is not fused, the plugin's forces are specific to it
TEST_F(FuseStaticTest, FuseStaticPlugin) {
  // passive plugin applying an upward force to the center of mass of its bodies
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.lift";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int type) {
    for (int i = 1; i < m->nbody; i++) {
      if (m->body_plugin[i] == instance) {
        mjtNum force[3] = {0, 0, 10}, torque[3] = {0};
        mj_applyFT(m, d, force, torque, d->xipos + 3 * i, i, d->qfrc_passive);
      }
    }
  };
  mjp_registerPlugin(&plugin);

  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <extension>
      <plugin plugin="mujoco.test.lift"/>
    </extension>
    <worldbody>
      <body name="moving" pos="0 0 1">
        <joint name="hinge" axis="0 1 0"/>
        <geom name="moving" size=".1"/>
        <body name="static" pos="1 0 0">
          <geom name="static" size=".1"/>
          <plugin plugin="mujoco.test.lift"/>
        </body>
        <body name="fused" pos="0 1 0">
          <geom name="fused" size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::string fuse_xml = absl::StrFormat(xml, "true");
  MjModelPtr fuse = LoadModelFromString(fuse_xml);
  ASSERT_THAT(fuse.get(), NotNull());
  EXPECT_EQ(fuse->nbody, 3);
  ExpectCoherentFuse(fuse_xml, absl::StrFormat(xml, "false"), 1e-5);
}

// the sleep policy of a static body is an error, with or without fusing
TEST_F(FuseStaticTest, FuseStaticSleepPolicy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler fusestatic="%s"/>
    <worldbody>
      <body>
        <joint/>
        <geom size=".1"/>
        <body sleep="never">
          <geom size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  for (const char* fusestatic : {"false", "true"}) {
    std::array<char, 1024> error;
    MjModelPtr m = LoadModelFromString(absl::StrFormat(xml, fusestatic),
                                       error.data(), error.size());
    EXPECT_THAT(m.get(), IsNull()) << "fusestatic " << fusestatic;
    EXPECT_THAT(error.data(), HasSubstr("sleep policy only allowed"));
  }
}

// ------------- test discardvisual --------------------------------------------

using DiscardVisualTest = MujocoTest;
TEST_F(DiscardVisualTest, DiscardVisualKeepsInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler discardvisual="true"/>

    <asset>
      <mesh name="visual_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="1 0 0  0 1 0  0 0 1  0.707 0 0.707"
        face="0 2 1  0 3 2" />

      <mesh name="collision_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="1 0 0  0 1 0  0 0 1  0.707 0 0.707"
        face="0 2 1  0 3 2" />
    </asset>

    <worldbody>
      <body>
        <geom type="mesh" mesh="visual_mesh" contype="0" conaffinity="0"/>
      </body>
      <body>
        <geom type="mesh" mesh="collision_mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_THAT(model->nmesh, 1);
  EXPECT_THAT(model->body_inertia[3], model->body_inertia[6]);
  EXPECT_THAT(model->body_inertia[4], model->body_inertia[7]);
  EXPECT_THAT(model->body_inertia[5], model->body_inertia[8]);
}

TEST_F(DiscardVisualTest, DiscardVisualEquivalent) {
  char error[1024];
  size_t error_sz = 1024;

  static const char* const kDiscardvisualPath =
      "user/testdata/discardvisual.xml";
  static const char* const kDiscardvisualFalsePath =
      "user/testdata/discardvisual_false.xml";

  const std::string xml_path1 = GetTestDataFilePath(kDiscardvisualPath);
  mjModel* model1 = mj_loadXML(xml_path1.c_str(), 0, error, error_sz);
  EXPECT_THAT(model1, NotNull()) << error;

  const std::string xml_path2 = GetTestDataFilePath(kDiscardvisualFalsePath);
  mjModel* model2 = mj_loadXML(xml_path2.c_str(), 0, error, error_sz);
  EXPECT_THAT(model2, NotNull()) << error;

  EXPECT_THAT(model1->nq, model2->nq);
  EXPECT_THAT(model1->nmat, 0);
  EXPECT_THAT(model1->ntex, 0);
  EXPECT_THAT(model2->ngeom - model1->ngeom, 3);
  EXPECT_THAT(model2->nmesh - model1->nmesh, 2);
  EXPECT_THAT(model1->npair, model2->npair);
  EXPECT_THAT(model1->nsensor, model2->nsensor);
  EXPECT_THAT(model1->nwrap, model2->nwrap);

  for (int i = 0; i < model1->ngeom; i++) {
    std::string name = std::string(model1->names + model1->name_geomadr[i]);
    EXPECT_NE(name.find("kept"), std::string::npos);
    EXPECT_EQ(name.find("discard"), std::string::npos);
  }

  for (int i = 0; i < model1->npair; i++) {
    int adr1 = model1->name_geomadr[model1->pair_geom1[i]];
    int adr2 = model2->name_geomadr[model2->pair_geom1[i]];
    EXPECT_STREQ(model1->names + adr1, model2->names + adr2);
    adr1 = model1->name_geomadr[model1->pair_geom2[i]];
    adr2 = model2->name_geomadr[model2->pair_geom2[i]];
    EXPECT_STREQ(model1->names + adr1, model2->names + adr2);
  }

  for (int i = 0; i < model1->nsensor; i++) {
    int adr1 = model1->name_geomadr[model1->sensor_objid[i]];
    int adr2 = model2->name_geomadr[model2->sensor_objid[i]];
    EXPECT_STREQ(model1->names + adr1, model2->names + adr2);
  }

  for (int i = 0; i < model1->nwrap; i++) {
    int adr1 = model1->name_geomadr[model1->wrap_objid[i]];
    int adr2 = model2->name_geomadr[model2->wrap_objid[i]];
    EXPECT_STREQ(model1->names + adr1, model2->names + adr2);
  }

  mjData* d1 = mj_makeData(model1);
  mjData* d2 = mj_makeData(model2);
  for (int i = 0; i < 100; i++) {
    mj_step(model1, d1);
    mj_step(model2, d2);
  }

  for (int i = 0; i < model1->nq; i++) {
    EXPECT_THAT(d1->qpos[i], d2->qpos[i]);
  }

  mj_deleteModel(model1);
  mj_deleteModel(model2);
  mj_deleteData(d1);
  mj_deleteData(d2);
}

TEST_F(DiscardVisualTest, FindElementAfterDiscard) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler discardvisual="true"/>
    <asset>
      <texture name="tex" type="2d" builtin="checker" width="8" height="8"/>
      <material name="mat" texture="tex"/>
      <mesh name="vismesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="colmesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <body name="b">
        <joint/>
        <geom name="vis1" type="mesh" mesh="vismesh" contype="0" conaffinity="0" material="mat"/>
        <geom name="vis2" size=".2" contype="0" conaffinity="0"/>
        <geom name="col1" size=".1"/>
        <geom name="col2" type="mesh" mesh="colmesh"/>
        <geom name="col3" size=".1" pos="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->ngeom, 3);
  EXPECT_EQ(model->nmesh, 1);

  // discarded elements are not found
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "vis1"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "vis2"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "vismesh"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MATERIAL, "mat"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_TEXTURE, "tex"), IsNull());

  // surviving elements are found under their own names, at their new ids
  for (const char* name : {"col1", "col2", "col3"}) {
    mjsElement* el = mjs_findElement(spec, mjOBJ_GEOM, name);
    ASSERT_THAT(el, NotNull()) << name;
    EXPECT_STREQ(mjs_getString(mjs_getName(el)), name);
    EXPECT_EQ(mjs_getId(el), mj_name2id(model, mjOBJ_GEOM, name));
  }
  mjsElement* colmesh = mjs_findElement(spec, mjOBJ_MESH, "colmesh");
  ASSERT_THAT(colmesh, NotNull());
  EXPECT_STREQ(mjs_getString(mjs_getName(colmesh)), "colmesh");
  EXPECT_EQ(mjs_getId(colmesh), mj_name2id(model, mjOBJ_MESH, "colmesh"));

  // the spec recompiles to the same sizes
  mjModel* model2 = mj_compile(spec, 0);
  ASSERT_THAT(model2, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model2->ngeom, 3);
  EXPECT_EQ(model2->nmesh, 1);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "vis1"), IsNull());

  mj_deleteModel(model2);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// discarding is an operation on the spec, which the compiler option applies
TEST_F(DiscardVisualTest, DiscardVisualOperation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option density="1.2" viscosity="1e-5"/>
    <asset>
      <texture name="tex" type="2d" builtin="checker" width="8" height="8"/>
      <material name="mat" texture="tex"/>
      <mesh name="visual" vertex="0 0 0  1 0 0  0 1 0  0 0 1" material="mat"/>
      <mesh name="collision" vertex="0 0 0  1 0 0  0 1 0  0 0 1" material="mat"/>
      <mesh name="site" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="tuple" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <mesh name="unused" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <default>
      <geom material="mat"/>
      <site material="mat"/>
      <default class="visual">
        <geom contype="0" conaffinity="0"/>
      </default>
    </default>
    <worldbody>
      <light name="light" texture="tex"/>
      <site name="s0"/>
      <geom name="visual_world" class="visual" size=".1"/>
      <body name="inferred">
        <joint/>
        <geom name="collision" type="mesh" mesh="collision"/>
        <geom name="visual" class="visual" type="mesh" mesh="visual" pos="1 0 0"/>
        <geom name="visual_massless" class="visual" size=".1" mass="0"/>
        <geom name="wing" class="visual" type="ellipsoid" size=".3 .2 .01" fluidshape="ellipsoid"/>
        <site name="s1" type="mesh" mesh="site" pos="0 0 1"/>
        <body name="explicit" pos="0 0 1">
          <joint/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <geom name="visual_explicit" class="visual" size=".1"/>
          <geom name="sensor" class="visual" size=".1"/>
          <geom name="pair" class="visual" size=".1"/>
          <geom name="wrap" class="visual" size=".1" pos="0 .5 0"/>
        </body>
      </body>
    </worldbody>
    <contact>
      <pair geom1="collision" geom2="pair"/>
    </contact>
    <tendon>
      <spatial name="tendon" material="mat">
        <site site="s0"/>
        <geom geom="wrap"/>
        <site site="s1"/>
      </spatial>
    </tendon>
    <sensor>
      <framepos objtype="geom" objname="sensor"/>
    </sensor>
    <custom>
      <tuple name="tuple">
        <element objtype="mesh" objname="tuple"/>
      </tuple>
    </custom>
  </mujoco>
  )";
  std::array<char, 1024> error;
  std::string field;

  // the model with its visual elements
  mjSpec* spec_full = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec_full, NotNull()) << error.data();
  mjModel* m_full = mj_compile(spec_full, nullptr);
  ASSERT_THAT(m_full, NotNull()) << mjs_getError(spec_full);

  // compiled with the option
  mjSpec* spec_option = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec_option, NotNull()) << error.data();
  spec_option->compiler.discardvisual = 1;
  mjModel* m_option = mj_compile(spec_option, nullptr);
  ASSERT_THAT(m_option, NotNull()) << mjs_getError(spec_option);

  // the operation followed by a compilation without the option
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 8) << mjs_getError(spec);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(CompareModel(m_option, m, field), 0) << field;

  // materials and textures are gone, with the references to them
  EXPECT_EQ(m->nmat, 0);
  EXPECT_EQ(m->ntex, 0);
  EXPECT_EQ(m->light_texid[0], -1);

  // geoms are kept if they collide, have fluid forces or are referenced
  for (const char* name : {"collision", "wing", "sensor", "pair", "wrap"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, name), NotNull()) << name;
  }
  for (const char* name :
       {"visual_world", "visual", "visual_massless", "visual_explicit"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, name), IsNull()) << name;
  }

  // meshes are kept if a remaining geom or a site uses them, or if they are
  // referenced
  for (const char* name : {"collision", "site", "tuple"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, name), NotNull()) << name;
  }
  for (const char* name : {"visual", "unused"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, name), IsNull()) << name;
  }

  // inertia which was inferred from a discarded geom is kept, as an explicit
  // inertial
  int inferred = mj_name2id(m, mjOBJ_BODY, "inferred");
  EXPECT_EQ(mjs_findBody(spec, "inferred")->explicitinertial, 1);
  EXPECT_NEAR(m->body_mass[inferred], m_full->body_mass[inferred],
              MjTol(1e-12, 1e-6));
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(m->body_inertia[3 * inferred + i],
                m_full->body_inertia[3 * inferred + i], MjTol(1e-12, 1e-6));
    EXPECT_NEAR(m->body_ipos[3 * inferred + i],
                m_full->body_ipos[3 * inferred + i], MjTol(1e-12, 1e-6));
  }

  // applying it again changes nothing
  mjSpec* discarded = mj_copySpec(spec);
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 0);
  EXPECT_THAT(CompareSpec(discarded, spec), IsEmpty());

  // compiling again and saving reproduce the model
  mjModel* m_again = mj_compile(spec, nullptr);
  ASSERT_THAT(m_again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(CompareModel(m, m_again, field), 0) << field;
  {
    FullFloatPrecision increase_precision;
    MjModelPtr saved =
        LoadModelFromString(SaveAndReadXml(spec), error.data(), error.size());
    ASSERT_THAT(saved.get(), NotNull()) << error.data();
    EXPECT_LE(CompareModel(m, saved.get(), field), MjTol(1e-12, 1e-6)) << field;
  }

  mj_deleteModel(m_again);
  mj_deleteModel(m);
  mj_deleteModel(m_option);
  mj_deleteModel(m_full);
  mj_deleteSpec(discarded);
  mj_deleteSpec(spec);
  mj_deleteSpec(spec_option);
  mj_deleteSpec(spec_full);
}

// a material or texture which a sensor or tuple refers to is kept, as an
// element which nothing renders with
TEST_F(DiscardVisualTest, DiscardVisualKeepsReferencedAssets) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="discarded" type="2d" builtin="checker" width="8" height="8"/>
      <texture name="in_material" type="2d" builtin="checker" width="8" height="8"/>
      <texture name="in_tuple" type="2d" builtin="checker" width="8" height="8"/>
      <material name="discarded" texture="discarded"/>
      <material name="in_tuple" texture="in_material"/>
    </asset>
    <worldbody>
      <body>
        <joint/>
        <geom name="geom" size=".1" material="in_tuple"/>
      </body>
    </worldbody>
    <custom>
      <tuple name="tuple">
        <element objtype="material" objname="in_tuple"/>
        <element objtype="texture" objname="in_tuple"/>
      </tuple>
    </custom>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 3) << mjs_getError(spec);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  // what the tuple refers to is kept, and the tuple refers to it
  EXPECT_EQ(m->nmat, 1);
  EXPECT_EQ(m->ntex, 1);
  EXPECT_EQ(m->tuple_objid[0], mj_name2id(m, mjOBJ_MATERIAL, "in_tuple"));
  EXPECT_EQ(m->tuple_objid[1], mj_name2id(m, mjOBJ_TEXTURE, "in_tuple"));

  // nothing uses it: the geom has no material, the material no texture
  EXPECT_EQ(m->geom_matid[0], -1);
  for (int i = 0; i < mjNTEXROLE; i++) {
    EXPECT_EQ(m->mat_texid[i], -1) << i;
  }

  // applying it again changes nothing
  mjSpec* discarded = mj_copySpec(spec);
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 0);
  EXPECT_THAT(CompareSpec(discarded, spec), IsEmpty());

  mj_deleteModel(m);
  mj_deleteSpec(discarded);
  mj_deleteSpec(spec);
}

// with inertiafromgeom "true", inertia which is inferred from a discarded geom
// cannot be kept
TEST_F(DiscardVisualTest, DiscardVisualInertiaFromGeomTrue) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiafromgeom="true"/>
    <asset>
      <material name="mat"/>
    </asset>
    <worldbody>
      <body name="body">
        <joint/>
        <geom size=".1" material="mat"/>
        <geom name="visual" size=".2" contype="0" conaffinity="0" %s/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;

  // the geom has mass: an error, and nothing is discarded
  std::string xml_mass = absl::StrFormat(xml, "");
  mjSpec* spec =
      mj_parseXMLString(xml_mass.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjSpec* copy = mj_copySpec(spec);
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("inertiafromgeom 'true'"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("body"));
  EXPECT_THAT(CompareSpec(copy, spec), IsEmpty());

  // the same through the compiler option
  spec->compiler.discardvisual = 1;
  EXPECT_THAT(mj_compile(spec, nullptr), IsNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("inertiafromgeom 'true'"));
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "visual"), NotNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MATERIAL, "mat"), NotNull());
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);

  // the geom has no mass: it is discarded
  std::string xml_massless = absl::StrFormat(xml, R"(mass="0")");
  spec = mj_parseXMLString(xml_massless.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m_full = mj_compile(spec, nullptr);
  ASSERT_THAT(m_full, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 2) << mjs_getError(spec);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "visual"), IsNull());
  EXPECT_EQ(mjs_findBody(spec, "body")->explicitinertial, 0);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->body_mass[1], m_full->body_mass[1]);
  mj_deleteModel(m);
  mj_deleteModel(m_full);
  mj_deleteSpec(spec);
}

// assets are read only if inertia is inferred from a discarded geom
TEST_F(DiscardVisualTest, DiscardVisualReadsAssetsIfNeeded) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="visual" file="this_file_does_not_exist.obj"/>
    </asset>
    <worldbody>
      <body name="body">
        <joint/>
        %s
        <geom size=".1"/>
        <geom name="visual" type="mesh" mesh="visual" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;

  // the body has an explicit inertial: the visual mesh is discarded without
  // being read
  std::string xml_explicit = absl::StrFormat(
      xml, R"(<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>)");
  mjSpec* spec =
      mj_parseXMLString(xml_explicit.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), 2) << mjs_getError(spec);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nmesh, 0);
  EXPECT_EQ(m->ngeom, 1);
  mj_deleteModel(m);
  mj_deleteSpec(spec);

  // its inertia is inferred: the mesh is needed, so it is an error and nothing
  // is discarded
  std::string xml_inferred = absl::StrFormat(xml, "");
  spec = mj_parseXMLString(xml_inferred.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_discardVisual(spec, nullptr), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("this_file_does_not_exist.obj"));
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "visual"), NotNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "visual"), NotNull());
  mj_deleteSpec(spec);
}

// if compilation fails after the discard, the discard stays and the error says
// so
TEST_F(DiscardVisualTest, DiscardVisualThenError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler discardvisual="true"/>
    <worldbody>
      <body name="body">
        <joint/>
        <geom size=".1"/>
        <geom name="visual" size=".2" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key" qpos="1 2 3"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_THAT(mj_compile(spec, nullptr), IsNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("were discarded"));
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "visual"), IsNull());

  // the spec is valid once the error is fixed
  EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_KEY, "key")), 0);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->ngeom, 1);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// discarding visual elements deletes the plugin instances which only they refer
// to, also in a spec which was never compiled, and keeps those which another
// element refers to
TEST_F(DiscardVisualTest, DiscardVisualDeletesPluginInstances) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="shared">
          <config key="radius1" value="0.35"/>
          <config key="radius2" value="0.15"/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="shared">
        <plugin instance="shared"/>
      </mesh>
      <mesh name="visual">
        <plugin plugin="mujoco.sdf.torus">
          <config key="radius1" value="0.25"/>
          <config key="radius2" value="0.125"/>
        </plugin>
      </mesh>
    </asset>
    <worldbody>
      <body>
        <freejoint/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom name="collides" type="sdf" mesh="shared">
          <plugin instance="shared"/>
        </geom>
        <geom name="visual" type="sdf" mesh="visual" contype="0" conaffinity="0">
          <plugin plugin="mujoco.sdf.torus">
            <config key="radius1" value="0.25"/>
            <config key="radius2" value="0.125"/>
          </plugin>
        </geom>
        <geom name="shares" type="sdf" mesh="shared" contype="0" conaffinity="0">
          <plugin instance="shared"/>
        </geom>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto instances = [](mjSpec* spec) {
    int n = 0;
    for (mjsElement* plugin = mjs_firstElement(spec, mjOBJ_PLUGIN); plugin;
         plugin = mjs_nextElement(spec, plugin)) {
      n++;
    }
    return n;
  };
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  ASSERT_EQ(instances(spec), 3);

  ASSERT_EQ(mjs_discardVisual(spec, nullptr), 3) << mjs_getError(spec);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "visual"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_GEOM, "shares"), IsNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "visual"), IsNull());
  ASSERT_EQ(instances(spec), 1);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_PLUGIN, "shared"), NotNull());

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 1);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// ------------- test mjs_dedupAssets ------------------------------------------

using DedupAssetsTest = MujocoTest;

// the assets of a model, with duplicates of some of them when dup is true and
// references to the duplicates in every kind of element, including a default
// class, a sensor and a tuple; near-duplicates differ in one field each
std::string DedupAssetsXml(bool dup) {
  const char* tex = dup ? "tex_dup" : "tex";
  const char* cube = dup ? "cube_dup" : "cube";
  const char* mesh = dup ? "mesh_dup" : "mesh";
  std::string assets = dup ? R"(
      <texture name="tex_dup" type="2d" builtin="checker" width="8" height="8" rgb1="1 0 0"/>
      <texture name="cube_dup" type="cube" builtin="gradient" width="4" height="4"/>
      <mesh name="mesh_dup" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
  )"
                           : "";
  return absl::StrFormat(R"(
  <mujoco>
    <asset>
      <texture name="tex" type="2d" builtin="checker" width="8" height="8" rgb1="1 0 0"/>
      <texture name="cube" type="cube" builtin="gradient" width="4" height="4"/>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      %s
      <texture name="tex_hflip" type="2d" builtin="checker" width="8" height="8" rgb1="1 0 0"
        hflip="true"/>
      <texture name="tex_rgb" type="2d" builtin="checker" width="8" height="8" rgb1="0 1 0"/>
      <texture name="tex_height" type="2d" builtin="checker" width="8" height="16" rgb1="1 0 0"/>
      <texture name="tex_type" type="cube" builtin="checker" width="8" height="8" rgb1="1 0 0"/>
      <mesh name="mesh_scale" vertex="0 0 0  1 0 0  0 1 0  0 0 1" scale="2 2 2"/>
      <mesh name="mesh_vert" vertex="0 0 0  1 0 0  0 1 0  0 0 2"/>
      <mesh name="mesh_inertia" vertex="0 0 0  1 0 0  0 1 0  0 0 1" inertia="shell"/>
      <mesh name="mesh_hull" vertex="0 0 0  1 0 0  0 1 0  0 0 1" maxhullvert="4"/>
      <mesh name="mesh_material" vertex="0 0 0  1 0 0  0 1 0  0 0 1" material="mat"/>
      <material name="mat" texture="tex"/>
      <material name="mat_dup" texture="%s"/>
      <material name="mat_hflip" texture="tex_hflip"/>
      <material name="mat_rgb" texture="tex_rgb"/>
      <material name="mat_height" texture="tex_height"/>
      <material name="mat_type" texture="tex_type"/>
      <material name="mat_cube" texture="%s"/>
      <material name="mat_class" class="dup"/>
    </asset>
    <default>
      <default class="dup">
        <geom type="mesh" mesh="%s" material="mat_dup"/>
        <site type="mesh" mesh="%s"/>
        <material texture="%s"/>
      </default>
    </default>
    <worldbody>
      <light name="light" texture="%s"/>
      <body>
        <joint/>
        <geom name="g" type="mesh" mesh="mesh" material="mat"/>
        <geom name="g_dup" type="mesh" mesh="%s" material="mat_dup"/>
        <geom name="g_class" class="dup"/>
        <geom name="g_scale" type="mesh" mesh="mesh_scale"/>
        <geom name="g_vert" type="mesh" mesh="mesh_vert"/>
        <geom name="g_inertia" type="mesh" mesh="mesh_inertia"/>
        <geom name="g_hull" type="mesh" mesh="mesh_hull"/>
        <geom name="g_material" type="mesh" mesh="mesh_material"/>
        <site name="s_dup" type="mesh" mesh="%s"/>
        <site name="s_class" class="dup"/>
      </body>
    </worldbody>
    <sensor>
      <tactile name="tactile" geom="g" mesh="%s"/>
    </sensor>
    <custom>
      <tuple name="tuple">
        <element objtype="texture" objname="%s"/>
        <element objtype="mesh" objname="%s"/>
      </tuple>
    </custom>
  </mujoco>
  )",
                         assets, tex, cube, mesh, mesh, tex, tex, mesh, mesh,
                         mesh, tex, mesh);
}

// duplicates are merged into the first, with every reference to them;
// near-duplicates are kept
TEST_F(DedupAssetsTest, DedupAssets) {
  std::array<char, 1024> error;
  std::string field;

  // the model written without the duplicates
  std::string xml_merged = DedupAssetsXml(/*dup=*/false);
  mjSpec* spec_merged =
      mj_parseXMLString(xml_merged.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec_merged, NotNull()) << error.data();
  mjModel* m_merged = mj_compile(spec_merged, nullptr);
  ASSERT_THAT(m_merged, NotNull()) << mjs_getError(spec_merged);

  // the operation on the model with the duplicates
  std::string xml = DedupAssetsXml(/*dup=*/true);
  mjSpec* spec = mj_parseXMLString(xml.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjsMesh* mesh_dup = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh_dup"));
  ASSERT_THAT(mesh_dup, NotNull());
  EXPECT_EQ(mjs_dedupAssets(spec), 3) << mjs_getError(spec);

  // the duplicates are gone, the survivors keep their names, the rest stays
  for (const char* name : {"tex_dup", "cube_dup"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_TEXTURE, name), IsNull()) << name;
  }
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "mesh_dup"), IsNull());
  for (const char* name :
       {"tex", "cube", "tex_hflip", "tex_rgb", "tex_height", "tex_type"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_TEXTURE, name), NotNull()) << name;
  }
  for (const char* name : {"mesh", "mesh_scale", "mesh_vert", "mesh_inertia",
                           "mesh_hull", "mesh_material"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, name), NotNull()) << name;
  }

  // the references point at the survivors
  mjsMaterial* mat_dup =
      mjs_asMaterial(mjs_findElement(spec, mjOBJ_MATERIAL, "mat_dup"));
  mjsMaterial* mat_cube =
      mjs_asMaterial(mjs_findElement(spec, mjOBJ_MATERIAL, "mat_cube"));
  mjsGeom* g_dup = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "g_dup"));
  mjsSite* s_dup = mjs_asSite(mjs_findElement(spec, mjOBJ_SITE, "s_dup"));
  mjsLight* light = mjs_asLight(mjs_findElement(spec, mjOBJ_LIGHT, "light"));
  mjsTuple* tuple = mjs_asTuple(mjs_findElement(spec, mjOBJ_TUPLE, "tuple"));
  mjsSensor* tactile =
      mjs_asSensor(mjs_findElement(spec, mjOBJ_SENSOR, "tactile"));
  mjsDefault* def = mjs_findDefault(spec, "dup");
  ASSERT_THAT(mat_dup, NotNull());
  ASSERT_THAT(mat_cube, NotNull());
  ASSERT_THAT(g_dup, NotNull());
  ASSERT_THAT(s_dup, NotNull());
  ASSERT_THAT(light, NotNull());
  ASSERT_THAT(tuple, NotNull());
  ASSERT_THAT(tactile, NotNull());
  ASSERT_THAT(def, NotNull());
  EXPECT_EQ((*mat_dup->textures)[mjTEXROLE_RGB], "tex");
  EXPECT_EQ((*mat_cube->textures)[mjTEXROLE_RGB], "cube");
  EXPECT_EQ(*g_dup->meshname, "mesh");
  EXPECT_EQ(*s_dup->meshname, "mesh");
  EXPECT_EQ(*light->texture, "tex");
  EXPECT_THAT(*tuple->objname, ElementsAre("tex", "mesh"));
  EXPECT_EQ(*tactile->objname, "mesh");
  EXPECT_EQ(*def->geom->meshname, "mesh");
  EXPECT_EQ(*def->site->meshname, "mesh");
  EXPECT_EQ((*def->material->textures)[mjTEXROLE_RGB], "tex");

  // the compiled model is that of the model written without the duplicates
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->ntex, 6);
  EXPECT_EQ(m->nmesh, 6);
  EXPECT_EQ(CompareModel(m_merged, m, field), 0) << field;

  // applying it again changes nothing
  mjSpec* once = mj_copySpec(spec);
  EXPECT_EQ(mjs_dedupAssets(spec), 0) << mjs_getError(spec);
  EXPECT_THAT(CompareSpec(once, spec), IsEmpty());

  // compiling again gives the same model
  mjModel* m_again = mj_compile(spec, nullptr);
  ASSERT_THAT(m_again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(CompareModel(m, m_again, field), 0) << field;

  mj_deleteModel(m_again);
  mj_deleteModel(m);
  mj_deleteModel(m_merged);
  mj_deleteSpec(once);
  mj_deleteSpec(spec);
  mj_deleteSpec(spec_merged);
}

// compiled ids are resolved from names: merging after a compilation and
// compiling again gives the merged model
TEST_F(DedupAssetsTest, DedupAssetsAfterCompile) {
  std::array<char, 1024> error;
  std::string field;
  std::string xml = DedupAssetsXml(/*dup=*/true);
  mjSpec* spec = mj_parseXMLString(xml.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m_dup = mj_compile(spec, nullptr);
  ASSERT_THAT(m_dup, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m_dup->ntex, 8);
  EXPECT_EQ(m_dup->nmesh, 7);
  uint64_t signature = spec->element->signature;
  EXPECT_NE(signature, 0);

  EXPECT_EQ(mjs_dedupAssets(spec), 3) << mjs_getError(spec);
  EXPECT_EQ(spec->element->signature, 0);

  // no compiled pointer to a merged asset is left: a copy of the spec made
  // before compiling it again compiles to the same model
  mjSpec* copy = mj_copySpec(spec);
  ASSERT_THAT(copy, NotNull());
  mjModel* m_copy = mj_compile(copy, nullptr);
  ASSERT_THAT(m_copy, NotNull()) << mjs_getError(copy);

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(CompareModel(m, m_copy, field), 0) << field;
  mj_deleteModel(m_copy);
  mj_deleteSpec(copy);
  EXPECT_EQ(m->ntex, 6);
  EXPECT_EQ(m->nmesh, 6);
  EXPECT_NE(m->signature, signature);

  // the geoms and the materials refer to the survivors
  int mesh = mj_name2id(m, mjOBJ_MESH, "mesh");
  int tex = mj_name2id(m, mjOBJ_TEXTURE, "tex");
  for (const char* name : {"g", "g_dup", "g_class"}) {
    EXPECT_EQ(m->geom_dataid[mj_name2id(m, mjOBJ_GEOM, name)], mesh) << name;
  }
  for (const char* name : {"s_dup", "s_class"}) {
    EXPECT_EQ(m->site_dataid[mj_name2id(m, mjOBJ_SITE, name)], mesh) << name;
  }
  for (const char* name : {"mat", "mat_dup", "mat_class"}) {
    int mat = mj_name2id(m, mjOBJ_MATERIAL, name);
    EXPECT_EQ(m->mat_texid[mat * mjNTEXROLE + mjTEXROLE_RGB], tex) << name;
  }
  EXPECT_EQ(m->light_texid[0], tex);
  EXPECT_EQ(m->tuple_objid[0], tex);
  EXPECT_EQ(m->tuple_objid[1], mesh);
  EXPECT_EQ(m->sensor_objid[0], mesh);

  mj_deleteModel(m);
  mj_deleteModel(m_dup);
  mj_deleteSpec(spec);
}

// meshes with a plugin are never merged, nor are assets which differ only in
// their buffers
TEST_F(DedupAssetsTest, DedupAssetsKeepsDistinct) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus"/>
    </extension>
    <asset>
      <mesh name="plugin1">
        <plugin plugin="mujoco.sdf.torus"/>
      </mesh>
      <mesh name="plugin2">
        <plugin plugin="mujoco.sdf.torus"/>
      </mesh>
      <mesh name="normal1" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="1 0 0  1 0 0  1 0 0  1 0 0"/>
      <mesh name="normal2" vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        normal="0 1 0  0 1 0  0 1 0  0 1 0"/>
    </asset>
    <worldbody>
      <geom type="sdf" mesh="plugin1">
        <plugin plugin="mujoco.sdf.torus"/>
      </geom>
      <geom type="sdf" mesh="plugin2">
        <plugin plugin="mujoco.sdf.torus"/>
      </geom>
      <geom type="mesh" mesh="normal1"/>
      <geom type="mesh" mesh="normal2"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_dedupAssets(spec), 0) << mjs_getError(spec);
  for (const char* name : {"plugin1", "plugin2", "normal1", "normal2"}) {
    EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, name), NotNull()) << name;
  }
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nmesh, 4);
  mj_deleteModel(m);
  mj_deleteSpec(spec);

  // textures from buffers are merged only if the buffers are the same
  spec = mj_makeSpec();
  mjsTexture* tex1 = mjs_addTexture(spec, nullptr);
  mjsTexture* tex2 = mjs_addTexture(spec, nullptr);
  mjs_setName(tex1->element, "buffer1");
  mjs_setName(tex2->element, "buffer2");
  std::vector<std::byte> pixels1(3, std::byte{0});
  std::vector<std::byte> pixels2(3, std::byte{255});
  for (mjsTexture* tex : {tex1, tex2}) {
    tex->type = mjTEXTURE_2D;
    tex->width = 1;
    tex->height = 1;
    tex->nchannel = 3;
  }
  mjs_setBuffer(tex1->data, pixels1.data(), pixels1.size());
  mjs_setBuffer(tex2->data, pixels2.data(), pixels2.size());
  mjSpec* copy = mj_copySpec(spec);
  EXPECT_EQ(mjs_dedupAssets(spec), 0) << mjs_getError(spec);
  EXPECT_THAT(CompareSpec(copy, spec), IsEmpty());

  mjs_setBuffer(tex2->data, pixels1.data(), pixels1.size());
  EXPECT_EQ(mjs_dedupAssets(spec), 1) << mjs_getError(spec);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_TEXTURE, "buffer1"), NotNull());
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_TEXTURE, "buffer2"), IsNull());
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// assets whose names repeat cannot be merged, and nothing is changed
TEST_F(DedupAssetsTest, DedupAssetsRepeatedName) {
  mjSpec* spec = mj_makeSpec();
  for (int i = 0; i < 3; i++) {
    mjsMesh* mesh = mjs_addMesh(spec, nullptr);
    std::vector<float> vert = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
    mjs_setFloat(mesh->uservert, vert.data(), vert.size());

    // mjs_setName refuses a repeated name, write it directly
    *mjs_getName(mesh->element) = i == 2 ? "other" : "mesh";
  }
  auto names = [](mjSpec* spec) {
    std::vector<std::string> names;
    for (mjsElement* el = mjs_firstElement(spec, mjOBJ_MESH); el;
         el = mjs_nextElement(spec, el)) {
      names.push_back(*mjs_getName(el));
    }
    return names;
  };
  EXPECT_EQ(mjs_dedupAssets(spec), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("repeated name 'mesh'"));
  EXPECT_THAT(names(spec), ElementsAre("mesh", "mesh", "other"));
  mj_deleteSpec(spec);
}

// the assets of an attached spec belong to its parent, they cannot be merged
TEST_F(DedupAssetsTest, DedupAssetsOfAttachedSpec) {
  mjSpec* child = mj_makeSpec();
  mjs_addBody(mjs_findBody(child, "world"), nullptr);
  mjSpec* parent = mj_makeSpec();
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);
  ASSERT_THAT(mjs_attach(frame->element, child->element, "child-", ""),
              NotNull())
      << mjs_getError(parent);
  EXPECT_EQ(mjs_dedupAssets(child), -1);
  EXPECT_THAT(mjs_getError(child),
              HasSubstr("Cannot merge the assets of an attached mjSpec"));
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

// a model which attaches the same sub-model three times has three copies of
// its assets: they are merged into those of the first attachment, and the
// model renders the same
TEST_F(DedupAssetsTest, DedupAssetsAttached) {
  static const char* const kParentPath =
      "user/testdata/dedup_assets_parent.xml";
  std::array<char, 1024> error;
  std::string field;
  auto parse = [&]() {
    return mj_parseXML(GetTestDataFilePath(kParentPath).c_str(), 0,
                       error.data(), error.size());
  };

  // the model with three copies of the assets
  mjSpec* spec_full = parse();
  ASSERT_THAT(spec_full, NotNull()) << error.data();
  mjModel* m_full = mj_compile(spec_full, nullptr);
  ASSERT_THAT(m_full, NotNull()) << mjs_getError(spec_full);
  EXPECT_EQ(m_full->ntex, 6);
  EXPECT_EQ(m_full->nmesh, 3);
  EXPECT_EQ(m_full->nmat, 6);

  // merged by hand: the assets of the first attachment are kept, the others
  // are deleted and what referred to them refers to the first
  mjSpec* spec_hand = parse();
  ASSERT_THAT(spec_hand, NotNull()) << error.data();
  for (const char* prefix : {"b_", "c_"}) {
    std::string p = prefix;
    for (const char* tex : {"carpet", "checker"}) {
      mjsMaterial* mat = mjs_asMaterial(
          mjs_findElement(spec_hand, mjOBJ_MATERIAL, (p + tex).c_str()));
      ASSERT_THAT(mat, NotNull());
      (*mat->textures)[mjTEXROLE_RGB] = std::string("a_") + tex;
      mjsElement* el =
          mjs_findElement(spec_hand, mjOBJ_TEXTURE, (p + tex).c_str());
      ASSERT_THAT(el, NotNull());
      EXPECT_EQ(mjs_delete(spec_hand, el), 0) << mjs_getError(spec_hand);
    }
    mjsBody* body = mjs_findBody(spec_hand, (p + "body").c_str());
    ASSERT_THAT(body, NotNull());
    for (mjsElement* el = mjs_firstChild(body, mjOBJ_GEOM, 0); el;
         el = mjs_nextChild(body, el, 0)) {
      mjs_setString(mjs_asGeom(el)->meshname, "a_torus");
    }
    mjsElement* el =
        mjs_findElement(spec_hand, mjOBJ_MESH, (p + "torus").c_str());
    ASSERT_THAT(el, NotNull());
    EXPECT_EQ(mjs_delete(spec_hand, el), 0) << mjs_getError(spec_hand);
  }
  mjModel* m_hand = mj_compile(spec_hand, nullptr);
  ASSERT_THAT(m_hand, NotNull()) << mjs_getError(spec_hand);

  // merged by the operation
  mjSpec* spec = parse();
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(mjs_dedupAssets(spec), 6) << mjs_getError(spec);
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->ntex, 2);
  EXPECT_EQ(m->nmesh, 1);
  EXPECT_EQ(m->nmat, 6);
  EXPECT_EQ(CompareModel(m_hand, m, field), 0) << field;

  // every geom and material refers to the same asset as in the full model,
  // which has the same data
  EXPECT_EQ(m->ngeom, m_full->ngeom);
  for (int i = 0; i < m->ngeom; i++) {
    int full = m_full->geom_dataid[i];
    int merged = m->geom_dataid[i];
    ASSERT_GE(merged, 0);
    EXPECT_EQ(m_full->mesh_vertnum[full], m->mesh_vertnum[merged]) << i;
    EXPECT_EQ(m_full->mesh_facenum[full], m->mesh_facenum[merged]) << i;
    EXPECT_EQ(m->geom_matid[i], m_full->geom_matid[i]) << i;
  }
  for (int i = 0; i < m->nmat; i++) {
    int full = m_full->mat_texid[i * mjNTEXROLE + mjTEXROLE_RGB];
    int merged = m->mat_texid[i * mjNTEXROLE + mjTEXROLE_RGB];
    ASSERT_GE(merged, 0);
    EXPECT_EQ(m_full->tex_width[full], m->tex_width[merged]) << i;
    EXPECT_EQ(m_full->tex_height[full], m->tex_height[merged]) << i;
    EXPECT_EQ(m_full->tex_nchannel[full], m->tex_nchannel[merged]) << i;
    int size =
        m->tex_width[merged] * m->tex_height[merged] * m->tex_nchannel[merged];
    EXPECT_EQ(std::memcmp(m_full->tex_data + m_full->tex_adr[full],
                          m->tex_data + m->tex_adr[merged], size),
              0)
        << i;
  }

  // the physics is the same
  EXPECT_EQ(m->nbody, m_full->nbody);
  for (int i = 0; i < m->nbody; i++) {
    EXPECT_EQ(m->body_mass[i], m_full->body_mass[i]) << i;
    for (int j = 0; j < 3; j++) {
      EXPECT_EQ(m->body_inertia[3 * i + j], m_full->body_inertia[3 * i + j]);
      EXPECT_EQ(m->body_ipos[3 * i + j], m_full->body_ipos[3 * i + j]);
    }
  }

  mj_deleteModel(m);
  mj_deleteModel(m_hand);
  mj_deleteModel(m_full);
  mj_deleteSpec(spec);
  mj_deleteSpec(spec_hand);
  mj_deleteSpec(spec_full);
}

// ------------- test lengthrange ----------------------------------------------

using LengthRangeTest = MujocoTest;

TEST_F(LengthRangeTest, LengthRangeThreading) {
  char error[1024];
  size_t error_sz = 1024;
  std::string field = "";

  static const char* const kLengthrangePath = "user/testdata/lengthrange.xml";

  const std::string xml_path1 = GetTestDataFilePath(kLengthrangePath);
  mjSpec* spec = mj_parseXML(xml_path1.c_str(), 0, error, error_sz);
  EXPECT_THAT(spec, NotNull()) << error;
  mjModel* model1 = mj_compile(spec, 0);
  EXPECT_THAT(model1, NotNull()) << error;

  // model is such that the lengthrange for first actuator is [1, sqrt(5)]
  EXPECT_NEAR(model1->actuator_lengthrange[0], 1.0, 1e-3);
  EXPECT_NEAR(model1->actuator_lengthrange[1], std::sqrt(5.0), 1e-3);

  // recompile without threads
  ASSERT_EQ(spec->compiler.usethread, 1);
  spec->compiler.usethread = 0;
  mjModel* model2 = mj_compile(spec, 0);
  EXPECT_THAT(model2, NotNull()) << error;

  // expect threaded and unthreaded models to be identical
  EXPECT_LE(CompareModel(model1, model2, field), 0)
      << "Threaded and unthreaded lengthrange models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteModel(model1);
  mj_deleteModel(model2);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, ResolvePluginMissingInstanceThrowsError) {
  // Instance name="my_pid_config" dos not match actuator plugin's
  // instance="pid_config", this should throw an appropriate error
  static constexpr char xml_mismatch[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.pid">
        <instance name="my_pid_config" />
      </plugin>
    </extension>
    <worldbody>
      <body name="block" pos="0 0 0.5">
        <joint name="slide_z" type="slide" axis="0 0 1" />
        <geom type="box" size="0.1 0.1 0.1" mass="1.0" rgba="0 0.7 0 1"/>
      </body>
    </worldbody>
    <actuator>
      <plugin name="pid_actuator" joint="slide_z" plugin="mujoco.pid" instance="pid_config" />
    </actuator>
  </mujoco>
  )";

  std::array<char, 1024> error_buffer;
  mjSpec* spec = mj_parseXMLString(xml_mismatch, 0, error_buffer.data(),
                                   error_buffer.size());
  ASSERT_THAT(spec, NotNull()) << error_buffer.data();

  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_THAT(model, IsNull());

  std::string error_msg = mjs_getError(spec);
  EXPECT_THAT(error_msg,
              HasSubstr("unrecognized name 'pid_config' for plugin instance"));

  if (model) mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// ----------------------------- test modeldir  --------------------------------

TEST_F(MujocoTest, Modeldir) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "meshdir/cube.obj", cube, sizeof(cube));

  // child with the asset
  mjSpec* child = mj_makeSpec();
  mjsMesh* mesh = mjs_addMesh(child, 0);
  mjsFrame* frame = mjs_addFrame(mjs_findBody(child, "world"), 0);
  mjsGeom* geom = mjs_addGeom(mjs_findBody(child, "world"), 0);
  mjs_setString(child->compiler.meshdir, "meshdir");
  mjs_setString(mesh->file, "cube.obj");
  mjs_setName(mesh->element, "cube");
  mjs_setString(geom->meshname, "cube");
  mjs_setFrame(geom->element, frame);
  geom->type = mjGEOM_MESH;

  // parent attaching the child
  mjSpec* spec = mj_makeSpec();
  mjs_setDeepCopy(spec, true);
  mjs_setString(spec->compiler.meshdir, "asset");
  mjs_attach(mjs_findBody(spec, "world")->element, frame->element, "_", "");
  mjModel* model = mj_compile(spec, vfs.get());
  EXPECT_THAT(model, NotNull());

  mj_deleteSpec(child);
  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, NestedMeshDir) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";

  static constexpr char child_xml[] = R"(
  <mujoco>
    <compiler meshdir="child_meshdir"/>

    <asset>
      <mesh name="m" file="child_mesh.obj"/>
    </asset>

    <worldbody>
      <body name="child">
        <geom type="mesh" mesh="m"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char parent_xml[] = R"(
  <mujoco>
    <compiler meshdir="parent_meshdir"/>

    <asset>
      <mesh name="m" file="parent_mesh.obj"/>
      <model name="child" file="child.xml"/>
    </asset>

    <worldbody>
      <body name="parent">
        <geom type="mesh" mesh="m"/>
        <attach model="child" body="child" prefix="child_"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char grandparent_xml[] = R"(
  <mujoco>
    <compiler meshdir="grandparent_meshdir"/>

    <asset>
      <mesh name="m" file="grandparent_mesh.obj"/>
      <model name="parent" file="parent.xml"/>
    </asset>

    <worldbody>
      <geom type="mesh" mesh="m"/>
      <attach model="parent" body="parent" prefix="parent_"/>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child_meshdir/child_mesh.obj", cube,
                  sizeof(cube));
  mj_addBufferVFS(vfs.get(), "child.xml", child_xml, sizeof(child_xml));
  mj_addBufferVFS(vfs.get(), "parent_meshdir/parent_mesh.obj", cube,
                  sizeof(cube));
  mj_addBufferVFS(vfs.get(), "parent.xml", parent_xml, sizeof(parent_xml));
  mj_addBufferVFS(vfs.get(), "grandparent_meshdir/grandparent_mesh.obj", cube,
                  sizeof(cube));

  std::array<char, 1024> error;
  MjModelPtr child_model =
      LoadModelFromString(child_xml, error.data(), error.size(), vfs.get());
  EXPECT_THAT(child_model.get(), NotNull()) << error.data();

  MjModelPtr parent_model =
      LoadModelFromString(parent_xml, error.data(), error.size(), vfs.get());
  EXPECT_THAT(parent_model.get(), NotNull()) << error.data();

  MjModelPtr grandparent_model = LoadModelFromString(
      grandparent_xml, error.data(), error.size(), vfs.get());
  EXPECT_THAT(grandparent_model.get(), NotNull()) << error.data();

  mj_deleteVFS(vfs.get());
}

// ----------------------------- test paths ------------------------------------

static constexpr char kCubeObj[] = R"(
v -1 -1  1
v  1 -1  1
v -1  1  1
v  1  1  1
v -1  1 -1
v  1  1 -1
v -1 -1 -1
v  1 -1 -1)";

// the path stored for an asset, "" if none
static std::string PathAt(const mjModel* m, int pathadr) {
  return pathadr < 0 ? "" : std::string(m->paths + pathadr);
}

// normalized filesystem path for platform-independent comparison
static std::filesystem::path NormalizedPath(const std::string& path) {
  return std::filesystem::path(path).lexically_normal();
}

// whether a path stored in mjModel opens on its own, from any directory
static bool Opens(const std::string& path) {
  std::array<char, 1024> error;
  mjResource* resource =
      mju_openResource("", path.c_str(), nullptr, error.data(), error.size());
  if (!resource) return false;
  mju_closeResource(resource);
  return true;
}

// write content to dir/name, creating dir
static void WriteFile(const std::filesystem::path& dir, const std::string& name,
                      std::string_view content) {
  std::filesystem::create_directories(dir);
  std::ofstream((dir / name).string(), std::ios::binary) << content;
}

// the content of a file under test/
static std::string ReadTestData(const std::string& path) {
  std::ifstream file(GetTestDataFilePath(path), std::ios::binary);
  return std::string((std::istreambuf_iterator<char>(file)), {});
}

// a child model with a mesh, a height field and a texture from files, under
// its own asset directories
static std::string ChildXml(const std::string& meshdir,
                            const std::string& texturedir) {
  return absl::StrFormat(R"(
  <mujoco>
    <compiler meshdir="%s" texturedir="%s"/>
    <asset>
      <mesh name="cube" file="cube.obj"/>
      <hfield name="terrain" file="hfield.png" size="1 1 .1 .1"/>
      <texture name="tex" type="2d" file="hfield.png"/>
      <material name="mat" texture="tex"/>
    </asset>
    <worldbody>
      <body name="box">
        <geom type="mesh" mesh="cube" material="mat"/>
      </body>
    </worldbody>
  </mujoco>
  )",
                         meshdir, texturedir);
}

// paths in mjModel are resolved against the modelfiledir and the asset
// directories of the spec that owns each asset, so that two attached models
// with different directories keep their own files
TEST_F(MujocoTest, PathsResolvedPerAttachedModel) {
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <asset>
      <model name="a" file="a/a.xml"/>
      <model name="b" file="b/b.xml"/>
    </asset>
    <worldbody>
      <attach model="a" body="box" prefix="a_"/>
      <attach model="b" body="box" prefix="b_"/>
    </worldbody>
  </mujoco>
  )";

  // the two children in sibling directories with different asset directories
  std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "resolved_paths";
  std::filesystem::remove_all(root);
  std::string png = ReadTestData("user/testdata/hfield.png");
  WriteFile(root, "parent.xml", parent_xml);
  WriteFile(root / "a", "a.xml", ChildXml("meshes", "textures"));
  WriteFile(root / "a" / "meshes", "cube.obj", kCubeObj);
  WriteFile(root / "a" / "meshes", "hfield.png", png);
  WriteFile(root / "a" / "textures", "hfield.png", png);
  WriteFile(root / "b", "b.xml", ChildXml("geom", "img"));
  WriteFile(root / "b" / "geom", "cube.obj", kCubeObj);
  WriteFile(root / "b" / "geom", "hfield.png", png);
  WriteFile(root / "b" / "img", "hfield.png", png);

  std::array<char, 1024> error;
  mjModel* m = mj_loadXML((root / "parent.xml").string().c_str(), nullptr,
                          error.data(), error.size());
  ASSERT_THAT(m, NotNull()) << error.data();
  ASSERT_EQ(m->nmesh, 2);
  ASSERT_EQ(m->nhfield, 2);
  ASSERT_EQ(m->ntex, 2);

  // each asset's path is its own file, which opens from any directory
  auto expected = [&](mjtObj type, int i, const std::string& file) {
    bool a = mj_id2name(m, type, i)[0] == 'a';
    std::string dir = a ? "a/" : "b/";
    bool mesh = type != mjOBJ_TEXTURE;
    dir += mesh ? (a ? "meshes/" : "geom/") : (a ? "textures/" : "img/");
    return (root / (dir + file)).lexically_normal();
  };
  for (int i = 0; i < 2; i++) {
    EXPECT_EQ(NormalizedPath(PathAt(m, m->mesh_pathadr[i])),
              expected(mjOBJ_MESH, i, "cube.obj"));
    EXPECT_EQ(NormalizedPath(PathAt(m, m->hfield_pathadr[i])),
              expected(mjOBJ_HFIELD, i, "hfield.png"));
    EXPECT_EQ(NormalizedPath(PathAt(m, m->tex_pathadr[i])),
              expected(mjOBJ_TEXTURE, i, "hfield.png"));
    EXPECT_TRUE(Opens(PathAt(m, m->mesh_pathadr[i])));
    EXPECT_TRUE(Opens(PathAt(m, m->hfield_pathadr[i])));
    EXPECT_TRUE(Opens(PathAt(m, m->tex_pathadr[i])));
  }
  EXPECT_NE(PathAt(m, m->mesh_pathadr[0]), PathAt(m, m->mesh_pathadr[1]));

  mj_deleteModel(m);
  std::filesystem::remove_all(root);
}

// the resource provider of the "cubeobj" scheme serves a cube
static int OpenCube(mjResource* resource) { return 1; }
static int ReadCube(mjResource* resource, const void** buffer) {
  *buffer = kCubeObj;
  return sizeof(kCubeObj) - 1;
}
static void CloseCube(mjResource* resource) {}

// an absolute path and a URI with a scheme are stored as authored, a relative
// one under modelfiledir and meshdir
TEST_F(MujocoTest, PathsAbsoluteAndScheme) {
  mjpResourceProvider provider = {
      .prefix = "cubeobj",
      .open = OpenCube,
      .read = ReadCube,
      .close = CloseCube,
  };
  ASSERT_GT(mjp_registerProvider(&provider), 0);

  std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "absolute_paths";
  std::filesystem::remove_all(root);
  WriteFile(root / "elsewhere", "cube.obj", kCubeObj);
  WriteFile(root / "meshes" / "sub", "cube.obj", kCubeObj);
  std::string absolute =
      (root / "elsewhere" / "cube.obj").lexically_normal().string();
  std::string xml = absl::StrFormat(R"(
  <mujoco>
    <compiler meshdir="meshes"/>
    <asset>
      <mesh name="absolute" file="%s"/>
      <mesh name="scheme" file="cubeobj://assets/cube.obj"/>
      <mesh name="relative" file="sub/cube.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="absolute"/>
      <geom type="mesh" mesh="scheme"/>
      <geom type="mesh" mesh="relative"/>
    </worldbody>
  </mujoco>
  )",
                                    absolute);

  std::array<char, 1024> error;
  mjSpec* spec =
      mj_parseXMLString(xml.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjs_setString(spec->modelfiledir, root.string().c_str());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  EXPECT_EQ(NormalizedPath(PathAt(m, m->mesh_pathadr[0])),
            NormalizedPath(absolute));
  EXPECT_EQ(PathAt(m, m->mesh_pathadr[1]), "cubeobj://assets/cube.obj");
  EXPECT_EQ(NormalizedPath(PathAt(m, m->mesh_pathadr[2])),
            (root / "meshes" / "sub" / "cube.obj").lexically_normal());
  for (int i = 0; i < 3; i++) {
    EXPECT_TRUE(Opens(PathAt(m, m->mesh_pathadr[i])));
  }

  mj_deleteModel(m);
  mj_deleteSpec(spec);
  std::filesystem::remove_all(root);
}

// strippath drops the directories of the file before it is resolved, and the
// spec keeps the file as authored
TEST_F(MujocoTest, PathsStripped) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler meshdir="meshes" strippath="true"/>
    <asset>
      <mesh name="cube" file="../ignored/cube.obj"/>
      <skin name="skin" file="skins/cube.skn"/>
    </asset>
    <worldbody>
      <body name="box">
        <geom type="mesh" mesh="cube"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "stripped_paths";
  std::filesystem::remove_all(root);
  WriteFile(root / "meshes", "cube.obj", kCubeObj);
  WriteFile(root / "meshes", "cube.skn",
            ReadTestData("user/testdata/cube.skn"));

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjs_setString(spec->modelfiledir, root.string().c_str());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  EXPECT_EQ(NormalizedPath(PathAt(m, m->mesh_pathadr[0])),
            (root / "meshes" / "cube.obj").lexically_normal());
  EXPECT_EQ(NormalizedPath(PathAt(m, m->skin_pathadr[0])),
            (root / "meshes" / "cube.skn").lexically_normal());
  EXPECT_TRUE(Opens(PathAt(m, m->mesh_pathadr[0])));
  EXPECT_TRUE(Opens(PathAt(m, m->skin_pathadr[0])));
  EXPECT_EQ(mjs_getString(
                mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "cube"))->file),
            std::string("../ignored/cube.obj"));
  EXPECT_EQ(mjs_getString(
                mjs_asSkin(mjs_findElement(spec, mjOBJ_SKIN, "skin"))->file),
            std::string("skins/cube.skn"));

  mj_deleteModel(m);
  mj_deleteSpec(spec);
  std::filesystem::remove_all(root);
}

// the content type of a texture is stored after the paths when authored, and
// is -1 otherwise, for embedded textures too; it survives an MJB round trip
TEST_F(MujocoTest, TextureContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="typed" type="2d" builtin="checker" width="4" height="4"
        content_type="image/png"/>
      <texture name="untyped" type="2d" builtin="checker" width="4" height="4"/>
    </asset>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m, NotNull()) << error.data();
  ASSERT_EQ(m->ntex, 2);

  // no files, so paths holds only the content type
  const mjtSize type_size = sizeof("image/png");
  EXPECT_EQ(m->tex_pathadr[0], -1);
  EXPECT_EQ(m->tex_pathadr[1], -1);
  EXPECT_EQ(PathAt(m.get(), m->tex_contenttypeadr[0]), "image/png");
  EXPECT_EQ(m->tex_contenttypeadr[1], -1);
  EXPECT_EQ(m->npaths, type_size);

  // round trip through MJB
  std::vector<char> mjb(mj_sizeModel(m.get()));
  mj_saveModel(m.get(), nullptr, mjb.data(), mjb.size());
  MjModelPtr loaded(mj_loadModelBuffer(mjb.data(), mjb.size()));
  ASSERT_THAT(loaded, NotNull());
  EXPECT_EQ(loaded->npaths, m->npaths);
  EXPECT_EQ(loaded->tex_contenttypeadr[0], m->tex_contenttypeadr[0]);
  EXPECT_EQ(loaded->tex_contenttypeadr[1], -1);
  EXPECT_EQ(PathAt(loaded.get(), loaded->tex_contenttypeadr[0]), "image/png");
}

// a model without content types stores only the paths, and attaching keeps the
// content type of every texture
TEST_F(MujocoTest, TextureContentTypeAttached) {
  static constexpr char plain_xml[] = R"(
  <mujoco>
    <asset>
      <texture name="plain" type="2d" file="hfield.png"/>
    </asset>
  </mujoco>
  )";
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <asset>
      <model name="a" file="child.xml"/>
      <texture name="plain" type="2d" file="hfield.png"/>
    </asset>
    <worldbody>
      <attach model="a" body="box" prefix="a_"/>
      <attach model="a" body="box" prefix="b_"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char child_xml[] = R"(
  <mujoco>
    <asset>
      <texture name="tex" type="2d" file="hfield.png" content_type="image/png"/>
      <material name="mat" texture="tex"/>
    </asset>
    <worldbody>
      <body name="box">
        <geom size="1" material="mat"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::filesystem::path root =
      std::filesystem::path(testing::TempDir()) / "content_types";
  std::filesystem::remove_all(root);
  WriteFile(root, "plain.xml", plain_xml);
  WriteFile(root, "parent.xml", parent_xml);
  WriteFile(root, "child.xml", child_xml);
  WriteFile(root, "hfield.png", ReadTestData("user/testdata/hfield.png"));

  // without content types, paths holds only the paths
  std::array<char, 1024> error;
  mjModel* m = mj_loadXML((root / "plain.xml").string().c_str(), nullptr,
                          error.data(), error.size());
  ASSERT_THAT(m, NotNull()) << error.data();
  ASSERT_EQ(m->ntex, 1);
  std::string path = PathAt(m, m->tex_pathadr[0]);
  const mjtSize path_size = path.size() + 1;
  const mjtSize type_size = sizeof("image/png");
  EXPECT_EQ(m->tex_contenttypeadr[0], -1);
  EXPECT_EQ(m->npaths, path_size);
  mj_deleteModel(m);

  // the attached textures keep their content type, the parent's has none
  m = mj_loadXML((root / "parent.xml").string().c_str(), nullptr, error.data(),
                 error.size());
  ASSERT_THAT(m, NotNull()) << error.data();
  ASSERT_EQ(m->ntex, 3);
  int plain = mj_name2id(m, mjOBJ_TEXTURE, "plain");
  ASSERT_GE(plain, 0);
  for (int i = 0; i < m->ntex; i++) {
    EXPECT_EQ(PathAt(m, m->tex_pathadr[i]), path);
    EXPECT_EQ(PathAt(m, m->tex_contenttypeadr[i]),
              i == plain ? "" : "image/png");
  }
  EXPECT_EQ(m->npaths, 3 * path_size + 2 * type_size);

  mj_deleteModel(m);
  std::filesystem::remove_all(root);
}

TEST_F(MujocoTest, ConvertSpringdamper) {
  static constexpr char xml[] = R"(
    <mujoco>
    <worldbody>
      <body>
        <joint axis="0 1 0" springdamper="1 1"/>
        <geom size="0.2 0.2 0.2" type="box"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << err.data();
  std::array<char, 1024> str;
  mj_saveXMLString(spec, str.data(), str.size(), err.data(), err.size());
  EXPECT_THAT(str.data(), HasSubstr("damping"));
  EXPECT_THAT(str.data(), HasSubstr("stiffness"));
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// ------------- test history buffer computation -------------------------------

using DelayBufferTest = MujocoTest;

TEST_F(DelayBufferTest, ActuatorDelayBufferSizes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1"/>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt1"/>
        <joint name="jnt2" axis="1 0 0"/>
        <joint name="jnt3" axis="0 1 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt1"/>
      <motor joint="jnt2" delay="3" nsample="3"/>
      <motor joint="jnt3" delay="10" nsample="10"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr m = LoadModelFromString(xml);
  ASSERT_THAT(m.get(), NotNull());
  ASSERT_EQ(m->nu, 3);

  // nhistory = (2+2*3) + (2+2*10) = 8 + 22 = 30
  EXPECT_EQ(m->nhistory, 30);

  // verify per-actuator delay and addresses
  EXPECT_EQ(m->actuator_history[0], 0);
  EXPECT_EQ(m->actuator_history[2], 3);
  EXPECT_EQ(m->actuator_history[4], 10);
  EXPECT_EQ(m->actuator_historyadr[0], -1);
  EXPECT_EQ(m->actuator_historyadr[1], 0);
  EXPECT_EQ(m->actuator_historyadr[2], 8);
}

// ------------------------------ frame scale ----------------------------------

using FrameScaleTest = MujocoTest;

// a pendulum in a frame, with the given frame attributes
static std::string Pendulum(const char* frame) {
  return absl::StrFormat(R"(
  <mujoco>
    <worldbody>
      <frame name="frame" %s>
        <body name="arm" pos="0 0 1">
          <joint name="hinge" axis="0 1 0" damping="0.1" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/>
          <site name="tip" pos="0 0 -0.5" size="0.01"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )",
                         frame);
}

// a unit scale compiles to the same model as no scale
TEST_F(FrameScaleTest, IdentityScale) {
  MjModelPtr m = LoadModelFromString(Pendulum(""));
  ASSERT_THAT(m.get(), NotNull());
  for (const char* frame :
       {R"(scale="1")", R"(scale="1" scaling="geometry")"}) {
    MjModelPtr m1 = LoadModelFromString(Pendulum(frame));
    ASSERT_THAT(m1.get(), NotNull());
    std::string field;
    EXPECT_EQ(CompareModel(m.get(), m1.get(), field), 0) << field;
  }
}

// the compiled values of the pendulum of the design, under both policies
TEST_F(FrameScaleTest, Pendulum) {
  MjModelPtr m = LoadModelFromString(Pendulum(""));
  MjModelPtr similar = LoadModelFromString(Pendulum(R"(scale="2")"));
  MjModelPtr geometry =
      LoadModelFromString(Pendulum(R"(scale="2" scaling="geometry")"));
  ASSERT_THAT(m.get(), NotNull());
  ASSERT_THAT(similar.get(), NotNull());
  ASSERT_THAT(geometry.get(), NotNull());

  const mjtNum tol = MjTol(1e-12, 1e-6);
  for (const mjModel* scaled : {similar.get(), geometry.get()}) {
    EXPECT_EQ(scaled->body_pos[5], 2);
    EXPECT_NEAR(scaled->geom_size[0], 0.1, tol);
    EXPECT_EQ(scaled->site_pos[2], -1);
    EXPECT_NEAR(scaled->body_mass[1], 8 * m->body_mass[1], 8 * tol);
    for (int i = 0; i < 3; i++) {
      EXPECT_NEAR(scaled->body_inertia[3 + i], 32 * m->body_inertia[3 + i],
                  32 * tol);
    }
  }

  // similar: hinge damping and armature are M L^2 T^-1 and M L^2
  EXPECT_NEAR(similar->dof_damping[0], 3.2, tol);
  EXPECT_NEAR(similar->dof_armature[0], 0.32, tol);

  // geometry: forces are unchanged
  EXPECT_EQ(geometry->dof_damping[0], m->dof_damping[0]);
  EXPECT_EQ(geometry->dof_armature[0], m->dof_armature[0]);
}

// a frame scales its contents about its own origin, and nested scales compose
TEST_F(FrameScaleTest, NestedTranslatedFrames) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame pos="1 0 0" euler="0 0 90" scale="2">
        <frame pos="0 1 0" scale="3">
          <geom pos="0 0 1" size="0.1"/>
        </frame>
        <body pos="1 0 0">
          <geom size="0.1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_THAT(GetRow(m->geom_pos, 3, 0),
              Pointwise(DoubleNear(tol), {-1, 0, 6}));
  EXPECT_NEAR(m->geom_size[0], 0.6, tol);
  EXPECT_THAT(GetRow(m->body_pos, 3, 1), Pointwise(DoubleNear(tol), {1, 2, 0}));
  EXPECT_NEAR(m->geom_size[3], 0.2, tol);
}

// mixed policies compose: similar 2 around geometry 0.5 keeps the shape and
// inertia and multiplies a hinge stiffness by 2^2 * 2^3
TEST_F(FrameScaleTest, MixedPolicies) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="outer">
        <frame name="inner" scaling="geometry">
          <body>
            <joint stiffness="1" damping="1"/>
            <geom pos="0 0 1" size="0.1"/>
          </body>
        </frame>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  mjs_findFrame(spec, "outer")->scale[0] = 2;
  mjs_findFrame(spec, "outer")->scale[1] = 2;
  mjs_findFrame(spec, "outer")->scale[2] = 2;
  mjs_findFrame(spec, "inner")->scale[0] = 0.5;
  mjs_findFrame(spec, "inner")->scale[1] = 0.5;
  mjs_findFrame(spec, "inner")->scale[2] = 0.5;
  mjModel* scaled = mj_compile(spec, nullptr);
  ASSERT_THAT(scaled, NotNull()) << mjs_getError(spec);

  EXPECT_EQ(scaled->geom_size[0], m->geom_size[0]);
  EXPECT_EQ(scaled->body_mass[1], m->body_mass[1]);
  EXPECT_EQ(scaled->jnt_stiffness[0], 32);
  EXPECT_EQ(scaled->dof_damping[0], 32);

  mj_deleteModel(scaled);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// masses follow massscale, by default the ratio of volumes: inertias take its
// factor times the square of the scale, and under similar forces follow it
TEST_F(FrameScaleTest, MassScale) {
  MjModelPtr m = LoadModelFromString(Pendulum(""));
  ASSERT_THAT(m.get(), NotNull());
  const mjtNum tol = MjTol(1e-12, 1e-6);
  struct Case {
    const char* frame;
    double length, mass, force;  // factors of lengths, masses, and damping
  };
  const Case kCases[] = {
      {R"(scale="2" scaling="geometry" massscale="1")", 2, 1, 1},
      {R"(massscale="2" scaling="geometry")", 1, 2, 1},
      {R"(scale="2" massscale="1")", 2, 1, 4},
      {R"(scale="2" massscale="2")", 2, 2, 8},
  };
  for (const Case& c : kCases) {
    std::array<char, 1024> error;
    MjModelPtr s =
        LoadModelFromString(Pendulum(c.frame), error.data(), error.size());
    ASSERT_THAT(s.get(), NotNull()) << error.data();
    EXPECT_NEAR(s->geom_size[0], c.length * m->geom_size[0], tol) << c.frame;
    EXPECT_NEAR(s->body_mass[1], c.mass * m->body_mass[1], c.mass * tol)
        << c.frame;
    for (int i = 0; i < 3; i++) {
      double f = c.mass * c.length * c.length;
      EXPECT_NEAR(s->body_inertia[3 + i], f * m->body_inertia[3 + i], f * tol)
          << c.frame;
    }
    EXPECT_NEAR(s->dof_damping[0], c.force * m->dof_damping[0], c.force * tol)
        << c.frame;
  }
}

// a body with an explicit inertial and one with a geom of given mass, in a
// frame of the given scale around a frame of the given mass factor
static std::string NestedMass(const char* scale, const char* massscale) {
  return absl::StrFormat(R"(
  <mujoco>
    <worldbody>
      <frame scale="%s" scaling="geometry">
        <frame massscale="%s" scaling="geometry">
          <body>
            <freejoint/>
            <inertial pos="0 0 0" mass="2" diaginertia="1 1 1"/>
          </body>
          <body>
            <freejoint/>
            <geom size="0.1" mass="3"/>
          </body>
        </frame>
      </frame>
    </worldbody>
  </mujoco>
  )",
                         scale, massscale);
}

// mass factors compose like scales: massscale is relative to the enclosing
// frame, and explicit inertials and geom masses follow it
TEST_F(FrameScaleTest, NestedMassScale) {
  MjModelPtr m = LoadModelFromString(NestedMass("1", "0"));
  MjModelPtr s = LoadModelFromString(NestedMass("2", "0.5"));
  ASSERT_THAT(m.get(), NotNull());
  ASSERT_THAT(s.get(), NotNull());
  const mjtNum tol = MjTol(1e-12, 1e-6);
  for (int b = 1; b < 3; b++) {
    EXPECT_NEAR(s->body_mass[b], 4 * m->body_mass[b], tol);
    EXPECT_NEAR(s->body_inertia[3 * b], 16 * m->body_inertia[3 * b], 16 * tol);
  }
}

// compilation starts from the spec: compiling at 2, 0.5 and 2 again gives the
// same model the first and the last time
TEST_F(FrameScaleTest, NoAccumulation) {
  mjSpec* spec = mj_parseXMLString(Pendulum("").c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjsFrame* frame = mjs_findFrame(spec, "frame");
  std::vector<mjModel*> models;
  for (double s : {2.0, 0.5, 2.0}) {
    frame->scale[0] = frame->scale[1] = frame->scale[2] = s;
    models.push_back(mj_compile(spec, nullptr));
    ASSERT_THAT(models.back(), NotNull()) << mjs_getError(spec);
  }
  std::string field;
  EXPECT_EQ(CompareModel(models[0], models[2], field), 0) << field;
  EXPECT_GT(CompareModel(models[0], models[1], field), 0);
  for (mjModel* m : models) mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// saved compiled, the scale is baked into the values, which reload to the same
// model
TEST_F(FrameScaleTest, SavesCompiled) {
  mjSpec* spec = mj_parseXMLString(Pendulum(R"(scale="1.5")").c_str(), nullptr,
                                   nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, Not(HasSubstr("scale")));
  MjModelPtr reloaded = LoadModelFromString(saved);
  ASSERT_THAT(reloaded.get(), NotNull()) << saved;
  std::string field;
  EXPECT_LE(CompareModel(m, reloaded.get(), field), MjTol(1e-12, 1e-6))
      << field;
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a mesh used at three scales: by geoms of bodies in frames, and by a site
static std::string Tetra(double s1, double s2) {
  return absl::StrFormat(R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <body name="b0">
        <freejoint/>
        <geom type="mesh" mesh="tetra"/>
      </body>
      <frame name="f1" scale="%.17g">
        <body name="b1" pos="2 0 0">
          <freejoint/>
          <geom type="mesh" mesh="tetra"/>
        </body>
        <site type="mesh" mesh="tetra"/>
      </frame>
      <frame name="f2" scale="%.17g">
        <body name="b2" pos="4 0 0">
          <freejoint/>
          <geom type="mesh" mesh="tetra"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )",
                         s1, s2);
}

// a mesh used at several scales compiles to an instance per scale, each as the
// asset compiled with its scale multiplied by that of its use
TEST_F(FrameScaleTest, MeshInstances) {
  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(Tetra(1.5, 0.5), error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nmesh, 3);
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_MESH, 0), "tetra");
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_MESH, 1), "tetra@1");
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_MESH, 2), "tetra@2");
  EXPECT_THAT(AsVector(m->geom_dataid, 3), ElementsAre(0, 1, 2));
  EXPECT_EQ(m->site_dataid[0], 1);
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_NEAR(m->body_mass[2] / m->body_mass[1], 3.375, tol);
  EXPECT_NEAR(m->body_mass[3] / m->body_mass[1], 0.125, tol);

  static constexpr char reference[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="tetra@1" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
            scale="1.5 1.5 1.5"/>
      <mesh name="tetra@2" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
            scale="0.5 0.5 0.5"/>
    </asset>
    <worldbody>
      <body name="b0">
        <freejoint/>
        <geom type="mesh" mesh="tetra"/>
      </body>
      <body name="b1" pos="3 0 0">
        <freejoint/>
        <geom type="mesh" mesh="tetra@1"/>
      </body>
      <site type="mesh" mesh="tetra@1"/>
      <body name="b2" pos="2 0 0">
        <freejoint/>
        <geom type="mesh" mesh="tetra@2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr ref = LoadModelFromString(reference, error.data(), error.size());
  ASSERT_THAT(ref.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nmeshvert, ref->nmeshvert);
  auto same = [&](const auto* a, const auto* b, int n) {
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(a[i], b[i], tol * std::max(1.0, std::abs((double)b[i]))) << i;
    }
  };
  same(m->mesh_vert, ref->mesh_vert, 3 * m->nmeshvert);
  same(m->mesh_scale, ref->mesh_scale, 3 * m->nmesh);
  same(m->mesh_pos, ref->mesh_pos, 3 * m->nmesh);
  same(m->geom_size, ref->geom_size, 3 * m->ngeom);
  same(m->geom_pos, ref->geom_pos, 3 * m->ngeom);
  same(m->geom_rbound, ref->geom_rbound, m->ngeom);
  same(m->site_size, ref->site_size, 3 * m->nsite);
  same(m->body_pos, ref->body_pos, 3 * m->nbody);
  same(m->body_ipos, ref->body_ipos, 3 * m->nbody);
  same(m->body_mass, ref->body_mass, m->nbody);
  same(m->body_inertia, ref->body_inertia, 3 * m->nbody);
}

// the instances are not in the spec, which saves without them; saved compiled,
// they are assets of their own
TEST_F(FrameScaleTest, MeshInstancesSave) {
  mjSpec* spec =
      mj_parseXMLString(Tetra(1.5, 0.5).c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "tetra@1"), IsNull());

  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr("tetra@2"));
  MjModelPtr reloaded = LoadModelFromString(saved);
  ASSERT_THAT(reloaded.get(), NotNull()) << saved;
  std::string field;
  EXPECT_LE(CompareModel(m, reloaded.get(), field), MjTol(1e-12, 1e-6))
      << field;

  spec->compiler.savecompiled = 0;
  std::string written = SaveAndReadXml(spec);
  EXPECT_THAT(written, Not(HasSubstr("@")));
  MjModelPtr rewritten = LoadModelFromString(written);
  ASSERT_THAT(rewritten.get(), NotNull()) << written;
  EXPECT_EQ(CompareModel(m, rewritten.get(), field), 0) << field;

  // a copy of the compiled spec keeps the instances
  mjSpec* copy = mj_copySpec(spec);
  copy->compiler.savecompiled = 1;
  EXPECT_EQ(SaveAndReadXml(copy), saved);
  mj_deleteSpec(copy);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// instances are made again at each compilation, from the spec
TEST_F(FrameScaleTest, MeshInstancesRecompile) {
  mjSpec* spec = mj_parseXMLString(Tetra(1, 1).c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjsFrame* frame = mjs_findFrame(spec, "f1");
  std::vector<mjModel*> models;
  for (double s : {2.0, 0.5, 1.0, 2.0}) {
    frame->scale[0] = frame->scale[1] = frame->scale[2] = s;
    models.push_back(mj_compile(spec, nullptr));
    ASSERT_THAT(models.back(), NotNull()) << mjs_getError(spec);
  }
  EXPECT_EQ(models[0]->nmesh, 2);
  EXPECT_EQ(models[2]->nmesh, 1);
  std::string field;
  EXPECT_EQ(CompareModel(models[0], models[3], field), 0) << field;
  MjModelPtr fresh = LoadModelFromString(Tetra(2, 1));
  ASSERT_THAT(fresh.get(), NotNull());
  EXPECT_EQ(CompareModel(models[3], fresh.get(), field), 0) << field;
  for (mjModel* m : models) mj_deleteModel(m);
  mj_deleteSpec(spec);

  // the first use of the mesh changes scale
  auto xml = [](const char* scale) {
    return std::string(R"(
    <mujoco>
      <asset>
        <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      </asset>
      <worldbody>
        <frame name="f" scale=")") +
           scale + R"(">
          <geom type="mesh" mesh="tetra"/>
        </frame>
      </worldbody>
    </mujoco>
    )";
  };
  spec = mj_parseXMLString(xml("1").c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  frame = mjs_findFrame(spec, "f");
  frame->scale[0] = frame->scale[1] = frame->scale[2] = 2;
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  MjModelPtr fresh2 = LoadModelFromString(xml("2"));
  ASSERT_THAT(fresh2.get(), NotNull());
  EXPECT_EQ(CompareModel(m2, fresh2.get(), field), 0) << field;
  EXPECT_EQ(m2->mesh_scale[0], 2);
  mj_deleteModel(m2);
  mj_deleteModel(m1);
  mj_deleteSpec(spec);
}

// the name of an instance cannot be that of another asset
TEST_F(FrameScaleTest, MeshInstanceName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="tetra@1" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="tetra"/>
      <geom type="mesh" mesh="tetra@1"/>
      <frame scale="2">
        <geom type="mesh" mesh="tetra"/>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("'tetra@1' of an instance of this asset is taken"));
}

// discarding visual geoms compiles the tree first, whose instances are kept for
// those which remain
TEST_F(FrameScaleTest, MeshInstancesDiscardVisual) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler discardvisual="true"/>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="visual" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="tetra"/>
      <frame scale="2">
        <body>
          <freejoint/>
          <geom type="mesh" mesh="tetra"/>
          <geom type="mesh" mesh="visual" contype="0" conaffinity="0"/>
          <frame scale="1.5">
            <geom type="mesh" mesh="tetra" contype="0" conaffinity="0"/>
          </frame>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nmesh, 2);
  EXPECT_THAT(AsVector(m->geom_dataid, m->ngeom), ElementsAre(0, 1));
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  std::string field;
  EXPECT_LE(CompareModel(m, again, field), MjTol(1e-12, 1e-6)) << field;
  mj_deleteModel(again);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// deleting a mesh deletes its instances, so that the spec can be copied
TEST_F(FrameScaleTest, MeshInstancesDelete) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh name="other" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <geom name="small" type="mesh" mesh="tetra"/>
      <frame scale="2">
        <geom name="large" type="mesh" mesh="tetra"/>
      </frame>
      <geom type="mesh" mesh="other"/>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nmesh, 3);
  for (const char* name : {"small", "large"}) {
    ASSERT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_GEOM, name)), 0);
  }
  ASSERT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_MESH, "tetra")), 0);
  mjSpec* copy = mj_copySpec(spec);
  ASSERT_THAT(copy, NotNull());
  mjModel* m2 = mj_compile(copy, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(copy);
  EXPECT_EQ(m2->nmesh, 1);
  mj_deleteModel(m2);
  mj_deleteSpec(copy);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a flex made from a mesh asset takes the asset at its own scale, also after a
// compilation which scaled the asset for a geom
TEST_F(FrameScaleTest, MeshInstanceMakeFlex) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <frame scale="2">
        <geom type="mesh" mesh="tetra"/>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<mjModel*, 2> m;
  for (int compiled = 0; compiled < 2; compiled++) {
    mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
    ASSERT_THAT(spec, NotNull());
    if (compiled) {
      mjModel* first = mj_compile(spec, nullptr);
      ASSERT_THAT(first, NotNull()) << mjs_getError(spec);
      mj_deleteModel(first);
    }
    mjsFlex* flex =
        mjs_makeFlex(mjs_findBody(spec, "world"), "flex", "mesh", 2, nullptr,
                     nullptr, nullptr, nullptr, nullptr, 0.001, 1, 0, 0, 0, 0,
                     0, nullptr, nullptr, nullptr, "tetra", nullptr);
    ASSERT_THAT(flex, NotNull()) << mjs_getError(spec);
    m[compiled] = mj_compile(spec, nullptr);
    ASSERT_THAT(m[compiled], NotNull()) << mjs_getError(spec);
    mj_deleteSpec(spec);
  }
  std::string field;
  EXPECT_LE(CompareModel(m[0], m[1], field), MjTol(1e-12, 1e-6)) << field;
  mj_deleteModel(m[0]);
  mj_deleteModel(m[1]);
}

// a height field used at two scales; its elevation data is copied back if its
// instances agree
TEST_F(FrameScaleTest, HFieldInstances) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="terrain" nrow="2" ncol="2" elevation="0 1 1 0"
              size="1 1 0.5 0.1"/>
    </asset>
    <worldbody>
      <geom type="hfield" hfield="terrain"/>
      <frame scale="2">
        <geom type="hfield" hfield="terrain" pos="5 0 0"/>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nhfield, 2);
  EXPECT_STREQ(mj_id2name(m, mjOBJ_HFIELD, 1), "terrain@1");
  EXPECT_THAT(AsVector(m->hfield_size + 4, 4), ElementsAre(2, 2, 1, 0.2));
  EXPECT_THAT(AsVector(m->geom_dataid, 2), ElementsAre(0, 1));

  float edited[4] = {0, 0, 1, 1};
  std::copy(edited, edited + 4, m->hfield_data + m->hfield_adr[1]);
  EXPECT_EQ(mj_copyBack(spec, m), 0);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("instances differ"));

  std::copy(edited, edited + 4, m->hfield_data + m->hfield_adr[0]);
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  mjModel* recompiled = mj_compile(spec, nullptr);
  ASSERT_THAT(recompiled, NotNull()) << mjs_getError(spec);
  std::string field;
  EXPECT_EQ(CompareModel(m, recompiled, field), 0) << field;
  mj_deleteModel(recompiled);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// the density of a mesh with shell inertia is a surface density, and the mass
// still follows the volume ratio
TEST_F(FrameScaleTest, ShellMeshMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1" inertia="shell"/>
    </asset>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="mesh" mesh="tetra"/>
      </body>
      <frame scale="2" scaling="geometry">
        <body>
          <freejoint/>
          <geom type="mesh" mesh="tetra"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_NEAR(m->body_mass[2] / m->body_mass[1], 8, MjTol(1e-12, 1e-5));
}

// flexes on bodies in a frame of the given scale: a rope with edge stiffness, a
// rope with an edge equality, and a triangle with its vertices in one body
static std::string Ropes(double scale, const char* scaling) {
  return absl::StrFormat(R"(
  <mujoco>
    <default>
      <joint type="slide" axis="1 0 0"/>
      <geom size="0.01"/>
    </default>
    <worldbody>
      <frame scale="%.17g" scaling="%s">
        <body name="b0" pos="0 0 1"><joint/><geom/></body>
        <body name="b1" pos="0.2 0 1"><joint/><geom/></body>
        <body name="b2" pos="0.4 0 1"><joint/><geom/></body>
        <body name="b3" pos="0 1 1"><joint/><geom/></body>
        <body name="b4" pos="0.2 1 1"><joint/><geom/></body>
        <body name="plate" pos="0 0 2"/>
      </frame>
    </worldbody>
    <deformable>
      <flex name="rope" dim="1" body="b0 b1 b2" element="0 1 1 2" radius="0.02">
        <edge stiffness="10" damping="0.5"/>
        <contact margin="0.01" gap="0.005" solimp="0.9 0.95 0.001 0.5 2"
                 friction="1 0.005 0.0001"/>
        <elasticity damping="0.3"/>
      </flex>
      <flex name="rope2" dim="1" body="b3 b4" element="0 1" radius="0.02"/>
      <flex name="tri" dim="2" body="plate" vertex="0 0 0 0.1 0 0 0 0.1 0"
            element="0 1 2" radius="0.01"/>
    </deformable>
    <equality>
      <flex flex="rope2" solimp="0.9 0.95 0.001 0.5 2"/>
    </equality>
  </mujoco>
  )",
                         scale, scaling);
}

// the properties of a flex take the gauge of its bodies, and the positions of
// its vertices those of their own bodies
TEST_F(FrameScaleTest, Flexes) {
  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(Ropes(1, "similar"), error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-12, 1e-6);
  for (const char* scaling : {"similar", "geometry"}) {
    MjModelPtr s =
        LoadModelFromString(Ropes(2, scaling), error.data(), error.size());
    ASSERT_THAT(s.get(), NotNull()) << error.data();
    const double force = std::string(scaling) == "similar" ? 8 : 1;
    for (int f = 0; f < m->nflex; f++) {
      EXPECT_NEAR(s->flex_radius[f], 2 * m->flex_radius[f], tol);
    }
    EXPECT_NEAR(s->flex_margin[0], 0.02, tol);
    EXPECT_NEAR(s->flex_gap[0], 0.01, tol);
    EXPECT_NEAR(s->flex_solimp[2], 0.002, tol);
    EXPECT_NEAR(s->flex_friction[0], 1, tol);
    EXPECT_NEAR(s->flex_friction[1], 0.01, tol);
    EXPECT_NEAR(s->flex_friction[2], 0.0002, tol);
    EXPECT_NEAR(s->flex_edgestiffness[0], 10 * force, force * tol);
    EXPECT_NEAR(s->flex_edgedamping[0], 0.5 * force, force * tol);
    EXPECT_EQ(s->flex_damping[0], m->flex_damping[0]);
    EXPECT_NEAR(s->eq_solimp[2], 0.002, tol);
    int adr = s->flex_vertadr[2];
    for (int i = 0; i < 9; i++) {
      EXPECT_NEAR(s->flex_vert[3 * adr + i], 2 * m->flex_vert[3 * adr + i],
                  tol);
    }
    for (int i = 0; i < 3; i++) {
      EXPECT_NEAR(s->flex_size[3 * 2 + i], 2 * m->flex_size[3 * 2 + i], tol);
    }
  }
}

// a cloth in a frame of the given scale, in the given gravity
static std::string Cloth(double scale, double gravity) {
  return absl::StrFormat(R"(
  <mujoco>
    <option gravity="0 0 %.17g"/>
    <worldbody>
      <frame scale="%.17g">
        <body name="holder" pos="0 0 1">
          <flexcomp type="grid" count="4 4 1" spacing="0.1 0.1 0.1"
                    radius="0.01" name="cloth" dim="2" mass="0.5">
            <elasticity young="1e3" poisson="0.2" thickness="0.01"
                        damping="0.01" elastic2d="stretch"/>
            <edge equality="false" damping="0.1"/>
            <contact contype="0" conaffinity="0"/>
            <pin id="0 3"/>
          </flexcomp>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )",
                         gravity, scale);
}

// a scaled cloth moves like the original in proportionally scaled gravity
TEST_F(FrameScaleTest, FlexSimilarity) {
  constexpr double s = 1.7;
  constexpr double g = -9.81;
  std::array<char, 1024> error;
  MjModelPtr m0 =
      LoadModelFromString(Cloth(1, g / s), error.data(), error.size());
  ASSERT_THAT(m0.get(), NotNull()) << error.data();
  MjModelPtr m1 = LoadModelFromString(Cloth(s, g), error.data(), error.size());
  ASSERT_THAT(m1.get(), NotNull()) << error.data();
  ASSERT_GT(m0->nq, 0);
  MjDataPtr d0 = MakeData(m0);
  MjDataPtr d1 = MakeData(m1);
  for (int step = 0; step < 300; step++) {
    mj_step(m0.get(), d0.get());
    mj_step(m1.get(), d1.get());
    for (int i = 0; i < m0->nq; i++) {
      EXPECT_NEAR(d1->qpos[i] / s, d0->qpos[i], MjTol(1e-10, 1e-4))
          << "step " << step << ", qpos " << i;
    }
    if (HasFailure()) break;
  }
  EXPECT_GT(std::abs(d0->qpos[m0->nq - 1]), 0.01);
}

// a skin scales with its bones, which must have the same scale
TEST_F(FrameScaleTest, Skins) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="f" scale="2">
        <body name="b0" pos="0 0 1"/>
        <body name="b1" pos="1 0 1"/>
      </frame>
      <frame name="g">
        <body name="b2" pos="0 1 1"/>
      </frame>
    </worldbody>
    <deformable>
      <skin name="skin" vertex="0 0 1 1 0 1 0 1 1" face="0 1 2" inflate="0.01">
        <bone body="b0" bindpos="0 0 1" bindquat="1 0 0 0" vertid="0 2"
              vertweight="1 1"/>
        <bone body="b1" bindpos="1 0 1" bindquat="1 0 0 0" vertid="1"
              vertweight="1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(AsVector(m->skin_vert, 9),
              ElementsAre(0, 0, 2, 2, 0, 2, 0, 2, 2));
  EXPECT_THAT(AsVector(m->skin_bonebindpos, 6), ElementsAre(0, 0, 2, 2, 0, 2));
  EXPECT_FLOAT_EQ(m->skin_inflate[0], 0.02);

  // saved compiled, the scaled skin reloads to the same model
  std::string saved = SaveAndReadXml(spec);
  MjModelPtr reloaded = LoadModelFromString(saved);
  ASSERT_THAT(reloaded.get(), NotNull()) << saved;
  std::string field;
  EXPECT_LE(CompareModel(m, reloaded.get(), field), MjTol(1e-12, 1e-6))
      << field;
  mj_deleteModel(m);

  // bones of different scales
  mjsFrame* frame = mjs_findFrame(spec, "g");
  frame->scale[0] = frame->scale[1] = frame->scale[2] = 3;
  mjsSkin* skin = mjs_asSkin(mjs_findElement(spec, mjOBJ_SKIN, "skin"));
  mjs_setInStringVec(skin->bodyname, 1, "b2");
  EXPECT_THAT(mj_compile(spec, nullptr), IsNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("must have the same scale"));
  mj_deleteSpec(spec);
}

// a chain of hinge and slide joints, and a free sphere on a plane, all in a
// frame of the given scale, in the given gravity
static std::string Chain(double scale, double gravity) {
  return absl::StrFormat(R"(
  <mujoco>
    <option gravity="0 0 %.17g" tolerance="0" iterations="20"/>
    <worldbody>
      <frame scale="%.17g">
        <geom type="plane" size="5 5 0.1"/>
        <body pos="0 0 1">
          <joint axis="0 1 0" stiffness="2" damping="0.3" armature="0.01"
                 springref="10" range="-30 30"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 -0.5" size="0.05"/>
          <body pos="0.3 0 -0.5">
            <joint type="slide" axis="1 0 0" stiffness="50" damping="1"
                   range="-0.05 0.05" frictionloss="0.1" ref="0.02"/>
            <geom type="box" size="0.1 0.05 0.05" mass="0.5"/>
          </body>
        </body>
        <body pos="1 0 0.3">
          <freejoint/>
          <geom type="sphere" size="0.1" margin="0.01"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )",
                         gravity, scale);
}

// the statement of the design: scaled by s under similar, in gravity g, a model
// moves like the original in gravity g / s, with lengths multiplied by s
TEST_F(FrameScaleTest, Similarity) {
  constexpr double s = 1.7;
  constexpr double g = -9.81;
  MjModelPtr m0 = LoadModelFromString(Chain(1, g / s));
  MjModelPtr m1 = LoadModelFromString(Chain(s, g));
  ASSERT_THAT(m0.get(), NotNull());
  ASSERT_THAT(m1.get(), NotNull());
  MjDataPtr d0 = MakeData(m0);
  MjDataPtr d1 = MakeData(m1);

  // factors of qpos: slide and free positions are lengths
  std::vector<mjtNum> factor(m0->nq, 1);
  factor[1] = s;
  for (int i = 2; i < 5; i++) factor[i] = s;

  bool contact = false;
  for (int step = 0; step < 500; step++) {
    mj_step(m0.get(), d0.get());
    mj_step(m1.get(), d1.get());
    contact |= d0->ncon > 0;
    for (int i = 0; i < m0->nq; i++) {
      EXPECT_NEAR(d1->qpos[i] / factor[i], d0->qpos[i], MjTol(1e-10, 1e-4))
          << "step " << step << ", qpos " << i;
    }
    if (HasFailure()) break;
  }
  EXPECT_TRUE(contact);
}

// a model scaled under similar with a mass factor moves like the original in
// gravity g / s, whatever the factor
TEST_F(FrameScaleTest, MassScaleSimilarity) {
  constexpr double s = 1.7;
  constexpr double g = -9.81;
  MjModelPtr m0 = LoadModelFromString(Chain(1, g / s));
  std::string scaled = Chain(s, g);
  std::string::size_type at = scaled.find("<frame scale=");
  ASSERT_NE(at, std::string::npos);
  scaled.insert(at + 7, R"(massscale="0.3" )");
  MjModelPtr m1 = LoadModelFromString(scaled);
  ASSERT_THAT(m0.get(), NotNull());
  ASSERT_THAT(m1.get(), NotNull()) << scaled;
  MjDataPtr d0 = MakeData(m0);
  MjDataPtr d1 = MakeData(m1);
  std::vector<mjtNum> factor(m0->nq, 1);
  for (int i = 1; i < 5; i++) factor[i] = s;
  for (int step = 0; step < 300; step++) {
    mj_step(m0.get(), d0.get());
    mj_step(m1.get(), d1.get());
    for (int i = 0; i < m0->nq; i++) {
      EXPECT_NEAR(d1->qpos[i] / factor[i], d0->qpos[i], MjTol(1e-10, 1e-4))
          << "step " << step << ", qpos " << i;
    }
    if (HasFailure()) break;
  }
}

// the actuators of the design: a position servo and a motor, on a hinge and a
// slide, under both policies
TEST_F(FrameScaleTest, Actuators) {
  auto model = [](const char* frame) {
    return absl::StrFormat(R"(
    <mujoco>
      <worldbody>
        <frame %s>
          <body pos="0 0 1">
            <joint name="hinge" axis="0 1 0"/>
            <joint name="slide" type="slide" range="0 0.2"/>
            <geom size="0.1"/>
          </body>
        </frame>
      </worldbody>
      <actuator>
        <position joint="hinge" kp="100" kv="10" ctrlrange="-1 1"
                  forcerange="-50 50"/>
        <position joint="slide" kp="300" ctrlrange="0 0.2"
                  forcerange="-40 40"/>
        <motor joint="slide" gear="50" ctrlrange="-1 1"/>
        <motor joint="hinge" gear="1 0 0 1 1 1"/>
      </actuator>
    </mujoco>
    )",
                           frame);
  };
  MjModelPtr similar = LoadModelFromString(model(R"(scale="2")"));
  MjModelPtr geometry =
      LoadModelFromString(model(R"(scale="2" scaling="geometry")"));
  ASSERT_THAT(similar.get(), NotNull());
  ASSERT_THAT(geometry.get(), NotNull());
  const mjtNum tol = MjTol(1e-12, 1e-5);

  // hinge servo: kp and kv are torques, the control an angle
  EXPECT_NEAR(similar->actuator_gainprm[0], 3200, 3200 * tol);
  EXPECT_NEAR(similar->actuator_biasprm[1], -3200, 3200 * tol);
  EXPECT_NEAR(similar->actuator_biasprm[2], -320, 320 * tol);
  EXPECT_NEAR(similar->actuator_forcerange[1], 1600, 1600 * tol);
  EXPECT_EQ(similar->actuator_ctrlrange[1], 1);
  EXPECT_EQ(geometry->actuator_gainprm[0], 100);
  EXPECT_EQ(geometry->actuator_forcerange[1], 50);

  // slide servo: the control is a position, kp a force per length
  EXPECT_NEAR(similar->actuator_ctrlrange[3], 0.4, tol);
  EXPECT_NEAR(similar->actuator_gainprm[mjNGAIN], 2400, 2400 * tol);
  EXPECT_NEAR(similar->actuator_forcerange[3], 640, 640 * tol);
  EXPECT_NEAR(geometry->actuator_ctrlrange[3], 0.4, tol);
  EXPECT_EQ(geometry->actuator_gainprm[mjNGAIN], 300);

  // motor: the control is a command, which the gain absorbs under similar
  EXPECT_EQ(similar->actuator_ctrlrange[5], 1);
  EXPECT_EQ(similar->actuator_gear[12], 50);
  EXPECT_NEAR(similar->actuator_gainprm[2 * mjNGAIN], 16, 16 * tol);
  EXPECT_EQ(geometry->actuator_gainprm[2 * mjNGAIN], 1);

  // the gear of a joint is a scalar, its other entries are not scaled
  EXPECT_THAT(GetRow(similar->actuator_gear, 6, 3),
              ElementsAre(1, 0, 0, 1, 1, 1));
}

// elements outside the tree take the gauge of the elements they refer to: the
// boundary of the design, an arm scaled by 2 in an unscaled world
TEST_F(FrameScaleTest, Boundary) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 0.1"/>
      <site name="hook" pos="0 0 3"/>
      <frame scale="2">
        <body name="arm" pos="0 0 1">
          <joint axis="0 1 0"/>
          <geom name="hand" size="0.1"/>
          <site name="tip" pos="0 0 -0.5"/>
          <site name="tip2" pos="0.1 0 -0.5"/>
        </body>
      </frame>
    </worldbody>
    <contact>
      <pair geom1="hand" geom2="floor" margin="0.01"
            friction="1 1 0.005 0.0001 0.0001"/>
    </contact>
    <tendon>
      <spatial springlength="0.3" stiffness="100">
        <site site="tip"/>
        <site site="tip2"/>
      </spatial>
      <spatial springlength="0.8" stiffness="100">
        <site site="tip"/>
        <site site="hook"/>
      </spatial>
    </tendon>
    <equality>
      <connect body1="arm" body2="world" anchor="0 0 0.2"/>
    </equality>
    <sensor>
      <framepos objtype="site" objname="tip" reftype="site" refname="hook"
                cutoff="4"/>
      <touch site="tip" cutoff="10"/>
      <rangefinder site="tip" cutoff="1"/>
      <distance geom1="hand" geom2="floor" cutoff="1"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-12, 1e-6);

  // a pair takes the geometric mean of the gauges of its geoms
  EXPECT_NEAR(m->pair_margin[0], 0.01 * std::sqrt(2.0), tol);
  EXPECT_NEAR(m->pair_friction[2], 0.005 * std::sqrt(2.0), tol);

  // a tendon takes the gauge of the lowest frame or body containing its path
  EXPECT_NEAR(m->tendon_lengthspring[0], 0.6, tol);
  EXPECT_NEAR(m->tendon_stiffness[0], 800, 800 * tol);
  EXPECT_NEAR(m->tendon_lengthspring[2], 0.8, tol);
  EXPECT_NEAR(m->tendon_stiffness[1], 100, 100 * tol);

  // a connect anchor is in the coordinates of its first body
  EXPECT_NEAR(m->eq_data[2], 0.4, tol);

  // a sensor takes the gauge of its object, not of its reference; a distance
  // between two objects takes the geometric mean of their gauges
  EXPECT_NEAR(m->sensor_cutoff[0], 8, tol);
  EXPECT_NEAR(m->sensor_cutoff[1], 160, 160 * tol);
  EXPECT_NEAR(m->sensor_cutoff[2], 2, tol);
  EXPECT_NEAR(m->sensor_cutoff[3], std::sqrt(2.0), tol);
}

// a model with actuators, tendons, an equality, a contact pair and sensors, all
// in a frame of the given scale, in the given gravity
static std::string Mechanism(double scale, double gravity) {
  return absl::StrFormat(R"(
  <mujoco>
    <option gravity="0 0 %.17g" tolerance="0" iterations="30"/>
    <worldbody>
      <frame scale="%.17g">
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <site name="anchor" pos="0.5 0 1.5"/>
        <body pos="0 0 1">
          <joint name="hinge" axis="0 1 0" damping="0.3" armature="0.01"
                 range="-30 30"/>
          <geom type="capsule" fromto="0 0 0 0.3 0 -0.5" size="0.05"/>
          <site name="elbow" pos="0.3 0 -0.5"/>
          <body pos="0.3 0 -0.5">
            <joint name="slide" type="slide" axis="1 0 0" damping="1"
                   range="-0.05 0.05" frictionloss="0.1"/>
            <geom type="box" size="0.1 0.05 0.05" mass="0.5"/>
          </body>
        </body>
        <body name="link" pos="-1 0 1">
          <joint type="ball" damping="0.1"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 -0.4" size="0.04"/>
        </body>
        <body pos="1 0 0.3">
          <freejoint/>
          <geom name="sphere" type="sphere" size="0.1"/>
        </body>
      </frame>
    </worldbody>
    <contact>
      <pair geom1="sphere" geom2="floor" margin="0.01"/>
    </contact>
    <tendon>
      <fixed stiffness="5" springlength="0.01">
        <joint joint="hinge" coef="0.1"/>
        <joint joint="slide" coef="1"/>
      </fixed>
      <spatial name="bungee" stiffness="20" damping="0.5" range="0.1 2">
        <site site="anchor"/>
        <site site="elbow"/>
      </spatial>
    </tendon>
    <equality>
      <connect body1="link" anchor="0.2 0 -0.4"/>
    </equality>
    <actuator>
      <position joint="slide" kp="300" kv="5" ctrlrange="-0.04 0.04"
                forcerange="-40 40"/>
      <motor joint="hinge" gear="2" ctrlrange="-1 1"/>
      <motor tendon="bungee" ctrlrange="-1 1"/>
    </actuator>
    <sensor>
      <jointpos joint="slide"/>
      <framepos objtype="site" objname="elbow"/>
    </sensor>
  </mujoco>
  )",
                         gravity, scale);
}

// the statement of the design, with the elements outside the tree: controls
// declared as positions are lengths, commands are unchanged
TEST_F(FrameScaleTest, SimilarityWithActuators) {
  constexpr double s = 1.7;
  constexpr double g = -9.81;
  std::array<char, 1024> error;
  MjModelPtr m0 =
      LoadModelFromString(Mechanism(1, g / s), error.data(), error.size());
  ASSERT_THAT(m0.get(), NotNull()) << error.data();
  MjModelPtr m1 = LoadModelFromString(Mechanism(s, g));
  ASSERT_THAT(m1.get(), NotNull());
  MjDataPtr d0 = MakeData(m0);
  MjDataPtr d1 = MakeData(m1);

  // factors: slide, free position; the servo control; slide and site positions
  std::vector<mjtNum> qfactor(m0->nq, 1);
  qfactor[1] = s;
  for (int i = 6; i < 9; i++) qfactor[i] = s;
  const mjtNum ufactor[3] = {s, 1, 1};
  const mjtNum sfactor[4] = {s, s, s, s};

  for (int step = 0; step < 500; step++) {
    for (int i = 0; i < 3; i++) {
      d0->ctrl[i] = 0.03 * std::sin(0.02 * step + i);
      d1->ctrl[i] = ufactor[i] * d0->ctrl[i];
    }
    mj_step(m0.get(), d0.get());
    mj_step(m1.get(), d1.get());
    for (int i = 0; i < m0->nq; i++) {
      EXPECT_NEAR(d1->qpos[i] / qfactor[i], d0->qpos[i], MjTol(1e-10, 1e-4))
          << "step " << step << ", qpos " << i;
    }
    for (int i = 0; i < 4; i++) {
      EXPECT_NEAR(d1->sensordata[i] / sfactor[i], d0->sensordata[i],
                  MjTol(1e-10, 1e-4))
          << "step " << step << ", sensor " << i;
    }
    if (HasFailure()) break;
  }
}

// actuator parameters which the gauge cannot interpret are an error under a
// force gauge, and compile under geometry
TEST_F(FrameScaleTest, UserActuator) {
  auto model = [](const char* scaling) {
    return absl::StrFormat(R"(
    <mujoco>
      <worldbody>
        <frame scale="2" scaling="%s">
          <body>
            <joint name="hinge"/>
            <geom size="0.1"/>
          </body>
        </frame>
      </worldbody>
      <actuator>
        <general joint="hinge" gaintype="user" biastype="user"/>
      </actuator>
    </mujoco>
    )",
                           scaling);
  };
  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(model("similar"), error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("cannot be scaled"));
  m = LoadModelFromString(model("geometry"), error.data(), error.size());
  EXPECT_THAT(m.get(), NotNull()) << error.data();
}

// keyframes are given in the units of the spec: free joints and mocap bodies in
// the world, about the origin of their innermost frame, and slides, activations
// and controls by the gauges of their joints and actuators
TEST_F(FrameScaleTest, Keyframes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame pos="1 0 0" euler="0 0 90" scale="2">
        <frame pos="0 1 0" scale="3">
          <body pos="0 0 1">
            <freejoint name="free"/>
            <geom size="0.1"/>
          </body>
        </frame>
        <body name="mocap" mocap="true" pos="1 0 0">
          <geom size="0.1" contype="0" conaffinity="0"/>
        </body>
        <body pos="0 0 2">
          <joint name="slide" type="slide" axis="1 0 0" ref="0.05"/>
          <geom size="0.1"/>
        </body>
      </frame>
    </worldbody>
    <actuator>
      <general joint="slide" input="vel" dyntype="integrator" gainprm="10"/>
      <pid joint="slide" kp="10" ki="1" slewmax="1" input="pos vel ff"/>
    </actuator>
    <keyframe>
      <key name="authored" qpos="0 0 1 1 0 0 0 0.1" qvel="1 0 0 0 0 1 0.5"
           act="0.1 0.1 0.1" ctrl="0.1 0.1 0.1 0.1" mpos="1 1 0"/>
      <key name="partial" qpos="0 0 1 1 0 0 0"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  const mjtNum tol = MjTol(1e-12, 1e-6);
  const int free = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "free")];
  const int slide = m->jnt_qposadr[mj_name2id(m, mjOBJ_JOINT, "slide")];
  const int freedof = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "free")];
  const int slidedof = m->jnt_dofadr[mj_name2id(m, mjOBJ_JOINT, "slide")];
  const int mocap = mj_name2id(m, mjOBJ_BODY, "mocap");
  ASSERT_EQ(m->nkey, 2);
  ASSERT_EQ(m->na, 3);
  ASSERT_EQ(m->nu, 4);

  // the free body where the spec puts it, and a slide of 0.1, scaled by 6 and 2
  const mjtNum freepos[3] = {-1, 0, 6};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(m->qpos0[free + i], freepos[i], tol);
    EXPECT_NEAR(m->key_qpos[free + i], freepos[i], tol);
    EXPECT_NEAR(m->key_qpos[m->nq + free + i], freepos[i], tol);
  }
  EXPECT_NEAR(m->qpos0[slide], 0.1, tol);
  EXPECT_NEAR(m->key_qpos[slide], 0.2, tol);

  // linear velocities are scaled, angular ones are not
  const mjtNum qvel[6] = {6, 0, 0, 0, 0, 1};
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(m->key_qvel[freedof + i], qvel[i], tol);
  }
  EXPECT_NEAR(m->key_qvel[slidedof], 1, tol);

  // activations: an integrated velocity, a setpoint and an integral; controls:
  // a velocity, then a position, a velocity and a force
  for (int i = 0; i < 3; i++) EXPECT_NEAR(m->key_act[i], 0.2, tol);
  const mjtNum ctrl[4] = {0.2, 0.2, 0.2, 1.6};
  for (int i = 0; i < 4; i++) EXPECT_NEAR(m->key_ctrl[i], ctrl[i], tol);

  // the mocap body where the spec puts it
  const mjtNum mpos[3] = {1, 2, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(m->body_pos[3 * mocap + i], mpos[i], tol);
    EXPECT_NEAR(m->key_mpos[i], mpos[i], tol);
    EXPECT_NEAR(m->key_mpos[3 + i], mpos[i], tol);
  }

  // a partial keyframe is completed in the spec with the default configuration,
  // in the units of the spec
  EXPECT_NEAR(m->key_qpos[m->nq + slide], 0.1, tol);
  mjsKey* partial = mjs_asKey(mjs_findElement(spec, mjOBJ_KEY, "partial"));
  ASSERT_THAT(partial, NotNull());
  ASSERT_EQ(partial->qpos->size(), m->nq);
  EXPECT_NEAR((*partial->qpos)[slide], 0.05, tol);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// mj_copyBack writes the values of a scaled model in the units of the spec: an
// unchanged model leaves the spec as it is, and edits compile to themselves
TEST_F(FrameScaleTest, CopyBack) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame pos="0.5 0 0" euler="0 0 30" scale="2">
        <body name="arm" pos="0 0 1">
          <joint name="slide" type="slide" axis="1 0 0" range="-0.1 0.1"
                 damping="1"/>
          <geom name="hand" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.05"/>
          <site name="tip" pos="0 0 -0.5" size="0.01"/>
          <camera name="cam" pos="0 -1 0"/>
        </body>
        <body name="ball" pos="1 0 0.3">
          <freejoint/>
          <geom size="0.1"/>
          <site name="mark" pos="0 0 0.1"/>
        </body>
      </frame>
    </worldbody>
    <tendon>
      <spatial stiffness="10">
        <site site="tip"/>
        <site site="mark"/>
      </spatial>
    </tendon>
    <actuator>
      <position joint="slide" kp="100" ctrlrange="-0.1 0.1"/>
    </actuator>
    <sensor>
      <jointpos joint="slide" cutoff="1"/>
    </sensor>
    <keyframe>
      <key qpos="0.05 1 0 1 1 0 0 0"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  // unchanged
  mjSpec* copy = mj_copySpec(spec);
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  EXPECT_THAT(CompareSpec(copy, spec), IsEmpty());
  mj_deleteSpec(copy);

  // edits of the compiled values, which the frame scales by 2
  const int arm = mj_name2id(m, mjOBJ_BODY, "arm");
  const int hand = mj_name2id(m, mjOBJ_GEOM, "hand");
  const int tip = mj_name2id(m, mjOBJ_SITE, "tip");
  const int cam = mj_name2id(m, mjOBJ_CAMERA, "cam");
  m->body_pos[3 * arm + 2] += 0.5;
  m->body_mass[arm] = 5;
  m->jnt_range[1] = 0.3;
  m->dof_damping[0] = 20;
  m->geom_size[3 * hand] = 0.2;
  m->site_pos[3 * tip + 2] = -2;
  m->cam_pos[3 * cam + 1] = -3;
  m->tendon_stiffness[0] = 50;
  m->actuator_gainprm[0] = 1000;
  m->actuator_biasprm[1] = -1000;
  m->sensor_cutoff[0] = 3;
  m->key_qpos[0] = 0.15;
  m->key_qpos[1] += 1;
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);

  // the spec has them in its units
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_NEAR(mjs_findBody(spec, "arm")->pos[2], 1.25, tol);
  EXPECT_NEAR(mjs_findBody(spec, "arm")->mass, 5.0 / 8, tol);
  EXPECT_NEAR(
      mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "slide"))->range[1], 0.15,
      tol);
  EXPECT_NEAR(mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "hand"))->size[0],
              0.1, tol);

  // and compile to the edited model
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  auto expect = [&](const mjtNum* a, const mjtNum* b, int n, const char* name) {
    for (int i = 0; i < n; i++) {
      EXPECT_NEAR(a[i], b[i], tol * std::max(1.0, std::abs((double)a[i])))
          << name << "[" << i << "]";
    }
  };
  expect(m->body_pos, m2->body_pos, 3 * m->nbody, "body_pos");
  expect(m->body_mass, m2->body_mass, m->nbody, "body_mass");
  expect(m->body_inertia, m2->body_inertia, 3 * m->nbody, "body_inertia");
  expect(m->jnt_range, m2->jnt_range, 2 * m->njnt, "jnt_range");
  expect(m->dof_damping, m2->dof_damping, m->nv, "dof_damping");
  expect(m->geom_size, m2->geom_size, 3 * m->ngeom, "geom_size");
  expect(m->site_pos, m2->site_pos, 3 * m->nsite, "site_pos");
  expect(m->cam_pos, m2->cam_pos, 3 * m->ncam, "cam_pos");
  expect(m->tendon_stiffness, m2->tendon_stiffness, m->ntendon,
         "tendon_stiffness");
  expect(m->actuator_gainprm, m2->actuator_gainprm, mjNGAIN * m->nu,
         "actuator_gainprm");
  expect(m->actuator_biasprm, m2->actuator_biasprm, mjNBIAS * m->nu,
         "actuator_biasprm");
  expect(m->sensor_cutoff, m2->sensor_cutoff, m->nsensor, "sensor_cutoff");
  expect(m->key_qpos, m2->key_qpos, m->nq, "key_qpos");

  mj_deleteModel(m2);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a new radius of a capsule given by fromto leaves fromto in the spec, at any
// scale
TEST_F(FrameScaleTest, CopyBackFromtoRadius) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="1.7">
        <body>
          <geom name="g" type="capsule" fromto="0 0 0 0.3 0.1 -0.5" size="0.05"/>
          <site name="s" type="capsule" fromto="0 0 0 0.3 0.1 -0.5" size="0.05"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  m->geom_size[0] = 0.2;
  m->site_size[0] = 0.2;
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  mjsGeom* geom = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "g"));
  mjsSite* site = mjs_asSite(mjs_findElement(spec, mjOBJ_SITE, "s"));
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_EQ(geom->fromto[3], 0.3);
  EXPECT_EQ(site->fromto[3], 0.3);
  EXPECT_NEAR(geom->size[0], 0.2 / 1.7, tol);
  EXPECT_NEAR(site->size[0], 0.2 / 1.7, tol);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// copy-back divides by the mass factor
TEST_F(FrameScaleTest, MassScaleCopyBack) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame massscale="2" scaling="geometry">
        <body name="body">
          <freejoint/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->body_mass[1], 2);
  m->body_mass[1] = 3;
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  mjsBody* body = mjs_findBody(spec, "body");
  EXPECT_EQ(body->mass, 1.5);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// inertials which are written to the spec are in its units: adopted, merged,
// fused, and saved with saveinertial
TEST_F(FrameScaleTest, Inertials) {
  const mjtNum tol = MjTol(1e-12, 1e-6);

  // adopted
  mjSpec* spec =
      mj_parseXMLString(Pendulum(R"(scale="2")").c_str(), nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  mjsBody* arm = mjs_findBody(spec, "arm");
  ASSERT_EQ(mjs_adoptInertial(arm, nullptr), 0) << mjs_getError(spec);
  EXPECT_NEAR(arm->mass, m->body_mass[1] / 8, tol);
  mjModel* adopted = mj_compile(spec, nullptr);
  ASSERT_THAT(adopted, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(adopted->body_mass[1], m->body_mass[1], tol);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(adopted->body_ipos[3 + i], m->body_ipos[3 + i], tol);
    EXPECT_NEAR(adopted->body_inertia[3 + i], m->body_inertia[3 + i], tol);
  }
  mj_deleteModel(adopted);
  mj_deleteModel(m);
  mj_deleteSpec(spec);

  // merged: a child of mass 1 under a scale of 2 weighs 8 in its parent
  static constexpr char merge[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent" pos="0 0 1">
        <joint/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <frame pos="1 0 0" scale="2">
          <body name="child" pos="0 0 0.5">
            <inertial pos="0 0 0" mass="1" diaginertia="0.1 0.1 0.1"/>
          </body>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";
  spec = mj_parseXMLString(merge, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(mjs_findBody(spec, "parent")->mass, 9, tol);
  m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(m->body_mass[1], 9, 9 * tol);
  EXPECT_NEAR(m->body_ipos[3], 8.0 / 9, tol);
  EXPECT_NEAR(m->body_ipos[5], 8.0 / 9, tol);
  mj_deleteModel(m);
  mj_deleteSpec(spec);

  // fused, and saved with saveinertial: the same model as before
  static constexpr char fuse[] = R"(
  <mujoco>
    <compiler savecompiled="false" saveinertial="true"/>
    <worldbody>
      <frame pos="0 0 1" scale="2">
        <body name="arm">
          <joint/>
          <geom type="box" size="0.1 0.2 0.3"/>
          <body name="weight" pos="0.5 0 0">
            <inertial pos="0.1 0 0" mass="1" diaginertia="0.1 0.2 0.3"/>
          </body>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  spec = mj_parseXMLString(fuse, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  mjModel* saved = nullptr;
  {
    std::string xml = SaveAndReadXml(spec);
    EXPECT_THAT(xml, HasSubstr("scale=\"2\""));
    saved = LoadModelFromString(xml).release();
    ASSERT_THAT(saved, NotNull()) << xml;
  }
  spec->compiler.fusestatic = 1;
  mjModel* fused = mj_compile(spec, nullptr);
  ASSERT_THAT(fused, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(fused->nbody, 2);
  EXPECT_NEAR(fused->body_mass[1], m->body_mass[1] + m->body_mass[2],
              tol * fused->body_mass[1]);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(saved->body_inertia[3 + i], m->body_inertia[3 + i], tol);
    EXPECT_NEAR(saved->body_ipos[3 + i], m->body_ipos[3 + i], tol);
  }
  EXPECT_NEAR(saved->body_mass[1], m->body_mass[1], tol * m->body_mass[1]);
  mj_deleteModel(fused);
  mj_deleteModel(saved);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// ------------------------- anisotropic frame scale ---------------------------

// an anisotropic scale requires the geometry policy
TEST_F(FrameScaleTest, AnisotropicPolicy) {
  for (const char* scaling : {"similar", "geometry"}) {
    std::string xml = absl::StrFormat(R"(
    <mujoco>
      <worldbody>
        <frame scale="1 2 3" scaling="%s">
          <geom type="box" size="0.1 0.1 0.1"/>
        </frame>
      </worldbody>
    </mujoco>
    )",
                                      scaling);
    std::array<char, 1024> error;
    MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
    if (std::string(scaling) == "similar") {
      EXPECT_THAT(m.get(), IsNull());
      EXPECT_THAT(error.data(), HasSubstr("requires scaling 'geometry'"));
    } else {
      EXPECT_THAT(m.get(), NotNull()) << error.data();
    }
  }
}

// shapes aligned with the axes of the scale take its factors along their own
// axes; a sphere site takes the geometric mean, and other sites stretch along
// their own axes even when they are not aligned
TEST_F(FrameScaleTest, AnisotropicShapes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="2 3 4" scaling="geometry">
        <body name="b" pos="1 1 1">
          <geom name="box" type="box" size="0.1 0.2 0.3"/>
          <geom name="ellipsoid" type="ellipsoid" size="0.1 0.2 0.3"/>
          <geom name="turned" type="box" size="0.1 0.2 0.3" euler="0 0 90"/>
          <geom name="tilted" type="box" size="0.1 0.2 0.3" euler="0 0 1"/>
          <site name="ball" size="0.1"/>
          <site name="tiltedsite" type="box" size="0.1 0.2 0.3" euler="0 0 45"/>
          <site name="capsulesite" type="capsule" size="0.1 0.2"/>
        </body>
      </frame>
      <frame scale="2 2 3" scaling="geometry">
        <body>
          <geom name="capsule" type="capsule" size="0.1 0.2"/>
          <geom name="cylinder" type="cylinder" size="0.1 0.2"/>
        </body>
      </frame>
      <frame scale="3 2 2" scaling="geometry">
        <body>
          <geom name="fromto" type="capsule" fromto="0 0 0 1 0 0" size="0.1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-12, 1e-6);
  auto size = [&](const char* name) {
    return GetRow(m->geom_size, 3, mj_name2id(m.get(), mjOBJ_GEOM, name));
  };
  EXPECT_THAT(GetRow(m->body_pos, 3, 1), Pointwise(DoubleNear(tol), {2, 3, 4}));
  EXPECT_THAT(size("box"), Pointwise(DoubleNear(tol), {0.2, 0.6, 1.2}));
  EXPECT_THAT(size("ellipsoid"), Pointwise(DoubleNear(tol), {0.2, 0.6, 1.2}));
  EXPECT_THAT(size("turned"), Pointwise(DoubleNear(tol), {0.3, 0.4, 1.2}));
  EXPECT_THAT(size("tilted"), Pointwise(DoubleNear(1e-3), {0.2, 0.6, 1.2}));
  EXPECT_NEAR(m->site_size[0], 0.1 * std::cbrt(24), tol);
  auto sitesize = [&](const char* name) {
    return GetRow(m->site_size, 3, mj_name2id(m.get(), mjOBJ_SITE, name));
  };
  const double f = std::sqrt(6.5);
  EXPECT_THAT(sitesize("tiltedsite"),
              Pointwise(DoubleNear(tol), {0.1 * f, 0.2 * f, 1.2}));
  EXPECT_THAT(GetRow(sitesize("capsulesite").data(), 2, 0),
              Pointwise(DoubleNear(tol), {0.1 * std::sqrt(6.0), 0.8}));
  EXPECT_THAT(GetRow(size("capsule").data(), 2, 0),
              Pointwise(DoubleNear(tol), {0.2, 0.6}));
  EXPECT_THAT(GetRow(size("cylinder").data(), 2, 0),
              Pointwise(DoubleNear(tol), {0.2, 0.6}));
  EXPECT_THAT(GetRow(size("fromto").data(), 2, 0),
              Pointwise(DoubleNear(tol), {0.2, 1.5}));
}

// shapes which cannot represent their scaled selves are errors
TEST_F(FrameScaleTest, AnisotropicErrors) {
  constexpr std::array<std::array<const char*, 3>, 6> kCases = {{
      {"1 2 1", R"(<geom type="box" size="0.1 0.2 0.3" euler="0 0 45"/>)",
       "not aligned with the axes of the scale"},
      {"2 3 4", R"(<geom type="box" size="0.1 0.2 0.3" euler="0 0 5"/>)",
       "not aligned with the axes of the scale"},
      {"1 2 1", R"(<geom size="0.1"/>)", "use an ellipsoid instead"},
      {"1 2 3", R"(<geom type="capsule" size="0.1 0.2"/>)",
       "scaled equally across its axis"},
      {"1 2 1",
       R"(<body><joint axis="1 1 0"/><geom type="box" size="0.1 0.1 0.1"/></body>)",
       "axis of this joint is not aligned"},
      {"1 2 1",
       R"(<body><joint axis="1 0.0524 0"/><geom type="box" size="0.1 0.1 0.1"/></body>)",
       "axis of this joint is not aligned"},
  }};
  for (const auto& [scale, element, message] : kCases) {
    std::string xml = absl::StrFormat(R"(
    <mujoco>
      <worldbody>
        <frame scale="%s" scaling="geometry">%s</frame>
      </worldbody>
    </mujoco>
    )",
                                      scale, element);
    std::array<char, 1024> error;
    MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
    EXPECT_THAT(m.get(), IsNull()) << element;
    EXPECT_THAT(error.data(), HasSubstr(message)) << element;
  }
}

// a joint along an axis of the scale takes its factor; one within the
// alignment tolerance is accepted
TEST_F(FrameScaleTest, AnisotropicJoints) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="2 3 4" scaling="geometry">
        <body>
          <joint name="slide" type="slide" axis="0 1 0" range="-1 1"/>
          <joint name="hinge" axis="1 0 0" range="-30 30"/>
          <joint name="tilted" axis="1 0.0175 0"/>
          <geom type="box" size="0.1 0.1 0.1"/>
        </body>
      </frame>
    </worldbody>
    <actuator>
      <position joint="slide" kp="10" ctrlrange="-0.5 0.5"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_THAT(GetRow(m->jnt_range, 2, 0), Pointwise(DoubleNear(tol), {-3, 3}));
  EXPECT_NEAR(m->jnt_range[3], 30 * mjPI / 180, tol);
  EXPECT_THAT(GetRow(m->actuator_ctrlrange, 2, 0),
              Pointwise(DoubleNear(tol), {-1.5, 1.5}));
}

// the first mesh, body and geom of two models are the same, up to the order of
// the vertices; vertices are compared in the body frame, since the sign of each
// axis of the mesh frame is arbitrary
static void ExpectSameMeshBody(const mjModel* a, const mjModel* b) {
  const mjtNum tol = MjTol(1e-6, 1e-5);
  ASSERT_EQ(a->mesh_vertnum[0], b->mesh_vertnum[0]);
  auto sorted = [](const mjModel* m) {
    std::vector<std::array<mjtNum, 3>> v(m->mesh_vertnum[0]);
    for (int i = 0; i < v.size(); i++) {
      mjtNum vert[3];
      for (int k = 0; k < 3; k++) {
        vert[k] = m->mesh_vert[3 * (m->mesh_vertadr[0] + i) + k];
      }
      mju_rotVecQuat(v[i].data(), vert, m->geom_quat);
      for (int k = 0; k < 3; k++) v[i][k] += m->geom_pos[k];
    }
    std::sort(v.begin(), v.end());
    return v;
  };
  auto va = sorted(a), vb = sorted(b);
  for (int i = 0; i < va.size(); i++) {
    for (int k = 0; k < 3; k++) {
      EXPECT_NEAR(va[i][k], vb[i][k], MjTol(1e-5, 1e-4)) << i;
    }
  }
  EXPECT_NEAR(a->body_mass[1], b->body_mass[1], tol * b->body_mass[1]);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(a->body_inertia[3 + i], b->body_inertia[3 + i],
                tol * b->body_inertia[3 + i]);
  }
  EXPECT_THAT(GetRow(a->geom_size, 3, 0),
              Pointwise(DoubleNear(tol), GetRow(b->geom_size, 3, 0)));
  EXPECT_NEAR(a->geom_rbound[0], b->geom_rbound[0], tol);
}

// a mesh in a body which is rotated relative to the axes of the scale is
// sheared in its own coordinates, and compiles like the sheared mesh; the same
// mesh in an aligned body is another instance
TEST_F(FrameScaleTest, AnisotropicMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"
            face="0 2 1 0 1 3 0 3 2 1 2 3" texcoord="0 0 1 0 0 1 1 1"/>
    </asset>
    <worldbody>
      <frame scale="1 2 1" scaling="geometry">
        <body name="body" euler="0 0 45">
          <freejoint/>
          <geom type="mesh" mesh="tetra"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  // the map in the coordinates of the body: R^T diag(1, 2, 1) R for a turn of
  // 45 degrees about z
  static constexpr char reference[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1.5 0.5 0 0.5 1.5 0 0 0 1"/>
    </asset>
    <worldbody>
      <body name="body" euler="0 0 45">
        <freejoint/>
        <geom type="mesh" mesh="tetra"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  MjModelPtr ref = LoadModelFromString(reference, error.data(), error.size());
  ASSERT_THAT(ref.get(), NotNull()) << error.data();
  ExpectSameMeshBody(m.get(), ref.get());

  // saved compiled, the sheared instance reloads to the same model
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* compiled = mj_compile(spec, nullptr);
  ASSERT_THAT(compiled, NotNull()) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);
  MjModelPtr reloaded = LoadModelFromString(saved);
  ASSERT_THAT(reloaded.get(), NotNull()) << saved;
  ExpectSameMeshBody(compiled, reloaded.get());
  ASSERT_EQ(compiled->mesh_texcoordnum[0], 4);
  ASSERT_EQ(reloaded->mesh_texcoordnum[0], 4);
  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(reloaded->mesh_texcoord[i], compiled->mesh_texcoord[i]) << i;
  }
  mj_deleteModel(compiled);
  mj_deleteSpec(spec);
}

// full inertia tensor of a body in its coordinates
static std::array<mjtNum, 9> FullInertia(const mjModel* m, int body) {
  mjtNum mat[9], res[9], tmp[9];
  mju_quat2Mat(mat, m->body_iquat + 4 * body);
  mjtNum diag[9] = {0};
  for (int i = 0; i < 3; i++) diag[4 * i] = m->body_inertia[3 * body + i];
  mju_mulMatMat(tmp, mat, diag, 3, 3, 3);
  mju_mulMatMatT(res, tmp, mat, 3, 3, 3);
  std::array<mjtNum, 9> out;
  for (int i = 0; i < 9; i++) out[i] = res[i];
  return out;
}

// an explicit inertial transforms exactly: it equals the inertia which the
// scaled geometry gives
TEST_F(FrameScaleTest, AnisotropicInertia) {
  static constexpr char geometry[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <frame scale="%s" scaling="geometry">
        <body euler="0 0 30">
          <geom type="mesh" mesh="tetra" euler="10 20 0"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr m0 = LoadModelFromString(absl::StrFormat(geometry, "1"));
  ASSERT_THAT(m0.get(), NotNull());
  std::string explicit_xml = absl::StrFormat(
      R"(
  <mujoco>
    <worldbody>
      <frame scale="1 2 1.5" scaling="geometry">
        <body euler="0 0 30">
          <inertial pos="%.17g %.17g %.17g" quat="%.17g %.17g %.17g %.17g"
                    mass="%.17g" diaginertia="%.17g %.17g %.17g"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )",
      m0->body_ipos[3], m0->body_ipos[4], m0->body_ipos[5], m0->body_iquat[4],
      m0->body_iquat[5], m0->body_iquat[6], m0->body_iquat[7], m0->body_mass[1],
      m0->body_inertia[3], m0->body_inertia[4], m0->body_inertia[5]);
  std::array<char, 1024> error;
  MjModelPtr inferred = LoadModelFromString(
      absl::StrFormat(geometry, "1 2 1.5"), error.data(), error.size());
  ASSERT_THAT(inferred.get(), NotNull()) << error.data();
  MjModelPtr given =
      LoadModelFromString(explicit_xml, error.data(), error.size());
  ASSERT_THAT(given.get(), NotNull()) << error.data();
  const mjtNum tol = MjTol(1e-6, 1e-4);
  EXPECT_NEAR(given->body_mass[1], inferred->body_mass[1], tol);
  EXPECT_NEAR(given->body_mass[1], 3 * m0->body_mass[1], tol);
  EXPECT_THAT(GetRow(given->body_ipos, 3, 1),
              Pointwise(DoubleNear(tol), GetRow(inferred->body_ipos, 3, 1)));
  EXPECT_THAT(FullInertia(given.get(), 1),
              Pointwise(DoubleNear(tol), FullInertia(inferred.get(), 1)));
}

// a box given by fromto whose size is edited is written with the size rule of a
// box, which applies once fromto is replaced
TEST_F(FrameScaleTest, AnisotropicFromtoCopyBack) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="3 2 2" scaling="geometry">
        <body pos="0.1 0.2 0.3">
          <geom name="g" type="box" fromto="0 0 0 1 0 0" size="0.1"/>
          <site name="s" type="box" fromto="0 0 0 1 0 0" size="0.1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  const mjtNum tol = MjTol(1e-12, 1e-6);
  EXPECT_THAT(GetRow(m->geom_size, 3, 0),
              Pointwise(DoubleNear(tol), {0.2, 0.2, 1.5}));
  m->geom_size[1] = 0.4;
  m->site_size[1] = 0.4;
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  EXPECT_TRUE(std::isnan(
      mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "g"))->fromto[0]));
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  auto expect = [&](const mjtNum* a, const mjtNum* b, int n) {
    EXPECT_THAT(GetRow(a, n, 0), Pointwise(DoubleNear(tol), GetRow(b, n, 0)));
  };
  expect(m->geom_size, m2->geom_size, 3);
  expect(m->geom_pos, m2->geom_pos, 3);
  expect(m->geom_quat, m2->geom_quat, 4);
  expect(m->site_size, m2->site_size, 3);
  expect(m->site_pos, m2->site_pos, 3);
  expect(m->site_quat, m2->site_quat, 4);
  mj_deleteModel(m2);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// copy-back, saving as written and recompiling after a change of scale
TEST_F(FrameScaleTest, AnisotropicRoundTrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <frame name="frame" scale="1 2 1.5" scaling="geometry">
        <body name="arm" pos="0.1 0.2 0.3" euler="0 0 30">
          <joint type="slide" axis="0 0 1" range="-1 1"/>
          <geom type="box" size="0.1 0.2 0.3" euler="0 0 -30"/>
          <geom type="mesh" mesh="tetra"/>
          <inertial pos="0 0 0" mass="1" diaginertia="2 3 4"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = mj_parseXMLString(xml, nullptr, nullptr, 0);
  ASSERT_THAT(spec, NotNull());
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  // copy-back of unchanged values leaves the spec as it is
  mjSpec* copy = mj_copySpec(spec);
  ASSERT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  EXPECT_THAT(CompareSpec(copy, spec), IsEmpty());
  mj_deleteSpec(copy);

  // saved as written, the model reloads to the same compiled model
  spec->compiler.savecompiled = 0;
  std::string written = SaveAndReadXml(spec);
  MjModelPtr rewritten = LoadModelFromString(written);
  ASSERT_THAT(rewritten.get(), NotNull()) << written;
  std::string field;
  EXPECT_EQ(CompareModel(m, rewritten.get(), field), 0) << field;

  // recompiling after a change of scale gives what compiling afresh gives
  mjsFrame* frame = mjs_findFrame(spec, "frame");
  frame->scale[2] = 3;
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  std::string changed = xml;
  changed.replace(changed.find("1 2 1.5"), 7, "1 2 3");
  MjModelPtr fresh = LoadModelFromString(changed);
  ASSERT_THAT(fresh.get(), NotNull());
  EXPECT_EQ(CompareModel(m2, fresh.get(), field), 0) << field;

  mj_deleteModel(m2);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a skin whose bones share one map in its coordinates is mapped by it
TEST_F(FrameScaleTest, AnisotropicSkin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="2 3 4" scaling="geometry">
        <body name="b0" pos="0 0 1"/>
        <body name="b1" pos="1 0 1" euler="0 0 90"/>
      </frame>
    </worldbody>
    <deformable>
      <skin name="skin" vertex="0 0 1 1 0 1 0 1 1" face="0 1 2">
        <bone body="b0" bindpos="0 0 1" bindquat="1 0 0 0" vertid="0 2"
              vertweight="1 1"/>
        <bone body="b1" bindpos="1 0 1" bindquat="0.70710678 0 0 0.70710678"
              vertid="1" vertweight="1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_THAT(AsVector(m->skin_vert, 9),
              Pointwise(FloatNear(1e-5), {0, 0, 4, 2, 0, 4, 0, 3, 4}));
  EXPECT_THAT(AsVector(m->skin_bonebindpos, 6),
              Pointwise(FloatNear(1e-5), {0, 0, 4, 2, 0, 4}));
}

// ------------------------- mju_error during compile --------------------------

// mjCModel::Compile turns mju_error into a compile error via longjmp, skipping
// the destructors of C++ objects on the stack, so errors from bad input must
// not reach mju_error; the leak checks below come from running under asan

// SDF attribute error in the serial mesh compile path
TEST_F(UserModelTest, SdfAttributeErrorDoesNotLeak) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler usethread="false"/>
    <extension>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="torus_instance_with_a_long_name_to_defeat_sso">
          <config key="radius1" value="not_a_number_and_longer_than_sso"/>
          <config key="radius2" value="0.15"/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="torus">
        <plugin instance="torus_instance_with_a_long_name_to_defeat_sso"/>
      </mesh>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("invalid attribute value for 'radius1'"));
}

// an SDF attribute with trailing characters is invalid
TEST_F(UserModelTest, SdfAttributeTrailingCharacters) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="torus">
          <config key="radius1" value="0.35m"/>
          <config key="radius2" value=" 0.15 "/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="torus">
        <plugin instance="torus"/>
      </mesh>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("invalid attribute value for 'radius1'"));
}

// SDF attribute error in the threaded mesh compile path
TEST_F(UserModelTest, SdfAttributeErrorInWorkerThread) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="torus">
          <config key="radius1" value="not_a_number_and_longer_than_sso"/>
          <config key="radius2" value="0.15"/>
        </instance>
      </plugin>
    </extension>
    <asset>
      <mesh name="torus1">
        <plugin instance="torus"/>
      </mesh>
      <mesh name="torus2">
        <plugin instance="torus"/>
      </mesh>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("invalid attribute value for 'radius1'"));
}

// SDF attribute error in plugin init, for an instance which no mesh uses
TEST_F(UserModelTest, SdfInitErrorDoesNotLeak) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.gear">
        <instance name="gear">
          <config key="thickness" value="not_a_number_and_longer_than_sso"/>
        </instance>
      </plugin>
    </extension>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("Invalid parameter specification in Gear plugin"));
}

// ------------- test material textures ----------------------------------------

using MaterialTest = MujocoTest;

// the textures of a material are resolved from the spec: a material which is
// pointed at another texture compiles once the first texture is deleted
TEST_F(MaterialTest, TextureFromSpec) {
  mjSpec* spec = mj_makeSpec();
  mjsTexture* tex1 = mjs_addTexture(spec, nullptr);
  mjsTexture* tex2 = mjs_addTexture(spec, nullptr);
  for (mjsTexture* tex : {tex1, tex2}) {
    tex->type = mjTEXTURE_2D;
    tex->builtin = mjBUILTIN_CHECKER;
    tex->width = 8;
    tex->height = 8;
  }
  mjs_setName(tex1->element, "tex1");
  mjs_setName(tex2->element, "tex2");
  mjsMaterial* mat = mjs_addMaterial(spec, nullptr);
  mjs_setName(mat->element, "mat");
  (*mat->textures)[mjTEXROLE_RGB] = "tex1";
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->mat_texid[mjTEXROLE_RGB], 0);
  mj_deleteModel(m);

  (*mat->textures)[mjTEXROLE_RGB] = "tex2";
  EXPECT_EQ(mjs_delete(spec, tex1->element), 0) << mjs_getError(spec);
  m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->ntex, 1);
  EXPECT_EQ(m->mat_texid[mjTEXROLE_RGB], mj_name2id(m, mjOBJ_TEXTURE, "tex2"));
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(UserModelTest, DeleteDefaultErasesFromDefMap) {
  mjSpec* spec = mj_makeSpec();
  mjsDefault* def = mjs_addDefault(spec, "test_class", nullptr);
  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* body = mjs_addBody(world, nullptr);
  mjsGeom* geom = mjs_addGeom(body, def);
  geom->type = mjGEOM_SPHERE;
  geom->size[0] = 0.1;

  EXPECT_THAT(mjs_getDefault(geom->element), NotNull());
  ASSERT_EQ(mjs_delete(spec, def->element), 0) << mjs_getError(spec);
  EXPECT_THAT(mjs_getDefault(geom->element), IsNull());

  mj_deleteSpec(spec);
}

TEST_F(UserModelTest, CopyBackMaterialPbrAndTexuniform) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <material name="mat" texuniform="false" metallic="0.1" roughness="0.2"/>
    </asset>
    <worldbody>
      <geom size="1" material="mat"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nmat, 1);

  m->mat_texuniform[0] = 1;
  m->mat_metallic[0] = 0.75f;
  m->mat_roughness[0] = 0.85f;

  EXPECT_EQ(mj_copyBack(spec, m), 1) << mjs_getError(spec);
  mjsMaterial* mat =
      mjs_asMaterial(mjs_findElement(spec, mjOBJ_MATERIAL, "mat"));
  ASSERT_THAT(mat, NotNull());
  EXPECT_EQ(mat->texuniform, 1);
  EXPECT_FLOAT_EQ(mat->metallic, 0.75f);
  EXPECT_FLOAT_EQ(mat->roughness, 0.85f);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}
}  // namespace
}  // namespace mujoco
