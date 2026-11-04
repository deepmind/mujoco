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

// Tests for engine/engine_collision_convex.c.

#include "src/engine/engine_collision_convex.h"

#include <string>
#include <string_view>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

static const char* const kFramelessContactPath =
    "engine/testdata/collision_convex/frameless_contact.xml";
static const char* const kFramelessContactHfieldPath =
    "engine/testdata/collision_convex/frameless_contact_hfield.xml";
static const char* const kCylinderBoxPath =
    "engine/testdata/collision_convex/cylinder_box.xml";

using MjcConvexTest = MujocoTest;

TEST_F(MjcConvexTest, FramelessContact) {
  const std::string xml_path = GetTestDataFilePath(kFramelessContactPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  // Loading used to fail with "engine error: xaxis of contact frame undefined".
  EXPECT_THAT(model, NotNull()) << "Failed to load model: " << error;
  mj_deleteModel(model);
}

TEST_F(MjcConvexTest, FramelessContactHfield) {
  const std::string xml_path = GetTestDataFilePath(kFramelessContactHfieldPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  // Loading used to fail with "engine error: xaxis of contact frame undefined".
  EXPECT_THAT(model, NotNull()) << "Failed to load model: " << error;
  mj_deleteModel(model);
}

TEST_F(MjcConvexTest, CylinderBox) {
  const std::string xml_path = GetTestDataFilePath(kCylinderBoxPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << "Failed to load model: " << error;
  mjData* data = mj_makeData(model);

  // with multiCCD enabled, should find 4 contacts
  mj_forward(model, data);
  EXPECT_EQ(data->ncon, 4);

  // with multiCCD disabled, should find 1 contact
  model->opt.disableflags |= mjDSBL_MULTICCD;
  mj_forward(model, data);
  EXPECT_EQ(data->ncon, 1);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(MjcConvexTest, PlaneConvexMultipleFaces) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="wedge" vertex="1 1 -1 1 -1 -1 -1 -1 -1 -1 1 -1 0 1 1 0 -1 1"/>
    </asset>
    <worldbody>
      <geom type="plane" size="10 10 0.1"/>
      <body pos="0 -2 0">
        <freejoint/>
        <geom type="mesh" mesh="wedge"/>
      </body>
      <body pos="0 2 0.8" euler="-90 0 0">
        <freejoint/>
        <geom type="mesh" mesh="wedge"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());

  // quad bottom has 4 contacts, triangular side has 3 contacts
  EXPECT_EQ(data->ncon, 7);
}

TEST_F(MjcConvexTest, CylinderBoxHorizontal) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" size="1 1 1" pos="0 0 0"/>
      <body pos="0 0 1.49">
        <freejoint/>
        <geom type="cylinder" size="0.5 1" euler="90 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 2);

  model->opt.disableflags |= mjDSBL_MULTICCD;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 1);
}

TEST_F(MjcConvexTest, CylinderCylinderFaceToFace) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="cylinder" size="1 1" pos="0 0 0"/>
      <body pos="0 0 1.99">
        <freejoint/>
        <geom type="cylinder" size="1 1"/>
      </body>
    </worldbody>
  </mujoco>)";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 4);

  model->opt.disableflags |= mjDSBL_MULTICCD;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 1);
}

TEST_F(MjcConvexTest, CylinderCylinderSideToSide) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="cylinder" size="1 1" pos="0 0 0" euler="90 0 0"/>
      <body pos="0 0 1.99">
        <freejoint/>
        <geom type="cylinder" size="1 1" euler="90 0 0"/>
      </body>
    </worldbody>
  </mujoco>)";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  // TODO(kylebayes): support edge-edge cylinder multicontact
  EXPECT_EQ(data->ncon, 1);

  model->opt.disableflags |= mjDSBL_MULTICCD;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 1);
}

TEST_F(MjcConvexTest, PlaneFittedEllipsoid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="blob"
        vertex="0 0 .3  .2 0 0  0 .2 0  -.2 0 0  0 -.2 0  0 0 -.3"
        face="0 1 2  0 2 3  0 3 4  0 4 1  5 2 1  5 3 2  5 4 3  5 1 4"/>
    </asset>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body pos="0 0 .1">
        <freejoint/>
        <geom type="ellipsoid" mesh="blob"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // plane vs ellipsoid is a single contact, whether or not it was mesh-fitted
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->ncon, 1);
}

mjtNum HausdorffDist(const MjModelPtr& m, const MjDataPtr& d,
                     std::string_view name1, std::string_view name2,
                     int nitermax = 50, mjtNum stepsize = 0.5,
                     mjtNum tol = 1e-6) {
  int g1 = mj_name2id(m.get(), mjOBJ_GEOM, std::string(name1).c_str());
  int g2 = mj_name2id(m.get(), mjOBJ_GEOM, std::string(name2).c_str());
  mjCCDObj obj1, obj2;
  mjc_initCCDObj(&obj1, m.get(), d.get(), g1, 0);
  mjc_initCCDObj(&obj2, m.get(), d.get(), g2, 0);
  return mjc_hausdorff(&obj1, &obj2, nitermax, stepsize, tol);
}

int CheckEnclosed(const MjModelPtr& m, const MjDataPtr& d,
                  std::string_view name1, std::string_view name2,
                  int nitermax = 50, mjtNum stepsize = 0.5, mjtNum tol = 1e-6) {
  return HausdorffDist(m, d, name1, name2, nitermax, stepsize, tol) <= 0;
}

TEST_F(MjcConvexTest, IsEnclosedSpheres) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="outer" type="sphere" pos="0 0 0" size="1.0"/>
      <geom name="inner_center" type="sphere" pos="0 0 0" size="0.4"/>
      <geom name="inner_axial" type="sphere" pos="0.59 0 0" size="0.4"/>
      <geom name="protrude_axial" type="sphere" pos="0.61 0 0" size="0.4"/>
      <geom name="inner_diag" type="sphere" pos="0.4 0.4 0.4" size="0.3"/>
      <geom name="protrude_diag" type="sphere" pos="0.42 0.42 0.42" size="0.3"/>
      <geom name="disjoint" type="sphere" pos="3.0 0 0" size="0.2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // concentric and identical
  EXPECT_NEAR(HausdorffDist(model, data, "inner_center", "outer"), -0.6, 1e-5);
  EXPECT_NEAR(HausdorffDist(model, data, "outer", "inner_center"), 0.6, 1e-5);
  EXPECT_NEAR(HausdorffDist(model, data, "outer", "outer"), 0.0, 1e-5);

  // offset along X axis: 0.59 + 0.4 - 1.0 = -0.01 vs 0.61 + 0.4 - 1.0 = +0.01
  EXPECT_NEAR(HausdorffDist(model, data, "inner_axial", "outer"), -0.01, 1e-5);
  EXPECT_NEAR(HausdorffDist(model, data, "protrude_axial", "outer"), 0.01,
              1e-5);

  // offset along (1,1,1) diagonal:
  // inner_diag: 0.4 * sqrt(3) + 0.3 - 1.0 = -0.00717968 (enclosed)
  // protrude_diag: 0.42 * sqrt(3) + 0.3 - 1.0 = +0.02745626 (protrudes)
  EXPECT_NEAR(HausdorffDist(model, data, "inner_diag", "outer"),
              0.4 * mju_sqrt(3.0) - 0.7, 1e-4);
  EXPECT_NEAR(HausdorffDist(model, data, "protrude_diag", "outer"),
              0.42 * mju_sqrt(3.0) - 0.7, 1e-4);

  // completely separated: 3.0 + 0.2 - 1.0 = 2.2
  EXPECT_NEAR(HausdorffDist(model, data, "disjoint", "outer"), 2.2, 1e-5);
}

TEST_F(MjcConvexTest, IsEnclosedBoxesAndRotations) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="box_outer" type="box" pos="0 0 0" size="1 1 1"/>
      <geom name="box_inner" type="box" pos="0.2 -0.3 0.1" size="0.5 0.5 0.5"/>
      <geom name="box_rotated_fit" type="box" pos="0 0 0" euler="35 25 45" size="0.5 0.5 0.5"/>
      <geom name="box_rotated_poke" type="box" pos="0 0 0" euler="0 0 45" size="0.75 0.75 0.75"/>
      <body pos="1.5 -2.0 0.5" euler="30 45 60">
        <geom name="tilted_outer" type="box" size="1.0 0.8 0.6"/>
        <geom name="tilted_inner" type="box" pos="0.1 0.1 -0.05" euler="15 -20 10" size="0.4 0.3 0.2"/>
        <geom name="tilted_poke" type="box" pos="0.5 0.0 0.0" euler="0 45 0" size="0.5 0.3 0.5"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // closest face of box_inner is along -Y: 0.3 + 0.5 - 1.0 = -0.2
  EXPECT_NEAR(HausdorffDist(model, data, "box_inner", "box_outer"), -0.2, 1e-4);
  EXPECT_GT(HausdorffDist(model, data, "box_outer", "box_inner"), 0.0);
  EXPECT_EQ(CheckEnclosed(model, data, "box_rotated_fit", "box_outer"), 1);

  // corner reaches 0.75 * sqrt(2) = 1.06066 > 1.0 along X/Y faces
  EXPECT_NEAR(HausdorffDist(model, data, "box_rotated_poke", "box_outer"),
              0.75 * mju_sqrt(2.0) - 1.0, 1e-4);

  // arbitrarily rotated outer box container
  EXPECT_EQ(CheckEnclosed(model, data, "tilted_inner", "tilted_outer"), 1);
  EXPECT_EQ(CheckEnclosed(model, data, "tilted_outer", "tilted_inner"), 0);
  EXPECT_EQ(CheckEnclosed(model, data, "tilted_poke", "tilted_outer"), 0);
}

TEST_F(MjcConvexTest, IsEnclosedBoxInSphereCornerProtrusion) {
  // stress test for S^2 hill-climbing: a box centered in a unit sphere has
  // axial extent h < 1.0 (so all 6 k=0 axial checks are negative), but its 8
  // corners sit at distance h * sqrt(3) along the (+/-1, +/-1, +/-1) diagonals
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="sphere" type="sphere" size="1.0"/>
      <geom name="box_in" type="box" size="0.57 0.57 0.57"/>
      <geom name="box_out" type="box" size="0.59 0.59 0.59"/>
      <geom name="box_rot_in" type="box" euler="23 41 17" size="0.57 0.57 0.57"/>
      <geom name="box_rot_out" type="box" euler="23 41 17" size="0.59 0.59 0.59"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // 0.57 * sqrt(3) - 1.0 = -0.01273 < 0 -> enclosed
  EXPECT_NEAR(HausdorffDist(model, data, "box_in", "sphere"),
              0.57 * mju_sqrt(3.0) - 1.0, 1e-4);
  EXPECT_NEAR(HausdorffDist(model, data, "box_rot_in", "sphere"),
              0.57 * mju_sqrt(3.0) - 1.0, 1e-4);

  // 0.59 * sqrt(3) - 1.0 = +0.02191 > 0 -> corners poke out of sphere
  EXPECT_NEAR(HausdorffDist(model, data, "box_out", "sphere"),
              0.59 * mju_sqrt(3.0) - 1.0, 1e-4);
  EXPECT_NEAR(HausdorffDist(model, data, "box_rot_out", "sphere"),
              0.59 * mju_sqrt(3.0) - 1.0, 1e-4);

  // sphere inside box: protrudes by 1.0 - 0.57 = 0.43 along box face normals
  EXPECT_NEAR(HausdorffDist(model, data, "sphere", "box_in"), 0.43, 1e-4);
}

TEST_F(MjcConvexTest, IsEnclosedSmoothAndPolytopeShapes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="ellipsoid" type="ellipsoid" euler="20 -35 50" size="1.2 0.8 0.6"/>
      <geom name="cap_in" type="capsule" euler="20 -35 50" size="0.2 0.3"/>
      <geom name="cap_out" type="capsule" euler="20 -35 50" size="0.25 0.4"/>
      <geom name="cyl_outer" type="cylinder" euler="15 25 -40" size="1.0 1.0"/>
      <geom name="cyl_inner" type="cylinder" euler="15 25 -40" size="0.7 0.8"/>
      <geom name="box_in_cyl" type="box" euler="15 25 -40" size="0.69 0.69 0.95"/>
      <geom name="box_out_cyl" type="box" euler="15 25 -40" size="0.72 0.72 0.95"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  EXPECT_EQ(CheckEnclosed(model, data, "cap_in", "ellipsoid"), 1);
  EXPECT_NEAR(HausdorffDist(model, data, "cap_out", "ellipsoid"), 0.05, 1e-4);
  EXPECT_NEAR(HausdorffDist(model, data, "cyl_inner", "cyl_outer"), -0.2, 1e-4);
  // Rim of cyl_outer (r=1, z=1) to rim of cyl_inner (r=0.7, z=0.8) has
  // Euclidean Hausdorff distance sqrt(0.3^2 + 0.2^2) = sqrt(0.13)
  EXPECT_NEAR(HausdorffDist(model, data, "cyl_outer", "cyl_inner"),
              mju_sqrt(0.3 * 0.3 + 0.2 * 0.2), 1e-4);

  // box in cylinder: radial corner distance 0.69 * sqrt(2) = 0.9758 < 1.0 vs
  // 0.72 * sqrt(2) = 1.0182 > 1.0
  EXPECT_EQ(CheckEnclosed(model, data, "box_in_cyl", "cyl_outer"), 1);
  EXPECT_NEAR(HausdorffDist(model, data, "box_out_cyl", "cyl_outer"),
              0.72 * mju_sqrt(2.0) - 1.0, 1e-4);
}

TEST_F(MjcConvexTest, IsEnclosedConvexMeshes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="octa_outer"
            vertex=" 1  0  0  -1  0  0   0  1  0   0 -1  0   0  0  1   0  0 -1"/>
      <mesh name="octa_inner"
            vertex=" 0.8  0  0  -0.8  0  0   0  0.8  0   0 -0.8  0   0  0  0.8   0  0 -0.8"/>
      <mesh name="cube_in_octa"
            vertex="-0.32 -0.32 -0.32   0.32 -0.32 -0.32   0.32  0.32 -0.32  -0.32  0.32 -0.32
                    -0.32 -0.32  0.32   0.32 -0.32  0.32   0.32  0.32  0.32  -0.32  0.32  0.32"/>
      <mesh name="cube_out_octa"
            vertex="-0.35 -0.35 -0.35   0.35 -0.35 -0.35   0.35  0.35 -0.35  -0.35  0.35 -0.35
                    -0.35 -0.35  0.35   0.35 -0.35  0.35   0.35  0.35  0.35  -0.35  0.35  0.35"/>
    </asset>
    <worldbody>
      <geom name="outer" type="mesh" mesh="octa_outer"/>
      <geom name="inner" type="mesh" mesh="octa_inner"/>
      <geom name="cube_in" type="mesh" mesh="cube_in_octa"/>
      <geom name="cube_out" type="mesh" mesh="cube_out_octa"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // octahedron face plane is |x|+|y|+|z| <= 1
  // (face distance 1/sqrt(3) = 0.57735)
  EXPECT_EQ(CheckEnclosed(model, data, "inner", "outer"), 1);
  EXPECT_EQ(CheckEnclosed(model, data, "outer", "inner"), 0);

  // cube with half-side 0.32 has |x|+|y|+|z| = 0.96 < 1.0 (inside octahedron)
  EXPECT_EQ(CheckEnclosed(model, data, "cube_in", "outer"), 1);

  // cube with half-side 0.35 has |x|+|y|+|z| = 1.05 > 1.0 (corners poke through
  // the 8 diagonal faces of the octahedron by (1.05 - 1.0) / sqrt(3))
  EXPECT_NEAR(HausdorffDist(model, data, "cube_out", "outer"),
              0.05 / mju_sqrt(3.0), 1e-4);
}

TEST_F(MjcConvexTest, IsEnclosedDisjointInsideWorldAABB) {
  // verify that non-intersecting objects whose world AABB is strictly inside
  // obj2's world AABB are rejected without needing GJK first.
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="tilted_slab" type="ellipsoid" euler="0 45 0" size="1.0 1.0 0.05"/>
      <geom name="tucked_sphere" type="sphere" pos="-0.35 0 0.35" size="0.05"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  EXPECT_EQ(CheckEnclosed(model, data, "tucked_sphere", "tilted_slab"), 0);
}

TEST_F(MjcConvexTest, IsEnclosedScaleInvariance) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="micro_sphere" type="sphere" size="1e-4"/>
      <geom name="micro_box_in" type="box" size="5.7e-5 5.7e-5 5.7e-5"/>
      <geom name="micro_box_out" type="box" size="5.9e-5 5.9e-5 5.9e-5"/>
      <geom name="macro_sphere" type="sphere" size="1e3"/>
      <geom name="macro_box_in" type="box" size="570 570 570"/>
      <geom name="macro_box_out" type="box" size="590 590 590"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // micro scale (1e-4 m)
  EXPECT_NEAR(HausdorffDist(model, data, "micro_box_in", "micro_sphere"),
              (0.57 * mju_sqrt(3.0) - 1.0) * 1e-4, 1e-8);
  EXPECT_NEAR(HausdorffDist(model, data, "micro_box_out", "micro_sphere"),
              (0.59 * mju_sqrt(3.0) - 1.0) * 1e-4, 1e-8);

  // macro scale (1e3 m)
  EXPECT_NEAR(HausdorffDist(model, data, "macro_box_in", "macro_sphere"),
              (0.57 * mju_sqrt(3.0) - 1.0) * 1e3, 1e-1);
  EXPECT_NEAR(HausdorffDist(model, data, "macro_box_out", "macro_sphere"),
              (0.59 * mju_sqrt(3.0) - 1.0) * 1e3, 1e-1);
}

TEST_F(MjcConvexTest, IsEnclosedSite) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="outer_site" type="box" size="1.0 1.0 1.0" pos="0 0 0"/>
      <site name="inner_site" type="sphere" size="0.8" pos="0.1 0.1 0.1"/>
      <site name="protrude_site" type="sphere" size="0.8" pos="0.3 0 0"/>
      <geom name="inner_geom" type="capsule" size="0.2 0.5" pos="0 0 0"/>
      <geom name="protrude_geom" type="capsule" size="0.2 0.9" pos="0 0 0"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  int outer_s = mj_name2id(model.get(), mjOBJ_SITE, "outer_site");
  int inner_s = mj_name2id(model.get(), mjOBJ_SITE, "inner_site");
  int protrude_s = mj_name2id(model.get(), mjOBJ_SITE, "protrude_site");
  int inner_g = mj_name2id(model.get(), mjOBJ_GEOM, "inner_geom");
  int protrude_g = mj_name2id(model.get(), mjOBJ_GEOM, "protrude_geom");

  mjCCDObj outer_obj, inner_s_obj, protrude_s_obj, inner_g_obj, protrude_g_obj;
  mjc_initCCDObjSite(&outer_obj, model.get(), data.get(), outer_s, 0);
  mjc_initCCDObjSite(&inner_s_obj, model.get(), data.get(), inner_s, 0);
  mjc_initCCDObjSite(&protrude_s_obj, model.get(), data.get(), protrude_s, 0);
  mjc_initCCDObj(&inner_g_obj, model.get(), data.get(), inner_g, 0);
  mjc_initCCDObj(&protrude_g_obj, model.get(), data.get(), protrude_g, 0);

  // inner site is completely inside outer box: max dist along axes is 0.1 + 0.8
  // - 1.0 = -0.1
  EXPECT_NEAR(mjc_hausdorff(&inner_s_obj, &outer_obj, 50, 0.5, 1e-6), -0.1,
              1e-5);

  // protrude site extends to x = 0.3 + 0.8 = 1.1, outer box is 1.0 -> dist =
  // +0.1
  EXPECT_NEAR(mjc_hausdorff(&protrude_s_obj, &outer_obj, 50, 0.5, 1e-6), 0.1,
              1e-5);

  // capsule geom inside site: max z is 0.5 + 0.2 = 0.7, box is 1.0 -> dist =
  // -0.3
  EXPECT_NEAR(mjc_hausdorff(&inner_g_obj, &outer_obj, 50, 0.5, 1e-6), -0.3,
              1e-5);

  // capsule geom protruding: max z is 0.9 + 0.2 = 1.1, box is 1.0 -> dist =
  // +0.1
  EXPECT_NEAR(mjc_hausdorff(&protrude_g_obj, &outer_obj, 50, 0.5, 1e-6), 0.1,
              1e-5);
}

TEST_F(MjcConvexTest, HFieldMarginAndGap) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="hf" nrow="3" ncol="3" size="1 1 0.1 0.1"/>
    </asset>
    <worldbody>
      <geom name="terrain" type="hfield" hfield="hf" margin="0.02" gap="0.01"/>
      <body pos="0.1 0.1 0.065">
        <freejoint/>
        <geom type="sphere" size="0.05"/>
      </body>
      <flexcomp name="flex" type="grid" count="2 2 1" spacing="0.2 0.2 0.2"
                pos="-0.2 -0.2 0.065" radius="0.05" dim="2"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // sphere and flex bottoms are at z = 0.065 - 0.05 = 0.015, within margin
  // (0.02)
  int ngeom = 0, nflex = 0;
  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].geom[1] >= 0) ngeom++;
    if (data->contact[i].flex[1] >= 0) nflex++;
    EXPECT_NEAR(data->contact[i].dist, 0.015, MjTol(1e-6, 1e-5));
    EXPECT_NEAR(data->contact[i].pos[2], 0.0075, MjTol(1e-6, 1e-5));
  }
  EXPECT_GT(ngeom, 0);
  EXPECT_GT(nflex, 0);
}


TEST_F(MjcConvexTest, CylinderBoxNearZeroPenetrationMulticcd) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="cylinder" type="cylinder" size="0.005 0.01"/>
      <body pos="-0.0012280108103396095 0.006366577323866548 0.0050013282943414845"
            quat="0.6374277150568654 2.4329189787778094e-05 -9.472579190314077e-05 0.7705101547108238">
        <freejoint/>
        <geom name="box" type="box" size="0.001483693 0.0003489551 0.005"/>
      </body>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum geom_dist = mj_geomDistance(model.get(), data.get(), 0, 1, 0.05, nullptr);
  EXPECT_NEAR(geom_dist, -0.00044e-3, 1e-6);

  // Contact distance should match true distance and not phantom box width (~ -2.967 mm)
  ASSERT_GT(data->ncon, 0);
  for (int i = 0; i < data->ncon; i++) {
    EXPECT_NEAR(data->contact[i].dist, geom_dist, 1e-4);
  }
}

TEST_F(MjcConvexTest, BoxCylinderNearZeroPenetrationMulticcd) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="-0.0012280108103396095 0.006366577323866548 0.0050013282943414845"
            quat="0.6374277150568654 2.4329189787778094e-05 -9.472579190314077e-05 0.7705101547108238">
        <freejoint/>
        <geom name="box" type="box" size="0.001483693 0.0003489551 0.005"/>
      </body>
      <geom name="cylinder" type="cylinder" size="0.005 0.01"/>
    </worldbody>
  </mujoco>)";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum geom_dist = mj_geomDistance(model.get(), data.get(), 0, 1, 0.05, nullptr);
  EXPECT_NEAR(geom_dist, -0.00044e-3, 1e-6);

  // Contact distance should match true distance and not phantom box width (~ -2.967 mm)
  ASSERT_GT(data->ncon, 0);
  for (int i = 0; i < data->ncon; i++) {
    EXPECT_NEAR(data->contact[i].dist, geom_dist, 1e-4);
  }
}

}  // namespace
}  // namespace mujoco
