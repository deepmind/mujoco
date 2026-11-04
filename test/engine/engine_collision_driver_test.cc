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

// Tests for engine/engine_collision_driver.c.

#include "src/engine/engine_collision_driver.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using MjCollisionTest = MujocoTest;
using GeomPair = std::pair<std::string, std::string>;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::NotNull;

// Returns a sorted list of pairs of colliding geom names, where each pair of
// geom names is sorted.
static std::vector<GeomPair> colliding_pairs(const mjModel* model,
                                             const mjData* data) {
  std::vector<GeomPair> result;
  for (int i = 0; i < data->ncon; i++) {
    std::string geom1 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom[0]);
    std::string geom2 = mj_id2name(model, mjOBJ_GEOM, data->contact[i].geom[1]);
    result.push_back(GeomPair(std::min(geom1, geom2), std::max(geom1, geom2)));
  }
  std::sort(result.begin(), result.end());
  return result;
}

TEST_F(MjCollisionTest, AllCollisions) {
  static const char* const kModelFilePath = "engine/testdata/collisions.xml";
  const std::string xml_path = GetTestDataFilePath(kModelFilePath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  // mjCOL_ALL is the default
  mj_fwdPosition(model, data);
  EXPECT_THAT(colliding_pairs(model, data),
              ElementsAre(GeomPair("box", "sphere_collides"),
                          GeomPair("box", "sphere_predefined")));

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(MjCollisionTest, EmptyModel) {
  char error[1024];
  MjModelPtr model = LoadModelFromString("<mujoco/>", error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_fwdPosition(model.get(), data.get());
  EXPECT_THAT(colliding_pairs(model.get(), data.get()), IsEmpty());
}

TEST_F(MjCollisionTest, ZeroedHessian) {
  static const char* const kModelFilePath = "engine/testdata/collisions.xml";
  const std::string xml_path = GetTestDataFilePath(kModelFilePath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  mjData* data = mj_makeData(model);

  mj_fwdPosition(model, data);
  for (int i = 0; i < data->ncon; i++) {
    for (int j = 0; j < 36; j++) {
      EXPECT_FALSE(isnan(data->contact[i].H[j]))
          << "NaN in contact[" << i << "].H[" << j << "]";
    }
  }
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(MjCollisionTest, ContactCount) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom type="plane" size="5 5 .01"/>
      </body>
      <body pos="0 0 0.9">
        <freejoint/>
        <geom type="sphere" size="1" pos="-1 -1 0"/>
        <geom type="sphere" size="1" pos="-1  1 0"/>
        <geom type="sphere" size="1" pos=" 1 -1 0"/>
        <geom type="sphere" size="1" pos=" 1  1 0"/>
        <geom type="sphere" size="1" pos="-2 -2 0"/>
        <geom type="sphere" size="1" pos="-2  2 0"/>
        <geom type="sphere" size="1" pos=" 2 -2 0"/>
        <geom type="sphere" size="1" pos=" 2  2 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_forward(m.get(), d.get());

  // there are 8 spheres, all touching the floor
  EXPECT_EQ(d->ncon, 8);
}

TEST_F(MjCollisionTest, InGapContactsMultiGeomBody) {
  // In-gap contacts survive mid-phase BVH pruning. Both bodies
  // have a "far" geom that pulls the BVH root box away, so detecting the A-B
  // pair requires the descent filter to account for gap (body_margin).
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom name="A" type="box" size=".005 .01 .3" pos="-.015 0 0" gap=".004"/>
        <geom name="far1" type="box" size=".005 .01 .3" pos="-.2 0 0"/>
      </body>
      <body pos="0 0 1">
        <freejoint/>
        <geom name="B" type="box" size=".005 .01 .3" pos="-.0022 0 0" gap=".004"/>
        <geom name="far2" type="box" size=".005 .01 .3" pos=".1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_forward(m.get(), d.get());

  // geoms A and B are separated by 2.8mm, within the 8mm combined gap:
  // inactive contacts are expected
  EXPECT_GT(d->ncon, 0);
  for (int i = 0; i < d->ncon; i++) {
    EXPECT_GT(d->contact[i].dist, 0);
    EXPECT_LT(d->contact[i].dist, 0.008);
    EXPECT_EQ(d->contact[i].efc_address, -1);
  }
}

TEST_F(MjCollisionTest, MidphaseDescendsLargerTree) {
  // The small body sits inside the large body's root box but touches neither
  // leaf. Descending the larger tree first prunes immediately, so the small
  // body's children are never tested. The inertial offset makes BVH center ==
  // size, which zeroes the area if (center, size) is mistaken for (min, max).
  static constexpr char xml[] = R"(
  <mujoco>
    <visual>
      <global bvactive="true"/>
    </visual>
    <worldbody>
      <body>
        <freejoint/>
        <inertial pos="-1.1 -.1 -.1" mass="1" diaginertia="1 1 1"/>
        <geom type="box" size=".1 .1 .1" pos="-1 0 0"/>
        <geom type="box" size=".1 .1 .1" pos="1 0 0"/>
      </body>
      <body>
        <freejoint/>
        <geom type="box" size=".02 .02 .02" pos="-.05 0 0"/>
        <geom type="box" size=".02 .02 .02" pos=".05 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_forward(m.get(), d.get());

  // only the two root nodes are active
  int bvhadr = m->body_bvhadr[2];
  EXPECT_TRUE(d->bvh_active[m->body_bvhadr[1]]);
  EXPECT_TRUE(d->bvh_active[bvhadr]);
  for (int i = 1; i < m->body_bvhnum[2]; i++) {
    EXPECT_FALSE(d->bvh_active[bvhadr + i]);
  }
}

TEST_F(MjCollisionTest, FilterParent) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <freejoint/>
        <geom name="colliding1" size="1" pos="0 0 100"/>
        <body>
          <geom size="1"/>
          <body>
            <joint axis="1 0 0"/>
            <geom size="1" pos="0 0 50"/>
            <body>
              <geom name="colliding2" size="1" pos="0 0 99.5"/>
            </body>
          </body>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_fwdPosition(m.get(), d.get());

  // there should be zero contacts, because colliding1 and colliding2 are in
  // bodies that have a parent-child relationship, through welds
  EXPECT_EQ(d->ncon, 0);

  // when this filtering is disabled, the geoms should collide
  m->opt.disableflags |= mjDSBL_FILTERPARENT;
  mj_fwdPosition(m.get(), d.get());

  EXPECT_THAT(colliding_pairs(m.get(), d.get()),
              ElementsAre(GeomPair("colliding1", "colliding2")));
}

TEST_F(MjCollisionTest, FilterParentDoesntAffectWorldBody) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="colliding1" size="1" pos="0 0 100"/>
      <body pos="0 0 0">
        <joint axis="1 0 0"/>
        <geom name="colliding2" size="1" pos="0 0 99.5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_fwdPosition(m.get(), d.get());

  // even though colliding1 and colliding2 are have a parent-child relationship,
  // they collide because colliding1 is in <worldbody>
  EXPECT_THAT(colliding_pairs(m.get(), d.get()),
              ElementsAre(GeomPair("colliding1", "colliding2")));
}

TEST_F(MjCollisionTest, FilterStaticFlex) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body name="static">
        <geom name="static" size=".1"/>
      </body>
      <body name="rigid">
        <flexcomp name="rigid" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" rigid="true"/>
      </body>
      <body name="mocap" mocap="true" pos=".05 .05 .001">
        <flexcomp name="mocap" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2" rigid="true"/>
      </body>
      <body name="ball" pos="0 0 .09">
        <freejoint/>
        <geom name="ball" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_forward(m.get(), d.get());

  // the floor, the static geom and the two flexes overlap but have no dofs:
  // they collide only with the ball
  int ball = mj_name2id(m.get(), mjOBJ_GEOM, "ball");
  int nflexcon[2] = {0, 0};
  for (int i = 0; i < d->ncon; i++) {
    const mjContact& con = d->contact[i];
    EXPECT_TRUE(con.geom[0] == ball || con.geom[1] == ball);
    if (con.flex[1] >= 0) {
      nflexcon[con.flex[1]]++;
    }
  }
  EXPECT_GT(nflexcon[0], 0);
  EXPECT_GT(nflexcon[1], 0);
}

TEST_F(MjCollisionTest, FilterStaticFlexSelfCollision) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a" pos="-1 0 0"/>
      <body name="b" pos="1 0 0"/>
      <body name="c" pos="0 -1 .01"/>
      <body name="d" pos="0 1 .01"/>
      <body pos="0 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="rope" dim="1" radius=".01" body="a b c d" element="0 1 2 3">
        <edge stiffness="1"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_forward(m.get(), d.get());

  // the two elements of the flex cross, but its vertices are in static bodies
  EXPECT_EQ(d->ncon, 0);
}

TEST_F(MjCollisionTest, TestOBB) {
  mjtNum bvh1[6] = {-1, -1, -1, 1, 1, 1};
  mjtNum bvh2[6] = {-1, -1, -1, 1, 1, 1};
  mjtNum pos1[3] = {0, 0, 0};
  mjtNum mat1[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
  mjtNum pos2[3] = {1.71, 1.71, 0};  // just a little more than 1+sqrt(2)/2
  mjtNum mat2[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

  EXPECT_THAT(
      mj_collideOBB(bvh1, bvh2, pos1, mat1, pos2, mat2, 0, NULL, NULL, 0),
      true);

  // rotate by 45 degrees
  mat2[0] = 1. / mju_sqrt(2.);
  mat2[1] = -1. / mju_sqrt(2.);
  mat2[3] = 1. / mju_sqrt(2.);
  mat2[4] = 1. / mju_sqrt(2.);

  EXPECT_THAT(
      mj_collideOBB(bvh1, bvh2, pos1, mat1, pos2, mat2, 0, NULL, NULL, 0),
      false);
}

TEST_F(MjCollisionTest, PlaneInBody) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom pos="0 0 0" type="plane" size="1 1 .01"/>
      </body>
      <body pos="0 0 .0499">
        <joint type="slide" axis="0 0 1"/>
        <geom size=".05"/>
      </body>
    </worldbody>
    </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_step(m.get(), d.get());
}

TEST_F(MjCollisionTest, PinchingSucceeds) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" gravity="0 0 -9.81"/>
    <worldbody>
      <geom name="floor" type="plane" size="0 0 1"/>

      <body name="gripper" pos="0 0 0.5">
        <joint name="lift" type="slide" axis="0 0 1" damping="50"/>
        <geom type="box" size="0.2 0.05 0.02" rgba="0.5 0.5 0.5 1"/> <!-- base -->

        <body name="left_finger" pos="-0.1 0 -0.1">
          <joint name="left_slide" type="slide" axis="1 0 0" damping="10"/>
          <geom type="box" size="0.02 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
        </body>

        <body name="right_finger" pos="0.1 0 -0.1">
          <joint name="right_slide" type="slide" axis="-1 0 0" damping="10"/>
          <geom type="box" size="0.02 0.1 0.1" rgba="0.8 0.2 0.2 1"/>
        </body>
      </body>

      <flexcomp name="cloth" type="grid" dim="2" count="9 9 1" spacing="0.05 0.05 0.05"
                pos="0 0 0.1" radius="0.01">
        <edge equality="true"/>
      </flexcomp>
    </worldbody>

    <equality>
      <joint joint1="right_slide" joint2="left_slide"/>
    </equality>

    <tendon>
      <fixed name="grasp">
        <joint joint="right_slide" coef="1"/>
        <joint joint="left_slide" coef="1"/>
      </fixed>
    </tendon>

    <actuator>
      <position name="lift" joint="lift" kp="600" dampratio="1" ctrlrange="-1 1"/>
      <position name="grasp" tendon="grasp" kp="200" dampratio="1" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  int lift_id = mj_name2id(m.get(), mjOBJ_ACTUATOR, "lift");
  int grasp_id = mj_name2id(m.get(), mjOBJ_ACTUATOR, "grasp");

  // Phase 1: Lower gripper.
  // The gripper base starts at z=0.5. The finger has length 0.2 (size 0.1),
  // extending from z=0.4 to z=0.2 relative to base (center at -0.1).
  // The cloth is at z=0.1. We need to lower the gripper so the fingertips
  // reach the cloth. A lift value of -0.35 places the fingertips near z=0.05.

  for (int i = 0; i < 500; ++i) {
    d->ctrl[lift_id] = -0.35;  // Lower
    d->ctrl[grasp_id] = 0;     // Open
    mj_step(m.get(), d.get());
  }

  // Phase 2: Pinch
  for (int i = 0; i < 100; ++i) {
    d->ctrl[lift_id] = -0.35;  // Hold height
    d->ctrl[grasp_id] = 0.8;   // Close (max 1)
    mj_step(m.get(), d.get());
  }

  // Phase 3: Lift
  for (int i = 0; i < 1000; ++i) {
    d->ctrl[lift_id] = 0.5;   // Lift up
    d->ctrl[grasp_id] = 0.8;  // Keep closed
    mj_step(m.get(), d.get());
  }

  // Check if cloth is lifted
  // flex verts are in d->flexvert_xpos
  // original z is ~0.1 (falling to floor ~0.0)
  // gripper lifted to > 0.5 probably

  // Find average Z of cloth
  double avg_z = 0;
  int nvert = m->flex_vertnum[0];
  for (int i = 0; i < nvert; ++i) {
    avg_z += d->flexvert_xpos[3 * i + 2];
  }
  avg_z /= nvert;

  // If lifted, avg_z should be significantly > 0.1
  // If failed (slipped), avg_z should be near 0 (floor)

  // Specialized primitives (mjraw_BoxTriangle, mjraw_CapsuleTriangle) should
  // enable stable pinching, so we expect the cloth to be lifted.
  EXPECT_GT(avg_z, 0.2) << "Cloth slipped out of gripper!";
}

TEST_F(MjCollisionTest, MarginSumming) {
  // Two spheres with size 0.1, placed 0.21 apart (distance of 0.01).
  // With margin summing, margin1 + margin2 = 0.00999 + 0.00999 = 0.01998 > 0.01
  // so a contact should be generated.
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom name="sphere1" type="sphere" size=".1" margin="0.00999"/>
        <joint type="slide" axis="1 0 0"/>
      </body>
      <body pos=".21 0 0">
        <geom name="sphere2" type="sphere" size=".1" margin="0.00999"/>
        <joint type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_fwdPosition(m.get(), d.get());

  // With margin summing, we expect 1 contact
  EXPECT_EQ(d->ncon, 1);
}

TEST_F(MjCollisionTest, MaxContact) {
  constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag multiccd="enable"/>
    </option>
    <asset>
      <mesh name="smallbox"
        vertex="-1 -1 -1  1 -1 -1   1  1 -1
                 1  1  1  1 -1  1  -1  1 -1
                -1  1  1 -1 -1  1"/>
    </asset>
    <worldbody>
      <geom name="mesh" type="mesh" mesh="smallbox"/>
      <geom name="box" type="box" size="1 1 1"/>
      <geom name="plane" type="plane" size="1 1 1"/>
      <geom name="sphere" type="sphere" size="1"/>
      <geom name="capsule" type="capsule" size="1 1"/>
      <geom name="ellipsoid" type="ellipsoid" size="1 1 1"/>
      <geom name="cylinder" type="cylinder" size="1 1"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  int mesh = mj_name2id(m.get(), mjOBJ_GEOM, "mesh");
  int box = mj_name2id(m.get(), mjOBJ_GEOM, "box");
  int plane = mj_name2id(m.get(), mjOBJ_GEOM, "plane");
  int sphere = mj_name2id(m.get(), mjOBJ_GEOM, "sphere");
  int capsule = mj_name2id(m.get(), mjOBJ_GEOM, "capsule");
  int ellipsoid = mj_name2id(m.get(), mjOBJ_GEOM, "ellipsoid");
  int cylinder = mj_name2id(m.get(), mjOBJ_GEOM, "cylinder");

  EXPECT_EQ(mj_maxContact(m.get(), mesh, box, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), mesh, plane, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), box, plane, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), mesh, mesh, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), box, box, -1), 8);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, capsule, -1), 2);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, box, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, plane, -1), 2);
  EXPECT_EQ(mj_maxContact(m.get(), cylinder, plane, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, sphere, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, capsule, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, box, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, mesh, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, plane, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, cylinder, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, ellipsoid, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, box, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, mesh, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, plane, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, cylinder, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), ellipsoid, capsule, -1), 1);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, cylinder, -1), 2);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, mesh, -1), 2);
  EXPECT_EQ(mj_maxContact(m.get(), cylinder, cylinder, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), cylinder, box, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), cylinder, mesh, -1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), mesh, box, 1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), mesh, mesh, 1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), cylinder, box, 1), 4);
  EXPECT_EQ(mj_maxContact(m.get(), capsule, mesh, 1), 2);
}

// flex lying on a plane and on 16 spheres of the world body
std::string FlexOnGeoms(const std::string& flex) {
  return R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="0 0 1" pos="0 0 .05"/>
      <replicate count="4" offset=".2 0 0">
        <replicate count="4" offset="0 .2 0">
          <geom size=".05" pos="-.3 -.3 0"/>
        </replicate>
      </replicate>
      )" +
         flex + R"(
    </worldbody>
  </mujoco>
  )";
}

// contacts with the plane, all contacts, maximum contacts with one sphere
struct FlexContacts {
  int nplane = 0;
  int ncon = 0;
  int maxsphere = 0;
};

FlexContacts CountFlexContacts(const mjModel* m, mjData* d) {
  mj_fwdPosition(m, d);
  int plane = mj_name2id(m, mjOBJ_GEOM, "plane");
  std::vector<int> ngeom(m->ngeom, 0);
  FlexContacts count;
  count.ncon = d->ncon;
  for (int i = 0; i < d->ncon; i++) {
    int g = d->contact[i].geom[0] >= 0 ? d->contact[i].geom[0]
                                       : d->contact[i].geom[1];
    ngeom[g]++;
  }
  for (int g = 0; g < m->ngeom; g++) {
    if (g == plane) {
      count.nplane = ngeom[g];
    } else if (m->geom_type[g] == mjGEOM_SPHERE) {
      count.maxsphere = std::max(count.maxsphere, ngeom[g]);
    }
  }
  return count;
}

TEST_F(MjCollisionTest, VertexFlexContactsAreNotLimited) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(FlexOnGeoms(R"(
      <flexcomp name="cloth" type="grid" count="15 15 1" spacing=".05 .05 .05" dim="2"
                pos="0 0 .053" radius=".005">
        <edge equality="true"/>
        <contact selfcollide="none"/>
      </flexcomp>)"),
                                     error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  ASSERT_EQ(m->flex_rigid[0], 0);
  ASSERT_EQ(m->flex_interp[0], 0);

  MjDataPtr d_bvh = MakeData(m);
  FlexContacts bvh = CountFlexContacts(m.get(), d_bvh.get());

  // every vertex touches the plane, all contacts are kept
  EXPECT_EQ(bvh.nplane, m->flex_vertnum[0]);
  EXPECT_GT(bvh.ncon - bvh.nplane, 0);

  // same contacts without midphase
  m->opt.disableflags |= mjDSBL_MIDPHASE;
  MjDataPtr d_all = MakeData(m);
  FlexContacts all = CountFlexContacts(m.get(), d_all.get());
  EXPECT_EQ(all.nplane, bvh.nplane);
  EXPECT_EQ(all.ncon, bvh.ncon);
}

TEST_F(MjCollisionTest, ReducibleFlexContactLimitIsPerGeom) {
  // rigid cloth in a free body, trilinear soft box with its bottom layer on the
  // geoms
  const char* rigid = R"(
      <body name="cloth" pos="0 0 .053">
        <freejoint/>
        <inertial pos="0 0 0" mass="1" diaginertia=".1 .1 .1"/>
        <flexcomp name="cloth" type="grid" count="15 15 1" spacing=".05 .05 .05" dim="2"
                  radius=".005" rigid="true">
          <contact selfcollide="none"/>
        </flexcomp>
      </body>)";
  const char* trilinear = R"(
      <flexcomp name="box" type="grid" count="15 15 2" spacing=".05 .05 .05" dim="3"
                pos="0 0 .078" radius=".005" mass="1" dof="trilinear">
        <contact selfcollide="none"/>
      </flexcomp>)";
  for (const char* attrib : {rigid, trilinear}) {
    char error[1024];
    MjModelPtr m =
        LoadModelFromString(FlexOnGeoms(attrib), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << attrib << ": " << error;
    ASSERT_TRUE(m->flex_rigid[0] || m->flex_interp[0]) << attrib;

    MjDataPtr d_bvh = MakeData(m);
    FlexContacts bvh = CountFlexContacts(m.get(), d_bvh.get());

    // every vertex touches the plane, mjMAXCONPAIR contacts are kept
    EXPECT_EQ(bvh.nplane, mjMAXCONPAIR) << attrib;

    // the spheres have more contacts, each sphere at most mjMAXCONPAIR
    EXPECT_GT(bvh.ncon - bvh.nplane, mjMAXCONPAIR) << attrib;
    EXPECT_LE(bvh.maxsphere, mjMAXCONPAIR) << attrib;

    // same contacts without midphase
    m->opt.disableflags |= mjDSBL_MIDPHASE;
    MjDataPtr d_all = MakeData(m);
    FlexContacts all = CountFlexContacts(m.get(), d_all.get());
    EXPECT_EQ(all.nplane, bvh.nplane) << attrib;
    EXPECT_EQ(all.ncon, bvh.ncon) << attrib;
  }
}

TEST_F(MjCollisionTest, Flex3DActiveLayersMidphaseDisabled) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="box" type="box" size="0.15 0.15 0.05" pos="0 0 0"/>
      <flexcomp name="vol1" type="grid" dim="3" count="4 4 4"
                spacing="0.06 0.06 0.06" pos="0 0 0.09" radius="0.005" mass="1">
        <contact selfcollide="none" activelayers="1"/>
        <edge equality="true"/>
      </flexcomp>
      <flexcomp name="vol2" type="grid" dim="3" count="4 4 4"
                spacing="0.06 0.06 0.06" pos="0 0 0.22" radius="0.005" mass="1">
        <contact selfcollide="none" activelayers="1"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  MjDataPtr d_bvh = MakeData(m);
  mj_forward(m.get(), d_bvh.get());
  EXPECT_GT(d_bvh->ncon, 0);

  m->opt.disableflags |= mjDSBL_MIDPHASE;
  MjDataPtr d_all = MakeData(m);
  mj_forward(m.get(), d_all.get());
  EXPECT_EQ(d_all->ncon, d_bvh->ncon);

  for (int i = 0; i < d_all->ncon; ++i) {
    for (int k = 0; k < 2; ++k) {
      int f = d_all->contact[i].flex[k];
      int e = d_all->contact[i].elem[k];
      if (f >= 0 && e >= 0) {
        int layer = m->flex_elemlayer[m->flex_elemadr[f] + e];
        EXPECT_LT(layer, m->flex_activelayers[f]);
      }
    }
  }
}

// closed box surface with n[i] points along edge i and half-sizes size[i], as
// point and element strings of a flexcomp of type direct
void BoxSurface(const int n[3], const mjtNum size[3], std::string* point,
                std::string* element) {
  // index surface points
  std::vector<int> id(n[0] * n[1] * n[2], -1);
  int npoint = 0;
  for (int i = 0; i < n[0]; i++) {
    for (int j = 0; j < n[1]; j++) {
      for (int k = 0; k < n[2]; k++) {
        if (i == 0 || i == n[0] - 1 || j == 0 || j == n[1] - 1 || k == 0 ||
            k == n[2] - 1) {
          int ijk[3] = {i, j, k};
          id[(i * n[1] + j) * n[2] + k] = npoint++;
          for (int a = 0; a < 3; a++) {
            *point += std::to_string(size[a] * (2.0 * ijk[a] / (n[a] - 1) - 1));
            *point += ' ';
          }
        }
      }
    }
  }

  // two triangles per quad on each face
  for (int a = 0; a < 3; a++) {
    int b = (a + 1) % 3, c = (a + 2) % 3;
    for (int s : {0, n[a] - 1}) {
      for (int u = 0; u < n[b] - 1; u++) {
        for (int v = 0; v < n[c] - 1; v++) {
          int quad[4];
          int corner[4][2] = {{u, v}, {u + 1, v}, {u + 1, v + 1}, {u, v + 1}};
          for (int q = 0; q < 4; q++) {
            int ijk[3];
            ijk[a] = s;
            ijk[b] = corner[q][0];
            ijk[c] = corner[q][1];
            quad[q] = id[(ijk[0] * n[1] + ijk[1]) * n[2] + ijk[2]];
          }
          int tri[2][3] = {{quad[0], quad[1], quad[2]},
                           {quad[0], quad[2], quad[3]}};
          for (auto& t : tri) {
            if (s == 0) {
              std::swap(t[1], t[2]);
            }
            for (int vid : t) {
              *element += std::to_string(vid) + " ";
            }
          }
        }
      }
    }
  }
}

// pillow: closed box surface of 10 x 10 x 2 cm with interpolated dofs, resting
// on a floor, optionally pressed by a plate
std::string Pillow(const std::string& cellcount, const std::string& selfcollide,
                   const std::string& dof = "trilinear", bool plate = false) {
  const int n[3] = {11, 11, 3};
  const mjtNum size[3] = {.05, .05, .01};
  std::string point, element;
  BoxSurface(n, size, &point, &element);

  std::string plate_xml = plate ? R"(
      <body name="plate" pos="0 0 .028">
        <joint name="press" type="slide" axis="0 0 1" damping="5"/>
        <geom type="box" size=".07 .07 .005" mass=".1"/>
      </body>)"
                                : "";
  std::string actuator_xml = plate ? R"(
    <actuator>
      <motor joint="press" gear="-1"/>
    </actuator>)"
                                   : "";
  // narrow phase only: all element pairs are allocated on the arena
  std::string memory = selfcollide == "narrow" ? "200M" : "10M";
  return R"(
  <mujoco>
    <option timestep="1e-3" solver="CG" tolerance="1e-8" integrator="discrete"/>
    <size memory=")" +
         memory + R"("/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>)" +
         plate_xml + R"(
      <flexcomp name="pillow" type="direct" dim="2" pos="0 0 .0115"
                radius=".001" mass=".1" dof=")" +
         dof + R"(" cellcount=")" + cellcount + R"("
                point=")" +
         point + R"(" element=")" + element + R"(">
        <elasticity young="300" poisson=".2" damping=".002" elastic2d="none"/>
        <contact selfcollide=")" +
         selfcollide + R"(" solref=".005 1"/>
      </flexcomp>
    </worldbody>)" +
         actuator_xml + R"(
  </mujoco>
  )";
}

// self contacts of a flex: total, between its top and bottom faces, smallest
// distance
struct SelfContacts {
  int nself = 0;
  int ntopbottom = 0;
  mjtNum mindist = mjMAXVAL;
};

// vertex v (global index) of an interpolated flex on the top (side=1) or
// bottom (side=0) face of the cage, from its parametric coordinates
bool VertOnFace(const mjModel* m, int v, int side) {
  return std::abs(m->flex_vert0[3 * v + 2] - side) < 1e-6;
}

// element of interpolated flex f on the top (side=1) or bottom (side=0) face of
// the cage
bool OnFace(const mjModel* m, int f, int e, int side) {
  int dim = m->flex_dim[f];
  const int* edata = m->flex_elem + m->flex_elemdataadr[f] + e * (dim + 1);
  for (int i = 0; i <= dim; i++) {
    if (!VertOnFace(m, m->flex_vertadr[f] + edata[i], side)) {
      return false;
    }
  }
  return true;
}

SelfContacts CountSelfContacts(const mjModel* m, const mjData* d, int f) {
  SelfContacts count;
  for (int i = 0; i < d->ncon; i++) {
    const mjContact& con = d->contact[i];
    if (con.flex[0] != f || con.flex[1] != f) {
      continue;
    }
    count.nself++;
    count.mindist = std::min(count.mindist, con.dist);
    if ((OnFace(m, f, con.elem[0], 0) && OnFace(m, f, con.elem[1], 1)) ||
        (OnFace(m, f, con.elem[0], 1) && OnFace(m, f, con.elem[1], 0))) {
      count.ntopbottom++;
    }
  }
  return count;
}

// move the nodes of interpolated flex f such that the cage is flattened along z
// to thickness eps above its lowest node
void FlattenCage(const mjModel* m, mjData* d, int f, mjtNum eps) {
  mj_resetData(m, d);
  mj_kinematics(m, d);
  int order = std::abs(m->flex_interp[f]);
  int nz = m->flex_cellnum[3 * f + 2] * order + 1;
  int nodeadr = m->flex_nodeadr[f];
  mjtNum zmin = mjMAXVAL;
  for (int i = 0; i < m->flex_nodenum[f]; i++) {
    zmin = std::min(zmin, d->xpos[3 * m->flex_nodebodyid[nodeadr + i] + 2]);
  }
  for (int i = 0; i < m->flex_nodenum[f]; i++) {
    int b = m->flex_nodebodyid[nodeadr + i];
    mjtNum ztarget = zmin + eps * (i % nz) / (nz - 1);
    for (int j = m->body_jntadr[b]; j < m->body_jntadr[b] + m->body_jntnum[b];
         j++) {
      if (m->jnt_type[j] == mjJNT_SLIDE && m->jnt_axis[3 * j + 2] == 1) {
        d->qpos[m->jnt_qposadr[j]] += ztarget - d->xpos[3 * b + 2];
      }
    }
  }
  mj_fwdPosition(m, d);
}

TEST_F(MjCollisionTest, InterpolatedSelfCollisionNoContactsAtRest) {
  for (const char* cellcount : {"1 1 1", "2 2 3", "4 4 4"}) {
    for (const char* selfcollide : {"narrow", "bvh", "sap", "auto"}) {
      for (const char* dof : {"trilinear", "quadratic"}) {
        char error[1024];
        MjModelPtr m = LoadModelFromString(Pillow(cellcount, selfcollide, dof),
                                           error, sizeof(error));
        ASSERT_THAT(m.get(), NotNull()) << error;
        ASSERT_NE(m->flex_interp[0], 0);
        MjDataPtr d = MakeData(m);
        mj_fwdPosition(m.get(), d.get());

        // adjacent elements touch, but they share cage nodes: no self contacts
        EXPECT_EQ(CountSelfContacts(m.get(), d.get(), 0).nself, 0)
            << cellcount << ", " << selfcollide << ", " << dof;
      }
    }
  }
}

TEST_F(MjCollisionTest, InterpolatedSelfCollisionExcludesSharedNodes) {
  for (const char* selfcollide : {"narrow", "bvh", "sap", "auto"}) {
    // top and bottom faces are in cells 2 and 0, which have no common node
    char error[1024];
    MjModelPtr m =
        LoadModelFromString(Pillow("2 2 3", selfcollide), error, sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);
    FlattenCage(m.get(), d.get(), 0, 0.5 * m->flex_radius[0]);
    SelfContacts flat = CountSelfContacts(m.get(), d.get(), 0);
    EXPECT_GT(flat.ntopbottom, 0) << selfcollide;
    EXPECT_LE(flat.nself, mjMAXCONPAIR) << selfcollide;

    // top and bottom faces are in neighboring cells, which share the middle
    // layer of nodes: no self contacts
    for (const char* cellcount : {"2 2 1", "2 2 2"}) {
      MjModelPtr m_near = LoadModelFromString(Pillow(cellcount, selfcollide),
                                              error, sizeof(error));
      ASSERT_THAT(m_near.get(), NotNull()) << error;
      MjDataPtr d_near = MakeData(m_near);
      FlattenCage(m_near.get(), d_near.get(), 0, 0.5 * m->flex_radius[0]);
      EXPECT_EQ(CountSelfContacts(m_near.get(), d_near.get(), 0).nself, 0)
          << cellcount << ", " << selfcollide;
    }
  }
}

TEST_F(MjCollisionTest, InterpolatedSelfCollisionMidphaseParity) {
  // compare selfcollide="auto" (cell-level midphase) against "narrow", "bvh",
  // and "sap" on trilinear and quadratic pillows (including a 3x3x3 cell grid
  // where the 10x10x2 vertex grid has elements straddling cell boundaries)
  for (const char* cellcount : {"2 2 3", "3 3 3"}) {
    for (const char* dof : {"trilinear", "quadratic"}) {
      char error[1024];
      MjModelPtr m_auto = LoadModelFromString(Pillow(cellcount, "auto", dof),
                                              error, sizeof(error));
      ASSERT_THAT(m_auto.get(), NotNull()) << error;
      MjDataPtr d_auto = MakeData(m_auto);

      // push the top-corner cage node down to the bottom face so a small patch
      // of opposite cells collides (< mjMAXCONPAIR contacts, avoiding tie-break
      // truncation differences when bvh/sap swap (e1, e2) in GJK)
      mj_resetData(m_auto.get(), d_auto.get());
      mj_kinematics(m_auto.get(), d_auto.get());
      int nodeadr = m_auto->flex_nodeadr[0];
      mjtNum zmin = mjMAXVAL;
      for (int i = 0; i < m_auto->flex_nodenum[0]; i++) {
        zmin = std::min(
            zmin, d_auto->xpos[3 * m_auto->flex_nodebodyid[nodeadr + i] + 2]);
      }
      int b = m_auto->flex_nodebodyid[nodeadr + m_auto->flex_nodenum[0] - 1];
      mjtNum ztarget = zmin - 0.5 * m_auto->flex_radius[0];
      for (int j = m_auto->body_jntadr[b];
           j < m_auto->body_jntadr[b] + m_auto->body_jntnum[b]; j++) {
        if (m_auto->jnt_type[j] == mjJNT_SLIDE &&
            m_auto->jnt_axis[3 * j + 2] == 1) {
          d_auto->qpos[m_auto->jnt_qposadr[j]] +=
              ztarget - d_auto->xpos[3 * b + 2];
        }
      }
      mj_forward(m_auto.get(), d_auto.get());
      SelfContacts c_auto = CountSelfContacts(m_auto.get(), d_auto.get(), 0);
      ASSERT_GT(c_auto.ntopbottom, 0);
      ASSERT_LT(c_auto.nself, mjMAXCONPAIR);

      for (const char* mode : {"narrow", "bvh", "sap"}) {
        MjModelPtr m = LoadModelFromString(Pillow(cellcount, mode, dof), error,
                                           sizeof(error));
        ASSERT_THAT(m.get(), NotNull()) << error;
        MjDataPtr d = MakeData(m);
        mju_copy(d->qpos, d_auto->qpos, m->nq);
        mj_forward(m.get(), d.get());

        ASSERT_EQ(d->ncon, d_auto->ncon)
            << cellcount << ", " << dof << ", " << mode;

        if (std::string(mode) == "narrow") {
          // auto and narrow both canonicalize (e1 < e2) and sort pairs, so
          // their contacts are bit-for-bit identical
          for (int i = 0; i < d->ncon; i++) {
            EXPECT_EQ(d->contact[i].elem[0], d_auto->contact[i].elem[0])
                << cellcount << ", " << dof << ", contact " << i;
            EXPECT_EQ(d->contact[i].elem[1], d_auto->contact[i].elem[1])
                << cellcount << ", " << dof << ", contact " << i;
            EXPECT_EQ(d->contact[i].dist, d_auto->contact[i].dist)
                << cellcount << ", " << dof << ", contact " << i;
            for (int k = 0; k < 3; k++) {
              EXPECT_EQ(d->contact[i].pos[k], d_auto->contact[i].pos[k]);
              EXPECT_EQ(d->contact[i].frame[k], d_auto->contact[i].frame[k]);
            }
          }
          for (int i = 0; i < m->nv; i++) {
            EXPECT_EQ(d->qfrc_constraint[i], d_auto->qfrc_constraint[i])
                << cellcount << ", " << dof << ", dof " << i;
          }
        } else {
          // bvh and sap do not canonicalize e1 < e2 before GJK/EPA, so GJK
          // distances can differ by ~1e-7 when swapping (e1, e2)
          SelfContacts c_mode = CountSelfContacts(m.get(), d.get(), 0);
          EXPECT_EQ(c_mode.nself, c_auto.nself);
          EXPECT_EQ(c_mode.ntopbottom, c_auto.ntopbottom);
          EXPECT_NEAR(c_mode.mindist, c_auto.mindist, 1e-6);
          for (int i = 0; i < m->nv; i++) {
            EXPECT_THAT(d->qfrc_constraint[i],
                        MjNear(d_auto->qfrc_constraint[i], 1e-4, 1e-3))
                << cellcount << ", " << dof << ", " << mode << ", dof " << i;
          }
        }
      }
    }
  }

  // also test an element spanning >= 2 cells colliding with another element in
  // the same primary cell (both have e_lo = 0, but one spans cells [0, 3] and
  // has only 1 vertex in cell 0 and 2 vertices in cell 3, while the other is
  // in cell 0; after flattening the cage, only the vertices in cell 3 touch
  // cell 0)
  std::string xml_span = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="span" type="direct" dim="2" dof="trilinear"
                cellcount="1 1 4" radius=".01" mass="1"
                element="0 1 2  3 4 5  6 7 8"
                point="0 0 0  1 0 0  0 1 0
                       1 1 0  1 0 1  0 1 1
                       5 0 0  5 1 0  5 0 1">
        <elasticity young="1"/>
        <contact selfcollide="auto"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m_span_auto =
      LoadModelFromString(xml_span.c_str(), error, sizeof(error));
  ASSERT_THAT(m_span_auto.get(), NotNull()) << error;
  MjDataPtr d_span_auto = MakeData(m_span_auto);
  FlattenCage(m_span_auto.get(), d_span_auto.get(), 0,
              0.5 * m_span_auto->flex_radius[0]);
  mj_forward(m_span_auto.get(), d_span_auto.get());

  for (const char* mode : {"narrow", "bvh", "sap"}) {
    std::string xml_mode = xml_span;
    size_t pos = xml_mode.find("auto");
    xml_mode.replace(pos, 4, mode);
    MjModelPtr m_span =
        LoadModelFromString(xml_mode.c_str(), error, sizeof(error));
    ASSERT_THAT(m_span.get(), NotNull()) << error;
    MjDataPtr d_span = MakeData(m_span);
    FlattenCage(m_span.get(), d_span.get(), 0, 0.5 * m_span->flex_radius[0]);
    mj_forward(m_span.get(), d_span.get());
    ASSERT_EQ(d_span->ncon, d_span_auto->ncon) << mode;
    for (int i = 0; i < d_span->ncon; i++) {
      EXPECT_EQ(
          std::min(d_span->contact[i].elem[0], d_span->contact[i].elem[1]),
          d_span_auto->contact[i].elem[0]);
      EXPECT_EQ(
          std::max(d_span->contact[i].elem[0], d_span->contact[i].elem[1]),
          d_span_auto->contact[i].elem[1]);
      EXPECT_NEAR(d_span->contact[i].dist, d_span_auto->contact[i].dist, 1e-6);
    }
  }
}

// number of contacts between elements 0 and 1 of the flex with two horizontal
// triangles at heights z1 and z2 (and a vertical triangle far away, so that the
// cage spans z in [0, 1] with 4 cells along z), after flattening the cage
int FlattenedTrianglePairContacts(mjtNum z1, mjtNum z2) {
  char point[512];
  std::snprintf(point, sizeof(point),
                "0 0 %.17g  1 0 %.17g  0 1 %.17g  0 0 %.17g  1 0 %.17g  "
                "0 1 %.17g  5 0 0  5 1 0  5 0 1",
                z1, z1, z1, z2, z2, z2);
  std::string xml = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="tris" type="direct" dim="2" dof="trilinear"
                cellcount="1 1 4" radius=".001" mass="1"
                element="0 1 2  3 4 5  6 7 8" point=")" +
                    std::string(point) + R"(">
        <contact selfcollide="narrow"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
  EXPECT_THAT(m.get(), NotNull()) << error;
  if (!m) return -1;
  MjDataPtr d = MakeData(m);
  FlattenCage(m.get(), d.get(), 0, 0.5 * m->flex_radius[0]);
  int ncon = 0;
  for (int i = 0; i < d->ncon; i++) {
    const mjContact& con = d->contact[i];
    if (con.flex[0] == 0 && con.flex[1] == 0 &&
        std::min(con.elem[0], con.elem[1]) == 0 &&
        std::max(con.elem[0], con.elem[1]) == 1) {
      ncon++;
    }
  }
  return ncon;
}

TEST_F(MjCollisionTest, InterpolatedSelfCollisionBoundarySymmetry) {
  // with 4 cells along z, heights 0, .25, .5, .75 and 1 are cell planes 0 to 4;
  // a triangle on an internal plane belongs to the cells on both sides

  // symmetric about plane 2: a triangle on plane 2 is in cells 1 and 2, which
  // neighbor cells 0 and 3
  EXPECT_EQ(FlattenedTrianglePairContacts(0, .5), 0);
  EXPECT_EQ(FlattenedTrianglePairContacts(.5, 1), 0);
  EXPECT_GT(FlattenedTrianglePairContacts(0, .75), 0);
  EXPECT_GT(FlattenedTrianglePairContacts(.25, 1), 0);
  EXPECT_GT(FlattenedTrianglePairContacts(0, 1), 0);

  // rounding below a cell plane does not change the cells of a triangle
  EXPECT_EQ(FlattenedTrianglePairContacts(0, .5 - 1e-16), 0);
  EXPECT_EQ(FlattenedTrianglePairContacts(.5 - 1e-16, 1), 0);

  // a triangle inside cell 1 neighbors cells 0 and 2, not 3
  EXPECT_EQ(FlattenedTrianglePairContacts(.375, 0), 0);
  EXPECT_GT(FlattenedTrianglePairContacts(.375, 1), 0);
}

// lowest vertex of the top face and highest vertex of the bottom face of the
// cage
std::pair<mjtNum, mjtNum> FaceHeights(const mjModel* m, const mjData* d,
                                      int f) {
  mjtNum ztop = mjMAXVAL, zbottom = -mjMAXVAL;
  for (int v = m->flex_vertadr[f]; v < m->flex_vertadr[f] + m->flex_vertnum[f];
       v++) {
    mjtNum z = d->flexvert_xpos[3 * v + 2];
    if (VertOnFace(m, v, 1)) {
      ztop = std::min(ztop, z);
    } else if (VertOnFace(m, v, 0)) {
      zbottom = std::max(zbottom, z);
    }
  }
  return {ztop, zbottom};
}

TEST_F(MjCollisionTest, InterpolatedSelfCollisionPressedPillow) {
  for (const char* selfcollide : {"auto", "none"}) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(
        Pillow("2 2 3", selfcollide, "trilinear", /*plate=*/true), error,
        sizeof(error));
    ASSERT_THAT(m.get(), NotNull()) << error;
    MjDataPtr d = MakeData(m);

    // press the plate down with a force that collapses the cage, then hold it
    // until the pillow settles; track the smallest gap between the faces
    mjtNum mingap = mjMAXVAL;
    for (int i = 0; i < 2000; i++) {
      d->ctrl[0] = 3 * std::min(1.0, i / 1000.0);
      mj_step(m.get(), d.get());
      auto [top, bottom] = FaceHeights(m.get(), d.get(), 0);
      mingap = std::min(mingap, top - bottom);
    }
    SelfContacts pressed = CountSelfContacts(m.get(), d.get(), 0);
    auto [ztop, zbottom] = FaceHeights(m.get(), d.get(), 0);
    mjtNum vmax = 0;
    for (int i = 0; i < m->nv; i++) {
      vmax = std::max(vmax, std::abs(d->qvel[i]));
    }

    // simulation is stable
    for (int i = 0; i < m->nq; i++) {
      EXPECT_TRUE(std::isfinite(d->qpos[i])) << selfcollide;
    }
    for (int i = 0; i < mjNWARNING; i++) {
      EXPECT_EQ(d->warning[i].number, 0) << selfcollide << ", warning " << i;
    }

    if (std::string(selfcollide) == "auto") {
      // settled: top and bottom faces are in contact, penetration is bounded
      // (only the settled state is checked: the deepest transient penetration
      // depends on the platform)
      EXPECT_LT(vmax, 0.06);
      EXPECT_GT(pressed.ntopbottom, 0);
      EXPECT_GT(pressed.mindist, -m->flex_radius[0]);
      EXPECT_GT(ztop, zbottom);
    } else {
      // without self-collision, the top face passes through the bottom face
      EXPECT_EQ(pressed.ntopbottom, 0);
      EXPECT_LT(mingap, 0);
    }
  }
}

TEST_F(MjCollisionTest, ParallelCapsuleCapsule) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body euler="30 45 60">
        <geom name="a" type="capsule" size="0.01 2.0" pos="0 0.005 0"/>
      </body>
      <body euler="30 0 0">
        <body euler="0 45 60">
          <geom name="b" type="capsule" size="0.01 1.5" pos="0 -0.005 0.3"/>
          <geom name="c" type="capsule" size="0.01 1.5" pos="0 -0.095 0.3"/>
          <geom name="d" type="capsule" size="0.01 1.5" pos="0 -0.095 0.3" euler="0 0.1 0"/>
          <geom name="e" type="capsule" size="0.01 1.5" pos="0 -0.015 0.3" euler="0 0.01 0"/>
        </body>
      </body>
    </worldbody>
    <contact>
      <pair geom1="a" geom2="b"/>
    </contact>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());

  mj_forward(m.get(), d.get());

  // Penetrating rotated parallel capsules (a, b) should generate 2 contacts.
  ASSERT_EQ(d->ncon, 2);
  EXPECT_THAT(d->contact[0].dist, MjNear(-0.01, 1e-12, 1e-5));
  EXPECT_THAT(d->contact[1].dist, MjNear(-0.01, 1e-12, 1e-5));

  // Separated rotated parallel capsules (a, c) have surface distance 0.08.
  int a = mj_name2id(m.get(), mjOBJ_GEOM, "a");
  int c = mj_name2id(m.get(), mjOBJ_GEOM, "c");
  EXPECT_THAT(mj_geomDistance(m.get(), d.get(), a, c, 0.2, nullptr),
              MjNear(0.08, 1e-12, 1e-5));

  // Near-parallel (0.1 deg) capsules (a, d): surface distance 0.08, closest
  // point on d is at its center cross-section (distance 0.01 from d's xpos).
  int d_id = mj_name2id(m.get(), mjOBJ_GEOM, "d");
  mjtNum fromto[6];
  EXPECT_THAT(mj_geomDistance(m.get(), d.get(), a, d_id, 0.2, fromto),
              MjNear(0.08, 1e-12, 1e-5));
  EXPECT_THAT(mju_dist3(fromto + 3, d->geom_xpos + 3 * d_id),
              MjNear(0.01, 1e-12, 1e-2));

  // Near-parallel (0.01 deg) capsules (a, e): surface distance 0.0, closest
  // point on e is at its center cross-section (distance 0.01 from e's xpos).
  int e_id = mj_name2id(m.get(), mjOBJ_GEOM, "e");
  EXPECT_THAT(mj_geomDistance(m.get(), d.get(), a, e_id, 0.2, fromto),
              MjNear(0.0, 1e-12, 1e-5));
  EXPECT_THAT(mju_dist3(fromto + 3, d->geom_xpos + 3 * e_id),
              MjNear(0.01, 1e-12, 2e-1));
}

// Parallel capsules where the first one overhangs the second: the closest
// points come from the second capsule's endpoints, so the distance must not
// depend on distmax (the collision margin) or on the geom order.
TEST_F(MjCollisionTest, ParallelCapsuleCapsuleOverhang) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="long" type="capsule" size="0.05 0.3"/>
      <body>
        <geom name="short" type="capsule" size="0.08 0.2" pos="0.2 0 0.05"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="long" geom2="short" margin="1"/>
    </contact>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  ASSERT_THAT(d, NotNull());
  mj_forward(m.get(), d.get());

  // Surface distance: 0.2 - (0.05 + 0.08) = 0.07 at both ends of short's span
  // [-0.15, 0.25].
  ASSERT_EQ(d->ncon, 2);
  EXPECT_MJTNUM_EQ(d->contact[0].dist, 0.07);
  EXPECT_MJTNUM_EQ(d->contact[1].dist, 0.07);
  EXPECT_MJTNUM_EQ(d->contact[0].pos[2], -0.15);
  EXPECT_MJTNUM_EQ(d->contact[1].pos[2], 0.25);

  int l = mj_name2id(m.get(), mjOBJ_GEOM, "long");
  int s = mj_name2id(m.get(), mjOBJ_GEOM, "short");
  for (mjtNum distmax : {0.08, 1.0}) {
    mjtNum fromto[6];
    EXPECT_MJTNUM_EQ(mj_geomDistance(m.get(), d.get(), l, s, distmax, fromto),
                     0.07);
    EXPECT_MJTNUM_EQ(fromto[2], -0.15);
    EXPECT_MJTNUM_EQ(fromto[5], -0.15);

    EXPECT_MJTNUM_EQ(mj_geomDistance(m.get(), d.get(), s, l, distmax, fromto),
                     0.07);
    EXPECT_MJTNUM_EQ(fromto[2], -0.15);
    EXPECT_MJTNUM_EQ(fromto[5], -0.15);
  }
}

TEST_F(MjCollisionTest, MaxContactSdfPairs) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" vertex="-0.5 -0.5 -0.5  0.5 -0.5 -0.5  -0.5 0.5 -0.5  0.5 0.5 -0.5
                                -0.5 -0.5  0.5  0.5 -0.5  0.5  -0.5 0.5  0.5  0.5 0.5  0.5"/>
      <hfield name="hf" nrow="2" ncol="2" size="1 1 1 1" elevation="0 0 0 0"/>
    </asset>
    <worldbody>
      <geom name="sdf" type="sdf" mesh="cube"/>
      <geom name="plane" type="plane" size="1 1 0.1"/>
      <geom name="mesh" type="mesh" mesh="cube"/>
      <geom name="sphere" type="sphere" size="0.5"/>
      <geom name="hfield" type="hfield" hfield="hf"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  int sdf = mj_name2id(m.get(), mjOBJ_GEOM, "sdf");
  int plane = mj_name2id(m.get(), mjOBJ_GEOM, "plane");
  int mesh = mj_name2id(m.get(), mjOBJ_GEOM, "mesh");
  int sphere = mj_name2id(m.get(), mjOBJ_GEOM, "sphere");
  int hfield = mj_name2id(m.get(), mjOBJ_GEOM, "hfield");

  m->opt.sdf_initpoints = 2;
  EXPECT_EQ(mj_maxContact(m.get(), plane, sdf, 0), 4);
  EXPECT_EQ(mj_maxContact(m.get(), sdf, plane, 0), 4);
  EXPECT_EQ(mj_maxContact(m.get(), mesh, sdf, 0), mjMAXCONPAIR);
  EXPECT_EQ(mj_maxContact(m.get(), sdf, mesh, 0), mjMAXCONPAIR);
  EXPECT_EQ(mj_maxContact(m.get(), hfield, sdf, 0), 0);
  EXPECT_EQ(mj_maxContact(m.get(), sphere, sdf, 0), 2);

  m->opt.sdf_initpoints = 2 * mjMAXCONPAIR;
  EXPECT_EQ(mj_maxContact(m.get(), sphere, sdf, 0), mjMAXCONPAIR);

  m->opt.sdf_initpoints = -5;
  EXPECT_EQ(mj_maxContact(m.get(), sphere, sdf, 0), 0);
}

TEST_F(MjCollisionTest, FlexBVHPruningIncludesFlexGap) {
  // flex : flex intermediate BVH pruning includes flex_gap.
  static constexpr char kFlexFlexXml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="vol1" type="grid" dim="3" count="4 4 4"
                spacing="0.06 0.06 0.06" pos="0 0 0" radius="0.005" mass="1">
        <contact selfcollide="none" gap="0.05"/>
        <edge equality="true"/>
      </flexcomp>
      <flexcomp name="vol2" type="grid" dim="3" count="4 4 4"
                spacing="0.06 0.06 0.06" pos="0 0 0.22" radius="0.005" mass="1">
        <contact selfcollide="none" gap="0.05"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(kFlexFlexXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  m->opt.disableflags |= mjDSBL_MIDPHASE;
  MjDataPtr d_all = MakeData(m);
  mj_forward(m.get(), d_all.get());
  ASSERT_GT(d_all->ncon, 0);

  m->opt.disableflags &= ~mjDSBL_MIDPHASE;
  MjDataPtr d_bvh = MakeData(m);
  mj_forward(m.get(), d_bvh.get());
  EXPECT_EQ(d_bvh->ncon, d_all->ncon);

  // body : flex intermediate BVH pruning includes flex_gap.
  static constexpr char kBodyFlexXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 -0.15">
        <geom type="sphere" pos="-0.03 0 0.03" size="0.01"/>
        <geom type="sphere" pos="0.03 0 -0.03" size="0.01"/>
      </body>
      <flexcomp name="vol" type="grid" dim="3" count="4 4 4"
                spacing="0.06 0.06 0.06" pos="0 0 0" radius="0.005" mass="1">
        <contact selfcollide="none" gap="0.05"/>
        <edge equality="true"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr mb = LoadModelFromString(kBodyFlexXml, error, sizeof(error));
  ASSERT_THAT(mb.get(), NotNull()) << error;

  mb->opt.disableflags |= mjDSBL_MIDPHASE;
  MjDataPtr db_all = MakeData(mb);
  mj_forward(mb.get(), db_all.get());
  ASSERT_GT(db_all->ncon, 0);

  mb->opt.disableflags &= ~mjDSBL_MIDPHASE;
  MjDataPtr db_bvh = MakeData(mb);
  mj_forward(mb.get(), db_bvh.get());
  EXPECT_EQ(db_bvh->ncon, db_all->ncon);
}

TEST_F(MjCollisionTest, PlaneAndSdfFlexRespectGeomBitmaskInMidphase) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" vertex="-0.5 -0.5 -0.5  0.5 -0.5 -0.5  -0.5 0.5 -0.5  0.5 0.5 -0.5
                                -0.5 -0.5  0.5  0.5 -0.5  0.5  -0.5 0.5  0.5  0.5 0.5  0.5"/>
    </asset>
    <worldbody>
      <geom name="far_sphere" type="sphere" size="0.01" pos="10 0 0"
            contype="1" conaffinity="1"/>
      <geom name="plane_filtered" type="plane" size="1 1 0.1" pos="0 0 0.05"
            contype="0" conaffinity="0"/>
      <geom name="sdf_filtered" type="sdf" mesh="cube" pos="0 0 -0.445"
            contype="2" conaffinity="2"/>
      <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                dim="2" pos="0 0 0.053" radius=".005">
        <edge equality="true"/>
        <contact selfcollide="none" contype="1" conaffinity="1"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  MjDataPtr d_bvh = MakeData(m);
  mj_fwdPosition(m.get(), d_bvh.get());
  EXPECT_EQ(d_bvh->ncon, 0);

  m->opt.disableflags |= mjDSBL_MIDPHASE;
  MjDataPtr d_all = MakeData(m);
  mj_fwdPosition(m.get(), d_all.get());
  EXPECT_EQ(d_all->ncon, 0);
}

// Exclude two-pointer merge with out-of-order excludes, midphase, and <pair>.
TEST_F(MjCollisionTest, ExcludeMergeAndPairOverride) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="gw" size="0.3"/>
      <replicate count="20" offset="0.001 0 0">
        <body name="r">
          <freejoint/>
          <geom name="gr" size="0.3" contype="2" conaffinity="0"/>
        </body>
      </replicate>
      <body name="a" pos="0.01 0 0">
        <freejoint/>
        <geom name="ga" size="0.3"/>
      </body>
      <body name="b" pos="0.02 0 0">
        <freejoint/>
        <geom name="gb0" size="0.3"/>
        <geom name="gb1" pos="0 0 0.01" size="0.3"/>
      </body>
    </worldbody>
    <contact>
      <exclude body1="b" body2="a"/>
      <exclude body1="b" body2="r10"/>
      <exclude body1="a" body2="world"/>
      <exclude body1="r05" body2="world"/>
      <pair geom1="gb1" geom2="ga"/>
    </contact>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  for (int disable : {0, static_cast<int>(mjDSBL_MIDPHASE)}) {
    m->opt.disableflags = disable;
    mj_forward(m.get(), d.get());
    EXPECT_THAT(colliding_pairs(m.get(), d.get()),
                ElementsAre(GeomPair("ga", "gb1"), GeomPair("gb0", "gw"),
                            GeomPair("gb1", "gw")));
  }
}

// Narrowphase skips CCD stack buffers unless convex or custom colliders run.
TEST_F(MjCollisionTest, NarrowphaseCCDBuffer) {
  static constexpr char sphere_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="20" offset="0.01 0 0">
        <body>
          <freejoint/>
          <geom type="sphere" size="0.2"/>
        </body>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  static constexpr char ellipsoid_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="20" offset="0.01 0 0">
        <body>
          <freejoint/>
          <geom type="ellipsoid" size="0.2 0.2 0.2"/>
        </body>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  auto max_stack = [](const char* xml, int workers, int disable = 0) {
    char error[1024];
    MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
    EXPECT_THAT(m.get(), NotNull()) << error;
    m->opt.disableflags = disable;
    MjDataPtr d = MakeData(m);
    mju_threadpool(d.get(), workers);
    mj_forward(m.get(), d.get());
    d->maxuse_stack = 0;
    mj_collision(m.get(), d.get());
    EXPECT_EQ(d->ncon, 190);
    return static_cast<ptrdiff_t>(d->maxuse_stack);
  };

  ptrdiff_t prim = max_stack(sphere_xml, 4);
  ptrdiff_t convex = max_stack(ellipsoid_xml, 4);
  ptrdiff_t ccd_buf = max_stack(ellipsoid_xml, 0) - max_stack(sphere_xml, 0);
  ASSERT_GT(ccd_buf, 0);
  EXPECT_LE(prim, max_stack(ellipsoid_xml, 4, mjDSBL_NATIVECCD));
  EXPECT_LT(prim - max_stack(sphere_xml, 0), ccd_buf / 10);
  EXPECT_GE(convex - prim, 4 * ccd_buf);

  static mjfCollision ccd = mjCOLLISIONFUNC[mjGEOM_ELLIPSOID][mjGEOM_ELLIPSOID];
  mjfCollision orig = mjCOLLISIONFUNC[mjGEOM_SPHERE][mjGEOM_SPHERE];
  mjCOLLISIONFUNC[mjGEOM_SPHERE][mjGEOM_SPHERE] =
      [](const mjModel* m, mjData* d, mjPreContact* c, int g1, int g2,
         mjtNum margin) { return ccd(m, d, c, g1, g2, margin); };
  ptrdiff_t custom = max_stack(sphere_xml, 4);
  mjCOLLISIONFUNC[mjGEOM_SPHERE][mjGEOM_SPHERE] = orig;
  EXPECT_NEAR(custom, convex, ccd_buf / 4);
  EXPECT_LT(custom - prim, 10 * ccd_buf);
}

}  // namespace
}  // namespace mujoco
