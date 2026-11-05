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

// Tests for xml/xml_api.cc.

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

// ---------------------------- test capsule --------------------------------

TEST_F(MujocoTest, ReadsCapsule) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <link name="torso">
    <collision>
      <origin rpy="-1.57080 0.00000 0.00000" xyz="0.00000 0.00000 0.00000"/>
      <geometry>
        <capsule length="0.14000" radius="0.07000"/>
      </geometry>
    </collision>
  </link>
  </robot>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
}

TEST_F(MujocoTest, ReadsGeomNames) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <mujoco>
    <compiler discardvisual="false"/>
  </mujoco>

  <link name="torso">
    <collision name="collision_box">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.2 0.3"/>
      </geometry>
    </collision>
    <visual name="visual_sphere">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.5"/>
      </geometry>
    </visual>
  </link>
  </robot>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // Check the geoms have been loaded with the right names
  int collision_box_id =
      mj_name2id(model.get(), mjtObj::mjOBJ_GEOM, "collision_box");
  ASSERT_GE(collision_box_id, 0);
  EXPECT_EQ(model->geom_type[collision_box_id], mjtGeom::mjGEOM_BOX);

  int visual_sphere_id =
      mj_name2id(model.get(), mjtObj::mjOBJ_GEOM, "visual_sphere");
  ASSERT_GE(visual_sphere_id, 0);
  EXPECT_EQ(model->geom_type[visual_sphere_id], mjtGeom::mjGEOM_SPHERE);
}

TEST_F(MujocoTest, CanLoadUrdfWithNonUniqueNamesCollisionBeforeVisual) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <mujoco>
    <compiler discardvisual="false"/>
  </mujoco>

  <link name="torso">
    <collision name="shared_name">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.2 0.3"/>
      </geometry>
    </collision>
    <visual name="shared_name">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.5"/>
      </geometry>
    </visual>
  </link>
  </robot>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // Check the collision geom gets its name from the URDF. The visual sphere
  // should not have a name to avoid duplicates.
  int collision_box_id =
      mj_name2id(model.get(), mjtObj::mjOBJ_GEOM, "shared_name");
  ASSERT_GE(collision_box_id, 0);
  EXPECT_EQ(model->geom_type[collision_box_id], mjtGeom::mjGEOM_BOX);
}

TEST_F(MujocoTest, CanLoadUrdfWithNonUniqueNamesVisualBeforeCollision) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <mujoco>
    <compiler discardvisual="false"/>
  </mujoco>

  <link name="torso">
    <visual name="shared_name">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <sphere radius="0.5"/>
      </geometry>
    </visual>
    <collision name="shared_name">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <box size="0.1 0.2 0.3"/>
      </geometry>
    </collision>
  </link>
  </robot>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // Check the visual geom gets its name from the URDF. The collision geom
  // should not have a name to avoid duplicates.
  int visual_sphere_id =
      mj_name2id(model.get(), mjtObj::mjOBJ_GEOM, "shared_name");
  ASSERT_GE(visual_sphere_id, 0);
  EXPECT_EQ(model->geom_type[visual_sphere_id], mjtGeom::mjGEOM_SPHERE);
}

TEST_F(MujocoTest, ReadsJointTypes) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <mujoco>
    <compiler discardvisual="false"/>
  </mujoco>

  <link name="world"/>
  <joint type="floating" name="floating">
      <parent link="world"/>
      <child link="link1"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
  </joint>
  <link name="link1">
      <collision>
        <geometry>
          <box size="0.1 0.2 0.3"/>
        </geometry>
      </collision>
  </link>

  <joint type="revolute" name="revolute">
      <parent link="world"/>
      <child link="link2"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <axis xyz="1.0 0.0 -1.0"/>
  </joint>
  <link name="link2">
      <collision>
        <geometry>
          <box size="0.1 0.2 0.3"/>
        </geometry>
      </collision>
  </link>

  <joint type="spherical" name="spherical">
      <parent link="world"/>
      <child link="link3"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
  </joint>
  <link name="link3">
      <collision>
        <geometry>
          <box size="0.1 0.2 0.3"/>
        </geometry>
      </collision>
  </link>

  <joint type="prismatic" name="prismatic">
      <parent link="world"/>
      <child link="link4"/>
      <origin rpy="0 0 0" xyz="0.0 0.0 0.0"/>
      <axis xyz="0.0 0.0 1.0"/>
  </joint>
  <link name="link4">
      <collision>
        <geometry>
          <box size="0.1 0.2 0.3"/>
        </geometry>
      </collision>
  </link>

  </robot>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  constexpr float eps = 1e-6;

  std::vector<std::string> joint_names = {"floating", "revolute", "spherical",
                                          "prismatic"};
  std::vector<mjtJoint> expected_joint_types = {
      mjtJoint::mjJNT_FREE, mjtJoint::mjJNT_HINGE, mjtJoint::mjJNT_BALL,
      mjtJoint::mjJNT_SLIDE};
  std::vector<std::vector<float>> expected_axis = {{0.0, 0.0, 1.0},
                                                   {0.707107, 0.0, -0.707107},
                                                   {0.0, 0.0, 1.0},
                                                   {0.0, 0.0, 1.0}};
  for (int i = 0; i < joint_names.size(); ++i) {
    int id =
        mj_name2id(model.get(), mjtObj::mjOBJ_JOINT, joint_names[i].c_str());
    EXPECT_EQ(model->jnt_type[id], expected_joint_types[i]);
    EXPECT_NEAR(model->jnt_axis[3 * id], expected_axis[i][0], eps);
    EXPECT_NEAR(model->jnt_axis[3 * id + 1], expected_axis[i][1], eps);
    EXPECT_NEAR(model->jnt_axis[3 * id + 2], expected_axis[i][2], eps);
  }
}

// ---------------------------- inertial orientation ---------------------------

TEST_F(MujocoTest, UrdfInertialOriginRotation) {
  // the inertial origin rotation must not be lost during compilation,
  // for both the default (alignfree) and non-aligned paths
  static constexpr char urdf[] = R"(
  <robot name="inertia_frame">
    <mujoco><compiler fusestatic="false"/></mujoco>
    <link name="world"/>
    <link name="body">
      <inertial>
        <origin xyz="0 0 0" rpy="0 0 1.5707963267948966"/>
        <mass value="1"/>
        <inertia ixx="1" iyy="2" izz="2.5" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <joint name="free" type="floating">
      <parent link="world"/>
      <child link="body"/>
    </joint>
  </robot>
  )";

  for (bool alignfree : {true, false}) {
    std::array<char, 1000> error;
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parseXMLString(urdf, nullptr, error.data(), error.size()),
        mj_deleteSpec);
    ASSERT_THAT(spec.get(), NotNull()) << error.data();
    spec->compiler.alignfree = alignfree;
    MjModelPtr model(mj_compile(spec.get(), nullptr));
    ASSERT_THAT(model.get(), NotNull()) << mjs_getError(spec.get());

    int body_id = mj_name2id(model.get(), mjtObj::mjOBJ_BODY, "body");

    // world-frame inertia tensor must be diag(2, 1, 2.5)
    mjtNum rot_body[9], mat[9], rot_world[9], tensor[9];
    mju_quat2Mat(rot_body, model->body_iquat + 4 * body_id);
    mju_quat2Mat(mat, model->body_quat + 4 * body_id);
    mju_mulMatMat(rot_world, mat, rot_body, 3, 3, 3);
    for (int r = 0; r < 3; r++) {
      for (int c = 0; c < 3; c++) {
        mjtNum sum = 0;
        for (int k = 0; k < 3; k++) {
          sum += rot_world[3 * r + k] * rot_world[3 * c + k] *
                 model->body_inertia[3 * body_id + k];
        }
        tensor[3 * r + c] = sum;
      }
    }
    constexpr double diag2[9] = {2, 0, 0, 0, 1, 0, 0, 0, 2.5};
    for (int i = 0; i < 9; i++) {
      EXPECT_THAT(tensor[i], MjNear(diag2[i], 1e-7, 1e-5))
          << "alignfree=" << alignfree;
    }

    // X torque for 1 rad/s^2 must equal the x principal moment R*D*R'[0][0]
    MjDataPtr data = MakeData(model);
    data->qacc[3] = 1.0;
    mj_inverse(model.get(), data.get());
    mjtNum expected = 0;
    for (int k = 0; k < 3; k++) {
      expected +=
          rot_body[k] * rot_body[k] * model->body_inertia[3 * body_id + k];
    }
    EXPECT_THAT(data->qfrc_inverse[3], MjNear(expected, 1e-7, 1e-5))
        << "alignfree=" << alignfree;
  }
}

TEST_F(MujocoTest, UrdfObliqueInertia) {
  // q = (1, 2, 3, 4) / sqrt(30), so R = [-10 2 11; 10 -5 10; 5 14 2] / 15.
  // For I = [4 .3 .2; .3 3 .1; .2 .1 2], these are the exact entries of R I R'.
  // I is positive definite, with every eigenvalue less than trace(I)/2.
  constexpr mjtNum expected[9] = {
      1004.0 / 375, -127.0 / 150, -49.0 / 125, -127.0 / 150, 3,
      61.0 / 150,   -49.0 / 125,  61.0 / 150,  1246.0 / 375};
  static constexpr char urdf[] = R"(
  <robot name="oblique">
    <mujoco><compiler fusestatic="false"/></mujoco>
    <link name="world"/>
    <link name="body">
      <inertial>
        <origin rpy="1.4288992721907328 -0.3398369094541219 2.356194490192345"/>
        <mass value="1"/>
        <inertia ixx="4" iyy="3" izz="2" ixy="0.3" ixz="0.2" iyz="0.1"/>
      </inertial>
    </link>
    <joint name="free" type="floating">
      <parent link="world"/>
      <child link="body"/>
    </joint>
  </robot>
  )";

  // World axes include the frame change made by alignfree.
  auto check_inertia = [&](const MjModelPtr& model) {
    int body = mj_name2id(model.get(), mjOBJ_BODY, "body");
    ASSERT_GE(body, 0);
    MjDataPtr data = MakeData(model);
    mj_forward(model.get(), data.get());
    const mjtNum* rot = data->ximat + 9 * body;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        mjtNum inertia = 0;
        for (int k = 0; k < 3; ++k) {
          inertia += rot[3 * r + k] * model->body_inertia[3 * body + k] *
                     rot[3 * c + k];
        }
        EXPECT_THAT(inertia, MjNear(expected[3 * r + c], 1e-7, 1e-5));
      }
    }
  };

  for (bool alignfree : {false, true}) {
    SCOPED_TRACE(alignfree);
    std::array<char, 1000> error{};
    std::unique_ptr<mjSpec, decltype(&mj_deleteSpec)> spec(
        mj_parseXMLString(urdf, nullptr, error.data(), error.size()),
        mj_deleteSpec);
    ASSERT_THAT(spec.get(), NotNull()) << error.data();
    spec->compiler.alignfree = alignfree;
    MjModelPtr model(mj_compile(spec.get(), nullptr));
    ASSERT_THAT(model.get(), NotNull()) << mjs_getError(spec.get());
    check_inertia(model);
  }
}

TEST_F(MujocoTest, RepeatedMeshName) {
  static constexpr char urdf[] = R"(
  <robot name="">
  <mujoco>
    <compiler discardvisual="false"/>
  </mujoco>

  <link name="geom1">
    <visual name="vis1">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh.obj" scale="1 1 1"/>
      </geometry>
    </visual>
  </link>

  <link name="geom2">
    <visual name="vis2">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh.obj" scale="2 2 2"/>
      </geometry>
    </visual>
  </link>

  <link name="geom3">
    <visual name="vis3">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh.obj" scale="3 3 3"/>
      </geometry>
    </visual>
  </link>

  <link name="geom4">
    <visual name="vis4">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="mesh.obj" scale="2 2 2"/>
      </geometry>
    </visual>
  </link>
  </robot>
  )";

  std::array<char, 1000> error;
  mjSpec* spec = mj_parseXMLString(urdf, 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();

  mjsMesh* mesh = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh"));
  mjsMesh* mesh1 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh1"));
  mjsMesh* mesh2 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh2"));
  mjsMesh* mesh3 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh3"));
  EXPECT_THAT(mesh, NotNull());
  EXPECT_THAT(mesh1, NotNull());
  EXPECT_THAT(mesh2, NotNull());
  EXPECT_THAT(mesh3, IsNull());
  EXPECT_STREQ(mjs_getName(mesh->element)->c_str(), "mesh");
  EXPECT_STREQ(mjs_getName(mesh1->element)->c_str(), "mesh1");
  EXPECT_STREQ(mjs_getName(mesh2->element)->c_str(), "mesh2");

  mjsGeom* vis1 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis1"));
  mjsGeom* vis2 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis2"));
  mjsGeom* vis3 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis3"));
  mjsGeom* vis4 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis4"));
  EXPECT_STREQ(mjs_getString(vis1->meshname), "mesh");
  EXPECT_STREQ(mjs_getString(vis2->meshname), "mesh1");
  EXPECT_STREQ(mjs_getString(vis3->meshname), "mesh2");
  EXPECT_STREQ(mjs_getString(vis4->meshname), "mesh1");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, MimicBecomesJointEquality) {
  // gripper with mirrored fingers: right_joint = -1 * left_joint + 0.02. the
  // target joint is declared after the joint that mimics it, on purpose
  static constexpr char urdf[] = R"(
  <robot name="gripper">
    <mujoco><option gravity="0 0 0"/></mujoco>
    <link name="base">
      <inertial><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <link name="left_finger">
      <inertial><mass value="0.1"/>
        <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <link name="right_finger">
      <inertial><mass value="0.1"/>
        <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <joint name="right_joint" type="prismatic">
      <parent link="base"/><child link="right_finger"/><axis xyz="1 0 0"/>
      <dynamics damping="1"/>
      <mimic joint="left_joint" multiplier="-1" offset="0.02"/>
    </joint>
    <joint name="left_joint" type="prismatic">
      <parent link="base"/><child link="left_finger"/><axis xyz="1 0 0"/>
      <dynamics damping="1"/>
    </joint>
  </robot>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  ASSERT_EQ(model->neq, 1);
  EXPECT_EQ(model->eq_type[0], mjEQ_JOINT);
  int right = mj_name2id(model.get(), mjOBJ_JOINT, "right_joint");
  int left = mj_name2id(model.get(), mjOBJ_JOINT, "left_joint");
  EXPECT_EQ(model->eq_obj1id[0], right);
  EXPECT_EQ(model->eq_obj2id[0], left);
  EXPECT_MJTNUM_EQ(model->eq_data[0], 0.02);  // offset
  EXPECT_EQ(model->eq_data[1], -1.0);         // multiplier

  // simulate from a driver displacement and check the follower tracks
  // q_right = -1 * q_left + 0.02
  MjDataPtr data = MakeData(model);
  data->qpos[model->jnt_qposadr[left]] = 0.03;
  while (data->time < 2) {
    mj_step(model.get(), data.get());
  }
  mjtNum q_left = data->qpos[model->jnt_qposadr[left]];
  mjtNum q_right = data->qpos[model->jnt_qposadr[right]];
  EXPECT_NEAR(q_right, -1.0 * q_left + 0.02, 1e-4);
}

TEST_F(MujocoTest, MimicTargetingIncompatibleJointIsIgnored) {
  // a <mimic> pointing at a fixed joint used to load with the mimic dropped;
  // it must still load rather than turn into a hard failure
  static constexpr char urdf[] = R"(
  <robot name="robot">
    <link name="base">
      <inertial><mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <link name="a">
      <inertial><mass value="0.1"/>
        <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <link name="b">
      <inertial><mass value="0.1"/>
        <inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/></inertial>
    </link>
    <joint name="mover" type="prismatic">
      <parent link="base"/><child link="a"/><axis xyz="1 0 0"/>
      <mimic joint="welded"/>
    </joint>
    <joint name="welded" type="fixed">
      <parent link="a"/><child link="b"/>
    </joint>
  </robot>)";

  std::array<char, 1024> error;
  mock_warning_handler.ExpectWarnings("<mimic> on joint 'mover' ignored");
  MjModelPtr model = LoadModelFromString(urdf, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->neq, 0);
}

TEST_F(MujocoTest, MeshesWithSameBasenameDifferentDirectories) {
  static constexpr char urdf[] = R"(
  <robot name="robot">
    <mujoco>
      <compiler discardvisual="false"/>
    </mujoco>
    <link name="geom1">
      <visual name="vis1">
        <geometry>
          <mesh filename="dir1/mesh.stl" scale="1 1 1"/>
        </geometry>
      </visual>
    </link>
    <link name="geom2">
      <visual name="vis2">
        <geometry>
          <mesh filename="dir2/mesh.stl" scale="1 1 1"/>
        </geometry>
      </visual>
    </link>
    <link name="geom3">
      <visual name="vis3">
        <geometry>
          <mesh filename="dir2/mesh.stl" scale="1 1 1"/>
        </geometry>
      </visual>
    </link>
  </robot>
  )";

  std::array<char, 1000> error;
  mjSpec* spec = mj_parseXMLString(urdf, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjsMesh* mesh0 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh"));
  mjsMesh* mesh1 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh1"));
  mjsMesh* mesh2 = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "mesh2"));
  ASSERT_THAT(mesh0, NotNull());
  ASSERT_THAT(mesh1, NotNull());
  EXPECT_THAT(mesh2, IsNull());
  EXPECT_STREQ(mjs_getString(mesh0->file), "dir1/mesh.stl");
  EXPECT_STREQ(mjs_getString(mesh1->file), "dir2/mesh.stl");

  mjsGeom* vis1 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis1"));
  mjsGeom* vis2 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis2"));
  mjsGeom* vis3 = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "vis3"));
  ASSERT_THAT(vis1, NotNull());
  ASSERT_THAT(vis2, NotNull());
  ASSERT_THAT(vis3, NotNull());
  EXPECT_STREQ(mjs_getString(vis1->meshname), "mesh");
  EXPECT_STREQ(mjs_getString(vis2->meshname), "mesh1");
  EXPECT_STREQ(mjs_getString(vis3->meshname), "mesh1");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, UnnamedGeomAfterNamedVisualAndMaterialLookup) {
  static constexpr char urdf[] = R"(
  <robot name="robot">
    <mujoco>
      <compiler discardvisual="false"/>
    </mujoco>
    <material name="blue">
      <color rgba="0 0 1 1"/>
    </material>
    <link name="base">
      <visual name="vis1">
        <material name="blue"/>
        <geometry><sphere radius="0.1"/></geometry>
      </visual>
      <collision>
        <geometry><box size="0.1 0.1 0.1"/></geometry>
      </collision>
      <visual>
        <geometry><sphere radius="0.2"/></geometry>
      </visual>
    </link>
  </robot>
  )";

  std::array<char, 1000> error;
  testing::internal::CaptureStderr();
  mjSpec* spec = mj_parseXMLString(urdf, nullptr, error.data(), error.size());
  std::string stderr_output = testing::internal::GetCapturedStderr();
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(stderr_output, "");

  mjsBody* body = mjs_findBody(spec, "base");
  ASSERT_THAT(body, NotNull());

  mjsElement* elem1 = mjs_firstChild(body, mjOBJ_GEOM, 0);
  ASSERT_THAT(elem1, NotNull());
  mjsGeom* geom1 = mjs_asGeom(elem1);
  ASSERT_THAT(geom1, NotNull());
  EXPECT_STREQ(mjs_getString(mjs_getName(elem1)), "vis1");
  EXPECT_FLOAT_EQ(geom1->rgba[0], 0.0f);
  EXPECT_FLOAT_EQ(geom1->rgba[1], 0.0f);
  EXPECT_FLOAT_EQ(geom1->rgba[2], 1.0f);
  EXPECT_FLOAT_EQ(geom1->rgba[3], 1.0f);

  mjsElement* elem2 = mjs_nextChild(body, elem1, 0);
  ASSERT_THAT(elem2, NotNull());
  EXPECT_STREQ(mjs_getString(mjs_getName(elem2)), "");

  mjsElement* elem3 = mjs_nextChild(body, elem2, 0);
  ASSERT_THAT(elem3, NotNull());
  EXPECT_STREQ(mjs_getString(mjs_getName(elem3)), "");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, UrdfJointLimitsEqualPartialAndContinuous) {
  mock_warning_handler.ExpectWarnings(
      "joint 'j_eq' has equal lower and upper limits");
  static constexpr char urdf[] = R"(
  <robot name="test_robot">
    <link name="base">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <link name="link_eq">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <link name="link_partial">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <link name="link_cont">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <joint name="j_eq" type="revolute">
      <parent link="base"/>
      <child link="link_eq"/>
      <axis xyz="0 0 1"/>
      <limit lower="1.0" upper="1.0" effort="10" velocity="1"/>
    </joint>
    <joint name="j_partial" type="prismatic">
      <parent link="base"/>
      <child link="link_partial"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.5" effort="8" velocity="1"/>
    </joint>
    <joint name="j_cont" type="continuous">
      <parent link="base"/>
      <child link="link_cont"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.0" upper="1.0" effort="15" velocity="1"/>
    </joint>
  </robot>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(urdf, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjsJoint* j_eq_spec = mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "j_eq"));
  mjsJoint* j_partial_spec =
      mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "j_partial"));
  mjsJoint* j_cont_spec =
      mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "j_cont"));
  ASSERT_THAT(j_eq_spec, NotNull());
  ASSERT_THAT(j_partial_spec, NotNull());
  ASSERT_THAT(j_cont_spec, NotNull());
  EXPECT_EQ(j_eq_spec->limited, mjLIMITED_FALSE);
  EXPECT_EQ(j_partial_spec->limited, mjLIMITED_FALSE);
  EXPECT_EQ(j_cont_spec->limited, mjLIMITED_FALSE);
  EXPECT_DOUBLE_EQ(j_cont_spec->range[0], 0.0);
  EXPECT_DOUBLE_EQ(j_cont_spec->range[1], 0.0);

  MjModelPtr model(mj_compile(spec, nullptr));
  ASSERT_THAT(model.get(), NotNull()) << mjs_getError(spec);

  int j_eq = mj_name2id(model.get(), mjOBJ_JOINT, "j_eq");
  int j_partial = mj_name2id(model.get(), mjOBJ_JOINT, "j_partial");
  int j_cont = mj_name2id(model.get(), mjOBJ_JOINT, "j_cont");
  ASSERT_GE(j_eq, 0);
  ASSERT_GE(j_partial, 0);
  ASSERT_GE(j_cont, 0);
  EXPECT_EQ(model->jnt_limited[j_eq], 0);
  EXPECT_EQ(model->jnt_actfrclimited[j_eq], 1);
  EXPECT_EQ(model->jnt_limited[j_partial], 0);
  EXPECT_EQ(model->jnt_actfrclimited[j_partial], 1);
  EXPECT_EQ(model->jnt_limited[j_cont], 0);
  EXPECT_EQ(model->jnt_actfrclimited[j_cont], 1);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, UrdfJointLimitsEqualAutolimitsFalse) {
  // with autolimits=false, an equal-bounds limit used to fail compilation;
  // effort is omitted since it would also need actuatorfrclimited
  mock_warning_handler.ExpectWarnings(
      "joint 'j_eq' has equal lower and upper limits");
  static constexpr char urdf[] = R"(
  <robot name="test_robot">
    <mujoco>
      <compiler autolimits="false"/>
    </mujoco>
    <link name="base">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <link name="link_eq">
      <inertial>
        <mass value="1"/>
        <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
      </inertial>
    </link>
    <joint name="j_eq" type="revolute">
      <parent link="base"/>
      <child link="link_eq"/>
      <axis xyz="0 0 1"/>
      <limit lower="1.0" upper="1.0" velocity="1"/>
    </joint>
  </robot>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(urdf, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(spec->compiler.autolimits, 0);

  MjModelPtr model(mj_compile(spec, nullptr));
  ASSERT_THAT(model.get(), NotNull()) << mjs_getError(spec);
  int j_eq = mj_name2id(model.get(), mjOBJ_JOINT, "j_eq");
  ASSERT_GE(j_eq, 0);
  EXPECT_EQ(model->jnt_limited[j_eq], 0);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, UrdfIndexedMeshNamesAreUnique) {
  // the second link1 mesh gets the indexed name link11, which must not clash
  // with a mesh whose own basename is link11, in either declaration order
  static constexpr char urdf_indexed_first[] = R"(
  <robot name="robot">
    <mujoco>
      <compiler discardvisual="false"/>
    </mujoco>
    <link name="a">
      <visual name="vis_a">
        <geometry><mesh filename="visual/link1.stl"/></geometry>
      </visual>
      <collision name="col_a">
        <geometry><mesh filename="collision/link1.stl"/></geometry>
      </collision>
    </link>
    <link name="b">
      <visual name="vis_b">
        <geometry><mesh filename="visual/link11.stl"/></geometry>
      </visual>
    </link>
  </robot>
  )";

  static constexpr char urdf_plain_first[] = R"(
  <robot name="robot">
    <mujoco>
      <compiler discardvisual="false"/>
    </mujoco>
    <link name="b">
      <visual name="vis_b">
        <geometry><mesh filename="visual/link11.stl"/></geometry>
      </visual>
    </link>
    <link name="a">
      <visual name="vis_a">
        <geometry><mesh filename="visual/link1.stl"/></geometry>
      </visual>
      <collision name="col_a">
        <geometry><mesh filename="collision/link1.stl"/></geometry>
      </collision>
    </link>
  </robot>
  )";

  for (const char* urdf : {urdf_indexed_first, urdf_plain_first}) {
    std::array<char, 1000> error;
    mjSpec* spec = mj_parseXMLString(urdf, nullptr, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << error.data();

    // three distinct meshes with distinct names
    std::vector<std::string> names;
    for (mjsElement* el = mjs_firstElement(spec, mjOBJ_MESH); el;
         el = mjs_nextElement(spec, el)) {
      names.push_back(mjs_getString(mjs_getName(el)));
    }
    ASSERT_EQ(names.size(), 3);
    EXPECT_NE(names[0], names[1]);
    EXPECT_NE(names[0], names[2]);
    EXPECT_NE(names[1], names[2]);

    // each geom references the mesh with its own file
    for (auto [geom, file] : {std::pair{"vis_a", "visual/link1.stl"},
                              std::pair{"col_a", "collision/link1.stl"},
                              std::pair{"vis_b", "visual/link11.stl"}}) {
      mjsGeom* g = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, geom));
      ASSERT_THAT(g, NotNull()) << geom;
      mjsMesh* m = mjs_asMesh(
          mjs_findElement(spec, mjOBJ_MESH, mjs_getString(g->meshname)));
      ASSERT_THAT(m, NotNull()) << geom;
      EXPECT_STREQ(mjs_getString(m->file), file) << geom;
    }

    mj_deleteSpec(spec);
  }
}

}  // namespace
}  // namespace mujoco
