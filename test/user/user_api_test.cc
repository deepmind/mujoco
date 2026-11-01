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

// Tests for user/user_api.cc.

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <functional>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/str_format.h"
#include <mujoco/mjplugin.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "src/user/user_api.h"
#include "src/xml/xml_api.h"
#include "test/compare_model.h"
#include "test/compare_spec.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsNull;
using ::testing::NotNull;

// -------------------------- test model manipulation  -------------------------

TEST_F(MujocoTest, GetSetData) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* body = mjs_addBody(world, 0);
  mjsSite* site = mjs_addSite(body, 0);

  {
    double vec[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    const char* str = "sitename";

    mjs_setName(site->element, str);
    mjs_setDouble(site->userdata, vec, 10);
  }

  EXPECT_THAT(mjs_getName(site->element)->c_str(), HasSubstr("sitename"));

  int nsize;
  const double* vec = mjs_getDouble(site->userdata, &nsize);
  for (int i = 0; i < nsize; ++i) {
    EXPECT_EQ(vec[i], i);
  }

  mj_deleteSpec(spec);
}
TEST_F(MujocoTest, TreeTraversal) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <site name="site1"/>
        <body name="body2">
          <site name="site4"/>
        </body>
        <geom name="geom1" size="1"/>
        <geom name="geom2" size="1"/>
        <site name="site2"/>
        <site name="site3"/>
        <geom name="geom3" size="1"/>
      </body>
      <body name="body3">
        <site name="site5"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();

  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* body1 = mjs_findBody(spec, "body1");
  mjsBody* body2 = mjs_findBody(spec, "body2");
  mjsBody* body3 = mjs_findBody(spec, "body3");
  mjsElement* b1 = body1->element;
  mjsElement* b2 = body2->element;
  mjsElement* b3 = body3->element;
  mjsElement* site1 = mjs_findElement(spec, mjOBJ_SITE, "site1");
  mjsElement* site2 = mjs_findElement(spec, mjOBJ_SITE, "site2");
  mjsElement* site3 = mjs_findElement(spec, mjOBJ_SITE, "site3");
  mjsElement* site4 = mjs_findElement(spec, mjOBJ_SITE, "site4");
  mjsElement* site5 = mjs_findElement(spec, mjOBJ_SITE, "site5");
  mjsElement* geom1 = mjs_findElement(spec, mjOBJ_GEOM, "geom1");
  mjsElement* geom2 = mjs_findElement(spec, mjOBJ_GEOM, "geom2");
  mjsElement* geom3 = mjs_findElement(spec, mjOBJ_GEOM, "geom3");

  // test nonexistent
  EXPECT_EQ(mjs_firstElement(spec, mjOBJ_ACTUATOR), nullptr);
  EXPECT_EQ(mjs_firstElement(spec, mjOBJ_LIGHT), nullptr);
  EXPECT_EQ(mjs_firstChild(body1, mjOBJ_CAMERA, /*recurse=*/true), nullptr);
  EXPECT_EQ(mjs_firstChild(body1, mjOBJ_TENDON, /*recurse=*/true), nullptr);

  // test first, nonrecursive
  EXPECT_EQ(mjs_firstElement(spec, mjOBJ_BODY), world->element);
  EXPECT_EQ(site1, mjs_firstElement(spec, mjOBJ_SITE));
  EXPECT_EQ(site1, mjs_firstChild(body1, mjOBJ_SITE, /*recurse=*/false));
  EXPECT_EQ(geom1, mjs_firstChild(body1, mjOBJ_GEOM, /*recurse=*/false));
  EXPECT_EQ(site4, mjs_firstChild(body2, mjOBJ_SITE, /*recurse=*/false));
  EXPECT_EQ(site5, mjs_firstChild(body3, mjOBJ_SITE, /*recurse=*/false));

  // test first, recursive
  EXPECT_EQ(site1, mjs_firstChild(world, mjOBJ_SITE, /*recurse=*/true));
  EXPECT_EQ(geom1, mjs_firstChild(world, mjOBJ_GEOM, /*recurse=*/true));
  EXPECT_EQ(site4, mjs_firstChild(body2, mjOBJ_SITE, /*recurse=*/true));
  EXPECT_EQ(site5, mjs_firstChild(body3, mjOBJ_SITE, /*recurse=*/true));

  // test next, nonrecursive
  EXPECT_EQ(site2, mjs_nextChild(body1, site1, /*recursive=*/false));
  EXPECT_EQ(site3, mjs_nextChild(body1, site2, /*recursive=*/false));
  EXPECT_EQ(nullptr, mjs_nextChild(body1, site3, /*recursive=*/false));
  EXPECT_EQ(geom2, mjs_nextChild(body1, geom1, /*recursive=*/false));
  EXPECT_EQ(geom3, mjs_nextChild(body1, geom2, /*recursive=*/false));
  EXPECT_EQ(nullptr, mjs_nextChild(body1, geom3, /*recursive=*/false));
  EXPECT_EQ(mjs_nextElement(spec, site1), site2);
  EXPECT_EQ(mjs_nextElement(spec, site2), site3);
  EXPECT_EQ(mjs_nextElement(spec, site3), site4);
  EXPECT_EQ(mjs_nextElement(spec, site4), site5);
  EXPECT_EQ(mjs_nextElement(spec, site5), nullptr);
  EXPECT_EQ(mjs_nextElement(spec, geom1), geom2);
  EXPECT_EQ(mjs_nextElement(spec, geom2), geom3);
  EXPECT_EQ(mjs_nextElement(spec, geom3), nullptr);

  // test next, recursive
  EXPECT_EQ(b2, mjs_nextChild(world, b1, /*recursive=*/true));
  EXPECT_EQ(b3, mjs_nextChild(world, b2, /*recursive=*/true));
  EXPECT_EQ(site2, mjs_nextChild(body1, site1, /*recursive=*/true));
  EXPECT_EQ(site3, mjs_nextChild(body1, site2, /*recursive=*/true));
  EXPECT_EQ(site4, mjs_nextChild(body1, site3, /*recursive=*/true));
  EXPECT_EQ(site4, mjs_nextChild(world, site3, /*recursive=*/true));
  EXPECT_EQ(site5, mjs_nextChild(world, site4, /*recursive=*/true));
  EXPECT_EQ(nullptr, mjs_nextChild(body1, site5, /*recursive=*/true));
  EXPECT_EQ(geom2, mjs_nextChild(body1, geom1, /*recursive=*/true));
  EXPECT_EQ(geom3, mjs_nextChild(body1, geom2, /*recursive=*/true));
  EXPECT_EQ(nullptr, mjs_nextChild(body1, geom3, /*recursive=*/true));

  // check compilation ordering of sites
  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_THAT(model, NotNull());
  EXPECT_EQ(mjs_getId(site1), 0);
  EXPECT_EQ(mjs_getId(site2), 1);
  EXPECT_EQ(mjs_getId(site3), 2);
  EXPECT_EQ(mjs_getId(site4), 3);
  EXPECT_EQ(mjs_getId(site5), 4);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, AttachAndChildDeletion) {
  mjSpec* child_spec = mj_makeSpec();
  mjsBody* child_world = mjs_findBody(child_spec, "world");
  mjsBody* child_body = mjs_addBody(child_world, 0);
  mjsJoint* freejoint = mjs_addJoint(child_body, 0);
  freejoint->type = mjJNT_FREE;
  mjs_setName(freejoint->element, "child_freejoint");

  mjSpec* parent_spec = mj_makeSpec();
  mjsBody* parent_world = mjs_findBody(parent_spec, "world");
  mjsBody* parent_body = mjs_addBody(parent_world, 0);

  // Attach child spec to parent_body
  mjsElement* attached =
      mjs_attach(parent_body->element, child_spec->element, "pre_", "");
  ASSERT_THAT(attached, NotNull());

  // Delete freejoint from child_spec, should fail because it is attached
  int result = mjs_delete(child_spec, freejoint->element);
  EXPECT_EQ(result, -1);

  // The freejoint should still be in parent_spec because deletion failed
  mjsElement* found_joint =
      mjs_findElement(parent_spec, mjOBJ_JOINT, "pre_child_freejoint");
  EXPECT_THAT(found_joint, NotNull());

  mj_deleteSpec(child_spec);
  mj_deleteSpec(parent_spec);
}

TEST_F(MujocoTest, OriginSpecInvariantToAttachment) {
  mjSpec* child_spec = mj_makeSpec();
  mjsBody* child_world = mjs_findBody(child_spec, "world");
  mjsBody* child_body = mjs_addBody(child_world, 0);
  mjsJoint* freejoint = mjs_addJoint(child_body, 0);
  freejoint->type = mjJNT_FREE;
  mjs_setName(freejoint->element, "child_freejoint");

  mjSpec* parent_spec = mj_makeSpec();
  mjsBody* parent_world = mjs_findBody(parent_spec, "world");
  mjsBody* parent_body = mjs_addBody(parent_world, 0);
  mjs_setName(parent_body->element, "parent_body");

  // Attach child spec to parent_body
  mjsElement* attached =
      mjs_attach(parent_body->element, child_spec->element, "pre_", "");
  ASSERT_THAT(attached, NotNull());

  // The freejoint should still be in parent_spec because deletion failed
  mjsElement* child_spec_joint =
      mjs_findElement(parent_spec, mjOBJ_JOINT, "pre_child_freejoint");
  EXPECT_EQ(mjs_getSpec(child_spec_joint), parent_spec);
  EXPECT_EQ(mjs_getOriginSpec(child_spec_joint), child_spec);

  mjsElement* parent_spec_body =
      mjs_findElement(parent_spec, mjOBJ_BODY, "parent_body");
  EXPECT_EQ(mjs_getOriginSpec(parent_spec_body), parent_spec);

  mj_deleteSpec(child_spec);
  mj_deleteSpec(parent_spec);
}

int open_mock(mjResource* resource) {
  static const char parent_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent_body"/>
    </worldbody>
  </mujoco>
  )";
  resource->data = mju_malloc(sizeof(parent_xml));
  absl::SNPrintF(static_cast<char*>(resource->data), sizeof(parent_xml), "%s",
                 parent_xml);
  return 1;
}

int read_mock(mjResource* resource, const void** buffer) {
  *buffer = resource->data;
  return std::strlen((const char*)resource->data);
}

void close_mock(mjResource* resource) {
  mju_free(resource->data);
  resource->data = nullptr;
}

TEST_F(MujocoTest, AttachedSpecDoesNotInheritURI) {
  // This test checks that when we attach a child spec to a parent spec that was
  // loaded from a resource provider, the child spec does not inherit the
  // resource URI from the parent. This allows the child spec to specify assets
  // relative to its model file or in the VFS.
  mjpResourceProvider provider = {
      .prefix = "fakeprovider",
      .open = open_mock,
      .read = read_mock,
      .close = close_mock,
  };

  mjp_registerProvider(&provider);

  std::array<char, 1024> err;
  mjSpec* parent_spec =
      mj_parseXML("fakeprovider:parent.xml", nullptr, err.data(), err.size());
  mjs_setString(parent_spec->modelname, "parent");
  ASSERT_THAT(parent_spec, NotNull()) << err.data();

  // Create child spec
  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="child_body">
        <geom type="mesh" mesh="asset"/>
      </body>
    </worldbody>
    <asset>
      <mesh name="asset" file="asset.obj"/>
    </asset>
  </mujoco>
  )";

  // Setup VFS with asset
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  static constexpr char asset_data[] = R"(
  v 0 0 0
  v 1 0 0
  v 0 1 0
  v 0 0 1
  f 1 2 3
  f 1 2 4
  f 2 3 4
  f 3 1 4
  )";
  mj_addBufferVFS(&vfs, "asset.obj", asset_data, sizeof(asset_data));

  mjSpec* child_spec =
      mj_parseXMLString(child_xml, &vfs, err.data(), err.size());
  mjs_setString(child_spec->modelname, "child");
  ASSERT_THAT(child_spec, NotNull()) << err.data();

  // Attach child spec to parent spec's world body
  mjsBody* world = mjs_findBody(parent_spec, "world");
  ASSERT_THAT(world, NotNull());

  mjsElement* attached =
      mjs_attach(world->element, child_spec->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  mjModel* model = mj_compile(parent_spec, &vfs);
  mj_deleteVFS(&vfs);

  EXPECT_THAT(model, NotNull()) << mjs_getError(parent_spec);

  if (model) {
    mj_deleteModel(model);
  }
  mj_deleteSpec(parent_spec);
  mj_deleteSpec(child_spec);
}

TEST_F(MujocoTest, ActivatePlugin) {
  mjSpec* spec = mj_makeSpec();
  mjs_activatePlugin(spec, "mujoco.elasticity.cable");

  // associate plugin to body
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjs_setString(body->plugin.plugin_name, "mujoco.elasticity.cable");
  body->plugin.element = mjs_addPlugin(spec)->element;
  body->plugin.active = true;
  mjsGeom* geom = mjs_addGeom(body, 0);
  geom->type = mjGEOM_BOX;
  geom->size[0] = 1;
  geom->size[1] = 1;
  geom->size[2] = 1;

  // compile and check that the plugin is present
  mjModel* model = mj_compile(spec, NULL);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nplugin, 1);
  EXPECT_THAT(model->body_plugin[1], 0);

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, DeletePlugin) {
  mjSpec* spec = mj_makeSpec();
  mjs_activatePlugin(spec, "mujoco.pid");

  // create body
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjsJoint* joint = mjs_addJoint(body, 0);
  mjsGeom* geom = mjs_addGeom(body, 0);
  mjs_setName(joint->element, "j1");
  joint->type = mjJNT_SLIDE;
  geom->size[0] = 1;

  // add actuator
  mjsActuator* actuator = mjs_addActuator(spec, 0);
  mjs_setString(actuator->target, "j1");
  mjs_setString(actuator->plugin.plugin_name, "mujoco.pid");
  actuator->plugin.element = mjs_addPlugin(spec)->element;
  actuator->plugin.active = true;
  actuator->trntype = mjTRN_JOINT;

  // compile and check that the plugin is present
  mjModel* model = mj_compile(spec, NULL);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nu, 1);
  EXPECT_THAT(model->nplugin, 1);
  EXPECT_THAT(model->actuator_plugin[0], 0);

  // delete actuator
  mjs_delete(spec, actuator->element);

  // recompile and check that the plugin is not present
  mjModel* newmodel = mj_compile(spec, NULL);
  EXPECT_THAT(newmodel, NotNull());
  EXPECT_THAT(newmodel->nu, 0);
  EXPECT_THAT(newmodel->nplugin, 0);

  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteModel(newmodel);
}

TEST_F(MujocoTest, SetToDCMotorNullable) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  double motorconst[2] = {0.05, 0.05};
  double resistance = 2.0;

  const char* err =
      mjs_setToDCMotor(actuator, motorconst, resistance, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, nullptr, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->gainprm[0], 2.0);
  EXPECT_EQ(actuator->gainprm[1], 0.05);
  EXPECT_EQ(actuator->gainprm[4], 0);
  EXPECT_EQ(actuator->gainprm[5], 0);
  EXPECT_EQ(actuator->gainprm[6], 0);
  EXPECT_EQ(actuator->dynprm[7], 0);
  EXPECT_EQ(actuator->dynprm[8], 0);

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToDCMotorDeriveKe) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  double resistance = 2.0;
  double nominal[3] = {12.0, 0, 100.0};  // vn=12, omega0=100

  const char* err =
      mjs_setToDCMotor(actuator, nullptr, resistance, nominal, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->gainprm[0], 2.0);
  EXPECT_NEAR(actuator->gainprm[1], 0.12, 1e-5);

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToDCMotorFull) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  double motorconst[2] = {0.05, 0.05};
  double resistance = 2.0;
  double saturation[3] = {1.0, 2.0, 3.0};
  double controller[6] = {10.0, 20.0, 30.0, 40.0, 50.0, 60.0};

  const char* err =
      mjs_setToDCMotor(actuator, motorconst, resistance, nullptr, saturation,
                       nullptr, nullptr, controller, nullptr, nullptr, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->gainprm[0], 2.0);   // resistance
  EXPECT_EQ(actuator->gainprm[1], 0.05);  // K
  EXPECT_EQ(actuator->gainprm[4], 10.0);  // kp
  EXPECT_EQ(actuator->gainprm[5], 20.0);  // ki
  EXPECT_EQ(actuator->gainprm[6], 30.0);  // kd
  EXPECT_EQ(actuator->dynprm[7], 40.0);   // slewmax
  EXPECT_EQ(actuator->dynprm[8], 50.0);   // Imax
  EXPECT_EQ(actuator->gainprm[7], 60.0);  // Vmax
  EXPECT_EQ(actuator->dynprm[1], 3.0);    // (di/dt)_max

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToDCMotorLuGre) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  double motorconst[2] = {0.05, 0.05};
  double resistance = 2.0;
  double lugre[5] = {100.0, 1.0, 0.5, 0.7, 10.0};

  const char* err =
      mjs_setToDCMotor(actuator, motorconst, resistance, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr, lugre, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->dynprm[5], 100.0);  // stiffness
  EXPECT_EQ(actuator->dynprm[6], 1.0);    // damping
  EXPECT_EQ(actuator->biasprm[3], 0.5);   // coulomb
  EXPECT_EQ(actuator->biasprm[4], 0.7);   // static
  EXPECT_EQ(actuator->biasprm[5], 10.0);  // stribeck

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToIntVelocityReportsErrors) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  // position servo errors are returned; timeconst is not an MJCF attribute
  double timeconst = -1.0;
  const char* err =
      mjs_setToIntVelocity(actuator, 5.0, nullptr, nullptr, &timeconst, 0);
  EXPECT_STREQ(err, "timeconst cannot be negative");

  // inheritrange sets actrange, so the two are exclusive
  actuator->actrange[1] = 1.0;
  err = mjs_setToIntVelocity(actuator, 5.0, nullptr, nullptr, nullptr, 1.0);
  EXPECT_STREQ(err, "actrange and inheritrange cannot both be defined");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToOrientation) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  // kv variant, default (expmap) chart
  double kv = 2.0;
  const char* err = mjs_setToOrientation(actuator, 5.0, &kv, nullptr, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->gaintype, mjGAIN_SO3);
  EXPECT_EQ(actuator->biastype, mjBIAS_SO3);
  EXPECT_EQ(actuator->dyntype, mjDYN_NONE);
  EXPECT_EQ(actuator->gainprm[0], 5.0);
  EXPECT_EQ(actuator->biasprm[1], -5.0);
  EXPECT_EQ(actuator->biasprm[2], -2.0);
  EXPECT_EQ(actuator->ctrlspec, 0);

  // dampratio variant, quat chart
  double dampratio = 1.0;
  err = mjs_setToOrientation(actuator, 5.0, nullptr, &dampratio, mjCHART_QUAT);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->biasprm[2], 1.0);
  EXPECT_EQ(actuator->ctrlspec, mjCHART_QUAT);

  // kv and dampratio are mutually exclusive
  err = mjs_setToOrientation(actuator, 5.0, &kv, &dampratio, 0);
  EXPECT_STREQ(err, "kv and dampratio cannot both be defined");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToPID) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);

  // stateless PID with kv, default input signature
  double kv = 3.0;
  const char* err = mjs_setToPID(actuator, 5.0, &kv, nullptr, nullptr, nullptr,
                                 nullptr, 0, 0);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->gaintype, mjGAIN_PID);
  EXPECT_EQ(actuator->biastype, mjBIAS_AFFINE);
  EXPECT_EQ(actuator->dyntype, mjDYN_NONE);
  EXPECT_EQ(actuator->biasprm[1], -5.0);
  EXPECT_EQ(actuator->biasprm[2], -3.0);
  EXPECT_EQ(actuator->gainprm[0], 0.0);

  // integral action with anti-windup, pos-only signature
  double ki = 0.5, imax = 2.0, dampratio = 1.0;
  err = mjs_setToPID(actuator, 5.0, nullptr, &dampratio, &ki, &imax, nullptr, 0,
                     mjINPUT_POS);
  EXPECT_STREQ(err, "");
  EXPECT_EQ(actuator->dyntype, mjDYN_PID);
  EXPECT_EQ(actuator->gainprm[0], 0.5);
  EXPECT_EQ(actuator->dynprm[0], 2.0);
  EXPECT_EQ(actuator->biasprm[2], 1.0);
  EXPECT_EQ(actuator->ctrlspec, mjINPUT_POS);

  // kv and dampratio are mutually exclusive
  err = mjs_setToPID(actuator, 5.0, &kv, &dampratio, nullptr, nullptr, nullptr,
                     0, 0);
  EXPECT_STREQ(err, "kv and dampratio cannot both be defined");

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToDeclaresInput) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);
  double timeconst[2] = {-1, -1};
  double range[2] = {-1, -1};

  // setpoint and pressure shortcuts declare an input, command ones clear it
  mjs_setToPosition(actuator, 1, nullptr, nullptr, nullptr, 0);
  EXPECT_EQ(actuator->ctrlspec, mjINPUT_POS);
  mjs_setToMotor(actuator);
  EXPECT_EQ(actuator->ctrlspec, 0);
  mjs_setToVelocity(actuator, 1);
  EXPECT_EQ(actuator->ctrlspec, mjINPUT_VEL);
  mjs_setToDamper(actuator, 1);
  EXPECT_EQ(actuator->ctrlspec, 0);
  mjs_setToIntVelocity(actuator, 1, nullptr, nullptr, nullptr, 0);
  EXPECT_EQ(actuator->ctrlspec, mjINPUT_VEL);
  mjs_setToMuscle(actuator, timeconst, 0, range, -1, -1, -1, -1, -1, -1, -1);
  EXPECT_EQ(actuator->ctrlspec, 0);
  mjs_setToCylinder(actuator, 1, 0, 1, -1);
  EXPECT_EQ(actuator->ctrlspec, mjINPUT_PRESSURE);
  mjs_setToAdhesion(actuator, 1);
  EXPECT_EQ(actuator->ctrlspec, 0);

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToOwnsControlModel) {
  mjSpec* spec = mj_makeSpec();
  mjsActuator* actuator = mjs_addActuator(spec, 0);
  EXPECT_EQ(actuator->type, mjACTUATOR_GENERAL);

  // a position servo with damping and filter dynamics
  double kv = 10, timeconst = 0.1;
  mjs_setToPosition(actuator, 100, &kv, nullptr, &timeconst, 0);
  EXPECT_EQ(actuator->type, mjACTUATOR_POSITION);
  EXPECT_EQ(actuator->dyntype, mjDYN_FILTEREXACT);

  // a motor set afterwards is a plain motor
  mjs_setToMotor(actuator);
  EXPECT_EQ(actuator->type, mjACTUATOR_MOTOR);
  EXPECT_EQ(actuator->gainprm[0], 1);
  EXPECT_EQ(actuator->biasprm[1], 0);
  EXPECT_EQ(actuator->biasprm[2], 0);
  EXPECT_EQ(actuator->biastype, mjBIAS_NONE);
  EXPECT_EQ(actuator->dyntype, mjDYN_NONE);
  EXPECT_EQ(actuator->dynprm[0], 1);

  // intvelocity is its own type
  mjs_setToIntVelocity(actuator, 1, nullptr, nullptr, nullptr, 0);
  EXPECT_EQ(actuator->type, mjACTUATOR_INTVELOCITY);

  mj_deleteSpec(spec);
}

static constexpr char xml_plugin_1[] = R"(
  <mujoco model="MuJoCo Model">
    <worldbody>
      <body name="body"/>
    </worldbody>
  </mujoco>)";

static constexpr char xml_plugin_2[] = R"(
  <mujoco model="MuJoCo Model">
    <extension>
      <plugin plugin="mujoco.pid">
        <instance name="actuator-1">
          <config key="ki" value="4.0"/>
          <config key="slewmax" value="3.14159"/>
        </instance>
      </plugin>
    </extension>
    <worldbody>
      <body name="empty"/>
      <body name="body">
        <joint name="joint"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <plugin name="actuator-1" plugin="mujoco.pid" instance="actuator-1"
              joint="joint" actdim="2"/>
    </actuator>
  </mujoco>)";

TEST_F(MujocoTest, AttachPlugin) {
  std::array<char, 1000> err;
  mjSpec* parent = mj_parseXMLString(xml_plugin_1, 0, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();
  mjSpec* spec_1 = mj_parseXMLString(xml_plugin_2, 0, err.data(), err.size());
  ASSERT_THAT(spec_1, NotNull()) << err.data();

  // do a copy before attaching
  mjSpec* spec_2 = mj_copySpec(spec_1);
  mjs_setString(spec_2->modelname, "first_copy");
  mjSpec* spec_3 = mj_copySpec(spec_1);
  mjs_setString(spec_3->modelname, "second_copy");
  ASSERT_THAT(spec_3, NotNull()) << err.data();

  // attach a body referencing the plugin to the frame and compile
  mjsBody* body_1 = mjs_findBody(parent, "body");
  EXPECT_THAT(body_1, NotNull());
  mjsFrame* attachment_frame = mjs_addFrame(body_1, 0);
  EXPECT_THAT(attachment_frame, NotNull());
  mjs_attach(attachment_frame->element, mjs_findBody(spec_1, "body")->element,
             "child-", "");
  mjModel* model_1 = mj_compile(parent, nullptr);
  EXPECT_THAT(model_1, NotNull());
  EXPECT_THAT(model_1->nbody, 3);

  // attach it a second time to test namespacing and compile
  ASSERT_THAT(spec_2, NotNull()) << err.data();
  mjs_attach(attachment_frame->element, mjs_findBody(spec_2, "body")->element,
             "copy-", "");
  mjModel* model_2 = mj_compile(parent, nullptr);
  EXPECT_THAT(model_2, NotNull());
  EXPECT_THAT(model_2->nbody, 4);

  // attach a body not referencing the plugin and compile
  mjs_attach(attachment_frame->element, mjs_findBody(spec_3, "empty")->element,
             "empty-", "");
  mjModel* model_3 = mj_compile(parent, nullptr);
  EXPECT_THAT(model_3, NotNull());
  EXPECT_THAT(model_3->nbody, 5);

  mj_deleteModel(model_1);
  mj_deleteModel(model_2);
  mj_deleteModel(model_3);
  mj_deleteSpec(parent);
  mj_deleteSpec(spec_1);
  mj_deleteSpec(spec_2);
  mj_deleteSpec(spec_3);
}

TEST_F(MujocoTest, DetachPlugin) {
  std::array<char, 1000> err;
  mjSpec* parent = mj_parseXMLString(xml_plugin_1, 0, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();
  mjSpec* child = mj_parseXMLString(xml_plugin_2, 0, err.data(), err.size());
  ASSERT_THAT(child, NotNull()) << err.data();

  // attach a body referencing the plugin to the frame
  mjsElement* frame = mjs_addFrame(mjs_findBody(parent, "world"), 0)->element;
  mjsElement* body = mjs_findBody(child, "body")->element;
  EXPECT_THAT(mjs_attach(frame, body, "child-", ""), NotNull());

  // detach the body and compile
  mjsBody* body_to_detach = mjs_findBody(parent, "child-body");
  EXPECT_THAT(body_to_detach, NotNull());
  EXPECT_THAT(mjs_delete(parent, body_to_detach->element), 0);
  mjModel* model = mj_compile(parent, nullptr);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nbody, 2);
  EXPECT_THAT(model->nplugin, 0);

  mj_deleteModel(model);
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, DetachPluginKeepsRemainingInstances) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.pid">
        <instance name="pid1"/>
        <instance name="pid2"/>
      </plugin>
    </extension>
    <worldbody>
      <body name="body1">
        <joint name="joint1" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="body2">
        <joint name="joint2" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <plugin plugin="mujoco.pid" instance="pid1" joint="joint1"/>
      <plugin plugin="mujoco.pid" instance="pid2" joint="joint2"/>
    </actuator>
  </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();

  // the first instance is deleted with the actuator, the second one is kept
  mjsBody* body = mjs_findBody(spec, "body1");
  EXPECT_EQ(mjs_delete(spec, body->element), 0) << mjs_getError(spec);
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 1);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, DeleteBodyKeepsPluginReferencedByPointer) {
  mjSpec* spec = mj_makeSpec();
  mjs_activatePlugin(spec, "mujoco.pid");
  mjsBody* world = mjs_findBody(spec, "world");

  // a body with the actuated joint and an unrelated body
  mjsBody* body = mjs_addBody(world, 0);
  mjsJoint* joint = mjs_addJoint(body, 0);
  mjs_setName(joint->element, "joint");
  joint->type = mjJNT_SLIDE;
  mjs_addGeom(body, 0)->size[0] = 0.1;
  mjsBody* other = mjs_addBody(world, 0);
  mjs_addGeom(other, 0)->size[0] = 0.1;

  // a named instance that the actuator references only through its element
  mjsPlugin* plugin = mjs_addPlugin(spec);
  mjs_setName(plugin->element, "pid");
  mjs_setString(plugin->plugin_name, "mujoco.pid");
  mjsActuator* actuator = mjs_addActuator(spec, 0);
  mjs_setString(actuator->target, "joint");
  actuator->trntype = mjTRN_JOINT;
  mjs_setString(actuator->plugin.plugin_name, "mujoco.pid");
  actuator->plugin.element = plugin->element;
  actuator->plugin.active = true;

  // deleting the unrelated body keeps the instance
  EXPECT_EQ(mjs_delete(spec, other->element), 0) << mjs_getError(spec);
  ASSERT_THAT(mjs_findElement(spec, mjOBJ_PLUGIN, "pid"), NotNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 1);
  EXPECT_EQ(model->actuator_plugin[0], 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, DeleteBodyDeletesImplicitPlugins) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.pid"/>
      <plugin plugin="mujoco.sdf.torus"/>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <asset>
      <mesh name="mesh" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <body>
        <joint name="kept"/>
        <geom size=".1"/>
      </body>
      <frame name="frame">
        <body name="body">
          <joint name="joint"/>
          <geom size=".1"/>
          <site name="site"/>
          <body>
            <geom type="sdf" mesh="mesh">
              <plugin plugin="mujoco.sdf.torus"/>
            </geom>
          </body>
        </body>
      </frame>
    </worldbody>
    <actuator>
      <plugin name="kept" plugin="mujoco.pid" joint="kept"/>
      <plugin plugin="mujoco.pid" joint="joint"/>
    </actuator>
    <sensor>
      <plugin plugin="mujoco.sensor.touch_grid" objtype="site" objname="site">
        <config key="size" value="2 2"/>
        <config key="fov" value="45 45"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
    </sensor>
  </mujoco>)";

  for (mjtObj type : {mjOBJ_BODY, mjOBJ_FRAME}) {
    for (bool compile : {false, true}) {
      std::array<char, 1000> err;
      mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
      ASSERT_THAT(spec, NotNull()) << err.data();
      if (compile) {
        mjModel* model = mj_compile(spec, nullptr);
        ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
        EXPECT_EQ(model->nplugin, 4);
        mj_deleteModel(model);
      }

      // the instances of the geom in the subtree, and of the actuator and the
      // sensor removed because they reference it, are deleted with it
      const char* name = type == mjOBJ_BODY ? "body" : "frame";
      mjsElement* element = mjs_findElement(spec, type, name);
      EXPECT_EQ(mjs_delete(spec, element), 0) << mjs_getError(spec);
      mjsElement* kept = mjs_findElement(spec, mjOBJ_ACTUATOR, "kept");
      mjsElement* plugin = mjs_firstElement(spec, mjOBJ_PLUGIN);
      EXPECT_EQ(plugin, mjs_asActuator(kept)->plugin.element);
      ASSERT_THAT(mjs_nextElement(spec, plugin), IsNull());
      mjModel* model = mj_compile(spec, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
      EXPECT_EQ(model->nplugin, 1);

      mj_deleteModel(model);
      mj_deleteSpec(spec);
    }
  }
}

TEST_F(MujocoTest, DeleteActuatorBeforeCompileDeletesImplicitPlugin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.pid"/>
    </extension>
    <worldbody>
      <body>
        <joint name="joint"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <plugin plugin="mujoco.pid" joint="joint"/>
    </actuator>
  </mujoco>)";

  // the instance is deleted with the actuator from a spec that was never
  // compiled, or from an uncompiled copy of a compiled spec
  for (bool copy : {false, true}) {
    std::array<char, 1000> err;
    mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
    ASSERT_THAT(spec, NotNull()) << err.data();
    if (copy) {
      mjModel* model = mj_compile(spec, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
      mj_deleteModel(model);
      mjSpec* original = spec;
      spec = mj_copySpec(original);
      mj_deleteSpec(original);
    }

    mjsElement* actuator = mjs_firstElement(spec, mjOBJ_ACTUATOR);
    EXPECT_EQ(mjs_delete(spec, actuator), 0) << mjs_getError(spec);
    ASSERT_THAT(mjs_firstElement(spec, mjOBJ_PLUGIN), IsNull());
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    EXPECT_EQ(model->nplugin, 0);

    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

TEST_F(MujocoTest, DeleteBodyDeletesPluginReferencedByPointer) {
  mjSpec* spec = mj_makeSpec();
  mjs_activatePlugin(spec, "mujoco.elasticity.cable");

  // a named instance that only a body references, through its element
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjs_addGeom(body, 0)->size[0] = 0.1;
  mjsPlugin* plugin = mjs_addPlugin(spec);
  mjs_setName(plugin->element, "cable");
  mjs_setString(plugin->plugin_name, "mujoco.elasticity.cable");
  mjs_setString(body->plugin.plugin_name, "mujoco.elasticity.cable");
  body->plugin.element = plugin->element;
  body->plugin.active = true;

  // the instance is deleted once, with the body
  EXPECT_EQ(mjs_delete(spec, body->element), 0) << mjs_getError(spec);
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_PLUGIN), IsNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, DeleteActuatorKeepsSharedPlugin) {
  for (bool by_name : {false, true}) {
    mjSpec* spec = mj_makeSpec();
    mjs_activatePlugin(spec, "mujoco.pid");
    mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
    mjsJoint* joint = mjs_addJoint(body, 0);
    mjs_setName(joint->element, "joint");
    mjs_addGeom(body, 0)->size[0] = 0.1;

    // two actuators that share a named instance, the first one through its
    // element, the second one through its element or by name
    mjsPlugin* plugin = mjs_addPlugin(spec);
    mjs_setName(plugin->element, "pid");
    mjs_setString(plugin->plugin_name, "mujoco.pid");
    mjsActuator* actuator[2];
    for (int i = 0; i < 2; i++) {
      actuator[i] = mjs_addActuator(spec, 0);
      mjs_setString(actuator[i]->target, "joint");
      actuator[i]->trntype = mjTRN_JOINT;
      mjs_setString(actuator[i]->plugin.plugin_name, "mujoco.pid");
      actuator[i]->plugin.active = true;
    }
    actuator[0]->plugin.element = plugin->element;
    if (by_name) {
      mjs_setString(actuator[1]->plugin.name, "pid");
    } else {
      actuator[1]->plugin.element = plugin->element;
    }
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    mj_deleteModel(model);

    // deleting the first actuator keeps the instance for the second one
    EXPECT_EQ(mjs_delete(spec, actuator[0]->element), 0) << mjs_getError(spec);
    ASSERT_THAT(mjs_findElement(spec, mjOBJ_PLUGIN, "pid"), NotNull());
    model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    EXPECT_EQ(model->nplugin, 1);
    EXPECT_EQ(model->actuator_plugin[0], 0);
    mj_deleteModel(model);

    // deleting the second one deletes the instance
    EXPECT_EQ(mjs_delete(spec, actuator[1]->element), 0) << mjs_getError(spec);
    EXPECT_THAT(mjs_firstElement(spec, mjOBJ_PLUGIN), IsNull());

    mj_deleteSpec(spec);
  }
}

TEST_F(MujocoTest, DeleteBodyKeepsPluginOfDefault) {
  mjSpec* spec = mj_makeSpec();
  mjs_activatePlugin(spec, "mujoco.pid");
  mjsBody* world = mjs_findBody(spec, "world");
  for (const char* name : {"joint1", "joint2"}) {
    mjsBody* body = mjs_addBody(world, 0);
    mjsJoint* joint = mjs_addJoint(body, 0);
    mjs_setName(joint->element, name);
    mjs_addGeom(body, 0)->size[0] = 0.1;
  }

  // a default class that points to a named instance, used by an actuator
  mjsPlugin* plugin = mjs_addPlugin(spec);
  mjs_setName(plugin->element, "pid");
  mjs_setString(plugin->plugin_name, "mujoco.pid");
  mjsDefault* def = mjs_addDefault(spec, "pid", nullptr);
  mjs_setString(def->actuator->plugin.plugin_name, "mujoco.pid");
  def->actuator->plugin.element = plugin->element;
  def->actuator->plugin.active = true;
  mjsActuator* actuator = mjs_addActuator(spec, def);
  mjs_setString(actuator->target, "joint1");
  actuator->trntype = mjTRN_JOINT;

  // deleting the body of the actuator keeps the instance of the class
  mjsBody* body1 = mjs_getParent(mjs_findElement(spec, mjOBJ_JOINT, "joint1"));
  EXPECT_EQ(mjs_delete(spec, body1->element), 0) << mjs_getError(spec);
  ASSERT_THAT(mjs_findElement(spec, mjOBJ_PLUGIN, "pid"), NotNull());
  actuator = mjs_addActuator(spec, def);
  mjs_setString(actuator->target, "joint2");
  actuator->trntype = mjTRN_JOINT;
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 1);
  EXPECT_EQ(model->actuator_plugin[0], 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, DeleteBodyKeepsUnreferencedPlugin) {
  for (const char* name : {"", "pid"}) {
    mjSpec* spec = mj_makeSpec();
    mjs_activatePlugin(spec, "mujoco.pid");
    mjsBody* world = mjs_findBody(spec, "world");
    mjsBody* body = mjs_addBody(world, 0);
    mjsJoint* joint = mjs_addJoint(body, 0);
    mjs_setName(joint->element, "joint");
    mjs_addGeom(body, 0)->size[0] = 0.1;
    mjsBody* other = mjs_addBody(world, 0);
    mjs_addGeom(other, 0)->size[0] = 0.1;

    // an instance that no element references yet is kept
    mjsPlugin* plugin = mjs_addPlugin(spec);
    mjs_setName(plugin->element, name);
    mjs_setString(plugin->plugin_name, "mujoco.pid");
    EXPECT_EQ(mjs_delete(spec, other->element), 0) << mjs_getError(spec);
    ASSERT_THAT(mjs_firstElement(spec, mjOBJ_PLUGIN), NotNull());

    // and can be referenced afterwards
    mjsActuator* actuator = mjs_addActuator(spec, 0);
    mjs_setString(actuator->target, "joint");
    actuator->trntype = mjTRN_JOINT;
    mjs_setString(actuator->plugin.plugin_name, "mujoco.pid");
    actuator->plugin.element = plugin->element;
    actuator->plugin.active = true;
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    EXPECT_EQ(model->nplugin, 1);

    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

TEST_F(MujocoTest, BodyToFrameDeletesImplicitPlugin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.elasticity.cable"/>
    </extension>
    <worldbody>
      <body name="body">
        <geom size=".1"/>
        <plugin plugin="mujoco.elasticity.cable"/>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();

  // the instance is deleted with the body, frames have no plugins
  mjsBody* body = mjs_findBody(spec, "body");
  ASSERT_THAT(mjs_bodyToFrame(&body), NotNull()) << mjs_getError(spec);
  ASSERT_THAT(mjs_firstElement(spec, mjOBJ_PLUGIN), IsNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, AttachExplicitPlugin) {
  static constexpr char xml_parent[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <body name="body"/>
      </worldbody>
    </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();

  mjSpec* child = mj_makeSpec();
  mjsBody* body = mjs_addBody(mjs_findBody(child, "world"), 0);
  mjsGeom* geom = mjs_addGeom(body, 0);
  mjsSite* site = mjs_addSite(body, 0);
  mjsSensor* sensor = mjs_addSensor(child);
  mjsPlugin* plugin = mjs_addPlugin(child);
  mjs_activatePlugin(child, "mujoco.sensor.touch_grid");
  mjs_setString(plugin->plugin_name, "mujoco.sensor.touch_grid");
  mjs_setString(sensor->plugin.plugin_name, "mujoco.sensor.touch_grid");
  mjs_setName(body->element, "body");
  mjs_setName(sensor->element, "touch2");
  mjs_setString(sensor->objname, "touch2");
  mjs_setName(site->element, "touch2");
  geom->size[0] = 0.1;
  site->size[0] = 0.001;
  sensor->type = mjSENS_PLUGIN;
  sensor->objtype = mjOBJ_SITE;
  sensor->plugin.element = plugin->element;
  sensor->plugin.active = true;
  std::map<std::string, std::string, std::less<>> config_attribs;
  config_attribs["size"] = "8 12";
  config_attribs["fov"] = "10 13";
  config_attribs["gamma"] = "0";
  config_attribs["nchannel"] = "1";
  mjs_setPluginAttributes(plugin, &config_attribs);

  mjsBody* body_parent = mjs_findBody(parent, "body");
  EXPECT_THAT(body_parent, NotNull());
  mjsFrame* attachment_frame = mjs_addFrame(body_parent, 0);
  EXPECT_THAT(attachment_frame, NotNull());

  mjs_attach(attachment_frame->element, mjs_findBody(child, "body")->element,
             "child-", "");
  mjModel* model = mj_compile(parent, nullptr);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nplugin, 1);

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, ReplicatePlugin) {
  static constexpr char xml[] = R"(
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sdf.torus">
          <instance name="torus" />
        </plugin>
      </extension>
      <asset>
        <mesh name="torus">
          <plugin instance="torus" />
        </mesh>
      </asset>
      <worldbody>
        <geom type="sdf" mesh="torus">
          <plugin instance="torus" />
        </geom>
        <replicate count="100">
          <body />
        </replicate>
      </worldbody>
    </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nplugin, 1);
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, ReplicateExplicitPlugin) {
  static constexpr char xml[] = R"(
    <mujoco>
      <extension>
        <plugin plugin="mujoco.sensor.touch_grid"/>
      </extension>
      <worldbody>
        <body name="tactile_sensor_2">
          <replicate count="2" offset="0 0.1 0">
            <geom type="sphere" size=".0008" />
          </replicate>
          <site name="touch2" size="0.001"/>
        </body>
      </worldbody>
      <sensor>
        <plugin name="touch2" plugin="mujoco.sensor.touch_grid" objtype="site" objname="touch2">
          <config key="size" value="8 12"/>
          <config key="fov" value="10 13"/>
          <config key="gamma" value="0"/>
          <config key="nchannel" value="1"/>
        </plugin>
      </sensor>
    </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nplugin, 1);
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, SetNameDetectsRepeatedNames) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsGeom* geom1 = mjs_addGeom(world, 0);
  mjsGeom* geom2 = mjs_addGeom(world, 0);
  mjsSite* site = mjs_addSite(world, 0);

  // repeated name within a type is an error
  EXPECT_EQ(mjs_setName(geom1->element, "a"), 0);
  EXPECT_EQ(mjs_setName(geom2->element, "a"), -1);
  EXPECT_STREQ(mjs_getError(spec), "Error: repeated name 'a' in geom");

  // same name across types is fine
  EXPECT_EQ(mjs_setName(site->element, "a"), 0);

  // renaming an element frees its old name
  EXPECT_EQ(mjs_setName(geom1->element, "b"), 0);
  EXPECT_EQ(mjs_setName(geom2->element, "a"), 0);
  EXPECT_EQ(mjs_setName(geom1->element, "a"), -1);

  // a failed rename leaves the previous name in place
  EXPECT_STREQ(mjs_getName(geom1->element)->c_str(), "b");

  // renaming to the current name is fine
  EXPECT_EQ(mjs_setName(geom2->element, "a"), 0);

  // empty names may repeat
  EXPECT_EQ(mjs_setName(geom1->element, ""), 0);
  mjsGeom* geom3 = mjs_addGeom(world, 0);
  EXPECT_EQ(mjs_setName(geom3->element, ""), 0);

  // deleting an element frees its name
  EXPECT_EQ(mjs_delete(spec, geom2->element), 0);
  EXPECT_EQ(mjs_setName(geom3->element, "a"), 0);

  // frames are checked too
  mjsFrame* frame1 = mjs_addFrame(world, nullptr);
  mjsFrame* frame2 = mjs_addFrame(world, nullptr);
  EXPECT_EQ(mjs_setName(frame1->element, "f"), 0);
  EXPECT_EQ(mjs_setName(frame2->element, "f"), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("repeated name 'f'"));
  EXPECT_EQ(mjs_setName(frame2->element, "g"), 0);

  // names that arrived through attach are visible to the check
  mjSpec* child = mj_makeSpec();
  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);
  EXPECT_EQ(mjs_setName(child_body->element, "body"), 0);
  mjsFrame* frame = mjs_addFrame(world, nullptr);
  ASSERT_THAT(mjs_attach(frame->element, child_body->element, "attached_", ""),
              NotNull());
  mjsBody* body = mjs_addBody(world, 0);
  EXPECT_EQ(mjs_setName(body->element, "attached_body"), -1);
  EXPECT_STREQ(mjs_getError(spec),
               "Error: repeated name 'attached_body' in body");
  EXPECT_EQ(mjs_setName(body->element, "own"), 0);

  // the model compiles once names are unique
  geom1->size[0] = geom3->size[0] = 1;
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mj_name2id(model, mjOBJ_GEOM, "a"), 1);
  EXPECT_EQ(mj_name2id(model, mjOBJ_BODY, "attached_body"), 1);

  mj_deleteModel(model);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetNameDetectsRepeatedNamesFromXML) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 1"/>
      <body name="box">
        <geom name="boxgeom" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // names loaded from XML are visible to the check
  mjsBody* world = mjs_findBody(spec, "world");
  mjsGeom* geom = mjs_addGeom(world, 0);
  geom->size[0] = 1;
  EXPECT_EQ(mjs_setName(geom->element, "floor"), -1);
  EXPECT_STREQ(mjs_getError(spec), "Error: repeated name 'floor' in geom");
  EXPECT_EQ(mjs_setName(geom->element, "boxgeom"), -1);
  EXPECT_EQ(mjs_setName(geom->element, "box"), 0);  // body name, other type

  // still checked after a compile
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjsBody* body = mjs_addBody(world, 0);
  EXPECT_EQ(mjs_setName(body->element, "box"), -1);
  EXPECT_EQ(mjs_setName(body->element, "box2"), 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SignatureTracksCompilation) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjsGeom* geom = mjs_addGeom(body, 0);
  geom->size[0] = 1;

  // an edited spec has no signature until it is compiled
  EXPECT_EQ(spec->element->signature, 0);
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(spec->element->signature, model->signature);

  // adding an element clears it, and deleting the element does not bring it
  // back
  mjsGeom* added = mjs_addGeom(body, 0);
  EXPECT_EQ(spec->element->signature, 0);
  EXPECT_EQ(mjs_delete(spec, added->element), 0);
  EXPECT_EQ(spec->element->signature, 0);

  // recompiling restores it
  mjModel* model2 = mj_compile(spec, 0);
  ASSERT_THAT(model2, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(spec->element->signature, model2->signature);
  EXPECT_EQ(model2->signature, model->signature);

  // copying a compiled spec preserves the signature, so the copy still binds
  mjSpec* clean_copy = mj_copySpec(spec);
  EXPECT_EQ(clean_copy->element->signature, model2->signature);

  // attaching clears it
  mjSpec* child = mj_makeSpec();
  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);
  mjsFrame* frame = mjs_addFrame(body, nullptr);
  ASSERT_THAT(mjs_attach(frame->element, child_body->element, "c_", ""),
              NotNull());
  EXPECT_EQ(spec->element->signature, 0);

  // copying an edited spec keeps it cleared
  mjSpec* dirty_copy = mj_copySpec(spec);
  EXPECT_EQ(dirty_copy->element->signature, 0);

  mjModel* model3 = mj_compile(spec, 0);
  ASSERT_THAT(model3, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(spec->element->signature, model3->signature);
  EXPECT_NE(model3->signature, model2->signature);

  mj_deleteModel(model3);
  mj_deleteModel(model2);
  mj_deleteModel(model);
  mj_deleteSpec(dirty_copy);
  mj_deleteSpec(clean_copy);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, RecompileFails) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjsGeom* geom = mjs_addGeom(body, 0);
  geom->type = mjGEOM_SPHERE;
  geom->size[0] = 1;

  mjModel* model = mj_compile(spec, 0);
  mjData* data = mj_makeData(model);

  mjsMaterial* mat1 = mjs_addMaterial(spec, 0);
  mjsMaterial* mat2 = mjs_addMaterial(spec, 0);
  EXPECT_EQ(mjs_setName(mat1->element, "yellow"), 0);
  EXPECT_EQ(mjs_setName(mat2->element, "yellow"), -1);
  EXPECT_STREQ(mjs_getError(spec), "Error: repeated name 'yellow' in material");

  EXPECT_EQ(mj_recompile(spec, 0, model, data), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("empty name in material"));

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, ModifyShellInertiaFails) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  1 1 1e-6"
        face="0 1 2  2 1 3" inertia="shell"/>
    </asset>
    <worldbody>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();

  // add a geom to spec
  mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "world"), nullptr);
  geom->type = mjGEOM_MESH;
  mjs_setString(geom->meshname, "example_mesh");
  geom->typeinertia = mjINERTIA_SHELL;

  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_THAT(model, IsNull());
  EXPECT_THAT(mjs_getError(spec),
              HasSubstr("inertia should be specified in the mesh asset"));
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

// the bounding volume hierarchy of a body is rebuilt by every compilation
TEST_F(MujocoTest, RecompileEditGeoms) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom name="a" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_two_geoms[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom name="a" size=".1"/>
        <geom name="b" size=".1" pos="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull());
  EXPECT_EQ(m1->nbvh, 1);

  // add a geom: the root is no longer a leaf
  mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "body"), nullptr);
  mjs_setName(geom->element, "b");
  geom->size[0] = .1;
  geom->pos[0] = 1;
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  MjModelPtr expected =
      LoadModelFromString(xml_two_geoms, er.data(), er.size());
  ASSERT_THAT(expected.get(), NotNull()) << er.data();
  ASSERT_EQ(m2->nbvh, expected->nbvh);
  for (int i = 0; i < m2->nbvh; i++) {
    EXPECT_EQ(m2->bvh_nodeid[i], expected->bvh_nodeid[i]) << "node " << i;
  }

  // delete both geoms: no hierarchy is left
  EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_GEOM, "a")), 0);
  EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_GEOM, "b")), 0);
  mjModel* m3 = mj_compile(spec, nullptr);
  ASSERT_THAT(m3, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m3->nbvh, 0);
  EXPECT_EQ(m3->body_bvhnum[1], 0);

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteModel(m3);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, RecompileEdit) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, nullptr);
  EXPECT_THAT(m1, NotNull());

  // add a geom
  mjsBody* world = mjs_findBody(spec, "world");
  mjsGeom* geom = mjs_addGeom(world, nullptr);
  geom->size[0] = 1;

  // compile again
  mjModel* m2 = mj_compile(spec, nullptr);
  EXPECT_THAT(m2, NotNull());

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteSpec(spec);
}

// editing a frame's pose after a compile takes effect in the next compile
TEST_F(MujocoTest, RecompileEditFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="outer" pos="0 0 1">
        <frame name="inner" pos="1 0 0">
          <geom name="geom" size=".1"/>
          <body name="body">
            <freejoint/>
            <geom size=".1"/>
          </body>
        </frame>
      </frame>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull());
  int geom = mj_name2id(m1, mjOBJ_GEOM, "geom");
  int body = mj_name2id(m1, mjOBJ_BODY, "body");
  const mjtNum tol = MjTol(1e-12, 1e-6);
  const mjtNum pos1[3] = {1, 0, 1};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(m1->geom_pos[3 * geom + i], pos1[i], tol);
    EXPECT_NEAR(m1->body_pos[3 * body + i], pos1[i], tol);
  }

  // move the outer frame; move the inner one and rotate it 90 degrees about z
  mjsFrame* outer = mjs_findFrame(spec, "outer");
  mjsFrame* inner = mjs_findFrame(spec, "inner");
  outer->pos[2] = 5;
  inner->pos[0] = 0;
  inner->pos[1] = 2;
  inner->quat[0] = inner->quat[3] = mju_sqrt(0.5);

  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull());
  const mjtNum pos2[3] = {0, 2, 5};
  const mjtNum quat2[4] = {mju_sqrt(0.5), 0, 0, mju_sqrt(0.5)};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(m2->geom_pos[3 * geom + i], pos2[i], tol);
    EXPECT_NEAR(m2->body_pos[3 * body + i], pos2[i], tol);
    EXPECT_NEAR(m2->qpos0[i], pos2[i], tol);
  }
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(m2->body_quat[4 * body + i], quat2[i], tol);
  }

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteSpec(spec);
}

// an element which is added after a compilation is found by its name
// a reference which is removed after compiling is not in the next model
TEST_F(MujocoTest, RecompileRemovedReferences) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="tex" type="2d" builtin="checker" width="8" height="8"/>
      <texture name="tex2" type="2d" builtin="checker" width="8" height="8"/>
      <material name="mat" %s/>
      <mesh name="mesh" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <hfield name="hf" nrow="2" ncol="2" size="1 1 1 1"/>
    </asset>
    <worldbody>
      <light name="light" %s/>
      <camera name="cam" %s/>
      <geom name="hfgeom" %s size=".2"/>
      <site name="s0"/>
      <body name="a">
        <joint/>
        <geom size=".1"/>
      </body>
      <body name="b" pos="1 0 0">
        <joint/>
        <geom name="g" %s size=".1"/>
        <site name="s" %s/>
        <site name="s1" pos="0 0 1"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="t" %s>
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
    <actuator>
      <general name="act" site="s1" %s/>
    </actuator>
  </mujoco>
  )";
  std::string with_references = absl::StrFormat(
      xml, R"(texture="tex")", R"(mode="targetbody" target="b" texture="tex2")",
      R"(mode="targetbody" target="b")", R"(type="hfield" hfield="hf")",
      R"(type="mesh" mesh="mesh" material="mat")",
      R"(type="mesh" mesh="mesh" material="mat")", R"(material="mat")",
      R"(refsite="s0")");
  std::string without = absl::StrFormat(xml, "", "", "", "", "", "", "", "");

  std::array<char, 1024> error;
  mjSpec* spec =
      mj_parseXMLString(with_references.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m1 = mj_compile(spec, 0);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m1->geom_matid[mj_name2id(m1, mjOBJ_GEOM, "g")], 0);
  EXPECT_EQ(m1->cam_targetbodyid[0], mj_name2id(m1, mjOBJ_BODY, "b"));

  // remove the references
  mjsMaterial* material =
      mjs_asMaterial(mjs_findElement(spec, mjOBJ_MATERIAL, "mat"));
  mjs_setInStringVec(material->textures, mjTEXROLE_RGB, "");
  mjsLight* light = mjs_asLight(mjs_findElement(spec, mjOBJ_LIGHT, "light"));
  light->mode = mjCAMLIGHT_FIXED;
  mjs_setString(light->targetbody, "");
  mjs_setString(light->texture, "");
  mjsCamera* camera = mjs_asCamera(mjs_findElement(spec, mjOBJ_CAMERA, "cam"));
  camera->mode = mjCAMLIGHT_FIXED;
  mjs_setString(camera->targetbody, "");
  mjsGeom* hfgeom = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "hfgeom"));
  hfgeom->type = mjGEOM_SPHERE;
  mjs_setString(hfgeom->hfieldname, "");
  mjsGeom* geom = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "g"));
  geom->type = mjGEOM_SPHERE;
  mjs_setString(geom->meshname, "");
  mjs_setString(geom->material, "");
  mjsSite* site = mjs_asSite(mjs_findElement(spec, mjOBJ_SITE, "s"));
  site->type = mjGEOM_SPHERE;
  mjs_setString(site->meshname, "");
  mjs_setString(site->material, "");
  mjsTendon* tendon = mjs_asTendon(mjs_findElement(spec, mjOBJ_TENDON, "t"));
  mjs_setString(tendon->material, "");
  mjsActuator* actuator =
      mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, "act"));
  mjs_setString(actuator->refsite, "");

  // the model is that of a spec which never had them
  mjModel* m2 = mj_compile(spec, 0);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  MjModelPtr expected =
      LoadModelFromString(without, error.data(), error.size());
  ASSERT_THAT(expected.get(), NotNull()) << error.data();
  std::string field;
  EXPECT_EQ(CompareModel(m2, expected.get(), field), 0) << field;

  mj_deleteModel(m2);
  mj_deleteModel(m1);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, FindElementAddedAfterCompile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom name="geom" size=".1"/>
      </body>
    </worldbody>
    <custom>
      <numeric name="first" data="1"/>
      <numeric name="second" data="2"/>
    </custom>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "body"), nullptr);
  mjs_setName(geom->element, "added");
  EXPECT_EQ(mjs_findElement(spec, mjOBJ_GEOM, "added"), geom->element);

  mjsNumeric* numeric = mjs_addNumeric(spec);
  mjs_setName(numeric->element, "added");
  EXPECT_EQ(mjs_findElement(spec, mjOBJ_NUMERIC, "added"), numeric->element);

  // a name which was changed since the compilation finds its element too
  mjsElement* second = mjs_findElement(spec, mjOBJ_NUMERIC, "second");
  mjs_setName(second, "renamed");
  EXPECT_EQ(mjs_findElement(spec, mjOBJ_NUMERIC, "renamed"), second);
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_NUMERIC, "second"), IsNull());

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// compiling does not change what was authored in the spec
TEST_F(MujocoTest, CompileLeavesSpecUnchanged) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree" meshdir="meshes" texturedir="textures"/>
    <default>
      <default class="arm">
        <joint damping="1"/>
        <geom type="capsule" size=".05"/>
      </default>
    </default>
    <asset>
      <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
      <mesh file="cube.obj" scale=".1 .1 .1"/>
      <texture name="grid" type="2d" builtin="checker" width="8" height="8"/>
      <material name="blue" texture="grid" rgba="0 0 1 1"/>
    </asset>
    <worldbody>
      <light pos="0 0 3"/>
      <camera name="fixed" pos="0 -2 1" xyaxes="1 0 0 0 1 2"/>
      <geom name="floor" type="plane" size="1 1 .1" material="blue"/>
      <frame name="base" pos="0 0 1" euler="0 0 30">
        <body name="upper" childclass="arm">
          <joint name="shoulder" axis="0 1 0" range="-90 90"/>
          <geom name="upper" fromto="0 0 0 0 0 -.3"/>
          <body name="lower" pos="0 0 -.3">
            <joint name="elbow" axis="0 1 0"/>
            <geom name="lower" fromto="0 0 0 0 0 -.3"/>
            <geom name="pulley" type="cylinder" size=".02 .02" pos=".1 0 -.3"/>
            <site name="side" pos=".2 0 -.3"/>
            <site name="hand" pos="0 0 -.3" zaxis="0 1 1"/>
          </body>
        </body>
      </frame>
      <body name="free" pos="1 0 1" axisangle="0 0 1 45">
        <freejoint align="true"/>
        <geom name="tetra" type="mesh" mesh="tetra" pos=".1 0 0"/>
        <geom name="cube" type="mesh" mesh="cube" pos="-.1 0 0"/>
        <site name="top" pos="0 0 .2"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="lower" geom2="tetra"/>
      <pair geom1="upper" geom2="tetra"/>
      <exclude body1="lower" body2="free"/>
      <exclude body1="upper" body2="lower"/>
    </contact>
    <equality>
      <connect body1="lower" body2="free" anchor="0 0 -.3"/>
    </equality>
    <tendon>
      <spatial name="rope" range="0 1">
        <site site="hand"/>
        <geom geom="pulley" sidesite="side"/>
        <site site="top"/>
      </spatial>
    </tendon>
    <actuator>
      <position name="shoulder" joint="shoulder" kp="10"/>
      <motor name="rope" tendon="rope"/>
    </actuator>
    <sensor>
      <jointpos joint="elbow"/>
      <framepos objtype="site" objname="hand"/>
    </sensor>
    <custom>
      <numeric name="gain" data="1 2 3"/>
    </custom>
    <keyframe>
      <key name="bent" qpos="45 90 1 0 1 1 0 0 0"/>
    </keyframe>
  </mujoco>
  )";

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
  mj_addBufferVFS(vfs.get(), "meshes/cube.obj", cube, sizeof(cube));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjSpec* authored = mj_copySpec(spec);
  ASSERT_THAT(CompareSpec(authored, spec), IsEmpty());

  // a compilation, and a second one
  for (int i = 0; i < 2; i++) {
    mjModel* model = mj_compile(spec, vfs.get());
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    EXPECT_THAT(CompareSpec(authored, spec), IsEmpty());
    mj_deleteModel(model);
  }

  // a compilation which fails at its very end, in the keyframes
  for (mjSpec* s : {spec, authored}) {
    mjs_asKey(mjs_findElement(s, mjOBJ_KEY, "bent"))->qpos->push_back(0);
  }
  EXPECT_THAT(mj_compile(spec, vfs.get()), IsNull());
  EXPECT_THAT(CompareSpec(authored, spec), IsEmpty());

  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// the model holds pairs and excludes sorted, the spec in the authored order
TEST_F(MujocoTest, CompileKeepsPairOrder) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint/>
        <geom name="a" size=".1"/>
      </body>
      <body name="b" pos="1 0 0">
        <joint/>
        <geom name="b" size=".1"/>
      </body>
      <body name="c" pos="2 0 0">
        <joint/>
        <geom name="c" size=".1"/>
      </body>
    </worldbody>
    <contact>
      <pair name="late" geom1="b" geom2="c"/>
      <pair name="early" geom1="a" geom2="b"/>
      <exclude name="late" body1="b" body2="c"/>
      <exclude name="early" body1="a" body2="b"/>
    </contact>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  for (mjtObj type : {mjOBJ_PAIR, mjOBJ_EXCLUDE}) {
    // the model is sorted
    EXPECT_EQ(mj_name2id(model, type, "early"), 0);
    EXPECT_EQ(mj_name2id(model, type, "late"), 1);

    // the spec is not
    mjsElement* first = mjs_firstElement(spec, type);
    EXPECT_STREQ(mjs_getString(mjs_getName(first)), "late");

    // a name finds its element, whose id is its row in the model
    for (const char* name : {"late", "early"}) {
      mjsElement* element = mjs_findElement(spec, type, name);
      ASSERT_THAT(element, NotNull());
      EXPECT_STREQ(mjs_getString(mjs_getName(element)), name);
      EXPECT_EQ(mjs_getId(element), mj_name2id(model, type, name));
    }
  }

  // a value edited in the model is copied back to its own pair
  model->pair_margin[mj_name2id(model, mjOBJ_PAIR, "late")] = 0.5;
  EXPECT_EQ(mj_copyBack(spec, model), 1);
  std::array<char, 2000> saved;
  mj_saveXMLString(spec, saved.data(), saved.size(), er.data(), er.size());
  EXPECT_THAT(saved.data(),
              ::testing::ContainsRegex("name=\"late\".*margin=\"0.5\""));

  // compiling again gives the same model
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mj_name2id(again, mjOBJ_PAIR, "early"), 0);
  EXPECT_EQ(mj_name2id(again, mjOBJ_EXCLUDE, "early"), 0);

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// changing the file of an asset after a compile takes effect in the next one,
// and the name which the parser derived from the first file stays
TEST_F(MujocoTest, RecompileEditMeshFile) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";
  static constexpr char tetra[] = R"(
  v 0 0 0
  v 1 0 0
  v 0 1 0
  v 0 0 1)";
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh file="cube.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="cube"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_tetra[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" file="tetra.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="cube"/>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));
  mj_addBufferVFS(vfs.get(), "tetra.obj", tetra, sizeof(tetra));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m1->nmeshvert, 8);

  mjsMesh* mesh = mjs_asMesh(mjs_findElement(spec, mjOBJ_MESH, "cube"));
  ASSERT_THAT(mesh, NotNull());
  mjs_setString(mesh->file, "tetra.obj");
  mjModel* m2 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);

  mjSpec* spec_tetra =
      mj_parseXMLString(xml_tetra, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec_tetra, NotNull()) << er.data();
  mjModel* expected = mj_compile(spec_tetra, vfs.get());
  ASSERT_THAT(expected, NotNull()) << mjs_getError(spec_tetra);
  std::string field;
  EXPECT_EQ(CompareModel(m2, expected, field), 0) << field;

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteModel(expected);
  mj_deleteSpec(spec_tetra);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// ------------------- test cache with modified assets -------------------------

TEST_F(MujocoTest, RecompileCompareObjCache) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh file="cube.obj"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="cube"/>
    </worldbody>
  </mujoco>)";

  static constexpr char cube1[] = R"(
  v -0.500000 -0.500000  0.500000
  v  0.500000 -0.500000  0.500000
  v -0.500000  0.500000  0.500000
  v  0.500000  0.500000  0.500000
  v -0.500000  0.500000 -0.500000
  v  0.500000  0.500000 -0.500000
  v -0.500000 -0.500000 -0.500000
  v  0.500000 -0.500000 -0.500000)";

  static constexpr char cube2[] = R"(
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
  mj_addBufferVFS(vfs.get(), "cube.obj", cube1, sizeof(cube1));

  std::array<char, 1024> error;

  // load model once
  MjModelPtr m =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  EXPECT_EQ(m->mesh_vert[0], -0.5);

  // update cube.obj, load again
  mj_deleteFileVFS(vfs.get(), "cube.obj");
  mj_addBufferVFS(vfs.get(), "cube.obj", cube2, sizeof(cube2));
  m = LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  EXPECT_EQ(m->mesh_vert[0], -1);

  mj_deleteVFS(vfs.get());
}

// tiny RGB 2 x 3 PNG file
static constexpr uint8_t tex1[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0x12, 0x16, 0xf1, 0x4d,
    0x00, 0x00, 0x00, 0x1c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
    0x78, 0xc1, 0xc0, 0xc0, 0xc0, 0xf0, 0xbf, 0xb8, 0xb8, 0x98, 0x81,
    0xe1, 0x3f, 0xc3, 0xff, 0xff, 0xff, 0xc5, 0xc4, 0xc4, 0x00, 0x46,
    0xd7, 0x07, 0x7f, 0xd2, 0x52, 0xa1, 0x41, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

// previous PNG file, but rotated by 180 degrees
static constexpr uint8_t tex2[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0x12, 0x16, 0xf1, 0x4d,
    0x00, 0x00, 0x00, 0x1c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
    0x10, 0x13, 0x13, 0xfb, 0xff, 0xff, 0x3f, 0xc3, 0x7f, 0x06, 0x96,
    0xd8, 0xd8, 0x58, 0x46, 0x46, 0x86, 0x17, 0x0c, 0x0c, 0x00, 0x49,
    0x22, 0x06, 0x44, 0xe4, 0x91, 0xb8, 0x83, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

TEST_F(MujocoTest, RecompileComparePngCache) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture content_type="image/png" file="tex.png" type="2d"/>
      <material name="material" texture="tex"/>
    </asset>

    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "tex.png", tex1, sizeof(tex1));

  std::array<char, 1024> error;

  // load model once
  MjModelPtr m =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  EXPECT_EQ(m->ntexdata, 18);  // w x h x rgb = 3 x 2 x 3
  mjtByte byte = m->tex_data[0];

  // update tex.png, load again
  mj_deleteFileVFS(vfs.get(), "tex.png");
  mj_addBufferVFS(vfs.get(), "tex.png", tex2, sizeof(tex2));
  m = LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  EXPECT_NE(m->tex_data[0], byte);
  EXPECT_EQ(m->tex_data[15], byte);  // first pixel is now last pixel

  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, DisableCache) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture content_type="image/png" file="tex.png" type="2d"/>
      <material name="material" texture="tex"/>
    </asset>

    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
)";

  mjCache* cache = mj_getCache();
  std::size_t capacity = mj_getCacheCapacity(cache);
  mj_setCacheCapacity(cache, 0);

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "tex.png", tex1, sizeof(tex1));

  std::array<char, 1024> error;

  // load model once
  MjModelPtr m =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());

  EXPECT_EQ(mj_getCacheSize(cache), 0);
  EXPECT_EQ(m->ntexdata, 18);  // w x h x rgb = 3 x 2 x 3

  mj_setCacheCapacity(cache, capacity);
  mj_deleteVFS(vfs.get());
}

// -------------------------------- test textures ------------------------------

TEST_F(MujocoTest, TextureFromBuffer) {
  mjSpec* spec = mj_makeSpec();

  mjsTexture* t1 = mjs_addTexture(spec, nullptr);
  mjs_setName(t1->element, "tex1");
  t1->type = mjTEXTURE_2D;
  t1->width = 3;
  t1->height = 2;
  t1->nchannel = 3;
  mjs_setBuffer(t1->data, (std::byte*)tex1, 18);

  mjsTexture* t2 = mjs_addTexture(spec, nullptr);
  mjs_setName(t2->element, "tex2");
  t2->type = mjTEXTURE_2D;
  t2->width = 3;
  t2->height = 2;
  t2->nchannel = 3;
  mjs_setBuffer(t2->data, (std::byte*)tex2, 18);

  mjsMaterial* mat = mjs_addMaterial(spec, nullptr);
  mjs_setName(mat->element, "mat");
  mjs_setInStringVec(mat->textures, mjTEXROLE_RGB, "tex1");
  mjs_setInStringVec(mat->textures, mjTEXROLE_ORM, "tex2");

  mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "world"), nullptr);
  mjs_setString(geom->material, "mat");
  geom->size[0] = 1;

  mjModel* m = mj_compile(spec, nullptr);
  EXPECT_THAT(m, NotNull());
  EXPECT_EQ(m->ntex, 2);

  std::array<char, 1024> err;
  std::array<char, 1024> str;
  mj_saveXMLString(spec, str.data(), str.size(), err.data(), err.size());
  EXPECT_STREQ(err.data(), "XML Error: no support for buffer textures.");

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// the buffer of a texture belongs to the user: compiling reads it, and does
// not fill it for textures made from a file or a builtin
TEST_F(MujocoTest, TextureBufferIsAuthored) {
  mjSpec* spec = mj_makeSpec();
  mjsTexture* texture = mjs_addTexture(spec, nullptr);
  mjs_setName(texture->element, "texture");
  texture->type = mjTEXTURE_2D;
  texture->width = 2;
  texture->height = 1;

  // a builtin texture: the pixels are in the model only
  texture->builtin = mjBUILTIN_FLAT;
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  EXPECT_GT(m1->ntexdata, 0);
  EXPECT_THAT(*texture->data, IsEmpty());

  // a buffer given after a compilation is used by the next one; a flip is
  // applied to the model and not to the buffer, however often it is compiled
  const std::vector<std::byte> pixels = {std::byte{1}, std::byte{2}};
  texture->builtin = mjBUILTIN_NONE;
  texture->nchannel = 1;
  texture->hflip = 1;
  mjs_setBuffer(texture->data, pixels.data(), pixels.size());
  for (int i = 0; i < 2; i++) {
    mjModel* m2 = mj_compile(spec, nullptr);
    ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
    ASSERT_EQ(m2->ntexdata, 2);
    EXPECT_EQ(m2->tex_data[0], 2);
    EXPECT_EQ(m2->tex_data[1], 1);
    EXPECT_EQ(*texture->data, pixels);
    mj_deleteModel(m2);
  }

  mj_deleteModel(m1);
  mj_deleteSpec(spec);
}

// a texture added with a class takes the class's texture defaults, one
// added without a class takes the main defaults
TEST_F(MujocoTest, TextureDefaults) {
  mjSpec* spec = mj_makeSpec();
  mjsDefault* main = mjs_getSpecDefault(spec);
  ASSERT_THAT(main->texture, NotNull());
  EXPECT_EQ(main->texture->nchannel, 3);
  EXPECT_EQ(main->texture->type, mjTEXTURE_CUBE);

  mjsDefault* def = mjs_addDefault(spec, "flat", main);
  def->texture->type = mjTEXTURE_2D;
  def->texture->builtin = mjBUILTIN_FLAT;
  def->texture->width = 2;
  def->texture->height = 3;
  def->texture->rgb1[0] = 0.25;
  EXPECT_EQ(mjs_findDefault(spec, "flat")->texture->builtin, mjBUILTIN_FLAT);

  // a child class inherits the texture defaults of its parent
  mjsDefault* child = mjs_addDefault(spec, "child", def);
  EXPECT_EQ(child->texture->builtin, mjBUILTIN_FLAT);
  EXPECT_EQ(child->texture->height, 3);

  mjsTexture* with = mjs_addTexture(spec, def);
  mjs_setName(with->element, "with");
  EXPECT_EQ(with->type, mjTEXTURE_2D);
  EXPECT_EQ(with->builtin, mjBUILTIN_FLAT);
  EXPECT_EQ(with->width, 2);
  EXPECT_EQ(with->height, 3);
  EXPECT_EQ(with->rgb1[0], 0.25);
  EXPECT_EQ(mjs_getDefault(with->element), def);

  mjsTexture* without = mjs_addTexture(spec, nullptr);
  mjs_setName(without->element, "without");
  EXPECT_EQ(without->type, mjTEXTURE_CUBE);
  EXPECT_EQ(without->builtin, mjBUILTIN_NONE);
  EXPECT_EQ(without->width, 0);
  EXPECT_EQ(mjs_getDefault(without->element), main);
  without->type = mjTEXTURE_2D;
  without->builtin = mjBUILTIN_CHECKER;
  without->width = 4;
  without->height = 4;

  // the model has what the class gave the texture
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->tex_type[0], mjTEXTURE_2D);
  EXPECT_EQ(model->tex_width[0], 2);
  EXPECT_EQ(model->tex_height[0], 3);
  EXPECT_EQ(model->tex_data[0], (mjtByte)(255 * 0.25));
  EXPECT_EQ(model->tex_width[1], 4);

  // the class and its texture defaults are saved
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr("<default class=\"flat\">"));
  EXPECT_THAT(saved, HasSubstr("name=\"with\" class=\"flat\"/>"));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// texture classes of an attached spec are kept, under the attach prefix
TEST_F(MujocoTest, AttachKeepsTextureDefaults) {
  static constexpr char xml_child[] = R"(
  <mujoco>
    <default>
      <default class="tex">
        <texture type="2d" builtin="checker" width="4" height="4"/>
      </default>
    </default>
    <asset>
      <texture name="checker" class="tex"/>
      <material name="mat" texture="checker"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom size="1" material="mat"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <frame name="frame"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> err;
  mjSpec* child = mj_parseXMLString(xml_child, 0, err.data(), err.size());
  ASSERT_THAT(child, NotNull()) << err.data();
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();

  mjsElement* attached =
      mjs_attach(mjs_findFrame(parent, "frame")->element,
                 mjs_findBody(child, "body")->element, "child-", "");
  ASSERT_THAT(attached, NotNull()) << mjs_getError(parent);

  mjsDefault* def = mjs_findDefault(parent, "child-tex");
  ASSERT_THAT(def, NotNull());
  EXPECT_EQ(def->texture->builtin, mjBUILTIN_CHECKER);
  mjsTexture* texture =
      mjs_asTexture(mjs_findElement(parent, mjOBJ_TEXTURE, "child-checker"));
  ASSERT_THAT(texture, NotNull());
  EXPECT_EQ(mjs_getDefault(texture->element), def);

  mjModel* model = mj_compile(parent, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
  EXPECT_EQ(model->ntex, 1);
  EXPECT_EQ(model->tex_width[0], 4);
  EXPECT_THAT(SaveAndReadXml(parent),
              HasSubstr("name=\"child-checker\" class=\"child-tex\"/>"));

  mj_deleteModel(model);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

TEST_F(MujocoTest, TestTextureFlip) {
  mjSpec* spec = mj_makeSpec();
  EXPECT_THAT(spec, NotNull());

  std::vector<std::byte> texture_data = {std::byte{1}, std::byte{2},
                                         std::byte{3}, std::byte{4}};
  mjsTexture* texture = mjs_addTexture(spec, nullptr);
  mjs_setName(texture->element, "hflipped");
  texture->type = mjTEXTURE_2D;
  texture->width = 2;
  texture->height = 2;
  texture->nchannel = 1;
  texture->hflip = true;
  mjs_setBuffer(texture->data, texture_data.data(), texture_data.size());

  texture_data = {std::byte{11}, std::byte{12}, std::byte{13}, std::byte{21},
                  std::byte{22}, std::byte{23}, std::byte{31}, std::byte{32},
                  std::byte{33}, std::byte{41}, std::byte{42}, std::byte{43}};

  texture = mjs_addTexture(spec, nullptr);
  mjs_setName(texture->element, "vflipped");
  texture->type = mjTEXTURE_2D;
  texture->width = 2;
  texture->height = 2;
  texture->nchannel = 3;
  texture->vflip = true;
  mjs_setBuffer(texture->data, texture_data.data(), texture_data.size());

  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull()) << mjs_getError(spec);

  EXPECT_THAT(model->tex_data[0], 2);
  EXPECT_THAT(model->tex_data[1], 1);
  EXPECT_THAT(model->tex_data[2], 4);
  EXPECT_THAT(model->tex_data[3], 3);

  EXPECT_THAT(model->tex_data[4], 31);
  EXPECT_THAT(model->tex_data[5], 32);
  EXPECT_THAT(model->tex_data[6], 33);
  EXPECT_THAT(model->tex_data[7], 41);
  EXPECT_THAT(model->tex_data[8], 42);
  EXPECT_THAT(model->tex_data[9], 43);
  EXPECT_THAT(model->tex_data[10], 11);
  EXPECT_THAT(model->tex_data[11], 12);
  EXPECT_THAT(model->tex_data[12], 13);
  EXPECT_THAT(model->tex_data[13], 21);
  EXPECT_THAT(model->tex_data[14], 22);
  EXPECT_THAT(model->tex_data[15], 23);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// -------------------------------- test attach --------------------------------

static constexpr char xml_child[] = R"(
  <mujoco model="child">
    <default>
      <default class="cylinder">
        <geom type="cylinder" size=".1 1 0"/>
      </default>
    </default>

    <custom>
      <numeric data="10" name="constant"/>
    </custom>

    <asset>
      <texture name="texture" type="2d" builtin="checker" width="32" height="32"/>
      <material name="material" texture="texture" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
      <frame name="pframe">
        <frame name="cframe">
          <body name="body">
            <joint type="hinge" name="hinge"/>
            <geom class="cylinder" material="material"/>
            <light mode="targetbody" target="targetbody"/>
            <site name="site" material="material"/>
            <body name="targetbody"/>
            <body/>
          </body>
        </frame>
      </frame>
      <body name="ignore">
        <geom size=".1"/>
        <joint type="slide"/>
      </body>
      <frame name="frame" pos=".1 0 0" euler="0 90 0"/>
    </worldbody>

    <sensor>
      <framepos name="sensor" objtype="body" objname="body"/>
      <framepos name="ignore" objtype="body" objname="ignore"/>
    </sensor>

    <tendon>
      <fixed name="fixed">
        <joint joint="hinge" coef="2"/>
      </fixed>
    </tendon>

    <actuator>
      <position name="hinge" joint="hinge" timeconst=".01"/>
      <position name="fixed" tendon="fixed" timeconst=".01"/>
      <position name="site" site="site" timeconst=".01"/>
    </actuator>

    <contact>
      <exclude body1="body" body2="targetbody"/>
    </contact>

    <keyframe>
      <key name="two" time="2" qpos="2 22" act="1 2 3" ctrl="1 2 3"/>
      <key name="three" time="3" qpos="3 33" act="4 5 6" ctrl="4 5 6"/>
    </keyframe>
  </mujoco>)";

TEST_F(MujocoTest, AttachSame) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_result[] = R"(
  <mujoco model="child">
    <default>
      <default class="cylinder">
        <geom type="cylinder" size=".1 1 0"/>
      </default>
    </default>

    <custom>
      <numeric data="10" name="constant"/>
    </custom>

    <asset>
      <texture name="texture" type="2d" builtin="checker" width="32" height="32"/>
      <material name="material" texture="texture" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
      <body name="body">
        <joint type="hinge" name="hinge"/>
        <geom class="cylinder" material="material"/>
        <light mode="targetbody" target="targetbody"/>
        <site name="site" material="material"/>
        <body name="targetbody"/>
        <body/>
      </body>
      <body name="ignore">
        <geom size=".1"/>
        <joint type="slide"/>
      </body>
      <frame name="frame" pos=".1 0 0" euler="0 90 0">
        <body name="attached-body-1">
          <joint type="hinge" name="attached-hinge-1"/>
          <geom class="cylinder" material="material"/>
          <light mode="targetbody" target="attached-targetbody-1"/>
          <site name="attached-site-1" material="material"/>
          <body name="attached-targetbody-1"/>
          <body/>
        </body>
      </frame>
    </worldbody>

    <sensor>
      <framepos name="sensor" objtype="body" objname="body"/>
      <framepos name="ignore" objtype="body" objname="ignore"/>
      <framepos name="attached-sensor-1" objtype="body" objname="attached-body-1"/>
    </sensor>

    <tendon>
      <fixed name="fixed">
        <joint joint="hinge" coef="2"/>
      </fixed>
      <fixed name="attached-fixed-1">
        <joint joint="attached-hinge-1" coef="2"/>
      </fixed>
    </tendon>

    <actuator>
      <position name="hinge" joint="hinge" timeconst=".01"/>
      <position name="fixed" tendon="fixed" timeconst=".01"/>
      <position name="site" site="site" timeconst=".01"/>
      <position name="attached-hinge-1" joint="attached-hinge-1" timeconst=".01"/>
      <position name="attached-fixed-1" tendon="attached-fixed-1" timeconst=".01"/>
      <position name="attached-site-1" site="attached-site-1" timeconst=".01"/>
    </actuator>

    <contact>
      <exclude body1="body" body2="targetbody"/>
      <exclude body1="attached-body-1" body2="attached-targetbody-1"/>
    </contact>

    <keyframe>
      <key name="two" time="2" qpos="2 22 0" act="1 2 3 0 0 0" ctrl="1 2 3 0 0 0"/>
      <key name="three" time="3" qpos="3 33 0" act="4 5 6 0 0 0" ctrl="4 5 6 0 0 0"/>
      <key name="attached-two-1" time="2" qpos="0 22 2" act="0 0 0 1 2 3" ctrl="0 0 0 1 2 3"/>
      <key name="attached-three-1" time="3" qpos="0 33 3" act="0 0 0 4 5 6" ctrl="0 0 0 4 5 6"/>
    </keyframe>
  </mujoco>)";

  // create parent
  mjSpec* parent = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjs_setDeepCopy(parent, true);  // needed for self-attach

  // get frame
  mjsFrame* frame = mjs_findFrame(parent, "frame");
  EXPECT_THAT(frame, NotNull());

  // get subtree
  mjsBody* body = mjs_findBody(parent, "body");
  EXPECT_THAT(body, NotNull());

  // attach child to parent frame
  mjsBody* attached =
      mjs_asBody(mjs_attach(frame->element, body->element, "attached-", "-1"));
  EXPECT_THAT(attached, mjs_findBody(parent, "attached-body-1"));

  // check that the spec was not copied
  EXPECT_THAT(mjs_findSpec(parent, "child"), IsNull());

  // compile new model
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull());

  // check full name stored in mjModel
  EXPECT_STREQ(mj_id2name(m_attached, mjOBJ_BODY, 5), "attached-body-1");

  // check body 3 is attached to the world
  EXPECT_THAT(m_attached->body_parentid[4], 0);

  // compare with expected XML
  MjModelPtr m_expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_attached, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';
  ;

  // destroy everything
  mj_deleteSpec(parent);
  mj_deleteModel(m_attached);
}

TEST_F(MujocoTest, AttachSpatialTendonWithoutSidesite) {
  mock_warning_handler.ExpectWarnings();
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent_body">
        <geom size="0.1" type="sphere"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="child_body">
        <geom name="wrap_geom" size="0.05" type="sphere"/>
        <site name="site_A" pos="0 0 0.1"/>
        <site name="site_B" pos="0 0 -0.1"/>
        <site name="side_site" pos="0.05 0 0"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="tendon_with_sidesite">
        <site site="site_A"/>
        <geom geom="wrap_geom" sidesite="side_site"/>
        <site site="site_B"/>
      </spatial>
      <spatial name="tendon_without_sidesite">
        <site site="site_A"/>
        <geom geom="wrap_geom"/>
        <site site="site_B"/>
      </spatial>
    </tendon>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  mjsBody* parent_body = mjs_findBody(parent, "parent_body");
  ASSERT_THAT(parent_body, NotNull());
  mjsSite* attach_site = mjs_addSite(parent_body, 0);
  mjs_setName(attach_site->element, "attach_site");

  mjs_attach(attach_site->element, mjs_findBody(child, "child_body")->element,
             "", "_child");

  EXPECT_THAT(
      mjs_findElement(parent, mjOBJ_TENDON, "tendon_with_sidesite_child"),
      NotNull());
  EXPECT_THAT(
      mjs_findElement(parent, mjOBJ_TENDON, "tendon_without_sidesite_child"),
      NotNull());

  mjsTendon* tendon = mjs_asTendon(
      mjs_findElement(parent, mjOBJ_TENDON, "tendon_with_sidesite_child"));
  ASSERT_THAT(tendon, NotNull());
  ASSERT_EQ(mjs_getWrapNum(tendon), 3);

  EXPECT_EQ(mjs_getWrapTarget(mjs_getWrap(tendon, 0)),
            mjs_findElement(parent, mjOBJ_SITE, "site_A_child"));
  EXPECT_EQ(mjs_getWrapTarget(mjs_getWrap(tendon, 1)),
            mjs_findElement(parent, mjOBJ_GEOM, "wrap_geom_child"));
  EXPECT_EQ(mjs_getWrapTarget(mjs_getWrap(tendon, 2)),
            mjs_findElement(parent, mjOBJ_SITE, "site_B_child"));
  EXPECT_EQ(mjs_getWrapSideSite(mjs_getWrap(tendon, 1)),
            mjs_asSite(mjs_findElement(parent, mjOBJ_SITE, "side_site_child")));

  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
  EXPECT_EQ(model->ntendon, 2);

  mj_deleteModel(model);
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachSpatialTendonGitHubIssue3119) {
  mock_warning_handler.ExpectWarnings();
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent_body">
        <geom size="0.1" type="sphere"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="child_body">
        <geom name="wrap_geom" size="0.05" type="sphere"/>
        <site name="site_A" pos="0 0 0.1"/>
        <site name="site_B" pos="0 0 -0.1"/>
        <site name="side_site" pos="0.05 0 0"/>
      </body>
    </worldbody>
    <tendon>
      <spatial name="tendon_with_sidesite">
        <site site="site_A"/>
        <geom geom="wrap_geom" sidesite="side_site"/>
        <site site="site_B"/>
      </spatial>
      <spatial name="tendon_without_sidesite">
        <site site="site_A"/>
        <geom geom="wrap_geom"/>
        <site site="site_B"/>
      </spatial>
    </tendon>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* parent_spec = mj_parseXMLString(parent_xml, 0, er.data(), er.size());
  ASSERT_THAT(parent_spec, NotNull()) << er.data();
  mjSpec* child_spec = mj_parseXMLString(child_xml, 0, er.data(), er.size());
  ASSERT_THAT(child_spec, NotNull()) << er.data();

  mjsBody* parent_body = mjs_findBody(parent_spec, "parent_body");
  ASSERT_THAT(parent_body, NotNull());
  mjsSite* attach_site = mjs_addSite(parent_body, 0);
  mjs_setName(attach_site->element, "attach_site");

  mjs_attach(attach_site->element,
             mjs_findBody(child_spec, "child_body")->element, "", "_child");

  EXPECT_THAT(
      mjs_findElement(parent_spec, mjOBJ_TENDON, "tendon_with_sidesite_child"),
      NotNull());
  EXPECT_THAT(mjs_findElement(parent_spec, mjOBJ_TENDON,
                              "tendon_without_sidesite_child"),
              NotNull());

  mjModel* model = mj_compile(parent_spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent_spec);
  EXPECT_EQ(model->ntendon, 2);

  mj_deleteModel(model);
  mj_deleteSpec(parent_spec);
  mj_deleteSpec(child_spec);
}

TEST_F(MujocoTest, AttachDifferent) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco model="parent">
    <default>
      <default class="geom_size">
        <geom size="0.1"/>
      </default>
    </default>

    <worldbody>
      <body name="sphere">
        <freejoint/>
        <geom class="geom_size"/>
        <frame name="frame" pos=".1 0 0" euler="0 90 0"/>
      </body>
    </worldbody>

    <keyframe>
      <key name="one" time="1" qpos="1 1 1 1 0 0 0"/>
    </keyframe>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco model="parent">
    <default>
      <default class="geom_size">
        <geom size="0.1"/>
      </default>
      <default class="attached-cylinder-1">
        <geom type="cylinder" size=".1 1 0"/>
      </default>
    </default>

    <custom>
      <numeric data="10" name="attached-constant-1"/>
    </custom>

    <asset>
      <texture name="attached-texture-1" type="2d" builtin="checker" width="32" height="32"/>
      <material name="attached-material-1" texture="attached-texture-1" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
      <body name="sphere">
        <freejoint/>
        <geom class="geom_size"/>
        <frame name="frame" pos=".1 0 0" euler="0 90 0">
          <body name="attached-body-1">
            <joint type="hinge" name="attached-hinge-1"/>
            <geom class="attached-cylinder-1" material="attached-material-1"/>
            <light mode="targetbody" target="attached-targetbody-1"/>
            <site name="attached-site-1" material="attached-material-1"/>
            <body name="attached-targetbody-1"/>
            <body/>
          </body>
        </frame>
      </body>
    </worldbody>

    <sensor>
      <framepos name="attached-sensor-1" objtype="body" objname="attached-body-1"/>
    </sensor>

    <tendon>
      <fixed name="attached-fixed-1">
        <joint joint="attached-hinge-1" coef="2"/>
      </fixed>
    </tendon>

    <actuator>
      <position name="attached-hinge-1" joint="attached-hinge-1" timeconst=".01"/>
      <position name="attached-fixed-1" tendon="attached-fixed-1" timeconst=".01"/>
      <position name="attached-site-1" site="attached-site-1" timeconst=".01"/>
    </actuator>

    <contact>
      <exclude body1="attached-body-1" body2="attached-targetbody-1"/>
    </contact>

    <keyframe>
      <key name="one" time="1" qpos="1 1 1 1 0 0 0 0"/>
      <key name="attached-two-1" time="2" qpos="0 0 0 1 0 0 0 2" act="1 2 3" ctrl="1 2 3"/>
      <key name="attached-three-1" time="3" qpos="0 0 0 1 0 0 0 3" act="4 5 6" ctrl="4 5 6"/>
    </keyframe>
  </mujoco>)";

  // model with one free sphere and a frame
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();

  // get frame
  mjsFrame* frame = mjs_findFrame(parent, "frame");
  EXPECT_THAT(frame, NotNull());

  // model with one cylinder and a hinge
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  // get subtree
  mjsBody* body = mjs_findBody(child, "body");
  EXPECT_THAT(body, NotNull());

  // attach child to parent frame
  mjsBody* attached =
      mjs_asBody(mjs_attach(frame->element, body->element, "attached-", "-1"));
  EXPECT_THAT(attached, mjs_findBody(parent, "attached-body-1"));

  // check that the spec was copied
  EXPECT_THAT(mjs_findSpec(parent, "child"), NotNull());

  // compile new model
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull());

  // check frame is present
  EXPECT_THAT(mjs_findFrame(parent, "frame"), NotNull());

  // check full name stored in mjModel
  EXPECT_STREQ(mj_id2name(m_attached, mjOBJ_BODY, 2), "attached-body-1");

  // check body 2 is attached to body 1
  EXPECT_THAT(m_attached->body_parentid[2], 1);

  // check that the correct defaults are present
  EXPECT_THAT(mjs_findDefault(parent, "main"), NotNull());
  EXPECT_THAT(mjs_findDefault(parent, "geom_size"), NotNull());
  EXPECT_THAT(mjs_findDefault(parent, "attached-cylinder-1"), NotNull());

  // compare with expected XML
  MjModelPtr m_expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_attached, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';
  ;

  // destroy everything
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(m_attached);
}

TEST_F(MujocoTest, AttachFrame) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco model="parent">
    <worldbody>
      <body name="sphere">
        <freejoint/>
        <geom size=".1"/>
        <frame name="frame" pos=".1 0 0" euler="0 90 0"/>
      </body>
    </worldbody>

    <keyframe>
      <key name="one" time="1" qpos="1 1 1 1 0 0 0"/>
    </keyframe>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco model="parent">
    <default>
      <default class="attached-cylinder-1">
        <geom type="cylinder" size=".1 1 0"/>
      </default>
    </default>

    <custom>
      <numeric data="10" name="attached-constant-1"/>
    </custom>

    <asset>
      <texture name="attached-texture-1" type="2d" builtin="checker" width="32" height="32"/>
      <material name="attached-material-1" texture="attached-texture-1" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
      <body name="sphere">
        <freejoint/>
        <geom size=".1"/>
        <frame name="frame" pos=".1 0 0" euler="0 90 0"/>
        <frame name="pframe">
          <frame name="cframe">
            <body name="attached-body-1">
              <joint type="hinge" name="attached-hinge-1"/>
              <geom class="attached-cylinder-1" material="attached-material-1"/>
              <light mode="targetbody" target="attached-targetbody-1"/>
              <site name="attached-site-1" material="attached-material-1"/>
              <body name="attached-targetbody-1"/>
              <body/>
            </body>
          </frame>
        </frame>
      </body>
    </worldbody>

    <sensor>
      <framepos name="attached-sensor-1" objtype="body" objname="attached-body-1"/>
    </sensor>

    <tendon>
      <fixed name="attached-fixed-1">
        <joint joint="attached-hinge-1" coef="2"/>
      </fixed>
    </tendon>

    <actuator>
      <position name="attached-hinge-1" joint="attached-hinge-1" timeconst=".01"/>
      <position name="attached-fixed-1" tendon="attached-fixed-1" timeconst=".01"/>
      <position name="attached-site-1" site="attached-site-1" timeconst=".01"/>
    </actuator>

    <contact>
      <exclude body1="attached-body-1" body2="attached-targetbody-1"/>
    </contact>

    <keyframe>
      <key name="one" time="1" qpos="1 1 1 1 0 0 0 0"/>
      <key name="attached-two-1" time="2" qpos="0 0 0 1 0 0 0 2" act="1 2 3" ctrl="1 2 3"/>
      <key name="attached-three-1" time="3" qpos="0 0 0 1 0 0 0 3" act="4 5 6" ctrl="4 5 6"/>
    </keyframe>
  </mujoco>)";

  // model with one free sphere and a frame
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();

  // get body
  mjsBody* body = mjs_findBody(parent, "sphere");
  EXPECT_THAT(body, NotNull());

  // model with one cylinder and a hinge
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  // get subtree
  mjsFrame* frame = mjs_findFrame(child, "pframe");
  EXPECT_THAT(frame, NotNull());

  // attach child frame to parent body
  mjsFrame* attached =
      mjs_asFrame(mjs_attach(body->element, frame->element, "attached-", "-1"));
  EXPECT_THAT(attached, mjs_findFrame(parent, "attached-pframe-1"));

  // check that the spec was copied
  EXPECT_THAT(mjs_findSpec(parent, "child"), NotNull());

  // check that the parent body was not namespaced
  EXPECT_THAT(mjs_findBody(child, "world"), NotNull());

  // compile new model
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull());

  // check full name stored in mjModel
  EXPECT_STREQ(mj_id2name(m_attached, mjOBJ_BODY, 2), "attached-body-1");

  // check body 2 is attached to body 1
  EXPECT_THAT(m_attached->body_parentid[2], 1);

  // compare with expected XML
  MjModelPtr m_expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_attached, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';
  ;

  // destroy everything
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(m_attached);
}

TEST_F(MujocoTest, AttachCompiled) {
  std::array<char, 1000> er;

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" pos="0 0 0" size="0 0 0.05" type="plane"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="base">
        <freejoint/>
        <geom name="geom1" size="0.1" type="sphere"/>
      </body>
    </worldbody>
  </mujoco>)";

  // load parent
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();

  // load child
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  // compile child
  mjModel* m_child = mj_compile(child, 0);
  EXPECT_THAT(m_child, NotNull()) << mjs_getError(child);

  // add frame to the parent to attach
  mjsBody* world = mjs_findBody(parent, "world");
  EXPECT_THAT(world, NotNull()) << mjs_getError(parent);
  mjsFrame* frame = mjs_addFrame(world, 0);
  EXPECT_THAT(frame, NotNull()) << mjs_getError(parent);

  // attach child body to the frame
  mjsBody* to_attach = mjs_findBody(child, "base");
  EXPECT_THAT(to_attach, NotNull()) << mjs_getError(child);
  mjs_attach(frame->element, to_attach->element, "child_", "");

  // check that attached model can be compiled
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull())
      << "Failed to compile attached model" << mjs_getError(parent);

  // destroy everything
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(m_attached);
  mj_deleteModel(m_child);
}

void TestDetachBody(bool compile) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_result[] = R"(
  <mujoco model="child">
    <asset>
      <texture name="texture" type="2d" builtin="checker" width="32" height="32"/>
      <material name="material" texture="texture" texrepeat="1 1" texuniform="true"/>
    </asset>

    <custom>
      <numeric data="10" name="constant"/>
    </custom>

    <worldbody>
      <frame name="pframe">
        <frame name="cframe">
        </frame>
      </frame>
      <body name="ignore">
        <geom size=".1"/>
        <joint type="slide"/>
      </body>
      <frame name="frame" pos=".1 0 0" euler="0 90 0"/>
    </worldbody>

    <sensor>
      <framepos name="ignore" objtype="body" objname="ignore"/>
    </sensor>

    <keyframe>
      <key name="two" time="2" qpos="22"/>
      <key name="three" time="3" qpos="33"/>
    </keyframe>
  </mujoco>)";

  // model with one cylinder and a hinge
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  // compile model (for testing double compilation)
  mjModel* m_child = compile ? mj_compile(child, 0) : nullptr;

  // get subtree
  mjsBody* body = mjs_findBody(child, "body");
  EXPECT_THAT(body, NotNull());

  // delete subtree
  EXPECT_THAT(mjs_delete(child, body->element), 0);

  // try saving to XML before compiling again
  std::array<char, 1024> e;
  std::array<char, 1024> s;
  EXPECT_EQ(mj_saveXMLString(child, s.data(), 1024, e.data(), 1024), -1);
  EXPECT_THAT(e.data(), compile
                            ? HasSubstr("must be recompiled")
                            : HasSubstr("Only compiled model can be written"));

  // compile new model
  mjModel* m_detached = mj_compile(child, 0);
  EXPECT_THAT(m_detached, NotNull());

  // compare with expected XML
  MjModelPtr m_expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_detached, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  // destroy everything
  mj_deleteSpec(child);
  mj_deleteModel(m_detached);
  if (m_child) mj_deleteModel(m_child);
}

TEST_F(MujocoTest, DetachBody) {
  TestDetachBody(/*compile=*/false);
  TestDetachBody(/*compile=*/true);
}

TEST_F(MujocoTest, DeleteBodyAfterCompile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler discardvisual="true"/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 1"/>
      <site name="site"/>
      <body>
        <geom size=".1" contype="0" conaffinity="0"/>
      </body>
      <body name="body">
        <geom name="geom" size=".1"/>
        <site name="site_body"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="geom" geom2="floor"/>
    </contact>
    <tendon>
      <spatial>
        <site site="site"/>
        <geom geom="geom"/>
        <site site="site_body"/>
      </spatial>
    </tendon>
    <sensor>
      <framepos objtype="geom" objname="geom"/>
    </sensor>
  </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // the elements that reference the body are deleted with it, also after the
  // visual geom that precedes it was discarded by the compiler
  mjsBody* body = mjs_findBody(spec, "body");
  EXPECT_EQ(mjs_delete(spec, body->element), 0) << mjs_getError(spec);
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_PAIR), IsNull());
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_TENDON), IsNull());
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_SENSOR), IsNull());
  mjModel* deleted = mj_compile(spec, nullptr);
  ASSERT_THAT(deleted, NotNull()) << mjs_getError(spec);

  // same for elements added after compiling
  mjsBody* added = mjs_addBody(mjs_findBody(spec, "world"), nullptr);
  mjs_setName(added->element, "added");
  mjsGeom* geom = mjs_addGeom(added, nullptr);
  mjs_setName(geom->element, "added");
  mjsPair* pair = mjs_addPair(spec, nullptr);
  mjs_setString(pair->geomname1, "added");
  mjs_setString(pair->geomname2, "floor");
  EXPECT_EQ(mjs_delete(spec, added->element), 0) << mjs_getError(spec);
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_PAIR), IsNull());
  mjModel* readded = mj_compile(spec, nullptr);
  EXPECT_THAT(readded, NotNull()) << mjs_getError(spec);

  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteModel(deleted);
  mj_deleteModel(readded);
}

void TestDeleteFrame(bool compile) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="hinge"/>
        <geom name="geom" size=".1"/>
        <frame name="frame" pos="1 0 0">
          <joint name="slide" type="slide"/>
          <geom name="in_frame" size=".1"/>
          <site name="in_frame"/>
          <camera name="in_frame"/>
          <light name="in_frame"/>
          <frame name="nested" pos="0 1 0">
            <geom name="in_nested" size=".1"/>
            <body name="in_nested">
              <joint name="in_body"/>
              <geom name="in_body" size=".1"/>
            </body>
          </frame>
        </frame>
        <frame name="sibling" pos="0 0 1">
          <geom name="in_sibling" size=".1"/>
        </frame>
      </body>
    </worldbody>

    <sensor>
      <framepos name="geom" objtype="geom" objname="geom"/>
      <framepos name="in_frame" objtype="site" objname="in_frame"/>
      <framepos name="in_body" objtype="geom" objname="in_body"/>
    </sensor>

    <actuator>
      <motor name="hinge" joint="hinge"/>
      <motor name="slide" joint="slide"/>
    </actuator>

    <keyframe>
      <key name="key" qpos="1 2 3"/>
    </keyframe>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="hinge"/>
        <geom name="geom" size=".1"/>
        <frame name="sibling" pos="0 0 1">
          <geom name="in_sibling" size=".1"/>
        </frame>
      </body>
    </worldbody>

    <sensor>
      <framepos name="geom" objtype="geom" objname="geom"/>
    </sensor>

    <actuator>
      <motor name="hinge" joint="hinge"/>
    </actuator>

    <keyframe>
      <key name="key" qpos="1"/>
    </keyframe>
  </mujoco>)";

  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();

  // compile model (for testing double compilation)
  mjModel* m_before = compile ? mj_compile(spec, 0) : nullptr;

  // count the elements that are released
  int released = 0;
  auto cleanup = +[](const void* data) {
    *static_cast<int*>(const_cast<void*>(data)) += 1;
  };
  mjsFrame* frame = mjs_findFrame(spec, "frame");
  EXPECT_THAT(frame, NotNull());
  for (mjsElement* element :
       {frame->element, mjs_findFrame(spec, "nested")->element,
        mjs_findElement(spec, mjOBJ_GEOM, "in_frame"),
        mjs_findElement(spec, mjOBJ_GEOM, "in_body"),
        mjs_findElement(spec, mjOBJ_GEOM, "in_sibling")}) {
    mjs_setUserValueWithCleanup(element, "released", &released, cleanup);
  }

  // delete the frame, everything inside it and everything that references it
  EXPECT_EQ(mjs_delete(spec, frame->element), 0);
  EXPECT_THAT(mjs_findFrame(spec, "frame"), IsNull());
  EXPECT_THAT(mjs_findFrame(spec, "nested"), IsNull());
  EXPECT_THAT(mjs_findFrame(spec, "sibling"), NotNull());

  // the frame and its contents are released immediately, the sibling is not
  EXPECT_EQ(released, 4);

  // compare with expected XML
  mjModel* m_deleted = mj_compile(spec, 0);
  EXPECT_THAT(m_deleted, NotNull()) << mjs_getError(spec);
  MjModelPtr m_expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_deleted, m_expected.get(), field), tol)
      << "Expected and deleted models are different!\n"
      << "Different field: " << field << '\n';

  // destroy everything
  mj_deleteSpec(spec);
  EXPECT_EQ(released, 5);
  mj_deleteModel(m_deleted);
  if (m_before) mj_deleteModel(m_before);
}

TEST_F(MujocoTest, DeleteFrame) {
  TestDeleteFrame(/*compile=*/false);
  TestDeleteFrame(/*compile=*/true);
}

TEST_F(MujocoTest, DeleteFrameInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint/>
        <geom size=".5" mass="2"/>
        <frame name="outer" pos="1 0 0">
          <frame name="inner" pos="0 1 0">
            <inertial pos="0 0 1" mass="1" diaginertia="1 1 1"/>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->body_mass[1], 1);
  EXPECT_EQ(model->body_ipos[3], 1);

  // the inertial is deleted along with the frame enclosing it
  EXPECT_EQ(mjs_delete(spec, mjs_findFrame(spec, "outer")->element), 0);
  mjModel* deleted = mj_compile(spec, 0);
  ASSERT_THAT(deleted, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(deleted->body_mass[1], 2);
  EXPECT_EQ(deleted->body_ipos[3], 0);

  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteModel(deleted);
}

TEST_F(MujocoTest, DeleteFramePlugin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.elasticity.cable"/>
      <plugin plugin="mujoco.sdf.torus">
        <instance name="torus"/>
      </plugin>
    </extension>

    <asset>
      <mesh name="torus">
        <plugin instance="torus"/>
      </mesh>
    </asset>

    <worldbody>
      <frame name="frame">
        <geom type="sdf" mesh="torus">
          <plugin plugin="mujoco.sdf.torus"/>
        </geom>
        <body>
          <geom size=".1"/>
          <plugin plugin="mujoco.elasticity.cable"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, 0, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->nplugin, 3);

  // the plugin instances created by the geom and the body are deleted with them
  EXPECT_EQ(mjs_delete(spec, mjs_findFrame(spec, "frame")->element), 0);
  mjModel* newmodel = mj_compile(spec, nullptr);
  ASSERT_THAT(newmodel, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(newmodel->nplugin, 1);

  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteModel(newmodel);
}

TEST_F(MujocoTest, AttachToSite) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <site name="site" pos="1 0 0" quat="0 1 0 0"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="sphere">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <site name="site" pos="1 0 0" quat="0 1 0 0"/>
      <frame pos="1 0 0" quat="0 1 0 0">
        <body name="attached-sphere-1">
          <joint type="slide"/>
          <geom size=".1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  mjsBody* world = mjs_findBody(parent, "world");
  EXPECT_THAT(world, NotNull());
  mjsSite* site = mjs_asSite(mjs_firstChild(world, mjOBJ_SITE, 0));
  EXPECT_THAT(site, NotNull());
  mjsBody* body = mjs_findBody(child, "sphere");
  EXPECT_THAT(body, NotNull());
  mjsBody* attached =
      mjs_asBody(mjs_attach(site->element, body->element, "attached-", "-1"));
  EXPECT_THAT(attached, NotNull());

  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, AttachFrameToSite) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <site name="site" pos="1 0 0" quat="0 1 0 0"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <frame name="frame">
        <body name="sphere">
          <joint type="slide"/>
          <geom size=".1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <site name="site" pos="1 0 0" quat="0 1 0 0"/>
      <frame pos="1 0 0" quat="0 1 0 0">
        <body name="attached-sphere-1">
          <joint type="slide"/>
          <geom size=".1"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  mjsBody* world = mjs_findBody(parent, "world");
  EXPECT_THAT(world, NotNull());
  mjsSite* site = mjs_asSite(mjs_firstChild(world, mjOBJ_SITE, 0));
  EXPECT_THAT(site, NotNull());
  mjsFrame* frame = mjs_findFrame(child, "frame");
  EXPECT_THAT(frame, NotNull());
  mjsFrame* attached =
      mjs_asFrame(mjs_attach(site->element, frame->element, "attached-", "-1"));
  EXPECT_THAT(attached, NotNull());

  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

// an operation which fails leaves the spec as it was, keyframes included
TEST_F(MujocoTest, FailedOperationLeavesKeyframes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nkey="3"/>
    <asset>
      <mesh name="mesh" file="this_file_does_not_exist.obj"/>
    </asset>
    <worldbody>
      <body name="body">
        <joint/>
        <geom type="mesh" mesh="mesh"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjSpec* copy = mj_copySpec(spec);

  EXPECT_EQ(mjs_adoptInertial(mjs_findBody(spec, "body"), nullptr), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("this_file_does_not_exist.obj"));
  EXPECT_THAT(mjs_firstElement(spec, mjOBJ_KEY), IsNull());
  EXPECT_THAT(CompareSpec(copy, spec), IsEmpty());

  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// the inertial which compilation infers for a body can be adopted: it becomes
// part of the spec and no longer follows the geoms
TEST_F(MujocoTest, AdoptInertial) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree" settotalmass="10"/>
    <asset>
      <mesh name="cube" file="cube.obj" scale=".1 .1 .1"/>
    </asset>
    <worldbody>
      <body name="arm" pos="0 0 1">
        <joint/>
        <frame pos=".1 0 0" euler="0 0 30">
          <geom name="capsule" type="capsule" fromto="0 0 0 .3 0 0" size=".05"/>
        </frame>
        <geom name="cube" type="mesh" mesh="cube" pos="0 .2 0"/>
      </body>
      <body name="free" pos="1 0 1">
        <freejoint align="true"/>
        <geom name="box" type="box" size=".1 .2 .3" pos=".1 .2 .3" euler="10 20 30"/>
        <geom name="ball" size=".1" pos="-.2 0 0"/>
      </body>
      <body name="given" pos="2 0 1">
        <joint/>
        <inertial pos="0 0 .1" mass="2" fullinertia="1 1 1 0 0 0"/>
        <geom size=".1"/>
      </body>
      <body name="inferred" pos="3 0 1">
        <joint/>
        <geom name="inferred" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  // settotalmass is deprecated
  mock_warning_handler.ExpectWarnings("settotalmass");

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m0 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m0, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);
  mjsBody* arm = mjs_findBody(spec, "arm");
  mjsBody* free = mjs_findBody(spec, "free");
  mjsBody* given = mjs_findBody(spec, "given");
  mjsBody* inferred = mjs_findBody(spec, "inferred");

  // assets are read as when compiling: this mesh is in the VFS only
  EXPECT_EQ(mjs_adoptInertial(arm, nullptr), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("cube.obj"));
  EXPECT_FALSE(arm->explicitinertial);

  EXPECT_EQ(mjs_adoptInertial(arm, vfs.get()), 0);
  EXPECT_EQ(mjs_adoptInertial(free, vfs.get()), 0);
  EXPECT_TRUE(arm->explicitinertial);
  EXPECT_TRUE(free->explicitinertial);

  // an inertial which was given is left as it was
  EXPECT_EQ(mjs_adoptInertial(given, vfs.get()), 0);
  EXPECT_THAT(CompareSpec(authored, spec), testing::Not(HasSubstr("'given'")));

  // the model is the same, whichever bodies were adopted: the values are
  // those before the total mass is set
  mjModel* m1 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  std::string field;
  EXPECT_LE(CompareModel(m0, m1, field), MjTol(1e-12, 1e-5)) << field;

  // an adopted inertial does not follow the geoms, an inferred one does
  spec->compiler.settotalmass = -1;
  mjModel* m2 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "capsule"))->size[0] *= 2;
  mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "inferred"))->size[0] *= 2;
  mjModel* m3 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m3, NotNull()) << mjs_getError(spec);
  int arm_id = mjs_getId(arm->element);
  int inferred_id = mjs_getId(inferred->element);
  EXPECT_EQ(m3->body_mass[arm_id], m2->body_mass[arm_id]);
  EXPECT_NEAR(m3->body_mass[inferred_id], 8 * m2->body_mass[inferred_id],
              MjTol(1e-10, 1e-3));

  // there is nothing to adopt when every inertia is inferred, whatever is given
  spec->compiler.inertiafromgeom = mjINERTIAFROMGEOM_TRUE;
  EXPECT_EQ(mjs_adoptInertial(inferred, vfs.get()), -1);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("inertiafromgeom"));
  EXPECT_FALSE(inferred->explicitinertial);
  EXPECT_EQ(mjs_adoptInertial(mjs_findBody(spec, "world"), vfs.get()), -1);

  mj_deleteModel(m0);
  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteModel(m3);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, BodyToFrame) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <frame name="frame" pos="1 2 3"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="sphere">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <camera pos="0 0 0" quat="1 0 0 0"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <frame name="frame" pos="1 2 3">
          <body name="attached-sphere-1">
            <joint type="slide"/>
            <geom size=".1"/>
          </body>
          <frame name="attached-world-2">
            <body name="attached-sphere-2">
              <joint type="slide"/>
              <geom size=".1"/>
            </body>
            <camera pos="0 0 0" quat="1 0 0 0"/>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child1 = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child1, NotNull()) << er.data();
  mjSpec* child2 = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child2, NotNull()) << er.data();

  // attach a body to the frame
  mjsFrame* frame = mjs_findFrame(parent, "frame");
  EXPECT_THAT(frame, NotNull());
  mjsBody* body = mjs_findBody(child1, "sphere");
  EXPECT_THAT(body, NotNull());
  mjsBody* attached =
      mjs_asBody(mjs_attach(frame->element, body->element, "attached-", "-1"));
  EXPECT_THAT(attached, NotNull());
  mjModel* model1 = mj_compile(parent, 0);
  EXPECT_THAT(model1, NotNull());

  // attach the world to the same frame and convert it to a frame
  mjsBody* world = mjs_findBody(child2, "world");
  EXPECT_THAT(world, NotNull());
  mjsBody* child_world =
      mjs_asBody(mjs_attach(frame->element, world->element, "attached-", "-2"));
  EXPECT_THAT(child_world, NotNull());
  mjsFrame* frame_world = mjs_bodyToFrame(&child_world);
  EXPECT_THAT(frame_world, NotNull());
  EXPECT_THAT(child_world, IsNull());

  // compile and compare
  mjModel* model2 = mj_compile(parent, 0);
  EXPECT_THAT(model2, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model2, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(parent);
  mj_deleteSpec(child1);
  mj_deleteSpec(child2);
  mj_deleteModel(model1);
  mj_deleteModel(model2);
}

TEST_F(MujocoTest, BodyToFrameOrientation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <body name="child" pos="0 0 1" euler="0 0 90">
          <geom size=".1" pos="1 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_expected[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <frame pos="0 0 1" euler="0 0 90">
          <geom size=".1" pos="1 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull());

  // the frame has the orientation of the body
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  MjModelPtr expected = LoadModelFromString(xml_expected, er.data(), er.size());
  ASSERT_THAT(expected.get(), NotNull()) << er.data();
  std::string field = "";
  EXPECT_LE(CompareModel(model, expected.get(), field), 0)
      << "Expected and converted models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, BodyToFrameNestedFrames) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <body name="child" pos="1 0 0">
          <frame pos="0 1 0" euler="0 0 90">
            <joint pos="1 0 0" axis="1 0 0"/>
            <geom size=".1" pos="1 0 0"/>
            <camera pos="1 0 0"/>
            <frame pos="0 0 1">
              <site pos="1 0 0"/>
              <light pos="1 0 0" dir="1 0 0"/>
            </frame>
            <body name="grandchild" pos="1 0 0">
              <geom size=".1"/>
            </body>
          </frame>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_expected[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <frame pos="1 0 0">
          <frame pos="0 1 0" euler="0 0 90">
            <joint pos="1 0 0" axis="1 0 0"/>
            <geom size=".1" pos="1 0 0"/>
            <camera pos="1 0 0"/>
            <frame pos="0 0 1">
              <site pos="1 0 0"/>
              <light pos="1 0 0" dir="1 0 0"/>
            </frame>
            <body name="grandchild" pos="1 0 0">
              <geom size=".1"/>
            </body>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);

  // elements in frames of the converted body stay in them
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  MjModelPtr expected = LoadModelFromString(xml_expected, er.data(), er.size());
  ASSERT_THAT(expected.get(), NotNull()) << er.data();
  std::string field = "";
  EXPECT_LE(CompareModel(model, expected.get(), field), 0)
      << "Expected and converted models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, BodyToFrameWorld) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  EXPECT_THAT(mjs_bodyToFrame(&world), IsNull());
  EXPECT_THAT(world, NotNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("world body"));
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, BodyToFrameWithInertial) {
  static constexpr char xml_child[] = R"(
    <mujoco>
      <worldbody>
        <body name="parent">
          <body name="child">
            <inertial mass="1" pos="0 0 0" quat="1 0 0 0" diaginertia="1 2 3"/>
          </body>
        </body>
      </worldbody>
    </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull());
  mjsBody* parent = mjs_findBody(spec, "parent");
  EXPECT_THAT(parent, NotNull());
  mjsBody* child = mjs_findBody(spec, "child");
  EXPECT_THAT(child, NotNull());
  mjs_bodyToFrame(&child);
  EXPECT_THAT(parent->mass, 1);
  EXPECT_THAT(parent->fullinertia[0], 1);
  EXPECT_THAT(parent->fullinertia[1], 2);
  EXPECT_THAT(parent->fullinertia[2], 3);

  // the merged inertial compiles
  mjModel* merged = mj_compile(spec, 0);
  ASSERT_THAT(merged, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(merged->body_mass[1], 1);
  mj_deleteSpec(spec);
  mj_deleteModel(model);
  mj_deleteModel(merged);
}

// inertia matrix of a body in body coordinates, as (xx, yy, zz, xy, xz, yz)
std::array<mjtNum, 6> BodyInertia(const mjModel* m, int body) {
  static constexpr int row[6] = {0, 1, 2, 0, 0, 1};
  static constexpr int col[6] = {0, 1, 2, 1, 2, 2};
  const mjtNum* inertia = m->body_inertia + 3 * body;
  mjtNum mat[9];
  mju_quat2Mat(mat, m->body_iquat + 4 * body);
  std::array<mjtNum, 6> res;
  for (int i = 0; i < 6; i++) {
    res[i] = 0;
    for (int k = 0; k < 3; k++) {
      res[i] += mat[3 * row[i] + k] * inertia[k] * mat[3 * col[i] + k];
    }
  }
  return res;
}

void BodyToFrameMergeInertial(bool compile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <inertial mass="1" pos="0 0 0" diaginertia="2 3 4"/>
        <body name="child" pos="1 0 0">
          <inertial mass="1" pos="0 0 0" diaginertia="2 3 4"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  if (compile) {
    mjModel* model = mj_compile(spec, 0);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    mj_deleteModel(model);
  }
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);

  // two unit masses, 1 apart along x
  const double ipos[3] = {.5, 0, 0};
  const double inertia[6] = {4, 6.5, 8.5, 0, 0, 0};
  mjsBody* parent = mjs_findBody(spec, "parent");
  ASSERT_THAT(parent, NotNull());
  EXPECT_EQ(parent->mass, 2);
  EXPECT_THAT(parent->ipos, ElementsAreArray(ipos));
  EXPECT_THAT(parent->fullinertia, ElementsAreArray(inertia));

  // the merged inertial compiles
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->body_mass[1], 2);
  std::array<mjtNum, 6> body_inertia = BodyInertia(model, 1);
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(model->body_ipos[3 + i], ipos[i]);
  }
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(body_inertia[i], inertia[i], MjTol(1e-12, 1e-5));
  }

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, BodyToFrameMergeInertial) {
  BodyToFrameMergeInertial(/*compile=*/false);
  BodyToFrameMergeInertial(/*compile=*/true);
}

TEST_F(MujocoTest, BodyToFrameInertialPose) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <joint/>
        <inertial mass="1" pos=".1 .2 .3" quat="1 2 3 4" diaginertia="2 3 4"/>
        <frame pos="0 0 1" euler="0 0 30">
          <frame pos="0 1 0" axisangle="1 1 0 60">
            <body name="child" pos="1 0 0" euler="0 40 0">
              <inertial mass="2" pos="0 .5 0" euler="50 0 0" diaginertia="3 4 5"/>
            </body>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  // fuse the static child into the parent at compile time
  std::array<char, 1000> er;
  mjSpec* spec_fused = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec_fused, NotNull()) << er.data();
  spec_fused->compiler.fusestatic = 1;
  mjModel* fused = mj_compile(spec_fused, 0);
  ASSERT_THAT(fused, NotNull()) << mjs_getError(spec_fused);

  // convert the child to a frame
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // principal axes are not unique, compare inertia matrices rather than iquat
  ASSERT_EQ(model->nbody, 2);
  ASSERT_EQ(fused->nbody, 2);
  EXPECT_EQ(model->body_mass[1], fused->body_mass[1]);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(model->body_ipos[3 + i], fused->body_ipos[3 + i],
                MjTol(1e-14, 1e-6));
  }
  std::array<mjtNum, 6> inertia = BodyInertia(model, 1);
  std::array<mjtNum, 6> expected = BodyInertia(fused, 1);
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(inertia[i], expected[i], MjTol(1e-13, 1e-6));
  }

  mj_deleteSpec(spec_fused);
  mj_deleteSpec(spec);
  mj_deleteModel(fused);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, BodyToFrameInertialInFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <joint/>
        <frame pos="0 0 1" euler="0 0 30">
          <inertial mass="1" pos=".1 .2 .3" euler="10 20 30" diaginertia="2 3 4"/>
        </frame>
        <body name="child" pos="1 0 0" euler="0 40 0">
          <frame pos="0 1 0" axisangle="1 1 0 60">
            <inertial mass="2" pos="0 .5 0" euler="50 0 0" diaginertia="3 4 5"/>
          </frame>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  // fuse the static child into the parent at compile time
  std::array<char, 1000> er;
  mjSpec* spec_fused = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec_fused, NotNull()) << er.data();
  spec_fused->compiler.fusestatic = 1;
  mjModel* fused = mj_compile(spec_fused, 0);
  ASSERT_THAT(fused, NotNull()) << mjs_getError(spec_fused);

  // convert the child to a frame
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // both inertials follow their frames, the merged one is in body coordinates
  ASSERT_EQ(model->nbody, 2);
  ASSERT_EQ(fused->nbody, 2);
  EXPECT_EQ(model->body_mass[1], fused->body_mass[1]);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(model->body_ipos[3 + i], fused->body_ipos[3 + i],
                MjTol(1e-14, 1e-6));
  }
  std::array<mjtNum, 6> inertia = BodyInertia(model, 1);
  std::array<mjtNum, 6> expected = BodyInertia(fused, 1);
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(inertia[i], expected[i], MjTol(1e-13, 1e-6));
  }

  mj_deleteSpec(spec_fused);
  mj_deleteSpec(spec);
  mj_deleteModel(fused);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, BodyToFrameFullInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <body name="child" pos="0 0 1">
          <inertial mass="1" pos="0 1 0" fullinertia="2 3 4 .1 .2 .3"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());

  // rotate the inertial frame 90 degrees around x, which MJCF does not allow
  // with fullinertia
  child->iquat[1] = 1;
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);

  // the full inertia of the child in the coordinates of the parent
  const double ipos[3] = {0, 1, 1};
  const double inertia[6] = {2, 4, 3, -.2, .1, -.3};
  mjsBody* parent = mjs_findBody(spec, "parent");
  ASSERT_THAT(parent, NotNull());
  EXPECT_EQ(parent->mass, 1);
  EXPECT_THAT(parent->ipos, ElementsAreArray(ipos));
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(parent->fullinertia[i], inertia[i], 1e-14);
  }

  mj_deleteSpec(spec);
}

// an inertial which is inferred from geoms is merged like one which is given,
// whether or not the spec was compiled before
TEST_F(MujocoTest, BodyToFrameInferredInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="inferred">
        <joint/>
        <geom size=".1"/>
        <body name="child1" pos="1 0 0">
          <geom size=".1"/>
        </body>
      </body>
      <body name="given">
        <joint/>
        <inertial mass="1" pos="0 0 0" diaginertia="2 3 4"/>
        <body name="child2" pos="1 0 0">
          <geom size=".1"/>
        </body>
      </body>
      <body name="geoms">
        <joint/>
        <geom size=".1"/>
        <body name="child3" pos="1 0 0">
          <inertial mass="2" pos="0 0 0" diaginertia="1 1 1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  // a sphere: mass and inertia about its center
  const double ms = 4.0 / 3.0 * mjPI * 1e-3 * 1000;
  const double is = 0.4 * ms * 1e-2;

  for (bool compile_first : {false, true}) {
    std::array<char, 1000> er;
    mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
    ASSERT_THAT(spec, NotNull()) << er.data();
    if (compile_first) {
      mjModel* compiled = mj_compile(spec, 0);
      ASSERT_THAT(compiled, NotNull()) << mjs_getError(spec);
      mj_deleteModel(compiled);
    }
    for (const char* name : {"child1", "child2", "child3"}) {
      mjsBody* child = mjs_findBody(spec, name);
      ASSERT_THAT(child, NotNull());
      ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
    }

    // two bodies whose inertia is inferred: the parent has both geoms
    mjsBody* inferred = mjs_findBody(spec, "inferred");
    EXPECT_FALSE(inferred->explicitinertial);

    mjModel* m = mj_compile(spec, 0);
    ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
    ASSERT_EQ(m->nbody, 4);
    mjtNum tol = MjTol(1e-9, 1e-5);

    // total mass, center of mass along x, and inertia about it: xx and yy = zz
    struct Expected {
      double mass, com, ixx, iyy;
    };
    double c2 = ms / (1 + ms);
    double c3 = 2 / (ms + 2);
    Expected expected[3] = {
        {2 * ms, 0.5, 2 * is, 2 * is + 2 * ms * 0.25},
        {1 + ms, c2, 2 + is, 3 + is + c2 * c2 + ms * (1 - c2) * (1 - c2)},
        {ms + 2, c3, is + 1, is + 1 + ms * c3 * c3 + 2 * (1 - c3) * (1 - c3)},
    };
    for (int i = 0; i < 3; i++) {
      int id = i + 1;
      std::array<mjtNum, 6> inertia = BodyInertia(m, id);
      EXPECT_NEAR(m->body_mass[id], expected[i].mass, tol) << id;
      EXPECT_NEAR(m->body_ipos[3 * id], expected[i].com, tol) << id;
      EXPECT_NEAR(inertia[0], expected[i].ixx, tol) << id;
      EXPECT_NEAR(inertia[1], expected[i].iyy, tol) << id;
      EXPECT_NEAR(inertia[2], expected[i].iyy + (i == 1 ? 1 : 0), tol) << id;
    }

    mj_deleteModel(m);
    mj_deleteSpec(spec);
  }
}

// the inertia which an earlier compilation inferred for the parent is not used
// once the parent has lost its geoms
TEST_F(MujocoTest, BodyToFrameStaleInferredInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <joint type="slide"/>
        <geom name="geom" size=".1" mass="2"/>
        <body name="child" pos="1 0 0">
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  for (bool compile_first : {false, true}) {
    SCOPED_TRACE(compile_first ? "compiled first" : "not compiled first");
    mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << error.data();
    if (compile_first) {
      mjModel* m = mj_compile(spec, nullptr);
      ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
      EXPECT_EQ(m->body_mass[1], 2);
      mj_deleteModel(m);
    }

    // the parent loses its geom, then receives the inertial of the child
    EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_GEOM, "geom")), 0);
    mjsBody* child = mjs_findBody(spec, "child");
    EXPECT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
    mjModel* m = mj_compile(spec, nullptr);
    ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
    EXPECT_EQ(m->nbody, 2);
    EXPECT_EQ(m->body_mass[1], 1);
    EXPECT_EQ(m->body_ipos[3], 1);
    mj_deleteModel(m);
    mj_deleteSpec(spec);
  }
}

// inferred inertia needs the assets: without them the conversion fails and
// says how to proceed
TEST_F(MujocoTest, BodyToFrameInferredInertiaAssets) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" file="cube.obj" scale=".1 .1 .1"/>
    </asset>
    <worldbody>
      <body name="parent">
        <joint/>
        <inertial mass="1" pos="0 0 0" diaginertia="2 3 4"/>
        <body name="child" pos="1 0 0">
          <geom type="mesh" mesh="cube"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjSpec* authored = mj_copySpec(spec);
  mjModel* m0 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m0, NotNull()) << mjs_getError(spec);

  // the mesh cannot be read without the VFS: nothing is changed
  mjsBody* child = mjs_findBody(spec, "child");
  EXPECT_THAT(mjs_bodyToFrame(&child), IsNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("mjs_adoptInertial"));
  EXPECT_THAT(CompareSpec(authored, spec), IsEmpty());

  // once the inertial is adopted, the conversion does not need the mesh
  ASSERT_THAT(child, NotNull());
  EXPECT_EQ(mjs_adoptInertial(child, vfs.get()), 0);
  ASSERT_THAT(mjs_bodyToFrame(&child), NotNull()) << mjs_getError(spec);
  mjModel* m1 = mj_compile(spec, vfs.get());
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(m1->body_mass[1], m0->body_mass[1] + m0->body_mass[2],
              MjTol(1e-12, 1e-6));

  mj_deleteModel(m0);
  mj_deleteModel(m1);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, BodyToFrameInvalidInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <inertial mass="1" pos="0 0 0" diaginertia="2 3 4"/>
        <body name="child">
          <inertial mass="1" pos="0 0 0" diaginertia="2 3 4" fullinertia="2 3 4 0 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjsBody* child = mjs_findBody(spec, "child");
  ASSERT_THAT(child, NotNull());

  // the conversion fails like the compiler would, and changes nothing
  EXPECT_THAT(mjs_bodyToFrame(&child), IsNull());
  EXPECT_THAT(mjs_getError(spec), HasSubstr("cannot both be specified"));
  EXPECT_THAT(child, NotNull());
  EXPECT_EQ(mjs_findBody(spec, "parent")->mass, 1);
  EXPECT_THAT(mjs_findBody(spec, "child"), NotNull());
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, BodyToFrameAttach) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <body name="child" pos="1 0 0">
          <geom name="geom" size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <frame pos="1 0 0">
        <geom name="attached-geom" size=".1"/>
      </frame>
    </worldbody>
  </mujoco>)";

  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();
  mjsBody* parent = mjs_findBody(spec, "parent");
  EXPECT_THAT(parent, NotNull());
  mjsBody* child = mjs_findBody(spec, "child");
  EXPECT_THAT(child, NotNull());

  // the frame belongs to the parent of the converted body
  mjsFrame* frame = mjs_bodyToFrame(&child);
  EXPECT_THAT(frame, NotNull());
  EXPECT_EQ(mjs_getParent(frame->element), parent);

  // attach the frame to another spec
  mjSpec* other = mj_makeSpec();
  mjsBody* world = mjs_findBody(other, "world");
  EXPECT_THAT(mjs_attach(world->element, frame->element, "attached-", ""),
              NotNull());

  // compile and compare
  mjModel* model = mj_compile(other, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(spec);
  mj_deleteSpec(other);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, AttachSpecToSite) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <site name="site" pos="1 2 3"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="sphere">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <camera pos="0 0 0" quat="1 0 0 0"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <site name="site" pos="1 2 3"/>
        <frame name="attached-world-1" pos="1 2 3">
          <body name="attached-sphere-1">
            <joint type="slide"/>
            <geom size=".1"/>
          </body>
          <camera pos="0 0 0" quat="1 0 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();
  mjsSite* site = mjs_asSite(mjs_findElement(parent, mjOBJ_SITE, "site"));
  EXPECT_THAT(site, NotNull());

  // add a frame to the child
  mjsBody* world = mjs_findBody(child, "world");
  EXPECT_THAT(world, NotNull());
  mjsFrame* frame = mjs_addFrame(world, 0);
  EXPECT_THAT(frame, NotNull());
  mjs_setName(frame->element, "world");
  mjs_setFrame(mjs_firstChild(world, mjOBJ_BODY, 0), frame);
  mjs_setFrame(mjs_firstChild(world, mjOBJ_CAMERA, 0), frame);

  // attach the entire spec to the site
  mjsFrame* worldframe =
      mjs_asFrame(mjs_attach(site->element, frame->element, "attached-", "-1"));
  EXPECT_THAT(worldframe, NotNull());

  // compile and compare
  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  // check that the child world still exists
  mjsBody* child_world = mjs_findBody(child, "world");
  EXPECT_THAT(child_world, NotNull());

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, AttachSpecToBody) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="body"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="sphere">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <camera pos="0 0 0" quat="1 0 0 0"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_result[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <frame name="attached-world-1" pos="1 2 3">
          <body name="attached-sphere-1">
            <joint type="slide"/>
            <geom size=".1"/>
          </body>
          <camera pos="0 0 0" quat="1 0 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();
  mjsBody* body = mjs_findBody(parent, "body");
  EXPECT_THAT(body, NotNull());

  // add a frame to the child
  mjsBody* world = mjs_findBody(child, "world");
  EXPECT_THAT(world, NotNull());
  mjsFrame* frame = mjs_addFrame(world, 0);
  EXPECT_THAT(frame, NotNull());
  mjs_setName(frame->element, "world");
  mjs_setFrame(mjs_firstChild(world, mjOBJ_BODY, 0), frame);
  mjs_setFrame(mjs_firstChild(world, mjOBJ_CAMERA, 0), frame);

  // attach the entire spec to the site
  mjsFrame* worldframe =
      mjs_asFrame(mjs_attach(body->element, frame->element, "attached-", "-1"));
  EXPECT_THAT(worldframe, NotNull());
  worldframe->pos[0] = 1;
  worldframe->pos[1] = 2;
  worldframe->pos[2] = 3;

  // compile and compare
  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr expected = LoadModelFromString(xml_result, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  // check that the child world still exists
  mjsBody* child_world = mjs_findBody(child, "world");
  EXPECT_THAT(child_world, NotNull());

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, PreserveState) {
  std::array<char, 1000> er;
  std::string field = "";

  static constexpr char xml_full[] = R"(
  <mujoco>
    <worldbody>
      <body name="detachable" pos="1 0 0">
        <joint type="hinge" axis="0 0 1" name="hinge"/>
        <geom type="sphere" size=".1"/>
      </body>
      <body name="persistent">
        <joint type="slide" axis="0 0 1" name="slide"/>
        <geom type="sphere" size=".2"/>
      </body>
      <body name="mocap_detach" mocap="true"/>
      <body name="mocap" mocap="true"/>
    </worldbody>
    <actuator>
      <position name="hinge" joint="hinge" timeconst=".01"/>
      <position name="slide" joint="slide" timeconst=".01"/>
    </actuator>
  </mujoco>)";

  static constexpr char xml_expected[] = R"(
  <mujoco>
    <worldbody>
      <body name="persistent">
        <joint type="slide" axis="0 0 1" name="slide"/>
        <geom type="sphere" size=".2"/>
      </body>
      <body name="newbody" pos="2 0 0">
        <joint type="slide" axis="0 0 1"/>
        <geom type="sphere" size=".3"/>
      </body>
      <body name="mocap" mocap="true"/>
    </worldbody>
    <actuator>
      <position name="slide" joint="slide" timeconst=".01"/>
    </actuator>
  </mujoco>)";

  // load spec
  mjSpec* spec = mj_parseXMLString(xml_full, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();

  // compile models
  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull());
  MjModelPtr m_expected =
      LoadModelFromString(xml_expected, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull());

  // create data
  mjData* data = mj_makeData(model);
  EXPECT_THAT(data, NotNull());
  MjDataPtr d_expected = MakeData(m_expected);
  EXPECT_THAT(d_expected.get(), NotNull());

  // set ctrl
  data->ctrl[0] = 1;
  data->ctrl[1] = 2;
  d_expected.get()->ctrl[0] = 2;

  // set mocap
  data->mocap_pos[3] = 1;
  data->mocap_quat[4] = 0;
  data->mocap_quat[5] = 1;
  d_expected.get()->mocap_pos[0] = 1;
  d_expected.get()->mocap_quat[0] = 0;
  d_expected.get()->mocap_quat[1] = 1;

  // step models
  mj_step(model, data);
  mj_step(m_expected.get(), d_expected.get());
  EXPECT_THAT(data->time, model->opt.timestep);

  // detach subtree
  mjsBody* body = mjs_findBody(spec, "detachable");
  EXPECT_THAT(body, NotNull());
  EXPECT_THAT(mjs_delete(spec, body->element), 0);

  // detach mocap
  mjsBody* mocap_body = mjs_findBody(spec, "mocap_detach");
  EXPECT_THAT(mocap_body, NotNull());
  EXPECT_THAT(mjs_delete(spec, mocap_body->element), 0);

  // add body
  mjsBody* newbody = mjs_addBody(mjs_findBody(spec, "world"), 0);
  EXPECT_THAT(newbody, NotNull());

  // add geom and joint
  mjsGeom* geom = mjs_addGeom(newbody, 0);
  mjsJoint* joint = mjs_addJoint(newbody, 0);

  // set properties
  newbody->pos[0] = 2;
  geom->size[0] = .3;
  joint->type = mjJNT_SLIDE;
  joint->axis[0] = 0;
  joint->axis[1] = 0;
  joint->axis[2] = 1;
  joint->ref = d_expected.get()->qpos[m_expected->nq - 1];

  // compile new model
  mj_recompile(spec, 0, model, data);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(data->time, model->opt.timestep);

  // compare qpos
  EXPECT_EQ(model->nq, m_expected->nq);
  for (int i = 0; i < model->nq; ++i) {
    EXPECT_EQ(data->qpos[i], d_expected.get()->qpos[i]) << i;
  }

  // compare qvel
  EXPECT_EQ(model->nv, m_expected->nv);
  for (int i = 0; i < model->nv - 1; ++i) {
    EXPECT_EQ(data->qvel[i], d_expected.get()->qvel[i]) << i;
  }

  // second body was added after stepping so qvel should be zero
  EXPECT_EQ(data->qvel[model->nv - 1], 0);

  // compare act
  EXPECT_EQ(model->na, m_expected->na);
  for (int i = 0; i < model->na; ++i) {
    EXPECT_EQ(data->act[i], d_expected.get()->act[i]) << i;
  }

  // compare mocap
  EXPECT_EQ(model->nmocap, m_expected->nmocap);
  for (int i = 0; i < model->nmocap; ++i) {
    for (int j = 0; j < 3; ++j) {
      EXPECT_EQ(data->mocap_pos[3 * i + j],
                d_expected.get()->mocap_pos[3 * i + j])
          << i;
    }
    for (int j = 0; j < 4; ++j) {
      EXPECT_EQ(data->mocap_quat[4 * i + j],
                d_expected.get()->mocap_quat[4 * i + j])
          << i;
    }
  }

  // check that the function is callable with no data
  mj_deleteData(data);
  mj_recompile(spec, 0, model, nullptr);

  // destroy everything
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, RecompileAttach) {
  std::array<char, 1000> er;
  std::string field = "";

  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint type="slide" axis="0 0 1"/>
        <geom size=".2"/>
      </body>
    </worldbody>
  </mujoco>)";

  mjSpec* parent = mj_makeSpec();
  EXPECT_THAT(parent, NotNull());

  mjSpec* child1 = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(child1, NotNull());
  mjSpec* child2 = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(child2, NotNull());

  mjsElement* frame1 = mjs_addFrame(mjs_findBody(parent, "world"), 0)->element;
  mjs_attach(frame1, mjs_findBody(child1, "body")->element, "child-", "-1");

  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nq, 1);
  mjData* data = mj_makeData(model);
  EXPECT_THAT(data, NotNull());

  for (int i = 0; i < 100; i++) {
    mj_step(model, data);
  }

  mjsElement* frame2 = mjs_addFrame(mjs_findBody(parent, "world"), 0)->element;
  mjs_attach(frame2, mjs_findBody(child2, "body")->element, "child-", "-2");

  EXPECT_EQ(mj_recompile(parent, 0, model, data), 0);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->nq, 2);
  EXPECT_NE(data->qpos[0], data->qpos[1]);
  EXPECT_EQ(data->qpos[1], 0);

  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteSpec(child1);
  mj_deleteSpec(child2);
  mj_deleteSpec(parent);
}

// the state is kept when a model is attached ahead of a joint of a spec which
// has keyframes
TEST_F(MujocoTest, RecompileAttachKeepsState) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
  mjData* data = mj_makeData(model);
  data->qpos[0] = 5;
  data->qpos[1] = 7;
  data->qvel[0] = 50;
  data->qvel[1] = 70;

  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                         mjs_findBody(child, "b")->element, "c_", ""),
              NotNull())
      << mjs_getError(parent);
  ASSERT_EQ(mj_recompile(parent, nullptr, model, data), 0)
      << mjs_getError(parent);
  ASSERT_EQ(model->nq, 3);
  EXPECT_THAT(AsVector(data->qpos, 3), ElementsAreArray({5, 0, 7}));
  EXPECT_THAT(AsVector(data->qvel, 3), ElementsAreArray({50, 0, 70}));

  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// the elements of a child attached by reference start from their initial state
// when the parent is recompiled, as attached copies do
TEST_F(MujocoTest, RecompileAttachByReference) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="f"/>
    </worldbody>
    <equality>
      <joint joint1="a"/>
    </equality>
    <actuator>
      <general joint="a" dyntype="filter" dynprm="1"/>
    </actuator>
    <sensor>
      <jointpos joint="a" nsample="2" delay="0.01"/>
    </sensor>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <joint name="x" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="x"/>
    </equality>
    <actuator>
      <general joint="x" dyntype="filter" dynprm="1"/>
    </actuator>
    <sensor>
      <jointpos joint="x" nsample="2" delay="0.01"/>
    </sensor>
  </mujoco>
  )";

  std::array<char, 1000> er;
  for (bool compiled : {false, true}) {
    for (bool key : {false, true}) {
      std::vector<mjtNum> state[2];
      for (bool deepcopy : {false, true}) {
        mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
        ASSERT_THAT(parent, NotNull()) << er.data();
        mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
        ASSERT_THAT(child, NotNull()) << er.data();
        mjs_setDeepCopy(parent, deepcopy);
        if (key) {
          mjsKey* k = mjs_addKey(child);
          double qpos = 1, ctrl = 2, act = 3;
          mjs_setDouble(k->qpos, &qpos, 1);
          mjs_setDouble(k->ctrl, &ctrl, 1);
          mjs_setDouble(k->act, &act, 1);
        }
        if (compiled) {
          mjModel* m = mj_compile(child, nullptr);
          ASSERT_THAT(m, NotNull()) << mjs_getError(child);
          mj_deleteModel(m);
        }

        mjModel* model = mj_compile(parent, nullptr);
        ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
        mjData* data = mj_makeData(model);
        data->qpos[0] = 5;
        data->ctrl[0] = 7;
        data->act[0] = 9;
        data->eq_active[0] = 0;
        for (int i = 0; i < model->nhistory; i++) {
          data->history[i] = 10 + i;
        }

        ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f")->element,
                               mjs_findBody(child, "x")->element, "c_", ""),
                    NotNull())
            << mjs_getError(parent);
        ASSERT_EQ(mj_recompile(parent, nullptr, model, data), 0)
            << mjs_getError(parent);
        EXPECT_THAT(AsVector(data->ctrl, 2), ElementsAreArray({7, 0}));
        EXPECT_THAT(AsVector(data->act, 2), ElementsAreArray({9, 0}));
        state[deepcopy].resize(mj_stateSize(model, mjSTATE_INTEGRATION));
        mj_getState(model, data, state[deepcopy].data(), mjSTATE_INTEGRATION);

        mj_deleteData(data);
        mj_deleteModel(model);
        mj_deleteSpec(child);
        mj_deleteSpec(parent);
      }
      EXPECT_EQ(state[0], state[1])
          << "compiled " << compiled << ", key " << key;
    }
  }
}

// a joint inside a frame attached by reference starts from its initial state
// when the parent is recompiled
TEST_F(MujocoTest, RecompileAttachFrameByReference) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint type="hinge" axis="1 0 0"/>
        <joint type="hinge" axis="0 1 0"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="hinge" axis="1 0 0"/>
        <geom size=".1"/>
        <frame name="f">
          <joint type="hinge" axis="0 0 1"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  for (bool deepcopy : {false, true}) {
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjs_setDeepCopy(parent, deepcopy);

    mjModel* model = mj_compile(parent, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
    mjData* data = mj_makeData(model);
    data->qpos[0] = 5;
    data->qpos[1] = 6;
    data->qvel[0] = 50;
    data->qvel[1] = 60;

    ASSERT_THAT(mjs_attach(mjs_findBody(parent, "A")->element,
                           mjs_findFrame(child, "f")->element, "c_", ""),
                NotNull())
        << mjs_getError(parent);
    ASSERT_EQ(mj_recompile(parent, nullptr, model, data), 0)
        << mjs_getError(parent);
    ASSERT_EQ(model->nq, 3);
    EXPECT_THAT(AsVector(data->qpos, 3), ElementsAreArray({5, 6, 0}));
    EXPECT_THAT(AsVector(data->qvel, 3), ElementsAreArray({50, 60, 0}));

    mj_deleteData(data);
    mj_deleteModel(model);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

// a child attached by reference again lays out its keyframes with the bodies it
// attached before, which keep their state in the parent
TEST_F(MujocoTest, RecompileAttachByReferenceAgain) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body mocap="true"/>
      <body>
        <joint type="hinge"/>
        <geom size=".1"/>
      </body>
      <frame name="f1"/>
      <frame name="f2"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="m" mocap="true">
        <body>
          <joint type="hinge"/>
          <geom size=".1"/>
        </body>
      </body>
      <body name="y">
        <joint type="hinge"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  for (bool compiled : {false, true}) {
    std::vector<mjtNum> state[2];
    std::vector<mjtNum> keys[2];
    for (bool deepcopy : {false, true}) {
      mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
      ASSERT_THAT(parent, NotNull()) << er.data();
      mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
      ASSERT_THAT(child, NotNull()) << er.data();
      mjs_setDeepCopy(parent, deepcopy);
      if (compiled) {
        mjModel* m = mj_compile(child, nullptr);
        ASSERT_THAT(m, NotNull()) << mjs_getError(child);
        mj_deleteModel(m);
      }

      ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element,
                             mjs_findBody(child, "m")->element, "m_", ""),
                  NotNull())
          << mjs_getError(parent);
      mjModel* model = mj_compile(parent, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
      mjData* data = mj_makeData(model);
      for (int i = 0; i < model->nq; i++) {
        data->qpos[i] = 5 + i;
        data->qvel[i] = 50 + i;
      }
      for (int i = 0; i < 6 * model->nbody; i++) {
        data->xfrc_applied[i] = 100 + i;
      }
      for (int i = 0; i < 3 * model->nmocap; i++) {
        data->mocap_pos[i] = 200 + i;
      }

      ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element,
                             mjs_findBody(child, "y")->element, "y_", ""),
                  NotNull())
          << mjs_getError(parent);
      ASSERT_EQ(mj_recompile(parent, nullptr, model, data), 0)
          << mjs_getError(parent);
      ASSERT_EQ(model->nq, 3);
      EXPECT_THAT(AsVector(data->qpos, 3), ElementsAreArray({5, 6, 0}));
      state[deepcopy].resize(mj_stateSize(model, mjSTATE_INTEGRATION));
      mj_getState(model, data, state[deepcopy].data(), mjSTATE_INTEGRATION);
      keys[deepcopy] = AsVector(model->key_qpos, model->nkey * model->nq);

      mj_deleteData(data);
      mj_deleteModel(model);
      mj_deleteSpec(child);
      mj_deleteSpec(parent);
    }
    EXPECT_EQ(state[0], state[1]) << "compiled " << compiled;

    // the place of the body attached before is lost from the compiled layout
    if (compiled) {
      EXPECT_EQ(keys[0], keys[1]);
    }
  }
}

// a child attached by reference again gives its elements outside the tree to
// the attachment whose bodies they refer to, as copies do
TEST_F(MujocoTest, AttachByReferenceAgain) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a"/>
        <geom size=".1"/>
      </body>
      <frame name="f1"/>
      <frame name="f2"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <joint name="x"/>
        <geom size=".1"/>
      </body>
      <body name="y">
        <joint name="y"/>
        <geom name="y1" size=".1"/>
        <geom name="y2" size=".1"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="y1" geom2="y2"/>
    </contact>
    <tendon>
      <fixed name="t">
        <joint joint="y" coef="1"/>
      </fixed>
    </tendon>
    <equality>
      <joint joint1="y"/>
    </equality>
    <actuator>
      <general joint="x"/>
      <general name="u" joint="y"/>
    </actuator>
    <sensor>
      <jointpos joint="y"/>
    </sensor>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjModel* model[2];
  for (bool deepcopy : {false, true}) {
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjs_setDeepCopy(parent, deepcopy);
    mjsElement* y = mjs_findBody(child, "y")->element;
    mjsActuator* u =
        mjs_asActuator(mjs_findElement(child, mjOBJ_ACTUATOR, "u"));

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element,
                           mjs_findBody(child, "x")->element, "c_", ""),
                NotNull())
        << mjs_getError(parent);

    // the actuator of y stays in the child as it is
    EXPECT_STREQ(mjs_getString(mjs_getName(u->element)), "u");
    EXPECT_STREQ(mjs_getString(u->target), "y");
    EXPECT_EQ(mjs_getSpec(u->element), child);

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element, y, "d_", ""),
                NotNull())
        << mjs_getError(parent);
    model[deepcopy] = mj_compile(parent, nullptr);
    ASSERT_THAT(model[deepcopy], NotNull()) << mjs_getError(parent);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
  ASSERT_EQ(model[1]->nu, 2);
  std::string field;
  EXPECT_EQ(CompareModel(model[0], model[1], field), 0) << field;
  mj_deleteModel(model[0]);
  mj_deleteModel(model[1]);
}

// attaching a frame by reference leaves the rest of its body in the child
TEST_F(MujocoTest, AttachFrameByReferenceLeavesChild) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <joint name="x0"/>
        <geom name="gx" size=".1"/>
        <frame name="f">
          <joint name="jx" axis="1 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjModel* model[2];
  for (bool deepcopy : {false, true}) {
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjs_setDeepCopy(parent, deepcopy);
    mjsElement* x0 = mjs_findElement(child, mjOBJ_JOINT, "x0");
    mjsElement* gx = mjs_findElement(child, mjOBJ_GEOM, "gx");

    ASSERT_THAT(mjs_attach(mjs_findBody(parent, "A")->element,
                           mjs_findFrame(child, "f")->element, "c_", ""),
                NotNull())
        << mjs_getError(parent);
    EXPECT_STREQ(mjs_getString(mjs_getName(x0)), "x0");
    EXPECT_STREQ(mjs_getString(mjs_getName(gx)), "gx");
    EXPECT_EQ(mjs_getSpec(x0), child);
    EXPECT_EQ(mjs_getSpec(gx), child);

    model[deepcopy] = mj_compile(parent, nullptr);
    ASSERT_THAT(model[deepcopy], NotNull()) << mjs_getError(parent);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
  std::string field;
  EXPECT_EQ(CompareModel(model[0], model[1], field), 0) << field;
  mj_deleteModel(model[0]);
  mj_deleteModel(model[1]);
}

// a frame attached to a body is posed relative to that body, the frames which
// enclose it in the child are not attached
TEST_F(MujocoTest, AttachNestedFrameToBody) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <frame name="fy" pos="1 0 0">
          <frame name="fz" pos="0 1 0">
            <geom name="gz" size=".1"/>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  for (bool deepcopy : {false, true}) {
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjs_setDeepCopy(parent, deepcopy);
    mjsFrame* fy = mjs_findFrame(child, "fy");

    mjsElement* attached =
        mjs_attach(mjs_findBody(parent, "A")->element,
                   mjs_findFrame(child, "fz")->element, "c_", "");
    ASSERT_THAT(attached, NotNull()) << mjs_getError(parent);
    EXPECT_THAT(mjs_getFrame(attached), IsNull());

    // edit the enclosing frame of the child, delete it in a copy
    fy->pos[0] = 2;
    if (deepcopy) {
      EXPECT_EQ(mjs_delete(child, fy->element), 0);
    }

    mjModel* model = mj_compile(parent, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
    int gz = mj_name2id(model, mjOBJ_GEOM, "c_gz");
    ASSERT_GE(gz, 0);
    EXPECT_EQ(model->geom_pos[3 * gz + 0], 0);
    EXPECT_EQ(model->geom_pos[3 * gz + 1], 1);
    EXPECT_EQ(model->geom_pos[3 * gz + 2], 0);
    mj_deleteModel(model);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

// the elements of a spec attached by reference are in the parent, the spec
// cannot be attached again as a whole
TEST_F(MujocoTest, AttachSpecByReferenceAgain) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <frame name="f1"/>
      <frame name="f2"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element, child->element,
                         "r0/", ""),
              NotNull())
      << mjs_getError(parent);
  EXPECT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element, child->element,
                         "r1/", ""),
              IsNull());
  EXPECT_THAT(mjs_getError(parent), HasSubstr("already attached by reference"));

  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// attaching by reference an element which is in the parent, its own or one
// moved there by an earlier attachment, or a body which contains one, fails and
// leaves the parent as it was; a copy can be attached
TEST_F(MujocoTest, AttachByReferenceInParent) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint/>
        <geom size=".1"/>
      </body>
      <frame name="f1"/>
      <frame name="f2"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <joint/>
        <geom size=".1"/>
        <frame name="fx">
          <joint axis="1 0 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr const char* kError[] = {
      "in the parent spec", "already attached by reference",
      "in the parent spec", "in the parent spec", "in the parent spec"};

  std::array<char, 1000> er;
  for (int i = 0; i < 5; i++) {
    for (bool deepcopy : {false, true}) {
      mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
      ASSERT_THAT(parent, NotNull()) << er.data();
      mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
      ASSERT_THAT(child, NotNull()) << er.data();
      mjsElement* f1 = mjs_findFrame(parent, "f1")->element;
      mjsElement* f2 = mjs_findFrame(parent, "f2")->element;
      mjsElement* a = mjs_findBody(parent, "A")->element;
      mjsElement* x = mjs_findBody(child, "x")->element;
      mjsElement* fx = mjs_findFrame(child, "fx")->element;

      // parent and child of the first attachment, if any, and of the second
      mjsElement* attach[5][4] = {
          {f1, x, f2, x},               // the same body twice
          {a, fx, f2, x},               // a body after one of its frames
          {f1, child->element, f2, x},  // a body of a spec attached whole
          {nullptr, nullptr, f1, a},    // a body of the parent
          {nullptr, nullptr, a, f2},    // a frame of the parent
      };
      if (attach[i][0]) {
        ASSERT_THAT(mjs_attach(attach[i][0], attach[i][1], "c_", ""), NotNull())
            << mjs_getError(parent);
      }
      mjModel* model = mj_compile(parent, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(parent);

      mjs_setDeepCopy(parent, deepcopy);
      mjsElement* attached = mjs_attach(attach[i][2], attach[i][3], "d_", "");
      if (deepcopy) {
        EXPECT_THAT(attached, NotNull()) << mjs_getError(parent);
        mjModel* copied = mj_compile(parent, nullptr);
        EXPECT_THAT(copied, NotNull()) << mjs_getError(parent);
        mj_deleteModel(copied);
      } else {
        EXPECT_THAT(attached, IsNull()) << i;
        EXPECT_THAT(mjs_getError(parent), HasSubstr(kError[i]));

        // the parent has each body once, and compiles to the same model
        int nbody = 0;
        for (mjsElement* el = mjs_firstElement(parent, mjOBJ_BODY);
             el && nbody <= model->nbody; el = mjs_nextElement(parent, el)) {
          nbody++;
        }
        EXPECT_EQ(nbody, model->nbody) << i;
        mjModel* again = mj_compile(parent, nullptr);
        ASSERT_THAT(again, NotNull()) << mjs_getError(parent);
        std::string field;
        EXPECT_EQ(CompareModel(model, again, field), 0) << field;
        mj_deleteModel(again);
      }

      mj_deleteModel(model);
      mj_deleteSpec(child);
      mj_deleteSpec(parent);
    }
  }
}

TEST_F(MujocoTest, RecompileControlBlocks) {
  std::array<char, 1000> er;

  // 1. Deleting first actuator preserves retained actuator's control
  static constexpr char xml_delete[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j1" type="slide"/>
        <geom size="0.1"/>
      </body>
      <body>
        <joint name="j2" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="a1" joint="j1"/>
      <motor name="a2" joint="j2"/>
    </actuator>
  </mujoco>)";

  mjSpec* s1 = mj_parseXMLString(xml_delete, 0, er.data(), er.size());
  ASSERT_THAT(s1, NotNull()) << er.data();
  mjModel* m1 = mj_compile(s1, 0);
  ASSERT_THAT(m1, NotNull());
  mjData* d1 = mj_makeData(m1);
  d1->ctrl[0] = 11.0;
  d1->ctrl[1] = 22.0;

  mjsElement* a1 = mjs_findElement(s1, mjOBJ_ACTUATOR, "a1");
  ASSERT_THAT(a1, NotNull());
  EXPECT_EQ(mjs_delete(s1, a1), 0);
  EXPECT_EQ(mj_recompile(s1, 0, m1, d1), 0);
  EXPECT_EQ(m1->nu, 1);
  EXPECT_MJTNUM_EQ(d1->ctrl[0], 22.0);
  mj_deleteData(d1);
  mj_deleteModel(m1);
  mj_deleteSpec(s1);

  // 2. Multi-input PID actuator preserves all control slots
  static constexpr char xml_pid[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <pid name="pid" joint="j" kp="10" kv="1" input="pos vel ff"/>
    </actuator>
  </mujoco>)";

  mjSpec* s2 = mj_parseXMLString(xml_pid, 0, er.data(), er.size());
  ASSERT_THAT(s2, NotNull()) << er.data();
  mjModel* m2 = mj_compile(s2, 0);
  ASSERT_THAT(m2, NotNull());
  ASSERT_EQ(m2->nu, 3);
  mjData* d2 = mj_makeData(m2);
  d2->ctrl[0] = 1.0;
  d2->ctrl[1] = 2.0;
  d2->ctrl[2] = 5.0;

  EXPECT_EQ(mj_recompile(s2, 0, m2, d2), 0);
  ASSERT_EQ(m2->nu, 3);
  EXPECT_MJTNUM_EQ(d2->ctrl[0], 1.0);
  EXPECT_MJTNUM_EQ(d2->ctrl[1], 2.0);
  EXPECT_MJTNUM_EQ(d2->ctrl[2], 5.0);
  mj_deleteData(d2);
  mj_deleteModel(m2);
  mj_deleteSpec(s2);

  // 3. Zero-input actuator followed by scalar actuator (nactuator=2, nu=1)
  static constexpr char xml_zero[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j1" type="slide"/>
        <geom size="0.1"/>
      </body>
      <body>
        <joint name="j2" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor name="dc0" joint="j1" input="none" motorconst="1" resistance="1"/>
      <motor name="m1" joint="j2"/>
    </actuator>
  </mujoco>)";

  mjSpec* s3 = mj_parseXMLString(xml_zero, 0, er.data(), er.size());
  ASSERT_THAT(s3, NotNull()) << er.data();
  mjModel* m3 = mj_compile(s3, 0);
  ASSERT_THAT(m3, NotNull());
  ASSERT_EQ(m3->nactuator, 2);
  ASSERT_EQ(m3->nu, 1);
  mjData* d3 = mj_makeData(m3);
  d3->ctrl[0] = 33.0;

  EXPECT_EQ(mj_recompile(s3, 0, m3, d3), 0);
  ASSERT_EQ(m3->nactuator, 2);
  ASSERT_EQ(m3->nu, 1);
  EXPECT_MJTNUM_EQ(d3->ctrl[0], 33.0);
  mj_deleteData(d3);
  mj_deleteModel(m3);
  mj_deleteSpec(s3);

  // 4. Expanding actuator actdim from 1 to 3 does not read past saved act
  static constexpr char xml_actdim[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="hinge"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <general name="a" joint="j" dyntype="integrator"/>
    </actuator>
  </mujoco>)";

  mjSpec* s4 = mj_parseXMLString(xml_actdim, 0, er.data(), er.size());
  ASSERT_THAT(s4, NotNull()) << er.data();
  mjModel* m4 = mj_compile(s4, 0);
  ASSERT_THAT(m4, NotNull());
  ASSERT_EQ(m4->na, 1);
  mjData* d4 = mj_makeData(m4);
  d4->act[0] = 42.0;

  mjsActuator* act4 = mjs_asActuator(mjs_findElement(s4, mjOBJ_ACTUATOR, "a"));
  ASSERT_THAT(act4, NotNull());
  act4->dyntype = mjDYN_USER;
  act4->actdim = 3;

  EXPECT_EQ(mj_recompile(s4, 0, m4, d4), 0);
  ASSERT_EQ(m4->na, 3);
  EXPECT_MJTNUM_EQ(d4->act[0], 42.0);
  EXPECT_MJTNUM_EQ(d4->act[1], 0.0);
  EXPECT_MJTNUM_EQ(d4->act[2], 0.0);
  mj_deleteData(d4);
  mj_deleteModel(m4);
  mj_deleteSpec(s4);
}

TEST_F(MujocoTest, RecompileIntegrationState) {
  std::array<char, 1000> er;
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuserdata="2"/>
    <worldbody>
      <body name="b1">
        <joint name="j1" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body name="b2">
        <joint name="j2" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <equality>
      <joint name="eq1" joint1="j1" active="false"/>
      <joint name="eq2" joint1="j2" active="false"/>
    </equality>
    <actuator>
      <motor name="a1" joint="j1" nsample="2" delay="0.01"/>
    </actuator>
    <sensor>
      <jointpos name="s1" joint="j2" nsample="2" delay="0.01"/>
    </sensor>
  </mujoco>)";

  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m = mj_compile(spec, 0);
  ASSERT_THAT(m, NotNull());
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());

  d->time = 3.25;
  d->qpos[0] = 0.4;
  d->qpos[1] = -0.2;
  d->qvel[0] = 0.3;
  d->qvel[1] = -0.1;
  d->qfrc_applied[0] = 7.0;
  d->qfrc_applied[1] = 9.0;
  d->xfrc_applied[6 * 1 + 0] = 8.0;
  d->xfrc_applied[6 * 2 + 2] = 12.0;
  d->eq_active[0] = 1;
  d->eq_active[1] = 1;
  d->userdata[0] = 123.0;
  d->userdata[1] = 321.0;
  d->qacc_warmstart[0] = 456.0;
  d->qacc_warmstart[1] = 654.0;
  d->ctrl[0] = 4.5;
  for (int i = 0; i < m->nhistory; i++) {
    d->history[i] = 10.0 + i;
  }

  int nstate = mj_stateSize(m, mjSTATE_INTEGRATION);
  std::vector<mjtNum> state_before(nstate);
  mj_getState(m, d, state_before.data(), mjSTATE_INTEGRATION);

  // no-op recompile preserves the entire mjSTATE_INTEGRATION
  EXPECT_EQ(mj_recompile(spec, 0, m, d), 0);
  std::vector<mjtNum> state_after(nstate);
  mj_getState(m, d, state_after.data(), mjSTATE_INTEGRATION);
  EXPECT_EQ(state_before, state_after);

  // delete b1 (along with j1, eq1, a1) and recompile: b2, j2, eq2, s1 state
  // preserved
  mjsBody* b1 = mjs_findBody(spec, "b1");
  ASSERT_THAT(b1, NotNull());
  EXPECT_EQ(mjs_delete(spec, b1->element), 0);
  EXPECT_EQ(mj_recompile(spec, 0, m, d), 0);
  EXPECT_EQ(m->nq, 1);
  EXPECT_EQ(m->nv, 1);
  EXPECT_EQ(m->nbody, 2);
  EXPECT_EQ(m->neq, 1);
  EXPECT_MJTNUM_EQ(d->qpos[0], -0.2);
  EXPECT_MJTNUM_EQ(d->qvel[0], -0.1);
  EXPECT_MJTNUM_EQ(d->qfrc_applied[0], 9.0);
  EXPECT_MJTNUM_EQ(d->qacc_warmstart[0], 654.0);
  EXPECT_MJTNUM_EQ(d->xfrc_applied[6 * 1 + 2], 12.0);
  EXPECT_EQ(d->eq_active[0], 1);
  EXPECT_MJTNUM_EQ(d->userdata[0], 123.0);
  EXPECT_MJTNUM_EQ(d->userdata[1], 321.0);
  ASSERT_EQ(m->nhistory, 6);
  for (int i = 0; i < 6; i++) {
    EXPECT_MJTNUM_EQ(d->history[i], 16.0 + i);
  }

  // add a new body/joint/actuator/equality and mutate j2 from slide to ball
  mjsBody* b3 = mjs_addBody(mjs_findBody(spec, "world"), 0);
  mjsGeom* g3 = mjs_addGeom(b3, 0);
  g3->size[0] = 0.1;
  g3->mass = 1.0;
  mjsJoint* j3 = mjs_addJoint(b3, 0);
  mjs_setName(j3->element, "j3");
  j3->type = mjJNT_SLIDE;
  j3->ref = 0.75;
  mjsActuator* a3 = mjs_addActuator(spec, 0);
  a3->trntype = mjTRN_JOINT;
  mjs_setString(a3->target, "j3");
  mjsEquality* eq3 = mjs_addEquality(spec, 0);
  eq3->type = mjEQ_JOINT;
  eq3->active = 1;
  mjs_setString(eq3->name1, "j3");

  // delete eq2 and s1 before mutating j2 to ball joint
  EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_EQUALITY, "eq2")), 0);
  EXPECT_EQ(mjs_delete(spec, mjs_findElement(spec, mjOBJ_SENSOR, "s1")), 0);
  mjsJoint* j2 = mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "j2"));
  ASSERT_THAT(j2, NotNull());
  j2->type = mjJNT_BALL;

  EXPECT_EQ(mj_recompile(spec, 0, m, d), 0);
  ASSERT_EQ(m->nq, 5);
  ASSERT_EQ(m->nv, 4);
  // j2 changed type from slide to ball: initialized to unit quaternion from
  // qpos0 and zero vel
  EXPECT_MJTNUM_EQ(d->qpos[0], 1.0);
  EXPECT_MJTNUM_EQ(d->qpos[1], 0.0);
  EXPECT_MJTNUM_EQ(d->qpos[2], 0.0);
  EXPECT_MJTNUM_EQ(d->qpos[3], 0.0);
  EXPECT_MJTNUM_EQ(d->qvel[0], 0.0);
  EXPECT_MJTNUM_EQ(d->qfrc_applied[0], 0.0);
  EXPECT_MJTNUM_EQ(d->qacc_warmstart[0], 0.0);
  // newly added j3, a3, eq3 receive defaults while b2 retains xfrc_applied
  EXPECT_MJTNUM_EQ(d->qpos[4], 0.75);
  EXPECT_MJTNUM_EQ(d->qvel[3], 0.0);
  EXPECT_MJTNUM_EQ(d->ctrl[0], 0.0);
  EXPECT_EQ(d->eq_active[0], 1);
  EXPECT_MJTNUM_EQ(d->xfrc_applied[6 * 1 + 2], 12.0);
  EXPECT_MJTNUM_EQ(d->xfrc_applied[6 * 2 + 2], 0.0);

  mj_deleteData(d);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, AttachMocap) {
  std::array<char, 1000> er;
  mjtNum tol = 0;
  std::string field = "";

  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 1 1" quat="0 1 0 0" name="mocap" mocap="true"/>
    </worldbody>
    <keyframe>
      <key name="key" time="1" mpos="2 2 2" mquat="0 0 0 1"/>
    </keyframe>
  </mujoco>)";

  static constexpr char xml_expected[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 1 1" quat="0 1 0 0" name="mocap" mocap="true"/>
      <body pos="1 1 1" quat="0 1 0 0" name="attached-mocap-1" mocap="true"/>
    </worldbody>
    <keyframe>
      <key name="key" time="1" mpos="2 2 2 1 1 1" mquat="0 0 0 1 0 1 0 0"/>
      <key name="attached-key-1" time="1" mpos="1 1 1 2 2 2" mquat="0 1 0 0 0 0 0 1"/>
    </keyframe>
  </mujoco>)";

  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();
  mjs_setDeepCopy(spec, true);  // needed for self-attach

  mjsBody* body = mjs_findBody(spec, "mocap");
  EXPECT_THAT(body, NotNull());

  mjsBody* world = mjs_findBody(spec, "world");
  EXPECT_THAT(world, NotNull());

  mjsElement* frame = mjs_addFrame(world, NULL)->element;
  mjs_attach(frame, body->element, "attached-", "-1");

  mjsBody* attached_body = mjs_findBody(spec, "attached-mocap-1");
  EXPECT_THAT(attached_body, NotNull());

  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull());

  MjModelPtr m_expected =
      LoadModelFromString(xml_expected, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, ReplicateKeyframe) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="1" euler="0 0 1.8">
        <body name="body" pos="0 -1 0">
          <joint type="slide"/>
          <geom name="g" size="1"/>
        </body>
      </replicate>
    </worldbody>

    <keyframe>
      <key name="keyframe" qpos="1"/>
    </keyframe>
  </mujoco>

  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), testing::NotNull()) << error.data();
  EXPECT_THAT(m->ngeom, 1);
  EXPECT_THAT(m->nbody, 2);

  // check that the keyframe is not replicated
  EXPECT_THAT(m->nkey, 1);
  EXPECT_THAT(m->nq, 1);
  EXPECT_THAT(m->key_qpos[0], 1);
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_KEY, 0), "keyframe");
}

TEST_F(MujocoTest, AttachUnnamedAssets) {
  static constexpr char cube[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";

  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh file="cube.obj"/>
    </asset>
    <worldbody>
      <frame name="frame">
        <geom type="mesh" mesh="cube"/>
      </frame>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  // the parser has named the mesh after its file
  std::array<char, 1000> er;
  mjSpec* child = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();
  mjsElement* mesh = mjs_firstElement(child, mjOBJ_MESH);
  EXPECT_STREQ(mjs_getString(mjs_getName(mesh)), "cube");

  // the name takes the prefix of the attachment
  mjSpec* spec = mj_makeSpec();
  mjs_attach(mjs_findBody(spec, "world")->element,
             mjs_findFrame(child, "frame")->element, "_", "");

  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(model->nmesh, 1);
  EXPECT_STREQ(mj_id2name(model, mjOBJ_MESH, 0), "_cube");

  mj_deleteVFS(vfs.get());
  mj_deleteSpec(spec);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

// an asset created through the API is not named after its file
TEST_F(MujocoTest, ApiAssetNeedsName) {
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
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  mjSpec* spec = mj_makeSpec();
  mjsMesh* mesh = mjs_addMesh(spec, nullptr);
  mjs_setString(mesh->file, "cube.obj");
  mjsGeom* geom = mjs_addGeom(mjs_findBody(spec, "world"), nullptr);
  geom->type = mjGEOM_MESH;
  mjs_setString(geom->meshname, "cube");

  // the mesh has no name, and the error says what to do
  EXPECT_THAT(mjs_findElement(spec, mjOBJ_MESH, "cube"), IsNull());
  EXPECT_THAT(mj_compile(spec, vfs.get()), IsNull());
  EXPECT_THAT(mjs_getError(spec),
              HasSubstr("mesh with file 'cube.obj' has no name"));
  EXPECT_THAT(mjs_getString(mjs_getName(mesh->element)), IsEmpty());

  // with a name it compiles
  mjs_setName(mesh->element, "cube");
  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_STREQ(mj_id2name(model, mjOBJ_MESH, 0), "cube");

  mj_deleteVFS(vfs.get());
  mj_deleteSpec(spec);
  mj_deleteModel(model);
}

TEST_F(MujocoTest, AttachMaterialReferences) {
  static constexpr char xml_child[] = R"(
  <mujoco>
    <asset>
      <material name="material" rgba="1 0 0 1"/>
      <mesh name="mesh" material="material" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
    </asset>
    <worldbody>
      <body name="body">
        <joint/>
        <geom type="mesh" mesh="mesh"/>
        <site name="site0"/>
        <site name="site1" pos="1 0 0"/>
      </body>
    </worldbody>
    <tendon>
      <spatial material="material">
        <site site="site0"/>
        <site site="site1"/>
      </spatial>
    </tendon>
    <deformable>
      <skin name="skin" material="material" vertex="0 0 0 1 0 0 0 1 0" face="0 1 2">
        <bone body="body" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2" vertweight="1 1 1"/>
      </skin>
    </deformable>
  </mujoco>)";

  // attach two children whose skin, tendon and mesh reference a material, by
  // deep copy and by reference
  std::array<char, 1000> er;
  std::array<mjModel*, 2> models;
  for (bool deepcopy : {true, false}) {
    mjSpec* parent = mj_makeSpec();
    mjs_setDeepCopy(parent, deepcopy);
    std::array<mjSpec*, 2> children;
    for (int i = 0; i < 2; i++) {
      children[i] = mj_parseXMLString(xml_child, 0, er.data(), er.size());
      ASSERT_THAT(children[i], NotNull()) << er.data();
      mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), 0);
      mjsBody* body = mjs_findBody(children[i], "body");
      mjsElement* attached =
          mjs_attach(frame->element, body->element, i ? "b-" : "a-", "");
      EXPECT_THAT(attached, NotNull()) << mjs_getError(parent);
    }

    // skin names and material references are namespaced
    mjModel* model = mj_compile(parent, 0);
    ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
    ASSERT_THAT(model->nskin, 2);
    ASSERT_THAT(model->ntendon, 2);
    ASSERT_THAT(model->ngeom, 2);
    EXPECT_STREQ(mj_id2name(model, mjOBJ_SKIN, 0), "a-skin");
    EXPECT_STREQ(mj_id2name(model, mjOBJ_SKIN, 1), "b-skin");
    for (int i = 0; i < 2; i++) {
      const char* material = i ? "b-material" : "a-material";
      EXPECT_STREQ(mj_id2name(model, mjOBJ_MATERIAL, model->skin_matid[i]),
                   material);
      EXPECT_STREQ(mj_id2name(model, mjOBJ_MATERIAL, model->tendon_matid[i]),
                   material);
      EXPECT_STREQ(mj_id2name(model, mjOBJ_MATERIAL, model->geom_matid[i]),
                   material);
    }
    models[deepcopy] = model;

    mj_deleteSpec(parent);
    for (mjSpec* child : children) {
      mj_deleteSpec(child);
    }
  }

  // deep copy and attachment by reference give the same model
  std::string field = "";
  EXPECT_LE(CompareModel(models[0], models[1], field), 0)
      << "Different field: " << field;

  mj_deleteModel(models[0]);
  mj_deleteModel(models[1]);
}

TEST_F(MujocoTest, AttachSkinFromFile) {
  static constexpr char xml_child[] = R"(
  <mujoco>
    <asset>
      <skin file="triangle_bone.skn"/>
    </asset>
    <worldbody>
      <body name="box">
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>)";

  // a triangle with one bone on body "box"
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  ASSERT_EQ(
      mj_addFileVFS(vfs.get(), GetTestDataFilePath("user/testdata").c_str(),
                    "triangle_bone.skn"),
      0);

  // the bones loaded from file take the prefixes and suffixes of two nested
  // attachments, by deep copy and by reference
  std::array<char, 1000> er;
  std::array<mjModel*, 2> models;
  for (bool deepcopy : {true, false}) {
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjSpec* middle = mj_makeSpec();
    mjSpec* parent = mj_makeSpec();
    mjs_setDeepCopy(middle, deepcopy);
    mjs_setDeepCopy(parent, deepcopy);
    mjsBody* body = mjs_addBody(mjs_findBody(middle, "world"), 0);
    mjs_setName(body->element, "middle");
    mjsFrame* frame = mjs_addFrame(body, 0);
    EXPECT_THAT(mjs_attach(frame->element, mjs_findBody(child, "box")->element,
                           "a_", "_x"),
                NotNull())
        << mjs_getError(middle);
    frame = mjs_addFrame(mjs_findBody(parent, "world"), 0);
    EXPECT_THAT(mjs_attach(frame->element,
                           mjs_findBody(middle, "middle")->element, "b_", "_y"),
                NotNull())
        << mjs_getError(parent);

    mjModel* model = mj_compile(parent, vfs.get());
    ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
    ASSERT_EQ(model->nskin, 1);
    for (int i = 0; i < model->skin_bonenum[0]; i++) {
      EXPECT_STREQ(mj_id2name(model, mjOBJ_BODY, model->skin_bonebodyid[i]),
                   "b_a_box_x_y");
    }

    // the saved model has the bones as data, and loads without the file
    std::string saved = SaveAndReadXml(parent);
    EXPECT_EQ(saved.find("triangle_bone.skn"), std::string::npos);
    MjModelPtr reloaded = LoadModelFromString(saved, er.data(), er.size());
    ASSERT_THAT(reloaded, NotNull()) << er.data();
    ASSERT_EQ(reloaded->nskin, 1);
    ASSERT_EQ(reloaded->nskinbone, model->nskinbone);
    ASSERT_EQ(reloaded->nskinvert, model->nskinvert);
    for (int i = 0; i < model->nskinbone; i++) {
      EXPECT_STREQ(
          mj_id2name(reloaded.get(), mjOBJ_BODY, reloaded->skin_bonebodyid[i]),
          "b_a_box_x_y");
    }
    for (int i = 0; i < 3 * model->nskinvert; i++) {
      EXPECT_EQ(reloaded->skin_vert[i], model->skin_vert[i]);
    }
    for (int i = 0; i < model->nskinbonevert; i++) {
      EXPECT_EQ(reloaded->skin_bonevertid[i], model->skin_bonevertid[i]);
      EXPECT_EQ(reloaded->skin_bonevertweight[i],
                model->skin_bonevertweight[i]);
    }

    models[deepcopy] = model;
    mj_deleteSpec(parent);
    mj_deleteSpec(middle);
    mj_deleteSpec(child);
  }

  // deep copy and attachment by reference give the same model
  std::string field = "";
  EXPECT_LE(CompareModel(models[0], models[1], field), 0)
      << "Different field: " << field;

  mj_deleteModel(models[0]);
  mj_deleteModel(models[1]);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, InitTexture) {
  mjSpec* spec = mj_makeSpec();
  EXPECT_THAT(spec, NotNull());

  mjsTexture* texture = mjs_addTexture(spec, nullptr);
  mjs_setName(texture->element, "checker");
  texture->type = mjTEXTURE_CUBE;
  texture->builtin = mjBUILTIN_CHECKER;
  texture->width = 300;
  texture->height = 300;

  mjsMaterial* material = mjs_addMaterial(spec, 0);
  mjs_setName(material->element, "floor");
  mjs_setInStringVec(material->textures, mjTEXROLE_RGB, "checker");

  mjsGeom* floor = mjs_addGeom(mjs_findBody(spec, "world"), 0);
  mjs_setString(floor->material, "floor");
  floor->type = mjGEOM_PLANE;
  floor->size[0] = 1;
  floor->size[1] = 1;
  floor->size[2] = 0.01;
  mjs_setString(floor->material, "floor");

  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull());

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

void AttachNestedKeyframe(bool compile) {
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="frame"/>
    </worldbody>
  </mujoco>)";

  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size=".2"/>
        <frame name="frame"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key2" time="2" qpos="2"/>
    </keyframe>
  </mujoco>)";

  static constexpr char gchild_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size=".3"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key1" time="1" qpos="1"/>
    </keyframe>
  </mujoco>)";

  static constexpr char expected_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="frame">
        <body name="child-body">
          <joint name="child-joint" type="slide"/>
          <geom size=".2"/>
          <frame name="child-frame">
            <body name="child-gchild-body">
              <joint name="child-gchild-joint" type="slide"/>
              <geom size=".3"/>
            </body>
          </frame>
        </body>
      </frame>
    </worldbody>
    <keyframe>
      <key name="child-key2" time="2" qpos="0 2 0"/>
      <key name="child-gchild-key1" time="1" qpos="0 0 1"/>
    </keyframe>
  </mujoco>)";

  static constexpr char expected_xml_uncompiled[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="frame">
        <body name="child-body">
          <joint name="child-joint" type="slide"/>
          <geom size=".2"/>
          <frame name="child-frame">
            <body name="child-gchild-body">
              <joint name="child-gchild-joint" type="slide"/>
              <geom size=".3"/>
            </body>
          </frame>
        </body>
      </frame>
    </worldbody>
    <keyframe>
      <key name="child-key2" time="2" qpos="0 2 0"/>
      <key name="gchild-key1" time="1" qpos="0 0 1"/>
    </keyframe>
  </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(parent_xml, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(child_xml, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();
  mjSpec* gchild = mj_parseXMLString(gchild_xml, 0, er.data(), er.size());
  EXPECT_THAT(gchild, NotNull()) << er.data();

  mjs_setDeepCopy(parent, true);
  mjs_setDeepCopy(child, true);

  // attach gchild to child
  mjs_attach(mjs_findFrame(child, "frame")->element,
             mjs_findBody(gchild, "body")->element, "gchild-", "");

  // compile required before further attachment
  mjModel* m_child = compile ? mj_compile(child, 0) : nullptr;

  // check warning is issued, empty for a compiled model
  MockWarningHandler warning_handler;
  if (!compile) {
    warning_handler.ExpectWarnings("model has pending keyframes");
  }

  // attach child to parent
  mjs_attach(mjs_findFrame(parent, "frame")->element,
             mjs_findBody(child, "body")->element, "child-", "");

  if (compile) {
    EXPECT_EQ(mjs_numWarnings(parent), 0);
  } else {
    EXPECT_GE(mjs_numWarnings(parent), 1);
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("model has pending keyframes"));
  }
  // compare models
  mjtNum tol = 0;
  std::string field = "";
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull());
  MjModelPtr m_expected = LoadModelFromString(
      compile ? expected_xml : expected_xml_uncompiled, er.data(), er.size());
  EXPECT_THAT(m_expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(m_attached, m_expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';
  ;

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteSpec(gchild);
  mj_deleteModel(m_attached);
  mj_deleteModel(m_child);
}

TEST_F(MujocoTest, TestAttachNestedKeyframe) {
  mock_warning_handler.ExpectWarnings();
  AttachNestedKeyframe(/*compile=*/true);
  AttachNestedKeyframe(/*compile=*/false);
}

TEST_F(MujocoTest, RepeatedAttachKeyframe) {
  static constexpr char xml_1[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <body name="body"/>
      </worldbody>
    </mujoco>)";

  static constexpr char xml_2[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <body name="b1">
          <joint/>
          <geom size="0.1"/>
        </body>
        <body name="b2">
          <joint/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="home" qpos="1 2" />
      </keyframe>
    </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_1, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_2, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  mjsBody* body_1 = mjs_findBody(parent, "body");
  mjsElement* attachment_frame = mjs_addFrame(body_1, 0)->element;
  mjs_attach(attachment_frame, mjs_findBody(child, "b1")->element, "b1-", "");
  mjModel* model_1 = mj_compile(parent, 0);
  EXPECT_THAT(model_1, NotNull());
  mjs_attach(attachment_frame, mjs_findBody(child, "b2")->element, "b2-", "");
  mjModel* model_2 = mj_compile(parent, 0);
  EXPECT_THAT(model_2, NotNull());

  EXPECT_EQ(model_1->nkey, 1);
  EXPECT_EQ(model_2->nkey, 2);
  EXPECT_STREQ(mj_id2name(model_2, mjOBJ_KEY, 0), "b1-home");
  EXPECT_STREQ(mj_id2name(model_2, mjOBJ_KEY, 1), "b2-home");

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model_1);
  mj_deleteModel(model_2);
}

// a keyframe stays in the spec when the tree changes; the next compilation
// only completes its vectors
TEST_F(MujocoTest, KeyframesSurviveTreeChanges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint name="a"/>
        <geom size=".1"/>
      </body>
      <body name="b">
        <joint name="b"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="b" joint="b"/>
    </actuator>
    <keyframe>
      <key name="home" qpos="1 2" ctrl="3"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="c">
        <joint name="c"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="pose" qpos="4"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();
  mjsElement* home = mjs_findElement(spec, mjOBJ_KEY, "home");
  ASSERT_THAT(home, NotNull());

  // deleting a body keeps the keyframe, which is completed when compiling
  EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "a")->element), 0);
  EXPECT_EQ(mjs_findElement(spec, mjOBJ_KEY, "home"), home);
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(mjs_findElement(spec, mjOBJ_KEY, "home"), home);
  ASSERT_EQ(m1->nkey, 1);
  EXPECT_EQ(m1->key_qpos[0], 2);
  EXPECT_EQ(m1->key_ctrl[0], 3);

  // the keyframe of an attached model is in the parent before compiling
  mjsFrame* frame = mjs_addFrame(mjs_findBody(spec, "world"), nullptr);
  mjs_attach(frame->element, mjs_findBody(child, "c")->element, "child-", "");
  mjsElement* pose = mjs_findElement(spec, mjOBJ_KEY, "child-pose");
  ASSERT_THAT(pose, NotNull());
  EXPECT_EQ(mjs_nextElement(spec, home), pose);

  // it can be renamed, and a vector which is set meanwhile is kept
  mjs_setName(pose, "posture");
  std::vector<double> qvel = {5, 6};
  mjs_setDouble(mjs_asKey(home)->qvel, qvel.data(), qvel.size());

  // a copy made meanwhile compiles to the same model
  mjSpec* copy = mj_copySpec(spec);
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  mjModel* m2_copy = mj_compile(copy, nullptr);
  ASSERT_THAT(m2_copy, NotNull()) << mjs_getError(copy);
  std::string field;
  EXPECT_EQ(CompareModel(m2, m2_copy, field), 0) << field;

  ASSERT_EQ(m2->nkey, 2);
  ASSERT_EQ(m2->nq, 2);
  EXPECT_EQ(mj_name2id(m2, mjOBJ_KEY, "home"), 0);
  EXPECT_EQ(mj_name2id(m2, mjOBJ_KEY, "posture"), 1);
  EXPECT_THAT(AsVector(m2->key_qpos, 4), ElementsAreArray({2, 0, 0, 4}));
  EXPECT_THAT(AsVector(m2->key_qvel, 4), ElementsAreArray({5, 6, 0, 0}));
  EXPECT_THAT(AsVector(m2->key_ctrl, 2), ElementsAreArray({3, 0}));

  // the spec has the completed vectors
  EXPECT_THAT(*mjs_asKey(pose)->qpos, ElementsAreArray({0, 4}));

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteModel(m2_copy);
  mj_deleteSpec(copy);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
}

// a pending keyframe can be found, edited and renamed before it is compiled
TEST_F(MujocoTest, PendingKeyframeEdits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint/>
        <geom size=".1"/>
      </body>
      <body name="b">
        <joint/>
        <geom size=".1"/>
      </body>
      <body name="c">
        <joint/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="home" qpos="1 2 3" qvel="4 5 6"/>
    </keyframe>
  </mujoco>
  )";

  // the same holds for a spec which was compiled before: a vector given to a
  // keyframe after the tree changed is laid out for the tree as it is then
  for (bool compile_first : {false, true}) {
    SCOPED_TRACE(compile_first ? "compiled first" : "not compiled first");
    std::array<char, 1000> er;
    mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
    ASSERT_THAT(spec, NotNull()) << er.data();
    mjModel* model = nullptr;
    mjData* data = nullptr;
    if (compile_first) {
      model = mj_compile(spec, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
      data = mj_makeData(model);
      data->qpos[0] = 0.25;
      data->qpos[1] = 0.5;
      data->qpos[2] = 0.75;
    }
    mjsElement* home = mjs_findElement(spec, mjOBJ_KEY, "home");
    ASSERT_THAT(home, NotNull());

    // the keyframe is pending after a deletion; one which is added then is
    // placed before it, and each is found by its name
    EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "a")->element), 0);
    mjsKey* added = mjs_addKey(spec);
    mjs_setName(added->element, "added");
    EXPECT_EQ(mjs_findElement(spec, mjOBJ_KEY, "home"), home);
    EXPECT_EQ(mjs_findElement(spec, mjOBJ_KEY, "added"), added->element);

    // a vector given to the pending keyframe replaces what was stored of it
    const std::vector<double> qpos = {20, 30};
    mjs_setDouble(mjs_asKey(home)->qpos, qpos.data(), qpos.size());

    // it is renamed, and its name is given to another keyframe
    EXPECT_EQ(mjs_setName(home, "old"), 0);
    mjsKey* reused = mjs_addKey(spec);
    EXPECT_EQ(mjs_setName(reused->element, "home"), 0);
    const std::vector<double> qpos_reused = {200, 300};
    mjs_setDouble(reused->qpos, qpos_reused.data(), qpos_reused.size());

    // after a second deletion, every keyframe has its own values
    EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "b")->element), 0)
        << mjs_getError(spec);
    if (compile_first) {
      // recompiling carries the state over: the joint which is left has the
      // address which the first compilation gave it
      ASSERT_EQ(mj_recompile(spec, nullptr, model, data), 0)
          << mjs_getError(spec);
      EXPECT_EQ(data->qpos[0], 0.75);
    } else {
      model = mj_compile(spec, nullptr);
      ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    }
    ASSERT_EQ(model->nq, 1);
    ASSERT_EQ(model->nkey, 3);
    int old_id = mj_name2id(model, mjOBJ_KEY, "old");
    int home_id = mj_name2id(model, mjOBJ_KEY, "home");
    int added_id = mj_name2id(model, mjOBJ_KEY, "added");
    EXPECT_EQ(model->key_qpos[old_id], 30);
    EXPECT_EQ(model->key_qvel[old_id], 6);
    EXPECT_EQ(model->key_qpos[home_id], 300);
    EXPECT_EQ(model->key_qvel[home_id], 0);
    EXPECT_EQ(model->key_qpos[added_id], 0);

    mj_deleteData(data);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

// a keyframe which awaits compilation keeps its values when the spec is
// copied or one of its bodies is deleted
TEST_F(MujocoTest, PendingKeyframeSurvivesCopy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="frame"/>
      <body name="other">
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="c">
        <joint name="c"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="c" dyntype="filter" dynprm="1"/>
    </actuator>
    <keyframe>
      <key name="pose" qpos="1" act="2" ctrl="3"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  // the keyframe of the attached model awaits compilation
  mjs_attach(mjs_findFrame(spec, "frame")->element,
             mjs_findBody(child, "c")->element, "child-", "");

  mjSpec* copy = mj_copySpec(spec);
  EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "other")->element), 0);

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nkey, 1);
  EXPECT_EQ(m->key_qpos[0], 1);
  EXPECT_EQ(m->key_act[0], 2);
  EXPECT_EQ(m->key_ctrl[0], 3);

  mj_deleteModel(m);
  mj_deleteSpec(copy);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
}

// a copy of a compiled spec keeps the values of its keyframes when a body is
// deleted from it, or when it is attached
TEST_F(MujocoTest, KeyframeOfCopySurvivesTreeChanges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="b">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key" qpos="1 2"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  mjSpec* copy = mj_copySpec(spec);
  EXPECT_EQ(mjs_delete(copy, mjs_findBody(copy, "a")->element), 0);
  mjModel* m_copy = mj_compile(copy, nullptr);
  ASSERT_THAT(m_copy, NotNull()) << mjs_getError(copy);
  ASSERT_EQ(m_copy->nq, 1);
  EXPECT_EQ(m_copy->key_qpos[0], 2);

  mjSpec* child = mj_copySpec(spec);
  mjSpec* parent = mj_makeSpec();
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);
  ASSERT_THAT(mjs_attach(frame->element, mjs_findBody(child, "b")->element,
                         "child-", ""),
              NotNull())
      << mjs_getError(parent);
  mjModel* m_parent = mj_compile(parent, nullptr);
  ASSERT_THAT(m_parent, NotNull()) << mjs_getError(parent);
  ASSERT_EQ(m_parent->nq, 1);
  EXPECT_EQ(m_parent->key_qpos[0], 2);

  mj_deleteModel(m);
  mj_deleteModel(m_copy);
  mj_deleteModel(m_parent);
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// a copy of a compiled spec is saved as the original is, also once the original
// is deleted
TEST_F(MujocoTest, SaveCopyOfCompiledSpec) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetrahedron" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <body>
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="mocap" mocap="true" pos="1 0 0"/>
      <body pos="0 1 0">
        <freejoint/>
        <geom name="tetrahedron" type="mesh" mesh="tetrahedron"/>
      </body>
    </worldbody>
    <sensor>
      <contact geom1="tetrahedron"/>
    </sensor>
    <keyframe>
      <key qpos="1 0 1 0 1 0 0 0" mpos="2 0 0"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);
  ASSERT_FALSE(saved.empty());

  mjSpec* copy = mj_copySpec(spec);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
  EXPECT_EQ(SaveAndReadXml(copy), saved);

  mj_deleteSpec(copy);
}

// recompiling a copy of a compiled spec with the model and data of the original
// keeps the state, as recompiling the original does
TEST_F(MujocoTest, RecompileCopyOfCompiledSpec) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuserdata="1"/>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="mocap" mocap="true"/>
    </worldbody>
    <equality>
      <joint joint1="a" active="false"/>
    </equality>
    <actuator>
      <general joint="b" dyntype="filter" dynprm="1" nsample="2" delay=".01"/>
    </actuator>
    <sensor>
      <jointpos joint="a" nsample="2" delay=".01"/>
    </sensor>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjData* data = mj_makeData(model);

  data->time = 3;
  data->qpos[0] = 5;
  data->qpos[1] = 7;
  data->qvel[1] = 2;
  data->act[0] = 4;
  data->ctrl[0] = 6;
  data->mocap_pos[0] = 8;
  data->eq_active[0] = 1;
  data->userdata[0] = 9;
  data->qfrc_applied[1] = 10;
  data->xfrc_applied[6 * 2 + 2] = 11;
  for (int i = 0; i < model->nhistory; i++) {
    data->history[i] = 12 + i;
  }
  int nstate = mj_stateSize(model, mjSTATE_INTEGRATION);
  std::vector<mjtNum> state(nstate);
  mj_getState(model, data, state.data(), mjSTATE_INTEGRATION);

  mjSpec* copy = mj_copySpec(spec);
  mjsGeom* geom = mjs_addGeom(mjs_findBody(copy, "A"), nullptr);
  geom->size[0] = 0.1;
  EXPECT_EQ(mj_recompile(copy, nullptr, model, data), 0) << mjs_getError(copy);
  ASSERT_EQ(model->ngeom, 3);
  std::vector<mjtNum> recompiled(nstate);
  mj_getState(model, data, recompiled.data(), mjSTATE_INTEGRATION);
  EXPECT_EQ(recompiled, state);

  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// a copy of a compiled spec numbers its pairs and excludes as the compiled
// model does, rather than in the order in which they were written
TEST_F(MujocoTest, CopyOfCompiledSpecNumbersPairs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="1">
        <freejoint/>
        <geom name="1" size=".1"/>
      </body>
      <body name="2">
        <freejoint/>
        <geom name="2" size=".1"/>
      </body>
      <body name="3">
        <freejoint/>
        <geom name="3" size=".1"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="2" geom2="3"/>
      <pair geom1="1" geom2="2"/>
      <exclude body1="2" body2="3"/>
      <exclude body1="1" body2="2"/>
    </contact>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  mjSpec* copy = mj_copySpec(spec);
  for (mjtObj type : {mjOBJ_PAIR, mjOBJ_EXCLUDE}) {
    mjsElement* element = mjs_firstElement(spec, type);
    mjsElement* copied = mjs_firstElement(copy, type);
    EXPECT_EQ(mjs_getId(element), 1);
    while (element && copied) {
      EXPECT_EQ(mjs_getId(copied), mjs_getId(element));
      element = mjs_nextElement(spec, element);
      copied = mjs_nextElement(copy, copied);
    }
  }

  mj_deleteModel(model);
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// a copy of a compiled spec holds what was authored in the original and is
// saved as it: a contact pair and an exclude name their geoms and bodies in the
// order in which they were written, which compilation swaps, and a tendon which
// wraps a cylinder keeps what the compilation gave it
TEST_F(MujocoTest, CopyOfCompiledSpecKeepsPairsAndWraps) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_tendon="3"/>
    <worldbody>
      <geom name="floor" size=".2" contype="0" conaffinity="0"/>
      <geom name="post" type="cylinder" size=".1 .5" pos=".5 0 .5"/>
      <site name="start" pos="0 0 1"/>
      <body name="first" pos="0 0 1">
        <freejoint/>
        <geom name="ball" size=".1"/>
        <site name="end"/>
      </body>
      <body name="second" pos="1 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="ball" geom2="floor"/>
      <exclude body1="second" body2="first"/>
    </contact>
    <tendon>
      <spatial name="rope" user="1">
        <site site="start"/>
        <geom geom="post"/>
        <site site="end"/>
      </spatial>
    </tendon>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  mjSpec* copy = mj_copySpec(spec);
  EXPECT_THAT(CompareSpec(spec, copy), IsEmpty());
  EXPECT_EQ(SaveAndReadXml(copy), SaveAndReadXml(spec));

  mj_deleteModel(model);
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

// a joint whose type was changed after compiling has no value in the keyframes
// which are stored when the tree changes: they take its default configuration
TEST_F(MujocoTest, KeyframeSkipsJointOfChangedType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint name="a"/>
        <geom size=".1"/>
      </body>
      <body name="b">
        <joint name="b"/>
        <geom size=".1"/>
      </body>
      <body name="c">
        <joint name="c"/>
        <geom size=".1"/>
      </body>
      <body name="d">
        <joint name="d"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key" qpos="1 2 3 4"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);

  mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "a"))->type = mjJNT_BALL;
  EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "d")->element), 0);

  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m2->nq, 6);
  EXPECT_THAT(AsVector(m2->key_qpos, 6), ElementsAreArray({1, 0, 0, 0, 2, 3}));

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, ResizeParentKeyframe) {
  static constexpr char xml_parent[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <frame name="frame"/>
        <body name="body">
          <joint/>
          <geom size="0.1"/>
        </body>
      </worldbody>
      <keyframe>
        <key name="home" qpos="1"/>
      </keyframe>
    </mujoco>)";

  static constexpr char xml_child[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <body name="body">
          <joint/>
          <geom size="0.1"/>
        </body>
      </worldbody>
    </mujoco>)";

  static constexpr char xml_expected[] = R"(
    <mujoco model="MuJoCo Model">
      <worldbody>
        <body name="body">
          <joint/>
          <geom size="0.1"/>
        </body>
        <frame name="frame">
          <body name="child-body">
            <joint/>
            <geom size="0.1"/>
          </body>
        </frame>
      </worldbody>
      <keyframe>
        <key name="home" qpos="1 0"/>
      </keyframe>
    </mujoco>)";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  mjs_attach(mjs_findFrame(parent, "frame")->element,
             mjs_findBody(child, "body")->element, "child-", "");

  mjModel* model = mj_compile(parent, 0);
  EXPECT_THAT(model, NotNull());

  mjtNum tol = 0;
  std::string field = "";
  MjModelPtr expected = LoadModelFromString(xml_expected, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();
  EXPECT_LE(CompareModel(model, expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteModel(model);
}

// the keyframes of a parent keep their values and their places when a model
// is attached ahead of some of its joints
TEST_F(MujocoTest, AttachKeepsParentKeyframe) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2" qvel="3 4"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // whether or not the parent was compiled before the attachment
  for (bool compile : {false, true}) {
    std::array<char, 1000> er;
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjModel* m_parent = compile ? mj_compile(parent, nullptr) : nullptr;

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                           mjs_findBody(child, "b")->element, "c_", ""),
                NotNull())
        << mjs_getError(parent);

    // a keyframe which is added now comes after the one of the parent
    mjs_setName(mjs_addKey(parent)->element, "added");

    mjModel* m = mj_compile(parent, nullptr);
    ASSERT_THAT(m, NotNull()) << mjs_getError(parent);

    // the attached joint is between the two joints of the parent
    ASSERT_EQ(m->nq, 3);
    EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "c_j"), 1);
    ASSERT_EQ(m->nkey, 2);
    EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "k"), 0);
    EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "added"), 1);
    EXPECT_THAT(AsVector(m->key_qpos, 3), ElementsAreArray({1, 0, 2}));
    EXPECT_THAT(AsVector(m->key_qvel, 3), ElementsAreArray({3, 0, 4}));

    mj_deleteModel(m);
    mj_deleteModel(m_parent);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

// a keyframe of a parent which does not fit its model, such as one which is
// written ahead for the model being assembled, is left as it is
TEST_F(MujocoTest, AttachLeavesKeyframeWrittenAhead) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2"/>
      <key name="ahead" qpos="3 4 5"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                         mjs_findBody(child, "b")->element, "c_", ""),
              NotNull())
      << mjs_getError(parent);
  mjModel* m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);

  // the keyframe which fits the parent follows its joints, the other does not
  ASSERT_EQ(m->nq, 3);
  ASSERT_EQ(m->nkey, 2);
  EXPECT_THAT(AsVector(m->key_qpos, 3), ElementsAreArray({1, 0, 2}));
  EXPECT_THAT(AsVector(m->key_qpos + 3, 3), ElementsAreArray({3, 4, 5}));

  mj_deleteModel(m);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// the keyframes keep their values when a body of the parent is deleted after
// a model was attached to it, with no compilation in between
TEST_F(MujocoTest, AttachThenDeleteKeepsKeyframes) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="C">
        <joint name="c" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2 3"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="ck" qpos="4"/>
    </keyframe>
  </mujoco>
  )";

  // whether or not the parent was compiled before the attachment
  for (bool compile : {false, true}) {
    std::array<char, 1000> er;
    mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
    ASSERT_THAT(parent, NotNull()) << er.data();
    mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
    ASSERT_THAT(child, NotNull()) << er.data();
    mjModel* m_parent = compile ? mj_compile(parent, nullptr) : nullptr;

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                           mjs_findBody(child, "b")->element, "c_", ""),
                NotNull())
        << mjs_getError(parent);
    ASSERT_EQ(mjs_delete(parent, mjs_findBody(parent, "B")->element), 0)
        << mjs_getError(parent);

    mjModel* m = mj_compile(parent, nullptr);
    ASSERT_THAT(m, NotNull()) << mjs_getError(parent);
    ASSERT_EQ(m->nq, 3);
    EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "c_j"), 1);

    // the deletion moves the keyframe of the parent after the attached one
    ASSERT_EQ(m->nkey, 2);
    EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "c_ck"), 0);
    EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "k"), 1);
    EXPECT_THAT(AsVector(m->key_qpos, 3), ElementsAreArray({0, 4, 0}));
    EXPECT_THAT(AsVector(m->key_qpos + 3, 3), ElementsAreArray({1, 0, 3}));

    mj_deleteModel(m);
    mj_deleteModel(m_parent);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

// a vector which is given to a keyframe between two attachments is laid out
// for the tree as it is then, also when the parent was compiled before, and
// recompiling keeps the state of the joints of the parent
TEST_F(MujocoTest, KeyframeSetBetweenAttachments) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="C">
        <joint name="c" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="x">
        <joint name="x" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="y">
        <joint name="y" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  // whether the parent was compiled before the first attachment, and whether
  // it had the keyframe then or it was added after it
  for (bool compile : {false, true}) {
    for (bool key_first : {true, false}) {
      SCOPED_TRACE(std::string(compile ? "compiled" : "not compiled") +
                   (key_first ? ", keyframe first" : ", keyframe added"));
      std::array<char, 1000> er;
      mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
      ASSERT_THAT(parent, NotNull()) << er.data();
      mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
      ASSERT_THAT(child, NotNull()) << er.data();
      mjsKey* key = nullptr;
      if (key_first) {
        key = mjs_addKey(parent);
        const std::vector<double> qpos = {1, 2, 3};
        mjs_setDouble(key->qpos, qpos.data(), qpos.size());
      }
      mjModel* model = nullptr;
      mjData* data = nullptr;
      if (compile) {
        model = mj_compile(parent, nullptr);
        ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
        data = mj_makeData(model);
        data->qpos[0] = 0.25;
        data->qpos[1] = 0.5;
        data->qpos[2] = 0.75;
        data->qvel[0] = 1.5;
        data->qvel[1] = 2.5;
        data->qvel[2] = 3.5;
      }

      // after the first attachment, the keyframe is given a vector for the
      // joints a, c_x, b, c
      ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                             mjs_findBody(child, "x")->element, "c_", ""),
                  NotNull())
          << mjs_getError(parent);
      if (!key_first) {
        key = mjs_addKey(parent);
      }
      const std::vector<double> qpos = {11, 12, 13, 14};
      mjs_setDouble(key->qpos, qpos.data(), qpos.size());

      // the second attachment puts c_y between c_x and b
      ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "fa")->element,
                             mjs_findBody(child, "y")->element, "c_", ""),
                  NotNull())
          << mjs_getError(parent);
      if (compile) {
        ASSERT_EQ(mj_recompile(parent, nullptr, model, data), 0)
            << mjs_getError(parent);
        EXPECT_THAT(AsVector(data->qpos, 5),
                    ElementsAreArray({0.25, 0.0, 0.0, 0.5, 0.75}));
        EXPECT_THAT(AsVector(data->qvel, 5),
                    ElementsAreArray({1.5, 0.0, 0.0, 2.5, 3.5}));
      } else {
        model = mj_compile(parent, nullptr);
        ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
      }
      ASSERT_EQ(model->nq, 5);
      EXPECT_EQ(mj_name2id(model, mjOBJ_JOINT, "c_y"), 2);
      ASSERT_EQ(model->nkey, 1);
      EXPECT_THAT(AsVector(model->key_qpos, 5),
                  ElementsAreArray({11, 12, 0, 13, 14}));

      mj_deleteData(data);
      mj_deleteModel(model);
      mj_deleteSpec(child);
      mj_deleteSpec(parent);
    }
  }
}

// once a spec which was changed is compiled again, its keyframes are laid out
// for the compiled model until the tree changes, as before the first change
TEST_F(MujocoTest, KeyframeLayoutAfterCompilingChange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="C">
        <joint name="c" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2 3"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(mjs_delete(spec, mjs_findBody(spec, "A")->element), 0)
      << mjs_getError(spec);
  mjModel* m2 = mj_compile(spec, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(spec);

  // a joint which is added between b and c has no value in the keyframe
  mjsBody* c = mjs_findBody(spec, "C");
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "B"), nullptr);
  mjs_addJoint(body, nullptr)->type = mjJNT_SLIDE;
  mjs_addGeom(body, nullptr)->size[0] = 0.1;
  ASSERT_EQ(mjs_delete(spec, c->element), 0) << mjs_getError(spec);

  mjModel* m3 = mj_compile(spec, nullptr);
  ASSERT_THAT(m3, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m3->nq, 2);
  EXPECT_THAT(AsVector(m3->key_qpos, 2), ElementsAreArray({2, 0}));

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteModel(m3);
  mj_deleteSpec(spec);
}

// a keyframe which a deletion stores after an attachment has the time it was
// given meanwhile, when its model is attached to another
TEST_F(MujocoTest, AttachThenDeleteKeepsKeyframeTime) {
  mock_warning_handler.ExpectWarnings();
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="f"/>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();
  mjSpec* grandparent = mj_makeSpec();

  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f")->element,
                         mjs_findBody(child, "b")->element, "c_", ""),
              NotNull())
      << mjs_getError(parent);
  mjs_asKey(mjs_findElement(parent, mjOBJ_KEY, "k"))->time = 5;
  ASSERT_EQ(mjs_delete(parent, mjs_findBody(parent, "B")->element), 0)
      << mjs_getError(parent);

  mjsFrame* frame = mjs_addFrame(mjs_findBody(grandparent, "world"), nullptr);
  ASSERT_THAT(
      mjs_attach(frame->element, mjs_findBody(parent, "A")->element, "p_", ""),
      NotNull())
      << mjs_getError(grandparent);
  mjModel* m = mj_compile(grandparent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(grandparent);
  ASSERT_EQ(m->nq, 1);
  ASSERT_EQ(m->nkey, 1);
  EXPECT_EQ(m->key_time[0], 5);
  EXPECT_EQ(m->key_qpos[0], 1);

  mj_deleteModel(m);
  mj_deleteSpec(grandparent);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// a keyframe of a parent does not take the values which a copy of the parent
// holds for it, when a body of that copy is attached to the parent
TEST_F(MujocoTest, AttachCopyOfParentKeepsKeyframe) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="f1"/>
      <frame name="f2"/>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  // the keyframe is stored in the parent, then in its copy as well
  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element,
                         mjs_findBody(child, "b")->element, "c_", ""),
              NotNull())
      << mjs_getError(parent);
  mjSpec* copy = mj_copySpec(parent);
  ASSERT_THAT(copy, NotNull());
  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element,
                         mjs_findBody(copy, "A")->element, "x_", ""),
              NotNull())
      << mjs_getError(parent);

  mjModel* m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);
  ASSERT_EQ(m->nq, 3);
  EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "x_a"), 2);
  ASSERT_EQ(m->nkey, 2);
  EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "k"), 0);
  EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "x_k"), 1);
  EXPECT_THAT(AsVector(m->key_qpos, 3), ElementsAreArray({1, 0, 0}));
  EXPECT_THAT(AsVector(m->key_qpos + 3, 3), ElementsAreArray({0, 0, 1}));

  mj_deleteModel(m);
  mj_deleteSpec(parent);
  mj_deleteSpec(copy);
  mj_deleteSpec(child);
}

// a keyframe keeps its values when a subtree of its model is attached to the
// model itself, and every attachment adds a copy of it for the new subtree
TEST_F(MujocoTest, SelfAttachKeepsKeyframe) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="A">
        <joint name="a" type="slide"/>
        <geom size=".1"/>
        <frame name="fa"/>
      </body>
      <body name="B">
        <joint name="b" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1 2"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjs_setDeepCopy(spec, true);

  // two copies of B, ahead of B in the tree
  mjsElement* frame = mjs_findFrame(spec, "fa")->element;
  const mjsElement* body = mjs_findBody(spec, "B")->element;
  ASSERT_THAT(mjs_attach(frame, body, "c1_", ""), NotNull())
      << mjs_getError(spec);
  ASSERT_THAT(mjs_attach(frame, body, "c2_", ""), NotNull())
      << mjs_getError(spec);

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nq, 4);
  EXPECT_EQ(mj_name2id(m, mjOBJ_JOINT, "b"), 3);
  ASSERT_EQ(m->nkey, 3);
  EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "k"), 0);
  EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "c1_k"), 1);
  EXPECT_EQ(mj_name2id(m, mjOBJ_KEY, "c2_k"), 2);
  EXPECT_THAT(AsVector(m->key_qpos, 4), ElementsAreArray({1, 0, 0, 2}));
  EXPECT_THAT(AsVector(m->key_qpos + 4, 4), ElementsAreArray({1, 2, 0, 0}));
  EXPECT_THAT(AsVector(m->key_qpos + 8, 4), ElementsAreArray({1, 0, 2, 0}));

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a keyframe of a parent has the default configuration of the joints of a
// model which is attached to it
TEST_F(MujocoTest, AttachDefaultsInParentKeyframe) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <frame name="frame" pos="0 0 1"/>
    </worldbody>
    <keyframe>
      <key name="k" qpos="1"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="free" pos="1 0 0">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "frame")->element,
                         mjs_findBody(child, "free")->element, "c_", ""),
              NotNull())
      << mjs_getError(parent);
  mjModel* m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);

  // the free body is where the frame it is attached to puts it
  ASSERT_EQ(m->nq, 8);
  ASSERT_EQ(m->nkey, 1);
  EXPECT_THAT(AsVector(m->key_qpos, 8),
              ElementsAreArray({1, 1, 0, 1, 1, 0, 0, 0}));

  mj_deleteModel(m);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// a keyframe which is shorter than the model survives a change to the tree:
// what it does not give takes the default configuration
TEST_F(MujocoTest, ShortKeyframeSurvivesTreeChanges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <body name="b">
        <joint type="slide"/>
        <geom size=".1"/>
      </body>
      <frame pos="1 0 0" euler="0 0 90">
        <body name="c" pos="0 0 1">
          <freejoint/>
          <geom size=".1"/>
        </body>
      </frame>
    </worldbody>
    <keyframe>
      <key name="short" qpos="2 3"/>
      <key name="default"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  // delete a body: the free joint is as in the default keyframe
  EXPECT_EQ(mjs_delete(spec, mjs_findBody(spec, "a")->element), 0)
      << mjs_getError(spec);
  mjModel* m1 = mj_compile(spec, nullptr);
  ASSERT_THAT(m1, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m1->nq, 8);
  ASSERT_EQ(m1->nkey, 2);
  EXPECT_EQ(m1->key_qpos[0], 3);
  EXPECT_EQ(AsVector(m1->key_qpos + 1, 7), AsVector(m1->key_qpos + 8 + 1, 7));

  // attach the model: likewise
  mjSpec* parent = mj_makeSpec();
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);
  ASSERT_THAT(mjs_attach(frame->element, child->element, "child-", ""),
              NotNull())
      << mjs_getError(parent);
  mjModel* m2 = mj_compile(parent, nullptr);
  ASSERT_THAT(m2, NotNull()) << mjs_getError(parent);
  ASSERT_EQ(m2->nq, 9);
  ASSERT_EQ(m2->nkey, 2);
  EXPECT_THAT(AsVector(m2->key_qpos, 2), ElementsAreArray({2, 3}));
  EXPECT_EQ(AsVector(m2->key_qpos + 2, 7), AsVector(m2->key_qpos + 9 + 2, 7));

  mj_deleteModel(m1);
  mj_deleteModel(m2);
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, KeyframeSizeError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset="0 1 0">
        <body name="ball" pos="0 0 1">
          <geom type="sphere" size="0.1"/>
          <joint/>
        </body>
      </replicate>
    </worldbody>

    <keyframe>
      <key name="valid_qpos" qpos="0.5"/>
      <key name="invalid_qpos" qpos="0.5 0.25 0.1"/>
    </keyframe>
  </mujoco>
  )";

  // the size is checked against the model with its replicas in place
  std::array<char, 1000> er;
  MjModelPtr model = LoadModelFromString(xml, er.data(), er.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(er.data(), HasSubstr("keyframe 'invalid_qpos': invalid qpos "
                                   "size, expected 2, got 3"));
}

TEST_F(MujocoTest, DifferentUnitsAllowed) {
  static constexpr char gchild_xml[] = R"(
  <mujoco>
    <compiler angle="degree"/>

    <worldbody>
      <body name="gchild" euler="-90 0 0">
        <geom type="box" size="1 1 1"/>
        <joint name="gchild_joint" range="-180 180"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char child_xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>

    <worldbody>
      <body name="child" euler="-1.5707963 0 0">
        <geom type="box" size="1 1 1"/>
        <joint name="child_joint" range="-3.1415926 3.1415926"/>
        <frame name="frame" euler="1.5707963 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char parent_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom type="box" size="1 1 1"/>
        <joint name="parent_joint" range="-180 180"/>
        <frame name="frame" euler="90 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* gchild = mj_parseXMLString(gchild_xml, 0, error.data(), error.size());
  mjSpec* child = mj_parseXMLString(child_xml, 0, error.data(), error.size());
  mjSpec* spec = mj_parseXMLString(parent_xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjs_attach(mjs_findFrame(child, "frame")->element,
             mjs_findBody(gchild, "gchild")->element, "gchild_", "");
  mjs_attach(mjs_findFrame(spec, "frame")->element,
             mjs_findBody(child, "child")->element, "child_", "");

  mjModel* model = mj_compile(spec, 0);
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->njnt, 3);
  EXPECT_NEAR(model->jnt_range[0], -mjPI, 1e-6);
  EXPECT_NEAR(model->jnt_range[1], mjPI, 1e-6);
  EXPECT_NEAR(model->jnt_range[2], -mjPI, 1e-6);
  EXPECT_NEAR(model->jnt_range[3], mjPI, 1e-6);
  EXPECT_NEAR(model->jnt_range[4], -mjPI, 1e-6);
  EXPECT_NEAR(model->jnt_range[5], mjPI, 1e-6);
  EXPECT_NEAR(model->body_quat[4], 1, 1e-12);
  EXPECT_NEAR(model->body_quat[5], 0, 1e-12);
  EXPECT_NEAR(model->body_quat[6], 0, 1e-12);
  EXPECT_NEAR(model->body_quat[7], 0, 1e-12);

  mjSpec* copied_spec = mj_copySpec(spec);
  ASSERT_THAT(copied_spec, NotNull());
  mj_deleteSpec(gchild);
  mj_deleteSpec(child);
  mj_deleteSpec(spec);
  mj_deleteModel(model);

  // check that deleting `parent` or `child` does not invalidate the copy
  mjModel* copied_model = mj_compile(copied_spec, 0);
  EXPECT_THAT(copied_model, NotNull());
  EXPECT_THAT(copied_model->njnt, 3);
  EXPECT_NEAR(copied_model->jnt_range[0], -mjPI, 1e-6);
  EXPECT_NEAR(copied_model->jnt_range[1], mjPI, 1e-6);
  EXPECT_NEAR(copied_model->jnt_range[2], -mjPI, 1e-6);
  EXPECT_NEAR(copied_model->jnt_range[3], mjPI, 1e-6);
  EXPECT_NEAR(copied_model->jnt_range[4], -mjPI, 1e-6);
  EXPECT_NEAR(copied_model->jnt_range[5], mjPI, 1e-6);
  EXPECT_NEAR(copied_model->body_quat[4], 1, 1e-12);
  EXPECT_NEAR(copied_model->body_quat[5], 0, 1e-12);
  EXPECT_NEAR(copied_model->body_quat[6], 0, 1e-12);
  EXPECT_NEAR(copied_model->body_quat[7], 0, 1e-12);

  mj_deleteSpec(copied_spec);
  mj_deleteModel(copied_model);
}

TEST_F(MujocoTest, DifferentOptionsInAttachedFrame) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody/>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <compiler eulerseq="zyx"/>
    <worldbody>
      <frame name="child" >
        <site euler="0 90 180"/>
      </frame>
    </worldbody>
  </mujoco>
  )";

  // load specs and compile child
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, nullptr, 0);
  EXPECT_THAT(parent, NotNull());
  mjSpec* child1 = mj_parseXMLString(xml_child, 0, nullptr, 0);
  EXPECT_THAT(child1, NotNull());
  mjModel* m_child1 = mj_compile(child1, 0);
  EXPECT_THAT(m_child1, NotNull());
  mjSpec* child2 = mj_parseXMLString(xml_child, 0, nullptr, 0);
  EXPECT_THAT(child2, NotNull());
  mjModel* m_child2 = mj_compile(child1, 0);
  EXPECT_THAT(m_child2, NotNull());

  // attach child frame to parent worldbody
  mjsBody* world = mjs_findBody(parent, "world");
  EXPECT_THAT(world, NotNull());
  mjsFrame* child1_frame = mjs_findFrame(child1, "child");
  EXPECT_THAT(child1_frame, NotNull());
  mjsFrame* child2_frame = mjs_findFrame(child2, "child");
  EXPECT_THAT(child2_frame, NotNull());
  mjsElement* attached_frame1 =
      mjs_attach(world->element, child1_frame->element, "child-", "-1");
  EXPECT_THAT(attached_frame1, NotNull());
  mjsElement* attached_frame2 =
      mjs_attach(world->element, child2_frame->element, "child-", "-2");
  EXPECT_THAT(attached_frame2, NotNull());

  // wrap the child frame in the parent frame and compile
  mjModel* m_attached = mj_compile(parent, 0);
  EXPECT_THAT(m_attached, NotNull());
  EXPECT_NEAR(m_attached->site_quat[0], m_child1->site_quat[0], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[1], m_child1->site_quat[1], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[2], m_child1->site_quat[2], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[3], m_child1->site_quat[3], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[4], m_child2->site_quat[0], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[5], m_child2->site_quat[1], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[6], m_child2->site_quat[2], 1e-6);
  EXPECT_NEAR(m_attached->site_quat[7], m_child2->site_quat[3], 1e-6);

  mj_deleteSpec(parent);
  mj_deleteSpec(child1);
  mj_deleteModel(m_child1);
  mj_deleteSpec(child2);
  mj_deleteModel(m_child2);
  mj_deleteModel(m_attached);
}

TEST_F(MujocoTest, NotCopyAttachedSpec) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <asset>
      <model name="child" file="xml_child.xml"/>
    </asset>
    <worldbody>
      <body name="parent">
        <geom name="geom" size="2"/>
        <attach model="child" body="body" prefix="other"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <geom name="geom" size="1" pos="2 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "xml_child.xml", xml_child, sizeof(xml_child));

  std::array<char, 1024> er;
  mjSpec* spec = mj_parseXMLString(xml_parent, vfs.get(), er.data(), er.size());
  EXPECT_THAT(spec, NotNull()) << er.data();

  mjModel* model = mj_compile(spec, vfs.get());
  EXPECT_THAT(model, NotNull()) << er.data();

  mjSpec* child = mjs_findSpec(spec, "child");
  EXPECT_THAT(child, NotNull());

  mjSpec* copy = mj_copySpec(spec);
  EXPECT_THAT(copy, NotNull());

  mjSpec* child_copy = mjs_findSpec(copy, "child");
  EXPECT_THAT(child_copy, NotNull());
  EXPECT_EQ(child_copy, child);

  mj_deleteSpec(spec);
  mj_deleteSpec(copy);
  mj_deleteModel(model);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, ApplyNameSpaceToDefaults) {
  static constexpr char xml_c[] = R"(
  <mujoco>
    <default>
      <default class="mesh">
        <mesh scale="0.001 0.001 0.001"/>
      </default>
    </default>
    <asset>
      <mesh file="cube.obj" class="mesh"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom type="mesh" mesh="cube"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char xml_p[] = R"(
  <mujoco>
    <worldbody>
      <frame name="parent"/>
    </worldbody>
  </mujoco>
  )";

  static constexpr char cube[] = R"(
  v -0.500000 -0.500000  0.500000
  v  0.500000 -0.500000  0.500000
  v -0.500000  0.500000  0.500000
  v  0.500000  0.500000  0.500000
  v -0.500000  0.500000 -0.500000
  v  0.500000  0.500000 -0.500000
  v -0.500000 -0.500000 -0.500000
  v  0.500000 -0.500000 -0.500000)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1024> err;
  mjSpec* child = mj_parseXMLString(xml_c, vfs.get(), err.data(), err.size());
  EXPECT_THAT(child, NotNull()) << err.data();
  mjSpec* parent = mj_parseXMLString(xml_p, 0, err.data(), err.size());
  EXPECT_THAT(parent, NotNull()) << err.data();

  mjsElement* attached =
      mjs_attach(mjs_findFrame(parent, "parent")->element,
                 mjs_findBody(child, "body")->element, "child-", "");
  EXPECT_THAT(attached, NotNull());

  mjModel* model = mj_compile(parent, vfs.get());
  EXPECT_THAT(model, NotNull());

  mj_deleteSpec(child);
  mj_deleteSpec(parent);
  mj_deleteModel(model);
  mj_deleteVFS(vfs.get());
}

TEST_F(MujocoTest, DetachDefault) {
  static constexpr char xml_c[] = R"(
  <mujoco>
    <default>
      <default class="parent">
        <default class="child1">
          <mesh scale="0.001 0.001 0.001"/>
        </default>
        <default class="child2">
          <mesh scale="0.001 0.001 0.001"/>
        </default>
      </default>
    </default>
    <asset>
      <mesh file="cube.obj" class="child2"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom type="mesh" mesh="cube"/>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char cube[] = R"(
  v -0.500000 -0.500000  0.500000
  v  0.500000 -0.500000  0.500000
  v -0.500000  0.500000  0.500000
  v  0.500000  0.500000  0.500000
  v -0.500000  0.500000 -0.500000
  v  0.500000  0.500000 -0.500000
  v -0.500000 -0.500000 -0.500000
  v  0.500000 -0.500000 -0.500000)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1024> err;
  mjSpec* spec = mj_parseXMLString(xml_c, vfs.get(), err.data(), err.size());
  EXPECT_THAT(spec, NotNull()) << err.data();

  // get default
  mjsDefault* child = mjs_findDefault(spec, "child1");
  EXPECT_THAT(child, NotNull());

  // delete default
  EXPECT_EQ(mjs_delete(spec, child->element), 0);
  child = mjs_findDefault(spec, "child1");
  EXPECT_THAT(child, IsNull());

  // try and detach previously detached default, should fail
  EXPECT_EQ(mjs_delete(spec, nullptr), -1);
  child = mjs_findDefault(spec, "child1");
  EXPECT_THAT(child, IsNull());

  // detach parent
  mjsDefault* parent = mjs_findDefault(spec, "parent");
  EXPECT_THAT(parent, NotNull());
  mjs_delete(spec, parent->element);

  // both parent and remaining child should be removed
  parent = mjs_findDefault(spec, "parent");
  EXPECT_THAT(parent, IsNull());
  child = mjs_findDefault(spec, "child2");
  EXPECT_THAT(child, IsNull());

  // error when trying to detach the 'main' default
  mjsDefault* main = mjs_findDefault(spec, "main");
  EXPECT_THAT(main, NotNull());
  EXPECT_EQ(mjs_delete(spec, main->element), -1);
  EXPECT_THAT(mjs_getError(spec),
              HasSubstr("cannot remove the global default ('main')"));

  main = mjs_findDefault(spec, "main");
  EXPECT_THAT(main, NotNull());

  mj_deleteVFS(vfs.get());
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, DeleteForeignDefault) {
  mjSpec* owner = mj_makeSpec();
  mjSpec* other = mj_makeSpec();
  mjsDefault* foreign = mjs_addDefault(owner, "foreign", nullptr);
  EXPECT_THAT(mjs_addDefault(other, "mine", nullptr), NotNull());

  // deleting a default owned by another spec fails
  EXPECT_EQ(mjs_delete(other, foreign->element), -1);
  EXPECT_THAT(mjs_getError(other), HasSubstr("default is not in this model"));

  // neither spec is modified
  EXPECT_THAT(mjs_findDefault(owner, "foreign"), NotNull());
  EXPECT_THAT(mjs_findDefault(other, "mine"), NotNull());

  // both default trees are intact, saving traverses them
  for (mjSpec* spec : {owner, other}) {
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    mj_deleteModel(model);
  }
  EXPECT_THAT(SaveAndReadXml(owner), HasSubstr("class=\"foreign\""));
  EXPECT_THAT(SaveAndReadXml(other), HasSubstr("class=\"mine\""));

  mj_deleteSpec(other);
  mj_deleteSpec(owner);
}

TEST_F(MujocoTest, DeleteForeignElement) {
  mjSpec* owner = mj_makeSpec();
  mjSpec* other = mj_makeSpec();
  mjsBody* body = mjs_addBody(mjs_findBody(owner, "world"), nullptr);
  mjsGeom* geom = mjs_addGeom(body, nullptr);
  geom->size[0] = 1;

  // deleting an element owned by another spec fails
  EXPECT_EQ(mjs_delete(other, body->element), -1);
  EXPECT_THAT(mjs_getError(other), HasSubstr("element is not in this model"));

  // the owner still compiles with the element
  mjModel* model = mj_compile(owner, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(owner);
  EXPECT_EQ(model->nbody, 2);
  EXPECT_EQ(model->ngeom, 1);

  mj_deleteModel(model);
  mj_deleteSpec(other);
  mj_deleteSpec(owner);
}

TEST_F(MujocoTest, ErrorWhenCompilingOrphanedSpec) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> er;
  mjSpec* child = mj_parseXMLString(xml, 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();
  mjSpec* parent = mj_makeSpec();
  EXPECT_THAT(parent, NotNull());
  mjsBody* body = mjs_findBody(child, "a");
  EXPECT_THAT(body, NotNull());
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);
  EXPECT_THAT(frame, NotNull());
  mjs_attach(frame->element, body->element, "child-", "");
  mj_deleteSpec(parent);
  mjModel* model = mj_compile(child, 0);
  EXPECT_THAT(model, IsNull());
  EXPECT_THAT(mjs_getError(child), HasSubstr("by reference to a parent"));
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, SetFrameReverseOrder) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsFrame* child = mjs_addFrame(world, nullptr);
  mjsFrame* parent = mjs_addFrame(world, nullptr);
  mjs_setName(child->element, "child");
  mjs_setName(parent->element, "parent");
  mjs_setFrame(child->element, parent);
  mjSpec* copy = mj_copySpec(spec);
  EXPECT_THAT(copy, NotNull());
  EXPECT_THAT(mjs_findFrame(copy, "child"), NotNull());
  EXPECT_THAT(mjs_findFrame(copy, "parent"), NotNull());
  mj_deleteSpec(spec);
  mj_deleteSpec(copy);
}

TEST_F(MujocoTest, CopyInertialInFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <frame name="frame" pos="1 0 0">
          <inertial pos="0 1 0" mass="1" diaginertia="1 1 1"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // move the frame in a copy, the copied inertial follows its own frame
  mjSpec* copy = mj_copySpec(spec);
  ASSERT_THAT(copy, NotNull());
  mjs_findFrame(copy, "frame")->pos[0] = 2;
  mjModel* model = mj_compile(copy, 0);
  ASSERT_THAT(model, NotNull());
  EXPECT_EQ(model->body_ipos[3], 2);
  EXPECT_EQ(model->body_ipos[4], 1);
  EXPECT_EQ(model->body_ipos[5], 0);

  mj_deleteModel(model);
  mj_deleteSpec(copy);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, UserValue) {
  mjSpec* spec = mj_makeSpec();
  EXPECT_THAT(spec, NotNull());
  mjsBody* body = mjs_addBody(mjs_findBody(spec, "world"), nullptr);
  EXPECT_THAT(body, NotNull());
  std::string data = "data";
  mjs_setUserValue(body->element, "key", data.data());
  EXPECT_THAT(mjs_getUserValue(body->element, "invalid_key"), IsNull());
  const void* payload = mjs_getUserValue(body->element, "key");
  EXPECT_STREQ(static_cast<const char*>(payload), data.c_str());
  mjs_deleteUserValue(body->element, "key");
  EXPECT_THAT(mjs_getUserValue(body->element, "key"), IsNull());

  std::string* heap_data = new std::string("heap_data");
  mjs_setUserValueWithCleanup(
      body->element, "key", heap_data,
      [](const void* data) { delete static_cast<const std::string*>(data); });
  payload = mjs_getUserValue(body->element, "key");
  EXPECT_STREQ(static_cast<const std::string*>(payload)->c_str(),
               heap_data->c_str());
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, CompilerTimers) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="grid" type="2d" builtin="checker" width="300" height="300" rgb1=".1 .2 .3" rgb2=".2 .3 .4"/>
      <material name="grid" texture="grid"/>
    </asset>
    <worldbody>
      <geom type="plane" size="1 1 1" material="grid"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());

  EXPECT_GT(mjs_getTimer(spec)[mjCTIMER_TOTAL], 0);
  EXPECT_GT(mjs_getTimer(spec)[mjCTIMER_ASSETS], 0);
  EXPECT_GT(mjs_getTimer(spec)[mjCTIMER_TEXTURE], 0);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// -------------------- test compile warning infrastructure --------------------

TEST_F(MujocoTest, CompileWarningCount) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom size="1"/>
        <flexcomp name="grid" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                  dim="2" radius="0.01">
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());

  // flex with no passive forces should produce a warning
  EXPECT_GT(mjs_numWarnings(spec), 0);
  EXPECT_THAT(mjs_getWarning(spec, 0), HasSubstr("not rigid"));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, CompileWarningOutOfBounds) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsGeom* geom = mjs_addGeom(world, 0);
  geom->size[0] = 1;

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());

  // no warnings expected for simple model
  EXPECT_EQ(mjs_numWarnings(spec), 0);
  EXPECT_THAT(mjs_getWarning(spec, 0), IsNull());
  EXPECT_THAT(mjs_getWarning(spec, -1), IsNull());

  // nullptr spec should not crash
  EXPECT_EQ(mjs_numWarnings(nullptr), 0);
  EXPECT_THAT(mjs_getWarning(nullptr, 0), IsNull());

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, RecompileClearsCompileWarnings) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom size="1"/>
        <flexcomp name="grid" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                  dim="2" radius="0.01">
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());
  int first_count = mjs_numWarnings(spec);
  EXPECT_GT(first_count, 0);

  // recompile — warnings should be regenerated, not accumulated
  mj_deleteModel(model);
  model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());
  EXPECT_EQ(mjs_numWarnings(spec), first_count);

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, LoadXMLWarningInErrorBuffer) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom size="1"/>
        <flexcomp name="grid" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                  dim="2" radius="0.01">
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  // write xml to VFS
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));

  std::array<char, 1024> error;
  error[0] = '\0';
  mjModel* model = mj_loadXML("model.xml", &vfs, error.data(), error.size());
  ASSERT_THAT(model, NotNull());

  // warning should be in the error buffer
  EXPECT_THAT(error.data(), HasSubstr("not rigid"));

  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(MujocoTest, CompileWarningChainedToHandler) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom size="1"/>
        <flexcomp name="grid" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                  dim="2" radius="0.01">
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // install a custom log handler that captures warnings
  std::vector<std::string> captured_warnings;
  static thread_local std::vector<std::string>* capture_ptr = nullptr;
  capture_ptr = &captured_warnings;

  // install custom log handler (replaces global, so mock is bypassed)

  mjfLogHandler prev = mju_setLogHandler([](const mjLogMessage* msg) {
    if (msg->level == mjLOG_WARNING && capture_ptr) {
      capture_ptr->push_back(msg->subject);
    }
  });

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull());

  // restore log handler
  mju_setLogHandler(prev);
  capture_ptr = nullptr;

  // chaining should have forwarded warnings to our handler
  EXPECT_THAT(captured_warnings, testing::Contains(HasSubstr("not rigid")));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, WarningAccumulationAndRetrieval) {
  mock_warning_handler.ExpectWarnings();
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <worldbody>
      <frame name="parent"/>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="child">
        <frame name="child_frame"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_gchild[] = R"(
  <mujoco>
    <worldbody>
      <body name="gchild"/>
    </worldbody>
    <keyframe>
      <key name="k1"/>
    </keyframe>
  </mujoco>
  )";

  std::array<char, 1024> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();
  mjSpec* gchild = mj_parseXMLString(xml_gchild, 0, er.data(), er.size());
  ASSERT_THAT(gchild, NotNull()) << er.data();

  mjs_setDeepCopy(parent, true);
  mjs_setDeepCopy(child, true);
  mjs_setDeepCopy(gchild, true);

  mjs_attach(mjs_findFrame(child, "child_frame")->element,
             mjs_findBody(gchild, "gchild")->element, "gchild-", "");

  mjs_attach(mjs_findFrame(parent, "parent")->element,
             mjs_findBody(child, "child")->element, "child-", "");

  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0),
              HasSubstr("Child model has pending keyframes"));
  EXPECT_THAT(mjs_getWarning(parent, 1), IsNull());

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
  mj_deleteSpec(gchild);
}

TEST_F(MujocoTest, CompilationWarningsClearedOnRecompile) {
  mock_warning_handler.ExpectWarnings();
  // flex with no edge stiffness or equality triggers passive forces warning
  static constexpr char xml[] = R"(
  <mujoco>
  <worldbody>
    <flexcomp name="test" type="grid" count="4 4 1" spacing=".2 .2 .2"
              dim="2" radius=".1"/>
  </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();

  // first compile: should generate flex warning
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  int n1 = mjs_numWarnings(spec);
  EXPECT_EQ(n1, 1);
  EXPECT_THAT(mjs_getWarning(spec, 0),
              HasSubstr("no equality constraints or passive forces"));

  // recompile: warnings should be cleared and regenerated
  mj_deleteModel(m);
  m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  EXPECT_EQ(mjs_numWarnings(spec), n1);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, MjEncodeNativeFormats) {
  // simple test spec
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* body = mjs_addBody(world, nullptr);
  mjsGeom* geom = mjs_addGeom(body, nullptr);
  geom->size[0] = 1.0;
  geom->size[1] = 1.0;
  geom->size[2] = 1.0;

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull());

  std::filesystem::path tmp_dir = std::filesystem::temp_directory_path();
  std::string xml_path = (tmp_dir / "test_encode.xml").string();
  std::string mjb_path = (tmp_dir / "test_encode.mjb").string();
  std::string txt_path = (tmp_dir / "test_encode.txt").string();

  char error[1000];

  // XML Encoding with spec
  EXPECT_GT(mj_encode(spec, model, xml_path.c_str(), nullptr, nullptr, error,
                      sizeof(error)),
            0);
  EXPECT_TRUE(std::filesystem::exists(xml_path));
  EXPECT_GT(std::filesystem::file_size(xml_path), 0);

  // XML Encoding without spec (should fail because no XML loaded in global
  // spec)
  std::filesystem::remove(xml_path);
  // free global spec
  mj_freeLastXML();
  EXPECT_EQ(mj_encode(nullptr, model, xml_path.c_str(), nullptr, nullptr, error,
                      sizeof(error)),
            -1);
  EXPECT_THAT(error, HasSubstr("No XML model loaded"));

  // a save which fails does not create the file
  EXPECT_FALSE(std::filesystem::exists(xml_path));

  // MJB Encoding (spec can be null)
  EXPECT_GT(mj_encode(nullptr, model, mjb_path.c_str(), nullptr, nullptr, error,
                      sizeof(error)),
            0);
  EXPECT_TRUE(std::filesystem::exists(mjb_path));
  EXPECT_GT(std::filesystem::file_size(mjb_path), 0);

  mjModel* model2 = mj_loadModel(mjb_path.c_str(), nullptr);
  EXPECT_THAT(model2, NotNull());
  mj_deleteModel(model2);

  // TXT Encoding (spec can be null)
  EXPECT_GT(mj_encode(nullptr, model, txt_path.c_str(), nullptr, nullptr, error,
                      sizeof(error)),
            0);
  EXPECT_TRUE(std::filesystem::exists(txt_path));
  EXPECT_GT(std::filesystem::file_size(txt_path), 0);

  // Clean up
  std::filesystem::remove(mjb_path);
  std::filesystem::remove(txt_path);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// Tests that joint ordering is preserved after attach operations
TEST_F(MujocoTest, AttachPreservesJointOrder) {
  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="slider_body">
        <joint name="slide_joint" type="slide"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
      <body name="free_body" pos="1 0 0">
        <freejoint name="free_joint"/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  mjSpec* parent = mj_makeSpec();
  mjsBody* parent_world = mjs_findBody(parent, "world");
  mjsFrame* attach_frame = mjs_addFrame(parent_world, nullptr);
  mjs_setName(attach_frame->element, "attach_frame");

  std::array<char, 1000> error;
  mjSpec* child =
      mj_parseXMLString(child_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child, NotNull()) << error.data();

  mjsElement* attached =
      mjs_attach(attach_frame->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull());

  // Verify joint order: slide joint should come before free joint
  int slide_id = mj_name2id(model, mjOBJ_JOINT, "child_slide_joint");
  int free_id = mj_name2id(model, mjOBJ_JOINT, "child_free_joint");
  ASSERT_GE(slide_id, 0);
  ASSERT_GE(free_id, 0);
  EXPECT_LT(slide_id, free_id)
      << "Slide joint should have a lower ID than free joint";

  // Export and re-import to test round-trip preservation
  std::string exported_xml = SaveAndReadXml(parent);

  mjSpec* reimported = mj_parseXMLString(exported_xml.c_str(), nullptr,
                                         error.data(), error.size());
  ASSERT_THAT(reimported, NotNull()) << error.data();

  mjModel* reimported_model = mj_compile(reimported, nullptr);
  ASSERT_THAT(reimported_model, NotNull());

  // Verify joint order is preserved after round-trip
  int reimported_slide_id =
      mj_name2id(reimported_model, mjOBJ_JOINT, "child_slide_joint");
  int reimported_free_id =
      mj_name2id(reimported_model, mjOBJ_JOINT, "child_free_joint");
  ASSERT_GE(reimported_slide_id, 0);
  ASSERT_GE(reimported_free_id, 0);
  EXPECT_EQ(reimported_slide_id, slide_id)
      << "Joint IDs should be the same after XML round-trip";
  EXPECT_EQ(reimported_free_id, free_id)
      << "Joint IDs should be the same after XML round-trip";

  mj_deleteModel(model);
  mj_deleteModel(reimported_model);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
  mj_deleteSpec(reimported);
}

TEST_F(MujocoTest, DeleteReclaimsMemoryImmediately) {
  mjSpec* spec = mj_makeSpec();
  mjsBody* world = mjs_findBody(spec, "world");

  int cleanup_count = 0;
  auto cleanup = +[](const void* data) {
    *static_cast<int*>(const_cast<void*>(data)) += 1;
  };

  mjsBody* body = mjs_addBody(world, nullptr);
  mjsGeom* geom = mjs_addGeom(body, nullptr);
  mjs_setUserValueWithCleanup(body->element, "tracker", &cleanup_count,
                              cleanup);
  mjs_setUserValueWithCleanup(geom->element, "tracker", &cleanup_count,
                              cleanup);

  EXPECT_EQ(cleanup_count, 0);
  EXPECT_EQ(mjs_delete(spec, body->element), 0);
  EXPECT_EQ(cleanup_count, 2);

  mj_deleteSpec(spec);
  EXPECT_EQ(cleanup_count, 2);
}

// ------------------------------- mj_copyBack ---------------------------------

using CopyBackTest = MujocoTest;

// The fields which differ between two specs, each as "geom[0] 'name': field".
std::vector<std::string> ChangedFields(const mjSpec* before,
                                       const mjSpec* after) {
  std::vector<std::string> fields;
  std::istringstream lines(CompareSpec(before, after, 1000));
  std::string line;
  while (std::getline(lines, line)) {
    fields.push_back(line.substr(0, line.rfind(": ")));
  }
  return fields;
}

// The indices of the arrays, among those given with their lengths, which
// differ between two models.
std::vector<int> DifferentArrays(const mjModel* m1, const mjModel* m2,
                                 std::vector<mjtNum * mjModel::*> arrays,
                                 std::vector<int> lengths) {
  std::vector<int> different;
  for (int i = 0; i < arrays.size(); i++) {
    for (int j = 0; j < lengths[i]; j++) {
      mjtNum a = (m1->*arrays[i])[j];
      mjtNum b = (m2->*arrays[i])[j];
      if (mju_abs(a - b) > MjTol(1e-13, 1e-5)) {
        different.push_back(i);
        break;
      }
    }
  }
  return different;
}

// what was changed in the model is written to the spec, where it can be read
// and from where it is compiled again; nothing else in the spec is touched
TEST_F(CopyBackTest, WritesChangedValues) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree"/>
    <default>
      <geom friction=".5 .25 .125"/>
    </default>
    <worldbody>
      <body name="arm" pos="0 0 1" euler="0 0 90">
        <joint name="hinge" range="-45 45" damping="1"/>
        <geom name="edited" size=".25" euler="0 90 0"/>
        <geom name="kept" size=".5" pos="1 0 0" euler="0 90 0"/>
      </body>
    </worldbody>
    <actuator>
      <position name="servo" joint="hinge" kp="8"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);

  // a model which was not changed changes nothing
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(ChangedFields(authored, spec), IsEmpty());

  model->geom_friction[0] = 0.75;
  model->geom_rgba[1] = 0.25f;
  model->dof_damping[0] = 2;
  model->jnt_range[1] = 0.5;
  model->actuator_gainprm[0] = 16;
  model->actuator_biasprm[1] = -16;
  model->opt.gravity[2] = -5;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  // the changes can be read in the spec, in its units
  EXPECT_THAT(
      ChangedFields(authored, spec),
      ElementsAre("spec: option.gravity[2]", "spec: authored.option",
                  "joint[0] 'hinge': range[1]", "joint[0] 'hinge': damping[0]",
                  "geom[0] 'edited': friction[0]", "geom[0] 'edited': rgba[1]",
                  "actuator[0] 'servo': gainprm[0]",
                  "actuator[0] 'servo': biasprm[1]"));
  mjsGeom* edited = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "edited"));
  mjsJoint* hinge = mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "hinge"));
  EXPECT_EQ(edited->friction[0], 0.75);
  EXPECT_EQ(edited->friction[1], 0.25);
  EXPECT_EQ(edited->alt.type, mjORIENTATION_EULER);
  EXPECT_EQ(hinge->damping[0], 2);
  EXPECT_EQ(hinge->range[0], -45);
  EXPECT_NEAR(hinge->range[1], 0.5 * 180 / mjPI, 1e-12);
  EXPECT_EQ(spec->option.gravity[2], -5);
  EXPECT_EQ(mjs_isAuthored(spec, spec->option.gravity), 1);

  // and are in the model which is compiled from it
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  std::string field;
  EXPECT_LE(CompareModel(model, again, field), MjTol(1e-14, 1e-6)) << field;
  EXPECT_EQ(again->opt.gravity[2], -5);

  // copying back again changes nothing
  mjSpec* copied = mj_copySpec(spec);
  ASSERT_EQ(mj_copyBack(spec, again), 1) << mjs_getError(spec);
  EXPECT_THAT(ChangedFields(copied, spec), IsEmpty());

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(copied);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
}

// a pose is written in the frame which the element is in; an orientation which
// was not changed stays as it was written
TEST_F(CopyBackTest, WritesPoses) {
  static constexpr char cube[] = R"(
  v 0 0 0
  v 1 0 0
  v 0 2 0
  v 1 2 0
  v 0 0 4
  v 1 0 4
  v 0 2 4
  v 1 2 4)";
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree"/>
    <asset>
      <mesh name="cube" file="cube.obj"/>
    </asset>
    <worldbody>
      <frame pos="1 0 0" euler="0 0 90">
        <body name="moved" pos="0 1 0" euler="90 0 0">
          <joint name="hinge" pos=".25 0 0" axis="0 0 2"/>
          <frame name="inner" pos="0 0 .5" euler="0 90 0">
            <geom name="turned" size=".125" pos=".25 0 0" euler="0 0 45"/>
            <geom name="mesh" type="mesh" mesh="cube" pos="0 .5 0" euler="0 0 45"/>
            <site name="site" pos="0 .25 0" euler="0 45 0"/>
            <camera name="camera" pos="0 0 1" euler="45 0 0"/>
            <light name="light" pos="0 0 2" dir="0 0 -1"/>
            <joint name="slide" type="slide" pos="0 0 .25" axis="1 0 0"/>
          </frame>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);

  // move everything, and turn the first geom, the light and the hinge
  const mjtNum turn[4] = {0.8, 0, 0.6, 0};
  mjtNum turned[4];
  for (mjtNum* pos : {model->body_pos + 3, model->geom_pos, model->geom_pos + 3,
                      model->site_pos, model->cam_pos, model->light_pos,
                      model->jnt_pos, model->jnt_pos + 3}) {
    pos[0] += 0.5;
    pos[1] -= 0.25;
  }
  mju_mulQuat(turned, model->geom_quat, turn);
  mju_copy4(model->geom_quat, turned);
  mju_rotVecQuat(model->light_dir, model->light_dir, turn);
  mju_rotVecQuat(model->jnt_axis, model->jnt_axis, turn);
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  // only the orientation which was changed is now a quaternion
  auto orientation = [&](mjtObj type, const char* name) {
    mjsElement* element = mjs_findElement(spec, type, name);
    return type == mjOBJ_BODY   ? mjs_asBody(element)->alt.type
           : type == mjOBJ_GEOM ? mjs_asGeom(element)->alt.type
           : type == mjOBJ_SITE ? mjs_asSite(element)->alt.type
                                : mjs_asCamera(element)->alt.type;
  };
  EXPECT_EQ(orientation(mjOBJ_GEOM, "turned"), mjORIENTATION_QUAT);
  EXPECT_EQ(orientation(mjOBJ_BODY, "moved"), mjORIENTATION_EULER);
  EXPECT_EQ(orientation(mjOBJ_GEOM, "mesh"), mjORIENTATION_EULER);
  EXPECT_EQ(orientation(mjOBJ_SITE, "site"), mjORIENTATION_EULER);
  EXPECT_EQ(orientation(mjOBJ_CAMERA, "camera"), mjORIENTATION_EULER);
  for (const std::string& changed : ChangedFields(authored, spec)) {
    EXPECT_THAT(changed,
                testing::AnyOf(HasSubstr(": pos["), HasSubstr(": dir["),
                               HasSubstr("'hinge': axis["),
                               HasSubstr("'turned': quat["),
                               HasSubstr("'turned': alt.")));
  }

  // the body moved in the frame it is in: by (-.25, -.5, 0) there
  mjsBody* moved = mjs_findBody(spec, "moved");
  EXPECT_NEAR(moved->pos[0], -0.25, 1e-12);
  EXPECT_NEAR(moved->pos[1], 0.5, 1e-12);
  EXPECT_NEAR(moved->pos[2], 0, 1e-12);

  // the model which is compiled from the spec has the poses of the changed one
  mjModel* again = mj_compile(spec, vfs.get());
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(DifferentArrays(
                  model, again,
                  {&mjModel::body_pos, &mjModel::body_quat, &mjModel::geom_pos,
                   &mjModel::geom_quat, &mjModel::site_pos, &mjModel::site_quat,
                   &mjModel::cam_pos, &mjModel::cam_quat, &mjModel::light_pos,
                   &mjModel::light_dir, &mjModel::jnt_pos, &mjModel::jnt_axis},
                  {6, 8, 6, 8, 3, 4, 3, 4, 3, 3, 6, 6}),
              IsEmpty());

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// a position and an orientation are written each if the model changed it, so
// that what was written in the spec since stays; a mesh whose frame is offset
// offsets the position by the orientation, which then gives both. Any other
// attribute is written whole
TEST_F(CopyBackTest, WritesPositionAndOrientationApart) {
  static constexpr char cube[] = R"(
  v 0 0 0
  v 1 0 0
  v 0 2 0
  v 1 2 0
  v 0 0 4
  v 1 0 4
  v 0 2 4
  v 1 2 4)";
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" file="cube.obj"/>
    </asset>
    <worldbody>
      <body name="body">
        <geom name="geom" size=".25" friction=".5 .25 .125"/>
        <geom name="mesh" type="mesh" mesh="cube"/>
        <site name="site"/>
        <camera name="camera"/>
        <light name="light" dir="0 0 -1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // positions and a friction coefficient written in the spec since
  mjsBody* body = mjs_findBody(spec, "body");
  mjsGeom* geom = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "geom"));
  mjsGeom* mesh = mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, "mesh"));
  mjsSite* site = mjs_asSite(mjs_findElement(spec, mjOBJ_SITE, "site"));
  mjsCamera* camera =
      mjs_asCamera(mjs_findElement(spec, mjOBJ_CAMERA, "camera"));
  mjsLight* light = mjs_asLight(mjs_findElement(spec, mjOBJ_LIGHT, "light"));
  for (double* pos :
       {body->pos, geom->pos, mesh->pos, site->pos, camera->pos, light->pos}) {
    pos[0] = 3;
  }
  geom->friction[1] = 0.0625;

  // orientations, a direction and another friction coefficient in the model
  const mjtNum turn[4] = {0.8, 0, 0.6, 0};
  for (mjtNum* quat :
       {model->body_quat + 4, model->geom_quat, model->geom_quat + 4,
        model->site_quat, model->cam_quat}) {
    mjtNum turned[4];
    mju_mulQuat(turned, quat, turn);
    mju_copy4(quat, turned);
  }
  mju_rotVecQuat(model->light_dir, model->light_dir, turn);
  model->geom_friction[0] = 2;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  // the new orientations, and the positions as they were written
  for (const double* pos :
       {body->pos, geom->pos, site->pos, camera->pos, light->pos}) {
    EXPECT_EQ(pos[0], 3);
  }
  for (auto [quat, compiled] :
       std::vector<std::pair<const double*, const mjtNum*>>{
           {body->quat, model->body_quat + 4},
           {geom->quat, model->geom_quat},
           {site->quat, model->site_quat},
           {camera->quat, model->cam_quat}}) {
    for (int i = 0; i < 4; i++) EXPECT_EQ(quat[i], compiled[i]);
  }
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(light->dir[i], model->light_dir[i], 1e-15);
  }

  // the position of the mesh geom follows its orientation
  EXPECT_NE(mesh->pos[0], 3);
  mjModel* again = mj_compile(spec, vfs.get());
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(again->geom_pos[3 + i], model->geom_pos[3 + i],
                MjTol(1e-12, 1e-5));
  }
  for (int i = 0; i < 4; i++) {
    EXPECT_NEAR(again->geom_quat[4 + i], model->geom_quat[4 + i],
                MjTol(1e-12, 1e-6));
  }

  // friction is one attribute: the model gives all of it
  EXPECT_EQ(geom->friction[0], 2);
  EXPECT_EQ(geom->friction[1], model->geom_friction[1]);

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// a size and pose which were written as fromto, or came from fitting the geom
// to a mesh, are written in its place
TEST_F(CopyBackTest, WritesShapes) {
  static constexpr char cube[] = R"(
  v 0 0 0
  v 1 0 0
  v 0 2 0
  v 1 2 0
  v 0 0 4
  v 1 0 4
  v 0 2 4
  v 1 2 4)";
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="cube" file="cube.obj"/>
    </asset>
    <worldbody>
      <geom name="thicker" type="capsule" fromto="0 0 0 0 0 1" size=".125"/>
      <geom name="longer" type="capsule" fromto="1 0 0 1 0 1" size=".125"/>
      <geom name="fitted" type="box" mesh="cube" pos="2 0 0"/>
      <geom name="mesh" type="mesh" mesh="cube" pos="4 0 0"/>
      <site name="longer" type="cylinder" fromto="0 1 0 0 1 1" size=".125"/>
    </worldbody>
  </mujoco>
  )";
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "cube.obj", cube, sizeof(cube));

  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);

  model->geom_size[3 * 0] = 0.25;      // the radius, which fromto does not give
  model->geom_size[3 * 1 + 1] = 0.75;  // the length, which it does
  model->geom_size[3 * 2] *= 2;
  model->site_size[1] = 0.75;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  auto geom = [&](const char* name) {
    return mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, name));
  };
  EXPECT_TRUE(std::isnan(geom("longer")->fromto[0]));
  EXPECT_EQ(geom("thicker")->fromto[5], 1);
  EXPECT_EQ(geom("thicker")->size[0], 0.25);
  EXPECT_EQ(geom("longer")->size[1], 0.75);
  EXPECT_EQ(geom("longer")->pos[2], 0.5);
  EXPECT_STREQ(mjs_getString(geom("fitted")->meshname), "");
  EXPECT_STREQ(mjs_getString(geom("mesh")->meshname), "cube");
  for (const std::string& changed : ChangedFields(authored, spec)) {
    EXPECT_THAT(changed, testing::Not(HasSubstr("'mesh'")));
    EXPECT_THAT(changed, testing::Not(HasSubstr("'thicker': fromto")));
    EXPECT_THAT(changed, testing::Not(HasSubstr("'thicker': pos")));
  }

  // the model which is compiled from the spec has the sizes and poses of the
  // changed one
  mjModel* again = mj_compile(spec, vfs.get());
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(DifferentArrays(model, again,
                              {&mjModel::geom_size, &mjModel::geom_pos,
                               &mjModel::geom_quat, &mjModel::site_size,
                               &mjModel::site_pos, &mjModel::site_quat},
                              {12, 12, 16, 3, 3, 4}),
              IsEmpty());

  // the size of a mesh geom is that of its mesh
  model->geom_size[3 * 3] *= 2;
  mjSpec* before = mj_copySpec(spec);
  EXPECT_EQ(mj_copyBack(spec, model), 0);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("the size of a mesh geom"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("'mesh'"));
  EXPECT_THAT(ChangedFields(before, spec), IsEmpty());

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(before);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// a geom fitted to a mesh which has the material of the mesh keeps it when it
// no longer refers to the mesh
TEST_F(CopyBackTest, FittedGeomKeepsMaterialOfMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <material name="red" rgba="1 0 0 1"/>
      <material name="blue" rgba="0 0 1 1"/>
      <mesh name="cube" material="red"
            vertex="0 0 0  1 0 0  0 2 0  1 2 0  0 0 4  1 0 4  0 2 4  1 2 4"/>
    </asset>
    <worldbody>
      <geom name="inherited" type="box" mesh="cube"/>
      <geom name="own" type="box" mesh="cube" material="blue" pos="2 0 0"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  int red = mj_name2id(model, mjOBJ_MATERIAL, "red");
  int blue = mj_name2id(model, mjOBJ_MATERIAL, "blue");
  ASSERT_EQ(model->geom_matid[0], red);
  ASSERT_EQ(model->geom_matid[1], blue);

  model->geom_size[3 * 0] *= 2;
  model->geom_size[3 * 1] *= 2;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  auto geom = [&](const char* name) {
    return mjs_asGeom(mjs_findElement(spec, mjOBJ_GEOM, name));
  };
  EXPECT_STREQ(mjs_getString(geom("inherited")->meshname), "");
  EXPECT_STREQ(mjs_getString(geom("inherited")->material), "red");
  EXPECT_STREQ(mjs_getString(geom("own")->material), "blue");

  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(again->geom_matid[0], red);
  EXPECT_EQ(again->geom_matid[1], blue);
  EXPECT_THAT(DifferentArrays(model, again, {&mjModel::geom_size}, {6}),
              IsEmpty());

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a mass or inertia which was changed gives the body an inertial of its own
TEST_F(CopyBackTest, WritesInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiafromgeom="%s" %s/>
    <worldbody>
      <body name="inferred">
        <joint/>
        <geom name="ball" size=".5" pos="1 0 0"/>
      </body>
      <body name="given" pos="0 2 0">
        <joint/>
        <inertial pos="0 0 .5" mass="2" fullinertia="2 2 1 .25 0 0"/>
        <geom size=".5"/>
      </body>
      <body name="aligned" pos="0 4 0">
        <freejoint align="true"/>
        <geom type="box" size=".5 .25 .125" pos="1 0 0" euler="0 0 30"/>
      </body>
      <body name="kept" pos="0 6 0">
        <joint/>
        <geom size=".5" pos="1 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> er;
  std::string auto_xml = absl::StrFormat(xml, "auto", "");
  mjSpec* spec = mj_parseXMLString(auto_xml.c_str(), 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);

  // the mass of every body but the last, and the inertia of the first
  for (int i = 1; i < 4; i++) model->body_mass[i] *= 2;
  for (int i = 0; i < 3; i++) model->body_inertia[3 + i] *= 2;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  // a body which infers its inertial now has one, as the model has it
  mjsBody* inferred = mjs_findBody(spec, "inferred");
  EXPECT_TRUE(inferred->explicitinertial);
  EXPECT_EQ(inferred->mass, static_cast<double>(model->body_mass[1]));
  EXPECT_EQ(inferred->ipos[0], 1);

  // one which has an inertial keeps it as it was written, but for the mass
  mjsBody* given = mjs_findBody(spec, "given");
  EXPECT_EQ(given->mass, 4);
  EXPECT_EQ(given->fullinertia[3], 0.25);

  // what was not changed still follows its geoms
  EXPECT_FALSE(mjs_findBody(spec, "kept")->explicitinertial);
  for (const std::string& changed : ChangedFields(authored, spec)) {
    EXPECT_THAT(changed, testing::Not(HasSubstr("'kept'")));
    EXPECT_THAT(changed, testing::Not(HasSubstr("geom[")));
  }

  // the model which is compiled from the spec has the inertials of the changed
  // one, and the geoms where they were
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(DifferentArrays(model, again,
                              {&mjModel::body_mass, &mjModel::body_inertia,
                               &mjModel::body_ipos, &mjModel::body_iquat,
                               &mjModel::body_pos, &mjModel::body_quat,
                               &mjModel::geom_pos, &mjModel::geom_quat},
                              {5, 15, 15, 20, 15, 20, 12, 16}),
              IsEmpty());

  // the frame of a body which is aligned with its free joint is the inertial
  // frame: moving the one would move the other
  model->body_ipos[3 * 3] = 0.5;
  EXPECT_EQ(mj_copyBack(spec, model), 0);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("aligned with its free joint"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("'aligned'"));
  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);

  // an inertia which is inferred whatever is given, or scaled with all the
  // others, cannot be given: nothing is copied
  mock_warning_handler.ExpectWarnings("settotalmass");
  for (const auto& [inertiafromgeom, totalmass, reason] :
       {std::array<const char*, 3>{"true", "", "inertiafromgeom"},
        std::array<const char*, 3>{"auto", "settotalmass='8'",
                                   "settotalmass"}}) {
    std::string fixed_xml = absl::StrFormat(xml, inertiafromgeom, totalmass);
    spec = mj_parseXMLString(fixed_xml.c_str(), 0, er.data(), er.size());
    ASSERT_THAT(spec, NotNull()) << er.data();
    model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    authored = mj_copySpec(spec);

    model->geom_friction[0] = 0.25;
    ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
    model->geom_friction[0] = 0.75;
    model->body_mass[1] *= 2;
    mjSpec* before = mj_copySpec(spec);
    EXPECT_EQ(mj_copyBack(spec, model), 0);
    EXPECT_THAT(mjs_getError(spec), HasSubstr(reason));
    EXPECT_THAT(mjs_getError(spec), HasSubstr("'inferred'"));
    EXPECT_THAT(ChangedFields(before, spec), IsEmpty());
    EXPECT_THAT(ChangedFields(authored, spec),
                ElementsAre("geom[0] 'ball': friction[0]"));

    mj_deleteModel(model);
    mj_deleteSpec(before);
    mj_deleteSpec(authored);
    mj_deleteSpec(spec);
  }
}

// a range is written in the unit of the spec, and what it is a range of keeps
// the limited state which it has in the model
TEST_F(CopyBackTest, WritesRanges) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="degree" autolimits="%s"/>
    <worldbody>
      <body>
        <joint name="limited" range="-45 45" %s/>
        <joint name="free" axis="0 1 0"/>
        <joint name="slide" type="slide" axis="1 0 0" ref="0.5"/>
        <geom size=".5"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="limited" joint="limited" ctrlrange="-1 1" %s/>
      <motor name="free" joint="free"/>
    </actuator>
  </mujoco>
  )";
  for (bool autolimits : {true, false}) {
    std::string limits_xml = absl::StrFormat(
        xml, autolimits ? "true" : "false", autolimits ? "" : "limited='true'",
        autolimits ? "" : "ctrllimited='true'");
    std::array<char, 1000> er;
    mjSpec* spec =
        mj_parseXMLString(limits_xml.c_str(), 0, er.data(), er.size());
    ASSERT_THAT(spec, NotNull()) << er.data();
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    ASSERT_EQ(model->jnt_limited[0], 1);
    ASSERT_EQ(model->jnt_limited[1], 0);

    // give ranges to what is limited and to what is not, and a reference
    model->jnt_range[1] = 0.5;
    model->jnt_range[2] = -0.25;
    model->jnt_range[3] = 0.25;
    model->actuator_ctrlrange[1] = 2;
    model->actuator_ctrlrange[2] = -2;
    model->actuator_ctrlrange[3] = 2;
    model->qpos0[0] = 0.25;
    model->qpos0[2] = 0.75;
    ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

    // angles are in degrees in this spec, where the joint is limited
    auto joint = [&](const char* name) {
      return mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, name));
    };
    EXPECT_NEAR(joint("limited")->range[1], 0.5 * 180 / mjPI, 1e-12);
    EXPECT_NEAR(joint("limited")->ref, 0.25 * 180 / mjPI, 1e-12);
    EXPECT_EQ(joint("slide")->ref, 0.75);
    EXPECT_EQ(joint("free")->range[1], 0.25);
    EXPECT_EQ(joint("free")->limited, mjLIMITED_FALSE);
    EXPECT_EQ(joint("limited")->limited,
              autolimits ? mjLIMITED_AUTO : mjLIMITED_TRUE);

    // the model which is compiled from the spec has the ranges of the changed
    // one, which limit what they limited
    mjModel* again = mj_compile(spec, nullptr);
    ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
    EXPECT_THAT(DifferentArrays(model, again,
                                {&mjModel::jnt_range, &mjModel::qpos0,
                                 &mjModel::actuator_ctrlrange},
                                {6, 3, 4}),
                IsEmpty());
    EXPECT_THAT(AsVector(again->jnt_limited, 3), ElementsAre(1, 0, 0));
    EXPECT_THAT(AsVector(again->actuator_ctrllimited, 2), ElementsAre(1, 0));

    mj_deleteModel(again);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

static constexpr char kCopyBackToCopy[] = R"(
<mujoco>
  <worldbody>
    <body name="arm">
      <joint name="hinge" range="-45 45"/>
      <geom name="geom" size=".25"/>
    </body>
    <body name="other" pos="1 0 0">
      <joint/>
      <geom size=".25"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="arm" body2="other"/>
  </equality>
  <actuator>
    <position name="servo" joint="hinge" kp="8" dampratio="1"/>
  </actuator>
</mujoco>
)";

// a copy of a compiled spec holds what the compilation gave the model, so it
// takes what was changed in the model as the original does
TEST_F(CopyBackTest, CopiesBackToCopy) {
  std::array<char, 1000> er;
  mjSpec* spec =
      mj_parseXMLString(kCopyBackToCopy, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // nothing was changed in the model: the copy stays as it is
  mjSpec* copy = mj_copySpec(spec);
  mjSpec* before = mj_copySpec(copy);
  ASSERT_EQ(mj_copyBack(copy, model), 1) << mjs_getError(copy);
  EXPECT_THAT(CompareSpec(before, copy, 1000), IsEmpty());

  // a change is copied to the copy, and nothing else is
  model->geom_size[0] = 0.5;
  ASSERT_EQ(mj_copyBack(copy, model), 1) << mjs_getError(copy);
  EXPECT_THAT(ChangedFields(before, copy),
              ElementsAre("geom[0] 'geom': size[0]"));

  mj_deleteSpec(before);
  mj_deleteSpec(copy);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// an element of a copy whose references are not found as the compilation named
// them is taken again from its spec: the copy does not hold what the model was
// given, so nothing can be told to have changed
TEST_F(CopyBackTest, RefusesCopyWhichLostCompiledValues) {
  std::array<char, 1000> er;
  mjSpec* spec =
      mj_parseXMLString(kCopyBackToCopy, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // the weld refers to a body which is renamed after the compilation
  mjs_setName(mjs_findBody(spec, "other")->element, "renamed");
  mjsEquality* weld = mjs_asEquality(mjs_firstElement(spec, mjOBJ_EQUALITY));
  mjs_setString(weld->name2, "renamed");

  mjSpec* copy = mj_copySpec(spec);
  mjSpec* before = mj_copySpec(copy);
  EXPECT_EQ(mj_copyBack(copy, model), 0);
  EXPECT_THAT(mjs_getError(copy), HasSubstr("copy"));
  EXPECT_THAT(CompareSpec(before, copy, 1000), IsEmpty());

  // once the copy is compiled, it takes what was changed in a model of its
  // structure
  mjModel* compiled = mj_compile(copy, nullptr);
  ASSERT_THAT(compiled, NotNull()) << mjs_getError(copy);
  compiled->geom_size[0] = 0.5;
  ASSERT_EQ(mj_copyBack(copy, compiled), 1) << mjs_getError(copy);
  EXPECT_THAT(ChangedFields(before, copy),
              ElementsAre("geom[0] 'geom': size[0]"));

  mj_deleteModel(compiled);
  mj_deleteSpec(before);
  mj_deleteSpec(copy);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// keyframes, custom data and the elevation data of a height field
TEST_F(CopyBackTest, WritesVectors) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="given" nrow="2" ncol="3" size="1 1 1 1" elevation="0 2 4 4 2 0"/>
    </asset>
    <worldbody>
      <geom type="hfield" hfield="given"/>
      <body name="free" pos="0 0 2">
        <freejoint/>
        <geom size=".5"/>
      </body>
    </worldbody>
    <custom>
      <numeric name="numeric" data="1 2 3"/>
    </custom>
    <keyframe>
      <key name="rest"/>
      <key name="up" qpos="0 0 4 1 0 0 0"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* authored = mj_copySpec(spec);

  model->key_qpos[2] = 8;
  model->key_qvel[6 + 1] = 0.5;
  model->numeric_data[1] = 5;
  model->hfield_data[1] = 0.25f;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);

  // a keyframe vector which was changed is given in full
  mjsKey* rest = mjs_asKey(mjs_findElement(spec, mjOBJ_KEY, "rest"));
  mjsKey* up = mjs_asKey(mjs_findElement(spec, mjOBJ_KEY, "up"));
  EXPECT_THAT(*rest->qpos, ElementsAre(0, 0, 8, 1, 0, 0, 0));
  EXPECT_THAT(*up->qvel, ElementsAre(0, 0.5, 0, 0, 0, 0));
  EXPECT_THAT(*up->qpos, ElementsAre(0, 0, 4, 1, 0, 0, 0));
  mjsNumeric* numeric =
      mjs_asNumeric(mjs_findElement(spec, mjOBJ_NUMERIC, "numeric"));
  EXPECT_THAT(*numeric->data, ElementsAre(1, 5, 3));

  // elevation data is given as the model holds it, scaled to [0, 1]
  mjsHField* hfield =
      mjs_asHField(mjs_findElement(spec, mjOBJ_HFIELD, "given"));
  EXPECT_THAT(*hfield->userdata, ElementsAre(1, 0.25f, 0, 0, 0.5f, 1));

  // the model which is compiled from the spec is the changed one
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  std::string field;
  EXPECT_LE(CompareModel(model, again, field), MjTol(1e-13, 1e-5)) << field;

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(authored);
  mj_deleteSpec(spec);
}

// a spring length which was changed is written, in place of one which
// compilation computed
TEST_F(CopyBackTest, WritesSpringLength) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slider" type="slide"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed stiffness="1">
        <joint joint="slider" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjsTendon* tendon = mjs_asTendon(mjs_firstElement(spec, mjOBJ_TENDON));

  // unchanged, the computed length stays computed
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_EQ(tendon->springlength[0], -1);
  EXPECT_EQ(tendon->springlength[1], -1);

  model->tendon_lengthspring[0] = 0.25;
  model->tendon_lengthspring[1] = 0.5;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_EQ(tendon->springlength[0], 0.25);
  EXPECT_EQ(tendon->springlength[1], 0.5);
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(again->tendon_lengthspring[0], 0.25);
  EXPECT_EQ(again->tendon_lengthspring[1], 0.5);

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// compilation scales elevation data so that its lowest value is 0 and its
// highest is 1: data which it would scale again cannot be copied back
TEST_F(CopyBackTest, RefusesHeightFieldWhichIsScaledAgain) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="terrain" nrow="2" ncol="2" size="1 1 1 1" elevation="0 2 4 1"/>
    </asset>
    <worldbody>
      <geom type="hfield" hfield="terrain"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* before = mj_copySpec(spec);
  const std::vector<float> scaled = AsVector(model->hfield_data, 4);

  // between 0.25 and 0.75
  for (int i = 0; i < 4; i++) model->hfield_data[i] = 0.25f + 0.5f * scaled[i];
  EXPECT_EQ(mj_copyBack(spec, model), 0);
  EXPECT_THAT(mjs_getError(spec),
              HasSubstr("the elevation data of a height field"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("'terrain'"));
  EXPECT_THAT(ChangedFields(before, spec), IsEmpty());

  // other data between 0 and 1 is copied, and is what the spec compiles to
  for (int i = 0; i < 4; i++) model->hfield_data[i] = scaled[i] * scaled[i];
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(AsVector(again->hfield_data, 4), AsVector(model->hfield_data, 4));

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(before);
  mj_deleteSpec(spec);
}

static constexpr char kCopyBackEquality[] = R"(
<mujoco>
  <worldbody>
    <body name="first" pos="0 0 1">
      <freejoint/>
      <geom size=".25"/>
    </body>
    <body name="second" pos="1 0 1">
      <freejoint/>
      <geom size=".25"/>
    </body>
  </worldbody>
  <equality>
    <connect name="pin" body1="first" body2="second" anchor="0 0 0"/>
    <weld name="weld" body1="first" body2="second"/>
  </equality>
</mujoco>
)";

// a connect between bodies is given its anchor in the first body, and
// compilation computes the one in the second body: a change to that one which
// does not follow from the model cannot be copied back
TEST_F(CopyBackTest, RefusesComputedAnchorOfConnect) {
  std::array<char, 1000> er;
  mjSpec* spec =
      mj_parseXMLString(kCopyBackEquality, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* before = mj_copySpec(spec);
  ASSERT_EQ(model->eq_data[3], -1);

  model->eq_data[3] = 0.5;
  EXPECT_EQ(mj_copyBack(spec, model), 0);
  EXPECT_THAT(mjs_getError(spec),
              HasSubstr("the anchor of a connect in its second body"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("'pin'"));
  EXPECT_THAT(ChangedFields(before, spec), IsEmpty());
  model->eq_data[3] = -1;

  // the anchor in the first body is copied, and the other one follows from it
  model->eq_data[0] = 0.25;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(ChangedFields(before, spec),
              ElementsAre("equality[0] 'pin': data[0]"));
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(again->eq_data[0], 0.25);
  EXPECT_EQ(again->eq_data[3], -0.75);

  // one which mj_setConst computed, after a body was moved, is accepted
  mjData* data = mj_makeData(again);
  again->qpos0[7] = 2;
  mj_setConst(again, data);
  ASSERT_EQ(again->eq_data[3], -1.75);
  ASSERT_EQ(mj_copyBack(spec, again), 1) << mjs_getError(spec);
  EXPECT_EQ(mjs_findBody(spec, "second")->pos[0], 2);
  mjModel* moved = mj_compile(spec, nullptr);
  ASSERT_THAT(moved, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(moved->eq_data[3], -1.75);

  mj_deleteModel(moved);
  mj_deleteData(data);
  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(before);
  mj_deleteSpec(spec);
}

// a weld between bodies which has no relative pose is marked for mj_setConst
// to compute it: a change to the pose which does not follow from the model
// cannot be copied back, and a weld which is no longer marked has it written
TEST_F(CopyBackTest, WritesPoseOfUnmarkedWeld) {
  std::array<char, 1000> er;
  mjSpec* spec =
      mj_parseXMLString(kCopyBackEquality, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjData* data = mj_makeData(model);
  mjSpec* before = mj_copySpec(spec);
  mjtNum* weld = model->eq_data + mjNEQDATA;
  ASSERT_THAT(AsVector(weld, mjNEQDATA),
              ElementsAre(0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1));

  // the torque scale
  weld[10] = 2;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(ChangedFields(before, spec),
              ElementsAre("equality[1] 'weld': data[10]"));

  // a pose which mj_setConst does not compute
  weld[3] = 0.5;
  EXPECT_EQ(mj_copyBack(spec, model), 0);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("the relative pose of a weld"));
  EXPECT_THAT(mjs_getError(spec), HasSubstr("'weld'"));
  weld[3] = 1;

  // the anchor, and the pose which mj_setConst computes from it
  weld[0] = 0.25;
  mj_setConst(model, data);
  ASSERT_EQ(weld[3], 1.25);
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(ChangedFields(before, spec),
              ElementsAre("equality[1] 'weld': data[0]",
                          "equality[1] 'weld': data[10]"));
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(AsVector(again->eq_data + mjNEQDATA, mjNEQDATA),
            AsVector(weld, mjNEQDATA));

  // no longer marked: the pose is given
  weld[11] = 0;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(
      ChangedFields(before, spec),
      ElementsAre("equality[1] 'weld': data[0]", "equality[1] 'weld': data[3]",
                  "equality[1] 'weld': data[6]",
                  "equality[1] 'weld': data[10]"));
  mjModel* unmarked = mj_compile(spec, nullptr);
  ASSERT_THAT(unmarked, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(AsVector(unmarked->eq_data + mjNEQDATA, mjNEQDATA),
            AsVector(weld, mjNEQDATA));

  mj_deleteModel(unmarked);
  mj_deleteModel(again);
  mj_deleteData(data);
  mj_deleteModel(model);
  mj_deleteSpec(before);
  mj_deleteSpec(spec);
}

// the spec gives one value to all the degrees of freedom of a ball or a free
// joint: values which differ between them cannot be copied back
TEST_F(CopyBackTest, RefusesValuesWhichDifferBetweenDofs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <geom size=".25"/>
      </body>
      <body pos="1 0 0">
        <freejoint name="free"/>
        <geom size=".25"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mjSpec* before = mj_copySpec(spec);

  struct Case {
    mjtNum* value;
    mjtNum changed;
    const char* what;
    const char* joint;
  };
  const Case cases[] = {
      {model->dof_damping + 1, 2, "the damping of a joint", "'ball'"},
      {model->dof_dampingpoly + mjNPOLY * 2, 2, "the damping of a joint",
       "'ball'"},
      {model->dof_armature + 3 + 5, 2, "the armature of a joint", "'free'"},
      {model->dof_frictionloss + 3 + 1, 2, "the friction loss of a joint",
       "'free'"},
      {model->dof_solref + mjNREF * 1, 0.5,
       "the solver parameters of the friction loss of a joint", "'ball'"},
      {model->dof_solimp + mjNIMP * 2 + 1, 0.5,
       "the solver parameters of the friction loss of a joint", "'ball'"},
  };
  for (const Case& c : cases) {
    const mjtNum compiled = *c.value;
    *c.value = c.changed;
    EXPECT_EQ(mj_copyBack(spec, model), 0) << c.what;
    EXPECT_THAT(mjs_getError(spec), HasSubstr(c.what));
    EXPECT_THAT(mjs_getError(spec), HasSubstr(c.joint));
    EXPECT_THAT(ChangedFields(before, spec), IsEmpty());
    *c.value = compiled;
  }

  // the same value in all of them is the value of the joint
  for (int i = 0; i < 3; i++) model->dof_damping[i] = 2;
  for (int i = 3; i < 9; i++) model->dof_armature[i] = 0.5;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  EXPECT_THAT(
      ChangedFields(before, spec),
      ElementsAre("joint[0] 'ball': damping[0]", "joint[1] 'free': armature"));
  mjModel* again = mj_compile(spec, nullptr);
  ASSERT_THAT(again, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(AsVector(again->dof_damping, 9), AsVector(model->dof_damping, 9));
  EXPECT_EQ(AsVector(again->dof_armature, 9), AsVector(model->dof_armature, 9));

  mj_deleteModel(again);
  mj_deleteModel(model);
  mj_deleteSpec(before);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, AttachEmptyOrNullPrefixWarning) {
  mock_warning_handler.ExpectWarnings();

  for (const char* prefix : {"", static_cast<const char*>(nullptr)}) {
    mjSpec* parent = mj_makeSpec();
    mjSpec* child = mj_makeSpec();
    mjsBody* world = mjs_findBody(parent, "world");

    mjsElement* attached =
        mjs_attach(world->element, child->element, prefix, "");
    ASSERT_THAT(attached, NotNull());
    EXPECT_EQ(mjs_numWarnings(parent), 1);
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("may prevent the model from being saved correctly"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }
}
// a passive plugin whose init fails for the given mjData
static mjData* fail_init_data = nullptr;
static void RegisterFailInitPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.fail_init";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    return d == fail_init_data ? -1 : 0;
  };
  plugin.compute =
      +[](const mjModel* m, mjData* d, int instance, int capability_bit) {};
  mjp_registerPlugin(&plugin);
}

// a plugin init which fails when mj_recompile remakes the caller's mjData is a
// recompile error, and m and d are deleted, as when compilation fails
TEST_F(MujocoTest, RecompilePluginInitFailure) {
  RegisterFailInitPlugin();
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.test.fail_init">
        <instance name="instance"/>
      </plugin>
    </extension>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());

  fail_init_data = d;
  EXPECT_NE(mj_recompile(spec, nullptr, m, d), 0);
  fail_init_data = nullptr;
  EXPECT_THAT(mjs_getError(spec), HasSubstr("plugin init failed"));
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, KeyframePlasticExpandAttachScale) {
  // 1. ExpandKeyframe: adding a plastic joint to a spec with an existing key
  static constexpr char expand_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b1">
        <joint name="j1" type="slide" stiffness="10" springyield="0.1" springref="0.2"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k0" plastic="0.5"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1000> er;
  mjSpec* spec = mj_parseXMLString(expand_xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();

  mjsBody* world = mjs_findBody(spec, "world");
  mjsBody* b2 = mjs_addBody(world, 0);
  mjsJoint* j2 = mjs_addJoint(b2, 0);
  j2->type = mjJNT_SLIDE;
  j2->stiffness[0] = 10;
  j2->springyield = 0.1;
  j2->springref = 0.3;
  mjsGeom* g2 = mjs_addGeom(b2, 0);
  g2->size[0] = 0.1;

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nplastic, 2);
  EXPECT_NEAR(m->key_plastic[0], 0.5, 1e-6);
  EXPECT_NEAR(m->key_plastic[1], 0.3, 1e-6);
  mj_deleteModel(m);
  mj_deleteSpec(spec);

  // 2. StoreKeyframes (attach): attaching a child spec with a plastic keyframe
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="p">
        <joint name="pj" type="slide" stiffness="10" springyield="0.1" springref="0.1"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" plastic="0.4"/>
    </keyframe>
  </mujoco>
  )";
  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="c">
        <joint name="cj" type="slide" stiffness="10" springyield="0.1" springref="0.2"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="k" plastic="0.7"/>
    </keyframe>
  </mujoco>
  )";
  mjSpec* parent = mj_parseXMLString(parent_xml, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(child_xml, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();
  mjsBody* pworld = mjs_findBody(parent, "world");
  ASSERT_THAT(mjs_attach(pworld->element, child->element, "c_", ""), NotNull());

  m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);
  ASSERT_EQ(m->nplastic, 2);
  ASSERT_EQ(m->nkey, 2);
  // key 0 ("k"): parent plastic = 0.4, child default = 0.2
  EXPECT_NEAR(m->key_plastic[0], 0.4, 1e-6);
  EXPECT_NEAR(m->key_plastic[1], 0.2, 1e-6);
  // key 1 ("c_k"): parent default = 0.1, child plastic = 0.7
  EXPECT_NEAR(m->key_plastic[2], 0.1, 1e-6);
  EXPECT_NEAR(m->key_plastic[3], 0.7, 1e-6);
  mj_deleteModel(m);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);

  // 3. ScaleKeyframe: scaling a body with a slide joint and tendon plastic key
  static constexpr char scale_xml[] = R"(
  <mujoco>
    <worldbody>
      <frame scale="2">
        <body name="b">
          <joint name="j" type="slide" axis="1 0 0" stiffness="10" springyield="0.1"/>
          <geom size="0.1"/>
          <site name="s1" pos="0 0 0"/>
        </body>
        <site name="s0" pos="-1 0 0"/>
      </frame>
    </worldbody>
    <tendon>
      <spatial name="t" stiffness="10" springyield="0.1" springlength="1 1">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
    <keyframe>
      <key name="k" plastic="0.5 1.2 1.4"/>
    </keyframe>
  </mujoco>
  )";
  spec = mj_parseXMLString(scale_xml, 0, er.data(), er.size());
  ASSERT_THAT(spec, NotNull()) << er.data();
  m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  ASSERT_EQ(m->nplastic, 3);
  EXPECT_NEAR(m->key_plastic[0], 1.0, 1e-6);
  EXPECT_NEAR(m->key_plastic[1], 2.4, 1e-6);
  EXPECT_NEAR(m->key_plastic[2], 2.8, 1e-6);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, FindChildDoesNotReturnSelf) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <body name="leaf">
          <geom name="leaf" size="0.1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> err;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();

  mjsBody* parent = mjs_findBody(spec, "parent");
  ASSERT_THAT(parent, NotNull());
  mjsBody* leaf = mjs_findChild(parent, "leaf");
  ASSERT_THAT(leaf, NotNull());

  EXPECT_THAT(mjs_findChild(leaf, "leaf"), IsNull());
  EXPECT_THAT(mjs_findChild(parent, "parent"), IsNull());

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, SetToMuscleNullArrayPointers) {
  mjSpec* spec = mj_makeSpec();
  ASSERT_THAT(spec, NotNull());
  mjsActuator* act = mjs_addActuator(spec, nullptr);
  ASSERT_THAT(act, NotNull());

  mjs_setToMuscle(act, /*timeconst=*/nullptr, /*tausmooth=*/0,
                  /*range=*/nullptr, /*force=*/-1, /*scale=*/-1, /*lmin=*/-1,
                  /*lmax=*/-1, /*vmax=*/-1, /*fpmax=*/-1, /*fvmax=*/-1);
  EXPECT_EQ(act->dyntype, mjDYN_MUSCLE);
  EXPECT_EQ(act->gaintype, mjGAIN_MUSCLE);
  EXPECT_EQ(act->biastype, mjBIAS_MUSCLE);
  EXPECT_DOUBLE_EQ(act->dynprm[0], 0.01);
  EXPECT_DOUBLE_EQ(act->dynprm[1], 0.04);
  EXPECT_DOUBLE_EQ(act->gainprm[0], 0.75);
  EXPECT_DOUBLE_EQ(act->gainprm[1], 1.05);

  mj_deleteSpec(spec);
}

TEST_F(MujocoTest, EncodeMjbContentType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1000> err;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, err.data(), err.size());
  ASSERT_THAT(spec, NotNull()) << err.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  const std::string out_path =
      (std::filesystem::path(testing::TempDir()) / "encoded_mjb_no_ext")
          .string();
  mjtSize written = mj_encode(spec, m, out_path.c_str(), "model/vnd.mujoco.mjb",
                              nullptr, err.data(), err.size());
  EXPECT_GT(written, 0) << err.data();
  EXPECT_EQ(written, mj_sizeModel(m));

  std::filesystem::remove(out_path);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

}  // namespace
}  // namespace mujoco
