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

#include <array>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_api.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;

TEST_F(MujocoTest, AttachWarningsSurviveRecompile) {
  mock_warning_handler.ExpectWarnings();
  // attach warnings should persist through recompilation
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <option timestep="0.005"/>
    <worldbody>
      <frame name="attachment"/>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <option timestep="0.001"/>
    <worldbody>
      <body name="child">
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> er;
  mjSpec* parent = mj_parseXMLString(xml_parent, 0, er.data(), er.size());
  ASSERT_THAT(parent, NotNull()) << er.data();
  mjSpec* child = mj_parseXMLString(xml_child, 0, er.data(), er.size());
  ASSERT_THAT(child, NotNull()) << er.data();

  mjsFrame* frame = mjs_findFrame(parent, "attachment");
  ASSERT_THAT(frame, NotNull());
  mjsElement* attached = mjs_attach(
      frame->element, mjs_findBody(child, "child")->element, "child-", "");
  ASSERT_THAT(attached, NotNull());

  // attach warning should be present
  int attach_warnings = mjs_numWarnings(parent);
  EXPECT_GE(attach_warnings, 1);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));

  // compile
  mjModel* m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);

  // attach warnings should survive compilation
  EXPECT_GE(mjs_numWarnings(parent), attach_warnings);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));

  // recompile
  mj_deleteModel(m);
  m = mj_compile(parent, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(parent);

  // attach warnings should still be there after recompile
  EXPECT_GE(mjs_numWarnings(parent), attach_warnings);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));

  mj_deleteModel(m);
  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictWarningDefault) {
  mock_warning_handler.ExpectWarnings();
  // default mode (warning): conflicting values should keep parent, set warning

  mjSpec* parent = mj_makeSpec();
  parent->option.timestep = 0.005;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // parent value should be unchanged
  EXPECT_EQ(parent->option.timestep, mjtNum(0.005));

  // one grouped warning per attach
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  std::string w = mjs_getWarning(parent, 0);
  EXPECT_THAT(w, HasSubstr("policy is 'warning'"));
  EXPECT_THAT(w, HasSubstr("timestep: parent has 0.005, child has 0.001,"
                           " keeping parent value"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictWarningNoConflict) {
  mock_warning_handler.ExpectWarnings();
  // warning mode: child has non-default, parent is default → warn + keep parent
  mjSpec* parent = mj_makeSpec();

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // warning produced when child is non-default
  EXPECT_TRUE(mjs_isWarning(parent));
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeMin) {
  mock_warning_handler.ExpectWarnings();
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.timestep = 0.005;
  parent->option.tolerance = 1e-10;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;
  child->option.tolerance = 1e-12;
  child->option.sleep_tolerance = 0.005;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // min-fields should take the smaller value
  EXPECT_EQ(parent->option.timestep, mjtNum(0.001));
  EXPECT_EQ(parent->option.tolerance, mjtNum(1e-12));
  // one side is default -> copy and warn
  EXPECT_EQ(parent->option.sleep_tolerance, mjtNum(0.005));

  // one grouped warning containing all three fields
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  std::string w = mjs_getWarning(parent, 0);
  EXPECT_THAT(w, HasSubstr("policy is 'merge'"));
  EXPECT_THAT(w, HasSubstr("timestep: parent has 0.005, child has 0.001,"
                           " taking the minimum"));
  EXPECT_THAT(w, HasSubstr("tolerance"));
  EXPECT_THAT(w, HasSubstr("sleep_tolerance"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeMax) {
  mock_warning_handler.ExpectWarnings();
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.iterations = 150;
  parent->nkey = 5;
  parent->nuser_body = 2;

  mjSpec* child = mj_makeSpec();
  child->option.iterations = 200;
  child->nkey = 8;
  child->nuser_body = 4;

  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);

  mjsElement* attached =
      mjs_attach(frame->element, child_body->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // max-fields should take the larger value
  EXPECT_EQ(parent->option.iterations, 200);
  EXPECT_EQ(parent->nkey, 8);
  EXPECT_EQ(parent->nuser_body, 4);

  // one grouped warning containing all three fields
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  std::string w = mjs_getWarning(parent, 0);
  EXPECT_THAT(w, HasSubstr("iterations: parent has 150, child has 200,"
                           " taking the maximum"));
  EXPECT_THAT(w, HasSubstr("nkey"));
  EXPECT_THAT(w, HasSubstr("nuser_body"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeError) {
  // merge mode: conflicting error-fields should produce an error
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.integrator = mjINT_RK4;

  mjSpec* child = mj_makeSpec();
  child->option.integrator = mjINT_IMPLICIT;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached = mjs_attach(world->element, child->element, "", "");

  // error-fields should fail on conflict
  EXPECT_THAT(attached, IsNull());
  EXPECT_THAT(mjs_getError(parent), HasSubstr("integrator: parent has"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictErrorMode) {
  // error mode: any conflict -> error
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_ERROR;
  parent->option.timestep = 0.005;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached = mjs_attach(world->element, child->element, "", "");

  EXPECT_THAT(attached, IsNull());
  std::string error = mjs_getError(parent);
  EXPECT_THAT(error, HasSubstr("policy is 'error'"));
  EXPECT_THAT(error, HasSubstr("timestep: parent has 0.005, child has 0.001"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictErrorNoConflict) {
  // error mode: no conflict (one side default) -> succeeds
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_ERROR;
  parent->option.timestep = 0.005;

  mjSpec* child = mj_makeSpec();
  // child timestep is default -> no conflict

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // parent unchanged
  EXPECT_EQ(parent->option.timestep, mjtNum(0.005));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeFromBody) {
  mock_warning_handler.ExpectWarnings();
  // merge mode: attach from body (not spec) still merges

  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.timestep = 0.005;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;
  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);

  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);

  mjsElement* attached =
      mjs_attach(frame->element, child_body->element, "", "");
  ASSERT_THAT(attached, NotNull());

  EXPECT_EQ(parent->option.timestep, mjtNum(0.001));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeGravityError) {
  // merge mode: conflicting gravity -> error
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.gravity[2] = -10.0;

  mjSpec* child = mj_makeSpec();
  child->option.gravity[2] = -1.62;

  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);

  mjs_attach(frame->element, child_body->element, "child_", "");

  EXPECT_THAT(mjs_getError(parent),
              HasSubstr("gravity: parent has 0 0 -10, child has 0 0 -1.62"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeRealtimeMin) {
  mock_warning_handler.ExpectWarnings();
  // merge mode: conflicting realtime -> take min

  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->visual.global.realtime = 2.0f;

  mjSpec* child = mj_makeSpec();
  child->visual.global.realtime = 0.5f;

  mjsBody* child_body = mjs_addBody(mjs_findBody(child, "world"), 0);
  mjsFrame* frame = mjs_addFrame(mjs_findBody(parent, "world"), nullptr);

  mjsElement* attached =
      mjs_attach(frame->element, child_body->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  EXPECT_EQ(parent->visual.global.realtime, 0.5f);

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictWarningBitfields) {
  mock_warning_handler.ExpectWarnings();
  // warning mode: conflicting disable flags report individual flag names

  mjSpec* parent = mj_makeSpec();
  parent->option.disableflags = mjDSBL_GRAVITY | mjDSBL_CONTACT;
  parent->authored.disableflags =
      mjDSBL_GRAVITY | mjDSBL_CONTACT | mjDSBL_ACTUATION;

  mjSpec* child = mj_makeSpec();
  child->option.disableflags = mjDSBL_GRAVITY | mjDSBL_ACTUATION;
  child->authored.disableflags =
      mjDSBL_GRAVITY | mjDSBL_CONTACT | mjDSBL_ACTUATION;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // Contact and Actuation differ, Gravity agrees -> 1 grouped warning
  EXPECT_TRUE(mjs_isWarning(parent));
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  std::string w = mjs_getWarning(parent, 0);
  EXPECT_THAT(w, HasSubstr("flag 'Contact': parent set, child unset,"
                           " keeping parent"));
  EXPECT_THAT(w, HasSubstr("flag 'Actuation': parent unset, child set,"
                           " keeping parent"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeMultipleWarnings) {
  mock_warning_handler.ExpectWarnings();
  // merge mode: multiple min/max conflicts should be in one grouped warning

  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.timestep = 0.005;
  parent->option.iterations = 150;
  parent->visual.global.realtime = 2.0f;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;        // min conflict
  child->option.iterations = 200;        // max conflict
  child->visual.global.realtime = 0.5f;  // min conflict

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // values should be merged
  EXPECT_EQ(parent->option.timestep, mjtNum(0.001));
  EXPECT_EQ(parent->option.iterations, 200);
  EXPECT_EQ(parent->visual.global.realtime, 0.5f);

  // one grouped warning containing all three fields
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  std::string w = mjs_getWarning(parent, 0);
  EXPECT_THAT(w, HasSubstr("policy is 'merge'"));
  EXPECT_THAT(w, HasSubstr("timestep: parent has 0.005, child has 0.001,"
                           " taking the minimum"));
  EXPECT_THAT(w, HasSubstr("iterations: parent has 150, child has 200,"
                           " taking the maximum"));
  EXPECT_THAT(w, HasSubstr("realtime"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictWarningBitfieldsNoConflict) {
  // warning mode: one side has zero disableflags -> no conflict
  mjSpec* parent = mj_makeSpec();
  parent->option.disableflags = mjDSBL_GRAVITY;

  mjSpec* child = mj_makeSpec();
  // child disableflags is 0 (default) -> no conflict

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());
  EXPECT_FALSE(mjs_isWarning(parent));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMergeBitfields) {
  mock_warning_handler.ExpectWarnings();
  // merge mode: flags are ORed together
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.disableflags = mjDSBL_GRAVITY | mjDSBL_CONTACT;
  parent->option.enableflags = mjENBL_OVERRIDE;

  mjSpec* child = mj_makeSpec();
  child->option.disableflags = mjDSBL_GRAVITY | mjDSBL_ACTUATION;
  child->option.enableflags = mjENBL_ENERGY;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // union of flags
  EXPECT_EQ(parent->option.disableflags,
            mjDSBL_GRAVITY | mjDSBL_CONTACT | mjDSBL_ACTUATION);
  EXPECT_EQ(parent->option.enableflags, mjENBL_OVERRIDE | mjENBL_ENERGY);

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictErrorBitfields) {
  // error mode: differing flags -> per-flag error
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_ERROR;
  parent->option.disableflags = mjDSBL_GRAVITY;
  parent->authored.disableflags = mjDSBL_GRAVITY | mjDSBL_CONTACT;

  mjSpec* child = mj_makeSpec();
  child->option.disableflags = mjDSBL_CONTACT;
  child->authored.disableflags = mjDSBL_GRAVITY | mjDSBL_CONTACT;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached = mjs_attach(world->element, child->element, "", "");

  EXPECT_THAT(attached, IsNull());
  // should mention the specific flag name, not "disableflags"
  std::string error = mjs_getError(parent);
  EXPECT_THAT(error, HasSubstr("flag 'Contact'"));
  EXPECT_THAT(error, HasSubstr("flag 'Gravity'"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictErrorNoMutation) {
  // failed attach should not mutate the parent spec
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_MERGE;
  parent->option.timestep = 0.005;
  parent->option.iterations = 150;
  parent->option.integrator = mjINT_RK4;
  parent->option.gravity[2] = -10.0;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;             // min-mergeable
  child->option.iterations = 200;             // max-mergeable
  child->option.integrator = mjINT_IMPLICIT;  // unmergeable -> error
  child->option.gravity[2] = -1.62;           // unmergeable array -> error

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached = mjs_attach(world->element, child->element, "", "");

  // attach should fail
  EXPECT_THAT(attached, IsNull());

  // parent should be completely unchanged
  EXPECT_EQ(parent->option.timestep, mjtNum(0.005));
  EXPECT_EQ(parent->option.iterations, 150);
  EXPECT_EQ(parent->option.integrator, mjINT_RK4);
  EXPECT_EQ(parent->option.gravity[2], mjtNum(-10.0));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictMultipleErrors) {
  // error mode: multiple conflicts should all be reported in the error message
  mjSpec* parent = mj_makeSpec();
  parent->compiler.conflict = mjCONFLICT_ERROR;
  parent->option.timestep = 0.005;
  parent->option.iterations = 150;
  parent->option.gravity[2] = -10.0;

  mjSpec* child = mj_makeSpec();
  child->option.timestep = 0.001;
  child->option.iterations = 200;
  child->option.gravity[2] = -1.62;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached = mjs_attach(world->element, child->element, "", "");

  EXPECT_THAT(attached, IsNull());
  // all three conflicting fields should appear in the error message
  std::string error = mjs_getError(parent);
  EXPECT_THAT(error, HasSubstr("timestep"));
  EXPECT_THAT(error, HasSubstr("iterations"));
  EXPECT_THAT(error, HasSubstr("gravity"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictWarningArray) {
  mock_warning_handler.ExpectWarnings();
  // warning mode: conflicting array -> keep parent, warn
  mjSpec* parent = mj_makeSpec();
  parent->option.gravity[2] = -10.0;

  mjSpec* child = mj_makeSpec();
  child->option.gravity[2] = -1.62;

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull());

  // parent gravity should be unchanged
  EXPECT_EQ(parent->option.gravity[2], mjtNum(-10.0));

  // one grouped warning containing gravity
  EXPECT_TRUE(mjs_isWarning(parent));
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0),
              HasSubstr("gravity: parent has 0 0 -10, child has 0 0 -1.62,"
                        " keeping parent value"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictSubjectNames) {
  mock_warning_handler.ExpectWarnings();

  // Test case 1: both custom names
  {
    mjSpec* parent = mj_makeSpec();
    mjs_setString(parent->modelname, "parent_model");
    parent->option.timestep = 0.005;

    mjSpec* child = mj_makeSpec();
    mjs_setString(child->modelname, "child_model");
    child->option.timestep = 0.001;

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    ASSERT_THAT(attached, NotNull());

    EXPECT_EQ(mjs_numWarnings(parent), 1);
    std::string w = mjs_getWarning(parent, 0);
    EXPECT_THAT(w, HasSubstr("Attach conflict when attaching 'child_model' to "
                             "'parent_model', policy is 'warning'"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // Test case 2: only child custom name
  {
    mjSpec* parent = mj_makeSpec();
    parent->option.timestep = 0.005;

    mjSpec* child = mj_makeSpec();
    mjs_setString(child->modelname, "child_model");
    child->option.timestep = 0.001;

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    ASSERT_THAT(attached, NotNull());

    EXPECT_EQ(mjs_numWarnings(parent), 1);
    std::string w = mjs_getWarning(parent, 0);
    EXPECT_THAT(w, HasSubstr("Attach conflict when attaching 'child_model', "
                             "policy is 'warning'"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // Test case 3: only parent custom name
  {
    mjSpec* parent = mj_makeSpec();
    mjs_setString(parent->modelname, "parent_model");
    parent->option.timestep = 0.005;

    mjSpec* child = mj_makeSpec();
    child->option.timestep = 0.001;

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    ASSERT_THAT(attached, NotNull());

    EXPECT_EQ(mjs_numWarnings(parent), 1);
    std::string w = mjs_getWarning(parent, 0);
    EXPECT_THAT(w, HasSubstr("Attach conflict when attaching to 'parent_model',"
                             " policy is 'warning'"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // Test case 4: no custom name (both "MuJoCo Model")
  {
    mjSpec* parent = mj_makeSpec();
    parent->option.timestep = 0.005;

    mjSpec* child = mj_makeSpec();
    child->option.timestep = 0.001;

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    ASSERT_THAT(attached, NotNull());

    EXPECT_EQ(mjs_numWarnings(parent), 1);
    std::string w = mjs_getWarning(parent, 0);
    EXPECT_THAT(w, HasSubstr("Attach conflict on attach, "
                             "policy is 'warning'"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }
}

TEST_F(MujocoTest, AttachConflictAuthoredDefault) {
  // child explicitly sets timestep to its default value (0.002) in XML,
  // which should still conflict with a non-default parent timestep
  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="error"/>
      <option timestep="0.005"/>
      <worldbody>
        <body name="parent_body"><geom size="1"/></body>
      </worldbody>
    </mujoco>
  )";

  static constexpr char child_xml[] = R"(
    <mujoco>
      <option timestep="0.002"/>
      <worldbody>
        <body name="child_body"><geom size="1"/></body>
      </worldbody>
    </mujoco>
  )";

  std::array<char, 1024> err;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();
  mjSpec* child = mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
  ASSERT_THAT(child, NotNull()) << err.data();

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");

  // attach should fail: child authored timestep=0.002 (the default),
  // but parent has timestep=0.005 — authored tracking catches this
  EXPECT_THAT(attached, IsNull());
  EXPECT_THAT(mjs_getError(parent), HasSubstr("timestep"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictNonAuthoredDefault) {
  // child does NOT set timestep in XML, parent has non-default timestep.
  // no conflict should be detected (child didn't author the field)
  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="error"/>
      <option timestep="0.005"/>
      <worldbody>
        <body name="parent_body"><geom size="1"/></body>
      </worldbody>
    </mujoco>
  )";

  static constexpr char child_xml[] = R"(
    <mujoco>
      <worldbody>
        <body name="child_body"><geom size="1"/></body>
      </worldbody>
    </mujoco>
  )";

  std::array<char, 1024> err;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
  ASSERT_THAT(parent, NotNull()) << err.data();
  mjSpec* child = mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
  ASSERT_THAT(child, NotNull()) << err.data();

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");

  // attach should succeed: child didn't author timestep
  EXPECT_THAT(attached, NotNull()) << mjs_getError(parent);
  // parent timestep should be unchanged
  EXPECT_EQ(parent->option.timestep, mjtNum(0.005));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictXMLMergeAuthoredDefault) {
  mock_warning_handler.ExpectWarnings();

  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="merge"/>
      <option timestep="0.005"/>
      <worldbody/>
    </mujoco>
  )";

  static constexpr char child_xml[] = R"(
    <mujoco>
      <option timestep="0.002"/>
      <worldbody/>
    </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull()) << error.data();

  mjSpec* child =
      mj_parseXMLString(child_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child, NotNull()) << error.data();

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull()) << "Error details: " << mjs_getError(parent);

  // Child explicitly authored timestep=0.002. Parent has timestep=0.005. So
  // they conflict on both-authored. Under merge mode, min-merge applies ->
  // child wins.
  EXPECT_EQ(parent->option.timestep, mjtNum(0.002));
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0),
              HasSubstr("timestep: parent has 0.005, child has 0.002, "
                        "taking the minimum"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictXMLBitfieldSilentAdoption) {
  mock_warning_handler.ExpectWarnings();

  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="merge"/>
      <worldbody/>
    </mujoco>
  )";

  static constexpr char child_xml[] = R"(
    <mujoco>
      <option>
        <flag constraint="disable"/>
      </option>
      <worldbody/>
    </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull()) << error.data();

  mjSpec* child =
      mj_parseXMLString(child_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child, NotNull()) << error.data();

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull()) << "Error details: " << mjs_getError(parent);

  // Parent should adopt child's constraint disable flag since parent didn't
  // restrict flags. In merge mode, this logging counts as 1 warning.
  EXPECT_EQ(parent->option.disableflags, mjDSBL_CONSTRAINT);
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0),
              HasSubstr("flag 'Constraint': added from child"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachWarningBoundaryAndPreservation) {
  mock_warning_handler.ExpectWarnings();

  // 1. Build a parent XML that generates a compile warning
  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="warning"/>
      <option timestep="0.005"/>
      <worldbody>
        <body name="parent">
          <flexcomp name="grid" type="grid" count="3 3 1" spacing="0.1 0.1 0.1"
                    dim="2" radius="0.01">
          </flexcomp>
        </body>
      </worldbody>
    </mujoco>
  )";

  // 2. Child 1 has conflicting timestep (0.001)
  static constexpr char child1_xml[] = R"(
    <mujoco>
      <option timestep="0.001"/>
      <worldbody/>
    </mujoco>
  )";

  // 3. Child 2 has conflicting sleep_tolerance (0.005)
  static constexpr char child2_xml[] = R"(
    <mujoco>
      <option sleep_tolerance="0.005"/>
      <worldbody/>
    </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull()) << error.data();

  mjSpec* child1 =
      mj_parseXMLString(child1_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child1, NotNull()) << error.data();

  mjSpec* child2 =
      mj_parseXMLString(child2_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child2, NotNull()) << error.data();

  mjsBody* world = mjs_findBody(parent, "world");

  // Perform Attach 1
  mjsElement* attached1 =
      mjs_attach(world->element, child1->element, "c1_", "");
  ASSERT_THAT(attached1, NotNull())
      << "Error details: " << mjs_getError(parent);
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));

  // Perform Attach 2
  mjsElement* attached2 =
      mjs_attach(world->element, child2->element, "c2_", "");
  ASSERT_THAT(attached2, NotNull())
      << "Error details: " << mjs_getError(parent);
  EXPECT_EQ(mjs_numWarnings(parent), 2);
  EXPECT_THAT(mjs_getWarning(parent, 1), HasSubstr("sleep_tolerance"));

  // Compile 1: compiles and generates a compile warning (flex is not rigid)
  mjModel* model1 = mj_compile(parent, nullptr);
  ASSERT_THAT(model1, NotNull());

  // Warnings should include both attach warnings (indices 0 and 1) AND the
  // compile warning (at index 2)
  int num_warnings_after_compile = mjs_numWarnings(parent);
  EXPECT_GE(num_warnings_after_compile, 3);
  EXPECT_THAT(mjs_getWarning(parent, 2), HasSubstr("not rigid"));

  // Compile 2 (Recompile): Recompiling should CLEAR compile warnings but
  // PRESERVE all attach warnings!
  mjModel* model2 = mj_compile(parent, nullptr);
  ASSERT_THAT(model2, NotNull());

  // Attach warnings must still be preserved and compile warnings refreshed
  EXPECT_EQ(mjs_numWarnings(parent), num_warnings_after_compile);
  EXPECT_THAT(mjs_getWarning(parent, 0), HasSubstr("timestep"));
  EXPECT_THAT(mjs_getWarning(parent, 1), HasSubstr("sleep_tolerance"));
  EXPECT_THAT(mjs_getWarning(parent, 2), HasSubstr("not rigid"));

  mj_deleteModel(model1);
  mj_deleteModel(model2);
  mj_deleteSpec(parent);
  mj_deleteSpec(child1);
  mj_deleteSpec(child2);
}

TEST_F(MujocoTest, AttachConflictWarningZFarDefaultMessage) {
  mock_warning_handler.ExpectWarnings();

  static constexpr char parent_xml[] = R"(
    <mujoco>
      <compiler conflict="warning"/>
      <worldbody/>
    </mujoco>
  )";

  static constexpr char child_xml[] = R"(
    <mujoco>
      <visual>
        <map zfar="30"/>
      </visual>
      <worldbody/>
    </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* parent =
      mj_parseXMLString(parent_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull()) << error.data();

  mjSpec* child =
      mj_parseXMLString(child_xml, nullptr, error.data(), error.size());
  ASSERT_THAT(child, NotNull()) << error.data();

  mjsBody* world = mjs_findBody(parent, "world");
  mjsElement* attached =
      mjs_attach(world->element, child->element, "child_", "");
  ASSERT_THAT(attached, NotNull()) << "Error details: " << mjs_getError(parent);

  // Since only the child authored 'zfar' and the parent relied on defaults,
  // we warning-log and state that the parent has the default value.
  EXPECT_TRUE(mjs_isWarning(parent));
  EXPECT_EQ(mjs_numWarnings(parent), 1);
  EXPECT_THAT(mjs_getWarning(parent, 0),
              HasSubstr("zfar: parent has 50 (default), child has "
                        "30, keeping parent value"));

  mj_deleteSpec(parent);
  mj_deleteSpec(child);
}

TEST_F(MujocoTest, AttachConflictXMLBitfieldAuthoredDefault) {
  mock_warning_handler.ExpectWarnings();
  std::array<char, 1024> err;

  // error mode: child explicitly authors constraint="enable" (0), conflicting
  // with parent's constraint="disable" (1)
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="error"/>
        <option>
          <flag constraint="disable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    static constexpr char child_xml[] = R"(
      <mujoco>
        <option>
          <flag constraint="enable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";

    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();
    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    EXPECT_THAT(attached, IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("flag 'Constraint': parent set, child unset"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // warning mode: authored zero bits in disableflags and enableflags emit
  // warnings and preserve parent flags
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="warning"/>
        <option>
          <flag constraint="disable" energy="enable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    static constexpr char child_xml[] = R"(
      <mujoco>
        <option>
          <flag constraint="enable" energy="disable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";

    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();
    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    ASSERT_THAT(attached, NotNull()) << mjs_getError(parent);
    EXPECT_EQ(parent->option.disableflags, mjDSBL_CONSTRAINT);
    EXPECT_EQ(parent->option.enableflags, mjENBL_ENERGY);
    EXPECT_EQ(mjs_numWarnings(parent), 1);
    std::string w = mjs_getWarning(parent, 0);
    EXPECT_THAT(w, HasSubstr("flag 'Constraint': parent set, child unset"));
    EXPECT_THAT(w, HasSubstr("flag 'Energy': parent set, child unset"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // disableactuator: child clears authored group bit via C API
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="error"/>
        <option actuatorgroupdisable="2"/>
        <worldbody/>
      </mujoco>
    )";
    static constexpr char child_xml[] = R"(
      <mujoco>
        <option actuatorgroupdisable="2"/>
        <worldbody/>
      </mujoco>
    )";

    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();
    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();
    child->option.disableactuator = 0;

    mjsBody* world = mjs_findBody(parent, "world");
    mjsElement* attached =
        mjs_attach(world->element, child->element, "child_", "");
    EXPECT_THAT(attached, IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("disableactuator group 2: parent set, child unset"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }
}

TEST_F(MujocoTest, AttachConflictMixedXMLAndCApi) {
  std::array<char, 1024> err;

  // scalar: XML parent + C-API child
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="error"/>
        <option timestep="0.005"/>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();

    mjSpec* child = mj_makeSpec();
    child->option.timestep = 0.01;

    mjsBody* world = mjs_findBody(parent, "world");
    EXPECT_THAT(mjs_attach(world->element, child->element, "child_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("timestep: parent has 0.005, child has 0.01"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // scalar: C-API parent + XML child explicitly authoring default value
  {
    mjSpec* parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_ERROR;
    parent->option.timestep = 0.01;

    static constexpr char child_xml[] = R"(
      <mujoco>
        <option timestep="0.002"/>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    EXPECT_THAT(mjs_attach(world->element, child->element, "child_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("timestep: parent has 0.01, child has 0.002"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // array: C-API parent + XML child explicitly authoring default gravity
  {
    mjSpec* parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_ERROR;
    parent->option.gravity[2] = -1.62;

    static constexpr char child_xml[] = R"(
      <mujoco>
        <option gravity="0 0 -9.81"/>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    EXPECT_THAT(mjs_attach(world->element, child->element, "child_", ""),
                IsNull());
    EXPECT_THAT(
        mjs_getError(parent),
        HasSubstr("gravity: parent has 0 0 -1.62, child has 0 0 -9.81"));

    mj_deleteSpec(parent);
    mj_deleteSpec(child);
  }

  // bitfield: XML parent + C-API child, and post-parse C-API bitfield mutation
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="error"/>
        <option>
          <flag gravity="enable" contact="enable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();

    mjSpec* child = mj_makeSpec();
    child->option.disableflags = mjDSBL_GRAVITY;

    mjsBody* world = mjs_findBody(parent, "world");
    EXPECT_THAT(mjs_attach(world->element, child->element, "c1_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("flag 'Gravity': parent unset, child set"));

    static constexpr char xml_child_str[] = R"(
      <mujoco>
        <option>
          <flag gravity="enable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* xml_child =
        mj_parseXMLString(xml_child_str, nullptr, err.data(), err.size());
    ASSERT_THAT(xml_child, NotNull()) << err.data();
    xml_child->option.disableflags |= mjDSBL_CONTACT;

    EXPECT_THAT(mjs_attach(world->element, xml_child->element, "c2_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent),
                HasSubstr("flag 'Contact': parent unset, child set"));

    mj_deleteSpec(xml_child);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }

  // disjoint bitfields: XML parent (<flag contact="disable"/>) + C-API child
  // (mjDSBL_GRAVITY) in error and merge modes
  {
    static constexpr char parent_error_xml[] = R"(
      <mujoco>
        <compiler conflict="error"/>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    mjSpec* parent =
        mj_parseXMLString(parent_error_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();

    mjSpec* child = mj_makeSpec();
    child->option.disableflags = mjDSBL_GRAVITY;

    mjsBody* world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_FALSE(mjs_isWarning(parent));
    EXPECT_EQ(parent->option.disableflags, mjDSBL_CONTACT);

    mj_deleteSpec(child);
    mj_deleteSpec(parent);

    MockWarningHandler merge_warnings;
    merge_warnings.ExpectWarnings();
    static constexpr char parent_merge_xml[] = R"(
      <mujoco>
        <compiler conflict="merge"/>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";
    parent =
        mj_parseXMLString(parent_merge_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();

    child = mj_makeSpec();
    child->option.disableflags = mjDSBL_GRAVITY;

    world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_EQ(parent->option.disableflags, mjDSBL_CONTACT | mjDSBL_GRAVITY);
    EXPECT_EQ(mjs_numWarnings(parent), 1);
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("flag 'Gravity': added from child"));
    EXPECT_THAT(mjs_getWarning(parent, 0), Not(HasSubstr("Contact")));

    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }

  // disjoint bitfields: C-API parent (mjDSBL_GRAVITY) + XML child
  // (<flag contact="disable"/>) in error and merge modes
  {
    static constexpr char child_xml[] = R"(
      <mujoco>
        <option>
          <flag contact="disable"/>
        </option>
        <worldbody/>
      </mujoco>
    )";

    mjSpec* parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_ERROR;
    parent->option.disableflags = mjDSBL_GRAVITY;

    mjSpec* child =
        mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_FALSE(mjs_isWarning(parent));
    EXPECT_EQ(parent->option.disableflags, mjDSBL_GRAVITY);

    mj_deleteSpec(child);
    mj_deleteSpec(parent);

    MockWarningHandler merge_warnings;
    merge_warnings.ExpectWarnings();
    parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_MERGE;
    parent->option.disableflags = mjDSBL_GRAVITY;

    child = mj_parseXMLString(child_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child, NotNull()) << err.data();

    world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_EQ(parent->option.disableflags, mjDSBL_GRAVITY | mjDSBL_CONTACT);
    EXPECT_EQ(mjs_numWarnings(parent), 1);
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("flag 'Contact': added from child"));
    EXPECT_THAT(mjs_getWarning(parent, 0), Not(HasSubstr("Gravity")));

    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }

  // disjoint bitfields: two C-API specs setting disjoint flags in error and
  // merge modes
  {
    mjSpec* parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_ERROR;
    parent->option.disableflags = mjDSBL_GRAVITY;
    parent->option.disableactuator = 1 << 1;

    mjSpec* child = mj_makeSpec();
    child->option.disableflags = mjDSBL_CONTACT;
    child->option.disableactuator = 1 << 2;

    mjsBody* world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_FALSE(mjs_isWarning(parent));
    EXPECT_EQ(parent->option.disableflags, mjDSBL_GRAVITY);
    EXPECT_EQ(parent->option.disableactuator, 1 << 1);

    mj_deleteSpec(child);
    mj_deleteSpec(parent);

    MockWarningHandler merge_warnings;
    merge_warnings.ExpectWarnings();
    parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_MERGE;
    parent->option.disableflags = mjDSBL_GRAVITY;
    parent->option.disableactuator = 1 << 1;

    child = mj_makeSpec();
    child->option.disableflags = mjDSBL_CONTACT;
    child->option.disableactuator = 1 << 2;

    world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child->element, "c_", ""), NotNull())
        << mjs_getError(parent);
    EXPECT_EQ(parent->option.disableflags, mjDSBL_GRAVITY | mjDSBL_CONTACT);
    EXPECT_EQ(parent->option.disableactuator, (1 << 1) | (1 << 2));
    EXPECT_EQ(mjs_numWarnings(parent), 1);
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("flag 'Contact': added from child"));
    EXPECT_THAT(mjs_getWarning(parent, 0),
                HasSubstr("disableactuator group 2: added from child"));

    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

TEST_F(MujocoTest, AttachMergeAdoptedAttributeMarkedAuthored) {
  mock_warning_handler.ExpectWarnings();
  std::array<char, 1024> err;

  // non-default scalar adopted from child1 is not overwritten by child2
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="merge"/>
        <worldbody>
          <frame name="f1"/>
          <frame name="f2"/>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child1_xml[] = R"(
      <mujoco>
        <option timestep="0.001"/>
        <worldbody>
          <body name="b1">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child2_xml[] = R"(
      <mujoco>
        <option timestep="0.005"/>
        <worldbody>
          <body name="b2">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";

    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();
    mjSpec* child1 =
        mj_parseXMLString(child1_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child1, NotNull()) << err.data();
    mjSpec* child2 =
        mj_parseXMLString(child2_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child2, NotNull()) << err.data();

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element,
                           mjs_findBody(child1, "b1")->element, "c1_", ""),
                NotNull());
    EXPECT_EQ(parent->option.timestep, mjtNum(0.001));

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element,
                           mjs_findBody(child2, "b2")->element, "c2_", ""),
                NotNull());
    EXPECT_EQ(parent->option.timestep, mjtNum(0.001));

    mj_deleteSpec(child2);
    mj_deleteSpec(child1);
    mj_deleteSpec(parent);
  }

  // explicitly authored default scalar and array adopted from child1 are
  // preserved when child2 attaches
  {
    static constexpr char parent_xml[] = R"(
      <mujoco>
        <compiler conflict="merge"/>
        <worldbody>
          <frame name="f1"/>
          <frame name="f2"/>
          <frame name="f3"/>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child1_xml[] = R"(
      <mujoco>
        <option timestep="0.002" gravity="0 0 -9.81">
          <flag constraint="disable"/>
        </option>
        <worldbody>
          <body name="b1">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child2_xml[] = R"(
      <mujoco>
        <option timestep="0.005"/>
        <worldbody>
          <body name="b2">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child3_xml[] = R"(
      <mujoco>
        <option gravity="0 0 -1.62"/>
        <worldbody>
          <body name="b3">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";

    mjSpec* parent =
        mj_parseXMLString(parent_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(parent, NotNull()) << err.data();
    mjSpec* child1 =
        mj_parseXMLString(child1_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child1, NotNull()) << err.data();
    mjSpec* child2 =
        mj_parseXMLString(child2_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child2, NotNull()) << err.data();
    mjSpec* child3 =
        mj_parseXMLString(child3_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child3, NotNull()) << err.data();

    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f1")->element,
                           mjs_findBody(child1, "b1")->element, "c1_", ""),
                NotNull());
    ASSERT_THAT(mjs_attach(mjs_findFrame(parent, "f2")->element,
                           mjs_findBody(child2, "b2")->element, "c2_", ""),
                NotNull());
    EXPECT_EQ(parent->option.timestep, mjtNum(0.002));

    // child3's unmergeable gravity conflicts with child1's adopted default
    EXPECT_THAT(mjs_attach(mjs_findFrame(parent, "f3")->element,
                           mjs_findBody(child3, "b3")->element, "c3_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent), HasSubstr("gravity"));

    // child1's adopted disableflag is also marked authored on parent
    parent->compiler.conflict = mjCONFLICT_ERROR;
    static constexpr char child4_xml[] = R"(
      <mujoco>
        <option>
          <flag constraint="enable"/>
        </option>
        <worldbody>
          <body name="b4">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    mjSpec* child4 =
        mj_parseXMLString(child4_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child4, NotNull()) << err.data();
    EXPECT_THAT(mjs_attach(mjs_findFrame(parent, "f3")->element,
                           mjs_findBody(child4, "b4")->element, "c4_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent), HasSubstr("flag 'Constraint'"));

    mj_deleteSpec(child4);
    mj_deleteSpec(child3);
    mj_deleteSpec(child2);
    mj_deleteSpec(child1);
    mj_deleteSpec(parent);
  }

  // equal C-API scalar and array values on parent are marked authored on merge
  {
    mjSpec* parent = mj_makeSpec();
    parent->compiler.conflict = mjCONFLICT_MERGE;
    parent->option.timestep = 0.001;
    parent->option.gravity[2] = -1.62;
    EXPECT_EQ(parent->authored.option, 0);

    static constexpr char child1_xml[] = R"(
      <mujoco>
        <option timestep="0.001" gravity="0 0 -1.62"/>
        <worldbody>
          <body name="b1">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child2_xml[] = R"(
      <mujoco>
        <option timestep="0.005"/>
        <worldbody>
          <body name="b2">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    static constexpr char child3_xml[] = R"(
      <mujoco>
        <option gravity="0 0 -3.71"/>
        <worldbody>
          <body name="b3">
            <geom size="0.1"/>
          </body>
        </worldbody>
      </mujoco>
    )";
    mjSpec* child1 =
        mj_parseXMLString(child1_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child1, NotNull()) << err.data();
    mjSpec* child2 =
        mj_parseXMLString(child2_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child2, NotNull()) << err.data();
    mjSpec* child3 =
        mj_parseXMLString(child3_xml, nullptr, err.data(), err.size());
    ASSERT_THAT(child3, NotNull()) << err.data();

    mjsBody* world = mjs_findBody(parent, "world");
    ASSERT_THAT(mjs_attach(world->element, child1->element, "c1_", ""),
                NotNull())
        << mjs_getError(parent);
    EXPECT_EQ(parent->authored.option, child1->authored.option);

    // after resetting to defaults via C API, authored bits still protect both
    parent->option.timestep = 0.002;
    parent->option.gravity[2] = -9.81;
    ASSERT_THAT(mjs_attach(world->element, child2->element, "c2_", ""),
                NotNull())
        << mjs_getError(parent);
    EXPECT_EQ(parent->option.timestep, mjtNum(0.002));
    EXPECT_THAT(mjs_attach(world->element, child3->element, "c3_", ""),
                IsNull());
    EXPECT_THAT(mjs_getError(parent), HasSubstr("gravity"));

    mj_deleteSpec(child3);
    mj_deleteSpec(child2);
    mj_deleteSpec(child1);
    mj_deleteSpec(parent);
  }
}

}  // namespace
}  // namespace mujoco
