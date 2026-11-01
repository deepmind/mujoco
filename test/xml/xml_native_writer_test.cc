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

// Tests for xml/xml_native_writer.cc.

#include <memory>
#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
  #include <unistd.h>
#endif

#include <array>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "src/xml/xml_numeric_format.h"
#include "test/compare_model.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::DoubleNear;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pointwise;

using XMLWriterTest = MujocoTest;

static const char* const kNonRgbTextureXMLPath =
    "xml/testdata/hfield_png_nonrgb.xml";

TEST_F(XMLWriterTest, EmptyModel) {
  static constexpr char xml[] = "<mujoco/>";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("default")));
}

TEST_F(XMLWriterTest, KeepsDCMotorNoInputs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="jnt"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0" input="none"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("input=\"none\""));

  // reload: the empty input block survives the round trip
  MjModelPtr model2 = LoadModelFromString(saved_xml.c_str());
  ASSERT_THAT(model2.get(), NotNull());
  EXPECT_EQ(model2->nu, 0);
  EXPECT_EQ(model2->actuator_ctrlnum[0], 0);
}

TEST_F(XMLWriterTest, KeepsDeclaredInputs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="servo">
        <position kp="10"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="jnt"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt"/>
      <velocity joint="jnt"/>
      <intvelocity joint="jnt"/>
      <cylinder joint="jnt"/>
      <motor joint="jnt"/>
      <position class="servo" joint="jnt"/>
      <velocity class="servo" joint="jnt"/>
      <motor class="servo" joint="jnt"/>
      <damper class="servo" joint="jnt" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::vector<int> ctrlspec =
      AsVector(model->actuator_ctrlspec, model->nactuator);
  EXPECT_THAT(ctrlspec,
              ElementsAre(mjINPUT_POS, mjINPUT_VEL, mjINPUT_VEL,
                          mjINPUT_PRESSURE, 0, mjINPUT_POS, mjINPUT_VEL, 0, 0));

  // saved as general: the declarations are written and survive reloading
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("input=\"pos\""));
  EXPECT_THAT(saved_xml, HasSubstr("input=\"vel\""));
  EXPECT_THAT(saved_xml, HasSubstr("input=\"pressure\""));

  // the motor in the servo class clears the declaration it would inherit
  EXPECT_THAT(saved_xml, HasSubstr("input=\"\""));
  MjModelPtr model2 = LoadModelFromString(saved_xml.c_str());
  ASSERT_THAT(model2.get(), NotNull());
  EXPECT_EQ(AsVector(model2->actuator_ctrlspec, model2->nactuator), ctrlspec);
}

TEST_F(XMLWriterTest, SavesMemory) {
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory=" 1023 "/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    std::string saved_xml = SaveAndReadXml(model.get());
    EXPECT_THAT(saved_xml, HasSubstr("memory=\"1023\""));
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1024"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    std::string saved_xml = SaveAndReadXml(model.get());
    EXPECT_THAT(saved_xml, HasSubstr("memory=\"1K\""));
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="4096"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    std::string saved_xml = SaveAndReadXml(model.get());
    EXPECT_THAT(saved_xml, HasSubstr("memory=\"4K\""));
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1048576"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    std::string saved_xml = SaveAndReadXml(model.get());
    EXPECT_THAT(saved_xml, HasSubstr("memory=\"1M\""));
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1047552"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    std::string saved_xml = SaveAndReadXml(model.get());
    EXPECT_THAT(saved_xml, HasSubstr("memory=\"1023K\""));
  }
}

TEST_F(XMLWriterTest, SavesDisableSensor) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sensor="disable"/>
    </option>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("sensor=\"disable\""));
}

TEST_F(XMLWriterTest, SavesInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler saveinertial="true"/>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("mass=\"1\""));
}

// the attributes which say how to save are read into the spec and not saved
TEST_F(XMLWriterTest, SavingControlsAreNotSaved) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler saveinertial="true" savecompiled="true" savecanonical="false"/>
    <worldbody>
      <body>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  EXPECT_EQ(spec->compiler.savecompiled, 1);
  EXPECT_EQ(spec->compiler.savecanonical, 0);

  // both are "true" unless set
  mjSpec* empty = mj_makeSpec();
  EXPECT_EQ(empty->compiler.savecompiled, 1);
  EXPECT_EQ(empty->compiler.savecanonical, 1);
  mj_deleteSpec(empty);

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr("<inertial"));
  EXPECT_THAT(saved, Not(HasSubstr("save")));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(XMLWriterTest, EmptyUserSensor) {
  static constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <user dim="2" needstage="vel"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("dim=\"2\""));
  EXPECT_THAT(saved_xml, HasSubstr("needstage=\"vel\""));
  EXPECT_THAT(saved_xml, HasSubstr("datatype=\"real\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("objtype")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("objname")));
}

TEST_F(XMLWriterTest, KeepsEmptyClasses) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="empty_referenced"/>
      <default class="empty_unreferenced"/>
      <default class="regular">
        <geom size="0.3"/>
      </default>
    </default>
    <worldbody>
      <geom class="regular"/>
      <geom class="empty_referenced" size="0.2"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("default class=\"regular\""));
  EXPECT_THAT(saved_xml, HasSubstr("default class=\"empty_referenced\""));
  EXPECT_THAT(saved_xml, HasSubstr("default class=\"empty_unreferenced\""));
}

TEST_F(XMLWriterTest, KeepsExplicitInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="0.2"/>
        <inertial pos="0 1 2" mass="3"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("<inertial pos=\"0 1 2\" mass=\"3\""));
}

TEST_F(XMLWriterTest, NotAddsInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="0.2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("inertial")));
}

// the masses of a model which sets its total mass are saved as they were
// scaled, for the bodies with an inertial and for those without
TEST_F(XMLWriterTest, KeepsTotalMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler settotalmass="8"/>
    <worldbody>
      <body name="given">
        <joint/>
        <inertial pos="0 0 .5" mass="1" diaginertia=".5 .5 .25"/>
        <geom size=".25"/>
      </body>
      <body name="inferred" pos="1 0 0">
        <joint/>
        <geom type="box" size=".25 .25 .5"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  mock_warning_handler.ExpectWarnings("settotalmass");
  FullFloatPrecision increase_precision;

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(mj_getTotalmass(model), 8, MjTol(1e-12, 1e-5));

  // saving the spec, or the model after loading the file
  MjModelPtr loaded = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(loaded.get(), NotNull()) << error.data();
  for (const std::string& saved :
       {SaveAndReadXml(spec), SaveAndReadXml(loaded.get())}) {
    EXPECT_THAT(saved, Not(HasSubstr("settotalmass")));
    MjModelPtr reloaded =
        LoadModelFromString(saved.c_str(), error.data(), error.size());
    ASSERT_THAT(reloaded.get(), NotNull()) << error.data();
    std::string field;
    EXPECT_LE(CompareModel(model, reloaded.get(), field), MjTol(1e-12, 1e-5))
        << field;
  }

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a body of an attached model is compiled with the settings of that model; the
// saved file has those of the model it is attached to, and gives the body its
// compiled inertia where it would not infer that inertia alike
TEST_F(XMLWriterTest, KeepsInertiaOfAttachedBodies) {
  struct Case {
    const char* parent;  // compiler attributes of the model
    const char* child;   // compiler attributes of the attached model
    const char* body;    // contents of the attached body
  };
  const Case cases[] = {
      // an inertial which "true" would replace with that of the geom, whose
      // mass is then not computed
      {R"(inertiafromgeom="true")", "",
       R"(<joint/><inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
          <geom size=".1" mass="10"/>)"},
      // inertia inferred from the geom, which "false" would make massless
      {R"(inertiafromgeom="false")", "",
       R"(<joint/><geom size=".1" mass="2"/>)"},
      // a massless body, which "auto" would give the mass of its geom
      {"", R"(inertiafromgeom="false")", R"(<geom size=".1" mass="5"/>)"},
      // inertia inferred from fewer geoms than the model counts; the geom which
      // it does not count is saved with no mass, but the inertia is given
      {"", R"(inertiagrouprange="0 1")",
       R"(<joint/><geom size=".1" mass="1"/>
          <geom size=".2" mass="4" group="3"/>)"},
  };
  FullFloatPrecision increase_precision;
  for (const Case& c : cases) {
    std::string parent_xml =
        std::string(R"(<mujoco><compiler savecompiled="true" )") + c.parent +
        R"(/><worldbody><frame name="frame" pos="1 0 0"/></worldbody></mujoco>)";
    std::string child_xml = std::string("<mujoco><compiler ") + c.child +
                            R"(/><worldbody><body name="body">)" + c.body +
                            "</body></worldbody></mujoco>";
    std::array<char, 1024> error;
    mjSpec* parent = mj_parseXMLString(parent_xml.c_str(), nullptr,
                                       error.data(), error.size());
    mjSpec* child = mj_parseXMLString(child_xml.c_str(), nullptr, error.data(),
                                      error.size());
    ASSERT_THAT(parent, NotNull()) << error.data();
    ASSERT_THAT(child, NotNull()) << error.data();
    mjsFrame* frame = mjs_findFrame(parent, "frame");
    mjsBody* body = mjs_findBody(child, "body");
    ASSERT_THAT(mjs_attach(frame->element, body->element, "child_", ""),
                NotNull())
        << mjs_getError(parent);
    mjModel* model = mj_compile(parent, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(parent);

    std::string saved = SaveAndReadXml(parent);
    EXPECT_THAT(saved, HasSubstr("<inertial"));
    MjModelPtr reloaded =
        LoadModelFromString(saved.c_str(), error.data(), error.size());
    EXPECT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
    if (reloaded) {
      std::string field;
      EXPECT_LE(CompareModel(model, reloaded.get(), field), MjTol(1e-12, 1e-5))
          << field << "\n"
          << saved;
    }

    mj_deleteModel(model);
    mj_deleteSpec(child);
    mj_deleteSpec(parent);
  }
}

// the inertia of a body is saved as the last compilation gave it, also if the
// spec was edited since, or as a model which was copied back to it gives it
TEST_F(XMLWriterTest, KeepsCompiledInertiaOfEditedSpec) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler savecompiled="true"/>
    <worldbody>
      <body name="given">
        <joint/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom size=".1" mass="10"/>
      </body>
      <body name="inferred" pos="1 0 0">
        <joint/>
        <geom size=".1" mass="2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  FullFloatPrecision increase_precision;
  for (int edit = 0; edit < 3; edit++) {
    std::array<char, 1024> error;
    mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << error.data();
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

    if (edit == 0) {
      // the spec would infer the inertia of every body
      spec->compiler.inertiafromgeom = mjINERTIAFROMGEOM_TRUE;
    } else if (edit == 1) {
      // the spec would infer the inertia of the body which was given one
      mjs_findBody(spec, "given")->ipos[0] = std::nan("");
    } else {
      // the model gives the other body a mass
      model->body_mass[2] = 4;
      ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
    }

    std::string saved = SaveAndReadXml(spec);
    MjModelPtr reloaded =
        LoadModelFromString(saved.c_str(), error.data(), error.size());
    EXPECT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
    if (reloaded) {
      for (int i = 0; i < 3; i++) {
        EXPECT_NEAR(reloaded->body_mass[i], model->body_mass[i],
                    MjTol(1e-12, 1e-5))
            << "edit " << edit << ", body " << i << "\n"
            << saved;
      }
      for (int i = 0; i < 9; i++) {
        EXPECT_NEAR(reloaded->body_inertia[i], model->body_inertia[i],
                    MjTol(1e-12, 1e-5))
            << "edit " << edit << "\n"
            << saved;
      }
    }

    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

TEST_F(XMLWriterTest, KeepsBoundMassInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler boundmass="0.1" boundinertia="0.2"/>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("boundmass=\"0.1\""));
  EXPECT_THAT(saved_xml, HasSubstr("boundinertia=\"0.2\""));
}

TEST_F(XMLWriterTest, DropsZeroBoundMassInertia) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler boundmass="0" boundinertia="0"/>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("boundmass")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("boundinertia")));
}

TEST_F(XMLWriterTest, KeepsInertiaFromGeomFalse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiafromgeom="false"/>
    <worldbody>
      <body>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <joint/>
        <body>
          <geom size="0.1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("inertiafromgeom=\"false\""));

  // the welded body stays massless, its inertia is not inferred from the geom
  MjModelPtr reloaded = LoadModelFromString(saved_xml);
  ASSERT_THAT(reloaded.get(), NotNull());
  EXPECT_EQ(AsVector(reloaded->body_mass, 3), AsVector(model->body_mass, 3));
  EXPECT_EQ(AsVector(reloaded->body_inertia, 9),
            AsVector(model->body_inertia, 9));
}

TEST_F(XMLWriterTest, DropsInertialIfFromGeom) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiafromgeom="true"/>
    <worldbody>
      <body>
        <inertial pos="0 1 2" mass="3"/>
        <geom size="0.2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("inertial")));
}

TEST_F(XMLWriterTest, KeepsInertiaGroupRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiagrouprange="0 1"/>
    <worldbody>
      <body>
        <geom size="0.1"/>
        <geom size="0.2" group="2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("inertiagrouprange=\"0 1\""));

  // the body inertia is inferred from the same geoms
  MjModelPtr reloaded = LoadModelFromString(saved_xml);
  ASSERT_THAT(reloaded.get(), NotNull());
  EXPECT_EQ(reloaded->body_mass[1], model->body_mass[1]);
  EXPECT_EQ(AsVector(reloaded->body_inertia + 3, 3),
            AsVector(model->body_inertia + 3, 3));
}

TEST_F(XMLWriterTest, KeepsAutoLimitsFalse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="false"/>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("autolimits=\"false\""));
}

TEST_F(XMLWriterTest, DoesNotKeepInferredJointLimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge" range="-1 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1 1\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("limited=\"true\"")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
}

TEST_F(XMLWriterTest, KeepsExplicitJointLimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge" limited="true" range="-1 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("limited=\"true\""));
}

TEST_F(XMLWriterTest, KeepsJointLimitedFalseIfAutoLimits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge" limited="false" range="-1 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1 1\" limited=\"false\""));
}

TEST_F(XMLWriterTest, DoesNotKeepInferredTendonLimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint type="slide"/>
        <geom size="1"/>
        <site name="s1"/>
      </body>
      <site name="s2"/>
    </worldbody>
    <tendon>
      <spatial range="-1 1">
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1 1\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("limited=\"true\"")));
}

TEST_F(XMLWriterTest, KeepsExplicitTendonLimitedIfAutoLimits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint type="slide"/>
        <geom size="1"/>
        <site name="s1"/>
      </body>
      <site name="s2"/>
    </worldbody>
    <tendon>
      <spatial limited="true" range="-1 1">
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("limited=\"true\""));
}

TEST_F(XMLWriterTest, KeepsTendonLimitedFalseIfAutoLimits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian" autolimits="true" />
    <worldbody>
      <body>
        <joint type="slide"/>
        <geom size="1"/>
        <site name="s1"/>
      </body>
      <site name="s2"/>
    </worldbody>
    <tendon>
      <spatial limited="false" range="-1 1">
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("limited=\"false\" range=\"-1 1\""));
}

TEST_F(XMLWriterTest, DoesNotKeepInferredActlimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general dyntype="filter" joint="hinge" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("actrange=\"-1 1\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("actlimited=\"true\"")));
}

TEST_F(XMLWriterTest, KeepsExplicitActlimitedIfAutoLimits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general dyntype="filter" joint="hinge" actlimited="true" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("actrange=\"-1 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("actlimited=\"true\""));
}

TEST_F(XMLWriterTest, KeepsActlimitedFalse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general dyntype="filter" joint="hinge" actlimited="false" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("actlimited=\"false\" actrange=\"-1 1\""));
}

TEST_F(XMLWriterTest, DoesNotKeepInferredCtrllimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" ctrlrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("ctrlrange=\"-1 1\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("ctrllimited=\"true\"")));
}

TEST_F(XMLWriterTest, KeepsExplicitCtrllimitedIfAutoLimits) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" ctrllimited="true" ctrlrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("ctrlrange=\"-1 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("ctrllimited=\"true\""));
}

TEST_F(XMLWriterTest, KeepsCtrllimitedFalse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" ctrllimited="false" ctrlrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("ctrlrange=\"-1 1\" ctrllimited=\"false\""));
}

TEST_F(XMLWriterTest, DoesNotKeepInferredForcelimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler autolimits="true" />
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" forcerange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("forcerange=\"-1 1\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("forcelimited=\"true\"")));
}

TEST_F(XMLWriterTest, KeepsExplicitForcelimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" forcelimited="true" forcerange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("autolimits=\"true\"")));
  EXPECT_THAT(saved_xml, HasSubstr("forcerange=\"-1 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("forcelimited=\"true\""));
}

TEST_F(XMLWriterTest, KeepsForcelimitedFalse) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" forcelimited="false" forcerange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml,
              HasSubstr("forcelimited=\"false\" forcerange=\"-1 1\""));
}

TEST_F(XMLWriterTest, WritesGravComp) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body gravcomp=".25"/>
      <body/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("gravcomp=\"0.25\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("gravcomp=\"0\"")));
}

TEST_F(XMLWriterTest, WritesBodyPos) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 2 3"/>
      <body pos="0 0 0"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("pos=\"1 2 3\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("pos=\"0 0 0\"")));
}

TEST_F(XMLWriterTest, UndefinedMassDensity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("density")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("mass")));
}

TEST_F(XMLWriterTest, WritesDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <geom density="100"/>
    </default>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("mass")));
  EXPECT_THAT(saved_xml, HasSubstr("<geom density=\"100\"/>"));
}

TEST_F(XMLWriterTest, WritesActuatorDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="position">
        <position kp="3" kv="4" />
      </default>
      <default class="intvelocity">
        <intvelocity kp="5" kv="6" />
      </default>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="0 2"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" class="position"/>
      <intvelocity joint="jnt" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("mass")));
  EXPECT_THAT(
      saved_xml,
      HasSubstr("<general input=\"pos\" biastype=\"affine\" gainprm=\"3\""));
}

TEST_F(XMLWriterTest, WritesFrameDefaults) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "Quat values differ in float32 (euler->quat rounding)";
#endif
  static constexpr char xml[] = R"(
  <mujoco model="test">
    <default>
      <default class="dframe">
        <geom size=".1"/>
      </default>
    </default>

    <worldbody>
      <frame name="f1" euler="0 0 30">
        <geom size=".5" euler="0 0 20"/>
      </frame>

      <body name="body">
        <frame pos="0 1 0" name="f2" childclass="dframe">
          <geom pos="0 1 0"/>
          <frame pos="0 1 0" name="f3">
            <frame pos="0 1 0">
              <body pos="1 0 0">
                <geom pos="0 0 1"/>
              </body>
            </frame>
          </frame>
          <frame name="f4" childclass="main">
            <geom size=".2"/>
          </frame>
        </frame>
        <frame>
          <light pos="0 0 1"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_expected[] = R"(<mujoco model="test">
  <compiler angle="radian"/>

  <default>
    <default class="dframe">
      <geom size="0.1 0 0"/>
    </default>
  </default>

  <worldbody>
    <body name="body">
      <frame name="f2" childclass="dframe" pos="0 1 0">
        <geom pos="0 1 0"/>
        <frame name="f3" pos="0 1 0">
          <frame pos="0 1 0">
            <body pos="1 0 0">
              <geom pos="0 0 1"/>
            </body>
          </frame>
        </frame>
        <frame name="f4" childclass="main">
          <geom size="0.2"/>
        </frame>
      </frame>
      <light pos="0 0 1"/>
    </body>
    <frame name="f1" quat="0.965926 0 0 0.258819">
      <geom size="0.5" quat="0.984808 0 0 0.173648"/>
    </frame>
  </worldbody>
</mujoco>
)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_STREQ(saved_xml.c_str(), xml_expected);
}

// frames survive save/load with their authored pose, every element type under
// them is written frame-relative, and the compiled model is unchanged
TEST_F(XMLWriterTest, FramesRoundTrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetra" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
    </asset>
    <worldbody>
      <frame name="f1" pos="1 2 3" euler="0 0 90">
        <geom name="ball" size=".1" pos=".1 .2 .3" euler="30 0 0"/>
        <geom name="mesh" type="mesh" mesh="tetra" pos=".1 0 0" euler="0 30 0"/>
        <site name="site" pos="0 .1 0" euler="0 45 0"/>
        <camera name="cam" pos="0 0 1" euler="0 0 45"/>
        <light name="light" pos="0 0 2" dir="1 0 -1"/>
        <frame pos="0 0 1" euler="90 0 0">
          <body name="b1" pos="1 0 0" euler="0 90 0">
            <joint name="hinge" pos=".1 .2 .3" axis="0 1 0"/>
            <geom size=".1"/>
            <body name="b2" pos="0 1 0">
              <frame pos=".3 .2 .1" euler="0 0 90">
                <joint name="framed_slide" type="slide" pos=".1 0 0" axis="1 0 0"/>
                <frame euler="90 0 0">
                  <joint name="framed_hinge" pos="0 .2 .3" axis="0 0 1"/>
                </frame>
              </frame>
              <geom size=".1"/>
            </body>
          </body>
        </frame>
      </frame>
    </worldbody>
  </mujoco>
  )";

  FullFloatPrecision increase_precision;
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull());

  // save and reload
  std::string saved = SaveAndReadXml(spec);
  mjSpec* spec2 =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec2, NotNull()) << error.data() << "\n" << saved;
  mjModel* model2 = mj_compile(spec2, nullptr);
  ASSERT_THAT(model2, NotNull());

  // the frame is still there with its authored pose
  mjsFrame* f1 = mjs_findFrame(spec2, "f1");
  ASSERT_THAT(f1, NotNull());
  EXPECT_THAT(f1->pos, ElementsAre(1, 2, 3));
  EXPECT_THAT(f1->quat, ElementsAre(DoubleNear(std::sqrt(0.5), 1e-12), 0, 0,
                                    DoubleNear(std::sqrt(0.5), 1e-12)));

  // joints inside frames keep their authored, frame-relative anchor and axis
  mjsJoint* slide =
      mjs_asJoint(mjs_findElement(spec2, mjOBJ_JOINT, "framed_slide"));
  mjsJoint* hinge =
      mjs_asJoint(mjs_findElement(spec2, mjOBJ_JOINT, "framed_hinge"));
  ASSERT_THAT(slide, NotNull());
  ASSERT_THAT(hinge, NotNull());
  const double slide_pos[3] = {.1, 0, 0}, slide_axis[3] = {1, 0, 0};
  const double hinge_pos[3] = {0, .2, .3}, hinge_axis[3] = {0, 0, 1};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(slide->pos[i], slide_pos[i], 1e-12);
    EXPECT_NEAR(slide->axis[i], slide_axis[i], 1e-12);
    EXPECT_NEAR(hinge->pos[i], hinge_pos[i], 1e-12);
    EXPECT_NEAR(hinge->axis[i], hinge_axis[i], 1e-12);
  }

  // the compiled model is unchanged
  std::string field;
  EXPECT_LT(CompareModel(model, model2, field), MjTol(1e-12, 1e-6)) << field;

  mj_deleteModel(model2);
  mj_deleteSpec(spec2);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// elements of one kind keep their order, and so their ids, when frames hold
// some of them
TEST_F(XMLWriterTest, KeepsOrderAroundFrames) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="g0" size="1"/>
      <frame name="outer">
        <geom name="g1" size="1"/>
        <body name="b1"/>
        <body name="b2"/>
        <frame name="inner">
          <body name="b3"/>
          <geom name="g2" size="1"/>
        </frame>
        <site name="s0"/>
      </frame>
      <geom name="g3" size="1"/>
      <body name="b4"/>
      <site name="s1"/>
      <frame name="last">
        <site name="s2"/>
      </frame>
      <site name="s3"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved = SaveAndReadXml(model.get());
  std::array<char, 1024> error;
  MjModelPtr reloaded =
      LoadModelFromString(saved.c_str(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
  for (int i = 0; i < 4; i++) {
    EXPECT_STREQ(mj_id2name(reloaded.get(), mjOBJ_GEOM, i),
                 ("g" + std::to_string(i)).c_str())
        << saved;
    EXPECT_STREQ(mj_id2name(reloaded.get(), mjOBJ_BODY, i + 1),
                 ("b" + std::to_string(i + 1)).c_str())
        << saved;
    EXPECT_STREQ(mj_id2name(reloaded.get(), mjOBJ_SITE, i),
                 ("s" + std::to_string(i)).c_str())
        << saved;
  }
}

// an element or a body which has the main class inside a body with another
// childclass is saved with its class
TEST_F(XMLWriterTest, KeepsMainClassUnderChildclass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <geom size="1"/>
      <default class="small">
        <geom size="0.5"/>
      </default>
    </default>

    <worldbody>
      <body childclass="small">
        <geom name="small"/>
        <geom name="main" class="main"/>
        <body childclass="main">
          <geom name="inner"/>
          <frame childclass="small">
            <geom name="framed"/>
            <frame childclass="main">
              <geom name="nested"/>
            </frame>
          </frame>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_THAT(AsVector(model->geom_size, 15),
              ElementsAre(0.5, 0, 0, 1, 0, 0, 1, 0, 0, 0.5, 0, 0, 1, 0, 0));
  std::string saved = SaveAndReadXml(model.get());
  std::array<char, 1024> error;
  MjModelPtr reloaded =
      LoadModelFromString(saved.c_str(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
  EXPECT_THAT(AsVector(reloaded->geom_size, 15),
              ElementsAre(0.5, 0, 0, 1, 0, 0, 1, 0, 0, 0.5, 0, 0, 1, 0, 0))
      << saved;
}

// quaternion product and pose composition in double precision
static void MulQuat(double res[4], const double a[4], const double b[4]) {
  double q[4] = {a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                 a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                 a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                 a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]};
  for (int i = 0; i < 4; i++) {
    res[i] = q[i];
  }
}

static void MulPose(double pos[3], double quat[4], const double pos1[3],
                    const double quat1[4], const double pos2[3],
                    const double quat2[4]) {
  // rotate pos2 by quat1: q * (0, v) * q'
  double v[4] = {0, pos2[0], pos2[1], pos2[2]};
  double conj[4] = {quat1[0], -quat1[1], -quat1[2], -quat1[3]};
  double tmp[4];
  MulQuat(tmp, quat1, v);
  MulQuat(tmp, tmp, conj);
  for (int i = 0; i < 3; i++) {
    pos[i] = pos1[i] + tmp[i + 1];
  }
  MulQuat(quat, quat1, quat2);
}

// a frame inside a free-aligned body stays where it was and keeps the authored
// poses of its contents
TEST_F(XMLWriterTest, FramesInAlignedBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body" pos="0 0 1" euler="0 30 0">
        <freejoint align="true"/>
        <geom type="box" size=".1 .2 .3" pos=".5 0 0" euler="0 0 30"/>
        <frame name="outer" pos="0 1 0" euler="0 0 90">
          <geom name="geom" size=".1" pos=".2 0 0"/>
          <frame name="inner" pos="0 0 .5" euler="90 0 0">
            <site name="site" pos="0 .3 0"/>
            <camera name="camera" pos="0 0 .4"/>
            <light name="light" pos="0 0 .5" dir="1 0 0"/>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  FullFloatPrecision increase_precision;
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull());

  // world pose of the outer frame, before alignment: body pose times frame pose
  mjsBody* body = mjs_findBody(spec, "body");
  mjsFrame* outer = mjs_findFrame(spec, "outer");
  double body_quat[4], outer_quat[4];
  mjs_resolveOrientation(body_quat, true, "xyz", &body->alt);
  mjs_resolveOrientation(outer_quat, true, "xyz", &outer->alt);
  double pos[3], quat[4];
  MulPose(pos, quat, body->pos, body_quat, outer->pos, outer_quat);

  // save and reload
  std::string saved = SaveAndReadXml(spec);
  mjSpec* spec2 =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec2, NotNull()) << error.data() << "\n" << saved;
  mjModel* model2 = mj_compile(spec2, nullptr);
  ASSERT_THAT(model2, NotNull());

  // the body was aligned, the frame did not move
  mjsBody* body2 = mjs_findBody(spec2, "body");
  mjsFrame* outer2 = mjs_findFrame(spec2, "outer");
  ASSERT_THAT(outer2, NotNull());
  EXPECT_GT(std::abs(body->pos[0] - body2->pos[0]), 0.1)
      << "body was not aligned";
  double pos2[3], quat2[4];
  MulPose(pos2, quat2, body2->pos, body2->quat, outer2->pos, outer2->quat);
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(pos2[i], pos[i], 1e-12) << saved;
  }
  double dot = 0;
  for (int i = 0; i < 4; i++) {
    dot += quat[i] * quat2[i];
  }
  EXPECT_NEAR(std::abs(dot), 1, 1e-12) << saved;

  // contents keep their authored frame-relative positions
  mjsGeom* geom = mjs_asGeom(mjs_findElement(spec2, mjOBJ_GEOM, "geom"));
  mjsSite* site = mjs_asSite(mjs_findElement(spec2, mjOBJ_SITE, "site"));
  mjsCamera* camera =
      mjs_asCamera(mjs_findElement(spec2, mjOBJ_CAMERA, "camera"));
  mjsLight* light = mjs_asLight(mjs_findElement(spec2, mjOBJ_LIGHT, "light"));
  const double geom_pos[3] = {.2, 0, 0}, site_pos[3] = {0, .3, 0};
  const double camera_pos[3] = {0, 0, .4}, light_pos[3] = {0, 0, .5},
               light_dir[3] = {1, 0, 0};
  for (int i = 0; i < 3; i++) {
    EXPECT_NEAR(geom->pos[i], geom_pos[i], 1e-12) << saved;
    EXPECT_NEAR(site->pos[i], site_pos[i], 1e-12) << saved;
    EXPECT_NEAR(camera->pos[i], camera_pos[i], 1e-12) << saved;
    EXPECT_NEAR(light->pos[i], light_pos[i], 1e-12) << saved;
    EXPECT_NEAR(light->dir[i], light_dir[i], 1e-12) << saved;
  }

  // the compiled model is unchanged
  std::string field;
  EXPECT_LT(CompareModel(model, model2, field), MjTol(1e-12, 1e-6)) << field;

  mj_deleteModel(model2);
  mj_deleteSpec(spec2);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a body's plugin is written once, not again inside each of the body's frames
TEST_F(XMLWriterTest, BodyPluginWrittenOnce) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.elasticity.cable"/>
    </extension>
    <worldbody>
      <body name="body">
        <joint type="ball"/>
        <geom type="capsule" size=".01" fromto="0 0 0 .1 0 0"/>
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="1e6"/>
          <config key="bend" value="1e6"/>
        </plugin>
        <frame name="frame" pos="0 0 1">
          <site name="site"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);

  // one plugin element inside the body, none inside the frame
  size_t body_begin = saved.find("<body name=\"body\"");
  size_t frame_begin = saved.find("<frame name=\"frame\"");
  size_t frame_end = saved.find("</frame>");
  ASSERT_NE(body_begin, std::string::npos) << saved;
  ASSERT_NE(frame_begin, std::string::npos) << saved;
  int nplugin = 0;
  for (size_t at = saved.find("<plugin", body_begin); at != std::string::npos;
       at = saved.find("<plugin", at + 1)) {
    nplugin++;
    EXPECT_FALSE(at > frame_begin && at < frame_end) << saved;
  }
  EXPECT_EQ(nplugin, 1) << saved;

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// an inertial nested in a frame is saved under the body, in body coordinates
TEST_F(XMLWriterTest, InertialInFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint/>
        <frame name="frame" pos="1 0 0" euler="0 0 90">
          <inertial pos=".5 0 0" euler="90 0 0" mass="1" diaginertia="2 3 4"/>
          <site name="site" pos="0 .5 0"/>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  FullFloatPrecision increase_precision;
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_NEAR(model->body_ipos[3], 1, MjTol(1e-14, 1e-6));
  EXPECT_NEAR(model->body_ipos[4], .5, MjTol(1e-14, 1e-6));

  // the inertial is written before the frame, which keeps its other contents
  std::string saved = SaveAndReadXml(spec);
  size_t inertial = saved.find("<inertial");
  size_t frame = saved.find("<frame name=\"frame\"");
  ASSERT_NE(inertial, std::string::npos) << saved;
  ASSERT_NE(frame, std::string::npos) << saved;
  EXPECT_LT(inertial, frame) << saved;

  // the compiled model is unchanged
  mjSpec* spec2 =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec2, NotNull()) << error.data() << "\n" << saved;
  mjModel* model2 = mj_compile(spec2, nullptr);
  ASSERT_THAT(model2, NotNull()) << mjs_getError(spec2);
  std::string field;
  EXPECT_LT(CompareModel(model, model2, field), MjTol(1e-12, 1e-6)) << field;

  mj_deleteModel(model2);
  mj_deleteSpec(spec2);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(XMLWriterTest, WritesDensity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05" density="100"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("density=\"100\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("mass")));
}

TEST_F(XMLWriterTest, WritesMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05" mass="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("density")));
  EXPECT_THAT(saved_xml, HasSubstr("mass=\"0.1\""));
}

TEST_F(XMLWriterTest, ZeroMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom type="box" size=".05 .05 .05" mass="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("density")));
  EXPECT_THAT(saved_xml, HasSubstr("mass=\"0\""));
}

TEST_F(XMLWriterTest, OverwritesDensity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="0.2" density="100" mass="100"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("density")));
  EXPECT_THAT(saved_xml, HasSubstr("mass=\"100\""));
}

TEST_F(XMLWriterTest, SaveDefaultMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="2 0 3  0 1 3  1 2 3  0 2 1" />
    </asset>
    <default class="main">
      <geom type="mesh" mass="1"/>
    </default>
    <worldbody>
      <body>
        <geom mesh="example" size=".1 .2 .3"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  EXPECT_THAT(model.get(), NotNull());
  std::string content = SaveAndReadXml(model.get());
  EXPECT_THAT(content, HasSubstr("mass=\"1\""));
}

TEST_F(XMLWriterTest, SaveMeshSite) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="2 0 3  0 1 3  1 2 3  0 2 1" />
    </asset>
    <worldbody>
      <body>
        <site name="s1" type="mesh" mesh="example"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  EXPECT_THAT(model.get(), NotNull());
  std::string content = SaveAndReadXml(model.get());
  EXPECT_THAT(content, HasSubstr("type=\"mesh\""));
  EXPECT_THAT(content, HasSubstr("mesh=\"example\""));
}

TEST_F(XMLWriterTest, UsesTwoSpaces) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("  "));
  EXPECT_THAT(saved_xml, Not(HasSubstr("    ")));
}

TEST_F(XMLWriterTest, WritesSkin) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 -1 .6" name="B0_parent">
        <flexcomp name="B0" type="grid" count="4 4 1" spacing=".2 .2 .2" radius=".1" dim="2"/>
      </body>
    </worldbody>
    <deformable>
      <skin name="B0Skin" rgba="1 1 0 1" inflate="0.1"
        vertex="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
        face="0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 4 8 9 4 9 5 5 9 10 5 10 6 6 10 11 6 11 7 8
              12 13 8 13 9 9 13 14 9 14 10 10 14 15 10 15 11 16 21 20 16 17 21 17 22 21 17 18
              22 18 23 22 18 19 23 20 25 24 20 21 25 21 26 25 21 22 26 22 27 26 22 23 27 24 29
              28 24 25 29 25 30 29 25 26 30 26 31 30 26 27 31 0 20 4 0 16 20 4 24 8 4 20 24 8
              28 12 8 24 28 3 7 23 3 23 19 7 11 27 7 27 23 11 15 31 11 31 27 0 1 17 0 17 16 1
              2 18 1 18 17 2 3 19 2 19 18 12 29 13 12 28 29 13 30 14 13 29 30 14 31 15 14 30 31">
        <bone body="B0_0" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 16" vertweight="1 1"/>
        <bone body="B0_1" bindpos="0 0 0" bindquat="1 0 0 0" vertid="1 17" vertweight="1 1"/>
        <bone body="B0_2" bindpos="0 0 0" bindquat="1 0 0 0" vertid="2 18" vertweight="1 1"/>
        <bone body="B0_3" bindpos="0 0 0" bindquat="1 0 0 0" vertid="3 19" vertweight="1 1"/>
        <bone body="B0_4" bindpos="0 0 0" bindquat="1 0 0 0" vertid="4 20" vertweight="1 1"/>
        <bone body="B0_5" bindpos="0 0 0" bindquat="1 0 0 0" vertid="5 21" vertweight="1 1"/>
        <bone body="B0_6" bindpos="0 0 0" bindquat="1 0 0 0" vertid="6 22" vertweight="1 1"/>
        <bone body="B0_7" bindpos="0 0 0" bindquat="1 0 0 0" vertid="7 23" vertweight="1 1"/>
        <bone body="B0_8" bindpos="0 0 0" bindquat="1 0 0 0" vertid="8 24" vertweight="1 1"/>
        <bone body="B0_9" bindpos="0 0 0" bindquat="1 0 0 0" vertid="9 25" vertweight="1 1"/>
        <bone body="B0_10" bindpos="0 0 0" bindquat="1 0 0 0" vertid="10 26" vertweight="1 1"/>
        <bone body="B0_11" bindpos="0 0 0" bindquat="1 0 0 0" vertid="11 27" vertweight="1 1"/>
        <bone body="B0_12" bindpos="0 0 0" bindquat="1 0 0 0" vertid="12 28" vertweight="1 1"/>
        <bone body="B0_13" bindpos="0 0 0" bindquat="1 0 0 0" vertid="13 29" vertweight="1 1"/>
        <bone body="B0_14" bindpos="0 0 0" bindquat="1 0 0 0" vertid="14 30" vertweight="1 1"/>
        <bone body="B0_15" bindpos="0 0 0" bindquat="1 0 0 0" vertid="15 31" vertweight="1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_THAT(model->nskin, 1);

  char error[1024];
  MjModelPtr mtemp =
      LoadModelFromString(SaveAndReadXml(model.get()), error, sizeof(error));
  ASSERT_THAT(mtemp.get(), NotNull()) << error;
  EXPECT_THAT(mtemp->nskin, 1);
}

TEST_F(XMLWriterTest, WritesPinnedFlexNodes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <flexcomp name="soft" type="box" count="3 3 3" spacing=".03 .01 .01" mass=".5" dof="trilinear">
          <contact selfcollide="none"/>
          <edge equality="true"/>
          <pin id="4 5 6 7"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  // pinned nodes have no body of their own: their coordinates in the parent
  // body frame must be saved or the interpolation grid degenerates on reload
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("nodecoord"));

  char error[1024];
  MjModelPtr mtemp = LoadModelFromString(saved_xml, error, sizeof(error));
  ASSERT_THAT(mtemp.get(), NotNull()) << error;
  EXPECT_EQ(SaveAndReadXml(mtemp.get()), saved_xml);
}

TEST_F(XMLWriterTest, KeepsEmptyFlexCells) {
  // no element uses the last point, which widens the grid: cells 1 and 2 are
  // empty and have no stiffness; the second flexcomp is in a rotated body
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <flexcomp name="test" type="direct" dim="3" dof="quadratic" mass="1" cellcount="3 1 1" point="0.0 0.0 0.0  0.3 0.0 0.0  0.0 1.0 0.0  0.3 1.0 0.0  0.0 0.0 1.0  0.3 0.0 1.0  0.0 1.0 1.0  0.3 1.0 1.0  1.0 0.5 0.5" element="0 1 3 2 4 5 7 6">
          <contact selfcollide="none"/>
          <elasticity young="1"/>
        </flexcomp>
      </body>
      <body name="rotated" euler="30 20 10">
        <flexcomp name="turned" type="direct" dim="3" dof="quadratic" mass="1" cellcount="3 1 1" point="0.0 0.0 0.0  0.3 0.0 0.0  0.0 1.0 0.0  0.3 1.0 0.0  0.0 0.0 1.0  0.3 0.0 1.0  0.0 1.0 1.0  0.3 1.0 1.0  1.0 0.5 0.5" element="0 1 3 2 4 5 7 6">
          <contact selfcollide="none"/>
          <elasticity young="1"/>
        </flexcomp>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  // the saved flexes find the same empty cells
  FullFloatPrecision increase_precision;
  MjModelPtr saved = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(saved.get(), NotNull());
  std::string field;
  EXPECT_EQ(CompareModel(model.get(), saved.get(), field), 0) << field;
}

TEST_F(XMLWriterTest, WritesHfield) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="hf" nrow="3" ncol="2" size="0.5 0.5 1 0.1"
              elevation="1 2
                         3 4
                         5 6"/>
    </asset>
  </mujoco>
  )";
  // load model
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  int size = model->hfield_nrow[0] * model->hfield_ncol[0];
  EXPECT_EQ(size, 6);

  // check that the data is normalized and in row-major, bottom-to-top order
  EXPECT_THAT(AsVector(model->hfield_data, 6),
              ElementsAre(.8, 1, .4, .6, 0, .2));

  // save and read, compare data
  MjModelPtr mtemp = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(mtemp.get(), NotNull());
  for (int i = 0; i < size; ++i) {
    EXPECT_THAT(mtemp->hfield_data[i], FloatEq(model->hfield_data[i]));
  }

  // modify data, save read and compare
  model->hfield_data[0] = 0.25;
  model->hfield_data[1] = 0.0;
  model->hfield_data[2] = 0.5;
  model->hfield_data[3] = 0.0;
  model->hfield_data[4] = 1.0;
  model->hfield_data[5] = 0.0;

  mtemp = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(mtemp.get(), NotNull());
  for (int i = 0; i < size; ++i) {
    EXPECT_THAT(mtemp->hfield_data[i], FloatEq(model->hfield_data[i]));
  }
}

TEST_F(XMLWriterTest, WritesLight) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="r1">
        <light bulbradius="1"/>
      </default>
    </default>
    <worldbody>
      <light/>
      <light class="r1"/>
      <light class="r1" bulbradius="2"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  // save and read, compare data
  MjModelPtr mtemp = LoadModelFromString(SaveAndReadXml(model.get()));
  EXPECT_EQ(mtemp->nlight, 3);
  EXPECT_FLOAT_EQ(mtemp->light_bulbradius[0], 0.02);
  EXPECT_FLOAT_EQ(mtemp->light_bulbradius[1], 1);
  EXPECT_FLOAT_EQ(mtemp->light_bulbradius[2], 2);
}

TEST_F(XMLWriterTest, WritesMaterial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="mat">
        <material metallic="2" roughness="3"/>
      </default>
    </default>
    <asset>
      <material name="0" class="mat"/>
      <material name="1" class="mat" metallic="4" roughness="5"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // save and read, compare data
  MjModelPtr mtemp = LoadModelFromString(SaveAndReadXml(model.get()));
  EXPECT_EQ(mtemp->nmat, 2);
  EXPECT_EQ(mtemp->mat_metallic[0], 2);
  EXPECT_EQ(mtemp->mat_metallic[1], 4);
  EXPECT_EQ(mtemp->mat_roughness[0], 3);
  EXPECT_EQ(mtemp->mat_roughness[1], 5);
}

TEST_F(XMLWriterTest, SpringlengthOneValue) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="1"/>
      <site name="2" pos="0 0 1"/>
    </worldbody>

    <tendon>
      <spatial springlength="0.5">
        <site site="1"/>
        <site site="2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("springlength=\"0.5\""));
}

TEST_F(XMLWriterTest, SpringlengthTwoValues) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="1"/>
      <site name="2" pos="0 0 1"/>
    </worldbody>

    <tendon>
      <spatial springlength="0 0.5">
        <site site="1"/>
        <site site="2"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("springlength=\"0 0.5\""));
}

TEST_F(XMLWriterTest, Actdim) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" dyntype="user" actdim="2"/>
      <general joint="hinge" dyntype="filter" dynprm="1"/>
      <motor joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("actdim=\"2\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("actdim=\"1\"")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("actdim=\"0\"")));
}

TEST_F(XMLWriterTest, ActdimDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <general actdim="2"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" dyntype="user"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("actdim=\"2\""));
}

TEST_F(XMLWriterTest, TrimsDefaults) {
#ifdef mjUSESINGLE
  GTEST_SKIP() << "float(0.9) != double(0.9), so default-trimming fails";
#endif
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" friction="0.9" solref="0.1" solimp="0.1 0.2"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="0 2"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" kp="3" kv="4"/>
      <damper joint="jnt" kv="10" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("friction=\"0.9\""));
  EXPECT_THAT(saved_xml, HasSubstr("solref=\"0.1\""));
  EXPECT_THAT(saved_xml, HasSubstr("solimp=\"0.1 0.2\""));
  EXPECT_THAT(saved_xml, HasSubstr("gainprm=\"3\" biasprm=\"0 -3 -4\""));
  EXPECT_THAT(saved_xml, HasSubstr("gainprm=\"0 0 -10\""));
}

TEST_F(XMLWriterTest, DoesntSaveGlobal) {
  static constexpr char xml[] = "<mujoco/>";
  std::array<char, 1000> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("global")));
}

TEST_F(XMLWriterTest, InheritrangeSavedAsRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="slide" type="slide" range="0 2"/>
      </body>
      <body>
        <geom size="1"/>
        <joint name="hinge" type="hinge" range="-2 0"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="slide" inheritrange="2"/>
      <intvelocity joint="hinge" inheritrange="0.5"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("ctrlrange=\"-1 3\""));
  EXPECT_THAT(saved_xml, HasSubstr("actrange=\"-1.5 -0.5\""));
}

TEST_F(XMLWriterTest, InheritedInheritrangeSavedAsRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="position">
        <position kp="3" kv="4" inheritrange="2"/>
      </default>
      <default class="intvelocity">
        <intvelocity kp="5" kv="6" inheritrange="0.5"/>
      </default>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" range="0 2"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" class="position"/>
      <intvelocity joint="jnt" class="intvelocity"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("inheritrange")));
  EXPECT_THAT(saved_xml, HasSubstr("ctrlrange=\"-1 3\""));
  EXPECT_THAT(saved_xml, HasSubstr("actrange=\"0.5 1.5\""));
}

// check that no precision is lost when saving XMLs with FullFloatPrecision
TEST_F(XMLWriterTest, SetPrecision) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" size="0.1 0.123456 0.1234567812345678"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  // save to XML and re-load, expect to lose precision
  MjModelPtr model_lo = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(model_lo.get(), NotNull());

  EXPECT_EQ(model->geom_size[1], model_lo->geom_size[1]);
  EXPECT_NE(model->geom_size[2], model_lo->geom_size[2]);
  {
    // save to XML and reload with FullFloatPrecision
    // expect to maintain precision
    FullFloatPrecision increase_precision;
    MjModelPtr model_hi = LoadModelFromString(SaveAndReadXml(model.get()));
    EXPECT_EQ(model->geom_size[2], model_hi->geom_size[2]);
  }
}

// with ExactFloatPrecision each number is saved as the shortest text which
// reads back as the same number, and a value is left out only if it is its
// default
TEST_F(XMLWriterTest, ExactPrecision) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" size="0.1 0.123456 0.1234567812345678" pos="1e-17 100 0.30000000000000004"/>
      <site pos="0.3333333333333333 1e+20 1.5e-07" rgba="0.1 0.33333334 16777216 1"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string exact;
  {
    ExactFloatPrecision exact_precision;
    exact = SaveAndReadXml(model.get());
    EXPECT_THAT(exact, HasSubstr(R"(size="0.1 0.123456 0.1234567812345678")"));
    EXPECT_THAT(exact, HasSubstr(R"(pos="1e-17 100 0.30000000000000004")"));
    EXPECT_THAT(exact, HasSubstr(R"(pos="0.3333333333333333 1e+20 1.5e-07")"));
    EXPECT_THAT(exact, HasSubstr(R"(rgba="0.1 0.33333334 16777216 1")"));
  }

  // the precision which was set before is restored
  std::string saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, HasSubstr(R"(size="0.1 0.123456 0.123457")"));
  EXPECT_THAT(saved, HasSubstr(R"(pos="0 100 0.3")"));

  // the exact numbers read back as they were
  MjModelPtr reloaded = LoadModelFromString(exact.c_str());
  ASSERT_THAT(reloaded.get(), NotNull());
  EXPECT_EQ(model->geom_size[2], reloaded->geom_size[2]);
  EXPECT_EQ(model->geom_pos[0], reloaded->geom_pos[0]);
}

// check that vector attributes keep their precision with FullFloatPrecision
TEST_F(XMLWriterTest, SetPrecisionVectors) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tetrahedron" vertex="0 0 0 0.123456789 0 0 0 0.123456789 0 0 0 0.123456789"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="tetrahedron"/>
      <flexcomp name="triangle" type="direct" dim="2" rigid="true" euler="30 20 10" point="0 0 0 1 0 0 0 1 0" element="0 1 2"/>
    </worldbody>
  </mujoco>
  )";
  FullFloatPrecision increase_precision;
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjModelPtr saved = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(saved.get(), NotNull());

  EXPECT_EQ(AsVector(saved->mesh_vert, 3 * saved->nmeshvert),
            AsVector(model->mesh_vert, 3 * model->nmeshvert));
  EXPECT_EQ(AsVector(saved->flex_vert, 3 * saved->nflexvert),
            AsVector(model->flex_vert, 3 * model->nflexvert));
}

class XMLWriterLocaleTest : public MujocoTest {
 public:
  XMLWriterLocaleTest() : old_locale_(std::setlocale(LC_ALL, nullptr)) {}

 protected:
  void SetUp() override {
    if (!std::setlocale(LC_ALL, "de_DE.UTF-8")) {
      GTEST_SKIP() << "This system doesn't support the de_DE.UTF-8 locale";
    }
  }

  void TearDown() override { std::setlocale(LC_ALL, old_locale_.c_str()); }

 private:
  std::string old_locale_;
};

TEST_F(XMLWriterLocaleTest, IgnoresLocale) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="box" size="0.1 1.23 2.345"/>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("0.1 1.23 2.345"));

  // Test that MuJoCo doesn't override locales for subsequent calls.
  char formatted[8];
  std::snprintf(formatted, sizeof(formatted), "%.4f", 3.9375);
  EXPECT_EQ(std::string(formatted), "3,9375");
}

TEST_F(XMLWriterTest, NonRGBTextures) {
  const std::string xml_path = GetTestDataFilePath(kNonRgbTextureXMLPath);
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();

  std::string saved_xml = SaveAndReadXml(model);
  EXPECT_FALSE(saved_xml.empty());

  // check that layers are written correctly
  EXPECT_THAT(saved_xml, HasSubstr("<material name=\"hfield\">"));
  EXPECT_THAT(saved_xml, HasSubstr("<layer texture=\"hfield\" role=\"rgb\"/>"));
  EXPECT_THAT(saved_xml, HasSubstr("<layer texture=\"hfield\" role=\"orm\"/>"));
  EXPECT_THAT(saved_xml,
              HasSubstr("<layer texture=\"hfield\" role=\"normal\"/>"));

  mj_deleteModel(model);
}

// the number of channels which a texture file is read with is saved
TEST_F(XMLWriterTest, KeepsTextureChannels) {
  const std::string xml_path = GetTestDataFilePath(kNonRgbTextureXMLPath);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXML(xml_path.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjsTexture* texture =
      mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, "hfield"));
  ASSERT_THAT(texture, NotNull());
  texture->nchannel = 4;
  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->tex_nchannel[0], 4);

  EXPECT_THAT(SaveAndReadXml(spec), HasSubstr("nchannel=\"4\""));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// texture defaults are inherited through classes and survive a round trip,
// the writer saving only what differs from the parent class
TEST_F(XMLWriterTest, TextureDefaultsRoundTrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="c">
        <texture type="2d" builtin="checker" rgb1="0.1 0.2 0.3" mark="edge"
          markrgb="1 0 0" width="4" height="4"/>
        <default class="d">
          <texture rgb1="0.9 0.8 0.7"/>
        </default>
      </default>
    </default>
    <asset>
      <texture name="a" class="c"/>
      <texture name="b" class="d" height="8"/>
      <texture name="e" type="2d" builtin="flat" width="2" height="2"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  auto check = [](mjSpec* spec) {
    mjsTexture* a = mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, "a"));
    mjsTexture* b = mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, "b"));
    mjsTexture* e = mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, "e"));
    ASSERT_THAT(a, NotNull());
    ASSERT_THAT(b, NotNull());
    ASSERT_THAT(e, NotNull());
    EXPECT_EQ(a->type, mjTEXTURE_2D);
    EXPECT_EQ(a->builtin, mjBUILTIN_CHECKER);
    EXPECT_EQ(a->mark, mjMARK_EDGE);
    EXPECT_THAT(a->rgb1, ElementsAre(0.1, 0.2, 0.3));
    EXPECT_THAT(a->markrgb, ElementsAre(1, 0, 0));
    EXPECT_EQ(a->width, 4);
    EXPECT_EQ(a->height, 4);
    EXPECT_EQ(b->builtin, mjBUILTIN_CHECKER);
    EXPECT_THAT(b->rgb1, ElementsAre(0.9, 0.8, 0.7));
    EXPECT_EQ(b->height, 8);
    EXPECT_EQ(e->builtin, mjBUILTIN_FLAT);
    EXPECT_THAT(e->rgb1, ElementsAre(0.8, 0.8, 0.8));
    mjsDefault* c = mjs_findDefault(spec, "c");
    ASSERT_THAT(c, NotNull());
    EXPECT_EQ(c->texture->builtin, mjBUILTIN_CHECKER);
    EXPECT_EQ(mjs_findDefault(spec, "d")->texture->rgb1[0], 0.9);
    EXPECT_EQ(mjs_getSpecDefault(spec)->texture->builtin, mjBUILTIN_NONE);
  };
  check(spec);

  mjModel* model = mj_compile(spec, 0);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->tex_type[0], mjTEXTURE_2D);
  EXPECT_EQ(model->tex_height[1], 8);

  // the class is saved, and each level writes only what it sets
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr("<texture type=\"2d\" colorspace=\"auto\" "
                               "name=\"a\" class=\"c\"/>"));
  EXPECT_THAT(saved, HasSubstr("<texture type=\"2d\" colorspace=\"auto\" "
                               "name=\"b\" class=\"d\" height=\"8\"/>"));
  EXPECT_THAT(saved,
              HasSubstr("<default class=\"c\">\n      <texture type=\"2d\" "
                        "builtin=\"checker\" rgb1=\"0.1 0.2 0.3\" "
                        "mark=\"edge\" markrgb=\"1 0 0\" width=\"4\" "
                        "height=\"4\"/>"));
  EXPECT_THAT(saved, HasSubstr("<default class=\"d\">\n        <texture "
                               "rgb1=\"0.9 0.8 0.7\"/>"));

  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), 0, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data();
  check(reloaded);

  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// defaults which hold for a file texture, and an inherited flag turned off
TEST_F(XMLWriterTest, TextureDefaultsFile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="f">
        <texture type="2d" nchannel="4" hflip="true" vflip="true"/>
      </default>
    </default>
    <asset>
      <texture name="t" class="f" file="hfield.png"/>
      <texture name="u" class="f" file="hfield.png" vflip="false"/>
    </asset>
  </mujoco>
  )";
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  ASSERT_EQ(
      mj_addFileVFS(vfs.get(), "",
                    GetTestDataFilePath("xml/testdata/hfield.png").c_str()),
      0);
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, vfs.get(), error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, vfs.get());
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->tex_type[0], mjTEXTURE_2D);
  EXPECT_EQ(model->tex_nchannel[0], 4);
  EXPECT_EQ(model->tex_nchannel[1], 4);

  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr("<texture type=\"2d\" hflip=\"true\" "
                               "vflip=\"true\" nchannel=\"4\"/>"));
  EXPECT_THAT(saved, HasSubstr("name=\"t\" class=\"f\" file=\"hfield.png\"/>"));
  EXPECT_THAT(saved, HasSubstr("name=\"u\" class=\"f\" file=\"hfield.png\" "
                               "vflip=\"false\"/>"));

  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), vfs.get(), error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data();
  mjsTexture* u = mjs_asTexture(mjs_findElement(reloaded, mjOBJ_TEXTURE, "u"));
  ASSERT_THAT(u, NotNull());
  EXPECT_TRUE(u->hflip);
  EXPECT_FALSE(u->vflip);
  EXPECT_EQ(u->nchannel, 4);

  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

// attributes which identify the file cannot be defaults
TEST_F(XMLWriterTest, TextureDefaultsRejectFile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <texture file="hfield.png"/>
    </default>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(), HasSubstr("file"));
}

// a model without texture defaults is saved as before
TEST_F(XMLWriterTest, TextureWithoutDefaultsUnchanged) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="checker" type="2d" builtin="checker" rgb1="0.1 0.2 0.3"
        width="8" height="8"/>
      <texture name="sky" type="skybox" builtin="gradient" rgb2="0 0 1"
        width="4" height="4" mark="cross"/>
    </asset>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, Not(HasSubstr("<default")));
  EXPECT_THAT(saved, Not(HasSubstr("class=")));
  EXPECT_THAT(saved, HasSubstr("<texture type=\"2d\" colorspace=\"auto\" "
                               "name=\"checker\" builtin=\"checker\" "
                               "rgb1=\"0.1 0.2 0.3\" width=\"8\" "
                               "height=\"8\"/>"));
  EXPECT_THAT(saved, HasSubstr("<texture type=\"skybox\" colorspace=\"auto\" "
                               "name=\"sky\" builtin=\"gradient\" "
                               "mark=\"cross\" rgb2=\"0 0 1\" width=\"4\" "
                               "height=\"24\"/>"));
}

// ---------------- test CopyBack functionality (decompiler) ------------------
using DecompilerTest = MujocoTest;
TEST_F(DecompilerTest, SavesStatistics) {
  static constexpr char xml[] = R"(
  <mujoco>
    <statistic meansize="2" extent="3" center="4 5 6" meanmass="7" meaninertia="8"/>
  </mujoco>
  )";
  mock_warning_handler.ExpectWarnings("meaninertia");
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  model->stat.meansize = 7;
  model->stat.extent = 8;
  model->stat.center[0] = 9;
  model->stat.center[1] = 10;
  model->stat.center[2] = 11;
  model->stat.meanmass = 12;
  model->stat.meaninertia = 13;
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("meansize=\"7\""));
  EXPECT_THAT(saved_xml, HasSubstr("extent=\"8\""));
  EXPECT_THAT(saved_xml, HasSubstr("center=\"9 10 11\""));
  EXPECT_THAT(saved_xml, HasSubstr("meanmass=\"12\""));
  EXPECT_THAT(saved_xml, HasSubstr("meaninertia=\"13\""));
}

TEST_F(DecompilerTest, SaveAndReadXml) {
  static constexpr char xml1[] = R"(
  <mujoco>
    <worldbody>
      <geom size="1"/>
      <geom size="2"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml2[] = R"(
  <mujoco>
    <worldbody>
      <geom size="1"/>
      <geom size="2"/>
      <geom size="3"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m1 = LoadModelFromString(xml1, error.data(), error.size());
  ASSERT_THAT(m1.get(), NotNull()) << error.data();
  m1->geom_size[0] = 10;
  m1->geom_size[3] = 20;
  std::string saved_xml = SaveAndReadXml(m1.get());
  EXPECT_THAT(saved_xml, HasSubstr("geom size=\"10\""));
  EXPECT_THAT(saved_xml, HasSubstr("geom size=\"20\""));

  // parse the mjSpec, save it and read it back
  mjSpec* spec = mj_parseXMLString(xml2, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  mjModel* m2 = mj_compile(spec, nullptr);
  std::string saved_xml1 = SaveAndReadXml(spec);
  EXPECT_THAT(saved_xml1, HasSubstr("geom size=\"1\""));
  EXPECT_THAT(saved_xml1, HasSubstr("geom size=\"2\""));
  EXPECT_THAT(saved_xml1, HasSubstr("geom size=\"3\""));

  // modify the mjModel, save it and read it back
  m2->geom_size[0] = .1;
  m2->geom_size[3] = .2;
  m2->geom_size[6] = .3;
  EXPECT_EQ(mj_copyBack(spec, m1.get()), 0);
  EXPECT_THAT(mjs_getError(spec), HasSubstr("CopyBack"));
  EXPECT_EQ(mj_copyBack(spec, m2), 1);
  std::string saved_xml2 = SaveAndReadXml(spec);
  EXPECT_THAT(saved_xml2, HasSubstr("geom size=\"0.1\""));
  EXPECT_THAT(saved_xml2, HasSubstr("geom size=\"0.2\""));
  EXPECT_THAT(saved_xml2, HasSubstr("geom size=\"0.3\""));

  // a model which was not compiled from the XML which was loaded last is not
  // saved with it
  EXPECT_EQ(mj_saveLastXML(nullptr, m2, error.data(), error.size()), 0);
  EXPECT_THAT(error.data(), HasSubstr("CopyBack"));

  mj_deleteSpec(spec);
  mj_deleteModel(m2);
}

// the reference pose of a body with a free joint is qpos0, which is where the
// simulation reads it
TEST_F(DecompilerTest, FreeJointReferencePose) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="free" pos="1 2 3">
        <freejoint/>
        <geom size="1"/>
      </body>
      <body name="fixed" pos="5 6 7">
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  model->qpos0[2] = 4;
  model->body_pos[3 * 2 + 2] = 8;
  std::string saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, HasSubstr("name=\"free\" pos=\"1 2 4\""));
  EXPECT_THAT(saved, HasSubstr("name=\"fixed\" pos=\"5 6 8\""));

  // the pose of the body is used if qpos0 was not changed
  model->body_pos[3 * 1 + 2] = 9;
  saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, HasSubstr("name=\"free\" pos=\"1 2 9\""));
}

// a numeric with less data than its size is saved, and copied back, with the
// zeros which fill it in the model; the XML parser fills it, the API does not
TEST_F(DecompilerTest, NumericLargerThanData) {
  mjSpec* spec = mj_makeSpec();
  mjsNumeric* numeric = mjs_addNumeric(spec);
  mjs_setName(numeric->element, "padded");
  numeric->size = 5;
  const double data[2] = {1, 2};
  mjs_setDouble(numeric->data, data, 2);
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(SaveAndReadXml(spec), HasSubstr("data=\"1 2 0 0 0\""));

  model->numeric_data[4] = 7;
  EXPECT_EQ(mj_copyBack(spec, model), 1);
  EXPECT_THAT(SaveAndReadXml(spec), HasSubstr("data=\"1 2 0 0 7\""));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(DecompilerTest, DoesntSaveInferredStatistics) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="0.2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("meansize")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("meanmass")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("meaninertia")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("center")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("extent")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("statistic")));
}

TEST_F(DecompilerTest, VeryLargeNumbers) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>
    <worldbody>
      <camera focal="16777217 1" sensorsize="1 1" resolution="100 100"/>
      <body pos="1e+20 0 0">
        <geom size="1"/>
        <joint axis="1 0 0" range="-1e+10 1e+10"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  std::string saved_xml = SaveAndReadXml(model.get());
  // note, focal is float and loses precision 16777217 -> 16777216
  EXPECT_THAT(saved_xml, HasSubstr("focal=\"16777216 1\""));
  EXPECT_THAT(saved_xml, HasSubstr("pos=\"1e+20 0 0\""));
  EXPECT_THAT(saved_xml, HasSubstr("range=\"-1e+10 1e+10\""));
}

TEST_F(XMLWriterTest, ExpandAttach) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <asset>
      <model name="b" file="b.xml" />
    </asset>

    <worldbody>
      <attach model="b" body="b" prefix="b" />
    </worldbody>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco>
    <default>
      <default class="b"/>
    </default>

    <worldbody>
      <body name="b">
        <geom type="box" size="0.1 0.1 0.1"/>
        <joint name="b"/>
      </body>
    </worldbody>

    <actuator>
      <position joint="b" class="b"/>
    </actuator>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "b.xml", xml_child, sizeof(xml_child));

  std::array<char, 1024> er;
  MjModelPtr m =
      LoadModelFromString(xml_parent, er.data(), er.size(), vfs.get());
  ASSERT_THAT(m.get(), NotNull()) << er.data();

  std::string saved_xml = SaveAndReadXml(m.get());
  EXPECT_THAT(saved_xml, HasSubstr("class=\"bb\""));
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLWriterTest, WritesCameraOutput) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="default_cam"/>
      <camera name="multi_cam" output="depth normal"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  // default output="rgb" should not be written
  EXPECT_THAT(saved_xml, Not(HasSubstr("output=\"rgb\"")));
  // non-default output should be written
  EXPECT_THAT(saved_xml, HasSubstr("output=\"depth normal\""));
}

TEST_F(XMLWriterTest, WritesCameraOutputDefault) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="multi">
        <camera output="depth distance"/>
      </default>
    </default>
    <worldbody>
      <camera name="default_cam"/>
      <camera name="class_cam" class="multi"/>
      <camera name="override_cam" class="multi" output="segmentation"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  // save and reload to verify round-trip
  MjModelPtr mtemp = LoadModelFromString(saved_xml);
  ASSERT_THAT(mtemp.get(), NotNull());
  EXPECT_EQ(mtemp->ncam, 3);
  EXPECT_EQ(mtemp->cam_output[0], mjCAMOUT_RGB);
  EXPECT_EQ(mtemp->cam_output[1], mjCAMOUT_DEPTH | mjCAMOUT_DIST);
  EXPECT_EQ(mtemp->cam_output[2], mjCAMOUT_SEG);
}

// the energy sensors are saved under the names which the reader knows
TEST_F(XMLWriterTest, WritesEnergySensors) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <geom size="1"/>
      </body>
    </worldbody>

    <sensor>
      <e_potential name="potential"/>
      <e_kinetic name="kinetic"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, HasSubstr("<e_potential name=\"potential\"/>"));
  EXPECT_THAT(saved, HasSubstr("<e_kinetic name=\"kinetic\"/>"));
  std::array<char, 1024> error;
  MjModelPtr reloaded =
      LoadModelFromString(saved.c_str(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
  EXPECT_EQ(reloaded->nsensor, 2);
}

TEST_F(XMLWriterTest, WritesSensorDelayIntervalHistory) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" delay="0.05" interval="0.1 -0.02" nsample="20" interp="linear"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("delay=\"0.05\""));
  EXPECT_THAT(saved_xml, HasSubstr("interval=\"0.1 -0.02\""));
  EXPECT_THAT(saved_xml, HasSubstr("nsample=\"20\""));
  EXPECT_THAT(saved_xml, HasSubstr("interp=\"linear\""));
}

TEST_F(XMLWriterTest, DoesNotWriteDefaultSensorAttributes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("delay=")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("interval=")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("nsample=")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("interp=")));
}

TEST_F(XMLWriterTest, WritesActuatorDelayHistory) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="slide" delay="0.03" nsample="15" interp="cubic"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("delay=\"0.03\""));
  EXPECT_THAT(saved_xml, HasSubstr("nsample=\"15\""));
  EXPECT_THAT(saved_xml, HasSubstr("interp=\"cubic\""));
}

TEST_F(XMLWriterTest, BodySimpleRoundtrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1" simple="false">
        <joint type="ball"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("simple=\"false\""));

  // auto should not be written
  static constexpr char xml_auto[] = R"(
  <mujoco>
    <worldbody>
      <body name="B1">
        <joint type="ball"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model_auto = LoadModelFromString(xml_auto);
  ASSERT_THAT(model_auto.get(), NotNull());
  std::string saved_auto = SaveAndReadXml(model_auto.get());
  EXPECT_THAT(saved_auto, Not(HasSubstr("simple=")));

  // nC should increase when simple is disabled
  EXPECT_GT(model->nC, model_auto->nC);
}

// ------------------------ saving a spec as it was written -------------------

using AsWrittenTest = MujocoTest;

// a model in the order and the format of the writer
static constexpr char kWritten[] = R"(<mujoco model="written">
  <compiler autolimits="true" angle="degree" eulerseq="zyx"/>

  <option timestep="0.002" gravity="0 0 -5">
    <flag gravity="enable" energy="enable"/>
  </option>

  <size memory="1M"/>

  <statistic extent="2"/>

  <visual>
    <global azimuth="90"/>
  </visual>

  <default>
    <geom size="0.1" euler="90 0 0"/>
    <default class="arm">
      <joint range="-45 45" damping="0.5"/>
    </default>
  </default>

  <asset>
    <texture type="2d" name="checker" builtin="checker" rgb1="0.1 0.2 0.3" rgb2="0.2 0.3 0.4" width="8" height="8"/>
    <material name="floor" texture="checker" texrepeat="2 2"/>
    <mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1" scale="2 2 2"/>
  </asset>

  <worldbody>
    <geom name="floor" size="0 0 0.1" type="plane" material="floor"/>
    <frame name="tilted" pos="0 0 1" axisangle="0 1 0 30">
      <geom name="capsule" fromto="0 0 0 0 0 1" type="capsule"/>
      <body name="link" childclass="arm" pos="0.1 0.2 0.30000000000000004" euler="10 20 30">
        <inertial pos="0 0 0.1" mass="2" fullinertia="1 1 1 0.1 0 0"/>
        <joint name="hinge" axis="0 1 0"/>
        <joint name="slide" type="slide" range="-1 1" ref="0.5"/>
        <geom name="ball" quat="1 0 0 0"/>
        <geom name="fitted" type="box" mesh="tetra"/>
        <site name="tip" pos="0 0 0.2" zaxis="1 0 0"/>
        <camera name="eye" xyaxes="1 0 0 0 0 1"/>
      </body>
    </frame>
    <body name="floating" pos="0 0 2">
      <freejoint name="free" align="false"/>
      <geom name="box" size="0.1 0.2 0.3" type="box" mass="1"/>
    </body>
  </worldbody>

  <contact>
    <exclude body1="link" body2="floating"/>
  </contact>

  <tendon>
    <fixed name="coupled" springlength="0.1 0.2">
      <joint joint="hinge" coef="1"/>
      <joint joint="slide" coef="-2"/>
    </fixed>
  </tendon>

  <equality>
    <weld body1="link" body2="floating"/>
  </equality>

  <actuator>
    <general name="servo" joint="hinge" biastype="affine" gainprm="10" biasprm="0 -10 1"/>
  </actuator>

  <sensor>
    <jointpos joint="hinge" name="angle"/>
  </sensor>

  <custom>
    <numeric name="gains" size="3" data="1 2 0"/>
  </custom>

  <keyframe>
    <key name="start" qpos="0.5 0.25 0 0 2 1 0 0 0"/>
    <key/>
  </keyframe>
</mujoco>
)";

// parse a model and ask for it to be saved as it is written
static mjSpec* ParseAsWritten(const char* xml, bool canonical = false) {
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  if (spec) {
    spec->compiler.savecompiled = 0;
    spec->compiler.savecanonical = canonical;
  }
  return spec;
}

// the spec as it is saved to a string; the error if it cannot be saved
static std::string SaveToString(const mjSpec* spec,
                                std::string* error = nullptr) {
  std::array<char, 1024> err = {0};
  int size = mj_saveXMLString(spec, nullptr, 0, err.data(), err.size());
  if (error) *error = err.data();
  if (size <= 0) return "";
  std::string xml(size + 1, '\0');
  mj_saveXMLString(spec, xml.data(), size + 1, err.data(), err.size());
  xml.resize(size);
  return xml;
}

// an element type which the writer does not support is a write error, not an
// mju_error which would longjmp past the writer's C++ objects (and, from
// mj_saveLastXML, its lock); leaks are detected when running under asan
TEST_F(AsWrittenTest, UnknownTypeIsWriteError) {
  mjSpec* spec = mj_makeSpec();
  spec->compiler.savecompiled = 0;
  mjs_setString(spec->modelname, "a model name longer than the sso buffer");
  mjs_addEquality(spec, nullptr)->type = mjEQ_DISTANCE;
  std::array<char, 1024> error = {0};
  auto save = MjuErrorMessageFrom(mj_saveXMLString);
  EXPECT_EQ(save(spec, nullptr, 0, error.data(), error.size()), "");
  EXPECT_THAT(error.data(), HasSubstr("unknown equality type"));
  mj_deleteSpec(spec);
}

// every attribute is saved as it was written, also one which was written with
// its default value, and numbers are saved exactly
TEST_F(AsWrittenTest, SavesWhatWasWritten) {
  mjSpec* spec = ParseAsWritten(kWritten);
  ASSERT_THAT(spec, NotNull());

  // the spec need not be compiled
  EXPECT_EQ(SaveAndReadXml(spec), kWritten);

  // compiling it does not change what is saved
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(SaveAndReadXml(spec), kWritten);

  // the values which compilation made of the model are saved if asked for
  spec->compiler.savecompiled = 1;
  std::string compiled = SaveAndReadXml(spec);
  EXPECT_THAT(compiled, HasSubstr("angle=\"radian\""));
  EXPECT_THAT(compiled, Not(HasSubstr("euler")));
  EXPECT_THAT(compiled, Not(HasSubstr("fromto")));
  EXPECT_THAT(compiled, HasSubstr("<geom name=\"fitted\" size="));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a frame keeps its scale and scaling when saved as written, one value if the
// scale is uniform; saved compiled, the scale is baked into the values
TEST_F(AsWrittenTest, SavesFrameScale) {
  static constexpr char xml[] = R"(
<mujoco>
  <worldbody>
    <frame name="uniform" scale="2">
      <geom size="0.1"/>
    </frame>
    <frame name="vector" scale="1 2 3" scaling="geometry"/>
    <frame name="unit" pos="1 0 0" scale="1"/>
    <frame name="mass" massscale="2"/>
  </worldbody>
</mujoco>
)";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  std::string saved = SaveToString(spec);
  EXPECT_THAT(saved, HasSubstr(R"(<frame name="uniform" scale="2">)"));
  EXPECT_THAT(
      saved,
      HasSubstr(R"(<frame name="vector" scale="1 2 3" scaling="geometry"/>)"));
  EXPECT_THAT(saved, HasSubstr(R"(<frame name="unit" pos="1 0 0"/>)"));
  EXPECT_THAT(saved, HasSubstr(R"(<frame name="mass" massscale="2"/>)"));

  // reloaded, the frames have the same scale and scaling
  std::array<char, 1024> error;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data();
  for (const char* name : {"uniform", "vector", "unit", "mass"}) {
    mjsFrame* f1 = mjs_findFrame(spec, name);
    mjsFrame* f2 = mjs_findFrame(reloaded, name);
    EXPECT_THAT(f2->scale, ElementsAreArray(f1->scale)) << name;
    EXPECT_EQ(f2->scaling, f1->scaling) << name;
    EXPECT_EQ(f2->massscale, f1->massscale) << name;
  }
  mj_deleteSpec(reloaded);

  // saved compiled, neither attribute is written
  mjs_delete(spec, mjs_findFrame(spec, "vector")->element);
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  spec->compiler.savecompiled = 1;
  saved = SaveToString(spec);
  EXPECT_THAT(saved, Not(HasSubstr("scale")));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// in the canonical notation, what was written is saved as quaternions, radians,
// sizes with poses, and principal moments of inertia
TEST_F(AsWrittenTest, SavesCanonicalNotation) {
  mjSpec* spec = ParseAsWritten(kWritten, /*canonical=*/true);
  ASSERT_THAT(spec, NotNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved,
              HasSubstr("<compiler autolimits=\"true\" angle=\"radian\"/>"));
  for (const char* notation :
       {"euler", "axisangle", "xyaxes", "zaxis", "fromto", "fullinertia"}) {
    EXPECT_THAT(saved, Not(HasSubstr(notation))) << saved;
  }
  EXPECT_THAT(
      saved,
      HasSubstr(
          R"(<joint range="-0.7853981633974483 0.7853981633974483" damping="0.5"/>)"));
  EXPECT_THAT(
      saved,
      HasSubstr(R"(<geom name="capsule" size="0.1 0.5" pos="0 0 0.5" quat=)"));
  EXPECT_THAT(saved, HasSubstr(R"(<inertial pos="0 0 0.1" quat=)"));
  EXPECT_THAT(saved, HasSubstr("diaginertia="));

  // the rest is as it was written
  EXPECT_THAT(saved,
              HasSubstr(R"(<option timestep="0.002" gravity="0 0 -5">)"));
  EXPECT_THAT(saved, HasSubstr(R"(pos="0.1 0.2 0.30000000000000004")"));
  EXPECT_THAT(saved,
              HasSubstr(R"(<geom name="fitted" type="box" mesh="tetra"/>)"));

  // it is the same model
  std::array<char, 1024> error;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjModel* remodel = mj_compile(reloaded, nullptr);
  ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
  std::string field;
  EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

  mj_deleteModel(remodel);
  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// what is copied back from a model is saved in the notation of the spec
TEST_F(AsWrittenTest, SavesWhatWasCopiedBack) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame pos="1 0 0">
        <body name="body" pos="1 0 0">
          <joint name="hinge" range="-90 90"/>
          <geom name="geom" fromto="0 0 0 0 0 1" size="0.5" type="capsule"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  // move the body, halve the upper limit, make the capsule thinner
  model->body_pos[3] = 4;
  model->jnt_range[1] /= 2;
  model->geom_size[0] = 0.25;
  ASSERT_EQ(mj_copyBack(spec, model), 1) << mjs_getError(spec);
  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr(R"(<body name="body" pos="3 0 0">)"));
  EXPECT_THAT(
      saved,
      HasSubstr(R"(<geom name="geom" size="0.25" fromto="0 0 0 0 0 1")"));
  EXPECT_THAT(saved, Not(HasSubstr("angle=")));

  // the limit is in degrees, to roundoff
  std::array<char, 1024> error;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjsJoint* hinge =
      mjs_asJoint(mjs_findElement(reloaded, mjOBJ_JOINT, "hinge"));
  ASSERT_THAT(hinge, NotNull());
  EXPECT_EQ(hinge->range[0], -90);
  EXPECT_NEAR(hinge->range[1], 45, MjTol(1e-12, 1e-5));

  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// an inertial which compilation does not use is saved as it was written
TEST_F(AsWrittenTest, SavesUnusedInertial) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler inertiafromgeom="true"/>
    <worldbody>
      <body>
        <inertial pos="0 1 2" mass="3"/>
        <geom size="0.2"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  EXPECT_THAT(SaveAndReadXml(spec),
              HasSubstr(R"(<inertial pos="0 1 2" mass="3"/>)"));
  mj_deleteSpec(spec);
}

// with saveinertial, the inertial which compilation calculated is saved with
// the mass before settotalmass scaled it, as the inertials which the spec
// gives: the saved file scales them all again
TEST_F(AsWrittenTest, SavesCalculatedInertialBeforeTotalMass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler settotalmass="4" saveinertial="true"/>
    <worldbody>
      <body name="given">
        <joint/>
        <inertial pos="0 0 0" mass="1" diaginertia=".5 .5 .5"/>
      </body>
      <body name="inferred" pos="1 0 0">
        <joint/>
        <geom size=".5" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  mock_warning_handler.ExpectWarnings("settotalmass");
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->body_mass[1], 2);
  EXPECT_EQ(model->body_mass[2], 2);

  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr(R"(settotalmass="4")"));
  EXPECT_THAT(saved, HasSubstr(R"(<inertial pos="0 0 0" mass="1" )"));
  std::array<char, 1024> error;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjModel* remodel = mj_compile(reloaded, nullptr);
  ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
  std::string field;
  EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

  mj_deleteModel(remodel);
  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a range which an actuator inherits stays inherited, in the actuator and in
// its class
TEST_F(AsWrittenTest, SavesInheritedRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="inherits">
        <position inheritrange="0.5"/>
      </default>
    </default>

    <worldbody>
      <body>
        <joint name="slide" type="slide" range="0 2"/>
        <geom size="1"/>
      </body>
    </worldbody>

    <actuator>
      <position joint="slide" inheritrange="2"/>
      <position joint="slide" class="inherits"/>
      <intvelocity joint="slide" inheritrange="1"/>
    </actuator>
  </mujoco>
  )";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  std::string saved = SaveAndReadXml(spec);
  EXPECT_THAT(saved, HasSubstr(R"(<position inheritrange="0.5")"));
  EXPECT_THAT(saved, HasSubstr(R"(<position joint="slide" inheritrange="2")"));
  EXPECT_THAT(saved,
              HasSubstr(R"(<position class="inherits" joint="slide"/>)"));
  EXPECT_THAT(saved,
              HasSubstr(R"(<intvelocity joint="slide" inheritrange="1")"));
  EXPECT_THAT(saved, Not(HasSubstr("ctrlrange")));
  EXPECT_THAT(saved, Not(HasSubstr("actrange")));

  std::array<char, 1024> error;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjModel* remodel = mj_compile(reloaded, nullptr);
  ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
  std::string field;
  EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

  // the values which compilation made have the ranges
  spec->compiler.savecompiled = 1;
  std::string compiled = SaveAndReadXml(spec);
  EXPECT_THAT(compiled, HasSubstr(R"(ctrlrange="-1 3")"));
  EXPECT_THAT(compiled, HasSubstr(R"(actrange="0 2")"));
  EXPECT_THAT(compiled, Not(HasSubstr("inheritrange")));

  mj_deleteModel(remodel);
  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// an actuator is saved with the shortcut it was written with, in both modes
TEST_F(AsWrittenTest, SavesShortcuts) {
  static constexpr char xml[] = R"(<mujoco model="MuJoCo Model">
  <worldbody>
    <body name="b">
      <joint name="j" type="slide" range="0 2"/>
      <joint name="ball" type="ball"/>
      <geom size="1"/>
    </body>
  </worldbody>

  <actuator>
    <motor joint="j"/>
    <position joint="j" kp="3" kv="4"/>
    <position joint="j" kp="3" dampratio="0.5" timeconst="0.1"/>
    <velocity joint="j" kv="2"/>
    <intvelocity joint="j" actrange="-1 1" kp="5"/>
    <damper joint="j" ctrlrange="0 1" kv="10"/>
    <cylinder joint="j" timeconst="2" area="3" bias="1 2 3"/>
    <muscle joint="j" lengthrange="0 1" force="100"/>
    <adhesion body="b" ctrlrange="0 1" gain="2"/>
    <pid joint="j" kp="2" ki="1" imax="3"/>
    <orientation joint="ball" kp="2"/>
    <dcmotor joint="j" motorconst="0.5" resistance="2"/>
  </actuator>
</mujoco>
)";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  EXPECT_EQ(SaveAndReadXml(spec), xml);
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(SaveAndReadXml(spec), xml);

  // the canonical notation, which compiled values are always saved in, has
  // every actuator as general; a damping ratio which compilation resolved into
  // a gain is written rounded
  for (bool compiled : {false, true}) {
    spec->compiler.savecompiled = compiled;
    spec->compiler.savecanonical = 1;
    std::string canonical = SaveAndReadXml(spec);
    for (const char* tag :
         {"motor", "position", "velocity", "intvelocity", "damper", "cylinder",
          "muscle", "adhesion", "pid", "orientation", "dcmotor"}) {
      EXPECT_THAT(canonical, Not(HasSubstr(std::string("<") + tag + " ")))
          << tag;
    }
    EXPECT_THAT(canonical,
                HasSubstr(R"(<general joint="j" input="pos" biastype="affine" )"
                          R"(gainprm="3" biasprm="0 -3 -4"/>)"));

    std::array<char, 1024> error;
    mjSpec* reloaded = mj_parseXMLString(canonical.c_str(), nullptr,
                                         error.data(), error.size());
    ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << canonical;
    mjModel* remodel = mj_compile(reloaded, nullptr);
    ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
    std::string field;
    EXPECT_LE(CompareModel(model, remodel, field), compiled ? 1e-6 : 0)
        << field;
    mj_deleteModel(remodel);
    mj_deleteSpec(reloaded);
  }

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a class is saved with the shortcut of its actuator default; a child which
// only repeats the shortcut of its parent is left out
TEST_F(AsWrittenTest, SavesShortcutDefaults) {
  static constexpr char xml[] = R"(<mujoco model="MuJoCo Model">
  <default>
    <default class="servo">
      <position kp="5" kv="1"/>
      <default class="fast">
        <position kp="50"/>
      </default>
      <default class="vel">
        <velocity kv="2"/>
      </default>
      <default class="mot">
        <motor/>
      </default>
      <default class="same">
        <position/>
      </default>
    </default>
  </default>

  <worldbody>
    <body>
      <joint name="j" type="slide" range="0 2"/>
      <geom size="1"/>
    </body>
  </worldbody>

  <actuator>
    <position class="fast" joint="j"/>
    <position class="servo" joint="j" kv="0"/>
    <position class="servo" joint="j" dampratio="1"/>
    <position class="same" joint="j"/>
    <velocity class="vel" joint="j"/>
    <motor class="mot" joint="j"/>
    <motor class="servo" joint="j"/>
  </actuator>
</mujoco>
)";
  static constexpr char expected[] = R"(<mujoco model="MuJoCo Model">
  <default>
    <default class="servo">
      <position kp="5" kv="1"/>
      <default class="fast">
        <position kp="50"/>
      </default>
      <default class="vel">
        <velocity kv="2"/>
      </default>
      <default class="mot">
        <motor/>
      </default>
      <default class="same"/>
    </default>
  </default>

  <worldbody>
    <body>
      <joint name="j" type="slide" range="0 2"/>
      <geom size="1"/>
    </body>
  </worldbody>

  <actuator>
    <position class="fast" joint="j"/>
    <position class="servo" joint="j" kv="0"/>
    <position class="servo" joint="j" dampratio="1"/>
    <position class="same" joint="j"/>
    <velocity class="vel" joint="j"/>
    <motor class="mot" joint="j"/>
    <motor class="servo" joint="j"/>
  </actuator>
</mujoco>
)";
  for (bool savecompiled : {false, true}) {
    mjSpec* spec = ParseAsWritten(xml);
    ASSERT_THAT(spec, NotNull());
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    spec->compiler.savecompiled = savecompiled;
    std::string saved = SaveAndReadXml(spec);
    if (!savecompiled) {
      EXPECT_EQ(saved, expected);
    } else {
      EXPECT_THAT(saved, Not(HasSubstr("<position")));
      EXPECT_THAT(saved, Not(HasSubstr("<velocity")));
      EXPECT_THAT(saved, Not(HasSubstr("<motor")));
      EXPECT_THAT(saved, HasSubstr(R"(<default class="same"/>)"));
      EXPECT_THAT(saved, HasSubstr(R"(<general class="servo" joint="j")"));
    }

    std::array<char, 1024> error;
    mjSpec* reloaded =
        mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
    mjModel* remodel = mj_compile(reloaded, nullptr);
    ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
    std::string field;
    EXPECT_LE(CompareModel(model, remodel, field), savecompiled ? 1e-6 : 0)
        << field << "\n"
        << saved;

    mj_deleteModel(remodel);
    mj_deleteSpec(reloaded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

// a shortcut under a general default with its gain model, or under a general
// child class of a shortcut default, keeps its tag; an inherited range is
// written only where it is not already inherited
TEST_F(AsWrittenTest, SavesShortcutsUnderGeneralDefaults) {
  static constexpr char xml[] = R"(<mujoco model="MuJoCo Model">
  <default>
    <default class="lift">
      <general forcerange="-70 70" biastype="affine" gainprm="400" biasprm="0 -200 -100"/>
    </default>
    <default class="servo">
      <position kp="100" kv="10"/>
      <default class="limited">
        <general forcerange="-50 50"/>
      </default>
      <default class="range">
        <position inheritrange="1"/>
      </default>
      <default class="arm">
        <velocity kv="20"/>
      </default>
    </default>
    <default class="pid">
      <pid kp="7" ki="3"/>
      <default class="pidlimited">
        <general forcerange="-5 5"/>
      </default>
    </default>
  </default>

  <worldbody>
    <body>
      <joint name="j" type="slide" range="0 2"/>
      <geom size="1"/>
    </body>
  </worldbody>

  <actuator>
    <position class="lift" joint="j"/>
    <position class="limited" joint="j"/>
    <position class="range" joint="j"/>
    <intvelocity class="range" joint="j" actrange="-1 1"/>
    <position class="arm" joint="j"/>
    <pid class="pidlimited" joint="j"/>
  </actuator>
</mujoco>
)";
  for (bool savecompiled : {false, true}) {
    mjSpec* spec = ParseAsWritten(xml);
    ASSERT_THAT(spec, NotNull());
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    EXPECT_EQ(model->actuator_gainprm[0], 400);
    EXPECT_EQ(model->actuator_biasprm[2], -100);
    EXPECT_EQ(model->actuator_gainprm[mjNGAIN], 100);
    EXPECT_EQ(model->actuator_ctrlrange[5], 2);
    EXPECT_EQ(model->actuator_gainprm[4 * mjNGAIN], 100);
    EXPECT_EQ(model->actuator_biasprm[4 * mjNBIAS + 2], -10);
    EXPECT_EQ(model->actuator_biasprm[5 * mjNBIAS + 1], -7);
    EXPECT_EQ(model->actuator_gainprm[5 * mjNGAIN], 3);
    EXPECT_EQ(model->actuator_forcerange[11], 5);
    spec->compiler.savecompiled = savecompiled;
    std::string saved = SaveAndReadXml(spec);
    if (!savecompiled) {
      EXPECT_EQ(saved, xml);
    }

    std::array<char, 1024> error;
    mjSpec* reloaded =
        mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
    mjModel* remodel = mj_compile(reloaded, nullptr);
    ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
    std::string field;
    EXPECT_EQ(CompareModel(model, remodel, field), 0) << field << "\n" << saved;

    mj_deleteModel(remodel);
    mj_deleteSpec(reloaded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

// an actuator which its shortcut does not give back is saved as general; one
// which was written as general stays general; compiled values are all general
TEST_F(AsWrittenTest, SavesGeneralWhenShortcutDoesNotFit) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide" range="0 2"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <position name="edited" joint="j" kp="3"/>
      <dcmotor name="asymmetric" joint="j" motorconst="0.5" resistance="2"/>
      <general name="written" joint="j" biastype="affine" gainprm="3" biasprm="0 -3"/>
      <position name="kept" joint="j" kp="3"/>
    </actuator>
  </mujoco>
  )";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  mjsActuator* edited =
      mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, "edited"));
  edited->gainprm[1] = 7;
  mjsActuator* dc =
      mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, "asymmetric"));
  dc->forcelimited = mjLIMITED_TRUE;
  dc->forcerange[0] = -1;
  dc->forcerange[1] = 2;
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  for (bool savecompiled : {false, true}) {
    spec->compiler.savecompiled = savecompiled;
    std::string saved = SaveAndReadXml(spec);
    EXPECT_THAT(saved, HasSubstr(R"(<general name="edited" joint="j")"));
    EXPECT_THAT(saved, HasSubstr(R"(gainprm="3 7")"));
    EXPECT_THAT(saved, HasSubstr(R"(<general name="asymmetric" joint="j")"));
    EXPECT_THAT(saved, HasSubstr(R"(<general name="written" joint="j")"));
    EXPECT_THAT(saved,
                HasSubstr(savecompiled
                              ? R"(<general name="kept" joint="j")"
                              : R"(<position name="kept" joint="j" kp="3"/>)"));

    std::array<char, 1024> error;
    mjSpec* reloaded =
        mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
    mjModel* remodel = mj_compile(reloaded, nullptr);
    ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
    std::string field;
    EXPECT_EQ(CompareModel(model, remodel, field), 0) << field << "\n" << saved;
    mj_deleteModel(remodel);
    mj_deleteSpec(reloaded);
  }

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// a full inertia which the spec gives in a rotated frame, as one which is read
// from URDF, is saved rotated: MJCF has no orientation beside a full inertia
TEST_F(AsWrittenTest, SavesFullInertiaOfRotatedFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <inertial pos="0 0 0.1" mass="2" fullinertia="0.2 0.3 0.4 0.01 0.02 0.03"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  for (bool canonical : {false, true}) {
    mjSpec* spec = ParseAsWritten(xml, canonical);
    ASSERT_THAT(spec, NotNull());
    mjsBody* body = mjs_findBody(spec, "body");
    body->iquat[0] = 0;
    body->iquat[1] = 1;
    mjModel* model = mj_compile(spec, nullptr);
    ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
    std::string saved = SaveAndReadXml(spec);
    if (canonical) {
      EXPECT_THAT(saved, HasSubstr("diaginertia=")) << saved;
    } else {
      EXPECT_THAT(
          saved,
          HasSubstr(
              R"(<inertial pos="0 0 0.1" mass="2" fullinertia="0.2 0.3 0.4 -0.01 -0.02 0.03"/>)"))
          << saved;
    }

    std::array<char, 1024> error;
    mjSpec* reloaded =
        mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
    mjModel* remodel = mj_compile(reloaded, nullptr);
    ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
    std::string field;
    EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

    mj_deleteModel(remodel);
    mj_deleteSpec(reloaded);
    mj_deleteModel(model);
    mj_deleteSpec(spec);
  }
}

// the vectors which a keyframe has are saved, also those which are what the
// model has without them; a keyframe which has none is saved without any, also
// once the spec is compiled. A vector which was the reference configuration
// stays what it was when the spec gives the model another one
TEST_F(AsWrittenTest, SavesKeyframeVectorsWhichAreGiven) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.25"/>
      </body>
      <body name="target" mocap="true"/>
    </worldbody>
    <actuator>
      <motor joint="slide"/>
    </actuator>
    <keyframe>
      <key name="given" qpos="0" qvel="0" mpos="0 0 0" mquat="1 0 0 0" ctrl="0"/>
      <key name="rest"/>
    </keyframe>
  </mujoco>
  )";
  mjSpec* spec = ParseAsWritten(xml);
  ASSERT_THAT(spec, NotNull());
  const char given[] =
      "<key name=\"given\" qpos=\"0\" qvel=\"0\" mpos=\"0 0 0\" "
      "mquat=\"1 0 0 0\" ctrl=\"0\"/>\n    <key name=\"rest\"/>";
  EXPECT_THAT(SaveToString(spec), HasSubstr(given));
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(SaveToString(spec), HasSubstr(given));

  // the joint and the mocap body are given another reference
  mjs_asJoint(mjs_findElement(spec, mjOBJ_JOINT, "slide"))->ref = 2;
  mjsBody* target = mjs_findBody(spec, "target");
  target->pos[0] = 3;
  target->alt.type = mjORIENTATION_EULER;
  target->alt.euler[2] = 90;
  std::string saved = SaveToString(spec);
  EXPECT_THAT(saved, HasSubstr(given));

  // the saved file compiles to the keyframes which the spec compiles to
  MjModelPtr reloaded = LoadModelFromString(saved);
  ASSERT_THAT(reloaded.get(), NotNull());
  mjModel* direct = mj_compile(spec, nullptr);
  ASSERT_THAT(direct, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(AsVector(direct->key_qpos, 2), ElementsAre(0, 2));
  EXPECT_EQ(AsVector(reloaded->key_qpos, 2), AsVector(direct->key_qpos, 2));
  EXPECT_EQ(AsVector(reloaded->key_mpos, 6), AsVector(direct->key_mpos, 6));
  EXPECT_EQ(AsVector(reloaded->key_mquat, 8), AsVector(direct->key_mquat, 8));

  mj_deleteModel(direct);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// mj_saveLastXML saves the model which was loaded as it was written, with what
// was changed in it
TEST_F(AsWrittenTest, SavesLastXML) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler savecompiled="false" savecanonical="false"/>
    <worldbody>
      <body pos="1 0 0" euler="0 0 90">
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  model->geom_size[0] = 2;
  std::string saved = SaveAndReadXml(model.get());
  EXPECT_THAT(saved, HasSubstr(R"(<body pos="1 0 0" euler="0 0 90">)"));
  EXPECT_THAT(saved, HasSubstr(R"(<geom size="2"/>)"));
}

// the angles of an attached spec are saved in the unit of the spec which it is
// attached to
TEST_F(AsWrittenTest, SavesAttachedInUnitOfParent) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <compiler angle="radian"/>
    <worldbody>
      <frame name="frame"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <worldbody>
      <body name="body" euler="0 0 90">
        <joint name="hinge" range="-90 90" ref="45"/>
        <joint name="loose" axis="1 0 0" range="-90 90" limited="false"/>
        <geom name="geom" size="1" axisangle="0 1 0 90"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* parent = ParseAsWritten(xml_parent);
  mjSpec* child =
      mj_parseXMLString(xml_child, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull());
  ASSERT_THAT(child, NotNull()) << error.data();
  mjsFrame* frame = mjs_findFrame(parent, "frame");
  mjsBody* body = mjs_findBody(child, "body");
  ASSERT_THAT(mjs_attach(frame->element, body->element, "child_", ""),
              NotNull())
      << mjs_getError(parent);
  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);

  // a range which does not limit is not an angle
  std::string saved = SaveAndReadXml(parent);
  EXPECT_THAT(saved, HasSubstr(R"(euler="0 0 1.5707963267948966")"));
  EXPECT_THAT(
      saved,
      HasSubstr(
          R"(range="-1.5707963267948966 1.5707963267948966" ref="0.7853981633974483")"));
  EXPECT_THAT(saved,
              HasSubstr(R"(range="-90 90" axis="1 0 0" limited="false")"));
  EXPECT_THAT(saved, HasSubstr(R"(axisangle="0 1 0 1.5707963267948966")"));

  // it is the same model
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjModel* remodel = mj_compile(reloaded, nullptr);
  ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
  std::string field;
  EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

  mj_deleteModel(remodel);
  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// an attached model whose compiler settings change what compilation infers for
// its elements is not saved as it is written if they differ from those of the
// model: the saved file has one compiler element. Whether a range limits is
// inferred alike under either autolimits
TEST_F(AsWrittenTest, RefusesAttachedCompilerSettings) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <compiler inertiafromgeom="true"/>
    <worldbody>
      <frame name="frame"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char xml_child[] = R"(
  <mujoco>
    <compiler autolimits="false"/>
    <worldbody>
      <body name="body">
        <joint name="hinge" range="-1 1" limited="true"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom size=".1" mass="10"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* parent = ParseAsWritten(xml_parent);
  mjSpec* child =
      mj_parseXMLString(xml_child, nullptr, error.data(), error.size());
  ASSERT_THAT(parent, NotNull());
  ASSERT_THAT(child, NotNull()) << error.data();
  mjsFrame* frame = mjs_findFrame(parent, "frame");
  mjsBody* body = mjs_findBody(child, "body");
  ASSERT_THAT(mjs_attach(frame->element, body->element, "child_", ""),
              NotNull())
      << mjs_getError(parent);

  // the attached body has the mass of its inertial, which inertiafromgeom
  // "true" in the saved file would replace with that of its geom
  mjModel* model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
  EXPECT_EQ(model->body_mass[1], 1);
  std::string message;
  EXPECT_EQ(SaveToString(parent, &message), "");
  EXPECT_THAT(message, HasSubstr("'child_body' is compiled with the setting "
                                 "inertiafromgeom of an attached model"));

  // with the same inertia settings only autolimits differs, which changes
  // nothing: the model is saved, and it is the same model
  parent->compiler.inertiafromgeom = mjINERTIAFROMGEOM_AUTO;
  mj_deleteModel(model);
  model = mj_compile(parent, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(parent);
  std::string saved = SaveToString(parent, &message);
  ASSERT_FALSE(saved.empty()) << message;
  mjSpec* reloaded =
      mj_parseXMLString(saved.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(reloaded, NotNull()) << error.data() << "\n" << saved;
  mjModel* remodel = mj_compile(reloaded, nullptr);
  ASSERT_THAT(remodel, NotNull()) << mjs_getError(reloaded);
  std::string field;
  EXPECT_EQ(CompareModel(model, remodel, field), 0) << field;

  // a saved file without autolimits would make "auto" with a range an error
  parent->compiler.autolimits = 0;
  child->compiler.autolimits = 1;
  EXPECT_EQ(SaveToString(parent, &message), "");
  EXPECT_THAT(message, HasSubstr("'child_hinge' is compiled with the setting "
                                 "autolimits of an attached model"));

  mj_deleteModel(remodel);
  mj_deleteSpec(reloaded);
  mj_deleteModel(model);
  mj_deleteSpec(child);
  mj_deleteSpec(parent);
}

// compiled values, and inertials which are calculated, are those of a
// compilation
TEST_F(AsWrittenTest, CompiledValuesNeedCompilation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  std::string message;
  EXPECT_EQ(SaveToString(spec, &message), "");
  EXPECT_THAT(message, HasSubstr("Only compiled model can be written"));

  spec->compiler.savecompiled = 0;
  EXPECT_THAT(SaveToString(spec, &message), HasSubstr("<geom size=\"1\"/>"));

  spec->compiler.saveinertial = 1;
  EXPECT_EQ(SaveToString(spec, &message), "");
  EXPECT_THAT(message, HasSubstr("Only compiled model can be written"));

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(SaveToString(spec, &message), HasSubstr("<inertial"));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(XMLWriterTest, EmptyFlagsAttribute) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="cam" output=""/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("output=")));
}

TEST_F(XMLWriterTest, OmitsDefaultJointPosAxis) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="hinge"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("pos=\"0 0 0\"")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("axis=")));
}

TEST_F(XMLWriterTest, KeepsAuthoredJointPosAxis) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <joint name="j" type="hinge" pos="0 0 0.5" axis="1 0 0"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("pos=\"0 0 0.5\""));
  EXPECT_THAT(saved_xml, HasSubstr("axis=\"1 0 0\""));
}

TEST_F(XMLWriterTest, OmitsDefaultEqualityData) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b1"><joint name="j1" type="hinge"/><geom size=".1"/></body>
      <body name="b2"><joint name="j2" type="hinge"/><geom size=".1"/></body>
    </worldbody>
    <equality>
      <weld body1="b1" body2="b2"/>
      <joint joint1="j1" joint2="j2"/>
    </equality>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, Not(HasSubstr("anchor=")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("torquescale=")));
  EXPECT_THAT(saved_xml, Not(HasSubstr("polycoef=")));
  // the relative pose, which mj_setConst computes
  EXPECT_THAT(saved_xml, Not(HasSubstr("relpose=")));
}

TEST_F(XMLWriterTest, WritesDefaultClassJointPosAxis) {
  // pos and axis set in a default class used to be lost when saving
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <joint pos="1 2 3" axis="0 1 0"/>
    </default>
    <worldbody>
      <body name="b">
        <joint name="j" type="hinge"/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("<joint pos=\"1 2 3\" axis=\"0 1 0\"/>"));
}

TEST_F(XMLWriterTest, MeshSiteRoundTripPreservesFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="m" vertex="-1 -1 -1  1 -1 -1  1 1 -1  -1 1 -1  -1 -1 1  1 -1 1  1 1 1  -1 1 1"
            refpos="1 2 3" refquat="0 1 0 0"/>
    </asset>
    <worldbody>
      <site name="s" type="mesh" mesh="m" pos="4 5 6" quat="1 0 0 0"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  std::string saved_xml = SaveAndReadXml(model.get());
  MjModelPtr reloaded = LoadModelFromString(saved_xml);
  ASSERT_THAT(reloaded.get(), NotNull());

  EXPECT_NEAR(model->site_pos[0], 4, 1e-6);
  EXPECT_NEAR(model->site_pos[1], 5, 1e-6);
  EXPECT_NEAR(model->site_pos[2], 6, 1e-6);
  EXPECT_NEAR(model->site_quat[0], 1, 1e-6);
  EXPECT_NEAR(model->site_quat[1], 0, 1e-6);
  EXPECT_NEAR(model->site_quat[2], 0, 1e-6);
  EXPECT_NEAR(model->site_quat[3], 0, 1e-6);

  EXPECT_NEAR(model->site_pos[0], reloaded->site_pos[0], 1e-6);
  EXPECT_NEAR(model->site_pos[1], reloaded->site_pos[1], 1e-6);
  EXPECT_NEAR(model->site_pos[2], reloaded->site_pos[2], 1e-6);
  EXPECT_NEAR(model->site_quat[0], reloaded->site_quat[0], 1e-6);
  EXPECT_NEAR(model->site_quat[1], reloaded->site_quat[1], 1e-6);
  EXPECT_NEAR(model->site_quat[2], reloaded->site_quat[2], 1e-6);
  EXPECT_NEAR(model->site_quat[3], reloaded->site_quat[3], 1e-6);
}

// the compiled frame of a mesh geom absorbs the mesh frame (centroid offset,
// rotation to principal axes), surfacevel is saved in the authored frame
TEST_F(XMLWriterTest, MeshGeomRoundTripPreservesSurfacevel) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box" vertex=".7 1.8 2.9  1.3 1.8 2.9  .7 2.2 2.9  1.3 2.2 2.9  .7 1.8 3.1  1.3 1.8 3.1  .7 2.2 3.1  1.3 2.2 3.1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="box" surfacevel="1 0 0  0 0 2"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());

  MjModelPtr reloaded = LoadModelFromString(SaveAndReadXml(model.get()));
  ASSERT_THAT(reloaded.get(), NotNull());

  EXPECT_THAT(
      AsVector(reloaded->geom_surfacevel, 6),
      Pointwise(MjNear(1e-12, 1e-6), AsVector(model->geom_surfacevel, 6)));
}

TEST_F(XMLWriterTest, TextureGridsizeWithoutGridlayoutAndCompiledNchannel) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="grid_cls">
        <texture type="skybox" gridsize="2 4" gridlayout="LFRB.D.."/>
      </default>
    </default>
    <asset>
      <texture name="sky1" type="skybox" file="dummy.png" gridsize="2 3"/>
      <texture name="sky2" class="grid_cls" file="dummy.png" gridsize="2 3"/>
      <texture name="ktx" type="2d" file="tex.ktx" content_type="image/ktx"/>
    </asset>
  </mujoco>
  )";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  static constexpr char kDummy[] = "ktxblob";
  mj_addBufferVFS(&vfs, "tex.ktx", kDummy, sizeof(kDummy));

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, &vfs, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // authored save: sky1 and sky2 omit default-dots gridlayout, resetting sky2
  spec->compiler.savecompiled = 0;
  std::string saved_authored = SaveToString(spec);
  EXPECT_THAT(saved_authored, Not(HasSubstr("gridlayout=\".")));
  mjSpec* reloaded_authored = mj_parseXMLString(saved_authored.c_str(), &vfs,
                                                error.data(), error.size());
  ASSERT_THAT(reloaded_authored, NotNull()) << error.data() << "\n"
                                            << saved_authored;
  mjsTexture* sky2 =
      mjs_asTexture(mjs_findElement(reloaded_authored, mjOBJ_TEXTURE, "sky2"));
  ASSERT_THAT(sky2, NotNull());
  EXPECT_EQ(std::string(sky2->gridlayout, 12), "............");
  mj_deleteSpec(reloaded_authored);

  // compiled save on ktx texture: compiled nchannel=1 is written
  mjs_delete(spec, mjs_findElement(spec, mjOBJ_TEXTURE, "sky1"));
  mjs_delete(spec, mjs_findElement(spec, mjOBJ_TEXTURE, "sky2"));
  mjModel* model = mj_compile(spec, &vfs);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->tex_nchannel[0], 1);
  spec->compiler.savecompiled = 1;
  std::string saved_compiled = SaveToString(spec);
  EXPECT_THAT(saved_compiled, HasSubstr("nchannel=\"1\""));

  mj_deleteModel(model);
  mj_deleteSpec(spec);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLWriterTest, MaterialDefaultLayerRoundtrip) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="layered">
        <material>
          <layer role="normal" texture="norm"/>
        </material>
      </default>
    </default>
    <asset>
      <texture name="rgb" type="2d" builtin="flat" width="4" height="4"/>
      <texture name="norm" type="2d" builtin="flat" width="4" height="4"/>
      <material name="m" class="layered" texture="rgb"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  std::string saved = SaveAndReadXml(model.get());
  MjModelPtr reloaded =
      LoadModelFromString(saved.c_str(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
  EXPECT_EQ(reloaded->mat_texid[mjNTEXROLE * 0 + mjTEXROLE_RGB], 0);
  EXPECT_EQ(reloaded->mat_texid[mjNTEXROLE * 0 + mjTEXROLE_NORMAL], 1);
}

TEST_F(XMLWriterTest, CameraDefaultSensorsizeAndDefaultLightTexture) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="cam_cls">
        <camera sensorsize="0.036 0.024" focal="0.05 0.05" resolution="640 480"/>
        <light directional="true"/>
      </default>
    </default>
    <worldbody>
      <camera name="c1" class="cam_cls"/>
      <camera name="c2" class="cam_cls" sensorsize="0.02 0.02"/>
      <light name="l1" class="cam_cls"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjsDefault* def = mjs_findDefault(spec, "cam_cls");
  ASSERT_THAT(def, NotNull());
  mjs_setString(def->light->texture, "ignored_in_default");

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  spec->compiler.savecompiled = 1;
  std::string saved = SaveToString(spec);
  EXPECT_THAT(saved, HasSubstr("<camera name=\"c1\" class=\"cam_cls\"/>"));
  EXPECT_THAT(
      saved,
      HasSubstr(
          "<camera name=\"c2\" class=\"cam_cls\" sensorsize=\"0.02 0.02\"/>"));
  EXPECT_THAT(saved, Not(HasSubstr("ignored_in_default")));

  MjModelPtr reloaded =
      LoadModelFromString(saved.c_str(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(XMLWriterTest, PluginMeshPreservesAttributes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sdf.torus"/>
    </extension>
    <asset>
      <material name="mat"/>
      <mesh name="torus" refpos="0.1 0.2 0.3" refquat="0 1 0 0"
            maxhullvert="16" material="mat">
        <plugin plugin="mujoco.sdf.torus"/>
      </mesh>
    </asset>
    <worldbody>
      <geom type="sdf" mesh="torus">
        <plugin plugin="mujoco.sdf.torus"/>
      </geom>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  std::string compile_error = model ? "" : mjs_getError(spec);
  if (!model) {
    mj_deleteSpec(spec);
  }
  ASSERT_THAT(model, NotNull()) << compile_error;

  for (int savecompiled : {0, 1}) {
    spec->compiler.savecompiled = savecompiled;
    std::string saved = SaveToString(spec);
    EXPECT_THAT(saved, HasSubstr("refpos=\"0.1 0.2 0.3\""));
    EXPECT_THAT(saved, HasSubstr("refquat=\"0 1 0 0\""));
    EXPECT_THAT(saved, HasSubstr("maxhullvert=\"16\""));
    EXPECT_THAT(saved, HasSubstr("material=\"mat\""));
    EXPECT_THAT(saved, Not(HasSubstr("vertex=")));
    MjModelPtr reloaded =
        LoadModelFromString(saved.c_str(), error.data(), error.size());
    ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
    EXPECT_EQ(reloaded->geom_matid[0], 0);
  }

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(XMLWriterTest, PluginSensorPreservesReftypeRefname) {
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.sensor.touch_grid"/>
    </extension>
    <worldbody>
      <site name="ref_site"/>
      <body>
        <geom size="0.1"/>
        <site name="touch_site"/>
      </body>
    </worldbody>
    <sensor>
      <plugin name="touch" plugin="mujoco.sensor.touch_grid"
              objtype="site" objname="touch_site"
              reftype="site" refname="ref_site">
        <config key="size" value="4 4"/>
        <config key="fov" value="10 10"/>
        <config key="gamma" value="0"/>
        <config key="nchannel" value="1"/>
      </plugin>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);

  for (int savecompiled : {0, 1}) {
    spec->compiler.savecompiled = savecompiled;
    std::string saved = SaveToString(spec);
    EXPECT_THAT(saved, HasSubstr("reftype=\"site\""));
    EXPECT_THAT(saved, HasSubstr("refname=\"ref_site\""));
    MjModelPtr reloaded =
        LoadModelFromString(saved.c_str(), error.data(), error.size());
    ASSERT_THAT(reloaded.get(), NotNull()) << error.data() << "\n" << saved;
    EXPECT_EQ(reloaded->sensor_reftype[0], mjOBJ_SITE);
    EXPECT_EQ(reloaded->sensor_refid[0],
              mj_name2id(reloaded.get(), mjOBJ_SITE, "ref_site"));
  }

  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

}  // namespace
}  // namespace mujoco
