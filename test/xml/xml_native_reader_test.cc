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

// Tests for xml/xml_native_reader.cc.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjspec.h>
#include <mujoco/mujoco.h>
#include "src/cc/array_safety.h"
#include "src/user/user_api.h"
#include "src/xml/xml_api.h"
#include "test/compare_model.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::string;
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::FloatEq;
using ::testing::HasSubstr;
using ::testing::IsNan;
using ::testing::IsNull;
using ::testing::NotNull;

using XMLReaderTest = MujocoTest;

TEST_F(XMLReaderTest, UniqueElementTest) {
  std::array<char, 1024> error;
  static constexpr char xml[] = R"(
  <mujoco>
   <option>
      <flag sensor="disable"/>
      <flag sensor="disable"/>
    </option>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("unique element 'flag' found 2 times"));
}

TEST_F(XMLReaderTest, SchemaLocationHeaderAccepted) {
  // XML namespace machinery is tolerated on the root element, so models can
  // reference mjcf.xsd for editor completion; see the linked example in the
  // XML reference intro
  std::array<char, 1024> error;
  const std::string path =
      GetTestDataFilePath("xml/testdata/schema_location.xml");
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull());
  mj_deleteModel(model);
  mj_deleteSpec(spec);

  // only on the root: namespaced attributes elsewhere are still rejected
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom xsi:type="box" size="1"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr bad = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(bad.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("unrecognized attribute: 'xsi:type'"));
}

TEST_F(XMLReaderTest, AuthoredFromXml) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler boundmass="1"/>
    <option timestep="0.01" gravity="0 0 -5">
      <flag constraint="disable"/>
    </option>
    <visual>
      <global fovy="60"/>
      <quality shadowsize="1024"/>
    </visual>
    <worldbody/>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // option: timestep authored, gravity authored, density not authored
  EXPECT_EQ(mjs_isAuthored(spec, &spec->option.timestep), 1);
  EXPECT_EQ(mjs_isAuthored(spec, spec->option.gravity), 1);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->option.density), 0);

  // compiler: boundmass authored, boundinertia not authored
  EXPECT_EQ(mjs_isAuthored(spec, &spec->compiler.boundmass), 1);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->compiler.boundinertia), 0);

  // flags: constraint disable authored, contact not authored
  EXPECT_NE(spec->authored.disableflags & mjDSBL_CONSTRAINT, 0);
  EXPECT_EQ(spec->authored.disableflags & mjDSBL_CONTACT, 0);

  // visual global: fovy authored, ipd not authored
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.global.fovy), 1);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.global.ipd), 0);

  // visual quality: shadowsize authored, offsamples not authored
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.quality.shadowsize), 1);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.quality.offsamples), 0);

  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, AuthoredDefaultsZero) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody/>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // nothing authored in an empty model
  EXPECT_EQ(mjs_isAuthored(spec, &spec->option.timestep), 0);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->option.gravity), 0);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->compiler.boundmass), 0);
  EXPECT_EQ(spec->authored.disableflags, 0);
  EXPECT_EQ(spec->authored.enableflags, 0);
  EXPECT_EQ(spec->authored.disableactuator, 0);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.global.fovy), 0);
  EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.map.znear), 0);

  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, MemorySize) {
  std::array<char, 1024> error;
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="512"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 512);
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1K "/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 1024);
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="  10K"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 10240);
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory=" 4M  "/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 4 * 1024 * 1024);
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1G"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 1024 * 1024 * 1024);
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1073741824"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), NotNull()) << error.data();
    EXPECT_EQ(model->narena, 1024 * 1024 * 1024);
  }
}

TEST_F(XMLReaderTest, InvalidMemorySize) {
  std::array<char, 1024> error;
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="-3"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), IsNull());
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="1 M"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), IsNull());
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="2X"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), IsNull());
  }
  {
    static constexpr char xml[] = R"(
    <mujoco>
      <size memory="K"/>
    </mujoco>
    )";
    MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
    ASSERT_THAT(model.get(), IsNull());
  }
}

TEST_F(XMLReaderTest, InvalidNUserBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_body="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_body"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserJoint) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_jnt="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_jnt"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserGeom) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_geom="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_geom"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserSite) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_site="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_site"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserCamera) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_cam="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_cam"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_tendon="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_tendon"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserActuator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_actuator="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_actuator"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, InvalidNUserSensor) {
  static constexpr char xml[] = R"(
  <mujoco>
    <size nuser_sensor="-2"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("nuser_sensor"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, CanParseInf) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" pos="5e-1 -INF iNf"/>
        <geom size="1" pos="inF -inf Inf"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  const double inf = std::numeric_limits<double>::infinity();
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(model->geom_pos[0], 0.5);
  EXPECT_EQ(model->geom_pos[1], -inf);
  EXPECT_THAT(model->geom_pos[2], inf);
  EXPECT_EQ(model->geom_pos[3], inf);
  EXPECT_EQ(model->geom_pos[4], -inf);
  EXPECT_EQ(model->geom_pos[5], inf);
}

TEST_F(XMLReaderTest, CanParseNanAndRaisesWarning) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" pos="nan NaN NAN"/>
        <geom size="1" pos="1.0 0.0 nAn"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings("XML contains a 'NaN'");
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_THAT(model->geom_pos[0], IsNan());
  EXPECT_THAT(model->geom_pos[1], IsNan());
  EXPECT_THAT(model->geom_pos[2], IsNan());
  EXPECT_EQ(model->geom_pos[3], 1);
  EXPECT_EQ(model->geom_pos[4], 0);
  EXPECT_THAT(model->geom_pos[5], IsNan());
}

TEST_F(XMLReaderTest, InvalidArrayElement) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" axisangle="1.0 0.0 0.0 [[1]]"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("bad format in attribute 'axisangle'"));
  EXPECT_THAT(error.data(), HasSubstr("line 5"));
}

TEST_F(XMLReaderTest, InvalidArrayLength) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" axisangle="1 0 0 0 1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("has too much data"));
  EXPECT_THAT(error.data(), HasSubstr("line 5"));
}

TEST_F(XMLReaderTest, InvalidQuaternion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" quat="0 0 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("zero quaternion is not allowed"));
  EXPECT_THAT(error.data(), HasSubstr("line 5"));
}

TEST_F(XMLReaderTest, InvalidNumber) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" axisangle="1 0.1.2.3"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("bad format in attribute 'axisangle'"));
  EXPECT_THAT(error.data(), HasSubstr("line 5"));
}

TEST_F(XMLReaderTest, InvalidNumberRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="mesh" file="mesh.stl" face="100000000000000000000000"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("number is too large in attribute 'face'"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(XMLReaderTest, InvalidNumberOfAttributes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 .3">
        <freejoint/>
        <geom name="ellipsoid" type="ellipsoid" size=".1 .1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("size 2 must be positive"));
  EXPECT_THAT(error.data(), HasSubstr("line 6"));
}

TEST_F(XMLReaderTest, AllowsSpaces) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1" axisangle="1 0   0 0 "/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
}

TEST_F(XMLReaderTest, QuatReplacesDefaultOrientation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <geom size="1" euler="0 0 90"/>
      <site euler="0 0 90"/>
      <camera euler="0 0 90"/>
    </default>

    <worldbody>
      <geom name="inherits"/>
      <geom name="replaces" quat="1 0 0 0"/>
      <site name="inherits"/>
      <site name="replaces" quat="1 0 0 0"/>
      <camera name="inherits"/>
      <camera name="replaces" quat="1 0 0 0"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_NE(model->geom_quat[0], 1);
  EXPECT_THAT(AsVector(model->geom_quat + 4, 4), ElementsAre(1, 0, 0, 0));
  EXPECT_NE(model->site_quat[0], 1);
  EXPECT_THAT(AsVector(model->site_quat + 4, 4), ElementsAre(1, 0, 0, 0));
  EXPECT_NE(model->cam_quat[0], 1);
  EXPECT_THAT(AsVector(model->cam_quat + 4, 4), ElementsAre(1, 0, 0, 0));
}

TEST_F(XMLReaderTest, InvalidDoubleOrientation) {
  std::string prefix = "<mujoco><worldbody><";
  std::string suffix = "/></worldbody></mujoco>";
  std::vector<std::string> orientations = {
      R"( quat="0 1 0 0" )",
      R"( euler="1.7 2.9 0.1" )",
      R"( zaxis="1.7 2.9 0.1" )",
      R"( axisangle="1.7 2.9 0.1 0" )",
      R"( xyaxes="1.7 2.9 0.1 0.4 1.4 0.6" )",
  };
  std::vector<std::string> fields = {"geom", "body", "camera", "site"};
  for (auto const& field : fields) {
    for (auto const& orient1 : orientations) {
      for (auto const& orient2 : orientations) {
        if (orient1 == orient2) continue;
        std::string xml = prefix + field + orient1 + orient2 + suffix;
        std::array<char, 1024> error;
        MjModelPtr model =
            LoadModelFromString(xml.c_str(), error.data(), error.size());
        ASSERT_THAT(model.get(), IsNull());
        EXPECT_THAT(
            error.data(),
            HasSubstr("multiple orientation specifiers are not allowed"));
      }
    }
  }
}

TEST_F(XMLReaderTest, ClassOverridesChildclass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="size2">
        <geom size="2"/>
      </default>
      <default class="size3">
        <geom size="3"/>
      </default>
    </default>
    <worldbody>
      <frame childclass="size2">
        <geom/>
        <geom class="size3"/>
        <body childclass="size2">
          <geom/>
          <geom class="size3"/>
        </body>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->geom_size[3 * 0], 2);
  EXPECT_EQ(model->geom_size[3 * 1], 3);
  EXPECT_EQ(model->geom_size[3 * 2], 2);
  EXPECT_EQ(model->geom_size[3 * 3], 3);
}

TEST_F(XMLReaderTest, RepeatedDefaultName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="sphere">
        <geom type="sphere" size="1"/>
      </default>
      <default class="sphere">
        <geom type="capsule" size="1 1"/>
      </default>
    </default>
    <worldbody>
      <body>
        <geom class="sphere"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull()) << error.data();
  EXPECT_THAT(error.data(), HasSubstr("repeated default class name"));
}

TEST_F(XMLReaderTest, InvalidDefaultClassName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="sphere">
        <geom type="sphere" size="1"/>
      </default>
    </default>
    <worldbody>
      <body>
        <geom class="invalid"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull()) << error.data();
  EXPECT_THAT(error.data(),
              AllOf(HasSubstr("unknown default class name 'invalid'"),
                    HasSubstr("Element 'geom'"), HasSubstr("line 10")));
}

TEST_F(XMLReaderTest, InvalidTopDefaultClassName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default class="sphere">
      <geom type="sphere" size="1"/>
    </default>
    <worldbody>
      <body>
        <geom class="sphere"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull()) << error.data();
  EXPECT_THAT(error.data(),
              HasSubstr("top-level default class 'main' cannot be renamed"));
}

TEST_F(XMLReaderTest, ValidTopDefaultClassName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default class="main">
      <geom type="sphere" size="1"/>
    </default>
    <worldbody>
      <body>
        <geom class="main"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
}

// ------------------------ test including -------------------------------------

// tiny RGB 2 x 3 PNG file
static constexpr unsigned char kTinyPng[] = {
    0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00,
    0x0d, 0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
    0x00, 0x02, 0x08, 0x02, 0x00, 0x00, 0x00, 0x12, 0x16, 0xf1, 0x4d,
    0x00, 0x00, 0x00, 0x1c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63,
    0x78, 0xc1, 0xc0, 0xc0, 0xc0, 0xf0, 0xbf, 0xb8, 0xb8, 0x98, 0x81,
    0xe1, 0x3f, 0xc3, 0xff, 0xff, 0xff, 0xc5, 0xc4, 0xc4, 0x00, 0x46,
    0xd7, 0x07, 0x7f, 0xd2, 0x52, 0xa1, 0x41, 0x00, 0x00, 0x00, 0x00,
    0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82};

// mesh OBJ file of a cube
static constexpr char kTinyObj[] = R"(
  v -1 -1  1
  v  1 -1  1
  v -1  1  1
  v  1  1  1
  v -1  1 -1
  v  1  1 -1
  v -1 -1 -1
  v  1 -1 -1)";

TEST_F(XMLReaderTest, IncludeTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="1 1 1"/>
      <include file="model1.xml"/>
      <include file="model2.xml"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml1[] = R"(
  <mujoco>
    <geom name="box" type="box" size="1 1 1"/>
  </mujoco>)";

  static constexpr char xml2[] = R"(
  <mujoco>
    <geom name="ball" type="sphere" size="2"/>
    <include file="model3.xml"/>
  </mujoco>)";

  static constexpr char xml3[] = R"(
  <mujoco>
    <geom name="another_box" type="box" size="2 2 2"/>
  </mujoco>)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  mj_addBufferVFS(vfs.get(), "model1.xml", xml1, sizeof(xml1));
  mj_addBufferVFS(vfs.get(), "model2.xml", xml2, sizeof(xml2));
  mj_addBufferVFS(vfs.get(), "model3.xml", xml3, sizeof(xml3));

  std::array<char, 1024> error;
  MjModelPtr model =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(mj_name2id(model.get(), mjOBJ_GEOM, "ball"), 2);
  EXPECT_EQ(mj_name2id(model.get(), mjOBJ_GEOM, "another_box"), 3);
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, IncludeChildTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="1 1 1"/>
      <include file="model1.xml">
        <geom name="box" type="box" size="1 1 1"/>
      </include>
    </worldbody>
  </mujoco>)";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("Include element cannot have children"));
}

TEST_F(XMLReaderTest, IncludeSameFileTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <include file="model1.xml"/>
    <include file="model1.xml"/>
  </mujoco>)";

  static constexpr char xml1[] = R"(
  <mujoco>
    <geom name="box" type="box" size="1 1 1"/>
  </mujoco>)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());

  mj_addBufferVFS(vfs.get(), "model1.xml", xml1, sizeof(xml1));

  std::array<char, 1024> error;
  MjModelPtr model =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("File 'model1.xml' already included"));
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, IncludePathTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="1 1 1"/>
      <include file="submodels/model1.xml"/>
      <include file="submodels/model2.xml"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml1[] = R"(
  <mujoco>
    <geom name="box" type="box" size="1 1 1"/>
  </mujoco>)";

  static constexpr char xml2[] = R"(
  <mujoco>
    <geom name="ball" type="sphere" size="2"/>
    <include file="subsubmodels/model3.xml"/>
  </mujoco>)";

  static constexpr char xml3[] = R"(
  <mujoco>
    <geom name="another_box" type="box" size="2 2 2"/>
  </mujoco>)";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));
  mj_addBufferVFS(&vfs, "submodels/model1.xml", xml1, sizeof(xml1));
  mj_addBufferVFS(&vfs, "submodels/model2.xml", xml2, sizeof(xml2));
  mj_addBufferVFS(&vfs, "submodels/subsubmodels/model3.xml", xml3,
                  sizeof(xml3));

  std::array<char, 1024> error;
  mjModel* model = mj_loadXML("model.xml", &vfs, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_EQ(mj_name2id(model, mjOBJ_GEOM, "ball"), 2);
  EXPECT_EQ(mj_name2id(model, mjOBJ_GEOM, "another_box"), 3);
  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, FallbackIncludePathTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="1 1 1"/>
      <include file="model1.xml"/>
      <include file="submodels/model2.xml"/>
    </worldbody>
  </mujoco>)";

  static constexpr char xml1[] = R"(
  <mujoco>
    <geom name="box" type="box" size="1 1 1"/>
  </mujoco>)";

  static constexpr char xml2[] = R"(
  <mujoco>
    <geom name="ball" type="sphere" size="2"/>
    <include file="subsubmodels/model3.xml"/>
  </mujoco>)";

  static constexpr char xml3[] = R"(
  <mujoco>
    <geom name="another_box" type="box" size="2 2 2"/>
  </mujoco>)";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));
  mj_addBufferVFS(&vfs, "model1.xml", xml1, sizeof(xml1));
  mj_addBufferVFS(&vfs, "submodels/model2.xml", xml2, sizeof(xml2));
  mj_addBufferVFS(&vfs, "subsubmodels/model3.xml", xml3, sizeof(xml3));

  std::array<char, 1024> error;
  mjModel* model = mj_loadXML("model.xml", &vfs, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();
  EXPECT_EQ(mj_name2id(model, mjOBJ_GEOM, "ball"), 2);
  EXPECT_EQ(mj_name2id(model, mjOBJ_GEOM, "another_box"), 3);
  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, MaterialTextureTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny0.png" type="2d" name="tiny0"/>
      <texture file="tiny1.png" type="2d" name="tiny1"/>
      <material name="material">
        <layer role="occlusion" texture="tiny0"/>
        <layer role="roughness" texture="tiny0"/>
        <layer role="metallic" texture="tiny0"/>
        <layer role="rgb" texture="tiny1"/>
      </material>
    </asset>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "tiny0.png", kTinyPng, sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "tiny1.png", kTinyPng, sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));

  char error[1024];
  mjModel* model = mj_loadXML("model.xml", &vfs, error, 1024);

  EXPECT_THAT(model, NotNull()) << error;
  EXPECT_EQ(model->mat_texid[mjTEXROLE_RGB], 1);
  EXPECT_EQ(model->mat_texid[mjTEXROLE_METALLIC], 0);
  EXPECT_EQ(model->mat_texid[mjTEXROLE_ROUGHNESS], 0);
  EXPECT_EQ(model->mat_texid[mjTEXROLE_OCCLUSION], 0);

  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, LegacyMaterialTextureTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny0.png" type="2d" name="tiny0"/>
      <texture file="tiny1.png" type="2d" name="tiny1"/>
      <material name="material" texture="tiny1"/>
    </asset>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "tiny0.png", kTinyPng, sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "tiny1.png", kTinyPng, sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));

  char error[1024];
  mjModel* model = mj_loadXML("model.xml", &vfs, error, 1024);

  EXPECT_THAT(model, NotNull()) << error;
  EXPECT_EQ(model->mat_texid[mjTEXROLE_RGB], 1);

  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, MaterialTextureFailTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny0.png" type="2d" name="tiny0"/>
      <texture file="tiny1.png" type="2d" name="tiny1"/>
      <material name="material" texture="tiny1">
        <layer role="rgb" texture="tiny1"/>
        <layer role="occlusion" texture="tiny0"/>
      </material>
    </asset>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("A material with a texture attribute "
                                      "cannot have layer sub-elements"));
}

// an unknown layer role is a parse error, not an mju_error which would longjmp
// past the parser's C++ objects; leaks are detected when running under asan
TEST_F(XMLReaderTest, MaterialUnknownLayerRole) {
  static constexpr char xml[] = R"(
  <mujoco model="a model name which is longer than the sso buffer">
    <asset>
      <texture name="texture_name_longer_than_sso" type="2d" builtin="checker"
        width="8" height="8"/>
      <material name="material_name_longer_than_sso">
        <layer role="not_a_role" texture="texture_name_longer_than_sso"/>
      </material>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error = {0};
  auto parse = MjuErrorMessageFrom(mj_parseXMLString);
  EXPECT_EQ(parse(xml, nullptr, error.data(), error.size()), "");
  EXPECT_THAT(error.data(), HasSubstr("unknown layer role 'not_a_role'"));
}

TEST_F(XMLReaderTest, LargeTextureAddressTest) {
#if defined(__has_feature)
  #if __has_feature(address_sanitizer) || __has_feature(memory_sanitizer) || \
      __has_feature(thread_sanitizer)
  GTEST_SKIP() << "Skipping large texture allocation under sanitizers";
  #endif
#elif defined(ADDRESS_SANITIZER) || defined(MEMORY_SANITIZER) || \
    defined(THREAD_SANITIZER)
  GTEST_SKIP() << "Skipping large texture allocation under sanitizers";
#endif
  static constexpr char xml[] = R"(
  <mujoco>
  <asset>
    <!--
      Test that tex_adr can correctly index into a texture buffer larger than
      what a 32-bit signed integer can represent.
      The first texture has size 6*10923*10923*3 > 2^31-1 bytes.
      tex_adr[1] should correctly point beyond this offset.
    -->
    <texture name="tex0" builtin="gradient" width="10923" height="2"/>
    <texture name="tex1" builtin="flat" width="2" height="2"/>
  </asset>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());

  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->ntex, 2);
  EXPECT_GT(model->tex_adr[1], INT32_MAX);
}

TEST_F(XMLReaderTest, IncludeAssetsTest) {
  static constexpr char xml[] = R"(
  <mujoco>
   <include file="assets/assets.xml"/>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char assets[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny.png" type="2d"/>
      <material name="material" texture="tiny"/>
      <include file="subassets/assets.xml"/>
    </asset>
  </mujoco>
  )";
  static constexpr char subassets[] = R"(
  <mujoco>
    <texture file="subtiny.png" type="2d"/>
    <mesh name="cube" file="cube.obj"/>
    <material name="submaterial" texture="subtiny"/>
  </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "assets/tiny.png", kTinyPng, sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "assets/subassets/subtiny.png", kTinyPng,
                  sizeof(kTinyPng));
  mj_addBufferVFS(&vfs, "assets/subassets/cube.obj", kTinyObj,
                  sizeof(kTinyObj));
  mj_addBufferVFS(&vfs, "assets/assets.xml", assets, sizeof(assets));
  mj_addBufferVFS(&vfs, "assets/subassets/assets.xml", subassets,
                  sizeof(subassets));
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));

  // loading the file should be successful
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML("model.xml", &vfs, error.data(), error.size());

  ASSERT_THAT(model, NotNull()) << error.data();

  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, FallbackIncludeAssetsTest) {
  static constexpr char xml[] = R"(
  <mujoco>
   <include file="assets/assets.xml"/>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char assets[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny.png" type="2d"/>
      <material name="material" texture="tiny"/>
      <include file="subassets/assets.xml"/>
    </asset>
  </mujoco>
  )";
  static constexpr char subassets[] = R"(
  <mujoco>
    <texture file="subtiny.png" type="2d"/>
    <material name="submaterial" texture="subtiny"/>
  </mujoco>
  )";

  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "assets/tiny.png", kTinyPng, sizeof(kTinyPng));

  // need to fallback for backwards compatibility
  mj_addBufferVFS(&vfs, "subtiny.png", kTinyPng, sizeof(kTinyPng));

  mj_addBufferVFS(&vfs, "assets/assets.xml", assets, sizeof(assets));
  mj_addBufferVFS(&vfs, "assets/subassets/assets.xml", subassets,
                  sizeof(subassets));
  mj_addBufferVFS(&vfs, "model.xml", xml, sizeof(xml));

  // loading the file should be successful
  std::array<char, 1024> error;
  mjModel* model = mj_loadXML("model.xml", &vfs, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();

  mj_deleteModel(model);
  mj_deleteVFS(&vfs);
}

TEST_F(XMLReaderTest, IncludeAbsoluteTest) {
  static constexpr char xml[] = R"(
  <mujoco>
   <include file="assets/assets.xml"/>
    <worldbody>
      <geom type="plane" material="material" size="4 4 4"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char assets[] = R"(
  <mujoco>
    <asset>
      <texture file="tiny.png" type="2d"/>
      <material name="material" texture="tiny"/>
      <include file="subassets/assets.xml"/>
    </asset>
  </mujoco>
  )";
  static constexpr char subassets[] = R"(
  <mujoco>
    <texture file="MjMock.IncludeAbsoluteTest:assets/subtiny.png" type="2d"/>
    <material name="submaterial" texture="subtiny"/>
  </mujoco>
  )";

  MockFilesystem fs("IncludeAbsoluteTest");
  fs.AddFile("assets/tiny.png", kTinyPng, sizeof(kTinyPng));
  fs.AddFile("assets/subtiny.png", kTinyPng, sizeof(kTinyPng));
  fs.AddFile("assets/assets.xml", (const unsigned char*)assets, sizeof(assets));
  fs.AddFile("assets/subassets/assets.xml", (const unsigned char*)subassets,
             sizeof(subassets));
  fs.AddFile("model.xml", (const unsigned char*)xml, sizeof(xml));
  std::string modelpath = fs.FullPath("model.xml");

  std::array<char, 1024> error;
  // loading the file should be successful
  mjModel* model =
      mj_loadXML(modelpath.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();

  mj_deleteModel(model);
}

TEST_F(XMLReaderTest, IncludeAbsoluteMeshDirTest) {
  static constexpr char xml[] = R"(
  <mujoco>
    <include file="assets.xml"/>
  </mujoco>
  )";
  static constexpr char assets[] = R"(
  <mujoco>
    <compiler meshdir="mjMock.IncludeAbsoluteMeshDirTest:/assets"/>
    <asset>
    <mesh file="cube.obj"/>
  </asset>
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

  MockFilesystem fs("IncludeAbsoluteMeshDirTest");
  fs.AddFile("/assets/cube.obj", (const unsigned char*)cube, sizeof(cube));
  fs.AddFile("assets.xml", (const unsigned char*)assets, sizeof(assets));
  fs.AddFile("model.xml", (const unsigned char*)xml, sizeof(xml));
  std::string modelpath = fs.FullPath("model.xml");

  std::array<char, 1024> error;
  // loading the file should be successful
  mjModel* model =
      mj_loadXML(modelpath.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(model, NotNull()) << error.data();

  mj_deleteModel(model);
}

TEST_F(XMLReaderTest, ParsePolycoef) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="0"/>
        <geom size="1"/>
      </body>
      <body>
        <joint name="1"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <equality>
      <joint joint1="0" joint2="1"/>
      <joint joint1="0" joint2="1" polycoef="2"/>
      <joint joint1="0" joint2="1" polycoef="3 4"/>
      <joint joint1="0" joint2="1" polycoef="5 6 7 8 9"/>
    </equality>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_THAT(AsVector(m->eq_data + 0 * mjNEQDATA, 5),
              ElementsAre(0, 1, 0, 0, 0));
  EXPECT_THAT(AsVector(m->eq_data + 1 * mjNEQDATA, 5),
              ElementsAre(2, 1, 0, 0, 0));
  EXPECT_THAT(AsVector(m->eq_data + 2 * mjNEQDATA, 5),
              ElementsAre(3, 4, 0, 0, 0));
  EXPECT_THAT(AsVector(m->eq_data + 3 * mjNEQDATA, 5),
              ElementsAre(5, 6, 7, 8, 9));
}

TEST_F(XMLReaderTest, TendonArmature) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="a"/>
      <body pos="1 0 0">
        <joint name="slide" type="slide"/>
        <geom size=".1"/>
        <site name="b"/>
      </body>
    </worldbody>

    <tendon>
      <spatial armature="1.5">
        <site site="a"/>
        <site site="b"/>
      </spatial>
      <fixed armature="2.5">
        <joint joint="slide" coef="1"/>
      </fixed>
      <fixed>
        <joint joint="slide" coef="2"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_EQ(m->ntendon, 3);
  EXPECT_MJTNUM_EQ(m->tendon_armature[0], 1.5);
  EXPECT_MJTNUM_EQ(m->tendon_armature[1], 2.5);
  EXPECT_MJTNUM_EQ(m->tendon_armature[2], 0);
}

TEST_F(XMLReaderTest, TendonArmatureNegative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="a"/>
      <site name="b" pos="1 0 0"/>
    </worldbody>

    <tendon>
      <spatial armature="-1.5">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("tendon armature cannot be negative"));
}

TEST_F(XMLReaderTest, TendonArmatureGeomWrap) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="a"/>
      <geom name="g" type="sphere" size=".1"/>
      <site name="b" pos="1 0 0"/>
    </worldbody>

    <tendon>
      <spatial armature="1.5">
        <site site="a"/>
        <geom geom="g"/>
        <site site="b"/>
      </spatial>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("geom wrapping not supported"));
}

// ------------------------ test frame parsing ---------------------------------
// an asset with a file and no name is named after the file when it is parsed
TEST_F(XMLReaderTest, AssetNamedAfterFile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh file="meshes/cube.obj"/>
      <mesh name="given" file="meshes/cube.obj"/>
      <texture type="2d" file="textures/tiles.png"/>
      <texture type="skybox" builtin="gradient" width="2" height="12"/>
      <hfield file="terrain.png" size="1 1 1 1"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  auto names = [spec](mjtObj type) {
    std::vector<std::string> names;
    for (mjsElement* element = mjs_firstElement(spec, type); element;
         element = mjs_nextElement(spec, element)) {
      names.push_back(mjs_getString(mjs_getName(element)));
    }
    return names;
  };
  EXPECT_THAT(names(mjOBJ_MESH), ElementsAre("cube", "given"));
  EXPECT_THAT(names(mjOBJ_TEXTURE), ElementsAre("tiles", ""));
  EXPECT_THAT(names(mjOBJ_HFIELD), ElementsAre("terrain"));
  mj_deleteSpec(spec);

  // two files with the same name cannot both give it to their mesh
  static constexpr char xml_repeated[] = R"(
  <mujoco>
    <asset>
      <mesh file="left/cube.obj"/>
      <mesh file="right/cube.obj"/>
    </asset>
  </mujoco>
  )";
  spec = mj_parseXMLString(xml_repeated, 0, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(), HasSubstr("repeated name 'cube' in mesh"));
}

TEST_F(XMLReaderTest, ParseFrame) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="frame">
        <geom size=".1"/>
      </default>
      <default class="body">
        <geom size=".2"/>
      </default>
      <default class="geom">
        <geom size=".3"/>
      </default>
    </default>

    <worldbody>
      <frame euler="0 0 30">
        <geom size=".5" euler="0 0 20"/>
      </frame>

      <body>
        <frame pos="0 1 0" childclass="frame">
          <geom pos="0 1 0"/>
          <body pos="1 0 0" childclass="body">
            <geom pos="0 0 1"/>
            <geom pos="0 0 1" class="geom"/>
          </body>
        </frame>
      </body>

      <frame euler="0 0 30">
        <frame euler="0 0 20">
          <geom size=".6"/>
        </frame>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_THAT(m->geom_size[0], .5);
  EXPECT_THAT(m->geom_size[3], .6);
  EXPECT_THAT(m->geom_size[6], .1);
  EXPECT_THAT(m->geom_size[9], .2);
  EXPECT_THAT(m->geom_size[12], .3);
}

TEST_F(XMLReaderTest, DuplicateFrameName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="frame1" euler="0 0 30">
        <geom size=".1"/>
      </frame>

      <frame name="frame1" euler="0 0 30">
        <geom size=".1"/>
      </frame>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull()) << error.data();
  EXPECT_THAT(error.data(), HasSubstr("repeated name 'frame1'"));
}

TEST_F(XMLReaderTest, ParseFrameScale) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <frame name="default"/>
      <frame name="uniform" scale="2"/>
      <frame name="vector" scale="1 2 3" scaling="geometry"/>
      <frame name="similar" scaling="similar"/>
      <frame name="mass" massscale="2"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjsFrame* frame = mjs_findFrame(spec, "default");
  EXPECT_THAT(frame->scale, ElementsAre(1, 1, 1));
  EXPECT_EQ(frame->scaling, mjSCALING_SIMILAR);

  frame = mjs_findFrame(spec, "uniform");
  EXPECT_THAT(frame->scale, ElementsAre(2, 2, 2));
  EXPECT_EQ(frame->scaling, mjSCALING_SIMILAR);

  frame = mjs_findFrame(spec, "vector");
  EXPECT_THAT(frame->scale, ElementsAre(1, 2, 3));
  EXPECT_EQ(frame->scaling, mjSCALING_GEOMETRY);

  frame = mjs_findFrame(spec, "similar");
  EXPECT_EQ(frame->scaling, mjSCALING_SIMILAR);
  EXPECT_EQ(frame->massscale, 0);

  frame = mjs_findFrame(spec, "mass");
  EXPECT_EQ(frame->massscale, 2);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, InvalidFrameScale) {
  constexpr std::pair<const char*, const char*> kCases[] = {
      {R"(<frame scale="1 2"/>)", "must have 1 or 3 values"},
      {R"(<frame scale="1 2 3 4"/>)", "too much data"},
      {R"(<frame scaling="material"/>)", "invalid keyword"},
      {R"(<frame scale="0"/>)", "frame scale must be finite and positive"},
      {R"(<frame scale="1 -1 1"/>)", "frame scale must be finite and positive"},
      {R"(<frame scale="1 2 1"/>)",
       "anisotropic frame scale requires scaling 'geometry'"},
      {R"(<frame massscale="-1"/>)",
       "frame massscale must be finite and non-negative"},
  };
  for (const auto& [frame, message] : kCases) {
    std::string xml =
        std::string("<mujoco><worldbody>") + frame + "</worldbody></mujoco>";
    std::array<char, 1024> error;
    MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
    EXPECT_THAT(m.get(), IsNull()) << frame;
    EXPECT_THAT(error.data(), HasSubstr(message)) << frame;
  }
}

// inertia tensor of a body, in body coordinates
static std::array<mjtNum, 9> BodyInertia(const mjModel* m, int body) {
  const mjtNum* inertia = m->body_inertia + 3 * body;
  mjtNum mat[9];
  mju_quat2Mat(mat, m->body_iquat + 4 * body);
  std::array<mjtNum, 9> res;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      res[3 * i + j] = 0;
      for (int k = 0; k < 3; k++) {
        res[3 * i + j] += mat[3 * i + k] * inertia[k] * mat[3 * j + k];
      }
    }
  }
  return res;
}

TEST_F(XMLReaderTest, InertialInFrame) {
  static constexpr char framed_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <frame pos="1 0 0" euler="0 0 90">
          <inertial pos=".5 0 0" euler="90 0 0" mass="1" diaginertia="2 3 4"/>
        </frame>
      </body>
      <body>
        <frame pos="0 1 0" euler="90 0 0">
          <inertial pos="0 0 .5" mass="2" fullinertia="2 3 4 .1 .2 .3"/>
        </frame>
      </body>
      <body>
        <frame pos="1 0 0" euler="0 0 90">
          <frame pos="0 0 1" euler="0 90 0">
            <inertial pos="0 .5 0" mass="3" diaginertia="2 3 4"/>
          </frame>
        </frame>
      </body>
    </worldbody>
  </mujoco>
  )";

  // the same inertial frames, written in body coordinates
  static constexpr char flat_xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <inertial pos="1 .5 0" quat="1 1 1 1" mass="1" diaginertia="2 3 4"/>
      </body>
      <body>
        <inertial pos="0 .5 0" mass="2" fullinertia="2 4 3 -.2 .1 -.3"/>
      </body>
      <body>
        <inertial pos=".5 0 1" quat="1 -1 1 1" mass="3" diaginertia="2 3 4"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::array<char, 1024> error;
  MjModelPtr framed =
      LoadModelFromString(framed_xml, error.data(), error.size());
  ASSERT_THAT(framed.get(), NotNull()) << error.data();
  MjModelPtr flat = LoadModelFromString(flat_xml, error.data(), error.size());
  ASSERT_THAT(flat.get(), NotNull()) << error.data();

  // principal axes are not unique, compare inertia tensors rather than iquat
  const mjtNum tol = MjTol(1e-13, 1e-6);
  for (int i = 1; i < flat->nbody; i++) {
    EXPECT_EQ(framed->body_mass[i], flat->body_mass[i]) << i;
    for (int j = 0; j < 3; j++) {
      EXPECT_NEAR(framed->body_ipos[3 * i + j], flat->body_ipos[3 * i + j], tol)
          << "body " << i << " ipos " << j;
    }
    std::array<mjtNum, 9> inertia = BodyInertia(framed.get(), i);
    std::array<mjtNum, 9> expected = BodyInertia(flat.get(), i);
    for (int j = 0; j < 9; j++) {
      EXPECT_NEAR(inertia[j], expected[j], tol)
          << "body " << i << " inertia " << j;
    }
  }
}

// ---------------------- test replicate parsing -------------------------------

TEST_F(XMLReaderTest, ParseReplicate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="texture" type="2d" builtin="checker" width="32" height="32"/>
      <material name="material" texture="texture" texrepeat="1 1" texuniform="true"/>
    </asset>

    <worldbody>
      <replicate count="101" offset="3 0 .1" euler="0 0 1.8">
        <body name="body" pos="0 -1 0">
          <joint type="slide"/>
          <geom name="g" size="1"/>
        </body>
      </replicate>

      <replicate count="2" offset="1 0 0">
        <replicate count="2" offset="0 1 0" sep="_">
          <geom name="geom" size="1" pos="0 0 1" material="material"/>
          <site name="site" pos="1 0 0"/>
        </replicate>
      </replicate>
    </worldbody>

    <sensor>
      <framepos name="sensor" objtype="site" objname="site"/>
    </sensor>

    <keyframe>
      <key name="keyframe" qpos="1"/>
    </keyframe>
  </mujoco>

  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), testing::NotNull()) << error.data();
  EXPECT_THAT(m->ngeom, 105);
  EXPECT_THAT(m->nsensor, 4);
  EXPECT_THAT(m->nbody, 102);

  // check that the separator is used correctly
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      char geom_name[mjMAXUINAME] = "";
      util::strcat_arr(geom_name, m->names + m->name_geomadr[2 * i + j]);
      EXPECT_THAT(m->geom_pos[6 * i + 3 * j + 0], i);
      EXPECT_THAT(m->geom_pos[6 * i + 3 * j + 1], j);
      EXPECT_THAT(m->geom_pos[6 * i + 3 * j + 2], 1);
      EXPECT_THAT(std::string(geom_name),
                  "geom_" + std::to_string(j) + std::to_string(i));
    }
  }

  // check that 3 digits are used if count > 99
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 10; ++k) {
        int ngeom = 100 * i + 10 * j + k;
        if (ngeom > 99) break;
        char geom_name[mjMAXUINAME] = "";
        util::strcat_arr(geom_name, m->names + m->name_geomadr[4 + ngeom]);
        EXPECT_THAT(
            std::string(geom_name),
            "g" + std::to_string(i) + std::to_string(j) + std::to_string(k));
      }
    }
  }

  // check body positions
  mjtNum pos[2] = {0, 0};
  const mjtNum tol = MjTol(1e-8, 1e-3);
  for (int i = 1; i < 102; ++i) {
    mjtNum theta = (i - 1) * 1.8 * mjPI / 180;
    EXPECT_NEAR(m->body_pos[3 * i + 0], pos[0] + sin(theta), tol) << i;
    EXPECT_NEAR(m->body_pos[3 * i + 1], pos[1] - cos(theta), tol) << i;
    EXPECT_NEAR(m->body_pos[3 * i + 2], (i - 1) * .1, tol);
    pos[0] += 3 * cos(theta);
    pos[1] += 3 * sin(theta);
  }

  // check that the final pose is correct
  int n = m->nbody - 1;
  EXPECT_NEAR(m->body_quat[4 * n + 0], 0, tol);
  EXPECT_EQ(m->body_quat[4 * n + 1], 0);
  EXPECT_EQ(m->body_quat[4 * n + 2], 0);
  EXPECT_EQ(m->body_quat[4 * n + 3], 1);

  // check that the keyframe is not replicated and is completed with defaults
  EXPECT_THAT(m->nkey, 1);
  EXPECT_THAT(m->nq, 101);
  for (int j = 0; j < m->nq; j++) {
    EXPECT_THAT(m->key_qpos[j], j == 0 ? 1 : 0) << j;
  }
}

TEST_F(XMLReaderTest, ParseReplicatePartialReference) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 1"/>
      <site name="world"/>

      <replicate count="2" offset="1 0 0">
        <body name="replicated">
          <freejoint/>
          <site name="replicated"/>
          <geom name="replicated" size="1"/>
        </body>
      </replicate>
    </worldbody>

    <contact>
      <pair geom1="floor" geom2="replicated" condim="1"/>
      <exclude body1="world" body2="replicated" />
    </contact>

    <tendon>
      <spatial>
        <site site="world"/>
        <site site="replicated"/>
      </spatial>
    </tendon>

    <sensor>
      <framepos name="sensor" objtype="site" objname="replicated" reftype="site" refname="world"/>
    </sensor>
  </mujoco>

  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), testing::NotNull()) << error.data();

  EXPECT_THAT(m->nbody, 3);
  EXPECT_THAT(m->ngeom, 3);
  EXPECT_THAT(m->npair, 2);
  EXPECT_THAT(m->nexclude, 2);
  EXPECT_THAT(m->ntendon, 2);
  EXPECT_THAT(m->nsensor, 2);
}

TEST_F(XMLReaderTest, ReplicateCannotHaveJoints) {
  static constexpr char joint_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset="2 0 0">
        <joint type="hinge"/>
        <geom size=".1"/>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(joint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("joint cannot be a direct child of replicate"));
  EXPECT_THAT(error.data(), HasSubstr("line 5"));

  static constexpr char freejoint_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <replicate count="2" offset="2 0 0">
          <freejoint/>
          <geom size=".1"/>
        </replicate>
      </body>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(freejoint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("joint cannot be a direct child of replicate"));

  static constexpr char body_joint_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset="1 0 0">
        <body>
          <joint type="hinge"/>
          <geom size=".1"/>
        </body>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(body_joint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->njnt, 2);
}

TEST_F(XMLReaderTest, ReplicateCannotHaveJointsInFrame) {
  static constexpr char joint_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <replicate count="2" offset="2 0 0">
          <frame pos="0 0 1">
            <joint type="hinge"/>
            <geom size=".1"/>
          </frame>
        </replicate>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(joint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("joint cannot be a direct child of replicate"));
  EXPECT_THAT(error.data(), HasSubstr("line 7"));

  static constexpr char nested_freejoint_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <replicate count="2" offset="2 0 0">
          <frame>
            <frame>
              <freejoint/>
              <geom size=".1"/>
            </frame>
          </frame>
        </replicate>
      </body>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(nested_freejoint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("joint cannot be a direct child of replicate"));
  EXPECT_THAT(error.data(), HasSubstr("line 8"));

  // a joint inside a body nested in a frame inside replicate is allowed
  static constexpr char body_joint_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset="1 0 0">
        <frame>
          <body>
            <joint type="hinge"/>
            <geom size=".1"/>
          </body>
        </frame>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(body_joint_xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->njnt, 2);
}

TEST_F(XMLReaderTest, ReplicateCannotHaveJointsFromInclude) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b">
        <replicate count="2" offset="2 0 0">
          <include file="joint.xml"/>
        </replicate>
      </body>
    </worldbody>
  </mujoco>)";

  static constexpr char joint_xml[] = R"(
  <mujoco>
    <joint type="hinge"/>
    <geom size=".1"/>
  </mujoco>)";

  static constexpr char body_xml[] = R"(
  <mujoco>
    <body>
      <joint type="hinge"/>
      <geom size=".1"/>
    </body>
  </mujoco>)";

  static constexpr char body_include_xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset="2 0 0">
        <include file="body.xml"/>
      </replicate>
    </worldbody>
  </mujoco>)";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "joint.xml", joint_xml, sizeof(joint_xml));
  mj_addBufferVFS(vfs.get(), "body.xml", body_xml, sizeof(body_xml));

  std::array<char, 1024> error;
  MjModelPtr model =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("joint cannot be a direct child of replicate"));

  // a joint inside an included body nested in replicate is allowed
  model = LoadModelFromString(body_include_xml, error.data(), error.size(),
                              vfs.get());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  if (model) {
    EXPECT_EQ(model->njnt, 2);
  }
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, ParseReplicateDefaultPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="body">
        <geom type="capsule"/>
      </default>
    </default>

    <worldbody>
      <replicate count="2" euler="0 0 16.36" sep="-">
        <body name="torso" pos="-5 0 1.282" childclass="body">
          <body name="head" pos="0 0 .19">
            <geom name="head" type="sphere" size=".09"/>
          </body>
        </body>
      </replicate>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();

  mjsBody* torso = mjs_findBody(spec, "torso-0");
  EXPECT_THAT(torso, NotNull());

  mjsDefault* def = mjs_getDefault(torso->element);
  EXPECT_THAT(def, NotNull());
  EXPECT_THAT(def->geom->type, mjGEOM_CAPSULE);

  mjsBody* head = mjs_findBody(spec, "head-0");
  EXPECT_THAT(head, NotNull());

  def = mjs_getDefault(head->element);
  EXPECT_THAT(def, NotNull());
  EXPECT_THAT(def->geom->type, mjGEOM_CAPSULE);

  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, ParseReplicateRepeatedName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <compiler>
      <lengthrange mode="all"/>
    </compiler>

    <worldbody>
      <light pos="0 0 10"/>

      <replicate offset="0 .1 0" count="2">
        <site name="a" size=".02"/>
        <body pos="1 0 1">
          <joint axis="0 -1 0" range="0 90"/>
          <geom type="capsule" size=".02" fromto="0 0 0 0 0 -1"/>
          <site name="b" pos="0 0 -1"/>
        </body>
      </replicate>
    </worldbody>

    <tendon>
      <spatial name="b">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>

    <actuator>
      <position name="b" tendon="b0" ctrlrange="0 3" kp="100" dampratio="1"/>
      <position name="b" tendon="b1" ctrlrange="0 3" kp="100" dampratio="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, IsNull()) << error.data();
  EXPECT_THAT(error.data(), HasSubstr("repeated name 'b' in actuator"));
  EXPECT_THAT(error.data(), HasSubstr("Element 'position'"));
}

TEST_F(XMLReaderTest, RepeatedPrefix) {
  static constexpr char parent[] = R"(
  <mujoco>
    <asset>
      <model name="1" file="child_1.xml" content_type="text/xml" />
      <model name="2" file="child_2.xml" content_type="text/xml" />
    </asset>

    <worldbody>
      <attach model="1" body="1" prefix="prefix-"/>
      <replicate count="1">
        <attach model="2" body="2" prefix="prefix-"/>
      </replicate>
    </worldbody>
  </mujoco>
  )";

  static constexpr char child_1[] = R"(
  <mujoco>
    <asset>
      <model name="2" file="child_2.xml" content_type="text/xml"/>
    </asset>

    <worldbody>
      <body name="1">
        <body name="2">
          <attach model="2" body="2" prefix="prefix2"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char child_2[] = R"(
  <mujoco>
    <worldbody>
      <body name="2"/>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child_1.xml", child_1, sizeof(child_1));
  mj_addBufferVFS(vfs.get(), "child_2.xml", child_2, sizeof(child_2));

  std::array<char, 1024> err;
  mjSpec* c2 = mj_parseXMLString(child_2, 0, err.data(), err.size());
  EXPECT_THAT(c2, NotNull()) << err.data();
  mjSpec* c1 = mj_parseXMLString(child_1, vfs.get(), err.data(), err.size());
  EXPECT_THAT(c1, NotNull()) << err.data();
  mj_deleteSpec(c1);
  mj_deleteSpec(c2);

  mjSpec* spec = mj_parseXMLString(parent, vfs.get(), err.data(), err.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(err.data(), HasSubstr("cannot attach"));
  mj_deleteSpec(spec);
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, ParseReplicateExcludeTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="winch" pos="-.01 0 .35">
        <joint name="winch" damping="1"/>
        <geom type="cylinder" size=".015 .01"/>
        <site name="anchor" pos=".1 0 .04"/>
      </body>
      <site name="pulley" pos=".1 0 .32"/>
      <site name="hook_left" pos=".08 0 .3"/>
      <site name="hook_right" pos=".12 0 .3"/>
      <body name="sphere" pos=".1 0 .2">
        <freejoint/>
        <geom type="sphere" size=".03"/>
        <site name="pin_left" pos="-.025 0 .025"/>
        <site name="pin_right" pos=".025 0 .025"/>
      </body>

      <replicate count="4" offset=".025 0 0">
        <replicate count="4" offset="0 .025 0">
          <replicate count="4" offset="0 0 .025">
            <body pos=".06 -.04 .05">
              <geom type="sphere" size=".012"/>
            </body>
          </replicate>
        </replicate>
      </replicate>
    </worldbody>

    <tendon>
      <spatial range="0 .19" limited="true">
        <site site="anchor"/>
        <site site="pulley"/>
        <pulley divisor="3"/>
        <site site="pulley"/>
        <site site="hook_left"/>
        <site site="pin_left"/>
        <pulley divisor="3"/>
        <site site="pulley"/>
        <site site="hook_right"/>
        <site site="pin_right"/>
      </spatial>
    </tendon>

    <actuator>
      <position name="winch" joint="winch" ctrlrange="-.7 .5" ctrllimited="true" kp="10"/>
  </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, 0);
  EXPECT_THAT(m, NotNull());
  EXPECT_THAT(m->nbody, 67);
  EXPECT_THAT(m->ngeom, 66);
  EXPECT_THAT(m->nsite, 6);
  EXPECT_THAT(m->nu, 1);
  EXPECT_THAT(m->ntendon, 1);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, ParseReplicateWithTendon) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <replicate count="2" offset=".025 0 0">
        <replicate count="2" offset="0 .025 0">
          <replicate count="2" offset="0 0 .025">
            <body name="winch" pos="-.01 0 .35">
              <joint name="winch" damping="1"/>
              <geom type="cylinder" size=".015 .01"/>
              <site name="anchor" pos=".1 0 .04"/>
            </body>
            <site name="pulley" pos=".1 0 .32"/>
            <site name="hook_left" pos=".08 0 .3"/>
            <site name="hook_right" pos=".12 0 .3"/>
            <body name="sphere" pos=".1 0 .2">
              <freejoint/>
              <geom type="sphere" size=".03"/>
              <site name="pin_left" pos="-.025 0 .025"/>
              <site name="pin_right" pos=".025 0 .025"/>
            </body>
            <body pos=".06 -.04 .05">
              <geom type="sphere" size=".012"/>
            </body>
          </replicate>
        </replicate>
      </replicate>
    </worldbody>

    <tendon>
      <spatial range="0 .19" limited="true" name="tendon">
        <site site="anchor"/>
        <site site="pulley"/>
        <pulley divisor="3"/>
        <site site="pulley"/>
        <site site="hook_left"/>
        <site site="pin_left"/>
        <pulley divisor="3"/>
        <site site="pulley"/>
        <site site="hook_right"/>
        <site site="pin_right"/>
      </spatial>
    </tendon>

    <actuator>
      <position name="winch" joint="winch" ctrlrange="-.7 .5" ctrllimited="true" kp="10"/>
      <position name="tendon" tendon="tendon" ctrlrange="0 1" kp="100" dampratio="1"/>
  </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, 0);
  EXPECT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_THAT(m->nbody, 25);
  EXPECT_THAT(m->ngeom, 24);
  EXPECT_THAT(m->nsite, 48);
  EXPECT_THAT(m->nu, 16);
  EXPECT_THAT(m->ntendon, 8);
  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a keyframe describes the model with its replicas in place
TEST_F(XMLReaderTest, ParseReplicateKeyframe) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="before">
        <joint type="slide"/>
        <geom size="1"/>
      </body>
      <replicate count="2" offset="3 0 0">
        <body name="replica">
          <joint name="replica" type="slide"/>
          <geom size="1"/>
        </body>
        <body name="mocap" mocap="true"/>
      </replicate>
      <body name="after">
        <joint type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="replica" dyntype="filter" dynprm="1"/>
    </actuator>

    <keyframe>
      <key name="key" qpos="1 2 3 4" qvel="5 6 7 8" act="1 2" ctrl="3 4" mpos="1 2 3 4 5 6" mquat="0 1 0 0 0 0 1 0"/>
    </keyframe>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nq, 4);
  ASSERT_EQ(m->na, 2);
  ASSERT_EQ(m->nmocap, 2);

  // the keyframe is not replicated and has the values which were written
  ASSERT_EQ(m->nkey, 1);
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_KEY, 0), "key");
  EXPECT_THAT(AsVector(m->key_qpos, 4), ElementsAre(1, 2, 3, 4));
  EXPECT_THAT(AsVector(m->key_qvel, 4), ElementsAre(5, 6, 7, 8));
  EXPECT_THAT(AsVector(m->key_act, 2), ElementsAre(1, 2));
  EXPECT_THAT(AsVector(m->key_ctrl, 2), ElementsAre(3, 4));
  EXPECT_THAT(AsVector(m->key_mpos, 6), ElementsAre(1, 2, 3, 4, 5, 6));
  EXPECT_THAT(AsVector(m->key_mquat, 8), ElementsAre(0, 1, 0, 0, 0, 0, 1, 0));
}

// the model of issue #3071: a replicate in an included file is expanded
// before the bodies which follow it are parsed
TEST_F(XMLReaderTest, ParseReplicateKeyframeInclude) {
  static constexpr char robot_xml[] = R"(
  <mujoco model="robot">
    <worldbody>
      <body name="arm">
        <joint name="j1" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.1 0.1 0.1"/>
        <body name="replicate_body">
          <replicate count="2" sep="-">
            <site name="s"/>
          </replicate>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char scene_xml[] = R"(
  <mujoco model="scene">
    <include file="robot.xml"/>

    <worldbody>
      <body name="cube" pos="0 0 1">
        <freejoint/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>

    <keyframe>
      <key name="test" qpos="0.5 1 2 3 0 1 0 0"/>
    </keyframe>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "robot.xml", robot_xml, sizeof(robot_xml));

  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(scene_xml, error.data(), error.size(), vfs.get());
  mj_deleteVFS(vfs.get());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nq, 8);
  ASSERT_EQ(m->nkey, 1);
  EXPECT_THAT(AsVector(m->key_qpos, 8), ElementsAre(0.5, 1, 2, 3, 0, 1, 0, 0));
}

// the keyframes of a model which is attached inside a replicate are added
// once and set the first replica
TEST_F(XMLReaderTest, ParseReplicateAttachKeyframe) {
  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint" type="slide"/>
        <geom size="1"/>
      </body>
    </worldbody>

    <actuator>
      <general joint="joint" dyntype="filter" dynprm="1"/>
    </actuator>

    <keyframe>
      <key name="key" qpos="1" act="2" ctrl="3"/>
    </keyframe>
  </mujoco>
  )";

  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <model name="child" file="child.xml"/>
    </asset>

    <worldbody>
      <replicate count="2" offset="3 0 0">
        <attach model="child" body="body" prefix="child_"/>
      </replicate>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child.xml", child_xml, sizeof(child_xml));

  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(xml, error.data(), error.size(), vfs.get());
  mj_deleteVFS(vfs.get());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nq, 2);
  ASSERT_EQ(m->na, 2);
  ASSERT_EQ(m->nu, 2);

  ASSERT_EQ(m->nkey, 1);
  EXPECT_STREQ(mj_id2name(m.get(), mjOBJ_KEY, 0), "child_key");
  EXPECT_THAT(AsVector(m->key_qpos, 2), ElementsAre(1, 0));
  EXPECT_THAT(AsVector(m->key_act, 2), ElementsAre(2, 0));
  EXPECT_THAT(AsVector(m->key_ctrl, 2), ElementsAre(3, 0));
}

// ---------------------- test spec assets parsing -----------------------------

TEST_F(XMLReaderTest, ParseSpecAssets) {
  std::array<char, 1024> er;

  static const char* const kParentPath = "xml/testdata/parent.xml";
  static const char* const kChildPath = "xml/testdata/child.xml";

  const std::string xml_parent = GetTestDataFilePath(kParentPath);
  const std::string xml_child = GetTestDataFilePath(kChildPath);

  mjModel* parent = mj_loadXML(xml_parent.c_str(), 0, er.data(), er.size());
  EXPECT_THAT(parent, NotNull()) << er.data();

  mjModel* child = mj_loadXML(xml_child.c_str(), 0, er.data(), er.size());
  EXPECT_THAT(child, NotNull()) << er.data();

  mj_deleteModel(parent);
  mj_deleteModel(child);
}

TEST_F(XMLReaderTest, AttachSpecAssets) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <asset>
      <model name="child" file="xml_child.xml"/>
    </asset>
    <worldbody>
      <body name="parent">
        <geom name="geom" size="2"/>
        <replicate count="2" offset="0 0 1">
          <attach model="child" body="body" prefix="other"/>
        </replicate>
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

  static constexpr char xml_expected[] = R"(
  <mujoco>
    <worldbody>
      <body name="parent">
        <geom name="geom" size="2"/>
        <body name="otherbody0">
          <geom name="othergeom0" size="1" pos="2 0 0"/>
        </body>
        <body name="otherbody1" pos="0 0 1">
          <geom name="othergeom1" size="1" pos="2 0 0"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "xml_child.xml", xml_child, sizeof(xml_child));

  std::array<char, 1024> er;
  MjModelPtr model =
      LoadModelFromString(xml_parent, er.data(), er.size(), vfs.get());
  EXPECT_THAT(model.get(), NotNull()) << er.data();

  MjModelPtr expected = LoadModelFromString(xml_expected, er.data(), er.size());
  EXPECT_THAT(expected.get(), NotNull()) << er.data();

  mjtNum tol = 0;
  std::string field = "";
  EXPECT_LE(CompareModel(model.get(), expected.get(), field), tol)
      << "Expected and attached models are different!\n"
      << "Different field: " << field << '\n';
  ;
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, InvalidAttach) {
  static constexpr char xml_parent[] = R"(
  <mujoco>
    <asset>
      <model name="other" file="child.xml"/>
    </asset>
    <worldbody>
      <body name="parent">
        <joint name="joint1"/>
        <geom size="2"/>
        <attach model="other" body="body" prefix="_"/>
      </body>
    </worldbody>

    <actuator>
      <motor name="_actuator" joint="joint1"/>
    </actuator>
  </mujoco>
  )";

  static constexpr char xml_child[] = R"(
  <mujoco model="child">
    <worldbody>
      <body name="body">
        <joint name="joint2"/>
        <geom size="1" pos="2 0 0"/>
      </body>
    </worldbody>

    <actuator>
      <motor name="actuator" joint="joint2"/>
    </actuator>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child.xml", xml_child, sizeof(xml_child));

  std::array<char, 1024> er;
  MjModelPtr model =
      LoadModelFromString(xml_parent, er.data(), er.size(), vfs.get());

  EXPECT_THAT(model.get(), IsNull()) << er.data();
  EXPECT_THAT(er.data(), HasSubstr("repeated name '_actuator' in actuator"));
  EXPECT_THAT(er.data(), HasSubstr("Element 'attach'"));
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, LookupCompilerOptionWithoutSpecCopy) {
  static constexpr char child_xml[] = R"(
  <mujoco>
    <compiler angle="radian"/>

    <worldbody>
      <body name="child">
        <geom type="box" size="1 1 1"/>
        <joint name="child_joint" range="-3.1415926 3.1415926"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char parent_xml[] = R"(
  <mujoco>
    <asset>
      <model name="child" file="child.xml"/>
    </asset>

    <worldbody>
      <body name="parent">
        <geom type="box" size="1 1 1"/>
        <joint name="parent_joint" range="-180 180"/>
        <attach model="child" body="child" prefix="child_"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child.xml", child_xml, sizeof(child_xml));
  mj_addBufferVFS(vfs.get(), "parent.xml", parent_xml, sizeof(parent_xml));

  std::array<char, 1024> error;
  auto* spec =
      mj_parseXMLString(parent_xml, vfs.get(), error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* model = mj_compile(spec, vfs.get());
  EXPECT_THAT(model, NotNull());
  EXPECT_THAT(model->njnt, 2);
  EXPECT_NEAR(model->jnt_range[0], -3.14159, 1e-5);
  EXPECT_NEAR(model->jnt_range[1], 3.14159, 1e-5);
  EXPECT_NEAR(model->jnt_range[2], -3.14159, 1e-5);
  EXPECT_NEAR(model->jnt_range[3], 3.14159, 1e-5);

  mjSpec* copied_spec = mj_copySpec(spec);
  ASSERT_THAT(copied_spec, NotNull());
  mjModel* copied_model = mj_compile(copied_spec, vfs.get());
  EXPECT_THAT(copied_model, NotNull());
  EXPECT_THAT(copied_model->njnt, 2);
  EXPECT_NEAR(copied_model->jnt_range[0], -3.14159, 1e-5);
  EXPECT_NEAR(copied_model->jnt_range[1], 3.14159, 1e-5);
  EXPECT_NEAR(copied_model->jnt_range[2], -3.14159, 1e-5);
  EXPECT_NEAR(copied_model->jnt_range[3], 3.14159, 1e-5);

  mj_deleteSpec(spec);
  mj_deleteSpec(copied_spec);
  mj_deleteModel(model);
  mj_deleteModel(copied_model);

  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, ResizeKeyframeAfterParsing) {
  static constexpr char parent_xml[] = R"(
  <mujoco>
    <asset>
      <model name="child" file="child.xml"/>
    </asset>
    <worldbody>
      <attach model="child" body="world" prefix="child_"/>
      <body name="body">
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <keyframe>
      <key name="key" qpos="1"/>
    </keyframe>
  </mujoco>
  )";

  static constexpr char child_xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="body">
        <joint name="joint"/>
        <geom size="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_addBufferVFS(vfs.get(), "child.xml", child_xml, sizeof(child_xml));

  std::array<char, 1024> error;
  MjModelPtr m =
      LoadModelFromString(parent_xml, error.data(), error.size(), vfs.get());
  EXPECT_THAT(m.get(), NotNull()) << error.data();
  mj_deleteVFS(vfs.get());
}

TEST_F(XMLReaderTest, CheckTextureGridlayoutParsing) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <texture name="test_skybox" type="skybox" file="dummy_image.png"
               gridsize="2 4" gridlayout="LFRB.D.."/>
    </asset>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  EXPECT_THAT(spec, NotNull()) << error.data();
  auto skybox_spec =
      mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, "test_skybox"));
  EXPECT_THAT(skybox_spec, NotNull()) << error.data();
  EXPECT_EQ(std::strlen(skybox_spec->gridlayout), 8);
  EXPECT_TRUE(std::strcmp(skybox_spec->gridlayout, "LFRB.D..") == 0);

  mj_deleteSpec(spec);
}

// ----------------------- test camera parsing ---------------------------------

TEST_F(XMLReaderTest, CameraInvalidFovyAndSensorsize) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <camera fovy="1" sensorsize="1 1" resolution="100 100"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("at most one of 'fovy', 'sensorsize'"));
  EXPECT_THAT(error.data(), HasSubstr("line 6"));
}

TEST_F(XMLReaderTest, CameraPrincipalRequiresSensorsize) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <camera principal="1 1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("focal/principal require sensorsize"));
  EXPECT_THAT(error.data(), HasSubstr("line 6"));
}

TEST_F(XMLReaderTest, CameraSensorsizeRequiresResolution) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <camera sensorsize="1 1" resolution="0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(m.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("requires positive resolution"));
  EXPECT_THAT(error.data(), HasSubstr("line 6"));
}

// ----------------------- test inertia parsing --------------------------------

TEST_F(XMLReaderTest, InvalidInertialOrientation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <inertial pos="0 0 0" mass="1" quat="1 0 0 0" fullinertia="1 1 1 0 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("at most one of 'fullinertia', 'quat', 'axisangle', "
                        "'xyaxes', 'zaxis', 'euler' can be specified"));
}

TEST_F(XMLReaderTest, ReadShellParameter) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="example_mesh"
        vertex="0 0 0  1 0 0  0 1 0  0 0 1"
        face="0 2 1  2 0 3" inertia="shell"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="example_mesh"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
}

// ----------------------- test builtin mesh parsing ---------------------------

TEST_F(XMLReaderTest, ReadWedgeMesh) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="wedge" builtin="wedge" params="25 25 180 90 0"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="wedge"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
}

TEST_F(XMLReaderTest, BuiltinAndFile) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="wedge" builtin="wedge" vertex="0 0 0 1 0 0 0 1 0 0 1 0"
            params="25 25 180 90 0"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="wedge"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("at most one of 'builtin', 'vertex' can be specified"));
}

TEST_F(XMLReaderTest, MakePlateNoParameters) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="plate" builtin="plate"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="plate" contype="0" conaffinity="0"/>
    </worldbody>
  </mujoco>)";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("required attribute missing: 'params'"));
}

TEST_F(XMLReaderTest, MakePlateTooFewParameters) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="plate" builtin="plate" params="1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="plate" contype="0" conaffinity="0"/>
    </worldbody>
  </mujoco>)";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("Plate builtin mesh type requires 2 parameters"));
}

TEST_F(XMLReaderTest, MakePlateInvalidParameters) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="plate" builtin="plate" params="-1 -1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="plate" contype="0" conaffinity="0"/>
    </worldbody>
  </mujoco>)";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("resolutions must be positive"));
}

// ----------------------- test flex yield parsing ---------------------------

TEST_F(XMLReaderTest, FlexYieldOneOrTwoValues) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="one" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" elastic2d="both" yield="0.02"/>
      </flexcomp>
      <flexcomp name="two" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" elastic2d="both" yield="0.03 0.004"/>
      </flexcomp>
      <flexcomp name="box" type="grid" count="2 2 2" spacing=".1 .1 .1" dim="3">
        <elasticity young="1000" yield="0.05"/>
      </flexcomp>
    </worldbody>
    <deformable>
      <flex name="direct" dim="2" body="world world world" vertex="0 0 0 1 0 0 0 1 0"
            element="0 1 2">
        <elasticity yield="0.5 0.25"/>
      </flex>
      <flex name="direct3d" dim="3" body="world world world world"
            vertex="0 0 0 1 0 0 0 1 0 0 0 1" element="0 1 2 3">
        <elasticity yield="0.5"/>
      </flex>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // a single value applies to both stretching and bending of 2D flexes, and is
  // only the stretching yield strain of 3D flexes
  mjsFlex* one = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "one"));
  mjsFlex* two = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "two"));
  mjsFlex* box = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "box"));
  mjsFlex* direct = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "direct"));
  mjsFlex* direct3d = mjs_asFlex(mjs_findElement(spec, mjOBJ_FLEX, "direct3d"));
  ASSERT_THAT(one, NotNull());
  ASSERT_THAT(two, NotNull());
  ASSERT_THAT(box, NotNull());
  ASSERT_THAT(direct, NotNull());
  ASSERT_THAT(direct3d, NotNull());
  EXPECT_THAT(one->stressyield, ElementsAre(0.02, 0.02));
  EXPECT_THAT(two->stressyield, ElementsAre(0.03, 0.004));
  EXPECT_THAT(box->stressyield, ElementsAre(0.05, 0));
  EXPECT_THAT(direct->stressyield, ElementsAre(0.5, 0.25));
  EXPECT_THAT(direct3d->stressyield, ElementsAre(0.5, 0));
  mj_deleteSpec(spec);

  // the compiled model has one row of (stretch, bend) yield strains per flex
  static constexpr char xml_model[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="one" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" elastic2d="both" yield="0.02"/>
      </flexcomp>
      <flexcomp name="two" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" elastic2d="both" yield="0.03 0.004"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml_model, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  int id1 = mj_name2id(model.get(), mjOBJ_FLEX, "one");
  int id2 = mj_name2id(model.get(), mjOBJ_FLEX, "two");
  EXPECT_FLOAT_EQ(model->flex_yield[2 * id1], 0.02);
  EXPECT_FLOAT_EQ(model->flex_yield[2 * id1 + 1], 0.02);
  EXPECT_FLOAT_EQ(model->flex_yield[2 * id2], 0.03);
  EXPECT_FLOAT_EQ(model->flex_yield[2 * id2 + 1], 0.004);
}

TEST_F(XMLReaderTest, FlexYieldErrors) {
  std::array<char, 1024> error;

  // at most two values
  static constexpr char xml_three[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" yield="0.1 0.2 0.3"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml_three, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("has too much data"));

  // the bending yield strain cannot be negative
  static constexpr char xml_negative[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="3 3 1" spacing=".1 .1 .1" dim="2">
        <elasticity young="1000" thickness="0.01" yield="0.1 -0.2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(xml_negative, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("yield cannot be negative"));

  // 3D and interpolated flexes have no bending plasticity: a positive bending
  // yield strain is an error
  static constexpr char xml_3d[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="2 2 2" spacing=".1 .1 .1" dim="3">
        <elasticity young="1000" yield="0.05 0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(xml_3d, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("bending yield requires a non-interpolated 2d flex"));
  static constexpr char xml_interp[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="2 2 2" spacing=".1 .1 .1" dim="3"
                dof="trilinear">
        <elasticity young="1000" yield="0.05 0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(xml_interp, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("bending yield requires a non-interpolated 2d flex"));

  // a single value on an interpolated flex is only the stretching yield strain
  static constexpr char xml_interp_single[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="2 2 2" spacing=".1 .1 .1" dim="3"
                dof="trilinear">
        <elasticity young="1000" yield="0.05"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(xml_interp_single, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_FLOAT_EQ(model->flex_yield[0], 0.05);
  EXPECT_EQ(model->flex_yield[1], 0);
  EXPECT_EQ(model->flex_plasticmode[0], 2);

  // 1D flexes have no plasticity: a positive bending yield strain is an error
  static constexpr char xml_1d[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="f" type="grid" count="4 1 1" spacing=".1 .1 .1" dim="1">
        <elasticity yield="0 0.2"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  model = LoadModelFromString(xml_1d, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("flex plasticity requires a 2d or 3d flex"));
}

// ----------------------- test skin parsing --------------------------------

TEST_F(XMLReaderTest, ReadsSkinGroups) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 -1 .6" name="B0_parent">
        <flexcomp name="B0" type="grid" count="4 4 1" spacing=".2 .2 .2" group="2" radius=".1" dim="2">
          <edge equality="true"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
      <body pos="-1 1 .6" name="B1_parent">
        <flexcomp name="B1" type="grid" count="4 4 1" spacing=".2 .2 .2" group="4" radius=".1" dim="2">
          <edge equality="true"/>
          <contact selfcollide="none"/>
        </flexcomp>
      </body>
    </worldbody>
    <deformable>
      <skin name="B0Skin" group="4" rgba="1 1 0 1" inflate="0.1"
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
      <skin name="B1Skin" group="2" rgba="0 1 1 1" inflate="0.1"
        vertex="0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"
        face="0 4 5 0 5 1 1 5 6 1 6 2 2 6 7 2 7 3 4 8 9 4 9 5 5 9 10 5 10 6 6 10 11 6 11 7 8
              12 13 8 13 9 9 13 14 9 14 10 10 14 15 10 15 11 16 21 20 16 17 21 17 22 21 17 18
              22 18 23 22 18 19 23 20 25 24 20 21 25 21 26 25 21 22 26 22 27 26 22 23 27 24 29
              28 24 25 29 25 30 29 25 26 30 26 31 30 26 27 31 0 20 4 0 16 20 4 24 8 4 20 24 8
              28 12 8 24 28 3 7 23 3 23 19 7 11 27 7 27 23 11 15 31 11 31 27 0 1 17 0 17 16 1
              2 18 1 18 17 2 3 19 2 19 18 12 29 13 12 28 29 13 30 14 13 29 30 14 31 15 14 30 31">
        <bone body="B1_0" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 16" vertweight="1 1"/>
        <bone body="B1_1" bindpos="0 0 0" bindquat="1 0 0 0" vertid="1 17" vertweight="1 1"/>
        <bone body="B1_2" bindpos="0 0 0" bindquat="1 0 0 0" vertid="2 18" vertweight="1 1"/>
        <bone body="B1_3" bindpos="0 0 0" bindquat="1 0 0 0" vertid="3 19" vertweight="1 1"/>
        <bone body="B1_4" bindpos="0 0 0" bindquat="1 0 0 0" vertid="4 20" vertweight="1 1"/>
        <bone body="B1_5" bindpos="0 0 0" bindquat="1 0 0 0" vertid="5 21" vertweight="1 1"/>
        <bone body="B1_6" bindpos="0 0 0" bindquat="1 0 0 0" vertid="6 22" vertweight="1 1"/>
        <bone body="B1_7" bindpos="0 0 0" bindquat="1 0 0 0" vertid="7 23" vertweight="1 1"/>
        <bone body="B1_8" bindpos="0 0 0" bindquat="1 0 0 0" vertid="8 24" vertweight="1 1"/>
        <bone body="B1_9" bindpos="0 0 0" bindquat="1 0 0 0" vertid="9 25" vertweight="1 1"/>
        <bone body="B1_10" bindpos="0 0 0" bindquat="1 0 0 0" vertid="10 26" vertweight="1 1"/>
        <bone body="B1_11" bindpos="0 0 0" bindquat="1 0 0 0" vertid="11 27" vertweight="1 1"/>
        <bone body="B1_12" bindpos="0 0 0" bindquat="1 0 0 0" vertid="12 28" vertweight="1 1"/>
        <bone body="B1_13" bindpos="0 0 0" bindquat="1 0 0 0" vertid="13 29" vertweight="1 1"/>
        <bone body="B1_14" bindpos="0 0 0" bindquat="1 0 0 0" vertid="14 30" vertweight="1 1"/>
        <bone body="B1_15" bindpos="0 0 0" bindquat="1 0 0 0" vertid="15 31" vertweight="1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  int flexid1 = mj_name2id(model.get(), mjOBJ_FLEX, "B0");
  int flexid2 = mj_name2id(model.get(), mjOBJ_FLEX, "B1");
  EXPECT_THAT(model->flex_group[flexid1], 2);
  EXPECT_THAT(model->skin_group[0], 4);
  EXPECT_THAT(model->flex_group[flexid2], 4);
  EXPECT_THAT(model->skin_group[1], 2);
}

TEST_F(XMLReaderTest, InvalidSkinGroup) {
  static constexpr char xml[] = R"(
  <mujoco>
    <deformable>
      <skin group="6"/>
    </deformable>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr("skin group must be between 0 and 5\nElement 'skin', line 4"));
}

TEST_F(XMLReaderTest, Orthographic) {
  static constexpr char xml[] = R"(
  <mujoco>
    <visual>
      <global orthographic="true" fovy="5"/>
      <map znear="0.01"/>
    </visual>

    <default>
      <camera projection="orthographic"/>
    </default>

    <worldbody>
      <camera name="fovy=1" pos=".5 .5 2" projection="orthographic" fovy="1"/>
      <camera name="fovy=2" pos="0 0 2" fovy="2"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();

  EXPECT_EQ(model->vis.global.orthographic, 1);
  EXPECT_EQ(model->cam_projection[0], mjPROJ_ORTHOGRAPHIC);
  EXPECT_EQ(model->cam_projection[1], mjPROJ_ORTHOGRAPHIC);
  EXPECT_EQ(model->cam_fovy[0], 1);
  EXPECT_EQ(model->cam_fovy[1], 2);
}

// ------------- test height-field parsing -------------------------------------

using HfieldParsingTest = MujocoTest;

TEST_F(HfieldParsingTest, NoData) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="hf" nrow="4" ncol="3" size="0.5 0.5 1 0.1"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->hfield_nrow[0], 4);
  EXPECT_EQ(model->hfield_ncol[0], 3);
  EXPECT_EQ(model->hfield_size[0], 0.5);
  EXPECT_EQ(model->hfield_size[1], 0.5);
  EXPECT_EQ(model->hfield_size[2], 1);
  EXPECT_MJTNUM_EQ(model->hfield_size[3], 0.1);
}

TEST_F(HfieldParsingTest, HasDataBadSize) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <hfield name="hf" nrow="3" ncol="2" size="0.5 0.5 1 0.1"
              elevation="1 2
                         3 4
                         5 6 7"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("data length must match nrow*ncol"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(HfieldParsingTest, HasData) {
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
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->hfield_nrow[0], 3);
  EXPECT_EQ(model->hfield_ncol[0], 2);
  EXPECT_EQ(model->hfield_size[0], 0.5);
  EXPECT_EQ(model->hfield_size[1], 0.5);
  EXPECT_EQ(model->hfield_size[2], 1);
  EXPECT_MJTNUM_EQ(model->hfield_size[3], 0.1);

  // offset (minimum) and scaling (maximum) from normalizing operation
  float offset = 1.0;
  float scale = 6.0 - offset;

  // compare data, note: reverse row order
  EXPECT_THAT(model->hfield_data[0], FloatEq((5 - offset) / scale));
  EXPECT_THAT(model->hfield_data[1], FloatEq((6 - offset) / scale));
  EXPECT_THAT(model->hfield_data[2], FloatEq((3 - offset) / scale));
  EXPECT_THAT(model->hfield_data[3], FloatEq((4 - offset) / scale));
  EXPECT_THAT(model->hfield_data[4], FloatEq((1 - offset) / scale));
  EXPECT_THAT(model->hfield_data[5], FloatEq((2 - offset) / scale));
}

// ------------- test relative frame sensor parsing ----------------------------

using RelativeFrameSensorParsingTest = MujocoTest;

TEST_F(RelativeFrameSensorParsingTest, RefNameButNoType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="reference"/>
      <body name="sensorized"/>
    </worldbody>
    <sensor>
      <framepos objname="sensorized" objtype="body" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(
      error.data(),
      HasSubstr("attributes 'reftype', 'refname' must be specified together"));
  EXPECT_THAT(error.data(), HasSubstr("line 8"));
}

TEST_F(RelativeFrameSensorParsingTest, RefTypeButNoName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="reference"/>
      <body name="sensorized"/>
    </worldbody>
    <sensor>
      <framepos objname="sensorized" objtype="body" reftype="site"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  LoadModelFromString(xml, error.data(), error.size());
  EXPECT_THAT(
      error.data(),
      HasSubstr("attributes 'reftype', 'refname' must be specified together"));
  EXPECT_THAT(error.data(), HasSubstr("line 8"));
}

// ------------- test actlimited parsing ---------------------------------------

using ActuatorTest = MujocoTest;

TEST_F(ActuatorTest, InvalidActlimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge" actlimited="invalid" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("unrecognized attribute"));
  EXPECT_THAT(error.data(), HasSubstr("line 10"));
}

TEST_F(ActuatorTest, IncompleteActlimited) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" actlimited="true" dyntype="filter" actrange="-1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("attribute 'actrange' does not have enough data"));
}

TEST_F(ActuatorTest, ReadsByte) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <general joint="hinge" dyntype="filter" actlimited="true" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(*(model->actuator_actlimited), true);
}

// ---------------- test actuator parsing --------------------------------------

using ActuatorParseTest = MujocoTest;

TEST_F(ActuatorParseTest, PositionTimeconst) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" timeconst="2"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  ASSERT_NEAR(model->actuator_dynprm[0], 2.0, 1e-6);
  EXPECT_THAT(model->actuator_dyntype[0], Eq(mjDYN_FILTEREXACT));
}

TEST_F(ActuatorParseTest, PositionTimeconstInheritrange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" range="-1 1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" inheritrange="1" timeconst="2"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
}

// a general actuator which is a position servo inherits its range like the
// shortcuts which create one
TEST_F(ActuatorParseTest, GeneralInheritrange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="inherits">
        <general inheritrange="0.5"/>
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
      <general joint="slide" inheritrange="2" biastype="affine" biasprm="0 -1"/>
      <general joint="slide" class="inherits" biastype="affine" biasprm="0 -1"/>
      <general joint="slide" inheritrange="2" dyntype="integrator" biastype="affine" biasprm="0 -1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_THAT(AsVector(model->actuator_ctrlrange, 8),
              ElementsAre(-1, 3, -1, 3, 0.5, 1.5, 0, 0));
  EXPECT_THAT(AsVector(model->actuator_actrange + 6, 2), ElementsAre(-1, 3));
  EXPECT_THAT(AsVector(model->actuator_ctrllimited, 4),
              ElementsAre(1, 1, 1, 0));
  EXPECT_EQ(model->actuator_actlimited[3], 1);
}

TEST_F(ActuatorParseTest, PositionTimeconstDefault) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <position timeconst="1"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  ASSERT_NEAR(model->actuator_dynprm[0], 1.0, 1e-6);
  EXPECT_THAT(model->actuator_dyntype[0], Eq(mjDYN_FILTEREXACT));
}

TEST_F(ActuatorParseTest, PositionTimeconstDefaultOverride) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <position timeconst="1"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" timeconst="0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_FALSE(model->actuator_dynprm[0]);
  EXPECT_THAT(model->actuator_dyntype[0], Eq(mjDYN_NONE));
}

TEST_F(ActuatorParseTest, ReadsDamper) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="-10 10"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="jnt" ctrlrange="0 1"/>
      <general joint="jnt" ctrllimited="true" ctrlrange="0 1" gaintype="affine" biastype="none"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_THAT(model->actuator_gaintype[0], Eq(mjGAIN_AFFINE));
  EXPECT_THAT(model->actuator_gaintype[1], Eq(mjGAIN_AFFINE));
  EXPECT_THAT(model->actuator_biastype[0], Eq(mjBIAS_NONE));
  EXPECT_THAT(model->actuator_biastype[1], Eq(mjBIAS_NONE));
}

TEST_F(ActuatorParseTest, DamperRequiresPositiveDamping) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="-10 10"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="jnt" kv="-1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("damping coefficient cannot be negative"));
}

TEST_F(ActuatorParseTest, DamperRequiresControlRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="-10 10" limited="true"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="jnt" kv="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("invalid control range"));
  EXPECT_THAT(error.data(), HasSubstr("line 10"));
}

TEST_F(ActuatorParseTest, DamperPositiveControlRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="-10 10"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="jnt" kv="1" ctrlrange="-1 0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("control range cannot be negative"));
  EXPECT_THAT(error.data(), HasSubstr("line 10"));
}

TEST_F(ActuatorParseTest, ReadsPositionIntvelKv) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" kv="2"/>
      <intvelocity joint="jnt" actrange="-1 1" kv="3"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_THAT(model->actuator_biasprm[0 * mjNBIAS + 2], Eq(-2.0));
  EXPECT_THAT(model->actuator_biasprm[1 * mjNBIAS + 2], Eq(-3.0));
}

TEST_F(ActuatorParseTest, RequirePositiveKv) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="jnt" kv="-2"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("kv cannot be negative"));
  EXPECT_THAT(error.data(), HasSubstr("line 10"));
}

TEST_F(ActuatorParseTest, IntvelocityRejectsInvalidDamping) {
  static constexpr char xml_kv_dampratio[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="jnt" kv="1" dampratio="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model =
      LoadModelFromString(xml_kv_dampratio, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("kv and dampratio cannot both be defined"));

  static constexpr char xml_negative_kv[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="jnt" kv="-1"/>
    </actuator>
  </mujoco>
  )";
  model = LoadModelFromString(xml_negative_kv, error.data(), error.size());
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("kv cannot be negative"));
}

TEST_F(ActuatorParseTest, PositionIntvelocityVelocityDefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="position">
        <position kp="3" kv="4" inheritrange="2"/>
      </default>
      <default class="intvelocity">
        <intvelocity kp="5" kv="6" inheritrange="0.5"/>
      </default>
      <default class="velocity">
        <velocity kv="7"/>
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
      <intvelocity joint="jnt" class="intvelocity"/>
      <velocity joint="jnt" class="velocity"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_gainprm[0 * mjNGAIN + 0], 3.0);
  EXPECT_EQ(model->actuator_biasprm[0 * mjNBIAS + 1], -3.0);
  EXPECT_EQ(model->actuator_biasprm[0 * mjNBIAS + 2], -4.0);
  EXPECT_EQ(model->actuator_gainprm[1 * mjNGAIN + 0], 5.0);
  EXPECT_EQ(model->actuator_biasprm[1 * mjNBIAS + 1], -5.0);
  EXPECT_EQ(model->actuator_biasprm[1 * mjNBIAS + 2], -6.0);
  EXPECT_EQ(model->actuator_gainprm[2 * mjNGAIN + 0], 7.0);
  EXPECT_EQ(model->actuator_biasprm[2 * mjNBIAS + 1], 0.0);
  EXPECT_EQ(model->actuator_biasprm[2 * mjNBIAS + 2], -7.0);
  for (int i = 0; i < model->nu; i++) {
    for (int j = 3; j < mjNBIAS; j++) {
      EXPECT_EQ(model->actuator_biasprm[i * mjNBIAS + j], 0.0);
    }
    for (int j = 3; j < mjNGAIN; j++) {
      EXPECT_EQ(model->actuator_gainprm[i * mjNGAIN + j], 0.0);
    }
  }
  EXPECT_EQ(model->actuator_ctrlrange[0 * 2 + 0], -1.0);
  EXPECT_EQ(model->actuator_ctrlrange[0 * 2 + 1], 3.0);
  EXPECT_EQ(model->actuator_actrange[1 * 2 + 0], 0.5);
  EXPECT_EQ(model->actuator_actrange[1 * 2 + 1], 1.5);
}

TEST_F(ActuatorParseTest, IntvelocityCheckEquivalence) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="hinge" kp="2.5" actrange="-1.57 1.57"/>
      <general joint="hinge" actlimited="true" actrange="-1.57 1.57"
      dyntype="integrator" biastype="affine" gainprm="2.5" biasprm="0 -2.5 0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  // same actlimited
  EXPECT_EQ(model->actuator_actlimited[0], 1);
  EXPECT_EQ(model->actuator_actlimited[1], 1);
  // same dyntype
  EXPECT_EQ(model->actuator_dyntype[0], mjDYN_INTEGRATOR);
  EXPECT_EQ(model->actuator_dyntype[1], mjDYN_INTEGRATOR);
  // same biastype
  EXPECT_EQ(model->actuator_biastype[0], mjBIAS_AFFINE);
  EXPECT_EQ(model->actuator_biastype[1], mjBIAS_AFFINE);
  // same gaintype
  EXPECT_EQ(model->actuator_gaintype[0], mjGAIN_FIXED);
  EXPECT_EQ(model->actuator_gaintype[1], mjGAIN_FIXED);
  // same gainprm
  EXPECT_DOUBLE_EQ(model->actuator_gainprm[0], 2.5);
  EXPECT_DOUBLE_EQ(model->actuator_gainprm[mjNGAIN], 2.5);
  // same biasprm
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[0], 0.0);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[1], -2.5);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[2], 0.0);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[mjNBIAS], 0.0);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[mjNBIAS + 1], -2.5);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[mjNBIAS + 2], 0.0);
  // same actrange
  EXPECT_MJTNUM_EQ(model->actuator_actrange[0 + 0], -1.57);
  EXPECT_MJTNUM_EQ(model->actuator_actrange[0 + 1], 1.57);
  EXPECT_MJTNUM_EQ(model->actuator_actrange[0 + 2], -1.57);
  EXPECT_MJTNUM_EQ(model->actuator_actrange[0 + 3], 1.57);
}

TEST_F(ActuatorParseTest, IntvelocityCheckDefaultsIfNotSpecified) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="hinge" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  // check that by default kp = 1
  EXPECT_DOUBLE_EQ(model->actuator_gainprm[0], 1.0);
  // check that biasprm is (0, -1, 0)
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[0], 0.0);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[1], -1.0);
  EXPECT_DOUBLE_EQ(model->actuator_biasprm[2], 0.0);
}

TEST_F(ActuatorParseTest, IntvelocityNoActrangeIsValid) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  // actlimited resolves to false when no actrange is provided
  EXPECT_EQ(model->actuator_actlimited[0], 0);
}

TEST_F(ActuatorParseTest, IntvelocityInheritrangeAllowsCtrlrange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0" range="-2 2"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="jnt" ctrlrange="-1 1" inheritrange="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_biastype[0], mjBIAS_AFFINE);
  EXPECT_EQ(model->actuator_ctrlrange[0], -1.0);
  EXPECT_EQ(model->actuator_ctrlrange[1], 1.0);
  EXPECT_EQ(model->actuator_actrange[0], -2.0);
  EXPECT_EQ(model->actuator_actrange[1], 2.0);
}

TEST_F(ActuatorParseTest, IntvelocityDefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <intvelocity kp="5"/>
      <default class="withactrange">
        <intvelocity kp="1" actrange="-1 1"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="hinge1"/>
        <joint name="hinge2" axis="1 0 0"/>
        <geom type="box" size=".025 .025 .025"/>
      </body>
    </worldbody>
    <actuator>
      <intvelocity joint="hinge1" actrange="0 1"/>
      <intvelocity joint="hinge2" class="withactrange"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_DOUBLE_EQ(model->actuator_gainprm[0], 5);
  EXPECT_DOUBLE_EQ(model->actuator_gainprm[mjNGAIN], 1);
  EXPECT_DOUBLE_EQ(model->actuator_actrange[0 + 0], 0);
  EXPECT_DOUBLE_EQ(model->actuator_actrange[0 + 1], 1);
  EXPECT_DOUBLE_EQ(model->actuator_actrange[0 + 2], -1);
  EXPECT_DOUBLE_EQ(model->actuator_actrange[0 + 3], 1);
}

TEST_F(ActuatorParseTest, AdhesionDefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <adhesion ctrlrange="0 3"/>
    </default>
    <worldbody>
      <body name="sphere">
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion body="sphere"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(model->actuator_ctrlrange[0], 0);
  EXPECT_EQ(model->actuator_ctrlrange[1], 3);
}

TEST_F(ActuatorParseTest, ErrorBadAdhesionDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <adhesion ctrlrange="-1 0"/>
    </default>
    <worldbody>
      <body name="sphere">
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion body="sphere"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("adhesion control range cannot be negative"));
}

// make sure range requirement is not enforced at parse time
TEST_F(ActuatorParseTest, DampersDontRequireRange) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <damper ctrlrange="0 2"/>
    </default>
    <worldbody>
      <body name="sphere">
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(model->actuator_ctrlrange[0], 0);
  EXPECT_EQ(model->actuator_ctrlrange[1], 2);
}

TEST_F(ActuatorParseTest, DamperInheritsKv) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <damper kv="5" ctrlrange="0 2"/>
    </default>
    <worldbody>
      <body name="sphere">
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <damper joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_gainprm[2], -5.0);
}

// a general default is the base of every shortcut with its gain model; a
// shortcut sets its own dynamics and inherits mechanical attributes
TEST_F(ActuatorParseTest, ShortcutsInheritFromGeneral) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <general dyntype="filter" dynprm="123" gainprm="5" ctrlrange="0 2"/>
      <default class="lift">
        <general biastype="affine" gainprm="400" biasprm="0 -200 -100" forcerange="-70 70"/>
      </default>
      <default class="servo">
        <position kp="100" kv="10"/>
        <default class="limited">
          <general forcerange="-50 50"/>
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
      <body name="sphere">
        <joint name="hinge"/>
        <geom name="sphere" size="1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion name="adhere" body="sphere"/>
      <position name="lift" class="lift" joint="hinge"/>
      <position name="limited" class="limited" joint="hinge"/>
      <damper name="damper" joint="hinge"/>
      <position name="arm" class="arm" joint="hinge"/>
      <velocity name="armvel" class="arm" joint="hinge"/>
      <pid name="pid" class="pidlimited" joint="hinge"/>
    </actuator>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  auto actuator = [spec](const char* name) {
    return mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, name));
  };

  // adhesion: gain from the fixed-gain general default, its own dynamics
  EXPECT_EQ(actuator("adhere")->gainprm[0], 5);
  EXPECT_EQ(actuator("adhere")->dynprm[0], 1);
  EXPECT_EQ(actuator("adhere")->dyntype, mjDYN_NONE);
  EXPECT_EQ(actuator("adhere")->ctrlrange[1], 2);

  // position: kp and kv from a general default written as a position servo
  EXPECT_EQ(actuator("lift")->gainprm[0], 400);
  EXPECT_EQ(actuator("lift")->biasprm[1], -400);
  EXPECT_EQ(actuator("lift")->biasprm[2], -100);
  EXPECT_EQ(actuator("lift")->forcerange[1], 70);

  // a general child class which sets only mechanical attributes keeps the
  // parent's position parameters
  EXPECT_EQ(actuator("limited")->gainprm[0], 100);
  EXPECT_EQ(actuator("limited")->biasprm[2], -10);
  EXPECT_EQ(actuator("limited")->forcerange[1], 50);

  // damper: an affine gain is not found in a fixed-gain general default
  EXPECT_EQ(actuator("damper")->gainprm[2], 0);
  EXPECT_EQ(actuator("damper")->ctrlrange[1], 2);

  // a class written with another shortcut is skipped: the position parameters
  // come from the nearest ancestor written with position (or general), the
  // velocity parameters from the class itself
  EXPECT_EQ(actuator("arm")->gainprm[0], 100);
  EXPECT_EQ(actuator("arm")->biasprm[2], -10);
  EXPECT_EQ(actuator("armvel")->gainprm[0], 20);

  // pid: a general child class with the pid gain type is inherited from
  EXPECT_EQ(actuator("pid")->biasprm[1], -7);
  EXPECT_EQ(actuator("pid")->gainprm[0], 3);
  EXPECT_EQ(actuator("pid")->forcerange[1], 5);

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// the activation layout of a dcmotor default and the inheritrange of a position
// default do not carry over to other shortcuts
TEST_F(ActuatorParseTest, ShortcutsDoNotInheritLayoutOfOtherShortcuts) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="dc">
        <dcmotor motorconst="0.5" resistance="2" inductance="0.1 0" controller="0 1 0 0 0 0"/>
      </default>
      <default class="range">
        <position inheritrange="1"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="hinge" type="slide" range="0 1"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor name="motor" class="dc" joint="hinge"/>
      <position name="position" class="range" joint="hinge"/>
      <intvelocity name="intvelocity" class="range" joint="hinge" actrange="-1 1"/>
    </actuator>
  </mujoco>
  )";

  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  auto actuator = [spec](const char* name) {
    return mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, name));
  };
  EXPECT_EQ(actuator("motor")->actdim, -1);
  EXPECT_EQ(actuator("motor")->actearly, 0);
  EXPECT_EQ(actuator("motor")->actlimited, mjLIMITED_AUTO);
  EXPECT_EQ(actuator("position")->inheritrange, 1);
  EXPECT_EQ(actuator("intvelocity")->inheritrange, 0);

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(model->actuator_actnum[0], 0);
  EXPECT_EQ(model->actuator_ctrlrange[3], 1);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// github issue #3561: a motor in a class with a position default is a plain
// motor
TEST_F(ActuatorParseTest, ShortcutsDoNotInheritFromOtherShortcuts) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <position kp="100" kv="10" timeconst="0.1" ctrlrange="-1 1" gear="2"/>
    </default>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="hinge"/>
      <cylinder joint="hinge"/>
      <pid joint="hinge"/>
      <position joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // motor: documented row, mechanical attributes inherited
  EXPECT_EQ(model->actuator_gainprm[0], 1);
  EXPECT_EQ(model->actuator_biasprm[1], 0);
  EXPECT_EQ(model->actuator_biasprm[2], 0);
  EXPECT_EQ(model->actuator_biastype[0], mjBIAS_NONE);
  EXPECT_EQ(model->actuator_dyntype[0], mjDYN_NONE);
  EXPECT_EQ(model->actuator_ctrlrange[0], -1);
  EXPECT_EQ(model->actuator_ctrlrange[1], 1);
  EXPECT_EQ(model->actuator_gear[0], 2);

  // cylinder: documented defaults, no position bias
  EXPECT_EQ(model->actuator_gainprm[mjNGAIN], 1);
  EXPECT_EQ(model->actuator_biasprm[mjNBIAS], 0);
  EXPECT_EQ(model->actuator_biasprm[mjNBIAS + 1], 0);
  EXPECT_EQ(model->actuator_biasprm[mjNBIAS + 2], 0);
  EXPECT_EQ(model->actuator_dynprm[mjNDYN], 1);

  // pid: documented defaults
  EXPECT_EQ(model->actuator_biasprm[2 * mjNBIAS + 1], -1);
  EXPECT_EQ(model->actuator_biasprm[2 * mjNBIAS + 2], 0);
  EXPECT_EQ(model->actuator_dyntype[2], mjDYN_NONE);

  // position: inherits kp, kv and timeconst from the position default
  EXPECT_EQ(model->actuator_gainprm[3 * mjNGAIN], 100);
  EXPECT_EQ(model->actuator_biasprm[3 * mjNBIAS + 1], -100);
  EXPECT_EQ(model->actuator_biasprm[3 * mjNBIAS + 2], -10);
  EXPECT_EQ(model->actuator_dyntype[3], mjDYN_FILTEREXACT);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[3 * mjNDYN], 0.1);
}

TEST_F(ActuatorParseTest, ShortcutParametersInheritedFromSameShortcut) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <position kp="100" dampratio="2"/>
      <default class="kv">
        <position kv="10"/>
      </default>
      <default class="muscle">
        <muscle scale="300" lmin="0.3"/>
      </default>
      <default class="pid">
        <pid kp="7" ki="3" imax="5" slewmax="2" input="pos ff"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="hinge" axis="1 0 0"/>
        <geom size="1"/>
        <site name="a" pos="0 1 0"/>
      </body>
      <site name="b" pos="0 0 1"/>
    </worldbody>
    <tendon>
      <spatial name="tendon">
        <site site="a"/>
        <site site="b"/>
      </spatial>
    </tendon>
    <actuator>
      <position name="dampratio" joint="hinge"/>
      <position name="kv" joint="hinge" kv="1"/>
      <position name="child" class="kv" joint="hinge"/>
      <muscle name="muscle" class="muscle" tendon="tendon"/>
      <pid name="pid" class="pid" joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, 0, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  auto actuator = [spec](const char* name) {
    return mjs_asActuator(mjs_findElement(spec, mjOBJ_ACTUATOR, name));
  };

  // dampratio is inherited, and replaced by an explicit kv
  EXPECT_EQ(actuator("dampratio")->type, mjACTUATOR_POSITION);
  EXPECT_EQ(actuator("dampratio")->biasprm[2], 2);
  EXPECT_EQ(actuator("kv")->biasprm[2], -1);

  // a child class keeps the parent's kp, its kv replaces the dampratio
  EXPECT_EQ(actuator("child")->gainprm[0], 100);
  EXPECT_EQ(actuator("child")->biasprm[2], -10);

  // muscle: only the given parameters differ from the muscle defaults
  EXPECT_EQ(actuator("muscle")->type, mjACTUATOR_MUSCLE);
  EXPECT_EQ(actuator("muscle")->gainprm[0], 0.75);
  EXPECT_EQ(actuator("muscle")->gainprm[3], 300);
  EXPECT_EQ(actuator("muscle")->gainprm[4], 0.3);
  EXPECT_EQ(actuator("muscle")->gainprm[5], 1.6);

  // pid: all parameters inherited
  EXPECT_EQ(actuator("pid")->biasprm[1], -7);
  EXPECT_EQ(actuator("pid")->gainprm[0], 3);
  EXPECT_EQ(actuator("pid")->dynprm[0], 5);
  EXPECT_EQ(actuator("pid")->dynprm[1], 2);
  EXPECT_EQ(actuator("pid")->ctrlspec, mjINPUT_POS | mjINPUT_FF);

  mjModel* model = mj_compile(spec, nullptr);
  ASSERT_THAT(model, NotNull()) << mjs_getError(spec);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

TEST_F(ActuatorParseTest, DCMotorBasicParsing) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0" damping="1 2 3" armature="0.1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_dyntype[0], mjDYN_DCMOTOR);
  EXPECT_EQ(model->actuator_gaintype[0], mjGAIN_DCMOTOR);
  EXPECT_EQ(model->actuator_biastype[0], mjBIAS_DCMOTOR);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[0], 2.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[1], 0.05);
  EXPECT_MJTNUM_EQ(model->actuator_damping[0], 1.0);
  EXPECT_MJTNUM_EQ(model->actuator_dampingpoly[0], 2.0);
  EXPECT_MJTNUM_EQ(model->actuator_dampingpoly[1], 3.0);
  EXPECT_MJTNUM_EQ(model->actuator_armature[0], 0.1);
}

TEST_F(ActuatorParseTest, DCMotorNominalDerivation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <!-- no damping: Ke = vn/omega0 -->
      <dcmotor joint="jnt" nominal="12 0.6 600"/>
      <!-- B > 0, R given: quadratic -->
      <dcmotor joint="jnt" nominal="12 0 600" resistance="0.4" damping="0.0001"/>
      <!-- B > 0, R from nominal: linear -->
      <dcmotor joint="jnt" nominal="12 0.6 600" damping="0.0001"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // actuator 0: B = 0, Ke = vn/omega0
  {
    double K = 12.0 / 600.0;
    double R = K * 12.0 / 0.6;
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[0 * mjNGAIN + 0], R);
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[0 * mjNGAIN + 1], K);
  }

  // actuator 1: B > 0, R given, quadratic Ke^2*omega0 - Ke*vn + R*B*omega0 = 0
  {
    double B = 0.0001, R = 0.4, vn = 12.0, omega0 = 600.0;
    double disc = vn * vn - 4 * R * B * omega0 * omega0;
    double Ke = (vn + sqrt(disc)) / (2 * omega0);
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[1 * mjNGAIN + 0], R);
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[1 * mjNGAIN + 1], Ke);
  }

  // actuator 2: B > 0, R from nominal, Ke = vn/omega0 - vn*B/tau0
  {
    double B = 0.0001, vn = 12.0, tau0 = 0.6, omega0 = 600.0;
    double Ke = vn / omega0 - vn * B / tau0;
    double R = Ke * vn / tau0;
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[2 * mjNGAIN + 0], R);
    EXPECT_MJTNUM_EQ(model->actuator_gainprm[2 * mjNGAIN + 1], Ke);
  }
}

TEST_F(ActuatorParseTest, DCMotorSaturation) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"
               saturation="1.5 0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_forcelimited[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_forcerange[0], -1.5);
  EXPECT_MJTNUM_EQ(model->actuator_forcerange[1], 1.5);
}

TEST_F(ActuatorParseTest, DCMotorInheritedDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <dcmotor motorconst="1.0" resistance="1.0" controller="2.0 0.5 0.1 10.0 5.0 12.0"
      saturation="0 0 0" inductance="0 0.01" input="pos vel"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // check motorconst and resistance are overridden by instance
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[1], 0.05);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[0], 2.0);

  // check controller gains (kp, ki, kd) in gainprm[4:6]
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[4], 2.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[5], 0.5);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[6], 0.1);

  // check controller limits (slewmax, Imax) in dynprm[7,8]
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[7], 10.0);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[8], 5.0);

  // check Vmax in gainprm[7]
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[7], 12.0);

  // check setpoint input block (gainprm[8] is retired)
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[8], 0.0);
  EXPECT_EQ(model->actuator_ctrlnum[0], 2);

  // check inductance (te) in dynprm[0]
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[0], 0.01);
}

TEST_F(ActuatorParseTest, DCMotorControllerFull) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0" input="pos vel"
               controller="1.0 2.0 3.0 4.0 5.0 6.0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  EXPECT_MJTNUM_EQ(model->actuator_gainprm[4], 1.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[5], 2.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[6], 3.0);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[7], 4.0);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[8], 5.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[7], 6.0);
}

TEST_F(ActuatorParseTest, DCMotorLuGreRemapping) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0" damping="0.01"
               lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[5], 100);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[6], 1);
  EXPECT_MJTNUM_EQ(model->actuator_damping[0], 0.01);
  EXPECT_MJTNUM_EQ(model->actuator_biasprm[3], 0.5);
  EXPECT_MJTNUM_EQ(model->actuator_biasprm[4], 0.7);
  EXPECT_MJTNUM_EQ(model->actuator_biasprm[5], 10);
}

TEST_F(ActuatorParseTest, DCMotorLuGreInheritedDefaults) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <dcmotor motorconst="0.05" resistance="2.0" damping="0.01" lugre="100 1 0.5 0.7 10"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_MJTNUM_EQ(model->actuator_damping[0], 0.01);
}

TEST_F(ActuatorParseTest, DCMotorActdimStateless) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 0);
  EXPECT_EQ(model->actuator_actadr[0], -1);
}

TEST_F(ActuatorParseTest, DCMotorActdimCurrentOnly) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"
               inductance="0.001 0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[0], 0.001 / 2.0);
}

TEST_F(ActuatorParseTest, DCMotorActdimThermalOnly) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"
               thermal="10 5 0 0 0 25"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[2], 10);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[3], 5);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[4], 25);
}

TEST_F(ActuatorParseTest, DCMotorActdimLuGreOnly) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"
               lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[5], 100);
}

TEST_F(ActuatorParseTest, DCMotorActdimAllThree) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2.0"
               inductance="0.001 0"
               thermal="10 5 0 0 0 25"
               lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 3);
}

TEST_F(ActuatorParseTest, DCMotorMissingKError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("motor constant K must be positive"));
}

TEST_F(ActuatorParseTest, DCMotorDefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <dcmotor motorconst="0.03" resistance="1.5"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[0], 1.5);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[1], 0.03);
}

TEST_F(ActuatorParseTest, DCMotorMotorconstGeometricMean) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.03 0.05" resistance="2.0"/>
      <dcmotor joint="jnt" motorconst="0.03" resistance="2.0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  double K = std::sqrt(0.03 * 0.05);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[0], 2.0);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[1], K);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[mjNGAIN + 1], 0.03);
}

TEST_F(ActuatorParseTest, DCMotorPassiveZeroMotorconst) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" input="none" motorconst="0" resistance="1"
               lugre="100 1 0.5 0.7 10"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_gainprm[1], 0);
  EXPECT_EQ(model->actuator_actnum[0], 1);
  MjDataPtr data = MakeData(model);
  mj_step(model.get(), data.get());
}

TEST_F(ActuatorParseTest, DCMotorNegativeMotorconstError) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" input="none" motorconst="-0.05" resistance="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("motor constants must be non-negative"));
}

TEST_F(ActuatorParseTest, DCMotorThermalTimeconstPrecedence) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" motorconst="0.05" resistance="2" thermal="2 5 4"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_actnum[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[2], 2);  // resistance
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[3], 2);  // timeconst / resistance
}

TEST_F(ActuatorParseTest, DCMotorZeroOverridesClass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <dcmotor motorconst="0.05" resistance="2" input="pos"
               saturation="1.5 0 100" inductance="0.002"
               controller="1 0 0 0 0 12" thermal="2 5"
               lugre="100 1 0.5 0.7 10"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt"/>
      <dcmotor joint="jnt" saturation="0 0 0" inductance="0"
               controller="1 0 0 0 0 0" thermal="0 0 0" lugre="0"/>
      <dcmotor joint="jnt" saturation="0" inductance="0 0.001"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // inherited: torque limit, current, Vmax, thermal and LuGre
  EXPECT_EQ(model->actuator_forcelimited[0], 1);
  EXPECT_MJTNUM_EQ(model->actuator_forcerange[1], 1.5);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[0], 0.001);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[1], 100);
  EXPECT_MJTNUM_EQ(model->actuator_gainprm[7], 12);
  EXPECT_EQ(model->actuator_actnum[0], 3);

  // explicit zeros disable the inherited features
  const mjtNum* dynprm = model->actuator_dynprm + mjNDYN;
  const mjtNum* gainprm = model->actuator_gainprm + mjNGAIN;
  const mjtNum* biasprm = model->actuator_biasprm + mjNBIAS;
  EXPECT_EQ(model->actuator_forcelimited[1], 0);
  EXPECT_EQ(model->actuator_forcerange[2], 0);
  EXPECT_EQ(model->actuator_forcerange[3], 0);
  for (int i = 0; i < 7; i++) EXPECT_EQ(dynprm[i], 0) << i;
  EXPECT_EQ(gainprm[2], 0);
  EXPECT_EQ(gainprm[3], 0);
  EXPECT_EQ(gainprm[7], 0);
  for (int i = 3; i < 6; i++) EXPECT_EQ(biasprm[i], 0) << i;
  EXPECT_EQ(model->actuator_actnum[1], 0);

  // a single zero clears the inherited torque limit; the timeconst is explicit
  EXPECT_EQ(model->actuator_forcelimited[2], 0);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[2 * mjNDYN + 0], 0.001);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[2 * mjNDYN + 1], 100);

  // activation slots agree with the parameters
  MjDataPtr data = MakeData(model);
  mj_step(model.get(), data.get());
}

TEST_F(ActuatorParseTest, DCMotorZeroInductanceOverridesClass) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <dcmotor motorconst="0.05" resistance="2" inductance="0.002"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <dcmotor joint="jnt" inductance="0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_dynprm[0], 0);
  EXPECT_EQ(model->actuator_actnum[0], 0);
}

TEST_F(ActuatorParseTest, ActdimDefaultsPropagate) {
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
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // expect that actdim was inherited from the general default
  EXPECT_EQ(model->actuator_actnum[0], 2);
}

TEST_F(ActuatorParseTest, MusclesParseSmoothdyn) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge" limited="true" range="-1 1"/>
      </body>
    </worldbody>
    <actuator>
      <muscle joint="hinge"/>
      <muscle joint="hinge" tausmooth="0.4"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_dynprm[2], 0.0);
  EXPECT_MJTNUM_EQ(model->actuator_dynprm[mjNDYN + 2], 0.4);
}

TEST_F(ActuatorParseTest, MusclesSmoothdynNegative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="hinge" limited="true" range="-1 1"/>
      </body>
    </worldbody>
    <actuator>
      <muscle joint="hinge" tausmooth="-0.4"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("muscle tausmooth cannot be negative"));
  EXPECT_THAT(error.data(), HasSubstr("line 10"));
}

TEST_F(ActuatorParseTest, GroupDisable) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option actuatorgroupdisable="0 3 8 3"/>
    <!-- note: repeated numbers are okay -->
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(model->opt.disableactuator, (1 << 0) + (1 << 3) + (1 << 8));
}

TEST_F(ActuatorParseTest, GroupDisableNegative) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option actuatorgroupdisable="0 -3 5"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("must be non-negative"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(ActuatorParseTest, GroupDisableTooBig) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option actuatorgroupdisable="0 31"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("cannot exceed 30"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

// ------------- test sensor parsing -------------------------------------------

using SensorParseTest = MujocoTest;

TEST_F(SensorParseTest, UserObjTypeNoName) {
  static constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <user dim="1" needstage="vel" objtype="site"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr("attributes 'objtype', 'objname' must be specified together"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(SensorParseTest, UserObjNameNoType) {
  static constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <user dim="1" needstage="vel" objname="kevin"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr("attributes 'objtype', 'objname' must be specified together"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(SensorParseTest, UserNeedstageAcc) {
  static constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <user dim="1" needstage="vel"/>
      <user dim="1"/>
      <user dim="1" needstage="pos"/>
    </sensor>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->sensor_needstage[0], mjSTAGE_VEL);
  EXPECT_EQ(model->sensor_needstage[1], mjSTAGE_ACC);
  EXPECT_EQ(model->sensor_needstage[2], mjSTAGE_POS);
}

// ------------- test general parsing ------------------------------------------

TEST_F(XMLReaderTest, ZnearZeroNotAllowed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <visual>
      <map znear="0"/>
    </visual>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("znear must be strictly positive"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(XMLReaderTest, ZnearNegativeNotAllowed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <visual>
      <map znear="-1"/>
    </visual>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("znear must be strictly positive"));
  EXPECT_THAT(error.data(), HasSubstr("line 4"));
}

TEST_F(XMLReaderTest, ExtentZeroNotAllowed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <statistic extent="0"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("extent must be strictly positive"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, ExtentNegativeNotAllowed) {
  static constexpr char xml[] = R"(
  <mujoco>
    <statistic extent="-1"/>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("extent must be strictly positive"));
  EXPECT_THAT(error.data(), HasSubstr("line 3"));
}

TEST_F(XMLReaderTest, LightRadius) {
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
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_FLOAT_EQ(model->light_bulbradius[0], 0.02);
  EXPECT_FLOAT_EQ(model->light_bulbradius[1], 1);
  EXPECT_FLOAT_EQ(model->light_bulbradius[2], 2);
}

TEST_F(XMLReaderTest, CameraOutput) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="default_cam"/>
      <camera name="single_cam" output="depth"/>
      <camera name="multi_cam" output="rgb normal segmentation"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->ncam, 3);
  EXPECT_EQ(model->cam_output[0], mjCAMOUT_RGB);  // default
  EXPECT_EQ(model->cam_output[1], mjCAMOUT_DEPTH);
  EXPECT_EQ(model->cam_output[2],
            mjCAMOUT_RGB | mjCAMOUT_NORMAL | mjCAMOUT_SEG);
}

TEST_F(XMLReaderTest, CameraOutputDefault) {
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
      <camera name="override_cam" class="multi" output="normal"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->ncam, 3);
  EXPECT_EQ(model->cam_output[0], mjCAMOUT_RGB);  // main default
  EXPECT_EQ(model->cam_output[1], mjCAMOUT_DEPTH | mjCAMOUT_DIST);
  EXPECT_EQ(model->cam_output[2], mjCAMOUT_NORMAL);
}

// ------------- test delay attribute parsing ----------------------------------

TEST_F(ActuatorParseTest, ActuatorDelayParsed) {
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
      <motor joint="jnt2" delay="3" nsample="3" interp="zoh"/>
      <motor joint="jnt3" delay="10" nsample="10" interp="cubic"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  ASSERT_EQ(model->nu, 3);
  // actuator_history[2*i] = nsample, actuator_history[2*i+1] = interp
  EXPECT_EQ(model->actuator_history[0], 0);   // jnt1 nsample
  EXPECT_EQ(model->actuator_history[1], 0);   // jnt1 interp (no buffer, def 0)
  EXPECT_EQ(model->actuator_history[2], 3);   // jnt2 nsample
  EXPECT_EQ(model->actuator_history[3], 0);   // jnt2 interp (ZOH)
  EXPECT_EQ(model->actuator_history[4], 10);  // jnt3 nsample
  EXPECT_EQ(model->actuator_history[5], 2);   // jnt3 interp (cubic)
  EXPECT_EQ(model->actuator_historyadr[0], -1);
  EXPECT_EQ(model->actuator_historyadr[1], 0);
  EXPECT_EQ(model->actuator_historyadr[2], 8);  // 2+2*3 = 8
}

TEST_F(ActuatorParseTest, ActuatorDelayDefault) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1"/>
    <default>
      <motor delay="5" nsample="5"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt1"/>
        <joint name="jnt2" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt1"/>
      <motor joint="jnt2" delay="0" nsample="0"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  ASSERT_EQ(model->nu, 2);
  EXPECT_EQ(model->actuator_history[0], 5);
  EXPECT_EQ(model->actuator_history[2], 0);
  EXPECT_EQ(model->actuator_historyadr[0], 0);
  EXPECT_EQ(model->actuator_historyadr[1], -1);
}

TEST_F(ActuatorParseTest, ActuatorDelayRequiresHistory) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt" delay="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("setting delay > 0 without a history buffer"));
}

TEST_F(ActuatorParseTest, DampingArmatureDefaultsPropagate) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <motor damping="3" armature="0.5"/>
    </default>
    <worldbody>
      <body>
        <geom size="1"/>
        <joint name="jnt" type="slide" axis="1 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <motor joint="jnt"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_EQ(model->actuator_damping[0], 3);
  EXPECT_EQ(model->actuator_armature[0], 0.5);
}

TEST_F(ActuatorParseTest, DeclaredInputs) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <position joint="hinge"/>
      <velocity joint="hinge"/>
      <intvelocity joint="hinge"/>
      <cylinder joint="hinge"/>
      <motor joint="hinge"/>
      <damper joint="hinge" ctrlrange="0 1"/>
      <muscle joint="hinge" lengthrange="-1 1"/>
      <general joint="hinge" input="pos"/>
      <general joint="hinge" gaintype="affine" input="vel"/>
      <general joint="hinge" input="pressure"/>
      <general joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();
  EXPECT_THAT(
      AsVector(model->actuator_ctrlspec, model->nactuator),
      ElementsAre(mjINPUT_POS, mjINPUT_VEL, mjINPUT_VEL, mjINPUT_PRESSURE, 0, 0,
                  0, mjINPUT_POS, mjINPUT_VEL, mjINPUT_PRESSURE, 0));

  // declared inputs are named, commands are not
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 0, 0), "pos");
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 1, 0), "vel");
  EXPECT_STREQ(mj_actuatorInputName(model.get(), 3, 0), "pressure");
  EXPECT_EQ(mj_actuatorInputName(model.get(), 3, 1), nullptr);
  EXPECT_EQ(mj_actuatorInputName(model.get(), 4, 0), nullptr);
  EXPECT_EQ(mj_actuatorInputName(model.get(), 10, 0), nullptr);
}

TEST_F(ActuatorParseTest, DeclaredInputErrors) {
  std::array<char, 1024> error;
  auto load = [&error](const string& actuator) {
    string xml = R"(
    <mujoco>
      <worldbody>
        <body>
          <joint name="hinge"/>
          <geom size="1"/>
        </body>
      </worldbody>
      <actuator>)" +
                 actuator + R"(</actuator>
    </mujoco>
    )";
    return LoadModelFromString(xml, error.data(), error.size());
  };

  // fixed and affine gains take one input
  for (const char* input : {"pos vel", "ff", "voltage", "none"}) {
    MjModelPtr model =
        load(string(R"(<general joint="hinge" input=")") + input + R"("/>)");
    EXPECT_THAT(model.get(), IsNull()) << input;
    EXPECT_THAT(error.data(),
                HasSubstr("fixed and affine gains take one input"))
        << input;
  }

  // muscle gains take none
  MjModelPtr model =
      load(R"(<general joint="hinge" gaintype="muscle" input="pos"/>)");
  EXPECT_THAT(model.get(), IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("input is not available for muscle and user gains"));
}

TEST_F(ActuatorParseTest, InputInheritedFromSameGaintype) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="velocity">
        <velocity kv="1"/>
      </default>
      <default class="pid">
        <pid kp="1" input="pos"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <geom size="1"/>
      </body>
      <body>
        <joint name="hinge"/>
        <geom size="1"/>
      </body>
    </worldbody>
    <actuator>
      <orientation class="velocity" joint="ball" kp="1"/>
      <pid class="velocity" joint="hinge"/>
      <general class="velocity" joint="hinge" gaintype="pid" gainprm="0"/>
      <general class="velocity" joint="hinge"/>
      <motor class="velocity" joint="hinge"/>
      <pid class="pid" joint="hinge"/>
      <general class="pid" joint="hinge"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr model = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(model.get(), NotNull()) << error.data();

  // another gaintype's input is not inherited: expmap chart, default pid inputs
  EXPECT_EQ(model->actuator_ctrlspec[0], mjCHART_EXPMAP);
  EXPECT_EQ(model->actuator_ctrlspec[1], mjINPUT_POS | mjINPUT_VEL);
  EXPECT_EQ(model->actuator_ctrlspec[2], mjINPUT_POS | mjINPUT_VEL);

  // the same gaintype's input is inherited, unless a shortcut sets its own
  EXPECT_EQ(model->actuator_ctrlspec[3], mjINPUT_VEL);
  EXPECT_EQ(model->actuator_ctrlspec[4], 0);
  EXPECT_EQ(model->actuator_ctrlspec[5], mjINPUT_POS);
  EXPECT_EQ(model->actuator_ctrlspec[6], mjINPUT_POS);
}

TEST_F(XMLReaderTest, AttachConflictXMLWarning) {
  mock_warning_handler.ExpectWarnings();
  std::array<char, 1024> error;
  std::string path = GetTestDataFilePath("xml/testdata/parent_warn.xml");
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // one grouped warning per attach, containing gravity and damper flag
  // (constraint is only authored by child, not a conflict)
  EXPECT_EQ(mjs_numWarnings(spec), 1);
  std::string w = mjs_getWarning(spec, 0);
  EXPECT_THAT(w, HasSubstr("gravity: parent has 0 0 -10, child has 0 0 0,"
                           " keeping parent value"));
  EXPECT_THAT(w, HasSubstr("flag 'Damper'"));

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  // Gravity should be parent's value (0 0 -10)
  EXPECT_MJTNUM_EQ(m->opt.gravity[0], 0);
  EXPECT_MJTNUM_EQ(m->opt.gravity[1], 0);
  EXPECT_MJTNUM_EQ(m->opt.gravity[2], -10);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, AttachConflictXMLMerge) {
  mock_warning_handler.ExpectWarnings();
  std::array<char, 1024> error;
  std::string path = GetTestDataFilePath("xml/testdata/parent_merge.xml");
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  // one grouped warning per attach, containing timestep, iterations, and flag
  EXPECT_GE(mjs_numWarnings(spec), 1);
  std::string w = mjs_getWarning(spec, 0);
  EXPECT_THAT(w, HasSubstr("timestep: parent has 0.005, child has 0.002,"
                           " taking the minimum"));
  EXPECT_THAT(w, HasSubstr("iterations: parent has 50, child has 100,"
                           " taking the maximum"));

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());
  // Timestep should be min (0.002)
  EXPECT_MJTNUM_EQ(m->opt.timestep, 0.002);
  // Iterations should be max (100)
  EXPECT_EQ(m->opt.iterations, 100);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, AttachConflictXMLError) {
  std::array<char, 1024> error;
  std::string path = GetTestDataFilePath("xml/testdata/parent_error.xml");
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());

  // Should fail to parse because of conflict in timestep in error mode
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("timestep: parent has 0.005, child has 0.002"));
}

TEST_F(XMLReaderTest, AttachConflictXMLMergeUnmergableError) {
  std::array<char, 1024> error;
  std::string path =
      GetTestDataFilePath("xml/testdata/parent_merge_unmergable.xml");
  mjSpec* spec = mj_parseXML(path.c_str(), nullptr, error.data(), error.size());

  // Should fail to parse because gravity is unmergeable in merge mode
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("gravity: parent has 0 0 -10, child has 0 0 0"));
}

TEST_F(XMLReaderTest, SelfAttach) {
  static constexpr char xml[] = R"(
    <mujoco model="self-attach-test">
      <worldbody>
        <body name="body1">
          <geom name="geom1" size="1"/>
        </body>
        <body name="body2">
          <attach body="body1" prefix="attached_"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull());

  int body2_id = mj_name2id(m, mjOBJ_BODY, "body2");
  int attached_body1_id = mj_name2id(m, mjOBJ_BODY, "attached_body1");

  EXPECT_GE(body2_id, 0);
  EXPECT_GE(attached_body1_id, 0);
  EXPECT_EQ(m->body_parentid[attached_body1_id], body2_id);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, SelfAttachCollisionError) {
  static constexpr char xml[] = R"(
    <mujoco model="self-attach-collision">
      <worldbody>
        <body name="body1"/>
        <body name="attached_body1"/>
        <body name="body2">
          <attach body="body1" prefix="attached_"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr("cannot self-attach: element attached_body1 already exists"));
}

TEST_F(XMLReaderTest, SelfAttachMissingError) {
  static constexpr char xml[] = R"(
    <mujoco model="self-attach-missing">
      <worldbody>
        <body name="body1">
          <attach body="nonexistent" prefix="attached_"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(), HasSubstr("could not find body 'nonexistent' in "
                                      "the current model for self-attachment"));
}

TEST_F(XMLReaderTest, SelfAttachFrame) {
  static constexpr char xml[] = R"(
    <mujoco model="self-attach-frame-test">
      <worldbody>
        <frame name="frame1">
          <body name="body1" pos="-1 0 0">
            <geom name="geom1" type="box" size="1 1 1"/>
          </body>
        </frame>
        <body name="body2">
          <attach frame="frame1" prefix="attached_"/>
        </body>
      </worldbody>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  int body2_id = mj_name2id(m, mjOBJ_BODY, "body2");
  int attached_body1_id = mj_name2id(m, mjOBJ_BODY, "attached_body1");

  EXPECT_GE(body2_id, 0);
  EXPECT_GE(attached_body1_id, 0);
  EXPECT_EQ(m->body_parentid[attached_body1_id], body2_id);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

// a keyframe describes the model with its self-attached subtrees in place
TEST_F(XMLReaderTest, SelfAttachKeyframe) {
  static constexpr char xml[] = R"(
    <mujoco model="self-attach-keyframe">
      <worldbody>
        <body name="body1">
          <joint type="slide"/>
          <geom size="1"/>
        </body>
        <attach body="body1" prefix="attached_"/>
      </worldbody>
      <keyframe>
        <key name="key" qpos="1 2"/>
      </keyframe>
    </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nq, 2);
  ASSERT_EQ(m->nkey, 1);
  EXPECT_THAT(AsVector(m->key_qpos, 2), ElementsAre(1, 2));
}

TEST_F(XMLReaderTest, CustomTextFromAttribute) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-attr">
      <custom>
        <text name="my_text" data="attribute_value"/>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  int text_id = mj_name2id(m, mjOBJ_TEXT, "my_text");
  ASSERT_GE(text_id, 0);
  EXPECT_STREQ(m->text_data + m->text_adr[text_id], "attribute_value");

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, CustomTextFromCData) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-cdata">
      <custom>
        <text name="my_text"><![CDATA[cdata_value]]></text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  int text_id = mj_name2id(m, mjOBJ_TEXT, "my_text");
  ASSERT_GE(text_id, 0);
  EXPECT_STREQ(m->text_data + m->text_adr[text_id], "cdata_value");

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, CustomTextFromCDataTextProto) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-proto">
      <custom>
        <text name="my_proto">
<![CDATA[
message {
  field_str: "hello <world> & \"quotes\""
  field_int: 42
  field_float: 3.14
}
]]>
        </text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  int text_id = mj_name2id(m, mjOBJ_TEXT, "my_proto");
  ASSERT_GE(text_id, 0);
  const char* text_val = m->text_data + m->text_adr[text_id];
  EXPECT_THAT(text_val, HasSubstr("message {"));
  EXPECT_THAT(text_val,
              HasSubstr("field_str: \"hello <world> & \\\"quotes\\\"\""));
  EXPECT_THAT(text_val, HasSubstr("field_int: 42"));
  EXPECT_THAT(text_val, HasSubstr("field_float: 3.14"));

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, CustomTextRawTextIgnored) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-raw">
      <custom>
        <text name="my_text">raw_text_value</text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(), HasSubstr("text field cannot be empty"));
}

TEST_F(XMLReaderTest, CustomTextRawTextWithAttributeIgnored) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-raw-attr">
      <custom>
        <text name="my_text" data="attr_val">raw_text_value</text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);

  int text_id = mj_name2id(m, mjOBJ_TEXT, "my_text");
  ASSERT_GE(text_id, 0);
  EXPECT_STREQ(m->text_data + m->text_adr[text_id], "attr_val");

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, CustomTextBothAttributeAndCDataError) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-both-cdata">
      <custom>
        <text name="my_text" data="from_attr"><![CDATA[from_cdata]]></text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(
      error.data(),
      HasSubstr(
          "text field data cannot be specified as both attribute and CDATA"));
}

TEST_F(XMLReaderTest, CustomTextMultipleCDataError) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-mult-cdata">
      <custom>
        <text name="my_text"><![CDATA[first]]><![CDATA[second]]></text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  EXPECT_THAT(spec, IsNull());
  EXPECT_THAT(error.data(),
              HasSubstr("text field cannot have multiple CDATA sections"));
}

TEST_F(XMLReaderTest, CustomTextEmptyError) {
  static constexpr char xml1[] = R"(
    <mujoco model="custom-text-empty-1">
      <custom>
        <text name="my_text"/>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error1;
  mjSpec* spec1 =
      mj_parseXMLString(xml1, nullptr, error1.data(), error1.size());
  EXPECT_THAT(spec1, IsNull());
  EXPECT_THAT(error1.data(), HasSubstr("text field cannot be empty"));

  static constexpr char xml2[] = R"(
    <mujoco model="custom-text-empty-2">
      <custom>
        <text name="my_text"></text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error2;
  mjSpec* spec2 =
      mj_parseXMLString(xml2, nullptr, error2.data(), error2.size());
  EXPECT_THAT(spec2, IsNull());
  EXPECT_THAT(error2.data(), HasSubstr("text field cannot be empty"));

  static constexpr char xml3[] = R"(
    <mujoco model="custom-text-empty-3">
      <custom>
        <text name="my_text" data=""/>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error3;
  mjSpec* spec3 =
      mj_parseXMLString(xml3, nullptr, error3.data(), error3.size());
  EXPECT_THAT(spec3, IsNull());
  EXPECT_THAT(error3.data(), HasSubstr("text field cannot be empty"));

  static constexpr char xml4[] = R"(
    <mujoco model="custom-text-empty-4">
      <custom>
        <text name="my_text"><![CDATA[]]></text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error4;
  mjSpec* spec4 =
      mj_parseXMLString(xml4, nullptr, error4.data(), error4.size());
  EXPECT_THAT(spec4, IsNull());
  EXPECT_THAT(error4.data(), HasSubstr("text field cannot be empty"));
}

TEST_F(XMLReaderTest, CustomTextSaveXmlRoundTrip) {
  static constexpr char xml[] = R"(
    <mujoco model="custom-text-roundtrip">
      <custom>
        <text name="text_attr" data="simple_value"/>
        <text name="text_cdata">
<![CDATA[
message {
  field_str: "hello <world> & \"quotes\""
  field_int: 42
}
]]>
        </text>
      </custom>
    </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m_orig = mj_compile(spec, nullptr);
  ASSERT_THAT(m_orig, NotNull()) << mjs_getError(spec);

  std::array<char, 2048> saved_xml;
  int save_res = mj_saveXMLString(spec, saved_xml.data(), saved_xml.size(),
                                  error.data(), error.size());
  ASSERT_EQ(save_res, 0) << error.data();

  // verify text_attr uses attribute data="..." while text_cdata uses CDATA
  EXPECT_THAT(saved_xml.data(),
              HasSubstr(R"(<text name="text_attr" data="simple_value"/>)"));
  EXPECT_THAT(saved_xml.data(),
              HasSubstr(R"(<text name="text_cdata"><![CDATA[)"));

  // reload saved XML
  mjSpec* saved_spec =
      mj_parseXMLString(saved_xml.data(), nullptr, error.data(), error.size());
  ASSERT_THAT(saved_spec, NotNull()) << error.data();

  mjModel* m = mj_compile(saved_spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(saved_spec);

  int id_attr = mj_name2id(m, mjOBJ_TEXT, "text_attr");
  ASSERT_GE(id_attr, 0);
  EXPECT_STREQ(m->text_data + m->text_adr[id_attr], "simple_value");

  int id_cdata = mj_name2id(m, mjOBJ_TEXT, "text_cdata");
  ASSERT_GE(id_cdata, 0);
  const char* cdata_val = m->text_data + m->text_adr[id_cdata];
  EXPECT_THAT(cdata_val, HasSubstr("message {"));
  EXPECT_THAT(cdata_val,
              HasSubstr("field_str: \"hello <world> & \\\"quotes\\\"\""));
  EXPECT_THAT(cdata_val, HasSubstr("field_int: 42"));

  mj_deleteModel(m);
  mj_deleteModel(m_orig);
  mj_deleteSpec(saved_spec);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, DeepBodyNesting) {
  constexpr int kDepth = 100;
  std::string xml = "<mujoco><worldbody>";
  for (int i = 0; i < kDepth; ++i) {
    xml +=
        "<body pos=\"0 .05 0\"><joint type=\"hinge\" armature=\"0.1\"/>"
        "<geom type=\"capsule\" size=\"0.01 0.02\"/>";
  }
  for (int i = 0; i < kDepth; ++i) {
    xml += "</body>";
  }
  xml += "</worldbody></mujoco>";

  std::array<char, 1024> error;
  mjSpec* spec =
      mj_parseXMLString(xml.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_EQ(m->nbody, kDepth + 1);
  EXPECT_EQ(m->njnt, kDepth);
  EXPECT_EQ(m->ngeom, kDepth);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, DefaultTendonArmatureAndActuatorfrc) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <tendon armature="0.5" actuatorfrclimited="true" actuatorfrcrange="-2 3"/>
    </default>
    <worldbody>
      <body>
        <joint name="j" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t">
        <joint joint="j" coef="1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();
  mjModel* m = mj_compile(spec, nullptr);
  ASSERT_THAT(m, NotNull()) << mjs_getError(spec);
  EXPECT_MJTNUM_EQ(m->tendon_armature[0], 0.5);
  EXPECT_EQ(m->tendon_actfrclimited[0], 1);
  EXPECT_MJTNUM_EQ(m->tendon_actfrcrange[0], -2);
  EXPECT_MJTNUM_EQ(m->tendon_actfrcrange[1], 3);

  std::array<char, 2048> saved_xml;
  ASSERT_EQ(mj_saveXMLString(spec, saved_xml.data(), saved_xml.size(),
                             error.data(), error.size()),
            0)
      << error.data();
  MjModelPtr reloaded =
      LoadModelFromString(saved_xml.data(), error.data(), error.size());
  ASSERT_THAT(reloaded.get(), NotNull()) << error.data();
  EXPECT_MJTNUM_EQ(reloaded->tendon_armature[0], 0.5);
  EXPECT_EQ(reloaded->tendon_actfrclimited[0], 1);
  EXPECT_MJTNUM_EQ(reloaded->tendon_actfrcrange[0], -2);
  EXPECT_MJTNUM_EQ(reloaded->tendon_actfrcrange[1], 3);

  mj_deleteModel(m);
  mj_deleteSpec(spec);
}

TEST_F(XMLReaderTest, NestedFallbackAndDuplicateInclude) {
  static constexpr char root_ok[] = R"(
  <mujoco>
    <worldbody>
      <include file="sub/mid.xml"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char mid_xml[] = R"(
  <mujoco>
    <include file="other/leaf.xml"/>
  </mujoco>
  )";
  static constexpr char leaf_xml[] = R"(
  <mujoco>
    <include file="deep.xml"/>
  </mujoco>
  )";
  static constexpr char deep_xml[] = R"(
  <mujoco>
    <geom name="deep_geom" size="0.1"/>
  </mujoco>
  )";

  MockFilesystem fs("NestedFallbackAndDuplicateInclude");
  fs.AddFile("root.xml", (const unsigned char*)root_ok, sizeof(root_ok));
  fs.AddFile("sub/mid.xml", (const unsigned char*)mid_xml, sizeof(mid_xml));
  fs.AddFile("other/leaf.xml", (const unsigned char*)leaf_xml,
             sizeof(leaf_xml));
  fs.AddFile("other/deep.xml", (const unsigned char*)deep_xml,
             sizeof(deep_xml));

  std::array<char, 1024> error;
  std::string root_path = fs.FullPath("root.xml");
  mjModel* m =
      mj_loadXML(root_path.c_str(), nullptr, error.data(), error.size());
  ASSERT_THAT(m, NotNull()) << error.data();
  EXPECT_GE(mj_name2id(m, mjOBJ_GEOM, "deep_geom"), 0);
  mj_deleteModel(m);

  // duplicate include via fallback path and direct path must be rejected
  static constexpr char root_dup_fallback[] = R"(
  <mujoco>
    <worldbody>
      <include file="other/leaf.xml"/>
      <include file="sub/mid.xml"/>
    </worldbody>
  </mujoco>
  )";
  fs.AddFile("root_dup1.xml", (const unsigned char*)root_dup_fallback,
             sizeof(root_dup_fallback));
  std::string dup1_path = fs.FullPath("root_dup1.xml");
  EXPECT_THAT(
      mj_loadXML(dup1_path.c_str(), nullptr, error.data(), error.size()),
      IsNull());
  EXPECT_THAT(error.data(), HasSubstr("already included"));

  // duplicate include via relative path after direct include must be rejected
  static constexpr char root_dup_rel[] = R"(
  <mujoco>
    <worldbody>
      <include file="other/deep.xml"/>
      <include file="other/leaf.xml"/>
    </worldbody>
  </mujoco>
  )";
  fs.AddFile("root_dup2.xml", (const unsigned char*)root_dup_rel,
             sizeof(root_dup_rel));
  std::string dup2_path = fs.FullPath("root_dup2.xml");
  EXPECT_THAT(
      mj_loadXML(dup2_path.c_str(), nullptr, error.data(), error.size()),
      IsNull());
  EXPECT_THAT(error.data(), HasSubstr("already included"));

  // duplicate nested include in VFS must not double-prefix filename
  static constexpr char vfs_root[] = R"(
  <mujoco>
    <worldbody>
      <include file="submodels/model1.xml"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char vfs_mid[] = R"(
  <mujoco>
    <include file="subsubmodels/model2.xml"/>
    <include file="subsubmodels/model2.xml"/>
  </mujoco>
  )";
  mjVFS vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "model.xml", vfs_root, sizeof(vfs_root));
  mj_addBufferVFS(&vfs, "submodels/model1.xml", vfs_mid, sizeof(vfs_mid));
  mj_addBufferVFS(&vfs, "submodels/subsubmodels/model2.xml", deep_xml,
                  sizeof(deep_xml));
  EXPECT_THAT(mj_loadXML("model.xml", &vfs, error.data(), error.size()),
              IsNull());
  EXPECT_THAT(error.data(), HasSubstr("already included"));
  mj_deleteVFS(&vfs);

  // nested relative includes when root model itself is loaded by relative path
  static constexpr char rel_root[] = R"(
  <mujoco>
    <worldbody>
      <include file="sub/a.xml"/>
    </worldbody>
  </mujoco>
  )";
  static constexpr char rel_a[] = R"(
  <mujoco>
    <include file="b.xml"/>
  </mujoco>
  )";
  static constexpr char rel_b[] = R"(
  <mujoco>
    <include file="other/c.xml"/>
  </mujoco>
  )";
  static constexpr char rel_c[] = R"(
  <mujoco>
    <geom name="rel_geom" size="0.1"/>
  </mujoco>
  )";
  mjVFS rel_vfs;
  mj_defaultVFS(&rel_vfs);
  mj_addBufferVFS(&rel_vfs, "models/root.xml", rel_root, sizeof(rel_root));
  mj_addBufferVFS(&rel_vfs, "models/sub/a.xml", rel_a, sizeof(rel_a));
  mj_addBufferVFS(&rel_vfs, "models/sub/b.xml", rel_b, sizeof(rel_b));
  mj_addBufferVFS(&rel_vfs, "models/other/c.xml", rel_c, sizeof(rel_c));
  mjModel* m_rel =
      mj_loadXML("models/root.xml", &rel_vfs, error.data(), error.size());
  ASSERT_THAT(m_rel, NotNull()) << error.data();
  EXPECT_GE(mj_name2id(m_rel, mjOBJ_GEOM, "rel_geom"), 0);
  mj_deleteModel(m_rel);
  mj_deleteVFS(&rel_vfs);
}

TEST_F(XMLReaderTest, TextureGridsizePreservesSameAndResetsDifferentLayout) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="grid_cls">
        <texture type="skybox" gridsize="3 4" gridlayout=".U..LFRB.D.."/>
      </default>
    </default>
    <asset>
      <texture name="inherited" class="grid_cls" file="sky.png"/>
      <texture name="same_size" class="grid_cls" file="sky.png" gridsize="3 4"/>
      <texture name="diff_size" class="grid_cls" file="sky.png" gridsize="2 3"/>
    </asset>
  </mujoco>
  )";
  std::array<char, 1024> error;
  mjSpec* spec = mj_parseXMLString(xml, nullptr, error.data(), error.size());
  ASSERT_THAT(spec, NotNull()) << error.data();

  auto tex = [spec](const char* name) {
    return mjs_asTexture(mjs_findElement(spec, mjOBJ_TEXTURE, name));
  };
  ASSERT_THAT(tex("inherited"), NotNull());
  ASSERT_THAT(tex("same_size"), NotNull());
  ASSERT_THAT(tex("diff_size"), NotNull());
  EXPECT_EQ(std::string(tex("inherited")->gridlayout, 12), ".U..LFRB.D..");
  EXPECT_EQ(std::string(tex("same_size")->gridlayout, 12), ".U..LFRB.D..");
  EXPECT_EQ(std::string(tex("diff_size")->gridlayout, 12), "............");

  mj_deleteSpec(spec);
}

TEST_F(ActuatorParseTest, ShortcutInheritrangeDefaultOverride) {
  static constexpr char xml[] = R"(
  <mujoco>
    <default>
      <default class="pos_inh">
        <position inheritrange="1"/>
      </default>
      <default class="intvel_inh">
        <intvelocity inheritrange="1"/>
      </default>
      <default class="pid_inh">
        <pid inheritrange="1" kp="2"/>
      </default>
      <default class="pos_exp">
        <position ctrlrange="-3 3"/>
      </default>
      <default class="intvel_exp">
        <intvelocity actrange="-4 4"/>
      </default>
      <default class="pid_exp">
        <pid ctrlrange="-5 5" kp="2"/>
      </default>
    </default>
    <worldbody>
      <body>
        <joint name="j" type="slide" range="-1 2"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <actuator>
      <position class="pos_inh" joint="j" ctrlrange="-3 3"/>
      <intvelocity class="intvel_inh" joint="j" actrange="-4 4"/>
      <pid class="pid_inh" joint="j" ctrlrange="-5 5"/>
      <position class="pos_exp" joint="j" inheritrange="1"/>
      <intvelocity class="intvel_exp" joint="j" inheritrange="1"/>
      <pid class="pid_exp" joint="j" inheritrange="1"/>
    </actuator>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[0] + 0], -3);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[0] + 1], 3);
  EXPECT_MJTNUM_EQ(m->actuator_actrange[1 * 2 + 0], -4);
  EXPECT_MJTNUM_EQ(m->actuator_actrange[1 * 2 + 1], 4);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[2] + 0], -5);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[2] + 1], 5);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[3] + 0], -1);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[3] + 1], 2);
  EXPECT_MJTNUM_EQ(m->actuator_actrange[4 * 2 + 0], -1);
  EXPECT_MJTNUM_EQ(m->actuator_actrange[4 * 2 + 1], 2);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[5] + 0], -1);
  EXPECT_MJTNUM_EQ(m->actuator_ctrlrange[2 * m->actuator_ctrladr[5] + 1], 2);
}

TEST_F(XMLReaderTest, SubnormalFloatAndOverflowAttributes) {
  static constexpr char xml_subnormal[] = R"(
  <mujoco>
    <worldbody>
      <geom size="0.1" rgba="1e-40 1e-45 1e-50 1" pos="1e-315 0 0"/>
    </worldbody>
  </mujoco>
  )";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml_subnormal, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  EXPECT_GT(m->geom_rgba[0], 0.0f);
  EXPECT_GT(m->geom_rgba[1], 0.0f);
  EXPECT_EQ(m->geom_rgba[2], 0.0f);
  EXPECT_EQ(m->geom_rgba[3], 1.0f);

  static constexpr char xml_overflow[] = R"(
  <mujoco>
    <worldbody>
      <geom size="0.1" rgba="1e40 1 1 1"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr bad =
      LoadModelFromString(xml_overflow, error.data(), error.size());
  EXPECT_THAT(bad.get(), IsNull());
  EXPECT_THAT(error.data(), HasSubstr("number is too large"));
}

TEST_F(XMLReaderTest, CustomNumericLargeAndInvalidSizes) {
  std::string values;
  for (int i = 0; i < 600; ++i) {
    if (i > 0) values += ' ';
    values += std::to_string(i + 1);
  }
  std::string xml_large =
      "<mujoco><custom>"
      "<numeric name=\"explicit\" size=\"650\" data=\"1 2 3\"/>"
      "<numeric name=\"inferred\" data=\"" +
      values +
      "\"/>"
      "</custom></mujoco>";
  std::array<char, 1024> error;
  MjModelPtr m = LoadModelFromString(xml_large, error.data(), error.size());
  ASSERT_THAT(m.get(), NotNull()) << error.data();
  ASSERT_EQ(m->nnumeric, 2);
  EXPECT_EQ(m->numeric_size[0], 650);
  EXPECT_MJTNUM_EQ(m->numeric_data[0], 1);
  EXPECT_MJTNUM_EQ(m->numeric_data[2], 3);
  EXPECT_MJTNUM_EQ(m->numeric_data[649], 0);
  EXPECT_EQ(m->numeric_size[1], 600);
  EXPECT_MJTNUM_EQ(m->numeric_data[m->numeric_adr[1] + 599], 600);

  for (const char* bad_xml : {
           "<mujoco><custom>"
           "<numeric name=\"a\" size=\"0\" data=\"1\"/>"
           "</custom></mujoco>",
           "<mujoco><custom><numeric name=\"a\" "
           "size=\"-2\"/></custom></mujoco>",
           "<mujoco><custom><numeric name=\"a\"/></custom></mujoco>",
           "<mujoco><custom>"
           "<numeric name=\"a\" size=\"2\" data=\"1 2 3\"/>"
           "</custom></mujoco>",
       }) {
    MjModelPtr bad = LoadModelFromString(bad_xml, error.data(), error.size());
    EXPECT_THAT(bad.get(), IsNull()) << bad_xml;
  }

  MjModelPtr bad_size = LoadModelFromString(
      "<mujoco><custom><numeric name=\"a\" "
      "size=\"2000000000\"/></custom></mujoco>",
      error.data(), error.size());
  EXPECT_THAT(bad_size.get(), IsNull());
  EXPECT_THAT(error.data(),
              testing::HasSubstr("custom field size exceeds maximum"));
}

TEST_F(XMLReaderTest, HeadlightActive) {
  struct Case {
    const char* attr;
    int expected;
  };
  const Case valid_cases[] = {
      {"false", 0}, {"true", 1}, {"auto", 2}, {"0", 0}, {"1", 1}, {"2", 2},
  };
  std::array<char, 1024> error;
  for (const Case& c : valid_cases) {
    std::string xml = std::string("<mujoco><visual><headlight active=\"") +
                      c.attr + "\"/></visual></mujoco>";
    mjSpec* spec =
        mj_parseXMLString(xml.c_str(), nullptr, error.data(), error.size());
    ASSERT_THAT(spec, NotNull()) << c.attr << ": " << error.data();
    EXPECT_EQ(spec->visual.headlight.active, c.expected) << c.attr;
    EXPECT_EQ(mjs_isAuthored(spec, &spec->visual.headlight.active), 1)
        << c.attr;
    mjModel* m = mj_compile(spec, nullptr);
    ASSERT_THAT(m, NotNull()) << c.attr << ": " << mjs_getError(spec);
    EXPECT_EQ(m->vis.headlight.active, c.expected) << c.attr;
    mj_deleteModel(m);
    mj_deleteSpec(spec);
  }

  for (const char* out_of_range : {"-1", "3"}) {
    std::string xml = std::string("<mujoco><visual><headlight active=\"") +
                      out_of_range + "\"/></visual></mujoco>";
    MjModelPtr bad =
        LoadModelFromString(xml.c_str(), error.data(), error.size());
    EXPECT_THAT(bad.get(), IsNull()) << out_of_range;
    EXPECT_THAT(error.data(), HasSubstr("invalid keyword")) << out_of_range;
  }
}

}  // namespace
}  // namespace mujoco
