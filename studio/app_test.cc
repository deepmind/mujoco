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

#include "studio/app.h"

#include <cstddef>
#include <cstdlib>
#include <span>

#include "testing/base/public/gunit.h"
#include <mujoco/mujoco.h>
#include "studio/hal/graphics_mode.h"

namespace mujoco::studio {

struct AppTestPeer {
  static mjSpec* spec(App& app) { return app.has_spec() ? app.spec() : nullptr; }
  static mjsElement* curr_element(const App& app) { return app.tmp_.curr_element; }
  static void set_curr_element(App& app, mjsElement* element) { app.tmp_.curr_element = element; }
};

namespace {

TEST(AppTest, CurrElementResetOnModelReloadAndEmptyInit) {
  setenv("SDL_VIDEODRIVER", "dummy", 1);

  App::Config config;
  config.width = 640;
  config.height = 480;
  config.gfx_mode = GraphicsMode::FilamentOpenGlSoftware;
  App app(config);

  static constexpr char kXml1[] = R"(
  <mujoco>
    <worldbody>
      <body name="body1">
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  static constexpr char kXml2[] = R"(
  <mujoco>
    <worldbody>
      <body name="body2">
        <geom type="box" size="0.1 0.1 0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  std::span<const std::byte> buf1(reinterpret_cast<const std::byte*>(kXml1), sizeof(kXml1) - 1);
  std::span<const std::byte> buf2(reinterpret_cast<const std::byte*>(kXml2), sizeof(kXml2) - 1);

  app.LoadModelFromBuffer(buf1, "text/xml", "m1.xml");
  ASSERT_NE(AppTestPeer::spec(app), nullptr);
  mjsBody* b1 = mjs_findBody(AppTestPeer::spec(app), "body1");
  ASSERT_NE(b1, nullptr);
  mjsElement* el = b1->element;
  ASSERT_NE(el, nullptr);

  AppTestPeer::set_curr_element(app, el);
  EXPECT_EQ(AppTestPeer::curr_element(app), el);

  app.LoadModelFromBuffer(buf2, "text/xml", "m2.xml");
  EXPECT_EQ(AppTestPeer::curr_element(app), nullptr);

  mjsBody* b2 = mjs_findBody(AppTestPeer::spec(app), "body2");
  ASSERT_NE(b2, nullptr);
  AppTestPeer::set_curr_element(app, b2->element);
  EXPECT_EQ(AppTestPeer::curr_element(app), b2->element);

  app.InitEmptyModel();
  EXPECT_EQ(AppTestPeer::curr_element(app), nullptr);
}

}  // namespace
}  // namespace mujoco::studio
