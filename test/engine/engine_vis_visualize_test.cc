// Copyright 2024 DeepMind Technologies Limited
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

#include <cmath>
#include <cstddef>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;

static const char* const kModelPath = "testdata/model.xml";

class MjvSceneTest : public MujocoTest {
 protected:
  static constexpr int kMaxGeom = 10000;

  void InitSceneObjects(mjModel* model, int maxgeom = kMaxGeom) {
    mjv_defaultScene(&scn_);
    mjv_makeScene(model, &scn_, maxgeom);
    mjv_defaultOption(&opt_);
    mjv_defaultPerturb(&pert_);
    mjv_defaultFreeCamera(model, &cam_);

    // enable flags to exercise additional code paths
    for (int i = 0; i < mjNVISFLAG; ++i) {
      opt_.flags[i] = 1;
    }
  }

  void FreeSceneObjects() { mjv_freeScene(&scn_); }

  mjvScene scn_;
  mjvOption opt_;
  mjvPerturb pert_;
  mjvCamera cam_;
};

TEST_F(MjvSceneTest, UpdateScene) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull())
      << "Failed to load model from " << kModelPath << ": " << error;

  InitSceneObjects(model);

  mjData* data = mj_makeData(model);
  while (data->time < .2) {
    mj_step(model, data);
  }

  mjv_updateScene(model, data, &opt_, &pert_, &cam_, mjCAT_ALL, &scn_);
  EXPECT_EQ(scn_.status, 0);
  EXPECT_GT(scn_.ngeom, 0);
  EXPECT_GT(scn_.nlight, 0);
  if (model->nskin) EXPECT_GT(scn_.nskin, 0);
  if (model->nflex) EXPECT_GT(scn_.nflex, 0);

  mjv_updateScene(model, data, &opt_, &pert_, &cam_, mjCAT_ALL, &scn_);

  // call mj_copyData to expose any memory leaks in mjv_updateScene
  mjData* data_copy = mj_copyData(nullptr, model, data);

  mj_deleteData(data_copy);
  mj_deleteData(data);
  FreeSceneObjects();
  mj_deleteModel(model);
}

TEST_F(MjvSceneTest, UpdateSceneGeomsExhausted) {
  const std::string xml_path = GetTestDataFilePath(kModelPath);
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull())
      << "Failed to load model from " << kModelPath << ": " << error;

  const int maxgeoms = 1;
  InitSceneObjects(model, maxgeoms);

  mjData* data = mj_makeData(model);
  mj_forward(model, data);

  // clear handlers to avoid test failure; we are explicitly expecting a warning
  mju_clearHandlers();
  mjv_updateScene(model, data, &opt_, &pert_, &cam_, mjCAT_ALL, &scn_);
  EXPECT_EQ(scn_.status, 1);
  EXPECT_EQ(scn_.ngeom, maxgeoms);
  mj_deleteData(data);
  FreeSceneObjects();
  mj_deleteModel(model);
}

TEST_F(MjvSceneTest, PrincipalPointFrustumSign) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="cam" pos="0 0 1" zaxis="0 0 1"
              sensorsize="0.01 0.01" focal="0.01 0.01"
              principal="0 0.002"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());

  // point camera at the fixed cam
  cam_.type = mjCAMERA_FIXED;
  cam_.fixedcamid = 0;
  mjv_updateCamera(model.get(), data.get(), &cam_, &scn_);

  float top = scn_.camera[0].frustum_top;
  float bottom = scn_.camera[0].frustum_bottom;

  // with cy > 0 the principal point is above center, so the frustum should
  // extend further downward than upward: |bottom| > top
  EXPECT_GT(-bottom, top);

  // verify exact values against the pinhole model
  float znear = model->vis.map.znear * model->stat.extent;
  float cy = model->cam_intrinsic[3];
  float fy = model->cam_intrinsic[1];
  float sh = model->cam_sensorsize[1];
  float half = znear / fy * (sh / 2);
  float offset = znear / fy * cy;

  EXPECT_FLOAT_EQ(top, half - offset);
  EXPECT_FLOAT_EQ(bottom, -(half + offset));

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, InvalidFixedCamId) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="cam" pos="0 0 1"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  cam_.type = mjCAMERA_FIXED;

  cam_.fixedcamid = -1;
  EXPECT_THAT(MjuErrorMessageFrom(mjv_updateCamera)(model.get(), data.get(),
                                                    &cam_, &scn_),
              ::testing::HasSubstr("fixed camera id is outside valid range"));
  EXPECT_THAT(MjuErrorMessageFrom(mjv_cameraFrame)(nullptr, nullptr, nullptr,
                                                   nullptr, data.get(), &cam_),
              ::testing::HasSubstr("fixed camera id is outside valid range"));

  cam_.fixedcamid = model->ncam;
  EXPECT_THAT(MjuErrorMessageFrom(mjv_updateCamera)(model.get(), data.get(),
                                                    &cam_, &scn_),
              ::testing::HasSubstr("fixed camera id is outside valid range"));

  FreeSceneObjects();
}

static constexpr char kMeshSiteXml[] = R"(
<mujoco>
  <asset>
    <mesh name="offset_box"
          vertex="1 -.5 0  3 -.5 0  3 .5 0  1 .5 0
                  1 -.5 4  3 -.5 4  3 .5 4  1 .5 4"/>
  </asset>
  <worldbody>
    <body pos=".4 .5 .6" euler="10 20 30">
      <freejoint/>
      <inertial pos="0 0 0" mass="1" diaginertia="1 2 3"/>
      <frame pos=".3 -.2 .1" euler="30 10 20">
        <site name="plain" pos=".1 .2 .3" euler="20 30 10" size=".01"/>
        <site name="mesh" type="mesh" mesh="offset_box"
              pos=".1 .2 .3" euler="20 30 10"/>
        <geom name="reference" type="mesh" mesh="offset_box"
              pos=".1 .2 .3" euler="20 30 10" contype="0" conaffinity="0"/>
      </frame>
    </body>
  </worldbody>
  <sensor>
    <framepos objtype="site" objname="plain"/>
    <framepos objtype="site" objname="mesh"/>
    <framequat objtype="site" objname="plain"/>
    <framequat objtype="site" objname="mesh"/>
    <gyro site="plain"/>
    <gyro site="mesh"/>
    <accelerometer site="plain"/>
    <accelerometer site="mesh"/>
  </sensor>
</mujoco>
)";

TEST_F(MjvSceneTest, MeshSitePreservesAuthoredFrame) {
  MjModelPtr model = LoadModelFromString(kMeshSiteXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  int plain = mj_name2id(model.get(), mjOBJ_SITE, "plain");
  int mesh = mj_name2id(model.get(), mjOBJ_SITE, "mesh");
  for (int j = 0; j < 3; ++j) {
    EXPECT_EQ(model->site_pos[3 * plain + j], model->site_pos[3 * mesh + j]);
  }
  for (int j = 0; j < 4; ++j) {
    EXPECT_EQ(model->site_quat[4 * plain + j], model->site_quat[4 * mesh + j]);
  }
  for (int i = 0; i < model->nv; ++i) {
    data->qvel[i] = 0.1 * (i + 1);
    data->qfrc_applied[i] = 0.2 * (i + 1);
  }
  mj_forward(model.get(), data.get());
  for (int sensor = 0; sensor < model->nsensor; sensor += 2) {
    int adr1 = model->sensor_adr[sensor];
    int adr2 = model->sensor_adr[sensor + 1];
    for (int j = 0; j < model->sensor_dim[sensor]; ++j) {
      EXPECT_NEAR(data->sensordata[adr1 + j], data->sensordata[adr2 + j], 1e-6);
    }
  }
}

TEST_F(MjvSceneTest, MeshSitePreservesVisualPoseAndVolume) {
  MjModelPtr model = LoadModelFromString(kMeshSiteXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  int site = mj_name2id(model.get(), mjOBJ_SITE, "mesh");
  int geom = mj_name2id(model.get(), mjOBJ_GEOM, "reference");
  InitSceneObjects(model.get());
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);
  const mjvGeom* mesh_site = nullptr;
  const mjvGeom* mesh_geom = nullptr;
  for (int i = 0; i < scn_.ngeom; ++i) {
    const mjvGeom* item = scn_.geoms + i;
    if (item->objtype == mjOBJ_SITE && item->objid == site) mesh_site = item;
    if (item->objtype == mjOBJ_GEOM && item->objid == geom) mesh_geom = item;
  }
  ASSERT_THAT(mesh_site, NotNull());
  ASSERT_THAT(mesh_geom, NotNull());
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(mesh_site->pos[j], mesh_geom->pos[j], 1e-6);
  }
  for (int j = 0; j < 9; ++j) {
    EXPECT_NEAR(mesh_site->mat[j], mesh_geom->mat[j], 1e-6);
  }
  EXPECT_EQ(
      mj_insideSite(model.get(), data.get(), site, data->geom_xpos + 3 * geom),
      1);
  mjtNum outside[3] = {100, 100, 100};
  EXPECT_EQ(mj_insideSite(model.get(), data.get(), site, outside), 0);
  FreeSceneObjects();
}

TEST_F(MjvSceneTest, TendonWrapGeodesicLength) {
  constexpr char kWrapXml[] = R"(
  <mujoco>
    <worldbody>
      <site name="0a" pos="0 0 1"/>
      <geom name="0g" type="sphere" size=".1" pos="0 0 0.5"/>
      <site name="0b" pos="0 .05 .15"/>

      <site name="1a" pos=".4 0 1"/>
      <geom name="1g" type="sphere" size=".1" pos=".4 0 0.5"/>
      <site name="1s" pos=".52 0 .5"/>
      <site name="1b" pos=".25 0 .15"/>

      <site name="2a" pos="-.4 0 1"/>
      <geom name="2g" type="sphere" size=".1" pos="-.4 0 0.5"/>
      <site name="2s" pos="-.49 0 .5"/>
      <site name="2b" pos="-.4 0 .15"/>

      <site name="3a" pos="0 1 1"/>
      <geom name="3g" type="cylinder" size=".05 .2" zaxis="0 1 .2" pos="0 1 0.5"/>
      <site name="3s" pos=".12 1 .5"/>
      <site name="3b" pos="-.15 1.1 .15"/>
    </worldbody>
    <tendon>
      <spatial>
        <site site="0a"/>
        <geom geom="0g"/>
        <site site="0b"/>
      </spatial>
      <spatial>
        <site site="1a"/>
        <geom geom="1g" sidesite="1s"/>
        <site site="1b"/>
      </spatial>
      <spatial>
        <site site="2a"/>
        <geom geom="2g" sidesite="2s"/>
        <site site="2b"/>
      </spatial>
      <spatial>
        <site site="3a"/>
        <geom geom="3g" sidesite="3s"/>
        <site site="3b"/>
      </spatial>
    </tendon>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kWrapXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  for (int i = 0; i < model->ntendon; ++i) {
    mjtNum vis_length = 0;
    for (int g = 0; g < scn_.ngeom; ++g) {
      const mjvGeom* geom = scn_.geoms + g;
      if (geom->objtype == mjOBJ_TENDON && geom->objid == i) {
        vis_length += 2 * geom->size[2];
      }
    }
    EXPECT_NEAR(vis_length, data->ten_length[i], 1e-3) << "tendon=" << i;
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, FlexDim2FlatskinNormalsPointOutward) {
  constexpr char kFlexXml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="sheet" type="direct" dim="2" radius="0.01" flatskin="true"
                point="0 0 0  1 0 0  0 1 0" element="0 1 2"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kFlexXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  // First 3 vertices belong to the +radius face (normal +z), next 3 to the
  // -radius face (normal -z).
  for (int k = 0; k < 3; ++k) {
    EXPECT_NEAR(scn_.flexnormal[3 * k + 0], 0.0f, 1e-5f);
    EXPECT_NEAR(scn_.flexnormal[3 * k + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(scn_.flexnormal[3 * k + 2], 1.0f, 1e-5f);
  }
  for (int k = 3; k < 6; ++k) {
    EXPECT_NEAR(scn_.flexnormal[3 * k + 0], 0.0f, 1e-5f);
    EXPECT_NEAR(scn_.flexnormal[3 * k + 1], 0.0f, 1e-5f);
    EXPECT_NEAR(scn_.flexnormal[3 * k + 2], -1.0f, 1e-5f);
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, ContactSurfaceVelocityArrow) {
  // Static belt before body, static belt after body, and welded static body
  // declared after movable body (higher bodyid, still static).
  constexpr char kBeltFirstXml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="belt" type="plane" size="5 5 .1" surfacevel="1 0 0  0 0 0"/>
      <body pos="0 0 .09">
        <freejoint/>
        <geom type="sphere" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  constexpr char kBeltSecondXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 .09">
        <freejoint/>
        <geom type="sphere" size=".1"/>
      </body>
      <geom name="belt" type="plane" size="5 5 .1" surfacevel="1 0 0  0 0 0"/>
    </worldbody>
  </mujoco>
  )";
  constexpr char kWeldedBeltAfterBodyXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 .09">
        <freejoint/>
        <geom type="sphere" size=".1"/>
      </body>
      <body name="conveyor">
        <geom name="belt" type="plane" size="5 5 .1"
          surfacevel="1 0 0  0 0 0"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  for (const char* xml :
       {kBeltFirstXml, kBeltSecondXml, kWeldedBeltAfterBodyXml}) {
    MjModelPtr model = LoadModelFromString(xml);
    ASSERT_THAT(model.get(), NotNull());
    MjDataPtr data = MakeData(model);
    mj_forward(model.get(), data.get());
    ASSERT_EQ(data->ncon, 1);

    InitSceneObjects(model.get());
    mjv_defaultOption(&opt_);
    opt_.flags[mjVIS_CONTACTPOINT] = 1;
    mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                    &scn_);

    int n_arrows = 0;
    int n_arrows2 = 0;
    const mjvGeom* arrow = nullptr;
    for (int i = 0; i < scn_.ngeom; ++i) {
      if (scn_.geoms[i].type == mjGEOM_ARROW) {
        arrow = &scn_.geoms[i];
        ++n_arrows;
      } else if (scn_.geoms[i].type == mjGEOM_ARROW2) {
        ++n_arrows2;
      }
    }
    EXPECT_EQ(n_arrows, 1);
    EXPECT_EQ(n_arrows2, 0);
    if (arrow) {
      EXPECT_NEAR(arrow->size[2], 0.5f, 1e-5f);
      // Arrow +z axis (third column of mat) points along +x belt velocity.
      EXPECT_NEAR(arrow->mat[2], 1.0f, 1e-5f);
      EXPECT_NEAR(arrow->mat[5], 0.0f, 1e-5f);
      EXPECT_NEAR(arrow->mat[8], 0.0f, 1e-5f);
    }

    // Frictionless contacts (dim == 1) do not draw a surface velocity arrow.
    data->contact[0].dim = 1;
    mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                    &scn_);
    n_arrows = 0;
    for (int i = 0; i < scn_.ngeom; ++i) {
      if (scn_.geoms[i].type == mjGEOM_ARROW ||
          scn_.geoms[i].type == mjGEOM_ARROW2) {
        ++n_arrows;
      }
    }
    EXPECT_EQ(n_arrows, 0);

    FreeSceneObjects();
  }

  // Two movable bodies: double-sided arrow (mjGEOM_ARROW2) along tangent axis.
  constexpr char kTwoMovableXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <freejoint/>
        <geom type="sphere" size=".1" surfacevel="1 0 0  0 0 0"/>
      </body>
      <body pos="0 0 .19">
        <freejoint/>
        <geom type="sphere" size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(kTwoMovableXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 1);

  InitSceneObjects(model.get());
  mjv_defaultOption(&opt_);
  opt_.flags[mjVIS_CONTACTPOINT] = 1;
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  int n_arrows = 0;
  int n_arrows2 = 0;
  const mjvGeom* arrow2 = nullptr;
  for (int i = 0; i < scn_.ngeom; ++i) {
    if (scn_.geoms[i].type == mjGEOM_ARROW) {
      ++n_arrows;
    } else if (scn_.geoms[i].type == mjGEOM_ARROW2) {
      arrow2 = &scn_.geoms[i];
      ++n_arrows2;
    }
  }
  EXPECT_EQ(n_arrows, 0);
  EXPECT_EQ(n_arrows2, 1);
  if (arrow2) {
    EXPECT_NEAR(arrow2->size[2], 0.5f, 1e-5f);
    EXPECT_NEAR(std::abs(arrow2->mat[2]), 1.0f, 1e-5f);
    EXPECT_NEAR(arrow2->mat[5], 0.0f, 1e-5f);
    EXPECT_NEAR(arrow2->mat[8], 0.0f, 1e-5f);
  }
  FreeSceneObjects();
}

TEST_F(MjvSceneTest, GeomCastshadow) {
  constexpr char kXml[] = R"(
  <mujoco>
    <default>
      <default class="noshadow">
        <geom castshadow="false"/>
      </default>
    </default>
    <worldbody>
      <geom name="default_geom" type="sphere" size="0.1"/>
      <geom name="noshadow_explicit" type="sphere" size="0.1" castshadow="false"/>
      <geom name="noshadow_class" class="noshadow" type="sphere" size="0.1"/>
      <geom name="override_class" class="noshadow" type="sphere" size="0.1" castshadow="true"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  ASSERT_EQ(model->ngeom, 4);
  EXPECT_EQ(model->geom_castshadow[0], 1);
  EXPECT_EQ(model->geom_castshadow[1], 0);
  EXPECT_EQ(model->geom_castshadow[2], 0);
  EXPECT_EQ(model->geom_castshadow[3], 1);

  // XML save/load round-trip preserves geom_castshadow
  const std::string saved_xml = SaveAndReadXml(model.get());
  MjModelPtr reloaded = LoadModelFromString(saved_xml);
  ASSERT_THAT(reloaded.get(), NotNull());
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(reloaded->geom_castshadow[i], model->geom_castshadow[i])
        << "i=" << i;
  }

  // mjv_initGeom defaults castshadow to 1
  mjvGeom init_geom;
  mjv_initGeom(&init_geom, mjGEOM_SPHERE, nullptr, nullptr, nullptr, nullptr);
  EXPECT_EQ(init_geom.castshadow, 1);

  // mjv_updateScene propagates model->geom_castshadow and picks up runtime
  // edits
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());
  InitSceneObjects(model.get());
  mjv_defaultOption(&opt_);
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_STATIC,
                  &scn_);

  for (int g = 0; g < scn_.ngeom; ++g) {
    const mjvGeom& geom = scn_.geoms[g];
    if (geom.objtype == mjOBJ_GEOM) {
      EXPECT_EQ(geom.castshadow, model->geom_castshadow[geom.objid])
          << "objid=" << geom.objid;
    }
  }

  model->geom_castshadow[0] = 0;
  model->geom_castshadow[1] = 1;
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_STATIC,
                  &scn_);
  for (int g = 0; g < scn_.ngeom; ++g) {
    const mjvGeom& geom = scn_.geoms[g];
    if (geom.objtype == mjOBJ_GEOM) {
      EXPECT_EQ(geom.castshadow, model->geom_castshadow[geom.objid])
          << "objid=" << geom.objid;
    }
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, AddFlexBvhGeomsDoesNotLeakStack) {
  static constexpr char kFlexXml[] = R"(
  <mujoco>
    <worldbody>
      <body name="softbody">
        <flexcomp name="soft" type="grid" count="2 2 2" spacing=".05 .05 .05"
                  dim="3" radius=".001" mass=".005" dof="trilinear"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(kFlexXml);
  ASSERT_THAT(model.get(), NotNull());
  ASSERT_GT(model->nflex, 0);
  ASSERT_GT(model->flex_interp[0], 0);

  InitSceneObjects(model.get());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  size_t pstack_before = data->pstack;
  for (int frame = 0; frame < 5; ++frame) {
    mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                    &scn_);
    EXPECT_EQ(data->pstack, pstack_before)
        << "pstack leaked at frame " << frame;
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, AddMeshBvhGeomsSkipsHfield) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <asset>
      <mesh name="m" vertex="0 0 0  1 0 0  0 1 0  0 0 1"/>
      <hfield name="h1" nrow="2" ncol="2" size="1 1 1 1"/>
      <hfield name="h2" nrow="2" ncol="2" size="1 1 1 1"/>
    </asset>
    <worldbody>
      <geom type="mesh" mesh="m" pos="0 0 0"/>
      <geom type="hfield" hfield="h1" pos="10 0 0"/>
      <geom type="hfield" hfield="h2" pos="20 0 0"/>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  EXPECT_EQ(model->nmesh, 1);
  EXPECT_EQ(model->nhfield, 2);

  InitSceneObjects(model.get());
  for (int i = 0; i < mjNVISFLAG; ++i) {
    opt_.flags[i] = 0;
  }
  opt_.flags[mjVIS_MESHBVH] = 1;

  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  int mesh_lineboxes = 0;
  int hfield_lineboxes = 0;
  for (int g = 0; g < scn_.ngeom; ++g) {
    if (scn_.geoms[g].type == mjGEOM_LINEBOX &&
        scn_.geoms[g].category == mjCAT_DECOR) {
      if (scn_.geoms[g].pos[0] > 5.0f) {
        hfield_lineboxes++;
      } else {
        mesh_lineboxes++;
      }
    }
  }
  EXPECT_GT(mesh_lineboxes, 0);
  EXPECT_EQ(hfield_lineboxes, 0);

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, MakeSceneFlexDim1AllocatesZeroFaces) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <flexcomp name="cable" type="direct" dim="1"
                point="0 0 0  1 0 0  2 0 0" element="0 1  1 2"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  ASSERT_EQ(model->nflex, 1);
  ASSERT_EQ(model->flex_dim[0], 1);
  ASSERT_EQ(model->flex_elemnum[0], 2);

  InitSceneObjects(model.get());
  EXPECT_EQ(scn_.flexfacenum[0], 0);
  EXPECT_EQ(scn_.flexface, nullptr);
  FreeSceneObjects();
}

TEST_F(MjvSceneTest, CatenaryTendonLabelsOnlyFirstSegment) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <site name="s0" pos="0 0 1"/>
      <site name="s1" pos="1 0 1"/>
    </worldbody>
    <tendon>
      <spatial name="rope" limited="true" range="0 2.0">
        <site site="s0"/>
        <site site="s1"/>
      </spatial>
    </tendon>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_defaultOption(&opt_);
  opt_.label = mjLABEL_TENDON;
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  int tendon_geoms = 0;
  int labeled_geoms = 0;
  for (int i = 0; i < scn_.ngeom; ++i) {
    const mjvGeom& geom = scn_.geoms[i];
    if (geom.objtype == mjOBJ_TENDON && geom.objid == 0) {
      if (tendon_geoms == 0) {
        EXPECT_NE(geom.label[0], '\0');
      }
      tendon_geoms++;
      if (geom.label[0] != '\0') {
        labeled_geoms++;
      }
    }
  }
  EXPECT_GT(tendon_geoms, 1);
  EXPECT_EQ(labeled_geoms, 1);
  FreeSceneObjects();
}

TEST_F(MjvSceneTest, Model2RoomDoesNotScaleTranslation) {
  mjvScene scn;
  mjv_defaultScene(&scn);
  scn.enabletransform = 1;
  scn.translate[0] = 10.0f;
  scn.translate[1] = -4.0f;
  scn.translate[2] = 6.0f;
  scn.rotate[0] = 1.0f;
  scn.rotate[1] = 0.0f;
  scn.rotate[2] = 0.0f;
  scn.rotate[3] = 0.0f;
  scn.scale = 2.0f;

  mjtNum modelpos[3] = {3.0, 5.0, -2.0};
  mjtNum modelquat[4] = {1.0, 0.0, 0.0, 0.0};
  mjtNum roompos[3] = {0.0, 0.0, 0.0};
  mjtNum roomquat[4] = {1.0, 0.0, 0.0, 0.0};
  mjtNum roundtrip_pos[3] = {0.0, 0.0, 0.0};
  mjtNum roundtrip_quat[4] = {1.0, 0.0, 0.0, 0.0};

  mjv_model2room(roompos, roomquat, modelpos, modelquat, &scn);
  EXPECT_MJTNUM_EQ(roompos[0], static_cast<mjtNum>(10.0 + 2.0 * 3.0));
  EXPECT_MJTNUM_EQ(roompos[1], static_cast<mjtNum>(-4.0 + 2.0 * 5.0));
  EXPECT_MJTNUM_EQ(roompos[2], static_cast<mjtNum>(6.0 + 2.0 * (-2.0)));

  mjv_room2model(roundtrip_pos, roundtrip_quat, roompos, roomquat, &scn);
  EXPECT_MJTNUM_EQ(roundtrip_pos[0], modelpos[0]);
  EXPECT_MJTNUM_EQ(roundtrip_pos[1], modelpos[1]);
  EXPECT_MJTNUM_EQ(roundtrip_pos[2], modelpos[2]);
}

TEST_F(MjvSceneTest, MovePerturbInvalidSelect) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  for (int bad_sel : {0, -1, static_cast<int>(model->nbody)}) {
    mjv_defaultPerturb(&pert_);
    pert_.select = bad_sel;

    mjv_movePerturb(model.get(), data.get(), mjMOUSE_MOVE_V_REL, 0.25, 0.35,
                    &scn_, &pert_);
    EXPECT_MJTNUM_EQ(pert_.refpos[0], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;
    EXPECT_MJTNUM_EQ(pert_.refpos[1], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;
    EXPECT_MJTNUM_EQ(pert_.refpos[2], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;

    mjv_movePerturb(model.get(), data.get(), mjMOUSE_ROTATE_V, 0.25, 0.35,
                    &scn_, &pert_);
    EXPECT_MJTNUM_EQ(pert_.refquat[0], static_cast<mjtNum>(1.0))
        << "bad_sel=" << bad_sel;
    EXPECT_MJTNUM_EQ(pert_.refquat[1], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;
    EXPECT_MJTNUM_EQ(pert_.refquat[2], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;
    EXPECT_MJTNUM_EQ(pert_.refquat[3], static_cast<mjtNum>(0.0))
        << "bad_sel=" << bad_sel;
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, ApplyPerturbForceCombinesTranslateAndRotateTorque) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <body name="box" pos="0 0 1">
        <freejoint/>
        <geom type="box" size="0.2 0.2 0.2" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);
  mjv_defaultPerturb(&pert_);
  pert_.select = 1;
  pert_.localpos[0] = 0.2;
  pert_.localpos[1] = 0.0;
  pert_.localpos[2] = 0.0;

  pert_.active = mjPERT_TRANSLATE;
  pert_.active2 = mjPERT_ROTATE;
  mjv_initPerturb(model.get(), data.get(), &scn_, &pert_);

  pert_.refselpos[1] += 0.5;
  const mjtNum axis[3] = {0, 0, 1};
  mjtNum dquat[4];
  mju_axisAngle2Quat(dquat, axis, 0.2);
  mju_mulQuat(pert_.refquat, dquat, pert_.refquat);

  pert_.active = mjPERT_TRANSLATE;
  pert_.active2 = 0;
  mju_zero(data->xfrc_applied, 6 * model->nbody);
  mjv_applyPerturbForce(model.get(), data.get(), &pert_);
  mjtNum torque_translate[3];
  mju_copy3(torque_translate, data->xfrc_applied + 6 * pert_.select + 3);
  EXPECT_GT(mju_norm3(torque_translate), 1e-3);

  pert_.active = 0;
  pert_.active2 = mjPERT_ROTATE;
  mju_zero(data->xfrc_applied, 6 * model->nbody);
  mjv_applyPerturbForce(model.get(), data.get(), &pert_);
  mjtNum torque_rotate[3];
  mju_copy3(torque_rotate, data->xfrc_applied + 6 * pert_.select + 3);
  EXPECT_GT(mju_norm3(torque_rotate), 1e-3);

  pert_.active = mjPERT_TRANSLATE;
  pert_.active2 = mjPERT_ROTATE;
  mju_zero(data->xfrc_applied, 6 * model->nbody);
  mjv_applyPerturbForce(model.get(), data.get(), &pert_);
  const mjtNum* torque_both = data->xfrc_applied + 6 * pert_.select + 3;
  for (int j = 0; j < 3; ++j) {
    EXPECT_MJTNUM_EQ(torque_both[j], torque_translate[j] + torque_rotate[j])
        << "j=" << j;
  }

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, SelectRespectsCalibratedFrustumWidth) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <camera name="cam" pos="0 0 1" zaxis="0 0 1"
              sensorsize="0.04 0.02" focal="0.01 0.01"/>
      <body name="target_body" pos="1.9 0 0">
        <geom name="target_geom" type="sphere" size="0.05"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  cam_.type = mjCAMERA_FIXED;
  cam_.fixedcamid = 0;
  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);
  ASSERT_GT(scn_.camera[0].frustum_width, 0.0f);

  mjtNum selpnt[3] = {0, 0, 0};
  int geomid = -1;
  int flexid = -1;
  int skinid = -1;
  int bodyid = mjv_select(model.get(), data.get(), &opt_,
                          /*aspectratio=*/1.0, /*relx=*/0.975, /*rely=*/0.5,
                          &scn_, selpnt, &geomid, &flexid, &skinid);
  EXPECT_EQ(bodyid, 1);
  EXPECT_EQ(geomid, 0);

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, MoveCameraTurnVClampedElevationPreservesHeadpos) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="sphere" size="0.1"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());

  mjvCamera cam;
  mjv_defaultCamera(&cam);
  cam.type = mjCAMERA_FREE;
  cam.azimuth = 0.0;
  cam.elevation = 85.0;
  cam.distance = 2.0;

  mjtNum headpos_before[3] = {0, 0, 0};
  mjtNum forward_before[3] = {0, 0, 0};
  mjv_cameraFrame(headpos_before, forward_before, nullptr, nullptr, nullptr,
                  &cam);

  mjv_moveCamera(model.get(), mjMOUSE_TURN_V, 0.0, -0.1, &cam);
  EXPECT_MJTNUM_EQ(cam.elevation, static_cast<mjtNum>(89.0));

  mjtNum headpos_after[3] = {0, 0, 0};
  mjtNum forward_after[3] = {0, 0, 0};
  mjv_cameraFrame(headpos_after, forward_after, nullptr, nullptr, nullptr,
                  &cam);
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(headpos_after[j], headpos_before[j], MjTol(1e-12, 1e-5));
  }
}

TEST_F(MjvSceneTest, CameraOrthographicGovernedByCameraNotModel) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <visual>
      <global orthographic="true" fovy="2"/>
    </visual>
    <worldbody>
      <geom type="sphere" size="0.1"/>
    </worldbody>
  </mujoco>
  )";

  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // 1) mjv_defaultFreeCamera initializes cam.orthographic from
  // model->vis.global.orthographic.
  mjvCamera cam_default;
  mjv_defaultFreeCamera(model.get(), &cam_default);
  EXPECT_EQ(cam_default.orthographic, 1);

  const float expected_ortho_half_height = model->vis.global.fovy / 2;
  const float znear = model->vis.map.znear * model->stat.extent;
  const float expected_persp_half_height =
      znear * mju_tan(model->vis.global.fovy * mjPI / 360.0);

  InitSceneObjects(model.get());

  // Helper lambda to test frustum, gl_camera, and scene camera
  auto verify_camera = [&](const mjvCamera& cam, bool expect_ortho) {
    float zver[2] = {0, 0};
    mjv_cameraFrustum(zver, nullptr, nullptr, model.get(), &cam);
    if (expect_ortho) {
      EXPECT_FLOAT_EQ(zver[0], expected_ortho_half_height);
    } else {
      EXPECT_NEAR(zver[0], expected_persp_half_height, 1e-6);
    }

    mjvGLCamera gl_cam = mjv_camera2GLCamera(model.get(), data.get(), &cam);
    EXPECT_EQ(gl_cam.orthographic, expect_ortho ? 1 : 0);

    mjvCamera cam_copy = cam;
    mjv_updateCamera(model.get(), data.get(), &cam_copy, &scn_);
    EXPECT_EQ(scn_.camera[0].orthographic, expect_ortho ? 1 : 0);
  };

  // Case A: model->vis.global.orthographic = 0, cam.orthographic = 1 (behaves
  // orthographic)
  model->vis.global.orthographic = 0;
  mjvCamera cam_ortho = cam_default;
  cam_ortho.orthographic = 1;
  cam_ortho.azimuth = 0.0;
  cam_ortho.elevation = 0.0;
  cam_ortho.distance = 2.0;
  verify_camera(cam_ortho, /*expect_ortho=*/true);

  // Case B: model->vis.global.orthographic = 1, cam.orthographic = 0 (behaves
  // perspective)
  model->vis.global.orthographic = 1;
  mjvCamera cam_persp = cam_default;
  cam_persp.orthographic = 0;
  cam_persp.azimuth = 0.0;
  cam_persp.elevation = 0.0;
  cam_persp.distance = 2.0;
  verify_camera(cam_persp, /*expect_ortho=*/false);

  // Verify mjv_moveCamera: cam.orthographic controls moveCamera, independent of
  // model
  mjvCamera cam_ortho_m0 = cam_ortho;
  model->vis.global.orthographic = 0;
  mjv_moveCamera(model.get(), mjMOUSE_MOVE_H, 0.1, 0.0, &cam_ortho_m0);

  mjvCamera cam_ortho_m1 = cam_ortho;
  model->vis.global.orthographic = 1;
  mjv_moveCamera(model.get(), mjMOUSE_MOVE_H, 0.1, 0.0, &cam_ortho_m1);

  // Both orthographic moves must produce identical results regardless of model
  // flag
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(cam_ortho_m0.lookat[j], cam_ortho_m1.lookat[j],
                MjTol(1e-12, 1e-5));
  }

  mjvCamera cam_persp_m0 = cam_persp;
  model->vis.global.orthographic = 0;
  mjv_moveCamera(model.get(), mjMOUSE_MOVE_H, 0.1, 0.0, &cam_persp_m0);

  mjvCamera cam_persp_m1 = cam_persp;
  model->vis.global.orthographic = 1;
  mjv_moveCamera(model.get(), mjMOUSE_MOVE_H, 0.1, 0.0, &cam_persp_m1);

  // Both perspective moves must produce identical results regardless of model
  // flag
  for (int j = 0; j < 3; ++j) {
    EXPECT_NEAR(cam_persp_m0.lookat[j], cam_persp_m1.lookat[j],
                MjTol(1e-12, 1e-5));
  }

  // Orthographic and perspective move amounts must differ
  EXPECT_GT(mju_dist3(cam_ortho_m0.lookat, cam_persp_m0.lookat), 1e-3);

  FreeSceneObjects();
}

TEST_F(MjvSceneTest, MjvSelectRespectsFlexAndSkinGroups) {
  static constexpr char kXml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b0" pos="0 0 0"/>
    </worldbody>
    <deformable>
      <flex name="f0" dim="2" group="3" body="b0 b0 b0"
            vertex="-1 -1 0  1 -1 0  0 1 0" element="0 1 2"/>
      <skin name="s0" group="3" vertex="-1 -1 0.5  1 -1 0.5  0 1 0.5" face="0 1 2">
        <bone body="b0" bindpos="0 0 0" bindquat="1 0 0 0" vertid="0 1 2" vertweight="1 1 1"/>
      </skin>
    </deformable>
  </mujoco>
  )";

  char error[1024] = "";
  MjModelPtr model = LoadModelFromString(kXml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  InitSceneObjects(model.get());
  mjv_defaultOption(&opt_);
  ASSERT_EQ(opt_.flexgroup[3], 0);
  ASSERT_EQ(opt_.skingroup[3], 0);

  cam_.type = mjCAMERA_FREE;
  cam_.lookat[0] = 0.0;
  cam_.lookat[1] = 0.0;
  cam_.lookat[2] = 0.0;
  cam_.distance = 2.0;
  cam_.azimuth = 90.0;
  cam_.elevation = -90.0;

  mjv_updateScene(model.get(), data.get(), &opt_, &pert_, &cam_, mjCAT_ALL,
                  &scn_);

  mjtNum selpnt[3] = {0, 0, 0};
  int geomid = -1, flexid = -1, skinid = -1;
  int selbody = mjv_select(model.get(), data.get(), &opt_, 1.0, 0.5, 0.5, &scn_,
                           selpnt, &geomid, &flexid, &skinid);
  EXPECT_EQ(selbody, -1);
  EXPECT_EQ(flexid, -1);
  EXPECT_EQ(skinid, -1);

  FreeSceneObjects();
}
}  // namespace
}  // namespace mujoco
