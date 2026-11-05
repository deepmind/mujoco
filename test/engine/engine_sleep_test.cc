// Copyright 2025 DeepMind Technologies Limited
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

// Tests for engine/engine_sleep.c.

#include "src/engine/engine_sleep.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjplugin.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::string;
using ::std::vector;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

using SleepTest = MujocoTest;

static constexpr char kSimple[] = R"(
<mujoco>
  <option>
    <flag constraint="disable" contact="disable"/>
  </option>

  <default>
    <geom size="1"/>
  </default>

  <worldbody>
    <body>
      <joint type="ball"/>
      <geom/>
    </body>

    <body>
      <geom/>
    </body>

    <geom/>

    <body>
      <joint/>
      <geom/>
      <geom/>
      <body pos="1 0 0">
        <joint/>
        <geom/>
      </body>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr int kAwake = -(1 + mjMINAWAKE);

TEST_F(SleepTest, MjSleepUpdate) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kSimple, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // ntree = 2, nbody = 5, nv = 5, njnt = 3, ngeom = 6
  // body 0: world, 1 geom
  // body 1: 1 ball join, 3 dofs, 1 geom
  // body 2: no joint, 1 geom
  // body 3: hinge joint, 1 dof, 2 geoms
  // body 4: child of body 2, hinge joint, 1 dof, 1 geom

  EXPECT_EQ(m->ntree, 2);
  EXPECT_EQ(m->nbody, 5);
  EXPECT_EQ(m->nv, 5);
  EXPECT_EQ(m->njnt, 3);
  EXPECT_EQ(m->ngeom, 6);

  EXPECT_THAT(AsVector(m->body_treeid, m->nbody), ElementsAre(-1, 0, -1, 1, 1));
  EXPECT_THAT(AsVector(m->dof_bodyid, m->nv), ElementsAre(1, 1, 1, 3, 4));
  EXPECT_THAT(AsVector(m->geom_bodyid, m->ngeom),
              ElementsAre(0, 1, 2, 3, 3, 4));
  EXPECT_THAT(AsVector(m->jnt_bodyid, m->njnt), ElementsAre(1, 3, 4));

  // Test Case 1: Initial state
  EXPECT_THAT(AsVector(d->tree_asleep, m->ntree), ElementsAre(kAwake, kAwake));
  EXPECT_EQ(d->ntree_awake, 2);
  EXPECT_EQ(d->nv_awake, 5);
  EXPECT_THAT(AsVector(d->dof_awake_ind, d->nv_awake),
              ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(AsVector(d->tree_awake, m->ntree), ElementsAre(1, 1));
  EXPECT_THAT(
      AsVector(d->body_awake, m->nbody),
      ElementsAre(mjS_STATIC, mjS_AWAKE, mjS_STATIC, mjS_AWAKE, mjS_AWAKE));

  // Test Case 2: Call mj_sleepUpdate, expect no changes
  mj_updateSleep(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 2);
  EXPECT_EQ(d->nv_awake, 5);
  EXPECT_THAT(AsVector(d->dof_awake_ind, d->nv_awake),
              ElementsAre(0, 1, 2, 3, 4));
  EXPECT_THAT(
      AsVector(d->body_awake, m->nbody),
      ElementsAre(mjS_STATIC, mjS_AWAKE, mjS_STATIC, mjS_AWAKE, mjS_AWAKE));
  EXPECT_THAT(AsVector(d->tree_awake, m->ntree), ElementsAre(1, 1));

  // Test Case 3: Tree 0 asleep
  d->tree_asleep[0] = 0;
  d->tree_asleep[1] = -1;
  mj_updateSleep(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 1);
  EXPECT_EQ(d->nv_awake, 2);
  EXPECT_THAT(AsVector(d->dof_awake_ind, d->nv_awake), ElementsAre(3, 4));
  EXPECT_THAT(AsVector(d->tree_awake, m->ntree), ElementsAre(0, 1));
  EXPECT_THAT(
      AsVector(d->body_awake, m->nbody),
      ElementsAre(mjS_STATIC, mjS_ASLEEP, mjS_STATIC, mjS_AWAKE, mjS_AWAKE));

  // Test Case 4: Tree 1 asleep
  d->tree_asleep[0] = -1;
  d->tree_asleep[1] = 1;
  mj_updateSleep(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 1);
  EXPECT_EQ(d->nv_awake, 3);
  EXPECT_THAT(AsVector(d->dof_awake_ind, d->nv_awake), ElementsAre(0, 1, 2));
  EXPECT_THAT(AsVector(d->tree_awake, m->ntree), ElementsAre(1, 0));
  EXPECT_THAT(
      AsVector(d->body_awake, m->nbody),
      ElementsAre(mjS_STATIC, mjS_AWAKE, mjS_STATIC, mjS_ASLEEP, mjS_ASLEEP));

  // Test Case 5: All trees asleep
  d->tree_asleep[0] = 0;
  d->tree_asleep[1] = 1;
  mj_updateSleep(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 0);
  EXPECT_EQ(d->nv_awake, 0);
  EXPECT_THAT(AsVector(d->dof_awake_ind, d->nv_awake), ElementsAre());
  EXPECT_THAT(AsVector(d->tree_awake, m->ntree), ElementsAre(0, 0));
  EXPECT_THAT(
      AsVector(d->body_awake, m->nbody),
      ElementsAre(mjS_STATIC, mjS_ASLEEP, mjS_STATIC, mjS_ASLEEP, mjS_ASLEEP));
}

TEST_F(SleepTest, MjWakeIsland) {
  // one awake tree and two cycles
  int asleep[] = {kAwake, 2, 1, 3};
  EXPECT_EQ(mj_wakeIsland(asleep, 4, 0, kAwake, nullptr, 0), 0);
  EXPECT_THAT(AsVector(asleep, 4), ElementsAre(kAwake, 2, 1, 3));
  EXPECT_EQ(mj_wakeIsland(asleep, 4, 1, kAwake, nullptr, 0), 2);
  EXPECT_THAT(AsVector(asleep, 4), ElementsAre(kAwake, kAwake, kAwake, 3));
  EXPECT_EQ(mj_wakeIsland(asleep, 4, 3, kAwake, nullptr, 0), 1);
  EXPECT_THAT(AsVector(asleep, 4), ElementsAre(kAwake, kAwake, kAwake, kAwake));
}

TEST_F(SleepTest, BadWakeIsland) {
  EXPECT_FATAL_FAILURE(([] {
                         int asleep_bad1[] = {-1, 0};
                         mj_wakeIsland(asleep_bad1, 2, 1, kAwake, nullptr, 0);
                       }()),
                       "invalid sleep state index -1 when waking tree 1");

  EXPECT_FATAL_FAILURE(([] {
                         int asleep_bad2[] = {-1, 2};
                         mj_wakeIsland(asleep_bad2, 2, 1, kAwake, nullptr, 0);
                       }()),
                       "invalid sleep state index 2 when waking tree 1");

  EXPECT_FATAL_FAILURE(([] {
                         int asleep_bad3[] = {1, 2, 1};
                         mj_wakeIsland(asleep_bad3, 3, 0, kAwake, nullptr, 0);
                       }()),
                       "tree 0 is not in a cycle");
}

static const char* const kStaticModel = "engine/testdata/sleep/static.xml";
static const char* const kMocapcModel = "engine/testdata/sleep/mocap.xml";
static const char* const kSmoothModel = "engine/testdata/sleep/smooth.xml";
static const char* const kInitModel = "engine/testdata/sleep/init.xml";
static const char* const kInitIslandModel =
    "engine/testdata/sleep/init_island.xml";
static const char* const kTendonModel = "engine/testdata/sleep/tendon.xml";
static const char* const kContactModel = "engine/testdata/sleep/contact.xml";
static const char* const kPairModel = "engine/testdata/sleep/contactpair.xml";
static const char* const kSensorModel = "engine/testdata/sleep/sensor.xml";

// roll out some models with sleeping enabled, valuable under ASAN and MSAN
TEST_F(SleepTest, KickTires) {
  for (const char* path :
       {kStaticModel, kMocapcModel, kInitModel, kInitIslandModel, kSensorModel,
        kTendonModel, kContactModel, kPairModel, kSmoothModel}) {
    const std::string xml_path = GetTestDataFilePath(path);
    char error[1024];
    mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
    ASSERT_THAT(m, NotNull()) << error;

    int duration_id = mj_name2id(m, mjOBJ_NUMERIC, "duration");
    ASSERT_GE(duration_id, 0);
    mjtNum duration = m->numeric_data[m->numeric_adr[duration_id]];

    mjData* d = mj_makeData(m);
    while (d->time < duration) {
      mj_step(m, d);
    }

    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// Test that sleeping does not affect the simulation of awake trees:
// Roll out kSmoothModel, where all trees go to sleep within `duration` seconds
// in two mjData's, one with sleeping enabled and one without; expect the same
// values (for selected arrays) in awake trees in both.
TEST_F(SleepTest, WakingUnaffectedBySleeping) {
  const std::string xml_path = GetTestDataFilePath(kSmoothModel);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;

  int duration_id = mj_name2id(m, mjOBJ_NUMERIC, "duration");
  ASSERT_GE(duration_id, 0);
  mjtNum duration = m->numeric_data[m->numeric_adr[duration_id]];

  for (mjtJacobian jacobian : {mjJAC_DENSE, mjJAC_SPARSE}) {
    m->opt.jacobian = jacobian;
    for (mjtIntegrator integrator :  // TODO: b/457674312 - Add support for RK4.
         {mjINT_EULER, mjINT_IMPLICITFAST, mjINT_IMPLICIT, mjINT_DISCRETE}) {
      m->opt.integrator = integrator;

      // make data with sleeping enabled
      m->opt.enableflags |= mjENBL_SLEEP;
      mjData* d_sleep = mj_makeData(m);

      // make data with sleeping disabled
      m->opt.enableflags &= ~mjENBL_SLEEP;
      mjData* d_nosleep = mj_makeData(m);

      // disable constraints, contacts
      m->opt.disableflags |= mjDSBL_CONSTRAINT | mjDSBL_CONTACT;

      ASSERT_EQ(d_sleep->nbody_awake, m->nbody);
      int nbody_awake = -1;

      while (d_nosleep->time < duration) {
        m->opt.enableflags |= mjENBL_SLEEP;
        mj_step(m, d_sleep);
        m->opt.enableflags &= ~mjENBL_SLEEP;
        mj_step(m, d_nosleep);

        // if nbody_awake is not changed, skip
        if (d_sleep->nbody_awake == nbody_awake) {
          continue;
        }

        // compare xpos
        for (int i = 0; i < m->nbody; i++) {
          if (d_sleep->body_awake[i] == mjS_ASLEEP) continue;
          auto xpos1 = AsVector(d_nosleep->xpos + 3 * i, 3);
          auto xpos2 = AsVector(d_sleep->xpos + 3 * i, 3);
          EXPECT_EQ(xpos1, xpos2)
              << " xpos[" << i << "] at time " << d_nosleep->time;
        }

        // compare M and qLD
        for (int i = 0; i < d_sleep->nv_awake; i++) {
          int j = d_sleep->dof_awake_ind[i];
          auto M1 = AsVector(d_nosleep->M + m->M_rowadr[j], m->M_rownnz[j]);
          auto M2 = AsVector(d_sleep->M + m->M_rowadr[j], m->M_rownnz[j]);
          EXPECT_EQ(M1, M2) << " M[" << j << ",:] at time " << d_nosleep->time;
          auto qLD1 = AsVector(d_nosleep->qLD + m->M_rowadr[j], m->M_rownnz[j]);
          auto qLD2 = AsVector(d_sleep->qLD + m->M_rowadr[j], m->M_rownnz[j]);
          EXPECT_EQ(qLD1, qLD2)
              << " qLD[" << j << ",:] at time " << d_nosleep->time;
        }

        // compare cvel
        for (int i = 0; i < d_sleep->nbody_awake; i++) {
          if (d_sleep->body_awake[i] == mjS_ASLEEP) continue;
          auto cvel1 = AsVector(d_nosleep->cvel + 6 * i, 6);
          auto cvel2 = AsVector(d_sleep->cvel + 6 * i, 6);
          EXPECT_EQ(cvel1, cvel2)
              << " cvel[" << i << "] at time " << d_nosleep->time;
        }

        // compare subtree_angmom, only for dynamic bodies
        for (int i = 0; i < d_sleep->nbody_awake; i++) {
          if (d_sleep->body_awake[i] != mjS_AWAKE) continue;
          auto subtree_angmom1 = AsVector(d_nosleep->subtree_angmom + 3 * i, 3);
          auto subtree_angmom2 = AsVector(d_sleep->subtree_angmom + 3 * i, 3);
          EXPECT_EQ(subtree_angmom1, subtree_angmom2)
              << " subtree_angmom[" << i << "] at time " << d_nosleep->time;
        }

        // compare qfrc/qacc arrays
        for (int i = 0; i < d_sleep->nv_awake; i++) {
          int j = d_sleep->dof_awake_ind[i];
          EXPECT_EQ(d_nosleep->qfrc_smooth[j], d_sleep->qfrc_smooth[j])
              << " qfrc_smooth[" << j << "] at time " << d_nosleep->time;
          EXPECT_EQ(d_nosleep->qacc_smooth[j], d_sleep->qacc_smooth[j])
              << " qacc_smooth[" << j << "] at time " << d_nosleep->time;
          EXPECT_EQ(d_nosleep->qacc[j], d_sleep->qacc[j])
              << " qacc[" << j << "] at time " << d_nosleep->time;
        }

        nbody_awake = d_sleep->nbody_awake;
      }

      mj_deleteData(d_sleep);
      mj_deleteData(d_nosleep);
    }
  }
  mj_deleteModel(m);
}

// Test that waking does not affect sleeping trees for pos/vel-dependent arrays.
// Roll out models where some trees wake and/or sleep. At kCompare intervals,
// copy the state from the mjData with sleeping enabled to another mjData and
// call mj_forward with sleeping disabled. Expect pos/vel-dependent arrays to be
// unchanged for all trees and frc/acc-dependent arrays to be the same for awake
// trees.
TEST_F(SleepTest, SleepingUnaffectedByWaking) {
  for (const char* path :
       {kInitModel, kMocapcModel, kInitIslandModel, kTendonModel, kContactModel,
        kSensorModel, kSmoothModel}) {
    const std::string xml_path = GetTestDataFilePath(path);
    char error[1024];
    mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
    ASSERT_THAT(m, NotNull()) << error;

    const int kCompare = 10;  // number of comparisons per rollout

    // for some sensors, the comparison is expected to fail (at least once)
    vector<bool> sensor_mismatch(m->nsensor, false);

    // TODO: b/457674312 - Add support for RK4.
    for (mjtIntegrator integrator :
         {mjINT_EULER, mjINT_IMPLICITFAST, mjINT_IMPLICIT, mjINT_DISCRETE}) {
      m->opt.integrator = integrator;

      // make data with sleeping enabled
      m->opt.enableflags |= mjENBL_SLEEP;
      mjData* d_sleep = mj_makeData(m);

      // make data with sleeping disabled
      m->opt.enableflags &= ~mjENBL_SLEEP;
      mjData* d_nosleep = mj_makeData(m);

      int duration_id = mj_name2id(m, mjOBJ_NUMERIC, "duration");
      ASSERT_GE(duration_id, 0);
      mjtNum duration = m->numeric_data[m->numeric_adr[duration_id]];

      int compare_interval = duration / (m->opt.timestep * kCompare);
      int nsteps = 0;
      while (d_sleep->time < duration) {
        // step d_sleep with sleeping enabled
        m->opt.enableflags |= mjENBL_SLEEP;
        mj_step(m, d_sleep);
        nsteps++;

        // every compare_interval steps, compare with d_nosleep
        if (nsteps % compare_interval != 0) {
          continue;
        }

        // call mj_forward to update d_sleep
        mj_forward(m, d_sleep);

        // copy state from d_sleep to d_nosleep
        mj_copyData(d_nosleep, m, d_sleep);

        // forward d_nosleep with sleeping disabled
        m->opt.enableflags &= ~mjENBL_SLEEP;
        mj_forward(m, d_nosleep);

        // ==== compare arrays for all dofs / bodies / sensors ====

        // compare xpos
        for (int i = 0; i < m->nbody; i++) {
          auto xpos1 = AsVector(d_sleep->xpos + 3 * i, 3);
          auto xpos2 = AsVector(d_nosleep->xpos + 3 * i, 3);
          EXPECT_EQ(xpos1, xpos2)
              << " xpos[" << i << "] at time " << d_sleep->time;
        }

        // compare M and qLD
        for (int i = 0; i < m->nv; i++) {
          auto M1 = AsVector(d_sleep->M + m->M_rowadr[i], m->M_rownnz[i]);
          auto M2 = AsVector(d_nosleep->M + m->M_rowadr[i], m->M_rownnz[i]);
          EXPECT_EQ(M1, M2) << " M[" << i << ",:] at time " << d_sleep->time;
          auto qLD1 = AsVector(d_sleep->qLD + m->M_rowadr[i], m->M_rownnz[i]);
          auto qLD2 = AsVector(d_nosleep->qLD + m->M_rowadr[i], m->M_rownnz[i]);
          EXPECT_EQ(qLD1, qLD2)
              << " qLD[" << i << ",:] at time " << d_sleep->time;
        }

        // compare cvel
        for (int i = 0; i < m->nbody; i++) {
          auto cvel1 = AsVector(d_sleep->cvel + 6 * i, 6);
          auto cvel2 = AsVector(d_nosleep->cvel + 6 * i, 6);
          EXPECT_EQ(cvel1, cvel2)
              << " cvel[" << i << "] at time " << d_sleep->time;
        }

        // compare qfrc arrays
        for (int i = 0; i < m->nv; i++) {
          EXPECT_EQ(d_sleep->qfrc_fluid[i], d_nosleep->qfrc_fluid[i])
              << " qfrc_fluid[" << i << "] at time " << d_sleep->time;
          EXPECT_EQ(d_sleep->qfrc_damper[i], d_nosleep->qfrc_damper[i])
              << " qfrc_damper[" << i << "] at time " << d_sleep->time;
          EXPECT_EQ(d_sleep->qfrc_spring[i], d_nosleep->qfrc_spring[i])
              << " qfrc_spring[" << i << "] at time " << d_sleep->time;
          EXPECT_EQ(d_sleep->qfrc_gravcomp[i], d_nosleep->qfrc_gravcomp[i])
              << " qfrc_gravcomp[" << i << "] at time " << d_sleep->time;
          EXPECT_EQ(d_sleep->qfrc_bias[i], d_nosleep->qfrc_bias[i])
              << " qfrc_bias[" << i << "] at time " << d_sleep->time;
        }

        // compare sensordata
        for (int i = 0; i < m->nsensor; i++) {
          int dim = m->sensor_dim[i];
          int adr = m->sensor_adr[i];
          auto data1 = AsVector(d_sleep->sensordata + adr, dim);
          auto data2 = AsVector(d_nosleep->sensordata + adr, dim);
          if (m->nuser_sensor == 1 && m->sensor_user[i] == 1) {
            // user=1 means sensor value cannot be determined at sleep time
            sensor_mismatch[i] = sensor_mismatch[i] || (data1 != data2);
            EXPECT_EQ(mj_sleepState(m, d_sleep, mjOBJ_SENSOR, i), mjS_AWAKE);
          } else {
            // otherwise expect perfect match
            EXPECT_EQ(data1, data2)
                << " sensor " << i << " at time " << d_sleep->time;
          }
        }

        // ==== compare arrays for awake dofs only ====

        // compare qacc arrays for awake dofs
        for (int j = 0; j < d_sleep->nv_awake; j++) {
          int i = d_sleep->dof_awake_ind[j];
          EXPECT_EQ(d_sleep->qacc_smooth[i], d_nosleep->qacc_smooth[i])
              << " qacc_smooth[" << i << "] at time " << d_sleep->time;
          EXPECT_EQ(d_sleep->qacc[i], d_nosleep->qacc[i])
              << " qacc[" << i << "] at time " << d_sleep->time;
        }
      }

      for (int i = 0; i < m->nsensor; i++) {
        if (m->nuser_sensor == 1 && m->sensor_user[i] == 1) {
          EXPECT_TRUE(sensor_mismatch[i])
              << "contact sensor " << i << " comparison was expected to fail";
        }
      }

      mj_deleteData(d_nosleep);
      mj_deleteData(d_sleep);
    }
    mj_deleteModel(m);
  }
}

static const char* const kEqualityModel = "engine/testdata/sleep/equality.xml";

// Activate equality between sleeping and awake trees, useful under ASAN/MSAN.
TEST_F(SleepTest, Equality) {
  const std::string xml_path = GetTestDataFilePath(kEqualityModel);
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;

  mjData* d = mj_makeData(m);
  while (d->ntree_awake == m->ntree) {
    mj_step(m, d);
  }

  int dd = mj_name2id(m, mjOBJ_EQUALITY, "dyn/dyn");
  ASSERT_GE(dd, 0);

  mj_step(m, d);
  d->eq_active[dd] = 1;
  mj_step(m, d);

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test that the free-body implicit (gyroscopic) solve doesn't break the sleep
// qvel=0 invariant. A standalone free body with high viscosity should
// eventually go to sleep, and after sleeping, qvel/qacc must be exactly zero.
TEST_F(SleepTest, FreeBodySleepZeroVelocity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" viscosity="10"
            sleep_tolerance="0.01">
      <flag sleep="enable" gravity="disable" constraint="disable"
            contact="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1" euler="10 20 30"
              pos=".03 .02 .01"/>
      </body>
    </worldbody>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // give initial velocity (both translational and angular)
  d->qvel[0] = 0.5;
  d->qvel[1] = 0.5;
  d->qvel[2] = 0.5;
  d->qvel[3] = 1.0;
  d->qvel[4] = 2.0;
  d->qvel[5] = 3.0;

  // step until body goes to sleep
  for (int step = 0; step < 1000; step++) {
    mj_step(m.get(), d.get());
    if (d->ntree_awake == 0) break;
  }

  // body should have gone to sleep
  ASSERT_EQ(d->ntree_awake, 0) << "body did not go to sleep";

  // qvel and qacc must be exactly zero for sleeping body
  for (int i = 0; i < 6; i++) {
    EXPECT_EQ(d->qvel[i], 0.0) << "qvel[" << i << "] not zero after sleep";
    EXPECT_EQ(d->qacc[i], 0.0) << "qacc[" << i << "] not zero after sleep";
  }
}

static const char* const kInitIslandFailModel =
    "engine/testdata/sleep/init_island_fail.xml";

// trees marked sleep='init' that cannot be slept compile with a warning and
// start awake
TEST_F(SleepTest, InitIslandFail) {
  const std::string xml_path = GetTestDataFilePath(kInitIslandFailModel);
  mock_warning_handler.ExpectWarnings(
      "3 trees were marked as sleep='init' but only 0 could be slept. Body "
      "'asleep_init0' (id=1) is the root of the first tree that could not be "
      "slept.");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);
  ASSERT_THAT(d, NotNull());
  for (int i = 0; i < m->ntree; i++) {
    EXPECT_LT(d->tree_asleep[i], 0) << i;
  }
  mj_deleteData(d);
  mj_deleteModel(m);
}

// applied force keeps a tree awake
void PushControl(const mjModel* m, mjData* d) { d->qfrc_applied[0] = 1; }

// MuJoCo allocations not yet freed, counted by CountingMalloc and CountingFree
std::atomic<int> live_allocations = 0;

void* CountingMalloc(std::size_t size) {
  live_allocations++;
  return std::malloc(size);
}

void CountingFree(void* block) {
  live_allocations--;
  std::free(block);
}

// passive plugin whose instance data is allocated by init and read by compute
int RegisterDataPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.sleepdata";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nstate = +[](const mjModel* m, int instance) { return 1; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    d->plugin_data[instance] =
        reinterpret_cast<uintptr_t>(mju_malloc(sizeof(mjtNum)));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    mju_free(reinterpret_cast<void*>(d->plugin_data[instance]));
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) { *static_cast<mjtNum*>(plugin_data) = 0; };
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int type) {
    *reinterpret_cast<mjtNum*>(d->plugin_data[instance]) += 1;
  };
  return mjp_registerPlugin(&plugin);
}

// a reset that cannot put sleep='init' trees to sleep warns and leaves them
// awake, allocating and freeing nothing
TEST_F(SleepTest, InitFailureWarnsAndLeavesTreesAwake) {
  static const int plugin_slot = RegisterDataPlugin();
  ASSERT_GE(plugin_slot, 0);
  static constexpr char xml[] = R"(
  <mujoco>
    <extension>
      <plugin plugin="mujoco.test.sleepdata">
        <instance name="data"/>
      </plugin>
    </extension>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body sleep="init">
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  mju_user_malloc = CountingMalloc;
  mju_user_free = CountingFree;
  mjData* d = mj_makeData(model.get());
  int allocated = live_allocations;
  EXPECT_GE(d->tree_asleep[0], 0);
  mock_warning_handler.ExpectWarnings("could be slept");
  mjcb_control = PushControl;
  mj_resetData(model.get(), d);
  EXPECT_LT(d->tree_asleep[0], 0);
  EXPECT_EQ(live_allocations, allocated);
  mj_resetDataKeyframe(model.get(), d, -1);
  EXPECT_LT(d->tree_asleep[0], 0);
  EXPECT_EQ(live_allocations, allocated);
  mjData* d2 = mj_makeData(model.get());
  EXPECT_THAT(d2, NotNull());
  if (d2) {
    EXPECT_LT(d2->tree_asleep[0], 0);
    mj_step(model.get(), d2);
    mj_deleteData(d2);
  }
  EXPECT_EQ(live_allocations, allocated);
  mjcb_control = nullptr;
  mj_deleteData(d);
  EXPECT_EQ(live_allocations, 0);
  mju_user_malloc = nullptr;
  mju_user_free = nullptr;
}

// Test that a constrained flex eventually goes to sleep.
TEST_F(SleepTest, FlexEdgeSleep) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_edge.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  EXPECT_GT(m->flex_edgeequality[0], 0);
  mjData* d = mj_makeData(m);

  // give initial velocity
  d->qvel[0] = 0.1;

  // step until body goes to sleep
  for (int step = 0; step < 2000; step++) {
    mj_step(m, d);
    if (d->ntree_awake == 0) break;
  }

  EXPECT_EQ(d->ntree_awake, 0) << "flex did not go to sleep";

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test that a constraint-free flex never sleeps.
TEST_F(SleepTest, FlexNoConstraintNeverSleeps) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_nocnstr.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  EXPECT_EQ(m->flex_edgeequality[0], 0);
  mjData* d = mj_makeData(m);

  // step for a while
  for (int step = 0; step < 500; step++) {
    mj_step(m, d);
  }

  // tree should have AUTO_NEVER policy
  EXPECT_EQ(m->tree_sleep_policy[0], mjSLEEP_AUTO_NEVER);
  EXPECT_GT(d->ntree_awake, 0) << "constraint-free flex should not sleep";

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test that mj_sleepState returns correct values for flex objects.
TEST_F(SleepTest, FlexSleepState) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_state.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  // initially awake
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, 0), mjS_AWAKE);

  // step until flex goes to sleep
  for (int step = 0; step < 1000; step++) {
    mj_step(m, d);
    if (d->ntree_awake == 0) break;
  }

  // flex should be asleep
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, 0), mjS_ASLEEP);

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test that a sleeping flex wakes on contact with a falling ball,
// then both go back to sleep after the ball rolls off.
TEST_F(SleepTest, FlexWakeContact) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_contact.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  int flexid = mj_name2id(m, mjOBJ_FLEX, "f1");
  ASSERT_GE(flexid, 0);

  // phase 1: flex settles and goes to sleep
  for (int step = 0; step < 5000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, flexid) == mjS_ASLEEP) break;
  }
  ASSERT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, flexid), mjS_ASLEEP)
      << "flex did not go to sleep";

  // phase 2: ball hits flex, flex wakes up
  for (int step = 0; step < 5000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, flexid) == mjS_AWAKE) break;
  }
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, flexid), mjS_AWAKE)
      << "flex should have been woken by ball contact";

  // phase 3: ball rolls off, everything goes back to sleep
  for (int step = 0; step < 10000; step++) {
    mj_step(m, d);
    if (d->ntree_awake == 0) break;
  }
  EXPECT_EQ(d->ntree_awake, 0) << "all trees should be asleep again";

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test full sleep/wake lifecycle with two grippers and two flex objects.
TEST_F(SleepTest, HollowVsSolidSleep) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/hollow_vs_solid.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  // look up flex IDs
  int solid_flex = mj_name2id(m, mjOBJ_FLEX, "soft_mesh");
  int hollow_flex = mj_name2id(m, mjOBJ_FLEX, "soft_mesh_2");
  ASSERT_GE(solid_flex, 0);
  ASSERT_GE(hollow_flex, 0);

  // look up actuator IDs
  int grasp_r = mj_name2id(m, mjOBJ_ACTUATOR, "grasp_r");
  int grasp_s = mj_name2id(m, mjOBJ_ACTUATOR, "grasp_s");
  ASSERT_GE(grasp_r, 0);
  ASSERT_GE(grasp_s, 0);

  // phase 1: everything starts awake
  mj_forward(m, d);
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, solid_flex), mjS_AWAKE);
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex), mjS_AWAKE);

  // phase 2: both flexes go to sleep within 500 steps
  for (int step = 0; step < 500; step++) {
    mj_step(m, d);
  }
  ASSERT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, solid_flex), mjS_ASLEEP)
      << "solid flex did not go to sleep";
  ASSERT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex), mjS_ASLEEP)
      << "hollow flex did not go to sleep";

  // phase 3: close gripper_solid (grasp_r), it wakes soft_mesh_2 (hollow)
  d->ctrl[grasp_r] = 1;
  for (int step = 0; step < 2000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex) == mjS_AWAKE) break;
  }
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex), mjS_AWAKE)
      << "hollow flex should be woken by gripper_solid";
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, solid_flex), mjS_ASLEEP)
      << "solid flex should still be asleep";

  // phase 4: close gripper_hollow (grasp_s), it wakes soft_mesh (solid)
  d->ctrl[grasp_s] = 1;
  for (int step = 0; step < 2000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, solid_flex) == mjS_AWAKE) break;
  }
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, solid_flex), mjS_AWAKE)
      << "solid flex should be woken by gripper_hollow";

  // phase 5: open gripper_solid (grasp_r=0), hollow flex goes back to sleep
  d->ctrl[grasp_r] = 0;
  for (int step = 0; step < 2000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex) == mjS_ASLEEP) break;
  }
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, hollow_flex), mjS_ASLEEP)
      << "hollow flex should go back to sleep after gripper release";

  // phase 6: open gripper_hollow (grasp_s=0), solid flex goes back to sleep
  d->ctrl[grasp_s] = 0;
  for (int step = 0; step < 2000; step++) {
    mj_step(m, d);
    if (mj_sleepState(m, d, mjOBJ_FLEX, solid_flex) == mjS_ASLEEP) break;
  }
  EXPECT_EQ(mj_sleepState(m, d, mjOBJ_FLEX, solid_flex), mjS_ASLEEP)
      << "solid flex should go back to sleep after gripper release";

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test sleep/wake lifecycle with elastic flexes under the discrete integrator
// (both CG and Newton solvers).
TEST_F(SleepTest, FlexSleepDiscrete) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" viscosity="2" sleep_tolerance="0.1">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                radius=".01" dim="2" mass=".3" pos="0 0 .01" dof="full">
        <contact selfcollide="none"/>
        <edge equality="true" solref="0.02 1"/>
        <elasticity young="1e3" poisson="0.2" damping="0.2" elastic2d="bend" thickness="0.01"/>
      </flexcomp>
      <flexcomp name="hollow" type="grid" count="2 2 2" spacing=".05 .05 .05"
                radius=".01" dim="3" mass=".4" pos="0.5 0 .035" dof="trilinear">
        <contact selfcollide="none"/>
        <edge equality="strain" solref="0.02 1"/>
      </flexcomp>
      <body name="ball" pos="0 0.01 0.5">
        <freejoint/>
        <geom type="sphere" size=".03" mass=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  int cloth_id = mj_name2id(m.get(), mjOBJ_FLEX, "cloth");
  int hollow_id = mj_name2id(m.get(), mjOBJ_FLEX, "hollow");
  ASSERT_GE(cloth_id, 0);
  ASSERT_GE(hollow_id, 0);

  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    m->opt.solver = solver;
    MjDataPtr d = MakeData(m);

    // phase 1: cloth and hollow flexes settle onto the plane and fall asleep
    // before the ball reaches them
    for (int step = 0; step < 500; step++) {
      mj_step(m.get(), d.get());
      if (mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id) == mjS_ASLEEP &&
          mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, hollow_id) ==
              mjS_ASLEEP) {
        break;
      }
    }
    ASSERT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id), mjS_ASLEEP)
        << "solver " << solver;
    ASSERT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, hollow_id),
              mjS_ASLEEP)
        << "solver " << solver;

    // phase 2: falling ball impacts cloth and wakes it while hollow stays
    // asleep
    for (int step = 0; step < 2000; step++) {
      mj_step(m.get(), d.get());
      if (mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id) == mjS_AWAKE) {
        break;
      }
    }
    EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id), mjS_AWAKE)
        << "solver " << solver;
    EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, hollow_id),
              mjS_ASLEEP)
        << "solver " << solver;

    // phase 3: everything settles back to sleep
    for (int step = 0; step < 5000; step++) {
      mj_step(m.get(), d.get());
      if (d->ntree_awake == 0) {
        break;
      }
    }
    EXPECT_EQ(d->ntree_awake, 0) << "solver " << solver;
  }
}

// Test sleep/wake lifecycle with purely passive elastic flexes (no equality
// constraints, both non-interpolated and trilinear interpolated, including a
// pinned node 0 and a zero-contact floating flex) under the discrete
// integrator.
TEST_F(SleepTest, PassiveFlexSleepDiscrete) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" viscosity="2" sleep_tolerance="0.1">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <flexcomp name="passive_cloth" type="grid" count="3 3 1" spacing=".05 .05 .05"
                radius=".01" dim="2" mass=".3" pos="0 0 .01" dof="full">
        <pin id="0"/>
        <contact selfcollide="none"/>
        <elasticity young="1e3" poisson="0.2" damping="0.2" elastic2d="both" thickness="0.01"/>
      </flexcomp>
      <flexcomp name="passive_solid" type="grid" count="2 2 2" spacing=".05 .05 .05"
                radius=".01" dim="3" mass=".4" pos="0.5 0 .06" dof="trilinear">
        <pin id="0"/>
        <contact selfcollide="none"/>
        <elasticity young="2e3" poisson="0.2" damping="0.2"/>
      </flexcomp>
      <body name="ball_cloth" pos="0 0.01 0.5">
        <freejoint/>
        <geom type="sphere" size=".03" mass=".1"/>
      </body>
      <body name="ball_solid" pos="0.52 0.02 1.2">
        <freejoint/>
        <geom type="sphere" size=".03" mass=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  int cloth_id = mj_name2id(m.get(), mjOBJ_FLEX, "passive_cloth");
  int solid_id = mj_name2id(m.get(), mjOBJ_FLEX, "passive_solid");
  ASSERT_GE(cloth_id, 0);
  ASSERT_GE(solid_id, 0);
  EXPECT_EQ(m->flex_edgeequality[cloth_id], 0);
  EXPECT_EQ(m->flex_edgeequality[solid_id], 0);

  for (mjtSolver solver : {mjSOL_CG, mjSOL_NEWTON}) {
    m->opt.solver = solver;
    MjDataPtr d = MakeData(m);

    // phase 1: both passive flexes settle and fall asleep before the balls
    // reach them
    for (int step = 0; step < 500; step++) {
      mj_step(m.get(), d.get());
      if (mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id) == mjS_ASLEEP &&
          mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, solid_id) == mjS_ASLEEP) {
        break;
      }
    }
    ASSERT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id), mjS_ASLEEP)
        << "solver " << solver;
    ASSERT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, solid_id), mjS_ASLEEP)
        << "solver " << solver;

    // phase 2: ball_cloth impacts passive_cloth and wakes all of its trees
    // while passive_solid stays asleep
    for (int step = 0; step < 2000; step++) {
      mj_step(m.get(), d.get());
      if (mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id) == mjS_AWAKE) {
        break;
      }
    }
    EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, cloth_id), mjS_AWAKE)
        << "solver " << solver;
    EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, solid_id), mjS_ASLEEP)
        << "solver " << solver;

    // phase 3: ball_solid impacts passive_solid (whose node 0 is pinned) and
    // wakes all of its dynamic node trees
    for (int step = 0; step < 2000; step++) {
      mj_step(m.get(), d.get());
      if (mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, solid_id) == mjS_AWAKE) {
        break;
      }
    }
    EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, solid_id), mjS_AWAKE)
        << "solver " << solver;

    // phase 4: everything settles back to sleep
    for (int step = 0; step < 5000; step++) {
      mj_step(m.get(), d.get());
      if (d->ntree_awake == 0) {
        break;
      }
    }
    EXPECT_EQ(d->ntree_awake, 0) << "solver " << solver;
  }

  // also verify a zero-contact floating passive flex sleeps and wakes as one
  // cycle when a single vertex is perturbed
  static constexpr char free_xml[] = R"(
  <mujoco>
    <option timestep="0.002" integrator="discrete" viscosity="5" sleep_tolerance="0.05">
      <flag sleep="enable" gravity="disable" contact="disable"/>
    </option>
    <worldbody>
      <flexcomp name="free_cloth" type="grid" count="2 2 1" spacing=".05 .05 .05"
                radius=".01" dim="2" mass=".2" dof="full">
        <elasticity young="1e3" poisson="0.2" damping="0.2" elastic2d="both" thickness="0.01"/>
      </flexcomp>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr mf = LoadModelFromString(free_xml, error, sizeof(error));
  ASSERT_THAT(mf.get(), NotNull()) << error;
  MjDataPtr df = MakeData(mf);
  df->qvel[0] = 0.01;
  for (int step = 0; step < 500; step++) {
    mj_step(mf.get(), df.get());
    if (df->ntree_awake == 0) {
      break;
    }
  }
  ASSERT_EQ(df->ntree_awake, 0);
  // Switching to a non-metric integrator at runtime wakes the passive flex and
  // prevents individual vertices from sleeping without the metric union
  mf->opt.integrator = mjINT_EULER;
  for (int step = 0; step < 200; step++) {
    mj_step(mf.get(), df.get());
  }
  EXPECT_EQ(df->ntree_awake, mf->ntree);

  // Switching back to discrete allows the passive flex to sleep again as one
  // cycle, and perturbing one vertex wakes all 4 vertex trees
  mf->opt.integrator = mjINT_DISCRETE;
  for (int step = 0; step < 500; step++) {
    mj_step(mf.get(), df.get());
    if (df->ntree_awake == 0) {
      break;
    }
  }
  ASSERT_EQ(df->ntree_awake, 0);
  df->qvel[0] = 1.0;
  mj_step(mf.get(), df.get());
  EXPECT_EQ(df->ntree_awake, mf->ntree);
}

// Test that a sleeping flex touching a non-world static body doesn't crash.
TEST_F(SleepTest, FlexStaticContact) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_static.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  // step until flex goes to sleep, should not throw
  for (int step = 0; step < 1000; step++) {
    mj_step(m, d);
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}

// Test that a sleeping flex near a mocap body doesn't crash.
TEST_F(SleepTest, FlexMocapContact) {
  const std::string xml_path =
      GetTestDataFilePath("engine/testdata/sleep/flex_mocap.xml");
  char error[1024];
  mjModel* m = mj_loadXML(xml_path.c_str(), 0, error, sizeof(error));
  ASSERT_THAT(m, NotNull()) << error;
  mjData* d = mj_makeData(m);

  for (int step = 0; step < 200; step++) {
    mj_step(m, d);
  }

  mj_deleteData(d);
  mj_deleteModel(m);
}

// mocap bodies are their own weld root: dragging into a sleeping tree wakes it
TEST_F(SleepTest, MocapWakes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body name="hand" mocap="true" pos="2 0 .1">
        <geom type="sphere" size=".1"/>
        <body name="finger">
          <geom type="sphere" size=".05" pos=".1 0 0"/>
        </body>
      </body>
      <body name="box" pos="0 0 .1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  // mocap body is its own weld root, static child inherits it,
  // box is its own (jointed)
  int hand = mj_name2id(m.get(), mjOBJ_BODY, "hand");
  int finger = mj_name2id(m.get(), mjOBJ_BODY, "finger");
  EXPECT_EQ(m->body_weldid[hand], hand);
  EXPECT_EQ(m->body_weldid[finger], hand);

  // let the box fall asleep
  MjDataPtr d = MakeData(m);
  int steps = 0;
  while (d->ntree_awake > 0 && steps++ < 10000) {
    mj_step(m.get(), d.get());
  }
  ASSERT_EQ(d->ntree_awake, 0);

  // drag the mocap hand into the sleeping box: box wakes
  d->mocap_pos[0] = 0;
  mj_step(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 1);
}

// mocap bodies do not generate contacts with static bodies or each other
TEST_F(SleepTest, MocapNoStaticContacts) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body mocap="true" pos="0 0 .05">
        <geom type="sphere" size=".1"/>
      </body>
      <body mocap="true" pos=".05 0 .05">
        <geom type="sphere" size=".1"/>
      </body>
      <geom type="box" size=".1 .1 .1" pos="-.05 0 .05"/>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // everything interpenetrates, nothing can move: no contacts
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->ncon, 0);
}

// a plane on a mocap body collides like a plane on a static body
TEST_F(SleepTest, MocapPlane) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body mocap="true">
        <geom type="plane" size="5 5 .1"/>
      </body>
      <body pos="0 0 .5">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // box falls onto the mocap plane and rests on it
  while (d->time < 2) {
    mj_step(m.get(), d.get());
  }
  EXPECT_GT(d->ncon, 0);
  EXPECT_NEAR(d->qpos[2], 0.1, 1e-3);
}

// a tree welded by equality to a mocap body is woken when asleep
TEST_F(SleepTest, MocapWeldEqualityWakes) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom type="plane" size="5 5 .1"/>
      <body name="target" mocap="true" pos="2 0 .5">
        <geom type="sphere" size=".05" contype="0" conaffinity="0"/>
      </body>
      <body name="box" pos="0 0 .1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
    <equality>
      <weld name="grab" body1="target" body2="box" active="false"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // equality inactive: box falls asleep on the floor
  int steps = 0;
  while (d->ntree_awake > 0 && steps++ < 10000) {
    mj_step(m.get(), d.get());
  }
  ASSERT_EQ(d->ntree_awake, 0);

  // activate the weld to the mocap body: box wakes
  d->eq_active[mj_name2id(m.get(), mjOBJ_EQUALITY, "grab")] = 1;
  mj_step(m.get(), d.get());
  EXPECT_EQ(d->ntree_awake, 1);
}

// plugin whose init runs kinematics before mj_resetData, like the cable plugin
// state: xpos of the last body, computed at init
void RegisterKinematicsPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.kinematics";
  plugin.nstate = +[](const mjModel* m, int instance) { return 3; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    mju_copy(d->qpos, m->qpos0, m->nq);
    mj_kinematics(m, d);
    mju_copy3(d->plugin_state + m->plugin_stateadr[instance],
              d->xpos + 3 * (m->nbody - 1));
    return 0;
  };
  mjp_registerPlugin(&plugin);
}

// plugin init sees all trees awake and static bodies posed
TEST_F(SleepTest, PluginInitKinematics) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <extension>
      <plugin plugin="mujoco.test.kinematics">
        <instance name="kinematics"/>
      </plugin>
    </extension>
    <worldbody>
      <body>
        <joint/>
        <geom size=".1"/>
      </body>
      <body pos="0 0 1">
        <body pos="1 0 0">
          <joint/>
          <geom size=".1"/>
        </body>
      </body>
    </worldbody>
  </mujoco>
  )";
  RegisterKinematicsPlugin();
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;

  // zero-filled buffers: uninitialized sleep state would put both trees in one
  // sleep cycle through tree 0; mjData buffers must be 64-byte aligned
  mju_user_malloc = +[](size_t size) {
    void* ptr = ::operator new(size, std::align_val_t(64));
    std::memset(ptr, 0, size);
    return ptr;
  };
  mju_user_free =
      +[](void* ptr) { ::operator delete(ptr, std::align_val_t(64)); };
  mjData* d = mj_makeData(m.get());
  mj_forward(m.get(), d);
  EXPECT_EQ(AsVector(d->plugin_state, 3),
            AsVector(d->xpos + 3 * (m->nbody - 1), 3));
  mj_deleteData(d);
  mju_user_malloc = nullptr;
  mju_user_free = nullptr;
}

// tactile sensors reference mjOBJ_MESH without aborting in mj_sleepState
TEST_F(SleepTest, TactileSensorSleepState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <asset>
      <mesh name="sensor_mesh"
            vertex="0 0 0  1 0 0  0 1 0  0 0 1"
            face="0 2 1  0 1 3  0 3 2  1 2 3"/>
    </asset>
    <worldbody>
      <geom name="static_sensor_geom" type="mesh" mesh="sensor_mesh"/>
      <body name="awake_collider" pos="0.1 0.1 0.05">
        <freejoint/>
        <geom type="sphere" size="0.2"/>
      </body>
      <body name="sleeping_body" pos="5 0 1" sleep="init">
        <freejoint/>
        <geom name="sleeping_sensor_geom" type="mesh" mesh="sensor_mesh"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="static_sensor_geom" mesh="sensor_mesh"/>
      <tactile geom="sleeping_sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_forward(m.get(), d.get());
  EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_MESH, 0), mjS_STATIC);
  EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_SENSOR, 0), mjS_AWAKE);
  EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_SENSOR, 1), mjS_ASLEEP);

  // static tactile sensor still detects penetration from awake_collider
  MjDataPtr d_nosleep = MakeData(m);
  m->opt.enableflags &= ~mjENBL_SLEEP;
  mj_forward(m.get(), d_nosleep.get());
  for (int k = 0; k < m->sensor_dim[0]; k++) {
    EXPECT_EQ(d->sensordata[k], d_nosleep->sensordata[k]);
  }
}

// 2-tree tendon with frictionloss wakes sleeping tree when coupled tree moves
TEST_F(SleepTest, WakeTendonFrictionloss) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.01">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body name="b0" pos="0 0 0">
        <joint name="j0" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
      <body name="b1" pos="0 1 0">
        <joint name="j1" type="slide" axis="1 0 0"/>
        <geom size="0.1" mass="1"/>
      </body>
    </worldbody>
    <tendon>
      <fixed name="t" frictionloss="5.0">
        <joint joint="j0" coef="1"/>
        <joint joint="j1" coef="-1"/>
      </fixed>
    </tendon>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m_sleep = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m_sleep.get(), NotNull()) << error;
  MjDataPtr d_sleep = MakeData(m_sleep);
  mj_forward(m_sleep.get(), d_sleep.get());
  d_sleep->tree_asleep[1] = 1;
  mj_updateSleep(m_sleep.get(), d_sleep.get());
  ASSERT_EQ(d_sleep->tree_awake[1], 0);

  MjModelPtr m_nosleep = LoadModelFromString(xml, error, sizeof(error));
  m_nosleep->opt.enableflags &= ~mjENBL_SLEEP;
  MjDataPtr d_nosleep = MakeData(m_nosleep);

  d_sleep->qvel[0] = 2.0;
  d_nosleep->qvel[0] = 2.0;

  mj_step(m_sleep.get(), d_sleep.get());
  mj_step(m_nosleep.get(), d_nosleep.get());

  EXPECT_EQ(d_sleep->tree_awake[1], 1);
  EXPECT_NEAR(d_sleep->qvel[0], d_nosleep->qvel[0], MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d_sleep->qvel[1], d_nosleep->qvel[1], MjTol(1e-12, 1e-5));
}

// flex equality wakes all sleeping trees, and mj_sleepState(mjOBJ_FLEX) checks
// all vertices
TEST_F(SleepTest, WakeEqualityFlexMultipleSleepingTreesAndSleepState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.01">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body name="v0" pos="0 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="v1" pos="1 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      </body>
      <body name="v2" pos="2 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.05" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
    <deformable>
      <flex name="chain" dim="1" body="v0 v1 v2" element="0 1 1 2"/>
    </deformable>
    <equality>
      <flex flex="chain"/>
    </equality>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_forward(m.get(), d.get());
  d->tree_asleep[0] = 0;
  d->tree_asleep[2] = 2;
  mj_updateSleep(m.get(), d.get());

  // before mj_wakeEquality: v0 and v2 are asleep, v1 is awake -> flex must
  // report AWAKE
  EXPECT_EQ(d->tree_awake[0], 0);
  EXPECT_EQ(d->tree_awake[1], 1);
  EXPECT_EQ(d->tree_awake[2], 0);
  EXPECT_EQ(mj_sleepState(m.get(), d.get(), mjOBJ_FLEX, 0), mjS_AWAKE);

  // mj_forward triggers mj_wakeEquality, which must wake BOTH v0 and v2
  d->qvel[1] = 1.0;
  mj_step(m.get(), d.get());
  EXPECT_EQ(d->tree_awake[0], 1);
  EXPECT_EQ(d->tree_awake[1], 1);
  EXPECT_EQ(d->tree_awake[2], 1);
}

// site-transmission actuator with refsite checks sleep state of both site and
// refsite
TEST_F(SleepTest, SiteActuatorRefsiteSleepState) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0" timestep="0.01">
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <site name="world_site1" pos="0 0 0"/>
      <site name="world_site2" pos="0 1 0"/>
      <body name="moving_ref" pos="0 0 0">
        <joint name="j_ref" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="ref_site" pos="0 0 0"/>
      </body>
      <body name="sleeper" pos="3 0 0" sleep="init">
        <freejoint/>
        <geom type="sphere" size="0.1" mass="1"/>
        <site name="sleep_site" pos="0 0 0"/>
      </body>
    </worldbody>
    <actuator>
      <general site="world_site1" refsite="ref_site" gear="1 0 0 0 0 0"/>
      <general site="world_site1" refsite="sleep_site" gear="1 0 0 0 0 0"/>
      <general site="world_site1" refsite="world_site2" gear="1 0 0 0 0 0"/>
    </actuator>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d_sleep = MakeData(m);

  d_sleep->qpos[0] = 0.5;
  d_sleep->qvel[0] = 1.2;
  d_sleep->ctrl[0] = 3.0;
  mj_forward(m.get(), d_sleep.get());

  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_ACTUATOR, 0),
            mjS_AWAKE);
  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_ACTUATOR, 1),
            mjS_ASLEEP);
  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_ACTUATOR, 2),
            mjS_STATIC);

  MjDataPtr d_nosleep = MakeData(m);
  mj_copyData(d_nosleep.get(), m.get(), d_sleep.get());
  m->opt.enableflags &= ~mjENBL_SLEEP;
  mj_forward(m.get(), d_nosleep.get());

  EXPECT_NEAR(d_sleep->actuator_length[0], d_nosleep->actuator_length[0],
              MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d_sleep->actuator_velocity[0], d_nosleep->actuator_velocity[0],
              MjTol(1e-12, 1e-5));
  EXPECT_NEAR(d_sleep->qfrc_actuator[0], d_nosleep->qfrc_actuator[0],
              MjTol(1e-12, 1e-5));
}

// subtree contact and insidesite sensors on a static parent body remain awake
// when a descendant kinematic tree is awake
TEST_F(SleepTest, SubtreeSensorStaticParentWithAwakeChild) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <geom name="floor" type="plane" size="5 5 .1"/>
      <site name="zone" type="box" size=".5 .5 .5" pos="0 0 .1"/>
      <body name="container" pos="0 0 1">
        <body name="ball" pos="0 0 0">
          <joint type="slide" axis="0 0 1"/>
          <geom type="sphere" size=".15"/>
        </body>
      </body>
      <body name="sleeper" pos="3 0 1" sleep="init">
        <freejoint/>
        <geom type="sphere" size=".1"/>
      </body>
    </worldbody>
    <sensor>
      <contact subtree1="container" geom2="floor"/>
      <insidesite site="zone" objtype="body" objname="container"/>
      <insidesite site="zone" objtype="xbody" objname="container" enclosed="true"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d_sleep = MakeData(m);
  ASSERT_EQ(d_sleep->ntree_awake, 1);
  ASSERT_LT(d_sleep->nbody_awake, m->nbody);

  // move awake child ball down into the site and into contact with the floor
  d_sleep->qpos[0] = -0.9;
  mj_forward(m.get(), d_sleep.get());

  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_SENSOR, 0), mjS_AWAKE);
  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_SENSOR, 1), mjS_AWAKE);
  EXPECT_EQ(mj_sleepState(m.get(), d_sleep.get(), mjOBJ_SENSOR, 2), mjS_AWAKE);

  MjDataPtr d_nosleep = MakeData(m);
  mj_copyData(d_nosleep.get(), m.get(), d_sleep.get());
  m->opt.enableflags &= ~mjENBL_SLEEP;
  mj_forward(m.get(), d_nosleep.get());

  EXPECT_EQ(d_nosleep->sensordata[0], 1);
  EXPECT_EQ(d_nosleep->sensordata[1], 1);
  EXPECT_EQ(d_sleep->sensordata[0], d_nosleep->sensordata[0]);
  EXPECT_EQ(d_sleep->sensordata[1], d_nosleep->sensordata[1]);
  EXPECT_NEAR(d_sleep->sensordata[2], d_nosleep->sensordata[2],
              MjTol(1e-12, 1e-5));
}

// mjTOPIC_SLEEP debug logging safely clamps buffer position when >300 trees
// sleep and wake
TEST_F(SleepTest, SleepTopicLoggingTruncatesSafelyWithManyTrees) {
  constexpr int kNumTrees = 350;
  string xml =
      R"(<mujoco><option gravity="0 0 0"><flag sleep="enable"/></option><worldbody>)";
  for (int i = 0; i < kNumTrees; i++) {
    xml += "<body pos=\"" + std::to_string(i) +
           " 0 0\">"
           "<joint name=\"j" +
           std::to_string(i) +
           "\" type=\"slide\" axis=\"1 0 0\"/>"
           "<geom type=\"sphere\" size=\"0.05\" mass=\"1\" contype=\"0\" "
           "conaffinity=\"0\"/>"
           "</body>";
  }
  xml += "</worldbody><equality>";
  for (int i = 0; i + 1 < kNumTrees; i++) {
    xml += "<joint joint1=\"j" + std::to_string(i) + "\" joint2=\"j" +
           std::to_string(i + 1) + "\"/>";
  }
  xml += "</equality></mujoco>";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml.c_str(), error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  static vector<string>* subjects = nullptr;
  vector<string> captured;
  subjects = &captured;

  mjLogConfig orig_cfg = mju_getLogConfig();
  mjLogConfig cfg = orig_cfg;
  cfg.logto_console = false;
  cfg.logto_file = false;
  cfg.topics = (1 << (mjTOPIC_SLEEP - 1));
  mju_setLogConfig(cfg);
  mjfLogHandler prev_handler = mju_setLogHandler(+[](const mjLogMessage* msg) {
    if (subjects && msg->topic == mjTOPIC_SLEEP) {
      subjects->emplace_back(msg->subject);
    }
  });

  // mark all 350 coupled trees ready to sleep (-1); step 1 sleeps the 350-tree
  // island via mj_sleepTrees, step 2 wakes it via mj_wakeIsland
  for (int i = 0; i < kNumTrees; i++) {
    d->tree_asleep[i] = -1;
  }
  mj_step(m.get(), d.get());
  int ntree_after_sleep = d->ntree_awake;

  d->qvel[0] = 1.0;
  mj_step(m.get(), d.get());
  int ntree_after_wake = d->ntree_awake;

  mju_setLogHandler(prev_handler);
  mju_setLogConfig(orig_cfg);
  subjects = nullptr;

  EXPECT_EQ(ntree_after_sleep, 0);
  EXPECT_EQ(ntree_after_wake, kNumTrees);
#ifndef MJ_DISABLE_DEBUG_TRACING
  ASSERT_EQ(captured.size(), 2);
  EXPECT_THAT(captured[0], HasSubstr("slept trees 0 1 2"));
  EXPECT_THAT(captured[1], HasSubstr("woke due to perturbation trees 0 1 2"));
  EXPECT_LT(captured[0].size(), 1024u);
  EXPECT_LT(captured[1].size(), 1024u);
#endif
}

}  // namespace
}  // namespace mujoco
