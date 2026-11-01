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

// Tests for engine/engine_sensor.c.

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjtype.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_support.h"
#include "src/engine/engine_util_blas.h"
#include "src/engine/engine_util_spatial.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::std::string;
using ::std::vector;

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::Not;
using ::testing::NotNull;
using ::testing::Pointwise;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::testing::WhenSorted;

const mjtNum tol = 1e-14;  // nearness tolerance for floating point numbers

// returns as a vector the measured values from sensor with index `id`
static vector<mjtNum> GetSensor(const mjModel* model, const mjData* data,
                                int id) {
  return vector<mjtNum>(
      data->sensordata + model->sensor_adr[id],
      data->sensordata + model->sensor_adr[id] + model->sensor_dim[id]);
}

// returns as a vector the measured values from sensor with name `name
static vector<mjtNum> GetSensor(const mjModel* model, const mjData* data,
                                const char* name) {
  int id = mj_name2id(model, mjOBJ_SENSOR, name);
  return vector<mjtNum>(
      data->sensordata + model->sensor_adr[id],
      data->sensordata + model->sensor_adr[id] + model->sensor_dim[id]);
}

using SensorTest = MujocoTest;

// --------------------- test sensor disable flag ------------------------------

TEST_F(SensorTest, DisableSensors) {
  constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <clock/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // call mj_forward, expect clock to report 0
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->sensordata[0], 0.0);

  // call mj_step, mj_step1, expect clock to be incremented by timestep
  mj_step(model.get(), data.get());
  mj_step1(model.get(), data.get());
  EXPECT_EQ(data->sensordata[0], model->opt.timestep);

  // disable sensors, call mj_step, mj_step1, expect clock to not increment
  model->opt.disableflags |= mjDSBL_SENSOR;
  mj_step(model.get(), data.get());
  mj_step1(model.get(), data.get());
  EXPECT_EQ(data->time, 2 * model->opt.timestep);
  EXPECT_EQ(data->sensordata[0], model->opt.timestep);

  // re-enable sensors, call mj_step, mj_step1, expect clock to match time
  model->opt.disableflags = 0;
  mj_step(model.get(), data.get());
  mj_step1(model.get(), data.get());
  EXPECT_EQ(data->time, data->sensordata[0]);
}

// --------------------- test relative frame sensors  --------------------------

using RelativeFrameSensorTest = MujocoTest;

// hand-picked positions and orientations for simple expected values
TEST_F(RelativeFrameSensorTest, ReferencePosMat) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="reference" pos="3 -4 0" xyaxes="4 3 0 -3 4 0"/>
      <site name="object" pos="4 3 0" xyaxes="3 -4 0 4 3 0"/>
    </worldbody>
    <sensor>
      <framepos objtype="site" objname="object"
                reftype="xbody" refname="reference"/>
      <framexaxis objtype="site" objname="object"
                  reftype="xbody" refname="reference"/>
      <frameyaxis objtype="site" objname="object"
                  reftype="xbody" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // compare actual and expected values
  vector pos = GetSensor(model.get(), data.get(), 0);
  EXPECT_THAT(pos, Pointwise(MjNear(tol, 1e-6), {5, 5, 0}));

  vector xaxis = GetSensor(model.get(), data.get(), 1);
  EXPECT_THAT(xaxis, Pointwise(MjNear(tol, 1e-6), {0, -1, 0}));

  vector yaxis = GetSensor(model.get(), data.get(), 2);
  EXPECT_THAT(yaxis, Pointwise(MjNear(tol, 1e-6), {1, 0, 0}));
}

// orientations given by quaternion and by orientation matrix are identical
TEST_F(RelativeFrameSensorTest, ReferenceQuatMat) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="reference" euler="10 20 30"/>
      <site name="object" euler="20 40 60"/>
    </worldbody>
    <sensor>
      <framexaxis objtype="site" objname="object"
                  reftype="site" refname="reference"/>
      <frameyaxis objtype="site" objname="object"
                  reftype="site" refname="reference"/>
      <framezaxis objtype="site" objname="object"
                  reftype="site" refname="reference"/>
      <framequat objtype="site" objname="object"
                  reftype="site" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // call mj_forward and convert orientation matrix to quaternion
  mj_forward(model.get(), data.get());
  mjtNum mat[9], converted_quat[4];
  mju_transpose(mat, data->sensordata, 3, 3);
  mju_mat2Quat(converted_quat, mat);

  // compare quaternion sensor and quat derived from orientation matrix
  vector quat = GetSensor(model.get(), data.get(), 3);
  EXPECT_THAT(quat, Pointwise(MjNear(tol, 1e-6), converted_quat));
}

// compare global frame and initially co-located relative frame on same body
TEST_F(RelativeFrameSensorTest, ReferencePosMatQuat) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <freejoint/>
        <site name="reference"/>
        <geom name="object" euler="20 40 60" pos="1 2 3" size="1"/>
      </body>
    </worldbody>
    <sensor>
      <framepos objtype="geom" objname="object"/>
      <framexaxis objtype="geom" objname="object"/>
      <frameyaxis objtype="geom" objname="object"/>
      <framezaxis objtype="geom" objname="object"/>
      <framequat objtype="geom" objname="object"/>
      <framepos objtype="geom" objname="object"
                reftype="site" refname="reference"/>
      <framexaxis objtype="geom" objname="object"
                  reftype="site" refname="reference"/>
      <frameyaxis objtype="geom" objname="object"
                  reftype="site" refname="reference"/>
      <framezaxis objtype="geom" objname="object"
                  reftype="site" refname="reference"/>
      <framequat objtype="geom" objname="object"
                  reftype="site" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  constexpr int nsensordata = 32;
  ASSERT_EQ(model->nsensordata, nsensordata);
  MjDataPtr data = MakeData(model);

  // call mj_forward, save global sensors (colocated with reference frame)
  mj_forward(model.get(), data.get());
  vector expected_values(data->sensordata, data->sensordata + nsensordata / 2);

  // set qpos to arbitrary values, call mj_forward
  for (int i = 0; i < 7; i++) {
    data->qpos[i] = i + 1;
  }
  mj_forward(model.get(), data.get());

  // get values from relative sensors after moving the object
  vector actual_values(data->sensordata + nsensordata / 2,
                       data->sensordata + nsensordata);

  // object and reference have moved together, we expect values to not change
  EXPECT_THAT(actual_values, Pointwise(MjNear(tol, 1e-6), expected_values));
}

// hand-picked velocities and orientations for simple expected values
TEST_F(RelativeFrameSensorTest, FrameVelLinearFixed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body xyaxes="1 -1 0 1 1 0">
        <joint type="slide" axis="1 0 0"/>
        <geom name="reference" size="1"/>
      </body>
      <body>
        <joint type="slide" axis="1 0 0"/>
        <geom name="object" size="1"/>
      </body>
    </worldbody>
    <sensor>
      <framelinvel objtype="geom" objname="object"
                   reftype="geom" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  data->qvel[0] = mju_sqrt(2);
  data->qvel[1] = 1;
  mj_forward(model.get(), data.get());

  // compare to expected values
  vector linvel = GetSensor(model.get(), data.get(), 0);
  const mjtNum expected_linvel[3] = {-mju_sqrt(0.5), mju_sqrt(0.5), 0};
  EXPECT_THAT(linvel, Pointwise(MjNear(tol, 1e-6), expected_linvel));
}

// object and reference in the same body, expect angular velocities to be zero
TEST_F(RelativeFrameSensorTest, FrameVelAngFixed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint type="hinge" axis="1 2 3"/>
        <geom name="reference" size="1" pos="1 2 3"/>
        <geom name="object" size="1" pos="-3 -2 -1"/>
      </body>
    </worldbody>
    <sensor>
      <frameangvel objtype="geom" objname="object"
                   reftype="geom" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // set joint velocities and call forward dynamics
  data->qvel[0] = 1;
  mj_forward(model.get(), data.get());

  // obj and ref rotate together, relative angular velocities should be zero
  vector angvel = GetSensor(model.get(), data.get(), 0);
  EXPECT_THAT(angvel, Pointwise(MjNear(tol, 1e-6), {0, 0, 0}));
}

// object and reference rotate on the same global axis
TEST_F(RelativeFrameSensorTest, FrameVelAngOpposing) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body xyaxes="0 -1 0 1 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom name="reference" size="1"/>
      </body>
      <body>
        <joint type="hinge" axis="1 0 0"/>
        <geom name="object" size="1" pos="-3 -2 -1"/>
      </body>
    </worldbody>
    <sensor>
      <frameangvel objtype="geom" objname="object"
                   reftype="geom" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // set joint velocities and call forward dynamics
  data->qvel[0] = -1;
  data->qvel[1] = 1;
  mj_forward(model.get(), data.get());

  // obj and ref rotate on same axis, we can just difference the velocities
  vector angvel = GetSensor(model.get(), data.get(), 0);
  const mjtNum expected_angvel[3] = {0, data->qvel[1] - data->qvel[0], 0};
  EXPECT_THAT(angvel, Pointwise(MjNear(tol, 1e-6), expected_angvel));
}

// two arbitrary frames, compare velocity sensors and fin-diffed positions
TEST_F(RelativeFrameSensorTest, FrameVelGeneral) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1 2 3" euler="10 20 30">
        <joint type="hinge" axis="2 3 4"/>
        <geom name="reference" size="1" pos="0 1 2"/>
      </body>
      <body pos="-3 -2 -1" euler="20 40 60">
        <joint type="hinge" axis="2 3 4"/>
        <geom name="object" size="1" pos="1 2 3"/>
      </body>
    </worldbody>
    <sensor>
      <framepos objtype="geom" objname="object"
                reftype="geom" refname="reference"/>
      <framequat objtype="geom" objname="object"
                 reftype="geom" refname="reference"/>
      <framelinvel objtype="geom" objname="object"
                   reftype="geom" refname="reference"/>
      <frameangvel objtype="geom" objname="object"
                   reftype="geom" refname="reference"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  mjtNum dt = MjTol(1e-6, 1e-3);  // timestep used for finite differencing

  // set (arbitrary) joint velocities and call forward dynamics
  data->qvel[0] = 1;
  data->qvel[1] = -1;
  mj_forward(model.get(), data.get());

  // save measured linear and angular velocities as vectors
  vector linvel = GetSensor(model.get(), data.get(), 2);
  vector angvel = GetSensor(model.get(), data.get(), 3);

  // save current position, quaternion as arrays
  mjtNum pos0[3], quat0[4];
  mju_copy3(pos0, data->sensordata);
  mju_copy4(quat0, data->sensordata + 3);

  // explicit Euler integration with small dt
  mju_addToScl(data->qpos, data->qvel, dt, 2);

  // call mj_forward again, save new position and quaternion
  mj_forward(model.get(), data.get());
  mjtNum pos1[3], quat1[4];
  mju_copy3(pos1, data->sensordata);
  mju_copy4(quat1, data->sensordata + 3);

  // compute expected linear velocities using finite differencing
  mjtNum linvel_findiff[3];
  mju_sub3(linvel_findiff, pos1, pos0);
  mju_scl3(linvel_findiff, linvel_findiff, 1 / dt);

  // compute expected angular velocities using finite differencing
  mjtNum dquat[4], angvel_findiff[3];
  mju_negQuat(quat0, quat0);
  mju_mulQuat(dquat, quat1, quat0);
  mju_quat2Vel(angvel_findiff, dquat, dt);

  // compare analytic and finite-differenced relative velocities
  EXPECT_THAT(linvel, Pointwise(MjNear(10 * dt, 1.5e-2), linvel_findiff));
  EXPECT_THAT(angvel, Pointwise(MjNear(10 * dt, 2e-3), angvel_findiff));
}

// ------------------------- general sensor tests  -----------------------------
using SensorTest = MujocoTest;

TEST_F(SensorTest, EnableEnergy) {
  constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -5">
     <flag energy="enable"/>
    </option>
    <worldbody>
      <body pos="0 0 2">
        <geom size="1" mass="3"/>
        <freejoint/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->energy[0], 2 * 3 * 5, MjTol(1e-12, 1e-5));

  model->opt.enableflags &= ~mjENBL_ENERGY;
  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->energy[0], 0, MjTol(1e-12, 1e-5));
}

TEST_F(SensorTest, PotentialEnergy) {
  constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -5"/>
    <worldbody>
      <body pos="0 0 2">
        <geom size="1" mass="3"/>
        <freejoint/>
      </body>
    </worldbody>
    <sensor>
      <e_potential/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 2 * 3 * 5, MjTol(1e-12, 1e-5));

  data->qpos[2] = 7;
  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 7 * 3 * 5, MjTol(1e-12, 1e-5));
}

TEST_F(SensorTest, PotentialEnergyFreeJointSpring) {
  constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body>
        <geom size="1" mass="3"/>
        <joint type="free" stiffness="2"/>
      </body>
    </worldbody>
    <sensor>
      <e_potential/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);
  data->qpos[0] = 1;
  data->qpos[1] = 2;
  data->qpos[2] = 3;
  mj_forward(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 0.5 * 2 * 14, MjTol(1e-12, 1e-5));
}

TEST_F(SensorTest, KineticEnergy) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 2">
        <geom size="1" mass="3"/>
        <freejoint/>
      </body>
    </worldbody>
    <sensor>
      <e_kinetic/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  while (data->time < 1.5) {
    mj_step(model.get(), data.get());
  }
  mj_forward(model.get(), data.get());

  mjtNum mass = 3;
  mjtNum speed = data->time * mju_norm3(model->opt.gravity);
  EXPECT_NEAR(data->sensordata[0], 0.5 * mass * speed * speed,
              MjTol(1e-7, 1e-2));
}

TEST_F(SensorTest, PolyStiffnessEnergy) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.0001">
      <flag energy="enable"/>
    </option>

    <worldbody>
      <body>
        <joint type="slide" stiffness="10 5 1"/>
        <geom size="1" mass="1"/>
      </body>
    </worldbody>

    <keyframe>
      <key qpos="2"/>
    </keyframe>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  mj_resetDataKeyframe(m.get(), d.get(), 0);

  mj_forward(m.get(), d.get());
  mjtNum total_energy = d->energy[0] + d->energy[1];

  for (int i = 0; i < 100; i++) {
    mj_step(m.get(), d.get());
    EXPECT_NEAR(d->energy[0] + d->energy[1], total_energy, 0.003);
  }
}

// test clock sensor
TEST_F(SensorTest, Clock) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1e-3"/>
    <sensor>
      <clock/>
      <clock name="clampedclock" cutoff="3e-3"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // call step 4 times, checking that clock works as expected
  for (int i = 0; i < 5; i++) {
    mj_step(model.get(), data.get());
    mj_step1(model.get(),
             data.get());  // update values of position-based sensors
    EXPECT_EQ(data->sensordata[0], data->time);
    EXPECT_EQ(data->sensordata[1], mju_min(data->time, 3e-3));
  }

  // check names
  const char* name0 = mj_id2name(model.get(), mjOBJ_SENSOR, 0);
  EXPECT_EQ(name0, nullptr);
  const char* name1 = mj_id2name(model.get(), mjOBJ_SENSOR, 1);
  EXPECT_THAT(name1, StrEq("clampedclock"));
}

// test that integer parameters pass through
TEST_F(SensorTest, IntPrm) {
  constexpr char xml[] = R"(
  <mujoco>
    <sensor>
      <clock name="dummy"/>
    </sensor>
  </mujoco>
  )";
  ASSERT_EQ(mjNSENS, 3);

  char err[1024];
  mjSpec* spec = mj_parseXMLString(xml, 0, err, sizeof(err));
  ASSERT_THAT(spec, NotNull()) << err;

  mjModel* model = mj_compile(spec, nullptr);
  EXPECT_EQ(model->sensor_intprm[0], 0);
  EXPECT_EQ(model->sensor_intprm[1], 0);
  mj_deleteModel(model);

  mjsSensor* s = mjs_asSensor(mjs_findElement(spec, mjOBJ_SENSOR, "dummy"));
  s->intprm[0] = 3;
  s->intprm[1] = 4;
  s->intprm[2] = 5;
  model = mj_compile(spec, nullptr);
  EXPECT_EQ(model->sensor_intprm[0], 3);
  EXPECT_EQ(model->sensor_intprm[1], 4);
  EXPECT_EQ(model->sensor_intprm[2], 5);
  mj_deleteModel(model);
  mj_deleteSpec(spec);
}

// test sequential collision sensors
TEST_F(SensorTest, CollisionSequential) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="plane" type="plane" size="1 1 1"/>
      <geom name="sphere1" pos="0 0 1" size="0.2"/>
      <geom name="sphere2" pos="1 0 1" size="0.3"/>
    </worldbody>
    <sensor>
      <distance name="0"  geom1="plane"   geom2="sphere1" cutoff="1"/>
      <distance name="1"  geom1="sphere2" geom2="plane" cutoff="1"/>
      <distance name="2"  geom1="sphere1" geom2="sphere2" cutoff="1"/>
      <normal   name="3"  geom1="plane"   geom2="sphere1" cutoff="1"/>
      <normal   name="4"  geom1="sphere2" geom2="plane" cutoff="1"/>
      <normal   name="5"  geom1="sphere1" geom2="sphere2" cutoff="1"/>
      <fromto   name="6"  geom1="plane"   geom2="sphere1" cutoff="1"/>
      <fromto   name="7"  geom1="sphere2" geom2="plane" cutoff="1"/>
      <fromto   name="8"  geom1="sphere1" geom2="sphere2" cutoff="1"/>
      <!-- sequential sensors with identical signature -->
      <distance name="9"  geom1="plane"   geom2="sphere1" cutoff="1"/>
      <fromto   name="10" geom1="plane"   geom2="sphere1" cutoff="1"/>
      <normal   name="11" geom1="plane"   geom2="sphere1" cutoff="1"/>
      <normal   name="12"  geom1="sphere1" geom2="sphere2" cutoff="1"/>
      <fromto   name="13"  geom1="sphere1" geom2="sphere2" cutoff="1"/>
      <distance name="14"  geom1="sphere1" geom2="sphere2" cutoff="1"/>

    </sensor>
  </mujoco>
  )";
  MjModelPtr model_ptr = LoadModelFromString(xml);
  MjDataPtr data_ptr = MakeData(model_ptr);

  mjModel* model = model_ptr.get();
  mjData* data = data_ptr.get();
  mj_forward(model, data);

  mjtNum eps = 1e-14;
  EXPECT_NEAR(data->sensordata[0], 0.8, MjTol(eps, 1e-7));
  EXPECT_NEAR(data->sensordata[1], 0.7, MjTol(eps, 1e-7));
  EXPECT_NEAR(data->sensordata[2], 0.5, MjTol(eps, 1e-7));

  EXPECT_THAT(GetSensor(model, data, 3),
              Pointwise(MjNear(eps, 1e-7), vector<mjtNum>{0, 0, 1}));
  EXPECT_THAT(GetSensor(model, data, 4),
              Pointwise(MjNear(eps, 1e-7), vector<mjtNum>{0, 0, -1}));
  EXPECT_THAT(GetSensor(model, data, 5),
              Pointwise(MjNear(eps, 1e-7), vector<mjtNum>{1, 0, 0}));
  EXPECT_THAT(GetSensor(model, data, 6),
              Pointwise(MjNear(eps, 1e-6), vector<mjtNum>{0, 0, 0, 0, 0, .8}));
  EXPECT_THAT(GetSensor(model, data, 7),
              Pointwise(MjNear(eps, 1e-6), vector<mjtNum>{1, 0, .7, 1, 0, 0}));
  EXPECT_THAT(GetSensor(model, data, 8),
              Pointwise(MjNear(eps, 1e-6), vector<mjtNum>{.2, 0, 1, .7, 0, 1}));

  EXPECT_THAT(GetSensor(model, data, 9),
              Pointwise(MjNear(eps, 1e-7), GetSensor(model, data, 0)));
  EXPECT_THAT(GetSensor(model, data, 10),
              Pointwise(MjNear(eps, 1e-6), GetSensor(model, data, 6)));
  EXPECT_THAT(GetSensor(model, data, 11),
              Pointwise(MjNear(eps, 1e-7), GetSensor(model, data, 3)));
  EXPECT_THAT(GetSensor(model, data, 12),
              Pointwise(MjNear(eps, 1e-7), GetSensor(model, data, 5)));
  EXPECT_THAT(GetSensor(model, data, 13),
              Pointwise(MjNear(eps, 1e-6), GetSensor(model, data, 8)));
  EXPECT_THAT(GetSensor(model, data, 14),
              Pointwise(MjNear(eps, 1e-7), GetSensor(model, data, 2)));
}

TEST_F(SensorTest, BadContact) {
  string xml_template = R"(
  <mujoco>
    <worldbody>
      <geom name="sphere1" pos="0 0 1" size="0.2"/>
      <body name="body">
        <freejoint/>
        <geom name="sphere2" pos="1 0 1" size="0.3"/>
        <site name="site" pos="1 0 1" size="0.3"/>
        <body name="non_root">
          <geom name="sphere3" pos="1 0 1" size="0.3"/>
        </body>
      </body>
    </worldbody>
    <sensor>
      <contact BAD_ATTR/>
    </sensor>
  </mujoco>
  )";

  struct Case {
    string bad_attr;
    string expected_error;
  };

  Case test_cases[] = {
      {"geom1='sphere1' geom2='sphere2' data='dist force normal'",
       "must be in order: found, force, torque, dist, pos, normal, tangent, "
       "linvel, angvel"},
      {"geom1='sphere1' geom2='sphere2' num='-3'",
       "'num' must be positive in sensor"},
      {"geom1='sphere1' geom2='sphere2' site='site'",
       "at most one of 'geom1', 'body1', 'subtree1', 'site' can be specified"},
      {"geom2='sphere1' body2='body'",
       "at most one of 'geom2', 'body2', 'subtree2' can be specified"},
      {"geom1='sphere1' geom2='sphere2' reduce='netforce' data='linvel'",
       "netforce reduction is incompatible with linvel or angvel"},
  };

  for (const auto& test : test_cases) {
    string xml = xml_template;
    size_t pos = xml.find("BAD_ATTR");
    ASSERT_NE(pos, string::npos);
    xml.replace(pos, 8, test.bad_attr);

    char error[1024];
    MjModelPtr model = LoadModelFromString(xml.c_str(), error, sizeof(error));
    ASSERT_THAT(model.get(), IsNull()) << "Test case: " << test.bad_attr;
    EXPECT_THAT(error, HasSubstr(test.expected_error))
        << "Test case: " << test.bad_attr;
  }
}

TEST_F(SensorTest, Contact) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/contact.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  mjData* data = mj_makeData(model);

  for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
    model->opt.cone = cone;

    mj_resetData(model, data);
    while (data->time < 2) {
      mj_step(model, data);
    }

    vector all = GetSensor(model, data, "all");
    EXPECT_EQ(all, vector<mjtNum>{4});

    vector world = GetSensor(model, data, "world");
    EXPECT_EQ(world, vector<mjtNum>{3});

    vector b1 = GetSensor(model, data, "b1");
    EXPECT_EQ(b1, vector<mjtNum>{3});

    vector g1 = GetSensor(model, data, "g1");
    EXPECT_EQ(g1, vector<mjtNum>{3});

    vector b1g2 = GetSensor(model, data, "b1:g2");
    EXPECT_EQ(b1g2, vector<mjtNum>{1});

    vector b1world = GetSensor(model, data, "b1:world");
    EXPECT_EQ(b1world, vector<mjtNum>{2});

    vector site = GetSensor(model, data, "site");
    EXPECT_EQ(site, vector<mjtNum>{2});

    vector sitewall = GetSensor(model, data, "site:wall");
    EXPECT_EQ(sitewall, vector<mjtNum>{1});

    vector wall = GetSensor(model, data, "wall");
    EXPECT_THAT(wall, Pointwise(MjNear(1e-4, 0.02),
                                {1, 8, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0}));

    // normals points *away* from b2 (towards floor / b1)
    vector b2 = GetSensor(model, data, "b2");
    EXPECT_THAT(b2, Pointwise(MjNear(1e-4, 0.02),
                              {3, 0, 0, 0, 0, -1, 4, 0, 0, 1, 0, 0}));

    // normal points *towards* b2
    vector b2f = GetSensor(model, data, "b2_flipped");
    EXPECT_THAT(b2f, Pointwise(MjNear(1e-4, 0.02),
                               {3, 0, 0, 0, 0, 1, 4, 0, 0, -1, 0, 0}));

    vector b2r = GetSensor(model, data, "b2_reduced");
    EXPECT_THAT(b2r, Pointwise(MjNear(1e-4, 0.02), {4, 0, 0, -1, 0, 0}));
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(SensorTest, ContactSorted) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/contact_sorted.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  mjData* data = mj_makeData(model);
  while (data->time < .5) {
    mj_step(model, data);
  }

  vector unsorted = GetSensor(model, data, "unsorted");
  EXPECT_THAT(unsorted, SizeIs(4));
  EXPECT_THAT(unsorted, Not(WhenSorted(ElementsAreArray(unsorted))));

  vector sorted = GetSensor(model, data, "sorted dist");
  EXPECT_THAT(sorted, SizeIs(4));
  EXPECT_THAT(sorted, WhenSorted(ElementsAreArray(sorted)));

  vector sorted_force = GetSensor(model, data, "sorted force");
  EXPECT_THAT(sorted_force, SizeIs(12));
  vector<mjtNum> nnorms;
  for (size_t i = 0; i < sorted_force.size(); i += 3) {
    nnorms.push_back(-sorted_force[i] * sorted_force[i] +
                     -sorted_force[i + 1] * sorted_force[i + 1] +
                     -sorted_force[i + 2] * sorted_force[i + 2]);
  }
  EXPECT_THAT(nnorms, WhenSorted(ElementsAreArray(nnorms)));

  vector smallest = GetSensor(model, data, "smallest dist");
  EXPECT_THAT(smallest, SizeIs(1));
  EXPECT_EQ(smallest[0], sorted[0]);

  vector largest = GetSensor(model, data, "largest force");
  EXPECT_THAT(largest, SizeIs(3));
  EXPECT_THAT(largest,
              ElementsAre(sorted_force[0], sorted_force[1], sorted_force[2]));

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(SensorTest, ContactSubtree) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/contact_subtree.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  mjData* data = mj_makeData(model);

  while (data->time < 0.2) {
    mj_step(model, data);

    int all = GetSensor(model, data, "all")[0];
    int w_t1 = GetSensor(model, data, "w_t1")[0];
    int w_t2 = GetSensor(model, data, "w_t2")[0];
    int t1 = GetSensor(model, data, "t1")[0];
    int t2 = GetSensor(model, data, "t2")[0];
    int t1_t1 = GetSensor(model, data, "t1_t1")[0];
    int t2_t2 = GetSensor(model, data, "t2_t2")[0];
    int t1_t2 = GetSensor(model, data, "t1_t2")[0];
    int t2_t1 = GetSensor(model, data, "t2_t1")[0];

    // compute the number of first tree contacts in two different ways
    EXPECT_EQ(t1, w_t1 + t1_t1 + t1_t2);

    // compute the number of second tree contacts in two different ways
    EXPECT_EQ(t2, w_t2 + t2_t2 + t2_t1);

    // compute the number of all contacts in two different ways
    EXPECT_EQ(all, w_t1 + w_t2 + t1_t1 + t2_t2 + t1_t2);
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(SensorTest, ContactSubtreePartial) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/contact_subtree_partial.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  mjData* data = mj_makeData(model);

  while (data->time < 0.6) {
    mj_step(model, data);
  }

  EXPECT_EQ(GetSensor(model, data, "all")[0], 4);
  EXPECT_EQ(GetSensor(model, data, "world")[0], 4);
  EXPECT_EQ(GetSensor(model, data, "thigh")[0], 4);
  EXPECT_EQ(GetSensor(model, data, "shin")[0], 2);
  EXPECT_EQ(GetSensor(model, data, "foot")[0], 1);
  EXPECT_EQ(GetSensor(model, data, "foot_w")[0], 0);
  EXPECT_EQ(GetSensor(model, data, "foot_w2")[0], 1);

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(SensorTest, ContactNet) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/contact_net.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;
  int b1 = mj_name2id(model, mjOBJ_BODY, "b1");
  int b2 = mj_name2id(model, mjOBJ_BODY, "b2");
  int nv = model->nv;

  mjData* data = mj_makeData(model);

  for (mjtCone cone : {mjCONE_PYRAMIDAL, mjCONE_ELLIPTIC}) {
    model->opt.cone = cone;
    mj_resetData(model, data);

    // for each timestep, compare the net force computation to qfrc_constraint
    // data->ncon varies in [0, 6]
    int nconmax = 0;
    while (data->time < 0.2) {
      mj_step(model, data);
      vector<mjtNum> qfrc_expected = AsVector(data->qfrc_constraint, nv);

      // check net force, sensor returns body1 -> body2
      vector net12 = GetSensor(model, data, "net12");
      EXPECT_EQ(net12.size(), 9);
      mjtNum* force = net12.data();
      mjtNum* torque = net12.data() + 3;
      mjtNum* point = net12.data() + 6;

      // apply wrench to b2
      vector<mjtNum> qfrc(nv, 0.0);
      mj_applyFT(model, data, force, torque, point, b2, qfrc.data());

      // apply opposite wrench to b1
      mju_scl3(force, force, -1);
      mju_scl3(torque, torque, -1);
      mj_applyFT(model, data, force, torque, point, b1, qfrc.data());

      // compare
      EXPECT_THAT(qfrc, Pointwise(MjNear(1e-6, 2e-4), qfrc_expected));

      // check net force, sensor returns body2 -> body1
      vector net21 = GetSensor(model, data, "net21");
      EXPECT_EQ(net21.size(), 9);
      force = net21.data();
      torque = net21.data() + 3;
      point = net21.data() + 6;
      qfrc.assign(nv, 0.0);

      // apply wrench to b1
      mj_applyFT(model, data, force, torque, point, b1, qfrc.data());

      // apply opposite wrench to b2
      mju_scl3(force, force, -1);
      mju_scl3(torque, torque, -1);
      mj_applyFT(model, data, force, torque, point, b2, qfrc.data());

      // compare
      EXPECT_THAT(qfrc, Pointwise(MjNear(1e-6, 2e-4), qfrc_expected));

      nconmax = std::max(nconmax, data->ncon);
    }

    // at least 5 contacts happened
    EXPECT_GT(nconmax, 4);
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST_F(SensorTest, ContactNetEmpty) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="b1" pos="0 0 1">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
      <body name="b2" pos="0 0 5">
        <freejoint/>
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <contact body1="b1" body2="b2" reduce="netforce"
               data="found force torque dist pos normal tangent"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);

  mj_forward(model.get(), data.get());
  ASSERT_EQ(data->ncon, 0);
  EXPECT_THAT(GetSensor(model.get(), data.get(), 0),
              ElementsAre(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0));
}

TEST_F(SensorTest, ContactStage) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 1"/>
      <body name="b" pos="0 0 0.08">
        <freejoint/>
        <geom name="sphere" type="sphere" size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <contact name="pos_found" data="found dist pos normal tangent" reduce="mindist"/>
      <contact name="vel_lin" data="found linvel angvel"/>
      <contact name="acc_force" data="found force"/>
      <contact name="acc_torque" data="torque"/>
      <contact name="acc_force_vel" data="force linvel"/>
      <contact name="acc_maxforce" data="found dist" reduce="maxforce"/>
      <contact name="acc_netforce" data="found dist" reduce="netforce"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model.get(), NotNull());
  MjDataPtr data = MakeData(model);
  mju_zero(data->sensordata, model->nsensordata);

  EXPECT_EQ(model->sensor_needstage[0], mjSTAGE_POS);
  EXPECT_EQ(model->sensor_needstage[1], mjSTAGE_VEL);
  EXPECT_EQ(model->sensor_needstage[2], mjSTAGE_ACC);
  EXPECT_EQ(model->sensor_needstage[3], mjSTAGE_ACC);
  EXPECT_EQ(model->sensor_needstage[4], mjSTAGE_ACC);
  EXPECT_EQ(model->sensor_needstage[5], mjSTAGE_ACC);
  EXPECT_EQ(model->sensor_needstage[6], mjSTAGE_ACC);

  data->qvel[0] = 1.0;
  data->qvel[1] = 2.0;
  data->qvel[2] = -0.5;
  data->qvel[3] = 0.3;
  data->qvel[4] = -0.4;
  data->qvel[5] = 0.7;

  // mj_fwdPosition + mj_sensorPos should populate only the POS-stage sensor
  mj_fwdPosition(model.get(), data.get());
  mj_sensorPos(model.get(), data.get());
  EXPECT_THAT(
      GetSensor(model.get(), data.get(), "pos_found"),
      Pointwise(MjNear(1e-6, 1e-5),
                vector<mjtNum>{1, -0.02, 0, 0, -0.01, 0, 0, 1, 0, 1, 0}));
  EXPECT_THAT(GetSensor(model.get(), data.get(), "vel_lin"),
              ElementsAre(0, 0, 0, 0, 0, 0, 0));
  EXPECT_THAT(GetSensor(model.get(), data.get(), "acc_force"),
              ElementsAre(0, 0, 0, 0));

  // mj_fwdVelocity + mj_sensorVel should populate the VEL-stage sensor
  mj_fwdVelocity(model.get(), data.get());
  mj_sensorVel(model.get(), data.get());
  EXPECT_EQ(GetSensor(model.get(), data.get(), "vel_lin")[0], 1);
  EXPECT_NE(GetSensor(model.get(), data.get(), "vel_lin")[1], 0);
  EXPECT_THAT(GetSensor(model.get(), data.get(), "acc_force"),
              ElementsAre(0, 0, 0, 0));
}

TEST_F(SensorTest, ContactVelocity) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 1" condim="6"/>
      <body name="b" pos="0 0 0.08">
        <freejoint/>
        <geom name="sphere" type="sphere" size="0.1" condim="6" friction="0.8 0.1 0.05"/>
      </body>
    </worldbody>
    <sensor>
      <contact name="vel12" geom1="floor" geom2="sphere" data="linvel angvel"/>
      <contact name="vel21" geom1="sphere" geom2="floor" data="linvel angvel"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  const mjtNum qvel[6] = {1.2, -0.7, 0.4, 0.5, -1.1, 0.8};

  // with elliptic cone, efc_vel is directly the 6D contact velocity
  m->opt.cone = mjCONE_ELLIPTIC;
  mj_resetData(m.get(), d.get());
  mju_copy(d->qvel, qvel, 6);
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->ncon, 1);

  const mjContact& con = d->contact[0];
  const mjtNum* efc_vel6 = d->efc_vel + con.efc_address;

  // vel12 (unflipped) should match efc_vel in all 6 components
  vector vel12 = GetSensor(m.get(), d.get(), "vel12");
  EXPECT_THAT(vel12, Pointwise(MjNear(1e-12, 1e-5),
                               vector<mjtNum>(efc_vel6, efc_vel6 + 6)));

  // vel21 (flipped) should negate the z-component (index 2 and 5)
  vector vel21 = GetSensor(m.get(), d.get(), "vel21");
  vector<mjtNum> expected_flipped = {vel12[0], vel12[1], -vel12[2],
                                     vel12[3], vel12[4], -vel12[5]};
  EXPECT_THAT(vel21, Pointwise(MjNear(1e-12, 1e-5), expected_flipped));

  // pyramidal cone should report the exact same velocity
  m->opt.cone = mjCONE_PYRAMIDAL;
  mj_resetData(m.get(), d.get());
  mju_copy(d->qvel, qvel, 6);
  mj_forward(m.get(), d.get());
  EXPECT_THAT(GetSensor(m.get(), d.get(), "vel12"),
              Pointwise(MjNear(1e-12, 1e-5), vel12));

  // even when condim = 1 (frictionless), sensor still reports full 6D velocity
  vector vel_condim6 = GetSensor(m.get(), d.get(), "vel12");
  m->geom_condim[0] = 1;
  m->geom_condim[1] = 1;
  mj_resetData(m.get(), d.get());
  mju_copy(d->qvel, qvel, 6);
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->contact[0].dim, 1);
  EXPECT_THAT(GetSensor(m.get(), d.get(), "vel12"),
              Pointwise(MjNear(1e-12, 1e-5), vel_condim6));
}

TEST_F(SensorTest, CameraProjection) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1.1 0 1">
        <geom type="box" size=".1 .6 .375"/>
        <site name="frontorigin" pos="-.1  .6  .375"/>
        <site name="frontcorner" pos="-.1 -.6 -.375"/>
      </body>
      <body pos="-1.1 0 1">
        <geom type="box" size=".1 .6 .375"/>
        <site name="backcenter" pos="-.1 0 0"/>
      </body>
      <camera pos="0 0 1" xyaxes="0 -1 0 0 0 1" fovy="41.11209"
              resolution="1920 1200" name="fixedcamera"/>
    </worldbody>
    <sensor>
      <camprojection site="frontorigin" camera="fixedcamera"/>
      <camprojection site="frontcorner" camera="fixedcamera"/>
      <camprojection site="backcenter" camera="fixedcamera"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // update positions and sensors
  mj_fwdPosition(model.get(), data.get());
  mj_sensorPos(model.get(), data.get());
  EXPECT_THAT(model->cam_resolution[0], 1920);
  EXPECT_THAT(model->cam_resolution[1], 1200);
  mjtNum eps = 1e-4;
  EXPECT_NEAR(data->sensordata[0], 0, eps);
  EXPECT_NEAR(data->sensordata[1], 0, eps);
  EXPECT_NEAR(data->sensordata[2], 1920, eps);
  EXPECT_NEAR(data->sensordata[3], 1200, eps);
  EXPECT_NEAR(data->sensordata[4], 960, eps);
  EXPECT_NEAR(data->sensordata[5], 600, eps);
}

// previous implementation of cam_project to verify the new one
static void cam_project_old(mjtNum sensordata[2], const mjtNum target_xpos[3],
                            const mjtNum cam_xpos[3], const mjtNum cam_xmat[9],
                            const int cam_res[2], mjtNum cam_fovy,
                            const float cam_intrinsic[4],
                            const float cam_sensorsize[2]) {
  mjtNum fx, fy;

  // translation matrix (4x4)
  mjtNum translation[4][4] = {};
  translation[0][0] = 1;
  translation[1][1] = 1;
  translation[2][2] = 1;
  translation[3][3] = 1;
  translation[0][3] = -cam_xpos[0];
  translation[1][3] = -cam_xpos[1];
  translation[2][3] = -cam_xpos[2];

  // rotation matrix (4x4)
  mjtNum rotation[4][4] = {};
  rotation[0][0] = 1;
  rotation[1][1] = 1;
  rotation[2][2] = 1;
  rotation[3][3] = 1;
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      rotation[i][j] = cam_xmat[j * 3 + i];
    }
  }

  // focal transformation matrix (3x4)
  if (cam_sensorsize[0] && cam_sensorsize[1]) {
    fx = cam_intrinsic[0] / cam_sensorsize[0] * cam_res[0];
    fy = cam_intrinsic[1] / cam_sensorsize[1] * cam_res[1];
  } else {
    fx = fy = .5 / mju_tan(cam_fovy * mjPI / 360.) * cam_res[1];
  }

  mjtNum focal[3][4] = {};
  focal[0][0] = -fx;
  focal[1][1] = fy;
  focal[2][2] = 1.0;

  // image matrix (3x3)
  mjtNum image[3][3] = {};
  image[0][0] = 1;
  image[1][1] = 1;
  image[2][2] = 1;
  image[0][2] = (mjtNum)cam_res[0] / 2.0;
  image[1][2] = (mjtNum)cam_res[1] / 2.0;

  // projection matrix (3x4): product of all 4 matrices
  mjtNum proj[3][4] = {};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 4; k++) {
        for (int l = 0; l < 4; l++) {
          for (int n = 0; n < 4; n++) {
            proj[i][n] +=
                image[i][j] * focal[j][k] * rotation[k][l] * translation[l][n];
          }
        }
      }
    }
  }

  // projection matrix multiplies homogenous [x, y, z, 1] vectors
  mjtNum pos_hom[4] = {0, 0, 0, 1};
  mju_copy3(pos_hom, target_xpos);

  // project world coordinates into pixel space, see:
  // https://en.wikipedia.org/wiki/3D_projection#Mathematical_formula
  mjtNum pixel_coord_hom[3] = {0};
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      pixel_coord_hom[i] += proj[i][j] * pos_hom[j];
    }
  }

  // avoid dividing by tiny numbers
  mjtNum denom = pixel_coord_hom[2];
  if (mju_abs(denom) < mjMINVAL) {
    if (denom < 0) {
      denom = mju_min(denom, -mjMINVAL);
    } else {
      denom = mju_max(denom, mjMINVAL);
    }
  }

  // compute projection
  sensordata[0] = pixel_coord_hom[0] / denom;
  sensordata[1] = pixel_coord_hom[1] / denom;
}

TEST_F(SensorTest, CameraProjectionComparison) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="1.5 0.5 1.2">
        <site name="target1" pos="0 0 0"/>
      </body>
      <body pos="-0.8 -1.2 0.5">
        <site name="target2" pos="0 0 0"/>
      </body>
      <body pos="-0.5 0.5 0.5">
        <site name="target3" pos="0 0 0"/>
      </body>

      <camera name="cam1" pos="0 0 1" xyaxes="0 -1 0 0 0 1" fovy="45" resolution="800 600"/>
      <camera name="cam2" pos="1 -1 2" euler="10 20 30" fovy="60" resolution="1024 768"/>
      <camera name="cam3" pos="-1 1 1.5" euler="-10 -20 30"
              resolution="640 480" sensorsize="0.032 0.024"
              focalpixel="500 500"/>
    </worldbody>
    <sensor>
      <camprojection site="target1" camera="cam1"/>
      <camprojection site="target2" camera="cam1"/>
      <camprojection site="target3" camera="cam1"/>

      <camprojection site="target1" camera="cam2"/>
      <camprojection site="target2" camera="cam2"/>
      <camprojection site="target3" camera="cam2"/>

      <camprojection site="target1" camera="cam3"/>
      <camprojection site="target2" camera="cam3"/>
      <camprojection site="target3" camera="cam3"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  mj_fwdPosition(model.get(), data.get());
  mj_sensorPos(model.get(), data.get());

  for (int i = 0; i < model->nsensor; ++i) {
    if (model->sensor_type[i] == mjSENS_CAMPROJECTION) {
      int objid = model->sensor_objid[i];
      int refid = model->sensor_refid[i];

      mjtNum expected[2];
      cam_project_old(expected, data->site_xpos + 3 * objid,
                      data->cam_xpos + 3 * refid, data->cam_xmat + 9 * refid,
                      model->cam_resolution + 2 * refid, model->cam_fovy[refid],
                      model->cam_intrinsic + 4 * refid,
                      model->cam_sensorsize + 2 * refid);

      int adr = model->sensor_adr[i];
      EXPECT_NEAR(data->sensordata[adr], expected[0], MjTol(1e-10, 1e-3));
      EXPECT_NEAR(data->sensordata[adr + 1], expected[1], MjTol(1e-10, 1e-3));
    }
  }
}

TEST_F(SensorTest, InsideSite) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 1">
        <joint type="slide" axis="1 0 0" range="-.75 .75"/>
        <geom name="query" type="sphere" size=".01" rgba="1 0 0 1"/>
      </body>

      <site name="sphere"    type="sphere"    size=".11"                           pos="-.5 0 1"/>
      <site name="capsule"   type="capsule"   size=".08 .15"     euler="20 -40 60" pos="-.25 0 1"/>
      <site name="ellipsoid" type="ellipsoid" size=".11 .15 .09" euler="20 40 -60" pos="0 0 1"/>
      <site name="cylinder"  type="cylinder"  size=".09 .12"     euler="-20 40 60" pos=".25 0 1"/>
      <site name="box"       type="box"       size=".08 .1 .14"  euler="20 -40 60" pos=".5 0 1"/>
    </worldbody>

    <sensor>
      <insidesite name="sphere"    site="sphere"    objtype="geom" objname="query"/>
      <insidesite name="capsule"   site="capsule"   objtype="geom" objname="query"/>
      <insidesite name="ellipsoid" site="ellipsoid" objtype="geom" objname="query"/>
      <insidesite name="cylinder"  site="cylinder"  objtype="geom" objname="query"/>
      <insidesite name="box"       site="box"       objtype="geom" objname="query"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nsensordata, 5);
  MjDataPtr data = MakeData(model);

  mjtNum hpos[5] = {-.5, -.25, 0, .25, .5};
  for (int i = 0; i < 5; i++) {
    data->qpos[0] = hpos[i];
    mj_forward(model.get(), data.get());
    vector<mjtNum> expected(5, 0.0);
    expected[i] = 1.0;
    EXPECT_EQ(AsVector(data->sensordata, model->nsensordata), expected);
  }
}

TEST_F(SensorTest, InsideSiteMesh) {
  constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="box_mesh" vertex="-0.1 -0.1 -0.1  0.1 -0.1 -0.1  0.1 0.1 -0.1  -0.1 0.1 -0.1  -0.1 -0.1 0.1  0.1 -0.1 0.1  0.1 0.1 0.1  -0.1 0.1 0.1"/>
    </asset>

    <worldbody>
      <body pos="0 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom name="query" type="sphere" size=".001"/>
      </body>

      <site name="mesh_site" type="mesh" mesh="box_mesh" pos="1.0 0 0"/>
    </worldbody>

    <sensor>
      <insidesite name="inside" site="mesh_site" objtype="geom" objname="query"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nsensordata, 1);
  MjDataPtr data = MakeData(model);

  // Position query inside mesh site
  data->qpos[0] = 1.0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->sensordata[0], 1.0);

  // Position query outside mesh site (to the left)
  data->qpos[0] = 0.0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->sensordata[0], 0.0);

  // Position query outside mesh site (to the right)
  data->qpos[0] = 2.0;
  mj_forward(model.get(), data.get());
  EXPECT_EQ(data->sensordata[0], 0.0);
}

TEST_F(SensorTest, InsideSiteEnclosed) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="box_site" type="box" size="1.0 1.0 1.0" pos="0 0 0"/>

      <geom name="inner_sphere" type="sphere" size="0.4" pos="0.5 0 0"/>
      <geom name="protrude_sphere" type="sphere" size="0.4" pos="0.7 0 0"/>

      <site name="inner_site" type="sphere" size="0.5" pos="0 0 0"/>
      <site name="protrude_site" type="sphere" size="0.5" pos="0.6 0 0"/>

      <body name="body_enclosed" pos="0 0 0">
        <geom name="bg1" type="sphere" size="0.2" pos="-0.5 0 0"/>
        <geom name="bg2" type="sphere" size="0.3" pos="0.5 0 0"/>
      </body>

      <body name="body_protrude" pos="0 0 0">
        <geom name="bg3" type="sphere" size="0.2" pos="-0.5 0 0"/>
        <geom name="bg4" type="sphere" size="0.3" pos="0.8 0 0"/>
      </body>

      <body name="xbody_parent" pos="0 0 0">
        <geom name="p_geom" type="sphere" size="0.2" pos="0 0 0"/>
        <body name="xbody_child" pos="0.9 0 0">
          <geom name="c_geom" type="sphere" size="0.3" pos="0 0 0"/>
        </body>
      </body>
    </worldbody>

    <sensor>
      <insidesite name="s_inner_sphere"    site="box_site" objtype="geom"   objname="inner_sphere"    enclosed="true"/>
      <insidesite name="s_protrude_sphere" site="box_site" objtype="geom"   objname="protrude_sphere" enclosed="true"/>
      <insidesite name="s_inner_site"      site="box_site" objtype="site"   objname="inner_site"      enclosed="true"/>
      <insidesite name="s_protrude_site"   site="box_site" objtype="site"   objname="protrude_site"   enclosed="true"/>
      <insidesite name="s_body_enclosed"   site="box_site" objtype="body"   objname="body_enclosed"   enclosed="true"/>
      <insidesite name="s_body_protrude"   site="box_site" objtype="body"   objname="body_protrude"   enclosed="true"/>
      <insidesite name="s_xbody"           site="box_site" objtype="xbody"  objname="xbody_parent"    enclosed="true"/>
    </sensor>
  </mujoco>
  )";

  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_EQ(model->nsensordata, 7);
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // inner_sphere: x max = 0.5 + 0.4 = 0.9, box size = 1.0 -> dist = -0.1
  EXPECT_NEAR(data->sensordata[0], -0.1, 1e-5);

  // protrude_sphere: x max = 0.7 + 0.4 = 1.1, box size = 1.0 -> dist = 0.1
  EXPECT_NEAR(data->sensordata[1], 0.1, 1e-5);

  // inner_site: radius = 0.5, box size = 1.0 -> dist = -0.5
  EXPECT_NEAR(data->sensordata[2], -0.5, 1e-5);

  // protrude_site: x max = 0.6 + 0.5 = 1.1, box size = 1.0 -> dist = 0.1
  EXPECT_NEAR(data->sensordata[3], 0.1, 1e-5);

  // body_enclosed: max(bg1: -0.3, bg2: 0.5+0.3-1.0 = -0.2) = -0.2
  EXPECT_NEAR(data->sensordata[4], -0.2, 1e-5);

  // body_protrude: max(bg3: -0.3, bg4: 0.8+0.3-1.0 = 0.1) = 0.1
  EXPECT_NEAR(data->sensordata[5], 0.1, 1e-5);

  // xbody_parent: max(p_geom: -0.8, c_geom: 0.9+0.3-1.0 = 0.2) = 0.2
  EXPECT_NEAR(data->sensordata[6], 0.2, 1e-5);

  // non-convex geom on body should fail compilation
  constexpr char bad_xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="box_site" type="box" size="1 1 1"/>
      <body name="b">
        <geom name="plane_geom" type="plane" size="1 1 1"/>
      </body>
    </worldbody>
    <sensor>
      <insidesite site="box_site" objtype="body" objname="b" enclosed="true"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr bad_model = LoadModelFromString(bad_xml, error, sizeof(error));
  EXPECT_THAT(bad_model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("compact convex shape"));

  // body with 0 geoms should fail compilation
  constexpr char empty_body_xml[] = R"(
  <mujoco>
    <worldbody>
      <site name="box_site" type="box" size="1 1 1"/>
      <body name="empty_b"/>
    </worldbody>
    <sensor>
      <insidesite site="box_site" objtype="body" objname="empty_b" enclosed="true"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr empty_body_model =
      LoadModelFromString(empty_body_xml, error, sizeof(error));
  EXPECT_THAT(empty_body_model.get(), IsNull());
  EXPECT_THAT(error, HasSubstr("must have at least one geom"));
}

TEST_F(SensorTest, RangefinderCamera) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 .1"/>
      <body pos="0 0 2">
        <camera name="persp" xyaxes="1 0 0 0 1 0" resolution="3 3" fovy="90"/>
        <camera name="ortho" euler="0 45 0" resolution="3 3"
          projection="orthographic" fovy="2"/>
      </body>
    </worldbody>

    <sensor>
      <rangefinder camera="persp" data="dist depth"/>
      <rangefinder camera="ortho" data="dist dir origin point"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;

  // first sensor: data="dist depth" => (1+1)*9 = 18
  // second sensor: data="dist dir origin point" => (1+3+3+3)*9 = 90
  EXPECT_EQ(model->nsensordata, 108);

  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  mjtNum tol = 1e-6;
  mjtNum height = 2.0;
  mjtNum fy = 1.5;
  mjtNum offsets[3] = {-1.0, 0.0, 1.0};  // pixel center - principal point

  // test 1: perspective camera - rays diverge, distance varies with angle
  int adr0 = model->sensor_adr[0];
  constexpr int stride0 = 2;  // dist(1) + depth(1)
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      int idx = row * 3 + col;
      mjtNum dx = offsets[col] / fy;
      mjtNum dy = offsets[row] / fy;
      mjtNum expected_dist = height * mju_sqrt(1 + dx * dx + dy * dy);
      mjtNum dist = data->sensordata[adr0 + idx * stride0];
      EXPECT_NEAR(dist, expected_dist, tol)
          << "perspective dist pixel (" << row << ", " << col << ")";

      // depth should equal camera height (2.0) for all pixels
      mjtNum depth = data->sensordata[adr0 + idx * stride0 + 1];
      EXPECT_NEAR(depth, height, tol)
          << "perspective depth pixel (" << row << ", " << col << ")";
    }
  }

  // test 2: orthographic camera distance - tilted 45 degrees around Y axis
  mjtNum extent = 2.0;  // fovy for orthographic
  mjtNum half_extent = extent / 2;
  mjtNum fx = 1.5;  // width / 2 for 3x3 image
  mjtNum cx = 1.5;  // principal point
  mjtNum cos45 = mju_sqrt(0.5);
  mjtNum sin45 = mju_sqrt(0.5);
  int adr1 = model->sensor_adr[1];
  constexpr int stride1 = 10;  // dist(1) + dir(3) + origin(3) + point(3)
  for (int row = 0; row < 3; row++) {
    for (int col = 0; col < 3; col++) {
      int idx = row * 3 + col;

      // pixel offset in camera frame: matches mju_camPixelRay formula
      mjtNum px_cam = (col + 0.5 - cx) / fx * half_extent;

      // camera tilted 45 around Y: local +X maps to world (+cos45, 0, -sin45)
      mjtNum origin_z = height - px_cam * sin45;

      // ray hits z=0 plane: distance = origin_z / cos45
      mjtNum expected_dist = origin_z / cos45;
      mjtNum dist = data->sensordata[adr1 + idx * stride1];
      EXPECT_NEAR(dist, expected_dist, tol)
          << "orthographic dist pixel (" << row << ", " << col << ")";

      // verify point = origin + dir * dist
      mjtNum* dir = data->sensordata + adr1 + idx * stride1 + 1;
      mjtNum* origin = data->sensordata + adr1 + idx * stride1 + 4;
      mjtNum* point = data->sensordata + adr1 + idx * stride1 + 7;
      mjtNum expected_point[3];
      mju_addScl3(expected_point, origin, dir, dist);
      EXPECT_NEAR(point[0], expected_point[0], tol)
          << "ortho point[0] pixel (" << row << ", " << col << ")";
      EXPECT_NEAR(point[1], expected_point[1], tol)
          << "ortho point[1] pixel (" << row << ", " << col << ")";
      EXPECT_NEAR(point[2], expected_point[2], tol)
          << "ortho point[2] pixel (" << row << ", " << col << ")";
    }
  }
}

TEST_F(SensorTest, RFCamera) {
  const string xml_path =
      GetTestDataFilePath("engine/testdata/sensor/rfcamera.xml");
  char error[1024];
  mjModel* model = mj_loadXML(xml_path.c_str(), nullptr, error, sizeof(error));
  ASSERT_THAT(model, NotNull()) << error;

  // both sensors have data="dist point normal" => (1+3+3)*16 = 112
  ASSERT_EQ(model->nsensor, 2);
  EXPECT_EQ(model->sensor_dim[0], 112);
  EXPECT_EQ(model->sensor_dim[1], 112);

  mjData* data = mj_makeData(model);
  mj_step(model, data);

  // check both sensors: dist, point, normal
  constexpr int stride = 7;  // dist(1) + point(3) + normal(3)
  for (int s = 0; s < 2; s++) {
    int adr = model->sensor_adr[s];
    for (int i = 0; i < 16; i++) {
      mjtNum dist = data->sensordata[adr + i * stride];
      mjtNum* point = data->sensordata + adr + i * stride + 1;
      mjtNum* normal = data->sensordata + adr + i * stride + 4;

      EXPECT_TRUE(dist > 0 || dist == -1) << "sensor " << s << " pixel " << i;

      if (dist > 0) {
        EXPECT_GT(mju_norm3(point), 0.0) << "sensor " << s << " point " << i;
        EXPECT_NEAR(mju_norm3(normal), 1.0, 1e-6)
            << "sensor " << s << " normal " << i;
      } else {
        EXPECT_NEAR(mju_norm3(point), 0.0, 1e-6)
            << "sensor " << s << " point " << i;
        EXPECT_NEAR(mju_norm3(normal), 0.0, 1e-6)
            << "sensor " << s << " normal " << i;
      }
    }
  }

  mj_deleteData(data);
  mj_deleteModel(model);
}

// ----------------------- rangefinder cutoff (maxdist) ------------------------

TEST_F(SensorTest, RangefinderCutoffBeyond) {
  // object at 5m, cutoff=2m: should return -1 (no detection)
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <site name="rf" pos="0 0 0" zaxis="1 0 0"/>
        <geom size="0.01"/>
      </body>
      <body pos="5 0 0">
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder site="rf" cutoff="2"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // object beyond cutoff: sensor returns -1
  EXPECT_MJTNUM_EQ(data->sensordata[0], -1);
}

TEST_F(SensorTest, RangefinderCutoffWithin) {
  // object at 1.5m, cutoff=2m: should return actual distance
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <site name="rf" pos="0 0 0" zaxis="1 0 0"/>
        <geom size="0.01"/>
      </body>
      <body pos="1.5 0 0">
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder site="rf" cutoff="2"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // object within cutoff: sensor returns positive distance
  EXPECT_GT(data->sensordata[0], 0);
  // distance should be approximately 1.5 - sphere_radius = 1.4
  EXPECT_NEAR(data->sensordata[0], 1.4, MjTol(1e-5, 1e-3));
}

TEST_F(SensorTest, RangefinderCutoffNoHit) {
  // no object in ray path, cutoff set: should return -1
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <site name="rf" pos="0 0 0" zaxis="1 0 0"/>
        <geom size="0.01"/>
      </body>
      <body pos="0 5 0">
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder site="rf" cutoff="10"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // no hit: -1 regardless of cutoff
  EXPECT_MJTNUM_EQ(data->sensordata[0], -1);
}

TEST_F(SensorTest, RangefinderCutoffNoCutoff) {
  // object at 5m, no cutoff (cutoff=0): should return actual distance
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body pos="0 0 0">
        <site name="rf" pos="0 0 0" zaxis="1 0 0"/>
        <geom size="0.01"/>
      </body>
      <body pos="5 0 0">
        <geom type="sphere" size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder site="rf"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // no cutoff: sensor returns actual distance (~4.9)
  EXPECT_GT(data->sensordata[0], 0);
  EXPECT_NEAR(data->sensordata[0], 4.9, MjTol(1e-5, 1e-3));
}

TEST_F(SensorTest, RangefinderCameraCutoff) {
  // camera rangefinder: plane at z=0, camera at z=2 looking down, cutoff=1.5
  // all pixels hit the plane at distance ~2.0, which is > cutoff
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 .1"/>
      <body pos="0 0 2">
        <camera name="cam" xyaxes="1 0 0 0 1 0" resolution="3 3" fovy="90"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder camera="cam" cutoff="1.5"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  // all 9 pixels: center pixel dist=2.0, corner pixels > 2.0, all > cutoff=1.5
  int adr = model->sensor_adr[0];
  for (int i = 0; i < 9; i++) {
    EXPECT_MJTNUM_EQ(data->sensordata[adr + i], -1)
        << "pixel " << i << " should be -1 (beyond cutoff)";
  }
}

TEST_F(SensorTest, RangefinderCameraCutoffWithin) {
  // camera rangefinder 1m above a plane whose origin is 5m away, cutoff=1.5
  // the plane must not be culled: the hit at distance 1 is within the cutoff
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 .1"/>
      <body pos="5 0 1">
        <camera name="cam" xyaxes="1 0 0 0 1 0" resolution="1 1" fovy="10"/>
      </body>
    </worldbody>
    <sensor>
      <rangefinder camera="cam" cutoff="1.5"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);
  mj_forward(model.get(), data.get());

  EXPECT_NEAR(data->sensordata[0], 1, MjTol(1e-10, 1e-5));
}

// ------------------------------- sensor delays -------------------------------

TEST_F(SensorTest, SensorDelay) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" delay="0.02" nsample="3"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // delay = 0.02 seconds, timestep = 0.01
  // history = 3 (more than delay/timestep=2) to ensure buffer coverage
  EXPECT_EQ(model->sensor_history[0], 3);
  EXPECT_NEAR(model->sensor_delay[0], 0.02, MjTol(1e-10, 1e-7));

  // Use different values to verify exact delay timing.
  // With delay=0.02 and timestep=0.01, we expect 2-step delay:
  // - At step N, sensordata should reflect qpos from step N-2.

  // step 0: qpos=10, read from initial buffer
  data->qpos[0] = 10.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 0.0, 1e-10) << "step 0";

  // step 1: qpos=20, still reading initial buffer
  data->qpos[0] = 20.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 0.0, 1e-10) << "step 1";

  // step 2: qpos=30, read value from step 0 (delay=2 steps)
  data->qpos[0] = 30.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 10.0, 1e-10) << "step 2";

  // step 3: qpos=40, read value from step 1 (delay=2 steps)
  data->qpos[0] = 40.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 20.0, 1e-10) << "step 3";

  // step 4: qpos=50, read value from step 2 (delay=2 steps)
  data->qpos[0] = 50.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 30.0, 1e-10) << "step 4";
}

// Test sensor delay with linear interpolation (interp=1)
// Uses delay = 1.5*timestep so interpolation is meaningful
TEST_F(SensorTest, SensorDelayLinearInterp) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" delay="0.015" nsample="3" interp="linear"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // delay = 0.015 seconds = 1.5*timestep, nsample=3, interp=1 (linear)
  // With linear interpolation and 1.5*timestep delay, the read time falls
  // exactly between two buffer samples, so we should get the average.
  EXPECT_EQ(model->sensor_history[0], 3);
  EXPECT_EQ(model->sensor_history[1], 1);  // interp=1 (linear)
  EXPECT_NEAR(model->sensor_delay[0], 0.015, MjTol(1e-10, 1e-7));

  // Set increasing qpos values: step i -> qpos = (i+1)*10
  // Buffer has samples at times: -0.02, -0.01, 0 (initialized)
  // After step 0 at time=0.01: buffer has times -0.01, 0, 0.01 with values 0,
  // 0, 10 Read at time 0.01 - 0.015 = -0.005: interpolate between t=-0.01
  // (val=0) and t=0 (val=0) Expected: 0 * 0.5 + 0 * 0.5 = 0

  data->qpos[0] = 10.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 0.0, 1e-10) << "step 0";

  // After step 1 at time=0.02: buffer has times 0, 0.01, 0.02 with values 0,
  // 10, 20 Read at time 0.02 - 0.015 = 0.005: interpolate between t=0 (val=0)
  // and t=0.01 (val=10) Expected: 0 * 0.5 + 10 * 0.5 = 5

  data->qpos[0] = 20.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 5.0, 1e-10) << "step 1";

  // After step 2 at time=0.03: buffer has times 0.01, 0.02, 0.03 with values
  // 10, 20, 30 Read at 0.03 - 0.015 = 0.015: interpolate between t=0.01
  // (val=10) and t=0.02 (val=20) Expected: 10 * 0.5 + 20 * 0.5 = 15

  data->qpos[0] = 30.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 15.0, 1e-10) << "step 2";
}

TEST_F(SensorTest, SensorInterval) {
  // This test uses the exact values from the documentation for interval:
  // timestep=1, interval=2.5, producing times 0, 3, 5, 8, 10, 13, ...
  // with interval="2.5 -1.5", producing times 1, 4, 6, 9, 11, 14, ...
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="1" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos name="default_phase" joint="slide" interval="2.5 0" nsample="10"/>
      <jointpos name="offset_phase" joint="slide" interval="2.5 -1.5" nsample="10"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  int sensor0 = mj_name2id(model.get(), mjOBJ_SENSOR, "default_phase");
  int sensor1 = mj_name2id(model.get(), mjOBJ_SENSOR, "offset_phase");
  int adr0 = model->sensor_adr[sensor0];
  int adr1 = model->sensor_adr[sensor1];

  // Verify initial buffer timestamps (after mj_makeData/mj_resetData)
  // With period=2.5, dt=1.0, nsample=10:
  // sensor0 (phase = -period = -2.5): continuous times are -2.5, -5, -7.5, ...
  //   rounded up to dt: -2, -5, -7, -10, -12, -15, -17, -20, -22, -25
  // sensor1 (phase = -1.5): continuous times are -1.5, -4, -6.5, ...
  //   rounded up to dt: -1, -4, -6, -9, -11, -14, -16, -19, -21, -24
  int n0 = model->sensor_history[2 * sensor0];
  int n1 = model->sensor_history[2 * sensor1];
  mjtNum* buf0 = data->history + model->sensor_historyadr[sensor0];
  mjtNum* buf1 = data->history + model->sensor_historyadr[sensor1];
  mjtNum* times0 = buf0 + 2;
  mjtNum* times1 = buf1 + 2;
  mjtNum expected_times0[] = {-25, -22, -20, -17, -15, -12, -10, -7, -5, -2};
  mjtNum expected_times1[] = {-24, -21, -19, -16, -14, -11, -9, -6, -4, -1};
  for (int i = 0; i < n0; i++) {
    EXPECT_NEAR(times0[i], expected_times0[i], 1e-10);
  }
  for (int i = 0; i < n1; i++) {
    EXPECT_NEAR(times1[i], expected_times1[i], 1e-10);
  }

  // sensor0: interval="2.5 0" -> time_prev starts at -2.5
  //   triggers at: 0, 3, 5, 8, 10, 13, ... (gaps: 3,2,3,2,3,...)
  // sensor1: interval="2.5 -1.5" -> time_prev starts at -1.5
  //   triggers at: 1, 4, 6, 9, 11, 14, ... (gaps: 3,2,3,2,3,...)

  // Arrays tracking when each sensor triggers (1=triggers, 0=holds)
  // Times:         0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
  int triggers0[] = {1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0};
  int triggers1[] = {0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1};

  mjtNum value0 = 0, value1 = 0;
  for (int t = 0; t < 15; t++) {
    // set position to current time (so we can track when sensor was computed)
    data->qpos[0] = t;
    mj_step(model.get(), data.get());

    // update expected values based on trigger pattern
    if (triggers0[t]) value0 = t;
    if (triggers1[t]) value1 = t;

    EXPECT_NEAR(data->sensordata[adr0], value0, MjTol(1e-10, 1e-7))
        << "sensor0 at t=" << t;
    EXPECT_NEAR(data->sensordata[adr1], value1, MjTol(1e-10, 1e-7))
        << "sensor1 at t=" << t;
  }
}

TEST_F(SensorTest, SensorDelayInterval) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" delay="0.02" interval="0.03 0" nsample="5"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // Combined delay and interval
  EXPECT_EQ(model->sensor_history[0], 5);
  EXPECT_NEAR(model->sensor_delay[0], 0.02, MjTol(1e-10, 1e-7));
  EXPECT_NEAR(model->sensor_interval[2 * 0], 0.03, MjTol(1e-10, 1e-7));

  // Verify initial buffer timestamps (after mj_makeData/mj_resetData)
  // With period=0.03, dt=0.01, nsample=5, phase=0 (means -period=-0.03):
  // continuous times: -0.03, -0.06, -0.09, -0.12, -0.15
  // rounded up to dt: -0.03, -0.06, -0.09, -0.12, -0.15 (multiples of dt)
  int n = model->sensor_history[0];
  mjtNum* buf = data->history + model->sensor_historyadr[0];
  mjtNum* times = buf + 2;
  mjtNum expected_times[] = {-0.15, -0.12, -0.09, -0.06, -0.03};
  for (int i = 0; i < n; i++) {
    EXPECT_NEAR(times[i], expected_times[i], MjTol(1e-10, 0.015));
  }

  // set position
  data->qpos[0] = 5.0;

  // initial steps: reading from buffer (initially 0)
  // With delay=0.02, interval=0.03:
  // - At t=0, interval satisfied: compute 5.0, insert at t=0 (current time)
  // - Reading happens at d->time - delay; at t=0.02, reads at t=0.00 (5.0)
  for (int i = 0; i < 2; i++) {
    mj_step(model.get(), data.get());
    // sensor reads delayed value (0.0 from initial buffer)
    EXPECT_NEAR(data->sensordata[0], 0.0, MjTol(1e-10, 1e-7)) << "step " << i;
  }

  // step 3 (i=2): reading at t=0.00 now returns the inserted value 5.0
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 5.0, MjTol(1e-10, 1e-7));
}

TEST_F(SensorTest, SensorHistoryOnly) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" nsample="5"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // history only, no delay or interval
  EXPECT_EQ(model->sensor_history[0], 5);
  EXPECT_NEAR(model->sensor_delay[0], 0.0, MjTol(1e-10, 1e-7));
  EXPECT_NEAR(model->sensor_interval[2 * 0], 0.0, MjTol(1e-10, 1e-7));

  // set position
  data->qpos[0] = 3.0;

  // without delay, sensordata reflects current value immediately
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 3.0, 1e-10);

  // change position, check again
  data->qpos[0] = 7.0;
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 7.0, 1e-10);
}

TEST_F(SensorTest, SensorDelayMultiDim) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01"/>
    <worldbody>
      <body>
        <joint name="ball" type="ball"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <ballangvel joint="ball" delay="0.02" nsample="2"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // ballangvel is 3D
  EXPECT_EQ(model->sensor_dim[0], 3);
  EXPECT_EQ(model->sensor_history[0], 2);

  // set angular velocity
  data->qvel[0] = 1.0;
  data->qvel[1] = 2.0;
  data->qvel[2] = 3.0;

  // step: reading delayed value (initially 0)
  mj_step(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[0], 0.0, 1e-10);
  EXPECT_NEAR(data->sensordata[1], 0.0, 1e-10);
  EXPECT_NEAR(data->sensordata[2], 0.0, 1e-10);

  // after delay, values should propagate
  mj_step(model.get(), data.get());
  mj_step(model.get(), data.get());
  mj_step(model.get(), data.get());
  // angular velocity is affected by dynamics, just check the buffer works
  EXPECT_THAT(AsVector(data->sensordata, 3), Not(ElementsAre(0.0, 0.0, 0.0)));
}

TEST_F(SensorTest, ReadSensor) {
  constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.01" gravity="0 0 0"/>
    <worldbody>
      <body>
        <joint name="slide" type="slide"/>
        <geom size="0.1"/>
      </body>
    </worldbody>
    <sensor>
      <jointpos joint="slide" nsample="5"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  MjDataPtr data = MakeData(model);

  // step with different qpos values to populate buffer
  // mj_advance inserts at current time, then time advances
  data->qpos[0] = 1.0;
  mj_step(model.get(), data.get());  // inserts 1.0 at t=0, time -> 0.01

  data->qpos[0] = 2.0;
  mj_step(model.get(), data.get());  // inserts 2.0 at t=0.01, time -> 0.02

  data->qpos[0] = 3.0;
  mj_step(model.get(), data.get());  // inserts 3.0 at t=0.02, time -> 0.03

  // now time=0.03, buffer has: [t=0: 1.0, t=0.01: 2.0, t=0.02: 3.0]

  // read at different times from history
  mjtNum result[1];
  const mjtNum* ptr;

  // read at t=0 -> returns 1.0
  ptr = mj_readSensor(model.get(), data.get(), 0, 0.0, result, /*order=*/0);
  EXPECT_NEAR(*ptr, 1.0, 1e-10);

  // read at t=0.01 -> returns 2.0 (ZOH: exactly at insertion time)
  ptr = mj_readSensor(model.get(), data.get(), 0, 0.01, result, /*order=*/0);
  EXPECT_NEAR(*ptr, 2.0, 1e-10);

  // read at t=0.02 -> returns 3.0
  ptr = mj_readSensor(model.get(), data.get(), 0, 0.02, result, /*order=*/0);
  EXPECT_NEAR(*ptr, 3.0, 1e-10);
}

// mj_sensorAcc returns correct accelerometer after mj_step1 (issue #3133)
TEST_F(SensorTest, AccelerometerAfterStep1) {
  constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="10 10 0.1"/>
      <body pos="0 0 0.1">
        <freejoint/>
        <geom type="box" size="0.1 0.1 0.1" mass="10"/>
        <site name="imu"/>
      </body>
    </worldbody>
    <sensor>
      <accelerometer site="imu"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  MjDataPtr data = MakeData(model);

  // settle the simulation with split-step loop
  for (int i = 0; i < 100; i++) {
    mj_step1(model.get(), data.get());
    mj_step2(model.get(), data.get());
  }

  // call mj_step1 + mj_sensorAcc, expect gravity reading (~9.81 m/s^2)
  mj_step1(model.get(), data.get());
  mj_sensorAcc(model.get(), data.get());
  EXPECT_NEAR(data->sensordata[2], 9.81, 1e-2);
}

// Test tactile sensor reads non-zero values when contacts occur
TEST_F(SensorTest, TactileSkipTangents) {
  constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag multiccd="enable"/>
    </option>
    <asset>
      <mesh name="sensor_mesh" builtin="sphere" params="0"/>
    </asset>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
      </body>
      <body>
        <geom type="box" size=".7 .7 .3"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_GT(model->nsensordata, 0) << "No sensor data allocated";
  EXPECT_EQ(model->sensor_needstage[0], mjSTAGE_VEL);
  MjDataPtr data = MakeData(model);

  // Compute position and velocity stages only (no mj_sensorAcc needed)
  mj_fwdPosition(model.get(), data.get());
  mj_fwdVelocity(model.get(), data.get());
  mj_sensorVel(model.get(), data.get());

  // Verify initial state
  EXPECT_EQ(data->time, 0.0);
  EXPECT_GT(data->ncon, 0) << "No contacts generated";

  // Tactile sensor layout: [depths..., tang1_vel..., tang2_vel...]
  int ntaxel = model->nsensordata / 3;
  ASSERT_EQ(model->nsensordata % 3, 0) << "Sensor dim should be divisible by 3";

  // No tangents, so tangent components should be zero
  for (int i = ntaxel; i < model->nsensordata; i++) {
    EXPECT_EQ(data->sensordata[i], 0.0)
        << "Tangent component at index " << i << " should be 0";
  }

  // Penetration depth components: verify count, sign, and magnitude ~0.2
  int nonzero_count = 0;
  for (int i = 0; i < ntaxel; i++) {
    if (data->sensordata[i] != 0) {
      nonzero_count++;
      EXPECT_NEAR(data->sensordata[i], 0.2, 0.1)
          << "Penetration depth at taxel " << i;
    }
  }
  EXPECT_EQ(nonzero_count, 2) << "Expected 2 taxels in contact";
}

// Test tactile sensor cutoff attribute, engine clamping, and round-trip XML
// save/load
TEST_F(SensorTest, TactileCutoff) {
  constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag multiccd="enable"/>
    </option>
    <asset>
      <mesh name="sensor_mesh" builtin="sphere" params="0"/>
    </asset>
    <worldbody>
      <body pos="0 0 1">
        <freejoint/>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
      </body>
      <body>
        <geom type="box" size=".7 .7 .3"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh" cutoff="0.05"/>
    </sensor>
  </mujoco>
  )";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  ASSERT_GT(model->nsensordata, 0) << "No sensor data allocated";
  EXPECT_MJTNUM_EQ(model->sensor_cutoff[0], 0.05);

  MjDataPtr data = MakeData(model);

  // Compute collisions and sensors at t=0
  mj_forward(model.get(), data.get());

  int ntaxel = model->nsensordata / 3;
  int nonzero_count = 0;
  for (int i = 0; i < ntaxel; i++) {
    if (data->sensordata[i] != 0) {
      nonzero_count++;
      // Without cutoff, penetration is ~0.2; with cutoff=0.05, it must be
      // clamped to 0.05
      EXPECT_NEAR(data->sensordata[i], 0.05, MjTol(1e-6, 1e-4))
          << "Penetration depth at taxel " << i
          << " should be clamped to cutoff";
    }
  }
  EXPECT_EQ(nonzero_count, 2) << "Expected 2 taxels in contact";

  // Verify XML round-trip preserves cutoff without schema errors
  std::string saved_xml = SaveAndReadXml(model.get());
  EXPECT_THAT(saved_xml, HasSubstr("cutoff=\"0.05\""));
  EXPECT_THAT(saved_xml, Not(HasSubstr("noise")));

  MjModelPtr reloaded =
      LoadModelFromString(saved_xml.c_str(), error, sizeof(error));
  ASSERT_THAT(reloaded.get(), NotNull()) << error;
  EXPECT_MJTNUM_EQ(reloaded->sensor_cutoff[0], 0.05);
}

// insidesite uses subtree_com for massless flex parent bodies
TEST_F(SensorTest, InsideSiteFlexBody) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 0"/>
    <worldbody>
      <body name="parent">
        <flexcomp name="soft" type="grid" count="3 3 3"
                  radius="0.01" dim="3" mass="1">
            <elasticity young="5e4" poisson="0.2"/>
        </flexcomp>
      </body>

      <!-- large site centered at origin, should contain the flex -->
      <site name="container" type="box" size="2 2 2"/>
    </worldbody>

    <sensor>
      <insidesite name="inside" site="container"
                  objtype="body" objname="parent"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  EXPECT_CALL(mock_warning_handler, Warn(testing::HasSubstr("is not rigid")))
      .WillOnce(testing::Return());
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // flex is at origin, site is a large box at origin — should be inside
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->sensordata[0], 1)
      << "flex body should be inside the container site";

  // shift all vertex/node bodies far outside the site via qpos
  // each body has 3 slide joints (x, y, z); shift z by +10
  int parent_id = mj_name2id(m.get(), mjOBJ_BODY, "parent");
  for (int b = parent_id + 1; b < m->nbody; b++) {
    if (m->body_parentid[b] == parent_id) {
      int jadr = m->body_jntadr[b];
      if (jadr >= 0 && m->body_jntnum[b] == 3) {
        // z-slide is the 3rd joint
        d->qpos[m->jnt_qposadr[jadr + 2]] = 10.0;
      }
    }
  }
  mj_forward(m.get(), d.get());

  // subtree_com should now be far outside; sensor should read 0
  EXPECT_EQ(d->sensordata[0], 0)
      << "flex body should be outside the container site after displacement";
}

// Test that a tactile sensor's compile-time body-collision check correctly
// uses the referenced Geom ID instead of mistakenly indexing by Mesh ID.
TEST_F(SensorTest, TactileMeshIdMismatchedValidator) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="sensor_mesh" builtin="sphere" params="0"/>
    </asset>
    <worldbody>
      <body>
        <geom size="0.1" contype="0" conaffinity="0"/>
      </body>
      <body>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
}

// Test that tactile tangent channels correctly project relative velocity
// when the tactile sensor geom is rotated relative to the world.
TEST_F(SensorTest, TactileRotatedGeomTangents) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag multiccd="enable"/>
    </option>
    <asset>
      <mesh name="sensor_mesh" builtin="wedge" params="3 3 45 45 0" scale=".2 .2 .2"/>
    </asset>
    <worldbody>
      <body>
        <geom type="box" size=".25 .25 .25"/>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh" euler="0 90 0"
              mass="0" contype="0" conaffinity="0"/>
      </body>
      <body name="slider" pos="0 0 0">
        <joint type="slide" axis="1 0 0"/>
        <geom type="box" size=".3 .3 .3"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Set sliding velocity along world X
  d->qvel[0] = 2.0;

  mj_forward(m.get(), d.get());

  int ntaxel = m->mesh_vertnum[0];
  ASSERT_EQ(ntaxel, 9);
  ASSERT_EQ(m->nsensordata, 27);

  // At least some taxels should be in contact and register penetration
  int in_contact = 0;
  mjtNum total_tangent_slip = 0;
  for (int j = 0; j < ntaxel; j++) {
    if (d->sensordata[0 * ntaxel + j] > 0) {
      in_contact++;
      total_tangent_slip +=
          d->sensordata[1 * ntaxel + j] + d->sensordata[2 * ntaxel + j];
    }
  }
  EXPECT_GT(in_contact, 0);
  EXPECT_GT(total_tangent_slip, 0.0);
}

// Test that self-geom contacts do not evaluate distance against the sensor
// itself.
TEST_F(SensorTest, TactileSelfGeomExclusion) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="sensor_mesh" builtin="sphere" params="0"/>
    </asset>
    <worldbody>
      <body>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"/>
      </body>
    </worldbody>
    <contact>
      <pair geom1="sensor_geom" geom2="sensor_geom"/>
    </contact>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  mj_forward(m.get(), d.get());

  // With self-geom exclusion, sensor readings must remain 0
  for (int i = 0; i < m->nsensordata; i++) {
    EXPECT_EQ(d->sensordata[i], 0.0);
  }
}

// Test that rotating bodies in contact produce tangential velocity via surface
// contact kinematics.
TEST_F(SensorTest, TactileSpinningContactVelocity) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag multiccd="enable"/>
    </option>
    <asset>
      <mesh name="sensor_mesh" builtin="wedge" params="3 3 45 45 0" scale=".2 .2 .2"/>
    </asset>
    <worldbody>
      <body>
        <geom type="box" size=".25 .25 .25"/>
        <geom name="sensor_geom" type="mesh" mesh="sensor_mesh"
              mass="0" contype="0" conaffinity="0"/>
      </body>
      <body name="spinner" pos="0 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <geom type="sphere" size=".5"/>
      </body>
    </worldbody>
    <sensor>
      <tactile geom="sensor_geom" mesh="sensor_mesh"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  // Set angular velocity about Y axis (pure rotation, zero linear velocity at
  // body origin)
  d->qvel[0] = 10.0;

  mj_forward(m.get(), d.get());

  int ntaxel = m->mesh_vertnum[0];
  ASSERT_EQ(ntaxel, 9);
  ASSERT_EQ(m->nsensordata, 27);

  mjtNum total_tangent_slip = 0;
  for (int j = 0; j < ntaxel; j++) {
    total_tangent_slip +=
        d->sensordata[1 * ntaxel + j] + d->sensordata[2 * ntaxel + j];
  }
  EXPECT_GT(total_tangent_slip, 0.0);
}

// Test that contact and touch sensors work with Flex contacts.
TEST_F(SensorTest, FlexContactSensors) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option gravity="0 0 -9.81"/>
    <worldbody>
      <geom name="floor" type="plane" size="1 1 10"/>
      <body name="parent" pos="0 0 0.005">
        <flexcomp name="soft" type="grid" dof="trilinear" cellcount="2 2 2"
                  count="7 8 9" radius="0.01" dim="3" mass="1"
                  spacing="0.05 0.05 0.05">
            <elasticity young="5e4" poisson="0.2"/>
            <contact selfcollide="none"/>
        </flexcomp>
      </body>
      <site name="floor_site" type="box" size="1 1 0.01" pos="0 0 0"/>
    </worldbody>
    <sensor>
      <contact name="flex_contact_subtree" subtree1="parent"/>
      <contact name="flex_contact_body" body1="parent"/>
      <contact name="flex_vel" geom1="floor" subtree2="parent" data="linvel angvel"/>
      <touch name="floor_touch" site="floor_site"/>
    </sensor>
  </mujoco>
  )";

  char error[1024] = {0};
  EXPECT_CALL(mock_warning_handler, Warn(testing::HasSubstr("is not rigid")))
      .WillOnce(testing::Return());
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);

  m->opt.cone = mjCONE_ELLIPTIC;
  mj_step(m.get(), d.get());
  mj_forward(m.get(), d.get());

  // We expect at least one contact between the flex and the floor
  ASSERT_GT(d->ncon, 0) << "No contacts generated";

  // Check the contact sensor "flex_contact_subtree"
  vector flex_contact_subtree =
      GetSensor(m.get(), d.get(), "flex_contact_subtree");
  EXPECT_GT(flex_contact_subtree[0], 0)
      << "Flex contact sensor (subtree) did not detect any contacts";

  // Check the contact sensor "flex_contact_body"
  vector flex_contact_body = GetSensor(m.get(), d.get(), "flex_contact_body");
  EXPECT_EQ(flex_contact_body[0], 0)
      << "Flex contact sensor (body) should not match contacts on child bodies";

  // Check flex contact velocity against solver constraint velocity
  const mjContact& con = d->contact[0];
  const mjtNum* efc_vel = d->efc_vel + con.efc_address;
  vector flex_vel = GetSensor(m.get(), d.get(), "flex_vel");
  EXPECT_THAT(vector<mjtNum>(flex_vel.begin(), flex_vel.begin() + con.dim),
              Pointwise(MjNear(1e-12, 1e-5),
                        vector<mjtNum>(efc_vel, efc_vel + con.dim)));

  // Check the touch sensor "floor_touch"
  vector floor_touch = GetSensor(m.get(), d.get(), "floor_touch");
  EXPECT_GT(floor_touch[0], 0.0)
      << "Floor touch sensor did not detect any force";
}

}  // namespace
}  // namespace mujoco
