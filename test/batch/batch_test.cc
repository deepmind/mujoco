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

// Tests for mjbatch.h.

#include <mujoco/mjbatch.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjxmacro.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using BatchTest = MujocoTest;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;

// activation dynamics, a mocap weld, a keyframe, and sensors that set mjData's
// lazy-evaluation flags (accelerometer, subtreelinvel)
static constexpr char kXml[] = R"(
<mujoco>
  <option timestep="0.002"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body name="cart" pos="0 0 .1">
      <joint name="slide" type="slide" axis="1 0 0"/>
      <geom type="box" size=".1 .1 .05" mass="1"/>
      <body name="pole" pos="0 0 .05">
        <joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0 0 .5" size=".02" mass=".1"/>
        <site name="tip" pos="0 0 .5"/>
      </body>
    </body>
    <body name="ball" pos="0 1 .5">
      <freejoint/>
      <geom type="sphere" size=".05" mass=".2"/>
    </body>
    <body name="mocap" mocap="true" pos="0 1 .5">
      <geom type="sphere" size=".02" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
  <equality>
    <weld body1="mocap" body2="ball"/>
  </equality>
  <actuator>
    <motor joint="slide" gear="5"/>
    <general joint="hinge" dyntype="filter" dynprm="0.02" gainprm="5"
      biastype="affine" biasprm="0 -5 0"/>
  </actuator>
  <sensor>
    <jointpos joint="hinge"/>
    <accelerometer site="tip"/>
    <subtreelinvel body="cart"/>
  </sensor>
  <keyframe>
    <key qpos="0.3 0.5 0 1 .5 1 0 0 0" ctrl="0.1 0.2" act="0.2" mpos="0 1 .5"/>
  </keyframe>
</mujoco>
)";

// free bodies on a plane and on each other, with elliptic cones and torsional
// friction
static constexpr char kContactXml[] = R"(
<mujoco>
  <option cone="elliptic" timestep="0.004"/>
  <worldbody>
    <geom type="plane" size="2 2 .1"/>
    <body pos="0 0 .12">
      <freejoint/>
      <geom type="box" size=".1 .1 .1" condim="6"/>
    </body>
    <body pos=".05 0 .35">
      <freejoint/>
      <geom size=".1" condim="4"/>
    </body>
    <body pos="-.05 .02 .6">
      <freejoint/>
      <geom type="capsule" size=".05 .15" euler="0 90 0"/>
    </body>
    <body pos=".3 0 .1">
      <freejoint/>
      <geom type="ellipsoid" size=".1 .08 .06"/>
    </body>
    <body pos=".3 .02 .3">
      <freejoint/>
      <geom type="cylinder" size=".08 .05"/>
    </body>
  </worldbody>
</mujoco>
)";

static constexpr int kNsim = 8;

// offset of a state component in an mjSTATE_INTEGRATION row
int Offset(const mjModel* m, mjtState component) {
  return mj_stateSize(m,
                      mjSTATE_INTEGRATION & (static_cast<int>(component) - 1));
}

std::vector<mjtNum> GetState(const mjModel* m, const mjData* d) {
  std::vector<mjtNum> row(mj_stateSize(m, mjSTATE_INTEGRATION));
  mj_getState(m, d, row.data(), mjSTATE_INTEGRATION);
  return row;
}

// expect models a and b, of the same sizes, to have the same arrays and the
// same scalars that mj_setConst writes
void ExpectSameArrays(const mjModel* a, const mjModel* b,
                      const std::string& what) {
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-variable"
#endif
  const mjModel* m = a;
  MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc)                                              \
  EXPECT_EQ(std::memcmp(a->name, b->name, sizeof(type) * m->nr * (nc)), 0) \
      << #name << ", " << what;
  MJMODEL_POINTERS
#undef X
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif
#define X(name) EXPECT_EQ(a->name, b->name) << #name << ", " << what;
  MJSETCONST_SCALARS
#undef X
}

// mjb_makeBatch with nthread threads, persistent or not
mjBatch* MakeBatch(const mjModel* m, int nsim, int nthread, int persistent,
                   char* error = nullptr, int error_sz = 0) {
  mjBatchConfig config;
  mjb_defaultConfig(&config);
  config.nthread = nthread;
  config.persistent = persistent;
  return mjb_makeBatch(m, nsim, &config, error, error_sz);
}

struct BatchPtr {
  mjBatch* b;
  explicit BatchPtr(mjBatch* batch) : b(batch) {}
  ~BatchPtr() { mjb_deleteBatch(b); }
  mjtNum* Row(int i) {
    return mjb_state(b) + static_cast<size_t>(i) * mjb_nstate(b);
  }
};

// sets a global, such as a callback or an allocator, until the end of the scope
template <typename T>
class ScopedSet {
 public:
  ScopedSet(T& global, T value) : global_(global), saved_(global) {
    global_ = value;
  }
  ~ScopedSet() { global_ = saved_; }

 private:
  T& global_;
  T saved_;
};
template <typename T>
ScopedSet(T&, T) -> ScopedSet<T>;

// 50 calls on a batch and on one mjData per simulation, with writes to every
// kind of input, a keyframe reset and subsets; the states and outputs agree
// after each
void Lockstep(int nthread, int nstep, int persistent = 0) {
  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, nthread, persistent));
  ASSERT_THAT(batch.b, NotNull());
  mjBatch* b = batch.b;
  EXPECT_EQ(mjb_nthread(b), nthread);
  EXPECT_EQ(mjb_persistent(b), persistent);

  std::vector<MjDataPtr> ref;
  for (int i = 0; i < kNsim; ++i) ref.push_back(MakeData(model));

  int size, elemsize;
  auto* sensordata =
      static_cast<mjtNum*>(mjb_output(b, "sensordata", &size, &elemsize));
  ASSERT_THAT(sensordata, NotNull());
  EXPECT_EQ(size, m->nsensordata);
  EXPECT_EQ(elemsize, sizeof(mjtNum));
  auto* site_xpos =
      static_cast<mjtNum*>(mjb_output(b, "site_xpos", &size, nullptr));
  ASSERT_THAT(site_xpos, NotNull());

  int ctrl = Offset(m, mjSTATE_CTRL), qpos = Offset(m, mjSTATE_QPOS);
  int mocap = Offset(m, mjSTATE_MOCAP_POS),
      xfrc = Offset(m, mjSTATE_XFRC_APPLIED);
  int eq = Offset(m, mjSTATE_EQ_ACTIVE);
  int ball = mj_name2id(m, mjOBJ_BODY, "ball");

  // a write lands in the batch's row and in the reference's data
  auto write = [&](int i, int offset, mjtNum value, mjtNum* field) {
    batch.Row(i)[offset] = value;
    *field = value;
  };

  for (int call = 0; call < 50; ++call) {
    for (int i = 0; i < kNsim; ++i) {
      for (int k = 0; k < m->nu; ++k) {
        write(i, ctrl + k, 0.5 * mju_sin(0.3 * call + i + k), ref[i]->ctrl + k);
      }
      if (call == 10 && i % 3 == 1)
        write(i, qpos + 1, 0.1 * i, ref[i]->qpos + 1);
      if (call == 15) write(i, mocap + 0, 0.02 * i, ref[i]->mocap_pos + 0);
      if (call == 20 && (i == 2 || i == 3)) {
        write(i, xfrc + 6 * ball + 3, 0.3, ref[i]->xfrc_applied + 6 * ball + 3);
      }
      if (call == 30 && i % 2 == 0) {
        batch.Row(i)[eq] = 0;
        ref[i]->eq_active[0] = 0;
      }
    }

    std::vector<int> ids;
    if (call == 25) {
      ids = {1, 5};
      std::vector<int> keys(ids.size(), 0);
      ASSERT_EQ(mjb_reset(b, ids.data(), ids.size(), keys.data(), mjSTAGE_ACC),
                0);
      for (int i : ids) {
        mj_resetDataKeyframe(m, ref[i].get(), 0);
        mj_forward(m, ref[i].get());
      }
    } else {
      if (call % 4 == 3) ids = {0, 2, 3, 7};
      const int* p = ids.empty() ? nullptr : ids.data();
      ASSERT_EQ(mjb_step(b, p, ids.size(), nstep, mjSTAGE_NONE), 0)
          << mjb_error(b, 0);
      for (int i = 0; i < kNsim; ++i) {
        if (!ids.empty() && std::find(ids.begin(), ids.end(), i) == ids.end())
          continue;
        for (int k = 0; k < nstep; ++k) mj_step(m, ref[i].get());
      }
    }

    // rows of simulations that did not run hold the writes not yet applied
    for (int i = 0; i < kNsim; ++i) {
      bool ran =
          ids.empty() || std::find(ids.begin(), ids.end(), i) != ids.end();
      if (!ran) continue;
      std::vector<mjtNum> expected = GetState(m, ref[i].get());
      std::vector<mjtNum> actual(batch.Row(i), batch.Row(i) + mjb_nstate(b));
      ASSERT_EQ(actual, expected) << "call " << call << " sim " << i;
      for (int k = 0; k < m->nsensordata; ++k) {
        ASSERT_EQ(sensordata[i * m->nsensordata + k], ref[i]->sensordata[k])
            << "call " << call << " sim " << i;
      }
      for (int k = 0; k < 3 * m->nsite; ++k) {
        ASSERT_EQ(site_xpos[i * 3 * m->nsite + k], ref[i]->site_xpos[k]);
      }
    }
  }
}

// a NULL config is the defaults: a thread per logical CPU, capped at nsim, and
// an mjData per thread
TEST_F(BatchTest, DefaultConfig) {
  MjModelPtr model = LoadModelFromString(kXml);
  mjBatchConfig config;
  mjb_defaultConfig(&config);
  EXPECT_EQ(config.nthread, 0);
  EXPECT_EQ(config.persistent, 0);
  BatchPtr batch(mjb_makeBatch(model.get(), 2, nullptr, nullptr, 0));
  ASSERT_THAT(batch.b, NotNull());
  int ncpu = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
  EXPECT_EQ(mjb_nthread(batch.b), std::min(ncpu, 2));
  EXPECT_EQ(mjb_persistent(batch.b), 0);
}

TEST_F(BatchTest, LockstepOneThread) { Lockstep(1, 1); }
TEST_F(BatchTest, LockstepThreeThreads) { Lockstep(3, 1); }
TEST_F(BatchTest, LockstepSevenThreadsMultiStep) { Lockstep(7, 3); }
TEST_F(BatchTest, LockstepPersistent) { Lockstep(3, 2, 1); }

// per-simulation masses with derived constants, against one model per
// simulation
TEST_F(BatchTest, ExpandAndSetConst) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  int pole = mj_name2id(m, mjOBJ_BODY, "pole");
  std::vector<std::vector<mjtNum>> finals;
  for (int nthread : {1, 4}) {
    BatchPtr batch(MakeBatch(m, kNsim, nthread, 0));
    mjBatch* b = batch.b;
    auto* mass =
        static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
    ASSERT_THAT(mass, NotNull());
    for (int i = 0; i < kNsim; ++i) mass[i * m->nbody + pole] = 0.1 + 0.2 * i;
    ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);

    // mj_setConst's outputs now have per-simulation storage
    auto* subtree = static_cast<mjtNum*>(
        mjb_expand(b, "body_subtreemass", nullptr, nullptr));
    for (int i = 0; i < kNsim; ++i) {
      EXPECT_EQ(subtree[i * m->nbody + pole], mass[i * m->nbody + pole]);
    }

    int ctrl = Offset(m, mjSTATE_CTRL);
    for (int i = 0; i < kNsim; ++i) batch.Row(i)[ctrl] = 1.0;
    ASSERT_EQ(mjb_step(b, nullptr, 0, 20, mjSTAGE_NONE), 0);
    std::vector<mjtNum> all;
    for (int i = 0; i < kNsim; ++i) {
      all.insert(all.end(), batch.Row(i), batch.Row(i) + mjb_nstate(b));
    }
    finals.push_back(all);

    for (int i = 0; i < kNsim; ++i) {
      mjModel* mi = mj_copyModel(nullptr, m);
      mi->body_mass[pole] = mass[i * m->nbody + pole];  // as stored
      mjData* d = mj_makeData(mi);
      mj_setConst(mi, d);
      mj_resetData(mi, d);
      d->ctrl[0] = 1.0;
      for (int k = 0; k < 20; ++k) mj_step(mi, d);
      EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
                GetState(mi, d))
          << "sim " << i;
      mj_deleteData(d);
      mj_deleteModel(mi);
    }
  }
  EXPECT_EQ(finals[0], finals[1]);
}

// the scalars mj_setConst writes (gravity compensation flags) are per
// simulation
TEST_F(BatchTest, GravcompIsPerSimulation) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  int pole = mj_name2id(m, mjOBJ_BODY, "pole");
  BatchPtr batch(MakeBatch(m, kNsim, 3, 0));
  mjBatch* b = batch.b;
  auto* gravcomp =
      static_cast<mjtNum*>(mjb_expand(b, "body_gravcomp", nullptr, nullptr));
  for (int i = 0; i < kNsim; i += 2) gravcomp[i * m->nbody + pole] = 1;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  int qpos = Offset(m, mjSTATE_QPOS), qvel = Offset(m, mjSTATE_QVEL);
  for (int i = 0; i < kNsim; ++i) batch.Row(i)[qpos + 1] = 0.3;
  ASSERT_EQ(mjb_step(b, nullptr, 0, 5, mjSTAGE_NONE), 0);
  for (int i = 0; i < kNsim; i += 2) {
    EXPECT_NE(batch.Row(i)[qvel + 1], batch.Row(i + 1)[qvel + 1]);
    EXPECT_EQ(batch.Row(i)[qvel + 1], batch.Row(0)[qvel + 1]);
  }
}

// reset uses each simulation's expanded qpos0 and discards writes made before
// it
TEST_F(BatchTest, ResetExpandedQpos0) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 2, 0));
  mjBatch* b = batch.b;
  auto* qpos0 = static_cast<mjtNum*>(mjb_expand(b, "qpos0", nullptr, nullptr));
  for (int i = 0; i < kNsim; ++i) qpos0[i * m->nq + 1] = 0.1 * i;
  auto* xpos = static_cast<mjtNum*>(mjb_output(b, "xpos", nullptr, nullptr));
  int qpos = Offset(m, mjSTATE_QPOS);
  batch.Row(5)[qpos] = 0.25;  // discarded by the reset
  int ids[] = {2, 5};
  ASSERT_EQ(mjb_reset(b, ids, 2, nullptr, mjSTAGE_ACC), 0);
  for (int i : ids) {
    EXPECT_EQ(batch.Row(i)[qpos], 0);
    EXPECT_EQ(batch.Row(i)[qpos + 1], qpos0[i * m->nq + 1]);
    EXPECT_NE(xpos[i * 3 * m->nbody + 3 * 1 + 2], 0);  // reset runs mj_forward
  }
  EXPECT_EQ(xpos[0 * 3 * m->nbody + 3 * 1 + 2],
            0);  // not run: output not filled yet
}

// a MuJoCo error in one simulation is reported; the others run, and it keeps
// its state
TEST_F(BatchTest, ErrorKeepsFailingSimulation) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  for (int nthread : {1, 3}) {
    BatchPtr batch(MakeBatch(m, kNsim, nthread, 0));
    mjBatch* b = batch.b;
    auto* eq_type =
        static_cast<int*>(mjb_expand(b, "eq_type", nullptr, nullptr));
    eq_type[2] = 99;
    std::vector<mjtNum> before(batch.Row(2), batch.Row(2) + mjb_nstate(b));
    EXPECT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_NONE), 1);
    EXPECT_NE(mjb_status(b)[2], 0);
    EXPECT_EQ(mjb_status(b)[0], 0);
    EXPECT_THAT(std::string(mjb_error(b, 2)), Not(testing::IsEmpty()));
    EXPECT_STREQ(mjb_error(b, 0), "");
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(2), batch.Row(2) + mjb_nstate(b)),
              before);
    EXPECT_GT(batch.Row(0)[0], 0);  // time advanced in the others
    eq_type[2] = m->eq_type[0];
    EXPECT_EQ(mjb_step(b, nullptr, 0, 1, mjSTAGE_NONE), 0);
    EXPECT_STREQ(mjb_error(b, 2), "");
    EXPECT_EQ(mjb_status(b)[2], 0);
    EXPECT_EQ(batch.Row(2)[0], m->opt.timestep);
    EXPECT_EQ(batch.Row(0)[0], 4 * m->opt.timestep);
  }
}

TEST_F(BatchTest, SleepIsRejected) {
  std::string xml = kXml;
  xml.replace(xml.find("<option"), 7,
              "<option><flag sleep=\"enable\"/></option><option");
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  char error[256];
  EXPECT_THAT(MakeBatch(model.get(), kNsim, 1, 0, error, sizeof(error)),
              IsNull());
  EXPECT_THAT(std::string(error), HasSubstr("persistent"));
  BatchPtr batch(MakeBatch(model.get(), kNsim, 1, 1, error, sizeof(error)));
  EXPECT_THAT(batch.b, NotNull());
}

TEST_F(BatchTest, OutputsAndExpandRefuse) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 1, 0));
  EXPECT_THAT(mjb_output(batch.b, "qpos", nullptr, nullptr),
              IsNull());  // in the state
  EXPECT_THAT(mjb_output(batch.b, "nope", nullptr, nullptr), IsNull());
  EXPECT_THAT(mjb_expand(batch.b, "mesh_vert", nullptr, nullptr),
              IsNull());  // asset
  EXPECT_THAT(mjb_expand(batch.b, "nope", nullptr, nullptr), IsNull());
}

TEST_F(BatchTest, BadIds) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 2, 0));
  int unsorted[] = {3, 1};
  EXPECT_THAT(
      MjuErrorMessageFrom(mjb_step)(batch.b, unsorted, 2, 1, mjSTAGE_NONE),
      HasSubstr("mjb_step: simulation ids must be sorted, unique and in "
                "[0, 8); ids[1] is 1"));
  int out_of_range[] = {kNsim};
  EXPECT_THAT(
      MjuErrorMessageFrom(mjb_forward)(batch.b, out_of_range, 1, mjSTAGE_ACC),
      HasSubstr("mjb_forward: "));
  // the batch is still usable: no lock was left held
  EXPECT_EQ(mjb_step(batch.b, nullptr, 0, 1, mjSTAGE_NONE), 0);
}

// apply runs any function on each simulation; with save=0 the state is
// untouched
TEST_F(BatchTest, Apply) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 3, 0));
  mjBatch* b = batch.b;
  int qpos = Offset(m, mjSTATE_QPOS);
  for (int i = 0; i < kNsim; ++i) batch.Row(i)[qpos + 1] = 0.1 * i;
  std::vector<mjtNum> before(mjb_state(b),
                             mjb_state(b) + kNsim * mjb_nstate(b));

  std::vector<mjtNum> jacp(kNsim * 3 * m->nv);
  auto jac_tip = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    mj_kinematics(m, d);
    mj_comPos(m, d);
    mjtNum* out = static_cast<mjtNum*>(arg) + sim * 3 * m->nv;
    mj_jacSite(m, d, out, nullptr, mj_name2id(m, mjOBJ_SITE, "tip"));
  };
  ASSERT_EQ(mjb_apply(b, nullptr, 0, jac_tip, jacp.data(), 0), 0);
  EXPECT_EQ(
      std::vector<mjtNum>(mjb_state(b), mjb_state(b) + kNsim * mjb_nstate(b)),
      before);

  MjDataPtr d = MakeData(model);
  std::vector<mjtNum> expected(3 * m->nv);
  for (int i = 0; i < kNsim; ++i) {
    mj_setState(m, d.get(), batch.Row(i), mjSTATE_INTEGRATION);
    mj_kinematics(m, d.get());
    mj_comPos(m, d.get());
    mj_jacSite(m, d.get(), expected.data(), nullptr,
               mj_name2id(m, mjOBJ_SITE, "tip"));
    EXPECT_EQ(std::vector<mjtNum>(jacp.begin() + i * 3 * m->nv,
                                  jacp.begin() + (i + 1) * 3 * m->nv),
              expected)
        << "sim " << i;
  }
}

// apply publishes no outputs: a callback that computes only some fields must
// not overwrite rows with what the worker's data holds from another simulation
TEST_F(BatchTest, ApplyLeavesOutputsAlone) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body>
        <joint name="j" type="slide" axis="0 0 1"/>
        <geom size=".1"/>
      </body>
    </worldbody>
    <sensor>
      <jointvel joint="j"/>
    </sensor>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, 4, 1, 0));
  mjBatch* b = batch.b;
  auto* sensordata =
      static_cast<mjtNum*>(mjb_output(b, "sensordata", nullptr, nullptr));
  int qvel = Offset(m, mjSTATE_QVEL);
  for (int i = 0; i < 4; ++i) batch.Row(i)[qvel] = i + 1;
  ASSERT_EQ(mjb_forward(b, nullptr, 0, mjSTAGE_ACC), 0);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(sensordata[i], i + 1);
  auto kinematics =
      +[](const mjModel* m, mjData* d, int, void*) { mj_kinematics(m, d); };
  ASSERT_EQ(mjb_apply(b, nullptr, 0, kinematics, nullptr, 0), 0);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(sensordata[i], i + 1);
}

// a simulation that fails set_const does not cost the others their derived
// constants
TEST_F(BatchTest, SetConstErrorKeepsOthersComplete) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option timestep="0.002"/>
    <worldbody>
      <body pos="0 0 1">
        <joint type="slide" axis="0 0 1"/>
        <geom size=".1" mass="1" contype="0" conaffinity="0"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, 2, 1, 0));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  mass[0 * m->nbody + 1] = 2;
  mass[1 * m->nbody + 1] = 0;  // singular: mj_setConst raises in simulation 1

  // the singular simulation also warns, on the caller's thread (nthread is 1)
  mock_warning_handler.ExpectWarnings("singular");
  EXPECT_EQ(mjb_setConst(b, nullptr, 0), 1);
  EXPECT_EQ(mjb_status(b)[0], 0);
  EXPECT_NE(mjb_status(b)[1], 0);
  int ids[] = {0};
  ASSERT_EQ(mjb_step(b, ids, 1, 1, mjSTAGE_NONE), 0);

  mjModel* ref = mj_copyModel(nullptr, m);
  ref->body_mass[1] = 2;
  mjData* d = mj_makeData(ref);
  mj_setConst(ref, d);
  mj_resetData(ref, d);
  mj_step(ref, d);
  EXPECT_EQ(std::vector<mjtNum>(batch.Row(0), batch.Row(0) + mjb_nstate(b)),
            GetState(ref, d));
  mj_deleteData(d);
  mj_deleteModel(ref);
}

// a persistent batch keeps one mjData per simulation, so sleeping works: trees
// fall asleep and wake on writes exactly as in a loop over one mjData per
// simulation
TEST_F(BatchTest, PersistentSleepMatchesLoop) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" viscosity="10" sleep_tolerance="0.01">
      <flag sleep="enable" gravity="disable" constraint="disable"
        contact="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1" euler="10 20 30"/>
      </body>
      <body pos="1 0 0">
        <freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  constexpr int nsim = 4;
  BatchPtr batch(MakeBatch(m, nsim, 2, 1));
  ASSERT_THAT(batch.b, NotNull());
  mjBatch* b = batch.b;
  std::vector<MjDataPtr> ref;
  int qvel = Offset(m, mjSTATE_QVEL);
  for (int i = 0; i < nsim; ++i) {
    ref.push_back(MakeData(model));
    for (int k = 0; k < m->nv; ++k) {
      mjtNum v = 0.1 * (i + 1) * (k % 3 + 1);
      batch.Row(i)[qvel + k] = v;
      ref[i]->qvel[k] = v;
    }
  }
  bool slept = false;
  for (int call = 0; call < 100; ++call) {
    if (call == 60) {  // a write to a sleeping simulation wakes it
      batch.Row(1)[qvel] = 0.7;
      ref[1]->qvel[0] = 0.7;
    }
    ASSERT_EQ(mjb_step(b, nullptr, 0, 10, mjSTAGE_NONE), 0) << mjb_error(b, 0);
    for (int i = 0; i < nsim; ++i) {
      for (int k = 0; k < 10; ++k) mj_step(m, ref[i].get());
      slept = slept || ref[i]->ntree_awake < m->ntree;
      ASSERT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
                GetState(m, ref[i].get()))
          << "call " << call << " sim " << i;
    }
  }
  EXPECT_TRUE(slept) << "no tree fell asleep: the test does not exercise sleep";
}

// sleep enabled per simulation through mjOption needs a persistent batch too
TEST_F(BatchTest, ExpandedSleepFlagNeedsPersistent) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 1, 0));
  mjBatch* b = batch.b;
  auto* enableflags =
      static_cast<int*>(mjb_expand(b, "opt.enableflags", nullptr, nullptr));
  ASSERT_THAT(enableflags, NotNull());
  enableflags[3] |= mjENBL_SLEEP;
  EXPECT_EQ(mjb_step(b, nullptr, 0, 1, mjSTAGE_NONE), 1);
  EXPECT_NE(mjb_status(b)[3], 0);
  EXPECT_THAT(std::string(mjb_error(b, 3)), HasSubstr("persistent"));
}

// mjOption fields are expandable: per-simulation gravity, against one model
// each
TEST_F(BatchTest, ExpandOption) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 3, 0));
  mjBatch* b = batch.b;
  int size, elemsize;
  auto* gravity =
      static_cast<mjtNum*>(mjb_expand(b, "opt.gravity", &size, &elemsize));
  ASSERT_THAT(gravity, NotNull());
  EXPECT_EQ(size, 3);
  EXPECT_EQ(elemsize, sizeof(mjtNum));
  EXPECT_THAT(mjb_expand(b, "opt.timestep", &size, nullptr), NotNull());
  EXPECT_EQ(size, 1);
  for (int i = 0; i < kNsim; ++i) gravity[3 * i + 2] = -1.0 * i;
  ASSERT_EQ(mjb_step(b, nullptr, 0, 10, mjSTAGE_NONE), 0);
  for (int i = 0; i < kNsim; ++i) {
    mjModel* mi = mj_copyModel(nullptr, m);
    mi->opt.gravity[2] = gravity[3 * i + 2];
    mjData* d = mj_makeData(mi);
    for (int k = 0; k < 10; ++k) mj_step(mi, d);
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(mi, d))
        << "sim " << i;
    mj_deleteData(d);
    mj_deleteModel(mi);
  }
}

// every simulation that raises is reported, with its own message
TEST_F(BatchTest, AllFailuresReported) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 3, 0));
  mjBatch* b = batch.b;
  auto* eq_type = static_cast<int*>(mjb_expand(b, "eq_type", nullptr, nullptr));
  eq_type[1] = eq_type[5] = 99;
  EXPECT_EQ(mjb_step(b, nullptr, 0, 1, mjSTAGE_NONE), 2);
  for (int i = 0; i < kNsim; ++i) {
    bool failed = i == 1 || i == 5;
    EXPECT_EQ(mjb_status(b)[i] != 0, failed) << "sim " << i;
    EXPECT_EQ(std::string(mjb_error(b, i)).empty(), !failed) << "sim " << i;
  }
}

// mjb_rollout is a loop of mj_setState(control) and mj_step with records after
// each substep, on the selected simulations, in call order
TEST_F(BatchTest, RolloutMatchesLoop) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  constexpr int nstep = 6;
  for (int nthread : {1, 3}) {
    BatchPtr batch(MakeBatch(m, kNsim, nthread, 0));
    mjBatch* b = batch.b;
    int ids[] = {1, 4, 6};
    int n = 3;
    int spec = mjSTATE_CTRL | mjSTATE_QFRC_APPLIED;
    int ncontrol = mj_stateSize(m, spec);
    std::vector<mjtNum> control(n * nstep * ncontrol);
    for (int k = 0; k < static_cast<int>(control.size()); ++k)
      control[k] = 0.3 * mju_sin(0.7 * k);
    int nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS);
    std::vector<mjtNum> states(n * nstep * nstate),
        sensors(n * nstep * m->nsensordata);
    mjBatchRecord record[] = {{nullptr, mjSTATE_FULLPHYSICS, states.data()},
                              {"sensordata", 0, sensors.data()}};
    ASSERT_EQ(mjb_rollout(b, ids, n, nstep, control.data(), spec, record, 2),
              0);

    for (int j = 0; j < n; ++j) {
      MjDataPtr d = MakeData(model);
      std::vector<mjtNum> row(nstate);
      for (int k = 0; k < nstep; ++k) {
        mj_setState(m, d.get(), control.data() + (j * nstep + k) * ncontrol,
                    spec);
        mj_step(m, d.get());
        mj_getState(m, d.get(), row.data(), mjSTATE_FULLPHYSICS);
        size_t at = j * nstep + k;
        EXPECT_EQ(std::vector<mjtNum>(states.begin() + at * nstate,
                                      states.begin() + (at + 1) * nstate),
                  row)
            << "j " << j << " k " << k;
        EXPECT_EQ(
            std::vector<mjtNum>(sensors.begin() + at * m->nsensordata,
                                sensors.begin() + (at + 1) * m->nsensordata),
            std::vector<mjtNum>(d->sensordata, d->sensordata + m->nsensordata));
      }
      // the row continues from the rollout's last state
      EXPECT_EQ(std::vector<mjtNum>(batch.Row(ids[j]),
                                    batch.Row(ids[j]) + mjb_nstate(b)),
                GetState(m, d.get()));
    }
    EXPECT_EQ(batch.Row(0)[0], 0);  // not selected: did not run
  }
}

// a rollout on a persistent batch starts afresh, though an earlier call left
// stages to continue from, and leaves none for the next step
TEST_F(BatchTest, RolloutOnPersistentBatch) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  constexpr int nsim = 2, nstep = 5;
  int spec = mjSTATE_CTRL;
  int ncontrol = mj_stateSize(m, spec);
  std::vector<mjtNum> control(nsim * nstep * ncontrol);
  for (int k = 0; k < static_cast<int>(control.size()); ++k)
    control[k] = 0.3 * mju_sin(0.7 * k);
  int nstate = mj_stateSize(m, mjSTATE_FULLPHYSICS);
  std::vector<std::vector<mjtNum>> states, rows;
  for (int persistent : {0, 1}) {
    BatchPtr batch(MakeBatch(m, nsim, 1, persistent));
    mjBatch* b = batch.b;
    ASSERT_EQ(mjb_forward(b, nullptr, 0, mjSTAGE_ACC), 0);
    std::vector<mjtNum> recorded(nsim * nstep * nstate);
    mjBatchRecord record = {nullptr, mjSTATE_FULLPHYSICS, recorded.data()};
    ASSERT_EQ(
        mjb_rollout(b, nullptr, 0, nstep, control.data(), spec, &record, 1), 0);
    ASSERT_EQ(mjb_step(b, nullptr, 0, 2, mjSTAGE_VEL), 0);
    states.push_back(recorded);
    rows.emplace_back(batch.Row(0), batch.Row(0) + nsim * mjb_nstate(b));
  }
  EXPECT_EQ(states[1], states[0]);
  EXPECT_EQ(rows[1], rows[0]);
}

// a simulation whose warning counters rise stops, and its remaining records
// repeat
TEST_F(BatchTest, RolloutStopsAtWarning) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, 2, 1, 0));  // on the caller's thread
  mjBatch* b = batch.b;
  batch.Row(0)[Offset(m, mjSTATE_QPOS)] =
      std::numeric_limits<mjtNum>::quiet_NaN();
  mock_warning_handler.ExpectWarnings("QPOS");
  constexpr int nstep = 4;
  std::vector<mjtNum> qpos(2 * nstep * m->nq);
  mjBatchRecord record[] = {{"qpos", 0, qpos.data()}};
  ASSERT_EQ(mjb_rollout(b, nullptr, 0, nstep, nullptr, 0, record, 1), 0);
  for (int k = 1; k < nstep; ++k) {
    EXPECT_EQ(std::vector<mjtNum>(qpos.begin() + k * m->nq,
                                  qpos.begin() + (k + 1) * m->nq),
              std::vector<mjtNum>(qpos.begin(), qpos.begin() + m->nq))
        << "k " << k;
  }
  // the other simulation ran all its steps
  EXPECT_NE(std::vector<mjtNum>(qpos.begin() + (nstep + 1) * m->nq,
                                qpos.begin() + (nstep + 2) * m->nq),
            std::vector<mjtNum>(qpos.begin() + nstep * m->nq,
                                qpos.begin() + (nstep + 1) * m->nq));
}

// thread models share the asset arrays with the batch's model: a mesh and a
// heightfield in contact, with per-simulation masses and constants, against one
// model each
TEST_F(BatchTest, AssetsAreShared) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
      <hfield name="bumps" nrow="3" ncol="3" size="1 1 .1 .1"
        elevation="0 1 0 1 0 1 0 1 0"/>
    </asset>
    <worldbody>
      <geom type="hfield" hfield="bumps"/>
      <body pos="0 0 .5">
        <freejoint/>
        <geom type="mesh" mesh="tet"/>
      </body>
      <body pos=".3 .3 .6">
        <freejoint/>
        <geom type="box" size=".05 .05 .05"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  constexpr int nsim = 6;
  BatchPtr batch(MakeBatch(m, nsim, 3, 0));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  for (int i = 0; i < nsim; ++i) mass[i * m->nbody + 1] = 0.5 + 0.25 * i;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  ASSERT_EQ(mjb_step(b, nullptr, 0, 200, mjSTAGE_NONE), 0);
  for (int i = 0; i < nsim; ++i) {
    mjModel* mi = mj_copyModel(nullptr, m);
    mi->body_mass[1] = mass[i * m->nbody + 1];
    mjData* d = mj_makeData(mi);
    mj_setConst(mi, d);
    mj_resetData(mi, d);
    for (int k = 0; k < 200; ++k) mj_step(mi, d);
    EXPECT_GT(d->ncon, 0)
        << "no contact: the test does not exercise the assets";
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(mi, d))
        << "sim " << i;
    mj_deleteData(d);
    mj_deleteModel(mi);
  }
}

// the thread models share the arrays that nothing writes: in every call, each
// array that is neither expanded nor written by mj_setConst holds the model's
// values
TEST_F(BatchTest, ThreadModelsKeepUnwrittenArrays) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 3, 0));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  auto* size =
      static_cast<mjtNum*>(mjb_expand(b, "geom_size", nullptr, nullptr));
  for (int i = 0; i < kNsim; ++i) {
    mass[i * m->nbody + 1] *= 1 + 0.1 * i;
    size[i * 3 * m->ngeom + 3] *= 1 + 0.05 * i;
  }
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  ASSERT_EQ(mjb_step(b, nullptr, 0, 5, mjSTAGE_NONE), 0);

  struct Args {
    const mjModel* model;
    std::mutex mu;
    std::set<std::string> changed;  // unwritten arrays which differ
  } args;
  args.model = m;
  auto check = +[](const mjModel* mi, mjData*, int, void* arg) {
    auto* a = static_cast<Args*>(arg);
    const mjModel* m = a->model;
    std::set<std::string> written = {"body_mass", "geom_size"};
#define X(name, fixed, kin0, inert0, spring)                 \
  if (((fixed) | (kin0) | (inert0) | (spring)) & mjSC_OUT) { \
    written.insert(#name);                                   \
  }
    MJSETCONST_ARRAYS
#undef X
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wunused-variable"
#endif
    MJMODEL_POINTERS_PREAMBLE(m)
#define X(type, name, nr, nc)                                        \
  if (!written.count(#name) &&                                       \
      std::memcmp(mi->name, m->name, sizeof(type) * m->nr * (nc))) { \
    std::lock_guard<std::mutex> lock(a->mu);                         \
    a->changed.insert(#name);                                        \
  }
    MJMODEL_POINTERS
#undef X
#if defined(__clang__) || defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif
  };
  ASSERT_EQ(mjb_apply(b, nullptr, 0, check, &args, 0), 0);
  EXPECT_THAT(args.changed, ::testing::IsEmpty());
}

// mj_setConst clears the sleep flag while it runs and does not restore it if it
// raises; the thread's model still has the model's options for the simulation
// that follows on it
TEST_F(BatchTest, SetConstErrorKeepsOptions) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option>
      <flag sleep="enable"/>
    </option>
    <worldbody>
      <body>
        <joint type="slide" axis="1 0 0" armature="1"/>
        <geom size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  ASSERT_EQ(m->body_simple[1], 2);        // translation only: no mass raises
  BatchPtr batch(MakeBatch(m, 2, 1, 1));  // one thread: sim 1 follows sim 0
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  mass[1] = 0;  // sim 0: a moving body with no mass raises in mj_setConst
  EXPECT_EQ(mjb_setConst(b, nullptr, 0), 1);
  EXPECT_THAT(mjb_error(b, 0), HasSubstr("0 mass"));
  EXPECT_EQ(mjb_status(b)[1], 0);
  int flags[2] = {0, 0};
  auto record = +[](const mjModel* m, mjData*, int sim, void* arg) {
    static_cast<int*>(arg)[sim] = m->opt.enableflags;
  };
  ASSERT_EQ(mjb_apply(b, nullptr, 0, record, flags, 0), 0);
  EXPECT_EQ(flags[0], m->opt.enableflags);
  EXPECT_EQ(flags[1], m->opt.enableflags);
}

// set_const does not depend on what another simulation left in a shared
// mjData: its contacts, which the moment of an adhesion actuator averages, and
// its mocap pose, from which a connect anchor is computed
TEST_F(BatchTest, SetConstIgnoresSharedData) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body name="box" pos="0 0 .1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
      <body name="target" mocap="true" pos="0 0 .5"/>
    </worldbody>
    <equality>
      <connect body1="target" body2="box" anchor="0 0 0" active="false"/>
    </equality>
    <actuator>
      <adhesion body="box" ctrlrange="0 1"/>
    </actuator>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, 2, 1, 0));  // one thread: one shared mjData
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  mass[m->nbody + 1] *= 2;  // sim 1

  // sim 0 moves its mocap body and steps into contact
  int mocap = Offset(m, mjSTATE_MOCAP_POS);
  batch.Row(0)[mocap] = 0.3;
  int ids0[] = {0};
  ASSERT_EQ(mjb_step(b, ids0, 1, 1, mjSTAGE_NONE), 0);
  int ids1[] = {1};
  ASSERT_EQ(mjb_setConst(b, ids1, 1), 0);

  MjModelPtr got(mj_copyModel(nullptr, m));
  mjb_getModel(b, 1, got.get());
  MjModelPtr want(mj_copyModel(nullptr, m));
  want->body_mass[1] *= 2;
  MjDataPtr d(mj_makeData(want.get()));
  mj_setConst(want.get(), d.get());
  ExpectSameArrays(got.get(), want.get(), "sim 1");
}

// mjb_setConst runs only the groups of mj_setConst that read a per-simulation
// field. For each mjtNum array mj_setConst reads, perturbed in the second of
// two simulations, the models are those of a batch which also expands an
// option, so that every group runs
void ExpectSetConstGroupsMatchAllGroups(const mjModel* m,
                                        const std::string& path) {
  constexpr int nsim = 2;
  int persistent = (m->opt.enableflags & (mjENBL_SLEEP | mjENBL_IPC)) != 0;
  MjModelPtr got(mj_copyModel(nullptr, m));
  MjModelPtr want(mj_copyModel(nullptr, m));
  char error[1000] = "";
  std::set<std::string> num;  // arrays of mjtNum, whatever its size
#define X(type, name, nr, nc) \
  if (std::is_same_v<type, mjtNum>) num.insert(#name);
  MJMODEL_POINTERS
#undef X
  int ntested = 0;
#define X(name, fixed, kin0, inert0, spring)                                 \
  if (((fixed) | (kin0) | (inert0) | (spring)) & mjSC_IN) {                  \
    BatchPtr batch(MakeBatch(m, nsim, 1, persistent, error, sizeof(error))); \
    ASSERT_THAT(batch.b, NotNull()) << path << ": " << error;                \
    BatchPtr all(MakeBatch(m, nsim, 1, persistent));                         \
    int size = 0, elemsize = 0;                                              \
    auto* a =                                                                \
        static_cast<mjtNum*>(mjb_expand(batch.b, #name, &size, &elemsize));  \
    auto* b =                                                                \
        static_cast<mjtNum*>(mjb_expand(all.b, #name, nullptr, nullptr));    \
    if (a && size && num.count(#name)) {                                     \
      ntested++;                                                             \
      for (int k = 0; k < size; ++k) b[size + k] = a[size + k] *= 1.01;      \
      mjb_expand(all.b, "opt.timestep", nullptr, nullptr);                   \
      EXPECT_EQ(mjb_setConst(batch.b, nullptr, 0),                           \
                mjb_setConst(all.b, nullptr, 0))                             \
          << #name << " in " << path;                                        \
      for (int i = 0; i < nsim; ++i) {                                       \
        EXPECT_EQ(mjb_status(batch.b)[i], mjb_status(all.b)[i])              \
            << #name << " in " << path;                                      \
        mjb_getModel(batch.b, i, got.get());                                 \
        mjb_getModel(all.b, i, want.get());                                  \
        ExpectSameArrays(got.get(), want.get(),                              \
                         path + " " + #name + " sim " + std::to_string(i));  \
      }                                                                      \
    }                                                                        \
  }
  MJSETCONST_ARRAYS
#undef X
  EXPECT_GT(ntested, 0) << path;
}

TEST_F(BatchTest, SetConstGroupsMatchAllGroups) {
  static constexpr char kRichXml[] = R"(
  <mujoco>
    <worldbody>
      <light pos="0 0 3" mode="targetbody" target="arm"/>
      <camera pos="1 1 1" mode="targetbodycom" target="arm"/>
      <body name="arm" pos="0 0 1">
        <joint name="slide" type="slide" axis="0 0 1" springref=".1"
          stiffness="10" armature=".1"/>
        <geom size=".1" mass="1"/>
        <site name="s1" pos=".1 0 0"/>
        <body pos=".3 0 0">
          <joint name="hinge" axis="0 1 0" springref=".2" stiffness="1"/>
          <geom type="capsule" fromto="0 0 0 .3 0 0" size=".05"/>
          <site name="s2" pos=".3 0 0"/>
        </body>
      </body>
      <body pos="1 0 1">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
    <tendon>
      <spatial stiffness="5" springlength=".2">
        <site site="s1"/>
        <site site="s2"/>
      </spatial>
    </tendon>
    <equality>
      <connect body1="arm" anchor=".1 0 0"/>
    </equality>
    <actuator>
      <position joint="hinge" kp="10"/>
    </actuator>
  </mujoco>
  )";

  ExpectSetConstGroupsMatchAllGroups(LoadModelFromString(kXml).get(), "kXml");
  ExpectSetConstGroupsMatchAllGroups(LoadModelFromString(kRichXml).get(),
                                     "kRichXml");
}

// the same, on the models on which engine_setconst_test sees every mark of the
// table; under model/ or the test data
using SetConstGroupsCorpusTest = ::testing::TestWithParam<const char*>;

TEST_P(SetConstGroupsCorpusTest, MatchAllGroups) {
  std::string path = GetParam();
  std::string file = path.substr(0, 6) == "model/"
                         ? GetModelPath(path.substr(6))
                         : GetTestDataFilePath(path);
  MjModelPtr model(mj_loadXML(file.c_str(), nullptr, nullptr, 0));
  ASSERT_THAT(model, NotNull()) << path;
  ExpectSetConstGroupsMatchAllGroups(model.get(), path);
}

INSTANTIATE_TEST_SUITE_P(
    Corpus, SetConstGroupsCorpusTest,
    ::testing::Values(
        "model/flex/bag.xml", "model/flex/mechanisms/cantilever.xml",
        "model/flex/mechanisms/dipper.xml", "model/flex/poncho.xml",
        "model/slider_crank/slider_crank.xml",
        "engine/testdata/core_constraint/dofless_contact.xml",
        "engine/testdata/derivative/dcmotor.xml",
        "engine/testdata/flex_setconst.xml",
        "engine/testdata/sleep/init_island.xml", "testdata/model.xml",
        "user/testdata/flex_line_obj.xml", "user/testdata/mesh.xml"));

// a function run by mjb_apply may call another batch: the enclosing
// simulation's error trap and the caller's log handler survive the nested call
TEST_F(BatchTest, ApplyCallsAnotherBatch) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr outer(MakeBatch(m, kNsim, 2, 0));
  BatchPtr inner(MakeBatch(m, kNsim, 2, 0));
  auto step_inner = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    mjb_step(static_cast<mjBatch*>(arg), &sim, 1, 1, mjSTAGE_NONE);
    if (sim == 2) mju_error("raised after the nested call");
  };
  mjfLogHandler handler = _mjPRIVATE_setTlsLogHandler(nullptr);
  _mjPRIVATE_setTlsLogHandler(handler);

  EXPECT_EQ(mjb_apply(outer.b, nullptr, 0, step_inner, inner.b, 0), 1);
  EXPECT_EQ(_mjPRIVATE_setTlsLogHandler(handler), handler);
  for (int i = 0; i < kNsim; ++i) {
    EXPECT_EQ(mjb_status(outer.b)[i], i == 2) << "sim " << i;
    EXPECT_EQ(mjb_status(inner.b)[i], 0) << "sim " << i;
    EXPECT_EQ(inner.Row(i)[0], m->opt.timestep) << "sim " << i;
  }
  EXPECT_THAT(mjb_error(outer.b, 2), HasSubstr("raised after the nested call"));
}

// a function run by mjb_apply that calls its own batch fails instead of
// deadlocking
TEST_F(BatchTest, ApplyCallingItsOwnBatchFails) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 2, 0));
  auto step_self = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    mjb_step(static_cast<mjBatch*>(arg), &sim, 1, 1, mjSTAGE_NONE);
  };
  EXPECT_EQ(mjb_apply(batch.b, nullptr, 0, step_self, batch.b, 0), kNsim);
  EXPECT_THAT(mjb_error(batch.b, 0), HasSubstr("its own batch"));
  EXPECT_EQ(batch.Row(0)[0], 0);
}

// set_const in a persistent batch computes on scratch data: a sleeping
// simulation's mjData would skip its trees, and must keep its sleep state
TEST_F(BatchTest, PersistentSetConstIgnoresSleep) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" viscosity="10" sleep_tolerance="0.01">
      <flag sleep="enable" gravity="disable" constraint="disable"
        contact="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  constexpr int nsim = 2;
  BatchPtr batch(MakeBatch(m, nsim, 1, 1));
  ASSERT_THAT(batch.b, NotNull());
  mjBatch* b = batch.b;
  int qvel = Offset(m, mjSTATE_QVEL);
  for (int i = 0; i < nsim; ++i) batch.Row(i)[qvel] = 0.1;
  ASSERT_EQ(mjb_step(b, nullptr, 0, 1000, mjSTAGE_NONE), 0);

  auto awake = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    static_cast<int*>(arg)[sim] = d->ntree_awake;
  };
  int ntree_awake[nsim];
  ASSERT_EQ(mjb_apply(b, nullptr, 0, awake, ntree_awake, 0), 0);
  ASSERT_EQ(ntree_awake[0], 0) << "the test needs a sleeping simulation";

  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  mass[1 * m->nbody + 1] = 2;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);

  mjModel* m2 = mj_copyModel(nullptr, m);
  m2->body_mass[1] = 2;
  mjData* d2 = mj_makeData(m2);
  mj_setConst(m2, d2);
  auto* invweight =
      static_cast<mjtNum*>(mjb_expand(b, "body_invweight0", nullptr, nullptr));
  ASSERT_THAT(invweight, NotNull());
  EXPECT_EQ(invweight[0 * 2 * m->nbody + 2], m->body_invweight0[2]);
  EXPECT_EQ(invweight[1 * 2 * m->nbody + 2], m2->body_invweight0[2]);
  EXPECT_NE(m2->body_invweight0[2], m->body_invweight0[2]);
  mj_deleteData(d2);
  mj_deleteModel(m2);

  ASSERT_EQ(mjb_apply(b, nullptr, 0, awake, ntree_awake, 0), 0);
  EXPECT_EQ(ntree_awake[0], 0) << "set_const woke a simulation";
}

// a field that set_const expanded becomes an input once expanded explicitly:
// connect anchors written to eq_data survive the next set_const
TEST_F(BatchTest, ExpandedSetConstFieldIsAnInput) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a" pos="0 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
      <body name="b" pos=".5 0 1">
        <geom size=".1"/>
      </body>
    </worldbody>
    <equality>
      <connect body1="a" body2="b" anchor=".25 0 0"/>
    </equality>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  int body_b = mj_name2id(m, mjOBJ_BODY, "b");
  BatchPtr batch(MakeBatch(m, 2, 1, 0));
  mjBatch* b = batch.b;
  auto* pos = static_cast<mjtNum*>(mjb_expand(b, "body_pos", nullptr, nullptr));
  pos[1 * 3 * m->nbody + 3 * body_b] = 0.7;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0),
            0);  // expands eq_data: its body2 anchor moved

  auto* eq_data =
      static_cast<mjtNum*>(mjb_expand(b, "eq_data", nullptr, nullptr));
  mjtNum* data1 = eq_data + 1 * mjNEQDATA * m->neq;
  ASSERT_NE(data1[3], m->eq_data[3]);
  const mjtNum anchor = 0.3;
  data1[0] = anchor;  // the first anchor of simulation 1
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  EXPECT_EQ(data1[0], anchor);

  mjModel* ref = mj_copyModel(nullptr, m);
  ref->body_pos[3 * body_b] = pos[1 * 3 * m->nbody + 3 * body_b];
  ref->eq_data[0] = anchor;
  mjData* d = mj_makeData(ref);
  mj_setConst(ref, d);
  EXPECT_EQ(std::vector<mjtNum>(data1, data1 + mjNEQDATA),
            std::vector<mjtNum>(ref->eq_data, ref->eq_data + mjNEQDATA));
  mj_deleteData(d);
  mj_deleteModel(ref);
}

// mjb_expanded reads a field's per-simulation values without expanding it;
// a field that set_const derived stays derived, recomputed by the next one
TEST_F(BatchTest, ExpandedReadsWithoutExpanding) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <body name="a" pos="0 0 1">
        <freejoint/>
        <geom size=".1"/>
      </body>
      <body name="b" pos=".5 0 1">
        <geom size=".1"/>
      </body>
    </worldbody>
    <equality>
      <connect body1="a" body2="b" anchor=".25 0 0"/>
    </equality>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  int body_b = mj_name2id(m, mjOBJ_BODY, "b");
  BatchPtr batch(MakeBatch(m, 2, 1, 0));
  mjBatch* b = batch.b;

  int size = 0, elemsize = 0;
  // not expanded, and still not after the query
  EXPECT_THAT(mjb_expanded(b, "eq_data", &size, &elemsize), IsNull());
  EXPECT_EQ(size, mjNEQDATA * m->neq);
  EXPECT_EQ(elemsize, sizeof(mjtNum));
  EXPECT_THAT(mjb_expanded(b, "eq_data", nullptr, nullptr), IsNull());
  EXPECT_THAT(mjb_expanded(b, "mesh_vert", nullptr, nullptr), IsNull());
  EXPECT_THAT(mjb_expanded(b, "nope", nullptr, nullptr), IsNull());

  auto* pos = static_cast<mjtNum*>(mjb_expand(b, "body_pos", nullptr, nullptr));
  EXPECT_EQ(mjb_expanded(b, "body_pos", nullptr, nullptr), pos);
  pos[1 * 3 * m->nbody + 3 * body_b] = 0.7;

  // expands eq_data: its body2 anchor moved
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  const auto* eq_data =
      static_cast<const mjtNum*>(mjb_expanded(b, "eq_data", nullptr, nullptr));
  ASSERT_THAT(eq_data, NotNull());
  const mjtNum* data1 = eq_data + 1 * mjNEQDATA * m->neq;
  mjtNum anchor2 = data1[3];
  EXPECT_NE(anchor2, m->eq_data[3]);

  // still derived: moving body b again moves the anchor at the next set_const
  pos[1 * 3 * m->nbody + 3 * body_b] = 0.9;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  EXPECT_EQ(mjb_expanded(b, "eq_data", nullptr, nullptr), eq_data);
  EXPECT_NE(data1[3], anchor2);
}

// every mjtState component is kept in the rows, never copied out as an output;
// a component added to mjtState must be added here and to InState in batch.cc
TEST_F(BatchTest, StateComponentsAreNotOutputs) {
  const char* fields[] = {"time",      "qpos",         "qvel",
                          "act",       "history",      "qacc_warmstart",
                          "ctrl",      "qfrc_applied", "xfrc_applied",
                          "eq_active", "mocap_pos",    "mocap_quat",
                          "userdata",  "plugin_state", "plastic"};
  ASSERT_EQ(sizeof(fields) / sizeof(fields[0]), mjNSTATE);
  ASSERT_EQ(mjSTATE_INTEGRATION, (1 << mjNSTATE) - 1);
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), 2, 1, 0));
  for (int c = 1; c < mjNSTATE; ++c) {  // time is a scalar, not an array field
    EXPECT_THAT(mjb_output(batch.b, fields[c], nullptr, nullptr), IsNull())
        << fields[c];
  }
  EXPECT_THAT(mjb_output(batch.b, "xpos", nullptr, nullptr), NotNull());
}

// the scalars of mjModel that mj_setConst changes are the ones the batch keeps
// per simulation (Scalars in batch.cc); a new one must be added there
TEST_F(BatchTest, SetConstChangesNoOtherScalar) {
  static constexpr char xml[] = R"(
  <mujoco>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .5" gravcomp="1">
        <freejoint/>
        <geom size=".1"/>
      </body>
      <body name="b" pos=".5 0 .5">
        <freejoint/>
        <geom type="box" size=".1 .1 .1"/>
      </body>
    </worldbody>
    <actuator>
      <adhesion body="b" ctrlrange="0 1" gain="10"/>
    </actuator>
  </mujoco>
  )";
  for (const char* text : {kXml, xml}) {
    MjModelPtr model = LoadModelFromString(text);
    ASSERT_THAT(model, NotNull());
    mjModel* m = mj_copyModel(nullptr, model.get());
    for (int i = 1; i < m->nbody; ++i) {
      m->body_mass[i] *= 2;
      m->body_gravcomp[i] = 1;
    }
    std::vector<mjModel> snapshot(2);
    std::memcpy(&snapshot[0], m, sizeof(mjModel));
    mjData* d = mj_makeData(m);
    mj_setConst(m, d);
    std::memcpy(&snapshot[1], m, sizeof(mjModel));
    EXPECT_NE(snapshot[0].ngravcomp,
              snapshot[1].ngravcomp);  // the edits reach a scalar
    for (mjModel& s : snapshot) {
#define X(type, name, nr, nc) s.name = nullptr;
      MJMODEL_POINTERS
#undef X
#define X(name) s.name = {};
      MJSETCONST_SCALARS
#undef X
    }
    EXPECT_EQ(std::memcmp(&snapshot[0], &snapshot[1], sizeof(mjModel)), 0)
        << "mj_setConst changed an mjModel scalar the batch does not keep";
    mj_deleteData(d);
    mj_deleteModel(m);
  }
}

// contact-rich lockstep: free bodies on a plane and on each other, with
// elliptic cones and torsional friction, against one mjData per simulation
TEST_F(BatchTest, ContactLockstep) {
  MjModelPtr model = LoadModelFromString(kContactXml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  int qvel = Offset(m, mjSTATE_QVEL);
  for (int persistent : {0, 1}) {
    for (int nthread : {1, 3}) {
      BatchPtr batch(MakeBatch(m, kNsim, nthread, persistent));
      mjBatch* b = batch.b;
      std::vector<MjDataPtr> ref;
      for (int i = 0; i < kNsim; ++i) {
        ref.push_back(MakeData(model));
        for (int k = 0; k < m->nv; ++k) {
          mjtNum v = 0.5 * std::sin(i + 0.7 * k);
          batch.Row(i)[qvel + k] = v;
          ref[i]->qvel[k] = v;
        }
      }
      int maxcon = 0;
      for (int call = 0; call < 40; ++call) {
        ASSERT_EQ(mjb_step(b, nullptr, 0, 5, mjSTAGE_NONE), 0)
            << mjb_error(b, 0);
        for (int i = 0; i < kNsim; ++i) {
          for (int k = 0; k < 5; ++k) mj_step(m, ref[i].get());
          maxcon = std::max(maxcon, ref[i]->ncon);
          ASSERT_EQ(
              std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(m, ref[i].get()))
              << "persistent " << persistent << " nthread " << nthread
              << " call " << call << " sim " << i;
        }
      }
      EXPECT_GE(maxcon, 6)
          << "too few contacts: the test does not exercise them";
    }
  }
}

// a mix of every call, with writes to controls, state and model fields between
// calls, on a persistent batch that continues steps from its previous call's
// position and velocity stages, and on a shared batch that cannot: rows and
// outputs agree bitwise
void ContinuationMatchesShared(const char* xml) {
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  BatchPtr persistent(MakeBatch(m, kNsim, 2, 1));
  BatchPtr shared(MakeBatch(m, kNsim, 3, 0));
  int nstate = mjb_nstate(persistent.b);
  int qpos = Offset(m, mjSTATE_QPOS);
  int ctrl = Offset(m, mjSTATE_CTRL);
  std::vector<mjtNum*> mass, xpos, sensordata;
  for (mjBatch* b : {persistent.b, shared.b}) {
    mass.push_back(
        static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr)));
    xpos.push_back(
        static_cast<mjtNum*>(mjb_output(b, "xpos", nullptr, nullptr)));
    sensordata.push_back(
        static_cast<mjtNum*>(mjb_output(b, "sensordata", nullptr, nullptr)));
  }
  auto noop = +[](const mjModel* m, mjData* d, int sim, void* arg) {};
  int subset[] = {1, 2, 6};
  int reset[] = {0, 5};
  int zero[] = {0, 0};  // keyframe 0
  // whether a simulation's acceleration-stage outputs were computed by a call:
  // forward and reset below mjSTAGE_ACC leave them to whatever the mjData held
  std::vector<bool> acc(kNsim, true);

  for (int round = 0; round < 60; ++round) {
    std::string op;
    for (int b = 0; b < 2; ++b) {
      BatchPtr& batch = b ? shared : persistent;
      for (int i = 0; i < kNsim; ++i) {
        for (int k = 0; k < m->nu; ++k) {
          batch.Row(i)[ctrl + k] = std::sin(0.3 * round + i + k);
        }
      }
      if (round % 7 ==
          3) {  // a state write, of a coordinate the forces depend on
        for (int i : {1, 4}) batch.Row(i)[qpos + 1] += 0.01;
      }
      if (round % 9 == 5) {  // a model write, without set_const
        mass[b][2 * m->nbody + 1] *= 1.1;
      }

      int n = 0;
      if (round % 11 == 7 && m->nkey) {
        op = "reset";
        n = mjb_reset(batch.b, reset, 2, zero,
                      round % 22 == 7 ? mjSTAGE_ACC : mjSTAGE_NONE);
        for (int i : reset) acc[i] = round % 22 == 7;
      } else if (round % 19 == 14 && m->nkey) {
        op = "reset vel";
        n = mjb_reset(batch.b, reset, 2, zero, mjSTAGE_VEL);
        for (int i : reset) acc[i] = false;
      } else if (round % 13 == 9) {
        op = "set_const";
        n = mjb_setConst(batch.b, nullptr, 0);
      } else if (round % 10 == 4) {
        op = "apply";
        n = mjb_apply(batch.b, nullptr, 0, noop, nullptr, 0);
      } else if (round % 12 == 6) {
        op = "forward";
        n = mjb_forward(batch.b, nullptr, 0,
                        round % 24 == 6 ? mjSTAGE_ACC : mjSTAGE_VEL);
        acc.assign(kNsim, round % 24 == 6);
      } else if (round % 8 == 2) {
        op = "step none";
        n = mjb_step(batch.b, nullptr, 0, 2, mjSTAGE_NONE);
        acc.assign(kNsim, true);
      } else if (round % 17 == 11) {
        op = "step acc";
        n = mjb_step(batch.b, nullptr, 0, 2, mjSTAGE_ACC);
        acc.assign(kNsim, true);
      } else if (round % 5 == 1) {
        op = "step vel, subset";
        n = mjb_step(batch.b, subset, 3, 2, mjSTAGE_VEL);
        for (int i : subset) acc[i] = true;
      } else {
        op = "step vel";
        n = mjb_step(batch.b, nullptr, 0, 1 + round % 3, mjSTAGE_VEL);
        acc.assign(kNsim, true);
      }
      ASSERT_EQ(n, 0) << op << ": " << mjb_error(batch.b, 0);
    }

    std::string at = "round " + std::to_string(round) + " (" + op + ")";
    for (int i = 0; i < kNsim; ++i) {
      EXPECT_EQ(std::memcmp(persistent.Row(i), shared.Row(i),
                            sizeof(mjtNum) * nstate),
                0)
          << at << ", state of simulation " << i;
    }
    EXPECT_EQ(
        std::memcmp(xpos[0], xpos[1], sizeof(mjtNum) * kNsim * 3 * m->nbody), 0)
        << at << ", xpos";
    for (int i = 0; i < kNsim; ++i) {
      if (!acc[i]) continue;
      EXPECT_EQ(std::memcmp(sensordata[0] + i * m->nsensordata,
                            sensordata[1] + i * m->nsensordata,
                            sizeof(mjtNum) * m->nsensordata),
                0)
          << at << ", sensordata of simulation " << i;
    }
  }
}

TEST_F(BatchTest, ContinuationMatchesShared) {
  ContinuationMatchesShared(kXml);
}

TEST_F(BatchTest, ContinuationMatchesSharedWithContacts) {
  ContinuationMatchesShared(kContactXml);
}

// position-stage passes of each simulation's mjData, read through mjb_apply
// (which ends any continuation)
std::vector<int> PositionPasses(mjBatch* b) {
  std::vector<int> passes(mjb_nsim(b));
  auto read = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    static_cast<int*>(arg)[sim] = d->timer[mjTIMER_POS_KINEMATICS].number;
  };
  EXPECT_EQ(mjb_apply(b, nullptr, 0, read, passes.data(), 0), 0);
  return passes;
}

std::vector<int> StepCounts(mjBatch* b) {
  std::vector<int> steps(mjb_nsim(b));
  auto read = +[](const mjModel* m, mjData* d, int sim, void* arg) {
    static_cast<int*>(arg)[sim] = d->timer[mjTIMER_STEP].number;
  };
  EXPECT_EQ(mjb_apply(b, nullptr, 0, read, steps.data(), 0), 0);
  return steps;
}

void NoopControl(const mjModel* m, mjData* d) {}

// a persistent batch continues a step from the position and velocity stages its
// previous call computed, unless anything they depend on changed
TEST_F(BatchTest, ContinuationSkipsPositionPass) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, 3, 2, 1));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  std::vector<int> before = PositionPasses(b);
  ASSERT_EQ(mjb_forward(b, nullptr, 0, mjSTAGE_ACC), 0);  // one pass
  for (int call = 0; call < 2; ++call) {  // three each, continuing
    ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_VEL), 0);
  }
  mass[1 * m->nbody + 1] *= 2;                   // simulation 1: a model write
  batch.Row(2)[Offset(m, mjSTATE_QPOS)] += 0.1;  // simulation 2: a state write
  ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_VEL), 0);
  std::vector<int> after = PositionPasses(b);
  EXPECT_EQ(after[0] - before[0], 1 + 3 + 3 + 3);
  EXPECT_EQ(after[1] - before[1], 1 + 3 + 3 + 4);
  EXPECT_EQ(after[2] - before[2], 1 + 3 + 3 + 4);

  // continued steps count once
  for (int steps : StepCounts(b)) EXPECT_EQ(steps, 3 + 3 + 3);

  // after mjb_apply, and with a control callback installed, every call starts
  // afresh
  ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_VEL),
            0);  // four: apply ended it
  {
    ScopedSet control(mjcb_control, NoopControl);
    ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_VEL), 0);  // four
  }
  std::vector<int> last = PositionPasses(b);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(last[i] - after[i], 4 + 4) << "simulation " << i;

  // stage NONE leaves nothing to continue from
  ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_NONE), 0);
  ASSERT_EQ(mjb_step(b, nullptr, 0, 3, mjSTAGE_VEL), 0);
  std::vector<int> none = PositionPasses(b);
  for (int i = 0; i < 3; ++i)
    EXPECT_EQ(none[i] - last[i], 3 + 4) << "simulation " << i;
}

// a persistent batch continues its next step from the stages of an
// acceleration-stage call, so that call checks the state as mj_step would
TEST_F(BatchTest, ForwardChecksStateBeforeContinuing) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  mock_warning_handler.ExpectWarnings("QPOS");
  std::vector<std::vector<mjtNum>> rows;
  for (int persistent : {0, 1}) {
    BatchPtr batch(MakeBatch(m, 1, 1, persistent));
    batch.Row(0)[Offset(m, mjSTATE_QPOS)] =
        std::numeric_limits<mjtNum>::quiet_NaN();
    ASSERT_EQ(mjb_forward(batch.b, nullptr, 0, mjSTAGE_ACC), 0);
    ASSERT_EQ(mjb_step(batch.b, nullptr, 0, 1, mjSTAGE_VEL), 0);
    rows.emplace_back(batch.Row(0), batch.Row(0) + mjb_nstate(batch.b));
    EXPECT_EQ(mjb_warning(batch.b)[mjWARN_BADQPOS].number, 1);
  }
  for (mjtNum x : rows[1]) EXPECT_TRUE(std::isfinite(x));
  EXPECT_EQ(rows[1], rows[0]);
}

// RK4 cannot continue: mj_step2 integrates with Euler
TEST_F(BatchTest, ContinuationNotWithRungeKutta) {
  std::string xml(kXml);
  xml.replace(xml.find("<option timestep=\"0.002\"/>"), 26,
              "<option timestep=\"0.002\" integrator=\"RK4\"/>");
  MjModelPtr model = LoadModelFromString(xml.c_str());
  ASSERT_THAT(model, NotNull());
  BatchPtr batch(MakeBatch(model.get(), 2, 1, 1));
  std::vector<int> before = PositionPasses(batch.b);
  ASSERT_EQ(mjb_forward(batch.b, nullptr, 0, mjSTAGE_ACC), 0);
  ASSERT_EQ(mjb_step(batch.b, nullptr, 0, 3, mjSTAGE_VEL), 0);
  std::vector<int> after = PositionPasses(batch.b);
  EXPECT_EQ(after[0] - before[0],
            1 + 3 * 4 + 1);  // RK4 evaluates four stages per step
  ContinuationMatchesShared(xml.c_str());
}

// stage sets what is computed at the new state: VEL gives positions of the new
// state, NONE those of the state before the last substep
TEST_F(BatchTest, StepStage) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  for (int persistent : {0, 1}) {
    BatchPtr batch(MakeBatch(m, kNsim, 2, persistent));
    mjBatch* b = batch.b;
    auto* xpos = static_cast<mjtNum*>(mjb_output(b, "xpos", nullptr, nullptr));
    int qvel = Offset(m, mjSTATE_QVEL);
    for (int i = 0; i < kNsim; ++i) batch.Row(i)[qvel] = 1;
    MjDataPtr d = MakeData(model);
    for (int stage : {mjSTAGE_VEL, mjSTAGE_NONE}) {
      std::vector<mjtNum> before(mjb_state(b),
                                 mjb_state(b) + kNsim * mjb_nstate(b));
      ASSERT_EQ(mjb_step(b, nullptr, 0, 2, stage), 0);
      for (int i = 0; i < kNsim; ++i) {
        mj_setState(m, d.get(), batch.Row(i), mjSTATE_INTEGRATION);
        mj_kinematics(m, d.get());
        bool fresh = !std::memcmp(xpos + i * 3 * m->nbody, d->xpos,
                                  sizeof(mjtNum) * 3 * m->nbody);
        EXPECT_EQ(fresh, stage == mjSTAGE_VEL)
            << "persistent " << persistent << " sim " << i;
      }
    }
    EXPECT_THAT(
        MjuErrorMessageFrom(mjb_step)(b, nullptr, 0, 1, mjSTAGE_ACC + 1),
        HasSubstr("stage"));
  }
}

// reset at a lower stage, followed by randomization and forward, as in
// reinforcement learning, gives what a reset at mjSTAGE_ACC does, with one
// forward pass fewer
TEST_F(BatchTest, ResetStage) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  int qpos = Offset(m, mjSTATE_QPOS);
  for (int persistent : {0, 1}) {
    BatchPtr full(MakeBatch(m, kNsim, 2, persistent));
    BatchPtr none(MakeBatch(m, kNsim, 2, persistent));
    std::vector<mjtNum*> xpos;
    for (BatchPtr* batch : {&full, &none}) {
      xpos.push_back(
          static_cast<mjtNum*>(mjb_output(batch->b, "xpos", nullptr, nullptr)));
      ASSERT_EQ(mjb_step(batch->b, nullptr, 0, 3, mjSTAGE_NONE), 0);
    }
    std::vector<int> keys(kNsim, 0);
    ASSERT_EQ(mjb_reset(full.b, nullptr, 0, keys.data(), mjSTAGE_ACC), 0);
    ASSERT_EQ(mjb_reset(none.b, nullptr, 0, keys.data(), mjSTAGE_NONE), 0);
    for (BatchPtr* batch : {&full, &none}) {
      for (int i = 0; i < kNsim; ++i) batch->Row(i)[qpos + 1] = 0.1 * i;
      ASSERT_EQ(mjb_forward(batch->b, nullptr, 0, mjSTAGE_ACC), 0);
    }
    std::vector<int> passes_full = PositionPasses(full.b);
    std::vector<int> passes_none = PositionPasses(none.b);
    EXPECT_EQ(std::memcmp(full.Row(0), none.Row(0),
                          sizeof(mjtNum) * kNsim * mjb_nstate(full.b)),
              0)
        << "persistent " << persistent;
    EXPECT_EQ(
        std::memcmp(xpos[0], xpos[1], sizeof(mjtNum) * kNsim * 3 * m->nbody), 0)
        << "persistent " << persistent;
    if (persistent) {                // mj_resetData clears the counts
      EXPECT_EQ(passes_full[0], 2);  // reset and forward
      EXPECT_EQ(passes_none[0], 1);  // forward only
    }
  }
}

// a field registered in a persistent batch starts with each simulation's
// current values; in a shared batch it is zero until the next call
TEST_F(BatchTest, OutputStartsFromPersistentData) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  MjDataPtr d = MakeData(model);
  for (int persistent : {0, 1}) {
    BatchPtr batch(MakeBatch(m, kNsim, 2, persistent));
    mjBatch* b = batch.b;
    int qpos = Offset(m, mjSTATE_QPOS);
    for (int i = 0; i < kNsim; ++i) batch.Row(i)[qpos + 1] = 0.1 * i;
    ASSERT_EQ(mjb_forward(b, nullptr, 0, mjSTAGE_ACC), 0);
    auto* xquat =
        static_cast<mjtNum*>(mjb_output(b, "xquat", nullptr, nullptr));
    for (int i = 0; i < kNsim; ++i) {
      mj_setState(m, d.get(), batch.Row(i), mjSTATE_INTEGRATION);
      mj_forward(m, d.get());
      std::vector<mjtNum> got(xquat + 4 * m->nbody * i,
                              xquat + 4 * m->nbody * (i + 1));
      if (persistent) {
        EXPECT_EQ(got, std::vector<mjtNum>(d->xquat, d->xquat + 4 * m->nbody))
            << "sim " << i;
      } else {
        EXPECT_EQ(got, std::vector<mjtNum>(4 * m->nbody, 0)) << "sim " << i;
      }
    }
  }
}

// calls on fewer simulations than threads use only as many threads, and the
// others skip them; interleaved with full calls, results match one mjData per
// simulation
TEST_F(BatchTest, SmallCallsOnManyThreads) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  constexpr int nsim = 16;
  BatchPtr batch(MakeBatch(m, nsim, 8, 0));
  ASSERT_EQ(mjb_nthread(batch.b), 8);
  std::vector<MjDataPtr> ref;
  for (int i = 0; i < nsim; ++i) ref.push_back(MakeData(model));
  int ctrl = Offset(m, mjSTATE_CTRL);
  const int sizes[] = {1, 2, 3, 5, 7, 16};
  for (int call = 0; call < 200; ++call) {
    int n = sizes[call % 6];
    std::vector<int> ids;
    for (int j = 0; j < n; ++j) ids.push_back((call + j * (nsim / n)) % nsim);
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    for (int i : ids) {
      mjtNum u = std::sin(0.1 * call + i);
      batch.Row(i)[ctrl] = u;
      ref[i]->ctrl[0] = u;
    }
    ASSERT_EQ(mjb_step(batch.b, ids.data(), ids.size(), 1, mjSTAGE_NONE), 0);
    for (int i : ids) mj_step(m, ref[i].get());
  }
  for (int i = 0; i < nsim; ++i) {
    EXPECT_EQ(
        std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(batch.b)),
        GetState(m, ref[i].get()))
        << "sim " << i;
  }
}

// the address and size of each state component in the rows
TEST_F(BatchTest, StateComponent) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 1, 0));
  mjBatch* b = batch.b;
  for (int k = 0; k < mjNSTATE; ++k) {
    mjtState component = static_cast<mjtState>(1 << k);
    int size = -1;
    mjtNum* adr = mjb_stateComponent(b, component, &size);
    EXPECT_EQ(adr - mjb_state(b), Offset(m, component)) << "component " << k;
    EXPECT_EQ(size, mj_stateSize(m, component)) << "component " << k;
  }
  EXPECT_THAT(MjuErrorMessageFrom(mjb_stateComponent)(
                  b, mjSTATE_QPOS | mjSTATE_QVEL, nullptr),
              HasSubstr("one"));
  EXPECT_THAT(MjuErrorMessageFrom(mjb_stateComponent)(b, 0, nullptr),
              HasSubstr("one"));
}

// a force applied to a sleeping tree between calls wakes it: a step that would
// continue from the last call's position stage, which ran before the force,
// starts afresh
TEST_F(BatchTest, ContinuationSeesForcesOnSleepingTrees) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="implicitfast" viscosity="10" sleep_tolerance="0.01">
      <flag sleep="enable" gravity="disable" constraint="disable"
        contact="disable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom type="box" size=".1 .2 .3" mass="1"/>
      </body>
      <body pos="1 0 0">
        <freejoint/>
        <geom type="sphere" size=".1" mass="1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  constexpr int nsim = 2;
  BatchPtr batch(MakeBatch(m, nsim, 1, 1));
  mjBatch* b = batch.b;
  int qvel = Offset(m, mjSTATE_QVEL);
  int qfrc = Offset(m, mjSTATE_QFRC_APPLIED);
  int xfrc = Offset(m, mjSTATE_XFRC_APPLIED);
  std::vector<MjDataPtr> ref;
  for (int i = 0; i < nsim; ++i) {
    ref.push_back(MakeData(model));
    for (int k = 0; k < m->nv; ++k) {
      batch.Row(i)[qvel + k] = ref[i]->qvel[k] = 0.1 * (k % 3 + 1);
    }
  }
  bool pushed[nsim] = {false, false};
  for (int call = 0; call < 100; ++call) {
    for (int i = 0; i < nsim; ++i) {
      for (int tree = 0; tree < m->ntree && !pushed[i]; ++tree) {
        if (ref[i]->tree_awake[tree]) continue;
        int body = m->tree_bodyadr[tree];
        if (i ==
            0) {  // simulation 0: a body force; simulation 1: a joint force
          batch.Row(i)[xfrc + 6 * body] = ref[i]->xfrc_applied[6 * body] = 1;
        } else {
          int dof = m->body_dofadr[body];
          batch.Row(i)[qfrc + dof] = ref[i]->qfrc_applied[dof] = 1;
        }
        pushed[i] = true;
      }
    }
    ASSERT_EQ(mjb_step(b, nullptr, 0, 10, mjSTAGE_VEL), 0) << mjb_error(b, 0);
    for (int i = 0; i < nsim; ++i) {
      for (int k = 0; k < 10; ++k) mj_step(m, ref[i].get());
      ASSERT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
                GetState(m, ref[i].get()))
          << "call " << call << " sim " << i;
    }
  }
  EXPECT_TRUE(pushed[0] && pushed[1])
      << "no tree fell asleep: the test does not exercise sleep";
}

void FailingControl(const mjModel* m, mjData* d) {
  mju_error("control failed");
}

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

// trees that start asleep make mj_makeData run mj_forward and put them to
// sleep; an error raised by a callback there fails mjb_makeBatch with its
// message, and frees everything it allocated
TEST_F(BatchTest, MakeDataErrorFailsMakeBatch) {
  static constexpr char xml[] = R"(
  <mujoco>
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
  char error[300];
  mjBatch* b;
  {
    ScopedSet control(mjcb_control, FailingControl);
    ScopedSet malloc_hook(mju_user_malloc, CountingMalloc);
    ScopedSet free_hook(mju_user_free, CountingFree);
    b = MakeBatch(model.get(), 4, 2, 1, error, sizeof(error));
  }
  EXPECT_THAT(b, IsNull());
  EXPECT_THAT(error, HasSubstr("control failed"));
  EXPECT_EQ(live_allocations, 0);
  BatchPtr batch(MakeBatch(model.get(), 4, 2, 1));
  EXPECT_THAT(batch.b, NotNull());
}

// passive plugin with a step counter in its instance data, outside
// plugin_state, that pushes the first dof harder after each step
int RegisterCounterPlugin() {
  mjpPlugin plugin;
  mjp_defaultPlugin(&plugin);
  plugin.name = "mujoco.test.batchcounter";
  plugin.capabilityflags |= mjPLUGIN_PASSIVE;
  plugin.nstate = +[](const mjModel* m, int instance) { return 0; };
  plugin.init = +[](const mjModel* m, mjData* d, int instance) {
    d->plugin_data[instance] = reinterpret_cast<uintptr_t>(new int(0));
    return 0;
  };
  plugin.destroy = +[](mjData* d, int instance) {
    delete reinterpret_cast<int*>(d->plugin_data[instance]);
    d->plugin_data[instance] = 0;
  };
  plugin.reset = +[](const mjModel* m, mjtNum* plugin_state, void* plugin_data,
                     int instance) { *static_cast<int*>(plugin_data) = 0; };
  plugin.compute = +[](const mjModel* m, mjData* d, int instance, int type) {
    int count = *reinterpret_cast<int*>(d->plugin_data[instance]);
    d->qfrc_passive[0] += 0.01 * count;
  };
  plugin.advance = +[](const mjModel* m, mjData* d, int instance) {
    ++*reinterpret_cast<int*>(d->plugin_data[instance]);
  };
  return mjp_registerPlugin(&plugin);
}

// a persistent batch gives each simulation its own plugin instances, so even
// plugins that keep state outside plugin_state match a loop, with continuation
TEST_F(BatchTest, PersistentBatchHasPluginInstancesPerSimulation) {
  static const int slot = RegisterCounterPlugin();
  ASSERT_GE(slot, 0);
  std::string xml(kXml);
  xml.replace(xml.find("<worldbody>"), 11,
              "<extension><plugin plugin=\"mujoco.test.batchcounter\">"
              "<instance name=\"counter\"/></plugin></extension><worldbody>");
  MjModelPtr model = LoadModelFromString(xml.c_str());
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  constexpr int nsim = 3, ncall = 4, nstep = 2;
  BatchPtr batch(MakeBatch(m, nsim, 2, 1));
  mjBatch* b = batch.b;
  ASSERT_EQ(mjb_reset(b, nullptr, 0, nullptr, mjSTAGE_ACC), 0);
  for (int call = 0; call < ncall; ++call) {
    ASSERT_EQ(mjb_step(b, nullptr, 0, nstep, mjSTAGE_VEL), 0);
  }
  for (int i = 0; i < nsim; ++i) {
    MjDataPtr d = MakeData(model);
    mj_forward(m, d.get());
    for (int k = 0; k < ncall * nstep; ++k) mj_step(m, d.get());
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(m, d.get()))
        << "simulation " << i;
  }
}

// copies of model m with the pole's mass of simulation i, constants included
std::vector<MjModelPtr> MassVariants(const mjModel* m, int nsim) {
  int pole = mj_name2id(m, mjOBJ_BODY, "pole");
  std::vector<MjModelPtr> variants;
  for (int i = 0; i < nsim; ++i) {
    variants.emplace_back(mj_copyModel(nullptr, m));
    variants.back()->body_mass[pole] = 0.1 * (i + 1);
    mjData* d = mj_makeData(variants.back().get());
    mj_setConst(variants.back().get(), d);
    mj_deleteData(d);
  }
  return variants;
}

// each simulation runs on its caller model as in a loop, from a step after a
// call on the batch's own model, which a persistent batch must not continue
void CallerModels(int persistent) {
  MjModelPtr model = LoadModelFromString(kXml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  std::vector<MjModelPtr> variants = MassVariants(m, kNsim);
  std::vector<const mjModel*> models;
  for (auto& v : variants) models.push_back(v.get());
  BatchPtr batch(MakeBatch(m, kNsim, 3, persistent));
  mjBatch* b = batch.b;
  ASSERT_EQ(mjb_reset(b, nullptr, 0, nullptr, mjSTAGE_VEL), 0);
  mjb_setModels(b, models.data());
  mjtNum* ctrl = mjb_stateComponent(b, mjSTATE_CTRL, nullptr);
  for (int i = 0; i < kNsim; ++i) ctrl[i * mjb_nstate(b)] = 0.5;
  ASSERT_EQ(mjb_step(b, nullptr, 0, 20, mjSTAGE_NONE), 0);
  mjModel* copy = mj_copyModel(nullptr, m);
  for (int i = 0; i < kNsim; ++i) {
    MjDataPtr d = MakeData(model);
    mj_resetData(m, d.get());
    d->ctrl[0] = 0.5;
    for (int k = 0; k < 20; ++k) mj_step(models[i], d.get());
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(m, d.get()))
        << "sim " << i;
    mjb_getModel(b, i, copy);
    int pole = mj_name2id(m, mjOBJ_BODY, "pole");
    EXPECT_EQ(copy->body_mass[pole], models[i]->body_mass[pole]);
  }
  mj_deleteModel(copy);

  // back on the batch's own model
  mjb_setModels(b, nullptr);
  std::vector<mjtNum> row(batch.Row(0), batch.Row(0) + mjb_nstate(b));
  ASSERT_EQ(mjb_step(b, nullptr, 0, 1, mjSTAGE_NONE), 0);
  MjDataPtr d = MakeData(model);
  mj_setState(m, d.get(), row.data(), mjSTATE_INTEGRATION);
  mj_step(m, d.get());
  EXPECT_EQ(std::vector<mjtNum>(batch.Row(0), batch.Row(0) + mjb_nstate(b)),
            GetState(m, d.get()));
}

TEST_F(BatchTest, CallerModels) { CallerModels(0); }

TEST_F(BatchTest, CallerModelsPersistent) { CallerModels(1); }

// caller models may differ in the shape of their meshes: mjData holds only the
// number of their bounding volumes
TEST_F(BatchTest, CallerModelsWithDifferentMeshes) {
  std::string xml = R"(
  <mujoco>
    <asset>
      <mesh name="rock" vertex="VERTICES"/>
    </asset>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .3">
        <freejoint/>
        <geom type="mesh" mesh="rock"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  std::string tet = xml, skewed = xml;
  tet.replace(tet.find("VERTICES"), 8, "0 0 0  .2 0 0  0 .2 0  0 0 .2");
  skewed.replace(skewed.find("VERTICES"), 8,
                 "0 0 0  .3 0 0  0 .1 0  .05 0 .25");
  MjModelPtr m0 = LoadModelFromString(tet.c_str());
  MjModelPtr m1 = LoadModelFromString(skewed.c_str());
  ASSERT_THAT(m0, NotNull());
  ASSERT_THAT(m1, NotNull());
  ASSERT_NE(m0->mesh_vert[3], m1->mesh_vert[3]);
  const mjModel* models[] = {m0.get(), m1.get()};
  BatchPtr batch(MakeBatch(m0.get(), 2, 2, 0));
  mjb_setModels(batch.b, models);
  ASSERT_EQ(mjb_reset(batch.b, nullptr, 0, nullptr, mjSTAGE_NONE), 0);
  ASSERT_EQ(mjb_step(batch.b, nullptr, 0, 300, mjSTAGE_NONE), 0);
  for (int i = 0; i < 2; ++i) {
    mjData* d = mj_makeData(models[i]);
    for (int k = 0; k < 300; ++k) mj_step(models[i], d);
    EXPECT_GT(d->ncon, 0);
    EXPECT_EQ(
        std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(batch.b)),
        GetState(models[i], d))
        << "sim " << i;
    mj_deleteData(d);
  }
}

// caller models must give mjData the batch model's layout, and replace the
// batch's own per-simulation fields rather than combine with them
TEST_F(BatchTest, CallerModelsChecks) {
  MjModelPtr model = LoadModelFromString(kXml);
  MjModelPtr other = LoadModelFromString(kContactXml);
  const mjModel* m = model.get();
  std::vector<MjModelPtr> variants = MassVariants(m, 2);
  const mjModel* good[] = {variants[0].get(), variants[1].get()};
  const mjModel* mismatched[] = {variants[0].get(), other.get()};
  const mjModel* missing[] = {variants[0].get(), nullptr};
  {
    BatchPtr batch(MakeBatch(m, 2, 1, 0));
    EXPECT_THAT(MjuErrorMessageFrom(mjb_setModels)(batch.b, mismatched),
                HasSubstr("layout"));
    EXPECT_THAT(MjuErrorMessageFrom(mjb_setModels)(batch.b, missing),
                HasSubstr("NULL"));
    mjb_setModels(batch.b, good);
    EXPECT_THAT(
        MjuErrorMessageFrom(mjb_expand)(batch.b, "body_mass", nullptr, nullptr),
        HasSubstr("caller"));
    EXPECT_THAT(MjuErrorMessageFrom(mjb_setConst)(batch.b, nullptr, 0),
                HasSubstr("caller"));
  }
  {
    BatchPtr batch(MakeBatch(m, 2, 1, 0));
    ASSERT_THAT(mjb_expand(batch.b, "body_mass", nullptr, nullptr), NotNull());
    EXPECT_THAT(MjuErrorMessageFrom(mjb_setModels)(batch.b, good),
                HasSubstr("per-simulation"));
  }
  {
    // sleep, enabled by a caller model, needs a persistent batch
    MjModelPtr sleepy(mj_copyModel(nullptr, m));
    sleepy->opt.enableflags |= mjENBL_SLEEP;
    const mjModel* models[] = {variants[0].get(), sleepy.get()};
    BatchPtr batch(MakeBatch(m, 2, 1, 0));
    EXPECT_THAT(MjuErrorMessageFrom(mjb_setModels)(batch.b, models),
                HasSubstr("persistent"));
  }
  {
    // keyframes are each model's
    std::string xml(kXml);
    xml.replace(xml.find("</keyframe>"), 11,
                "<key qpos=\"-0.3 -0.5 0 1 .5 1 0 0 0\"/></keyframe>");
    MjModelPtr twokeys = LoadModelFromString(xml.c_str());
    ASSERT_EQ(twokeys->nkey, 2);
    const mjModel* models[] = {twokeys.get(), variants[1].get()};
    BatchPtr batch(MakeBatch(m, 2, 1, 0));
    mjb_setModels(batch.b, models);
    int key0[] = {1, 0};
    EXPECT_EQ(mjb_reset(batch.b, nullptr, 0, key0, mjSTAGE_NONE), 0);
    EXPECT_EQ(batch.Row(0)[Offset(m, mjSTATE_QPOS)], twokeys->key_qpos[m->nq]);
    int key1[] = {0, 1};
    EXPECT_THAT(
        MjuErrorMessageFrom(mjb_reset)(batch.b, nullptr, 0, key1, mjSTAGE_NONE),
        HasSubstr("keyframe"));
  }
}

// IPC, like sleep, keeps state in mjData from step to step: it needs a
// persistent batch
TEST_F(BatchTest, IpcNeedsPersistent) {
  static constexpr char xml[] = R"(
  <mujoco>
    <option integrator="discrete">
      <flag ipc="enable"/>
    </option>
    <worldbody>
      <body>
        <freejoint/>
        <geom size=".1"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  char error[256];
  EXPECT_THAT(MakeBatch(model.get(), 2, 1, 0, error, sizeof(error)), IsNull());
  EXPECT_THAT(std::string(error), HasSubstr("persistent"));
  BatchPtr persistent(MakeBatch(model.get(), 2, 1, 1, error, sizeof(error)));
  ASSERT_THAT(persistent.b, NotNull()) << error;
  EXPECT_EQ(mjb_step(persistent.b, nullptr, 0, 3, mjSTAGE_NONE), 0);

  // enabled for one simulation through an expanded flag, it is rejected there
  std::string off(xml);
  off.replace(off.find("<flag ipc=\"enable\"/>"), 20, "");
  MjModelPtr model_off = LoadModelFromString(off.c_str());
  ASSERT_THAT(model_off, NotNull());
  BatchPtr shared(MakeBatch(model_off.get(), 2, 1, 0));
  auto* enableflags = static_cast<int*>(
      mjb_expand(shared.b, "opt.enableflags", nullptr, nullptr));
  enableflags[1] |= mjENBL_IPC;
  EXPECT_EQ(mjb_step(shared.b, nullptr, 0, 1, mjSTAGE_NONE), 1);
  EXPECT_THAT(std::string(mjb_error(shared.b, 1)), HasSubstr("persistent"));
}

// set_const recomputes only the simulations of the call; the fields it expands
// start from the model's values, which are the others' as long as their own
// set_const left them
TEST_F(BatchTest, SetConstRecomputesOnlyItsSimulations) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  int pole = mj_name2id(m, mjOBJ_BODY, "pole");
  BatchPtr batch(MakeBatch(m, kNsim, 2, 0));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  for (int i = 0; i < kNsim; ++i) mass[i * m->nbody + pole] = 0.1 + 0.2 * i;
  int ids[] = {0, 1};
  ASSERT_EQ(mjb_setConst(b, ids, 2), 0);
  auto* subtree =
      static_cast<mjtNum*>(mjb_expand(b, "body_subtreemass", nullptr, nullptr));
  for (int i = 0; i < kNsim; ++i) {
    EXPECT_EQ(subtree[i * m->nbody + pole],
              i < 2 ? mass[i * m->nbody + pole] : m->body_subtreemass[pole])
        << "sim " << i;
  }
}

// mj_setConst refits each simulation's body bounding volume hierarchies; the
// meshes' hierarchies, after the bodies' in bvh_aabb, stay shared
TEST_F(BatchTest, SetConstRefitsBodyHierarchies) {
  static constexpr char xml[] = R"(
  <mujoco>
    <asset>
      <mesh name="tet" vertex="0 0 0  .2 0 0  0 .2 0  0 0 .2"/>
    </asset>
    <worldbody>
      <geom type="plane" size="1 1 .1"/>
      <body pos="0 0 .5">
        <freejoint/>
        <geom name="ball" size=".05" pos=".1 0 0"/>
        <geom size=".05" pos="-.1 0 0"/>
        <geom type="mesh" mesh="tet"/>
      </body>
    </worldbody>
  </mujoco>
  )";
  MjModelPtr model = LoadModelFromString(xml);
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  int ball = mj_name2id(m, mjOBJ_GEOM, "ball");
  int nbody_bvh = m->body_bvhadr[1] + m->body_bvhnum[1];
  ASSERT_EQ(m->mesh_bvhadr[0], nbody_bvh);
  ASSERT_GT(m->nbvhstatic, nbody_bvh);
  std::vector<mjtNum> aabb(m->bvh_aabb, m->bvh_aabb + 6 * m->nbvhstatic);
  constexpr int nsim = 4;
  BatchPtr batch(MakeBatch(m, nsim, 2, 0));
  mjBatch* b = batch.b;
  auto* size =
      static_cast<mjtNum*>(mjb_expand(b, "geom_size", nullptr, nullptr));
  for (int i = 0; i < nsim; ++i) size[(i * m->ngeom + ball) * 3] *= i + 1;
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  ASSERT_EQ(mjb_step(b, nullptr, 0, 1000, mjSTAGE_NONE), 0);
  EXPECT_EQ(std::vector<mjtNum>(mjb_model(b)->bvh_aabb,
                                mjb_model(b)->bvh_aabb + 6 * m->nbvhstatic),
            aabb);
  mjModel* mb = mj_copyModel(nullptr, m);
  for (int i = 0; i < nsim; ++i) {
    mjModel* mi = mj_copyModel(nullptr, m);
    mi->geom_size[3 * ball] = size[(i * m->ngeom + ball) * 3];
    mjData* d = mj_makeData(mi);
    mj_setConst(mi, d);
    mjb_getModel(b, i, mb);
    EXPECT_EQ(mb->geom_rbound[ball], mi->geom_rbound[ball]) << "sim " << i;
    EXPECT_EQ(
        std::vector<mjtNum>(mb->bvh_aabb, mb->bvh_aabb + 6 * m->nbvhstatic),
        std::vector<mjtNum>(mi->bvh_aabb, mi->bvh_aabb + 6 * m->nbvhstatic))
        << "sim " << i;
    mj_resetData(mi, d);
    for (int k = 0; k < 1000; ++k) mj_step(mi, d);
    EXPECT_GT(d->ncon, 0) << "sim " << i;
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(i), batch.Row(i) + mjb_nstate(b)),
              GetState(mi, d))
        << "sim " << i;
    mj_deleteData(d);
    mj_deleteModel(mi);
  }
  mj_deleteModel(mb);
  int n = 0;
  ASSERT_THAT(mjb_expand(b, "bvh_aabb", &n, nullptr), NotNull());
  EXPECT_EQ(n, 6 * nbody_bvh);
}

// fixed-size numeric mjData arrays are outputs too
TEST_F(BatchTest, FixedSizeOutputs) {
  MjModelPtr model = LoadModelFromString(kXml);
  BatchPtr batch(MakeBatch(model.get(), kNsim, 2, 0));
  int size = 0, elemsize = 0;
  auto* niter =
      static_cast<int*>(mjb_output(batch.b, "solver_niter", &size, &elemsize));
  ASSERT_THAT(niter, NotNull());
  EXPECT_EQ(size, mjNISLAND);
  EXPECT_EQ(elemsize, sizeof(int));
  EXPECT_THAT(mjb_output(batch.b, "energy", &size, nullptr), NotNull());
  EXPECT_EQ(size, 2);
  EXPECT_THAT(mjb_output(batch.b, "solver", nullptr, nullptr),
              IsNull());  // a struct array
  ASSERT_EQ(mjb_step(batch.b, nullptr, 0, 1, mjSTAGE_NONE), 0);
  for (int i = 0; i < kNsim; ++i)
    EXPECT_GT(niter[i * mjNISLAND], 0) << "sim " << i;
}

// the scalars that count what a call computed are outputs, each simulation's
// own though simulations share an mjData; the other scalars are not
TEST_F(BatchTest, CountOutputs) {
  MjModelPtr model = LoadModelFromString(kContactXml);
  const mjModel* m = model.get();
  BatchPtr batch(MakeBatch(m, kNsim, 2, 0));
  mjBatch* b = batch.b;
  int size = 0, elemsize = 0;
  auto* ncon = static_cast<int*>(mjb_output(b, "ncon", &size, &elemsize));
  ASSERT_THAT(ncon, NotNull());
  EXPECT_EQ(size, 1);
  EXPECT_EQ(elemsize, sizeof(int));
  auto* nefc = static_cast<int*>(mjb_output(b, "nefc", nullptr, nullptr));
  auto* nisland = static_cast<int*>(mjb_output(b, "nisland", nullptr, nullptr));
  ASSERT_THAT(nefc, NotNull());
  ASSERT_THAT(nisland, NotNull());
  EXPECT_THAT(mjb_output(b, "pstack", nullptr, nullptr), IsNull());
  EXPECT_THAT(mjb_output(b, "maxuse_con", nullptr, nullptr), IsNull());

  // lift the bodies of every other simulation apart, off the plane
  int qpos = Offset(m, mjSTATE_QPOS);
  for (int i = 1; i < kNsim; i += 2) {
    for (int j = 0; j < m->njnt; ++j) {
      batch.Row(i)[qpos + m->jnt_qposadr[j] + 2] += 1 + j;
    }
  }
  std::vector<std::vector<mjtNum>> rows;
  for (int i = 0; i < kNsim; ++i) {
    rows.emplace_back(batch.Row(i), batch.Row(i) + mjb_nstate(b));
  }
  // long enough for the resting bodies to land
  const int nstep = 30;
  ASSERT_EQ(mjb_step(b, nullptr, 0, nstep, mjSTAGE_NONE), 0);
  MjDataPtr d = MakeData(model);
  for (int i = 0; i < kNsim; ++i) {
    mj_setState(m, d.get(), rows[i].data(), mjSTATE_INTEGRATION);
    for (int k = 0; k < nstep; ++k) mj_step(m, d.get());
    EXPECT_EQ(ncon[i], d->ncon) << "sim " << i;
    EXPECT_EQ(nefc[i], d->nefc) << "sim " << i;
    EXPECT_EQ(nisland[i], d->nisland) << "sim " << i;
  }
  EXPECT_GT(ncon[0], 0);
  EXPECT_EQ(ncon[1], 0);
}

// a simulation materialized with mjb_getModel and mjb_getData, including writes
// to its row since its last call, continues exactly as in the batch
void MaterializedContinues(int persistent) {
  MjModelPtr model = LoadModelFromString(kXml);
  const mjModel* m = model.get();
  int pole = mj_name2id(m, mjOBJ_BODY, "pole");
  BatchPtr batch(MakeBatch(m, kNsim, 2, persistent));
  mjBatch* b = batch.b;
  auto* mass =
      static_cast<mjtNum*>(mjb_expand(b, "body_mass", nullptr, nullptr));
  auto* timestep =
      static_cast<mjtNum*>(mjb_expand(b, "opt.timestep", nullptr, nullptr));
  for (int i = 0; i < kNsim; ++i) {
    mass[i * m->nbody + pole] = 0.1 + 0.2 * i;
    timestep[i] = 0.001 * (1 + i % 3);
  }
  ASSERT_EQ(mjb_setConst(b, nullptr, 0), 0);
  ASSERT_EQ(mjb_step(b, nullptr, 0, 5, mjSTAGE_VEL), 0);
  int ctrl = Offset(m, mjSTATE_CTRL);
  for (int i = 0; i < kNsim; ++i)
    batch.Row(i)[ctrl] = 0.5 - 0.1 * i;  // pending writes

  for (int sim : {0, 3, 7}) {
    mjModel* ms = mj_copyModel(nullptr, mjb_model(b));
    mjb_getModel(b, sim, ms);
    EXPECT_EQ(ms->body_mass[pole], mass[sim * m->nbody + pole]);
    EXPECT_EQ(ms->opt.timestep, timestep[sim]);
    mjData* ds = mj_makeData(ms);
    mjb_getData(b, sim, ds);
    EXPECT_EQ(
        GetState(ms, ds),
        std::vector<mjtNum>(batch.Row(sim), batch.Row(sim) + mjb_nstate(b)));
    for (int k = 0; k < 5; ++k) mj_step(ms, ds);
    ASSERT_EQ(mjb_step(b, &sim, 1, 5, mjSTAGE_NONE), 0);
    EXPECT_EQ(
        std::vector<mjtNum>(batch.Row(sim), batch.Row(sim) + mjb_nstate(b)),
        GetState(ms, ds))
        << "persistent " << persistent << " sim " << sim;
    mj_deleteData(ds);
    mj_deleteModel(ms);
  }
}

TEST_F(BatchTest, MaterializedContinues) { MaterializedContinues(0); }

TEST_F(BatchTest, MaterializedContinuesPersistent) { MaterializedContinues(1); }

// the model and data written into must have the batch's sizes
TEST_F(BatchTest, MaterializeChecksSizes) {
  MjModelPtr model = LoadModelFromString(kXml);
  MjModelPtr other = LoadModelFromString(kContactXml);
  BatchPtr batch(MakeBatch(model.get(), 2, 1, 0));
  mjModel* m = mj_copyModel(nullptr, other.get());
  mjData* d = mj_makeData(other.get());
  EXPECT_THAT(MjuErrorMessageFrom(mjb_getModel)(batch.b, 0, m),
              HasSubstr("sizes"));
  EXPECT_THAT(MjuErrorMessageFrom(mjb_getData)(batch.b, 0, d),
              HasSubstr("sizes"));
  EXPECT_THAT(MjuErrorMessageFrom(mjb_getData)(batch.b, 2, d),
              HasSubstr("mjb_getData: sim must be in [0, 2), got 2"));
  mj_deleteData(d);
  mj_deleteModel(m);
}

// each simulation of a reset takes its own keyframe, or the defaults for -1
TEST_F(BatchTest, ResetKeyframePerSimulation) {
  std::string xml(kXml);
  xml.replace(
      xml.find("</keyframe>"), 11,
      "<key qpos=\"-0.3 -0.5 0 1 .5 1 0 0 0\" mpos=\"0 1 .5\"/></keyframe>");
  MjModelPtr model = LoadModelFromString(xml.c_str());
  ASSERT_THAT(model, NotNull());
  const mjModel* m = model.get();
  ASSERT_EQ(m->nkey, 2);
  BatchPtr batch(MakeBatch(m, kNsim, 3, 0));
  mjBatch* b = batch.b;
  ASSERT_EQ(mjb_step(b, nullptr, 0, 5, mjSTAGE_NONE), 0);
  int ids[] = {1, 4, 6};
  int keys[] = {0, -1, 1};
  ASSERT_EQ(mjb_reset(b, ids, 3, keys, mjSTAGE_ACC), 0);
  MjDataPtr d = MakeData(model);
  for (int j = 0; j < 3; ++j) {
    if (keys[j] < 0) {
      mj_resetData(m, d.get());
    } else {
      mj_resetDataKeyframe(m, d.get(), keys[j]);
    }
    mj_forward(m, d.get());
    EXPECT_EQ(std::vector<mjtNum>(batch.Row(ids[j]),
                                  batch.Row(ids[j]) + mjb_nstate(b)),
              GetState(m, d.get()))
        << "sim " << ids[j];
  }
  int bad[] = {0, 2, 0};
  EXPECT_THAT(MjuErrorMessageFrom(mjb_reset)(b, ids, 3, bad, mjSTAGE_ACC),
              HasSubstr("keyframe"));
}

}  // namespace
}  // namespace mujoco
